"""
Gradient descent optimizer for AP position using differentiable ray tracing.

This module implements gradient-based optimization using PyTorch and DrJit
to leverage Sionna's differentiable ray tracing capabilities.
"""

import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import drjit as dr
import matplotlib.pyplot as plt
import sionna.rt
from sionna.rt import RadioMapSolver
import copy

from .base_optimizer import BaseAPOptimizer
from ..metrics import (
    POWER_EPSILON,
    compute_min_rss_metric,
    compute_soft_min_rss_metric,
    normalized_softmin_loss,
    compute_coverage_metric,
    rss_to_dbm,
)


class GradientDescentAPOptimizer(BaseAPOptimizer):
    """
    Gradient descent optimizer for AP position using differentiable ray tracing.

    This class uses PyTorch with DrJit integration (@dr.wrap) to compute gradients
    of the RSS metric with respect to the AP position and optimizes using gradient descent.
    """

    def __init__(
        self,
        scene: sionna.rt.Scene,
        initial_position: Tuple[float, float],
        fixed_z: float = 3.8,
        position_bounds: Dict[str, float] = None,
        initial_direction_xy: Tuple[float, float] = (0.0, 0.0),
        fixed_dir_z: float = -0.5,
        optimize_orientation: bool = True,
    ):
        """
        Initialize gradient descent optimizer.

        Args:
            scene: Sionna Scene object
            initial_position: [x, y] initial AP position (z is fixed)
            fixed_z: Fixed height for AP (z-coordinate)
            position_bounds: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max' for constraints
            initial_direction_xy: Initial horizontal look-at direction ``(dx, dy)``.
                Default ``(0, 0)`` means no horizontal tilt (straight down when
                combined with the fixed ``dir_z``).
            fixed_dir_z: Fixed z-component of the look-at direction vector.
                Kept constant during optimisation for stability.  Default
                ``-0.5`` (downward bias for a ceiling-mounted AP).  Only
                ``dx`` and ``dy`` receive gradients.
            optimize_orientation: Whether to include orientation in the
                optimization.  When ``False`` the direction is kept fixed and
                only the position is optimized (legacy behaviour).
        """
        super().__init__(scene=scene, fixed_z=fixed_z, position_bounds=position_bounds)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimize_orientation = optimize_orientation

        # --- Position normalisation ---
        # Trainable x, y live in [0, 1]; the scene receives denormalised
        # coordinates in [pos_min, pos_max].  This keeps Adam step sizes
        # uniform regardless of the physical coordinate range.
        self.pos_min = position_bounds.get("x_min", 5.0) if position_bounds else 5.0
        self.pos_max = position_bounds.get("x_max", 25.0) if position_bounds else 25.0
        self.pos_range = self.pos_max - self.pos_min  # 20.0 by default

        # Initialize trainable (normalised) position
        self.tx_x = torch.tensor(
            self._normalize(initial_position[0]),
            dtype=torch.float32, requires_grad=True, device=self.device,
        )
        self.tx_y = torch.tensor(
            self._normalize(initial_position[1]),
            dtype=torch.float32, requires_grad=True, device=self.device,
        )

        # --- Orientation (unit-vector look-at) ---
        # Only the horizontal (xy) components are trainable.  The z-component
        # is fixed at ``fixed_dir_z`` to keep the AP tilted downwards and to
        # make gradient-based optimisation more numerically stable (avoids
        # the antenna flipping upside-down or degenerating to the horizon).
        #
        # IMPORTANT: when both dx and dy are zero the direction is purely
        # vertical.  Sionna's look_at() converts to spherical coords via
        # atan2(dy, dx) and acos(dz).  At (0, 0) atan2 produces NaN
        # gradients in DrJit, killing all orientation learning.  We add a
        # small perturbation to guarantee a well-defined azimuth.
        dx0, dy0 = initial_direction_xy
        if abs(dx0) < 1e-6 and abs(dy0) < 1e-6:
            dx0, dy0 = 1e-2, 1e-2
        self.tx_dir_xy = torch.tensor(
            [dx0, dy0], dtype=torch.float32,
            requires_grad=optimize_orientation, device=self.device,
        )
        self.fixed_dir_z = fixed_dir_z

        self.radio_solver = RadioMapSolver()
        self.radio_solver.loop_mode = "evaluated"  # Needed for gradient computation

        # History tracking
        self.history = {
            "positions": [],
            "directions": [],
            "look_at_targets": [],
            "min_rss_values": [],
            "min_rss_dbm_values": [],
            "coverage_values": [],
            "losses": [],
            "gradients": [],
            "direction_gradients": [],
        }

    # --- Normalisation helpers -------------------------------------------
    def _normalize(self, v: float) -> float:
        """Map a physical coordinate to [0, 1]."""
        return (v - self.pos_min) / self.pos_range

    def _denormalize_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Map a normalised [0, 1] tensor back to physical coordinates."""
        return t * self.pos_range + self.pos_min

    def get_full_position(self) -> np.ndarray:
        """Get full 3D position [x, y, z] as numpy array (physical coords)."""
        phys_x = self._denormalize_tensor(self.tx_x).item()
        phys_y = self._denormalize_tensor(self.tx_y).item()
        return np.array([phys_x, phys_y, self.fixed_z])

    def apply_position_constraints(self) -> None:
        """Clamp normalised position to [0, 1] (equivalent to physical bounds)."""
        with torch.no_grad():
            self.tx_x.clamp_(0.0, 1.0)
            self.tx_y.clamp_(0.0, 1.0)

    def get_current_direction(self) -> torch.Tensor:
        """Return the L2-normalised look-at direction (unit vector).

        The full 3-D direction is built from the trainable ``tx_dir_xy``
        and the fixed ``fixed_dir_z``, then normalised to length 1.
        """
        dir_z = torch.tensor(
            [self.fixed_dir_z], dtype=torch.float32, device=self.device,
        )
        full_dir = torch.cat([self.tx_dir_xy, dir_z])
        return F.normalize(full_dir, dim=0)

    def compute_loss(
        self,
        samples_per_tx: int = 1_000_000,
        max_depth: int = 13,
        use_soft_min: bool = True,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute loss for current AP position and orientation with
        differentiable ray tracing.

        This uses @dr.wrap to bridge PyTorch and DrJit for gradient
        computation through the RadioMapSolver.  The transmitter
        orientation is set via ``tx.look_at(target)`` where
        ``target = position + normalised_direction``, so that Dr.Jit
        traces all geometric operations (cross products, rotation
        matrices) and gradients flow back to ``self.tx_direction``.

        When ``use_soft_min=True`` (default), uses ``normalized_softmin_loss``
        which converts Watts → dBm → [0, 1] scores → logsumexp SoftMin.
        This eliminates the manual dBm-scaling heuristic and produces
        well-bounded gradients regardless of absolute power level.

        When ``use_soft_min=False``, falls back to the legacy hard-min loss
        (``-min_rss_dbm / 100``), which can produce sparse gradients.

        Args:
            samples_per_tx: Number of ray tracing samples
            max_depth: Maximum ray tracing depth
            use_soft_min: Use normalized SoftMin loss (recommended) vs hard minimum
            temperature: Temperature for SoftMin (lower = closer to hard min).
                Typical values: 0.05 (sharp), 0.1 (balanced), 0.5 (smooth).

        Returns:
            loss: Scalar tensor to minimise (lower ⇒ higher worst-case RSS)
        """

        # L2-normalise direction to stay on the unit sphere
        direction = self.get_current_direction()

        # Wrap the radio map computation for gradient flow
        @dr.wrap(source="torch", target="drjit")
        def compute_rss(tx_x, tx_y, tx_dir_x, tx_dir_y):
            for tx in self.scene.transmitters.values():
                # Denormalise position: [0,1] → [pos_min, pos_max]
                phys_x = tx_x * self.pos_range + self.pos_min
                phys_y = tx_y * self.pos_range + self.pos_min
                tx.position = [phys_x.array, phys_y.array, self.fixed_z]

                # Set orientation via look_at: target = position + direction
                # The look_at call is inside @dr.wrap so Dr.Jit traces it
                # and gradients flow through the rotation math.
                # dir_z is fixed (not a DrJit variable) — no gradient.
                target = [
                    phys_x.array + tx_dir_x.array,
                    phys_y.array + tx_dir_y.array,
                    self.fixed_z + direction[2].item(),
                ]
                tx.look_at(target)

            # Compute radio map
            rm = self.radio_solver(
                self.scene,
                cell_size=(1.0, 1.0),
                samples_per_tx=samples_per_tx,
                max_depth=max_depth,
                refraction=True,
                diffraction=True,
            )

            return rm.rss

        # Compute RSS tensor with gradient tracking
        # Only dir_x and dir_y flow through @dr.wrap; dir_z is a constant.
        rss = compute_rss(
            self.tx_x, self.tx_y,
            direction[0], direction[1],
        )

        # Stash a detached copy of the RSS for metric logging so that we
        # do NOT need a second radio-map evaluation per iteration.
        self._last_rss = rss.detach().clone()

        if use_soft_min:
            # --- Normalized SoftMin loss (new, numerically stable) ---------
            # Flatten the radio map to (1, num_cells) for normalized_softmin_loss
            loss = normalized_softmin_loss(
                rss.flatten().unsqueeze(0),
                temperature=temperature,
            )
        else:
            # --- Legacy hard-min loss (kept for comparison) ----------------
            min_rss = compute_min_rss_metric(rss)
            log_min_rss = rss_to_dbm(min_rss)
            loss = -log_min_rss / 100.0  # Scale down for stability

        return loss

    def _eval_barrier_radio_map(
        self,
        samples_per_tx: int = 1_000_000,
        max_depth: int = 13,
    ) -> None:
        """Run a detached radio-map evaluation as a DrJit evaluation barrier.

        In DrJit's *evaluated* loop mode the backward pass tries to compile
        the entire reverse-AD graph into one CUDA (PTX) kernel.  For complex
        scenes this kernel exceeds the PTX assembler's limits and compilation
        fails with an unrecoverable error.

        Inserting a second, **gradient-free** radio-map computation between
        the forward pass and ``loss.backward()`` forces DrJit's JIT to
        compile and execute intermediate kernels, effectively splitting the
        backward kernel into smaller, compilable pieces.

        The RSS values produced here are **not** used for gradient updates.
        Metrics are taken from ``self._last_rss`` (cached during the forward
        pass) for consistency with the gradient computation.

        ``dr.suspend_grad()`` / ``dr.resume_grad()`` ensure this evaluation
        does not pollute the AD tape that ``loss.backward()`` will traverse.
        """
        dr.suspend_grad()
        try:
            _rm = self.radio_solver(
                self.scene,
                cell_size=(1.0, 1.0),
                samples_per_tx=samples_per_tx,
                max_depth=max_depth,
                refraction=True,
                diffraction=True,
            )
            # Accessing .rss materialises the result on the GPU,
            # completing the evaluation barrier.
            _ = _rm.rss
        finally:
            dr.resume_grad()

    def optimize(
        self,
        num_iterations: int = 50,
        learning_rate: float = 0.5,
        samples_per_tx: int = 1_000_000,
        max_depth: int = 13,
        use_soft_min: bool = True,
        temperature: float = 0.2,
        coverage_threshold_dbm: float = -120.0,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Run gradient descent optimization.

        Args:
            num_iterations: Number of optimization steps
            learning_rate: Learning rate for gradient descent
            samples_per_tx: Number of ray tracing samples per iteration
            max_depth: Maximum ray tracing depth
            use_soft_min: Use soft minimum for better gradients
            temperature: Temperature for soft minimum
            coverage_threshold_dbm: Threshold for coverage metric
            verbose: Print progress

        Returns:
            final_position: Optimized AP position [x, y, z]
            final_min_rss: Final minimum RSS value
        """
        # Build parameter groups.
        #
        # Position LR compensation: the trainable tx_x, tx_y live in [0, 1]
        # (normalised).  Adam's per-step displacement ≈ lr in parameter
        # space, which maps to lr × pos_range metres.  Dividing by
        # pos_range keeps the user-facing 'learning_rate' in physical
        # units (≈ metres per Adam step).
        #
        # Orientation gets DIR_LR_MULTIPLIER× learning rate for faster
        # adaptation (its gradient magnitudes are typically smaller).
        DIR_LR_MULTIPLIER = 3.0
        pos_lr = learning_rate / self.pos_range
        param_groups = [
            {"params": [self.tx_x, self.tx_y], "lr": pos_lr},
        ]
        if self.optimize_orientation:
            param_groups.append(
                {"params": [self.tx_dir_xy], "lr": learning_rate * DIR_LR_MULTIPLIER},
            )

        # Create optimizer
        optimizer = torch.optim.AdamW(param_groups)

        init_dir = self.get_current_direction().detach().cpu().numpy()
        if verbose:
            print(f"Starting Gradient Descent Optimization (PyTorch + DrJit)")
            print(f"  Device: {self.device}")
            init_pos = self.get_full_position()
            print(
                f"  Initial position: ({init_pos[0]:.2f}, "
                f"{init_pos[1]:.2f}, {self.fixed_z:.2f})"
            )
            print(
                f"  Normalised pos:   ({self.tx_x.item():.4f}, "
                f"{self.tx_y.item():.4f})  [0–1 range]"
            )
            print(
                f"  Initial direction: ({init_dir[0]:.4f}, "
                f"{init_dir[1]:.4f}, {init_dir[2]:.4f})"
            )
            print(f"  Optimize orientation: {self.optimize_orientation}")
            print(f"  LR (position):   {pos_lr:.4f}  (lr={learning_rate} / range={self.pos_range:.0f})")
            if self.optimize_orientation:
                print(f"  LR (direction):  {learning_rate * DIR_LR_MULTIPLIER} ({DIR_LR_MULTIPLIER}×)")
            print(f"  Iterations: {num_iterations}")
            print(f"  Samples per iteration: {samples_per_tx}")
            print(f"  Use soft minimum: {use_soft_min}")
            if use_soft_min:
                print(f"  Temperature: {temperature}")
            print("-" * 70)

        start_time = time.time()

        for iteration in range(num_iterations):
            iter_start = time.time()

            # Zero gradients
            optimizer.zero_grad()

            # Compute loss with gradient tracking
            loss = self.compute_loss(
                samples_per_tx=samples_per_tx,
                max_depth=max_depth,
                use_soft_min=use_soft_min,
                temperature=temperature,
            )

            # ── DrJit evaluation barrier ─────────────────────────────
            # In evaluated loop mode, DrJit's backward pass compiles the
            # full reverse-AD graph into one PTX kernel.  For complex
            # scenes this kernel is too large and PTX compilation fails.
            # Running a *detached* radio-map evaluation between forward
            # and backward forces DrJit to flush intermediate kernels,
            # splitting the backward compilation into manageable pieces.
            # self._eval_barrier_radio_map(
            #     samples_per_tx=samples_per_tx,
            #     max_depth=max_depth,
            # )

            # ── Eval radio map for metrics (no gradient tracking) ────────
            # This is a separate forward pass for logging; it does NOT
            # participate in the computation graph.
            with torch.no_grad():
                cur_dir = self.get_current_direction()
                phys_pos = self.get_full_position()
                for tx in self.scene.transmitters.values():
                    tx.position = [float(phys_pos[0]), float(phys_pos[1]), self.fixed_z]
                    target = [
                        float(phys_pos[0]) + float(cur_dir[0].item()),
                        float(phys_pos[1]) + float(cur_dir[1].item()),
                        self.fixed_z + float(cur_dir[2].item()),
                    ]
                    tx.look_at(target)

                rm = self.radio_solver(
                    self.scene,
                    cell_size=(1.0, 1.0),
                    samples_per_tx=samples_per_tx,
                    max_depth=max_depth,
                    refraction=True,
                    diffraction=True,
                )
                rss = np.array(rm.rss)
                rss = torch.from_numpy(rss)

            # ── Backward pass ────────────────────────────────────────────
            loss.backward()

            # Check gradients — position
            grad_x = self.tx_x.grad.item() if self.tx_x.grad is not None else 0.0
            grad_y = self.tx_y.grad.item() if self.tx_y.grad is not None else 0.0

            # Check gradients — direction (xy only; z is fixed)
            if self.optimize_orientation and self.tx_dir_xy.grad is not None:
                dir_grad_xy = self.tx_dir_xy.grad.detach().cpu().numpy()
                dir_grad = np.array([dir_grad_xy[0], dir_grad_xy[1], 0.0])
            else:
                dir_grad = np.zeros(3)

            # ── NaN guard ────────────────────────────────────────────────
            # DrJit can occasionally return NaN gradients (e.g. degenerate
            # ray paths).  Replace NaN with zero to keep Adam state clean.
            nan_detected = False
            if self.tx_x.grad is not None and torch.isnan(self.tx_x.grad):
                self.tx_x.grad.zero_()
                grad_x = 0.0
                nan_detected = True
            if self.tx_y.grad is not None and torch.isnan(self.tx_y.grad):
                self.tx_y.grad.zero_()
                grad_y = 0.0
                nan_detected = True
            if self.optimize_orientation and self.tx_dir_xy.grad is not None:
                if torch.any(torch.isnan(self.tx_dir_xy.grad)):
                    self.tx_dir_xy.grad.zero_()
                    dir_grad = np.zeros(3)
                    nan_detected = True
            if nan_detected and verbose:
                print(f"WARNING: NaN gradients detected at iteration {iteration+1}, zeroed out")

            if abs(grad_x) < 1e-12 and abs(grad_y) < 1e-12 and not nan_detected and verbose:
                print(f"WARNING: Position gradients are near zero at iteration {iteration+1}")

            # Apply gradients
            optimizer.step()

            # No explicit re-normalisation of tx_dir_xy needed here.
            # get_current_direction() always normalises the full (dx, dy, fixed_dz)
            # vector, so the unit-sphere constraint is enforced at forward time.
            # Allowing the raw xy magnitudes to vary freely lets Adam maintain
            # proper momentum/variance estimates without interference.

            # Apply position constraints
            self.apply_position_constraints()

            # ── NaN position guard ───────────────────────────────────────
            # If tx_x or tx_y became NaN (e.g. from residual NaN in Adam
            # momentum buffers), reset to the last valid position.
            with torch.no_grad():
                if torch.isnan(self.tx_x) or torch.isnan(self.tx_y):
                    if self.history["positions"]:
                        last_pos = self.history["positions"][-1]
                        self.tx_x.fill_(self._normalize(last_pos[0]))
                        self.tx_y.fill_(self._normalize(last_pos[1]))
                    else:
                        self.tx_x.fill_(0.5)
                        self.tx_y.fill_(0.5)
                    if verbose:
                        print(f"WARNING: NaN position at iteration {iteration+1}, reset to last valid")

            # # ── Eval radio map for metrics (no gradient tracking) ────────
            # # This is a separate forward pass for logging; it does NOT
            # # participate in the computation graph.
            # with torch.no_grad():
            #     cur_dir = self.get_current_direction()
            #     phys_pos = self.get_full_position()
            #     for tx in self.scene.transmitters.values():
            #         tx.position = [float(phys_pos[0]), float(phys_pos[1]), self.fixed_z]
            #         target = [
            #             float(phys_pos[0]) + float(cur_dir[0].item()),
            #             float(phys_pos[1]) + float(cur_dir[1].item()),
            #             self.fixed_z + float(cur_dir[2].item()),
            #         ]
            #         tx.look_at(target)

            #     rm = self.radio_solver(
            #         self.scene,
            #         cell_size=(1.0, 1.0),
            #         samples_per_tx=samples_per_tx,
            #         max_depth=max_depth,
            #         refraction=True,
            #         diffraction=True,
            #     )
            #     rss = np.array(rm.rss)
            #     rss = torch.from_numpy(rss)

            # Compute metrics for logging
            min_rss = compute_min_rss_metric(rss.cpu())
            min_rss_dbm = rss_to_dbm(min_rss)
            coverage = compute_coverage_metric(rss, coverage_threshold_dbm)

            # Store history
            current_pos = self.get_full_position()
            current_dir = self.get_current_direction().detach().cpu().numpy()
            # Compute the actual look_at target point for debugging
            look_at_target = current_pos + current_dir
            self.history["positions"].append(current_pos.copy())
            self.history["directions"].append(current_dir.copy())
            self.history["look_at_targets"].append(look_at_target.copy())
            self.history["min_rss_values"].append(min_rss.item())
            self.history["min_rss_dbm_values"].append(min_rss_dbm.item())
            self.history["coverage_values"].append(coverage.item())
            self.history["losses"].append(loss.item())
            self.history["gradients"].append([grad_x, grad_y])
            self.history["direction_gradients"].append(dir_grad.tolist())

            iter_time = time.time() - iter_start

            if verbose:
                pos_grad_norm = np.sqrt(grad_x**2 + grad_y**2)
                dir_grad_norm = float(np.linalg.norm(dir_grad))
                print(
                    f"Iter {iteration+1:3d}/{num_iterations} | "
                    f"Pos: ({current_pos[0]:.2f}, {current_pos[1]:.2f}) | "
                    f"Dir: ({current_dir[0]:.3f}, {current_dir[1]:.3f}, {current_dir[2]:.3f}) | "
                    f"LookAt: ({look_at_target[0]:.2f}, {look_at_target[1]:.2f}, {look_at_target[2]:.2f}) | "
                    f"Min RSS: {min_rss_dbm:.2f} dBm | "
                    f"Cov: {coverage:.1f}% | "
                    f"Loss: {loss.item():.2e} | "
                    f"∇pos: {pos_grad_norm:.2e} | "
                    f"∇dir: {dir_grad_norm:.2e} | "
                    f"Time: {iter_time:.1f}s"
                )

        elapsed_time = time.time() - start_time
        final_position = self.get_full_position()
        final_min_rss = self.history["min_rss_values"][-1]

        final_dir = self.get_current_direction().detach().cpu().numpy()
        final_look_at = final_position + final_dir

        if verbose:
            print("-" * 70)
            print(f"Gradient Descent Complete!")
            print(
                f"  Initial position: ({self.history['positions'][0][0]:.2f}, "
                f"{self.history['positions'][0][1]:.2f}, {self.history['positions'][0][2]:.2f})"
            )
            print(
                f"  Final position: ({final_position[0]:.2f}, {final_position[1]:.2f}, {final_position[2]:.2f})"
            )
            init_d = self.history["directions"][0]
            print(
                f"  Initial direction: ({init_d[0]:.4f}, {init_d[1]:.4f}, {init_d[2]:.4f})"
            )
            print(
                f"  Final direction:   ({final_dir[0]:.4f}, {final_dir[1]:.4f}, {final_dir[2]:.4f})"
            )
            init_la = self.history["look_at_targets"][0]
            print(
                f"  Initial look_at:   ({init_la[0]:.2f}, {init_la[1]:.2f}, {init_la[2]:.2f})"
            )
            print(
                f"  Final look_at:     ({final_look_at[0]:.2f}, {final_look_at[1]:.2f}, {final_look_at[2]:.2f})"
            )
            # Also report best iteration orientation
            if self.history["min_rss_values"]:
                best_idx = int(np.argmax(self.history["min_rss_values"]))
                best_d = self.history["directions"][best_idx]
                best_la = self.history["look_at_targets"][best_idx]
                best_p = self.history["positions"][best_idx]
                print(
                    f"  Best iter {best_idx+1} pos:     ({best_p[0]:.2f}, {best_p[1]:.2f}, {best_p[2]:.2f})"
                )
                print(
                    f"  Best iter {best_idx+1} dir:     ({best_d[0]:.4f}, {best_d[1]:.4f}, {best_d[2]:.4f})"
                )
                print(
                    f"  Best iter {best_idx+1} look_at: ({best_la[0]:.2f}, {best_la[1]:.2f}, {best_la[2]:.2f})"
                )
            print(f"  Initial min RSS: {self.history['min_rss_dbm_values'][0]:.2f} dBm")
            final_min_rss_dbm = rss_to_dbm(torch.tensor(final_min_rss)).item()
            print(f"  Final min RSS: {final_min_rss_dbm:.2f} dBm")
            print(
                f"  Improvement: {self.history['min_rss_dbm_values'][-1] - self.history['min_rss_dbm_values'][0]:.2f} dB"
            )
            print(f"  Final coverage: {self.history['coverage_values'][-1]:.1f}%")
            print(f"  Total time: {elapsed_time:.2f}s")
            print(f"  Time per iteration: {elapsed_time/num_iterations:.2f}s")

        return final_position, final_min_rss

    def plot_results(self, **kwargs) -> None:
        """
        Plot the optimization results.
        
        Implements the abstract method from BaseAPOptimizer by plotting
        the optimization trajectory and convergence metrics.
        
        Args:
            **kwargs: Not used, accepts for interface compatibility
        """
        self.plot_optimization_trajectory()
    
    def plot_optimization_trajectory(self) -> None:
        """Plot the optimization trajectory."""
        positions = np.array(self.history["positions"])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Position trajectory
        ax = axes[0, 0]
        ax.plot(positions[:, 0], positions[:, 1], "b-o", markersize=4, linewidth=1.5)
        ax.plot(positions[0, 0], positions[0, 1], "go", markersize=12, label="Start")
        ax.plot(positions[-1, 0], positions[-1, 1], "r*", markersize=15, label="End")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("AP Position Trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Min RSS over iterations
        ax = axes[0, 1]
        ax.plot(self.history["min_rss_dbm_values"], "b-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Min RSS (dBm)")
        ax.set_title("Minimum RSS Evolution")
        ax.grid(True, alpha=0.3)

        # 3. Coverage over iterations
        ax = axes[1, 0]
        ax.plot(self.history["coverage_values"], "g-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Coverage (%)")
        ax.set_title("Coverage Evolution")
        ax.grid(True, alpha=0.3)

        # 4. Gradient norms over iterations (position + direction)
        ax = axes[1, 1]
        pos_grad_norms = [np.sqrt(g[0] ** 2 + g[1] ** 2) for g in self.history["gradients"]]
        ax.semilogy(pos_grad_norms, "r-", linewidth=2, label="∇pos")
        if self.history.get("direction_gradients"):
            dir_grad_norms = [
                np.linalg.norm(g) for g in self.history["direction_gradients"]
            ]
            ax.semilogy(dir_grad_norms, "m--", linewidth=2, label="∇dir")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm Evolution (log scale)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Optional 5th figure: direction components over iterations ---
        if self.history.get("directions") and len(self.history["directions"]) > 0:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            dirs = np.array(self.history["directions"])
            ax2.plot(dirs[:, 0], label="dx", linewidth=2)
            ax2.plot(dirs[:, 1], label="dy", linewidth=2)
            ax2.plot(dirs[:, 2], label="dz", linewidth=2)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Direction Component")
            ax2.set_title("AP Look-At Direction Evolution")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()

        plt.tight_layout()
        plt.show()
