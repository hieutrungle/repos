"""
Gradient descent optimizer for AP position using differentiable ray tracing.

This module implements gradient-based optimization using PyTorch and DrJit
to leverage Sionna's differentiable ray tracing capabilities.
"""

import time
from typing import Dict, Tuple

import numpy as np
import torch
import drjit as dr
import matplotlib.pyplot as plt
import sionna.rt
from sionna.rt import RadioMapSolver
import copy

from .base_optimizer import BaseAPOptimizer
from ..metrics import (
    compute_min_rss_metric,
    compute_soft_min_rss_metric,
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
    ):
        """
        Initialize gradient descent optimizer.

        Args:
            scene: Sionna Scene object
            initial_position: [x, y] initial AP position (z is fixed)
            fixed_z: Fixed height for AP (z-coordinate)
            position_bounds: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max' for constraints
        """
        super().__init__(scene=scene, fixed_z=fixed_z, position_bounds=position_bounds)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize trainable position using PyTorch tensors on appropriate device
        self.tx_x = torch.tensor(
            initial_position[0], dtype=torch.float32, requires_grad=True, device=self.device
        )
        self.tx_y = torch.tensor(
            initial_position[1], dtype=torch.float32, requires_grad=True, device=self.device
        )
        self.radio_solver = RadioMapSolver()
        self.radio_solver.loop_mode = "evaluated"  # Needed for gradient computation

        # History tracking
        self.history = {
            "positions": [],
            "min_rss_values": [],
            "min_rss_dbm_values": [],
            "coverage_values": [],
            "losses": [],
            "gradients": [],
        }

    def get_full_position(self) -> np.ndarray:
        """Get full 3D position [x, y, z] as numpy array."""
        return np.array([self.tx_x.item(), self.tx_y.item(), self.fixed_z])

    def apply_position_constraints(self) -> None:
        """Project position back to valid bounds if constraints are specified."""
        if self.position_bounds and len(self.position_bounds) > 0:
            with torch.no_grad():
                self.tx_x.clamp_(self.position_bounds["x_min"], self.position_bounds["x_max"])
                self.tx_y.clamp_(self.position_bounds["y_min"], self.position_bounds["y_max"])

    def compute_loss(
        self,
        samples_per_tx: int = 100_000,
        max_depth: int = 10,
        use_soft_min: bool = True,
        temperature: float = 0.1,
    ) -> Tuple[torch.Tensor, sionna.rt.RadioMap]:
        """
        Compute loss for current AP position with differentiable ray tracing.

        This uses @dr.wrap to bridge PyTorch and DrJit for gradient computation through
        the RadioMapSolver.

        Args:
            samples_per_tx: Number of ray tracing samples
            max_depth: Maximum ray tracing depth
            use_soft_min: Use soft minimum (better gradients) vs hard minimum
            temperature: Temperature for soft minimum (lower = closer to hard min)

        Returns:
            loss: Negative of min RSS (we minimize loss = maximize min RSS)
            rm: Radio map object (computed outside of gradient tape)
        """

        # Wrap the radio map computation for gradient flow
        @dr.wrap(source="torch", target="drjit")
        def compute_rss(tx_x, tx_y):
            # Update transmitter position using DrJit arrays
            for tx in self.scene.transmitters.values():
                tx.position = [tx_x.array, tx_y.array, self.fixed_z]

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

        # Compute metric with gradient tracking
        rss = compute_rss(self.tx_x, self.tx_y)
        if use_soft_min:
            min_rss = compute_soft_min_rss_metric(rss, temperature=temperature)
        else:
            min_rss = compute_min_rss_metric(rss)

        # Convert to dBm scale for better loss landscape
        log_min_rss = rss_to_dbm(min_rss)
        average_rss_dbm = torch.mean(rss_to_dbm(rss))

        # Loss is negative of min RSS (minimize loss = maximize min RSS)
        loss = -(log_min_rss + average_rss_dbm) / 100.0  # Scale down for stability

        return loss

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
        # Create optimizer
        optimizer = torch.optim.Adam([self.tx_x, self.tx_y], lr=learning_rate)

        if verbose:
            print(f"Starting Gradient Descent Optimization (PyTorch + DrJit)")
            print(f"  Device: {self.device}")
            print(
                f"  Initial position: ({self.tx_x.item():.2f}, "
                f"{self.tx_y.item():.2f}, {self.fixed_z:.2f})"
            )
            print(f"  Learning rate: {learning_rate}")
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
            
            with torch.no_grad():
                for tx in self.scene.transmitters.values():
                    tx.position = [float(self.tx_x.item()), float(self.tx_y.item()), self.fixed_z]

                rm = self.radio_solver(
                    self.scene,
                    cell_size=(1.0, 1.0),
                    samples_per_tx=int(samples_per_tx / 10),  # Use fewer samples for logging
                    max_depth=max_depth,
                    refraction=True,
                    diffraction=True,
                )
                rss = np.array(rm.rss)
                rss = torch.from_numpy(rss)

            # Backward pass to compute gradients
            loss.backward()

            # Check gradients
            grad_x = self.tx_x.grad.item() if self.tx_x.grad is not None else 0.0
            grad_y = self.tx_y.grad.item() if self.tx_y.grad is not None else 0.0

            if abs(grad_x) < 1e-12 and abs(grad_y) < 1e-12 and verbose:
                print(f"WARNING: Gradients are near zero at iteration {iteration+1}")

            # Apply gradients
            optimizer.step()

            # Apply position constraints
            self.apply_position_constraints()
            

            # Compute metrics for logging
            # Compute radio map separately for visualization (without gradient tracking)
            
            min_rss = compute_min_rss_metric(rss.clone().detach().cpu())
            min_rss_dbm = rss_to_dbm(min_rss)
            coverage = compute_coverage_metric(rss, coverage_threshold_dbm)

            # Store history
            current_pos = self.get_full_position()
            self.history["positions"].append(current_pos.copy())
            self.history["min_rss_values"].append(min_rss.item())
            self.history["min_rss_dbm_values"].append(min_rss_dbm.item())
            self.history["coverage_values"].append(coverage.item())
            self.history["losses"].append(loss.item())
            self.history["gradients"].append([grad_x, grad_y])

            iter_time = time.time() - iter_start

            if verbose:
                grad_norm = np.sqrt(grad_x**2 + grad_y**2)
                print(
                    f"Iter {iteration+1:3d}/{num_iterations} | "
                    f"Pos: ({current_pos[0]:.2f}, {current_pos[1]:.2f}) | "
                    f"P5 RSS: {min_rss_dbm:.2f} dBm | "
                    f"Coverage: {coverage:.1f}% | "
                    f"Loss: {loss.item():.2e} | "
                    f"Grad norm: {grad_norm:.2e} | "
                    f"Time: {iter_time:.1f}s"
                )

        elapsed_time = time.time() - start_time
        final_position = self.get_full_position()
        final_min_rss = self.history["min_rss_values"][-1]

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

        # 2. P5 RSS over iterations
        ax = axes[0, 1]
        ax.plot(self.history["min_rss_dbm_values"], "b-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("P5 RSS (dBm)")
        ax.set_title("Minimum RSS Evolution")
        ax.grid(True, alpha=0.3)

        # 3. Coverage over iterations
        ax = axes[1, 0]
        ax.plot(self.history["coverage_values"], "g-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Coverage (%)")
        ax.set_title("Coverage Evolution")
        ax.grid(True, alpha=0.3)

        # 4. Gradient norm over iterations
        ax = axes[1, 1]
        grad_norms = [np.sqrt(g[0] ** 2 + g[1] ** 2) for g in self.history["gradients"]]
        ax.semilogy(grad_norms, "r-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm Evolution (log scale)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
