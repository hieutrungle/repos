"""Dedicated gradient descent optimizer for the memetic fusion pipeline.

This module implements a focused gradient-based optimizer that minimizes the
same ``MemeticCompositeLoss`` landscape used by the memetic GA evaluator.
The implementation intentionally excludes legacy fairness switches and masked
objectives so the optimization surface remains a single, explicit manifold.

Design goals
------------
1. Use one universal objective: ``MemeticCompositeLoss``.
2. Keep the optimizer lean and specific to the memetic pipeline.
3. Preserve multi-AP repulsion to avoid AP collapse.
4. Compute human-readable physical metrics only under ``torch.no_grad()`` for
   logging and analysis, never for backpropagation.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import drjit as dr
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import sionna.rt
from sionna.rt import RadioMapSolver, Transmitter

from reflector_position.metrics import (
    compute_thresholded_reporting_metrics,
)
from reflector_position.optimizers.base_optimizer import BaseAPOptimizer
from reflector_position.optimizers.memetic.memetic_loss import MemeticCompositeLoss
from reflector_position.reflector_model import ReflectorController


class MemeticGradientDescentOptimizer(BaseAPOptimizer):
    """Memetic GD optimizer built around one composite differentiable loss.

    Parameters
    ----------
    scene : sionna.rt.Scene
        Scene to optimize in-place.
    initial_positions : list[tuple[float, float]] | tuple[float, float] | None
        Initial AP XY coordinates. A bare ``(x, y)`` tuple is treated as a
        single-AP configuration.
    initial_position : tuple[float, float] | None
        Backward-compatible single-AP alias.
    num_aps : int | None
        Number of APs if positions must be auto-generated.
    fixed_z : float
        Shared AP height.
    position_bounds : dict[str, float] | None
        Optional XY clamp bounds.
    initial_directions_xy : list[tuple[float, float]] | None
        Optional per-AP XY look directions.
    initial_direction_xy : tuple[float, float] | None
        Optional single direction replicated across APs.
    fixed_dir_z : float
        Shared Z component for normalized AP look directions.
    optimize_orientation : bool
        Whether AP directions are trainable.
    repulsion_weight : float
        Pairwise inverse-square repulsion multiplier for multi-AP runs.
    reflector_controller : ReflectorController | None
        Optional reflector controller reused from the loaded worker scene.
    reflector_u : float | None
        Initial reflector wall-coordinate ``u`` in [0, 1].
    reflector_v : float | None
        Initial reflector wall-coordinate ``v`` in [0, 1].
    reflector_target : tuple[float, float, float] | None
        Initial reflector focal point.
    initial_focal_point : tuple[float, float, float] | None
        Alias for the reflector focal point used by bridge outputs.
    """

    def __init__(
        self,
        scene: sionna.rt.Scene,
        initial_positions: Union[List[Tuple[float, float]], Tuple[float, float], None] = None,
        initial_position: Optional[Tuple[float, float]] = None,
        num_aps: Optional[int] = None,
        fixed_z: float = 3.8,
        position_bounds: Optional[Dict[str, float]] = None,
        initial_directions_xy: Optional[List[Tuple[float, float]]] = None,
        initial_direction_xy: Optional[Tuple[float, float]] = None,
        fixed_dir_z: float = -0.3,
        optimize_orientation: bool = True,
        repulsion_weight: float = 1.0,
        reflector_controller: Optional[ReflectorController] = None,
        reflector_u: Optional[float] = None,
        reflector_v: Optional[float] = None,
        reflector_target: Optional[Tuple[float, float, float]] = None,
        initial_focal_point: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        super().__init__(scene=scene, fixed_z=fixed_z, position_bounds=position_bounds)

        self.device = (
            reflector_controller.device
            if reflector_controller is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.optimize_orientation = bool(optimize_orientation)
        self.repulsion_weight = float(repulsion_weight)
        self.fixed_dir_z = float(fixed_dir_z)

        positions = self._resolve_initial_positions(
            initial_positions=initial_positions,
            initial_position=initial_position,
            num_aps=num_aps,
        )
        self.num_aps = len(positions)
        self._ensure_transmitters(positions)

        self.tx_x = torch.tensor(
            [float(position[0]) for position in positions],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.tx_y = torch.tensor(
            [float(position[1]) for position in positions],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        direction_inits = self._resolve_initial_directions(
            initial_directions_xy=initial_directions_xy,
            initial_direction_xy=initial_direction_xy,
        )
        self.tx_dir_xy = torch.tensor(
            direction_inits,
            dtype=torch.float32,
            device=self.device,
            requires_grad=self.optimize_orientation,
        )

        self.reflector_controller = reflector_controller
        self.reflector_u_raw: Optional[torch.Tensor] = None
        self.reflector_v_raw: Optional[torch.Tensor] = None
        self.reflector_target: Optional[torch.Tensor] = None
        self._configure_reflector_params(
            reflector_u=reflector_u,
            reflector_v=reflector_v,
            reflector_target=reflector_target,
            initial_focal_point=initial_focal_point,
        )

        self.radio_solver = RadioMapSolver()
        self.radio_solver.loop_mode = "evaluated"
        self.loss_module = MemeticCompositeLoss(
            alpha=0.99,
            beta=0.01,
            softmin_temperature=0.15,
            coverage_threshold_dbm=-120.0,
            coverage_temperature=2.0,
        )

        self.history: Dict[str, List[Any]] = {
            "positions": [],
            "directions": [],
            "primary_loss": [],
            "loss_components": [],
            "physical_metrics": [],
            "repulsion_losses": [],
            "gradients": [],
            "direction_gradients": [],
            "ap_distances": [],
            "reflector_u": [],
            "reflector_v": [],
            "reflector_target": [],
            "reflector_position": [],
        }
        self.results: Dict[str, Any] = {}

    def optimize(
        self,
        num_iterations: int = 50,
        learning_rate: float = 0.1,
        samples_per_tx: int = 1_000_000,
        max_depth: int = 13,
        temperature: float = 0.1,
        softmin_temperature: Optional[float] = None,
        alpha: float = 0.99,
        beta: float = 0.01,
        coverage_threshold_dbm: float = -120.0,
        coverage_temperature: float = 2.0,
        verbose: bool = True,
        use_soft_min: bool = True,
        shadow_quantile: float = 0.05,
        fairness_loss_type: str = "auto",
        **_: Any,
    ) -> Tuple[np.ndarray, float]:
        """Run memetic gradient descent on the universal composite loss.

        Parameters other than the composite-loss hyperparameters are retained
        only for pipeline compatibility. Legacy fairness toggles are accepted
        but intentionally ignored.
        """
        del use_soft_min, shadow_quantile, fairness_loss_type

        effective_softmin_temperature = (
            softmin_temperature if softmin_temperature is not None else temperature
        )

        self.loss_module = MemeticCompositeLoss(
            alpha=float(alpha),
            beta=float(beta),
            softmin_temperature=float(effective_softmin_temperature),
            coverage_threshold_dbm=float(coverage_threshold_dbm),
            coverage_temperature=float(coverage_temperature),
        )

        param_groups: List[Dict[str, Any]] = [
            {"params": [self.tx_x, self.tx_y], "lr": float(learning_rate)},
        ]
        if self.optimize_orientation:
            param_groups.append(
                {"params": [self.tx_dir_xy], "lr": float(learning_rate) * 2.0}
            )
        reflector_params = self._reflector_parameter_list()
        if reflector_params:
            param_groups.append(
                {"params": reflector_params, "lr": float(learning_rate) * 0.5}
            )

        # Keep the GD step aligned with the reported memetic objective.
        # AdamW's default weight decay adds an implicit regularizer that is not
        # part of the memetic loss and can make the logged loss rise even when
        # the optimizer is following its internal objective.
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0, betas=(0.0, 0.999))
        scheduler = CosineAnnealingLR(optimizer, T_max=int(num_iterations), eta_min=1e-5)

        start_time = time.time()
        with torch.no_grad():
            initial_snapshot = self._evaluate_metrics(
                samples_per_tx=int(samples_per_tx),
                max_depth=int(max_depth),
            )
            self._record_history(initial_snapshot)

        best_snapshot = self._clone_snapshot(initial_snapshot)
        for iteration in range(int(num_iterations)):
            optimizer.zero_grad()
            total_loss, loss_components = self.compute_loss(
                samples_per_tx=int(samples_per_tx),
                max_depth=int(max_depth),
            )
            
            # print(f"Iteration {iteration + 1}/{num_iterations} - Loss: {total_loss.item():.6f}")
            with torch.no_grad():
                rm = self.radio_solver(
                    self.scene,
                    cell_size=(5.0, 5.0),
                    samples_per_tx=100_000,
                    max_depth=3,
                    refraction=False,
                    diffraction=False,
                )
                rss = torch.from_numpy(np.array(rm.rss))
            
            total_loss.backward()

            if reflector_params:
                self._estimate_reflector_gradients(
                    samples_per_tx=int(samples_per_tx),
                    max_depth=int(max_depth),
                )

            self._sanitize_gradients()
            optimizer.step()
            self._apply_position_constraints()

            with torch.no_grad():
                snapshot = self._evaluate_metrics(
                    samples_per_tx=int(samples_per_tx),
                    max_depth=int(max_depth),
                )
                self._record_history(snapshot)
                if snapshot["primary_loss"] < best_snapshot["primary_loss"]:
                    best_snapshot = self._clone_snapshot(snapshot)

            if verbose:
                self._log_iteration(iteration + 1, int(num_iterations), snapshot)
                scheduler.step()
                scheduler.zero_grad()

        with torch.no_grad():
            snapshot = self._evaluate_metrics(
                samples_per_tx=int(samples_per_tx),
                max_depth=int(max_depth),
            )
            self._record_history(snapshot)
            if snapshot["primary_loss"] < best_snapshot["primary_loss"]:
                best_snapshot = self._clone_snapshot(snapshot)

        elapsed_time = time.time() - start_time
        final_positions = self.get_full_positions()
        final_snapshot = self._clone_snapshot(self._history_snapshot(-1))

        self.results = {
            "num_aps": self.num_aps,
            "best_configuration": {
                "positions": np.asarray(best_snapshot["positions"]).tolist(),
                "directions": np.asarray(best_snapshot["directions"]).tolist(),
                "reflector": best_snapshot["reflector_snapshot"],
            },
            "primary_loss": float(best_snapshot["primary_loss"]),
            "loss_components": dict(best_snapshot["loss_components"]),
            "physical_metrics": dict(best_snapshot["physical_metrics"]),
            "final_configuration": {
                "positions": np.asarray(final_snapshot["positions"]).tolist(),
                "directions": np.asarray(final_snapshot["directions"]).tolist(),
                "reflector": final_snapshot["reflector_snapshot"],
            },
            "final_primary_loss": float(final_snapshot["primary_loss"]),
            "final_loss_components": dict(final_snapshot["loss_components"]),
            "final_physical_metrics": dict(final_snapshot["physical_metrics"]),
            "elapsed_time": elapsed_time,
            "positions": final_positions.tolist() if self.num_aps > 1 else final_positions[0].tolist(),
        }

        if verbose:
            print("-" * 80)
            print("MEMETIC GD COMPLETE")
            print(f"  Best primary loss: {best_snapshot['primary_loss']:.6f}")
            if best_snapshot["loss_components"]:
                print("  Best loss components:")
                for name, value in best_snapshot["loss_components"].items():
                    print(f"    {name}: {value:.6f}")
            if final_snapshot["physical_metrics"]:
                print("  Final physical metrics:")
                for name, value in final_snapshot["physical_metrics"].items():
                    print(f"    {name}: {value:.6f}")
            print(f"  Total time: {elapsed_time:.2f}s")

        if self.num_aps == 1:
            return final_positions[0], float(best_snapshot["primary_loss"])
        return final_positions, float(best_snapshot["primary_loss"])

    def compute_loss(
        self,
        samples_per_tx: int,
        max_depth: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the memetic composite loss plus multi-AP repulsion."""
        if self.reflector_controller is not None:
            self._apply_reflector_state()

        directions = self.get_current_directions()
        dir_z_values = [float(directions[i, 2].item()) for i in range(self.num_aps)]
        tx_list = [self.scene.transmitters[name] for name in self.tx_names]

        wrap_args: List[torch.Tensor] = []
        wrap_args.extend(self.tx_x[i] for i in range(self.num_aps))
        wrap_args.extend(self.tx_y[i] for i in range(self.num_aps))
        wrap_args.extend(directions[i, 0] for i in range(self.num_aps))
        wrap_args.extend(directions[i, 1] for i in range(self.num_aps))

        @dr.wrap(source="torch", target="drjit")
        def compute_rss(*args: torch.Tensor) -> torch.Tensor:
            n_aps = self.num_aps
            for index in range(n_aps):
                tx = tx_list[index]
                x_coord = args[index]
                y_coord = args[n_aps + index]
                dir_x = args[2 * n_aps + index]
                dir_y = args[3 * n_aps + index]

                tx.position = [x_coord.array, y_coord.array, self.fixed_z]
                target = [
                    x_coord.array + dir_x.array,
                    y_coord.array + dir_y.array,
                    self.fixed_z + dir_z_values[index],
                ]
                tx.look_at(target)

            radio_map = self.radio_solver(
                self.scene,
                cell_size=(1.0, 1.0),
                samples_per_tx=samples_per_tx,
                max_depth=max_depth,
                refraction=True,
                diffraction=True,
            )
            return radio_map.rss

        coverage_map = compute_rss(*wrap_args)
        total_loss, loss_components = self.loss_module(coverage_map)

        repulsion_loss = self._compute_repulsion_loss()
        total_loss = total_loss + self.repulsion_weight * repulsion_loss

        merged_components = dict(loss_components)
        merged_components["repulsion_loss"] = float(repulsion_loss.detach().item())
        return total_loss, merged_components

    def plot_results(self, **kwargs: Any) -> None:
        """No-op placeholder for API compatibility."""
        del kwargs

    def get_full_positions(self) -> np.ndarray:
        """Return all AP positions as a ``[num_aps, 3]`` NumPy array."""
        x_coords = self.tx_x.detach().cpu().numpy()
        y_coords = self.tx_y.detach().cpu().numpy()
        z_coords = np.full_like(x_coords, self.fixed_z)
        return np.stack([x_coords, y_coords, z_coords], axis=-1)

    def get_current_directions(self) -> torch.Tensor:
        """Return normalized AP look directions of shape ``[num_aps, 3]``."""
        dir_z = torch.full(
            (self.num_aps, 1),
            self.fixed_dir_z,
            dtype=torch.float32,
            device=self.device,
        )
        full_dir = torch.cat([self.tx_dir_xy, dir_z], dim=1)
        return F.normalize(full_dir, dim=1)

    def _resolve_initial_positions(
        self,
        initial_positions: Union[List[Tuple[float, float]], Tuple[float, float], None],
        initial_position: Optional[Tuple[float, float]],
        num_aps: Optional[int],
    ) -> List[Tuple[float, float]]:
        """Resolve initial AP positions from the accepted constructor inputs."""
        if initial_positions is not None:
            if (
                isinstance(initial_positions, tuple)
                and len(initial_positions) == 2
                and all(isinstance(value, (int, float)) for value in initial_positions)
            ):
                return [
                    (float(initial_positions[0]), float(initial_positions[1]))
                ]
            return [
                (float(position[0]), float(position[1]))
                for position in initial_positions
            ]

        if initial_position is not None:
            return [(float(initial_position[0]), float(initial_position[1]))]

        effective_num_aps = 1 if num_aps is None else int(num_aps)
        return self._generate_spread_positions(effective_num_aps)

    def _resolve_initial_directions(
        self,
        initial_directions_xy: Optional[List[Tuple[float, float]]],
        initial_direction_xy: Optional[Tuple[float, float]],
    ) -> List[List[float]]:
        """Resolve per-AP initial XY directions."""
        if initial_directions_xy is not None:
            if len(initial_directions_xy) != self.num_aps:
                raise ValueError(
                    "'initial_directions_xy' must match the number of AP positions."
                )
            return [
                [float(direction[0]), float(direction[1])]
                for direction in initial_directions_xy
            ]

        if initial_direction_xy is not None:
            return [
                [float(initial_direction_xy[0]), float(initial_direction_xy[1])]
                for _ in range(self.num_aps)
            ]

        return [[1.0, 0.0] for _ in range(self.num_aps)]

    def _configure_reflector_params(
        self,
        reflector_u: Optional[float],
        reflector_v: Optional[float],
        reflector_target: Optional[Tuple[float, float, float]],
        initial_focal_point: Optional[Tuple[float, float, float]],
    ) -> None:
        """Initialize reflector trainable tensors when a controller is active."""
        if self.reflector_controller is None:
            return

        u_init = 0.5 if reflector_u is None else float(reflector_u)
        v_init = 0.5 if reflector_v is None else float(reflector_v)
        target_init = initial_focal_point or reflector_target
        if target_init is None:
            if self.position_bounds:
                x_center = 0.5 * (
                    float(self.position_bounds.get("x_min", 0.0))
                    + float(self.position_bounds.get("x_max", 0.0))
                )
                y_center = 0.5 * (
                    float(self.position_bounds.get("y_min", 0.0))
                    + float(self.position_bounds.get("y_max", 0.0))
                )
            else:
                x_center = float(self.tx_x[0].detach().cpu().item())
                y_center = float(self.tx_y[0].detach().cpu().item())
            target_init = (x_center, y_center, 1.5)

        self.reflector_u_raw = torch.tensor(
            self._inverse_sigmoid(u_init),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.reflector_v_raw = torch.tensor(
            self._inverse_sigmoid(v_init),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.reflector_target = torch.tensor(
            target_init,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

    def _reflector_parameter_list(self) -> List[torch.Tensor]:
        """Return active reflector parameters for the optimizer."""
        params: List[torch.Tensor] = []
        if self.reflector_u_raw is not None:
            params.append(self.reflector_u_raw)
        if self.reflector_v_raw is not None:
            params.append(self.reflector_v_raw)
        if self.reflector_target is not None:
            params.append(self.reflector_target)
        return params

    def _ensure_transmitters(
        self,
        initial_positions: List[Tuple[float, float]],
    ) -> None:
        """Ensure the scene exposes one transmitter per AP."""
        existing = list(self.scene.transmitters.keys())
        if len(existing) >= self.num_aps:
            self.tx_names = existing[: self.num_aps]
            return

        ref_power = 5.0
        if existing:
            raw_power = list(self.scene.transmitters.values())[0].power_dbm
            try:
                ref_power = float(raw_power)
            except (TypeError, ValueError):
                ref_power = float(raw_power[0]) if hasattr(raw_power, "__getitem__") else 5.0

        for index in range(len(existing), self.num_aps):
            x_coord, y_coord = initial_positions[index]
            transmitter = Transmitter(
                name=f"Tx{index:02d}",
                position=[float(x_coord), float(y_coord), self.fixed_z],
                power_dbm=ref_power,
            )
            self.scene.add(transmitter)

        self.tx_names = list(self.scene.transmitters.keys())[: self.num_aps]

    def _generate_spread_positions(self, num_aps: int) -> List[Tuple[float, float]]:
        """Generate simple spread-out initial positions inside the bounds."""
        x_min = float(self.position_bounds.get("x_min", 5.0))
        x_max = float(self.position_bounds.get("x_max", 25.0))
        y_min = float(self.position_bounds.get("y_min", 5.0))
        y_max = float(self.position_bounds.get("y_max", 25.0))

        if num_aps <= 1:
            return [((x_min + x_max) * 0.5, (y_min + y_max) * 0.5)]

        x_values = np.linspace(x_min, x_max, num_aps)
        y_values = np.linspace(y_min, y_max, num_aps)
        return [(float(x_values[i]), float(y_values[::-1][i])) for i in range(num_aps)]

    def _apply_position_constraints(self) -> None:
        """Clamp physical XY coordinates to the configured bounds."""
        if not self.position_bounds:
            return

        x_min = float(self.position_bounds.get("x_min", float("-inf")))
        x_max = float(self.position_bounds.get("x_max", float("inf")))
        y_min = float(self.position_bounds.get("y_min", float("-inf")))
        y_max = float(self.position_bounds.get("y_max", float("inf")))

        with torch.no_grad():
            self.tx_x.clamp_(min=x_min, max=x_max)
            self.tx_y.clamp_(min=y_min, max=y_max)

    def _compute_repulsion_loss(self) -> torch.Tensor:
        """Compute inverse-square pairwise repulsion in physical space."""
        if self.num_aps <= 1 or self.repulsion_weight <= 0.0:
            return torch.tensor(0.0, device=self.device)

        repulsion = torch.tensor(0.0, device=self.device)
        eps = 1e-4
        for i, j in combinations(range(self.num_aps), 2):
            dx = self.tx_x[i] - self.tx_x[j]
            dy = self.tx_y[i] - self.tx_y[j]
            repulsion = repulsion + 1.0 / (dx * dx + dy * dy + eps)
        return repulsion

    def _apply_reflector_state(self) -> None:
        """Push current reflector parameters into the scene graph."""
        controller = self.reflector_controller
        if controller is None:
            return
        if self.reflector_u_raw is None or self.reflector_v_raw is None:
            return
        if self.reflector_target is None:
            return

        controller.u = torch.sigmoid(self.reflector_u_raw)
        controller.v = torch.sigmoid(self.reflector_v_raw)
        controller.set_tx_position(self.get_full_positions()[0])
        controller.set_focal_point(self.reflector_target, requires_grad=True)
        controller.orient_to_target()
        controller.apply_to_scene()

    def _evaluate_metrics(
        self,
        samples_per_tx: int,
        max_depth: int,
        primary_loss: Optional[float] = None,
        loss_components: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compute detached physical metrics and objective diagnostics."""
        tx_list = [self.scene.transmitters[name] for name in self.tx_names]
        directions = self.get_current_directions().detach().cpu().numpy()
        positions = self.get_full_positions()
        for index, tx in enumerate(tx_list):
            position = positions[index]
            direction = directions[index]
            tx.position = [float(position[0]), float(position[1]), float(position[2])]
            tx.look_at(
                [
                    float(position[0] + direction[0]),
                    float(position[1] + direction[1]),
                    float(position[2] + direction[2]),
                ]
            )

        if self.reflector_controller is not None:
            self._apply_reflector_state()

        radio_map = self.radio_solver(
            self.scene,
            cell_size=(1.0, 1.0),
            samples_per_tx=samples_per_tx,
            max_depth=max_depth,
            refraction=True,
            diffraction=True,
        )
        coverage_map = torch.from_numpy(np.array(radio_map.rss)).to(self.device)

        computed_total_loss, computed_loss_components = self.loss_module(coverage_map)
        physical_metrics = compute_thresholded_reporting_metrics(
            coverage_map,
            threshold_dbm=self.loss_module.coverage_loss.threshold_dbm,
        )
        repulsion_loss = float(self._compute_repulsion_loss().item())

        effective_loss_components = dict(
            computed_loss_components if loss_components is None else loss_components
        )
        effective_loss_components.setdefault("repulsion_loss", repulsion_loss)
        effective_primary_loss = (
            float(computed_total_loss.item()) + self.repulsion_weight * repulsion_loss
            if primary_loss is None
            else float(primary_loss)
        )

        reflector_snapshot = self._snapshot_reflector()
        return {
            "positions": positions,
            "directions": directions,
            "primary_loss": effective_primary_loss,
            "loss_components": effective_loss_components,
            "physical_metrics": physical_metrics,
            "reflector_snapshot": reflector_snapshot,
        }

    def _record_history(
        self,
        snapshot: Dict[str, Any],
    ) -> None:
        """Append one detached iteration snapshot to the optimizer history."""
        positions = np.asarray(snapshot["positions"])
        directions = np.asarray(snapshot["directions"])

        self.history["positions"].append(positions.copy())
        self.history["directions"].append(directions.copy())
        self.history["primary_loss"].append(float(snapshot["primary_loss"]))
        self.history["loss_components"].append(dict(snapshot["loss_components"]))
        self.history["physical_metrics"].append(dict(snapshot["physical_metrics"]))
        self.history["repulsion_losses"].append(
            float(snapshot["loss_components"].get("repulsion_loss", 0.0))
        )

        grad_x = (
            self.tx_x.grad.detach().cpu().numpy()
            if self.tx_x.grad is not None
            else np.zeros(self.num_aps)
        )
        grad_y = (
            self.tx_y.grad.detach().cpu().numpy()
            if self.tx_y.grad is not None
            else np.zeros(self.num_aps)
        )
        self.history["gradients"].append(
            [[float(grad_x[i]), float(grad_y[i])] for i in range(self.num_aps)]
        )

        if self.tx_dir_xy.grad is not None:
            dir_grad = self.tx_dir_xy.grad.detach().cpu().numpy()
            self.history["direction_gradients"].append(dir_grad.tolist())
        else:
            self.history["direction_gradients"].append(
                np.zeros((self.num_aps, 2)).tolist()
            )

        self.history["ap_distances"].append(self._pairwise_distances())

        reflector_snapshot = snapshot["reflector_snapshot"]
        self.history["reflector_u"].append(reflector_snapshot["u"])
        self.history["reflector_v"].append(reflector_snapshot["v"])
        self.history["reflector_target"].append(reflector_snapshot["target"])
        self.history["reflector_position"].append(reflector_snapshot["position"])

    def _log_iteration(
        self,
        iteration: int,
        num_iterations: int,
        snapshot: Dict[str, Any],
    ) -> None:
        """Print one concise iteration summary."""
        position_summary = " | ".join(
            f"AP{i}:({snapshot['positions'][i][0]:.2f},{snapshot['positions'][i][1]:.2f})"
            for i in range(self.num_aps)
        )
        component_summary = " | ".join(
            f"{name}:{value:.4f}"
            for name, value in snapshot["loss_components"].items()
        )
        print(
            f"Iter {iteration:3d}/{num_iterations} | {position_summary} | "
            f"Loss:{snapshot['primary_loss']:.4e}"
            + (f" | {component_summary}" if component_summary else "")
        )

    def _sanitize_gradients(self) -> None:
        """Zero out NaN gradients before the optimizer step."""
        params: List[torch.Tensor] = [self.tx_x, self.tx_y, self.tx_dir_xy]
        params.extend(self._reflector_parameter_list())
        for param in params:
            if param.grad is None:
                continue
            nan_mask = torch.isnan(param.grad)
            if torch.any(nan_mask):
                param.grad[nan_mask] = 0.0

    def _evaluate_loss_no_grad(
        self,
        samples_per_tx: int,
        max_depth: int,
    ) -> float:
        """Evaluate the current composite loss without building gradients."""
        with torch.no_grad():
            snapshot = self._evaluate_metrics(
                samples_per_tx=samples_per_tx,
                max_depth=max_depth,
            )
            return float(snapshot["primary_loss"])

    def _history_snapshot(self, index: int) -> Dict[str, Any]:
        """Build one standardized snapshot from the stored history buffers."""
        return {
            "positions": np.asarray(self.history["positions"][index]),
            "directions": np.asarray(self.history["directions"][index]),
            "primary_loss": float(self.history["primary_loss"][index]),
            "loss_components": dict(self.history["loss_components"][index]),
            "physical_metrics": dict(self.history["physical_metrics"][index]),
            "reflector_snapshot": {
                "u": self.history["reflector_u"][index],
                "v": self.history["reflector_v"][index],
                "target": self.history["reflector_target"][index],
                "position": self.history["reflector_position"][index],
            },
        }

    def _clone_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Clone a snapshot into JSON-friendly values for persistent storage."""
        return {
            "positions": np.asarray(snapshot["positions"]).copy(),
            "directions": np.asarray(snapshot["directions"]).copy(),
            "primary_loss": float(snapshot["primary_loss"]),
            "loss_components": dict(snapshot.get("loss_components", {})),
            "physical_metrics": dict(snapshot.get("physical_metrics", {})),
            "reflector_snapshot": dict(snapshot.get("reflector_snapshot", {})),
        }

    def _estimate_reflector_gradients(
        self,
        samples_per_tx: int,
        max_depth: int,
        perturbation_scale: float = 0.05,
    ) -> None:
        """Estimate reflector gradients with SPSA and assign them to ``.grad``."""
        params = self._reflector_parameter_list()
        if not params:
            return

        deltas = [
            (torch.bernoulli(torch.full_like(param, 0.5)) * 2.0 - 1.0)
            for param in params
        ]

        with torch.no_grad():
            for param, delta in zip(params, deltas):
                param.add_(perturbation_scale * delta)
        loss_plus = self._evaluate_loss_no_grad(samples_per_tx, max_depth)

        with torch.no_grad():
            for param, delta in zip(params, deltas):
                param.add_(-2.0 * perturbation_scale * delta)
        loss_minus = self._evaluate_loss_no_grad(samples_per_tx, max_depth)

        with torch.no_grad():
            for param, delta in zip(params, deltas):
                param.add_(perturbation_scale * delta)

        diff = loss_plus - loss_minus
        for param, delta in zip(params, deltas):
            grad_estimate = diff / (2.0 * perturbation_scale * delta)
            if param.grad is None:
                param.grad = grad_estimate.clone()
            else:
                param.grad.copy_(grad_estimate)

    def _pairwise_distances(self) -> List[float]:
        """Return all pairwise AP distances in meters."""
        if self.num_aps <= 1:
            return []

        positions = self.get_full_positions()
        distances: List[float] = []
        for i, j in combinations(range(self.num_aps), 2):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            distances.append(distance)
        return distances

    def _snapshot_reflector(self) -> Dict[str, Optional[Any]]:
        """Return current reflector parameters as JSON-friendly values."""
        if self.reflector_controller is None:
            return {
                "u": None,
                "v": None,
                "target": None,
                "position": None,
            }

        u_value = (
            float(torch.sigmoid(self.reflector_u_raw).detach().cpu().item())
            if self.reflector_u_raw is not None
            else None
        )
        v_value = (
            float(torch.sigmoid(self.reflector_v_raw).detach().cpu().item())
            if self.reflector_v_raw is not None
            else None
        )
        target_value = (
            self.reflector_target.detach().cpu().tolist()
            if self.reflector_target is not None
            else None
        )
        try:
            position_value = self.reflector_controller.get_position().tolist()
        except Exception:
            position_value = None

        return {
            "u": u_value,
            "v": v_value,
            "target": target_value,
            "position": position_value,
        }

    @staticmethod
    def _inverse_sigmoid(value: float) -> float:
        """Map a bounded value in ``(0, 1)`` into unconstrained logit space."""
        clipped = min(max(value, 1e-4), 1.0 - 1e-4)
        return float(np.log(clipped / (1.0 - clipped)))


__all__ = ["MemeticGradientDescentOptimizer"]