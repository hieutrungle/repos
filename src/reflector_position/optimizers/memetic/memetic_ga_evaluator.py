"""Static configuration evaluator for memetic GA chromosome scoring.

This module provides a lean, single-responsibility evaluator for the memetic
GA stage. It configures one exact AP and optional reflector state, runs one
radio-map forward pass, and scores the resulting coverage map using the shared
``MemeticCompositeLoss`` objective.

There is intentionally no sweep logic, no percentile fallback, and no legacy
reporting metric computation here. The class exists solely to evaluate one
static configuration on the same continuous loss manifold used by the GD
stage.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import sionna.rt
from sionna.rt import RadioMapSolver

from reflector_position.metrics import (
    compute_thresholded_reporting_metrics,
)
from reflector_position.optimizers.memetic.memetic_loss import MemeticCompositeLoss
from reflector_position.reflector_model import ReflectorController


class StaticConfigurationEvaluator:
    """Evaluate one AP and reflector configuration without building gradients.

    Parameters
    ----------
    scene : sionna.rt.Scene
        Scene whose transmitters are updated in-place before each evaluation.
    reflector_controller : ReflectorController | None
        Optional controller for passive reflector placement and orientation.
    loss_kwargs : Mapping[str, Any]
        Keyword arguments forwarded to :class:`MemeticCompositeLoss`.
    """

    def __init__(
        self,
        scene: sionna.rt.Scene,
        reflector_controller: Optional[ReflectorController],
        loss_kwargs: Mapping[str, Any],
    ) -> None:
        self.scene = scene
        self.reflector_controller = reflector_controller
        self.loss_kwargs = dict(loss_kwargs)
        self.loss_module = MemeticCompositeLoss(**self.loss_kwargs)
        self.solver = RadioMapSolver()
        if self.reflector_controller is not None:
            self.device = self.reflector_controller.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        """Evaluate a single static configuration and return standardized output.

        Parameters
        ----------
        task : Mapping[str, Any]
            Task dictionary describing one concrete state. Supported keys are:

            - ``initial_positions``: sequence of ``(x, y)`` AP positions
            - ``fixed_z``: shared AP height
            - ``initial_directions_xy``: optional sequence of ``(dx, dy)``
            - ``reflector_u`` / ``reflector_v`` / ``reflector_target``
            - ``samples_per_tx`` / ``max_depth`` for the radio-map solver

        Returns
        -------
        dict[str, Any]
            Dictionary containing primary fitness, auxiliary losses, and
            detached physical metrics.
        """
        with torch.no_grad():
            tx_positions = self._build_tx_positions(task)
            self._configure_transmitters(tx_positions, task.get("initial_directions_xy"))
            self._configure_reflector(tx_positions, task)

            radio_map = self.solver(
                self.scene,
                cell_size=(1.0, 1.0),
                samples_per_tx=int(task.get("samples_per_tx", 1_000_000)),
                max_depth=int(task.get("max_depth", 13)),
                refraction=True,
                diffraction=True,
            )
            coverage_map = torch.from_numpy(np.array(radio_map.rss)).to(self.device)

            total_loss, loss_components = self.loss_module(coverage_map)
            physical_metrics = self._compute_physical_metrics(coverage_map)
            return {
                "primary_fitness": -float(total_loss.item()),
                "loss_components": dict(loss_components),
                "physical_metrics": physical_metrics,
            }

    def _compute_physical_metrics(self, coverage_map: torch.Tensor) -> Dict[str, float]:
        """Compute detached physical observation metrics for one radio map."""
        metrics = compute_thresholded_reporting_metrics(
            coverage_map,
            threshold_dbm=self.loss_module.coverage_loss.threshold_dbm,
        )
        return dict(metrics)

    def _build_tx_positions(self, task: Mapping[str, Any]) -> list[list[float]]:
        """Convert task AP coordinates into 3-D transmitter positions."""
        raw_positions = task.get("initial_positions")
        if not isinstance(raw_positions, Sequence) or len(raw_positions) == 0:
            raise ValueError("task must include non-empty 'initial_positions'")

        fixed_z = float(task.get("fixed_z", 0.0))
        tx_positions: list[list[float]] = []
        for position in raw_positions:
            if not isinstance(position, Sequence) or len(position) < 2:
                raise ValueError("each entry in 'initial_positions' must contain at least (x, y)")
            tx_positions.append([
                float(position[0]),
                float(position[1]),
                fixed_z,
            ])
        return tx_positions

    def _configure_transmitters(
        self,
        tx_positions: Sequence[Sequence[float]],
        directions_xy: Any,
    ) -> None:
        """Update scene transmitter positions and optional look directions."""
        transmitters = list(self.scene.transmitters.values())
        if len(tx_positions) > len(transmitters):
            raise ValueError(
                f"task defines {len(tx_positions)} APs but scene only has {len(transmitters)} transmitters"
            )

        directions_list: Optional[Sequence[Any]]
        if directions_xy is None:
            directions_list = None
        elif isinstance(directions_xy, Sequence):
            directions_list = directions_xy
            if len(directions_list) != len(tx_positions):
                raise ValueError(
                    "'initial_directions_xy' must have the same length as 'initial_positions'"
                )
        else:
            raise ValueError("'initial_directions_xy' must be a sequence or None")

        for index, transmitter in enumerate(transmitters[: len(tx_positions)]):
            position = [float(coord) for coord in tx_positions[index]]
            transmitter.position = position

            if directions_list is None or directions_list[index] is None:
                continue

            direction_xy = directions_list[index]
            if not isinstance(direction_xy, Sequence) or len(direction_xy) < 2:
                raise ValueError(
                    "each entry in 'initial_directions_xy' must contain at least (dx, dy)"
                )

            dx = float(direction_xy[0])
            dy = float(direction_xy[1])
            target = [position[0] + dx, position[1] + dy, position[2]]
            transmitter.look_at(target)

    def _configure_reflector(
        self,
        tx_positions: Sequence[Sequence[float]],
        task: Mapping[str, Any],
    ) -> None:
        """Apply reflector state when a controller and task parameters exist."""
        controller = self.reflector_controller
        if controller is None:
            return

        required_keys = ("reflector_u", "reflector_v", "reflector_target")
        if not all(key in task and task[key] is not None for key in required_keys):
            return

        reflector_target = task["reflector_target"]
        if not isinstance(reflector_target, Sequence) or len(reflector_target) != 3:
            raise ValueError("'reflector_target' must be a 3-D point")

        controller.u = torch.tensor(
            float(task["reflector_u"]),
            dtype=torch.float32,
            device=self.device,
        )
        controller.v = torch.tensor(
            float(task["reflector_v"]),
            dtype=torch.float32,
            device=self.device,
        )
        controller.set_tx_position(np.asarray(tx_positions[0], dtype=np.float32))
        controller.set_focal_point(
            torch.tensor(reflector_target, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        controller.orient_to_target()
        controller.apply_to_scene()


__all__ = ["StaticConfigurationEvaluator"]