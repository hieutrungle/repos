"""Memetic-focused Ray parallel optimizer with raw worker outputs.

This module intentionally avoids the heavy post-processing logic from
``ray_parallel_optimizer.py``. Each worker returns raw optimizer artifacts,
and downstream memetic components are responsible for interpreting metrics,
plotting, and summaries.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import ray
import torch
from ray.util.actor_pool import ActorPool
from sionna.rt import RadioMapSolver

from reflector_position.metrics import POWER_EPSILON
from reflector_position.optimizers.memetic.demand_weights import (
    generate_spatial_weight_map,
)
from reflector_position.optimizers.memetic.memetic_ga_evaluator import (
    StaticConfigurationEvaluator,
)
from reflector_position.optimizers.memetic.memetic_gd_optimizer import (
    MemeticGradientDescentOptimizer,
)
from reflector_position.optimizers.optimizer_factory import OptimizerFactory


def _to_serializable(value: Any) -> Any:
    """Recursively convert values to Ray/JSON-friendly Python objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Mapping):
        return {str(k): _to_serializable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]

    if hasattr(value, "tolist"):
        try:
            return _to_serializable(value.tolist())
        except Exception:
            pass

    if hasattr(value, "item"):
        try:
            return _to_serializable(value.item())
        except Exception:
            pass

    # Last-resort fallback for custom objects that are not JSON serializable.
    return repr(value)


@ray.remote
class RawOptimizationWorker:
    """Reusable actor that returns raw optimizer payloads per task."""

    def __init__(
        self,
        worker_id: int,
        scene_config: Dict[str, Any],
        demand_config: Optional[Mapping[str, Any]] = None,
    ):
        self.worker_id = worker_id
        self._run_count = 0
        self.scene_config = dict(scene_config)
        self.demand_config = dict(demand_config or {})
        self.scene, self.reflector_controller = self._load_scene(self.scene_config)
        self.device = self._resolve_device()
        self.radio_map_geometry: Dict[str, Any] = {}
        self.spatial_weights = self._build_spatial_weights()

    def _load_scene(self, scene_config: Dict[str, Any]) -> Tuple[Any, Optional[Any]]:
        """Load scene and optional reflector controller."""
        from reflector_position.scene_setup import setup_building_floor_scene

        loaded = setup_building_floor_scene(
            scene_path=str(scene_config["scene_path"]),
            frequency=scene_config.get("frequency", 6e9),
            tx_positions=scene_config.get("tx_positions", None),
            tx_power_dbm=scene_config.get("tx_power_dbm", 5.0),
            rx_position=scene_config.get("rx_position", (16.0, 16.5, 1.5)),
            reflector_enabled=scene_config.get("reflector_enabled", False),
            reflector_size=tuple(scene_config.get("reflector_size", (2.0, 2.0))),
            wall_top_left=scene_config.get("wall_top_left", None),
            wall_bottom_right=scene_config.get("wall_bottom_right", None),
            focal_point=scene_config.get("focal_point", None),
            device=scene_config.get("device", "cuda"),
        )
        if isinstance(loaded, tuple) and len(loaded) == 2:
            scene, reflector_controller = loaded
            return scene, reflector_controller
        return loaded, None

    def _resolve_device(self) -> torch.device:
        """Resolve the torch device used for worker-local tensors."""
        if self.reflector_controller is not None:
            return torch.device(self.reflector_controller.device)

        configured_device = str(self.scene_config.get("device", "cuda"))
        if configured_device.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(configured_device)

    def _build_default_task(self) -> Dict[str, Any]:
        """Build a default task payload used for worker-level map initialization.

        This mirrors the configuration flow used by ``StaticConfigurationEvaluator``
        so the precomputed demand map is generated on the same scene state
        convention as GA/GD evaluations.
        """
        transmitters = list(self.scene.transmitters.values())
        if not transmitters:
            raise ValueError("Scene must contain at least one transmitter")

        first_tx_position = np.array(transmitters[0].position, dtype=np.float32)
        fixed_z = float(first_tx_position[2])
        task: Dict[str, Any] = {
            "initial_positions": [
                (float(np.array(tx.position)[0]), float(np.array(tx.position)[1]))
                for tx in transmitters
            ],
            "fixed_z": fixed_z,
            "initial_directions_xy": None,
        }

        focal_point = self.scene_config.get("focal_point")
        if self.reflector_controller is not None and focal_point is not None:
            if isinstance(focal_point, Sequence) and len(focal_point) == 3:
                task["reflector_u"] = float(0.5)
                task["reflector_v"] = float(0.5)
                task["reflector_target"] = (
                    float(focal_point[0]),
                    float(focal_point[1]),
                    float(focal_point[2]),
                )

        return task

    def _build_tx_positions(self, task: Mapping[str, Any]) -> List[List[float]]:
        """Convert task AP coordinates into 3-D transmitter positions."""
        raw_positions = task.get("initial_positions")
        if not isinstance(raw_positions, Sequence) or len(raw_positions) == 0:
            raise ValueError("task must include non-empty 'initial_positions'")

        fixed_z = float(task.get("fixed_z", 0.0))
        tx_positions: List[List[float]] = []
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

        controller.u = torch.tensor(float(task["reflector_u"]), dtype=torch.float32, device=self.device)
        controller.v = torch.tensor(float(task["reflector_v"]), dtype=torch.float32, device=self.device)
        controller.set_tx_position(np.asarray(tx_positions[0], dtype=np.float32))
        controller.set_focal_point(
            torch.tensor(reflector_target, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        controller.orient_to_target()
        controller.apply_to_scene()

    def _build_spatial_weights(self) -> torch.Tensor:
        """Build one static normalized demand map for this worker's scene."""
        task = self._build_default_task()
        tx_positions = self._build_tx_positions(task)
        self._configure_transmitters(tx_positions, task.get("initial_directions_xy"))
        self._configure_reflector(tx_positions, task)

        solver = RadioMapSolver()
        radio_map = solver(
            self.scene,
            cell_size=(1.0, 1.0),
            samples_per_tx=100_000,
            max_depth=3,
            refraction=True,
            diffraction=True,
        )

        # Lock a single map geometry for all subsequent evaluations in this
        # worker so demand weights and coverage maps always share HxW shape.
        self.radio_map_geometry = {
            "center": np.array(radio_map.center).tolist(),
            "size": np.array(radio_map.size).tolist(),
            "orientation": np.array(radio_map.orientation).tolist(),
        }

        coverage_map = torch.from_numpy(np.array(radio_map.rss)).to(self.device)
        flat_batches = coverage_map.reshape(-1, coverage_map.shape[-2], coverage_map.shape[-1])
        valid_mask = (flat_batches > POWER_EPSILON).any(dim=0)
        
        if not torch.any(valid_mask):
            valid_mask = torch.ones_like(valid_mask, dtype=torch.bool, device=self.device)

        num_rows, num_cols = int(valid_mask.shape[0]), int(valid_mask.shape[1])

        return generate_spatial_weight_map(
            num_rows=num_rows,
            num_cols=num_cols,
            demand_config=self.demand_config,
            valid_mask=valid_mask,
        )

    def _has_explicit_weighted_reporting_config(self) -> bool:
        """Return True when weighted reporting was explicitly requested."""
        if "_report_weighted_stats" in self.demand_config:
            return bool(self.demand_config.get("_report_weighted_stats", False))

        # Backward-compatible fallback for older configs that set these keys.
        bounding_boxes = self.demand_config.get("bounding_boxes")
        box_weights = self.demand_config.get("box_weights")
        return (
            isinstance(bounding_boxes, Sequence)
            and isinstance(box_weights, Sequence)
            and len(bounding_boxes) > 0
            and len(box_weights) > 0
        )

    def optimize(
        self,
        task_id: int,
        optimizer_method: str,
        optimizer_kwargs: Dict[str, Any],
        optimization_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run one optimizer task and return raw outputs only."""
        self._run_count += 1

        if optimizer_method == "memetic_eval":
            return self._run_memetic_eval(task_id, optimizer_kwargs)

        if optimizer_method == "memetic_gd":
            return self._run_memetic_gd(
                task_id=task_id,
                optimizer_kwargs=optimizer_kwargs,
                optimization_params=optimization_params,
            )

        optimizer_kwargs_local = self._prepare_optimizer_kwargs(optimizer_kwargs)
        optimizer = self._create_optimizer(optimizer_method, optimizer_kwargs_local)

        start_time = time.time()
        optimizer_result = optimizer.optimize(**optimization_params)
        elapsed_time = time.time() - start_time
        return self._build_raw_output(
            task_id=task_id,
            optimizer_method=optimizer_method,
            optimizer_kwargs=optimizer_kwargs,
            optimization_params=optimization_params,
            optimizer=optimizer,
            optimizer_result=optimizer_result,
            elapsed_time=elapsed_time,
        )

    def _run_memetic_eval(
        self,
        task_id: int,
        task_kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate one static memetic GA task on the shared scene."""
        task_payload = dict(task_kwargs)
        loss_kwargs = task_payload.pop("loss_kwargs", {})
        if not isinstance(loss_kwargs, Mapping):
            raise ValueError("'loss_kwargs' must be a mapping for memetic_eval tasks.")

        evaluator = StaticConfigurationEvaluator(
            scene=self.scene,
            reflector_controller=self.reflector_controller,
            loss_kwargs=loss_kwargs,
            spatial_weights=self.spatial_weights,
            radio_map_geometry=self.radio_map_geometry,
            include_weighted_reporting=self._has_explicit_weighted_reporting_config(),
        )

        start_time = time.time()
        evaluation = evaluator.evaluate(task=task_payload)
        elapsed_time = time.time() - start_time

        primary_fitness = float(evaluation.get("primary_fitness", float("-inf")))
        loss_components = evaluation.get("loss_components", {})
        physical_metrics = evaluation.get("physical_metrics", {})

        return {
            "task_id": int(task_id),
            "worker_id": int(self.worker_id),
            "optimizer_method": "memetic_eval",
            "time_elapsed": float(elapsed_time),
            "optimizer_kwargs": _to_serializable(task_kwargs),
            "optimization_params": {},
            "primary_fitness": primary_fitness,
            "loss_components": _to_serializable(loss_components),
            "physical_metrics": _to_serializable(physical_metrics),
            "raw_output": True,
        }

    def _run_memetic_gd(
        self,
        task_id: int,
        optimizer_kwargs: Dict[str, Any],
        optimization_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run one memetic gradient-descent task and return raw outputs."""
        optimizer_kwargs_local = dict(optimizer_kwargs)
        if (
            self.reflector_controller is not None
            and "reflector_controller" not in optimizer_kwargs_local
        ):
            optimizer_kwargs_local["reflector_controller"] = self.reflector_controller

        optimizer = MemeticGradientDescentOptimizer(
            scene=self.scene,
            spatial_weights=self.spatial_weights,
            radio_map_geometry=self.radio_map_geometry,
            include_weighted_reporting=self._has_explicit_weighted_reporting_config(),
            **optimizer_kwargs_local,
        )

        start_time = time.time()
        optimizer_result = optimizer.optimize(**optimization_params)
        elapsed_time = time.time() - start_time
        return self._build_raw_output(
            task_id=task_id,
            optimizer_method="memetic_gd",
            optimizer_kwargs=optimizer_kwargs,
            optimization_params=optimization_params,
            optimizer=optimizer,
            optimizer_result=optimizer_result,
            elapsed_time=elapsed_time,
        )

    def _prepare_optimizer_kwargs(self, optimizer_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare per-task optimizer kwargs for local worker execution."""
        optimizer_kwargs_local = dict(optimizer_kwargs)

        # Reuse per-worker reflector controller instead of serializing it in tasks.
        if self.reflector_controller is not None and "reflector_controller" not in optimizer_kwargs_local:
            optimizer_kwargs_local["reflector_controller"] = self.reflector_controller

        # Build percentile objective in-worker from scalar quantile payload.
        if (
            "percentile_target_quantile" in optimizer_kwargs_local
            and "percentile_objective" not in optimizer_kwargs_local
        ):
            from reflector_position.metrics import PercentileCoverageObjective

            target_quantile = float(optimizer_kwargs_local.pop("percentile_target_quantile"))
            optimizer_kwargs_local["percentile_objective"] = PercentileCoverageObjective(
                target_quantile=target_quantile,
                mode="maximize",
            )

        return optimizer_kwargs_local

    def _create_optimizer(self, optimizer_method: str, optimizer_kwargs: Dict[str, Any]) -> Any:
        """Instantiate optimizer for one task."""
        return OptimizerFactory.create(
            method=optimizer_method,
            scene=self.scene,
            **optimizer_kwargs,
        )

    def _build_raw_output(
        self,
        task_id: int,
        optimizer_method: str,
        optimizer_kwargs: Dict[str, Any],
        optimization_params: Dict[str, Any],
        optimizer: Any,
        optimizer_result: Any,
        elapsed_time: float,
    ) -> Dict[str, Any]:
        """Build final worker payload with raw optimizer artifacts."""
        output: Dict[str, Any] = {
            "task_id": int(task_id),
            "worker_id": int(self.worker_id),
            "optimizer_method": str(optimizer_method),
            "time_elapsed": float(elapsed_time),
            "optimizer_kwargs": _to_serializable(optimizer_kwargs),
            "optimization_params": _to_serializable(optimization_params),
            "optimizer_result": _to_serializable(optimizer_result),
            "num_aps": int(getattr(optimizer, "num_aps", 1)),
        }

        if hasattr(optimizer, "history"):
            output["history"] = _to_serializable(getattr(optimizer, "history"))

        if hasattr(optimizer, "results"):
            output["results"] = _to_serializable(getattr(optimizer, "results"))

        if hasattr(optimizer, "_snapshot_reflector") and callable(optimizer._snapshot_reflector):
            try:
                output["reflector_snapshot"] = _to_serializable(optimizer._snapshot_reflector())
            except Exception:
                output["reflector_snapshot"] = None

        return output

    def get_worker_id(self) -> int:
        """Return worker id."""
        return self.worker_id

    def get_run_count(self) -> int:
        """Return number of processed tasks."""
        return self._run_count


class RawRayParallelOptimizer:
    """ActorPool orchestrator that preserves raw worker outputs."""

    def __init__(
        self,
        num_workers: int = 4,
        gpu_fraction: float = 0.25,
        demand_config: Optional[Mapping[str, Any]] = None,
    ):
        self.num_workers = int(num_workers)
        self.gpu_fraction = float(gpu_fraction)
        self.demand_config = dict(demand_config or {})

        self._workers: List[Any] = []
        self._pool: Optional[ActorPool] = None
        self._scene_config: Optional[Dict[str, Any]] = None
        self._demand_config: Dict[str, Any] = dict(self.demand_config)

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def _kill_workers(self) -> None:
        """Terminate all currently tracked worker actors."""
        for worker in self._workers:
            try:
                ray.kill(worker)
            except Exception:
                continue
        self._workers = []
        self._pool = None

    def shutdown(self) -> None:
        """Destroy actor pool resources owned by this object."""
        self._kill_workers()

    def _ensure_pool(self, scene_config: Dict[str, Any], verbose: bool = False) -> None:
        """Create or reuse actor pool for the given scene config."""
        if (
            self._pool is not None
            and self._scene_config == scene_config
            and self._demand_config == self.demand_config
        ):
            return

        if self._workers:
            self._kill_workers()

        if verbose:
            print(f"[raw-ray] Spawning {self.num_workers} workers...")

        actor_options: Dict[str, Any] = {"num_cpus": 1}
        if self.gpu_fraction > 0:
            actor_options["num_gpus"] = self.gpu_fraction

        self._workers = [
            RawOptimizationWorker.options(**actor_options).remote(
                worker_id=i,
                scene_config=scene_config,
                demand_config=self.demand_config,
            )
            for i in range(self.num_workers)
        ]
        self._pool = ActorPool(self._workers)
        self._scene_config = dict(scene_config)
        self._demand_config = dict(self.demand_config)

    def _build_task_configs(
        self,
        optimizer_method: str,
        work_items: List[Dict[str, Any]],
        optimization_params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build per-task payloads consumed by actor pool."""
        return [
            {
                "task_id": i,
                "optimizer_method": optimizer_method,
                "optimizer_kwargs": kwargs,
                "optimization_params": optimization_params,
            }
            for i, kwargs in enumerate(work_items)
        ]

    def _collect_results(
        self,
        task_configs: List[Dict[str, Any]],
        num_tasks: int,
        verbose: bool,
    ) -> List[Dict[str, Any]]:
        """Execute tasks via ActorPool and collect unordered results."""
        assert self._pool is not None
        results_iter = self._pool.map_unordered(
            lambda actor, cfg: actor.optimize.remote(
                cfg["task_id"],
                cfg["optimizer_method"],
                cfg["optimizer_kwargs"],
                cfg["optimization_params"],
            ),
            task_configs,
        )

        all_results: List[Dict[str, Any]] = []
        progress_every = max(1, num_tasks // 20)
        for completed, result in enumerate(results_iter, start=1):
            all_results.append(result)
            if verbose and (completed % progress_every == 0 or completed == num_tasks):
                print(f"[raw-ray] Progress: {completed}/{num_tasks}")

        return all_results

    def _build_aggregate_stats(
        self,
        all_results: List[Dict[str, Any]],
        num_tasks: int,
        total_time: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute run-level timing and worker-utilization aggregates."""
        worker_task_counts: Dict[int, int] = {}
        for result in all_results:
            wid = int(result.get("worker_id", -1))
            worker_task_counts[wid] = worker_task_counts.get(wid, 0) + 1

        task_times = [float(r.get("time_elapsed", 0.0)) for r in all_results]
        aggregate_stats = {
            "num_tasks": int(num_tasks),
            "mean_time_per_task": float(np.mean(task_times)) if task_times else 0.0,
            "total_sequential_time": float(np.sum(task_times)) if task_times else 0.0,
            "total_wall_clock_time": float(total_time),
            "speedup": float(np.sum(task_times) / total_time) if total_time > 0 and task_times else 0.0,
        }
        pool_info = {
            "num_workers": self.num_workers,
            "num_tasks": int(num_tasks),
            "tasks_per_worker": worker_task_counts,
        }
        return aggregate_stats, pool_info

    def run(
        self,
        scene_config: Dict[str, Any],
        optimizer_method: str,
        work_items: List[Dict[str, Any]],
        optimization_params: Dict[str, Any],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Execute tasks in parallel and return raw task results."""
        num_tasks = len(work_items)
        if num_tasks == 0:
            raise ValueError("work_items cannot be empty")

        self._ensure_pool(scene_config, verbose=verbose)

        start_time = time.time()
        task_configs = self._build_task_configs(
            optimizer_method=optimizer_method,
            work_items=work_items,
            optimization_params=optimization_params,
        )
        all_results = self._collect_results(
            task_configs=task_configs,
            num_tasks=num_tasks,
            verbose=verbose,
        )
        total_time = float(time.time() - start_time)
        aggregate_stats, pool_info = self._build_aggregate_stats(
            all_results=all_results,
            num_tasks=num_tasks,
            total_time=total_time,
        )

        return {
            "all_results": all_results,
            "total_time": total_time,
            "aggregate_stats": aggregate_stats,
            "pool_info": pool_info,
            "optimizer_method": optimizer_method,
            "raw_output": True,
        }


class RawRayActorPoolExecutor:
    """Ordered ActorPool executor for memetic GA evaluations with raw outputs."""

    def __init__(
        self,
        scene_config: Dict[str, Any],
        demand_config: Optional[Mapping[str, Any]] = None,
        num_workers: int = 4,
        gpu_fraction: float = 0.25,
        verbose: bool = True,
    ):
        self.num_workers = int(num_workers)
        self.gpu_fraction = float(gpu_fraction)
        self.scene_config = dict(scene_config)
        self.demand_config = dict(demand_config or {})

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        actor_options: Dict[str, Any] = {"num_cpus": 1}
        if self.gpu_fraction > 0:
            actor_options["num_gpus"] = self.gpu_fraction

        if verbose:
            print(
                f"  Spawning {self.num_workers} raw Ray workers "
                f"(GPU fraction={self.gpu_fraction}) ..."
            )

        self._workers = [
            RawOptimizationWorker.options(**actor_options).remote(
                worker_id=i,
                scene_config=self.scene_config,
                demand_config=self.demand_config,
            )
            for i in range(self.num_workers)
        ]
        self._pool = ActorPool(self._workers)

        if verbose:
            print(f"  Raw ActorPool ready: {self.num_workers} workers")

    def map(self, func: Any, iterable: Any) -> List[Dict[str, Any]]:
        """Apply func to iterable and evaluate tasks in input order."""
        items = list(iterable)
        if not items:
            return []

        task_args = [func(item) for item in items]
        return list(
            self._pool.map(
                lambda actor, args: actor.optimize.remote(*args),
                task_args,
            )
        )

    def shutdown(self) -> None:
        """Kill all worker actors and release ActorPool resources."""
        for worker in self._workers:
            try:
                ray.kill(worker)
            except Exception:
                continue
        self._workers = []
        self._pool = None
        print("  Raw ActorPool shut down.")
