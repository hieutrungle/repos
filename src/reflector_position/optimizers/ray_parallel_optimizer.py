"""
Ray-based parallel optimizer using ActorPool for efficient resource reuse.

This module implements distributed parallel optimization using Ray's ActorPool
pattern, enabling many independent optimization trajectories (e.g., 32) to be
processed by a fixed pool of reusable workers (e.g., 6). Each worker loads the
heavy Scene object once and reuses it for multiple optimization tasks.

Key Design Principles:
1. Actor reuse: Workers load the Scene once, then accept many tasks
2. Automatic work distribution: ActorPool queues tasks and assigns to idle actors
3. GPU efficiency: Configurable GPU fraction per worker (0.25 = 4 workers/GPU)
4. Decoupled task count: Number of tasks can far exceed number of workers

Architecture (ActorPool pattern):
- Orchestrator (Driver): Manages pool, submits tasks, aggregates results
- Actor Pool (Workers): Fixed set of reusable actors, each with persistent Scene
- Work Queue: Tasks are automatically queued and assigned to next available actor
- Reduction Phase: Winner selection from all task results

Usage pattern:
    1. Create orchestrator with num_workers (pool size) and gpu_fraction
    2. Submit N work items (N can be >> num_workers)
    3. ActorPool auto-distributes: workers process tasks as they become available
    4. Collect and aggregate all results
"""

import os
import time
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import ray
from ray.util.actor_pool import ActorPool
# Create optimizer for this specific task (fresh each time)
from reflector_position.optimizers.optimizer_factory import OptimizerFactory
from reflector_position.metrics import POWER_EPSILON


def _rss_watts_to_dbm(rss_watt: float) -> float:
    """Convert RSS from Watts (linear) to dBm. Mirrors metrics.rss_to_dbm."""
    return 10.0 * np.log10(max(rss_watt, POWER_EPSILON)) + 30.0


def _fmt_dir(direction) -> str:
    """Format a direction vector for display. Returns 'N/A' when missing."""
    if direction is None:
        return "N/A"
    # Multi-AP: nested list [[dx0,dy0,dz0],[dx1,dy1,dz1]]
    if isinstance(direction, (list, np.ndarray)) and len(direction) > 0:
        if isinstance(direction[0], (list, np.ndarray)):
            parts = [f"({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f})" for d in direction]
            return " | ".join(parts)
    return f"({direction[0]:+.4f}, {direction[1]:+.4f}, {direction[2]:+.4f})"


def _fmt_pos(position) -> str:
    """Format a position for display. Handles multi-AP nested lists."""
    if position is None:
        return "N/A"
    if isinstance(position, (list, np.ndarray)) and len(position) > 0:
        if isinstance(position[0], (list, np.ndarray)):
            parts = [f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" for p in position]
            return " | ".join(parts)
    return f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"


# Per-AP colours and markers for trajectory plots
_AP_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
_AP_MARKERS = ["o", "s", "^", "D", "v", "P"]


@ray.remote
class OptimizationWorker:
    """
    Reusable Ray actor that runs independent optimization tasks.

    Each worker:
    1. Loads its own Scene instance ONCE via setup_building_floor_scene
    2. Accepts multiple optimization tasks via optimize()
    3. Creates a fresh optimizer for each task (different starting positions/bounds)
    4. Returns serializable results to the orchestrator

    The Scene is heavy (XML parsing, BVH building) and persists in memory.
    The optimizer is lightweight and recreated for each task.
    """

    def __init__(
        self,
        worker_id: int,
        scene_config: Dict[str, Any],
    ):
        """
        Initialize worker — loads Scene only. No optimizer is created here.

        Args:
            worker_id: Unique identifier for this worker (actor).
            scene_config: Configuration for setup_building_floor_scene():
                - scene_path (str): Path to the scene XML file (required)
                - frequency (float): Operating frequency in Hz (default: 5.18e9)
                - tx_positions (list): Transmitter positions [(x,y,z), ...]
                - tx_power_dbm (float): Transmitter power in dBm (default: 5.0)
                - rx_position (tuple): Receiver position (x,y,z)
        """
        self.worker_id = worker_id
        self._run_count = 0

        # Load Scene instance (CRITICAL: each worker gets its own Scene)
        # and optional reflector controller.
        self.scene, self.reflector_controller = self._load_scene(scene_config)

    def _load_scene(self, scene_config: Dict[str, Any]) -> Tuple[Any, Optional[Any]]:
        """
        Load a fully configured Scene instance using setup_building_floor_scene.

        This mirrors the pattern from full_comparison.py and
        optimizer_factory_example.py, ensuring Tx/Rx arrays, transmitters,
        and receivers are properly configured.

        Args:
            scene_config: Dict with scene_path (required) and optional params.

        Returns:
            Fully configured sionna.rt.Scene instance.
        """
        from reflector_position.scene_setup import setup_building_floor_scene

        loaded = setup_building_floor_scene(
            scene_path=str(scene_config["scene_path"]),
            frequency=scene_config.get("frequency", 5.18e9),
            tx_positions=scene_config.get("tx_positions", None),
            tx_power_dbm=scene_config.get("tx_power_dbm", 5.0),
            rx_position=scene_config.get("rx_position", (16.0, 6.5, 1.5)),
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

    def optimize(
        self,
        task_id: int,
        optimizer_method: str,
        optimizer_kwargs: Dict[str, Any],
        optimization_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create an optimizer on-the-fly and run one optimization task.

        The Scene persists from __init__; the optimizer is freshly created
        each time so the worker can handle different starting positions,
        search bounds, etc.

        Args:
            task_id: Identifier for this specific task.
            optimizer_method: 'gradient_descent', 'grid_search', or 'grid_search_point'.
            optimizer_kwargs: Keyword arguments for OptimizerFactory.create().
                For gradient_descent: initial_position, position_bounds, fixed_z
                For grid_search: search_bounds, grid_resolution, fixed_z
                For grid_search_point: evaluation_position, fixed_z
            optimization_params: Keyword arguments for optimizer.optimize().
                For gradient_descent: num_iterations, learning_rate, samples_per_tx,
                    max_depth, use_soft_min, temperature, verbose
                For grid_search: samples_per_tx, max_depth,
                    coverage_threshold_dbm, verbose

        Returns:
            Dictionary with:
                - task_id: Task identifier
                - worker_id: Physical actor that processed this task
                - best_position: Optimized position [x, y, z] as list
                - best_metric: Best 5th-percentile RSS value (linear Watts)
                - best_metric_dbm: Best 5th-percentile RSS value in dBm
                - time_elapsed: Optimization time in seconds
                - history: Optimization history (gradient descent)
                - grid_results: Grid search results (grid search)
        """
        self._run_count += 1

        optimizer_kwargs_local = dict(optimizer_kwargs)

        # Attach per-worker reflector controller (do not serialize controller
        # through every task payload).
        if self.reflector_controller is not None and "reflector_controller" not in optimizer_kwargs_local:
            optimizer_kwargs_local["reflector_controller"] = self.reflector_controller

        # Construct percentile objective inside the worker from a scalar
        # quantile to avoid shipping torch module objects per task.
        if (
            "percentile_target_quantile" in optimizer_kwargs_local
            and "percentile_objective" not in optimizer_kwargs_local
        ):
            from reflector_position.metrics import PercentileCoverageObjective

            optimizer_kwargs_local["percentile_objective"] = PercentileCoverageObjective(
                target_quantile=float(optimizer_kwargs_local.pop("percentile_target_quantile")),
                mode="maximize",
            )

        optimizer = OptimizerFactory.create(
            method=optimizer_method,
            scene=self.scene,
            **optimizer_kwargs_local,
        )

        # Detect multi-AP configuration
        num_aps = getattr(optimizer, "num_aps", 1)

        # Run optimization (same call pattern as full_comparison.py)
        start_time = time.time()
        result = optimizer.optimize(**optimization_params)
        elapsed_time = time.time() - start_time

        # Unpack result tuple.
        # GD returns (position, metric).
        # SinglePointGridSearch returns (position, orientation, metric).
        result_orientation = None
        if isinstance(result, tuple) and len(result) == 3:
            final_position, result_orientation, final_metric = result
        elif isinstance(result, tuple) and len(result) == 2:
            final_position, final_metric = result
        else:
            final_position = None
            final_metric = float("-inf")

        # For gradient descent: find the BEST position across all iterations,
        # not just the final one, since the goal is to find the best
        # possible 5th-percentile RSS value.
        best_position = final_position
        best_metric = final_metric
        best_iteration = -1  # -1 means "final" (no history or non-GD)

        if hasattr(optimizer, "history") and "min_rss_values" in optimizer.history:
            rss_values = optimizer.history["min_rss_values"]
            positions = optimizer.history.get("positions", [])
            if rss_values and positions and len(rss_values) == len(positions):
                best_iter_idx = int(np.argmax(rss_values))
                best_metric = float(rss_values[best_iter_idx])
                best_position = positions[best_iter_idx]
                best_iteration = best_iter_idx

        # Extract orientation (direction + look_at) from history if available
        best_direction = None
        final_direction = None
        best_look_at = None
        final_look_at = None

        # Case 1: Orientation returned directly from optimizer (grid_search_point 8-dir sweep)
        if result_orientation is not None:
            best_direction = np.asarray(result_orientation).tolist()
            final_direction = best_direction  # same for single-point eval
            if best_position is not None:
                pos_arr = np.asarray(best_position)
                dir_arr = np.asarray(result_orientation)
                best_look_at = (pos_arr + dir_arr).tolist()
                final_look_at = best_look_at

        # Case 2: Orientation stored in GD history
        elif hasattr(optimizer, "history"):
            directions = optimizer.history.get("directions", [])
            look_at_targets = optimizer.history.get("look_at_targets", [])
            if directions:
                final_direction = np.asarray(directions[-1]).tolist()
                if 0 <= best_iteration < len(directions):
                    best_direction = np.asarray(directions[best_iteration]).tolist()
                else:
                    best_direction = final_direction
            if look_at_targets:
                final_look_at = np.asarray(look_at_targets[-1]).tolist()
                if 0 <= best_iteration < len(look_at_targets):
                    best_look_at = np.asarray(look_at_targets[best_iteration]).tolist()
                else:
                    best_look_at = final_look_at

        # Convert to serializable types
        if best_position is not None:
            best_position = np.asarray(best_position).tolist()
        else:
            best_position = [0.0, 0.0, 0.0]

        # Also keep final position for reference
        if final_position is not None:
            final_position = np.asarray(final_position).tolist()
        else:
            final_position = [0.0, 0.0, 0.0]

        metric_linear = float(best_metric) if best_metric is not None else 0.0
        metric_dbm = _rss_watts_to_dbm(metric_linear)

        output = {
            "task_id": task_id,
            "worker_id": self.worker_id,
            "num_aps": num_aps,
            "best_position": best_position,
            "best_metric": metric_linear,
            "best_metric_dbm": metric_dbm,
            "best_iteration": best_iteration,
            "final_position": final_position,
            "best_direction": best_direction,
            "final_direction": final_direction,
            "best_look_at": best_look_at,
            "final_look_at": final_look_at,
            "time_elapsed": elapsed_time,
        }

        # Reflector state: for grid search the reflector (u, v, target) is
        # specified per work item; for GD the optimizer *learns* them.
        # Prefer the optimizer's snapshot when available.
        if hasattr(optimizer, "_snapshot_reflector") and callable(optimizer._snapshot_reflector):
            refl_snap = optimizer._snapshot_reflector()
            output["reflector_u"] = refl_snap.get("u")
            output["reflector_v"] = refl_snap.get("v")
            output["reflector_focal_point"] = refl_snap.get("focal_point")
            output["reflector_position"] = refl_snap.get("position")
            # Keep backward-compat key used by grid search callers
            output["reflector_target"] = refl_snap.get("focal_point")
        else:
            output["reflector_u"] = optimizer_kwargs_local.get("reflector_u")
            output["reflector_v"] = optimizer_kwargs_local.get("reflector_v")
            output["reflector_target"] = optimizer_kwargs_local.get("reflector_target")

        # Include history if optimizer tracks it (gradient descent has history)
        if hasattr(optimizer, "history"):
            history = {}
            for k, v in optimizer.history.items():
                if isinstance(v, list):
                    serialized = []
                    for item in v:
                        if hasattr(item, "tolist"):
                            serialized.append(item.tolist())
                        elif hasattr(item, "item"):
                            serialized.append(item.item())
                        else:
                            serialized.append(item)
                    history[k] = serialized
                else:
                    history[k] = v
            output["history"] = history

        # Include grid search results if available
        if hasattr(optimizer, "results"):
            gs_results = {}
            for k, v in optimizer.results.items():
                if k == "radio_maps":
                    continue  # Skip non-serializable radio maps
                if isinstance(v, list):
                    serialized = []
                    for item in v:
                        if hasattr(item, "tolist"):
                            serialized.append(item.tolist())
                        elif hasattr(item, "item"):
                            serialized.append(item.item())
                        else:
                            serialized.append(item)
                    gs_results[k] = serialized
                else:
                    gs_results[k] = v
            output["grid_results"] = gs_results

            # Promote key reflector metadata to top-level result for easier
            # plotting/summary consumption.
            for key in (
                "reflector_position",
                "reflector_u",
                "reflector_v",
                "reflector_target",
                "reflector_focal_point",
            ):
                if key in gs_results and (key not in output or output.get(key) is None):
                    output[key] = gs_results[key]

        return output

    def get_worker_id(self) -> int:
        """Return the worker ID."""
        return self.worker_id

    def get_run_count(self) -> int:
        """Return how many tasks this worker has processed."""
        return self._run_count


class RayParallelOptimizer:
    """
    Orchestrator for distributed parallel optimization using Ray ActorPool.

    Implements the "Pool -> Map -> Reduce" workflow:
    1. Pool: Spawn a fixed number of persistent actors, each loading Scene once
    2. Map: Submit all work items to ActorPool for automatic distribution
    3. Reduce: Aggregate results and select best configuration

    The pool is created lazily on the first run() call and reused for
    subsequent calls with the same scene_config. This avoids reloading the
    heavy Scene object when running multiple experiments.

    Key difference from the old architecture:
    - OLD: num_workers == num_tasks (create/destroy actor per task)
    - NEW: num_workers = pool size (fixed), num_tasks can be much larger
           ActorPool handles the queuing and work distribution automatically

    Example:
        >>> import ray
        >>> from pathlib import Path
        >>> from reflector_position.optimizers import (
        ...     RayParallelOptimizer, generate_random_initial_positions
        ... )
        >>>
        >>> ray.init(ignore_reinit_error=True)
        >>>
        >>> scene_config = {
        ...     "scene_path": str(Path.home() / "blender/models/building_floor/building_floor.xml"),
        ...     "frequency": 5.18e9,
        ...     "tx_power_dbm": 5.0,
        ... }
        >>>
        >>> parallel_opt = RayParallelOptimizer(num_workers=6, gpu_fraction=0.25)
        >>>
        >>> bounds = {"x_min": 5.0, "x_max": 25.0, "y_min": 5.0, "y_max": 25.0}
        >>> positions = generate_random_initial_positions(32, bounds, seed=42)
        >>>
        >>> # 32 tasks processed by 6 reusable workers
        >>> work_items = [
        ...     {"initial_position": (pos[0], pos[1]),
        ...      "position_bounds": bounds}
        ...     for pos in positions
        ... ]
        >>>
        >>> results = parallel_opt.run(
        ...     scene_config=scene_config,
        ...     optimizer_method="gradient_descent",
        ...     work_items=work_items,
        ...     optimization_params={
        ...         "num_iterations": 30,
        ...         "learning_rate": 0.5,
        ...         "samples_per_tx": 1_000_000,
        ...         "max_depth": 13,
        ...         "verbose": False,
        ...     },
        ... )
        >>>
        >>> print(f"Best: Task #{results['best_task_id']}")
        >>> parallel_opt.shutdown()
    """

    def __init__(
        self,
        num_workers: int = 4,
        gpu_fraction: float = 0.25,
    ):
        """
        Initialize Ray parallel optimizer (the orchestrator).

        Args:
            num_workers: Number of persistent actors in the pool.
                Each actor loads the Scene once and reuses it.
                E.g., 4 workers can process 32+ tasks via queuing.
            gpu_fraction: GPU fraction per worker. E.g., 0.25 = 4 workers per GPU,
                0.5 = 2 workers per GPU, 0.0 = CPU-only.
        """
        self.num_workers = num_workers
        self.gpu_fraction = gpu_fraction

        # Pool state (created lazily on first run)
        self._workers: List = []
        self._pool: Optional[ActorPool] = None
        self._scene_config: Optional[Dict[str, Any]] = None

        # Initialize Ray if not already running
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def _ensure_pool(
        self,
        scene_config: Dict[str, Any],
        verbose: bool = False,
    ) -> None:
        """
        Create or reuse the actor pool.

        If a pool already exists with the same scene_config, it is reused
        (avoiding expensive Scene reloading). Otherwise, old actors are killed
        and a new pool is created.

        Args:
            scene_config: Scene configuration dict.
            verbose: Print pool creation info.
        """
        # Reuse existing pool if scene config hasn't changed
        if self._pool is not None and self._scene_config == scene_config:
            if verbose:
                print(f"  Reusing existing pool of {self.num_workers} workers")
            return

        # Shut down old pool if exists
        if self._workers:
            self._kill_workers()

        # Spawn new workers
        if verbose:
            print(f"  Spawning {self.num_workers} persistent workers...")

        actor_options = {"num_cpus": 1}
        if self.gpu_fraction > 0:
            actor_options["num_gpus"] = self.gpu_fraction

        self._workers = [
            OptimizationWorker.options(**actor_options).remote(
                worker_id=i,
                scene_config=scene_config,
            )
            for i in range(self.num_workers)
        ]
        self._pool = ActorPool(self._workers)
        self._scene_config = scene_config

        if verbose:
            print(f"  Pool ready: {self.num_workers} workers (Scene loaded once per worker)")

    def run(
        self,
        scene_config: Dict[str, Any],
        optimizer_method: str,
        work_items: List[Dict[str, Any]],
        optimization_params: Dict[str, Any],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run parallel optimization across the actor pool.

        This implements the full "Pool -> Map -> Reduce" workflow:
        1. Pool: Create/reuse persistent actors (each with its own Scene)
        2. Map: Submit all work items to ActorPool (automatic queuing)
        3. Reduce: Aggregate results and select winner

        The number of work_items can exceed num_workers - ActorPool
        automatically queues tasks and assigns them to the next idle actor.

        Args:
            scene_config: Scene configuration dict passed to each worker's
                setup_building_floor_scene(). Must contain 'scene_path'.
                Optional: 'frequency', 'tx_power_dbm', 'tx_positions', 'rx_position'.
            optimizer_method: 'gradient_descent' or 'grid_search'.
            work_items: List of optimizer kwargs dicts, one per task.
                Can be any length (decoupled from num_workers).
                For gradient_descent: [{'initial_position': (x,y), 'position_bounds': {...}}, ...]
                For grid_search: [{'search_bounds': {...}, 'grid_resolution': 2.0}, ...]
            optimization_params: Dict of keyword arguments for optimizer.optimize().
                Shared across all tasks.
                For gradient_descent: {'num_iterations': 30, 'learning_rate': 0.5, ...}
                For grid_search: {'samples_per_tx': 1_000_000, 'max_depth': 13, ...}
            verbose: Print progress information.

        Returns:
            Dictionary with:
                - all_results: List of result dicts from all tasks
                - best_result: Result dict from best task
                - best_task_id: ID of the task with the best metric
                - total_time: Total wall-clock time
                - aggregate_stats: Statistics across all tasks
                - pool_info: Worker utilization info
        """
        num_tasks = len(work_items)

        if num_tasks == 0:
            raise ValueError("work_items must contain at least one task")

        start_time = time.time()

        if verbose:
            print("=" * 80)
            print("RAY PARALLEL OPTIMIZATION (ActorPool)")
            print("=" * 80)
            print(f"  Pool size (workers):  {self.num_workers}")
            print(f"  Total tasks:          {num_tasks}")
            print(f"  GPU fraction/worker:  {self.gpu_fraction}")
            print(f"  Optimizer method:     {optimizer_method}")
            if self.gpu_fraction > 0:
                print(f"  Workers per GPU:      {int(1.0 / self.gpu_fraction)}")
            else:
                print("  Mode: CPU-only")
            if num_tasks > self.num_workers:
                print(f"  Queuing: {num_tasks} tasks across {self.num_workers} workers "
                      f"(~{num_tasks / self.num_workers:.1f} tasks/worker)")
            print("-" * 80)

        # -- Phase 1: POOL -- Create or reuse persistent actors ---------
        if verbose:
            print("Phase 1/3: Setting up actor pool...")

        self._ensure_pool(scene_config, verbose=verbose)

        if verbose:
            print("-" * 80)

        # -- Phase 2: MAP -- Submit all tasks via ActorPool -------------
        if verbose:
            print("Phase 2/3: Submitting tasks to pool...")
            print(f"  ({num_tasks} tasks -> {self.num_workers} workers, auto-queued)")

        # Prepare task configs: each item = (task_id, method, kwargs, params)
        task_configs = [
            {
                "task_id": i,
                "optimizer_method": optimizer_method,
                "optimizer_kwargs": kwargs,
                "optimization_params": optimization_params,
            }
            for i, kwargs in enumerate(work_items)
        ]

        # ActorPool.map_unordered distributes tasks to idle workers automatically.
        # If there are more tasks than workers, excess tasks are queued.
        all_results = []
        result_iter = self._pool.map_unordered(
            lambda actor, cfg: actor.optimize.remote(
                cfg["task_id"],
                cfg["optimizer_method"],
                cfg["optimizer_kwargs"],
                cfg["optimization_params"],
            ),
            task_configs,
        )

        progress_every = max(1, num_tasks // 20)  # ~5% updates
        for completed, result in enumerate(result_iter, start=1):
            all_results.append(result)
            if verbose and (completed == 1 or completed % progress_every == 0 or completed == num_tasks):
                print(f"  Progress: {completed}/{num_tasks} tasks completed")

        if verbose:
            print(f"  All {len(all_results)} tasks completed")
            print("-" * 80)

        # -- Phase 3: REDUCE -- Winner selection ------------------------
        if verbose:
            print("Phase 3/3: Aggregating results and selecting winner...")

        # Find best task (maximize metric - higher p5_rss is better)
        best_idx = int(np.argmax([r["best_metric"] for r in all_results]))
        best_result = all_results[best_idx]

        total_time = time.time() - start_time

        # Compute aggregate statistics (in both linear and dBm)
        metrics_linear = [r["best_metric"] for r in all_results]
        metrics_dbm = [r["best_metric_dbm"] for r in all_results]
        times = [r["time_elapsed"] for r in all_results]

        aggregate_stats = {
            # Linear (Watts)
            "mean_metric": float(np.mean(metrics_linear)),
            "std_metric": float(np.std(metrics_linear)),
            "min_metric": float(np.min(metrics_linear)),
            "max_metric": float(np.max(metrics_linear)),
            # dBm (for analysis / plotting)
            "mean_metric_dbm": float(np.mean(metrics_dbm)),
            "std_metric_dbm": float(np.std(metrics_dbm)),
            "min_metric_dbm": float(np.min(metrics_dbm)),
            "max_metric_dbm": float(np.max(metrics_dbm)),
            # Timing
            "mean_time_per_task": float(np.mean(times)),
            "total_sequential_time": float(np.sum(times)),
            "total_wall_clock_time": total_time,
            # Speedup: total sequential time / actual wall-clock time
            "speedup": float(np.sum(times) / total_time) if total_time > 0 else 0.0,
        }

        # Optional percentile statistics (if produced by grid search objective)
        percentile_dbm_values = []
        for r in all_results:
            g = r.get("grid_results", {})
            if "percentile_score_dbm" in g and g["percentile_score_dbm"] is not None:
                percentile_dbm_values.append(float(g["percentile_score_dbm"]))
        if percentile_dbm_values:
            aggregate_stats.update({
                "mean_percentile_dbm": float(np.mean(percentile_dbm_values)),
                "std_percentile_dbm": float(np.std(percentile_dbm_values)),
                "min_percentile_dbm": float(np.min(percentile_dbm_values)),
                "max_percentile_dbm": float(np.max(percentile_dbm_values)),
            })

        # Worker utilization: how many tasks each worker processed
        worker_task_counts = {}
        for r in all_results:
            wid = r["worker_id"]
            worker_task_counts[wid] = worker_task_counts.get(wid, 0) + 1

        pool_info = {
            "num_workers": self.num_workers,
            "num_tasks": num_tasks,
            "tasks_per_worker": worker_task_counts,
        }

        if verbose:
            print(f"  Best task: #{best_result['task_id']} "
                  f"(processed by Worker #{best_result['worker_id']})")
            _n_aps = best_result.get("num_aps", 1)
            if _n_aps > 1:
                print(f"  Num APs: {_n_aps}")
            print(f"  Position: {_fmt_pos(best_result['best_position'])}")
            print(f"  P5 RSS: {best_result['best_metric_dbm']:.2f} dBm")
            _bd = best_result.get("best_direction")
            if _bd:
                print(f"  Direction: {_fmt_dir(_bd)}")
            _bla = best_result.get("best_look_at")
            if _bla:
                print(f"  Look-at:  {_fmt_pos(_bla)}")
            print()
            print("  Aggregate Statistics:")
            print(
                f"    P5 RSS range: [{aggregate_stats['min_metric_dbm']:.2f}, "
                f"{aggregate_stats['max_metric_dbm']:.2f}] dBm"
            )
            print(
                f"    Mean P5 RSS: {aggregate_stats['mean_metric_dbm']:.2f} "
                f"+/- {aggregate_stats['std_metric_dbm']:.2f} dBm"
            )
            if "mean_percentile_dbm" in aggregate_stats:
                print(
                    f"    5th percentile range: [{aggregate_stats['min_percentile_dbm']:.2f}, "
                    f"{aggregate_stats['max_percentile_dbm']:.2f}] dBm"
                )
                print(
                    f"    Mean 5th percentile: {aggregate_stats['mean_percentile_dbm']:.2f} "
                    f"+/- {aggregate_stats['std_percentile_dbm']:.2f} dBm"
                )
            print()
            print("  Performance:")
            print(f"    Mean time per task: {aggregate_stats['mean_time_per_task']:.2f}s")
            print(f"    Total sequential time: {aggregate_stats['total_sequential_time']:.2f}s")
            print(f"    Wall-clock time: {total_time:.2f}s")
            print(f"    Speedup: {aggregate_stats['speedup']:.2f}x")
            print()
            print("  Worker Utilization:")
            for wid in sorted(worker_task_counts):
                print(f"    Worker #{wid}: {worker_task_counts[wid]} tasks")
            print("=" * 80)

        return {
            "all_results": all_results,
            "best_result": best_result,
            "best_task_id": best_result["task_id"],
            "total_time": total_time,
            "aggregate_stats": aggregate_stats,
            "pool_info": pool_info,
        }

    def save_task_trajectory_plots(
        self,
        results: Dict[str, Any],
        save_dir: str,
        filename_prefix: str = "task",
        position_bounds: Optional[Dict[str, float]] = None,
        rss_range_dbm: Optional[tuple] = None,
    ) -> List[str]:
        """
        Save per-task trajectory plots for gradient descent tasks.

        Creates a 2x2 figure for each task that has history data:
        1. Position trajectory (start, end, best marked)
        2. P5 RSS over iterations (best iteration highlighted)
        3. Coverage over iterations
        4. Gradient norm (log scale)

        Only applicable to gradient descent tasks (which have history).

        Args:
            results: Results dictionary from run().
            save_dir: Directory to save plots.
            filename_prefix: Prefix for plot filenames.
                Files are named: {prefix}_{task_id}_trajectory.png
            position_bounds: Optional dict with 'x_min', 'x_max', 'y_min', 'y_max'
                to set consistent axis limits on position plots across all tasks.
            rss_range_dbm: Optional tuple (min_dbm, max_dbm) to set consistent
                y-axis limits on P5 RSS plots across all tasks and methods.

        Returns:
            List of saved file paths.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []

        for task_result in results["all_results"]:
            history = task_result.get("history")
            if not history or "positions" not in history:
                continue  # Skip non-GD tasks

            task_id = task_result["task_id"]
            num_aps_task = task_result.get("num_aps", 1)
            positions = np.array(history["positions"])
            min_rss_dbm_values = history.get("min_rss_dbm_values", [])
            coverage_values = history.get("coverage_values", [])
            gradients = history.get("gradients", [])

            if len(positions) == 0:
                continue

            # Normalize to [num_iter, num_aps, 3]
            if positions.ndim == 2:
                positions = positions[:, np.newaxis, :]  # [iter, 1, 3]
            num_aps_task = positions.shape[1]

            # Find the best iteration (highest min RSS dBm)
            best_iter = task_result.get("best_iteration", -1)
            if best_iter < 0 and min_rss_dbm_values:
                best_iter = int(np.argmax(min_rss_dbm_values))

            # Build orientation subtitle lines if available
            orientation_lines = ""
            best_dir = task_result.get("best_direction")
            final_dir = task_result.get("final_direction")
            best_la = task_result.get("best_look_at")
            final_la = task_result.get("final_look_at")
            if best_dir:
                orientation_lines += f"\nBest Dir: {_fmt_dir(best_dir)}"
            if best_la:
                orientation_lines += f"  LookAt: {_fmt_pos(best_la)}"
            if final_dir:
                orientation_lines += f"\nFinal Dir: {_fmt_dir(final_dir)}"
            if final_la:
                orientation_lines += f"  LookAt: {_fmt_pos(final_la)}"

            fig, axes = plt.subplots(2, 2, figsize=(14, 11))
            fig.suptitle(
                f"Task #{task_id} — Gradient Descent Trajectory\n"
                f"Best P5 RSS: {task_result['best_metric_dbm']:.2f} dBm "
                f"at iteration {best_iter + 1}"
                f"{orientation_lines}",
                fontsize=11,
                fontweight="bold",
            )

            # 1. Position trajectory (per-AP)
            ax = axes[0, 0]
            directions = history.get("directions", [])
            dirs_arr = np.array(directions) if directions else None
            if dirs_arr is not None and dirs_arr.ndim == 2:
                dirs_arr = dirs_arr[:, np.newaxis, :]  # [iter, 1, 3]
            arrow_scale = 2.0

            for k in range(num_aps_task):
                color = _AP_COLORS[k % len(_AP_COLORS)]
                marker = _AP_MARKERS[k % len(_AP_MARKERS)]
                lbl = f"AP{k} " if num_aps_task > 1 else ""
                ap_pos = positions[:, k, :]  # [iter, 3]
                ax.plot(
                    ap_pos[:, 0], ap_pos[:, 1],
                    f"-{marker}", color=color,
                    markersize=4, linewidth=1.5, alpha=0.6,
                    label=f"{lbl}path",
                )
                ax.plot(
                    ap_pos[0, 0], ap_pos[0, 1],
                    marker, color="green", markersize=12, zorder=5,
                    label=f"{lbl}Start" if k == 0 else None,
                )
                ax.plot(
                    ap_pos[-1, 0], ap_pos[-1, 1],
                    "s", color=color, markersize=12, zorder=5,
                    label=f"{lbl}End",
                )
                if 0 <= best_iter < len(ap_pos):
                    ax.plot(
                        ap_pos[best_iter, 0], ap_pos[best_iter, 1],
                        "*", color=color, markersize=18, zorder=6,
                        label=f"{lbl}Best (iter {best_iter + 1})" if k == 0 else None,
                    )
                # Direction arrows
                if dirs_arr is not None and len(dirs_arr) == len(positions):
                    ap_dir = dirs_arr[:, k, :]  # [iter, 3]
                    ax.annotate(
                        "", xy=(ap_pos[0, 0] + ap_dir[0, 0] * arrow_scale,
                                ap_pos[0, 1] + ap_dir[0, 1] * arrow_scale),
                        xytext=(ap_pos[0, 0], ap_pos[0, 1]),
                        arrowprops=dict(arrowstyle="->", color="green", lw=2),
                    )
                    ax.annotate(
                        "", xy=(ap_pos[-1, 0] + ap_dir[-1, 0] * arrow_scale,
                                ap_pos[-1, 1] + ap_dir[-1, 1] * arrow_scale),
                        xytext=(ap_pos[-1, 0], ap_pos[-1, 1]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2),
                    )

            if position_bounds:
                ax.set_xlim(position_bounds["x_min"], position_bounds["x_max"])
                ax.set_ylim(position_bounds["y_min"], position_bounds["y_max"])
            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Y Position (m)")
            title = "AP Position Trajectories" if num_aps_task > 1 else "AP Position Trajectory + Direction"
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")

            # 2. P5 RSS over iterations
            ax = axes[0, 1]
            if min_rss_dbm_values:
                iters = list(range(1, len(min_rss_dbm_values) + 1))
                ax.plot(iters, min_rss_dbm_values, "b-", linewidth=2)
                if 0 <= best_iter < len(min_rss_dbm_values):
                    ax.axvline(
                        best_iter + 1, color="red", linestyle="--",
                        alpha=0.7, label=f"Best iter {best_iter + 1}",
                    )
                    ax.plot(
                        best_iter + 1, min_rss_dbm_values[best_iter],
                        "r*", markersize=15, zorder=5,
                    )
            if rss_range_dbm:
                ax.set_ylim(rss_range_dbm[0], rss_range_dbm[1])
            ax.set_xlabel("Iteration")
            ax.set_ylabel("5th Percentile RSS (dBm)")
            ax.set_title("5th Percentile RSS Evolution")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            # 3. Coverage over iterations
            ax = axes[1, 0]
            if coverage_values:
                iters = list(range(1, len(coverage_values) + 1))
                ax.plot(iters, coverage_values, "g-", linewidth=2)
                if 0 <= best_iter < len(coverage_values):
                    ax.axvline(
                        best_iter + 1, color="red", linestyle="--",
                        alpha=0.7, label=f"Best iter {best_iter + 1}",
                    )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Coverage (%)")
            ax.set_title("Coverage Evolution")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            # 4. Gradient norm (log scale)
            ax = axes[1, 1]
            if gradients:
                grad_arr = np.array(gradients)
                if grad_arr.ndim == 3:
                    # Multi-AP: [iter, num_aps, 2] -> total norm
                    grad_norms = [
                        float(np.sqrt(np.sum(np.array(g) ** 2)))
                        for g in gradients
                    ]
                else:
                    grad_norms = [
                        np.sqrt(g[0] ** 2 + g[1] ** 2) for g in gradients
                    ]
                iters = list(range(1, len(grad_norms) + 1))
                ax.semilogy(iters, grad_norms, "r-", linewidth=2)
                if 0 <= best_iter < len(grad_norms):
                    ax.axvline(
                        best_iter + 1, color="red", linestyle="--",
                        alpha=0.7, label=f"Best iter {best_iter + 1}",
                    )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Gradient Norm")
            ax.set_title("Gradient Norm Evolution (log scale)")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"{filename_prefix}_{task_id}_trajectory.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(save_path)

        if saved_paths:
            print(f"Saved {len(saved_paths)} trajectory plots to: {save_dir}/")
        else:
            print("No trajectory plots saved (no tasks with history data)")

        return saved_paths

    def save_results_plot(
        self,
        results: Dict[str, Any],
        save_path: str,
        metric_name: str = "P5 RSS",
        position_bounds: Optional[Dict[str, float]] = None,
        rss_range_dbm: Optional[tuple] = None,
    ) -> None:
        """
        Save visualization of parallel optimization results to a file.

        Creates a 2x2 figure:
        1. P5 RSS distribution across tasks (dBm)
        2. Final positions scatter plot (color = P5 RSS dBm)
        3. Time per task bar chart
        4. Summary statistics text

        Args:
            results: Results dictionary from run().
            save_path: File path to save the plot (e.g., 'results/parallel.png').
            metric_name: Label for the metric axis.
            position_bounds: Optional dict with 'x_min', 'x_max', 'y_min', 'y_max'
                to set consistent axis limits on the positions scatter plot.
            rss_range_dbm: Optional tuple (min_dbm, max_dbm) to set consistent
                axis limits on P5 RSS histogram and colorbar across methods.
        """
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for saving
        import matplotlib.pyplot as plt

        all_results = results["all_results"]
        best_result = results["best_result"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Use dBm values for all analysis plots
        metrics_dbm = [r["best_metric_dbm"] for r in all_results]
        best_dbm = best_result["best_metric_dbm"]

        percentile_dbm = [
            r.get("grid_results", {}).get("percentile_score_dbm")
            for r in all_results
        ]
        percentile_dbm = [float(v) for v in percentile_dbm if v is not None]
        has_percentile = len(percentile_dbm) > 0
        best_pct_dbm = best_result.get("grid_results", {}).get("percentile_score_dbm")

        best_reflector_pos = (
            best_result.get("reflector_position")
            or best_result.get("grid_results", {}).get("reflector_position")
        )
        best_reflector_target = (
            best_result.get("reflector_target")
            or best_result.get("grid_results", {}).get("reflector_target")
        )

        # 1. Distribution of final P5 RSS (dBm)
        ax = axes[0, 0]
        hist_range = rss_range_dbm if rss_range_dbm else None
        ax.hist(
            metrics_dbm, bins=max(5, len(metrics_dbm) // 2),
            edgecolor="black", alpha=0.7, range=hist_range,
        )
        ax.axvline(
            best_dbm,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Best: {best_dbm:.2f} dBm",
        )
        if rss_range_dbm:
            ax.set_xlim(rss_range_dbm[0], rss_range_dbm[1])
        ax.set_xlabel(f"{metric_name} (dBm)")
        ax.set_ylabel("Number of Tasks")
        ax.set_title("Distribution of P5 RSS Across Tasks")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Final positions scatter (color = P5 RSS dBm)
        ax = axes[0, 1]
        sample_pos = all_results[0]["best_position"]
        _is_multi_ap = (
            isinstance(sample_pos, (list, np.ndarray))
            and len(sample_pos) > 0
            and isinstance(sample_pos[0], (list, np.ndarray))
        )
        if _is_multi_ap:
            n_aps = len(sample_pos)
            for k in range(n_aps):
                ap_positions = np.array([r["best_position"][k] for r in all_results])
                scatter = ax.scatter(
                    ap_positions[:, 0],
                    ap_positions[:, 1],
                    c=metrics_dbm,
                    s=80,
                    cmap="viridis",
                    edgecolor="black",
                    alpha=0.7,
                    marker=_AP_MARKERS[k % len(_AP_MARKERS)],
                    label=f"AP{k}",
                    vmin=rss_range_dbm[0] if rss_range_dbm else None,
                    vmax=rss_range_dbm[1] if rss_range_dbm else None,
                )
            # Connect APs of the same task with thin lines
            for r in all_results:
                pos = np.array(r["best_position"])
                ax.plot(pos[:, 0], pos[:, 1], "k-", alpha=0.15, linewidth=0.5)
            # Mark best task's APs
            best_pos = np.array(best_result["best_position"])
            for k in range(n_aps):
                ax.plot(best_pos[k, 0], best_pos[k, 1], "r*", markersize=18, zorder=6)
            # Draw orientation arrows for best result
            _best_dir = best_result.get("best_direction")
            if _best_dir is not None:
                _arrow_sc = 2.5
                _best_dir_arr = np.array(_best_dir)
                if _best_dir_arr.ndim == 2:
                    for k in range(min(n_aps, len(_best_dir_arr))):
                        ax.annotate(
                            "",
                            xy=(best_pos[k, 0] + _best_dir_arr[k, 0] * _arrow_sc,
                                best_pos[k, 1] + _best_dir_arr[k, 1] * _arrow_sc),
                            xytext=(best_pos[k, 0], best_pos[k, 1]),
                            arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
                            zorder=7,
                        )
                else:
                    ax.annotate(
                        "",
                        xy=(best_pos[0, 0] + _best_dir_arr[0] * _arrow_sc,
                            best_pos[0, 1] + _best_dir_arr[1] * _arrow_sc),
                        xytext=(best_pos[0, 0], best_pos[0, 1]),
                        arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
                        zorder=7,
                    )
            # Overlay reflector placement and focal target for best task
            if best_reflector_pos is not None:
                rp = np.asarray(best_reflector_pos)
                ax.plot(
                    rp[0], rp[1], marker="X", color="magenta", markersize=14,
                    markeredgecolor="black", label="Reflector", zorder=8,
                )
            if best_reflector_target is not None:
                rt = np.asarray(best_reflector_target)
                ax.plot(
                    rt[0], rt[1], marker="P", color="orange", markersize=13,
                    markeredgecolor="black", label="Focal Point", zorder=8,
                )
            if best_reflector_pos is not None and best_reflector_target is not None:
                rp = np.asarray(best_reflector_pos)
                rt = np.asarray(best_reflector_target)
                ax.plot(
                    [rp[0], rt[0]], [rp[1], rt[1]], "--", color="magenta",
                    linewidth=1.5, alpha=0.8, label="Reflector→Focal", zorder=7,
                )
        else:
            positions = np.array([r["best_position"] for r in all_results])
            scatter = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=metrics_dbm,
                s=100,
                cmap="viridis",
                edgecolor="black",
                alpha=0.7,
                vmin=rss_range_dbm[0] if rss_range_dbm else None,
                vmax=rss_range_dbm[1] if rss_range_dbm else None,
            )
            ax.plot(
                best_result["best_position"][0],
                best_result["best_position"][1],
                "r*",
                markersize=20,
                label="Best",
            )
            # Draw orientation arrow for best result
            _best_dir = best_result.get("best_direction")
            if _best_dir is not None:
                _arrow_sc = 2.5
                _bd = np.array(_best_dir)
                _bp = best_result["best_position"]
                ax.annotate(
                    "",
                    xy=(_bp[0] + _bd[0] * _arrow_sc,
                        _bp[1] + _bd[1] * _arrow_sc),
                    xytext=(_bp[0], _bp[1]),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
                    zorder=7,
                )
            if best_reflector_pos is not None:
                rp = np.asarray(best_reflector_pos)
                ax.plot(
                    rp[0], rp[1], marker="X", color="magenta", markersize=14,
                    markeredgecolor="black", label="Reflector", zorder=8,
                )
            if best_reflector_target is not None:
                rt = np.asarray(best_reflector_target)
                ax.plot(
                    rt[0], rt[1], marker="P", color="orange", markersize=13,
                    markeredgecolor="black", label="Focal Point", zorder=8,
                )
            if best_reflector_pos is not None and best_reflector_target is not None:
                rp = np.asarray(best_reflector_pos)
                rt = np.asarray(best_reflector_target)
                ax.plot(
                    [rp[0], rt[0]], [rp[1], rt[1]], "--", color="magenta",
                    linewidth=1.5, alpha=0.8, label="Reflector→Focal", zorder=7,
                )
        if position_bounds:
            ax.set_xlim(position_bounds["x_min"], position_bounds["x_max"])
            ax.set_ylim(position_bounds["y_min"], position_bounds["y_max"])
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Final Positions (color = P5 RSS dBm)")
        plt.colorbar(scatter, ax=ax, label=f"{metric_name} (dBm)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        # 3. Time per task
        ax = axes[1, 0]
        task_ids = [r["task_id"] for r in all_results]
        times = [r["time_elapsed"] for r in all_results]
        # Sort by task_id for display
        sorted_pairs = sorted(zip(task_ids, times))
        sorted_ids, sorted_times = zip(*sorted_pairs) if sorted_pairs else ([], [])
        ax.bar(sorted_ids, sorted_times, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Task ID")
        ax.set_ylabel("Time (s)")
        ax.set_title("Optimization Time per Task")
        ax.grid(True, alpha=0.3, axis="y")

        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis("off")
        stats = results["aggregate_stats"]
        pool_info = results.get("pool_info", {})
        best_dbm_val = best_result["best_metric_dbm"]
        summary = (
            f"PARALLEL OPTIMIZATION SUMMARY\n"
            f"\n"
            f"Pool: {pool_info.get('num_workers', '?')} workers, "
            f"{pool_info.get('num_tasks', len(all_results))} tasks\n"
            f"\n"
            f"Best Task: #{best_result['task_id']} "
            f"(Worker #{best_result['worker_id']})\n"
            f"Best AP Position(s): {_fmt_pos(best_result['best_position'])}\n"
            f"Best AP Direction(s): {_fmt_dir(best_result.get('best_direction'))}\n"
            f"Best Reflector Position: {_fmt_pos(best_reflector_pos)}\n"
            f"Best Reflector Focal Point: {_fmt_pos(best_reflector_target)}\n"
            f"Best P5 RSS: {best_dbm_val:.2f} dBm\n"
        )
        if best_pct_dbm is not None:
            summary += f"Best 5th Percentile: {float(best_pct_dbm):.2f} dBm\n"
        summary += (
            f"\n"
            f"P5 RSS Statistics (dBm):\n"
            f"  Mean: {stats['mean_metric_dbm']:.2f} +/- {stats['std_metric_dbm']:.2f}\n"
            f"  Range: [{stats['min_metric_dbm']:.2f}, {stats['max_metric_dbm']:.2f}]\n"
        )
        if has_percentile:
            summary += (
                f"\n"
                f"5th Percentile Statistics (dBm):\n"
                f"  Mean: {stats.get('mean_percentile_dbm', float('nan')):.2f} +/- "
                f"{stats.get('std_percentile_dbm', float('nan')):.2f}\n"
                f"  Range: [{stats.get('min_percentile_dbm', float('nan')):.2f}, "
                f"{stats.get('max_percentile_dbm', float('nan')):.2f}]\n"
            )
        ax.text(
            0.1,
            0.5,
            summary,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to: {save_path}")

    def shutdown(self):
        """Shut down the actor pool and kill all workers."""
        self._kill_workers()
        self._pool = None
        self._scene_config = None

    def _kill_workers(self):
        """Explicitly kill all worker actors."""
        for w in self._workers:
            try:
                ray.kill(w)
            except Exception:
                pass  # Actor may already be dead
        self._workers = []


def generate_random_initial_positions(
    num_positions: int,
    bounds: Dict[str, float],
    fixed_z: float = 3.8,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Generate random initial positions for parallel optimization tasks.

    Args:
        num_positions: Number of random positions to generate.
        bounds: Dictionary with spatial bounds keys
            ``x_min``, ``x_max``, ``y_min``, ``y_max``.
        fixed_z: Fixed height (z-coordinate) for all generated positions.
        seed: Random seed for reproducibility.

    Returns:
        List of position arrays [x, y, z].
    """
    rng = np.random.default_rng(seed)

    x_positions = rng.uniform(bounds["x_min"], bounds["x_max"], num_positions)
    y_positions = rng.uniform(bounds["y_min"], bounds["y_max"], num_positions)

    return [
        np.array([x, y, fixed_z])
        for x, y in zip(x_positions, y_positions)
    ]
