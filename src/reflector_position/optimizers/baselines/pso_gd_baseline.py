"""Particle Swarm Optimization followed by GD micro-tuning baseline."""

from __future__ import annotations

import inspect
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from reflector_position.optimizers.memetic.memetic_gd_logic import (
    run_targeted_gd_exploitation,
)

from .baseline_utils import build_evaluator_task, format_baseline_result


def _format_pool_task(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, str, Dict[str, Any], Dict[str, Any]]:
    """Convert one evaluator task into the raw actor invocation tuple."""
    task_id, task = item
    return (int(task_id), "memetic_eval", task, {})


def _extract_primary_loss(result: Mapping[str, Any]) -> float:
    """Read primary loss from one worker result with fitness fallback."""
    if result.get("primary_loss") is not None:
        return float(result["primary_loss"])

    primary_fitness = result.get("primary_fitness")
    if primary_fitness is not None:
        return -float(primary_fitness)

    return float("inf")


def _extract_selected_physical_metrics(result: Mapping[str, Any]) -> Dict[str, float]:
    """Extract selected RSSI metrics from one evaluator result."""
    physical_metrics = result.get("physical_metrics", {})
    if not isinstance(physical_metrics, Mapping):
        return {}

    selected: Dict[str, float] = {}
    for metric_key in ("mean_rss_dbm", "min_rss_dbm", "p5_rss_dbm"):
        raw_value = physical_metrics.get(metric_key)
        if raw_value is None:
            continue
        try:
            selected[metric_key] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return selected


def _project_directions(raw_directions: np.ndarray) -> np.ndarray:
    """Project direction vectors onto unit sphere with strict downward z."""
    z = np.clip(raw_directions[..., 2], -1.0, -1e-4)
    z_exp = z[..., None]

    xy = raw_directions[..., :2]
    xy_norm = np.linalg.norm(xy, axis=-1, keepdims=True)
    safe_xy_norm = np.maximum(xy_norm, 1e-12)
    xy_unit = xy / safe_xy_norm

    xy_mag = np.sqrt(np.maximum(1.0 - z_exp * z_exp, 0.0))
    projected_xy = xy_unit * xy_mag

    near_zero_mask = xy_norm[..., 0] < 1e-12
    if np.any(near_zero_mask):
        projected_xy[near_zero_mask, 0] = xy_mag[near_zero_mask, 0]
        projected_xy[near_zero_mask, 1] = 0.0

    projected = np.concatenate([projected_xy, z_exp], axis=-1)
    norm = np.linalg.norm(projected, axis=-1, keepdims=True)
    return projected / np.maximum(norm, 1e-12)


def _submit_eval_tasks(ray_optimizer: Any, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate one swarm state through the active Ray actor pool when possible."""
    indexed_tasks = list(enumerate(tasks))
    pool = getattr(ray_optimizer, "pool", None)
    if pool is None:
        pool = getattr(ray_optimizer, "_pool", None)

    if pool is not None and hasattr(pool, "map"):
        task_args = [_format_pool_task(item) for item in indexed_tasks]
        results = list(
            pool.map(
                lambda actor, args: actor.optimize.remote(*args),
                task_args,
            )
        )
    else:
        scene_config = getattr(ray_optimizer, "_scene_config", None)
        if scene_config is None:
            scene_config = getattr(ray_optimizer, "scene_config", None)

        if not isinstance(scene_config, Mapping):
            raise RuntimeError(
                "ray_optimizer does not expose an active pool or scene configuration "
                "required for static PSO evaluations."
            )

        run_output = ray_optimizer.run(
            scene_config=dict(scene_config),
            optimizer_method="memetic_eval",
            work_items=tasks,
            optimization_params={},
            verbose=False,
        )
        raw_results = run_output.get("all_results") if isinstance(run_output, Mapping) else None
        if not isinstance(raw_results, list):
            raise RuntimeError("ray_optimizer.run returned malformed 'all_results' payload.")
        results = raw_results

    # Normalize to task-id order for stable particle-to-result alignment.
    ordered: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
    for fallback_idx, raw_result in enumerate(results):
        if not isinstance(raw_result, Mapping):
            continue
        raw_task_id = raw_result.get("task_id", fallback_idx)
        try:
            task_id = int(raw_task_id)
        except (TypeError, ValueError):
            task_id = fallback_idx

        if 0 <= task_id < len(tasks):
            ordered[task_id] = dict(raw_result)

    missing_ids = [i for i, item in enumerate(ordered) if item is None]
    if missing_ids:
        raise RuntimeError(f"Missing evaluator results for particle ids: {missing_ids}")

    return [item for item in ordered if item is not None]


def _to_positions_xy(value: Any) -> Optional[List[Tuple[float, float]]]:
    """Convert candidate position payloads into ``[(x, y), ...]`` form."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) == 0:
        return None

    first = value[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        positions: List[Tuple[float, float]] = []
        for entry in value:
            if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)) or len(entry) < 2:
                return None
            positions.append((float(entry[0]), float(entry[1])))
        return positions

    if len(value) >= 2:
        return [(float(value[0]), float(value[1]))]

    return None


def _to_directions_xyz(value: Any) -> Optional[List[Tuple[float, float, float]]]:
    """Convert candidate direction payloads into ``[(dx, dy, dz), ...]`` form."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) == 0:
        return None

    first = value[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        directions: List[Tuple[float, float, float]] = []
        for entry in value:
            if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)) or len(entry) < 3:
                return None
            directions.append((float(entry[0]), float(entry[1]), float(entry[2])))
        return directions

    if len(value) >= 3:
        return [(float(value[0]), float(value[1]), float(value[2]))]

    return None


def _extract_best_primary_loss(global_best: Mapping[str, Any]) -> Optional[float]:
    """Extract best primary loss from common GD payload layouts."""
    for key in ("best_primary_loss", "primary_loss", "final_primary_loss"):
        raw_value = global_best.get(key)
        if raw_value is not None:
            try:
                return float(raw_value)
            except (TypeError, ValueError):
                pass

    raw_fitness = global_best.get("primary_fitness")
    if raw_fitness is not None:
        try:
            return -float(raw_fitness)
        except (TypeError, ValueError):
            pass

    result_summary = global_best.get("results")
    if isinstance(result_summary, Mapping):
        for key in ("primary_loss", "final_primary_loss"):
            raw_value = result_summary.get(key)
            if raw_value is not None:
                try:
                    return float(raw_value)
                except (TypeError, ValueError):
                    pass

    return None


def _extract_best_loss_components(global_best: Mapping[str, Any]) -> Dict[str, float]:
    """Extract standardized best loss components from a GD global-best payload."""
    for container in (global_best, global_best.get("results")):
        if not isinstance(container, Mapping):
            continue

        for key in ("best_loss_components", "loss_components"):
            candidate = container.get(key)
            if isinstance(candidate, Mapping):
                return {
                    str(name): float(value)
                    for name, value in candidate.items()
                    if value is not None
                }
    return {}


def _extract_best_physical_metrics(global_best: Mapping[str, Any]) -> Dict[str, float]:
    """Extract standardized best physical metrics from a GD global-best payload."""
    for container in (global_best, global_best.get("results")):
        if not isinstance(container, Mapping):
            continue

        for key in ("best_physical_metrics", "physical_metrics"):
            candidate = container.get(key)
            if isinstance(candidate, Mapping):
                return {
                    str(name): float(value)
                    for name, value in candidate.items()
                    if value is not None
                }
    return {}


def _extract_best_positions(global_best: Mapping[str, Any]) -> Optional[List[Tuple[float, float]]]:
    """Extract best/final AP positions from GD outputs with robust fallbacks."""
    for key in ("final_positions", "best_positions", "positions"):
        positions = _to_positions_xy(global_best.get(key))
        if positions is not None:
            return positions

    result_summary = global_best.get("results")
    if isinstance(result_summary, Mapping):
        for section_name in ("final_configuration", "best_configuration"):
            section = result_summary.get(section_name)
            if isinstance(section, Mapping):
                positions = _to_positions_xy(section.get("positions"))
                if positions is not None:
                    return positions

        positions = _to_positions_xy(result_summary.get("positions"))
        if positions is not None:
            return positions

    optimizer_result = global_best.get("optimizer_result")
    if isinstance(optimizer_result, Sequence) and len(optimizer_result) > 0:
        positions = _to_positions_xy(optimizer_result[0])
        if positions is not None:
            return positions

    optimizer_kwargs = global_best.get("optimizer_kwargs")
    if isinstance(optimizer_kwargs, Mapping):
        positions = _to_positions_xy(optimizer_kwargs.get("initial_positions"))
        if positions is not None:
            return positions

    return None


def _extract_best_directions(global_best: Mapping[str, Any]) -> Optional[List[Tuple[float, float, float]]]:
    """Extract best/final AP directions from GD outputs with robust fallbacks."""
    for key in ("final_directions", "best_directions", "directions"):
        directions = _to_directions_xyz(global_best.get(key))
        if directions is not None:
            return directions

    result_summary = global_best.get("results")
    if isinstance(result_summary, Mapping):
        for section_name in ("final_configuration", "best_configuration"):
            section = result_summary.get(section_name)
            if isinstance(section, Mapping):
                directions = _to_directions_xyz(section.get("directions"))
                if directions is not None:
                    return directions

    optimizer_kwargs = global_best.get("optimizer_kwargs")
    if isinstance(optimizer_kwargs, Mapping):
        directions = _to_directions_xyz(optimizer_kwargs.get("initial_directions_xyz"))
        if directions is not None:
            return directions

    return None


def _run_gd_orchestrator(
    ray_optimizer: Any,
    tasks: List[Dict[str, Any]],
    gd_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Run targeted GD with compatibility for multiple call signatures."""
    gd_callable: Any = run_targeted_gd_exploitation
    parameter_names = set(inspect.signature(run_targeted_gd_exploitation).parameters.keys())
    if {"tasks", "gd_params", "ray_optimizer"}.issubset(parameter_names):
        return gd_callable(
            ray_optimizer=ray_optimizer,
            tasks=tasks,
            gd_params=gd_params,
        )

    cached_scene_config = getattr(ray_optimizer, "_scene_config", None)
    scene_config = (
        dict(cached_scene_config)
        if isinstance(cached_scene_config, Mapping)
        else None
    )

    gd_tasks: List[Dict[str, Any]] = []
    for task in tasks:
        merged_task = {**task, **dict(gd_params)}
        if scene_config is not None and "scene_config" not in merged_task:
            merged_task["scene_config"] = dict(scene_config)
        gd_tasks.append(merged_task)

    return gd_callable(
        gd_tasks=gd_tasks,
        ray_optimizer=ray_optimizer,
        verbose=bool(gd_params.get("verbose", False)),
    )


def _extract_gd_iteration_trace(global_best: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Extract one GD loss trajectory from global-best history when available."""
    history = global_best.get("history", {})
    if not isinstance(history, Mapping):
        return []

    primary_series = history.get("primary_loss", [])
    if not isinstance(primary_series, Sequence):
        return []

    trace: List[Dict[str, Any]] = []
    running_best = float("inf")
    for iteration_idx, raw_value in enumerate(primary_series, start=1):
        try:
            loss_value = float(raw_value)
        except (TypeError, ValueError):
            continue

        running_best = min(running_best, loss_value)
        trace.append(
            {
                "iteration": int(iteration_idx),
                "primary_loss": float(loss_value),
                "running_best_primary_loss": float(running_best),
            }
        )
    return trace


def run_pso_gd_baseline(
    ray_optimizer: Any,
    num_aps: int,
    fixed_z: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    pso_params: Dict[str, Any],
    gd_params: Dict[str, Any],
    optimize_orientation: bool = True,
    loss_kwargs: Optional[Mapping[str, Any]] = None,
    evaluation_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run PSO macro-search followed by GD micro-tuning and format best result."""
    total_aps = int(num_aps)
    if total_aps < 1:
        raise ValueError(f"num_aps must be >= 1, got {num_aps}")

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    if x_min >= x_max:
        raise ValueError(f"x_bounds must satisfy min < max, got {x_bounds}")
    if y_min >= y_max:
        raise ValueError(f"y_bounds must satisfy min < max, got {y_bounds}")

    swarm_size = int(pso_params.get("swarm_size", 20))
    num_iterations = int(pso_params.get("num_iterations", 10))
    w = float(pso_params.get("w", 0.5))
    c1 = float(pso_params.get("c1", 1.5))
    c2 = float(pso_params.get("c2", 1.5))

    if swarm_size < 1:
        raise ValueError(f"swarm_size must be >= 1, got {swarm_size}")
    if num_iterations < 1:
        raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")

    start_time = time.perf_counter()
    rng = np.random.default_rng()

    # Particle state: [swarm, num_aps, dim].
    positions = np.empty((swarm_size, total_aps, 2), dtype=np.float64)
    positions[..., 0] = rng.uniform(x_min, x_max, size=(swarm_size, total_aps))
    positions[..., 1] = rng.uniform(y_min, y_max, size=(swarm_size, total_aps))

    pos_vel = rng.uniform(-1.0, 1.0, size=(swarm_size, total_aps, 2)).astype(np.float64)

    directions: Optional[np.ndarray] = None
    dir_vel: Optional[np.ndarray] = None
    if bool(optimize_orientation):
        raw_dir = np.empty((swarm_size, total_aps, 3), dtype=np.float64)
        raw_dir[..., :2] = rng.uniform(-1.0, 1.0, size=(swarm_size, total_aps, 2))
        raw_dir[..., 2] = rng.uniform(-1.0, -1e-4, size=(swarm_size, total_aps))
        directions = _project_directions(raw_dir)

        dir_vel = rng.uniform(-0.5, 0.5, size=(swarm_size, total_aps, 3)).astype(np.float64)

    pbest_positions = positions.copy()
    pbest_directions = directions.copy() if directions is not None else None
    pbest_losses = np.full((swarm_size,), np.inf, dtype=np.float64)
    pso_iteration_trace: List[Dict[str, Any]] = []

    gbest_loss = float("inf")
    gbest_positions: Optional[np.ndarray] = None
    gbest_directions: Optional[np.ndarray] = None

    for iteration_idx in range(num_iterations):
        tasks: List[Dict[str, Any]] = []
        for particle_index in range(swarm_size):
            particle_positions = [
                (float(positions[particle_index, ap_index, 0]), float(positions[particle_index, ap_index, 1]))
                for ap_index in range(total_aps)
            ]

            particle_directions: Optional[List[Tuple[float, float, float]]] = None
            if directions is not None:
                particle_directions = [
                    (
                        float(directions[particle_index, ap_index, 0]),
                        float(directions[particle_index, ap_index, 1]),
                        float(directions[particle_index, ap_index, 2]),
                    )
                    for ap_index in range(total_aps)
                ]

            tasks.append(
                build_evaluator_task(
                    positions_xy=particle_positions,
                    directions_xyz=particle_directions,
                    fixed_z=float(fixed_z),
                    num_aps=total_aps,
                    optimize_orientation=bool(optimize_orientation),
                    loss_kwargs=loss_kwargs,
                    evaluation_kwargs=evaluation_params,
                )
            )

        eval_results = _submit_eval_tasks(ray_optimizer=ray_optimizer, tasks=tasks)
        losses = np.asarray(
            [
                _extract_primary_loss(result)
                if isinstance(result, Mapping)
                else float("inf")
                for result in eval_results
            ],
            dtype=np.float64,
        )

        improved = losses < pbest_losses
        if np.any(improved):
            pbest_losses[improved] = losses[improved]
            pbest_positions[improved] = positions[improved]
            if directions is not None and pbest_directions is not None:
                pbest_directions[improved] = directions[improved]

        candidate_index = int(np.argmin(pbest_losses))
        candidate_loss = float(pbest_losses[candidate_index])
        if np.isfinite(candidate_loss) and candidate_loss < gbest_loss:
            gbest_loss = candidate_loss
            gbest_positions = pbest_positions[candidate_index].copy()
            if pbest_directions is not None:
                gbest_directions = pbest_directions[candidate_index].copy()

        swarm_best = float(np.min(losses)) if losses.size else float("inf")
        swarm_mean = float(np.mean(losses)) if losses.size else float("inf")
        global_best_for_trace = float(gbest_loss) if np.isfinite(gbest_loss) else float(swarm_best)
        trace_row: Dict[str, Any] = {
            "iteration": int(iteration_idx + 1),
            "swarm_best_primary_loss": float(swarm_best),
            "swarm_mean_primary_loss": float(swarm_mean),
            "global_best_primary_loss": float(global_best_for_trace),
            "running_best_primary_loss": float(global_best_for_trace),
        }
        if losses.size and eval_results:
            best_particle_idx = int(np.argmin(losses))
            if 0 <= best_particle_idx < len(eval_results):
                trace_row.update(
                    _extract_selected_physical_metrics(eval_results[best_particle_idx])
                )

        pso_iteration_trace.append(trace_row)

        if gbest_positions is None:
            continue

        r1_pos = rng.random(size=positions.shape)
        r2_pos = rng.random(size=positions.shape)
        gbest_pos_broadcast = gbest_positions[np.newaxis, ...]

        pos_vel = (
            w * pos_vel
            + c1 * r1_pos * (pbest_positions - positions)
            + c2 * r2_pos * (gbest_pos_broadcast - positions)
        )
        positions = positions + pos_vel

        positions[..., 0] = np.clip(positions[..., 0], x_min, x_max)
        positions[..., 1] = np.clip(positions[..., 1], y_min, y_max)

        if directions is not None and dir_vel is not None and pbest_directions is not None and gbest_directions is not None:
            r1_dir = rng.random(size=directions.shape)
            r2_dir = rng.random(size=directions.shape)
            gbest_dir_broadcast = gbest_directions[np.newaxis, ...]

            dir_vel = (
                w * dir_vel
                + c1 * r1_dir * (pbest_directions - directions)
                + c2 * r2_dir * (gbest_dir_broadcast - directions)
            )
            directions = _project_directions(directions + dir_vel)

    if gbest_positions is None or not np.isfinite(gbest_loss):
        raise RuntimeError("PSO phase failed to discover a valid global-best configuration.")

    gbest_task = build_evaluator_task(
        positions_xy=[
            (float(gbest_positions[ap_index, 0]), float(gbest_positions[ap_index, 1]))
            for ap_index in range(total_aps)
        ],
        directions_xyz=(
            [
                (
                    float(gbest_directions[ap_index, 0]),
                    float(gbest_directions[ap_index, 1]),
                    float(gbest_directions[ap_index, 2]),
                )
                for ap_index in range(total_aps)
            ]
            if gbest_directions is not None
            else None
        ),
        fixed_z=float(fixed_z),
        num_aps=total_aps,
        optimize_orientation=bool(optimize_orientation),
        loss_kwargs=loss_kwargs,
        evaluation_kwargs=evaluation_params,
    )

    gd_output = _run_gd_orchestrator(
        ray_optimizer=ray_optimizer,
        tasks=[gbest_task],
        gd_params=dict(gd_params),
    )

    global_best = gd_output.get("global_best_result") if isinstance(gd_output, Mapping) else None
    if not isinstance(global_best, Mapping):
        raise RuntimeError("PSO+GD baseline returned no global_best_result from GD phase.")

    best_primary_loss = _extract_best_primary_loss(global_best)
    if best_primary_loss is None:
        raise RuntimeError("PSO+GD baseline global-best GD result lacks valid primary loss.")

    best_positions = _extract_best_positions(global_best)
    if not best_positions:
        raise RuntimeError("PSO+GD baseline global-best GD result lacks position payload.")

    best_directions = _extract_best_directions(global_best)

    elapsed = time.perf_counter() - start_time
    formatted = format_baseline_result(
        algorithm_name="pso_gd",
        best_positions=best_positions,
        best_directions=best_directions,
        best_primary_loss=float(best_primary_loss),
        loss_components=_extract_best_loss_components(global_best),
        physical_metrics=_extract_best_physical_metrics(global_best),
        time_elapsed=float(elapsed),
    )
    gd_iteration_trace = _extract_gd_iteration_trace(global_best)
    combined_trace: List[Dict[str, Any]] = []
    for row in pso_iteration_trace:
        combined_trace.append(
            {
                "phase": "pso",
                **row,
            }
        )
    for row in gd_iteration_trace:
        combined_trace.append(
            {
                "phase": "gd",
                **row,
            }
        )

    formatted["pso_iteration_trace"] = pso_iteration_trace
    formatted["gd_iteration_trace"] = gd_iteration_trace
    formatted["iteration_trace"] = combined_trace
    formatted["num_iterations"] = len(combined_trace)
    if isinstance(gd_output, Mapping):
        formatted["gd_metrics"] = dict(gd_output.get("metrics", {}))
    return formatted


__all__ = ["run_pso_gd_baseline"]
