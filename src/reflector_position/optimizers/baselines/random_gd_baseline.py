"""Random multi-start baseline followed by targeted GD micro-tuning."""

from __future__ import annotations

import time
import inspect
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from reflector_position.optimizers.memetic.memetic_gd_logic import (
    run_targeted_gd_exploitation,
)

from .baseline_utils import build_evaluator_task, format_baseline_result


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
    top_level_keys = (
        "final_positions",
        "best_positions",
        "positions",
    )
    for key in top_level_keys:
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
    top_level_keys = (
        "final_directions",
        "best_directions",
        "directions",
    )
    for key in top_level_keys:
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
        # Forward-compatible path for adapters exposing the requested API shape.
        return gd_callable(
            ray_optimizer=ray_optimizer,
            tasks=tasks,
            gd_params=gd_params,
        )

    # Current project signature: run_targeted_gd_exploitation(gd_tasks, ray_optimizer, verbose)
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


def _extract_gd_iteration_traces(
    gd_output: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract aggregate and per-seed GD iteration traces from raw outputs."""
    raw_results = gd_output.get("all_fine_tuned_results", [])
    if not isinstance(raw_results, list):
        return [], []

    per_seed_traces: List[Dict[str, Any]] = []
    all_series: List[List[float]] = []
    running_global_best = float("inf")

    for seed_index, raw_result in enumerate(raw_results):
        if not isinstance(raw_result, Mapping):
            continue

        history = raw_result.get("history", {})
        if not isinstance(history, Mapping):
            continue

        primary_series = history.get("primary_loss", [])
        if not isinstance(primary_series, Sequence):
            continue

        series: List[float] = []
        seed_rows: List[Dict[str, Any]] = []
        running_seed_best = float("inf")
        for iteration_idx, value in enumerate(primary_series, start=1):
            try:
                loss_value = float(value)
            except (TypeError, ValueError):
                continue

            series.append(loss_value)
            running_seed_best = min(running_seed_best, loss_value)
            seed_rows.append(
                {
                    "iteration": int(iteration_idx),
                    "primary_loss": float(loss_value),
                    "running_best_primary_loss": float(running_seed_best),
                }
            )

        if not series:
            continue

        all_series.append(series)
        per_seed_traces.append(
            {
                "seed_index": int(seed_index),
                "task_id": int(raw_result.get("task_id", seed_index)),
                "trace": seed_rows,
            }
        )

    aggregate_trace: List[Dict[str, Any]] = []
    if all_series:
        max_len = max(len(series) for series in all_series)
        for iteration_idx in range(max_len):
            bucket = [series[iteration_idx] for series in all_series if iteration_idx < len(series)]
            if not bucket:
                continue

            min_loss = float(min(bucket))
            mean_loss = float(np.mean(bucket))
            max_loss = float(max(bucket))
            running_global_best = min(running_global_best, min_loss)
            aggregate_trace.append(
                {
                    "iteration": int(iteration_idx + 1),
                    "min_primary_loss": min_loss,
                    "mean_primary_loss": mean_loss,
                    "max_primary_loss": max_loss,
                    "running_best_primary_loss": float(running_global_best),
                }
            )

    return aggregate_trace, per_seed_traces


def run_random_multi_start_gd(
    ray_optimizer: Any,
    num_aps: int,
    fixed_z: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    gd_params: Dict[str, Any],
    num_samples: int = 10,
    optimize_orientation: bool = True,
) -> Dict[str, Any]:
    """Run random multi-start generation followed by targeted GD exploitation.

    Args:
        ray_optimizer: Initialized ``RawRayParallelOptimizer`` instance.
        num_aps: Number of APs per sampled configuration.
        fixed_z: Shared AP z-coordinate.
        x_bounds: Sampling bounds for x coordinates.
        y_bounds: Sampling bounds for y coordinates.
        gd_params: Shared GD optimization parameters.
        num_samples: Number of random starts.
        optimize_orientation: Whether to include orientation vectors.

    Returns:
        A formatted GA-schema-compatible baseline result dictionary.
    """
    if int(num_aps) < 1:
        raise ValueError(f"num_aps must be >= 1, got {num_aps}")
    if int(num_samples) < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    if x_min >= x_max:
        raise ValueError(f"x_bounds must satisfy min < max, got {x_bounds}")
    if y_min >= y_max:
        raise ValueError(f"y_bounds must satisfy min < max, got {y_bounds}")

    start_time = time.perf_counter()

    rng = np.random.default_rng()
    total_aps = int(num_aps)
    total_samples = int(num_samples)

    tasks: List[Dict[str, Any]] = []
    for _ in range(total_samples):
        positions = np.empty((total_aps, 2), dtype=np.float64)
        positions[:, 0] = rng.uniform(x_min, x_max, size=total_aps)
        positions[:, 1] = rng.uniform(y_min, y_max, size=total_aps)

        directions: Optional[List[Tuple[float, float, float]]] = None
        if bool(optimize_orientation):
            vectors = np.empty((total_aps, 3), dtype=np.float64)
            vectors[:, :2] = rng.uniform(-1.0, 1.0, size=(total_aps, 2))
            vectors[:, 2] = rng.uniform(-1.0, -1e-4, size=total_aps)

            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            vectors = vectors / norms

            directions = [
                (float(vector[0]), float(vector[1]), float(vector[2]))
                for vector in vectors
            ]

        task = build_evaluator_task(
            positions_xy=[(float(pos[0]), float(pos[1])) for pos in positions],
            directions_xyz=directions,
            fixed_z=float(fixed_z),
            num_aps=total_aps,
            optimize_orientation=bool(optimize_orientation),
        )
        tasks.append(task)

    gd_output = _run_gd_orchestrator(
        ray_optimizer=ray_optimizer,
        tasks=tasks,
        gd_params=dict(gd_params),
    )

    global_best = gd_output.get("global_best_result") if isinstance(gd_output, Mapping) else None
    if not isinstance(global_best, Mapping):
        raise RuntimeError("Random multi-start GD returned no global_best_result.")

    best_primary_loss = _extract_best_primary_loss(global_best)
    if best_primary_loss is None:
        raise RuntimeError("Global best GD result is missing a valid primary loss.")

    best_positions = _extract_best_positions(global_best)
    if not best_positions:
        raise RuntimeError("Global best GD result is missing AP position data.")

    best_directions = _extract_best_directions(global_best)

    elapsed = time.perf_counter() - start_time
    formatted = format_baseline_result(
        algorithm_name="random_multi_start_gd",
        best_positions=best_positions,
        best_directions=best_directions,
        best_primary_loss=float(best_primary_loss),
        loss_components=_extract_best_loss_components(global_best),
        physical_metrics=_extract_best_physical_metrics(global_best),
        time_elapsed=float(elapsed),
    )
    aggregate_trace, per_seed_traces = _extract_gd_iteration_traces(
        gd_output if isinstance(gd_output, Mapping) else {}
    )
    formatted["iteration_trace"] = aggregate_trace
    formatted["per_seed_iteration_traces"] = per_seed_traces
    formatted["num_iterations"] = len(aggregate_trace)
    formatted["num_random_starts"] = total_samples
    if isinstance(gd_output, Mapping):
        formatted["gd_metrics"] = dict(gd_output.get("metrics", {}))
    return formatted


__all__ = ["run_random_multi_start_gd"]
