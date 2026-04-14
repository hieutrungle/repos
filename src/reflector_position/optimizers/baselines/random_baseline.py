"""Pure random (Monte Carlo) baseline for AP placement evaluation.

This module generates random AP configurations, evaluates them through an
injected Ray actor pool, and returns a standardized result payload.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .baseline_utils import build_evaluator_task, format_baseline_result

if TYPE_CHECKING:
    from reflector_position.optimizers.memetic.raw_ray_parallel_optimizer import (
        RawRayActorPoolExecutor,
    )


def _format_pool_task(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, str, Dict[str, Any], Dict[str, Any]]:
    """Convert one evaluator task into the raw actor invocation tuple."""
    task_id, task = item
    return (int(task_id), "memetic_eval", task, {})


def _submit_eval_tasks(
    ray_pool: "RawRayActorPoolExecutor",
    tasks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Submit evaluator tasks through the injected pool and return ordered results."""
    indexed_tasks = list(enumerate(tasks))

    try:
        return list(ray_pool.map(_format_pool_task, indexed_tasks))
    except TypeError as exc:
        error_text = str(exc)
        is_signature_mismatch = (
            "missing 1 required positional argument: 'iterable'" in error_text
            or "takes 2 positional arguments" in error_text
            or "takes 3 positional arguments" in error_text
        )
        if not is_signature_mismatch:
            raise

        # Compatibility fallback for alternate pools exposing map(iterable).
        return list(ray_pool.map(tasks))  # type: ignore[misc, call-arg]


def _extract_primary_loss(result: Mapping[str, Any]) -> float:
    """Read primary loss from one worker result with fitness fallback."""
    if result.get("primary_loss") is not None:
        return float(result["primary_loss"])

    primary_fitness = result.get("primary_fitness")
    if primary_fitness is not None:
        return -float(primary_fitness)

    return float("inf")


def _extract_physical_metric(result: Mapping[str, Any], metric_key: str) -> Optional[float]:
    """Extract one numeric physical metric from worker payload when available."""
    physical_metrics = result.get("physical_metrics", {})
    if not isinstance(physical_metrics, Mapping):
        return None

    raw_value = physical_metrics.get(metric_key)
    if raw_value is None:
        return None

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    """Convert value to float when possible."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _aggregate_mean_metric_map(
    results: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> Dict[str, float]:
    """Aggregate one metric mapping field across results using arithmetic means."""
    buckets: Dict[str, List[float]] = {}
    for result in results:
        raw_metrics = result.get(metric_key)
        if not isinstance(raw_metrics, Mapping):
            continue

        for name, raw_value in raw_metrics.items():
            numeric = _as_float(raw_value)
            if numeric is None:
                continue
            buckets.setdefault(str(name), []).append(float(numeric))

    return {
        name: float(np.mean(values))
        for name, values in buckets.items()
        if values
    }


def _to_positions_xy(array: np.ndarray) -> List[Tuple[float, float]]:
    """Convert a ``[N,2]`` array into a list of ``(x, y)`` float tuples."""
    return [(float(row[0]), float(row[1])) for row in array]


def _to_directions_xyz(array: np.ndarray) -> List[Tuple[float, float, float]]:
    """Convert a ``[N,3]`` array into a list of ``(dx, dy, dz)`` float tuples."""
    return [(float(row[0]), float(row[1]), float(row[2])) for row in array]


def run_random_monte_carlo(
    ray_pool: "RawRayActorPoolExecutor",
    num_aps: int,
    fixed_z: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    num_samples: int = 100,
    optimize_orientation: bool = True,
    random_seed: Optional[int] = None,
    loss_kwargs: Optional[Mapping[str, Any]] = None,
    evaluation_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a pure random Monte Carlo baseline and return standardized output.

    Args:
        ray_pool: Initialized raw Ray actor-pool executor.
        num_aps: Number of APs per sampled configuration.
        fixed_z: Shared AP height used by the evaluator.
        x_bounds: Inclusive sampling bounds for x coordinates.
        y_bounds: Inclusive sampling bounds for y coordinates.
        num_samples: Number of random configurations to evaluate.
        optimize_orientation: Whether to sample and evaluate AP orientations.
        random_seed: Optional seed used to initialize numpy random generator.
        loss_kwargs: Optional memetic objective kwargs forwarded to evaluator.
        evaluation_params: Optional evaluator runtime kwargs (samples/max depth).

    Returns:
        GA-schema-compatible baseline result dictionary.
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
    rng = np.random.default_rng(None if random_seed is None else int(random_seed))
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
            directions = _to_directions_xyz(vectors / norms)

        task = build_evaluator_task(
            positions_xy=_to_positions_xy(positions),
            directions_xyz=directions,
            fixed_z=float(fixed_z),
            num_aps=total_aps,
            optimize_orientation=bool(optimize_orientation),
            loss_kwargs=loss_kwargs,
            evaluation_kwargs=evaluation_params,
        )
        tasks.append(task)

    results = _submit_eval_tasks(ray_pool=ray_pool, tasks=tasks)
    if not results:
        raise RuntimeError("Random baseline received no worker results.")

    task_by_id = {task_id: task for task_id, task in enumerate(tasks)}

    ordered_results: List[Optional[Mapping[str, Any]]] = [None] * len(tasks)
    for fallback_idx, raw_result in enumerate(results):
        if not isinstance(raw_result, Mapping):
            continue

        raw_task_id = raw_result.get("task_id", fallback_idx)
        try:
            task_id = int(raw_task_id)
        except (TypeError, ValueError):
            task_id = fallback_idx

        if 0 <= task_id < len(tasks):
            ordered_results[task_id] = raw_result

    for fallback_idx, raw_result in enumerate(results):
        if fallback_idx >= len(ordered_results):
            break
        if ordered_results[fallback_idx] is None and isinstance(raw_result, Mapping):
            ordered_results[fallback_idx] = raw_result

    iteration_trace: List[Dict[str, Any]] = []
    running_best_loss = float("inf")
    for iteration_idx, raw_result in enumerate(ordered_results, start=1):
        candidate_loss = (
            _extract_primary_loss(raw_result)
            if isinstance(raw_result, Mapping)
            else float("inf")
        )
        running_best_loss = min(running_best_loss, candidate_loss)
        task_id = iteration_idx - 1
        if isinstance(raw_result, Mapping):
            try:
                task_id = int(raw_result.get("task_id", task_id))
            except (TypeError, ValueError):
                task_id = iteration_idx - 1

        trace_row: Dict[str, Any] = {
            "iteration": int(iteration_idx),
            "task_id": int(task_id),
            "primary_loss": float(candidate_loss),
            "running_best_primary_loss": float(running_best_loss),
        }
        if isinstance(raw_result, Mapping):
            for metric_key in (
                "mean_rss_dbm",
                "min_rss_dbm",
                "p5_rss_dbm",
                "priority_mean_rss_dbm",
                "priority_min_rss_dbm",
                "priority_p5_rss_dbm",
            ):
                metric_value = _extract_physical_metric(raw_result, metric_key)
                if metric_value is not None:
                    trace_row[metric_key] = float(metric_value)

        iteration_trace.append(trace_row)

    valid_results: List[Mapping[str, Any]] = [
        result
        for result in ordered_results
        if isinstance(result, Mapping)
    ]
    primary_loss_samples: List[float] = []
    for result in valid_results:
        primary_loss = _extract_primary_loss(result)
        if np.isfinite(primary_loss):
            primary_loss_samples.append(float(primary_loss))

    reporting_primary_loss = (
        float(np.mean(primary_loss_samples))
        if primary_loss_samples
        else None
    )
    reporting_loss_components = _aggregate_mean_metric_map(
        results=valid_results,
        metric_key="loss_components",
    )
    reporting_physical_metrics = _aggregate_mean_metric_map(
        results=valid_results,
        metric_key="physical_metrics",
    )

    best_loss = float("inf")
    best_result: Optional[Mapping[str, Any]] = None
    best_task: Optional[Mapping[str, Any]] = None

    for fallback_idx, raw_result in enumerate(ordered_results):
        if not isinstance(raw_result, Mapping):
            continue

        candidate_loss = _extract_primary_loss(raw_result)
        if candidate_loss >= best_loss:
            continue

        raw_task_id = raw_result.get("task_id", fallback_idx)
        try:
            task_id = int(raw_task_id)
        except (TypeError, ValueError):
            task_id = fallback_idx

        resolved_task = task_by_id.get(task_id)
        if resolved_task is None and 0 <= fallback_idx < len(tasks):
            resolved_task = tasks[fallback_idx]
        if resolved_task is None:
            continue

        best_loss = candidate_loss
        best_result = raw_result
        best_task = resolved_task

    if best_result is None or best_task is None:
        raise RuntimeError("Random baseline could not resolve a valid best result.")

    best_positions = [
        (float(pos[0]), float(pos[1]))
        for pos in best_task.get("initial_positions", [])
    ]
    raw_directions = best_task.get("initial_directions_xyz")
    best_directions = (
        [(float(vec[0]), float(vec[1]), float(vec[2])) for vec in raw_directions]
        if isinstance(raw_directions, Sequence)
        else None
    )

    loss_components_raw = best_result.get("loss_components", {})
    physical_metrics_raw = best_result.get("physical_metrics", {})
    loss_components = (
        {
            str(name): float(value)
            for name, value in loss_components_raw.items()
            if value is not None
        }
        if isinstance(loss_components_raw, Mapping)
        else {}
    )
    physical_metrics = (
        {
            str(name): float(value)
            for name, value in physical_metrics_raw.items()
            if value is not None
        }
        if isinstance(physical_metrics_raw, Mapping)
        else {}
    )

    elapsed = time.perf_counter() - start_time
    formatted = format_baseline_result(
        algorithm_name="random_monte_carlo",
        best_positions=best_positions,
        best_directions=best_directions,
        best_primary_loss=best_loss,
        loss_components=loss_components,
        physical_metrics=physical_metrics,
        time_elapsed=float(elapsed),
    )
    formatted["reporting_mode"] = "mean_initializations"
    formatted["reporting_num_initializations"] = int(total_samples)
    formatted["reporting_num_valid_results"] = int(len(valid_results))
    formatted["reporting_primary_loss"] = reporting_primary_loss
    formatted["reporting_primary_fitness"] = (
        float(-reporting_primary_loss)
        if reporting_primary_loss is not None
        else None
    )
    formatted["reporting_loss_components"] = reporting_loss_components
    formatted["reporting_physical_metrics"] = reporting_physical_metrics
    formatted["iteration_trace"] = iteration_trace
    formatted["num_iterations"] = len(iteration_trace)
    return formatted


__all__ = ["run_random_monte_carlo"]
