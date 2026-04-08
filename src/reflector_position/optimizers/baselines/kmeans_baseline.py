"""K-Means centroid baseline for AP placement evaluation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans

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


def run_kmeans_baseline(
    ray_pool: "RawRayActorPoolExecutor",
    num_aps: int,
    fixed_z: float,
    floorplan_coords: np.ndarray,
    optimize_orientation: bool = True,
    loss_kwargs: Optional[Mapping[str, Any]] = None,
    evaluation_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run K-Means centroid placement baseline and return standardized output.

    Args:
        ray_pool: Initialized raw Ray actor-pool executor.
        num_aps: Number of APs (and clusters) to place.
        fixed_z: Shared AP height used by the evaluator.
        floorplan_coords: Valid floorplan samples as ``[N, 2]`` array.
        optimize_orientation: Whether to include AP orientation in the task.
        loss_kwargs: Optional memetic objective kwargs forwarded to evaluator.
        evaluation_params: Optional evaluator runtime kwargs (samples/max depth).

    Returns:
        GA-schema-compatible baseline result dictionary.
    """
    total_aps = int(num_aps)
    if total_aps < 1:
        raise ValueError(f"num_aps must be >= 1, got {num_aps}")

    coords = np.asarray(floorplan_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            "floorplan_coords must be a 2-D array with shape [N, 2], "
            f"got {coords.shape}"
        )
    if coords.shape[0] < total_aps:
        raise ValueError(
            "floorplan_coords must contain at least num_aps points "
            f"({coords.shape[0]} < {total_aps})"
        )

    start_time = time.perf_counter()

    kmeans = KMeans(n_clusters=total_aps)
    kmeans.fit(coords)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)

    best_positions = [
        (float(center[0]), float(center[1]))
        for center in centers
    ]
    best_directions: Optional[List[Tuple[float, float, float]]] = (
        [(0.0, 0.0, -1.0) for _ in range(total_aps)] if bool(optimize_orientation) else None
    )

    task = build_evaluator_task(
        positions_xy=best_positions,
        directions_xyz=best_directions,
        fixed_z=float(fixed_z),
        num_aps=total_aps,
        optimize_orientation=bool(optimize_orientation),
        loss_kwargs=loss_kwargs,
        evaluation_kwargs=evaluation_params,
    )

    results = _submit_eval_tasks(ray_pool=ray_pool, tasks=[task])
    if not results:
        raise RuntimeError("K-Means baseline received no worker results.")

    first_result = results[0]
    if not isinstance(first_result, Mapping):
        raise RuntimeError("K-Means baseline received an invalid worker result payload.")

    best_primary_loss = _extract_primary_loss(first_result)

    loss_components_raw = first_result.get("loss_components", {})
    physical_metrics_raw = first_result.get("physical_metrics", {})
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
        algorithm_name="kmeans",
        best_positions=best_positions,
        best_directions=best_directions,
        best_primary_loss=best_primary_loss,
        loss_components=loss_components,
        physical_metrics=physical_metrics,
        time_elapsed=float(elapsed),
    )
    iteration_row: Dict[str, Any] = {
        "iteration": 1,
        "task_id": int(first_result.get("task_id", 0)) if isinstance(first_result, Mapping) else 0,
        "primary_loss": float(best_primary_loss),
        "running_best_primary_loss": float(best_primary_loss),
    }
    for metric_key in ("mean_rss_dbm", "min_rss_dbm", "p5_rss_dbm"):
        if metric_key in physical_metrics:
            iteration_row[metric_key] = float(physical_metrics[metric_key])

    formatted["iteration_trace"] = [iteration_row]
    formatted["num_iterations"] = 1
    return formatted


__all__ = ["run_kmeans_baseline"]
