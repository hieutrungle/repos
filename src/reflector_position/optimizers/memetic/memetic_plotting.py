"""Plotting helpers for memetic optimization outputs.

This module converts raw memetic GA/GD outputs into publication-ready plots.
It is intentionally independent from the Ray orchestration layer so the
pipeline can remain focused on execution and artifact routing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_AP_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
_AP_MARKERS = ["o", "s", "^", "D", "v", "P"]
_PHYSICAL_METRIC_PRIORITY = ("coverage_pct", "p5_rss_dbm", "min_rss_dbm", "mean_rss_dbm")
_RSSI_Y_AXIS_ROUND_STEP = 5.0
_DEFAULT_MEAN_P5_COMBINED_Y_RANGE = (-80.0, -40.0)
_GA_BEST_METRIC_PLOT_SPECS = (
    (
        "mean_rss_dbm",
        "priority_mean_rss_dbm",
        "Mean RSSI (dBm)",
        "ga_best_mean_rssi_trend_plot",
        "ga_best_mean_rssi_trend.png",
        "ga_gd_stitched_mean_rssi_plot",
        "ga_gd_stitched_mean_rssi.png",
    ),
    (
        "p5_rss_dbm",
        "priority_p5_rss_dbm",
        "P5 RSSI (dBm)",
        "ga_best_p5_rssi_trend_plot",
        "ga_best_p5_rssi_trend.png",
        "ga_gd_stitched_p5_rssi_plot",
        "ga_gd_stitched_p5_rssi.png",
    ),
    (
        "min_rss_dbm",
        "priority_min_rss_dbm",
        "Min RSSI (dBm)",
        "ga_best_min_rssi_trend_plot",
        "ga_best_min_rssi_trend.png",
        "ga_gd_stitched_min_rssi_plot",
        "ga_gd_stitched_min_rssi.png",
    ),
)


def _set_integer_x_ticks(ax: Any) -> None:
    """Force integer-only major ticks for generation/iteration axes."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _extract_history(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return history payload when present, else empty mapping."""
    history = result.get("history")
    return history if isinstance(history, Mapping) else {}


def _extract_results_payload(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return optimizer results payload when present, else empty mapping."""
    payload = result.get("results")
    return payload if isinstance(payload, Mapping) else {}


def _extract_reflector_snapshot(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return reflector snapshot payload when present, else empty mapping."""
    snapshot = result.get("reflector_snapshot")
    return snapshot if isinstance(snapshot, Mapping) else {}


def _extract_primary_loss_series(result: Mapping[str, Any]) -> List[float]:
    """Return the primary-loss history as floats when available."""
    history = _extract_history(result)
    values = history.get("primary_loss")
    if isinstance(values, Sequence):
        return [float(v) for v in values]
    return []


def _extract_physical_metric_series(result: Mapping[str, Any], metric_name: str) -> List[float]:
    """Return one physical-metric history series when available."""
    history = _extract_history(result)
    physical_metrics = history.get("physical_metrics")
    if isinstance(physical_metrics, Sequence):
        output: List[float] = []
        for item in physical_metrics:
            if isinstance(item, Mapping) and item.get(metric_name) is not None:
                output.append(float(item[metric_name]))
        if output:
            return output

    legacy_key_map = {
        "coverage_pct": "coverage_values",
        "p5_rss_dbm": "p5_rss_dbm_values",
        "min_rss_dbm": "min_rss_dbm_values",
    }
    legacy_key = legacy_key_map.get(metric_name)
    if legacy_key is None:
        return []

    values = history.get(legacy_key)
    if isinstance(values, Sequence):
        return [float(v) for v in values]
    return []


def _extract_priority_metric_series(result: Mapping[str, Any], metric_name: str) -> List[float]:
    """Return one priority-report metric series when available."""
    history = _extract_history(result)
    physical_metrics = history.get("physical_metrics")
    if not isinstance(physical_metrics, Sequence):
        return []

    series: List[float] = []
    for item in physical_metrics:
        if not isinstance(item, Mapping):
            continue
        value = item.get(metric_name)
        if value is not None:
            series.append(float(value))
    return series


def _extract_spatial_priority_map(result: Mapping[str, Any]) -> Optional[np.ndarray]:
    """Return the spatial-priority map (2-D) when present in result payload."""
    raw = result.get("spatial_weights")
    if raw is None:
        raw = _extract_results_payload(result).get("spatial_weights")
    if raw is None:
        return None

    try:
        arr = np.asarray(raw, dtype=np.float32)
    except Exception:
        return None

    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        side = int(np.sqrt(arr.size))
        if side * side == arr.size:
            arr = arr.reshape(side, side)
        else:
            return None
    if arr.ndim != 2:
        return None
    return arr


def _extract_best_iteration(result: Mapping[str, Any]) -> int:
    """Return best iteration index based on primary loss when available."""
    primary_loss = _extract_primary_loss_series(result)
    if primary_loss:
        return int(np.argmin(primary_loss))
    return -1


def _extract_best_primary_loss(result: Mapping[str, Any]) -> Optional[float]:
    """Return best-observed primary loss from raw result payload."""
    primary_loss = _extract_primary_loss_series(result)
    if primary_loss:
        return float(min(primary_loss))

    results_payload = _extract_results_payload(result)
    value = results_payload.get("primary_loss")
    return float(value) if value is not None else None


def _extract_final_primary_loss(result: Mapping[str, Any]) -> Optional[float]:
    """Return final primary loss from raw result payload."""
    primary_loss = _extract_primary_loss_series(result)
    if primary_loss:
        return float(primary_loss[-1])

    results_payload = _extract_results_payload(result)
    value = results_payload.get("final_primary_loss", results_payload.get("primary_loss"))
    return float(value) if value is not None else None


def _extract_best_position(result: Mapping[str, Any]) -> Optional[Any]:
    """Return best AP position(s) from history or standardized result payload."""
    history = _extract_history(result)
    positions = history.get("positions")
    best_iter = _extract_best_iteration(result)
    if isinstance(positions, Sequence) and len(positions) > 0:
        idx = best_iter if 0 <= best_iter < len(positions) else len(positions) - 1
        return positions[idx]

    results_payload = _extract_results_payload(result)
    best_configuration = results_payload.get("best_configuration")
    if isinstance(best_configuration, Mapping) and best_configuration.get("positions") is not None:
        return best_configuration["positions"]
    if results_payload.get("positions") is not None:
        return results_payload["positions"]
    return None


def _extract_final_position(result: Mapping[str, Any]) -> Optional[Any]:
    """Return final AP position(s) from history or standardized result payload."""
    history = _extract_history(result)
    positions = history.get("positions")
    if isinstance(positions, Sequence) and len(positions) > 0:
        return positions[-1]

    results_payload = _extract_results_payload(result)
    final_configuration = results_payload.get("final_configuration")
    if isinstance(final_configuration, Mapping) and final_configuration.get("positions") is not None:
        return final_configuration["positions"]
    if results_payload.get("positions") is not None:
        return results_payload["positions"]
    return None


def _extract_best_direction(result: Mapping[str, Any]) -> Optional[Any]:
    """Return best direction(s) from history or standardized result payload."""
    history = _extract_history(result)
    directions = history.get("directions")
    best_iter = _extract_best_iteration(result)
    if isinstance(directions, Sequence) and len(directions) > 0:
        idx = best_iter if 0 <= best_iter < len(directions) else len(directions) - 1
        return directions[idx]

    results_payload = _extract_results_payload(result)
    best_configuration = results_payload.get("best_configuration")
    if isinstance(best_configuration, Mapping) and best_configuration.get("directions") is not None:
        return best_configuration["directions"]
    return None


def _extract_final_direction(result: Mapping[str, Any]) -> Optional[Any]:
    """Return final direction(s) from history or standardized result payload."""
    history = _extract_history(result)
    directions = history.get("directions")
    if isinstance(directions, Sequence) and len(directions) > 0:
        return directions[-1]

    results_payload = _extract_results_payload(result)
    final_configuration = results_payload.get("final_configuration")
    if isinstance(final_configuration, Mapping) and final_configuration.get("directions") is not None:
        return final_configuration["directions"]
    return None


def _extract_best_look_at(result: Mapping[str, Any]) -> Optional[Any]:
    """Return best look-at target(s) from history when available."""
    history = _extract_history(result)
    look_at_targets = history.get("look_at_targets")
    best_iter = _extract_best_iteration(result)
    if isinstance(look_at_targets, Sequence) and len(look_at_targets) > 0:
        idx = best_iter if 0 <= best_iter < len(look_at_targets) else len(look_at_targets) - 1
        return look_at_targets[idx]
    return None


def _extract_final_look_at(result: Mapping[str, Any]) -> Optional[Any]:
    """Return final look-at target(s) from history when available."""
    history = _extract_history(result)
    look_at_targets = history.get("look_at_targets")
    if isinstance(look_at_targets, Sequence) and len(look_at_targets) > 0:
        return look_at_targets[-1]
    return None


def _extract_reflector_position(result: Mapping[str, Any]) -> Optional[Any]:
    """Return reflector position from raw result payload when available."""
    snapshot = _extract_reflector_snapshot(result)
    if snapshot.get("position") is not None:
        return snapshot["position"]

    results_payload = _extract_results_payload(result)
    best_configuration = results_payload.get("best_configuration")
    if isinstance(best_configuration, Mapping):
        reflector = best_configuration.get("reflector")
        if isinstance(reflector, Mapping) and reflector.get("position") is not None:
            return reflector["position"]
    return results_payload.get("reflector_position")


def _extract_reflector_target(result: Mapping[str, Any]) -> Optional[Any]:
    """Return reflector target from raw result payload when available."""
    snapshot = _extract_reflector_snapshot(result)
    if snapshot.get("target") is not None:
        return snapshot["target"]

    results_payload = _extract_results_payload(result)
    best_configuration = results_payload.get("best_configuration")
    if isinstance(best_configuration, Mapping):
        reflector = best_configuration.get("reflector")
        if isinstance(reflector, Mapping) and reflector.get("target") is not None:
            return reflector["target"]
    return results_payload.get("reflector_target")


def _extract_best_physical_metrics(result: Mapping[str, Any]) -> Dict[str, float]:
    """Return the standardized physical metrics dictionary."""
    results_payload = _extract_results_payload(result)
    metrics = results_payload.get("physical_metrics")
    if isinstance(metrics, Mapping):
        return {str(name): float(value) for name, value in metrics.items() if value is not None}
    return {}


def _select_secondary_metric(result: Mapping[str, Any]) -> Tuple[str, List[float]]:
    """Choose one physical metric series for secondary trajectory plots."""
    for metric_name in _PHYSICAL_METRIC_PRIORITY:
        values = _extract_physical_metric_series(result, metric_name)
        if values:
            return metric_name, values
    return "physical_metric", []


def _fmt_dir(direction: Optional[list]) -> str:
    """Format one or more direction vectors for display."""
    if direction is None:
        return "N/A"

    if isinstance(direction, (list, tuple, np.ndarray)) and len(direction) > 0:
        if isinstance(direction[0], (list, tuple, np.ndarray)):
            parts = [f"({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f})" for d in direction]
            return " | ".join(parts)

    return f"({direction[0]:+.4f}, {direction[1]:+.4f}, {direction[2]:+.4f})"


def _fmt_pos(position: Optional[list]) -> str:
    """Format one or more positions for display."""
    if position is None:
        return "N/A"

    if isinstance(position, (list, tuple, np.ndarray)) and len(position) > 0:
        if isinstance(position[0], (list, tuple, np.ndarray)):
            parts = [f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" for p in position]
            return " | ".join(parts)

    return f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"


def save_ga_training_curve(ga_results: Mapping[str, Any], save_path: Path) -> Optional[str]:
    """Save GA primary-fitness curve with optional second-rank history."""
    details = ga_results.get("generation_details", [])
    if not isinstance(details, list) or len(details) == 0:
        return None

    generations: List[int] = []
    best_values: List[float] = []
    second_values: List[float] = []
    mean_values: List[float] = []

    for fallback_gen, row in enumerate(details):
        if not isinstance(row, Mapping):
            continue

        raw_gen = row.get("gen", fallback_gen)
        try:
            generation_index = int(raw_gen)
        except (TypeError, ValueError):
            generation_index = fallback_gen
        generations.append(generation_index)

        best_values.append(_coerce_optional_float(row.get("best_primary_fitness")))
        mean_values.append(_coerce_optional_float(row.get("mean_primary_fitness")))

        second_fitness = row.get("second_primary_fitness")
        top_individuals = row.get("top_individuals")
        if (
            second_fitness is None
            and isinstance(top_individuals, Sequence)
            and len(top_individuals) > 1
            and isinstance(top_individuals[1], Mapping)
        ):
            second_fitness = top_individuals[1].get("primary_fitness")
        second_values.append(_coerce_optional_float(second_fitness))

    if not generations:
        return None

    best_array = np.asarray(best_values, dtype=np.float64)
    mean_array = np.asarray(mean_values, dtype=np.float64)
    second_array = np.asarray(second_values, dtype=np.float64)
    has_best = bool(np.any(np.isfinite(best_array)))
    has_mean = bool(np.any(np.isfinite(mean_array)))
    has_second = bool(np.any(np.isfinite(second_array)))
    if not any((has_best, has_mean, has_second)):
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    if has_best:
        ax.plot(generations, best_values, marker="o", linewidth=2.0, label="GA Best")
    if has_second:
        ax.plot(generations, second_values, marker="^", linewidth=1.8, label="GA 2nd Best")
    if has_mean:
        ax.plot(generations, mean_values, marker="s", linewidth=1.6, label="GA Mean")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Primary Fitness")
    ax.set_title("Memetic Phase-1 Training Curve")
    _set_integer_x_ticks(ax)
    ax.grid(True, alpha=0.3)
    # legend at bottom right
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def _coerce_optional_float(value: Any) -> float:
    """Convert one value to float; return NaN when conversion fails."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _coerce_metric(metrics: Mapping[str, Any], metric_key: str) -> float:
    """Read one metric as float when possible, else NaN."""
    candidate = metrics.get(metric_key)
    if candidate is None:
        return float("nan")
    try:
        return float(candidate)
    except (TypeError, ValueError):
        return float("nan")


def _round_down_to_step(value: float, step: float) -> float:
    """Round one value down to the nearest multiple of ``step``."""
    return float(step * np.floor(float(value) / step))


def _round_up_to_step(value: float, step: float) -> float:
    """Round one value up to the nearest multiple of ``step``."""
    return float(step * np.ceil(float(value) / step))


def _collect_finite(values: Sequence[float]) -> List[float]:
    """Collect finite float values from one sequence."""
    return [float(value) for value in values if np.isfinite(value)]


def _compute_rssi_metric_y_limits(
    ga_results: Mapping[str, Any],
    gd_results: Optional[Mapping[str, Any]] = None,
    rounding_step: float = _RSSI_Y_AXIS_ROUND_STEP,
) -> Dict[str, Tuple[float, float]]:
    """Compute shared per-metric RSSI y-limits across all ranks.

    Limits are expanded to outer multiples of ``rounding_step`` so all plots for
    the same metric share a stable, easy-to-read range.
    """
    ranks = _resolve_ga_plot_ranks(ga_results)
    limits: Dict[str, Tuple[float, float]] = {}

    for (
        metric_key,
        priority_metric_key,
        _,
        _,
        _,
        _,
        _,
    ) in _GA_BEST_METRIC_PLOT_SPECS:
        collected: List[float] = []

        for rank in ranks:
            _, area_values, priority_values = _extract_ga_ranked_metric_series(
                ga_results=ga_results,
                metric_key=metric_key,
                priority_metric_key=priority_metric_key,
                rank=rank,
            )
            collected.extend(_collect_finite(area_values))
            collected.extend(_collect_finite(priority_values))

            if gd_results is not None:
                gd_area, gd_priority = _extract_gd_seed_metric_pair(
                    gd_results=gd_results,
                    seed_index=rank - 1,
                    metric_key=metric_key,
                    priority_metric_key=priority_metric_key,
                )
                use_seed_specific = bool(np.isfinite(gd_area) or np.isfinite(gd_priority))
                if not use_seed_specific:
                    gd_area, gd_priority = _extract_gd_best_metric_pair(
                        gd_results=gd_results,
                        metric_key=metric_key,
                        priority_metric_key=priority_metric_key,
                    )
                if np.isfinite(gd_area):
                    collected.append(float(gd_area))
                if np.isfinite(gd_priority):
                    collected.append(float(gd_priority))

        if not collected:
            continue

        raw_min = min(collected)
        raw_max = max(collected)
        ymin = _round_down_to_step(raw_min, rounding_step)
        ymax = _round_up_to_step(raw_max, rounding_step)
        if ymin == ymax:
            ymin -= rounding_step
            ymax += rounding_step

        limits[metric_key] = (ymin, ymax)

    return limits


def _resolve_ga_plot_ranks(ga_results: Mapping[str, Any]) -> List[int]:
    """Resolve GA rank indices to render based on available payloads."""
    max_rank = 1

    raw_num_selected = ga_results.get("num_selected_seeds")
    if raw_num_selected is not None:
        try:
            max_rank = max(max_rank, int(raw_num_selected))
        except (TypeError, ValueError):
            pass

    generation_details = ga_results.get("generation_details")
    if isinstance(generation_details, Sequence):
        for row in generation_details:
            if not isinstance(row, Mapping):
                continue

            top_individuals = row.get("top_individuals")
            if isinstance(top_individuals, Sequence):
                for fallback_rank, ranked in enumerate(top_individuals, start=1):
                    if not isinstance(ranked, Mapping):
                        continue
                    raw_rank = ranked.get("rank", fallback_rank)
                    try:
                        rank = int(raw_rank)
                    except (TypeError, ValueError):
                        rank = fallback_rank
                    max_rank = max(max_rank, rank)
            elif row.get("second_primary_fitness") is not None:
                max_rank = max(max_rank, 2)

    if max_rank <= 0:
        return [1]
    return list(range(1, max_rank + 1))


def _ranked_artifact_key(base_key: str, rank: int) -> str:
    """Build a rank-aware artifact key while preserving rank-1 keys."""
    if rank == 1:
        return base_key
    if base_key.startswith("ga_best_"):
        return base_key.replace("ga_best_", f"ga_rank{rank}_", 1)
    if base_key.startswith("ga_gd_stitched_"):
        return base_key.replace("ga_gd_stitched_", f"ga_gd_stitched_rank{rank}_", 1)
    return f"{base_key}_rank{rank}"


def _ranked_filename(base_filename: str, rank: int) -> str:
    """Build a rank-aware filename with explicit rank numbering."""
    if base_filename.startswith("ga_best_"):
        return base_filename.replace("ga_best_", f"ga_rank{rank}_", 1)
    if base_filename.startswith("ga_gd_stitched_"):
        return base_filename.replace("ga_gd_stitched_", f"ga_gd_stitched_rank{rank}_", 1)

    suffix = Path(base_filename).suffix
    stem = Path(base_filename).stem
    return f"{stem}_rank{rank}{suffix}"


def _extract_ga_ranked_metric_series(
    ga_results: Mapping[str, Any],
    metric_key: str,
    priority_metric_key: str,
    rank: int = 1,
) -> Tuple[List[int], List[float], List[float]]:
    """Extract one GA-rank metric series per generation for all/priority metrics."""
    if rank <= 0:
        return [], [], []

    generation_details = ga_results.get("generation_details")
    if not isinstance(generation_details, Sequence):
        return [], [], []

    generations: List[int] = []
    area_values: List[float] = []
    priority_values: List[float] = []

    for fallback_gen, row in enumerate(generation_details):
        if not isinstance(row, Mapping):
            continue

        raw_gen = row.get("gen", fallback_gen)
        try:
            generation_index = int(raw_gen)
        except (TypeError, ValueError):
            generation_index = fallback_gen

        area_metric = float("nan")
        priority_metric = float("nan")

        metrics: Optional[Mapping[str, Any]] = None
        top_individuals = row.get("top_individuals")
        if (
            isinstance(top_individuals, Sequence)
            and len(top_individuals) >= rank
            and isinstance(top_individuals[rank - 1], Mapping)
        ):
            ranked_metrics = top_individuals[rank - 1].get("physical_metrics")
            if isinstance(ranked_metrics, Mapping):
                metrics = ranked_metrics

        if metrics is None:
            if rank == 1:
                fallback_metrics = row.get("best_physical_metrics")
                if isinstance(fallback_metrics, Mapping):
                    metrics = fallback_metrics
            elif rank == 2:
                fallback_metrics = row.get("second_physical_metrics")
                if isinstance(fallback_metrics, Mapping):
                    metrics = fallback_metrics

        if isinstance(metrics, Mapping):
            area_metric = _coerce_metric(metrics, metric_key)
            priority_metric = _coerce_metric(metrics, priority_metric_key)

        generations.append(generation_index)
        area_values.append(area_metric)
        priority_values.append(priority_metric)

    return generations, area_values, priority_values


def _extract_ga_best_metric_series(
    ga_results: Mapping[str, Any],
    metric_key: str,
    priority_metric_key: str,
) -> Tuple[List[int], List[float], List[float]]:
    """Backward-compatible wrapper for rank-1 GA metric extraction."""
    return _extract_ga_ranked_metric_series(
        ga_results=ga_results,
        metric_key=metric_key,
        priority_metric_key=priority_metric_key,
        rank=1,
    )


def save_ga_generation_best_metric_trend_plot(
    ga_results: Mapping[str, Any],
    save_path: Path,
    metric_key: str,
    priority_metric_key: str,
    metric_label: str,
    rank: int = 1,
    y_limits: Optional[Tuple[float, float]] = None,
    gd_results: Optional[Mapping[str, Any]] = None,
    gd_seed_index: Optional[int] = None,
) -> Optional[str]:
    """Plot one GA rank metric trend for all-area vs priority-area."""
    generations, area_values, priority_values = _extract_ga_ranked_metric_series(
        ga_results=ga_results,
        metric_key=metric_key,
        priority_metric_key=priority_metric_key,
        rank=rank,
    )
    area_array = np.asarray(area_values, dtype=np.float64)
    priority_array = np.asarray(priority_values, dtype=np.float64)
    has_area = bool(np.any(np.isfinite(area_array)))
    has_priority = bool(np.any(np.isfinite(priority_array)))

    gd_area_value = float("nan")
    gd_priority_value = float("nan")
    use_seed_specific_label = False
    if isinstance(gd_results, Mapping):
        if gd_seed_index is not None:
            gd_area_value, gd_priority_value = _extract_gd_seed_metric_pair(
                gd_results=gd_results,
                seed_index=gd_seed_index,
                metric_key=metric_key,
                priority_metric_key=priority_metric_key,
            )
            use_seed_specific_label = bool(np.isfinite(gd_area_value) or np.isfinite(gd_priority_value))

        if not use_seed_specific_label:
            gd_area_value, gd_priority_value = _extract_gd_best_metric_pair(
                gd_results=gd_results,
                metric_key=metric_key,
                priority_metric_key=priority_metric_key,
            )

    has_gd_area = bool(np.isfinite(gd_area_value))
    has_gd_priority = bool(np.isfinite(gd_priority_value))

    if not any((has_area, has_priority, has_gd_area, has_gd_priority)):
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    if has_area:
        ax.plot(
            generations,
            area_values,
            marker="o",
            linewidth=2.0,
            label="All-area",
        )
    if has_priority:
        ax.plot(
            generations,
            priority_values,
            marker="s",
            linewidth=2.0,
            label="Priority-area",
        )

    gd_x = (max(generations) + 1) if generations else 0
    if has_gd_area:
        ax.scatter(
            [gd_x],
            [gd_area_value],
            marker="*",
            s=180,
            color="tab:blue",
            zorder=6,
            label="GD all-area",
        )
        if has_area and generations and np.isfinite(area_values[-1]):
            ax.plot([generations[-1], gd_x], [area_values[-1], gd_area_value], "--", color="tab:blue", alpha=0.65)

    if has_gd_priority:
        ax.scatter(
            [gd_x],
            [gd_priority_value],
            marker="*",
            s=180,
            color="tab:orange",
            zorder=6,
            label="GD priority-area",
        )
        if has_priority and generations and np.isfinite(priority_values[-1]):
            ax.plot(
                [generations[-1], gd_x],
                [priority_values[-1], gd_priority_value],
                "--",
                color="tab:orange",
                alpha=0.65,
            )

    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel(metric_label)
    # title_prefix = f"GA Rank-{rank} Individual"
    title_prefix = ""
    ax.set_title(f"{title_prefix} {metric_label} Trend")
    if y_limits is not None:
        ax.set_ylim(float(y_limits[0]), float(y_limits[1]))
    _set_integer_x_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def save_ga_generation_best_metric_trend_plots(
    ga_results: Mapping[str, Any],
    save_dir: Path,
    gd_results: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Save GA rank-aware metric-trend plots for key all/priority metrics."""
    save_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, str] = {"ga_best_metric_trend_dir": str(save_dir)}

    ranks = _resolve_ga_plot_ranks(ga_results)
    artifacts["ga_best_metric_ranks"] = ",".join(str(rank) for rank in ranks)
    y_limits_by_metric = _compute_rssi_metric_y_limits(
        ga_results=ga_results,
        gd_results=gd_results,
    )

    rendered_count = 0
    for rank in ranks:
        for (
            metric_key,
            priority_metric_key,
            metric_label,
            ga_artifact_key,
            ga_filename,
            _,
            _,
        ) in _GA_BEST_METRIC_PLOT_SPECS:
            rendered = save_ga_generation_best_metric_trend_plot(
                ga_results=ga_results,
                save_path=save_dir / _ranked_filename(ga_filename, rank),
                metric_key=metric_key,
                priority_metric_key=priority_metric_key,
                metric_label=metric_label,
                rank=rank,
                y_limits=y_limits_by_metric.get(metric_key),
                gd_results=gd_results,
                gd_seed_index=rank - 1,
            )
            if rendered is not None:
                rendered_count += 1
                artifacts[_ranked_artifact_key(ga_artifact_key, rank)] = rendered

    artifacts["ga_best_metric_trend_plot_count"] = str(rendered_count)
    return artifacts


def save_ga_generation_combined_plot(
    ga_results: Mapping[str, Any],
    save_path: Path,
    rank: int = 1,
    y_limits: Optional[Tuple[float, float]] = _DEFAULT_MEAN_P5_COMBINED_Y_RANGE,
    gd_results: Optional[Mapping[str, Any]] = None,
    gd_seed_index: Optional[int] = None,
) -> Optional[str]:
    """Plot one rank with mean, p5, and min RSSI trends in one combined chart."""
    metric_specs: Tuple[Tuple[str, str, str, str, str, str], ...] = (
        ("mean_rss_dbm", "priority_mean_rss_dbm", "Mean RSSI", "tab:blue", "o", "s"),
        ("p5_rss_dbm", "priority_p5_rss_dbm", "P5 RSSI", "tab:orange", "^", "D"),
        ("min_rss_dbm", "priority_min_rss_dbm", "Min RSSI", "tab:green", "v", "P"),
    )

    series_by_metric: Dict[str, Tuple[List[int], List[float], List[float]]] = {}
    has_area_by_metric: Dict[str, bool] = {}
    has_priority_by_metric: Dict[str, bool] = {}
    for metric_key, priority_metric_key, _, _, _, _ in metric_specs:
        generations, area_values, priority_values = _extract_ga_ranked_metric_series(
            ga_results=ga_results,
            metric_key=metric_key,
            priority_metric_key=priority_metric_key,
            rank=rank,
        )
        series_by_metric[metric_key] = (generations, area_values, priority_values)
        has_area_by_metric[metric_key] = bool(np.any(np.isfinite(np.asarray(area_values, dtype=np.float64))))
        has_priority_by_metric[metric_key] = bool(np.any(np.isfinite(np.asarray(priority_values, dtype=np.float64))))

    gd_by_metric: Dict[str, Tuple[float, float]] = {
        metric_key: (float("nan"), float("nan"))
        for metric_key, _, _, _, _, _ in metric_specs
    }
    use_seed_specific_label = False
    if isinstance(gd_results, Mapping):
        if gd_seed_index is not None:
            gd_by_metric = {}
            for metric_key, priority_metric_key, _, _, _, _ in metric_specs:
                gd_by_metric[metric_key] = _extract_gd_seed_metric_pair(
                    gd_results=gd_results,
                    seed_index=gd_seed_index,
                    metric_key=metric_key,
                    priority_metric_key=priority_metric_key,
                )
            use_seed_specific_label = any(
                np.isfinite(area_value) or np.isfinite(priority_value)
                for area_value, priority_value in gd_by_metric.values()
            )

        if not use_seed_specific_label:
            gd_by_metric = {}
            for metric_key, priority_metric_key, _, _, _, _ in metric_specs:
                gd_by_metric[metric_key] = _extract_gd_best_metric_pair(
                    gd_results=gd_results,
                    metric_key=metric_key,
                    priority_metric_key=priority_metric_key,
                )

    if not any(
        has_area_by_metric[metric_key]
        or has_priority_by_metric[metric_key]
        or np.isfinite(gd_by_metric[metric_key][0])
        or np.isfinite(gd_by_metric[metric_key][1])
        for metric_key, _, _, _, _, _ in metric_specs
    ):
        return None

    fig, ax = plt.subplots(figsize=(11, 5))

    for metric_key, _, display_label, color, area_marker, priority_marker in metric_specs:
        generations, area_values, priority_values = series_by_metric[metric_key]
        if has_area_by_metric[metric_key]:
            ax.plot(
                generations,
                area_values,
                marker=area_marker,
                linewidth=2.0,
                color=color,
                label=f"{display_label} (All-area)",
            )
        if has_priority_by_metric[metric_key]:
            ax.plot(
                generations,
                priority_values,
                marker=priority_marker,
                linewidth=2.0,
                linestyle="--",
                color=color,
                label=f"{display_label} (Priority-area)",
            )

    gd_x = max(
        max(generations) if generations else 0
        for generations, _, _ in series_by_metric.values()
    ) + 1

    finite_values: List[float] = []
    for metric_key, _, display_label, color, _, _ in metric_specs:
        generations, area_values, priority_values = series_by_metric[metric_key]
        finite_values.extend(_collect_finite(area_values))
        finite_values.extend(_collect_finite(priority_values))

        gd_area_value, gd_priority_value = gd_by_metric[metric_key]
        if np.isfinite(gd_area_value):
            finite_values.append(float(gd_area_value))
            ax.scatter(
                [gd_x],
                [gd_area_value],
                marker="*",
                s=180,
                color=color,
                zorder=6,
                label=f"GD {display_label} (All-area)",
            )
            if has_area_by_metric[metric_key] and generations and np.isfinite(area_values[-1]):
                ax.plot(
                    [generations[-1], gd_x],
                    [area_values[-1], gd_area_value],
                    "--",
                    color=color,
                    alpha=0.65,
                )

        if np.isfinite(gd_priority_value):
            finite_values.append(float(gd_priority_value))
            ax.scatter(
                [gd_x],
                [gd_priority_value],
                marker="*",
                s=180,
                color=color,
                zorder=6,
                label=f"GD {display_label} (Priority-area)",
            )
            if has_priority_by_metric[metric_key] and generations and np.isfinite(priority_values[-1]):
                ax.plot(
                    [generations[-1], gd_x],
                    [priority_values[-1], gd_priority_value],
                    "--",
                    color=color,
                    alpha=0.65,
                )

    if y_limits is None:
        if finite_values:
            ymin = _round_down_to_step(min(finite_values), _RSSI_Y_AXIS_ROUND_STEP)
            ymax = _round_up_to_step(max(finite_values), _RSSI_Y_AXIS_ROUND_STEP)
            ymin = min(ymin, _DEFAULT_MEAN_P5_COMBINED_Y_RANGE[0])
            ymax = max(ymax, _DEFAULT_MEAN_P5_COMBINED_Y_RANGE[1])
            if ymin == ymax:
                ymin -= _RSSI_Y_AXIS_ROUND_STEP
                ymax += _RSSI_Y_AXIS_ROUND_STEP
            resolved_y_limits = (ymin, ymax)
        else:
            resolved_y_limits = _DEFAULT_MEAN_P5_COMBINED_Y_RANGE
    else:
        resolved_y_limits = y_limits

    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel("RSSI (dBm)")
    ax.set_title("Mean, P5, and Min RSSI Trend")
    ax.set_ylim(float(resolved_y_limits[0]), float(resolved_y_limits[1]))
    _set_integer_x_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def save_ga_generation_combined_plots(
    ga_results: Mapping[str, Any],
    save_dir: Path,
    gd_results: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Save one combined mean+p5+min RSSI trend plot per GA rank."""
    save_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, str] = {"ga_combined_trend_dir": str(save_dir)}

    ranks = _resolve_ga_plot_ranks(ga_results)
    artifacts["ga_combined_ranks"] = ",".join(str(rank) for rank in ranks)

    rendered_count = 0
    base_artifact_key = "ga_combined_trend_plot"
    base_filename = "ga_combined_trend.png"
    for rank in ranks:
        rendered = save_ga_generation_combined_plot(
            ga_results=ga_results,
            save_path=save_dir / _ranked_filename(base_filename, rank),
            rank=rank,
            y_limits=None,
            gd_results=gd_results,
            gd_seed_index=rank - 1,
        )
        if rendered is not None:
            rendered_count += 1
            artifacts[_ranked_artifact_key(base_artifact_key, rank)] = rendered

    artifacts["ga_combined_trend_plot_count"] = str(rendered_count)
    return artifacts


def _extract_gd_best_metric_pair(
    gd_results: Mapping[str, Any],
    metric_key: str,
    priority_metric_key: str,
) -> Tuple[float, float]:
    """Extract best GD all-area and priority-area metric values."""
    best_result = gd_results.get("global_best_result")
    if not isinstance(best_result, Mapping):
        return float("nan"), float("nan")

    metrics = _extract_best_physical_metrics(best_result)
    if not metrics:
        return float("nan"), float("nan")

    return (
        _coerce_metric(metrics, metric_key),
        _coerce_metric(metrics, priority_metric_key),
    )


def _extract_gd_seed_metric_pair(
    gd_results: Mapping[str, Any],
    seed_index: int,
    metric_key: str,
    priority_metric_key: str,
) -> Tuple[float, float]:
    """Extract one seed-specific GD metric pair using per-seed analysis first."""
    if seed_index < 0:
        return float("nan"), float("nan")

    per_seed = gd_results.get("per_seed_analysis")
    if (
        isinstance(per_seed, Sequence)
        and len(per_seed) > seed_index
        and isinstance(per_seed[seed_index], Mapping)
    ):
        metrics = per_seed[seed_index].get("physical_metrics")
        if isinstance(metrics, Mapping):
            return (
                _coerce_metric(metrics, metric_key),
                _coerce_metric(metrics, priority_metric_key),
            )

    all_results = gd_results.get("all_fine_tuned_results")
    if isinstance(all_results, Sequence):
        task_result: Optional[Mapping[str, Any]] = None
        for candidate in all_results:
            if not isinstance(candidate, Mapping):
                continue
            raw_task_id = candidate.get("task_id")
            if raw_task_id is None:
                continue
            try:
                task_id = int(raw_task_id)
            except (TypeError, ValueError):
                continue
            if task_id == seed_index:
                task_result = candidate
                break

        if task_result is None:
            if len(all_results) > seed_index and isinstance(all_results[seed_index], Mapping):
                task_result = all_results[seed_index]

        if task_result is not None:
            metrics = _extract_best_physical_metrics(task_result)
            if metrics:
                return (
                    _coerce_metric(metrics, metric_key),
                    _coerce_metric(metrics, priority_metric_key),
                )

    return float("nan"), float("nan")


def save_ga_gd_stitched_metric_trend_plot(
    ga_results: Mapping[str, Any],
    gd_results: Mapping[str, Any],
    save_path: Path,
    metric_key: str,
    priority_metric_key: str,
    metric_label: str,
    rank: int = 1,
    gd_seed_index: Optional[int] = None,
    y_limits: Optional[Tuple[float, float]] = None,
) -> Optional[str]:
    """Plot one GA-rank trend stitched with one final GD metric point."""
    generations, area_values, priority_values = _extract_ga_ranked_metric_series(
        ga_results=ga_results,
        metric_key=metric_key,
        priority_metric_key=priority_metric_key,
        rank=rank,
    )

    gd_area_value = float("nan")
    gd_priority_value = float("nan")
    use_seed_specific_label = False
    if gd_seed_index is not None:
        gd_area_value, gd_priority_value = _extract_gd_seed_metric_pair(
            gd_results=gd_results,
            seed_index=gd_seed_index,
            metric_key=metric_key,
            priority_metric_key=priority_metric_key,
        )
        use_seed_specific_label = bool(
            np.isfinite(gd_area_value) or np.isfinite(gd_priority_value)
        )

    if not use_seed_specific_label:
        gd_area_value, gd_priority_value = _extract_gd_best_metric_pair(
            gd_results=gd_results,
            metric_key=metric_key,
            priority_metric_key=priority_metric_key,
        )

    area_array = np.asarray(area_values, dtype=np.float64)
    priority_array = np.asarray(priority_values, dtype=np.float64)
    has_ga_area = bool(generations) and bool(np.any(np.isfinite(area_array)))
    has_ga_priority = bool(generations) and bool(np.any(np.isfinite(priority_array)))
    has_gd_area = bool(np.isfinite(gd_area_value))
    has_gd_priority = bool(np.isfinite(gd_priority_value))
    if not any((has_ga_area, has_ga_priority, has_gd_area, has_gd_priority)):
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    if has_ga_area:
        ax.plot(
            generations,
            area_values,
            marker="o",
            linewidth=2.0,
            label="GA all-area",
        )
    if has_ga_priority:
        ax.plot(
            generations,
            priority_values,
            marker="s",
            linewidth=2.0,
            label="GA priority-area",
        )

    gd_x = (max(generations) + 1) if generations else 0
    if has_gd_area:
        ax.scatter(
            [gd_x],
            [gd_area_value],
            marker="*",
            s=180,
            color="tab:blue",
            zorder=6,
            label="GD all-area",
        )
        if has_ga_area and np.isfinite(area_values[-1]):
            ax.plot([generations[-1], gd_x], [area_values[-1], gd_area_value], "--", color="tab:blue", alpha=0.65)
    if has_gd_priority:
        ax.scatter(
            [gd_x],
            [gd_priority_value],
            marker="*",
            s=180,
            color="tab:orange",
            zorder=6,
            label="GD priority-area",
        )
        if has_ga_priority and np.isfinite(priority_values[-1]):
            ax.plot([generations[-1], gd_x], [priority_values[-1], gd_priority_value], "--", color="tab:orange", alpha=0.65)

    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel(metric_label)
    # title_prefix = f"GA Rank-{rank} to GD Stitched"
    title_prefix = ""
    ax.set_title(f"{title_prefix} {metric_label} Trend")
    if y_limits is not None:
        ax.set_ylim(float(y_limits[0]), float(y_limits[1]))
    _set_integer_x_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def save_ga_gd_stitched_metric_trend_plots(
    ga_results: Mapping[str, Any],
    gd_results: Mapping[str, Any],
    save_dir: Path,
) -> Dict[str, str]:
    """Save stitched GA-rank and GD endpoint plots for key metrics."""
    save_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, str] = {"ga_gd_stitched_metric_trend_dir": str(save_dir)}

    ranks = _resolve_ga_plot_ranks(ga_results)
    artifacts["ga_gd_stitched_metric_ranks"] = ",".join(str(rank) for rank in ranks)
    y_limits_by_metric = _compute_rssi_metric_y_limits(
        ga_results=ga_results,
        gd_results=gd_results,
    )

    rendered_count = 0
    for rank in ranks:
        for (
            metric_key,
            priority_metric_key,
            metric_label,
            _,
            _,
            stitched_artifact_key,
            stitched_filename,
        ) in _GA_BEST_METRIC_PLOT_SPECS:
            rendered = save_ga_gd_stitched_metric_trend_plot(
                ga_results=ga_results,
                gd_results=gd_results,
                save_path=save_dir / _ranked_filename(stitched_filename, rank),
                metric_key=metric_key,
                priority_metric_key=priority_metric_key,
                metric_label=metric_label,
                rank=rank,
                gd_seed_index=rank - 1,
                y_limits=y_limits_by_metric.get(metric_key),
            )
            if rendered is not None:
                rendered_count += 1
                artifacts[_ranked_artifact_key(stitched_artifact_key, rank)] = rendered

    artifacts["ga_gd_stitched_metric_trend_plot_count"] = str(rendered_count)
    return artifacts


def save_gd_seed_improvements(gd_results: Mapping[str, Any], save_path: Path) -> Optional[str]:
    """Save per-seed baseline vs refined primary losses and reductions."""
    analysis = gd_results.get("per_seed_analysis", [])
    if not isinstance(analysis, list) or len(analysis) == 0:
        return None

    seed_ids: List[int] = []
    initial_values: List[float] = []
    final_values: List[float] = []
    deltas: List[float] = []

    for row in analysis:
        initial_loss = row.get("initial_primary_loss")
        best_loss = row.get("best_primary_loss", row.get("final_primary_loss"))
        delta_loss = row.get("delta_best_loss", row.get("delta_loss"))
        if initial_loss is None or best_loss is None or delta_loss is None:
            continue
        seed_ids.append(int(row.get("seed_index", len(seed_ids))))
        initial_values.append(float(initial_loss))
        final_values.append(float(best_loss))
        deltas.append(float(delta_loss))

    if not seed_ids:
        return None

    x = list(range(len(seed_ids)))
    width = 0.36
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    ax1.bar([i - width / 2 for i in x], initial_values, width=width, label="Initial")
    ax1.bar([i + width / 2 for i in x], final_values, width=width, label="Best GD")
    ax1.set_ylabel("Primary Loss")
    ax1.set_title("Memetic Phase-3 Refinement by Seed")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend()

    bars = ax2.bar(x, deltas, width=0.55, color="tab:green")
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.set_xlabel("Seed Index")
    ax2.set_ylabel("Loss Reduction")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(seed_id) for seed_id in seed_ids])
    ax2.grid(True, axis="y", alpha=0.25)

    for bar, delta in zip(bars, deltas):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{delta:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def save_phase_timing_plot(timings: Mapping[str, Any], save_path: Path) -> str:
    """Save runtime breakdown plot for GA, GD, and total wall clock time."""
    labels = ["GA", "GD", "Total"]
    values = [
        float(timings.get("ga_duration_sec", 0.0)),
        float(timings.get("gd_duration_sec", 0.0)),
        float(timings.get("total_duration_sec", 0.0)),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:purple"])
    ax.set_ylabel("Seconds")
    ax.set_title("Memetic Pipeline Runtime Breakdown")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def save_gd_parallel_summary_plot(
    gd_results: Mapping[str, Any],
    save_path: Path,
    position_bounds: Optional[Mapping[str, Any]] = None,
    rss_range_dbm: Optional[Tuple[float, float]] = None,
) -> Optional[str]:
    """Save a parallel GD summary plot derived from standardized task outputs."""
    del rss_range_dbm

    all_results = gd_results.get("all_fine_tuned_results", [])
    if not isinstance(all_results, list) or len(all_results) == 0:
        return None

    best_result = gd_results.get("global_best_result")
    if not isinstance(best_result, Mapping):
        return None

    primary_losses = [
        value for value in (_extract_best_primary_loss(result) for result in all_results)
        if value is not None
    ]
    if not primary_losses:
        return None

    best_primary_loss = _extract_best_primary_loss(best_result)
    if best_primary_loss is None:
        return None

    best_position = _extract_best_position(best_result)
    if best_position is None:
        return None

    metadata = gd_results.get("parallel_run_metadata", {})
    aggregate_stats = dict(metadata.get("aggregate_stats", {})) if isinstance(metadata, Mapping) else {}
    metric_stats = {
        "mean_primary_loss": float(np.mean(primary_losses)),
        "std_primary_loss": float(np.std(primary_losses)),
        "min_primary_loss": float(np.min(primary_losses)),
        "max_primary_loss": float(np.max(primary_losses)),
    }
    for key, value in metric_stats.items():
        aggregate_stats.setdefault(key, value)

    pool_info = dict(metadata.get("pool_info", {})) if isinstance(metadata, Mapping) else {}
    pool_info.setdefault("num_tasks", len(all_results))

    best_reflector_pos = _extract_reflector_position(best_result)
    best_reflector_target = _extract_reflector_target(best_result)
    best_physical_metrics = _extract_best_physical_metrics(best_result)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(
        primary_losses,
        bins=max(5, len(all_results) // 2),
        edgecolor="black",
        alpha=0.7,
    )
    ax.axvline(best_primary_loss, color="red", linestyle="--", linewidth=2, label=f"Best: {best_primary_loss:.6f}")
    ax.set_xlabel("Primary Loss")
    ax.set_ylabel("Number of Tasks")
    ax.set_title("Distribution of Primary Loss Across GD Tasks")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    sample_pos = best_position
    is_multi_ap = (
        isinstance(sample_pos, (list, tuple, np.ndarray))
        and len(sample_pos) > 0
        and isinstance(sample_pos[0], (list, tuple, np.ndarray))
    )

    if is_multi_ap:
        n_aps = len(sample_pos)
        for ap_idx in range(n_aps):
            ap_positions = np.array([
                _extract_best_position(result)[ap_idx]
                for result in all_results
                if _extract_best_position(result) is not None
            ])
            scatter = ax.scatter(
                ap_positions[:, 0],
                ap_positions[:, 1],
                c=primary_losses,
                s=80,
                cmap="viridis_r",
                edgecolor="black",
                alpha=0.7,
                marker=_AP_MARKERS[ap_idx % len(_AP_MARKERS)],
                label=f"AP{ap_idx}",
            )

        for result in all_results:
            pos = _extract_best_position(result)
            if pos is None:
                continue
            pos_arr = np.array(pos)
            ax.plot(pos_arr[:, 0], pos_arr[:, 1], "k-", alpha=0.15, linewidth=0.5)

        best_pos_arr = np.array(best_position)
        for ap_idx in range(n_aps):
            ax.plot(best_pos_arr[ap_idx, 0], best_pos_arr[ap_idx, 1], "r*", markersize=18, zorder=6)

        best_direction = _extract_best_direction(best_result)
        if best_direction is not None:
            best_dir_arr = np.array(best_direction)
            if best_dir_arr.ndim == 2:
                for ap_idx in range(min(n_aps, len(best_dir_arr))):
                    ax.annotate(
                        "",
                        xy=(
                            best_pos_arr[ap_idx, 0] + best_dir_arr[ap_idx, 0] * 2.5,
                            best_pos_arr[ap_idx, 1] + best_dir_arr[ap_idx, 1] * 2.5,
                        ),
                        xytext=(best_pos_arr[ap_idx, 0], best_pos_arr[ap_idx, 1]),
                        arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
                        zorder=7,
                    )
    else:
        positions = np.array([
            _extract_best_position(result)
            for result in all_results
            if _extract_best_position(result) is not None
        ])
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=primary_losses,
            s=100,
            cmap="viridis_r",
            edgecolor="black",
            alpha=0.7,
        )
        ax.plot(best_position[0], best_position[1], "r*", markersize=20, label="Best")

        best_direction = _extract_best_direction(best_result)
        if best_direction is not None:
            best_dir_arr = np.array(best_direction)
            ax.annotate(
                "",
                xy=(best_position[0] + best_dir_arr[0] * 2.5, best_position[1] + best_dir_arr[1] * 2.5),
                xytext=(best_position[0], best_position[1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
                zorder=7,
            )

    if best_reflector_pos is not None:
        rp = np.asarray(best_reflector_pos)
        ax.plot(rp[0], rp[1], marker="X", color="magenta", markersize=14, markeredgecolor="black", label="Reflector", zorder=8)
    if best_reflector_target is not None:
        rt = np.asarray(best_reflector_target)
        ax.plot(rt[0], rt[1], marker="P", color="orange", markersize=13, markeredgecolor="black", label="Target", zorder=8)
    if best_reflector_pos is not None and best_reflector_target is not None:
        rp = np.asarray(best_reflector_pos)
        rt = np.asarray(best_reflector_target)
        ax.plot([rp[0], rt[0]], [rp[1], rt[1]], "--", color="magenta", linewidth=1.5, alpha=0.8, label="Reflector→Target", zorder=7)

    if position_bounds:
        ax.set_xlim(position_bounds["x_min"], position_bounds["x_max"])
        ax.set_ylim(position_bounds["y_min"], position_bounds["y_max"])
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Best GD Positions (color = primary loss)")
    plt.colorbar(scatter, ax=ax, label="Primary Loss")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1, 0]
    task_ids = [int(result.get("task_id", idx)) for idx, result in enumerate(all_results)]
    times = [float(result.get("time_elapsed", 0.0)) for result in all_results]
    sorted_pairs = sorted(zip(task_ids, times))
    sorted_ids, sorted_times = zip(*sorted_pairs) if sorted_pairs else ([], [])
    ax.bar(sorted_ids, sorted_times, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Task ID")
    ax.set_ylabel("Time (s)")
    ax.set_title("GD Time per Task")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        f"MEMETIC GD SUMMARY\n\n"
        f"Pool: {pool_info.get('num_workers', '?')} workers, {pool_info.get('num_tasks', len(all_results))} tasks\n\n"
        f"Best Task: #{best_result.get('task_id')} (Worker #{best_result.get('worker_id')})\n"
        f"Best AP Position(s): {_fmt_pos(best_position)}\n"
        f"Best AP Direction(s): {_fmt_dir(_extract_best_direction(best_result))}\n"
        f"Best Reflector Position: {_fmt_pos(best_reflector_pos)}\n"
        f"Best Reflector Target: {_fmt_pos(best_reflector_target)}\n"
        f"Best Primary Loss: {best_primary_loss:.6f}\n"
        f"\nPrimary Loss Statistics:\n"
        f"  Mean: {aggregate_stats['mean_primary_loss']:.6f} +/- {aggregate_stats['std_primary_loss']:.6f}\n"
        f"  Range: [{aggregate_stats['min_primary_loss']:.6f}, {aggregate_stats['max_primary_loss']:.6f}]\n"
    )
    if best_physical_metrics:
        summary += "\nBest Physical Metrics:\n"
        for key, value in best_physical_metrics.items():
            summary += f"  {key}: {value:.6f}\n"
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

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def save_gd_trajectory_plots(
    gd_results: Mapping[str, Any],
    save_dir: Path,
    filename_prefix: str = "gd_task",
    position_bounds: Optional[Mapping[str, Any]] = None,
    rss_range_dbm: Optional[Tuple[float, float]] = None,
) -> List[str]:
    """Save per-task GD trajectory plots from raw history payloads."""
    del rss_range_dbm

    all_results = gd_results.get("all_fine_tuned_results", [])
    if not isinstance(all_results, list):
        return []

    save_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    for result in all_results:
        history = _extract_history(result)
        positions = history.get("positions")
        if not isinstance(positions, Sequence) or len(positions) == 0:
            continue

        task_id = int(result.get("task_id", len(saved_paths)))
        positions_arr = np.array(positions)
        if positions_arr.ndim == 2:
            positions_arr = positions_arr[:, np.newaxis, :]
        num_aps = positions_arr.shape[1]

        directions_arr = np.array(history.get("directions", [])) if history.get("directions") else None
        if directions_arr is not None and directions_arr.ndim == 2:
            directions_arr = directions_arr[:, np.newaxis, :]

        primary_loss_values = _extract_primary_loss_series(result)
        coverage_values = _extract_physical_metric_series(result, "coverage_pct")
        secondary_metric_name, secondary_metric_values = _select_secondary_metric(result)
        priority_mean_values = _extract_priority_metric_series(result, "priority_mean_rss_dbm")
        priority_min_values = _extract_priority_metric_series(result, "priority_min_rss_dbm")
        priority_p5_values = _extract_priority_metric_series(result, "priority_p5_rss_dbm")
        spatial_priority_map = _extract_spatial_priority_map(result)
        gradients = history.get("gradients", [])
        best_iter = _extract_best_iteration(result)

        orientation_lines = ""
        best_dir = _extract_best_direction(result)
        final_dir = _extract_final_direction(result)
        best_look_at = _extract_best_look_at(result)
        final_look_at = _extract_final_look_at(result)
        if best_dir is not None:
            orientation_lines += f"\nBest Dir: {_fmt_dir(best_dir)}"
        if best_look_at is not None:
            orientation_lines += f"  LookAt: {_fmt_pos(best_look_at)}"
        if final_dir is not None:
            orientation_lines += f"\nFinal Dir: {_fmt_dir(final_dir)}"
        if final_look_at is not None:
            orientation_lines += f"  LookAt: {_fmt_pos(final_look_at)}"

        best_primary_loss = _extract_best_primary_loss(result)
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        fig.suptitle(
            f"Task #{task_id} — Gradient Descent Trajectory\n"
            f"Best Primary Loss: {best_primary_loss:.6f} at iteration {best_iter + 1 if best_iter >= 0 else 'N/A'}"
            f"{orientation_lines}",
            fontsize=11,
            fontweight="bold",
        )

        ax = axes[0, 0]
        for ap_idx in range(num_aps):
            color = _AP_COLORS[ap_idx % len(_AP_COLORS)]
            marker = _AP_MARKERS[ap_idx % len(_AP_MARKERS)]
            label_prefix = f"AP{ap_idx} " if num_aps > 1 else ""
            ap_positions = positions_arr[:, ap_idx, :]
            ax.plot(ap_positions[:, 0], ap_positions[:, 1], f"-{marker}", color=color, markersize=4, linewidth=1.5, alpha=0.6, label=f"{label_prefix}path")
            ax.plot(ap_positions[0, 0], ap_positions[0, 1], marker, color="green", markersize=12, zorder=5, label=f"{label_prefix}Start" if ap_idx == 0 else None)
            ax.plot(ap_positions[-1, 0], ap_positions[-1, 1], "s", color=color, markersize=12, zorder=5, label=f"{label_prefix}End")
            if 0 <= best_iter < len(ap_positions):
                ax.plot(ap_positions[best_iter, 0], ap_positions[best_iter, 1], "*", color=color, markersize=18, zorder=6, label=f"{label_prefix}Best (iter {best_iter + 1})" if ap_idx == 0 else None)

            if directions_arr is not None and len(directions_arr) == len(positions_arr):
                ap_directions = directions_arr[:, ap_idx, :]
                ax.annotate(
                    "",
                    xy=(ap_positions[0, 0] + ap_directions[0, 0] * 2.0, ap_positions[0, 1] + ap_directions[0, 1] * 2.0),
                    xytext=(ap_positions[0, 0], ap_positions[0, 1]),
                    arrowprops=dict(arrowstyle="->", color="green", lw=2),
                )
                ax.annotate(
                    "",
                    xy=(ap_positions[-1, 0] + ap_directions[-1, 0] * 2.0, ap_positions[-1, 1] + ap_directions[-1, 1] * 2.0),
                    xytext=(ap_positions[-1, 0], ap_positions[-1, 1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2),
                )

        if position_bounds:
            ax.set_xlim(position_bounds["x_min"], position_bounds["x_max"])
            ax.set_ylim(position_bounds["y_min"], position_bounds["y_max"])
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("AP Position Trajectories")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        ax = axes[0, 1]
        if primary_loss_values:
            iterations = list(range(1, len(primary_loss_values) + 1))
            ax.plot(iterations, primary_loss_values, "b-", linewidth=2)
            if 0 <= best_iter < len(primary_loss_values):
                ax.axvline(best_iter + 1, color="red", linestyle="--", alpha=0.7, label=f"Best iter {best_iter + 1}")
                ax.plot(best_iter + 1, primary_loss_values[best_iter], "r*", markersize=15, zorder=5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Primary Loss")
        ax.set_title("Primary Loss Evolution")
        _set_integer_x_ticks(ax)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if coverage_values:
            iterations = list(range(1, len(coverage_values) + 1))
            ax.plot(iterations, coverage_values, "g-", linewidth=2, label="coverage_pct")
            if 0 <= best_iter < len(coverage_values):
                ax.axvline(best_iter + 1, color="red", linestyle="--", alpha=0.7, label=f"Best iter {best_iter + 1}")
        elif secondary_metric_values:
            iterations = list(range(1, len(secondary_metric_values) + 1))
            ax.plot(iterations, secondary_metric_values, "g-", linewidth=2, label=secondary_metric_name)
            if 0 <= best_iter < len(secondary_metric_values):
                ax.axvline(best_iter + 1, color="red", linestyle="--", alpha=0.7, label=f"Best iter {best_iter + 1}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("coverage_pct")
        ax.set_title("Coverage Evolution")
        _set_integer_x_ticks(ax)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if gradients:
            grad_arr = np.array(gradients)
            if grad_arr.ndim == 3:
                grad_norms = [float(np.sqrt(np.sum(np.array(grad) ** 2))) for grad in gradients]
            else:
                grad_norms = [float(np.sqrt(grad[0] ** 2 + grad[1] ** 2)) for grad in gradients]
            iterations = list(range(1, len(grad_norms) + 1))
            ax.semilogy(iterations, grad_norms, "r-", linewidth=2)
            if 0 <= best_iter < len(grad_norms):
                ax.axvline(best_iter + 1, color="red", linestyle="--", alpha=0.7, label=f"Best iter {best_iter + 1}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm Evolution (log scale)")
        _set_integer_x_ticks(ax)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[2, 0]
        any_priority_series = False
        if priority_mean_values:
            any_priority_series = True
            ax.plot(range(1, len(priority_mean_values) + 1), priority_mean_values, "b-", linewidth=2, label="priority_mean_rss_dbm")
        if priority_min_values:
            any_priority_series = True
            ax.plot(range(1, len(priority_min_values) + 1), priority_min_values, "m-", linewidth=2, label="priority_min_rss_dbm")
        if priority_p5_values:
            any_priority_series = True
            ax.plot(range(1, len(priority_p5_values) + 1), priority_p5_values, "c-", linewidth=2, label="priority_p5_rss_dbm")
        if any_priority_series and 0 <= best_iter < max(len(priority_mean_values), len(priority_min_values), len(priority_p5_values)):
            ax.axvline(best_iter + 1, color="red", linestyle="--", alpha=0.7, label=f"Best iter {best_iter + 1}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RSSI (dBm)")
        ax.set_title("Priority Metrics Evolution")
        _set_integer_x_ticks(ax)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[2, 1]
        if spatial_priority_map is not None:
            if position_bounds:
                extent = (
                    float(position_bounds["x_min"]),
                    float(position_bounds["x_max"]),
                    float(position_bounds["y_min"]),
                    float(position_bounds["y_max"]),
                )
            else:
                extent = None

            image = ax.imshow(
                spatial_priority_map,
                cmap="magma",
                origin="lower",
                extent=extent,
                aspect="equal",
            )
            ax.set_title("Priority Map Heatmap")
            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Y Position (m)")
            if position_bounds:
                ax.set_xlim(position_bounds["x_min"], position_bounds["x_max"])
                ax.set_ylim(position_bounds["y_min"], position_bounds["y_max"])
            ax.grid(False)
            # Keep heatmap panel size comparable to other subplots.
            cax = ax.inset_axes([1.02, 0.0, 0.035, 1.0])
            cbar = fig.colorbar(image, cax=cax)
            cbar.set_label("Priority")
        else:
            ax.axis("off")
            note_lines = [
                f"Task #{task_id}",
                "Priority metrics computed on explicit priority area only.",
                "Series: priority_mean_rss_dbm, priority_min_rss_dbm, priority_p5_rss_dbm.",
                "Priority map unavailable in this artifact.",
            ]
            if not any_priority_series:
                note_lines.append("No priority metric history available for this task.")
            ax.text(
                0.02,
                0.95,
                "\n".join(note_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
            )

        fig.tight_layout()
        save_path = save_dir / f"{filename_prefix}_{task_id}_trajectory.png"
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(save_path))

    return saved_paths


def save_memetic_plots(
    summary: Mapping[str, Any],
    output_dir: Path,
    position_bounds: Optional[Mapping[str, Any]] = None,
    rss_range_dbm: Tuple[float, float] = (-130.0, -80.0),
) -> Dict[str, str]:
    """Save the full memetic plot set and return generated artifact paths."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, str] = {"plots_dir": str(plots_dir)}
    ga_plot = save_ga_training_curve(summary.get("ga_results", {}), plots_dir / "ga_training_curve.png")
    gd_plot = save_gd_seed_improvements(summary.get("gd_results", {}), plots_dir / "gd_seed_improvements.png")
    timing_plot = save_phase_timing_plot(summary.get("timings", {}), plots_dir / "pipeline_timing_breakdown.png")
    gd_summary_plot = save_gd_parallel_summary_plot(
        summary.get("gd_results", {}),
        plots_dir / "gd_parallel_summary.png",
        position_bounds=position_bounds,
        rss_range_dbm=rss_range_dbm,
    )
    gd_trajectory_paths = save_gd_trajectory_plots(
        summary.get("gd_results", {}),
        plots_dir / "gd_trajectories",
        filename_prefix="gd_task",
        position_bounds=position_bounds,
        rss_range_dbm=rss_range_dbm,
    )
    ga_metric_trend_artifacts = save_ga_generation_best_metric_trend_plots(
        ga_results=summary.get("ga_results", {}),
        save_dir=plots_dir / "ga_generation_metric_trends",
        gd_results=summary.get("gd_results", {}),
    )
    ga_gd_stitched_metric_trend_artifacts = save_ga_gd_stitched_metric_trend_plots(
        ga_results=summary.get("ga_results", {}),
        gd_results=summary.get("gd_results", {}),
        save_dir=plots_dir / "ga_gd_stitched_metric_trends",
    )
    ga_combined_artifacts = save_ga_generation_combined_plots(
        ga_results=summary.get("ga_results", {}),
        save_dir=plots_dir / "ga_combined_trends",
        gd_results=summary.get("gd_results", {}),
    )

    if ga_plot is not None:
        artifacts["ga_training_plot"] = ga_plot
    if gd_plot is not None:
        artifacts["gd_seed_plot"] = gd_plot
    artifacts["timing_plot"] = timing_plot
    if gd_summary_plot is not None:
        artifacts["gd_parallel_summary_plot"] = gd_summary_plot
    artifacts["gd_trajectory_dir"] = str(plots_dir / "gd_trajectories")
    artifacts["gd_trajectory_count"] = str(len(gd_trajectory_paths))
    artifacts.update(ga_metric_trend_artifacts)
    artifacts.update(ga_gd_stitched_metric_trend_artifacts)
    artifacts.update(ga_combined_artifacts)
    return artifacts


def _as_positions_3d(raw: Any) -> Optional[List[List[float]]]:
    """Normalize raw position payloads to a list of 3D positions."""
    if raw is None or not isinstance(raw, Sequence) or len(raw) == 0:
        return None

    first = raw[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return [[float(p[0]), float(p[1]), float(p[2])] for p in raw]

    if len(raw) >= 3:
        return [[float(raw[0]), float(raw[1]), float(raw[2])]]
    return None


def _as_positions_3d_with_fixed_z(raw: Any, fixed_z: float) -> Optional[List[List[float]]]:
    """Normalize position payloads that may be 2D XY with a shared fixed Z."""
    if raw is None or not isinstance(raw, Sequence) or len(raw) == 0:
        return None

    first = raw[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        output: List[List[float]] = []
        for position in raw:
            if len(position) >= 3:
                output.append([float(position[0]), float(position[1]), float(position[2])])
            elif len(position) >= 2:
                output.append([float(position[0]), float(position[1]), float(fixed_z)])
        return output or None

    if len(raw) >= 3:
        return [[float(raw[0]), float(raw[1]), float(raw[2])]]
    if len(raw) >= 2:
        return [[float(raw[0]), float(raw[1]), float(fixed_z)]]
    return None


def _as_directions_3d(raw: Any) -> Optional[List[List[float]]]:
    """Normalize raw direction payloads to a list of 3D vectors."""
    if raw is None or not isinstance(raw, Sequence) or len(raw) == 0:
        return None

    first = raw[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        output: List[List[float]] = []
        for direction in raw:
            if len(direction) >= 3:
                output.append([float(direction[0]), float(direction[1]), float(direction[2])])
            elif len(direction) >= 2:
                output.append([float(direction[0]), float(direction[1]), 0.0])
        return output or None

    if len(raw) >= 3:
        return [[float(raw[0]), float(raw[1]), float(raw[2])]]
    if len(raw) >= 2:
        return [[float(raw[0]), float(raw[1]), 0.0]]
    return None


def _extract_reflector_state(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract reflector state in a uniform form: ``u``, ``v``, ``target``."""
    reflector = payload.get("reflector")
    if isinstance(reflector, Mapping):
        if {
            "u",
            "v",
            "focal_x",
            "focal_y",
            "focal_z",
        }.issubset(reflector.keys()):
            return {
                "u": float(reflector["u"]),
                "v": float(reflector["v"]),
                "target": [
                    float(reflector["focal_x"]),
                    float(reflector["focal_y"]),
                    float(reflector["focal_z"]),
                ],
            }
        if {"u", "v", "target"}.issubset(reflector.keys()):
            target = reflector["target"]
            if isinstance(target, Sequence) and len(target) >= 3:
                return {
                    "u": float(reflector["u"]),
                    "v": float(reflector["v"]),
                    "target": [float(target[0]), float(target[1]), float(target[2])],
                }

    if {"reflector_u", "reflector_v", "reflector_target"}.issubset(payload.keys()):
        target = payload["reflector_target"]
        if isinstance(target, Sequence) and len(target) >= 3:
            return {
                "u": float(payload["reflector_u"]),
                "v": float(payload["reflector_v"]),
                "target": [float(target[0]), float(target[1]), float(target[2])],
            }

    return None


def _build_ga_snapshots(summary: Mapping[str, Any], scene_config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build GA initial/best/final snapshots for coverage-map rendering."""
    ga_results = summary.get("ga_results", {})
    if not isinstance(ga_results, Mapping):
        return {}

    initial_positions = _as_positions_3d(scene_config.get("tx_positions"))
    initial_reflector = None
    focal_point = scene_config.get("focal_point")
    if isinstance(focal_point, Sequence) and len(focal_point) >= 3:
        initial_reflector = {
            "u": 0.5,
            "v": 0.5,
            "target": [float(focal_point[0]), float(focal_point[1]), float(focal_point[2])],
        }

    snapshots: Dict[str, Dict[str, Any]] = {}
    if initial_positions is not None:
        snapshots["initial"] = {
            "positions": initial_positions,
            "directions": None,
            "reflector": initial_reflector,
        }

    best_entry: Optional[Mapping[str, Any]] = None
    hall = ga_results.get("hall_of_fame")
    if isinstance(hall, Sequence) and len(hall) > 0 and isinstance(hall[0], Mapping):
        best_entry = hall[0]
    if best_entry is None:
        seeds = ga_results.get("seeds")
        if isinstance(seeds, Sequence) and len(seeds) > 0 and isinstance(seeds[0], Mapping):
            best_entry = seeds[0]

    if best_entry is not None:
        best_positions = _as_positions_3d(best_entry.get("ap_positions", best_entry.get("positions")))
        best_directions = _as_directions_3d(best_entry.get("ap_directions", best_entry.get("directions")))
        best_reflector = _extract_reflector_state(best_entry)
        if best_positions is not None:
            snapshots["best"] = {
                "positions": best_positions,
                "directions": best_directions,
                "reflector": best_reflector,
            }

            # GA exposes one final chosen configuration in this pipeline.
            snapshots["final"] = {
                "positions": best_positions,
                "directions": best_directions,
                "reflector": best_reflector,
            }

    return snapshots


def _build_ga_generation_best_snapshots(
    summary: Mapping[str, Any],
) -> List[Tuple[int, Dict[str, Any]]]:
    """Build per-generation GA-best snapshots for optional coverage rendering."""
    ga_results = summary.get("ga_results", {})
    if not isinstance(ga_results, Mapping):
        return []

    generation_details = ga_results.get("generation_details")
    if not isinstance(generation_details, Sequence):
        return []

    snapshots: List[Tuple[int, Dict[str, Any]]] = []
    for row in generation_details:
        if not isinstance(row, Mapping):
            continue

        raw_gen = row.get("gen")
        try:
            generation_index = int(raw_gen)
        except (TypeError, ValueError):
            continue

        positions = _as_positions_3d(row.get("best_ap_positions", row.get("ap_positions")))
        if positions is None:
            continue

        directions = _as_directions_3d(
            row.get("best_ap_directions", row.get("ap_directions"))
        )

        reflector = None
        best_reflector = row.get("best_reflector")
        if isinstance(best_reflector, Mapping):
            reflector = _extract_reflector_state({"reflector": best_reflector})
        if reflector is None:
            reflector = _extract_reflector_state(row)

        snapshots.append(
            (
                generation_index,
                {
                    "positions": positions,
                    "directions": directions,
                    "reflector": reflector,
                },
            )
        )

    snapshots.sort(key=lambda item: item[0])
    return snapshots


def _build_gd_snapshots(summary: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build GD initial/best/final snapshots from global best task history."""
    gd_results = summary.get("gd_results", {})
    if not isinstance(gd_results, Mapping):
        return {}
    best_task = gd_results.get("global_best_result")
    if not isinstance(best_task, Mapping):
        return {}

    history = best_task.get("history")
    if not isinstance(history, Mapping):
        return {}

    positions_seq = history.get("positions")
    primary_seq = history.get("primary_loss")
    if not isinstance(positions_seq, Sequence) or len(positions_seq) == 0:
        return {}

    best_index = len(positions_seq) - 1
    if isinstance(primary_seq, Sequence) and len(primary_seq) == len(positions_seq):
        best_index = int(np.argmin([float(value) for value in primary_seq]))

    direction_seq = history.get("directions") if isinstance(history.get("directions"), Sequence) else None
    reflector_u = history.get("reflector_u") if isinstance(history.get("reflector_u"), Sequence) else None
    reflector_v = history.get("reflector_v") if isinstance(history.get("reflector_v"), Sequence) else None
    reflector_target = history.get("reflector_target") if isinstance(history.get("reflector_target"), Sequence) else None

    def _snapshot_at(index: int) -> Optional[Dict[str, Any]]:
        positions = _as_positions_3d(positions_seq[index])
        if positions is None:
            return None
        directions = None
        if direction_seq is not None and len(direction_seq) > index:
            directions = _as_directions_3d(direction_seq[index])

        reflector = None
        if (
            reflector_u is not None
            and reflector_v is not None
            and reflector_target is not None
            and len(reflector_u) > index
            and len(reflector_v) > index
            and len(reflector_target) > index
        ):
            target = reflector_target[index]
            if isinstance(target, Sequence) and len(target) >= 3:
                reflector = {
                    "u": float(reflector_u[index]),
                    "v": float(reflector_v[index]),
                    "target": [float(target[0]), float(target[1]), float(target[2])],
                }

        return {
            "positions": positions,
            "directions": directions,
            "reflector": reflector,
        }

    initial_snapshot = _snapshot_at(0)
    best_snapshot = _snapshot_at(best_index)
    final_snapshot = _snapshot_at(len(positions_seq) - 1)

    snapshots: Dict[str, Dict[str, Any]] = {}
    if initial_snapshot is not None:
        snapshots["initial"] = initial_snapshot
    if best_snapshot is not None:
        snapshots["best"] = best_snapshot
    if final_snapshot is not None:
        snapshots["final"] = final_snapshot
    return snapshots


def _build_gd_task_ga_snapshot(result: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Build one GA-seed snapshot that initialized this GD trajectory."""
    optimizer_kwargs = result.get("optimizer_kwargs")
    if not isinstance(optimizer_kwargs, Mapping):
        return None

    fixed_z = float(optimizer_kwargs.get("fixed_z", 0.0))
    positions = _as_positions_3d_with_fixed_z(optimizer_kwargs.get("initial_positions"), fixed_z)
    if positions is None:
        return None

    directions = _as_directions_3d(
        optimizer_kwargs.get("initial_directions_xyz", optimizer_kwargs.get("initial_directions_xy"))
    )
    reflector = _extract_reflector_state(optimizer_kwargs)
    return {
        "positions": positions,
        "directions": directions,
        "reflector": reflector,
    }


def _build_gd_task_trajectory_snapshots(result: Mapping[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Build labeled per-iteration GD snapshots for one task result."""
    history = result.get("history")
    if not isinstance(history, Mapping):
        return []

    positions_seq = history.get("positions")
    if not isinstance(positions_seq, Sequence) or len(positions_seq) == 0:
        return []

    direction_seq = history.get("directions") if isinstance(history.get("directions"), Sequence) else None
    reflector_u = history.get("reflector_u") if isinstance(history.get("reflector_u"), Sequence) else None
    reflector_v = history.get("reflector_v") if isinstance(history.get("reflector_v"), Sequence) else None
    reflector_target = history.get("reflector_target") if isinstance(history.get("reflector_target"), Sequence) else None

    snapshots: List[Tuple[str, Dict[str, Any]]] = []
    for index, raw_positions in enumerate(positions_seq):
        positions = _as_positions_3d(raw_positions)
        if positions is None:
            continue

        directions = None
        if direction_seq is not None and len(direction_seq) > index:
            directions = _as_directions_3d(direction_seq[index])

        reflector = None
        if (
            reflector_u is not None
            and reflector_v is not None
            and reflector_target is not None
            and len(reflector_u) > index
            and len(reflector_v) > index
            and len(reflector_target) > index
        ):
            target = reflector_target[index]
            if isinstance(target, Sequence) and len(target) >= 3:
                reflector = {
                    "u": float(reflector_u[index]),
                    "v": float(reflector_v[index]),
                    "target": [float(target[0]), float(target[1]), float(target[2])],
                }

        snapshots.append(
            (
                f"iter_{index:04d}",
                {
                    "positions": positions,
                    "directions": directions,
                    "reflector": reflector,
                },
            )
        )

    return snapshots


def _resolve_render_scene_config(
    config_args: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Resolve plotting scene config with optional visualization overrides.

    Priority order:
    1. Base from ``scene_config`` (required)
    2. Apply ``visualization_scene_config`` mapping overrides (optional)
    3. Apply ``visualization_scene_path`` override for convenience (optional)
    """
    base_scene_config = config_args.get("scene_config")
    if not isinstance(base_scene_config, Mapping):
        return None

    render_scene_config: Dict[str, Any] = dict(base_scene_config)

    raw_visualization_scene_config = config_args.get("visualization_scene_config")
    if isinstance(raw_visualization_scene_config, Mapping):
        render_scene_config.update(dict(raw_visualization_scene_config))

    visualization_scene_path = config_args.get("visualization_scene_path")
    if visualization_scene_path is not None:
        render_scene_config["scene_path"] = str(visualization_scene_path)

    # Preserve top-level placement controls used by scene setup when they are
    # not embedded inside scene_config.
    if "num_aps" in config_args and "num_aps" not in render_scene_config:
        render_scene_config["num_aps"] = config_args.get("num_aps")
    if "position_bounds" in config_args and "position_bounds" not in render_scene_config:
        render_scene_config["position_bounds"] = config_args.get("position_bounds")

    return render_scene_config


def _coerce_xyz_triplet(
    raw_value: Any,
    default: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Convert a raw value into an XYZ triplet, or return default."""
    if (
        not isinstance(raw_value, Sequence)
        or isinstance(raw_value, (str, bytes))
        or len(raw_value) < 3
    ):
        return default

    try:
        return (float(raw_value[0]), float(raw_value[1]), float(raw_value[2]))
    except (TypeError, ValueError):
        return default


def _resolve_render_camera(
    config_args: Mapping[str, Any],
    render_settings: Mapping[str, Any],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Resolve render camera position/look_at with optional config overrides."""
    default_position = (20.0, 20.0, 55.0)
    default_look_at = (20.0, 20.1, 1.5)

    merged_camera: Dict[str, Any] = {}

    raw_top_level_camera = config_args.get("camera")
    if isinstance(raw_top_level_camera, Mapping):
        merged_camera.update(dict(raw_top_level_camera))

    raw_plot_camera = render_settings.get("camera")
    if isinstance(raw_plot_camera, Mapping):
        merged_camera.update(dict(raw_plot_camera))

    if "camera_position" in config_args:
        merged_camera["position"] = config_args.get("camera_position")
    if "camera_look_at" in config_args:
        merged_camera["look_at"] = config_args.get("camera_look_at")
    if "camera_position" in render_settings:
        merged_camera["position"] = render_settings.get("camera_position")
    if "camera_look_at" in render_settings:
        merged_camera["look_at"] = render_settings.get("camera_look_at")

    position = _coerce_xyz_triplet(merged_camera.get("position"), default_position)
    look_at = _coerce_xyz_triplet(merged_camera.get("look_at"), default_look_at)
    return position, look_at


def _apply_snapshot_to_scene(
    scene: Any,
    reflector_controller: Any,
    snapshot: Mapping[str, Any],
) -> None:
    """Apply AP and reflector snapshot values to the mutable scene."""
    import torch

    positions = snapshot.get("positions")
    directions = snapshot.get("directions")
    reflector = snapshot.get("reflector")

    transmitters = list(scene.transmitters.values())
    if not isinstance(positions, Sequence) or len(positions) == 0:
        return

    for index, position in enumerate(positions[: len(transmitters)]):
        pos = [float(position[0]), float(position[1]), float(position[2])]
        transmitters[index].position = pos
        if isinstance(directions, Sequence) and len(directions) > index and directions[index] is not None:
            direction = directions[index]
            dx = float(direction[0])
            dy = float(direction[1])
            dz = float(direction[2]) if len(direction) >= 3 else 0.0
            target = [pos[0] + dx, pos[1] + dy, pos[2] + dz]
            transmitters[index].look_at(target)

    if reflector_controller is None or not isinstance(reflector, Mapping):
        return

    target = reflector.get("target")
    if not isinstance(target, Sequence) or len(target) < 3:
        return

    device = getattr(reflector_controller, "device", "cpu")
    reflector_controller.u = torch.tensor(float(reflector.get("u", 0.5)), dtype=torch.float32, device=device)
    reflector_controller.v = torch.tensor(float(reflector.get("v", 0.5)), dtype=torch.float32, device=device)
    reflector_controller.set_tx_position(np.asarray(positions[0], dtype=np.float32))
    reflector_controller.set_focal_point(
        torch.tensor([float(target[0]), float(target[1]), float(target[2])], dtype=torch.float32, device=device),
        requires_grad=False,
    )
    reflector_controller.orient_to_target()
    reflector_controller.apply_to_scene()


def _render_coverage_snapshot(
    scene_config: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    save_path: Path,
    samples_per_tx: int,
    max_depth: int,
    resolution: Tuple[int, int],
    camera_position: Tuple[float, float, float],
    camera_look_at: Tuple[float, float, float],
) -> Optional[str]:
    """Render one coverage-map image using Sionna Scene.render_to_file."""
    from sionna.rt import RadioMapSolver

    from reflector_position.scene_setup import create_camera, setup_building_floor_scene

    effective_num_aps = scene_config.get("num_aps")
    if effective_num_aps is None:
        snapshot_positions = snapshot.get("positions")
        if isinstance(snapshot_positions, Sequence) and not isinstance(snapshot_positions, (str, bytes)):
            effective_num_aps = int(len(snapshot_positions))

    loaded = setup_building_floor_scene(
        scene_path=str(scene_config["scene_path"]),
        frequency=scene_config.get("frequency", 6e9),
        tx_positions=scene_config.get("tx_positions", None),
        num_aps=effective_num_aps,
        position_bounds=scene_config.get("position_bounds", None),
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
    else:
        scene, reflector_controller = loaded, None

    _apply_snapshot_to_scene(scene, reflector_controller, snapshot)

    solver = RadioMapSolver()
    radio_map = solver(
        scene,
        cell_size=(1.0, 1.0),
        samples_per_tx=int(samples_per_tx),
        max_depth=int(max_depth),
        refraction=True,
        diffraction=True,
    )

    camera = create_camera(position=camera_position, look_at=camera_look_at)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render_to_file(
        camera=camera,
        filename=str(save_path),
        radio_map=radio_map,
        rm_metric="rss",
        rm_db_scale=True,
        rm_vmin=-80,
        rm_vmax=-40,
        resolution=resolution,
        show_devices=True,
        show_orientations=False,
    )
    return str(save_path)


def save_memetic_coverage_maps(
    summary: Mapping[str, Any],
    config_args: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, str]:
    """Save GA/GD coverage maps, including optional GA/GD trajectory frames."""
    render_scene_config = _resolve_render_scene_config(config_args)
    if render_scene_config is None:
        return {}

    render_settings = config_args.get("coverage_plot_settings")
    if not isinstance(render_settings, Mapping):
        render_settings = {}

    camera_position, camera_look_at = _resolve_render_camera(
        config_args=config_args,
        render_settings=render_settings,
    )

    samples_per_tx = int(render_settings.get("samples_per_tx", 1_000_000))
    max_depth = int(render_settings.get("max_depth", 13))
    resolution_raw = render_settings.get("resolution", (1200, 900))
    if isinstance(resolution_raw, Sequence) and len(resolution_raw) >= 2:
        resolution = (int(resolution_raw[0]), int(resolution_raw[1]))
    else:
        resolution = (1200, 900)
    render_gd_trajectory_coverage_maps = bool(
        render_settings.get(
            "render_gd_trajectory_coverage_maps",
            render_settings.get("render_all_gd_trajectory_frames", True),
        )
    )
    render_ga_generation_best_coverage_maps = bool(
        render_settings.get("render_ga_generation_best_coverage_maps", False)
    )
    ga_generation_frame_stride = max(
        1,
        int(render_settings.get("ga_generation_frame_stride", 1)),
    )
    ga_generation_max_frames_raw = render_settings.get("ga_generation_max_frames")
    ga_generation_max_frames: Optional[int] = None
    if ga_generation_max_frames_raw is not None:
        try:
            ga_generation_max_frames = max(1, int(ga_generation_max_frames_raw))
        except (TypeError, ValueError):
            ga_generation_max_frames = None

    render_all_gd_trajectory_frames = bool(render_settings.get("render_all_gd_trajectory_frames", True))
    gd_trajectory_frame_stride = max(1, int(render_settings.get("gd_trajectory_frame_stride", 1)))
    gd_trajectory_max_frames_raw = render_settings.get("gd_trajectory_max_frames")
    gd_trajectory_max_frames: Optional[int] = None
    if gd_trajectory_max_frames_raw is not None:
        try:
            gd_trajectory_max_frames = max(1, int(gd_trajectory_max_frames_raw))
        except (TypeError, ValueError):
            gd_trajectory_max_frames = None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, str] = {}
    states_by_method = {
        "ga": _build_ga_snapshots(summary, render_scene_config),
        "gd": _build_gd_snapshots(summary),
    }

    for method, snapshots in states_by_method.items():
        for stage in ("initial", "best", "final"):
            snapshot = snapshots.get(stage)
            if not isinstance(snapshot, Mapping):
                continue

            image_path = plots_dir / f"coverage_map_{method}_{stage}.png"
            key = f"coverage_map_{method}_{stage}"
            try:
                rendered = _render_coverage_snapshot(
                    scene_config=render_scene_config,
                    snapshot=snapshot,
                    save_path=image_path,
                    samples_per_tx=samples_per_tx,
                    max_depth=max_depth,
                    resolution=resolution,
                    camera_position=camera_position,
                    camera_look_at=camera_look_at,
                )
                if rendered is not None:
                    artifacts[key] = rendered
            except Exception as exc:
                artifacts[f"coverage_map_{method}_{stage}_error"] = str(exc)

    if render_ga_generation_best_coverage_maps:
        generation_snapshots = _build_ga_generation_best_snapshots(summary)
        if ga_generation_frame_stride > 1:
            generation_snapshots = [
                pair
                for frame_index, pair in enumerate(generation_snapshots)
                if frame_index % ga_generation_frame_stride == 0
            ]
        if ga_generation_max_frames is not None:
            generation_snapshots = generation_snapshots[:ga_generation_max_frames]

        base_dir = plots_dir / "ga_generation_best_coverage"
        rendered_generation_count = 0
        for generation_index, snapshot in generation_snapshots:
            image_path = base_dir / f"gen_{generation_index:04d}.png"
            try:
                rendered = _render_coverage_snapshot(
                    scene_config=render_scene_config,
                    snapshot=snapshot,
                    save_path=image_path,
                    samples_per_tx=samples_per_tx,
                    max_depth=max_depth,
                    resolution=resolution,
                    camera_position=camera_position,
                    camera_look_at=camera_look_at,
                )
                if rendered is not None:
                    rendered_generation_count += 1
                    artifacts[f"coverage_map_ga_generation_{generation_index:04d}"] = rendered
            except Exception as exc:
                artifacts[
                    f"coverage_map_ga_generation_{generation_index:04d}_error"
                ] = str(exc)

        artifacts["coverage_map_ga_generation_best_dir"] = str(base_dir)
        artifacts["coverage_map_ga_generation_best_count"] = str(rendered_generation_count)

    if render_gd_trajectory_coverage_maps and render_all_gd_trajectory_frames:
        gd_results = summary.get("gd_results", {})
        all_results = gd_results.get("all_fine_tuned_results", []) if isinstance(gd_results, Mapping) else []
        if isinstance(all_results, Sequence):
            base_dir = plots_dir / "gd_trajectory_coverage"
            rendered_image_count = 0
            rendered_task_count = 0

            for fallback_task_index, result in enumerate(all_results):
                if not isinstance(result, Mapping):
                    continue

                task_id = int(result.get("task_id", fallback_task_index))
                task_dir = base_dir / f"task_{task_id:03d}"
                task_rendered = 0

                ga_snapshot = _build_gd_task_ga_snapshot(result)
                if isinstance(ga_snapshot, Mapping):
                    ga_path = task_dir / "ga_final.png"
                    try:
                        rendered = _render_coverage_snapshot(
                            scene_config=render_scene_config,
                            snapshot=ga_snapshot,
                            save_path=ga_path,
                            samples_per_tx=samples_per_tx,
                            max_depth=max_depth,
                            resolution=resolution,
                            camera_position=camera_position,
                            camera_look_at=camera_look_at,
                        )
                        if rendered is not None:
                            task_rendered += 1
                    except Exception as exc:
                        artifacts[f"coverage_map_gd_task_{task_id}_ga_final_error"] = str(exc)

                snapshots = _build_gd_task_trajectory_snapshots(result)
                if gd_trajectory_frame_stride > 1:
                    snapshots = [
                        pair for frame_index, pair in enumerate(snapshots) if frame_index % gd_trajectory_frame_stride == 0
                    ]
                if gd_trajectory_max_frames is not None:
                    snapshots = snapshots[:gd_trajectory_max_frames]

                for label, snapshot in snapshots:
                    image_path = task_dir / f"{label}.png"
                    try:
                        rendered = _render_coverage_snapshot(
                            scene_config=render_scene_config,
                            snapshot=snapshot,
                            save_path=image_path,
                            samples_per_tx=samples_per_tx,
                            max_depth=max_depth,
                            resolution=resolution,
                            camera_position=camera_position,
                            camera_look_at=camera_look_at,
                        )
                        if rendered is not None:
                            task_rendered += 1
                    except Exception as exc:
                        artifacts[f"coverage_map_gd_task_{task_id}_{label}_error"] = str(exc)

                if task_rendered > 0:
                    rendered_task_count += 1
                    rendered_image_count += task_rendered
                    artifacts[f"coverage_map_gd_task_{task_id}_dir"] = str(task_dir)
                    artifacts[f"coverage_map_gd_task_{task_id}_image_count"] = str(task_rendered)

            artifacts["coverage_map_gd_trajectory_dir"] = str(base_dir)
            artifacts["coverage_map_gd_trajectory_task_count"] = str(rendered_task_count)
            artifacts["coverage_map_gd_trajectory_image_count"] = str(rendered_image_count)

    return artifacts