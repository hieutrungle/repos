"""Static comparison plotting utilities for experiment outputs.

This module centralizes matplotlib-based plotting so orchestration scripts can
keep method execution logic separate from plotting/reporting concerns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - plotting optional at runtime
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting optional at runtime
    plt = None


_PRIMARY_LOSS_KEYS: Tuple[str, ...] = (
    "running_best_primary_loss",
    "global_best_primary_loss",
    "min_primary_loss",
    "primary_loss",
    "swarm_best_primary_loss",
)

_RSSI_METRIC_KEYS: Tuple[str, ...] = (
    "mean_rss_dbm",
    "min_rss_dbm",
    "p5_rss_dbm",
)

_RSSI_METRIC_LABELS: Dict[str, str] = {
    "mean_rss_dbm": "Mean RSSI (dBm)",
    "min_rss_dbm": "Minimum RSSI (dBm)",
    "p5_rss_dbm": "5th Percentile RSSI (dBm)",
}


def _as_float(value: Any) -> Optional[float]:
    """Convert arbitrary value to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_best_physical_metric(
    method: str,
    result_payload: Mapping[str, Any],
    metric_key: str,
) -> Optional[float]:
    """Extract one best physical metric from method result payloads."""
    candidate_maps: List[Mapping[str, Any]] = []

    top_level_metrics = result_payload.get("best_physical_metrics")
    if isinstance(top_level_metrics, Mapping):
        candidate_maps.append(top_level_metrics)

    if method == "memetic":
        global_best = result_payload.get("global_best_result")
        if isinstance(global_best, Mapping):
            result_summary = global_best.get("results")
            if isinstance(result_summary, Mapping):
                for key in ("best_physical_metrics", "physical_metrics"):
                    metrics = result_summary.get(key)
                    if isinstance(metrics, Mapping):
                        candidate_maps.append(metrics)

            for key in ("best_physical_metrics", "physical_metrics"):
                metrics = global_best.get(key)
                if isinstance(metrics, Mapping):
                    candidate_maps.append(metrics)

    for metric_map in candidate_maps:
        metric_value = _as_float(metric_map.get(metric_key))
        if metric_value is not None:
            return metric_value

    return None


def _extract_trace_series(
    trace_rows: Sequence[Mapping[str, Any]],
    preferred_keys: Sequence[str],
) -> List[float]:
    """Extract a numeric series from trace rows using preferred key ordering."""
    values: List[float] = []
    for row in trace_rows:
        selected: Optional[float] = None
        for key in preferred_keys:
            if key not in row:
                continue
            selected = _as_float(row.get(key))
            if selected is not None:
                break
        if selected is not None:
            values.append(float(selected))
    return values


def _expand_single_point_series(series: Sequence[float], target_len: int) -> List[float]:
    """Expand one-point series to a straight line across the target length."""
    values = [float(value) for value in series]
    if target_len <= 1:
        return values
    if len(values) == 1:
        return [values[0]] * int(target_len)
    return values


def _build_primary_loss_series(
    method: str,
    trace_rows: Sequence[Mapping[str, Any]],
    result_payload: Mapping[str, Any],
) -> List[float]:
    """Build primary-loss series for one method with robust fallback."""
    series = _extract_trace_series(trace_rows, _PRIMARY_LOSS_KEYS)
    if series:
        return series

    if method == "memetic":
        gd_results = result_payload.get("gd_results", {})
        if isinstance(gd_results, Mapping):
            metrics = gd_results.get("metrics", {})
            if isinstance(metrics, Mapping):
                best_loss = _as_float(metrics.get("best_primary_loss"))
                if best_loss is not None:
                    return [best_loss]

    best_primary_loss = _as_float(result_payload.get("best_primary_loss"))
    if best_primary_loss is not None:
        return [best_primary_loss]

    return []


def _build_rssi_metric_series(
    method: str,
    trace_rows: Sequence[Mapping[str, Any]],
    result_payload: Mapping[str, Any],
    metric_key: str,
    fallback_length: int,
) -> List[float]:
    """Build one RSSI metric series from traces with best-metric fallback."""
    series = _extract_trace_series(trace_rows, [metric_key])
    if series:
        return series

    best_metric = _extract_best_physical_metric(
        method=method,
        result_payload=result_payload,
        metric_key=metric_key,
    )
    if best_metric is None:
        return []

    length = max(1, int(fallback_length))
    return [float(best_metric)] * length


def _plot_primary_loss_comparison(
    methods: Sequence[str],
    method_results: Mapping[str, Mapping[str, Any]],
    method_trace_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    plots_dir: Path,
) -> Optional[str]:
    """Save one static all-method primary-loss comparison plot."""
    if plt is None:
        return None

    raw_series_map: Dict[str, List[float]] = {}
    for method in methods:
        raw_series_map[method] = _build_primary_loss_series(
            method=method,
            trace_rows=method_trace_rows.get(method, []),
            result_payload=method_results.get(method, {}),
        )

    max_len = max((len(series) for series in raw_series_map.values()), default=0)
    if max_len < 1:
        return None

    figure, axis = plt.subplots(figsize=(12.0, 6.0))
    for method in methods:
        series = _expand_single_point_series(raw_series_map.get(method, []), target_len=max_len)
        if not series:
            continue

        x_values = np.arange(1, len(series) + 1, dtype=np.int64)
        axis.plot(x_values, series, linewidth=2.0, label=method)

    if not axis.lines:
        plt.close(figure)
        return None

    axis.set_title("Primary Loss Trend - All Methods (Static)")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Primary Loss")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")

    output_path = plots_dir / "all_methods_primary_loss_static.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return str(output_path)


def _plot_per_method_rssi_triplets(
    methods: Sequence[str],
    method_results: Mapping[str, Mapping[str, Any]],
    method_trace_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    plots_dir: Path,
    y_limits: Tuple[float, float],
) -> Dict[str, Optional[str]]:
    """Save one per-method static plot containing mean/min/p5 RSSI trends."""
    output_paths: Dict[str, Optional[str]] = {}
    if plt is None:
        for method in methods:
            output_paths[method] = None
        return output_paths

    y_min, y_max = float(y_limits[0]), float(y_limits[1])
    for method in methods:
        trace_rows = method_trace_rows.get(method, [])
        fallback_len = max(1, len(trace_rows))

        metric_series_map: Dict[str, List[float]] = {}
        for metric_key in _RSSI_METRIC_KEYS:
            metric_series_map[metric_key] = _build_rssi_metric_series(
                method=method,
                trace_rows=trace_rows,
                result_payload=method_results.get(method, {}),
                metric_key=metric_key,
                fallback_length=fallback_len,
            )

        target_len = max((len(series) for series in metric_series_map.values()), default=0)
        if target_len < 1:
            output_paths[method] = None
            continue

        figure, axis = plt.subplots(figsize=(12.0, 6.0))
        for metric_key in _RSSI_METRIC_KEYS:
            series = _expand_single_point_series(metric_series_map.get(metric_key, []), target_len=target_len)
            if not series:
                continue

            x_values = np.arange(1, len(series) + 1, dtype=np.int64)
            axis.plot(
                x_values,
                series,
                linewidth=2.0,
                label=_RSSI_METRIC_LABELS.get(metric_key, metric_key),
            )

        if not axis.lines:
            plt.close(figure)
            output_paths[method] = None
            continue

        axis.set_title(f"{method} RSSI Metric Trends (Static)")
        axis.set_xlabel("Iteration")
        axis.set_ylabel("RSSI (dBm)")
        axis.set_ylim(y_min, y_max)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

        output_path = plots_dir / f"{method}_rssi_triplet_static.png"
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        output_paths[method] = str(output_path)

    return output_paths


def _plot_cross_method_rssi_metrics(
    methods: Sequence[str],
    method_results: Mapping[str, Mapping[str, Any]],
    method_trace_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    plots_dir: Path,
    y_limits: Tuple[float, float],
) -> Dict[str, Optional[str]]:
    """Save static cross-method RSSI metric plots (mean/min/p5 separately)."""
    output_paths: Dict[str, Optional[str]] = {}
    if plt is None:
        for metric_key in _RSSI_METRIC_KEYS:
            output_paths[metric_key] = None
        return output_paths

    y_min, y_max = float(y_limits[0]), float(y_limits[1])
    for metric_key in _RSSI_METRIC_KEYS:
        raw_series_by_method: Dict[str, List[float]] = {}
        for method in methods:
            trace_rows = method_trace_rows.get(method, [])
            fallback_len = max(1, len(trace_rows))
            raw_series_by_method[method] = _build_rssi_metric_series(
                method=method,
                trace_rows=trace_rows,
                result_payload=method_results.get(method, {}),
                metric_key=metric_key,
                fallback_length=fallback_len,
            )

        target_len = max((len(series) for series in raw_series_by_method.values()), default=0)
        if target_len < 1:
            output_paths[metric_key] = None
            continue

        figure, axis = plt.subplots(figsize=(12.0, 6.0))
        for method in methods:
            series = _expand_single_point_series(raw_series_by_method.get(method, []), target_len=target_len)
            if not series:
                continue

            x_values = np.arange(1, len(series) + 1, dtype=np.int64)
            axis.plot(x_values, series, linewidth=2.0, label=method)

        if not axis.lines:
            plt.close(figure)
            output_paths[metric_key] = None
            continue

        metric_label = _RSSI_METRIC_LABELS.get(metric_key, metric_key)
        axis.set_title(f"{metric_label} Trend - All Methods (Static)")
        axis.set_xlabel("Iteration")
        axis.set_ylabel("RSSI (dBm)")
        axis.set_ylim(y_min, y_max)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

        output_path = plots_dir / f"all_methods_{metric_key}_static.png"
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        output_paths[metric_key] = str(output_path)

    return output_paths


def save_static_comparison_plots(
    methods: Sequence[str],
    method_results: Mapping[str, Mapping[str, Any]],
    method_trace_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    plots_dir: Path,
    rssi_y_limits: Tuple[float, float] = (-100.0, -40.0),
) -> Dict[str, Any]:
    """Generate static matplotlib comparison plots for experiment methods.

    Returns a serializable artifact dictionary containing output paths.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    primary_loss_plot = _plot_primary_loss_comparison(
        methods=methods,
        method_results=method_results,
        method_trace_rows=method_trace_rows,
        plots_dir=plots_dir,
    )
    per_method_rssi = _plot_per_method_rssi_triplets(
        methods=methods,
        method_results=method_results,
        method_trace_rows=method_trace_rows,
        plots_dir=plots_dir,
        y_limits=rssi_y_limits,
    )
    all_methods_rssi = _plot_cross_method_rssi_metrics(
        methods=methods,
        method_results=method_results,
        method_trace_rows=method_trace_rows,
        plots_dir=plots_dir,
        y_limits=rssi_y_limits,
    )

    return {
        "all_methods_primary_loss_static_png": primary_loss_plot,
        "per_method_rssi_triplet_static_png": per_method_rssi,
        "all_methods_rssi_metric_static_png": all_methods_rssi,
    }


__all__ = ["save_static_comparison_plots"]
