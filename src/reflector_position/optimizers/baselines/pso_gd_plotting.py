"""PSO+GD specific plotting helpers.

This module renders baseline-specific trajectory plots for PSO followed by
gradient-descent refinement. The plotting contract intentionally mirrors the
style used by memetic plotting while remaining decoupled from memetic code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except Exception:  # pragma: no cover - plotting optional at runtime
    plt = None


_PRIORITY_METRIC_KEYS: Tuple[str, ...] = (
    "priority_mean_rss_dbm",
    "priority_min_rss_dbm",
    "priority_p5_rss_dbm",
)

_ALL_REGION_METRIC_KEYS: Tuple[str, ...] = (
    "mean_rss_dbm",
    "min_rss_dbm",
    "p5_rss_dbm",
)

_PRIMARY_LOSS_KEYS: Tuple[str, ...] = (
    "primary_loss",
    "min_primary_loss",
    "global_best_primary_loss",
    "swarm_best_primary_loss",
    "running_best_primary_loss",
)


def _as_float(value: Any) -> float:
    """Convert one value to float, returning NaN on invalid inputs."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _set_integer_x_ticks(axis: Any) -> None:
    """Force integer ticks for optimization-step axes."""
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))


def _extract_trace_rows(result_payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Extract normalized iteration rows from one PSO+GD result payload."""
    raw_rows = result_payload.get("iteration_trace")
    if not isinstance(raw_rows, Sequence):
        return []

    rows: List[Mapping[str, Any]] = []
    for row in raw_rows:
        if isinstance(row, Mapping):
            rows.append(row)
    return rows


def _extract_series(trace_rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    """Extract one numeric series from trace rows as a NumPy array."""
    return np.asarray([_as_float(row.get(key)) for row in trace_rows], dtype=np.float64)


def _extract_primary_loss_series(trace_rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Extract primary-loss series using ordered key fallbacks."""
    values: List[float] = []
    for row in trace_rows:
        selected = float("nan")
        for key in _PRIMARY_LOSS_KEYS:
            candidate = _as_float(row.get(key))
            if np.isfinite(candidate):
                selected = candidate
                break
        values.append(selected)
    return np.asarray(values, dtype=np.float64)


def _extract_best_iteration(primary_loss: np.ndarray) -> Optional[int]:
    """Return one zero-based best iteration index for finite loss series."""
    if primary_loss.size == 0:
        return None

    finite_mask = np.isfinite(primary_loss)
    if not np.any(finite_mask):
        return None

    finite_indices = np.where(finite_mask)[0]
    finite_values = primary_loss[finite_mask]
    best_local_index = int(np.argmin(finite_values))
    return int(finite_indices[best_local_index])


def _extract_priority_map(result_payload: Mapping[str, Any]) -> Optional[np.ndarray]:
    """Extract spatial priority-map array from one PSO+GD result payload."""
    raw_map = result_payload.get("spatial_weights")
    if raw_map is None:
        return None

    try:
        array = np.asarray(raw_map, dtype=np.float64)
    except Exception:
        return None

    if array.ndim == 2 and array.size > 0:
        return array
    return None


def _extract_position_extent(result_payload: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Extract heatmap extent from payload position_bounds when available."""
    raw_bounds = result_payload.get("position_bounds")
    if not isinstance(raw_bounds, Mapping):
        return None

    required_keys = ("x_min", "x_max", "y_min", "y_max")
    if not all(key in raw_bounds for key in required_keys):
        return None

    x_min = _as_float(raw_bounds.get("x_min"))
    x_max = _as_float(raw_bounds.get("x_max"))
    y_min = _as_float(raw_bounds.get("y_min"))
    y_max = _as_float(raw_bounds.get("y_max"))
    if not all(np.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
        return None

    if x_min >= x_max or y_min >= y_max:
        return None

    return float(x_min), float(x_max), float(y_min), float(y_max)


def _plot_metrics_panel(
    axis: Any,
    steps: np.ndarray,
    trace_rows: Sequence[Mapping[str, Any]],
    metric_keys: Sequence[str],
    title: str,
) -> None:
    """Render one metric-evolution panel for a list of metric keys."""
    color_cycle = ("tab:blue", "tab:purple", "tab:cyan")
    plotted = False

    for index, metric_key in enumerate(metric_keys):
        series = _extract_series(trace_rows, metric_key)
        finite_mask = np.isfinite(series)
        if not np.any(finite_mask):
            continue

        axis.plot(
            steps[finite_mask],
            series[finite_mask],
            linewidth=2.0,
            color=color_cycle[index % len(color_cycle)],
            label=metric_key,
        )
        plotted = True

    axis.set_xlabel("Optimization Step")
    axis.set_ylabel("RSSI (dBm)")
    axis.set_title(title)
    _set_integer_x_ticks(axis)
    axis.grid(True, alpha=0.3)
    if plotted:
        axis.legend(loc="best", fontsize=9)


def save_pso_gd_trajectory_plot(
    result_payload: Mapping[str, Any],
    save_path: Path,
) -> Optional[str]:
    """Save one PSO+GD trajectory plot with focused 4-panel layout."""
    if plt is None:
        return None

    trace_rows = _extract_trace_rows(result_payload)
    if not trace_rows:
        return None

    steps = np.arange(1, len(trace_rows) + 1, dtype=np.int64)
    primary_loss = _extract_primary_loss_series(trace_rows)
    running_best = _extract_series(trace_rows, "running_best_primary_loss")
    best_iteration = _extract_best_iteration(primary_loss)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    primary_axis = axes[0, 0]
    finite_primary = np.isfinite(primary_loss)
    if np.any(finite_primary):
        primary_axis.plot(
            steps[finite_primary],
            primary_loss[finite_primary],
            color="tab:blue",
            linewidth=2.0,
            label="primary_loss",
        )

    finite_running = np.isfinite(running_best)
    if np.any(finite_running):
        primary_axis.plot(
            steps[finite_running],
            running_best[finite_running],
            color="tab:orange",
            linewidth=1.7,
            linestyle="--",
            label="running_best_primary_loss",
        )

    if best_iteration is not None:
        best_x = int(best_iteration + 1)
        primary_axis.axvline(
            best_x,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Best iter {best_x}",
        )

    primary_axis.set_xlabel("Optimization Step")
    primary_axis.set_ylabel("Primary Loss")
    primary_axis.set_title("Primary Loss Evolution")
    _set_integer_x_ticks(primary_axis)
    primary_axis.grid(True, alpha=0.3)
    if primary_axis.lines:
        primary_axis.legend(loc="best", fontsize=9)

    _plot_metrics_panel(
        axis=axes[0, 1],
        steps=steps,
        trace_rows=trace_rows,
        metric_keys=_PRIORITY_METRIC_KEYS,
        title="Priority Metrics Evolution",
    )

    _plot_metrics_panel(
        axis=axes[1, 0],
        steps=steps,
        trace_rows=trace_rows,
        metric_keys=_ALL_REGION_METRIC_KEYS,
        title="Metrics Evolution (All-region)",
    )

    heatmap_axis = axes[1, 1]
    priority_map = _extract_priority_map(result_payload)
    if priority_map is None:
        heatmap_axis.axis("off")
        heatmap_axis.text(
            0.02,
            0.95,
            "Priority map unavailable in PSO+GD payload.",
            transform=heatmap_axis.transAxes,
            va="top",
            ha="left",
            fontsize=10,
        )
    else:
        extent = _extract_position_extent(result_payload)
        image = heatmap_axis.imshow(
            priority_map,
            cmap="magma",
            origin="lower",
            extent=extent,
            aspect="equal",
        )
        heatmap_axis.set_title("Priority Map Heatmap")
        heatmap_axis.set_xlabel("X Position (m)")
        heatmap_axis.set_ylabel("Y Position (m)")
        heatmap_axis.grid(False)
        colorbar = figure.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Priority")

    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return str(save_path)


def save_pso_gd_plots(
    result_payload: Mapping[str, Any],
    plots_dir: Path,
) -> Dict[str, str]:
    """Save all PSO+GD-specific plots and return artifact-path mapping."""
    artifacts: Dict[str, str] = {}
    trajectory_path = save_pso_gd_trajectory_plot(
        result_payload=result_payload,
        save_path=plots_dir / "pso_gd_trajectory.png",
    )
    if trajectory_path is not None:
        artifacts["pso_gd_trajectory_plot_png"] = trajectory_path
    return artifacts


__all__ = ["save_pso_gd_plots", "save_pso_gd_trajectory_plot"]
