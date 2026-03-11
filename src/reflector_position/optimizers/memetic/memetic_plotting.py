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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_AP_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
_AP_MARKERS = ["o", "s", "^", "D", "v", "P"]
_PHYSICAL_METRIC_PRIORITY = ("coverage_pct", "p5_rss_dbm", "min_rss_dbm", "mean_rss_dbm")


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


def _fmt_dir(direction: Any) -> str:
    """Format one or more direction vectors for display."""
    if direction is None:
        return "N/A"

    if isinstance(direction, (list, tuple, np.ndarray)) and len(direction) > 0:
        if isinstance(direction[0], (list, tuple, np.ndarray)):
            parts = [f"({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f})" for d in direction]
            return " | ".join(parts)

    return f"({direction[0]:+.4f}, {direction[1]:+.4f}, {direction[2]:+.4f})"


def _fmt_pos(position: Any) -> str:
    """Format one or more positions for display."""
    if position is None:
        return "N/A"

    if isinstance(position, (list, tuple, np.ndarray)) and len(position) > 0:
        if isinstance(position[0], (list, tuple, np.ndarray)):
            parts = [f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" for p in position]
            return " | ".join(parts)

    return f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"


def save_ga_training_curve(ga_results: Mapping[str, Any], save_path: Path) -> Optional[str]:
    """Save GA best/mean primary-fitness curve."""
    details = ga_results.get("generation_details", [])
    if not isinstance(details, list) or len(details) == 0:
        return None

    generations = [int(row.get("gen", idx)) for idx, row in enumerate(details)]
    best_values = [float(row.get("best_primary_fitness")) for row in details if row.get("best_primary_fitness") is not None]
    mean_values = [float(row.get("mean_primary_fitness")) for row in details if row.get("mean_primary_fitness") is not None]
    if len(generations) != len(best_values) or len(generations) != len(mean_values):
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, best_values, marker="o", linewidth=2.0, label="GA Best")
    ax.plot(generations, mean_values, marker="s", linewidth=1.6, label="GA Mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Primary Fitness")
    ax.set_title("Memetic Phase-1 Training Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


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
    ax.legend()
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
    ax.legend()
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
        secondary_metric_name, secondary_metric_values = _select_secondary_metric(result)
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
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
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
        ax.legend(fontsize=8)
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
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if secondary_metric_values:
            iterations = list(range(1, len(secondary_metric_values) + 1))
            ax.plot(iterations, secondary_metric_values, "g-", linewidth=2)
            if 0 <= best_iter < len(secondary_metric_values):
                ax.axvline(best_iter + 1, color="red", linestyle="--", alpha=0.7, label=f"Best iter {best_iter + 1}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(secondary_metric_name)
        ax.set_title(f"{secondary_metric_name} Evolution")
        ax.legend(fontsize=9)
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
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

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

    if ga_plot is not None:
        artifacts["ga_training_plot"] = ga_plot
    if gd_plot is not None:
        artifacts["gd_seed_plot"] = gd_plot
    artifacts["timing_plot"] = timing_plot
    if gd_summary_plot is not None:
        artifacts["gd_parallel_summary_plot"] = gd_summary_plot
    artifacts["gd_trajectory_dir"] = str(plots_dir / "gd_trajectories")
    artifacts["gd_trajectory_count"] = str(len(gd_trajectory_paths))
    return artifacts