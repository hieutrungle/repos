"""Human-readable reporting helpers for memetic optimization runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


def _extract_history(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return history payload when present, else empty mapping."""
    history = result.get("history")
    return history if isinstance(history, Mapping) else {}


def _extract_results_payload(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return optimizer results payload when present, else empty mapping."""
    payload = result.get("results")
    return payload if isinstance(payload, Mapping) else {}


def _extract_best_primary_loss(result: Mapping[str, Any]) -> Optional[float]:
    """Return best-observed primary loss from raw result payload."""
    history = _extract_history(result)
    primary_loss = history.get("primary_loss")
    if isinstance(primary_loss, Sequence) and len(primary_loss) > 0:
        return float(min(float(value) for value in primary_loss))

    results_payload = _extract_results_payload(result)
    value = results_payload.get("primary_loss")
    return float(value) if value is not None else None


def _extract_best_loss_components(result: Mapping[str, Any]) -> Dict[str, float]:
    """Return best auxiliary loss dictionary from raw result payload."""
    results_payload = _extract_results_payload(result)
    components = results_payload.get("loss_components")
    if isinstance(components, Mapping):
        return {str(name): float(value) for name, value in components.items() if value is not None}
    return {}


def _extract_best_physical_metrics(result: Mapping[str, Any]) -> Dict[str, float]:
    """Return best physical metrics dictionary from raw result payload."""
    results_payload = _extract_results_payload(result)
    metrics = results_payload.get("physical_metrics")
    if isinstance(metrics, Mapping):
        return {str(name): float(value) for name, value in metrics.items() if value is not None}
    return {}


def _extract_best_position(result: Mapping[str, Any]) -> Any:
    """Return best AP position(s) from history or standardized result payload."""
    history = _extract_history(result)
    positions = history.get("positions")
    primary_loss = history.get("primary_loss")
    if isinstance(positions, Sequence) and len(positions) > 0:
        if isinstance(primary_loss, Sequence) and len(primary_loss) == len(positions):
            best_idx = int(np.argmin([float(value) for value in primary_loss]))
            return positions[best_idx]
        return positions[-1]

    results_payload = _extract_results_payload(result)
    best_configuration = results_payload.get("best_configuration")
    if isinstance(best_configuration, Mapping) and best_configuration.get("positions") is not None:
        return best_configuration.get("positions")

    return None


def _fmt_value(value: Optional[float], precision: int = 2) -> str:
    """Format numeric values for summary output."""
    if value is None:
        return "N/A"
    return f"{float(value):.{precision}f}"


def _fmt_position(position: Any) -> str:
    """Format one or more 3D positions for report output."""
    if position is None:
        return "N/A"
    if isinstance(position, (list, tuple)) and len(position) > 0 and isinstance(position[0], (list, tuple)):
        return " | ".join(f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" for p in position)
    return f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"


def _append_mapping_section(
    report_lines: List[str],
    title: str,
    values: Mapping[str, Any],
    precision: int = 6,
) -> None:
    """Append one formatted mapping section when values are present."""
    if not isinstance(values, Mapping) or not values:
        return
    report_lines.extend(["", title])
    for key, value in values.items():
        report_lines.append(f"- {key}: {_fmt_value(value, precision=precision)}")


def _coerce_optional_float(value: Any) -> Optional[float]:
    """Coerce value to float when possible, else return None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_metric(metrics: Mapping[str, Any], metric_key: str) -> Optional[float]:
    """Read one metric with float coercion."""
    return _coerce_optional_float(metrics.get(metric_key))


def build_memetic_summary_report(summary: Mapping[str, Any]) -> str:
    """Build a Markdown summary report for one memetic run."""
    ga_results = summary.get("ga_results", {})
    gd_results = summary.get("gd_results", {})
    timings = summary.get("timings", {})
    counts = summary.get("counts", {})
    global_best = summary.get("global_best_result")

    best_primary_loss = gd_results.get("metrics", {}).get("best_primary_loss")
    if best_primary_loss is None and isinstance(global_best, Mapping):
        best_primary_loss = _extract_best_primary_loss(global_best)

    report_lines: List[str] = [
        "# Memetic Optimization Summary",
        "",
        "## Runtime",
        f"- GA duration: {_fmt_value(timings.get('ga_duration_sec'))} s",
        f"- GD duration: {_fmt_value(timings.get('gd_duration_sec'))} s",
        f"- Total duration: {_fmt_value(timings.get('total_duration_sec'))} s",
        "",
        "## Counts",
        f"- Selected GA seeds: {counts.get('num_ga_seeds', 'N/A')}",
        f"- GD tasks launched: {counts.get('num_gd_tasks', 'N/A')}",
        f"- GD task results: {counts.get('num_gd_results', 'N/A')}",
        "",
        "## GA Outcome",
        f"- Best GA primary fitness: {_fmt_value(ga_results.get('best_primary_fitness'), precision=6)}",
        f"- Selected seeds: {ga_results.get('num_selected_seeds', 'N/A')}",
        "",
        "## GD Outcome",
        f"- Best GD primary loss: {_fmt_value(best_primary_loss, precision=6)}",
        f"- Mean loss reduction: {_fmt_value(gd_results.get('metrics', {}).get('mean_loss_reduction'), precision=6)}",
        f"- Max loss reduction: {_fmt_value(gd_results.get('metrics', {}).get('max_loss_reduction'), precision=6)}",
        f"- Min loss reduction: {_fmt_value(gd_results.get('metrics', {}).get('min_loss_reduction'), precision=6)}",
    ]

    _append_mapping_section(report_lines, "## GA Best Loss Components", ga_results.get("best_loss_components", {}))
    _append_mapping_section(report_lines, "## GA Best Physical Metrics", ga_results.get("best_physical_metrics", {}))

    generation_details = ga_results.get("generation_details")
    final_generation_top_individuals: List[Dict[str, Any]] = []
    if isinstance(generation_details, Sequence) and generation_details:
        last_generation = generation_details[-1]
        if isinstance(last_generation, Mapping):
            top_individuals = last_generation.get("top_individuals")
            if isinstance(top_individuals, Sequence):
                for fallback_rank, ranked in enumerate(top_individuals, start=1):
                    if not isinstance(ranked, Mapping):
                        continue
                    payload = dict(ranked)
                    if "rank" not in payload:
                        payload["rank"] = fallback_rank
                    final_generation_top_individuals.append(payload)

            if not final_generation_top_individuals:
                best_payload = {
                    "rank": 1,
                    "primary_fitness": last_generation.get("best_primary_fitness"),
                    "physical_metrics": last_generation.get(
                        "best_physical_metrics",
                        ga_results.get("best_physical_metrics", {}),
                    ),
                }
                if best_payload["primary_fitness"] is not None or isinstance(
                    best_payload.get("physical_metrics"),
                    Mapping,
                ):
                    final_generation_top_individuals.append(best_payload)

                second_payload = {
                    "rank": 2,
                    "primary_fitness": last_generation.get("second_primary_fitness"),
                    "physical_metrics": last_generation.get("second_physical_metrics", {}),
                }
                if second_payload["primary_fitness"] is not None or isinstance(
                    second_payload.get("physical_metrics"),
                    Mapping,
                ):
                    final_generation_top_individuals.append(second_payload)

    if final_generation_top_individuals:
        report_lines.extend(["", "## GA Final Generation Top Individuals"])
        for ranked in final_generation_top_individuals:
            raw_rank = ranked.get("rank", "?")
            try:
                rank = int(raw_rank)
            except (TypeError, ValueError):
                rank = raw_rank

            primary_fitness = _coerce_optional_float(ranked.get("primary_fitness"))
            primary_loss = (
                -float(primary_fitness)
                if primary_fitness is not None
                else None
            )
            metrics = ranked.get("physical_metrics")
            metrics_mapping = metrics if isinstance(metrics, Mapping) else {}

            mean_rssi = _coerce_metric(metrics_mapping, "mean_rss_dbm")
            priority_mean_rssi = _coerce_metric(metrics_mapping, "priority_mean_rss_dbm")
            p5_rssi = _coerce_metric(metrics_mapping, "p5_rss_dbm")
            priority_p5_rssi = _coerce_metric(metrics_mapping, "priority_p5_rss_dbm")

            report_lines.append(
                "- Rank #{rank}: fitness={fitness}, loss={loss}, "
                "mean_rssi={mean_rssi}, priority_mean_rssi={priority_mean_rssi}, "
                "p5_rssi={p5_rssi}, priority_p5_rssi={priority_p5_rssi}".format(
                    rank=rank,
                    fitness=_fmt_value(primary_fitness, precision=6),
                    loss=_fmt_value(primary_loss, precision=6),
                    mean_rssi=_fmt_value(mean_rssi, precision=3),
                    priority_mean_rssi=_fmt_value(priority_mean_rssi, precision=3),
                    p5_rssi=_fmt_value(p5_rssi, precision=3),
                    priority_p5_rssi=_fmt_value(priority_p5_rssi, precision=3),
                )
            )

    selected_seeds = ga_results.get("seeds")
    if isinstance(selected_seeds, Sequence) and selected_seeds:
        report_lines.extend(["", "## GA Seeds Used For GD"])
        for fallback_seed_index, seed in enumerate(selected_seeds, start=1):
            if not isinstance(seed, Mapping):
                continue

            raw_rank = seed.get("rank", fallback_seed_index)
            try:
                seed_rank = int(raw_rank)
            except (TypeError, ValueError):
                seed_rank = fallback_seed_index

            primary_fitness = _coerce_optional_float(seed.get("primary_fitness"))
            primary_loss = (
                -float(primary_fitness)
                if primary_fitness is not None
                else None
            )
            min_distance = _coerce_optional_float(seed.get("min_distance_to_previous"))

            metrics = seed.get("physical_metrics")
            metrics_mapping = metrics if isinstance(metrics, Mapping) else {}
            mean_rssi = _coerce_metric(metrics_mapping, "mean_rss_dbm")
            priority_mean_rssi = _coerce_metric(metrics_mapping, "priority_mean_rss_dbm")

            report_lines.append(
                "- Seed rank #{rank}: fitness={fitness}, loss={loss}, "
                "mean_rssi={mean_rssi}, priority_mean_rssi={priority_mean_rssi}, "
                "min_distance_to_previous={min_distance}".format(
                    rank=seed_rank,
                    fitness=_fmt_value(primary_fitness, precision=6),
                    loss=_fmt_value(primary_loss, precision=6),
                    mean_rssi=_fmt_value(mean_rssi, precision=3),
                    priority_mean_rssi=_fmt_value(priority_mean_rssi, precision=3),
                    min_distance=_fmt_value(min_distance, precision=3),
                )
            )

    if isinstance(global_best, Mapping):
        best_components = _extract_best_loss_components(global_best)
        best_physical_metrics = _extract_best_physical_metrics(global_best)
        report_lines.extend(
            [
                "",
                "## Global Best Task",
                f"- Task ID: {global_best.get('task_id', 'N/A')}",
                f"- Worker ID: {global_best.get('worker_id', 'N/A')}",
                f"- Best primary loss: {_fmt_value(_extract_best_primary_loss(global_best), precision=6)}",
                f"- Best AP position(s): {_fmt_position(_extract_best_position(global_best))}",
            ]
        )
        _append_mapping_section(report_lines, "## Global Best Loss Components", best_components)
        _append_mapping_section(report_lines, "## Global Best Physical Metrics", best_physical_metrics)

    per_seed_analysis = gd_results.get("per_seed_analysis", [])
    if isinstance(per_seed_analysis, list) and per_seed_analysis:
        report_lines.extend(["", "## Per-Seed Improvements"])
        for row in per_seed_analysis:
            report_lines.append(
                "- Seed #{seed}: initial={initial}, best={best}, final={final}, delta={delta}, status={status}".format(
                    seed=row.get("seed_index", "?"),
                    initial=_fmt_value(row.get("initial_primary_loss"), precision=6),
                    best=_fmt_value(row.get("best_primary_loss"), precision=6),
                    final=_fmt_value(row.get("final_primary_loss"), precision=6),
                    delta=_fmt_value(row.get("delta_loss"), precision=6),
                    status=row.get("status", "unknown"),
                )
            )

    return "\n".join(report_lines) + "\n"


def save_memetic_summary_report(summary: Mapping[str, Any], save_path: Path) -> str:
    """Write a Markdown summary report for one memetic run."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(build_memetic_summary_report(summary), encoding="utf-8")
    return str(save_path)