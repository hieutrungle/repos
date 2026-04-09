#!/usr/bin/env python3
"""Sweep AP count and random seeds across memetic and baseline methods.

Instructions
------------
1. Run one command over AP count and seed ranges, for one or many methods.
2. Each (AP, seed) trial now saves run_experiments-style plots under:
    <run_root>/plots/per_trial/aps_XX_seed_YYYY/
    - <method>_trend.html
    - method_comparison_trend.html (when >= 2 successful methods)
    - all_methods_primary_loss_static.png
    - <method>_rssi_triplet_static.png
    - all_methods_mean_rss_dbm_static.png
    - all_methods_min_rss_dbm_static.png
    - all_methods_p5_rss_dbm_static.png
3. Sweep-level artifacts are still saved in one run folder:
    - artifacts/sweep_results.json
    - artifacts/plot_artifacts.json
    - plots/iteration_traces/*.png
    - plots/elbow/*.png

Example
-------
python scripts/run_all_methods_ap_sweep.py \
     --methods all \
     --config configs/run_experiments_cuda_hrbb.json \
     --ap_min 1 --ap_max 8 \
     --seeds 41 42 43 \
     --output_dir results/ap_sweep
"""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import ray

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from reflector_position.optimizers.baselines import (
    run_kmeans_baseline,
    run_pso_gd_baseline,
    run_random_monte_carlo,
    run_random_multi_start_gd,
    run_weighted_kmeans_baseline,
)
from reflector_position.optimizers.baselines.static_comparison_plotting import (
    save_static_comparison_plots,
)
from reflector_position.optimizers.memetic.raw_ray_parallel_optimizer import (
    RawRayActorPoolExecutor,
    RawRayParallelOptimizer,
)
from reflector_position.optimizers.memetic.run_memetic_pipeline import (
    _default_memetic_config,
    run_memetic_optimization,
)
from run_experiments import (
    _bind_shared_actor_pool as _runner_bind_shared_actor_pool,
    _build_weighted_kmeans_sample_weights as _runner_build_weighted_kmeans_sample_weights,
    _enforce_baseline_cuda as _runner_enforce_baseline_cuda,
    _extract_floorplan_coords as _runner_extract_floorplan_coords,
    _extract_xy_bounds as _runner_extract_xy_bounds,
    _get_optional_mapping as _runner_get_optional_mapping,
    _resolve_demand_config as _runner_resolve_demand_config,
    _resolve_ga_evaluation_params as _runner_resolve_ga_evaluation_params,
    _resolve_gd_params as _runner_resolve_gd_params,
    _resolve_num_workers as _runner_resolve_num_workers,
    _resolve_objective_params as _runner_resolve_objective_params,
    _resolve_pso_params as _runner_resolve_pso_params,
    _save_comparison_plot as _runner_save_comparison_plot,
    _save_method_trend_plot as _runner_save_method_trend_plot,
    _warmup_actor_pool as _runner_warmup_actor_pool,
)

Bounds = Tuple[float, float]
TraceRows = List[Dict[str, Any]]
RunEntry = Dict[str, Any]
SweepStore = Dict[str, Dict[str, Dict[str, RunEntry]]]
AggregateStats = Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]]

_SINGLE_METHODS: List[str] = [
    "memetic",
    "random",
    "kmeans",
    "weighted_kmeans",
    "random_gd",
    "pso_gd",
]
_BASELINE_METHODS: List[str] = [
    "random",
    "kmeans",
    "weighted_kmeans",
    "random_gd",
    "pso_gd",
]

_PRIMARY_TRACE_KEYS: Tuple[str, ...] = (
    "running_best_primary_loss",
    "global_best_primary_loss",
    "min_primary_loss",
    "primary_loss",
    "swarm_best_primary_loss",
)

_RSSI_KEYS: Tuple[str, ...] = (
    "min_rss_dbm",
    "mean_rss_dbm",
    "p5_rss_dbm",
)

_ELBOW_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("primary_loss", "AP Count vs Primary Loss", "Primary Loss"),
    ("mean_rss_dbm", "AP Count vs Mean RSSI", "RSSI (dBm)"),
    ("min_rss_dbm", "AP Count vs Minimum RSSI", "RSSI (dBm)"),
    ("p5_rss_dbm", "AP Count vs 5th Percentile RSSI", "RSSI (dBm)"),
)

_ITERATION_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("primary_loss", "Iteration vs Primary Loss", "Primary Loss"),
    ("min_rss_dbm", "Iteration vs Min RSSI", "RSSI (dBm)"),
    ("mean_rss_dbm", "Iteration vs Mean RSSI", "RSSI (dBm)"),
    ("p5_rss_dbm", "Iteration vs P5 RSSI", "RSSI (dBm)"),
)

_METHOD_STYLES: Dict[str, Dict[str, str]] = {
    "memetic": {"color": "tab:blue", "marker": "o"},
    "random": {"color": "tab:orange", "marker": "s"},
    "kmeans": {"color": "tab:red", "marker": "x"},
    "weighted_kmeans": {"color": "tab:purple", "marker": "D"},
    "random_gd": {"color": "tab:brown", "marker": "v"},
    "pso_gd": {"color": "tab:green", "marker": "^"},
}

_PER_TRIAL_RSSI_Y_LIMITS: Tuple[float, float] = (-100.0, -40.0)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for AP/seed sweep across multiple methods."""
    parser = argparse.ArgumentParser(
        description=(
            "Sweep AP counts and random seeds for requested methods, then "
            "generate iteration and elbow comparison plots."
        )
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help=(
            "Methods to run. Allowed values: memetic random kmeans "
            "weighted_kmeans random_gd pso_gd all_baselines all"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config used as sweep baseline.",
    )
    parser.add_argument(
        "--ap_min",
        type=int,
        required=True,
        help="Minimum AP count to evaluate (inclusive).",
    )
    parser.add_argument(
        "--ap_max",
        type=int,
        required=True,
        help="Maximum AP count to evaluate (inclusive).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Random seeds to evaluate per AP count.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory for sweep artifacts.",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional fixed y-axis minimum for elbow plots.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional fixed y-axis maximum for elbow plots.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logs from underlying optimization runs.",
    )
    return parser.parse_args()


def _normalize_methods(raw_methods: Sequence[str]) -> List[str]:
    """Normalize method tokens into an ordered unique concrete method list."""
    cleaned = [str(token).strip() for token in raw_methods if str(token).strip()]
    if not cleaned:
        raise ValueError("--methods must contain at least one method token")

    expanded: List[str] = []
    for token in cleaned:
        if token == "all":
            expanded.extend(_SINGLE_METHODS)
            continue
        if token == "all_baselines":
            expanded.extend(_BASELINE_METHODS)
            continue
        if token not in _SINGLE_METHODS:
            raise ValueError(f"Unsupported method token: {token!r}")
        expanded.append(token)

    ordered_unique: List[str] = []
    seen = set()
    for method in expanded:
        if method in seen:
            continue
        seen.add(method)
        ordered_unique.append(method)
    return ordered_unique


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON config with object-root validation."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a JSON object: {path}")
    return payload


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively update nested dictionaries."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, Mapping):
            _deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _to_jsonable(value: Any) -> Any:
    """Recursively coerce values into JSON-serializable payloads."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "tolist"):
        try:
            return _to_jsonable(value.tolist())
        except Exception:
            pass

    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass

    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    """Write one JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2)


def _as_float(value: Any) -> Optional[float]:
    """Convert value to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_non_string_sequence(value: Any) -> bool:
    """Return True when value is a sequence and not text/bytes."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _method_style(method: str) -> Dict[str, str]:
    """Return stable color/marker style for one method."""
    return _METHOD_STYLES.get(method, {"color": "tab:gray", "marker": "o"})


def _expand_single_point_series(series: Sequence[float], target_len: int) -> List[float]:
    """Expand one-point data into a horizontal line for visual comparability."""
    values = [float(value) for value in series]
    if target_len <= 1:
        return values
    if len(values) == 1:
        return [values[0]] * int(target_len)
    return values


def _extract_best_primary_loss(method: str, result_payload: Mapping[str, Any]) -> Optional[float]:
    """Extract best primary loss from method payload using robust fallbacks."""
    raw_top_level = _as_float(result_payload.get("best_primary_loss"))
    if raw_top_level is not None:
        return raw_top_level

    if method == "memetic":
        gd_results = result_payload.get("gd_results", {})
        if isinstance(gd_results, Mapping):
            metrics = gd_results.get("metrics", {})
            if isinstance(metrics, Mapping):
                metric_loss = _as_float(metrics.get("best_primary_loss"))
                if metric_loss is not None:
                    return metric_loss

            global_best = gd_results.get("global_best_result", {})
            if isinstance(global_best, Mapping):
                for key in ("best_primary_loss", "primary_loss", "final_primary_loss"):
                    metric_loss = _as_float(global_best.get(key))
                    if metric_loss is not None:
                        return metric_loss

                result_summary = global_best.get("results", {})
                if isinstance(result_summary, Mapping):
                    for key in ("best_primary_loss", "primary_loss", "final_primary_loss"):
                        metric_loss = _as_float(result_summary.get(key))
                        if metric_loss is not None:
                            return metric_loss

    return None


def _extract_best_physical_metrics(
    method: str,
    result_payload: Mapping[str, Any],
) -> Dict[str, float]:
    """Extract best physical metrics from method payloads with memetic fallbacks."""
    candidates: List[Mapping[str, Any]] = []

    top_level_metrics = result_payload.get("best_physical_metrics")
    if isinstance(top_level_metrics, Mapping):
        candidates.append(top_level_metrics)

    if method == "memetic":
        global_best = result_payload.get("global_best_result")
        if isinstance(global_best, Mapping):
            result_summary = global_best.get("results")
            if isinstance(result_summary, Mapping):
                for key in ("best_physical_metrics", "physical_metrics"):
                    value = result_summary.get(key)
                    if isinstance(value, Mapping):
                        candidates.append(value)

            for key in ("best_physical_metrics", "physical_metrics"):
                value = global_best.get(key)
                if isinstance(value, Mapping):
                    candidates.append(value)

        gd_results = result_payload.get("gd_results")
        if isinstance(gd_results, Mapping):
            global_best_gd = gd_results.get("global_best_result")
            if isinstance(global_best_gd, Mapping):
                result_summary = global_best_gd.get("results")
                if isinstance(result_summary, Mapping):
                    for key in ("best_physical_metrics", "physical_metrics"):
                        value = result_summary.get(key)
                        if isinstance(value, Mapping):
                            candidates.append(value)

    for candidate in candidates:
        normalized = {
            str(name): float(metric)
            for name, raw_value in candidate.items()
            if (metric := _as_float(raw_value)) is not None
        }
        if normalized:
            return normalized

    return {}


def _extract_memetic_iteration_trace(result_payload: Mapping[str, Any]) -> TraceRows:
    """Extract memetic iteration trace with primary and RSSI metrics when possible."""
    gd_results = result_payload.get("gd_results", {})
    if not isinstance(gd_results, Mapping):
        return []

    global_best = gd_results.get("global_best_result", {})
    if isinstance(global_best, Mapping):
        history = global_best.get("history", {})
        if isinstance(history, Mapping):
            primary_series = history.get("primary_loss", [])
            physical_series = history.get("physical_metrics", [])
            if _is_non_string_sequence(primary_series):
                trace_rows: TraceRows = []
                running_best = float("inf")
                for idx, raw_loss in enumerate(primary_series, start=1):
                    loss_value = _as_float(raw_loss)
                    if loss_value is None:
                        continue

                    running_best = min(running_best, loss_value)
                    row: Dict[str, Any] = {
                        "iteration": int(idx),
                        "primary_loss": float(loss_value),
                        "running_best_primary_loss": float(running_best),
                    }

                    if _is_non_string_sequence(physical_series) and (idx - 1) < len(physical_series):
                        metric_row = physical_series[idx - 1]
                        if isinstance(metric_row, Mapping):
                            for metric_key in _RSSI_KEYS:
                                metric_value = _as_float(metric_row.get(metric_key))
                                if metric_value is not None:
                                    row[metric_key] = float(metric_value)

                    trace_rows.append(row)

                if trace_rows:
                    return trace_rows

    all_results = gd_results.get("all_fine_tuned_results", [])
    if not isinstance(all_results, list):
        return []

    primary_series_by_seed: List[List[float]] = []
    rssi_series_by_seed: Dict[str, List[List[float]]] = {metric: [] for metric in _RSSI_KEYS}

    for raw_result in all_results:
        if not isinstance(raw_result, Mapping):
            continue

        history = raw_result.get("history", {})
        if not isinstance(history, Mapping):
            continue

        raw_primary_series = history.get("primary_loss", [])
        if not _is_non_string_sequence(raw_primary_series):
            continue

        primary_values: List[float] = []
        for raw_value in raw_primary_series:
            parsed = _as_float(raw_value)
            if parsed is not None:
                primary_values.append(float(parsed))

        if not primary_values:
            continue

        primary_series_by_seed.append(primary_values)

        raw_physical_series = history.get("physical_metrics", [])
        if _is_non_string_sequence(raw_physical_series):
            for metric_key in _RSSI_KEYS:
                metric_values: List[float] = []
                for raw_metric_row in raw_physical_series:
                    if not isinstance(raw_metric_row, Mapping):
                        continue
                    metric_value = _as_float(raw_metric_row.get(metric_key))
                    if metric_value is not None:
                        metric_values.append(float(metric_value))
                if metric_values:
                    rssi_series_by_seed[metric_key].append(metric_values)

    if not primary_series_by_seed:
        return []

    max_len = max(len(series) for series in primary_series_by_seed)
    running_best = float("inf")
    aggregated_rows: TraceRows = []

    for idx in range(max_len):
        primary_bucket = [series[idx] for series in primary_series_by_seed if idx < len(series)]
        if not primary_bucket:
            continue

        min_primary = float(min(primary_bucket))
        mean_primary = float(np.mean(primary_bucket))
        max_primary = float(max(primary_bucket))
        running_best = min(running_best, min_primary)

        row: Dict[str, Any] = {
            "iteration": int(idx + 1),
            "min_primary_loss": min_primary,
            "mean_primary_loss": mean_primary,
            "max_primary_loss": max_primary,
            "running_best_primary_loss": float(running_best),
            "primary_loss": min_primary,
        }

        for metric_key in _RSSI_KEYS:
            metric_bucket = [series[idx] for series in rssi_series_by_seed[metric_key] if idx < len(series)]
            if metric_bucket:
                row[metric_key] = float(np.mean(metric_bucket))

        aggregated_rows.append(row)

    return aggregated_rows


def _extract_method_iteration_trace(method: str, result_payload: Mapping[str, Any]) -> TraceRows:
    """Normalize one method payload into iteration trace rows."""
    if method == "memetic":
        memetic_trace = _extract_memetic_iteration_trace(result_payload)
        if memetic_trace:
            return memetic_trace

    raw_trace = result_payload.get("iteration_trace")
    if not isinstance(raw_trace, list):
        return []

    trace_rows: TraceRows = []
    for index, row in enumerate(raw_trace, start=1):
        if not isinstance(row, Mapping):
            continue
        normalized = dict(row)
        normalized.setdefault("iteration", int(index))
        trace_rows.append({str(key): _to_jsonable(value) for key, value in normalized.items()})
    return trace_rows


def _extract_scalar_metrics(method: str, result_payload: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Extract final scalar metrics for one method trial."""
    physical_metrics = _extract_best_physical_metrics(method=method, result_payload=result_payload)

    metrics: Dict[str, Optional[float]] = {
        "primary_loss": _extract_best_primary_loss(method=method, result_payload=result_payload),
        "min_rss_dbm": _as_float(physical_metrics.get("min_rss_dbm")),
        "mean_rss_dbm": _as_float(physical_metrics.get("mean_rss_dbm")),
        "p5_rss_dbm": _as_float(physical_metrics.get("p5_rss_dbm")),
        "coverage_pct": _as_float(physical_metrics.get("coverage_pct")),
    }

    for alias_key in (
        "priority_min_rss_dbm",
        "priority_mean_rss_dbm",
        "priority_p5_rss_dbm",
    ):
        if alias_key in physical_metrics:
            metrics[alias_key] = _as_float(physical_metrics.get(alias_key))

    return metrics


def _build_trial_config(
    base_config: Mapping[str, Any],
    num_aps: int,
    seed: int,
    verbose_override: bool,
) -> Dict[str, Any]:
    """Build per-(num_aps, seed) config payload."""
    config = deepcopy(dict(base_config))
    config["num_aps"] = int(num_aps)
    config["random_seed"] = int(seed)
    config["verbose"] = bool(verbose_override)

    ga_params = config.get("ga_params")
    if isinstance(ga_params, Mapping):
        ga_payload = dict(ga_params)
        ga_payload["seed"] = int(seed)
        config["ga_params"] = ga_payload

    scene_config = config.get("scene_config")
    if not isinstance(scene_config, Mapping):
        raise ValueError("Config must contain a scene_config mapping")

    updated_scene = dict(scene_config)
    updated_scene["num_aps"] = int(num_aps)

    tx_positions_raw = updated_scene.get("tx_positions")
    if isinstance(tx_positions_raw, Sequence) and not isinstance(tx_positions_raw, (str, bytes)):
        tx_positions = list(tx_positions_raw)
        if len(tx_positions) >= int(num_aps):
            updated_scene["tx_positions"] = [
                list(tx_positions[idx])
                for idx in range(int(num_aps))
            ]

    config["scene_config"] = updated_scene
    return config


def _run_memetic_for_trial(
    trial_config: Mapping[str, Any],
    run_root: Path,
    num_aps: int,
    seed: int,
) -> Dict[str, Any]:
    """Run memetic pipeline for one trial and return raw summary payload."""
    memetic_config = deepcopy(dict(trial_config))
    memetic_config["output_dir"] = str(run_root / "raw_method_runs" / "memetic")
    memetic_config["run_name"] = f"aps_{int(num_aps):02d}_seed_{int(seed):04d}_memetic"
    return run_memetic_optimization(memetic_config)


def _run_requested_baselines_for_trial(
    trial_config: Mapping[str, Any],
    baseline_methods: Sequence[str],
    seed: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Run requested baselines under one shared baseline Ray lifecycle."""
    scene_config_value = trial_config.get("scene_config")
    if not isinstance(scene_config_value, Mapping):
        raise ValueError("Config must contain scene_config mapping for baseline execution")

    scene_config = dict(scene_config_value)
    demand_config = _runner_resolve_demand_config(trial_config)

    num_workers = _runner_resolve_num_workers(trial_config)
    configured_gpu_fraction = float(trial_config.get("gpu_fraction", 0.0))
    gpu_fraction = _runner_enforce_baseline_cuda(
        scene_config=scene_config,
        num_workers=num_workers,
        configured_gpu_fraction=configured_gpu_fraction,
    )

    num_aps = int(trial_config.get("num_aps", 2))
    fixed_z = float(trial_config.get("fixed_z", 3.8))
    optimize_orientation = bool(trial_config.get("optimize_orientation", True))

    x_bounds, y_bounds = _runner_extract_xy_bounds(trial_config)
    scene_config["num_aps"] = int(num_aps)
    scene_config["position_bounds"] = {
        "x_min": float(x_bounds[0]),
        "x_max": float(x_bounds[1]),
        "y_min": float(y_bounds[0]),
        "y_max": float(y_bounds[1]),
    }

    random_params = _runner_get_optional_mapping(trial_config, "random_params")
    random_num_samples = int(random_params.get("num_samples", trial_config.get("num_samples", 100)))

    random_gd_params = _runner_get_optional_mapping(trial_config, "random_gd_params")
    random_gd_num_samples = int(random_gd_params.get("num_samples", 10))

    objective_params = _runner_resolve_objective_params(trial_config)
    ga_evaluation_params = _runner_resolve_ga_evaluation_params(trial_config)
    gd_params = _runner_resolve_gd_params(trial_config, objective_params=objective_params)
    pso_params = _runner_resolve_pso_params(trial_config)

    warmup_eval_params = {
        "samples_per_tx": int(min(int(ga_evaluation_params.get("samples_per_tx", 10_000)), 10_000)),
        "max_depth": int(min(int(ga_evaluation_params.get("max_depth", 3)), 3)),
    }

    results_by_method: Dict[str, Dict[str, Any]] = {}
    errors_by_method: Dict[str, str] = {}

    pool_executor: Optional[RawRayActorPoolExecutor] = None
    ray_optimizer: Optional[RawRayParallelOptimizer] = None
    warmup_done = False

    try:
        pool_executor = RawRayActorPoolExecutor(
            scene_config=scene_config,
            demand_config=demand_config,
            num_workers=num_workers,
            gpu_fraction=gpu_fraction,
            verbose=bool(trial_config.get("verbose", True)),
        )

        ray_optimizer = RawRayParallelOptimizer(
            num_workers=num_workers,
            gpu_fraction=gpu_fraction,
            demand_config=demand_config,
        )
        _runner_bind_shared_actor_pool(ray_optimizer, pool_executor)
        ray_optimizer._scene_config = dict(scene_config)  # type: ignore[attr-defined]

        for method in baseline_methods:
            random.seed(int(seed))
            np.random.seed(int(seed))

            started_at = time.perf_counter()
            try:
                if method == "random":
                    payload = run_random_monte_carlo(
                        ray_pool=pool_executor,
                        num_aps=num_aps,
                        fixed_z=fixed_z,
                        x_bounds=x_bounds,
                        y_bounds=y_bounds,
                        num_samples=random_num_samples,
                        optimize_orientation=optimize_orientation,
                        random_seed=int(seed),
                        loss_kwargs=objective_params,
                        evaluation_params=ga_evaluation_params,
                    )
                elif method == "kmeans":
                    floorplan_coords = _runner_extract_floorplan_coords(
                        config=trial_config,
                        x_bounds=x_bounds,
                        y_bounds=y_bounds,
                    )
                    payload = run_kmeans_baseline(
                        ray_pool=pool_executor,
                        num_aps=num_aps,
                        fixed_z=fixed_z,
                        floorplan_coords=floorplan_coords,
                        optimize_orientation=optimize_orientation,
                        random_seed=int(seed),
                        loss_kwargs=objective_params,
                        evaluation_params=ga_evaluation_params,
                    )
                elif method == "weighted_kmeans":
                    floorplan_coords = _runner_extract_floorplan_coords(
                        config=trial_config,
                        x_bounds=x_bounds,
                        y_bounds=y_bounds,
                    )
                    spatial_weights = _runner_build_weighted_kmeans_sample_weights(
                        floorplan_coords=floorplan_coords,
                        x_bounds=x_bounds,
                        y_bounds=y_bounds,
                        demand_config=demand_config,
                    )
                    payload = run_weighted_kmeans_baseline(
                        ray_pool=pool_executor,
                        num_aps=num_aps,
                        fixed_z=fixed_z,
                        floorplan_coords=floorplan_coords,
                        spatial_weights=spatial_weights,
                        optimize_orientation=optimize_orientation,
                        random_seed=int(seed),
                        loss_kwargs=objective_params,
                        evaluation_params=ga_evaluation_params,
                    )
                elif method == "random_gd":
                    if not warmup_done:
                        _runner_warmup_actor_pool(
                            executor=pool_executor,
                            scene_config=scene_config,
                            num_aps=num_aps,
                            fixed_z=fixed_z,
                            optimize_orientation=optimize_orientation,
                            objective_params=objective_params,
                            warmup_eval_params=warmup_eval_params,
                            x_bounds=x_bounds,
                            y_bounds=y_bounds,
                        )
                        warmup_done = True

                    payload = run_random_multi_start_gd(
                        ray_optimizer=ray_optimizer,
                        num_aps=num_aps,
                        fixed_z=fixed_z,
                        x_bounds=x_bounds,
                        y_bounds=y_bounds,
                        gd_params=gd_params,
                        num_samples=random_gd_num_samples,
                        optimize_orientation=optimize_orientation,
                        random_seed=int(seed),
                    )
                elif method == "pso_gd":
                    if not warmup_done:
                        _runner_warmup_actor_pool(
                            executor=pool_executor,
                            scene_config=scene_config,
                            num_aps=num_aps,
                            fixed_z=fixed_z,
                            optimize_orientation=optimize_orientation,
                            objective_params=objective_params,
                            warmup_eval_params=warmup_eval_params,
                            x_bounds=x_bounds,
                            y_bounds=y_bounds,
                        )
                        warmup_done = True

                    payload = run_pso_gd_baseline(
                        ray_optimizer=ray_optimizer,
                        num_aps=num_aps,
                        fixed_z=fixed_z,
                        x_bounds=x_bounds,
                        y_bounds=y_bounds,
                        pso_params=pso_params,
                        gd_params=gd_params,
                        optimize_orientation=optimize_orientation,
                        random_seed=int(seed),
                        loss_kwargs=objective_params,
                        evaluation_params=ga_evaluation_params,
                    )
                else:
                    raise ValueError(f"Unsupported baseline method: {method!r}")

                elapsed_sec = float(time.perf_counter() - started_at)
                results_by_method[method] = {
                    "payload": payload,
                    "elapsed_sec": elapsed_sec,
                }
            except Exception as exc:
                errors_by_method[method] = f"{type(exc).__name__}: {exc}"

    finally:
        if ray_optimizer is not None:
            try:
                ray_optimizer.shutdown()
            except Exception:
                pass

        if pool_executor is not None:
            try:
                pool_executor.shutdown()
            except Exception:
                pass

        if ray.is_initialized():
            ray.shutdown()

    return results_by_method, errors_by_method


def _set_run_entry(
    store: SweepStore,
    method: str,
    num_aps: int,
    seed: int,
    entry: RunEntry,
) -> None:
    """Insert one run entry into nested method->ap->seed storage."""
    method_bucket = store.setdefault(method, {})
    ap_bucket = method_bucket.setdefault(str(int(num_aps)), {})
    ap_bucket[str(int(seed))] = entry


def _extract_trace_metric_series(trace_rows: Sequence[Mapping[str, Any]], metric_key: str) -> List[float]:
    """Extract one metric series from iteration trace rows."""
    values: List[float] = []
    if metric_key == "primary_loss":
        for row in trace_rows:
            selected: Optional[float] = None
            for key in _PRIMARY_TRACE_KEYS:
                if key not in row:
                    continue
                selected = _as_float(row.get(key))
                if selected is not None:
                    break
            if selected is not None:
                values.append(float(selected))
        return values

    for row in trace_rows:
        selected = _as_float(row.get(metric_key))
        if selected is not None:
            values.append(float(selected))
    return values


def _compute_aggregate_stats(
    store: SweepStore,
    methods: Sequence[str],
    ap_values: Sequence[int],
    metric_keys: Sequence[str],
) -> AggregateStats:
    """Compute per-method, per-AP aggregate stats across seeds for metrics."""
    aggregates: AggregateStats = {}

    for method in methods:
        method_bucket: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
        for num_aps in ap_values:
            ap_entries = store.get(method, {}).get(str(int(num_aps)), {})
            metric_stats: Dict[str, Dict[str, Optional[float]]] = {}

            for metric_key in metric_keys:
                values: List[float] = []
                for run_entry in ap_entries.values():
                    if not isinstance(run_entry, Mapping):
                        continue
                    if run_entry.get("status") != "ok":
                        continue

                    metrics = run_entry.get("metrics", {})
                    if not isinstance(metrics, Mapping):
                        continue

                    metric_value = _as_float(metrics.get(metric_key))
                    if metric_value is not None:
                        values.append(float(metric_value))

                if values:
                    array = np.asarray(values, dtype=np.float64)
                    metric_stats[metric_key] = {
                        "count": float(len(values)),
                        "mean": float(np.mean(array)),
                        "std": float(np.std(array, ddof=0)),
                        "min": float(np.min(array)),
                        "max": float(np.max(array)),
                    }
                else:
                    metric_stats[metric_key] = {
                        "count": 0.0,
                        "mean": None,
                        "std": None,
                        "min": None,
                        "max": None,
                    }

            method_bucket[str(int(num_aps))] = metric_stats

        aggregates[method] = method_bucket

    return aggregates


def _build_per_trial_plot_inputs(
    store: SweepStore,
    methods: Sequence[str],
    num_aps: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, TraceRows]]:
    """Build method payload and trace maps for one trial from sweep storage."""
    method_results: Dict[str, Dict[str, Any]] = {}
    method_trace_rows: Dict[str, TraceRows] = {}

    for method in methods:
        run_entry = store.get(method, {}).get(str(int(num_aps)), {}).get(str(int(seed)), {})
        if not isinstance(run_entry, Mapping):
            continue
        if run_entry.get("status") != "ok":
            continue

        metrics = run_entry.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}

        best_physical_metrics: Dict[str, float] = {}
        for key in (
            "min_rss_dbm",
            "mean_rss_dbm",
            "p5_rss_dbm",
            "coverage_pct",
            "priority_min_rss_dbm",
            "priority_mean_rss_dbm",
            "priority_p5_rss_dbm",
        ):
            metric_value = _as_float(metrics.get(key))
            if metric_value is not None:
                best_physical_metrics[key] = float(metric_value)

        method_results[method] = {
            "best_primary_loss": _as_float(metrics.get("primary_loss")),
            "best_physical_metrics": best_physical_metrics,
        }

        iteration_trace = run_entry.get("iteration_trace")
        if isinstance(iteration_trace, list):
            method_trace_rows[method] = [
                dict(row)
                for row in iteration_trace
                if isinstance(row, Mapping)
            ]
        else:
            method_trace_rows[method] = []

    return method_results, method_trace_rows


def _save_per_trial_plots(
    methods: Sequence[str],
    method_results: Mapping[str, Mapping[str, Any]],
    method_trace_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    num_aps: int,
    seed: int,
    plots_dir: Path,
) -> Dict[str, Any]:
    """Save run_experiments-style plots for one (AP, seed) trial."""
    successful_methods = [method for method in methods if method in method_results]
    trial_key = f"aps_{int(num_aps):02d}_seed_{int(seed):04d}"
    trial_dir = plots_dir / "per_trial" / trial_key
    trial_dir.mkdir(parents=True, exist_ok=True)

    if not successful_methods:
        return {
            "plot_dir": str(trial_dir),
            "methods_plotted": [],
            "method_trend_html": {},
            "comparison_plot_html": None,
            "static_plot_artifacts": {},
            "warning": "No successful method payloads for this trial",
        }

    method_trend_html: Dict[str, Optional[str]] = {}
    for method in successful_methods:
        trend_path = trial_dir / f"{method}_trend.html"
        method_trend_html[method] = _runner_save_method_trend_plot(
            method=method,
            trace_rows=method_trace_rows.get(method, []),
            plot_path=trend_path,
        )

    comparison_plot_html: Optional[str] = None
    if len(successful_methods) > 1:
        comparison_plot_html = _runner_save_comparison_plot(
            method_traces={
                method: method_trace_rows.get(method, [])
                for method in successful_methods
            },
            plot_path=trial_dir / "method_comparison_trend.html",
        )

    static_plot_artifacts = save_static_comparison_plots(
        methods=successful_methods,
        method_results={
            method: method_results[method]
            for method in successful_methods
        },
        method_trace_rows={
            method: method_trace_rows.get(method, [])
            for method in successful_methods
        },
        plots_dir=trial_dir,
        rssi_y_limits=_PER_TRIAL_RSSI_Y_LIMITS,
    )

    return {
        "plot_dir": str(trial_dir),
        "methods_plotted": successful_methods,
        "method_trend_html": method_trend_html,
        "comparison_plot_html": comparison_plot_html,
        "static_plot_artifacts": static_plot_artifacts,
    }


def plot_iteration_traces(
    store: SweepStore,
    methods: Sequence[str],
    ap_values: Sequence[int],
    representative_seed: int,
    plots_dir: Path,
) -> Dict[str, str]:
    """Plot representative iteration convergence traces for each AP count."""
    output_paths: Dict[str, str] = {}
    iter_dir = plots_dir / "iteration_traces"
    iter_dir.mkdir(parents=True, exist_ok=True)

    for num_aps in ap_values:
        traces_by_method: Dict[str, TraceRows] = {}
        for method in methods:
            run_entry = store.get(method, {}).get(str(int(num_aps)), {}).get(str(int(representative_seed)), {})
            if not isinstance(run_entry, Mapping):
                continue
            if run_entry.get("status") != "ok":
                continue

            iteration_trace = run_entry.get("iteration_trace")
            if not isinstance(iteration_trace, list) or not iteration_trace:
                continue

            trace_rows = [
                dict(row)
                for row in iteration_trace
                if isinstance(row, Mapping)
            ]
            if trace_rows:
                traces_by_method[method] = trace_rows

        if not traces_by_method:
            continue

        figure, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
        flattened_axes = list(axes.reshape(-1))

        for axis, (metric_key, title, ylabel) in zip(flattened_axes, _ITERATION_METRICS):
            series_map: Dict[str, List[float]] = {
                method: _extract_trace_metric_series(trace_rows, metric_key)
                for method, trace_rows in traces_by_method.items()
            }
            max_len = max((len(series) for series in series_map.values()), default=0)

            for method in methods:
                series = series_map.get(method, [])
                if not series:
                    continue

                expanded = _expand_single_point_series(series, target_len=max_len)
                x_values = np.arange(1, len(expanded) + 1, dtype=np.int64)
                style = _method_style(method)
                axis.plot(
                    x_values,
                    expanded,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.8,
                    markersize=3.8,
                    label=method,
                )

            axis.set_title(title)
            axis.set_xlabel("Iteration")
            axis.set_ylabel(ylabel)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
            axis.grid(True, alpha=0.28)
            axis.legend(loc="best")

        figure.suptitle(
            f"Iteration Convergence Comparison | AP={int(num_aps)} | Seed={int(representative_seed)}",
            fontsize=13,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

        output_path = iter_dir / f"aps_{int(num_aps):02d}_seed_{int(representative_seed):04d}_convergence.png"
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)

        output_paths[str(int(num_aps))] = str(output_path)

    return output_paths


def _resolve_auto_y_limits(lower_values: Sequence[float], upper_values: Sequence[float]) -> Optional[Tuple[float, float]]:
    """Resolve auto y-limits from lower/upper envelopes with margin."""
    if not lower_values or not upper_values:
        return None

    y_min = float(min(lower_values))
    y_max = float(max(upper_values))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return None

    if y_max <= y_min:
        return (y_min - 1.0, y_max + 1.0)

    margin = 0.05 * (y_max - y_min)
    return (y_min - margin, y_max + margin)


def plot_elbow_curves(
    aggregates: AggregateStats,
    methods: Sequence[str],
    ap_values: Sequence[int],
    plots_dir: Path,
    fixed_y_limits: Optional[Tuple[float, float]],
) -> Dict[str, str]:
    """Plot elbow curves with mean/std shading for all requested methods."""
    output_paths: Dict[str, str] = {}
    elbow_dir = plots_dir / "elbow"
    elbow_dir.mkdir(parents=True, exist_ok=True)

    for metric_key, title, y_label in _ELBOW_METRICS:
        figure, axis = plt.subplots(figsize=(9.0, 5.6))
        plotted_any = False
        lower_envelope: List[float] = []
        upper_envelope: List[float] = []

        for method in methods:
            x_points: List[int] = []
            y_mean: List[float] = []
            y_std: List[float] = []

            method_aggregate = aggregates.get(method, {})
            for num_aps in ap_values:
                ap_stats = method_aggregate.get(str(int(num_aps)), {})
                if not isinstance(ap_stats, Mapping):
                    continue

                metric_stats = ap_stats.get(metric_key, {})
                if not isinstance(metric_stats, Mapping):
                    continue

                mean_value = _as_float(metric_stats.get("mean"))
                if mean_value is None:
                    continue
                std_value = _as_float(metric_stats.get("std"))
                std_numeric = float(std_value) if std_value is not None else 0.0

                x_points.append(int(num_aps))
                y_mean.append(float(mean_value))
                y_std.append(std_numeric)

            if not x_points:
                continue

            x_array = np.asarray(x_points, dtype=np.int64)
            y_array = np.asarray(y_mean, dtype=np.float64)
            std_array = np.asarray(y_std, dtype=np.float64)

            style = _method_style(method)
            axis.plot(
                x_array,
                y_array,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=5.0,
                label=method,
            )
            axis.fill_between(
                x_array,
                y_array - std_array,
                y_array + std_array,
                color=style["color"],
                alpha=0.16,
            )

            plotted_any = True
            lower_envelope.extend((y_array - std_array).tolist())
            upper_envelope.extend((y_array + std_array).tolist())

        if not plotted_any:
            plt.close(figure)
            continue

        axis.set_title(title)
        axis.set_xlabel("Number of APs")
        axis.set_ylabel(y_label)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))

        if fixed_y_limits is not None:
            axis.set_ylim(float(fixed_y_limits[0]), float(fixed_y_limits[1]))
        else:
            auto_limits = _resolve_auto_y_limits(lower_values=lower_envelope, upper_values=upper_envelope)
            if auto_limits is not None:
                axis.set_ylim(float(auto_limits[0]), float(auto_limits[1]))

        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        figure.tight_layout()

        output_path = elbow_dir / f"elbow_{metric_key}.png"
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)

        output_paths[metric_key] = str(output_path)

    return output_paths


def main() -> int:
    """Run full AP/seed sweep for requested methods and generate plots."""
    args = _parse_args()

    if int(args.ap_min) <= 0 or int(args.ap_max) <= 0:
        raise ValueError("--ap_min and --ap_max must be positive")
    if int(args.ap_min) > int(args.ap_max):
        raise ValueError("--ap_min must be <= --ap_max")

    if not args.seeds:
        raise ValueError("--seeds must not be empty")
    seeds = [int(seed) for seed in args.seeds]

    if (args.y_min is None) != (args.y_max is None):
        raise ValueError("Provide both --y-min and --y-max together, or omit both")

    fixed_y_limits: Optional[Tuple[float, float]] = None
    if args.y_min is not None and args.y_max is not None:
        if float(args.y_min) >= float(args.y_max):
            raise ValueError("--y-min must be smaller than --y-max")
        fixed_y_limits = (float(args.y_min), float(args.y_max))

    methods = _normalize_methods(args.methods)
    ap_values = list(range(int(args.ap_min), int(args.ap_max) + 1))

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_config = _default_memetic_config()
    _deep_update(base_config, _load_json(config_path))

    output_base = Path(args.output_dir).expanduser().resolve()
    run_root = output_base / f"all_methods_ap_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifacts_dir = run_root / "artifacts"
    plots_dir = run_root / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] config: {config_path}")
    print(f"[sweep] run root: {run_root}")
    print(f"[sweep] methods: {methods}")
    print(f"[sweep] AP range: {ap_values[0]}..{ap_values[-1]}")
    print(f"[sweep] seeds: {seeds}")
    if fixed_y_limits is None:
        print("[sweep] elbow y-range: auto (unified per metric across all methods)")
    else:
        print(f"[sweep] elbow y-range: fixed {fixed_y_limits}")

    store: SweepStore = {}

    for num_aps in ap_values:
        for seed in seeds:
            run_name = f"aps_{int(num_aps):02d}_seed_{int(seed)}"
            print(
                f"[run] start num_aps={int(num_aps)}, seed={int(seed)}, "
                f"run_name={run_name}"
            )
            trial_config = _build_trial_config(
                base_config=base_config,
                num_aps=num_aps,
                seed=seed,
                verbose_override=bool(args.verbose),
            )

            if "memetic" in methods:
                memetic_start = time.perf_counter()
                try:
                    memetic_payload = _run_memetic_for_trial(
                        trial_config=trial_config,
                        run_root=run_root,
                        num_aps=num_aps,
                        seed=seed,
                    )
                    memetic_elapsed = float(time.perf_counter() - memetic_start)
                    memetic_metrics = _extract_scalar_metrics("memetic", memetic_payload)
                    memetic_trace = _extract_method_iteration_trace("memetic", memetic_payload)

                    _set_run_entry(
                        store=store,
                        method="memetic",
                        num_aps=num_aps,
                        seed=seed,
                        entry={
                            "status": "ok",
                            "elapsed_sec": memetic_elapsed,
                            "metrics": memetic_metrics,
                            "iteration_trace": memetic_trace,
                            "saved_artifacts": memetic_payload.get("saved_artifacts", {}),
                        },
                    )
                except Exception as exc:
                    _set_run_entry(
                        store=store,
                        method="memetic",
                        num_aps=num_aps,
                        seed=seed,
                        entry={
                            "status": "error",
                            "elapsed_sec": None,
                            "metrics": {},
                            "iteration_trace": [],
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        },
                    )
                    print(f"[run] memetic failed for AP={num_aps}, seed={seed}: {type(exc).__name__}: {exc}")

            baseline_methods = [method for method in methods if method in _BASELINE_METHODS]
            if baseline_methods:
                baseline_results, baseline_errors = _run_requested_baselines_for_trial(
                    trial_config=trial_config,
                    baseline_methods=baseline_methods,
                    seed=seed,
                )

                for method in baseline_methods:
                    method_result = baseline_results.get(method)
                    if isinstance(method_result, Mapping):
                        payload = method_result.get("payload", {})
                        elapsed_sec = _as_float(method_result.get("elapsed_sec"))
                        if isinstance(payload, Mapping):
                            metrics = _extract_scalar_metrics(method, payload)
                            trace_rows = _extract_method_iteration_trace(method, payload)
                            _set_run_entry(
                                store=store,
                                method=method,
                                num_aps=num_aps,
                                seed=seed,
                                entry={
                                    "status": "ok",
                                    "elapsed_sec": elapsed_sec,
                                    "metrics": metrics,
                                    "iteration_trace": trace_rows,
                                },
                            )
                            continue

                    error_text = baseline_errors.get(method, "Unknown baseline execution error")
                    _set_run_entry(
                        store=store,
                        method=method,
                        num_aps=num_aps,
                        seed=seed,
                        entry={
                            "status": "error",
                            "elapsed_sec": None,
                            "metrics": {},
                            "iteration_trace": [],
                            "error": error_text,
                        },
                    )
                    print(f"[run] {method} failed for AP={num_aps}, seed={seed}: {error_text}")

            trial_mean_rss_values: List[float] = []
            for method in methods:
                run_entry = store.get(method, {}).get(str(int(num_aps)), {}).get(str(int(seed)), {})
                if not isinstance(run_entry, Mapping):
                    continue
                if run_entry.get("status") != "ok":
                    continue

                metrics = run_entry.get("metrics", {})
                if not isinstance(metrics, Mapping):
                    continue

                mean_rss_dbm = _as_float(metrics.get("mean_rss_dbm"))
                if mean_rss_dbm is not None:
                    trial_mean_rss_values.append(float(mean_rss_dbm))

            if trial_mean_rss_values:
                trial_mean_rss_dbm = float(np.mean(trial_mean_rss_values))
                print(
                    f"[run] done  num_aps={int(num_aps)}, seed={int(seed)}, "
                    f"mean_rss_dbm={trial_mean_rss_dbm:.4f}"
                )
            else:
                print(
                    f"[run] done  num_aps={int(num_aps)}, seed={int(seed)}, "
                    "mean_rss_dbm=nan"
                )

    metric_keys = {
        "primary_loss",
        "min_rss_dbm",
        "mean_rss_dbm",
        "p5_rss_dbm",
        "coverage_pct",
        "priority_min_rss_dbm",
        "priority_mean_rss_dbm",
        "priority_p5_rss_dbm",
    }

    aggregates = _compute_aggregate_stats(
        store=store,
        methods=methods,
        ap_values=ap_values,
        metric_keys=sorted(metric_keys),
    )

    sweep_results_path = artifacts_dir / "sweep_results.json"
    sweep_payload = {
        "generated_at": datetime.now().isoformat(),
        "config_path": str(config_path),
        "run_root": str(run_root),
        "methods": methods,
        "ap_values": ap_values,
        "seeds": seeds,
        "fixed_elbow_y_limits": {
            "y_min": fixed_y_limits[0],
            "y_max": fixed_y_limits[1],
        }
        if fixed_y_limits is not None
        else None,
        "results": store,
        "aggregates": aggregates,
    }
    _write_json(sweep_results_path, sweep_payload)
    print(f"[save] sweep results: {sweep_results_path}")

    per_trial_plot_artifacts: Dict[str, Any] = {}
    for num_aps in ap_values:
        for seed in seeds:
            trial_key = f"aps_{int(num_aps):02d}_seed_{int(seed):04d}"
            try:
                trial_method_results, trial_method_traces = _build_per_trial_plot_inputs(
                    store=store,
                    methods=methods,
                    num_aps=num_aps,
                    seed=seed,
                )
                per_trial_plot_artifacts[trial_key] = _save_per_trial_plots(
                    methods=methods,
                    method_results=trial_method_results,
                    method_trace_rows=trial_method_traces,
                    num_aps=num_aps,
                    seed=seed,
                    plots_dir=plots_dir,
                )

                warning = per_trial_plot_artifacts[trial_key].get("warning")
                if warning is None:
                    print(
                        "[plot] per-trial plots saved: "
                        f"AP={int(num_aps)} seed={int(seed)} -> "
                        f"{per_trial_plot_artifacts[trial_key].get('plot_dir')}"
                    )
                else:
                    print(
                        "[plot] per-trial plots skipped: "
                        f"AP={int(num_aps)} seed={int(seed)} ({warning})"
                    )
            except Exception as exc:
                per_trial_plot_artifacts[trial_key] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                print(f"[plot] per-trial plotting failed for AP={num_aps}, seed={seed}: {type(exc).__name__}: {exc}")

    plot_artifacts: Dict[str, Any] = {
        "per_trial_plots": per_trial_plot_artifacts,
        "iteration_trace_plots": {},
        "elbow_plots": {},
    }
    try:
        representative_seed = int(seeds[0])
        plot_artifacts["iteration_trace_plots"] = plot_iteration_traces(
            store=store,
            methods=methods,
            ap_values=ap_values,
            representative_seed=representative_seed,
            plots_dir=plots_dir,
        )
        plot_artifacts["elbow_plots"] = plot_elbow_curves(
            aggregates=aggregates,
            methods=methods,
            ap_values=ap_values,
            plots_dir=plots_dir,
            fixed_y_limits=fixed_y_limits,
        )
    except Exception as exc:
        plot_artifacts["plotting_error"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        print(f"[plot] failed: {type(exc).__name__}: {exc}")

    plot_artifacts_path = artifacts_dir / "plot_artifacts.json"
    _write_json(plot_artifacts_path, plot_artifacts)
    print(f"[save] plot artifacts: {plot_artifacts_path}")

    print("[done] sweep complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
