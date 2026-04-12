#!/usr/bin/env python3
"""Sweep AP count and random seeds across memetic and baseline methods.

Instructions
------------
1. Run one command over AP count and seed ranges, for one or many methods.
2. Each (AP, seed) trial now saves full run_experiments-style artifacts under:
    <run_root>/per_trial_runs/aps_XX_seed_YYYY/
    - artifacts/experiment_summary.json
    - artifacts/method_summary.csv
    - artifacts/final_analysis.txt
    - artifacts/<method>_results.json
    - artifacts/<method>_iteration_trace.json
    - artifacts/<method>_iteration_trace.csv
    - artifacts/launcher_summary.json
    - plots/<method>_trend.html
    - plots/method_comparison_trend.html (when >= 2 successful methods)
    - plots/all_methods_primary_loss_static.png
    - plots/<method>_rssi_triplet_static.png
    - plots/all_methods_mean_rss_dbm_static.png
    - plots/all_methods_min_rss_dbm_static.png
    - plots/all_methods_p5_rss_dbm_static.png
3. Sweep-level artifacts are still saved in one run folder:
    - artifacts/sweep_results.json
    - artifacts/plot_artifacts.json
    - plots/iteration_traces/*.png
    - plots/elbow/elbow_primary_loss.png
    - plots/elbow/elbow_mean_rss_dbm.png
    - plots/elbow/elbow_min_rss_dbm.png
    - plots/elbow/elbow_p5_rss_dbm.png
    - plots/elbow/elbow_priority_mean_rss_dbm.png
    - plots/elbow/elbow_priority_min_rss_dbm.png
    - plots/elbow/elbow_priority_p5_rss_dbm.png

Equalization config
-------------------
- Configure equalization under iteration_equalization in the JSON config:
    - enabled: true|false
    - target_iterations: optional integer target
- This sweep runner reads equalization settings only from config.

Example
-------
python scripts/run_all_methods_ap_sweep.py \
     --methods all \
     --config configs/run_experiments_cuda_hrbb.json \
     --ap_min 1 --ap_max 8 \
     --seeds 41 42 43 \
     --output_dir results/ap_sweep

python scripts/run_all_methods_ap_sweep.py \
         --methods all \
         --config configs/run_experiments_cuda_hrbb.smoke_ap_sweep.json \
         --ap_min 2 --ap_max 3 \
         --seeds 301 \
         --output_dir tmp_results/smoke_alignment
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
    _build_final_analysis_report as _runner_build_final_analysis_report,
    _build_weighted_kmeans_sample_weights as _runner_build_weighted_kmeans_sample_weights,
    _enforce_baseline_cuda as _runner_enforce_baseline_cuda,
    _extract_floorplan_coords as _runner_extract_floorplan_coords,
    _extract_method_iteration_trace as _runner_extract_method_iteration_trace,
    _extract_xy_bounds as _runner_extract_xy_bounds,
    _get_optional_mapping as _runner_get_optional_mapping,
    _resolve_demand_config as _runner_resolve_demand_config,
    _resolve_ga_evaluation_params as _runner_resolve_ga_evaluation_params,
    _resolve_gd_params as _runner_resolve_gd_params,
    _resolve_iteration_equalization as _runner_resolve_iteration_equalization,
    _resolve_method_sequence as _runner_resolve_method_sequence,
    _resolve_num_workers as _runner_resolve_num_workers,
    _resolve_objective_params as _runner_resolve_objective_params,
    _resolve_pso_params as _runner_resolve_pso_params,
    _save_comparison_plot as _runner_save_comparison_plot,
    _save_method_artifacts as _runner_save_method_artifacts,
    _write_csv as _runner_write_csv,
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

_ELBOW_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("primary_loss", "AP Count vs Primary Loss", "Primary Loss"),
    ("mean_rss_dbm", "AP Count vs Mean RSSI", "RSSI (dBm)"),
    ("min_rss_dbm", "AP Count vs Minimum RSSI", "RSSI (dBm)"),
    ("p5_rss_dbm", "AP Count vs 5th Percentile RSSI", "RSSI (dBm)"),
    ("priority_mean_rss_dbm", "AP Count vs Priority Mean RSSI", "RSSI (dBm)"),
    ("priority_min_rss_dbm", "AP Count vs Priority Minimum RSSI", "RSSI (dBm)"),
    ("priority_p5_rss_dbm", "AP Count vs Priority 5th Percentile RSSI", "RSSI (dBm)"),
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

_METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "memetic": "GA + GD",
    "pso_gd": "PSO + GD",
    "random_gd": "random + GD",
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
        resolved_sequence = _runner_resolve_method_sequence(token)
        if not resolved_sequence:
            raise ValueError(f"Unsupported method token: {token!r}")

        for method in resolved_sequence:
            if method not in _SINGLE_METHODS:
                raise ValueError(f"Unsupported method token: {token!r}")
            expanded.append(method)

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


def _method_style(method: str) -> Dict[str, str]:
    """Return stable color/marker style for one method."""
    return _METHOD_STYLES.get(method, {"color": "tab:gray", "marker": "o"})


def _display_method_name(method: str) -> str:
    """Return one user-facing method name for plot legends."""
    return _METHOD_DISPLAY_NAMES.get(method, str(method).replace("_", " "))


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


def _extract_method_iteration_trace(
    method: str,
    result_payload: Mapping[str, Any],
    method_config: Optional[Mapping[str, Any]] = None,
) -> TraceRows:
    """Normalize one method payload into iteration trace rows.

    Delegates to the shared run_experiments extractor so AP sweep and
    run_experiments report identical per-method iteration counting.
    """
    trace_rows = _runner_extract_method_iteration_trace(
        method=method,
        result_payload=result_payload,
        method_config=method_config,
    )
    return [
        {str(key): _to_jsonable(value) for key, value in dict(row).items()}
        for row in trace_rows
        if isinstance(row, Mapping)
    ]


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
    method_configs: Mapping[str, Mapping[str, Any]],
    baseline_methods: Sequence[str],
    seed: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Run requested baselines under one shared baseline Ray lifecycle."""
    if not baseline_methods:
        return {}, {}

    reference_method = str(baseline_methods[0])
    reference_config = method_configs.get(reference_method)
    if not isinstance(reference_config, Mapping):
        raise ValueError(
            "method_configs must provide one mapping config for each requested baseline method"
        )

    scene_config_value = reference_config.get("scene_config")
    if not isinstance(scene_config_value, Mapping):
        raise ValueError("Config must contain scene_config mapping for baseline execution")

    scene_config = dict(scene_config_value)
    demand_config = _runner_resolve_demand_config(reference_config)

    num_workers = _runner_resolve_num_workers(reference_config)
    configured_gpu_fraction = float(reference_config.get("gpu_fraction", 0.0))
    gpu_fraction = _runner_enforce_baseline_cuda(
        scene_config=scene_config,
        num_workers=num_workers,
        configured_gpu_fraction=configured_gpu_fraction,
    )

    num_aps = int(reference_config.get("num_aps", 2))
    fixed_z = float(reference_config.get("fixed_z", 3.8))
    optimize_orientation = bool(reference_config.get("optimize_orientation", True))

    x_bounds, y_bounds = _runner_extract_xy_bounds(reference_config)
    scene_config["num_aps"] = int(num_aps)
    scene_config["position_bounds"] = {
        "x_min": float(x_bounds[0]),
        "x_max": float(x_bounds[1]),
        "y_min": float(y_bounds[0]),
        "y_max": float(y_bounds[1]),
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
            verbose=bool(reference_config.get("verbose", True)),
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

            method_config = method_configs.get(method, reference_config)
            if not isinstance(method_config, Mapping):
                method_config = reference_config

            random_params = _runner_get_optional_mapping(method_config, "random_params")
            random_num_samples = int(
                random_params.get("num_samples", method_config.get("num_samples", 100))
            )
            random_gd_params = _runner_get_optional_mapping(method_config, "random_gd_params")
            random_gd_num_samples = int(random_gd_params.get("num_samples", 10))

            objective_params = _runner_resolve_objective_params(method_config)
            ga_evaluation_params = _runner_resolve_ga_evaluation_params(method_config)
            gd_params = _runner_resolve_gd_params(method_config, objective_params=objective_params)
            pso_params = _runner_resolve_pso_params(method_config)

            warmup_eval_params = {
                "samples_per_tx": int(min(int(ga_evaluation_params.get("samples_per_tx", 10_000)), 10_000)),
                "max_depth": int(min(int(ga_evaluation_params.get("max_depth", 3)), 3)),
            }

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
                        config=method_config,
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
                        config=method_config,
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


def _build_running_best_series(trace_rows: Sequence[Mapping[str, Any]]) -> List[float]:
    """Build running-best primary-loss series from one method trace."""
    values: List[float] = []
    running_best = float("inf")

    for row in trace_rows:
        selected: Optional[float] = None
        for key in _PRIMARY_TRACE_KEYS:
            if key not in row:
                continue
            selected = _as_float(row.get(key))
            if selected is not None:
                break

        if selected is None:
            continue

        running_best = min(running_best, float(selected))
        values.append(float(running_best))

    return values


def _metric_key_with_priority_fallback(metric_key: str) -> Optional[str]:
    """Map priority metric keys to corresponding all-region metric keys."""
    if not str(metric_key).startswith("priority_"):
        return None
    fallback = str(metric_key)[len("priority_") :]
    if fallback in ("mean_rss_dbm", "min_rss_dbm", "p5_rss_dbm"):
        return fallback
    return None


def _build_primary_loss_series_for_static(
    method: str,
    trace_rows: Sequence[Mapping[str, Any]],
    result_payload: Mapping[str, Any],
) -> List[float]:
    """Build one primary-loss series for static-plot data export."""
    series = _extract_trace_metric_series(trace_rows, "primary_loss")
    if series:
        return [float(value) for value in series]

    best_primary = _extract_best_primary_loss(method=method, result_payload=result_payload)
    if best_primary is not None:
        return [float(best_primary)]

    return []


def _build_rssi_series_for_static(
    method: str,
    trace_rows: Sequence[Mapping[str, Any]],
    result_payload: Mapping[str, Any],
    metric_key: str,
    fallback_length: int,
) -> List[float]:
    """Build one RSSI series for static-plot data export."""
    resolved_length = max(1, int(fallback_length))
    series = _extract_trace_metric_series(trace_rows, metric_key)
    if series:
        return [float(value) for value in series]

    best_metrics = _extract_best_physical_metrics(method=method, result_payload=result_payload)
    metric_value = _as_float(best_metrics.get(metric_key))
    if metric_value is not None:
        return [float(metric_value)] * int(resolved_length)

    fallback_metric = _metric_key_with_priority_fallback(metric_key)
    if fallback_metric is not None:
        fallback_series = _extract_trace_metric_series(trace_rows, fallback_metric)
        if fallback_series:
            return [float(value) for value in fallback_series]

        fallback_value = _as_float(best_metrics.get(fallback_metric))
        if fallback_value is not None:
            return [float(fallback_value)] * int(resolved_length)

    return []


def _build_per_trial_plot_data(
    methods: Sequence[str],
    method_results: Mapping[str, Mapping[str, Any]],
    method_trace_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Build raw x/y series payload for all per-trial generated plots."""
    trend_keys: Tuple[str, ...] = (
        "running_best_primary_loss",
        "global_best_primary_loss",
        "min_primary_loss",
        "mean_primary_loss",
        "max_primary_loss",
        "primary_loss",
        "swarm_best_primary_loss",
        "swarm_mean_primary_loss",
    )
    all_region_metrics: Tuple[str, ...] = ("mean_rss_dbm", "min_rss_dbm", "p5_rss_dbm")
    priority_metrics: Tuple[str, ...] = (
        "priority_mean_rss_dbm",
        "priority_min_rss_dbm",
        "priority_p5_rss_dbm",
    )

    method_trends: Dict[str, Any] = {}
    comparison_running_best: Dict[str, Any] = {}

    for method in methods:
        trace_rows = list(method_trace_rows.get(method, []))
        if not trace_rows:
            continue

        x_values = list(range(1, len(trace_rows) + 1))
        phases = [str(row.get("phase", "")) for row in trace_rows]
        series_payload: Dict[str, List[Optional[float]]] = {}
        for key in trend_keys:
            series_payload[key] = [_as_float(row.get(key)) for row in trace_rows]

        method_trends[method] = {
            "x": x_values,
            "phase": phases,
            "series": series_payload,
        }

        running_best = _build_running_best_series(trace_rows)
        if running_best:
            comparison_running_best[method] = {
                "x": list(range(1, len(running_best) + 1)),
                "y": [float(value) for value in running_best],
            }

    static_primary_raw: Dict[str, List[float]] = {}
    for method in methods:
        static_primary_raw[method] = _build_primary_loss_series_for_static(
            method=method,
            trace_rows=method_trace_rows.get(method, []),
            result_payload=method_results.get(method, {}),
        )
    primary_target_len = max((len(series) for series in static_primary_raw.values()), default=0)
    static_primary: Dict[str, Any] = {}
    for method in methods:
        expanded = _expand_single_point_series(
            static_primary_raw.get(method, []),
            target_len=primary_target_len,
        )
        if not expanded:
            continue
        static_primary[method] = {
            "x": list(range(1, len(expanded) + 1)),
            "y": [float(value) for value in expanded],
        }

    def _build_per_method_triplet_payload(metric_keys: Sequence[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for method in methods:
            trace_rows = method_trace_rows.get(method, [])
            fallback_len = max(1, len(trace_rows))
            raw_by_metric: Dict[str, List[float]] = {}
            for metric_key in metric_keys:
                raw_by_metric[metric_key] = _build_rssi_series_for_static(
                    method=method,
                    trace_rows=trace_rows,
                    result_payload=method_results.get(method, {}),
                    metric_key=metric_key,
                    fallback_length=fallback_len,
                )

            target_len = max((len(series) for series in raw_by_metric.values()), default=0)
            if target_len < 1:
                continue

            metric_payload: Dict[str, Any] = {}
            for metric_key in metric_keys:
                expanded = _expand_single_point_series(raw_by_metric.get(metric_key, []), target_len=target_len)
                if not expanded:
                    continue
                metric_payload[str(metric_key)] = {
                    "x": list(range(1, len(expanded) + 1)),
                    "y": [float(value) for value in expanded],
                }

            if metric_payload:
                payload[method] = metric_payload

        return payload

    def _build_cross_method_metric_payload(metric_keys: Sequence[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for metric_key in metric_keys:
            raw_by_method: Dict[str, List[float]] = {}
            for method in methods:
                trace_rows = method_trace_rows.get(method, [])
                fallback_len = max(1, len(trace_rows))
                raw_by_method[method] = _build_rssi_series_for_static(
                    method=method,
                    trace_rows=trace_rows,
                    result_payload=method_results.get(method, {}),
                    metric_key=metric_key,
                    fallback_length=fallback_len,
                )

            target_len = max((len(series) for series in raw_by_method.values()), default=0)
            if target_len < 1:
                continue

            per_method_payload: Dict[str, Any] = {}
            for method in methods:
                expanded = _expand_single_point_series(raw_by_method.get(method, []), target_len=target_len)
                if not expanded:
                    continue
                per_method_payload[method] = {
                    "x": list(range(1, len(expanded) + 1)),
                    "y": [float(value) for value in expanded],
                }

            if per_method_payload:
                payload[str(metric_key)] = {
                    "per_method": per_method_payload,
                }

        return payload

    pso_payload = method_results.get("pso_gd", {})
    pso_trace_rows = list(method_trace_rows.get("pso_gd", []))
    pso_trajectory_data: Optional[Dict[str, Any]] = None
    if isinstance(pso_payload, Mapping) and pso_trace_rows:
        pso_steps = list(range(1, len(pso_trace_rows) + 1))
        pso_trajectory_data = {
            "x": pso_steps,
            "series": {
                "primary_loss": _extract_trace_metric_series(pso_trace_rows, "primary_loss"),
                "running_best_primary_loss": [
                    _as_float(row.get("running_best_primary_loss"))
                    for row in pso_trace_rows
                ],
                "mean_rss_dbm": [_as_float(row.get("mean_rss_dbm")) for row in pso_trace_rows],
                "min_rss_dbm": [_as_float(row.get("min_rss_dbm")) for row in pso_trace_rows],
                "p5_rss_dbm": [_as_float(row.get("p5_rss_dbm")) for row in pso_trace_rows],
                "priority_mean_rss_dbm": [
                    _as_float(row.get("priority_mean_rss_dbm"))
                    for row in pso_trace_rows
                ],
                "priority_min_rss_dbm": [
                    _as_float(row.get("priority_min_rss_dbm"))
                    for row in pso_trace_rows
                ],
                "priority_p5_rss_dbm": [
                    _as_float(row.get("priority_p5_rss_dbm"))
                    for row in pso_trace_rows
                ],
            },
            "position_bounds": _to_jsonable(pso_payload.get("position_bounds")),
            "spatial_weights": _to_jsonable(pso_payload.get("spatial_weights")),
        }

    return {
        "method_trends": method_trends,
        "method_comparison_running_best": comparison_running_best,
        "static_primary_loss": static_primary,
        "static_per_method_rssi_triplets": _build_per_method_triplet_payload(all_region_metrics),
        "static_cross_method_rssi": _build_cross_method_metric_payload(all_region_metrics),
        "static_per_method_priority_rssi_triplets": _build_per_method_triplet_payload(priority_metrics),
        "static_cross_method_priority_rssi": _build_cross_method_metric_payload(priority_metrics),
        "pso_gd_trajectory": pso_trajectory_data,
    }


def _save_per_trial_experiment_outputs(
    methods: Sequence[str],
    method_results: Mapping[str, Mapping[str, Any]],
    method_elapsed_sec: Mapping[str, Optional[float]],
    method_configs: Mapping[str, Mapping[str, Any]],
    num_aps: int,
    seed: int,
    run_root: Path,
) -> Dict[str, Any]:
    """Save full run_experiments-style outputs for one (AP, seed) trial."""
    trial_key = f"aps_{int(num_aps):02d}_seed_{int(seed):04d}"
    trial_run_dir = run_root / "per_trial_runs" / trial_key
    trial_artifacts_dir = trial_run_dir / "artifacts"
    trial_plots_dir = trial_run_dir / "plots"
    trial_artifacts_dir.mkdir(parents=True, exist_ok=True)
    trial_plots_dir.mkdir(parents=True, exist_ok=True)

    successful_methods = [
        method
        for method in methods
        if isinstance(method_results.get(method), Mapping)
    ]
    failed_methods = [
        method
        for method in methods
        if method not in successful_methods
    ]

    if not successful_methods:
        return {
            "run_dir": str(trial_run_dir),
            "artifacts_dir": str(trial_artifacts_dir),
            "plots_dir": str(trial_plots_dir),
            "methods_plotted": [],
            "failed_methods": list(failed_methods),
            "method_artifacts": {},
            "summary_json": None,
            "summary_csv": None,
            "final_analysis_txt": None,
            "launcher_summary_json": None,
            "warning": "No successful method payloads for this trial",
        }

    normalized_method_results: Dict[str, Dict[str, Any]] = {
        method: dict(method_results[method])
        for method in successful_methods
    }

    method_artifacts: Dict[str, Dict[str, Optional[str]]] = {}
    method_trace_rows: Dict[str, TraceRows] = {}
    method_summary_rows: List[Dict[str, Any]] = []

    for method in successful_methods:
        payload = normalized_method_results[method]
        method_config = method_configs.get(method)
        method_artifacts[method] = _runner_save_method_artifacts(
            method=method,
            result_payload=payload,
            method_config=method_config,
            artifacts_dir=trial_artifacts_dir,
            plots_dir=trial_plots_dir,
        )

        trace_rows = _extract_method_iteration_trace(
            method=method,
            result_payload=payload,
            method_config=method_config,
        )
        method_trace_rows[method] = trace_rows

        raw_num_iterations = _as_float(payload.get("num_iterations"))
        reported_num_iterations = (
            int(raw_num_iterations)
            if raw_num_iterations is not None
            else len(trace_rows)
        )

        method_summary_rows.append(
            {
                "method": method,
                "elapsed_sec": _as_float(method_elapsed_sec.get(method)),
                "best_primary_loss": _extract_best_primary_loss(method, payload),
                "num_iterations": int(reported_num_iterations),
                "result_json": method_artifacts[method].get("result_json"),
                "trace_csv": method_artifacts[method].get("trace_csv"),
                "trend_plot_html": method_artifacts[method].get("trend_plot_html"),
            }
        )

    comparison_plot_html: Optional[str] = None
    if len(successful_methods) > 1:
        comparison_plot_html = _runner_save_comparison_plot(
            method_traces={
                method: method_trace_rows[method]
                for method in successful_methods
            },
            plot_path=trial_plots_dir / "method_comparison_trend.html",
        )

    static_plot_artifacts = save_static_comparison_plots(
        methods=successful_methods,
        method_results={
            method: normalized_method_results[method]
            for method in successful_methods
        },
        method_trace_rows={
            method: method_trace_rows[method]
            for method in successful_methods
        },
        plots_dir=trial_plots_dir,
        rssi_y_limits=_PER_TRIAL_RSSI_Y_LIMITS,
    )

    per_trial_plot_data = _build_per_trial_plot_data(
        methods=successful_methods,
        method_results=normalized_method_results,
        method_trace_rows=method_trace_rows,
    )
    per_trial_plot_data_path = trial_artifacts_dir / "plot_data.json"
    _write_json(per_trial_plot_data_path, per_trial_plot_data)

    final_analysis_text, ranked_analysis_rows = _runner_build_final_analysis_report(
        run_dir=trial_run_dir,
        methods=successful_methods,
        method_summary_rows=method_summary_rows,
        method_trace_rows=method_trace_rows,
        method_results=normalized_method_results,
    )

    final_analysis_txt_path = trial_artifacts_dir / "final_analysis.txt"
    final_analysis_txt_path.write_text(final_analysis_text, encoding="utf-8")

    summary_payload: Dict[str, Any] = {
        "run_dir": str(trial_run_dir),
        "methods": list(successful_methods),
        "failed_methods": list(failed_methods),
        "method_summary": method_summary_rows,
        "method_artifacts": method_artifacts,
        "analysis": {
            "final_analysis_txt": str(final_analysis_txt_path),
            "ranked_methods": ranked_analysis_rows,
            "static_plot_artifacts": static_plot_artifacts,
            "plot_data_json": str(per_trial_plot_data_path),
        },
    }

    summary_json_path = trial_artifacts_dir / "experiment_summary.json"
    summary_csv_path = trial_artifacts_dir / "method_summary.csv"
    _write_json(summary_json_path, summary_payload)
    _runner_write_csv(summary_csv_path, method_summary_rows)

    launcher_summary_path = trial_artifacts_dir / "launcher_summary.json"
    launcher_payload = {
        "run_dir": str(trial_run_dir),
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
        "comparison_plot_html": comparison_plot_html,
        "final_analysis_txt": str(final_analysis_txt_path),
        "static_plot_artifacts": static_plot_artifacts,
        "plot_data_json": str(per_trial_plot_data_path),
    }
    _write_json(launcher_summary_path, launcher_payload)

    return {
        "run_dir": str(trial_run_dir),
        "artifacts_dir": str(trial_artifacts_dir),
        "plots_dir": str(trial_plots_dir),
        "methods_plotted": list(successful_methods),
        "failed_methods": list(failed_methods),
        "method_artifacts": method_artifacts,
        "comparison_plot_html": comparison_plot_html,
        "static_plot_artifacts": static_plot_artifacts,
        "plot_data_json": str(per_trial_plot_data_path),
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
        "final_analysis_txt": str(final_analysis_txt_path),
        "launcher_summary_json": str(launcher_summary_path),
    }


def _build_iteration_trace_plot_data(
    store: SweepStore,
    methods: Sequence[str],
    ap_values: Sequence[int],
    representative_seed: int,
) -> Dict[str, Any]:
    """Build raw x/y payload for sweep-level iteration-trace plots."""
    payload: Dict[str, Any] = {}

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

        metric_payload: Dict[str, Any] = {}
        for metric_key, title, ylabel in _ITERATION_METRICS:
            series_map: Dict[str, List[float]] = {
                method: _extract_trace_metric_series(trace_rows, metric_key)
                for method, trace_rows in traces_by_method.items()
            }
            max_len = max((len(series) for series in series_map.values()), default=0)

            per_method_payload: Dict[str, Any] = {}
            for method in methods:
                series = series_map.get(method, [])
                if not series:
                    continue
                expanded = _expand_single_point_series(series, target_len=max_len)
                per_method_payload[method] = {
                    "x": list(range(1, len(expanded) + 1)),
                    "y": [float(value) for value in expanded],
                }

            if per_method_payload:
                metric_payload[str(metric_key)] = {
                    "title": str(title),
                    "ylabel": str(ylabel),
                    "per_method": per_method_payload,
                }

        if metric_payload:
            payload[str(int(num_aps))] = {
                "representative_seed": int(representative_seed),
                "metrics": metric_payload,
            }

    return payload


def _build_elbow_plot_data(
    aggregates: AggregateStats,
    methods: Sequence[str],
    ap_values: Sequence[int],
    fixed_y_limits: Optional[Tuple[float, float]],
) -> Dict[str, Any]:
    """Build raw x/y payload for sweep-level elbow plots."""
    payload: Dict[str, Any] = {}

    for metric_key, title, y_label in _ELBOW_METRICS:
        metric_payload: Dict[str, Any] = {
            "title": str(title),
            "y_label": str(y_label),
            "per_method": {},
            "y_limits": None,
        }

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
                y_std.append(float(std_numeric))

            if not x_points:
                continue

            lower_values = [float(mean - std) for mean, std in zip(y_mean, y_std)]
            upper_values = [float(mean + std) for mean, std in zip(y_mean, y_std)]
            lower_envelope.extend(lower_values)
            upper_envelope.extend(upper_values)

            metric_payload["per_method"][method] = {
                "x": [int(value) for value in x_points],
                "mean": [float(value) for value in y_mean],
                "std": [float(value) for value in y_std],
                "lower": lower_values,
                "upper": upper_values,
            }

        if not metric_payload["per_method"]:
            continue

        if fixed_y_limits is not None:
            metric_payload["y_limits"] = {
                "y_min": float(fixed_y_limits[0]),
                "y_max": float(fixed_y_limits[1]),
            }
        else:
            auto_limits = _resolve_auto_y_limits(
                lower_values=lower_envelope,
                upper_values=upper_envelope,
            )
            if auto_limits is not None:
                metric_payload["y_limits"] = {
                    "y_min": float(auto_limits[0]),
                    "y_max": float(auto_limits[1]),
                }

        payload[str(metric_key)] = metric_payload

    return payload


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
                    label=_display_method_name(method),
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
                label=_display_method_name(method),
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

    equalization_config = _runner_get_optional_mapping(base_config, "iteration_equalization")
    config_equalization_enabled = bool(equalization_config.get("enabled", False))
    config_target_iterations: Optional[int] = None
    if "target_iterations" in equalization_config:
        raw_target_iterations = equalization_config.get("target_iterations")
        if raw_target_iterations is not None:
            try:
                config_target_iterations = int(raw_target_iterations)
            except (TypeError, ValueError) as exc:
                raise ValueError("iteration_equalization.target_iterations must be an integer") from exc

    equalization_enabled = bool(config_equalization_enabled)
    target_iterations = config_target_iterations
    if target_iterations is not None and int(target_iterations) <= 0:
        raise ValueError("Target iterations must be > 0")

    if equalization_enabled:
        print("[iterations] equalization enabled")
        print(f"[iterations] target={target_iterations if target_iterations is not None else 'auto'}")

    store: SweepStore = {}
    per_trial_plot_artifacts: Dict[str, Any] = {}
    per_trial_iteration_plans: Dict[str, Dict[str, Any]] = {}

    for num_aps in ap_values:
        for seed in seeds:
            run_name = f"aps_{int(num_aps):02d}_seed_{int(seed)}"
            trial_key = f"aps_{int(num_aps):02d}_seed_{int(seed):04d}"
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
            method_trial_configs, trial_iteration_plan = _runner_resolve_iteration_equalization(
                methods=methods,
                base_config=trial_config,
                enabled=equalization_enabled,
                target_iterations=target_iterations,
            )
            per_trial_iteration_plans[trial_key] = dict(trial_iteration_plan)

            trial_method_results: Dict[str, Dict[str, Any]] = {}
            trial_method_elapsed_sec: Dict[str, Optional[float]] = {}

            if "memetic" in methods:
                memetic_start = time.perf_counter()
                try:
                    memetic_config = method_trial_configs.get("memetic", trial_config)
                    memetic_payload = _run_memetic_for_trial(
                        trial_config=memetic_config,
                        run_root=run_root,
                        num_aps=num_aps,
                        seed=seed,
                    )
                    memetic_elapsed = float(time.perf_counter() - memetic_start)
                    memetic_metrics = _extract_scalar_metrics("memetic", memetic_payload)
                    memetic_trace = _extract_method_iteration_trace(
                        "memetic",
                        memetic_payload,
                        method_config=memetic_config,
                    )

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
                    trial_method_results["memetic"] = dict(memetic_payload)
                    trial_method_elapsed_sec["memetic"] = float(memetic_elapsed)
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
                    trial_method_elapsed_sec["memetic"] = None
                    print(f"[run] memetic failed for AP={num_aps}, seed={seed}: {type(exc).__name__}: {exc}")

            baseline_methods = [method for method in methods if method in _BASELINE_METHODS]
            if baseline_methods:
                baseline_results, baseline_errors = _run_requested_baselines_for_trial(
                    method_configs=method_trial_configs,
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
                            method_config = method_trial_configs.get(method, trial_config)
                            trace_rows = _extract_method_iteration_trace(
                                method,
                                payload,
                                method_config=method_config,
                            )
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
                            trial_method_results[method] = dict(payload)
                            trial_method_elapsed_sec[method] = elapsed_sec
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
                    trial_method_elapsed_sec[method] = None
                    print(f"[run] {method} failed for AP={num_aps}, seed={seed}: {error_text}")

            try:
                per_trial_plot_artifacts[trial_key] = _save_per_trial_experiment_outputs(
                    methods=methods,
                    method_results=trial_method_results,
                    method_elapsed_sec=trial_method_elapsed_sec,
                    method_configs=method_trial_configs,
                    num_aps=num_aps,
                    seed=seed,
                    run_root=run_root,
                )

                warning = per_trial_plot_artifacts[trial_key].get("warning")
                if warning is None:
                    print(
                        "[save] per-trial artifacts saved: "
                        f"AP={int(num_aps)} seed={int(seed)} -> "
                        f"{per_trial_plot_artifacts[trial_key].get('run_dir')}"
                    )
                else:
                    print(
                        "[save] per-trial artifacts skipped: "
                        f"AP={int(num_aps)} seed={int(seed)} ({warning})"
                    )
            except Exception as exc:
                per_trial_plot_artifacts[trial_key] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                print(f"[save] per-trial artifact save failed for AP={num_aps}, seed={seed}: {type(exc).__name__}: {exc}")

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
        "iteration_equalization": {
            "enabled": bool(equalization_enabled),
            "target_iterations": int(target_iterations) if target_iterations is not None else None,
            "config_enabled": bool(config_equalization_enabled),
            "config_target_iterations": (
                int(config_target_iterations)
                if config_target_iterations is not None
                else None
            ),
            "per_trial_plans": per_trial_iteration_plans,
        },
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

    representative_seed = int(seeds[0])
    iteration_trace_plot_data = _build_iteration_trace_plot_data(
        store=store,
        methods=methods,
        ap_values=ap_values,
        representative_seed=representative_seed,
    )
    elbow_plot_data = _build_elbow_plot_data(
        aggregates=aggregates,
        methods=methods,
        ap_values=ap_values,
        fixed_y_limits=fixed_y_limits,
    )

    iteration_trace_plot_data_path = artifacts_dir / "iteration_trace_plot_data.json"
    elbow_plot_data_path = artifacts_dir / "elbow_plot_data.json"
    _write_json(iteration_trace_plot_data_path, iteration_trace_plot_data)
    _write_json(elbow_plot_data_path, elbow_plot_data)

    plot_artifacts: Dict[str, Any] = {
        "per_trial_plots": per_trial_plot_artifacts,
        "iteration_trace_plots": {},
        "elbow_plots": {},
        "iteration_trace_plot_data_json": str(iteration_trace_plot_data_path),
        "elbow_plot_data_json": str(elbow_plot_data_path),
    }
    try:
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
