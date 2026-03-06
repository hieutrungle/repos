"""Top-level orchestration for the memetic optimization pipeline.

This module stitches together:
1. Phase 1  - GA macro-exploration (`memetic_ga_logic.py`)
2. Phase 2  - Seed-to-GD bridge (`memetic_bridge.py`)
3. Phase 3  - Targeted GD exploitation (`memetic_gd_logic.py`)

Design goal
-----------
Keep Ray workers/actor pool hot across GA and GD phases to avoid repeated heavy
scene loading (XML parsing, BVH setup, GPU context warmup).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import matplotlib
import ray

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reflector_position.optimizers.memetic.memetic_bridge import (
    generate_gd_tasks_from_seeds,
)
from reflector_position.optimizers.memetic.memetic_ga_logic import (
    MemeticGeneticAlgorithmRunner,
)
from reflector_position.optimizers.memetic.memetic_gd_logic import (
    run_targeted_gd_exploitation,
)
from reflector_position.optimizers.ray_evaluator import RayActorPoolExecutor
from reflector_position.optimizers.ray_parallel_optimizer import RayParallelOptimizer


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge mapping values from updates into base."""
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_json_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration JSON from disk."""
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return payload


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build CLI parser for running the memetic pipeline."""
    parser = argparse.ArgumentParser(
        description="Run memetic optimization pipeline (GA + GD) with optional JSON config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file for run_memetic_optimization().",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments/",
        help="Directory where plots and all run artifacts are saved.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run subfolder name; default is timestamp-based.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging regardless of config file value.",
    )
    return parser


def _to_jsonable(value: Any) -> Any:
    """Recursively coerce arbitrary objects into JSON-serializable structures."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    if hasattr(value, "tolist"):
        try:
            return _to_jsonable(value.tolist())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return _to_jsonable(vars(value))
        except Exception:
            pass

    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    """Write payload as formatted JSON after safe coercion."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: List[Mapping[str, Any]], fieldnames: List[str]) -> None:
    """Write rows to CSV; missing columns are emitted as empty fields."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _plot_ga_training_curve(ga_results: Mapping[str, Any], save_path: Path) -> None:
    """Plot GA max/mean training curves in dBm over generations."""
    details = ga_results.get("generation_details", [])
    if not isinstance(details, list) or len(details) == 0:
        return

    generations = [int(d.get("gen", i)) for i, d in enumerate(details)]
    max_dbm = [float(d.get("max_dbm")) for d in details if d.get("max_dbm") is not None]
    mean_dbm = [float(d.get("mean_dbm")) for d in details if d.get("mean_dbm") is not None]

    if len(max_dbm) != len(generations) or len(mean_dbm) != len(generations):
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, max_dbm, marker="o", linewidth=2.0, label="GA Best (dBm)")
    ax.plot(generations, mean_dbm, marker="s", linewidth=1.6, label="GA Mean (dBm)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("P5 RSS (dBm)")
    ax.set_title("Memetic Phase-1 Training Curve (GA)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_gd_seed_improvements(gd_results: Mapping[str, Any], save_path: Path) -> None:
    """Plot per-seed initial/final metrics and delta improvements for GD."""
    analysis = gd_results.get("per_seed_analysis", [])
    if not isinstance(analysis, list) or len(analysis) == 0:
        return

    seed_ids: List[int] = []
    init_vals: List[float] = []
    final_vals: List[float] = []
    deltas: List[float] = []

    for row in analysis:
        init_dbm = row.get("initial_ga_rss_dbm")
        final_dbm = row.get("best_gd_rss_dbm", row.get("final_gd_rss_dbm"))
        delta_db = row.get("delta_best_improvement_db", row.get("delta_improvement_db"))
        if init_dbm is None or final_dbm is None or delta_db is None:
            continue
        seed_ids.append(int(row.get("seed_index", len(seed_ids))))
        init_vals.append(float(init_dbm))
        final_vals.append(float(final_dbm))
        deltas.append(float(delta_db))

    if not seed_ids:
        return

    x = list(range(len(seed_ids)))
    width = 0.36

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    ax1.bar([i - width / 2 for i in x], init_vals, width=width, label="Initial GA")
    ax1.bar([i + width / 2 for i in x], final_vals, width=width, label="Final GD")
    ax1.set_ylabel("P5 RSS (dBm)")
    ax1.set_title("Memetic Phase-3 Refinement by Seed")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend()

    bars = ax2.bar(x, deltas, width=0.55, color="tab:green")
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.set_xlabel("Seed Index")
    ax2.set_ylabel("Delta Improvement (dB)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(seed_id) for seed_id in seed_ids])
    ax2.grid(True, axis="y", alpha=0.25)

    for bar, delta in zip(bars, deltas):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{delta:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_phase_timing(timings: Mapping[str, Any], save_path: Path) -> None:
    """Plot GA/GD/total timing summary for publication-ready reporting."""
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
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_memetic_artifacts(
    summary: Mapping[str, Any],
    config_args: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, str]:
    """Save complete memetic run artifacts (JSON/CSV/plots) into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = output_dir / "artifacts"
    plots_dir = output_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    ga_results = dict(summary.get("ga_results", {}))
    gd_results = dict(summary.get("gd_results", {}))

    summary_json = artifacts_dir / "memetic_summary.json"
    ga_json = artifacts_dir / "ga_results.json"
    gd_json = artifacts_dir / "gd_results.json"
    config_json = artifacts_dir / "run_config.json"
    best_json = artifacts_dir / "global_best_result.json"

    _write_json(summary_json, summary)
    _write_json(ga_json, ga_results)
    _write_json(gd_json, gd_results)
    _write_json(config_json, config_args)
    _write_json(best_json, summary.get("global_best_result"))

    generation_rows = ga_results.get("generation_details", [])
    if isinstance(generation_rows, list) and generation_rows:
        fieldnames = [
            "gen",
            "nevals",
            "max_dbm",
            "mean_dbm",
            "std",
            "time",
        ]
        _write_csv(artifacts_dir / "ga_generation_details.csv", generation_rows, fieldnames)

    seed_rows = gd_results.get("per_seed_analysis", [])
    if isinstance(seed_rows, list) and seed_rows:
        fieldnames = [
            "seed_index",
            "task_id",
            "worker_id",
            "initial_ga_rss_dbm",
            "best_gd_rss_dbm",
            "final_gd_rss_dbm",
            "delta_improvement_db",
            "delta_best_improvement_db",
            "delta_final_improvement_db",
            "status",
        ]
        _write_csv(artifacts_dir / "gd_per_seed_analysis.csv", seed_rows, fieldnames)

    ga_plot = plots_dir / "ga_training_curve.png"
    gd_plot = plots_dir / "gd_seed_improvements.png"
    timing_plot = plots_dir / "pipeline_timing_breakdown.png"

    _plot_ga_training_curve(ga_results, ga_plot)
    _plot_gd_seed_improvements(gd_results, gd_plot)
    _plot_phase_timing(dict(summary.get("timings", {})), timing_plot)

    return {
        "output_dir": str(output_dir),
        "artifacts_dir": str(artifacts_dir),
        "plots_dir": str(plots_dir),
        "summary_json": str(summary_json),
        "ga_results_json": str(ga_json),
        "gd_results_json": str(gd_json),
        "run_config_json": str(config_json),
        "global_best_json": str(best_json),
        "ga_generation_csv": str(artifacts_dir / "ga_generation_details.csv"),
        "gd_per_seed_csv": str(artifacts_dir / "gd_per_seed_analysis.csv"),
        "ga_training_plot": str(ga_plot),
        "gd_seed_plot": str(gd_plot),
        "timing_plot": str(timing_plot),
    }


def _build_ray_style_gd_plot_payload(
    gd_results: Mapping[str, Any],
    num_workers: int,
) -> Optional[Dict[str, Any]]:
    """Build a RayParallelOptimizer-compatible plotting payload from memetic GD results."""
    all_results = gd_results.get("all_fine_tuned_results", [])
    best_result = gd_results.get("global_best_result")
    metadata = gd_results.get("parallel_run_metadata", {})

    if not isinstance(all_results, list) or len(all_results) == 0 or not isinstance(best_result, Mapping):
        return None

    metrics_dbm = [
        float(r["best_metric_dbm"])
        for r in all_results
        if isinstance(r, Mapping) and r.get("best_metric_dbm") is not None
    ]
    if not metrics_dbm:
        return None

    mean_dbm = sum(metrics_dbm) / len(metrics_dbm)
    std_dbm = math.sqrt(sum((x - mean_dbm) ** 2 for x in metrics_dbm) / len(metrics_dbm))

    aggregate_stats = {
        "mean_metric_dbm": mean_dbm,
        "std_metric_dbm": std_dbm,
        "min_metric_dbm": min(metrics_dbm),
        "max_metric_dbm": max(metrics_dbm),
        "mean_percentile_dbm": mean_dbm,
        "std_percentile_dbm": std_dbm,
        "min_percentile_dbm": min(metrics_dbm),
        "max_percentile_dbm": max(metrics_dbm),
    }

    total_time = metadata.get("total_time")
    if total_time is None:
        total_time = sum(
            float(r.get("time_elapsed", 0.0))
            for r in all_results
            if isinstance(r, Mapping)
        )

    pool_info = dict(metadata.get("pool_info", {})) if isinstance(metadata, Mapping) else {}
    pool_info.setdefault("num_workers", num_workers)
    pool_info.setdefault("num_tasks", len(all_results))

    return {
        "all_results": all_results,
        "best_result": dict(best_result),
        "aggregate_stats": aggregate_stats,
        "pool_info": pool_info,
        "total_time": float(total_time),
    }


def _save_ray_style_gd_plots(
    ray_parallel_optimizer: RayParallelOptimizer,
    gd_results: Mapping[str, Any],
    output_dir: Path,
    position_bounds: Mapping[str, Any],
    num_workers: int,
) -> Dict[str, str]:
    """Save Ray-style GD summary plot and one trajectory plot per task."""
    payload = _build_ray_style_gd_plot_payload(gd_results, num_workers=num_workers)
    if payload is None:
        return {}

    plots_dir = output_dir / "plots"
    traj_dir = plots_dir / "gd_trajectories"
    plots_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    summary_plot_path = plots_dir / "gd_parallel_summary.png"
    ray_parallel_optimizer.save_results_plot(
        payload,
        save_path=str(summary_plot_path),
        metric_name="P5 RSS",
        position_bounds=dict(position_bounds),
        rss_range_dbm=(-130.0, -80.0),
    )

    trajectory_paths = ray_parallel_optimizer.save_task_trajectory_plots(
        payload,
        save_dir=str(traj_dir),
        filename_prefix="gd_task",
        position_bounds=dict(position_bounds),
        rss_range_dbm=(-130.0, -80.0),
    )

    return {
        "gd_ray_style_summary_plot": str(summary_plot_path),
        "gd_trajectory_dir": str(traj_dir),
        "gd_trajectory_count": str(len(trajectory_paths)),
    }


def _bind_shared_actor_pool(
    ray_parallel_optimizer: RayParallelOptimizer,
    executor: RayActorPoolExecutor,
) -> None:
    """Bind an existing RayActorPoolExecutor pool into RayParallelOptimizer.

    This intentionally reuses private pool state so both GA (executor.map) and
    GD (`RayParallelOptimizer.run`) operate on the same hot actors.
    """
    ray_parallel_optimizer._workers = executor._workers  # type: ignore[attr-defined]
    ray_parallel_optimizer._pool = executor._pool  # type: ignore[attr-defined]
    ray_parallel_optimizer._scene_config = dict(executor.scene_config)  # type: ignore[attr-defined]


def run_memetic_optimization(config_args: Mapping[str, Any]) -> Dict[str, Any]:
    """Run full memetic optimization pipeline (GA -> bridge -> targeted GD).

    Parameters
    ----------
    config_args : Mapping[str, Any]
        Pipeline configuration dictionary. Expected top-level keys:

        - ``scene_config``: dict for scene setup (required by Ray workers)
        - ``position_bounds``: dict with ``x_min``, ``x_max``, ``y_min``, ``y_max``
        - ``fixed_z``: float AP height
        - ``num_pool_workers``: int actor pool size
        - ``gpu_fraction``: float GPU fraction per worker
        - ``num_aps``: int number of APs
        - ``min_ap_separation``: float minimum AP separation (meters) for GA
        - ``optimize_orientation``: bool
        - ``reflector_enabled``: bool
        - ``focal_z``: float reflector focal z
        - ``ga_params``: dict DEAP hyperparameters
        - ``ga_optimization_params``: dict worker eval params for GA
        - ``k_seeds``: int number of spatial seeds to extract
        - ``d_corr``: float topological distance threshold
        - ``gd_hyperparams``: dict GD optimization params
        - ``output_dir``: str|Path base folder for run artifacts
        - ``run_name``: optional run label subfolder name
        - ``verbose``: bool

    Returns
    -------
    dict[str, Any]
        Comprehensive pipeline output:
        - ``ga_results``
        - ``gd_results``
        - ``global_best_result``
        - ``timings`` (GA/GD/total seconds)
        - ``counts`` (seed/task counts)
        - ``saved_artifacts`` (paths to saved JSON/CSV/plots)
    """
    verbose = bool(config_args.get("verbose", True))

    scene_config = dict(config_args["scene_config"])
    position_bounds = dict(config_args["position_bounds"])
    fixed_z = float(config_args.get("fixed_z", 3.8))

    num_pool_workers = int(config_args.get("num_pool_workers", 4))
    gpu_fraction = float(config_args.get("gpu_fraction", 0.25))

    num_aps = int(config_args.get("num_aps", 2))
    min_ap_separation = float(config_args.get("min_ap_separation", 2.0))
    optimize_orientation = bool(config_args.get("optimize_orientation", True))
    reflector_enabled = bool(config_args.get("reflector_enabled", False))
    focal_z = float(config_args.get("focal_z", 1.5))

    ga_params = dict(config_args.get("ga_params", {}))
    ga_optimization_params = dict(config_args.get("ga_optimization_params", {}))
    k_seeds = int(config_args.get("k_seeds", 5))
    d_corr = float(config_args.get("d_corr", 5.0))

    gd_hyperparams = dict(config_args.get("gd_hyperparams", {}))

    # ------------------------------------------------------------------
    # GA/GD objective alignment: inject composite-loss hyperparameters
    # from gd_hyperparams into ga_optimization_params so that Ray workers
    # compute the *same* smoothed landscape during the GA evaluation pass.
    # Existing ga_optimization_params values take precedence (explicit
    # override).
    # ------------------------------------------------------------------
    _ALIGNMENT_KEYS = (
        "use_soft_min",
        "temperature",
        "shadow_quantile",
        "fairness_loss_type",
        "alpha",
        "beta",
        "coverage_threshold_dbm",
        "coverage_temperature",
    )
    for _key in _ALIGNMENT_KEYS:
        if _key not in ga_optimization_params and _key in gd_hyperparams:
            ga_optimization_params[_key] = gd_hyperparams[_key]

    output_base_dir = Path(str(config_args.get("output_dir", "results/experiments/")))
    run_name = config_args.get("run_name")
    if run_name:
        run_dir = output_base_dir / str(run_name)
    else:
        run_dir = output_base_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")

    if verbose:
        print("\n" + "=" * 80)
        print("MEMETIC PIPELINE START")
        print("=" * 80)

    pipeline_start = time.perf_counter()

    executor: Optional[RayActorPoolExecutor] = None
    ray_parallel_optimizer: Optional[RayParallelOptimizer] = None

    try:
        # -----------------------------------------------------------------
        # Step 1: Resource initialization
        # -----------------------------------------------------------------
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Intentionally keep this executor alive for both GA and GD phases.
        executor = RayActorPoolExecutor(
            scene_config=scene_config,
            num_workers=num_pool_workers,
            gpu_fraction=gpu_fraction,
            verbose=verbose,
        )

        # -----------------------------------------------------------------
        # Step 2: Phase 1 - GA macro-exploration
        # -----------------------------------------------------------------
        ga_start = time.perf_counter()

        ga_runner = MemeticGeneticAlgorithmRunner(
            position_bounds=position_bounds,
            fixed_z=fixed_z,
            executor_map=executor.map,
            optimize_orientation=optimize_orientation,
            num_aps=num_aps,
            min_ap_separation=min_ap_separation,
            reflector_enabled=reflector_enabled,
            focal_z=focal_z,
            k_seeds=k_seeds,
            d_corr=d_corr,
        )

        ga_results = ga_runner.run(
            optimization_params=ga_optimization_params,
            ga_params=ga_params,
            seed=config_args.get("random_seed"),
            verbose=verbose,
            k_seeds=k_seeds,
            d_corr=d_corr,
        )

        ga_duration = time.perf_counter() - ga_start

        seeds = list(ga_results.get("seeds", []))

        # -----------------------------------------------------------------
        # Step 3: Phase 2 - Bridge seeds -> GD tasks
        # -----------------------------------------------------------------
        gd_tasks = generate_gd_tasks_from_seeds(
            seeds=seeds,
            num_aps=num_aps,
            optimize_orientation=optimize_orientation,
            reflector_enabled=reflector_enabled,
            gd_hyperparams=gd_hyperparams,
        )

        # Attach scene + baseline metric metadata for Phase-3 analysis.
        for seed, task in zip(seeds, gd_tasks):
            task["scene_config"] = scene_config
            seed_metric_dbm = seed.get("fitness_dbm")
            if seed_metric_dbm is not None:
                task["seed_fitness_dbm"] = float(seed_metric_dbm)

        # -----------------------------------------------------------------
        # Step 4: Phase 3 - Targeted GD micro-exploitation
        # -----------------------------------------------------------------
        gd_start = time.perf_counter()

        ray_parallel_optimizer = RayParallelOptimizer(
            num_workers=num_pool_workers,
            gpu_fraction=gpu_fraction,
        )

        # Reuse existing ActorPool from GA executor (hot worker contexts).
        _bind_shared_actor_pool(ray_parallel_optimizer, executor)

        gd_results = run_targeted_gd_exploitation(
            gd_tasks=gd_tasks,
            ray_optimizer=ray_parallel_optimizer,
            verbose=verbose,
        )

        gd_duration = time.perf_counter() - gd_start
        total_duration = time.perf_counter() - pipeline_start

        summary = {
            "ga_results": ga_results,
            "gd_results": gd_results,
            "global_best_result": gd_results.get("global_best_result"),
            "timings": {
                "ga_duration_sec": ga_duration,
                "gd_duration_sec": gd_duration,
                "total_duration_sec": total_duration,
            },
            "counts": {
                "num_ga_seeds": len(seeds),
                "num_gd_tasks": len(gd_tasks),
                "num_gd_results": len(gd_results.get("all_fine_tuned_results", [])),
            },
        }

        saved_artifacts = _save_memetic_artifacts(
            summary=summary,
            config_args=config_args,
            output_dir=run_dir,
        )

        # Additional Ray-style GD visualizations (same look as experiment runner).
        try:
            if ray_parallel_optimizer is not None:
                ray_plot_artifacts = _save_ray_style_gd_plots(
                    ray_parallel_optimizer=ray_parallel_optimizer,
                    gd_results=gd_results,
                    output_dir=run_dir,
                    position_bounds=position_bounds,
                    num_workers=num_pool_workers,
                )
                saved_artifacts.update(ray_plot_artifacts)
        except Exception as exc:
            if verbose:
                print(f"[memetic-pipeline] Warning: failed to save Ray-style GD plots: {exc}")

        summary["saved_artifacts"] = saved_artifacts

        if verbose:
            print("\n" + "=" * 80)
            print("MEMETIC PIPELINE COMPLETE")
            print("=" * 80)
            print(f"GA duration:    {ga_duration:.2f}s")
            print(f"GD duration:    {gd_duration:.2f}s")
            print(f"Total duration: {total_duration:.2f}s")
            print(f"Results saved:  {saved_artifacts['output_dir']}")

        return summary

    finally:
        # Always tear down resources, even on exceptions.
        if executor is not None:
            try:
                executor.shutdown()
            except Exception:
                pass

        # ray_parallel_optimizer shares executor workers above, so avoid
        # calling ray_parallel_optimizer.shutdown() to prevent double-kill.
        if ray.is_initialized():
            ray.shutdown()


def _default_scene_config() -> Dict[str, Any]:
    """Build a default 2-AP + reflector scene configuration for demo runs."""
    scene_path = (
        Path.home()
        / "blender"
        / "models"
        / "building_floor"
        / "building_floor.xml"
    )
    return {
        "scene_path": str(scene_path),
        "frequency": 5.18e9,
        "tx_power_dbm": 5.0,
        "tx_positions": [(7.0, 7.0, 3.8), (23.0, 23.0, 3.8)],
        "reflector_enabled": True,
        "reflector_size": (2.0, 2.0),
        "wall_top_left": [15.0, 34.0, 3.0],
        "wall_bottom_right": [34.0, 34.0, 1.0],
        "focal_point": [20.0, 20.0, 1.5],
        "device": "cuda",
    }


def _default_memetic_config() -> Dict[str, Any]:
    """Default memetic pipeline config used when no --config is provided."""
    return {
        "scene_config": _default_scene_config(),
        "output_dir": "results/experiments/",
        "position_bounds": {
            "x_min": 5.5,
            "x_max": 34.5,
            "y_min": 5.5,
            "y_max": 34.5,
        },
        "fixed_z": 3.8,
        "num_pool_workers": 2,
        "gpu_fraction": 0.5,
        "random_seed": 4,
        "num_aps": 2,
        "min_ap_separation": 5.0,
        "optimize_orientation": True,
        "reflector_enabled": True,
        "focal_z": 1.5,
        "ga_params": {
            "pop_size": 150,
            "n_gen": 50,
            "cxpb": 0.7,
            "mutpb": 0.3,
            "tournsize": 3,
            "hof_size": 20,
        },
        "ga_optimization_params": {
            "samples_per_tx": 1_000_000,
            "max_depth": 13,
            "verbose": False,
        },
        "k_seeds": 3,
        "d_corr": 5.0,
        "gd_hyperparams": {
            "num_iterations": 50,
            "learning_rate": 0.1,
            "samples_per_tx": 1_000_000,
            "max_depth": 13,
            "use_soft_min": True,
            "temperature": 0.15,
            "shadow_quantile": 0.05,
            "fairness_loss_type": "auto",
            "alpha": 0.95,
            "beta": 0.05,
            "coverage_threshold_dbm": -120.0,
            "coverage_temperature": 2.0,
            "verbose": False,
        },
        "verbose": True,
    }


if __name__ == "__main__":
    parser = _build_cli_parser()
    args = parser.parse_args()

    run_config = _default_memetic_config()

    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        loaded = _load_json_config(config_path)
        run_config = _deep_update(run_config, loaded)

    run_config["output_dir"] = args.output_dir
    if args.run_name:
        run_config["run_name"] = args.run_name
    if args.verbose:
        run_config["verbose"] = True

    output = run_memetic_optimization(run_config)
    global_best = output.get("global_best_result")
    if global_best is not None:
        print(
            "\nGlobal best result: "
            f"task #{global_best.get('task_id')} | "
            f"best_metric_dbm={float(global_best.get('best_metric_dbm')):.2f}"
        )
