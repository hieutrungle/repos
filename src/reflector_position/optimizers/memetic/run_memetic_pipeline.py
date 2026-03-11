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
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import ray

from reflector_position.optimizers.memetic.memetic_bridge import (
    generate_gd_tasks_from_seeds,
)
from reflector_position.optimizers.memetic.memetic_ga_logic import (
    MemeticGeneticAlgorithmRunner,
)
from reflector_position.optimizers.memetic.memetic_gd_logic import (
    run_targeted_gd_exploitation,
)
from reflector_position.optimizers.memetic.memetic_plotting import (
    save_memetic_plots,
)
from reflector_position.optimizers.memetic.raw_ray_parallel_optimizer import (
    RawRayActorPoolExecutor,
    RawRayParallelOptimizer,
)
from reflector_position.optimizers.memetic.memetic_summary import (
    save_memetic_summary_report,
)


_OBJECTIVE_PARAM_DEFAULTS: Dict[str, Any] = {
    "alpha": 0.95,
    "beta": 0.05,
    "softmin_temperature": 0.15,
    "coverage_threshold_dbm": -120.0,
    "coverage_temperature": 2.0,
}

_GA_EVALUATION_PARAM_DEFAULTS: Dict[str, Any] = {
    "samples_per_tx": 1_000_000,
    "max_depth": 13,
    "verbose": False,
}

_GD_OPTIMIZATION_PARAM_DEFAULTS: Dict[str, Any] = {
    "num_iterations": 50,
    "learning_rate": 0.1,
    "samples_per_tx": 1_000_000,
    "max_depth": 13,
    "verbose": False,
}

_LEGACY_OBJECTIVE_KEY_MAP = {
    "temperature": "softmin_temperature",
}


def _coerce_mapping(
    config_args: Mapping[str, Any],
    primary_key: str,
    legacy_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a copied config subsection, with optional legacy-key fallback."""
    section = config_args.get(primary_key)
    if section is None and legacy_key is not None:
        section = config_args.get(legacy_key)
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise ValueError(f"'{primary_key}' must be a mapping when provided")
    return dict(section)


def _resolve_objective_params(config_args: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve shared memetic objective parameters from new or legacy schema."""
    objective_params = dict(_OBJECTIVE_PARAM_DEFAULTS)
    explicit_objective_params = _coerce_mapping(config_args, "objective_params")
    objective_params.update(explicit_objective_params)

    legacy_sources = (
        _coerce_mapping(config_args, "ga_evaluation_params", "ga_optimization_params"),
        _coerce_mapping(config_args, "gd_optimization_params", "gd_hyperparams"),
    )
    explicit_objective_keys = set(explicit_objective_params.keys())
    for source in legacy_sources:
        for legacy_key, normalized_key in _LEGACY_OBJECTIVE_KEY_MAP.items():
            if normalized_key not in explicit_objective_keys and legacy_key in source:
                objective_params[normalized_key] = source[legacy_key]
        for key in _OBJECTIVE_PARAM_DEFAULTS:
            if key not in explicit_objective_keys and key in source:
                objective_params[key] = source[key]

    return objective_params


def _resolve_ga_evaluation_params(
    config_args: Mapping[str, Any],
    objective_params: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build GA evaluator worker params from separated config sections."""
    ga_evaluation_params = dict(_GA_EVALUATION_PARAM_DEFAULTS)
    ga_evaluation_params.update(
        _coerce_mapping(config_args, "ga_evaluation_params", "ga_optimization_params")
    )
    ga_evaluation_params.update(objective_params)
    return ga_evaluation_params


def _resolve_gd_optimization_params(
    config_args: Mapping[str, Any],
    objective_params: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build GD optimize params from separated config sections."""
    gd_optimization_params = dict(_GD_OPTIMIZATION_PARAM_DEFAULTS)
    gd_source = _coerce_mapping(config_args, "gd_optimization_params", "gd_hyperparams")

    if "softmin_temperature" not in gd_source and "temperature" in gd_source:
        gd_source["softmin_temperature"] = gd_source["temperature"]

    gd_source.pop("temperature", None)
    gd_source.pop("use_soft_min", None)
    gd_source.pop("shadow_quantile", None)
    gd_source.pop("fairness_loss_type", None)

    gd_optimization_params.update(gd_source)
    gd_optimization_params.update(objective_params)
    return gd_optimization_params


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


##################################################
# Save Summary of All Runs, including GA/GD results, timings, and tabular artifacts.
##################################################


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


def _flatten_row(row: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested mappings into one CSV-friendly row."""
    flattened: Dict[str, Any] = {}
    for key, value in row.items():
        flat_key = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flattened.update(_flatten_row(value, prefix=f"{flat_key}__"))
        else:
            flattened[flat_key] = value
    return flattened


def _collect_fieldnames(rows: List[Mapping[str, Any]]) -> List[str]:
    """Collect stable CSV fieldnames from a sequence of flattened rows."""
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    return fieldnames


def _save_memetic_artifacts(
    summary: Mapping[str, Any],
    config_args: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, str]:
    """Save complete memetic run artifacts (JSON/CSV/plots/report) into output_dir."""
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
    report_md = artifacts_dir / "memetic_report.md"

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
            "best_primary_fitness",
            "mean_primary_fitness",
            "std",
            "time",
        ]
        _write_csv(artifacts_dir / "ga_generation_details.csv", generation_rows, fieldnames)

    seed_rows = gd_results.get("per_seed_analysis", [])
    if isinstance(seed_rows, list) and seed_rows:
        flattened_seed_rows = [_flatten_row(row) for row in seed_rows]
        fieldnames = _collect_fieldnames(flattened_seed_rows)
        _write_csv(artifacts_dir / "gd_per_seed_analysis.csv", flattened_seed_rows, fieldnames)

    plot_artifacts = save_memetic_plots(
        summary=summary,
        output_dir=output_dir,
        position_bounds=config_args.get("position_bounds"),
    )
    report_path = save_memetic_summary_report(summary, report_md)

    artifacts = {
        "output_dir": str(output_dir),
        "artifacts_dir": str(artifacts_dir),
        "plots_dir": str(plots_dir),
        "summary_json": str(summary_json),
        "ga_results_json": str(ga_json),
        "gd_results_json": str(gd_json),
        "run_config_json": str(config_json),
        "global_best_json": str(best_json),
        "report_markdown": report_path,
        "ga_generation_csv": str(artifacts_dir / "ga_generation_details.csv"),
        "gd_per_seed_csv": str(artifacts_dir / "gd_per_seed_analysis.csv"),
    }
    artifacts.update(plot_artifacts)
    return artifacts


def _bind_shared_actor_pool(
    ray_parallel_optimizer: RawRayParallelOptimizer,
    executor: RawRayActorPoolExecutor,
) -> None:
    """Bind an existing raw ActorPool executor into RawRayParallelOptimizer.

    This intentionally reuses private pool state so both GA (executor.map) and
    GD (``RawRayParallelOptimizer.run``) operate on the same hot actors.
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
        - ``objective_params``: shared memetic loss settings for GA and GD
        - ``ga_params``: dict DEAP hyperparameters
        - ``ga_evaluation_params``: dict worker eval params for GA
        - ``k_seeds``: int number of spatial seeds to extract
        - ``d_corr``: float topological distance threshold
        - ``gd_optimization_params``: dict GD optimizer settings
        - ``output_dir``: str|Path base folder for run artifacts
        - ``run_name``: optional run label subfolder name
        - ``verbose``: bool

        Legacy config keys ``ga_optimization_params`` and ``gd_hyperparams``
        are still accepted for backward compatibility.

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

    num_pool_workers = int(config_args.get("num_pool_workers", 3))
    gpu_fraction = float(config_args.get("gpu_fraction", 0.33))

    num_aps = int(config_args.get("num_aps", 2))
    min_ap_separation = float(config_args.get("min_ap_separation", 5.0))
    optimize_orientation = bool(config_args.get("optimize_orientation", True))
    reflector_enabled = bool(config_args.get("reflector_enabled", True))
    focal_z = float(config_args.get("focal_z", 1.5))

    ga_params = dict(config_args.get("ga_params", {}))
    k_seeds = int(config_args.get("k_seeds", 5))
    d_corr = float(config_args.get("d_corr", 3.0))

    objective_params = _resolve_objective_params(config_args)
    ga_evaluation_params = _resolve_ga_evaluation_params(config_args, objective_params)
    gd_optimization_params = _resolve_gd_optimization_params(config_args, objective_params)

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

    executor: Optional[RawRayActorPoolExecutor] = None
    ray_parallel_optimizer: Optional[RawRayParallelOptimizer] = None

    try:
        # -----------------------------------------------------------------
        # Step 1: Resource initialization
        # -----------------------------------------------------------------
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Intentionally keep this executor alive for both GA and GD phases.
        executor = RawRayActorPoolExecutor(
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
            optimization_params=ga_evaluation_params,
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
            gd_optimization_params=gd_optimization_params,
        )

        # Attach scene + baseline metric metadata for Phase-3 analysis.
        for seed, task in zip(seeds, gd_tasks):
            task["scene_config"] = scene_config
            seed_primary_fitness = seed.get("primary_fitness")
            if seed_primary_fitness is not None:
                task["initial_primary_loss"] = -float(seed_primary_fitness)

        # -----------------------------------------------------------------
        # Step 4: Phase 3 - Targeted GD micro-exploitation
        # -----------------------------------------------------------------
        gd_start = time.perf_counter()

        ray_parallel_optimizer = RawRayParallelOptimizer(
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
        "objective_params": {
            "alpha": 0.95,
            "beta": 0.05,
            "softmin_temperature": 0.15,
            "coverage_threshold_dbm": -120.0,
            "coverage_temperature": 2.0,
        },
        "ga_params": {
            "pop_size": 150,
            "n_gen": 50,
            "cxpb": 0.7,
            "mutpb": 0.3,
            "tournsize": 3,
            "hof_size": 20,
        },
        "ga_evaluation_params": {
            "samples_per_tx": 1_000_000,
            "max_depth": 13,
            "verbose": False,
        },
        "k_seeds": 3,
        "d_corr": 5.0,
        "gd_optimization_params": {
            "num_iterations": 50,
            "learning_rate": 0.1,
            "samples_per_tx": 1_000_000,
            "max_depth": 13,
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
    best_primary_loss = (
        output.get("gd_results", {})
        .get("metrics", {})
        .get("best_primary_loss")
    )
    if global_best is not None:
        print(
            "\nGlobal best result: "
            f"task #{global_best.get('task_id')} | "
            f"primary_loss={float(best_primary_loss):.6f}"
        )
