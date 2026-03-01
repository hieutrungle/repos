"""
Unified experiment runner for ``ray_parallel_example.py``.

This script replaces both:
- ``examples/ray_hparam_tuning.py``
- ``examples/ray_parallel_sweep_runner.py``

Key behavior:
1. Each trial runs exactly ONE method: ``gd``, ``gs``, or ``ga``.
2. Each trial runs exactly ONE mode: ``1ap`` or ``2ap``.
3. Supports explicit ``trials`` and automatic trial generation from
   ``sweep_groups`` (Cartesian product over hyperparameter grids).

Usage:
    python examples/ray_experiment_runner.py \
        --config examples/ray_experiment_runner_config.example.json

Generate expanded config only (no execution):
    python examples/ray_experiment_runner.py \
        --config examples/ray_experiment_runner_config.example.json \
        --generate-only \
        --generated-config examples/generated_trials.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

try:
    import ray
except ModuleNotFoundError:
    ray = None

try:
    import ray_parallel_example as exp
except ModuleNotFoundError:
    exp = None


VALID_METHODS = {"gd", "gs", "ga"}
VALID_MODES = {"1ap", "2ap", "2ap_reflector"}

# Default reflector geometry — matches ray_parallel_example.py defaults
_DEFAULT_REFLECTOR = {
    "wall_top_left": [15.0, 34.0, 3.0],
    "wall_bottom_right": [34.0, 34.0, 1.0],
    "reflector_size": [2.0, 2.0],
    "focal_z": 1.5,
    "target_quantile": 0.05,
}


class TeeStream:
    """Write to both terminal and file."""

    def __init__(self, terminal_stream, file_stream):
        self.terminal_stream = terminal_stream
        self.file_stream = file_stream

    def write(self, data: str) -> None:
        self.terminal_stream.write(data)
        self.file_stream.write(data)

    def flush(self) -> None:
        self.terminal_stream.flush()
        self.file_stream.flush()


def _merge_dict(base: dict[str, Any], updates: dict[str, Any] | None) -> dict[str, Any]:
    result = deepcopy(base)
    if not updates:
        return result
    for key, value in updates.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)


def _abbr_float(value: float) -> str:
    s = f"{value:g}"
    return s.replace("-", "m").replace(".", "")


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = mapping
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _validate_trial(trial: dict[str, Any], trial_idx: int) -> None:
    method = trial.get("method")
    mode = trial.get("mode")

    if method not in VALID_METHODS:
        raise ValueError(
            f"Trial #{trial_idx}: invalid method '{method}'. "
            f"Expected one of {sorted(VALID_METHODS)}"
        )
    if mode not in VALID_MODES:
        raise ValueError(
            f"Trial #{trial_idx}: invalid mode '{mode}'. "
            f"Expected one of {sorted(VALID_MODES)}"
        )


def _default_trial_name(trial_idx: int, trial: dict[str, Any]) -> str:
    method = trial["method"]
    mode = trial["mode"]
    seed = trial.get("random_seed", exp.RANDOM_SEED)

    suffix_parts = [f"seed{seed}"]

    if method == "gd":
        overrides = trial.get("gd_optimization_overrides", {})
        suffix_parts.extend(
            [
                f"lr{_abbr_float(float(overrides.get('learning_rate', 0.5)))}",
                f"t{_abbr_float(float(overrides.get('temperature', 0.15)))}",
                f"a{_abbr_float(float(overrides.get('alpha', 0.9)))}",
                f"b{_abbr_float(float(overrides.get('beta', 0.1)))}",
            ]
        )
        if mode == "2ap_reflector":
            flt = trial.get("gd_fairness_loss_type", "auto")
            suffix_parts.append(f"flt{flt}")
    elif method == "gs":
        suffix_parts.append(f"gr{_abbr_float(float(trial.get('gs_grid_resolution', 1.0)))}")
        if mode in {"2ap", "2ap_reflector"}:
            suffix_parts.append(f"r{int(trial.get('gs_num_rounds', exp.ALTERNATING_ROUNDS))}")
        if mode == "2ap_reflector":
            suffix_parts.append(f"or{int(trial.get('gs_outer_rounds', 3))}")
            suffix_parts.append(f"q{_abbr_float(float(trial.get('gs_target_quantile', 0.05)))}")
    elif method == "ga":
        ga_params = trial.get("ga_params", {})
        suffix_parts.extend(
            [
                f"p{int(ga_params.get('pop_size', exp.GA_PARAMS['pop_size']))}",
                f"g{int(ga_params.get('n_gen', exp.GA_PARAMS['n_gen']))}",
                f"m{_abbr_float(float(ga_params.get('mutpb', exp.GA_PARAMS['mutpb'])))}",
            ]
        )
        if mode == "2ap_reflector":
            suffix_parts.append(f"q{_abbr_float(float(trial.get('ga_target_quantile', 0.05)))}")

    suffix = "_".join(suffix_parts)
    return f"trial_{trial_idx:03d}_{method}_{mode}_{suffix}"


def _build_trials(config: dict[str, Any]) -> list[dict[str, Any]]:
    shared = config.get("shared", {})
    explicit_trials = config.get("trials", [])
    sweep_groups = config.get("sweep_groups", [])

    all_trials: list[dict[str, Any]] = []

    for trial in explicit_trials:
        # Skip comment-only entries (no method/mode)
        if "method" not in trial and "mode" not in trial:
            continue
        merged = _merge_dict(shared, trial)
        all_trials.append(merged)

    for group in sweep_groups:
        base = _merge_dict(shared, group.get("base", {}))

        method = group.get("method")
        mode_values = _ensure_list(group.get("mode", "1ap"))
        seed_values = _ensure_list(group.get("random_seed", exp.RANDOM_SEED))

        grid = group.get("grid", {})
        grid_keys = list(grid.keys())
        grid_values = [_ensure_list(grid[key]) for key in grid_keys]

        if not grid_keys:
            grid_product = [tuple()]
        else:
            grid_product = list(product(*grid_values))

        name_prefix = group.get("name_prefix")

        for mode in mode_values:
            for seed in seed_values:
                for combo in grid_product:
                    trial = deepcopy(base)
                    trial["method"] = method
                    trial["mode"] = mode
                    trial["random_seed"] = int(seed)

                    for key, value in zip(grid_keys, combo):
                        _set_nested(trial, key, value)

                    if name_prefix and "name" not in trial:
                        combo_suffix = []
                        for key, value in zip(grid_keys, combo):
                            leaf = key.split(".")[-1]
                            combo_suffix.append(f"{leaf}{_abbr_float(float(value))}" if isinstance(value, float) else f"{leaf}{value}")
                        trial["name"] = _sanitize_name(
                            f"{name_prefix}_{mode}_seed{seed}_" + "_".join(combo_suffix)
                        )

                    all_trials.append(trial)

    for i, trial in enumerate(all_trials, start=1):
        _validate_trial(trial, i)
        if "name" not in trial:
            trial["name"] = _default_trial_name(i, trial)
        trial["name"] = _sanitize_name(str(trial["name"]))

    return all_trials


def _get_reflector_cfg(trial: dict[str, Any]) -> dict[str, Any]:
    """Extract reflector geometry from trial, falling back to defaults."""
    return {
        "wall_top_left": trial.get(
            "reflector_wall_top_left", _DEFAULT_REFLECTOR["wall_top_left"]
        ),
        "wall_bottom_right": trial.get(
            "reflector_wall_bottom_right", _DEFAULT_REFLECTOR["wall_bottom_right"]
        ),
        "reflector_size": tuple(
            trial.get("reflector_size", _DEFAULT_REFLECTOR["reflector_size"])
        ),
        "focal_z": float(
            trial.get("reflector_focal_z", _DEFAULT_REFLECTOR["focal_z"])
        ),
        "target_quantile": float(
            trial.get("reflector_target_quantile", _DEFAULT_REFLECTOR["target_quantile"])
        ),
    }


def _run_trial_method(trial: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    method = trial["method"]
    mode = trial["mode"]
    seed = int(trial.get("random_seed", exp.RANDOM_SEED))

    num_pool_workers = int(trial.get("num_pool_workers", exp.NUM_POOL_WORKERS))
    gpu_fraction = float(trial.get("gpu_fraction", exp.GPU_FRACTION))

    # -----------------------------------------------------------------
    # Gradient Descent
    # -----------------------------------------------------------------
    if method == "gd":
        gd_num_tasks = int(trial.get("gd_num_tasks", 100))
        gd_num_iterations = int(trial.get("gd_num_iterations", 50))
        gd_samples_per_tx = int(trial.get("gd_samples_per_tx", 1_000_000))
        gd_repulsion_weight = float(trial.get("gd_repulsion_weight", 0.3))
        gd_fairness_loss_type = str(trial.get("gd_fairness_loss_type", "auto"))

        parallel_opt = exp.RayParallelOptimizer(
            num_workers=num_pool_workers,
            gpu_fraction=gpu_fraction,
        )
        try:
            if mode == "2ap_reflector":
                rcfg = _get_reflector_cfg(trial)
                return exp.example_parallel_gd_2ap_with_reflector(
                    parallel_opt,
                    num_tasks=gd_num_tasks,
                    num_iterations=gd_num_iterations,
                    repulsion_weight=gd_repulsion_weight,
                    fairness_loss_type=gd_fairness_loss_type,
                    output_dir=str(output_dir),
                    random_seed=seed,
                    wall_top_left=rcfg["wall_top_left"],
                    wall_bottom_right=rcfg["wall_bottom_right"],
                    reflector_size=rcfg["reflector_size"],
                    target_z=rcfg["focal_z"],
                    optimization_overrides=trial.get("gd_optimization_overrides"),
                )

            if mode == "1ap":
                return exp.example_parallel_gradient_descent(
                    parallel_opt,
                    num_aps=1,
                    num_tasks=gd_num_tasks,
                    num_iterations=gd_num_iterations,
                    samples_per_tx=gd_samples_per_tx,
                    output_dir=str(output_dir),
                    scene_config=exp.SCENE_CONFIG,
                    random_seed=seed,
                    optimization_overrides=trial.get("gd_optimization_overrides"),
                )

            # mode == "2ap"
            return exp.example_parallel_gradient_descent(
                parallel_opt,
                num_aps=2,
                num_tasks=gd_num_tasks,
                num_iterations=gd_num_iterations,
                repulsion_weight=gd_repulsion_weight,
                samples_per_tx=gd_samples_per_tx,
                output_dir=str(output_dir),
                scene_config=exp.SCENE_CONFIG_2AP,
                random_seed=seed,
                optimization_overrides=trial.get("gd_optimization_overrides"),
            )
        finally:
            parallel_opt.shutdown()

    # -----------------------------------------------------------------
    # Grid Search
    # -----------------------------------------------------------------
    if method == "gs":
        parallel_opt = exp.RayParallelOptimizer(
            num_workers=num_pool_workers,
            gpu_fraction=gpu_fraction,
        )
        try:
            if mode == "2ap_reflector":
                rcfg = _get_reflector_cfg(trial)
                return exp.example_parallel_grid_search_2ap_with_reflector(
                    parallel_opt,
                    grid_resolution=float(trial.get("gs_grid_resolution", 1.0)),
                    num_rounds=int(trial.get("gs_num_rounds", exp.ALTERNATING_ROUNDS)),
                    outer_rounds=int(trial.get("gs_outer_rounds", 3)),
                    output_dir=str(output_dir),
                    scene_config=exp.SCENE_CONFIG_2AP,
                    wall_top_left=rcfg["wall_top_left"],
                    wall_bottom_right=rcfg["wall_bottom_right"],
                    reflector_size=rcfg["reflector_size"],
                    target_z=rcfg["focal_z"],
                    target_quantile=rcfg["target_quantile"],
                    u_steps=int(trial.get("gs_u_steps", 3)),
                    v_steps=int(trial.get("gs_v_steps", 3)),
                    target_resolution=float(trial.get("gs_target_resolution", 10.0)),
                    min_ap_separation=float(trial.get("gs_min_ap_separation", 10.0)),
                )

            if mode == "1ap":
                return exp.example_parallel_grid_search(
                    parallel_opt,
                    grid_resolution=float(trial.get("gs_grid_resolution", 1.0)),
                    output_dir=str(output_dir),
                    scene_config=exp.SCENE_CONFIG,
                )

            # mode == "2ap"
            return exp.example_parallel_grid_search_2ap(
                parallel_opt,
                grid_resolution=float(trial.get("gs_grid_resolution", 1.0)),
                num_rounds=int(trial.get("gs_num_rounds", exp.ALTERNATING_ROUNDS)),
                output_dir=str(output_dir),
                scene_config=exp.SCENE_CONFIG_2AP,
            )
        finally:
            parallel_opt.shutdown()

    # -----------------------------------------------------------------
    # Genetic Algorithm
    # -----------------------------------------------------------------
    # method == "ga"
    ga_params = _merge_dict(exp.GA_PARAMS, trial.get("ga_params"))

    if mode == "2ap_reflector":
        rcfg = _get_reflector_cfg(trial)
        target_quantile = float(
            trial.get("ga_target_quantile", rcfg["target_quantile"])
        )
        focal_z = float(trial.get("ga_focal_z", rcfg["focal_z"]))

        # Build reflector-enabled scene config for GA workers
        ga_scene_config = {
            **exp.SCENE_CONFIG_2AP,
            "reflector_enabled": True,
            "reflector_size": rcfg["reflector_size"],
            "wall_top_left": rcfg["wall_top_left"],
            "wall_bottom_right": rcfg["wall_bottom_right"],
            "focal_point": [20.0, 20.0, focal_z],
            "device": "cpu",
        }
        ga_executor = exp.RayActorPoolExecutor(
            scene_config=ga_scene_config,
            num_workers=num_pool_workers,
            gpu_fraction=gpu_fraction,
            verbose=True,
        )
        try:
            return exp.example_deap_ga_2ap_with_reflector(
                ga_executor,
                ga_params=ga_params,
                min_ap_separation=float(trial.get("ga_min_ap_separation", 5.0)),
                output_dir=str(output_dir),
                random_seed=seed,
                wall_top_left=rcfg["wall_top_left"],
                wall_bottom_right=rcfg["wall_bottom_right"],
                reflector_size=rcfg["reflector_size"],
                focal_z=focal_z,
                target_quantile=target_quantile,
            )
        finally:
            ga_executor.shutdown()

    # Non-reflector GA (1ap or 2ap)
    ga_executor = exp.RayActorPoolExecutor(
        scene_config=exp.SCENE_CONFIG if mode == "1ap" else exp.SCENE_CONFIG_2AP,
        num_workers=num_pool_workers,
        gpu_fraction=gpu_fraction,
        verbose=True,
    )
    try:
        if mode == "1ap":
            return exp.example_deap_ga_1ap(
                ga_executor,
                ga_params=ga_params,
                output_dir=str(output_dir),
                random_seed=seed,
            )

        return exp.example_deap_ga_2ap(
            ga_executor,
            ga_params=ga_params,
            min_ap_separation=float(trial.get("ga_min_ap_separation", 5.0)),
            output_dir=str(output_dir),
            random_seed=seed,
        )
    finally:
        ga_executor.shutdown()


def _extract_summary_row(trial: dict[str, Any], results: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    method = trial["method"]
    mode = trial["mode"]

    best_dbm: float | None
    time_s: float | None

    if method in {"gd", "gs"}:
        best_dbm = results["best_result"]["best_metric_dbm"]
        time_s = results["total_time"]
    else:
        best_dbm = results["best_fitness_dbm"]
        time_s = results["total_time"]

    row: dict[str, Any] = {
        "trial": trial["name"],
        "method": trial["method"],
        "mode": trial["mode"],
        "random_seed": trial.get("random_seed", exp.RANDOM_SEED),
        "best_p5_rss_dbm": best_dbm,
        "time_s": time_s,
        "run_dir": str(run_dir),
    }

    # Reflector metadata
    if mode == "2ap_reflector":
        if method == "ga":
            refl = results.get("best_reflector")
            if refl:
                row["reflector_u"] = refl.get("u")
                row["reflector_v"] = refl.get("v")
                row["focal_x"] = refl.get("focal_x")
                row["focal_y"] = refl.get("focal_y")
        elif method in {"gd", "gs"}:
            br = results.get("best_result", {})
            row["reflector_u"] = br.get("reflector_u")
            row["reflector_v"] = br.get("reflector_v")
            focal = br.get("reflector_focal_point") or br.get("reflector_target")
            if focal and len(focal) >= 2:
                row["focal_x"] = focal[0]
                row["focal_y"] = focal[1]

    return row


def _save_summary_files(run_root: Path, rows: list[dict[str, Any]], detailed: list[dict[str, Any]]) -> None:
    summary_json = run_root / "summary.json"
    summary_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    summary_csv = run_root / "summary.csv"
    if rows:
        # Collect all unique keys across rows (reflector trials add extra columns)
        seen: set[str] = set()
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    detailed_json = run_root / "all_trials_detailed.json"
    detailed_json.write_text(json.dumps(detailed, indent=2, default=str), encoding="utf-8")


def generate_expanded_config(config_path: Path, generated_config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    trials = _build_trials(config)

    payload = {
        "shared": config.get("shared", {}),
        "trials": trials,
    }
    generated_config_path.parent.mkdir(parents=True, exist_ok=True)
    generated_config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Generated expanded trial config")
    print(f"Input config: {config_path}")
    print(f"Trials: {len(trials)}")
    print(f"Output config: {generated_config_path}")
    print("=" * 80)


def run_experiments(config_path: Path, output_root: Path) -> None:
    if exp is None or ray is None:
        raise RuntimeError(
            "Missing dependencies. Activate the project environment and install Ray "
            "before running (e.g., source .venv/bin/activate)."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    trials = _build_trials(config)
    if not trials:
        raise ValueError("No trials found. Provide 'trials' and/or 'sweep_groups'.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_root / f"ray_experiments_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "used_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print(f"Running {len(trials)} trial(s)")
    print(f"Config: {config_path}")
    print(f"Output: {run_root}")
    print("=" * 80)

    summary_rows: list[dict[str, Any]] = []
    detailed_results: list[dict[str, Any]] = []

    ray.init(ignore_reinit_error=True)
    failed_trials: list[str] = []
    try:
        total = len(trials)
        for idx, trial in enumerate(trials, start=1):
            trial_name = trial["name"]
            trial_dir = run_root / trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)

            output_txt = trial_dir / "output.txt"
            try:
                with output_txt.open("w", encoding="utf-8") as f:
                    tee_out = TeeStream(sys.stdout, f)
                    tee_err = TeeStream(sys.stderr, f)
                    with redirect_stdout(tee_out), redirect_stderr(tee_err):
                        print("\n" + "-" * 80)
                        print(
                            f"[{idx}/{total}] {trial_name} | "
                            f"method={trial['method']} | mode={trial['mode']} | "
                            f"seed={trial.get('random_seed', exp.RANDOM_SEED)}"
                        )
                        print("-" * 80)

                        result = _run_trial_method(trial, output_dir=trial_dir)

                        trial_record = {
                            "trial": trial_name,
                            "config": trial,
                            "result": result,
                        }
                        (trial_dir / "trial_record.json").write_text(
                            json.dumps(trial_record, indent=2, default=str),
                            encoding="utf-8",
                        )

                summary_row = _extract_summary_row(trial, result, trial_dir)
                summary_rows.append(summary_row)
                detailed_results.append(
                    {
                        "trial": trial_name,
                        "config": trial,
                        "result": result,
                        "summary": summary_row,
                    }
                )
            except Exception as exc:
                failed_trials.append(trial_name)
                tb_str = traceback.format_exc()
                print(f"\n*** Trial {trial_name} FAILED ***\n{tb_str}")
                # Persist failure info so the user can inspect later
                (trial_dir / "FAILED.txt").write_text(
                    f"Trial: {trial_name}\nError: {exc}\n\n{tb_str}",
                    encoding="utf-8",
                )
                # Save partial summaries after each failure
                _save_summary_files(run_root, summary_rows, detailed_results)

    finally:
        ray.shutdown()

    _save_summary_files(run_root, summary_rows, detailed_results)

    print("\n" + "=" * 80)
    print("Experiment run complete")
    if failed_trials:
        print(f"FAILED trials ({len(failed_trials)}/{total}): {', '.join(failed_trials)}")
    print(f"Successful: {total - len(failed_trials)}/{total}")
    print(f"Summary CSV: {run_root / 'summary.csv'}")
    print(f"Summary JSON: {run_root / 'summary.json'}")
    print(f"Detailed JSON: {run_root / 'all_trials_detailed.json'}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified runner for GD/GS/GA trials from ray_parallel_example.py"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/ray_experiment_runner_config.example.json"),
        help="Path to runner config JSON.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/experiments"),
        help="Root directory for run outputs.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only expand config (trials + sweep_groups) into explicit trials.",
    )
    parser.add_argument(
        "--generated-config",
        type=Path,
        default=Path("results/generated_trials.json"),
        help="Output path for --generate-only expanded config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.generate_only:
        generate_expanded_config(args.config, args.generated_config)
    else:
        run_experiments(config_path=args.config, output_root=args.output_root)
