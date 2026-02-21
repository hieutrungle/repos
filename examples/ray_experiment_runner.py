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
VALID_MODES = {"1ap", "2ap"}


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
    elif method == "gs":
        suffix_parts.append(f"gr{_abbr_float(float(trial.get('gs_grid_resolution', 1.0)))}")
        if mode == "2ap":
            suffix_parts.append(f"r{int(trial.get('gs_num_rounds', exp.ALTERNATING_ROUNDS))}")
    elif method == "ga":
        ga_params = trial.get("ga_params", {})
        suffix_parts.extend(
            [
                f"p{int(ga_params.get('pop_size', exp.GA_PARAMS['pop_size']))}",
                f"g{int(ga_params.get('n_gen', exp.GA_PARAMS['n_gen']))}",
                f"m{_abbr_float(float(ga_params.get('mutpb', exp.GA_PARAMS['mutpb'])))}",
            ]
        )

    suffix = "_".join(suffix_parts)
    return f"trial_{trial_idx:03d}_{method}_{mode}_{suffix}"


def _build_trials(config: dict[str, Any]) -> list[dict[str, Any]]:
    shared = config.get("shared", {})
    explicit_trials = config.get("trials", [])
    sweep_groups = config.get("sweep_groups", [])

    all_trials: list[dict[str, Any]] = []

    for trial in explicit_trials:
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


def _run_trial_method(trial: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    method = trial["method"]
    mode = trial["mode"]
    seed = int(trial.get("random_seed", exp.RANDOM_SEED))

    num_pool_workers = int(trial.get("num_pool_workers", exp.NUM_POOL_WORKERS))
    gpu_fraction = float(trial.get("gpu_fraction", exp.GPU_FRACTION))

    if method in {"gd", "gs"}:
        parallel_opt = exp.RayParallelOptimizer(
            num_workers=num_pool_workers,
            gpu_fraction=gpu_fraction,
        )
        try:
            if method == "gd":
                gd_num_tasks = int(trial.get("gd_num_tasks", 100))
                gd_num_iterations = int(trial.get("gd_num_iterations", 50))
                gd_samples_per_tx = int(trial.get("gd_samples_per_tx", 1_000_000))
                gd_repulsion_weight = float(trial.get("gd_repulsion_weight", 0.3))

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

            if mode == "1ap":
                return exp.example_parallel_grid_search(
                    parallel_opt,
                    grid_resolution=float(trial.get("gs_grid_resolution", 1.0)),
                    output_dir=str(output_dir),
                    scene_config=exp.SCENE_CONFIG,
                )

            return exp.example_parallel_grid_search_2ap(
                parallel_opt,
                grid_resolution=float(trial.get("gs_grid_resolution", 1.0)),
                num_rounds=int(trial.get("gs_num_rounds", exp.ALTERNATING_ROUNDS)),
                output_dir=str(output_dir),
                scene_config=exp.SCENE_CONFIG_2AP,
            )
        finally:
            parallel_opt.shutdown()

    ga_executor = exp.RayActorPoolExecutor(
        scene_config=exp.SCENE_CONFIG if mode == "1ap" else exp.SCENE_CONFIG_2AP,
        num_workers=num_pool_workers,
        gpu_fraction=gpu_fraction,
        verbose=True,
    )
    try:
        ga_params = _merge_dict(exp.GA_PARAMS, trial.get("ga_params"))
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

    best_dbm: float | None
    time_s: float | None

    if method in {"gd", "gs"}:
        best_dbm = results["best_result"]["best_metric_dbm"]
        time_s = results["total_time"]
    else:
        best_dbm = results["best_fitness_dbm"]
        time_s = results["total_time"]

    return {
        "trial": trial["name"],
        "method": trial["method"],
        "mode": trial["mode"],
        "random_seed": trial.get("random_seed", exp.RANDOM_SEED),
        "best_dbm": best_dbm,
        "time_s": time_s,
        "run_dir": str(run_dir),
    }


def _save_summary_files(run_root: Path, rows: list[dict[str, Any]], detailed: list[dict[str, Any]]) -> None:
    summary_json = run_root / "summary.json"
    summary_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    summary_csv = run_root / "summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
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
    try:
        total = len(trials)
        for idx, trial in enumerate(trials, start=1):
            trial_name = trial["name"]
            trial_dir = run_root / trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)

            output_txt = trial_dir / "output.txt"
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

    finally:
        ray.shutdown()

    _save_summary_files(run_root, summary_rows, detailed_results)

    print("\n" + "=" * 80)
    print("Experiment run complete")
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
