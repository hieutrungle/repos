"""
Automated hyperparameter tuning runner for ``ray_parallel_example.py``.

Runs multiple trials with different random seeds and optimizer hyperparameters,
then saves per-trial artifacts and a summary CSV/JSON for easy comparison.

Usage:
	python examples/ray_hparam_tuning.py \
		--config examples/ray_hparam_tuning_config.example.json
"""

from __future__ import annotations

import argparse
import csv
import json
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


def _build_trials(config: dict[str, Any]) -> list[dict[str, Any]]:
	if "trials" in config:
		return config["trials"]

	if "sweep" not in config:
		raise ValueError("Config must contain either 'trials' or 'sweep'.")

	sweep = config["sweep"]
	random_seeds = sweep.get("random_seeds", [exp.RANDOM_SEED])
	modes = sweep.get("modes", ["1ap"])

	gd_grid = sweep.get("gd", {})
	ga_grid = sweep.get("ga", {})

	gd_learning_rates = gd_grid.get("learning_rate", [0.5])
	gd_temperatures = gd_grid.get("temperature", [0.15])
	gd_alphas = gd_grid.get("alpha", [0.9])
	gd_betas = gd_grid.get("beta", [0.1])

	ga_pop_sizes = ga_grid.get("pop_size", [150])
	ga_n_gens = ga_grid.get("n_gen", [50])
	ga_mutpbs = ga_grid.get("mutpb", [0.3])

	trials: list[dict[str, Any]] = []
	trial_idx = 1
	for mode, seed, lr, temp, alpha, beta, pop_size, n_gen, mutpb in product(
		modes,
		random_seeds,
		gd_learning_rates,
		gd_temperatures,
		gd_alphas,
		gd_betas,
		ga_pop_sizes,
		ga_n_gens,
		ga_mutpbs,
	):
		trial_name = (
			f"trial_{trial_idx:03d}_{mode}_seed{seed}_"
			f"lr{lr}_temp{temp}_a{alpha}_b{beta}_pop{pop_size}_gen{n_gen}_mut{mutpb}"
		)
		trials.append(
			{
				"name": trial_name,
				"mode": mode,
				"random_seed": seed,
				"gd_optimization_overrides": {
					"learning_rate": lr,
					"temperature": temp,
					"alpha": alpha,
					"beta": beta,
				},
				"ga_params": {
					"pop_size": pop_size,
					"n_gen": n_gen,
					"mutpb": mutpb,
				},
			}
		)
		trial_idx += 1

	return trials


def _extract_metrics(trial_name: str, mode: str, results: dict[str, Any]) -> dict[str, Any]:
	gd = results.get("gd")
	gs = results.get("gs")
	ga = results.get("ga")

	return {
		"trial": trial_name,
		"mode": mode,
		"gd_best_dbm": None if gd is None else gd["best_result"]["best_metric_dbm"],
		"gd_time_s": None if gd is None else gd["total_time"],
		"gs_best_dbm": None if gs is None else gs["best_result"]["best_metric_dbm"],
		"gs_time_s": None if gs is None else gs["total_time"],
		"ga_best_dbm": None if ga is None else ga["best_fitness_dbm"],
		"ga_time_s": None if ga is None else ga["total_time"],
	}


def _run_mode(
	mode: str,
	trial: dict[str, Any],
	trial_output_dir: Path,
	include_grid_search: bool,
) -> dict[str, Any]:
	seed = int(trial.get("random_seed", exp.RANDOM_SEED))
	gd_optimization_overrides = trial.get("gd_optimization_overrides")
	ga_params = _merge_dict(exp.GA_PARAMS, trial.get("ga_params"))

	num_pool_workers = int(trial.get("num_pool_workers", exp.NUM_POOL_WORKERS))
	gpu_fraction = float(trial.get("gpu_fraction", exp.GPU_FRACTION))

	gd_num_tasks = int(trial.get("gd_num_tasks", 100))
	gd_num_iterations = int(trial.get("gd_num_iterations", 50))
	gd_samples_per_tx = int(trial.get("gd_samples_per_tx", 1_000_000))
	gd_repulsion_weight = float(trial.get("gd_repulsion_weight", 0.3))

	results: dict[str, Any] = {}

	parallel_opt = exp.RayParallelOptimizer(
		num_workers=num_pool_workers,
		gpu_fraction=gpu_fraction,
	)
	try:
		if mode == "1ap":
			results["gd"] = exp.example_parallel_gradient_descent(
				parallel_opt,
				num_aps=1,
				num_tasks=gd_num_tasks,
				num_iterations=gd_num_iterations,
				samples_per_tx=gd_samples_per_tx,
				output_dir=str(trial_output_dir),
				scene_config=exp.SCENE_CONFIG,
				random_seed=seed,
				optimization_overrides=gd_optimization_overrides,
			)
			if include_grid_search:
				results["gs"] = exp.example_parallel_grid_search(
					parallel_opt,
					grid_resolution=float(trial.get("gs_grid_resolution", 1.0)),
					output_dir=str(trial_output_dir),
					scene_config=exp.SCENE_CONFIG,
				)
		elif mode == "2ap":
			results["gd"] = exp.example_parallel_gradient_descent(
				parallel_opt,
				num_aps=2,
				num_tasks=gd_num_tasks,
				num_iterations=gd_num_iterations,
				repulsion_weight=gd_repulsion_weight,
				samples_per_tx=gd_samples_per_tx,
				output_dir=str(trial_output_dir),
				scene_config=exp.SCENE_CONFIG_2AP,
				random_seed=seed,
				optimization_overrides=gd_optimization_overrides,
			)
			if include_grid_search:
				results["gs"] = exp.example_parallel_grid_search_2ap(
					parallel_opt,
					grid_resolution=float(trial.get("gs_grid_resolution", 1.0)),
					num_rounds=int(trial.get("gs_num_rounds", exp.ALTERNATING_ROUNDS)),
					output_dir=str(trial_output_dir),
					scene_config=exp.SCENE_CONFIG_2AP,
				)
		else:
			raise ValueError(f"Unsupported mode: {mode}")
	finally:
		parallel_opt.shutdown()

	ga_executor = exp.RayActorPoolExecutor(
		scene_config=exp.SCENE_CONFIG if mode == "1ap" else exp.SCENE_CONFIG_2AP,
		num_workers=num_pool_workers,
		gpu_fraction=gpu_fraction,
		verbose=True,
	)
	try:
		if mode == "1ap":
			results["ga"] = exp.example_deap_ga_1ap(
				ga_executor,
				ga_params=ga_params,
				output_dir=str(trial_output_dir),
				random_seed=seed,
			)
		else:
			results["ga"] = exp.example_deap_ga_2ap(
				ga_executor,
				ga_params=ga_params,
				min_ap_separation=float(trial.get("ga_min_ap_separation", 5.0)),
				output_dir=str(trial_output_dir),
				random_seed=seed,
			)
	finally:
		ga_executor.shutdown()

	return results


def run_tuning(config_path: Path, output_root: Path) -> None:
	if exp is None or ray is None:
		raise RuntimeError(
			"Missing dependencies. Activate the project environment and install Ray "
			"before running tuning (e.g., source .venv/bin/activate)."
		)

	config = json.loads(config_path.read_text())
	trials = _build_trials(config)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = output_root / f"ray_tuning_{timestamp}"
	run_dir.mkdir(parents=True, exist_ok=True)

	(run_dir / "used_config.json").write_text(json.dumps(config, indent=2))

	include_grid_search = bool(config.get("include_grid_search", False))

	print("=" * 80)
	print(f"Running {len(trials)} trial(s)")
	print(f"Config: {config_path}")
	print(f"Output: {run_dir}")
	print(f"Include grid search: {include_grid_search}")
	print("=" * 80)

	summary_rows: list[dict[str, Any]] = []
	detailed_results: list[dict[str, Any]] = []

	ray.init(ignore_reinit_error=True)
	try:
		for idx, trial in enumerate(trials, start=1):
			trial_name = trial.get("name", f"trial_{idx:03d}")
			mode = trial.get("mode", "1ap")
			if mode not in {"1ap", "2ap", "both"}:
				raise ValueError(f"Invalid mode '{mode}' in trial '{trial_name}'")

			print("\n" + "-" * 80)
			print(f"[{idx}/{len(trials)}] {trial_name} | mode={mode}")
			print("-" * 80)

			trial_dir = run_dir / trial_name
			trial_dir.mkdir(parents=True, exist_ok=True)

			trial_record: dict[str, Any] = {
				"trial": trial_name,
				"config": trial,
				"results": {},
			}

			if mode in {"1ap", "both"}:
				mode_dir = trial_dir / "1ap"
				mode_dir.mkdir(exist_ok=True)
				results_1ap = _run_mode(
					mode="1ap",
					trial=trial,
					trial_output_dir=mode_dir,
					include_grid_search=include_grid_search,
				)
				trial_record["results"]["1ap"] = results_1ap
				summary_rows.append(_extract_metrics(trial_name, "1ap", results_1ap))

			if mode in {"2ap", "both"}:
				mode_dir = trial_dir / "2ap"
				mode_dir.mkdir(exist_ok=True)
				results_2ap = _run_mode(
					mode="2ap",
					trial=trial,
					trial_output_dir=mode_dir,
					include_grid_search=include_grid_search,
				)
				trial_record["results"]["2ap"] = results_2ap
				summary_rows.append(_extract_metrics(trial_name, "2ap", results_2ap))

			detailed_results.append(trial_record)

			(trial_dir / "trial_record.json").write_text(
				json.dumps(trial_record, indent=2, default=str)
			)
	finally:
		ray.shutdown()

	summary_json_path = run_dir / "summary.json"
	summary_json_path.write_text(json.dumps(summary_rows, indent=2, default=str))

	summary_csv_path = run_dir / "summary.csv"
	if summary_rows:
		fieldnames = list(summary_rows[0].keys())
		with summary_csv_path.open("w", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(summary_rows)

	detailed_path = run_dir / "all_trials_detailed.json"
	detailed_path.write_text(json.dumps(detailed_results, indent=2, default=str))

	print("\n" + "=" * 80)
	print("Tuning complete")
	print(f"Summary CSV: {summary_csv_path}")
	print(f"Summary JSON: {summary_json_path}")
	print(f"Detailed: {detailed_path}")
	print("=" * 80)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Hyperparameter tuning runner for examples/ray_parallel_example.py"
	)
	parser.add_argument(
		"--config",
		type=Path,
		default=Path("examples/ray_hparam_tuning_config.example.json"),
		help="Path to tuning config JSON.",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=Path("results/hparam_tuning"),
		help="Root directory for tuning outputs.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run_tuning(config_path=args.config, output_root=args.output_root)
