"""
Run `ray_parallel_example.py` across multiple seeds and key hyperparameter configs.

This runner:
- Executes 1-AP and/or 2-AP experiments for each trial configuration
- Saves each trial to `results/ray_parallel_<abbr>`
- Renames all generated graph files (`.png`) to include:
  - prefix: `ray_parallel_`
  - postfix: abbreviated key hyperparameters
- Captures all terminal output to `output.txt` inside each trial folder

Usage:
    python examples/ray_parallel_sweep_runner.py \
        --config examples/ray_parallel_sweep_config.example.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
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


@dataclass
class TrialConfig:
    mode: str
    random_seed: int
    gd_learning_rate: float
    gd_temperature: float
    gd_alpha: float
    gd_beta: float
    ga_pop_size: int
    ga_n_gen: int
    ga_mutpb: float
    num_pool_workers: int
    gpu_fraction: float
    gd_num_tasks: int
    gd_num_iterations: int
    gd_samples_per_tx: int
    gd_repulsion_weight: float
    ga_min_ap_separation: float


class TeeStream:
    """Write to both terminal and file."""

    def __init__(self, terminal_stream, file_stream):
        self.terminal_stream = terminal_stream
        self.file_stream = file_stream

    def write(self, data):
        self.terminal_stream.write(data)
        self.file_stream.write(data)

    def flush(self):
        self.terminal_stream.flush()
        self.file_stream.flush()


def _abbr_float(value: float) -> str:
    s = f"{value:g}"
    s = s.replace("-", "m")
    s = s.replace(".", "")
    return s


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)


def _trial_suffix(t: TrialConfig) -> str:
    return (
        f"s{t.random_seed}_"
        f"lr{_abbr_float(t.gd_learning_rate)}_"
        f"t{_abbr_float(t.gd_temperature)}_"
        f"a{_abbr_float(t.gd_alpha)}_"
        f"b{_abbr_float(t.gd_beta)}_"
        f"p{t.ga_pop_size}_"
        f"g{t.ga_n_gen}_"
        f"m{_abbr_float(t.ga_mutpb)}"
    )


def _rename_graphs_with_prefix_suffix(run_dir: Path, suffix: str) -> None:
    for png_path in run_dir.rglob("*.png"):
        stem = png_path.stem
        if stem.startswith("ray_parallel_") and stem.endswith(f"_{suffix}"):
            continue
        new_name = f"ray_parallel_{stem}_{suffix}{png_path.suffix}"
        png_path.rename(png_path.with_name(new_name))


def _build_trials(config: dict[str, Any]) -> list[TrialConfig]:
    shared = config.get("shared", {})
    sweep = config.get("sweep", {})

    modes = sweep.get("modes", ["both"])
    seeds = sweep.get("random_seeds", [exp.RANDOM_SEED])

    gd = sweep.get("gd", {})
    ga = sweep.get("ga", {})

    lr_values = gd.get("learning_rate", [0.5])
    temp_values = gd.get("temperature", [0.15])
    alpha_values = gd.get("alpha", [0.9])
    beta_values = gd.get("beta", [0.1])

    pop_values = ga.get("pop_size", [150])
    n_gen_values = ga.get("n_gen", [50])
    mutpb_values = ga.get("mutpb", [0.3])

    trials: list[TrialConfig] = []
    for mode, seed, lr, temp, alpha, beta, pop_size, n_gen, mutpb in product(
        modes,
        seeds,
        lr_values,
        temp_values,
        alpha_values,
        beta_values,
        pop_values,
        n_gen_values,
        mutpb_values,
    ):
        if mode not in {"1ap", "2ap", "both"}:
            raise ValueError(f"Unsupported mode: {mode}")

        trials.append(
            TrialConfig(
                mode=mode,
                random_seed=int(seed),
                gd_learning_rate=float(lr),
                gd_temperature=float(temp),
                gd_alpha=float(alpha),
                gd_beta=float(beta),
                ga_pop_size=int(pop_size),
                ga_n_gen=int(n_gen),
                ga_mutpb=float(mutpb),
                num_pool_workers=int(shared.get("num_pool_workers", exp.NUM_POOL_WORKERS)),
                gpu_fraction=float(shared.get("gpu_fraction", exp.GPU_FRACTION)),
                gd_num_tasks=int(shared.get("gd_num_tasks", 100)),
                gd_num_iterations=int(shared.get("gd_num_iterations", 50)),
                gd_samples_per_tx=int(shared.get("gd_samples_per_tx", 1_000_000)),
                gd_repulsion_weight=float(shared.get("gd_repulsion_weight", 0.3)),
                ga_min_ap_separation=float(shared.get("ga_min_ap_separation", 5.0)),
            )
        )

    return trials


def _run_single_trial(base_results_dir: Path, trial: TrialConfig, index: int, total: int) -> dict[str, Any]:
    suffix = _trial_suffix(trial)
    run_dir_name = _sanitize_name(f"ray_parallel_{suffix}")
    run_dir = base_results_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    output_txt = run_dir / "output.txt"
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "suffix": suffix,
        "mode": trial.mode,
        "random_seed": trial.random_seed,
    }

    gd_overrides = {
        "learning_rate": trial.gd_learning_rate,
        "temperature": trial.gd_temperature,
        "alpha": trial.gd_alpha,
        "beta": trial.gd_beta,
    }

    ga_params = {
        **exp.GA_PARAMS,
        "pop_size": trial.ga_pop_size,
        "n_gen": trial.ga_n_gen,
        "mutpb": trial.ga_mutpb,
    }

    with output_txt.open("w", encoding="utf-8") as f:
        tee_out = TeeStream(sys.stdout, f)
        tee_err = TeeStream(sys.stderr, f)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print("=" * 80)
            print(f"[{index}/{total}] Running {run_dir_name}")
            print(f"mode={trial.mode} | seed={trial.random_seed}")
            print(f"GD: lr={trial.gd_learning_rate}, temp={trial.gd_temperature}, "
                  f"alpha={trial.gd_alpha}, beta={trial.gd_beta}")
            print(f"GA: pop={trial.ga_pop_size}, gen={trial.ga_n_gen}, mutpb={trial.ga_mutpb}")
            print("=" * 80)

            if trial.mode in {"1ap", "both"}:
                results_1ap = exp.run_all_1ap(
                    output_dir=str(run_dir),
                    num_pool_workers=trial.num_pool_workers,
                    gpu_fraction=trial.gpu_fraction,
                    random_seed=trial.random_seed,
                    gd_num_tasks=trial.gd_num_tasks,
                    gd_num_iterations=trial.gd_num_iterations,
                    gd_optimization_overrides=gd_overrides,
                    ga_params=ga_params,
                )
                summary["gd_1ap_best_dbm"] = results_1ap["gd_1ap"]["best_result"]["best_metric_dbm"]
                summary["ga_1ap_best_dbm"] = results_1ap["ga_1ap"]["best_fitness_dbm"]

            if trial.mode in {"2ap", "both"}:
                results_2ap = exp.run_all_2ap(
                    output_dir=str(run_dir),
                    num_pool_workers=trial.num_pool_workers,
                    gpu_fraction=trial.gpu_fraction,
                    random_seed=trial.random_seed,
                    gd_num_tasks=trial.gd_num_tasks,
                    gd_num_iterations=trial.gd_num_iterations,
                    gd_repulsion_weight=trial.gd_repulsion_weight,
                    gd_samples_per_tx=trial.gd_samples_per_tx,
                    gd_optimization_overrides=gd_overrides,
                    ga_params=ga_params,
                    ga_min_ap_separation=trial.ga_min_ap_separation,
                )
                summary["gd_2ap_best_dbm"] = results_2ap["gd_2ap"]["best_result"]["best_metric_dbm"]
                summary["ga_2ap_best_dbm"] = results_2ap["ga_2ap"]["best_fitness_dbm"]

            print("\nRun completed.")

    _rename_graphs_with_prefix_suffix(run_dir, suffix)

    # Save per-run metadata
    metadata_path = run_dir / "run_metadata.json"
    metadata_payload = {
        "trial": trial.__dict__,
        "suffix": suffix,
        "summary": summary,
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    return summary


def run_sweep(config_path: Path, results_dir: Path) -> None:
    if exp is None or ray is None:
        raise RuntimeError(
            "Missing dependencies. Activate project env and install Ray first "
            "(e.g., source .venv/bin/activate)."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    trials = _build_trials(config)
    if not trials:
        raise ValueError("No trials generated from config.")

    results_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    ray.init(ignore_reinit_error=True)
    try:
        total = len(trials)
        for idx, trial in enumerate(trials, start=1):
            summary = _run_single_trial(results_dir, trial, idx, total)
            all_summaries.append(summary)
    finally:
        ray.shutdown()

    sweep_summary_path = results_dir / "ray_parallel_sweep_summary.json"
    sweep_summary_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Sweep finished")
    print(f"Summary: {sweep_summary_path}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ray_parallel_example.py for multiple seeds/hyperparameters"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/ray_parallel_sweep_config.example.json"),
        help="Path to sweep config JSON",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory where `ray_parallel_*` run folders are created",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(config_path=args.config, results_dir=args.results_dir)
