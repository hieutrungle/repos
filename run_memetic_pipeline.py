#!/usr/bin/env python3
"""Convenience entrypoint for the Memetic Fusion pipeline.

Why this file exists
--------------------
This launcher lets you run memetic optimization from the repository root with
short commands, instead of typing the long module path every time.

Quick start
-----------
1) Run with default config:
   python run_memetic_pipeline.py

2) Run with a custom config:
   python run_memetic_pipeline.py --config configs/memetic_pipeline_config.json

3) Override output directory from CLI:
   python run_memetic_pipeline.py --output-dir results/experiments --run-name journal_run_01

4) See full options and helper notes:
   python run_memetic_pipeline.py --help
   python run_memetic_pipeline.py --hints

Config behavior
---------------
- Default config path: ``configs/memetic_pipeline_config.json``
- If config file is missing, this script writes a template automatically.
- CLI flags override values loaded from config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "memetic_pipeline_config.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reflector_position.optimizers.memetic.run_memetic_pipeline import (  # noqa: E402
    _default_memetic_config,
    run_memetic_optimization,
)


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge mapping values from updates into base."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, Mapping):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_json(path: Path) -> Dict[str, Any]:
    """Load config JSON and assert dictionary root type."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON file with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the convenience launcher."""
    parser = argparse.ArgumentParser(
        description="Run Memetic Fusion (GA + GD) using a short root-level command.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to pipeline config JSON (default: configs/memetic_pipeline_config.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override for artifact output directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run subfolder name under output_dir.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Force verbose logging, regardless of config file value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit without launching optimization.",
    )
    parser.add_argument(
        "--hints",
        action="store_true",
        help="Print practical config-editing and execution hints, then exit.",
    )
    return parser


def _print_hints(config_path: Path) -> None:
    """Print practical helper text for collaborators and future users."""
    print("\nMemetic Pipeline Hints")
    print("=" * 80)
    print("1) Primary config file:")
    print(f"   {config_path}")
    print("\n2) Most-used tuning keys:")
    print("   - objective_params.softmin_temperature / alpha / beta")
    print("   - ga_params.pop_size / ga_params.n_gen")
    print("   - ga_evaluation_params.samples_per_tx / max_depth")
    print("   - k_seeds / d_corr")
    print("   - gd_optimization_params.num_iterations / learning_rate")
    print("   - num_pool_workers / gpu_fraction")
    print("\n3) Fast smoke-test profile (recommended before long runs):")
    print("   - ga_params.pop_size: 12-30")
    print("   - ga_params.n_gen: 1-5")
    print("   - gd_optimization_params.num_iterations: 3-20")
    print("\n4) Journal-quality profile:")
    print("   - Increase pop_size, n_gen, num_iterations")
    print("   - Keep d_corr meaningful to avoid redundant seed exploitation")
    print("\n5) Typical commands:")
    print("   - python run_memetic_pipeline.py")
    print("   - python run_memetic_pipeline.py --run-name exp01")
    print("   - python run_memetic_pipeline.py --output-dir results/experiments")
    print("   - python run_memetic_pipeline.py --config configs/memetic_pipeline_config.json")
    print("=" * 80)


def main() -> int:
    """Resolve config, apply CLI overrides, and launch memetic optimization."""
    args = _build_parser().parse_args()

    config_path = Path(args.config).expanduser().resolve()

    if args.hints:
        _print_hints(config_path)
        return 0

    config = _default_memetic_config()

    if config_path.exists():
        loaded = _load_json(config_path)
        config = _deep_update(config, loaded)
    else:
        _write_json(config_path, config)
        print(f"[info] Config not found. Wrote template to: {config_path}")

    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.run_name:
        config["run_name"] = args.run_name
    if args.verbose:
        config["verbose"] = True

    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0

    output = run_memetic_optimization(config)

    saved = output.get("saved_artifacts", {})
    if isinstance(saved, dict) and saved.get("output_dir"):
        print(f"\nSaved artifacts: {saved['output_dir']}")

    global_best = output.get("global_best_result")
    best_primary_loss = (
        output.get("gd_results", {})
        .get("metrics", {})
        .get("best_primary_loss")
    )
    if isinstance(global_best, dict) and best_primary_loss is not None:
        print(
            "Global best: "
            f"task #{global_best.get('task_id')} | "
            f"primary_loss={float(best_primary_loss):.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
