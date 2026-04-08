#!/usr/bin/env python3
"""Sweep AP count across multiple seeds and plot RSSI metrics with std shading.

How to run
----------
Example command:

        python scripts/run_num_aps_seed_sweep.py \
                --config configs/memetic_pipeline_config_hrbb.json \
                --num-aps 1 2 3 \
                --seeds 41 42 43 44 45

Outputs are written under one timestamped sweep folder and include:
- Per-run CSV and aggregate CSV/JSON.
- One plot per metric key:
    - mean_rss_dbm, min_rss_dbm, p5_rss_dbm
    - priority_mean_rss_dbm, priority_min_rss_dbm, priority_p5_rss_dbm

By default, all plots use the same fixed y-axis range [-100, -40] dBm for
easy comparison. You can override this range with --y-min and --y-max.

This script runs the memetic pipeline for each pair in:
- AP count list (num_aps)
- random seed list

For each AP count, it computes:
- mean RSSI across successful seed runs
- standard deviation of RSSI across successful seed runs

The final artifact is an elbow-style plot:
- x-axis: number of APs
- y-axis: mean RSSI
- shaded band: +/- one standard deviation

Important constraint enforced by this script:
- For every run, len(scene_config.tx_positions) == num_aps
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_BASE_CONFIG_PATH = REPO_ROOT / "configs" / "memetic_pipeline_config.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reflector_position.optimizers.memetic.run_memetic_pipeline import (  # noqa: E402
    _default_memetic_config,
    run_memetic_optimization,
)


_PLOT_METRIC_KEYS = (
    "mean_rss_dbm",
    "min_rss_dbm",
    "p5_rss_dbm",
    "priority_mean_rss_dbm",
    "priority_min_rss_dbm",
    "priority_p5_rss_dbm",
)

_METRIC_LABELS = {
    "mean_rss_dbm": "Mean RSSI (All Region)",
    "min_rss_dbm": "Min RSSI (All Region)",
    "p5_rss_dbm": "P5 RSSI (All Region)",
    "priority_mean_rss_dbm": "Mean RSSI (Priority Region)",
    "priority_min_rss_dbm": "Min RSSI (Priority Region)",
    "priority_p5_rss_dbm": "P5 RSSI (Priority Region)",
}

_COMPARISON_METRIC_PAIRS = (
    ("mean_rss_dbm", "priority_mean_rss_dbm", "Mean RSSI"),
    ("min_rss_dbm", "priority_min_rss_dbm", "Min RSSI"),
    ("p5_rss_dbm", "priority_p5_rss_dbm", "P5 RSSI"),
)

_DEFAULT_PLOT_Y_RANGE = (-100.0, -40.0)


def _load_json(path: Path) -> Dict[str, Any]:
    """Load one JSON file and validate mapping root."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a JSON object: {path}")
    return payload


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge mapping values from updates into base."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, Mapping):
            _deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _write_json(path: Path, payload: Any) -> None:
    """Write one JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write one CSV file from rows and explicit fieldnames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _coerce_xyz_positions(raw_positions: Any) -> List[List[float]]:
    """Validate and coerce config tx positions as a list of 3D points."""
    if not isinstance(raw_positions, Sequence) or isinstance(raw_positions, (str, bytes)):
        raise ValueError("scene_config.tx_positions must be a sequence of [x, y, z] points")

    positions: List[List[float]] = []
    for idx, item in enumerate(raw_positions):
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 3:
            raise ValueError(
                f"scene_config.tx_positions[{idx}] must be a length-3 sequence [x, y, z]"
            )
        try:
            positions.append([float(item[0]), float(item[1]), float(item[2])])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"scene_config.tx_positions[{idx}] must contain numeric values"
            ) from exc

    if not positions:
        raise ValueError("scene_config.tx_positions must not be empty")

    return positions


def _extract_best_rssi_metric(summary: Mapping[str, Any], metric_key: str) -> float:
    """Extract one best-run RSSI metric from the memetic summary payload."""
    best_candidates = [
        summary.get("global_best_result"),
        summary.get("gd_results", {}).get("global_best_result"),
    ]

    for candidate in best_candidates:
        if not isinstance(candidate, Mapping):
            continue
        results = candidate.get("results")
        if not isinstance(results, Mapping):
            continue
        physical = results.get("physical_metrics")
        if not isinstance(physical, Mapping):
            continue
        value = physical.get(metric_key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    ga_results = summary.get("ga_results")
    if isinstance(ga_results, Mapping):
        ga_best_physical = ga_results.get("best_physical_metrics")
        if isinstance(ga_best_physical, Mapping):
            value = ga_best_physical.get(metric_key)
            if value is not None:
                return float(value)

    raise KeyError(
        "Could not find metric "
        f"'{metric_key}' in gd_results.global_best_result.results.physical_metrics "
        "or ga_results.best_physical_metrics"
    )


def _summarize_metric_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Return basic descriptive stats for one metric vector."""
    if not values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _plot_metric_with_std(
    aggregate_rows: Sequence[Mapping[str, Any]],
    series_specs: Sequence[Mapping[str, Any]],
    save_path: Path,
    title: str,
    y_axis_label: str = "RSSI (dBm)",
    y_limits: Optional[Tuple[float, float]] = None,
) -> bool:
    """Plot one or more metric mean curves with +/- std shading across AP counts."""
    if not series_specs:
        return False

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted_any = False

    for series in series_specs:
        mean_field = str(series.get("mean_field", ""))
        std_field = str(series.get("std_field", ""))
        label = str(series.get("label", mean_field))
        color = series.get("color")
        linestyle = str(series.get("linestyle", "-"))

        plotted_rows = [row for row in aggregate_rows if row.get(mean_field) is not None]
        if not plotted_rows:
            continue

        x = np.asarray([int(row["num_aps"]) for row in plotted_rows], dtype=np.int64)
        y = np.asarray([float(row[mean_field]) for row in plotted_rows], dtype=np.float64)
        std = np.asarray([float(row.get(std_field) or 0.0) for row in plotted_rows], dtype=np.float64)

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            linestyle=linestyle,
            color=color,
            label=label,
        )
        ax.fill_between(
            x,
            y - std,
            y + std,
            alpha=0.16,
            color=color,
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return False

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Number of APs")
    ax.set_ylabel(y_axis_label)
    ax.set_title(title)
    if y_limits is not None:
        ax.set_ylim(float(y_limits[0]), float(y_limits[1]))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _build_run_config(
    base_config: Mapping[str, Any],
    tx_pool: Sequence[Sequence[float]],
    num_aps: int,
    seed: int,
    output_root: Path,
    run_name: str,
    verbose: bool,
) -> Dict[str, Any]:
    """Build one trial config while enforcing tx_positions length == num_aps."""
    if num_aps <= 0:
        raise ValueError(f"num_aps must be positive, got {num_aps}")
    if len(tx_pool) < num_aps:
        raise ValueError(
            "Not enough tx_positions for requested num_aps: "
            f"need {num_aps}, have {len(tx_pool)}"
        )

    cfg = deepcopy(dict(base_config))
    scene_config = cfg.get("scene_config")
    if not isinstance(scene_config, Mapping):
        raise ValueError("Config must contain scene_config mapping")

    selected_positions = [list(point) for point in tx_pool[:num_aps]]
    if len(selected_positions) != num_aps:
        raise ValueError(
            "Internal error while selecting tx_positions: "
            f"expected {num_aps}, got {len(selected_positions)}"
        )

    updated_scene = dict(scene_config)
    updated_scene["tx_positions"] = selected_positions

    cfg["scene_config"] = updated_scene
    cfg["num_aps"] = int(num_aps)
    cfg["random_seed"] = int(seed)
    cfg["output_dir"] = str(output_root)
    cfg["run_name"] = run_name
    cfg["verbose"] = bool(verbose)

    # Enforce the user-requested contract explicitly.
    if len(cfg["scene_config"]["tx_positions"]) != int(cfg["num_aps"]):
        raise ValueError("len(scene_config.tx_positions) must equal num_aps")

    return cfg


def _parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Sweep num_aps across multiple random seeds and plot all/priority "
            "mean, min, and p5 RSSI with standard deviation shading."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_BASE_CONFIG_PATH),
        help="Path to base memetic JSON config.",
    )
    parser.add_argument(
        "--num-aps",
        type=int,
        nargs="+",
        default=None,
        help=(
            "List of AP counts to evaluate (example: --num-aps 1 2 3 4). "
            "Default: 1..len(scene_config.tx_positions)."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds to evaluate for each AP count.",
    )
    parser.add_argument(
        "--rssi-metric",
        type=str,
        default="mean_rss_dbm",
        help=(
            "Optional extra metric key in physical_metrics to aggregate/plot "
            "in addition to the built-in six metrics."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output root directory for this sweep.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first failed trial.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose memetic pipeline logging per run.",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=_DEFAULT_PLOT_Y_RANGE[0],
        help="Shared y-axis lower limit for all generated plots.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=_DEFAULT_PLOT_Y_RANGE[1],
        help="Shared y-axis upper limit for all generated plots.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the AP-count-by-seed sweep and save aggregate artifacts."""
    args = _parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_config = _default_memetic_config()
    _deep_update(base_config, _load_json(config_path))

    scene_config = base_config.get("scene_config")
    if not isinstance(scene_config, Mapping):
        raise ValueError("Base config must contain scene_config mapping")

    tx_pool = _coerce_xyz_positions(scene_config.get("tx_positions"))

    if args.num_aps is None:
        num_aps_values = list(range(1, len(tx_pool) + 1))
    else:
        num_aps_values = sorted({int(value) for value in args.num_aps})

    if not num_aps_values:
        raise ValueError("num_aps list must not be empty")
    if min(num_aps_values) <= 0:
        raise ValueError("All num_aps values must be positive")
    if max(num_aps_values) > len(tx_pool):
        raise ValueError(
            "Requested num_aps exceeds available tx_positions in base config: "
            f"max num_aps requested={max(num_aps_values)}, "
            f"available tx positions={len(tx_pool)}"
        )

    seeds = [int(seed) for seed in args.seeds]
    if not seeds:
        raise ValueError("Seed list must not be empty")

    if float(args.y_min) >= float(args.y_max):
        raise ValueError("--y-min must be smaller than --y-max")
    y_limits = (float(args.y_min), float(args.y_max))

    configured_output_root = args.output_dir or str(base_config.get("output_dir", "results/experiments"))
    output_root = Path(configured_output_root).expanduser().resolve() / (
        "num_aps_seed_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] config: {config_path}")
    print(f"[sweep] output: {output_root}")
    print(f"[sweep] num_aps values: {num_aps_values}")
    print(f"[sweep] seeds: {seeds}")
    print(f"[sweep] metric: {args.rssi_metric}")
    print(f"[sweep] plotted metric keys: {list(_PLOT_METRIC_KEYS)}")
    print(f"[sweep] fixed y-axis range: {y_limits}")

    run_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []

    for num_aps in num_aps_values:
        metric_values: List[float] = []
        metric_values_by_key: Dict[str, List[float]] = {
            metric_key: [] for metric_key in _PLOT_METRIC_KEYS
        }

        for seed in seeds:
            run_name = f"aps_{num_aps:02d}_seed_{seed}"
            trial_cfg = _build_run_config(
                base_config=base_config,
                tx_pool=tx_pool,
                num_aps=num_aps,
                seed=seed,
                output_root=output_root,
                run_name=run_name,
                verbose=bool(args.verbose),
            )

            print(f"[run] start num_aps={num_aps}, seed={seed}, run_name={run_name}")
            try:
                summary = run_memetic_optimization(trial_cfg)
                metric_value = _extract_best_rssi_metric(summary, args.rssi_metric)
                metric_values.append(metric_value)

                metric_row_values: Dict[str, float] = {}
                for metric_key in _PLOT_METRIC_KEYS:
                    value = _extract_best_rssi_metric(summary, metric_key)
                    metric_values_by_key[metric_key].append(value)
                    metric_row_values[metric_key] = value

                saved_artifacts = summary.get("saved_artifacts", {})
                run_rows.append(
                    {
                        "num_aps": num_aps,
                        "seed": seed,
                        "run_name": run_name,
                        "status": "ok",
                        "metric_key": args.rssi_metric,
                        "rssi_value": metric_value,
                        **{f"{key}_value": metric_row_values[key] for key in _PLOT_METRIC_KEYS},
                        "output_dir": saved_artifacts.get("output_dir"),
                        "error": "",
                    }
                )
                print(
                    f"[run] done  num_aps={num_aps}, seed={seed}, "
                    f"{args.rssi_metric}={metric_value:.4f}"
                )

            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                run_rows.append(
                    {
                        "num_aps": num_aps,
                        "seed": seed,
                        "run_name": run_name,
                        "status": "error",
                        "metric_key": args.rssi_metric,
                        "rssi_value": None,
                        **{f"{key}_value": None for key in _PLOT_METRIC_KEYS},
                        "output_dir": None,
                        "error": error_text,
                    }
                )
                print(f"[run] error num_aps={num_aps}, seed={seed}: {error_text}")
                if args.verbose:
                    traceback.print_exc()
                if args.fail_fast:
                    raise

        success_count = len(metric_values)
        total_count = len(seeds)
        primary_stats = _summarize_metric_values(metric_values)

        aggregate_row: Dict[str, Any] = {
            "num_aps": num_aps,
            "metric_key": args.rssi_metric,
            "num_success": success_count,
            "num_total": total_count,
            "mean_rssi": primary_stats["mean"],
            "std_rssi": primary_stats["std"],
            "min_rssi": primary_stats["min"],
            "max_rssi": primary_stats["max"],
        }
        for metric_key in _PLOT_METRIC_KEYS:
            stats = _summarize_metric_values(metric_values_by_key[metric_key])
            aggregate_row[f"{metric_key}_mean"] = stats["mean"]
            aggregate_row[f"{metric_key}_std"] = stats["std"]
            aggregate_row[f"{metric_key}_min"] = stats["min"]
            aggregate_row[f"{metric_key}_max"] = stats["max"]

        aggregate_rows.append(aggregate_row)

        print(
            f"[agg] num_aps={num_aps}: successes={success_count}/{total_count}, "
            f"mean={primary_stats['mean']}, std={primary_stats['std']}"
        )

    run_rows_path = output_root / "per_run_results.csv"
    aggregate_rows_path = output_root / "aggregate_by_num_aps.csv"
    aggregate_json_path = output_root / "aggregate_by_num_aps.json"

    run_fieldnames = [
        "num_aps",
        "seed",
        "run_name",
        "status",
        "metric_key",
        "rssi_value",
        *[f"{metric_key}_value" for metric_key in _PLOT_METRIC_KEYS],
        "output_dir",
        "error",
    ]

    aggregate_fieldnames = [
        "num_aps",
        "metric_key",
        "num_success",
        "num_total",
        "mean_rssi",
        "std_rssi",
        "min_rssi",
        "max_rssi",
        *[
            field
            for metric_key in _PLOT_METRIC_KEYS
            for field in (
                f"{metric_key}_mean",
                f"{metric_key}_std",
                f"{metric_key}_min",
                f"{metric_key}_max",
            )
        ],
    ]

    _write_csv(
        run_rows_path,
        run_rows,
        fieldnames=run_fieldnames,
    )
    _write_csv(
        aggregate_rows_path,
        aggregate_rows,
        fieldnames=aggregate_fieldnames,
    )
    _write_json(
        aggregate_json_path,
        {
            "generated_at": datetime.now().isoformat(),
            "config_path": str(config_path),
            "output_root": str(output_root),
            "metric_key": args.rssi_metric,
            "plot_metric_keys": list(_PLOT_METRIC_KEYS),
            "comparison_metric_pairs": [
                {
                    "all_metric": all_metric,
                    "priority_metric": priority_metric,
                    "title": title,
                }
                for all_metric, priority_metric, title in _COMPARISON_METRIC_PAIRS
            ],
            "plot_y_range": {
                "y_min": y_limits[0],
                "y_max": y_limits[1],
            },
            "num_aps_values": num_aps_values,
            "seeds": seeds,
            "aggregate": aggregate_rows,
        },
    )

    for all_metric_key, priority_metric_key, metric_title in _COMPARISON_METRIC_PAIRS:
        comparison_plot_path = output_root / f"num_aps_vs_{all_metric_key}_all_vs_priority.png"
        if _plot_metric_with_std(
            aggregate_rows=aggregate_rows,
            series_specs=[
                {
                    "mean_field": f"{all_metric_key}_mean",
                    "std_field": f"{all_metric_key}_std",
                    "label": _METRIC_LABELS.get(all_metric_key, all_metric_key),
                    "color": "tab:blue",
                    "linestyle": "-",
                },
                {
                    "mean_field": f"{priority_metric_key}_mean",
                    "std_field": f"{priority_metric_key}_std",
                    "label": _METRIC_LABELS.get(priority_metric_key, priority_metric_key),
                    "color": "tab:orange",
                    "linestyle": "--",
                },
            ],
            save_path=comparison_plot_path,
            title=f"AP Count Sweep: {metric_title} (All vs Priority)",
            y_axis_label="RSSI (dBm)",
            y_limits=y_limits,
        ):
            print(f"[plot] saved: {comparison_plot_path}")
        else:
            print(
                "[plot] skipped comparison "
                f"{all_metric_key} vs {priority_metric_key}: no successful runs with metric values"
            )

    if args.rssi_metric not in _PLOT_METRIC_KEYS:
        extra_plot_path = output_root / f"num_aps_vs_{args.rssi_metric}.png"
        if _plot_metric_with_std(
            aggregate_rows=aggregate_rows,
            series_specs=[
                {
                    "mean_field": "mean_rssi",
                    "std_field": "std_rssi",
                    "label": args.rssi_metric,
                    "color": "tab:green",
                    "linestyle": "-",
                }
            ],
            save_path=extra_plot_path,
            title=f"AP Count Sweep: {args.rssi_metric} vs Number of APs",
            y_axis_label="RSSI (dBm)",
            y_limits=y_limits,
        ):
            print(f"[plot] saved extra metric: {extra_plot_path}")
        else:
            print(f"[plot] skipped extra metric {args.rssi_metric}: no successful runs")

    print(f"[done] per-run CSV: {run_rows_path}")
    print(f"[done] aggregate CSV: {aggregate_rows_path}")
    print(f"[done] aggregate JSON: {aggregate_json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
