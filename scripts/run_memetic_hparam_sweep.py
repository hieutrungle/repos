#!/usr/bin/env python3
"""Sequential hyperparameter sweep runner for the memetic pipeline.

This script expands a sweep specification into concrete memetic pipeline runs,
executes them one by one, and stores each run in its own subfolder.

Typical usage:

    python scripts/run_memetic_hparam_sweep.py \
        --base-config configs/memetic_pipeline_config.json \
        --sweep-config configs/memetic_hparam_sweep.example.json

The per-trial output layout is:

    <output_root>/
      <trial_name>/
        artifacts/
        plots/
      logs/
        <trial_name>.log
      sweep_plan.json
      sweep_summary.json
      sweep_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_BASE_CONFIG_PATH = REPO_ROOT / "configs" / "memetic_pipeline_config.json"
DEFAULT_SWEEP_CONFIG_PATH = REPO_ROOT / "configs" / "memetic_hparam_sweep.example.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reflector_position.optimizers.memetic.run_memetic_pipeline import (  # noqa: E402
    _default_memetic_config,
    run_memetic_optimization,
)


class TeeStream:
    """Write output to the terminal and to a log file."""

    def __init__(self, terminal_stream: Any, file_stream: Any) -> None:
        self.terminal_stream = terminal_stream
        self.file_stream = file_stream
        self.encoding = getattr(terminal_stream, "encoding", None)
        self.errors = getattr(terminal_stream, "errors", None)

    def write(self, data: str) -> None:
        self.terminal_stream.write(data)
        if getattr(self.file_stream, "closed", False):
            return
        try:
            self.file_stream.write(data)
        except ValueError:
            # Some libraries keep a reference to stdout/stderr and write during
            # process teardown after the log file has been closed.
            return

    def flush(self) -> None:
        self.terminal_stream.flush()
        if getattr(self.file_stream, "closed", False):
            return
        try:
            self.file_stream.flush()
        except ValueError:
            return

    def fileno(self) -> int:
        """Expose the terminal file descriptor for libraries that require it."""
        return self.terminal_stream.fileno()

    def isatty(self) -> bool:
        """Report whether the wrapped terminal behaves like a TTY."""
        return bool(getattr(self.terminal_stream, "isatty", lambda: False)())

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown stream attributes to the terminal stream."""
        return getattr(self.terminal_stream, name)


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` in place."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, Mapping):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file and assert a dictionary root."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    """Write formatted JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: List[Mapping[str, Any]], fieldnames: List[str]) -> None:
    """Write summary rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _sanitize_name(name: str) -> str:
    """Convert one trial name into a safe folder name."""
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name).strip("_") or "trial"


def _abbr_value(value: Any) -> str:
    """Create a short readable token for one sweep value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}".replace("-", "m").replace(".", "p")
    return _sanitize_name(str(value))


def _set_nested(mapping: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Assign one dotted key into a nested dictionary."""
    parts = dotted_key.split(".")
    cursor = mapping
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _ensure_list(value: Any) -> List[Any]:
    """Wrap non-list values so sweep expansion can iterate uniformly."""
    if isinstance(value, list):
        return value
    return [value]


def _extract_trial_overrides(trial: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the effective override payload for one explicit trial."""
    if "overrides" in trial:
        overrides = trial.get("overrides")
        if not isinstance(overrides, Mapping):
            raise ValueError("trial 'overrides' must be a mapping when provided")
        return deepcopy(dict(overrides))

    excluded = {"name", "enabled", "description"}
    return {
        str(key): deepcopy(value)
        for key, value in trial.items()
        if key not in excluded and not str(key).startswith("_")
    }


def _build_explicit_trials(spec: Mapping[str, Any], shared: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Build explicit trials from the sweep specification."""
    results: List[Dict[str, Any]] = []
    for index, trial in enumerate(spec.get("trials", []), start=1):
        if not isinstance(trial, Mapping):
            raise ValueError(f"Trial #{index} must be a mapping")
        if trial.get("enabled", True) is False:
            continue

        overrides = deepcopy(dict(shared))
        _deep_update(overrides, _extract_trial_overrides(trial))

        name = trial.get("name") or f"trial_{index:03d}"
        results.append(
            {
                "name": _sanitize_name(str(name)),
                "overrides": overrides,
                "source": "explicit",
            }
        )
    return results


def _build_sweep_group_trials(spec: Mapping[str, Any], shared: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expand sweep groups into concrete trial definitions."""
    results: List[Dict[str, Any]] = []
    for group_index, group in enumerate(spec.get("sweep_groups", []), start=1):
        if not isinstance(group, Mapping):
            raise ValueError(f"Sweep group #{group_index} must be a mapping")
        if group.get("enabled", True) is False:
            continue

        grid = group.get("grid", {})
        if not isinstance(grid, Mapping) or not grid:
            raise ValueError(f"Sweep group #{group_index} must define a non-empty 'grid'")

        base_overrides = deepcopy(dict(shared))
        base_payload = group.get("base_overrides", group.get("base", {}))
        if base_payload:
            if not isinstance(base_payload, Mapping):
                raise ValueError(f"Sweep group #{group_index} base overrides must be a mapping")
            _deep_update(base_overrides, dict(base_payload))

        grid_keys = [str(key) for key in grid.keys()]
        grid_values = [_ensure_list(grid[key]) for key in grid_keys]
        name_prefix = _sanitize_name(str(group.get("name_prefix", f"group_{group_index:02d}")))

        for combo_index, combo in enumerate(product(*grid_values), start=1):
            overrides = deepcopy(base_overrides)
            combo_parts: List[str] = []
            for key, value in zip(grid_keys, combo):
                _set_nested(overrides, key, value)
                combo_parts.append(f"{key.split('.')[-1]}_{_abbr_value(value)}")

            name = f"{name_prefix}_{combo_index:03d}_{'_'.join(combo_parts)}"
            results.append(
                {
                    "name": _sanitize_name(name),
                    "overrides": overrides,
                    "source": "sweep_group",
                }
            )
    return results


def _build_trials(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Build the full ordered trial plan from explicit trials and sweep groups."""
    shared = spec.get("shared", {})
    if shared and not isinstance(shared, Mapping):
        raise ValueError("'shared' must be a mapping when provided")

    shared_mapping = dict(shared) if isinstance(shared, Mapping) else {}
    trials = _build_explicit_trials(spec, shared_mapping)
    trials.extend(_build_sweep_group_trials(spec, shared_mapping))

    seen = set()
    for trial in trials:
        if trial["name"] in seen:
            raise ValueError(f"Duplicate trial name detected: {trial['name']}")
        seen.add(trial["name"])

    return trials


def _safe_dict(value: Any) -> Dict[str, Any]:
    """Return value as a dict, or empty dict if not a mapping."""
    return dict(value) if isinstance(value, Mapping) else {}


def _extract_physical_metric_fields(d: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    """Extract RSSI and coverage fields from a physical_metrics snapshot."""
    return {
        f"{prefix}mean_rssi": d.get("mean_rss_dbm"),
        f"{prefix}min_rssi": d.get("min_rss_dbm"),
        f"{prefix}p5_rssi": d.get("p5_rss_dbm"),
        f"{prefix}coverage_pct": d.get("coverage_pct"),
    }


def _extract_loss_component_fields(d: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    """Extract loss component scalar fields from a loss_components snapshot."""
    return {
        f"{prefix}softmin_loss": d.get("softmin_loss"),
        f"{prefix}coverage_loss": d.get("coverage_loss"),
        f"{prefix}repulsion_loss": d.get("repulsion_loss"),
    }


def _summarize_trial(
    trial_name: str,
    status: str,
    output_dir: Path,
    overrides: Mapping[str, Any],
    result: Mapping[str, Any] | None = None,
    error: str | None = None,
    wall_clock_sec: float | None = None,
) -> Dict[str, Any]:
    """Build one comparable summary row for a trial.

    Extracts per-method (GA and GD) metrics for three checkpoints:
    - initial  : state before any GD update (first snapshot in GD history)
    - best     : configuration with the lowest observed primary loss
    - final    : configuration at the last GD iteration

    Physical metrics reported: mean_rssi, min_rssi, p5_rssi, coverage_pct
    Loss components reported : softmin_loss, coverage_loss, repulsion_loss
    """
    result = result or {}
    gd_results = _safe_dict(result.get("gd_results")) if isinstance(result, Mapping) else {}
    ga_results = _safe_dict(result.get("ga_results")) if isinstance(result, Mapping) else {}
    timings = _safe_dict(result.get("timings")) if isinstance(result, Mapping) else {}
    counts = _safe_dict(result.get("counts")) if isinstance(result, Mapping) else {}

    # ------------------------------------------------------------------
    # GA metrics (best individual across the entire evolution)
    # ------------------------------------------------------------------
    ga_best_primary_fitness = ga_results.get("best_primary_fitness")
    ga_best_primary_loss = (
        -float(ga_best_primary_fitness) if ga_best_primary_fitness is not None else None
    )
    ga_best_loss_components = _safe_dict(ga_results.get("best_loss_components"))
    ga_best_physical_metrics = _safe_dict(ga_results.get("best_physical_metrics"))

    ga_fields: Dict[str, Any] = {
        "ga_best_primary_loss": ga_best_primary_loss,
        **_extract_loss_component_fields(ga_best_loss_components, "ga_best_"),
        **_extract_physical_metric_fields(ga_best_physical_metrics, "ga_best_"),
    }

    # ------------------------------------------------------------------
    # GD metrics — extracted from the global best seed's raw worker output
    # ------------------------------------------------------------------
    global_best = _safe_dict(gd_results.get("global_best_result"))
    gd_history = _safe_dict(global_best.get("history"))
    gd_result_summary = _safe_dict(global_best.get("results"))

    # Snapshot lists from GD history (index 0 = initial, -1 = final state)
    history_primary_loss: List[Any] = gd_history.get("primary_loss") or []
    history_loss_components: List[Any] = gd_history.get("loss_components") or []
    history_physical_metrics: List[Any] = gd_history.get("physical_metrics") or []

    def _nth(lst: List[Any], n: int) -> Dict[str, Any]:
        """Safely return the nth element of a list as a dict."""
        if lst and abs(n) < len(lst):
            return _safe_dict(lst[n])
        return {}

    # Initial checkpoint
    gd_initial_loss = float(history_primary_loss[0]) if history_primary_loss else None
    gd_initial_loss_components = _nth(history_loss_components, 0)
    gd_initial_physical_metrics = _nth(history_physical_metrics, 0)

    # Best checkpoint (taken from standardized results payload)
    gd_best_primary_loss = gd_result_summary.get("primary_loss")
    if gd_best_primary_loss is not None:
        gd_best_primary_loss = float(gd_best_primary_loss)
    else:
        gd_best_primary_loss = gd_results.get("metrics", {}).get("best_primary_loss")
    gd_best_loss_components = _safe_dict(gd_result_summary.get("loss_components"))
    gd_best_physical_metrics = _safe_dict(gd_result_summary.get("physical_metrics"))

    # Final checkpoint
    gd_final_primary_loss = gd_result_summary.get("final_primary_loss")
    if gd_final_primary_loss is not None:
        gd_final_primary_loss = float(gd_final_primary_loss)
    elif history_primary_loss:
        gd_final_primary_loss = float(history_primary_loss[-1])
    gd_final_loss_components = _safe_dict(
        gd_result_summary.get("final_loss_components") or _nth(history_loss_components, -1)
    )
    gd_final_physical_metrics = _safe_dict(
        gd_result_summary.get("final_physical_metrics") or _nth(history_physical_metrics, -1)
    )

    gd_fields: Dict[str, Any] = {
        "gd_initial_primary_loss": gd_initial_loss,
        **_extract_loss_component_fields(gd_initial_loss_components, "gd_initial_"),
        **_extract_physical_metric_fields(gd_initial_physical_metrics, "gd_initial_"),
        "gd_best_primary_loss": gd_best_primary_loss,
        **_extract_loss_component_fields(gd_best_loss_components, "gd_best_"),
        **_extract_physical_metric_fields(gd_best_physical_metrics, "gd_best_"),
        "gd_final_primary_loss": gd_final_primary_loss,
        **_extract_loss_component_fields(gd_final_loss_components, "gd_final_"),
        **_extract_physical_metric_fields(gd_final_physical_metrics, "gd_final_"),
    }

    return {
        "trial_name": trial_name,
        "status": status,
        "output_dir": str(output_dir),
        **ga_fields,
        **gd_fields,
        # Keep legacy key for backward compatibility
        "best_primary_loss": gd_best_primary_loss,
        "best_primary_fitness": ga_best_primary_fitness,
        "ga_duration_sec": timings.get("ga_duration_sec"),
        "gd_duration_sec": timings.get("gd_duration_sec"),
        "total_duration_sec": timings.get("total_duration_sec"),
        "runner_wall_clock_sec": wall_clock_sec,
        "num_ga_seeds": counts.get("num_ga_seeds"),
        "num_gd_tasks": counts.get("num_gd_tasks"),
        "num_gd_results": counts.get("num_gd_results"),
        "error": error,
        "overrides_json": json.dumps(overrides, ensure_ascii=False, sort_keys=True),
    }


def _rank_successful_trials(rows: List[Dict[str, Any]]) -> None:
    """Add best-loss ranks to successful rows in place."""
    successful = [
        row for row in rows
        if row.get("status") == "ok" and row.get("best_primary_loss") is not None
    ]
    successful.sort(key=lambda row: float(row["best_primary_loss"]))
    for rank, row in enumerate(successful, start=1):
        row["rank_by_best_primary_loss"] = rank
    for row in rows:
        row.setdefault("rank_by_best_primary_loss", None)


def _write_trial_trends(
    trial_name: str,
    result: Mapping[str, Any],
    trends_dir: Path,
) -> None:
    """Write per-trial GD iteration history and GA generation history CSVs.

    Files written (under ``trends_dir``):
    - ``<trial_name>_gd_history.csv``  — one row per GD snapshot (initial +
      one per iteration + final of the global best seed).
    - ``<trial_name>_ga_generations.csv`` — one row per GA generation.
    """
    trends_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # GD iteration history from the global best seed
    # ------------------------------------------------------------------
    gd_results = _safe_dict(result.get("gd_results"))
    global_best = _safe_dict(gd_results.get("global_best_result"))
    gd_history = _safe_dict(global_best.get("history"))

    primary_losses: List[Any] = gd_history.get("primary_loss") or []
    loss_components_list: List[Any] = gd_history.get("loss_components") or []
    physical_metrics_list: List[Any] = gd_history.get("physical_metrics") or []

    if primary_losses:
        n = len(primary_losses)
        gd_rows: List[Dict[str, Any]] = []
        for step in range(n):
            lc = _safe_dict(loss_components_list[step]) if step < len(loss_components_list) else {}
            pm = _safe_dict(physical_metrics_list[step]) if step < len(physical_metrics_list) else {}
            gd_rows.append({
                "step": step,
                "primary_loss": primary_losses[step],
                "softmin_loss": lc.get("softmin_loss"),
                "coverage_loss": lc.get("coverage_loss"),
                "repulsion_loss": lc.get("repulsion_loss"),
                "mean_rssi": pm.get("mean_rss_dbm"),
                "min_rssi": pm.get("min_rss_dbm"),
                "p5_rssi": pm.get("p5_rss_dbm"),
                "coverage_pct": pm.get("coverage_pct"),
            })
        gd_fieldnames = [
            "step", "primary_loss", "softmin_loss", "coverage_loss",
            "repulsion_loss", "mean_rssi", "min_rssi", "p5_rssi", "coverage_pct",
        ]
        _write_csv(trends_dir / f"{trial_name}_gd_history.csv", gd_rows, gd_fieldnames)

    # ------------------------------------------------------------------
    # GA generation history
    # ------------------------------------------------------------------
    ga_results = _safe_dict(result.get("ga_results"))
    generation_details: List[Any] = ga_results.get("generation_details") or []

    if generation_details:
        ga_rows: List[Dict[str, Any]] = []
        for entry in generation_details:
            entry_dict = _safe_dict(entry)
            ga_rows.append({
                "gen": entry_dict.get("gen"),
                "nevals": entry_dict.get("nevals"),
                "best_primary_fitness": entry_dict.get("best_primary_fitness"),
                "best_primary_loss": (
                    -float(entry_dict["best_primary_fitness"])
                    if entry_dict.get("best_primary_fitness") is not None else None
                ),
                "mean_primary_fitness": entry_dict.get("mean_primary_fitness"),
                "std": entry_dict.get("std"),
                "feasible_count": entry_dict.get("feasible_count"),
                "penalized_count": entry_dict.get("penalized_count"),
                "time": entry_dict.get("time"),
            })
        ga_fieldnames = [
            "gen", "nevals", "best_primary_fitness", "best_primary_loss",
            "mean_primary_fitness", "std", "feasible_count", "penalized_count", "time",
        ]
        _write_csv(trends_dir / f"{trial_name}_ga_generations.csv", ga_rows, ga_fieldnames)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run sequential hyperparameter sweeps for the memetic pipeline.",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=str(DEFAULT_BASE_CONFIG_PATH),
        help="Base memetic pipeline config JSON.",
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default=str(DEFAULT_SWEEP_CONFIG_PATH),
        help="Sweep specification JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional root directory for all sweep runs.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Expand the sweep plan and write plan files without running trials.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of trials to execute after expansion.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one trial fails.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Force verbose logging inside each memetic pipeline run.",
    )
    return parser


def main() -> int:
    """Expand a sweep plan, run trials sequentially, and save a summary."""
    args = _build_parser().parse_args()

    base_config_path = Path(args.base_config).expanduser().resolve()
    sweep_config_path = Path(args.sweep_config).expanduser().resolve()

    base_config = _default_memetic_config()
    if base_config_path.exists():
        _deep_update(base_config, _load_json(base_config_path))
    else:
        print(f"[info] Base config not found; using in-code defaults: {base_config_path}")

    if not sweep_config_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {sweep_config_path}")
    sweep_spec = _load_json(sweep_config_path)

    trials = _build_trials(sweep_spec)
    if args.limit is not None:
        trials = trials[: max(0, int(args.limit))]
    if not trials:
        raise ValueError("Sweep plan expanded to zero trials")

    default_output_root = Path(str(base_config.get("output_dir", "results/experiments"))) / (
        "memetic_hparam_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_root = Path(args.output_dir) if args.output_dir else Path(str(sweep_spec.get("output_dir", default_output_root)))
    output_root = output_root.expanduser().resolve()
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan_payload = {
        "generated_at": datetime.now().isoformat(),
        "base_config_path": str(base_config_path),
        "sweep_config_path": str(sweep_config_path),
        "output_root": str(output_root),
        "num_trials": len(trials),
        "trials": trials,
    }
    _write_json(output_root / "sweep_plan.json", plan_payload)

    if args.generate_only:
        print(f"Wrote expanded sweep plan to: {output_root / 'sweep_plan.json'}")
        return 0

    summary_rows: List[Dict[str, Any]] = []
    print(f"[sweep] Output root: {output_root}")
    print(f"[sweep] Trials to run: {len(trials)}")

    for index, trial in enumerate(trials, start=1):
        trial_name = str(trial["name"])
        overrides = dict(trial["overrides"])
        trial_config = deepcopy(base_config)
        _deep_update(trial_config, overrides)
        trial_config["output_dir"] = str(output_root)
        trial_config["run_name"] = trial_name
        if args.verbose:
            trial_config["verbose"] = True

        log_path = logs_dir / f"{trial_name}.log"
        print(f"\n[sweep] Trial {index}/{len(trials)}: {trial_name}")

        start_time = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf-8") as handle:
                tee_out = TeeStream(sys.stdout, handle)
                tee_err = TeeStream(sys.stderr, handle)
                with redirect_stdout(tee_out), redirect_stderr(tee_err):
                    result = run_memetic_optimization(trial_config)

            wall_clock_sec = float(time.perf_counter() - start_time)
            trial_output_dir = output_root / trial_name
            row = _summarize_trial(
                trial_name=trial_name,
                status="ok",
                output_dir=trial_output_dir,
                overrides=overrides,
                result=result,
                wall_clock_sec=wall_clock_sec,
            )
            summary_rows.append(row)

            # Write per-trial trend CSVs (GD iteration history + GA generation stats)
            _write_trial_trends(
                trial_name=trial_name,
                result=result,
                trends_dir=output_root / "trends",
            )

            best_primary_loss = row.get("best_primary_loss")
            if best_primary_loss is not None:
                print(f"[sweep] Completed {trial_name} | best_primary_loss={float(best_primary_loss):.6f}")
            else:
                print(f"[sweep] Completed {trial_name}")
        except Exception as exc:
            wall_clock_sec = float(time.perf_counter() - start_time)
            trial_output_dir = output_root / trial_name
            error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            traceback_text = traceback.format_exc()
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("\n\n=== TRACEBACK ===\n")
                handle.write(traceback_text)

            summary_rows.append(
                _summarize_trial(
                    trial_name=trial_name,
                    status="failed",
                    output_dir=trial_output_dir,
                    overrides=overrides,
                    error=error_text,
                    wall_clock_sec=wall_clock_sec,
                )
            )
            print(f"[sweep] FAILED {trial_name} | {error_text}")
            if args.fail_fast:
                break

        _rank_successful_trials(summary_rows)
        summary_payload = {
            "generated_at": datetime.now().isoformat(),
            "base_config_path": str(base_config_path),
            "sweep_config_path": str(sweep_config_path),
            "output_root": str(output_root),
            "rows": summary_rows,
        }
        _write_json(output_root / "sweep_summary.json", summary_payload)
        fieldnames = [
            "trial_name",
            "status",
            "rank_by_best_primary_loss",
            # GA — best individual across entire evolution
            "ga_best_primary_loss",
            "ga_best_softmin_loss",
            "ga_best_coverage_loss",
            "ga_best_repulsion_loss",
            "ga_best_mean_rssi",
            "ga_best_min_rssi",
            "ga_best_p5_rssi",
            "ga_best_coverage_pct",
            # GD — initial snapshot (before any GD update)
            "gd_initial_primary_loss",
            "gd_initial_softmin_loss",
            "gd_initial_coverage_loss",
            "gd_initial_repulsion_loss",
            "gd_initial_mean_rssi",
            "gd_initial_min_rssi",
            "gd_initial_p5_rssi",
            "gd_initial_coverage_pct",
            # GD — best snapshot (lowest primary loss)
            "gd_best_primary_loss",
            "gd_best_softmin_loss",
            "gd_best_coverage_loss",
            "gd_best_repulsion_loss",
            "gd_best_mean_rssi",
            "gd_best_min_rssi",
            "gd_best_p5_rssi",
            "gd_best_coverage_pct",
            # GD — final snapshot (last iteration)
            "gd_final_primary_loss",
            "gd_final_softmin_loss",
            "gd_final_coverage_loss",
            "gd_final_repulsion_loss",
            "gd_final_mean_rssi",
            "gd_final_min_rssi",
            "gd_final_p5_rssi",
            "gd_final_coverage_pct",
            # Timing and counts
            "ga_duration_sec",
            "gd_duration_sec",
            "total_duration_sec",
            "runner_wall_clock_sec",
            "num_ga_seeds",
            "num_gd_tasks",
            "num_gd_results",
            # Legacy keys kept for backward compatibility
            "best_primary_loss",
            "best_primary_fitness",
            "output_dir",
            "error",
            "overrides_json",
        ]
        _write_csv(output_root / "sweep_summary.csv", summary_rows, fieldnames)

    print(f"\n[sweep] Summary written to: {output_root / 'sweep_summary.json'}")
    print(f"[sweep] CSV written to: {output_root / 'sweep_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())