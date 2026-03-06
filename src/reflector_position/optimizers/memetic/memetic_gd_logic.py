"""Memetic targeted GD micro-exploitation orchestration.

Phase 3 module for running gradient-descent fine-tuning on a curated set of
GA-derived seeds using an already initialized ``RayParallelOptimizer``.

The function in this module intentionally does not initialize Ray, create actor
pools, or mutate external orchestrator state beyond invoking
``ray_optimizer.run(...)``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from reflector_position.metrics import POWER_EPSILON


_GD_OPTIMIZE_PARAM_KEYS = {
    "num_iterations",
    "learning_rate",
    "samples_per_tx",
    "max_depth",
    "use_soft_min",
    "temperature",
    "shadow_quantile",
    "fairness_loss_type",
    "coverage_threshold_dbm",
    "alpha",
    "beta",
    "coverage_temperature",
    "verbose",
}

# Keys that can appear in memetic bridge/orchestration payloads but must never
# be forwarded to GradientDescentAPOptimizer.__init__(...).
_GD_NON_INIT_TASK_KEYS = {
    "scene_config",
    "initial_orientations",  # bridge alias (3D), GD expects initial_directions_xy
    "reflector_target",  # bridge alias, GD expects initial_focal_point
    "reflector_u",  # not accepted by current GD optimizer constructor
    "reflector_v",  # not accepted by current GD optimizer constructor
    "initial_ga_rss_dbm",  # analysis metadata
    "ga_seed_metric_dbm",  # analysis metadata
    "seed_fitness_dbm",  # analysis metadata
    "initial_metric_dbm",  # analysis metadata
}


def _rss_watts_to_dbm(rss_watt: float) -> float:
    """Convert linear Watts to dBm."""
    return 10.0 * np.log10(max(float(rss_watt), POWER_EPSILON)) + 30.0


def _split_task_and_opt_params(task: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split one bridge task into optimizer-init kwargs and optimize kwargs."""
    init_kwargs: Dict[str, Any] = {}
    optimize_kwargs: Dict[str, Any] = {}
    for key, value in task.items():
        if key in _GD_OPTIMIZE_PARAM_KEYS:
            optimize_kwargs[key] = value
        elif key in _GD_NON_INIT_TASK_KEYS:
            continue
        else:
            init_kwargs[key] = value
    return init_kwargs, optimize_kwargs


def _extract_initial_metric_dbm(
    task: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Optional[float]:
    """Extract initial metric (dBm) for one targeted GD run.

    Priority:
    1. Explicit GA-seed metric on task payload.
    2. First value from GD history ``min_rss_dbm_values``.
    3. First value from GD history ``min_rss_values`` converted to dBm.
    """
    for key in (
        "initial_ga_rss_dbm",
        "ga_seed_metric_dbm",
        "seed_fitness_dbm",
        "initial_metric_dbm",
    ):
        if key in task and task[key] is not None:
            return float(task[key])

    history = result.get("history", {})
    if isinstance(history, Mapping):
        dbm_series = history.get("min_rss_dbm_values")
        if isinstance(dbm_series, Sequence) and len(dbm_series) > 0:
            return float(dbm_series[0])

        lin_series = history.get("min_rss_values")
        if isinstance(lin_series, Sequence) and len(lin_series) > 0:
            return _rss_watts_to_dbm(float(lin_series[0]))

    return None


def _extract_final_metric_dbm(result: Mapping[str, Any]) -> Optional[float]:
    """Extract final-iteration GD metric (dBm) when available."""
    history = result.get("history", {})
    if isinstance(history, Mapping):
        dbm_series = history.get("min_rss_dbm_values")
        if isinstance(dbm_series, Sequence) and len(dbm_series) > 0:
            return float(dbm_series[-1])

        lin_series = history.get("min_rss_values")
        if isinstance(lin_series, Sequence) and len(lin_series) > 0:
            return _rss_watts_to_dbm(float(lin_series[-1]))

    best_dbm = result.get("best_metric_dbm")
    return float(best_dbm) if best_dbm is not None else None


def _extract_scene_config(
    gd_tasks: List[Dict[str, Any]],
    ray_optimizer: Any,
) -> Dict[str, Any]:
    """Resolve scene configuration without instantiating Ray inside this module."""
    # Preferred: explicit scene config in any task
    for task in gd_tasks:
        if "scene_config" in task and task["scene_config"] is not None:
            if not isinstance(task["scene_config"], Mapping):
                raise ValueError("'scene_config' in gd_tasks must be a mapping/dict.")
            return dict(task["scene_config"])

    # Fallback: existing optimizer cached scene config
    cached = getattr(ray_optimizer, "_scene_config", None)
    if isinstance(cached, Mapping):
        return dict(cached)

    raise ValueError(
        "Scene configuration is missing. Provide 'scene_config' in gd_tasks or "
        "ensure ray_optimizer has a cached _scene_config from a prior run."
    )


def run_targeted_gd_exploitation(
    gd_tasks: List[Dict[str, Any]],
    ray_optimizer: Any,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run targeted GD exploitation for K memetic seeds and aggregate outcomes.

    Parameters
    ----------
    gd_tasks : list[dict[str, Any]]
        List of translated work-item dictionaries from Phase 2 bridge.
        Each item must be compatible with ``GradientDescentAPOptimizer``
        initialization fields (e.g. ``initial_positions``, ``fixed_z``, etc.).

        Optional per-item metadata keys used by this function:
        - ``scene_config``: scene config dict (if optimizer has no cached scene)
        - ``initial_ga_rss_dbm`` / ``ga_seed_metric_dbm`` / ``seed_fitness_dbm``:
          baseline GA metric for delta reporting.

        Optional optimize-parameter keys are recognized and removed from task
        kwargs before submission, then passed as shared ``optimization_params``
        to Ray when consistent across tasks.

    ray_optimizer : RayParallelOptimizer
        Already instantiated distributed optimizer. This function does not
        create Ray runtime or ActorPool.

    verbose : bool, default=True
        If True, prints per-seed initial/final/delta metrics and summary stats.

    Returns
    -------
    dict[str, Any]
        {
          "global_best_result": dict | None,
          "all_fine_tuned_results": list[dict],
                    "parallel_run_metadata": dict,
          "metrics": {
              "num_tasks": int,
              "num_results": int,
              "max_improvement_db": float | None,
              "mean_improvement_db": float | None,
              "min_improvement_db": float | None,
              "best_final_metric_dbm": float | None,
          },
          "per_seed_analysis": list[dict],
        }

    Notes
    -----
    - Supports both result contracts:
      1) tuple ``(best_result_dict, all_results_list)``
      2) dict with keys ``best_result`` and ``all_results``.
    - If ``gd_tasks`` is empty, returns empty structures and skips Ray call.
    """
    if not gd_tasks:
        if verbose:
            print("[memetic-gd] Warning: no gd_tasks provided; skipping exploitation run.")
        return {
            "global_best_result": None,
            "all_fine_tuned_results": [],
            "metrics": {
                "num_tasks": 0,
                "num_results": 0,
                "max_improvement_db": None,
                "mean_improvement_db": None,
                "min_improvement_db": None,
                "best_final_metric_dbm": None,
            },
            "per_seed_analysis": [],
                "parallel_run_metadata": {},
        }

    scene_config = _extract_scene_config(gd_tasks, ray_optimizer)

    init_work_items: List[Dict[str, Any]] = []
    optimize_param_candidates: List[Dict[str, Any]] = []

    for task in gd_tasks:
        init_kwargs, optimize_kwargs = _split_task_and_opt_params(task)
        init_work_items.append(init_kwargs)
        optimize_param_candidates.append(optimize_kwargs)

    # Ensure optimize params are consistent across tasks because
    # RayParallelOptimizer.run accepts one shared optimization_params dict.
    optimization_params: Dict[str, Any] = optimize_param_candidates[0] if optimize_param_candidates else {}
    for i, candidate in enumerate(optimize_param_candidates[1:], start=1):
        if candidate != optimization_params:
            raise ValueError(
                "Inconsistent per-task GD optimize parameters detected. "
                f"Task #0 params={optimization_params}, task #{i} params={candidate}. "
                "Use shared gd_hyperparams in bridge stage for Ray batch execution."
            )

    if "verbose" not in optimization_params:
        optimization_params["verbose"] = False

    run_output = ray_optimizer.run(
        scene_config=scene_config,
        optimizer_method="gradient_descent",
        work_items=init_work_items,
        optimization_params=optimization_params,
        verbose=verbose,
    )

    parallel_run_metadata: Dict[str, Any] = {}
    if isinstance(run_output, tuple) and len(run_output) == 2:
        best_result = run_output[0]
        all_results = run_output[1]
    elif isinstance(run_output, Mapping):
        best_result = run_output.get("best_result")
        all_results = run_output.get("all_results", [])
        for key in ("aggregate_stats", "pool_info", "total_time"):
            if key in run_output:
                parallel_run_metadata[key] = run_output[key]
    else:
        raise RuntimeError(
            "Unsupported return type from ray_optimizer.run(). Expected tuple or mapping."
        )

    if not isinstance(all_results, list):
        raise RuntimeError("ray_optimizer.run() returned malformed all_results payload.")

    # Task-ID keyed lookup for deterministic per-seed pairing.
    task_by_id: Dict[int, Dict[str, Any]] = {
        int(task_id): result
        for task_id, result in (
            (int(r.get("task_id", i)), r) for i, r in enumerate(all_results)
        )
    }

    per_seed_analysis: List[Dict[str, Any]] = []
    deltas: List[float] = []

    for idx, original_task in enumerate(gd_tasks):
        result = task_by_id.get(idx)
        if result is None:
            analysis = {
                "seed_index": idx,
                "initial_ga_rss_dbm": None,
                "final_gd_rss_dbm": None,
                "delta_improvement_db": None,
                "status": "missing_result",
            }
            per_seed_analysis.append(analysis)
            continue

        initial_dbm = _extract_initial_metric_dbm(original_task, result)
        best_dbm = float(result.get("best_metric_dbm")) if result.get("best_metric_dbm") is not None else None
        final_dbm = _extract_final_metric_dbm(result)
        delta_best_db = (best_dbm - initial_dbm) if (best_dbm is not None and initial_dbm is not None) else None
        delta_final_db = (final_dbm - initial_dbm) if (final_dbm is not None and initial_dbm is not None) else None

        if delta_best_db is not None:
            deltas.append(float(delta_best_db))

        analysis = {
            "seed_index": idx,
            "task_id": int(result.get("task_id", idx)),
            "worker_id": result.get("worker_id"),
            "initial_ga_rss_dbm": initial_dbm,
            "best_gd_rss_dbm": best_dbm,
            "final_gd_rss_dbm": final_dbm,
            "delta_improvement_db": delta_best_db,
            "delta_best_improvement_db": delta_best_db,
            "delta_final_improvement_db": delta_final_db,
            "status": "ok",
        }
        per_seed_analysis.append(analysis)

    # Identify absolute global optimum by final GD metric.
    global_best_result: Optional[Dict[str, Any]] = None
    best_final_metric_dbm: Optional[float] = None
    for res in all_results:
        res_dbm = res.get("best_metric_dbm")
        if res_dbm is None:
            continue
        res_dbm = float(res_dbm)
        if best_final_metric_dbm is None or res_dbm > best_final_metric_dbm:
            best_final_metric_dbm = res_dbm
            global_best_result = res

    metrics = {
        "num_tasks": len(gd_tasks),
        "num_results": len(all_results),
        "max_improvement_db": float(max(deltas)) if deltas else None,
        "mean_improvement_db": float(np.mean(deltas)) if deltas else None,
        "min_improvement_db": float(min(deltas)) if deltas else None,
        "best_final_metric_dbm": best_final_metric_dbm,
    }

    if verbose:
        print("\n" + "=" * 80)
        print("TARGETED GD MICRO-EXPLOITATION SUMMARY")
        print("=" * 80)
        print("Seed | Initial GA (dBm) | Best GD (dBm) | Delta_best (dB)")
        print("-" * 80)
        for row in per_seed_analysis:
            seed_idx = row["seed_index"]
            init_dbm = row["initial_ga_rss_dbm"]
            fin_dbm = row["best_gd_rss_dbm"]
            delta_db = row["delta_best_improvement_db"]

            init_txt = f"{init_dbm:>16.2f}" if init_dbm is not None else "       N/A       "
            fin_txt = f"{fin_dbm:>14.2f}" if fin_dbm is not None else "      N/A      "
            delta_txt = f"{delta_db:>10.2f}" if delta_db is not None else "   N/A    "
            print(f"{seed_idx:>4d} | {init_txt} | {fin_txt} | {delta_txt}")

        print("-" * 80)
        print(
            "Max/Mean/Min Delta (dB): "
            f"{metrics['max_improvement_db'] if metrics['max_improvement_db'] is not None else 'N/A'} / "
            f"{metrics['mean_improvement_db'] if metrics['mean_improvement_db'] is not None else 'N/A'} / "
            f"{metrics['min_improvement_db'] if metrics['min_improvement_db'] is not None else 'N/A'}"
        )
        if global_best_result is not None:
            print(
                "Global best GD: "
                f"task #{global_best_result.get('task_id')} | "
                f"worker #{global_best_result.get('worker_id')} | "
                f"best_metric_dbm={float(global_best_result.get('best_metric_dbm')):.2f}"
            )
        else:
            print("Global best GD: N/A")
        print("=" * 80)

    return {
        "global_best_result": global_best_result,
        "all_fine_tuned_results": all_results,
        "parallel_run_metadata": parallel_run_metadata,
        "metrics": metrics,
        "per_seed_analysis": per_seed_analysis,
    }
