"""Memetic targeted GD micro-exploitation orchestration.

Phase 3 module for running gradient-descent fine-tuning on a curated set of
GA-derived seeds using an already initialized raw Ray optimizer.

The function in this module intentionally does not initialize Ray, create actor
pools, or mutate external orchestrator state beyond invoking
``ray_optimizer.run(...)``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_GD_OPTIMIZE_PARAM_KEYS = {
    "num_iterations",
    "learning_rate",
    "samples_per_tx",
    "max_depth",
    "softmin_temperature",
    "temperature",
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
    "initial_primary_loss",  # analysis metadata
}


def _extract_history(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the standardized GD history mapping when present."""
    history = result.get("history", {})
    return history if isinstance(history, Mapping) else {}


def _extract_result_summary(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the standardized GD result summary when present."""
    summary = result.get("results", {})
    return summary if isinstance(summary, Mapping) else {}


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


def _extract_initial_primary_loss(
    task: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Optional[float]:
    """Extract the starting primary loss for one targeted GD run."""
    task_value = task.get("initial_primary_loss")
    if task_value is not None:
        return float(task_value)

    history = _extract_history(result)
    primary_series = history.get("primary_loss")
    if isinstance(primary_series, Sequence) and len(primary_series) > 0:
        return float(primary_series[0])

    return None


def _extract_final_primary_loss(result: Mapping[str, Any]) -> Optional[float]:
    """Extract the final primary loss when available."""
    history = _extract_history(result)
    primary_series = history.get("primary_loss")
    if isinstance(primary_series, Sequence) and len(primary_series) > 0:
        return float(primary_series[-1])

    summary = _extract_result_summary(result)
    summary_value = summary.get("final_primary_loss", summary.get("primary_loss"))
    return float(summary_value) if summary_value is not None else None


def _extract_best_primary_loss(result: Mapping[str, Any]) -> Optional[float]:
    """Extract the best observed primary loss from raw or processed payloads."""
    history = _extract_history(result)
    primary_series = history.get("primary_loss")
    if isinstance(primary_series, Sequence) and len(primary_series) > 0:
        return float(min(float(value) for value in primary_series))

    summary = _extract_result_summary(result)
    summary_value = summary.get("primary_loss")
    return float(summary_value) if summary_value is not None else None


def _extract_best_loss_components(result: Mapping[str, Any]) -> Dict[str, float]:
    """Extract the best standardized auxiliary loss dictionary."""
    summary = _extract_result_summary(result)
    components = summary.get("loss_components")
    if isinstance(components, Mapping):
        return {str(name): float(value) for name, value in components.items()}
    return {}


def _extract_best_physical_metrics(result: Mapping[str, Any]) -> Dict[str, float]:
    """Extract the best standardized physical metrics dictionary."""
    summary = _extract_result_summary(result)
    metrics = summary.get("physical_metrics")
    if isinstance(metrics, Mapping):
        return {str(name): float(value) for name, value in metrics.items()}
    return {}


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
        - ``initial_primary_loss``: optional baseline loss for delta reporting.

        Optional optimize-parameter keys are recognized and removed from task
        kwargs before submission, then passed as shared ``optimization_params``
        to Ray when consistent across tasks.

    ray_optimizer : RawRayParallelOptimizer
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
              "max_loss_reduction": float | None,
              "mean_loss_reduction": float | None,
              "min_loss_reduction": float | None,
              "best_primary_loss": float | None,
          },
          "per_seed_analysis": list[dict],
        }

    Notes
    -----
        - Consumes raw worker outputs from ``RawRayParallelOptimizer``.
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
                "max_loss_reduction": None,
                "mean_loss_reduction": None,
                "min_loss_reduction": None,
                "best_primary_loss": None,
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
    # RawRayParallelOptimizer.run accepts one shared optimization_params dict.
    optimization_params: Dict[str, Any] = optimize_param_candidates[0] if optimize_param_candidates else {}
    for i, candidate in enumerate(optimize_param_candidates[1:], start=1):
        if candidate != optimization_params:
            raise ValueError(
                "Inconsistent per-task GD optimize parameters detected. "
                f"Task #0 params={optimization_params}, task #{i} params={candidate}. "
                "Use shared gd_optimization_params in bridge stage for Ray batch execution."
            )

    if "verbose" not in optimization_params:
        optimization_params["verbose"] = False

    run_output = ray_optimizer.run(
        scene_config=scene_config,
        optimizer_method="memetic_gd",
        work_items=init_work_items,
        optimization_params=optimization_params,
        verbose=verbose,
    )

    parallel_run_metadata: Dict[str, Any] = {}
    if isinstance(run_output, Mapping):
        all_results = run_output.get("all_results", [])
        for key in ("aggregate_stats", "pool_info", "total_time"):
            if key in run_output:
                parallel_run_metadata[key] = run_output[key]
    else:
        raise RuntimeError(
            "Unsupported return type from ray_optimizer.run(). Expected mapping."
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
                "initial_primary_loss": None,
                "best_primary_loss": None,
                "final_primary_loss": None,
                "delta_loss": None,
                "delta_best_loss": None,
                "loss_components": {},
                "physical_metrics": {},
                "status": "missing_result",
            }
            per_seed_analysis.append(analysis)
            continue

        initial_primary_loss = _extract_initial_primary_loss(original_task, result)
        best_primary_loss = _extract_best_primary_loss(result)
        final_primary_loss = _extract_final_primary_loss(result)
        delta_best_loss = (
            initial_primary_loss - best_primary_loss
            if (best_primary_loss is not None and initial_primary_loss is not None)
            else None
        )
        delta_loss = (
            initial_primary_loss - final_primary_loss
            if (final_primary_loss is not None and initial_primary_loss is not None)
            else None
        )

        if delta_loss is not None:
            deltas.append(float(delta_loss))

        analysis = {
            "seed_index": idx,
            "task_id": int(result.get("task_id", idx)),
            "worker_id": result.get("worker_id"),
            "initial_primary_loss": initial_primary_loss,
            "best_primary_loss": best_primary_loss,
            "final_primary_loss": final_primary_loss,
            "delta_loss": delta_loss,
            "delta_best_loss": delta_best_loss,
            "loss_components": _extract_best_loss_components(result),
            "physical_metrics": _extract_best_physical_metrics(result),
            "status": "ok",
        }
        per_seed_analysis.append(analysis)

    # Identify absolute global optimum by the minimum observed primary loss.
    global_best_result: Optional[Dict[str, Any]] = None
    best_primary_loss: Optional[float] = None
    for res in all_results:
        res_loss = _extract_best_primary_loss(res)
        if res_loss is None:
            continue
        res_loss = float(res_loss)
        if best_primary_loss is None or res_loss < best_primary_loss:
            best_primary_loss = res_loss
            global_best_result = res

    metrics = {
        "num_tasks": len(gd_tasks),
        "num_results": len(all_results),
        "max_loss_reduction": float(max(deltas)) if deltas else None,
        "mean_loss_reduction": float(np.mean(deltas)) if deltas else None,
        "min_loss_reduction": float(min(deltas)) if deltas else None,
        "best_primary_loss": best_primary_loss,
    }

    if verbose:
        print("\n" + "=" * 80)
        print("TARGETED GD MICRO-EXPLOITATION SUMMARY")
        print("=" * 80)
        print("Seed | Initial Loss | Best Loss | Final Loss | Delta")
        print("-" * 80)
        for row in per_seed_analysis:
            seed_idx = row["seed_index"]
            initial_value = row["initial_primary_loss"]
            best_value = row["best_primary_loss"]
            final_value = row["final_primary_loss"]
            delta_value = row["delta_loss"]

            initial_txt = f"{initial_value:>12.6f}" if initial_value is not None else "    N/A     "
            best_txt = f"{best_value:>9.6f}" if best_value is not None else "   N/A   "
            final_txt = f"{final_value:>10.6f}" if final_value is not None else "    N/A   "
            delta_txt = f"{delta_value:>10.6f}" if delta_value is not None else "   N/A    "
            print(f"{seed_idx:>4d} | {initial_txt} | {best_txt} | {final_txt} | {delta_txt}")

            loss_components = row.get("loss_components", {})
            if isinstance(loss_components, Mapping) and loss_components:
                for name, value in loss_components.items():
                    print(f"     component {name}: {float(value):.6f}")

        print("-" * 80)
        print(
            "Max/Mean/Min Loss Reduction: "
            f"{metrics['max_loss_reduction'] if metrics['max_loss_reduction'] is not None else 'N/A'} / "
            f"{metrics['mean_loss_reduction'] if metrics['mean_loss_reduction'] is not None else 'N/A'} / "
            f"{metrics['min_loss_reduction'] if metrics['min_loss_reduction'] is not None else 'N/A'}"
        )
        if global_best_result is not None:
            print(
                "Global best GD: "
                f"task #{global_best_result.get('task_id')} | "
                f"worker #{global_best_result.get('worker_id')} | "
                f"primary_loss={float(best_primary_loss):.6f}"
            )
            best_components = _extract_best_loss_components(global_best_result)
            if best_components:
                for name, value in best_components.items():
                    print(f"  {name}: {value:.6f}")
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
