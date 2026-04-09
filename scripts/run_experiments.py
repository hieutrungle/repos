"""Unified experiments entry point for memetic and baseline benchmarking.

Instructions
------------
1. Use one shared JSON config (HRBB-style schema is recommended).
2. Baseline methods are always forced to CUDA execution in this runner.
3. Choose a single method or run multiple methods in one command:
    - `memetic`, `random`, `kmeans`, `weighted_kmeans`, `random_gd`, `pso_gd`
   - `all_baselines`, `all`
4. Artifacts are saved in memetic-like folders per run:
   - `<run_dir>/artifacts/*.json|*.csv`
   - `<run_dir>/plots/*.html`
5. Per-method iteration traces are exported to CSV and plotted as trend charts.

python scripts/run_experiments.py --method all \
    --config configs/run_experiments_cuda_hrbb.json \
    --output_dir results/experiments
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import ray

from reflector_position.optimizers.baselines import (
    build_evaluator_task,
    run_kmeans_baseline,
    run_pso_gd_baseline,
    run_random_monte_carlo,
    run_random_multi_start_gd,
    run_weighted_kmeans_baseline,
)
from reflector_position.optimizers.baselines.static_comparison_plotting import (
    save_static_comparison_plots,
)
from reflector_position.optimizers.memetic.demand_weights import (
    generate_spatial_priority_map,
)
from reflector_position.optimizers.memetic.raw_ray_parallel_optimizer import (
    RawRayActorPoolExecutor,
    RawRayParallelOptimizer,
)
from reflector_position.optimizers.memetic.run_memetic_pipeline import (
    run_memetic_optimization,
)

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - plotting optional at runtime
    go = None


Bounds = Tuple[float, float]

_SINGLE_METHODS = ["memetic", "random", "kmeans", "weighted_kmeans", "random_gd", "pso_gd"]
_BASELINE_METHODS = ["random", "kmeans", "weighted_kmeans", "random_gd", "pso_gd"]
_METHOD_CHOICES = [*_SINGLE_METHODS, "all_baselines", "all"]

_OBJECTIVE_PARAM_DEFAULTS: Dict[str, Any] = {
    "alpha": 0.95,
    "beta": 0.05,
    "softmin_temperature": 0.15,
    "softmin_floor_dbm": -120.0,
    "softmin_ceil_dbm": -60.0,
    "coverage_threshold_dbm": -120.0,
    "coverage_temperature": 2.0,
}

_GA_EVALUATION_PARAM_DEFAULTS: Dict[str, Any] = {
    "samples_per_tx": 1_000_000,
    "max_depth": 13,
    "verbose": False,
}

_DEMAND_CONFIG_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "bounding_boxes": [],
    "box_weights": [],
    "apply_blur": False,
}


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for unified experiments execution."""
    parser = argparse.ArgumentParser(
        description=(
            "Run one or many optimization methods and save memetic-style "
            "artifacts/plots."
        )
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=_METHOD_CHOICES,
        help="Method to execute: single, all_baselines, or all.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to base JSON config file.",
    )
    parser.add_argument(
        "--output_dir",
        default="./results/experiments",
        type=str,
        help="Directory where run folders and artifacts are written.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        help="Optional run folder name.",
    )
    return parser.parse_args()


def _load_json_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate a JSON config payload."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError("Configuration JSON root must be an object/mapping.")

    return dict(payload)


def _require_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    """Return mapping as dict or raise a clear schema error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"'{field_name}' must be a mapping/object in the config.")
    return dict(value)


def _get_optional_mapping(config: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """Get optional config subsection as dict."""
    value = config.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"'{key}' must be a mapping/object when provided.")
    return dict(value)


def _coerce_bounds_pair(raw_value: Any, field_name: str) -> Bounds:
    """Coerce one axis bound pair ``(min, max)`` and validate ordering."""
    if (
        not isinstance(raw_value, Sequence)
        or isinstance(raw_value, (str, bytes))
        or len(raw_value) != 2
    ):
        raise ValueError(
            f"'{field_name}' must be a 2-item sequence [min, max], got {raw_value!r}."
        )

    low = float(raw_value[0])
    high = float(raw_value[1])
    if low >= high:
        raise ValueError(f"'{field_name}' must satisfy min < max, got {raw_value!r}.")

    return (low, high)


def _extract_xy_bounds(config: Mapping[str, Any]) -> Tuple[Bounds, Bounds]:
    """Resolve x/y bounds from supported schema variants."""
    sections: List[Tuple[str, Mapping[str, Any]]] = [("config", config)]
    for key in ("bounds", "room_dimensions", "position_bounds"):
        value = config.get(key)
        if isinstance(value, Mapping):
            sections.append((key, value))

    for section_name, section in sections:
        has_pair_form = "x_bounds" in section or "y_bounds" in section
        if has_pair_form:
            if "x_bounds" not in section or "y_bounds" not in section:
                raise ValueError(
                    f"Both x_bounds and y_bounds are required in '{section_name}'."
                )
            x_bounds = _coerce_bounds_pair(
                section["x_bounds"],
                f"{section_name}.x_bounds",
            )
            y_bounds = _coerce_bounds_pair(
                section["y_bounds"],
                f"{section_name}.y_bounds",
            )
            return x_bounds, y_bounds

        min_max_keys = ("x_min", "x_max", "y_min", "y_max")
        if all(key in section for key in min_max_keys):
            x_bounds = _coerce_bounds_pair(
                [section["x_min"], section["x_max"]],
                f"{section_name}.[x_min,x_max]",
            )
            y_bounds = _coerce_bounds_pair(
                [section["y_min"], section["y_max"]],
                f"{section_name}.[y_min,y_max]",
            )
            return x_bounds, y_bounds

    raise ValueError(
        "Could not resolve x/y bounds from config. Please add either: "
        "(1) x_bounds and y_bounds, or (2) bounds.x_bounds/bounds.y_bounds, "
        "or (3) room_dimensions.x_bounds/room_dimensions.y_bounds, "
        "or (4) position_bounds with x_min/x_max/y_min/y_max."
    )


def _extract_floorplan_coords(
    config: Mapping[str, Any],
    x_bounds: Bounds,
    y_bounds: Bounds,
) -> np.ndarray:
    """Resolve floorplan points for K-Means; fallback to synthetic grid."""
    raw_coords = config.get("floorplan_coords")
    if raw_coords is not None:
        coords = np.asarray(raw_coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                "'floorplan_coords' must have shape [N, 2] when provided. "
                f"Got {coords.shape}."
            )
        if coords.shape[0] < 1:
            raise ValueError("'floorplan_coords' must contain at least one point.")
        return coords

    kmeans_params = _get_optional_mapping(config, "kmeans_params")
    grid_size = int(kmeans_params.get("grid_size", 40))
    if grid_size < 2:
        raise ValueError("kmeans_params.grid_size must be >= 2 when provided.")

    x_values = np.linspace(x_bounds[0], x_bounds[1], grid_size, dtype=np.float64)
    y_values = np.linspace(y_bounds[0], y_bounds[1], grid_size, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
    return np.column_stack((grid_x.ravel(), grid_y.ravel()))


def _resolve_objective_params(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve memetic objective hyperparameters from config with defaults."""
    params = dict(_OBJECTIVE_PARAM_DEFAULTS)
    for key in (
        "objective_params",
        "ga_evaluation_params",
        "gd_params",
        "gd_optimization_params",
        "gd_hyperparams",
    ):
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"'{key}' must be a mapping/object when provided.")

        source = dict(value)
        if "softmin_temperature" not in source and "temperature" in source:
            source["softmin_temperature"] = source["temperature"]

        for objective_key in _OBJECTIVE_PARAM_DEFAULTS:
            if objective_key in source:
                params[objective_key] = source[objective_key]

    return params


def _resolve_gd_params(
    config: Mapping[str, Any],
    objective_params: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve GD params using modern and legacy config keys."""
    params = {
        "num_iterations": 50,
        "learning_rate": 0.1,
        "samples_per_tx": 1_000_000,
        "max_depth": 13,
        "verbose": False,
    }
    for key in ("gd_params", "gd_optimization_params", "gd_hyperparams"):
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"'{key}' must be a mapping/object when provided.")
        source = dict(value)
        if "softmin_temperature" not in source and "temperature" in source:
            source["softmin_temperature"] = source["temperature"]
        params.update(source)

    params.update(dict(objective_params))
    return params


def _resolve_pso_params(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve PSO hyperparameters with safe defaults."""
    params = {
        "swarm_size": 24,
        "num_iterations": 20,
        "w": 0.6,
        "c1": 1.5,
        "c2": 1.5,
    }
    pso_params = config.get("pso_params")
    if pso_params is not None:
        if not isinstance(pso_params, Mapping):
            raise ValueError("'pso_params' must be a mapping/object when provided.")
        params.update(dict(pso_params))
    return params


def _resolve_num_workers(config: Mapping[str, Any]) -> int:
    """Resolve pool worker count from config."""
    return int(config.get("num_pool_workers", config.get("num_workers", 1)))


def _resolve_ga_evaluation_params(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve static evaluator runtime params from shared schema."""
    params = dict(_GA_EVALUATION_PARAM_DEFAULTS)
    for key in ("ga_evaluation_params", "ga_optimization_params"):
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"'{key}' must be a mapping/object when provided.")
        params.update(dict(value))
    return params


def _resolve_demand_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve worker demand config with memetic-pipeline-compatible defaults."""
    demand_config = dict(_DEMAND_CONFIG_DEFAULTS)
    raw_demand = config.get("demand_config")
    if raw_demand is not None:
        if not isinstance(raw_demand, Mapping):
            raise ValueError("'demand_config' must be a mapping/object when provided.")
        demand_config.update(dict(raw_demand))

    explicit_priority_stats = False
    boxes = demand_config.get("bounding_boxes")
    weights = demand_config.get("box_weights")
    if isinstance(boxes, list) and isinstance(weights, list) and boxes and weights:
        explicit_priority_stats = True

    if isinstance(config.get("position_bounds"), Mapping):
        demand_config["position_bounds"] = dict(config["position_bounds"])

    demand_config["_report_priority_stats"] = bool(explicit_priority_stats)
    demand_config["_report_weighted_stats"] = bool(explicit_priority_stats)
    return demand_config


def _resolve_demand_config_for_method(
    config: Mapping[str, Any],
    method: str,
) -> Dict[str, Any]:
    """Resolve demand behavior per method.

    Regular geometric K-Means intentionally runs without demand weighting,
    while all other methods reuse memetic-style demand weighting.
    """
    if method != "kmeans":
        return _resolve_demand_config(config)

    demand_config = dict(_DEMAND_CONFIG_DEFAULTS)
    if isinstance(config.get("position_bounds"), Mapping):
        demand_config["position_bounds"] = dict(config["position_bounds"])
    demand_config["_report_priority_stats"] = False
    demand_config["_report_weighted_stats"] = False
    return demand_config


def _infer_floorplan_grid_shape(coords: np.ndarray) -> Tuple[int, int]:
    """Infer ``(rows, cols)`` from floorplan coordinates with safe fallback."""
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"floorplan_coords must have shape [N,2], got {coords.shape}")

    total_points = int(coords.shape[0])
    if total_points < 1:
        raise ValueError("floorplan_coords must contain at least one point")

    unique_x = np.unique(coords[:, 0])
    unique_y = np.unique(coords[:, 1])
    num_cols = int(unique_x.shape[0])
    num_rows = int(unique_y.shape[0])
    if num_rows >= 1 and num_cols >= 1 and num_rows * num_cols == total_points:
        return num_rows, num_cols

    side = int(np.ceil(np.sqrt(float(total_points))))
    return max(1, side), max(1, side)


def _build_weighted_kmeans_sample_weights(
    floorplan_coords: np.ndarray,
    x_bounds: Bounds,
    y_bounds: Bounds,
    demand_config: Mapping[str, Any],
) -> np.ndarray:
    """Build per-point sample weights using memetic demand-priority map logic."""
    coords = np.asarray(floorplan_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"floorplan_coords must have shape [N,2], got {coords.shape}")

    total_points = int(coords.shape[0])
    if total_points < 1:
        raise ValueError("floorplan_coords must contain at least one point")

    if not bool(demand_config.get("enabled", False)):
        return np.ones((total_points,), dtype=np.float64)

    num_rows, num_cols = _infer_floorplan_grid_shape(coords)

    map_demand_config = dict(demand_config)
    if "position_bounds" not in map_demand_config:
        map_demand_config["position_bounds"] = {
            "x_min": float(x_bounds[0]),
            "x_max": float(x_bounds[1]),
            "y_min": float(y_bounds[0]),
            "y_max": float(y_bounds[1]),
        }

    priority_map_tensor = generate_spatial_priority_map(
        num_rows=num_rows,
        num_cols=num_cols,
        demand_config=map_demand_config,
    )

    if hasattr(priority_map_tensor, "detach"):
        priority_map = np.asarray(priority_map_tensor.detach().cpu().numpy(), dtype=np.float64)
    else:
        priority_map = np.asarray(priority_map_tensor, dtype=np.float64)

    if priority_map.shape != (num_rows, num_cols):
        priority_map = np.reshape(priority_map, (num_rows, num_cols))

    x_span = float(x_bounds[1] - x_bounds[0])
    y_span = float(y_bounds[1] - y_bounds[0])
    if x_span <= 0.0 or y_span <= 0.0:
        raise ValueError("x_bounds and y_bounds must satisfy min < max")

    x_norm = (coords[:, 0] - float(x_bounds[0])) / x_span
    y_norm = (coords[:, 1] - float(y_bounds[0])) / y_span

    col_idx = np.rint(x_norm * float(max(1, num_cols - 1))).astype(np.int64)
    row_idx = np.rint(y_norm * float(max(1, num_rows - 1))).astype(np.int64)
    col_idx = np.clip(col_idx, 0, num_cols - 1)
    row_idx = np.clip(row_idx, 0, num_rows - 1)

    sample_weights = priority_map[row_idx, col_idx].astype(np.float64)
    if sample_weights.shape[0] != total_points:
        raise RuntimeError("weighted_kmeans sample-weight generation returned invalid shape")

    return sample_weights


def _bind_shared_actor_pool(
    ray_parallel_optimizer: RawRayParallelOptimizer,
    executor: RawRayActorPoolExecutor,
) -> None:
    """Bind one existing raw ActorPool executor into RawRayParallelOptimizer."""
    ray_parallel_optimizer._workers = executor._workers  # type: ignore[attr-defined]
    ray_parallel_optimizer._pool = executor._pool  # type: ignore[attr-defined]
    ray_parallel_optimizer._scene_config = dict(executor.scene_config)  # type: ignore[attr-defined]
    ray_parallel_optimizer._demand_config = dict(executor.demand_config)  # type: ignore[attr-defined]


def _warmup_actor_pool(
    executor: RawRayActorPoolExecutor,
    scene_config: Mapping[str, Any],
    num_aps: int,
    fixed_z: float,
    optimize_orientation: bool,
    objective_params: Mapping[str, Any],
    warmup_eval_params: Mapping[str, Any],
    x_bounds: Bounds,
    y_bounds: Bounds,
) -> None:
    """Run one tiny eval task so workers are initialized before GD dispatch."""
    tx_positions = scene_config.get("tx_positions", [])
    positions_xy: List[Tuple[float, float]] = []
    if isinstance(tx_positions, Sequence) and not isinstance(tx_positions, (str, bytes)):
        for tx in tx_positions:
            if (
                isinstance(tx, Sequence)
                and not isinstance(tx, (str, bytes))
                and len(tx) >= 2
            ):
                positions_xy.append((float(tx[0]), float(tx[1])))
            if len(positions_xy) >= int(num_aps):
                break

    while len(positions_xy) < int(num_aps):
        ratio = (len(positions_xy) + 1) / float(int(num_aps) + 1)
        x_val = x_bounds[0] + ratio * (x_bounds[1] - x_bounds[0])
        y_val = y_bounds[0] + ratio * (y_bounds[1] - y_bounds[0])
        positions_xy.append((float(x_val), float(y_val)))

    directions = (
        [(0.0, 0.0, -1.0) for _ in range(int(num_aps))]
        if bool(optimize_orientation)
        else None
    )

    warmup_task = build_evaluator_task(
        positions_xy=positions_xy,
        directions_xyz=directions,
        fixed_z=float(fixed_z),
        num_aps=int(num_aps),
        optimize_orientation=bool(optimize_orientation),
        loss_kwargs=objective_params,
        evaluation_kwargs=warmup_eval_params,
    )
    executor.map(lambda item: item, [(0, "memetic_eval", warmup_task, {})])


def _enforce_baseline_cuda(
    scene_config: Dict[str, Any],
    num_workers: int,
    configured_gpu_fraction: float,
) -> float:
    """Force baseline methods to CUDA/GPU execution."""
    scene_config["device"] = "cuda"

    default_fraction = 1.0 / float(max(1, int(num_workers)))
    gpu_fraction = float(configured_gpu_fraction)
    if gpu_fraction <= 0.0:
        gpu_fraction = default_fraction

    return float(gpu_fraction)


def _run_baseline_method(method: str, config: Mapping[str, Any]) -> Dict[str, Any]:
    """Run one baseline method with persistent worker lifecycle."""
    scene_config = _require_mapping(config.get("scene_config"), "scene_config")
    demand_config = _resolve_demand_config_for_method(config, method=method)

    num_workers = _resolve_num_workers(config)
    configured_gpu_fraction = float(config.get("gpu_fraction", 0.0))
    gpu_fraction = _enforce_baseline_cuda(
        scene_config=scene_config,
        num_workers=num_workers,
        configured_gpu_fraction=configured_gpu_fraction,
    )

    num_aps = int(config.get("num_aps", 2))
    fixed_z = float(config.get("fixed_z", 3.8))
    optimize_orientation = bool(config.get("optimize_orientation", True))

    x_bounds, y_bounds = _extract_xy_bounds(config)

    scene_config["num_aps"] = num_aps
    scene_config["position_bounds"] = {
        "x_min": float(x_bounds[0]),
        "x_max": float(x_bounds[1]),
        "y_min": float(y_bounds[0]),
        "y_max": float(y_bounds[1]),
    }

    random_params = _get_optional_mapping(config, "random_params")
    random_num_samples = int(random_params.get("num_samples", config.get("num_samples", 100)))

    random_gd_params = _get_optional_mapping(config, "random_gd_params")
    random_gd_num_samples = int(random_gd_params.get("num_samples", 10))

    objective_params = _resolve_objective_params(config)
    ga_evaluation_params = _resolve_ga_evaluation_params(config)
    gd_params = _resolve_gd_params(config, objective_params=objective_params)
    pso_params = _resolve_pso_params(config)

    warmup_eval_params = {
        "samples_per_tx": int(min(int(ga_evaluation_params.get("samples_per_tx", 10_000)), 10_000)),
        "max_depth": int(min(int(ga_evaluation_params.get("max_depth", 3)), 3)),
    }

    pool_executor: Optional[RawRayActorPoolExecutor] = None
    ray_optimizer: Optional[RawRayParallelOptimizer] = None
    shared_pool_bound = False

    try:
        pool_executor = RawRayActorPoolExecutor(
            scene_config=scene_config,
            demand_config=demand_config,
            num_workers=num_workers,
            gpu_fraction=gpu_fraction,
            verbose=bool(config.get("verbose", True)),
        )

        if method == "random":
            return run_random_monte_carlo(
                ray_pool=pool_executor,
                num_aps=num_aps,
                fixed_z=fixed_z,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                num_samples=random_num_samples,
                optimize_orientation=optimize_orientation,
                loss_kwargs=objective_params,
                evaluation_params=ga_evaluation_params,
            )

        if method == "kmeans":
            floorplan_coords = _extract_floorplan_coords(
                config=config,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
            )
            return run_kmeans_baseline(
                ray_pool=pool_executor,
                num_aps=num_aps,
                fixed_z=fixed_z,
                floorplan_coords=floorplan_coords,
                optimize_orientation=optimize_orientation,
                loss_kwargs=objective_params,
                evaluation_params=ga_evaluation_params,
            )

        if method == "weighted_kmeans":
            floorplan_coords = _extract_floorplan_coords(
                config=config,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
            )
            spatial_weights = _build_weighted_kmeans_sample_weights(
                floorplan_coords=floorplan_coords,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                demand_config=demand_config,
            )
            return run_weighted_kmeans_baseline(
                ray_pool=pool_executor,
                num_aps=num_aps,
                fixed_z=fixed_z,
                floorplan_coords=floorplan_coords,
                spatial_weights=spatial_weights,
                optimize_orientation=optimize_orientation,
                loss_kwargs=objective_params,
                evaluation_params=ga_evaluation_params,
            )

        ray_optimizer = RawRayParallelOptimizer(
            num_workers=num_workers,
            gpu_fraction=gpu_fraction,
            demand_config=demand_config,
        )
        _bind_shared_actor_pool(ray_optimizer, pool_executor)
        shared_pool_bound = True

        ray_optimizer._scene_config = dict(scene_config)  # type: ignore[attr-defined]

        _warmup_actor_pool(
            executor=pool_executor,
            scene_config=scene_config,
            num_aps=num_aps,
            fixed_z=fixed_z,
            optimize_orientation=optimize_orientation,
            objective_params=objective_params,
            warmup_eval_params=warmup_eval_params,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )

        if method == "random_gd":
            return run_random_multi_start_gd(
                ray_optimizer=ray_optimizer,
                num_aps=num_aps,
                fixed_z=fixed_z,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                gd_params=gd_params,
                num_samples=random_gd_num_samples,
                optimize_orientation=optimize_orientation,
            )

        if method == "pso_gd":
            return run_pso_gd_baseline(
                ray_optimizer=ray_optimizer,
                num_aps=num_aps,
                fixed_z=fixed_z,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                pso_params=pso_params,
                gd_params=gd_params,
                optimize_orientation=optimize_orientation,
                loss_kwargs=objective_params,
                evaluation_params=ga_evaluation_params,
            )

        raise ValueError(f"Unsupported baseline method: {method!r}")

    finally:
        if pool_executor is not None:
            try:
                pool_executor.shutdown()
            except Exception:
                pass

        if ray_optimizer is not None and not shared_pool_bound:
            try:
                ray_optimizer.shutdown()
            except Exception:
                pass

        if ray.is_initialized():
            ray.shutdown()


def _run_single_method(
    method: str,
    config: Dict[str, Any],
    run_dir: Path,
    run_name: Optional[str],
) -> Dict[str, Any]:
    """Dispatch one method and return result payload."""
    if method == "memetic":
        memetic_config = dict(config)
        memetic_output_dir = run_dir / "memetic"
        memetic_output_dir.mkdir(parents=True, exist_ok=True)
        memetic_config["output_dir"] = str(memetic_output_dir)
        memetic_config["run_name"] = (
            f"{run_name}_memetic"
            if run_name
            else "memetic"
        )
        return run_memetic_optimization(memetic_config)

    return _run_baseline_method(method=method, config=config)


def _resolve_method_sequence(method: str) -> List[str]:
    """Resolve execution sequence from requested method selector."""
    if method == "all_baselines":
        return list(_BASELINE_METHODS)
    if method == "all":
        return list(_SINGLE_METHODS)
    return [method]


def _resolve_run_directory(output_dir: Path, run_name: Optional[str]) -> Path:
    """Resolve run root directory for all artifacts and plots."""
    if run_name:
        run_dir = output_dir / run_name
    else:
        run_dir = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _to_jsonable(value: Any) -> Any:
    """Recursively coerce arbitrary objects into JSON-serializable values."""
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

    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
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
        json.dump(_to_jsonable(payload), handle, indent=2)


def _flatten_row(row: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested mappings into one CSV-friendly row."""
    flattened: Dict[str, Any] = {}
    for key, value in row.items():
        joined_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_row(value, prefix=joined_key))
        elif isinstance(value, list):
            flattened[joined_key] = json.dumps(_to_jsonable(value), ensure_ascii=True)
        else:
            flattened[joined_key] = _to_jsonable(value)
    return flattened


def _collect_fieldnames(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Collect stable CSV fieldnames from flattened rows."""
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows to CSV; nested mappings are flattened."""
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    flat_rows = [_flatten_row(row) for row in rows]
    fieldnames = _collect_fieldnames(flat_rows)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def _aggregate_gd_iteration_trace(gd_results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Build aggregate GD trend from all fine-tuned worker histories."""
    all_results = gd_results.get("all_fine_tuned_results", [])
    if not isinstance(all_results, list):
        return []

    all_series: List[List[float]] = []
    for raw_result in all_results:
        if not isinstance(raw_result, Mapping):
            continue
        history = raw_result.get("history", {})
        if not isinstance(history, Mapping):
            continue
        primary_series = history.get("primary_loss", [])
        if not isinstance(primary_series, Sequence):
            continue

        series: List[float] = []
        for raw_value in primary_series:
            try:
                series.append(float(raw_value))
            except (TypeError, ValueError):
                continue
        if series:
            all_series.append(series)

    if not all_series:
        return []

    trace: List[Dict[str, Any]] = []
    running_best = float("inf")
    max_len = max(len(series) for series in all_series)
    for iteration_idx in range(max_len):
        bucket = [series[iteration_idx] for series in all_series if iteration_idx < len(series)]
        if not bucket:
            continue

        min_loss = float(min(bucket))
        mean_loss = float(np.mean(bucket))
        max_loss = float(max(bucket))
        running_best = min(running_best, min_loss)
        trace.append(
            {
                "iteration": int(iteration_idx + 1),
                "min_primary_loss": min_loss,
                "mean_primary_loss": mean_loss,
                "max_primary_loss": max_loss,
                "running_best_primary_loss": float(running_best),
            }
        )

    return trace


def _extract_method_iteration_trace(
    method: str,
    result_payload: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Extract normalized per-iteration trace rows for one method result."""
    raw_trace = result_payload.get("iteration_trace")
    if isinstance(raw_trace, list):
        trace_rows: List[Dict[str, Any]] = []
        for index, row in enumerate(raw_trace, start=1):
            if not isinstance(row, Mapping):
                continue
            normalized = dict(row)
            normalized.setdefault("iteration", int(index))
            trace_rows.append({str(k): _to_jsonable(v) for k, v in normalized.items()})
        if trace_rows:
            return trace_rows

    if method == "memetic":
        gd_results = result_payload.get("gd_results", {})
        if isinstance(gd_results, Mapping):
            return _aggregate_gd_iteration_trace(gd_results)

    return []


def _extract_best_primary_loss(method: str, result_payload: Mapping[str, Any]) -> Optional[float]:
    """Extract one best primary loss value for method-level summary tables."""
    if method == "memetic":
        gd_results = result_payload.get("gd_results", {})
        if isinstance(gd_results, Mapping):
            metrics = gd_results.get("metrics", {})
            if isinstance(metrics, Mapping):
                raw = metrics.get("best_primary_loss")
                if raw is not None:
                    try:
                        return float(raw)
                    except (TypeError, ValueError):
                        pass

    raw = result_payload.get("best_primary_loss")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    return None


def _as_float(value: Any) -> Optional[float]:
    """Convert a value to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _save_method_trend_plot(method: str, trace_rows: Sequence[Mapping[str, Any]], plot_path: Path) -> Optional[str]:
    """Save one interactive method trend plot when plotly is available."""
    if go is None or not trace_rows:
        return None

    x_values = list(range(1, len(trace_rows) + 1))
    phase_labels = [str(row.get("phase", "")) for row in trace_rows]

    keys_in_order = [
        "running_best_primary_loss",
        "global_best_primary_loss",
        "min_primary_loss",
        "mean_primary_loss",
        "max_primary_loss",
        "primary_loss",
        "swarm_best_primary_loss",
        "swarm_mean_primary_loss",
    ]

    figure = go.Figure()
    for key in keys_in_order:
        y_values: List[Optional[float]] = [_as_float(row.get(key)) for row in trace_rows]
        if not any(value is not None for value in y_values):
            continue

        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=key,
                customdata=phase_labels,
                hovertemplate=(
                    "iter=%{x}<br>value=%{y:.6f}<br>phase=%{customdata}<extra>"
                    + key
                    + "</extra>"
                ),
            )
        )

    if not figure.data:
        return None

    figure.update_layout(
        title=f"{method} Trend",
        xaxis_title="Iteration",
        yaxis_title="Primary Loss",
        template="plotly_white",
    )
    figure.update_xaxes(tickmode="linear", tick0=1, dtick=1, tickformat="d")

    figure.write_html(str(plot_path), include_plotlyjs="cdn")
    return str(plot_path)


def _build_running_best_series(trace_rows: Sequence[Mapping[str, Any]]) -> List[float]:
    """Build a comparable running-best loss curve from arbitrary trace rows."""
    preferred_keys = (
        "running_best_primary_loss",
        "global_best_primary_loss",
        "min_primary_loss",
        "primary_loss",
        "swarm_best_primary_loss",
    )

    values: List[float] = []
    running_best = float("inf")
    for row in trace_rows:
        candidate: Optional[float] = None
        for key in preferred_keys:
            if key in row:
                candidate = _as_float(row.get(key))
                if candidate is not None:
                    break

        if candidate is None:
            continue

        running_best = min(running_best, float(candidate))
        values.append(float(running_best))

    return values


def _save_comparison_plot(
    method_traces: Mapping[str, Sequence[Mapping[str, Any]]],
    plot_path: Path,
) -> Optional[str]:
    """Save multi-method running-best comparison plot."""
    if go is None or not method_traces:
        return None

    figure = go.Figure()
    for method, trace_rows in method_traces.items():
        running_best = _build_running_best_series(trace_rows)
        if not running_best:
            continue

        x_values = list(range(1, len(running_best) + 1))
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=running_best,
                mode="lines+markers",
                name=method,
            )
        )

    if not figure.data:
        return None

    figure.update_layout(
        title="Method Running-Best Trend Comparison",
        xaxis_title="Iteration",
        yaxis_title="Running Best Primary Loss",
        template="plotly_white",
    )
    figure.update_xaxes(tickmode="linear", tick0=1, dtick=1, tickformat="d")
    figure.write_html(str(plot_path), include_plotlyjs="cdn")
    return str(plot_path)


def _extract_best_physical_metrics(
    method: str,
    result_payload: Mapping[str, Any],
) -> Dict[str, float]:
    """Extract normalized best physical metrics from method result payloads."""
    candidates: List[Mapping[str, Any]] = []

    top_level_metrics = result_payload.get("best_physical_metrics")
    if isinstance(top_level_metrics, Mapping):
        candidates.append(top_level_metrics)

    if method == "memetic":
        global_best = result_payload.get("global_best_result")
        if isinstance(global_best, Mapping):
            result_summary = global_best.get("results")
            if isinstance(result_summary, Mapping):
                for key in ("best_physical_metrics", "physical_metrics"):
                    value = result_summary.get(key)
                    if isinstance(value, Mapping):
                        candidates.append(value)

            for key in ("best_physical_metrics", "physical_metrics"):
                value = global_best.get(key)
                if isinstance(value, Mapping):
                    candidates.append(value)

    for raw_metrics in candidates:
        normalized = {
            str(name): float(metric)
            for name, raw_value in raw_metrics.items()
            if (metric := _as_float(raw_value)) is not None
        }
        if normalized:
            return normalized

    return {}


def _format_optional_float(value: Optional[float], precision: int = 6) -> str:
    """Format a float for table output; returns ``N/A`` for invalid values."""
    if value is None:
        return "N/A"

    numeric = _as_float(value)
    if numeric is None or not np.isfinite(numeric):
        return "N/A"

    return f"{numeric:.{precision}f}"


def _render_text_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a compact ASCII table suitable for terminal and text files."""
    str_headers = [str(header) for header in headers]
    str_rows = [[str(cell) for cell in row] for row in rows]

    widths = [len(header) for header in str_headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            if idx < len(widths):
                widths[idx] = max(widths[idx], len(cell))

    def _render_row(row_values: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row_values))

    divider = "-+-".join("-" * width for width in widths)
    lines = [_render_row(str_headers), divider]
    lines.extend(_render_row(row) for row in str_rows)
    return "\n".join(lines)


def _build_final_analysis_report(
    run_dir: Path,
    methods: Sequence[str],
    method_summary_rows: Sequence[Mapping[str, Any]],
    method_trace_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    method_results: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build ranked final analysis report and normalized ranking rows."""
    summary_by_method = {
        str(row.get("method")): row
        for row in method_summary_rows
        if row.get("method") is not None
    }

    analysis_rows: List[Dict[str, Any]] = []
    for method in methods:
        summary_row = summary_by_method.get(method, {})
        trace_rows = method_trace_rows.get(method, [])
        running_best_series = _build_running_best_series(trace_rows)

        start_running_best = running_best_series[0] if running_best_series else None
        end_running_best = running_best_series[-1] if running_best_series else None
        improvement = (
            float(start_running_best - end_running_best)
            if (start_running_best is not None and end_running_best is not None)
            else None
        )

        best_physical_metrics = _extract_best_physical_metrics(
            method=method,
            result_payload=method_results.get(method, {}),
        )

        analysis_rows.append(
            {
                "method": method,
                "best_primary_loss": _as_float(summary_row.get("best_primary_loss")),
                "elapsed_sec": _as_float(summary_row.get("elapsed_sec")),
                "num_iterations": int(summary_row.get("num_iterations", len(trace_rows))),
                "start_running_best": _as_float(start_running_best),
                "end_running_best": _as_float(end_running_best),
                "running_best_improvement": _as_float(improvement),
                "coverage_pct": _as_float(best_physical_metrics.get("coverage_pct")),
                "min_rss_dbm": _as_float(best_physical_metrics.get("min_rss_dbm")),
                "p5_rss_dbm": _as_float(best_physical_metrics.get("p5_rss_dbm")),
                "mean_rss_dbm": _as_float(best_physical_metrics.get("mean_rss_dbm")),
            }
        )

    ranked_rows = sorted(
        analysis_rows,
        key=lambda row: (
            float("inf") if row["best_primary_loss"] is None else float(row["best_primary_loss"]),
            float("inf") if row["elapsed_sec"] is None else float(row["elapsed_sec"]),
        ),
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    ranking_table_rows = [
        [
            row["rank"],
            row["method"],
            _format_optional_float(row["best_primary_loss"], precision=9),
            _format_optional_float(row["elapsed_sec"], precision=2),
            row["num_iterations"],
            _format_optional_float(row["running_best_improvement"], precision=6),
        ]
        for row in ranked_rows
    ]
    ranking_table = _render_text_table(
        headers=[
            "Rank",
            "Method",
            "Best Primary Loss",
            "Elapsed (s)",
            "Iterations",
            "Running-Best Gain",
        ],
        rows=ranking_table_rows,
    )

    quality_table_rows = [
        [
            row["method"],
            _format_optional_float(row["coverage_pct"], precision=3),
            _format_optional_float(row["min_rss_dbm"], precision=3),
            _format_optional_float(row["p5_rss_dbm"], precision=3),
            _format_optional_float(row["mean_rss_dbm"], precision=3),
        ]
        for row in ranked_rows
    ]
    quality_table = _render_text_table(
        headers=[
            "Method",
            "Coverage (%)",
            "Min RSS (dBm)",
            "P5 RSS (dBm)",
            "Mean RSS (dBm)",
        ],
        rows=quality_table_rows,
    )

    lines = [
        "=" * 80,
        "FINAL MULTI-METHOD ANALYSIS",
        "=" * 80,
        f"Run directory: {run_dir}",
        "",
        "Ranking by best primary loss (lower is better):",
        ranking_table,
        "",
        "Best-configuration quality snapshot:",
        quality_table,
        "",
        "Note: Running-Best Gain = first running-best value - final running-best value.",
    ]

    if ranked_rows:
        winner = ranked_rows[0]
        lines.extend(
            [
                "",
                "Winner:",
                (
                    f"  method={winner['method']} | "
                    f"best_primary_loss={_format_optional_float(winner['best_primary_loss'], precision=9)} | "
                    f"elapsed_sec={_format_optional_float(winner['elapsed_sec'], precision=2)}"
                ),
            ]
        )

    lines.append("=" * 80)
    report_text = "\n".join(lines) + "\n"
    return report_text, ranked_rows


def _save_method_artifacts(
    method: str,
    result_payload: Mapping[str, Any],
    artifacts_dir: Path,
    plots_dir: Path,
) -> Dict[str, Optional[str]]:
    """Save one method's JSON result, iteration traces, CSVs, and plots."""
    method_artifacts: Dict[str, Optional[str]] = {}

    result_json_path = artifacts_dir / f"{method}_results.json"
    _write_json(result_json_path, result_payload)
    method_artifacts["result_json"] = str(result_json_path)

    trace_rows = _extract_method_iteration_trace(method, result_payload)
    if trace_rows:
        trace_json_path = artifacts_dir / f"{method}_iteration_trace.json"
        trace_csv_path = artifacts_dir / f"{method}_iteration_trace.csv"
        _write_json(trace_json_path, trace_rows)
        _write_csv(trace_csv_path, trace_rows)
        method_artifacts["trace_json"] = str(trace_json_path)
        method_artifacts["trace_csv"] = str(trace_csv_path)

        trend_plot_path = plots_dir / f"{method}_trend.html"
        method_artifacts["trend_plot_html"] = _save_method_trend_plot(
            method=method,
            trace_rows=trace_rows,
            plot_path=trend_plot_path,
        )
    else:
        method_artifacts["trace_json"] = None
        method_artifacts["trace_csv"] = None
        method_artifacts["trend_plot_html"] = None

    return method_artifacts


def main() -> None:
    """CLI entry point for running and saving experiment methods."""
    args = _parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_json_config(config_path)
    run_dir = _resolve_run_directory(output_dir=output_dir, run_name=args.run_name)

    artifacts_dir = run_dir / "artifacts"
    plots_dir = run_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    methods = _resolve_method_sequence(str(args.method))

    method_results: Dict[str, Dict[str, Any]] = {}
    method_artifacts: Dict[str, Dict[str, Optional[str]]] = {}
    method_trace_rows: Dict[str, List[Dict[str, Any]]] = {}
    method_summary_rows: List[Dict[str, Any]] = []

    for method in methods:
        method_start = time.perf_counter()
        result_payload = _run_single_method(
            method=method,
            config=dict(config),
            run_dir=run_dir,
            run_name=args.run_name,
        )
        method_elapsed = float(time.perf_counter() - method_start)

        if not isinstance(result_payload, Mapping):
            raise RuntimeError(f"Method {method!r} returned non-mapping payload.")

        payload = dict(result_payload)
        payload.setdefault("orchestrator", {})
        if isinstance(payload["orchestrator"], Mapping):
            orchestrator_metadata = dict(payload["orchestrator"])
            orchestrator_metadata["method"] = method
            orchestrator_metadata["config_path"] = str(config_path)
            orchestrator_metadata["elapsed_sec"] = method_elapsed
            orchestrator_metadata["run_dir"] = str(run_dir)
            payload["orchestrator"] = orchestrator_metadata

        method_results[method] = payload
        method_artifacts[method] = _save_method_artifacts(
            method=method,
            result_payload=payload,
            artifacts_dir=artifacts_dir,
            plots_dir=plots_dir,
        )

        trace_rows = _extract_method_iteration_trace(method, payload)
        method_trace_rows[method] = trace_rows

        method_summary_rows.append(
            {
                "method": method,
                "elapsed_sec": method_elapsed,
                "best_primary_loss": _extract_best_primary_loss(method, payload),
                "num_iterations": len(trace_rows),
                "result_json": method_artifacts[method].get("result_json"),
                "trace_csv": method_artifacts[method].get("trace_csv"),
                "trend_plot_html": method_artifacts[method].get("trend_plot_html"),
            }
        )

    final_analysis_text, ranked_analysis_rows = _build_final_analysis_report(
        run_dir=run_dir,
        methods=methods,
        method_summary_rows=method_summary_rows,
        method_trace_rows=method_trace_rows,
        method_results=method_results,
    )
    final_analysis_txt_path = artifacts_dir / "final_analysis.txt"
    final_analysis_txt_path.write_text(final_analysis_text, encoding="utf-8")

    summary_payload = {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "methods": methods,
        "method_summary": method_summary_rows,
        "method_artifacts": method_artifacts,
        "analysis": {
            "final_analysis_txt": str(final_analysis_txt_path),
            "ranked_methods": ranked_analysis_rows,
        },
    }

    summary_json_path = artifacts_dir / "experiment_summary.json"
    summary_csv_path = artifacts_dir / "method_summary.csv"
    _write_json(summary_json_path, summary_payload)
    _write_csv(summary_csv_path, method_summary_rows)

    comparison_plot_path: Optional[str] = None
    if len(methods) > 1:
        comparison_plot = plots_dir / "method_comparison_trend.html"
        comparison_plot_path = _save_comparison_plot(
            method_traces=method_trace_rows,
            plot_path=comparison_plot,
        )

    static_plot_artifacts = save_static_comparison_plots(
        methods=methods,
        method_results=method_results,
        method_trace_rows=method_trace_rows,
        plots_dir=plots_dir,
        rssi_y_limits=(-100.0, -40.0),
    )

    launcher_payload = {
        "run_dir": str(run_dir),
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
        "comparison_plot_html": comparison_plot_path,
        "final_analysis_txt": str(final_analysis_txt_path),
        "static_plot_artifacts": static_plot_artifacts,
    }
    _write_json(artifacts_dir / "launcher_summary.json", launcher_payload)

    if isinstance(summary_payload.get("analysis"), Mapping):
        summary_analysis = dict(summary_payload["analysis"])
        summary_analysis["static_plot_artifacts"] = static_plot_artifacts
        summary_payload["analysis"] = summary_analysis
        _write_json(summary_json_path, summary_payload)

    print(final_analysis_text.rstrip())
    print(f"Saved run artifacts: {run_dir}")
    print(f"Saved analysis report: {final_analysis_txt_path}")


if __name__ == "__main__":
    main()
