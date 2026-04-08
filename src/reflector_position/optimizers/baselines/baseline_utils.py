"""Universal data adapters for baseline optimization algorithms.

This module intentionally contains only data-structure translation utilities.
It decouples baseline candidate generators (random, k-means, PSO, etc.) from
the underlying Ray/Sionna execution and physics-evaluation layers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _to_xy_tuple(value: Tuple[float, float], index: int) -> Tuple[float, float]:
    """Return one AP XY coordinate as validated float tuple."""
    if len(value) != 2:
        raise ValueError(
            f"positions_xy[{index}] must contain exactly 2 values, got {len(value)}"
        )
    return (float(value[0]), float(value[1]))


def _to_xyz_tuple(value: Tuple[float, float, float], index: int, label: str) -> Tuple[float, float, float]:
    """Return one 3-D vector as validated float tuple."""
    if len(value) != 3:
        raise ValueError(
            f"{label}[{index}] must contain exactly 3 values, got {len(value)}"
        )
    return (float(value[0]), float(value[1]), float(value[2]))


def _build_best_individual(
    positions_xy: Sequence[Tuple[float, float]],
    directions_xyz: Optional[Sequence[Tuple[float, float, float]]],
) -> List[float]:
    """Flatten best XY positions and optional XYZ directions into one vector."""
    genes = []
    for x_val, y_val in positions_xy:
        genes.extend([float(x_val), float(y_val)])
    if directions_xyz is not None:
        for dx_val, dy_val, dz_val in directions_xyz:
            genes.extend([float(dx_val), float(dy_val), float(dz_val)])
    return genes


def build_evaluator_task(
    positions_xy: Sequence[Tuple[float, float]],
    directions_xyz: Optional[Sequence[Tuple[float, float, float]]],
    fixed_z: float,
    num_aps: int,
    optimize_orientation: bool,
    loss_kwargs: Optional[Mapping[str, Any]] = None,
    evaluation_kwargs: Optional[Mapping[str, Any]] = None,
    reflector_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the canonical evaluator task payload for Ray-based physics evaluation.

    This universal adapter converts baseline-generated candidate variables into
    the normalized dictionary contract consumed by the existing evaluator stack.
    It exists to decouple baseline search logic from the Ray/Sionna engine.

    Args:
        positions_xy: Candidate AP positions as ``(x, y)`` pairs.
        directions_xyz: Optional AP look directions as ``(dx, dy, dz)`` tuples.
        fixed_z: Fixed AP z-coordinate applied by the evaluator.
        num_aps: Expected number of APs in the task payload.
        optimize_orientation: Whether orientation is optimized for this run.
        loss_kwargs: Optional memetic objective kwargs forwarded to
            ``StaticConfigurationEvaluator``.
        evaluation_kwargs: Optional static-evaluator runtime kwargs such as
            ``samples_per_tx`` and ``max_depth``.
        reflector_params: Optional reflector payload containing
            ``reflector_u``, ``reflector_v``, and ``reflector_target``.

    Returns:
        A flat evaluator task dictionary with GA/GD-compatible keys:
        ``initial_positions``, ``fixed_z``, ``num_aps``,
        ``optimize_orientation``, ``initial_directions_xyz`` and optional
        reflector keys.

    Raises:
        ValueError: If AP counts are inconsistent or reflector inputs are
            incomplete/invalid.
    """
    expected_num_aps = int(num_aps)
    normalized_positions = [
        _to_xy_tuple(position, index)
        for index, position in enumerate(positions_xy)
    ]
    if len(normalized_positions) != expected_num_aps:
        raise ValueError(
            "positions_xy length must match num_aps "
            f"({len(normalized_positions)} != {expected_num_aps})"
        )

    normalized_directions: Optional[Sequence[Tuple[float, float, float]]]
    if bool(optimize_orientation):
        if directions_xyz is not None:
            normalized = [
                _to_xyz_tuple(direction, index, "directions_xyz")
                for index, direction in enumerate(directions_xyz)
            ]
            if len(normalized) != expected_num_aps:
                raise ValueError(
                    "directions_xyz length must match num_aps when "
                    f"optimize_orientation=True ({len(normalized)} != {expected_num_aps})"
                )
            normalized_directions = normalized
        else:
            normalized_directions = None
    else:
        normalized_directions = None

    task: Dict[str, Any] = {
        "initial_positions": normalized_positions,
        "fixed_z": float(fixed_z),
        "num_aps": expected_num_aps,
        "optimize_orientation": bool(optimize_orientation),
        "initial_directions_xyz": normalized_directions,
    }

    if loss_kwargs is not None:
        if not isinstance(loss_kwargs, Mapping):
            raise ValueError("loss_kwargs must be a mapping when provided")
        task["loss_kwargs"] = dict(loss_kwargs)

    if evaluation_kwargs is not None:
        if not isinstance(evaluation_kwargs, Mapping):
            raise ValueError("evaluation_kwargs must be a mapping when provided")
        if "samples_per_tx" in evaluation_kwargs:
            task["samples_per_tx"] = int(evaluation_kwargs["samples_per_tx"])
        if "max_depth" in evaluation_kwargs:
            task["max_depth"] = int(evaluation_kwargs["max_depth"])

    if reflector_params is not None:
        reflector_view = dict(reflector_params)
        missing_keys = [
            key
            for key in ("reflector_u", "reflector_v", "reflector_target")
            if key not in reflector_view
        ]
        if missing_keys:
            raise ValueError(
                "reflector_params is missing required keys: "
                f"{', '.join(missing_keys)}"
            )

        reflector_target_raw = reflector_view["reflector_target"]
        if not isinstance(reflector_target_raw, (list, tuple)):
            raise ValueError("reflector_target must be a tuple/list of (x, y, z)")
        if len(reflector_target_raw) != 3:
            raise ValueError(
                "reflector_target must contain exactly 3 values, "
                f"got {len(reflector_target_raw)}"
            )

        reflector_target = (
            float(reflector_target_raw[0]),
            float(reflector_target_raw[1]),
            float(reflector_target_raw[2]),
        )

        task.update(
            {
                "reflector_u": float(reflector_view["reflector_u"]),
                "reflector_v": float(reflector_view["reflector_v"]),
                "reflector_target": reflector_target,
            }
        )

    return task


def format_baseline_result(
    algorithm_name: str,
    best_positions: Sequence[Tuple[float, float]],
    best_directions: Optional[Sequence[Tuple[float, float, float]]],
    best_primary_loss: float,
    loss_components: Dict[str, float],
    physical_metrics: Dict[str, float],
    time_elapsed: float,
) -> Dict[str, Any]:
    """Format baseline outputs into a GA-compatible JSON result contract.

    This universal adapter standardizes baseline algorithm results into the
    key naming convention used by GA artifacts so existing reporting/plotting
    code can consume outputs without engine-specific branching.

    Args:
        algorithm_name: Baseline identifier such as ``random_monte_carlo``.
        best_positions: Best AP positions as ``(x, y)`` tuples.
        best_directions: Optional best AP directions as ``(dx, dy, dz)`` tuples.
        best_primary_loss: Best objective value in loss space.
        loss_components: Best detached loss components.
        physical_metrics: Best detached physical metrics.
        time_elapsed: Wall-clock runtime in seconds.

    Returns:
        A standardized dictionary using GA-style keys including
        ``best_primary_fitness``, ``best_primary_loss``,
        ``best_loss_components``, and ``best_physical_metrics``.
    """
    normalized_positions = [
        _to_xy_tuple(position, index)
        for index, position in enumerate(best_positions)
    ]
    normalized_directions = (
        [
            _to_xyz_tuple(direction, index, "best_directions")
            for index, direction in enumerate(best_directions)
        ]
        if best_directions is not None
        else None
    )
    if normalized_directions is not None and len(normalized_directions) != len(normalized_positions):
        raise ValueError(
            "best_directions length must match best_positions length "
            f"({len(normalized_directions)} != {len(normalized_positions)})"
        )

    best_loss = float(best_primary_loss)
    normalized_loss_components = {
        str(name): float(value)
        for name, value in dict(loss_components).items()
    }
    normalized_physical_metrics = {
        str(name): float(value)
        for name, value in dict(physical_metrics).items()
    }
    best_individual = _build_best_individual(normalized_positions, normalized_directions)

    seed_payload: Dict[str, Any] = {
        "rank": 1,
        "primary_fitness": float(-best_loss),
        "ap_positions": [tuple(position) for position in normalized_positions],
        "ap_directions": [tuple(direction) for direction in normalized_directions]
        if normalized_directions is not None
        else None,
        "reflector": None,
        "chromosome": list(best_individual),
        "loss_components": dict(normalized_loss_components),
        "physical_metrics": dict(normalized_physical_metrics),
        "min_distance_to_previous": None,
    }

    hall_of_fame_payload: Dict[str, Any] = {
        "rank": 1,
        "primary_fitness": float(-best_loss),
        "loss_components": dict(normalized_loss_components),
        "physical_metrics": dict(normalized_physical_metrics),
        "ap_positions": [tuple(position) for position in normalized_positions],
        "ap_directions": [tuple(direction) for direction in normalized_directions]
        if normalized_directions is not None
        else None,
        "reflector": None,
        "chromosome": list(best_individual),
    }

    result: Dict[str, Any] = {
        "algorithm_name": str(algorithm_name),
        "num_aps": len(normalized_positions),
        "optimize_orientation": normalized_directions is not None,
        "reflector_enabled": False,
        "best_primary_fitness": float(-best_loss),
        "best_primary_loss": best_loss,
        "best_individual": list(best_individual),
        "best_positions": normalized_positions,
        "best_directions": normalized_directions,
        "best_loss_components": normalized_loss_components,
        "best_physical_metrics": normalized_physical_metrics,
        "seeds": [seed_payload],
        "num_selected_seeds": 1,
        "generation_top_k": 1,
        "seed_extraction": {
            "k_requested": 1,
            "d_corr": 0.0,
            "hof_size": 1,
        },
        "hall_of_fame": [hall_of_fame_payload],
        "logbook": [],
        "generation_details": [],
        "total_time": float(time_elapsed),
        "total_evaluations": 1,
        "ga_params": {},
        "time_elapsed": float(time_elapsed),
    }
    return result
