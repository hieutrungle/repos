"""Memetic bridge: translate GA seeds into Ray GD work items.

This module converts the Phase-1 memetic GA outputs (topological seed dicts)
into self-contained task dictionaries for distributed gradient-descent runs via
``RayParallelOptimizer.run(..., optimizer_method="gradient_descent")``.

The bridge is intentionally pure and side-effect free:
- It does not mutate input seed dictionaries.
- It validates required schema per seed.
- It emits deterministic, serializable task payloads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple


def _coerce_vec3(name: str, value: Any) -> Tuple[float, float, float]:
    """Validate and coerce a 3D vector-like value to ``(x, y, z)`` floats."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"'{name}' must be a length-3 tuple/list, got: {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _extract_positions(seed: Mapping[str, Any], seed_index: int) -> List[Tuple[float, float, float]]:
    """Extract AP positions from common seed schemas.

    Supported keys:
    - ``positions`` (requested Phase-2 schema)
    - ``ap_positions`` (Phase-1 memetic runner schema)
    """
    positions_raw = seed.get("positions", seed.get("ap_positions"))
    if positions_raw is None:
        raise ValueError(
            f"Seed #{seed_index} is missing required key 'positions' (or 'ap_positions')."
        )
    if not isinstance(positions_raw, (list, tuple)) or len(positions_raw) == 0:
        raise ValueError(f"Seed #{seed_index} has invalid 'positions': {positions_raw!r}")

    return [
        _coerce_vec3(f"seed[{seed_index}].positions[{i}]", p)
        for i, p in enumerate(positions_raw)
    ]


def _extract_directions(seed: Mapping[str, Any]) -> Optional[List[Tuple[float, float, float]]]:
    """Extract AP directions from common seed schemas.

    Supported keys:
    - ``directions`` (requested Phase-2 schema)
    - ``ap_directions`` (Phase-1 memetic runner schema)
    """
    directions_raw = seed.get("directions", seed.get("ap_directions"))
    if directions_raw is None:
        return None

    if not isinstance(directions_raw, (list, tuple)):
        raise ValueError("'directions' must be a list/tuple when provided.")

    return [
        _coerce_vec3(f"directions[{i}]", d)
        for i, d in enumerate(directions_raw)
    ]


def _extract_reflector(seed: Mapping[str, Any], seed_index: int) -> Tuple[float, float, Tuple[float, float, float]]:
    """Extract reflector parameters from supported seed schemas.

    Supported forms:
    1. Flat keys:
       - ``reflector_u``, ``reflector_v``, ``focal_point`` (or ``reflector_target``)
    2. Nested Phase-1 style:
       - ``reflector`` dict with ``u``, ``v``, ``focal_x``, ``focal_y``, ``focal_z``
    """
    if "reflector" in seed and isinstance(seed["reflector"], Mapping):
        reflector = seed["reflector"]
        missing = [k for k in ("u", "v", "focal_x", "focal_y", "focal_z") if k not in reflector]
        if missing:
            raise ValueError(
                f"Seed #{seed_index} reflector dict is missing required keys: {missing}."
            )
        u_val = float(reflector["u"])
        v_val = float(reflector["v"])
        focal = (
            float(reflector["focal_x"]),
            float(reflector["focal_y"]),
            float(reflector["focal_z"]),
        )
        return u_val, v_val, focal

    missing_flat = [
        key
        for key in ("reflector_u", "reflector_v")
        if key not in seed
    ]
    if missing_flat:
        raise ValueError(
            f"Seed #{seed_index} is missing reflector keys {missing_flat} while reflector_enabled=True."
        )

    focal_value = seed.get("focal_point", seed.get("reflector_target"))
    if focal_value is None:
        raise ValueError(
            f"Seed #{seed_index} is missing 'focal_point' (or 'reflector_target') while reflector_enabled=True."
        )

    u_val = float(seed["reflector_u"])
    v_val = float(seed["reflector_v"])
    focal = _coerce_vec3(f"seed[{seed_index}].focal_point", focal_value)
    return u_val, v_val, focal


def generate_gd_tasks_from_seeds(
    seeds: List[Dict[str, Any]],
    num_aps: int,
    optimize_orientation: bool,
    reflector_enabled: bool,
    gd_hyperparams: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Translate GA seeds into Ray-compatible GD work items.

    Parameters
    ----------
    seeds : list[dict]
        Topological seeds from memetic GA. Each seed must provide AP positions:

        Required:
        - ``positions``: list of ``(x, y, z)`` tuples (or ``ap_positions``).

        Optional orientation:
        - ``directions``: list of ``(dx, dy, dz)`` tuples (or ``ap_directions``).

        Optional reflector (required when ``reflector_enabled=True``):
        - flat schema: ``reflector_u``, ``reflector_v``,
          and ``focal_point`` (or ``reflector_target``), OR
        - nested schema: ``reflector`` dict with
          ``u``, ``v``, ``focal_x``, ``focal_y``, ``focal_z``.

    num_aps : int
        Expected number of APs in each seed.
    optimize_orientation : bool
        Whether GD will optimize AP orientation. If ``True``, bridge includes
        orientation initialization keys when available.
    reflector_enabled : bool
        Whether reflector initialization is required.
    gd_hyperparams : dict, optional
        Additional key-value pairs injected into every returned task dict
        (e.g. ``num_iterations``, ``learning_rate_pos``, ``learning_rate_dir``).

    Returns
    -------
    list[dict[str, Any]]
        One task dict per seed. Each task is self-contained and intended for
        ``RayParallelOptimizer.run(..., optimizer_method="gradient_descent")``
        ``work_items`` usage.

        Per task, emitted keys include:
        - ``initial_positions``: list[(x, y)]
        - ``fixed_z``: float
        - ``num_aps``: int
        - ``optimize_orientation``: bool
        - ``initial_orientations``: list[(dx, dy, dz)] | None (bridge alias)
        - ``initial_directions_xy``: list[(dx, dy)] | None (GD-compatible)
        - ``reflector_u`` / ``reflector_v`` / ``reflector_target`` (bridge keys)
        - ``initial_focal_point`` (GD-compatible reflector focal-point init)
        - + any ``gd_hyperparams`` keys

    Raises
    ------
    ValueError
        If seed schema is invalid, AP count mismatches ``num_aps``, or
        reflector keys are missing when ``reflector_enabled=True``.
    """
    if num_aps < 1:
        raise ValueError(f"num_aps must be >= 1, got {num_aps}")

    hyperparams: Dict[str, Any] = dict(gd_hyperparams or {})
    tasks: List[Dict[str, Any]] = []

    for idx, seed in enumerate(seeds):
        # Never mutate caller-owned seed dicts.
        seed_view: Mapping[str, Any] = dict(seed)

        positions_xyz = _extract_positions(seed_view, idx)
        if len(positions_xyz) != num_aps:
            raise ValueError(
                f"Seed #{idx} has {len(positions_xyz)} AP positions but num_aps={num_aps}."
            )

        fixed_z_values = {float(p[2]) for p in positions_xyz}
        if len(fixed_z_values) != 1:
            raise ValueError(
                f"Seed #{idx} has non-uniform AP z-values: {sorted(fixed_z_values)}."
            )
        fixed_z = float(next(iter(fixed_z_values)))

        task_kwargs: Dict[str, Any] = {
            "initial_positions": [(float(p[0]), float(p[1])) for p in positions_xyz],
            "fixed_z": fixed_z,
            "num_aps": int(num_aps),
            "optimize_orientation": bool(optimize_orientation),
        }

        directions = _extract_directions(seed_view)
        if optimize_orientation:
            if directions is not None and len(directions) != num_aps:
                raise ValueError(
                    f"Seed #{idx} has {len(directions)} directions but num_aps={num_aps}."
                )

            task_kwargs["initial_orientations"] = directions
            task_kwargs["initial_directions_xy"] = (
                [(float(d[0]), float(d[1])) for d in directions]
                if directions is not None
                else None
            )
        else:
            task_kwargs["initial_orientations"] = None
            task_kwargs["initial_directions_xy"] = None

        if reflector_enabled:
            reflector_u, reflector_v, focal_point = _extract_reflector(seed_view, idx)
            task_kwargs["reflector_u"] = reflector_u
            task_kwargs["reflector_v"] = reflector_v
            task_kwargs["reflector_target"] = focal_point
            task_kwargs["initial_focal_point"] = focal_point

        task_kwargs.update(hyperparams)
        tasks.append(task_kwargs)

    return tasks
