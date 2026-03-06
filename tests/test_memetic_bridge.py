"""Unit tests for memetic bridge translation (Phase 2)."""

from __future__ import annotations

from copy import deepcopy

import pytest

from reflector_position.optimizers.memetic.memetic_bridge import (
    generate_gd_tasks_from_seeds,
)


def test_generate_gd_tasks_from_seeds_schema_2ap_reflector() -> None:
    """Bridge should emit GD-ready task kwargs without mutating input seeds."""
    seeds = [
        {
            "positions": [(10.0, 12.0, 3.8), (22.0, 24.0, 3.8)],
            "directions": [(0.1, 0.9, -0.4), (-0.8, 0.3, -0.5)],
            "reflector_u": 0.25,
            "reflector_v": 0.75,
            "focal_point": (20.0, 20.0, 1.5),
        },
        {
            "positions": [(11.5, 13.5, 3.8), (25.0, 26.0, 3.8)],
            "directions": [(0.0, 1.0, -0.5), (-1.0, 0.0, -0.5)],
            "reflector_u": 0.45,
            "reflector_v": 0.55,
            "focal_point": (18.0, 19.0, 1.5),
        },
    ]
    seeds_before = deepcopy(seeds)

    gd_hyperparams = {
        "num_iterations": 50,
        "learning_rate": 0.1,
    }

    tasks = generate_gd_tasks_from_seeds(
        seeds=seeds,
        num_aps=2,
        optimize_orientation=True,
        reflector_enabled=True,
        gd_hyperparams=gd_hyperparams,
    )

    assert len(tasks) == 2

    for task in tasks:
        # Core GD initialization schema
        assert "initial_positions" in task
        assert "fixed_z" in task
        assert "num_aps" in task
        assert "optimize_orientation" in task

        # Orientation mapping (bridge alias + GD-compatible key)
        assert "initial_orientations" in task
        assert "initial_directions_xy" in task

        # Reflector mapping
        assert "reflector_u" in task
        assert "reflector_v" in task
        assert "reflector_target" in task
        assert "initial_focal_point" in task

        # Hyperparameters propagated into each task
        assert task["num_iterations"] == 50
        assert task["learning_rate"] == 0.1

        # 2-AP shape checks
        assert len(task["initial_positions"]) == 2
        assert len(task["initial_directions_xy"]) == 2

    # Specific value checks for first task
    assert tasks[0]["initial_positions"] == [(10.0, 12.0), (22.0, 24.0)]
    assert tasks[0]["fixed_z"] == 3.8
    assert tasks[0]["reflector_u"] == 0.25
    assert tasks[0]["reflector_v"] == 0.75
    assert tasks[0]["reflector_target"] == (20.0, 20.0, 1.5)

    # Ensure immutability (input seeds unchanged)
    assert seeds == seeds_before


def test_generate_gd_tasks_from_seeds_missing_reflector_keys_raises() -> None:
    """Bridge should fail fast when reflector mode is enabled but keys are missing."""
    seeds = [
        {
            "positions": [(10.0, 12.0, 3.8), (22.0, 24.0, 3.8)],
            "reflector_u": 0.25,
            # reflector_v and focal_point intentionally missing
        }
    ]

    with pytest.raises(ValueError, match="missing reflector keys|missing 'focal_point'"):
        generate_gd_tasks_from_seeds(
            seeds=seeds,
            num_aps=2,
            optimize_orientation=False,
            reflector_enabled=True,
            gd_hyperparams=None,
        )
