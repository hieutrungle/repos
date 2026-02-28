"""
Example: Full comparison of grid search vs gradient descent (single-process).

Runs **four** configurations:

1. Grid Search — 1 AP  (8-direction sweep per grid point)
2. Grid Search — 2 APs (alternating optimisation with joint orientation sweep)
3. Gradient Descent — 1 AP  (position + orientation)
4. Gradient Descent — 2 APs (cooperative placement with repulsion)

This script uses the same ``OptimizerFactory`` workflow as the Ray parallel
workers but runs everything sequentially in a single process.  Both methods
report position **and** orientation (direction / look_at).

Workflow per method:
    1. ``OptimizerFactory.create(method, scene, **kwargs)``
    2. ``optimizer.optimize(**params)``
    3. Unpack result:
       - GD returns ``(position, metric)`` with orientation in ``history``
       - Grid search point returns ``(position, orientation, metric)``

Grid search iterates over all points via ``generate_grid_positions()`` (1-AP)
or ``generate_alternating_grid_tasks()`` (multi-AP, alternating) and evaluates
each with ``SinglePointGridSearchOptimizer``.

Usage:
    python examples/full_comparison.py
"""

import time
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import sionna.rt
from sionna.rt import RadioMapSolver

from reflector_position import (
    setup_building_floor_scene,
    OptimizerFactory,
    rss_to_dbm,
    ReflectorController,
    PercentileCoverageObjective,
)
from reflector_position.optimizers.grid_search import (
    generate_grid_positions,
    generate_alternating_grid_tasks,
    generate_reflector_grid_tasks,
    SinglePointGridSearchOptimizer,
)
from reflector_position.metrics import POWER_EPSILON


# -- Configuration (mirrors ray_parallel_example.py) ---------------------------

SCENE_PATH = Path.home() / "blender" / "models" / "building_floor" / "building_floor.xml"

POSITION_BOUNDS = {
    "x_min": 5.0,
    "x_max": 35.0,
    "y_min": 5.0,
    "y_max": 35.0,
}

FIXED_Z = 3.8

# Shared ray-tracing parameters
RT_PARAMS = {
    "samples_per_tx": 1_000_000,
    "max_depth": 13,
}

OUTPUT_DIR = "results/full_comparison"
VERIFICATION_DIR = "results/verification"

# 2-AP alternating optimisation defaults
ALTERNATING_ROUNDS = 2  # number of full AP1→AP2 sweeps
INITIAL_AP_POSITIONS_2AP = [(7.0, 7.0), (23.0, 23.0)]


def _rss_watts_to_dbm(rss_watt: float) -> float:
    """Convert RSS from Watts (linear) to dBm."""
    return 10.0 * np.log10(max(rss_watt, POWER_EPSILON)) + 30.0


# -- Helper: unpack result (same logic as OptimizationWorker.optimize) ---------

def _unpack_result(optimizer, result):
    """
    Unpack optimizer result into a unified dict.

    Follows the same logic as ``OptimizationWorker.optimize()`` in
    ``ray_parallel_optimizer.py`` so that downstream analysis code
    can consume results from either pathway without changes.
    """
    # Unpack result tuple
    result_orientation = None
    if isinstance(result, tuple) and len(result) == 3:
        final_position, result_orientation, final_metric = result
    elif isinstance(result, tuple) and len(result) == 2:
        final_position, final_metric = result
    else:
        final_position = None
        final_metric = float("-inf")

    best_position = final_position
    best_metric = final_metric
    best_iteration = -1

    # GD: find the BEST iteration, not just the final one
    if hasattr(optimizer, "history") and "min_rss_values" in optimizer.history:
        rss_values = optimizer.history["min_rss_values"]
        positions = optimizer.history.get("positions", [])
        if rss_values and positions and len(rss_values) == len(positions):
            best_iter_idx = int(np.argmax(rss_values))
            best_metric = float(rss_values[best_iter_idx])
            best_position = positions[best_iter_idx]
            best_iteration = best_iter_idx

    # Extract orientation
    best_direction = None
    final_direction = None
    best_look_at = None
    final_look_at = None

    # Case 1: orientation returned directly (grid_search_point 8-dir sweep)
    if result_orientation is not None:
        best_direction = np.asarray(result_orientation).tolist()
        final_direction = best_direction
        if best_position is not None:
            pos_arr = np.asarray(best_position)
            dir_arr = np.asarray(result_orientation)
            best_look_at = (pos_arr + dir_arr).tolist()
            final_look_at = best_look_at

    # Case 2: orientation stored in GD history
    elif hasattr(optimizer, "history"):
        directions = optimizer.history.get("directions", [])
        look_at_targets = optimizer.history.get("look_at_targets", [])
        if directions:
            final_direction = np.asarray(directions[-1]).tolist()
            if 0 <= best_iteration < len(directions):
                best_direction = np.asarray(directions[best_iteration]).tolist()
            else:
                best_direction = final_direction
        if look_at_targets:
            final_look_at = np.asarray(look_at_targets[-1]).tolist()
            if 0 <= best_iteration < len(look_at_targets):
                best_look_at = np.asarray(look_at_targets[best_iteration]).tolist()
            else:
                best_look_at = final_look_at

    # Serialise
    if best_position is not None:
        best_position = np.asarray(best_position).tolist()
    else:
        best_position = [0.0, 0.0, 0.0]

    if final_position is not None:
        final_position = np.asarray(final_position).tolist()
    else:
        final_position = [0.0, 0.0, 0.0]

    metric_linear = float(best_metric) if best_metric is not None else 0.0
    metric_dbm = _rss_watts_to_dbm(metric_linear)

    return {
        "best_position": best_position,
        "best_metric": metric_linear,
        "best_metric_dbm": metric_dbm,
        "best_iteration": best_iteration,
        "final_position": final_position,
        "best_direction": best_direction,
        "final_direction": final_direction,
        "best_look_at": best_look_at,
        "final_look_at": final_look_at,
    }


# -- Run methods ---------------------------------------------------------------

def run_gradient_descent(scene, num_aps: int = 1, verbose: bool = True):
    """Run gradient descent via OptimizerFactory (single or multi-AP)."""
    print("\n" + "=" * 80)
    print(f"GRADIENT DESCENT OPTIMIZATION ({num_aps} AP{'s' if num_aps > 1 else ''}, with orientation)")
    print("=" * 80)

    if num_aps == 1:
        gd_kwargs = dict(
            initial_position=(15.0, 15.0),
            position_bounds=POSITION_BOUNDS,
            fixed_z=FIXED_Z,
        )
    else:
        gd_kwargs = dict(
            initial_positions=[(7.0, 7.0), (23.0, 23.0)],
            position_bounds=POSITION_BOUNDS,
            fixed_z=FIXED_Z,
            repulsion_weight=1.0,
        )

    optimizer = OptimizerFactory.create(
        method="gradient_descent",
        scene=scene,
        **gd_kwargs,
    )

    optimization_params = {
        **RT_PARAMS,
        "num_iterations": 30,
        "learning_rate": 0.5,
        "use_soft_min": True,
        "temperature": 0.2,
        "verbose": verbose,
    }

    start = time.time()
    result = optimizer.optimize(**optimization_params)
    elapsed = time.time() - start

    info = _unpack_result(optimizer, result)
    info["time_elapsed"] = elapsed
    info["num_iterations"] = len(optimizer.history["positions"])

    return info, optimizer


def run_grid_search_1ap(scene, grid_resolution: float = 5.0, verbose: bool = True):
    """
    Run 1-AP grid search with 8-direction sweep via OptimizerFactory.

    Each grid point is evaluated using ``SinglePointGridSearchOptimizer``
    (the same class used by the Ray worker), which sweeps 8 orientations
    per point and picks the best.
    """
    print("\n" + "=" * 80)
    print("GRID SEARCH — 1 AP (8-direction sweep per point)")
    print("=" * 80)

    grid_positions = generate_grid_positions(
        search_bounds=POSITION_BOUNDS,
        grid_resolution=grid_resolution,
        fixed_z=FIXED_Z,
    )
    num_points = len(grid_positions)

    print(f"  Grid resolution: {grid_resolution}m")
    print(f"  Search space: x=[{POSITION_BOUNDS['x_min']}, {POSITION_BOUNDS['x_max']}], "
          f"y=[{POSITION_BOUNDS['y_min']}, {POSITION_BOUNDS['y_max']}]")
    print(f"  Total grid points: {num_points}")
    print(f"  Evaluations per point: 8 (cardinal directions)")
    print(f"  Total evaluations: {num_points * 8}")
    print("-" * 70)

    optimization_params = {
        **RT_PARAMS,
        "verbose": False,  # per-direction verbose is too noisy
    }

    best_overall = None
    all_point_results = []
    start = time.time()

    for idx, pos in enumerate(grid_positions):
        optimizer = OptimizerFactory.create(
            method="grid_search_point",
            scene=scene,
            evaluation_position=(float(pos[0]), float(pos[1])),
            fixed_z=FIXED_Z,
        )

        result = optimizer.optimize(**optimization_params)
        info = _unpack_result(optimizer, result)
        all_point_results.append(info)

        if best_overall is None or info["best_metric"] > best_overall["best_metric"]:
            best_overall = info

        if verbose and (idx + 1) % 5 == 0:
            elapsed = time.time() - start
            eta = elapsed / (idx + 1) * (num_points - idx - 1)
            print(
                f"  Progress: {idx+1}/{num_points} | "
                f"Current: ({pos[0]:.1f}, {pos[1]:.1f}) -> {info['best_metric_dbm']:.2f} dBm "
                f"dir={info['best_direction']} | "
                f"Best so far: {best_overall['best_metric_dbm']:.2f} dBm | "
                f"ETA: {eta:.1f}s"
            )

    total_time = time.time() - start
    best_overall["time_elapsed"] = total_time
    best_overall["num_evaluations"] = num_points
    best_overall["all_point_results"] = all_point_results

    if verbose:
        print("-" * 70)
        print(f"  Grid Search 1-AP Complete! ({total_time:.1f}s)")

    return best_overall


def run_grid_search_2ap(
    scene,
    grid_resolution: float = 5.0,
    num_rounds: int = ALTERNATING_ROUNDS,
    initial_positions=None,
    verbose: bool = True,
):
    """
    Run 2-AP grid search using alternating optimisation.

    In each round every AP is swept over the grid while the other is fixed.
    Both APs' orientations are swept (``None`` → 8 dirs each for the
    active AP; fixed AP keeps its last-best orientation).

    Args:
        scene: Sionna Scene with 2 transmitters.
        grid_resolution: Grid spacing in metres.
        num_rounds: Number of full alternation rounds (AP0→AP1 counts as 1).
        initial_positions: Starting ``[(x,y), (x,y)]`` for the two APs.
        verbose: Print progress.

    Returns:
        Dict with best positions, orientations, metric, and timing info.
    """
    if initial_positions is None:
        initial_positions = list(INITIAL_AP_POSITIONS_2AP)

    print("\n" + "=" * 80)
    print("GRID SEARCH — 2 APs (alternating optimisation)")
    print("=" * 80)
    print(f"  Grid resolution: {grid_resolution}m")
    print(f"  Alternating rounds: {num_rounds}")
    print(f"  Initial AP positions: {initial_positions}")
    print(f"  Search space: x=[{POSITION_BOUNDS['x_min']}, {POSITION_BOUNDS['x_max']}], "
          f"y=[{POSITION_BOUNDS['y_min']}, {POSITION_BOUNDS['y_max']}]")

    optimization_params = {
        **RT_PARAMS,
        "verbose": False,
    }

    # Current best state for each AP
    current_positions = list(initial_positions)  # [(x,y), (x,y)]
    current_orientations = [None, None]  # sweep both initially

    best_metric = -float("inf")
    best_metric_dbm = -200.0
    best_positions = list(current_positions)
    best_directions = [None, None]

    start = time.time()

    for rnd in range(num_rounds):
        for active_idx in range(2):
            fixed_idx = 1 - active_idx

            # Build orientations: active AP sweeps, fixed uses last-best
            orientations_for_tasks = [None, None]
            orientations_for_tasks[fixed_idx] = current_orientations[fixed_idx]
            # active AP's orientation is None → sweep

            tasks = generate_alternating_grid_tasks(
                active_ap_idx=active_idx,
                search_bounds=POSITION_BOUNDS,
                fixed_positions=current_positions,
                fixed_orientations=orientations_for_tasks,
                grid_resolution=grid_resolution,
                fixed_z=FIXED_Z,
            )
            num_tasks = len(tasks)

            if verbose:
                print(f"\n  Round {rnd+1}/{num_rounds} — sweeping AP{active_idx} "
                      f"({num_tasks} grid points, AP{fixed_idx} fixed at "
                      f"{current_positions[fixed_idx]})")

            round_best_metric = -float("inf")
            round_best_info = None

            for t_idx, task_kwargs in enumerate(tasks):
                optimizer = OptimizerFactory.create(
                    method="grid_search_point",
                    scene=scene,
                    **task_kwargs,
                )
                result = optimizer.optimize(**optimization_params)
                info = _unpack_result(optimizer, result)

                if info["best_metric"] > round_best_metric:
                    round_best_metric = info["best_metric"]
                    round_best_info = info
                    # Store per-AP orientations from the optimizer
                    round_best_orientations = optimizer.results.get(
                        "best_orientations", None
                    )

                if verbose and (t_idx + 1) % max(1, num_tasks // 5) == 0:
                    elapsed = time.time() - start
                    print(
                        f"    Progress: {t_idx+1}/{num_tasks} | "
                        f"Best so far: {_rss_watts_to_dbm(round_best_metric):.2f} dBm"
                    )

            # Update current state with round winner
            if round_best_info is not None:
                # Multi-AP: best_position is a list of lists
                pos = round_best_info["best_position"]
                if isinstance(pos[0], (list, tuple)):
                    current_positions = [(p[0], p[1]) for p in pos]
                else:
                    # Single result but we know the full config from task
                    current_positions[active_idx] = (pos[0], pos[1])

                # Update orientations
                if round_best_orientations is not None:
                    for i, d in enumerate(round_best_orientations):
                        current_orientations[i] = tuple(d) if d is not None else None

                if round_best_metric > best_metric:
                    best_metric = round_best_metric
                    best_metric_dbm = _rss_watts_to_dbm(best_metric)
                    best_positions = list(current_positions)
                    best_directions = list(current_orientations)

            if verbose:
                print(
                    f"    → AP{active_idx} best: {current_positions[active_idx]}, "
                    f"joint P5 RSS = {_rss_watts_to_dbm(round_best_metric):.2f} dBm"
                )

    total_time = time.time() - start

    # Build result dict
    best_pos_3d = [
        [p[0], p[1], FIXED_Z] for p in best_positions
    ]
    result_info = {
        "best_position": best_pos_3d,
        "best_metric": best_metric,
        "best_metric_dbm": best_metric_dbm,
        "best_direction": [
            list(d) if d is not None else None for d in best_directions
        ],
        "best_look_at": None,
        "final_position": best_pos_3d,
        "final_direction": [
            list(d) if d is not None else None for d in best_directions
        ],
        "final_look_at": None,
        "best_iteration": -1,
        "time_elapsed": total_time,
        "num_rounds": num_rounds,
        "grid_resolution": grid_resolution,
    }

    if verbose:
        print("-" * 70)
        print(f"  Grid Search 2-AP Complete! ({total_time:.1f}s)")
        print(f"  Best AP0: ({best_positions[0][0]:.2f}, {best_positions[0][1]:.2f})")
        print(f"  Best AP1: ({best_positions[1][0]:.2f}, {best_positions[1][1]:.2f})")
        print(f"  Best Min RSS: {best_metric_dbm:.2f} dBm")

    return result_info


# -- Reflector integration verification ----------------------------------------

def run_reflector_verification(verbose: bool = True):
    """Visual verification of reflector integration into the scene.

    Creates two scenes (with and without the reflector), computes radio
    maps for both, and renders four images:

    1. ``scene_no_reflector.png``  — geometry only, baseline
    2. ``scene_with_reflector.png`` — geometry with reflector visible
    3. ``coverage_no_reflector.png``  — radio map overlay, baseline
    4. ``coverage_with_reflector.png`` — radio map overlay with reflector

    All outputs are written to ``VERIFICATION_DIR``.
    """
    os.makedirs(VERIFICATION_DIR, exist_ok=True)

    # Camera -------------------------------------------------------------------
    cam = sionna.rt.Camera(position=[20, 20, 60],
                           look_at=[20, 20.1, 1.5])

    # Reflector wall bounding box:
    # Two corner points define the area the reflector can slide on.
    # u ∈ [0,1] sweeps horizontally (left→right), v ∈ [0,1] vertically (bottom→top).
    wall_top_left     = [15.0, 34.0, 3.0]   # (x1, y1, z_top)
    wall_bottom_right = [34.0, 34.0, 1.0]   # (x2, y2, z_bottom)

    # Radio-map solver parameters (lighter than optimisation runs)
    rm_samples = 500_000
    rm_depth   = 13

    # ==================================================================
    # A. Baseline scene — NO reflector
    # ==================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("REFLECTOR VERIFICATION — setting up baseline scene (no reflector)")
        print("=" * 80)

    scene_base = setup_building_floor_scene(
        scene_path=str(SCENE_PATH),
        frequency=5.18e9,
        tx_power_dbm=5.0,
        reflector_enabled=False,
    )

    # Render geometry ----------------------------------------------------------
    if verbose:
        print("  Rendering baseline geometry ...")
    scene_base.render_to_file(
        camera=cam,
        filename=os.path.join(VERIFICATION_DIR, "scene_no_reflector.png"),
        resolution=(1280, 960),
        num_samples=512,
        show_devices=True,
        show_orientations=True,
    )
    if verbose:
        print("    -> saved scene_no_reflector.png")

    # Compute radio map --------------------------------------------------------
    if verbose:
        print("  Computing baseline radio map ...")
    rm_solver = RadioMapSolver()
    rm_base = rm_solver(
        scene_base,
        cell_size=(1.0, 1.0),
        samples_per_tx=rm_samples,
        max_depth=rm_depth,
        refraction=True,
        diffraction=True,
    )

    # Render coverage ----------------------------------------------------------
    if verbose:
        print("  Rendering baseline coverage map ...")
    scene_base.render_to_file(
        camera=cam,
        filename=os.path.join(VERIFICATION_DIR, "coverage_no_reflector.png"),
        resolution=(1280, 960),
        num_samples=512,
        radio_map=rm_base,
        rm_metric="rss",
        rm_db_scale=True,
        show_devices=True,
        show_orientations=True,
    )
    if verbose:
        print("    -> saved coverage_no_reflector.png")

    # ==================================================================
    # B. Scene WITH reflector
    # ==================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("REFLECTOR VERIFICATION — setting up scene WITH reflector")
        print("=" * 80)

    scene_refl, reflector_ctrl = setup_building_floor_scene(
        scene_path=str(SCENE_PATH),
        frequency=5.18e9,
        tx_power_dbm=5.0,
        reflector_enabled=True,
        reflector_size=(2.0, 2.0),
        wall_top_left=wall_top_left,
        wall_bottom_right=wall_bottom_right,
        focal_point=[10.0, 20.0, 1.5],  # aim reflected energy toward the Rx
        device="cpu",  # verification runs on CPU; no GPU optimiser needed
    )

    # Place reflector at u=0.5, v=0.5 (centre of wall patch) and orient it
    reflector_ctrl.u = reflector_ctrl._to_tensor(np.array(1.0))
    reflector_ctrl.v = reflector_ctrl._to_tensor(np.array(0.5))
    reflector_ctrl.orient_to_target()   # orient toward focal_point using Tx
    reflector_ctrl.apply_to_scene()     # push into Mitsuba scene graph

    if verbose:
        print(f"  Reflector state:\n{reflector_ctrl}")

    # Render geometry ----------------------------------------------------------
    if verbose:
        print("  Rendering scene with reflector ...")
    scene_refl.render_to_file(
        camera=cam,
        filename=os.path.join(VERIFICATION_DIR, "scene_with_reflector.png"),
        resolution=(1280, 960),
        num_samples=512,
        show_devices=True,
        show_orientations=True,
    )
    if verbose:
        print("    -> saved scene_with_reflector.png")

    # Compute radio map --------------------------------------------------------
    if verbose:
        print("  Computing radio map with reflector ...")
    rm_refl = rm_solver(
        scene_refl,
        cell_size=(1.0, 1.0),
        samples_per_tx=rm_samples,
        max_depth=rm_depth,
        refraction=True,
        diffraction=True,
    )

    # Render coverage ----------------------------------------------------------
    if verbose:
        print("  Rendering coverage map with reflector ...")
    scene_refl.render_to_file(
        camera=cam,
        filename=os.path.join(VERIFICATION_DIR, "coverage_with_reflector.png"),
        resolution=(1280, 960),
        num_samples=512,
        radio_map=rm_refl,
        rm_metric="rss",
        rm_db_scale=True,
        show_devices=True,
        show_orientations=True,
    )
    if verbose:
        print("    -> saved coverage_with_reflector.png")

    # Summary ------------------------------------------------------------------
    if verbose:
        print("\n" + "-" * 80)
        print(f"  Verification images saved to: {VERIFICATION_DIR}/")
        print("    - scene_no_reflector.png")
        print("    - scene_with_reflector.png")
        print("    - coverage_no_reflector.png")
        print("    - coverage_with_reflector.png")
        print("-" * 80)

    return scene_base, scene_refl, reflector_ctrl


# -- Reflector grid search (sequential) ----------------------------------------

def run_reflector_grid_search(
    verbose: bool = True,
    u_steps: int = 3,
    v_steps: int = 3,
    target_resolution: float = 5.0,
) -> Dict[str, Any]:
    """Sequential grid search over reflector wall position and focal target.

    Phase A is assumed complete — APs are frozen at known-good positions
    and orientations.  This function sweeps the reflector's *(u, v)* wall
    coordinates and the focal-target *(x, y)* grid, evaluating each
    configuration with :class:`SinglePointGridSearchOptimizer`.

    Parameters
    ----------
    verbose : bool
        Print per-task progress.
    u_steps : int
        Number of uniformly-spaced *u* samples in [0, 1].
    v_steps : int
        Number of uniformly-spaced *v* samples in [0, 1].
    target_resolution : float
        Spacing (metres) for the focal-target XY grid.

    Returns
    -------
    dict
        Best reflector configuration and metric.
    """
    print("\n" + "=" * 80)
    print("REFLECTOR GRID SEARCH — sequential validation")
    print("=" * 80)

    # Frozen AP configuration (from prior Phase A) ----------------------------
    fixed_ap_positions: List[Tuple[float, float, float]] = [(10.0, 20.0, 3.8)]
    fixed_ap_orientations: List[Tuple[float, float, float]] = [(0.0, 1.0, 0.0)]

    # Wall bounding box -------------------------------------------------------
    wall_top_left     = [15.0, 34.0, 3.0]
    wall_bottom_right = [34.0, 34.0, 1.0]

    # Focal-target search area ------------------------------------------------
    target_bounds = {
        "x_min": 5.0,
        "x_max": 35.0,
        "y_min": 5.0,
        "y_max": 35.0,
    }
    target_z = 1.5  # receiver height

    # Generate work items ------------------------------------------------------
    tasks = generate_reflector_grid_tasks(
        fixed_ap_positions=fixed_ap_positions,
        fixed_ap_orientations=fixed_ap_orientations,
        u_steps=u_steps,
        v_steps=v_steps,
        target_bounds=target_bounds,
        target_resolution=target_resolution,
        target_z=target_z,
    )
    num_tasks = len(tasks)

    print(f"  Wall: top_left={wall_top_left}, bottom_right={wall_bottom_right}")
    print(f"  u_steps={u_steps}, v_steps={v_steps}")
    print(f"  Target bounds: {target_bounds}")
    print(f"  Target resolution: {target_resolution} m  (z={target_z})")
    print(f"  Total tasks: {num_tasks}")
    print("-" * 70)

    # Create a single scene + controller (re-used across all tasks) -----------
    scene, reflector_ctrl = setup_building_floor_scene(
        scene_path=str(SCENE_PATH),
        frequency=5.18e9,
        tx_power_dbm=5.0,
        reflector_enabled=True,
        reflector_size=(2.0, 2.0),
        wall_top_left=wall_top_left,
        wall_bottom_right=wall_bottom_right,
        focal_point=[20.0, 20.0, target_z],  # placeholder, overwritten per task
        device="cpu",
    )

    # Ray-tracing parameters (lighter for validation) --------------------------
    rt_params = {
        "samples_per_tx": 500_000,
        "max_depth": 13,
        "coverage_threshold_dbm": -100.0,
        "verbose": False,
    }

    best_rss = -float("inf")
    best_task: Optional[Dict[str, Any]] = None
    best_rss_dbm = -200.0

    start = time.time()

    for idx, task in enumerate(tasks):
        optimizer = SinglePointGridSearchOptimizer(
            scene=scene,
            evaluation_positions=task["evaluation_positions"],
            evaluation_orientations=task["evaluation_orientations"],
            fixed_z=task["fixed_z"],
            reflector_controller=reflector_ctrl,
            reflector_u=task["reflector_u"],
            reflector_v=task["reflector_v"],
            reflector_target=task["reflector_target"],
        )

        _pos, _orient, rss_watts = optimizer.optimize(**rt_params)
        rss_dbm = _rss_watts_to_dbm(rss_watts)

        if rss_watts > best_rss:
            best_rss = rss_watts
            best_rss_dbm = rss_dbm
            best_task = {
                "reflector_u": task["reflector_u"],
                "reflector_v": task["reflector_v"],
                "reflector_target": task["reflector_target"],
                "rss_watts": rss_watts,
                "rss_dbm": rss_dbm,
            }

        if verbose and (idx % max(1, num_tasks // 20) == 0 or idx == num_tasks - 1):
            u_val = task["reflector_u"]
            v_val = task["reflector_v"]
            tgt = task["reflector_target"]
            print(
                f"  [{idx + 1:>{len(str(num_tasks))}}/{num_tasks}] "
                f"u={u_val:.2f} v={v_val:.2f} "
                f"target=({tgt[0]:.1f},{tgt[1]:.1f},{tgt[2]:.1f}) "
                f"-> {rss_dbm:.2f} dBm  "
                f"(best so far: {best_rss_dbm:.2f} dBm)"
            )

    total_time = time.time() - start

    print("-" * 70)
    print(f"  Reflector Grid Search Complete! ({total_time:.1f}s)")
    if best_task is not None:
        print(f"  Best u={best_task['reflector_u']:.3f}, "
              f"v={best_task['reflector_v']:.3f}")
        print(f"  Best target=({best_task['reflector_target'][0]:.1f}, "
              f"{best_task['reflector_target'][1]:.1f}, "
              f"{best_task['reflector_target'][2]:.1f})")
        print(f"  Best Min RSS: {best_rss_dbm:.2f} dBm")
    print("=" * 80)

    result = {
        "best_task": best_task,
        "best_rss": best_rss,
        "best_rss_dbm": best_rss_dbm,
        "num_tasks": num_tasks,
        "time_elapsed": total_time,
        "u_steps": u_steps,
        "v_steps": v_steps,
        "target_resolution": target_resolution,
    }
    return result


# -- Joint GD optimization: AP + Reflector -------------------------------------

def run_gradient_descent_with_reflector(
    scene_path: str = str(SCENE_PATH),
    *,
    num_aps: int = 2,
    initial_ap_positions: Optional[List[Tuple[float, float]]] = None,
    reflector_size: Tuple[float, float] = (2.0, 2.0),
    wall_top_left: List[float] = [15.0, 34.0, 3.0],
    wall_bottom_right: List[float] = [34.0, 34.0, 1.0],
    initial_focal_point: Optional[Tuple[float, float, float]] = None,
    num_iterations: int = 30,
    learning_rate: float = 0.5,
    samples_per_tx: int = 1_000_000,
    max_depth: int = 13,
    temperature: float = 0.2,
    shadow_quantile: float = 0.05,
    alpha: float = 0.6,
    beta: float = 0.4,
    coverage_threshold_dbm: float = -120.0,
    coverage_temperature: float = 2.0,
    fairness_loss_type: str = "auto",
    rx_position: Tuple[float, float, float] = (16.0, 6.5, 1.5),
    frequency: float = 5.18e9,
    tx_power_dbm: float = 5.0,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Any]:
    """Run joint AP + Reflector gradient-descent optimisation.

    Creates a scene with a passive reflector, a :class:`ReflectorController`,
    and a :class:`GradientDescentAPOptimizer` that jointly learns AP
    positions/orientations **and** reflector wall placement + focal point.

    Parameters
    ----------
    scene_path : str
        Path to the Mitsuba/Sionna XML scene file.
    num_aps : int
        Number of APs to optimise.
    initial_ap_positions : list of (float, float), optional
        Starting ``(x, y)`` per AP.  Defaults to ``INITIAL_AP_POSITIONS_2AP``.
    reflector_size : tuple of float
        ``(width, height)`` of the reflector mesh in metres.
    wall_top_left, wall_bottom_right : list of float
        3-D corner points of the reflector's wall bounding box.
    initial_focal_point : tuple of (float, float, float), optional
        Starting focal point for reflector beam-forming.  Defaults to
        centre of position bounds at z = 1.5.
    num_iterations : int
        Gradient-descent iterations.
    learning_rate : float
        Base learning rate for AP position parameters.
    samples_per_tx, max_depth : int
        Ray-tracing fidelity knobs.
    temperature : float
        Softmin temperature for fairness loss.  When a reflector is active
        this is the ``MaskedSoftMinLoss`` temperature (higher → sharper min);
        otherwise it is the legacy ``normalized_softmin_loss`` temperature.
    shadow_quantile : float
        Fraction of lowest-signal cells to mask out inside
        ``MaskedSoftMinLoss`` (reflector shadow dead zone).  Default 0.05.
    alpha, beta : float
        Weights for fairness and coverage losses.
    coverage_threshold_dbm : float
        Threshold for coverage-sigmoid loss (dBm).
    coverage_temperature : float
        Sigmoid sharpness for coverage loss.
    fairness_loss_type : str
        Which fairness loss to use: ``"auto"`` (default — masked_softmin
        when reflector is active, softmin otherwise), ``"softmin"``,
        ``"masked_softmin"``, or ``"percentile"``.
    rx_position : tuple of float
        Receiver position ``(x, y, z)``.
    frequency : float
        Operating frequency in Hz.
    tx_power_dbm : float
        Total transmitter power in dBm.
    verbose : bool
        Print per-iteration progress.

    Returns
    -------
    info : dict
        Unpacked result dictionary (same schema as other ``run_*`` helpers).
    optimizer : GradientDescentAPOptimizer
        The optimizer instance (for plotting / further inspection).
    """
    from reflector_position.optimizers.gradient_descent import (
        GradientDescentAPOptimizer,
    )

    print("\n" + "=" * 80)
    print(f"GRADIENT DESCENT — {num_aps} AP(s) + REFLECTOR (joint optimisation)")
    print("=" * 80)

    if initial_ap_positions is None:
        initial_ap_positions = list(INITIAL_AP_POSITIONS_2AP[:num_aps])

    # Build TX position tuples with z
    tx_positions_3d = [
        (p[0], p[1], FIXED_Z) for p in initial_ap_positions
    ]

    # Scene + reflector
    scene, reflector_ctrl = setup_building_floor_scene(
        scene_path=scene_path,
        frequency=frequency,
        tx_power_dbm=tx_power_dbm,
        tx_positions=tx_positions_3d,
        rx_position=rx_position,
        reflector_enabled=True,
        reflector_size=reflector_size,
        wall_top_left=wall_top_left,
        wall_bottom_right=wall_bottom_right,
        focal_point=initial_focal_point,
    )

    # Instantiate the joint optimizer
    optimizer = GradientDescentAPOptimizer(
        scene=scene,
        initial_positions=initial_ap_positions,
        fixed_z=FIXED_Z,
        position_bounds=POSITION_BOUNDS,
        optimize_orientation=True,
        repulsion_weight=1.0 if num_aps > 1 else 0.0,
        reflector_controller=reflector_ctrl,
        initial_focal_point=initial_focal_point,
    )

    start = time.time()
    result = optimizer.optimize(
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        samples_per_tx=samples_per_tx,
        max_depth=max_depth,
        use_soft_min=True,
        temperature=temperature,
        shadow_quantile=shadow_quantile,
        alpha=alpha,
        beta=beta,
        coverage_threshold_dbm=coverage_threshold_dbm,
        coverage_temperature=coverage_temperature,
        fairness_loss_type=fairness_loss_type,
        verbose=verbose,
    )
    elapsed = time.time() - start

    info = _unpack_result(optimizer, result)
    info["time_elapsed"] = elapsed
    info["num_iterations"] = len(optimizer.history["positions"])

    # Append reflector-specific summary
    refl_snap = optimizer._snapshot_reflector()
    info["reflector_u"] = refl_snap["u"]
    info["reflector_v"] = refl_snap["v"]
    info["reflector_focal_point"] = refl_snap["focal_point"]
    info["reflector_position"] = refl_snap["position"]

    return info, optimizer


# -- Physically-Aware Alternating Grid Search ----------------------------------

def run_grid_search_physically_aware(
    scene_path: str = str(SCENE_PATH),
    *,
    # --- AP grid-search parameters ---
    ap_grid_resolution: float = 5.0,
    ap_alternating_rounds: int = ALTERNATING_ROUNDS,
    initial_ap_positions: Optional[List[Tuple[float, float]]] = None,
    # --- Reflector grid-search parameters ---
    reflector_enabled: bool = True,
    reflector_size: Tuple[float, float] = (2.0, 2.0),
    wall_top_left: List[float] = [15.0, 34.0, 3.0],
    wall_bottom_right: List[float] = [34.0, 34.0, 1.0],
    u_steps: int = 5,
    v_steps: int = 5,
    target_bounds: Optional[Dict[str, float]] = None,
    target_resolution: float = 5.0,
    target_z: float = 1.5,
    # --- Objective / quantile parameters ---
    target_quantile: float = 0.05,
    shadow_fraction: Optional[float] = None,
    # --- Outer alternation ---
    outer_rounds: int = 2,
    # --- Ray-tracing parameters ---
    samples_per_tx: int = 1_000_000,
    max_depth: int = 13,
    coverage_threshold_dbm: float = -100.0,
    # --- Misc ---
    rx_position: Tuple[float, float, float] = (16.0, 6.5, 1.5),
    frequency: float = 5.18e9,
    tx_power_dbm: float = 5.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Physically-Aware Alternating Grid Search for 2 APs + 1 reflector.

    Orchestrates a derivative-free optimisation loop that alternates
    between two phases:

    **Phase A — AP placement:**
        Alternating grid search over 2 AP positions and orientations
        (8-cardinal sweep per active AP, fixed reflector configuration).

    **Phase B — Reflector placement:**
        Grid search over the reflector's wall coordinates ``(u, v)`` and
        focal-target ``(x, y)`` with APs frozen at their Phase-A winners.

    Phases A and B are repeated for ``outer_rounds`` iterations.  Each
    evaluation uses the :class:`PercentileCoverageObjective` so that the
    RF shadow cast by the opaque reflector does not dominate the metric.

    Parameters
    ----------
    scene_path : str
        Path to the Mitsuba/Sionna XML scene file.
    ap_grid_resolution : float
        Grid spacing (metres) for AP position sweep.
    ap_alternating_rounds : int
        Number of inner AP0 → AP1 alternation sweeps per Phase A.
    initial_ap_positions : list of (float, float), optional
        Starting ``(x, y)`` for AP0 and AP1.  Defaults to
        ``INITIAL_AP_POSITIONS_2AP``.
    reflector_enabled : bool
        If *False* the reflector phase is skipped entirely.
    reflector_size : tuple of float
        ``(width, height)`` of the reflector in metres.
    wall_top_left, wall_bottom_right : list of float
        3-D corner points of the reflector's wall bounding box.
    u_steps, v_steps : int
        Number of uniform samples along each wall dimension.
    target_bounds : dict, optional
        ``{x_min, x_max, y_min, y_max}`` for the focal-target grid.
        Defaults to the full building floor ``(5 – 35)``.
    target_resolution : float
        Grid spacing (metres) for the focal-target XY grid.
    target_z : float
        Fixed height for focal-target evaluation points.
    target_quantile : float
        Percentile used by :class:`PercentileCoverageObjective`.  **Must**
        exceed the fraction of the coverage grid swallowed by the
        reflector's RF shadow (see *Shadow-Area Constraint* below).
    shadow_fraction : float, optional
        Estimated shadow-area fraction.  When provided the objective
        constructor validates ``target_quantile > shadow_fraction`` and
        raises ``ValueError`` otherwise.
    outer_rounds : int
        Number of full Phase-A → Phase-B alternation cycles.
    samples_per_tx, max_depth : int
        Ray-tracing fidelity parameters.
    coverage_threshold_dbm : float
        Threshold for the auxiliary hard-coverage metric.
    rx_position : tuple of float
        Receiver position ``(x, y, z)``.
    frequency : float
        Operating frequency in Hz.
    tx_power_dbm : float
        Total transmitter power in dBm.
    verbose : bool
        Print per-phase progress.

    Returns
    -------
    dict
        Summary with keys: ``best_ap_positions``, ``best_ap_orientations``,
        ``best_reflector_u``, ``best_reflector_v``,
        ``best_reflector_target``, ``best_percentile_score``,
        ``best_percentile_score_dbm``, ``best_min_rss_dbm``,
        ``time_elapsed``, ``history``.

    Shadow-Area Constraint
    ~~~~~~~~~~~~~~~~~~~~~~
    The ``target_quantile`` must be **strictly larger** than the fraction
    of the coverage grid covered by the reflector's physical shadow.  For
    a 2 m × 2 m reflector in a 30 m × 30 m room the shadow is roughly:

    .. math::

        \\frac{2 \\times 30}{30 \\times 30} \\approx 7\\%

    (conservatively, the shadow behind the reflector extends the full
    room depth).  A ``target_quantile`` of 0.05 (5 %) may still clip the
    shadow edge; 0.10 is safer.  If you know the exact shadow area, pass
    ``shadow_fraction`` to enable a strict validation check.
    """
    # ------------------------------------------------------------------
    # 0. Instantiate the percentile objective
    # ------------------------------------------------------------------
    objective = PercentileCoverageObjective(
        target_quantile=target_quantile,
        mode="maximize",
        shadow_fraction=shadow_fraction,
    )

    if initial_ap_positions is None:
        initial_ap_positions = list(INITIAL_AP_POSITIONS_2AP)

    if target_bounds is None:
        target_bounds = {
            "x_min": 5.0,
            "x_max": 35.0,
            "y_min": 5.0,
            "y_max": 35.0,
        }

    rt_params = {
        "samples_per_tx": samples_per_tx,
        "max_depth": max_depth,
        "coverage_threshold_dbm": coverage_threshold_dbm,
        "verbose": False,
    }

    # ------------------------------------------------------------------
    # 1. Create the scene with 2 APs + reflector
    # ------------------------------------------------------------------
    tx_positions_3d = [
        (p[0], p[1], FIXED_Z) for p in initial_ap_positions
    ]
    scene_setup_kwargs: Dict[str, Any] = dict(
        scene_path=scene_path,
        frequency=frequency,
        tx_positions=tx_positions_3d,
        tx_power_dbm=tx_power_dbm,
        rx_position=rx_position,
        reflector_enabled=reflector_enabled,
        reflector_size=reflector_size,
        wall_top_left=wall_top_left,
        wall_bottom_right=wall_bottom_right,
        focal_point=[20.0, 20.0, target_z],  # placeholder
        device="cpu",  # grid search is derivative-free
    )

    if reflector_enabled:
        scene, reflector_ctrl = setup_building_floor_scene(**scene_setup_kwargs)
    else:
        scene = setup_building_floor_scene(**scene_setup_kwargs)
        reflector_ctrl = None

    # ------------------------------------------------------------------
    # State tracking
    # ------------------------------------------------------------------
    current_ap_positions: List[Tuple[float, float]] = list(initial_ap_positions)
    current_ap_orientations: List[Optional[Tuple[float, float, float]]] = [
        None, None,
    ]
    current_reflector_u: float = 0.5
    current_reflector_v: float = 0.5
    current_reflector_target: Tuple[float, float, float] = (20.0, 20.0, target_z)

    best_overall_pct: float = -float("inf")
    best_overall_rss: float = -float("inf")
    best_ap_positions: List[Tuple[float, float]] = list(current_ap_positions)
    best_ap_orientations: List[Optional[Tuple[float, float, float]]] = [None, None]
    best_reflector_u: float = current_reflector_u
    best_reflector_v: float = current_reflector_v
    best_reflector_target: Tuple[float, float, float] = current_reflector_target

    history: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("PHYSICALLY-AWARE ALTERNATING GRID SEARCH")
    print(f"  2 APs + {'1 reflector' if reflector_enabled else 'no reflector'}")
    print(f"  Objective: {target_quantile*100:.1f}th-percentile coverage")
    print(f"  Outer rounds: {outer_rounds}")
    print(f"  AP grid resolution: {ap_grid_resolution} m")
    if reflector_enabled:
        print(f"  Reflector u_steps={u_steps}, v_steps={v_steps}, "
              f"target_resolution={target_resolution} m")
    print("=" * 80)

    global_start = time.time()

    # ==================================================================
    # OUTER ALTERNATION LOOP
    # ==================================================================
    with torch.no_grad():
        for outer_rnd in range(outer_rounds):
            print(f"\n{'─'*70}")
            print(f"  OUTER ROUND {outer_rnd + 1}/{outer_rounds}")
            print(f"{'─'*70}")

            # ==========================================================
            # PHASE A: AP position & orientation sweep (alternating)
            # ==========================================================
            print(f"\n  Phase A: AP placement sweep "
                  f"({ap_alternating_rounds} inner rounds)")

            for inner_rnd in range(ap_alternating_rounds):
                for active_idx in range(2):
                    fixed_idx = 1 - active_idx

                    # Build orientations: active AP sweeps, fixed uses best
                    orientations_for_tasks: List[
                        Optional[Tuple[float, float, float]]
                    ] = [None, None]
                    orientations_for_tasks[fixed_idx] = (
                        current_ap_orientations[fixed_idx]
                    )

                    tasks = generate_alternating_grid_tasks(
                        active_ap_idx=active_idx,
                        search_bounds=POSITION_BOUNDS,
                        fixed_positions=current_ap_positions,
                        fixed_orientations=orientations_for_tasks,
                        grid_resolution=ap_grid_resolution,
                        fixed_z=FIXED_Z,
                    )
                    num_tasks = len(tasks)

                    if verbose:
                        print(
                            f"    Inner {inner_rnd+1}/{ap_alternating_rounds} "
                            f"— sweeping AP{active_idx} "
                            f"({num_tasks} pts, AP{fixed_idx} fixed at "
                            f"{current_ap_positions[fixed_idx]})"
                        )

                    phase_best_pct = -float("inf")
                    phase_best_info: Optional[Dict[str, Any]] = None
                    phase_best_orientations: Optional[List] = None

                    for t_idx, task_kwargs in enumerate(tasks):
                        optimizer = SinglePointGridSearchOptimizer(
                            scene=scene,
                            reflector_controller=reflector_ctrl,
                            reflector_u=current_reflector_u,
                            reflector_v=current_reflector_v,
                            reflector_target=current_reflector_target,
                            percentile_objective=objective,
                            **task_kwargs,
                        )
                        _pos, _orient, _rss = optimizer.optimize(**rt_params)

                        pct_score = optimizer.results.get(
                            "percentile_score", _rss
                        )
                        if pct_score > phase_best_pct:
                            phase_best_pct = pct_score
                            phase_best_info = {
                                "position": _pos,
                                "orientation": _orient,
                                "rss": _rss,
                                "percentile_score": pct_score,
                                "percentile_score_dbm": optimizer.results.get(
                                    "percentile_score_dbm",
                                    _rss_watts_to_dbm(_rss),
                                ),
                            }
                            phase_best_orientations = optimizer.results.get(
                                "best_orientations", None
                            )

                        if verbose and (
                            t_idx + 1
                        ) % max(1, num_tasks // 5) == 0:
                            print(
                                f"      {t_idx + 1}/{num_tasks} | "
                                f"best pct so far: "
                                f"{_rss_watts_to_dbm(phase_best_pct):.2f} dBm"
                            )

                    # Update AP state with phase winner
                    if phase_best_info is not None:
                        pos = phase_best_info["position"]
                        if isinstance(pos, np.ndarray):
                            pos = pos.tolist()
                        if isinstance(pos[0], (list, tuple)):
                            current_ap_positions = [
                                (p[0], p[1]) for p in pos
                            ]
                        else:
                            current_ap_positions[active_idx] = (
                                pos[0], pos[1],
                            )

                        if phase_best_orientations is not None:
                            for i, d in enumerate(phase_best_orientations):
                                current_ap_orientations[i] = (
                                    tuple(d) if d is not None else None
                                )

                    if verbose:
                        pct_dbm = _rss_watts_to_dbm(phase_best_pct)
                        print(
                            f"    → AP{active_idx} best at "
                            f"{current_ap_positions[active_idx]}, "
                            f"pct = {pct_dbm:.2f} dBm"
                        )

            # ==========================================================
            # PHASE B: Reflector sweep (u, v, focal target)
            # ==========================================================
            if reflector_enabled and reflector_ctrl is not None:
                print(f"\n  Phase B: Reflector placement sweep")

                # Freeze AP positions & orientations for reflector sweep
                frozen_ap_3d: List[Tuple[float, float, float]] = [
                    (p[0], p[1], FIXED_Z) for p in current_ap_positions
                ]
                frozen_ap_dirs: List[Tuple[float, float, float]] = [
                    o if o is not None else (0.0, 1.0, 0.0)
                    for o in current_ap_orientations
                ]

                refl_tasks = generate_reflector_grid_tasks(
                    fixed_ap_positions=frozen_ap_3d,
                    fixed_ap_orientations=frozen_ap_dirs,
                    u_steps=u_steps,
                    v_steps=v_steps,
                    target_bounds=target_bounds,
                    target_resolution=target_resolution,
                    target_z=target_z,
                )
                num_refl_tasks = len(refl_tasks)

                if verbose:
                    print(f"    {num_refl_tasks} reflector configurations")

                refl_best_pct = -float("inf")
                refl_best_task: Optional[Dict[str, Any]] = None

                for r_idx, task in enumerate(refl_tasks):
                    optimizer = SinglePointGridSearchOptimizer(
                        scene=scene,
                        evaluation_positions=task["evaluation_positions"],
                        evaluation_orientations=task[
                            "evaluation_orientations"
                        ],
                        fixed_z=task["fixed_z"],
                        reflector_controller=reflector_ctrl,
                        reflector_u=task["reflector_u"],
                        reflector_v=task["reflector_v"],
                        reflector_target=task["reflector_target"],
                        percentile_objective=objective,
                    )
                    _pos, _orient, _rss = optimizer.optimize(**rt_params)

                    pct_score = optimizer.results.get(
                        "percentile_score", _rss
                    )
                    if pct_score > refl_best_pct:
                        refl_best_pct = pct_score
                        refl_best_task = {
                            "reflector_u": task["reflector_u"],
                            "reflector_v": task["reflector_v"],
                            "reflector_target": task["reflector_target"],
                            "rss": _rss,
                            "percentile_score": pct_score,
                            "percentile_score_dbm": optimizer.results.get(
                                "percentile_score_dbm",
                                _rss_watts_to_dbm(_rss),
                            ),
                        }

                    if verbose and (
                        r_idx + 1
                    ) % max(1, num_refl_tasks // 10) == 0:
                        print(
                            f"      {r_idx + 1}/{num_refl_tasks} | "
                            f"best pct: "
                            f"{_rss_watts_to_dbm(refl_best_pct):.2f} dBm"
                        )

                # Update reflector state
                if refl_best_task is not None:
                    current_reflector_u = refl_best_task["reflector_u"]
                    current_reflector_v = refl_best_task["reflector_v"]
                    current_reflector_target = refl_best_task[
                        "reflector_target"
                    ]

                    if verbose:
                        print(
                            f"    → Reflector best: u={current_reflector_u:.3f}, "
                            f"v={current_reflector_v:.3f}, "
                            f"target={current_reflector_target}, "
                            f"pct = {refl_best_task['percentile_score_dbm']:.2f} dBm"
                        )

            # ----------------------------------------------------------
            # Track global best across outer rounds
            # ----------------------------------------------------------
            # Re-evaluate current best config to get canonical score
            round_pct = phase_best_pct
            if reflector_enabled and refl_best_task is not None:
                round_pct = max(round_pct, refl_best_pct)

            if round_pct > best_overall_pct:
                best_overall_pct = round_pct
                best_ap_positions = list(current_ap_positions)
                best_ap_orientations = list(current_ap_orientations)
                best_reflector_u = current_reflector_u
                best_reflector_v = current_reflector_v
                best_reflector_target = current_reflector_target
                if reflector_enabled and refl_best_task is not None:
                    best_overall_rss = refl_best_task["rss"]
                elif phase_best_info is not None:
                    best_overall_rss = phase_best_info["rss"]

            history.append({
                "outer_round": outer_rnd + 1,
                "ap_positions": [list(p) for p in current_ap_positions],
                "ap_orientations": [
                    list(o) if o is not None else None
                    for o in current_ap_orientations
                ],
                "reflector_u": current_reflector_u,
                "reflector_v": current_reflector_v,
                "reflector_target": list(current_reflector_target),
                "percentile_score": float(round_pct),
                "percentile_score_dbm": _rss_watts_to_dbm(round_pct),
            })

            if verbose:
                print(
                    f"\n  Round {outer_rnd + 1} summary: "
                    f"pct = {_rss_watts_to_dbm(round_pct):.2f} dBm  "
                    f"(global best: {_rss_watts_to_dbm(best_overall_pct):.2f} dBm)"
                )

    # ==================================================================
    # Summary
    # ==================================================================
    total_time = time.time() - global_start
    best_pct_dbm = _rss_watts_to_dbm(best_overall_pct)
    best_rss_dbm = _rss_watts_to_dbm(best_overall_rss)

    best_ap_3d = [[p[0], p[1], FIXED_Z] for p in best_ap_positions]

    print("\n" + "=" * 80)
    print("PHYSICALLY-AWARE GRID SEARCH — COMPLETE")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Best AP positions: {best_ap_positions}")
    print(f"  Best AP orientations: {best_ap_orientations}")
    if reflector_enabled:
        print(f"  Best reflector u={best_reflector_u:.3f}, "
              f"v={best_reflector_v:.3f}")
        print(f"  Best reflector target: {best_reflector_target}")
    print(f"  Best {target_quantile*100:.0f}th-percentile: {best_pct_dbm:.2f} dBm")
    print(f"  P5 RSS at best config: {best_rss_dbm:.2f} dBm")
    print("=" * 80)

    return {
        "best_ap_positions": best_ap_3d,
        "best_ap_orientations": [
            list(o) if o is not None else None
            for o in best_ap_orientations
        ],
        "best_reflector_u": best_reflector_u,
        "best_reflector_v": best_reflector_v,
        "best_reflector_target": list(best_reflector_target),
        "best_percentile_score": float(best_overall_pct),
        "best_percentile_score_dbm": best_pct_dbm,
        "best_min_rss": float(best_overall_rss),
        "best_min_rss_dbm": best_rss_dbm,
        "target_quantile": target_quantile,
        "outer_rounds": outer_rounds,
        "ap_grid_resolution": ap_grid_resolution,
        "reflector_u_steps": u_steps,
        "reflector_v_steps": v_steps,
        "time_elapsed": total_time,
        "history": history,
        # Standard interface keys for comparison table
        "best_position": best_ap_3d,
        "best_metric": float(best_overall_pct),
        "best_metric_dbm": best_pct_dbm,
        "best_direction": [
            list(o) if o is not None else None
            for o in best_ap_orientations
        ],
        "best_look_at": None,
        "final_position": best_ap_3d,
        "final_direction": [
            list(o) if o is not None else None
            for o in best_ap_orientations
        ],
        "final_look_at": None,
        "best_iteration": -1,
    }


def run_grid_search_2ap_with_reflector(
    scene_path: str = str(SCENE_PATH),
    *,
    ap_grid_resolution: float = 5.0,
    ap_alternating_rounds: int = ALTERNATING_ROUNDS,
    initial_ap_positions: Optional[List[Tuple[float, float]]] = None,
    reflector_size: Tuple[float, float] = (2.0, 2.0),
    wall_top_left: List[float] = [15.0, 34.0, 3.0],
    wall_bottom_right: List[float] = [34.0, 34.0, 1.0],
    u_steps: int = 5,
    v_steps: int = 5,
    target_bounds: Optional[Dict[str, float]] = None,
    target_resolution: float = 5.0,
    target_z: float = 1.5,
    target_quantile: float = 0.05,
    shadow_fraction: Optional[float] = None,
    outer_rounds: int = 2,
    samples_per_tx: int = 1_000_000,
    max_depth: int = 13,
    coverage_threshold_dbm: float = -100.0,
    rx_position: Tuple[float, float, float] = (16.0, 6.5, 1.5),
    frequency: float = 5.18e9,
    tx_power_dbm: float = 5.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run joint 2-AP + reflector position/orientation grid search.

    This is a convenience wrapper around
    :func:`run_grid_search_physically_aware` dedicated to the use case of
    optimizing two APs together with one passive reflector.

    Notes
    -----
    - APs: alternating grid search over positions + orientation sweep.
    - Reflector: grid search over wall placement ``(u, v)`` + focal target.
    - Objective: quantile-based robustness metric (default 5th percentile).
    - Entire evaluation runs under ``torch.no_grad()`` in the underlying
      implementation for derivative-free execution.
    """
    return run_grid_search_physically_aware(
        scene_path=scene_path,
        ap_grid_resolution=ap_grid_resolution,
        ap_alternating_rounds=ap_alternating_rounds,
        initial_ap_positions=initial_ap_positions,
        reflector_enabled=True,
        reflector_size=reflector_size,
        wall_top_left=wall_top_left,
        wall_bottom_right=wall_bottom_right,
        u_steps=u_steps,
        v_steps=v_steps,
        target_bounds=target_bounds,
        target_resolution=target_resolution,
        target_z=target_z,
        target_quantile=target_quantile,
        shadow_fraction=shadow_fraction,
        outer_rounds=outer_rounds,
        samples_per_tx=samples_per_tx,
        max_depth=max_depth,
        coverage_threshold_dbm=coverage_threshold_dbm,
        rx_position=rx_position,
        frequency=frequency,
        tx_power_dbm=tx_power_dbm,
        verbose=verbose,
    )


# -- Comparison and reporting --------------------------------------------------

def _fmt_dir(d):
    """Format a direction vector for printing.  Handles nested lists for multi-AP."""
    if d is None:
        return "N/A"
    if isinstance(d[0], (list, tuple, np.ndarray)):
        return " | ".join(_fmt_dir(sub) for sub in d)
    return f"({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f})"


def _fmt_pos(p):
    """Format a position for printing.  Handles nested lists for multi-AP."""
    if p is None:
        return "N/A"
    # Multi-AP: list of lists, e.g. [[x0,y0,z0],[x1,y1,z1]]
    if isinstance(p[0], (list, tuple, np.ndarray)):
        return " | ".join(_fmt_pos(sub) for sub in p)
    return f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"


def main():
    # ==================================================================
    # 0. REFLECTOR INTEGRATION VERIFICATION
    # ==================================================================
    # run_reflector_verification(verbose=True)

    # ==================================================================
    # 0b. REFLECTOR GRID SEARCH (sequential validation)
    # ==================================================================
    # refl_gs_info = run_reflector_grid_search(
    #     verbose=True,
    #     u_steps=3,
    #     v_steps=3,
    #     target_resolution=10.0,  # coarse grid for validation
    # )

    # # ==================================================================
    # # 1-AP RUNS (scene with 1 transmitter)
    # # ==================================================================
    # print("Setting up 1-AP scene...")
    # scene_1ap = setup_building_floor_scene(
    #     scene_path=str(SCENE_PATH),
    #     frequency=5.18e9,
    #     tx_power_dbm=5.0,
    # )

    # # -- 1-AP Grid Search ------------------------------------------------------
    # gs_1ap_info = run_grid_search_1ap(scene_1ap, grid_resolution=5.0, verbose=True)

    # # -- 1-AP Gradient Descent -------------------------------------------------
    # gd_1ap_info, gd_1ap_opt = run_gradient_descent(
    #     scene_1ap, num_aps=1, verbose=True,
    # )

    # ==================================================================
    # 2-AP + Reflector RUN (joint physically-aware grid search)
    # ==================================================================
    # print("\n\nRunning 2-AP + Reflector grid search...")
    # gs_2ap_refl_info = run_grid_search_2ap_with_reflector(
    #     scene_path=str(SCENE_PATH),
    #     ap_grid_resolution=5.0,
    #     ap_alternating_rounds=ALTERNATING_ROUNDS,
    #     initial_ap_positions=list(INITIAL_AP_POSITIONS_2AP),
    #     u_steps=3,
    #     v_steps=3,
    #     target_resolution=10.0,
    #     target_quantile=0.05,
    #     outer_rounds=2,
    #     verbose=True,
    # )

    # # -- 2-AP Gradient Descent -------------------------------------------------
    # gd_2ap_info, gd_2ap_opt = run_gradient_descent(
    #     scene_2ap, num_aps=2, verbose=True,
    # )

    # ==================================================================
    # 2-AP + Reflector GD (joint differentiable optimisation)
    # ==================================================================
    print("\n\nRunning 2-AP + Reflector Gradient Descent...")
    gd_2ap_refl_info, gd_2ap_refl_opt = run_gradient_descent_with_reflector(
        scene_path=str(SCENE_PATH),
        num_aps=2,
        initial_ap_positions=list(INITIAL_AP_POSITIONS_2AP),
        wall_top_left=[15.0, 34.0, 3.0],
        wall_bottom_right=[34.0, 34.0, 1.0],
        initial_focal_point=(20.0, 20.0, 1.5),
        num_iterations=30,
        learning_rate=0.5,
        samples_per_tx=1_000_000,
        max_depth=13,
        fairness_loss_type="auto",
        verbose=True,
    )

    # ==================================================================
    # COMPARISON TABLE
    # ==================================================================
    print("\n" + "=" * 80)
    print("RUN SUMMARY — 2 APs + Reflector: Grid Search vs Gradient Descent")
    print("=" * 80)

    configs = [
        # ("GS 2-AP + Reflector", gs_2ap_refl_info),
        ("GD 2-AP + Reflector", gd_2ap_refl_info),
    ]

    for label, info in configs:
        print(f"\n  {label}:")
        print(f"    Best position:  {_fmt_pos(info['best_position'])}")
        print(f"    Best direction: {_fmt_dir(info['best_direction'])}")
        if info.get("best_look_at"):
            print(f"    Best look_at:   {_fmt_pos(info['best_look_at'])}")
        if info.get("final_position") and info["final_position"] != info["best_position"]:
            print(f"    Final position: {_fmt_pos(info['final_position'])}")
        if info.get("final_direction") and info["final_direction"] != info["best_direction"]:
            print(f"    Final dir:      {_fmt_dir(info['final_direction'])}")
        if info.get("best_iteration", -1) >= 0:
            print(f"    Best iteration: {info['best_iteration'] + 1}"
                  f"/{info.get('num_iterations', '?')}")
        print(f"    Min RSS:        {info['best_metric_dbm']:.2f} dBm")
        print(f"    Time:           {info['time_elapsed']:.1f}s")
        # Reflector details (if present)
        if info.get("reflector_u") is not None:
            print(f"    Reflector u={info['reflector_u']:.4f}, v={info['reflector_v']:.4f}")
        if info.get("reflector_focal_point") is not None:
            fp = info["reflector_focal_point"]
            print(f"    Reflector focal: ({fp[0]:.2f}, {fp[1]:.2f}, {fp[2]:.2f})")
        if info.get("reflector_position") is not None:
            rp = info["reflector_position"]
            print(f"    Reflector pos:   ({rp[0]:.2f}, {rp[1]:.2f}, {rp[2]:.2f})")

    # Quick summary line
    print("\n" + "-" * 80)
    print("  Summary (Best Metric in dBm):")
    for label, info in configs:
        print(f"    {label}: {info['best_metric_dbm']:.2f} dBm  ({info['time_elapsed']:.1f}s)")

    # -- Save results ----------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {}
    for label, info in configs:
        key = label.strip().lower().replace(" ", "_").replace("-", "")
        # Drop non-serialisable bits
        summary[key] = {
            k: v for k, v in info.items()
            if k != "all_point_results"
        }
    out_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # # -- Plot results ----------------------------------------------------------
    # for name, opt in [("1-AP", gd_1ap_opt), ("2-AP", gd_2ap_opt)]:
    #     try:
    #         opt.plot_optimization_trajectory()
    #     except Exception as e:
    #         print(f"\nNote: Could not plot {name} GD results (requires display): {e}")


# def main():
#     # ==================================================================
#     # 0. REFLECTOR INTEGRATION VERIFICATION
#     # ==================================================================
#     run_reflector_verification(verbose=True)

#     # ==================================================================
#     # 0b. REFLECTOR GRID SEARCH (sequential validation)
#     # ==================================================================
#     refl_gs_info = run_reflector_grid_search(
#         verbose=True,
#         u_steps=3,
#         v_steps=3,
#         target_resolution=10.0,  # coarse grid for validation
#     )

#     # ==================================================================
#     # 1-AP RUNS (scene with 1 transmitter)
#     # ==================================================================
#     print("Setting up 1-AP scene...")
#     scene_1ap = setup_building_floor_scene(
#         scene_path=str(SCENE_PATH),
#         frequency=5.18e9,
#         tx_power_dbm=5.0,
#     )

#     # -- 1-AP Grid Search ------------------------------------------------------
#     gs_1ap_info = run_grid_search_1ap(scene_1ap, grid_resolution=5.0, verbose=True)

#     # -- 1-AP Gradient Descent -------------------------------------------------
#     gd_1ap_info, gd_1ap_opt = run_gradient_descent(
#         scene_1ap, num_aps=1, verbose=True,
#     )

#     # ==================================================================
#     # 2-AP RUNS (scene with 2 transmitters)
#     # ==================================================================
#     print("\n\nSetting up 2-AP scene...")
#     scene_2ap = setup_building_floor_scene(
#         scene_path=str(SCENE_PATH),
#         frequency=5.18e9,
#         tx_power_dbm=5.0,
#         tx_positions=[
#             (INITIAL_AP_POSITIONS_2AP[0][0], INITIAL_AP_POSITIONS_2AP[0][1], FIXED_Z),
#             (INITIAL_AP_POSITIONS_2AP[1][0], INITIAL_AP_POSITIONS_2AP[1][1], FIXED_Z),
#         ],
#     )

#     # -- 2-AP Grid Search (alternating) ----------------------------------------
#     gs_2ap_info = run_grid_search_2ap(
#         scene_2ap, grid_resolution=5.0, num_rounds=ALTERNATING_ROUNDS, verbose=True,
#     )

#     # -- 2-AP Gradient Descent -------------------------------------------------
#     gd_2ap_info, gd_2ap_opt = run_gradient_descent(
#         scene_2ap, num_aps=2, verbose=True,
#     )

#     # ==================================================================
#     # COMPARISON TABLE
#     # ==================================================================
#     print("\n" + "=" * 80)
#     print("FULL COMPARISON — 1 AP vs 2 APs × Grid Search vs Gradient Descent")
#     print("=" * 80)

#     configs = [
#         ("GS  1-AP", gs_1ap_info),
#         ("GD  1-AP", gd_1ap_info),
#         ("GS  2-AP", gs_2ap_info),
#         ("GD  2-AP", gd_2ap_info),
#     ]

#     for label, info in configs:
#         print(f"\n  {label}:")
#         print(f"    Best position:  {_fmt_pos(info['best_position'])}")
#         print(f"    Best direction: {_fmt_dir(info['best_direction'])}")
#         if info.get("best_look_at"):
#             print(f"    Best look_at:   {_fmt_pos(info['best_look_at'])}")
#         if info.get("final_position") and info["final_position"] != info["best_position"]:
#             print(f"    Final position: {_fmt_pos(info['final_position'])}")
#         if info.get("final_direction") and info["final_direction"] != info["best_direction"]:
#             print(f"    Final dir:      {_fmt_dir(info['final_direction'])}")
#         if info.get("best_iteration", -1) >= 0:
#             print(f"    Best iteration: {info['best_iteration'] + 1}"
#                   f"/{info.get('num_iterations', '?')}")
#         print(f"    Min RSS:        {info['best_metric_dbm']:.2f} dBm")
#         print(f"    Time:           {info['time_elapsed']:.1f}s")

#     # Quick summary line
#     print("\n" + "-" * 80)
#     print("  Summary (P5 RSS in dBm):")
#     for label, info in configs:
#         print(f"    {label}: {info['best_metric_dbm']:.2f} dBm  ({info['time_elapsed']:.1f}s)")

#     # -- Save results ----------------------------------------------------------
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     summary = {}
#     for label, info in configs:
#         key = label.strip().lower().replace(" ", "_").replace("-", "")
#         # Drop non-serialisable bits
#         summary[key] = {
#             k: v for k, v in info.items()
#             if k != "all_point_results"
#         }
#     out_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
#     with open(out_path, "w") as f:
#         json.dump(summary, f, indent=2, default=str)
#     print(f"\nResults saved to: {out_path}")

#     # -- Plot results ----------------------------------------------------------
#     for name, opt in [("1-AP", gd_1ap_opt), ("2-AP", gd_2ap_opt)]:
#         try:
#             opt.plot_optimization_trajectory()
#         except Exception as e:
#             print(f"\nNote: Could not plot {name} GD results (requires display): {e}")

if __name__ == "__main__":
    main()
