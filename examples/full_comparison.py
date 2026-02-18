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
from pathlib import Path

import numpy as np

from reflector_position import (
    setup_building_floor_scene,
    OptimizerFactory,
    rss_to_dbm,
)
from reflector_position.optimizers.grid_search import (
    generate_grid_positions,
    generate_alternating_grid_tasks,
)
from reflector_position.metrics import POWER_EPSILON


# -- Configuration (mirrors ray_parallel_example.py) ---------------------------

SCENE_PATH = Path.home() / "blender" / "models" / "building_floor" / "building_floor.xml"

POSITION_BOUNDS = {
    "x_min": 5.0,
    "x_max": 25.0,
    "y_min": 5.0,
    "y_max": 25.0,
}

FIXED_Z = 3.8

# Shared ray-tracing parameters
RT_PARAMS = {
    "samples_per_tx": 1_000_000,
    "max_depth": 13,
}

OUTPUT_DIR = "results/full_comparison"

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


def run_grid_search(scene, grid_resolution: float = 5.0, verbose: bool = True):
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
                    f"joint min RSS = {_rss_watts_to_dbm(round_best_metric):.2f} dBm"
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
    # 1-AP RUNS (scene with 1 transmitter)
    # ==================================================================
    print("Setting up 1-AP scene...")
    scene_1ap = setup_building_floor_scene(
        scene_path=str(SCENE_PATH),
        frequency=5.18e9,
        tx_power_dbm=5.0,
    )

    # -- 1-AP Grid Search ------------------------------------------------------
    gs_1ap_info = run_grid_search(scene_1ap, grid_resolution=5.0, verbose=True)

    # -- 1-AP Gradient Descent -------------------------------------------------
    gd_1ap_info, gd_1ap_opt = run_gradient_descent(
        scene_1ap, num_aps=1, verbose=True,
    )

    # ==================================================================
    # 2-AP RUNS (scene with 2 transmitters)
    # ==================================================================
    print("\n\nSetting up 2-AP scene...")
    scene_2ap = setup_building_floor_scene(
        scene_path=str(SCENE_PATH),
        frequency=5.18e9,
        tx_power_dbm=5.0,
        tx_positions=[
            (INITIAL_AP_POSITIONS_2AP[0][0], INITIAL_AP_POSITIONS_2AP[0][1], FIXED_Z),
            (INITIAL_AP_POSITIONS_2AP[1][0], INITIAL_AP_POSITIONS_2AP[1][1], FIXED_Z),
        ],
    )

    # -- 2-AP Grid Search (alternating) ----------------------------------------
    gs_2ap_info = run_grid_search_2ap(
        scene_2ap, grid_resolution=5.0, num_rounds=ALTERNATING_ROUNDS, verbose=True,
    )

    # -- 2-AP Gradient Descent -------------------------------------------------
    gd_2ap_info, gd_2ap_opt = run_gradient_descent(
        scene_2ap, num_aps=2, verbose=True,
    )

    # ==================================================================
    # COMPARISON TABLE
    # ==================================================================
    print("\n" + "=" * 80)
    print("FULL COMPARISON — 1 AP vs 2 APs × Grid Search vs Gradient Descent")
    print("=" * 80)

    configs = [
        ("GS  1-AP", gs_1ap_info),
        ("GD  1-AP", gd_1ap_info),
        ("GS  2-AP", gs_2ap_info),
        ("GD  2-AP", gd_2ap_info),
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

    # Quick summary line
    print("\n" + "-" * 80)
    print("  Summary (Min RSS in dBm):")
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

    # -- Plot results ----------------------------------------------------------
    for name, opt in [("1-AP", gd_1ap_opt), ("2-AP", gd_2ap_opt)]:
        try:
            opt.plot_optimization_trajectory()
        except Exception as e:
            print(f"\nNote: Could not plot {name} GD results (requires display): {e}")


if __name__ == "__main__":
    main()
