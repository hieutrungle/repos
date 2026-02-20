"""
Example: Full comparison of grid search vs gradient descent (single-process).

This script uses the same ``OptimizerFactory`` workflow as the Ray parallel
workers but runs everything sequentially in a single process.  Both methods
now report position **and** orientation (direction / look_at).

Workflow per method:
    1. ``OptimizerFactory.create(method, scene, **kwargs)``
    2. ``optimizer.optimize(**params)``
    3. Unpack result:
       - GD returns ``(position, metric)`` with orientation in ``history``
       - Grid search point returns ``(position, orientation, metric)``

Grid search iterates over all points via ``generate_grid_positions()`` and
evaluates each with ``SinglePointGridSearchOptimizer`` (8-direction sweep),
mirroring the Ray parallel pattern.

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
from reflector_position.optimizers.grid_search import generate_grid_positions
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

def run_gradient_descent(scene, verbose: bool = True):
    """Run gradient descent via OptimizerFactory (single trajectory)."""
    print("\n" + "=" * 80)
    print("GRADIENT DESCENT OPTIMIZATION (with orientation)")
    print("=" * 80)

    optimizer = OptimizerFactory.create(
        method="gradient_descent",
        scene=scene,
        initial_position=(15.0, 15.0),
        position_bounds=POSITION_BOUNDS,
        fixed_z=FIXED_Z,
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
    Run grid search with 8-direction sweep via OptimizerFactory.

    Each grid point is evaluated using ``SinglePointGridSearchOptimizer``
    (the same class used by the Ray worker), which sweeps 8 orientations
    per point and picks the best.
    """
    print("\n" + "=" * 80)
    print("GRID SEARCH OPTIMIZATION (8-direction sweep per point)")
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
        print(f"  Grid Search Complete! ({total_time:.1f}s)")

    return best_overall


# -- Comparison and reporting --------------------------------------------------

def _fmt_dir(d):
    """Format a direction vector for printing."""
    if d is None:
        return "N/A"
    return f"({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f})"


def _fmt_pos(p):
    """Format a position for printing."""
    if p is None:
        return "N/A"
    return f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"


def main():
    # -- Setup scene -----------------------------------------------------------
    print("Setting up scene...")
    scene = setup_building_floor_scene(
        scene_path=str(SCENE_PATH),
        frequency=5.18e9,
        tx_power_dbm=5.0,
    )

    # -- Run Grid Search -------------------------------------------------------
    gs_info = run_grid_search(scene, grid_resolution=5.0, verbose=True)

    # -- Run Gradient Descent --------------------------------------------------
    gd_info, gd_optimizer = run_gradient_descent(scene, verbose=True)

    # -- Comparison ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(f"\nGrid Search (8-dir sweep per point):")
    print(f"  Best position:  {_fmt_pos(gs_info['best_position'])}")
    print(f"  Best direction: {_fmt_dir(gs_info['best_direction'])}")
    print(f"  Best look_at:   {_fmt_pos(gs_info['best_look_at'])}")
    print(f"  Min RSS:        {gs_info['best_metric_dbm']:.2f} dBm")
    print(f"  Evaluations:    {gs_info['num_evaluations']} points × 8 dirs")
    print(f"  Time:           {gs_info['time_elapsed']:.1f}s")

    print(f"\nGradient Descent (position + orientation):")
    print(f"  Best position:  {_fmt_pos(gd_info['best_position'])}")
    print(f"  Best direction: {_fmt_dir(gd_info['best_direction'])}")
    print(f"  Best look_at:   {_fmt_pos(gd_info['best_look_at'])}")
    print(f"  Final position: {_fmt_pos(gd_info['final_position'])}")
    print(f"  Final direction:{_fmt_dir(gd_info['final_direction'])}")
    print(f"  Final look_at:  {_fmt_pos(gd_info['final_look_at'])}")
    print(f"  Best iteration: {gd_info['best_iteration'] + 1}/{gd_info['num_iterations']}")
    print(f"  Min RSS:        {gd_info['best_metric_dbm']:.2f} dBm")
    print(f"  Iterations:     {gd_info['num_iterations']}")
    print(f"  Time:           {gd_info['time_elapsed']:.1f}s")

    gs_evals = gs_info["num_evaluations"] * 8
    gd_evals = gd_info["num_iterations"]
    print(f"\nEvaluation count: GS={gs_evals}, GD={gd_evals}")
    if gd_evals > 0:
        print(f"Efficiency: GD uses {gs_evals / gd_evals:.1f}× fewer evaluations")

    # -- Save results ----------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {
        "grid_search": {k: v for k, v in gs_info.items() if k != "all_point_results"},
        "gradient_descent": gd_info,
    }
    out_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # -- Plot results ----------------------------------------------------------
    try:
        gd_optimizer.plot_optimization_trajectory()
    except Exception as e:
        print(f"\nNote: Could not plot results (requires display): {e}")


if __name__ == "__main__":
    main()
