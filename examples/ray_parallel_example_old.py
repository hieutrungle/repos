"""
Example: Distributed parallel optimization using Ray.

Demonstrates how to use RayParallelOptimizer to run multiple independent
optimization trajectories in parallel for both gradient descent and grid search.

The scene setup follows the same pattern as full_comparison.py and
optimizer_factory_example.py, using setup_building_floor_scene() to properly
configure transmitters, receivers, and antenna arrays.

Results are saved to files instead of displayed interactively.

Usage:
    python examples/ray_parallel_example.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import ray

from reflector_position.optimizers.ray_parallel_optimizer import (
    RayParallelOptimizer,
    generate_random_initial_positions,
)

# ── Configuration ─────────────────────────────────────────────────────────────

# Scene path — same as full_comparison.py and optimizer_factory_example.py
SCENE_PATH = Path.home() / "blender" / "models" / "building_floor" / "building_floor.xml"

# Scene configuration matching setup_building_floor_scene() defaults
SCENE_CONFIG = {
    "scene_path": str(SCENE_PATH),
    "frequency": 5.18e9,
    "tx_power_dbm": 5.0,
    # tx_positions and rx_position use defaults from setup_building_floor_scene
}

# Search space bounds (same as optimizer_factory_example.py)
POSITION_BOUNDS = {
    "x_min": 5.0,
    "x_max": 25.0,
    "y_min": 5.0,
    "y_max": 25.0,
}

OUTPUT_DIR = "results/ray_parallel"


# ── Example 1: Parallel Gradient Descent ──────────────────────────────────────

def example_parallel_gradient_descent():
    """
    Run multiple gradient descent optimizations in parallel.

    Each worker starts from a different random position and optimizes
    independently. The best result is selected at the end.

    This follows the same optimizer initialization pattern as
    optimizer_factory_example.py:
        OptimizerFactory.create(
            method="gradient_descent",
            scene=scene,
            initial_position=(x, y),
            position_bounds={...},
        )
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Parallel Gradient Descent")
    print("=" * 80)

    NUM_WORKERS = 8
    GPU_FRACTION = 0.25  # 4 workers per GPU

    # Generate diverse starting positions
    initial_positions = generate_random_initial_positions(
        num_positions=NUM_WORKERS,
        bounds=POSITION_BOUNDS,
        fixed_z=3.8,
        seed=42,
    )

    print(f"\nGenerated {NUM_WORKERS} initial positions:")
    for i, pos in enumerate(initial_positions):
        print(f"  Worker {i}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    # Build per-worker optimizer kwargs
    # These match the kwargs you'd pass to OptimizerFactory.create()
    worker_optimizer_kwargs = [
        {
            "initial_position": (float(pos[0]), float(pos[1])),
            "position_bounds": POSITION_BOUNDS,
            "fixed_z": 3.8,
        }
        for pos in initial_positions
    ]

    # Optimization parameters — passed to optimizer.optimize()
    # Same parameters as in full_comparison.py
    optimization_params = {
        "num_iterations": 10,
        "learning_rate": 0.5,
        "samples_per_tx": 500_000,
        "max_depth": 13,
        "use_soft_min": True,
        "temperature": 0.2,
        "verbose": False,
    }

    # Create orchestrator and run
    parallel_opt = RayParallelOptimizer(
        num_workers=NUM_WORKERS,
        gpu_fraction=GPU_FRACTION,
    )

    results = parallel_opt.run(
        scene_config=SCENE_CONFIG,
        optimizer_method="gradient_descent",
        worker_optimizer_kwargs=worker_optimizer_kwargs,
        optimization_params=optimization_params,
        verbose=True,
    )

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _save_results(results, os.path.join(OUTPUT_DIR, "gd_results.json"))

    # Save plot
    parallel_opt.save_results_plot(
        results,
        save_path=os.path.join(OUTPUT_DIR, "gd_parallel_results.png"),
        metric_name="Min RSS",
    )

    return results


# ── Example 2: Parallel Grid Search ──────────────────────────────────────────

def example_parallel_grid_search():
    """
    Run grid search optimizations in parallel, each covering a different
    spatial quadrant.

    Each worker receives different search_bounds, splitting the full search
    space into quadrants. This follows the same optimizer initialization
    pattern as full_comparison.py:
        GridSearchAPOptimizer(
            scene=scene,
            search_bounds={...},
            grid_resolution=2.0,
            fixed_z=3.8,
        )
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Parallel Grid Search (quadrant split)")
    print("=" * 80)

    NUM_WORKERS = 4
    GPU_FRACTION = 0.25

    # Divide the search space into 4 quadrants
    x_mid = (POSITION_BOUNDS["x_min"] + POSITION_BOUNDS["x_max"]) / 2
    y_mid = (POSITION_BOUNDS["y_min"] + POSITION_BOUNDS["y_max"]) / 2

    quadrants = [
        {"x_min": POSITION_BOUNDS["x_min"], "x_max": x_mid,
         "y_min": POSITION_BOUNDS["y_min"], "y_max": y_mid},    # SW
        {"x_min": x_mid,                    "x_max": POSITION_BOUNDS["x_max"],
         "y_min": POSITION_BOUNDS["y_min"], "y_max": y_mid},    # SE
        {"x_min": POSITION_BOUNDS["x_min"], "x_max": x_mid,
         "y_min": y_mid,                    "y_max": POSITION_BOUNDS["y_max"]},  # NW
        {"x_min": x_mid,                    "x_max": POSITION_BOUNDS["x_max"],
         "y_min": y_mid,                    "y_max": POSITION_BOUNDS["y_max"]},  # NE
    ]

    print(f"\nDividing space into {NUM_WORKERS} quadrants:")
    for i, q in enumerate(quadrants):
        print(f"  Worker {i}: x=[{q['x_min']:.1f}, {q['x_max']:.1f}], "
              f"y=[{q['y_min']:.1f}, {q['y_max']:.1f}]")

    # Build per-worker optimizer kwargs for grid search
    # These match the kwargs you'd pass to OptimizerFactory.create()
    worker_optimizer_kwargs = [
        {
            "search_bounds": q,
            "grid_resolution": 5.0,
            "fixed_z": 3.8,
        }
        for q in quadrants
    ]

    # Optimization parameters — passed to optimizer.optimize()
    optimization_params = {
        "samples_per_tx": 500_000,
        "max_depth": 13,
        "verbose": False,
    }

    # Create orchestrator and run
    parallel_opt = RayParallelOptimizer(
        num_workers=NUM_WORKERS,
        gpu_fraction=GPU_FRACTION,
    )

    results = parallel_opt.run(
        scene_config=SCENE_CONFIG,
        optimizer_method="grid_search",
        worker_optimizer_kwargs=worker_optimizer_kwargs,
        optimization_params=optimization_params,
        verbose=True,
    )

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _save_results(results, os.path.join(OUTPUT_DIR, "gs_results.json"))

    # Save plot
    parallel_opt.save_results_plot(
        results,
        save_path=os.path.join(OUTPUT_DIR, "gs_parallel_results.png"),
        metric_name="Min RSS",
    )

    return results


# ── Utilities ─────────────────────────────────────────────────────────────────

def _save_results(results: dict, path: str) -> None:
    """Save results dict to JSON, converting numpy types."""
    serializable = {
        "best_worker_id": results["best_worker_id"],
        "total_time": results["total_time"],
        "aggregate_stats": results["aggregate_stats"],
        "best_result": {
            "worker_id": results["best_result"]["worker_id"],
            "best_position": results["best_result"]["best_position"],
            "best_metric": results["best_result"]["best_metric"],
            "best_metric_dbm": results["best_result"]["best_metric_dbm"],
            "time_elapsed": results["best_result"]["time_elapsed"],
        },
        "all_metrics_dbm": [r["best_metric_dbm"] for r in results["all_results"]],
        "all_positions": [r["best_position"] for r in results["all_results"]],
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Results saved to: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Initialize Ray once (all examples share the same cluster)
    ray.init(ignore_reinit_error=True)

    try:
        # Example 1: Parallel gradient descent
        gd_results = example_parallel_gradient_descent()

        # # Example 2: Parallel grid search
        # gs_results = example_parallel_grid_search()

        # ── Summary ───────────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        print(f"\nGradient Descent (parallel):")
        print(f"  Best position: {gd_results['best_result']['best_position']}")
        print(f"  Best Min RSS:  {gd_results['best_result']['best_metric_dbm']:.2f} dBm")
        print(f"  Wall-clock:    {gd_results['total_time']:.2f}s")
        # print(f"\nGrid Search (parallel):")
        # print(f"  Best position: {gs_results['best_result']['best_position']}")
        # print(f"  Best Min RSS:  {gs_results['best_result']['best_metric_dbm']:.2f} dBm")
        # print(f"  Wall-clock:    {gs_results['total_time']:.2f}s")

    finally:
        ray.shutdown()
