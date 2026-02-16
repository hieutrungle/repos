"""
Example: Distributed parallel optimization using Ray ActorPool.

Demonstrates how to use RayParallelOptimizer with the ActorPool pattern to
process many independent optimization tasks (e.g., 32) using a small fixed
pool of reusable workers (e.g., 6). Each worker loads the heavy Scene once
and reuses it for multiple tasks.

Key concepts demonstrated:
1. Decoupled task count: 32 tasks processed by 6 workers
2. Actor reuse: Scene loaded once per worker, not once per task
3. Pool reuse: Same pool shared between gradient descent and grid search
4. Automatic queuing: ActorPool distributes work to idle workers

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
from reflector_position.optimizers.grid_search import generate_grid_positions
from reflector_position.optimizers.ray_evaluator import RayActorPoolExecutor
from reflector_position.optimizers.deap_logic import GeneticAlgorithmRunner

# -- Configuration -------------------------------------------------------------

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

# Pool configuration — fixed pool size, decoupled from task count
NUM_POOL_WORKERS = 4        # Fixed pool size (actors loading Scene)
GPU_FRACTION = 0.25         # 4 workers per GPU

# Unified Min RSS scale (dBm) for all plots — enables visual comparison
# between GD and GS results. Set to None for auto-scaling.
RSS_RANGE_DBM = (-130.0, -90.0)  # (min_dbm, max_dbm)

# Shared ray-tracing parameters (used by GD, GS, and GA)
OPTIMIZATION_PARAMS = {
    "samples_per_tx": 1_000_000,
    "max_depth": 13,
    "verbose": False,
}

# GA-specific evolutionary parameters
GA_PARAMS = {
    "pop_size": 100,
    "n_gen": 20,
    "cxpb": 0.7,
    "mutpb": 0.3,
    "tournsize": 10,
    "cx_alpha": 0.5,
    "mut_mu": 0.0,
    "mut_sigma": 2.0,
    "mut_indpb": 0.2,
    "hof_size": 5,
}


# -- Example 1: Parallel Gradient Descent (many tasks, small pool) -------------

def example_parallel_gradient_descent(parallel_opt: RayParallelOptimizer):
    """
    Run 16 gradient descent trajectories using a pool of 4 reusable workers.

    Each task starts from a different random position and optimizes
    independently. The ActorPool automatically queues the 16 tasks across
    4 workers (~4 tasks/worker). The Scene is loaded only 4 times total.

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
    print("EXAMPLE 1: Parallel Gradient Descent (ActorPool)")
    print("=" * 80)

    NUM_TASKS = 80  # Tasks >> pool workers -> queuing

    # Generate diverse starting positions for all tasks
    initial_positions = generate_random_initial_positions(
        num_positions=NUM_TASKS,
        bounds=POSITION_BOUNDS,
        fixed_z=3.8,
        seed=42,
    )

    print(f"\nPool: {NUM_POOL_WORKERS} workers | Tasks: {NUM_TASKS}")
    print(f"~{NUM_TASKS / NUM_POOL_WORKERS:.0f} tasks per worker (auto-queued)")
    print(f"\nGenerated {NUM_TASKS} initial positions:")
    for i, pos in enumerate(initial_positions):
        print(f"  Task {i:2d}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    # Build per-task optimizer kwargs
    # These match the kwargs you'd pass to OptimizerFactory.create()
    work_items = [
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
        **OPTIMIZATION_PARAMS,
        "num_iterations": 30,
        "learning_rate": 0.5,
        "use_soft_min": True,
        "temperature": 0.2,
    }

    # Run all tasks through the pool
    results = parallel_opt.run(
        scene_config=SCENE_CONFIG,
        optimizer_method="gradient_descent",
        work_items=work_items,
        optimization_params=optimization_params,
        verbose=True,
    )

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _save_results(results, os.path.join(OUTPUT_DIR, "gd_results.json"))

    # Save overview plot
    parallel_opt.save_results_plot(
        results,
        save_path=os.path.join(OUTPUT_DIR, "gd_parallel_results.png"),
        metric_name="Min RSS",
        position_bounds=POSITION_BOUNDS,
        rss_range_dbm=RSS_RANGE_DBM,
    )

    # Save per-task trajectory plots for detailed analysis
    trajectory_dir = os.path.join(OUTPUT_DIR, "gd_trajectories")
    parallel_opt.save_task_trajectory_plots(
        results,
        save_dir=trajectory_dir,
        filename_prefix="gd_task",
        position_bounds=POSITION_BOUNDS,
        rss_range_dbm=RSS_RANGE_DBM,
    )

    return results


# -- Example 2: Parallel Grid Search (true parallel, one point per task) -------

def example_parallel_grid_search(parallel_opt: RayParallelOptimizer):
    """
    Run a true parallel grid search over the entire search space.

    All grid points are generated upfront with the specified resolution,
    and each point is submitted as an independent Ray task. The ActorPool
    distributes single-point evaluations across workers in parallel.

    This follows the pattern:
        1. generate_grid_positions() -> list of [x, y, z] positions
        2. Each position becomes a SinglePointGridSearchOptimizer task
        3. ActorPool evaluates all points in parallel
        4. Best position is selected from all evaluations
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Parallel Grid Search (true parallel, one point per task)")
    print("=" * 80)

    GRID_RESOLUTION = 1.0  # meters between grid points

    # Generate ALL grid positions upfront
    grid_positions = generate_grid_positions(
        search_bounds=POSITION_BOUNDS,
        grid_resolution=GRID_RESOLUTION,
        fixed_z=3.8,
    )
    num_tasks = len(grid_positions)

    print(f"\nGrid resolution: {GRID_RESOLUTION}m")
    print(f"Search space: x=[{POSITION_BOUNDS['x_min']}, {POSITION_BOUNDS['x_max']}], "
          f"y=[{POSITION_BOUNDS['y_min']}, {POSITION_BOUNDS['y_max']}]")
    print(f"Generated {num_tasks} grid points")
    print(f"Pool: {NUM_POOL_WORKERS} workers | Tasks: {num_tasks}")
    print(f"~{num_tasks / NUM_POOL_WORKERS:.0f} tasks per worker (auto-queued)")

    # Build per-task optimizer kwargs — each task evaluates ONE grid point
    work_items = [
        {
            "evaluation_position": (float(pos[0]), float(pos[1])),
            "fixed_z": 3.8,
        }
        for pos in grid_positions
    ]

    # Optimization parameters — passed to optimizer.optimize()
    optimization_params = OPTIMIZATION_PARAMS

    # Run — pool is reused from GD example (no Scene reload!)
    results = parallel_opt.run(
        scene_config=SCENE_CONFIG,
        optimizer_method="grid_search_point",
        work_items=work_items,
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
        position_bounds=POSITION_BOUNDS,
        rss_range_dbm=RSS_RANGE_DBM,
    )

    return results


# -- Example 3: DEAP Genetic Algorithm (modular IoC, Ray-parallel evaluation) --

def example_deap_ga(ga_runner: GeneticAlgorithmRunner):
    """
    Run a Genetic Algorithm using DEAP with Ray-parallel fitness evaluation.

    The GA logic (selection, crossover, mutation) runs on the driver via
    ``GeneticAlgorithmRunner`` (pure DEAP, no Ray imports).  Fitness
    evaluation is delegated to ``RayActorPoolExecutor.map()`` which
    distributes ``SinglePointGridSearchOptimizer`` tasks across the
    Ray ActorPool.

    Architecture (IoC pattern)::

        GeneticAlgorithmRunner  ── executor_map ──▶  RayActorPoolExecutor
        (algorithm logic)         (dependency          (Ray ActorPool)
                                   injection)
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: DEAP Genetic Algorithm (Modular IoC, Ray-parallel evaluation)")
    print("=" * 80)

    print(f"\nGA params: pop={GA_PARAMS['pop_size']}, gen={GA_PARAMS['n_gen']}")
    print(f"Pool: {NUM_POOL_WORKERS} workers | GPU fraction: {GPU_FRACTION}")

    results = ga_runner.run(
        optimization_params=OPTIMIZATION_PARAMS,
        ga_params=GA_PARAMS,
        seed=42,
        verbose=True,
    )

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _save_ga_results(results, os.path.join(OUTPUT_DIR, "ga_results.json"))

    # Save evolution plot
    ga_runner.save_evolution_plot(
        results,
        save_path=os.path.join(OUTPUT_DIR, "ga_evolution.png"),
        position_bounds=POSITION_BOUNDS,
        rss_range_dbm=RSS_RANGE_DBM,
    )

    return results


# -- Utilities -----------------------------------------------------------------

def _save_results(results: dict, path: str) -> None:
    """Save results dict to JSON, converting numpy types."""
    serializable = {
        "best_task_id": results["best_task_id"],
        "total_time": results["total_time"],
        "aggregate_stats": results["aggregate_stats"],
        "pool_info": results["pool_info"],
        "best_result": {
            "task_id": results["best_result"]["task_id"],
            "worker_id": results["best_result"]["worker_id"],
            "best_position": results["best_result"]["best_position"],
            "best_metric": results["best_result"]["best_metric"],
            "best_metric_dbm": results["best_result"]["best_metric_dbm"],
            "best_iteration": results["best_result"].get("best_iteration", -1),
            "final_position": results["best_result"].get("final_position"),
            "time_elapsed": results["best_result"]["time_elapsed"],
        },
        "all_metrics_dbm": [r["best_metric_dbm"] for r in results["all_results"]],
        "all_positions": [r["best_position"] for r in results["all_results"]],
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Results saved to: {path}")


def _save_ga_results(results: dict, path: str) -> None:
    """Save GA results dict to JSON."""
    serializable = {
        "best_individual": results["best_individual"],
        "best_fitness": results["best_fitness"],
        "best_fitness_dbm": results["best_fitness_dbm"],
        "best_position": results["best_position"],
        "hall_of_fame": results["hall_of_fame"],
        "total_time": results["total_time"],
        "total_evaluations": results["total_evaluations"],
        "ga_params": results["ga_params"],
        "generation_details": results["generation_details"],
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"GA results saved to: {path}")


# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    # Initialize Ray once (all examples share the same cluster)
    ray.init(ignore_reinit_error=True)
    
    # ---------------------------------------------------------
    # PART 1: Run Gradient Descent & Grid Search (Pool A)
    # ---------------------------------------------------------
    # Create a single orchestrator with a fixed pool size.
    # The pool is shared across both GD and GS examples — workers persist
    # and the Scene is loaded only once per worker (not once per task).
    parallel_opt = RayParallelOptimizer(
        num_workers=NUM_POOL_WORKERS,
        gpu_fraction=GPU_FRACTION,
    )
    
    try:
        # Run GD and GS
        gd_results = example_parallel_gradient_descent(parallel_opt)
        gs_results = example_parallel_grid_search(parallel_opt)

        print("\nGradient Descent & Grid Search Complete.")
        
    finally:
        # CRITICAL: Shut down Pool A to release GPUs/CPUs
        parallel_opt.shutdown()
        print("Parallel Optimizer pool shut down. Resources released.")
        
    # ---------------------------------------------------------
    # PART 2: Run Genetic Algorithm (Pool B)
    # ---------------------------------------------------------
    # Now that resources are free, we can create the GA pool

    # Modular GA: separate execution engine + algorithm runner (IoC pattern)
    ga_executor = RayActorPoolExecutor(
        scene_config=SCENE_CONFIG,
        num_workers=NUM_POOL_WORKERS,
        gpu_fraction=GPU_FRACTION,
        verbose=True,
    )
    ga_runner = GeneticAlgorithmRunner(
        position_bounds=POSITION_BOUNDS,
        fixed_z=3.8,
        executor_map=ga_executor.map,  # Dependency Injection
    )
    
    try:
        ga_results = example_deap_ga(ga_runner)
        
        # -- Summary -----------------------------------------------------------
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        
        print(f"\nPool: {NUM_POOL_WORKERS} workers (Scene loaded once per worker)")
        print(f"\nGradient Descent ({len(gd_results['all_results'])} tasks):")
        print(f"  Best task:     #{gd_results['best_task_id']}")
        print(f"  Best iteration: {gd_results['best_result'].get('best_iteration', -1) + 1}")
        print(f"  Best position: {gd_results['best_result']['best_position']}")
        print(f"  Best Min RSS:  {gd_results['best_result']['best_metric_dbm']:.2f} dBm")
        print(f"  Wall-clock:    {gd_results['total_time']:.2f}s")
        print(f"  Speedup:       {gd_results['aggregate_stats']['speedup']:.2f}x")
        print(f"\nGrid Search ({len(gs_results['all_results'])} grid points):")
        print(f"  Best task:     #{gs_results['best_task_id']}")
        print(f"  Best position: {gs_results['best_result']['best_position']}")
        print(f"  Best Min RSS:  {gs_results['best_result']['best_metric_dbm']:.2f} dBm")
        print(f"  Wall-clock:    {gs_results['total_time']:.2f}s")
        print(f"  Speedup:       {gs_results['aggregate_stats']['speedup']:.2f}x")
        print(f"\nGenetic Algorithm ({ga_results['ga_params']['pop_size']} pop, "
              f"{ga_results['ga_params']['n_gen']} gen):")
        print(f"  Best position: {ga_results['best_position']}")
        print(f"  Best Min RSS:  {ga_results['best_fitness_dbm']:.2f} dBm")
        print(f"  Total evals:   {ga_results['total_evaluations']}")
        print(f"  Wall-clock:    {ga_results['total_time']:.2f}s")
    finally:
        # Shut down Pool B
        ga_executor.shutdown()
        ray.shutdown()

    # try:
    #     # Example 1: 64 GD trajectories -> 4-worker pool
    #     gd_results = example_parallel_gradient_descent(parallel_opt)

    #     # Example 2: parallel grid search -> same 4-worker pool (reused!)
    #     gs_results = example_parallel_grid_search(parallel_opt)

    #     # Example 3: DEAP Genetic Algorithm -> modular IoC (separate pool)
    #     ga_results = example_deap_ga(ga_runner)

    #     # -- Summary -----------------------------------------------------------
    #     print("\n" + "=" * 80)
    #     print("OVERALL SUMMARY")
    #     print("=" * 80)
    #     print(f"\nPool: {NUM_POOL_WORKERS} workers (Scene loaded once per worker)")
    #     print(f"\nGradient Descent ({len(gd_results['all_results'])} tasks):")
    #     print(f"  Best task:     #{gd_results['best_task_id']}")
    #     print(f"  Best iteration: {gd_results['best_result'].get('best_iteration', -1) + 1}")
    #     print(f"  Best position: {gd_results['best_result']['best_position']}")
    #     print(f"  Best Min RSS:  {gd_results['best_result']['best_metric_dbm']:.2f} dBm")
    #     print(f"  Wall-clock:    {gd_results['total_time']:.2f}s")
    #     print(f"  Speedup:       {gd_results['aggregate_stats']['speedup']:.2f}x")
    #     print(f"\nGrid Search ({len(gs_results['all_results'])} grid points):")
    #     print(f"  Best task:     #{gs_results['best_task_id']}")
    #     print(f"  Best position: {gs_results['best_result']['best_position']}")
    #     print(f"  Best Min RSS:  {gs_results['best_result']['best_metric_dbm']:.2f} dBm")
    #     print(f"  Wall-clock:    {gs_results['total_time']:.2f}s")
    #     print(f"  Speedup:       {gs_results['aggregate_stats']['speedup']:.2f}x")
    #     print(f"\nGenetic Algorithm ({ga_results['ga_params']['pop_size']} pop, "
    #           f"{ga_results['ga_params']['n_gen']} gen):")
    #     print(f"  Best position: {ga_results['best_position']}")
    #     print(f"  Best Min RSS:  {ga_results['best_fitness_dbm']:.2f} dBm")
    #     print(f"  Total evals:   {ga_results['total_evaluations']}")
    #     print(f"  Wall-clock:    {ga_results['total_time']:.2f}s")

    # finally:
    #     # Explicitly kill pool actors and shutdown Ray
    #     parallel_opt.shutdown()
    #     ga_executor.shutdown()
    #     ray.shutdown()
