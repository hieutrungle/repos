"""
Entry point: Run Genetic Algorithm with modular Ray evaluation.

Wires together:
    - ``RayActorPoolExecutor`` (execution engine — manages Ray ActorPool)
    - ``GeneticAlgorithmRunner``  (algorithm logic — pure DEAP, no Ray imports)

The executor's ``map`` method is injected into the GA runner via DEAP's
``toolbox.register("map", executor.map)``.  This Inversion of Control (IoC)
pattern strictly separates the execution engine from the algorithm logic,
preventing freeze issues caused by ``map_unordered`` and resource contention.

Usage::

    python examples/run_ga_modular.py
"""

import json
import os
from pathlib import Path

import ray

from reflector_position.optimizers.ray_evaluator import RayActorPoolExecutor
from reflector_position.optimizers.deap_logic import GeneticAlgorithmRunner

# ===========================================================================
# Configuration (matches ray_parallel_example.py Example 3)
# ===========================================================================

SCENE_PATH = (
    Path.home() / "blender" / "models" / "building_floor" / "building_floor.xml"
)

SCENE_CONFIG = {
    "scene_path": str(SCENE_PATH),
    "frequency": 5.18e9,
    "tx_power_dbm": 5.0,
    # tx_positions and rx_position use defaults from setup_building_floor_scene
}

POSITION_BOUNDS = {
    "x_min": 5.0,
    "x_max": 25.0,
    "y_min": 5.0,
    "y_max": 25.0,
}

OUTPUT_DIR = "results/ray_parallel"

NUM_POOL_WORKERS = 4
GPU_FRACTION = 0.25

RSS_RANGE_DBM = (-120.0, -90.0)

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

OPTIMIZATION_PARAMS = {
    "samples_per_tx": 1_000_000,
    "max_depth": 13,
    "verbose": False,
}


# ===========================================================================
# Utilities
# ===========================================================================


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


# ===========================================================================
# Main
# ===========================================================================


if __name__ == "__main__":
    # Initialize Ray once
    ray.init(ignore_reinit_error=True)

    print("=" * 80)
    print("MODULAR GA: RayActorPoolExecutor + GeneticAlgorithmRunner (IoC)")
    print("=" * 80)

    # 1. Create the execution engine (Ray ActorPool)
    executor = RayActorPoolExecutor(
        scene_config=SCENE_CONFIG,
        num_workers=NUM_POOL_WORKERS,
        gpu_fraction=GPU_FRACTION,
        verbose=True,
    )

    # 2. Create the algorithm runner, injecting executor.map
    ga = GeneticAlgorithmRunner(
        position_bounds=POSITION_BOUNDS,
        fixed_z=3.8,
        executor_map=executor.map,  # <--- Dependency Injection
    )

    try:
        # 3. Run the GA
        results = ga.run(
            optimization_params=OPTIMIZATION_PARAMS,
            ga_params=GA_PARAMS,
            seed=42,
            verbose=True,
        )

        # 4. Save results
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        _save_ga_results(results, os.path.join(OUTPUT_DIR, "ga_modular_results.json"))

        # 5. Save evolution plot
        ga.save_evolution_plot(
            results,
            save_path=os.path.join(OUTPUT_DIR, "ga_modular_evolution.png"),
            position_bounds=POSITION_BOUNDS,
            rss_range_dbm=RSS_RANGE_DBM,
        )

        # 6. Summary
        print(f"\nGenetic Algorithm ({GA_PARAMS['pop_size']} pop, "
              f"{GA_PARAMS['n_gen']} gen):")
        print(f"  Best position: {results['best_position']}")
        print(f"  Best Min RSS:  {results['best_fitness_dbm']:.2f} dBm")
        print(f"  Total evals:   {results['total_evaluations']}")
        print(f"  Wall-clock:    {results['total_time']:.2f}s")

    finally:
        # 7. Clean up
        executor.shutdown()
        ray.shutdown()
