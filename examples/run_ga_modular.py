"""
Entry point: Run Genetic Algorithm with modular evaluation.

Supports two execution modes via ``--mode``:

``local`` (default)
    Single-process evaluation.  Loads the Scene once and evaluates all
    individuals sequentially.  No Ray dependency — ideal for debugging,
    profiling, and quick smoke tests.

``ray``
    Distributed evaluation via ``RayActorPoolExecutor``.  Spawns a pool
    of persistent actors, each holding its own Scene, and distributes
    fitness evaluations across workers automatically.

In both modes the ``GeneticAlgorithmRunner`` (pure DEAP, no Ray imports)
is identical — only the injected ``executor_map`` changes.

Usage::

    # Quick local test (small pop, few generations)
    python examples/run_ga_modular.py --mode local

    # Full distributed run
    python examples/run_ga_modular.py --mode ray

    # Custom parameters
    python examples/run_ga_modular.py --mode local --pop-size 20 --n-gen 5
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

from reflector_position.optimizers.deap_logic import GeneticAlgorithmRunner


# ===========================================================================
# Configuration (matches ray_parallel_example.py Example 3)
# ===========================================================================

SCENE_PATH = (
    Path.home() / "blender" / "models" / "building_floor" / "building_floor.xml"
)

SCENE_CONFIG_1AP = {
    "scene_path": str(SCENE_PATH),
    "frequency": 5.18e9,
    "tx_power_dbm": 5.0,
}

FIXED_Z = 3.8
INITIAL_AP_POSITIONS_2AP = [(7.0, 7.0), (23.0, 23.0)]

SCENE_CONFIG_2AP = {
    **SCENE_CONFIG_1AP,
    "tx_positions": [
        (pos[0], pos[1], FIXED_Z) for pos in INITIAL_AP_POSITIONS_2AP
    ],
}

MIN_AP_SEPARATION = 5.0

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
    "mut_sigma_pos": 2.0,
    "mut_sigma_dir": 0.3,
    "mut_indpb": 0.2,
    "hof_size": 5,
}

OPTIMIZATION_PARAMS = {
    "samples_per_tx": 1_000_000,
    "max_depth": 13,
    "verbose": False,
}


# ===========================================================================
# Local (single-process) executor
# ===========================================================================


class LocalExecutor:
    """
    Single-process executor that mimics the ``RayActorPoolExecutor`` interface.

    Loads the Scene once, then evaluates individuals sequentially using
    ``OptimizerFactory``.  Compatible with DEAP's ``toolbox.register("map", ...)``
    pattern via the :meth:`map` method.
    """

    def __init__(
        self,
        scene_config: Dict[str, Any],
        verbose: bool = True,
    ):
        from reflector_position.scene_setup import setup_building_floor_scene

        if verbose:
            print("  Loading scene (single-process mode) ...")

        self.scene = setup_building_floor_scene(
            scene_path=str(scene_config["scene_path"]),
            frequency=scene_config.get("frequency", 5.18e9),
            tx_positions=scene_config.get("tx_positions", None),
            tx_power_dbm=scene_config.get("tx_power_dbm", 5.0),
            rx_position=scene_config.get("rx_position", (16.0, 6.5, 1.5)),
        )
        if verbose:
            print("  Scene loaded.")

    def map(
        self,
        func: Callable,
        iterable: Iterable,
    ) -> List[Dict[str, Any]]:
        """Evaluate items sequentially — drop-in replacement for Ray pool.map."""
        from reflector_position.optimizers.optimizer_factory import OptimizerFactory

        import numpy as np

        items = list(iterable)
        if not items:
            return []

        task_args = [func(item) for item in items]
        results: List[Dict[str, Any]] = []

        for task_id, method, kwargs, opt_params in task_args:
            optimizer = OptimizerFactory.create(
                method=method,
                scene=self.scene,
                **kwargs,
            )
            start = time.time()
            result_tuple = optimizer.optimize(**opt_params)
            elapsed = time.time() - start

            # Unpack — matches OptimizationWorker.optimize() output format.
            result_orientation = None
            if isinstance(result_tuple, tuple) and len(result_tuple) == 3:
                final_position, result_orientation, final_metric = result_tuple
            elif isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                final_position, final_metric = result_tuple
            else:
                final_position = None
                final_metric = float("-inf")

            best_position = (
                np.asarray(final_position).tolist()
                if final_position is not None
                else [0.0, 0.0, 0.0]
            )
            best_direction = (
                np.asarray(result_orientation).tolist()
                if result_orientation is not None
                else None
            )
            metric_linear = float(final_metric) if final_metric is not None else 0.0

            results.append({
                "task_id": task_id,
                "worker_id": 0,
                "best_position": best_position,
                "best_metric": metric_linear,
                "best_direction": best_direction,
                "time_elapsed": elapsed,
                "grid_results": (
                    {k: v for k, v in optimizer.results.items() if k != "radio_maps"}
                    if hasattr(optimizer, "results")
                    else {}
                ),
            })

        return results

    def shutdown(self) -> None:
        """No-op for single-process mode."""
        print("  Local executor shut down (no-op).")


# ===========================================================================
# Utilities
# ===========================================================================


def _save_ga_results(results: dict, path: str) -> None:
    """Save GA results dict to JSON."""
    serializable = {
        "best_individual": results["best_individual"],
        "best_fitness": results["best_fitness"],
        "best_fitness_dbm": results["best_fitness_dbm"],
        "optimize_orientation": results.get("optimize_orientation"),
        "num_aps": results.get("num_aps", 1),
        "hall_of_fame": results["hall_of_fame"],
        "total_time": results["total_time"],
        "total_evaluations": results["total_evaluations"],
        "ga_params": results["ga_params"],
        "generation_details": results["generation_details"],
    }
    # 1-AP fields
    if "best_position" in results:
        serializable["best_position"] = results["best_position"]
        serializable["best_direction"] = results.get("best_direction")
    # Multi-AP fields
    if "best_positions" in results:
        serializable["best_positions"] = results["best_positions"]
        serializable["best_directions"] = results.get("best_directions")
        serializable["best_ap_separation"] = results.get("best_ap_separation")
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"GA results saved to: {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GA for AP position & orientation optimization.",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "ray"],
        default="local",
        help="Execution mode: 'local' (single-process) or 'ray' (distributed). "
             "Default: local.",
    )
    parser.add_argument("--pop-size", type=int, default=None, help="Population size.")
    parser.add_argument("--n-gen", type=int, default=None, help="Number of generations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no-orientation",
        action="store_true",
        help="Disable orientation optimization (legacy 2-gene mode).",
    )
    parser.add_argument(
        "--num-aps",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of Access Points to optimise (1 or 2). Default: 1.",
    )
    parser.add_argument(
        "--min-sep",
        type=float,
        default=None,
        help="Minimum AP separation in metres (only for num_aps>=2). "
             "Default: 2.0.",
    )
    return parser.parse_args()


# ===========================================================================
# Main
# ===========================================================================


if __name__ == "__main__":
    args = _parse_args()

    # Override GA params from CLI if provided
    ga_params = dict(GA_PARAMS)
    if args.pop_size is not None:
        ga_params["pop_size"] = args.pop_size
    if args.n_gen is not None:
        ga_params["n_gen"] = args.n_gen

    optimize_orientation = not args.no_orientation
    num_aps = args.num_aps
    min_ap_separation = args.min_sep if args.min_sep is not None else MIN_AP_SEPARATION

    # Select scene config based on num_aps
    scene_config = SCENE_CONFIG_2AP if num_aps >= 2 else SCENE_CONFIG_1AP

    print("=" * 80)
    mode_label = "LOCAL (single-process)" if args.mode == "local" else "RAY (distributed)"
    if num_aps == 1:
        orient_label = "4D [x,y,dx,dy]" if optimize_orientation else "2D [x,y]"
    else:
        orient_label = (
            f"{num_aps * 4}D [{num_aps}AP orient]" if optimize_orientation
            else f"{num_aps * 2}D [{num_aps}AP pos-only]"
        )
    print(f"MODULAR GA: {mode_label} | Chromosome: {orient_label} | APs: {num_aps}")
    print("=" * 80)

    # -- Create executor ------------------------------------------------
    executor: Any  # LocalExecutor or RayActorPoolExecutor
    if args.mode == "local":
        executor = LocalExecutor(
            scene_config=scene_config,
            verbose=True,
        )
    else:
        import ray
        from reflector_position.optimizers.ray_evaluator import RayActorPoolExecutor

        ray.init(ignore_reinit_error=True)
        executor = RayActorPoolExecutor(
            scene_config=scene_config,
            num_workers=NUM_POOL_WORKERS,
            gpu_fraction=GPU_FRACTION,
            verbose=True,
        )

    # -- Create GA runner -----------------------------------------------
    ga = GeneticAlgorithmRunner(
        position_bounds=POSITION_BOUNDS,
        fixed_z=FIXED_Z,
        executor_map=executor.map,
        optimize_orientation=optimize_orientation,
        num_aps=num_aps,
        min_ap_separation=min_ap_separation,
    )

    try:
        # -- Run GA -----------------------------------------------------
        results = ga.run(
            optimization_params=OPTIMIZATION_PARAMS,
            ga_params=ga_params,
            seed=args.seed,
            verbose=True,
        )

        # -- Save results -----------------------------------------------
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        suffix = "local" if args.mode == "local" else "ray"
        _save_ga_results(
            results,
            os.path.join(OUTPUT_DIR, f"ga_modular_{suffix}_results.json"),
        )

        # -- Save evolution plot ----------------------------------------
        ga.save_evolution_plot(
            results,
            save_path=os.path.join(OUTPUT_DIR, f"ga_modular_{suffix}_evolution.png"),
            position_bounds=POSITION_BOUNDS,
            rss_range_dbm=RSS_RANGE_DBM,
        )

        # -- Summary ----------------------------------------------------
        print(f"\nGenetic Algorithm ({ga_params['pop_size']} pop, "
              f"{ga_params['n_gen']} gen, mode={args.mode}, "
              f"APs={num_aps}):")
        if num_aps == 1:
            print(f"  Best position:  {results['best_position']}")
            bd = results.get("best_direction")
            if bd:
                print(f"  Best direction: ({bd[0]:+.4f}, {bd[1]:+.4f}, {bd[2]:+.4f})")
        else:
            for ap_idx in range(num_aps):
                pos = results['best_positions'][ap_idx]
                print(f"  AP{ap_idx} position:  ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                bd = results.get('best_directions')
                if bd and bd[ap_idx]:
                    d = bd[ap_idx]
                    print(f"  AP{ap_idx} direction: ({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f})")
            print(f"  AP separation:  {results.get('best_ap_separation', 0):.2f}m")
        print(f"  Best Min RSS:   {results['best_fitness_dbm']:.2f} dBm")
        print(f"  Total evals:    {results['total_evaluations']}")
        print(f"  Wall-clock:     {results['total_time']:.2f}s")

    finally:
        executor.shutdown()
        if args.mode == "ray":
            ray.shutdown()
