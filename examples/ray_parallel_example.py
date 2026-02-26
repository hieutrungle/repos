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
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import ray

from reflector_position.optimizers.ray_parallel_optimizer import (
    RayParallelOptimizer,
    generate_random_initial_positions,
)
from reflector_position.optimizers.grid_search import (
    generate_grid_positions,
    generate_alternating_grid_tasks,
    generate_reflector_grid_tasks,
)
from reflector_position.optimizers.ray_evaluator import RayActorPoolExecutor
from reflector_position.optimizers.deap_logic import GeneticAlgorithmRunner

# -- Configuration -------------------------------------------------------------

# Scene path — same as full_comparison.py and optimizer_factory_example.py
SCENE_PATH = Path.home() / "blender" / "models" / "building_floor" / "building_floor.xml"

FIXED_Z = 3.8

# Scene configuration matching setup_building_floor_scene() defaults
SCENE_CONFIG = {
    "scene_path": str(SCENE_PATH),
    "frequency": 5.18e9,
    "tx_power_dbm": 5.0,
    # tx_positions and rx_position use defaults from setup_building_floor_scene
}

# 2-AP initial positions and scene config
INITIAL_AP_POSITIONS_2AP = [(7.0, 7.0), (23.0, 23.0)]

SCENE_CONFIG_2AP = {
    **SCENE_CONFIG,
    "tx_positions": [
        (pos[0], pos[1], FIXED_Z) for pos in INITIAL_AP_POSITIONS_2AP
    ],
}

# Search space bounds (same as optimizer_factory_example.py)
POSITION_BOUNDS = {
    "x_min": 5.5,
    "x_max": 34.5,
    "y_min": 5.5,
    "y_max": 34.5,
}

OUTPUT_DIR = "results/ray_parallel"

# Alternating optimisation rounds for 2-AP grid search
ALTERNATING_ROUNDS = 3

# Pool configuration — fixed pool size, decoupled from task count
NUM_POOL_WORKERS = 2        # Fixed pool size (actors loading Scene)
GPU_FRACTION = 0.5         # 4 workers per GPU

# Unified Min RSS scale (dBm) for all plots — enables visual comparison
# between GD and GS results. Set to None for auto-scaling.
RSS_RANGE_DBM = (-130.0, -80.0)  # (min_dbm, max_dbm)

# Shared ray-tracing parameters (used by GD, GS, and GA)
OPTIMIZATION_PARAMS = {
    "samples_per_tx": 1_000_000,
    "max_depth": 13,
    "verbose": False,
}

# GA-specific evolutionary parameters
GA_PARAMS = {
    "pop_size": 150,
    "n_gen": 50,
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

RANDOM_SEED = 4


# -- Example 1: Parallel Gradient Descent (many tasks, small pool) -------------

def example_parallel_gradient_descent(
    parallel_opt: RayParallelOptimizer,
    num_aps: int = 1,
    num_tasks: int = 6,
    num_iterations: int = 30,
    repulsion_weight: float = 1.0,
    samples_per_tx: int = 1_000_000,
    output_dir: str = OUTPUT_DIR,
    scene_config: dict = None,
    random_seed: int = RANDOM_SEED,
    optimization_overrides: Optional[dict] = None,
):
    """
    Deprecated: this AP-only gradient-descent runner is being phased out in
    favor of reflector-aware optimization flows.

    Run gradient descent trajectories using a pool of reusable workers.

    Supports single-AP (num_aps=1) and multi-AP (num_aps>=2) configurations.
    For multi-AP, each task starts from a different random set of AP positions
    and uses repulsion loss to prevent AP merging.
    """
    warnings.warn(
        "example_parallel_gradient_descent() is deprecated and will be removed "
        "in a future release; use reflector-aware optimization entrypoints instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if scene_config is None:
        scene_config = SCENE_CONFIG_2AP if num_aps >= 2 else SCENE_CONFIG
    print("\n" + "=" * 80)
    print(f"Parallel Gradient Descent — {num_aps} AP(s), {num_tasks} tasks")
    print("=" * 80)

    rng = np.random.default_rng(random_seed)

    if num_aps == 1:
        # Single-AP: one random starting position per task
        initial_positions = generate_random_initial_positions(
            num_positions=num_tasks,
            bounds=POSITION_BOUNDS,
            fixed_z=3.8,
            seed=random_seed,
        )
        print(f"\nPool: {parallel_opt.num_workers} workers | Tasks: {num_tasks}")
        print(f"\nGenerated {num_tasks} initial positions:")
        for i, pos in enumerate(initial_positions):
            print(f"  Task {i:2d}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

        work_items = [
            {
                "initial_position": (float(pos[0]), float(pos[1])),
                "position_bounds": POSITION_BOUNDS,
                "fixed_z": 3.8,
                "optimize_orientation": True,
            }
            for pos in initial_positions
        ]
    else:
        # Multi-AP: random position set per task
        print(f"\nPool: {parallel_opt.num_workers} workers | Tasks: {num_tasks}")
        print(f"Repulsion weight: {repulsion_weight}")
        print(f"\nGenerated {num_tasks} initial position sets ({num_aps} APs each):")

        work_items = []
        for i in range(num_tasks):
            positions = [
                (
                    float(rng.uniform(POSITION_BOUNDS["x_min"], POSITION_BOUNDS["x_max"])),
                    float(rng.uniform(POSITION_BOUNDS["y_min"], POSITION_BOUNDS["y_max"])),
                )
                for _ in range(num_aps)
            ]
            pos_str = "  ".join(f"({p[0]:.1f},{p[1]:.1f})" for p in positions)
            print(f"  Task {i:2d}: {pos_str}")
            work_items.append({
                "initial_positions": positions,
                "position_bounds": POSITION_BOUNDS,
                "fixed_z": 3.8,
                "optimize_orientation": True,
                "repulsion_weight": repulsion_weight,
            })

    # Optimization parameters — passed to optimizer.optimize()
    optimization_params = {
        **OPTIMIZATION_PARAMS,
        "samples_per_tx": samples_per_tx,
        "num_iterations": num_iterations,
        "learning_rate": 0.5,
        "use_soft_min": True,
        "temperature": 0.15,
        "alpha": 0.9,  # coverage loss weight
        "beta": 0.1,   # Sigmoid temperature for coverage loss.
    }
    if optimization_overrides:
        optimization_params.update(optimization_overrides)

    # Run all tasks through the pool
    results = parallel_opt.run(
        scene_config=scene_config,
        optimizer_method="gradient_descent",
        work_items=work_items,
        optimization_params=optimization_params,
        verbose=True,
    )

    # Save results
    ap_tag = f"{num_aps}ap"
    os.makedirs(output_dir, exist_ok=True)
    _save_results(results, os.path.join(output_dir, f"gd_{ap_tag}_results.json"))

    # Save overview plot
    parallel_opt.save_results_plot(
        results,
        save_path=os.path.join(output_dir, f"gd_{ap_tag}_parallel_results.png"),
        metric_name="Min RSS",
        position_bounds=POSITION_BOUNDS,
        rss_range_dbm=RSS_RANGE_DBM,
    )

    # Save per-task trajectory plots for detailed analysis
    trajectory_dir = os.path.join(output_dir, f"gd_{ap_tag}_trajectories")
    parallel_opt.save_task_trajectory_plots(
        results,
        save_dir=trajectory_dir,
        filename_prefix=f"gd_{ap_tag}_task",
        position_bounds=POSITION_BOUNDS,
        rss_range_dbm=RSS_RANGE_DBM,
    )

    return results


# -- Example 2: Parallel Grid Search (true parallel, one point per task) -------

def example_parallel_grid_search(
    parallel_opt: RayParallelOptimizer,
    grid_resolution: float = 0.5,
    output_dir: str = OUTPUT_DIR,
    scene_config: dict = None,
):
    """
    Run a true parallel 1-AP grid search over the entire search space.

    All grid points are generated upfront with the specified resolution,
    and each point is submitted as an independent Ray task. The ActorPool
    distributes single-point evaluations across workers in parallel.

    This follows the pattern:
        1. generate_grid_positions() -> list of [x, y, z] positions
        2. Each position becomes a SinglePointGridSearchOptimizer task
        3. ActorPool evaluates all points in parallel
        4. Best position is selected from all evaluations
    """
    if scene_config is None:
        scene_config = SCENE_CONFIG

    print("\n" + "=" * 80)
    print("Parallel Grid Search — 1 AP (true parallel, one point per task)")
    print("=" * 80)

    # Generate ALL grid positions upfront
    grid_positions = generate_grid_positions(
        search_bounds=POSITION_BOUNDS,
        grid_resolution=grid_resolution,
        fixed_z=FIXED_Z,
    )
    num_tasks = len(grid_positions)

    print(f"\nGrid resolution: {grid_resolution}m")
    print(f"Search space: x=[{POSITION_BOUNDS['x_min']}, {POSITION_BOUNDS['x_max']}], "
          f"y=[{POSITION_BOUNDS['y_min']}, {POSITION_BOUNDS['y_max']}]")
    print(f"Generated {num_tasks} grid points")
    print(f"Pool: {parallel_opt.num_workers} workers | Tasks: {num_tasks}")
    print(f"~{num_tasks / parallel_opt.num_workers:.0f} tasks per worker (auto-queued)")

    # Build per-task optimizer kwargs — each task evaluates ONE grid point
    work_items = [
        {
            "evaluation_position": (float(pos[0]), float(pos[1])),
            "fixed_z": FIXED_Z,
        }
        for pos in grid_positions
    ]

    # Optimization parameters — passed to optimizer.optimize()
    optimization_params = OPTIMIZATION_PARAMS

    # Run — pool is reused if scene_config matches
    results = parallel_opt.run(
        scene_config=scene_config,
        optimizer_method="grid_search_point",
        work_items=work_items,
        optimization_params=optimization_params,
        verbose=True,
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    _save_results(results, os.path.join(output_dir, "gs_1ap_results.json"))

    # Save plot
    parallel_opt.save_results_plot(
        results,
        save_path=os.path.join(output_dir, "gs_1ap_parallel_results.png"),
        metric_name="Min RSS",
        position_bounds=POSITION_BOUNDS,
        rss_range_dbm=RSS_RANGE_DBM,
    )

    return results


# -- Example 2b: Parallel Grid Search — 2 APs (alternating optimisation) ------

def example_parallel_grid_search_2ap(
    parallel_opt: RayParallelOptimizer,
    grid_resolution: float = 1.0,
    num_rounds: int = ALTERNATING_ROUNDS,
    initial_positions=None,
    output_dir: str = OUTPUT_DIR,
    scene_config: dict = None,
):
    """
    Run 2-AP grid search using alternating optimisation with Ray parallelism.

    In each round every AP is swept over the grid while the other is fixed.
    All grid points for one AP are submitted as parallel Ray tasks.

    Args:
        parallel_opt: RayParallelOptimizer instance.
        grid_resolution: Grid spacing in metres.
        num_rounds: Number of full alternation rounds (AP0→AP1 counts as 1).
        initial_positions: Starting ``[(x,y), (x,y)]`` for the two APs.
        output_dir: Directory for output files.
        scene_config: Scene config dict (must have 2 TX positions).

    Returns:
        Results dict from the final sweep (compatible with save_results_plot).
    """
    if initial_positions is None:
        initial_positions = list(INITIAL_AP_POSITIONS_2AP)
    if scene_config is None:
        scene_config = SCENE_CONFIG_2AP

    num_aps = len(initial_positions)

    print("\n" + "=" * 80)
    print(f"Parallel Grid Search — {num_aps} APs (alternating, {num_rounds} rounds)")
    print("=" * 80)
    print(f"  Grid resolution: {grid_resolution}m")
    print(f"  Alternating rounds: {num_rounds}")
    print(f"  Initial AP positions: {initial_positions}")

    current_positions = list(initial_positions)
    # None → sweep 8 cardinal directions for that AP
    current_orientations: list = [None] * num_aps

    best_overall_metric_dbm = float("-inf")
    best_overall_results = None
    total_evaluations = 0
    start_time = time.time()

    for round_idx in range(num_rounds):
        print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")

        for active_ap in range(num_aps):
            fixed_label = ", ".join(
                f"AP{k}=({current_positions[k][0]:.1f},{current_positions[k][1]:.1f})"
                for k in range(num_aps) if k != active_ap
            )
            print(f"\n  Sweeping AP{active_ap}  (fixed: {fixed_label})")

            # Generate work items for this sweep
            work_items = generate_alternating_grid_tasks(
                active_ap_idx=active_ap,
                search_bounds=POSITION_BOUNDS,
                fixed_positions=current_positions,
                fixed_orientations=current_orientations,
                grid_resolution=grid_resolution,
                fixed_z=FIXED_Z,
            )

            print(f"  {len(work_items)} grid points × orientations → parallel tasks")

            # Run in parallel — pool is reused across sweeps
            results = parallel_opt.run(
                scene_config=scene_config,
                optimizer_method="grid_search_point",
                work_items=work_items,
                optimization_params=OPTIMIZATION_PARAMS,
                verbose=False,
            )

            total_evaluations += len(work_items)

            # Extract best from this sweep
            sweep_best = results["best_result"]
            sweep_dbm = sweep_best["best_metric_dbm"]

            # Update active AP position and orientation from the best result
            sweep_pos = sweep_best["best_position"]
            sweep_dir = sweep_best.get("best_direction")

            if isinstance(sweep_pos[0], (list, np.ndarray)):
                # Multi-AP positions returned — update all
                current_positions = [tuple(p[:2]) for p in sweep_pos]
                if sweep_dir and isinstance(sweep_dir[0], (list, np.ndarray)):
                    current_orientations = [
                        tuple(d) if d is not None else None for d in sweep_dir
                    ]
            else:
                current_positions[active_ap] = tuple(sweep_pos[:2])
                if sweep_dir:
                    current_orientations[active_ap] = tuple(sweep_dir)

            print(f"  Best AP{active_ap}: ({current_positions[active_ap][0]:.1f}, "
                  f"{current_positions[active_ap][1]:.1f})  "
                  f"metric={sweep_dbm:.2f} dBm")

            # Track overall best
            if sweep_dbm > best_overall_metric_dbm:
                best_overall_metric_dbm = sweep_dbm
                best_overall_results = results

        # End-of-round summary
        print(f"\n  Round {round_idx + 1} complete:")
        for i in range(num_aps):
            print(f"    AP{i}: pos=({current_positions[i][0]:.1f}, "
                  f"{current_positions[i][1]:.1f})  "
                  f"orient={current_orientations[i]}")
        print(f"    Best metric so far: {best_overall_metric_dbm:.2f} dBm")

    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time:.1f}s | "
          f"Evaluations: {total_evaluations} | "
          f"Best: {best_overall_metric_dbm:.2f} dBm")

    # Save results and plot from the best-performing sweep
    os.makedirs(output_dir, exist_ok=True)
    _save_results(
        best_overall_results,
        os.path.join(output_dir, "gs_2ap_results.json"),
    )
    parallel_opt.save_results_plot(
        best_overall_results,
        save_path=os.path.join(output_dir, "gs_2ap_parallel_results.png"),
        metric_name="Min RSS",
        position_bounds=POSITION_BOUNDS,
        rss_range_dbm=RSS_RANGE_DBM,
    )

    return best_overall_results


# -- Example 2c: Parallel Grid Search — 2 APs + Reflector --------------------

def example_parallel_grid_search_2ap_with_reflector(
    parallel_opt: RayParallelOptimizer,
    grid_resolution: float = 1.0,
    num_rounds: int = ALTERNATING_ROUNDS,
    outer_rounds: int = 2,
    initial_positions=None,
    output_dir: str = OUTPUT_DIR,
    scene_config: dict = None,
    reflector_size: tuple[float, float] = (2.0, 2.0),
    wall_top_left: list[float] = [15.0, 34.0, 3.0],
    wall_bottom_right: list[float] = [34.0, 34.0, 1.0],
    u_steps: int = 3,
    v_steps: int = 3,
    target_bounds: Optional[dict] = None,
    target_resolution: float = 10.0,
    target_z: float = 1.5,
    target_quantile: float = 0.05,
    min_ap_separation: float = 10.0,
) -> dict:
    """Run physically-aware 2-AP + reflector grid search with Ray parallelism.

    Mirrors ``run_grid_search_2ap_with_reflector`` from full_comparison.py,
    but evaluates each AP-grid and reflector-grid candidate in parallel via
    ``RayParallelOptimizer``.

    Ranking is based on ``percentile_score`` (when available), with fallback
    to ``best_metric``.
    """
    if initial_positions is None:
        initial_positions = list(INITIAL_AP_POSITIONS_2AP)

    if target_bounds is None:
        target_bounds = {
            "x_min": 5.0,
            "x_max": 35.0,
            "y_min": 5.0,
            "y_max": 35.0,
        }

    base_scene_config = SCENE_CONFIG_2AP if scene_config is None else scene_config
    scene_config = {
        **base_scene_config,
        "reflector_enabled": True,
        "reflector_size": reflector_size,
        "wall_top_left": wall_top_left,
        "wall_bottom_right": wall_bottom_right,
        "focal_point": [20.0, 20.0, target_z],
        "device": "cpu",
    }

    print("\n" + "=" * 80)
    print("Parallel Grid Search — 2 APs + Reflector (physically-aware)")
    print("=" * 80)
    print(f"  AP grid resolution: {grid_resolution}m")
    print(f"  AP alternating rounds/outer round: {num_rounds}")
    print(f"  Outer rounds: {outer_rounds}")
    print(f"  Reflector grid: u_steps={u_steps}, v_steps={v_steps}, target_res={target_resolution}m")
    print(f"  Objective quantile: {target_quantile:.2f}")
    print(f"  Min AP separation: {min_ap_separation:.2f}m")

    current_positions = list(initial_positions)
    current_orientations: list = [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    current_reflector_u = 0.5
    current_reflector_v = 0.5
    current_reflector_target = (20.0, 20.0, target_z)

    best_overall_score = float("-inf")
    best_overall_results = None
    total_evaluations = 0
    start_time = time.time()

    def _coerce_percentile_metrics(run_results: dict) -> dict:
        """Promote percentile score into best_metric fields when present."""
        for row in run_results.get("all_results", []):
            grid_results = row.get("grid_results", {})
            if "percentile_score" in grid_results:
                row["best_metric"] = float(grid_results["percentile_score"])
                row["best_metric_dbm"] = float(grid_results.get("percentile_score_dbm", row["best_metric_dbm"]))

        if run_results.get("all_results"):
            best_idx = int(np.argmax([r["best_metric"] for r in run_results["all_results"]]))
            run_results["best_result"] = run_results["all_results"][best_idx]
            run_results["best_task_id"] = run_results["best_result"]["task_id"]
        return run_results

    for outer_idx in range(outer_rounds):
        print(f"\n--- Outer Round {outer_idx + 1}/{outer_rounds} ---")

        # Phase A: AP alternating sweep (reflector fixed)
        for round_idx in range(num_rounds):
            print(f"\n  AP Alternating Round {round_idx + 1}/{num_rounds}")
            for active_ap in range(2):
                # Sweep orientation for the active AP only; keep other AP fixed.
                sweep_orientations = list(current_orientations)
                sweep_orientations[active_ap] = None

                work_items = generate_alternating_grid_tasks(
                    active_ap_idx=active_ap,
                    search_bounds=POSITION_BOUNDS,
                    fixed_positions=current_positions,
                    fixed_orientations=sweep_orientations,
                    grid_resolution=grid_resolution,
                    fixed_z=FIXED_Z,
                    min_ap_separation=min_ap_separation,
                )

                for wi in work_items:
                    wi["reflector_u"] = float(current_reflector_u)
                    wi["reflector_v"] = float(current_reflector_v)
                    wi["reflector_target"] = tuple(float(c) for c in current_reflector_target)
                    wi["percentile_target_quantile"] = float(target_quantile)

                results = parallel_opt.run(
                    scene_config=scene_config,
                    optimizer_method="grid_search_point",
                    work_items=work_items,
                    optimization_params=OPTIMIZATION_PARAMS,
                    verbose=True,
                )
                results = _coerce_percentile_metrics(results)
                total_evaluations += len(work_items)

                sweep_best = results["best_result"]
                sweep_score_dbm = sweep_best["best_metric_dbm"]
                print(f"    AP{active_ap} sweep: {len(work_items)} tasks, best={sweep_score_dbm:.2f} dBm")

                sweep_pos = sweep_best["best_position"]
                if isinstance(sweep_pos[0], (list, np.ndarray)):
                    current_positions = [tuple(p[:2]) for p in sweep_pos]
                else:
                    current_positions[active_ap] = tuple(sweep_pos[:2])

                grid_results = sweep_best.get("grid_results", {})
                best_orients = grid_results.get("best_orientations")
                if best_orients and isinstance(best_orients, list):
                    current_orientations = [tuple(d) if d is not None else None for d in best_orients]

                if sweep_best["best_metric"] > best_overall_score:
                    best_overall_score = sweep_best["best_metric"]
                    best_overall_results = results

        # Phase B: reflector sweep (AP fixed)
        reflector_tasks = generate_reflector_grid_tasks(
            fixed_ap_positions=[(p[0], p[1], FIXED_Z) for p in current_positions],
            fixed_ap_orientations=[
                tuple(o) if o is not None else (0.0, 1.0, 0.0)
                for o in current_orientations
            ],
            u_steps=u_steps,
            v_steps=v_steps,
            target_bounds=target_bounds,
            target_resolution=target_resolution,
            target_z=target_z,
        )
        for wi in reflector_tasks:
            wi["percentile_target_quantile"] = float(target_quantile)

        refl_results = parallel_opt.run(
            scene_config=scene_config,
            optimizer_method="grid_search_point",
            work_items=reflector_tasks,
            optimization_params=OPTIMIZATION_PARAMS,
            verbose=True,
        )
        refl_results = _coerce_percentile_metrics(refl_results)
        total_evaluations += len(reflector_tasks)

        refl_best = refl_results["best_result"]
        refl_best_task = reflector_tasks[refl_results["best_task_id"]]
        current_reflector_u = refl_best_task["reflector_u"]
        current_reflector_v = refl_best_task["reflector_v"]
        current_reflector_target = refl_best_task["reflector_target"]

        print(
            "  Reflector sweep: "
            f"{len(reflector_tasks)} tasks, best={refl_best['best_metric_dbm']:.2f} dBm, "
            f"u={current_reflector_u:.3f}, v={current_reflector_v:.3f}, "
            f"target={current_reflector_target}"
        )

        if refl_best["best_metric"] > best_overall_score:
            best_overall_score = refl_best["best_metric"]
            best_overall_results = refl_results

    total_time = time.time() - start_time
    print(
        f"\nTotal time: {total_time:.1f}s | "
        f"Evaluations: {total_evaluations} | "
        f"Best: {best_overall_results['best_result']['best_metric_dbm']:.2f} dBm"
    )

    os.makedirs(output_dir, exist_ok=True)
    _save_results(
        best_overall_results,
        os.path.join(output_dir, "gs_2ap_reflector_results.json"),
    )
    parallel_opt.save_results_plot(
        best_overall_results,
        save_path=os.path.join(output_dir, "gs_2ap_reflector_parallel_results.png"),
        metric_name="Percentile Score",
        position_bounds=POSITION_BOUNDS,
        rss_range_dbm=RSS_RANGE_DBM,
    )

    return best_overall_results


# -- Example 3a: DEAP GA — 1 AP -----------------------------------------------

def example_deap_ga_1ap(
    ga_executor: RayActorPoolExecutor,
    ga_params: dict = None,
    output_dir: str = OUTPUT_DIR,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """
    Run a 1-AP Genetic Algorithm (4-gene chromosome ``[x, y, dx, dy]``).

    Creates a ``GeneticAlgorithmRunner`` with ``num_aps=1``, injects the
    ``RayActorPoolExecutor.map`` as the evaluation engine, runs the GA,
    saves results and an evolution plot.

    Args:
        ga_executor: ``RayActorPoolExecutor`` bound to a *1-TX* scene.
        ga_params: Override for evolutionary hyper-parameters.
        output_dir: Directory for output files.

    Returns:
        GA results dict.
    """
    if ga_params is None:
        ga_params = GA_PARAMS

    print("\n" + "=" * 80)
    print("DEAP Genetic Algorithm — 1 AP (4D [x, y, dx, dy])")
    print("=" * 80)

    ga_runner = GeneticAlgorithmRunner(
        position_bounds=POSITION_BOUNDS,
        fixed_z=FIXED_Z,
        executor_map=ga_executor.map,
        optimize_orientation=True,
        num_aps=1,
    )

    results = ga_runner.run(
        optimization_params=OPTIMIZATION_PARAMS,
        ga_params=ga_params,
        seed=random_seed,
        verbose=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    _save_ga_results(results, os.path.join(output_dir, "ga_1ap_results.json"))

    ga_runner.save_evolution_plot(
        results,
        save_path=os.path.join(output_dir, "ga_1ap_evolution.png"),
        position_bounds=POSITION_BOUNDS,
        rss_range_dbm=RSS_RANGE_DBM,
    )

    return results


# -- Example 3b: DEAP GA — 2 APs ----------------------------------------------

def example_deap_ga_2ap(
    ga_executor: RayActorPoolExecutor,
    ga_params: dict = None,
    min_ap_separation: float = 5.0,
    output_dir: str = OUTPUT_DIR,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """
    Run a 2-AP Genetic Algorithm (8-gene chromosome
    ``[x1, y1, x2, y2, dx1, dy1, dx2, dy2]``).

    Creates a ``GeneticAlgorithmRunner`` with ``num_aps=2`` and a
    separation constraint.  Fitness evaluation is delegated to the
    injected ``RayActorPoolExecutor``.

    Args:
        ga_executor: ``RayActorPoolExecutor`` bound to a *2-TX* scene.
        ga_params: Override for evolutionary hyper-parameters.
        min_ap_separation: Minimum allowed AP distance (metres).
        output_dir: Directory for output files.

    Returns:
        GA results dict.
    """
    if ga_params is None:
        ga_params = GA_PARAMS

    print("\n" + "=" * 80)
    print("DEAP Genetic Algorithm — 2 APs (8D [x1,y1,x2,y2,dx1,dy1,dx2,dy2])")
    print("=" * 80)
    print(f"  min_ap_separation = {min_ap_separation}m")

    ga_runner = GeneticAlgorithmRunner(
        position_bounds=POSITION_BOUNDS,
        fixed_z=FIXED_Z,
        executor_map=ga_executor.map,
        optimize_orientation=True,
        num_aps=2,
        min_ap_separation=min_ap_separation,
    )

    results = ga_runner.run(
        optimization_params=OPTIMIZATION_PARAMS,
        ga_params=ga_params,
        seed=random_seed,
        verbose=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    _save_ga_results(results, os.path.join(output_dir, "ga_2ap_results.json"))

    ga_runner.save_evolution_plot(
        results,
        save_path=os.path.join(output_dir, "ga_2ap_evolution.png"),
        position_bounds=POSITION_BOUNDS,
        rss_range_dbm=RSS_RANGE_DBM,
    )

    return results


# -- Utilities -----------------------------------------------------------------

def _fmt_dir(d):
    """Format a direction vector for printing."""
    if d is None:
        return " N/A"
    # Multi-AP: nested list
    if isinstance(d, (list, np.ndarray)) and len(d) > 0:
        if isinstance(d[0], (list, np.ndarray)):
            parts = [f"({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})" for v in d]
            return " " + " | ".join(parts)
    return f" ({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f})"


def _fmt_pos(p):
    """Format a position for printing."""
    if p is None:
        return "N/A"
    if isinstance(p, (list, np.ndarray)) and len(p) > 0:
        if isinstance(p[0], (list, np.ndarray)):
            parts = [f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})" for v in p]
            return " | ".join(parts)
    return f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"


def _save_results(results: dict, path: str) -> None:
    """Save results dict to JSON, converting numpy types."""
    all_results = results["all_results"]
    percentile_values_dbm = [
        r.get("grid_results", {}).get("percentile_score_dbm")
        for r in all_results
        if r.get("grid_results", {}).get("percentile_score_dbm") is not None
    ]

    summary_stats = {
        "min_rss_dbm": {
            "mean": results["aggregate_stats"].get("mean_metric_dbm"),
            "std": results["aggregate_stats"].get("std_metric_dbm"),
            "range": [
                results["aggregate_stats"].get("min_metric_dbm"),
                results["aggregate_stats"].get("max_metric_dbm"),
            ],
        }
    }
    if percentile_values_dbm:
        summary_stats["percentile_5_dbm"] = {
            "mean": float(np.mean(percentile_values_dbm)),
            "std": float(np.std(percentile_values_dbm)),
            "range": [float(np.min(percentile_values_dbm)), float(np.max(percentile_values_dbm))],
        }

    serializable = {
        "best_task_id": results["best_task_id"],
        "total_time": results["total_time"],
        "aggregate_stats": results["aggregate_stats"],
        "summary_stats": summary_stats,
        "pool_info": results["pool_info"],
        "best_result": {
            "task_id": results["best_result"]["task_id"],
            "worker_id": results["best_result"]["worker_id"],
            "best_position": results["best_result"]["best_position"],
            "best_metric": results["best_result"]["best_metric"],
            "best_metric_dbm": results["best_result"]["best_metric_dbm"],
            "best_iteration": results["best_result"].get("best_iteration", -1),
            "final_position": results["best_result"].get("final_position"),
            "best_direction": results["best_result"].get("best_direction"),
            "final_direction": results["best_result"].get("final_direction"),
            "best_look_at": results["best_result"].get("best_look_at"),
            "final_look_at": results["best_result"].get("final_look_at"),
            "reflector_position": results["best_result"].get("reflector_position"),
            "reflector_target": results["best_result"].get("reflector_target"),
            "reflector_u": results["best_result"].get("reflector_u"),
            "reflector_v": results["best_result"].get("reflector_v"),
            "percentile_score_dbm": results["best_result"].get("grid_results", {}).get("percentile_score_dbm"),
            "time_elapsed": results["best_result"]["time_elapsed"],
        },
        "all_metrics_dbm": [r["best_metric_dbm"] for r in results["all_results"]],
        "all_positions": [r["best_position"] for r in results["all_results"]],
        "all_best_directions": [r.get("best_direction") for r in results["all_results"]],
        "all_final_directions": [r.get("final_direction") for r in results["all_results"]],
        "all_best_look_ats": [r.get("best_look_at") for r in results["all_results"]],
        "all_final_look_ats": [r.get("final_look_at") for r in results["all_results"]],
        "all_reflector_positions": [r.get("reflector_position") for r in results["all_results"]],
        "all_reflector_targets": [r.get("reflector_target") for r in results["all_results"]],
        "all_percentile_scores_dbm": percentile_values_dbm,
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Results saved to: {path}")


def _save_ga_results(results: dict, path: str) -> None:
    """Save GA results dict to JSON (supports 1-AP and multi-AP)."""
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


# -- Summary helpers -----------------------------------------------------------

def _print_gd_summary(label: str, results: dict) -> None:
    """Print a detailed gradient descent summary block."""
    b = results["best_result"]
    stats = results["aggregate_stats"]
    print(f"\n{label} ({len(results['all_results'])} tasks):")
    print(f"  Best task:      #{results['best_task_id']}")
    print(f"  Best iteration: {b.get('best_iteration', -1) + 1}")
    print(f"  Best position:  {_fmt_pos(b['best_position'])}")
    print(f"  Best direction: {_fmt_dir(b.get('best_direction'))}")
    print(f"  Best Min RSS:   {b['best_metric_dbm']:.2f} dBm")
    print(f"  Wall-clock:     {results['total_time']:.2f}s")
    print(f"  Speedup:        {stats['speedup']:.2f}x")


def _print_gs_summary(label: str, results: dict) -> None:
    """Print a detailed grid search summary block."""
    b = results["best_result"]
    stats = results["aggregate_stats"]
    pct_values = [
        r.get("grid_results", {}).get("percentile_score_dbm")
        for r in results["all_results"]
        if r.get("grid_results", {}).get("percentile_score_dbm") is not None
    ]

    print(f"\n{label} ({len(results['all_results'])} grid points):")
    print(f"  Best task:      #{results['best_task_id']}")
    print(f"  Best AP pos:    {_fmt_pos(b['best_position'])}")
    print(f"  Best AP dir:    {_fmt_dir(b.get('best_direction'))}")
    print(f"  Reflector pos:  {_fmt_pos(b.get('reflector_position'))}")
    print(f"  Focal point:    {_fmt_pos(b.get('reflector_target'))}")
    print(f"  Best Min RSS:   {b['best_metric_dbm']:.2f} dBm")
    if b.get("grid_results", {}).get("percentile_score_dbm") is not None:
        print(f"  Best 5th pct:   {b['grid_results']['percentile_score_dbm']:.2f} dBm")
    print(
        f"  Min RSS stats:  mean={stats['mean_metric_dbm']:.2f}±{stats['std_metric_dbm']:.2f} dBm, "
        f"range=[{stats['min_metric_dbm']:.2f}, {stats['max_metric_dbm']:.2f}]"
    )
    if pct_values:
        print(
            f"  5th pct stats:  mean={np.mean(pct_values):.2f}±{np.std(pct_values):.2f} dBm, "
            f"range=[{np.min(pct_values):.2f}, {np.max(pct_values):.2f}]"
        )
    print(f"  Wall-clock:     {results['total_time']:.2f}s")
    print(f"  Speedup:        {stats['speedup']:.2f}x")


def _print_ga_summary(label: str, results: dict) -> None:
    """Print a detailed GA summary block (1-AP or multi-AP)."""
    num_aps = results.get("num_aps", 1)
    print(f"\n{label} (pop={results['ga_params']['pop_size']}, "
          f"gen={results['ga_params']['n_gen']}, APs={num_aps}):")
    if num_aps == 1:
        print(f"  Best position:  {_fmt_pos(results['best_position'])}")
        print(f"  Best direction: {_fmt_dir(results.get('best_direction'))}")
    else:
        for i in range(num_aps):
            pos = results['best_positions'][i]
            print(f"  AP{i} position:  ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            dirs = results.get('best_directions')
            if dirs and dirs[i]:
                d = dirs[i]
                print(f"  AP{i} direction: ({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f})")
        sep = results.get('best_ap_separation', 0)
        print(f"  AP separation:  {sep:.2f}m")
    print(f"  Best Min RSS:   {results['best_fitness_dbm']:.2f} dBm")
    print(f"  Total evals:    {results['total_evaluations']}")
    print(f"  Wall-clock:     {results['total_time']:.2f}s")


# -- Wrapper functions for 1-AP and 2-AP experiments ---------------------------

def run_all_1ap(
    output_dir: str = OUTPUT_DIR,
    num_pool_workers: int = NUM_POOL_WORKERS,
    gpu_fraction: float = GPU_FRACTION,
    random_seed: int = RANDOM_SEED,
    gd_num_tasks: int = 100,
    gd_num_iterations: int = 50,
    gd_optimization_overrides: Optional[dict] = None,
    ga_params: Optional[dict] = None,
) -> dict:
    """
    Run all 1-AP experiments: Gradient Descent + Grid Search + GA.

    Executors are created and destroyed **sequentially** to avoid
    deadlocks from concurrent Ray actor pools on the same GPU.

    Lifecycle::

        parallel_opt  (GD + GS)  →  shutdown  →  ga_executor (GA)  →  shutdown

    Args:
        output_dir: Directory for output files.
        num_pool_workers: Number of Ray workers per pool.
        gpu_fraction: GPU fraction per worker.

    Returns:
        Dict with ``gd_1ap``, ``gs_1ap``, and ``ga_1ap`` result dicts.
    """
    print("\n" + "#" * 80)
    print("#  ALL 1-AP EXPERIMENTS (GD + GS + GA)")
    print("#" * 80)

    all_results = {}

    # --- Phase 1: GD + GS via RayParallelOptimizer -----------------------
    parallel_opt = RayParallelOptimizer(
        num_workers=num_pool_workers,
        gpu_fraction=gpu_fraction,
    )
    try:
        all_results["gd_1ap"] = example_parallel_gradient_descent(
            parallel_opt,
            num_aps=1,
            num_tasks=gd_num_tasks,
            num_iterations=gd_num_iterations,
            output_dir=output_dir,
            scene_config=SCENE_CONFIG,
            random_seed=random_seed,
            optimization_overrides=gd_optimization_overrides,
        )

        all_results["gs_1ap"] = example_parallel_grid_search(
            parallel_opt,
            grid_resolution=1.0,
            output_dir=output_dir,
            scene_config=SCENE_CONFIG,
        )
    finally:
        parallel_opt.shutdown()
        print("1-AP GD+GS pool shut down.")

    # --- Phase 2: GA via RayActorPoolExecutor ----------------------------
    ga_executor = RayActorPoolExecutor(
        scene_config=SCENE_CONFIG,
        num_workers=num_pool_workers,
        gpu_fraction=gpu_fraction,
        verbose=True,
    )
    try:
        all_results["ga_1ap"] = example_deap_ga_1ap(
            ga_executor,
            ga_params=ga_params,
            output_dir=output_dir,
            random_seed=random_seed,
        )
    finally:
        ga_executor.shutdown()
        print("1-AP GA pool shut down.")

    print("\nAll 1-AP experiments (GD + GS + GA) complete.")

    return all_results


def run_all_2ap(
    output_dir: str = OUTPUT_DIR,
    num_pool_workers: int = NUM_POOL_WORKERS,
    gpu_fraction: float = GPU_FRACTION,
    random_seed: int = RANDOM_SEED,
    gd_num_tasks: int = 100,
    gd_num_iterations: int = 50,
    gd_repulsion_weight: float = 0.3,
    gd_samples_per_tx: int = 1_000_000,
    gd_optimization_overrides: Optional[dict] = None,
    ga_params: Optional[dict] = None,
    ga_min_ap_separation: float = 5.0,
) -> dict:
    """
    Run all 2-AP experiments: Gradient Descent + Grid Search (alternating) + GA.

    Executors are created and destroyed **sequentially** to avoid
    deadlocks from concurrent Ray actor pools on the same GPU.

    Lifecycle::

        parallel_opt  (GD + GS)  →  shutdown  →  ga_executor (GA)  →  shutdown

    Args:
        output_dir: Directory for output files.
        num_pool_workers: Number of Ray workers per pool.
        gpu_fraction: GPU fraction per worker.

    Returns:
        Dict with ``gd_2ap``, ``gs_2ap``, and ``ga_2ap`` result dicts.
    """
    print("\n" + "#" * 80)
    print("#  ALL 2-AP EXPERIMENTS (GD + GS + GA)")
    print("#" * 80)

    all_results = {}

    # --- Phase 1: GD + GS via RayParallelOptimizer -----------------------
    parallel_opt = RayParallelOptimizer(
        num_workers=num_pool_workers,
        gpu_fraction=gpu_fraction,
    )
    try:
        all_results["gd_2ap"] = example_parallel_gradient_descent(
            parallel_opt,
            num_aps=2,
            num_tasks=gd_num_tasks,
            num_iterations=gd_num_iterations,
            repulsion_weight=gd_repulsion_weight,
            samples_per_tx=gd_samples_per_tx,
            output_dir=output_dir,
            scene_config=SCENE_CONFIG_2AP,
            random_seed=random_seed,
            optimization_overrides=gd_optimization_overrides,
        )

        all_results["gs_2ap"] = example_parallel_grid_search_2ap_with_reflector(
            parallel_opt,
            grid_resolution=1.0,
            num_rounds=ALTERNATING_ROUNDS,
            outer_rounds=2,
            output_dir=output_dir,
            scene_config=SCENE_CONFIG_2AP,
            target_quantile=0.05,
        )
    finally:
        parallel_opt.shutdown()
        print("2-AP GD+GS pool shut down.")

    # --- Phase 2: GA via RayActorPoolExecutor ----------------------------
    ga_executor_2ap = RayActorPoolExecutor(
        scene_config=SCENE_CONFIG_2AP,
        num_workers=num_pool_workers,
        gpu_fraction=gpu_fraction,
        verbose=True,
    )
    try:
        all_results["ga_2ap"] = example_deap_ga_2ap(
            ga_executor_2ap,
            ga_params=ga_params,
            min_ap_separation=ga_min_ap_separation,
            output_dir=output_dir,
            random_seed=random_seed,
        )
    finally:
        ga_executor_2ap.shutdown()
        print("2-AP GA pool shut down.")

    print("\nAll 2-AP experiments (GD + GS + GA) complete.")

    return all_results


# -- Main ----------------------------------------------------------------------
def run_reflector_aware_grid_search_only(
    output_dir: str = OUTPUT_DIR,
    num_pool_workers: int = NUM_POOL_WORKERS,
    gpu_fraction: float = GPU_FRACTION,
    grid_resolution: float = 1.0,
    num_rounds: int = ALTERNATING_ROUNDS,
    outer_rounds: int = 3,
    target_quantile: float = 0.05,
    min_ap_separation: float = 10.0,
) -> dict:
    """
    Canonical reflector-aware entrypoint (grid search only).

    Runs only ``example_parallel_grid_search_2ap_with_reflector`` and prints a
    concise summary block.
    """
    # Initialize Ray once (all examples share the same cluster)
    ray.init(ignore_reinit_error=True)

    parallel_opt = RayParallelOptimizer(
        num_workers=num_pool_workers,
        gpu_fraction=gpu_fraction,
    )
    try:
        gs_results = example_parallel_grid_search_2ap_with_reflector(
            parallel_opt,
            grid_resolution=grid_resolution,
            num_rounds=num_rounds,
            outer_rounds=outer_rounds,
            output_dir=output_dir,
            scene_config=SCENE_CONFIG_2AP,
            target_quantile=target_quantile,
            min_ap_separation=min_ap_separation,
        )
    finally:
        parallel_opt.shutdown()
        print("Reflector-aware GS pool shut down.")

    print("\n" + "=" * 80)
    print("REFLECTOR-AWARE GRID SEARCH SUMMARY")
    print("=" * 80)
    print(f"\nPool: {num_pool_workers} workers (Scene loaded once per worker)")
    _print_gs_summary("Grid Search (2-AP + reflector)", gs_results)

    ray.shutdown()
    return {"gs_2ap": gs_results}


def run_reflector_aware():
    """
    Deprecated compatibility wrapper.

    Use run_reflector_aware_grid_search_only() instead.
    """
    warnings.warn(
        "run_reflector_aware() is deprecated and will be removed in a future "
        "release; use run_reflector_aware_grid_search_only() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_reflector_aware_grid_search_only()


def run_ap_only():
    """
    Deprecated legacy entrypoint.

    Use run_reflector_aware() instead.
    """
    warnings.warn(
        "run_ap_only() is deprecated and will be removed in a future release; "
        "use run_reflector_aware_grid_search_only() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_reflector_aware_grid_search_only()


if __name__ == "__main__":
    run_reflector_aware_grid_search_only(
        num_rounds=ALTERNATING_ROUNDS,
        outer_rounds=3,
        min_ap_separation=10.0,
    )