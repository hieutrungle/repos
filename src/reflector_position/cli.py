"""
Command-line interface for reflector position optimization.

This module provides the main entry point for running AP position optimization
using either grid search, gradient descent, or both methods.
"""

import argparse
import sys
from pathlib import Path

from .config import SceneConfig, GridSearchConfig, GradientDescentConfig
from .scene_setup import setup_building_floor_scene
from .optimizers import GridSearchAPOptimizer, GradientDescentAPOptimizer
from .utils import compute_radio_map_with_tx_position
from .metrics import rss_to_dbm


def run_grid_search(scene, config: GridSearchConfig, verbose: bool = True):
    """
    Run grid search optimization.

    Args:
        scene: Sionna Scene object
        config: Grid search configuration
        verbose: Print detailed output

    Returns:
        GridSearchAPOptimizer instance with results
    """
    optimizer = GridSearchAPOptimizer(
        scene=scene,
        search_bounds=config.search_bounds,
        grid_resolution=config.grid_resolution,
        fixed_z=config.fixed_z,
    )

    best_position, best_rss = optimizer.optimize(
        samples_per_tx=config.samples_per_tx,
        max_depth=config.max_depth,
        coverage_threshold_dbm=config.coverage_threshold_dbm,
        verbose=verbose,
    )

    return optimizer


def run_gradient_descent(scene, config: GradientDescentConfig, verbose: bool = True):
    """
    Run gradient descent optimization.

    Args:
        scene: Sionna Scene object
        config: Gradient descent configuration
        verbose: Print detailed output

    Returns:
        GradientDescentAPOptimizer instance with results
    """
    optimizer = GradientDescentAPOptimizer(
        scene=scene,
        initial_position=config.initial_position,
        fixed_z=config.fixed_z,
        position_bounds=config.position_bounds,
    )

    final_position, final_rss = optimizer.optimize(
        num_iterations=config.num_iterations,
        learning_rate=config.learning_rate,
        samples_per_tx=config.samples_per_tx,
        max_depth=config.max_depth,
        use_soft_min=config.use_soft_min,
        temperature=config.temperature,
        coverage_threshold_dbm=config.coverage_threshold_dbm,
        verbose=verbose,
    )

    return optimizer


def compare_results(grid_optimizer, gradient_optimizer):
    """
    Compare results from both optimizers.

    Args:
        grid_optimizer: GridSearchAPOptimizer instance
        gradient_optimizer: GradientDescentAPOptimizer instance
    """
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPARISON: Grid Search vs. Gradient Descent")
    print("=" * 80)

    # Grid search statistics
    gs_best_idx = max(range(len(grid_optimizer.results["min_rss_values"])),
                      key=lambda i: grid_optimizer.results["min_rss_values"][i])
    gs_best_pos = grid_optimizer.results["positions"][gs_best_idx]
    gs_best_rss = grid_optimizer.results["min_rss_dbm_values"][gs_best_idx]
    gs_best_coverage = grid_optimizer.results["coverage_values"][gs_best_idx]
    gs_num_evals = len(grid_optimizer.results["positions"])

    # Gradient descent statistics
    gd_initial_pos = gradient_optimizer.history["positions"][0]
    gd_final_pos = gradient_optimizer.history["positions"][-1]
    gd_initial_rss = gradient_optimizer.history["min_rss_dbm_values"][0]
    gd_final_rss = gradient_optimizer.history["min_rss_dbm_values"][-1]
    gd_final_coverage = gradient_optimizer.history["coverage_values"][-1]
    gd_num_iters = len(gradient_optimizer.history["positions"])
    gd_improvement = gd_final_rss - gd_initial_rss

    print("\n📊 GRID SEARCH RESULTS:")
    print(f"  ├─ Positions evaluated: {gs_num_evals}")
    print(f"  ├─ Best position: ({gs_best_pos[0]:.2f}, {gs_best_pos[1]:.2f}, {gs_best_pos[2]:.2f})")
    print(f"  ├─ Best min RSS: {gs_best_rss:.2f} dBm")
    print(f"  └─ Best coverage: {gs_best_coverage:.1f}%")

    print("\n🎯 GRADIENT DESCENT RESULTS:")
    print(f"  ├─ Iterations: {gd_num_iters}")
    print(f"  ├─ Initial position: ({gd_initial_pos[0]:.2f}, {gd_initial_pos[1]:.2f}, {gd_initial_pos[2]:.2f})")
    print(f"  ├─ Final position: ({gd_final_pos[0]:.2f}, {gd_final_pos[1]:.2f}, {gd_final_pos[2]:.2f})")
    print(f"  ├─ Initial min RSS: {gd_initial_rss:.2f} dBm")
    print(f"  ├─ Final min RSS: {gd_final_rss:.2f} dBm")
    print(f"  ├─ Improvement: {gd_improvement:.2f} dB")
    print(f"  └─ Final coverage: {gd_final_coverage:.1f}%")

    print("\n⚡ EFFICIENCY COMPARISON:")
    efficiency_ratio = gs_num_evals / gd_num_iters
    print(f"  ├─ Grid search evaluations: {gs_num_evals}")
    print(f"  ├─ Gradient descent iterations: {gd_num_iters}")
    print(f"  └─ Efficiency ratio: {efficiency_ratio:.1f}× (GD is {efficiency_ratio:.1f}× more efficient)")

    print("\n" + "=" * 80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AP Position Optimization using Grid Search and/or Gradient Descent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "scene_path",
        type=str,
        help="Path to the scene XML file",
    )

    # Method selection
    parser.add_argument(
        "--method",
        type=str,
        choices=["grid-search", "gradient-descent", "all"],
        default="all",
        help="Optimization method to run",
    )

    # Scene configuration
    parser.add_argument("--frequency", type=float, default=5.18e9, help="Operating frequency in Hz")
    parser.add_argument("--tx-power", type=float, default=5.0, help="Transmitter power in dBm")

    # Grid search configuration
    parser.add_argument("--gs-x-min", type=float, default=5.0, help="Grid search X minimum")
    parser.add_argument("--gs-x-max", type=float, default=35.0, help="Grid search X maximum")
    parser.add_argument("--gs-y-min", type=float, default=5.0, help="Grid search Y minimum")
    parser.add_argument("--gs-y-max", type=float, default=35.0, help="Grid search Y maximum")
    parser.add_argument("--gs-resolution", type=float, default=5.0, help="Grid resolution in meters")
    parser.add_argument("--gs-samples", type=int, default=500_000, help="Samples per TX for grid search")
    parser.add_argument("--gs-max-depth", type=int, default=13, help="Max ray tracing depth for grid search")

    # Gradient descent configuration
    parser.add_argument("--gd-init-x", type=float, default=20.0, help="GD initial X position")
    parser.add_argument("--gd-init-y", type=float, default=20.0, help="GD initial Y position")
    parser.add_argument("--gd-x-min", type=float, default=5.0, help="GD X minimum bound")
    parser.add_argument("--gd-x-max", type=float, default=35.0, help="GD X maximum bound")
    parser.add_argument("--gd-y-min", type=float, default=5.0, help="GD Y minimum bound")
    parser.add_argument("--gd-y-max", type=float, default=35.0, help="GD Y maximum bound")
    parser.add_argument("--gd-iterations", type=int, default=10, help="Number of GD iterations")
    parser.add_argument("--gd-lr", type=float, default=0.5, help="Learning rate for GD")
    parser.add_argument("--gd-samples", type=int, default=1_000_000, help="Samples per TX for GD")
    parser.add_argument("--gd-max-depth", type=int, default=15, help="Max ray tracing depth for GD")
    parser.add_argument("--gd-temperature", type=float, default=0.2, help="Temperature for soft minimum")

    # Common settings
    parser.add_argument("--fixed-z", type=float, default=3.8, help="Fixed Z height for AP")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    # Validate scene path
    scene_path = Path(args.scene_path)
    if not scene_path.exists():
        print(f"Error: Scene file not found: {scene_path}", file=sys.stderr)
        sys.exit(1)

    verbose = not args.quiet

    # Setup scene
    if verbose:
        print("=" * 80)
        print("AP POSITION OPTIMIZATION")
        print("=" * 80)
        print(f"\n📍 Loading scene: {scene_path}")

    scene_config = SceneConfig(
        scene_path=str(scene_path),
        frequency=args.frequency,
        tx_power_dbm=args.tx_power,
    )

    scene = setup_building_floor_scene(
        scene_path=scene_config.scene_path,
        frequency=scene_config.frequency,
        tx_power_dbm=scene_config.tx_power_dbm,
    )

    if verbose:
        print(f"✓ Scene loaded successfully")
        print(f"  - Frequency: {args.frequency / 1e9:.2f} GHz")
        print(f"  - TX Power: {args.tx_power} dBm")
        print(f"  - Transmitters: {len(scene.transmitters)}")
        print(f"  - Receivers: {len(scene.receivers)}")

    # Run optimization(s)
    grid_optimizer = None
    gradient_optimizer = None

    if args.method in ["grid-search", "all"]:
        if verbose:
            print("\n" + "=" * 80)
            print("GRID SEARCH OPTIMIZATION")
            print("=" * 80)

        gs_config = GridSearchConfig(
            x_min=args.gs_x_min,
            x_max=args.gs_x_max,
            y_min=args.gs_y_min,
            y_max=args.gs_y_max,
            grid_resolution=args.gs_resolution,
            fixed_z=args.fixed_z,
            samples_per_tx=args.gs_samples,
            max_depth=args.gs_max_depth,
        )

        grid_optimizer = run_grid_search(scene, gs_config, verbose=verbose)

    if args.method in ["gradient-descent", "all"]:
        if verbose:
            print("\n" + "=" * 80)
            print("GRADIENT DESCENT OPTIMIZATION")
            print("=" * 80)

        gd_config = GradientDescentConfig(
            initial_x=args.gd_init_x,
            initial_y=args.gd_init_y,
            x_min=args.gd_x_min,
            x_max=args.gd_x_max,
            y_min=args.gd_y_min,
            y_max=args.gd_y_max,
            fixed_z=args.fixed_z,
            num_iterations=args.gd_iterations,
            learning_rate=args.gd_lr,
            samples_per_tx=args.gd_samples,
            max_depth=args.gd_max_depth,
            temperature=args.gd_temperature,
        )

        gradient_optimizer = run_gradient_descent(scene, gd_config, verbose=verbose)

    # Compare results if both methods were run
    if args.method == "all" and grid_optimizer and gradient_optimizer:
        compare_results(grid_optimizer, gradient_optimizer)

    if verbose:
        print("\n✓ Optimization complete!")


if __name__ == "__main__":
    main()
