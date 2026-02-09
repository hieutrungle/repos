"""
Example: Full comparison of grid search vs gradient descent

This script runs both optimization methods and compares their results.
"""

from pathlib import Path
from reflector_position import (
    setup_building_floor_scene,
    GridSearchAPOptimizer,
    GradientDescentAPOptimizer,
    GridSearchConfig,
    GradientDescentConfig,
)


def main():
    # Path to your scene file
    scene_path = Path.home() / "blender" / "models" / "building_floor" / "building_floor.xml"
    
    # Setup scene
    print("Setting up scene...")
    scene = setup_building_floor_scene(
        scene_path=str(scene_path),
        frequency=5.18e9,
        tx_power_dbm=5.0,
    )
    
    # Grid search configuration
    gs_config = GridSearchConfig(
        x_min=5.0,
        x_max=25.0,
        y_min=5.0,
        y_max=25.0,
        grid_resolution=2.0,  # 2 meter spacing
        fixed_z=3.8,
        samples_per_tx=1_000_000,
        max_depth=13,
    )
    
    # Gradient descent configuration
    gd_config = GradientDescentConfig(
        initial_x=15.0,
        initial_y=15.0,
        x_min=5.0,
        x_max=25.0,
        y_min=5.0,
        y_max=25.0,
        fixed_z=3.8,
        num_iterations=30,
        learning_rate=0.25,
        samples_per_tx=1_000_000,
        max_depth=13,
        use_soft_min=True,
        temperature=0.2,
    )
    
    # Run grid search
    print("\n" + "=" * 80)
    print("GRID SEARCH OPTIMIZATION")
    print("=" * 80)
    
    grid_optimizer = GridSearchAPOptimizer(
        scene=scene,
        search_bounds=gs_config.search_bounds,
        grid_resolution=gs_config.grid_resolution,
        fixed_z=gs_config.fixed_z,
    )
    
    grid_optimizer.optimize(
        samples_per_tx=gs_config.samples_per_tx,
        max_depth=gs_config.max_depth,
        verbose=True,
    )
    
    # Run gradient descent
    print("\n" + "=" * 80)
    print("GRADIENT DESCENT OPTIMIZATION")
    print("=" * 80)
    
    gd_optimizer = GradientDescentAPOptimizer(
        scene=scene,
        initial_position=gd_config.initial_position,
        fixed_z=gd_config.fixed_z,
        position_bounds=gd_config.position_bounds,
    )
    
    gd_optimizer.optimize(
        num_iterations=gd_config.num_iterations,
        learning_rate=gd_config.learning_rate,
        samples_per_tx=gd_config.samples_per_tx,
        max_depth=gd_config.max_depth,
        use_soft_min=gd_config.use_soft_min,
        temperature=gd_config.temperature,
        verbose=True,
    )
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    gs_best_idx = max(range(len(grid_optimizer.results["min_rss_values"])),
                      key=lambda i: grid_optimizer.results["min_rss_values"][i])
    gs_best_pos = grid_optimizer.results["positions"][gs_best_idx]
    gs_best_rss = grid_optimizer.results["min_rss_dbm_values"][gs_best_idx]
    
    gd_final_pos = gd_optimizer.history["positions"][-1]
    gd_final_rss = gd_optimizer.history["min_rss_dbm_values"][-1]
    
    print(f"\nGrid Search Best:")
    print(f"  Position: ({gs_best_pos[0]:.2f}, {gs_best_pos[1]:.2f}, {gs_best_pos[2]:.2f})")
    print(f"  Min RSS: {gs_best_rss:.2f} dBm")
    print(f"  Evaluations: {len(grid_optimizer.results['positions'])}")
    
    print(f"\nGradient Descent Final:")
    print(f"  Position: ({gd_final_pos[0]:.2f}, {gd_final_pos[1]:.2f}, {gd_final_pos[2]:.2f})")
    print(f"  Min RSS: {gd_final_rss:.2f} dBm")
    print(f"  Iterations: {len(gd_optimizer.history['positions'])}")
    
    efficiency = len(grid_optimizer.results['positions']) / len(gd_optimizer.history['positions'])
    print(f"\nEfficiency: Gradient descent is {efficiency:.1f}× faster")
    
    # Plot results
    try:
        grid_optimizer.plot_results(metric='min_rss_dbm')
        gd_optimizer.plot_optimization_trajectory()
    except Exception as e:
        print(f"\nNote: Could not plot results (requires display): {e}")


if __name__ == "__main__":
    main()
