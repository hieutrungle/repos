"""
Example: Quick comparison test between grid search and gradient descent

This script demonstrates a fast comparison using both optimizers with reduced computational requirements.
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
        frequency=5.18e9,  # 5.18 GHz
        tx_power_dbm=5.0,
    )
    
    print("\n" + "=" * 70)
    print("GRID SEARCH OPTIMIZATION")
    print("=" * 70)
    
    # Configure grid search with smaller grid for speed
    gs_config = GridSearchConfig(
        x_min=15.0,
        x_max=25.0,
        y_min=15.0,
        y_max=25.0,
        grid_resolution=5.0,  # Coarse grid (3x3 = 9 positions)
        samples_per_tx=50_000,  # Fewer samples for speed
        max_depth=10,  # Lower depth for speed
        fixed_z=3.8,
    )
    
    # Run grid search
    gs_optimizer = GridSearchAPOptimizer(
        scene=scene,
        search_bounds=gs_config.search_bounds,
        grid_resolution=gs_config.grid_resolution,
        fixed_z=gs_config.fixed_z,
    )
    
    gs_position, gs_rss = gs_optimizer.optimize(
        samples_per_tx=gs_config.samples_per_tx,
        max_depth=gs_config.max_depth,
        verbose=True,
    )
    
    print("\n" + "=" * 70)
    print("GRADIENT DESCENT OPTIMIZATION")
    print("=" * 70)
    
    # Configure gradient descent starting from grid search result
    gd_config = GradientDescentConfig(
        initial_x=float(gs_position[0]),
        initial_y=float(gs_position[1]),
        x_min=15.0,
        x_max=25.0,
        y_min=15.0,
        y_max=25.0,
        fixed_z=3.8,
        num_iterations=5,  # Few iterations for quick test
        learning_rate=0.5,
        samples_per_tx=50_000,  # Match grid search samples
        max_depth=10,  # Match grid search depth
        use_soft_min=True,
        temperature=0.1,
    )
    
    # Run gradient descent
    gd_optimizer = GradientDescentAPOptimizer(
        scene=scene,
        initial_position=gd_config.initial_position,
        fixed_z=gd_config.fixed_z,
        position_bounds=gd_config.position_bounds,
    )
    
    gd_position, gd_rss = gd_optimizer.optimize(
        num_iterations=gd_config.num_iterations,
        learning_rate=gd_config.learning_rate,
        samples_per_tx=gd_config.samples_per_tx,
        max_depth=gd_config.max_depth,
        use_soft_min=gd_config.use_soft_min,
        temperature=gd_config.temperature,
        verbose=True,
    )
    
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\nGrid Search:")
    print(f"  Best position: ({gs_position[0]:.2f}, {gs_position[1]:.2f}, {gs_position[2]:.2f})")
    print(f"  Best min RSS: {gs_rss:.6e} W")
    
    print(f"\nGradient Descent:")
    print(f"  Final position: ({gd_position[0]:.2f}, {gd_position[1]:.2f}, {gd_position[2]:.2f})")
    print(f"  Final min RSS: {gd_rss:.6e} W")
    
    print(f"\nImprovement:")
    improvement = ((gd_rss - gs_rss) / gs_rss) * 100 if gs_rss > 0 else 0
    print(f"  RSS improvement: {improvement:.2f}%")
    
    print(f"\nPosition change:")
    pos_change = ((gd_position[0] - gs_position[0])**2 + 
                  (gd_position[1] - gs_position[1])**2)**0.5
    print(f"  Distance moved: {pos_change:.2f} m")
    
    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
