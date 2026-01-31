"""
Example: Quick grid search optimization test

This script demonstrates a fast test using grid search with reduced computational requirements.
"""

from pathlib import Path
from reflector_position import (
    setup_building_floor_scene,
    GridSearchAPOptimizer,
    GridSearchConfig,
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
    
    # Configure quick test parameters with smaller grid for speed
    config = GridSearchConfig(
        x_min=15.0,
        x_max=25.0,
        y_min=15.0,
        y_max=25.0,
        grid_resolution=5.0,  # Coarse grid for quick test (only 3x3 = 9 positions)
        samples_per_tx=50_000,  # Fewer samples for speed
        max_depth=10,  # Lower depth for speed
        fixed_z=3.8,
    )
    
    # Create optimizer
    print("\nRunning grid search optimization...")
    optimizer = GridSearchAPOptimizer(
        scene=scene,
        search_bounds=config.search_bounds,
        grid_resolution=config.grid_resolution,
        fixed_z=config.fixed_z,
    )
    
    # Run optimization
    best_position, best_rss = optimizer.optimize(
        samples_per_tx=config.samples_per_tx,
        max_depth=config.max_depth,
        verbose=True,
    )
    
    print(f"\n✓ Optimization complete!")
    print(f"  Best position: ({best_position[0]:.2f}, {best_position[1]:.2f}, {best_position[2]:.2f})")
    print(f"  Best min RSS: {best_rss:.6e} W")
    
    # Optionally plot results
    try:
        optimizer.plot_results(metric='min_rss_dbm')
    except Exception as e:
        print(f"Note: Could not plot results (requires display): {e}")


if __name__ == "__main__":
    main()
