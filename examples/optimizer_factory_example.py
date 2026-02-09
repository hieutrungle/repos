"""
Example: Using the Optimizer Factory

This example demonstrates how to use the optimizer factory pattern
to easily switch between different optimization methods.
"""

from pathlib import Path
from reflector_position import (
    setup_building_floor_scene,
    OptimizerFactory,
    create_optimizer,
)


def main():
    # Setup scene (same for all methods)
    scene = setup_building_floor_scene(
        scene_path=Path.home() / "blender" / "models" / "building_floor" / "building_floor.xml",
        frequency=5.18e9,
        tx_power_dbm=5.0,
    )
    
    # Define common parameters
    position_bounds = {
        'x_min': 0.0,
        'x_max': 20.0,
        'y_min': 0.0,
        'y_max': 20.0,
    }
    
    # Method 1: Using OptimizerFactory.create()
    print("=" * 70)
    print("Method 1: Using OptimizerFactory.create()")
    print("=" * 70)
    
    # Create gradient descent optimizer
    gd_optimizer = OptimizerFactory.create(
        method="gradient_descent",
        scene=scene,
        initial_position=(10.0, 10.0),
        position_bounds=position_bounds,
    )
    
    # Run optimization
    gd_position, gd_rss = gd_optimizer.optimize(
        num_iterations=10,
        learning_rate=0.5,
        samples_per_tx=500_000,
    )
    
    print(f"\nGradient Descent Result:")
    print(f"  Position: {gd_position}")
    print(f"  Min RSS: {gd_rss:.6f}")
    
    # Method 2: Using the convenience function
    print("\n" + "=" * 70)
    print("Method 2: Using create_optimizer() convenience function")
    print("=" * 70)
    
    # Create grid search optimizer
    gs_optimizer = create_optimizer(
        method="grid-search",  # Note: hyphens are automatically converted
        scene=scene,
        search_bounds=position_bounds,
        grid_resolution=5.0,
    )
    
    # Run optimization
    gs_position, gs_rss = gs_optimizer.optimize(
        samples_per_tx=500_000,
    )
    
    print(f"\nGrid Search Result:")
    print(f"  Position: {gs_position}")
    print(f"  Min RSS: {gs_rss:.6f}")
    
    # Method 3: Switch methods easily with a parameter
    print("\n" + "=" * 70)
    print("Method 3: Easy method switching")
    print("=" * 70)
    
    def run_optimization(method_name: str):
        """Helper function to run any optimization method."""
        if method_name == "gradient_descent":
            optimizer = create_optimizer(
                method=method_name,
                scene=scene,
                initial_position=(15.0, 15.0),
                position_bounds=position_bounds,
            )
            return optimizer.optimize(
                num_iterations=5,
                learning_rate=0.5,
                samples_per_tx=500_000,
            )
        elif method_name == "grid_search":
            optimizer = create_optimizer(
                method=method_name,
                scene=scene,
                search_bounds=position_bounds,
                grid_resolution=5.0,
            )
            return optimizer.optimize(samples_per_tx=500_000)
    
    # Run different methods
    for method in ["gradient_descent", "grid_search"]:
        position, rss = run_optimization(method)
        print(f"\n{method.replace('_', ' ').title()}:")
        print(f"  Position: {position}")
        print(f"  Min RSS: {rss:.6f}")
    
    # Method 4: List available methods
    print("\n" + "=" * 70)
    print("Available Optimization Methods")
    print("=" * 70)
    available_methods = OptimizerFactory.list_methods()
    print(f"Available methods: {', '.join(available_methods)}")
    
    # Method 5: Visualize results
    print("\n" + "=" * 70)
    print("Visualizing Results")
    print("=" * 70)
    
    # Re-run with visualization
    optimizer = create_optimizer(
        method="gradient_descent",
        scene=scene,
        initial_position=(10.0, 10.0),
        position_bounds=position_bounds,
    )
    
    position, rss = optimizer.optimize(
        num_iterations=10,
        learning_rate=0.5,
        samples_per_tx=500_000,
        verbose=False,
    )
    
    # Plot trajectory
    optimizer.plot_optimization_trajectory()


if __name__ == "__main__":
    main()
