"""
Example: Quick gradient descent optimization test

This script demonstrates a fast test using gradient descent with reduced computational requirements.
"""

from pathlib import Path
from reflector_position import (
    setup_building_floor_scene,
    GradientDescentAPOptimizer,
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
    
    # Configure quick test parameters
    config = GradientDescentConfig(
        initial_x=20.0,
        initial_y=20.0,
        x_min=5.0,
        x_max=35.0,
        y_min=5.0,
        y_max=35.0,
        fixed_z=3.8,
        num_iterations=5,  # Few iterations for quick test
        learning_rate=0.5,
        samples_per_tx=100_000,  # Fewer samples for speed
        max_depth=10,  # Lower depth for speed
        use_soft_min=True,
        temperature=0.1,
    )
    
    # Create and run optimizer
    print("\nRunning gradient descent optimization...")
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
        verbose=True,
    )
    
    print(f"\n✓ Optimization complete!")
    print(f"  Final position: ({final_position[0]:.2f}, {final_position[1]:.2f}, {final_position[2]:.2f})")
    
    # Optionally plot trajectory
    try:
        optimizer.plot_optimization_trajectory()
    except Exception as e:
        print(f"Note: Could not plot trajectory (requires display): {e}")


if __name__ == "__main__":
    main()
