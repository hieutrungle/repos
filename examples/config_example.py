"""
Example configuration file for reflector position optimization.

Copy this file and modify parameters as needed.
"""

from reflector_position.config import (
    SceneConfig,
    GridSearchConfig,
    GradientDescentConfig,
    OptimizationConfig,
)

# Scene configuration
scene_config = SceneConfig(
    scene_path="/path/to/your/scene.xml",
    frequency=5.18e9,  # 5.18 GHz WiFi
    tx_positions=[(10.0, 20.0, 3.8)],  # Single TX at ceiling height
    tx_power_dbm=5.0,
    rx_position=(16.0, 6.5, 1.5),  # Receiver at user height
)

# Grid search configuration
grid_search_config = GridSearchConfig(
    x_min=5.0,
    x_max=35.0,
    y_min=5.0,
    y_max=35.0,
    grid_resolution=5.0,  # 5 meter spacing
    fixed_z=3.8,  # Ceiling height
    samples_per_tx=500_000,
    max_depth=13,
    coverage_threshold_dbm=-100.0,
)

# Gradient descent configuration
gradient_descent_config = GradientDescentConfig(
    initial_x=20.0,  # Start in center
    initial_y=20.0,
    x_min=5.0,
    x_max=35.0,
    y_min=5.0,
    y_max=35.0,
    fixed_z=3.8,
    num_iterations=10,
    learning_rate=0.5,
    samples_per_tx=1_000_000,
    max_depth=15,
    use_soft_min=True,
    temperature=0.2,
    coverage_threshold_dbm=-100.0,
)

# Complete configuration
config = OptimizationConfig(
    scene=scene_config,
    grid_search=grid_search_config,
    gradient_descent=gradient_descent_config,
)
