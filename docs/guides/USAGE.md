# Usage Guide

## Overview

This guide demonstrates different ways to use the reflector position optimization package.

## Table of Contents

1. [Command-Line Interface](#command-line-interface)
2. [Python API](#python-api)
3. [Configuration](#configuration)
4. [Advanced Usage](#advanced-usage)
5. [Output and Results](#output-and-results)

## Command-Line Interface

The CLI is the quickest way to run optimizations.

### Basic Usage

```bash
# Run gradient descent (default parameters)
reflector-optimize /path/to/scene.xml --method gradient-descent

# Run grid search
reflector-optimize /path/to/scene.xml --method grid-search

# Run both and compare
reflector-optimize /path/to/scene.xml --method all
```

### Customizing Parameters

#### Grid Search Example

```bash
reflector-optimize building_floor.xml \
    --method grid-search \
    --gs-x-min 10 \
    --gs-x-max 30 \
    --gs-y-min 10 \
    --gs-y-max 30 \
    --gs-resolution 2.0 \
    --gs-samples 500000 \
    --gs-max-depth 13
```

#### Gradient Descent Example

```bash
reflector-optimize building_floor.xml \
    --method gradient-descent \
    --gd-init-x 20 \
    --gd-init-y 20 \
    --gd-iterations 20 \
    --gd-lr 0.5 \
    --gd-samples 1000000 \
    --gd-max-depth 15 \
    --gd-temperature 0.2
```

#### Full Comparison

```bash
reflector-optimize building_floor.xml \
    --method all \
    --frequency 5.18e9 \
    --tx-power 5.0 \
    --fixed-z 3.8
```

### Quiet Mode

Suppress detailed output:

```bash
reflector-optimize scene.xml --method gradient-descent --quiet
```

## Python API

### Quick Start

```python
from reflector_position import (
    setup_building_floor_scene,
    GradientDescentAPOptimizer,
)

# Load scene
scene = setup_building_floor_scene(
    scene_path="building_floor.xml",
    frequency=5.18e9,
)

# Create optimizer
optimizer = GradientDescentAPOptimizer(
    scene=scene,
    initial_position=(20.0, 20.0),
    fixed_z=3.8,
    position_bounds={
        'x_min': 5.0, 'x_max': 35.0,
        'y_min': 5.0, 'y_max': 35.0,
    }
)

# Run optimization
final_pos, final_rss = optimizer.optimize(
    num_iterations=10,
    learning_rate=0.5,
    samples_per_tx=1_000_000,
    verbose=True
)

print(f"Optimal position: {final_pos}")
```

### Using Configuration Objects

```python
from reflector_position import (
    SceneConfig,
    GradientDescentConfig,
    setup_building_floor_scene,
    GradientDescentAPOptimizer,
)

# Define configurations
scene_config = SceneConfig(
    scene_path="building_floor.xml",
    frequency=5.18e9,
    tx_power_dbm=5.0,
)

gd_config = GradientDescentConfig(
    initial_x=20.0,
    initial_y=20.0,
    num_iterations=20,
    learning_rate=0.5,
    samples_per_tx=1_000_000,
)

# Setup and run
scene = setup_building_floor_scene(
    scene_path=scene_config.scene_path,
    frequency=scene_config.frequency,
    tx_power_dbm=scene_config.tx_power_dbm,
)

optimizer = GradientDescentAPOptimizer(
    scene=scene,
    initial_position=gd_config.initial_position,
    position_bounds=gd_config.position_bounds,
)

final_pos, final_rss = optimizer.optimize(
    num_iterations=gd_config.num_iterations,
    learning_rate=gd_config.learning_rate,
    samples_per_tx=gd_config.samples_per_tx,
)
```

### Grid Search with Visualization

```python
from reflector_position import GridSearchAPOptimizer

optimizer = GridSearchAPOptimizer(
    scene=scene,
    search_bounds={
        'x_min': 5.0, 'x_max': 35.0,
        'y_min': 5.0, 'y_max': 35.0,
    },
    grid_resolution=2.0,
    fixed_z=3.8,
)

best_pos, best_rss = optimizer.optimize(
    samples_per_tx=500_000,
    max_depth=13,
    verbose=True,
)

# Plot results
optimizer.plot_results(metric='min_rss_dbm')
optimizer.plot_results(metric='coverage')
```

### Accessing Results

```python
# Gradient descent history
positions = optimizer.history['positions']
min_rss_values = optimizer.history['min_rss_dbm_values']
coverage_values = optimizer.history['coverage_values']
gradients = optimizer.history['gradients']

# Grid search results
all_positions = optimizer.results['positions']
all_rss_values = optimizer.results['min_rss_dbm_values']
radio_maps = optimizer.results['radio_maps']

# Find best from grid search
import numpy as np
best_idx = np.argmax(optimizer.results['min_rss_values'])
best_position = optimizer.results['positions'][best_idx]
best_radio_map = optimizer.results['radio_maps'][best_idx]
```

## Configuration

### Scene Configuration

```python
from reflector_position import SceneConfig

config = SceneConfig(
    scene_path="/path/to/scene.xml",
    frequency=5.18e9,  # Operating frequency in Hz
    tx_positions=[(10.0, 20.0, 3.8)],  # List of TX positions
    tx_power_dbm=5.0,  # Total TX power
    rx_position=(16.0, 6.5, 1.5),  # RX position
)
```

### Optimizer Configuration

```python
from reflector_position import GridSearchConfig, GradientDescentConfig

# Grid search
gs_config = GridSearchConfig(
    x_min=5.0,
    x_max=35.0,
    y_min=5.0,
    y_max=35.0,
    grid_resolution=5.0,
    fixed_z=3.8,
    samples_per_tx=500_000,
    max_depth=13,
    coverage_threshold_dbm=-100.0,
)

# Gradient descent
gd_config = GradientDescentConfig(
    initial_x=20.0,
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
```

## Advanced Usage

### Custom Metrics

```python
from reflector_position.metrics import (
    compute_min_rss_metric,
    compute_soft_min_rss_metric,
    compute_coverage_metric,
)

import torch

# Compute custom metrics from radio map
radio_map = optimizer.results['radio_maps'][0]
rss_tensor = torch.from_numpy(radio_map.rss)

min_rss = compute_min_rss_metric(rss_tensor)
soft_min = compute_soft_min_rss_metric(rss_tensor, temperature=0.1)
coverage = compute_coverage_metric(rss_tensor, threshold_dbm=-90.0)

print(f"Min RSS: {min_rss:.2e} W")
print(f"Soft min RSS: {soft_min:.2e} W")
print(f"Coverage (>-90 dBm): {coverage:.1f}%")
```

### Manual Radio Map Computation

```python
from reflector_position.utils import compute_radio_map_with_tx_position

# Compute radio map for specific position
tx_position = (15.0, 25.0, 3.8)
radio_map = compute_radio_map_with_tx_position(
    scene,
    tx_position,
    cell_size=(1.0, 1.0),
    samples_per_tx=1_000_000,
    max_depth=15,
)

# Access RSS values
rss_watts = radio_map.rss  # In Watts
from reflector_position.metrics import rss_to_dbm
rss_dbm = rss_to_dbm(rss_watts)  # Convert to dBm
```

### Visualization

```python
# Plot gradient descent trajectory
optimizer.plot_optimization_trajectory()

# Plot grid search heatmap
optimizer.plot_results(metric='min_rss_dbm')

# Render scene with radio map
from reflector_position import create_camera

camera = create_camera(
    position=(20, 20, 50),
    look_at=(20, 20, 1.5)
)

scene.render(
    camera=camera,
    radio_map=radio_map,
    rm_metric="rss",
    rm_vmax=-30,
    rm_vmin=-100,
    rm_db_scale=True,
)
```

## Output and Results

### Console Output

Typical gradient descent output:

```
Starting Gradient Descent Optimization (PyTorch + DrJit)
  Device: cuda
  Initial position: (20.00, 20.00, 3.80)
  Learning rate: 0.5
  Iterations: 10
----------------------------------------------------------------------
Iter   1/10 | Pos: (20.45, 20.12) | Min RSS: -85.23 dBm | Coverage: 78.3% | ...
Iter   2/10 | Pos: (20.87, 20.23) | Min RSS: -84.56 dBm | Coverage: 79.1% | ...
...
----------------------------------------------------------------------
Gradient Descent Complete!
  Initial position: (20.00, 20.00, 3.80)
  Final position: (22.34, 21.45, 3.80)
  Initial min RSS: -85.23 dBm
  Final min RSS: -82.10 dBm
  Improvement: 3.13 dB
```

### Accessing Data Programmatically

```python
# Save results to file
import json
import numpy as np

results = {
    'final_position': optimizer.get_full_position().tolist(),
    'min_rss_history': [float(x) for x in optimizer.history['min_rss_dbm_values']],
    'coverage_history': [float(x) for x in optimizer.history['coverage_values']],
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save numpy arrays
np.save('positions.npy', np.array(optimizer.history['positions']))
np.save('gradients.npy', np.array(optimizer.history['gradients']))
```

## Tips and Best Practices

1. **Start with grid search** on a coarse grid to understand the landscape
2. **Use gradient descent** with grid search best as initial position
3. **Reduce samples_per_tx** for faster iteration during development
4. **Increase max_depth** for more accurate results in complex scenes
5. **Use soft minimum** (temperature ~0.1-0.3) for better gradients
6. **Monitor gradient norms** - if too small, increase learning rate
7. **Save results frequently** for long-running optimizations
8. **Visualize trajectories** to debug optimization behavior

## Troubleshooting

**Optimization gets stuck:**
- Try different initial positions
- Reduce learning rate
- Increase samples_per_tx for more stable gradients

**Poor convergence:**
- Use soft minimum instead of hard minimum
- Adjust temperature parameter
- Check position bounds are reasonable

**Slow performance:**
- Reduce samples_per_tx
- Lower max_depth
- Use GPU if available
