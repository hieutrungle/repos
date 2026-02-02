# Optimizer Factory Pattern

## Overview

The optimizer factory pattern provides a consistent interface for creating and using different optimization methods. This makes it easy to:

- Switch between optimization methods
- Compare different approaches
- Extend with custom optimizers
- Maintain clean, modular code

## Architecture

### Base Class: `BaseAPOptimizer`

All optimizers inherit from this abstract base class, ensuring a consistent API:

```python
from reflector_position.optimizers import BaseAPOptimizer

class BaseAPOptimizer(ABC):
    @abstractmethod
    def optimize(self, samples_per_tx, max_depth, verbose, **kwargs):
        """Run optimization to find best AP position."""
        pass
    
    @abstractmethod
    def plot_results(self, **kwargs):
        """Visualize optimization results."""
        pass
```

### Available Optimizers

1. **GradientDescentAPOptimizer** - Physics-aware gradient-based optimization
2. **GridSearchAPOptimizer** - Exhaustive spatial search

### Factory: `OptimizerFactory`

Creates optimizer instances with a consistent interface:

```python
from reflector_position import OptimizerFactory

optimizer = OptimizerFactory.create(
    method="gradient_descent",
    scene=scene,
    **method_specific_params
)
```

## Usage

### Basic Usage

```python
from reflector_position import create_optimizer, setup_building_floor_scene

# Setup scene
scene = setup_building_floor_scene("scene.xml")

# Create optimizer
optimizer = create_optimizer(
    method="gradient_descent",
    scene=scene,
    initial_position=(10.0, 10.0),
    position_bounds={'x_min': 0, 'x_max': 20, 'y_min': 0, 'y_max': 20}
)

# Run optimization
position, rss = optimizer.optimize(
    num_iterations=10,
    learning_rate=0.5,
    samples_per_tx=1_000_000
)

# Visualize
optimizer.plot_results()
```

### Method-Specific Parameters

#### Gradient Descent

```python
optimizer = create_optimizer(
    method="gradient_descent",
    scene=scene,
    initial_position=(10.0, 10.0),  # Required: starting point
    position_bounds={'x_min': 0, 'x_max': 20, 'y_min': 0, 'y_max': 20},
    fixed_z=3.8,
)

position, rss = optimizer.optimize(
    num_iterations=50,
    learning_rate=0.5,
    samples_per_tx=1_000_000,
    max_depth=13,
    use_soft_min=True,
    temperature=0.2,
)
```

#### Grid Search

```python
optimizer = create_optimizer(
    method="grid_search",
    scene=scene,
    search_bounds={'x_min': 0, 'x_max': 20, 'y_min': 0, 'y_max': 20},  # Required
    grid_resolution=2.0,  # Grid spacing in meters
    fixed_z=3.8,
)

position, rss = optimizer.optimize(
    samples_per_tx=500_000,
    max_depth=13,
)
```

### Switching Methods Easily

```python
def run_comparison(method: str, scene):
    """Run optimization with any method."""
    if method == "gradient_descent":
        optimizer = create_optimizer(
            method=method,
            scene=scene,
            initial_position=(10.0, 10.0),
            position_bounds={'x_min': 0, 'x_max': 20, 'y_min': 0, 'y_max': 20}
        )
        return optimizer.optimize(num_iterations=10, learning_rate=0.5)
    
    elif method == "grid_search":
        optimizer = create_optimizer(
            method=method,
            scene=scene,
            search_bounds={'x_min': 0, 'x_max': 20, 'y_min': 0, 'y_max': 20},
            grid_resolution=5.0
        )
        return optimizer.optimize()

# Compare methods
for method in ["gradient_descent", "grid_search"]:
    position, rss = run_comparison(method, scene)
    print(f"{method}: {position}, RSS={rss:.2f}")
```

### List Available Methods

```python
from reflector_position import OptimizerFactory

methods = OptimizerFactory.list_methods()
print(f"Available: {methods}")
# Output: Available: ['gradient_descent', 'grid_search']
```

## Extending with Custom Optimizers

You can add your own optimization methods:

```python
from reflector_position.optimizers import BaseAPOptimizer, OptimizerFactory
import numpy as np

class MyCustomOptimizer(BaseAPOptimizer):
    def __init__(self, scene, custom_param, **kwargs):
        super().__init__(scene=scene, **kwargs)
        self.custom_param = custom_param
    
    def optimize(self, samples_per_tx=1_000_000, max_depth=13, verbose=True, **kwargs):
        # Your optimization logic here
        position = np.array([10.0, 10.0, self.fixed_z])
        rss = -50.0  # Example value
        return position, rss
    
    def plot_results(self, **kwargs):
        # Your visualization logic
        print("Plotting results...")

# Register the custom optimizer
OptimizerFactory.register("my_custom", MyCustomOptimizer)

# Use it
optimizer = create_optimizer(
    method="my_custom",
    scene=scene,
    custom_param=42,
)
```

## Best Practices

### 1. Use Factory for Configuration Flexibility

```python
# Good: Easy to configure via command line or config file
method = "gradient_descent"  # Could come from argparse or config
optimizer = create_optimizer(method=method, scene=scene, **params)
```

### 2. Consistent Error Handling

```python
from reflector_position import OptimizerFactory

try:
    optimizer = OptimizerFactory.create(
        method=user_input_method,
        scene=scene,
        **params
    )
except ValueError as e:
    print(f"Invalid method: {e}")
    available = OptimizerFactory.list_methods()
    print(f"Available methods: {available}")
```

### 3. Type Hints for Clarity

```python
from reflector_position.optimizers import BaseAPOptimizer

def optimize_scene(
    optimizer: BaseAPOptimizer,
    samples: int = 1_000_000
) -> tuple[np.ndarray, float]:
    return optimizer.optimize(samples_per_tx=samples)
```

## Benefits

1. **Consistency**: All optimizers follow the same interface
2. **Extensibility**: Easy to add new optimization methods
3. **Maintainability**: Changes to one optimizer don't affect others
4. **Testability**: Easy to mock and test individual components
5. **Flexibility**: Switch methods without changing calling code

## Ray Integration (Future)

The factory pattern is designed to support Ray-based distributed optimization:

```python
# Future: Ray-based multi-start gradient descent
optimizer = create_optimizer(
    method="ray_multi_start",
    scene=scene,
    num_workers=32,
    initial_positions=generate_random_starts(32),
    gpu_fraction=0.1,
)

best_position, best_rss = optimizer.optimize()
```

See [docs/methodology/RAY_ARCHITECTURE.md](../../docs/methodology/RAY_ARCHITECTURE.md) for details on the Ray-based architecture.
