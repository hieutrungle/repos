# Project Structure Overview

This document provides an overview of the refactored project structure.

## Directory Layout

```
reflector-position/
│
├── src/reflector_position/          # Main package source code
│   ├── __init__.py                  # Package initialization & exports
│   ├── cli.py                       # Command-line interface
│   ├── config.py                    # Configuration dataclasses
│   ├── metrics.py                   # RSS metrics (min, soft-min, coverage)
│   ├── scene_setup.py               # Scene loading & configuration
│   ├── utils.py                     # Utility functions
│   └── optimizers/                  # Optimizer implementations
│       ├── __init__.py
│       ├── grid_search.py           # Grid search optimizer
│       └── gradient_descent.py      # Gradient descent optimizer
│
├── examples/                         # Example scripts
│   ├── quick_test.py                # Fast test with reduced params
│   ├── full_comparison.py           # Compare both methods
│   └── config_example.py            # Example configuration
│
├── notebooks/                        # Jupyter notebooks (preserved)
│   ├── building_floor.ipynb         # Original development notebook
│   └── ...
│
├── diff-rt-calibration/             # External dependency (preserved)
│
├── context/                          # Documentation & planning
│
├── pyproject.toml                   # Package configuration & metadata
├── requirements.txt                 # Pinned dependencies
├── README.md                        # Main documentation
├── INSTALL.md                       # Installation guide
├── USAGE.md                         # Detailed usage guide
├── LICENSE                          # MIT License
├── main.py                          # Development entry point
└── .gitignore                       # Git ignore patterns

```

## Key Components

### Source Code (`src/reflector_position/`)

#### Core Modules

1. **`__init__.py`** - Package initialization
   - Exports public API
   - Version information
   - Convenience imports

2. **`cli.py`** - Command-line interface
   - Argument parsing
   - Method selection (grid-search, gradient-descent, all)
   - Progress reporting
   - Results comparison

3. **`config.py`** - Configuration management
   - `SceneConfig`: Scene setup parameters
   - `GridSearchConfig`: Grid search parameters
   - `GradientDescentConfig`: Gradient descent parameters
   - `OptimizationConfig`: Complete configuration container

4. **`metrics.py`** - RSS metrics
   - `compute_min_rss_metric()`: Hard minimum RSS
   - `compute_soft_min_rss_metric()`: Differentiable soft minimum
   - `compute_coverage_metric()`: Coverage area calculation
   - `rss_to_dbm()`: Unit conversion

5. **`scene_setup.py`** - Scene configuration
   - `setup_building_floor_scene()`: Load and configure scene
   - `create_camera()`: Camera for visualization

6. **`utils.py`** - Utility functions
   - `compute_radio_map_with_tx_position()`: Radio map computation

#### Optimizers (`src/reflector_position/optimizers/`)

1. **`grid_search.py`** - Grid search implementation
   - `GridSearchAPOptimizer` class
   - Exhaustive spatial search
   - Result visualization

2. **`gradient_descent.py`** - Gradient descent implementation
   - `GradientDescentAPOptimizer` class
   - Differentiable ray tracing
   - PyTorch + DrJit integration
   - Trajectory visualization

### Examples (`examples/`)

1. **`quick_test.py`** - Fast test
   - Reduced parameters for quick validation
   - Good for development/debugging

2. **`full_comparison.py`** - Complete comparison
   - Runs both methods
   - Compares results
   - Shows efficiency gains

3. **`config_example.py`** - Configuration template
   - Shows all configuration options
   - Copy and modify for your use case

### Documentation

1. **`README.md`** - Main documentation
   - Project overview
   - Installation instructions
   - Quick start guide
   - API examples

2. **`INSTALL.md`** - Detailed installation
   - Step-by-step installation
   - GPU support
   - Troubleshooting

3. **`USAGE.md`** - Usage guide
   - CLI examples
   - Python API examples
   - Advanced usage
   - Tips and best practices

## Package Configuration

### `pyproject.toml`

Modern Python packaging configuration:
- Project metadata
- Dependencies
- Development tools (black, ruff, pytest)
- CLI entry points

### Entry Points

The package defines a CLI entry point:
```toml
[project.scripts]
reflector-optimize = "reflector_position.cli:main"
```

After installation, you can run:
```bash
reflector-optimize scene.xml --method gradient-descent
```

## Migration from Notebook

The code has been refactored from [building_floor.ipynb](../notebooks/building_floor.ipynb):

### What was migrated:

1. **Helper functions** → `metrics.py` and `utils.py`
   - RSS computation functions
   - Metric calculations
   - Radio map utilities

2. **Optimizer classes** → `optimizers/`
   - `GridSearchAPOptimizer` → `grid_search.py`
   - `GradientDescentAPOptimizer` → `gradient_descent.py`

3. **Scene setup** → `scene_setup.py`
   - Scene loading logic
   - Transmitter/receiver configuration
   - Camera setup

4. **Configuration** → `config.py`
   - Dataclasses for type safety
   - Default values
   - Parameter validation

### What was added:

1. **CLI interface** (`cli.py`)
   - Argument parsing
   - Method selection
   - Progress reporting

2. **Configuration management** (`config.py`)
   - Structured configuration
   - Type hints
   - Validation

3. **Package structure**
   - Proper imports
   - Public API
   - Documentation

4. **Examples**
   - Standalone scripts
   - Different use cases
   - Best practices

## Design Principles

### Clean Code

1. **Separation of Concerns**
   - Each module has single responsibility
   - Optimizers separated from metrics
   - Configuration separated from logic

2. **Type Hints**
   - All public functions have type annotations
   - Dataclasses for configuration
   - Better IDE support

3. **Documentation**
   - Docstrings for all public functions
   - Usage examples in README
   - Comprehensive guides

4. **Error Handling**
   - Validation of inputs
   - Clear error messages
   - Graceful degradation

### Senior Engineer Practices

1. **Modularity**
   - Reusable components
   - Clear interfaces
   - Minimal coupling

2. **Testability**
   - Functions are pure where possible
   - Configuration via dependency injection
   - Mockable dependencies

3. **Maintainability**
   - Consistent naming
   - Clear structure
   - Version control ready

4. **Usability**
   - CLI for quick experiments
   - API for integration
   - Examples for learning

## Usage Patterns

### Development Workflow

```bash
# Install in editable mode
pip install -e .

# Run quick test
python examples/quick_test.py

# Use CLI for experiments
reflector-optimize scene.xml --method gradient-descent --gd-iterations 5

# Format code
black src/ examples/

# Check code quality
ruff check src/
```

### Integration Pattern

```python
# Import as library
from reflector_position import (
    setup_building_floor_scene,
    GradientDescentAPOptimizer,
    GridSearchAPOptimizer,
)

# Use in your code
scene = setup_building_floor_scene("scene.xml")
optimizer = GradientDescentAPOptimizer(scene, (20, 20))
result = optimizer.optimize(num_iterations=10)
```

### Extension Pattern

```python
# Extend optimizers
from reflector_position.optimizers import GradientDescentAPOptimizer

class MyCustomOptimizer(GradientDescentAPOptimizer):
    def compute_loss(self, ...):
        # Custom loss function
        pass
```

## Future Enhancements

Potential additions (not yet implemented):

1. **Testing**
   - Unit tests in `tests/`
   - Integration tests
   - CI/CD pipeline

2. **Advanced Features**
   - Multi-objective optimization
   - Constrained optimization
   - Parallel evaluation

3. **Visualization**
   - Interactive plots
   - 3D visualization
   - Animation of optimization

4. **Performance**
   - Caching
   - Distributed computation
   - GPU batching

5. **Documentation**
   - API reference (Sphinx)
   - Tutorials
   - Performance benchmarks
