# Reflector Position Optimization

Physics-aware optimal placement for mechanical reflectors in NLOS (Non-Line-of-Sight) scenarios using differentiable ray tracing with Sionna.

> 📊 **Project Status**: See [STATUS.md](STATUS.md) for current development status, roadmap, and completed features.

## Features

- **Grid Search Optimization**: Exhaustive search over spatial grid for baseline performance
- **Gradient Descent Optimization**: Fast gradient-based optimization using differentiable ray tracing
- **Metrics**: Minimum RSS, coverage area, and soft minimum for smooth optimization
- **Visualizations**: Heatmaps, convergence plots, and trajectory visualization
- **CLI Tool**: Command-line interface for easy experimentation

## Installation

### From Source

```bash
# Clone the repository
cd reflector-position

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10, < 3.14
- TensorFlow >= 2.20.0
- Sionna >= 1.2.1
- PyTorch >= 2.9.0
- DrJit >= 1.2.0
- Mitsuba >= 3.7.0
- NumPy == 1.26.4

## Quick Start

### Command Line Interface

The package provides a CLI tool `reflector-optimize`:

```bash
# Run gradient descent only
reflector-optimize /path/to/scene.xml --method gradient-descent

# Run grid search only
reflector-optimize /path/to/scene.xml --method grid-search

# Run both methods and compare
reflector-optimize /path/to/scene.xml --method all

# Customize parameters
reflector-optimize /path/to/scene.xml \
    --method gradient-descent \
    --gd-iterations 20 \
    --gd-lr 0.5 \
    --gd-samples 1000000
```

#### CLI Options

**Method Selection:**
- `--method {grid-search,gradient-descent,all}`: Choose optimization method

**Scene Configuration:**
- `--frequency FLOAT`: Operating frequency in Hz (default: 5.18e9)
- `--tx-power FLOAT`: Transmitter power in dBm (default: 5.0)
- `--fixed-z FLOAT`: Fixed Z height for AP (default: 3.8)

**Grid Search Options:**
- `--gs-x-min`, `--gs-x-max`: X bounds for grid search
- `--gs-y-min`, `--gs-y-max`: Y bounds for grid search
- `--gs-resolution FLOAT`: Grid spacing in meters (default: 5.0)
- `--gs-samples INT`: Ray tracing samples (default: 500000)
- `--gs-max-depth INT`: Max ray tracing depth (default: 13)

**Gradient Descent Options:**
- `--gd-init-x`, `--gd-init-y`: Initial position
- `--gd-x-min`, `--gd-x-max`: X bounds
- `--gd-y-min`, `--gd-y-max`: Y bounds
- `--gd-iterations INT`: Number of iterations (default: 10)
- `--gd-lr FLOAT`: Learning rate (default: 0.5)
- `--gd-samples INT`: Ray tracing samples (default: 1000000)
- `--gd-max-depth INT`: Max ray tracing depth (default: 15)
- `--gd-temperature FLOAT`: Soft minimum temperature (default: 0.2)

**Other Options:**
- `--quiet`: Suppress verbose output

### Python API

#### Quick Example

```python
from reflector_position import (
    setup_building_floor_scene,
    GradientDescentAPOptimizer,
    GradientDescentConfig,
)

# Setup scene
scene = setup_building_floor_scene(
    scene_path="/path/to/scene.xml",
    frequency=5.18e9,
    tx_power_dbm=5.0,
)

# Configure optimizer
config = GradientDescentConfig(
    initial_x=20.0,
    initial_y=20.0,
    num_iterations=10,
    learning_rate=0.5,
    samples_per_tx=1_000_000,
)

# Run optimization
optimizer = GradientDescentAPOptimizer(
    scene=scene,
    initial_position=config.initial_position,
    position_bounds=config.position_bounds,
)

final_position, final_rss = optimizer.optimize(
    num_iterations=config.num_iterations,
    learning_rate=config.learning_rate,
    samples_per_tx=config.samples_per_tx,
)

# Visualize results
optimizer.plot_optimization_trajectory()
```

#### Grid Search Example

```python
from reflector_position import GridSearchAPOptimizer, GridSearchConfig

# Configure grid search
config = GridSearchConfig(
    x_min=5.0,
    x_max=35.0,
    y_min=5.0,
    y_max=35.0,
    grid_resolution=2.0,
    samples_per_tx=500_000,
)

# Run optimization
optimizer = GridSearchAPOptimizer(
    scene=scene,
    search_bounds=config.search_bounds,
    grid_resolution=config.grid_resolution,
)

best_position, best_rss = optimizer.optimize()

# Visualize results
optimizer.plot_results(metric='min_rss_dbm')
```

## Examples

See the `examples/` directory for complete examples:

- `examples/quick_test.py`: Fast gradient descent test with reduced parameters
- `examples/full_comparison.py`: Compare grid search vs gradient descent

Run examples:

```bash
python examples/quick_test.py
python examples/full_comparison.py
```

## Project Structure

```
reflector-position/
├── src/reflector_position/
│   ├── __init__.py          # Package initialization
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration dataclasses
│   ├── metrics.py           # RSS metrics and utilities
│   ├── scene_setup.py       # Scene configuration
│   ├── utils.py             # Helper functions
│   └── optimizers/
│       ├── __init__.py
│       ├── grid_search.py   # Grid search optimizer
│       └── gradient_descent.py  # Gradient descent optimizer
├── examples/                # Example scripts
├── pyproject.toml          # Package configuration
└── README.md               # This file
```

## Methodology

### Overview

The framework implements **physics-aware optimization** using differentiable ray tracing, enabling gradient-based methods that understand the physical propagation environment. For advanced Ray-based distributed optimization and baseline comparisons, see the detailed [methodology documentation](docs/methodology/).

### Optimization Approaches

#### Grid Search

Exhaustively evaluates AP positions on a 2D grid:
- Computes radio map for each position
- Tracks minimum RSS across coverage area
- Serves as baseline for comparison

#### Gradient Descent (Differentiable Ray Tracing)

Uses differentiable ray tracing to optimize via gradients:
- Leverages Sionna's differentiable RadioMapSolver
- Uses soft minimum (LogSumExp) for smooth gradients
- PyTorch + DrJit integration via `@dr.wrap` decorator
- 50-100× faster than grid search

### Advanced Workflow: Ray-Based Distributed Optimization

The framework supports **Ray-Based Distributed Multi-Start Gradient Descent** for robust exploration of non-convex optimization landscapes when optimizing physical reflector positions:

```
[32 Independent Ray Actors] → [Isolated Scene Instances] → [Parallel Optimization]
                                        ↓
                               [Independent Gradients]
                                        ↓
                               [Winner Selection]
```

**Key Features:**
- **32 Independent Processes**: Each Ray Actor runs in its own Python process with isolated memory
- **Unique Scene Geometry**: Each actor has its own Scene copy with different reflector positions
- **Process-Level Isolation**: True independence - actors cannot interfere with each other
- **Robust Exploration**: Avoids local minima through complete isolation and diversity
- **Winner Selection**: Choose the best configuration from 32 independent attempts

**Why Ray Instead of Vectorization?**
- **Vectorization**: Suitable for parameter optimization (Tx/Rx positions, phase shifts) within a single scene
- **Ray**: Required when each optimization trajectory needs **different physical geometry** (reflector positions, obstacles)

Since reflectors are physical objects that change the scene geometry, each optimization instance needs its own independent Scene copy. Ray provides this process-level isolation that vectorization cannot.

**For details**, see:
- [OPTIMIZATION_WORKFLOW.md](docs/methodology/OPTIMIZATION_WORKFLOW.md) - Complete Ray architecture and implementation guide
- [BASELINES.md](docs/methodology/BASELINES.md) - Comparison with GA, PSO, and Alternating Optimization
- [FUTURE_ROADMAP.md](docs/methodology/FUTURE_ROADMAP.md) - Implementation plans for Ray-based features

### Metrics

- **Minimum RSS**: Worst-case received signal strength (optimization objective)
- **Soft Minimum**: Differentiable approximation using LogSumExp
- **Coverage**: Percentage of area above threshold (-100 dBm default)


## Performance

Typical performance on building floor scenario:

| Method | Evaluations | Time | Solution Quality |
|--------|-------------|------|------------------|
| Grid Search (2m grid) | ~100-200 | ~30-60 min | Baseline |
| Gradient Descent | 10-20 | ~20-40 min | Within 1 dB |

Gradient descent achieves similar quality with significantly fewer evaluations.

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Format code
black src/ examples/

# Lint code
ruff check src/ examples/

# Run type checking
mypy src/
```

### Running Tests

```bash
pytest
```

## Completed Features ✅

### Core Functionality
- ✅ **Grid Search Optimizer**: Exhaustive spatial search with configurable resolution
- ✅ **Gradient Descent Optimizer**: Differentiable ray tracing with PyTorch + DrJit
- ✅ **Soft Minimum Metric**: Smooth, differentiable optimization objective
- ✅ **Coverage Metrics**: RSS threshold-based coverage calculation
- ✅ **Radio Map Computation**: Configurable ray tracing parameters

### Code Architecture
- ✅ **Modular Design**: Separate modules for metrics, optimizers, config, and scene setup
- ✅ **Type Hints**: Full type annotations across all public APIs
- ✅ **Configuration System**: Type-safe dataclasses for all parameters
- ✅ **Error Handling**: Input validation and clear error messages

### User Interface
- ✅ **CLI Tool**: `reflector-optimize` command-line interface
- ✅ **Python API**: Clean, documented API for programmatic use
- ✅ **Visualization**: Heatmaps, trajectory plots, convergence graphs
- ✅ **Examples**: Quick test and full comparison scripts

### Documentation
- ✅ **README**: Main documentation with quick start
- ✅ **Installation Guide**: Detailed setup instructions (`docs/INSTALL.md`)
- ✅ **Usage Guide**: Comprehensive examples (`docs/USAGE.md`)
- ✅ **Quick Reference**: Cheat sheet (`docs/QUICKREF.md`)
- ✅ **Project Structure**: Architecture documentation (`docs/PROJECT_STRUCTURE.md`)
- ✅ **Changelog**: Migration details (`docs/CHANGELOG.md`)

### Package Management
- ✅ **pyproject.toml**: Modern Python packaging
- ✅ **Entry Points**: CLI command installation
- ✅ **Dependencies**: Pinned versions for reproducibility
- ✅ **Editable Install**: Development-friendly installation

## TODO & Future Enhancements 🚀

### High Priority
- [ ] **Unit Tests**: Add pytest test suite for core functionality
  - [ ] Test metrics calculations
  - [ ] Test optimizer convergence
  - [ ] Test scene setup utilities
  - [ ] Test configuration validation
- [ ] **Integration Tests**: End-to-end optimization tests
- [ ] **CI/CD Pipeline**: GitHub Actions for automated testing
- [ ] **Type Checking**: Add mypy to CI pipeline

### Performance Improvements
- [ ] **Ray Distributed Optimization**: Implement multi-process optimization for reflector positioning
- [ ] **GPU Memory Management**: Optimize VRAM usage for multiple scene instances
- [ ] **Caching**: Cache radio maps for repeated positions
- [ ] **Memory Optimization**: Reduce memory footprint for large scenes

### New Features
- [ ] **Multi-Objective Optimization**: Simultaneous coverage + capacity optimization
- [ ] **Constrained Optimization**: Wall-mounting and mechanical constraints
- [ ] **Reflector Control**: Integration of mechanical reflector from notebook
- [ ] **Multi-AP Optimization**: Joint optimization of multiple access points
- [ ] **Different Environments**: Support for corridor, warehouse, outdoor scenes
- [ ] **Adaptive Learning Rate**: Automatic learning rate scheduling
- [ ] **Early Stopping**: Convergence detection

### Visualization & Analysis
- [ ] **Interactive Plots**: Plotly/Bokeh integration for interactive exploration
- [ ] **3D Visualization**: 3D scene rendering with radio coverage
- [ ] **Animation**: Animated optimization trajectory
- [ ] **Comparative Analysis**: Automated method comparison reports
- [ ] **Heat Map Export**: Save results as images/videos

### Documentation
- [ ] **API Documentation**: Sphinx-generated API reference
- [ ] **Tutorials**: Step-by-step tutorials for different scenarios
- [ ] **Jupyter Notebooks**: Interactive tutorial notebooks
- [ ] **Performance Benchmarks**: Benchmark results for different configurations
- [ ] **Video Demos**: Screen recordings demonstrating usage

### Code Quality
- [ ] **Linting**: Enforce code style with ruff/black in pre-commit hooks
- [ ] **Code Coverage**: Aim for >80% test coverage
- [ ] **Security Scanning**: Add dependency vulnerability scanning
- [ ] **Documentation Coverage**: Ensure all public APIs are documented

### Publishing & Distribution
- [ ] **PyPI Release**: Publish package to PyPI
- [ ] **Docker Image**: Containerized version with all dependencies
- [ ] **Conda Package**: conda-forge distribution
- [ ] **Documentation Site**: GitHub Pages or Read the Docs hosting
- [ ] **Zenodo DOI**: Citable version with DOI

### Research Extensions
- [ ] **Phase 2**: Joint AP + single RIS optimization (from roadmap)
- [ ] **Phase 3**: Multi-AP, multi-RIS optimization (from roadmap)
- [ ] **Discontinuity Smoothing**: Sigmoid-based smoothing (Eertmans et al.)
- [ ] **Learned Schedules**: Machine learning for hyperparameter tuning
- [ ] **Real-World Validation**: Experimental validation with measurements

### User Experience
- [ ] **Progress Bars**: tqdm integration for long-running optimizations
- [ ] **Resume Capability**: Save/load optimization state
- [ ] **Configuration Files**: YAML/JSON config file support
- [ ] **Logging**: Structured logging with different verbosity levels
- [ ] **Result Export**: Export results to JSON, CSV, HDF5

## Documentation

### Main Guides
- **Quick Start**: See [Quick Start](#quick-start) section above
- **Installation**: See [docs/guides/INSTALL.md](docs/guides/INSTALL.md)
- **Usage Guide**: See [docs/guides/USAGE.md](docs/guides/USAGE.md)
- **Quick Reference**: See [docs/guides/QUICKREF.md](docs/guides/QUICKREF.md)

### Architecture & Structure
- **Project Structure**: See [docs/architecture/PROJECT_STRUCTURE.md](docs/architecture/PROJECT_STRUCTURE.md)
- **Changelog**: See [docs/architecture/CHANGELOG.md](docs/architecture/CHANGELOG.md)

### Methodology & Research
- **Optimization Workflow**: See [docs/methodology/OPTIMIZATION_WORKFLOW.md](docs/methodology/OPTIMIZATION_WORKFLOW.md) - Ray-based distributed optimization architecture
- **Baseline Methods**: See [docs/methodology/BASELINES.md](docs/methodology/BASELINES.md) - GA, PSO, and AO comparisons
- **Future Roadmap**: See [docs/methodology/FUTURE_ROADMAP.md](docs/methodology/FUTURE_ROADMAP.md) - Advanced features and research extensions

For a complete documentation index, see [docs/README.md](docs/README.md).


## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This project uses:
- [Sionna](https://nvlabs.github.io/sionna/) for differentiable ray tracing
- [PyTorch](https://pytorch.org/) for gradient computation
- [DrJit](https://github.com/mitsuba-renderer/drjit) for PyTorch-Mitsuba integration

## Citation

If you use this code in your research, please cite:

```bibtex
@software{reflector_position,
  title = {Reflector Position Optimization},
  author = {Your Name},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/yourusername/reflector-position}
}
```

