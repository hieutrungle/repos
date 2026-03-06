# Reflector Position Optimization

Physics-aware optimal placement for mechanical reflectors in NLOS (Non-Line-of-Sight) scenarios using differentiable ray tracing with Sionna.

> 📊 **Project Status**: See [STATUS.md](STATUS.md) for current development status, roadmap, and completed features.

## Features

- **Grid Search Optimization**: Exhaustive search over spatial grid for baseline performance; supports 1-AP, 2-AP alternating, and 2-AP + reflector modes
- **Gradient Descent Optimization**: Fast gradient-based optimization using differentiable ray tracing; joint AP position, orientation, and reflector placement
- **Genetic Algorithm (DEAP)**: Evolutionary optimization with population-based search; 12-gene chromosome encoding AP positions, orientations, and reflector parameters
- **Reflector-Aware Optimization**: All three methods support passive reflector placement and focal-point aiming on wall surfaces, with shadow-robust objectives (5th-percentile RSS)
- **Ray-Parallel Execution**: Distributed evaluation via Ray ActorPool — all three methods run in parallel across persistent GPU workers
- **Experiment Runner**: Config-driven batch hyperparameter sweeps across GD / GS / GA with JSON configs (259 production trials, 19 smoke-test trials)
- **Three Optimization Modes**: `1ap` (single AP), `2ap` (dual AP), `2ap_reflector` (dual AP + passive reflector)
- **Inversion of Control (IoC) Architecture**: Clean separation of algorithm logic (DEAP) from execution engine (Ray) via dependency injection
- **Metrics**: 5th-percentile RSS, soft minimum (LogSumExp), masked soft minimum (shadow-aware), percentile coverage objective, and differentiable coverage loss
- **Visualizations**: Heatmaps, convergence plots, trajectory visualization, GA evolution plots, and Hall of Fame
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
- `examples/ray_parallel_example.py`: Ray-parallel GD, GS, and GA across 1-AP, 2-AP, and 2-AP + reflector modes via ActorPool
- `examples/run_ga_modular.py`: **Modular GA** — DEAP genetic algorithm with Ray-parallel fitness evaluation (IoC pattern)
- `examples/ray_experiment_runner.py`: Unified config-driven experiment runner for hyperparameter sweeps (one method per trial)
- `examples/ray_experiment_runner_config.example.json`: Production config (259 trials across GD / GS / GA sweeps)
- `examples/ray_experiment_runner_config.smoke_test.json`: Quick validation config (19 trials)

Run examples:

```bash
python examples/quick_test.py
python examples/full_comparison.py
python examples/ray_parallel_example.py
python examples/run_ga_modular.py
python examples/ray_experiment_runner.py --config examples/ray_experiment_runner_config.example.json
```

## Ray Runner (Unified)

`examples/ray_experiment_runner.py` is the single entrypoint for Ray-based hyperparameter automation.

- Runs one method per trial (`gd`, `gs`, or `ga`) in one of three modes (`1ap`, `2ap`, `2ap_reflector`)
- Supports explicit trials and automatic hyperparameter sweeps via `sweep_groups` (Cartesian grid)
- Reflector-aware trials automatically configure wall geometry, focal point, and shadow-robust objectives
- Saves per-trial logs and consolidated summaries (`summary.csv`, `summary.json`, `all_trials_detailed.json`)

### Step 1: Generate explicit trial config (recommended)

```bash
python examples/ray_experiment_runner.py \
    --config examples/ray_experiment_runner_config.example.json \
    --generate-only \
    --generated-config results/generated_trials.json
```

### Step 2: Run hyperparameter optimization

Using the original config:

```bash
python examples/ray_experiment_runner.py \
    --config examples/ray_experiment_runner_config.example.json \
    --output-root results/experiments
```

Using the generated explicit trial config:

```bash
python examples/ray_experiment_runner.py \
    --config results/generated_trials.json \
    --output-root results/experiments
```

### Output location

Each run is stored under:

- `results/experiments/ray_experiments_<timestamp>/`

For full configuration details and tuning patterns, see [docs/guides/RAY_EXPERIMENT_RUNNER.md](docs/guides/RAY_EXPERIMENT_RUNNER.md).

## Project Structure

```
reflector-position/
├── src/reflector_position/
│   ├── __init__.py              # Package initialization
│   ├── cli.py                   # Command-line interface
│   ├── config.py                # Configuration dataclasses
│   ├── metrics.py               # RSS metrics and utilities
│   ├── scene_setup.py           # Scene configuration
│   ├── utils.py                 # Helper functions
│   └── optimizers/
│       ├── __init__.py
│       ├── base_optimizer.py        # Abstract base class
│       ├── gradient_descent.py      # Gradient descent (differentiable RT)
│       ├── grid_search.py           # Grid search + SinglePointGridSearch
│       ├── optimizer_factory.py     # Factory pattern for optimizer creation
│       ├── ray_parallel_optimizer.py # ActorPool orchestrator + OptimizationWorker
│       ├── ray_evaluator.py         # Generic Ray execution engine (IoC)
│       ├── deap_logic.py            # Pure DEAP GA logic (no Ray imports)
│       └── ray_deap_optimizer.py    # Monolithic DEAP+Ray (legacy)
│   ├── reflector_model.py           # ReflectorController + mesh creation
├── examples/
│   ├── ray_parallel_example.py  # GD, GS, GA across 1ap/2ap/2ap_reflector
│   ├── run_ga_modular.py        # Modular GA entry point (IoC)
│   ├── ray_experiment_runner.py # Config-driven batch experiment runner
│   └── ...                      # Other examples
├── docs/                        # Comprehensive documentation
├── tests/                       # pytest test suite (62 tests)
├── pyproject.toml               # Package configuration
└── README.md                    # This file
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
- Joint optimisation of AP position, orientation, and reflector placement (wall UV + focal point)
- Multiple loss functions: `softmin`, `masked_softmin` (shadow-aware), `percentile`, `auto`
- 50-100× faster than grid search

#### Genetic Algorithm (DEAP Library) ✅

Evolutionary optimisation using the DEAP framework:
- **Population-based search**: 50-200 individuals with configurable chromosome encoding
- **12-gene reflector chromosome**: `[x1, y1, x2, y2, dir1_x, dir1_y, dir2_x, dir2_y, refl_u, refl_v, focal_x, focal_y]`
- **Operators**: Blend crossover (`cxBlend`), split Gaussian mutation (separate σ for position/direction/reflector), tournament selection
- **Maximises 5th-percentile RSS** (linear Watts) as fitness; shadow-robust for reflector scenarios
- **Separation constraint**: penalises overlapping APs before expensive ray-tracing
- **Ray-parallel evaluation**: each individual evaluated via `SinglePointGridSearchOptimizer` on Ray ActorPool
- **Modular IoC architecture**: algorithm logic (no Ray imports) separated from execution engine

### Ray-Based Distributed Optimization ✅

All three methods (GD, GS, GA) run on a shared **Ray ActorPool** infrastructure:

```
┌─────────────────────────────────────────────────────────┐
│              Driver (Algorithm Logic)                    │
│  Gradient Descent / Grid Search / DEAP GA               │
│         │                                               │
│         │  toolbox.map(evaluate, population)             │
│         ▼                                               │
│  ┌─────────────────────────────┐                        │
│  │  RayActorPoolExecutor.map() │  ← Dependency Injection│
│  │  (pool.map — ordered, sync) │                        │
│  └──────────┬──────────────────┘                        │
│             │                                           │
│    ┌────────┼────────┬────────┐                         │
│    ▼        ▼        ▼        ▼                         │
│  Worker0  Worker1  Worker2  Worker3                     │
│  (Scene)  (Scene)  (Scene)  (Scene)                     │
│  GPU 0.25 GPU 0.25 GPU 0.25 GPU 0.25                   │
└─────────────────────────────────────────────────────────┘
```

**Key Features:**
- **ActorPool pattern**: Fixed pool of persistent workers; Scene loaded once per worker
- **Ordered synchronous map** (`pool.map`): prevents freeze issues from `map_unordered`
- **GPU efficiency**: Configurable fraction per worker (0.25 = 4 workers/GPU)
- **IoC pattern**: Algorithm logic knows nothing about Ray; uses injected `map` function
- **Three optimisation methods**: GD (multi-start), GS (true parallel), GA (DEAP evolutionary)

**Quick Example — Modular GA:**
```python
import ray
from reflector_position.optimizers import RayActorPoolExecutor, GeneticAlgorithmRunner

ray.init()
executor = RayActorPoolExecutor(scene_config={...}, num_workers=4, gpu_fraction=0.25)

ga = GeneticAlgorithmRunner(
    position_bounds={"x_min": 5, "x_max": 25, "y_min": 5, "y_max": 25},
    fixed_z=3.8,
    executor_map=executor.map,  # Dependency Injection
)

results = ga.run(
    optimization_params={"samples_per_tx": 1_000_000, "max_depth": 13},
    ga_params={"pop_size": 50, "n_gen": 20},
    seed=42,
)

print(f"Best: {results['best_position']}  RSS: {results['best_fitness_dbm']:.2f} dBm")
executor.shutdown()
```

**For complete details**, see:
- [RAY_PARALLEL_GUIDE.md](docs/methodology/RAY_PARALLEL_GUIDE.md) - Complete guide with examples
- [RAY_ARCHITECTURE.md](docs/methodology/RAY_ARCHITECTURE.md) - Why Ray vs vectorization
- [OPTIMIZATION_WORKFLOW.md](docs/methodology/OPTIMIZATION_WORKFLOW.md) - Complete architecture
- [RAY_IMPLEMENTATION_SUMMARY.md](docs/methodology/RAY_IMPLEMENTATION_SUMMARY.md) - Implementation status
- [BASELINES.md](docs/methodology/BASELINES.md) - Comparison with GA, PSO, and Alternating Optimization

### Reflector-Aware Optimization ✅

All three methods support joint AP + passive reflector placement in `2ap_reflector` mode:

- **Reflector parameterisation**: wall-surface UV coordinates `(u, v)` ∈ [0, 1]² + 3-D focal-point aiming
- **Grid Search**: outer loop sweeps reflector UV × focal-target grid; inner loop alternates AP positions
- **Gradient Descent**: `torch.sigmoid`-bounded raw parameters for reflector UV and focal point; differentiable through Sionna scene graph
- **Genetic Algorithm**: 4 extra genes `[refl_u, refl_v, focal_x, focal_y]` appended to AP chromosome; split mutation with separate σ for reflector genes
- **Shadow-robust objective**: 5th-percentile RSS (`PercentileCoverageObjective`) ignores the ~2-5% of cells shadowed by the reflector body
- **Scene integration**: `ReflectorController` manages mesh creation, wall placement, and focal-point orientation at each evaluation

For details see [docs/methodology/RAY_PARALLEL_GUIDE.md](docs/methodology/RAY_PARALLEL_GUIDE.md).

### Metrics

- **5th-Percentile RSS** (P5): Primary objective — robust worst-case coverage ignoring reflector shadows
- **Soft Minimum**: Differentiable approximation using LogSumExp
- **Masked SoftMin**: Shadow-aware variant that down-weights dead-zone cells
- **Percentile Coverage**: Configurable quantile-based objective (`PercentileCoverageObjective`)
- **Coverage**: Percentage of area above threshold (-100 dBm default)


## Performance

Typical performance on building floor scenario:

| Method | Evaluations | Time | Solution Quality |
|--------|-------------|------|------------------|
| Grid Search (2m grid) | ~100-200 | ~30-60 min | Baseline |
| Gradient Descent | 10-20 | ~20-40 min | Within 1 dB |
| Ray Parallel GD (4 workers, 64 tasks) | 640 | ~10-20 min | Best of 64 starts |
| Ray Parallel GS (4 workers, 441 pts) | 441 | ~5-15 min | Exhaustive |
| DEAP GA (pop=50, 20 gen, 4 workers) | ~700-1000 | ~15-30 min | Population-optimal |

Gradient descent achieves similar quality with significantly fewer evaluations. The DEAP GA explores the search space more broadly via an evolving population, complementing gradient-based methods. All three methods leverage the same Ray ActorPool for parallel evaluation with near-linear speedup.

**Testing Status**: Core optimizers validated with 62 unit and integration tests (82-92% coverage). Ray parallel and GA implementations complete.

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
- ✅ **Reflector-Aware Optimization**: Joint AP + reflector placement for all 3 methods (GD, GS, GA) with shadow-robust P5 RSS objective
- ✅ **Optimizer Factory**: Factory pattern for creating optimizers
- ✅ **Base Optimizer ABC**: Abstract base class enforcing optimizer interface

### Ray-Based Parallel Optimization ✅
- ✅ **RayParallelOptimizer**: ActorPool orchestrator for distributed multi-start GD and parallel GS
- ✅ **OptimizationWorker**: Persistent Ray actor with reusable Scene + optional ReflectorController
- ✅ **RayActorPoolExecutor**: Generic execution engine with ordered `pool.map` (IoC pattern)
- ✅ **GeneticAlgorithmRunner**: Pure DEAP GA logic — no Ray imports, uses injected `map`; 12-gene reflector chromosome
- ✅ **SinglePointGridSearchOptimizer**: Evaluates single (x, y) position for GA fitness
- ✅ **Experiment Runner**: Config-driven batch runner with JSON schema, sweep groups, and CSV/JSON summaries
- ✅ **Three Modes**: `1ap`, `2ap`, `2ap_reflector` — all methods × all modes
- ✅ **GPU Management**: Configurable GPU fraction per worker (0.25 = 4 workers/GPU)
- ✅ **Multi-GPU Validation**: Parallel Ray execution validated on multi-GPU setup
- ✅ **Non-Ray Validation**: Baseline single-process execution validated for reflector-aware paths
- ✅ **Freeze-safe**: Uses `pool.map` (ordered, synchronous) instead of `map_unordered`
- ✅ **Comprehensive Documentation**: Detailed guides with examples

### Testing & Quality Assurance ✅
- ✅ **Unit Tests**: 62 tests across 4 test files
- ✅ **Test Coverage**: 82-92% core coverage, 100% factory coverage
- ✅ **Test Framework**: pytest with markers (unit, integration, slow)
- ✅ **Shared Fixtures**: Efficient scene setup and reuse
- ✅ **Fast Execution**: ~10s for full test suite
- ✅ **Test Documentation**: Comprehensive guides in `docs/tests/`

### Code Architecture
- ✅ **Modular Design**: Separate modules for metrics, optimizers, config, and scene setup
- ✅ **Type Hints**: Full type annotations across all public APIs
- ✅ **Configuration System**: Type-safe dataclasses for all parameters
- ✅ **Error Handling**: Input validation and clear error messages
- ✅ **Factory Pattern**: Extensible optimizer creation system

### User Interface
- ✅ **CLI Tool**: `reflector-optimize` command-line interface
- ✅ **Python API**: Clean, documented API for programmatic use
- ✅ **Visualization**: Heatmaps, trajectory plots, convergence graphs
- ✅ **Examples**: Quick test, full comparison, Ray parallel examples

### Documentation
- ✅ **README**: Main documentation with quick start
- ✅ **Installation Guide**: Detailed setup instructions (`docs/guides/INSTALL.md`)
- ✅ **Usage Guide**: Comprehensive examples (`docs/guides/USAGE.md`)
- ✅ **Quick Reference**: Cheat sheet (`docs/guides/QUICKREF.md`)
- ✅ **Project Structure**: Architecture documentation (`docs/architecture/PROJECT_STRUCTURE.md`)
- ✅ **Changelog**: Migration details (`docs/architecture/CHANGELOG.md`)
- ✅ **Ray Parallel Guide**: Complete guide with examples (`docs/methodology/RAY_PARALLEL_GUIDE.md`)
- ✅ **Ray Architecture**: Why Ray vs vectorization (`docs/methodology/RAY_ARCHITECTURE.md`)
- ✅ **Test Documentation**: Testing guides and summaries (`docs/tests/`)
- ✅ **Methodology**: Optimization workflow and baselines

### Package Management
- ✅ **pyproject.toml**: Modern Python packaging
- ✅ **Entry Points**: CLI command installation
- ✅ **Dependencies**: Pinned versions for reproducibility (including Ray)
- ✅ **Editable Install**: Development-friendly installation

## TODO & Future Enhancements 🚀

### High Priority
- [x] **Unit Tests**: Add pytest test suite for core functionality ✅
  - [x] Test metrics calculations ✅
  - [x] Test optimizer convergence ✅
  - [x] Test scene setup utilities ✅
  - [x] Test configuration validation ✅
  - [x] Test factory pattern ✅
  - [x] Test base optimizer ABC ✅
- [x] **Integration Tests**: End-to-end optimization tests ✅
- [ ] **Ray Parallel Tests**: Unit and integration tests for Ray implementation
  - [ ] Test RayParallelOptimizer initialization and configuration
  - [ ] Test OptimizationWorker spawning and lifecycle
  - [ ] Test result aggregation and winner selection
  - [ ] Test GPU fraction allocation
  - [ ] Test error handling and recovery
- [ ] **CLI Tests**: Test command-line interface functionality
- [ ] **CI/CD Pipeline**: GitHub Actions for automated testing
- [ ] **Type Checking**: Add mypy to CI pipeline

### Performance Improvements
- [x] **Ray Distributed Optimization**: Implement multi-process optimization for reflector positioning ✅
  - [x] RayParallelOptimizer with configurable workers ✅
  - [x] OptimizationWorker with process isolation ✅
  - [x] GPU fraction management ✅
  - [ ] Testing and validation ⏳
- [x] **GPU Memory Management**: Configurable VRAM usage per worker ✅
- [ ] **Caching**: Cache radio maps for repeated positions
- [ ] **Memory Optimization**: Reduce memory footprint for large scenes
- [ ] **Async Result Collection**: Stream results as they complete
- [ ] **Checkpointing**: Save intermediate results for long-running optimizations

### New Features
- [ ] **Multi-Objective Optimization**: Simultaneous coverage + capacity optimization
- [ ] **Constrained Optimization**: Wall-mounting and mechanical constraints
- [x] **Reflector Control**: Mechanical reflector initialization and control integrated into main optimization path ✅
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

## Roadmap

### Phase 1: Core Functionality ✅ COMPLETE
- Grid search baseline
- Gradient descent optimization
- Basic visualization
- Package structure
- Documentation
- Unit and integration tests (62 tests, 82-92% coverage)
- Factory pattern and base classes

**Status**: ✅ Complete (January 2026)

### Phase 2: Ray-Based Parallel Optimization ✅ COMPLETE
- Ray distributed architecture (ActorPool pattern)
- Multi-start gradient descent (64 tasks → 4 workers)
- True parallel grid search (441 single-point tasks)
- DEAP genetic algorithm with Ray-parallel fitness evaluation
- Inversion of Control (IoC) architecture: `deap_logic.py` + `ray_evaluator.py`
- Ordered `pool.map` replacing `map_unordered` (prevents freezes)
- GPU memory management with configurable fraction
- Comprehensive documentation

**Status**: ✅ Complete (February 2026)

### Phase 3: Testing & Validation (Q1 2026) 🚧 ONGOING
- [x] Core optimizer unit tests (62 tests) ✅
- [x] Integration tests ✅
- [x] Test documentation ✅
- [ ] Ray parallel tests
- [ ] GA / DEAP tests
- [ ] CLI tests
- [ ] CI/CD pipeline
- [ ] Performance benchmarks

**Status**: 🚧 Core Tests Complete, Ray + GA Tests Pending  
**Started**: January 2026  
**Target**: February 2026

### Phase 4: Advanced Features (Q1-Q2 2026)
- Multi-objective optimization
- ✅ Reflector-aware optimization for all 3 methods (GD, GS, GA) with shadow-robust P5 objective
- ✅ Config-driven experiment runner with hyperparameter sweep support
- Multi-AP optimization (beyond 2-AP)
- Adaptive learning rate
- Coarse-to-fine Ray-based search
- Hybrid GA+GD (seed GD from GA best solutions)

**Status**: 🚧 In progress (reflector optimization complete for all methods; advanced extensions pending)  
**Target**: March-April 2026

### Phase 5: Publishing & Release (Q2 2026)
- PyPI publication
- Documentation site
- Tutorial materials
- Video demonstrations
- v1.0.0 release

**Status**: 📋 Planned  
**Target**: May 2026

## Documentation

### Main Guides
- **Quick Start**: See [Quick Start](#quick-start) section above
- **Installation**: See [docs/guides/INSTALL.md](docs/guides/INSTALL.md)
- **Usage Guide**: See [docs/guides/USAGE.md](docs/guides/USAGE.md)
- **Quick Reference**: See [docs/guides/QUICKREF.md](docs/guides/QUICKREF.md)
- **Ray Experiment Runner**: See [docs/guides/RAY_EXPERIMENT_RUNNER.md](docs/guides/RAY_EXPERIMENT_RUNNER.md)

### Architecture & Structure
- **Project Structure**: See [docs/architecture/PROJECT_STRUCTURE.md](docs/architecture/PROJECT_STRUCTURE.md)
- **Changelog**: See [docs/architecture/CHANGELOG.md](docs/architecture/CHANGELOG.md)

### Methodology & Research
- **Optimization Workflow**: See [docs/methodology/OPTIMIZATION_WORKFLOW.md](docs/methodology/OPTIMIZATION_WORKFLOW.md) - Ray-based distributed optimization architecture
- **Ray Parallel Guide**: See [docs/methodology/RAY_PARALLEL_GUIDE.md](docs/methodology/RAY_PARALLEL_GUIDE.md) - Complete guide to Ray parallel optimization
- **Ray Architecture**: See [docs/methodology/RAY_ARCHITECTURE.md](docs/methodology/RAY_ARCHITECTURE.md) - Why Ray vs vectorization
- **Ray Implementation**: See [docs/methodology/RAY_IMPLEMENTATION_SUMMARY.md](docs/methodology/RAY_IMPLEMENTATION_SUMMARY.md) - Implementation summary
- **Baseline Methods**: See [docs/methodology/BASELINES.md](docs/methodology/BASELINES.md) - GA, PSO, and AO comparisons
- **Future Roadmap**: See [docs/methodology/FUTURE_ROADMAP.md](docs/methodology/FUTURE_ROADMAP.md) - Advanced features and research extensions

### Testing & Quality
- **Test Hub**: See [docs/tests/README.md](docs/tests/README.md) - Complete testing documentation
- **Test Summary**: See [docs/tests/TEST_SUMMARY.md](docs/tests/TEST_SUMMARY.md) - Test statistics (62 tests, 82-92% coverage)
- **Testing Guide**: See [docs/tests/TESTING_GUIDE.md](docs/tests/TESTING_GUIDE.md) - How to run tests
- **Test Categories**: See [docs/tests/TEST_CATEGORIES.md](docs/tests/TEST_CATEGORIES.md) - Test organization

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

