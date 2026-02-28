# Ray Parallel Optimization Guide

**Date**: February 27, 2026
**Status**: Implementation Complete (GD + GS + GA, reflector-aware)

Comprehensive guide to using Ray for distributed parallel optimization of
access-point (AP) positions and intelligent reflecting surface (IRS)
reflector placement.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Running Experiments](#running-experiments)
5. [Optimization Methods](#optimization-methods)
6. [Reflector-Aware Modes](#reflector-aware-modes)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

---

## Overview

The Ray framework provides distributed parallel execution for three
optimization methods:

| Method | Class | Pattern | Use Case |
|--------|-------|---------|----------|
| **Gradient Descent** | `RayParallelOptimizer` | Multi-start random restarts | Continuous parameter search |
| **Grid Search** | `RayParallelOptimizer` | One grid point per task | Exhaustive evaluation |
| **Genetic Algorithm** | `RayActorPoolExecutor` + `GeneticAlgorithmRunner` | DEAP population evaluation | Global search with evolution |

### Key Features

- **ActorPool pattern**: Fixed pool of N workers processes M >> N tasks.
- **Scene reuse**: Each worker loads the heavy Scene once, reuses it.
- **GPU efficiency**: Configurable `gpu_fraction` per worker.
- **Reflector-aware**: All methods support joint AP + reflector optimization.
- **IoC architecture**: GA logic is fully decoupled from Ray.
- **Experiment runner**: Config-driven batch execution with hyperparameter sweeps.

---

## Architecture

### Components

```
src/reflector_position/optimizers/
  ray_parallel_optimizer.py   # OptimizationWorker actor + RayParallelOptimizer orchestrator
  ray_evaluator.py            # RayActorPoolExecutor (generic pool.map for DEAP)
  deap_logic.py               # GeneticAlgorithmRunner (pure DEAP, no Ray imports)
  optimizer_factory.py        # OptimizerFactory.create() — used by workers

examples/
  ray_parallel_example.py     # All optimization functions (GD, GS, GA) for 1ap/2ap/2ap_reflector
  ray_experiment_runner.py    # Config-driven batch runner with sweep groups
  ray_experiment_runner_config.example.json    # Full production config (259 trials)
  ray_experiment_runner_config.smoke_test.json # Minimal config for validation (19 trials)
  run_ga_modular.py           # Standalone GA entry point (IoC demo)
```

### Data Flow

```
                      ray_experiment_runner.py
                              |
                  (reads JSON config, expands sweeps)
                              |
                    ray_parallel_example.py
                     /        |         \
                GD (SPSA)   GS (grid)    GA (DEAP)
                    |         |            |
             RayParallel  RayParallel  RayActorPool
              Optimizer    Optimizer    Executor
                    \         |         /
                     OptimizationWorker (Ray Actor)
                              |
                    OptimizerFactory.create()
                              |
                    Scene + Optimizer + Ray tracing
```

---

## Quick Start

### Prerequisites

```bash
source .venv/bin/activate
pip install ray[default]

# Verify imports
python -c "import ray; from reflector_position.optimizers.ray_parallel_optimizer import RayParallelOptimizer; print('OK')"
```

### Minimal Example (Gradient Descent)

```python
import ray
from reflector_position.optimizers.ray_parallel_optimizer import (
    RayParallelOptimizer,
    generate_random_initial_positions,
)

ray.init(ignore_reinit_error=True)

scene_config = {
    "scene_path": "/path/to/building_floor.xml",
    "frequency": 5.18e9,
    "tx_power_dbm": 5.0,
}

bounds = {"x_min": 5.5, "x_max": 34.5, "y_min": 5.5, "y_max": 34.5}
positions = generate_random_initial_positions(32, bounds, seed=42)

parallel_opt = RayParallelOptimizer(num_workers=4, gpu_fraction=0.5)

work_items = [
    {"initial_position": (pos[0], pos[1]), "position_bounds": bounds}
    for pos in positions
]

results = parallel_opt.run(
    scene_config=scene_config,
    optimizer_method="gradient_descent",
    work_items=work_items,
    optimization_params={
        "num_iterations": 30,
        "learning_rate": 0.5,
        "samples_per_tx": 1_000_000,
        "max_depth": 13,
        "verbose": False,
    },
)

best = results["best_result"]
print(f"Best position: {best['best_position']}")
print(f"Best P5 RSS:   {best['best_metric_dbm']:.2f} dBm")
print(f"Speedup:       {results['aggregate_stats']['speedup']:.1f}x")

parallel_opt.shutdown()
ray.shutdown()
```

### Minimal Example (Genetic Algorithm)

```python
import ray
from reflector_position.optimizers.ray_evaluator import RayActorPoolExecutor
from reflector_position.optimizers.deap_logic import GeneticAlgorithmRunner

ray.init(ignore_reinit_error=True)

scene_config = {
    "scene_path": "/path/to/building_floor.xml",
    "frequency": 5.18e9,
    "tx_power_dbm": 5.0,
    "tx_positions": [(7.0, 7.0, 3.8), (23.0, 23.0, 3.8)],
}

executor = RayActorPoolExecutor(
    scene_config=scene_config,
    num_workers=2,
    gpu_fraction=0.5,
)

ga = GeneticAlgorithmRunner(
    position_bounds={"x_min": 5.5, "x_max": 34.5, "y_min": 5.5, "y_max": 34.5},
    fixed_z=3.8,
    executor_map=executor.map,
    optimize_orientation=True,
    num_aps=2,
    min_ap_separation=5.0,
)

results = ga.run(
    optimization_params={"samples_per_tx": 1_000_000, "max_depth": 13, "verbose": False},
    ga_params={"pop_size": 150, "n_gen": 50, "cxpb": 0.7, "mutpb": 0.3},
    random_seed=4,
)

print(f"Best fitness: {results['best_fitness_dbm']:.2f} dBm")
executor.shutdown()
ray.shutdown()
```

---

## Running Experiments

The **experiment runner** is the recommended way to execute hyperparameter
sweeps. See [RAY_EXPERIMENT_RUNNER.md](../guides/RAY_EXPERIMENT_RUNNER.md)
for full documentation.

### Quick Smoke Test

```bash
# Preview expanded trials (no GPU needed)
python examples/ray_experiment_runner.py \
  --config examples/ray_experiment_runner_config.smoke_test.json \
  --generate-only

# Run all 19 smoke-test trials
python examples/ray_experiment_runner.py \
  --config examples/ray_experiment_runner_config.smoke_test.json \
  --output-root results_smoke_test/experiments
```

### Full Production Run

```bash
# 259 trials across GD, GS, GA with AP-only and reflector modes
python examples/ray_experiment_runner.py \
  --config examples/ray_experiment_runner_config.example.json \
  --output-root results/experiments
```

### Config Structure

```jsonc
{
  "shared": {
    "num_pool_workers": 2,
    "gpu_fraction": 0.5,
    "random_seed": 4,
    "gd_num_tasks": 100,
    "gd_num_iterations": 50,
    "ga_min_ap_separation": 7.0,
    "reflector_wall_top_left": [15.0, 34.0, 3.0],
    "reflector_wall_bottom_right": [34.0, 34.0, 1.0],
    "reflector_size": [2.0, 2.0]
  },
  "trials": [
    {
      "name": "gd_2ap_baseline",
      "method": "gd",
      "mode": "2ap",
      "gd_optimization_overrides": { "learning_rate": 0.5 }
    }
  ],
  "sweep_groups": [
    {
      "name_prefix": "gd_lr_sweep",
      "method": "gd",
      "mode": "2ap_reflector",
      "random_seed": [51, 52, 53],
      "grid": {
        "gd_optimization_overrides.learning_rate": [0.3, 0.5, 0.7],
        "gd_optimization_overrides.temperature": [0.1, 0.15, 0.2]
      }
    }
  ]
}
```

Sweep groups generate a Cartesian product: 3 LR × 3 temp × 3 seeds = **27 trials**.

### Config Files

| Config | Trials | Purpose |
|--------|--------|---------|
| `ray_experiment_runner_config.example.json` | 259 | Full production sweep across GD, GS, GA |
| `ray_experiment_runner_config.smoke_test.json` | 19 | Fast validation (~5-15 min) |

### Output Structure

```
results/experiments/ray_experiments_20260227_143052/
  used_config.json              # Copy of input config
  summary.csv                   # One row per trial
  summary.json                  # Same data as JSON
  all_trials_detailed.json      # Full results per trial
  gd_2ap_baseline/              # Per-trial directory
    output.txt                  # Captured stdout/stderr
    trial_record.json           # Config + result
    gd_2ap_results.json         # Method-specific output
```

---

## Optimization Methods

### Gradient Descent (SPSA)

- Multi-start: M random initial positions processed by N workers.
- Each task runs independent gradient descent with SPSA (Simultaneous
  Perturbation Stochastic Approximation).
- Supports AP position, orientation, and repulsion loss for multi-AP.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gd_num_tasks` | 100 | Number of random-restart trajectories |
| `gd_num_iterations` | 50 | Steps per trajectory |
| `gd_samples_per_tx` | 1,000,000 | Ray-tracing samples per step |
| `gd_repulsion_weight` | 0.3 | Multi-AP repulsion loss weight |
| `learning_rate` | 0.5 | SPSA step size (in `gd_optimization_overrides`) |
| `temperature` | 0.15 | Soft-min temperature |
| `gd_fairness_loss_type` | `"auto"` | `auto`, `softmin`, `masked_softmin`, `percentile` |

### Grid Search (Alternating)

- 1-AP: exhaustive grid over (x, y) positions.
- 2-AP: alternating optimisation — sweep AP1 with AP2 fixed, then swap.
- 2-AP + Reflector: outer loop over reflector (u, v, focal-target),
  inner loop alternates APs.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gs_grid_resolution` | 1.0 | Grid spacing in metres |
| `gs_num_rounds` | 3 | Alternating sweeps per outer round |
| `gs_outer_rounds` | 3 | AP-sweep / reflector-sweep outer loops |
| `gs_u_steps` / `gs_v_steps` | 3 | Reflector wall-surface grid divisions |
| `gs_target_resolution` | 10.0 | Focal-target grid spacing (metres) |
| `gs_min_ap_separation` | 10.0 | Min distance between APs |

### Genetic Algorithm (DEAP)

- Population-based evolutionary search via the DEAP library.
- Chromosome encodes AP positions, orientations, and (optionally) reflector params.
- Fitness = P5 RSS (5th-percentile) in linear Watts (maximised).
- Each generation's population is evaluated in parallel via `RayActorPoolExecutor`.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 150 | Population size |
| `n_gen` | 50 | Number of generations |
| `cxpb` | 0.7 | Crossover probability |
| `mutpb` | 0.3 | Mutation probability |
| `tournsize` | 10 | Tournament selection size |
| `mut_sigma_pos` | 2.0 | Position gene mutation sigma (metres) |
| `mut_sigma_dir` | 0.3 | Direction gene mutation sigma (radians) |
| `ga_min_ap_separation` | 5.0 | Min AP-AP distance (penalty-based) |

---

## Reflector-Aware Modes

### Mode Summary

| Mode | APs | Reflector | GA Chromosome | Description |
|------|-----|-----------|---------------|-------------|
| `1ap` | 1 | No | 4 genes: `[x, y, dx, dy]` | Single AP placement |
| `2ap` | 2 | No | 8 genes: `[x1,y1,x2,y2, d1x,d1y, d2x,d2y]` | Dual AP placement |
| `2ap_reflector` | 2 | Yes | 12 genes: `[..8 AP.., u, v, fx, fy]` | Dual AP + IRS reflector |

### Reflector Parameterisation

The reflector is parameterised by its position on a wall surface and a
focal point:

- **u** in [0, 1]: lateral wall-surface coordinate
- **v** in [0, 1]: vertical wall-surface coordinate
- **focal_point** (x, y, z): the point the reflector aims at
  (z fixed at `reflector_focal_z`, typically 1.5 m)

### Reflector Geometry Config

These keys are used in the experiment runner config and apply to all
methods in `2ap_reflector` mode:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `reflector_wall_top_left` | `[x,y,z]` | `[15.0, 34.0, 3.0]` | Top-left corner of reflector wall |
| `reflector_wall_bottom_right` | `[x,y,z]` | `[34.0, 34.0, 1.0]` | Bottom-right corner |
| `reflector_size` | `[w, h]` | `[2.0, 2.0]` | Panel size in metres |
| `reflector_focal_z` | `float` | `1.5` | Focal point z-height |
| `reflector_target_quantile` | `float` | `0.05` | P5 RSS objective quantile |

### How Each Method Optimises the Reflector

| Method | Approach |
|--------|----------|
| **GD** | Reflector u, v, and focal point are differentiable parameters updated by SPSA alongside AP positions. |
| **GS** | Outer loop grids over (u, v) × focal-target positions; inner loop alternates AP placement. |
| **GA** | 4 extra genes appended to chromosome; evolved jointly with AP genes via crossover and mutation. |

---

## Performance Tuning

### GPU Fraction Selection

| GPU Memory | Workers per GPU | `gpu_fraction` | Scene |
|------------|----------------|----------------|-------|
| 12 GB | 2 | 0.5 | Complex |
| 12 GB | 4 | 0.25 | Simple |
| 24 GB | 4 | 0.25 | Complex |
| 24 GB | 8 | 0.125 | Simple |

**Rule**: Start with `gpu_fraction=0.5` (2 workers/GPU) and increase workers
until OOM.

### Worker Count Guidelines

- **Exploration** (GD multi-start): 32-64 tasks, 2-4 workers.
- **Grid Search**: task count = grid points; 2-4 workers.
- **GA**: workers evaluate the population each generation; 2-4 workers.
- Diminishing returns beyond 8 workers per GPU due to memory contention.

### Speedup Analysis

```
Speedup = Total Sequential Time / Wall-Clock Time
```

Typically 0.8 × N efficiency (80% of ideal linear scaling).
Check `results["aggregate_stats"]["speedup"]` from `RayParallelOptimizer.run()`.

### Memory Optimization

- Lower `samples_per_tx` (100K for smoke tests, 1M for production).
- Reduce `max_depth` (10 instead of 13).
- Simplify scene geometry.
- Monitor with `nvidia-smi` during runs.

---

## Troubleshooting

### Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Fix**: Decrease `num_pool_workers`, lower `gpu_fraction`, or reduce
`samples_per_tx` / `max_depth`.

### Ray Actor Startup Timeout

**Symptom**: Worker group startup timed out.

**Fix**: Reduce `num_pool_workers` to match available resources. Set
`RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S` if needed.

### Slow Convergence

**Symptom**: All trajectories converging to poor local minima.

**Fix**: Increase `gd_num_tasks` for more diverse starts. Use the GA for
global search, then refine with GD.

### Import Errors in Workers

**Symptom**: `ModuleNotFoundError` in Ray actors.

**Fix**: Install the package in editable mode (`pip install -e .`) or use
`ray.init(runtime_env={"pip": [...]})`.

### Ray Dashboard

Access the Ray Dashboard at `http://127.0.0.1:8265` (shown during
`ray.init()`) to monitor actor utilisation and resource allocation.

---

## API Reference

### `RayParallelOptimizer`

Orchestrator for GD and GS parallel optimization.

```python
class RayParallelOptimizer:
    def __init__(self, num_workers=4, gpu_fraction=0.25):
        """
        Args:
            num_workers: Pool size (persistent actors).
            gpu_fraction: GPU fraction per worker (0.5 = 2 workers/GPU).
        """

    def run(self, scene_config, optimizer_method, work_items, optimization_params, verbose=True):
        """
        Run M work items across N workers (Pool -> Map -> Reduce).

        Args:
            scene_config: Dict with 'scene_path' and optional params.
            optimizer_method: 'gradient_descent' or 'grid_search' or 'grid_search_point'.
            work_items: List of optimizer kwargs dicts (one per task).
            optimization_params: Dict of params for optimizer.optimize().

        Returns:
            Dict with 'all_results', 'best_result', 'best_task_id',
            'total_time', 'aggregate_stats', 'pool_info'.
        """

    def shutdown(self):
        """Kill all actors and release GPU memory."""
```

### `RayActorPoolExecutor`

Generic execution engine for DEAP GA integration.

```python
class RayActorPoolExecutor:
    def __init__(self, scene_config, num_workers=4, gpu_fraction=0.25, verbose=True):
        """
        Spawn persistent OptimizationWorker actors.

        Args:
            scene_config: Dict with 'scene_path' (required).
            num_workers: Pool size.
            gpu_fraction: GPU fraction per worker.
        """

    def map(self, func, iterable):
        """
        Map func over items using pool.map (ordered, synchronous).

        Compatible with DEAP's toolbox.register("map", executor.map).

        Args:
            func: Converts item -> (task_id, method, kwargs, params) tuple.
            iterable: Items to process (e.g. DEAP individuals).

        Returns:
            Ordered list of result dicts (result[i] <-> iterable[i]).
        """

    def shutdown(self):
        """Kill all worker actors."""
```

### `GeneticAlgorithmRunner`

Pure DEAP evolutionary algorithm (no Ray imports).

```python
class GeneticAlgorithmRunner:
    def __init__(self, position_bounds, fixed_z, executor_map,
                 optimize_orientation=True, num_aps=1,
                 min_ap_separation=2.0, reflector_enabled=False, ...):
        """
        Args:
            position_bounds: Dict with x_min, x_max, y_min, y_max.
            fixed_z: AP height.
            executor_map: Callable for parallel evaluation (injected).
            num_aps: 1 or 2.
            reflector_enabled: Whether to add 4 reflector genes.
        """

    def run(self, optimization_params, ga_params, random_seed=42):
        """
        Run evolutionary optimization.

        Args:
            optimization_params: Ray-tracing params (samples_per_tx, max_depth).
            ga_params: DEAP params (pop_size, n_gen, cxpb, mutpb, ...).
            random_seed: Seed for reproducibility.

        Returns:
            Dict with 'best_individual', 'best_fitness_dbm',
            'hall_of_fame', 'logbook', 'total_time', ...
        """
```

### `OptimizationWorker` (Ray Actor)

```python
@ray.remote
class OptimizationWorker:
    def __init__(self, worker_id, scene_config):
        """Load Scene once. Optionally load ReflectorController."""

    def optimize(self, task_id, optimizer_method, optimizer_kwargs, optimization_params):
        """
        Create a fresh optimizer and run one task.

        Returns:
            Dict with task_id, worker_id, best_position, best_metric_dbm,
            best_direction, reflector_u, reflector_v, time_elapsed, history, ...
        """
```

### Helper Functions

```python
def generate_random_initial_positions(num_positions, bounds, fixed_z=3.8, seed=None):
    """Generate random positions within bounds."""

def generate_grid_positions(bounds, grid_resolution, fixed_z=3.8):
    """Generate grid positions for exhaustive search."""

def generate_alternating_grid_tasks(bounds, grid_resolution, fixed_ap_positions, ...):
    """Generate work items for alternating 2-AP grid search."""

def generate_reflector_grid_tasks(wall_top_left, wall_bottom_right, u_steps, v_steps, ...):
    """Generate work items for reflector grid search."""
```

---

## See Also

- [RAY_ARCHITECTURE.md](RAY_ARCHITECTURE.md) — Why Ray instead of vectorisation.
- [RAY_IMPLEMENTATION_SUMMARY.md](RAY_IMPLEMENTATION_SUMMARY.md) — File structure
  and implementation status.
- [RAY_EXPERIMENT_RUNNER.md](../guides/RAY_EXPERIMENT_RUNNER.md) — Config-driven
  runner: config schema, all trial parameters, sweep groups, and recipes.
- [GA_DEAP_IMPLEMENTATION.md](GA_DEAP_IMPLEMENTATION.md) — DEAP GA internals.
