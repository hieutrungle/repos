# Ray Parallel Optimizer — Implementation Summary

**Date**: February 27, 2026
**Status**: Complete (GD + GS + GA, reflector-aware, experiment runner)

---

## Overview

The Ray-based distributed parallel optimization framework enables running
multiple independent optimization trajectories simultaneously. It supports
three optimization methods — Gradient Descent (SPSA), Grid Search, and
Genetic Algorithm (DEAP) — across three modes (1ap, 2ap, 2ap_reflector),
all sharing the same ActorPool infrastructure with an Inversion of Control
(IoC) architecture.

A unified **experiment runner** (`ray_experiment_runner.py`) drives batch
execution from a single JSON config, supporting explicit trials and
Cartesian-product hyperparameter sweeps.

---

## What Was Built

### 1. Core Ray Infrastructure

#### `OptimizationWorker` (Ray Actor)
**File**: `src/reflector_position/optimizers/ray_parallel_optimizer.py`

- Loads its own `Scene` instance once in `__init__`.
- Optionally loads a `ReflectorController` for reflector-aware modes.
- Accepts many tasks via `optimize(task_id, method, kwargs, params)`.
- Creates a fresh optimizer per task via `OptimizerFactory`.
- Returns serialisable result dict with position, metric (dBm),
  orientation, reflector state, and history.

#### `RayParallelOptimizer` (Orchestrator)
**File**: Same as above

- Pool → Map → Reduce pattern with `ActorPool`.
- Lazily creates and reuses the pool across calls.
- `run()` accepts M work items for N workers (M >> N).
- Uses `map_unordered` for GD/GS task distribution.
- Computes speedup, mean/std/min/max metrics.

#### `RayActorPoolExecutor` (Generic Execution Engine)
**File**: `src/reflector_position/optimizers/ray_evaluator.py`

- Algorithm-agnostic `map(func, iterable)` interface.
- Uses `pool.map` (ordered, synchronous) — not `map_unordered`.
- Preserves input-output ordering for DEAP.
- Injected into `GeneticAlgorithmRunner` via `toolbox.register("map", ...)`.

### 2. Genetic Algorithm (DEAP)

#### `GeneticAlgorithmRunner`
**File**: `src/reflector_position/optimizers/deap_logic.py`

- **No Ray imports** — pure DEAP algorithm module.
- Dependency injection via `executor_map` callable.
- Chromosome encoding:
  - 1-AP: `[x, y, dir_x, dir_y]` (4 genes)
  - 2-AP: `[x1, y1, x2, y2, d1x, d1y, d2x, d2y]` (8 genes)
  - 2-AP + Reflector: `[...8 AP genes..., refl_u, refl_v, focal_x, focal_y]` (12 genes)
- BLX-alpha crossover, split Gaussian mutation, tournament selection.
- Maximises P5 RSS (5th-percentile received signal strength).
- Hall of Fame, statistics logging, evolution plots.

### 3. Experiment Runner

#### `ray_experiment_runner.py`
**File**: `examples/ray_experiment_runner.py` (659 lines)

- Replaces both `ray_hparam_tuning.py` and `ray_parallel_sweep_runner.py`.
- Reads a single JSON config with `shared`, `trials`, and `sweep_groups`.
- Expands sweep groups into concrete trials (Cartesian product × seeds × modes).
- Executes each trial sequentially (one method × one mode).
- Captures per-trial stdout/stderr to `output.txt`.
- Writes `summary.csv`, `summary.json`, `all_trials_detailed.json`.
- CLI: `--config`, `--output-root`, `--generate-only`, `--generated-config`.

#### Config Files
**Full config**: `examples/ray_experiment_runner_config.example.json` (259 trials)
- GD sweeps: learning rate, temperature, fairness loss type
- GS sweeps: grid resolution, outer rounds, u/v steps
- GA sweeps: pop size, n_gen, mutation rate, crossover, tournament
- AP-only baselines and reflector-aware variants

**Smoke test**: `examples/ray_experiment_runner_config.smoke_test.json` (19 trials)
- All methods with minimal parameters for fast validation (~5-15 min)
- 7 explicit trials + 12 sweep-generated trials

### 4. Optimization Functions

**File**: `examples/ray_parallel_example.py`

| Function | Method | Mode | Description |
|----------|--------|------|-------------|
| `example_parallel_gradient_descent` | GD | 1ap, 2ap | Multi-start SPSA with random restarts |
| `example_parallel_grid_search` | GS | 1ap | Single-AP exhaustive grid |
| `example_parallel_grid_search_2ap` | GS | 2ap | Alternating 2-AP grid search |
| `example_parallel_grid_search_2ap_with_reflector` | GS | 2ap_reflector | Outer reflector grid × inner AP search |
| `example_parallel_gd_2ap_with_reflector` | GD | 2ap_reflector | SPSA with joint AP + reflector |
| `example_deap_ga_1ap` | GA | 1ap | Single-AP evolutionary search |
| `example_deap_ga_2ap` | GA | 2ap | Dual-AP evolutionary search |
| `example_deap_ga_2ap_with_reflector` | GA | 2ap_reflector | Full 12-gene evolutionary search |
| `run_reflector_aware_comparison` | All | 2ap_reflector | Run all three methods and compare |

### 5. Wrapper Entry Points

| Function | Description |
|----------|-------------|
| `run_all_1ap()` | Run GD + GS + GA for 1-AP mode |
| `run_all_2ap()` | Run GD + GS + GA for 2-AP mode |
| `run_reflector_aware_comparison()` | Run GD + GS + GA for 2-AP + reflector |
| `run_reflector_aware_grid_search_only()` | Run GS only for reflector mode |

---

## Key Design Decisions

### 1. ActorPool (Not Per-Task Actors)

The heavy `Scene` is loaded once per worker, not once per task. With
N = 2 workers and M = 64 tasks, only 2 scene loads occur.

### 2. Ordered vs Unordered Map

- `RayParallelOptimizer` uses `map_unordered` (GD/GS) for throughput.
- `RayActorPoolExecutor` uses `pool.map` (ordered) for DEAP correctness —
  `result[i]` must correspond to `individual[i]`.

### 3. IoC for GA

The `GeneticAlgorithmRunner` has zero Ray imports. The `map` function
is injected via DEAP's `toolbox.register`. This enables:
- Unit testing with `builtins.map`.
- Swapping to `multiprocessing.Pool.map` if needed.
- Clean separation of concerns.

### 4. Reflector Gene Encoding

Reflector genes use different mutation sigma (`sigma_reflector=0.1`)
than position genes (`sigma_pos=2.0`), since u/v are in [0, 1] while
positions are in [5.5, 34.5].

### 5. CSV Fieldnames Fix

`_save_summary_files` collects all unique keys across ALL rows before
writing CSV, since reflector trials add extra columns (`reflector_u`,
`reflector_v`, `focal_x`, `focal_y`) that non-reflector rows lack.

---

## File Structure

```
src/reflector_position/optimizers/
  ray_parallel_optimizer.py    # OptimizationWorker + RayParallelOptimizer (1314 lines)
  ray_evaluator.py             # RayActorPoolExecutor (168 lines)
  deap_logic.py                # GeneticAlgorithmRunner (1715 lines)
  optimizer_factory.py         # OptimizerFactory.create()
  grid_search.py               # GridSearch + SinglePointGridSearch + alternating logic
  gradient_descent.py          # GradientDescentAPOptimizer
  base_optimizer.py            # BaseAPOptimizer interface
  __init__.py                  # Exports all classes

examples/
  ray_parallel_example.py      # All optimization functions (1750+ lines)
  ray_experiment_runner.py     # Config-driven batch runner (659 lines)
  ray_experiment_runner_config.example.json    # Full config (259 trials)
  ray_experiment_runner_config.smoke_test.json # Smoke test (19 trials)
  run_ga_modular.py            # Standalone GA entry point (IoC demo)

docs/
  methodology/
    RAY_ARCHITECTURE.md        # Why Ray vs vectorisation
    RAY_PARALLEL_GUIDE.md      # Usage guide with API reference
    RAY_IMPLEMENTATION_SUMMARY.md  # This file
    GA_DEAP_IMPLEMENTATION.md  # DEAP GA internals
  guides/
    RAY_EXPERIMENT_RUNNER.md   # Experiment runner comprehensive guide
```

---

## Success Metrics

### Implementation Complete

- Three optimization methods: GD (SPSA), GS (alternating), GA (DEAP).
- Three modes: 1ap, 2ap, 2ap_reflector.
- Unified experiment runner with config-driven sweep groups.
- Two config templates: full production (259 trials), smoke test (19 trials).
- IoC architecture separating algorithm logic from execution engine.
- Ordered `pool.map` for DEAP, `map_unordered` for GD/GS.
- Comprehensive documentation (5 docs, 1000+ lines total).

### Reflector-Aware Features

- GD: differentiable reflector parameters (u, v, focal) via SPSA.
- GS: outer reflector grid × inner alternating AP search.
- GA: 12-gene chromosome with split mutation (position, direction, reflector).
- Reflector geometry configurable per trial in the experiment runner.

### Experiment Runner

- Explicit trials + Cartesian sweep groups.
- Deep-merge config hierarchy: shared → base → grid combo.
- Dotted-key notation for nested parameter overrides.
- Per-trial stdout capture, trial records, and global summary files.
- Comment-only entries in trials array for readability.
- Auto-generated trial names encoding key hyperparameters.

---

## See Also

- [RAY_ARCHITECTURE.md](RAY_ARCHITECTURE.md) — Why Ray instead of vectorisation.
- [RAY_PARALLEL_GUIDE.md](RAY_PARALLEL_GUIDE.md) — Usage guide, performance
  tuning, troubleshooting, and API reference.
- [RAY_EXPERIMENT_RUNNER.md](../guides/RAY_EXPERIMENT_RUNNER.md) — Config-driven
  runner: config schema, all trial parameters, sweep groups, and recipes.
- [GA_DEAP_IMPLEMENTATION.md](GA_DEAP_IMPLEMENTATION.md) — DEAP GA internals.
