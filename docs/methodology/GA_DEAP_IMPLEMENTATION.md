# DEAP Genetic Algorithm — Implementation Guide

**Date**: February 10, 2026  
**Status**: ✅ Complete

This document describes the implementation of the Genetic Algorithm (GA) optimizer using the DEAP library with Ray-parallel fitness evaluation, and the Inversion of Control (IoC) architecture that separates algorithm logic from the execution engine.

## Table of Contents

1. [Overview](#overview)
2. [Architecture (IoC Pattern)](#architecture-ioc-pattern)
3. [Module Reference](#module-reference)
4. [Algorithm Details](#algorithm-details)
5. [Configuration](#configuration)
6. [Entry Point Usage](#entry-point-usage)
7. [Comparison with Other Methods](#comparison-with-other-methods)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The GA optimizer searches for the optimal AP (x, y) position by evolving a population of candidate positions over multiple generations. Each individual's fitness is evaluated using **Sionna ray tracing** via the existing `OptimizationWorker` infrastructure.

### Key Properties

| Property | Value |
|----------|-------|
| Library | DEAP 1.4+ |
| Representation | `[x, y]` continuous genes |
| Fitness | Maximise minimum RSS (linear Watts) |
| Evaluation | `SinglePointGridSearchOptimizer` via Ray ActorPool |
| Crossover | Blend (`cxBlend`, α=0.5) |
| Mutation | Gaussian (`mutGaussian`, σ=2.0, indpb=0.2) |
| Selection | Tournament (k=3) |
| Parallelism | Ray `pool.map` (ordered, synchronous) |

---

## Architecture (IoC Pattern)

### The Problem with Monolithic Design

The initial implementation (`ray_deap_optimizer.py`) coupled DEAP logic with Ray pool management in a single class. This caused:

1. **Freeze issues**: `map_unordered` caused hangs when results arrived out-of-order and task_id tracking desynchronised on worker crash.
2. **Tight coupling**: Algorithm logic could not be tested or reused without Ray.
3. **Resource contention**: Nested parallelism (actor spawning actors) caused deadlocks.

### The IoC Solution

The refactored architecture splits into three files with clear responsibilities:

```
┌──────────────────────────────────────────────────────────────────┐
│                   Entry Point (run_ga_modular.py)                │
│         Wires executor.map into GA runner via DI                 │
└───────────┬───────────────────────────────────┬──────────────────┘
            │                                   │
            ▼                                   ▼
┌──────────────────────────┐    ┌──────────────────────────────────┐
│   ray_evaluator.py       │    │        deap_logic.py              │
│   RayActorPoolExecutor   │    │    GeneticAlgorithmRunner         │
│                          │    │                                   │
│  • Manages ActorPool     │    │  • Pure DEAP (NO ray imports)     │
│  • pool.map (ordered)    │    │  • Registers injected map         │
│  • Knows NOTHING about   │    │  • _format_individual → task args │
│    DEAP or fitness       │    │  • GA loop: select/cross/mutate   │
│  • Returns result dicts  │    │  • Extracts fitness from results  │
└──────────────────────────┘    └──────────────────────────────────┘
         ▲                                    │
         │    toolbox.register("map",         │
         │        executor.map)               │
         └────────────────────────────────────┘
              Dependency Injection
```

### Injection Point

DEAP is designed for this pattern via `toolbox.register("map", ...)`:

```python
# In GeneticAlgorithmRunner.__init__:
self.toolbox.register("map", self._executor_map)   # injected
self.toolbox.register("evaluate", self._format_individual)

# During evaluation:
results = toolbox.map(toolbox.evaluate, invalid_ind)
#         ^^^^^^^^^   ^^^^^^^^^^^^^^^^
#         executor     formats ind → (task_id, method, kwargs, params)
#         .map()
```

### Data Flow Per Generation

```
DEAP Population [ind0, ind1, ..., ind49]
         │
         ▼ (filter invalid fitness)
[ind2, ind7, ind15, ...]  (e.g. 35 individuals)
         │
         ▼ toolbox.evaluate(ind) = _format_individual(ind)
[(task_id, "grid_search_point", {eval_pos, z}, opt_params), ...]
         │
         ▼ executor.map(func, iterable)
   ┌─────┴─────┬─────────┬─────────┐
   Worker0     Worker1   Worker2   Worker3
   (Scene)     (Scene)   (Scene)   (Scene)
   ┌─────┐    ┌─────┐   ┌─────┐   ┌─────┐
   │eval │    │eval │   │eval │   │eval │
   │ind2 │    │ind7 │   │ind15│   │...  │
   └──┬──┘    └──┬──┘   └──┬──┘   └──┬──┘
      │          │         │         │
      ▼          ▼         ▼         ▼
   result0    result1   result2   result3 ...
         │
         ▼ pool.map guarantees order
[result0, result1, result2, ...]  (ordered)
         │
         ▼ extract fitness
ind.fitness.values = (result["best_metric"],)
```

---

## Module Reference

### `ray_evaluator.py` — Execution Engine

**Class**: `RayActorPoolExecutor`

```python
class RayActorPoolExecutor:
    def __init__(self, scene_config, num_workers=4, gpu_fraction=0.25, verbose=True)
    def map(self, func, iterable) -> List[Dict]
    def shutdown() -> None
```

**Key method — `map(func, iterable)`:**
1. Applies `func` to each item → list of arg tuples `(task_id, method, kwargs, params)`
2. Submits via `pool.map(lambda actor, args: actor.optimize.remote(*args), task_args)`
3. Returns **ordered** list of result dicts

**Why `pool.map` (not `map_unordered`):**
- Result[i] corresponds to Input[i] — no complex task_id ↔ individual mapping needed
- Blocks until all items complete → natural generation barrier
- If a worker crashes, error propagates immediately instead of hanging indefinitely

### `deap_logic.py` — Algorithm Logic

**Class**: `GeneticAlgorithmRunner`

```python
class GeneticAlgorithmRunner:
    def __init__(self, position_bounds, fixed_z, executor_map)
    def run(self, optimization_params=None, ga_params=None, seed=None, verbose=True) -> Dict
    def save_evolution_plot(self, results, save_path, position_bounds=None, rss_range_dbm=None)
```

**No Ray imports.** Can be tested with `map=builtins.map` or any custom map.

**Key method — `_format_individual(ind)`:**
```python
def _format_individual(self, ind) -> Tuple:
    return (
        task_id,                 # sequential counter
        "grid_search_point",     # optimizer method
        {
            "evaluation_position": (float(ind[0]), float(ind[1])),
            "fixed_z": self.fixed_z,
        },
        self._opt_params,        # {"samples_per_tx": ..., "max_depth": ...}
    )
```

### `run_ga_modular.py` — Entry Point

Wires executor and GA runner together:

```python
executor = RayActorPoolExecutor(scene_config, num_workers=4, gpu_fraction=0.25)
ga = GeneticAlgorithmRunner(bounds, fixed_z=3.8, executor_map=executor.map)
try:
    results = ga.run(optimization_params=..., ga_params=..., seed=42)
    ga.save_evolution_plot(results, "results/ga_evolution.png")
finally:
    executor.shutdown()
```

---

## Algorithm Details

### GA Operators

| Operator | DEAP Function | Default Parameters |
|----------|--------------|-------------------|
| Crossover | `tools.cxBlend` | α = 0.5 |
| Mutation | `tools.mutGaussian` | μ=0, σ=2.0, indpb=0.2 |
| Selection | `tools.selTournament` | tournsize = 3 |

### Generational Loop (eaSimple-style)

```
for gen in 1..n_gen:
    offspring = select(population, len(population))
    offspring = clone(offspring)
    
    for child1, child2 in pairs(offspring):
        if random() < cxpb:
            mate(child1, child2)
            invalidate fitness
    
    for mutant in offspring:
        if random() < mutpb:
            mutate(mutant)
            invalidate fitness
    
    clamp all to position_bounds
    
    evaluate individuals with invalid fitness  ← parallel via Ray
    
    population = offspring
    update HallOfFame
    record statistics
```

### Bounds Enforcement

After crossover/mutation, genes are clamped to `[x_min, x_max]` and `[y_min, y_max]`:

```python
ind[0] = max(x_min, min(ind[0], x_max))
ind[1] = max(y_min, min(ind[1], y_max))
```

### Statistics & Tracking

- **Logbook**: Per-generation max, mean, std, min (linear + dBm)
- **HallOfFame**: Top-k best solutions across all generations
- **Generation details**: Best position, evaluation count, wall-clock time per generation

### Results Dictionary

```python
{
    "best_individual": [x, y],
    "best_fitness": float,           # linear Watts
    "best_fitness_dbm": float,       # dBm
    "best_position": [x, y, z],
    "hall_of_fame": [{position, fitness, fitness_dbm}, ...],
    "logbook": tools.Logbook,
    "total_time": float,
    "total_evaluations": int,
    "ga_params": {...},
    "generation_details": [{gen, nevals, max_dbm, mean_dbm, ...}, ...],
}
```

---

## Configuration

### GA Hyper-parameters (`ga_params`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 50 | Population size |
| `n_gen` | 20 | Number of generations |
| `cxpb` | 0.7 | Crossover probability |
| `mutpb` | 0.3 | Mutation probability |
| `tournsize` | 3 | Tournament selection size |
| `cx_alpha` | 0.5 | Blend crossover alpha |
| `mut_mu` | 0.0 | Gaussian mutation mean |
| `mut_sigma` | 2.0 | Gaussian mutation std-dev |
| `mut_indpb` | 0.2 | Per-gene mutation probability |
| `hof_size` | 5 | Hall-of-Fame size |

### Radio Map Evaluation (`optimization_params`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `samples_per_tx` | 1,000,000 | Ray tracing samples per evaluation |
| `max_depth` | 13 | Maximum ray tracing depth |
| `verbose` | False | Worker-level verbosity |

### Pool Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_workers` | 4 | Persistent actors in the pool |
| `gpu_fraction` | 0.25 | GPU fraction per worker (4 workers/GPU) |

---

## Entry Point Usage

### Basic

```bash
python examples/run_ga_modular.py
```

### Customised (modify `run_ga_modular.py` constants)

```python
GA_PARAMS = {
    "pop_size": 100,
    "n_gen": 50,
    "cxpb": 0.8,
    "mutpb": 0.2,
    "tournsize": 5,
    "mut_sigma": 1.5,
}
NUM_POOL_WORKERS = 8
GPU_FRACTION = 0.125  # 8 workers per GPU
```

### Programmatic

```python
from reflector_position.optimizers import RayActorPoolExecutor, GeneticAlgorithmRunner

executor = RayActorPoolExecutor(scene_config, num_workers=4, gpu_fraction=0.25)
ga = GeneticAlgorithmRunner(
    position_bounds={"x_min": 5, "x_max": 25, "y_min": 5, "y_max": 25},
    fixed_z=3.8,
    executor_map=executor.map,
)
results = ga.run(
    optimization_params={"samples_per_tx": 1_000_000, "max_depth": 13},
    ga_params={"pop_size": 50, "n_gen": 20},
    seed=42,
)
ga.save_evolution_plot(results, "ga_evolution.png")
executor.shutdown()
```

---

## Comparison with Other Methods

### Method Summary

| Aspect | Grid Search | Gradient Descent | DEAP GA |
|--------|-------------|-----------------|---------|
| **Search** | Exhaustive grid | Local gradient | Population-based |
| **Evaluations** | O(grid²) | O(iterations) | O(pop × gen) |
| **Gradients** | None | Full (differentiable RT) | None (black-box) |
| **Local minima** | Not affected | Susceptible | More robust |
| **Parallelism** | Embarrassingly parallel | Multi-start parallel | Per-generation parallel |
| **Ray method** | `grid_search_point` | `gradient_descent` | `grid_search_point` |
| **Best for** | Baseline / exhaustive | Fast convergence | Broad exploration |

### When to Use Each

- **Grid Search**: When you want a guaranteed exhaustive survey of the space
- **Gradient Descent**: When you want fast convergence from a good starting point
- **DEAP GA**: When the landscape is complex and you want population-based exploration without needing differentiable evaluation
- **Hybrid GA→GD**: Run GA for broad exploration, then seed GD from the top GA solutions for fine-tuning (planned Phase 4)

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| GA hangs during evaluation | Worker OOM or crash | Reduce `samples_per_tx`, increase `gpu_fraction`, reduce `pop_size` |
| All individuals have same fitness | Bounds too tight or population collapsed | Increase `mut_sigma`, widen bounds |
| Poor convergence | Too few generations or low diversity | Increase `n_gen`, `pop_size`, or `mut_sigma` |
| Import error for DEAP | DEAP not installed | `pip install deap>=1.4.0` |

### Verifying the IoC Pattern

```python
# deap_logic.py should have ZERO ray imports:
import ast, inspect
from reflector_position.optimizers.deap_logic import GeneticAlgorithmRunner
source = inspect.getsource(GeneticAlgorithmRunner)
assert "import ray" not in source
assert "from ray" not in source
```

### Testing with Sequential Map

For debugging without Ray:

```python
def sequential_evaluator(func, iterable):
    """Drop-in replacement using Python's built-in map."""
    items = list(iterable)
    task_args = [func(item) for item in items]
    # Simulate worker results (requires actual scene)
    return [worker.optimize(*args) for args in task_args]

ga = GeneticAlgorithmRunner(bounds, 3.8, executor_map=sequential_evaluator)
```

---

## File Structure

```
src/reflector_position/optimizers/
├── ray_evaluator.py          # RayActorPoolExecutor (execution engine)
├── deap_logic.py             # GeneticAlgorithmRunner (pure DEAP)
├── ray_deap_optimizer.py     # RayDEAPOptimizer (monolithic, legacy)
├── ray_parallel_optimizer.py # ActorPool orchestrator + OptimizationWorker
├── grid_search.py            # GridSearch + SinglePointGridSearchOptimizer
├── gradient_descent.py       # GradientDescentAPOptimizer
├── optimizer_factory.py      # Factory with "grid_search_point" registered
├── base_optimizer.py         # Abstract base class
└── __init__.py               # Exports all classes

examples/
├── run_ga_modular.py         # IoC entry point for GA
└── ray_parallel_example.py   # GD + GS examples via ActorPool
```
