# DEAP Genetic Algorithm — Implementation Guide

**Last Updated**: February 27, 2026
**Status**: ✅ Complete (reflector-aware, multi-AP, IoC architecture)

This document describes the implementation of the Genetic Algorithm (GA) optimizer using the DEAP library with Ray-parallel fitness evaluation, the Inversion of Control (IoC) architecture, and the reflector-aware 12-gene chromosome encoding.

## Table of Contents

1. [Overview](#overview)
2. [Architecture (IoC Pattern)](#architecture-ioc-pattern)
3. [Chromosome Encoding](#chromosome-encoding)
4. [Reflector Extension](#reflector-extension)
5. [Evolutionary Operators](#evolutionary-operators)
6. [Constraints](#constraints)
7. [Module Reference](#module-reference)
8. [Configuration](#configuration)
9. [Entry Point Usage](#entry-point-usage)
10. [Comparison with Other Methods](#comparison-with-other-methods)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The GA optimizer searches for optimal AP positions, orientations, and (optionally) reflector placement by evolving a population of candidate solutions over multiple generations. Each individual's fitness is evaluated using **Sionna ray tracing** via the existing `OptimizationWorker` infrastructure.

### Key Properties

| Property | Value |
|----------|-------|
| Library | DEAP 1.4+ |
| Modes | `1ap`, `2ap`, `2ap_reflector` |
| Chromosome | 4 / 8 / 12 genes (depending on mode) |
| Fitness | Maximise 5th-percentile RSS (linear Watts) |
| Evaluation | `SinglePointGridSearchOptimizer` via Ray ActorPool |
| Crossover | Blend (`cxBlend`, α=0.5) |
| Mutation | Split Gaussian (σ_pos=2.0, σ_dir=0.3, σ_reflector=0.1) |
| Selection | Tournament (k=10) |
| Parallelism | Ray `pool.map` (ordered, synchronous) |

---

## Architecture (IoC Pattern)

### The Problem with Monolithic Design

The initial implementation (`ray_deap_optimizer.py`, legacy) coupled DEAP logic with Ray pool management in a single class. This caused:

1. **Freeze issues**: `map_unordered` caused hangs when results arrived out-of-order and task_id tracking desynchronised on worker crash.
2. **Tight coupling**: Algorithm logic could not be tested or reused without Ray.
3. **Resource contention**: Nested parallelism (actor spawning actors) caused deadlocks.

### The IoC Solution

The refactored architecture splits into three files with clear responsibilities:

```
┌──────────────────────────────────────────────────────────────────┐
│           Entry Point (run_ga_modular.py or ray_experiment_runner.py)  │
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
DEAP Population [ind0, ind1, ..., ind149]
         │
         ▼ (filter invalid fitness)
[ind2, ind7, ind15, ...]  (e.g. 100 individuals)
         │
         ▼ check separation constraint (2-AP modes)
┌─────────────────────────────┐
│ Penalised (too close)       │  → fitness = 1e-100 (≈ −970 dBm)
│ Valid (passed constraint)   │  → continue to evaluation
└─────────────────────────────┘
         │
         ▼ toolbox.evaluate(ind) = _format_individual(ind)
[(task_id, "grid_search_point", {positions, orientations, reflector_params}, opt_params), ...]
         │
         ▼ executor.map(func, iterable)
   ┌─────┴─────┬─────────┬─────────┐
   Worker0     Worker1   Worker2   Worker3
   (Scene +    (Scene +  (Scene +  (Scene +
    Refl.Ctrl)  Refl.Ctrl) Refl.Ctrl) Refl.Ctrl)
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
ind.reflector_u = result.get("reflector_u")
ind.reflector_target = result.get("reflector_target")
```

---

## Chromosome Encoding

### 1-AP Mode (4 genes)

```
[x, y, dir_x, dir_y]
```

- `x`, `y`: AP position (bounded by `position_bounds`)
- `dir_x`, `dir_y`: Horizontal look-at direction components ∈ [-1, 1]
- `dir_z` is fixed at -0.5 (downward bias); direction vector is L2-normalised before use

### 2-AP Mode (8 genes)

```
[x1, y1, x2, y2, dir1_x, dir1_y, dir2_x, dir2_y]
```

- Genes 0-3: positions for AP0 and AP1
- Genes 4-7: direction components for AP0 and AP1
- Separation constraint: penalises pairs closer than `min_ap_separation` metres

### 2-AP + Reflector Mode (12 genes)

```
[x1, y1, x2, y2, dir1_x, dir1_y, dir2_x, dir2_y, refl_u, refl_v, focal_x, focal_y]
```

- Genes 0-7: same as 2-AP mode
- Gene 8: `reflector_u` ∈ [0, 1] — lateral wall-surface coordinate
- Gene 9: `reflector_v` ∈ [0, 1] — vertical wall-surface coordinate
- Gene 10: `focal_x` — horizontal x-component of the focal point
- Gene 11: `focal_y` — horizontal y-component of the focal point
- `focal_z` is fixed at receiver height (default 1.5 m)

---

## Reflector Extension

When `reflector_enabled=True`:

### Chromosome Modification

4 additional genes are appended after the AP direction genes. The total chromosome length becomes `2*num_aps + 2*num_aps + 4`.

### Formatted Evaluation (Worker Interface)

`_format_individual()` injects reflector parameters into the worker kwargs:

```python
optimizer_kwargs["reflector_u"] = r_u           # gene value
optimizer_kwargs["reflector_v"] = r_v           # gene value
optimizer_kwargs["reflector_target"] = (fx, fy, focal_z)
optimizer_kwargs["percentile_target_quantile"] = 0.05
```

The `OptimizationWorker` automatically:
1. Constructs `PercentileCoverageObjective(target_quantile=0.05)` inside the worker (not serialised)
2. Passes `reflector_u`, `reflector_v`, and `reflector_target` to `SinglePointGridSearchOptimizer`
3. The optimizer uses `ReflectorController.set_position_uv(u, v)` and `.set_focal_point(target)` before ray-tracing

### Fitness Extraction

The worker returns the result dict with:
- `best_metric`: P5 RSS (linear Watts) — used as DEAP fitness
- `reflector_u`, `reflector_v`, `reflector_target`, `reflector_position`: stored on the individual for logging

### Attribute Storage

After evaluation, each individual gets:
```python
ind.reflector_u = result.get("reflector_u")
ind.reflector_v = result.get("reflector_v")
ind.reflector_target = result.get("reflector_target")
ind.reflector_position = result.get("reflector_position")
ind.percentile_score = grid_results.get("percentile_score")
ind.percentile_score_dbm = grid_results.get("percentile_score_dbm")
```

---

## Evolutionary Operators

### Split Mutation (`_split_mutate`)

Different gene groups need different mutation scales:

| Gene Group | σ (std-dev) | Rationale |
|------------|------------|-----------|
| Position genes (`x, y`) | `sigma_pos` = 2.0 | Metres — large steps to explore room |
| Direction genes (`dx, dy`) | `sigma_dir` = 0.3 | Unit-vector components — small perturbations |
| Reflector genes (`u, v, fx, fy`) | `sigma_reflector` = 0.1 | UV ∈ [0,1], focal bounded — fine-grained search |

```python
def _split_mutate(individual, mu, sigma_pos, sigma_dir, indpb,
                  num_pos_genes, sigma_reflector, reflector_gene_start):
    for i in range(len(individual)):
        if random.random() < indpb:
            if i < num_pos_genes:
                individual[i] += random.gauss(mu, sigma_pos)
            elif reflector_gene_start >= 0 and i >= reflector_gene_start:
                individual[i] += random.gauss(mu, sigma_reflector)
            else:
                individual[i] += random.gauss(mu, sigma_dir)
    return (individual,)
```

### Crossover

Standard blend crossover (`cxBlend`, α=0.5) applied uniformly across all genes.

### Selection

Tournament selection with `tournsize=10` (configurable). Larger tournament size increases selection pressure.

---

## Constraints

### Bounds Enforcement (`_clamp_individual`)

After every crossover and mutation, genes are clamped:

| Gene | Lower Bound | Upper Bound |
|------|-------------|-------------|
| Position x | `x_min` | `x_max` |
| Position y | `y_min` | `y_max` |
| Direction dx, dy | `-1.0` | `1.0` |
| Reflector u, v | `0.0` | `1.0` |
| Focal x | `fx_min` | `fx_max` |
| Focal y | `fy_min` | `fy_max` |

### Separation Constraint (2-AP Modes)

For `num_aps >= 2`, any individual where the Euclidean distance between AP0 and AP1 is less than `min_ap_separation` receives:
- Penalty fitness: `1e-100` Watts (≈ −970 dBm)
- `ind.penalized = True`
- **Not submitted to the worker pool** — saves expensive ray-tracing

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

**Key — `map(func, iterable)`:**
1. Applies `func` to each item → list of arg tuples `(task_id, method, kwargs, params)`
2. Submits via `pool.map(lambda actor, args: actor.optimize.remote(*args), task_args)`
3. Returns **ordered** list of result dicts

**Why `pool.map` (not `map_unordered`):**
- Result[i] corresponds to Input[i] — no complex task_id ↔ individual mapping needed
- Blocks until all items complete → natural generation barrier for GA
- If a worker crashes, error propagates immediately instead of hanging indefinitely

### `deap_logic.py` — Algorithm Logic

**Class**: `GeneticAlgorithmRunner`

```python
class GeneticAlgorithmRunner:
    def __init__(
        self,
        position_bounds: Dict[str, float],
        fixed_z: float,
        executor_map: Callable,
        optimize_orientation: bool = True,
        fixed_dir_z: float = -0.5,
        num_aps: int = 1,
        min_ap_separation: float = 2.0,
        reflector_enabled: bool = False,
        focal_bounds: Optional[Dict[str, float]] = None,
        focal_z: float = 1.5,
        percentile_target_quantile: float = 0.05,
    )
    def run(self, optimization_params=None, ga_params=None, seed=None, verbose=True) -> Dict
    def save_evolution_plot(self, results, save_path, position_bounds=None, rss_range_dbm=None)
```

**No Ray imports.** Can be tested with `map=builtins.map` or any custom map.

### Key Helper Methods

| Method | Purpose |
|--------|---------|
| `_format_individual(ind)` | Convert DEAP individual → worker arg tuple |
| `_clamp_individual(ind)` | Enforce bounds on all genes in-place |
| `_check_separation(ind)` | Check inter-AP distance constraint |
| `_evaluate_invalid(pop, toolbox)` | Evaluate + penalise + extract fitness |
| `_extract_positions(ind)` | Get `[x, y, z]` per AP from genes |
| `_extract_directions(ind)` | Get normalised directions from genes |
| `_extract_reflector(ind)` | Get `{u, v, focal_x, focal_y, focal_z}` from genes |

---

## Configuration

### GA Hyper-parameters (`ga_params`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 150 | Population size |
| `n_gen` | 50 | Number of generations |
| `cxpb` | 0.7 | Crossover probability |
| `mutpb` | 0.3 | Mutation probability |
| `tournsize` | 10 | Tournament selection size |
| `cx_alpha` | 0.5 | Blend crossover alpha |
| `mut_mu` | 0.0 | Gaussian mutation mean |
| `mut_sigma` | 2.0 | General mutation std-dev (legacy) |
| `mut_sigma_pos` | 2.0 | Position gene mutation std-dev |
| `mut_sigma_dir` | 0.3 | Direction gene mutation std-dev |
| `mut_indpb` | 0.2 | Per-gene mutation probability |
| `hof_size` | 5 | Hall-of-Fame size |

### Reflector-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reflector_enabled` | `False` | Append 4 reflector genes to chromosome |
| `focal_bounds` | Same as position bounds | `{fx_min, fx_max, fy_min, fy_max}` |
| `focal_z` | 1.5 | Fixed z-coordinate for focal point |
| `percentile_target_quantile` | 0.05 | Quantile for shadow-robust objective |

### Radio Map Evaluation (`optimization_params`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `samples_per_tx` | 1,000,000 | Ray tracing samples per evaluation |
| `max_depth` | 13 | Maximum ray tracing depth |
| `verbose` | `False` | Worker-level verbosity |

### Pool Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_workers` | 2 | Persistent actors in the pool |
| `gpu_fraction` | 0.5 | GPU fraction per worker |

---

## Entry Point Usage

### Via Experiment Runner (recommended)

```bash
python examples/ray_experiment_runner.py \
    --config examples/ray_experiment_runner_config.example.json
```

Example trial in config:
```json
{
  "name": "ga_2ap_reflector_baseline",
  "method": "ga",
  "mode": "2ap_reflector",
  "ga_min_ap_separation": 5.0,
  "ga_target_quantile": 0.05,
  "ga_focal_z": 1.5,
  "ga_params": {
    "pop_size": 150,
    "n_gen": 50,
    "cxpb": 0.7,
    "mutpb": 0.3,
    "tournsize": 10,
    "cx_alpha": 0.5,
    "mut_sigma_pos": 2.0,
    "mut_sigma_dir": 0.3,
    "mut_indpb": 0.2,
    "hof_size": 5
  }
}
```

### Programmatic (with reflector)

```python
import ray
from reflector_position.optimizers import RayActorPoolExecutor, GeneticAlgorithmRunner

ray.init()

scene_config = {
    "scene_path": "/path/to/scene.xml",
    "frequency": 5.18e9,
    "tx_positions": [(7.0, 7.0, 3.8), (23.0, 23.0, 3.8)],
    "tx_power_dbm": 5.0,
    "reflector_enabled": True,
    "reflector_size": (2.0, 2.0),
    "wall_top_left": [15.0, 34.0, 3.0],
    "wall_bottom_right": [34.0, 34.0, 1.0],
}

executor = RayActorPoolExecutor(scene_config, num_workers=2, gpu_fraction=0.5)

ga = GeneticAlgorithmRunner(
    position_bounds={"x_min": 5.5, "x_max": 34.5, "y_min": 5.5, "y_max": 34.5},
    fixed_z=3.8,
    executor_map=executor.map,
    optimize_orientation=True,
    num_aps=2,
    min_ap_separation=5.0,
    reflector_enabled=True,
    focal_z=1.5,
    percentile_target_quantile=0.05,
)

results = ga.run(
    optimization_params={"samples_per_tx": 1_000_000, "max_depth": 13},
    ga_params={"pop_size": 150, "n_gen": 50, "mut_sigma_pos": 2.0, "mut_sigma_dir": 0.3},
    seed=42,
)

print(f"Best P5 RSS: {results['best_fitness_dbm']:.2f} dBm")
print(f"Best positions: {results['best_position']}")
print(f"Best reflector: {results.get('best_reflector')}")

ga.save_evolution_plot(results, "results/ga_reflector_evolution.png")
executor.shutdown()
ray.shutdown()
```

### Standalone (without reflector, 1-AP)

```python
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

## Results Dictionary

```python
{
    "best_individual": [x1, y1, x2, y2, dx1, dy1, dx2, dy2, u, v, fx, fy],
    "best_fitness": float,              # linear Watts (P5 RSS)
    "best_fitness_dbm": float,          # dBm
    "best_position": [[x1,y1,z], [x2,y2,z]],   # per-AP positions
    "best_direction": [[nx,ny,nz], ...],         # per-AP normalised directions
    "best_reflector": {"u": ..., "v": ..., "focal_x": ..., "focal_y": ..., "focal_z": ...},
    "hall_of_fame": [{position, fitness, fitness_dbm, reflector}, ...],
    "logbook": tools.Logbook,
    "total_time": float,
    "total_evaluations": int,
    "penalized_count": int,
    "ga_params": {...},
    "generation_details": [{gen, nevals, max_dbm, mean_dbm, ...}, ...],
}
```

---

## Comparison with Other Methods

| Aspect | Grid Search | Gradient Descent | DEAP GA |
|--------|-------------|-----------------|---------|
| **Search** | Exhaustive grid | Local gradient | Population-based |
| **Evaluations** | O(grid²) | O(iterations × multi-start) | O(pop × gen) |
| **Gradients** | None | Full (differentiable RT) + SPSA (reflector) | None (black-box) |
| **Local minima** | Not affected | Susceptible (multi-start mitigates) | More robust |
| **Reflector** | Outer sweep grid | Joint differentiable + SPSA | 4 extra genes |
| **Parallelism** | Embarrassingly parallel | Multi-start parallel | Per-generation parallel |
| **Ray method** | `grid_search_point` | `gradient_descent` | `grid_search_point` |
| **Best for** | Baseline / exhaustive | Fast convergence | Broad exploration |

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| GA hangs during evaluation | Worker OOM or crash | Reduce `samples_per_tx`, increase `gpu_fraction`, reduce `pop_size` |
| All individuals have same fitness | Bounds too tight or pop collapsed | Increase `mut_sigma_pos`, widen bounds |
| Poor convergence | Too few generations or low diversity | Increase `n_gen`, `pop_size`, or mutation rates |
| Import error for DEAP | DEAP not installed | `pip install deap>=1.4.0` |
| Reflector genes stuck at 0/1 | Bounds clamping + low mutation | Increase `sigma_reflector` in `_split_mutate` |

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
ga = GeneticAlgorithmRunner(bounds, 3.8, executor_map=builtins.map)
# Note: builtins.map won't work directly because it calls
# _format_individual which returns tuples, not result dicts.
# Use a mock that simulates worker results for unit testing.
```

---

## File Structure

```
src/reflector_position/optimizers/
├── deap_logic.py             # GeneticAlgorithmRunner (pure DEAP, no Ray)
├── ray_evaluator.py          # RayActorPoolExecutor (execution engine)
├── ray_parallel_optimizer.py # OptimizationWorker + RayParallelOptimizer
├── grid_search.py            # SinglePointGridSearchOptimizer + grid helpers
├── gradient_descent.py       # GradientDescentAPOptimizer (joint reflector)
├── optimizer_factory.py      # Factory with "grid_search_point" registered
├── base_optimizer.py         # Abstract base class
├── ray_deap_optimizer.py     # Monolithic DEAP+Ray (legacy)
└── __init__.py               # Exports all classes

examples/
├── run_ga_modular.py              # Standalone GA entry point (IoC)
├── ray_parallel_example.py        # GD + GS + GA examples via ActorPool
└── ray_experiment_runner.py       # Config-driven batch runner (all methods)
```
