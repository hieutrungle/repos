# Ray Architecture for Reflector Optimization

**Date**: February 27, 2026
**Status**: Implementation Complete

This document explains the Ray-based distributed computing architecture used
for optimising access-point (AP) placement and intelligent reflecting surface
(IRS) reflector positioning.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Why Ray Instead of Vectorization?](#why-ray-instead-of-vectorization)
3. [Architecture Overview](#architecture-overview)
4. [Key Components](#key-components)
5. [ActorPool Pattern](#actorpool-pattern)
6. [Reflector-Aware Optimization](#reflector-aware-optimization)
7. [Inversion of Control (IoC)](#inversion-of-control-ioc)
8. [Resource Management](#resource-management)

---

## Problem Statement

For each optimisation step, the reflector and AP positions change the
physical scene geometry. Different optimisation trajectories need
independent scene instances — vectorised batching within a single scene
cannot support this because the scene mesh state is shared.

---

## Why Ray Instead of Vectorization?

### The "State Mutation" Problem

In Sionna's ray tracing engine, the `Scene` object holds the geometry of
walls, meshes, and reflector surfaces.

* **Vectorized approach (single Scene):** A single `Scene` instance cannot
  simultaneously host Reflector A at the North wall for Batch 1 *and*
  Reflector B at the South wall for Batch 2. The geometry is shared.
* **Ray approach (multiple Scenes):** Each Ray Actor loads its own private
  copy of the `Scene`. Worker 1 places the reflector on the North wall
  while Worker 2 places it on the South wall — no conflict.

### When to Use Each Approach

| Aspect | Vectorized Batching | Ray Architecture |
|--------|-------------------|------------------|
| **Use Case** | Changing parameters within a single scene | Modifying physical scene geometry |
| **Examples** | Tx positions, phase shifts, beam angles | Reflector positions, walls, obstacles |
| **Memory** | Shared scene, vectorised parameters | Independent scene copy per worker |
| **Parallelism** | GPU vectorisation (SIMD) | Process-level parallelism |
| **Isolation** | None (shared state) | Complete (separate processes) |
| **Best For** | Parameter sweeps, beamforming | Reflector / obstacle optimisation |

### Summary

* **Vectorisation** → changing *wave parameters* (phases, amplitudes) or
  *Tx/Rx positions* (points).
* **Ray / Multi-process** → changing *scene geometry* (meshes, walls,
  reflectors, obstacles).

---

## Architecture Overview

The framework uses two complementary Ray patterns:

### 1. `RayParallelOptimizer` — Pool → Map → Reduce

Used by **Gradient Descent (GD)** and **Grid Search (GS)** methods.

```
+---------------------------------------------------------+
|  1. POOL - Spawn N persistent OptimizationWorker actors  |
|     Each loads its own Scene once via setup_building_... |
+---------------------------------------------------------+
|  2. MAP - Submit M work items (M >> N) to ActorPool      |
|     map_unordered distributes tasks to idle workers      |
+---------------------------------------------------------+
|  3. REDUCE - Collect results, argmax -> best_result      |
|     Compute aggregate statistics and speedup             |
+---------------------------------------------------------+
```

### 2. `RayActorPoolExecutor` — IoC Map Interface

Used by the **Genetic Algorithm (GA)** via DEAP's `toolbox.map`.

```
+---------------------------------------------------------+
|  RayActorPoolExecutor (ray_evaluator.py)                 |
|    pool.map(func, population) - ORDERED, synchronous     |
|    Knows nothing about DEAP, GA, or fitness              |
+---------------------------------------------------------+
|  GeneticAlgorithmRunner (deap_logic.py)                   |
|    toolbox.register("map", executor.map)                 |
|    Pure DEAP logic - no Ray imports                      |
+---------------------------------------------------------+
```

---

## Key Components

### `OptimizationWorker` (Ray Actor)

**File**: `src/reflector_position/optimizers/ray_parallel_optimizer.py`

A Ray remote actor that:

1. **Loads the Scene once** in `__init__` via `setup_building_floor_scene()`.
2. **Optionally loads a `ReflectorController`** when reflector is enabled.
3. **Accepts many tasks** via `optimize(task_id, method, kwargs, params)`.
4. **Creates a fresh optimiser** per task using `OptimizerFactory`.
5. **Returns a serialisable result dict** with position, metric (dBm),
   orientation, reflector state, and timing.

```python
@ray.remote
class OptimizationWorker:
    def __init__(self, worker_id, scene_config):
        self.scene, self.reflector_controller = self._load_scene(scene_config)

    def optimize(self, task_id, optimizer_method, optimizer_kwargs, optimization_params):
        optimizer = OptimizerFactory.create(
            method=optimizer_method, scene=self.scene, **optimizer_kwargs
        )
        result = optimizer.optimize(**optimization_params)
        return {
            "task_id": task_id,
            "best_position": ...,
            "best_metric_dbm": ...,
            "reflector_u": ...,
            "reflector_v": ...,
            ...
        }
```

### `RayParallelOptimizer` (Orchestrator)

**File**: Same as above

- Creates an `ActorPool` of `N` persistent workers.
- `run()` accepts `M` work items (M can be >> N).
- Uses `pool.map_unordered` for GD/GS task distribution.
- Computes aggregate statistics (mean, std, speedup).
- Pool is **lazily created** and **reused** across calls.

### `RayActorPoolExecutor` (Generic Execution Engine)

**File**: `src/reflector_position/optimizers/ray_evaluator.py`

- Wraps the same `OptimizationWorker` actors in an `ActorPool`.
- Provides `map(func, iterable)` — compatible with DEAP's `toolbox.map`.
- Uses `pool.map` (**ordered, synchronous**) — not `map_unordered` —
  to preserve input-output ordering for DEAP.
- Knows nothing about genetic algorithms.

### `GeneticAlgorithmRunner` (Pure DEAP Logic)

**File**: `src/reflector_position/optimizers/deap_logic.py`

- **No Ray imports.** Receives `executor_map` via dependency injection.
- Chromosome encoding:
  - 1-AP: `[x, y, dir_x, dir_y]` (4 genes)
  - 2-AP: `[x1, y1, x2, y2, dir1_x, dir1_y, dir2_x, dir2_y]` (8 genes)
  - 2-AP + Reflector: `[...8 AP genes..., refl_u, refl_v, focal_x, focal_y]` (12 genes)
- BLX-alpha crossover, split Gaussian mutation (different sigma for position,
  direction, and reflector genes), tournament selection.
- Maximises P5 RSS (5th-percentile received signal strength).

---

## ActorPool Pattern

### Why ActorPool Instead of Per-Task Actors

| Approach | Scene Loads | Memory | Setup Overhead |
|----------|------------|--------|---------------|
| Per-task actors (old) | M (one per task) | M x scene_size | High |
| ActorPool (current) | N (one per worker) | N x scene_size | Low |

With N = 2 workers and M = 64 tasks, the scene is loaded only 2 times
instead of 64. This saves several minutes of setup time.

### Decoupled Task Count

```
N workers = 2 (pool size, each loading Scene once)
M tasks   = 64 (work items queued to the pool)
Queuing   = ActorPool auto-distributes tasks to idle workers
```

---

## Reflector-Aware Optimization

All three methods support the `2ap_reflector` mode:

| Method | How Reflector Is Optimised |
|--------|--------------------------|
| **GD** | Reflector u/v and focal point are differentiable parameters learned alongside AP positions. |
| **GS** | Outer loop sweeps reflector (u, v) x focal-target grid; inner loop alternates AP grid search. |
| **GA** | 4 extra genes (`refl_u`, `refl_v`, `focal_x`, `focal_y`) appended to the chromosome. |

The reflector is parameterised by:
- **(u, v)** in [0, 1]^2: position on the wall surface.
- **focal_point (x, y, z)**: the 3-D point the reflector orients toward
  (z is fixed at receiver height).

### Modes

| Mode | APs | Reflector | GA Chromosome | Description |
|------|-----|-----------|---------------|-------------|
| `1ap` | 1 | No | 4 genes | Single AP placement |
| `2ap` | 2 | No | 8 genes | Dual AP placement |
| `2ap_reflector` | 2 | Yes | 12 genes | Dual AP + IRS reflector |

**Objective metric**: P5 RSS (5th-percentile received signal strength in dBm).

---

## Inversion of Control (IoC)

The GA layer is completely decoupled from Ray:

```
+----------------------------+     +----------------------------+
|  GeneticAlgorithmRunner    | --> |  RayActorPoolExecutor      |
|  (deap_logic.py)           |     |  (ray_evaluator.py)        |
|                            |     |                            |
|  toolbox.map(eval, pop)    |     |  pool.map(func, items)     |
|  No Ray imports            |     |  OptimizationWorker x N    |
+----------------------------+     +----------------------------+
```

Benefits:
- GA logic is testable without Ray.
- Swap `executor.map` for `builtins.map` for single-process debugging.
- Same pattern works with `multiprocessing.Pool.map` if needed.

---

## Resource Management

### GPU Fraction

| `gpu_fraction` | Workers per GPU | Typical Use |
|----------------|----------------|-------------|
| `1.0` | 1 | Single worker, full GPU |
| `0.5` | 2 | Default in experiment runner |
| `0.25` | 4 | Light scenes |
| `0.125` | 8 | Very light scenes |

### Memory Considerations

Each worker loads a complete Scene (XML parsing, BVH tree, textures).
VRAM is the primary constraint:

- **Start with `gpu_fraction=0.5`** (2 workers/GPU).
- Increase workers until OOM.
- Lower `samples_per_tx` or `max_depth` to reduce per-task VRAM.

### Ray Lifecycle

```python
ray.init(ignore_reinit_error=True)

# RayParallelOptimizer / RayActorPoolExecutor manage the pool internally.
# Always call shutdown() when done:
parallel_opt.shutdown()   # Kills worker actors, frees GPU memory

ray.shutdown()
```

---

## See Also

- [RAY_PARALLEL_GUIDE.md](RAY_PARALLEL_GUIDE.md) — Detailed usage guide
  with code examples, performance tuning, and troubleshooting.
- [RAY_IMPLEMENTATION_SUMMARY.md](RAY_IMPLEMENTATION_SUMMARY.md) — Implementation
  status and file structure.
- [RAY_EXPERIMENT_RUNNER.md](../guides/RAY_EXPERIMENT_RUNNER.md) — Config-driven
  hyperparameter sweep runner (the primary entry point for batch experiments).
- [GA_DEAP_IMPLEMENTATION.md](GA_DEAP_IMPLEMENTATION.md) — DEAP GA
  implementation details.
