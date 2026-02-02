# Ray Parallel Optimization

**Date**: January 31, 2026  
**Status**: Implementation Complete

This document provides comprehensive guidance on using Ray for distributed parallel optimization of reflector positions.

## Table of Contents

1. [Overview](#overview)
2. [Why Ray?](#why-ray)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Advanced Usage](#advanced-usage)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)

---

## Overview

The `RayParallelOptimizer` enables distributed parallel execution of optimization algorithms, allowing you to explore multiple starting positions simultaneously to find global optima in non-convex optimization landscapes.

### Key Features

- **Process-level Isolation**: Each Ray actor has its own Scene instance
- **True Parallelism**: Multiple "parallel universes" exploring different trajectories
- **GPU Efficiency**: Configurable GPU fraction per worker (e.g., 0.1 = 10 workers per GPU)
- **Optimizer Agnostic**: Works with any optimizer inheriting from `BaseAPOptimizer`
- **Automatic Aggregation**: Winner selection from all parallel trajectories

---

## Why Ray?

### The Critical Distinction

Ray is necessary when optimizing **physical scene geometry** (reflector positions, wall placements, obstacle locations) rather than just wave parameters or Tx/Rx coordinates.

#### Vectorized Batching vs Ray Architecture

| Aspect | Vectorized Batching | Ray Architecture |
|--------|-------------------|------------------|
| **Use Case** | Changing parameters within single scene | Modifying physical scene geometry |
| **Examples** | Tx positions, phase shifts, beam angles | Reflector positions, walls, obstacles |
| **Memory** | Shared scene, vectorized parameters | Independent scene copies per worker |
| **Parallelism** | GPU vectorization (SIMD) | Process-level parallelism |
| **Isolation** | None (shared state) | Complete (separate processes) |
| **Best For** | Parameter sweeps, beamforming | Reflector/obstacle optimization |

### When to Use Ray

✅ **Use Ray When:**
- Optimizing physical reflector positions
- Moving walls or obstacles
- Each optimization needs different scene geometry
- Exploring multiple local minima in parallel
- Need process-level isolation

❌ **Don't Use Ray When:**
- Only changing Tx/Rx positions (use vectorization)
- Optimizing beamforming coefficients (use vectorization)
- Single optimization trajectory is sufficient
- Memory is extremely constrained

---

## Architecture

### Three-Phase Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. FORK PHASE                                 │
│                 Spawn Ray Actors                                 │
├─────────────────────────────────────────────────────────────────┤
│  Orchestrator                                                    │
│      │                                                           │
│      ├──> Worker 0: Scene @ Position P0, Optimizer Init         │
│      ├──> Worker 1: Scene @ Position P1, Optimizer Init         │
│      ├──> Worker 2: Scene @ Position P2, Optimizer Init         │
│      └──> ...                                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    2. MAP PHASE                                  │
│              Parallel Optimization                               │
├─────────────────────────────────────────────────────────────────┤
│  Worker 0                Worker 1                Worker N        │
│  ┌────────────┐         ┌────────────┐         ┌────────────┐  │
│  │ Scene P0   │         │ Scene P1   │         │ Scene PN   │  │
│  │ Optimizer  │         │ Optimizer  │         │ Optimizer  │  │
│  │  Iterate   │         │  Iterate   │         │  Iterate   │  │
│  └────────────┘         └────────────┘         └────────────┘  │
│       ↓                      ↓                      ↓           │
│  Result R0              Result R1              Result RN        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    3. REDUCE PHASE                               │
│             Aggregation & Winner Selection                       │
├─────────────────────────────────────────────────────────────────┤
│  Orchestrator                                                    │
│      │                                                           │
│      ├─ Collect all results [R0, R1, ..., RN]                   │
│      ├─ Find best: best_idx = argmax(metrics)                   │
│      ├─ Compute statistics: mean, std, range                    │
│      └─ Return winner + aggregate stats                         │
└─────────────────────────────────────────────────────────────────┘
```

### Ray Actor Structure

Each `OptimizationWorker` is a Ray actor with:

1. **Isolated Memory**: Own copy of Scene, optimizer, tensors
2. **GPU Allocation**: Configurable GPU fraction (e.g., 0.1)
3. **Independent Execution**: No communication with other workers ("Asocial")
4. **Result Return**: Sends final result back to orchestrator

---

## Quick Start

### Installation

Ensure Ray is installed:

```bash
pip install ray[default]
```

### Basic Example

```python
import ray
from reflector_position.optimizers import (
    RayParallelOptimizer,
    generate_random_initial_positions,
)

# Initialize Ray
ray.init()

# Configuration
NUM_WORKERS = 8
bounds = {"x_min": 0.0, "x_max": 20.0, "y_min": 0.0, "y_max": 20.0}

# Generate diverse starting positions
initial_positions = generate_random_initial_positions(
    num_positions=NUM_WORKERS,
    bounds=bounds,
    fixed_z=3.8,
    seed=42,
)

# Create parallel optimizer
parallel_opt = RayParallelOptimizer(
    num_workers=NUM_WORKERS,
    gpu_fraction=0.25,  # 4 workers per GPU
    optimizer_method="gradient_descent",
)

# Scene configuration
scene_config = {
    "xml_path": "l_shape_scene.xml",
    "reflector_name": "reflector",
}

# Optimization parameters
opt_params = {
    "num_iterations": 50,
    "learning_rate": 0.5,
    "samples_per_tx": 1_000_000,
    "max_depth": 13,
}

# Run parallel optimization
results = parallel_opt.optimize(
    scene_config=scene_config,
    initial_positions=initial_positions,
    optimization_params=opt_params,
    verbose=True,
)

# Access best result
best = results["best_result"]
print(f"Best position: {best['best_position']}")
print(f"Best metric: {best['best_metric']:.4f}")

# Plot results
parallel_opt.plot_results(results)

# Cleanup
parallel_opt.shutdown()
ray.shutdown()
```

---

## Advanced Usage

### 1. Parallel Grid Search

Divide the search space into regions, one per worker:

```python
# Divide space into 4 quadrants
quadrant_bounds = [
    {"x_min": 0.0, "x_max": 10.0, "y_min": 0.0, "y_max": 10.0},   # Q1
    {"x_min": 10.0, "x_max": 20.0, "y_min": 0.0, "y_max": 10.0},  # Q2
    {"x_min": 0.0, "x_max": 10.0, "y_min": 10.0, "y_max": 20.0},  # Q3
    {"x_min": 10.0, "x_max": 20.0, "y_min": 10.0, "y_max": 20.0}, # Q4
]

worker_configs = [
    {"search_bounds": bounds, "grid_resolution": 2.0}
    for bounds in quadrant_bounds
]

parallel_opt = RayParallelOptimizer(
    num_workers=4,
    gpu_fraction=0.25,
    optimizer_method="grid_search",
)

# Dummy initial positions (not used for grid search)
dummy_positions = [np.array([10.0, 10.0, 3.8])] * 4

results = parallel_opt.optimize(
    scene_config=scene_config,
    initial_positions=dummy_positions,
    optimization_params=opt_params,
    optimizer_configs=worker_configs,  # Different config per worker
    verbose=True,
)
```

### 2. Hyperparameter Tuning

Test different hyperparameters in parallel:

```python
learning_rates = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]
NUM_WORKERS = len(learning_rates)

base_position = np.array([10.0, 10.0, 3.8])
initial_positions = [base_position.copy() for _ in range(NUM_WORKERS)]

# Run separate optimization for each learning rate
for i, lr in enumerate(learning_rates):
    opt_params = {
        "num_iterations": 30,
        "learning_rate": lr,
        "samples_per_tx": 500_000,
    }
    
    single_opt = RayParallelOptimizer(num_workers=1, gpu_fraction=0.5)
    result = single_opt.optimize(
        scene_config=scene_config,
        initial_positions=[initial_positions[i]],
        optimization_params=opt_params,
    )
    
    print(f"LR {lr}: Metric = {result['best_result']['best_metric']:.4f}")
    single_opt.shutdown()
```

**Note**: For more sophisticated hyperparameter tuning, consider using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html).

### 3. Multi-Stage Optimization

Coarse search followed by fine refinement:

```python
# Stage 1: Coarse parallel search
coarse_positions = generate_random_initial_positions(
    num_positions=32,
    bounds={"x_min": 0, "x_max": 20, "y_min": 0, "y_max": 20},
    seed=42,
)

coarse_opt = RayParallelOptimizer(num_workers=32, gpu_fraction=0.1)
coarse_results = coarse_opt.optimize(
    scene_config=scene_config,
    initial_positions=coarse_positions,
    optimization_params={"num_iterations": 20, "learning_rate": 1.0},
)
coarse_opt.shutdown()

# Stage 2: Fine search around top 8 positions
top_positions = sorted(
    coarse_results["all_results"],
    key=lambda x: x["best_metric"],
    reverse=True,
)[:8]

fine_positions = [r["best_position"] for r in top_positions]

fine_opt = RayParallelOptimizer(num_workers=8, gpu_fraction=0.25)
fine_results = fine_opt.optimize(
    scene_config=scene_config,
    initial_positions=fine_positions,
    optimization_params={"num_iterations": 50, "learning_rate": 0.1},
)
fine_opt.shutdown()

print(f"Final best: {fine_results['best_result']['best_position']}")
```

---

## Performance Tuning

### GPU Fraction Selection

Choose `gpu_fraction` based on your GPU memory:

| GPU Memory | Workers per GPU | gpu_fraction | Scene Complexity |
|------------|----------------|--------------|------------------|
| 12GB | 4 | 0.25 | Simple |
| 12GB | 8 | 0.125 | Simple |
| 24GB | 8 | 0.125 | Complex |
| 24GB | 16 | 0.0625 | Simple |
| 48GB | 16 | 0.0625 | Complex |

**Rule of Thumb**: Start with `gpu_fraction=0.25` (4 workers/GPU) and increase workers until you hit OOM errors.

### Worker Count Guidelines

**Optimal Number of Workers**:
- **Exploration**: 16-32 workers for diverse initial positions
- **Refinement**: 4-8 workers for local search
- **Hyperparameter**: N workers for N hyperparameter combinations
- **Grid Search**: 4-16 workers for spatial subdivision

**Diminishing Returns**: Beyond 32-64 workers, the orchestration overhead may outweigh parallelism benefits.

### Memory Optimization

**Reduce Memory Usage**:
1. Lower `samples_per_tx` (500K instead of 1M)
2. Simplify scene meshes
3. Use lower `max_depth` (10 instead of 13)
4. Reduce number of concurrent workers

**Monitor Memory**:
```python
import GPUtil

# Before optimization
GPUtil.showUtilization()

# Run optimization
results = parallel_opt.optimize(...)

# After optimization
GPUtil.showUtilization()
```

### Speedup Analysis

Expected speedup with N workers:

```
Speedup = (Total Worker Time) / (Wall-clock Time)
```

**Ideal**: Speedup ≈ N (linear scaling)  
**Typical**: Speedup ≈ 0.8 * N (80% efficiency due to overhead)  
**Poor**: Speedup << N (bottleneck in orchestration or I/O)

The `aggregate_stats["speedup"]` in results provides this metric.

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `gpu_fraction` (more workers per GPU)
- Decrease `num_workers`
- Lower `samples_per_tx` or `max_depth`
- Simplify scene geometry

#### 2. Ray Actor Startup Timeout

**Symptoms**: `The worker group startup timed out after 30.0 seconds`

**Causes**:
- Insufficient cluster resources
- Ray autoscaler still provisioning nodes

**Solutions**:
- Set `RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S` environment variable
- Reduce `num_workers` to match available resources
- Wait for autoscaler to provision nodes (expected in first run)

#### 3. Slow Convergence

**Symptoms**: All workers converging to similar (poor) local minima

**Causes**:
- Insufficient diversity in initial positions
- Learning rate too high (overshooting) or too low (stuck)

**Solutions**:
- Increase variance in `generate_random_initial_positions` seed
- Test different learning rates (hyperparameter search)
- Use multi-stage optimization (coarse → fine)

#### 4. Import Errors in Workers

**Symptoms**: `ModuleNotFoundError` in Ray actors

**Causes**:
- Package not installed in worker environment
- Import path issues

**Solutions**:
- Ensure all dependencies in `requirements.txt`
- Use `ray.init(runtime_env={"pip": ["package1", "package2"]})`
- Install package in editable mode: `pip install -e .`

### Debugging Tips

**Enable Ray Logging**:
```python
import ray
ray.init(logging_level="DEBUG")
```

**Check Ray Dashboard**:
```bash
# Ray dashboard URL printed during ray.init()
# Navigate to: http://127.0.0.1:8265
```

**Test Single Worker First**:
```python
# Before running 32 workers, test with 1
parallel_opt = RayParallelOptimizer(num_workers=1, gpu_fraction=1.0)
```

---

## API Reference

### `RayParallelOptimizer`

Main orchestrator class for distributed parallel optimization.

#### Constructor

```python
RayParallelOptimizer(
    num_workers: int = 32,
    gpu_fraction: float = 0.1,
    optimizer_method: str = "gradient_descent"
)
```

**Parameters**:
- `num_workers`: Number of parallel Ray actors
- `gpu_fraction`: GPU fraction per worker (0.1 = 10 workers per GPU)
- `optimizer_method`: Base optimizer to parallelize (`"gradient_descent"`, `"grid_search"`)

#### `optimize()`

Run parallel optimization across all workers.

```python
optimize(
    scene_config: Dict[str, Any],
    initial_positions: List[np.ndarray],
    optimization_params: Dict[str, Any],
    optimizer_configs: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = True
) -> Dict[str, Any]
```

**Parameters**:
- `scene_config`: Scene configuration (XML path, reflector name)
- `initial_positions`: List of starting positions [x, y, z] (length = num_workers)
- `optimization_params`: Parameters for `optimizer.optimize()` call
- `optimizer_configs`: Optional per-worker optimizer configs
- `verbose`: Print progress information

**Returns**: Dictionary with:
- `all_results`: List of results from all workers
- `best_result`: Result from best worker
- `best_worker_id`: ID of best worker
- `total_time`: Wall-clock time
- `aggregate_stats`: Statistics (mean, std, speedup)

#### `plot_results()`

Visualize parallel optimization results.

```python
plot_results(
    results: Dict[str, Any],
    metric_name: str = "Min RSS (dBm)"
) -> None
```

**Parameters**:
- `results`: Results dictionary from `optimize()`
- `metric_name`: Label for metric in plots

**Plots**:
1. Distribution of final metrics
2. Final positions scatter (colored by metric)
3. Execution time per worker
4. Summary statistics

#### `shutdown()`

Shutdown Ray cluster.

```python
shutdown() -> None
```

---

### `OptimizationWorker`

Ray actor that runs independent optimization trajectory.

**Note**: Typically not used directly; created internally by `RayParallelOptimizer`.

#### Constructor

```python
@ray.remote
OptimizationWorker(
    worker_id: int,
    scene_config: Dict[str, Any],
    optimizer_method: str,
    optimizer_config: Dict[str, Any],
    optimization_params: Dict[str, Any],
    gpu_fraction: float = 0.1
)
```

#### `optimize()`

Run optimization and return results.

```python
optimize() -> Dict[str, Any]
```

**Returns**: Dictionary with:
- `worker_id`: Worker identifier
- `best_position`: Optimized position [x, y, z]
- `best_metric`: Best metric achieved
- `time_elapsed`: Optimization time
- `history`: Optimization history (if available)

---

### Helper Functions

#### `generate_random_initial_positions()`

Generate diverse initial positions for parallel optimization.

```python
generate_random_initial_positions(
    num_positions: int,
    bounds: Dict[str, float],
    fixed_z: float = 3.8,
    seed: Optional[int] = None
) -> List[np.ndarray]
```

**Parameters**:
- `num_positions`: Number of positions to generate
- `bounds`: Dictionary with `x_min`, `x_max`, `y_min`, `y_max`
- `fixed_z`: Fixed z-coordinate (height)
- `seed`: Random seed for reproducibility

**Returns**: List of position arrays [x, y, z]

**Example**:
```python
positions = generate_random_initial_positions(
    num_positions=32,
    bounds={"x_min": 0, "x_max": 20, "y_min": 0, "y_max": 20},
    fixed_z=3.8,
    seed=42,
)
```

---

## Production Deployment

### Batch Job Execution

For production workloads, use Ray Jobs:

```python
# Save configuration and run as job
import json

config = {
    "num_workers": 32,
    "scene_config": {"xml_path": "scene.xml"},
    "bounds": {"x_min": 0, "x_max": 20, "y_min": 0, "y_max": 20},
    "optimization_params": {...},
}

with open("ray_job_config.json", "w") as f:
    json.dump(config, f)

# Submit job
# ray job submit --working-dir . -- python ray_optimization_job.py
```

### Resource Configuration

Specify compute requirements in cluster config:

```yaml
# cluster_config.yaml
cluster_name: reflector-opt

max_workers: 8

available_node_types:
  ray.worker.gpu:
    node_config:
      InstanceType: g5.4xlarge  # 1x A10G GPU
    min_workers: 0
    max_workers: 8
    resources: {"GPU": 1}
```

Launch cluster:
```bash
ray up cluster_config.yaml
ray attach cluster_config.yaml
```

### Checkpointing

Save intermediate results for fault tolerance:

```python
import os

def optimize_with_checkpoints(parallel_opt, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    results = parallel_opt.optimize(...)
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "latest.pkl")
    import pickle
    with open(checkpoint_path, "wb") as f:
        pickle.dump(results, f)
    
    return results
```

---

## Summary

The `RayParallelOptimizer` provides a production-ready solution for distributed parallel optimization of reflector positions. Key takeaways:

1. **Use Ray for scene geometry optimization** (reflectors, walls, obstacles)
2. **Start with 8-16 workers** and scale up based on memory
3. **Set `gpu_fraction=0.25`** as a baseline (4 workers per GPU)
4. **Generate diverse initial positions** to avoid local minima
5. **Monitor speedup** to ensure efficient parallelism

For questions or issues, refer to:
- [Ray Documentation](https://docs.ray.io)
- [Examples](../examples/ray_parallel_example.py)
- [RAY_ARCHITECTURE.md](RAY_ARCHITECTURE.md)
