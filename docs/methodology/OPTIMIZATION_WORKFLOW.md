# Optimization Workflow

**Last Updated**: February 27, 2026

## Overview

This document describes the **distributed parallel optimization workflow** used in the Reflector Position Optimization framework. The system implements three optimization methods — **Gradient Descent (GD)**, **Grid Search (GS)**, and **Genetic Algorithm (GA)** — all sharing the same Ray ActorPool infrastructure. Each method supports three operating modes: `1ap` (single AP), `2ap` (dual AP), and `2ap_reflector` (dual AP + passive reflector).

### Key Design Decisions

1. **Ray ActorPool** (not per-task actors): Fixed pool of N persistent workers process M >> N tasks. Scene loaded once per worker, reused for all tasks.
2. **Inversion of Control (IoC)**: Algorithm logic (DEAP GA) is decoupled from execution engine (Ray) via dependency injection of a `map` callable.
3. **Shadow-robust objective**: 5th-percentile RSS (`PercentileCoverageObjective`) replaces hard minimum, which is trapped inside reflector dead zones.
4. **SPSA for reflector gradients**: SceneObject vertex manipulation is incompatible with DrJit AD, so reflector parameters use 2-point finite-difference gradient estimates within the GD optimizer.

---

## System Architecture

### ActorPool Pattern (All Methods)

```
┌─────────────────────────────────────────────────────────────────┐
│          Orchestrator / Driver  (single process)                │
│                                                                 │
│   RayParallelOptimizer        or        RayActorPoolExecutor    │
│   (GD, GS methods)                      (GA via DEAP)          │
│                                                                 │
│         │                                    │                  │
│    pool.map_unordered()              pool.map (ordered)         │
│         │                                    │                  │
│    ┌────┼────┬────────┬──────┐    ┌──────────┼──────────┐       │
│    ▼    ▼    ▼        ▼      ▼    ▼          ▼          ▼       │
│   W0   W1   W2  ...  Wk    W0   W1         W2    ...  Wk       │
│                                                                 │
│   Each Worker has:                                              │
│     • Persistent Scene (loaded once)                            │
│     • Optional ReflectorController                              │
│     • Fresh optimizer per task                                  │
│     • Configurable GPU fraction (0.5 = 2 workers/GPU)           │
└─────────────────────────────────────────────────────────────────┘
```

### Component File Map

| Component | File | Purpose |
|-----------|------|---------|
| `OptimizationWorker` | `ray_parallel_optimizer.py` | Ray actor — persistent Scene + optimizer creation per task |
| `RayParallelOptimizer` | `ray_parallel_optimizer.py` | Orchestrator for GD and GS tasks |
| `RayActorPoolExecutor` | `ray_evaluator.py` | Generic ordered `pool.map` engine (IoC target) |
| `GeneticAlgorithmRunner` | `deap_logic.py` | Pure DEAP logic — no Ray imports |
| `GradientDescentAPOptimizer` | `gradient_descent.py` | Differentiable AP + reflector optimizer |
| `SinglePointGridSearchOptimizer` | `grid_search.py` | Single-position evaluator (used by GA + GS) |
| `OptimizerFactory` | `optimizer_factory.py` | Factory dispatching to GD/GS/single-point |

---

## Three Optimization Modes

### `1ap` — Single Access Point

- **Chromosome (GA)**: `[x, y, dir_x, dir_y]` (4 genes)
- **GD**: 1 trainable position + 1 trainable direction
- **GS**: Grid sweep over (x, y), 8-direction orientation sweep per point

### `2ap` — Dual Access Point

- **Chromosome (GA)**: `[x1, y1, x2, y2, dir1_x, dir1_y, dir2_x, dir2_y]` (8 genes)
- **GD**: 2 trainable positions + 2 trainable directions + repulsion loss
- **GS**: Alternating grid search — fix AP1, sweep AP2; fix AP2, sweep AP1; repeat

### `2ap_reflector` — Dual AP + Passive Reflector

- **Chromosome (GA)**: `[x1, y1, x2, y2, d1x, d1y, d2x, d2y, refl_u, refl_v, focal_x, focal_y]` (12 genes)
- **GD**: 2 AP positions + 2 directions + reflector `(u, v)` + focal point `(x, y, z)` — all jointly optimised. Reflector gradients via SPSA.
- **GS**: Outer loop sweeps reflector UV × focal-target grid; inner loop alternates AP positions via grid search.

---

## Method 1: Gradient Descent (GD)

### Architecture

Multi-start gradient descent via Ray: the orchestrator generates N random initial configurations and distributes them across the worker pool. Each worker runs a full GD trajectory independently.

```
Orchestrator
  │
  ├── generate_random_initial_positions(N, bounds, seed)
  │
  └── pool.map_unordered(worker.optimize, tasks)
        │
        ├─ Worker0: GradientDescentAPOptimizer(pos_0) → 50 iterations → result_0
        ├─ Worker1: GradientDescentAPOptimizer(pos_1) → 50 iterations → result_1
        ├─ ...
        └─ WorkerN: GradientDescentAPOptimizer(pos_N) → 50 iterations → result_N
        │
        └── Winner selection: argmax(best_metric across all tasks)
```

### Trainable Parameters

| Parameter | Tensor | Gradient Source | Learning Rate |
|-----------|--------|----------------|---------------|
| AP positions `(x, y)` | `tx_x`, `tx_y` (normalised) | DrJit AD (differentiable RT) | `lr / pos_range` |
| AP directions `(dx, dy)` | `tx_dir_xy` | DrJit AD | `lr × 10` |
| Reflector wall UV | `reflector_u_raw`, `reflector_v_raw` | SPSA (2-point finite diff) | `lr × 0.5` |
| Reflector focal point | `focal_point_raw` | SPSA (2-point finite diff) | `lr × 0.5` |

### Loss Functions

The total loss is a weighted combination:

$$L = \alpha \cdot L_{\text{fairness}} + \beta \cdot L_{\text{coverage}} + L_{\text{repulsion}}$$

Where `fairness_loss_type` selects:

| Type | Class | When Used |
|------|-------|-----------|
| `softmin` | `normalized_softmin_loss` | Default for AP-only modes |
| `masked_softmin` | `MaskedSoftMinLoss` | Default for reflector modes (`auto`) |
| `percentile` | `PercentileCoverageObjective` | Shadow-robust; ignores bottom q% |
| `auto` | — | Selects `masked_softmin` if reflector present, else `softmin` |

The coverage component uses `differentiable_coverage_loss` (sigmoid approximation).

### History Tracking

Every iteration records: positions, directions, look-at targets, P5 RSS, coverage, total loss, fairness loss, coverage loss, repulsion loss, gradients, AP distances, reflector UV, reflector focal point, reflector world position.

---

## Method 2: Grid Search (GS)

### 1-AP Mode

Simple parallel grid: generate all `(x, y)` points at the specified resolution, submit each as an independent task. Each task evaluates 8 cardinal orientations.

```
generate_grid_positions(bounds, resolution)  →  N grid points
pool.map_unordered(worker.optimize, grid_tasks)
Winner = argmax(best_metric)
```

### 2-AP Mode (Alternating Optimisation)

Alternating grid search cycles:

```
for round in 1..num_rounds:
    Fix AP2 at current best, sweep AP1 over grid  →  update AP1 best
    Fix AP1 at current best, sweep AP2 over grid  →  update AP2 best
```

A `min_ap_separation` filter prunes grid points too close to the fixed AP.

### 2-AP + Reflector Mode

Nested outer/inner loops:

```
for outer_round in 1..outer_rounds:
    1. Fix APs at current best
    2. Generate reflector grid: u_steps × v_steps × target_grid
    3. Pool-evaluate all reflector configurations
    4. Update best reflector (u, v, target)

    for inner_round in 1..num_rounds:
        5. Fix reflector + AP2, sweep AP1  →  update AP1
        6. Fix reflector + AP1, sweep AP2  →  update AP2
```

The reflector grid is generated by `generate_reflector_grid_tasks()`:
- `u` ∈ linspace(0, 1, u_steps), `v` ∈ linspace(0, 1, v_steps)
- Target points: grid over position bounds at `target_resolution`
- Each task evaluates one (u, v, target) with fixed APs

Worker evaluation uses `PercentileCoverageObjective` for shadow-robust scoring.

---

## Method 3: Genetic Algorithm (GA)

### Architecture (IoC Pattern)

```
┌───────────────────────────────────────────────────┐
│  run_ga_modular.py / ray_experiment_runner.py     │
│  Wire executor.map  →  GA runner                  │
└───────┬───────────────────────────┬───────────────┘
        │                           │
        ▼                           ▼
┌──────────────────────┐  ┌──────────────────────────┐
│  ray_evaluator.py    │  │    deap_logic.py          │
│  RayActorPoolExecutor│  │  GeneticAlgorithmRunner   │
│                      │  │                           │
│  pool.map (ordered)  │  │  Pure DEAP (no Ray)       │
│  Returns result dicts│  │  toolbox.map = injected   │
└──────────────────────┘  └──────────────────────────┘
```

### Chromosome Encoding

| Mode | Genes | Layout |
|------|-------|--------|
| 1-AP | 4 | `[x, y, dir_x, dir_y]` |
| 2-AP | 8 | `[x1, y1, x2, y2, dx1, dy1, dx2, dy2]` |
| 2-AP + reflector | 12 | `[x1, y1, x2, y2, dx1, dy1, dx2, dy2, refl_u, refl_v, focal_x, focal_y]` |

### Evolutionary Operators

| Operator | DEAP Function | Parameters |
|----------|--------------|------------|
| Crossover | `cxBlend` | α = 0.5 |
| Mutation | `_split_mutate` | σ_pos=2.0, σ_dir=0.3, σ_reflector=0.1, indpb=0.2 |
| Selection | `selTournament` | tournsize = 10 |

Split mutation applies different σ values to position, direction, and reflector genes — reflecting their different scales.

### Constraints

- **Bounds**: Position genes clamped to `[x_min, x_max]`, direction to `[-1, 1]`, reflector UV to `[0, 1]`, focal to spatial bounds.
- **Separation** (2-AP): Individuals with inter-AP distance < `min_ap_separation` receive penalty fitness (`1e-100` Watts ≈ −970 dBm) without ray-tracing.

### Fitness

- 1-AP and 2-AP modes: maximise `best_metric` (P5 RSS in linear Watts)
- 2-AP + reflector mode: maximise `best_metric` via `PercentileCoverageObjective` (shadow-robust)

### Per-Generation Flow

```
Population  →  filter invalid fitness
             →  check separation constraint (penalise violators)
             →  format valid individuals via _format_individual()
             →  executor.map(evaluate, iterable)  →  ordered results
             →  assign fitness from result["best_metric"]
             →  tournament selection → crossover → mutation → clamp
             →  update HallOfFame + statistics
```

---

## Experiment Runner

The unified `ray_experiment_runner.py` automates batch execution across all methods and modes.

### JSON Config Schema

```json
{
  "shared": { "num_pool_workers": 2, "gpu_fraction": 0.5, ... },
  "trials": [ { "name": "...", "method": "gd", "mode": "2ap_reflector", ... } ],
  "sweep_groups": [ { "name_prefix": "...", "grid": { "param": [v1, v2] } } ]
}
```

- **`shared`**: Default parameters merged into every trial.
- **`trials`**: Explicit trial definitions (one method × one mode each).
- **`sweep_groups`**: Cartesian product over hyperparameter grids × random seeds.

### Trial Execution Flow

```
1. Load JSON config
2. _build_trials()  →  expand explicit + sweep_groups → flat list
3. For each trial:
   a. Create output directory
   b. TeeStream captures stdout to terminal + log file
   c. _run_trial_method(trial, output_dir)  →  dispatches to GD/GS/GA
   d. _extract_summary_row(trial, results, run_dir)
4. _save_summary_files(run_root, rows, detailed)
   → summary.csv, summary.json, all_trials_detailed.json
```

### Config Sizes

- **Production**: `ray_experiment_runner_config.example.json` — 259 trials
- **Smoke test**: `ray_experiment_runner_config.smoke_test.json` — 19 trials

---

## Reflector-Aware Optimization (All Methods)

### Reflector Parameterisation

The reflector is characterised by:
- **Wall-surface coordinates** `(u, v)` ∈ [0, 1]²: position on the wall bounding box defined by `wall_top_left` and `wall_bottom_right`
- **Focal point** `(x, y, z)`: 3-D target the reflector aims at; the `z` component is fixed at receiver height

`ReflectorController` translates `(u, v)` → world position and computes the orientation to aim at the focal point, then updates the scene mesh vertices.

### Per-Method Approach

| Aspect | GD | GS | GA |
|--------|----|----|-----|
| **Reflector params** | `torch.sigmoid`-bounded raw tensors | Discrete grid (u_steps × v_steps × target_grid) | 4 continuous genes `[u, v, fx, fy]` |
| **Gradient source** | SPSA (2 extra forward passes/iter) | N/A (exhaustive) | Evolutionary (crossover + mutation) |
| **Integration** | ReflectorController in optimizer | ReflectorController in worker per task | Worker constructs `PercentileCoverageObjective` from gene values |
| **Objective** | Configurable (`auto` → masked_softmin) | `PercentileCoverageObjective` | `PercentileCoverageObjective` |

### Shadow-Robust Objective

The passive reflector creates a physical dead zone (shadow) behind it, affecting ~2-5% of coverage cells. The hard minimum RSS is always trapped in this dead zone. The **5th-percentile RSS** (P5) is robust:

$$P_5 = \text{quantile}(\{r \in \mathcal{R} : r > \epsilon\},\; q=0.05)$$

This is implemented as `PercentileCoverageObjective(target_quantile=0.05, mode="maximize")`.

---

## Winner Selection and Output

### Per-Task Result Dictionary

Every worker returns a standardised dict:

```python
{
    "task_id": int,
    "worker_id": int,
    "num_aps": int,
    "best_position": [x, y, z],       # or [[x1,y1,z1], [x2,y2,z2]]
    "best_metric": float,              # P5 RSS (linear Watts)
    "best_metric_dbm": float,          # P5 RSS (dBm)
    "best_direction": [...],
    "reflector_u": float,
    "reflector_v": float,
    "reflector_target": [fx, fy, fz],
    "reflector_position": [x, y, z],
    "time_elapsed": float,
    "history": {...},                  # GD iterations
    "grid_results": {...},             # GS evaluation details
}
```

### Aggregation

- **GD**: Best task = `argmax(best_metric)` across all multi-start trajectories
- **GS**: Best task = `argmax(best_metric)` across all grid points
- **GA**: Best individual from Hall of Fame after final generation

---

## Resource Management

### GPU Fraction

Each `OptimizationWorker` is allocated a configurable GPU fraction:

| Configuration | Workers/GPU | Use Case |
|---------------|------------|----------|
| `gpu_fraction=1.0` | 1 | Large scenes, high sample count |
| `gpu_fraction=0.5` | 2 | Default for most runs |
| `gpu_fraction=0.25` | 4 | Small scenes, many parallel tasks |

### Memory Considerations

- Scene + BVH + ray-tracing buffers: ~1-2 GB per worker
- Each worker has its own `ReflectorController` (mesh vertices modified in-place)
- No scene state is shared across workers — full process-level isolation

---

## Cross-References

- **[RAY_ARCHITECTURE.md](RAY_ARCHITECTURE.md)** — Why Ray vs vectorised batching
- **[RAY_PARALLEL_GUIDE.md](RAY_PARALLEL_GUIDE.md)** — Complete usage guide with code examples
- **[RAY_IMPLEMENTATION_SUMMARY.md](RAY_IMPLEMENTATION_SUMMARY.md)** — Implementation status and file map
- **[GA_DEAP_IMPLEMENTATION.md](GA_DEAP_IMPLEMENTATION.md)** — Detailed GA implementation guide
- **[BASELINES.md](BASELINES.md)** — Comparison with PSO and Alternating Optimisation
- **[RAY_EXPERIMENT_RUNNER.md](../../docs/guides/RAY_EXPERIMENT_RUNNER.md)** — Config-driven batch runner guide
