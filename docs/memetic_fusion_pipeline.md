# Memetic Fusion Pipeline — Comprehensive Documentation

> **Scope:** Complete method-by-method trace of the Memetic Fusion pipeline,
> covering Phase 1 (Genetic Algorithm macro-exploration), Phase 2 (Seed-to-GD
> bridge), and Phase 3 (Gradient Descent micro-exploitation).  Includes
> objective-function alignment analysis, decision-variable comparison, and
> simplification recommendations.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Entry Point and Configuration](#2-entry-point-and-configuration)
3. [Resource Initialization (Ray Actor Pool)](#3-resource-initialization-ray-actor-pool)
4. [Phase 1 — GA Macro-Exploration](#4-phase-1--ga-macro-exploration)
   - 4.1 [Chromosome Encoding](#41-chromosome-encoding)
   - 4.2 [GA Initialization (DEAP Setup)](#42-ga-initialization-deap-setup)
   - 4.3 [Evaluation: `_evaluate_invalid()`](#43-evaluation-_evaluate_invalid)
   - 4.4 [Individual Formatting: `_format_individual()`](#44-individual-formatting-_format_individual)
   - 4.5 [Worker Dispatch: `RayActorPoolExecutor.map()`](#45-worker-dispatch-rayactorpoolexecutormap)
   - 4.6 [Worker Execution: `OptimizationWorker.optimize()`](#46-worker-execution-optimizationworkeroptimize)
   - 4.7 [Grid Search Evaluation: `SinglePointGridSearchOptimizer`](#47-grid-search-evaluation-singlepointgridsearchoptimizer)
   - 4.8 [Radio Map and Metrics: `_compute_metrics()`](#48-radio-map-and-metrics-_compute_metrics)
   - 4.9 [Fitness Extraction and Logging](#49-fitness-extraction-and-logging)
   - 4.10 [GA Loop (Selection, Crossover, Mutation)](#410-ga-loop-selection-crossover-mutation)
   - 4.11 [Hall of Fame and Seed Extraction](#411-hall-of-fame-and-seed-extraction)
5. [Phase 2 — Seed-to-GD Bridge](#5-phase-2--seed-to-gd-bridge)
6. [Phase 3 — Targeted GD Micro-Exploitation](#6-phase-3--targeted-gd-micro-exploitation)
   - 6.1 [Task Splitting and Dispatch](#61-task-splitting-and-dispatch)
   - 6.2 [GD Optimizer: `GradientDescentAPOptimizer`](#62-gd-optimizer-gradientdescentapoptimizer)
   - 6.3 [Loss Computation: `compute_loss()`](#63-loss-computation-compute_loss)
   - 6.4 [Differentiable Ray Tracing via `@dr.wrap`](#64-differentiable-ray-tracing-via-drwrap)
   - 6.5 [Reflector Optimization (SPSA)](#65-reflector-optimization-spsa)
   - 6.6 [Optimization Loop](#66-optimization-loop)
   - 6.7 [Result Aggregation](#67-result-aggregation)
7. [Loss Function Alignment Analysis](#7-loss-function-alignment-analysis)
   - 7.1 [GA Composite Loss (in `grid_search.py`)](#71-ga-composite-loss-in-grid_searchpy)
   - 7.2 [GD Composite Loss (in `gradient_descent.py`)](#72-gd-composite-loss-in-gradient_descentpy)
   - 7.3 [Alignment Guarantee](#73-alignment-guarantee)
   - 7.4 [Remaining Difference: Repulsion Loss](#74-remaining-difference-repulsion-loss)
8. [Decision Variables: What Does Each Phase Optimize?](#8-decision-variables-what-does-each-phase-optimize)
9. [Loss Functions Reference](#9-loss-functions-reference)
10. [File Dependency Map](#10-file-dependency-map)
11. [Simplification Recommendations](#11-simplification-recommendations)
12. [Configuration Reference](#12-configuration-reference)

---

## 1. High-Level Architecture

```
┌─────────────────────────────────┐
│  run_memetic_pipeline.py        │  Top-level launcher
│  (root convenience entry point) │
└──────────────┬──────────────────┘
               │ calls
               ▼
┌─────────────────────────────────────────────────────────┐
│  run_memetic_pipeline.py  (src/.../memetic/)            │
│  run_memetic_optimization(config)                       │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Step 1: Resource Init                            │   │
│  │   ray.init() → RayActorPoolExecutor              │   │
│  │   (spawn N OptimizationWorker actors)            │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Step 2: Phase 1 — GA Macro-Exploration           │   │
│  │   MemeticGeneticAlgorithmRunner.run()             │   │
│  │   → Uses executor.map for parallel fitness eval   │   │
│  │   → Returns seeds[]                              │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Step 3: Phase 2 — Bridge                         │   │
│  │   generate_gd_tasks_from_seeds()                 │   │
│  │   → Translates GA seeds → GD work items          │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Step 4: Phase 3 — GD Micro-Exploitation          │   │
│  │   RayParallelOptimizer (reuses hot actor pool)   │   │
│  │   run_targeted_gd_exploitation()                 │   │
│  │   → GradientDescentAPOptimizer per seed          │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  Step 5: Save artifacts (JSON, CSV, plots)              │
└─────────────────────────────────────────────────────────┘
```

The pipeline keeps Ray workers **hot** across both GA and GD phases through a
shared `ActorPool`, avoiding repeated scene loading (XML parsing + BVH + GPU
context warmup).

---

## 2. Entry Point and Configuration

### Root Launcher

**File:** `run_memetic_pipeline.py` (repository root)

1. Parses CLI args (`--config`, `--output-dir`, `--run-name`, `--verbose`, `--dry-run`, `--hints`).
2. Loads or generates `configs/memetic_pipeline_config.json`.
3. Deep-merges loaded config onto `_default_memetic_config()`.
4. Calls `run_memetic_optimization(config)` from the `src` module.

### Orchestrator

**File:** `src/reflector_position/optimizers/memetic/run_memetic_pipeline.py`

**Function:** `run_memetic_optimization(config_args)`

This is the true orchestration function. It:

1. Extracts all configuration fields from `config_args`.
2. **Injects alignment hyperparameters** from `gd_hyperparams` into `ga_optimization_params`:
   ```python
   _ALIGNMENT_KEYS = (
       "use_soft_min", "temperature", "shadow_quantile",
       "fairness_loss_type", "alpha", "beta",
       "coverage_threshold_dbm", "coverage_temperature",
   )
   for key in _ALIGNMENT_KEYS:
       if key not in ga_optimization_params and key in gd_hyperparams:
           ga_optimization_params[key] = gd_hyperparams[key]
   ```
   This ensures GA workers evaluate individuals using the **same composite loss
   landscape** as GD.
3. Initializes Ray and the actor pool.
4. Runs phases 1 → 2 → 3 sequentially.
5. Saves artifacts (JSON, CSV, plots) and returns a summary dict.

### Default Configuration (`_default_memetic_config`)

| Key | Default Value | Description |
|-----|--------------|-------------|
| `num_aps` | 2 | Number of access points |
| `optimize_orientation` | `True` | Whether to optimize AP look-at direction |
| `reflector_enabled` | `True` | Whether to optimize a passive reflector |
| `ga_params.pop_size` | 150 | GA population size |
| `ga_params.n_gen` | 50 | Number of GA generations |
| `ga_params.cxpb` | 0.7 | Crossover probability |
| `ga_params.mutpb` | 0.3 | Mutation probability |
| `ga_params.hof_size` | 20 | Hall of Fame size |
| `k_seeds` | 3 | Number of spatially distinct seeds to extract |
| `d_corr` | 5.0 | Min topological distance (meters) between seeds |
| `gd_hyperparams.num_iterations` | 50 | GD iterations per seed |
| `gd_hyperparams.alpha` | 0.95 | Weight for fairness loss |
| `gd_hyperparams.beta` | 0.05 | Weight for coverage loss |
| `gd_hyperparams.temperature` | 0.15 | Softmin temperature τ |

---

## 3. Resource Initialization (Ray Actor Pool)

**File:** `src/reflector_position/optimizers/ray_evaluator.py`

**Class:** `RayActorPoolExecutor`

```python
executor = RayActorPoolExecutor(
    scene_config=scene_config,
    num_workers=num_pool_workers,  # default: 2
    gpu_fraction=gpu_fraction,     # default: 0.5
)
```

**What happens:**
1. `ray.init()` is called (if not already initialized).
2. `num_workers` instances of `OptimizationWorker` are created as Ray remote
   actors. Each actor:
   - Loads the Sionna `Scene` from XML (`setup_building_floor_scene()`).
   - Parses the BVH spatial acceleration structure.
   - Initializes GPU context.
   - Optionally creates a `ReflectorController` instance.
3. All actors are placed into a `ray.util.actor_pool.ActorPool`.

The `executor.map()` interface is registered as DEAP's `toolbox.map`, enabling
the GA to evaluate individuals in parallel without knowing about Ray.

**Pool Reuse for GD:**
```python
ray_parallel_optimizer = RayParallelOptimizer(num_workers, gpu_fraction)
_bind_shared_actor_pool(ray_parallel_optimizer, executor)
```
The same hot actors serve both GA and GD phases — no scene reload.

---

## 4. Phase 1 — GA Macro-Exploration

**File:** `src/reflector_position/optimizers/memetic/memetic_ga_logic.py`

**Class:** `MemeticGeneticAlgorithmRunner`

### 4.1 Chromosome Encoding

Each DEAP individual is a flat `list[float]` of `_n_genes` values:

```
[x₀, y₀, ..., x_{N-1}, y_{N-1},   ← N·2 position genes
 dx₀, dy₀, ..., dx_{N-1}, dy_{N-1}, ← N·2 direction genes (if optimize_orientation)
 refl_u, refl_v, focal_x, focal_y]   ← 4 reflector genes (if reflector_enabled)
```

| Gene Group | Count | Bounds | Description |
|-----------|-------|--------|-------------|
| Position (x, y) per AP | `2 × num_aps` | `[x_min, x_max]`, `[y_min, y_max]` | AP physical coordinates |
| Direction (dx, dy) per AP | `2 × num_aps` | `[-1, 1]` | Look-at direction (L2-normalized with fixed `dir_z`) |
| Reflector u, v | 2 | `[0, 1]` | Wall-surface parametric coordinates |
| Reflector focal_x, focal_y | 2 | `[fx_min, fx_max]`, `[fy_min, fy_max]` | Focal-point target (in floor plane) |

**Example for 2-AP with orientation and reflector (12 genes):**
```
[x₀, y₀, x₁, y₁, dx₀, dy₀, dx₁, dy₁, u, v, focal_x, focal_y]
```

### 4.2 GA Initialization (DEAP Setup)

Inside `MemeticGeneticAlgorithmRunner.run()`:

```python
# DEAP creator types (module-level, created once)
creator.create("MemeticFitnessMax", base.Fitness, weights=(1.0,))
creator.create("MemeticIndividual", list, fitness=creator.MemeticFitnessMax)
```

**Toolbox registration:**

| Registration | Function | Notes |
|-------------|----------|-------|
| `toolbox.attr_float` | `random.uniform` per gene | Position genes: `[x_min, x_max]`; direction: `[-1, 1]`; reflector u/v: `[0, 1]`; focal: `[fx_min, fx_max]` |
| `toolbox.individual` | `tools.initCycle(MemeticIndividual, ...)` | Creates one individual from gene generators |
| `toolbox.population` | `tools.initRepeat(list, toolbox.individual)` | Creates N individuals |
| `toolbox.evaluate` | `self._format_individual` | Converts individual → worker task tuple |
| `toolbox.map` | `self._executor_map` | Injected `RayActorPoolExecutor.map` |
| `toolbox.select` | `tools.selTournament(tournsize=3)` | Tournament selection |
| `toolbox.mate` | `tools.cxBlend(alpha=0.5)` | Blend crossover |
| `toolbox.mutate` | `_split_mutate(sigma_pos=3.0, sigma_dir=0.3, ...)` | Gaussian mutation with region-specific sigmas |

### 4.3 Evaluation: `_evaluate_invalid()`

**Method:** `MemeticGeneticAlgorithmRunner._evaluate_invalid(population, toolbox)`

```
For each individual with invalid fitness:
  1. If num_aps >= 2: check _check_separation(individual)
     - If fails: assign PENALTY_SOFTMIN_FITNESS (-1e15)
     - Store ind.penalized = True, ind.best_metric_dbm = -999.0
  2. For valid individuals:
     results = toolbox.map(toolbox.evaluate, valid_inds)
     For each (ind, res):
       ind.fitness.values = (res["softmin_fitness"],)
       ind.best_metric_dbm = res["best_metric_dbm"]
       ind.best_coverage = res["grid_results"]["coverage_values"][-1]
```

**Key:** Fitness = `softmin_fitness = -total_loss` (maximized by GA).

### 4.4 Individual Formatting: `_format_individual()`

**Method:** `MemeticGeneticAlgorithmRunner._format_individual(individual)`

Returns a 4-tuple: `(task_id, "grid_search_point", optimizer_kwargs, self._opt_params)`

```python
optimizer_kwargs = {
    "evaluation_positions": [(x₀, y₀), (x₁, y₁)],  # or evaluation_position for 1-AP
    "fixed_z": 3.8,
    "evaluation_orientations": [(dx₀, dy₀, dz₀), (dx₁, dy₁, dz₁)],  # normalized
    # If reflector_enabled:
    "reflector_u": individual[rg],
    "reflector_v": individual[rg + 1],
    "reflector_target": (individual[rg+2], individual[rg+3], focal_z),
    "percentile_target_quantile": 0.05,
}
```

The second element `"grid_search_point"` is the **optimizer method** — this is
why `SinglePointGridSearchOptimizer` is the GA's **evaluation engine**.

### 4.5 Worker Dispatch: `RayActorPoolExecutor.map()`

**File:** `src/reflector_position/optimizers/ray_evaluator.py`

```python
def map(self, func, iterable):
    items = list(iterable)
    task_args = [func(item) for item in items]  # func = _format_individual
    results = list(self._pool.map(
        lambda actor, args: actor.optimize.remote(*args),
        task_args,
    ))
    return results
```

Uses **ordered** `pool.map` (not `map_unordered`) to ensure `results[i]`
corresponds to `population[i]`.

### 4.6 Worker Execution: `OptimizationWorker.optimize()`

**File:** `src/reflector_position/optimizers/ray_parallel_optimizer.py`

**Class:** `OptimizationWorker` (Ray remote actor)

```python
def optimize(self, task_id, optimizer_method, optimizer_kwargs, optimization_params):
    # 1. Attach per-worker reflector_controller (not serialized per task)
    # 2. Create PercentileCoverageObjective from scalar quantile
    # 3. Create optimizer via factory:
    optimizer = OptimizerFactory.create(
        method="grid_search_point",  # creates SinglePointGridSearchOptimizer
        scene=self.scene,
        **optimizer_kwargs,
    )
    # 4. Run optimization:
    result = optimizer.optimize(**optimization_params)
    # 5. Unpack result tuple: (position, orientation, metric)
    # 6. Build output dict with best_position, best_metric, best_metric_dbm, etc.
    # 7. Promote softmin_fitness from optimizer.results to top-level output:
    if "softmin_fitness" in gs_results:
        output["softmin_fitness"] = gs_results["softmin_fitness"]
    return output
```

### 4.7 Grid Search Evaluation: `SinglePointGridSearchOptimizer`

**File:** `src/reflector_position/optimizers/grid_search.py`

**Class:** `SinglePointGridSearchOptimizer`

**`optimize()` method:**

```python
def optimize(self, samples_per_tx, max_depth, coverage_threshold_dbm,
             use_soft_min, temperature, shadow_quantile,
             fairness_loss_type, alpha, beta, coverage_temperature,
             verbose=False):
    # 1. Build 3-D positions from evaluation_positions
    tx_positions = [[x, y, fixed_z] for (x, y) in self.evaluation_positions]

    # 2. Build orientation combos (Cartesian product of sweeps)
    combos = self._build_orientation_combos()
    #   - Fixed orientations: kept as-is (1 option)
    #   - None orientations: swept over 8 cardinal directions
    #   - Example: 2 APs both None → 64 combos (8×8)

    # 3. For each combo:
    for directions, dir_names in combos:
        self._configure_transmitters(tx_positions, directions)
        self._apply_reflector(tx_positions)  # no-op if no controller

        with torch.no_grad():
            metrics = self._compute_metrics(
                samples_per_tx, max_depth, coverage_threshold_dbm,
                use_soft_min=use_soft_min, temperature=temperature,
                shadow_quantile=shadow_quantile,
                fairness_loss_type=fairness_loss_type,
                alpha=alpha, beta=beta,
                coverage_temperature=coverage_temperature,
            )

        # Track best by RSS (or percentile_score if configured)
        if ranking > best_ranking_score:
            best = this_combo

    # 4. Store best metrics in self.results, including softmin_fitness
    # 5. Return (positions, orientations, min_rss_watts)
```

### 4.8 Radio Map and Metrics: `_compute_metrics()`

**Method:** `SinglePointGridSearchOptimizer._compute_metrics()`

This is **where the actual composite loss is computed** for GA evaluation:

```python
def _compute_metrics(self, samples_per_tx, max_depth, coverage_threshold_dbm,
                     use_soft_min, temperature, shadow_quantile,
                     fairness_loss_type, alpha, beta, coverage_temperature):
    # 1. Compute radio map via Sionna ray tracing
    rm = self._solver(self.scene, cell_size=(1.0, 1.0),
                      samples_per_tx=samples_per_tx,
                      max_depth=max_depth,
                      refraction=True, diffraction=True)
    rss_tensor = torch.from_numpy(np.array(rm.rss))

    # 2. Standard metrics
    min_rss = compute_p5_rss_metric(rss_tensor)          # 5th percentile
    min_rss_dbm = rss_to_dbm(min_rss)
    coverage = compute_coverage_metric(rss_tensor, coverage_threshold_dbm)

    # 3. Resolve fairness loss type
    effective_type = fairness_loss_type  # "auto" → masked_softmin if reflector, else softmin

    # 4. Compute fairness loss
    if effective_type == "masked_softmin":
        fairness_loss = MaskedSoftMinLoss(shadow_quantile, temperature)(rss_tensor)
    elif effective_type == "softmin":
        fairness_loss = normalized_softmin_loss(rss_tensor.flatten().unsqueeze(0), temperature)
    elif effective_type == "percentile":
        fairness_loss = -rss_to_dbm(compute_p5_rss_metric(rss_tensor)) / 100.0

    # 5. Compute coverage loss
    cov_loss = differentiable_coverage_loss(rss_tensor, coverage_threshold_dbm, coverage_temperature)

    # 6. Composite loss (SAME FORMULA AS GD)
    total_loss = alpha * fairness_loss + beta * cov_loss

    # 7. Invert for GA maximization
    result["softmin_fitness"] = -float(total_loss.item())

    return result
```

### 4.9 Fitness Extraction and Logging

Back in `_evaluate_invalid()`:

```python
ind.fitness.values = (float(res.get("softmin_fitness", PENALTY_SOFTMIN_FITNESS)),)
ind.best_metric_dbm = float(res.get("best_metric_dbm", -999.0))
```

- **GA fitness** = `softmin_fitness` = `-total_loss` (maximized)
- **Logging** uses `best_metric_dbm` (human-readable 5th-percentile RSS in dBm)

Generation logging in `run()`:
```python
gen_dbm_values = [getattr(ind, "best_metric_dbm", -999.0) for ind in population]
max_dbm = max(gen_dbm_values)
mean_dbm = np.mean(gen_dbm_values)
```

### 4.10 GA Loop (Selection, Crossover, Mutation)

```python
for gen in range(1, n_gen + 1):
    # 1. Selection (tournament, size=3)
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # 2. Crossover (blend, alpha=0.5)
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # 3. Mutation (split Gaussian: σ_pos=3.0, σ_dir=0.3, σ_refl=0.1)
    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 4. Clamping (position bounds, direction [-1,1], reflector [0,1])
    for ind in offspring:
        self._clamp_individual(ind)

    # 5. Evaluate invalid individuals (parallel via Ray)
    nevals = self._evaluate_invalid(offspring, toolbox)

    # 6. Elitism: replace population with offspring
    population[:] = offspring

    # 7. Update Hall of Fame
    hof.update(population)
```

### 4.11 Hall of Fame and Seed Extraction

**After the GA loop completes:**

```python
# Extract K spatially distinct seeds from HoF
seeds = self._extract_topological_seeds(hof, k_seeds, d_corr)
```

**`_extract_topological_seeds(hof, k_seeds, d_corr)`:**

1. Always accept `hof[0]` (the best individual).
2. Iterate remaining HoF entries in fitness order.
3. For each candidate, compute `_topological_distance()` to every accepted seed.
4. Accept if minimum distance ≥ `d_corr` meters.
5. Stop when K seeds collected.

**`_topological_distance(a, b)`:**
- 1-AP: Euclidean XY distance.
- 2-AP: `min(direct_assignment_distance, swapped_assignment_distance)` — handles
  AP label symmetry.

**Each seed is a `MemeticSeed` dataclass:**
```python
MemeticSeed(
    rank=1,
    fitness=-0.142,           # softmin_fitness
    fitness_dbm=-101.5,       # best_metric_dbm
    ap_positions=[(7.2, 8.1, 3.8), (25.0, 22.3, 3.8)],
    ap_directions=[(0.5, 0.7, -0.5), (-0.3, 0.9, -0.5)],
    reflector={"u": 0.3, "v": 0.6, "focal_x": 20.0, "focal_y": 15.0, "focal_z": 1.5},
    chromosome=[7.2, 8.1, 25.0, 22.3, 0.5, 0.7, -0.3, 0.9, 0.3, 0.6, 20.0, 15.0],
    coverage=92.5,
    min_distance_to_previous=None,
)
```

Seeds are serialized to dicts via `dataclasses.asdict()` and returned in `ga_results["seeds"]`.

---

## 5. Phase 2 — Seed-to-GD Bridge

**File:** `src/reflector_position/optimizers/memetic/memetic_bridge.py`

**Function:** `generate_gd_tasks_from_seeds(seeds, num_aps, optimize_orientation, reflector_enabled, gd_hyperparams)`

This is a **pure, side-effect-free** translation layer.

### Transformation per Seed

| GA Seed Field | GD Task Field | Transformation |
|--------------|---------------|----------------|
| `ap_positions: [(x,y,z), ...]` | `initial_positions: [(x,y), ...]` | Strip z |
| `ap_positions[0][2]` | `fixed_z` | Extract shared z |
| `ap_directions: [(dx,dy,dz), ...]` | `initial_directions_xy: [(dx,dy), ...]` | Strip z |
| `ap_directions` | `initial_orientations` | Full 3D (bridge alias) |
| `reflector.u` | `reflector_u` | Direct |
| `reflector.v` | `reflector_v` | Direct |
| `reflector.focal_*` | `reflector_target: (fx,fy,fz)` | Combine |
| `reflector.focal_*` | `initial_focal_point: (fx,fy,fz)` | GD-compatible alias |
| — | `gd_hyperparams.*` | All GD hyperparams injected |

### Validation
- AP count matches `num_aps`.
- All APs share the same z-coordinate.
- Reflector keys present when `reflector_enabled=True`.

### Result
Each task dict is **self-contained** for `RayParallelOptimizer.run(...)`.

**After bridge, the orchestrator also attaches metadata:**
```python
for seed, task in zip(seeds, gd_tasks):
    task["scene_config"] = scene_config
    task["seed_fitness_dbm"] = seed.get("fitness_dbm")
```

---

## 6. Phase 3 — Targeted GD Micro-Exploitation

### 6.1 Task Splitting and Dispatch

**File:** `src/reflector_position/optimizers/memetic/memetic_gd_logic.py`

**Function:** `run_targeted_gd_exploitation(gd_tasks, ray_optimizer, verbose)`

```python
def run_targeted_gd_exploitation(gd_tasks, ray_optimizer, verbose):
    # 1. Extract scene_config from tasks or ray_optimizer cache
    scene_config = _extract_scene_config(gd_tasks, ray_optimizer)

    # 2. Split each task into init_kwargs and optimize_kwargs
    for task in gd_tasks:
        init_kwargs, optimize_kwargs = _split_task_and_opt_params(task)
        #   init_kwargs: initial_positions, fixed_z, num_aps,
        #                optimize_orientation, initial_directions_xy,
        #                initial_focal_point
        #   optimize_kwargs: num_iterations, learning_rate, alpha, beta,
        #                    temperature, etc.

    # 3. Validate optimize_kwargs are consistent across tasks
    #    (Ray batch requires shared optimization_params)

    # 4. Dispatch via RayParallelOptimizer
    run_output = ray_optimizer.run(
        scene_config=scene_config,
        optimizer_method="gradient_descent",
        work_items=init_work_items,       # list of per-task init kwargs
        optimization_params=optimize_kwargs,  # shared GD params
        verbose=verbose,
    )

    # 5. Aggregate per-seed analysis (initial_dbm, final_dbm, delta)
    # 6. Identify global best by final GD metric
    return {
        "global_best_result": ...,
        "all_fine_tuned_results": all_results,
        "metrics": { num_tasks, max_improvement_db, mean_improvement_db, ... },
        "per_seed_analysis": [{ seed_index, initial_ga_rss_dbm, best_gd_rss_dbm, delta, ... }],
    }
```

### 6.2 GD Optimizer: `GradientDescentAPOptimizer`

**File:** `src/reflector_position/optimizers/gradient_descent.py`

**Class:** `GradientDescentAPOptimizer`

**Trainable parameters:**

| Parameter | Shape | Bounds | Description |
|-----------|-------|--------|-------------|
| `tx_x` | `[num_aps]` | `[0, 1]` (normalized) | AP x-positions |
| `tx_y` | `[num_aps]` | `[0, 1]` (normalized) | AP y-positions |
| `tx_dir_xy` | `[num_aps, 2]` | Unconstrained | AP look-at direction (dx, dy) |
| `reflector_u_raw` | scalar | Unconstrained | `sigmoid(u_raw)` → wall u ∈ [0,1] |
| `reflector_v_raw` | scalar | Unconstrained | `sigmoid(v_raw)` → wall v ∈ [0,1] |
| `focal_point_raw` | `[3]` | Unconstrained | Focal point (x, y, z) |

### 6.3 Loss Computation: `compute_loss()`

**Method:** `GradientDescentAPOptimizer.compute_loss()`

```python
def compute_loss(self, ...):
    # 1. Apply reflector params to scene (outside @dr.wrap)
    if self.reflector_controller is not None:
        self._apply_reflector_params()
        # sigmoid(u_raw) → u,  sigmoid(v_raw) → v
        # focal_point_raw → orient_to_target() → apply_to_scene()

    # 2. Build differentiable radio map via @dr.wrap
    @dr.wrap(source="torch", target="drjit")
    def compute_rss(*args):
        # Set AP positions/orientations from args
        # Run RadioMapSolver
        return rm.rss

    rss = compute_rss(tx_x[0], ..., tx_y[0], ..., dirs_x[0], ..., dirs_y[0], ...)

    # 3. Fairness loss (same dispatch as GA)
    fairness_loss = self._compute_fairness_loss(rss, fairness_loss_type, temperature, shadow_quantile)

    # 4. Coverage loss (same function as GA)
    cov_loss = differentiable_coverage_loss(rss, coverage_threshold_dbm, coverage_temperature)

    # 5. Composite (SAME FORMULA AS GA)
    coverage_loss = alpha * fairness_loss + beta * cov_loss

    # 6. Repulsion (multi-AP only, NOT in GA)
    if self.num_aps > 1 and self.repulsion_weight > 0:
        repulsion_loss = self._compute_repulsion_loss()
        total_loss = coverage_loss + self.repulsion_weight * repulsion_loss
    else:
        total_loss = coverage_loss

    return total_loss
```

### 6.4 Differentiable Ray Tracing via `@dr.wrap`

The `compute_rss` inner function uses `@dr.wrap(source="torch", target="drjit")`
to bridge PyTorch tensors into DrJit for differentiable ray tracing:

1. PyTorch tensors (AP positions, directions) → scalar DrJit arrays.
2. DrJit runs Sionna's `RadioMapSolver` with gradient tracking.
3. Output RSS tensor is automatically converted back to PyTorch.
4. `loss.backward()` propagates gradients through DrJit → PyTorch.

**Important:** Reflector geometry changes are **incompatible** with `@dr.wrap`'s AD
graph because `SceneObject.position` and `.look_at()` mutate mesh vertices via
`scene_params.update()`. Hence reflector uses SPSA.

### 6.5 Reflector Optimization (SPSA)

**Method:** `GradientDescentAPOptimizer._reflector_spsa_gradients()`

SPSA (Simultaneous Perturbation Stochastic Approximation):

```
For each iteration:
  1. Δ = random Bernoulli ±1 perturbation for each reflector param
  2. Apply +c·Δ to params → compute loss_plus
  3. Apply -c·Δ to params → compute loss_minus
  4. Restore original params
  5. grad_i ≈ (loss_plus - loss_minus) / (2·c·Δ_i)
  6. Set .grad on reflector_u_raw, reflector_v_raw, focal_point_raw
```

This requires **2 extra forward passes** per iteration (in addition to the main
differentiable forward pass for AP positions/orientations).

### 6.6 Optimization Loop

**Method:** `GradientDescentAPOptimizer.optimize()`

```python
def optimize(self, num_iterations, learning_rate, ...):
    # Parameter groups with different learning rates:
    param_groups = [
        {"params": [tx_x, tx_y],     "lr": learning_rate / pos_range},
        {"params": [tx_dir_xy],       "lr": learning_rate * 10.0},
        {"params": [refl_u_raw, refl_v_raw, focal_point_raw], "lr": learning_rate * 0.5},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Forward: differentiable loss (DrJit + PyTorch)
        loss = self.compute_loss(...)

        # Backward: automatic differentiation for AP params
        loss.backward()

        # Numerical gradients for reflector params (SPSA)
        if self.reflector_controller is not None:
            self._reflector_spsa_gradients(...)

        # Gradient clipping
        grad_x = self.tx_x.grad  # clip to [-1, 1]
        grad_y = self.tx_y.grad  # clip to [-1, 1]

        # Step
        optimizer.step()

        # Project positions back to [0, 1]
        self.apply_position_constraints()

        # Log metrics to history
```

### 6.7 Result Aggregation

Back in `run_targeted_gd_exploitation()`:

```python
# Per-seed analysis
for each (task, result):
    initial_dbm = seed's GA metric
    final_dbm = GD's best_metric_dbm
    delta = final_dbm - initial_dbm

# Global best = result with highest best_metric_dbm across all seeds
```

Output:
```json
{
  "global_best_result": { "task_id": 2, "best_metric_dbm": -98.3, ... },
  "metrics": {
    "num_tasks": 3,
    "max_improvement_db": 2.1,
    "mean_improvement_db": 1.5
  },
  "per_seed_analysis": [
    { "seed_index": 0, "initial_ga_rss_dbm": -101.5, "best_gd_rss_dbm": -99.8, "delta": 1.7 },
    ...
  ]
}
```

---

## 7. Loss Function Alignment Analysis

### 7.1 GA Composite Loss (in `grid_search.py`)

Computed inside `SinglePointGridSearchOptimizer._compute_metrics()` under
`torch.no_grad()`:

```python
total_loss = alpha * fairness_loss + beta * coverage_loss
softmin_fitness = -total_loss  # GA maximizes this
```

### 7.2 GD Composite Loss (in `gradient_descent.py`)

Computed inside `GradientDescentAPOptimizer.compute_loss()` with full gradient
tracking:

```python
composite = alpha * fairness_loss + beta * coverage_loss
total_loss = composite + repulsion_weight * repulsion_loss  # GD minimizes this
```

### 7.3 Alignment Guarantee

**Yes, GA and GD use the same loss function** for the core composite:

| Component | GA (`grid_search.py`) | GD (`gradient_descent.py`) | Same? |
|-----------|----------------------|---------------------------|-------|
| Fairness loss dispatch | `fairness_loss_type → auto/softmin/masked_softmin/percentile` | Same dispatch via `_compute_fairness_loss()` | ✅ |
| `normalized_softmin_loss()` | `metrics.py` | `metrics.py` | ✅ Same function |
| `MaskedSoftMinLoss()` | `metrics.py` | `metrics.py` | ✅ Same class |
| `differentiable_coverage_loss()` | `metrics.py` | `metrics.py` | ✅ Same function |
| Composite formula | `α · fairness + β · coverage` | `α · fairness + β · coverage` | ✅ |
| Hyperparameters | Injected from `gd_hyperparams` via alignment block | Direct from `gd_hyperparams` | ✅ |

**The `run_memetic_pipeline.py` alignment block copies 8 keys** from
`gd_hyperparams` into `ga_optimization_params`:
```
use_soft_min, temperature, shadow_quantile, fairness_loss_type,
alpha, beta, coverage_threshold_dbm, coverage_temperature
```

### 7.4 Remaining Difference: Repulsion Loss

**GA does NOT have repulsion loss.** Instead, it uses a hard constraint:

```python
# GA: hard constraint (penalty if violated)
if not self._check_separation(individual):
    ind.fitness.values = (PENALTY_SOFTMIN_FITNESS,)  # -1e15

# GD: soft differentiable penalty
repulsion_loss = sum(1 / (dist²(APi, APj) + ε))  for all i < j
total_loss += repulsion_weight * repulsion_loss
```

**Impact:** The GD total loss can be slightly different from the GA fitness for
multi-AP configurations because repulsion shifts the GD loss surface. However,
this is intentional — the GA's hard constraint prevents degenerate individuals,
while GD needs a smooth gradient signal to push APs apart.

---

## 8. Decision Variables: What Does Each Phase Optimize?

| Decision Variable | Phase 1 (GA) | Phase 3 (GD) |
|-------------------|:------------:|:------------:|
| AP positions (x, y) | ✅ Position genes in chromosome | ✅ `tx_x`, `tx_y` trainable tensors |
| AP orientation (dx, dy) | ✅ Direction genes + 8-direction sweep | ✅ `tx_dir_xy` trainable tensor |
| AP height (z) | ❌ Fixed (`fixed_z`) | ❌ Fixed (`fixed_z`) |
| Reflector wall position (u, v) | ✅ Reflector genes in chromosome | ✅ `sigmoid(reflector_u_raw)`, `sigmoid(reflector_v_raw)` + SPSA |
| Reflector focal point (x, y, z) | ✅ `focal_x`, `focal_y` genes; `focal_z` fixed | ✅ `focal_point_raw` tensor + SPSA |

**Both phases optimize all 4 variable groups** (AP position, AP orientation,
reflector wall position, reflector focal point) when `optimize_orientation=True`
and `reflector_enabled=True`.

**GA encoding differences:**
- Reflector u, v are direct [0, 1] values in the chromosome.
- GD uses unbounded `_raw` params with `sigmoid()` mapping.
- GA direction genes are L2-normalized; some individuals also undergo
  8-cardinal sweep.
- GD direction uses L2-normalized `tx_dir_xy` with gradient-based updates.

---

## 9. Loss Functions Reference

All loss functions live in `src/reflector_position/metrics.py`.

### `normalized_softmin_loss(rssi_watts, temperature=0.1)`

```
Watts → dBm → [0,1] normalized scores → τ·logsumexp(-s/τ) → loss
```
- **Lower is better** (lower loss = higher worst-case RSSI)
- Range: approximately [0, 1]
- Used when: no reflector (fairness_loss_type="softmin" or "auto" without reflector)

### `MaskedSoftMinLoss(shadow_quantile=0.05, temperature=0.1)`

```
Watts → dBm → [0,1] scores → detach(quantile threshold) → mask bottom Q%
→ -τ·logsumexp(-s_surviving/τ) → loss
```
- Shadow-aware: masks bottom `shadow_quantile` cells via **detached** quantile
- Gradients flow through surviving cell values, NOT through the mask boundary
- Returns `-soft_min` (minimizing loss = maximizing worst-case non-shadow signal)
- Used when: reflector present (fairness_loss_type="masked_softmin" or "auto" with reflector)

### `differentiable_coverage_loss(rssi_watts, threshold_dbm, temperature=2.0)`

```
Watts → dBm → (dBm - threshold) / τ → sigmoid → mean → 1 - mean → loss
```
- **Lower is better** (lower loss = more users above threshold)
- Range: [0, 1]
- Provides non-zero gradients near threshold (unlike hard step function)

### `compute_p5_rss_metric(rss_map)`

- Returns 5th-percentile RSS in linear Watts
- Used for human-readable logging (not for optimization)
- Non-differentiable (uses `torch.quantile`)

### `PercentileCoverageObjective(target_quantile=0.05)`

- Wraps `torch.quantile` in an `nn.Module`
- Used by `SinglePointGridSearchOptimizer` as an additional ranking criterion
- Not used in the composite loss

---

## 10. File Dependency Map

```
run_memetic_pipeline.py (root)
  └── src/.../memetic/run_memetic_pipeline.py
        ├── memetic_ga_logic.py            Phase 1: GA logic
        │     └── (uses executor_map → ray_evaluator.py)
        ├── memetic_bridge.py              Phase 2: seed → GD task translation
        ├── memetic_gd_logic.py            Phase 3: GD exploitation orchestration
        ├── ray_evaluator.py               ActorPool executor (generic)
        │     └── ray_parallel_optimizer.py (OptimizationWorker actor)
        │           └── optimizer_factory.py
        │                 ├── grid_search.py   (GA evaluates here)
        │                 │     └── metrics.py
        │                 └── gradient_descent.py (GD optimizes here)
        │                       └── metrics.py
        └── ray_parallel_optimizer.py       RayParallelOptimizer (GD dispatch)
              └── (same worker actors reused via _bind_shared_actor_pool)
```

**Key files (6 core modules):**

| File | Lines | Role |
|------|------:|------|
| `memetic_ga_logic.py` | ~800 | GA chromosome, DEAP loop, HoF, seed extraction |
| `memetic_bridge.py` | ~250 | Pure translation: GA seeds → GD tasks |
| `memetic_gd_logic.py` | ~350 | GD orchestration, result aggregation |
| `run_memetic_pipeline.py` | ~800 | Top-level orchestration, artifact saving |
| `grid_search.py` | ~900 | GA evaluation engine + reflector grid tasks |
| `gradient_descent.py` | ~1450 | GD optimizer, DrJit AD, SPSA |
| `ray_parallel_optimizer.py` | ~1300 | Worker actor, result serialization |
| `ray_evaluator.py` | ~200 | Generic ActorPool executor |
| `metrics.py` | ~590 | All loss functions (shared by GA and GD) |

---

## 11. Simplification Recommendations

### 11.1 Extract a Standalone `CompositeObjective` Class

**Problem:** The composite loss `α·fairness + β·coverage` is **duplicated** in
two places:
- `grid_search.py::_compute_metrics()` (GA path)
- `gradient_descent.py::compute_loss()` (GD path)

Both implement the same fairness-loss dispatch logic (`auto → masked_softmin/softmin`,
`MaskedSoftMinLoss`, `normalized_softmin_loss`, `differentiable_coverage_loss`).

**Recommendation:** Create a `CompositeObjective` class in `metrics.py`:

```python
class CompositeObjective(nn.Module):
    def __init__(self, alpha, beta, fairness_loss_type, temperature,
                 shadow_quantile, coverage_threshold_dbm, coverage_temperature,
                 reflector_active=False):
        ...

    def forward(self, rss_tensor):
        fairness = self._compute_fairness(rss_tensor)
        coverage = differentiable_coverage_loss(rss_tensor, ...)
        return alpha * fairness + beta * coverage
```

Both `grid_search.py` and `gradient_descent.py` would use `CompositeObjective()(rss)`.
This eliminates the risk of the two implementations diverging.

### 11.2 Decouple GA Evaluation from `SinglePointGridSearchOptimizer`

**Problem:** The GA evaluation is coupled to the grid search optimizer:
- `_format_individual()` returns `optimizer_method="grid_search_point"`.
- `SinglePointGridSearchOptimizer` was not designed for GA evaluation — it also
  does orientation sweeps, reflector grid tasks, and other grid search concerns.

**Recommendation:** Create a lightweight `MemeticEvaluator` class that:
1. Takes AP positions, orientations, and reflector params.
2. Configures transmitters and reflector.
3. Runs one radio-map evaluation.
4. Computes the composite loss via `CompositeObjective`.
5. Returns fitness and metrics.

This would remove the orientation sweep overhead when the GA already specifies
exact orientations and would make the GA evaluation path simpler and faster.

### 11.3 Simplify the Alignment Block

**Problem:** The alignment block in `run_memetic_pipeline.py` copies 8 keys from
`gd_hyperparams` to `ga_optimization_params`. This is fragile — adding a new
loss hyperparameter requires updating `_ALIGNMENT_KEYS`.

**Recommendation:** Make the composite loss configuration its own top-level
config section (e.g., `loss_config`), shared by both GA and GD:

```json
{
  "loss_config": {
    "alpha": 0.95,
    "beta": 0.05,
    "temperature": 0.15,
    "fairness_loss_type": "auto",
    "shadow_quantile": 0.05,
    "coverage_threshold_dbm": -120.0,
    "coverage_temperature": 2.0
  },
  "ga_params": { ... },
  "gd_hyperparams": { "num_iterations": 50, "learning_rate": 0.1, ... }
}
```

Both GA and GD would receive `loss_config` directly. No alignment block needed.

### 11.4 Reduce the Worker Output Dict Size

**Problem:** `OptimizationWorker.optimize()` builds a large output dict with
many fields (history, grid_results, positions, directions, reflector state).
For GA evaluation, most of this is unused — only `softmin_fitness`,
`best_metric_dbm`, and `coverage_values` are read.

**Recommendation:** Add a `lite_output=True` flag for GA evaluations that
returns only the fields the GA actually uses. This reduces serialization
overhead and Ray object store pressure.

### 11.5 Move SPSA to a Separate Module

**Problem:** `gradient_descent.py` is ~1450 lines and handles both the main
optimization loop and SPSA reflector gradients. The SPSA implementation
(`_reflector_spsa_gradients`, `_eval_detached_loss`, `_apply_reflector_params`)
is ~150 lines of self-contained logic.

**Recommendation:** Extract SPSA into `optimizers/spsa_reflector.py` as a
mixin or standalone class. This improves readability and makes it reusable
for potential future non-differentiable parameters.

### 11.6 Standalone Maintenance Checklist

To make the memetic pipeline fully standalone for independent maintenance:

1. **Pin the 6 core files** listed in Section 10 as the minimal set.
2. **Create `CompositeObjective`** (Section 11.1) to unify the loss computation.
3. **Remove dependency on `optimizer_factory.py`** for GA evaluation by using
   `MemeticEvaluator` (Section 11.2).
4. **Extract `loss_config`** as a shared config section (Section 11.3).
5. **Add integration tests** that verify GA and GD produce identical composite
   loss values for the same radio map — this is the most critical invariant.
6. **Document the `@dr.wrap` + SPSA split** for reflector optimization (current
   code comments are excellent but external documentation helps onboarding).

---

## 12. Configuration Reference

### Default Config (`_default_memetic_config()`)

```json
{
  "scene_config": {
    "scene_path": "~/blender/models/building_floor/building_floor.xml",
    "frequency": 5.18e9,
    "tx_power_dbm": 5.0,
    "tx_positions": [[7.0, 7.0, 3.8], [23.0, 23.0, 3.8]],
    "reflector_enabled": true,
    "reflector_size": [2.0, 2.0],
    "wall_top_left": [15.0, 34.0, 3.0],
    "wall_bottom_right": [34.0, 34.0, 1.0],
    "focal_point": [20.0, 20.0, 1.5],
    "device": "cuda"
  },
  "position_bounds": { "x_min": 5.5, "x_max": 34.5, "y_min": 5.5, "y_max": 34.5 },
  "fixed_z": 3.8,
  "num_pool_workers": 2,
  "gpu_fraction": 0.5,
  "random_seed": 4,
  "num_aps": 2,
  "min_ap_separation": 5.0,
  "optimize_orientation": true,
  "reflector_enabled": true,
  "focal_z": 1.5,
  "ga_params": {
    "pop_size": 150,
    "n_gen": 50,
    "cxpb": 0.7,
    "mutpb": 0.3,
    "tournsize": 3,
    "hof_size": 20
  },
  "ga_optimization_params": {
    "samples_per_tx": 1000000,
    "max_depth": 13,
    "verbose": false
  },
  "k_seeds": 3,
  "d_corr": 5.0,
  "gd_hyperparams": {
    "num_iterations": 50,
    "learning_rate": 0.1,
    "samples_per_tx": 1000000,
    "max_depth": 13,
    "use_soft_min": true,
    "temperature": 0.15,
    "shadow_quantile": 0.05,
    "fairness_loss_type": "auto",
    "alpha": 0.95,
    "beta": 0.05,
    "coverage_threshold_dbm": -120.0,
    "coverage_temperature": 2.0,
    "verbose": false
  },
  "verbose": true
}
```

### Key Hyperparameter Interactions

| Parameter | Effect on Loss Landscape |
|-----------|-------------------------|
| `alpha` ↑ | More weight on worst-case signal fairness; GD focuses on raising the floor |
| `beta` ↑ | More weight on coverage percentage; GD focuses on broad coverage |
| `temperature` ↓ | Sharper soft-min; approaches true minimum (less smooth, harder to optimize) |
| `shadow_quantile` ↑ | More cells masked as shadow; less conservative optimization |
| `coverage_temperature` ↓ | Sharper sigmoid at coverage threshold; steeper gradient but narrower region |
| `repulsion_weight` ↑ | Stronger AP separation force (GD only) |
| `d_corr` ↑ | Fewer but more spatially diverse seeds extracted from HoF |
| `k_seeds` ↑ | More seeds → more GD runs → better chance of finding global optimum |

---

## Appendix: Complete Call Chain (Summary)

```
run_memetic_pipeline.py (root)
  └─ main()
       └─ run_memetic_optimization(config)  [src/.../run_memetic_pipeline.py]
            ├─ Alignment: copy gd_hyperparams → ga_optimization_params
            ├─ ray.init()
            ├─ RayActorPoolExecutor(scene_config, num_workers)
            │    └─ OptimizationWorker.remote(scene_config)  ×N
            │         └─ setup_building_floor_scene()
            │              └─ Scene.load(xml), ReflectorController()
            │
            ├─ Phase 1: MemeticGeneticAlgorithmRunner(executor.map).run()
            │    ├─ DEAP toolbox setup
            │    ├─ Initial population (random chromosomes)
            │    ├─ _evaluate_invalid(population)
            │    │    ├─ _check_separation() → penalty or proceed
            │    │    ├─ toolbox.map(toolbox.evaluate, valid_inds)
            │    │    │    ├─ _format_individual(ind) → (task_id, "grid_search_point", kwargs, params)
            │    │    │    ├─ executor.map() → pool.map()
            │    │    │    │    └─ OptimizationWorker.optimize(task_id, "grid_search_point", kwargs, params)
            │    │    │    │         ├─ OptimizerFactory.create("grid_search_point") → SinglePointGridSearchOptimizer
            │    │    │    │         ├─ optimizer.optimize(**params)
            │    │    │    │         │    ├─ _build_orientation_combos()
            │    │    │    │         │    ├─ For each combo:
            │    │    │    │         │    │    ├─ _configure_transmitters()
            │    │    │    │         │    │    ├─ _apply_reflector()
            │    │    │    │         │    │    └─ _compute_metrics()  [torch.no_grad]
            │    │    │    │         │    │         ├─ RadioMapSolver → rss_tensor
            │    │    │    │         │    │         ├─ fairness_loss (MaskedSoftMinLoss / normalized_softmin_loss)
            │    │    │    │         │    │         ├─ coverage_loss (differentiable_coverage_loss)
            │    │    │    │         │    │         └─ softmin_fitness = -(α·fairness + β·coverage)
            │    │    │    │         │    └─ Store best in self.results
            │    │    │    │         └─ Promote softmin_fitness to output
            │    │    │    └─ Return ordered results
            │    │    └─ ind.fitness.values = (res["softmin_fitness"],)
            │    ├─ GA loop: select → crossover → mutate → clamp → evaluate → elitism → HoF
            │    └─ _extract_topological_seeds(hof, k=3, d_corr=5.0)
            │         └─ Distance-filtered top-K seeds → MemeticSeed dataclass → dict
            │
            ├─ Phase 2: generate_gd_tasks_from_seeds(seeds, gd_hyperparams)
            │    └─ Per seed: extract positions/directions/reflector → GD task dict
            │
            ├─ Phase 3: RayParallelOptimizer (reuses hot actor pool)
            │    └─ run_targeted_gd_exploitation(gd_tasks, ray_optimizer)
            │         ├─ _split_task_and_opt_params() per task
            │         ├─ ray_optimizer.run("gradient_descent", work_items, opt_params)
            │         │    └─ OptimizationWorker.optimize(task_id, "gradient_descent", kwargs, params)
            │         │         ├─ OptimizerFactory.create("gradient_descent") → GradientDescentAPOptimizer
            │         │         └─ optimizer.optimize(**params)
            │         │              ├─ AdamW optimizer setup (3 param groups)
            │         │              └─ For each iteration:
            │         │                   ├─ compute_loss()
            │         │                   │    ├─ _apply_reflector_params() [sigmoid-bounded]
            │         │                   │    ├─ @dr.wrap compute_rss() [differentiable ray tracing]
            │         │                   │    ├─ _compute_fairness_loss() [same functions as GA]
            │         │                   │    ├─ differentiable_coverage_loss() [same function as GA]
            │         │                   │    ├─ total = α·fairness + β·coverage + repulsion
            │         │                   │    └─ return total_loss
            │         │                   ├─ loss.backward() [DrJit AD for AP params]
            │         │                   ├─ _reflector_spsa_gradients() [2 extra forward passes]
            │         │                   ├─ optimizer.step()
            │         │                   └─ apply_position_constraints()
            │         └─ Aggregate: per-seed delta analysis, global best
            │
            └─ _save_memetic_artifacts(summary, config, output_dir)
                 ├─ JSON: memetic_summary, ga_results, gd_results, global_best, run_config
                 ├─ CSV: ga_generation_details, gd_per_seed_analysis
                 └─ Plots: ga_training_curve, gd_seed_improvements, pipeline_timing
```
