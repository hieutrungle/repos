# Baseline Comparison Methods

**Last Updated**: February 27, 2026
**Status**: Grid Search ✅ | Genetic Algorithm ✅ | PSO 📋 | AO 📋

## Overview

To validate the effectiveness of Differentiable Ray Tracing (DRT) for AP and reflector positioning, we compare against established optimization baselines. This document outlines all implemented and planned baselines.

## Why Multiple Baselines Matter

Grid Search alone is insufficient for Tier-1 venue publications (e.g. *IEEE TWC*). You need baselines that represent the state of the art for non-convex wireless placement optimisation. Currently two are complete (GS, GA), and two are planned (PSO, AO).

---

## Implemented Baselines

### 1. Grid Search (GS) — Exhaustive Baseline ✅

**Implementation**: `src/reflector_position/optimizers/grid_search.py`

#### Modes

| Mode | Sweep Space | Complexity |
|------|------------|------------|
| `1ap` | Position grid × 8 cardinal directions | O(grid² × 8) |
| `2ap` | Alternating — fix AP0, sweep AP1, swap | O(grid² × 8 × 2 rounds) |
| `2ap_reflector` | Above + outer reflector sweep (u × v × focal_xy) | O(grid² × 8 × u × v × f²) |

#### Key Properties
- **Evaluator**: `SinglePointGridSearchOptimizer` — places AP(s) at a fixed grid point, sweeps 8 cardinal orientations, returns best RSS metric
- **Objective**: 5th-percentile RSS (linear Watts) via `PercentileCoverageObjective`
- **Reflector sweep**: `generate_reflector_grid_tasks()` produces the Cartesian product of uniform `(u, v)` surface coordinates and `(x, y)` focal-point targets
- **Parallelism**: Embarrassingly parallel — one Ray actor per grid point
- **Direction sweep**: 8 cardinal/intercardinal unit vectors on the XY plane (N, NE, E, SE, S, SW, W, NW)

#### Worker Interface
Each grid point is submitted as a task to `OptimizationWorker.optimize()` with:
```python
{
    "method": "grid_search_point",
    "evaluation_positions": [(x, y)],
    "evaluation_orientations": [(dx, dy, dz)],  # or None → sweep
    "reflector_u": 0.3,         # optional
    "reflector_v": 0.7,         # optional
    "reflector_target": (20, 15, 1.5),  # optional
    "percentile_target_quantile": 0.05,
}
```

#### Strengths & Weaknesses
- **Strength**: Guaranteed to find the global optimum within the grid resolution
- **Weakness**: Exponential scaling — intractable for high-dimensional joint optimisation; discrete resolution limits solution quality

---

### 2. Genetic Algorithm (GA) — Evolutionary Baseline ✅

**Implementation**: `src/reflector_position/optimizers/deap_logic.py` + `ray_evaluator.py`

**Full Details**: See [GA_DEAP_IMPLEMENTATION.md](GA_DEAP_IMPLEMENTATION.md)

#### Key Properties

| Property | Value |
|----------|-------|
| Library | DEAP 1.4+ |
| Architecture | IoC — `GeneticAlgorithmRunner` (pure DEAP) + `RayActorPoolExecutor` |
| Chromosome | 4 / 8 / 12 genes (1ap / 2ap / 2ap_reflector) |
| Fitness | Maximise P5 RSS (5th-percentile, linear Watts) |
| Crossover | Blend (`cxBlend`, α=0.5) |
| Mutation | Split Gaussian — σ_pos=2.0, σ_dir=0.3, σ_reflector=0.1 |
| Selection | Tournament (k=10) |
| Population | 150 individuals, 50 generations |

#### Reflector-Aware Encoding (12 genes)
```
[x1, y1, x2, y2, dx1, dy1, dx2, dy2, refl_u, refl_v, focal_x, focal_y]
```
- UV genes ∈ [0, 1] — wall-surface parameterisation
- Focal genes bounded by `focal_bounds` dict
- `focal_z` fixed at receiver height (1.5 m)
- Separation constraint on AP pairs (penalty fitness 1e-100)

#### Evaluation Flow
1. `_format_individual()` converts genes → worker kwargs (positions, orientations, reflector params)
2. `executor.map()` calls `OptimizationWorker.optimize()` per individual
3. Worker evaluates via `SinglePointGridSearchOptimizer` → P5 RSS
4. Fitness and reflector attributes stored on the DEAP individual

#### Strengths & Weaknesses
- **Strength**: Population-based — more robust to local minima; handles mixed variables; black-box (no gradient needed)
- **Weakness**: Requires O(pop × gen) forward simulations — typically 150 × 50 = 7,500 evaluations

---

### 3. Gradient Descent (GD) — Physics-Aware DRT Method ✅

**Implementation**: `src/reflector_position/optimizers/gradient_descent.py`

#### Key Properties

| Property | Value |
|----------|-------|
| Framework | Sionna + DrJit auto-differentiation |
| Modes | `1ap`, `2ap`, `2ap_reflector` |
| AP parameters | `tx_position`, `look_at_direction` (differentiable) |
| Reflector parameters | `reflector_u_raw`, `reflector_v_raw`, `focal_point_raw` (sigmoid-bounded, SPSA) |
| Loss functions | `softmin`, `masked_softmin`, `percentile` (fairness-mode configurable) |
| Multi-start | N random seeds → ActorPool → pick best |

#### Reflector Gradient Estimation
DrJit AD cannot differentiate through SceneObject vertex manipulation (mesh repositioning). The solution:

- **AP parameters**: Full auto-diff via DrJit tape
- **Reflector parameters**: SPSA (Simultaneous Perturbation Stochastic Approximation)
  - 2-point finite-difference: perturb all reflector params, forward pass twice, estimate gradient
  - Learning rate multiplier: `REFLECTOR_LR_MULTIPLIER = 0.5` (relative to AP LR)
  - Direction LR multiplier: `DIR_LR_MULTIPLIER = 10.0`

#### Trainable Parameters (sigmoid-bounded)

| Parameter | Raw Variable | Bounds |
|-----------|-------------|--------|
| AP position | `tx_position` | Soft-bounded by optimiser step |
| Look-at direction | `look_at_direction` | Free (normalised) |
| Reflector U | `reflector_u_raw` | sigmoid → [0, 1] |
| Reflector V | `reflector_v_raw` | sigmoid → [0, 1] |
| Focal point | `focal_point_raw` | sigmoid → [fx_min, fx_max] etc. |

#### Strengths & Weaknesses
- **Strength**: Exploits physics-based gradients — converges in ~100 steps; infinite position resolution
- **Weakness**: Susceptible to local minima (mitigated by multi-start); SPSA adds noise for reflector params

---

## Planned Baselines

### 4. Particle Swarm Optimization (PSO)

**Timeline**: Phase 3 (planned)

#### Why PSO?
- Fast convergence for continuous coordinate problems
- Standard benchmark in UAV/RIS placement literature
- Complements GA as a swarm-based (vs. evolutionary) heuristic

#### Planned Configuration
- **Particles**: 30–50
- **Iterations**: 50–100
- **Inertia weight**: 0.7–0.9
- **Cognitive/social coefficients**: 1.5–2.0 each

#### Integration Plan
- Follow the same IoC pattern: `PSORunner` + `executor.map`
- Fitness via `SinglePointGridSearchOptimizer` (same as GA)
- Report total evaluations = particles × iterations

---

### 5. Alternating Optimization (AO)

**Timeline**: Phase 4 (planned)

#### Why AO?
- Standard analytical method for coupled non-convex problems
- Demonstrates the benefit of joint optimisation (DRT advantage)

#### Planned Approach
- Cycle: fix reflector → optimize APs (GD) → fix APs → optimize reflector (GD) → repeat
- 10–20 cycles with convergence check

---

## Baseline Comparison Table

| Baseline | Type | Evaluations | Gradients | Reflector | Status |
|----------|------|-------------|-----------|-----------|--------|
| **Grid Search** | Exhaustive | O(grid² × dirs) | None | Outer sweep | ✅ |
| **Genetic Algorithm** | Evolutionary | O(pop × gen) | None | 4 extra genes | ✅ |
| **Gradient Descent** | Physics-aware | O(iters × starts) | AD + SPSA | Joint (sigmoid-bounded) | ✅ |
| **PSO** | Swarm | O(particles × iters) | None | — | 📋 |
| **AO** | Analytical | O(cycles × per-var) | Per-subproblem | Decoupled | 📋 |

### Comparison Axes

| Metric | Grid Search | GA | GD (Ours) |
|--------|-------------|-----|-----------|
| **Objective** | P5 RSS | P5 RSS | Fairness loss (P5) |
| **Solution quality** | Limited by grid | Stochastic convergence | Gradient-guided |
| **Compute cost** | Highest (exhaustive) | High (pop × gen) | Lowest (gradient) |
| **Local minima** | N/A (exhaustive) | More robust | Susceptible (multi-start) |
| **Reflector handling** | Outer sweep | Co-evolved | Joint SPSA |
| **Resolution** | Discrete (grid spacing) | Continuous (mutation) | Continuous (gradient) |

---

## Evaluation Metrics

All baselines are compared on the same metrics:
1. **P5 RSS (dBm)**: Primary — 5th-percentile received signal strength
2. **Mean RSS (dBm)**: Secondary — average coverage
3. **Total evaluations**: Number of ray-tracing forward passes
4. **Wall-clock time**: Including scene loading and overhead
5. **Robustness**: Variance across random seeds (10+ runs for stochastic methods)

---

## Implementation Priority

| Phase | Baseline | Status |
|-------|----------|--------|
| Phase 1 | Grid Search | ✅ Complete |
| Phase 2 | Gradient Descent (DRT) | ✅ Complete |
| Phase 2 | Genetic Algorithm (DEAP + IoC) | ✅ Complete |
| Phase 3 | PSO | 📋 Planned |
| Phase 4 | AO | 📋 Planned |

---

## References

- **GA for RIS Placement**: Uses GA as the standard heuristic benchmark in RIS/UAV literature
- **PSO for AP Placement**: Standard swarm-based benchmark for continuous positioning
- **AO for Joint Optimisation**: Traditional approach for coupled non-convex problems
