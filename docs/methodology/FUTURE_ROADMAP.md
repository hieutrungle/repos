# Future Roadmap

**Last Updated**: February 27, 2026
**Status**: Phases 1–2 Complete | Phase 3+ Planned

This document tracks completed milestones and planned enhancements for the reflector-position optimisation framework.

---

## Phase Summary

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** | Core optimisers (GD, GS) + Ray parallel | ✅ Complete |
| **Phase 2** | GA baseline + reflector-aware + experiment runner | ✅ Complete |
| **Phase 3** | PSO baseline + hybrid GA+GD | 📋 Planned |
| **Phase 4** | Joint multi-device optimisation + AO baseline | 📋 Planned |
| **Phase 5** | Advanced scene support + constraints | 💡 Future |
| **Research** | RL baselines, meta-learning, multi-objective | 💡 Future |

---

## Completed — Phase 1

### Ray-Parallel Infrastructure ✅
- Ray `ActorPool` with persistent `OptimizationWorker` actors
- Configurable GPU fraction per worker (e.g. 0.25 = 4 workers/GPU)
- Ordered `pool.map` (synchronous, freeze-safe)
- Scene loaded once per worker — reused across all tasks

### Gradient Descent (GD) ✅
- Sionna differentiable ray tracing with DrJit auto-diff
- Multi-start parallel via ActorPool (N seeds → pick best)
- Configurable fairness loss: `softmin`, `masked_softmin`, `percentile`
- Trainable: `tx_position`, `look_at_direction`

### Grid Search (GS) ✅
- `SinglePointGridSearchOptimizer` — place AP, sweep 8 cardinal orientations
- 1-AP and 2-AP modes (alternating optimisation for 2-AP)
- Embarrassingly parallel — one actor per grid point

---

## Completed — Phase 2

### Genetic Algorithm (GA) with IoC Architecture ✅
- **Library**: DEAP 1.4+ with `GeneticAlgorithmRunner` (pure DEAP, zero Ray imports)
- **Execution**: `RayActorPoolExecutor` injected via `toolbox.register("map", executor.map)`
- **Chromosome**: 4/8/12 genes for 1ap/2ap/2ap_reflector modes
- **Operators**: Blend crossover, split Gaussian mutation (σ_pos=2.0, σ_dir=0.3, σ_reflector=0.1), tournament selection (k=10)
- **Fitness**: P5 RSS (5th-percentile, linear Watts) via `PercentileCoverageObjective`
- **Constraints**: Inter-AP separation constraint with penalty fitness (1e-100)
- **Entry points**: `run_ga_modular.py`, `ray_parallel_example.py`, `ray_experiment_runner.py`

### Reflector-Aware Optimisation (All Methods) ✅

| Feature | GD | GS | GA |
|---------|----|----|-----|
| Reflector position | Sigmoid-bounded UV raw tensors | `generate_reflector_grid_tasks()` outer sweep | 4 extra genes (u, v, focal_x, focal_y) |
| Focal point | Sigmoid-bounded `focal_point_raw` tensor | Grid of (x, y) targets at fixed z | Genes clamped to `focal_bounds` |
| Gradient method | SPSA (2-point finite difference) | N/A (exhaustive) | N/A (evolutionary) |
| Objective | Fairness loss (P5 RSS) | P5 RSS via `PercentileCoverageObjective` | P5 RSS via `PercentileCoverageObjective` |
| LR multiplier | `REFLECTOR_LR_MULTIPLIER = 0.5` | — | σ_reflector = 0.1 |

### Shadow-Robust Objective ✅
- `PercentileCoverageObjective` — constructed inside worker from `percentile_target_quantile`
- Maximises 5th-percentile RSS (P5 RSS) instead of minimum or mean
- Handles antenna shadow naturally — robust to a few deep nulls
- Used across all three methods (GD, GS, GA)

### Experiment Runner ✅
- Config-driven batch execution: `ray_experiment_runner.py`
- JSON config with `global_defaults` + per-trial overrides
- 259 production trials (all method × mode × hyperparameter combinations)
- 19 smoke-test trials for quick validation
- Per-trial result JSON + summary CSV + top-K ranking
- Automatic `num_workers` / `gpu_fraction` from `gpu_config`

---

## Planned — Phase 3

### Particle Swarm Optimization (PSO)

**Priority**: Recommended — stronger continuous baseline for publication.

#### Plan
- Library: `pyswarm` or custom implementation
- Follow IoC pattern: `PSORunner` + `executor.map` (same as GA)
- Fitness via `SinglePointGridSearchOptimizer` (same worker interface)
- Swarm: 30–50 particles, 50–100 iterations
- Parameters: inertia ω=0.8, cognitive c1=2.0, social c2=2.0

#### Key Comparison
> PSO moves based on stochastic velocity vectors. DRT moves based on the physical gradient of the radio environment.

### Hybrid GA + GD

**Priority**: Research extension.

#### Plan
- Use GA to find promising regions (broad exploration)
- Refine top-K individuals with GD (local gradient descent)
- Compare with pure GA and pure multi-start GD

---

## Planned — Phase 4

### Alternating Optimisation (AO) Baseline

**Priority**: Optional — demonstrates benefit of joint optimisation.

#### Plan
- Cycle: fix reflector → optimise APs (GD) → fix APs → optimise reflector (GD) → repeat
- 10–20 cycles with convergence check
- Compare against joint GD to show local-minima trapping when variables are decoupled

### Joint Multi-Device Optimisation

#### Plan
- Extend GD to jointly optimise: AP positions + AP orientations + reflector placement
- This is partially done already (2ap_reflector mode), but future work includes:
  - Multiple reflectors
  - RIS phase-shift optimisation (requires Sionna RIS support)
  - Higher-dimensional search spaces

---

## Future — Phase 5+

### Advanced Scene Support
- **Multi-floor buildings**: Per-floor optimisation with inter-floor interference
- **Dynamic environments**: Time-varying obstacles, adaptive re-optimisation
- **Realistic constraints**: Wall-mount only, ceiling height limits, cable routing

### Performance Optimisations
- Mixed-precision training (float16 for ray tracing)
- Gradient checkpointing for large batch sizes
- Adaptive ray sampling (more rays near convergence)

### Research Extensions
- **Deep RL baseline**: PPO/SAC agent for sequential placement decisions
- **Meta-learning**: Train on multiple floor plans, few-shot adapt to new environments
- **Multi-objective**: Pareto front (coverage vs. power vs. cost) via NSGA-II

### Documentation & Tooling
- Auto-generated API reference (Sphinx from docstrings)
- Interactive Jupyter tutorials
- Visualisation dashboard (Plotly/Dash)

---

## Cross-Reference

| Document | Content |
|----------|---------|
| [OPTIMIZATION_WORKFLOW.md](OPTIMIZATION_WORKFLOW.md) | Full system architecture and per-method workflow |
| [GA_DEAP_IMPLEMENTATION.md](GA_DEAP_IMPLEMENTATION.md) | Detailed GA implementation guide |
| [BASELINES.md](BASELINES.md) | All baseline methods with comparison tables |
| [RAY_ARCHITECTURE.md](RAY_ARCHITECTURE.md) | Ray ActorPool architecture |
| [RAY_PARALLEL_GUIDE.md](RAY_PARALLEL_GUIDE.md) | Practical guide to running parallel experiments |
| [RAY_IMPLEMENTATION_SUMMARY.md](RAY_IMPLEMENTATION_SUMMARY.md) | Implementation summary and design decisions |
| [../guides/RAY_EXPERIMENT_RUNNER.md](../guides/RAY_EXPERIMENT_RUNNER.md) | Experiment runner usage and config format |
