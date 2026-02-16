# Baseline Comparison Methods

## Overview

To validate the effectiveness of Differentiable Ray Tracing (DRT) for AP and reflector positioning, we compare against established optimization baselines. This document outlines recommended baseline methods for Tier-1 publication venues.

## Why Multiple Baselines Matter

Grid Search alone is insufficient as a baseline because it is widely considered "naive" and scales poorly (O(n^d) complexity). For venues like *IEEE Transactions on Wireless Communications*, you need baselines that represent the "state of the art" for non-convex optimization.

## Recommended Baselines

### 1. Genetic Algorithm (GA) - **Gold Standard Baseline** ✅ IMPLEMENTED

**Priority**: **COMPLETE** — Implemented using DEAP library with Ray-parallel fitness evaluation.

**Full Implementation Details**: See [GA_DEAP_IMPLEMENTATION.md](GA_DEAP_IMPLEMENTATION.md)

#### Why Use GA?
- **Industry Standard**: In 2024-2025 literature, GA is the default "heuristic" benchmark for placement problems (RIS, UAV, Base Station)
- **Handles Mixed Variables**: Works well with discrete and continuous variables without needing gradients
- **Established Credibility**: If you beat GA, you beat the industry standard for heuristic optimization

#### Implementation Details
- **Population Size**: 50 individuals (random sets of AP + RIS coordinates)
- **Generations**: 50-100
- **Fitness Function**: Same `coverage_loss` used for DRT method
- **Selection**: Tournament or roulette wheel
- **Crossover**: Uniform or single-point for coordinates
- **Mutation**: Gaussian noise with adaptive rate

#### Expected Performance
- **Strengths**: Eventually finds good solutions, robust to local minima
- **Weaknesses**: Requires thousands of forward-pass simulations (Population × Generations)

#### Key Argument
> "While GA eventually finds good solutions, it requires thousands of forward-pass simulations (Population × Generations). Our DRT method converges in ~100 steps with gradient guidance."

#### Python Implementation ✅
Using `DEAP` framework (implemented in `src/reflector_position/optimizers/deap_logic.py`):
```python
from reflector_position.optimizers import RayActorPoolExecutor, GeneticAlgorithmRunner
import ray

ray.init()
executor = RayActorPoolExecutor(scene_config={...}, num_workers=4, gpu_fraction=0.25)

ga = GeneticAlgorithmRunner(
    position_bounds={"x_min": 5, "x_max": 25, "y_min": 5, "y_max": 25},
    fixed_z=3.8,
    executor_map=executor.map,  # Dependency Injection
)

results = ga.run(
    optimization_params={"samples_per_tx": 1_000_000, "max_depth": 13},
    ga_params={"pop_size": 50, "n_gen": 20},
    seed=42,
)

print(f"Best: {results['best_position']}  RSS: {results['best_fitness_dbm']:.2f} dBm")
executor.shutdown()
ray.shutdown()
```

---

### 2. Particle Swarm Optimization (PSO) - **Faster Heuristic**

**Priority**: **RECOMMENDED** - Stronger baseline for continuous position optimization.

#### Why Use PSO?
- **Fast Convergence**: Often faster than GA for continuous coordinate problems
- **Conceptual Similarity**: Models "particles" flying through solution space (similar to batching but without physics-aware gradients)
- **Strong Continuous Optimizer**: Better suited for position optimization than GA

#### Implementation Details
- **Particles**: 30-50
- **Iterations**: 50-100
- **Inertia Weight**: 0.7-0.9 (controls exploration vs exploitation)
- **Cognitive Coefficient**: 1.5-2.0 (personal best attraction)
- **Social Coefficient**: 1.5-2.0 (global best attraction)

#### Expected Performance
- **Strengths**: Fast convergence for continuous problems, simple implementation
- **Weaknesses**: Can converge prematurely, no gradient information

#### Key Argument
> "PSO moves blindly based on stochastic velocity vectors. DRT moves purposefully based on the gradient of the radio environment."

#### Python Implementation
Use `PySwarm` or `scikit-opt`:
```python
from pyswarm import pso

def objective(x):
    return coverage_loss(x)

lb = [0, 0, 0, 0, 0, 0]  # Lower bounds
ub = [10, 10, 3, 10, 10, 3]  # Upper bounds
xopt, fopt = pso(objective, lb, ub, swarmsize=50, maxiter=100)
```

---

### 3. Alternating Optimization (AO) - **Mathematical Baseline**

**Priority**: **OPTIONAL** - Represents traditional analytical approach.

#### Why Use AO?
- **Standard Analytical Method**: The traditional way to solve coupled non-convex problems
- **Theoretical Foundation**: Well-established in optimization literature
- **Demonstrates Joint Optimization Value**: Shows the benefit of optimizing all variables together

#### Implementation Details
- **Approach**: Optimize one variable while holding others fixed
- **Cycle**: Optimize AP position → RIS position → Phase Shifts → Repeat
- **Iterations**: 10-20 cycles
- **Per-Variable Optimizer**: Gradient descent or grid search

#### Expected Performance
- **Strengths**: Simple to implement, guaranteed not to increase objective
- **Weaknesses**: Gets stuck in local minima, slow convergence

#### Key Argument
> "AO often gets stuck in local minima because it decouples variables that are physically coupled. Our DRT method optimizes them jointly (or hierarchically) with full gradient awareness."

#### Python Implementation
```python
def alternating_optimization(ap_init, ris_init, max_cycles=20):
    ap_pos = ap_init
    ris_pos = ris_init
    
    for cycle in range(max_cycles):
        # Fix RIS, optimize AP
        ap_pos = optimize_ap_fixed_ris(ap_pos, ris_pos)
        
        # Fix AP, optimize RIS
        ris_pos = optimize_ris_fixed_ap(ap_pos, ris_pos)
        
        if converged(ap_pos, ris_pos):
            break
    
    return ap_pos, ris_pos
```

---

## Baseline Comparison Table

| Baseline | Type | Complexity | Strength | Weakness (Your Advantage) |
|----------|------|-----------|----------|---------------------------|
| **Grid Search** | Naive | Exponential O(n^d) | Guaranteed to find best in grid | Intractable for >2 devices; discrete resolution; poor scaling |
| **Genetic Algorithm** | Heuristic | High (Pop × Gen × Eval) | Robust; handles mixed variables | "Black box" optimization; thousands of simulations; blind to physics |
| **PSO** | Heuristic | Medium (Particles × Iter × Eval) | Fast for continuous; simple | Premature convergence; blind exploration; no gradient info |
| **Alternating Opt.** | Analytical | Low (Cycles × Per-Var) | Theoretical guarantees | Decouples coupled variables; local minima trapping |
| **Proposed DRT** | Physics-Aware | Low (w.r.t resolution) | Exploits gradients; 100x fewer sims; infinite resolution; parallel exploration | Requires differentiable simulator |

## Strategic Recommendations

### For IEEE Transactions on Wireless Communications
**Must Include**: Grid Search + Genetic Algorithm  
**Recommended**: Add PSO for stronger continuous baseline  
**Optional**: Add AO if discussing joint optimization benefits

### For ACM MobiCom / IEEE INFOCOM
**Must Include**: All four baselines (Grid Search, GA, PSO, AO)  
**Additional**: Consider Deep RL baseline if time permits

### Evaluation Metrics
Compare baselines on:
1. **Convergence Speed**: Iterations to reach 95% of optimal
2. **Solution Quality**: Final coverage percentage
3. **Computational Cost**: Total ray tracing calls
4. **Robustness**: Success rate across different random seeds

### Example Results Table
```
| Method | Coverage (%) | Iterations | Ray Traces | Time (s) |
|--------|-------------|-----------|-----------|----------|
| Grid Search | 87.3 | 1 | 10,000 | 450 |
| GA | 91.2 | 100 | 5,000 | 320 |
| PSO | 89.8 | 80 | 2,400 | 180 |
| AO | 85.6 | 15 | 300 | 95 |
| DRT (Ours) | 94.1 | 50 | 100 | 45 |
```

## Implementation Priority

**Phase 1** (Complete): Grid Search baseline ✅  
**Phase 2** (Complete): Genetic Algorithm (DEAP) with Ray-parallel evaluation ✅  
**Phase 3** (Next): PSO for stronger comparison  
**Phase 4** (Future): Alternating Optimization for joint optimization claims

## References

- **GA for RIS**: [Multiple RIS-Assisted UAV Communications](https://ieeexplore.ieee.org) - Uses GA for placement
- **PSO for AP Placement**: [UAV Base Station Placement](https://ieeexplore.ieee.org) - PSO benchmark
- **AO for Joint Optimization**: [RIS Phase Shift and Positioning](https://ieeexplore.ieee.org) - Standard AO approach
