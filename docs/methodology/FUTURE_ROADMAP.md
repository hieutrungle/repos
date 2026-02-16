# Future Roadmap

## Advanced Features for Phase 2+

This document outlines planned enhancements based on the optimization framework analysis.

## Parallel Batch Optimization ✅ COMPLETE

### Implementation
- Ray ActorPool with persistent `OptimizationWorker` actors
- Multi-start gradient descent (64 tasks → 4 workers)
- True parallel grid search (441 single-point tasks)
- Ordered `pool.map` (synchronous, freeze-safe)
- Configurable GPU fraction per worker (0.25 = 4 workers/GPU)

### Architecture

```python
# Actual Implementation (ray_parallel_optimizer.py)
import ray
from ray.util import ActorPool

@ray.remote(num_gpus=gpu_fraction)
class OptimizationWorker:
    def __init__(self, scene_config):
        self.scene = setup_building_floor_scene(**scene_config)

    def evaluate(self, task_config):
        optimizer = create_optimizer(self.scene, task_config)
        return optimizer.optimize()

# ActorPool with persistent workers
pool = ActorPool([OptimizationWorker.remote(cfg) for _ in range(num_workers)])
results = list(pool.map(lambda w, t: w.evaluate.remote(t), tasks))
```

### Results
- **Near-linear speedup** with number of workers on GPU
- **Scene loaded once** per worker (not per task)
- **Three methods supported**: GD, GS, GA

---

## Baseline Method Implementations

### 1. Genetic Algorithm (GA) ✅ COMPLETE

**Status**: Implemented using DEAP library with Ray-parallel fitness evaluation.

#### Implementation Summary
- **Library**: DEAP 1.4.1+ (not PyGAD)
- **Architecture**: IoC pattern — `GeneticAlgorithmRunner` (pure DEAP, no Ray) + `RayActorPoolExecutor` (generic Ray engine)
- **Population**: 50–100 individuals encoding (x, y) AP positions
- **Operators**: Blend crossover (`cxBlend`), Gaussian mutation, tournament selection
- **Fitness**: Maximises minimum RSS (linear Watts) via `SinglePointGridSearchOptimizer`
- **Evaluation**: Ray ActorPool with `pool.map` (ordered, synchronous)

**Full Details**: See [GA_DEAP_IMPLEMENTATION.md](GA_DEAP_IMPLEMENTATION.md)

**Entry Point**: `examples/run_ga_modular.py`

```python
from reflector_position.optimizers import RayActorPoolExecutor, GeneticAlgorithmRunner

executor = RayActorPoolExecutor(scene_config={...}, num_workers=4, gpu_fraction=0.25)
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
```

---

### 2. Particle Swarm Optimization (PSO)

**Timeline**: Phase 3 (1-2 weeks)

#### Implementation Plan
```python
from pyswarm import pso

class PSOOptimizer:
    def __init__(self, scene_config, swarm_size=50, max_iter=100):
        self.scene_config = scene_config
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        
    def objective(self, x):
        ap_pos = x.reshape(self.scene_config.num_aps, 3)
        return -self.evaluate_coverage(ap_pos)  # Minimize negative coverage
    
    def optimize(self):
        lb, ub = self.get_bounds()
        xopt, fopt = pso(
            self.objective,
            lb, ub,
            swarmsize=self.swarm_size,
            maxiter=self.max_iter,
            omega=0.8,  # Inertia
            phip=2.0,   # Cognitive parameter
            phig=2.0    # Social parameter
        )
        return xopt.reshape(self.scene_config.num_aps, 3)
```

**Key Metrics to Report**:
- Total ray tracing calls: `swarm_size × iterations`
- Premature convergence detection (particle variance)
- Comparison with batch gradient descent (both use populations)

---

### 3. Alternating Optimization (AO)

**Timeline**: Phase 4 (1 week)

#### Implementation Plan
```python
class AlternatingOptimizer:
    def __init__(self, scene_config, max_cycles=20):
        self.scene_config = scene_config
        self.max_cycles = max_cycles
        
    def optimize(self, ap_init, ris_init):
        ap_pos = ap_init
        ris_pos = ris_init
        
        for cycle in range(self.max_cycles):
            # Freeze RIS, optimize AP
            ap_optimizer = GradientDescentAPOptimizer(fixed_ris=ris_pos)
            ap_pos = ap_optimizer.optimize(ap_pos)
            
            # Freeze AP, optimize RIS
            ris_optimizer = GradientDescentRISOptimizer(fixed_ap=ap_pos)
            ris_pos = ris_optimizer.optimize(ris_pos)
            
            # Check convergence
            if self.has_converged(ap_pos, ris_pos):
                break
                
        return ap_pos, ris_pos
```

**Key Metrics to Report**:
- Total cycles to convergence
- Comparison with joint optimization (show local minima trapping)

---

## Multi-Device Joint Optimization

### Current State
- Single AP or RIS optimization
- No joint positioning

### Planned Enhancement
Optimize AP positions + RIS positions + RIS phase shifts simultaneously:

```python
class JointOptimizer:
    def __init__(self, scene_config):
        # Trainable variables
        self.ap_positions = tf.Variable([...])  # [num_aps, 3]
        self.ris_positions = tf.Variable([...])  # [num_ris, 3]
        self.ris_phases = tf.Variable([...])  # [num_ris, num_elements]
        
    def compute_joint_loss(self):
        # Sionna RT with all trainable parameters
        paths = self.scene.compute_paths(
            tx_pos=self.ap_positions,
            ris_pos=self.ris_positions,
            ris_phases=self.ris_phases
        )
        return coverage_loss(paths)
    
    def optimize(self):
        for step in range(max_steps):
            with tf.GradientTape() as tape:
                loss = self.compute_joint_loss()
            
            grads = tape.gradient(loss, [
                self.ap_positions,
                self.ris_positions,
                self.ris_phases
            ])
            
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
```

### Challenges
- **High dimensionality**: 2 APs × 3 coords + 1 RIS × 3 coords + N phase elements
- **Coupled variables**: AP position affects optimal RIS phases and vice versa
- **Solution**: Use hierarchical optimization (coarse positioning first, then fine-tuning)

---

## Advanced Scene Support

### Planned Enhancements

#### 1. Multi-Floor Buildings
```python
class MultiFloorScene:
    def __init__(self, num_floors=3):
        self.floors = [load_floor_plan(f"floor_{i}.xml") for i in range(num_floors)]
        
    def optimize_per_floor(self):
        results = {}
        for i, floor in enumerate(self.floors):
            optimizer = APOptimizer(scene=floor)
            results[f"floor_{i}"] = optimizer.optimize()
        return results
```

#### 2. Dynamic Environments
- Time-varying obstacles (moving furniture, people)
- Adaptive re-optimization when coverage drops
- Trigger-based re-positioning

#### 3. Realistic Constraints
```python
# Example: Mounting constraints
ap_constraints = {
    "height_range": (2.5, 3.0),  # Ceiling mount only
    "wall_mount_only": True,
    "avoid_windows": True
}
```

---

## Performance Optimizations

### Planned Improvements

1. **Mixed Precision Training**
   ```python
   policy = tf.keras.mixed_precision.Policy('mixed_float16')
   tf.keras.mixed_precision.set_global_policy(policy)
   ```

2. **Gradient Checkpointing**
   - Reduce memory for large batch sizes
   - Trade computation for memory

3. **Ray Tracing Approximations**
   - Adaptive sampling (more rays near convergence)
   - Importance sampling for critical paths

4. **Distributed Training**
   - Multi-GPU support for large batch sizes
   - MirroredStrategy for data parallelism

---

## Research Extensions

### 1. Deep Reinforcement Learning Baseline
**Why**: Compare against model-free RL methods

```python
# PPO or SAC agent
class RLOptimizer:
    def __init__(self):
        self.agent = PPO(
            state_dim=grid_size,
            action_dim=3,  # x, y, z movement
            policy_network=ActorCritic()
        )
    
    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            while not done:
                action = self.agent.select_action(state)
                next_state, reward = self.env.step(action)
                self.agent.update(state, action, reward, next_state)
```

### 2. Meta-Learning for Fast Adaptation
- Train on multiple floor plans
- Few-shot learning for new environments

### 3. Multi-Objective Optimization
- Pareto front: Coverage vs Power Consumption vs Cost
- NSGA-II or MOEA/D algorithms

---

## Documentation Enhancements

### Planned Additions

1. **API Reference**
   - Auto-generated from docstrings (Sphinx)
   - Interactive examples (Jupyter notebooks)

2. **Performance Benchmarks**
   - Comparison tables for all baselines
   - Scalability plots (devices vs time)

3. **Tutorial Videos**
   - YouTube series on using the framework
   - Walkthrough of research reproduction

4. **Publication Guide**
   - How to cite this framework
   - Example paper sections using this tool

---

## Implementation Timeline

| Phase | Features | Duration | Status |
|-------|----------|----------|--------|
| **Phase 1** | Basic gradient descent, grid search | Completed | ✅ |
| **Phase 2** | Ray parallel + DEAP genetic algorithm | Completed | ✅ |
| **Phase 3** | PSO baseline + testing | 2-3 weeks | 📋 Planned |
| **Phase 4** | Joint optimization (AP + RIS) | 3-4 weeks | 📋 Planned |
| **Phase 5** | Multi-floor + constraints | 2 weeks | 📋 Planned |
| **Phase 6** | Performance optimizations | 2 weeks | 📋 Planned |
| **Research** | RL baselines, meta-learning, hybrid GA+GD | Ongoing | 💡 Future |

---

## Community Contributions

### How to Contribute
See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

### Requested Features
- [ ] Support for other ray tracing engines (e.g., Wireless InSite)
- [ ] Integration with OpenAI Gym for RL research
- [ ] Real-world deployment tools (hardware interfacing)
- [ ] Visualization dashboard (Plotly/Dash)

---

## References

- **Batch Optimization**: [Parallel Evolutionary Algorithms](https://link.springer.com)
- **Hybrid Methods**: [GA + Gradient Descent](https://ieeexplore.ieee.org)
- **Multi-Objective**: [NSGA-II for Wireless Networks](https://ieeexplore.ieee.org)
