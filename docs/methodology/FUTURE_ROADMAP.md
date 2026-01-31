# Future Roadmap

## Advanced Features for Phase 2+

This document outlines planned enhancements based on the optimization framework analysis.

## Parallel Batch Optimization (Priority: HIGH)

### Current Implementation
- Single AP position optimization per run
- Sequential optimization (one configuration at a time)

### Planned Enhancement
Implement **Batch-Parallelized Multi-Start Gradient Descent** with parallel worlds:

```python
# Target Implementation
batch_size = 32  # 32 parallel optimization worlds

# Tensor shapes for vectorized operations
ap_positions = tf.Variable(
    shape=[32, 2, 3],  # [batch, num_APs, xyz]
    initial_value=random_init()
)

ris_positions = tf.Variable(
    shape=[32, 1, 1],  # [batch, num_RIS, wall_param]
    initial_value=random_init()
)

# Each world explores independently
for iteration in range(max_iter):
    # Parallel physics computation for all 32 worlds
    losses = parallel_ray_trace(ap_positions, ris_positions)  # [32,]
    
    # Parallel gradient computation
    gradients = tape.gradient(losses, [ap_positions, ris_positions])
    
    # Parallel updates (no communication between worlds)
    optimizer.apply_gradients(zip(gradients, variables))

# Winner selection
best_idx = tf.argmin(losses)
best_config = ap_positions[best_idx]
```

### Benefits
- **Robust Exploration**: 32 independent searches avoid local minima
- **GPU Efficiency**: Sionna RT handles batch processing natively
- **No Communication Overhead**: Worlds don't share information (unlike PSO)
- **Guaranteed Diversity**: Different random seeds ensure varied exploration

### Implementation Checklist
- [ ] Vectorize scene initialization for batched inputs
- [ ] Modify RadioMapSolver to accept batched positions
- [ ] Implement parallel loss computation
- [ ] Add winner selection logic
- [ ] Benchmark against single-instance optimization

---

## Baseline Method Implementations

### 1. Genetic Algorithm (GA) - **Next Priority**

**Timeline**: Phase 2 (2-3 weeks)

#### Implementation Plan
```python
# Using PyGAD library
import pygad

class GeneticAlgorithmOptimizer:
    def __init__(self, scene_config, population_size=50, num_generations=100):
        self.scene_config = scene_config
        self.pop_size = population_size
        self.generations = num_generations
        
    def fitness_function(self, ga_instance, solution, solution_idx):
        # Decode solution to AP positions
        ap_pos = solution.reshape(self.scene_config.num_aps, 3)
        
        # Compute coverage using Sionna
        coverage = self.evaluate_coverage(ap_pos)
        
        # GA maximizes, so negate loss
        return -coverage
    
    def optimize(self):
        ga_instance = pygad.GA(
            num_generations=self.generations,
            num_parents_mating=self.pop_size // 2,
            fitness_func=self.fitness_function,
            sol_per_pop=self.pop_size,
            num_genes=self.scene_config.num_aps * 3,
            gene_space=self.get_position_bounds(),
            parent_selection_type="tournament",
            crossover_type="uniform",
            mutation_type="adaptive",
            mutation_percent_genes=10
        )
        
        ga_instance.run()
        return ga_instance.best_solution()
```

**Key Metrics to Report**:
- Total ray tracing calls: `population_size × generations`
- Convergence rate (generations to 95% optimal)
- Final coverage vs DRT

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
| **Phase 2** | Genetic Algorithm baseline | 2-3 weeks | 📋 Planned |
| **Phase 3** | PSO baseline + parallel batching | 2-3 weeks | 📋 Planned |
| **Phase 4** | Joint optimization (AP + RIS) | 3-4 weeks | 📋 Planned |
| **Phase 5** | Multi-floor + constraints | 2 weeks | 📋 Planned |
| **Phase 6** | Performance optimizations | 2 weeks | 📋 Planned |
| **Research** | RL baselines, meta-learning | Ongoing | 💡 Future |

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
