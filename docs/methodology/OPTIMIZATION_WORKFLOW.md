# Optimization Workflow

## Overview

This document describes the **Ray-based distributed parallel optimization workflow** used in the Reflector Position Optimization framework. The approach implements **Distributed Multi-Start Gradient Descent** using Ray for exploring the non-convex optimization landscape when optimizing physical reflector positions.

### Why Ray Instead of Vectorized Batching?

**Critical Distinction**: Ray is necessary when optimizing **physical scene geometry** (reflector positions) rather than just wave parameters or Tx/Rx coordinates.

- **Vectorized Batching**: Suitable for changing parameters within a single scene (e.g., transmitter positions, phase shifts)
- **Ray Architecture**: Required when each optimization trajectory needs **independent scene geometry** (e.g., moving physical reflectors, walls, obstacles)

Since reflectors are physical objects in the scene, each optimization instance requires its own independent Scene copy with different reflector positions. Ray provides this process-level isolation.

## Overall Parallel System Architecture

```bash
+=============================================================================+
|                  1. PARALLEL SYSTEM CONFIGURATION                           |
+=============================================================================+
|                                                                             |
|   +---------------------------+       +---------------------------------+   |
|   |     USER REQUIREMENTS     |       |       BATCH CONFIGURATION       |   |
|   +---------------------------+       +---------------------------------+   |
|   | • Floor Plan: Office_v1   |       | • Batch Size (B): 32            |   |
|   | • Num. APs: 2             |       |   (Creates 32 "Parallel Worlds")|   |
|   | • Num. RIS: 1             |       | • Learning Rates: [0.1, 0.01]   |   |
|   +-------------+-------------+       +----------------+----------------+   |
|                 |                                      |                    |
+=================|======================================|====================+
                  |                                      |
                  v                                      v
+-----------------------------------------------------------------------------+
|                        2. VECTORIZED INITIALIZATION                         |
+-----------------------------------------------------------------------------+
| • Tensor Shapes:                                                            |
|   - AP Positions:  [32, 2, 3]  (32 worlds, 2 APs, xyz coords)               |
|   - RIS Position:  [32, 1, 1]  (32 worlds, 1 RIS, "t" scalar on wall)       |
|   - Focal Points:  [32, 1, 3]  (32 worlds, 1 target point)                  |
|                                                                             |
| • Random Seeding:                                                           |
|   - Each of the 32 worlds starts with APs/RIS in DIFFERENT random spots.    |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
            /-----------------------------------------------------\
            |            3. OPTIMIZATION LOOP                     |
            |     (All 32 worlds optimize simultaneously)         |
            \--------------------------+--------------------------/
                                       |
        +------------------------------<-------------------------------+
        |                                                              |
+-------v-------------------------------------------------------+      |
|  A. PARALLEL PHYSICS PASS (Sionna RT)                         |      |
|  • Ray Casting: Traces rays for 32 scenes at once.            |      |
|    (GPU efficiently handles this parallel load)               |      |
|  • Output: [32, Num_Users] coverage maps.                     |      |
|  • Loss:   [32] separate loss values.                         |      |
+-------+-------------------------------------------------------+      |
        |                                                              |
        v                                                              |
+-------+-------------------------------------------------------+      |
|  B. PARALLEL BACKPROPAGATION                                  |      |
|  • TensorFlow computes gradients for all 32 tensors.          |      |
|  • Result: Gradient vectors pointing in 32 different          |      |
|    directions, exploring different parts of the room.         |      |
+-------+-------------------------------------------------------+      |
        |                                                              |
        v                                                              |
+-------+-------------------------------------------------------+      |
|  C. UPDATE STEP (The Scheduler)                               |      |
|  • Update all 32 configurations.                              |      |
|  • Some "worlds" might get stuck in corners (local minima).   |      |
|  • Other "worlds" will find the perfect LOS paths.            |      |
+-------+-------------------------------------------------------+      |
        |                                                              |
        |   Converged?                                                 |
        |   NO --------------------------------------------------------+
        |
        v YES
+=============================================================================+
|                             4. WINNER SELECTION                             |
+=============================================================================+
| • Input: 32 Final Coverage Maps                                             |
| • Logic: `best_index = argmin(final_losses)`                                |
| • Output:                                                                   |
|   - The single best configuration found across all 32 attempts.             |
|   - "World #7 found the optimal placement."                                 |
+-----------------------------------------------------------------------------+
```

### MapReduce Analogy

The GPU execution model implements a massive, iterative MapReduce operation:

### Map Step (Parallel Physics)

```text
    [BATCH OF 32 RANDOM INITIAL POSITIONS]
                   |
                   v
+--------------------------------------------+
| MAP OPERATION (Parallel Physics)           |
| ------------------------------------------ |
| Universe 0 | Universe 1 | ... | Universe 31|  <-- GPU Threads
| (Ray Trace)| (Ray Trace)| ... | (Ray Trace)|
+--------------------------------------------+
                   |
                   v
        [32 INDEPENDENT LOSS VALUES]
```

## Detail Ray-Based Distributed Architecture

```text
+=============================================================================+
|                      1. ORCHESTRATOR (DRIVER PROCESS)                       |
+=============================================================================+
|  • Role: The "God" class that manages the parallel universes.               |
|  • Resources: CPU (Management), RAM (Object Store)                          |
|  • Configuration:                                                           |
|    - Num_Workers: 32                                                        |
|    - GPU_Fraction: 0.1 per worker (allows 10 workers per physical GPU)      |
+=============================================================================+
                                      |
         (Spawns Independent Actors via Ray Object Store)
                                      |
        +-----------------------------+-----------------------------+
        |                             |                             |
        v                             v                             v
+---------------------+     +---------------------+     +---------------------+
|    RAY ACTOR 1      |     |    RAY ACTOR 2      |     |    RAY ACTOR 32     |
|   (Process ID X)    |     |   (Process ID Y)    |     |   (Process ID Z)    |
+---------------------+     +---------------------+     +---------------------+
| [ISOLATED MEMORY]   |     | [ISOLATED MEMORY]   |     | [ISOLATED MEMORY]   |
|                     |     |                     |     |                     |
| 1. Scene Instance   |     | 1. Scene Instance   |     | 1. Scene Instance   |
|    - Reflector @ P1 |     |    - Reflector @ P2 |     |    - Reflector @ P32|
|    - Walls, Meshes  |     |    - Walls, Meshes  |     |    - Walls, Meshes  |
|                     |     |                     |     |                     |
| 2. Optimizer Class  |     | 2. Optimizer Class  |     | 2. Optimizer Class  |
|    (GradientDescent)|     |    (GradientDescent)|     |    (GradientDescent)|
|                     |     |                     |     |                     |
| 3. Local Loop       |     | 3. Local Loop       |     | 3. Local Loop       |
|    - Forward Pass   |     |    - Forward Pass   |     |    - Forward Pass   |
|    - Backward Pass  |     |    - Backward Pass  |     |    - Backward Pass  |
|    - Update Geom.   |     |    - Update Geom.   |     |    - Update Geom.   |
+----------+----------+     +----------+----------+     +----------+----------+
           |                           |                           |
+=============================================================================+
|                        3. AGGREGATION & SELECTION                           |
+=============================================================================+
| • Ray.get(futures) -> List of [Final_Pos, Min_RSS, History]                 |
| • Winner Selection: best_config = max(results, key=lambda x: x.rss)         |
| • Post-Processing:  Plot heatmaps of the best "Universe"                    |
+=============================================================================+
```

### Key Architectural Differences

| Aspect | Vectorized Batching | Ray Architecture |
|--------|-------------------|------------------|
| **Memory Model** | Shared Scene State | **Independent Scene Instances** |
| **Geometry** | Static (Fixed Meshes) | **Dynamic (Unique Positions per Worker)** |
| **Process Model** | Single Process, GPU Threads | **Multiple Python Processes** |
| **Use Case** | Parameter Optimization | **Physical Object Placement** |
| **Memory Usage** | Low (1 Scene, N Rays) | High (N Scenes, N Rays) |
| **Parallelism** | GPU Vectorization | **Process-Level Isolation** |

## Execution Flow: Three Phases

The execution is split into three distinct phases: **Initialization**, **Async Execution**, and **Reduction**.

### Phase A: Initialization (The "Fork")

1. **Define Search Space**: The Orchestrator defines 32 initial seed positions for the reflector
2. **Spawn Actors**: Ray spins up 32 Python processes (Ray Actors)
3. **Load Scenes**: *Critical Step* - Each Actor independently calls `load_scene()`:
   - *Actor 1*: Loads XML, then executes `scene.get("Reflector").position = [x1, y1, z]`
   - *Actor 2*: Loads XML, then executes `scene.get("Reflector").position = [x2, y2, z]`
   - *Actor 32*: Loads XML, then executes `scene.get("Reflector").position = [x32, y32, z]`
   - **Result**: The geometry is unique per worker - no shared state

### Phase B: Asynchronous Execution (The "Map")

1. **Trigger Optimization**: Orchestrator calls `worker.optimize.remote()` on all 32 actors (non-blocking)
2. **Local Gradient Descent**: Inside each actor, the `GradientDescentAPOptimizer` runs:
   - Computes Ray Tracing (Sionna RT with DrJit)
   - Calculates Loss (Min RSS or Soft Minimum)
   - Computes Gradient (∇Loss w.r.t. reflector position)
   - Updates the local position variable
   - Modifies the Scene geometry for next iteration
3. **VRAM Management**: Ray manages GPU memory via the `num_gpus` parameter
   - Example: 24GB VRAM, each scene takes 1GB → Ray queues tasks if requests exceed capacity
   - Automatic load balancing across available GPUs

### Phase C: Reduction (The "Gather")

1. **Await Completion**: The Orchestrator waits for all 32 futures to resolve: `results = ray.get(futures)`
2. **Optional Pruning**: For multi-stage approaches:
   - Stop bottom 50% of workers halfway through
   - Reallocate resources to top performers
3. **Winner Takes All**: Compare final configurations and return the single best: `best = max(results, key=lambda x: x.rss)`

## Comparison with Particle Swarm Optimization (PSO)

### Particle Swarm Optimization (PSO)
- **Social**: Particles "talk" to each other. Particle A knows that Particle B found a better spot
- **Behavior**: The swarm clusters together quickly
- **Risk**: If the swarm converges too fast to a "pretty good" spot, they all get stuck there (Premature Convergence)

### Ray-Based Distributed Multi-Start Gradient Descent
- **Asocial**: "Universe 1" has no idea "Universe 2" exists. They run in separate Python processes with isolated memory
- **Behavior**: 32 scouts explore 32 completely different valleys of the optimization landscape independently
- **Benefit**: Better for non-convex problems. If 31 scouts get stuck behind a wall (local minimum), the 1 scout who started near the door can still find the global optimum without being pulled back by the failures of the others
- **Gradient Information**: Unlike PSO, each worker has full gradient information from differentiable ray tracing

## Implementation Details

### Ray Actor Wrapper

```python
import ray
from gradient_descent import GradientDescentAPOptimizer

@ray.remote(num_gpus=0.1)  # Each actor uses 10% of a GPU
class ReflectorOptimizerWorker:
    def __init__(self, scene_config, initial_reflector_pos, worker_id):
        # 1. Load a fresh, independent copy of the scene
        self.scene = load_scene(scene_config)
        
        # 2. Set unique reflector position for this worker
        self.scene.get("Reflector").position = initial_reflector_pos
        
        # 3. Initialize optimizer with this worker's scene
        self.optimizer = GradientDescentAPOptimizer(
            scene=self.scene,
            initial_position=initial_reflector_pos
        )
        
        self.worker_id = worker_id

    def optimize(self, num_iterations, learning_rate):
        # Run gradient descent locally
        final_pos, final_rss = self.optimizer.optimize(
            num_iterations=num_iterations,
            learning_rate=learning_rate
        )
        return {
            'worker_id': self.worker_id,
            'final_position': final_pos,
            'final_rss': final_rss,
            'history': self.optimizer.history
        }

# --- Orchestrator ---
def run_distributed_optimization():
    # 1. Define 32 distinct starting positions
    initial_positions = [
        [2.0, 5.0, 1.5],   # World 1
        [8.0, 1.0, 1.5],   # World 2
        # ... 30 more positions
    ]
    
    # 2. Spawn Ray Actors
    workers = [
        ReflectorOptimizerWorker.remote(
            scene_config="office_v1.xml",
            initial_reflector_pos=pos,
            worker_id=i
        )
        for i, pos in enumerate(initial_positions)
    ]
    
    # 3. Start optimization (non-blocking)
    futures = [w.optimize.remote(num_iterations=10, learning_rate=0.5) 
               for w in workers]
    
    # 4. Gather results (blocking)
    results = ray.get(futures)
    
    # 5. Select winner
    best = max(results, key=lambda x: x['final_rss'])
    print(f"Winner: Worker {best['worker_id']} with RSS={best['final_rss']:.2f} dB")
    
    return best
```

### Memory Management Considerations

**Critical**: Each Ray Actor loads a full Scene copy (meshes, textures, BVH trees), which can consume significant VRAM.

**Strategies**:
1. **Limit GPU Fraction**: Use `num_gpus=0.1` to allow 10 workers per GPU
2. **Start Small**: Test with 4-8 workers before scaling to 32
3. **Monitor Memory**: Use `nvidia-smi` to track VRAM usage
4. **Staged Execution**: Run 8 workers at a time if memory constrained

**Example VRAM Calculation**:
- Scene size: 1 GB (meshes + BVH)
- Ray tracing buffer: 500 MB per worker
- 32 workers × 1.5 GB = 48 GB total
- With 4× RTX 4090 (24 GB each) = 96 GB available → Feasible

### Key Features

1. **Process-Level Isolation**: Each Ray Actor runs in a separate Python process with independent memory
2. **Scene Independence**: Each actor has its own Scene object with unique reflector positions
3. **Exploration Without Communication**: Each world explores independently, avoiding premature convergence
4. **Physics-Aware Gradients**: Uses differentiable ray tracing to compute exact gradients through the radio propagation model
5. **Efficient Winner Selection**: Simple max/min reduction to select the best solution from all parallel attempts
6. **GPU Resource Management**: Ray automatically balances GPU memory across workers

## Benefits Over Traditional Methods

| Aspect | Traditional PSO | Vectorized Batching | Ray Architecture |
|--------|----------------|---------------------|------------------|
| **Communication** | Particles share global best | No communication | No communication |
| **Convergence** | Fast but risky (premature) | Slower but robust | Slower but robust |
| **Gradient Usage** | No gradients (black box) | Full gradient information | Full gradient information |
| **Scene Geometry** | Can't modify | Static (shared) | **Dynamic (independent)** |
| **Memory Model** | Sequential updates | Shared GPU memory | **Isolated process memory** |
| **Use Case** | General optimization | Parameter tuning | **Physical object placement** |
| **Local Minima** | Susceptible to trapping | 32 independent escape attempts | 32 independent escape attempts |

## When to Use Ray vs Vectorization

### Use Vectorized Batching When:
- Optimizing **parameters** (e.g., transmitter power, phase shifts)
- Optimizing **Tx/Rx positions** (point coordinates)
- Scene geometry remains **static**
- Memory is limited (single scene instance)

### Use Ray Architecture When:
- Optimizing **physical object positions** (reflectors, walls, obstacles)
- Each optimization trajectory needs **different scene geometry**
- Sufficient memory/VRAM available (multiple scene instances)
- Need **true process isolation** for stability

## Future Enhancements

### Staged Optimization (Pruning)
1. Run all 32 workers for first 5 iterations
2. Rank workers by current RSS
3. Stop bottom 50% (16 workers)
4. Continue top 50% for remaining iterations
5. Benefit: Saves computation while maintaining diversity

### Adaptive Learning Rates
```python
def optimize_adaptive(self, num_iterations):
    for iteration in range(num_iterations):
        # High learning rate initially
        lr = 1.0 * (0.9 ** iteration)
        # ... optimization step
```

### Multi-Objective with Ray
```python
def optimize_multi_objective(self):
    results = {
        'min_rss': self.optimizer.compute_min_rss(),
        'coverage': self.optimizer.compute_coverage(),
        'interference': self.optimizer.compute_interference()
    }
    # Pareto frontier selection in orchestrator
    return results
```

### Integration with PSO
To add PSO-style social behavior while maintaining Ray architecture:
1. After each iteration, gather all positions: `positions = ray.get([w.get_position.remote() for w in workers])`
2. Identify global best: `g_best = max(positions, key=...)`
3. Add "pull" force toward `g_best` in each worker's update step
4. Tune social parameters (inertia, cognitive, social coefficients)

**Note**: This hybrid approach may reduce exploration benefits but could accelerate convergence if the landscape is not too non-convex.
