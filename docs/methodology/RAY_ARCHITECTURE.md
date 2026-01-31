# Ray Architecture for Reflector Optimization

**Date**: January 31, 2026  
**Status**: Design Document

This document explains why Ray (distributed computing framework) is the correct architectural choice for optimizing physical reflector positions, as opposed to vectorized batching within a single scene.

## Problem Statement

For each gradient descent iteration, the reflector (a real physical object) changes its position, effectively leading to different scene geometries between each optimization instance. This requires true process-level isolation rather than just parameter vectorization.

## Why Ray Instead of Vectorization?

### 1. The "State Mutation" Problem

In ray tracing engines like Sionna, the `Scene` object holds the "truth" of the physical world (the geometry of walls, meshes, and reflectors).

* **Vectorized Approach (Single Scene):** If you use a single `Scene` object and try to "vectorize" the optimization, you hit a hard limit. You cannot easily place a physical reflector mesh in Position A for "Batch 1" and Position B for "Batch 2" simultaneously within the same scene instance. The geometry is usually static or shared.
* **Ray Approach (Multiple Scenes):** By using Ray, you spawn 32 independent "Actors." Each Actor loads its own private copy of the `Scene`.
  - *Worker 1* moves its reflector to the North wall.
  - *Worker 2* moves its reflector to the South wall.
  - Because they are separate objects in memory, they do not conflict. This perfectly matches your need for "different scenes between each instance."

### 2. The "Parallel Worlds" Alignment

The [OPTIMIZATION_WORKFLOW.md](OPTIMIZATION_WORKFLOW.md) describes a "Parallel System Architecture" where distinct worlds evolve independently.

* Ray maps 1:1 to this concept. Each Ray Actor is literally a separate "Parallel World" with its own state, history, and physics.
* This creates the **"Asocial"** behavior described in the workflow, where "Universe 1 has no idea Universe 2 exists". This isolation is automatic with Ray, preventing the "Mode Collapse" (averaging) issue.

### 3. Implementation Recommendation

Since you are proceeding with Ray, here is how you should structure the wrapper to handle the "Reflector" specifically.

**Critical Note on VRAM:**
Because each Ray worker loads a full copy of the scene (meshes, textures, BVH trees), running 32 workers might explode your GPU memory.

* **Strategy:** Start with a smaller batch (e.g., 4 or 8) to test memory usage before scaling to 32.

#### The Ray Wrapper for Moving Reflectors

```python
import ray
import numpy as np
from gradient_descent import GradientDescentAPOptimizer
# Assume you have a function that loads your base scene
# from scene_loader import load_base_scene

@ray.remote(num_gpus=0.25) # Limits to 4 workers per GPU (adjust based on VRAM)
class ReflectorOptimizerWorker:
    def __init__(self, scene_config, initial_reflector_pos, worker_id):
        # 1. Load a fresh, independent copy of the scene
        # self.scene = load_base_scene(scene_config)
        self.worker_id = worker_id
        
        # 2. APPLY THE UNIQUE GEOMETRY FOR THIS "WORLD"
        # Move the reflector mesh to its starting position for this specific worker
        # self.scene.get_object("Reflector").position = initial_reflector_pos
        
        # 3. Initialize the low-level optimizer with this specific scene
        self.optimizer = GradientDescentAPOptimizer(
            scene=self.scene,
            initial_position=initial_reflector_pos
        )

    def step(self):
        # Run one step of gradient descent
        # The optimizer will modify self.scene internally
        pos, rss = self.optimizer.optimize(num_iterations=1, verbose=False)
        return pos, rss

# --- Orchestrator ---
# 1. Define distinct starting positions for the reflector
initial_positions = [
    [2.0, 5.0, 1.5], # World 1 starts here
    [8.0, 1.0, 1.5], # World 2 starts here
    # ... etc
]

# 2. Spawn Actors
workers = [
    ReflectorOptimizerWorker.remote(
        scene_config="office_v1", 
        initial_reflector_pos=pos,
        worker_id=i
    )
    for i, pos in enumerate(initial_positions)
]

# 3. Parallel Execution
futures = [w.step.remote() for w in workers]
results = ray.get(futures)

```

### Summary

You are making the right choice.

* **Vectorization** is best for changing *wave parameters* (phases, amplitudes) or *Tx/Rx positions* (points).
* **Ray/Multiprocessing** is best for changing *scene geometry* (meshes, walls, obstacles), which is exactly what you are doing.

Would you like me to generate a template for the `scene_loader` function to ensure it loads efficiently inside the Ray workers?




## Question

Give me a better workflow to reflect the use of Ray and independent instances.

## Answer

This updated workflow reflects the architectural shift from **Vectorized Batching** (single engine, multiple points) to **Distributed Parallelism** (multiple engines, multiple scenes) using Ray. This structure specifically addresses your need to modify physical scene geometry (reflector positions) independently for each optimization trajectory.

### **Distributed Optimization Workflow (Ray Architecture)**

#### **1. Conceptual Shift**

* **Previous Model (Vectorized):** One "Universe" with 32 ghosts. You could only change parameters that didn't affect the static scene geometry (like Tx/Rx coordinates).
* **New Model (Distributed):** 32 distinct "Parallel Universes." Each universe is a fully isolated process with its own copy of the physics engine. You can physically move walls, reflectors, or obstacles in Universe A without affecting Universe B.

#### **2. System Architecture Diagram**

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
           +-------------<-------------+-------------<-------------+
                                      |
                               (Returns Futures)
                                      |
+=============================================================================+
|                        3. AGGREGATION & SELECTION                           |
+=============================================================================+
| • Ray.get(futures) -> List of [Final_Pos, Min_RSS, History]                 |
| • Winner Selection: best_config = max(results, key=lambda x: x.rss)         |
| • Post-Processing:  Plot heatmaps of the best "Universe"                    |
+-----------------------------------------------------------------------------+

```

---

#### **3. Detailed Process Flow**

The execution is split into three distinct phases: **Initialization**, **Async Execution**, and **Reduction**.

**Phase A: Initialization (The "Fork")**

1. **Define Search Space:** The Orchestrator defines 32 initial seed positions for the reflector.
2. **Spawn Actors:** Ray spins up 32 Python processes.
3. **Load Scenes:** *Crucial Step.* Each Actor independently calls `load_scene()`.
* *Actor 1* loads the XML, then executes `scene.get("Reflector").position = [x1, y1, z]`.
* *Actor 2* loads the XML, then executes `scene.get("Reflector").position = [x2, y2, z]`.
* *Note:* This confirms that the geometry is unique per worker.



**Phase B: Asynchronous Execution (The "Map")**

1. **Trigger Optimization:** The Orchestrator calls `worker.optimize.remote()` on all 32 actors. This is non-blocking.
2. **Local Gradient Descent:** Inside each actor, your existing `GradientDescentAPOptimizer` runs:
* Computes Ray Tracing (DrJit/Sionna).
* Calculates Loss (Min RSS).
* Computes Gradient ().
* Updates the local position variable.


3. **VRAM Management:** Because 32 distinct Scenes are loaded, VRAM is the bottleneck. Ray manages this via the `num_gpus` parameter (e.g., if you have 24GB VRAM and each scene takes 1GB, Ray queues pending tasks if `num_gpus` requests exceed capacity).

**Phase C: Reduction (The "Gather")**

1. **Await Completion:** The Orchestrator waits for all 32 futures to resolve.
2. **Pruning:** (Optional) If implementing a "Multi-Stage" approach, the Orchestrator could stop the bottom 50% of workers halfway through and reallocate resources to the top performers.
3. **Winner Takes All:** The final configurations are compared, and the single best physical setup is returned.

---

#### **4. Comparison: Why this works for Reflector Optimization**

| Feature | Original "Batch" Workflow | New "Ray" Workflow |
| --- | --- | --- |
| **State** | Shared Scene State | **Independent Scene State** |
| **Geometry** | Static (Fixed Walls/Meshes) | **Dynamic (Unique Mesh Pos per Worker)** |
| **Memory** | Low (1 Scene, N Rays) | High (N Scenes, N Rays) |
| **Gradients** | Vectorized Tensor Gradients | Independent Scalar Gradients |
| **Suitability** | Optimizing Tx/Rx coordinates | **Optimizing Physical Object Placement** |

#### **5. Implementation Hook**

You can wrap your existing `GradientDescentAPOptimizer` without rewriting its internal math. The Ray Actor simply becomes the "container" for the class:

```python
# Pseudo-code for the Ray Actor wrapper
@ray.remote(num_gpus=0.1)
class OptimizationActor:
    def __init__(self, scene_file, reflector_start_pos):
        # 1. Load Scene
        self.scene = load_scene(scene_file)
        
        # 2. Mutate Scene (The specific advantage of Ray)
        self.scene.objects["Reflector"].position = reflector_start_pos
        
        # 3. Instantiate your existing class
        self.optimizer = GradientDescentAPOptimizer(
            scene=self.scene,
            initial_position=reflector_start_pos
        )
        
    def run(self):
        # 4. Run the existing logic
        return self.optimizer.optimize() 

```