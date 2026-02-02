"""
Ray-based parallel optimizer wrapper for distributed multi-start optimization.

This module implements distributed parallel optimization using Ray, enabling
multiple independent optimization trajectories to run simultaneously across
different workers/GPUs. This is essential when optimizing physical scene
geometry (reflector positions) where each instance needs independent Scene copies.

Key Design Principles:
1. Process-level isolation: Each Ray actor has its own Scene instance
2. True parallelism: Multiple "parallel universes" exploring different starting points
3. GPU efficiency: Configurable GPU fraction per worker (e.g., 0.1 allows 10 workers per GPU)
4. Optimizer agnostic: Works with any optimizer inheriting from BaseAPOptimizer

Architecture:
- Orchestrator (Driver): Manages Ray actors, aggregates results
- Ray Actors (Workers): Each runs independent optimization with its own Scene
- Reduction Phase: Winner selection from all parallel trajectories
"""

import time
from typing import Dict, List, Tuple, Optional, Any, Type
import numpy as np
import ray
import torch

from .base_optimizer import BaseAPOptimizer
from .optimizer_factory import OptimizerFactory


@ray.remote
class OptimizationWorker:
    """
    Ray actor that runs an independent optimization trajectory.
    
    Each worker:
    1. Loads its own Scene instance (isolated from other workers)
    2. Initializes its optimizer with unique starting position
    3. Runs optimization independently
    4. Returns final results to orchestrator
    
    This provides true process-level isolation needed for modifying
    physical scene geometry (reflectors, walls, obstacles).
    """
    
    def __init__(
        self,
        worker_id: int,
        scene_config: Dict[str, Any],
        optimizer_method: str,
        optimizer_config: Dict[str, Any],
        optimization_params: Dict[str, Any],
        gpu_fraction: float = 0.1,
    ):
        """
        Initialize optimization worker.
        
        Args:
            worker_id: Unique identifier for this worker
            scene_config: Configuration for loading Scene (XML path, materials, etc.)
            optimizer_method: Optimization method name ('gradient_descent', 'grid_search')
            optimizer_config: Method-specific initialization parameters
            optimization_params: Parameters for optimize() call
            gpu_fraction: Fraction of GPU to allocate (0.1 = 10 workers per GPU)
        """
        self.worker_id = worker_id
        self.optimizer_method = optimizer_method
        self.optimizer_config = optimizer_config
        self.optimization_params = optimization_params
        
        # Set GPU device based on fraction
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{worker_id % torch.cuda.device_count()}")
        else:
            self.device = torch.device("cpu")
        
        # Load Scene instance (CRITICAL: each worker gets its own Scene)
        self.scene = self._load_scene(scene_config)
        
        # Initialize optimizer with this worker's configuration
        self.optimizer = OptimizerFactory.create(
            method=optimizer_method,
            scene=self.scene,
            **optimizer_config
        )
        
    def _load_scene(self, scene_config: Dict[str, Any]):
        """
        Load Scene instance for this worker.
        
        Each worker loads its own Scene with potentially different
        initial geometry (reflector positions, obstacle placements).
        
        Args:
            scene_config: Configuration dict with:
                - xml_path: Path to scene XML file
                - reflector_position: [x, y, z] for this worker
                - other scene modifications
                
        Returns:
            Scene instance specific to this worker
        """
        import sionna.rt
        
        # Load base scene
        scene = sionna.rt.load_scene(scene_config["xml_path"])
        
        # Apply worker-specific modifications (e.g., reflector position)
        if "reflector_position" in scene_config:
            reflector_pos = scene_config["reflector_position"]
            # Modify scene geometry (this is why we need Ray!)
            if "reflector_name" in scene_config:
                reflector = scene.get(scene_config["reflector_name"])
                reflector.position = reflector_pos
        
        return scene
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization and return results.
        
        Returns:
            Dictionary with:
                - worker_id: Worker identifier
                - best_position: Optimized position [x, y, z]
                - best_metric: Best metric value achieved
                - history: Optimization history (if available)
                - time_elapsed: Total optimization time
        """
        start_time = time.time()
        
        # Run optimization
        best_position, best_metric = self.optimizer.optimize(
            verbose=False,  # Suppress individual worker logs
            **self.optimization_params
        )
        
        elapsed_time = time.time() - start_time
        
        # Collect results
        result = {
            "worker_id": self.worker_id,
            "best_position": best_position,
            "best_metric": best_metric,
            "time_elapsed": elapsed_time,
        }
        
        # Include history if optimizer tracks it
        if hasattr(self.optimizer, "history"):
            result["history"] = self.optimizer.history
        
        return result


class RayParallelOptimizer:
    """
    Orchestrator for distributed parallel optimization using Ray.
    
    This class implements the "Multiple Parallel Universes" architecture:
    1. Spawns multiple Ray actors (workers)
    2. Each worker runs independent optimization from different starting point
    3. Aggregates results and selects best configuration
    
    This is the CORRECT approach for optimizing physical scene geometry
    (reflectors, walls, obstacles) as opposed to vectorized batching which
    only works for changing wave parameters or Tx/Rx coordinates.
    
    Usage:
        >>> # Initialize Ray
        >>> ray.init()
        >>> 
        >>> # Create parallel optimizer
        >>> parallel_opt = RayParallelOptimizer(
        ...     num_workers=32,
        ...     gpu_fraction=0.1,
        ...     optimizer_method="gradient_descent"
        ... )
        >>> 
        >>> # Define search space (different starting positions)
        >>> initial_positions = generate_random_positions(32, bounds)
        >>> 
        >>> # Run parallel optimization
        >>> results = parallel_opt.optimize(
        ...     scene_config={"xml_path": "scene.xml"},
        ...     initial_positions=initial_positions,
        ...     optimization_params={"num_iterations": 50, "learning_rate": 0.5}
        ... )
        >>> 
        >>> # Get best configuration
        >>> best_result = results["best_result"]
        >>> print(f"Best position: {best_result['best_position']}")
    """
    
    def __init__(
        self,
        num_workers: int = 32,
        gpu_fraction: float = 0.1,
        optimizer_method: str = "gradient_descent",
    ):
        """
        Initialize Ray parallel optimizer.
        
        Args:
            num_workers: Number of parallel workers (optimization trajectories)
            gpu_fraction: GPU fraction per worker (0.1 = 10 workers per GPU)
            optimizer_method: Base optimization method to parallelize
        """
        self.num_workers = num_workers
        self.gpu_fraction = gpu_fraction
        self.optimizer_method = optimizer_method
        
        # Initialize Ray if not already running
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    
    def optimize(
        self,
        scene_config: Dict[str, Any],
        initial_positions: List[np.ndarray],
        optimization_params: Dict[str, Any],
        optimizer_configs: Optional[List[Dict[str, Any]]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run parallel optimization across multiple workers.
        
        This implements the full "Fork -> Map -> Reduce" workflow:
        1. Fork: Spawn workers with different initial positions
        2. Map: Each worker runs optimization independently
        3. Reduce: Aggregate results and select winner
        
        Args:
            scene_config: Base scene configuration (applied to all workers)
                - xml_path: Scene XML file
                - Common scene settings
            initial_positions: List of starting positions, one per worker
                Length must equal num_workers
            optimization_params: Parameters passed to optimize() method
                - num_iterations, learning_rate, samples_per_tx, etc.
            optimizer_configs: Optional list of optimizer-specific configs per worker
                If None, uses initial_positions to generate configs
            verbose: Print progress information
            
        Returns:
            Dictionary with:
                - all_results: List of results from all workers
                - best_result: Result from best worker
                - best_worker_id: ID of best worker
                - total_time: Total wall-clock time
                - aggregate_stats: Statistics across all workers
        """
        if len(initial_positions) != self.num_workers:
            raise ValueError(
                f"Number of initial positions ({len(initial_positions)}) "
                f"must equal num_workers ({self.num_workers})"
            )
        
        start_time = time.time()
        
        if verbose:
            print("=" * 80)
            print("RAY PARALLEL OPTIMIZATION")
            print("=" * 80)
            print(f"Workers: {self.num_workers}")
            print(f"GPU fraction per worker: {self.gpu_fraction}")
            print(f"Base optimizer method: {self.optimizer_method}")
            print(f"Expected workers per GPU: {int(1.0 / self.gpu_fraction)}")
            print("-" * 80)
        
        # Phase 1: FORK - Spawn Ray actors
        if verbose:
            print("Phase 1/3: Spawning Ray actors...")
        
        workers = []
        for i, init_pos in enumerate(initial_positions):
            # Create worker-specific scene config
            worker_scene_config = scene_config.copy()
            if "reflector_position" in worker_scene_config:
                worker_scene_config["reflector_position"] = init_pos
            
            # Create worker-specific optimizer config
            if optimizer_configs is not None:
                opt_config = optimizer_configs[i]
            else:
                # Default: use initial position for optimizer initialization
                opt_config = {"initial_position": init_pos[:2]}  # [x, y]
            
            # Spawn worker with GPU allocation
            worker = OptimizationWorker.remote(
                worker_id=i,
                scene_config=worker_scene_config,
                optimizer_method=self.optimizer_method,
                optimizer_config=opt_config,
                optimization_params=optimization_params,
                gpu_fraction=self.gpu_fraction,
            )
            workers.append(worker)
        
        if verbose:
            print(f"✓ Spawned {len(workers)} workers")
            print("-" * 80)
        
        # Phase 2: MAP - Parallel execution
        if verbose:
            print("Phase 2/3: Running parallel optimization...")
            print("(Each worker optimizing independently - 'Asocial' behavior)")
        
        # Launch all optimizations in parallel (non-blocking)
        futures = [worker.optimize.remote() for worker in workers]
        
        # Wait for all to complete (blocking)
        all_results = ray.get(futures)
        
        if verbose:
            print(f"✓ All {len(all_results)} workers completed")
            print("-" * 80)
        
        # Phase 3: REDUCE - Winner selection
        if verbose:
            print("Phase 3/3: Aggregating results and selecting winner...")
        
        # Find best worker (maximize metric, assuming higher is better)
        # NOTE: Adjust this logic based on your metric (min vs max)
        best_idx = np.argmax([r["best_metric"] for r in all_results])
        best_result = all_results[best_idx]
        
        total_time = time.time() - start_time
        
        # Compute aggregate statistics
        metrics = [r["best_metric"] for r in all_results]
        times = [r["time_elapsed"] for r in all_results]
        
        aggregate_stats = {
            "mean_metric": np.mean(metrics),
            "std_metric": np.std(metrics),
            "min_metric": np.min(metrics),
            "max_metric": np.max(metrics),
            "mean_time_per_worker": np.mean(times),
            "total_wall_clock_time": total_time,
            "speedup": np.sum(times) / total_time if total_time > 0 else 0,
        }
        
        if verbose:
            print(f"✓ Winner: Worker #{best_result['worker_id']}")
            print(f"  Position: {best_result['best_position']}")
            print(f"  Metric: {best_result['best_metric']:.4f}")
            print()
            print("Aggregate Statistics:")
            print(f"  Metric range: [{aggregate_stats['min_metric']:.4f}, "
                  f"{aggregate_stats['max_metric']:.4f}]")
            print(f"  Mean metric: {aggregate_stats['mean_metric']:.4f} "
                  f"± {aggregate_stats['std_metric']:.4f}")
            print(f"  Mean time per worker: {aggregate_stats['mean_time_per_worker']:.2f}s")
            print(f"  Total wall-clock time: {total_time:.2f}s")
            print(f"  Speedup: {aggregate_stats['speedup']:.2f}x")
            print("=" * 80)
        
        return {
            "all_results": all_results,
            "best_result": best_result,
            "best_worker_id": best_result["worker_id"],
            "total_time": total_time,
            "aggregate_stats": aggregate_stats,
        }
    
    def plot_results(
        self,
        results: Dict[str, Any],
        metric_name: str = "Min RSS (dBm)",
    ) -> None:
        """
        Visualize results from parallel optimization.
        
        Creates plots showing:
        1. Distribution of final metrics across workers
        2. Best vs worst trajectories
        3. Coverage map comparison
        
        Args:
            results: Results dictionary from optimize()
            metric_name: Name of metric for plot labels
        """
        import matplotlib.pyplot as plt
        
        all_results = results["all_results"]
        best_result = results["best_result"]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Distribution of final metrics
        ax = axes[0, 0]
        metrics = [r["best_metric"] for r in all_results]
        ax.hist(metrics, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(
            best_result["best_metric"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Best: {best_result['best_metric']:.2f}",
        )
        ax.set_xlabel(metric_name)
        ax.set_ylabel("Number of Workers")
        ax.set_title("Distribution of Final Metrics Across Workers")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Final positions scatter
        ax = axes[0, 1]
        positions = np.array([r["best_position"] for r in all_results])
        colors = metrics
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=colors,
            s=100,
            cmap="viridis",
            edgecolor="black",
            alpha=0.7,
        )
        ax.plot(
            best_result["best_position"][0],
            best_result["best_position"][1],
            "r*",
            markersize=20,
            label="Best",
        )
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Final Positions (color = metric)")
        plt.colorbar(scatter, ax=ax, label=metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Execution times
        ax = axes[1, 0]
        times = [r["time_elapsed"] for r in all_results]
        worker_ids = [r["worker_id"] for r in all_results]
        ax.bar(worker_ids, times, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Worker ID")
        ax.set_ylabel("Time (s)")
        ax.set_title("Optimization Time per Worker")
        ax.grid(True, alpha=0.3, axis="y")
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis("off")
        stats = results["aggregate_stats"]
        summary_text = f"""
        PARALLEL OPTIMIZATION SUMMARY
        
        Total Workers: {len(all_results)}
        
        Best Worker: #{best_result['worker_id']}
        Best Position: [{best_result['best_position'][0]:.2f}, 
                       {best_result['best_position'][1]:.2f}, 
                       {best_result['best_position'][2]:.2f}]
        Best Metric: {best_result['best_metric']:.4f}
        
        Metric Statistics:
          Mean: {stats['mean_metric']:.4f} ± {stats['std_metric']:.4f}
          Range: [{stats['min_metric']:.4f}, {stats['max_metric']:.4f}]
        
        Performance:
          Avg time per worker: {stats['mean_time_per_worker']:.2f}s
          Total wall-clock: {stats['total_wall_clock_time']:.2f}s
          Speedup: {stats['speedup']:.2f}x
        """
        ax.text(
            0.1, 0.5, summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )
        
        plt.tight_layout()
        plt.show()
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        ray.shutdown()


def generate_random_initial_positions(
    num_positions: int,
    bounds: Dict[str, float],
    fixed_z: float = 3.8,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Generate random initial positions for parallel optimization.
    
    This creates diverse starting points for exploring the optimization
    landscape, helping avoid getting trapped in poor local minima.
    
    Args:
        num_positions: Number of positions to generate
        bounds: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max'
        fixed_z: Fixed z-coordinate (height)
        seed: Random seed for reproducibility
        
    Returns:
        List of position arrays [x, y, z]
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_positions = np.random.uniform(
        bounds["x_min"], bounds["x_max"], num_positions
    )
    y_positions = np.random.uniform(
        bounds["y_min"], bounds["y_max"], num_positions
    )
    z_positions = np.full(num_positions, fixed_z)
    
    return [
        np.array([x, y, z])
        for x, y, z in zip(x_positions, y_positions, z_positions)
    ]
