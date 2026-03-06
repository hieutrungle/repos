"""
Ray-based parallel optimizer wrapper for distributed multi-start optimization.

This module implements distributed parallel optimization using Ray, enabling
multiple independent optimization trajectories to run simultaneously across
different workers/GPUs. This is essential when optimizing physical scene
geometry (reflector positions) where each instance needs independent Scene copies.

Key Design Principles:
1. Process-level isolation: Each Ray actor has its own Scene instance
2. True parallelism: Multiple "parallel universes" exploring different starting points
3. GPU efficiency: Configurable GPU fraction per worker (e.g., 0.25 allows 4 workers per GPU)
4. Optimizer agnostic: Works with any optimizer registered in OptimizerFactory

Architecture (inspired by Ray distributed training patterns):
- Orchestrator (Driver): Manages Ray actors, aggregates results
- Ray Actors (Workers): Each runs independent optimization with its own Scene
- Reduction Phase: Winner selection from all parallel trajectories

Usage pattern mirrors Ray's distributed training:
    1. Define worker configs (like ScalingConfig in Ray Train)
    2. Spawn actors with resource allocation
    3. Execute in parallel (like trainer.fit())
    4. Collect and aggregate results
"""

import os
import time
from typing import Dict, List, Optional, Any

import numpy as np
import ray
from ray.actor import ActorHandle

from reflector_position.metrics import POWER_EPSILON

def _rss_watts_to_dbm(rss_watt: float) -> float:
    """Convert RSS from Watts (linear) to dBm. Mirrors metrics.rss_to_dbm."""
    return 10.0 * np.log10(max(rss_watt, POWER_EPSILON)) + 30.0


@ray.remote
class OptimizationWorker:
    """
    Ray actor that runs an independent optimization trajectory.

    Each worker:
    1. Loads its own Scene instance via setup_building_floor_scene (isolated memory)
    2. Creates its optimizer via OptimizerFactory
    3. Runs optimization independently
    4. Returns serializable results to orchestrator

    This provides true process-level isolation needed for modifying
    physical scene geometry (reflectors, walls, obstacles).
    """

    def __init__(
        self,
        worker_id: int,
        scene_config: Dict[str, Any],
        optimizer_method: str,
        optimizer_kwargs: Dict[str, Any],
    ):
        """
        Initialize optimization worker.

        Args:
            worker_id: Unique identifier for this worker
            scene_config: Configuration for setup_building_floor_scene():
                - scene_path (str): Path to the scene XML file (required)
                - frequency (float): Operating frequency in Hz (default: 5.18e9)
                - tx_positions (list): Transmitter positions [(x,y,z), ...]
                - tx_power_dbm (float): Transmitter power in dBm (default: 5.0)
                - rx_position (tuple): Receiver position (x,y,z)
            optimizer_method: 'gradient_descent' or 'grid_search'
            optimizer_kwargs: Keyword arguments for OptimizerFactory.create(),
                e.g. for gradient_descent: initial_position, position_bounds, fixed_z
                e.g. for grid_search: search_bounds, grid_resolution, fixed_z
        """
        self.worker_id = worker_id
        self.optimizer_method = optimizer_method

        # Load Scene instance (CRITICAL: each worker gets its own Scene)
        self.scene = self._load_scene(scene_config)

        # Initialize optimizer using the factory pattern
        from reflector_position.optimizers.optimizer_factory import OptimizerFactory

        self.optimizer = OptimizerFactory.create(
            method=optimizer_method,
            scene=self.scene,
            **optimizer_kwargs,
        )

    def _load_scene(self, scene_config: Dict[str, Any]):
        """
        Load a fully configured Scene instance using setup_building_floor_scene.

        This mirrors the pattern from full_comparison.py and
        optimizer_factory_example.py, ensuring Tx/Rx arrays, transmitters,
        and receivers are properly configured.

        Args:
            scene_config: Dict with scene_path (required) and optional params.

        Returns:
            Fully configured sionna.rt.Scene instance.
        """
        from reflector_position.scene_setup import setup_building_floor_scene

        scene = setup_building_floor_scene(
            scene_path=str(scene_config["scene_path"]),
            frequency=scene_config.get("frequency", 5.18e9),
            tx_positions=scene_config.get("tx_positions", None),
            tx_power_dbm=scene_config.get("tx_power_dbm", 5.0),
            rx_position=scene_config.get("rx_position", (16.0, 6.5, 1.5)),
        )

        return scene

    def optimize(self, optimization_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run optimization and return serializable results.

        Args:
            optimization_params: Keyword arguments passed to optimizer.optimize().
                For gradient_descent: num_iterations, learning_rate, samples_per_tx,
                    max_depth, use_soft_min, temperature, verbose
                For grid_search: samples_per_tx, max_depth,
                    coverage_threshold_dbm, verbose

        Returns:
            Dictionary with:
                - worker_id: Worker identifier
                - best_position: Optimized position [x, y, z] as list
                - best_metric: Best metric value achieved (float)
                - time_elapsed: Total optimization time in seconds
                - history: Optimization history (if available)
        """
        start_time = time.time()

        # Run optimization (same call pattern as full_comparison.py)
        result = self.optimizer.optimize(**optimization_params)

        elapsed_time = time.time() - start_time

        # Unpack result tuple
        if isinstance(result, tuple) and len(result) == 2:
            best_position, best_metric = result
        else:
            best_position = None
            best_metric = float("-inf")

        # Convert to serializable types
        if best_position is not None:
            best_position = np.asarray(best_position).tolist()
        else:
            best_position = [0.0, 0.0, 0.0]

        metric_linear = float(best_metric) if best_metric is not None else 0.0
        metric_dbm = _rss_watts_to_dbm(metric_linear)

        output = {
            "worker_id": self.worker_id,
            "best_position": best_position,
            "best_metric": metric_linear,
            "best_metric_dbm": metric_dbm,
            "time_elapsed": elapsed_time,
        }

        # Include history if optimizer tracks it (gradient descent has history)
        if hasattr(self.optimizer, "history"):
            history = {}
            for k, v in self.optimizer.history.items():
                if isinstance(v, list):
                    serialized = []
                    for item in v:
                        if hasattr(item, "tolist"):
                            serialized.append(item.tolist())
                        elif hasattr(item, "item"):
                            serialized.append(item.item())
                        else:
                            serialized.append(item)
                    history[k] = serialized
                else:
                    history[k] = v
            output["history"] = history

        # Include grid search results if available
        if hasattr(self.optimizer, "results"):
            gs_results = {}
            for k, v in self.optimizer.results.items():
                if k == "radio_maps":
                    continue  # Skip non-serializable radio maps
                if isinstance(v, list):
                    serialized = []
                    for item in v:
                        if hasattr(item, "tolist"):
                            serialized.append(item.tolist())
                        elif hasattr(item, "item"):
                            serialized.append(item.item())
                        else:
                            serialized.append(item)
                    gs_results[k] = serialized
                else:
                    gs_results[k] = v
            output["grid_results"] = gs_results

        return output

    def get_worker_id(self) -> int:
        """Return the worker ID."""
        return self.worker_id


class RayParallelOptimizer:
    """
    Orchestrator for distributed parallel optimization using Ray.

    Implements the "Fork -> Map -> Reduce" workflow:
    1. Fork: Spawn Ray actors (workers), each with its own Scene instance
    2. Map: Each worker runs optimization independently and in parallel
    3. Reduce: Aggregate results and select best configuration

    Following Ray's patterns (similar to ScalingConfig in Ray Train):
    - num_workers controls parallelism level
    - gpu_fraction controls GPU sharing (0.25 = 4 workers/GPU)
    - Each worker is an isolated Ray actor with its own resources

    Example:
        >>> import ray
        >>> from pathlib import Path
        >>> from reflector_position.optimizers import (
        ...     RayParallelOptimizer, generate_random_initial_positions
        ... )
        >>>
        >>> ray.init(ignore_reinit_error=True)
        >>>
        >>> scene_config = {
        ...     "scene_path": Path.home() / "blender/models/building_floor/building_floor.xml",
        ...     "frequency": 5.18e9,
        ...     "tx_power_dbm": 5.0,
        ... }
        >>>
        >>> parallel_opt = RayParallelOptimizer(num_workers=4, gpu_fraction=0.25)
        >>>
        >>> bounds = {"x_min": 5.0, "x_max": 25.0, "y_min": 5.0, "y_max": 25.0}
        >>> positions = generate_random_initial_positions(4, bounds, seed=42)
        >>>
        >>> worker_optimizer_kwargs = [
        ...     {"initial_position": (pos[0], pos[1]),
        ...      "position_bounds": bounds}
        ...     for pos in positions
        ... ]
        >>>
        >>> results = parallel_opt.run(
        ...     scene_config=scene_config,
        ...     optimizer_method="gradient_descent",
        ...     worker_optimizer_kwargs=worker_optimizer_kwargs,
        ...     optimization_params={
        ...         "num_iterations": 30,
        ...         "learning_rate": 0.5,
        ...         "samples_per_tx": 1_000_000,
        ...         "max_depth": 13,
        ...         "verbose": False,
        ...     },
        ... )
        >>>
        >>> print(f"Best: Worker #{results['best_worker_id']}")
    """

    def __init__(
        self,
        num_workers: int = 4,
        gpu_fraction: float = 0.25,
    ):
        """
        Initialize Ray parallel optimizer (the orchestrator).

        Args:
            num_workers: Number of parallel workers (optimization trajectories).
            gpu_fraction: GPU fraction per worker. E.g., 0.25 = 4 workers per GPU,
                0.5 = 2 workers per GPU, 0.0 = CPU-only.
        """
        self.num_workers = num_workers
        self.gpu_fraction = gpu_fraction

        # Initialize Ray if not already running
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def run(
        self,
        scene_config: Dict[str, Any],
        optimizer_method: str,
        worker_optimizer_kwargs: List[Dict[str, Any]],
        optimization_params: Dict[str, Any],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run parallel optimization across multiple workers.

        This implements the full "Fork -> Map -> Reduce" workflow:
        1. Fork: Spawn workers with different optimizer configurations
        2. Map: Each worker runs optimization independently
        3. Reduce: Aggregate results and select winner

        Args:
            scene_config: Scene configuration dict passed to each worker's
                setup_building_floor_scene(). Must contain 'scene_path'.
                Optional: 'frequency', 'tx_power_dbm', 'tx_positions', 'rx_position'.
            optimizer_method: 'gradient_descent' or 'grid_search'.
            worker_optimizer_kwargs: List of dicts (one per worker), each containing
                keyword arguments for OptimizerFactory.create().
                For gradient_descent: {'initial_position': (x,y), 'position_bounds': {...}, ...}
                For grid_search: {'search_bounds': {...}, 'grid_resolution': 2.0, ...}
            optimization_params: Dict of keyword arguments for optimizer.optimize().
                For gradient_descent: {'num_iterations': 30, 'learning_rate': 0.5, ...}
                For grid_search: {'samples_per_tx': 1_000_000, 'max_depth': 13, ...}
            verbose: Print progress information.

        Returns:
            Dictionary with:
                - all_results: List of result dicts from all workers
                - best_result: Result dict from best worker
                - best_worker_id: ID of best worker
                - total_time: Total wall-clock time
                - aggregate_stats: Statistics across all workers
        """
        if len(worker_optimizer_kwargs) != self.num_workers:
            raise ValueError(
                f"Length of worker_optimizer_kwargs ({len(worker_optimizer_kwargs)}) "
                f"must equal num_workers ({self.num_workers})"
            )

        start_time = time.time()

        if verbose:
            print("=" * 80)
            print("RAY PARALLEL OPTIMIZATION")
            print("=" * 80)
            print(f"  Workers: {self.num_workers}")
            print(f"  GPU fraction per worker: {self.gpu_fraction}")
            print(f"  Optimizer method: {optimizer_method}")
            if self.gpu_fraction > 0:
                print(f"  Workers per GPU: {int(1.0 / self.gpu_fraction)}")
            else:
                print("  Mode: CPU-only")
            print("-" * 80)

        # ── Phase 1: FORK — Spawn Ray actors ──────────────────────────
        if verbose:
            print("Phase 1/3: Spawning Ray actors...")

        workers: List[ActorHandle] = []
        for i in range(self.num_workers):
            # Set per-actor resource allocation via .options()
            # This is the correct Ray pattern for fractional GPU allocation
            actor_options = {"num_cpus": 1}
            if self.gpu_fraction > 0:
                actor_options["num_gpus"] = self.gpu_fraction

            worker: ActorHandle = OptimizationWorker.options(**actor_options).remote(
                worker_id=i,
                scene_config=scene_config,
                optimizer_method=optimizer_method,
                optimizer_kwargs=worker_optimizer_kwargs[i],
            )
            workers.append(worker)

        if verbose:
            print(f"  Spawned {len(workers)} workers")
            print("-" * 80)

        # ── Phase 2: MAP — Parallel execution ─────────────────────────
        if verbose:
            print("Phase 2/3: Running parallel optimization...")
            print("  (Each worker optimizing independently)")

        # Launch all optimizations in parallel (non-blocking remote calls)
        futures = [
            worker.optimize.remote(optimization_params)
            for worker in workers
        ]

        # Wait for all to complete (blocking)
        all_results = ray.get(futures)

        if verbose:
            print(f"  All {len(all_results)} workers completed")
            print("-" * 80)

        # ── Phase 3: REDUCE — Winner selection ────────────────────────
        if verbose:
            print("Phase 3/3: Aggregating results and selecting winner...")

        # Find best worker (maximize metric — higher min_rss is better)
        best_idx = int(np.argmax([r["best_metric"] for r in all_results]))
        best_result = all_results[best_idx]

        total_time = time.time() - start_time

        # Compute aggregate statistics (in both linear and dBm)
        metrics_linear = [r["best_metric"] for r in all_results]
        metrics_dbm = [r["best_metric_dbm"] for r in all_results]
        times = [r["time_elapsed"] for r in all_results]

        aggregate_stats = {
            # Linear (Watts)
            "mean_metric": float(np.mean(metrics_linear)),
            "std_metric": float(np.std(metrics_linear)),
            "min_metric": float(np.min(metrics_linear)),
            "max_metric": float(np.max(metrics_linear)),
            # dBm (for analysis / plotting)
            "mean_metric_dbm": float(np.mean(metrics_dbm)),
            "std_metric_dbm": float(np.std(metrics_dbm)),
            "min_metric_dbm": float(np.min(metrics_dbm)),
            "max_metric_dbm": float(np.max(metrics_dbm)),
            # Timing
            "mean_time_per_worker": float(np.mean(times)),
            "total_wall_clock_time": total_time,
            # Speedup: total sequential time / actual wall-clock time
            "speedup": float(np.sum(times) / total_time) if total_time > 0 else 0.0,
        }

        if verbose:
            print(f"  Winner: Worker #{best_result['worker_id']}")
            print(f"  Position: {best_result['best_position']}")
            print(f"  P5 RSS: {best_result['best_metric_dbm']:.2f} dBm")
            print()
            print("  Aggregate Statistics:")
            print(
                f"    P5 RSS range: [{aggregate_stats['min_metric_dbm']:.2f}, "
                f"{aggregate_stats['max_metric_dbm']:.2f}] dBm"
            )
            print(
                f"    Mean P5 RSS: {aggregate_stats['mean_metric_dbm']:.2f} "
                f"+/- {aggregate_stats['std_metric_dbm']:.2f} dBm"
            )
            print(f"    Mean time per worker: {aggregate_stats['mean_time_per_worker']:.2f}s")
            print(f"    Total wall-clock time: {total_time:.2f}s")
            print(f"    Speedup: {aggregate_stats['speedup']:.2f}x")
            print("=" * 80)

        return {
            "all_results": all_results,
            "best_result": best_result,
            "best_worker_id": best_result["worker_id"],
            "total_time": total_time,
            "aggregate_stats": aggregate_stats,
        }

    def save_results_plot(
        self,
        results: Dict[str, Any],
        save_path: str,
        metric_name: str = "P5 RSS",
    ) -> None:
        """
        Save visualization of parallel optimization results to a file.

        Creates a 2x2 figure:
        1. Metric distribution across workers
        2. Final positions scatter plot
        3. Time per worker bar chart
        4. Summary statistics text

        Args:
            results: Results dictionary from run().
            save_path: File path to save the plot (e.g., 'results/parallel.png').
            metric_name: Label for the metric axis.
        """
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for saving
        import matplotlib.pyplot as plt

        all_results = results["all_results"]
        best_result = results["best_result"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Use dBm values for all analysis plots
        metrics_dbm = [r["best_metric_dbm"] for r in all_results]
        best_dbm = best_result["best_metric_dbm"]

        # 1. Distribution of final P5 RSS (dBm)
        ax = axes[0, 0]
        ax.hist(metrics_dbm, bins=max(5, len(metrics_dbm) // 2), edgecolor="black", alpha=0.7)
        ax.axvline(
            best_dbm,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Best: {best_dbm:.2f} dBm",
        )
        ax.set_xlabel(f"{metric_name} (dBm)")
        ax.set_ylabel("Number of Workers")
        ax.set_title("Distribution of P5 RSS Across Workers")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Final positions scatter (color = P5 RSS dBm)
        ax = axes[0, 1]
        positions = np.array([r["best_position"] for r in all_results])
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=metrics_dbm,
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
        ax.set_title("Final Positions (color = P5 RSS dBm)")
        plt.colorbar(scatter, ax=ax, label=f"{metric_name} (dBm)")
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
        best_dbm_val = best_result["best_metric_dbm"]
        summary = (
            f"PARALLEL OPTIMIZATION SUMMARY\n"
            f"\n"
            f"Total Workers: {len(all_results)}\n"
            f"\n"
            f"Best Worker: #{best_result['worker_id']}\n"
            f"Best Position: [{best_result['best_position'][0]:.2f}, "
            f"{best_result['best_position'][1]:.2f}, "
            f"{best_result['best_position'][2]:.2f}]\n"
            f"Best P5 RSS: {best_dbm_val:.2f} dBm\n"
            f"\n"
            f"P5 RSS Statistics (dBm):\n"
            f"  Mean: {stats['mean_metric_dbm']:.2f} +/- {stats['std_metric_dbm']:.2f}\n"
            f"  Range: [{stats['min_metric_dbm']:.2f}, {stats['max_metric_dbm']:.2f}]\n"
            f"\n"
            f"Performance:\n"
            f"  Avg time/worker: {stats['mean_time_per_worker']:.2f}s\n"
            f"  Wall-clock: {stats['total_wall_clock_time']:.2f}s\n"
            f"  Speedup: {stats['speedup']:.2f}x"
        )
        ax.text(
            0.1,
            0.5,
            summary,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to: {save_path}")

    def shutdown(self):
        """Shutdown Ray runtime."""
        if ray.is_initialized():
            ray.shutdown()


def generate_random_initial_positions(
    num_positions: int,
    bounds: Dict[str, float],
    fixed_z: float = 3.8,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Generate random initial positions for parallel optimization.

    Creates diverse starting points for exploring the optimization
    landscape, helping avoid getting trapped in poor local minima.

    Args:
        num_positions: Number of positions to generate.
        bounds: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max'.
        fixed_z: Fixed z-coordinate (height).
        seed: Random seed for reproducibility.

    Returns:
        List of position arrays [x, y, z].
    """
    rng = np.random.default_rng(seed)

    x_positions = rng.uniform(bounds["x_min"], bounds["x_max"], num_positions)
    y_positions = rng.uniform(bounds["y_min"], bounds["y_max"], num_positions)

    return [
        np.array([x, y, fixed_z])
        for x, y in zip(x_positions, y_positions)
    ]
