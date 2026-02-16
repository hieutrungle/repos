"""
Ray ActorPool execution engine for parallel optimization.

This module provides ``RayActorPoolExecutor`` — a **generic, algorithm-agnostic**
execution engine that manages a pool of persistent ``OptimizationWorker`` Ray
actors.  It knows nothing about genetic algorithms, DEAP, or fitness.  Its sole
job is to distribute work items to workers and return results **in order**.

Architecture — Inversion of Control (IoC):
    The execution engine is *injected* into the algorithm layer via DEAP's
    ``toolbox.register("map", executor.map)``.  This separates the
    "Execution Engine" (Ray) from the "Algorithm Logic" (DEAP) cleanly.

Key Design:
    * Uses ``pool.map`` (**ordered, synchronous**) instead of ``map_unordered``
      to prevent freeze issues caused by out-of-order result processing,
      desynchronised task-ID tracking, or silent worker failures.
    * Result[i] always corresponds to input[i] — no complex ID mapping needed.
    * Each ``map`` call acts as a **synchronisation barrier**: the calling code
      blocks until every item in the batch has been evaluated.

Usage::

    executor = RayActorPoolExecutor(scene_config, num_workers=4)
    results = executor.map(format_func, population)   # ordered results
    executor.shutdown()
"""

from typing import Any, Callable, Dict, Iterable, List

import ray
from ray.util.actor_pool import ActorPool

from reflector_position.optimizers.ray_parallel_optimizer import (
    OptimizationWorker,
)


class RayActorPoolExecutor:
    """
    Generic Ray execution engine managing an ActorPool of OptimizationWorkers.

    Provides a ``map(func, iterable)`` interface compatible with DEAP's
    ``toolbox.register("map", ...)``.  Knows **nothing** about genetic
    algorithms — only distributes work to persistent actors.

    Why ``pool.map`` (not ``map_unordered``):
        1. Preserves input order → Result[i] corresponds to Input[i].
        2. No need for complex task_id ↔ individual mapping.
        3. Blocks until all items complete → natural generation barrier.
        4. If a worker crashes, the error propagates immediately rather than
           hanging indefinitely.
    """

    def __init__(
        self,
        scene_config: Dict[str, Any],
        num_workers: int = 4,
        gpu_fraction: float = 0.25,
        verbose: bool = True,
    ):
        """
        Spawn persistent ``OptimizationWorker`` actors and build an ActorPool.

        Args:
            scene_config: Configuration dict for ``setup_building_floor_scene``
                (must contain ``scene_path``).
            num_workers: Number of persistent actors in the pool.
            gpu_fraction: GPU fraction per worker (0.25 → 4 workers / GPU).
            verbose: Print pool creation progress.
        """
        self.num_workers = num_workers
        self.gpu_fraction = gpu_fraction
        self.scene_config = scene_config

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        actor_options: Dict[str, Any] = {"num_cpus": 1}
        if gpu_fraction > 0:
            actor_options["num_gpus"] = gpu_fraction

        if verbose:
            print(
                f"  Spawning {num_workers} Ray workers "
                f"(GPU fraction={gpu_fraction}) ..."
            )

        self._workers = [
            OptimizationWorker.options(**actor_options).remote(
                worker_id=i,
                scene_config=scene_config,
            )
            for i in range(num_workers)
        ]
        self._pool = ActorPool(self._workers)

        if verbose:
            print(f"  ActorPool ready: {num_workers} workers")

    # ------------------------------------------------------------------
    # Core map interface
    # ------------------------------------------------------------------

    def map(self, func: Callable, iterable: Iterable) -> List[Dict[str, Any]]:
        """
        Map a function over items using the Ray ActorPool.

        This is the DEAP-compatible ``map`` interface.  DEAP calls::

            toolbox.map(toolbox.evaluate, population)

        which, after ``toolbox.register("map", executor.map)``, becomes::

            executor.map(format_func, population)

        Flow:
            1. Apply *func* to each item → list of worker arg tuples.
            2. Submit all tuples to ``pool.map`` (ordered, synchronous).
            3. Return ordered list of worker result dicts.

        Args:
            func: Callable that converts an item (e.g. DEAP individual)
                into a tuple ``(task_id, optimizer_method, optimizer_kwargs,
                optimization_params)`` for ``OptimizationWorker.optimize``.
            iterable: Items to process (e.g. DEAP individuals with
                invalidated fitness).

        Returns:
            Ordered list of result dicts from ``OptimizationWorker.optimize``
            (result[i] corresponds to iterable[i]).
        """
        items = list(iterable)
        if not items:
            return []

        # 1. Format: apply func to each item → worker arg tuples
        task_args = [func(item) for item in items]

        # 2. Submit to pool using pool.map (ORDERED, SYNCHRONOUS)
        #    ╔═══════════════════════════════════════════════════╗
        #    ║  NOT map_unordered — prevents freeze issues from ║
        #    ║  out-of-order results / silent worker failures.  ║
        #    ╚═══════════════════════════════════════════════════╝
        results = list(
            self._pool.map(
                lambda actor, args: actor.optimize.remote(*args),
                task_args,
            )
        )

        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Kill all worker actors and release resources."""
        for w in self._workers:
            try:
                ray.kill(w)
            except Exception:
                pass
        self._workers = []
        self._pool = None
        print("  ActorPool shut down.")
