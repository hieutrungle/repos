"""
Pure DEAP Genetic Algorithm logic for AP position optimization.

This module implements the evolutionary algorithm using the DEAP library.
It does **NOT** import Ray.  The parallel evaluation is injected via
dependency injection of a ``map`` function through DEAP's
``toolbox.register("map", ...)``.

Architecture — Inversion of Control (IoC):
    ``GeneticAlgorithmRunner`` accepts an ``executor_map`` callable in its
    constructor.  Internally it registers this callable as
    ``toolbox.register("map", executor_map)``, so every call to
    ``toolbox.map(toolbox.evaluate, population)`` is transparently routed
    to whatever execution engine was injected (Ray, multiprocessing,
    sequential ``map``, etc.).

Key Principles:
    1. **No Ray imports** — this file is a pure algorithm module.
    2. **Dependency injection** via ``executor_map``.
    3. ``_format_individual(ind)`` converts a DEAP ``[x, y]`` individual into
       the argument tuple expected by ``OptimizationWorker.optimize``.
    4. The injected ``map`` returns raw worker result dicts; fitness extraction
       (``result["best_metric"]``) happens here in the GA runner.

Usage::

    from reflector_position.optimizers.deap_logic import GeneticAlgorithmRunner

    ga = GeneticAlgorithmRunner(
        position_bounds={"x_min": 5, "x_max": 25, "y_min": 5, "y_max": 25},
        fixed_z=3.8,
        executor_map=executor.map,   # injected from RayActorPoolExecutor
    )
    results = ga.run(
        optimization_params={"samples_per_tx": 1_000_000, "max_depth": 13},
        ga_params={"pop_size": 50, "n_gen": 20},
    )
    ga.save_evolution_plot(results, "results/ga_evolution.png")
"""

import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from deap import base, creator, tools

from reflector_position.metrics import POWER_EPSILON

# ---------------------------------------------------------------------------
# DEAP creator types — created once at module level.
# Guard against duplicate creation when the module is reloaded.
# ---------------------------------------------------------------------------
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def _rss_watts_to_dbm(rss_watt: float) -> float:
    """Convert RSS from Watts (linear) to dBm."""
    return 10.0 * np.log10(max(rss_watt, POWER_EPSILON)) + 30.0


class GeneticAlgorithmRunner:
    """
    Pure DEAP implementation of a Genetic Algorithm for AP positioning.

    Decoupled from Ray via dependency injection of the ``map`` function.
    All evolutionary logic (selection, crossover, mutation, bounds clamping,
    statistics, plotting) lives here.  The expensive fitness evaluation is
    delegated to the injected ``executor_map``.

    The GA maximises the **minimum RSS** (linear Watts) across transmitters,
    which is the ``best_metric`` field returned by workers.
    """

    def __init__(
        self,
        position_bounds: Dict[str, float],
        fixed_z: float,
        executor_map: Callable,
    ):
        """
        Args:
            position_bounds: Search space ``{x_min, x_max, y_min, y_max}``.
            fixed_z: Fixed z-coordinate for all AP positions.
            executor_map: A ``map(func, iterable) -> list[result_dict]``
                callable.  Typically ``RayActorPoolExecutor.map``.
        """
        self.bounds = position_bounds
        self.fixed_z = fixed_z
        self._executor_map = executor_map

        # Task counter for sequential IDs (useful for logging)
        self._task_counter: int = 0

        # Set during run() so _format_individual can access them
        self._opt_params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Task formatting (the "evaluate" function registered on the toolbox)
    # ------------------------------------------------------------------

    def _format_individual(self, ind) -> Tuple:
        """
        Convert a DEAP individual ``[x, y]`` into the argument tuple
        expected by ``OptimizationWorker.optimize``.

        Returns:
            ``(task_id, "grid_search_point", optimizer_kwargs, optimization_params)``
        """
        task_id = self._task_counter
        self._task_counter += 1
        return (
            task_id,
            "grid_search_point",
            {
                "evaluation_position": (float(ind[0]), float(ind[1])),
                "fixed_z": self.fixed_z,
            },
            self._opt_params,
        )

    # ------------------------------------------------------------------
    # Bounds enforcement
    # ------------------------------------------------------------------

    def _clamp_individual(self, ind) -> None:
        """Clamp an individual's genes to the position bounds (in-place)."""
        ind[0] = max(self.bounds["x_min"], min(ind[0], self.bounds["x_max"]))
        ind[1] = max(self.bounds["y_min"], min(ind[1], self.bounds["y_max"]))

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def _evaluate_invalid(
        self,
        population: List,
        toolbox: base.Toolbox,
    ) -> int:
        """
        Evaluate individuals with invalidated fitness via the injected map.

        Flow:
            1. Collect individuals whose fitness is invalid.
            2. ``toolbox.map(toolbox.evaluate, invalid_ind)``
               → ``executor_map(_format_individual, invalid_ind)``
               → list of worker result dicts (ordered).
            3. Extract ``result["best_metric"]`` as DEAP fitness tuple.

        Returns:
            Number of individuals evaluated.
        """
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        if not invalid_ind:
            return 0

        # toolbox.map calls executor_map(_format_individual, invalid_ind)
        # → returns list of worker result dicts, preserving order
        results = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, res in zip(invalid_ind, results):
            # Fitness is a tuple for DEAP — higher RSS is better
            ind.fitness.values = (res["best_metric"],)

        return len(invalid_ind)

    # ------------------------------------------------------------------
    # Main GA loop
    # ------------------------------------------------------------------

    def run(
        self,
        optimization_params: Optional[Dict[str, Any]] = None,
        ga_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the Genetic Algorithm.

        Args:
            optimization_params: Kwargs for the radio-map evaluation, e.g.
                ``{"samples_per_tx": 1_000_000, "max_depth": 13}``.
            ga_params: GA hyper-parameters:
                - ``pop_size`` (int): Population size (default 50).
                - ``n_gen`` (int): Number of generations (default 20).
                - ``cxpb`` (float): Crossover probability (default 0.7).
                - ``mutpb`` (float): Mutation probability (default 0.3).
                - ``tournsize`` (int): Tournament size (default 3).
                - ``cx_alpha`` (float): Blend crossover alpha (default 0.5).
                - ``mut_mu`` (float): Gaussian mutation mean (default 0.0).
                - ``mut_sigma`` (float): Gaussian mutation std-dev (default 2.0).
                - ``mut_indpb`` (float): Per-gene mutation prob (default 0.2).
                - ``hof_size`` (int): Hall-of-fame size (default 5).
            seed: Random seed for reproducibility.
            verbose: Print generation-by-generation progress.

        Returns:
            Dictionary with ``best_individual``, ``best_fitness``,
            ``best_fitness_dbm``, ``best_position``, ``hall_of_fame``,
            ``logbook``, ``total_time``, ``total_evaluations``,
            ``ga_params``, ``generation_details``.
        """
        # -- Unpack GA hyper-parameters ---------------------------------
        ga = ga_params or {}
        pop_size = ga.get("pop_size", 50)
        n_gen = ga.get("n_gen", 20)
        cxpb = ga.get("cxpb", 0.7)
        mutpb = ga.get("mutpb", 0.3)
        tournsize = ga.get("tournsize", 3)
        cx_alpha = ga.get("cx_alpha", 0.5)
        mut_mu = ga.get("mut_mu", 0.0)
        mut_sigma = ga.get("mut_sigma", 2.0)
        mut_indpb = ga.get("mut_indpb", 0.2)
        hof_size = ga.get("hof_size", 5)

        # Store opt params so _format_individual can read them
        self._opt_params = optimization_params or {
            "samples_per_tx": 1_000_000,
            "max_depth": 13,
            "verbose": False,
        }
        self._opt_params.setdefault("verbose", False)

        # Reset task counter
        self._task_counter = 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        start_time = time.time()

        if verbose:
            print("=" * 80)
            print("DEAP GENETIC ALGORITHM (modular, IoC-injected map)")
            print("=" * 80)
            print(f"  Population: {pop_size} | Generations: {n_gen}")
            print(f"  cxpb={cxpb}  mutpb={mutpb}  tournsize={tournsize}")
            print(f"  cx_alpha={cx_alpha}  mut_sigma={mut_sigma}  "
                  f"mut_indpb={mut_indpb}")
            print(f"  Bounds: x=[{self.bounds['x_min']}, "
                  f"{self.bounds['x_max']}], "
                  f"y=[{self.bounds['y_min']}, {self.bounds['y_max']}]")
            print("-" * 80)

        # -- DEAP toolbox setup ----------------------------------------
        toolbox = base.Toolbox()

        # Gene generators (uniform within bounds)
        toolbox.register(
            "attr_x",
            random.uniform,
            self.bounds["x_min"],
            self.bounds["x_max"],
        )
        toolbox.register(
            "attr_y",
            random.uniform,
            self.bounds["y_min"],
            self.bounds["y_max"],
        )
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (toolbox.attr_x, toolbox.attr_y),
            n=1,
        )
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual,
        )

        # Evolutionary operators
        toolbox.register("mate", tools.cxBlend, alpha=cx_alpha)
        toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=mut_mu,
            sigma=mut_sigma,
            indpb=mut_indpb,
        )
        toolbox.register("select", tools.selTournament, tournsize=tournsize)

        # ╔═══════════════════════════════════════════════════════════╗
        # ║  DEPENDENCY INJECTION: register the injected map + eval  ║
        # ╚═══════════════════════════════════════════════════════════╝
        toolbox.register("map", self._executor_map)
        toolbox.register("evaluate", self._format_individual)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("max", np.max)
        stats.register("mean", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)

        logbook = tools.Logbook()
        logbook.header = [
            "gen", "nevals", "max", "mean", "std", "min",
            "max_dbm", "mean_dbm", "best_pos",
        ]

        hof = tools.HallOfFame(hof_size)

        # -- Initial population -----------------------------------------
        population = toolbox.population(n=pop_size)

        total_evaluations = 0
        generation_details: List[Dict[str, Any]] = []

        if verbose:
            print(f"\nGen  0: evaluating {pop_size} individuals ...")
        gen_start = time.time()
        nevals = self._evaluate_invalid(population, toolbox)
        gen_time = time.time() - gen_start
        total_evaluations += nevals

        hof.update(population)
        record = stats.compile(population)
        best_ind = hof[0]
        record["max_dbm"] = _rss_watts_to_dbm(record["max"])
        record["mean_dbm"] = _rss_watts_to_dbm(record["mean"])
        record["best_pos"] = f"({best_ind[0]:.2f}, {best_ind[1]:.2f})"
        logbook.record(gen=0, nevals=nevals, **record)

        generation_details.append({
            "gen": 0,
            "nevals": nevals,
            "max_dbm": record["max_dbm"],
            "mean_dbm": record["mean_dbm"],
            "std": record["std"],
            "best_x": float(best_ind[0]),
            "best_y": float(best_ind[1]),
            "time": gen_time,
        })

        if verbose:
            print(
                f"  Gen  0 | evals={nevals:>3d} | "
                f"best={record['max_dbm']:.2f} dBm | "
                f"mean={record['mean_dbm']:.2f} dBm | "
                f"pos=({best_ind[0]:.2f}, {best_ind[1]:.2f}) | "
                f"time={gen_time:.1f}s"
            )

        # -- Generational loop ------------------------------------------
        for gen in range(1, n_gen + 1):
            gen_start = time.time()

            # Selection
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Clamp all individuals to bounds
            for ind in offspring:
                self._clamp_individual(ind)

            # Evaluate individuals with invalid fitness
            nevals = self._evaluate_invalid(offspring, toolbox)
            total_evaluations += nevals

            # Replace population
            population[:] = offspring

            # Update hall of fame
            hof.update(population)

            # Record statistics
            record = stats.compile(population)
            best_ind = hof[0]
            record["max_dbm"] = _rss_watts_to_dbm(record["max"])
            record["mean_dbm"] = _rss_watts_to_dbm(record["mean"])
            record["best_pos"] = f"({best_ind[0]:.2f}, {best_ind[1]:.2f})"
            logbook.record(gen=gen, nevals=nevals, **record)
            gen_time = time.time() - gen_start

            generation_details.append({
                "gen": gen,
                "nevals": nevals,
                "max_dbm": record["max_dbm"],
                "mean_dbm": record["mean_dbm"],
                "std": record["std"],
                "best_x": float(best_ind[0]),
                "best_y": float(best_ind[1]),
                "time": gen_time,
            })

            if verbose:
                print(
                    f"  Gen {gen:>2d} | evals={nevals:>3d} | "
                    f"best={record['max_dbm']:.2f} dBm | "
                    f"mean={record['mean_dbm']:.2f} dBm | "
                    f"pos=({best_ind[0]:.2f}, {best_ind[1]:.2f}) | "
                    f"time={gen_time:.1f}s"
                )

        total_time = time.time() - start_time

        # -- Build results dict -----------------------------------------
        best = hof[0]
        best_fitness = best.fitness.values[0]

        hall_of_fame_list = []
        for ind in hof:
            hall_of_fame_list.append({
                "position": [float(ind[0]), float(ind[1]), self.fixed_z],
                "fitness": float(ind.fitness.values[0]),
                "fitness_dbm": _rss_watts_to_dbm(ind.fitness.values[0]),
            })

        results = {
            "best_individual": [float(best[0]), float(best[1])],
            "best_fitness": float(best_fitness),
            "best_fitness_dbm": _rss_watts_to_dbm(best_fitness),
            "best_position": [float(best[0]), float(best[1]), self.fixed_z],
            "hall_of_fame": hall_of_fame_list,
            "logbook": logbook,
            "total_time": total_time,
            "total_evaluations": total_evaluations,
            "ga_params": {
                "pop_size": pop_size,
                "n_gen": n_gen,
                "cxpb": cxpb,
                "mutpb": mutpb,
                "tournsize": tournsize,
                "cx_alpha": cx_alpha,
                "mut_mu": mut_mu,
                "mut_sigma": mut_sigma,
                "mut_indpb": mut_indpb,
                "hof_size": hof_size,
                "seed": seed,
            },
            "generation_details": generation_details,
        }

        if verbose:
            print("-" * 80)
            print("GA COMPLETE")
            print(f"  Best position: ({best[0]:.2f}, {best[1]:.2f}, "
                  f"{self.fixed_z})")
            print(f"  Best Min RSS:  {results['best_fitness_dbm']:.2f} dBm")
            print(f"  Total evals:   {total_evaluations}")
            print(f"  Wall-clock:    {total_time:.2f}s")
            print("=" * 80)

        return results

    # ------------------------------------------------------------------
    # Plotting (no Ray dependency — pure matplotlib)
    # ------------------------------------------------------------------

    def save_evolution_plot(
        self,
        results: Dict[str, Any],
        save_path: str,
        position_bounds: Optional[Dict[str, float]] = None,
        rss_range_dbm: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Save a 2×2 visualisation of the GA evolution.

        Panels:
            1. Best & mean fitness (dBm) vs generation
            2. Best position trajectory over generations
            3. Hall-of-fame positions scatter
            4. Summary text

        Args:
            results: Dictionary returned by :meth:`run`.
            save_path: Output file path (e.g. ``results/ga_evolution.png``).
            position_bounds: Optional axis limits for position plots.
                Defaults to ``self.bounds``.
            rss_range_dbm: Optional ``(min_dbm, max_dbm)`` for RSS y-axis.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bounds = position_bounds or self.bounds
        gen_details = results["generation_details"]
        gens = [g["gen"] for g in gen_details]
        best_dbm = [g["max_dbm"] for g in gen_details]
        mean_dbm = [g["mean_dbm"] for g in gen_details]
        best_xs = [g["best_x"] for g in gen_details]
        best_ys = [g["best_y"] for g in gen_details]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Fitness convergence
        ax = axes[0, 0]
        ax.plot(gens, best_dbm, "r-o", markersize=3, linewidth=1.5,
                label="Best (HoF)")
        ax.plot(gens, mean_dbm, "b--", linewidth=1.0, alpha=0.7,
                label="Pop Mean")
        if rss_range_dbm:
            ax.set_ylim(rss_range_dbm[0], rss_range_dbm[1])
        ax.set_xlabel("Generation")
        ax.set_ylabel("Min RSS (dBm)")
        ax.set_title("Fitness Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Best position trajectory
        ax = axes[0, 1]
        ax.plot(best_xs, best_ys, "b-o", markersize=4, linewidth=1.5,
                alpha=0.7)
        ax.plot(best_xs[0], best_ys[0], "go", markersize=12, label="Gen 0")
        ax.plot(best_xs[-1], best_ys[-1], "r*", markersize=15, label="Final")
        if bounds:
            ax.set_xlim(bounds["x_min"], bounds["x_max"])
            ax.set_ylim(bounds["y_min"], bounds["y_max"])
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Best Position Trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Hall-of-fame scatter
        ax = axes[1, 0]
        hof = results["hall_of_fame"]
        hof_x = [h["position"][0] for h in hof]
        hof_y = [h["position"][1] for h in hof]
        hof_dbm = [h["fitness_dbm"] for h in hof]
        sc = ax.scatter(
            hof_x, hof_y,
            c=hof_dbm, s=200, cmap="viridis", edgecolor="black",
            vmin=rss_range_dbm[0] if rss_range_dbm else None,
            vmax=rss_range_dbm[1] if rss_range_dbm else None,
        )
        for i, (x, y, d) in enumerate(zip(hof_x, hof_y, hof_dbm)):
            ax.annotate(
                f"#{i + 1}\n{d:.1f}", (x, y),
                textcoords="offset points", xytext=(8, 4), fontsize=7,
            )
        if bounds:
            ax.set_xlim(bounds["x_min"], bounds["x_max"])
            ax.set_ylim(bounds["y_min"], bounds["y_max"])
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Hall of Fame (top solutions)")
        plt.colorbar(sc, ax=ax, label="Min RSS (dBm)")
        ax.grid(True, alpha=0.3)

        # 4. Summary text
        ax = axes[1, 1]
        ax.axis("off")
        ga_p = results["ga_params"]
        summary = (
            f"DEAP GA SUMMARY (Modular IoC)\n"
            f"\n"
            f"Population: {ga_p['pop_size']} | Generations: {ga_p['n_gen']}\n"
            f"cxpb={ga_p['cxpb']}  mutpb={ga_p['mutpb']}  "
            f"tournsize={ga_p['tournsize']}\n"
            f"cx_alpha={ga_p['cx_alpha']}  "
            f"mut_sigma={ga_p['mut_sigma']}  "
            f"mut_indpb={ga_p['mut_indpb']}\n"
            f"\n"
            f"Best Position: ({results['best_position'][0]:.2f}, "
            f"{results['best_position'][1]:.2f}, "
            f"{results['best_position'][2]:.2f})\n"
            f"Best Min RSS:  {results['best_fitness_dbm']:.2f} dBm\n"
            f"\n"
            f"Total evaluations: {results['total_evaluations']}\n"
            f"Wall-clock time:   {results['total_time']:.2f}s\n"
            f"Avg time/gen:      "
            f"{results['total_time'] / (ga_p['n_gen'] + 1):.2f}s\n"
        )
        ax.text(
            0.1, 0.5, summary,
            transform=ax.transAxes, fontsize=10,
            verticalalignment="center", family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"GA evolution plot saved to: {save_path}")
