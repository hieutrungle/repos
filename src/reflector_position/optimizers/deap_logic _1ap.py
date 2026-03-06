"""
Pure DEAP Genetic Algorithm logic for AP position and orientation optimization.

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

Chromosome Encoding:
    Each individual is a 4-gene chromosome ``[x, y, dir_x, dir_y]``:

    - ``x, y``: AP position in the horizontal plane (bounded by
      ``position_bounds``).
    - ``dir_x, dir_y``: Horizontal components of the look-at direction
      vector.  The vertical component ``dir_z`` is fixed at ``-0.5``
      (downward bias) to prevent the AP from flipping upside down.
      Before evaluation the raw vector ``[dir_x, dir_y, -0.5]`` is
      L2-normalised to a unit vector, so the genes encode a pure
      direction regardless of their magnitude.

    When ``optimize_orientation=False`` the chromosome falls back to the
    legacy 2-gene ``[x, y]`` encoding (position only, 8-direction sweep).

Key Principles:
    1. **No Ray imports** — this file is a pure algorithm module.
    2. **Dependency injection** via ``executor_map``.
    3. ``_format_individual(ind)`` converts a DEAP individual into
       the argument tuple expected by ``OptimizationWorker.optimize``.
    4. The injected ``map`` returns raw worker result dicts; fitness extraction
       (``result["best_metric"]``) happens here in the GA runner.

Usage::

    from reflector_position.optimizers.deap_logic import GeneticAlgorithmRunner

    ga = GeneticAlgorithmRunner(
        position_bounds={"x_min": 5, "x_max": 25, "y_min": 5, "y_max": 25},
        fixed_z=3.8,
        executor_map=executor.map,   # injected from RayActorPoolExecutor
        optimize_orientation=True,   # 4D chromosome [x, y, dir_x, dir_y]
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


def _fmt_dir_summary(direction, name=None) -> str:
    """Format a direction vector for summary text (plots / terminal)."""
    if direction is None:
        return "N/A"
    tag = f" ({name})" if name else ""
    return f"({direction[0]:+.4f}, {direction[1]:+.4f}, {direction[2]:+.4f}){tag}"


# -- Default fixed dir_z (matches gradient_descent.py) ----------------------
FIXED_DIR_Z_DEFAULT = -0.5


def _normalize_direction(
    dx: float,
    dy: float,
    dir_z: float = FIXED_DIR_Z_DEFAULT,
) -> Tuple[float, float, float]:
    """L2-normalise ``[dx, dy, dir_z]`` onto the unit sphere.

    If the raw vector is near-zero, falls back to straight-down ``(0, 0, -1)``.

    Returns:
        ``(nx, ny, nz)`` — the unit direction vector.
    """
    raw = np.array([dx, dy, dir_z], dtype=np.float64)
    norm = np.linalg.norm(raw)
    if norm < 1e-8:
        return (0.0, 0.0, -1.0)
    unit = raw / norm
    return (float(unit[0]), float(unit[1]), float(unit[2]))


def _split_mutate(individual, mu, sigma_pos, sigma_dir, indpb):
    """Gaussian mutation with different sigmas for position vs direction.

    Genes 0–1 (``x, y``) receive ``sigma_pos``; genes 2–3
    (``dir_x, dir_y``) receive ``sigma_dir``.  This allows coarse spatial
    exploration and fine-grained angular tuning simultaneously.

    Returns:
        ``(individual,)`` as required by DEAP.
    """
    sigmas = [sigma_pos, sigma_pos, sigma_dir, sigma_dir]
    for i in range(len(individual)):
        if random.random() < indpb:
            sigma = sigmas[i] if i < len(sigmas) else sigma_dir
            individual[i] += random.gauss(mu, sigma)
    return (individual,)


class GeneticAlgorithmRunner:
    """
    Pure DEAP implementation of a Genetic Algorithm for AP positioning
    and orientation.

    Decoupled from Ray via dependency injection of the ``map`` function.
    All evolutionary logic (selection, crossover, mutation, bounds clamping,
    statistics, plotting) lives here.  The expensive fitness evaluation is
    delegated to the injected ``executor_map``.

    The GA maximises the **minimum RSS** (linear Watts) across transmitters,
    which is the ``best_metric`` field returned by workers.

    When ``optimize_orientation=True`` (default) the chromosome is
    ``[x, y, dir_x, dir_y]`` (4 genes).  ``dir_z`` is fixed at
    ``FIXED_DIR_Z_DEFAULT`` and the raw direction is L2-normalised before
    evaluation so the AP always points along a unit vector.

    When ``optimize_orientation=False`` the chromosome is ``[x, y]``
    (2 genes) and the worker performs an 8-direction sweep (legacy
    behaviour).
    """

    def __init__(
        self,
        position_bounds: Dict[str, float],
        fixed_z: float,
        executor_map: Callable,
        optimize_orientation: bool = True,
        fixed_dir_z: float = FIXED_DIR_Z_DEFAULT,
    ):
        """
        Args:
            position_bounds: Search space ``{x_min, x_max, y_min, y_max}``.
            fixed_z: Fixed z-coordinate for all AP positions.
            executor_map: A ``map(func, iterable) -> list[result_dict]``
                callable.  Typically ``RayActorPoolExecutor.map``.
            optimize_orientation: If ``True`` (default), evolve a 4-gene
                chromosome ``[x, y, dir_x, dir_y]`` and pass the
                L2-normalised direction to the worker.  If ``False``,
                use the legacy 2-gene ``[x, y]`` chromosome with an
                8-direction sweep on the worker.
            fixed_dir_z: Fixed z-component of the look-at direction
                (default ``-0.5``).  Only ``dir_x`` and ``dir_y`` are
                evolved; ``dir_z`` is held constant.  Matches
                ``gradient_descent.py``.
        """
        self.bounds = position_bounds
        self.fixed_z = fixed_z
        self._executor_map = executor_map
        self.optimize_orientation = optimize_orientation
        self.fixed_dir_z = fixed_dir_z

        # Task counter for sequential IDs (useful for logging)
        self._task_counter: int = 0

        # Set during run() so _format_individual can access them
        self._opt_params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Task formatting (the "evaluate" function registered on the toolbox)
    # ------------------------------------------------------------------

    def _format_individual(self, ind) -> Tuple:
        """
        Convert a DEAP individual into the argument tuple expected by
        ``OptimizationWorker.optimize``.

        * **4-gene mode** (``optimize_orientation=True``):
          ``ind = [x, y, dir_x, dir_y]``.  The raw direction
          ``[dir_x, dir_y, fixed_dir_z]`` is L2-normalised and passed as
          ``evaluation_orientation`` so the worker evaluates exactly that
          direction (no 8-dir sweep).

        * **2-gene mode** (``optimize_orientation=False``):
          ``ind = [x, y]``.  No orientation is passed; the worker
          performs an 8-direction sweep (legacy behaviour).

        Returns:
            ``(task_id, "grid_search_point", optimizer_kwargs, optimization_params)``
        """
        task_id = self._task_counter
        self._task_counter += 1

        optimizer_kwargs: Dict[str, Any] = {
            "evaluation_position": (float(ind[0]), float(ind[1])),
            "fixed_z": self.fixed_z,
        }

        if self.optimize_orientation:
            dx, dy = float(ind[2]), float(ind[3])
            nx, ny, nz = _normalize_direction(dx, dy, self.fixed_dir_z)
            optimizer_kwargs["evaluation_orientation"] = (nx, ny, nz)

        return (
            task_id,
            "grid_search_point",
            optimizer_kwargs,
            self._opt_params,
        )

    # ------------------------------------------------------------------
    # Bounds enforcement
    # ------------------------------------------------------------------

    def _clamp_individual(self, ind) -> None:
        """Clamp an individual's genes to valid ranges (in-place).

        Position genes (0–1) are clamped to ``position_bounds``.
        Direction genes (2–3), when present, are clamped to ``[-1, 1]``.
        """
        ind[0] = max(self.bounds["x_min"], min(ind[0], self.bounds["x_max"]))
        ind[1] = max(self.bounds["y_min"], min(ind[1], self.bounds["y_max"]))
        if self.optimize_orientation and len(ind) >= 4:
            ind[2] = max(-1.0, min(ind[2], 1.0))
            ind[3] = max(-1.0, min(ind[3], 1.0))

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

            if self.optimize_orientation and len(ind) >= 4:
                # Direction is determined by the chromosome — store the
                # normalised vector so HoF copies carry the info.
                nx, ny, nz = _normalize_direction(
                    float(ind[2]), float(ind[3]), self.fixed_dir_z,
                )
                ind.best_direction = [nx, ny, nz]
                pos = np.array(res["best_position"])
                ind.best_look_at = (pos + np.array([nx, ny, nz])).tolist()
                ind.best_orientation_name = None  # continuous, not discrete
            else:
                # Legacy 2-gene mode: orientation from 8-dir sweep
                ind.best_direction = res.get("best_direction")
                ind.best_look_at = res.get("best_look_at")
                gs = res.get("grid_results", {})
                ind.best_orientation_name = gs.get("best_orientation_name")

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
                - ``mut_sigma`` (float): Legacy uniform sigma (default 2.0).
                    Used in 2-gene mode or as fallback when
                    ``mut_sigma_pos`` is not set.
                - ``mut_sigma_pos`` (float): Gaussian sigma for position
                    genes (default: ``mut_sigma``, i.e. 2.0 metres).
                - ``mut_sigma_dir`` (float): Gaussian sigma for direction
                    genes (default 0.3 unitless).  Smaller than position
                    sigma for fine angular tuning.
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
        mut_sigma_pos = ga.get("mut_sigma_pos", mut_sigma)  # position sigma
        mut_sigma_dir = ga.get("mut_sigma_dir", 0.3)        # direction sigma
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
            _mode = "4D [x,y,dx,dy]" if self.optimize_orientation else "2D [x,y]"
            print(f"DEAP GENETIC ALGORITHM ({_mode}, IoC-injected map)")
            print("=" * 80)
            print(f"  Population: {pop_size} | Generations: {n_gen}")
            print(f"  cxpb={cxpb}  mutpb={mutpb}  tournsize={tournsize}")
            if self.optimize_orientation:
                print(f"  cx_alpha={cx_alpha}  mut_sigma_pos={mut_sigma_pos}  "
                      f"mut_sigma_dir={mut_sigma_dir}  mut_indpb={mut_indpb}")
                print(f"  fixed_dir_z={self.fixed_dir_z}")
            else:
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

        if self.optimize_orientation:
            # Direction genes: random in [-1, 1]
            toolbox.register("attr_dir", random.uniform, -1.0, 1.0)
            gene_cycle = (
                toolbox.attr_x,
                toolbox.attr_y,
                toolbox.attr_dir,  # dir_x
                toolbox.attr_dir,  # dir_y
            )
        else:
            gene_cycle = (toolbox.attr_x, toolbox.attr_y)

        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            gene_cycle,
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

        if self.optimize_orientation:
            # Split mutation: coarse sigma for position, fine for direction
            toolbox.register(
                "mutate",
                _split_mutate,
                mu=mut_mu,
                sigma_pos=mut_sigma_pos,
                sigma_dir=mut_sigma_dir,
                indpb=mut_indpb,
            )
        else:
            # Legacy uniform mutation (2-gene mode)
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

        _best_dir_name = getattr(best_ind, "best_orientation_name", None)
        _best_dir = getattr(best_ind, "best_direction", None)
        generation_details.append({
            "gen": 0,
            "nevals": nevals,
            "max_dbm": record["max_dbm"],
            "mean_dbm": record["mean_dbm"],
            "std": record["std"],
            "best_x": float(best_ind[0]),
            "best_y": float(best_ind[1]),
            "best_direction": _best_dir,
            "best_orientation_name": _best_dir_name,
            "time": gen_time,
        })

        if verbose:
            _dir_tag = ""
            if _best_dir is not None:
                _dir_tag = (f" | dir=({_best_dir[0]:+.3f}, "
                            f"{_best_dir[1]:+.3f}, {_best_dir[2]:+.3f})")
            elif _best_dir_name:
                _dir_tag = f" | dir={_best_dir_name}"
            print(
                f"  Gen  0 | evals={nevals:>3d} | "
                f"best={record['max_dbm']:.2f} dBm | "
                f"mean={record['mean_dbm']:.2f} dBm | "
                f"pos=({best_ind[0]:.2f}, {best_ind[1]:.2f})"
                f"{_dir_tag} | "
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

            _best_dir_name = getattr(best_ind, "best_orientation_name", None)
            _best_dir = getattr(best_ind, "best_direction", None)
            generation_details.append({
                "gen": gen,
                "nevals": nevals,
                "max_dbm": record["max_dbm"],
                "mean_dbm": record["mean_dbm"],
                "std": record["std"],
                "best_x": float(best_ind[0]),
                "best_y": float(best_ind[1]),
                "best_direction": _best_dir,
                "best_orientation_name": _best_dir_name,
                "time": gen_time,
            })

            if verbose:
                _dir_tag = ""
                if _best_dir is not None:
                    _dir_tag = (f" | dir=({_best_dir[0]:+.3f}, "
                                f"{_best_dir[1]:+.3f}, {_best_dir[2]:+.3f})")
                elif _best_dir_name:
                    _dir_tag = f" | dir={_best_dir_name}"
                print(
                    f"  Gen {gen:>2d} | evals={nevals:>3d} | "
                    f"best={record['max_dbm']:.2f} dBm | "
                    f"mean={record['mean_dbm']:.2f} dBm | "
                    f"pos=({best_ind[0]:.2f}, {best_ind[1]:.2f})"
                    f"{_dir_tag} | "
                    f"time={gen_time:.1f}s"
                )

        total_time = time.time() - start_time

        # -- Build results dict -----------------------------------------
        best = hof[0]
        best_fitness = best.fitness.values[0]

        hall_of_fame_list = []
        for ind in hof:
            _ind_dir = getattr(ind, "best_direction", None)
            hall_of_fame_list.append({
                "position": [float(ind[0]), float(ind[1]), self.fixed_z],
                "fitness": float(ind.fitness.values[0]),
                "fitness_dbm": _rss_watts_to_dbm(ind.fitness.values[0]),
                "direction": _ind_dir,
                "look_at": getattr(ind, "best_look_at", None),
                "orientation_name": getattr(ind, "best_orientation_name", None),
                "chromosome": [float(g) for g in ind],
            })

        # Best individual chromosome as list
        _best_genes = [float(g) for g in best]

        results = {
            "best_individual": _best_genes,
            "best_fitness": float(best_fitness),
            "best_fitness_dbm": _rss_watts_to_dbm(best_fitness),
            "best_position": [float(best[0]), float(best[1]), self.fixed_z],
            "best_direction": getattr(best, "best_direction", None),
            "best_look_at": getattr(best, "best_look_at", None),
            "best_orientation_name": getattr(best, "best_orientation_name", None),
            "optimize_orientation": self.optimize_orientation,
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
                "mut_sigma_pos": mut_sigma_pos,
                "mut_sigma_dir": mut_sigma_dir,
                "mut_indpb": mut_indpb,
                "hof_size": hof_size,
                "seed": seed,
                "optimize_orientation": self.optimize_orientation,
                "fixed_dir_z": self.fixed_dir_z,
            },
            "generation_details": generation_details,
        }

        if verbose:
            print("-" * 80)
            print("GA COMPLETE")
            print(f"  Best position: ({best[0]:.2f}, {best[1]:.2f}, "
                  f"{self.fixed_z})")
            _bd = results.get("best_direction")
            _bn = results.get("best_orientation_name")
            if _bd:
                print(f"  Best direction: ({_bd[0]:+.4f}, {_bd[1]:+.4f}, {_bd[2]:+.4f})"
                      f"{f'  ({_bn})' if _bn else ''}")
            _bla = results.get("best_look_at")
            if _bla:
                print(f"  Best look_at:  ({_bla[0]:.2f}, {_bla[1]:.2f}, {_bla[2]:.2f})")
            print(f"  Best P5 RSS:  {results['best_fitness_dbm']:.2f} dBm")
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
            2. Best position trajectory over generations (with direction
               arrows when orientation is evolved)
            3. Hall-of-fame positions scatter (with direction arrows)
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

        # Extract direction vectors per generation (may be None)
        best_dirs = [g.get("best_direction") for g in gen_details]
        has_dirs = any(d is not None for d in best_dirs)
        arrow_scale = 2.0  # visual length multiplier for direction arrows

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
        ax.set_ylabel("P5 RSS (dBm)")
        ax.set_title("Fitness Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Best position trajectory (with direction arrows)
        ax = axes[0, 1]
        ax.plot(best_xs, best_ys, "b-o", markersize=4, linewidth=1.5,
                alpha=0.7)
        ax.plot(best_xs[0], best_ys[0], "go", markersize=12, label="Gen 0")
        ax.plot(best_xs[-1], best_ys[-1], "r*", markersize=15, label="Final")

        # Draw direction arrows (quiver) if orientation was evolved
        if has_dirs:
            for i, (px, py, d) in enumerate(zip(best_xs, best_ys, best_dirs)):
                if d is not None:
                    alpha = 0.3 + 0.7 * (i / max(len(best_dirs) - 1, 1))
                    ax.quiver(
                        px, py, d[0] * arrow_scale, d[1] * arrow_scale,
                        angles="xy", scale_units="xy", scale=1,
                        color="red", alpha=alpha, width=0.005,
                        headwidth=3, headlength=4,
                    )

        if bounds:
            ax.set_xlim(bounds["x_min"], bounds["x_max"])
            ax.set_ylim(bounds["y_min"], bounds["y_max"])
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Best Position Trajectory" +
                      (" + Orientation" if has_dirs else ""))
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Hall-of-fame scatter (with direction arrows)
        ax = axes[1, 0]
        hof = results["hall_of_fame"]
        hof_x = [h["position"][0] for h in hof]
        hof_y = [h["position"][1] for h in hof]
        hof_dbm = [h["fitness_dbm"] for h in hof]
        hof_dirs = [h.get("direction") for h in hof]
        sc = ax.scatter(
            hof_x, hof_y,
            c=hof_dbm, s=200, cmap="viridis", edgecolor="black",
            vmin=rss_range_dbm[0] if rss_range_dbm else None,
            vmax=rss_range_dbm[1] if rss_range_dbm else None,
        )
        # Direction arrows on HoF entries
        for i, (hx, hy, hd) in enumerate(zip(hof_x, hof_y, hof_dirs)):
            label_text = f"#{i + 1}\n{hof_dbm[i]:.1f}"
            ax.annotate(
                label_text, (hx, hy),
                textcoords="offset points", xytext=(8, 4), fontsize=7,
            )
            if hd is not None:
                ax.quiver(
                    hx, hy, hd[0] * arrow_scale, hd[1] * arrow_scale,
                    angles="xy", scale_units="xy", scale=1,
                    color="darkred", alpha=0.8, width=0.006,
                    headwidth=3, headlength=4,
                )
        if bounds:
            ax.set_xlim(bounds["x_min"], bounds["x_max"])
            ax.set_ylim(bounds["y_min"], bounds["y_max"])
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Hall of Fame (top solutions)")
        plt.colorbar(sc, ax=ax, label="P5 RSS (dBm)")
        ax.grid(True, alpha=0.3)

        # 4. Summary text
        ax = axes[1, 1]
        ax.axis("off")
        ga_p = results["ga_params"]

        # Build mutation info line based on mode
        if ga_p.get("optimize_orientation"):
            mut_line = (f"σ_pos={ga_p.get('mut_sigma_pos', ga_p.get('mut_sigma', '?'))}  "
                        f"σ_dir={ga_p.get('mut_sigma_dir', '?')}  "
                        f"indpb={ga_p['mut_indpb']}")
            mode_str = "4D [x, y, dx, dy]"
        else:
            mut_line = (f"mut_sigma={ga_p.get('mut_sigma', '?')}  "
                        f"mut_indpb={ga_p['mut_indpb']}")
            mode_str = "2D [x, y]"

        summary = (
            f"DEAP GA SUMMARY ({mode_str})\n"
            f"\n"
            f"Population: {ga_p['pop_size']} | Generations: {ga_p['n_gen']}\n"
            f"cxpb={ga_p['cxpb']}  mutpb={ga_p['mutpb']}  "
            f"tournsize={ga_p['tournsize']}\n"
            f"cx_alpha={ga_p['cx_alpha']}  {mut_line}\n"
            f"\n"
            f"Best Position: ({results['best_position'][0]:.2f}, "
            f"{results['best_position'][1]:.2f}, "
            f"{results['best_position'][2]:.2f})\n"
            f"Best Direction: {_fmt_dir_summary(results.get('best_direction'),
                                                results.get('best_orientation_name'))}\n"
            f"Best P5 RSS:  {results['best_fitness_dbm']:.2f} dBm\n"
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
