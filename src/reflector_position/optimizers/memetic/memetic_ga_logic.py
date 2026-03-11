"""Memetic GA logic with Hall of Fame and topology-aware seed extraction.

This module is an isolated Phase-1 implementation for the Memetic Fusion
pipeline (GA + local search). It intentionally does not modify the existing
baseline implementation in ``reflector_position.optimizers.deap_logic``.

Key additions over baseline GA runner:
1. Hall-of-Fame tracking across all generations (default size: 50).
2. Physics-informed distance filter to extract K spatially distinct seeds.
3. Seed-centric return payload suitable for downstream GD exploitation.

Design notes
------------
- No Ray imports here. Parallelism is injected through ``executor_map`` via
    ``toolbox.register("map", executor_map)``.
- Fitness maximization uses the worker's generic ``primary_fitness`` value,
    aligning the GA landscape with the GD objective contract.
- Distance filtering compares AP coordinates only (macro topology), ignoring
    AP orientation and reflector genes by design.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from deap import base, creator, tools

# ---------------------------------------------------------------------------
# DEAP creator types (module-safe)
# ---------------------------------------------------------------------------
if not hasattr(creator, "MemeticFitnessMax"):
    creator.create("MemeticFitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "MemeticIndividual"):
    creator.create("MemeticIndividual", list, fitness=creator.MemeticFitnessMax)


FIXED_DIR_Z_DEFAULT = -0.5
PENALTY_FITNESS_LINEAR = 1e-100
PENALTY_PRIMARY_FITNESS: float = -1e15
MIN_AP_SEPARATION_DEFAULT = 2.0
DEFAULT_HOF_SIZE = 50
DEFAULT_K_SEEDS = 5
DEFAULT_D_CORR_METERS = 5.0


@dataclass(slots=True)
class MemeticSeed:
    """Structured GA seed for downstream local search.

    Attributes
    ----------
    rank : int
        Rank in the selected seed list (1-indexed).
    primary_fitness : float
        GA fitness equal to the negative primary loss.
    ap_positions : list[tuple[float, float, float]]
        Decoded AP coordinates.
    ap_directions : list[tuple[float, float, float]] | None
        Decoded AP look directions when available.
    reflector : dict[str, float] | None
        Reflector genes (u, v, focal_x, focal_y, focal_z) when enabled.
    chromosome : list[float]
        Full raw chromosome values for reproducibility.
    loss_components : dict[str, float]
        Detached auxiliary loss components reported by the worker.
    physical_metrics : dict[str, float]
        Detached observation metrics reported by the worker.
    min_distance_to_previous : float | None
        Minimum topological distance to previously accepted seeds.
    """

    rank: int
    primary_fitness: float
    ap_positions: List[Tuple[float, float, float]]
    ap_directions: Optional[List[Tuple[float, float, float]]]
    reflector: Optional[Dict[str, float]]
    chromosome: List[float]
    loss_components: Dict[str, float]
    physical_metrics: Dict[str, float]
    min_distance_to_previous: Optional[float]


def _extract_result_primary_fitness(result: Mapping[str, Any]) -> float:
    """Extract GA fitness from the standardized evaluator payload."""
    fitness = result.get("primary_fitness")
    if fitness is not None:
        return float(fitness)
    return PENALTY_PRIMARY_FITNESS


def _extract_result_loss_components(result: Mapping[str, Any]) -> Dict[str, float]:
    """Extract detached auxiliary loss components from the evaluator payload."""
    loss_components = result.get("loss_components")
    if isinstance(loss_components, Mapping):
        return {
            str(name): float(value)
            for name, value in loss_components.items()
            if value is not None
        }
    return {}


def _extract_result_physical_metrics(result: Mapping[str, Any]) -> Dict[str, float]:
    """Extract detached observation metrics from the evaluator payload."""
    physical_metrics = result.get("physical_metrics")
    if isinstance(physical_metrics, Mapping):
        return {
            str(name): float(value)
            for name, value in physical_metrics.items()
            if value is not None
        }
    return {}


def _compile_generation_stats(population: Sequence[Any]) -> Dict[str, float | int]:
    """Compile GA generation stats while excluding infeasible penalties from means.

    Infeasible individuals keep the large negative penalty fitness so selection
    pressure remains intact, but reporting means over those values is not
    informative. This helper reports the best score over the full population
    while computing mean/std/min over feasible individuals only.
    """
    all_values = [
        float(ind.fitness.values[0])
        for ind in population
        if getattr(ind.fitness, "valid", False)
    ]
    feasible_values = [
        float(ind.fitness.values[0])
        for ind in population
        if getattr(ind.fitness, "valid", False) and not getattr(ind, "penalized", False)
    ]

    if not all_values:
        raise RuntimeError("Population contains no valid fitness values.")

    stat_values = feasible_values if feasible_values else all_values
    penalized_count = max(0, len(all_values) - len(feasible_values))
    return {
        "max": float(np.max(all_values)),
        "mean": float(np.mean(stat_values)),
        "std": float(np.std(stat_values)),
        "min": float(np.min(stat_values)),
        "feasible_count": int(len(feasible_values)),
        "penalized_count": int(penalized_count),
        "mean_population": float(np.mean(all_values)),
    }


def _normalize_direction(
    dx: float,
    dy: float,
    dir_z: float = FIXED_DIR_Z_DEFAULT,
) -> Tuple[float, float, float]:
    """L2-normalize ``[dx, dy, dir_z]``; fallback to ``(0,0,-1)``."""
    raw = np.array([dx, dy, dir_z], dtype=np.float64)
    norm = np.linalg.norm(raw)
    if norm < 1e-8:
        return (0.0, 0.0, -1.0)
    unit = raw / norm
    return (float(unit[0]), float(unit[1]), float(unit[2]))


def _split_mutate(
    individual: Sequence[float],
    mu: float,
    sigma_pos: float,
    sigma_dir: float,
    indpb: float,
    num_pos_genes: int = 2,
    sigma_reflector: float = 0.1,
    reflector_gene_start: int = -1,
) -> Tuple[Sequence[float]]:
    """Gaussian mutation with region-specific sigmas.

    Position genes use ``sigma_pos``.
    Direction genes use ``sigma_dir``.
    Reflector genes (if present) use ``sigma_reflector``.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            if i < num_pos_genes:
                sigma = sigma_pos
            elif reflector_gene_start >= 0 and i >= reflector_gene_start:
                sigma = sigma_reflector
            else:
                sigma = sigma_dir
            individual[i] += random.gauss(mu, sigma)
    return (individual,)


class MemeticGeneticAlgorithmRunner:
    """DEAP GA runner with Hall-of-Fame and spatially distinct seed extraction.

    This class mirrors the baseline GA workflow while adding a memetic-oriented
    post-processing stage that extracts topologically diverse seeds.
    """

    def __init__(
        self,
        position_bounds: Dict[str, float],
        fixed_z: float,
        executor_map: Callable[[Callable, Iterable], List[Dict[str, Any]]],
        optimize_orientation: bool = True,
        fixed_dir_z: float = FIXED_DIR_Z_DEFAULT,
        num_aps: int = 1,
        min_ap_separation: float = MIN_AP_SEPARATION_DEFAULT,
        reflector_enabled: bool = False,
        focal_bounds: Optional[Dict[str, float]] = None,
        focal_z: float = 1.5,
        percentile_target_quantile: float = 0.05,
        k_seeds: int = DEFAULT_K_SEEDS,
        d_corr: float = DEFAULT_D_CORR_METERS,
    ) -> None:
        self.bounds = position_bounds
        self.fixed_z = fixed_z
        self._executor_map = executor_map
        self.optimize_orientation = optimize_orientation
        self.fixed_dir_z = fixed_dir_z
        self.num_aps = int(num_aps)
        self.min_ap_separation = float(min_ap_separation)

        self.reflector_enabled = reflector_enabled
        self.focal_z = float(focal_z)
        self.percentile_target_quantile = float(percentile_target_quantile)

        self.default_k_seeds = int(k_seeds)
        self.default_d_corr = float(d_corr)

        if self.num_aps < 1:
            raise ValueError("num_aps must be >= 1")

        if focal_bounds is not None:
            self.focal_bounds = focal_bounds
        else:
            self.focal_bounds = {
                "fx_min": position_bounds["x_min"],
                "fx_max": position_bounds["x_max"],
                "fy_min": position_bounds["y_min"],
                "fy_max": position_bounds["y_max"],
            }

        self._n_pos_genes = 2 * self.num_aps
        self._n_dir_genes = 2 * self.num_aps if self.optimize_orientation else 0
        self._n_reflector_genes = 4 if self.reflector_enabled else 0
        self._n_genes = self._n_pos_genes + self._n_dir_genes + self._n_reflector_genes

        self._reflector_gene_start = (
            self._n_pos_genes + self._n_dir_genes if self.reflector_enabled else -1
        )

        self._task_counter: int = 0
        self._opt_params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Constraints and decoding
    # ------------------------------------------------------------------

    def _check_separation(self, individual: Sequence[float]) -> bool:
        """Check minimum pairwise AP distance constraint."""
        if self.num_aps < 2:
            return True

        positions = self._extract_ap_xy(individual)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                if float(np.hypot(dx, dy)) < self.min_ap_separation:
                    return False
        return True

    def _extract_ap_xy(self, individual: Sequence[float]) -> np.ndarray:
        """Return AP XY coordinates as ``[num_aps, 2]`` numpy array."""
        arr = np.zeros((self.num_aps, 2), dtype=np.float64)
        for ap_idx in range(self.num_aps):
            arr[ap_idx, 0] = float(individual[2 * ap_idx])
            arr[ap_idx, 1] = float(individual[2 * ap_idx + 1])
        return arr

    def _extract_positions(self, individual: Sequence[float]) -> List[Tuple[float, float, float]]:
        """Decode AP XYZ positions."""
        return [
            (
                float(individual[2 * ap_idx]),
                float(individual[2 * ap_idx + 1]),
                self.fixed_z,
            )
            for ap_idx in range(self.num_aps)
        ]

    def _extract_directions(
        self,
        individual: Sequence[float],
    ) -> Optional[List[Tuple[float, float, float]]]:
        """Decode AP directions when orientation optimization is enabled."""
        if not self.optimize_orientation:
            return None

        directions: List[Tuple[float, float, float]] = []
        dir_base = self._n_pos_genes
        for ap_idx in range(self.num_aps):
            dx = float(individual[dir_base + 2 * ap_idx])
            dy = float(individual[dir_base + 2 * ap_idx + 1])
            directions.append(_normalize_direction(dx, dy, self.fixed_dir_z))
        return directions

    def _extract_reflector(self, individual: Sequence[float]) -> Optional[Dict[str, float]]:
        """Decode reflector genes when enabled."""
        if not self.reflector_enabled:
            return None

        rg = self._reflector_gene_start
        return {
            "u": float(individual[rg]),
            "v": float(individual[rg + 1]),
            "focal_x": float(individual[rg + 2]),
            "focal_y": float(individual[rg + 3]),
            "focal_z": self.focal_z,
        }

    # ------------------------------------------------------------------
    # Topological distance filtering
    # ------------------------------------------------------------------

    def _topological_distance(
        self,
        individual_a: Sequence[float],
        individual_b: Sequence[float],
    ) -> float:
        """Compute AP-topology distance between two individuals in meters.

        For 1-AP: Euclidean distance in XY.
        For 2-AP: minimum assignment distance between direct and swapped AP IDs.
        For N>2: direct index-wise RMS distance (current pipeline targets 1/2 AP).
        """
        pos_a = self._extract_ap_xy(individual_a)
        pos_b = self._extract_ap_xy(individual_b)

        if self.num_aps == 1:
            return float(np.linalg.norm(pos_a[0] - pos_b[0]))

        if self.num_aps == 2:
            d_direct = float(np.sqrt(np.sum((pos_a - pos_b) ** 2)))
            d_swap = float(np.sqrt(np.sum((pos_a - pos_b[[1, 0], :]) ** 2)))
            return min(d_direct, d_swap)

        return float(np.sqrt(np.mean(np.sum((pos_a - pos_b) ** 2, axis=1))))

    def _extract_topological_seeds(
        self,
        hof: tools.HallOfFame,
        k_seeds: int,
        d_corr: float,
    ) -> List[MemeticSeed]:
        """Select top-K spatially distinct seeds from Hall of Fame.

        Selection rule:
        1. Always accept HoF[0].
        2. Iterate remaining HoF entries in fitness order.
        3. Accept candidate iff distance to every accepted seed is >= d_corr.
        """
        if k_seeds <= 0:
            return []

        selected: List[MemeticSeed] = []
        selected_individuals: List[Sequence[float]] = []

        for candidate in hof:
            if not selected_individuals:
                min_dist = None
                accept = True
            else:
                distances = [
                    self._topological_distance(candidate, accepted)
                    for accepted in selected_individuals
                ]
                min_dist = float(min(distances)) if distances else None
                accept = min_dist is None or min_dist >= d_corr

            if not accept:
                continue

            seed = MemeticSeed(
                rank=len(selected) + 1,
                primary_fitness=float(candidate.fitness.values[0]),
                ap_positions=self._extract_positions(candidate),
                ap_directions=self._extract_directions(candidate),
                reflector=self._extract_reflector(candidate),
                chromosome=[float(g) for g in candidate],
                loss_components=dict(getattr(candidate, "loss_components", {})),
                physical_metrics=dict(getattr(candidate, "physical_metrics", {})),
                min_distance_to_previous=min_dist,
            )
            selected.append(seed)
            selected_individuals.append(candidate)

            if len(selected) >= k_seeds:
                break

        return selected

    # ------------------------------------------------------------------
    # Worker formatting and evaluation
    # ------------------------------------------------------------------

    def _format_individual(self, individual: Sequence[float]) -> Tuple[int, str, Dict[str, Any], Dict[str, Any]]:
        """Format a DEAP individual into worker task arguments."""
        task_id = self._task_counter
        self._task_counter += 1

        task_kwargs: Dict[str, Any] = {
            "initial_positions": [
                (
                    float(individual[2 * ap_idx]),
                    float(individual[2 * ap_idx + 1]),
                )
                for ap_idx in range(self.num_aps)
            ],
            "fixed_z": self.fixed_z,
            "samples_per_tx": int(self._opt_params.get("samples_per_tx", 1_000_000)),
            "max_depth": int(self._opt_params.get("max_depth", 13)),
            "loss_kwargs": {
                "alpha": float(self._opt_params.get("alpha", 0.6)),
                "beta": float(self._opt_params.get("beta", 0.4)),
                "softmin_temperature": float(
                    self._opt_params.get(
                        "softmin_temperature",
                        self._opt_params.get("temperature", 0.1),
                    )
                ),
                "coverage_threshold_dbm": float(
                    self._opt_params.get("coverage_threshold_dbm", -100.0)
                ),
                "coverage_temperature": float(
                    self._opt_params.get("coverage_temperature", 2.0)
                ),
            },
        }

        if self.optimize_orientation:
            dir_base = self._n_pos_genes
            task_kwargs["initial_directions_xy"] = [
                _normalize_direction(
                    float(individual[dir_base + 2 * ap_idx]),
                    float(individual[dir_base + 2 * ap_idx + 1]),
                    self.fixed_dir_z,
                )[:2]
                for ap_idx in range(self.num_aps)
            ]

        if self.reflector_enabled:
            rg = self._reflector_gene_start
            task_kwargs["reflector_u"] = float(individual[rg])
            task_kwargs["reflector_v"] = float(individual[rg + 1])
            task_kwargs["reflector_target"] = (
                float(individual[rg + 2]),
                float(individual[rg + 3]),
                self.focal_z,
            )

        return (task_id, "memetic_eval", task_kwargs, {})

    def _clamp_individual(self, individual: Sequence[float]) -> None:
        """Clamp chromosome values in-place to valid bounds."""
        for ap_idx in range(self.num_aps):
            individual[2 * ap_idx] = max(
                self.bounds["x_min"],
                min(individual[2 * ap_idx], self.bounds["x_max"]),
            )
            individual[2 * ap_idx + 1] = max(
                self.bounds["y_min"],
                min(individual[2 * ap_idx + 1], self.bounds["y_max"]),
            )

        if self.optimize_orientation:
            dir_end = self._n_pos_genes + self._n_dir_genes
            for i in range(self._n_pos_genes, dir_end):
                individual[i] = max(-1.0, min(individual[i], 1.0))

        if self.reflector_enabled:
            rg = self._reflector_gene_start
            individual[rg] = max(0.0, min(individual[rg], 1.0))
            individual[rg + 1] = max(0.0, min(individual[rg + 1], 1.0))
            individual[rg + 2] = max(
                self.focal_bounds["fx_min"],
                min(individual[rg + 2], self.focal_bounds["fx_max"]),
            )
            individual[rg + 3] = max(
                self.focal_bounds["fy_min"],
                min(individual[rg + 3], self.focal_bounds["fy_max"]),
            )

    def _evaluate_invalid(self, population: List[Any], toolbox: base.Toolbox) -> int:
        """Evaluate only individuals with invalid fitness."""
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        if not invalid_ind:
            return 0

        if self.num_aps >= 2:
            valid_inds: List[Any] = []
            penalized_inds: List[Any] = []
            for ind in invalid_ind:
                if self._check_separation(ind):
                    valid_inds.append(ind)
                else:
                    penalized_inds.append(ind)

            for ind in penalized_inds:
                ind.fitness.values = (PENALTY_PRIMARY_FITNESS,)
                ind.penalized = True
                ind.loss_components = {}
                ind.physical_metrics = {}
        else:
            valid_inds = invalid_ind

        if valid_inds:
            results = toolbox.map(toolbox.evaluate, valid_inds)
            for ind, res in zip(valid_inds, results):
                # Use the standardized primary fitness so the GA follows the
                # same objective contract as the downstream local search.
                ind.fitness.values = (
                    _extract_result_primary_fitness(res),
                )
                ind.penalized = False
                ind.loss_components = _extract_result_loss_components(res)
                ind.physical_metrics = _extract_result_physical_metrics(res)

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
        k_seeds: Optional[int] = None,
        d_corr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run GA and return Hall-of-Fame plus diverse topological seeds.

        Parameters
        ----------
        optimization_params : dict, optional
            Forwarded to worker optimizer calls.
        ga_params : dict, optional
            DEAP hyperparameters:
            ``pop_size``, ``n_gen``, ``cxpb``, ``mutpb``, ``tournsize``,
            ``cx_alpha``, ``mut_mu``, ``mut_sigma``, ``mut_sigma_pos``,
            ``mut_sigma_dir``, ``mut_sigma_reflector``, ``mut_indpb``,
            ``hof_size``.
        seed : int, optional
            RNG seed for reproducibility.
        verbose : bool
            Print per-generation progress if True.
        k_seeds : int, optional
            Number of spatially distinct seeds to extract from HoF.
        d_corr : float, optional
            Topological correlation distance threshold in meters.

        Returns
        -------
        dict
            Seed-centric payload with HoF archive and GA diagnostics.
        """
        ga = ga_params or {}

        pop_size = int(ga.get("pop_size", 50))
        n_gen = int(ga.get("n_gen", 20))
        cxpb = float(ga.get("cxpb", 0.7))
        mutpb = float(ga.get("mutpb", 0.3))
        tournsize = int(ga.get("tournsize", 3))
        cx_alpha = float(ga.get("cx_alpha", 0.5))
        mut_mu = float(ga.get("mut_mu", 0.0))
        mut_sigma = float(ga.get("mut_sigma", 2.0))
        mut_sigma_pos = float(ga.get("mut_sigma_pos", mut_sigma))
        mut_sigma_dir = float(ga.get("mut_sigma_dir", 0.3))
        mut_sigma_reflector = float(ga.get("mut_sigma_reflector", 0.1))
        mut_indpb = float(ga.get("mut_indpb", 0.2))
        hof_size = int(ga.get("hof_size", DEFAULT_HOF_SIZE))

        eff_k_seeds = int(self.default_k_seeds if k_seeds is None else k_seeds)
        eff_d_corr = float(self.default_d_corr if d_corr is None else d_corr)

        self._opt_params = optimization_params or {
            "samples_per_tx": 1_000_000,
            "max_depth": 13,
            "verbose": False,
        }
        self._opt_params.setdefault("verbose", False)
        self._task_counter = 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if verbose:
            print("=" * 80)
            print("MEMETIC GA (HoF + Topological Seed Filter)")
            print("=" * 80)
            print(
                f"  Population={pop_size} | Generations={n_gen} | "
                f"APs={self.num_aps} | HoF size={hof_size}"
            )
            print(f"  k_seeds={eff_k_seeds} | d_corr={eff_d_corr:.2f}m")
            print("-" * 80)

        start_time = time.time()

        toolbox = base.Toolbox()
        toolbox.register("attr_x", random.uniform, self.bounds["x_min"], self.bounds["x_max"])
        toolbox.register("attr_y", random.uniform, self.bounds["y_min"], self.bounds["y_max"])

        pos_genes = (toolbox.attr_x, toolbox.attr_y) * self.num_aps
        if self.optimize_orientation:
            toolbox.register("attr_dir", random.uniform, -1.0, 1.0)
            gene_cycle = pos_genes + (toolbox.attr_dir,) * self._n_dir_genes
        else:
            gene_cycle = pos_genes

        if self.reflector_enabled:
            toolbox.register("attr_refl_uv", random.uniform, 0.0, 1.0)
            toolbox.register(
                "attr_focal_x",
                random.uniform,
                self.focal_bounds["fx_min"],
                self.focal_bounds["fx_max"],
            )
            toolbox.register(
                "attr_focal_y",
                random.uniform,
                self.focal_bounds["fy_min"],
                self.focal_bounds["fy_max"],
            )
            gene_cycle = gene_cycle + (
                toolbox.attr_refl_uv,
                toolbox.attr_refl_uv,
                toolbox.attr_focal_x,
                toolbox.attr_focal_y,
            )

        toolbox.register("individual", tools.initCycle, creator.MemeticIndividual, gene_cycle, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxBlend, alpha=cx_alpha)

        if self.optimize_orientation or self.reflector_enabled:
            toolbox.register(
                "mutate",
                _split_mutate,
                mu=mut_mu,
                sigma_pos=mut_sigma_pos,
                sigma_dir=mut_sigma_dir,
                indpb=mut_indpb,
                num_pos_genes=self._n_pos_genes,
                sigma_reflector=mut_sigma_reflector,
                reflector_gene_start=self._reflector_gene_start,
            )
        else:
            toolbox.register("mutate", tools.mutGaussian, mu=mut_mu, sigma=mut_sigma, indpb=mut_indpb)

        toolbox.register("select", tools.selTournament, tournsize=tournsize)
        toolbox.register("map", self._executor_map)
        toolbox.register("evaluate", self._format_individual)

        logbook = tools.Logbook()
        logbook.header = [
            "gen",
            "nevals",
            "max",
            "mean",
            "std",
            "min",
            "feasible_count",
            "penalized_count",
        ]

        hof = tools.HallOfFame(maxsize=hof_size)

        population = toolbox.population(n=pop_size)
        total_evaluations = 0
        generation_details: List[Dict[str, Any]] = []

        # Initial evaluation of the randomly generated population
        
        nevals = self._evaluate_invalid(population, toolbox)
        total_evaluations += nevals
        hof.update(population)

        record = _compile_generation_stats(population)
        logbook.record(gen=0, nevals=nevals, **record)
        generation_details.append(
            {
                "gen": 0,
                "nevals": nevals,
                "best_primary_fitness": float(record["max"]),
                "mean_primary_fitness": float(record["mean"]),
                "std": float(record["std"]),
                "feasible_count": int(record["feasible_count"]),
                "penalized_count": int(record["penalized_count"]),
                "mean_population_fitness": float(record["mean_population"]),
            }
        )

        if verbose:
            print(
                f"  Gen  0 | evals={nevals:>3d} | best={record['max']:.6f} | "
                f"mean={record['mean']:.6f} | penalized={record['penalized_count']}"
            )

        for gen in range(1, n_gen + 1):
            gen_start = time.time()

            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            for individual in offspring:
                self._clamp_individual(individual)

            nevals = self._evaluate_invalid(offspring, toolbox)
            total_evaluations += nevals
            population[:] = offspring
            hof.update(population)

            record = _compile_generation_stats(population)
            logbook.record(gen=gen, nevals=nevals, **record)

            gen_time = time.time() - gen_start
            generation_details.append(
                {
                    "gen": gen,
                    "nevals": nevals,
                    "best_primary_fitness": float(record["max"]),
                    "mean_primary_fitness": float(record["mean"]),
                    "std": float(record["std"]),
                    "feasible_count": int(record["feasible_count"]),
                    "penalized_count": int(record["penalized_count"]),
                    "mean_population_fitness": float(record["mean_population"]),
                    "time": gen_time,
                }
            )

            if verbose:
                print(
                    f"  Gen {gen:>2d} | evals={nevals:>3d} | best={record['max']:.6f} | "
                    f"mean={record['mean']:.6f} | penalized={record['penalized_count']} | "
                    f"time={gen_time:.1f}s"
                )

        total_time = time.time() - start_time

        if len(hof) == 0:
            raise RuntimeError("Hall of Fame is empty. No feasible individuals evaluated.")

        best = hof[0]
        selected_seeds = self._extract_topological_seeds(
            hof=hof,
            k_seeds=eff_k_seeds,
            d_corr=eff_d_corr,
        )

        hall_of_fame = []
        for i, ind in enumerate(hof, start=1):
            hall_of_fame.append(
                {
                    "rank": i,
                    "primary_fitness": float(ind.fitness.values[0]),
                    "loss_components": dict(getattr(ind, "loss_components", {})),
                    "physical_metrics": dict(getattr(ind, "physical_metrics", {})),
                    "ap_positions": self._extract_positions(ind),
                    "ap_directions": self._extract_directions(ind),
                    "reflector": self._extract_reflector(ind),
                    "chromosome": [float(g) for g in ind],
                }
            )

        results: Dict[str, Any] = {
            "num_aps": self.num_aps,
            "optimize_orientation": self.optimize_orientation,
            "reflector_enabled": self.reflector_enabled,
            "best_primary_fitness": float(best.fitness.values[0]),
            "best_individual": [float(g) for g in best],
            "best_loss_components": dict(getattr(best, "loss_components", {})),
            "best_physical_metrics": dict(getattr(best, "physical_metrics", {})),
            "seeds": [asdict(seed) for seed in selected_seeds],
            "num_selected_seeds": len(selected_seeds),
            "seed_extraction": {
                "k_requested": eff_k_seeds,
                "d_corr": eff_d_corr,
                "hof_size": len(hof),
            },
            "hall_of_fame": hall_of_fame,
            "logbook": logbook,
            "generation_details": generation_details,
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
                "mut_sigma_reflector": mut_sigma_reflector,
                "mut_indpb": mut_indpb,
                "hof_size": hof_size,
                "seed": seed,
                "k_seeds": eff_k_seeds,
                "d_corr": eff_d_corr,
            },
        }

        if verbose:
            print("-" * 80)
            print("MEMETIC GA COMPLETE")
            print(f"  Best primary fitness: {results['best_primary_fitness']:.6f}")
            if results["best_loss_components"]:
                print("  Best loss components:")
                for name, value in results["best_loss_components"].items():
                    print(f"    {name}: {value:.6f}")
            if results["best_physical_metrics"]:
                print("  Best physical metrics:")
                for name, value in results["best_physical_metrics"].items():
                    print(f"    {name}: {value:.6f}")
            print(f"  Selected seeds: {results['num_selected_seeds']} / {eff_k_seeds}")
            print(f"  Total evals: {total_evaluations}")
            print(f"  Wall-clock: {total_time:.2f}s")
            print("=" * 80)

        return results
