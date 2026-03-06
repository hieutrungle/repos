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
    **1-AP mode** (``num_aps=1``, default):
      4-gene chromosome ``[x, y, dir_x, dir_y]`` when
      ``optimize_orientation=True``, or 2-gene ``[x, y]`` when
      ``optimize_orientation=False`` (legacy mode).

    **2-AP mode** (``num_aps=2``):
      8-gene chromosome ``[x1, y1, x2, y2, dir1_x, dir1_y, dir2_x, dir2_y]``
      when ``optimize_orientation=True``.
      Genes 0-3 are positions, genes 4-7 are look-at directions.
      Without orientation: 4-gene ``[x1, y1, x2, y2]``.

      A **separation constraint** penalises solutions where the two APs
      are closer than ``min_ap_separation`` metres, saving expensive
      ray-tracing by returning a penalty fitness immediately.

    **Reflector extension** (``reflector_enabled=True``):
      When a passive reflector is present, 4 additional genes are
      appended *after* the AP direction genes:

        ``[..., reflector_u, reflector_v, focal_x, focal_y]``

      - ``reflector_u`` ∈ [0, 1]: lateral wall-surface coordinate.
      - ``reflector_v`` ∈ [0, 1]: vertical wall-surface coordinate.
      - ``focal_x``, ``focal_y``: horizontal components of the 3-D
        focal point the reflector should aim at.

      The focal point's *z* component is fixed at the receiver height
      (``focal_z``) so the GA only searches in 2-D focal space.

    Direction genes encode the horizontal components of the look-at vector.
    The vertical component ``dir_z`` is fixed at ``-0.5`` (downward bias).
    Before evaluation the raw vector ``[dir_x, dir_y, -0.5]`` is
    L2-normalised to a unit vector.

Key Principles:
    1. **No Ray imports** — this file is a pure algorithm module.
    2. **Dependency injection** via ``executor_map``.
    3. ``_format_individual(ind)`` converts a DEAP individual into
       the argument tuple expected by ``OptimizationWorker.optimize``.
       When reflector genes are present the formatted kwargs include
       ``reflector_u``, ``reflector_v``, ``reflector_target``, and
       ``percentile_target_quantile`` so the worker configures the
       reflector and uses the shadow-robust objective automatically.
    4. The injected ``map`` returns raw worker result dicts; fitness extraction
       (``result["best_metric"]``) happens here in the GA runner.

Usage::

    from reflector_position.optimizers.deap_logic import GeneticAlgorithmRunner

    # 1-AP (default)
    ga = GeneticAlgorithmRunner(
        position_bounds={"x_min": 5, "x_max": 25, "y_min": 5, "y_max": 25},
        fixed_z=3.8,
        executor_map=executor.map,
        optimize_orientation=True,
    )

    # 2-AP with separation constraint
    ga = GeneticAlgorithmRunner(
        position_bounds={"x_min": 5, "x_max": 25, "y_min": 5, "y_max": 25},
        fixed_z=3.8,
        executor_map=executor.map,
        optimize_orientation=True,
        num_aps=2,
        min_ap_separation=2.0,
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

# -- Penalty fitness for constraint violations (≈ −970 dBm) -----------------
PENALTY_FITNESS_LINEAR = 1e-100

# -- Default minimum separation between APs (metres) ------------------------
MIN_AP_SEPARATION_DEFAULT = 2.0


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


def _split_mutate(
    individual,
    mu,
    sigma_pos,
    sigma_dir,
    indpb,
    num_pos_genes=2,
    sigma_reflector=0.1,
    reflector_gene_start=-1,
):
    """Gaussian mutation with different sigmas for position, direction, and
    reflector genes.

    Genes ``0`` to ``num_pos_genes - 1`` receive ``sigma_pos``;
    genes from ``num_pos_genes`` to ``reflector_gene_start - 1`` receive
    ``sigma_dir`` (AP orientation); genes from ``reflector_gene_start``
    onward receive ``sigma_reflector`` (reflector wall-surface *u*/*v*
    and focal-point coordinates).

    When ``reflector_gene_start < 0`` (default) no reflector region
    exists and the function behaves exactly as before.

    For 1-AP: ``num_pos_genes=2`` → genes 0–1 position, genes 2–3 direction.
    For 2-AP: ``num_pos_genes=4`` → genes 0–3 position, genes 4–7 direction.

    Returns:
        ``(individual,)`` as required by DEAP.
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


class GeneticAlgorithmRunner:
    """
    Pure DEAP implementation of a Genetic Algorithm for AP positioning
    and orientation.

    Decoupled from Ray via dependency injection of the ``map`` function.
    All evolutionary logic (selection, crossover, mutation, bounds clamping,
    statistics, plotting) lives here.  The expensive fitness evaluation is
    delegated to the injected ``executor_map``.

    The GA maximises the **5th-percentile RSS** (linear Watts) across transmitters,
    which is the ``best_metric`` field returned by workers.

    Supports ``num_aps=1`` (default) and ``num_aps=2``:

    - **1-AP with orientation:** 4-gene chromosome ``[x, y, dx, dy]``.
    - **1-AP without orientation:** 2-gene chromosome ``[x, y]``.
    - **2-AP with orientation:** 8-gene chromosome
      ``[x1, y1, x2, y2, dx1, dy1, dx2, dy2]``.
    - **2-AP without orientation:** 4-gene chromosome
      ``[x1, y1, x2, y2]``.

    When ``num_aps >= 2``, a **separation constraint** is enforced: any
    individual whose APs are closer than ``min_ap_separation`` metres
    receives a penalty fitness and is *not* evaluated via ray tracing,
    saving compute.

    When ``reflector_enabled=True``, 4 continuous reflector genes are
    appended after the AP genes:

    - ``reflector_u`` ∈ [0, 1]: lateral wall-surface coordinate.
    - ``reflector_v`` ∈ [0, 1]: vertical wall-surface coordinate.
    - ``focal_x``, ``focal_y``: horizontal focal-point components
      (bounded by ``focal_bounds``).

    The formatted evaluation kwargs include the reflector parameters and
    the ``percentile_target_quantile`` so the worker automatically
    constructs a :class:`PercentileCoverageObjective` and configures the
    reflector geometry before each ray-trace.
    """

    def __init__(
        self,
        position_bounds: Dict[str, float],
        fixed_z: float,
        executor_map: Callable,
        optimize_orientation: bool = True,
        fixed_dir_z: float = FIXED_DIR_Z_DEFAULT,
        num_aps: int = 1,
        min_ap_separation: float = MIN_AP_SEPARATION_DEFAULT,
        # ---- Reflector parameters (optional) ----
        reflector_enabled: bool = False,
        focal_bounds: Optional[Dict[str, float]] = None,
        focal_z: float = 1.5,
        percentile_target_quantile: float = 0.05,
    ):
        """
        Args:
            position_bounds: Search space ``{x_min, x_max, y_min, y_max}``.
            fixed_z: Fixed z-coordinate for all AP positions.
            executor_map: A ``map(func, iterable) -> list[result_dict]``
                callable.  Typically ``RayActorPoolExecutor.map``.
            optimize_orientation: If ``True`` (default), evolve direction
                genes alongside position genes.  If ``False``, use
                position-only chromosome with 8-direction sweep on worker.
            fixed_dir_z: Fixed z-component of the look-at direction
                (default ``-0.5``).
            num_aps: Number of Access Points to optimise (1 or 2).
            min_ap_separation: Minimum allowed Euclidean distance (metres)
                between APs.  Individuals violating this constraint receive
                a penalty fitness.  Only used when ``num_aps >= 2``.
            reflector_enabled: If ``True``, append 4 reflector genes
                (``u``, ``v``, ``focal_x``, ``focal_y``) to every
                individual.
            focal_bounds: Search-space bounds for the focal-point genes:
                ``{fx_min, fx_max, fy_min, fy_max}``.  Defaults to the
                same spatial bounds as AP positions when ``None``.
            focal_z: Fixed z-coordinate for the reflector focal point
                (typically the receiver height).
            percentile_target_quantile: Quantile passed to the
                ``PercentileCoverageObjective`` on each worker.  Default
                0.05 (5th percentile).  Must exceed the reflector shadow
                fraction.
        """
        self.bounds = position_bounds
        self.fixed_z = fixed_z
        self._executor_map = executor_map
        self.optimize_orientation = optimize_orientation
        self.fixed_dir_z = fixed_dir_z
        self.num_aps = num_aps
        self.min_ap_separation = min_ap_separation

        # ---- Reflector configuration ----
        self.reflector_enabled = reflector_enabled
        self.focal_z = focal_z
        self.percentile_target_quantile = percentile_target_quantile

        if focal_bounds is not None:
            self.focal_bounds = focal_bounds
        else:
            # Default: same spatial extent as AP position bounds
            self.focal_bounds = {
                "fx_min": position_bounds["x_min"],
                "fx_max": position_bounds["x_max"],
                "fy_min": position_bounds["y_min"],
                "fy_max": position_bounds["y_max"],
            }

        # Derived chromosome dimensions
        self._n_pos_genes = 2 * num_aps           # x, y per AP
        if optimize_orientation:
            self._n_dir_genes = 2 * num_aps       # dx, dy per AP
        else:
            self._n_dir_genes = 0
        self._n_reflector_genes = 4 if reflector_enabled else 0  # u, v, fx, fy
        self._n_genes = (
            self._n_pos_genes + self._n_dir_genes + self._n_reflector_genes
        )

        # Index where reflector genes start in the chromosome
        self._reflector_gene_start = (
            self._n_pos_genes + self._n_dir_genes
            if reflector_enabled else -1
        )

        # Task counter for sequential IDs (useful for logging)
        self._task_counter: int = 0

        # Set during run() so _format_individual can access them
        self._opt_params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Separation constraint
    # ------------------------------------------------------------------

    def _check_separation(self, ind) -> bool:
        """Return ``True`` if all AP pairs satisfy ``min_ap_separation``.

        Only meaningful when ``num_aps >= 2``.  For ``num_aps == 1``
        always returns ``True``.
        """
        if self.num_aps < 2:
            return True
        # Extract (x, y) for each AP  —  genes [0 .. 2*num_aps)
        positions = []
        for ap_idx in range(self.num_aps):
            x = float(ind[2 * ap_idx])
            y = float(ind[2 * ap_idx + 1])
            positions.append((x, y))
        # Check every pair
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < self.min_ap_separation:
                    return False
        return True

    @staticmethod
    def _ap_distance(ind, num_aps: int) -> float:
        """Euclidean distance between AP0 and AP1 (2-AP shortcut)."""
        if num_aps < 2:
            return float("inf")
        dx = float(ind[0]) - float(ind[2])
        dy = float(ind[1]) - float(ind[3])
        return (dx * dx + dy * dy) ** 0.5

    # ------------------------------------------------------------------
    # Task formatting (the "evaluate" function registered on the toolbox)
    # ------------------------------------------------------------------

    def _format_individual(self, ind) -> Tuple:
        """
        Convert a DEAP individual into the argument tuple expected by
        ``OptimizationWorker.optimize``.

        **1-AP, 4-gene** (``optimize_orientation=True``):
          ``ind = [x, y, dir_x, dir_y]``.  Uses legacy single-AP kwargs
          (``evaluation_position``, ``evaluation_orientation``).

        **1-AP, 2-gene** (``optimize_orientation=False``):
          ``ind = [x, y]``.  No orientation passed; worker does 8-dir sweep.

        **2-AP, 8-gene** (``optimize_orientation=True``):
          ``ind = [x1, y1, x2, y2, dx1, dy1, dx2, dy2]``.
          Uses multi-AP kwargs (``evaluation_positions``,
          ``evaluation_orientations``).

        **2-AP, 4-gene** (``optimize_orientation=False``):
          ``ind = [x1, y1, x2, y2]``.  Multi-AP positions, worker sweeps
          8 directions per AP.

        **Reflector extension:**
          When ``reflector_enabled=True`` the last 4 genes encode
          ``[reflector_u, reflector_v, focal_x, focal_y]``.  These are
          injected into ``optimizer_kwargs`` as ``reflector_u``,
          ``reflector_v``, ``reflector_target``, and
          ``percentile_target_quantile`` so the worker automatically
          configures the reflector and uses the shadow-robust objective.

        Returns:
            ``(task_id, "grid_search_point", optimizer_kwargs, optimization_params)``
        """
        task_id = self._task_counter
        self._task_counter += 1

        if self.num_aps == 1:
            # ---- Single-AP (legacy interface) -------------------------
            optimizer_kwargs: Dict[str, Any] = {
                "evaluation_position": (float(ind[0]), float(ind[1])),
                "fixed_z": self.fixed_z,
            }
            if self.optimize_orientation:
                dx, dy = float(ind[2]), float(ind[3])
                nx, ny, nz = _normalize_direction(dx, dy, self.fixed_dir_z)
                optimizer_kwargs["evaluation_orientation"] = (nx, ny, nz)
        else:
            # ---- Multi-AP interface -----------------------------------
            positions: List[Tuple[float, float]] = []
            for ap_idx in range(self.num_aps):
                x = float(ind[2 * ap_idx])
                y = float(ind[2 * ap_idx + 1])
                positions.append((x, y))

            optimizer_kwargs = {
                "evaluation_positions": positions,
                "fixed_z": self.fixed_z,
            }

            if self.optimize_orientation:
                orientations: List[Tuple[float, float, float]] = []
                dir_base = self._n_pos_genes
                for ap_idx in range(self.num_aps):
                    dx = float(ind[dir_base + 2 * ap_idx])
                    dy = float(ind[dir_base + 2 * ap_idx + 1])
                    nx, ny, nz = _normalize_direction(dx, dy, self.fixed_dir_z)
                    orientations.append((nx, ny, nz))
                optimizer_kwargs["evaluation_orientations"] = orientations

        # ---- Reflector genes ------------------------------------------
        if self.reflector_enabled:
            rg = self._reflector_gene_start
            r_u = float(ind[rg])
            r_v = float(ind[rg + 1])
            focal_x = float(ind[rg + 2])
            focal_y = float(ind[rg + 3])

            optimizer_kwargs["reflector_u"] = r_u
            optimizer_kwargs["reflector_v"] = r_v
            optimizer_kwargs["reflector_target"] = (
                focal_x, focal_y, self.focal_z,
            )
            optimizer_kwargs["percentile_target_quantile"] = (
                self.percentile_target_quantile
            )

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

        Position genes are clamped to ``position_bounds``.
        Direction genes, when present, are clamped to ``[-1, 1]``.
        Reflector genes, when present, are clamped:
            - ``u``, ``v`` to ``[0, 1]``
            - ``focal_x``, ``focal_y`` to ``focal_bounds``
        """
        # Clamp position genes (x, y pairs for each AP)
        for ap_idx in range(self.num_aps):
            ind[2 * ap_idx] = max(
                self.bounds["x_min"],
                min(ind[2 * ap_idx], self.bounds["x_max"]),
            )
            ind[2 * ap_idx + 1] = max(
                self.bounds["y_min"],
                min(ind[2 * ap_idx + 1], self.bounds["y_max"]),
            )
        # Clamp direction genes to [-1, 1]
        if self.optimize_orientation:
            dir_end = self._n_pos_genes + self._n_dir_genes
            for i in range(self._n_pos_genes, dir_end):
                ind[i] = max(-1.0, min(ind[i], 1.0))

        # Clamp reflector genes
        if self.reflector_enabled:
            rg = self._reflector_gene_start
            ind[rg] = max(0.0, min(ind[rg], 1.0))          # u ∈ [0, 1]
            ind[rg + 1] = max(0.0, min(ind[rg + 1], 1.0))  # v ∈ [0, 1]
            ind[rg + 2] = max(
                self.focal_bounds["fx_min"],
                min(ind[rg + 2], self.focal_bounds["fx_max"]),
            )
            ind[rg + 3] = max(
                self.focal_bounds["fy_min"],
                min(ind[rg + 3], self.focal_bounds["fy_max"]),
            )

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

        For multi-AP configurations, checks the separation constraint
        first.  Individuals that violate it receive a penalty fitness
        immediately (saving expensive ray-tracing).  Only valid
        individuals are submitted to the worker pool.

        Returns:
            Number of individuals processed (including penalised ones).
        """
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        if not invalid_ind:
            return 0

        # --- Separate valid vs. penalised (multi-AP constraint) --------
        if self.num_aps >= 2:
            valid_inds: List = []
            penalised_inds: List = []
            for ind in invalid_ind:
                if self._check_separation(ind):
                    valid_inds.append(ind)
                else:
                    penalised_inds.append(ind)

            # Assign penalty fitness to constraint violators
            for ind in penalised_inds:
                ind.fitness.values = (PENALTY_FITNESS_LINEAR,)
                ind.penalized = True
                ind.best_coverage = 0.0
                ind.best_directions = [None] * self.num_aps
                ind.best_look_ats = [None] * self.num_aps
                if not self.optimize_orientation:
                    ind.best_orientation_names = [None] * self.num_aps
        else:
            valid_inds = invalid_ind
            penalised_inds = []

        # --- Evaluate valid individuals via worker pool ----------------
        if valid_inds:
            results = toolbox.map(toolbox.evaluate, valid_inds)

            for ind, res in zip(valid_inds, results):
                ind.fitness.values = (res["best_metric"],)
                ind.penalized = False

                # Extract coverage percentage from worker result
                gs = res.get("grid_results", {})
                cov_vals = gs.get("coverage_values", [])
                ind.best_coverage = float(cov_vals[-1]) if cov_vals else 0.0

                if self.num_aps == 1:
                    # ---- Single-AP attribute storage ------------------
                    if self.optimize_orientation and len(ind) >= 4:
                        nx, ny, nz = _normalize_direction(
                            float(ind[2]), float(ind[3]), self.fixed_dir_z,
                        )
                        ind.best_direction = [nx, ny, nz]
                        pos = np.array(res["best_position"])
                        ind.best_look_at = (
                            pos + np.array([nx, ny, nz])
                        ).tolist()
                        ind.best_orientation_name = None
                    else:
                        ind.best_direction = res.get("best_direction")
                        ind.best_look_at = res.get("best_look_at")
                        gs = res.get("grid_results", {})
                        ind.best_orientation_name = gs.get(
                            "best_orientation_name",
                        )
                else:
                    # ---- Multi-AP attribute storage --------------------
                    if self.optimize_orientation:
                        dirs: List = []
                        look_ats: List = []
                        dir_base = self._n_pos_genes
                        for ap_idx in range(self.num_aps):
                            dx = float(ind[dir_base + 2 * ap_idx])
                            dy = float(ind[dir_base + 2 * ap_idx + 1])
                            nx, ny, nz = _normalize_direction(
                                dx, dy, self.fixed_dir_z,
                            )
                            dirs.append([nx, ny, nz])
                            pos = [
                                float(ind[2 * ap_idx]),
                                float(ind[2 * ap_idx + 1]),
                                self.fixed_z,
                            ]
                            look_ats.append([
                                pos[0] + nx,
                                pos[1] + ny,
                                pos[2] + nz,
                            ])
                        ind.best_directions = dirs
                        ind.best_look_ats = look_ats
                    else:
                        # Orientation from 8-dir sweep on worker
                        best_dir = res.get("best_direction")
                        best_la = res.get("best_look_at")
                        gs = res.get("grid_results", {})
                        if (isinstance(best_dir, list)
                                and len(best_dir) == self.num_aps):
                            ind.best_directions = best_dir
                        else:
                            ind.best_directions = [best_dir] * self.num_aps
                        if (isinstance(best_la, list)
                                and len(best_la) == self.num_aps):
                            ind.best_look_ats = best_la
                        else:
                            ind.best_look_ats = [best_la] * self.num_aps
                        names = gs.get("best_orientation_names")
                        if names is not None:
                            ind.best_orientation_names = names
                        else:
                            name = gs.get("best_orientation_name")
                            ind.best_orientation_names = [name] * self.num_aps

                # ---- Reflector attribute storage ----------------------
                if self.reflector_enabled:
                    ind.reflector_u = res.get("reflector_u")
                    ind.reflector_v = res.get("reflector_v")
                    ind.reflector_target = res.get("reflector_target")
                    ind.reflector_position = res.get("reflector_position")
                    # Percentile score (shadow-robust objective)
                    gs = res.get("grid_results", {})
                    ind.percentile_score = gs.get("percentile_score")
                    ind.percentile_score_dbm = gs.get(
                        "percentile_score_dbm",
                    )

        return len(invalid_ind)

    # ------------------------------------------------------------------
    # Result extraction helpers
    # ------------------------------------------------------------------

    def _extract_positions(self, ind) -> List[List[float]]:
        """Extract ``[x, y, z]`` per AP from an individual's genes."""
        positions = []
        for ap_idx in range(self.num_aps):
            positions.append([
                float(ind[2 * ap_idx]),
                float(ind[2 * ap_idx + 1]),
                self.fixed_z,
            ])
        return positions

    def _extract_directions(self, ind) -> Any:
        """Extract normalised direction(s) from an individual.

        Returns:
            1-AP: ``[nx, ny, nz]`` or ``None``.
            Multi-AP: ``[[nx, ny, nz], ...]`` or stored attribute.
        """
        if not self.optimize_orientation:
            if self.num_aps == 1:
                return getattr(ind, "best_direction", None)
            return getattr(ind, "best_directions", None)
        if self.num_aps == 1:
            if len(ind) >= 4:
                nx, ny, nz = _normalize_direction(
                    float(ind[2]), float(ind[3]), self.fixed_dir_z,
                )
                return [nx, ny, nz]
            return None
        # Multi-AP
        dirs = []
        dir_base = self._n_pos_genes
        for ap_idx in range(self.num_aps):
            dx = float(ind[dir_base + 2 * ap_idx])
            dy = float(ind[dir_base + 2 * ap_idx + 1])
            dirs.append(list(_normalize_direction(dx, dy, self.fixed_dir_z)))
        return dirs

    def _extract_reflector(self, ind) -> Optional[Dict[str, Any]]:
        """Extract reflector parameters from an individual's genes.

        Returns:
            Dict with ``u``, ``v``, ``focal_x``, ``focal_y``,
            ``focal_z`` keys, or ``None`` if reflector is disabled.
        """
        if not self.reflector_enabled:
            return None
        rg = self._reflector_gene_start
        return {
            "u": float(ind[rg]),
            "v": float(ind[rg + 1]),
            "focal_x": float(ind[rg + 2]),
            "focal_y": float(ind[rg + 3]),
            "focal_z": self.focal_z,
        }

    # ------------------------------------------------------------------
    # Verbose formatting helpers
    # ------------------------------------------------------------------

    def _fmt_best_pos(self, ind) -> str:
        """Format best position(s) for verbose logging."""
        if self.num_aps == 1:
            return f"({ind[0]:.2f}, {ind[1]:.2f})"
        parts = []
        for ap_idx in range(self.num_aps):
            parts.append(
                f"AP{ap_idx}({ind[2*ap_idx]:.2f}, {ind[2*ap_idx+1]:.2f})"
            )
        return " | ".join(parts)

    def _fmt_best_dir(self, ind) -> str:
        """Format best direction(s) for verbose logging."""
        dirs = self._extract_directions(ind)
        if dirs is None:
            return ""
        if self.num_aps == 1:
            if (isinstance(dirs, list)
                    and len(dirs) == 3
                    and not isinstance(dirs[0], list)):
                return (f" | dir=({dirs[0]:+.3f}, "
                        f"{dirs[1]:+.3f}, {dirs[2]:+.3f})")
            return ""
        # Multi-AP directions
        parts = []
        for ap_idx, d in enumerate(dirs):
            if d is not None:
                parts.append(
                    f"d{ap_idx}=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})"
                )
        return (" | " + " ".join(parts)) if parts else ""

    def _fmt_reflector(self, ind) -> str:
        """Format reflector genes for verbose logging."""
        refl = self._extract_reflector(ind)
        if refl is None:
            return ""
        return (
            f" | refl(u={refl['u']:.3f}, v={refl['v']:.3f}) "
            f"focal({refl['focal_x']:.2f}, {refl['focal_y']:.2f})"
        )

    def _build_generation_detail(
        self,
        gen: int,
        nevals: int,
        record: dict,
        best_ind,
        gen_time: float,
    ) -> Dict[str, Any]:
        """Build a generation-detail dict for the results."""
        detail: Dict[str, Any] = {
            "gen": gen,
            "nevals": nevals,
            "max_dbm": record["max_dbm"],
            "mean_dbm": record["mean_dbm"],
            "std": record["std"],
            "best_coverage": getattr(best_ind, "best_coverage", 0.0),
            "time": gen_time,
        }
        if self.num_aps == 1:
            detail["best_x"] = float(best_ind[0])
            detail["best_y"] = float(best_ind[1])
            detail["best_direction"] = getattr(
                best_ind, "best_direction", None,
            )
            detail["best_orientation_name"] = getattr(
                best_ind, "best_orientation_name", None,
            )
        else:
            positions = []
            for ap_idx in range(self.num_aps):
                positions.append([
                    float(best_ind[2 * ap_idx]),
                    float(best_ind[2 * ap_idx + 1]),
                ])
            detail["best_positions"] = positions
            detail["best_directions"] = getattr(
                best_ind, "best_directions", None,
            )
            detail["best_orientation_names"] = getattr(
                best_ind, "best_orientation_names", None,
            )
            detail["ap_separation"] = self._ap_distance(
                best_ind, self.num_aps,
            )

        # Reflector genes (when enabled)
        if self.reflector_enabled:
            detail["reflector"] = self._extract_reflector(best_ind)

        return detail

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
                - ``mut_sigma_pos`` (float): Gaussian sigma for position
                    genes (default: ``mut_sigma``).
                - ``mut_sigma_dir`` (float): Gaussian sigma for direction
                    genes (default 0.3).
                - ``mut_indpb`` (float): Per-gene mutation prob (default 0.2).
                - ``hof_size`` (int): Hall-of-fame size (default 5).
            seed: Random seed for reproducibility.
            verbose: Print generation-by-generation progress.

        Returns:
            Dictionary with ``best_individual``, ``best_fitness``,
            ``best_fitness_dbm``, ``best_position`` (1-AP) or
            ``best_positions`` (multi-AP), ``hall_of_fame``, ``logbook``,
            ``total_time``, ``total_evaluations``, ``ga_params``,
            ``generation_details``.
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
        mut_sigma_pos = ga.get("mut_sigma_pos", mut_sigma)
        mut_sigma_dir = ga.get("mut_sigma_dir", 0.3)
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
            if self.num_aps == 1:
                _mode = (
                    "4D [x,y,dx,dy]"
                    if self.optimize_orientation
                    else "2D [x,y]"
                )
            else:
                if self.optimize_orientation:
                    _mode = (
                        f"{self._n_pos_genes + self._n_dir_genes}D "
                        f"[{'x,y,' * self.num_aps}"
                        f"{'dx,dy,' * self.num_aps}] "
                        f"({self.num_aps} APs)"
                    )
                else:
                    _mode = (
                        f"{self._n_pos_genes}D "
                        f"[{'x,y,' * self.num_aps}] "
                        f"({self.num_aps} APs)"
                    )
            if self.reflector_enabled:
                _mode += " + Reflector [u,v,fx,fy]"
            _mode += f" → {self._n_genes} genes total"
            print(f"DEAP GENETIC ALGORITHM ({_mode}, IoC-injected map)")
            print("=" * 80)
            print(
                f"  Population: {pop_size} | Generations: {n_gen} "
                f"| APs: {self.num_aps}"
            )
            print(f"  cxpb={cxpb}  mutpb={mutpb}  tournsize={tournsize}")
            if self.optimize_orientation:
                print(
                    f"  cx_alpha={cx_alpha}  mut_sigma_pos={mut_sigma_pos}  "
                    f"mut_sigma_dir={mut_sigma_dir}  mut_indpb={mut_indpb}"
                )
                print(f"  fixed_dir_z={self.fixed_dir_z}")
            else:
                print(
                    f"  cx_alpha={cx_alpha}  mut_sigma={mut_sigma}  "
                    f"mut_indpb={mut_indpb}"
                )
            if self.num_aps >= 2:
                print(f"  min_ap_separation={self.min_ap_separation}m")
            print(
                f"  Bounds: x=[{self.bounds['x_min']}, "
                f"{self.bounds['x_max']}], "
                f"y=[{self.bounds['y_min']}, {self.bounds['y_max']}]"
            )
            if self.reflector_enabled:
                fb = self.focal_bounds
                print(
                    f"  REFLECTOR: genes +4 (u,v,fx,fy) | "
                    f"focal_z={self.focal_z} | "
                    f"quantile={self.percentile_target_quantile}"
                )
                print(
                    f"  Focal bounds: fx=[{fb['fx_min']}, {fb['fx_max']}], "
                    f"fy=[{fb['fy_min']}, {fb['fy_max']}]"
                )
            print("-" * 80)

        # -- DEAP toolbox setup -----------------------------------------
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

        # Build gene cycle: all position genes first, then direction genes,
        # then (optionally) reflector genes.
        #   1-AP orient:    (x, y, dx, dy)                      — 4 genes
        #   2-AP orient:    (x1, y1, x2, y2, dx1, dy1, dx2, dy2) — 8 genes
        #   1-AP no-orient: (x, y)                                — 2 genes
        #   2-AP no-orient: (x1, y1, x2, y2)                     — 4 genes
        #   + reflector:    (..., u, v, focal_x, focal_y)         — +4 genes
        pos_genes = (toolbox.attr_x, toolbox.attr_y) * self.num_aps

        if self.optimize_orientation:
            toolbox.register("attr_dir", random.uniform, -1.0, 1.0)
            dir_genes = (toolbox.attr_dir,) * self._n_dir_genes
            gene_cycle = pos_genes + dir_genes
        else:
            gene_cycle = pos_genes

        # Reflector genes: u ∈ [0,1], v ∈ [0,1], focal_x, focal_y
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
            reflector_genes = (
                toolbox.attr_refl_uv,   # u
                toolbox.attr_refl_uv,   # v
                toolbox.attr_focal_x,   # focal_x
                toolbox.attr_focal_y,   # focal_y
            )
            gene_cycle = gene_cycle + reflector_genes

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

        if self.optimize_orientation or self.reflector_enabled:
            # Split mutation: separate sigmas for position, direction,
            # and reflector genes.
            mut_sigma_reflector = ga.get("mut_sigma_reflector", 0.1)
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
            # Legacy uniform mutation (position-only mode)
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
        record["best_pos"] = self._fmt_best_pos(best_ind)
        logbook.record(gen=0, nevals=nevals, **record)

        generation_details.append(
            self._build_generation_detail(
                0, nevals, record, best_ind, gen_time,
            )
        )

        if verbose:
            _dir_tag = self._fmt_best_dir(best_ind)
            _sep_tag = ""
            if self.num_aps >= 2:
                sep = self._ap_distance(best_ind, self.num_aps)
                n_pen = sum(
                    1 for ind in population
                    if getattr(ind, "penalized", False)
                )
                _sep_tag = f" | sep={sep:.1f}m pen={n_pen}"
            _cov = getattr(best_ind, "best_coverage", 0.0)
            _refl_tag = self._fmt_reflector(best_ind)
            print(
                f"  Gen  0 | evals={nevals:>3d} | "
                f"best={record['max_dbm']:.2f} dBm | "
                f"mean={record['mean_dbm']:.2f} dBm | "
                f"cov={_cov:.1f}% | "
                f"pos={self._fmt_best_pos(best_ind)}"
                f"{_dir_tag}{_sep_tag}{_refl_tag} | "
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
            record["best_pos"] = self._fmt_best_pos(best_ind)
            logbook.record(gen=gen, nevals=nevals, **record)
            gen_time = time.time() - gen_start

            generation_details.append(
                self._build_generation_detail(
                    gen, nevals, record, best_ind, gen_time,
                )
            )

            if verbose:
                _dir_tag = self._fmt_best_dir(best_ind)
                _sep_tag = ""
                if self.num_aps >= 2:
                    sep = self._ap_distance(best_ind, self.num_aps)
                    n_pen = sum(
                        1 for ind in population
                        if getattr(ind, "penalized", False)
                    )
                    _sep_tag = f" | sep={sep:.1f}m pen={n_pen}"
                _cov = getattr(best_ind, "best_coverage", 0.0)
                _refl_tag = self._fmt_reflector(best_ind)
                print(
                    f"  Gen {gen:>2d} | evals={nevals:>3d} | "
                    f"best={record['max_dbm']:.2f} dBm | "
                    f"mean={record['mean_dbm']:.2f} dBm | "
                    f"cov={_cov:.1f}% | "
                    f"pos={self._fmt_best_pos(best_ind)}"
                    f"{_dir_tag}{_sep_tag}{_refl_tag} | "
                    f"time={gen_time:.1f}s"
                )

        total_time = time.time() - start_time

        # -- Build results dict -----------------------------------------
        best = hof[0]
        best_fitness = best.fitness.values[0]

        hall_of_fame_list = []
        for ind in hof:
            hof_entry: Dict[str, Any] = {
                "fitness": float(ind.fitness.values[0]),
                "fitness_dbm": _rss_watts_to_dbm(ind.fitness.values[0]),
                "coverage": getattr(ind, "best_coverage", 0.0),
                "chromosome": [float(g) for g in ind],
            }
            if self.num_aps == 1:
                hof_entry["position"] = [
                    float(ind[0]), float(ind[1]), self.fixed_z,
                ]
                hof_entry["direction"] = getattr(
                    ind, "best_direction", None,
                )
                hof_entry["look_at"] = getattr(ind, "best_look_at", None)
                hof_entry["orientation_name"] = getattr(
                    ind, "best_orientation_name", None,
                )
            else:
                hof_entry["positions"] = self._extract_positions(ind)
                hof_entry["directions"] = getattr(
                    ind, "best_directions", None,
                )
                hof_entry["look_ats"] = getattr(
                    ind, "best_look_ats", None,
                )
                hof_entry["orientation_names"] = getattr(
                    ind, "best_orientation_names", None,
                )
                hof_entry["ap_separation"] = self._ap_distance(
                    ind, self.num_aps,
                )
            if self.reflector_enabled:
                hof_entry["reflector"] = self._extract_reflector(ind)
            hall_of_fame_list.append(hof_entry)

        _best_genes = [float(g) for g in best]

        results: Dict[str, Any] = {
            "best_individual": _best_genes,
            "best_fitness": float(best_fitness),
            "best_fitness_dbm": _rss_watts_to_dbm(best_fitness),
            "best_coverage": getattr(best, "best_coverage", 0.0),
            "optimize_orientation": self.optimize_orientation,
            "num_aps": self.num_aps,
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
                "num_aps": self.num_aps,
                "min_ap_separation": self.min_ap_separation,
                "reflector_enabled": self.reflector_enabled,
            },
            "generation_details": generation_details,
        }

        # Populate position / direction fields based on num_aps
        if self.num_aps == 1:
            results["best_position"] = [
                float(best[0]), float(best[1]), self.fixed_z,
            ]
            results["best_direction"] = getattr(
                best, "best_direction", None,
            )
            results["best_look_at"] = getattr(best, "best_look_at", None)
            results["best_orientation_name"] = getattr(
                best, "best_orientation_name", None,
            )
        else:
            results["best_positions"] = self._extract_positions(best)
            results["best_directions"] = getattr(
                best, "best_directions", None,
            )
            results["best_look_ats"] = getattr(
                best, "best_look_ats", None,
            )
            results["best_orientation_names"] = getattr(
                best, "best_orientation_names", None,
            )
            results["best_ap_separation"] = self._ap_distance(
                best, self.num_aps,
            )

        # Reflector results
        if self.reflector_enabled:
            results["reflector_enabled"] = True
            results["best_reflector"] = self._extract_reflector(best)
            results["percentile_target_quantile"] = (
                self.percentile_target_quantile
            )

        if verbose:
            print("-" * 80)
            print("GA COMPLETE")
            if self.num_aps == 1:
                print(
                    f"  Best position: ({best[0]:.2f}, {best[1]:.2f}, "
                    f"{self.fixed_z})"
                )
                _bd = results.get("best_direction")
                _bn = results.get("best_orientation_name")
                if _bd:
                    print(
                        f"  Best direction: ({_bd[0]:+.4f}, {_bd[1]:+.4f}, "
                        f"{_bd[2]:+.4f})"
                        f"{f'  ({_bn})' if _bn else ''}"
                    )
                _bla = results.get("best_look_at")
                if _bla:
                    print(
                        f"  Best look_at:  ({_bla[0]:.2f}, {_bla[1]:.2f}, "
                        f"{_bla[2]:.2f})"
                    )
            else:
                for ap_idx in range(self.num_aps):
                    pos = results["best_positions"][ap_idx]
                    print(
                        f"  AP{ap_idx} position: ({pos[0]:.2f}, "
                        f"{pos[1]:.2f}, {pos[2]:.2f})"
                    )
                    bd = results.get("best_directions")
                    if bd and bd[ap_idx]:
                        d = bd[ap_idx]
                        print(
                            f"  AP{ap_idx} direction: ({d[0]:+.4f}, "
                            f"{d[1]:+.4f}, {d[2]:+.4f})"
                        )
                print(
                    f"  AP separation: "
                    f"{results['best_ap_separation']:.2f}m"
                )
            print(f"  Best P5 RSS:  {results['best_fitness_dbm']:.2f} dBm")
            print(f"  Coverage:      {results['best_coverage']:.1f}%")
            if self.reflector_enabled:
                refl = results.get("best_reflector", {})
                print(
                    f"  Reflector u,v: ({refl.get('u', 0):.4f}, "
                    f"{refl.get('v', 0):.4f})"
                )
                print(
                    f"  Focal point:   ({refl.get('focal_x', 0):.2f}, "
                    f"{refl.get('focal_y', 0):.2f}, "
                    f"{refl.get('focal_z', 0):.2f})"
                )
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
        Save a 2x2 visualisation of the GA evolution.

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

        num_aps = results.get("num_aps", self.num_aps)
        bounds = position_bounds or self.bounds
        gen_details = results["generation_details"]
        gens = [g["gen"] for g in gen_details]
        best_dbm = [g["max_dbm"] for g in gen_details]
        mean_dbm = [g["mean_dbm"] for g in gen_details]

        arrow_scale = 2.0

        _AP_COLORS = [
            "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
            "#9467bd", "#8c564b",
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ---- Panel 1: Fitness convergence -----------------------------
        ax = axes[0, 0]
        ax.plot(
            gens, best_dbm, "r-o", markersize=3, linewidth=1.5,
            label="Best (HoF)",
        )
        ax.plot(
            gens, mean_dbm, "b--", linewidth=1.0, alpha=0.7,
            label="Pop Mean",
        )
        if rss_range_dbm:
            ax.set_ylim(rss_range_dbm[0], rss_range_dbm[1])
        ax.set_xlabel("Generation")
        ax.set_ylabel("P5 RSS (dBm)")

        # Coverage % on secondary y-axis
        best_cov = [g.get("best_coverage", 0.0) for g in gen_details]
        if any(c > 0 for c in best_cov):
            ax2 = ax.twinx()
            ax2.plot(
                gens, best_cov, "g-s", markersize=2, linewidth=1.2,
                alpha=0.8, label="Coverage %",
            )
            ax2.set_ylabel("Coverage (%)", color="green")
            ax2.tick_params(axis="y", labelcolor="green")
            ax2.set_ylim(0, 105)
            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
        else:
            ax.legend()

        ax.set_title("Fitness Convergence")
        ax.grid(True, alpha=0.3)

        # ---- Panel 2: Best position trajectory ------------------------
        ax = axes[0, 1]

        if num_aps == 1:
            best_xs = [g["best_x"] for g in gen_details]
            best_ys = [g["best_y"] for g in gen_details]
            best_dirs = [g.get("best_direction") for g in gen_details]
            has_dirs = any(d is not None for d in best_dirs)

            ax.plot(
                best_xs, best_ys, "b-o", markersize=4, linewidth=1.5,
                alpha=0.7,
            )
            ax.plot(
                best_xs[0], best_ys[0], "go", markersize=12, label="Gen 0",
            )
            ax.plot(
                best_xs[-1], best_ys[-1], "r*", markersize=15, label="Final",
            )

            if has_dirs:
                for i, (px, py, d) in enumerate(
                    zip(best_xs, best_ys, best_dirs)
                ):
                    if d is not None:
                        alpha = 0.3 + 0.7 * (
                            i / max(len(best_dirs) - 1, 1)
                        )
                        ax.quiver(
                            px, py,
                            d[0] * arrow_scale, d[1] * arrow_scale,
                            angles="xy", scale_units="xy", scale=1,
                            color="red", alpha=alpha, width=0.005,
                            headwidth=3, headlength=4,
                        )
        else:
            # Multi-AP: one trajectory per AP
            for ap_idx in range(num_aps):
                xs = [
                    g["best_positions"][ap_idx][0] for g in gen_details
                ]
                ys = [
                    g["best_positions"][ap_idx][1] for g in gen_details
                ]
                color = _AP_COLORS[ap_idx % len(_AP_COLORS)]
                ax.plot(
                    xs, ys, "-o", color=color, markersize=4,
                    linewidth=1.5, alpha=0.7, label=f"AP{ap_idx}",
                )
                ax.plot(
                    xs[0], ys[0], "o", color=color, markersize=10,
                    markeredgecolor="green", markeredgewidth=2,
                )
                ax.plot(
                    xs[-1], ys[-1], "*", color=color, markersize=14,
                    markeredgecolor="red", markeredgewidth=1,
                )

                # Direction arrows along trajectory
                dirs_list = [
                    g.get("best_directions") for g in gen_details
                ]
                for i, (px, py, dirs) in enumerate(
                    zip(xs, ys, dirs_list)
                ):
                    if (dirs is not None
                            and ap_idx < len(dirs)
                            and dirs[ap_idx] is not None):
                        d = dirs[ap_idx]
                        alpha = 0.3 + 0.7 * (
                            i / max(len(dirs_list) - 1, 1)
                        )
                        ax.quiver(
                            px, py,
                            d[0] * arrow_scale, d[1] * arrow_scale,
                            angles="xy", scale_units="xy", scale=1,
                            color=color, alpha=alpha, width=0.005,
                            headwidth=3, headlength=4,
                        )

        if bounds:
            ax.set_xlim(bounds["x_min"], bounds["x_max"])
            ax.set_ylim(bounds["y_min"], bounds["y_max"])
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        title_suffix = (
            " + Orientation" if self.optimize_orientation else ""
        )
        title_ap = f" ({num_aps} APs)" if num_aps > 1 else ""
        ax.set_title(
            f"Best Position Trajectory{title_suffix}{title_ap}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ---- Panel 3: Hall-of-fame scatter ----------------------------
        ax = axes[1, 0]
        hof = results["hall_of_fame"]
        hof_dbm = [h["fitness_dbm"] for h in hof]

        if num_aps == 1:
            hof_x = [h["position"][0] for h in hof]
            hof_y = [h["position"][1] for h in hof]
            hof_dirs = [h.get("direction") for h in hof]
            sc = ax.scatter(
                hof_x, hof_y,
                c=hof_dbm, s=200, cmap="viridis", edgecolor="black",
                vmin=rss_range_dbm[0] if rss_range_dbm else None,
                vmax=rss_range_dbm[1] if rss_range_dbm else None,
            )
            for i, (hx, hy, hd) in enumerate(
                zip(hof_x, hof_y, hof_dirs)
            ):
                label_text = f"#{i + 1}\n{hof_dbm[i]:.1f}"
                ax.annotate(
                    label_text, (hx, hy),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=7,
                )
                if hd is not None:
                    ax.quiver(
                        hx, hy,
                        hd[0] * arrow_scale, hd[1] * arrow_scale,
                        angles="xy", scale_units="xy", scale=1,
                        color="darkred", alpha=0.8, width=0.006,
                        headwidth=3, headlength=4,
                    )
            plt.colorbar(sc, ax=ax, label="P5 RSS (dBm)")
        else:
            # Multi-AP HoF: plot each AP per HoF entry
            for hof_idx, h in enumerate(hof):
                positions = h.get("positions", [])
                directions = h.get("directions")
                for ap_idx, pos in enumerate(positions):
                    color = _AP_COLORS[ap_idx % len(_AP_COLORS)]
                    marker = "o" if ap_idx == 0 else "s"
                    ax.scatter(
                        pos[0], pos[1],
                        c=[hof_dbm[hof_idx]], s=180, cmap="viridis",
                        edgecolor=color, linewidths=2, marker=marker,
                        vmin=(
                            rss_range_dbm[0] if rss_range_dbm else None
                        ),
                        vmax=(
                            rss_range_dbm[1] if rss_range_dbm else None
                        ),
                    )
                    if (directions
                            and ap_idx < len(directions)
                            and directions[ap_idx] is not None):
                        d = directions[ap_idx]
                        ax.quiver(
                            pos[0], pos[1],
                            d[0] * arrow_scale, d[1] * arrow_scale,
                            angles="xy", scale_units="xy", scale=1,
                            color=color, alpha=0.8, width=0.005,
                            headwidth=3, headlength=4,
                        )
                # Annotate best HoF entry
                if hof_idx == 0 and len(positions) >= 2:
                    mid_x = (
                        sum(p[0] for p in positions) / len(positions)
                    )
                    mid_y = (
                        sum(p[1] for p in positions) / len(positions)
                    )
                    ax.annotate(
                        f"#{hof_idx+1} {hof_dbm[hof_idx]:.1f}dBm",
                        (mid_x, mid_y),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=7,
                    )
                # Dashed line between APs
                if len(positions) >= 2:
                    ax.plot(
                        [positions[0][0], positions[1][0]],
                        [positions[0][1], positions[1][1]],
                        "--", color="gray",
                        alpha=0.4 if hof_idx > 0 else 0.8,
                        linewidth=1,
                    )

        if bounds:
            ax.set_xlim(bounds["x_min"], bounds["x_max"])
            ax.set_ylim(bounds["y_min"], bounds["y_max"])
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Hall of Fame (top solutions)")
        ax.grid(True, alpha=0.3)

        # ---- Panel 4: Summary text -----------------------------------
        ax = axes[1, 1]
        ax.axis("off")
        ga_p = results["ga_params"]

        if ga_p.get("optimize_orientation"):
            mut_line = (
                f"sigma_pos={ga_p.get('mut_sigma_pos', ga_p.get('mut_sigma', '?'))}  "
                f"sigma_dir={ga_p.get('mut_sigma_dir', '?')}  "
                f"indpb={ga_p['mut_indpb']}"
            )
        else:
            mut_line = (
                f"mut_sigma={ga_p.get('mut_sigma', '?')}  "
                f"mut_indpb={ga_p['mut_indpb']}"
            )

        if num_aps == 1:
            mode_str = (
                "4D [x,y,dx,dy]"
                if ga_p.get("optimize_orientation")
                else "2D [x,y]"
            )
            pos_str = (
                f"Best Position: ({results['best_position'][0]:.2f}, "
                f"{results['best_position'][1]:.2f}, "
                f"{results['best_position'][2]:.2f})\n"
                f"Best Direction: "
                f"{_fmt_dir_summary(results.get('best_direction'), results.get('best_orientation_name'))}\n"
            )
        else:
            n_genes = ga_p.get("num_aps", num_aps) * (
                4 if ga_p.get("optimize_orientation") else 2
            )
            mode_str = f"{n_genes}D ({num_aps} APs)"
            pos_lines = []
            for ap_idx in range(num_aps):
                pos = results["best_positions"][ap_idx]
                pos_lines.append(
                    f"AP{ap_idx}: ({pos[0]:.2f}, "
                    f"{pos[1]:.2f}, {pos[2]:.2f})"
                )
                bd = results.get("best_directions")
                if bd and ap_idx < len(bd) and bd[ap_idx]:
                    d = bd[ap_idx]
                    pos_lines.append(
                        f"  dir=({d[0]:+.4f}, {d[1]:+.4f}, "
                        f"{d[2]:+.4f})"
                    )
            pos_str = "\n".join(pos_lines) + "\n"
            pos_str += (
                f"Separation: "
                f"{results.get('best_ap_separation', 0):.2f}m\n"
            )

        summary = (
            f"DEAP GA SUMMARY ({mode_str})\n"
            f"\n"
            f"Population: {ga_p['pop_size']} | "
            f"Generations: {ga_p['n_gen']}\n"
            f"cxpb={ga_p['cxpb']}  mutpb={ga_p['mutpb']}  "
            f"tournsize={ga_p['tournsize']}\n"
            f"cx_alpha={ga_p['cx_alpha']}  {mut_line}\n"
            f"\n"
            f"{pos_str}"
            f"Best P5 RSS:  {results['best_fitness_dbm']:.2f} dBm\n"
            f"Coverage:      {results.get('best_coverage', 0.0):.1f}%\n"
        )
        if num_aps >= 2:
            summary += (
                f"Min AP Sep:    "
                f"{ga_p.get('min_ap_separation', '?')}m\n"
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
