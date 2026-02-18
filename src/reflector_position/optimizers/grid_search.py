"""
Grid search optimizer for AP position (single-point, multi-AP aware).

This module implements single-point grid search evaluation for AP placement
optimization.  Each ``SinglePointGridSearchOptimizer`` evaluates one
configuration of AP positions and orientations, returning the combined
min-RSS metric.

Multi-AP support:
    The optimizer accepts a *list* of evaluation positions and orientations
    (one per AP).  When an AP's orientation entry is ``None``, the optimizer
    sweeps 8 cardinal directions for that AP.  ``itertools.product`` generates
    the full Cartesian product of sweep directions across all sweeping APs.

    ============  ============  ====================================
    AP1 orient.   AP2 orient.   Configurations evaluated
    ============  ============  ====================================
    ``None``      ``None``      64  (8 x 8)
    fixed         ``None``       8  (1 x 8)
    ``None``      fixed          8  (8 x 1)
    fixed         fixed          1  (direct evaluation)
    ============  ============  ====================================

Helper functions:
    * ``generate_grid_positions()`` — 1-AP legacy grid point generation.
    * ``generate_alternating_grid_tasks()`` — multi-AP alternating
      optimisation task generation for ``RayParallelOptimizer``.
"""

import itertools
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import sionna.rt
from sionna.rt import RadioMapSolver

from .base_optimizer import BaseAPOptimizer
from ..metrics import compute_min_rss_metric, compute_coverage_metric, rss_to_dbm


# ---------------------------------------------------------------------------
# Grid position generation
# ---------------------------------------------------------------------------

def generate_grid_positions(
    search_bounds: Dict[str, float],
    grid_resolution: float = 1.0,
    fixed_z: float = 3.8,
) -> List[np.ndarray]:
    """
    Generate uniformly-spaced grid positions for parallel grid search.

    Creates all grid points within the search bounds at the specified
    resolution. Each point can then be submitted as an independent
    Ray task for truly parallel grid search evaluation.

    Args:
        search_bounds: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max'.
        grid_resolution: Grid spacing in meters.
        fixed_z: Fixed z-coordinate (height) for all positions.

    Returns:
        List of position arrays [x, y, z].

    Example:
        >>> bounds = {'x_min': 5.0, 'x_max': 25.0, 'y_min': 5.0, 'y_max': 25.0}
        >>> positions = generate_grid_positions(bounds, grid_resolution=1.0)
        >>> print(f"{len(positions)} grid points")
        441 grid points
    """
    x_range = np.arange(
        search_bounds["x_min"],
        search_bounds["x_max"] + grid_resolution / 2,  # inclusive endpoint
        grid_resolution,
    )
    y_range = np.arange(
        search_bounds["y_min"],
        search_bounds["y_max"] + grid_resolution / 2,
        grid_resolution,
    )

    positions = []
    for x in x_range:
        for y in y_range:
            positions.append(np.array([x, y, fixed_z]))

    return positions


def generate_alternating_grid_tasks(
    active_ap_idx: int,
    search_bounds: Dict[str, float],
    fixed_positions: List[Tuple[float, float]],
    fixed_orientations: List[Optional[Tuple[float, float, float]]],
    grid_resolution: float = 1.0,
    fixed_z: float = 3.8,
) -> List[Dict]:
    """
    Generate work-item dicts for alternating-optimisation grid search.

    In alternating optimisation one AP at a time is optimised over a
    spatial grid while the remaining APs stay at their current (fixed)
    positions.  This function builds the list of kwargs dicts that can be
    submitted directly to ``RayParallelOptimizer.run()`` as ``work_items``.

    Args:
        active_ap_idx: Index of the AP being optimised (0-based).
        search_bounds: Spatial bounds for the active AP.
            Dictionary with ``x_min``, ``x_max``, ``y_min``, ``y_max``.
        fixed_positions: Current ``(x, y)`` positions for **all** APs.
            The entry at ``active_ap_idx`` is ignored and replaced by
            each grid point in turn.
        fixed_orientations: Current orientations for **all** APs.
            Each entry is either ``(dx, dy, dz)`` (fixed) or ``None``
            (sweep 8 cardinal directions).  The active AP's entry is
            typically ``None`` to sweep it, but the caller may override.
        grid_resolution: Grid spacing in metres for the active AP.
        fixed_z: Fixed z-coordinate (height) for all APs.

    Returns:
        List of kwargs dicts, each compatible with
        ``SinglePointGridSearchOptimizer.__init__``.  One dict per grid
        point for the active AP.

    Example:
        >>> tasks = generate_alternating_grid_tasks(
        ...     active_ap_idx=0,
        ...     search_bounds={'x_min': 5, 'x_max': 25, 'y_min': 5, 'y_max': 25},
        ...     fixed_positions=[(15.0, 15.0), (20.0, 10.0)],
        ...     fixed_orientations=[None, (0.0, 1.0, 0.0)],
        ...     grid_resolution=1.0,
        ... )
        >>> print(f"{len(tasks)} work items")
    """
    grid_points = generate_grid_positions(search_bounds, grid_resolution, fixed_z)
    num_aps = len(fixed_positions)

    work_items: List[Dict] = []
    for pt in grid_points:
        # Build per-AP position list: replace active AP with grid point
        positions: List[Tuple[float, float]] = []
        for i in range(num_aps):
            if i == active_ap_idx:
                positions.append((float(pt[0]), float(pt[1])))
            else:
                positions.append(tuple(float(c) for c in fixed_positions[i]))  # type: ignore[arg-type]

        # Mirror the fixed_orientations list as-is
        orientations: List[Optional[Tuple[float, float, float]]] = list(
            fixed_orientations
        )

        work_items.append({
            "evaluation_positions": positions,
            "evaluation_orientations": orientations,
            "fixed_z": fixed_z,
        })

    return work_items


# ---------------------------------------------------------------------------
# Cardinal directions for orientation sweep
# ---------------------------------------------------------------------------

# 8 cardinal / intercardinal direction unit vectors on the XY plane.
# Used by SinglePointGridSearchOptimizer to sweep orientations.
_SQRT2_2 = np.sqrt(2.0) / 2.0
CARDINAL_DIRECTIONS: List[Tuple[str, np.ndarray]] = [
    ("N",  np.array([0.0,  1.0,  0.0])),
    ("NE", np.array([_SQRT2_2,  _SQRT2_2, 0.0])),
    ("E",  np.array([1.0,  0.0,  0.0])),
    ("SE", np.array([_SQRT2_2, -_SQRT2_2, 0.0])),
    ("S",  np.array([0.0, -1.0,  0.0])),
    ("SW", np.array([-_SQRT2_2, -_SQRT2_2, 0.0])),
    ("W",  np.array([-1.0,  0.0,  0.0])),
    ("NW", np.array([-_SQRT2_2,  _SQRT2_2, 0.0])),
]


# ---------------------------------------------------------------------------
# SinglePointGridSearchOptimizer (multi-AP aware)
# ---------------------------------------------------------------------------

class SinglePointGridSearchOptimizer(BaseAPOptimizer):
    """
    Evaluate one multi-AP configuration with joint orientation sweep.

    Supports an arbitrary number of APs.  For each AP whose orientation is
    ``None``, the optimiser sweeps 8 cardinal directions.  The full
    Cartesian product of sweep directions is explored via
    ``itertools.product``, ensuring *joint* orientation optimisation.

    **Backward compatibility:**  The legacy single-AP parameters
    ``evaluation_position`` and ``evaluation_orientation`` are still
    accepted and automatically wrapped into length-1 lists.
    """

    def __init__(
        self,
        scene: sionna.rt.Scene,
        # ---- Multi-AP parameters (preferred) ----
        evaluation_positions: Optional[List[Tuple[float, float]]] = None,
        evaluation_orientations: Optional[
            List[Optional[Tuple[float, float, float]]]
        ] = None,
        # ---- Legacy single-AP parameters (backward compat) ----
        evaluation_position: Optional[Tuple[float, float]] = None,
        evaluation_orientation: Optional[Tuple[float, float, float]] = None,
        fixed_z: float = 3.8,
        position_bounds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialise single-point (possibly multi-AP) grid search evaluator.

        Use *either* the multi-AP parameters (``evaluation_positions``,
        ``evaluation_orientations``) *or* the legacy single-AP parameters
        (``evaluation_position``, ``evaluation_orientation``).  If both
        are provided, the multi-AP parameters take precedence.

        Args:
            scene: Sionna ``Scene`` object with transmitters already created.
            evaluation_positions: List of ``(x, y)`` positions, one per AP.
            evaluation_orientations: List of orientations, one per AP.
                An entry of ``None`` triggers an 8-direction sweep for
                that AP.  An entry of ``(dx, dy, dz)`` fixes it.
            evaluation_position: *(legacy)* Single ``(x, y)`` position.
            evaluation_orientation: *(legacy)* Single ``(dx, dy, dz)`` or
                ``None``.
            fixed_z: Height (z-coordinate) for all APs.
            position_bounds: Optional bounds (unused, for interface compat).
        """
        super().__init__(scene=scene, fixed_z=fixed_z, position_bounds=position_bounds)

        # --- Normalise to list-of-AP form ---
        if evaluation_positions is not None:
            self.evaluation_positions: List[Tuple[float, float]] = list(
                evaluation_positions
            )
        elif evaluation_position is not None:
            self.evaluation_positions = [evaluation_position]
        else:
            raise ValueError(
                "Either 'evaluation_positions' (multi-AP) or "
                "'evaluation_position' (single-AP) must be provided."
            )

        num_aps = len(self.evaluation_positions)

        if evaluation_orientations is not None:
            self.evaluation_orientations: List[
                Optional[Tuple[float, float, float]]
            ] = list(evaluation_orientations)
        elif evaluation_orientation is not None:
            self.evaluation_orientations = [evaluation_orientation]
        else:
            # Default: sweep all APs
            self.evaluation_orientations = [None] * num_aps

        # Pad orientations to match positions length if shorter
        while len(self.evaluation_orientations) < num_aps:
            self.evaluation_orientations.append(None)

        self.num_aps = num_aps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._solver = RadioMapSolver()

        # ---- Results storage ----
        self.results: Dict[str, Any] = {
            "num_aps": num_aps,
            "positions": [],
            "min_rss_values": [],
            "min_rss_dbm_values": [],
            "coverage_values": [],
            "best_orientations": None,       # list of [dx,dy,dz] per AP
            "best_orientation_names": None,  # list of cardinal names (or None)
            # Legacy aliases (single-AP consumers read these)
            "best_orientation": None,
            "best_orientation_name": None,
            "direction_sweep": [],           # per-combo metrics
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _configure_transmitters(
        self,
        positions: List[List[float]],
        directions: List[np.ndarray],
    ) -> None:
        """Set position *and* look-at orientation for every transmitter.

        ``positions`` and ``directions`` must be aligned with the scene's
        transmitters in iteration order.
        """
        tx_list = list(self.scene.transmitters.values())
        for i, tx in enumerate(tx_list[: len(positions)]):
            pos = positions[i]
            tx.position = pos
            target = [
                pos[0] + float(directions[i][0]),
                pos[1] + float(directions[i][1]),
                pos[2] + float(directions[i][2]),
            ]
            tx.look_at(target)

    def _compute_metrics(
        self,
        samples_per_tx: int,
        max_depth: int,
        coverage_threshold_dbm: float,
    ) -> Dict[str, float]:
        """Run one radio-map evaluation and return scalar metrics.

        Returns:
            Dictionary with ``rss`` (Watts), ``rss_dbm``, and ``coverage``
            (percentage) keys.
        """
        rm = self._solver(
            self.scene,
            cell_size=(1.0, 1.0),
            samples_per_tx=samples_per_tx,
            max_depth=max_depth,
            refraction=True,
            diffraction=True,
        )
        rss_tensor = torch.from_numpy(np.array(rm.rss)).to(self.device)

        min_rss = compute_min_rss_metric(rss_tensor)
        min_rss_dbm = rss_to_dbm(min_rss)
        coverage = compute_coverage_metric(rss_tensor, coverage_threshold_dbm)

        return {
            "rss": min_rss.cpu().item(),
            "rss_dbm": min_rss_dbm.cpu().item(),
            "coverage": coverage.cpu().item(),
        }

    # ------------------------------------------------------------------
    # Orientation sweep machinery
    # ------------------------------------------------------------------

    def _build_orientation_combos(
        self,
    ) -> List[Tuple[List[np.ndarray], List[Optional[str]]]]:
        """Build every orientation combination that must be evaluated.

        For APs with a fixed orientation the direction is kept as-is.
        For APs whose orientation is ``None``, 8 cardinal directions are
        expanded.  The Cartesian product across all sweeping APs is
        returned.

        Returns:
            List of ``(directions, direction_names)`` tuples.
            - ``directions``: list of ``np.ndarray`` (one per AP).
            - ``direction_names``: list of cardinal name ``str`` or
              ``None`` (one per AP).
        """
        per_ap_options: List[List[Tuple[Optional[str], np.ndarray]]] = []

        for orient in self.evaluation_orientations:
            if orient is not None:
                # Fixed orientation — single option
                per_ap_options.append(
                    [(None, np.asarray(orient, dtype=np.float64))]
                )
            else:
                # Sweep 8 cardinal directions
                per_ap_options.append(
                    [(name, vec.copy()) for name, vec in CARDINAL_DIRECTIONS]
                )

        combos: List[Tuple[List[np.ndarray], List[Optional[str]]]] = []
        for combo in itertools.product(*per_ap_options):
            names: List[Optional[str]] = [c[0] for c in combo]
            dirs: List[np.ndarray] = [c[1] for c in combo]
            combos.append((dirs, names))

        return combos

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        samples_per_tx: int = 1_000_000,
        max_depth: int = 13,
        coverage_threshold_dbm: float = -100.0,
        verbose: bool = False,
    ) -> Tuple[Any, Any, float]:
        """
        Evaluate the configuration (one or more APs).

        For each orientation combination produced by
        ``_build_orientation_combos``, all transmitters are configured and
        the radio map is computed.  The best joint configuration (highest
        min-RSS) is returned.

        Args:
            samples_per_tx: Number of ray-tracing samples.
            max_depth: Maximum ray-tracing depth.
            coverage_threshold_dbm: Threshold for coverage calculation.
            verbose: Print per-combo progress.

        Returns:
            For **single-AP** (backward compat):
                ``(position_array, orientation_array, min_rss_watts)``
            For **multi-AP**:
                ``(positions_list, orientations_list, min_rss_watts)``
        """
        start_time = time.time()

        # Build 3-D positions for all APs
        tx_positions: List[List[float]] = [
            [float(pos[0]), float(pos[1]), float(self.fixed_z)]
            for pos in self.evaluation_positions
        ]
        position_arrays: List[np.ndarray] = [np.array(p) for p in tx_positions]

        # Build orientation combos (Cartesian product of sweeps)
        combos = self._build_orientation_combos()
        num_combos = len(combos)

        best_rss = -float("inf")
        best_dirs: List[np.ndarray] = [
            CARDINAL_DIRECTIONS[0][1].copy() for _ in range(self.num_aps)
        ]
        best_names: List[Optional[str]] = [None] * self.num_aps
        best_metrics: Dict[str, float] = {"rss": 0, "rss_dbm": 0, "coverage": 0}

        for combo_idx, (directions, dir_names) in enumerate(combos):
            self._configure_transmitters(tx_positions, directions)
            metrics = self._compute_metrics(
                samples_per_tx, max_depth, coverage_threshold_dbm,
            )

            # Store per-combo sweep data
            self.results["direction_sweep"].append({
                "combo_idx": combo_idx,
                "direction_names": list(dir_names),
                "directions": [d.tolist() for d in directions],
                "min_rss": metrics["rss"],
                "min_rss_dbm": metrics["rss_dbm"],
                "coverage": metrics["coverage"],
            })

            if metrics["rss"] > best_rss:
                best_rss = metrics["rss"]
                best_dirs = [d.copy() for d in directions]
                best_names = list(dir_names)
                best_metrics = metrics

            if verbose:
                dir_str = " | ".join(
                    f"AP{i} {n or 'fixed'} "
                    f"({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})"
                    for i, (n, d) in enumerate(zip(dir_names, directions))
                )
                print(
                    f"    [{combo_idx + 1}/{num_combos}] {dir_str}: "
                    f"Min RSS = {metrics['rss_dbm']:.2f} dBm, "
                    f"Coverage = {metrics['coverage']:.1f}%"
                )

        # Persist best result
        self.results["positions"] = [p.tolist() for p in position_arrays]
        self.results["min_rss_values"].append(best_metrics["rss"])
        self.results["min_rss_dbm_values"].append(best_metrics["rss_dbm"])
        self.results["coverage_values"].append(best_metrics["coverage"])
        self.results["best_orientations"] = [d.tolist() for d in best_dirs]
        self.results["best_orientation_names"] = best_names
        # Legacy aliases for single-AP consumers
        self.results["best_orientation"] = best_dirs[0].tolist()
        self.results["best_orientation_name"] = best_names[0]

        elapsed = time.time() - start_time

        if verbose:
            self._log_result(
                position_arrays, best_dirs, best_names,
                best_rss, elapsed, num_combos,
            )

        # Return in single-AP or multi-AP format
        if self.num_aps == 1:
            return position_arrays[0], best_dirs[0], best_rss
        else:
            return (
                [p.tolist() for p in position_arrays],
                [d.tolist() for d in best_dirs],
                best_rss,
            )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_result(
        self,
        positions: List[np.ndarray],
        orientations: List[np.ndarray],
        dir_names: List[Optional[str]],
        rss_val: float,
        elapsed: float,
        num_combos: int,
    ) -> None:
        """Print a human-readable evaluation summary for all APs."""
        rss_dbm = self.results["min_rss_dbm_values"][-1]
        cov = self.results["coverage_values"][-1]

        parts: List[str] = []
        for i in range(self.num_aps):
            pos = positions[i]
            d = orientations[i]
            name = dir_names[i]
            pos_str = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            dir_str = f"({d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f})"
            if name:
                parts.append(f"AP{i} {pos_str} dir={name} {dir_str}")
            else:
                parts.append(f"AP{i} {pos_str} dir={dir_str}")

        ap_summary = " | ".join(parts)
        print(
            f"  {ap_summary}: "
            f"Min RSS = {rss_dbm:.2f} dBm, "
            f"Coverage = {cov:.1f}%, "
            f"Time = {elapsed:.2f}s ({num_combos} combos)"
        )

    def plot_results(self, **kwargs) -> None:
        """No-op for single point evaluation."""
        pass
