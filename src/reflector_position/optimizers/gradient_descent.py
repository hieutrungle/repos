"""
Gradient descent optimizer for AP position using differentiable ray tracing.

This module implements gradient-based optimization using PyTorch and DrJit
to leverage Sionna's differentiable ray tracing capabilities.

Supports single or multi-AP (N >= 1) cooperative placement.  When ``num_aps > 1``
a **repulsion loss** prevents APs from collapsing to the same position, promoting
diverse spatial coverage.

Key design choices for multi-AP:
- Sionna's ``RadioMapSolver`` natively sums power from all active transmitters.
  We use this sum-power radio map as a differentiable proxy for max-power coverage.
- A pairwise repulsion term ``weight / (distance^2 + eps)`` is added to the
  coverage loss to enforce spatial separation between APs.
- Each AP has independent trainable position ``(x, y)`` and orientation ``(dx, dy)``
  while sharing ``fixed_z`` and ``fixed_dir_z``.
"""

import time
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
import sionna.rt
from sionna.rt import RadioMapSolver, Transmitter
import copy

from .base_optimizer import BaseAPOptimizer
from ..metrics import (
    POWER_EPSILON,
    compute_min_rss_metric,
    compute_p5_rss_metric,
    compute_soft_min_rss_metric,
    normalized_softmin_loss,
    MaskedSoftMinLoss,
    differentiable_coverage_loss,
    compute_coverage_metric,
    rss_to_dbm,
)
from ..reflector_model import ReflectorController

# Per-AP colours and markers for trajectory plots
_AP_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
_AP_MARKERS = ["o", "s", "^", "D", "v", "P"]


class GradientDescentAPOptimizer(BaseAPOptimizer):
    """
    Gradient descent optimizer for AP positions using differentiable ray tracing.

    Supports **N >= 1** access points optimised simultaneously.  Each AP has
    independent trainable position ``(x, y)`` and look-at direction ``(dx, dy)``,
    while ``fixed_z`` and ``fixed_dir_z`` are shared across all APs.

    For ``num_aps > 1`` a repulsion loss prevents the APs from merging into the
    single strongest location.

    **Joint reflector optimisation:**  When an optional
    :class:`ReflectorController` is provided, the optimizer additionally
    learns the reflector's wall placement ``(u, v)`` and 3-D focal point
    ``(x, y, z)`` through ``torch.sigmoid``-bounded raw parameters that
    maintain continuous gradients into the Sionna scene graph.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        scene: sionna.rt.Scene,
        initial_positions: Union[List[Tuple[float, float]], Tuple[float, float], None] = None,
        initial_position: Optional[Tuple[float, float]] = None,
        num_aps: Optional[int] = None,
        fixed_z: float = 3.8,
        position_bounds: Optional[Dict[str, float]] = None,
        initial_directions_xy: Optional[List[Tuple[float, float]]] = None,
        initial_direction_xy: Optional[Tuple[float, float]] = None,
        fixed_dir_z: float = -0.3,
        optimize_orientation: bool = True,
        repulsion_weight: float = 1.0,
        reflector_controller: Optional[ReflectorController] = None,
        initial_focal_point: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Initialise gradient descent optimizer for one or more APs.

        Position resolution order:
        1. ``initial_positions`` – explicit list of ``(x, y)`` per AP.
        2. ``initial_position``  – single ``(x, y)`` (convenience, implies N=1).
        3. ``num_aps``           – auto-generate spread-out positions on opposite
           corners of the bounding box.

        Args:
            scene: Sionna Scene object (must already have Tx/Rx arrays configured).
            initial_positions: List of ``(x, y)`` tuples, one per AP.
                Also accepts a single ``(x, y)`` tuple for backward-compat
                positional calls.
            initial_position: Single ``(x, y)`` tuple (implies ``num_aps=1``).
            num_aps: Number of APs when positions are auto-generated.
            fixed_z: Fixed height for all APs.
            position_bounds: ``{'x_min', 'x_max', 'y_min', 'y_max'}``.
            initial_directions_xy: Per-AP ``(dx, dy)`` look-at direction.
            initial_direction_xy: Single ``(dx, dy)`` replicated to all APs.
            fixed_dir_z: Fixed z-component of look-at direction (shared).
            optimize_orientation: Whether to optimise orientation as well as
                position.
            repulsion_weight: Multiplier for the pairwise repulsion loss term.
                Only active when ``num_aps > 1``.  Higher values push APs
                further apart.  Set to ``0`` to disable.
            reflector_controller: Optional :class:`ReflectorController`
                instance.  When provided the optimizer jointly learns the
                reflector's wall placement and focal-point orientation.
            initial_focal_point: Initial 3-D focal point ``(x, y, z)`` for
                the reflector beam-forming.  Falls back to the centre of
                ``position_bounds`` at ``z = fixed_z`` if not given.
        """
        super().__init__(scene=scene, fixed_z=fixed_z, position_bounds=position_bounds)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimize_orientation = optimize_orientation
        self.repulsion_weight = repulsion_weight

        # --- Position normalisation setup ---------------------------------
        self.pos_min = position_bounds.get("x_min", 5.0) if position_bounds else 5.0
        self.pos_max = position_bounds.get("x_max", 25.0) if position_bounds else 25.0
        self.pos_range = self.pos_max - self.pos_min

        # --- Resolve initial positions ------------------------------------
        if initial_positions is not None:
            # Accept a bare (x, y) tuple passed positionally (old calling convention)
            if (
                isinstance(initial_positions, tuple)
                and len(initial_positions) == 2
                and all(isinstance(v, (int, float)) for v in initial_positions)
            ):
                initial_positions = [initial_positions]
            self.num_aps = len(initial_positions)
        elif initial_position is not None:
            initial_positions = [initial_position]
            self.num_aps = 1
        elif num_aps is not None:
            self.num_aps = num_aps
            initial_positions = self._generate_spread_positions(
                num_aps, position_bounds,
            )
        else:
            # Default: single AP at centre of bounds
            cx = (self.pos_min + self.pos_max) / 2
            initial_positions = [(cx, cx)]
            self.num_aps = 1

        # --- Trainable position tensors  [num_aps] -----------------------
        norm_xs = [self._normalize(p[0]) for p in initial_positions]
        norm_ys = [self._normalize(p[1]) for p in initial_positions]
        self.tx_x = torch.tensor(
            norm_xs, dtype=torch.float32, requires_grad=True, device=self.device,
        )
        self.tx_y = torch.tensor(
            norm_ys, dtype=torch.float32, requires_grad=True, device=self.device,
        )

        # --- Ensure scene has enough transmitters -------------------------
        self._ensure_transmitters(initial_positions)

        # --- Orientation tensors  [num_aps, 2] ----------------------------
        if initial_directions_xy is not None:
            dir_inits = list(initial_directions_xy)
        elif initial_direction_xy is not None:
            dir_inits = [initial_direction_xy] * self.num_aps
        else:
            # Random initial direction per AP so each task explores differently
            dir_inits = [
                (np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0))
                for _ in range(self.num_aps)
            ]

        # Perturb near-zero directions to avoid NaN from atan2(0, 0)
        sanitised = []
        for dx0, dy0 in dir_inits:
            if abs(dx0) < 1e-6 and abs(dy0) < 1e-6:
                dx0, dy0 = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
            sanitised.append([dx0, dy0])

        self.tx_dir_xy = torch.tensor(
            sanitised, dtype=torch.float32,
            requires_grad=optimize_orientation, device=self.device,
        )  # shape [num_aps, 2]
        self.fixed_dir_z = fixed_dir_z

        # --- Solver -------------------------------------------------------
        self.radio_solver = RadioMapSolver()
        self.radio_solver.loop_mode = "evaluated"

        # --- Reflector (optional) -----------------------------------------
        self.reflector_controller: Optional[ReflectorController] = reflector_controller
        self.reflector_u_raw: Optional[torch.Tensor] = None
        self.reflector_v_raw: Optional[torch.Tensor] = None
        self.focal_point_raw: Optional[torch.Tensor] = None

        if reflector_controller is not None:
            # Unbounded raw parameters — sigmoid maps them to [0, 1]
            # Initialise at 0.0 so sigmoid(0) = 0.5 (wall centre)
            self.reflector_u_raw = torch.tensor(
                0.0, dtype=torch.float32, device=self.device,
                requires_grad=True,
            )
            self.reflector_v_raw = torch.tensor(
                0.0, dtype=torch.float32, device=self.device,
                requires_grad=True,
            )

            # Focal-point initialisation
            if initial_focal_point is not None:
                fp_init = list(initial_focal_point)
            else:
                # Default: centre of position bounds at receiver height
                cx = (self.pos_min + self.pos_max) / 2.0
                cy = (self.pos_min + self.pos_max) / 2.0
                fp_init = [cx, cy, 1.5]

            self.focal_point_raw = torch.tensor(
                fp_init, dtype=torch.float32, device=self.device,
                requires_grad=True,
            )

        # --- History tracking ---------------------------------------------
        self.history: Dict[str, list] = {
            "positions": [],            # per-iter AP positions
            "directions": [],           # per-iter AP directions (normalised)
            "look_at_targets": [],      # per-iter AP look-at points
            "min_rss_values": [],       # scalar per iter (5th-percentile RSS)
            "min_rss_dbm_values": [],   # scalar per iter (5th-percentile RSS in dBm)
            "coverage_values": [],      # scalar per iter
            "losses": [],               # scalar per iter (total loss)
            "fairness_losses": [],      # scalar per iter (fairness/softmin component)
            "coverage_obj_losses": [],  # scalar per iter (sigmoid coverage component)
            "coverage_losses": [],      # scalar per iter (alpha*fairness + beta*coverage)
            "repulsion_losses": [],     # scalar per iter (repulsion component)
            "gradients": [],            # per-iter position gradients
            "direction_gradients": [],  # per-iter direction gradients
            "ap_distances": [],         # per-iter pairwise AP distances
            "reflector_u": [],          # per-iter reflector u coordinate
            "reflector_v": [],          # per-iter reflector v coordinate
            "reflector_focal_point": [],  # per-iter reflector focal point
            "reflector_position": [],   # per-iter reflector world position
        }

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_spread_positions(
        num_aps: int,
        bounds: Optional[Dict[str, float]],
    ) -> List[Tuple[float, float]]:
        """Generate spread-out initial positions to encourage separation.

        * N=1 → centre of the bounding box.
        * N=2 → opposite diagonal corners (with margin).
        * N>2 → evenly spaced on a circle inside the bounding box.
        """
        x_min = bounds.get("x_min", 5.0) if bounds else 5.0
        x_max = bounds.get("x_max", 25.0) if bounds else 25.0
        y_min = bounds.get("y_min", 5.0) if bounds else 5.0
        y_max = bounds.get("y_max", 25.0) if bounds else 25.0

        margin = 0.10 * min(x_max - x_min, y_max - y_min)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

        if num_aps == 1:
            return [(cx, cy)]
        elif num_aps == 2:
            return [
                (x_min + margin, y_min + margin),
                (x_max - margin, y_max - margin),
            ]
        else:
            r = min(x_max - x_min, y_max - y_min) / 3
            return [
                (cx + r * np.cos(2 * np.pi * i / num_aps),
                 cy + r * np.sin(2 * np.pi * i / num_aps))
                for i in range(num_aps)
            ]

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def _ensure_transmitters(
        self,
        initial_positions: List[Tuple[float, float]],
    ) -> None:
        """Add transmitters to the scene until it has ``num_aps`` TXs."""
        existing = list(self.scene.transmitters.keys())
        n_existing = len(existing)

        if n_existing >= self.num_aps:
            self.tx_names = existing[: self.num_aps]
            return

        # Inherit power from first existing TX (or default 5 dBm)
        ref_power = 5.0
        if n_existing > 0:
            raw = list(self.scene.transmitters.values())[0].power_dbm
            # power_dbm may be a DrJit scalar; convert safely
            try:
                ref_power = float(raw)
            except (TypeError, ValueError):
                ref_power = float(raw[0]) if hasattr(raw, '__getitem__') else 5.0

        for i in range(n_existing, self.num_aps):
            pos = initial_positions[i]
            tx = Transmitter(
                name=f"Tx{i:02d}",
                position=[float(pos[0]), float(pos[1]), self.fixed_z],
                power_dbm=ref_power,
            )
            self.scene.add(tx)

        self.tx_names = list(self.scene.transmitters.keys())[: self.num_aps]

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _normalize(self, v: float) -> float:
        """Map a physical coordinate to [0, 1]."""
        return (v - self.pos_min) / self.pos_range

    def _denormalize_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Map a normalised [0, 1] tensor back to physical coordinates."""
        return t * self.pos_range + self.pos_min

    # ------------------------------------------------------------------
    # Position / direction accessors
    # ------------------------------------------------------------------

    def get_full_positions(self) -> np.ndarray:
        """Return all AP positions as array of shape ``[num_aps, 3]``."""
        xs = self._denormalize_tensor(self.tx_x).detach().cpu().numpy()
        ys = self._denormalize_tensor(self.tx_y).detach().cpu().numpy()
        zs = np.full_like(xs, self.fixed_z)
        return np.stack([xs, ys, zs], axis=-1)  # [num_aps, 3]

    def get_full_position(self) -> np.ndarray:
        """Backward-compat: ``[3]`` for single AP, ``[num_aps, 3]`` otherwise."""
        positions = self.get_full_positions()
        if self.num_aps == 1:
            return positions[0]
        return positions

    def apply_position_constraints(self) -> None:
        """Clamp all APs' normalised positions to [0, 1]."""
        with torch.no_grad():
            self.tx_x.clamp_(0.0, 1.0)
            self.tx_y.clamp_(0.0, 1.0)

    def get_current_directions(self) -> torch.Tensor:
        """L2-normalised look-at directions for all APs.  Shape ``[num_aps, 3]``."""
        dir_z = torch.full(
            (self.num_aps, 1), self.fixed_dir_z,
            dtype=torch.float32, device=self.device,
        )
        full_dir = torch.cat([self.tx_dir_xy, dir_z], dim=1)  # [N, 3]
        return F.normalize(full_dir, dim=1)

    def get_current_direction(self) -> torch.Tensor:
        """Backward-compat: direction for first (or only) AP.  Shape ``[3]``."""
        return self.get_current_directions()[0]

    # ------------------------------------------------------------------
    # Pairwise AP distances
    # ------------------------------------------------------------------

    def _pairwise_distances(self) -> List[float]:
        """Return physical Euclidean distances for all AP pairs."""
        if self.num_aps <= 1:
            return []
        positions = self.get_full_positions()  # [N, 3]
        dists = []
        for i, j in combinations(range(self.num_aps), 2):
            d = float(np.linalg.norm(positions[i, :2] - positions[j, :2]))
            dists.append(d)
        return dists

    # ------------------------------------------------------------------
    # Repulsion loss
    # ------------------------------------------------------------------

    def _compute_repulsion_loss(self) -> torch.Tensor:
        """Pairwise inverse-square repulsion in *physical* coordinates.

        ``L_repulsion = sum_{i<j} 1 / (||p_i - p_j||^2 + eps)``

        The loss is large when APs are close and vanishes as they separate.
        """
        if self.num_aps <= 1:
            return torch.tensor(0.0, device=self.device)

        eps = 1e-4  # prevent division by zero
        repulsion = torch.zeros(1, device=self.device)

        for i, j in combinations(range(self.num_aps), 2):
            dx = self._denormalize_tensor(self.tx_x[i]) - self._denormalize_tensor(self.tx_x[j])
            dy = self._denormalize_tensor(self.tx_y[i]) - self._denormalize_tensor(self.tx_y[j])
            dist_sq = dx ** 2 + dy ** 2
            repulsion = repulsion + 1.0 / (dist_sq + eps)

        return repulsion.squeeze()

    # ------------------------------------------------------------------
    # Reflector parameter injection
    # ------------------------------------------------------------------

    def _apply_reflector_params(self) -> None:
        """Apply current reflector trainable parameters to the scene graph.

        Converts unbounded ``reflector_u_raw`` / ``reflector_v_raw`` through
        ``torch.sigmoid`` to obtain wall coordinates in [0, 1], injects them
        into the :class:`ReflectorController`, computes the specular-reflection
        orientation from ``focal_point_raw``, and pushes the updated geometry
        into the Mitsuba scene graph.

        This method is intentionally called **outside** ``torch.no_grad()``
        so the sigmoid / focal-point operations remain on the computational
        graph.  The ``apply_to_scene()`` call detaches values at the
        Mitsuba boundary, but the upstream PyTorch graph is unaffected.
        """
        ctrl = self.reflector_controller
        if ctrl is None:
            return

        # 1. Sigmoid-bounded wall coordinates  [0, 1]
        u_bounded = torch.sigmoid(self.reflector_u_raw)
        v_bounded = torch.sigmoid(self.reflector_v_raw)
        ctrl.u = u_bounded
        ctrl.v = v_bounded

        # 2. Update TX reference for reflection math (use first AP position)
        phys_positions = self.get_full_positions()  # [N, 3] — detached NumPy
        ctrl.set_tx_position(phys_positions[0])

        # 3. Focal-point injection and specular orientation
        ctrl.set_focal_point(self.focal_point_raw, requires_grad=True)
        ctrl.orient_to_target()

        # 4. Push geometry into Mitsuba scene graph
        ctrl.apply_to_scene()

    # ------------------------------------------------------------------
    # SPSA numerical gradient estimation for reflector parameters
    # ------------------------------------------------------------------

    def _eval_detached_loss(
        self,
        samples_per_tx: int,
        max_depth: int,
        use_soft_min: bool,
        temperature: float,
        shadow_quantile: float,
        fairness_loss_type: str,
        alpha: float,
        beta: float,
        coverage_threshold_dbm: float,
        coverage_temperature: float,
    ) -> float:
        """Evaluate the composite loss with current scene state (no grad).

        Sets AP positions from ``self.tx_x / tx_y`` (detached), runs the
        radio solver, computes the same fairness + coverage loss as
        ``compute_loss``, and returns the scalar value.  Used by SPSA.
        """
        dr.suspend_grad()
        try:
            cur_dirs = self.get_current_directions()
            phys_pos = self.get_full_positions()
            tx_list = [self.scene.transmitters[n] for n in self.tx_names]
            for i in range(self.num_aps):
                tx_list[i].position = [
                    float(phys_pos[i, 0]),
                    float(phys_pos[i, 1]),
                    self.fixed_z,
                ]
                target = [
                    float(phys_pos[i, 0]) + float(cur_dirs[i, 0].item()),
                    float(phys_pos[i, 1]) + float(cur_dirs[i, 1].item()),
                    self.fixed_z + float(cur_dirs[i, 2].item()),
                ]
                tx_list[i].look_at(target)

            rm = self.radio_solver(
                self.scene,
                cell_size=(1.0, 1.0),
                samples_per_tx=samples_per_tx,
                max_depth=max_depth,
                refraction=True,
                diffraction=True,
            )
            rss = torch.from_numpy(np.array(rm.rss))
        finally:
            dr.resume_grad()

        # Compute composite loss (pure PyTorch, no grad needed)
        with torch.no_grad():
            effective_type = fairness_loss_type
            if effective_type == "auto" and not use_soft_min:
                effective_type = "percentile"
            fairness = self._compute_fairness_loss(
                rss,
                fairness_loss_type=effective_type,
                temperature=temperature,
                shadow_quantile=shadow_quantile,
            )
            cov = differentiable_coverage_loss(
                rss,
                threshold_dbm=coverage_threshold_dbm,
                temperature=coverage_temperature,
            )
            loss_val = float((alpha * fairness + beta * cov).item())
        return loss_val

    def _reflector_spsa_gradients(
        self,
        samples_per_tx: int,
        max_depth: int,
        use_soft_min: bool,
        temperature: float,
        shadow_quantile: float,
        fairness_loss_type: str,
        alpha: float,
        beta: float,
        coverage_threshold_dbm: float,
        coverage_temperature: float,
        perturbation_scale: float = 0.1,
    ) -> None:
        """Estimate reflector parameter gradients via SPSA.

        SPSA (Simultaneous Perturbation Stochastic Approximation) estimates
        the gradient of a scalar loss w.r.t. all parameters using only **2**
        forward evaluations, regardless of the number of parameters.

        This is used because ``SceneObject.position`` / ``.look_at()``
        mutate mesh vertices via ``scene_params.update()``, which is
        incompatible with DrJit's AD graph inside ``@dr.wrap``.

        After this call, ``.grad`` is set on ``reflector_u_raw``,
        ``reflector_v_raw``, and ``focal_point_raw``.
        """
        if self.reflector_controller is None:
            return

        c = perturbation_scale

        # Collect reflector trainable parameters
        params: list = [self.reflector_u_raw, self.reflector_v_raw]
        if self.focal_point_raw is not None:
            params.append(self.focal_point_raw)

        # Bernoulli ±1 perturbation for each parameter tensor
        deltas = [
            (torch.bernoulli(torch.ones_like(p.data) * 0.5) * 2 - 1)
            for p in params
        ]

        eval_kwargs = dict(
            samples_per_tx=samples_per_tx,
            max_depth=max_depth,
            use_soft_min=use_soft_min,
            temperature=temperature,
            shadow_quantile=shadow_quantile,
            fairness_loss_type=fairness_loss_type,
            alpha=alpha,
            beta=beta,
            coverage_threshold_dbm=coverage_threshold_dbm,
            coverage_temperature=coverage_temperature,
        )

        # ---- Forward evaluation at +perturbation -------------------------
        with torch.no_grad():
            for p, d in zip(params, deltas):
                p.data.add_(c * d)
            self._apply_reflector_params()
        loss_plus = self._eval_detached_loss(**eval_kwargs)

        # ---- Forward evaluation at -perturbation -------------------------
        with torch.no_grad():
            for p, d in zip(params, deltas):
                p.data.sub_(2.0 * c * d)       # from +c*d to -c*d
            self._apply_reflector_params()
        loss_minus = self._eval_detached_loss(**eval_kwargs)

        # ---- Restore original parameter values ---------------------------
        with torch.no_grad():
            for p, d in zip(params, deltas):
                p.data.add_(c * d)              # back to original
            self._apply_reflector_params()

        # ---- Set .grad via SPSA formula ----------------------------------
        # g_i ≈ (loss+ − loss−) / (2 * c * Δ_i)
        diff = loss_plus - loss_minus
        for p, d in zip(params, deltas):
            grad = diff / (2.0 * c * d)
            if p.grad is None:
                p.grad = grad.clone()
            else:
                p.grad.copy_(grad)

    # ------------------------------------------------------------------
    # Reflector state snapshot (for logging)
    # ------------------------------------------------------------------

    def _snapshot_reflector(self) -> Dict[str, Optional[list]]:
        """Return a dict of current reflector parameter values for logging.

        Returns detached CPU lists suitable for JSON serialisation.
        All keys are present but set to ``None`` when no reflector is active.
        """
        if self.reflector_controller is None:
            return {
                "u": None,
                "v": None,
                "focal_point": None,
                "position": None,
            }
        ctrl = self.reflector_controller
        u_val = torch.sigmoid(self.reflector_u_raw).detach().cpu().item()
        v_val = torch.sigmoid(self.reflector_v_raw).detach().cpu().item()
        fp_val = self.focal_point_raw.detach().cpu().tolist()
        try:
            pos_val = ctrl.wall_position().detach().cpu().tolist()
        except RuntimeError:
            pos_val = ctrl.get_position().tolist()
        return {
            "u": u_val,
            "v": v_val,
            "focal_point": fp_val,
            "position": pos_val,
        }

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Fairness loss dispatch
    # ------------------------------------------------------------------

    _VALID_FAIRNESS_TYPES = ("auto", "softmin", "masked_softmin", "percentile")

    def _compute_fairness_loss(
        self,
        rss: torch.Tensor,
        fairness_loss_type: str,
        temperature: float,
        shadow_quantile: float,
    ) -> torch.Tensor:
        """Dispatch fairness loss by type.

        Parameters
        ----------
        rss : torch.Tensor
            Radio-map RSS in linear Watts (from ``compute_rss``).
        fairness_loss_type : str
            One of ``"auto"``, ``"softmin"``, ``"masked_softmin"``,
            ``"percentile"``.
        temperature : float
            τ parameter for soft-min variants.
        shadow_quantile : float
            Quantile mask for ``MaskedSoftMinLoss``.

        Returns
        -------
        torch.Tensor
            Scalar fairness loss to minimise.
        """
        kind = fairness_loss_type
        if kind == "auto":
            kind = (
                "masked_softmin"
                if self.reflector_controller is not None
                else "softmin"
            )

        if kind == "masked_softmin":
            return MaskedSoftMinLoss(
                shadow_quantile=shadow_quantile,
                temperature=temperature,
            )(rss)
        elif kind == "softmin":
            return normalized_softmin_loss(
                rss.flatten().unsqueeze(0),
                temperature=temperature,
            )
        elif kind == "percentile":
            p5_rss = compute_p5_rss_metric(rss)
            log_p5_rss = rss_to_dbm(p5_rss)
            return -log_p5_rss / 100.0
        else:
            raise ValueError(
                f"Unknown fairness_loss_type={fairness_loss_type!r}.  "
                f"Valid options: {self._VALID_FAIRNESS_TYPES}"
            )

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        samples_per_tx: int = 1_000_000,
        max_depth: int = 13,
        use_soft_min: bool = True,
        temperature: float = 0.1,
        shadow_quantile: float = 0.05,
        fairness_loss_type: str = "auto",
        alpha: float = 0.6,
        beta: float = 0.4,
        coverage_threshold_dbm: float = -120.0,
        coverage_temperature: float = 2.0,
    ) -> torch.Tensor:
        """
        Compute the total loss for all APs (fairness + coverage + repulsion).

        The loss is a weighted composite of three terms:

        1. **Fairness loss** (``alpha``): controlled by ``fairness_loss_type``.

           - ``"auto"`` (default): ``MaskedSoftMinLoss`` when a reflector is
             active, ``normalized_softmin_loss`` otherwise.
           - ``"softmin"``: always ``normalized_softmin_loss``.
           - ``"masked_softmin"``: always ``MaskedSoftMinLoss``
             (shadow-aware, excludes bottom ``shadow_quantile`` cells).
           - ``"percentile"``: 5th-percentile RSS converted to dBm.
             Non-differentiable through DrJit but works with SPSA.

        2. **Coverage loss** (``beta``): ``differentiable_coverage_loss`` —
           maximises the fraction of users above ``coverage_threshold_dbm``
           using a smooth Sigmoid approximation.
        3. **Repulsion loss** (multi-AP only): pairwise inverse-square penalty
           to prevent APs from collapsing.

        ``total_loss = alpha * fairness + beta * coverage + repulsion_weight * repulsion``

        Args:
            samples_per_tx: Number of ray samples per transmitter.
            max_depth: Maximum ray tracing bounces.
            use_soft_min: **Deprecated** — use ``fairness_loss_type`` instead.
                When ``fairness_loss_type="auto"`` and ``use_soft_min=False``,
                falls back to ``"percentile"`` for backward compatibility.
            temperature: Temperature (τ) for soft-min fairness losses.
                Lower → sharper approximation of true min.
            shadow_quantile: Fraction of lowest-signal cells to mask out when
                using ``MaskedSoftMinLoss``.  Default 0.05.
            fairness_loss_type: Fairness loss selector.  One of
                ``"auto"``, ``"softmin"``, ``"masked_softmin"``,
                ``"percentile"``.  Default ``"auto"``.
            alpha: Weight for fairness loss.  Default 0.6.
            beta: Weight for coverage loss.  Default 0.4.
            coverage_threshold_dbm: Threshold for coverage objective (dBm).
            coverage_temperature: Sigmoid sharpness for coverage loss.

        Returns:
            Scalar loss tensor to minimise.
        """
        # ---- Reflector: apply current params to scene (detached) ----------
        # The reflector is a SceneObject whose position/orientation setters
        # modify mesh vertices via scene_params.update().  This is
        # fundamentally incompatible with @dr.wrap's AD graph (Transmitters,
        # by contrast, just store a mi.Point3f — no mesh mutation).
        #
        # Strategy: apply reflector geometry *outside* @dr.wrap so the
        # radio-map correctly reflects the reflector's placement, then use
        # SPSA (see _reflector_spsa_gradients) to estimate numerical
        # gradients for reflector_u_raw / reflector_v_raw / focal_point_raw.
        if self.reflector_controller is not None:
            self._apply_reflector_params()

        # L2-normalised directions for all APs  [num_aps, 3]
        directions = self.get_current_directions()

        # Pre-compute constant z-components for the closure
        dir_z_values = [float(directions[i, 2].item()) for i in range(self.num_aps)]

        # Ordered list of Sionna Transmitter objects matching our APs
        tx_list = [self.scene.transmitters[n] for n in self.tx_names]

        # Build flat scalar arg list for @dr.wrap:
        #   AP args:  [x_0..x_{N-1}, y_0..y_{N-1}, dx_0..dx_{N-1}, dy_0..dy_{N-1}]
        wrap_args: list = []
        for i in range(self.num_aps):
            wrap_args.append(self.tx_x[i])
        for i in range(self.num_aps):
            wrap_args.append(self.tx_y[i])
        for i in range(self.num_aps):
            wrap_args.append(directions[i, 0])
        for i in range(self.num_aps):
            wrap_args.append(directions[i, 1])

        @dr.wrap(source="torch", target="drjit")
        def compute_rss(*args):
            N = self.num_aps
            for i in range(N):
                tx = tx_list[i]
                xi = args[i]
                yi = args[N + i]
                dxi = args[2 * N + i]
                dyi = args[3 * N + i]

                phys_x = xi * self.pos_range + self.pos_min
                phys_y = yi * self.pos_range + self.pos_min
                tx.position = [phys_x.array, phys_y.array, self.fixed_z]

                target = [
                    phys_x.array + dxi.array,
                    phys_y.array + dyi.array,
                    self.fixed_z + dir_z_values[i],
                ]
                tx.look_at(target)

            rm = self.radio_solver(
                self.scene,
                cell_size=(1.0, 1.0),
                samples_per_tx=samples_per_tx,
                max_depth=max_depth,
                refraction=True,
                diffraction=True,
            )
            return rm.rss

        rss = compute_rss(*wrap_args)

        # Cache for metric logging (avoids a second radio-map evaluation)
        self._last_rss = rss.detach().clone()

        # ---- Fairness loss -----------------------------------------------
        # Backward compat: use_soft_min=False → percentile when type="auto"
        effective_type = fairness_loss_type
        if effective_type == "auto" and not use_soft_min:
            effective_type = "percentile"

        fairness_loss = self._compute_fairness_loss(
            rss,
            fairness_loss_type=effective_type,
            temperature=temperature,
            shadow_quantile=shadow_quantile,
        )

        # ---- Coverage loss (sigmoid-smoothed coverage ratio) -------------
        cov_loss = differentiable_coverage_loss(
            rss,
            threshold_dbm=coverage_threshold_dbm,
            temperature=coverage_temperature,
        )

        # ---- Composite: alpha * fairness + beta * coverage ---------------
        coverage_loss = alpha * fairness_loss + beta * cov_loss

        # ---- Repulsion loss (pure PyTorch, no DrJit) ---------------------
        if self.num_aps > 1 and self.repulsion_weight > 0:
            repulsion_loss = self._compute_repulsion_loss()
            total_loss = coverage_loss + self.repulsion_weight * repulsion_loss
        else:
            repulsion_loss = torch.tensor(0.0, device=self.device)
            total_loss = coverage_loss

        # Stash component values for logging
        self._last_fairness_loss = float(fairness_loss.item())
        self._last_coverage_obj_loss = float(cov_loss.item())
        self._last_coverage_loss = float(coverage_loss.item())
        self._last_repulsion_loss = float(repulsion_loss.item())

        return total_loss

    # ------------------------------------------------------------------
    # DrJit evaluation barrier (unchanged)
    # ------------------------------------------------------------------

    def _eval_barrier_radio_map(
        self,
        samples_per_tx: int = 1_000_000,
        max_depth: int = 13,
    ) -> None:
        """Run a detached radio-map evaluation as a DrJit evaluation barrier."""
        dr.suspend_grad()
        try:
            _rm = self.radio_solver(
                self.scene,
                cell_size=(1.0, 1.0),
                samples_per_tx=samples_per_tx,
                max_depth=max_depth,
                refraction=True,
                diffraction=True,
            )
            _ = _rm.rss
        finally:
            dr.resume_grad()

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------

    def optimize(
        self,
        num_iterations: int = 50,
        learning_rate: float = 0.5,
        samples_per_tx: int = 1_000_000,
        max_depth: int = 13,
        use_soft_min: bool = True,
        temperature: float = 0.2,
        shadow_quantile: float = 0.05,
        fairness_loss_type: str = "auto",
        coverage_threshold_dbm: float = -120.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        coverage_temperature: float = 2.0,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Run gradient descent optimisation for all APs simultaneously.

        Args:
            num_iterations: Number of gradient steps.
            learning_rate: Base learning rate for position parameters.
            samples_per_tx: Ray samples per transmitter per iteration.
            max_depth: Maximum ray bounces.
            use_soft_min: **Deprecated** — use ``fairness_loss_type`` instead.
                Kept for backward compatibility.  When ``fairness_loss_type``
                is ``"auto"`` and ``use_soft_min=False``, the percentile loss
                is used.
            temperature: Softmin temperature (τ) for fairness loss.  Lower →
                sharper approximation of true min.
            shadow_quantile: Fraction of lowest-signal cells to mask out
                inside ``MaskedSoftMinLoss``.  Default 0.05 (5 %).
            fairness_loss_type: Fairness loss selector.  One of ``"auto"``,
                ``"softmin"``, ``"masked_softmin"``, ``"percentile"``.
                Default ``"auto"`` (masked_softmin if reflector present,
                else softmin).
            coverage_threshold_dbm: Coverage threshold in dBm.
            alpha: Weight for fairness (softmin) loss.  Default 0.6.
            beta: Weight for coverage (sigmoid) loss.  Default 0.4.
            coverage_temperature: Sigmoid temperature for coverage loss.
            verbose: Print per-iteration progress.

        Returns:
            final_positions: ``[3]`` for single AP or ``[num_aps, 3]``.
            final_min_rss: Best 5th-percentile RSS (linear Watts).
        """
        # ---- Parameter groups --------------------------------------------
        DIR_LR_MULTIPLIER = 10.0
        REFLECTOR_LR_MULTIPLIER = 0.5
        pos_lr = learning_rate / self.pos_range
        param_groups = [
            {"params": [self.tx_x, self.tx_y], "lr": pos_lr},
        ]
        if self.optimize_orientation:
            param_groups.append(
                {"params": [self.tx_dir_xy], "lr": learning_rate * DIR_LR_MULTIPLIER},
            )

        # Register reflector trainable parameters
        if self.reflector_controller is not None:
            reflector_params: list = [self.reflector_u_raw, self.reflector_v_raw]
            if self.focal_point_raw is not None:
                reflector_params.append(self.focal_point_raw)
            param_groups.append(
                {"params": reflector_params, "lr": learning_rate * REFLECTOR_LR_MULTIPLIER},
            )

        optimizer = torch.optim.AdamW(param_groups)

        # ---- Verbose header ----------------------------------------------
        if verbose:
            init_dirs = self.get_current_directions().detach().cpu().numpy()
            init_pos = self.get_full_positions()
            print("Starting Gradient Descent Optimization (PyTorch + DrJit)")
            print(f"  Device: {self.device}")
            print(f"  Number of APs: {self.num_aps}")
            for k in range(self.num_aps):
                print(
                    f"  AP{k} initial position:  "
                    f"({init_pos[k, 0]:.2f}, {init_pos[k, 1]:.2f}, {self.fixed_z:.2f})"
                )
                print(
                    f"  AP{k} initial direction: "
                    f"({init_dirs[k, 0]:.4f}, {init_dirs[k, 1]:.4f}, {init_dirs[k, 2]:.4f})"
                )
            print(f"  Optimize orientation: {self.optimize_orientation}")
            print(f"  LR (position):  {pos_lr:.4f}  (lr={learning_rate} / range={self.pos_range:.0f})")
            if self.optimize_orientation:
                print(f"  LR (direction): {learning_rate * DIR_LR_MULTIPLIER} ({DIR_LR_MULTIPLIER}x)")
            if self.num_aps > 1:
                print(f"  Repulsion weight: {self.repulsion_weight}")
            if self.reflector_controller is not None:
                refl_snap = self._snapshot_reflector()
                print(f"  Reflector: ENABLED")
                print(f"    Initial u={refl_snap['u']:.4f}, v={refl_snap['v']:.4f}")
                if refl_snap["focal_point"] is not None:
                    fp = refl_snap["focal_point"]
                    print(f"    Initial focal point: ({fp[0]:.2f}, {fp[1]:.2f}, {fp[2]:.2f})")
                print(f"    LR (reflector): {learning_rate * REFLECTOR_LR_MULTIPLIER}")
                print(f"    Shadow quantile (MaskedSoftMinLoss): {shadow_quantile}")
            print(f"  Iterations: {num_iterations}")
            print(f"  Samples per iteration: {samples_per_tx}")
            print(f"  Use soft minimum: {use_soft_min}")
            # Resolve display name for fairness loss
            _display_type = fairness_loss_type
            if _display_type == "auto" and not use_soft_min:
                _display_type = "percentile"
            elif _display_type == "auto":
                _display_type = (
                    "masked_softmin"
                    if self.reflector_controller is not None
                    else "softmin"
                )
            print(f"  Fairness loss: {_display_type} (temperature={temperature})")
            print(f"  Loss weights: alpha={alpha} (fairness), beta={beta} (coverage)")
            print(f"  Coverage threshold: {coverage_threshold_dbm} dBm")
            print(f"  Coverage temperature: {coverage_temperature}")
            print("-" * 80)

        start_time = time.time()

        for iteration in range(num_iterations):
            iter_start = time.time()

            # ---- Forward pass --------------------------------------------
            optimizer.zero_grad()
            loss = self.compute_loss(
                samples_per_tx=samples_per_tx,
                max_depth=max_depth,
                use_soft_min=use_soft_min,
                temperature=temperature,
                shadow_quantile=shadow_quantile,
                alpha=alpha,
                beta=beta,
                coverage_threshold_dbm=coverage_threshold_dbm,
                coverage_temperature=coverage_temperature,
                fairness_loss_type=fairness_loss_type,
            )

            # ---- Detached radio-map for metrics --------------------------
            with torch.no_grad():
                cur_dirs = self.get_current_directions()
                phys_pos = self.get_full_positions()
                tx_list = [self.scene.transmitters[n] for n in self.tx_names]
                for i in range(self.num_aps):
                    tx_list[i].position = [
                        float(phys_pos[i, 0]),
                        float(phys_pos[i, 1]),
                        self.fixed_z,
                    ]
                    target = [
                        float(phys_pos[i, 0]) + float(cur_dirs[i, 0].item()),
                        float(phys_pos[i, 1]) + float(cur_dirs[i, 1].item()),
                        self.fixed_z + float(cur_dirs[i, 2].item()),
                    ]
                    tx_list[i].look_at(target)

                rm = self.radio_solver(
                    self.scene,
                    cell_size=(1.0, 1.0),
                    samples_per_tx=samples_per_tx,
                    max_depth=max_depth,
                    refraction=True,
                    diffraction=True,
                )
                rss = torch.from_numpy(np.array(rm.rss))

            # ---- Backward pass -------------------------------------------
            loss.backward()

            # ---- SPSA reflector gradients --------------------------------
            # SceneObject vertex manipulation is incompatible with DrJit AD,
            # so reflector gradients are estimated numerically via SPSA
            # (2 extra forward passes per iteration).
            if self.reflector_controller is not None:
                self._reflector_spsa_gradients(
                    samples_per_tx=samples_per_tx,
                    max_depth=max_depth,
                    use_soft_min=use_soft_min,
                    temperature=temperature,
                    shadow_quantile=shadow_quantile,
                    alpha=alpha,
                    beta=beta,
                    coverage_threshold_dbm=coverage_threshold_dbm,
                    coverage_temperature=coverage_temperature,
                    fairness_loss_type=fairness_loss_type,
                )

            # ---- Extract gradients ---------------------------------------
            grad_x = (
                self.tx_x.grad.detach().cpu().numpy()
                if self.tx_x.grad is not None
                else np.zeros(self.num_aps)
            )
            grad_y = (
                self.tx_y.grad.detach().cpu().numpy()
                if self.tx_y.grad is not None
                else np.zeros(self.num_aps)
            )
            if self.optimize_orientation and self.tx_dir_xy.grad is not None:
                dir_grad = self.tx_dir_xy.grad.detach().cpu().numpy()  # [N, 2]
                dir_grad_full = np.concatenate(
                    [dir_grad, np.zeros((self.num_aps, 1))], axis=1,
                )  # [N, 3]
            else:
                dir_grad_full = np.zeros((self.num_aps, 3))

            # ---- NaN guard -----------------------------------------------
            nan_detected = False
            if self.tx_x.grad is not None and torch.any(torch.isnan(self.tx_x.grad)):
                self.tx_x.grad[torch.isnan(self.tx_x.grad)] = 0.0
                nan_detected = True
            if self.tx_y.grad is not None and torch.any(torch.isnan(self.tx_y.grad)):
                self.tx_y.grad[torch.isnan(self.tx_y.grad)] = 0.0
                nan_detected = True
            if self.optimize_orientation and self.tx_dir_xy.grad is not None:
                if torch.any(torch.isnan(self.tx_dir_xy.grad)):
                    self.tx_dir_xy.grad[torch.isnan(self.tx_dir_xy.grad)] = 0.0
                    nan_detected = True
            # NaN guard for reflector parameters
            if self.reflector_controller is not None:
                for rp in [self.reflector_u_raw, self.reflector_v_raw, self.focal_point_raw]:
                    if rp is not None and rp.grad is not None and torch.any(torch.isnan(rp.grad)):
                        rp.grad[torch.isnan(rp.grad)] = 0.0
                        nan_detected = True
            if nan_detected and verbose:
                print(f"WARNING: NaN gradients at iteration {iteration + 1}, zeroed out")

            all_pos_grad_zero = (
                np.all(np.abs(grad_x) < 1e-12) and np.all(np.abs(grad_y) < 1e-12)
            )
            if all_pos_grad_zero and not nan_detected and verbose:
                print(f"WARNING: Position gradients near zero at iteration {iteration + 1}")

            # ---- Step ---------------------------------------------------
            optimizer.step()
            self.apply_position_constraints()

            # ---- NaN position guard --------------------------------------
            with torch.no_grad():
                nan_x = torch.isnan(self.tx_x)
                nan_y = torch.isnan(self.tx_y)
                if torch.any(nan_x) or torch.any(nan_y):
                    if self.history["positions"]:
                        last_pos = self.history["positions"][-1]
                        last_pos = np.atleast_2d(last_pos)  # ensure [N, 3]
                        for k in range(self.num_aps):
                            if nan_x[k]:
                                self.tx_x[k] = self._normalize(float(last_pos[k, 0]))
                            if nan_y[k]:
                                self.tx_y[k] = self._normalize(float(last_pos[k, 1]))
                    else:
                        self.tx_x[nan_x] = 0.5
                        self.tx_y[nan_y] = 0.5
                    if verbose:
                        print(f"WARNING: NaN position at iteration {iteration + 1}, reset")

            # ---- Metrics -------------------------------------------------
            min_rss = compute_p5_rss_metric(rss.cpu())
            min_rss_dbm = rss_to_dbm(min_rss)
            coverage = compute_coverage_metric(rss, coverage_threshold_dbm)

            # ---- Store history -------------------------------------------
            current_pos = self.get_full_positions()       # [N, 3]
            current_dir = self.get_current_directions().detach().cpu().numpy()  # [N, 3]
            look_at = current_pos + current_dir            # [N, 3]
            ap_dists = self._pairwise_distances()

            if self.num_aps == 1:
                # Backward-compat: store flat [3] arrays
                self.history["positions"].append(current_pos[0].copy())
                self.history["directions"].append(current_dir[0].copy())
                self.history["look_at_targets"].append(look_at[0].copy())
                self.history["gradients"].append([float(grad_x[0]), float(grad_y[0])])
                self.history["direction_gradients"].append(dir_grad_full[0].tolist())
            else:
                self.history["positions"].append(current_pos.copy())
                self.history["directions"].append(current_dir.copy())
                self.history["look_at_targets"].append(look_at.copy())
                self.history["gradients"].append(
                    [[float(grad_x[k]), float(grad_y[k])] for k in range(self.num_aps)]
                )
                self.history["direction_gradients"].append(dir_grad_full.tolist())

            self.history["min_rss_values"].append(min_rss.item())
            self.history["min_rss_dbm_values"].append(min_rss_dbm.item())
            self.history["coverage_values"].append(coverage.item())
            self.history["losses"].append(loss.item())
            self.history["fairness_losses"].append(self._last_fairness_loss)
            self.history["coverage_obj_losses"].append(self._last_coverage_obj_loss)
            self.history["coverage_losses"].append(self._last_coverage_loss)
            self.history["repulsion_losses"].append(self._last_repulsion_loss)
            self.history["ap_distances"].append(ap_dists)

            # Reflector history
            refl_snap = self._snapshot_reflector()
            self.history["reflector_u"].append(refl_snap["u"])
            self.history["reflector_v"].append(refl_snap["v"])
            self.history["reflector_focal_point"].append(refl_snap["focal_point"])
            self.history["reflector_position"].append(refl_snap["position"])

            iter_time = time.time() - iter_start

            # ---- Verbose logging -----------------------------------------
            if verbose:
                parts = [f"Iter {iteration + 1:3d}/{num_iterations}"]
                for k in range(self.num_aps):
                    parts.append(
                        f"AP{k}:({current_pos[k, 0]:.2f},{current_pos[k, 1]:.2f})"
                    )
                if self.num_aps > 1 and ap_dists:
                    dist_str = ",".join(f"{d:.1f}" for d in ap_dists)
                    parts.append(f"Dist:{dist_str}m")
                parts.append(f"P5RSS:{min_rss_dbm:.2f}dBm")
                parts.append(f"Cov:{coverage:.1f}%")
                parts.append(
                    f"Loss:{loss.item():.2e}"
                    f"(Fair:{self._last_fairness_loss:.2e},"
                    f"Cov:{self._last_coverage_obj_loss:.2e})"
                )
                if self.num_aps > 1:
                    parts.append(f"Rep:{self._last_repulsion_loss:.2e}")
                pos_grad_norm = float(np.sqrt(np.sum(grad_x ** 2 + grad_y ** 2)))
                parts.append(f"∇pos:{pos_grad_norm:.2e}")
                if self.reflector_controller is not None and refl_snap["u"] is not None:
                    parts.append(
                        f"Refl:u={refl_snap['u']:.3f},v={refl_snap['v']:.3f}"
                    )
                    # Show reflector gradient magnitude for diagnostics
                    refl_grad_norms = []
                    for rp_name, rp in [("u", self.reflector_u_raw),
                                        ("v", self.reflector_v_raw),
                                        ("fp", self.focal_point_raw)]:
                        if rp is not None and rp.grad is not None:
                            refl_grad_norms.append(
                                f"{rp_name}:{float(rp.grad.norm()):.2e}"
                            )
                        else:
                            refl_grad_norms.append(f"{rp_name}:None")
                    parts.append(f"∇refl:[{','.join(refl_grad_norms)}]")
                parts.append(f"{iter_time:.1f}s")
                print(" | ".join(parts))

        # ---- Summary -----------------------------------------------------
        elapsed_time = time.time() - start_time
        final_positions = self.get_full_positions()           # [N, 3]
        final_min_rss = self.history["min_rss_values"][-1]
        final_dirs = self.get_current_directions().detach().cpu().numpy()
        final_look_at = final_positions + final_dirs

        if verbose:
            print("-" * 80)
            print("Gradient Descent Complete!")
            for k in range(self.num_aps):
                init_p = np.atleast_2d(self.history["positions"][0])
                init_d = np.atleast_2d(self.history["directions"][0])
                init_la = np.atleast_2d(self.history["look_at_targets"][0])
                print(f"  AP{k}:")
                print(
                    f"    Init pos:  ({init_p[k, 0]:.2f}, {init_p[k, 1]:.2f}, "
                    f"{init_p[k, 2]:.2f})"
                )
                print(
                    f"    Final pos: ({final_positions[k, 0]:.2f}, "
                    f"{final_positions[k, 1]:.2f}, {final_positions[k, 2]:.2f})"
                )
                print(
                    f"    Init dir:  ({init_d[k, 0]:.4f}, {init_d[k, 1]:.4f}, "
                    f"{init_d[k, 2]:.4f})"
                )
                print(
                    f"    Final dir: ({final_dirs[k, 0]:.4f}, {final_dirs[k, 1]:.4f}, "
                    f"{final_dirs[k, 2]:.4f})"
                )
                print(
                    f"    Final look_at: ({final_look_at[k, 0]:.2f}, "
                    f"{final_look_at[k, 1]:.2f}, {final_look_at[k, 2]:.2f})"
                )
            if self.num_aps > 1:
                dists = self._pairwise_distances()
                for idx, (i, j) in enumerate(combinations(range(self.num_aps), 2)):
                    print(f"  AP{i}-AP{j} distance: {dists[idx]:.2f} m")
            if self.history["min_rss_values"]:
                best_idx = int(np.argmax(self.history["min_rss_values"]))
                print(f"  Best iteration: {best_idx + 1}")
                print(f"  Best P5 RSS: {self.history['min_rss_dbm_values'][best_idx]:.2f} dBm")
            print(f"  Initial P5 RSS: {self.history['min_rss_dbm_values'][0]:.2f} dBm")
            final_min_rss_dbm = rss_to_dbm(torch.tensor(final_min_rss)).item()
            print(f"  Final P5 RSS: {final_min_rss_dbm:.2f} dBm")
            print(
                f"  Improvement: "
                f"{self.history['min_rss_dbm_values'][-1] - self.history['min_rss_dbm_values'][0]:.2f} dB"
            )
            print(f"  Final coverage: {self.history['coverage_values'][-1]:.1f}%")
            if self.reflector_controller is not None:
                refl_final = self._snapshot_reflector()
                print(f"  Reflector final:")
                print(f"    u={refl_final['u']:.4f}, v={refl_final['v']:.4f}")
                if refl_final["focal_point"] is not None:
                    fp = refl_final["focal_point"]
                    print(f"    Focal point: ({fp[0]:.2f}, {fp[1]:.2f}, {fp[2]:.2f})")
                if refl_final["position"] is not None:
                    rp = refl_final["position"]
                    print(f"    Wall position: ({rp[0]:.2f}, {rp[1]:.2f}, {rp[2]:.2f})")
            print(f"  Total time: {elapsed_time:.2f}s")
            print(f"  Time per iteration: {elapsed_time / num_iterations:.2f}s")

        # Return shape depends on num_aps
        if self.num_aps == 1:
            return final_positions[0], final_min_rss
        return final_positions, final_min_rss

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_results(self, **kwargs) -> None:
        """Plot the optimisation results (delegates to trajectory plotter)."""
        self.plot_optimization_trajectory()

    def plot_optimization_trajectory(self) -> None:
        """Plot per-AP trajectories and convergence metrics."""
        # Normalise history to always be [num_iter, num_aps, 3]
        raw_positions = self.history["positions"]
        if len(raw_positions) == 0:
            print("No history to plot.")
            return

        positions = np.array(raw_positions)
        if positions.ndim == 2:
            # Single-AP legacy format [num_iter, 3] → [num_iter, 1, 3]
            positions = positions[:, np.newaxis, :]
        num_iter, N, _ = positions.shape

        directions = np.array(self.history["directions"])
        if directions.ndim == 2:
            directions = directions[:, np.newaxis, :]

        n_plots = 4 + (1 if N > 1 else 0)
        fig, axes = plt.subplots(2, 3 if N > 1 else 2, figsize=(18 if N > 1 else 14, 10))
        axes = axes.flatten()

        # --- 1. Position trajectory (per-AP) ---
        ax = axes[0]
        for k in range(N):
            c = _AP_COLORS[k % len(_AP_COLORS)]
            m = _AP_MARKERS[k % len(_AP_MARKERS)]
            ax.plot(
                positions[:, k, 0], positions[:, k, 1],
                color=c, marker=m, markersize=3, linewidth=1.2,
                label=f"AP{k}",
            )
            ax.plot(
                positions[0, k, 0], positions[0, k, 1],
                "o", color=c, markersize=10,
            )
            ax.plot(
                positions[-1, k, 0], positions[-1, k, 1],
                "*", color=c, markersize=14,
            )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("AP Position Trajectories")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        # --- 2. P5 RSS over iterations ---
        ax = axes[1]
        ax.plot(self.history["min_rss_dbm_values"], "b-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("5th Percentile RSS (dBm)")
        ax.set_title("5th Percentile RSS Evolution (Sum-Power)" if N > 1 else "5th Percentile RSS Evolution")
        ax.grid(True, alpha=0.3)

        # --- 3. Coverage ---
        ax = axes[2]
        ax.plot(self.history["coverage_values"], "g-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Coverage (%)")
        ax.set_title("Coverage Evolution")
        ax.grid(True, alpha=0.3)

        # --- 4. Gradient norms (per-AP) ---
        ax = axes[3]
        raw_grads = self.history["gradients"]
        grads = np.array(raw_grads)
        if grads.ndim == 2:
            grads = grads[:, np.newaxis, :]  # [iter, 1, 2]
        for k in range(N):
            c = _AP_COLORS[k % len(_AP_COLORS)]
            norms = np.sqrt(grads[:, k, 0] ** 2 + grads[:, k, 1] ** 2)
            ax.semilogy(norms, color=c, linewidth=1.5, label=f"∇pos AP{k}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Position Gradient Norms (log)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- 5. AP distance (only for multi-AP) ---
        if N > 1:
            ax = axes[4]
            ap_dist_history = self.history.get("ap_distances", [])
            if ap_dist_history and len(ap_dist_history[0]) > 0:
                dists_arr = np.array(ap_dist_history)
                pair_labels = [
                    f"AP{i}-AP{j}"
                    for i, j in combinations(range(N), 2)
                ]
                for p_idx, label in enumerate(pair_labels):
                    ax.plot(
                        dists_arr[:, p_idx], linewidth=2, label=label,
                    )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Distance (m)")
            ax.set_title("Inter-AP Distance")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

        # --- Separate figure: direction components ---
        if self.history.get("directions") and len(self.history["directions"]) > 0:
            fig2, dir_axes = plt.subplots(1, N, figsize=(6 * N, 4), squeeze=False)
            for k in range(N):
                ax2 = dir_axes[0, k]
                ax2.plot(directions[:, k, 0], label="dx", linewidth=2)
                ax2.plot(directions[:, k, 1], label="dy", linewidth=2)
                ax2.plot(directions[:, k, 2], label="dz", linewidth=2)
                ax2.set_xlabel("Iteration")
                ax2.set_ylabel("Direction Component")
                ax2.set_title(f"AP{k} Look-At Direction")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
