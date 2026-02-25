"""
Reflector mesh generation and controller for differentiable ray-tracing optimisation.

This module provides:

* :func:`create_flat_reflector_mesh` — procedural generation of a planar
  rectangular mesh suitable for Sionna's ``SceneObject``.
* :class:`ReflectorController` — stateful wrapper around a Sionna
  ``SceneObject`` that exposes differentiable placement/orientation
  controls for the optimisation loop.

GPU / CPU Memory Boundary
-------------------------
Sionna offloads all ray-tracing computation to the GPU via Mitsuba / Dr.Jit.
The ``ReflectorController`` therefore keeps *two* representations in sync:

1. **Optimisation tensors** (PyTorch, on ``self.device``) — these are the
   parameters the gradient-based optimiser reads and writes.  During parallel
   simulation replicas each worker receives its *own* controller instance so
   there is no cross-thread mutation on these tensors.
2. **Scene-graph state** (Mitsuba ``SceneObject.position`` /
   ``.orientation``) — updated from the PyTorch tensors immediately before
   each forward ray-trace call via :meth:`apply_to_scene`.  The Mitsuba
   backend copies the values into its own CUDA buffers, so the PyTorch
   tensors are not aliased during the trace.

Thread-safety is guaranteed by the *scene-per-worker* architecture: each
parallel replica owns an independent ``Scene`` + ``ReflectorController``
pair.  No locking is required because no state is shared.
"""

from __future__ import annotations

from typing import Optional, Tuple

import drjit as dr
import mitsuba as mi
import numpy as np
import sionna.rt
import torch


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

def create_flat_reflector_mesh(
    width: float = 2.0,
    height: float = 2.0,
    name: str = "reflector_mesh",
) -> mi.Mesh:
    """Create a flat rectangular Mitsuba mesh centred at the origin.

    The mesh lies in the y-z plane with its outward normal pointing along
    the **-x** direction (vertices at ``x = -0.01``).  This convention
    simplifies wall-mounting: translate / rotate the parent
    ``SceneObject`` to place the reflector flush against any wall.

    Parameters
    ----------
    width : float
        Extent along the local y-axis (metres).
    height : float
        Extent along the local z-axis (metres).
    name : str
        Internal Mitsuba mesh identifier.

    Returns
    -------
    mi.Mesh
        A two-triangle quad ready to be wrapped in a ``sionna.rt.SceneObject``.
    """
    w2, h2 = width / 2.0, height / 2.0

    vertices = np.array(
        [
            [-0.01, -w2, -h2],
            [-0.01,  w2, -h2],
            [-0.01,  w2,  h2],
            [-0.01, -w2,  h2],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.uint32,
    )

    mesh = mi.Mesh(
        name,
        vertex_count=len(vertices),
        face_count=len(faces),
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )

    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = dr.ravel(mi.Point3f(vertices.T))
    mesh_params["faces"] = dr.ravel(mi.Vector3u(faces.T))
    mesh_params.update()

    return mesh


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class ReflectorController:
    """High-level controller for a flat reflector inside a Sionna scene.

    The controller maintains *differentiable* PyTorch parameters that
    describe the reflector's placement on a wall surface and the 3-D focal
    point used for beam-forming orientation.  An explicit
    :meth:`apply_to_scene` step pushes the current parameter state into
    the Mitsuba scene graph so that the next ray-trace observes the
    updated geometry.

    Wall-surface parameterisation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The reflector's centre is constrained to an axis-aligned bounding box
    on a wall surface defined by two 3-D corner points:

    * ``wall_top_left``      — top-left corner ``(x1, y1, z_top)``
    * ``wall_bottom_right``  — bottom-right corner ``(x2, y2, z_bottom)``

    The bounding box is computed automatically:

    * ``origin``  = ``(x1, y1, z_bottom)`` — bottom-left corner
    * ``u_axis``  = ``(x2 - x1, y2 - y1, 0)`` — horizontal span
    * ``v_axis``  = ``(0, 0, z_top - z_bottom)`` — vertical span

    Two scalar parameters *u, v* ∈ [0, 1] index the position:

    .. math::

        \\mathbf{p} = \\mathbf{o} + u\\,\\mathbf{u} + v\\,\\mathbf{v}

    At ``(u=0, v=0)`` the reflector sits at the bottom-left corner;
    at ``(u=1, v=1)`` it reaches the top-right corner.

    Focal-point orientation
    ~~~~~~~~~~~~~~~~~~~~~~~
    ``focal_point`` is a 3-D coordinate toward which the mechanical tile
    array should focus reflected energy.  Together with a known transmitter
    position, the ideal surface normal is the bisector of the incoming and
    outgoing unit vectors (law of reflection).

    GPU / CPU contract
    ~~~~~~~~~~~~~~~~~~
    All internal tensors live on ``self.device`` (typically ``"cuda"``).
    Values are detached and moved to CPU only when writing into the Mitsuba
    scene graph inside :meth:`apply_to_scene`, which copies them into
    Dr.Jit's own device memory.  This prevents PyTorch ↔ Mitsuba pointer
    aliasing and keeps gradient tape intact.

    Parameters
    ----------
    reflector : sionna.rt.SceneObject
        The Sionna scene object wrapping the reflector mesh.
    wall_top_left : array-like, shape (3,), optional
        Top-left corner of the wall bounding box ``(x1, y1, z_top)``.
    wall_bottom_right : array-like, shape (3,), optional
        Bottom-right corner of the wall bounding box ``(x2, y2, z_bottom)``.
    tx_position : array-like, shape (3,), optional
        Transmitter position for reflection-normal calculations.
    focal_point : array-like, shape (3,), optional
        Initial focal point for tile beam-forming.
    device : str or torch.device
        Torch device for all internal tensors.  Must match the device
        used by the rest of the optimisation loop.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        reflector: sionna.rt.SceneObject,
        wall_top_left: Optional[np.ndarray] = None,
        wall_bottom_right: Optional[np.ndarray] = None,
        tx_position: Optional[np.ndarray] = None,
        focal_point: Optional[np.ndarray] = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.reflector = reflector
        self.device = torch.device(device)

        # Wall-surface basis (constant, non-differentiable) ----------------
        self._wall_origin: Optional[torch.Tensor] = None
        self._wall_u_axis: Optional[torch.Tensor] = None
        self._wall_v_axis: Optional[torch.Tensor] = None

        if wall_top_left is not None and wall_bottom_right is not None:
            self.set_wall_bounds(wall_top_left, wall_bottom_right)

        # Differentiable placement parameters on the wall (u, v) ----------
        self._u = torch.tensor(0.5, dtype=torch.float32, device=self.device, requires_grad=True)
        self._v = torch.tensor(0.5, dtype=torch.float32, device=self.device, requires_grad=True)

        # Orientation Euler angles (α, β, γ) — differentiable -------------
        self._orientation = torch.zeros(3, dtype=torch.float32, device=self.device, requires_grad=True)

        # Transmitter position (constant reference) ------------------------
        self._tx_position: Optional[torch.Tensor] = None
        if tx_position is not None:
            self.set_tx_position(tx_position)

        # Focal point for beam-forming (differentiable) --------------------
        self._focal_point: Optional[torch.Tensor] = None
        if focal_point is not None:
            self.set_focal_point(focal_point)

    # ------------------------------------------------------------------ #
    # Wall-surface helpers
    # ------------------------------------------------------------------ #

    def set_wall_bounds(
        self,
        top_left: np.ndarray | torch.Tensor,
        bottom_right: np.ndarray | torch.Tensor,
    ) -> None:
        """Define the wall placement area from two corner points.

        The bounding box is derived automatically so that ``u`` ∈ [0, 1]
        sweeps horizontally (left → right) and ``v`` ∈ [0, 1] sweeps
        vertically (bottom → top).

        Parameters
        ----------
        top_left : array-like, shape (3,)
            Top-left corner of the wall area ``(x1, y1, z_top)``.
        bottom_right : array-like, shape (3,)
            Bottom-right corner of the wall area ``(x2, y2, z_bottom)``.
        """
        tl = np.asarray(top_left, dtype=np.float32).flatten()
        br = np.asarray(bottom_right, dtype=np.float32).flatten()

        # origin = bottom-left corner
        origin = np.array([tl[0], tl[1], br[2]], dtype=np.float32)
        # u_axis = horizontal span (xy plane)
        u_axis = np.array([br[0] - tl[0], br[1] - tl[1], 0.0], dtype=np.float32)
        # v_axis = vertical span (z direction)
        v_axis = np.array([0.0, 0.0, tl[2] - br[2]], dtype=np.float32)

        self._wall_origin = self._to_const_tensor(origin)
        self._wall_u_axis = self._to_const_tensor(u_axis)
        self._wall_v_axis = self._to_const_tensor(v_axis)

    def set_tx_position(self, position: np.ndarray | torch.Tensor) -> None:
        """Set the transmitter position (constant, non-differentiable)."""
        self._tx_position = self._to_const_tensor(position)

    def set_focal_point(
        self,
        point: np.ndarray | torch.Tensor,
        *,
        requires_grad: bool = True,
    ) -> None:
        """Set the 3-D focal point for tile beam-forming.

        When ``requires_grad=True`` (default) the focal point participates in
        the backward pass so the optimiser can move the focus.
        """
        t = self._to_tensor(point)
        if requires_grad and not t.requires_grad:
            t = t.detach().requires_grad_(True)
        self._focal_point = t

    # ------------------------------------------------------------------ #
    # Parameter access (for the optimiser)
    # ------------------------------------------------------------------ #

    @property
    def u(self) -> torch.Tensor:
        """Lateral wall coordinate (scalar, differentiable)."""
        return self._u

    @u.setter
    def u(self, value: torch.Tensor) -> None:
        self._u = value

    @property
    def v(self) -> torch.Tensor:
        """Vertical wall coordinate (scalar, differentiable)."""
        return self._v

    @v.setter
    def v(self, value: torch.Tensor) -> None:
        self._v = value

    @property
    def orientation(self) -> torch.Tensor:
        """Euler angles [α, β, γ] (shape (3,), differentiable)."""
        return self._orientation

    @orientation.setter
    def orientation(self, value: torch.Tensor) -> None:
        self._orientation = value

    @property
    def focal_point(self) -> Optional[torch.Tensor]:
        """3-D focal point for tile beam-forming, or *None*."""
        return self._focal_point

    @property
    def tx_position(self) -> Optional[torch.Tensor]:
        """Transmitter position (constant reference), or *None*."""
        return self._tx_position

    def optimisable_parameters(self) -> list[torch.Tensor]:
        """Return the list of tensors the optimiser should track.

        Includes ``u``, ``v``, ``orientation``, and (if set) ``focal_point``.
        """
        params = [self._u, self._v, self._orientation]
        if self._focal_point is not None and self._focal_point.requires_grad:
            params.append(self._focal_point)
        return params

    # ------------------------------------------------------------------ #
    # Derived position from wall parameters
    # ------------------------------------------------------------------ #

    def wall_position(self) -> torch.Tensor:
        """Compute the 3-D position from (u, v) on the wall rectangle.

        Returns
        -------
        torch.Tensor, shape (3,)
            World-space position, on ``self.device``.

        Raises
        ------
        RuntimeError
            If the wall surface has not been configured.
        """
        if self._wall_origin is None:
            raise RuntimeError(
                "Wall bounds not configured. Call set_wall_bounds() first."
            )
        u_clamped = torch.clamp(self._u, 0.0, 1.0)
        v_clamped = torch.clamp(self._v, 0.0, 1.0)
        return self._wall_origin + u_clamped * self._wall_u_axis + v_clamped * self._wall_v_axis

    # ------------------------------------------------------------------ #
    # Reflection geometry
    # ------------------------------------------------------------------ #

    def compute_reflection_normal(
        self,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the ideal specular-reflection normal.

        The bisector of the incoming (Tx → reflector) and outgoing
        (reflector → target) unit vectors gives the surface normal that
        satisfies the law of reflection.

        Parameters
        ----------
        target : torch.Tensor, shape (3,), optional
            Outgoing target point.  Falls back to ``self.focal_point`` if
            not provided.

        Returns
        -------
        normal : torch.Tensor, shape (3,)
            Unit surface normal.
        vec_in : torch.Tensor, shape (3,)
            Normalised Tx → reflector direction.
        vec_out : torch.Tensor, shape (3,)
            Normalised reflector → target direction.

        Raises
        ------
        RuntimeError
            If transmitter position or target is unavailable.
        """
        if self._tx_position is None:
            raise RuntimeError("Transmitter position not set.")
        if target is None:
            target = self._focal_point
        if target is None:
            raise RuntimeError(
                "No target provided and focal_point is not set."
            )

        pos = self.wall_position()

        vec_to_tx = self._tx_position - pos
        vec_in = vec_to_tx / (torch.linalg.norm(vec_to_tx) + 1e-12)

        vec_to_target = target - pos
        vec_out = vec_to_target / (torch.linalg.norm(vec_to_target) + 1e-12)

        normal_raw = vec_in + vec_out
        normal = normal_raw / (torch.linalg.norm(normal_raw) + 1e-12)

        return normal, vec_in, vec_out

    def orient_to_target(
        self,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Orient the reflector so that its surface normal satisfies the law
        of reflection for the current Tx position and *target*.

        This uses Sionna's ``look_at()`` helper under the hood then reads
        back the resulting Euler angles into ``self._orientation``.

        Parameters
        ----------
        target : torch.Tensor, shape (3,), optional
            Falls back to ``self.focal_point``.

        Returns
        -------
        torch.Tensor, shape (3,)
            The normal vector that was set.
        """
        normal, _, _ = self.compute_reflection_normal(target)
        pos = self.wall_position()

        # CRITICAL: set the SceneObject position FIRST so that look_at()
        # computes the direction from the correct world-space location.
        # Without this, look_at() uses the stale mesh-origin position.
        pos_cpu = pos.detach().cpu().tolist()
        self.reflector.position = pos_cpu

        # Point along the normal from the reflector position.
        look_target = pos + normal * 3.0
        look_target_cpu = look_target.detach().cpu().tolist()

        self.reflector.look_at(mi.Point3f(look_target_cpu))

        # Read back the orientation that Sionna/Mitsuba computed.
        orient_np = np.array(self.reflector.orientation, dtype=np.float32).flatten()
        self._orientation = torch.tensor(
            orient_np, dtype=torch.float32, device=self.device, requires_grad=True,
        )
        return normal

    # ------------------------------------------------------------------ #
    # Scene-graph synchronisation
    # ------------------------------------------------------------------ #

    def apply_to_scene(self) -> None:
        """Push current parameter state into the Mitsuba scene graph.

        This **must** be called before every forward ray-trace so the
        renderer sees the latest reflector placement.

        The method detaches tensors and copies them to CPU before writing
        into the ``SceneObject``, avoiding any PyTorch ↔ Dr.Jit pointer
        aliasing.  Gradient tape on the source tensors is unaffected.
        """
        # Position ----------------------------------------------------------
        pos = self.wall_position() if self._wall_origin is not None else None
        if pos is not None:
            self.reflector.position = pos.detach().cpu().tolist()

        # Orientation -------------------------------------------------------
        orient = self._orientation.detach().cpu().tolist()
        self.reflector.orientation = orient

    # ------------------------------------------------------------------ #
    # Convenience: direct position / orientation (non-wall mode)
    # ------------------------------------------------------------------ #

    def set_position(self, x: float, y: float, z: float) -> None:
        """Set the reflector position directly (bypasses wall parameterisation)."""
        self.reflector.position = [float(x), float(y), float(z)]

    def get_position(self) -> np.ndarray:
        """Return the current Mitsuba scene-graph position as a NumPy array."""
        return np.asarray(self.reflector.position, dtype=np.float32).flatten()

    def set_orientation_euler(self, alpha: float, beta: float, gamma: float) -> None:
        """Set orientation directly using Euler angles (radians)."""
        self.reflector.orientation = [float(alpha), float(beta), float(gamma)]
        self._orientation = torch.tensor(
            [alpha, beta, gamma], dtype=torch.float32, device=self.device,
            requires_grad=True,
        )

    def get_orientation(self) -> np.ndarray:
        """Return the current Mitsuba scene-graph orientation as a NumPy array."""
        return np.asarray(self.reflector.orientation, dtype=np.float32).flatten()

    # ------------------------------------------------------------------ #
    # Bulk parameter vector (legacy / grid-search compat)
    # ------------------------------------------------------------------ #

    def set_params(self, params: np.ndarray) -> None:
        """Set position and orientation from a flat 6-element array.

        Parameters
        ----------
        params : array-like, shape (6,)
            ``[x, y, z, alpha, beta, gamma]``
        """
        params = np.asarray(params, dtype=np.float32).flatten()
        assert params.shape == (6,), "params must be [x, y, z, α, β, γ]"
        self.set_position(params[0], params[1], params[2])
        self.set_orientation_euler(params[3], params[4], params[5])

    def get_params(self) -> np.ndarray:
        """Return ``[x, y, z, alpha, beta, gamma]`` as a flat NumPy array."""
        return np.concatenate([self.get_position(), self.get_orientation()])

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _to_tensor(self, arr) -> torch.Tensor:
        """Convert *arr* to a float32 tensor on ``self.device``."""
        if isinstance(arr, torch.Tensor):
            return arr.to(dtype=torch.float32, device=self.device)
        return torch.tensor(
            np.asarray(arr, dtype=np.float32).flatten(),
            dtype=torch.float32,
            device=self.device,
        )

    def _to_const_tensor(self, arr) -> torch.Tensor:
        """Convert *arr* to a **non-differentiable** tensor on ``self.device``."""
        return self._to_tensor(arr).detach().requires_grad_(False)

    # ------------------------------------------------------------------ #
    # Pretty-printing
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:  # noqa: D105
        pos = self.get_position()
        orient = self.get_orientation()
        lines = [
            "ReflectorController(",
            f"  position  : {pos} m",
            f"  orient    : {np.degrees(orient)} deg",
            f"  u={self._u.item():.4f}  v={self._v.item():.4f}",
            f"  device    : {self.device}",
        ]
        if self._wall_origin is not None:
            o = self._wall_origin.cpu().numpy()
            u = self._wall_u_axis.cpu().numpy()
            v = self._wall_v_axis.cpu().numpy()
            tl = o + v              # top-left
            br = o + u              # bottom-right
            lines.append(f"  wall_top_left    : {tl}")
            lines.append(f"  wall_bottom_right: {br}")
        if self._tx_position is not None:
            t = self._tx_position.cpu().numpy()
            lines.append(f"  tx_position: {t}")
        if self._focal_point is not None:
            f = self._focal_point.detach().cpu().numpy()
            lines.append(f"  focal_point: {f}")
        lines.append(")")
        return "\n".join(lines)
