"""
Scene setup and configuration for reflector position optimization.

This module handles loading and configuring Sionna scenes with transmitters,
receivers, and an optional passive mechanical reflector.

GPU / CPU Memory Boundary
-------------------------
Scene construction happens on the CPU (Python / NumPy).  Once the scene is
returned to the caller the ray-tracing backend (Mitsuba / Dr.Jit) transfers
geometry to GPU memory on first use.  Each parallel simulation worker must
call this function independently to obtain its **own** ``Scene`` +
``ReflectorController`` pair — no scene-graph state is shared across
threads, which eliminates the need for locks and prevents CUDA memory
collisions during concurrent evaluations.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import sionna.rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver
import mitsuba as mi
from .reflector_model import (
    ReflectorController,
    create_flat_reflector_mesh,
)


def _coerce_position_bounds(
    position_bounds: Optional[Mapping[str, float]],
) -> Optional[Tuple[float, float, float, float]]:
    """Return (x_min, x_max, y_min, y_max) when bounds are valid."""
    if not isinstance(position_bounds, Mapping):
        return None

    required_keys = ("x_min", "x_max", "y_min", "y_max")
    if not all(key in position_bounds for key in required_keys):
        return None

    x_min = float(position_bounds["x_min"])
    x_max = float(position_bounds["x_max"])
    y_min = float(position_bounds["y_min"])
    y_max = float(position_bounds["y_max"])

    if x_min >= x_max or y_min >= y_max:
        raise ValueError("position_bounds must satisfy x_min < x_max and y_min < y_max")

    return x_min, x_max, y_min, y_max


def _resolve_tx_positions(
    tx_positions: Optional[List[Tuple[float, float, float]]],
    num_aps: Optional[int],
    position_bounds: Optional[Mapping[str, float]],
    default_z: float = 3.8,
) -> List[Tuple[float, float, float]]:
    """Resolve final transmitter positions used to instantiate scene TX nodes.

    Rules:
    - If ``num_aps`` is not provided, use all provided positions.
    - If provided positions >= ``num_aps``, keep the first ``num_aps``.
    - If provided positions < ``num_aps``, append new positions sequentially.
      Each appended XY is the mean of all existing XY positions plus the four
      position-bound corners.
    """
    raw_positions = tx_positions if tx_positions is not None else [(25.0, 25.0, default_z)]

    resolved: List[Tuple[float, float, float]] = []
    for index, position in enumerate(raw_positions):
        if not isinstance(position, (list, tuple, np.ndarray)) or len(position) != 3:
            raise ValueError(
                f"tx_positions[{index}] must be a length-3 sequence (x, y, z)"
            )
        resolved.append((float(position[0]), float(position[1]), float(position[2])))

    if not resolved:
        resolved = [(25.0, 25.0, float(default_z))]

    if num_aps is None:
        return resolved

    target_num_aps = int(num_aps)
    if target_num_aps <= 0:
        raise ValueError(f"num_aps must be positive, got {target_num_aps}")

    if len(resolved) >= target_num_aps:
        return resolved[:target_num_aps]

    bounds = _coerce_position_bounds(position_bounds)
    if bounds is None:
        raise ValueError(
            "position_bounds with keys x_min/x_max/y_min/y_max is required when "
            "num_aps exceeds available tx_positions"
        )

    x_min, x_max, y_min, y_max = bounds
    corners_xy = [
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_min),
        (x_max, y_max),
    ]

    while len(resolved) < target_num_aps:
        xs = [position[0] for position in resolved] + [corner[0] for corner in corners_xy]
        ys = [position[1] for position in resolved] + [corner[1] for corner in corners_xy]
        z_value = float(resolved[-1][2]) if resolved else float(default_z)
        resolved.append((float(np.mean(xs)), float(np.mean(ys)), z_value))

    return resolved


def setup_building_floor_scene(
    scene_path: str,
    frequency: float = 5.18e9,
    tx_positions: Optional[List[Tuple[float, float, float]]] = None,
    num_aps: Optional[int] = None,
    position_bounds: Optional[Mapping[str, float]] = None,
    tx_power_dbm: float = 5.0,
    rx_position: Tuple[float, float, float] = (16.0, 16.5, 1.5),
    # --- reflector parameters (all optional) ---
    reflector_enabled: bool = False,
    reflector_size: Tuple[float, float] = (2.0, 2.0),
    wall_top_left: Optional[Union[np.ndarray, List[float]]] = None,
    wall_bottom_right: Optional[Union[np.ndarray, List[float]]] = None,
    focal_point: Optional[Union[np.ndarray, List[float]]] = None,
    device: str = "cuda",
) -> Union[sionna.rt.Scene, Tuple[sionna.rt.Scene, ReflectorController]]:
    """Setup the building floor scene with transmitters, receivers and an
    optional passive reflector.

    When ``reflector_enabled=True`` the function creates a flat rectangular
    metal reflector, wraps it in a :class:`ReflectorController`, and returns
    both the scene and the controller so the optimiser can interact with the
    reflector in later phases.

    Parameters
    ----------
    scene_path : str
        Path to the Mitsuba/Sionna XML scene file.
    frequency : float
        Operating frequency in Hz (default 5.18 GHz).
    tx_positions : list of (float, float, float), optional
        Transmitter world positions.  Defaults to ``[(10, 20, 3.8)]``.
    num_aps : int, optional
        Number of AP transmitters to instantiate in the scene.
        If smaller than ``len(tx_positions)``, only the first ``num_aps``
        positions are used.
        If larger than ``len(tx_positions)``, additional positions are
        generated sequentially using the mean of existing positions and the
        four corners from ``position_bounds``.
    position_bounds : mapping, optional
        Bounds with keys ``x_min``, ``x_max``, ``y_min``, ``y_max`` used when
        auto-generating additional transmitter positions.
    tx_power_dbm : float
        Total transmitter power in dBm, split equally across APs.
    rx_position : tuple of float
        Receiver position ``(x, y, z)``.
    reflector_enabled : bool
        If *True* a reflector mesh + controller are created and returned.
    reflector_size : tuple of float
        ``(width, height)`` of the reflector in metres.
    wall_top_left : array-like, shape (3,), optional
        Top-left corner ``(x1, y1, z_top)`` of the wall bounding box the
        reflector can slide on.
    wall_bottom_right : array-like, shape (3,), optional
        Bottom-right corner ``(x2, y2, z_bottom)`` of the wall bounding box.
    focal_point : array-like, shape (3,), optional
        Initial 3-D focal point for beam-forming orientation.
    device : str
        PyTorch device for differentiable controller tensors.

    Returns
    -------
    scene : sionna.rt.Scene
        The fully configured scene (always returned).
    controller : ReflectorController
        Only returned when ``reflector_enabled=True``.  The caller
        receives a ``(scene, controller)`` tuple in that case.
    """
    # ------------------------------------------------------------------
    # 1. Load scene & set frequency
    # ------------------------------------------------------------------
    scene = load_scene(scene_path)
    scene.frequency = frequency

    speed_of_light = 3e8  # m/s
    wavelength = speed_of_light / frequency  # noqa: F841 (kept for downstream use)

    # ------------------------------------------------------------------
    # 2. Transmitter array
    # ------------------------------------------------------------------
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=2,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="VH",
    )

    resolved_tx_positions = _resolve_tx_positions(
        tx_positions=tx_positions,
        num_aps=num_aps,
        position_bounds=position_bounds,
        default_z=3.8,
    )

    n_txs = 1  # power-split kept at 1 AP for legacy compat
    power_per_tx = tx_power_dbm / n_txs

    for i, (x, y, z) in enumerate(resolved_tx_positions):
        # yaw = i * 2 * np.pi / n_txs
        tx = Transmitter(
            name=f"Tx{i:02d}",
            position=mi.Point3f([x, y, z]),
            orientation=mi.Point3f([0, 0, 0]),
            power_dbm=int(power_per_tx),
        )
        scene.add(tx)

    # ------------------------------------------------------------------
    # 3. Receiver array
    # ------------------------------------------------------------------
    # scene.rx_array = PlanarArray(
    #     num_rows=2,
    #     num_cols=2,
    #     vertical_spacing=0.5,
    #     horizontal_spacing=0.5,
    #     pattern="iso",
    #     polarization="VH",
    # )

    # rx = Receiver(name="Rx", position=mi.Point3f(list(rx_position)), orientation=mi.Point3f([0, 0, 0]))
    # scene.add(rx)

    # ------------------------------------------------------------------
    # 4. (Optional) Passive Reflector
    # ------------------------------------------------------------------
    if not reflector_enabled:
        return scene

    # 4a. Material — ITU metal, highly reflective
    reflector_material = sionna.rt.ITURadioMaterial(
        name="reflector_metal",
        itu_type="metal",
        thickness=0.002,  # 2 mm metal plate
    )

    # 4b. Mesh
    r_width, r_height = reflector_size
    mesh = create_flat_reflector_mesh(width=r_width, height=r_height)

    # 4c. SceneObject
    reflector_obj = sionna.rt.SceneObject(
        mi_mesh=mesh,
        name="reflector",
        radio_material=reflector_material,
    )
    scene.edit(add=reflector_obj)

    # 4d. Controller
    tx_pos_arr = np.asarray(resolved_tx_positions[0], dtype=np.float32)

    controller = ReflectorController(
        reflector=reflector_obj,
        wall_top_left=np.asarray(wall_top_left, dtype=np.float32),
        wall_bottom_right=np.asarray(wall_bottom_right, dtype=np.float32),
        tx_position=tx_pos_arr,
        focal_point=np.asarray(focal_point, dtype=np.float32),
        device=device,
    )

    return scene, controller


def create_camera(
    position: Tuple[float, float, float] = (20.0, 20.0, 70.0),
    look_at: Tuple[float, float, float] = (20.0, 20.1, 1.5),
) -> sionna.rt.Camera:
    """Create a camera for scene visualization.

    Parameters
    ----------
    position : tuple of float
        Camera position ``(x, y, z)``.
    look_at : tuple of float
        Point the camera looks at ``(x, y, z)``.

    Returns
    -------
    sionna.rt.Camera
    """
    return sionna.rt.Camera(position=mi.Point3f(list(position)), look_at=mi.Point3f(list(look_at)))
