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

from typing import List, Optional, Tuple, Union

import numpy as np
import sionna.rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver

from .reflector_model import (
    ReflectorController,
    create_flat_reflector_mesh,
)


def setup_building_floor_scene(
    scene_path: str,
    frequency: float = 5.18e9,
    tx_positions: Optional[List[Tuple[float, float, float]]] = None,
    tx_power_dbm: float = 5.0,
    rx_position: Tuple[float, float, float] = (16.0, 6.5, 1.5),
    # --- reflector parameters (all optional) ---
    reflector_enabled: bool = False,
    reflector_size: Tuple[float, float] = (2.0, 2.0),
    wall_origin: Optional[Union[np.ndarray, List[float]]] = None,
    wall_u_axis: Optional[Union[np.ndarray, List[float]]] = None,
    wall_v_axis: Optional[Union[np.ndarray, List[float]]] = None,
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
    tx_power_dbm : float
        Total transmitter power in dBm, split equally across APs.
    rx_position : tuple of float
        Receiver position ``(x, y, z)``.
    reflector_enabled : bool
        If *True* a reflector mesh + controller are created and returned.
    reflector_size : tuple of float
        ``(width, height)`` of the reflector in metres.
    wall_origin : array-like, shape (3,), optional
        Corner of the wall rectangle the reflector can slide on.
    wall_u_axis : array-like, shape (3,), optional
        Lateral basis vector of the wall rectangle.
    wall_v_axis : array-like, shape (3,), optional
        Vertical basis vector of the wall rectangle.
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
        num_rows=2,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="VH",
    )

    if tx_positions is None:
        tx_positions = [(10.0, 20.0, 3.8)]

    n_txs = 1  # power-split kept at 1 AP for legacy compat
    power_per_tx = tx_power_dbm / n_txs

    for i, (x, y, z) in enumerate(tx_positions):
        yaw = i * 2 * np.pi / n_txs
        tx = Transmitter(
            name=f"Tx{i:02d}",
            position=[x, y, z],
            orientation=[yaw, 0, 0],
            power_dbm=power_per_tx,
        )
        scene.add(tx)

    # ------------------------------------------------------------------
    # 3. Receiver array
    # ------------------------------------------------------------------
    scene.rx_array = PlanarArray(
        num_rows=2,
        num_cols=2,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="VH",
    )

    rx = Receiver(name="Rx", position=list(rx_position), orientation=[0, 0, 0])
    scene.add(rx)

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
    tx_pos_arr = np.asarray(tx_positions[0], dtype=np.float32)

    controller = ReflectorController(
        reflector=reflector_obj,
        wall_origin=np.asarray(wall_origin, dtype=np.float32) if wall_origin is not None else None,
        wall_u_axis=np.asarray(wall_u_axis, dtype=np.float32) if wall_u_axis is not None else None,
        wall_v_axis=np.asarray(wall_v_axis, dtype=np.float32) if wall_v_axis is not None else None,
        tx_position=tx_pos_arr,
        focal_point=np.asarray(focal_point, dtype=np.float32) if focal_point is not None else None,
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
    return sionna.rt.Camera(position=list(position), look_at=list(look_at))
