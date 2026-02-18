"""
Scene setup and configuration for reflector position optimization.

This module handles loading and configuring Sionna scenes with transmitters and receivers.
"""

from typing import List, Tuple

import numpy as np
import sionna.rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver


def setup_building_floor_scene(
    scene_path: str,
    frequency: float = 5.18e9,
    tx_positions: List[Tuple[float, float, float]] = None,
    tx_power_dbm: float = 5.0,
    rx_position: Tuple[float, float, float] = (16.0, 6.5, 1.5),
) -> sionna.rt.Scene:
    """
    Setup the building floor scene with transmitters and receivers.

    Args:
        scene_path: Path to the scene XML file
        frequency: Operating frequency in Hz (default: 5.18 GHz)
        tx_positions: List of transmitter positions [(x, y, z), ...]
                     If None, uses default position at (10, 20, 3.8)
        tx_power_dbm: Total transmitter power in dBm
        rx_position: Receiver position (x, y, z)

    Returns:
        Configured Sionna Scene object
    """
    # Load the scene
    scene = load_scene(scene_path)
    scene.frequency = frequency

    # Speed of light for wavelength calculation
    speed_of_light = 3e8  # m/s
    wavelength = speed_of_light / frequency

    # Configure transmitter array
    scene.tx_array = PlanarArray(
        num_rows=2,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="VH",
    )

    # Set default transmitter positions if not provided
    if tx_positions is None:
        tx_positions = [(10.0, 20.0, 3.8)]

    n_txs = 1
    # n_txs = len(tx_positions)
    power_per_tx = tx_power_dbm / n_txs

    # Add transmitters
    for i, (x, y, z) in enumerate(tx_positions):
        # Calculate orientation to spread coverage
        yaw = i * 2 * np.pi / n_txs
        tx = Transmitter(
            name=f"Tx{i:02d}",
            position=[x, y, z],
            orientation=[yaw, 0, 0],
            power_dbm=power_per_tx,
        )
        scene.add(tx)

    # Configure receiver array
    scene.rx_array = PlanarArray(
        num_rows=2,
        num_cols=2,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="VH",
    )

    # Add receiver
    rx = Receiver(name="Rx", position=list(rx_position), orientation=[0, 0, 0])
    scene.add(rx)

    return scene


def create_camera(
    position: Tuple[float, float, float] = (20.0, 20.0, 50.0),
    look_at: Tuple[float, float, float] = (20.0, 20.1, 1.5),
) -> sionna.rt.Camera:
    """
    Create a camera for scene visualization.

    Args:
        position: Camera position (x, y, z)
        look_at: Point the camera looks at (x, y, z)

    Returns:
        Sionna Camera object
    """
    return sionna.rt.Camera(position=list(position), look_at=list(look_at))
