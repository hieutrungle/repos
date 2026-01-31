"""
Utility functions for radio map computation and scene manipulation.
"""

from typing import Tuple

import sionna.rt
from sionna.rt import RadioMapSolver, RadioMap


def compute_radio_map_with_tx_position(
    scene: sionna.rt.Scene,
    tx_position: Tuple[float, float, float],
    cell_size: Tuple[float, float] = (1.0, 1.0),
    samples_per_tx: int = 1_000_000,
    max_depth: int = 15,
) -> RadioMap:
    """
    Compute radio map for a given transmitter position.

    Args:
        scene: Sionna scene object
        tx_position: [x, y, z] position of transmitter
        cell_size: Radio map cell size in meters (width, height)
        samples_per_tx: Number of ray tracing samples
        max_depth: Maximum number of reflections/diffractions

    Returns:
        RadioMap object with RSS values
    """
    # Update transmitter position
    for tx in scene.transmitters.values():
        tx.position = tx_position

    # Compute radio map
    solver = RadioMapSolver()
    rm = solver(
        scene,
        cell_size=cell_size,
        samples_per_tx=samples_per_tx,
        max_depth=max_depth,
        refraction=True,
        diffraction=True,
    )

    return rm
