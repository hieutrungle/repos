"""
Grid search optimizer for AP position.

This module implements exhaustive grid search over a 2D spatial grid
to find optimal AP placement that maximizes minimum RSS.
"""

import time
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import sionna.rt

from ..utils import compute_radio_map_with_tx_position
from ..metrics import compute_min_rss_metric, compute_coverage_metric, rss_to_dbm


class GridSearchAPOptimizer:
    """
    Grid search optimizer for AP position.

    Exhaustively evaluates all positions on a grid to find the optimal AP placement.
    """

    def __init__(
        self,
        scene: sionna.rt.Scene,
        search_bounds: Dict[str, float],
        grid_resolution: float = 1.0,
        fixed_z: float = 3.8,
    ):
        """
        Initialize grid search optimizer.

        Args:
            scene: Sionna Scene object
            search_bounds: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max'
            grid_resolution: Grid spacing in meters
            fixed_z: Fixed height for AP (z-coordinate)
        """
        self.scene = scene
        self.search_bounds = search_bounds
        self.grid_resolution = grid_resolution
        self.fixed_z = fixed_z

        # Create grid
        x_range = np.arange(
            search_bounds["x_min"],
            search_bounds["x_max"] + grid_resolution,
            grid_resolution,
        )
        y_range = np.arange(
            search_bounds["y_min"],
            search_bounds["y_max"] + grid_resolution,
            grid_resolution,
        )

        self.x_grid, self.y_grid = np.meshgrid(x_range, y_range)
        self.total_positions = self.x_grid.size

        # Storage for results
        self.results = {
            "positions": [],
            "min_rss_values": [],
            "min_rss_dbm_values": [],
            "coverage_values": [],
            "radio_maps": [],
        }

    def optimize(
        self,
        samples_per_tx: int = 1_000_000,
        max_depth: int = 13,
        coverage_threshold_dbm: float = -100.0,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Run grid search optimization.

        Args:
            samples_per_tx: Number of ray tracing samples per position
            max_depth: Maximum ray tracing depth
            coverage_threshold_dbm: Threshold for coverage calculation
            verbose: Print progress

        Returns:
            best_position: [x, y, z] of best AP position
            best_min_rss: Best minimum RSS value found
        """
        start_time = time.time()

        if verbose:
            print(f"Starting Grid Search Optimization")
            print(f"  Grid size: {self.x_grid.shape}")
            print(f"  Total positions to evaluate: {self.total_positions}")
            print(
                f"  Search bounds: x=[{self.search_bounds['x_min']}, {self.search_bounds['x_max']}], "
                f"y=[{self.search_bounds['y_min']}, {self.search_bounds['y_max']}]"
            )
            print(f"  Samples per position: {samples_per_tx}")
            print("-" * 70)

        best_min_rss = -np.inf
        best_position = None
        best_coverage = 0.0

        # Flatten grid for iteration
        positions = np.stack(
            [
                self.x_grid.flatten(),
                self.y_grid.flatten(),
                np.full(self.x_grid.size, self.fixed_z),
            ],
            axis=1,
        )

        for idx, pos in enumerate(positions):
            # Compute radio map for this position
            tx_position = [float(pos[0]), float(pos[1]), float(pos[2])]

            try:
                rm = compute_radio_map_with_tx_position(
                    self.scene, tx_position, samples_per_tx=samples_per_tx, max_depth=max_depth
                )

                # Compute metrics
                min_rss = compute_min_rss_metric(rm.rss)
                min_rss_dbm = 10 * np.log10(min_rss.numpy() + 1e-12) + 30.0
                coverage = compute_coverage_metric(rm.rss, coverage_threshold_dbm)

                # Store results
                self.results["positions"].append(pos)
                self.results["min_rss_values"].append(min_rss.numpy())
                self.results["min_rss_dbm_values"].append(min_rss_dbm)
                self.results["coverage_values"].append(coverage.numpy())
                self.results["radio_maps"].append(rm)

                # Update best
                if min_rss.numpy() > best_min_rss:
                    best_min_rss = min_rss.numpy()
                    best_position = pos
                    best_coverage = coverage.numpy()

                if verbose and (idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    eta = elapsed / (idx + 1) * (self.total_positions - idx - 1)
                    best_min_rss_dbm = 10 * np.log10(best_min_rss + 1e-12) + 30.0
                    print(
                        f"Progress: {idx+1}/{self.total_positions} | "
                        f"Current: ({pos[0]:.1f}, {pos[1]:.1f}) -> {min_rss_dbm:.2f} dBm | "
                        f"Best: {best_min_rss_dbm:.2f} dBm | "
                        f"ETA: {eta:.1f}s"
                    )

            except Exception as e:
                if verbose:
                    print(f"  Error at position {pos}: {e}")
                continue

        elapsed_time = time.time() - start_time

        if verbose:
            print("-" * 70)
            print(f"Grid Search Complete!")
            print(
                f"  Best position: ({best_position[0]:.2f}, {best_position[1]:.2f}, {best_position[2]:.2f})"
            )
            best_min_rss_dbm = 10 * np.log10(best_min_rss + 1e-12) + 30.0
            print(f"  Best min RSS: {best_min_rss_dbm:.2f} dBm")
            print(f"  Coverage: {best_coverage:.1f}%")
            print(f"  Total time: {elapsed_time:.2f}s")
            print(f"  Time per position: {elapsed_time/self.total_positions:.2f}s")

        return best_position, best_min_rss

    def plot_results(self, metric: str = "min_rss_dbm") -> None:
        """
        Plot grid search results as a heatmap.

        Args:
            metric: 'min_rss_dbm' or 'coverage'
        """
        if metric == "min_rss_dbm":
            values = np.array(self.results["min_rss_dbm_values"])
            title = "Grid Search: Minimum RSS (dBm)"
            cmap = "viridis"
            label = "Min RSS (dBm)"
        elif metric == "coverage":
            values = np.array(self.results["coverage_values"])
            title = "Grid Search: Coverage (%)"
            cmap = "RdYlGn"
            label = "Coverage (%)"
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Reshape to grid
        values_grid = values.reshape(self.x_grid.shape)

        plt.figure(figsize=(10, 8))
        im = plt.contourf(self.x_grid, self.y_grid, values_grid, levels=20, cmap=cmap)
        plt.colorbar(im, label=label)

        # Mark best position
        best_idx = np.argmax(self.results["min_rss_values"])
        best_pos = self.results["positions"][best_idx]
        plt.plot(
            best_pos[0],
            best_pos[1],
            "r*",
            markersize=20,
            label=f"Best: ({best_pos[0]:.1f}, {best_pos[1]:.1f})",
        )

        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
