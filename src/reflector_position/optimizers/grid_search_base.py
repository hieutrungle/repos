"""
Grid search optimizer for AP position.

This module implements exhaustive grid search over a 2D spatial grid
to find optimal AP placement that maximizes minimum RSS.
"""

import time
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create grid using torch
        x_range = torch.arange(
            search_bounds["x_min"],
            search_bounds["x_max"] + grid_resolution,
            grid_resolution,
            device=self.device,
        )
        y_range = torch.arange(
            search_bounds["y_min"],
            search_bounds["y_max"] + grid_resolution,
            grid_resolution,
            device=self.device,
        )

        self.x_grid, self.y_grid = torch.meshgrid(x_range, y_range, indexing='ij')
        self.total_positions = self.x_grid.numel()

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
            print(f"  Device: {self.device}")
            print(f"  Grid size: {self.x_grid.shape}")
            print(f"  Total positions to evaluate: {self.total_positions}")
            print(
                f"  Search bounds: x=[{self.search_bounds['x_min']}, {self.search_bounds['x_max']}], "
                f"y=[{self.search_bounds['y_min']}, {self.search_bounds['y_max']}]"
            )
            print(f"  Samples per position: {samples_per_tx}")
            print("-" * 70)

        best_min_rss = torch.tensor(-float('inf'), device=self.device)
        best_position = None
        best_coverage = torch.tensor(0.0, device=self.device)

        # Flatten grid for iteration using torch
        positions = torch.stack(
            [
                self.x_grid.flatten(),
                self.y_grid.flatten(),
                torch.full((self.x_grid.numel(),), self.fixed_z, device=self.device),
            ],
            dim=1,
        )

        for idx, pos in enumerate(positions):
            # Compute radio map for this position
            tx_position = [float(pos[0].item()), float(pos[1].item()), float(pos[2].item())]

            try:
                rm = compute_radio_map_with_tx_position(
                    self.scene, tx_position, samples_per_tx=samples_per_tx, max_depth=max_depth
                )

                # Compute metrics using torch (keep on GPU)
                rss_tensor = torch.from_numpy(np.array(rm.rss)).to(self.device)
                min_rss = compute_min_rss_metric(rss_tensor)
                min_rss_dbm = rss_to_dbm(min_rss)
                coverage = compute_coverage_metric(rss_tensor, coverage_threshold_dbm)

                # Store results (convert to CPU only for storage)
                self.results["positions"].append(pos.cpu().numpy())
                self.results["min_rss_values"].append(min_rss.cpu().item())
                self.results["min_rss_dbm_values"].append(min_rss_dbm.cpu().item())
                self.results["coverage_values"].append(coverage.cpu().item())
                self.results["radio_maps"].append(rm)

                # Update best (keep comparison on GPU)
                if min_rss > best_min_rss:
                    best_min_rss = min_rss
                    best_position = pos.cpu().numpy()
                    best_coverage = coverage

                if verbose and (idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    eta = elapsed / (idx + 1) * (self.total_positions - idx - 1)
                    best_min_rss_dbm = rss_to_dbm(best_min_rss)
                    print(
                        f"Progress: {idx+1}/{self.total_positions} | "
                        f"Current: ({pos[0].item():.1f}, {pos[1].item():.1f}) -> {min_rss_dbm.item():.2f} dBm | "
                        f"Best: {best_min_rss_dbm.item():.2f} dBm | "
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
            best_min_rss_dbm = rss_to_dbm(best_min_rss)
            print(f"  Best min RSS: {best_min_rss_dbm.item():.2f} dBm")
            print(f"  Coverage: {best_coverage.item():.1f}%")
            print(f"  Total time: {elapsed_time:.2f}s")
            print(f"  Time per position: {elapsed_time/self.total_positions:.2f}s")

        return best_position, best_min_rss.cpu().item()

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

        # Convert grid to CPU for plotting
        x_grid_cpu = self.x_grid.cpu().numpy()
        y_grid_cpu = self.y_grid.cpu().numpy()

        plt.figure(figsize=(10, 8))
        im = plt.contourf(x_grid_cpu, y_grid_cpu, values_grid, levels=20, cmap=cmap)
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
