"""
Abstract base class for AP position optimizers.

This module defines the interface that all optimization methods must implement,
providing a consistent API for different optimization strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import numpy as np
import sionna.rt


class BaseAPOptimizer(ABC):
    """
    Abstract base class for AP position optimization.
    
    All optimization methods (gradient descent, grid search, genetic algorithm, etc.)
    should inherit from this class and implement the required abstract methods.
    """
    
    def __init__(
        self,
        scene: sionna.rt.Scene,
        fixed_z: float = 3.8,
        position_bounds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize base optimizer.
        
        Args:
            scene: Sionna Scene object
            fixed_z: Fixed height for AP (z-coordinate)
            position_bounds: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max' for constraints
        """
        self.scene = scene
        self.fixed_z = fixed_z
        self.position_bounds = position_bounds or {}
        
    @abstractmethod
    def optimize(
        self,
        samples_per_tx: int = 1_000_000,
        max_depth: int = 13,
        verbose: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, float]:
        """
        Run optimization to find best AP position.
        
        Args:
            samples_per_tx: Number of ray tracing samples per evaluation
            max_depth: Maximum ray tracing depth
            verbose: Print progress information
            **kwargs: Method-specific parameters
            
        Returns:
            best_position: Optimized AP position [x, y, z]
            best_metric: Best metric value achieved (e.g., min RSS)
        """
        pass
    
    @abstractmethod
    def plot_results(self, **kwargs) -> None:
        """
        Visualize optimization results.
        
        Args:
            **kwargs: Method-specific plotting parameters
        """
        pass
    
    def get_position_bounds(self) -> Dict[str, float]:
        """Get position bounds if defined."""
        return self.position_bounds.copy()
    
    def validate_position(self, position: np.ndarray) -> bool:
        """
        Check if a position is within bounds.
        
        Args:
            position: Position array [x, y, z] or [x, y]
            
        Returns:
            True if position is valid, False otherwise
        """
        if not self.position_bounds:
            return True
            
        x, y = position[0], position[1]
        
        return (
            self.position_bounds.get("x_min", -float('inf')) <= x <= self.position_bounds.get("x_max", float('inf'))
            and self.position_bounds.get("y_min", -float('inf')) <= y <= self.position_bounds.get("y_max", float('inf'))
        )
    
    def get_full_position(self, x: float, y: float) -> np.ndarray:
        """
        Convert 2D position to full 3D position with fixed z.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Full 3D position [x, y, z]
        """
        return np.array([x, y, self.fixed_z])
