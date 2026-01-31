"""
Configuration management for reflector position optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SceneConfig:
    """Configuration for scene setup."""

    scene_path: str
    frequency: float = 5.18e9  # 5.18 GHz
    tx_positions: List[Tuple[float, float, float]] = field(default_factory=lambda: [(10.0, 20.0, 3.8)])
    tx_power_dbm: float = 5.0
    rx_position: Tuple[float, float, float] = (16.0, 6.5, 1.5)


@dataclass
class GridSearchConfig:
    """Configuration for grid search optimization."""

    x_min: float = 5.0
    x_max: float = 35.0
    y_min: float = 5.0
    y_max: float = 35.0
    grid_resolution: float = 5.0
    fixed_z: float = 3.8
    samples_per_tx: int = 500_000
    max_depth: int = 13
    coverage_threshold_dbm: float = -100.0

    @property
    def search_bounds(self) -> Dict[str, float]:
        """Get search bounds as dictionary."""
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }


@dataclass
class GradientDescentConfig:
    """Configuration for gradient descent optimization."""

    initial_x: float = 20.0
    initial_y: float = 20.0
    x_min: float = 5.0
    x_max: float = 35.0
    y_min: float = 5.0
    y_max: float = 35.0
    fixed_z: float = 3.8
    num_iterations: int = 10
    learning_rate: float = 0.5
    samples_per_tx: int = 1_000_000
    max_depth: int = 15
    use_soft_min: bool = True
    temperature: float = 0.2
    coverage_threshold_dbm: float = -100.0

    @property
    def initial_position(self) -> Tuple[float, float]:
        """Get initial position as tuple."""
        return (self.initial_x, self.initial_y)

    @property
    def position_bounds(self) -> Dict[str, float]:
        """Get position bounds as dictionary."""
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }


@dataclass
class OptimizationConfig:
    """Main configuration container."""

    scene: SceneConfig
    grid_search: GridSearchConfig = field(default_factory=GridSearchConfig)
    gradient_descent: GradientDescentConfig = field(default_factory=GradientDescentConfig)
