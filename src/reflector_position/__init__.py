"""
Reflector Position Optimization Package.

Physics-aware optimal placement for mechanical reflectors in NLOS scenarios
using differentiable ray tracing with Sionna.
"""

__version__ = "0.1.0"

from .optimizers import GridSearchAPOptimizer, GradientDescentAPOptimizer
from .scene_setup import setup_building_floor_scene, create_camera
from .metrics import (
    compute_min_rss_metric,
    compute_soft_min_rss_metric,
    compute_coverage_metric,
    rss_to_dbm,
)
from .utils import compute_radio_map_with_tx_position
from .config import (
    SceneConfig,
    GridSearchConfig,
    GradientDescentConfig,
    OptimizationConfig,
)

__all__ = [
    "__version__",
    # Optimizers
    "GridSearchAPOptimizer",
    "GradientDescentAPOptimizer",
    # Scene setup
    "setup_building_floor_scene",
    "create_camera",
    # Metrics
    "compute_min_rss_metric",
    "compute_soft_min_rss_metric",
    "compute_coverage_metric",
    "rss_to_dbm",
    # Utils
    "compute_radio_map_with_tx_position",
    # Config
    "SceneConfig",
    "GridSearchConfig",
    "GradientDescentConfig",
    "OptimizationConfig",
]
