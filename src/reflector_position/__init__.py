"""
Reflector Position Optimization Package.

Physics-aware optimal placement for mechanical reflectors in NLOS scenarios
using differentiable ray tracing with Sionna.
"""

__version__ = "0.1.0"

from .optimizers import (
    BaseAPOptimizer,
    SinglePointGridSearchOptimizer,
    generate_alternating_grid_tasks,
    GradientDescentAPOptimizer,
    OptimizerFactory,
    create_optimizer,
)
from .scene_setup import setup_building_floor_scene, create_camera
from .metrics import (
    POWER_EPSILON,
    compute_min_rss_metric,
    compute_soft_min_rss_metric,
    normalized_softmin_loss,
    compute_coverage_metric,
    rss_to_dbm,
    dbm_to_rss,
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
    "BaseAPOptimizer",
    "SinglePointGridSearchOptimizer",
    "generate_alternating_grid_tasks",
    "GradientDescentAPOptimizer",
    "OptimizerFactory",
    "create_optimizer",
    # Scene setup
    "setup_building_floor_scene",
    "create_camera",
    # Metrics
    "POWER_EPSILON",
    "compute_min_rss_metric",
    "compute_soft_min_rss_metric",
    "normalized_softmin_loss",
    "compute_coverage_metric",
    "rss_to_dbm",
    "dbm_to_rss",
    # Utils
    "compute_radio_map_with_tx_position",
    # Config
    "SceneConfig",
    "GridSearchConfig",
    "GradientDescentConfig",
    "OptimizationConfig",
]
