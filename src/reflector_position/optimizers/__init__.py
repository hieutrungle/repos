"""
Optimizers package for AP position optimization.
"""

from .base_optimizer import BaseAPOptimizer
from .grid_search import (
    SinglePointGridSearchOptimizer,
    generate_grid_positions,
    generate_alternating_grid_tasks,
    CARDINAL_DIRECTIONS,
)
from .gradient_descent import GradientDescentAPOptimizer
from .optimizer_factory import OptimizerFactory, create_optimizer
from .ray_parallel_optimizer import (
    RayParallelOptimizer,
    OptimizationWorker,
    generate_random_initial_positions,
)
from .ray_evaluator import RayActorPoolExecutor
from .deap_logic import GeneticAlgorithmRunner

__all__ = [
    "BaseAPOptimizer",
    "SinglePointGridSearchOptimizer",
    "generate_grid_positions",
    "generate_alternating_grid_tasks",
    "CARDINAL_DIRECTIONS",
    "GradientDescentAPOptimizer",
    "OptimizerFactory",
    "create_optimizer",
    # Ray parallel
    "RayParallelOptimizer",
    "OptimizationWorker",
    "generate_random_initial_positions",
    # DEAP GA (modular IoC pattern)
    "RayActorPoolExecutor",
    "GeneticAlgorithmRunner",
]
