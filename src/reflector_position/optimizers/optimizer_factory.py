"""
Factory for creating optimizer instances.

This module provides a factory pattern for easily instantiating different
optimization methods with a consistent interface.
"""

from typing import Dict, Optional, Type
import sionna.rt

from .base_optimizer import BaseAPOptimizer
from .gradient_descent import GradientDescentAPOptimizer
from .grid_search import SinglePointGridSearchOptimizer


class OptimizerFactory:
    """
    Factory for creating optimizer instances.
    
    Provides a centralized way to create different types of optimizers
    with consistent configuration.
    """
    
    # Registry of available optimizers
    _optimizers: Dict[str, Type[BaseAPOptimizer]] = {
        "gradient_descent": GradientDescentAPOptimizer,
        "grid_search_point": SinglePointGridSearchOptimizer,
    }
    
    @classmethod
    def create(
        cls,
        method: str,
        scene: sionna.rt.Scene,
        **kwargs
    ) -> BaseAPOptimizer:
        """
        Create an optimizer instance.
        
        Args:
            method: Optimization method name ('gradient_descent', 'grid_search')
            scene: Sionna Scene object
            **kwargs: Method-specific initialization parameters
            
        Returns:
            Initialized optimizer instance
            
        Raises:
            ValueError: If method is not recognized
            
        Examples:
            >>> # Create gradient descent optimizer
            >>> optimizer = OptimizerFactory.create(
            ...     method="gradient_descent",
            ...     scene=scene,
            ...     initial_position=(10.0, 10.0),
            ...     position_bounds={'x_min': 0, 'x_max': 20, 'y_min': 0, 'y_max': 20}
            ... )
            
            >>> # Create grid search optimizer
            >>> optimizer = OptimizerFactory.create(
            ...     method="grid_search",
            ...     scene=scene,
            ...     search_bounds={'x_min': 0, 'x_max': 20, 'y_min': 0, 'y_max': 20},
            ...     grid_resolution=2.0
            ... )
        """
        method = method.lower().replace("-", "_")
        
        if method not in cls._optimizers:
            available = ", ".join(cls._optimizers.keys())
            raise ValueError(
                f"Unknown optimization method: '{method}'. "
                f"Available methods: {available}"
            )
        
        optimizer_class = cls._optimizers[method]
        return optimizer_class(scene=scene, **kwargs)
    
    @classmethod
    def register(cls, name: str, optimizer_class: Type[BaseAPOptimizer]) -> None:
        """
        Register a new optimizer type.
        
        Allows extending the factory with custom optimizer implementations.
        
        Args:
            name: Name for the optimizer method
            optimizer_class: Optimizer class (must inherit from BaseAPOptimizer)
            
        Raises:
            TypeError: If optimizer_class doesn't inherit from BaseAPOptimizer
            
        Example:
            >>> class MyCustomOptimizer(BaseAPOptimizer):
            ...     # Implementation
            ...     pass
            >>> OptimizerFactory.register("custom", MyCustomOptimizer)
        """
        if not issubclass(optimizer_class, BaseAPOptimizer):
            raise TypeError(
                f"{optimizer_class.__name__} must inherit from BaseAPOptimizer"
            )
        
        cls._optimizers[name.lower().replace("-", "_")] = optimizer_class
    
    @classmethod
    def list_methods(cls) -> list:
        """
        Get list of available optimization methods.
        
        Returns:
            List of registered method names
        """
        return list(cls._optimizers.keys())


# Convenience function for quick creation
def create_optimizer(
    method: str,
    scene: sionna.rt.Scene,
    **kwargs
) -> BaseAPOptimizer:
    """
    Convenience function to create an optimizer.
    
    This is a shorthand for OptimizerFactory.create().
    
    Args:
        method: Optimization method name
        scene: Sionna Scene object
        **kwargs: Method-specific parameters
        
    Returns:
        Initialized optimizer instance
        
    Example:
        >>> optimizer = create_optimizer(
        ...     method="gradient-descent",
        ...     scene=scene,
        ...     initial_position=(10.0, 10.0)
        ... )
    """
    return OptimizerFactory.create(method=method, scene=scene, **kwargs)
