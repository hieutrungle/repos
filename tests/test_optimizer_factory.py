"""
Unit tests for OptimizerFactory.

Tests factory pattern functionality including creation, registration, and error handling.
"""

import pytest
import numpy as np

from reflector_position.optimizers import (
    BaseAPOptimizer,
    OptimizerFactory,
    create_optimizer,
    GradientDescentAPOptimizer,
    GridSearchAPOptimizer,
)


@pytest.mark.unit
class TestOptimizerFactory:
    """Test suite for OptimizerFactory."""
    
    def test_list_methods(self):
        """Test that factory lists available methods."""
        methods = OptimizerFactory.list_methods()
        
        assert isinstance(methods, list)
        assert len(methods) >= 2
        assert "gradient_descent" in methods
        assert "grid_search" in methods
    
    def test_create_gradient_descent(self, test_scene, initial_position, position_bounds):
        """Test creating gradient descent optimizer."""
        optimizer = OptimizerFactory.create(
            method="gradient_descent",
            scene=test_scene,
            initial_position=initial_position,
            position_bounds=position_bounds,
        )
        
        assert isinstance(optimizer, GradientDescentAPOptimizer)
        assert isinstance(optimizer, BaseAPOptimizer)
    
    def test_create_grid_search(self, test_scene, position_bounds):
        """Test creating grid search optimizer."""
        optimizer = OptimizerFactory.create(
            method="grid_search",
            scene=test_scene,
            search_bounds=position_bounds,
            grid_resolution=5.0,
        )
        
        assert isinstance(optimizer, GridSearchAPOptimizer)
        assert isinstance(optimizer, BaseAPOptimizer)
    
    def test_create_with_hyphens(self, test_scene, initial_position):
        """Test that method names with hyphens are converted."""
        optimizer = OptimizerFactory.create(
            method="gradient-descent",  # Hyphenated
            scene=test_scene,
            initial_position=initial_position,
        )
        
        assert isinstance(optimizer, GradientDescentAPOptimizer)
    
    def test_create_case_insensitive(self, test_scene, initial_position):
        """Test that method names are case-insensitive."""
        optimizer1 = OptimizerFactory.create(
            method="GRADIENT_DESCENT",
            scene=test_scene,
            initial_position=initial_position,
        )
        
        optimizer2 = OptimizerFactory.create(
            method="Gradient_Descent",
            scene=test_scene,
            initial_position=initial_position,
        )
        
        assert isinstance(optimizer1, GradientDescentAPOptimizer)
        assert isinstance(optimizer2, GradientDescentAPOptimizer)
    
    def test_create_invalid_method_raises_error(self, test_scene):
        """Test that invalid method name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OptimizerFactory.create(
                method="invalid_method",
                scene=test_scene,
            )
        
        assert "Unknown optimization method" in str(exc_info.value)
        assert "invalid_method" in str(exc_info.value)
    
    def test_register_custom_optimizer(self, test_scene):
        """Test registering a custom optimizer."""
        
        class CustomOptimizer(BaseAPOptimizer):
            def optimize(self, **kwargs):
                return np.array([0, 0, 0]), 0.0
            
            def plot_results(self, **kwargs):
                pass
        
        # Register
        OptimizerFactory.register("custom_test", CustomOptimizer)
        
        # Verify it's in the list
        methods = OptimizerFactory.list_methods()
        assert "custom_test" in methods
        
        # Create instance
        optimizer = OptimizerFactory.create(
            method="custom_test",
            scene=test_scene,
        )
        
        assert isinstance(optimizer, CustomOptimizer)
        assert isinstance(optimizer, BaseAPOptimizer)
    
    def test_register_non_optimizer_raises_error(self):
        """Test that registering non-BaseAPOptimizer class raises TypeError."""
        
        class NotAnOptimizer:
            pass
        
        with pytest.raises(TypeError) as exc_info:
            OptimizerFactory.register("bad_optimizer", NotAnOptimizer)
        
        assert "must inherit from BaseAPOptimizer" in str(exc_info.value)
    
    def test_convenience_function(self, test_scene, initial_position):
        """Test create_optimizer convenience function."""
        optimizer = create_optimizer(
            method="gradient_descent",
            scene=test_scene,
            initial_position=initial_position,
        )
        
        assert isinstance(optimizer, GradientDescentAPOptimizer)
        assert isinstance(optimizer, BaseAPOptimizer)
    
    def test_pass_kwargs_to_optimizer(self, test_scene, initial_position):
        """Test that kwargs are passed to optimizer constructor."""
        fixed_z = 5.0
        
        optimizer = OptimizerFactory.create(
            method="gradient_descent",
            scene=test_scene,
            initial_position=initial_position,
            fixed_z=fixed_z,
        )
        
        assert optimizer.fixed_z == fixed_z
    
    def test_multiple_registrations_same_name(self, test_scene):
        """Test that re-registering overwrites previous registration."""
        
        class Optimizer1(BaseAPOptimizer):
            def optimize(self, **kwargs):
                return np.array([1, 1, 1]), 1.0
            def plot_results(self, **kwargs):
                pass
        
        class Optimizer2(BaseAPOptimizer):
            def optimize(self, **kwargs):
                return np.array([2, 2, 2]), 2.0
            def plot_results(self, **kwargs):
                pass
        
        OptimizerFactory.register("overwrite_test", Optimizer1)
        OptimizerFactory.register("overwrite_test", Optimizer2)
        
        optimizer = OptimizerFactory.create(
            method="overwrite_test",
            scene=test_scene,
        )
        
        # Should be Optimizer2 (the last registered)
        assert isinstance(optimizer, Optimizer2)
        assert not isinstance(optimizer, Optimizer1)
