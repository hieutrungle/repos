"""
Unit tests for BaseAPOptimizer abstract base class.

Tests the interface contract that all optimizers must follow.
"""

import pytest
import numpy as np
from abc import ABC

from reflector_position.optimizers import BaseAPOptimizer


class ConcreteOptimizer(BaseAPOptimizer):
    """Concrete implementation for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimize_called = False
        self.plot_called = False
    
    def optimize(self, samples_per_tx=1_000_000, max_depth=13, verbose=True, **kwargs):
        self.optimize_called = True
        return np.array([10.0, 10.0, self.fixed_z]), -60.0
    
    def plot_results(self, **kwargs):
        self.plot_called = True


@pytest.mark.unit
class TestBaseAPOptimizer:
    """Test suite for BaseAPOptimizer abstract base class."""
    
    def test_is_abstract(self):
        """Test that BaseAPOptimizer is an abstract base class."""
        assert issubclass(BaseAPOptimizer, ABC)
    
    def test_cannot_instantiate_directly(self):
        """Test that BaseAPOptimizer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAPOptimizer(scene=None)
    
    def test_requires_optimize_implementation(self, test_scene):
        """Test that subclasses must implement optimize()."""
        
        class IncompleteOptimizer(BaseAPOptimizer):
            def plot_results(self, **kwargs):
                pass
        
        with pytest.raises(TypeError):
            IncompleteOptimizer(scene=test_scene)
    
    def test_requires_plot_results_implementation(self, test_scene):
        """Test that subclasses must implement plot_results()."""
        
        class IncompleteOptimizer(BaseAPOptimizer):
            def optimize(self, **kwargs):
                return np.array([0, 0, 0]), 0.0
        
        with pytest.raises(TypeError):
            IncompleteOptimizer(scene=test_scene)
    
    def test_concrete_implementation_works(self, test_scene):
        """Test that a complete implementation can be instantiated."""
        optimizer = ConcreteOptimizer(scene=test_scene, fixed_z=3.8)
        assert isinstance(optimizer, BaseAPOptimizer)
    
    def test_get_position_bounds(self, test_scene, position_bounds):
        """Test get_position_bounds() method."""
        optimizer = ConcreteOptimizer(
            scene=test_scene,
            position_bounds=position_bounds
        )
        
        bounds = optimizer.get_position_bounds()
        assert bounds == position_bounds
        assert bounds is not position_bounds  # Should be a copy
    
    def test_get_position_bounds_empty(self, test_scene):
        """Test get_position_bounds() with no bounds set."""
        optimizer = ConcreteOptimizer(scene=test_scene)
        bounds = optimizer.get_position_bounds()
        assert bounds == {}
    
    def test_validate_position_within_bounds(self, test_scene, position_bounds):
        """Test validate_position() with position inside bounds."""
        optimizer = ConcreteOptimizer(
            scene=test_scene,
            position_bounds=position_bounds
        )
        
        # Position inside bounds
        assert optimizer.validate_position(np.array([10.0, 10.0, 3.8])) == True
        assert optimizer.validate_position(np.array([10.0, 10.0])) == True
        assert optimizer.validate_position(np.array([25.0, 25.0])) == True
    
    def test_validate_position_outside_bounds(self, test_scene, position_bounds):
        """Test validate_position() with position outside bounds."""
        optimizer = ConcreteOptimizer(
            scene=test_scene,
            position_bounds=position_bounds
        )
        
        # Positions outside bounds
        assert optimizer.validate_position(np.array([4.0, 10.0])) == False  # x too low
        assert optimizer.validate_position(np.array([26.0, 10.0])) == False  # x too high
        assert optimizer.validate_position(np.array([10.0, 4.0])) == False  # y too low
        assert optimizer.validate_position(np.array([10.0, 26.0])) == False  # y too high
    
    def test_validate_position_no_bounds(self, test_scene):
        """Test validate_position() with no bounds (should always return True)."""
        optimizer = ConcreteOptimizer(scene=test_scene)
        
        assert optimizer.validate_position(np.array([100.0, 100.0])) is True
        assert optimizer.validate_position(np.array([-100.0, -100.0])) is True
    
    def test_get_full_position(self, test_scene):
        """Test get_full_position() method."""
        fixed_z = 3.8
        optimizer = ConcreteOptimizer(scene=test_scene, fixed_z=fixed_z)
        
        x, y = 10.0, 15.0
        position = optimizer.get_full_position(x, y)
        
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
        assert position[0] == x
        assert position[1] == y
        assert position[2] == fixed_z
    
    def test_optimize_method_called(self, test_scene):
        """Test that optimize() method can be called."""
        optimizer = ConcreteOptimizer(scene=test_scene)
        
        position, rss = optimizer.optimize()
        
        assert optimizer.optimize_called is True
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
        assert isinstance(rss, float)
    
    def test_plot_results_method_called(self, test_scene):
        """Test that plot_results() method can be called."""
        optimizer = ConcreteOptimizer(scene=test_scene)
        
        optimizer.plot_results()
        
        assert optimizer.plot_called is True
    
    def test_fixed_z_stored(self, test_scene):
        """Test that fixed_z is stored correctly."""
        fixed_z = 5.5
        optimizer = ConcreteOptimizer(scene=test_scene, fixed_z=fixed_z)
        assert optimizer.fixed_z == fixed_z
    
    def test_scene_stored(self, test_scene):
        """Test that scene is stored correctly."""
        optimizer = ConcreteOptimizer(scene=test_scene)
        assert optimizer.scene is test_scene
    
    def test_position_bounds_stored(self, test_scene, position_bounds):
        """Test that position_bounds is stored correctly."""
        optimizer = ConcreteOptimizer(
            scene=test_scene,
            position_bounds=position_bounds
        )
        assert optimizer.position_bounds == position_bounds
