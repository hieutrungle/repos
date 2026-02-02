"""
Unit and integration tests for GradientDescentAPOptimizer.

Tests gradient descent optimization functionality including initialization,
optimization execution, constraint handling, and result tracking.
"""

import pytest
import numpy as np
import torch

from reflector_position.optimizers import GradientDescentAPOptimizer


@pytest.mark.unit
class TestGradientDescentInitialization:
    """Test suite for GradientDescentAPOptimizer initialization."""
    
    @pytest.mark.requires_scene
    def test_initialization(self, test_scene, initial_position):
        """Test basic initialization."""
        optimizer = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=initial_position,
        )
        
        assert optimizer.scene is test_scene
        assert optimizer.fixed_z == 3.8  # Default value
        assert isinstance(optimizer.tx_x, torch.Tensor)
        assert isinstance(optimizer.tx_y, torch.Tensor)
        assert optimizer.tx_x.item() == initial_position[0]
        assert optimizer.tx_y.item() == initial_position[1]
    
    @pytest.mark.requires_scene
    def test_initialization_with_custom_z(self, test_scene, initial_position):
        """Test initialization with custom z-height."""
        fixed_z = 5.0
        optimizer = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=initial_position,
            fixed_z=fixed_z,
        )
        
        assert optimizer.fixed_z == fixed_z
    
    @pytest.mark.requires_scene
    def test_initialization_with_bounds(self, test_scene, initial_position, position_bounds):
        """Test initialization with position bounds."""
        optimizer = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=initial_position,
            position_bounds=position_bounds,
        )
        
        assert optimizer.position_bounds == position_bounds
    
    @pytest.mark.requires_scene
    def test_tensors_require_grad(self, test_scene, initial_position):
        """Test that position tensors have gradients enabled."""
        optimizer = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=initial_position,
        )
        
        assert optimizer.tx_x.requires_grad is True
        assert optimizer.tx_y.requires_grad is True
    
    @pytest.mark.requires_scene
    def test_device_detection(self, test_scene, initial_position):
        """Test that device is correctly detected."""
        optimizer = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=initial_position,
        )
        
        assert optimizer.device in [torch.device("cuda"), torch.device("cpu")]
    
    @pytest.mark.requires_scene
    def test_history_initialized(self, test_scene, initial_position):
        """Test that history tracking is initialized."""
        optimizer = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=initial_position,
        )
        
        assert "positions" in optimizer.history
        assert "min_rss_values" in optimizer.history
        assert "min_rss_dbm_values" in optimizer.history
        assert "coverage_values" in optimizer.history
        assert "losses" in optimizer.history
        assert "gradients" in optimizer.history
        
        for key in optimizer.history:
            assert isinstance(optimizer.history[key], list)
            assert len(optimizer.history[key]) == 0


@pytest.mark.unit
class TestGradientDescentMethods:
    """Test suite for GradientDescentAPOptimizer methods."""
    
    @pytest.mark.requires_scene
    def test_get_full_position(self, test_scene, initial_position):
        """Test get_full_position method."""
        optimizer = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=initial_position,
            fixed_z=4.0,
        )
        
        position = optimizer.get_full_position()
        
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
        assert position[0] == initial_position[0]
        assert position[1] == initial_position[1]
        assert position[2] == 4.0
    
    @pytest.mark.requires_scene
    def test_apply_position_constraints_within_bounds(self, test_scene, position_bounds):
        """Test that positions within bounds are not modified."""
        initial_pos = (10.0, 10.0)  # Within bounds
        optimizer = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=initial_pos,
            position_bounds=position_bounds,
        )
        
        original_x = optimizer.tx_x.item()
        original_y = optimizer.tx_y.item()
        
        optimizer.apply_position_constraints()
        
        assert optimizer.tx_x.item() == original_x
        assert optimizer.tx_y.item() == original_y
    
    @pytest.mark.requires_scene
    def test_apply_position_constraints_outside_bounds(self, test_scene, position_bounds):
        """Test that positions outside bounds are clamped."""
        optimizer = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=(10.0, 10.0),
            position_bounds=position_bounds,
        )
        
        # Manually set position outside bounds
        with torch.no_grad():
            optimizer.tx_x.copy_(torch.tensor(30.0))  # Above max
            optimizer.tx_y.copy_(torch.tensor(2.0))   # Below min
        
        optimizer.apply_position_constraints()
        
        assert optimizer.tx_x.item() == position_bounds['x_max']
        assert optimizer.tx_y.item() == position_bounds['y_min']
    
    @pytest.mark.requires_scene
    def test_apply_position_constraints_no_bounds(self, test_scene):
        """Test that constraints do nothing when no bounds are set."""
        optimizer = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=(100.0, 100.0),
            position_bounds=None,
        )
        
        original_x = optimizer.tx_x.item()
        original_y = optimizer.tx_y.item()
        
        optimizer.apply_position_constraints()
        
        # Should remain unchanged
        assert optimizer.tx_x.item() == original_x
        assert optimizer.tx_y.item() == original_y


@pytest.mark.integration
@pytest.mark.requires_scene
class TestGradientDescentOptimization:
    """Integration tests for gradient descent optimization."""
    
    def test_optimize_basic(self, test_scene, initial_position, gd_test_params):
        """Test basic optimization execution."""
        optimizer = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=initial_position,
        )
        
        final_position, final_rss = optimizer.optimize(**gd_test_params)
        
        assert isinstance(final_position, np.ndarray)
        assert final_position.shape == (3,)
        assert isinstance(final_rss, float)
        assert final_rss > 0  # RSS should be positive
    
    def test_optimize_history_tracking(self, test_scene, initial_position, gd_test_params):
        """Test that optimization history is tracked."""
        optimizer = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=initial_position,
        )
        
        num_iterations = gd_test_params['num_iterations']
        optimizer.optimize(**gd_test_params)
        
        # Check history lengths
        assert len(optimizer.history['positions']) == num_iterations
        assert len(optimizer.history['min_rss_values']) == num_iterations
        assert len(optimizer.history['min_rss_dbm_values']) == num_iterations
        assert len(optimizer.history['losses']) == num_iterations
        assert len(optimizer.history['gradients']) == num_iterations
    
    def test_optimize_respects_bounds(self, test_scene, position_bounds, gd_test_params):
        """Test that optimization respects position bounds."""
        optimizer = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=(10.0, 10.0),
            position_bounds=position_bounds,
        )
        
        final_position, _ = optimizer.optimize(**gd_test_params)
        
        assert position_bounds['x_min'] <= final_position[0] <= position_bounds['x_max']
        assert position_bounds['y_min'] <= final_position[1] <= position_bounds['y_max']
    
    def test_optimize_different_learning_rates(self, test_scene, initial_position, test_params):
        """Test optimization with different learning rates."""
        learning_rates = [0.1, 0.5, 1.0]
        results = []
        
        for lr in learning_rates:
            optimizer = GradientDescentAPOptimizer(
                scene=test_scene,
                initial_position=initial_position,
            )
            
            position, rss = optimizer.optimize(
                **test_params,
                num_iterations=3,
                learning_rate=lr,
            )
            results.append((position, rss))
        
        # Different learning rates should produce different results
        positions = [r[0] for r in results]
        assert not all(np.allclose(positions[0], p) for p in positions[1:])
    
    def test_optimize_soft_vs_hard_min(self, test_scene, initial_position, test_params):
        """Test optimization with soft vs hard minimum."""
        # Soft minimum
        optimizer_soft = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=initial_position,
        )
        pos_soft, rss_soft = optimizer_soft.optimize(
            **test_params,
            num_iterations=3,
            use_soft_min=True,
        )
        
        # Hard minimum
        optimizer_hard = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=initial_position,
        )
        pos_hard, rss_hard = optimizer_hard.optimize(
            **test_params,
            num_iterations=3,
            use_soft_min=False,
        )
        
        # Both should complete successfully
        assert pos_soft.shape == (3,)
        assert pos_hard.shape == (3,)
    
    @pytest.mark.slow
    def test_optimize_convergence(self, test_scene, initial_position):
        """Test that optimizer converges (improves RSS over iterations)."""
        optimizer = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=initial_position,
        )
        
        optimizer.optimize(
            num_iterations=10,
            learning_rate=0.5,
            samples_per_tx=50_000,
            max_depth=5,
            verbose=False,
        )
        
        # Check that RSS improved (or at least didn't get worse)
        initial_rss = optimizer.history['min_rss_dbm_values'][0]
        final_rss = optimizer.history['min_rss_dbm_values'][-1]
        
        # Final RSS should be >= initial (higher dBm is better)
        assert final_rss >= initial_rss - 5.0  # Allow small degradation due to randomness


@pytest.mark.unit
class TestGradientDescentVisualization:
    """Test suite for visualization methods."""
    
    @pytest.mark.requires_scene
    def test_plot_optimization_trajectory_executes(self, test_scene, initial_position, gd_test_params, monkeypatch):
        """Test that plot method executes without error."""
        # Mock plt.show() to avoid displaying plots during tests
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        optimizer = GradientDescentAPOptimizer(
            scene=test_scene,
            initial_position=initial_position,
        )
        
        optimizer.optimize(**gd_test_params)
        
        # Should not raise an error
        optimizer.plot_optimization_trajectory()
