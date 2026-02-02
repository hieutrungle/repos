"""
Unit and integration tests for GridSearchAPOptimizer.

Tests grid search optimization functionality including initialization,
grid creation, optimization execution, and result visualization.
"""

import pytest
import numpy as np
import torch

from reflector_position.optimizers import GridSearchAPOptimizer


@pytest.mark.unit
class TestGridSearchInitialization:
    """Test suite for GridSearchAPOptimizer initialization."""
    
    @pytest.mark.requires_scene
    def test_initialization(self, test_scene, position_bounds):
        """Test basic initialization."""
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=position_bounds,
            grid_resolution=2.0,
        )
        
        assert optimizer.scene is test_scene
        assert optimizer.search_bounds == position_bounds
        assert optimizer.grid_resolution == 2.0
        assert optimizer.fixed_z == 3.8  # Default value
    
    @pytest.mark.requires_scene
    def test_initialization_with_custom_z(self, test_scene, position_bounds):
        """Test initialization with custom z-height."""
        fixed_z = 5.0
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=position_bounds,
            fixed_z=fixed_z,
        )
        
        assert optimizer.fixed_z == fixed_z
    
    @pytest.mark.requires_scene
    def test_grid_creation(self, test_scene, position_bounds):
        """Test that grid is created correctly."""
        resolution = 5.0
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=position_bounds,
            grid_resolution=resolution,
        )
        
        # Check grid dimensions
        expected_x_points = int((position_bounds['x_max'] - position_bounds['x_min']) / resolution) + 1
        expected_y_points = int((position_bounds['y_max'] - position_bounds['y_min']) / resolution) + 1
        
        assert optimizer.x_grid.shape[0] == expected_x_points
        assert optimizer.y_grid.shape[1] == expected_y_points
        assert optimizer.total_positions == expected_x_points * expected_y_points
    
    @pytest.mark.requires_scene
    def test_grid_bounds(self, test_scene, position_bounds):
        """Test that grid covers the specified bounds."""
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=position_bounds,
            grid_resolution=1.0,
        )
        
        # Convert to CPU for testing
        x_grid_cpu = optimizer.x_grid.cpu().numpy()
        y_grid_cpu = optimizer.y_grid.cpu().numpy()
        
        # Check min/max values
        assert x_grid_cpu.min() == position_bounds['x_min']
        assert x_grid_cpu.max() == position_bounds['x_max']
        assert y_grid_cpu.min() == position_bounds['y_min']
        assert y_grid_cpu.max() == position_bounds['y_max']
    
    @pytest.mark.requires_scene
    def test_results_storage_initialized(self, test_scene, position_bounds):
        """Test that results storage is initialized."""
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=position_bounds,
        )
        
        assert "positions" in optimizer.results
        assert "min_rss_values" in optimizer.results
        assert "min_rss_dbm_values" in optimizer.results
        assert "coverage_values" in optimizer.results
        assert "radio_maps" in optimizer.results
        
        for key in optimizer.results:
            assert isinstance(optimizer.results[key], list)
            assert len(optimizer.results[key]) == 0
    
    @pytest.mark.requires_scene
    def test_device_detection(self, test_scene, position_bounds):
        """Test that device is correctly detected."""
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=position_bounds,
        )
        
        assert optimizer.device in [torch.device("cuda"), torch.device("cpu")]


@pytest.mark.integration
@pytest.mark.requires_scene
class TestGridSearchOptimization:
    """Integration tests for grid search optimization."""
    
    def test_optimize_basic(self, test_scene, small_grid_bounds, gs_test_params):
        """Test basic optimization execution."""
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=small_grid_bounds,
            grid_resolution=5.0,
        )
        
        best_position, best_rss = optimizer.optimize(**gs_test_params)
        
        assert isinstance(best_position, np.ndarray)
        assert best_position.shape == (3,)
        assert isinstance(best_rss, float)
        assert best_rss > 0  # RSS should be positive
    
    def test_optimize_evaluates_all_positions(self, test_scene, small_grid_bounds, gs_test_params):
        """Test that all grid positions are evaluated."""
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=small_grid_bounds,
            grid_resolution=5.0,
        )
        
        optimizer.optimize(**gs_test_params)
        
        # Number of results should equal total positions
        assert len(optimizer.results['positions']) == optimizer.total_positions
        assert len(optimizer.results['min_rss_values']) == optimizer.total_positions
    
    def test_optimize_best_within_bounds(self, test_scene, small_grid_bounds, gs_test_params):
        """Test that best position is within search bounds."""
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=small_grid_bounds,
            grid_resolution=5.0,
        )
        
        best_position, _ = optimizer.optimize(**gs_test_params)
        
        assert small_grid_bounds['x_min'] <= best_position[0] <= small_grid_bounds['x_max']
        assert small_grid_bounds['y_min'] <= best_position[1] <= small_grid_bounds['y_max']
        assert np.isclose(best_position[2], optimizer.fixed_z)
    
    def test_optimize_best_is_maximum(self, test_scene, small_grid_bounds, gs_test_params):
        """Test that returned position has the best RSS."""
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=small_grid_bounds,
            grid_resolution=5.0,
        )
        
        best_position, best_rss = optimizer.optimize(**gs_test_params)
        
        # Best RSS should be the maximum in results
        max_rss = max(optimizer.results['min_rss_values'])
        assert np.isclose(best_rss, max_rss)
    
    def test_optimize_coarse_vs_fine_grid(self, test_scene, small_grid_bounds, test_params):
        """Test optimization with different grid resolutions."""
        # Coarse grid
        optimizer_coarse = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=small_grid_bounds,
            grid_resolution=5.0,
        )
        pos_coarse, rss_coarse = optimizer_coarse.optimize(**test_params)
        
        # Finer grid (more positions to evaluate)
        optimizer_fine = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=small_grid_bounds,
            grid_resolution=2.5,
        )
        
        # Fine grid should evaluate more positions
        assert optimizer_fine.total_positions > optimizer_coarse.total_positions
    
    def test_optimize_coverage_calculation(self, test_scene, small_grid_bounds, gs_test_params):
        """Test that coverage values are calculated."""
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=small_grid_bounds,
            grid_resolution=5.0,
        )
        
        optimizer.optimize(**gs_test_params)
        
        # All positions should have coverage values
        assert len(optimizer.results['coverage_values']) == optimizer.total_positions
        
        # Coverage should be between 0 and 100
        for coverage in optimizer.results['coverage_values']:
            assert 0.0 <= coverage <= 100.0
    
    @pytest.mark.slow
    def test_optimize_stores_radio_maps(self, test_scene, small_grid_bounds, gs_test_params):
        """Test that radio maps are stored."""
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=small_grid_bounds,
            grid_resolution=5.0,
        )
        
        optimizer.optimize(**gs_test_params)
        
        # Radio maps should be stored
        assert len(optimizer.results['radio_maps']) == optimizer.total_positions
        
        # Each should be a valid radio map object
        for rm in optimizer.results['radio_maps']:
            assert hasattr(rm, 'rss')


@pytest.mark.unit
class TestGridSearchVisualization:
    """Test suite for visualization methods."""
    
    @pytest.mark.requires_scene
    def test_plot_results_min_rss(self, test_scene, small_grid_bounds, gs_test_params, monkeypatch):
        """Test plotting minimum RSS results."""
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=small_grid_bounds,
            grid_resolution=5.0,
        )
        
        optimizer.optimize(**gs_test_params)
        
        # Should not raise an error
        optimizer.plot_results(metric='min_rss_dbm')
    
    @pytest.mark.requires_scene
    def test_plot_results_coverage(self, test_scene, small_grid_bounds, gs_test_params, monkeypatch):
        """Test plotting coverage results."""
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=small_grid_bounds,
            grid_resolution=5.0,
        )
        
        optimizer.optimize(**gs_test_params)
        
        # Should not raise an error
        optimizer.plot_results(metric='coverage')
    
    @pytest.mark.requires_scene
    def test_plot_results_invalid_metric(self, test_scene, small_grid_bounds, gs_test_params):
        """Test that invalid metric raises error."""
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds=small_grid_bounds,
            grid_resolution=5.0,
        )
        
        optimizer.optimize(**gs_test_params)
        
        with pytest.raises(ValueError) as exc_info:
            optimizer.plot_results(metric='invalid_metric')
        
        assert "Unknown metric" in str(exc_info.value)


@pytest.mark.unit
class TestGridSearchEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.requires_scene
    def test_single_point_grid(self, test_scene, test_params):
        """Test grid search with only one point."""
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds={'x_min': 10.0, 'x_max': 10.0, 'y_min': 10.0, 'y_max': 10.0},
            grid_resolution=1.0,
        )
        
        assert optimizer.total_positions == 1
        
        position, rss = optimizer.optimize(**test_params)
        assert position[0] == 10.0
        assert position[1] == 10.0
    
    @pytest.mark.requires_scene
    def test_small_bounds(self, test_scene, test_params):
        """Test grid search with very small bounds."""
        optimizer = GridSearchAPOptimizer(
            scene=test_scene,
            search_bounds={'x_min': 10.0, 'x_max': 11.0, 'y_min': 10.0, 'y_max': 11.0},
            grid_resolution=0.5,
        )
        
        position, rss = optimizer.optimize(**test_params)
        
        # Should complete successfully
        assert 10.0 <= position[0] <= 11.0
        assert 10.0 <= position[1] <= 11.0
