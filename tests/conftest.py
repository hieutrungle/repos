"""
Pytest configuration and shared fixtures.

This module provides fixtures that are shared across all test modules,
including scene setup, common parameters, and mock objects.
"""

import pytest
import numpy as np
import sionna.rt
from pathlib import Path


@pytest.fixture(scope="session")
def scene_path():
    """Get the path to the test scene file."""
    project_root = Path(__file__).parent.parent
    scene_file = project_root / "l_shape_scene.xml"
    
    if not scene_file.exists():
        pytest.skip(f"Scene file not found: {scene_file}")
    
    return str(scene_file)


@pytest.fixture(scope="session")
def test_scene(scene_path):
    """
    Create a test scene for optimization tests.
    
    This fixture is session-scoped to avoid recreating the scene for every test.
    """
    from reflector_position import setup_building_floor_scene
    
    scene = setup_building_floor_scene(
        scene_path=scene_path,
        frequency=5.18e9,
        tx_power_dbm=5.0,
    )
    
    return scene


@pytest.fixture
def position_bounds():
    """Standard position bounds for testing."""
    return {
        'x_min': 5.0,
        'x_max': 25.0,
        'y_min': 5.0,
        'y_max': 25.0,
    }


@pytest.fixture
def initial_position():
    """Standard initial position for gradient descent tests."""
    return (10.0, 10.0)


@pytest.fixture
def small_grid_bounds():
    """Small grid bounds for faster testing."""
    return {
        'x_min': 10.0,
        'x_max': 15.0,
        'y_min': 10.0,
        'y_max': 15.0,
    }


@pytest.fixture
def test_params():
    """Common test parameters with reduced samples for speed."""
    return {
        'samples_per_tx': 10_000,  # Reduced for testing speed
        'max_depth': 5,             # Reduced for testing speed
        'verbose': False,           # Suppress output in tests
    }


@pytest.fixture
def gd_test_params(test_params):
    """Gradient descent specific test parameters."""
    return {
        **test_params,
        'num_iterations': 3,        # Few iterations for testing
        'learning_rate': 0.5,
        'use_soft_min': True,
        'temperature': 0.2,
    }


@pytest.fixture
def gs_test_params(test_params):
    """Grid search specific test parameters (for optimize() method only)."""
    # grid_resolution is not passed to optimize(), it's an init parameter
    return test_params.copy()


@pytest.fixture
def mock_radio_map():
    """Create a mock radio map for testing without ray tracing."""
    class MockRadioMap:
        def __init__(self):
            # Create synthetic RSS data: 10x10 grid
            np.random.seed(42)
            self.rss = np.random.uniform(-100, -50, (10, 10))
    
    return MockRadioMap()


# Markers for different test types
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (run with 'pytest -m slow')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_scene: marks tests that require scene file"
    )
