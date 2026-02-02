# Testing Framework

## Overview

This directory contains comprehensive unit and integration tests for all optimizer methods in the Reflector Position Optimization framework. Tests ensure that each optimizer follows the required interface and produces correct results.

## Test Structure

```
tests/
├── __init__.py                    # Tests package
├── conftest.py                    # Pytest fixtures and configuration
├── test_base_optimizer.py         # Tests for BaseAPOptimizer ABC
├── test_optimizer_factory.py      # Tests for OptimizerFactory
├── test_gradient_descent.py       # Tests for GradientDescentAPOptimizer
├── test_grid_search.py            # Tests for GridSearchAPOptimizer
└── README.md                      # This file
```

## Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
# From project root
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src/reflector_position --cov-report=html
```

### Run Specific Test Files

```bash
# Test base optimizer only
pytest tests/test_base_optimizer.py

# Test factory pattern only
pytest tests/test_optimizer_factory.py

# Test gradient descent only
pytest tests/test_gradient_descent.py

# Test grid search only
pytest tests/test_grid_search.py
```

### Run by Test Markers

```bash
# Run only unit tests (fast)
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration

# Run tests that require scene file
pytest tests/ -m requires_scene

# Skip slow tests
pytest tests/ -m "not slow"
```

### Run Specific Tests

```bash
# Run a specific test class
pytest tests/test_gradient_descent.py::TestGradientDescentInitialization

# Run a specific test method
pytest tests/test_gradient_descent.py::TestGradientDescentInitialization::test_initialization
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Fast tests that don't require full scene initialization
- Test individual components in isolation
- Mock external dependencies when possible
- Should complete in < 1 second each

### Integration Tests (`@pytest.mark.integration`)
- Test full optimization workflows
- Require actual scene setup and ray tracing
- Test interaction between components
- May take several seconds each

### Slow Tests (`@pytest.mark.slow`)
- Long-running tests (> 10 seconds)
- Full optimization runs with many iterations
- Convergence tests
- Run separately in CI/CD: `pytest -m slow`

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

### Scene Fixtures
- `scene_path`: Path to test scene XML file
- `test_scene`: Fully configured Sionna scene (session-scoped)

### Parameter Fixtures
- `position_bounds`: Standard bounds for testing (5-25 m)
- `initial_position`: Standard initial position (10, 10)
- `small_grid_bounds`: Smaller bounds for faster grid search tests
- `test_params`: Common parameters with reduced samples
- `gd_test_params`: Gradient descent specific parameters
- `gs_test_params`: Grid search specific parameters

### Mock Fixtures
- `mock_radio_map`: Synthetic radio map for testing without ray tracing

## Writing New Tests

### 1. Test a New Optimizer

Create a new file `test_my_optimizer.py`:

```python
"""
Unit and integration tests for MyOptimizer.
"""

import pytest
import numpy as np
from reflector_position.optimizers import MyOptimizer


@pytest.mark.unit
class TestMyOptimizerInitialization:
    """Test initialization."""
    
    @pytest.mark.requires_scene
    def test_initialization(self, test_scene):
        optimizer = MyOptimizer(scene=test_scene)
        assert optimizer.scene is test_scene


@pytest.mark.integration
@pytest.mark.requires_scene
class TestMyOptimizerOptimization:
    """Test optimization execution."""
    
    def test_optimize_basic(self, test_scene, test_params):
        optimizer = MyOptimizer(scene=test_scene)
        position, rss = optimizer.optimize(**test_params)
        
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
```

### 2. Use Fixtures

```python
def test_with_fixtures(self, test_scene, position_bounds, test_params):
    """Test using predefined fixtures."""
    optimizer = MyOptimizer(
        scene=test_scene,
        bounds=position_bounds
    )
    
    optimizer.optimize(**test_params)
```

### 3. Mark Tests Appropriately

```python
@pytest.mark.unit
def test_fast_unit():
    """Fast test without scene."""
    pass

@pytest.mark.integration
@pytest.mark.requires_scene
def test_with_scene(test_scene):
    """Test requiring scene."""
    pass

@pytest.mark.slow
def test_long_running(test_scene):
    """Test that takes > 10 seconds."""
    pass
```

## Test Requirements

### For New Optimizers

Every new optimizer must pass these tests:

1. **Inheritance Test**: Must inherit from `BaseAPOptimizer`
2. **Interface Test**: Must implement `optimize()` and `plot_results()`
3. **Initialization Test**: Must store scene, fixed_z, and bounds correctly
4. **Bounds Test**: Must respect position bounds during optimization
5. **Return Type Test**: Must return `(np.ndarray, float)` from optimize()
6. **Visualization Test**: `plot_results()` must execute without error

### Minimum Test Coverage

- **Unit Tests**: Test all public methods
- **Integration Tests**: Test full optimization workflow
- **Edge Cases**: Test boundary conditions and error handling
- **Visualization**: Test that plotting methods don't crash

## Continuous Integration

### Pre-commit Checks

Before committing, run:

```bash
# Quick unit tests
pytest tests/ -m unit

# All tests except slow ones
pytest tests/ -m "not slow"
```

### CI/CD Pipeline

```yaml
# Example GitHub Actions workflow
test:
  runs-on: ubuntu-latest
  steps:
    - name: Run unit tests
      run: pytest tests/ -m unit
    
    - name: Run integration tests
      run: pytest tests/ -m integration
    
    - name: Run slow tests (nightly)
      run: pytest tests/ -m slow
```

## Test Data

Tests use reduced parameters for speed:
- `samples_per_tx`: 10,000 (vs 1,000,000 in production)
- `max_depth`: 5 (vs 13 in production)
- `num_iterations`: 3 (vs 50 in production)
- `grid_resolution`: 5.0 m (vs 1-2 m in production)

## Debugging Tests

### Run with Detailed Output

```bash
# Show print statements
pytest tests/ -s

# Show detailed failure info
pytest tests/ -vv

# Stop at first failure
pytest tests/ -x

# Enter debugger on failure
pytest tests/ --pdb
```

### Run Single Test in Debug Mode

```bash
# With full traceback
pytest tests/test_gradient_descent.py::test_optimize_basic -vv --tb=long

# With print statements
pytest tests/test_gradient_descent.py::test_optimize_basic -s
```

## Coverage Reports

Generate coverage reports to ensure comprehensive testing:

```bash
# Terminal coverage report
pytest tests/ --cov=src/reflector_position --cov-report=term-missing

# HTML coverage report
pytest tests/ --cov=src/reflector_position --cov-report=html
open htmlcov/index.html
```

### Coverage Goals

- **Overall**: > 80% code coverage
- **Critical paths**: > 95% coverage for optimizer core logic
- **Edge cases**: All error paths should be tested

## Best Practices

1. **Test Independence**: Each test should be self-contained
2. **Fast Tests**: Use fixtures to avoid redundant setup
3. **Clear Names**: Test names should describe what they test
4. **Assertions**: Include meaningful assertion messages
5. **Mock External Calls**: Mock slow operations when possible
6. **Test Edge Cases**: Test boundary conditions and error paths
7. **Document Tests**: Include docstrings explaining test purpose

## Common Issues

### Scene File Not Found

If tests fail with "Scene file not found":
```bash
# Make sure scene file exists
ls l_shape_scene.xml

# Or skip tests requiring scene
pytest tests/ -m "not requires_scene"
```

### CUDA/GPU Issues

If GPU tests fail:
```bash
# Tests should automatically fall back to CPU
# Check device detection in test output
pytest tests/ -v -s | grep device
```

### Slow Test Execution

If tests are too slow:
```bash
# Run only unit tests
pytest tests/ -m unit

# Skip slow tests
pytest tests/ -m "not slow"

# Run in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest tests/ -n auto
```

## Adding New Test Fixtures

Add shared fixtures to `conftest.py`:

```python
@pytest.fixture
def my_fixture():
    """Description of fixture."""
    return some_value
```

## Test Metrics

Track these metrics over time:
- Total tests: Should increase with new features
- Test execution time: Should stay reasonable (< 2 min for unit tests)
- Coverage: Should increase or stay high (> 80%)
- Flaky tests: Should be zero

## Future Enhancements

- [ ] Add property-based testing with Hypothesis
- [ ] Add performance benchmarking tests
- [ ] Add mutation testing with mutmut
- [ ] Add test parallelization for speed
- [ ] Add automated test generation for new optimizers
