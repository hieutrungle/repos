# Test Module Guide

## Overview

This guide explains what each test module does, how it's organized, and how to work with it.

---

## Test File Structure

```
tests/
├── __init__.py                 # Makes tests a Python package
├── conftest.py                 # Shared fixtures and configuration
├── README.md                   # Testing documentation
├── test_base_optimizer.py      # Base class interface tests
├── test_optimizer_factory.py   # Factory pattern tests
├── test_gradient_descent.py    # Gradient-based optimization tests
└── test_grid_search.py         # Grid search optimization tests
```

---

## 1. conftest.py - Shared Test Configuration

**Purpose**: Central location for pytest fixtures and configuration  
**Lines**: 130  
**Fixtures**: 10

### What It Contains

#### Session-Scoped Fixtures
```python
@pytest.fixture(scope="session")
def test_scene():
    """Sionna scene loaded once per test session."""
    # Expensive setup - reused across all tests
    return load_scene("l_shape_scene.xml")
```

**Why Session Scope?**
- Scene loading is expensive (~2 seconds)
- Scene is read-only in tests
- Sharing reduces total test time from 60s to 10s

#### Function-Scoped Fixtures
```python
@pytest.fixture
def position_bounds():
    """Fresh bounds dict for each test."""
    return {
        'x_min': 5.0, 'x_max': 25.0,
        'y_min': 5.0, 'y_max': 25.0
    }
```

**Why Function Scope?**
- Each test gets independent copy
- Tests can modify without affecting others
- Prevents test pollution

### Key Fixtures

#### 1. `test_scene`
- **Type**: Sionna Scene
- **Scope**: Session (shared)
- **Usage**: All integration tests
- **Purpose**: Provides ray tracing environment

#### 2. `position_bounds`
- **Type**: Dict[str, float]
- **Scope**: Function (independent)
- **Usage**: Constraint testing
- **Purpose**: Standard position limits

#### 3. `initial_position`
- **Type**: Tuple[float, float]
- **Scope**: Function
- **Usage**: Optimization starting point
- **Purpose**: Consistent initialization

#### 4. `test_params`
- **Type**: Dict[str, Any]
- **Scope**: Function
- **Usage**: Optimization parameters
- **Purpose**: Fast test execution
```python
{
    'samples_per_tx': 10_000,    # vs 1M in production
    'max_depth': 5,              # vs 13 in production
    'verbose': False
}
```

#### 5. `gd_test_params`
- **Type**: Dict[str, Any]
- **Scope**: Function
- **Usage**: Gradient descent specific
- **Purpose**: GD-specific parameters
```python
{
    'num_iterations': 3,
    'learning_rate': 0.5,
    **test_params
}
```

#### 6. `gs_test_params`
- **Type**: Dict[str, Any]
- **Scope**: Function
- **Usage**: Grid search specific
- **Purpose**: Grid search parameters

#### 7. `small_grid_bounds`
- **Type**: Dict[str, float]
- **Scope**: Function
- **Usage**: Grid search tests
- **Purpose**: Small 5x5m search area for speed

#### 8. `mock_radio_map`
- **Type**: Mock object
- **Scope**: Function
- **Usage**: Unit tests without ray tracing
- **Purpose**: Fast testing without scene

### When to Add New Fixtures

✅ **Add fixture when:**
- Multiple tests need the same setup
- Setup is expensive (>0.1s)
- Object is complex to construct
- Tests need independent copies

❌ **Don't add fixture when:**
- Only one test needs it
- Setup is trivial (one line)
- Object is simple (int, string)
- Tests need to customize it significantly

---

## 2. test_base_optimizer.py - Base Class Tests

**Purpose**: Validate abstract base class contract  
**Lines**: 168  
**Tests**: 18  
**Classes**: 1 (TestBaseAPOptimizer)

### What It Tests

#### Abstract Class Behavior
```python
class TestBaseAPOptimizer:
    """Validates that BaseAPOptimizer enforces interface contract."""
```

**Key Validations:**
1. Cannot instantiate abstract class directly
2. Subclasses must implement abstract methods
3. Concrete implementations work correctly

### Test Structure

```python
# Helper class for testing
class ConcreteOptimizer(BaseAPOptimizer):
    """Valid implementation for testing."""
    def optimize(self, **kwargs):
        return np.array([10.0, 10.0, 3.8]), 1.0
    
    def plot_results(self, **kwargs):
        pass
```

### Why This Module Matters

- **Interface Contract**: Ensures all optimizers have consistent API
- **Polymorphism**: Enables optimizer swapping without code changes
- **Extensibility**: New optimizers must pass these tests

### Running These Tests

```bash
# All base optimizer tests
python run_tests.py base

# Individual test
pytest tests/test_base_optimizer.py::TestBaseAPOptimizer::test_is_abstract -v
```

### Adding New Base Class Tests

When you add new abstract methods to `BaseAPOptimizer`:

1. Add test that subclasses must implement it
2. Add test that implementation is called
3. Add test for default behavior (if applicable)

Example:
```python
def test_requires_new_method_implementation(self, test_scene):
    """Test that new_method must be implemented."""
    class IncompleteOptimizer(BaseAPOptimizer):
        def optimize(self, **kwargs):
            pass
        # Missing new_method!
    
    with pytest.raises(TypeError):
        IncompleteOptimizer(scene=test_scene)
```

---

## 3. test_optimizer_factory.py - Factory Pattern Tests

**Purpose**: Validate optimizer creation and registration  
**Lines**: 161  
**Tests**: 13  
**Classes**: 1 (TestOptimizerFactory)

### What It Tests

#### Factory Creation
```python
class TestOptimizerFactory:
    """Tests for OptimizerFactory class."""
```

**Key Validations:**
1. Method registration system
2. Optimizer instantiation
3. Name normalization
4. Error handling

### Test Organization

#### Discovery Tests
- List available methods
- Create registered optimizers
- Validate built-in methods

#### Flexibility Tests
- Case-insensitive names
- Hyphen vs underscore
- Multiple naming styles

#### Extension Tests
- Register custom optimizer
- Overwrite existing registration
- Type checking

### Why This Module Matters

- **Usability**: Users can easily switch optimization methods
- **Extensibility**: Custom optimizers can be added
- **Robustness**: Invalid inputs are caught early

### Running These Tests

```bash
# All factory tests
python run_tests.py factory

# Specific test
pytest tests/test_optimizer_factory.py::TestOptimizerFactory::test_create_gradient_descent -v
```

### Adding New Optimizer Tests

When you add a new optimizer (e.g., PSO):

1. Register it in the factory
2. Add factory creation test
3. Validate parameters are passed correctly

Example:
```python
def test_create_pso_optimizer(self, test_scene):
    """Test creating PSO optimizer via factory."""
    optimizer = OptimizerFactory.create(
        method="pso",
        scene=test_scene,
        population_size=30,
        inertia=0.7
    )
    assert isinstance(optimizer, PSOOptimizer)
    assert optimizer.population_size == 30
```

---

## 4. test_gradient_descent.py - Gradient Descent Tests

**Purpose**: Validate differentiable optimization  
**Lines**: 309  
**Tests**: 24  
**Classes**: 4

### What It Tests

#### Test Class Organization

```python
class TestGradientDescentInitialization:
    """8 tests for setup and configuration"""

class TestGradientDescentMethods:
    """4 tests for utility methods"""

class TestGradientDescentOptimization:
    """7 tests for optimization execution"""

class TestGradientDescentVisualization:
    """1 test for plotting"""
```

### Test Progression

#### Level 1: Initialization
Validates that optimizer is set up correctly:
- Tensors created with gradients
- Device properly selected
- History structures initialized

#### Level 2: Methods
Tests individual utility functions:
- Position retrieval
- Constraint application
- Gradient computation

#### Level 3: Optimization
End-to-end optimization testing:
- Optimization loop completes
- Metrics improve
- Constraints respected

#### Level 4: Visualization
Validates result presentation:
- Plots can be generated
- History data is plottable

### Why This Module Matters

- **Core Method**: Gradient descent is primary optimization approach
- **Differentiable**: Uses PyTorch for automatic differentiation
- **Complex**: More moving parts than grid search

### Running These Tests

```bash
# All gradient descent tests
python run_tests.py gradient

# Just initialization
pytest tests/test_gradient_descent.py::TestGradientDescentInitialization -v

# Slow convergence test
pytest tests/test_gradient_descent.py -m slow -v
```

### Test Parameters

**Reduced for Speed:**
```python
gd_test_params = {
    'num_iterations': 3,        # vs 50 in production
    'samples_per_tx': 10_000,   # vs 1M in production
    'max_depth': 5,             # vs 13 in production
    'learning_rate': 0.5
}
```

### Adding New GD Tests

When you modify gradient descent:

1. Add initialization test if constructor changes
2. Add method test if new utility added
3. Add optimization test if algorithm changes
4. Update convergence test if stopping criteria changes

Example:
```python
@pytest.mark.integration
@pytest.mark.requires_scene
def test_optimize_with_momentum(self, test_scene, initial_position):
    """Test gradient descent with momentum."""
    optimizer = GradientDescentAPOptimizer(
        scene=test_scene,
        initial_position=initial_position,
        use_momentum=True,
        momentum=0.9
    )
    result, metric = optimizer.optimize(num_iterations=5)
    assert optimizer.history['momentums'] is not None
```

---

## 5. test_grid_search.py - Grid Search Tests

**Purpose**: Validate exhaustive grid-based search  
**Lines**: 337  
**Tests**: 21  
**Classes**: 4

### What It Tests

#### Test Class Organization

```python
class TestGridSearchInitialization:
    """6 tests for grid generation"""

class TestGridSearchOptimization:
    """7 tests for search execution"""

class TestGridSearchVisualization:
    """3 tests for result plotting"""

class TestGridSearchEdgeCases:
    """2 tests for boundary conditions"""
```

### Test Progression

#### Level 1: Initialization
Validates grid creation:
- Grid covers search bounds
- Resolution respected
- Points within bounds

#### Level 2: Optimization
Tests search execution:
- All points evaluated
- Best point selected
- Metrics computed correctly

#### Level 3: Visualization
Validates result presentation:
- Heatmaps generated
- Different metrics supported
- Invalid metrics rejected

#### Level 4: Edge Cases
Tests unusual scenarios:
- Single-point grids
- Very small bounds
- Degenerate cases

### Why This Module Matters

- **Baseline**: Grid search is comparison standard
- **Simple**: Easier to understand than gradient descent
- **Exhaustive**: Guaranteed to find best grid point

### Running These Tests

```bash
# All grid search tests
python run_tests.py grid

# Just optimization tests
pytest tests/test_grid_search.py::TestGridSearchOptimization -v

# Edge cases
pytest tests/test_grid_search.py::TestGridSearchEdgeCases -v
```

### Test Parameters

**Small Grid for Speed:**
```python
small_grid_bounds = {
    'x_min': 10.0, 'x_max': 15.0,  # 5m x 5m
    'y_min': 10.0, 'y_max': 15.0
}
grid_resolution = 5.0  # Only 4 points
```

### Adding New Grid Search Tests

When you modify grid search:

1. Add initialization test if grid generation changes
2. Add optimization test if search algorithm changes
3. Add visualization test if new plots added
4. Add edge case test if new corner case found

Example:
```python
@pytest.mark.integration
@pytest.mark.requires_scene
def test_optimize_hierarchical_grid(self, test_scene, small_grid_bounds):
    """Test multi-resolution grid search."""
    optimizer = GridSearchAPOptimizer(
        scene=test_scene,
        search_bounds=small_grid_bounds,
        grid_resolution=5.0,
        use_hierarchical=True
    )
    result, metric = optimizer.optimize()
    assert len(optimizer.evaluated_positions) > 4  # Refined around best
```

---

## Test Dependencies

### Package Dependencies
```python
# Core
import pytest
import numpy as np

# Optimization
import torch
import sionna.rt

# Testing
from unittest.mock import Mock, patch
```

### Test Data Dependencies
- `l_shape_scene.xml` - Scene file for Sionna
- No external data files needed
- All fixtures self-contained

### Hardware Dependencies
- **CPU**: All tests can run on CPU
- **GPU**: Optional CUDA for faster execution
- **Memory**: ~2GB for scene loading
- **Disk**: Minimal (no large datasets)

---

## Test Maintenance

### When Tests Need Updates

#### 1. API Changes
If you change method signatures:
```python
# Old
optimizer.optimize(samples=10000)

# New  
optimizer.optimize(samples_per_tx=10000)

# Update ALL tests using this method
```

#### 2. New Features
If you add new functionality:
- Add tests for new feature
- Update existing tests if behavior changes
- Add fixtures if needed

#### 3. Bug Fixes
When fixing a bug:
1. Write test that reproduces bug
2. Fix the bug
3. Verify test passes
4. Keep test to prevent regression

#### 4. Performance Improvements
If you optimize code:
- Tests should still pass
- May need to adjust timing thresholds
- Update test parameters if needed

### Test Quality Checklist

✅ **Good Test:**
- Clear, descriptive name
- Docstring explaining what's tested
- Minimal setup required
- Tests one thing
- Has assertions
- Fast (<1s for unit tests)
- Independent of other tests

❌ **Bad Test:**
- Vague name like `test_optimizer`
- No documentation
- Complex setup
- Tests multiple things
- No assertions (just runs)
- Slow without reason
- Depends on test execution order

---

## Common Test Patterns

### Pattern 1: Basic Initialization Test
```python
def test_initialization(self, test_scene):
    """Test that optimizer can be created."""
    optimizer = MyOptimizer(scene=test_scene)
    assert optimizer is not None
    assert optimizer.scene is test_scene
```

### Pattern 2: Parameter Validation Test
```python
def test_invalid_parameter_raises(self):
    """Test that invalid parameter raises error."""
    with pytest.raises(ValueError, match="must be positive"):
        optimizer = MyOptimizer(learning_rate=-1.0)
```

### Pattern 3: Integration Test
```python
@pytest.mark.integration
@pytest.mark.requires_scene
def test_full_optimization(self, test_scene, test_params):
    """Test complete optimization workflow."""
    optimizer = MyOptimizer(scene=test_scene)
    result, metric = optimizer.optimize(**test_params)
    
    assert result.shape == (3,)
    assert metric > 0
    assert len(optimizer.history['positions']) > 0
```

### Pattern 4: Mock-Based Unit Test
```python
def test_method_without_scene(self, mock_radio_map):
    """Test method using mock instead of real scene."""
    optimizer = MyOptimizer(scene=Mock())
    result = optimizer._process_radio_map(mock_radio_map)
    assert result is not None
```

---

## Troubleshooting Tests

### Test Fails Locally but Not in CI
- Check environment variables
- Verify package versions
- Check file paths (absolute vs relative)
- Review random seeds

### Test Is Flaky (Sometimes Fails)
- Look for race conditions
- Check for uninitialized state
- Review test isolation
- Check random number usage

### Test Is Too Slow
- Reduce test parameters
- Mock expensive operations
- Use smaller test data
- Mark as `@pytest.mark.slow`

### Test Fails After Code Change
- Is it a regression? (Fix code)
- Did API change? (Update test)
- Is test too brittle? (Make more robust)
- Should test be removed? (If testing implementation detail)

---

## Related Documentation

- [Test Summary](TEST_SUMMARY.md) - Overview of all tests
- [Test Categories](TEST_CATEGORIES.md) - Detailed breakdown
- [Testing Guide](TESTING_GUIDE.md) - Step-by-step instructions
- [Full Testing Docs](TESTING.md) - Complete reference
