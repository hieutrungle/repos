# Test Categories - Detailed Breakdown

## Category Overview

This document provides in-depth details about each test category, what they validate, and why they're important.

---

## 1. Base Optimizer Tests (18 tests)

**File**: `tests/test_base_optimizer.py`  
**Purpose**: Validate the abstract base class interface contract  
**Importance**: Ensures all optimizer implementations follow the same interface

### Test Classes

#### TestBaseAPOptimizer (18 tests)

##### Abstract Class Enforcement (5 tests)
```python
test_is_abstract()                          # BaseAPOptimizer is ABC
test_cannot_instantiate_directly()          # Cannot create instance
test_requires_optimize_implementation()     # Subclass must implement optimize()
test_requires_plot_results_implementation() # Subclass must implement plot_results()
test_concrete_implementation_works()        # Valid subclass can be instantiated
```

**What's Tested:**
- ABC metaclass properly applied
- Abstract methods enforced at instantiation
- Concrete implementations accepted

**Why Important:**
- Guarantees interface consistency across all optimizers
- Prevents incomplete implementations
- Enables polymorphic usage

##### Position Bounds Validation (5 tests)
```python
test_get_position_bounds()              # Returns bounds dict
test_get_position_bounds_empty()        # Handles no bounds case
test_validate_position_within_bounds()  # Accepts valid positions
test_validate_position_outside_bounds() # Rejects invalid positions
test_get_full_position()                # Converts 2D to 3D position
```

**What's Tested:**
- Boundary checking logic
- Edge cases (no bounds, on boundary)
- Position coordinate handling

**Why Important:**
- Prevents optimizers from exploring invalid regions
- Ensures physical constraints are respected
- Validates geometric transformations

##### Initialization & State (8 tests)
```python
test_optimize_method_called()      # optimize() is callable
test_plot_results_method_called()  # plot_results() is callable
test_fixed_z_stored()              # Z-height properly stored
test_scene_stored()                # Scene reference maintained
test_position_bounds_stored()      # Bounds properly stored
```

**What's Tested:**
- Constructor parameter handling
- State initialization
- Method accessibility

**Why Important:**
- Ensures consistent initialization across optimizers
- Validates state management
- Confirms required parameters are stored

### Coverage: 92%
**Missing Lines:**
- Line 61: Edge case in position validation
- Line 71: Unused utility method

---

## 2. Factory Pattern Tests (13 tests)

**File**: `tests/test_optimizer_factory.py`  
**Purpose**: Validate optimizer creation and registration mechanism  
**Importance**: Enables easy extension with new optimization methods

### Test Classes

#### TestOptimizerFactory (13 tests)

##### Method Discovery (3 tests)
```python
test_list_methods()                 # Lists all available methods
test_create_gradient_descent()      # Creates gradient descent optimizer
test_create_grid_search()           # Creates grid search optimizer
```

**What's Tested:**
- Method registration system
- Available method enumeration
- Successful optimizer instantiation

**Why Important:**
- Users can discover available optimization methods
- Factory correctly maps names to classes
- Proper initialization with correct parameters

##### Method Name Normalization (2 tests)
```python
test_create_with_hyphens()      # "gradient-descent" works
test_create_case_insensitive()  # "GRADIENT_DESCENT" works
```

**What's Tested:**
- Hyphen to underscore conversion
- Case-insensitive matching
- Name flexibility

**Why Important:**
- Improves user experience (flexible naming)
- Prevents common typos from breaking code
- Allows various naming conventions

##### Error Handling (2 tests)
```python
test_create_invalid_method_raises_error()      # Unknown method raises ValueError
test_register_non_optimizer_raises_error()     # Non-BaseAPOptimizer rejected
```

**What's Tested:**
- Invalid method name handling
- Type checking for registrations
- Informative error messages

**Why Important:**
- Clear error messages guide users
- Prevents registration of invalid classes
- Maintains type safety

##### Custom Registration (3 tests)
```python
test_register_custom_optimizer()          # Add new optimizer
test_convenience_function()               # create_optimizer() works
test_pass_kwargs_to_optimizer()           # Constructor kwargs passed
test_multiple_registrations_same_name()   # Overwrites allowed
```

**What's Tested:**
- Custom optimizer registration
- Convenience wrapper function
- Parameter forwarding
- Registration updates

**Why Important:**
- Enables plugin architecture
- Users can add custom optimizers
- Maintains backward compatibility

### Coverage: 100%
**All lines covered** ✅

---

## 3. Gradient Descent Tests (24 tests)

**File**: `tests/test_gradient_descent.py`  
**Purpose**: Validate gradient-based optimization using PyTorch  
**Importance**: Core differentiable optimization method

### Test Classes

#### TestGradientDescentInitialization (8 tests)

##### Basic Initialization (6 tests)
```python
test_initialization()                   # Default parameters work
test_initialization_with_custom_z()     # Custom height accepted
test_initialization_with_bounds()       # Position constraints stored
test_tensors_require_grad()             # Gradients enabled
test_device_detection()                 # CPU/CUDA handling
test_history_initialized()              # Tracking structures created
```

**What's Tested:**
- Parameter defaults
- Custom parameter handling
- PyTorch tensor setup
- Device selection logic
- History data structure initialization

**Why Important:**
- Ensures differentiable optimization is possible
- Validates hardware acceleration support
- Confirms tracking infrastructure ready

#### TestGradientDescentMethods (4 tests)

##### Utility Methods (4 tests)
```python
test_get_full_position()                        # Returns [x, y, z]
test_apply_position_constraints_within_bounds() # No change if valid
test_apply_position_constraints_outside_bounds() # Clamps to bounds
test_apply_position_constraints_no_bounds()     # No-op without bounds
```

**What's Tested:**
- Position retrieval
- Constraint projection logic
- Boundary clamping
- Empty bounds handling

**Why Important:**
- Ensures optimizer respects constraints during optimization
- Validates coordinate transformations
- Confirms safety mechanisms work

#### TestGradientDescentOptimization (7 tests)

##### Optimization Execution (7 tests)
```python
test_optimize_basic()                       # Optimization completes
test_optimize_history_tracking()            # All metrics recorded
test_optimize_respects_bounds()             # Stays within constraints
test_optimize_different_learning_rates()    # Learning rate effects
test_optimize_soft_vs_hard_min()            # Metric variants work
test_optimize_convergence()                 # Eventually converges (slow)
```

**What's Tested:**
- End-to-end optimization workflow
- History tracking accuracy
- Constraint enforcement
- Hyperparameter effects
- Convergence behavior

**Why Important:**
- Validates core optimization loop
- Ensures metrics are tracked correctly
- Confirms optimization actually improves objective
- Tests different configurations

#### TestGradientDescentVisualization (1 test)

##### Plotting (1 test)
```python
test_plot_optimization_trajectory_executes()  # Plot runs without error
```

**What's Tested:**
- Visualization code executes
- No display errors
- History data is plottable

**Why Important:**
- Users can visualize optimization progress
- Ensures plot code doesn't crash
- Validates data format for plotting

### Coverage: 82%
**Missing Lines (25):**
- Lines 177-189: Alternative loss computation paths
- Line 230: Error handling branch
- Lines 258-259: Verbose output formatting
- Lines 274-291: Extended history tracking
- Line 305: Edge case in convergence check

---

## 4. Grid Search Tests (21 tests)

**File**: `tests/test_grid_search.py`  
**Purpose**: Validate exhaustive search over position grid  
**Importance**: Baseline method for comparison

### Test Classes

#### TestGridSearchInitialization (6 tests)

##### Setup Validation (6 tests)
```python
test_initialization()               # Default parameters work
test_initialization_with_custom_z() # Custom height accepted
test_grid_creation()                # Grid properly generated
test_grid_bounds()                  # Grid within search bounds
test_results_storage_initialized()  # Storage structures ready
test_device_detection()             # CPU/CUDA handling
```

**What's Tested:**
- Parameter handling
- Grid generation algorithm
- Bounds validation
- Storage initialization
- Hardware support

**Why Important:**
- Ensures grid covers search space correctly
- Validates all positions will be evaluated
- Confirms result storage is ready

#### TestGridSearchOptimization (7 tests)

##### Search Execution (7 tests)
```python
test_optimize_basic()                   # Optimization completes
test_optimize_evaluates_all_positions() # Every grid point tested
test_optimize_best_within_bounds()      # Result is valid
test_optimize_best_is_maximum()         # Best RSS is highest
test_optimize_coarse_vs_fine_grid()     # Resolution effects
test_optimize_coverage_calculation()    # Coverage computed
test_optimize_stores_radio_maps()       # Maps saved for analysis
```

**What's Tested:**
- Complete grid evaluation
- Best position selection
- Result validity
- Metric computation
- Resolution impact
- Data persistence

**Why Important:**
- Guarantees exhaustive search
- Validates result selection logic
- Ensures metrics are accurate
- Confirms data is available for analysis

#### TestGridSearchVisualization (3 tests)

##### Result Plotting (3 tests)
```python
test_plot_results_min_rss()        # RSS heatmap works
test_plot_results_coverage()       # Coverage heatmap works
test_plot_results_invalid_metric() # Invalid metric raises error
```

**What's Tested:**
- Different metric visualizations
- Heatmap generation
- Error handling

**Why Important:**
- Users can visualize search results
- Multiple metrics supported
- Invalid inputs rejected gracefully

#### TestGridSearchEdgeCases (2 tests)

##### Boundary Conditions (2 tests)
```python
test_single_point_grid()  # Grid with one point works
test_small_bounds()       # Small search space handled
```

**What's Tested:**
- Degenerate grid cases
- Minimum viable grids
- Edge case handling

**Why Important:**
- Prevents crashes on unusual inputs
- Validates robustness
- Ensures algorithm handles edge cases

### Coverage: 77%
**Missing Lines (23):**
- Lines 98-107: Alternative grid generation logic
- Lines 152-165: Extended result analysis
- Lines 170-179: Advanced visualization options

---

## Test Markers Explained

### `@pytest.mark.unit`
- **Purpose**: Fast, isolated component tests
- **Characteristics**: 
  - Can run without full system setup
  - Mock external dependencies
  - Focus on single component
- **When to use**: Testing individual methods, classes, functions

### `@pytest.mark.integration`
- **Purpose**: Multi-component interaction tests
- **Characteristics**:
  - Tests component integration
  - May require system resources
  - Validates end-to-end workflows
- **When to use**: Testing optimizer with scene, full optimization runs

### `@pytest.mark.slow`
- **Purpose**: Long-running tests (>10 seconds)
- **Characteristics**:
  - Convergence validation
  - Performance benchmarks
  - Large-scale tests
- **When to use**: Tests that can't be sped up, convergence tests

### `@pytest.mark.requires_scene`
- **Purpose**: Tests needing Sionna ray tracing scene
- **Characteristics**:
  - Requires scene file (l_shape_scene.xml)
  - Uses RadioMapSolver
  - Longer setup time
- **When to use**: Tests involving actual ray tracing

---

## Test Patterns & Best Practices

### Fixture Usage
```python
def test_with_fixtures(self, test_scene, test_params, position_bounds):
    """Leverage shared fixtures for consistency."""
    optimizer = GradientDescentAPOptimizer(
        scene=test_scene,
        initial_position=(10.0, 10.0),
        position_bounds=position_bounds
    )
    result = optimizer.optimize(**test_params)
    assert result is not None
```

### Mocking Displays
```python
def test_plot(self, monkeypatch):
    """Prevent plot windows during tests."""
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: None)
    optimizer.plot_results()  # No window shown
```

### Parameterized Tests
```python
@pytest.mark.parametrize("learning_rate", [0.1, 0.5, 1.0])
def test_learning_rates(self, learning_rate):
    """Test multiple parameter values."""
    optimizer = GradientDescentAPOptimizer(...)
    result = optimizer.optimize(learning_rate=learning_rate)
    assert result is not None
```

### Error Testing
```python
def test_invalid_input_raises():
    """Validate error handling."""
    with pytest.raises(ValueError, match="Invalid method"):
        OptimizerFactory.create("nonexistent")
```

---

## Coverage Interpretation

### What 82% Coverage Means
- **82% of code lines executed** during tests
- **18% not tested**: Edge cases, error paths, verbose output
- **Missing lines**: Typically less common code paths

### Improving Coverage
1. **Add edge case tests**: Test boundary conditions
2. **Test error paths**: Validate exception handling
3. **Test verbose output**: Check logging and print statements
4. **Remove dead code**: Delete unused code

### Coverage ≠ Quality
Good coverage doesn't guarantee bug-free code:
- ✅ 82% coverage with good assertions > 100% coverage with weak tests
- ✅ Focus on testing behavior, not just executing lines
- ✅ Test edge cases and error conditions
- ✅ Validate outputs, not just that code runs

---

## Related Documentation

- [Test Summary](TEST_SUMMARY.md) - High-level overview
- [Module Guide](MODULE_GUIDE.md) - What each test file does
- [Testing Guide](TESTING_GUIDE.md) - Step-by-step instructions
- [Full Testing Docs](TESTING.md) - Complete reference
