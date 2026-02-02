# Step-by-Step Testing Guide

## Quick Reference

| Task | Command | Duration |
|------|---------|----------|
| Run all tests | `python run_tests.py all` | ~10s |
| Quick check | `python run_tests.py quick` | ~0.7s |
| With coverage | `python run_tests.py coverage` | ~11s |
| Specific module | `python run_tests.py gradient` | ~5s |

---

## Step 1: Initial Setup

### 1.1 Install Dependencies

```bash
cd /home/hieule/research/reflector-position

# Install package in development mode
pip install -e .

# Install test dependencies
pip install pytest pytest-cov
```

**Verify Installation:**
```bash
python -c "import reflector_position; print('✅ Package installed')"
python -c "import pytest; print('✅ Pytest installed')"
```

### 1.2 Verify Test Environment

```bash
# Check that scene file exists
ls l_shape_scene.xml

# Should output: l_shape_scene.xml
```

**If scene file missing:**
```bash
# Scene file should be in project root
# If not, copy from appropriate location
```

### 1.3 Make Test Runner Executable

```bash
chmod +x run_tests.py
```

---

## Step 2: Running Your First Tests

### 2.1 Quick Validation (No Scene Required)

**Best for:** Rapid feedback during development

```bash
python run_tests.py quick
```

**Expected Output:**
```
======================================================================
Running: Quick Tests (No Scene Required)
======================================================================
Command: pytest tests -m "unit and not requires_scene" -v

...

=================== 27 passed in 0.67s ===================

✅ Quick Tests PASSED
```

**What This Tests:**
- Abstract class enforcement
- Factory pattern
- Parameter validation
- Mock-based unit tests

**When to Use:**
- Pre-commit checks
- Rapid iteration
- CI/CD quick feedback

### 2.2 All Unit Tests

**Best for:** Comprehensive validation

```bash
python run_tests.py unit
```

**Expected Output:**
```
=================== 49 passed in 10.61s ===================

✅ Unit Tests PASSED
```

**What This Tests:**
- All quick tests PLUS
- Scene-based initialization
- Optimization execution (reduced params)
- Visualization code

**When to Use:**
- Before committing changes
- After modifying optimizers
- Pre-pull request validation

### 2.3 All Tests (Including Integration)

**Best for:** Final validation

```bash
python run_tests.py all
```

**Expected Output:**
```
=================== 62 passed in 9.81s ===================

✅ All Tests PASSED
```

**What This Tests:**
- All unit tests PLUS
- Integration tests
- End-to-end workflows
- Full optimization cycles

**When to Use:**
- Before merging to main
- Before releases
- Weekly comprehensive checks

---

## Step 3: Testing Individual Modules

### 3.1 Test Base Optimizer

```bash
python run_tests.py base
```

**What This Tests:**
- Abstract base class behavior
- Interface contract enforcement
- Position validation
- Utility methods

**Expected Tests:** 18  
**Duration:** ~2s

**Use Case:** After modifying `BaseAPOptimizer` class

### 3.2 Test Factory Pattern

```bash
python run_tests.py factory
```

**What This Tests:**
- Optimizer creation
- Method registration
- Name normalization
- Error handling

**Expected Tests:** 13  
**Duration:** ~3s

**Use Case:** After adding new optimizer or modifying factory

### 3.3 Test Gradient Descent

```bash
python run_tests.py gradient
```

**What This Tests:**
- Gradient descent initialization
- PyTorch tensor handling
- Optimization execution
- Convergence behavior

**Expected Tests:** 24  
**Duration:** ~5s

**Use Case:** After modifying gradient descent optimizer

### 3.4 Test Grid Search

```bash
python run_tests.py grid
```

**What This Tests:**
- Grid generation
- Exhaustive search
- Result selection
- Edge cases

**Expected Tests:** 21  
**Duration:** ~5s

**Use Case:** After modifying grid search optimizer

---

## Step 4: Coverage Analysis

### 4.1 Generate Coverage Report

```bash
python run_tests.py coverage
```

**Expected Output:**
```
=================== 62 passed in 11.25s ===================

Name                                    Stmts   Miss  Cover
-----------------------------------------------------------
src/reflector_position/__init__.py         7      0   100%
src/...optimizer_factory.py               25      0   100%
src/...gradient_descent.py               139     25    82%
src/...grid_search.py                     99     23    77%
-----------------------------------------------------------
TOTAL                                    855    504    41%

Coverage HTML written to dir htmlcov

✅ Tests with Coverage PASSED

📊 Coverage report generated in htmlcov/index.html
```

### 4.2 View HTML Coverage Report

```bash
# Open in browser
firefox htmlcov/index.html
# or
google-chrome htmlcov/index.html
# or
xdg-open htmlcov/index.html
```

**What You'll See:**
- Overall coverage percentage
- Coverage by file
- Line-by-line coverage visualization
- Missed lines highlighted in red

### 4.3 Interpret Coverage Results

**Good Coverage (Green):**
- `optimizer_factory.py`: 100%
- `base_optimizer.py`: 92%
- `gradient_descent.py`: 82%

**Areas to Improve:**
- CLI module: 0% (not yet tested)
- Backup files: 0% (can be deleted)

**Focus on Core:**
Core optimizer coverage (82-92%) exceeds 80% target ✅

---

## Step 5: Testing Specific Scenarios

### 5.1 Test a Single Test Function

```bash
pytest tests/test_gradient_descent.py::TestGradientDescentInitialization::test_initialization -v
```

**Output:**
```
tests/test_gradient_descent.py::TestGradientDescentInitialization::test_initialization PASSED
```

**Use Case:** Debugging a specific test failure

### 5.2 Test a Specific Class

```bash
pytest tests/test_gradient_descent.py::TestGradientDescentInitialization -v
```

**Use Case:** Testing all initialization scenarios

### 5.3 Run Tests Matching Pattern

```bash
pytest tests/ -k "initialization" -v
```

**Use Case:** Run all tests with "initialization" in name

### 5.4 Run Only Failed Tests

```bash
pytest tests/ --lf -v
```

**Use Case:** Re-run only tests that failed last time

### 5.5 Stop on First Failure

```bash
pytest tests/ -x -v
```

**Use Case:** Debug first failure without waiting for all tests

---

## Step 6: Testing During Development

### 6.1 Watch Mode (Manual)

```bash
# Terminal 1: Make code changes
vim src/reflector_position/optimizers/gradient_descent.py

# Terminal 2: Run tests repeatedly
while true; do 
    python run_tests.py quick
    sleep 2
done
```

### 6.2 Test-Driven Development Workflow

**Red-Green-Refactor Cycle:**

1. **Red**: Write failing test
```bash
# Add test to test_gradient_descent.py
def test_new_feature(self):
    optimizer = GradientDescentAPOptimizer(...)
    result = optimizer.new_method()
    assert result == expected

# Run tests (should fail)
pytest tests/test_gradient_descent.py::test_new_feature -v
```

2. **Green**: Implement feature
```python
# Add method to gradient_descent.py
def new_method(self):
    return calculated_result
```

3. **Refactor**: Clean up code
```bash
# Run tests (should pass)
pytest tests/test_gradient_descent.py::test_new_feature -v
```

### 6.3 Pre-Commit Checklist

Before committing code:

```bash
# 1. Quick validation
python run_tests.py quick

# 2. Full unit tests
python run_tests.py unit

# 3. Format check (if using formatters)
black src/reflector_position --check

# 4. Lint check (if using linters)
flake8 src/reflector_position

# 5. Type check (if using mypy)
mypy src/reflector_position

# If all pass:
git add -A
git commit -m "Add new feature with tests"
```

---

## Step 7: Debugging Test Failures

### 7.1 Get Detailed Failure Information

```bash
pytest tests/ -v --tb=long
```

**Shows:**
- Full traceback
- Variable values
- Assertion details

### 7.2 Drop into Debugger on Failure

```bash
pytest tests/ -v --pdb
```

**When test fails:**
- Drops into Python debugger
- Can inspect variables
- Can execute code

**Debugger Commands:**
```
p variable_name    # Print variable
l                  # List code around failure
c                  # Continue execution
q                  # Quit debugger
```

### 7.3 Print Debug Information

Add temporary print statements:

```python
def test_something(self):
    optimizer = create_optimizer()
    print(f"DEBUG: optimizer state = {optimizer.state}")  # Temporary debug
    result = optimizer.optimize()
    assert result is not None
```

Run with print output:
```bash
pytest tests/test_file.py -v -s
```

**Note:** Remove debug prints before committing

### 7.4 Common Test Failures

#### Import Error
```
ImportError: cannot import name 'GradientDescentAPOptimizer'
```

**Fix:** Ensure package is installed
```bash
pip install -e .
```

#### Scene Loading Error
```
FileNotFoundError: l_shape_scene.xml not found
```

**Fix:** Verify scene file location
```bash
ls l_shape_scene.xml  # Should be in project root
```

#### Device Error (CUDA)
```
RuntimeError: CUDA out of memory
```

**Fix:** Force CPU mode
```python
# In test file
optimizer.device = torch.device("cpu")
```

#### Assertion Error
```
AssertionError: assert 0.75 > 0.8
```

**Fix:** Check if test expectations are correct or if code needs fixing

---

## Step 8: Advanced Testing

### 8.1 Run Slow Tests

```bash
python run_tests.py slow
```

**What This Includes:**
- Convergence validation (>10s)
- Full optimization runs
- Performance benchmarks

**When to Use:**
- Weekly validation
- Before releases
- Performance regression testing

### 8.2 Parallel Test Execution

```bash
pip install pytest-xdist

# Run tests in parallel (4 workers)
pytest tests/ -n 4 -v
```

**Benefits:**
- Faster execution (if many tests)
- Better CPU utilization

**Caution:**
- Ensure tests are independent
- May use more memory

### 8.3 Generate JUnit XML Report

```bash
pytest tests/ --junitxml=test-results.xml
```

**Use Case:**
- CI/CD integration
- Test result visualization
- Historical tracking

### 8.4 Test with Different Python Versions

```bash
# Using tox (if configured)
tox

# Or manually with conda
conda create -n test-py310 python=3.10
conda activate test-py310
pip install -e .
pytest tests/
```

---

## Step 9: Continuous Integration Setup

### 9.1 GitHub Actions Example

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: '3.12'
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov
    - name: Run tests
      run: python run_tests.py all
    - name: Generate coverage
      run: python run_tests.py coverage
```

### 9.2 Pre-Commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
echo "Running quick tests before commit..."
python run_tests.py quick

if [ $? -ne 0 ]; then
    echo "❌ Tests failed! Commit aborted."
    exit 1
fi

echo "✅ Tests passed!"
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

---

## Step 10: Maintaining Tests

### 10.1 Regular Test Maintenance

**Weekly:**
- Run full test suite: `python run_tests.py all`
- Check coverage: `python run_tests.py coverage`
- Review slow tests (can they be optimized?)

**Monthly:**
- Review test documentation
- Update test parameters if needed
- Remove obsolete tests
- Add tests for new edge cases

### 10.2 Adding Tests for New Features

**Checklist when adding new optimizer:**

1. ✅ Add to factory registration
2. ✅ Create `test_new_optimizer.py`
3. ✅ Write initialization tests
4. ✅ Write method tests
5. ✅ Write optimization tests
6. ✅ Write visualization tests
7. ✅ Update test summary docs

**Template:**
```python
# tests/test_new_optimizer.py
import pytest
from reflector_position.optimizers import NewOptimizer

class TestNewOptimizerInitialization:
    @pytest.mark.unit
    def test_initialization(self, test_scene):
        """Test basic initialization."""
        optimizer = NewOptimizer(scene=test_scene)
        assert optimizer is not None

class TestNewOptimizerOptimization:
    @pytest.mark.integration
    @pytest.mark.requires_scene
    def test_optimize_basic(self, test_scene, test_params):
        """Test basic optimization."""
        optimizer = NewOptimizer(scene=test_scene)
        result, metric = optimizer.optimize(**test_params)
        assert result.shape == (3,)
```

### 10.3 Refactoring Tests

**When to refactor:**
- Multiple tests have duplicate setup
- Tests are hard to understand
- Tests are slow without reason
- Tests are brittle (break on minor changes)

**How to refactor:**
1. Extract common setup to fixtures
2. Simplify test logic
3. Reduce test parameters
4. Use parameterized tests for similar scenarios

---

## Troubleshooting Common Issues

### Issue 1: "No module named 'reflector_position'"

**Cause:** Package not installed  
**Fix:**
```bash
pip install -e .
```

### Issue 2: Tests are very slow

**Cause:** Using production parameters  
**Fix:** Tests should use reduced parameters (already configured in conftest.py)

### Issue 3: "Scene file not found"

**Cause:** Missing l_shape_scene.xml  
**Fix:**
```bash
# Ensure scene file is in project root
ls l_shape_scene.xml
```

### Issue 4: Coverage report is empty

**Cause:** Wrong module path  
**Fix:** Use `--cov=reflector_position` not `--cov=src/reflector_position`

### Issue 5: Tests fail randomly

**Cause:** Flaky test (timing, randomness)  
**Fix:** 
- Add random seed
- Increase tolerance
- Mock external dependencies

---

## Quick Command Reference

```bash
# Run specific test suites
python run_tests.py all         # All 62 tests (~10s)
python run_tests.py unit        # Unit tests only (49 tests)
python run_tests.py quick       # Fast tests (27 tests, ~0.7s)
python run_tests.py integration # Integration tests
python run_tests.py slow        # Slow tests only

# Run specific modules
python run_tests.py base        # Base optimizer (18 tests)
python run_tests.py factory     # Factory pattern (13 tests)
python run_tests.py gradient    # Gradient descent (24 tests)
python run_tests.py grid        # Grid search (21 tests)

# Coverage and reporting
python run_tests.py coverage    # Generate HTML coverage report

# Advanced pytest commands
pytest tests/ -v                              # Verbose output
pytest tests/ -v --tb=short                   # Short traceback
pytest tests/ -v --tb=long                    # Long traceback
pytest tests/ -k "initialization"             # Match pattern
pytest tests/ -m "unit"                       # Run marked tests
pytest tests/ -m "not slow"                   # Exclude slow tests
pytest tests/ --lf                            # Last failed
pytest tests/ -x                              # Stop on first failure
pytest tests/ --pdb                           # Debug on failure
pytest tests/ -s                              # Show print output
pytest tests/ --collect-only                  # List tests without running
pytest tests/test_file.py::TestClass::test_method  # Specific test
```

---

## Related Documentation

- [Test Summary](TEST_SUMMARY.md) - High-level overview
- [Test Categories](TEST_CATEGORIES.md) - Detailed breakdown  
- [Module Guide](MODULE_GUIDE.md) - What each test file does
- [Full Testing Docs](TESTING.md) - Complete reference
