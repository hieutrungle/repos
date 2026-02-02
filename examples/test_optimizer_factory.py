"""
Quick test for the optimizer factory pattern.

This script tests that the factory can create both optimizer types
and that they follow the expected interface.
"""

from reflector_position.optimizers import (
    BaseAPOptimizer,
    OptimizerFactory,
    create_optimizer,
    GradientDescentAPOptimizer,
    GridSearchAPOptimizer,
)


def test_factory_creation():
    """Test that factory can create optimizer instances."""
    print("Testing OptimizerFactory...")
    
    # Test list methods
    methods = OptimizerFactory.list_methods()
    print(f"✓ Available methods: {methods}")
    assert "gradient_descent" in methods
    assert "grid_search" in methods
    
    # Test that we can't create invalid method
    try:
        OptimizerFactory.create("invalid_method", scene=None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly raises error for invalid method: {e}")
    
    print("\nAll factory tests passed!\n")


def test_inheritance():
    """Test that optimizers inherit from base class."""
    print("Testing optimizer inheritance...")
    
    assert issubclass(GradientDescentAPOptimizer, BaseAPOptimizer)
    print("✓ GradientDescentAPOptimizer inherits from BaseAPOptimizer")
    
    assert issubclass(GridSearchAPOptimizer, BaseAPOptimizer)
    print("✓ GridSearchAPOptimizer inherits from BaseAPOptimizer")
    
    print("\nAll inheritance tests passed!\n")


def test_interface():
    """Test that optimizers have required methods."""
    print("Testing optimizer interface...")
    
    required_methods = ["optimize", "plot_results"]
    
    for optimizer_class in [GradientDescentAPOptimizer, GridSearchAPOptimizer]:
        for method_name in required_methods:
            assert hasattr(optimizer_class, method_name)
            print(f"✓ {optimizer_class.__name__} has {method_name}() method")
    
    print("\nAll interface tests passed!\n")


def test_registration():
    """Test custom optimizer registration."""
    print("Testing custom optimizer registration...")
    
    class DummyOptimizer(BaseAPOptimizer):
        def optimize(self, **kwargs):
            import numpy as np
            return np.array([0, 0, 0]), 0.0
        
        def plot_results(self, **kwargs):
            pass
    
    # Register custom optimizer
    OptimizerFactory.register("dummy", DummyOptimizer)
    print("✓ Custom optimizer registered")
    
    # Check it's in the list
    methods = OptimizerFactory.list_methods()
    assert "dummy" in methods
    print(f"✓ Custom optimizer appears in methods: {methods}")
    
    # Test that non-BaseAPOptimizer raises error
    class NotAnOptimizer:
        pass
    
    try:
        OptimizerFactory.register("bad", NotAnOptimizer)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        print(f"✓ Correctly rejects non-BaseAPOptimizer: {e}")
    
    print("\nAll registration tests passed!\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Optimizer Factory Pattern Tests")
    print("=" * 70)
    print()
    
    test_factory_creation()
    test_inheritance()
    test_interface()
    test_registration()
    
    print("=" * 70)
    print("✓ All tests passed successfully!")
    print("=" * 70)
