#!/usr/bin/env python3
"""
Test runner script for reflector_position package.

Provides convenient commands for running different test suites.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print results."""
    print("=" * 70)
    print(f"Running: {description}")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"\n❌ {description} FAILED")
        return False
    else:
        print(f"\n✅ {description} PASSED")
        return True


def main():
    """Main test runner."""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py <command>")
        print("\nAvailable commands:")
        print("  all          - Run all tests")
        print("  unit         - Run unit tests only (fast)")
        print("  integration  - Run integration tests")
        print("  slow         - Run slow tests")
        print("  coverage     - Run tests with coverage report")
        print("  factory      - Test factory pattern only")
        print("  gradient     - Test gradient descent only")
        print("  grid         - Test grid search only")
        print("  base         - Test base optimizer only")
        print("  quick        - Run unit tests without scene (fastest)")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    tests_dir = Path(__file__).parent / "tests"
    
    success = True
    
    if command == "all":
        success = run_command(
            ["pytest", str(tests_dir), "-v"],
            "All Tests"
        )
    
    elif command == "unit":
        success = run_command(
            ["pytest", str(tests_dir), "-m", "unit", "-v"],
            "Unit Tests"
        )
    
    elif command == "integration":
        success = run_command(
            ["pytest", str(tests_dir), "-m", "integration", "-v"],
            "Integration Tests"
        )
    
    elif command == "slow":
        success = run_command(
            ["pytest", str(tests_dir), "-m", "slow", "-v"],
            "Slow Tests"
        )
    
    elif command == "coverage":
        success = run_command(
            ["pytest", str(tests_dir), 
             "--cov=reflector_position",
             "--cov-report=html",
             "--cov-report=term-missing"],
            "Tests with Coverage"
        )
        if success:
            print("\n📊 Coverage report generated in htmlcov/index.html")
    
    elif command == "factory":
        success = run_command(
            ["pytest", str(tests_dir / "test_optimizer_factory.py"), "-v"],
            "Factory Pattern Tests"
        )
    
    elif command == "gradient":
        success = run_command(
            ["pytest", str(tests_dir / "test_gradient_descent.py"), "-v"],
            "Gradient Descent Tests"
        )
    
    elif command == "grid":
        success = run_command(
            ["pytest", str(tests_dir / "test_grid_search.py"), "-v"],
            "Grid Search Tests"
        )
    
    elif command == "base":
        success = run_command(
            ["pytest", str(tests_dir / "test_base_optimizer.py"), "-v"],
            "Base Optimizer Tests"
        )
    
    elif command == "quick":
        success = run_command(
            ["pytest", str(tests_dir), "-m", "unit and not requires_scene", "-v"],
            "Quick Unit Tests (No Scene Required)"
        )
    
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
