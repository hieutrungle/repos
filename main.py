"""Main entry point for running optimization examples."""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from reflector_position import __version__
from reflector_position import cli

if __name__ == "__main__":
    # If no arguments provided, show welcome message and help
    if len(sys.argv) == 1:
        print(f"Reflector Position Optimization v{__version__}")
        print("\nTo run optimization, use one of these methods:\n")
        print("1. Command-line interface:")
        print("   python main.py /path/to/scene.xml --method gradient-descent")
        print("   OR: reflector-optimize /path/to/scene.xml --method gradient-descent")
        print("\n2. Run example scripts:")
        print("   python examples/quick_test.py")
        print("   python examples/full_comparison.py")
        print("\n3. Use the Python API (see README.md for examples)")
        print("\nDocumentation:")
        print("   - Main README: README.md")
        print("   - Installation: docs/guides/INSTALL.md")
        print("   - Usage Guide: docs/guides/USAGE.md")
        print("   - Quick Reference: docs/guides/QUICKREF.md")
        print("   - Optimization Workflow: docs/methodology/OPTIMIZATION_WORKFLOW.md")
        print("   - Project Status: STATUS.md")
        print("\nFor help: python main.py --help")
        print()
    
    # Forward all arguments to the CLI
    cli.main()