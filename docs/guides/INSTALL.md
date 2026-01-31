# Installation Guide

## Prerequisites

- Python 3.10, 3.11, or 3.12
- pip (latest version recommended)
- Virtual environment tool (venv, conda, etc.)

## Option 1: Install from Source (Recommended)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd reflector-position
```

### 2. Create Virtual Environment

Using venv:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# OR
.venv\Scripts\activate  # On Windows
```

Using conda:
```bash
conda create -n reflector-position python=3.11
conda activate reflector-position
```

### 3. Install the Package

Development mode (editable):
```bash
pip install -e .
```

With development tools:
```bash
pip install -e ".[dev]"
```

Regular install:
```bash
pip install .
```

## Option 2: Install from requirements.txt

If you prefer to use requirements.txt:

```bash
pip install -r requirements.txt
pip install -e .
```

## Verify Installation

Test that the package is installed correctly:

```bash
# Check CLI is available
reflector-optimize --help

# Test in Python
python -c "import reflector_position; print(reflector_position.__version__)"
```

## GPU Support

### For NVIDIA GPUs (CUDA)

TensorFlow and PyTorch with CUDA support:

```bash
# Install CUDA-enabled versions
pip install tensorflow[and-cuda]
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Check GPU availability:

```python
import tensorflow as tf
import torch

print(f"TensorFlow GPUs: {tf.config.list_physical_devices('GPU')}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
```

## Troubleshooting

### Import Errors

If you get import errors for Sionna or Mitsuba:

```bash
# Reinstall with specific versions
pip install sionna==0.18.0
pip install mitsuba==3.5.0
```

### TensorFlow Compatibility

If TensorFlow conflicts with NumPy:

```bash
# Install compatible versions
pip install tensorflow==2.20.0 numpy==1.26.4
```

### DrJit or Mitsuba Issues

If DrJit or Mitsuba fail to install:

```bash
# Ensure you have the latest pip
pip install --upgrade pip

# Try installing separately
pip install drjit==1.2.0
pip install mitsuba==3.7.1
```

### Missing Dependencies

If matplotlib or other visualization tools fail:

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install python3-tk

# macOS
brew install python-tk
```

## Development Setup

For contributors:

```bash
# Install with all development tools
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Verify tools work
black --version
ruff --version
pytest --version
mypy --version
```

## Uninstallation

```bash
pip uninstall reflector-position
```

## Next Steps

- Read the [README.md](../README.md) for usage examples
- Check [examples/](../examples/) directory for sample scripts
- Run `reflector-optimize --help` to see CLI options
- See [USAGE.md](USAGE.md) for detailed usage guide
