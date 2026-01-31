# Quick Reference

## Installation

```bash
pip install -e .
```

## CLI Quick Commands

```bash
# Gradient descent (fast)
reflector-optimize scene.xml --method gradient-descent

# Grid search (thorough)
reflector-optimize scene.xml --method grid-search

# Both methods
reflector-optimize scene.xml --method all

# Custom parameters
reflector-optimize scene.xml --gd-iterations 20 --gd-lr 0.5
```

## Python API Cheat Sheet

### Gradient Descent

```python
from reflector_position import (
    setup_building_floor_scene,
    GradientDescentAPOptimizer,
)

scene = setup_building_floor_scene("scene.xml")
optimizer = GradientDescentAPOptimizer(
    scene, 
    initial_position=(20, 20),
    position_bounds={'x_min': 5, 'x_max': 35, 'y_min': 5, 'y_max': 35}
)
pos, rss = optimizer.optimize(num_iterations=10, learning_rate=0.5)
```

### Grid Search

```python
from reflector_position import GridSearchAPOptimizer

optimizer = GridSearchAPOptimizer(
    scene,
    search_bounds={'x_min': 5, 'x_max': 35, 'y_min': 5, 'y_max': 35},
    grid_resolution=2.0
)
pos, rss = optimizer.optimize()
```

## Key Parameters

### Gradient Descent
- `num_iterations`: 10-50 (more = better convergence)
- `learning_rate`: 0.1-1.0 (higher = faster, less stable)
- `samples_per_tx`: 100k-1M (more = accurate, slower)
- `temperature`: 0.1-0.3 (lower = closer to hard min)

### Grid Search
- `grid_resolution`: 1-5 meters (finer = accurate, slower)
- `samples_per_tx`: 100k-500k
- `max_depth`: 10-15 reflections

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Slow optimization | Reduce `samples_per_tx` or `max_depth` |
| Poor convergence | Increase `learning_rate` or use `use_soft_min=True` |
| Out of bounds | Check `position_bounds` match your scene |
| Import errors | `pip install -e .` in package directory |

## File Locations

- **Source code**: `src/reflector_position/`
- **Examples**: `examples/`
- **Documentation**: `README.md`, `USAGE.md`, `INSTALL.md`
- **CLI entry**: `src/reflector_position/cli.py`

## Help Commands

```bash
# CLI help
reflector-optimize --help

# Python help
python -c "from reflector_position import GradientDescentAPOptimizer; help(GradientDescentAPOptimizer)"

# Check installation
python -c "import reflector_position; print(reflector_position.__version__)"
```
