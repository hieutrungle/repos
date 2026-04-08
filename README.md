# Reflector Position Optimization

Physics-aware AP and reflector placement using a memetic pipeline (GA exploration plus GD exploitation) on Sionna ray tracing.

## Current Scope

The code under src/reflector_position is now memetic-first.

- Primary runtime path: src/reflector_position/optimizers/memetic/run_memetic_pipeline.py
- Worker execution: src/reflector_position/optimizers/memetic/raw_ray_parallel_optimizer.py
- GA phase: src/reflector_position/optimizers/memetic/memetic_ga_logic.py
- Seed bridge: src/reflector_position/optimizers/memetic/memetic_bridge.py
- GD phase: src/reflector_position/optimizers/memetic/memetic_gd_logic.py and src/reflector_position/optimizers/memetic/memetic_gd_optimizer.py
- Loss and reporting: src/reflector_position/optimizers/memetic/memetic_loss.py and src/reflector_position/metrics.py
- Artifacts and reporting: src/reflector_position/optimizers/memetic/memetic_plotting.py and src/reflector_position/optimizers/memetic/memetic_summary.py

Important:

- There is no installed reflector-optimize console command in pyproject.toml.
- The optimizer package currently exports BaseAPOptimizer only from src/reflector_position/optimizers/__init__.py.
- The package still includes compatibility config dataclasses in src/reflector_position/config.py.

## Installation

```bash
cd reflector-position
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start (Memetic Pipeline)

Run one memetic optimization run:

```bash
python run_memetic_pipeline.py --config configs/memetic_pipeline_config.json
```

Show launcher options and helper notes:

```bash
python run_memetic_pipeline.py --help
python run_memetic_pipeline.py --hints
```

Run a hyperparameter sweep:

```bash
python scripts/run_memetic_hparam_sweep.py \
  --base-config configs/memetic_pipeline_config.json \
  --sweep-config configs/memetic_hparam_sweep.example.json
```

Run AP-count by seed sweeps:

```bash
python scripts/run_num_aps_seed_sweep.py --help
```

## Unified Experiments Runner (Memetic + Baselines)

Use the unified runner to execute one method or compare multiple methods with
shared config and memetic-style artifacts.

Recommended config:

- configs/run_experiments_cuda_hrbb.json

Run one method:

```bash
python scripts/run_experiments.py \
  --method random_gd \
  --config configs/run_experiments_cuda_hrbb.json \
  --output_dir results/experiments \
  --run_name hrbb_random_gd
```

Run all baselines in one command:

```bash
python scripts/run_experiments.py \
  --method all_baselines \
  --config configs/run_experiments_cuda_hrbb.json \
  --output_dir results/experiments \
  --run_name hrbb_baselines
```

Run all methods (memetic + baselines):

```bash
python scripts/run_experiments.py \
  --method all \
  --config configs/run_experiments_cuda_hrbb.json \
  --output_dir results/experiments \
  --run_name hrbb_all
```

Important:

- Baseline methods are forced to CUDA/GPU execution in scripts/run_experiments.py.
- The runner keeps workers persistent for each method run and destroys them only
  after that method finishes.
- Per-method iteration traces are exported and plotted automatically.
- Output layout mirrors memetic style with artifacts and plots folders.

## Pipeline Flow

The memetic pipeline executes in three phases with shared Ray actors:

1. GA macro-exploration generates diverse high-quality seeds.
2. Bridge conversion maps GA seeds to GD task payloads.
3. Targeted GD micro-exploitation refines each seed and selects the global best result.

The same actor pool is reused across GA and GD to avoid repeated scene warmup and reduce overhead.

## Configuration (Top-Level Keys)

Primary keys consumed by run_memetic_optimization:

- scene_config
- position_bounds
- fixed_z
- num_pool_workers
- gpu_fraction
- random_seed
- num_aps
- min_ap_separation
- optimize_orientation
- reflector_enabled
- focal_z
- demand_config
- objective_params
- ga_params
- ga_evaluation_params
- k_seeds
- d_corr
- gd_optimization_params
- coverage_plot_settings
- camera
- output_dir
- run_name
- verbose

Legacy compatibility keys are still accepted:

- ga_optimization_params (fallback path)
- gd_hyperparams (fallback path)

## Output Artifacts

Each run writes to:

- output_dir/run_name (if run_name provided), or
- output_dir/run_YYYYMMDD_HHMMSS

Key artifacts:

- artifacts/memetic_summary.json
- artifacts/ga_results.json
- artifacts/gd_results.json
- artifacts/global_best_result.json
- artifacts/run_config.json
- artifacts/memetic_report.md
- artifacts/ga_generation_details.csv (when GA generation details are present)
- artifacts/gd_per_seed_analysis.csv (when per-seed GD analysis is present)
- plots/* (trend plots, trajectory plots, coverage maps)

Unified runner artifacts (scripts/run_experiments.py):

- `RUN_DIR/artifacts/experiment_summary.json`
- `RUN_DIR/artifacts/method_summary.csv`
- `RUN_DIR/artifacts/METHOD_results.json`
- `RUN_DIR/artifacts/METHOD_iteration_trace.csv`
- `RUN_DIR/plots/METHOD_trend.html`
- `RUN_DIR/plots/method_comparison_trend.html` (when running multiple methods)

## Python API

Minimal programmatic usage:

```python
from reflector_position.optimizers.memetic.run_memetic_pipeline import run_memetic_optimization

config = {
    "scene_config": {
        "scene_path": "/path/to/scene.xml",
        "frequency": 5.18e9,
        "tx_power_dbm": 5.0,
        "tx_positions": [(10.0, 10.0, 3.8), (25.0, 25.0, 3.8)],
        "reflector_enabled": True,
        "reflector_size": (2.0, 2.0),
        "wall_top_left": [15.0, 34.0, 3.0],
        "wall_bottom_right": [34.0, 34.0, 1.0],
        "focal_point": [20.0, 20.0, 1.5],
        "device": "cuda",
    },
    "position_bounds": {"x_min": 5.5, "x_max": 34.5, "y_min": 5.5, "y_max": 34.5},
    "num_aps": 2,
    "num_pool_workers": 2,
    "gpu_fraction": 0.5,
    "ga_params": {"pop_size": 80, "n_gen": 20, "cxpb": 0.7, "mutpb": 0.3, "tournsize": 3, "hof_size": 10},
    "ga_evaluation_params": {"samples_per_tx": 1_000_000, "max_depth": 13, "verbose": False},
    "gd_optimization_params": {"num_iterations": 50, "learning_rate": 0.1, "samples_per_tx": 1_000_000, "max_depth": 13, "verbose": False},
    "objective_params": {
        "alpha": 0.95,
        "beta": 0.05,
        "softmin_temperature": 0.15,
        "softmin_floor_dbm": -120.0,
        "softmin_ceil_dbm": -70.0,
        "coverage_threshold_dbm": -120.0,
        "coverage_temperature": 2.0,
    },
    "k_seeds": 3,
    "d_corr": 5.0,
    "output_dir": "results/experiments",
    "verbose": True,
}

summary = run_memetic_optimization(config)
print(summary["saved_artifacts"]["output_dir"])
```

## Package Surface Summary

Top-level exports in src/reflector_position/__init__.py currently include:

- BaseAPOptimizer
- setup_building_floor_scene, create_camera
- ReflectorController, create_flat_reflector_mesh
- metrics helpers (rss_to_dbm, dbm_to_rss, softmin and coverage utilities)
- compute_radio_map_with_tx_position
- SceneConfig, GridSearchConfig, GradientDescentConfig, OptimizationConfig

Memetic subpackage exports in src/reflector_position/optimizers/memetic/__init__.py include:

- MemeticGeneticAlgorithmRunner, MemeticSeed
- generate_gd_tasks_from_seeds
- run_targeted_gd_exploitation
- save_memetic_plots, save_memetic_summary_report
- RawRayActorPoolExecutor, RawRayParallelOptimizer, RawOptimizationWorker

## Source Layout (Current)

```text
src/reflector_position/
  __init__.py
  config.py
  metrics.py
  reflector_model.py
  scene_setup.py
  utils.py
  optimizers/
    __init__.py
    base_optimizer.py
    memetic/
      __init__.py
      demand_weights.py
      memetic_bridge.py
      memetic_ga_evaluator.py
      memetic_ga_logic.py
      memetic_gd_logic.py
      memetic_gd_optimizer.py
      memetic_loss.py
      memetic_plotting.py
      memetic_summary.py
      raw_ray_parallel_optimizer.py
      run_memetic_pipeline.py
```

## Completed Features (Done)

### Pipeline Execution

- [x] Three-phase memetic flow (GA -> bridge -> GD) implemented in src/reflector_position/optimizers/memetic/run_memetic_pipeline.py.
- [x] Shared Ray actor pool is reused across GA and GD phases to avoid repeated scene warmup.
- [x] Bridge payload generation converts GA seeds into GD-ready task schemas.

### Optimization Modules

- [x] GA exploration runner and evaluator are implemented in memetic_ga_logic.py and memetic_ga_evaluator.py.
- [x] Targeted GD exploitation is implemented in memetic_gd_logic.py and memetic_gd_optimizer.py.
- [x] Demand-weighted reporting and objective handling are integrated via demand_weights.py and memetic_loss.py.
- [x] num_aps-aware scene setup supports truncation and bounds-aware TX position auto-generation when needed.

### Artifacts and Reporting

- [x] Run artifact bundle is saved per execution (summary JSONs, config snapshot, global-best payload).
- [x] GA generation CSV and GD per-seed CSV exports are produced when data is available.
- [x] Trend plots, coverage maps, and markdown summary reporting are generated by memetic_plotting.py and memetic_summary.py.

### Public Package Surface

- [x] Memetic subpackage exports orchestration and executor building blocks.
- [x] Top-level package exports scene helpers, reflector model helpers, and metrics utilities.
- [x] Script-first workflow is established; no packaged reflector-optimize console entry point.

## Future Work (Planned)

### High Priority

- [ ] Add script-interface tests for launcher argument parsing and config override precedence.
- [ ] Expand integration tests for shared actor-pool reuse and failure handling across GA and GD phases.
- [ ] Strengthen config schema validation and user-facing error diagnostics for malformed configs.

### Performance and Scale

- [ ] Add caching for repeated scene/radio-map computations where correctness permits.
- [ ] Add resumable checkpoints for long hyperparameter sweeps.
- [ ] Add benchmark suites for worker-scaling behavior across gpu_fraction and pool sizes.

### Analysis and Usability

- [ ] Expand automated comparative reporting for sweep outputs and AP-count studies.
- [ ] Improve docs around demand-weight interpretation and priority-area metrics.
- [ ] Add optional structured logging outputs for experiment tracking pipelines.

## Documentation

- docs/memetic_fusion_pipeline.md
- docs/README.md
- docs/architecture/PROJECT_STRUCTURE.md
- docs/methodology/

## Development

Run tests:

```bash
pytest
```

Run formatting and linting:

```bash
black src tests scripts
ruff check src tests scripts
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This project uses:

- [Sionna](https://nvlabs.github.io/sionna/) for differentiable ray tracing
- [PyTorch](https://pytorch.org/) for gradient computation
- [DrJit](https://github.com/mitsuba-renderer/drjit) for PyTorch-Mitsuba integration

## Citation

If you use this code in your research, please cite:

```bibtex
@software{reflector_position,
  title = {Reflector Position Optimization},
  author = {Your Name},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/yourusername/reflector-position}
}
```
