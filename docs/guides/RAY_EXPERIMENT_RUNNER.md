# Ray Experiment Runner (Unified)

This project now uses a single script for Ray-based hyperparameter automation:

- `examples/ray_experiment_runner.py`

It replaces the old split workflow and enforces one trial = one method run.

## Prerequisites

```bash
source .venv/bin/activate
```

Ensure Ray and project dependencies are installed in the active environment.

## What this runner solves

- One unified config structure
- No repeated execution of unrelated methods
- Sweep generation for hyperparameter combinations
- Per-trial logs and summary exports (`CSV` + `JSON`)

## Core rule

Each trial must specify exactly:

- `method`: `gd` or `gs` or `ga`
- `mode`: `1ap` or `2ap`

This avoids the old repeated runs where one trial ran multiple optimization techniques.

## Config structure

Use `examples/ray_experiment_runner_config.example.json` as the template.

Top-level keys:

- `shared`: defaults merged into every trial/group
- `trials`: explicit trials (manual)
- `sweep_groups`: auto-generated trial groups from Cartesian grids

### `sweep_groups` format

Each group supports:

- `name_prefix` (optional)
- `method` (required)
- `mode` (required; string or list)
- `random_seed` (required; int or list)
- `base` (optional object)
- `grid` (optional object of key -> list)

`grid` keys support dotted paths, for example:

- `gd_optimization_overrides.learning_rate`
- `ga_params.pop_size`
- `gs_grid_resolution`

## 1. Generate expanded config (no run)

```bash
python examples/ray_experiment_runner.py \
  --config examples/ray_experiment_runner_config.example.json \
  --generate-only \
  --generated-config results/generated_trials.json
```

This writes explicit `trials` after expanding all sweep combinations.

You can open `results/generated_trials.json` to review exactly what will run.

## 2. Run hyperparameter experiments

```bash
python examples/ray_experiment_runner.py \
  --config examples/ray_experiment_runner_config.example.json \
  --output-root results/experiments
```

## 3. Run using the generated config (explicit trials)

```bash
python examples/ray_experiment_runner.py \
  --config results/generated_trials.json \
  --output-root results/experiments
```

Use this when you want a frozen, auditable list of trials.

## Output layout

Each run creates a timestamped folder:

- `results/experiments/ray_experiments_<timestamp>/`

Inside:

- `used_config.json`
- `summary.csv`
- `summary.json`
- `all_trials_detailed.json`
- One subfolder per trial with:
  - `output.txt`
  - `trial_record.json`
  - method-specific result files/plots from `ray_parallel_example.py`

## Typical tuning patterns

### Tune only 2-AP GD

Set a `sweep_group` with:

- `method: "gd"`
- `mode: "2ap"`
- grid keys under `gd_optimization_overrides.*`

### Tune only 2-AP GA

Set a `sweep_group` with:

- `method: "ga"`
- `mode: "2ap"`
- grid keys under `ga_params.*`

### Tune only 1-AP GS resolution

Set a `sweep_group` with:

- `method: "gs"`
- `mode: "1ap"`
- `grid: { "gs_grid_resolution": [...] }`
