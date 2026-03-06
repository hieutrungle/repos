# Ray Experiment Runner — Comprehensive Guide

> **Script**: `examples/ray_experiment_runner.py`
> **Config template**: `examples/ray_experiment_runner_config.example.json`

A unified, config-driven experiment runner that executes Gradient Descent (GD),
Grid Search (GS), and Genetic Algorithm (GA) trials — with optional IRS
reflector optimisation — over a Ray cluster and collects structured results.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [CLI Reference](#cli-reference)
5. [Config File Schema](#config-file-schema)
   - [Top-Level Structure](#top-level-structure)
   - [shared](#shared)
   - [trials](#trials)
   - [sweep_groups](#sweep_groups)
6. [Trial Parameters — Full Reference](#trial-parameters--full-reference)
   - [Common Parameters](#common-parameters)
   - [Gradient Descent (gd)](#gradient-descent-gd)
   - [Grid Search (gs)](#grid-search-gs)
   - [Genetic Algorithm (ga)](#genetic-algorithm-ga)
   - [Reflector Geometry](#reflector-geometry)
7. [Modes](#modes)
8. [Output Layout](#output-layout)
9. [Summary Files](#summary-files)
10. [Workflows & Recipes](#workflows--recipes)
11. [Config File Examples](#config-file-examples)
12. [Tips & Troubleshooting](#tips--troubleshooting)

---

## Overview

### What this runner does

- Reads a single JSON config that describes **explicit trials** and/or
  **sweep groups** (Cartesian product over hyperparameter grids).
- Expands sweep groups into concrete trials.
- Executes each trial sequentially (one method × one mode per trial).
- Captures stdout/stderr per trial to `output.txt`.
- Writes per-trial `trial_record.json` and global `summary.csv` / `summary.json`.

### Core design rules

| Rule | Detail |
|------|--------|
| **One method per trial** | `gd`, `gs`, or `ga` — never combined. |
| **One mode per trial** | `1ap`, `2ap`, or `2ap_reflector`. |
| **Shared defaults** | The `shared` object is deep-merged into every trial. |
| **Sweep groups auto-expand** | Cartesian product of `grid` values × `random_seed` × `mode`. |
| **Deterministic naming** | Auto-generated trial names encode key hyperparameters. |

---

## Prerequisites

```bash
# 1. Activate the project virtual environment
source .venv/bin/activate

# 2. Ensure Ray is installed
pip install ray[default]

# 3. Verify imports work
python -c "import ray; import ray_parallel_example; print('OK')"
```

The runner imports `ray_parallel_example` (the main experiment module) and
`ray`. Both must be importable from the working directory.

---

## Quick Start

### 1. Preview what will run (no GPU required)

```bash
python examples/ray_experiment_runner.py \
  --config examples/ray_experiment_runner_config.example.json \
  --generate-only \
  --generated-config results/generated_trials.json
```

Open `results/generated_trials.json` to inspect every expanded trial and its
merged parameters.

### 2. Run all trials

```bash
python examples/ray_experiment_runner.py \
  --config examples/ray_experiment_runner_config.example.json \
  --output-root results/experiments
```

### 3. Run from the generated (frozen) config

```bash
python examples/ray_experiment_runner.py \
  --config results/generated_trials.json \
  --output-root results/experiments
```

Use this when you want a fully auditable, reproducible trial list.

---

## CLI Reference

```
python examples/ray_experiment_runner.py [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `Path` | `examples/ray_experiment_runner_config.example.json` | Path to the runner config JSON. |
| `--output-root` | `Path` | `results/experiments` | Root directory for timestamped run outputs. |
| `--generate-only` | flag | — | Expand config into explicit trials without executing. |
| `--generated-config` | `Path` | `results/generated_trials.json` | Output path for `--generate-only`. |

---

## Config File Schema

### Top-Level Structure

```jsonc
{
  "shared": { ... },        // Defaults merged into every trial
  "trials": [ ... ],        // Explicit trial definitions
  "sweep_groups": [ ... ]   // Auto-generated trials (Cartesian grid)
}
```

All three sections are optional (but at least `trials` or `sweep_groups` must
produce at least one trial).

---

### `shared`

A flat or nested object of default values. Every explicit trial and every
sweep-group-generated trial receives a **deep merge** of `shared` as its base.
Trial-level values override `shared`.

```jsonc
{
  "shared": {
    // --- Ray pool ---
    "num_pool_workers": 2,
    "gpu_fraction": 0.5,
    "random_seed": 4,

    // --- GD defaults ---
    "gd_num_tasks": 100,
    "gd_num_iterations": 50,
    "gd_samples_per_tx": 1000000,
    "gd_repulsion_weight": 0.2,
    "gd_fairness_loss_type": "auto",

    // --- GA defaults ---
    "ga_min_ap_separation": 4.0,

    // --- Reflector geometry ---
    "reflector_wall_top_left": [15.0, 34.0, 3.0],
    "reflector_wall_bottom_right": [34.0, 34.0, 1.0],
    "reflector_size": [2.0, 2.0],
    "reflector_focal_z": 1.5,
    "reflector_target_quantile": 0.05
  }
}
```

---

### `trials`

An array of explicit trial objects. Each object must contain at minimum
`method` and `mode`. An optional `name` gives the trial a human-readable
label (auto-generated if omitted). Comment-only entries (with just
`_comment`) are allowed and silently skipped during validation.

```jsonc
{
  "trials": [
    { "_comment": "--- Baselines ---" },
    {
      "name": "gd_2ap_baseline",
      "method": "gd",
      "mode": "2ap",
      "random_seed": 4,
      "gd_optimization_overrides": {
        "learning_rate": 0.5,
        "temperature": 0.15
      }
    }
  ]
}
```

**Merge behavior**: Each trial object is deep-merged on top of `shared`.
Keys in the trial override `shared` values. Nested dicts (e.g.
`gd_optimization_overrides`) are merged recursively — you only need to
specify the keys you want to change.

---

### `sweep_groups`

Each sweep group generates a Cartesian product of trials from a parameter
grid, crossed with seed(s) and mode(s).

```jsonc
{
  "sweep_groups": [
    {
      "_comment": "Optional description",
      "name_prefix": "gd_refl_lr_temp",   // Prefix for auto-generated names
      "method": "gd",                      // Required: "gd" | "gs" | "ga"
      "mode": "2ap_reflector",             // Required: string or list of strings
      "random_seed": [51, 52, 53],         // Required: int or list of ints
      "base": {                            // Optional: extra defaults (merged on shared)
        "gd_num_tasks": 64
      },
      "grid": {                            // Parameter grid — Cartesian product
        "gd_optimization_overrides.learning_rate": [0.3, 0.5, 0.7],
        "gd_optimization_overrides.temperature": [0.1, 0.15, 0.2]
      }
    }
  ]
}
```

#### How expansion works

Given the above group:
- 3 learning rates × 3 temperatures × 3 seeds = **27 trials**.
- Each trial inherits `shared` → `base` → grid combo.

#### Dotted keys

Grid keys support dot notation to set nested values:

| Key | Sets |
|-----|------|
| `gd_optimization_overrides.learning_rate` | `trial["gd_optimization_overrides"]["learning_rate"]` |
| `ga_params.pop_size` | `trial["ga_params"]["pop_size"]` |
| `ga_params.mut_sigma_pos` | `trial["ga_params"]["mut_sigma_pos"]` |

#### Multi-mode and multi-seed

Both `mode` and `random_seed` accept a list to generate cross-product trials:

```jsonc
{
  "mode": ["2ap", "2ap_reflector"],
  "random_seed": [51, 52]
}
```

This alone produces 4 trials (2 modes × 2 seeds) before the grid product.

---

## Trial Parameters — Full Reference

### Common Parameters

These apply to every trial regardless of method.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `method` | `str` | *required* | `"gd"`, `"gs"`, or `"ga"` |
| `mode` | `str` | *required* | `"1ap"`, `"2ap"`, or `"2ap_reflector"` |
| `name` | `str` | auto | Human-readable trial name (sanitised to `[a-zA-Z0-9_-]`) |
| `random_seed` | `int` | `4` | Random seed for reproducibility |
| `num_pool_workers` | `int` | `2` | Number of Ray actors in the worker pool |
| `gpu_fraction` | `float` | `0.5` | GPU fraction per worker (e.g. `0.5` = 2 workers/GPU) |

---

### Gradient Descent (`gd`)

SPSA-based gradient descent with random restarts.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `gd_num_tasks` | `int` | `100` | Number of random-restart trajectories |
| `gd_num_iterations` | `int` | `50` | Iterations per trajectory |
| `gd_samples_per_tx` | `int` | `1000000` | Ray-tracing samples per transmitter |
| `gd_repulsion_weight` | `float` | `0.3` | AP repulsion loss weight (multi-AP) |
| `gd_fairness_loss_type` | `str` | `"auto"` | Fairness loss type (see below) |
| `gd_optimization_overrides` | `object` | `{}` | Passed directly to the optimizer (see below) |

#### `gd_fairness_loss_type` values

| Value | Description |
|-------|-------------|
| `"auto"` | Automatically selects best loss for the mode |
| `"softmin"` | Differentiable soft-minimum approximation |
| `"masked_softmin"` | Soft-minimum with coverage masking |
| `"percentile"` | 5th-percentile RSS objective |

#### `gd_optimization_overrides` keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `learning_rate` | `float` | `0.5` | SPSA step size |
| `temperature` | `float` | `0.15` | Soft-min temperature |
| `alpha` | `float` | `0.9` | Primary objective weight |
| `beta` | `float` | `0.1` | Regularisation weight |
| `shadow_quantile` | `float` | `0.05` | Quantile for percentile loss (reflector mode) |
| `coverage_threshold_dbm` | `float` | `-120.0` | Coverage threshold (reflector mode) |
| `coverage_temperature` | `float` | `2.0` | Coverage sigmoid temperature (reflector mode) |

---

### Grid Search (`gs`)

Exhaustive grid evaluation. In `2ap`/`2ap_reflector` mode, uses alternating
optimisation (sweep one AP while fixing the other).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `gs_grid_resolution` | `float` | `1.0` | Grid spacing in metres |
| `gs_num_rounds` | `int` | `3` | Alternating sweeps per outer round (2AP modes) |
| `gs_outer_rounds` | `int` | `3` | AP-sweep → reflector-sweep outer loops (reflector mode) |
| `gs_u_steps` | `int` | `3` | Reflector u-axis grid divisions (reflector mode) |
| `gs_v_steps` | `int` | `3` | Reflector v-axis grid divisions (reflector mode) |
| `gs_target_resolution` | `float` | `10.0` | Reflector focal-target grid spacing in metres |
| `gs_min_ap_separation` | `float` | `10.0` | Minimum allowed distance between APs (metres) |
| `gs_target_quantile` | `float` | `0.05` | Percentile for ranking (reflector mode) |

---

### Genetic Algorithm (`ga`)

DEAP-based evolutionary algorithm with tournament selection.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ga_min_ap_separation` | `float` | `5.0` | Minimum AP–AP distance enforced via penalty |
| `ga_target_quantile` | `float` | `0.05` | Percentile objective quantile (reflector mode) |
| `ga_focal_z` | `float` | `1.5` | Reflector focal-point z-height (reflector mode) |
| `ga_params` | `object` | `{}` | Evolutionary parameters (see below) |

#### `ga_params` keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `pop_size` | `int` | `150` | Population size |
| `n_gen` | `int` | `50` | Number of generations |
| `cxpb` | `float` | `0.7` | Crossover probability |
| `mutpb` | `float` | `0.3` | Mutation probability |
| `tournsize` | `int` | `10` | Tournament selection size |
| `cx_alpha` | `float` | `0.5` | BLX-α crossover blend factor |
| `mut_mu` | `float` | `0.0` | Gaussian mutation mean |
| `mut_sigma` | `float` | `2.0` | Base Gaussian mutation std (fallback) |
| `mut_sigma_pos` | `float` | `2.0` | Position gene mutation std (metres) |
| `mut_sigma_dir` | `float` | `0.3` | Direction gene mutation std (radians) |
| `mut_indpb` | `float` | `0.2` | Per-gene mutation probability |
| `hof_size` | `int` | `5` | Hall-of-fame (best individuals kept) |

---

### Reflector Geometry

These keys configure the IRS reflector surface and are used by any trial with
`mode: "2ap_reflector"`. They can be set in `shared` or overridden per trial.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `reflector_wall_top_left` | `[x,y,z]` | `[15.0, 34.0, 3.0]` | Top-left corner of the wall hosting the reflector |
| `reflector_wall_bottom_right` | `[x,y,z]` | `[34.0, 34.0, 1.0]` | Bottom-right corner of the wall |
| `reflector_size` | `[w, h]` | `[2.0, 2.0]` | Reflector panel size in metres |
| `reflector_focal_z` | `float` | `1.5` | Z-height of the reflector focal point |
| `reflector_target_quantile` | `float` | `0.05` | Target quantile for the P5 RSS objective |

The reflector is parameterised by `(u, v)` coordinates on the wall surface
(both in `[0, 1]`), plus a focal point `(x, y, z)` that the reflector orients
toward. All three optimization methods (GD, GS, GA) jointly optimize AP
positions and reflector configuration.

---

## Modes

| Mode | APs | Reflector | Chromosome (GA) | Description |
|------|-----|-----------|------------------|-------------|
| `1ap` | 1 | No | 5 genes: `[x, y, dx, dy, dz]` | Single AP placement |
| `2ap` | 2 | No | 10 genes: `[x₁,y₁,dx₁,dy₁,dz₁, x₂,y₂,dx₂,dy₂,dz₂]` | Dual AP placement |
| `2ap_reflector` | 2 | Yes | 12 genes: `[...2ap..., refl_u, refl_v, focal_x, focal_y]` | Dual AP + IRS reflector |

**Objective metric**: All modes report **P5 RSS** (5th-percentile received signal
strength in dBm) — the signal level exceeded by 95% of receiver locations.
This is a shadow-robust metric that emphasises worst-case coverage.

---

## Output Layout

Each run creates a timestamped directory:

```
results/experiments/ray_experiments_20260227_143052/
├── used_config.json                  # Copy of the input config
├── summary.csv                       # One row per trial (spreadsheet-friendly)
├── summary.json                      # Same data as JSON
├── all_trials_detailed.json          # Full results + config per trial
│
├── gd_2ap_reflector_baseline/        # Trial subfolder
│   ├── output.txt                    # Captured stdout + stderr
│   ├── trial_record.json             # Trial config + full result dict
│   ├── gd_2ap_results.json           # Method-specific result file
│   ├── gd_2ap_parallel_results.png   # Overview plot
│   └── gd_2ap_trajectories/          # Per-task trajectory plots (GD)
│
├── gs_2ap_reflector_baseline/
│   ├── output.txt
│   ├── trial_record.json
│   └── ...
│
└── ga_2ap_reflector_baseline/
    ├── output.txt
    ├── trial_record.json
    └── ...
```

---

## Summary Files

### `summary.csv` / `summary.json`

Each row contains:

| Column | Description |
|--------|-------------|
| `trial` | Trial name |
| `method` | `gd`, `gs`, or `ga` |
| `mode` | `1ap`, `2ap`, or `2ap_reflector` |
| `random_seed` | Seed used |
| `best_p5_rss_dbm` | Best 5th-percentile RSS in dBm (higher is better) |
| `time_s` | Wall-clock time in seconds |
| `run_dir` | Path to the trial's output directory |
| `reflector_u` | Reflector u-param (reflector trials only) |
| `reflector_v` | Reflector v-param (reflector trials only) |
| `focal_x` | Reflector focal-point x (reflector trials only) |
| `focal_y` | Reflector focal-point y (reflector trials only) |

### `all_trials_detailed.json`

Array of objects, each with:
- `trial`: name
- `config`: full merged trial config
- `result`: raw result dict from the optimization method
- `summary`: the summary row

---

## Workflows & Recipes

### Recipe 1: Single baseline comparison (AP-only vs. reflector)

```jsonc
{
  "shared": {
    "num_pool_workers": 2,
    "gpu_fraction": 0.5
  },
  "trials": [
    {
      "name": "gd_2ap_baseline",
      "method": "gd",
      "mode": "2ap",
      "random_seed": 4,
      "gd_num_tasks": 100,
      "gd_num_iterations": 50
    },
    {
      "name": "gd_2ap_reflector",
      "method": "gd",
      "mode": "2ap_reflector",
      "random_seed": 4,
      "gd_num_tasks": 64,
      "gd_num_iterations": 30,
      "gd_fairness_loss_type": "auto"
    }
  ]
}
```

### Recipe 2: Sweep GD learning rate with reflector

```jsonc
{
  "shared": { "num_pool_workers": 2, "gpu_fraction": 0.5 },
  "sweep_groups": [
    {
      "name_prefix": "gd_lr_sweep",
      "method": "gd",
      "mode": "2ap_reflector",
      "random_seed": [51, 52, 53],
      "base": {
        "gd_num_tasks": 64,
        "gd_num_iterations": 30
      },
      "grid": {
        "gd_optimization_overrides.learning_rate": [0.1, 0.3, 0.5, 0.7, 1.0]
      }
    }
  ]
}
```

This produces 5 LR values × 3 seeds = **15 trials**.

### Recipe 3: Sweep GA evolutionary parameters

```jsonc
{
  "shared": { "num_pool_workers": 2, "gpu_fraction": 0.5 },
  "sweep_groups": [
    {
      "name_prefix": "ga_evo_sweep",
      "method": "ga",
      "mode": "2ap_reflector",
      "random_seed": [51, 52],
      "grid": {
        "ga_params.pop_size": [100, 150, 200],
        "ga_params.n_gen": [30, 50, 80],
        "ga_params.mutpb": [0.2, 0.3]
      }
    }
  ]
}
```

3 pop × 3 gen × 2 mut × 2 seeds = **36 trials**.

### Recipe 4: Compare all three methods on reflector mode

```jsonc
{
  "shared": {
    "num_pool_workers": 2,
    "gpu_fraction": 0.5,
    "reflector_wall_top_left": [15.0, 34.0, 3.0],
    "reflector_wall_bottom_right": [34.0, 34.0, 1.0],
    "reflector_target_quantile": 0.05
  },
  "trials": [
    {
      "name": "gd_reflector",
      "method": "gd",
      "mode": "2ap_reflector",
      "gd_num_tasks": 64,
      "gd_num_iterations": 30
    },
    {
      "name": "gs_reflector",
      "method": "gs",
      "mode": "2ap_reflector",
      "gs_grid_resolution": 1.0,
      "gs_outer_rounds": 3
    },
    {
      "name": "ga_reflector",
      "method": "ga",
      "mode": "2ap_reflector",
      "ga_params": { "pop_size": 150, "n_gen": 50 }
    }
  ]
}
```

### Recipe 5: Multi-mode sweep (AP-only vs. reflector)

```jsonc
{
  "sweep_groups": [
    {
      "name_prefix": "gd_mode_compare",
      "method": "gd",
      "mode": ["2ap", "2ap_reflector"],
      "random_seed": [51, 52, 53],
      "grid": {
        "gd_optimization_overrides.learning_rate": [0.3, 0.5]
      }
    }
  ]
}
```

2 modes × 3 seeds × 2 LRs = **12 trials**.

### Recipe 6: Reflector grid search resolution sweep

```jsonc
{
  "sweep_groups": [
    {
      "name_prefix": "gs_refl_resolution",
      "method": "gs",
      "mode": "2ap_reflector",
      "random_seed": [51],
      "grid": {
        "gs_grid_resolution": [0.5, 1.0, 2.0],
        "gs_outer_rounds": [2, 3, 4],
        "gs_u_steps": [3, 5],
        "gs_v_steps": [3, 5]
      }
    }
  ]
}
```

---

## Config File Examples

### Minimal config (one trial)

```json
{
  "trials": [
    {
      "method": "gd",
      "mode": "2ap"
    }
  ]
}
```

All other parameters use defaults.

### Full example config

See `examples/ray_experiment_runner_config.example.json` for a complete
config with:
- AP-only baseline trials (GD, GA)
- Reflector baseline trials (GD, GS, GA)
- Sweep groups for GD, GS, and GA with reflector
- Legacy AP-only sweeps

---

## Tips & Troubleshooting

### Generate before running

Always run `--generate-only` first to review the expanded trial list:

```bash
python examples/ray_experiment_runner.py \
  --config my_config.json \
  --generate-only \
  --generated-config my_expanded.json

# Count trials
python -c "import json; t=json.load(open('my_expanded.json')); print(len(t['trials']), 'trials')"
```

### Estimate runtime

Each GD trial with 64 tasks × 30 iterations takes ~15–30 min (2 workers,
1 GPU). GA trials with pop=150 × 50 generations take ~30–60 min. Grid Search
depends on resolution. Plan accordingly when sweeping large grids.

### Resume after failure

The runner does not yet support resumption. If a run fails mid-way:

1. Open the timestamped output directory.
2. Check `summary.json` for completed trials.
3. Remove completed trials from your config.
4. Re-run with the trimmed config.

### Comment-only entries

You can add comment entries to the `trials` array for readability:

```json
{ "_comment": "--- Section: Reflector baselines ---" }
```

These are ignored during validation (they have no `method`/`mode`).
**Note**: Comment entries in `trials` that lack `method` will cause a
validation error. Use the `_comment` field alongside valid `method`/`mode`
fields in real trial objects, or keep comment-only entries as documentation
notes that you remove before running.

### Trial naming

If you omit `name`, the runner auto-generates one encoding the method, mode,
seed, and key hyperparameters:

```
trial_001_gd_2ap_reflector_seed4_lr05_t015_a095_b005_fltauto
trial_002_ga_2ap_reflector_seed51_p150_g50_m03_q005
```

Provide explicit `name` values for clarity in long sweep runs.

### GPU memory

With `gpu_fraction: 0.5`, each worker claims half a GPU. Set
`num_pool_workers` so that `num_pool_workers × gpu_fraction ≤ total_GPUs`.
For CPU-only runs (GA typically), the fraction is still reserved for the
worker but Ray-tracing may default to CPU.

### Logs

Every trial's full output is saved to `<trial_dir>/output.txt`. If a trial
fails, check this file first. The console also shows a live `[n/total]`
progress indicator.
