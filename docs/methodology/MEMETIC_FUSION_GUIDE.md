# Memetic Fusion Guide

## 1. Overview & Theory

Indoor RF optimization with AP placement and reflector control is a high-dimensional, non-convex problem. Two practical failure modes appear repeatedly:

- **Curse of Dimensionality**: As we optimize more variables (AP x/y positions, AP orientations, reflector \(u,v\), reflector focal target), the search space grows combinatorially.
- **Local Minima Trapping**: Gradient-based methods can converge quickly to a nearby basin but miss better global layouts separated by walls, corners, or topology changes.

### Why Memetic Fusion (GA + GD)

The pipeline uses a **memetic strategy** to combine complementary strengths:

- **GA for macro-exploration**
  - Samples broad, discontinuous regions of the environment.
  - Can "jump" to candidate layouts separated by obstacles/walls.
  - Maintains population diversity via selection/mutation/crossover.
- **GD for micro-exploitation**
  - Starts from strong GA seeds.
  - Performs fine local refinement (small positional/orientation/reflector updates).
  - Efficiently improves smooth local neighborhoods.

This hybrid setup gives robust global exploration first, then efficient local polishing.

### Physics-Informed Distance Filter \(D_{corr}\)

Running GD on many near-duplicate GA candidates wastes compute. To avoid this, Phase-1 applies a topology-aware filter to Hall-of-Fame individuals:

- Keep Hall-of-Fame sorted by fitness.
- Always accept the best individual.
- Accept next candidates only if their AP topology distance to all selected seeds is at least \(D_{corr}\).
- Stop after collecting \(K\) seeds.

Conceptually, the selected set is:

\[
\mathcal{S} = \{s_1,\dots,s_K\} \quad \text{such that} \quad d(s_i,s_j) \ge D_{corr}, \; i \ne j
\]

Result: fewer redundant GD jobs and better coverage of distinct spatial basins.

---

## 2. Architecture & Module Breakdown

The implementation is in:

- `src/reflector_position/optimizers/memetic/memetic_ga_logic.py`
- `src/reflector_position/optimizers/memetic/memetic_bridge.py`
- `src/reflector_position/optimizers/memetic/memetic_gd_logic.py`
- `src/reflector_position/optimizers/memetic/run_memetic_pipeline.py`

### Phase 1 — GA Macro-Exploration
**File:** `memetic_ga_logic.py`

- Implements `MemeticGeneticAlgorithmRunner`.
- Adds `tools.HallOfFame` tracking (default size 50).
- Extracts topologically distinct seeds with `_extract_topological_seeds(k_seeds, d_corr)`.
- Returns seed-rich payload for downstream exploitation:
  - AP positions/orientations
  - reflector genes
  - fitness/coverage metadata

### Phase 2 — Memetic Bridge
**File:** `memetic_bridge.py`

- Implements `generate_gd_tasks_from_seeds(...)`.
- Converts GA seed schema into GD-ready Ray work items.
- Validates required fields (positions, reflector keys when enabled).
- Preserves input immutability (no mutation of caller-owned seed dicts).
- Injects shared GD hyperparameters into each task.

### Phase 3 — Targeted GD Micro-Exploitation
**File:** `memetic_gd_logic.py`

- Implements `run_targeted_gd_exploitation(...)`.
- Executes K translated tasks via `RayParallelOptimizer.run(..., optimizer_method="gradient_descent")`.
- Computes per-seed initial/final/delta metrics.
- Extracts global best final result from all GD outcomes.
- Handles empty-task and mixed return-contract edge cases.

### Phase 4 — Full Pipeline Orchestration
**File:** `run_memetic_pipeline.py`

- Implements `run_memetic_optimization(config_args)`.
- Orchestrates end-to-end flow:
  1. Initialize Ray and actor pool
  2. Run GA and collect seeds
  3. Bridge seeds to GD tasks
  4. Run targeted GD and aggregate metrics
  5. Return summary with timings and best result

### Crucial Design Choice: Shared Hot Actor Pool

In `run_memetic_pipeline.py`, the same `RayActorPoolExecutor` resources are reused for both GA and GD phases by binding pool/worker state into `RayParallelOptimizer`.

Why this matters:

- Avoids repeated heavy scene initialization.
- Keeps Sionna scene graph and GPU contexts warm.
- Reduces startup overhead between phases.
- Improves practical throughput in long optimization runs.

---

## 3. Running the Pipeline (Execution Guide)

Run commands from the **project root** (`/home/hieule/research/reflector-position`).

### Option A (recommended, explicit interpreter)

```bash
cd /home/hieule/research/reflector-position
PYTHONPATH=src /home/hieule/research/reflector-position/.venv/bin/python \
  src/reflector_position/optimizers/memetic/run_memetic_pipeline.py
```

### Option B (activate environment first)

```bash
cd /home/hieule/research/reflector-position
source .venv/bin/activate
export PYTHONPATH=src
python src/reflector_position/optimizers/memetic/run_memetic_pipeline.py
```

## Key Configuration Parameters (in `run_memetic_pipeline.py`)

Primary knobs in `demo_config`:

- **GA search scope/intensity**
  - `pop_size`: number of individuals per generation.
  - `n_gen`: number of GA generations.
  - `cxpb`, `mutpb`, `tournsize`: GA dynamics.
  - `hof_size`: Hall-of-Fame archive size.
- **Seed extraction**
  - `k_seeds`: number of seeds forwarded to GD.
  - `d_corr`: topology distance threshold (meters) for seed distinctness.
- **GD exploitation behavior**
  - `num_iterations`: local optimization length.
  - `learning_rate`: step size for updates.
  - `temperature`: softness of soft-min objective aggregation.
  - `shadow_quantile`: fairness/risk sensitivity to lower-tail coverage.
  - `alpha`, `beta`: objective mixing weights.
- **Ray/runtime**
  - `num_pool_workers`, `gpu_fraction`: distributed resource profile.
  - `samples_per_tx`, `max_depth`: ray-tracing fidelity/performance controls.

### Quick Test vs Deep Run Tuning

- **Quick smoke runs (fast feedback)**
  - Lower `pop_size` (e.g., 20–40)
  - Lower `n_gen` (e.g., 5–10)
  - Smaller `num_iterations` (e.g., 20–50)
  - Fewer workers or lower sampling (`samples_per_tx`)
  - Keep `k_seeds` small (e.g., 2–3)
- **Deep research runs (higher quality)**
  - Increase `pop_size` (e.g., 80–200)
  - Increase `n_gen` (e.g., 20–80)
  - Increase `num_iterations` (e.g., 100–500)
  - Raise ray-tracing fidelity (`samples_per_tx`, `max_depth`)
  - Use larger `k_seeds` with meaningful `d_corr`

Practical note:

- If `d_corr` is too small, many near-duplicate seeds pass through.
- If `d_corr` is too large, you may get fewer than `k_seeds` accepted seeds.

---

## 4. Testing & Validation

## Unit Test Command

```bash
cd /home/hieule/research/reflector-position
/home/hieule/research/reflector-position/.venv/bin/python -m pytest tests/test_memetic_bridge.py -v
```

Alternative concise form:

```bash
/home/hieule/research/reflector-position/.venv/bin/python -m pytest -q tests/test_memetic_bridge.py
```

`tests/test_memetic_bridge.py` currently validates:

- Correct schema translation from GA seeds to GD tasks.
- Reflector-key validation failure behavior.
- Input seed immutability.

## What to Monitor During a Full Run

### GA Phase

- **Hall-of-Fame quality:** best fitness trend in dBm improves across generations.
- **Seed extraction count:** `Selected seeds: X / k_seeds` at GA completion.
- **Spatial/topological separation:** accepted seeds should represent distinct AP layouts, not micro-perturbations of one location.
- **Wall-clock vs evals:** ensure runtime is consistent with expected `pop_size × n_gen` scale.

### GD Phase

- **Per-seed delta table:** check `Initial GA (dBm)`, `Final GD (dBm)`, and `Delta (dB)`.
- **Positive delta signal:** most successful exploitation runs show \(\Delta > 0\) dB.
- **Global best extraction:** confirm final summary reports the best task/worker and best metric.
- **Failure patterns:** missing results, no-improvement seeds, or inconsistent optimize params indicate integration/config issues.

### Resource/Systems Checks

- GPU memory usage remains stable across GA→GD transition (hot pool reuse works).
- Ray workers remain alive; no repeated cold-start spikes between phases.
- End-to-end timing breakdown (`ga_duration_sec`, `gd_duration_sec`, `total_duration_sec`) is reported.

---

## Minimal Repro Checklist

1. Use the project `.venv` interpreter.
2. Ensure `PYTHONPATH=src` is set.
3. Confirm scene path in `_default_scene_config()` exists on your machine.
4. Run bridge unit tests first.
5. Run full pipeline script.
6. Verify GD delta improvements and global best summary.

---

## File Map (for Researchers)

- Fusion GA logic: `src/reflector_position/optimizers/memetic/memetic_ga_logic.py`
- Seed translation bridge: `src/reflector_position/optimizers/memetic/memetic_bridge.py`
- Targeted GD exploitation: `src/reflector_position/optimizers/memetic/memetic_gd_logic.py`
- End-to-end orchestrator: `src/reflector_position/optimizers/memetic/run_memetic_pipeline.py`
- Bridge unit test: `tests/test_memetic_bridge.py`

This guide is intended as the operational reference for running and evaluating the Memetic Fusion workflow in this repository.
