# Run Experiments Config Reference (HRBB Preset)

This guide explains every parameter in ../../configs/run_experiments_cuda_hrbb.json and how to tune it effectively.

## Quick tuning order

1. Set geometry and constraints first.
2. Set objective behavior next.
3. Set optimizer budgets.
4. Set simulation fidelity.
5. Enable iteration equalization only when comparing methods fairly.

## Runtime intuition

Most runtime comes from ray-tracing evaluations.

- GA cost is approximately pop_size x n_gen x samples_per_tx.
- Random baseline cost is approximately random_params.num_samples x samples_per_tx.
- Random plus GD cost is approximately random_gd_params.num_samples x gd_optimization_params.num_iterations x samples_per_tx.
- PSO plus GD cost is approximately pso_params.swarm_size x pso_params.num_iterations x samples_per_tx, plus GD refinement.

If jobs are too slow, lower samples_per_tx before changing algorithm logic.

## Top-level fields

- scene_config: Core RF simulator scene and execution settings.
- visualization_scene_config: Optional alternate scene for visualization outputs.
- position_bounds: Placement bounds for AP optimization.
- fixed_z: Fixed AP height.
- num_pool_workers: Number of Ray workers.
- gpu_fraction: GPU fraction assigned per worker.
- random_seed: Global random seed for reproducibility.
- num_aps: Number of APs to place.
- min_ap_separation: Minimum AP-to-AP distance constraint.
- optimize_orientation: Enables AP direction optimization.
- reflector_enabled: Legacy-compatible top-level reflector toggle.
- focal_z: Reflector focal height control.
- demand_config: Spatial-priority map definition.
- objective_params: Loss shaping and tradeoff controls.
- ga_params: GA search budget and evolutionary behavior.
- ga_evaluation_params: GA evaluator fidelity and depth.
- gd_optimization_params: GD refinement depth and step behavior.
- random_params: Random baseline budget.
- kmeans_params: K-means preprocessing density.
- random_gd_params: Number of random starts for random_gd baseline.
- pso_params: PSO behavior and budget.
- k_seeds: Number of GA seeds selected for GD stage.
- d_corr: Diversity control for seed selection.
- camera: Visualization camera defaults.
- coverage_plot_settings: Fidelity and output behavior for coverage-map artifacts.
- output_dir: Base output path.
- verbose: Verbose logging switch.
- iteration_equalization: Non-kmeans iteration-budget equalization.

## scene_config

- scene_config.scene_path: Path to the optimization scene XML. Keep this aligned with the geometry you want to optimize.
- scene_config.frequency: Carrier frequency in Hz. Match your target band, for example 5.18e9 for 5 GHz.
- scene_config.tx_power_dbm: TX power in dBm. Raising this improves RSSI globally and can reduce observable placement differences.
- scene_config.tx_positions: Anchor points as [x, y, z]. Use realistic user or receiver locations.
- scene_config.reflector_enabled: Enables reflector simulation path. Turn on only when reflector geometry is valid.
- scene_config.reflector_size: Reflector dimensions [u, v] in meters.
- scene_config.wall_top_left: One reflector wall boundary corner in scene coordinates.
- scene_config.wall_bottom_right: Opposite reflector wall boundary corner.
- scene_config.focal_point: Reflector focal point [x, y, z] in scene coordinates.
- scene_config.device: Device hint. Baselines in run_experiments are forced to CUDA when available.

## visualization_scene_config

- visualization_scene_config.scene_path: Scene used for visualization renders. Keep it consistent with scene_config.scene_path unless you intentionally use a render-only scene.

## position_bounds

- position_bounds.x_min: Minimum x bound.
- position_bounds.x_max: Maximum x bound.
- position_bounds.y_min: Minimum y bound.
- position_bounds.y_max: Maximum y bound.

Tuning guidance:

- Tighter bounds reduce search space and improve runtime stability.
- Keep bounds physically feasible for installation and coverage.

## Global execution and reproducibility fields

- fixed_z: AP height used by all evaluated candidates. Typical indoor ceiling-mounted AP values are around 2.8 to 4.0 meters.
- num_pool_workers: Number of parallel workers. Increase until CPU or GPU resources saturate.
- gpu_fraction: GPU fraction per worker. Lower this when you see GPU OOM.
- random_seed: Reproducibility seed. Keep fixed for direct method comparisons.
- num_aps: AP count. Higher values improve potential coverage but increase optimization complexity.
- min_ap_separation: Minimum spacing between APs. Increase to prevent clustering.
- optimize_orientation: If true, AP directions are optimized. Disable only for quick ablations or fixed orientation assumptions.
- reflector_enabled: Top-level compatibility flag. Keep consistent with scene_config.reflector_enabled.
- focal_z: Reflector target height. Tune to user-device elevation when reflector mode is used.

## demand_config

- demand_config.enabled: Enables weighted spatial priorities.
- demand_config.bounding_boxes: Priority regions defined by top-left and bottom-right XY corners.
- demand_config.box_weights: Relative importance of each region.
- demand_config.apply_blur: Smooths region transitions in the demand map.

Tuning guidance:

- Use a small number of high-confidence priority regions first.
- Increase a box weight only when that region is truly business-critical.
- Set apply_blur true when optimization is unstable around sharp region boundaries.

## objective_params

- objective_params.alpha: Primary objective weight. Increase to prioritize main quality objective.
- objective_params.beta: Penalty or secondary weight. Increase to enforce constraints or auxiliary goals.
- objective_params.softmin_temperature: Softmin smoothness. Lower is more min-like, higher is smoother.
- objective_params.softmin_floor_dbm: Lower clamp in softmin normalization.
- objective_params.softmin_ceil_dbm: Upper clamp in softmin normalization.
- objective_params.coverage_threshold_dbm: Coverage threshold in dBm.
- objective_params.coverage_temperature: Smoothness around coverage threshold transition.

Tuning guidance:

- Start from realistic coverage_threshold_dbm from your service objective.
- Reduce softmin_temperature if you want stronger worst-case focus.
- Increase coverage_temperature if optimization is too brittle near threshold.

## ga_params

- ga_params.pop_size: Population size per generation.
- ga_params.n_gen: Number of generations.
- ga_params.cxpb: Crossover probability.
- ga_params.mutpb: Mutation probability.
- ga_params.tournsize: Tournament selection pressure.
- ga_params.hof_size: Number of elites retained.

Tuning guidance:

- Increase pop_size for broader exploration.
- Increase n_gen for deeper search when improvement has not saturated.
- Increase mutpb if search stagnates early.
- Lower tournsize if diversity collapses too quickly.

## ga_evaluation_params

- ga_evaluation_params.samples_per_tx: Evaluation fidelity per TX.
- ga_evaluation_params.max_depth: Maximum propagation depth.
- ga_evaluation_params.verbose: Evaluator logging verbosity.

Tuning guidance:

- For smoke tests, reduce samples_per_tx and max_depth.
- For final reports, increase fidelity and keep values consistent across methods.

## gd_optimization_params

- gd_optimization_params.num_iterations: GD refinement steps.
- gd_optimization_params.learning_rate: GD step size.
- gd_optimization_params.samples_per_tx: GD evaluator fidelity.
- gd_optimization_params.max_depth: GD propagation depth.
- gd_optimization_params.verbose: GD logging verbosity.

Tuning guidance:

- Increase num_iterations if local refinement is clearly underfitting.
- Lower learning_rate when loss oscillates or diverges.
- Keep GD fidelity aligned with GA fidelity for fair comparison.

## Baseline-specific sections

random_params:

- random_params.num_samples: Number of random candidates. Increase to strengthen the random baseline.

kmeans_params:

- kmeans_params.grid_size: Grid density used when synthetic floorplan points are generated. Larger values improve spatial approximation but increase preprocessing cost.

random_gd_params:

- random_gd_params.num_samples: Number of random start points before GD exploitation.

pso_params:

- pso_params.swarm_size: Number of particles.
- pso_params.num_iterations: PSO macro-iterations.
- pso_params.w: Inertia weight.
- pso_params.c1: Cognitive coefficient.
- pso_params.c2: Social coefficient.

PSO tuning guidance:

- Increase swarm_size for exploration if runtime allows.
- Increase num_iterations when PSO remains far from GD-ready candidates.
- Raise w for exploration-heavy behavior.
- Raise c2 when coordinated convergence is too slow.

## Seed extraction and diversity

- k_seeds: Number of GA-derived seeds sent to GD stage.
- d_corr: Diversity threshold for seed selection.

Tuning guidance:

- Increase k_seeds for robustness when compute budget allows.
- Increase d_corr to force more diverse seeds.
- Set d_corr to 0.0 when pure top-fitness selection is preferred.

## camera

- camera.position: Viewer position for visual outputs.
- camera.look_at: Viewer target point.

These fields do not change optimization results.

## coverage_plot_settings

- coverage_plot_settings.samples_per_tx: Plot-generation fidelity.
- coverage_plot_settings.max_depth: Plot-generation propagation depth.
- coverage_plot_settings.resolution: Output map resolution [width, height].
- coverage_plot_settings.render_ga_generation_best_coverage_maps: Save GA trajectory maps.
- coverage_plot_settings.render_gd_trajectory_coverage_maps: Save GD trajectory maps.

Tuning guidance:

- Keep these low during experimentation.
- Raise fidelity and resolution only for final analysis artifacts.

## Output and reporting controls

- output_dir: Base destination for experiment outputs.
- verbose: Verbose logs from runner components.

## iteration_equalization

- iteration_equalization.enabled: Enables equal total iteration budgets for non-kmeans methods.
- iteration_equalization.target_iterations: Optional fixed target. If omitted and enabled, runner derives target from configured methods.

Tuning guidance:

- Enable this for fair method-comparison reports.
- Use explicit target_iterations for reproducible benchmarking protocols.

## Suggested presets

Use these as top-level override blocks in your config file.

Smoke preset (fast sanity checks)

```json
{
  "num_pool_workers": 1,
  "verbose": false,
  "ga_params": {
    "pop_size": 24,
    "n_gen": 4,
    "cxpb": 0.6,
    "mutpb": 0.4,
    "tournsize": 12,
    "hof_size": 8
  },
  "ga_evaluation_params": {
    "samples_per_tx": 5000,
    "max_depth": 3,
    "verbose": false
  },
  "gd_optimization_params": {
    "num_iterations": 8,
    "learning_rate": 0.01,
    "samples_per_tx": 5000,
    "max_depth": 3,
    "verbose": false
  },
  "random_params": {
    "num_samples": 20
  },
  "random_gd_params": {
    "num_samples": 3
  },
  "pso_params": {
    "swarm_size": 20,
    "num_iterations": 6,
    "w": 0.6,
    "c1": 1.5,
    "c2": 1.5
  },
  "coverage_plot_settings": {
    "samples_per_tx": 20000,
    "max_depth": 5,
    "resolution": [800, 600],
    "render_ga_generation_best_coverage_maps": false,
    "render_gd_trajectory_coverage_maps": false
  },
  "iteration_equalization": {
    "enabled": false
  }
}
```

Balanced preset (day-to-day experiments)

```json
{
  "num_pool_workers": 1,
  "verbose": true,
  "ga_params": {
    "pop_size": 80,
    "n_gen": 20,
    "cxpb": 0.6,
    "mutpb": 0.4,
    "tournsize": 40,
    "hof_size": 20
  },
  "ga_evaluation_params": {
    "samples_per_tx": 200000,
    "max_depth": 10,
    "verbose": false
  },
  "gd_optimization_params": {
    "num_iterations": 60,
    "learning_rate": 0.005,
    "samples_per_tx": 200000,
    "max_depth": 10,
    "verbose": false
  },
  "random_params": {
    "num_samples": 100
  },
  "random_gd_params": {
    "num_samples": 6
  },
  "pso_params": {
    "swarm_size": 80,
    "num_iterations": 20,
    "w": 0.6,
    "c1": 1.6,
    "c2": 1.6
  },
  "coverage_plot_settings": {
    "samples_per_tx": 300000,
    "max_depth": 11,
    "resolution": [1200, 900],
    "render_ga_generation_best_coverage_maps": false,
    "render_gd_trajectory_coverage_maps": false
  },
  "iteration_equalization": {
    "enabled": true,
    "target_iterations": 100
  }
}
```

Final-quality preset (reporting and publication)

```json
{
  "num_pool_workers": 1,
  "verbose": true,
  "ga_params": {
    "pop_size": 180,
    "n_gen": 50,
    "cxpb": 0.6,
    "mutpb": 0.4,
    "tournsize": 60,
    "hof_size": 30
  },
  "ga_evaluation_params": {
    "samples_per_tx": 1000000,
    "max_depth": 13,
    "verbose": false
  },
  "gd_optimization_params": {
    "num_iterations": 140,
    "learning_rate": 0.003,
    "samples_per_tx": 1000000,
    "max_depth": 13,
    "verbose": false
  },
  "random_params": {
    "num_samples": 250
  },
  "random_gd_params": {
    "num_samples": 12
  },
  "pso_params": {
    "swarm_size": 160,
    "num_iterations": 50,
    "w": 0.6,
    "c1": 1.6,
    "c2": 1.6
  },
  "coverage_plot_settings": {
    "samples_per_tx": 1000000,
    "max_depth": 13,
    "resolution": [1600, 1200],
    "render_ga_generation_best_coverage_maps": false,
    "render_gd_trajectory_coverage_maps": false
  },
  "iteration_equalization": {
    "enabled": true,
    "target_iterations": 250
  }
}
```
