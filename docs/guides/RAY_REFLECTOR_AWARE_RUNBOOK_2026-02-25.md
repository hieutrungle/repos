# Ray Reflector-Aware Runbook (2026-02-25)

## Recommended command patterns

### 1) Standard run

```bash
/home/hieule/research/reflector-position/.venv/bin/python examples/ray_parallel_example.py
```

### 2) Timeout-guarded run (recommended during tuning)

```bash
timeout 300s /home/hieule/research/reflector-position/.venv/bin/python -u examples/ray_parallel_example.py
```

### 3) Programmatic call with explicit controls

```python
from examples.ray_parallel_example import run_reflector_aware_grid_search_only

res = run_reflector_aware_grid_search_only(
    output_dir="results/ray_parallel_verify",
    num_pool_workers=2,
    gpu_fraction=0.5,
    grid_resolution=5.0,
    num_rounds=1,
    outer_rounds=1,
    target_quantile=0.05,
    min_ap_separation=10.0,
)
```

## What is now reported

- Best task, AP positions, AP directions
- Reflector position and focal point
- Best min RSS (dBm)
- Best 5th percentile (dBm)
- Min RSS stats: mean/std/range
- 5th percentile stats: mean/std/range
- Worker utilization and timing/speedup

## Output artifacts

- JSON summary:
  - `results/.../gs_2ap_reflector_results.json`
- Plot image:
  - `results/.../gs_2ap_reflector_parallel_results.png`

## Plot legend updates

In **Final Positions** panel:

- AP markers remain per-AP symbol
- Reflector shown as **magenta `X`**
- Focal point shown as **orange `P`**
- Reflector→Focal shown as **magenta dashed line**

## AP overlap guard

- Alternating AP task generation now enforces `min_ap_separation`.
- If no valid task survives filtering, a `ValueError` is raised with guidance.

## Troubleshooting checklist

1. Use timeout (`timeout 180s` or `300s`) to avoid indefinite blocking in terminal sessions.
2. Keep `verbose=True` paths enabled to see ActorPool progress.
3. If runs are too slow, temporarily reduce:
   - `grid_resolution` (larger step size),
   - `num_rounds`,
   - `outer_rounds`.
4. If AP candidates vanish, reduce `min_ap_separation`.
5. For memory diagnostics, use `scripts/monitor_ray_run.sh`.
