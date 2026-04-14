"""PSO+GD specific plotting helpers.

This module renders baseline-specific trajectory plots for PSO followed by
gradient-descent refinement. It can also render optional per-step coverage maps
when enabled via ``coverage_plot_settings`` in the run config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except Exception:  # pragma: no cover - plotting optional at runtime
    plt = None
    MaxNLocator = None  # type: ignore[assignment]


_PRIORITY_METRIC_KEYS: Tuple[str, ...] = (
    "priority_mean_rss_dbm",
    "priority_min_rss_dbm",
    "priority_p5_rss_dbm",
)

_ALL_REGION_METRIC_KEYS: Tuple[str, ...] = (
    "mean_rss_dbm",
    "min_rss_dbm",
    "p5_rss_dbm",
)

_PRIMARY_LOSS_KEYS: Tuple[str, ...] = (
    "primary_loss",
    "min_primary_loss",
    "global_best_primary_loss",
    "swarm_best_primary_loss",
    "running_best_primary_loss",
)

_DEFAULT_CAMERA_POSITION: Tuple[float, float, float] = (20.0, 20.0, 70.0)
_DEFAULT_CAMERA_LOOK_AT: Tuple[float, float, float] = (20.0, 20.1, 1.5)
_DEFAULT_COVERAGE_RESOLUTION: Tuple[int, int] = (1200, 900)


def _as_float(value: Any) -> float:
    """Convert one value to float, returning NaN on invalid inputs."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_optional_int(value: Any, default: int) -> int:
    """Convert one value to int with a safe default fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_optional_positive_int(value: Any) -> Optional[int]:
    """Convert one optional value to positive int when possible."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _set_integer_x_ticks(axis: Any) -> None:
    """Force integer ticks for optimization-step axes."""
    if MaxNLocator is None:
        return
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))


def _extract_trace_rows(result_payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Extract normalized iteration rows from one PSO+GD result payload."""
    raw_rows = result_payload.get("iteration_trace")
    if not isinstance(raw_rows, Sequence):
        return []

    rows: List[Mapping[str, Any]] = []
    for row in raw_rows:
        if isinstance(row, Mapping):
            rows.append(row)
    return rows


def _extract_series(trace_rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    """Extract one numeric series from trace rows as a NumPy array."""
    return np.asarray([_as_float(row.get(key)) for row in trace_rows], dtype=np.float64)


def _extract_primary_loss_series(trace_rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Extract primary-loss series using ordered key fallbacks."""
    values: List[float] = []
    for row in trace_rows:
        selected = float("nan")
        for key in _PRIMARY_LOSS_KEYS:
            candidate = _as_float(row.get(key))
            if np.isfinite(candidate):
                selected = candidate
                break
        values.append(selected)
    return np.asarray(values, dtype=np.float64)


def _extract_best_iteration(primary_loss: np.ndarray) -> Optional[int]:
    """Return one zero-based best iteration index for finite loss series."""
    if primary_loss.size == 0:
        return None

    finite_mask = np.isfinite(primary_loss)
    if not np.any(finite_mask):
        return None

    finite_indices = np.where(finite_mask)[0]
    finite_values = primary_loss[finite_mask]
    best_local_index = int(np.argmin(finite_values))
    return int(finite_indices[best_local_index])


def _extract_priority_map(result_payload: Mapping[str, Any]) -> Optional[np.ndarray]:
    """Extract spatial priority-map array from one PSO+GD result payload."""
    raw_map = result_payload.get("spatial_weights")
    if raw_map is None:
        return None

    try:
        array = np.asarray(raw_map, dtype=np.float64)
    except Exception:
        return None

    if array.ndim == 2 and array.size > 0:
        return array
    return None


def _extract_position_extent(result_payload: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Extract heatmap extent from payload position_bounds when available."""
    raw_bounds = result_payload.get("position_bounds")
    if not isinstance(raw_bounds, Mapping):
        return None

    required_keys = ("x_min", "x_max", "y_min", "y_max")
    if not all(key in raw_bounds for key in required_keys):
        return None

    x_min = _as_float(raw_bounds.get("x_min"))
    x_max = _as_float(raw_bounds.get("x_max"))
    y_min = _as_float(raw_bounds.get("y_min"))
    y_max = _as_float(raw_bounds.get("y_max"))
    if not all(np.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
        return None

    if x_min >= x_max or y_min >= y_max:
        return None

    return float(x_min), float(x_max), float(y_min), float(y_max)


def _coerce_xyz_triplet(
    raw_value: Any,
    default_value: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Coerce an XYZ triplet to floats with fallback to default."""
    if (
        not isinstance(raw_value, Sequence)
        or isinstance(raw_value, (str, bytes))
        or len(raw_value) < 3
    ):
        return default_value

    try:
        return (float(raw_value[0]), float(raw_value[1]), float(raw_value[2]))
    except (TypeError, ValueError):
        return default_value


def _resolve_render_scene_config(
    config_args: Mapping[str, Any],
    result_payload: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Resolve scene config used for optional PSO+GD coverage rendering."""
    raw_scene = config_args.get("scene_config")
    if not isinstance(raw_scene, Mapping):
        return None

    scene_config = dict(raw_scene)
    visualization_scene = config_args.get("visualization_scene_config")
    if isinstance(visualization_scene, Mapping):
        scene_path = visualization_scene.get("scene_path")
        if isinstance(scene_path, str) and scene_path:
            scene_config["scene_path"] = scene_path

    visualization_scene_path = config_args.get("visualization_scene_path")
    if isinstance(visualization_scene_path, str) and visualization_scene_path:
        scene_config["scene_path"] = visualization_scene_path

    if not isinstance(scene_config.get("scene_path"), str) or not str(scene_config.get("scene_path")).strip():
        return None

    if "num_aps" not in scene_config:
        num_aps = _as_optional_int(result_payload.get("num_aps"), default=0)
        if num_aps > 0:
            scene_config["num_aps"] = int(num_aps)

    position_bounds = result_payload.get("position_bounds")
    if isinstance(position_bounds, Mapping) and "position_bounds" not in scene_config:
        scene_config["position_bounds"] = dict(position_bounds)

    return scene_config


def _resolve_render_camera(
    config_args: Mapping[str, Any],
    render_settings: Mapping[str, Any],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Resolve camera position and look-at for coverage rendering."""
    merged_camera: Dict[str, Any] = {}

    raw_camera = config_args.get("camera")
    if isinstance(raw_camera, Mapping):
        merged_camera.update(dict(raw_camera))

    if "camera_position" in config_args:
        merged_camera["position"] = config_args.get("camera_position")
    if "camera_look_at" in config_args:
        merged_camera["look_at"] = config_args.get("camera_look_at")
    if "camera_position" in render_settings:
        merged_camera["position"] = render_settings.get("camera_position")
    if "camera_look_at" in render_settings:
        merged_camera["look_at"] = render_settings.get("camera_look_at")

    camera_position = _coerce_xyz_triplet(
        merged_camera.get("position"),
        _DEFAULT_CAMERA_POSITION,
    )
    camera_look_at = _coerce_xyz_triplet(
        merged_camera.get("look_at"),
        _DEFAULT_CAMERA_LOOK_AT,
    )
    return camera_position, camera_look_at


def _coerce_snapshot_positions(
    raw_positions: Any,
    fallback_z: float,
) -> Optional[List[List[float]]]:
    """Coerce snapshot positions into ``[[x, y, z], ...]`` form."""
    if (
        not isinstance(raw_positions, Sequence)
        or isinstance(raw_positions, (str, bytes))
        or len(raw_positions) == 0
    ):
        return None

    positions: List[List[float]] = []
    for raw_position in raw_positions:
        if (
            not isinstance(raw_position, Sequence)
            or isinstance(raw_position, (str, bytes))
            or len(raw_position) < 2
        ):
            return None

        try:
            x_coord = float(raw_position[0])
            y_coord = float(raw_position[1])
            z_coord = float(raw_position[2]) if len(raw_position) >= 3 else float(fallback_z)
        except (TypeError, ValueError):
            return None

        positions.append([x_coord, y_coord, z_coord])

    return positions


def _coerce_snapshot_directions(raw_directions: Any) -> Optional[List[List[float]]]:
    """Coerce snapshot directions into ``[[dx, dy, dz], ...]`` form."""
    if (
        not isinstance(raw_directions, Sequence)
        or isinstance(raw_directions, (str, bytes))
        or len(raw_directions) == 0
    ):
        return None

    directions: List[List[float]] = []
    for raw_direction in raw_directions:
        if (
            not isinstance(raw_direction, Sequence)
            or isinstance(raw_direction, (str, bytes))
            or len(raw_direction) < 3
        ):
            return None

        try:
            dx = float(raw_direction[0])
            dy = float(raw_direction[1])
            dz = float(raw_direction[2])
        except (TypeError, ValueError):
            return None

        directions.append([dx, dy, dz])

    return directions


def _coerce_reflector_snapshot(raw_reflector: Any) -> Optional[Dict[str, Any]]:
    """Coerce optional reflector snapshot payload from one frame."""
    if not isinstance(raw_reflector, Mapping):
        return None

    reflector: Dict[str, Any] = {}
    u_value = _as_float(raw_reflector.get("u"))
    v_value = _as_float(raw_reflector.get("v"))
    if np.isfinite(u_value):
        reflector["u"] = float(u_value)
    if np.isfinite(v_value):
        reflector["v"] = float(v_value)

    raw_target = raw_reflector.get("target")
    if (
        isinstance(raw_target, Sequence)
        and not isinstance(raw_target, (str, bytes))
        and len(raw_target) >= 3
    ):
        try:
            reflector["target"] = [
                float(raw_target[0]),
                float(raw_target[1]),
                float(raw_target[2]),
            ]
        except (TypeError, ValueError):
            pass

    raw_position = raw_reflector.get("position")
    if (
        isinstance(raw_position, Sequence)
        and not isinstance(raw_position, (str, bytes))
        and len(raw_position) >= 3
    ):
        try:
            reflector["position"] = [
                float(raw_position[0]),
                float(raw_position[1]),
                float(raw_position[2]),
            ]
        except (TypeError, ValueError):
            pass

    return reflector if reflector else None


def _extract_coverage_snapshots(
    result_payload: Mapping[str, Any],
    fallback_z: float,
) -> List[Dict[str, Any]]:
    """Extract normalized PSO+GD per-step snapshots from result payload."""
    raw_snapshots = result_payload.get("coverage_snapshots")
    if not isinstance(raw_snapshots, Sequence):
        return []

    snapshots: List[Dict[str, Any]] = []
    for index, raw_snapshot in enumerate(raw_snapshots, start=1):
        if not isinstance(raw_snapshot, Mapping):
            continue

        positions = _coerce_snapshot_positions(
            raw_snapshot.get("positions"),
            fallback_z=float(fallback_z),
        )
        if positions is None:
            continue

        snapshot: Dict[str, Any] = {
            "iteration": int(_as_optional_int(raw_snapshot.get("iteration"), default=index)),
            "phase": str(raw_snapshot.get("phase", "step")),
            "phase_iteration": int(_as_optional_int(raw_snapshot.get("phase_iteration"), default=index)),
            "positions": positions,
        }

        directions = _coerce_snapshot_directions(raw_snapshot.get("directions"))
        if directions is not None:
            snapshot["directions"] = directions

        reflector_snapshot = _coerce_reflector_snapshot(raw_snapshot.get("reflector"))
        if reflector_snapshot is not None:
            snapshot["reflector"] = reflector_snapshot

        snapshots.append(snapshot)

    return snapshots


def _sanitize_phase_token(raw_phase: str) -> str:
    """Convert arbitrary phase text into a filesystem-safe token."""
    token = "".join(
        char if (char.isalnum() or char in ("_", "-")) else "_"
        for char in str(raw_phase).strip().lower()
    )
    return token or "step"


def _render_coverage_snapshot(
    scene_config: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    save_path: Path,
    samples_per_tx: int,
    max_depth: int,
    resolution: Tuple[int, int],
    camera_position: Tuple[float, float, float],
    camera_look_at: Tuple[float, float, float],
) -> Optional[str]:
    """Render one PSO+GD snapshot coverage map to disk."""
    from sionna.rt import RadioMapSolver

    from reflector_position.scene_setup import create_camera, setup_building_floor_scene

    snapshot_positions = snapshot.get("positions")
    if (
        not isinstance(snapshot_positions, Sequence)
        or isinstance(snapshot_positions, (str, bytes))
        or len(snapshot_positions) == 0
    ):
        return None

    effective_num_aps = scene_config.get("num_aps")
    if effective_num_aps is None:
        effective_num_aps = int(len(snapshot_positions))

    raw_reflector_size = scene_config.get("reflector_size", (2.0, 2.0))
    reflector_size: Tuple[float, float] = (2.0, 2.0)
    if (
        isinstance(raw_reflector_size, Sequence)
        and not isinstance(raw_reflector_size, (str, bytes))
        and len(raw_reflector_size) >= 2
    ):
        try:
            reflector_size = (
                float(raw_reflector_size[0]),
                float(raw_reflector_size[1]),
            )
        except (TypeError, ValueError):
            reflector_size = (2.0, 2.0)

    loaded = setup_building_floor_scene(
        scene_path=str(scene_config["scene_path"]),
        frequency=scene_config.get("frequency", 6e9),
        tx_positions=scene_config.get("tx_positions", None),
        num_aps=effective_num_aps,
        position_bounds=scene_config.get("position_bounds", None),
        tx_power_dbm=scene_config.get("tx_power_dbm", 5.0),
        rx_position=scene_config.get("rx_position", (16.0, 16.5, 1.5)),
        reflector_enabled=scene_config.get("reflector_enabled", False),
        reflector_size=reflector_size,
        wall_top_left=scene_config.get("wall_top_left", None),
        wall_bottom_right=scene_config.get("wall_bottom_right", None),
        focal_point=scene_config.get("focal_point", None),
        device=scene_config.get("device", "cuda"),
    )
    reflector_controller = None
    if isinstance(loaded, tuple) and len(loaded) == 2:
        scene = loaded[0]
        reflector_controller = loaded[1]
    else:
        scene = loaded

    snapshot_directions = snapshot.get("directions")
    transmitters = list(scene.transmitters.values())
    for tx_index, raw_position in enumerate(snapshot_positions[: len(transmitters)]):
        if not isinstance(raw_position, Sequence) or len(raw_position) < 3:
            continue

        position = [
            float(raw_position[0]),
            float(raw_position[1]),
            float(raw_position[2]),
        ]
        transmitters[tx_index].position = position

        if (
            isinstance(snapshot_directions, Sequence)
            and tx_index < len(snapshot_directions)
            and isinstance(snapshot_directions[tx_index], Sequence)
            and len(snapshot_directions[tx_index]) >= 3
        ):
            direction = snapshot_directions[tx_index]
            target = [
                position[0] + float(direction[0]),
                position[1] + float(direction[1]),
                position[2] + float(direction[2]),
            ]
            transmitters[tx_index].look_at(target)

    if reflector_controller is not None:
        reflector_snapshot = snapshot.get("reflector")
        if not isinstance(reflector_snapshot, Mapping):
            reflector_snapshot = {}

        target_raw = reflector_snapshot.get("target")
        if (
            not isinstance(target_raw, Sequence)
            or isinstance(target_raw, (str, bytes))
            or len(target_raw) < 3
        ):
            target_raw = scene_config.get("focal_point")

        if (
            isinstance(target_raw, Sequence)
            and not isinstance(target_raw, (str, bytes))
            and len(target_raw) >= 3
            and len(snapshot_positions) > 0
            and isinstance(snapshot_positions[0], Sequence)
            and len(snapshot_positions[0]) >= 3
        ):
            import torch

            u_value = _as_float(reflector_snapshot.get("u"))
            v_value = _as_float(reflector_snapshot.get("v"))
            if not np.isfinite(u_value):
                u_value = 0.5
            if not np.isfinite(v_value):
                v_value = 0.5

            reflector_controller.u = torch.tensor(
                float(u_value),
                dtype=torch.float32,
                device=reflector_controller.device,
            )
            reflector_controller.v = torch.tensor(
                float(v_value),
                dtype=torch.float32,
                device=reflector_controller.device,
            )
            reflector_controller.set_tx_position(
                np.asarray(snapshot_positions[0], dtype=np.float32)
            )
            reflector_controller.set_focal_point(
                torch.tensor(
                    [
                        float(target_raw[0]),
                        float(target_raw[1]),
                        float(target_raw[2]),
                    ],
                    dtype=torch.float32,
                    device=reflector_controller.device,
                ),
                requires_grad=False,
            )
            reflector_controller.orient_to_target()
            reflector_controller.apply_to_scene()

    solver = RadioMapSolver()
    radio_map = solver(
        scene,
        cell_size=cast(Any, (1.0, 1.0)),
        samples_per_tx=int(samples_per_tx),
        max_depth=int(max_depth),
        refraction=True,
        diffraction=True,
    )

    camera = create_camera(
        position=camera_position,
        look_at=camera_look_at,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render_to_file(
        camera=camera,
        filename=str(save_path),
        radio_map=radio_map,
        rm_metric="rss",
        rm_db_scale=True,
        rm_vmin=-80,
        rm_vmax=-40,
        resolution=resolution,
        show_devices=True,
        show_orientations=False,
    )
    return str(save_path)


def _save_pso_gd_step_coverage_maps(
    result_payload: Mapping[str, Any],
    plots_dir: Path,
    config_args: Mapping[str, Any],
) -> Dict[str, Any]:
    """Render optional PSO+GD per-step coverage maps based on config flags."""
    render_settings = config_args.get("coverage_plot_settings")
    if not isinstance(render_settings, Mapping):
        render_settings = {}

    if not bool(render_settings.get("render_pso_gd_step_coverage_maps", False)):
        return {}

    scene_config = _resolve_render_scene_config(
        config_args=config_args,
        result_payload=result_payload,
    )
    if scene_config is None:
        return {
            "pso_gd_step_coverage_error": (
                "coverage_plot_settings.render_pso_gd_step_coverage_maps is true "
                "but scene_config/visualization scene path is missing"
            )
        }

    samples_per_tx = _as_optional_int(render_settings.get("samples_per_tx"), default=1_000_000)
    max_depth = _as_optional_int(render_settings.get("max_depth"), default=13)

    raw_resolution = render_settings.get("resolution", _DEFAULT_COVERAGE_RESOLUTION)
    if isinstance(raw_resolution, Sequence) and not isinstance(raw_resolution, (str, bytes)) and len(raw_resolution) >= 2:
        resolution = (
            max(1, int(raw_resolution[0])),
            max(1, int(raw_resolution[1])),
        )
    else:
        resolution = _DEFAULT_COVERAGE_RESOLUTION

    frame_stride = max(
        1,
        _as_optional_int(
            render_settings.get("pso_gd_step_coverage_frame_stride"),
            default=1,
        ),
    )
    max_frames = _as_optional_positive_int(
        render_settings.get("pso_gd_step_coverage_max_frames")
    )

    camera_position, camera_look_at = _resolve_render_camera(
        config_args=config_args,
        render_settings=render_settings,
    )

    fallback_z = float(config_args.get("fixed_z", 3.8))
    snapshots = _extract_coverage_snapshots(
        result_payload=result_payload,
        fallback_z=fallback_z,
    )
    if not snapshots:
        return {
            "pso_gd_step_coverage_error": (
                "No per-step coverage snapshots found in PSO+GD result payload"
            )
        }

    selected_snapshots = [
        snapshot
        for index, snapshot in enumerate(snapshots)
        if index % frame_stride == 0
    ]
    if max_frames is not None:
        selected_snapshots = selected_snapshots[:max_frames]

    base_dir = plots_dir / "pso_gd_step_coverage"
    frames: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}

    for frame_index, snapshot in enumerate(selected_snapshots, start=1):
        iteration = _as_optional_int(snapshot.get("iteration"), default=frame_index)
        phase = str(snapshot.get("phase", "step"))
        phase_iteration = _as_optional_int(snapshot.get("phase_iteration"), default=frame_index)
        phase_token = _sanitize_phase_token(phase)

        image_path = base_dir / f"step_{iteration:04d}_{phase_token}_{phase_iteration:04d}.png"
        try:
            rendered = _render_coverage_snapshot(
                scene_config=scene_config,
                snapshot=snapshot,
                save_path=image_path,
                samples_per_tx=samples_per_tx,
                max_depth=max_depth,
                resolution=resolution,
                camera_position=camera_position,
                camera_look_at=camera_look_at,
            )
            if rendered is not None:
                frames.append(
                    {
                        "iteration": int(iteration),
                        "phase": phase,
                        "phase_iteration": int(phase_iteration),
                        "image_png": rendered,
                    }
                )
        except Exception as exc:
            errors[str(image_path.name)] = f"{type(exc).__name__}: {exc}"

    artifacts: Dict[str, Any] = {
        "pso_gd_step_coverage_dir": str(base_dir),
        "pso_gd_step_coverage_image_count": int(len(frames)),
        "pso_gd_step_coverage_frames": frames,
        "pso_gd_step_coverage_enabled": True,
    }
    if errors:
        artifacts["pso_gd_step_coverage_errors"] = errors
    return artifacts


def _plot_metrics_panel(
    axis: Any,
    steps: np.ndarray,
    trace_rows: Sequence[Mapping[str, Any]],
    metric_keys: Sequence[str],
    title: str,
) -> None:
    """Render one metric-evolution panel for a list of metric keys."""
    color_cycle = ("tab:blue", "tab:purple", "tab:cyan")
    plotted = False

    for index, metric_key in enumerate(metric_keys):
        series = _extract_series(trace_rows, metric_key)
        finite_mask = np.isfinite(series)
        if not np.any(finite_mask):
            continue

        axis.plot(
            steps[finite_mask],
            series[finite_mask],
            linewidth=2.0,
            color=color_cycle[index % len(color_cycle)],
            label=metric_key,
        )
        plotted = True

    axis.set_xlabel("Optimization Step")
    axis.set_ylabel("RSSI (dBm)")
    axis.set_title(title)
    _set_integer_x_ticks(axis)
    axis.grid(True, alpha=0.3)
    if plotted:
        axis.legend(loc="best", fontsize=9)


def save_pso_gd_trajectory_plot(
    result_payload: Mapping[str, Any],
    save_path: Path,
) -> Optional[str]:
    """Save one PSO+GD trajectory plot with focused 4-panel layout."""
    if plt is None:
        return None

    trace_rows = _extract_trace_rows(result_payload)
    if not trace_rows:
        return None

    steps = np.arange(1, len(trace_rows) + 1, dtype=np.int64)
    primary_loss = _extract_primary_loss_series(trace_rows)
    running_best = _extract_series(trace_rows, "running_best_primary_loss")
    best_iteration = _extract_best_iteration(primary_loss)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    primary_axis = axes[0, 0]
    finite_primary = np.isfinite(primary_loss)
    if np.any(finite_primary):
        primary_axis.plot(
            steps[finite_primary],
            primary_loss[finite_primary],
            color="tab:blue",
            linewidth=2.0,
            label="primary_loss",
        )

    finite_running = np.isfinite(running_best)
    if np.any(finite_running):
        primary_axis.plot(
            steps[finite_running],
            running_best[finite_running],
            color="tab:orange",
            linewidth=1.7,
            linestyle="--",
            label="running_best_primary_loss",
        )

    if best_iteration is not None:
        best_x = int(best_iteration + 1)
        primary_axis.axvline(
            best_x,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Best iter {best_x}",
        )

    primary_axis.set_xlabel("Optimization Step")
    primary_axis.set_ylabel("Primary Loss")
    primary_axis.set_title("Primary Loss Evolution")
    _set_integer_x_ticks(primary_axis)
    primary_axis.grid(True, alpha=0.3)
    if primary_axis.lines:
        primary_axis.legend(loc="best", fontsize=9)

    _plot_metrics_panel(
        axis=axes[0, 1],
        steps=steps,
        trace_rows=trace_rows,
        metric_keys=_PRIORITY_METRIC_KEYS,
        title="Priority Metrics Evolution",
    )

    _plot_metrics_panel(
        axis=axes[1, 0],
        steps=steps,
        trace_rows=trace_rows,
        metric_keys=_ALL_REGION_METRIC_KEYS,
        title="Metrics Evolution (All-region)",
    )

    heatmap_axis = axes[1, 1]
    priority_map = _extract_priority_map(result_payload)
    if priority_map is None:
        heatmap_axis.axis("off")
        heatmap_axis.text(
            0.02,
            0.95,
            "Priority map unavailable in PSO+GD payload.",
            transform=heatmap_axis.transAxes,
            va="top",
            ha="left",
            fontsize=10,
        )
    else:
        extent = _extract_position_extent(result_payload)
        image = heatmap_axis.imshow(
            priority_map,
            cmap="magma",
            origin="lower",
            extent=extent,
            aspect="equal",
        )
        heatmap_axis.set_title("Priority Map Heatmap")
        heatmap_axis.set_xlabel("X Position (m)")
        heatmap_axis.set_ylabel("Y Position (m)")
        heatmap_axis.grid(False)
        colorbar = figure.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Priority")

    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return str(save_path)


def save_pso_gd_plots(
    result_payload: Mapping[str, Any],
    plots_dir: Path,
    config_args: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Save all PSO+GD-specific plots and return artifact-path mapping."""
    artifacts: Dict[str, Any] = {}

    trajectory_path = save_pso_gd_trajectory_plot(
        result_payload=result_payload,
        save_path=plots_dir / "pso_gd_trajectory.png",
    )
    if trajectory_path is not None:
        artifacts["pso_gd_trajectory_plot_png"] = trajectory_path

    if isinstance(config_args, Mapping):
        coverage_artifacts = _save_pso_gd_step_coverage_maps(
            result_payload=result_payload,
            plots_dir=plots_dir,
            config_args=config_args,
        )
        artifacts.update(coverage_artifacts)

    return artifacts


__all__ = ["save_pso_gd_plots", "save_pso_gd_trajectory_plot"]
