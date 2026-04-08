"""Tests for GA generation-best snapshot payloads and coverage rendering hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest
from matplotlib.ticker import MaxNLocator

from reflector_position.optimizers.memetic.memetic_ga_logic import (
    MemeticGeneticAlgorithmRunner,
)
from reflector_position.optimizers.memetic import memetic_plotting
from reflector_position.optimizers.memetic import memetic_summary
from reflector_position.optimizers.memetic import run_memetic_pipeline


def _fake_executor_map(
    func: Any,
    iterable: Iterable[Any],
) -> list[dict[str, Any]]:
    """Return deterministic evaluator payloads from GA worker-format tasks."""
    outputs: list[dict[str, Any]] = []
    for individual in iterable:
        _, _, task_kwargs, _ = func(individual)
        positions = task_kwargs.get("initial_positions", [])

        # Maximize fitness near the center to keep selection deterministic.
        score = 0.0
        for x, y in positions:
            score -= (float(x) - 20.0) ** 2 + (float(y) - 20.0) ** 2

        outputs.append(
            {
                "primary_fitness": score,
                "loss_components": {"proxy_loss": -score},
                "physical_metrics": {"coverage_pct": 50.0 + 0.001 * score},
            }
        )
    return outputs


def test_generation_details_include_best_snapshot_payloads() -> None:
    """GA generation details should include decoded best-individual snapshots."""
    runner = MemeticGeneticAlgorithmRunner(
        position_bounds={"x_min": 0.0, "x_max": 40.0, "y_min": 0.0, "y_max": 40.0},
        fixed_z=3.8,
        executor_map=_fake_executor_map,
        optimize_orientation=True,
        num_aps=1,
        reflector_enabled=True,
        focal_z=1.5,
    )

    results = runner.run(
        optimization_params={"samples_per_tx": 1, "max_depth": 1},
        ga_params={"pop_size": 8, "n_gen": 2, "hof_size": 4},
        seed=7,
        verbose=False,
        k_seeds=2,
    )

    generation_details = results["generation_details"]
    assert len(generation_details) == 3  # Gen0 + 2 generations

    for row in generation_details:
        assert int(row.get("generation_top_k", 0)) >= 1

        top_individuals = row.get("top_individuals")
        assert isinstance(top_individuals, list)
        assert len(top_individuals) >= 1
        assert int(top_individuals[0].get("rank", -1)) == 1

        assert isinstance(row.get("best_chromosome"), list)

        best_positions = row.get("best_ap_positions")
        assert isinstance(best_positions, list)
        assert len(best_positions) == 1
        assert len(best_positions[0]) == 3

        best_directions = row.get("best_ap_directions")
        assert isinstance(best_directions, list)
        assert len(best_directions) == 1
        assert len(best_directions[0]) == 3

        best_reflector = row.get("best_reflector")
        assert isinstance(best_reflector, Mapping)
        assert {"u", "v", "focal_x", "focal_y", "focal_z"}.issubset(best_reflector.keys())

        assert isinstance(row.get("best_loss_components"), Mapping)
        assert isinstance(row.get("best_physical_metrics"), Mapping)

        if len(top_individuals) > 1:
            assert row.get("second_primary_fitness") is not None
            assert row.get("second_primary_fitness") == top_individuals[1].get("primary_fitness")
            assert isinstance(row.get("second_physical_metrics"), Mapping)


def test_save_memetic_coverage_maps_renders_ga_generation_best_frames(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Coverage plotting should render one image per GA generation-best snapshot."""
    rendered_paths: list[Path] = []
    rendered_scene_paths: list[str] = []
    rendered_camera_positions: list[tuple[float, float, float]] = []
    rendered_camera_look_ats: list[tuple[float, float, float]] = []

    def _fake_render_coverage_snapshot(
        scene_config: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        save_path: Path,
        samples_per_tx: int,
        max_depth: int,
        resolution: tuple[int, int],
        camera_position: tuple[float, float, float],
        camera_look_at: tuple[float, float, float],
    ) -> str:
        del snapshot, samples_per_tx, max_depth, resolution
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text("stub", encoding="utf-8")
        rendered_paths.append(save_path)
        rendered_scene_paths.append(str(scene_config.get("scene_path")))
        rendered_camera_positions.append(tuple(float(v) for v in camera_position))
        rendered_camera_look_ats.append(tuple(float(v) for v in camera_look_at))
        return str(save_path)

    monkeypatch.setattr(memetic_plotting, "_render_coverage_snapshot", _fake_render_coverage_snapshot)

    summary = {
        "ga_results": {
            "generation_details": [
                {
                    "gen": 0,
                    "best_ap_positions": [[10.0, 10.0, 3.8]],
                    "best_ap_directions": [[1.0, 0.0, 0.0]],
                    "best_reflector": {
                        "u": 0.25,
                        "v": 0.75,
                        "focal_x": 20.0,
                        "focal_y": 21.0,
                        "focal_z": 1.5,
                    },
                },
                {
                    "gen": 1,
                    "best_ap_positions": [[12.0, 11.0, 3.8]],
                    "best_ap_directions": [[0.0, 1.0, 0.0]],
                    "best_reflector": {
                        "u": 0.30,
                        "v": 0.70,
                        "focal_x": 22.0,
                        "focal_y": 20.0,
                        "focal_z": 1.5,
                    },
                },
            ]
        },
        "gd_results": {},
    }

    config_args = {
        "scene_config": {
            "scene_path": "dummy_scene.xml",
            "tx_positions": [[15.0, 15.0, 3.8]],
            "reflector_enabled": True,
            "focal_point": [25.0, 25.0, 1.5],
        },
        "visualization_scene_config": {
            "scene_path": "dummy_visual_scene.xml",
        },
        "camera": {
            "position": [20.0, 20.0, 70.0],
            "look_at": [20.0, 20.1, 1.5],
        },
        "coverage_plot_settings": {
            "samples_per_tx": 1,
            "max_depth": 1,
            "resolution": [64, 48],
            "render_ga_generation_best_coverage_maps": True,
            "render_gd_trajectory_coverage_maps": False,
        },
    }

    artifacts = memetic_plotting.save_memetic_coverage_maps(
        summary=summary,
        config_args=config_args,
        output_dir=tmp_path,
    )

    assert artifacts["coverage_map_ga_generation_best_count"] == "2"
    assert "coverage_map_ga_generation_0000" in artifacts
    assert "coverage_map_ga_generation_0001" in artifacts

    rendered_names = {path.name for path in rendered_paths}
    assert "gen_0000.png" in rendered_names
    assert "gen_0001.png" in rendered_names
    assert rendered_scene_paths
    assert set(rendered_scene_paths) == {"dummy_visual_scene.xml"}
    assert set(rendered_camera_positions) == {(20.0, 20.0, 70.0)}
    assert set(rendered_camera_look_ats) == {(20.0, 20.1, 1.5)}


def test_save_ga_generation_best_metric_trend_plots(tmp_path: Path) -> None:
    """GA trend plotting should render all-area vs priority-area metric lines."""
    ga_results = {
        "generation_details": [
            {
                "gen": 0,
                "best_physical_metrics": {
                    "mean_rss_dbm": -82.0,
                    "p5_rss_dbm": -96.0,
                    "min_rss_dbm": -101.0,
                    "priority_mean_rss_dbm": -78.0,
                    "priority_p5_rss_dbm": -92.0,
                    "priority_min_rss_dbm": -99.0,
                },
            },
            {
                "gen": 1,
                "best_physical_metrics": {
                    "mean_rss_dbm": -80.5,
                    "p5_rss_dbm": -94.5,
                    "min_rss_dbm": -100.0,
                    "priority_mean_rss_dbm": -76.5,
                    "priority_p5_rss_dbm": -89.0,
                    "priority_min_rss_dbm": -97.0,
                },
            },
        ]
    }

    artifacts = memetic_plotting.save_ga_generation_best_metric_trend_plots(
        ga_results=ga_results,
        save_dir=tmp_path,
    )

    assert artifacts["ga_best_metric_trend_plot_count"] == "3"
    assert "ga_best_mean_rssi_trend_plot" in artifacts
    assert "ga_best_p5_rssi_trend_plot" in artifacts
    assert "ga_best_min_rssi_trend_plot" in artifacts
    assert Path(artifacts["ga_best_mean_rssi_trend_plot"]).exists()
    assert Path(artifacts["ga_best_p5_rssi_trend_plot"]).exists()
    assert Path(artifacts["ga_best_min_rssi_trend_plot"]).exists()


def test_save_ga_generation_ranked_metric_trend_plots(tmp_path: Path) -> None:
    """Rank-aware GA trend plotting should render one set per selected rank."""
    ga_results = {
        "num_selected_seeds": 2,
        "generation_details": [
            {
                "gen": 0,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -82.0,
                            "p5_rss_dbm": -96.0,
                            "min_rss_dbm": -101.0,
                            "priority_mean_rss_dbm": -78.0,
                            "priority_p5_rss_dbm": -92.0,
                            "priority_min_rss_dbm": -99.0,
                        },
                    },
                    {
                        "rank": 2,
                        "physical_metrics": {
                            "mean_rss_dbm": -83.5,
                            "p5_rss_dbm": -97.5,
                            "min_rss_dbm": -102.5,
                            "priority_mean_rss_dbm": -79.5,
                            "priority_p5_rss_dbm": -93.5,
                            "priority_min_rss_dbm": -100.5,
                        },
                    },
                ],
            },
            {
                "gen": 1,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -80.5,
                            "p5_rss_dbm": -94.5,
                            "min_rss_dbm": -100.0,
                            "priority_mean_rss_dbm": -76.5,
                            "priority_p5_rss_dbm": -89.0,
                            "priority_min_rss_dbm": -97.0,
                        },
                    },
                    {
                        "rank": 2,
                        "physical_metrics": {
                            "mean_rss_dbm": -81.5,
                            "p5_rss_dbm": -95.5,
                            "min_rss_dbm": -101.2,
                            "priority_mean_rss_dbm": -77.2,
                            "priority_p5_rss_dbm": -90.4,
                            "priority_min_rss_dbm": -98.0,
                        },
                    },
                ],
            },
        ],
    }

    artifacts = memetic_plotting.save_ga_generation_best_metric_trend_plots(
        ga_results=ga_results,
        save_dir=tmp_path,
    )

    assert artifacts["ga_best_metric_trend_plot_count"] == "6"
    assert "ga_best_mean_rssi_trend_plot" in artifacts
    assert "ga_rank2_mean_rssi_trend_plot" in artifacts
    assert "ga_rank2_p5_rssi_trend_plot" in artifacts
    assert "ga_rank2_min_rssi_trend_plot" in artifacts
    assert Path(artifacts["ga_rank2_mean_rssi_trend_plot"]).exists()
    assert Path(artifacts["ga_rank2_p5_rssi_trend_plot"]).exists()
    assert Path(artifacts["ga_rank2_min_rssi_trend_plot"]).exists()


def test_save_ga_gd_stitched_metric_trend_plots(tmp_path: Path) -> None:
    """Stitched trend plots should append best GD metrics to GA generation trends."""
    ga_results = {
        "generation_details": [
            {
                "gen": 0,
                "best_physical_metrics": {
                    "mean_rss_dbm": -82.0,
                    "p5_rss_dbm": -96.0,
                    "min_rss_dbm": -101.0,
                    "priority_mean_rss_dbm": -78.0,
                    "priority_p5_rss_dbm": -92.0,
                    "priority_min_rss_dbm": -99.0,
                },
            },
            {
                "gen": 1,
                "best_physical_metrics": {
                    "mean_rss_dbm": -80.5,
                    "p5_rss_dbm": -94.5,
                    "min_rss_dbm": -100.0,
                    "priority_mean_rss_dbm": -76.5,
                    "priority_p5_rss_dbm": -89.0,
                    "priority_min_rss_dbm": -97.0,
                },
            },
        ]
    }
    gd_results = {
        "global_best_result": {
            "results": {
                "physical_metrics": {
                    "mean_rss_dbm": -79.0,
                    "p5_rss_dbm": -92.0,
                    "min_rss_dbm": -98.5,
                    "priority_mean_rss_dbm": -75.5,
                    "priority_p5_rss_dbm": -87.0,
                    "priority_min_rss_dbm": -95.0,
                }
            }
        }
    }

    artifacts = memetic_plotting.save_ga_gd_stitched_metric_trend_plots(
        ga_results=ga_results,
        gd_results=gd_results,
        save_dir=tmp_path,
    )

    assert artifacts["ga_gd_stitched_metric_trend_plot_count"] == "3"
    assert "ga_gd_stitched_mean_rssi_plot" in artifacts
    assert "ga_gd_stitched_p5_rssi_plot" in artifacts
    assert "ga_gd_stitched_min_rssi_plot" in artifacts
    assert Path(artifacts["ga_gd_stitched_mean_rssi_plot"]).exists()
    assert Path(artifacts["ga_gd_stitched_p5_rssi_plot"]).exists()
    assert Path(artifacts["ga_gd_stitched_min_rssi_plot"]).exists()


def test_save_ga_gd_stitched_ranked_metric_trend_plots(tmp_path: Path) -> None:
    """Stitched trends should include per-rank GD endpoints when available."""
    ga_results = {
        "num_selected_seeds": 2,
        "generation_details": [
            {
                "gen": 0,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -82.0,
                            "p5_rss_dbm": -96.0,
                            "min_rss_dbm": -101.0,
                            "priority_mean_rss_dbm": -78.0,
                            "priority_p5_rss_dbm": -92.0,
                            "priority_min_rss_dbm": -99.0,
                        },
                    },
                    {
                        "rank": 2,
                        "physical_metrics": {
                            "mean_rss_dbm": -84.0,
                            "p5_rss_dbm": -98.0,
                            "min_rss_dbm": -103.0,
                            "priority_mean_rss_dbm": -80.0,
                            "priority_p5_rss_dbm": -94.0,
                            "priority_min_rss_dbm": -101.0,
                        },
                    },
                ],
            },
            {
                "gen": 1,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -80.5,
                            "p5_rss_dbm": -94.5,
                            "min_rss_dbm": -100.0,
                            "priority_mean_rss_dbm": -76.5,
                            "priority_p5_rss_dbm": -89.0,
                            "priority_min_rss_dbm": -97.0,
                        },
                    },
                    {
                        "rank": 2,
                        "physical_metrics": {
                            "mean_rss_dbm": -82.1,
                            "p5_rss_dbm": -96.3,
                            "min_rss_dbm": -101.8,
                            "priority_mean_rss_dbm": -78.2,
                            "priority_p5_rss_dbm": -91.3,
                            "priority_min_rss_dbm": -98.8,
                        },
                    },
                ],
            },
        ],
    }

    gd_results = {
        "per_seed_analysis": [
            {
                "seed_index": 0,
                "physical_metrics": {
                    "mean_rss_dbm": -79.0,
                    "p5_rss_dbm": -92.0,
                    "min_rss_dbm": -98.5,
                    "priority_mean_rss_dbm": -75.5,
                    "priority_p5_rss_dbm": -87.0,
                    "priority_min_rss_dbm": -95.0,
                },
            },
            {
                "seed_index": 1,
                "physical_metrics": {
                    "mean_rss_dbm": -80.0,
                    "p5_rss_dbm": -93.0,
                    "min_rss_dbm": -99.5,
                    "priority_mean_rss_dbm": -76.2,
                    "priority_p5_rss_dbm": -88.2,
                    "priority_min_rss_dbm": -96.0,
                },
            },
        ]
    }

    artifacts = memetic_plotting.save_ga_gd_stitched_metric_trend_plots(
        ga_results=ga_results,
        gd_results=gd_results,
        save_dir=tmp_path,
    )

    assert artifacts["ga_gd_stitched_metric_trend_plot_count"] == "6"
    assert "ga_gd_stitched_mean_rssi_plot" in artifacts
    assert "ga_gd_stitched_rank2_mean_rssi_plot" in artifacts
    assert "ga_gd_stitched_rank2_p5_rssi_plot" in artifacts
    assert "ga_gd_stitched_rank2_min_rssi_plot" in artifacts
    assert Path(artifacts["ga_gd_stitched_rank2_mean_rssi_plot"]).exists()
    assert Path(artifacts["ga_gd_stitched_rank2_p5_rssi_plot"]).exists()
    assert Path(artifacts["ga_gd_stitched_rank2_min_rssi_plot"]).exists()


def test_ranked_ga_metric_plot_uses_integer_x_ticks_and_rank_free_legend(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Ranked GA metric plot should keep rank in title only, not in legend."""
    ga_results = {
        "generation_details": [
            {
                "gen": 0,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -82.0,
                            "priority_mean_rss_dbm": -78.0,
                        },
                    },
                    {
                        "rank": 2,
                        "physical_metrics": {
                            "mean_rss_dbm": -84.0,
                            "priority_mean_rss_dbm": -80.0,
                        },
                    },
                ],
            },
            {
                "gen": 1,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -80.5,
                            "priority_mean_rss_dbm": -76.5,
                        },
                    },
                    {
                        "rank": 2,
                        "physical_metrics": {
                            "mean_rss_dbm": -82.2,
                            "priority_mean_rss_dbm": -78.6,
                        },
                    },
                ],
            },
        ]
    }

    captured: dict[str, Any] = {}
    real_subplots = memetic_plotting.plt.subplots

    def _capture_subplots(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(memetic_plotting.plt, "subplots", _capture_subplots)

    output_path = tmp_path / "rank2_ga_plot.png"
    rendered = memetic_plotting.save_ga_generation_best_metric_trend_plot(
        ga_results=ga_results,
        save_path=output_path,
        metric_key="mean_rss_dbm",
        priority_metric_key="priority_mean_rss_dbm",
        metric_label="Mean RSSI (dBm)",
        rank=2,
    )

    assert rendered is not None
    assert output_path.exists()

    ax = captured["ax"]
    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["All-area", "Priority-area"]
    assert all("rank" not in label.lower() for label in labels)
    assert "Mean RSSI (dBm) Trend" in ax.get_title()
    assert isinstance(ax.xaxis.get_major_locator(), MaxNLocator)


def test_ranked_ga_metric_plot_overlays_gd_star_points(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """GA metric trend plot should add GD best markers with star icons."""
    ga_results = {
        "generation_details": [
            {
                "gen": 0,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -82.0,
                            "priority_mean_rss_dbm": -78.0,
                        },
                    }
                ],
            },
            {
                "gen": 1,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -80.5,
                            "priority_mean_rss_dbm": -76.5,
                        },
                    }
                ],
            },
        ]
    }
    gd_results = {
        "global_best_result": {
            "results": {
                "physical_metrics": {
                    "mean_rss_dbm": -79.8,
                    "priority_mean_rss_dbm": -75.9,
                }
            }
        }
    }

    captured: dict[str, Any] = {}
    real_subplots = memetic_plotting.plt.subplots

    def _capture_subplots(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(memetic_plotting.plt, "subplots", _capture_subplots)

    output_path = tmp_path / "rank1_ga_with_gd_plot.png"
    rendered = memetic_plotting.save_ga_generation_best_metric_trend_plot(
        ga_results=ga_results,
        gd_results=gd_results,
        gd_seed_index=0,
        save_path=output_path,
        metric_key="mean_rss_dbm",
        priority_metric_key="priority_mean_rss_dbm",
        metric_label="Mean RSSI (dBm)",
        rank=1,
    )

    assert rendered is not None
    assert output_path.exists()

    ax = captured["ax"]
    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert "GD all-area" in labels
    assert "GD priority-area" in labels
    assert len(ax.collections) == 2


def test_ranked_stitched_plot_uses_integer_x_ticks_and_rank_free_legend(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Ranked stitched GA->GD plot should keep rank in title and omit it from legend."""
    ga_results = {
        "generation_details": [
            {
                "gen": 0,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -82.0,
                            "priority_mean_rss_dbm": -78.0,
                        },
                    },
                    {
                        "rank": 2,
                        "physical_metrics": {
                            "mean_rss_dbm": -84.0,
                            "priority_mean_rss_dbm": -80.0,
                        },
                    },
                ],
            },
            {
                "gen": 1,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -80.5,
                            "priority_mean_rss_dbm": -76.5,
                        },
                    },
                    {
                        "rank": 2,
                        "physical_metrics": {
                            "mean_rss_dbm": -82.2,
                            "priority_mean_rss_dbm": -78.6,
                        },
                    },
                ],
            },
        ]
    }
    gd_results = {
        "per_seed_analysis": [
            {
                "seed_index": 0,
                "physical_metrics": {
                    "mean_rss_dbm": -79.0,
                    "priority_mean_rss_dbm": -75.5,
                },
            },
            {
                "seed_index": 1,
                "physical_metrics": {
                    "mean_rss_dbm": -80.0,
                    "priority_mean_rss_dbm": -76.2,
                },
            },
        ]
    }

    captured: dict[str, Any] = {}
    real_subplots = memetic_plotting.plt.subplots

    def _capture_subplots(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(memetic_plotting.plt, "subplots", _capture_subplots)

    output_path = tmp_path / "rank2_stitched_plot.png"
    rendered = memetic_plotting.save_ga_gd_stitched_metric_trend_plot(
        ga_results=ga_results,
        gd_results=gd_results,
        save_path=output_path,
        metric_key="mean_rss_dbm",
        priority_metric_key="priority_mean_rss_dbm",
        metric_label="Mean RSSI (dBm)",
        rank=2,
        gd_seed_index=1,
    )

    assert rendered is not None
    assert output_path.exists()

    ax = captured["ax"]
    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["GA all-area", "GA priority-area", "GD all-area", "GD priority-area"]
    assert all("rank" not in label.lower() for label in labels)
    assert "Mean RSSI (dBm) Trend" in ax.get_title()
    assert isinstance(ax.xaxis.get_major_locator(), MaxNLocator)


def test_compute_rssi_metric_y_limits_rounds_shared_range_across_ranks() -> None:
    """RSSI y-limits should be shared across ranks and rounded to multiples of 5."""
    ga_results = {
        "num_selected_seeds": 2,
        "generation_details": [
            {
                "gen": 0,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -82.3,
                            "p5_rss_dbm": -96.7,
                            "min_rss_dbm": -104.1,
                            "priority_mean_rss_dbm": -79.1,
                            "priority_p5_rss_dbm": -93.2,
                            "priority_min_rss_dbm": -101.6,
                        },
                    },
                    {
                        "rank": 2,
                        "physical_metrics": {
                            "mean_rss_dbm": -84.4,
                            "p5_rss_dbm": -98.9,
                            "min_rss_dbm": -106.2,
                            "priority_mean_rss_dbm": -81.3,
                            "priority_p5_rss_dbm": -95.7,
                            "priority_min_rss_dbm": -103.4,
                        },
                    },
                ],
            },
            {
                "gen": 1,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -77.4,
                            "p5_rss_dbm": -90.2,
                            "min_rss_dbm": -101.2,
                            "priority_mean_rss_dbm": -74.8,
                            "priority_p5_rss_dbm": -87.5,
                            "priority_min_rss_dbm": -98.9,
                        },
                    },
                    {
                        "rank": 2,
                        "physical_metrics": {
                            "mean_rss_dbm": -80.1,
                            "p5_rss_dbm": -94.6,
                            "min_rss_dbm": -103.9,
                            "priority_mean_rss_dbm": -76.1,
                            "priority_p5_rss_dbm": -89.8,
                            "priority_min_rss_dbm": -99.5,
                        },
                    },
                ],
            },
        ],
    }

    limits = memetic_plotting._compute_rssi_metric_y_limits(ga_results=ga_results)

    assert limits["mean_rss_dbm"] == (-85.0, -70.0)
    assert limits["p5_rss_dbm"] == (-100.0, -85.0)
    assert limits["min_rss_dbm"] == (-110.0, -95.0)


def test_metric_trend_plots_apply_shared_rssi_y_limits(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """GA metric trend save loop should pass one shared y-limit per metric."""
    ga_results = {
        "num_selected_seeds": 2,
        "generation_details": [
            {
                "gen": 0,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -82.0,
                            "p5_rss_dbm": -96.0,
                            "min_rss_dbm": -101.0,
                            "priority_mean_rss_dbm": -78.0,
                            "priority_p5_rss_dbm": -92.0,
                            "priority_min_rss_dbm": -99.0,
                        },
                    },
                    {
                        "rank": 2,
                        "physical_metrics": {
                            "mean_rss_dbm": -84.0,
                            "p5_rss_dbm": -98.0,
                            "min_rss_dbm": -103.0,
                            "priority_mean_rss_dbm": -80.0,
                            "priority_p5_rss_dbm": -94.0,
                            "priority_min_rss_dbm": -101.0,
                        },
                    },
                ],
            },
            {
                "gen": 1,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -80.5,
                            "p5_rss_dbm": -94.5,
                            "min_rss_dbm": -100.0,
                            "priority_mean_rss_dbm": -76.5,
                            "priority_p5_rss_dbm": -89.0,
                            "priority_min_rss_dbm": -97.0,
                        },
                    },
                    {
                        "rank": 2,
                        "physical_metrics": {
                            "mean_rss_dbm": -82.1,
                            "p5_rss_dbm": -96.3,
                            "min_rss_dbm": -101.8,
                            "priority_mean_rss_dbm": -78.2,
                            "priority_p5_rss_dbm": -91.3,
                            "priority_min_rss_dbm": -98.8,
                        },
                    },
                ],
            },
        ],
    }

    captured: list[tuple[str, int, Any]] = []

    def _fake_save_plot(
        ga_results: Mapping[str, Any],
        save_path: Path,
        metric_key: str,
        priority_metric_key: str,
        metric_label: str,
        rank: int = 1,
        y_limits: Any = None,
        gd_results: Mapping[str, Any] | None = None,
        gd_seed_index: int | None = None,
    ) -> str:
        del ga_results, save_path, priority_metric_key, metric_label, gd_results, gd_seed_index
        captured.append((metric_key, rank, y_limits))
        return str(tmp_path / f"{metric_key}_rank{rank}.png")

    monkeypatch.setattr(memetic_plotting, "save_ga_generation_best_metric_trend_plot", _fake_save_plot)

    artifacts = memetic_plotting.save_ga_generation_best_metric_trend_plots(
        ga_results=ga_results,
        save_dir=tmp_path,
    )

    assert artifacts["ga_best_metric_trend_plot_count"] == "6"
    assert len(captured) == 6

    for metric_name in ("mean_rss_dbm", "p5_rss_dbm", "min_rss_dbm"):
        metric_limits = [y_limits for metric_key, _, y_limits in captured if metric_key == metric_name]
        assert len(metric_limits) == 2
        assert metric_limits[0] == metric_limits[1]
        assert metric_limits[0] is not None
        assert metric_limits[0][0] % 5.0 == 0.0
        assert metric_limits[0][1] % 5.0 == 0.0


def test_mean_p5_combined_plot_uses_default_y_range(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Combined mean+p5 plot should default to y-range [-80, -40]."""
    ga_results = {
        "generation_details": [
            {
                "gen": 0,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -74.0,
                            "p5_rss_dbm": -88.0,
                            "priority_mean_rss_dbm": -70.0,
                            "priority_p5_rss_dbm": -84.0,
                        },
                    },
                    {
                        "rank": 2,
                        "physical_metrics": {
                            "mean_rss_dbm": -76.0,
                            "p5_rss_dbm": -90.0,
                            "priority_mean_rss_dbm": -72.0,
                            "priority_p5_rss_dbm": -86.0,
                        },
                    },
                ],
            },
            {
                "gen": 1,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -72.0,
                            "p5_rss_dbm": -86.0,
                            "priority_mean_rss_dbm": -68.0,
                            "priority_p5_rss_dbm": -82.0,
                        },
                    },
                    {
                        "rank": 2,
                        "physical_metrics": {
                            "mean_rss_dbm": -75.0,
                            "p5_rss_dbm": -89.0,
                            "priority_mean_rss_dbm": -71.0,
                            "priority_p5_rss_dbm": -85.0,
                        },
                    },
                ],
            },
        ]
    }

    captured: dict[str, Any] = {}
    real_subplots = memetic_plotting.plt.subplots

    def _capture_subplots(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(memetic_plotting.plt, "subplots", _capture_subplots)

    output_path = tmp_path / "mean_p5_combo_rank2.png"
    rendered = memetic_plotting.save_ga_generation_mean_p5_combined_plot(
        ga_results=ga_results,
        save_path=output_path,
        rank=2,
    )

    assert rendered is not None
    assert output_path.exists()

    ax = captured["ax"]
    ymin, ymax = ax.get_ylim()
    assert ymin == pytest.approx(-80.0)
    assert ymax == pytest.approx(-40.0)


def test_mean_p5_combined_plots_render_per_rank(tmp_path: Path) -> None:
    """Combined mean+p5 plot generation should produce one artifact per rank."""
    ga_results = {
        "num_selected_seeds": 2,
        "generation_details": [
            {
                "gen": 0,
                "top_individuals": [
                    {
                        "rank": 1,
                        "physical_metrics": {
                            "mean_rss_dbm": -74.0,
                            "p5_rss_dbm": -88.0,
                            "priority_mean_rss_dbm": -70.0,
                            "priority_p5_rss_dbm": -84.0,
                        },
                    },
                    {
                        "rank": 2,
                        "physical_metrics": {
                            "mean_rss_dbm": -76.0,
                            "p5_rss_dbm": -90.0,
                            "priority_mean_rss_dbm": -72.0,
                            "priority_p5_rss_dbm": -86.0,
                        },
                    },
                ],
            },
        ],
    }

    artifacts = memetic_plotting.save_ga_generation_mean_p5_combined_plots(
        ga_results=ga_results,
        save_dir=tmp_path,
    )

    assert artifacts["ga_mean_p5_combined_trend_plot_count"] == "2"
    assert "ga_mean_p5_combined_trend_plot" in artifacts
    assert "ga_mean_p5_combined_trend_plot_rank2" in artifacts
    assert Path(artifacts["ga_mean_p5_combined_trend_plot"]).exists()
    assert Path(artifacts["ga_mean_p5_combined_trend_plot_rank2"]).exists()


def test_build_ga_generation_metric_rows_uses_priority_metrics() -> None:
    """GA generation CSV rows should expose canonical priority metric columns."""
    rows = run_memetic_pipeline._build_ga_generation_metric_rows(
        [
            {
                "gen": 0,
                "best_primary_fitness": -0.25,
                "top_individuals": [
                    {
                        "rank": 1,
                        "primary_fitness": -0.25,
                        "physical_metrics": {
                            "mean_rss_dbm": -82.0,
                            "p5_rss_dbm": -96.0,
                            "min_rss_dbm": -101.0,
                            "priority_mean_rss_dbm": -78.0,
                            "priority_p5_rss_dbm": -92.0,
                            "priority_min_rss_dbm": -99.0,
                        },
                    },
                    {
                        "rank": 2,
                        "primary_fitness": -0.30,
                        "physical_metrics": {
                            "mean_rss_dbm": -83.0,
                            "p5_rss_dbm": -97.0,
                            "min_rss_dbm": -102.0,
                            "priority_mean_rss_dbm": -79.5,
                            "priority_p5_rss_dbm": -93.3,
                            "priority_min_rss_dbm": -100.8,
                        },
                    },
                ],
                "best_physical_metrics": {
                    "mean_rss_dbm": -82.0,
                    "p5_rss_dbm": -96.0,
                    "min_rss_dbm": -101.0,
                    "priority_mean_rss_dbm": -78.0,
                    "priority_p5_rss_dbm": -92.0,
                    "priority_min_rss_dbm": -99.0,
                },
            }
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["best_primary_loss"] == 0.25
    assert row["priority_mean_rss_dbm"] == -78.0
    assert row["priority_p5_rss_dbm"] == -92.0
    assert row["priority_min_rss_dbm"] == -99.0
    assert row["rank_1_priority_mean_rss_dbm"] == -78.0
    assert row["rank_2_primary_fitness"] == -0.30
    assert row["rank_2_primary_loss"] == 0.30
    assert row["rank_2_priority_mean_rss_dbm"] == -79.5
    assert row["second_primary_fitness"] == -0.30
    assert row["second_primary_loss"] == 0.30


def test_memetic_summary_reports_final_generation_top_ranks() -> None:
    """Summary report should include rank-aware GA final-generation details."""
    summary = {
        "ga_results": {
            "best_primary_fitness": -0.25,
            "num_selected_seeds": 2,
            "generation_details": [
                {
                    "gen": 1,
                    "top_individuals": [
                        {
                            "rank": 1,
                            "primary_fitness": -0.25,
                            "physical_metrics": {
                                "mean_rss_dbm": -82.0,
                                "priority_mean_rss_dbm": -78.0,
                                "p5_rss_dbm": -96.0,
                                "priority_p5_rss_dbm": -92.0,
                            },
                        },
                        {
                            "rank": 2,
                            "primary_fitness": -0.30,
                            "physical_metrics": {
                                "mean_rss_dbm": -83.0,
                                "priority_mean_rss_dbm": -79.0,
                                "p5_rss_dbm": -97.0,
                                "priority_p5_rss_dbm": -93.0,
                            },
                        },
                    ],
                }
            ],
            "seeds": [
                {
                    "rank": 1,
                    "primary_fitness": -0.25,
                    "min_distance_to_previous": None,
                    "physical_metrics": {
                        "mean_rss_dbm": -82.0,
                        "priority_mean_rss_dbm": -78.0,
                    },
                },
                {
                    "rank": 2,
                    "primary_fitness": -0.30,
                    "min_distance_to_previous": 4.5,
                    "physical_metrics": {
                        "mean_rss_dbm": -83.0,
                        "priority_mean_rss_dbm": -79.0,
                    },
                },
            ],
        },
        "gd_results": {"metrics": {}},
        "timings": {},
        "counts": {},
        "global_best_result": None,
    }

    report = memetic_summary.build_memetic_summary_report(summary)

    assert "## GA Final Generation Top Individuals" in report
    assert "- Rank #1:" in report
    assert "- Rank #2:" in report
    assert "## GA Seeds Used For GD" in report
    assert "- Seed rank #1:" in report
    assert "- Seed rank #2:" in report
