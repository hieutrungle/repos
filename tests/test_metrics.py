import pytest
import torch

from reflector_position.metrics import (
    compute_coverage_metric,
    compute_thresholded_reporting_metrics,
    rss_to_dbm,
)


@pytest.mark.unit
def test_reporting_metrics_use_only_numerically_valid_cells() -> None:
    rss_map = torch.tensor(
        [
            [1.0e-16, 1.0e-15],
            [2.0e-15, 1.0e-13],
        ],
        dtype=torch.float32,
    )

    metrics = compute_thresholded_reporting_metrics(rss_map)

    valid_dbm = rss_to_dbm(torch.tensor([1.0e-15, 2.0e-15, 1.0e-13], dtype=torch.float32))
    assert metrics["coverage_pct"] == pytest.approx(75.0)
    assert metrics["valid_cell_count"] == pytest.approx(3.0)
    assert metrics["total_cell_count"] == pytest.approx(4.0)
    assert metrics["min_rss_dbm"] == pytest.approx(float(valid_dbm.min().item()))
    assert metrics["p5_rss_dbm"] == pytest.approx(float(torch.quantile(valid_dbm, 0.05).item()))
    assert metrics["mean_rss_dbm"] == pytest.approx(float(valid_dbm.mean().item()))


@pytest.mark.unit
def test_thresholded_reporting_metrics_fall_back_to_power_floor_when_no_cells_are_covered() -> None:
    rss_map = torch.full((2, 2), 1.0e-16, dtype=torch.float32)

    metrics = compute_thresholded_reporting_metrics(rss_map)

    assert metrics["coverage_pct"] == pytest.approx(0.0)
    assert metrics["valid_cell_count"] == pytest.approx(0.0)
    assert metrics["total_cell_count"] == pytest.approx(4.0)
    assert metrics["min_rss_dbm"] == pytest.approx(-130.0)
    assert metrics["p5_rss_dbm"] == pytest.approx(-130.0)
    assert metrics["mean_rss_dbm"] == pytest.approx(-130.0)


@pytest.mark.unit
def test_reporting_coverage_is_independent_from_threshold_coverage_metric() -> None:
    rss_map = torch.tensor(
        [
            [1.0e-16, 1.0e-15],
            [2.0e-15, 1.0e-13],
        ],
        dtype=torch.float32,
    )

    metrics = compute_thresholded_reporting_metrics(rss_map)
    coverage = compute_coverage_metric(rss_map, threshold_dbm=-120.0)

    assert metrics["coverage_pct"] == pytest.approx(75.0)
    assert float(coverage.item()) == pytest.approx(50.0)


@pytest.mark.unit
def test_priority_reporting_uses_only_explicit_priority_area_with_plain_stats() -> None:
    rss_map = torch.tensor(
        [
            [1.0e-10, 1.0e-12],
            [1.0e-14, 1.0e-15],
        ],
        dtype=torch.float32,
    )
    spatial_weights = torch.tensor(
        [
            [0.0, 3.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )

    metrics = compute_thresholded_reporting_metrics(
        rss_map,
        percentile=0.05,
        spatial_weights=spatial_weights,
        include_weighted=True,
    )

    selected_valid_dbm = rss_to_dbm(torch.tensor([1.0e-12, 1.0e-14], dtype=torch.float32))
    assert metrics["priority_total_cell_count"] == pytest.approx(2.0)
    assert metrics["priority_valid_cell_count"] == pytest.approx(2.0)
    assert metrics["priority_coverage_pct"] == pytest.approx(100.0)
    assert metrics["priority_min_rss_dbm"] == pytest.approx(float(selected_valid_dbm.min().item()))
    assert metrics["priority_p5_rss_dbm"] == pytest.approx(
        float(torch.quantile(selected_valid_dbm.float(), 0.05).item())
    )
    assert metrics["priority_mean_rss_dbm"] == pytest.approx(float(selected_valid_dbm.mean().item()))

@pytest.mark.unit
def test_priority_reporting_uses_emphasized_cells_when_weights_are_positive_everywhere() -> None:
    rss_map = torch.tensor(
        [
            [1.0e-13, 1.0e-12],
            [1.0e-11, 1.0e-10],
        ],
        dtype=torch.float32,
    )
    spatial_weights = torch.tensor(
        [
            [1.0, 1.0],
            [5.0, 5.0],
        ],
        dtype=torch.float32,
    )

    metrics = compute_thresholded_reporting_metrics(
        rss_map,
        percentile=0.05,
        spatial_weights=spatial_weights,
        include_weighted=True,
    )

    selected_valid_dbm = rss_to_dbm(torch.tensor([1.0e-11, 1.0e-10], dtype=torch.float32))
    assert metrics["total_cell_count"] == pytest.approx(4.0)
    assert metrics["priority_total_cell_count"] == pytest.approx(2.0)
    assert metrics["priority_valid_cell_count"] == pytest.approx(2.0)
    assert metrics["priority_coverage_pct"] == pytest.approx(100.0)
    assert metrics["priority_mean_rss_dbm"] == pytest.approx(float(selected_valid_dbm.mean().item()))
    assert metrics["priority_mean_rss_dbm"] != pytest.approx(metrics["mean_rss_dbm"])