import pytest
import torch

from reflector_position.metrics import (
    compute_coverage_metric,
    compute_thresholded_reporting_metrics,
    rss_to_dbm,
)


@pytest.mark.unit
def test_thresholded_reporting_metrics_use_only_cells_above_threshold() -> None:
    rss_map = torch.tensor(
        [
            [1.0e-16, 1.0e-15],
            [2.0e-15, 1.0e-13],
        ],
        dtype=torch.float32,
    )

    metrics = compute_thresholded_reporting_metrics(rss_map, threshold_dbm=-120.0)

    valid_dbm = rss_to_dbm(torch.tensor([2.0e-15, 1.0e-13], dtype=torch.float32))
    assert metrics["coverage_pct"] == pytest.approx(50.0)
    assert metrics["valid_cell_count"] == pytest.approx(2.0)
    assert metrics["total_cell_count"] == pytest.approx(4.0)
    assert metrics["min_rss_dbm"] == pytest.approx(float(valid_dbm.min().item()))
    assert metrics["p5_rss_dbm"] == pytest.approx(float(torch.quantile(valid_dbm, 0.05).item()))
    assert metrics["mean_rss_dbm"] == pytest.approx(float(valid_dbm.mean().item()))


@pytest.mark.unit
def test_thresholded_reporting_metrics_fall_back_to_threshold_when_no_cells_are_covered() -> None:
    rss_map = torch.full((2, 2), 1.0e-16, dtype=torch.float32)

    metrics = compute_thresholded_reporting_metrics(rss_map, threshold_dbm=-120.0)

    assert metrics["coverage_pct"] == pytest.approx(0.0)
    assert metrics["valid_cell_count"] == pytest.approx(0.0)
    assert metrics["total_cell_count"] == pytest.approx(4.0)
    assert metrics["min_rss_dbm"] == pytest.approx(-120.0)
    assert metrics["p5_rss_dbm"] == pytest.approx(-120.0)
    assert metrics["mean_rss_dbm"] == pytest.approx(-120.0)


@pytest.mark.unit
def test_coverage_metric_matches_thresholded_reporting_coverage() -> None:
    rss_map = torch.tensor(
        [
            [1.0e-16, 1.0e-15],
            [2.0e-15, 1.0e-13],
        ],
        dtype=torch.float32,
    )

    metrics = compute_thresholded_reporting_metrics(rss_map, threshold_dbm=-120.0)
    coverage = compute_coverage_metric(rss_map, threshold_dbm=-120.0)

    assert float(coverage.item()) == pytest.approx(metrics["coverage_pct"])