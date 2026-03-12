import pytest
import torch

from reflector_position.optimizers.memetic.memetic_loss import MemeticCompositeLoss


@pytest.mark.unit
def test_memetic_composite_loss_decreases_when_signal_strength_increases() -> None:
    loss_module = MemeticCompositeLoss(
        alpha=2.0,
        beta=0.001,
        softmin_temperature=0.2,
        coverage_threshold_dbm=-120.0,
        coverage_temperature=2.0,
    )

    weaker_map = torch.full((4, 4), 1.0e-11, dtype=torch.float32)
    stronger_map = torch.full((4, 4), 1.0e-10, dtype=torch.float32)

    weaker_loss, _ = loss_module(weaker_map)
    stronger_loss, _ = loss_module(stronger_map)

    assert float(stronger_loss.item()) < float(weaker_loss.item())


@pytest.mark.unit
def test_memetic_coverage_component_rewards_more_cells_above_threshold() -> None:
    loss_module = MemeticCompositeLoss(
        alpha=0.0,
        beta=1.0,
        softmin_temperature=0.2,
        coverage_threshold_dbm=-120.0,
        coverage_temperature=2.0,
    )

    mostly_uncovered = torch.full((4, 4), 1.0e-16, dtype=torch.float32)
    mostly_uncovered[0, 0] = 1.0e-10

    mostly_covered = torch.full((4, 4), 1.0e-10, dtype=torch.float32)

    uncovered_loss, uncovered_parts = loss_module(mostly_uncovered)
    covered_loss, covered_parts = loss_module(mostly_covered)

    assert float(covered_loss.item()) < float(uncovered_loss.item())
    assert covered_parts["coverage_loss"] < uncovered_parts["coverage_loss"]