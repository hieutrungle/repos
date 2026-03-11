"""Standalone differentiable objectives for the memetic optimization pipeline.

This module centralizes the continuous objective landscape shared by the
memetic Genetic Algorithm (GA) and Gradient Descent (GD) stages.

The design goal is to expose a single source of truth for the optimization
manifold:

1. ``SoftMinLoss`` rewards improvements in weak-signal regions through a
   smooth minimum surrogate.
2. ``SoftCoverageLoss`` rewards broader area coverage above a target
   threshold through a sigmoid relaxation.
3. ``MemeticCompositeLoss`` combines both terms into one scalar objective.

All returned values follow a minimization convention. When a metric is
conceptually something we want to maximize, the corresponding loss returns its
negative value so that standard optimizers can minimize it directly.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from reflector_position.metrics import POWER_EPSILON


def _flatten_spatial_dims(coverage_map: Tensor) -> Tensor:
    """Flatten spatial dimensions while preserving an optional batch axis.

    Supported input shapes include:

    - ``(H, W)`` for a single radio map
    - ``(B, H, W)`` for a batch of radio maps
    - ``(N,)`` for an already flattened single map
    - ``(B, d1, ..., dk)`` for any batch-plus-spatial layout

    Parameters
    ----------
    coverage_map : torch.Tensor
        Input tensor containing signal values in linear Watts.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(batch, num_cells)``.
    """
    if coverage_map.ndim == 0:
        raise ValueError("coverage_map must have at least one dimension")

    if coverage_map.ndim == 1:
        return coverage_map.unsqueeze(0)

    if coverage_map.ndim == 2:
        return coverage_map.reshape(1, -1)

    return coverage_map.reshape(coverage_map.shape[0], -1)


def _coverage_map_to_dbm(coverage_map: Tensor) -> Tensor:
    """Convert a linear-power radio map to dBm before loss evaluation."""
    return 10.0 * torch.log10(coverage_map + POWER_EPSILON) + 30.0


class SoftMinLoss(nn.Module):
    r"""Differentiable soft minimum loss for weak-signal improvement.

    This module applies the log-sum-exp soft minimum approximation

    .. math::

        \operatorname{softmin}(x) = -\frac{1}{T}\log\sum_i e^{-T x_i}

    over the spatial cells of a radio map expressed in dBm. In this project
    the optimizer performs minimization, while the physical objective is to
    maximize the soft-min signal level. Since stronger wireless signals
    correspond to less-negative dBm values, the returned loss is

    .. math::

        \mathcal{L}_{\mathrm{softmin}} = -\operatorname{softmin}(x)

    so that minimizing the loss is equivalent to maximizing the smooth
    weakest-link signal surrogate.
    """

    def __init__(self, temperature: float, coverage_threshold_dbm: float) -> None:
        """Initialize the soft minimum loss.

        Parameters
        ----------
        temperature : float
            Positive temperature controlling the sharpness of the soft minimum.
            Larger values weight lower-signal cells more aggressively.
        coverage_threshold_dbm : float
            Coverage threshold in dBm. Cells below this threshold are clamped to it before applying the soft minimum, 
            preventing excessively negative values from dominating the loss and stabilizing optimization in very weak-signal regimes
        """
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.coverage_threshold_dbm = float(coverage_threshold_dbm)

    def forward(self, coverage_map: Tensor) -> Tensor:
        """Compute the minimization-form soft minimum loss.

        Parameters
        ----------
        coverage_map : torch.Tensor
            Radio map tensor in dBm with shape ``(H, W)``,
            ``(B, H, W)``, or any batch-plus-spatial equivalent.

        Returns
        -------
        torch.Tensor
            Scalar loss. Lower is better.
        """
        flat_map = _flatten_spatial_dims(coverage_map)
        clamped_map = torch.clamp(flat_map, min=self.coverage_threshold_dbm)
        softmin_value = -torch.logsumexp(
            -self.temperature * clamped_map,
            dim=-1,
        ) / self.temperature
        return -softmin_value.mean()


class SoftCoverageLoss(nn.Module):
    r"""Differentiable soft coverage loss based on sigmoid occupancy.

    For each spatial cell, the hard coverage indicator

    .. math::

        \mathbf{1}[x_i \geq \theta]

    is replaced with the continuous relaxation

    .. math::

        \sigma\left(T (x_i - \theta)\right)

    where $\theta$ is the coverage threshold in dBm and $T$ controls the
    sigmoid steepness. Averaging these dBm-based occupancy values produces a
    differentiable approximation of coverage percentage.

    The returned loss is the negative of that percentage so that minimizing
    the loss increases soft coverage.
    """

    def __init__(self, threshold_dbm: float, temperature: float) -> None:
        """Initialize the soft coverage loss.

        Parameters
        ----------
        threshold_dbm : float
            Coverage threshold in dBm.
        temperature : float
            Positive sigmoid steepness parameter. Larger values approach a
            harder threshold.
        """
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.threshold_dbm = float(threshold_dbm)
        self.temperature = float(temperature)

    def forward(self, coverage_map: Tensor) -> Tensor:
        """Compute the minimization-form soft coverage loss.

        Parameters
        ----------
        coverage_map : torch.Tensor
            Radio map tensor in dBm with shape ``(H, W)``,
            ``(B, H, W)``, or any batch-plus-spatial equivalent.

        Returns
        -------
        torch.Tensor
            Scalar loss equal to the negative soft coverage percentage.
        """
        flat_map = _flatten_spatial_dims(coverage_map)
        soft_coverage = torch.sigmoid(
            self.temperature * (flat_map - self.threshold_dbm)
        ).mean(dim=-1)
        return -soft_coverage.mean()


class MemeticCompositeLoss(nn.Module):
    r"""Composite memetic objective shared by GA and GD.

    The total objective is

    .. math::

        \mathcal{L}_{\mathrm{total}} = \alpha
        \mathcal{L}_{\mathrm{softmin}} + \beta
        \mathcal{L}_{\mathrm{coverage}}

    where both component losses are already expressed in minimization form.
    The composite module converts the incoming radio map from linear Watts to
    dBm exactly once, then evaluates both terms on the same dBm manifold.
    ``SoftMinLoss`` returns the negative dBm soft minimum, and
    ``SoftCoverageLoss`` returns the negative soft coverage percentage. This
    keeps the optimization landscape fully continuous and avoids any masking
    or hard selection operations that would break gradient flow.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        softmin_temperature: float,
        coverage_threshold_dbm: float,
        coverage_temperature: float,
    ) -> None:
        """Initialize the composite memetic loss.

        Parameters
        ----------
        alpha : float
            Weight applied to the soft-min term.
        beta : float
            Weight applied to the soft-coverage term.
        softmin_temperature : float
            Temperature parameter for :class:`SoftMinLoss`.
        coverage_threshold_dbm : float
            Threshold in dBm used by :class:`SoftCoverageLoss`.
        coverage_temperature : float
            Sigmoid steepness used by :class:`SoftCoverageLoss`.
        """
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.softmin_loss = SoftMinLoss(temperature=softmin_temperature, coverage_threshold_dbm=coverage_threshold_dbm)
        self.coverage_loss = SoftCoverageLoss(
            threshold_dbm=coverage_threshold_dbm,
            temperature=coverage_temperature,
        )

    def forward(self, coverage_map: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the total loss and detached auxiliary loss components.

        Parameters
        ----------
        coverage_map : torch.Tensor
            Radio map tensor in linear Watts with shape ``(H, W)``,
            ``(B, H, W)``, or any batch-plus-spatial equivalent.

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            A tuple ``(total_loss, components)`` where ``total_loss`` is a
            scalar tensor suitable for backpropagation and ``components`` holds
            detached scalar components for logging.
        """
        coverage_map_dbm = _coverage_map_to_dbm(coverage_map)
        softmin_loss = self.softmin_loss(coverage_map_dbm)
        coverage_loss = self.coverage_loss(coverage_map_dbm)
        total_loss = self.alpha * softmin_loss + self.beta * coverage_loss

        components = {
            "softmin_loss": float(softmin_loss.detach().item()),
            "coverage_loss": float(coverage_loss.detach().item()),
        }
        return total_loss, components


__all__ = [
    "SoftMinLoss",
    "SoftCoverageLoss",
    "MemeticCompositeLoss",
]