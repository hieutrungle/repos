"""
Metrics for evaluating radio map quality and coverage.

This module provides functions for computing various metrics from radio maps,
including minimum RSS, soft minimum RSS, and coverage area calculations.
"""

import torch


def compute_min_rss_metric(rss_map: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum RSS value in the radio map (in linear scale).

    This is the optimization objective: we want to maximize the minimum RSS
    to improve worst-case coverage.

    Args:
        rss_map: Radio map RSS tensor (in Watts) - PyTorch tensor

    Returns:
        Minimum RSS value (scalar tensor)
    """
    # Filter out invalid values (zeros or very small values)
    valid_mask = rss_map > 1e-15
    valid_rss = rss_map[valid_mask]

    if valid_rss.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)

    # Return minimum RSS (we want to maximize this)
    return torch.min(valid_rss)


def compute_soft_min_rss_metric(
    rss_map: torch.Tensor, temperature: float = 0.2
) -> torch.Tensor:
    """
    Compute a soft (differentiable) minimum RSS using LogSumExp trick.

    The soft minimum provides smoother gradients than hard minimum.
    Lower temperature -> closer to true minimum.

    Args:
        rss_map: Radio map RSS tensor (in Watts) - PyTorch tensor
        temperature: Temperature parameter (lower = closer to hard min)

    Returns:
        Soft minimum RSS value (PyTorch tensor)
    """
    # Flatten to 1D for logsumexp
    rss_flat = rss_map.flatten()

    # Filter out invalid values (zeros or very small values) to avoid log issues
    valid_mask = rss_flat > 1e-15
    valid_rss = rss_flat[valid_mask]

    if valid_rss.numel() == 0:
        return torch.tensor(
            0.0, dtype=rss_map.dtype, device=rss_map.device, requires_grad=True
        )

    # Convert to log scale for numerical stability
    log_rss = torch.log(valid_rss)

    # Soft minimum using LogSumExp:
    # softmin(x) = -temperature * logsumexp(-x/temperature)
    soft_min = -temperature * torch.logsumexp(-log_rss / temperature, dim=0)

    return torch.exp(soft_min)


def compute_coverage_metric(
    rss_map: torch.Tensor, threshold_dbm: float = -120.0
) -> torch.Tensor:
    """
    Compute coverage area percentage (RSS above threshold).

    Args:
        rss_map: Radio map RSS tensor (in Watts)
        threshold_dbm: Coverage threshold in dBm

    Returns:
        Coverage percentage (0-100)
    """
    # Convert threshold to linear scale (Watts)
    # P_watt = 10^((P_dbm - 30) / 10)
    threshold_watt = dbm_to_rss(torch.tensor(threshold_dbm))

    # Count cells above threshold
    above_threshold = (rss_map > threshold_watt).float()
    coverage = torch.mean(above_threshold) * 100.0  # Convert to percentage

    return coverage


def rss_to_dbm(rss_watt: torch.Tensor) -> torch.Tensor:
    """
    Convert RSS from Watts to dBm.

    Args:
        rss_watt: RSS value(s) in Watts (torch tensor)

    Returns:
        RSS value(s) in dBm (torch tensor)
    """
    return 10 * torch.log10(rss_watt + 1e-16) + 30.0

def dbm_to_rss(rss_dbm: torch.Tensor) -> torch.Tensor:
    """
    Convert RSS from dBm to Watts.

    Args:
        rss_dbm: RSS value(s) in dBm (torch tensor)
    Returns:
        RSS value(s) in Watts (torch tensor)    
    """
    return 10 ** ((rss_dbm - 30.0) / 10)