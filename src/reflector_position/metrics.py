"""
Metrics for evaluating radio map quality and coverage.

This module provides functions for computing various metrics from radio maps,
including minimum RSS, soft minimum RSS, normalized SoftMin loss, and coverage
area calculations.

Constants:
    POWER_EPSILON: Minimum power floor (Watts) used throughout the module to
        guard against log(0) and to mask dead/invalid cells.  Every function
        that touches linear-Watt values uses this single constant so the
        masking and numerical-safety behaviour is consistent.
"""

import torch

# ---------------------------------------------------------------------------
# Global epsilon — single source of truth for all power-floor comparisons
# ---------------------------------------------------------------------------
POWER_EPSILON: float = 1e-15
"""Minimum receivable power (Watts).  Values below this are treated as
numerical noise and excluded from optimisation metrics.  Also added inside
``log10`` / ``log`` calls to prevent ``-inf``."""


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
    valid_mask = rss_map > POWER_EPSILON
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
    valid_mask = rss_flat > POWER_EPSILON
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
    return 10 * torch.log10(rss_watt + POWER_EPSILON) + 30.0


def dbm_to_rss(rss_dbm: torch.Tensor) -> torch.Tensor:
    """
    Convert RSS from dBm to Watts.

    Args:
        rss_dbm: RSS value(s) in dBm (torch tensor)
    Returns:
        RSS value(s) in Watts (torch tensor)
    """
    return 10 ** ((rss_dbm - 30.0) / 10)


# ---------------------------------------------------------------------------
# Normalized SoftMin Loss
# ---------------------------------------------------------------------------

def normalized_softmin_loss(
    rssi_watts: torch.Tensor,
    temperature: float = 0.1,
    floor_dbm: float = -120.0,
    ceil_dbm: float = -60.0,
) -> torch.Tensor:
    """
    Numerically stable *Normalized SoftMin* loss for AP placement.

    Converts raw linear-Watt RSSI values into a bounded [0, 1] score space
    (via dBm normalisation) and then computes a differentiable soft minimum
    using ``torch.logsumexp``.  Minimising this loss is equivalent to
    *maximising the weakest-link user RSSI* while keeping gradients smooth.

    **Pipeline**::

        Watts ──► dBm ──► [0, 1] scores ──► SoftMin (logsumexp) ──► loss

    Args:
        rssi_watts: Tensor of shape ``(batch_size, num_users)`` or
            ``(num_users,)`` containing received power **in linear Watts**.
            Values in the range 1e-15 … 1e-8 are typical.
        temperature: Positive float controlling the sharpness of the soft
            minimum.  Low values (0.05) approximate ``min()``; high values
            (1.0) approximate ``mean()``.  Default 0.1 is a good starting
            point for AP-placement.
        floor_dbm: dBm value mapped to score 0.0 ("dead zone").
            Default -120 dBm.
        ceil_dbm: dBm value mapped to score 1.0 ("strong signal").
            Default -60 dBm.

    Returns:
        Scalar loss (mean over batch).  Range is approximately [0, 1].
        **Lower is better** (lower loss ⇒ higher worst-case RSSI).

    Example::

        >>> rssi = torch.tensor([[1e-11, 1e-12, 1e-13]])
        >>> loss = normalized_softmin_loss(rssi, temperature=0.1)
        >>> loss.backward()          # gradients flow cleanly

    Notes:
        * Uses the module-level ``POWER_EPSILON`` constant for all numerical
          safety guards, keeping behaviour consistent with
          ``rss_to_dbm`` / ``compute_min_rss_metric``.
        * ``torch.logsumexp`` is used instead of manual ``log(sum(exp(...)))``
          to avoid overflow/underflow at extreme temperatures.
    """
    # -- Ensure 2-D input (batch_size, num_users) --------------------------
    if rssi_watts.dim() == 1:
        rssi_watts = rssi_watts.unsqueeze(0)  # (1, num_users)

    # -- Step 1: Watts → dBm with epsilon safety ---------------------------
    # P_dbm = 10 * log10(P_watts + ε) + 30
    rssi_dbm = 10.0 * torch.log10(rssi_watts + POWER_EPSILON) + 30.0

    # -- Step 2: Normalise dBm → [0, 1] scores ----------------------------
    # Linear map:  floor_dbm → 0.0,  ceil_dbm → 1.0
    #   score = (dBm - floor) / (ceil - floor)
    span = ceil_dbm - floor_dbm  # e.g. -60 - (-120) = 60
    scores = (rssi_dbm - floor_dbm) / span

    # Clamp so that out-of-window values do not produce unbounded gradients
    scores = torch.clamp(scores, min=0.0, max=1.0)

    # -- Step 3: SoftMin via logsumexp -------------------------------------
    # We want to MAXIMISE the minimum score.
    #   softmin(s) = -τ · logsumexp(-s / τ)
    # This is always ≤ min(s), and equals min(s) as τ → 0.
    #
    # To turn it into a LOSS to MINIMISE:
    #   loss = τ · logsumexp(-s / τ)      (positive quantity)
    # Lower loss ⟹ higher worst-case score ⟹ better coverage.
    loss_per_sample = temperature * torch.logsumexp(
        -scores / temperature, dim=-1
    )  # shape: (batch_size,)

    # -- Step 4: Mean over batch -------------------------------------------
    return loss_per_sample.mean()