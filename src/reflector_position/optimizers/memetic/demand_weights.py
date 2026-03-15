"""Utilities for constructing static spatial demand-weight maps.

The demand map is defined over the 2-D optimization grid and is intended to be
shared by all objective evaluations in a run. The returned weights obey a
conservation rule: after masking invalid cells and any optional smoothing, the
sum of all weights equals the number of valid cells. This keeps optimizer
hyperparameters numerically stable when demand weighting is enabled.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


def generate_spatial_weight_map(
    num_rows: int,
    num_cols: int,
    demand_config: dict[str, Any] | Mapping[str, Any],
    valid_mask: Tensor,
) -> Tensor:
    """Generate a normalized 2-D spatial demand-weight tensor.

    Parameters
    ----------
    num_rows : int
        Number of rows in the optimization grid.
    num_cols : int
        Number of columns in the optimization grid.
    demand_config : dict[str, Any] | Mapping[str, Any]
        Demand-weight configuration. Supported keys are ``enabled``,
        ``bounding_boxes``, ``box_weights``, and ``apply_blur``.
    valid_mask : torch.Tensor
        Boolean or numeric 2-D mask with shape ``(num_rows, num_cols)`` where
        valid walkable cells are 1 and walls/voids are 0.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape ``(num_rows, num_cols)`` on the same device as
        ``valid_mask`` whose sum equals the number of valid cells.
    """
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError("num_rows and num_cols must be positive")
    if valid_mask.shape != (num_rows, num_cols):
        raise ValueError(
            "valid_mask shape must match (num_rows, num_cols); "
            f"got {tuple(valid_mask.shape)}"
        )

    mask = valid_mask.to(dtype=torch.float32, device=valid_mask.device)
    weight_map = torch.ones((num_rows, num_cols), dtype=torch.float32, device=mask.device)

    if bool(demand_config.get("enabled", True)):
        bounding_boxes = list(demand_config.get("bounding_boxes", []))
        box_weights = list(demand_config.get("box_weights", []))

        if len(bounding_boxes) != len(box_weights):
            raise ValueError("bounding_boxes and box_weights must have the same length")

        for box, weight in zip(bounding_boxes, box_weights):
            if len(box) != 2 or len(box[0]) != 2 or len(box[1]) != 2:
                raise ValueError(
                    "each bounding box must be [[row_min, col_min], [row_max, col_max]]"
                )

            r1 = max(0, min(num_rows, int(box[0][0])))
            c1 = max(0, min(num_cols, int(box[0][1])))
            r2 = max(0, min(num_rows, int(box[1][0])))
            c2 = max(0, min(num_cols, int(box[1][1])))

            if r2 <= r1 or c2 <= c1:
                continue

            weight_map[r1:r2, c1:c2] = float(weight)

    weight_map = weight_map * mask

    if bool(demand_config.get("enabled", True)) and bool(demand_config.get("apply_blur", False)):
        blurred = F.avg_pool2d(
            weight_map.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_map = blurred.squeeze(0).squeeze(0)
        weight_map = weight_map * mask

    num_valid_cells = mask.sum()
    current_weight_sum = weight_map.sum()

    if num_valid_cells <= 0:
        raise ValueError("valid_mask must contain at least one valid cell")
    if current_weight_sum <= 0:
        raise ValueError("weight_map sum must be positive after masking and blur")

    normalized_weight_map = weight_map * (num_valid_cells / current_weight_sum)
    return normalized_weight_map


__all__ = ["generate_spatial_weight_map"]
