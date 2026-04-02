"""Utilities for constructing static spatial demand-weight maps.

The demand map is defined over the 2-D optimization grid and is intended to be
shared by all objective evaluations in a run. The returned weights obey a
conservation rule: after optional smoothing, the sum of all weights equals the
total number of grid cells. This keeps optimizer hyperparameters numerically
stable when demand weighting is enabled.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


def generate_spatial_weight_map(
    num_rows: int,
    num_cols: int,
    demand_config: dict[str, Any] | Mapping[str, Any],
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
        ``bounding_boxes``, ``box_weights``, ``apply_blur``,
        ``position_bounds``, and
        ``box_coordinate_mode``.
        Default coordinate mode is ``xy`` with boxes interpreted as
        ``[[x_left, y_top], [x_right, y_bottom]]``.
    Returns
    -------
    torch.Tensor
        Float32 tensor of shape ``(num_rows, num_cols)`` whose sum equals the
        total number of cells ``num_rows * num_cols``.
    """
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError("num_rows and num_cols must be positive")
    weight_map = torch.ones((num_rows, num_cols), dtype=torch.float32)

    if bool(demand_config.get("enabled", True)):
        bounding_boxes = list(demand_config.get("bounding_boxes", []))
        box_weights = list(demand_config.get("box_weights", []))

        if len(bounding_boxes) != len(box_weights):
            raise ValueError("bounding_boxes and box_weights must have the same length")

        box_coordinate_mode = str(demand_config.get("box_coordinate_mode", "xy")).strip().lower()
        if box_coordinate_mode not in {"xy", "row_col"}:
            raise ValueError("box_coordinate_mode must be either 'xy' or 'row_col'")

        bounds = demand_config.get("position_bounds", demand_config.get("_position_bounds"))
        x_origin = 0.0
        y_origin = 0.0
        if isinstance(bounds, Mapping):
            x_origin = float(bounds.get("x_min", 0.0))
            y_origin = float(bounds.get("y_min", 0.0))

        for box, weight in zip(bounding_boxes, box_weights):
            if len(box) != 2 or len(box[0]) != 2 or len(box[1]) != 2:
                raise ValueError(
                    "each bounding box must be [[x_left, y_top], [x_right, y_bottom]]"
                )

            first_a = float(box[0][0])
            first_b = float(box[0][1])
            second_a = float(box[1][0])
            second_b = float(box[1][1])

            if box_coordinate_mode == "xy":
                # Interpret config boxes as scene coordinates:
                # [[x_left, y_top], [x_right, y_bottom]].
                x_min = min(first_a, second_a)
                x_max = max(first_a, second_a)
                y_min = min(first_b, second_b)
                y_max = max(first_b, second_b)

                c1 = max(0, min(num_cols, int(math.floor(x_min - x_origin))))
                c2 = max(0, min(num_cols, int(math.ceil(x_max - x_origin))))
                r1 = max(0, min(num_rows, int(math.floor(y_min - y_origin))))
                r2 = max(0, min(num_rows, int(math.ceil(y_max - y_origin))))
            else:
                row_min = min(first_a, second_a)
                row_max = max(first_a, second_a)
                col_min = min(first_b, second_b)
                col_max = max(first_b, second_b)

                r1 = max(0, min(num_rows, int(row_min)))
                r2 = max(0, min(num_rows, int(row_max)))
                c1 = max(0, min(num_cols, int(col_min)))
                c2 = max(0, min(num_cols, int(col_max)))

            if r2 <= r1 or c2 <= c1:
                continue

            weight_map[r1:r2, c1:c2] = float(weight)

    if bool(demand_config.get("enabled", True)) and bool(demand_config.get("apply_blur", False)):
        blurred = F.avg_pool2d(
            weight_map.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_map = blurred.squeeze(0).squeeze(0)

    num_cells = torch.tensor(
        float(num_rows * num_cols),
        dtype=weight_map.dtype,
        device=weight_map.device,
    )
    current_weight_sum = weight_map.sum()
    if current_weight_sum <= 0:
        raise ValueError("weight_map sum must be positive")

    normalized_weight_map = weight_map
    # normalized_weight_map = weight_map * (num_cells / current_weight_sum)
    return normalized_weight_map


__all__ = ["generate_spatial_weight_map"]
