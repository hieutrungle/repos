"""Interactive plotting utilities for memetic loss-function intuition.

This script provides five Plotly-based figures that visualize the equations
documented in ``context/loss_function.md``:

Part 1: Demand-Weighted Normalized SoftMin Loss
1) Raw dBm vs normalized score
2) Raw dBm vs normalized penalty
3) Penalty vs weighted normalized softmin loss
4) Normalized score vs weighted normalized softmin loss
5) Raw dBm vs weighted normalized softmin loss

Part 2: Soft Coverage Loss
6) Raw dBm vs soft coverage loss

All figures are interactive and can be shown in a browser or exported as HTML.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

try:
	import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - import guard for optional dep
	raise ImportError(
		"plotly is required for interactive plots. Install with: pip install plotly"
	) from exc


def normalize_dbm_values(
	x_dbm: np.ndarray,
	floor_dbm: float,
	ceil_dbm: float,
) -> np.ndarray:
	"""Normalize raw dBm values into score space using floor/ceiling anchors."""
	if ceil_dbm <= floor_dbm:
		raise ValueError("ceil_dbm must be greater than floor_dbm")
	return (x_dbm - floor_dbm) / (ceil_dbm - floor_dbm)


def penalty_from_normalized_score(scores: np.ndarray) -> np.ndarray:
	"""Convert normalized scores to penalties with top-end capping only.

	Scores above 1.0 are capped, while low scores remain unbounded below.
	This mirrors ``WeightedNormalizedSoftMinLoss``.
	"""
	capped_scores = np.minimum(scores, 1.0)
	return 1.0 - capped_scores


def weighted_normalized_softmin_loss(
	penalties: np.ndarray,
	weights: np.ndarray,
	temperature: float,
) -> float:
	"""Compute scalar demand-weighted normalized softmin penalty."""
	if temperature <= 0.0:
		raise ValueError("temperature must be positive")
	if penalties.ndim != 1 or weights.ndim != 1:
		raise ValueError("penalties and weights must be 1D arrays")
	if penalties.shape != weights.shape:
		raise ValueError("penalties and weights must have the same shape")
	if np.any(weights < 0):
		raise ValueError("weights must be non-negative")

	scaled_penalties = penalties / temperature
	shifted_inputs = scaled_penalties + np.log(weights + 1e-9)
	weighted_lse = np.logaddexp.reduce(shifted_inputs)
	normalization = np.log(np.sum(weights) + 1e-9)
	return float(temperature * (weighted_lse - normalization))


def soft_coverage_loss(
	x_dbm: np.ndarray,
	threshold_dbm: float,
	temperature: float,
) -> np.ndarray:
	"""Compute pointwise soft coverage loss: ``-sigmoid(T * (x - theta))``."""
	if temperature <= 0.0:
		raise ValueError("temperature must be positive")
	z = (x_dbm - threshold_dbm) / temperature
	return -1.0 / (1.0 + np.exp(-z))


def plot_normalization_curve(
	floor_dbm: float = -120.0,
	ceil_dbm: float = -70.0,
	x_min_dbm: float = -140.0,
	x_max_dbm: float = -40.0,
	num_points: int = 500,
) -> go.Figure:
	"""Plot raw dBm vs normalized score."""
	x = np.linspace(x_min_dbm, x_max_dbm, num_points)
	s = normalize_dbm_values(x, floor_dbm, ceil_dbm)

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=x, y=s, mode="lines", name="Normalized score s(x)"))
	fig.add_vline(x=floor_dbm, line_dash="dash", annotation_text="floor_dbm")
	fig.add_vline(x=ceil_dbm, line_dash="dash", annotation_text="ceil_dbm")
	fig.update_layout(
		title="1) Raw dBm vs Normalized Score",
		xaxis_title="Raw signal x (dBm)",
		yaxis_title="Normalized score s",
		template="plotly_white",
	)
	return fig


def plot_penalty_curve(
	floor_dbm: float = -120.0,
	ceil_dbm: float = -70.0,
	x_min_dbm: float = -140.0,
	x_max_dbm: float = -40.0,
	num_points: int = 500,
) -> go.Figure:
	"""Plot raw dBm vs normalized penalty."""
	x = np.linspace(x_min_dbm, x_max_dbm, num_points)
	s = normalize_dbm_values(x, floor_dbm, ceil_dbm)
	p = penalty_from_normalized_score(s)

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=x, y=p, mode="lines", name="Penalty p(x)"))
	fig.add_vline(x=floor_dbm, line_dash="dash", annotation_text="floor_dbm")
	fig.add_vline(x=ceil_dbm, line_dash="dash", annotation_text="ceil_dbm")
	fig.update_layout(
		title="2) Raw dBm vs Normalized Penalty",
		xaxis_title="Raw signal x (dBm)",
		yaxis_title="Penalty p = 1 - min(s, 1)",
		template="plotly_white",
	)
	return fig


def plot_softmin_loss_vs_penalty(
	temperatures: tuple[float, ...] = (0.5, 0.7, 1.0, 2.0, 5.0),
	fixed_penalties: tuple[float, ...] = (0.2, 0.5, 1.0),
	fixed_weights: tuple[float, ...] = (1.0, 3.0, 0.8),
	varying_weight: float = 2.0,
	p_min: float = -0.5,
	p_max: float = 3.0,
	num_points: int = 400,
) -> go.Figure:
	"""Plot varying penalty value vs weighted normalized softmin loss.

	One cell's penalty is swept from ``p_min`` to ``p_max`` while all other
	penalties and weights stay fixed. One line is drawn per temperature value.
	"""
	p_values = np.linspace(p_min, p_max, num_points)
	fixed_penalties_arr = np.asarray(fixed_penalties, dtype=float)
	fixed_weights_arr = np.asarray(fixed_weights, dtype=float)

	fig = go.Figure()
	for temperature in temperatures:
		y_values = []
		for p in p_values:
			penalties = np.concatenate(([p], fixed_penalties_arr))
			weights = np.concatenate(([varying_weight], fixed_weights_arr))
			y_values.append(
				weighted_normalized_softmin_loss(penalties, weights, temperature)
			)

		fig.add_trace(
			go.Scatter(
				x=p_values,
				y=y_values,
				mode="lines",
				name=f"T={temperature}",
			)
		)
	fig.update_layout(
		title="3) Penalty vs Demand-Weighted Normalized SoftMin Loss",
		xaxis_title="Varying penalty p_var",
		yaxis_title="Weighted normalized softmin loss",
		template="plotly_white",
	)
	return fig


def plot_softmin_loss_vs_normalized_score(
	temperatures: tuple[float, ...] = (0.5, 0.7, 1.0, 2.0, 5.0),
	fixed_penalties: tuple[float, ...] = (0.2, 0.5, 1.0),
	fixed_weights: tuple[float, ...] = (1.0, 3.0, 0.8),
	varying_weight: float = 2.0,
	s_min: float = -2.0,
	s_max: float = 2.0,
	num_points: int = 400,
) -> go.Figure:
	"""Plot varying normalized score vs weighted normalized softmin loss.

	One line is drawn per temperature value.
	"""
	s_values = np.linspace(s_min, s_max, num_points)
	p_values = penalty_from_normalized_score(s_values)

	fixed_penalties_arr = np.asarray(fixed_penalties, dtype=float)
	fixed_weights_arr = np.asarray(fixed_weights, dtype=float)

	fig = go.Figure()
	for temperature in temperatures:
		y_values = []
		for p in p_values:
			penalties = np.concatenate(([p], fixed_penalties_arr))
			weights = np.concatenate(([varying_weight], fixed_weights_arr))
			y_values.append(
				weighted_normalized_softmin_loss(penalties, weights, temperature)
			)

		fig.add_trace(
			go.Scatter(
				x=s_values,
				y=y_values,
				mode="lines",
				name=f"T={temperature}",
			)
		)
	fig.add_vline(x=1.0, line_dash="dash", annotation_text="s=1 cap")
	fig.update_layout(
		title="4) Normalized Score vs Demand-Weighted Normalized SoftMin Loss",
		xaxis_title="Varying normalized score s_var",
		yaxis_title="Weighted normalized softmin loss",
		template="plotly_white",
	)
	return fig


def plot_soft_coverage_loss_curve(
	threshold_dbm: float = -75.0,
	temperatures: tuple[float, ...] = (0.1, 0.3, 0.7, 1.0, 2.0),
	x_min_dbm: float = -120.0,
	x_max_dbm: float = -40.0,
	num_points: int = 500,
) -> go.Figure:
	"""Plot raw dBm vs soft coverage loss for one or more temperatures."""
	x = np.linspace(x_min_dbm, x_max_dbm, num_points)
	fig = go.Figure()
	for t in temperatures:
		y = soft_coverage_loss(x, threshold_dbm=threshold_dbm, temperature=t)
		fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"T_cov={t}"))

	fig.add_vline(x=threshold_dbm, line_dash="dash", annotation_text="threshold")
	fig.update_layout(
		title="6) Raw dBm vs Soft Coverage Loss",
		xaxis_title="Raw signal x (dBm)",
		yaxis_title="L_coverage = -sigmoid(T_cov * (x - theta))",
		template="plotly_white",
	)
	return fig


def plot_softmin_loss_vs_raw_dbm(
	temperatures: tuple[float, ...] = (0.5, 0.7, 1.0, 2.0, 5.0),
	floor_dbm: float = -120.0,
	ceil_dbm: float = -70.0,
	fixed_penalties: tuple[float, ...] = (0.2, 0.5, 1.0),
	fixed_weights: tuple[float, ...] = (1.0, 3.0, 0.8),
	varying_weight: float = 2.0,
	x_min_dbm: float = -140.0,
	x_max_dbm: float = -40.0,
	num_points: int = 400,
) -> go.Figure:
	"""Plot varying raw dBm value vs weighted normalized softmin loss.

	One cell's raw signal is swept from ``x_min_dbm`` to ``x_max_dbm``.
	For each dBm value, score normalization and penalty conversion are applied,
	then the weighted normalized softmin loss is evaluated.
	"""
	x_values = np.linspace(x_min_dbm, x_max_dbm, num_points)
	s_values = normalize_dbm_values(x_values, floor_dbm, ceil_dbm)
	p_values = penalty_from_normalized_score(s_values)

	fixed_penalties_arr = np.asarray(fixed_penalties, dtype=float)
	fixed_weights_arr = np.asarray(fixed_weights, dtype=float)

	fig = go.Figure()
	for temperature in temperatures:
		y_values = []
		for p in p_values:
			penalties = np.concatenate(([p], fixed_penalties_arr))
			weights = np.concatenate(([varying_weight], fixed_weights_arr))
			y_values.append(
				weighted_normalized_softmin_loss(penalties, weights, temperature)
			)

		fig.add_trace(
			go.Scatter(
				x=x_values,
				y=y_values,
				mode="lines",
				name=f"T={temperature}",
			)
		)

	fig.add_vline(x=floor_dbm, line_dash="dash", annotation_text="floor_dbm")
	fig.add_vline(x=ceil_dbm, line_dash="dash", annotation_text="ceil_dbm")
	fig.update_layout(
		title="5) Raw dBm vs Demand-Weighted Normalized SoftMin Loss",
		xaxis_title="Varying raw signal x_var (dBm)",
		yaxis_title="Weighted normalized softmin loss",
		template="plotly_white",
	)
	return fig


def create_all_figures() -> Dict[str, go.Figure]:
	"""Build all six requested figures."""
	return {
		"01_normalization": plot_normalization_curve(),
		"02_penalty": plot_penalty_curve(),
		"03_softmin_vs_penalty": plot_softmin_loss_vs_penalty(),
		"04_softmin_vs_normalized": plot_softmin_loss_vs_normalized_score(),
		"05_softmin_vs_raw_dbm": plot_softmin_loss_vs_raw_dbm(),
		"06_soft_coverage": plot_soft_coverage_loss_curve(),
	}


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Interactive plotting for memetic loss-function components."
	)
	parser.add_argument(
		"--save-dir",
		type=Path,
		default=None,
		help="Optional directory to export each interactive figure as HTML.",
	)
	parser.add_argument(
		"--no-show",
		action="store_true",
		help="If set, do not open figures in browser windows.",
	)
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	figures = create_all_figures()

	if args.save_dir is not None:
		args.save_dir.mkdir(parents=True, exist_ok=True)
		for name, fig in figures.items():
			fig.write_html(args.save_dir / f"{name}.html", include_plotlyjs="cdn")

	if not args.no_show:
		for fig in figures.values():
			fig.show()


if __name__ == "__main__":
	main()
