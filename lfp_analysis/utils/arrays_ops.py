"""Utilities for repeatable tensor selection/aggregation pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


IndexSpec = Union[int, slice, Sequence[int], np.ndarray]


@dataclass(frozen=True)
class Selection:
	"""Simple data container describing which axis and indices to slice."""

	axis: int
	indices: IndexSpec


class ArrayReducer:
	"""Encapsulate the selection → aggregation → flattening pattern.

	Parameters
	----------
	default_selections:
		Iterable of ``Selection`` objects applied in order of ascending axis.
	default_aggregation_axes:
		Axes passed to ``np.nanmean`` after selection (``None`` disables the
		reduction step).
	flatten:
		Whether ``prepare`` should return a flattened 1-D array by default.
	"""

	def __init__(
		self,
		default_selections: Optional[Iterable[Selection]] = None,
		default_aggregation_axes: Optional[Union[int, Sequence[int]]] = None,
		flatten: bool = True,
	) -> None:
		self.default_selections: Tuple[Selection, ...] = tuple(default_selections or ())
		self.default_aggregation_axes = default_aggregation_axes
		self.flatten = flatten

	# ------------------------------------------------------------------
	# Selection helpers
	# ------------------------------------------------------------------
	def apply_selection(
		self,
		array: np.ndarray,
		selections: Optional[Iterable[Selection]] = None,
	) -> np.ndarray:
		result = array
		ops = tuple(selections) if selections is not None else self.default_selections

		for selection in sorted(ops, key=lambda sel: sel.axis):
			result = self._apply_single_selection(result, selection)
		return result

	def _apply_single_selection(self, array: np.ndarray, selection: Selection) -> np.ndarray:
		axis = selection.axis
		if axis < 0:
			axis += array.ndim
		if axis < 0 or axis >= array.ndim:
			raise ValueError(f"Selection axis {selection.axis} invalid for shape {array.shape}")

		indexer: List[Union[slice, IndexSpec]] = [slice(None)] * array.ndim
		indexer[axis] = selection.indices
		return array[tuple(indexer)]

	# ------------------------------------------------------------------
	# Aggregation helpers
	# ------------------------------------------------------------------
	def aggregate(
		self,
		array: np.ndarray,
		axes: Optional[Union[int, Sequence[int]]] = None,
	) -> np.ndarray:
		aggregate_axes = axes
		if aggregate_axes is None:
			aggregate_axes = self.default_aggregation_axes

		if aggregate_axes is None:
			return array

		axes_tuple = tuple(np.atleast_1d(aggregate_axes))
		return np.nanmean(array, axis=axes_tuple, keepdims=False)

	# ------------------------------------------------------------------
	# Full pipeline
	# ------------------------------------------------------------------
	def prepare(
		self,
		array: np.ndarray,
		selections: Optional[Iterable[Selection]] = None,
		aggregation_axes: Optional[Union[int, Sequence[int]]] = None,
		flatten: Optional[bool] = None,
	) -> np.ndarray:
		selected = self.apply_selection(array, selections)
		aggregated = self.aggregate(selected, aggregation_axes)

		do_flatten = self.flatten if flatten is None else flatten
		if do_flatten:
			return aggregated.reshape(-1)
		return aggregated

	# ------------------------------------------------------------------
	# Convenience constructors
	# ------------------------------------------------------------------
	@classmethod
	def from_config(cls, config: dict) -> "ArrayReducer":
		"""Instantiate from a dict (e.g., parsed YAML)."""

		selections_cfg = config.get("select", [])
		selections = [
			Selection(axis=sel["axis"], indices=sel.get("indices", sel.get("index")))
			for sel in selections_cfg
		]

		aggregation_axes = config.get("aggregate")
		flatten = bool(config.get("flatten", True))
		return cls(selections, aggregation_axes, flatten)

