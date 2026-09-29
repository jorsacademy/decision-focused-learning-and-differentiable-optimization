"""Problem definitions shared by predictor-embedding backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class BoxBudgetProblem:
    """Continuous decision variables with box bounds and one resource budget."""

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    budget_weights: NDArray[np.float64]
    budget: float

    @classmethod
    def from_arrays(
        cls,
        lower: ArrayLike,
        upper: ArrayLike,
        budget_weights: ArrayLike,
        budget: float,
    ) -> BoxBudgetProblem:
        lower_arr = np.asarray(lower, dtype=float)
        upper_arr = np.asarray(upper, dtype=float)
        weights_arr = np.asarray(budget_weights, dtype=float)

        if lower_arr.ndim != 1 or upper_arr.ndim != 1 or weights_arr.ndim != 1:
            raise ValueError("lower, upper, and budget_weights must be one-dimensional")
        if not (lower_arr.shape == upper_arr.shape == weights_arr.shape):
            raise ValueError("lower, upper, and budget_weights must have equal length")
        if lower_arr.size == 0:
            raise ValueError("at least one decision variable is required")
        if np.any(~np.isfinite(lower_arr)) or np.any(~np.isfinite(upper_arr)):
            raise ValueError("box bounds must be finite")
        if np.any(lower_arr > upper_arr):
            raise ValueError("every lower bound must be <= its upper bound")
        if np.any(weights_arr < 0.0) or np.any(~np.isfinite(weights_arr)):
            raise ValueError("budget weights must be finite and non-negative")
        if not np.isfinite(budget):
            raise ValueError("budget must be finite")
        if float(weights_arr @ lower_arr) > float(budget) + 1e-12:
            raise ValueError("the lower bounds already violate the resource budget")

        return cls(lower_arr, upper_arr, weights_arr, float(budget))

    @property
    def n_features(self) -> int:
        return int(self.lower.size)
