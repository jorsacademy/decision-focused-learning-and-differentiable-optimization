"""Small contextual newsvendor laboratory for integrated conditional estimation-optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass(frozen=True)
class LinearConditionalNormal:
    weights: NDArray[np.float64]
    sigma: float

    def mean(self, x: ArrayLike) -> NDArray[np.float64]:
        matrix = np.asarray(x, dtype=float)
        return matrix @ self.weights

    def decision(self, x: ArrayLike, underage: float, overage: float) -> NDArray[np.float64]:
        critical = underage / (underage + overage)
        return self.mean(x) + self.sigma * norm.ppf(critical)


def newsvendor_cost(q: ArrayLike, demand: ArrayLike, underage: float, overage: float) -> float:
    qv = np.asarray(q, dtype=float)
    d = np.asarray(demand, dtype=float)
    return float(np.mean(underage * np.maximum(d - qv, 0.0) + overage * np.maximum(qv - d, 0.0)))


def fit_mse(x: ArrayLike, demand: ArrayLike, sigma: float) -> LinearConditionalNormal:
    matrix = np.asarray(x, dtype=float)
    y = np.asarray(demand, dtype=float)
    weights, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    return LinearConditionalNormal(weights=np.asarray(weights), sigma=float(sigma))


def fit_iceo(
    x: ArrayLike,
    demand: ArrayLike,
    *,
    sigma: float,
    underage: float,
    overage: float,
    l2: float = 1e-4,
) -> LinearConditionalNormal:
    """Fit conditional mean parameters directly through downstream newsvendor cost."""

    matrix = np.asarray(x, dtype=float)
    y = np.asarray(demand, dtype=float)
    start = fit_mse(matrix, y, sigma).weights

    def objective(weights: NDArray[np.float64]) -> float:
        model = LinearConditionalNormal(weights=weights, sigma=sigma)
        q = model.decision(matrix, underage, overage)
        return newsvendor_cost(q, y, underage, overage) + l2 * float(weights @ weights)

    result = minimize(objective, start, method="Powell")
    if not result.success:
        raise RuntimeError(result.message)
    return LinearConditionalNormal(np.asarray(result.x, dtype=float), float(sigma))
