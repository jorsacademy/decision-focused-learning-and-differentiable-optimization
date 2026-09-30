"""Decision-driven regularization benchmark for contextual newsvendor decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize


def newsvendor_cost(
    order: ArrayLike,
    demand: ArrayLike,
    *,
    underage: float,
    overage: float,
) -> NDArray[np.float64]:
    q = np.asarray(order, dtype=float)
    y = np.asarray(demand, dtype=float)
    return underage * np.maximum(y - q, 0.0) + overage * np.maximum(q - y, 0.0)


@dataclass(frozen=True)
class DDRLinearModel:
    coefficients: NDArray[np.float64]
    intercept: float
    prediction_weight: float

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        matrix = np.asarray(x, dtype=float)
        return self.intercept + matrix @ self.coefficients

    def decision(self, x: ArrayLike) -> NDArray[np.float64]:
        return self.predict(x)


def fit_ddr_linear(
    x: ArrayLike,
    demand: ArrayLike,
    *,
    underage: float = 3.0,
    overage: float = 1.0,
    prediction_weight: float = 0.5,
    l2: float = 1e-4,
) -> DDRLinearModel:
    """Fit a linear model with a bi-objective prediction/decision criterion.

    prediction_weight=1 gives an OLS-style prediction model. Smaller values put
    progressively more emphasis on downstream asymmetric decision cost.
    """

    matrix = np.asarray(x, dtype=float)
    y = np.asarray(demand, dtype=float)
    if matrix.ndim != 2 or y.ndim != 1 or matrix.shape[0] != y.size or y.size < 4:
        raise ValueError("x/demand shapes are invalid")
    if not 0.0 <= prediction_weight <= 1.0:
        raise ValueError("prediction_weight must lie in [0, 1]")
    if underage <= 0.0 or overage <= 0.0:
        raise ValueError("costs must be positive")

    design = np.column_stack([np.ones(y.size), matrix])
    initial, *_ = np.linalg.lstsq(design, y, rcond=None)
    mse_scale = max(float(np.var(y)), 1e-8)
    decision_scale = max(float(np.std(y)) * (underage + overage), 1e-8)

    def objective(beta: NDArray[np.float64]) -> float:
        prediction = design @ beta
        mse = float(np.mean((prediction - y) ** 2)) / mse_scale
        cost = float(
            np.mean(newsvendor_cost(prediction, y, underage=underage, overage=overage))
        ) / decision_scale
        penalty = l2 * float(np.dot(beta[1:], beta[1:]))
        return prediction_weight * mse + (1.0 - prediction_weight) * cost + penalty

    result = minimize(objective, initial, method="Powell")
    if not result.success:
        raise RuntimeError(f"DDR fit failed: {result.message}")
    beta = np.asarray(result.x, dtype=float)
    return DDRLinearModel(
        coefficients=beta[1:],
        intercept=float(beta[0]),
        prediction_weight=float(prediction_weight),
    )
