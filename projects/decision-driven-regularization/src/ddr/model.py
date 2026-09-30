"""Decision-driven regularization in a transparent contextual inventory problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize


@dataclass(frozen=True)
class DDRModel:
    weights: NDArray[np.float64]

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(x, dtype=float) @ self.weights


def asymmetric_cost(q: ArrayLike, y: ArrayLike, underage: float, overage: float) -> float:
    qv = np.asarray(q, dtype=float)
    demand = np.asarray(y, dtype=float)
    return float(np.mean(underage * np.maximum(demand - qv, 0.0) + overage * np.maximum(qv - demand, 0.0)))


def fit_ddr(
    x: ArrayLike,
    y: ArrayLike,
    *,
    alpha: float,
    underage: float,
    overage: float,
    l2: float = 1e-4,
) -> DDRModel:
    """Blend predictive squared error and downstream asymmetric decision cost."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    matrix = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    start, *_ = np.linalg.lstsq(matrix, target, rcond=None)

    def objective(w: NDArray[np.float64]) -> float:
        pred = matrix @ w
        mse = float(np.mean((pred - target) ** 2))
        decision = asymmetric_cost(pred, target, underage, overage)
        return alpha * mse + (1.0 - alpha) * decision + l2 * float(w @ w)

    result = minimize(objective, np.asarray(start), method="Powell")
    if not result.success:
        raise RuntimeError(result.message)
    return DDRModel(np.asarray(result.x, dtype=float))
