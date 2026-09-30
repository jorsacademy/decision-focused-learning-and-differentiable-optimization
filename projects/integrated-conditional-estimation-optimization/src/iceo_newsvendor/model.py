"""Compact integrated conditional estimation-optimization benchmark for newsvendor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.stats import norm


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
class GaussianICEO:
    """Linear conditional Gaussian model trained with an optimization-aware objective."""

    coefficients: NDArray[np.float64]
    intercept: float
    log_scale: float
    underage: float
    overage: float
    decision_weight: float

    @property
    def scale(self) -> float:
        return float(np.exp(self.log_scale))

    @property
    def critical_fractile(self) -> float:
        return float(self.underage / (self.underage + self.overage))

    def mean(self, x: ArrayLike) -> NDArray[np.float64]:
        matrix = np.asarray(x, dtype=float)
        return self.intercept + matrix @ self.coefficients

    def decision(self, x: ArrayLike) -> NDArray[np.float64]:
        return self.mean(x) + self.scale * norm.ppf(self.critical_fractile)


def fit_gaussian_iceo(
    x: ArrayLike,
    demand: ArrayLike,
    *,
    underage: float = 3.0,
    overage: float = 1.0,
    decision_weight: float = 0.5,
    l2: float = 1e-4,
) -> GaussianICEO:
    """Fit conditional Gaussian parameters using likelihood and downstream decision cost.

    decision_weight=0 is the separate estimate-then-optimize baseline. Positive values
    make the conditional estimator account for the downstream newsvendor objective.
    This is an independent compact ICEO-style benchmark, not a reproduction of the
    full polynomial-optimization framework in Qi, Grigas, and Shen.
    """

    matrix = np.asarray(x, dtype=float)
    y = np.asarray(demand, dtype=float)
    if matrix.ndim != 2 or y.ndim != 1 or matrix.shape[0] != y.size or y.size < 4:
        raise ValueError("x/demand shapes are invalid")
    if underage <= 0.0 or overage <= 0.0:
        raise ValueError("underage and overage must be positive")
    if not 0.0 <= decision_weight <= 1.0:
        raise ValueError("decision_weight must lie in [0, 1]")

    design = np.column_stack([np.ones(y.size), matrix])
    beta0, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta0
    init_scale = max(float(np.std(resid)), 1e-3)
    initial = np.concatenate([beta0, [np.log(init_scale)]])

    alpha = underage / (underage + overage)
    z_alpha = float(norm.ppf(alpha))
    cost_scale = max(float(np.std(y)) * (underage + overage), 1e-6)

    def objective(params: NDArray[np.float64]) -> float:
        beta = params[:-1]
        log_scale = float(np.clip(params[-1], -8.0, 8.0))
        scale = float(np.exp(log_scale))
        mu = design @ beta
        nll = 0.5 * np.mean(((y - mu) / scale) ** 2) + log_scale
        q = mu + scale * z_alpha
        decision_loss = float(np.mean(newsvendor_cost(q, y, underage=underage, overage=overage)))
        penalty = l2 * float(np.dot(beta[1:], beta[1:]))
        return (1.0 - decision_weight) * float(nll) + decision_weight * (
            decision_loss / cost_scale
        ) + penalty

    result = minimize(objective, initial, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"ICEO fit failed: {result.message}")

    params = np.asarray(result.x, dtype=float)
    return GaussianICEO(
        coefficients=params[1:-1],
        intercept=float(params[0]),
        log_scale=float(params[-1]),
        underage=float(underage),
        overage=float(overage),
        decision_weight=float(decision_weight),
    )
