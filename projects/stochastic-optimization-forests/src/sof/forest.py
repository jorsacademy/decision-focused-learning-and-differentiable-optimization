"""Optimization-aware forest for a contextual newsvendor problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def quantile_decision(y: NDArray[np.float64], underage: float, overage: float) -> float:
    critical = underage / (underage + overage)
    return float(np.quantile(y, critical))


def leaf_cost(y: NDArray[np.float64], underage: float, overage: float) -> float:
    q = quantile_decision(y, underage, overage)
    return float(np.sum(underage * np.maximum(y - q, 0.0) + overage * np.maximum(q - y, 0.0)))


@dataclass(frozen=True)
class Stump:
    feature: int
    threshold: float
    left_q: float
    right_q: float

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        matrix = np.asarray(x, dtype=float)
        return np.where(matrix[:, self.feature] <= self.threshold, self.left_q, self.right_q)


def fit_stump(x: ArrayLike, y: ArrayLike, underage: float, overage: float) -> Stump:
    matrix = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    best: tuple[float, int, float, float, float] | None = None

    for j in range(matrix.shape[1]):
        values = np.unique(matrix[:, j])
        thresholds = (values[:-1] + values[1:]) / 2.0
        for threshold in thresholds:
            left = matrix[:, j] <= threshold
            if left.sum() < 5 or (~left).sum() < 5:
                continue
            cost = leaf_cost(target[left], underage, overage) + leaf_cost(target[~left], underage, overage)
            candidate = (
                cost,
                j,
                float(threshold),
                quantile_decision(target[left], underage, overage),
                quantile_decision(target[~left], underage, overage),
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise ValueError("no valid split")
    return Stump(feature=best[1], threshold=best[2], left_q=best[3], right_q=best[4])


@dataclass
class OptimizationForest:
    trees: list[Stump]

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        predictions = np.stack([tree.predict(x) for tree in self.trees], axis=0)
        return np.mean(predictions, axis=0)


def fit_forest(
    x: ArrayLike,
    y: ArrayLike,
    *,
    n_trees: int = 25,
    underage: float = 4.0,
    overage: float = 1.0,
    seed: int = 0,
) -> OptimizationForest:
    matrix = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    trees: list[Stump] = []
    for _ in range(n_trees):
        idx = rng.integers(0, len(target), len(target))
        trees.append(fit_stump(matrix[idx], target[idx], underage, overage))
    return OptimizationForest(trees)
