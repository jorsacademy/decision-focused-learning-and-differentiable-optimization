"""Optimization-aware trees and forests for contextual newsvendor decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


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


def _weighted_quantile(values: NDArray[np.float64], weights: NDArray[np.float64], q: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = q * cumulative[-1]
    return float(sorted_values[min(int(np.searchsorted(cumulative, cutoff, side="left")), values.size - 1)])


def _optimal_order(y: NDArray[np.float64], underage: float, overage: float) -> float:
    alpha = underage / (underage + overage)
    return float(np.quantile(y, alpha, method="higher"))


def _leaf_risk(y: NDArray[np.float64], underage: float, overage: float) -> float:
    q = _optimal_order(y, underage, overage)
    return float(np.sum(newsvendor_cost(q, y, underage=underage, overage=overage)))


@dataclass
class _Node:
    indices: NDArray[np.int64]
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None


class OptimizationTree:
    """Tree whose split criterion directly minimizes downstream newsvendor loss."""

    def __init__(
        self,
        *,
        max_depth: int = 3,
        min_leaf: int = 20,
        underage: float = 3.0,
        overage: float = 1.0,
        max_thresholds: int = 16,
    ) -> None:
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.underage = underage
        self.overage = overage
        self.max_thresholds = max_thresholds
        self.root_: _Node | None = None
        self.x_: NDArray[np.float64] | None = None
        self.y_: NDArray[np.float64] | None = None

    def fit(self, x: ArrayLike, demand: ArrayLike) -> "OptimizationTree":
        matrix = np.asarray(x, dtype=float)
        y = np.asarray(demand, dtype=float)
        if matrix.ndim != 2 or y.ndim != 1 or matrix.shape[0] != y.size:
            raise ValueError("x/demand shapes are invalid")
        if y.size < 2 * self.min_leaf:
            raise ValueError("not enough samples for the requested min_leaf")
        self.x_, self.y_ = matrix, y
        self.root_ = self._grow(np.arange(y.size, dtype=np.int64), depth=0)
        return self

    def _grow(self, indices: NDArray[np.int64], depth: int) -> _Node:
        node = _Node(indices=indices)
        if depth >= self.max_depth or indices.size < 2 * self.min_leaf:
            return node

        assert self.x_ is not None and self.y_ is not None
        current_risk = _leaf_risk(
            self.y_[indices], self.underage, self.overage
        )
        best: tuple[float, int, float, NDArray[np.int64], NDArray[np.int64]] | None = None

        for feature in range(self.x_.shape[1]):
            values = self.x_[indices, feature]
            quantiles = np.linspace(0.05, 0.95, self.max_thresholds)
            thresholds = np.unique(np.quantile(values, quantiles))
            for threshold in thresholds:
                left = indices[values <= threshold]
                right = indices[values > threshold]
                if left.size < self.min_leaf or right.size < self.min_leaf:
                    continue
                risk = _leaf_risk(self.y_[left], self.underage, self.overage)
                risk += _leaf_risk(self.y_[right], self.underage, self.overage)
                if best is None or risk < best[0] - 1e-12:
                    best = (risk, feature, float(threshold), left, right)

        if best is None or best[0] >= current_risk - 1e-12:
            return node

        _, node.feature, node.threshold, left_idx, right_idx = best
        node.left = self._grow(left_idx, depth + 1)
        node.right = self._grow(right_idx, depth + 1)
        return node

    def leaf_indices(self, row: ArrayLike) -> NDArray[np.int64]:
        if self.root_ is None:
            raise RuntimeError("tree is not fitted")
        vector = np.asarray(row, dtype=float)
        node = self.root_
        while node.feature is not None:
            assert node.threshold is not None and node.left is not None and node.right is not None
            node = node.left if vector[node.feature] <= node.threshold else node.right
        return node.indices

    def decision(self, x: ArrayLike) -> NDArray[np.float64]:
        if self.y_ is None:
            raise RuntimeError("tree is not fitted")
        matrix = np.asarray(x, dtype=float)
        return np.asarray(
            [
                _optimal_order(self.y_[self.leaf_indices(row)], self.underage, self.overage)
                for row in matrix
            ],
            dtype=float,
        )


class OptimizationForest:
    """Bootstrap forest that aggregates contextual leaf weights before optimizing."""

    def __init__(
        self,
        *,
        n_estimators: int = 20,
        max_depth: int = 3,
        min_leaf: int = 20,
        underage: float = 3.0,
        overage: float = 1.0,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.underage = underage
        self.overage = overage
        self.random_state = random_state
        self.trees_: list[OptimizationTree] = []

    def fit(self, x: ArrayLike, demand: ArrayLike) -> "OptimizationForest":
        matrix = np.asarray(x, dtype=float)
        y = np.asarray(demand, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for _ in range(self.n_estimators):
            sample = rng.integers(0, y.size, size=y.size)
            tree = OptimizationTree(
                max_depth=self.max_depth,
                min_leaf=self.min_leaf,
                underage=self.underage,
                overage=self.overage,
            ).fit(matrix[sample], y[sample])
            self.trees_.append(tree)
        return self

    def decision(self, x: ArrayLike) -> NDArray[np.float64]:
        if not self.trees_:
            raise RuntimeError("forest is not fitted")
        matrix = np.asarray(x, dtype=float)
        alpha = self.underage / (self.underage + self.overage)
        decisions: list[float] = []
        for row in matrix:
            values: list[float] = []
            weights: list[float] = []
            for tree in self.trees_:
                assert tree.y_ is not None
                leaf = tree.leaf_indices(row)
                mass = 1.0 / (len(self.trees_) * leaf.size)
                values.extend(tree.y_[leaf].tolist())
                weights.extend([mass] * leaf.size)
            decisions.append(
                _weighted_quantile(
                    np.asarray(values, dtype=float),
                    np.asarray(weights, dtype=float),
                    alpha,
                )
            )
        return np.asarray(decisions, dtype=float)
