"""Exact optimization formulations for selected trained scikit-learn regressors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from .problem import BoxBudgetProblem


@dataclass(frozen=True)
class OptimizationResult:
    """Solution returned by an embedded-predictor optimization model."""

    x: NDArray[np.float64]
    predicted_value: float
    solver_objective: float
    status: int
    message: str


def _validate_predictor_dimension(n_features: int, predictor_features: int) -> None:
    if predictor_features != n_features:
        raise ValueError(
            f"predictor expects {predictor_features} features but problem has {n_features}"
        )


def optimize_linear_regressor(
    model: LinearRegression,
    problem: BoxBudgetProblem,
) -> OptimizationResult:
    """Maximize a fitted single-output linear regressor over the operational polytope."""

    coef = np.asarray(model.coef_, dtype=float)
    if coef.ndim != 1:
        raise ValueError("only single-output LinearRegression models are supported")
    _validate_predictor_dimension(problem.n_features, coef.size)

    objective = -coef
    budget_constraint = LinearConstraint(
        problem.budget_weights.reshape(1, -1),
        lb=-np.inf,
        ub=np.array([problem.budget], dtype=float),
    )
    result = milp(
        c=objective,
        integrality=np.zeros(problem.n_features, dtype=int),
        bounds=Bounds(problem.lower, problem.upper),
        constraints=budget_constraint,
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP solve failed: {result.message}")

    x = np.asarray(result.x, dtype=float)
    predicted = float(model.predict(x.reshape(1, -1))[0])
    return OptimizationResult(
        x=x,
        predicted_value=predicted,
        solver_objective=float(-result.fun + float(model.intercept_)),
        status=int(result.status),
        message=str(result.message),
    )


def _leaf_regions(
    model: DecisionTreeRegressor,
    problem: BoxBudgetProblem,
    epsilon: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return per-leaf box regions and prediction values for a fitted decision tree."""

    tree = model.tree_
    regions: list[tuple[np.ndarray, np.ndarray, float]] = []

    def visit(node: int, lower: np.ndarray, upper: np.ndarray) -> None:
        feature = int(tree.feature[node])
        if feature < 0:
            value = float(np.asarray(tree.value[node]).reshape(-1)[0])
            if np.all(lower <= upper + 1e-12):
                regions.append((lower.copy(), upper.copy(), value))
            return

        threshold = float(tree.threshold[node])

        left_upper = upper.copy()
        left_upper[feature] = min(left_upper[feature], threshold)
        if lower[feature] <= left_upper[feature] + 1e-12:
            visit(int(tree.children_left[node]), lower.copy(), left_upper)

        right_lower = lower.copy()
        right_lower[feature] = max(right_lower[feature], threshold + epsilon)
        if right_lower[feature] <= upper[feature] + 1e-12:
            visit(int(tree.children_right[node]), right_lower, upper.copy())

    visit(0, problem.lower.copy(), problem.upper.copy())
    if not regions:
        raise ValueError("no tree leaf intersects the optimization box")

    lowers = np.stack([r[0] for r in regions])
    uppers = np.stack([r[1] for r in regions])
    values = np.asarray([r[2] for r in regions], dtype=float)
    return lowers, uppers, values


def optimize_tree_regressor(
    model: DecisionTreeRegressor,
    problem: BoxBudgetProblem,
    *,
    epsilon: float = 1e-7,
) -> OptimizationResult:
    """Maximize a fitted regression tree with an exact leaf-selection MILP."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    _validate_predictor_dimension(problem.n_features, int(model.n_features_in_))

    leaf_lower, leaf_upper, leaf_value = _leaf_regions(model, problem, epsilon)
    n = problem.n_features
    leaves = leaf_value.size
    total_vars = n + leaves

    c = np.concatenate([np.zeros(n), -leaf_value])
    integrality = np.concatenate([np.zeros(n, dtype=int), np.ones(leaves, dtype=int)])
    lb = np.concatenate([problem.lower, np.zeros(leaves)])
    ub = np.concatenate([problem.upper, np.ones(leaves)])

    rows: list[np.ndarray] = []
    row_lb: list[float] = []
    row_ub: list[float] = []

    one_leaf = np.zeros(total_vars)
    one_leaf[n:] = 1.0
    rows.append(one_leaf)
    row_lb.append(1.0)
    row_ub.append(1.0)

    budget_row = np.zeros(total_vars)
    budget_row[:n] = problem.budget_weights
    rows.append(budget_row)
    row_lb.append(-np.inf)
    row_ub.append(problem.budget)

    for j in range(n):
        upper_row = np.zeros(total_vars)
        upper_row[j] = 1.0
        upper_row[n:] = -leaf_upper[:, j]
        rows.append(upper_row)
        row_lb.append(-np.inf)
        row_ub.append(0.0)

        lower_row = np.zeros(total_vars)
        lower_row[j] = -1.0
        lower_row[n:] = leaf_lower[:, j]
        rows.append(lower_row)
        row_lb.append(-np.inf)
        row_ub.append(0.0)

    constraints = LinearConstraint(
        np.stack(rows),
        lb=np.asarray(row_lb, dtype=float),
        ub=np.asarray(row_ub, dtype=float),
    )
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP solve failed: {result.message}")

    x = np.asarray(result.x[:n], dtype=float)
    embedded_value = float(-result.fun)
    predictor_value = float(model.predict(x.reshape(1, -1))[0])
    if not np.isclose(predictor_value, embedded_value, atol=1e-6, rtol=1e-6):
        raise RuntimeError(
            "embedded tree prediction disagrees with sklearn at the optimizer solution: "
            f"embedded={embedded_value}, sklearn={predictor_value}"
        )

    return OptimizationResult(
        x=x,
        predicted_value=predictor_value,
        solver_objective=embedded_value,
        status=int(result.status),
        message=str(result.message),
    )
