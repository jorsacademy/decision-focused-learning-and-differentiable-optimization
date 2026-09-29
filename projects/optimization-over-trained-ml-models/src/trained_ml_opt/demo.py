"""Reproducible synthetic experiment for trained-model optimization."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from .embedding import (
    optimize_linear_regressor,
    optimize_random_forest_regressor,
    optimize_tree_regressor,
)
from .problem import BoxBudgetProblem


def generate_training_data(seed: int = 7, samples: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """Create historical process observations with a nonlinear response surface."""

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(samples, 3))
    noise = rng.normal(0.0, 0.04, size=samples)
    y = (
        2.8 * x[:, 0]
        + 1.8 * x[:, 1]
        + 3.4 * np.minimum(x[:, 2], 0.55)
        - 1.6 * np.maximum(x[:, 0] - 0.72, 0.0) ** 2
        - 0.9 * x[:, 1] * x[:, 2]
        + noise
    )
    return x, y


def run_demo(seed: int = 7) -> dict[str, object]:
    x_train, y_train = generate_training_data(seed=seed)
    linear = LinearRegression().fit(x_train, y_train)
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=seed).fit(
        x_train, y_train
    )
    forest = RandomForestRegressor(
        n_estimators=5,
        max_depth=4,
        min_samples_leaf=20,
        random_state=seed,
    ).fit(x_train, y_train)

    problem = BoxBudgetProblem.from_arrays(
        lower=[0.0, 0.0, 0.0],
        upper=[1.0, 1.0, 1.0],
        budget_weights=[1.0, 1.2, 0.8],
        budget=1.65,
    )

    return {
        "linear": asdict(optimize_linear_regressor(linear, problem)),
        "tree": asdict(optimize_tree_regressor(tree, problem)),
        "random_forest": asdict(optimize_random_forest_regressor(forest, problem)),
    }
