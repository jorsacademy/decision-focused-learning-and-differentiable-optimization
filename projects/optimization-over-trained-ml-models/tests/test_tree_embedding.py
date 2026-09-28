import itertools

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from trained_ml_opt import BoxBudgetProblem, optimize_tree_regressor


def test_tree_embedding_matches_bruteforce_grid() -> None:
    rng = np.random.default_rng(3)
    x_train = rng.uniform(0.0, 1.0, size=(500, 2))
    y_train = 1.5 * (x_train[:, 0] > 0.55) + 2.0 * (x_train[:, 1] > 0.35)
    model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=15, random_state=0).fit(
        x_train, y_train
    )
    problem = BoxBudgetProblem.from_arrays([0, 0], [1, 1], [1, 1], 1.25)

    result = optimize_tree_regressor(model, problem)

    grid = np.linspace(0.0, 1.0, 401)
    feasible = np.array(
        [[a, b] for a, b in itertools.product(grid, grid) if a + b <= 1.25 + 1e-12],
        dtype=float,
    )
    brute_value = float(np.max(model.predict(feasible)))

    assert result.x.sum() <= 1.25 + 1e-7
    assert np.all(result.x >= -1e-9)
    assert np.all(result.x <= 1.0 + 1e-9)
    assert np.isclose(result.predicted_value, brute_value, atol=1e-7)
    assert np.isclose(result.predicted_value, result.solver_objective, atol=1e-7)


def test_tree_embedding_respects_box_intersection() -> None:
    x_train = np.array([[0.1], [0.2], [0.8], [0.9]], dtype=float)
    y_train = np.array([0.0, 0.0, 5.0, 5.0])
    model = DecisionTreeRegressor(max_depth=1, random_state=0).fit(x_train, y_train)
    problem = BoxBudgetProblem.from_arrays([0.0], [0.4], [1.0], 0.4)

    result = optimize_tree_regressor(model, problem)

    assert result.x[0] <= 0.4 + 1e-8
    assert np.isclose(result.predicted_value, 0.0)
