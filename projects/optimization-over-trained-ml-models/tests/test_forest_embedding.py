import itertools

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from trained_ml_opt import BoxBudgetProblem, optimize_random_forest_regressor


def test_forest_embedding_matches_dense_grid() -> None:
    rng = np.random.default_rng(41)
    x_train = rng.uniform(0.0, 1.0, size=(900, 2))
    y_train = (
        2.0 * (x_train[:, 0] > 0.35)
        + 1.5 * (x_train[:, 1] > 0.55)
        + 0.7 * (x_train[:, 0] + x_train[:, 1] > 1.15)
    )
    model = RandomForestRegressor(
        n_estimators=4,
        max_depth=3,
        min_samples_leaf=20,
        random_state=7,
    ).fit(x_train, y_train)
    problem = BoxBudgetProblem.from_arrays([0, 0], [1, 1], [1, 1], 1.2)

    result = optimize_random_forest_regressor(model, problem)

    grid = np.linspace(0.0, 1.0, 501)
    feasible = np.array(
        [[a, b] for a, b in itertools.product(grid, grid) if a + b <= 1.2 + 1e-12],
        dtype=float,
    )
    brute = float(np.max(model.predict(feasible)))

    assert result.x.sum() <= 1.2 + 1e-7
    assert np.isclose(result.predicted_value, result.solver_objective, atol=1e-7)
    assert result.predicted_value >= brute - 1e-7
