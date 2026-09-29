import numpy as np
from sklearn.linear_model import LinearRegression

from trained_ml_opt import BoxBudgetProblem, optimize_linear_regressor


def test_linear_embedding_hits_analytic_optimum() -> None:
    x_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    y_train = 3.0 * x_train[:, 0] + 2.0 * x_train[:, 1] + 4.0
    model = LinearRegression().fit(x_train, y_train)
    problem = BoxBudgetProblem.from_arrays([0, 0], [1, 1], [1, 1], 1.0)

    result = optimize_linear_regressor(model, problem)

    assert np.allclose(result.x, [1.0, 0.0], atol=1e-7)
    assert np.isclose(result.predicted_value, 7.0, atol=1e-7)
    assert np.isclose(result.predicted_value, result.solver_objective, atol=1e-7)
