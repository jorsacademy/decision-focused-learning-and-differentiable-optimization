import pytest

from trained_ml_opt import BoxBudgetProblem


def test_problem_rejects_infeasible_lower_bounds() -> None:
    with pytest.raises(ValueError, match="lower bounds"):
        BoxBudgetProblem.from_arrays([1.0, 1.0], [2.0, 2.0], [1.0, 1.0], 1.5)


def test_problem_dimension() -> None:
    problem = BoxBudgetProblem.from_arrays([0, 0, 0], [1, 2, 3], [1, 1, 1], 4)
    assert problem.n_features == 3
