"""Optional Gurobi Machine Learning backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from .embedding import OptimizationResult
from .problem import BoxBudgetProblem


def optimize_with_gurobi_ml(predictor: Any, problem: BoxBudgetProblem) -> OptimizationResult:
    """Maximize any predictor supported by gurobi-machinelearning."""

    try:
        from gurobi_ml import add_predictor_constr
        import gurobipy as gp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Install the optional 'gurobi' dependencies and configure a valid Gurobi license"
        ) from exc

    model = gp.Model("trained-ml-optimization")
    model.Params.OutputFlag = 0
    x = model.addMVar(problem.n_features, lb=problem.lower, ub=problem.upper, name="x")
    y = model.addMVar(1, lb=-gp.GRB.INFINITY, name="prediction")
    model.addConstr(problem.budget_weights @ x <= problem.budget, name="resource_budget")
    predictor_constr = add_predictor_constr(model, predictor, x, y)
    model.setObjective(y[0], gp.GRB.MAXIMIZE)
    model.optimize()

    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not prove optimality; status={model.Status}")

    solution = np.asarray(x.X, dtype=float)
    prediction = float(np.asarray(predictor.predict(solution.reshape(1, -1))).reshape(-1)[0])
    embedded = float(y.X[0])
    if not np.isclose(prediction, embedded, atol=1e-6, rtol=1e-6):
        error = np.asarray(predictor_constr.get_error(), dtype=float)
        raise RuntimeError(f"predictor embedding mismatch; max_error={np.max(np.abs(error))}")

    return OptimizationResult(
        x=solution,
        predicted_value=prediction,
        solver_objective=float(model.ObjVal),
        status=int(model.Status),
        message="Gurobi optimal solution",
    )
