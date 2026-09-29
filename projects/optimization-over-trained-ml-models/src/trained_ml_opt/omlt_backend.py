"""Optional OMLT/Pyomo formulation boundary for trained neural networks."""

from __future__ import annotations

from typing import Any

import numpy as np

from .problem import BoxBudgetProblem


def build_omlt_pyomo_model(network_definition: Any, problem: BoxBudgetProblem):
    """Build a Pyomo model containing an OMLT neural-network formulation.

    `network_definition` is an OMLT network definition, for example one loaded
    from ONNX. Solver selection remains outside this function.
    """
    try:
        import pyomo.environ as pyo
        from omlt import OmltBlock
        from omlt.neuralnet import FullSpaceNNFormulation
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the optional 'omlt' dependencies") from exc

    model=pyo.ConcreteModel()
    n=problem.n_features
    model.x=pyo.Var(range(n),bounds=lambda _,j:(float(problem.lower[j]),float(problem.upper[j])))
    model.nn=OmltBlock()
    model.nn.build_formulation(FullSpaceNNFormulation(network_definition))
    for j in range(n):
        model.add_component(
            f"link_input_{j}",
            pyo.Constraint(expr=model.nn.inputs[j] == model.x[j]),
        )
    model.resource_budget=pyo.Constraint(
        expr=sum(float(problem.budget_weights[j])*model.x[j] for j in range(n)) <= problem.budget
    )
    output_keys=list(model.nn.outputs.keys())
    if len(output_keys) != 1:
        raise ValueError("v0.2 OMLT adapter supports one scalar network output")
    model.objective=pyo.Objective(expr=model.nn.outputs[output_keys[0]],sense=pyo.maximize)
    return model


def verify_omlt_solution(predictor: Any, x: np.ndarray, embedded_value: float, atol: float=1e-5) -> None:
    """Check an externally solved OMLT model against the original predictor."""
    predicted=float(np.asarray(predictor.predict(np.asarray(x).reshape(1,-1))).reshape(-1)[0])
    if not np.isclose(predicted,embedded_value,atol=atol,rtol=atol):
        raise RuntimeError(f"OMLT embedding mismatch: predictor={predicted}, embedded={embedded_value}")
