# Optimization over Trained ML Models

A research-oriented implementation of **ML-to-optimization embedding**: train a predictive model first, then place that fitted model *inside* a mathematical optimization problem so the optimizer chooses inputs while respecting operational constraints.

This direction is different from decision-focused learning. Decision-focused learning changes how a predictor is trained using downstream decision quality. Here the predictor is already trained and fixed; the optimization model treats its prediction function as part of the objective or feasible system.

## Research question

> Given a trained predictor `y = f(x)`, how can controllable inputs `x` be optimized under explicit resource constraints while preserving agreement between the optimization formulation and the original predictor?

The initial release keeps the formulation auditable:

- `LinearRegression` is embedded as an LP;
- `DecisionTreeRegressor` is embedded as an exact leaf-selection MILP over a declared input box;
- an optional Gurobi Machine Learning adapter uses `gurobi_ml.add_predictor_constr`;
- open-source CI uses SciPy/HiGHS, so correctness tests need no commercial solver.

## Core optimization problem

```text
maximize      fitted_predictor(x)
subject to    lower <= x <= upper
              resource_weights @ x <= budget
```

The fitted predictor is translated into algebraic constraints rather than used only as a post-solve scoring function.

## Linear-model embedding

For

```text
f(x) = beta_0 + beta^T x
```

the model remains linear. The implementation solves it with `scipy.optimize.milp`/HiGHS and checks the optimizer objective against `sklearn.predict`.

## Decision-tree embedding

Every feasible regression-tree leaf receives a binary variable `z_l` with

```text
sum_l z_l = 1.
```

Each root-to-leaf path defines a feature box. With one-hot leaf selection the formulation imposes

```text
sum_l lower[l,j] z_l <= x_j <= sum_l upper[l,j] z_l
```

for every feature `j`, and the embedded prediction is

```text
y = sum_l leaf_value[l] z_l.
```

A small epsilon represents strict right-branch inequalities.

## Optional Gurobi Machine Learning backend

Install the optional dependencies and configure a valid Gurobi license:

```bash
pip install -e ".[gurobi]"
```

The adapter follows the official interface:

```python
from gurobi_ml import add_predictor_constr

predictor_constr = add_predictor_constr(model, fitted_predictor, x, y)
```

This path is intended for predictor families supported by the installed `gurobi-machinelearning` version. It is isolated from the open-source CI path.

## Synthetic benchmark

The included experiment generates historical observations from a nonlinear three-input response surface, fits a linear model and a shallow regression tree, and optimizes each fitted predictor under the same weighted resource budget.

```bash
python -m pip install -e ".[dev]"
pytest
python scripts/run_benchmark.py
```

The benchmark tests **train -> freeze -> embed -> optimize -> verify**. It is not presented as evidence that a synthetic predictor is causally valid or production-ready.

## Validation contract

The project separates three questions:

1. **Predictor fit:** does the fitted model represent the underlying process?
2. **Embedding fidelity:** does the algebraic formulation reproduce that fitted predictor?
3. **Optimization quality:** does the solver find the best feasible decision for the embedded predictor?

The current tests target (2) and (3). In particular, the tree MILP is compared with brute-force enumeration on a dense feasible grid.

## Current scope

Implemented in v0.1:

- single-output scikit-learn linear regression;
- single-output scikit-learn decision-tree regression;
- continuous decisions with box bounds and one linear resource budget;
- optional generic Gurobi ML adapter;
- embedding-fidelity and optimization tests.

Next extensions:

- Random Forest / Gradient Boosting / XGBoost formulations;
- ReLU MLP embedding;
- OMLT + Pyomo / ONNX path;
- multiple operational constraints;
- learned models used as constraints, not only objectives;
- formulation-size and bound-tightening benchmarks;
- explicit support / out-of-distribution guards.

## References

- Gurobi Machine Learning documentation for `add_predictor_constr` and supported predictor formulations.
- OMLT (Optimization & Machine Learning Toolkit) for representing neural networks and gradient-boosted trees inside Pyomo models.
- Mixed-integer formulation literature for learned predictors and neural networks.

## License

This project inherits the umbrella repository's non-commercial/source-available licensing terms.
