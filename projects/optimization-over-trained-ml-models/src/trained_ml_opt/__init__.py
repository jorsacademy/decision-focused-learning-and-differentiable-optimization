"""Optimization over trained machine-learning predictors."""

from .embedding import OptimizationResult, optimize_linear_regressor, optimize_tree_regressor
from .problem import BoxBudgetProblem

__all__ = [
    "BoxBudgetProblem",
    "OptimizationResult",
    "optimize_linear_regressor",
    "optimize_tree_regressor",
]
