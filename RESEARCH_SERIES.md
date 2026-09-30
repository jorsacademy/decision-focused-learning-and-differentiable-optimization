# Decision-Focused and Differentiable Optimization Research Series

This repository is part of a broader set of independent projects connecting prediction, uncertainty, and optimization. The repositories remain separate when they use different downstream problems, differentiation mechanisms, or learning objectives.

## Decision-focused learning and predict-then-optimize

| Repository | Main focus | Role in the series |
|---|---|---|
| `decision-focused-learning-spo` | SPO/SPO+ on controlled combinatorial problems with exact downstream oracles | Foundational DFL laboratory |
| `predict-then-optimize-production-planning-spo-plus-pytorch` | SPO+ on multi-period production planning with an exact LP downstream problem | Applied PtO/DFL case study |
| `distributionally-robust-decision-focused-learning` | Decision-focused learning under distributional ambiguity | Robust DFL extension |
| `contextual-optimization-newsvendor` | Directly learning context-dependent operational decisions | Contextual optimization |

## Differentiable optimization

| Repository | Main focus | Role in the series |
|---|---|---|
| `differentiable-optimization-pytorch` | Differentiating through quadratic optimization, including a transparent PyTorch linear-solve layer and constrained cvxpylayers example | Tutorial/foundational differentiable optimization |
| `differentiable-optimization-portfolio` | Differentiable convex optimization for long-only mean-variance portfolio decisions | Applied convex-layer case study |
| `differentiable-black-box-supplier-selection-pytorch` | Gradient-based learning around a discrete/black-box downstream optimization problem | Black-box differentiation case study |
| `differentiable-mixed-integer-optimization` | Training through an explicitly mixed-integer downstream decision layer using relaxation, perturbation, and MIP-as-layer style signals | Planned discrete differentiable-optimization benchmark |

## Inverse and uncertainty-aware decision learning

| Repository | Main focus | Relationship |
|---|---|---|
| `inverse-optimization-shortest-path` | Recovering latent costs/preferences from observed optimal decisions | Inverse optimization foundation |
| `inverse-optimization-industrial-decisions` | Inverse optimization in industrial decision settings | Applied inverse optimization |
| `conformal-prediction-robust-inventory-optimization-python` | Turning calibrated predictive sets into robust operational decisions | Uncertainty-to-decision bridge |
| `wasserstein-dro-inventory-optimization-python` | Distributionally robust inventory optimization | Robust optimization companion |

## Why these repositories remain separate

SPO+ does not require differentiating through the optimizer in the same way as a differentiable convex layer. The mixed-integer layer project is separate from the convex-layer projects because integrality makes the exact solution map discrete and requires different gradient surrogates or estimators. Inverse optimization reverses the direction of the problem by inferring objective parameters from decisions. Contextual optimization may learn a decision rule directly. Distributionally robust and conformal approaches focus on uncertainty sets or ambiguity rather than only end-to-end prediction loss.

The repositories are therefore related but methodologically distinct.

## Suggested reading order

1. `decision-focused-learning-spo`
2. `predict-then-optimize-production-planning-spo-plus-pytorch`
3. `differentiable-optimization-pytorch`
4. `differentiable-optimization-portfolio`
5. `differentiable-black-box-supplier-selection-pytorch`
6. `differentiable-mixed-integer-optimization` *(planned)*
7. `contextual-optimization-newsvendor`
8. `distributionally-robust-decision-focused-learning`
9. `inverse-optimization-shortest-path`
10. `inverse-optimization-industrial-decisions`

The ordering is pedagogical rather than a ranking of methods.

## Contextual optimization frontier

| Repository | Main focus | Role |
|---|---|---|
| `integrated-conditional-estimation-optimization` | Conditional distribution estimation coupled to downstream newsvendor structure | ICEO-style estimation/optimization integration |
| `decision-driven-regularization` | Bi-objective prediction accuracy and decision-cost fitting | 2026 blended learning/optimization frontier |
| `stochastic-optimization-forests` | Trees split on downstream stochastic-optimization loss | Optimization-aware nonparametric contextual decisions |

These projects are compact independent implementations. They reproduce the central methodological distinctions while explicitly avoiding claims of full paper-level reproduction.
