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

## Inverse and uncertainty-aware decision learning

| Repository | Main focus | Relationship |
|---|---|---|
| `inverse-optimization-shortest-path` | Recovering latent costs/preferences from observed optimal decisions | Inverse optimization foundation |
| `inverse-optimization-industrial-decisions` | Inverse optimization in industrial decision settings | Applied inverse optimization |
| `conformal-prediction-robust-inventory-optimization-python` | Turning calibrated predictive sets into robust operational decisions | Uncertainty-to-decision bridge |
| `wasserstein-dro-inventory-optimization-python` | Distributionally robust inventory optimization | Robust optimization companion |

## Why these repositories remain separate

SPO+ does not require differentiating through the optimizer in the same way as a differentiable convex layer. Inverse optimization reverses the direction of the problem by inferring objective parameters from decisions. Contextual optimization may learn a decision rule directly. Distributionally robust and conformal approaches focus on uncertainty sets or ambiguity rather than only end-to-end prediction loss.

The repositories are therefore related but methodologically distinct.

## Suggested reading order

1. `decision-focused-learning-spo`
2. `predict-then-optimize-production-planning-spo-plus-pytorch`
3. `differentiable-optimization-pytorch`
4. `differentiable-optimization-portfolio`
5. `differentiable-black-box-supplier-selection-pytorch`
6. `contextual-optimization-newsvendor`
7. `distributionally-robust-decision-focused-learning`
8. `inverse-optimization-shortest-path`
9. `inverse-optimization-industrial-decisions`

The ordering is pedagogical rather than a ranking of methods.

## Contextual optimization frontier

| Repository | Main focus | Role |
|---|---|---|
| `integrated-conditional-estimation-optimization` | Fit conditional uncertainty models through downstream stochastic decision quality | Integrated estimation-optimization |
| `decision-driven-regularization` | Blend predictive accuracy with downstream operational loss | 2026 learning/optimization regularization |
| `stochastic-optimization-forests` | Optimization-aware tree splitting for contextual stochastic decisions | Nonparametric prescriptive learning |

These projects extend the series beyond SPO+/differentiable layers into conditional-distribution learning and optimization-aware nonparametric policies.
