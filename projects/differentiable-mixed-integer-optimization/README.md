# Differentiable Mixed-Integer Optimization

> **Status:** planned research scaffold. The project is registered in the portfolio, but no benchmark result or paper reproduction is claimed yet.

## Research question

How can a predictive model be trained end-to-end when the downstream decision is a mixed-integer program whose exact solution map is discrete, piecewise constant, and not directly differentiable?

This project is intended to bridge the existing convex differentiable-optimization work and the black-box discrete-differentiation work in this monorepo.

## Planned decision pipeline

```text
context / features
      |
      v
predictive model
      |
      v
predicted MILP coefficients
      |
      +---- exact MILP oracle --------------------+
      |                                           |
      +---- gradient / surrogate mechanism        |
      |                                           v
      +-------------------------------------- decision loss
                                                  |
                                                  v
                                            model update
```

The final downstream decision remains solver-grounded. Gradient estimators are training mechanisms, not feasibility mechanisms.

## Initial benchmark scope

The first benchmark should use a compact contextual binary MILP with an independently auditable exact oracle, such as binary packing or capacitated selection. The benchmark should remain small enough that every learned decision can be compared against the true optimum during development.

Planned comparisons:

1. **Predict-then-optimize baseline** — train coefficient predictions with MSE, then solve the exact MILP.
2. **LP-relaxation gradient** — differentiate through a continuous relaxation and evaluate the resulting integer decision.
3. **Black-box / perturbation differentiation** — estimate a useful gradient around the exact discrete solver.
4. **MIP-as-a-layer style surrogate** — train through a mixed-integer decision layer while preserving an exact solver at evaluation time.

## Evaluation contract

Report at least:

- downstream regret against the clairvoyant optimum;
- optimality gap of the final integer decision;
- feasibility rate;
- coefficient-prediction error;
- gradient variance / stability;
- training cost and downstream solve time;
- sensitivity to integrality gap;
- held-out size and distribution shift.

A method should not be promoted merely because its training loss decreases. The primary question is whether the gradient mechanism improves integer decision quality.

## Relationship to neighboring projects

- `differentiable-optimization-pytorch`: continuous/QP differentiation foundation.
- `differentiable-optimization-portfolio`: applied convex optimization layer.
- `differentiable-black-box-supplier-selection-pytorch`: discrete black-box differentiation.
- `decision-focused-learning-spo`: decision-aware learning without requiring a differentiable exact MILP map.

This project is distinct because the downstream optimization problem is explicitly mixed-integer and the research object is the training signal through that discrete optimization layer.

## Planned milestones

1. Define a compact contextual MILP benchmark and exact oracle.
2. Implement matched predict-then-optimize and LP-relaxation baselines.
3. Implement one black-box/perturbation gradient mechanism.
4. Implement a MIP-as-a-layer style training surrogate.
5. Add repeated-seed experiments, OOD blocks, and gradient diagnostics.
6. Add CI tests for feasibility, exact-oracle agreement, and deterministic benchmark generation.

## Research lineage

The project is motivated by the **MIPaaL / MIP-as-a-Layer** line and by **Differentiation of Blackbox Combinatorial Solvers**. It is literature-informed; until an implementation explicitly states otherwise, it should not be described as a reproduction of either system.
