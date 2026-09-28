# Research Notes

## Position in the research series

This project covers the complementary direction to decision-focused learning and differentiable optimization: a **frozen trained predictor becomes part of the mathematical optimization formulation**.

- predict-then-optimize: predictions parameterize an optimizer;
- decision-focused learning: downstream optimization affects the training loss;
- differentiable optimization: an optimization layer participates in gradient-based training;
- optimization over trained ML: a fixed predictor is translated into algebraic constraints and optimized over its inputs.

## Formulation boundary

The native tree formulation deliberately targets one scikit-learn regression tree so its exactness argument remains transparent. Tree ensembles require one leaf-selection system per constituent tree and a coupled ensemble output; independently optimizing trees would be incorrect.

## Extrapolation risk

Optimization over a learned surrogate can amplify model error because the optimizer actively seeks favorable regions. Production use should therefore consider support diagnostics, domain restrictions, uncertainty-aware objectives, robust formulations, and independent validation. Restricting decisions to the declared input box is necessary here but is not claimed to solve distribution shift.
