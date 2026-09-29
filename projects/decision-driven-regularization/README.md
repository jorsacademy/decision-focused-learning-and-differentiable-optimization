# Decision-Driven Regularization

A compact benchmark inspired by Loke, Tang, Xiao, and Zhang's 2026 **Decision-Driven Regularization** framework.

The project fits a contextual linear demand model using a bi-objective criterion that blends:

- prediction error; and
- downstream asymmetric newsvendor cost.

The mixing parameter makes the prediction-vs-decision trade-off explicit. A prediction weight of 1 is an OLS-style baseline; lower values increasingly regularize the model toward decision quality.

This implementation is intentionally small and independent. It demonstrates the central blended-objective idea but does not claim byte-for-byte reproduction of the paper's complete surrogate, robust, or regret formulations.

## Evaluation

Every fitted model is evaluated on held-out data with both MSE and operational newsvendor cost. The point is not to declare one universal mixing weight, but to expose when improved predictive fit and improved operational decisions diverge.

## Reference

Gar Goei Loke, Qinshen Tang, Yangge Xiao, Xun Zhang, **Decision-Driven Regularization: A Blended Model for Learning and Optimization**, INFORMS Journal on Computing, 2026.
