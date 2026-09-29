# Decision-Driven Regularization

A small benchmark for blending prediction fit with downstream asymmetric decision cost.

A linear model is trained with

```text
(1-alpha) * normalized MSE
+ alpha * normalized newsvendor cost
+ ridge penalty.
```

The purpose is to expose the trade-off between predictive accuracy and operational loss as the regularization weight changes. This is an independent benchmark inspired by the 2026 decision-driven-regularization direction in the supplied technology review; it does not claim paper-level reproduction.

The central evaluation rule is simple: report both prediction error and held-out operational cost. A method is not considered better merely because its MSE is lower.
