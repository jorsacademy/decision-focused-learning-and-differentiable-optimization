# Integrated Conditional Estimation-Optimization

A compact contextual newsvendor benchmark inspired by Qi, Grigas, and Shen's **Integrated Conditional Estimation-Optimization (ICEO)** framework.

The project estimates a conditional Gaussian demand distribution while explicitly including the downstream newsvendor cost in the fitting objective. Setting the decision weight to zero recovers a conventional estimate-then-optimize baseline; positive weights couple conditional estimation and optimization.

This is an independent small research implementation, **not** a reproduction of the paper's full polynomial/semi-algebraic solution framework.

## Model

For context x, demand is modeled as

`D | x ~ Normal(mu_beta(x), sigma^2)`.

The downstream order quantity is the conditional newsvendor quantile

`q(x) = mu_beta(x) + sigma Phi^{-1}(cu/(cu+co))`.

Training minimizes a blended criterion containing Gaussian negative log likelihood and realized downstream newsvendor cost. The benchmark therefore asks whether changing the estimator to respect optimization structure changes held-out decision quality.

## Validation boundary

- train/test samples are separated;
- decision quality is evaluated with asymmetric underage/overage cost;
- no claim is made that this simplified objective is identical to the full ICEO estimator;
- the goal is to make the estimation-vs-decision coupling explicit and auditable.

## Reference

Meng Qi, Paul Grigas, Zuo-Jun (Max) Shen, **Integrated Conditional Estimation-Optimization**, Operations Research, published online 2025.
