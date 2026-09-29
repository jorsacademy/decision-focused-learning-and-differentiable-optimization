# Integrated Conditional Estimation-Optimization

A compact contextual-newsvendor implementation inspired by Qi, Grigas, and Shen's **Integrated Conditional Estimation-Optimization (ICEO)** framework.

The project contrasts:

- separate conditional estimation by least squares, followed by optimization;
- integrated fitting of the conditional model through the downstream newsvendor objective.

For a conditional Normal demand model, the newsvendor decision is analytic, so the experiment isolates the central ICEO idea without hiding it behind a large neural architecture.

The implementation is intentionally a transparent special case, not a reproduction of the complete paper. It includes deterministic tests showing that the integrated fit cannot be worse than its MSE initialization on the training decision objective.

Reference: Qi, Grigas, Shen, *Operations Research* 74(3), 2025, DOI 10.1287/opre.2023.0427.
