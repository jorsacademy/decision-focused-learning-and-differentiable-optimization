# Decision-Driven Regularization

A small reproducible implementation of the 2026 **Decision-Driven Regularization** idea: balance predictive accuracy and downstream decision quality rather than optimizing either objective in isolation.

The benchmark uses contextual inventory with asymmetric shortage/overage cost. A hyperparameter `alpha` blends mean-squared prediction error with operational cost. The endpoints recover prediction-oriented fitting and decision-oriented fitting; intermediate values expose the regularization trade-off.

This is a pedagogical special case, not a reproduction of every robust/regret-equivalent formulation in the paper.

Reference: Loke, Tang, Xiao, Zhang (2026), *INFORMS Journal on Computing*, DOI 10.1287/ijoc.2024.0930.
