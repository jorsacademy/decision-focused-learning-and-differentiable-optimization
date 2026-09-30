# Stochastic Optimization Forests

A compact optimization-aware forest inspired by Kallus and Mao's **Stochastic Optimization Forests**.

Unlike a standard random forest that chooses splits to improve prediction error, this project chooses splits by **downstream newsvendor cost**. Each leaf stores the cost-optimal empirical quantile decision. Bootstrap aggregation produces a small contextual decision forest.

The implementation uses exact re-optimization of candidate stump splits for transparency. It does not yet implement the paper's perturbation-based approximate split criterion or asymptotic theory.

Reference: Kallus & Mao, *Management Science* 69(4):1975–1994.
