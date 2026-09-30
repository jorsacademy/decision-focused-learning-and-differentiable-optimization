# Stochastic Optimization Forests

A transparent small-scale implementation inspired by Kallus and Mao's **Stochastic Optimization Forests**.

Unlike a standard regression tree, candidate splits are scored by **downstream newsvendor cost**. For every candidate split, the left and right child decisions are re-optimized using their empirical demand distributions, and the split with the lowest total operational loss is selected.

The forest bootstraps these optimization-aware trees. At prediction time, matching leaves induce contextual weights over historical demand observations; the final decision is the weighted empirical newsvendor quantile.

## Why this is different from Random Forest prediction

The tree is not trained to minimize squared prediction error. Its split objective is the stochastic optimization loss itself.

## Scope

This implementation deliberately uses exact re-optimization for each candidate split to make the prescriptive criterion inspectable. The paper's scalable perturbation-analysis approximations are not reproduced here and are a natural next extension.

## Reference

Nathan Kallus and Xiaojie Mao, **Stochastic Optimization Forests**, Management Science 69(4):1975–1994.
