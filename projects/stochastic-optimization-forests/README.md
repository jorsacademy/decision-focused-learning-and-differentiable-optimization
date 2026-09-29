# Stochastic Optimization Forests

A contextual stochastic-optimization benchmark where a random forest is used to construct **local scenario weights**, not merely a point prediction.

For a new context, each tree assigns training observations to leaves. Training observations sharing leaves with the query receive proximity weights. The downstream newsvendor decision is then the weighted empirical critical fractile.

This project is an independent implementation inspired by the stochastic-optimization-forest research line referenced in the supplied 2015–2026 IE/OR review. It is not claimed as a reproduction of Kallus–Mao or any specific implementation.

The key comparison is contextual forest-weighted stochastic optimization versus an unconditional empirical decision. Tests verify probability weights and context-sensitive decisions.
