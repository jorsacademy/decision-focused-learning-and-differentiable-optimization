import numpy as np

from sof.forest import fit_forest, fit_stump


def test_optimization_aware_split_detects_context() -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, size=(400, 2))
    y = np.where(x[:, 0] <= 0.0, 5.0, 12.0) + rng.normal(scale=0.4, size=400)
    stump = fit_stump(x, y, 4.0, 1.0)
    assert stump.feature == 0
    assert abs(stump.threshold) < 0.25


def test_forest_returns_finite_contextual_decisions() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 2))
    y = 8.0 + 3.0 * (x[:, 0] > 0) + rng.normal(size=300)
    forest = fit_forest(x, y, n_trees=7)
    q = forest.predict(x[:20])
    assert q.shape == (20,)
    assert np.isfinite(q).all()
