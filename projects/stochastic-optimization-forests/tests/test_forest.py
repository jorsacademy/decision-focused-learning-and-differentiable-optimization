import numpy as np

from sof_newsvendor import OptimizationForest, OptimizationTree, newsvendor_cost


def _data(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = 5.0 + 2.5 * (x[:, 0] > 0.0) + 1.2 * x[:, 1] + rng.normal(0.0, 0.6, n)
    return x, y


def test_tree_returns_context_dependent_feasible_decisions() -> None:
    x, y = _data(1, 500)
    tree = OptimizationTree(max_depth=2, min_leaf=30).fit(x, y)
    q = tree.decision(np.array([[-0.8, 0.0], [0.8, 0.0]]))
    assert q.shape == (2,)
    assert q[1] > q[0]


def test_forest_optimizes_weighted_leaf_distribution() -> None:
    x_train, y_train = _data(2, 700)
    x_test, y_test = _data(3, 1200)
    forest = OptimizationForest(n_estimators=8, max_depth=3, min_leaf=25, random_state=5).fit(
        x_train, y_train
    )
    q = forest.decision(x_test)
    cost = newsvendor_cost(q, y_test, underage=3.0, overage=1.0)
    assert np.isfinite(q).all()
    assert np.isfinite(cost).all()
