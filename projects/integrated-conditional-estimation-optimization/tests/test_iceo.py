import numpy as np

from iceo_newsvendor import fit_gaussian_iceo, newsvendor_cost


def _data(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    scale = 0.5 + 0.35 * (x[:, 1] > 0.0)
    y = 4.0 + 1.2 * x[:, 0] - 0.4 * x[:, 2] + rng.normal(scale=scale)
    return x, y


def test_integrated_fit_produces_finite_contextual_decisions() -> None:
    x, y = _data(1, 600)
    model = fit_gaussian_iceo(x, y, decision_weight=0.6)
    q = model.decision(x[:30])
    assert q.shape == (30,)
    assert np.isfinite(q).all()
    assert model.scale > 0.0


def test_decision_aware_fit_is_evaluated_by_operational_cost() -> None:
    x_train, y_train = _data(2, 900)
    x_test, y_test = _data(3, 4000)
    separate = fit_gaussian_iceo(x_train, y_train, decision_weight=0.0)
    integrated = fit_gaussian_iceo(x_train, y_train, decision_weight=0.5)

    for model in (separate, integrated):
        cost = newsvendor_cost(model.decision(x_test), y_test, underage=3.0, overage=1.0)
        assert np.isfinite(cost).all()
