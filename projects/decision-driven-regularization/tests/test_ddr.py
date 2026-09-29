import numpy as np

from ddr_newsvendor import fit_ddr_linear, newsvendor_cost


def _data(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    y = 5.0 + 1.4 * x[:, 0] - 0.8 * x[:, 1] + 0.5 * x[:, 2] ** 2
    y += rng.normal(0.0, 1.0, size=n)
    return x, y


def test_extreme_weights_produce_finite_models() -> None:
    x, y = _data(4, 500)
    prediction_model = fit_ddr_linear(x, y, prediction_weight=1.0)
    decision_model = fit_ddr_linear(x, y, prediction_weight=0.0)
    assert np.isfinite(prediction_model.predict(x[:10])).all()
    assert np.isfinite(decision_model.predict(x[:10])).all()


def test_blended_model_reports_both_prediction_and_decision_metrics() -> None:
    x_train, y_train = _data(5, 700)
    x_test, y_test = _data(6, 1500)
    model = fit_ddr_linear(x_train, y_train, prediction_weight=0.4)
    pred = model.predict(x_test)
    mse = np.mean((pred - y_test) ** 2)
    cost = np.mean(newsvendor_cost(pred, y_test, underage=3.0, overage=1.0))
    assert np.isfinite(mse)
    assert np.isfinite(cost)
