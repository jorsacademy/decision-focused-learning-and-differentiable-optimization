import numpy as np

from ddr.model import asymmetric_cost, fit_ddr


def test_alpha_changes_prediction_decision_tradeoff() -> None:
    rng = np.random.default_rng(9)
    x = np.column_stack([np.ones(600), rng.normal(size=600)])
    y = 8.0 + 2.0 * x[:, 1] + rng.normal(size=600)

    predictive = fit_ddr(x, y, alpha=1.0, underage=5.0, overage=1.0)
    blended = fit_ddr(x, y, alpha=0.25, underage=5.0, overage=1.0)

    pred_cost = asymmetric_cost(predictive.predict(x), y, 5.0, 1.0)
    blend_cost = asymmetric_cost(blended.predict(x), y, 5.0, 1.0)
    assert blend_cost <= pred_cost + 1e-5
