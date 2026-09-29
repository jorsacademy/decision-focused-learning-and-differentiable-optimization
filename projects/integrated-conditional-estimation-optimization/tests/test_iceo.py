import numpy as np

from iceo.model import fit_iceo, fit_mse, newsvendor_cost


def test_iceo_is_decision_aligned_on_training_data() -> None:
    rng = np.random.default_rng(4)
    x = np.column_stack([np.ones(500), rng.normal(size=500), rng.normal(size=500)])
    demand = x @ np.array([10.0, 2.0, -1.0]) + 1.4 * rng.normal(size=500)

    mse = fit_mse(x, demand, sigma=1.4)
    iceo = fit_iceo(x, demand, sigma=1.4, underage=4.0, overage=1.0)

    mse_cost = newsvendor_cost(mse.decision(x, 4.0, 1.0), demand, 4.0, 1.0)
    iceo_cost = newsvendor_cost(iceo.decision(x, 4.0, 1.0), demand, 4.0, 1.0)
    assert iceo_cost <= mse_cost + 1e-5
