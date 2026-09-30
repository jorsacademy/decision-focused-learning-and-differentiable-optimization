import numpy as np

from decision_reg.model import decision_cost,fit_decision_regularized


def test_decision_regularization_changes_asymmetric_decision_bias():
    rng=np.random.default_rng(2)
    x=rng.normal(size=(180,1))
    y=8+2*x[:,0]+rng.exponential(1.2,size=180)
    mse=fit_decision_regularized(x,y,alpha=0.0,underage=8,overage=1)
    driven=fit_decision_regularized(x,y,alpha=0.8,underage=8,overage=1)
    assert np.mean(driven.predict(x))>np.mean(mse.predict(x))


def test_decision_cost_nonnegative():
    assert decision_cost([1,2],[2,1],4,1)>=0
