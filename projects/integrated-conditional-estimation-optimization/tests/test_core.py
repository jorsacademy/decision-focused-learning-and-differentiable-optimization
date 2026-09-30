import numpy as np

from conditional_opt.core import fit_by_decision_validation,newsvendor_cost,weighted_quantile


def test_weighted_quantile_matches_simple_case():
    assert weighted_quantile([1,2,3],[1,1,1],0.5)==2.0


def test_validation_selects_declared_bandwidth_and_decisions_are_finite():
    rng=np.random.default_rng(0)
    x=rng.normal(size=(120,1))
    d=10+3*x[:,0]+rng.normal(scale=.2,size=120)
    model=fit_by_decision_validation(x[:70],d[:70],x[70:95],d[70:95],[.1,.3,1.0])
    assert model.bandwidth in {.1,.3,1.0}
    q=np.array([model.decision(row) for row in x[95:]])
    assert np.isfinite(q).all()
    assert np.mean(newsvendor_cost(q,d[95:],4,1))>=0
