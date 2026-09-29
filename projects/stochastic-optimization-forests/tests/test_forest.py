import numpy as np

from sof import StochasticOptimizationForest


def test_weights_are_probability_vector():
    rng=np.random.default_rng(0)
    x=rng.normal(size=(120,2))
    y=3*x[:,0]+rng.normal(size=120)
    model=StochasticOptimizationForest(n_estimators=20,min_samples_leaf=5).fit(x,y)
    w=model.weights(x[0])
    assert np.isclose(w.sum(),1.0)
    assert np.all(w>=0)


def test_context_changes_decision_on_heterogeneous_data():
    rng=np.random.default_rng(1)
    x=rng.uniform(-1,1,size=(500,1))
    y=np.where(x[:,0]<0,5.0,20.0)+rng.normal(0,.2,size=500)
    model=StochasticOptimizationForest(n_estimators=60,min_samples_leaf=8).fit(x,y)
    assert model.newsvendor_decision([-0.8])<model.newsvendor_decision([0.8])
