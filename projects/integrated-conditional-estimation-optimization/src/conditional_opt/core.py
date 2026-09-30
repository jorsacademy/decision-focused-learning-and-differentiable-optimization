"""Conditional scenario weighting and downstream newsvendor optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def newsvendor_cost(
    quantity: ArrayLike,
    demand: ArrayLike,
    underage: float,
    overage: float,
) -> NDArray[np.float64]:
    q=np.asarray(quantity,dtype=float)
    d=np.asarray(demand,dtype=float)
    return underage*np.maximum(d-q,0.0)+overage*np.maximum(q-d,0.0)


def weighted_quantile(values: ArrayLike, weights: ArrayLike, probability: float) -> float:
    v=np.asarray(values,dtype=float)
    w=np.asarray(weights,dtype=float)
    if v.ndim!=1 or w.ndim!=1 or v.size!=w.size or v.size==0:
        raise ValueError("values and weights must be non-empty one-dimensional arrays")
    if np.any(w<0) or not np.isfinite(w).all() or w.sum()<=0:
        raise ValueError("weights must be finite, non-negative, and have positive sum")
    if not 0.0<probability<1.0:
        raise ValueError("probability must lie in (0,1)")
    order=np.argsort(v)
    vv=v[order]
    ww=w[order]/w.sum()
    return float(vv[np.searchsorted(np.cumsum(ww),probability,side="left")])


@dataclass(frozen=True)
class ConditionalNewsvendor:
    x_train: NDArray[np.float64]
    demand_train: NDArray[np.float64]
    bandwidth: float
    underage: float
    overage: float

    def weights(self,x: ArrayLike) -> NDArray[np.float64]:
        xx=np.asarray(x,dtype=float)
        if xx.shape!=(self.x_train.shape[1],):
            raise ValueError("query context has wrong dimension")
        z=(self.x_train-xx)/self.bandwidth
        logw=-0.5*np.sum(z*z,axis=1)
        logw-=np.max(logw)
        w=np.exp(logw)
        if w.sum()<=1e-15:
            w=np.ones_like(w)
        return w/w.sum()

    def decision(self,x: ArrayLike) -> float:
        p=self.underage/(self.underage+self.overage)
        return weighted_quantile(self.demand_train,self.weights(x),p)


def fit_by_decision_validation(
    x_train:ArrayLike,
    demand_train:ArrayLike,
    x_validation:ArrayLike,
    demand_validation:ArrayLike,
    bandwidths:ArrayLike,
    *,
    underage:float=4.0,
    overage:float=1.0,
)->ConditionalNewsvendor:
    xt=np.asarray(x_train,dtype=float)
    dt=np.asarray(demand_train,dtype=float)
    xv=np.asarray(x_validation,dtype=float)
    dv=np.asarray(demand_validation,dtype=float)
    hs=np.asarray(bandwidths,dtype=float)
    if xt.ndim!=2 or xv.ndim!=2 or xt.shape[1]!=xv.shape[1]:
        raise ValueError("training and validation contexts must be compatible matrices")
    if dt.shape!=(xt.shape[0],) or dv.shape!=(xv.shape[0],):
        raise ValueError("demand vectors must match their context matrices")
    if np.any(hs<=0) or hs.size==0:
        raise ValueError("bandwidths must be positive")
    best=None
    for h in hs:
        candidate=ConditionalNewsvendor(xt,dt,float(h),underage,overage)
        q=np.array([candidate.decision(row) for row in xv])
        score=float(np.mean(newsvendor_cost(q,dv,underage,overage)))
        if best is None or score<best[0]-1e-12:
            best=(score,candidate)
    assert best is not None
    return best[1]
