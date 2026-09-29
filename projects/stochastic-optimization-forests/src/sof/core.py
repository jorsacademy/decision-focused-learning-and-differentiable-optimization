"""Random-forest proximity weights for contextual stochastic decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike,NDArray
from sklearn.ensemble import RandomForestRegressor


def weighted_quantile(v:NDArray[np.float64],w:NDArray[np.float64],p:float)->float:
    order=np.argsort(v)
    vv=v[order]
    ww=w[order]/w.sum()
    return float(vv[np.searchsorted(np.cumsum(ww),p,side="left")])


@dataclass
class StochasticOptimizationForest:
    n_estimators:int=100
    min_samples_leaf:int=10
    random_state:int=0

    def fit(self,x:ArrayLike,y:ArrayLike)->"StochasticOptimizationForest":
        self.x_=np.asarray(x,dtype=float)
        self.y_=np.asarray(y,dtype=float)
        if self.x_.ndim!=2 or self.y_.shape!=(self.x_.shape[0],):
            raise ValueError("x/y shapes are incompatible")
        self.forest_=RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
        ).fit(self.x_,self.y_)
        self.train_leaves_=self.forest_.apply(self.x_)
        return self

    def weights(self,x:ArrayLike)->NDArray[np.float64]:
        if not hasattr(self,"forest_"):
            raise RuntimeError("fit must be called first")
        xx=np.asarray(x,dtype=float).reshape(1,-1)
        qleaves=self.forest_.apply(xx)[0]
        w=np.zeros(self.x_.shape[0],dtype=float)
        for t,leaf in enumerate(qleaves):
            members=self.train_leaves_[:,t]==leaf
            count=int(np.sum(members))
            if count:
                w[members]+=1.0/(self.n_estimators*count)
        if w.sum()<=0:
            w[:]=1.0/len(w)
        else:
            w/=w.sum()
        return w

    def newsvendor_decision(self,x:ArrayLike,underage:float=4.0,overage:float=1.0)->float:
        p=underage/(underage+overage)
        return weighted_quantile(self.y_,self.weights(x),p)
