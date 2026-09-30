"""Contextual forest-weighted newsvendor benchmark."""

from __future__ import annotations

import numpy as np

from .core import StochasticOptimizationForest


def generate(seed:int,n:int)->tuple[np.ndarray,np.ndarray]:
    rng=np.random.default_rng(seed)
    x=rng.uniform(-2,2,size=(n,2))
    y=20+5*np.sin(x[:,0])+3*(x[:,1]>0)+rng.normal(0,1.5,size=n)
    return x,y


def cost(q:np.ndarray,d:np.ndarray,u:float=4,o:float=1)->float:
    return float(np.mean(u*np.maximum(d-q,0)+o*np.maximum(q-d,0)))


def run(seed:int=0)->dict[str,float]:
    xt,yt=generate(seed,800)
    xe,ye=generate(seed+1,400)
    model=StochasticOptimizationForest(n_estimators=120,min_samples_leaf=12,random_state=seed).fit(xt,yt)
    q=np.array([model.newsvendor_decision(row) for row in xe])
    global_q=float(np.quantile(yt,.8))
    return {"forest_cost":cost(q,ye),"global_cost":cost(np.full_like(ye,global_q),ye)}


if __name__=="__main__":
    print(run())
