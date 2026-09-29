"""Held-out prediction/decision trade-off benchmark."""

from __future__ import annotations

import numpy as np

from .model import decision_cost,fit_decision_regularized


def generate(seed:int,n:int)->tuple[np.ndarray,np.ndarray]:
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,3))
    y=12+2*x[:,0]-x[:,1]+.5*x[:,2]+rng.exponential(1.4,size=n)
    return x,y


def run(seed:int=0)->list[dict[str,float]]:
    xt,yt=generate(seed,600)
    xe,ye=generate(seed+1,500)
    rows=[]
    for alpha in [0.0,0.2,0.5,0.8]:
        model=fit_decision_regularized(xt,yt,alpha=alpha,underage=6,overage=1)
        pred=model.predict(xe)
        rows.append({
            "alpha":alpha,
            "mse":float(np.mean((pred-ye)**2)),
            "decision_cost":decision_cost(pred,ye,6,1),
        })
    return rows


if __name__=="__main__":
    print(run())
