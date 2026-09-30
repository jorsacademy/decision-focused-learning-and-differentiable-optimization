"""Synthetic contextual-newsboy benchmark."""

from __future__ import annotations

import numpy as np

from .core import fit_by_decision_validation,newsvendor_cost,weighted_quantile


def generate(seed:int,n:int)->tuple[np.ndarray,np.ndarray]:
    rng=np.random.default_rng(seed)
    x=rng.uniform(-1.0,1.0,size=(n,2))
    demand=10+4*x[:,0]-2*x[:,1]+3*np.sin(2*x[:,0])+rng.normal(0,1.2,size=n)
    return x,np.maximum(demand,0.0)


def run(seed:int=0)->dict[str,float]:
    xt,dt=generate(seed,600)
    xv,dv=generate(seed+1,250)
    xe,de=generate(seed+2,500)
    u,o=4.0,1.0
    model=fit_by_decision_validation(xt,dt,xv,dv,[0.15,0.25,0.4,0.7,1.2],underage=u,overage=o)
    q=np.array([model.decision(row) for row in xe])
    global_q=weighted_quantile(dt,np.ones_like(dt),u/(u+o))
    return {
        "bandwidth":model.bandwidth,
        "conditional_cost":float(np.mean(newsvendor_cost(q,de,u,o))),
        "global_cost":float(np.mean(newsvendor_cost(global_q,de,u,o))),
    }


if __name__=="__main__":
    print(run())
