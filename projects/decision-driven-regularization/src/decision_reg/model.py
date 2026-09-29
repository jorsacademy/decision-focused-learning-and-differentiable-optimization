"""Linear prediction with a blended prediction/decision training objective."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike,NDArray
from scipy.optimize import minimize


def decision_cost(pred:ArrayLike,y:ArrayLike,underage:float,overage:float)->float:
    p=np.asarray(pred,dtype=float)
    t=np.asarray(y,dtype=float)
    return float(np.mean(underage*np.maximum(t-p,0)+overage*np.maximum(p-t,0)))


@dataclass(frozen=True)
class DecisionRegularizedLinearModel:
    coefficients:NDArray[np.float64]
    intercept:float
    alpha:float
    underage:float
    overage:float

    def predict(self,x:ArrayLike)->NDArray[np.float64]:
        xx=np.asarray(x,dtype=float)
        return self.intercept+xx@self.coefficients


def fit_decision_regularized(
    x:ArrayLike,
    y:ArrayLike,
    *,
    alpha:float,
    underage:float=4.0,
    overage:float=1.0,
    ridge:float=1e-4,
)->DecisionRegularizedLinearModel:
    xx=np.asarray(x,dtype=float)
    yy=np.asarray(y,dtype=float)
    if xx.ndim!=2 or yy.shape!=(xx.shape[0],):
        raise ValueError("x/y shapes are incompatible")
    if not 0<=alpha<=1:
        raise ValueError("alpha must lie in [0,1]")
    design=np.column_stack([np.ones(xx.shape[0]),xx])
    beta0=np.linalg.lstsq(design,yy,rcond=None)[0]
    scale=max(float(np.var(yy)),1e-8)
    def objective(beta:NDArray[np.float64])->float:
        pred=design@beta
        mse=float(np.mean((pred-yy)**2))/scale
        dcost=decision_cost(pred,yy,underage,overage)/(max(float(np.mean(np.abs(yy))),1.0))
        return (1-alpha)*mse+alpha*dcost+ridge*float(beta[1:]@beta[1:])
    result=minimize(objective,beta0,method="Powell",options={"maxiter":4000,"xtol":1e-9,"ftol":1e-9})
    if not result.success:
        raise RuntimeError(result.message)
    return DecisionRegularizedLinearModel(
        np.asarray(result.x[1:],dtype=float),float(result.x[0]),float(alpha),underage,overage
    )
