"""Training loops for prediction-focused and SPO+ models."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from dfl_spo.losses import mse_loss, spo_plus_loss


def train_model(
    model: nn.Module,
    features: Tensor,
    costs: Tensor,
    *,
    method: str,
    k: int,
    epochs: int = 100,
    learning_rate: float = 1e-2,
) -> list[float]:
    """Train a predictor with MSE or SPO+ and return epoch losses."""
    if method not in {"mse", "spo+"}:
        raise ValueError("method must be 'mse' or 'spo+'")
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn: Callable[[Tensor, Tensor], Tensor]
    if method == "mse":
        loss_fn = mse_loss
    else:
        def loss_fn(pred, true):
            return spo_plus_loss(pred, true, k)

    history: list[float] = []
    model.train()
    for _ in range(epochs):
        predicted = model(features)
        loss = loss_fn(predicted, costs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history
