"""Training and inference helpers for TON regression."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .regressor import TONRegressor


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_regressor(
    train_embeddings: torch.Tensor,
    train_targets: np.ndarray,
    val_embeddings: torch.Tensor,
    val_targets: np.ndarray,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 200,
    patience: int = 30,
    device: str = "cpu",
) -> tuple[TONRegressor, list[dict[str, float]], float, float]:
    """Train the regressor with early stopping on validation MAE."""
    train_x = train_embeddings.to(dtype=torch.float32)
    val_x = val_embeddings.to(dtype=torch.float32)
    train_y_np = np.asarray(train_targets, dtype=np.float32)
    val_y_np = np.asarray(val_targets, dtype=np.float32)

    target_mean = float(train_y_np.mean())
    target_std = float(train_y_np.std())
    if target_std < 1e-8:
        target_std = 1.0

    train_y = torch.tensor((train_y_np - target_mean) / target_std, dtype=torch.float32)
    val_y = torch.tensor(val_y_np, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    model = TONRegressor(input_dim=train_x.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    history: list[dict[str, float]] = []

    val_x_device = val_x.to(device)
    val_y_device = val_y.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x).squeeze(-1)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * batch_x.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_pred_norm = model(val_x_device).squeeze(-1)
            val_pred = val_pred_norm * target_std + target_mean
            val_mae = torch.mean(torch.abs(val_pred - val_y_device)).item()

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_mae": float(val_mae),
            }
        )

        if val_mae + 1e-8 < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history, target_mean, target_std


@torch.no_grad()
def predict_with_regressor(
    model: TONRegressor,
    embeddings: torch.Tensor,
    target_mean: float,
    target_std: float,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Predict TON values from embeddings."""
    model.eval()
    dataset = TensorDataset(embeddings.to(dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions: list[np.ndarray] = []

    for (batch_x,) in loader:
        pred_norm = model(batch_x.to(device)).squeeze(-1).cpu().numpy()
        predictions.append(pred_norm)

    concatenated = np.concatenate(predictions)
    return concatenated * target_std + target_mean
