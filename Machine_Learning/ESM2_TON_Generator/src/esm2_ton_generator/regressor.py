"""TON regressor model."""

from __future__ import annotations

import torch.nn as nn


class TONRegressor(nn.Module):
    """Small MLP head for TON prediction from ESM-2 embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):  # type: ignore[override]
        return self.net(x)
