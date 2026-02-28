"""Inference wrapper for ESM-2 + TON regressor."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .esm_backend import ESM2FeatureExtractor, resolve_device
from .regressor import TONRegressor
from .train import predict_with_regressor


class ESM2TONPredictor:
    """Predict TON and token log-probs for guided generation."""

    def __init__(
        self,
        feature_extractor: ESM2FeatureExtractor,
        regressor: TONRegressor,
        target_mean: float,
        target_std: float,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.regressor = regressor
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    @property
    def device(self) -> str:
        return self.feature_extractor.device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_name: str | None = None,
        device: str = "auto",
    ) -> "ESM2TONPredictor":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        resolved_model_name = model_name or checkpoint.get("model_name")
        if not resolved_model_name:
            raise ValueError("Model name not found in checkpoint and not provided.")

        resolved_device = resolve_device(device)
        feature_extractor = ESM2FeatureExtractor(resolved_model_name, device=resolved_device)

        regressor = TONRegressor(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            dropout=float(checkpoint.get("dropout", 0.1)),
        )
        regressor.load_state_dict(checkpoint["regressor_state"])
        regressor.to(feature_extractor.device)
        regressor.eval()

        return cls(
            feature_extractor=feature_extractor,
            regressor=regressor,
            target_mean=float(checkpoint["target_mean"]),
            target_std=float(checkpoint["target_std"]),
        )

    def predict(
        self,
        sequences: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        embeddings = self.feature_extractor.embed(
            sequences,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        return predict_with_regressor(
            model=self.regressor,
            embeddings=embeddings,
            target_mean=self.target_mean,
            target_std=self.target_std,
            device=self.device,
            batch_size=batch_size,
        )

    def residue_log_probs(self, sequence: str, residue_index: int) -> dict[str, float]:
        return self.feature_extractor.masked_token_log_probs(sequence, residue_index)
