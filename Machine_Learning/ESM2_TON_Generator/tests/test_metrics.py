from __future__ import annotations

import numpy as np

from esm2_ton_generator.metrics import regression_metrics


def test_regression_metrics_keys_and_ranges() -> None:
    y_true = np.array([2.0, 4.0, 6.0, 8.0])
    y_pred = np.array([2.5, 3.5, 6.5, 7.5])

    metrics = regression_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"mae", "mse", "rmse", "r2", "spearman"}
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert -1.0 <= metrics["r2"] <= 1.0
