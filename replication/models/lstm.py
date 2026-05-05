"""Stacked LSTM regressor.

Predicts the regression label (10-step log return of mid_price) from a window
of 60 LOB feature vectors using a grid search + top-10 ensemble pipeline.
"""
from __future__ import annotations

import torch.nn as nn

from .. import config
from .base import FamilyResult, run_torch_family


class StackedLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = self.dropout(out[:, -1, :])
        return self.head(last).squeeze(-1)


_FAST_GRID = {
    "hidden":        [32, 64],
    "n_layers":      [1, 2],
    "dropout":       [0.1, 0.3],
    "learning_rate": [1e-3],
    "batch_size":    [128],
}
_FULL_GRID = {
    "hidden":        [32, 64, 128],
    "n_layers":      [1, 2],
    "dropout":       [0.1, 0.2, 0.3],
    "learning_rate": [1e-3, 5e-4],
    "batch_size":    [128, 256],
}


def _ctor(cfg: dict, n_features: int) -> StackedLSTM:
    return StackedLSTM(n_features, cfg["hidden"], cfg["n_layers"], cfg["dropout"])


def run(
    X_train, y_train, X_val, y_val, X_test, y_test, test_dates,
    *, log=print,
) -> FamilyResult:
    grid = _FULL_GRID if config.GRID_PRESET == "full" else _FAST_GRID
    keys = ["hidden", "n_layers", "dropout", "learning_rate", "batch_size"]
    return run_torch_family(
        prefix="lstm",
        arch_name="LSTM",
        model_ctor=_ctor,
        param_grid_spec=grid,
        param_keys=keys,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        test_dates=test_dates,
        epochs=20 if config.GRID_PRESET == "fast" else 40,
        patience=5,
        log=log,
    )
