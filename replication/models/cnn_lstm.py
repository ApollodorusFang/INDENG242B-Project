"""1D causal CNN feeding into an LSTM regressor."""
from __future__ import annotations

import torch
import torch.nn as nn

from .. import config
from .base import FamilyResult, run_torch_family


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        cnn_channels: int,
        kernel_size: int,
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
    ):
        super().__init__()
        pad = kernel_size - 1
        self.conv1 = nn.Conv1d(n_features, cnn_channels, kernel_size, padding=pad, dilation=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=pad * 2, dilation=2)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        self.conv_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    @staticmethod
    def _causal(out: torch.Tensor, seq_len: int) -> torch.Tensor:
        return out[:, :, :seq_len]

    def forward(self, x):
        _, L, _ = x.shape
        h = x.transpose(1, 2)
        h = torch.relu(self.bn1(self._causal(self.conv1(h), L)))
        h = torch.relu(self.bn2(self._causal(self.conv2(h), L)))
        h = self.conv_dropout(h)
        h = h.transpose(1, 2)
        out, _ = self.lstm(h)
        return self.head(out[:, -1, :]).squeeze(-1)


_FAST_GRID = {
    "cnn_channels":  [32, 64],
    "kernel_size":   [3],
    "lstm_hidden":   [32, 64],
    "lstm_layers":   [1],
    "dropout":       [0.1, 0.3],
    "learning_rate": [1e-3],
    "batch_size":    [128],
}
_FULL_GRID = {
    "cnn_channels":  [32, 64],
    "kernel_size":   [3],
    "lstm_hidden":   [32, 64],
    "lstm_layers":   [1, 2],
    "dropout":       [0.1, 0.3],
    "learning_rate": [1e-3, 5e-4],
    "batch_size":    [128, 256],
}


def _ctor(cfg: dict, n_features: int) -> CNNLSTM:
    return CNNLSTM(
        n_features=n_features,
        cnn_channels=cfg["cnn_channels"],
        kernel_size=cfg["kernel_size"],
        lstm_hidden=cfg["lstm_hidden"],
        lstm_layers=cfg["lstm_layers"],
        dropout=cfg["dropout"],
    )


def run(
    X_train, y_train, X_val, y_val, X_test, y_test, test_dates,
    *, log=print,
) -> FamilyResult:
    grid = _FULL_GRID if config.GRID_PRESET == "full" else _FAST_GRID
    keys = [
        "cnn_channels", "kernel_size", "lstm_hidden", "lstm_layers",
        "dropout", "learning_rate", "batch_size",
    ]
    return run_torch_family(
        prefix="cnn_lstm",
        arch_name="CNN-LSTM",
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
