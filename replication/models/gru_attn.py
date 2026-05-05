"""GRU + Bahdanau-style attention (port of ESE 5460 EC Pytorch NN Modeling)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .. import config
from .base import FamilyResult, run_torch_family


class GRUAttention(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        enc_dim = hidden * 2
        self.attn_W = nn.Linear(enc_dim, enc_dim)
        self.attn_v = nn.Linear(enc_dim, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(enc_dim, enc_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_dim // 2, 1),
        )

    def forward(self, x):
        H, _ = self.gru(x)
        scores = self.attn_v(torch.tanh(self.attn_W(H)))
        weights = torch.softmax(scores, dim=1)
        ctx = (weights * H).sum(dim=1)
        return self.head(ctx).squeeze(-1)


_FAST_GRID = {
    "hidden":        [32, 64],
    "n_layers":      [1, 2],
    "dropout":       [0.1, 0.3],
    "learning_rate": [1e-3],
    "batch_size":    [128],
}
_FULL_GRID = {
    "hidden":        [32, 64],
    "n_layers":      [1, 2],
    "dropout":       [0.1, 0.3],
    "learning_rate": [1e-3, 5e-4],
    "batch_size":    [128, 256],
}


def _ctor(cfg: dict, n_features: int) -> GRUAttention:
    return GRUAttention(n_features, cfg["hidden"], cfg["n_layers"], cfg["dropout"])


def run(
    X_train, y_train, X_val, y_val, X_test, y_test, test_dates,
    *, log=print,
) -> FamilyResult:
    grid = _FULL_GRID if config.GRID_PRESET == "full" else _FAST_GRID
    keys = ["hidden", "n_layers", "dropout", "learning_rate", "batch_size"]
    return run_torch_family(
        prefix="gru_attn",
        arch_name="GRU+Attention",
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
