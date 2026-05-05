"""Dataset loading utilities for the replication package.

Loads the prebuilt LOB tensors (``X_*.npy``, ``y_*_reg.npy``,
``y_*_3class.npy``) and the raw mid-price series so downstream models and
the backtester can reuse a single source of truth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@dataclass
class LobDataset:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train_reg: np.ndarray
    y_val_reg: np.ndarray
    y_test_reg: np.ndarray
    y_train_3c: np.ndarray
    y_val_3c: np.ndarray
    y_test_3c: np.ndarray
    feature_names: list[str]
    metadata: dict
    test_timestamps_ms: np.ndarray
    test_mid_t: np.ndarray
    test_mid_t_plus_h: np.ndarray

    @property
    def lookback(self) -> int:
        return int(self.metadata["lookback_window"])

    @property
    def horizon(self) -> int:
        return int(self.metadata["prediction_horizon"])

    @property
    def num_features(self) -> int:
        return self.X_train.shape[2]


def _load_raw_frame() -> pd.DataFrame:
    raw_path = config.RAW_DIR / "btcusdt_lob_raw.parquet"
    df = pd.read_parquet(raw_path)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df.loc[df["mid_price"] > 0].reset_index(drop=True)
    return df


def load_dataset() -> LobDataset:
    p = config.PROCESSED_DIR
    metadata = json.loads((p / "dataset_metadata.json").read_text())
    feature_names = json.loads((p / "feature_names.json").read_text())

    raw = _load_raw_frame()
    lookback = int(metadata["lookback_window"])
    horizon = int(metadata["prediction_horizon"])
    n_train = int(metadata["split_sizes"]["train"])
    n_val = int(metadata["split_sizes"]["val"])
    n_test = int(metadata["split_sizes"]["test"])

    # Sample i (0-indexed) covers feature rows [i, i+lookback). Label looks at
    # mid_price[i+lookback-1] -> mid_price[i+lookback-1+horizon].
    sample_t_idx = np.arange(n_train + n_val + n_test) + (lookback - 1)
    test_offset = n_train + n_val
    test_t_idx = sample_t_idx[test_offset : test_offset + n_test]
    test_t_plus_h_idx = test_t_idx + horizon

    test_timestamps_ms = raw["timestamp"].to_numpy()[test_t_idx]
    test_mid_t = raw["mid_price"].to_numpy()[test_t_idx]
    test_mid_t_plus_h = raw["mid_price"].to_numpy()[test_t_plus_h_idx]

    return LobDataset(
        X_train=np.load(p / "X_train.npy"),
        X_val=np.load(p / "X_val.npy"),
        X_test=np.load(p / "X_test.npy"),
        y_train_reg=np.load(p / "y_train_reg.npy"),
        y_val_reg=np.load(p / "y_val_reg.npy"),
        y_test_reg=np.load(p / "y_test_reg.npy"),
        y_train_3c=np.load(p / "y_train_3class.npy"),
        y_val_3c=np.load(p / "y_val_3class.npy"),
        y_test_3c=np.load(p / "y_test_3class.npy"),
        feature_names=feature_names,
        metadata=metadata,
        test_timestamps_ms=test_timestamps_ms,
        test_mid_t=test_mid_t,
        test_mid_t_plus_h=test_mid_t_plus_h,
    )
