"""Build a supervised LOB dataset from raw BTCUSDT snapshots.

Produces sliding-window inputs X with shape (N, lookback, num_features) and
three label variants (regression / binary / 3-class) over a future
mid-price horizon. Train/val/test are split chronologically; the
StandardScaler is fit on train only to avoid look-ahead leakage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import FEATURE_COLS, setup_logger  # type: ignore
else:
    from .utils import FEATURE_COLS, setup_logger


def make_windows(
    features: np.ndarray,
    mid: np.ndarray,
    lookback: int,
    horizon: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y_reg, y_bin, y_3c) with non-overlapping label leakage.

    For sample i (0-indexed), X[i] covers feature rows [i, i+lookback) and the
    label is the log return of mid_price from row (i+lookback-1) to
    (i+lookback-1+horizon). That way, X uses only past information and the
    label looks strictly into the future.
    """
    n_total = features.shape[0]
    n_samples = n_total - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"not enough rows: have {n_total}, need >= {lookback + horizon}"
        )

    n_features = features.shape[1]
    X = np.empty((n_samples, lookback, n_features), dtype=np.float32)
    for i in range(n_samples):
        X[i] = features[i : i + lookback]

    t_idx = np.arange(n_samples) + (lookback - 1)
    y_reg = np.log(mid[t_idx + horizon] / mid[t_idx]).astype(np.float32)
    y_bin = (y_reg > 0).astype(np.int64)
    y_3c = np.where(
        y_reg > threshold, 2, np.where(y_reg < -threshold, 0, 1)
    ).astype(np.int64)
    return X, y_reg, y_bin, y_3c


def chronological_split(
    n_samples: int, train_frac: float, val_frac: float
) -> Tuple[slice, slice, slice]:
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    return (
        slice(0, n_train),
        slice(n_train, n_train + n_val),
        slice(n_train + n_val, n_samples),
    )


def normalize_X(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on flattened train rows and transform all splits."""
    n_features = X_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    def transform(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr.astype(np.float32, copy=False)
        flat = arr.reshape(-1, n_features)
        return scaler.transform(flat).reshape(arr.shape).astype(np.float32)

    return transform(X_train), transform(X_val), transform(X_test), scaler


def save_arrays(out_dir: str, name: str, X: np.ndarray, ys: Dict[str, np.ndarray]) -> None:
    np.save(os.path.join(out_dir, f"X_{name}.npy"), X)
    np.save(os.path.join(out_dir, f"y_{name}_reg.npy"), ys["reg"])
    np.save(os.path.join(out_dir, f"y_{name}_binary.npy"), ys["bin"])
    np.save(os.path.join(out_dir, f"y_{name}_3class.npy"), ys["3c"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LOB training dataset from raw parquet.")
    p.add_argument("--raw", default="data/raw/btcusdt_lob_raw.parquet")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--threshold", type=float, default=5e-5)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("build_dataset")
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.raw):
        raise FileNotFoundError(f"raw parquet not found: {args.raw}")

    df = pd.read_parquet(args.raw)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    missing = [c for c in FEATURE_COLS + ["mid_price"] if c not in df.columns]
    if missing:
        raise ValueError(f"raw parquet is missing columns: {missing}")

    # Drop rows where mid_price is non-positive (would break log return).
    valid = df["mid_price"] > 0
    if (~valid).any():
        logger.warning("dropping %d rows with non-positive mid_price", int((~valid).sum()))
        df = df.loc[valid].reset_index(drop=True)

    features = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    mid = df["mid_price"].to_numpy(dtype=np.float64)
    logger.info(
        "loaded %d rows; building windows lookback=%d horizon=%d threshold=%g",
        len(df),
        args.lookback,
        args.horizon,
        args.threshold,
    )

    X, y_reg, y_bin, y_3c = make_windows(
        features, mid, args.lookback, args.horizon, args.threshold
    )
    logger.info("built %d windows; X shape=%s", X.shape[0], X.shape)

    train_sl, val_sl, test_sl = chronological_split(
        X.shape[0], args.train_frac, args.val_frac
    )
    logger.info(
        "split sizes: train=%d val=%d test=%d",
        train_sl.stop - train_sl.start,
        val_sl.stop - val_sl.start,
        test_sl.stop - test_sl.start,
    )

    X_train, X_val, X_test = X[train_sl], X[val_sl], X[test_sl]
    X_train_n, X_val_n, X_test_n, scaler = normalize_X(X_train, X_val, X_test)

    splits = {
        "train": (train_sl, X_train_n),
        "val": (val_sl, X_val_n),
        "test": (test_sl, X_test_n),
    }
    for name, (sl, X_norm) in splits.items():
        save_arrays(
            args.out_dir,
            name,
            X_norm,
            {"reg": y_reg[sl], "bin": y_bin[sl], "3c": y_3c[sl]},
        )

    # Also save raw (un-normalized) X arrays for users who want them.
    np.save(os.path.join(args.out_dir, "X_train_raw.npy"), X_train)
    np.save(os.path.join(args.out_dir, "X_val_raw.npy"), X_val)
    np.save(os.path.join(args.out_dir, "X_test_raw.npy"), X_test)

    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.pkl"))
    with open(os.path.join(args.out_dir, "feature_names.json"), "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    label_dist_3c = {
        "train": np.bincount(y_3c[train_sl], minlength=3).tolist(),
        "val": np.bincount(y_3c[val_sl], minlength=3).tolist(),
        "test": np.bincount(y_3c[test_sl], minlength=3).tolist(),
    }
    metadata = {
        "symbol": "BTCUSDT",
        "raw_path": os.path.abspath(args.raw),
        "raw_rows": int(len(df)),
        "lookback_window": args.lookback,
        "prediction_horizon": args.horizon,
        "threshold": args.threshold,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": round(1.0 - args.train_frac - args.val_frac, 6),
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[2]),
        "feature_columns": FEATURE_COLS,
        "X_shape_per_sample": [args.lookback, len(FEATURE_COLS)],
        "split_sizes": {
            "train": int(train_sl.stop - train_sl.start),
            "val": int(val_sl.stop - val_sl.start),
            "test": int(test_sl.stop - test_sl.start),
        },
        "three_class_label_counts": label_dist_3c,
        "raw_timestamp_range_ms": [
            int(df["timestamp"].iloc[0]),
            int(df["timestamp"].iloc[-1]),
        ],
    }
    with open(os.path.join(args.out_dir, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("dataset written to %s", args.out_dir)


if __name__ == "__main__":
    main()
