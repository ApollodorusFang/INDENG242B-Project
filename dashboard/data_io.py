"""Data-loading utilities for the Streamlit dashboard.

All loaders fail gracefully: if a file is missing or the schema is
unexpected, return ``None`` (or a tiny demo dataframe) and let the caller
decide what to show.

The dashboard reads three things:
1. ``data/processed/dataset_metadata.json`` — sample sizes, lookback, horizon.
2. ``replication/results/_traditional_metrics.csv`` — RMSE / R^2 / DirAcc / AUC table.
3. ``dashboard/data/test_predictions.csv`` (preferred) or
   ``replication/results/{family}_predictions.csv`` — per-step predictions.

If none of the above exist, a small synthetic fallback is used so the UI
still renders cleanly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "replication" / "results"
DASHBOARD_DATA_DIR = REPO_ROOT / "dashboard" / "data"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Stable model family ordering. Value = (display name, plot colour).
MODEL_FAMILIES: dict[str, tuple[str, str]] = {
    "ts": ("ARIMA", "#7f8c8d"),
    "rf": ("Random Forest", "#3498db"),
    "rnn": ("Stacked RNN", "#9b59b6"),
    "lstm": ("Stacked LSTM", "#e91e63"),
    "cnn_lstm": ("CNN-LSTM", "#e67e22"),
    "gru_attn": ("GRU + Attention", "#2ecc71"),
}

# Hard-coded fallback regression metrics, copied from Table 2 of the report.
# Used only when ``_traditional_metrics.csv`` is missing from the repo so the
# dashboard still loads with sensible numbers in a fresh clone.
FALLBACK_REGRESSION = pd.DataFrame(
    [
        {"Model": "ARIMA",           "family": "ts",        "RMSE_x1e4": 1.25, "MAE_x1e4": 0.778, "R2": -0.002, "DirAcc": 0.521, "AUC": 0.528},
        {"Model": "Random Forest",   "family": "rf",        "RMSE_x1e4": 1.34, "MAE_x1e4": 0.871, "R2": -0.142, "DirAcc": 0.486, "AUC": 0.516},
        {"Model": "Stacked RNN",     "family": "rnn",       "RMSE_x1e4": 1.46, "MAE_x1e4": 0.985, "R2": -0.359, "DirAcc": 0.501, "AUC": 0.536},
        {"Model": "Stacked LSTM",    "family": "lstm",      "RMSE_x1e4": 1.52, "MAE_x1e4": 1.072, "R2": -0.477, "DirAcc": 0.471, "AUC": 0.513},
        {"Model": "CNN-LSTM",        "family": "cnn_lstm",  "RMSE_x1e4": 1.43, "MAE_x1e4": 0.968, "R2": -0.311, "DirAcc": 0.466, "AUC": 0.502},
        {"Model": "GRU + Attention", "family": "gru_attn",  "RMSE_x1e4": 1.36, "MAE_x1e4": 0.871, "R2": -0.178, "DirAcc": 0.497, "AUC": 0.509},
    ]
)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def load_dataset_metadata() -> dict:
    p = PROCESSED_DIR / "dataset_metadata.json"
    if not p.exists():
        return {
            "raw_rows": 14577,
            "num_features": 82,
            "lookback_window": 60,
            "prediction_horizon": 10,
            "split_sizes": {"train": 10204, "val": 2186, "test": 2187},
        }
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Regression metrics table
# ---------------------------------------------------------------------------
def load_regression_metrics() -> pd.DataFrame:
    """Try the merged CSV, then per-family summaries, else fall back to report numbers."""
    merged = RESULTS_DIR / "_traditional_metrics.csv"
    if merged.exists():
        try:
            df = pd.read_csv(merged)
            return _normalize_metrics_columns(df)
        except Exception:
            pass

    rows = []
    for fam, (name, _) in MODEL_FAMILIES.items():
        summ = RESULTS_DIR / f"{fam}_summary.csv"
        if not summ.exists():
            continue
        try:
            df = pd.read_csv(summ)
            row = df.iloc[0].to_dict() if len(df) else {}
            row["Model"] = name
            row["family"] = fam
            rows.append(row)
        except Exception:
            continue
    if rows:
        return _normalize_metrics_columns(pd.DataFrame(rows))

    return FALLBACK_REGRESSION.copy()


def _normalize_metrics_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort renaming so the dashboard always sees a stable column set."""
    rename_map = {}
    for c in df.columns:
        low = c.lower().strip()
        if low in ("model", "name", "family_name"):
            rename_map[c] = "Model"
        elif low in ("rmse",):
            rename_map[c] = "RMSE"
        elif low in ("mae",):
            rename_map[c] = "MAE"
        elif low in ("r2", "r_squared", "r^2"):
            rename_map[c] = "R2"
        elif low in ("dir_acc", "diracc", "directional_accuracy"):
            rename_map[c] = "DirAcc"
        elif low in ("auc", "roc_auc"):
            rename_map[c] = "AUC"
        elif low in ("f1", "f1_score"):
            rename_map[c] = "F1"
    df = df.rename(columns=rename_map)
    if "RMSE" in df.columns and "RMSE_x1e4" not in df.columns:
        df["RMSE_x1e4"] = df["RMSE"] * 1e4
    if "MAE" in df.columns and "MAE_x1e4" not in df.columns:
        df["MAE_x1e4"] = df["MAE"] * 1e4
    keep_order = [c for c in
                  ["Model", "RMSE_x1e4", "MAE_x1e4", "R2", "DirAcc", "AUC", "F1"]
                  if c in df.columns]
    other = [c for c in df.columns if c not in keep_order]
    return df[keep_order + other]


# ---------------------------------------------------------------------------
# Predictions panel  (one row per test sample, one pred_{fam} column per model)
# ---------------------------------------------------------------------------
def load_predictions_panel() -> Optional[pd.DataFrame]:
    """Preferred: a single consolidated CSV at dashboard/data/test_predictions.csv.

    Schema:
        timestamp_ms (int) · realized_log_return (float) · mid_price_t (float, optional) ·
        mid_price_t_plus_h (float, optional) · pred_ts · pred_rf · pred_rnn ·
        pred_lstm · pred_cnn_lstm · pred_gru_attn

    Run ``python -m replication.export_dashboard_data`` once after the
    replication pipeline finishes to generate it.
    """
    consolidated = DASHBOARD_DATA_DIR / "test_predictions.csv"
    if consolidated.exists():
        try:
            return pd.read_csv(consolidated)
        except Exception:
            pass

    # Fallback: try to stitch from per-family CSVs in replication/results/.
    realized = _load_realized_returns()
    if realized is None:
        return None

    panel = pd.DataFrame({"realized_log_return": realized})
    panel["timestamp_ms"] = np.arange(len(realized), dtype=np.int64) * 1000

    found_any = False
    for fam in MODEL_FAMILIES.keys():
        path = RESULTS_DIR / f"{fam}_predictions.csv"
        if not path.exists():
            continue
        pred = _extract_prediction_column(path)
        if pred is None:
            continue
        n = min(len(pred), len(panel))
        col = np.full(len(panel), np.nan)
        col[:n] = pred[:n]
        panel[f"pred_{fam}"] = col
        found_any = True

    return panel if found_any else None


def _load_realized_returns() -> Optional[np.ndarray]:
    p = PROCESSED_DIR / "y_test_reg.npy"
    if not p.exists():
        return None
    try:
        return np.load(p).astype(float)
    except Exception:
        return None


def _extract_prediction_column(path: Path) -> Optional[np.ndarray]:
    """Find the ensemble / prediction column in a model's predictions CSV."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    candidates = [
        "predicted", "pred", "y_pred", "ensemble", "forecast",
        "prediction", "pred_test", "y_hat", "yhat",
    ]
    for c in candidates:
        if c in df.columns:
            return df[c].to_numpy(dtype=float)
    # last numeric column heuristic
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        return df[num_cols[-1]].to_numpy(dtype=float)
    return None


# ---------------------------------------------------------------------------
# Demo fallback so the UI still renders when no data is committed
# ---------------------------------------------------------------------------
def fallback_predictions_demo(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    """Synthesise a tiny BTC-like price path so the dashboard always renders.

    Returns a panel with all six pred_{fam} columns. The synthetic predictions
    have the same near-zero, low-skill character as the real models in our
    test set, so the sandbox demo is qualitatively faithful even on a fresh
    clone with no committed data.
    """
    rng = np.random.default_rng(seed)
    base_ts = np.int64(1_715_000_000_000)
    timestamp_ms = base_ts + np.arange(n) * 1000

    # Random walk mid-price around 60k.
    rets = rng.normal(0.0, 1.5e-4, size=n + 10)
    log_price = np.cumsum(rets) + np.log(60_000.0)
    mid_price = np.exp(log_price)
    mid_t = mid_price[:n]
    mid_th = mid_price[10 : 10 + n]
    realized = np.log(mid_th / mid_t)

    df = pd.DataFrame({
        "timestamp_ms": timestamp_ms,
        "mid_price_t": mid_t,
        "mid_price_t_plus_h": mid_th,
        "realized_log_return": realized,
    })

    # Faux predictions: realized * tiny correlation + heavy noise → near-zero R^2.
    scales = {"ts": 0.05, "rf": 0.03, "rnn": -0.02, "lstm": -0.04,
              "cnn_lstm": 0.01, "gru_attn": 0.02}
    for fam, s in scales.items():
        noise = rng.normal(0.0, 1.0e-4, size=n)
        df[f"pred_{fam}"] = s * realized + noise
    return df
