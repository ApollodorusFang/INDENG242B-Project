"""Traditional regression + directional metrics for predicted log-returns.

For each model family the predictions CSV contains
``{prefix}_Model_1..K``, ``Ensemble_Prediction``, and ``Actual``. We score
each ensemble against the realized 10-step log-return.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .. import config


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def regression_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def directional_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    label = (y_true > 0).astype(int)
    pred = (y_pred > 0).astype(int)

    if len(label) == 0 or label.sum() in (0, len(label)):
        auc = float("nan")
    else:
        auc = float(roc_auc_score(label, y_pred))

    scale = float(np.std(y_true)) + 1e-12
    prob = np.clip(_sigmoid(y_pred / scale), 1e-6, 1 - 1e-6)

    return {
        "DirAccuracy": float(accuracy_score(label, pred)),
        "Precision": float(precision_score(label, pred, zero_division=0)),
        "Recall": float(recall_score(label, pred, zero_division=0)),
        "F1": float(f1_score(label, pred, zero_division=0)),
        "AUC": auc,
        "LogLoss": float(log_loss(label, prob)),
        "PosRate_Pred": float(pred.mean()),
        "PosRate_True": float(label.mean()),
    }


def evaluate_family(prefix: str) -> dict[str, float]:
    df = pd.read_csv(config.RESULTS_DIR / f"{prefix}_predictions.csv")
    y_true = df["Actual"].to_numpy(dtype=float)
    y_pred = df["Ensemble_Prediction"].to_numpy(dtype=float)
    out = {"family": prefix, "n": len(y_true)}
    out.update(regression_block(y_true, y_pred))
    out.update(directional_block(y_true, y_pred))
    return out


def evaluate_all() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(config.RESULTS_DIR / "*_predictions.csv"))):
        prefix = Path(path).stem.replace("_predictions", "")
        rows.append(evaluate_family(prefix))
    df = pd.DataFrame(rows)
    out_path = config.RESULTS_DIR / "_traditional_metrics.csv"
    df.to_csv(out_path, index=False)
    return df
