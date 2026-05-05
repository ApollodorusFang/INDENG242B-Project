"""Shared training / grid-search / ensemble plumbing for sequence models.

Every model family writes the same four-file result schema:

* ``<prefix>_grid_search_results.csv``
* ``<prefix>_top10_model_metrics.csv``
* ``<prefix>_predictions.csv``       (Date, <prefix>_Model_1..K, Ensemble_Prediction, Actual)
* ``<prefix>_summary.csv``           (Metric, Value)

For 1-step regression the target is the log return ``y_test_reg`` directly.
``Date`` is the BTC server timestamp at which the prediction was made.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from .. import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = config.SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class FamilyResult:
    """Convenience bundle returned by :func:`run_torch_family`."""

    prefix: str
    arch_name: str
    grid_results: pd.DataFrame
    top_metrics: pd.DataFrame
    predictions: pd.DataFrame
    summary: pd.DataFrame


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def as_loader(X_, y_, shuffle):
        ds = TensorDataset(
            torch.from_numpy(np.asarray(X_, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_, dtype=np.float32)),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return (
        as_loader(X_train, y_train, shuffle=True),
        as_loader(X_val, y_val, shuffle=False),
        as_loader(X_test, y_test, shuffle=False),
    )


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    patience: int,
) -> tuple[float, int]:
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    final_epoch = 0
    for ep in range(epochs):
        final_epoch = ep + 1
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                total += crit(pred, yb).item() * xb.size(0)
                n += xb.size(0)
        val_loss = total / max(n, 1)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, final_epoch


def predict_torch(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            out.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(out, axis=0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "test_mse": float(mse),
        "test_rmse": float(np.sqrt(mse)),
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_r2": float(r2_score(y_true, y_pred)),
    }


def write_family_outputs(
    *,
    prefix: str,
    arch_name: str,
    grid_results: pd.DataFrame,
    top_metrics: pd.DataFrame,
    test_predictions: dict[str, np.ndarray],
    y_test: np.ndarray,
    test_dates: pd.Series | np.ndarray,
) -> FamilyResult:
    results_dir = config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    preds_df = pd.DataFrame(test_predictions)
    preds_df["Ensemble_Prediction"] = preds_df.values.mean(axis=1)
    preds_df["Actual"] = y_test
    preds_df.insert(0, "Date", pd.to_datetime(test_dates, unit="ms", utc=True) if np.issubdtype(np.asarray(test_dates).dtype, np.integer) else test_dates)

    ens_pred = preds_df["Ensemble_Prediction"].to_numpy()
    ens_metrics = regression_metrics(y_test, ens_pred)

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Total Configurations",
                f"Top-{len(test_predictions)} Used",
                "Test Set Size",
                "Best Single RMSE",
                "Best Single R2",
                "Ensemble RMSE",
                "Ensemble R2",
                "Ensemble MAE",
                "Ensemble MSE",
            ],
            "Value": [
                len(grid_results),
                len(test_predictions),
                len(y_test),
                float(top_metrics["test_rmse"].min()),
                float(top_metrics["test_r2"].max()),
                ens_metrics["test_rmse"],
                ens_metrics["test_r2"],
                ens_metrics["test_mae"],
                ens_metrics["test_mse"],
            ],
        }
    )

    grid_results.to_csv(results_dir / f"{prefix}_grid_search_results.csv", index=False)
    top_metrics.to_csv(results_dir / f"{prefix}_top10_model_metrics.csv", index=False)
    preds_df.to_csv(results_dir / f"{prefix}_predictions.csv", index=False)
    summary_df.to_csv(results_dir / f"{prefix}_summary.csv", index=False)

    return FamilyResult(
        prefix=prefix,
        arch_name=arch_name,
        grid_results=grid_results,
        top_metrics=top_metrics,
        predictions=preds_df,
        summary=summary_df,
    )


def run_torch_family(
    *,
    prefix: str,
    arch_name: str,
    model_ctor: Callable[[dict, int], nn.Module],
    param_grid_spec: dict[str, list[Any]],
    param_keys: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_dates,
    epochs: int = 30,
    patience: int = 6,
    top_k: int = config.TOP_K,
    log: Callable[[str], None] = print,
) -> FamilyResult:
    """Targets are standardized using train-set mean/std so the regression
    head learns on unit-variance signals; predictions are inverse-transformed
    before scoring/saving so the test metrics are in original log-return
    units."""
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-12)
    ystd_train = (y_train - y_mean) / y_std
    ystd_val = (y_val - y_mean) / y_std
    ystd_test = (y_test - y_mean) / y_std
    """Grid search over ``param_grid_spec`` then train top-K and ensemble.

    Each config trains a fresh ``model_ctor(cfg, n_features)`` model on
    ``(X_train, y_train_std)`` and is scored on the val set. The K best
    configs by val MSE are retrained, predictions on the test set are
    inverse-scaled and saved, and an equal-weighted ensemble is computed.
    """
    set_seed()
    n_features = X_train.shape[2]

    combos = list(product(*[param_grid_spec[k] for k in param_keys]))
    log(f"[{arch_name}] grid size: {len(combos)} | n_features={n_features} | device={DEVICE}")

    grid_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for idx, values in enumerate(combos, start=1):
        cfg = dict(zip(param_keys, values))
        try:
            tr, va, _ = make_loaders(
                X_train, ystd_train, X_val, ystd_val, X_test, ystd_test, cfg["batch_size"]
            )
            model = model_ctor(cfg, n_features)
            val_loss, ep = train_torch_model(
                model, tr, va, epochs=epochs, lr=cfg["learning_rate"], patience=patience
            )
            row = {k: cfg[k] for k in param_keys}
            row["val_loss"] = val_loss
            row["epochs_trained"] = ep
            grid_rows.append(row)
        except Exception as exc:
            log(f"[{arch_name}] config {idx} failed: {exc}")
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        if idx % 5 == 0 or idx == len(combos):
            log(f"[{arch_name}] {idx}/{len(combos)} elapsed={time.time()-t0:.1f}s")

    if not grid_rows:
        raise RuntimeError(f"[{arch_name}] grid produced no successful configs")

    grid_df = pd.DataFrame(grid_rows).sort_values("val_loss").reset_index(drop=True)
    top_configs = grid_df.head(top_k).to_dict("records")

    test_preds: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(top_configs, start=1):
        tr, va, te = make_loaders(
            X_train, ystd_train, X_val, ystd_val, X_test, ystd_test, cfg["batch_size"]
        )
        model = model_ctor(cfg, n_features)
        train_torch_model(
            model, tr, va, epochs=epochs, lr=cfg["learning_rate"], patience=patience
        )
        y_pred_std = predict_torch(model, te)
        y_pred = y_pred_std * y_std + y_mean
        name = f"{prefix.upper()}_Model_{i}"
        test_preds[name] = y_pred
        m = regression_metrics(y_test, y_pred)
        m["model_name"] = name
        for k in param_keys:
            m[k] = cfg[k]
        metric_rows.append(m)
        log(f"  {name}: rmse={m['test_rmse']:.6e} r2={m['test_r2']:.4f}")

    metrics_df = pd.DataFrame(metric_rows)
    return write_family_outputs(
        prefix=prefix,
        arch_name=arch_name,
        grid_results=grid_df,
        top_metrics=metrics_df,
        test_predictions=test_preds,
        y_test=y_test,
        test_dates=test_dates,
    )
