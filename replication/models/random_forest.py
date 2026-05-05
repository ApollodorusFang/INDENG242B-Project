"""Random Forest regressor on flattened LOB windows.

Small grid search over n_estimators / max_depth / min_samples_leaf, top-K
models by val MSE, equal-weighted ensemble on the test set, written to the
unified four-CSV result schema.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .. import config
from .base import FamilyResult, regression_metrics, write_family_outputs


_FAST_GRID = {
    "n_estimators":     [100, 200],
    "max_depth":        [None, 12],
    "min_samples_leaf": [1, 5],
    "max_features":     ["sqrt"],
}
_FULL_GRID = {
    "n_estimators":     [100, 200, 400],
    "max_depth":        [None, 8, 12, 20],
    "min_samples_leaf": [1, 2, 5],
    "max_features":     ["sqrt", 0.3],
}


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def run(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_dates,
    *,
    log=print,
) -> FamilyResult:
    grid = _FULL_GRID if config.GRID_PRESET == "full" else _FAST_GRID
    keys = ["n_estimators", "max_depth", "min_samples_leaf", "max_features"]
    combos = list(product(*[grid[k] for k in keys]))
    log(f"[RandomForest] grid size: {len(combos)}")

    Xtr = _flatten(X_train)
    Xva = _flatten(X_val)
    Xte = _flatten(X_test)

    grid_rows = []
    for idx, values in enumerate(combos, start=1):
        cfg = dict(zip(keys, values))
        model = RandomForestRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            max_features=cfg["max_features"],
            n_jobs=-1,
            random_state=config.SEED,
        )
        model.fit(Xtr, y_train)
        val_pred = model.predict(Xva)
        val_loss = float(np.mean((val_pred - y_val) ** 2))
        row = {**cfg, "val_loss": val_loss, "epochs_trained": 1}
        grid_rows.append(row)
        log(f"  [RF] {idx}/{len(combos)} val_mse={val_loss:.6e}")

    # Sort the original config dicts (pandas would coerce None/ints to NaN/float).
    grid_rows_sorted = sorted(grid_rows, key=lambda r: r["val_loss"])
    grid_df = pd.DataFrame(grid_rows_sorted).reset_index(drop=True)
    top_configs = grid_rows_sorted[: config.TOP_K]

    test_preds: dict[str, np.ndarray] = {}
    metric_rows = []
    for i, cfg in enumerate(top_configs, start=1):
        model = RandomForestRegressor(
            n_estimators=int(cfg["n_estimators"]),
            max_depth=cfg["max_depth"] if cfg["max_depth"] is None else int(cfg["max_depth"]),
            min_samples_leaf=int(cfg["min_samples_leaf"]),
            max_features=cfg["max_features"],
            n_jobs=-1,
            random_state=config.SEED + i,  # diversify ensemble members
        )
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        name = f"RF_Model_{i}"
        test_preds[name] = y_pred
        m = regression_metrics(y_test, y_pred)
        m["model_name"] = name
        for k in keys:
            m[k] = cfg[k]
        metric_rows.append(m)
        log(f"  {name}: rmse={m['test_rmse']:.6e} r2={m['test_r2']:.4f}")

    metrics_df = pd.DataFrame(metric_rows)
    return write_family_outputs(
        prefix="rf",
        arch_name="RandomForest",
        grid_results=grid_df,
        top_metrics=metrics_df,
        test_predictions=test_preds,
        y_test=y_test,
        test_dates=test_dates,
    )
