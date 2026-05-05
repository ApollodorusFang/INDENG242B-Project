"""Classical time-series baselines (AR / ARMA / ARIMA on log mid-price returns).

Adapted to the high-frequency LOB data and tuned for tractable runtime:

* Each candidate model is fit *once* on the train portion of the 1-step
  log-return series of mid_price.
* For every val / test sample at row ``t = i + lookback - 1`` we form the
  predicted cumulative ``horizon``-step log return by recursively applying
  the fitted AR/ARMA coefficients to a sliding window of the most recent
  observations. This avoids the per-step ``ARIMA.append`` / Kalman update
  cost and lets the family run end-to-end in seconds rather than hours.
* The naive zero-forecast (random-walk / martingale) baseline is included
  to establish a no-skill reference.

The four-file output schema matches every other family.
"""
from __future__ import annotations

import os as _os
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from .. import config
from .base import FamilyResult, regression_metrics, write_family_outputs

# Cap BLAS threads — the statsmodels ARIMA fit otherwise pegs every core for
# tiny operations and dominates wall time.
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
try:  # pragma: no cover
    import threadpoolctl
    threadpoolctl.threadpool_limits(1)
except ImportError:  # pragma: no cover
    pass


_FAST_SPECS: list[dict[str, Any]] = [
    {"label": "Naive(0)",   "kind": "zero"},
    {"label": "AR(1)",      "kind": "arma", "p": 1, "q": 0},
    {"label": "AR(2)",      "kind": "arma", "p": 2, "q": 0},
    {"label": "AR(5)",      "kind": "arma", "p": 5, "q": 0},
    {"label": "ARMA(1,1)",  "kind": "arma", "p": 1, "q": 1},
    {"label": "ARMA(2,2)",  "kind": "arma", "p": 2, "q": 2},
]
_FULL_SPECS: list[dict[str, Any]] = [
    {"label": "Naive(0)",   "kind": "zero"},
    *(
        {"label": f"AR({p})",   "kind": "arma", "p": p, "q": 0}
        for p in (1, 2, 3, 5, 10)
    ),
    *(
        {"label": f"ARMA({p},{q})", "kind": "arma", "p": p, "q": q}
        for p in (1, 2, 3) for q in (1, 2)
    ),
]


def _fit_arma(returns: np.ndarray, p: int, q: int):
    """Fit ARMA(p, q) on the returns series (d=0). Single fit, no walk-forward."""
    return ARIMA(returns, order=(p, 0, q), trend="n").fit(
        method_kwargs={"warn_convergence": False}
    )


def _arma_coeffs(fit_obj) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract AR / MA polynomial coefficients and residual std from a fitted ARMA."""
    ar = np.asarray(fit_obj.arparams, dtype=np.float64).ravel()
    ma = np.asarray(fit_obj.maparams, dtype=np.float64).ravel()
    sigma2 = float(getattr(fit_obj, "params_variance", 0.0)) if hasattr(fit_obj, "params_variance") else 0.0
    return ar, ma, sigma2


def _h_step_forecast_arma(
    ar: np.ndarray,
    ma: np.ndarray,
    last_returns: np.ndarray,
    last_innovations: np.ndarray,
    horizon: int,
) -> float:
    """Recursive ``horizon``-step ahead forecast for ARMA(p, q).

    Uses the convention ``r_t = sum_i ar_i * r_{t-i} + sum_j ma_j * eps_{t-j} + eps_t``.
    Future innovations are zero in the conditional-mean forecast; future
    returns are replaced by their predicted values. Returns the *cumulative*
    h-step log return.
    """
    p = len(ar)
    q = len(ma)
    # FIFO buffers ordered most-recent-first.
    r_hist = list(last_returns[-p:][::-1]) if p > 0 else []
    e_hist = list(last_innovations[-q:][::-1]) if q > 0 else []
    total = 0.0
    for _ in range(horizon):
        ar_part = sum(ar[i] * r_hist[i] for i in range(p)) if p > 0 else 0.0
        ma_part = sum(ma[j] * e_hist[j] for j in range(q)) if q > 0 else 0.0
        r_hat = ar_part + ma_part  # eps_t = 0 in expectation
        total += r_hat
        if p > 0:
            r_hist.insert(0, r_hat)
            r_hist.pop()
        if q > 0:
            e_hist.insert(0, 0.0)
            e_hist.pop()
    return float(total)


def _zero_forecasts(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)


def _arma_walk_forecasts(
    fit_obj,
    returns: np.ndarray,
    start_idx: int,
    n_steps: int,
    horizon: int,
) -> np.ndarray:
    """For each i in [0, n_steps), forecast cumulative h-step return starting
    from row ``start_idx + i`` of ``returns`` using the fixed ARMA fit.
    """
    ar, ma, _ = _arma_coeffs(fit_obj)
    p = len(ar)
    q = len(ma)
    # Compute residuals across the entire series under the fitted coefficients.
    residuals = np.asarray(fit_obj.resid, dtype=np.float64)
    if len(residuals) < len(returns):
        # Pad with zeros at the front so indices align with `returns`.
        residuals = np.concatenate([np.zeros(len(returns) - len(residuals)), residuals])

    out = np.empty(n_steps, dtype=np.float64)
    for i in range(n_steps):
        t = start_idx + i
        r_window = returns[max(0, t - p) : t]
        e_window = residuals[max(0, t - q) : t]
        # Pad if the window doesn't cover p / q yet.
        if p > 0 and len(r_window) < p:
            r_window = np.concatenate([np.zeros(p - len(r_window)), r_window])
        if q > 0 and len(e_window) < q:
            e_window = np.concatenate([np.zeros(q - len(e_window)), e_window])
        out[i] = _h_step_forecast_arma(ar, ma, r_window, e_window, horizon)
    return out


def run(
    *,
    raw_mid: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    lookback: int,
    horizon: int,
    y_val_reg: np.ndarray,
    y_test_reg: np.ndarray,
    test_dates,
    log=print,
) -> FamilyResult:
    """Classical baselines on the log mid-price return series."""
    grid = _FULL_SPECS if config.GRID_PRESET == "full" else _FAST_SPECS
    log(f"[TimeSeries] grid size: {len(grid)}")

    log_mid = np.log(raw_mid)
    returns = np.diff(log_mid)

    # Sample i is the row at t = i + lookback - 1; cumulative return from
    # row t to row t+horizon equals returns[t : t+horizon].sum().
    val_start_t = (n_train + 0) + (lookback - 1)
    test_start_t = (n_train + n_val) + (lookback - 1)

    grid_rows: list[dict[str, Any]] = []
    val_preds_per: dict[str, np.ndarray] = {}
    test_preds_per: dict[str, np.ndarray] = {}

    for cfg_spec in grid:
        label = cfg_spec["label"]
        try:
            if cfg_spec["kind"] == "zero":
                val_preds = _zero_forecasts(n_val)
                test_preds = _zero_forecasts(n_test)
            else:
                p, q = cfg_spec["p"], cfg_spec["q"]
                fit_obj = _fit_arma(returns[:val_start_t], p=p, q=q)
                val_preds = _arma_walk_forecasts(
                    fit_obj, returns, val_start_t, n_val, horizon
                )
                test_preds = _arma_walk_forecasts(
                    fit_obj, returns, test_start_t, n_test, horizon
                )
            val_mse = float(mean_squared_error(y_val_reg, val_preds))
        except Exception as exc:
            log(f"  [TS] {label} failed: {exc}")
            continue

        row = {"label": label, "kind": cfg_spec["kind"],
               "p": cfg_spec.get("p", 0), "q": cfg_spec.get("q", 0),
               "val_loss": val_mse, "epochs_trained": 1}
        grid_rows.append(row)
        val_preds_per[label] = val_preds
        test_preds_per[label] = test_preds
        log(f"  [TS] {label}: val_mse={val_mse:.6e}")

    if not grid_rows:
        raise RuntimeError("[TimeSeries] no successful configurations")

    grid_rows_sorted = sorted(grid_rows, key=lambda r: r["val_loss"])
    grid_df = pd.DataFrame(grid_rows_sorted).reset_index(drop=True)
    top_labels = [r["label"] for r in grid_rows_sorted[: config.TOP_K]]

    test_preds_named: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    for i, label in enumerate(top_labels, start=1):
        y_pred = test_preds_per[label]
        name = f"TS_Model_{i}"
        test_preds_named[name] = y_pred
        m = regression_metrics(y_test_reg, y_pred)
        m["model_name"] = name
        m["spec"] = label
        metric_rows.append(m)
        log(f"  {name} ({label}): rmse={m['test_rmse']:.6e} r2={m['test_r2']:.4f}")

    metrics_df = pd.DataFrame(metric_rows)
    return write_family_outputs(
        prefix="ts",
        arch_name="TimeSeries-ARIMA",
        grid_results=grid_df,
        top_metrics=metrics_df,
        test_predictions=test_preds_named,
        y_test=y_test_reg,
        test_dates=test_dates,
    )
