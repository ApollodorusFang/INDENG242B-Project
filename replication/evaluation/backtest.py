"""High-frequency backtester for BTCUSDT.

Each test sample ``i`` carries:
  * ``ts_i``           — server timestamp at decision time
  * ``mid_t``          — mid_price at decision time
  * ``mid_t_plus_h``   — mid_price ``horizon`` steps later
  * ``y_test_reg[i]``  — realized log return over the horizon
  * ``Ensemble_Pred``  — predicted log return over the horizon

We translate model predictions into a **non-overlapping** sequence of
horizon-length bets. From the test set we keep samples at strides of
``horizon`` so that consecutive bets do not share return windows. The
position rule is:

    pos = +1   if pred >  +threshold
    pos = -1   if pred <  -threshold
    pos =  0   otherwise

Net per-bet log-return is ``pos * realized``. Transaction cost
``cost_bps`` is charged on changes in position (turnover ``|pos_t - pos_{t-1}|``,
0..2 per flip), expressed in arithmetic returns.

Outputs a standard summary (Cumulative Return, Sharpe, Sortino, Max Drawdown,
Calmar, Win Rate, etc.).
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config


@dataclass
class BacktestRecord:
    family: str
    nav: pd.Series
    arith_returns: pd.Series
    positions: pd.Series
    turnover: pd.Series
    metrics: dict[str, float]


def _annualize_factor() -> float:
    return float(config.PERIODS_PER_YEAR)


def _compute_metrics(nav: pd.Series, arith_ret: pd.Series, turnover: pd.Series) -> dict[str, float]:
    af = _annualize_factor()
    if len(arith_ret) == 0:
        return {k: 0.0 for k in [
            "CumulativeReturn", "AnnualizedReturn", "AnnualizedVolatility",
            "Sharpe", "Sortino", "MaxDrawdown", "Calmar", "WinRate", "AvgTurnover",
        ]}

    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    n = len(arith_ret)
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (af / max(n, 1)) - 1.0)
    vol = float(arith_ret.std(ddof=0) * np.sqrt(af))
    if arith_ret.std(ddof=0) == 0 or n < 2:
        sharpe = 0.0
    else:
        sharpe = float(arith_ret.mean() / arith_ret.std(ddof=0) * np.sqrt(af))
    downside = arith_ret[arith_ret < 0]
    if len(downside) == 0 or downside.std(ddof=0) == 0:
        sortino = 0.0
    else:
        sortino = float(arith_ret.mean() / downside.std(ddof=0) * np.sqrt(af))

    peak = nav.cummax()
    dd = nav / peak - 1.0
    mdd = float(dd.min())
    calmar = float(cagr / abs(mdd)) if mdd != 0 else 0.0
    nz = arith_ret[arith_ret != 0]
    win_rate = float((nz > 0).mean()) if len(nz) else 0.0

    return {
        "CumulativeReturn": total,
        "AnnualizedReturn": cagr,
        "AnnualizedVolatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDrawdown": mdd,
        "Calmar": calmar,
        "WinRate": win_rate,
        "AvgTurnover": float(turnover.mean()),
    }


def backtest_predictions(
    *,
    family: str,
    pred_csv: Path,
    test_timestamps_ms: np.ndarray,
    test_y_reg: np.ndarray,
    horizon_stride: int = config.HORIZON_STEPS,
    threshold: float = config.SIGNAL_THRESHOLD,
    cost_bps: float = config.COST_BPS,
    allow_short: bool = True,
) -> BacktestRecord:
    df = pd.read_csv(pred_csv)
    pred = df["Ensemble_Prediction"].to_numpy(dtype=float)
    if len(pred) != len(test_y_reg):
        raise ValueError(
            f"{family}: ensemble length {len(pred)} != test set length {len(test_y_reg)}"
        )

    # Stride to non-overlapping horizon windows.
    idx = np.arange(0, len(pred), horizon_stride)
    pred_s = pred[idx]
    real_s = test_y_reg[idx]
    ts_s = test_timestamps_ms[idx]

    pos = np.where(pred_s > threshold, 1.0,
          np.where(pred_s < -threshold, -1.0 if allow_short else 0.0, 0.0))
    log_ret = pos * real_s
    arith_ret = np.expm1(log_ret)

    prev_pos = np.concatenate([[0.0], pos[:-1]])
    turnover = np.abs(pos - prev_pos)
    cost = turnover * (cost_bps / 1e4)
    net_ret = arith_ret - cost

    nav = (1.0 + pd.Series(net_ret)).cumprod()
    nav.iloc[0] = (1.0 + net_ret[0])  # explicit first entry
    dt_index = pd.to_datetime(ts_s, unit="ms", utc=True)

    nav.index = dt_index
    arith = pd.Series(net_ret, index=dt_index)
    pos_series = pd.Series(pos, index=dt_index)
    turn_series = pd.Series(turnover, index=dt_index)

    metrics = _compute_metrics(nav, arith, turn_series)
    metrics["family"] = family
    metrics["n_bets"] = int(len(pos))
    metrics["pct_long"] = float((pos > 0).mean())
    metrics["pct_short"] = float((pos < 0).mean())
    metrics["pct_flat"] = float((pos == 0).mean())

    return BacktestRecord(
        family=family,
        nav=nav,
        arith_returns=arith,
        positions=pos_series,
        turnover=turn_series,
        metrics=metrics,
    )


def buy_and_hold_record(
    *,
    test_timestamps_ms: np.ndarray,
    test_mid_t: np.ndarray,
    test_mid_t_plus_h: np.ndarray,
    horizon_stride: int = config.HORIZON_STEPS,
    cost_bps: float = config.COST_BPS,
) -> BacktestRecord:
    idx = np.arange(0, len(test_timestamps_ms), horizon_stride)
    real_log = np.log(test_mid_t_plus_h[idx] / test_mid_t[idx])
    arith = np.expm1(real_log)
    arith[0] -= cost_bps / 1e4  # one-time entry cost
    nav = (1.0 + pd.Series(arith)).cumprod()
    dt_index = pd.to_datetime(test_timestamps_ms[idx], unit="ms", utc=True)
    nav.index = dt_index
    arith_s = pd.Series(arith, index=dt_index)
    turnover = pd.Series(np.zeros_like(arith), index=dt_index)
    turnover.iloc[0] = 1.0
    metrics = _compute_metrics(nav, arith_s, turnover)
    metrics["family"] = "buy_and_hold"
    metrics["n_bets"] = int(len(arith))
    metrics["pct_long"] = 1.0
    metrics["pct_short"] = 0.0
    metrics["pct_flat"] = 0.0
    return BacktestRecord(
        family="buy_and_hold",
        nav=nav,
        arith_returns=arith_s,
        positions=pd.Series(np.ones_like(arith), index=dt_index),
        turnover=turnover,
        metrics=metrics,
    )


def run_all(
    *,
    test_timestamps_ms: np.ndarray,
    test_y_reg: np.ndarray,
    test_mid_t: np.ndarray,
    test_mid_t_plus_h: np.ndarray,
    log=print,
) -> tuple[pd.DataFrame, dict[str, BacktestRecord]]:
    """Costed + frictionless backtest sweep over every family that has
    written a ``*_predictions.csv``."""
    pred_files = sorted(glob.glob(str(config.RESULTS_DIR / "*_predictions.csv")))
    records: dict[str, BacktestRecord] = {}

    bh = buy_and_hold_record(
        test_timestamps_ms=test_timestamps_ms,
        test_mid_t=test_mid_t,
        test_mid_t_plus_h=test_mid_t_plus_h,
        cost_bps=config.COST_BPS,
    )
    records["buy_and_hold"] = bh

    for path in pred_files:
        prefix = Path(path).stem.replace("_predictions", "")
        rec = backtest_predictions(
            family=prefix,
            pred_csv=Path(path),
            test_timestamps_ms=test_timestamps_ms,
            test_y_reg=test_y_reg,
            cost_bps=config.COST_BPS,
        )
        records[prefix] = rec
        log(
            f"[BT cost={config.COST_BPS}bps {prefix}] "
            f"Sharpe={rec.metrics['Sharpe']:.3f} "
            f"CumRet={rec.metrics['CumulativeReturn']:.4f} "
            f"MaxDD={rec.metrics['MaxDrawdown']:.4f} "
            f"Bets={rec.metrics['n_bets']}"
        )

    # Second pass: frictionless backtest to isolate model alpha.
    frictionless_rows = []
    for path in pred_files:
        prefix = Path(path).stem.replace("_predictions", "")
        rec0 = backtest_predictions(
            family=prefix,
            pred_csv=Path(path),
            test_timestamps_ms=test_timestamps_ms,
            test_y_reg=test_y_reg,
            cost_bps=0.0,
        )
        m0 = {f"NoCost_{k}": v for k, v in rec0.metrics.items() if k not in ("family", "n_bets", "pct_long", "pct_short", "pct_flat")}
        m0["family"] = prefix
        frictionless_rows.append(m0)

    rows = [r.metrics for r in records.values()]
    summary = pd.DataFrame(rows).set_index("family")
    fric_df = pd.DataFrame(frictionless_rows).set_index("family")
    summary = summary.join(fric_df, how="left")
    summary.to_csv(config.RESULTS_DIR / "_backtest_metrics.csv")
    return summary, records
