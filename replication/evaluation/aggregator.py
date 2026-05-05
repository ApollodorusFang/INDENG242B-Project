"""Cross-family comparison: merges traditional regression metrics with the
backtest summary into a single comparison table and emits NAV / drawdown
plots for the report.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config
from .backtest import BacktestRecord


def merged_summary(
    traditional: pd.DataFrame,
    backtest: pd.DataFrame,
) -> pd.DataFrame:
    t = traditional.set_index("family")
    out = backtest.join(t, how="left")
    out.to_csv(config.RESULTS_DIR / "_combined_summary.csv")
    return out


def plot_nav(records: dict[str, BacktestRecord], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for family, rec in records.items():
        ax.plot(rec.nav.index, rec.nav.values, label=family, linewidth=1.4)
    ax.set_title("Backtest NAV — BTCUSDT 10s horizon, non-overlapping bets")
    ax.set_ylabel("NAV (initial = 1.0)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_drawdown(records: dict[str, BacktestRecord], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    for family, rec in records.items():
        peak = rec.nav.cummax()
        dd = rec.nav / peak - 1.0
        ax.plot(dd.index, dd.values, label=family, linewidth=1.0)
    ax.fill_between(dd.index, dd.values, 0, alpha=0.0)
    ax.set_title("Backtest drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_predictions_vs_actual(out_path: Path) -> None:
    """Scatter-style plot of predicted vs realized log-return per family
    (down-sampled for readability)."""
    import glob
    pred_files = sorted(glob.glob(str(config.RESULTS_DIR / "*_predictions.csv")))
    n = len(pred_files)
    if n == 0:
        return
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    for ax, path in zip(axes.flat, pred_files):
        df = pd.read_csv(path)
        x = df["Actual"].to_numpy(dtype=float)
        y = df["Ensemble_Prediction"].to_numpy(dtype=float)
        if len(x) > 4000:
            sl = np.random.default_rng(0).choice(len(x), 4000, replace=False)
            x, y = x[sl], y[sl]
        ax.scatter(x, y, s=4, alpha=0.4)
        lim = float(np.max(np.abs([x.min(), x.max(), y.min(), y.max()])))
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=0.8)
        ax.set_xlabel("realized")
        ax.set_ylabel("predicted")
        ax.set_title(Path(path).stem.replace("_predictions", ""))
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
