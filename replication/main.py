"""End-to-end driver.

Pipeline
--------
1. Load processed LOB tensors and raw mid-price.
2. Run six model families (Time Series, Random Forest, LSTM, RNN, GRU+Attn,
   CNN-LSTM); each writes the four-file result schema.
3. Compute traditional regression + directional metrics.
4. Run the high-frequency backtester with INDENG-231-style metrics.
5. Produce combined comparison tables + NAV / drawdown / scatter plots.

Usage
-----
    cd INDENG\ 242B/INDENG242B-Project
    python -m replication.main                # fast preset (default)
    GRID_PRESET=full python -m replication.main

The "fast" preset shrinks each grid so the whole pipeline finishes in a few
minutes on an RTX 5090; "full" expands grids closer to the ESE 5460 EC sweep.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from . import config
from .data_loader import load_dataset
from .evaluation import aggregator, backtest, regression
from .models import cnn_lstm, gru_attn, lstm, random_forest, rnn, time_series


def setup_logging() -> logging.Logger:
    log_dir = config.OUTPUTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "replication.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("replication")


def main() -> None:
    log = setup_logging()
    log.info("=" * 70)
    log.info("Replication pipeline (preset=%s)", config.GRID_PRESET)
    log.info("=" * 70)

    ds = load_dataset()
    log.info(
        "dataset: train=%d val=%d test=%d  lookback=%d horizon=%d  features=%d",
        len(ds.X_train), len(ds.X_val), len(ds.X_test),
        ds.lookback, ds.horizon, ds.num_features,
    )

    test_dates = ds.test_timestamps_ms

    log.info("--- Time Series (ARIMA / AR / MA) ---")
    raw_mid = _raw_mid_series()
    n_tr = ds.metadata["split_sizes"]["train"]
    n_va = ds.metadata["split_sizes"]["val"]
    n_te = ds.metadata["split_sizes"]["test"]
    time_series.run(
        raw_mid=raw_mid,
        n_train=n_tr, n_val=n_va, n_test=n_te,
        lookback=ds.lookback, horizon=ds.horizon,
        y_val_reg=ds.y_val_reg, y_test_reg=ds.y_test_reg,
        test_dates=test_dates,
        log=log.info,
    )

    log.info("--- Random Forest ---")
    random_forest.run(
        ds.X_train, ds.y_train_reg,
        ds.X_val, ds.y_val_reg,
        ds.X_test, ds.y_test_reg,
        test_dates=test_dates,
        log=log.info,
    )

    log.info("--- LSTM ---")
    lstm.run(
        ds.X_train, ds.y_train_reg,
        ds.X_val, ds.y_val_reg,
        ds.X_test, ds.y_test_reg,
        test_dates=test_dates,
        log=log.info,
    )

    log.info("--- SimpleRNN ---")
    rnn.run(
        ds.X_train, ds.y_train_reg,
        ds.X_val, ds.y_val_reg,
        ds.X_test, ds.y_test_reg,
        test_dates=test_dates,
        log=log.info,
    )

    log.info("--- GRU + Attention ---")
    gru_attn.run(
        ds.X_train, ds.y_train_reg,
        ds.X_val, ds.y_val_reg,
        ds.X_test, ds.y_test_reg,
        test_dates=test_dates,
        log=log.info,
    )

    log.info("--- CNN-LSTM ---")
    cnn_lstm.run(
        ds.X_train, ds.y_train_reg,
        ds.X_val, ds.y_val_reg,
        ds.X_test, ds.y_test_reg,
        test_dates=test_dates,
        log=log.info,
    )

    log.info("--- Traditional metrics ---")
    trad = regression.evaluate_all()
    log.info("\n%s", trad.round(6).to_string(index=False))

    log.info("--- Backtest ---")
    summary, records = backtest.run_all(
        test_timestamps_ms=ds.test_timestamps_ms,
        test_y_reg=ds.y_test_reg,
        test_mid_t=ds.test_mid_t,
        test_mid_t_plus_h=ds.test_mid_t_plus_h,
        log=log.info,
    )
    log.info("\n%s", summary.round(6).to_string())

    combined = aggregator.merged_summary(trad, summary)
    log.info("--- Combined ---\n%s", combined.round(6).to_string())

    fig_dir = config.OUTPUTS_DIR / "figures"
    aggregator.plot_nav(records, fig_dir / "nav.png")
    aggregator.plot_drawdown(records, fig_dir / "drawdown.png")
    aggregator.plot_predictions_vs_actual(fig_dir / "pred_vs_actual.png")
    log.info("Figures written to %s", fig_dir)


def _raw_mid_series() -> np.ndarray:
    import pandas as pd
    raw_path = config.RAW_DIR / "btcusdt_lob_raw.parquet"
    df = pd.read_parquet(raw_path)
    df = (
        df.drop_duplicates(subset=["timestamp"])
          .sort_values("timestamp")
          .reset_index(drop=True)
    )
    df = df.loc[df["mid_price"] > 0].reset_index(drop=True)
    return df["mid_price"].to_numpy(dtype=np.float64)


if __name__ == "__main__":
    main()
