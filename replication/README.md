# INDENG 242B — Crypto LOB replication package

Reuses the BTCUSDT order-book dataset built by `src/build_dataset.py` and
applies the model families from **ESE 5460 EC** (oil-price prediction) plus
the backtesting / evaluation toolkit from **INDENG 231 Project 1** (Nasdaq-100
strategy backtest) so each model is judged by both standard regression
metrics and a realistic high-frequency PnL simulation.

## Pipeline

```
data/processed/*.npy
     │
     ▼
replication/data_loader.py  ──►  X_train / X_val / X_test, y_*_reg, raw mid prices
     │
     ▼
replication/models/
   ├── time_series.py    AR / MA / ARIMA on log-return series          (ts_*)
   ├── random_forest.py  RandomForestRegressor on flattened windows    (rf_*)
   ├── lstm.py           PyTorch stacked LSTM                          (lstm_*)
   ├── rnn.py            PyTorch stacked SimpleRNN                     (rnn_*)
   ├── gru_attn.py       PyTorch GRU + Bahdanau attention              (gru_attn_*)
   └── cnn_lstm.py       PyTorch causal-CNN ─► LSTM                    (cnn_lstm_*)
     │                                                              shared 4-file output schema:
     │                                                                _grid_search_results.csv
     │                                                                _top10_model_metrics.csv
     │                                                                _predictions.csv
     │                                                                _summary.csv
     ▼
replication/evaluation/
   ├── regression.py   RMSE / MAE / R² / directional accuracy / F1 / AUC
   ├── backtest.py     INDENG-231-style engine (Sharpe, MaxDD, Calmar, …)
   └── aggregator.py   merged CSV + NAV / drawdown / scatter plots
     │
     ▼
replication/results/        (per-family CSV outputs + combined tables)
replication/outputs/figures (NAV, drawdown, predicted vs actual)
replication/outputs/logs    (replication.log)
```

## Run

```bash
cd "INDENG 242B/INDENG242B-Project"

# Smoke / fast preset (default): small grids, few epochs, runs in ~5–15 min on RTX 5090
python -m replication.main

# Full preset: larger grids, longer training, closer to the ESE 5460 EC sweep
GRID_PRESET=full python -m replication.main
```

The pipeline assumes `data/processed/X_*.npy` exist; if they don't, build
them first:

```bash
python -m src.build_dataset \
    --raw data/raw/btcusdt_lob_raw.parquet \
    --out-dir data/processed
```

## Backtesting conventions

* **Bet cadence** — non-overlapping `horizon`-step bets (10 s each), so the
  realized log-returns of consecutive bets do not share future windows.
* **Position rule** — `+1 / 0 / -1` based on whether the ensemble
  prediction exceeds `±SIGNAL_THRESHOLD`. Threshold matches the
  `dataset_metadata.json` 3-class boundary.
* **Costs** — `COST_BPS` charged on `|Δposition|` per rebalance.
* **Annualization** — uses `PERIODS_PER_YEAR = seconds_per_year / horizon_s`,
  which for 1 Hz × 10 s ≈ 3.15 M periods.
* **Buy-and-hold** — included as a benchmark row.

The 231 backtester’s per-strategy comparison table (Cumulative Return,
Annualized Return, Volatility, Sharpe, Sortino, Max Drawdown, Calmar,
Win Rate, Avg Turnover) maps 1:1 to the columns in
`replication/results/_backtest_metrics.csv`.

## Outputs

After running, expect:

```
replication/results/
    ts_grid_search_results.csv     ts_top10_model_metrics.csv     ts_predictions.csv     ts_summary.csv
    rf_grid_search_results.csv     rf_top10_model_metrics.csv     rf_predictions.csv     rf_summary.csv
    lstm_*                          rnn_*                           gru_attn_*               cnn_lstm_*
    _traditional_metrics.csv       _backtest_metrics.csv          _combined_summary.csv

replication/outputs/figures/nav.png
replication/outputs/figures/drawdown.png
replication/outputs/figures/pred_vs_actual.png
```
