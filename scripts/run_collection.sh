#!/usr/bin/env bash
# Collect 6 hours of BTCUSDT top-20 LOB snapshots from Binance.
set -euo pipefail

cd "$(dirname "$0")/.."

DURATION_HOURS="${DURATION_HOURS:-6}"
SYMBOL="${SYMBOL:-BTCUSDT}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1.0}"
OUT="${OUT:-data/raw/btcusdt_lob_raw.parquet}"

python -m src.collect_orderbook \
    --symbol "$SYMBOL" \
    --duration-hours "$DURATION_HOURS" \
    --sample-interval "$SAMPLE_INTERVAL" \
    --out "$OUT"
