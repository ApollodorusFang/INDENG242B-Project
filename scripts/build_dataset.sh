#!/usr/bin/env bash
# Build supervised LOB dataset from collected raw parquet.
set -euo pipefail

cd "$(dirname "$0")/.."

RAW="${RAW:-data/raw/btcusdt_lob_raw.parquet}"
OUT_DIR="${OUT_DIR:-data/processed}"
LOOKBACK="${LOOKBACK:-60}"
HORIZON="${HORIZON:-10}"
THRESHOLD="${THRESHOLD:-0.00005}"

python -m src.build_dataset \
    --raw "$RAW" \
    --out-dir "$OUT_DIR" \
    --lookback "$LOOKBACK" \
    --horizon "$HORIZON" \
    --threshold "$THRESHOLD"
