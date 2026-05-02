"""Shared helpers for the BTCUSDT LOB pipeline."""
from __future__ import annotations

import logging
import os
from typing import List

DEPTH = 20

BID_PRICE_COLS: List[str] = [f"bid_price_{i}" for i in range(1, DEPTH + 1)]
BID_SIZE_COLS: List[str] = [f"bid_size_{i}" for i in range(1, DEPTH + 1)]
ASK_PRICE_COLS: List[str] = [f"ask_price_{i}" for i in range(1, DEPTH + 1)]
ASK_SIZE_COLS: List[str] = [f"ask_size_{i}" for i in range(1, DEPTH + 1)]

LOB_COLS: List[str] = BID_PRICE_COLS + BID_SIZE_COLS + ASK_PRICE_COLS + ASK_SIZE_COLS
DERIVED_COLS: List[str] = ["mid_price", "spread", "order_book_imbalance_20"]
RAW_COLS: List[str] = ["timestamp"] + LOB_COLS + DERIVED_COLS

# Features used for X. mid_price is excluded so the model cannot trivially read
# the label; timestamp is excluded because it is not a feature.
FEATURE_COLS: List[str] = LOB_COLS + ["spread", "order_book_imbalance_20"]


def setup_logger(name: str, log_path: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def compute_derived(bid_prices, bid_sizes, ask_prices, ask_sizes):
    """Return (mid_price, spread, imbalance_20) from top-DEPTH arrays."""
    best_bid = bid_prices[0]
    best_ask = ask_prices[0]
    mid_price = 0.5 * (best_bid + best_ask)
    spread = best_ask - best_bid
    bid_sum = float(sum(bid_sizes))
    ask_sum = float(sum(ask_sizes))
    denom = bid_sum + ask_sum
    imbalance = (bid_sum - ask_sum) / denom if denom > 0 else 0.0
    return mid_price, spread, imbalance
