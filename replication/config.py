"""Shared configuration for the replication package.

Edit ``GRID_PRESET`` to switch between fast smoke-test grids and the full
grids that mirror the ESE 5460 EC search spaces.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

REPLICATION_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPLICATION_ROOT / "results"
OUTPUTS_DIR = REPLICATION_ROOT / "outputs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "logs").mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 1 Hz sampling, 10-step (10 s) prediction horizon. Crypto trades 24/7.
SAMPLE_INTERVAL_S = 1.0
HORIZON_STEPS = 10
SECONDS_PER_YEAR = 365.25 * 24 * 3600
PERIODS_PER_YEAR = SECONDS_PER_YEAR / (SAMPLE_INTERVAL_S * HORIZON_STEPS)

# Trading parameters used by the backtester. ``COST_BPS`` is per-side in bps.
# At 1 Hz / 10 s horizon, |log return| std is ~2e-4; INDENG 231's daily-style
# 2 bps would dominate the signal. We use a HF-realistic 1 bp default and
# also report a frictionless backtest for separating model alpha from costs.
COST_BPS = 1.0
SIGNAL_THRESHOLD = 5e-5  # matches the 3-class label threshold

# Top-K used by the ensemble (matches ESE 5460 EC convention).
TOP_K = 10

# Toggle this to "fast" for quick iteration; "full" to match ESE 5460 EC grids.
GRID_PRESET = os.environ.get("GRID_PRESET", "fast").lower()
