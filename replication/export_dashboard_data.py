"""Export dashboard data — self-contained.

Reads ONLY from ``replication/results/*_predictions.csv`` and (optionally)
``data/raw/btcusdt_lob_raw.parquet``. It does **not** require the
``data/processed/*.npy`` tensors, which are heavyweight intermediates not
typically committed to the repo.

Run from repo root:
    python -m replication.export_dashboard_data

Output:
    dashboard/data/test_predictions.csv

The CSV column schema the dashboard expects:
    timestamp_ms · realized_log_return · pred_ts · pred_rf · pred_rnn ·
    pred_lstm · pred_cnn_lstm · pred_gru_attn
    (mid_price_t / mid_price_t_plus_h are added if the raw parquet is present)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "replication" / "results"
RAW_PARQUET = REPO_ROOT / "data" / "raw" / "btcusdt_lob_raw.parquet"
PROCESSED_META = REPO_ROOT / "data" / "processed" / "dataset_metadata.json"
DASHBOARD_DATA_DIR = REPO_ROOT / "dashboard" / "data"

FAMILIES = ["ts", "rf", "rnn", "lstm", "cnn_lstm", "gru_attn"]

# Common column names; we match case-insensitively.
PRED_CANDIDATES   = ["predicted", "pred", "y_pred", "ensemble", "forecast",
                     "prediction", "y_hat", "yhat", "ensemble_pred"]
ACTUAL_CANDIDATES = ["actual", "y_true", "realized", "y_actual", "true",
                     "target", "y_test", "actuals"]
TIME_CANDIDATES   = ["timestamp_ms", "timestamp", "date", "time", "ts",
                     "datetime", "test_date", "test_timestamp"]


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Case-insensitive lookup of the first matching column name."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def coerce_to_ms(values) -> np.ndarray:
    """Convert any timestamp-ish series to int64 milliseconds since epoch."""
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer):
        sample = float(arr.flat[0]) if arr.size else 0.0
        if sample > 1e15:                  # looks like nanoseconds
            return (arr // 10**6).astype(np.int64)
        if sample > 1e11:                  # already milliseconds
            return arr.astype(np.int64)
        if sample > 1e8:                   # seconds
            return (arr * 1000).astype(np.int64)
    # Strings, datetime, float, etc. Force ms resolution explicitly because
    # pandas 2.x can pick datetime64[s] when parsing whole-second strings,
    # whose .astype("int64") returns *seconds* rather than nanoseconds.
    dt64 = pd.to_datetime(values).to_numpy().astype("datetime64[ms]")
    return dt64.astype(np.int64)


def main() -> int:
    if not RESULTS_DIR.exists():
        print(f"[error] {RESULTS_DIR} not found", file=sys.stderr)
        return 1

    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    per_family_preds: dict[str, np.ndarray] = {}
    actual_arr: np.ndarray | None = None
    actual_source: tuple[str, str] | None = None
    time_arr: np.ndarray | None = None
    time_source: tuple[str, str] | None = None
    actual_lower = {c.lower() for c in ACTUAL_CANDIDATES}

    for fam in FAMILIES:
        path = RESULTS_DIR / f"{fam}_predictions.csv"
        if not path.exists():
            print(f"[skip] {path.name} — file not found", file=sys.stderr)
            continue

        df = pd.read_csv(path)
        print(f"[read] {path.name}: shape={df.shape}, columns={list(df.columns)}",
              file=sys.stderr)

        # --- prediction column ------------------------------------------------
        pcol = find_col(df, PRED_CANDIDATES)
        if pcol is None:
            # Heuristic: last numeric column that isn't an "actual" column.
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            num_cols = [c for c in num_cols if c.lower() not in actual_lower]
            if num_cols:
                pcol = num_cols[-1]
                print(f"       [heuristic] using '{pcol}' as the prediction column",
                      file=sys.stderr)
        if pcol is None:
            print(f"       [skip] could not find a prediction column", file=sys.stderr)
            continue
        per_family_preds[fam] = df[pcol].to_numpy(dtype=float)
        print(f"       prediction → '{pcol}'  (n={len(per_family_preds[fam])})",
              file=sys.stderr)

        # --- actual column (only need one valid source) ----------------------
        if actual_arr is None:
            acol = find_col(df, ACTUAL_CANDIDATES)
            if acol is not None:
                actual_arr = df[acol].to_numpy(dtype=float)
                actual_source = (fam, acol)
                print(f"       [picked] actual → '{acol}' from {fam}",
                      file=sys.stderr)

        # --- timestamp column (only need one valid source) -------------------
        if time_arr is None:
            tcol = find_col(df, TIME_CANDIDATES)
            if tcol is not None:
                try:
                    time_arr = coerce_to_ms(df[tcol].values)
                    time_source = (fam, tcol)
                    print(f"       [picked] timestamp → '{tcol}' from {fam}",
                          file=sys.stderr)
                except Exception as exc:
                    print(f"       [warn] couldn't parse '{tcol}' as time: {exc}",
                          file=sys.stderr)

    if not per_family_preds:
        print("\n[error] No usable predictions CSV found.", file=sys.stderr)
        return 1

    if actual_arr is None:
        print("\n[error] None of the predictions CSVs had a recognisable 'actual' column.",
              file=sys.stderr)
        print("        Tried any of: " + ", ".join(ACTUAL_CANDIDATES), file=sys.stderr)
        print("        Open one of the CSVs and check the column name — then add it",
              file=sys.stderr)
        print("        to ACTUAL_CANDIDATES at the top of this script.", file=sys.stderr)
        return 1

    n = len(actual_arr)

    out = pd.DataFrame({"realized_log_return": actual_arr})
    if time_arr is not None:
        out["timestamp_ms"] = time_arr[:n].astype(np.int64)
    else:
        # Synthetic 1 Hz axis so the plots still render with a sensible x-axis.
        base_ms = 1_715_000_000_000
        out["timestamp_ms"] = (base_ms + np.arange(n) * 1000).astype(np.int64)
        print("[info] no timestamp column found in any CSV — using a synthetic 1 Hz axis",
              file=sys.stderr)

    for fam, vals in per_family_preds.items():
        col = np.full(n, np.nan)
        m = min(n, len(vals))
        col[:m] = vals[:m]
        out[f"pred_{fam}"] = col

    # --- optional: attach mid_price from raw parquet -------------------------
    if RAW_PARQUET.exists():
        try:
            raw = pd.read_parquet(RAW_PARQUET, columns=["timestamp", "mid_price"])
            raw = (
                raw.drop_duplicates(subset=["timestamp"])
                   .sort_values("timestamp")
                   .reset_index(drop=True)
            )
            raw = raw.loc[raw["mid_price"] > 0].reset_index(drop=True)

            lookback, horizon = 60, 10
            n_train = n_val = None
            if PROCESSED_META.exists():
                meta = json.loads(PROCESSED_META.read_text())
                lookback = int(meta.get("lookback_window", 60))
                horizon = int(meta.get("prediction_horizon", 10))
                sizes = meta.get("split_sizes", {})
                n_train = int(sizes.get("train", 0))
                n_val = int(sizes.get("val", 0))

            if n_train is None or n_val is None:
                # Fallback: infer from total raw rows + test set length.
                # Test sample 0's "t" row index in raw = total_rows - n_test - horizon - 1
                # which assumes the chronological split used here.
                t_offset = len(raw) - n - horizon
                if t_offset < lookback - 1:
                    raise RuntimeError(f"raw parquet too short (rows={len(raw)})")
            else:
                t_offset = (n_train + n_val) + (lookback - 1)

            test_t = np.arange(n) + t_offset
            test_th = test_t + horizon
            if test_th[-1] < len(raw):
                mid_t = raw["mid_price"].to_numpy()[test_t]
                mid_th = raw["mid_price"].to_numpy()[test_th]
                ts_test = raw["timestamp"].to_numpy()[test_t]
                out["mid_price_t"] = mid_t
                out["mid_price_t_plus_h"] = mid_th
                # If we had no timestamp from the CSVs, prefer the raw one.
                if time_arr is None:
                    out["timestamp_ms"] = ts_test.astype(np.int64)
                print(f"[ok] attached mid_price from raw parquet (n={n})",
                      file=sys.stderr)
            else:
                print(f"[warn] raw parquet too short to attach mid_price"
                      f" (need index {test_th[-1]}, have {len(raw)})", file=sys.stderr)
        except Exception as exc:
            print(f"[warn] could not attach mid_price: {exc}", file=sys.stderr)
    else:
        print(f"[info] {RAW_PARQUET.name} not found — skipping mid_price"
              f" (the dashboard handles this gracefully)", file=sys.stderr)

    # --- write -----------------------------------------------------------------
    cols = ["timestamp_ms"]
    if "mid_price_t" in out.columns:
        cols += ["mid_price_t", "mid_price_t_plus_h"]
    cols += ["realized_log_return"]
    cols += [f"pred_{fam}" for fam in FAMILIES if f"pred_{fam}" in out.columns]
    out = out[[c for c in cols if c in out.columns]]

    out_path = DASHBOARD_DATA_DIR / "test_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[done] wrote {out_path}", file=sys.stderr)
    print(f"       shape  = {out.shape}", file=sys.stderr)
    print(f"       cols   = {list(out.columns)}", file=sys.stderr)
    print(f"       actual = {actual_source}", file=sys.stderr)
    print(f"       time   = {time_source}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
