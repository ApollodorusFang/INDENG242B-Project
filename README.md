# Crypto LOB Dataset (BTCUSDT)

A small, self-contained pipeline that collects Binance BTCUSDT top-20 limit
order book snapshots and turns them into a supervised learning dataset
similar in spirit to FI-2010.

> Academic research only. This is a teaching / coursework artifact and is
> **not** production trading infrastructure. Do not trade real capital with it.

---

## 1. Install

```bash
cd crypto_lob_dataset
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+.

## 2. Collect raw order book data

```bash
bash scripts/run_collection.sh
# or, with options:
python -m src.collect_orderbook \
    --symbol BTCUSDT \
    --duration-hours 6 \
    --sample-interval 1.0 \
    --out data/raw/btcusdt_lob_raw.parquet
```

What it does:

- Subscribes to the public WebSocket stream `btcusdt@depth20@100ms`, which
  delivers the top 20 bids/asks every 100 ms.
- Pulls one REST `/api/v3/depth` snapshot first as a connectivity/sanity check.
- Samples one snapshot per `--sample-interval` seconds (default 1s), keyed off
  the server clock so you never get duplicate timestamps.
- Reconnects with exponential backoff (capped at 30s) on disconnect.
- Logs progress once per minute and saves intermediate Parquet chunks into
  `data/raw/_chunks/`.
- On exit (duration reached or Ctrl-C), consolidates chunks into a single
  Parquet file at `data/raw/btcusdt_lob_raw.parquet`.

Default 6-hour, 1 Hz collection produces ~21,600 rows. The resulting Parquet
file is well under 500 MB (typically a few MB with Snappy compression).

### Raw schema

| column | description |
| --- | --- |
| `timestamp` | Binance event time, ms since epoch |
| `bid_price_1..20`, `bid_size_1..20` | Top 20 bid levels |
| `ask_price_1..20`, `ask_size_1..20` | Top 20 ask levels |
| `mid_price` | `(best_bid + best_ask) / 2` |
| `spread` | `best_ask - best_bid` |
| `order_book_imbalance_20` | `(sum bid sizes - sum ask sizes) / (sum bid sizes + sum ask sizes)` |

## 3. Build the training dataset

```bash
bash scripts/build_dataset.sh
# or, with options:
python -m src.build_dataset \
    --raw data/raw/btcusdt_lob_raw.parquet \
    --out-dir data/processed \
    --lookback 60 \
    --horizon 10 \
    --threshold 0.00005
```

What it does:

1. Loads the raw Parquet, drops duplicate timestamps, sorts by time.
2. Selects feature columns (see below) — `mid_price` is excluded from `X` and
   used only to construct labels.
3. Builds sliding windows of length `lookback`. For sample `i`, `X[i]` covers
   rows `[i, i+lookback)` and the labels look at the future from row
   `i + lookback - 1` to `i + lookback - 1 + horizon`.
4. Splits chronologically: 70% train, 15% val, 15% test (configurable).
5. Fits `StandardScaler` on the **training** windows only and transforms all
   three splits with it (no look-ahead leakage).
6. Saves normalized `X_*.npy`, raw `X_*_raw.npy`, three label variants, the
   scaler, the feature column list, and a metadata JSON.

### Output layout

```
data/
  raw/
    btcusdt_lob_raw.parquet
  processed/
    X_train.npy             X_val.npy             X_test.npy
    X_train_raw.npy         X_val_raw.npy         X_test_raw.npy
    y_train_reg.npy         y_val_reg.npy         y_test_reg.npy
    y_train_binary.npy      y_val_binary.npy      y_test_binary.npy
    y_train_3class.npy      y_val_3class.npy      y_test_3class.npy
    feature_names.json
    dataset_metadata.json
    scaler.pkl
```

### Feature columns (in order)

`bid_price_1..20`, `bid_size_1..20`, `ask_price_1..20`, `ask_size_1..20`,
`spread`, `order_book_imbalance_20` — **82 features** total.

`mid_price` and `timestamp` are intentionally excluded from `X`. `mid_price`
is only used to compute labels.

### Expected shapes

With defaults (`lookback=60`, `horizon=10`) on a 6-hour collection
(~21,600 rows):

- `X_train.npy`: `(num_train, 60, 82)` `float32`
- `X_val.npy`:   `(num_val,   60, 82)` `float32`
- `X_test.npy`:  `(num_test,  60, 82)` `float32`
- `y_*_reg.npy`: `(num_*,)` `float32`
- `y_*_binary.npy`: `(num_*,)` `int64` in `{0, 1}`
- `y_*_3class.npy`: `(num_*,)` `int64` in `{0, 1, 2}`

`num_train + num_val + num_test = raw_rows - lookback - horizon + 1`.

### Label semantics

For each sample `t` (with `t = i + lookback - 1`):

- `regression_label  = log(mid_price[t + horizon] / mid_price[t])`
- `binary_label      = 1 if regression_label > 0 else 0`
- `three_class_label = 0 (down)  if regression_label < -threshold`
- `three_class_label = 1 (flat)  if |regression_label| <= threshold`
- `three_class_label = 2 (up)    if regression_label > threshold`

Default `threshold = 5e-5`.

## 4. Loading the dataset

PyTorch:

```python
import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader

X = torch.from_numpy(np.load("data/processed/X_train.npy"))            # (N, 60, 82)
y = torch.from_numpy(np.load("data/processed/y_train_3class.npy"))     # (N,)
loader = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)
```

TensorFlow / Keras:

```python
import numpy as np, tensorflow as tf
X = np.load("data/processed/X_train.npy")
y = np.load("data/processed/y_train_3class.npy")
ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(4096).batch(256)
```

## 5. Notes & caveats

- The pipeline never trades, places orders, or authenticates with Binance — it
  only reads public market data.
- Binance public endpoints are rate-limited and may temporarily reject IPs that
  reconnect too aggressively. The collector backs off exponentially.
- Sampling at 1 Hz from a 100 ms stream means most messages are dropped on
  purpose. If you want denser data, lower `--sample-interval` (e.g. `0.1`).
- Crypto markets are non-stationary; train/val/test gaps are unavoidable on a
  short collection. For serious experiments, collect across multiple days.
