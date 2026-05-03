"""Collect BTCUSDT top-20 limit order book snapshots from Binance.

Uses the public partial book depth stream (`<symbol>@depth20@100ms`), which
already delivers the top 20 levels every 100ms. We sample one snapshot per
second and write everything to a single Parquet file at the end. Intermediate
chunks are flushed to disk so a crash mid-run still produces usable data.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
import requests
import websocket  # websocket-client

# Allow running this file as a script (python src/collect_orderbook.py)
# as well as a module (python -m src.collect_orderbook).
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import (  # type: ignore
        ASK_PRICE_COLS,
        ASK_SIZE_COLS,
        BID_PRICE_COLS,
        BID_SIZE_COLS,
        DEPTH,
        RAW_COLS,
        compute_derived,
        setup_logger,
    )
else:
    from .utils import (
        ASK_PRICE_COLS,
        ASK_SIZE_COLS,
        BID_PRICE_COLS,
        BID_SIZE_COLS,
        DEPTH,
        RAW_COLS,
        compute_derived,
        setup_logger,
    )


EXCHANGES = {
    "binance": {
        "ws": "wss://stream.binance.com:9443/ws/{stream}",
        "rest": "https://api.binance.com/api/v3/depth",
    },
    "binance-us": {
        "ws": "wss://stream.binance.us:9443/ws/{stream}",
        "rest": "https://api.binance.us/api/v3/depth",
    },
}


class OrderBookCollector:
    def __init__(
        self,
        symbol: str,
        duration_seconds: int,
        sample_interval_seconds: float,
        out_path: str,
        chunk_path: str,
        chunk_size: int,
        log_path: Optional[str] = None,
        exchange: str = "binance-us",
    ) -> None:
        if exchange not in EXCHANGES:
            raise ValueError(f"unknown exchange: {exchange}; choose from {list(EXCHANGES)}")
        self.exchange = exchange
        self.ws_url_tpl = EXCHANGES[exchange]["ws"]
        self.rest_url = EXCHANGES[exchange]["rest"]
        self.symbol = symbol.lower()
        self.duration_seconds = duration_seconds
        self.sample_interval_ms = int(sample_interval_seconds * 1000)
        self.out_path = out_path
        self.chunk_path = chunk_path
        self.chunk_size = chunk_size

        self.logger = setup_logger("collector", log_path)
        self.rows: List[Dict] = []
        self.last_sampled_bucket: Optional[int] = None  # second bucket id
        self.last_logged_minute: Optional[int] = None
        self.start_wall: Optional[float] = None
        self.start_server_ms: Optional[int] = None
        self.total_messages = 0
        self.malformed_messages = 0
        self.chunk_index = 0
        self._stop = False

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        os.makedirs(self.chunk_path, exist_ok=True)

        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    # ----- lifecycle -----

    def _on_signal(self, signum, _frame) -> None:
        self.logger.warning("signal %s received, stopping collection", signum)
        self._stop = True

    def run(self) -> None:
        self._rest_warmup()
        self.start_wall = time.time()
        self.logger.info(
            "starting collection: exchange=%s symbol=%s duration=%ds sample_every=%dms top=%d",
            self.exchange,
            self.symbol.upper(),
            self.duration_seconds,
            self.sample_interval_ms,
            DEPTH,
        )

        backoff = 1.0
        while not self._stop and not self._duration_elapsed():
            try:
                self._run_once()
                # clean disconnect (server closed). Reset backoff and reconnect.
                backoff = 1.0
            except Exception as exc:  # pragma: no cover - network failure paths
                self.logger.error("ws error: %s; reconnecting in %.1fs", exc, backoff)
                self._sleep_with_check(backoff)
                backoff = min(backoff * 2.0, 30.0)

        self._flush_chunk(force=True)
        self._consolidate()

    def _duration_elapsed(self) -> bool:
        if self.start_wall is None:
            return False
        return (time.time() - self.start_wall) >= self.duration_seconds

    def _sleep_with_check(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end and not self._stop and not self._duration_elapsed():
            time.sleep(min(0.5, end - time.time()))

    # ----- network -----

    def _rest_warmup(self) -> None:
        """Sanity-check connectivity with a REST depth snapshot before opening WS."""
        try:
            r = requests.get(
                self.rest_url,
                params={"symbol": self.symbol.upper(), "limit": 20},
                timeout=10,
            )
            r.raise_for_status()
            payload = r.json()
            if "bids" in payload and "asks" in payload:
                self.logger.info(
                    "REST snapshot OK: %d bids / %d asks",
                    len(payload["bids"]),
                    len(payload["asks"]),
                )
            else:
                self.logger.warning("REST snapshot returned unexpected payload")
        except Exception as exc:
            self.logger.warning("REST snapshot failed (%s); continuing to WS", exc)

    def _run_once(self) -> None:
        stream = f"{self.symbol}@depth{DEPTH}@100ms"
        url = self.ws_url_tpl.format(stream=stream)
        ws = websocket.create_connection(url, timeout=30)
        ws.settimeout(30)
        self.logger.info("ws connected: %s", url)
        try:
            while not self._stop and not self._duration_elapsed():
                try:
                    msg = ws.recv()
                except websocket.WebSocketTimeoutException:
                    self.logger.warning("ws recv timeout, reconnecting")
                    return
                if not msg:
                    return
                self._handle_message(msg)
        finally:
            try:
                ws.close()
            except Exception:
                pass

    # ----- message handling -----

    def _handle_message(self, msg: str) -> None:
        self.total_messages += 1
        try:
            data = json.loads(msg)
            bids_raw = data["bids"]
            asks_raw = data["asks"]
            event_ms = int(data.get("E") or int(time.time() * 1000))
        except (ValueError, KeyError, TypeError):
            self.malformed_messages += 1
            if self.malformed_messages <= 5:
                self.logger.warning("malformed message #%d", self.malformed_messages)
            return

        if len(bids_raw) < DEPTH or len(asks_raw) < DEPTH:
            self.malformed_messages += 1
            return

        if self.start_server_ms is None:
            self.start_server_ms = event_ms

        # Sample at most one snapshot per `sample_interval_ms` window, keyed off
        # the server clock. This guarantees no duplicate timestamps.
        bucket = event_ms // self.sample_interval_ms
        if self.last_sampled_bucket is not None and bucket <= self.last_sampled_bucket:
            return
        self.last_sampled_bucket = bucket

        try:
            bid_prices = [float(b[0]) for b in bids_raw[:DEPTH]]
            bid_sizes = [float(b[1]) for b in bids_raw[:DEPTH]]
            ask_prices = [float(a[0]) for a in asks_raw[:DEPTH]]
            ask_sizes = [float(a[1]) for a in asks_raw[:DEPTH]]
        except (ValueError, TypeError):
            self.malformed_messages += 1
            return

        mid_price, spread, imbalance = compute_derived(
            bid_prices, bid_sizes, ask_prices, ask_sizes
        )

        row: Dict = {"timestamp": event_ms}
        row.update(dict(zip(BID_PRICE_COLS, bid_prices)))
        row.update(dict(zip(BID_SIZE_COLS, bid_sizes)))
        row.update(dict(zip(ASK_PRICE_COLS, ask_prices)))
        row.update(dict(zip(ASK_SIZE_COLS, ask_sizes)))
        row["mid_price"] = mid_price
        row["spread"] = spread
        row["order_book_imbalance_20"] = imbalance
        self.rows.append(row)

        self._maybe_log_progress()
        if len(self.rows) >= self.chunk_size:
            self._flush_chunk()

    def _maybe_log_progress(self) -> None:
        if self.start_wall is None:
            return
        elapsed = time.time() - self.start_wall
        minute = int(elapsed // 60)
        if minute == self.last_logged_minute:
            return
        self.last_logged_minute = minute
        self.logger.info(
            "progress: elapsed=%ds rows=%d malformed=%d msgs=%d",
            int(elapsed),
            len(self.rows) + self.chunk_index * self.chunk_size,
            self.malformed_messages,
            self.total_messages,
        )

    # ----- persistence -----

    def _flush_chunk(self, force: bool = False) -> None:
        if not self.rows:
            return
        if not force and len(self.rows) < self.chunk_size:
            return
        df = pd.DataFrame(self.rows, columns=RAW_COLS)
        path = os.path.join(self.chunk_path, f"chunk_{self.chunk_index:05d}.parquet")
        df.to_parquet(path, index=False)
        self.logger.info("flushed %d rows to %s", len(df), path)
        self.chunk_index += 1
        self.rows = []

    def _consolidate(self) -> None:
        chunks = sorted(
            os.path.join(self.chunk_path, f)
            for f in os.listdir(self.chunk_path)
            if f.startswith("chunk_") and f.endswith(".parquet")
        )
        if not chunks:
            self.logger.error("no chunks were written; nothing to consolidate")
            return
        frames = [pd.read_parquet(c) for c in chunks]
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.to_parquet(self.out_path, index=False)
        self.logger.info(
            "wrote %s rows=%d duration=%.1fs malformed=%d",
            self.out_path,
            len(df),
            (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 1000.0,
            self.malformed_messages,
        )
        # Best-effort cleanup of intermediate chunks.
        for c in chunks:
            try:
                os.remove(c)
            except OSError:
                pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect BTCUSDT top-20 LOB snapshots.")
    p.add_argument(
        "--exchange",
        default="binance-us",
        choices=list(EXCHANGES),
        help="Exchange WebSocket source (default: binance-us, since binance.com is geo-restricted in the US).",
    )
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument(
        "--duration-hours",
        type=float,
        default=6.0,
        help="Total collection duration in hours (default: 6).",
    )
    p.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Seconds between sampled snapshots (default: 1.0).",
    )
    p.add_argument(
        "--out",
        default="data/raw/btcusdt_lob_raw.parquet",
        help="Final consolidated parquet output path.",
    )
    p.add_argument(
        "--chunk-dir",
        default="data/raw/_chunks",
        help="Directory for intermediate parquet chunks.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1800,
        help="Rows per intermediate chunk (default: 1800 ~ 30 min @ 1Hz).",
    )
    p.add_argument(
        "--log-file",
        default="data/raw/collector.log",
        help="Path to log file (in addition to stdout).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    collector = OrderBookCollector(
        symbol=args.symbol,
        duration_seconds=int(args.duration_hours * 3600),
        sample_interval_seconds=args.sample_interval,
        out_path=args.out,
        chunk_path=args.chunk_dir,
        chunk_size=args.chunk_size,
        log_path=args.log_file,
        exchange=args.exchange,
    )
    collector.run()


if __name__ == "__main__":
    main()
