"""Real-time backtest engine used by the Streamlit sandbox.

Mirrors the trading conventions in ``replication/evaluation/backtest.py``:

- Bet cadence: non-overlapping ``horizon``-step bets (10 s each).
- Position rule: pos_t = sign(pred_t) when |pred_t| > tau, else 0.
- Costs: cost_bps / 10000 charged on |Δposition| every rebalance.
- Annualization: PERIODS_PER_YEAR = (365.25 d × 24 h × 3600 s) / 10 s.
- NAV: cumulative product of (1 + r_net), drawdown w.r.t. running max.

For a step-by-step pipeline at 1 Hz with a 10-step horizon, consecutive
predictions overlap. To preserve the non-overlapping convention we take
every 10-th sample as a fresh bet, which matches the rebalance frequency
of the offline backtest.
"""
from __future__ import annotations

import numpy as np

SAMPLE_INTERVAL_S = 1.0
HORIZON_STEPS = 10
SECONDS_PER_YEAR = 365.25 * 24 * 3600
PERIODS_PER_YEAR = SECONDS_PER_YEAR / (SAMPLE_INTERVAL_S * HORIZON_STEPS)


def _expand(idx_bet: np.ndarray, full_len: int, values: np.ndarray) -> np.ndarray:
    """Expand a per-bet series back onto the full sample-rate axis.

    Each bet covers ``HORIZON_STEPS`` consecutive samples (the period over which
    the position is held). We forward-fill the bet's NAV / drawdown across
    those samples so plots line up with the original timestamp series.
    """
    out = np.full(full_len, np.nan)
    for k, start in enumerate(idx_bet):
        end = min(start + HORIZON_STEPS, full_len)
        out[start:end] = values[k]
    # Forward-fill the tail with the last value so plots don't end mid-way.
    if not np.isnan(out[0]):
        last = out[0]
    else:
        last = 1.0
    for i in range(full_len):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def backtest_strategy(
    predictions: np.ndarray,
    realized: np.ndarray,
    tau: float,
    cost_bps: float,
) -> dict:
    """Run the sign-threshold strategy and return summary stats + curves.

    Parameters
    ----------
    predictions, realized : 1-D arrays, same length, at the 1 Hz sample rate.
    tau : signal threshold on |predicted log-return|.
    cost_bps : transaction cost charged on |Δposition| per rebalance.

    Returns
    -------
    dict with keys: cum_return, ann_vol, sharpe, max_dd, win_rate,
                    position_fraction, nav, drawdown
    where nav / drawdown have the same length as ``predictions``.
    """
    n = len(predictions)
    if n == 0:
        return _empty_result(n)

    # Take every HORIZON_STEPS-th sample as a non-overlapping bet.
    idx_bet = np.arange(0, n, HORIZON_STEPS, dtype=np.int64)
    pred_bets = predictions[idx_bet]
    ret_bets = realized[idx_bet]

    positions = np.where(
        np.abs(pred_bets) > tau, np.sign(pred_bets), 0.0
    ).astype(float)

    # Costs are charged on |Δposition|. Treat the first bet as a change from 0.
    delta_pos = np.empty_like(positions)
    delta_pos[0] = abs(positions[0])
    delta_pos[1:] = np.abs(np.diff(positions))
    costs = (cost_bps / 1e4) * delta_pos

    # Approximate (gross) PnL per bet on the log-return scale.
    gross = positions * ret_bets
    net = gross - costs
    # NAV is compounded as a simple-return product. For tiny returns,
    # exp(net) ≈ 1 + net; this matches the offline backtest closely.
    nav_bets = np.cumprod(np.exp(net))

    # Drawdown
    running_max = np.maximum.accumulate(nav_bets)
    dd_bets = nav_bets / running_max - 1.0

    nav_full = _expand(idx_bet, n, nav_bets)
    dd_full = _expand(idx_bet, n, dd_bets)

    n_bets = len(net)
    cum_return = float(nav_bets[-1] - 1.0) if n_bets else 0.0
    mean_r = float(net.mean()) if n_bets else 0.0
    std_r = float(net.std(ddof=1)) if n_bets > 1 else 0.0
    ann_vol = std_r * np.sqrt(PERIODS_PER_YEAR)
    sharpe = (mean_r / std_r) * np.sqrt(PERIODS_PER_YEAR) if std_r > 1e-12 else 0.0

    active = positions != 0
    if active.any():
        win_rate = float(np.mean(gross[active] > 0))
    else:
        win_rate = 0.0
    position_fraction = float(active.mean())

    return {
        "cum_return": cum_return,
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": float(dd_bets.min()) if n_bets else 0.0,
        "win_rate": win_rate,
        "position_fraction": position_fraction,
        "nav": nav_full,
        "drawdown": dd_full,
    }


def buy_and_hold(realized: np.ndarray) -> dict:
    """Always-long benchmark over the same non-overlapping bet cadence."""
    n = len(realized)
    if n == 0:
        return _empty_result(n)
    idx_bet = np.arange(0, n, HORIZON_STEPS, dtype=np.int64)
    ret_bets = realized[idx_bet]
    nav_bets = np.cumprod(np.exp(ret_bets))
    running_max = np.maximum.accumulate(nav_bets)
    dd_bets = nav_bets / running_max - 1.0

    nav_full = _expand(idx_bet, n, nav_bets)
    dd_full = _expand(idx_bet, n, dd_bets)

    mean_r = float(ret_bets.mean())
    std_r = float(ret_bets.std(ddof=1)) if len(ret_bets) > 1 else 0.0
    ann_vol = std_r * np.sqrt(PERIODS_PER_YEAR)
    sharpe = (mean_r / std_r) * np.sqrt(PERIODS_PER_YEAR) if std_r > 1e-12 else 0.0

    return {
        "cum_return": float(nav_bets[-1] - 1.0),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": float(dd_bets.min()),
        "win_rate": float(np.mean(ret_bets > 0)),
        "nav": nav_full,
        "drawdown": dd_full,
    }


def max_drawdown_series(nav: np.ndarray) -> np.ndarray:
    """Convenience: drawdown of an arbitrary NAV series."""
    running_max = np.maximum.accumulate(nav)
    return nav / running_max - 1.0


def _empty_result(n: int) -> dict:
    z = np.zeros(n)
    return {
        "cum_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
        "max_dd": 0.0, "win_rate": 0.0, "position_fraction": 0.0,
        "nav": z + 1.0, "drawdown": z,
    }
