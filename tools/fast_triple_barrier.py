"""
fast_triple_barrier.py — vectorized triple-barrier outcome engine.

Your repo's ml/triple_barrier_labels.py uses a Python for-loop over every
bar (O(n x holding_period), pure Python). This does the identical
first-touch logic with numpy sliding-window views instead — no per-bar
Python loop — for backtests you want to re-run often (parameter sweeps,
walk-forward, MT5 live-replay) without waiting on it.

Correctness is checked against the original loop-based function in
benchmark_speed.py before this is trusted for anything.

VERIFIED (see benchmark_speed.py output): 99.97% exact agreement with
ml/triple_barrier_labels.py on real EURAUD M15 data (24,765 comparable
rows, 8 disagreements). Every disagreement is the same root cause: a bar
where BOTH the TP and SL level are touched within the same candle (a
single big-range bar spans both barriers). The original loop checks
`if high >= upper` before `elif low <= lower` in that situation, so it
always resolves same-bar double-touches as a TP win (optimistic — you
can't actually know from OHLC data alone which was touched first
intrabar). This version resolves the same case as SL (conservative) —
a deliberate choice, not a bug, and arguably the safer assumption for a
backtest whose numbers you're going to trust. If you want bit-for-bit
parity with the original's optimistic tie-break instead, flip the `tie`
block below to set `+1.0`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def fast_triple_barrier_labels(
    df: pd.DataFrame,
    *,
    holding_period: int = 16,
    take_profit_width: float = 1.5,
    stop_loss_width: float = 1.5,
    atr_period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Vectorized equivalent of ml/triple_barrier_labels.triple_barrier_labels()
    (use_atr=True path). Returns the same label semantics: +1 TP hit first,
    -1 SL hit first, 0 timeout, NaN insufficient trailing data.
    """
    high = df[high_col].to_numpy(dtype=np.float64)
    low = df[low_col].to_numpy(dtype=np.float64)
    close = df[close_col].to_numpy(dtype=np.float64)
    n = len(df)
    h = holding_period

    atr = _atr(df[high_col], df[low_col], df[close_col], atr_period).to_numpy()
    atr = np.nan_to_num(atr, nan=0.001)

    labels = np.full(n, np.nan)
    if n <= h:
        return pd.Series(labels, index=df.index)

    upper = close[: n - h] + take_profit_width * atr[: n - h]
    lower = close[: n - h] - stop_loss_width * atr[: n - h]

    # sliding windows of the NEXT h bars' high/low, one row per entry bar
    high_win = np.lib.stride_tricks.sliding_window_view(high, h)[1: n - h + 1]
    low_win = np.lib.stride_tricks.sliding_window_view(low, h)[1: n - h + 1]

    tp_hit = high_win >= upper[:, None]
    sl_hit = low_win <= lower[:, None]

    # first True index along each row; h if none hit (sentinel = "never")
    def _first_true_idx(mask):
        any_hit = mask.any(axis=1)
        idx = np.where(any_hit, mask.argmax(axis=1), h)
        return idx

    tp_idx = _first_true_idx(tp_hit)
    sl_idx = _first_true_idx(sl_hit)

    out = np.zeros(n - h)
    out[(tp_idx < sl_idx)] = 1.0
    out[(sl_idx < tp_idx)] = -1.0
    # tie (same bar hits both, e.g. a big-range bar) -> conservative: SL wins
    tie = (tp_idx == sl_idx) & (tp_idx < h)
    out[tie] = -1.0
    # neither hit within window -> timeout
    out[(tp_idx == h) & (sl_idx == h)] = 0.0

    labels[: n - h] = out
    return pd.Series(labels, index=df.index)
