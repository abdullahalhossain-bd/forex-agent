"""
research/candlestick_backtest.py
=================================
FAST, ISOLATED candlestick-pattern RESEARCH backtester.

Purpose (audit Part 4): answer "which candlestick patterns actually have
measurable predictive value, on which pairs/timeframes/sessions, and does
confirmation help" — for RESEARCH, not live trading. This module is
deliberately NOT wired into `analysis/candlestick_engine.py`'s unified
`evaluate()`/`evaluate_series()` (which also runs the ML/br-book pattern
sets, trend/location/volatility confidence scoring, agreement/conflict
resolution, etc. — all of that is overkill for "what happens after a
Hammer on EURUSD M15" and would make this an order of magnitude slower).

It has exactly one production dependency: `analysis.candlestick_patterns_mw`
(`compute()` + `add_confirmation()`), called ONCE per (pair, timeframe)
dataset. No ML engine, no LLM/API calls, no liquidity/news/consensus
modules, no per-row `evaluate()` calls. Forward returns and the RR
backtest are vectorized with NumPy (sliding-window first-touch detection
for SL/TP, not a bar-by-bar Python loop).

DATA
----
This module does NOT assume any filenames. Point it at a directory with
`discover_ohlc_files()` / `load_ohlc_file()`, which tolerate the same
column-naming variance as `backtest/data_loader.py::HistoricalDataLoader`
(time/date/datetime/timestamp, volume/vol/tickvol/tick_volume, BOM-safe,
EET->UTC localization for naive MT4/MT5 timestamps) WITHOUT importing that
module or any of its own heavy dependencies (`analysis.market_regime`,
`analysis.patterns`, `data.indicators`, `utils.logger`, ...) — this file
is meant to run standing completely alone.

ENTRY MODELS (Part 15 — do not mix these; every trades table is labeled)
--------------------------------------------------------------------
  baseline (unconfirmed):
      pattern detected at close(i)  ->  entry = open(i+1)
      Applies to every detected pattern (1/2/3-bar), confirmable or not.

  confirmed (1-bar reversal patterns only — Hammer, Inverted Hammer,
  Dragonfly Doji, Shooting Star, Hanging Man, Gravestone Doji):
      pattern detected at close(i)
      confirmation evaluated using close(i+1)   [mw.add_confirmation]
      entry = open(i+2)   <- only once bar i+1 has actually closed
      Only rows where `csp_confirmed[i] == True` enter a trade.

Both are produced from the SAME underlying pattern events but written to
two SEPARATE trade tables (`trades_baseline.csv` / not-mixed columns in
the combined ledger via an explicit `entry_model` column) so confirmed vs.
unconfirmed performance (Part 6H) can be compared honestly.

RISK MODEL
----------
Stop distance = 1x ATR(14) (Wilder-style, computed once per dataset) as
of the pattern bar. Take-profit distance = stop_distance * RR, for
RR in {1.0, 1.5, 2.0, 3.0} (Part 5). SL/TP "first touch" is resolved with
a vectorized sliding-window scan of `high`/`low` over the next
`max_holding_bars` bars (default 50): if both SL and TP fall inside the
SAME bar, the outcome is conservatively scored a loss (intrabar path is
unknown from OHLC data alone — this is a standard, deliberately
pessimistic backtesting convention, not a modeling error). If neither is
touched within the window, the trade is closed at the last available
close in the window ("no_hit", mark-to-market).

CAUSALITY
---------
Every entry price is drawn from a bar STRICTLY AFTER the bar that
produced the signal (open(i+1) or open(i+2), never close(i) or
close(i+1)). SL/TP touches are only ever evaluated on bars at or after
the entry bar. No column here is computed from `evaluate_series()`
output, so `candlestick_engine.py`'s own causal guarantees are inherited
by construction, not re-verified here — `test_candlestick_architecture.py`
already covers that at the detection layer.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import candlestick_patterns_mw as mw  # noqa: E402


# =============================================================================
# PART 14 — DATA DISCOVERY / LOADING (no assumed filenames)
# =============================================================================

REQUIRED_OHLC_COLS = ["open", "high", "low", "close"]
TIME_COL_CANDIDATES = ("time", "datetime", "datetime_utc", "date", "timestamp", "local time")
VOLUME_COL_CANDIDATES = ("volume", "vol", "tickvol", "tick_volume", "tick volume", "real_volume")

_TF_TOKENS = {
    "M1": "M1", "1M": "M1", "M5": "M5", "5M": "M5", "M15": "M15", "15M": "M15",
    "M30": "M30", "30M": "M30", "H1": "H1", "1H": "H1", "H4": "H4", "4H": "H4",
    "D1": "D1", "1D": "D1", "DAILY": "D1", "DAY": "D1",
}


def discover_ohlc_files(data_dir: str) -> list[str]:
    """Recursively search `data_dir` for candidate OHLC files (csv/parquet).
    Returns an empty list (never raises) if the directory doesn't exist —
    callers are expected to report that explicitly (Part 14: "stop the
    backtest portion and tell me exactly which data files are missing")."""
    if not data_dir or not os.path.isdir(data_dir):
        return []
    found: list[str] = []
    for pat in ("*.csv", "*.parquet", "*.pq"):
        found += glob.glob(os.path.join(data_dir, "**", pat), recursive=True)
    return sorted(found)


def infer_pair_timeframe(path: str) -> tuple[str, str]:
    """Best-effort pair/timeframe inference from a filename, e.g.
    'EURUSD_M15.csv' -> ('EURUSD', 'M15'). Falls back to ('UNKNOWN', 'UNKNOWN')
    rather than guessing wrong silently."""
    base = os.path.splitext(os.path.basename(path))[0].upper()
    for token, norm in _TF_TOKENS.items():
        if token in base:
            pair = base.replace(token, "").strip("_- ")
            return (pair or "UNKNOWN"), norm
    return "UNKNOWN", "UNKNOWN"


def load_ohlc_file(path: str) -> pd.DataFrame:
    """
    Load and normalize one OHLC file. Mirrors the column-handling
    conventions already established in `backtest/data_loader.py`
    (BOM-safe headers, tolerant time-column detection, EET->UTC
    localization of naive MT4/MT5 timestamps, volume defaulted to 0.0
    when genuinely absent) WITHOUT importing that module, so this
    research tool has zero dependency on the rest of the backtest/AI
    package (Part 4).
    """
    if path.lower().endswith((".parquet", ".pq")):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")

    df.columns = [str(c).strip().lstrip("\ufeff").lower() for c in df.columns]

    if "time" not in df.columns:
        for alt in TIME_COL_CANDIDATES:
            if alt in df.columns:
                df = df.rename(columns={alt: "time"})
                break
    if "time" not in df.columns:
        raise ValueError(
            f"{path}: no recognizable timestamp column "
            f"(looked for {TIME_COL_CANDIDATES}; got {list(df.columns)})"
        )

    missing = [c for c in REQUIRED_OHLC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required OHLC column(s) {missing}")

    if "volume" not in df.columns:
        for alt in VOLUME_COL_CANDIDATES:
            if alt in df.columns:
                df = df.rename(columns={alt: "volume"})
                break
        else:
            df["volume"] = 0.0

    ts = pd.to_datetime(df["time"], errors="coerce", utc=False)
    if ts.dt.tz is None:
        try:
            ts = ts.dt.tz_localize("EET").dt.tz_convert("UTC")
        except Exception:
            ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    df["time"] = ts
    df = df.dropna(subset=["time"]).drop_duplicates(subset=["time"]).sort_values("time")
    df = df.set_index("time")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < n_before:
        warnings.warn(f"{path}: dropped {n_before - len(df)} rows with non-numeric/NaN OHLC")

    real_volume_available = "volume" in df.columns and (df["volume"].fillna(0) != 0).any()
    df.attrs["real_volume_available"] = bool(real_volume_available)
    return df


# =============================================================================
# SESSIONS (UTC-hour based; documented explicitly per Part 6E)
# =============================================================================
# Asian:            00:00-08:00 UTC (Tokyo core hours)
# London:           08:00-16:00 UTC
# New York:         13:00-21:00 UTC
# London/NY overlap 13:00-16:00 UTC (subset of both — a bar is tagged with
#   the single most specific label; overlap wins over the two individual
#   sessions when both would otherwise apply).

def session_label(utc_hour: np.ndarray) -> np.ndarray:
    out = np.full(utc_hour.shape, "asian", dtype=object)
    london = (utc_hour >= 8) & (utc_hour < 16)
    ny = (utc_hour >= 13) & (utc_hour < 21)
    overlap = london & ny
    out[london] = "london"
    out[ny] = "new_york"
    out[overlap] = "london_ny_overlap"
    return out


# =============================================================================
# PART 16 — PATTERN INVENTORY
# =============================================================================

def pattern_inventory() -> pd.DataFrame:
    """Enumerate every pattern currently implemented in
    `candlestick_patterns_mw.py` (the scanner this research engine uses),
    with category/direction/bar-length/confirmability — Part 16."""
    rows = []
    all_names = set(mw.PATTERN_LENGTH.keys())
    confirmable = mw._CONFIRMABLE_BULLISH_1BAR | mw._CONFIRMABLE_BEARISH_1BAR
    for name in sorted(all_names):
        rows.append({
            "pattern": name,
            "category": mw._categorize(name),
            "bars_required": mw.PATTERN_LENGTH[name],
            "confirmation_required": name in confirmable,
            "source_module": "candlestick_patterns_mw.compute()",
        })
    return pd.DataFrame(rows).sort_values(["bars_required", "pattern"]).reset_index(drop=True)


# =============================================================================
# PART 4/13 — VECTORIZED PATTERN DETECTION (called ONCE per dataset)
# =============================================================================

def detect_pattern_events(df: pd.DataFrame, pair: str, timeframe: str) -> pd.DataFrame:
    """
    Run `mw.compute()` + `mw.add_confirmation()` exactly once and return a
    tidy long-format events table (one row per detected pattern
    occurrence), not a per-bar wide table — this is what both the RR
    backtest and the forward-horizon research consume.
    """
    out = mw.compute(df)
    out = mw.add_confirmation(out)

    pat = out["csp_pattern"].to_numpy(dtype=object)
    cat = out["csp_category"].to_numpy(dtype=object)
    confirmed = out["csp_confirmed"].to_numpy()
    pending = out["csp_confirmation_pending"].to_numpy()

    mask = np.array([not mw.is_no_pattern(p) for p in pat])
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return pd.DataFrame(columns=[
            "pair", "timeframe", "timestamp", "pattern", "direction",
            "pattern_bar", "confirmation_bar", "confirmable", "confirmed", "pending",
        ])

    confirmable_set = mw._CONFIRMABLE_BULLISH_1BAR | mw._CONFIRMABLE_BEARISH_1BAR
    events = pd.DataFrame({
        "pair": pair,
        "timeframe": timeframe,
        "timestamp": df.index[idx],
        "pattern": pat[idx],
        "direction": cat[idx],
        "pattern_bar": idx,
        "confirmation_bar": idx + 1,
        "confirmable": [p in confirmable_set for p in pat[idx]],
        "confirmed": confirmed[idx],
        "pending": pending[idx],
    })
    # Only bullish/bearish are tradeable directions; neutral (Doji family)
    # is kept in the events table (useful for inventory/frequency stats)
    # but excluded from the RR backtest downstream.
    return events.reset_index(drop=True)


# =============================================================================
# PART 7 — FORWARD-HORIZON RETURNS (fully vectorized)
# =============================================================================

FORWARD_HORIZONS = (1, 2, 3, 5, 10, 20)


def compute_forward_returns(df: pd.DataFrame, horizons=FORWARD_HORIZONS) -> pd.DataFrame:
    """
    For every bar i, the forward close-to-close return over each horizon
    h: (close[i+h] - close[i]) / close[i]. Entirely vectorized with
    shifted NumPy arrays — no Python loop over rows.
    """
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    cols = {}
    for h in horizons:
        fwd = np.full(n, np.nan)
        if h < n:
            fwd[: n - h] = (close[h:] - close[:-h]) / close[:-h]
        cols[f"fwd_ret_{h}"] = fwd
    return pd.DataFrame(cols, index=df.index)


# =============================================================================
# PART 5 — RR BACKTEST (vectorized sliding-window first-touch)
# =============================================================================

RR_RATIOS = (1.0, 1.5, 2.0, 3.0)
DEFAULT_MAX_HOLDING_BARS = 50
DEFAULT_ATR_PERIOD = 14


def _atr(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> np.ndarray:
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    # Wilder smoothing via pandas ewm (alpha = 1/period), matches the
    # conventional ATR(14) definition used elsewhere in this project.
    return pd.Series(tr).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().to_numpy()


def _vectorized_rr_outcomes(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    entry_bar: np.ndarray, direction: np.ndarray, atr_at_signal: np.ndarray,
    rr: float, max_holding_bars: int,
) -> dict[str, np.ndarray]:
    """
    For each event (entry_bar[i], direction[i]), simulate a fixed-RR
    trade using a vectorized sliding-window scan of `high`/`low` starting
    at entry_bar, up to `max_holding_bars` ahead. Returns arrays keyed by
    output column name — no per-event Python loop.
    """
    n_events = len(entry_bar)
    n_bars = len(open_)
    W = max_holding_bars

    # Pad price arrays so windows that run off the end of the data are
    # safely NaN (never treated as a touch).
    pad = np.full(W, np.nan)
    high_p = np.concatenate([high, pad])
    low_p = np.concatenate([low, pad])
    close_p = np.concatenate([close, pad])

    valid = (entry_bar >= 0) & (entry_bar < n_bars)
    entry_bar = np.where(valid, entry_bar, 0)  # placeholder; masked out later
    entry_price = open_[np.clip(entry_bar, 0, n_bars - 1)]

    stop_distance = np.maximum(atr_at_signal, 1e-12)  # guard against zero/NaN ATR
    stop_price = entry_price - direction * stop_distance
    tp_price = entry_price + direction * stop_distance * rr

    win_idx = entry_bar[:, None] + np.arange(W)[None, :]
    win_high = high_p[win_idx]
    win_low = low_p[win_idx]
    win_close = close_p[win_idx]

    dir_col = direction[:, None]
    tp_col = tp_price[:, None]
    sl_col = stop_price[:, None]

    hit_tp = np.where(dir_col > 0, win_high >= tp_col, win_low <= tp_col)
    hit_sl = np.where(dir_col > 0, win_low <= sl_col, win_high >= sl_col)
    hit_tp = np.where(np.isnan(win_high) | np.isnan(win_low), False, hit_tp)
    hit_sl = np.where(np.isnan(win_high) | np.isnan(win_low), False, hit_sl)

    def _first_true(mask: np.ndarray) -> np.ndarray:
        any_hit = mask.any(axis=1)
        first = np.argmax(mask, axis=1)
        return np.where(any_hit, first, W)

    tp_first = _first_true(hit_tp)
    sl_first = _first_true(hit_sl)

    outcome = np.where(
        tp_first < sl_first, "win",
        np.where(sl_first < tp_first, "loss",
                 np.where(tp_first < W, "loss_tie", "no_hit")),
    )

    # Last valid (non-NaN) bar available in each window, for the no_hit case.
    valid_mask = ~np.isnan(win_close)
    last_valid = np.where(valid_mask.any(axis=1), valid_mask.cumsum(axis=1).argmax(axis=1), 0)

    exit_idx = np.select(
        [outcome == "win", np.isin(outcome, ["loss", "loss_tie"]), outcome == "no_hit"],
        [tp_first, sl_first, last_valid],
    )
    exit_idx = np.clip(exit_idx, 0, W - 1).astype(int)

    exit_price = np.select(
        [outcome == "win", np.isin(outcome, ["loss", "loss_tie"]), outcome == "no_hit"],
        [tp_price, stop_price, np.take_along_axis(win_close, exit_idx[:, None], axis=1).ravel()],
    )

    # MFE/MAE up to (and including) the exit bar — cumulative running
    # max/min of the window, gathered at exit_idx (vectorized, no loop).
    cummax_high = np.fmax.accumulate(np.where(np.isnan(win_high), -np.inf, win_high), axis=1)
    cummin_low = np.fmin.accumulate(np.where(np.isnan(win_low), np.inf, win_low), axis=1)
    hi_reach = np.take_along_axis(cummax_high, exit_idx[:, None], axis=1).ravel()
    lo_reach = np.take_along_axis(cummin_low, exit_idx[:, None], axis=1).ravel()

    fav_if_bull = hi_reach - entry_price
    adv_if_bull = entry_price - lo_reach
    mfe = np.where(direction > 0, fav_if_bull, adv_if_bull)
    mae = np.where(direction > 0, adv_if_bull, fav_if_bull)

    r_multiple = direction * (exit_price - entry_price) / stop_distance
    ret_pct = direction * (exit_price - entry_price) / entry_price
    holding_period = exit_idx + 1

    result = {
        "entry_bar": entry_bar,
        "entry_price": entry_price,
        "stop_loss": stop_price,
        "take_profit": tp_price,
        "outcome": outcome,
        "exit_price": exit_price,
        "holding_period": holding_period,
        "return": ret_pct,
        "r_multiple": r_multiple,
        "mfe_r": mfe / stop_distance,
        "mae_r": mae / stop_distance,
    }
    for k in result:
        if isinstance(result[k], np.ndarray) and result[k].dtype != object:
            result[k] = np.where(valid, result[k], np.nan) if np.issubdtype(result[k].dtype, np.floating) else result[k]
    return result


def run_rr_backtest(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    rr_ratios=RR_RATIOS,
    max_holding_bars: int = DEFAULT_MAX_HOLDING_BARS,
    atr_period: int = DEFAULT_ATR_PERIOD,
) -> pd.DataFrame:
    """
    Build the full trades ledger (Part 5 columns) for BOTH entry models,
    for every RR ratio, for every tradeable (bullish/bearish) event in
    `events`. Returns one long DataFrame with an explicit `entry_model`
    column ("baseline" / "confirmed") so the two are never accidentally
    mixed in downstream aggregation (Part 15).
    """
    tradeable = events[events["direction"].isin(["bullish", "bearish"])].copy()
    if tradeable.empty:
        return pd.DataFrame()

    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr = _atr(df, period=atr_period)
    n_bars = len(df)
    dir_sign = np.where(tradeable["direction"].to_numpy() == "bullish", 1.0, -1.0)

    all_trades = []

    # ---- Baseline (unconfirmed) entry model: entry = open(pattern_bar+1) ----
    entry_bar_base = tradeable["pattern_bar"].to_numpy() + 1
    atr_at_signal_base = atr[np.clip(tradeable["pattern_bar"].to_numpy(), 0, n_bars - 1)]
    for rr in rr_ratios:
        res = _vectorized_rr_outcomes(
            open_, high, low, close, entry_bar_base, dir_sign, atr_at_signal_base,
            rr, max_holding_bars,
        )
        trades = tradeable[["pair", "timeframe", "timestamp", "pattern", "direction",
                             "pattern_bar", "confirmation_bar", "confirmable",
                             "confirmed", "pending"]].copy()
        trades["entry_model"] = "baseline"
        trades["rr_ratio"] = rr
        trades["confirmation_status"] = np.where(
            tradeable["confirmable"], np.where(tradeable["confirmed"], "confirmed_but_baseline_entry",
            np.where(tradeable["pending"], "pending", "not_confirmed")), "not_applicable")
        for k, v in res.items():
            trades[k] = v
        all_trades.append(trades)

    # ---- Confirmed entry model: entry = open(pattern_bar+2), confirmed==True only ----
    confirmed_mask = tradeable["confirmable"] & tradeable["confirmed"]
    conf_events = tradeable[confirmed_mask]
    if not conf_events.empty:
        entry_bar_conf = conf_events["pattern_bar"].to_numpy() + 2
        atr_at_signal_conf = atr[np.clip(conf_events["pattern_bar"].to_numpy(), 0, n_bars - 1)]
        dir_sign_conf = np.where(conf_events["direction"].to_numpy() == "bullish", 1.0, -1.0)
        for rr in rr_ratios:
            res = _vectorized_rr_outcomes(
                open_, high, low, close, entry_bar_conf, dir_sign_conf, atr_at_signal_conf,
                rr, max_holding_bars,
            )
            trades = conf_events[["pair", "timeframe", "timestamp", "pattern", "direction",
                                   "pattern_bar", "confirmation_bar", "confirmable",
                                   "confirmed", "pending"]].copy()
            trades["entry_model"] = "confirmed"
            trades["rr_ratio"] = rr
            trades["confirmation_status"] = "confirmed"
            for k, v in res.items():
                trades[k] = v
            all_trades.append(trades)

    out = pd.concat(all_trades, ignore_index=True)
    # Drop trades whose entry bar ran off the end of the dataset (no valid entry price).
    out = out.dropna(subset=["entry_price"]).reset_index(drop=True)
    return out


# =============================================================================
# PART 8/9 — STATISTICAL AGGREGATION (N thresholds, bootstrap CI, warnings)
# =============================================================================

N_THRESHOLDS = (30, 100, 300)


def _max_losing_streak(is_loss: np.ndarray) -> int:
    if len(is_loss) == 0:
        return 0
    streak = best = 0
    for v in is_loss:
        streak = streak + 1 if v else 0
        best = max(best, streak)
    return int(best)


def _max_drawdown_r(r_multiples: np.ndarray) -> float:
    """Max peak-to-trough drawdown of the cumulative R-multiple equity curve
    (in R units), in chronological trade order."""
    if len(r_multiples) == 0:
        return 0.0
    equity = np.cumsum(r_multiples)
    running_max = np.maximum.accumulate(equity)
    dd = equity - running_max
    return float(dd.min())


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Simple percentile bootstrap CI for the mean of `values`."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo = (1 - ci) / 2 * 100
    hi = 100 - lo
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a win-rate proportion — more reliable than
    a normal approximation at small N."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    margin = z * math_sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return ((centre - margin) / denom, (centre + margin) / denom)


def math_sqrt(x: float) -> float:
    return float(np.sqrt(max(x, 0.0)))


def summarize_trades(trades: pd.DataFrame, group_cols: list[str], min_n: int = 1) -> pd.DataFrame:
    """
    Core statistics table (Part 8). One row per group (whatever
    `group_cols` is — pattern, pattern+pair, pattern+timeframe, ...).
    NEVER ranks by win rate alone: reports N, win rate (+ Wilson CI),
    average/median R, expectancy, profit factor, max drawdown, max
    losing streak, and explicit low-sample flags at N>=30/100/300.
    """
    rows = []
    for keys, grp in trades.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(grp)
        if n < min_n:
            continue
        r = grp["r_multiple"].to_numpy(dtype=float)
        wins = int((grp["outcome"] == "win").sum())
        losses = int(grp["outcome"].isin(["loss", "loss_tie"]).sum())
        win_rate = wins / n if n else float("nan")
        avg_r = float(np.mean(r)) if n else float("nan")
        median_r = float(np.median(r)) if n else float("nan")
        expectancy = avg_r  # expectancy in R, since every trade is R-normalized
        gross_win = r[r > 0].sum()
        gross_loss = -r[r < 0].sum()
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else float("nan")
        wilson_lo, wilson_hi = _wilson_ci(wins, n)
        boot_lo, boot_hi = _bootstrap_ci(r) if n >= 10 else (float("nan"), float("nan"))
        max_dd = _max_drawdown_r(r)
        max_streak = _max_losing_streak((grp["outcome"].isin(["loss", "loss_tie"])).to_numpy())

        row = dict(zip(group_cols, keys))
        row.update({
            "n": n,
            "wins": wins,
            "losses": losses,
            "no_hit": int((grp["outcome"] == "no_hit").sum()),
            "win_rate": round(win_rate, 4),
            "win_rate_wilson_lo": round(wilson_lo, 4) if not math.isnan(wilson_lo) else None,
            "win_rate_wilson_hi": round(wilson_hi, 4) if not math.isnan(wilson_hi) else None,
            "avg_r": round(avg_r, 4),
            "median_r": round(median_r, 4),
            "expectancy_r": round(expectancy, 4),
            "expectancy_r_boot_ci_lo": round(boot_lo, 4) if not math.isnan(boot_lo) else None,
            "expectancy_r_boot_ci_hi": round(boot_hi, 4) if not math.isnan(boot_hi) else None,
            "profit_factor": round(profit_factor, 4) if math.isfinite(profit_factor) else profit_factor,
            "max_drawdown_r": round(max_dd, 4),
            "max_losing_streak": max_streak,
            "n_ge_30": n >= 30,
            "n_ge_100": n >= 100,
            "n_ge_300": n >= 300,
            "low_sample_warning": n < 30,
        })
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Part 9: rank favoring adequate N + positive expectancy + stability,
    # NOT win rate alone. Composite score = expectancy * log1p(n), zeroed
    # out for n<30 (too data-mined to trust).
    result["research_rank_score"] = np.where(
        result["n"] >= 30,
        result["expectancy_r"] * np.log1p(result["n"]),
        np.nan,
    )
    return result.sort_values("research_rank_score", ascending=False, na_position="last").reset_index(drop=True)


import math  # noqa: E402  (kept near point of use for readability above)


MULTIPLE_TESTING_WARNING = (
    "WARNING — MULTIPLE TESTING / DATA-MINING BIAS: this report evaluates "
    "many pattern x pair x timeframe x session x month x regime combinations. "
    "With enough combinations, some will show an apparently strong edge by "
    "chance alone even if no real edge exists. The highest observed win rate "
    "or expectancy in any single slice is NOT automatically the best "
    "strategy — prefer combinations with N>=100 (ideally >=300), positive "
    "out-of-sample expectancy, and stability across the train/test split "
    "over combinations that merely look best in one slice of the data."
)


# =============================================================================
# PART 10 — CHRONOLOGICAL TRAIN/TEST SPLIT (no shuffling)
# =============================================================================

def chronological_split(df: pd.DataFrame, train_frac: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split OHLC data chronologically (never shuffled) into an in-sample
    ("research/development") and out-of-sample slice, per (pair, timeframe)
    dataset, BEFORE pattern detection — so no pattern's context (trend MAs,
    ATR warm-up, etc.) ever sees across the split boundary."""
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# =============================================================================
# ORCHESTRATION — run everything for one or many (pair, timeframe) datasets
# =============================================================================

@dataclass
class DatasetSpec:
    pair: str
    timeframe: str
    df: pd.DataFrame


@dataclass
class BacktestResult:
    events: pd.DataFrame
    trades: pd.DataFrame
    forward_returns: pd.DataFrame
    pattern_stats: pd.DataFrame
    pair_pattern_stats: pd.DataFrame
    timeframe_stats: pd.DataFrame
    direction_stats: pd.DataFrame
    session_stats: pd.DataFrame
    month_stats: pd.DataFrame
    confirmation_stats: pd.DataFrame
    in_sample_stats: pd.DataFrame
    out_sample_stats: pd.DataFrame
    inventory: pd.DataFrame
    n_rows: int
    n_events: int
    n_trades: int
    runtime_sec: float


def run_research_backtest(
    datasets: list[DatasetSpec],
    *,
    rr_ratios=RR_RATIOS,
    max_holding_bars: int = DEFAULT_MAX_HOLDING_BARS,
    train_frac: float = 0.70,
) -> BacktestResult:
    t0 = time.perf_counter()

    all_events, all_trades, all_fwd = [], [], []
    in_events, out_events = [], []
    in_trades, out_trades = [], []
    n_rows = 0

    for ds in datasets:
        n_rows += len(ds.df)

        # Full-sample events/trades (used for the main discovery tables).
        events = detect_pattern_events(ds.df, ds.pair, ds.timeframe)
        trades = run_rr_backtest(ds.df, events, rr_ratios=rr_ratios, max_holding_bars=max_holding_bars)
        fwd = compute_forward_returns(ds.df)
        fwd = fwd.assign(pair=ds.pair, timeframe=ds.timeframe, timestamp=ds.df.index)
        if not events.empty:
            fwd_events = events.merge(fwd, on=["pair", "timeframe", "timestamp"], how="left")
            all_fwd.append(fwd_events)
        all_events.append(events)
        if not trades.empty:
            all_trades.append(trades)

        # Part 10: chronological in/out-of-sample split, detected independently.
        df_in, df_out = chronological_split(ds.df, train_frac)
        ev_in = detect_pattern_events(df_in, ds.pair, ds.timeframe)
        ev_out_raw = detect_pattern_events(df_out, ds.pair, ds.timeframe)
        tr_in = run_rr_backtest(df_in, ev_in, rr_ratios=rr_ratios, max_holding_bars=max_holding_bars)
        tr_out = run_rr_backtest(df_out, ev_out_raw, rr_ratios=rr_ratios, max_holding_bars=max_holding_bars)
        if not tr_in.empty:
            in_trades.append(tr_in)
        if not tr_out.empty:
            out_trades.append(tr_out)

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    fwd_df = pd.concat(all_fwd, ignore_index=True) if all_fwd else pd.DataFrame()
    in_trades_df = pd.concat(in_trades, ignore_index=True) if in_trades else pd.DataFrame()
    out_trades_df = pd.concat(out_trades, ignore_index=True) if out_trades else pd.DataFrame()

    baseline_1x = trades_df[(trades_df.get("entry_model") == "baseline") & (trades_df.get("rr_ratio") == rr_ratios[0])] \
        if not trades_df.empty else pd.DataFrame()

    def _safe_summary(tdf, cols, min_n=1):
        if tdf is None or tdf.empty:
            return pd.DataFrame()
        return summarize_trades(tdf, cols, min_n=min_n)

    pattern_stats = _safe_summary(baseline_1x, ["pattern"])
    pair_pattern_stats = _safe_summary(baseline_1x, ["pair", "pattern"])
    timeframe_stats = _safe_summary(baseline_1x, ["timeframe", "pattern"])
    direction_stats = _safe_summary(baseline_1x, ["pattern", "direction"])

    if not baseline_1x.empty:
        ts = pd.to_datetime(baseline_1x["timestamp"])
        session_df = baseline_1x.assign(session=session_label(ts.dt.hour.to_numpy()))
        month_df = baseline_1x.assign(month=ts.dt.to_period("M").astype(str))
        session_stats = summarize_trades(session_df, ["pattern", "session"])
        month_stats = summarize_trades(month_df, ["pattern", "month"])
    else:
        session_stats = pd.DataFrame()
        month_stats = pd.DataFrame()

    # Confirmation effect (Part 6H): baseline vs confirmed, RR=1.0, same patterns.
    if not trades_df.empty:
        confirmable_patterns = mw._CONFIRMABLE_BULLISH_1BAR | mw._CONFIRMABLE_BEARISH_1BAR
        conf_slice = trades_df[
            (trades_df["rr_ratio"] == rr_ratios[0]) & (trades_df["pattern"].isin(confirmable_patterns))
        ]
        confirmation_stats = summarize_trades(conf_slice, ["pattern", "entry_model"])
    else:
        confirmation_stats = pd.DataFrame()

    in_sample_stats = _safe_summary(
        in_trades_df[(in_trades_df.get("entry_model") == "baseline") & (in_trades_df.get("rr_ratio") == rr_ratios[0])]
        if not in_trades_df.empty else pd.DataFrame(), ["pattern"])
    out_sample_stats = _safe_summary(
        out_trades_df[(out_trades_df.get("entry_model") == "baseline") & (out_trades_df.get("rr_ratio") == rr_ratios[0])]
        if not out_trades_df.empty else pd.DataFrame(), ["pattern"])

    runtime = time.perf_counter() - t0

    return BacktestResult(
        events=events_df, trades=trades_df, forward_returns=fwd_df,
        pattern_stats=pattern_stats, pair_pattern_stats=pair_pattern_stats,
        timeframe_stats=timeframe_stats, direction_stats=direction_stats,
        session_stats=session_stats, month_stats=month_stats,
        confirmation_stats=confirmation_stats,
        in_sample_stats=in_sample_stats, out_sample_stats=out_sample_stats,
        inventory=pattern_inventory(),
        n_rows=n_rows, n_events=len(events_df), n_trades=len(trades_df),
        runtime_sec=runtime,
    )


# =============================================================================
# OUTPUT FILES (Part 12)
# =============================================================================

def write_outputs(result: BacktestResult, out_dir: str, data_is_synthetic: bool = False) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def _write_csv(df: pd.DataFrame, name: str):
        path = os.path.join(out_dir, name)
        df.to_csv(path, index=False)
        written.append(path)

    _write_csv(result.pattern_stats, "candlestick_pattern_stats.csv")
    _write_csv(result.pair_pattern_stats, "candlestick_pair_pattern_stats.csv")
    _write_csv(result.timeframe_stats, "candlestick_timeframe_stats.csv")
    _write_csv(result.session_stats, "candlestick_session_stats.csv")
    _write_csv(result.month_stats, "candlestick_month_stats.csv")
    _write_csv(result.confirmation_stats, "candlestick_confirmation_stats.csv")
    _write_csv(result.forward_returns, "candlestick_forward_returns.csv")
    _write_csv(result.inventory, "candlestick_pattern_inventory.csv")
    _write_csv(result.in_sample_stats, "candlestick_in_sample_stats.csv")
    _write_csv(result.out_sample_stats, "candlestick_out_sample_stats.csv")

    report_path = os.path.join(out_dir, "candlestick_backtest_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_build_report_markdown(result, data_is_synthetic))
    written.append(report_path)
    return written


def _top(df: pd.DataFrame, n=10, ascending=False) -> pd.DataFrame:
    if df.empty or "research_rank_score" not in df.columns:
        return df.head(0)
    return df.sort_values("research_rank_score", ascending=ascending, na_position="last").head(n)


def _build_report_markdown(result: BacktestResult, synthetic: bool) -> str:
    lines = []
    lines.append("# Candlestick Pattern Research Backtest Report\n")
    if synthetic:
        lines.append(
            "> **⚠ THIS RUN USED SYNTHETIC (randomly generated) DATA.** No real "
            "OHLC dataset was found in the uploaded project (see Part 14 notes "
            "in the accompanying summary). Every number below is a demonstration "
            "of the engine's correctness and speed, NOT a real trading result. "
            "Re-run against real EURUSD/GBPUSD/USDJPY/XAUUSD... history to get "
            "real answers.\n"
        )
    lines.append(f"- Rows processed: **{result.n_rows:,}**")
    lines.append(f"- Pattern occurrences (all patterns, all directions): **{result.n_events:,}**")
    lines.append(f"- Trades simulated (all RR ratios, both entry models): **{result.n_trades:,}**")
    lines.append(f"- Runtime: **{result.runtime_sec:.2f}s** "
                 f"({result.n_rows / max(result.runtime_sec, 1e-9):,.0f} rows/sec)\n")

    lines.append(f"\n{MULTIPLE_TESTING_WARNING}\n")

    def _section(title, df, cols=("pattern", "n", "win_rate", "avg_r", "expectancy_r", "profit_factor")):
        lines.append(f"\n## {title}\n")
        if df.empty:
            lines.append("_No qualifying data (N too small or no trades in this slice)._\n")
            return
        cols = [c for c in cols if c in df.columns]
        lines.append(_top(df, 10)[cols].to_markdown(index=False))
        lines.append("")

    _section("1-2. Best patterns overall (baseline entry, RR 1:1, ranked by expectancy x log(N))",
              result.pattern_stats)
    _section("3. Best pattern x pair combinations", result.pair_pattern_stats,
              cols=("pair", "pattern", "n", "win_rate", "avg_r", "expectancy_r", "profit_factor"))
    _section("4. Best pattern x timeframe combinations", result.timeframe_stats,
              cols=("timeframe", "pattern", "n", "win_rate", "avg_r", "expectancy_r", "profit_factor"))

    lines.append("\n## 5. Confirmed vs. unconfirmed (Part 6H)\n")
    if result.confirmation_stats.empty:
        lines.append("_No confirmable 1-bar patterns with enough occurrences in this dataset._\n")
    else:
        cols = ["pattern", "entry_model", "n", "win_rate", "expectancy_r", "profit_factor"]
        cols = [c for c in cols if c in result.confirmation_stats.columns]
        lines.append(result.confirmation_stats[cols].to_markdown(index=False))

    _section("6. Worst patterns (lowest expectancy, N>=30)",
              result.pattern_stats[result.pattern_stats.get("n_ge_30", False) == True].sort_values("expectancy_r").head(10)
              if not result.pattern_stats.empty else result.pattern_stats)

    lines.append("\n## 9. In-sample vs out-of-sample (chronological 70/30 split, Part 10)\n")
    if result.in_sample_stats.empty or result.out_sample_stats.empty:
        lines.append("_Not enough data on one side of the split to compare._\n")
    else:
        merged = result.in_sample_stats[["pattern", "n", "win_rate", "expectancy_r"]].merge(
            result.out_sample_stats[["pattern", "n", "win_rate", "expectancy_r"]],
            on="pattern", suffixes=("_in_sample", "_out_of_sample"), how="inner",
        )
        merged["sign_flip"] = np.sign(merged["expectancy_r_in_sample"]) != np.sign(merged["expectancy_r_out_of_sample"])
        lines.append(merged.to_markdown(index=False))
        lines.append("\n_A `sign_flip=True` row means the pattern's edge did NOT survive "
                     "out-of-sample — treat with suspicion regardless of how good the "
                     "in-sample number looked._\n")

    lines.append("\n## 10. Sample-size warnings\n")
    if not result.pattern_stats.empty:
        low = result.pattern_stats[result.pattern_stats["low_sample_warning"]]
        if not low.empty:
            lines.append(f"{len(low)} pattern(s) have fewer than 30 occurrences overall and "
                         "are excluded from the ranked tables above:\n")
            lines.append(", ".join(sorted(low["pattern"].astype(str).tolist())))
        else:
            lines.append("No patterns fell below the N>=30 threshold in the overall table.")
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# SYNTHETIC SELF-TEST / DEMO (used only when no real data is found)
# =============================================================================

def _make_synthetic_dataset(pair: str, timeframe: str, n: int, seed: int, base_price: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    freq_map = {"M15": "15min", "H1": "1h", "H4": "4h"}
    idx = pd.date_range("2018-01-01", periods=n, freq=freq_map.get(timeframe, "1h"), tz="UTC")
    close = base_price + np.cumsum(rng.normal(0, base_price * 0.0006, n))
    open_ = close + rng.normal(0, base_price * 0.0002, n)
    high = np.maximum(open_, close) + rng.uniform(base_price * 0.0001, base_price * 0.0008, n)
    low = np.minimum(open_, close) - rng.uniform(base_price * 0.0001, base_price * 0.0008, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def run_self_test_and_benchmark(out_dir: str) -> BacktestResult:
    """No real OHLC data was found (Part 14) — build a multi-pair,
    multi-timeframe SYNTHETIC dataset purely to prove the engine is
    correct and fast, and to produce example output files with the
    expected shape. Every number in the report is clearly labeled
    synthetic."""
    specs = []
    pair_prices = {"EURUSD": 1.0850, "GBPUSD": 1.2650, "USDJPY": 150.25, "XAUUSD": 2350.0, "AUDUSD": 0.6550}
    seed = 0
    for pair, price in pair_prices.items():
        for tf, n in (("M15", 20000), ("H1", 8000), ("H4", 3000)):
            specs.append(DatasetSpec(pair=pair, timeframe=tf,
                                      df=_make_synthetic_dataset(pair, tf, n, seed, price)))
            seed += 1

    result = run_research_backtest(specs)
    write_outputs(result, out_dir, data_is_synthetic=True)
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Candlestick pattern research backtester")
    ap.add_argument("--data-dir", default=None, help="Directory containing OHLC CSV/Parquet files")
    ap.add_argument("--out-dir", default="./candlestick_research_output")
    ap.add_argument("--demo", action="store_true", help="Force the synthetic self-test/demo")
    args = ap.parse_args()

    if args.demo or not args.data_dir:
        print("No --data-dir given (or --demo forced) — running the synthetic self-test/benchmark.")
        res = run_self_test_and_benchmark(args.out_dir)
    else:
        files = discover_ohlc_files(args.data_dir)
        if not files:
            print(f"No OHLC .csv/.parquet files found under: {args.data_dir}")
            print("Cannot run the research backtest without real data. "
                  "Provide a directory of OHLC files (any of the naming/column "
                  "conventions handled by load_ohlc_file()) and re-run.")
            sys.exit(1)
        specs = []
        for f in files:
            pair, tf = infer_pair_timeframe(f)
            try:
                df = load_ohlc_file(f)
            except Exception as e:
                print(f"  SKIPPED {f}: {e}")
                continue
            specs.append(DatasetSpec(pair=pair, timeframe=tf, df=df))
            print(f"  loaded {f}: pair={pair} timeframe={tf} rows={len(df):,}")
        res = run_research_backtest(specs)
        write_outputs(res, args.out_dir, data_is_synthetic=False)

    print(f"\nProcessed:")
    print(f"  {len(set(zip(res.events['pair'], res.events['timeframe']))) if not res.events.empty else 0} pair/timeframe datasets")
    print(f"  {res.n_rows:,} candles")
    print(f"  {res.n_events:,} pattern occurrences")
    print(f"  {res.n_trades:,} simulated trades (all RR ratios x both entry models)")
    print(f"\nRuntime: {res.runtime_sec:.2f} seconds")
    print(f"Throughput: {res.n_rows / max(res.runtime_sec, 1e-9):,.0f} rows/sec")
    print(f"\nOutputs written to: {args.out_dir}")