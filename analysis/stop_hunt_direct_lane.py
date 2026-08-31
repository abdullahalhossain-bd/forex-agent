"""
analysis/stop_hunt_direct_lane.py — Direct signal lane for the validated
StopHuntSignalEngine strategy.

PURPOSE
=======
This module exists to close the implementation-parity gap between
`backtest/per_strategy_tester.py::_test_stop_hunt` (which validated a real
out-of-sample edge of 76.4%->80.4% / 65.5%->69.5% across two independent
15-pair sets) and `main.py`'s backtest/live execution (which previously
produced ~20-30% win rate on the same setup).

ROOT CAUSE THIS MODULE FIXES
============================
`core/trader.py:1008-1050` (the "Stop Hunt Direct Lane" block) imports
`get_stop_hunt_direct_signal` from this module. Before this file existed,
that import raised `ModuleNotFoundError`, which was silently swallowed by
the surrounding `except Exception as _e_shdl: log.debug(...)` at line
1049-1050. As a result:

  1. The direct-lane code path NEVER fired — `dec_out["direct_lane"]`
     was never set, so the downstream `direct_lane` bypasses in
     `risk/trade_permission.py` (lines 160, 201, 810, 912) were dead.
  2. main.py fell back to the full ~29-module analysis blend, whose
     consensus-based confidence gating rejects ~90%+ of the bars the
     standalone StopHuntSignalEngine would have traded.

The validated tester setup runs StopHuntSignalEngine SOLO (not in a
blend), with exactly four filters applied in order:
  1. London/NY session window (08:00-22:00 GMT)
  2. Exclude Wednesday (underperforms every other weekday)
  3. H4 trend agreement (H4 EMA20 vs EMA50 in the signal's direction)
  4. S/R proximity (within 0.5x ATR of an S/R zone, correct side)

This module replicates that exact logic for the most recent closed bar
of `df` (i.e., df.iloc[-1]). It is called from
`core/trader.py:evaluate_decision_core` only when the blended pipeline
returned WAIT/NO_TRADE — so it never overrides a trade the blend already
approved, and only fires on bars the blend would otherwise have skipped.

NO-LOOK-AHEAD CONTRACT
======================
`df` passed in is `market_out["df"]`, which (per backtest/unified_engine.py
line 209) is the slice `df.iloc[:i+1]` for the current bar i. So df.iloc[-1]
IS the current closed bar, and all four filters below use only data at
or before that bar — never forward.

For the H4 trend filter, we use the most recent H4 bar whose close time
is <= the current H1 bar's close time. This matches the tester's exact
indexing convention (per_strategy_tester.py:996-1000).

For S/R proximity, we use df.iloc[max(0, i-200):i+1] (a 200-bar trailing
window ending at the current bar), exactly mirroring the tester.

PARAMETERS
==========
The four filter parameters below mirror the tester's defaults EXACTLY.
Do NOT tune them without re-validating against per_strategy_tester.py.
The tester's defaults were chosen because they reproduced on two
independent out-of-sample 15-pair sets — changing them silently
invalidates that validation.

RETURN
======
dict with keys:
    action:        "BUY" | "SELL"
    entry:         float
    stop_loss:     float
    take_profit:   float
    reason:        str  (human-readable summary of filters applied)
    confidence:    int  (0-100; not used for gating, only logging —
                          the direct_lane path bypasses the confidence
                          gate in trade_permission.py)

or None if:
    - df has < 100 bars (StopHuntSignalEngine needs a 100-bar window)
    - StopHuntSignalEngine returns no BUY/SELL signal at the last bar
    - any of the four filters rejects the signal
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Filter parameters (mirror per_strategy_tester._test_stop_hunt defaults) ──
# DO NOT tune without re-validating against the tester. See module docstring.
SESSION_START_HOUR_GMT = 8     # require_london_ny_session: 8 <= hour < 22
SESSION_END_HOUR_GMT = 22
EXCLUDE_WEEKDAY_WED = True     # exclude_wednesday (weekday()==2)
H4_EMA_FAST = 20               # _h4_ema20
H4_EMA_SLOW = 50               # _h4_ema50
SR_PROXIMITY_ATR_MULT = 0.5    # sr_proximity_atr
ATR_PERIOD = 14                # self._atr(df, idx, period=14)
RR_RATIO = 2.0                 # self.rr_ratio
ATR_STOP_MULT = 1.5            # 1.5 * ATR for SL distance (tester line 1032-1034)
MIN_BARS_FOR_ENGINE = 50       # StopHuntSignalEngine needs ~50 bars minimum
                                 # (mirrors tester's `start = max(50, ...)` floor)


def _pip_size(pair: str) -> float:
    pair = pair.upper()
    if "JPY" in pair:
        return 0.01
    if "XAU" in pair or "XAG" in pair:
        return 0.1
    return 0.0001


def _atr(df: pd.DataFrame, idx: int, period: int = ATR_PERIOD) -> float:
    """Compute ATR at idx, mirroring per_strategy_tester._atr (line 1502-1517)."""
    if idx < period:
        return 0.0
    window = df.iloc[idx - period:idx + 1]
    high = window["high"].values
    low = window["low"].values
    close = window["close"].values
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    return float(np.mean(tr[-period:]))


def get_stop_hunt_direct_signal(
    df: pd.DataFrame,
    symbol: str,
    df_h4: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """Compute the validated StopHuntSignalEngine signal at df.iloc[-1],
    applying the four-filter combo (session + noWed + H4 trend + SR proximity).

    Args:
        df: H1 OHLCV with DatetimeIndex (UTC). Must contain >= 100 bars.
            In backtest mode this is the no-look-ahead slice df.iloc[:i+1].
        symbol: Trading pair, e.g. "EURUSD". Used for pip size + logging.
        df_h4: Optional H4 OHLCV with DatetimeIndex (UTC). If provided AND
            has >= 55 bars before the current H1 bar's close time, the
            H4 EMA20/EMA50 trend-agreement filter is applied. If None or
            insufficient history, the H4 filter is skipped (matches tester
            behavior — see per_strategy_tester.py:961-964).

    Returns:
        dict with action/entry/stop_loss/take_profit/reason/confidence, or None.
    """
    if df is None or len(df) < MIN_BARS_FOR_ENGINE:
        # 2026-08-13: removed debug print() — was firing every bar
        return None

    # v3.21 (live parity): the "direct lane" is a bypass that generates a
    # SELL straight from StopHuntSignalEngine even when the DecisionAgent
    # said WAIT — i.e. a trade with no master-verdict approval. The
    # v3.18-v3.21 debug sessions proved this lane causes unauthorized
    # losses, so it is DISABLED by default now. Set env
    # SZ_DISABLE_STOP_HUNT_LANE=0 to restore the legacy behavior.
    if str(os.getenv("SZ_DISABLE_STOP_HUNT_LANE", "1")).strip() == "1":
        return None

    # Late import — same module the tester uses (analysis/stop_hunt_signal_engine.py)
    from analysis.stop_hunt_signal_engine import StopHuntSignalEngine

    # ── 0. Run StopHuntSignalEngine on the trailing 100-bar window ─────────
    # Mirrors tester line 980: window = df.iloc[max(0, i-100):i+1]
    i = len(df) - 1  # current bar index
    window = df.iloc[max(0, i - 100):i + 1]
    try:
        engine = StopHuntSignalEngine()
        sig = engine.analyze(window)
    except Exception as e:
        log.debug(f"[stop_hunt_direct_lane] engine.analyze failed: {e}")
        return None

    if sig is None:
        return None
    signal = sig.get("signal", {})
    action = signal.get("action", "NO_TRADE")
    if action not in ("BUY", "SELL"):
        return None

    direction = "long" if action == "BUY" else "short"
    current_time = df.index[i]
    reasons = []

    # ── 1. London/NY session filter (08:00-22:00 GMT) ─────────────────────
    # Mirrors tester line 991-992.
    hour = current_time.hour
    if not (SESSION_START_HOUR_GMT <= hour < SESSION_END_HOUR_GMT):
        return None
    reasons.append(f"session {hour:02d}:00 GMT ∈ [08,22]")

    # ── 2. Exclude Wednesday filter ──────────────────────────────────────
    # Mirrors tester line 993-994. weekday()==2 is Wednesday.
    if EXCLUDE_WEEKDAY_WED and current_time.weekday() == 2:
        return None
    reasons.append(f"weekday={current_time.weekday()} (!=2 Wed)")

    # ── 3. H4 trend-agreement filter (only if df_h4 provided & sufficient) ─
    # Mirrors tester lines 961-964 + 995-1003 exactly.
    if df_h4 is not None and len(df_h4) >= H4_EMA_SLOW + 5:
        _h4_ema20 = df_h4["close"].ewm(span=H4_EMA_FAST, adjust=False).mean()
        _h4_ema50 = df_h4["close"].ewm(span=H4_EMA_SLOW, adjust=False).mean()
        # Use the most recent H4 bar with close_time <= current H1 bar's close_time
        # — same convention as tester line 996.
        h4_idx_mask = _h4_ema20.index <= current_time
        h4_idx = _h4_ema20.index[h4_idx_mask]
        if len(h4_idx) < H4_EMA_SLOW:
            # Not enough H4 history yet — tester skips (line 997-998)
            return None
        hi = _h4_ema20.index.get_loc(h4_idx[-1])
        h4_trend = "up" if _h4_ema20.iloc[hi] > _h4_ema50.iloc[hi] else "down"
        if not (
            (direction == "long" and h4_trend == "up")
            or (direction == "short" and h4_trend == "down")
        ):
            return None
        reasons.append(f"H4 trend={h4_trend} agrees with {direction}")

    # ── 4. S/R proximity filter (within 0.5x ATR of correct-side zone) ────
    # Mirrors tester lines 1004-1022 exactly.
    atr_now = _atr(df, i)
    if atr_now <= 0:
        return None
    try:
        from analysis.support_resistance import SupportResistance
        sr_window = df.iloc[max(0, i - 200):i + 1]
        # FIX (Finding #8, S/R correctness audit): this module's contract
        # (see module docstring) is H1 OHLCV only, and the engine's
        # default timeframe already happens to be "H1" — so this was not
        # a live bug — but leaving it implicit means a future default
        # change elsewhere in support_resistance.py would silently break
        # this module's tester-parity guarantee. Made explicit.
        sr_res = SupportResistance(timeframe="H1").analyze(sr_window)
    except Exception as e:
        log.debug(f"[stop_hunt_direct_lane] S/R analyze failed: {e}")
        sr_res = {}
    price_now = float(df.iloc[i]["close"])
    max_dist = SR_PROXIMITY_ATR_MULT * atr_now
    if direction == "long":
        near = any(
            abs(price_now - z.get("zone_top", price_now)) <= max_dist
            for z in sr_res.get("support_zones", [])
        )
        if not near:
            return None
        reasons.append(f"near support (within {SR_PROXIMITY_ATR_MULT}×ATR)")
    else:
        near = any(
            abs(price_now - z.get("zone_bottom", price_now)) <= max_dist
            for z in sr_res.get("resistance_zones", [])
        )
        if not near:
            return None
        reasons.append(f"near resistance (within {SR_PROXIMITY_ATR_MULT}×ATR)")

    # ── 5. Compute entry / SL / TP — EXACTLY as tester does ──────────────
    # Mirrors tester lines 1024-1034.
    entry = signal.get("entry_price") or float(df.iloc[i]["close"])
    stop = signal.get("stop_loss")
    tp = signal.get("take_profit")
    if not stop or not tp:
        # Fallback: 1.5*ATR stop, 1.5*ATR*RR_RATIO take-profit
        # (tester lines 1032-1034)
        if direction == "long":
            stop = entry - ATR_STOP_MULT * atr_now
            tp = entry + ATR_STOP_MULT * atr_now * RR_RATIO
        else:
            stop = entry + ATR_STOP_MULT * atr_now
            tp = entry - ATR_STOP_MULT * atr_now * RR_RATIO

    confidence = signal.get("confidence", "Medium")
    # Numeric confidence for trade_permission.py's logging — direct_lane
    # bypasses the MIN_CONFIDENCE gate (see risk/trade_permission.py:810-817),
    # so this value is informational only.
    conf_numeric = 80 if confidence == "High" else 70 if confidence == "Medium" else 60

    return {
        "action": action,
        "entry": float(entry),
        "stop_loss": float(stop),
        "take_profit": float(tp),
        "reason": f"StopHunt direct lane: {'; '.join(reasons)}",
        "confidence": conf_numeric,
    }