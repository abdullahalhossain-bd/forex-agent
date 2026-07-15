"""execution/structure_exit.py — Structure-based exit
=================================================================
FIXES applied (audit follow-up):
  1. trade / df input validation — no more crashes on None or
     missing columns; return (False, reason) instead.
  2. NaN-safety — a NaN in the latest bar's high/low/close/atr no
     longer silently resolves to "Structure intact". We detect it
     and return an explicit reason instead of pretending the
     structure is fine.
  3. Unknown `method` value now logs a warning instead of silently
     falling through to "Structure intact".
  4. compute_structure_tp() now validates `direction` the same way
     check_structure_exit() does, instead of silently treating any
     non-"BUY" value as SELL.
  5. Logger is now actually used (trigger events + rejected inputs).
  6. Unused `pandas` import removed (df is duck-typed; we never call
     pd.* directly).
=================================================================
"""
from __future__ import annotations

import math
from utils.logger import get_logger

log = get_logger("structure_exit")

REQUIRED_COLUMNS = ("high", "low", "close")


def _is_nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True  # non-numeric / missing → treat as unusable


def _validate_common(df, trade, lookback) -> tuple[bool, str]:
    """Shared validation for both functions. Returns (ok, reason)."""
    if df is None or len(df) < lookback:
        return False, "Insufficient data"
    if not isinstance(trade, dict):
        return False, "Invalid trade object (expected dict)"
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"df missing required columns: {missing}"
    return True, ""


def check_structure_exit(df, trade, method="bos", lookback=20, buffer_atr_mult=0.2):
    """
    Structure-based exit check (Break of Structure / Change of Character).

    Returns (should_exit: bool, reason: str).
    """
    ok, reason = _validate_common(df, trade, lookback)
    if not ok:
        return False, reason

    last = df.iloc[-1]

    if _is_nan(last["close"]) or _is_nan(last["high"]) or _is_nan(last["low"]):
        log.warning("[StructureExit] Latest bar has NaN OHLC — refusing to evaluate structure")
        return False, "Latest bar has NaN price data — cannot evaluate structure"

    close = float(last["close"])
    atr_raw = last.get("atr", 0.001)
    atr = float(atr_raw) if not _is_nan(atr_raw) else 0.001
    atr = atr or 0.001

    direction = str(trade.get("direction", trade.get("type", ""))).upper()
    if direction not in ("BUY", "SELL"):
        return False, f"Invalid direction: {direction!r}"

    window = df.iloc[-lookback - 1:-1] if len(df) > lookback else df.iloc[:-1]
    if len(window) == 0:
        return False, "No prior bars"

    swing_high = float(window["high"].max())
    swing_low = float(window["low"].min())
    if _is_nan(swing_high) or _is_nan(swing_low):
        return False, "Prior swing window has NaN price data — cannot evaluate structure"

    buffer = atr * buffer_atr_mult

    if method == "bos":
        if direction == "BUY" and close < swing_low - buffer:
            msg = f"Bearish BOS: close {close:.5f} < swing {swing_low:.5f}"
            log.info(f"[StructureExit] Exit triggered — {msg}")
            return True, msg
        if direction == "SELL" and close > swing_high + buffer:
            msg = f"Bullish BOS: close {close:.5f} > swing {swing_high:.5f}"
            log.info(f"[StructureExit] Exit triggered — {msg}")
            return True, msg

    elif method == "choch":
        sw = df.iloc[-6:-1] if len(df) > 5 else df.iloc[:-1]
        if len(sw) == 0:
            return False, "No prior bars"
        sh = float(sw["high"].max())
        sl = float(sw["low"].min())
        if _is_nan(sh) or _is_nan(sl):
            return False, "CHoCH window has NaN price data — cannot evaluate structure"

        if direction == "BUY" and float(last["low"]) < sl and close < sl + buffer:
            log.info("[StructureExit] Exit triggered — CHoCH bearish")
            return True, "CHoCH bearish"
        if direction == "SELL" and float(last["high"]) > sh and close > sh - buffer:
            log.info("[StructureExit] Exit triggered — CHoCH bullish")
            return True, "CHoCH bullish"

    else:
        log.warning(f"[StructureExit] Unknown method {method!r} — no exit check performed")
        return False, f"Unknown method: {method!r}"

    return False, "Structure intact"


def compute_structure_tp(df, direction, entry, lookback=50):
    """
    Structure-based take-profit target (nearest opposing swing extreme,
    falling back to a symmetric projection if price hasn't reached entry yet).
    """
    if df is None or len(df) < 10:
        return entry

    direction_u = str(direction).upper()
    if direction_u not in ("BUY", "SELL"):
        log.warning(f"[StructureExit] Invalid direction {direction!r} in compute_structure_tp — returning entry")
        return entry

    window = df.iloc[-lookback:] if len(df) >= lookback else df
    if "high" not in window.columns or "low" not in window.columns:
        log.warning("[StructureExit] df missing high/low columns — returning entry")
        return entry

    if direction_u == "BUY":
        tp = float(window["high"].max())
        if _is_nan(tp):
            return entry
        low_min = float(window["low"].min())
        return tp if tp > entry else entry + (entry - low_min)
    else:
        tp = float(window["low"].min())
        if _is_nan(tp):
            return entry
        high_max = float(window["high"].max())
        return tp if tp < entry else entry - (high_max - entry)