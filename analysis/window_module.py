# analysis/window_module.py — Nison-style rising/falling "window" (gap) rules
# =============================================================================
# NOTE ON NAMING: this module is intentionally separate from
# analysis/fvg_detector.py. That module detects ICT-style "Fair Value Gaps"
# (a 3-candle body imbalance, filtered by a minimum ATR fraction, generally
# treated as a mean-reversion magnet to be filled). This module implements a
# different, older definition from candlestick theory (Nison):
#
#   - A window exists between two ADJACENT candles only if the SHADOWS
#     (not just the real bodies) do not overlap. Real-body-only gap checks
#     over-report windows.
#   - Window size is IRRELEVANT to validity — even a one-pip window is a
#     legitimate support/resistance zone. Do not filter windows by an
#     ATR/min-size threshold (that is the FVG module's job, not this one).
#   - A rising window is bullish-continuation support; a falling window is
#     bearish-continuation resistance. The support/resistance zone is the
#     ENTIRE window range (top to bottom), not a single price.
#   - Trend from a window stays intact no matter how many windows open in
#     sequence — it only breaks when the LAST (most recent) window's
#     support/resistance is closed through. Do NOT hardcode a "trend
#     exhausted after N windows" rule.
#   - A window (support/resistance) always overrides an opposing single
#     candle signal until the window itself is voided by a close through it.
#
# This module never fabricates a trade signal on its own — it exists to be
# queried by other modules (candlestick pattern detectors, the confluence
# engine) so they can check "is there an unclosed window between me and the
# proposed trade direction?" before acting on a pattern.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Window:
    kind: str            # "rising" (bullish support) or "falling" (bearish resistance)
    open_index: int       # index of the FIRST candle of the pair (the one before the gap)
    close_index: int      # index of the SECOND candle of the pair (the one after the gap)
    top: float            # top of the window range
    bottom: float         # bottom of the window range
    voided: bool = False
    voided_at_index: Optional[int] = None

    @property
    def signal(self) -> str:
        return "bullish_continuation" if self.kind == "rising" else "bearish_continuation"


def detect_windows(
    df: pd.DataFrame,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> List[Window]:
    """
    Scan a DataFrame for Nison-style windows (gaps) using SHADOW overlap, not
    just real-body overlap. Size is irrelevant — any non-overlap counts.

    A rising window exists when candle[i].low > candle[i-1].high
    (candle i's entire shadow sits above candle i-1's entire shadow).
    A falling window exists when candle[i].high < candle[i-1].low.

    Each window is checked bar-by-bar going forward for a void condition
    (close through the window boundary). Once voided, later bars can no
    longer reference that window as active support/resistance.
    """
    highs = df[high_col].to_numpy(dtype=float)
    lows = df[low_col].to_numpy(dtype=float)
    closes = df[close_col].to_numpy(dtype=float)
    n = len(df)

    windows: List[Window] = []

    for i in range(1, n):
        prev_h, prev_l = highs[i - 1], lows[i - 1]
        cur_h, cur_l = highs[i], lows[i]

        if cur_l > prev_h:
            windows.append(Window(
                kind="rising", open_index=i - 1, close_index=i,
                top=cur_l, bottom=prev_h,
            ))
        elif cur_h < prev_l:
            windows.append(Window(
                kind="falling", open_index=i - 1, close_index=i,
                top=prev_l, bottom=cur_h,
            ))

    # Apply void conditions: a rising window voids on close < bottom;
    # a falling window voids on close > top. Scan forward from the window's
    # formation bar.
    for w in windows:
        for j in range(w.close_index + 1, n):
            if w.kind == "rising" and closes[j] < w.bottom:
                w.voided = True
                w.voided_at_index = j
                break
            if w.kind == "falling" and closes[j] > w.top:
                w.voided = True
                w.voided_at_index = j
                break

    return windows


def active_window_bias(windows: List[Window], as_of_index: int) -> Optional[str]:
    """
    Per the meta priority order (window support/resistance ranks ABOVE any
    opposing single-candle signal): return the bias implied by the most
    recent NOT-YET-VOIDED window as of `as_of_index`, or None if there is
    no active window yet.

    Trend from windows persists across multiple stacked windows in the same
    direction — only the most recent window's boundary matters, per the
    author's updated rule (the old "3 windows = exhaustion" heuristic is
    deliberately NOT implemented here).
    """
    candidate: Optional[Window] = None
    for w in windows:
        if w.close_index > as_of_index:
            continue
        if w.voided and w.voided_at_index is not None and w.voided_at_index <= as_of_index:
            continue
        if candidate is None or w.close_index > candidate.close_index:
            candidate = w
    return candidate.signal if candidate else None


def window_overrides_candle_signal(
    windows: List[Window],
    as_of_index: int,
    candle_signal: str,
) -> bool:
    """
    Implements meta.priority_order_when_signals_conflict rule #1: an active
    window's support/resistance always overrides an opposing single-candle
    signal (e.g. a bullish hammer printed inside a still-open falling window
    is still bearish context until that window's resistance is closed
    through).

    Returns True if `candle_signal` ("bullish_reversal" / "bearish_reversal")
    should be SUPPRESSED because it conflicts with the active window bias.
    """
    bias = active_window_bias(windows, as_of_index)
    if bias is None:
        return False
    if bias == "bullish_continuation" and candle_signal == "bearish_reversal":
        return True
    if bias == "bearish_continuation" and candle_signal == "bullish_reversal":
        return True
    return False


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open":  [1.10, 1.11, 1.115, 1.13, 1.128, 1.127, 1.126, 1.10, 1.08, 1.07],
        "high":  [1.112, 1.113, 1.118, 1.135, 1.130, 1.129, 1.128, 1.115, 1.09, 1.075],
        "low":   [1.098, 1.108, 1.113, 1.127, 1.120, 1.124, 1.122, 1.075, 1.07, 1.065],
        "close": [1.11, 1.112, 1.116, 1.129, 1.121, 1.126, 1.124, 1.08, 1.075, 1.068],
    }, index=idx)

    ws = detect_windows(df)
    print(f"Windows found: {len(ws)}")
    for w in ws:
        print(f"  {w.kind} window bars[{w.open_index}->{w.close_index}] "
              f"range=({w.bottom:.4f},{w.top:.4f}) voided={w.voided} at={w.voided_at_index}")

    bias = active_window_bias(ws, as_of_index=9)
    print(f"\nActive bias at bar 9: {bias}")
    print("Window module smoke test passed.")