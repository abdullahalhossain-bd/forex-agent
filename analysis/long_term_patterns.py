# analysis/long_term_patterns.py — Long-term topping/bottoming patterns
# =============================================================================
# These are multi-week/multi-month reversal patterns built from candle-chart
# swing points (not single/two/three-candle patterns). They are the
# candlestick-chart equivalents of Western chart patterns:
#
#   three_mountain_top       ~ triple top
#   three_buddha_top         ~ head and shoulders top
#   three_river_bottom       ~ inverse head and shoulders (bottom)
#   inverted_three_buddha    ~ triple bottom
#   dumpling_top             ~ rounding top
#   frypan_bottom            ~ rounding bottom
#   tower_top                ~ sharp V-reversal top (fast up, fast down)
#   tower_bottom             ~ sharp V-reversal bottom (fast down, fast up)
#
# Detection here works on SWING points (local highs/lows over a lookback
# window), not raw bars, since these patterns only make sense at that scale.
# Each detector returns a confidence-scored candidate; nothing here fires a
# trade on its own — per global_filters, these require the same
# confirmation + confluence gating as any other reversal signal.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class LongTermPattern:
    id: str
    western_equiv: str
    start_index: int
    end_index: int
    confirm_index: Optional[int]
    neckline: Optional[float]
    confidence: float
    note: str = ""


def _swing_points(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                   lookback: int = 5) -> tuple[list[int], list[int]]:
    """Local swing highs/lows: index i is a swing high if it's the max high
    within +/-lookback bars, swing low if it's the min low within +/-lookback."""
    n = len(closes)
    swing_highs, swing_lows = [], []
    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback:i + lookback + 1]
        window_l = lows[i - lookback:i + lookback + 1]
        if highs[i] == window_h.max():
            swing_highs.append(i)
        if lows[i] == window_l.min():
            swing_lows.append(i)
    return swing_highs, swing_lows


def detect_three_mountain_top(df, swing_highs, highs, lows, closes,
                               tolerance_pct: float = 0.01) -> List[LongTermPattern]:
    """Three peaks at roughly the same level (triple top)."""
    out = []
    for a, b, c in zip(swing_highs, swing_highs[1:], swing_highs[2:]):
        ha, hb, hc = highs[a], highs[b], highs[c]
        avg = (ha + hb + hc) / 3.0
        if avg == 0:
            continue
        if max(ha, hb, hc) - min(ha, hb, hc) <= avg * tolerance_pct:
            trough_between = min(lows[a:c + 1]) if c > a else None
            out.append(LongTermPattern(
                id="three_mountain_top", western_equiv="triple_top",
                start_index=a, end_index=c, confirm_index=None,
                neckline=trough_between, confidence=0.6,
                note="Three peaks within tolerance; confirm on close below neckline.",
            ))
    return out


def detect_three_buddha_top(df, swing_highs, highs, lows, closes) -> List[LongTermPattern]:
    """Head & shoulders: middle peak clearly higher than two similar shoulders."""
    out = []
    for a, b, c in zip(swing_highs, swing_highs[1:], swing_highs[2:]):
        ha, hb, hc = highs[a], highs[b], highs[c]
        if hb > ha and hb > hc and abs(ha - hc) <= max(ha, hc) * 0.015:
            neckline = min(min(lows[a:b + 1]), min(lows[b:c + 1]))
            out.append(LongTermPattern(
                id="three_buddha_top", western_equiv="head_and_shoulders_top",
                start_index=a, end_index=c, confirm_index=None,
                neckline=neckline, confidence=0.65,
                note="Head clearly above two similar shoulders; confirm on neckline break.",
            ))
    return out


def detect_three_river_bottom(df, swing_lows, highs, lows, closes) -> List[LongTermPattern]:
    """Inverse head & shoulders: middle trough clearly lower than two similar troughs."""
    out = []
    for a, b, c in zip(swing_lows, swing_lows[1:], swing_lows[2:]):
        la, lb, lc = lows[a], lows[b], lows[c]
        if lb < la and lb < lc and abs(la - lc) <= max(la, lc) * 0.015:
            neckline = max(max(highs[a:b + 1]), max(highs[b:c + 1]))
            out.append(LongTermPattern(
                id="three_river_bottom", western_equiv="inverse_head_and_shoulders",
                start_index=a, end_index=c, confirm_index=None,
                neckline=neckline, confidence=0.65,
                note="Head clearly below two similar shoulders; confirm on neckline break.",
            ))
    return out


def detect_inverted_three_buddha(df, swing_lows, highs, lows, closes,
                                  tolerance_pct: float = 0.01) -> List[LongTermPattern]:
    """Triple bottom: three troughs at roughly the same level."""
    out = []
    for a, b, c in zip(swing_lows, swing_lows[1:], swing_lows[2:]):
        la, lb, lc = lows[a], lows[b], lows[c]
        avg = (la + lb + lc) / 3.0
        if avg == 0:
            continue
        if max(la, lb, lc) - min(la, lb, lc) <= avg * tolerance_pct:
            neckline = max(highs[a:c + 1]) if c > a else None
            out.append(LongTermPattern(
                id="inverted_three_buddha", western_equiv="triple_bottom",
                start_index=a, end_index=c, confirm_index=None,
                neckline=neckline, confidence=0.6,
                note="Three troughs within tolerance; confirm on close above neckline.",
            ))
    return out


def detect_dumpling_top(df, highs, lows, closes, window: int = 15,
                         flatness_pct: float = 0.003) -> List[LongTermPattern]:
    """Rounding top: a slow convex roll-over — rising then flattening then
    falling closes over `window` bars, with a small gap/drop-off at the end
    per Nison's original description."""
    out = []
    n = len(closes)
    for i in range(window, n - window):
        seg = closes[i - window:i + window]
        first_half_slope = seg[window // 2] - seg[0]
        second_half_slope = seg[-1] - seg[-window // 2]
        mid_flatness = np.std(seg[window // 2: window + window // 2])
        if first_half_slope > 0 and second_half_slope < 0 and mid_flatness < np.mean(seg) * flatness_pct:
            out.append(LongTermPattern(
                id="dumpling_top", western_equiv="rounding_top",
                start_index=i - window, end_index=i + window, confirm_index=i,
                neckline=None, confidence=0.5,
                note="Convex roll-over: rise, flatten, roll down.",
            ))
    return out


def detect_frypan_bottom(df, highs, lows, closes, window: int = 15,
                          flatness_pct: float = 0.003) -> List[LongTermPattern]:
    """Rounding bottom: mirror image of dumpling_top."""
    out = []
    n = len(closes)
    for i in range(window, n - window):
        seg = closes[i - window:i + window]
        first_half_slope = seg[window // 2] - seg[0]
        second_half_slope = seg[-1] - seg[-window // 2]
        mid_flatness = np.std(seg[window // 2: window + window // 2])
        if first_half_slope < 0 and second_half_slope > 0 and mid_flatness < np.mean(seg) * flatness_pct:
            out.append(LongTermPattern(
                id="frypan_bottom", western_equiv="rounding_bottom",
                start_index=i - window, end_index=i + window, confirm_index=i,
                neckline=None, confidence=0.5,
                note="Concave bowl: fall, flatten, roll up.",
            ))
    return out


def detect_tower_top(df, highs, lows, closes, run_len: int = 5,
                      move_pct: float = 0.02) -> List[LongTermPattern]:
    """Sharp V-reversal at a top: a fast multi-bar rally immediately followed
    by an equally fast multi-bar decline, with little consolidation."""
    out = []
    n = len(closes)
    for i in range(run_len, n - run_len):
        up_move = (closes[i] - closes[i - run_len]) / closes[i - run_len]
        down_move = (closes[i + run_len] - closes[i]) / closes[i]
        if up_move > move_pct and down_move < -move_pct:
            out.append(LongTermPattern(
                id="tower_top", western_equiv="v_reversal_top",
                start_index=i - run_len, end_index=i + run_len, confirm_index=i,
                neckline=None, confidence=0.55,
                note="Sharp rally immediately reversed by an equally sharp decline.",
            ))
    return out


def detect_tower_bottom(df, highs, lows, closes, run_len: int = 5,
                         move_pct: float = 0.02) -> List[LongTermPattern]:
    """Sharp V-reversal at a bottom: fast decline immediately followed by a
    fast rally."""
    out = []
    n = len(closes)
    for i in range(run_len, n - run_len):
        down_move = (closes[i] - closes[i - run_len]) / closes[i - run_len]
        up_move = (closes[i + run_len] - closes[i]) / closes[i]
        if down_move < -move_pct and up_move > move_pct:
            out.append(LongTermPattern(
                id="tower_bottom", western_equiv="v_reversal_bottom",
                start_index=i - run_len, end_index=i + run_len, confirm_index=i,
                neckline=None, confidence=0.55,
                note="Sharp decline immediately reversed by an equally sharp rally.",
            ))
    return out


def detect_all(df: pd.DataFrame, *, swing_lookback: int = 5,
                high_col: str = "high", low_col: str = "low",
                close_col: str = "close") -> List[LongTermPattern]:
    """Run all 8 long-term topping/bottoming detectors and return every
    candidate found (unfiltered — the confluence engine should gate these
    the same way it gates any other reversal signal)."""
    highs = df[high_col].to_numpy(dtype=float)
    lows = df[low_col].to_numpy(dtype=float)
    closes = df[close_col].to_numpy(dtype=float)

    swing_highs, swing_lows = _swing_points(closes, highs, lows, swing_lookback)

    results: List[LongTermPattern] = []
    results += detect_three_mountain_top(df, swing_highs, highs, lows, closes)
    results += detect_three_buddha_top(df, swing_highs, highs, lows, closes)
    results += detect_three_river_bottom(df, swing_lows, highs, lows, closes)
    results += detect_inverted_three_buddha(df, swing_lows, highs, lows, closes)
    results += detect_dumpling_top(df, highs, lows, closes)
    results += detect_frypan_bottom(df, highs, lows, closes)
    results += detect_tower_top(df, highs, lows, closes)
    results += detect_tower_bottom(df, highs, lows, closes)
    return results


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    rng = np.random.default_rng(7)

    # Build a synthetic triple-top-ish series: three similar peaks
    base = np.linspace(1.00, 1.20, n)
    wave = 0.03 * np.sin(np.linspace(0, 6 * np.pi, n))
    noise = rng.normal(0, 0.002, n)
    close = base + wave + noise
    high = close + rng.uniform(0.001, 0.003, n)
    low = close - rng.uniform(0.001, 0.003, n)
    open_ = close + rng.normal(0, 0.001, n)

    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    results = detect_all(df)
    print(f"Long-term patterns found: {len(results)}")
    by_id = {}
    for r in results:
        by_id.setdefault(r.id, 0)
        by_id[r.id] += 1
    for k, v in by_id.items():
        print(f"  {k}: {v}")
    print("\nLong-term pattern module smoke test passed.")