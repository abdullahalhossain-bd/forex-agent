# analysis/advanced_patterns.py
# ============================================================
# Day 39 — Advanced Chart Pattern Recognition
# AI Chart Intelligence Layer
#
# Patterns:
#   ✅ Head & Shoulders (Bullish/Bearish)
#   ✅ Double Top / Double Bottom
#   ✅ Ascending / Descending / Symmetrical Triangle
#   ✅ Bull Flag / Bear Flag
#   ✅ Rising Wedge / Falling Wedge
#   ✅ Cup & Handle
#
# Features:
#   ✅ Pattern Confidence Score
#   ✅ Target & Invalidation Level
#   ✅ False Pattern Filter
#   ✅ Pattern Combination Engine
#   ✅ Memory Integration (pattern_history)
#   ✅ Analysis Agent integration
# ============================================================

import pandas as pd
import numpy as np
from utils.logger import get_logger

log = get_logger(__name__)

# ── Minimum pattern confidence to report ──────────────────────
MIN_CONFIDENCE = 50

# ── Pattern signal mapping ─────────────────────────────────────
PATTERN_DIRECTION = {
    'HEAD_AND_SHOULDERS':         'BEARISH',
    'INVERSE_HEAD_AND_SHOULDERS': 'BULLISH',
    'DOUBLE_TOP':                 'BEARISH',
    'DOUBLE_BOTTOM':              'BULLISH',
    'ASCENDING_TRIANGLE':         'BULLISH',
    'DESCENDING_TRIANGLE':        'BEARISH',
    'SYMMETRICAL_TRIANGLE':       'NEUTRAL',
    'BULL_FLAG':                  'BULLISH',
    'BEAR_FLAG':                  'BEARISH',
    'RISING_WEDGE':               'BEARISH',
    'FALLING_WEDGE':              'BULLISH',
    'CUP_AND_HANDLE':             'BULLISH',
}


class AdvancedPatternDetector:
    """
    Advanced chart pattern detection engine.

    Takes an OHLCV DataFrame as input,
    returns detected patterns + confidence + targets.

    Usage:
        detector = AdvancedPatternDetector()
        patterns = detector.detect_all(df)
        context  = detector.get_ai_context(df, ind_ctx, sr_ctx, regime_ctx)
    """

    def __init__(self, lookback: int = 100):
        """
        lookback : how many candles to look at when searching for patterns
        """
        self.lookback = lookback

    # ═══════════════════════════════════════════════════════════
    # MAIN DETECTION METHOD
    # ═══════════════════════════════════════════════════════════

    def detect_all(self, df: pd.DataFrame) -> list[dict]:
        """
        Detect all patterns at once.
        Returns: list of pattern dicts, sorted by confidence.
        """
        if len(df) < 30:
            return []

        data = df.tail(self.lookback).copy()
        patterns = []

        detectors = [
            self.detect_head_and_shoulders,
            self.detect_double_top_bottom,
            self.detect_triple_top_bottom,  # Book Spread 6-7
            self.detect_triangle,
            self.detect_flag,
            self.detect_wedge,
            self.detect_cup_and_handle,
            self.detect_rectangle,        # Day 100+ (Page 112-113)
            self.detect_momentum_screen,  # Day 100+ (Page 120)
        ]

        for detect_fn in detectors:
            try:
                result = detect_fn(data)
                if result:
                    if isinstance(result, list):
                        patterns.extend(result)
                    else:
                        patterns.append(result)
            except Exception as e:
                log.warning(f"Pattern detection error in {detect_fn.__name__}: {e}")

        # Filter low confidence
        patterns = [p for p in patterns if p.get('confidence', 0) >= MIN_CONFIDENCE]

        # Sort by confidence descending
        patterns.sort(key=lambda p: p.get('confidence', 0), reverse=True)

        log.info(f"Advanced patterns detected: {len(patterns)}")
        return patterns

    # ═══════════════════════════════════════════════════════════
    # 1. HEAD & SHOULDERS
    # ═══════════════════════════════════════════════════════════

    def detect_head_and_shoulders(self, df: pd.DataFrame) -> list[dict]:
        """
        Head & Shoulders (Bearish) and Inverse H&S (Bullish)

        Structure:
          Left Shoulder → Head (higher) → Right Shoulder (≈ Left)
          Neckline = line connecting lows between peaks

        Inverse: opposite — detect bullish reversal at bottoms
        """
        results = []
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        n      = len(df)

        if n < 40:
            return results

        # Find swing highs (for H&S)
        swing_highs = self._find_swings(highs, mode='high', window=5)
        swing_lows  = self._find_swings(lows,  mode='low',  window=5)

        # Need at least 3 swing highs for H&S
        if len(swing_highs) >= 3:
            for i in range(len(swing_highs) - 2):
                ls_idx, ls_val = swing_highs[i]
                h_idx,  h_val  = swing_highs[i + 1]
                rs_idx, rs_val = swing_highs[i + 2]

                # Head must be highest
                if not (h_val > ls_val and h_val > rs_val):
                    continue

                # Shoulders should be roughly equal (within 5%)
                shoulder_diff = abs(ls_val - rs_val) / h_val
                if shoulder_diff > 0.05:
                    continue

                # Neckline = average of lows between shoulders
                between_lows = [v for idx, v in swing_lows
                                if ls_idx < idx < rs_idx]
                if not between_lows:
                    continue
                neckline = np.mean(between_lows)

                # Pattern height
                height     = h_val - neckline
                target     = round(neckline - height, 5)
                inval      = round(h_val + height * 0.1, 5)
                curr_price = closes[-1]

                # Confidence components
                symmetry_score = max(0, 100 - shoulder_diff * 1000)
                size_score     = min(100, height / curr_price * 10000)
                recency_score  = max(0, 100 - (n - rs_idx) * 2)

                confidence = int(
                    symmetry_score * 0.40 +
                    size_score     * 0.30 +
                    recency_score  * 0.30
                )

                if confidence >= MIN_CONFIDENCE:
                    results.append({
                        'pattern':      'HEAD_AND_SHOULDERS',
                        'direction':    'BEARISH',
                        'confidence':   confidence,
                        'neckline':     round(neckline, 5),
                        'head':         round(h_val, 5),
                        'left_shoulder': round(ls_val, 5),
                        'right_shoulder': round(rs_val, 5),
                        'entry':        round(neckline, 5),
                        'target':       target,
                        'invalidation': inval,
                        'note':         (
                            f"H&S: Head={h_val:.5f}, Neckline={neckline:.5f}. "
                            f"SELL on neckline break. Target={target:.5f}"
                        ),
                    })

        # Inverse H&S (swing lows → bullish reversal)
        if len(swing_lows) >= 3:
            for i in range(len(swing_lows) - 2):
                ls_idx, ls_val = swing_lows[i]
                h_idx,  h_val  = swing_lows[i + 1]
                rs_idx, rs_val = swing_lows[i + 2]

                if not (h_val < ls_val and h_val < rs_val):
                    continue

                shoulder_diff = abs(ls_val - rs_val) / max(abs(h_val), 1e-10)
                if shoulder_diff > 0.05:
                    continue

                between_highs = [v for idx, v in swing_highs
                                 if ls_idx < idx < rs_idx]
                if not between_highs:
                    continue
                neckline = np.mean(between_highs)

                height     = neckline - h_val
                target     = round(neckline + height, 5)
                inval      = round(h_val - height * 0.1, 5)
                curr_price = closes[-1]

                symmetry_score = max(0, 100 - shoulder_diff * 1000)
                size_score     = min(100, height / max(curr_price, 1e-5) * 10000)
                recency_score  = max(0, 100 - (n - rs_idx) * 2)

                confidence = int(
                    symmetry_score * 0.40 +
                    size_score     * 0.30 +
                    recency_score  * 0.30
                )

                if confidence >= MIN_CONFIDENCE:
                    results.append({
                        'pattern':      'INVERSE_HEAD_AND_SHOULDERS',
                        'direction':    'BULLISH',
                        'confidence':   confidence,
                        'neckline':     round(neckline, 5),
                        'head':         round(h_val, 5),
                        'entry':        round(neckline, 5),
                        'target':       target,
                        'invalidation': inval,
                        'note':         (
                            f"Inverse H&S: Head={h_val:.5f}, Neckline={neckline:.5f}. "
                            f"BUY on neckline break. Target={target:.5f}"
                        ),
                    })

        return results

    # ═══════════════════════════════════════════════════════════
    # 2. DOUBLE TOP / DOUBLE BOTTOM
    # ═══════════════════════════════════════════════════════════

    def detect_double_top_bottom(self, df: pd.DataFrame) -> list[dict]:
        """
        Double Top   : two roughly equal highs → Bearish reversal
        Double Bottom: two roughly equal lows  → Bullish reversal

        Target = Breakout ± Pattern Height
        """
        results = []
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        n      = len(df)

        if n < 20:
            return results

        swing_highs = self._find_swings(highs, mode='high', window=5)
        swing_lows  = self._find_swings(lows,  mode='low',  window=5)

        # ── Double Top ──────────────────────────────────────────
        for i in range(len(swing_highs) - 1):
            idx1, val1 = swing_highs[i]
            idx2, val2 = swing_highs[i + 1]

            # Peaks roughly equal (within 2%)
            diff = abs(val1 - val2) / max(val1, 1e-10)
            if diff > 0.02:
                continue

            # Minimum separation
            if idx2 - idx1 < 5:
                continue

            # Neckline = low between the two peaks
            between_lows = lows[idx1:idx2]
            if len(between_lows) == 0:
                continue
            neckline = min(between_lows)

            height     = ((val1 + val2) / 2) - neckline
            target     = round(neckline - height, 5)
            inval      = round(max(val1, val2) * 1.01, 5)
            curr_price = closes[-1]

            equality_score = max(0, 100 - diff * 2000)
            size_score     = min(100, height / max(curr_price, 1e-5) * 10000)
            recency_score  = max(0, 100 - (n - idx2) * 3)
            curr_near_neck = 100 if abs(curr_price - neckline) / max(curr_price, 1e-5) < 0.005 else 60

            confidence = int(
                equality_score  * 0.35 +
                size_score      * 0.25 +
                recency_score   * 0.25 +
                curr_near_neck  * 0.15
            )

            if confidence >= MIN_CONFIDENCE:
                results.append({
                    'pattern':      'DOUBLE_TOP',
                    'direction':    'BEARISH',
                    'confidence':   confidence,
                    'peak1':        round(val1, 5),
                    'peak2':        round(val2, 5),
                    'neckline':     round(neckline, 5),
                    'entry':        round(neckline, 5),
                    'target':       target,
                    'invalidation': inval,
                    'note':         (
                        f"Double Top at {val1:.5f}/{val2:.5f}. "
                        f"Neckline={neckline:.5f}. SELL on break. Target={target:.5f}"
                    ),
                })

        # ── Double Bottom ───────────────────────────────────────
        for i in range(len(swing_lows) - 1):
            idx1, val1 = swing_lows[i]
            idx2, val2 = swing_lows[i + 1]

            diff = abs(val1 - val2) / max(abs(val1), 1e-10)
            if diff > 0.02:
                continue

            if idx2 - idx1 < 5:
                continue

            between_highs = highs[idx1:idx2]
            if len(between_highs) == 0:
                continue
            neckline = max(between_highs)

            height     = neckline - ((val1 + val2) / 2)
            target     = round(neckline + height, 5)
            inval      = round(min(val1, val2) * 0.99, 5)
            curr_price = closes[-1]

            equality_score = max(0, 100 - diff * 2000)
            size_score     = min(100, height / max(curr_price, 1e-5) * 10000)
            recency_score  = max(0, 100 - (n - idx2) * 3)
            curr_near_neck = 100 if abs(curr_price - neckline) / max(curr_price, 1e-5) < 0.005 else 60

            confidence = int(
                equality_score  * 0.35 +
                size_score      * 0.25 +
                recency_score   * 0.25 +
                curr_near_neck  * 0.15
            )

            if confidence >= MIN_CONFIDENCE:
                results.append({
                    'pattern':      'DOUBLE_BOTTOM',
                    'direction':    'BULLISH',
                    'confidence':   confidence,
                    'trough1':      round(val1, 5),
                    'trough2':      round(val2, 5),
                    'neckline':     round(neckline, 5),
                    'entry':        round(neckline, 5),
                    'target':       target,
                    'invalidation': inval,
                    'note':         (
                        f"Double Bottom at {val1:.5f}/{val2:.5f}. "
                        f"Neckline={neckline:.5f}. BUY on break. Target={target:.5f}"
                    ),
                })

        return results

    # ═══════════════════════════════════════════════════════════
    # 2b. TRIPLE TOP / TRIPLE BOTTOM (Book Spreads 6-7)
    # ═══════════════════════════════════════════════════════════

    def detect_triple_top_bottom(self, df: pd.DataFrame) -> list[dict]:
        """
        Book Spread 6-7 — Triple Top / Triple Bottom
        =============================================
        Triple Top   : 3 roughly equal highs → Bearish reversal
        Triple Bottom: 3 roughly equal lows  → Bullish reversal

        Stronger signal than Double Top/Bottom (3 touches vs 2).
        Neckline = shared support (Triple Top) or resistance (Triple Bottom)
        connecting the troughs/peaks between the 3 extremes.
        Confirmed on close below/above neckline.

        Pseudocode (from book):
          IF three swing highs within tolerance AND shared support not broken:
              Pattern = "Triple Top"
              IF price closes_below neckline: Signal = "Bearish"
          IF three swing lows within tolerance AND shared resistance not broken:
              Pattern = "Triple Bottom"
              IF price closes_above neckline: Signal = "Bullish"
        """
        results = []
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        n      = len(df)

        if n < 30:
            return results

        swing_highs = self._find_swings(highs, mode='high', window=5)
        swing_lows  = self._find_swings(lows,  mode='low',  window=5)
        atr = self._atr_simple(df)

        # ── Triple Top (3 peaks) ───────────────────────────────
        for i in range(len(swing_highs) - 2):
            idx1, val1 = swing_highs[i]
            idx2, val2 = swing_highs[i + 1]
            idx3, val3 = swing_highs[i + 2]

            # All 3 peaks roughly equal (within 1.5% tolerance — tighter than double)
            d12 = abs(val1 - val2) / max(val1, val2, 1e-10)
            d23 = abs(val2 - val3) / max(val2, val3, 1e-10)
            d13 = abs(val1 - val3) / max(val1, val3, 1e-10)
            if d12 > 0.015 or d23 > 0.015 or d13 > 0.015:
                continue

            # Find troughs between peaks
            trough1 = float(min(lows[idx1:idx2])) if idx2 > idx1 else float(lows[idx1])
            trough2 = float(min(lows[idx2:idx3])) if idx3 > idx2 else float(lows[idx2])
            neckline = min(trough1, trough2)

            # Neckline must be meaningfully below peaks
            pattern_height = val1 - neckline
            if pattern_height < atr * 0.5:
                continue

            curr_price = float(closes[-1])
            broke_down = curr_price < neckline - atr * 0.1

            if broke_down:
                target = round(neckline - pattern_height, 5)
                inval  = round(val1 + atr * 0.3, 5)
                results.append({
                    'pattern':       'TRIPLE_TOP',
                    'direction':     'BEARISH',
                    'trade_action':  'SHORT',
                    'confidence':    78,
                    'peak1':         round(val1, 5),
                    'peak2':         round(val2, 5),
                    'peak3':         round(val3, 5),
                    'neckline':      round(neckline, 5),
                    'entry':         round(neckline, 5),
                    'target':        target,
                    'invalidation':  inval,
                    'note':          (
                        f"Triple Top: 3 peaks at {val1:.5f}/{val2:.5f}/{val3:.5f}, "
                        f"neckline={neckline:.5f}. Confirmed bearish on close below neckline. "
                        f"Target={target:.5f}"
                    ),
                })
            else:
                # Still forming — flag as NEUTRAL/NO_TRADE
                results.append({
                    'pattern':       'TRIPLE_TOP_FORMING',
                    'direction':     'NEUTRAL',
                    'trade_action':  'NO_TRADE',
                    'confidence':    60,
                    'peak1':         round(val1, 5),
                    'peak2':         round(val2, 5),
                    'peak3':         round(val3, 5),
                    'neckline':      round(neckline, 5),
                    'entry':         None,
                    'target':        None,
                    'invalidation':  None,
                    'note':          (
                        f"Triple Top forming (3 peaks at {val1:.5f}/{val2:.5f}/{val3:.5f}). "
                        f"Wait for neckline break at {neckline:.5f}. NO_TRADE until confirmed."
                    ),
                })

        # ── Triple Bottom (3 troughs) ─────────────────────────
        for i in range(len(swing_lows) - 2):
            idx1, val1 = swing_lows[i]
            idx2, val2 = swing_lows[i + 1]
            idx3, val3 = swing_lows[i + 2]

            # All 3 troughs roughly equal
            d12 = abs(val1 - val2) / max(val1, val2, 1e-10)
            d23 = abs(val2 - val3) / max(val2, val3, 1e-10)
            d13 = abs(val1 - val3) / max(val1, val3, 1e-10)
            if d12 > 0.015 or d23 > 0.015 or d13 > 0.015:
                continue

            # Find peaks between troughs
            peak1 = float(max(highs[idx1:idx2])) if idx2 > idx1 else float(highs[idx1])
            peak2 = float(max(highs[idx2:idx3])) if idx3 > idx2 else float(highs[idx2])
            neckline = max(peak1, peak2)

            pattern_height = neckline - val1
            if pattern_height < atr * 0.5:
                continue

            curr_price = float(closes[-1])
            broke_up = curr_price > neckline + atr * 0.1

            if broke_up:
                target = round(neckline + pattern_height, 5)
                inval  = round(val1 - atr * 0.3, 5)
                results.append({
                    'pattern':       'TRIPLE_BOTTOM',
                    'direction':     'BULLISH',
                    'trade_action':  'LONG',
                    'confidence':    78,
                    'trough1':       round(val1, 5),
                    'trough2':       round(val2, 5),
                    'trough3':       round(val3, 5),
                    'neckline':      round(neckline, 5),
                    'entry':         round(neckline, 5),
                    'target':        target,
                    'invalidation':  inval,
                    'note':          (
                        f"Triple Bottom: 3 troughs at {val1:.5f}/{val2:.5f}/{val3:.5f}, "
                        f"neckline={neckline:.5f}. Confirmed bullish on close above neckline. "
                        f"Target={target:.5f}"
                    ),
                })
            else:
                results.append({
                    'pattern':       'TRIPLE_BOTTOM_FORMING',
                    'direction':     'NEUTRAL',
                    'trade_action':  'NO_TRADE',
                    'confidence':    60,
                    'trough1':       round(val1, 5),
                    'trough2':       round(val2, 5),
                    'trough3':       round(val3, 5),
                    'neckline':      round(neckline, 5),
                    'entry':         None,
                    'target':        None,
                    'invalidation':  None,
                    'note':          (
                        f"Triple Bottom forming (3 troughs at {val1:.5f}/{val2:.5f}/{val3:.5f}). "
                        f"Wait for neckline break at {neckline:.5f}. NO_TRADE until confirmed."
                    ),
                })

        return results

    # ═══════════════════════════════════════════════════════════
    # 3. TRIANGLE PATTERNS
    # ═══════════════════════════════════════════════════════════

    def detect_triangle(self, df: pd.DataFrame) -> list[dict]:
        """
        Ascending  Triangle : Flat resistance + Higher lows → Bullish
        Descending Triangle : Flat support + Lower highs → Bearish
        Symmetrical Triangle: Lower highs + Higher lows → Wait for breakout
        """
        results = []
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        n      = len(df)

        if n < 30:
            return results

        # Use recent 40-60 candles
        window = min(n, 60)
        h_seg  = highs[-window:]
        l_seg  = lows[-window:]
        c_seg  = closes[-window:]
        w      = len(h_seg)

        # Fit linear trend to highs and lows
        x        = np.arange(w)
        h_slope, h_intercept = np.polyfit(x, h_seg, 1)
        l_slope, l_intercept = np.polyfit(x, l_seg, 1)

        curr_price = closes[-1]
        atr        = self._atr_simple(df)

        # Resistance and support at last candle
        resistance = h_intercept + h_slope * (w - 1)
        support    = l_intercept + l_slope * (w - 1)

        # Convergence — lines getting closer
        range_start = (h_intercept + h_slope * 0) - (l_intercept + l_slope * 0)
        range_end   = resistance - support
        converging  = range_end < range_start * 0.85 and range_end > 0

        if not converging:
            return results

        height = range_start  # initial range = pattern height

        # ── Ascending Triangle ──────────────────────────────────
        # Resistance flat, lows rising
        if abs(h_slope) < atr * 0.02 and l_slope > atr * 0.005:
            target  = round(resistance + height, 5)
            inval   = round(support - atr, 5)
            conf_base = 75

            # Bonus: price near resistance
            near_res = abs(curr_price - resistance) < atr * 0.5
            confidence = conf_base + (10 if near_res else 0)

            results.append({
                'pattern':      'ASCENDING_TRIANGLE',
                'direction':    'BULLISH',
                'confidence':   min(95, confidence),
                'resistance':   round(resistance, 5),
                'support':      round(support, 5),
                'entry':        round(resistance, 5),
                'target':       target,
                'invalidation': inval,
                'note':         (
                    f"Ascending Triangle: Flat resistance={resistance:.5f}, "
                    f"rising lows. BUY on breakout. Target={target:.5f}"
                ),
            })

        # ── Descending Triangle ─────────────────────────────────
        # Support flat, highs falling
        elif abs(l_slope) < atr * 0.02 and h_slope < -atr * 0.005:
            target  = round(support - height, 5)
            inval   = round(resistance + atr, 5)
            conf_base = 73

            near_sup = abs(curr_price - support) < atr * 0.5
            confidence = conf_base + (10 if near_sup else 0)

            results.append({
                'pattern':      'DESCENDING_TRIANGLE',
                'direction':    'BEARISH',
                'confidence':   min(95, confidence),
                'resistance':   round(resistance, 5),
                'support':      round(support, 5),
                'entry':        round(support, 5),
                'target':       target,
                'invalidation': inval,
                'note':         (
                    f"Descending Triangle: Flat support={support:.5f}, "
                    f"falling highs. SELL on breakdown. Target={target:.5f}"
                ),
            })

        # ── Symmetrical Triangle ────────────────────────────────
        elif h_slope < -atr * 0.003 and l_slope > atr * 0.003:
            midpoint   = (resistance + support) / 2
            target_up  = round(midpoint + height / 2, 5)
            target_dn  = round(midpoint - height / 2, 5)

            results.append({
                'pattern':      'SYMMETRICAL_TRIANGLE',
                'direction':    'NEUTRAL',
                'confidence':   68,
                'resistance':   round(resistance, 5),
                'support':      round(support, 5),
                'entry':        None,
                'target_bull':  target_up,
                'target_bear':  target_dn,
                'invalidation': None,
                'note':         (
                    f"Symmetrical Triangle: Converging lines. "
                    f"Wait for breakout above {resistance:.5f} (target {target_up:.5f}) "
                    f"or below {support:.5f} (target {target_dn:.5f})"
                ),
            })

        return results

    # ═══════════════════════════════════════════════════════════
    # 4. FLAG PATTERNS
    # ═══════════════════════════════════════════════════════════

    def detect_flag(self, df: pd.DataFrame) -> list[dict]:
        """
        Bull Flag: Strong impulse up → small downward channel → BUY continuation
        Bear Flag: Strong impulse down → small upward channel → SELL continuation

        Pole = impulse move
        Flag = consolidation channel
        """
        results = []
        closes = df['close'].values
        highs  = df['high'].values
        lows   = df['low'].values
        n      = len(df)

        if n < 25:
            return results

        atr    = self._atr_simple(df)
        window = min(n, 60)

        # Split into pole (first 60%) and flag (last 40%)
        pole_len = int(window * 0.60)
        flag_len = window - pole_len

        pole  = closes[-window:-flag_len]
        flag  = closes[-flag_len:]
        f_h   = highs[-flag_len:]
        f_l   = lows[-flag_len:]

        if len(pole) < 5 or len(flag) < 5:
            return results

        pole_move  = pole[-1] - pole[0]
        flag_range = max(f_h) - min(f_l)

        # Flag should be smaller than pole (30-70% of pole)
        ratio_ok = 0.1 < abs(flag_range / pole_move) < 0.7 if pole_move != 0 else False

        if not ratio_ok:
            return results

        # ── Bull Flag ───────────────────────────────────────────
        if pole_move > atr * 2:  # Strong upward impulse
            x_flag   = np.arange(flag_len)
            f_slope, _ = np.polyfit(x_flag, flag, 1)

            # Flag should slope slightly downward (correction)
            if f_slope < 0:
                breakout = max(f_h)
                target   = round(breakout + abs(pole_move), 5)
                inval    = round(min(f_l) - atr * 0.5, 5)

                confidence = 72
                if abs(pole_move) > atr * 3:   confidence += 10
                if flag_range < abs(pole_move) * 0.4: confidence += 8

                results.append({
                    'pattern':      'BULL_FLAG',
                    'direction':    'BULLISH',
                    'confidence':   min(95, confidence),
                    'pole_move':    round(pole_move, 5),
                    'flag_range':   round(flag_range, 5),
                    'breakout':     round(breakout, 5),
                    'entry':        round(breakout, 5),
                    'target':       target,
                    'invalidation': inval,
                    'note':         (
                        f"Bull Flag: Pole={pole_move:.5f}, "
                        f"Flag consolidation. BUY on breakout above {breakout:.5f}. "
                        f"Target={target:.5f}"
                    ),
                })

        # ── Bear Flag ───────────────────────────────────────────
        elif pole_move < -atr * 2:  # Strong downward impulse
            x_flag    = np.arange(flag_len)
            f_slope, _ = np.polyfit(x_flag, flag, 1)

            # Flag should slope slightly upward (correction)
            if f_slope > 0:
                breakdown = min(f_l)
                target    = round(breakdown + pole_move, 5)  # pole_move is negative
                inval     = round(max(f_h) + atr * 0.5, 5)

                confidence = 70
                if abs(pole_move) > atr * 3:   confidence += 10
                if flag_range < abs(pole_move) * 0.4: confidence += 8

                results.append({
                    'pattern':      'BEAR_FLAG',
                    'direction':    'BEARISH',
                    'confidence':   min(95, confidence),
                    'pole_move':    round(pole_move, 5),
                    'flag_range':   round(flag_range, 5),
                    'breakdown':    round(breakdown, 5),
                    'entry':        round(breakdown, 5),
                    'target':       target,
                    'invalidation': inval,
                    'note':         (
                        f"Bear Flag: Pole={pole_move:.5f}, "
                        f"Flag consolidation. SELL on breakdown below {breakdown:.5f}. "
                        f"Target={target:.5f}"
                    ),
                })

        return results

    # ═══════════════════════════════════════════════════════════
    # 5. WEDGE PATTERNS
    # ═══════════════════════════════════════════════════════════

    def detect_wedge(self, df: pd.DataFrame) -> list[dict]:
        """
        Rising Wedge  : Both lines slope up, converging → Bearish reversal
        Falling Wedge : Both lines slope down, converging → Bullish reversal
        """
        results = []
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        n      = len(df)

        if n < 25:
            return results

        window = min(n, 60)
        h_seg  = highs[-window:]
        l_seg  = lows[-window:]
        w      = len(h_seg)
        x      = np.arange(w)

        h_slope, h_int = np.polyfit(x, h_seg, 1)
        l_slope, l_int = np.polyfit(x, l_seg, 1)

        atr   = self._atr_simple(df)
        curr  = closes[-1]

        # Convergence check — both lines moving together but getting closer
        converging = abs(h_slope - l_slope) < abs(h_slope) * 0.7

        if not converging:
            return results

        range_now = (h_int + h_slope * (w-1)) - (l_int + l_slope * (w-1))
        height    = max(h_seg) - min(l_seg)

        # ── Rising Wedge (bearish) ──────────────────────────────
        if h_slope > atr * 0.003 and l_slope > atr * 0.003 and l_slope > h_slope * 0.8:
            # Both slopes positive but converging
            support    = l_int + l_slope * (w - 1)
            target     = round(support - height * 0.6, 5)
            inval      = round(max(h_seg) * 1.005, 5)

            confidence = 70
            if l_slope > h_slope:  confidence += 8   # lows rising faster = more wedge-like
            if range_now < height * 0.5: confidence += 7

            results.append({
                'pattern':      'RISING_WEDGE',
                'direction':    'BEARISH',
                'confidence':   min(90, confidence),
                'upper_slope':  round(h_slope, 6),
                'lower_slope':  round(l_slope, 6),
                'support':      round(support, 5),
                'entry':        round(support, 5),
                'target':       target,
                'invalidation': inval,
                'note':         (
                    f"Rising Wedge: Both lines ascending, converging. "
                    f"SELL on support break {support:.5f}. Target={target:.5f}"
                ),
            })

        # ── Falling Wedge (bullish) ─────────────────────────────
        elif h_slope < -atr * 0.003 and l_slope < -atr * 0.003 and h_slope < l_slope * 0.8:
            resistance = h_int + h_slope * (w - 1)
            target     = round(resistance + height * 0.6, 5)
            inval      = round(min(l_seg) * 0.995, 5)

            confidence = 72
            if h_slope < l_slope:  confidence += 8
            if range_now < height * 0.5: confidence += 7

            results.append({
                'pattern':      'FALLING_WEDGE',
                'direction':    'BULLISH',
                'confidence':   min(90, confidence),
                'upper_slope':  round(h_slope, 6),
                'lower_slope':  round(l_slope, 6),
                'resistance':   round(resistance, 5),
                'entry':        round(resistance, 5),
                'target':       target,
                'invalidation': inval,
                'note':         (
                    f"Falling Wedge: Both lines descending, converging. "
                    f"BUY on resistance break {resistance:.5f}. Target={target:.5f}"
                ),
            })

        return results

    # ═══════════════════════════════════════════════════════════
    # 6. CUP & HANDLE
    # ═══════════════════════════════════════════════════════════

    def detect_cup_and_handle(self, df: pd.DataFrame) -> list[dict]:
        """
        Cup & Handle: Long-term bullish continuation
        Cup = U-shaped rounded bottom
        Handle = small downward correction after cup
        Breakout = BUY signal
        """
        results = []
        closes = df['close'].values
        highs  = df['high'].values
        n      = len(df)

        if n < 50:
            return results

        # Cup uses first 70% of data, handle uses last 30%
        cup_len    = int(n * 0.70)
        handle_len = n - cup_len

        cup    = closes[:cup_len]
        handle = closes[cup_len:]
        h_high = highs[cup_len:]

        if len(handle) < 8:
            return results

        cup_left  = cup[0]
        cup_right = cup[-1]
        cup_low   = min(cup)
        cup_depth = ((cup_left + cup_right) / 2) - cup_low

        # Cup lips should be roughly equal
        lip_diff  = abs(cup_left - cup_right) / max(cup_left, 1e-5)
        if lip_diff > 0.03:
            return results

        # Cup should be rounded (not V-shaped)
        # Check that middle 40% is within 20% of the bottom
        mid_start = int(cup_len * 0.30)
        mid_end   = int(cup_len * 0.70)
        mid_range = max(cup[mid_start:mid_end]) - cup_low
        roundness = mid_range / max(cup_depth, 1e-10)
        if roundness > 0.4:   # too V-shaped
            return results

        # Handle: small pullback (10-30% of cup depth)
        handle_drop = max(h_high) - min(handle)
        handle_ok   = 0.05 < handle_drop / max(cup_depth, 1e-10) < 0.50

        if not handle_ok:
            return results

        breakout   = max(cup_left, cup_right)
        target     = round(breakout + cup_depth, 5)
        inval      = round(min(handle) - cup_depth * 0.1, 5)
        curr_price = closes[-1]

        # Confidence
        roundness_score  = max(0, 100 - roundness * 200)
        symmetry_score   = max(0, 100 - lip_diff * 2000)
        handle_score     = 80
        curr_near_break  = 100 if abs(curr_price - breakout) / max(curr_price, 1e-5) < 0.01 else 60

        confidence = int(
            roundness_score * 0.35 +
            symmetry_score  * 0.25 +
            handle_score    * 0.25 +
            curr_near_break * 0.15
        )

        if confidence >= MIN_CONFIDENCE:
            results.append({
                'pattern':      'CUP_AND_HANDLE',
                'direction':    'BULLISH',
                'confidence':   min(92, confidence),
                'cup_low':      round(cup_low, 5),
                'cup_depth':    round(cup_depth, 5),
                'breakout':     round(breakout, 5),
                'entry':        round(breakout, 5),
                'target':       target,
                'invalidation': inval,
                'note':         (
                    f"Cup & Handle: Cup low={cup_low:.5f}, depth={cup_depth:.5f}. "
                    f"BUY on breakout above {breakout:.5f}. Target={target:.5f}"
                ),
            })

        return results

    # ═══════════════════════════════════════════════════════════
    # CONFIDENCE BOOSTER (Combination Engine)
    # ═══════════════════════════════════════════════════════════

    def boost_confidence(
        self,
        patterns:   list[dict],
        ind_ctx:    dict = None,
        sr_ctx:     dict = None,
        regime_ctx: dict = None,
        pat_ctx:    dict = None,
    ) -> list[dict]:
        """
        Adjust confidence based on pattern confidence + indicator
        alignment + S/R + regime.

        Boosts:
          +10 : RSI confirms direction
          +10 : MACD confirms direction
          +8  : Price at S/R zone
          +7  : Trending regime
          +8  : Candlestick pattern confirms direction
          -15 : Counter-trend (regime opposes pattern)
          -10 : Overbought/oversold against direction
        """
        if not patterns:
            return patterns

        boosted = []
        for pat in patterns:
            direction = pat.get('direction', 'NEUTRAL')
            conf      = pat.get('confidence', 50)
            adjustment = 0
            reasons    = []

            if ind_ctx:
                rsi        = ind_ctx.get('rsi', 50)
                rsi_signal = ind_ctx.get('rsi_signal', 'neutral')
                macd_cross = ind_ctx.get('macd_cross', '')
                trend      = ind_ctx.get('trend', '')

                # RSI confirmation
                if direction == 'BULLISH' and rsi_signal in ('oversold', 'bullish_zone'):
                    adjustment += 10
                    reasons.append('+10 RSI bullish')
                elif direction == 'BEARISH' and rsi_signal in ('overbought', 'bearish_zone'):
                    adjustment += 10
                    reasons.append('+10 RSI bearish')
                elif direction == 'BULLISH' and rsi_signal == 'overbought':
                    adjustment -= 10
                    reasons.append('-10 RSI overbought against BUY')
                elif direction == 'BEARISH' and rsi_signal == 'oversold':
                    adjustment -= 10
                    reasons.append('-10 RSI oversold against SELL')

                # MACD confirmation
                if direction == 'BULLISH' and 'bullish_cross' in macd_cross:
                    adjustment += 10
                    reasons.append('+10 MACD bullish cross')
                elif direction == 'BEARISH' and 'bearish_cross' in macd_cross:
                    adjustment += 10
                    reasons.append('+10 MACD bearish cross')

                # Trend alignment
                if direction == 'BULLISH' and 'bullish' in trend:
                    adjustment += 5
                    reasons.append('+5 trend aligned')
                elif direction == 'BEARISH' and 'bearish' in trend:
                    adjustment += 5
                    reasons.append('+5 trend aligned')
                elif direction == 'BULLISH' and 'bearish' in trend:
                    adjustment -= 15
                    reasons.append('-15 counter-trend')
                elif direction == 'BEARISH' and 'bullish' in trend:
                    adjustment -= 15
                    reasons.append('-15 counter-trend')

            if sr_ctx:
                location = sr_ctx.get('price_location', 'mid_range')
                if direction == 'BULLISH' and location == 'near_support':
                    adjustment += 8
                    reasons.append('+8 near support')
                elif direction == 'BEARISH' and location == 'near_resistance':
                    adjustment += 8
                    reasons.append('+8 near resistance')

            if regime_ctx:
                regime = regime_ctx.get('market_regime', '')
                if regime == 'TRENDING':
                    adjustment += 7
                    reasons.append('+7 trending regime')
                elif regime == 'RANGING' and pat.get('pattern') not in (
                    'DOUBLE_TOP', 'DOUBLE_BOTTOM', 'ASCENDING_TRIANGLE', 'DESCENDING_TRIANGLE'
                ):
                    adjustment -= 8
                    reasons.append('-8 ranging regime (less reliable for this pattern)')

            if pat_ctx:
                candle_signal = pat_ctx.get('pattern_signal', '')
                if direction == 'BULLISH' and 'Bullish' in candle_signal:
                    adjustment += 8
                    reasons.append('+8 bullish candlestick confirms')
                elif direction == 'BEARISH' and 'Bearish' in candle_signal:
                    adjustment += 8
                    reasons.append('+8 bearish candlestick confirms')

            new_conf = max(10, min(98, conf + adjustment))
            pat = dict(pat)
            pat['confidence']        = new_conf
            pat['confidence_raw']    = conf
            pat['confidence_boost']  = adjustment
            pat['boost_reasons']     = reasons
            boosted.append(pat)

        boosted.sort(key=lambda p: p.get('confidence', 0), reverse=True)
        return boosted

    # ═══════════════════════════════════════════════════════════
    # FALSE PATTERN FILTER
    # ═══════════════════════════════════════════════════════════

    def filter_false_patterns(
        self,
        patterns:   list[dict],
        regime_ctx: dict = None,
        ind_ctx:    dict = None,
    ) -> list[dict]:
        """
        Detect and remove low-quality patterns.

        Rules:
          - Skip trending patterns in a RANGING regime
          - Confidence < 55 after boost → skip
          - Counter-trend high-volatility → skip
        """
        if not patterns:
            return patterns

        filtered = []
        trend_patterns = {
            'HEAD_AND_SHOULDERS', 'INVERSE_HEAD_AND_SHOULDERS',
            'BULL_FLAG', 'BEAR_FLAG', 'RISING_WEDGE', 'FALLING_WEDGE'
        }

        for pat in patterns:
            reason = None
            pattern_name = pat.get('pattern', '')
            conf         = pat.get('confidence', 0)
            direction    = pat.get('direction', 'NEUTRAL')

            # Low confidence → false signal
            if conf < 55:
                reason = f"Confidence {conf}% too low"

            # Trending patterns in ranging market
            elif (regime_ctx and
                  regime_ctx.get('market_regime') == 'RANGING' and
                  pattern_name in trend_patterns):
                reason = f"{pattern_name} unreliable in RANGING market"

            # Counter-trend in strong trend
            elif ind_ctx:
                trend = ind_ctx.get('trend', '')
                if (direction == 'BULLISH' and 'strong_bearish' in trend):
                    reason = "Bullish pattern against strong bearish trend"
                elif (direction == 'BEARISH' and 'strong_bullish' in trend):
                    reason = "Bearish pattern against strong bullish trend"

            if reason:
                log.info(f"False pattern filtered: {pattern_name} — {reason}")
                pat = dict(pat)
                pat['filtered']       = True
                pat['filter_reason']  = reason
            else:
                pat = dict(pat)
                pat['filtered']       = False
                pat['filter_reason']  = None
                filtered.append(pat)

        return filtered

    # ═══════════════════════════════════════════════════════════
    # AI CONTEXT — Analysis Agent Integration
    # ═══════════════════════════════════════════════════════════

    def get_ai_context(
        self,
        df:         pd.DataFrame,
        ind_ctx:    dict = None,
        sr_ctx:     dict = None,
        regime_ctx: dict = None,
        pat_ctx:    dict = None,
        patterns:   list = None,  # Round-5 audit fix
    ) -> dict:
        """
        Full pipeline:
        detect → boost → filter → return AI-ready context

        Compatible with AnalysisAgent and DecisionAgent.

        Round-5 audit fix: the optional `patterns` parameter lets callers
        REUSE an already-computed pattern list instead of re-running
        detect_all() + boost_confidence() + filter_false_patterns() on
        the same df. This eliminates the duplicate computation observed
        in the operator's logs (two identical "Advanced patterns
        detected: N" lines per cycle).

        Callers that still call get_ai_context() without pre-computed
        patterns get the old behavior (full pipeline runs internally)
        — backward compatible.
        """
        if patterns is None:
            # Legacy path: caller didn't pre-compute. Run the full
            # pipeline here (this is the duplicate path the audit
            # flagged when AnalysisAgent already called detect_all()
            # + boost + filter separately just before this).
            patterns = self.detect_all(df)
            patterns = self.boost_confidence(
                patterns, ind_ctx, sr_ctx, regime_ctx, pat_ctx
            )
            patterns = self.filter_false_patterns(patterns, regime_ctx, ind_ctx)
        # else: caller passed in already-computed, boosted, filtered
        # patterns — use them directly, skip the duplicate work.

        # Best pattern
        best = patterns[0] if patterns else None

        # Combine with candlestick patterns for signal
        combined_signal = self._combined_signal(best, pat_ctx)

        return {
            # Primary output
            'advanced_pattern':    best.get('pattern') if best else 'NONE',
            'pattern_direction':   best.get('direction') if best else 'NEUTRAL',
            'pattern_confidence':  best.get('confidence') if best else 0,
            'pattern_target':      best.get('target') if best else None,
            'pattern_inval':       best.get('invalidation') if best else None,
            'pattern_note':        best.get('note') if best else 'No advanced pattern',

            # All detected
            'all_patterns':        [p.get('pattern') for p in patterns],
            'pattern_count':       len(patterns),

            # Combined signal (advanced + candlestick)
            'combined_signal':     combined_signal,
            'signal_strength':     self._signal_strength(best, pat_ctx),

            # For DecisionAgent
            'has_pattern':         best is not None,
            'pattern_bullish':     best.get('direction') == 'BULLISH' if best else False,
            'pattern_bearish':     best.get('direction') == 'BEARISH' if best else False,
        }

    def _combined_signal(self, best_pattern: dict | None, pat_ctx: dict | None) -> str:
        """Advanced pattern + candlestick pattern combination"""
        adv_dir = best_pattern.get('direction', 'NEUTRAL') if best_pattern else 'NEUTRAL'
        adv_conf = best_pattern.get('confidence', 0) if best_pattern else 0

        candle_signal = pat_ctx.get('pattern_signal', '') if pat_ctx else ''

        if adv_dir == 'BULLISH' and 'Bullish' in candle_signal and adv_conf >= 70:
            return 'STRONG_BUY'
        if adv_dir == 'BEARISH' and 'Bearish' in candle_signal and adv_conf >= 70:
            return 'STRONG_SELL'
        if adv_dir == 'BULLISH' and adv_conf >= 60:
            return 'BUY'
        if adv_dir == 'BEARISH' and adv_conf >= 60:
            return 'SELL'
        return 'NEUTRAL'

    def _signal_strength(self, best: dict | None, pat_ctx: dict | None) -> str:
        if not best:
            return 'NONE'
        conf = best.get('confidence', 0)
        if conf >= 85:  return 'VERY_HIGH'
        if conf >= 70:  return 'HIGH'
        if conf >= 55:  return 'MEDIUM'
        return 'LOW'

    # ═══════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════════

    def print_summary(self, patterns: list[dict]):
        print("\n" + "═" * 58)
        print("  📈  ADVANCED PATTERN RECOGNITION  (Day 39)")
        print("═" * 58)

        if not patterns:
            print("  No advanced patterns detected in recent data.")
            print("═" * 58 + "\n")
            return

        for i, pat in enumerate(patterns[:5], 1):
            direction  = pat.get('direction', 'NEUTRAL')
            icon       = '🟢' if direction == 'BULLISH' else ('🔴' if direction == 'BEARISH' else '🟡')
            conf       = pat.get('confidence', 0)
            conf_bar   = '█' * (conf // 10) + '░' * (10 - conf // 10)
            boost      = pat.get('confidence_boost', 0)
            boost_str  = f" ({'+' if boost >= 0 else ''}{boost})" if boost != 0 else ''

            print(f"\n  {i}. {icon} {pat.get('pattern')}")
            print(f"     Direction   : {direction}")
            print(f"     Confidence  : [{conf_bar}] {conf}%{boost_str}")
            if pat.get('entry'):
                print(f"     Entry       : {pat.get('entry')}")
            if pat.get('target'):
                print(f"     Target      : {pat.get('target')}")
            if pat.get('invalidation'):
                print(f"     Invalidation: {pat.get('invalidation')}")
            if pat.get('boost_reasons'):
                print(f"     Boost       : {' | '.join(pat['boost_reasons'][:3])}")
            print(f"     Note        : {pat.get('note', '')[:65]}")

        print("\n" + "═" * 58 + "\n")

    # ═══════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════

    def _find_swings(
        self,
        values: np.ndarray,
        mode:   str = 'high',
        window: int = 5,
    ) -> list[tuple[int, float]]:
        """Detect swing highs or lows."""
        swings = []
        n = len(values)
        for i in range(window, n - window):
            segment = values[i - window: i + window + 1]
            center  = values[i]
            if mode == 'high' and center == max(segment):
                swings.append((i, center))
            elif mode == 'low' and center == min(segment):
                swings.append((i, center))
        return swings

    def _atr_simple(self, df: pd.DataFrame, period: int = 14) -> float:
        """Simple ATR calculation।"""
        try:
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else None
            if atr and not np.isnan(atr):
                return float(atr)
            # Fallback
            highs  = df['high'].values[-period:]
            lows   = df['low'].values[-period:]
            closes = df['close'].values[-period:]
            trs    = [max(h - l, abs(h - c), abs(l - c))
                      for h, l, c in zip(highs[1:], lows[1:], closes[:-1])]
            return np.mean(trs) if trs else 0.0001
        except Exception as e:
            return 0.0001

    # ═════════════════════════════════════════════════════════════
    # Day 100+ — Book Pages 112-113: RECTANGLE PATTERN
    # ═════════════════════════════════════════════════════════════

    def detect_rectangle(self, df: pd.DataFrame) -> list[dict]:
        """
        Book Page 112-113 — Rectangle Pattern
        =======================================
        Price bounces between two parallel horizontal S/R lines.
        - Longer duration = more significant pattern
        - Breakout often accompanied by volume surge
        - Long entry on breakout above resistance
        - Short entry on breakout below support
        - Optional: wait for retest/pullback before entry

        Pseudocode (from book):
          IF price_oscillates_between(support, resistance, min_touches)
             AND duration > min_bars:
              Pattern = "Rectangle"
          IF price breaks_above resistance AND volume_confirmed:
              Signal = "Long"
          IF price breaks_below support AND volume_confirmed:
              Signal = "Short"
        """
        results = []
        if len(df) < 30:
            return results

        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        vols   = df['volume'].values if 'volume' in df.columns else None
        n      = len(df)

        # Use recent 40-80 candles
        window = min(n, 80)
        h_seg  = highs[-window:]
        l_seg  = lows[-window:]
        c_seg  = closes[-window:]

        atr = self._atr_simple(df)
        if atr <= 0 or np.isnan(atr):
            return results

        # Check if both highs and lows are approximately horizontal (flat)
        x = np.arange(len(h_seg))
        h_slope, h_intercept = np.polyfit(x, h_seg, 1)
        l_slope, l_intercept = np.polyfit(x, l_seg, 1)

        slope_threshold = atr * 0.02
        if abs(h_slope) > slope_threshold or abs(l_slope) > slope_threshold:
            return results

        resistance = float(h_intercept + h_slope * (len(h_seg) - 1))
        support    = float(l_intercept + l_slope * (len(l_seg) - 1))
        height     = resistance - support

        if height < atr * 0.5:
            return results

        res_touches = sum(1 for h in h_seg if abs(h - resistance) < atr * 0.3)
        sup_touches = sum(1 for l in l_seg if abs(l - support) < atr * 0.3)

        if res_touches < 2 or sup_touches < 2:
            return results

        curr_price = float(closes[-1])
        avg_vol    = float(np.mean(vols[-20:])) if vols is not None else 0
        curr_vol   = float(vols[-1]) if vols is not None else 0
        vol_surge  = curr_vol > avg_vol * 1.3 if avg_vol > 0 else False

        broke_up   = curr_price > resistance + atr * 0.1
        broke_down = curr_price < support - atr * 0.1

        if not (broke_up or broke_down):
            results.append({
                'pattern':       'RECTANGLE',
                'direction':     'NEUTRAL',
                'trade_action':  'NO_TRADE',
                'confidence':    65,
                'resistance':    round(resistance, 5),
                'support':       round(support, 5),
                'res_touches':   res_touches,
                'sup_touches':   sup_touches,
                'entry':         None,
                'target':        None,
                'invalidation':  None,
                'note':          (
                    f"Rectangle forming [{support:.5f} – {resistance:.5f}] "
                    f"({res_touches} res touches, {sup_touches} sup touches). "
                    f"Wait for breakout. NO_TRADE until confirmed."
                ),
            })
        elif broke_up:
            target = round(resistance + height, 5)
            inval  = round(resistance - atr * 0.5, 5)
            conf   = 70 + (10 if vol_surge else 0)
            results.append({
                'pattern':       'RECTANGLE_BREAKOUT_UP',
                'direction':     'BULLISH',
                'trade_action':  'LONG',
                'confidence':    min(95, conf),
                'resistance':    round(resistance, 5),
                'support':       round(support, 5),
                'entry':         round(resistance, 5),
                'target':        target,
                'invalidation':  inval,
                'volume_confirmed': bool(vol_surge),
                'note':          (
                    f"Rectangle breakout UP at {resistance:.5f}. "
                    f"LONG entry. Target={target:.5f} "
                    f"({'volume confirmed' if vol_surge else 'no volume confirm'})"
                ),
            })
        elif broke_down:
            target = round(support - height, 5)
            inval  = round(support + atr * 0.5, 5)
            conf   = 70 + (10 if vol_surge else 0)
            results.append({
                'pattern':       'RECTANGLE_BREAKOUT_DOWN',
                'direction':     'BEARISH',
                'trade_action':  'SHORT',
                'confidence':    min(95, conf),
                'resistance':    round(resistance, 5),
                'support':       round(support, 5),
                'entry':         round(support, 5),
                'target':        target,
                'invalidation':  inval,
                'volume_confirmed': bool(vol_surge),
                'note':          (
                    f"Rectangle breakdown at {support:.5f}. "
                    f"SHORT entry. Target={target:.5f} "
                    f"({'volume confirmed' if vol_surge else 'no volume confirm'})"
                ),
            })

        return results

    # ═════════════════════════════════════════════════════════════
    # Day 100+ — Book Page 120: MOMENTUM SCREENER (52-week high)
    # ═════════════════════════════════════════════════════════════

    def detect_momentum_screen(self, df: pd.DataFrame) -> list[dict]:
        """
        Book Page 120 — Momentum Screener (52-Week High Proximity)
        ==========================================================
        Identifies securities with strong momentum by checking:
          1. Proximity to 52-week high (within X% — book suggests 10%)
          2. Trailing % price change ranking

        Pseudocode (from book):
          proximity_to_high = (high_52wk - current_price) / high_52wk
          IF proximity_to_high <= 0.10:
              momentum_candidate = TRUE

          momentum_rank = pct_change(price, lookback=12_or_24_periods)
          # Rank securities by momentum_rank descending

        For FX (which trades 24/5), 52-week high = highest high in last
        ~252 trading days (~1 year). For intraday data, we use lookback
        proportional to data length.

        This is a SCREENING rule, not a directional signal by itself —
        it identifies momentum candidates for further analysis.
        """
        results = []
        if len(df) < 30:
            return results

        highs  = df['high'].values
        closes = df['close'].values
        n      = len(df)

        # 52-week high — for daily data use 252 bars; for intraday, use
        # entire available lookback (capped at 1000 bars)
        lookback_high = min(n, 252) if n >= 252 else min(n, 1000)
        high_52wk     = float(np.max(highs[-lookback_high:]))
        curr_price    = float(closes[-1])

        if high_52wk <= 0:
            return results

        proximity_to_high = (high_52wk - curr_price) / high_52wk

        lookback_12 = min(n, 12)
        lookback_24 = min(n, 24)
        pct_12 = (curr_price - float(closes[-lookback_12])) / float(closes[-lookback_12]) if lookback_12 > 0 else 0
        pct_24 = (curr_price - float(closes[-lookback_24])) / float(closes[-lookback_24]) if lookback_24 > 0 else 0

        MOMENTUM_THRESHOLD = 0.10
        is_momentum_candidate = proximity_to_high <= MOMENTUM_THRESHOLD

        if is_momentum_candidate:
            conf = 70
            conf += int(25 * (1 - proximity_to_high / MOMENTUM_THRESHOLD))
            if pct_12 > 0:
                conf += min(10, int(pct_12 * 1000))
            confidence = min(95, conf)

            direction = 'BULLISH' if (pct_12 > 0 or pct_24 > 0) else 'NEUTRAL'

            results.append({
                'pattern':              'MOMENTUM_CANDIDATE',
                'direction':            direction,
                'trade_action':         'WATCH_LONG' if direction == 'BULLISH' else 'WATCH',
                'confidence':           confidence,
                'high_52wk':            round(high_52wk, 5),
                'current_price':        round(curr_price, 5),
                'proximity_to_high':    round(proximity_to_high * 100, 2),
                'pct_change_12':        round(pct_12 * 100, 2),
                'pct_change_24':        round(pct_24 * 100, 2),
                'lookback_bars':        lookback_high,
                'momentum_threshold_pct': MOMENTUM_THRESHOLD * 100,
                'note':                 (
                    f"Momentum candidate: price {proximity_to_high*100:.2f}% below "
                    f"{lookback_high}-bar high ({high_52wk:.5f}). "
                    f"12-bar change: {pct_12*100:+.2f}%, 24-bar: {pct_24*100:+.2f}%. "
                    f"{'Strong bullish momentum.' if direction == 'BULLISH' else 'Near high but no positive momentum.'}"
                ),
            })

        return results


# ═══════════════════════════════════════════════════════════════
# QUICK RUN — Direct test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data.fetcher import DataFetcher
    from data.indicators import Indicators

    fetcher = DataFetcher()
    ind     = Indicators()

    df = fetcher.fetch_ohlcv("EURUSD", "1h", limit=200)
    if df is not None:
        df      = ind.add_all(df)
        ind_ctx = ind.get_ai_context(df)

        detector = AdvancedPatternDetector(lookback=100)
        patterns = detector.detect_all(df)
        patterns = detector.boost_confidence(patterns, ind_ctx=ind_ctx)
        patterns = detector.filter_false_patterns(patterns)
        detector.print_summary(patterns)

        ctx = detector.get_ai_context(df, ind_ctx=ind_ctx)
        print("AI Context:")
        for k, v in ctx.items():
            if k != 'all_patterns':
                print(f"  {k:<25}: {v}")