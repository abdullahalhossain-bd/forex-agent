# analysis/candlestick_patterns_ml.py — Candlestick pattern detectors for ML
# =============================================================================
# Ported from: https://github.com/MaxwellMendenhall/ml_backtest
# Original: ml_backtest/data/data.py → CandleStickPatterns class
# Original author: Maxwell Mendenhall — MIT license
#
# Simple boolean candlestick pattern detectors. Each returns True/False —
# these are the SIGNAL generators, not feature extractors (those are in
# ml/pattern_features.py).
#
# These complement analysis/candlestick_patterns_mw.py (the 33-pattern
# MotiveWave port). The differences:
#
#   - This module: 8 patterns, boolean output (is/isn't), used as entry triggers.
#   - MotiveWave port: 33 patterns, named output, used as a broad scanner.
#
# The two can coexist. The Mendenhall patterns are tuned slightly differently
# (e.g., Hammer requires total_range > 3*body, vs MotiveWave's body/range<0.1
# + lower>2*body). They will sometimes agree, sometimes disagree — both
# perspectives are useful.
# =============================================================================

from __future__ import annotations


class CandleStickPatterns:
    """8 candlestick pattern detectors — boolean output."""

    # ── 1-bar patterns ───────────────────────────────────────────────────────

    @staticmethod
    def is_inverted_hammer(current_open, current_close, current_high, current_low):
        """
        Inverted Hammer / Shooting Star.
        Total range > 3x body, upper shadow > 60% of range, lower shadow < 40%.
        """
        body_length = abs(current_open - current_close)
        upper_shadow = current_high - max(current_open, current_close)
        total_length = current_high - current_low
        lower_shadow = min(current_open, current_close) - current_low
        return (
            (total_length > 3 * body_length)
            and (upper_shadow / (0.001 + total_length) > 0.6)
            and (lower_shadow / (0.001 + total_length) < 0.4)
        )

    @staticmethod
    def is_hammer(current_open, current_close, current_high, current_low):
        """
        Hammer / Hanging Man.
        Total range > 3x body, close in upper 60% of range, open in upper 60%.
        """
        return (
            ((current_high - current_low) > 3 * abs(current_open - current_close))
            and ((current_close - current_low) / (0.001 + current_high - current_low) > 0.6)
            and ((current_open - current_low) / (0.001 + current_high - current_low) > 0.6)
        )

    @staticmethod
    def is_dragonfly_doji(current_open, current_close, current_high, current_low):
        """
        Dragonfly Doji.
        Body < 10% of range, lower shadow > 3x body, upper shadow < body.
        """
        body_range = abs(current_close - current_open)
        total_range = current_high - current_low
        upper_shadow = current_high - max(current_close, current_open)
        lower_shadow = min(current_close, current_open) - current_low
        return (
            (body_range / (total_range + 0.001) < 0.1)
            and (lower_shadow > (3 * body_range))
            and (upper_shadow < body_range)
        )

    # ── 2-bar patterns ───────────────────────────────────────────────────────

    @staticmethod
    def is_bullish_engulfing(current_open, current_close, prev_open, prev_close):
        """
        Bullish Engulfing.
        prev bearish, current bullish, current body engulfs prev body.
        """
        return (
            current_close >= prev_open > prev_close >= current_open
            and current_close > current_open
            and (current_close - current_open) > (prev_open - prev_close)
        )

    @staticmethod
    def is_bullish_harami(current_open, current_close, prev_open, prev_close):
        """
        Bullish Harami.
        prev bearish large body, current bullish small body contained within prev body.
        """
        return (
            prev_open > prev_close
            and prev_close <= current_open < current_close <= prev_open
            and (current_close - current_open) < (prev_open - prev_close)
        )

    @staticmethod
    def is_piercing_pattern(current_open, current_close, prev_open, prev_close):
        """
        Piercing Line.
        prev bearish, current bullish, opens below prev close, closes above prev midpoint
        but below prev open.
        """
        if not (prev_close < prev_open and current_close > current_open):
            return False
        prev_mid = (prev_open + prev_close) / 2.0
        return (
            current_open < prev_close
            and current_close > prev_mid
            and current_close < prev_open
        )

    # ── 3-bar patterns ───────────────────────────────────────────────────────

    @staticmethod
    def is_morning_star(b_prev_open, b_prev_close,
                        prev_open, prev_close,
                        current_open, current_close):
        """
        Morning Star.
        1st: large bearish, 2nd: small body (star), 3rd: large bullish closing
        above midpoint of 1st.
        """
        # 1st bearish, 3rd bullish
        if not (b_prev_close < b_prev_open and current_close > current_open):
            return False
        first_body = abs(b_prev_close - b_prev_open)
        second_body = abs(prev_close - prev_open)
        third_body = abs(current_close - current_open)
        # 2nd small body
        if second_body >= first_body * 0.5:
            return False
        # 3rd closes above midpoint of 1st
        first_mid = (b_prev_open + b_prev_close) / 2.0
        return (current_close > first_mid) and (third_body > first_body * 0.5)

    @staticmethod
    def is_morning_star_doji(b_prev_open, b_prev_close,
                             prev_open, prev_close, prev_high, prev_low,
                             current_open, current_close):
        """
        Morning Doji Star.
        Like Morning Star but middle candle is a doji (body < 10% of its range).
        """
        # 1st bearish, 3rd bullish
        if not (b_prev_close < b_prev_open and current_close > current_open):
            return False
        # Middle is doji
        second_body = abs(prev_close - prev_open)
        second_range = prev_high - prev_low
        if second_range == 0 or second_body / (second_range + 0.001) >= 0.1:
            return False
        # 3rd closes above 1st's close (gap-recovery)
        return current_close > b_prev_close


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Hammer: total range > 3x body, close in upper 60%, open in upper 60%
    # body=0.05, need range > 0.15 → low=0.85, high=1.06, range=0.21 ✓
    # close=1.05, low=0.85 → (1.05-0.85)/(0.001+0.21)=0.95 > 0.6 ✓
    # open=1.0, low=0.85 → (1.0-0.85)/(0.001+0.21)=0.71 > 0.6 ✓
    assert CandleStickPatterns.is_hammer(1.0, 1.05, 1.06, 0.85)
    print("OK Hammer")

    # Inverted Hammer: upper shadow dominant
    # body=0.01, range=0.11, body*3=0.03 < 0.11 ✓
    # upper=1.10-1.01=0.09, upper/(0.001+0.11)=0.81 > 0.6 ✓
    # lower=1.0-0.99=0.01, lower/(0.001+0.11)=0.09 < 0.4 ✓
    assert CandleStickPatterns.is_inverted_hammer(1.0, 1.01, 1.10, 0.99)
    print("OK Inverted Hammer")

    # Bullish Engulfing: prev bearish, current bullish, current body engulfs prev body
    # Need: current_close >= prev_open > prev_close >= current_open
    # prev_open=1.05, prev_close=0.95 (bearish)
    # current_open=0.95, current_close=1.10 (bullish, engulfs)
    # Check: 1.10>=1.05 ✓, 1.05>0.95 ✓, 0.95>=0.95 ✓
    assert CandleStickPatterns.is_bullish_engulfing(
        current_open=0.95, current_close=1.10, prev_open=1.05, prev_close=0.95
    )
    print("OK Bullish Engulfing")

    # Bullish Harami
    assert CandleStickPatterns.is_bullish_harami(
        current_open=0.95, current_close=1.00, prev_open=1.10, prev_close=0.90
    )
    print("OK Bullish Harami")

    # Piercing Pattern
    assert CandleStickPatterns.is_piercing_pattern(
        current_open=0.85, current_close=1.05, prev_open=1.10, prev_close=0.90
    )
    print("OK Piercing Pattern")

    # Morning Star: large bearish, small star, large bullish closing above midpoint
    assert CandleStickPatterns.is_morning_star(
        b_prev_open=1.20, b_prev_close=1.00,   # 1st: body=0.20, mid=1.10
        prev_open=0.99, prev_close=1.00,        # 2nd: body=0.01 (small)
        current_open=1.00, current_close=1.15,  # 3rd: body=0.15 (>0.20*0.5=0.10), close 1.15 > mid 1.10
    )
    print("OK Morning Star")

    # Morning Doji Star: middle candle must be a doji (body/range < 0.1)
    # prev_open=1.000, prev_close=1.001 (body=0.001), prev_high=1.05, prev_low=0.95 (range=0.10)
    # body/range = 0.001/0.101 = 0.0099 < 0.1 ✓
    assert CandleStickPatterns.is_morning_star_doji(
        b_prev_open=1.20, b_prev_close=1.00,
        prev_open=1.000, prev_close=1.001, prev_high=1.05, prev_low=0.95,
        current_open=1.00, current_close=1.15,
    )
    print("OK Morning Doji Star")

    print("\nAll candlestick pattern (ML port) smoke tests passed.")
