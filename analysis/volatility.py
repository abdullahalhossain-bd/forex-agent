# analysis/volatility.py  —  Day 85 | Volatility & Bollinger Squeeze Engine
# ============================================================
# আপনার বলা "Low volatility compression → Big breakout probability"
# এটাই সেই module।
#
# এই module ৩টা জিনিস detect করে:
#
#   1. SQUEEZE STATE
#      Bollinger Band width historical low-এ গেছে কিনা।
#      Squeeze ON  = compression phase, breakout আসার সম্ভাবনা
#      SQUEEZE OFF = expansion phase শুরু, momentum বাড়ছে
#
#   2. ATR REGIME
#      ATR current vs historical percentile
#      VOLATILE (>75th), NORMAL (25-75), QUIET (<25)
#
#   3. SQUEEZE RELEASE
#      যখন squeeze off হয় এবং price BB band break করে →
#      breakout direction detect করে
#
# Output:
#   {
#     "squeeze_on":      True/False,
#     "squeeze_strength": "EXTREME"|"HIGH"|"MODERATE"|"NONE",
#     "bb_width":        float,
#     "bb_width_pct":    float,    # percentile rank
#     "atr":             float,
#     "atr_regime":      "VOLATILE"|"NORMAL"|"QUIET",
#     "atr_pct":         float,
#     "release":         "BULLISH"|"BEARISH"|"NONE",
#     "expansion_prob":  0-100,    # breakout আসার probability
#     "signal":          "WAIT_BREAKOUT"|"BREAKOUT_UP"|"BREAKOUT_DOWN"|"EXPANDED",
#     "note":            str
#   }
# ============================================================

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("volatility_engine")


class VolatilityEngine:
    """
    Bollinger Squeeze + ATR regime + breakout release detector।
    """

    def __init__(
        self,
        bb_window:         int = 20,
        bb_std:            float = 2.0,
        squeeze_lookback:  int = 120,
        squeeze_percentile: float = 20.0,   # BB width যদি historical lowest 20%-এ থাকে
        atr_period:        int = 14,
        atr_lookback:      int = 100,
    ):
        self.bb_window          = bb_window
        self.bb_std             = bb_std
        self.squeeze_lookback   = squeeze_lookback
        self.squeeze_percentile = squeeze_percentile
        self.atr_period         = atr_period
        self.atr_lookback       = atr_lookback

    # ═══════════════════════════════════════════════════════
    # MAIN ENTRY
    # ═══════════════════════════════════════════════════════

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        df-এ high/low/close লাগবে। indicators.add_all করা থাকলেও কাজ করবে,
        না থাকলেও নিজে calculate করবে।
        """
        if df is None or len(df) < max(self.bb_window * 3, self.squeeze_lookback):
            return self._empty_result("Insufficient data for volatility analysis")

        df = df.copy()

        # ── Bollinger Bands ──
        df = self._add_bollinger(df)

        # ── ATR ──
        df = self._add_atr(df)

        last = df.iloc[-1]
        close = float(last["close"])
        bb_upper = float(last["bb_upper"])
        bb_lower = float(last["bb_lower"])
        bb_middle = float(last["bb_middle"])
        bb_width = float(last["bb_width"])
        atr = float(last["atr"])

        # ── Squeeze detection ──
        bb_width_history = df["bb_width"].iloc[-self.squeeze_lookback:-1].dropna()
        if len(bb_width_history) < 20:
            return self._empty_result("Insufficient BB width history")

        width_pct = self._percentile(bb_width, bb_width_history)
        squeeze_on = width_pct <= self.squeeze_percentile
        squeeze_strength = self._squeeze_strength(width_pct)

        # ── ATR regime ──
        atr_history = df["atr"].iloc[-self.atr_lookback:-1].dropna()
        atr_pct = self._percentile(atr, atr_history) if len(atr_history) > 10 else 50.0
        atr_regime = self._atr_regime(atr_pct)

        # ── Squeeze release ──
        release = self._detect_release(df, close, bb_upper, bb_lower, squeeze_on)

        # ── Expansion probability ──
        expansion_prob = self._expansion_probability(
            width_pct, atr_pct, release
        )

        # ── Signal ──
        signal, note = self._signal(
            squeeze_on, squeeze_strength, release, expansion_prob
        )

        result = {
            "valid":            True,
            "squeeze_on":       squeeze_on,
            "squeeze_strength": squeeze_strength,
            "bb_upper":         round(bb_upper, 5),
            "bb_middle":        round(bb_middle, 5),
            "bb_lower":         round(bb_lower, 5),
            "bb_width":         round(bb_width, 5),
            "bb_width_pct":     round(width_pct, 1),
            "atr":              round(atr, 5),
            "atr_regime":       atr_regime,
            "atr_pct":          round(atr_pct, 1),
            "release":          release,
            "expansion_prob":   expansion_prob,
            "signal":           signal,
            "note":             note,
            "close":            round(close, 5),
        }

        log.info(
            f"[Volatility] squeeze={squeeze_on}({squeeze_strength}) | "
            f"BB_width_pct={width_pct:.1f}% | ATR_regime={atr_regime} | "
            f"release={release} | prob={expansion_prob}%"
        )
        return result

    # ═══════════════════════════════════════════════════════
    # BOLLINGER BANDS
    # ═══════════════════════════════════════════════════════

    def _add_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        যদি indicators.add_all আগে থেকেই column বানিয়ে থাকে, সেটা reuse করো।
        """
        if "bb_upper" in df.columns and "bb_lower" in df.columns and "bb_width" in df.columns:
            return df

        rolling = df["close"].rolling(window=self.bb_window, min_periods=1)
        mean = rolling.mean()
        std  = rolling.std()

        df["bb_upper"]  = mean + self.bb_std * std
        df["bb_middle"] = mean
        df["bb_lower"]  = mean - self.bb_std * std
        # Width as percentage of middle band
        df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / mean.replace(0, np.nan) * 100
        return df

    # ═══════════════════════════════════════════════════════
    # ATR
    # ═══════════════════════════════════════════════════════

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        if "atr" in df.columns:
            # Check no NaN at the tail
            if not np.isnan(df["atr"].iloc[-1]):
                return df

        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low  - close.shift(1)).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df["atr"] = tr.ewm(alpha=1 / self.atr_period, adjust=False).mean()
        return df

    # ═══════════════════════════════════════════════════════
    # SQUEEZE LOGIC
    # ═══════════════════════════════════════════════════════

    def _percentile(self, value: float, history: pd.Series) -> float:
        """
        Value-র historical distribution-এ percentile।
        """
        arr = history.values
        return float((arr < value).sum() / max(1, len(arr)) * 100)

    def _squeeze_strength(self, width_pct: float) -> str:
        """
        Lower percentile = stronger squeeze।
        """
        if width_pct <= 5:    return "EXTREME"
        if width_pct <= 15:   return "HIGH"
        if width_pct <= 25:   return "MODERATE"
        return "NONE"

    def _atr_regime(self, atr_pct: float) -> str:
        if atr_pct >= 75:   return "VOLATILE"
        if atr_pct <= 25:   return "QUIET"
        return "NORMAL"

    def _detect_release(
        self,
        df: pd.DataFrame,
        close: float,
        bb_upper: float,
        bb_lower: float,
        squeeze_on: bool,
    ) -> str:
        """
        Squeeze release = price breaks out of BB bands।
        - Bullish release: close > BB upper (and squeeze was on recently)
        - Bearish release: close < BB lower
        - Otherwise: NONE
        """
        # Current candle breakout
        # BUG FIX (institutional review): both branches of the old if/else
        # returned the same value regardless of the squeeze-confirmation
        # check, which silently disabled the entire "release must follow a
        # squeeze" premise of this module (every close beyond a BB band was
        # reported as a squeeze "release", even in already-expanded,
        # high-volatility conditions with no compression at all). The check
        # below now actually gates the result on recent compression, and
        # returns "NONE" (an ordinary band break, not a squeeze release)
        # otherwise.
        if close > bb_upper:
            recent = df["bb_width"].iloc[-6:-1].dropna()
            recent_pct_min = self._percentile(recent.min(), df["bb_width"].iloc[-self.squeeze_lookback:-1].dropna()) if len(recent) > 0 else 100
            if squeeze_on or recent_pct_min <= 25:
                return "BULLISH"
            return "NONE"

        if close < bb_lower:
            recent = df["bb_width"].iloc[-6:-1].dropna()
            recent_pct_min = self._percentile(recent.min(), df["bb_width"].iloc[-self.squeeze_lookback:-1].dropna()) if len(recent) > 0 else 100
            if squeeze_on or recent_pct_min <= 25:
                return "BEARISH"
            return "NONE"

        return "NONE"

    def _expansion_probability(self, width_pct: float, atr_pct: float, release: str) -> int:
        """
        Squeeze থেকে expansion আসার probability 0-100।
        Stronger squeeze + ATR still low → higher probability।
        Already released → 100।
        """
        if release != "NONE":
            return 100

        # Lower width percentile = tighter squeeze = higher expansion prob
        squeeze_score = max(0, 100 - width_pct * 2)   # 0% width → 100, 50% → 0

        # Low ATR confirms compression (high ATR means already expanded)
        atr_score = max(0, 100 - atr_pct * 1.5)

        return max(0, min(100, int((squeeze_score * 0.6 + atr_score * 0.4))))

    def _signal(
        self,
        squeeze_on: bool,
        squeeze_strength: str,
        release: str,
        expansion_prob: int,
    ) -> tuple[str, str]:
        if release == "BULLISH":
            return "BREAKOUT_UP", "Squeeze released upward — bullish breakout"
        if release == "BEARISH":
            return "BREAKOUT_DOWN", "Squeeze released downward — bearish breakout"
        if squeeze_on and squeeze_strength in ("EXTREME", "HIGH"):
            return "WAIT_BREAKOUT", f"Squeeze {squeeze_strength} — breakout imminent, wait for direction"
        if squeeze_on:
            return "WAIT_BREAKOUT", "Squeeze active — wait for release"
        return "EXPANDED", "Volatility expanded — no squeeze setup"

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT
    # ═══════════════════════════════════════════════════════

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not result.get("valid"):
            return {
                "volatility_valid":         False,
                "squeeze_on":               False,
                "squeeze_strength":         "NONE",
                "atr_regime":               "NORMAL",
                "expansion_prob":           0,
                "volatility_signal":        "WAIT",
            }

        return {
            "volatility_valid":         True,
            "squeeze_on":               result.get("squeeze_on", False),
            "squeeze_strength":         result.get("squeeze_strength", "NONE"),
            "bb_width_pct":             result.get("bb_width_pct", 0),
            "atr_regime":               result.get("atr_regime", "NORMAL"),
            "atr_pct":                  result.get("atr_pct", 0),
            "expansion_prob":           result.get("expansion_prob", 0),
            "volatility_release":       result.get("release", "NONE"),
            "volatility_signal":        result.get("signal", "WAIT"),
        }

    # ═══════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "valid":            False,
            "reason":           reason,
            "squeeze_on":       False,
            "squeeze_strength": "NONE",
            "atr_regime":       "NORMAL",
            "expansion_prob":   0,
            "release":          "NONE",
            "signal":           "WAIT",
            "note":             reason,
        }

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 56
        log.info(bar)
        log.info("  📊  VOLATILITY ENGINE  (Day 85)")
        log.info(bar)

        if not result.get("valid"):
            log.info(f"  ⚠️  {result.get('reason', 'No analysis')}")
            log.info(bar)
            return

        sq_icon = "🔴" if result["squeeze_on"] else "⚪"
        log.info(f"  Squeeze      : {sq_icon}  {result['squeeze_on']}  ({result['squeeze_strength']})")
        log.info(f"  BB Width     : {result['bb_width']} (pct {result['bb_width_pct']}%)")
        log.info(f"  BB Upper/Low : {result['bb_upper']} / {result['bb_lower']}")
        log.info(f"  ATR          : {result['atr']} ({result['atr_regime']}, pct {result['atr_pct']}%)")
        log.info(f"  Release      : {result['release']}")
        log.info(f"  Expansion %  : {result['expansion_prob']}")
        log.info(f"  Signal       : {result['signal']}")
        log.info(f"  Note         : {result['note']}")
        log.info(bar)


# ═══════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    n = 250
    # Quiet phase (squeeze) then breakout
    quiet_len = 80
    quiet = 1.1000 + np.random.randn(quiet_len) * 0.0002   # very tight
    breakout_len = n - quiet_len
    breakout = 1.1000 + np.cumsum(np.random.randn(breakout_len) * 0.001) + np.linspace(0, 0.005, breakout_len)
    prices = np.concatenate([quiet, breakout])

    df = pd.DataFrame({
        "open":  prices,
        "high":  prices + 0.0005,
        "low":   prices - 0.0005,
        "close": prices,
    })

    engine = VolatilityEngine()
    result = engine.analyze(df)
    engine.print_summary(result)

    ctx = engine.get_ai_context(result)
    print("\nAI Context:")
    for k, v in ctx.items():
        print(f"  {k:<28}: {v}")