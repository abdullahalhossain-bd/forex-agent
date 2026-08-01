# analysis/ichimoku.py  —  Day 84 | Ichimoku Kinko Hyo Engine
# ============================================================
# Ichimoku Forex-এ অন্যতম জনপ্রিয় trend + support/resistance system।
# এটা একসাথে ৫টা জিনিস দেয়:
#
#   1. Tenkan-sen (9)   : short-term momentum / pivot
#   2. Kijun-sen  (26)  : medium-term trend baseline
#   3. Senkou A         : (Tenkan+Kijun)/2 shifted 26 ahead → cloud upper
#   4. Senkou B         : (52H+52L)/2 shifted 26 ahead → cloud lower
#   5. Chikou Span      : close shifted 26 back → momentum confirm
#
# Cloud (Kumo) = Senkou A ও B এর মধ্যবর্তী অঞ্চল।
#   Price above cloud → bullish bias
#   Price below cloud → bearish bias
#   Price inside cloud → ranging/neutral
#   Green cloud (A>B)  → bullish
#   Red cloud   (A<B)  → bearish
#
# Output:
#   {
#     "trend":      "BULLISH" | "BEARISH" | "NEUTRAL",
#     "cloud":      "ABOVE" | "BELOW" | "INSIDE",
#     "cloud_color":"GREEN" | "RED",
#     "strength":   0-100,
#     "tenkan":     float,
#     "kijun":      float,
#     "senkou_a":   float,   # current cloud upper
#     "senkou_b":   float,   # current cloud lower
#     "tk_cross":   "BULLISH" | "BEARISH" | "NONE",
#     "price_vs_kijun": "ABOVE" | "BELOW",
#     "chikou_clear": True/False,
#     "signal":     "BUY" | "SELL" | "WAIT",
#     "note":       str
#   }
# ============================================================

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("ichimoku_engine")


class IchimokuEngine:
    """
    Standard parameters: 9 / 26 / 52 / 26
    কিছু trader 7/14/42 ব্যবহার করে — সেটা __init__-এ দিলে হবে।
    """

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period:  int = 26,
        senkou_b_period: int = 52,
        displacement:  int = 26,
    ):
        self.tenkan_period   = tenkan_period
        self.kijun_period    = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement    = displacement

    # ═══════════════════════════════════════════════════════
    # MAIN ENTRY
    # ═══════════════════════════════════════════════════════

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        df-এ high/low/close লাগবে। অন্তত 52+displacement টা candle।
        """
        if df is None or len(df) < self.senkou_b_period + self.displacement + 5:
            return self._empty_result(
                f"Need at least {self.senkou_b_period + self.displacement + 5} candles"
            )

        # Step 1: Calculate 5 components
        df = df.copy()
        df["tenkan"] = self._donchian_mid(df, self.tenkan_period, "high", "low")
        df["kijun"]  = self._donchian_mid(df, self.kijun_period,  "high", "low")

        # Senkou A = (Tenkan + Kijun) / 2, projected 26 candles FORWARD.
        # Ichimoku convention: the cloud value at candle T is computed from
        # candle T-26's tenkan/kijun. So at the LATEST candle, the relevant
        # cloud is the raw senkou_a/b from 26 candles ago.
        df["senkou_a_raw"] = (df["tenkan"] + df["kijun"]) / 2
        df["senkou_b_raw"] = self._donchian_mid(df, self.senkou_b_period, "high", "low")

        # For charting: forward-shift by displacement (last `displacement` rows become NaN).
        # For analysis at latest candle: read raw value from `displacement` rows back.
        df["senkou_a"] = df["senkou_a_raw"].shift(self.displacement)
        df["senkou_b"] = df["senkou_b_raw"].shift(self.displacement)

        # Chikou Span = close shifted 26 BACK (past)
        df["chikou"] = df["close"].shift(self.displacement)

        last = df.iloc[-1]
        close = float(last["close"])

        # Cloud values at current candle (the projection 26 candles ago)
        senkou_a = float(last.get("senkou_a", np.nan)) if not np.isnan(last.get("senkou_a", np.nan)) else 0.0
        senkou_b = float(last.get("senkou_b", np.nan)) if not np.isnan(last.get("senkou_b", np.nan)) else 0.0

        tenkan = float(last.get("tenkan", 0)) if not np.isnan(last.get("tenkan", np.nan)) else 0.0
        kijun  = float(last.get("kijun", 0))  if not np.isnan(last.get("kijun", np.nan)) else 0.0

        # Step 2: Cloud position
        cloud_pos, cloud_color = self._cloud_position(close, senkou_a, senkou_b)

        # Step 3: Tenkan/Kijun cross
        tk_cross = self._tk_cross(df)

        # Step 4: Price vs Kijun
        price_vs_kijun = "ABOVE" if close > kijun else "BELOW"

        # Step 5: Chikou clearance (chikou above/below close 26 candles ago)
        chikou_clear = self._chikou_clear(df)

        # Step 6: Trend + strength
        trend, strength = self._assess_trend(
            cloud_pos, cloud_color, tk_cross, price_vs_kijun, chikou_clear
        )

        # Step 7: Signal
        signal, note = self._signal(trend, tk_cross, cloud_pos, chikou_clear)

        result = {
            "valid":          True,
            "trend":          trend,
            "cloud":          cloud_pos,
            "cloud_color":    cloud_color,
            "strength":       strength,
            "tenkan":         round(tenkan, 5),
            "kijun":          round(kijun, 5),
            "senkou_a":       round(senkou_a, 5),
            "senkou_b":       round(senkou_b, 5),
            "tk_cross":       tk_cross,
            "price_vs_kijun": price_vs_kijun,
            "chikou_clear":   chikou_clear,
            "signal":         signal,
            "note":           note,
            "close":          round(close, 5),
        }

        log.info(
            f"[Ichimoku] trend={trend} | cloud={cloud_pos}({cloud_color}) | "
            f"TK={tk_cross} | strength={strength} | signal={signal}"
        )
        return result

    # ═══════════════════════════════════════════════════════
    # CALCULATION HELPERS
    # ═══════════════════════════════════════════════════════

    def _donchian_mid(
        self, df: pd.DataFrame, period: int, high_col: str, low_col: str
    ) -> pd.Series:
        """
        Donchian channel midpoint = (highest_high + lowest_low) / 2
        over rolling window of `period`।
        """
        hh = df[high_col].rolling(window=period, min_periods=1).max()
        ll = df[low_col].rolling(window=period, min_periods=1).min()
        return (hh + ll) / 2

    def _cloud_position(self, close: float, sa: float, sb: float) -> tuple[str, str]:
        """
        Price কোথায় cloud-এর relative।
        """
        if sa == 0 or sb == 0:
            return "UNKNOWN", "UNKNOWN"

        cloud_top = max(sa, sb)
        cloud_bot = min(sa, sb)

        # Color
        color = "GREEN" if sa > sb else "RED"

        # Position
        if close > cloud_top:   return "ABOVE", color
        if close < cloud_bot:   return "BELOW", color
        return "INSIDE", color

    def _tk_cross(self, df: pd.DataFrame) -> str:
        """
        Tenkan/Kijun cross — bullish if Tenkan > Kijun
        (Tenkan crossing above Kijun = bullish signal)
        """
        if len(df) < 2:
            return "NONE"
        last  = df.iloc[-1]
        prev  = df.iloc[-2]

        t_now, k_now = last.get("tenkan"), last.get("kijun")
        t_prev, k_prev = prev.get("tenkan"), prev.get("kijun")

        if any(pd.isna(v) for v in (t_now, k_now, t_prev, k_prev)):
            # Fall back to current state only
            if not pd.isna(t_now) and not pd.isna(k_now):
                return "BULLISH" if t_now > k_now else "BEARISH"
            return "NONE"

        # Cross detection
        if t_prev <= k_prev and t_now > k_now:
            return "BULLISH"
        if t_prev >= k_prev and t_now < k_now:
            return "BEARISH"
        # State (no cross this candle)
        return "BULLISH" if t_now > k_now else "BEARISH"

    def _chikou_clear(self, df: pd.DataFrame) -> bool:
        """
        Chikou Span = the CURRENT close, plotted `displacement` candles back on
        the chart. "Clear" (bullish confirmation) means the current close is
        trading above where price was `displacement` candles ago — i.e. no
        overhead resistance at the point where Chikou is plotted.

        BUGFIX (institutional review): the previous implementation compared
        `df['chikou'].iloc[-1]` (= close.shift(displacement) evaluated at the
        last row, i.e. close from `displacement` candles ago) against
        `df['close'].iloc[-(displacement+1)]` (ALSO close from `displacement`
        candles ago). Both sides were the identical value, so
        `chikou_now > compare_close` was comparing a number to itself and was
        ALWAYS False, regardless of real market conditions. This silently
        pushed one vote to bear_votes in `_assess_trend()` on every call and
        made the "all signals aligned bullish -> strong BUY" branch in
        `_signal()` unreachable. Fixed by comparing current close (the actual
        Chikou value) against price `displacement` candles ago (the level the
        Chikou line pokes through on the chart).
        """
        if len(df) < self.displacement + 1:
            return False
        chikou_now = float(df["close"].iloc[-1])
        compare_close = float(df["close"].iloc[-(self.displacement + 1)])
        return chikou_now > compare_close

    # ═══════════════════════════════════════════════════════
    # TREND + STRENGTH
    # ═══════════════════════════════════════════════════════

    def _assess_trend(
        self,
        cloud_pos:    str,
        cloud_color:  str,
        tk_cross:     str,
        price_vs_kijun: str,
        chikou_clear: bool,
    ) -> tuple[str, int]:
        """
        সব signal একসাথে দেখে trend + 0-100 strength বের করো।
        """
        bull_votes = 0
        bear_votes = 0

        # Cloud position
        if cloud_pos == "ABOVE":           bull_votes += 2
        elif cloud_pos == "BELOW":         bear_votes += 2
        # INSIDE = no vote

        # Cloud color
        if cloud_color == "GREEN":         bull_votes += 1
        elif cloud_color == "RED":         bear_votes += 1

        # TK cross
        if tk_cross == "BULLISH":          bull_votes += 1
        elif tk_cross == "BEARISH":        bear_votes += 1

        # Price vs Kijun
        if price_vs_kijun == "ABOVE":      bull_votes += 1
        elif price_vs_kijun == "BELOW":    bear_votes += 1

        # Chikou
        if chikou_clear:                   bull_votes += 1
        else:                              bear_votes += 1

        total = bull_votes + bear_votes
        if total == 0:
            return "NEUTRAL", 0

        if bull_votes > bear_votes:
            trend = "BULLISH"
            strength = int((bull_votes / total) * 100)
        elif bear_votes > bull_votes:
            trend = "BEARISH"
            strength = int((bear_votes / total) * 100)
        else:
            return "NEUTRAL", 50

        return trend, max(0, min(100, strength))

    def _signal(
        self, trend: str, tk_cross: str, cloud_pos: str, chikou_clear: bool
    ) -> tuple[str, str]:
        """
        Strong BUY = trend BULLISH + cloud ABOVE + chikou clear
        Strong SELL = trend BEARISH + cloud BELOW + chikou not clear
        Otherwise WAIT
        """
        if trend == "BULLISH" and cloud_pos == "ABOVE" and chikou_clear:
            return "BUY", "All Ichimoku signals aligned bullish — strong buy"
        if trend == "BEARISH" and cloud_pos == "BELOW" and not chikou_clear:
            return "SELL", "All Ichimoku signals aligned bearish — strong sell"
        if trend == "BULLISH" and cloud_pos in ("ABOVE", "INSIDE"):
            return "BUY", "Bullish bias — but cloud/chikou not fully confirmed"
        if trend == "BEARISH" and cloud_pos in ("BELOW", "INSIDE"):
            return "SELL", "Bearish bias — but cloud/chikou not fully confirmed"
        return "WAIT", "Mixed signals — wait for alignment"

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT
    # ═══════════════════════════════════════════════════════

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not result.get("valid"):
            return {
                "ichimoku_valid":      False,
                "ichimoku_trend":      "NEUTRAL",
                "ichimoku_cloud":      "UNKNOWN",
                "ichimoku_signal":     "WAIT",
                "ichimoku_strength":   0,
            }

        return {
            "ichimoku_valid":        True,
            "ichimoku_trend":        result.get("trend"),
            "ichimoku_cloud":        result.get("cloud"),
            "ichimoku_cloud_color":  result.get("cloud_color"),
            "ichimoku_strength":     result.get("strength"),
            "ichimoku_signal":       result.get("signal"),
            "ichimoku_tk_cross":     result.get("tk_cross"),
            "ichimoku_price_vs_kijun": result.get("price_vs_kijun"),
            "ichimoku_chikou_clear": result.get("chikou_clear"),
        }

    # ═══════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "valid":        False,
            "reason":       reason,
            "trend":        "NEUTRAL",
            "cloud":        "UNKNOWN",
            "cloud_color":  "UNKNOWN",
            "strength":     0,
            "signal":       "WAIT",
            "note":         reason,
        }

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 56
        log.info(bar)
        log.info("  ☁️  ICHIMOKU ENGINE  (Day 84)")
        log.info(bar)

        if not result.get("valid"):
            log.info(f"  ⚠️  {result.get('reason', 'No analysis')}")
            log.info(bar)
            return

        icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(
            result["trend"], "❓"
        )
        cloud_icon = {"ABOVE": "⬆️", "BELOW": "⬇️", "INSIDE": "☁️"}.get(
            result["cloud"], "❓"
        )
        color_icon = {"GREEN": "🟢", "RED": "🔴"}.get(result["cloud_color"], "❓")

        log.info(f"  Trend        : {icon}  {result['trend']}  (strength {result['strength']}/100)")
        log.info(f"  Cloud        : {cloud_icon}  Price {result['cloud']}  ({color_icon} {result['cloud_color']})")
        log.info(f"  Tenkan (9)   : {result['tenkan']}")
        log.info(f"  Kijun  (26)  : {result['kijun']}")
        log.info(f"  Senkou A     : {result['senkou_a']}")
        log.info(f"  Senkou B     : {result['senkou_b']}")
        log.info(f"  TK Cross     : {result['tk_cross']}")
        log.info(f"  Price/Kijun  : {result['price_vs_kijun']}")
        log.info(f"  Chikou Clear : {result['chikou_clear']}")
        log.info(f"  Signal       : {result['signal']}")
        log.info(f"  Note         : {result['note']}")
        log.info(bar)


# ═══════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    n = 200
    prices = 1.1000 + np.cumsum(np.random.randn(n) * 0.0005) + np.linspace(0, 0.005, n)

    df = pd.DataFrame({
        "open":  prices,
        "high":  prices + 0.0005,
        "low":   prices - 0.0005,
        "close": prices,
    })

    engine = IchimokuEngine()
    result = engine.analyze(df)
    engine.print_summary(result)

    ctx = engine.get_ai_context(result)
    print("\nAI Context:")
    for k, v in ctx.items():
        print(f"  {k:<28}: {v}")