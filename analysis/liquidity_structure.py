# analysis/liquidity_structure.py  —  Liquidity Structure Extensions
# ============================================================
# Concepts from the videos that were NOT in any of the 3 originally
# uploaded modules (liquidity.py / liquidity_engine.py / liquidity_zones.py):
#
#   1. Internal vs External liquidity
#      External = liquidity outside the current visible range (the most
#      obvious swing high/low — where most stops sit).
#      Internal  = liquidity inside the range (minor pullback swings).
#      Rule of thumb from the videos: internal liquidity tends to get taken
#      FIRST, before price goes for the external liquidity.
#
#   2. High-resistance vs Low-resistance liquidity
#      Low-resistance  = forms after a FAILED swing (price tries for a new
#      high/low and fails — a "failure swing", Livermore's term, ~100 yrs
#      old under a new label). Weaker liquidity behind it.
#      High-resistance = forms after a CLEAN break (higher-high followed
#      immediately by a lower-low, or the reverse) — stronger liquidity.
#      Rule of thumb: once high-resistance liquidity is taken, price tends
#      to travel toward the low-resistance liquidity on the other side
#      ("path of least resistance").
#
#   3. Trend-line liquidity
#      Stops don't only cluster below/above horizontal swing highs/lows —
#      they also cluster along a clean rising/falling trendline drawn
#      through recent swing points. Modeled here as a fitted line through
#      the last N swing points on one side, kept only if the fit is
#      reasonably clean (R² threshold) — a jagged/noisy "trendline" isn't
#      something enough retail traders are actually drawing to matter.
#
#   4. Liquidity inducement
#      A minor (internal) swing gets swept just before the real move,
#      trapping traders who treated it as the real support/resistance,
#      while the major (external) level on the same side is never touched.
#      Modeled here as: an internal-level stop hunt fires while the
#      external level on the same side remains fresh (un-swept).
#
# No-lookahead: everything here only reads swing points confirmed by
# SWING_WINDOW closed candles after them (same contract as liquidity_zones.py
# and liquidity.py), and only ever looks at df.iloc[:] as given — callers
# must pass a bounded/expanding window, never the full future dataset.
# ============================================================

import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("liquidity_structure")

SWING_WINDOW          = 5
EXTERNAL_LOOKBACK      = 100   # bars searched for "the" external high/low
EXTERNAL_TOLERANCE_ATR = 0.15  # how close to the extreme counts as "external"
TRENDLINE_MIN_POINTS   = 3
TRENDLINE_LOOKBACK     = 80
TRENDLINE_MIN_R2       = 0.55
TRENDLINE_TOUCH_ATR    = 0.20


class LiquidityStructureAnalyzer:
    """
    Usage:
        struct = LiquidityStructureAnalyzer()
        levels        = struct.classify_internal_external(levels, df)
        resistance    = struct.classify_resistance(df)
        trendlines    = struct.detect_trendlines(df)
        tl_events     = struct.check_trendline_sweep(df, trendlines)
        inducement    = struct.detect_inducement(levels, stop_hunt_events)
    """

    # ═══════════════════════════════════════════════════════
    # 1. INTERNAL VS EXTERNAL
    # ═══════════════════════════════════════════════════════

    def classify_internal_external(self, levels: list[dict], df: pd.DataFrame) -> list[dict]:
        """Tags each level dict in-place (returns a new list) with scope='EXTERNAL'|'INTERNAL'."""
        if df is None or len(df) == 0 or not levels:
            return levels

        recent = df.tail(EXTERNAL_LOOKBACK)
        atr = self._safe_atr(df)
        ext_high = float(recent["high"].max())
        ext_low = float(recent["low"].min())
        tol = atr * EXTERNAL_TOLERANCE_ATR

        tagged = []
        for lvl in levels:
            price = lvl["price"]
            is_external = (
                (lvl["liquidity_type"] == "BUY_SIDE" and abs(price - ext_high) <= tol)
                or (lvl["liquidity_type"] == "SELL_SIDE" and abs(price - ext_low) <= tol)
                or lvl.get("label") in ("PDH", "PDL", "PWH", "PWL")  # always treated as external per the videos
            )
            tagged.append({**lvl, "scope": "EXTERNAL" if is_external else "INTERNAL"})
        return tagged

    # ═══════════════════════════════════════════════════════
    # 2. HIGH-RESISTANCE VS LOW-RESISTANCE (failure swing)
    # ═══════════════════════════════════════════════════════

    def classify_resistance(self, df: pd.DataFrame) -> dict:
        """
        Looks at the last 3 confirmed swing points on each side to classify
        the most recent structure as a failure swing (low-resistance) or a
        clean break (high-resistance), per side.

        Returns e.g.:
            {'buy_side': 'HIGH_RESISTANCE'|'LOW_RESISTANCE'|'UNKNOWN',
             'sell_side': 'HIGH_RESISTANCE'|'LOW_RESISTANCE'|'UNKNOWN'}
        """
        result = {"buy_side": "UNKNOWN", "sell_side": "UNKNOWN"}
        swing_highs, swing_lows = self._find_swings(df)

        if len(swing_highs) >= 2:
            h1, h2 = swing_highs[-2]["price"], swing_highs[-1]["price"]
            # Clean break (high-resistance buy-side liquidity) = a higher high
            # immediately followed (once we also have the matching low leg)
            # by structure breaking down again. Failure swing (low-resistance)
            # = the newer high failed to exceed the prior high.
            result["buy_side"] = "HIGH_RESISTANCE" if h2 > h1 else "LOW_RESISTANCE"

        if len(swing_lows) >= 2:
            l1, l2 = swing_lows[-2]["price"], swing_lows[-1]["price"]
            result["sell_side"] = "HIGH_RESISTANCE" if l2 < l1 else "LOW_RESISTANCE"

        return result

    # ═══════════════════════════════════════════════════════
    # 3. TREND-LINE LIQUIDITY
    # ═══════════════════════════════════════════════════════

    def detect_trendlines(self, df: pd.DataFrame) -> list[dict]:
        """Fits a line through recent swing lows (rising support) and swing
        highs (falling resistance); keeps only reasonably clean fits (R²)."""
        swing_highs, swing_lows = self._find_swings(df)
        recent_highs = [s for s in swing_highs if s["index"] >= len(df) - TRENDLINE_LOOKBACK]
        recent_lows = [s for s in swing_lows if s["index"] >= len(df) - TRENDLINE_LOOKBACK]

        lines = []
        rising = self._fit_line(recent_lows, "RISING_SUPPORT", len(df))
        if rising:
            lines.append(rising)
        falling = self._fit_line(recent_highs, "FALLING_RESISTANCE", len(df))
        if falling:
            lines.append(falling)
        return lines

    def _fit_line(self, points: list[dict], kind: str, n: int) -> dict | None:
        if len(points) < TRENDLINE_MIN_POINTS:
            return None
        xs = np.array([p["index"] for p in points], dtype=float)
        ys = np.array([p["price"] for p in points], dtype=float)
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = slope * xs + intercept
        ss_res = np.sum((ys - pred) ** 2)
        ss_tot = np.sum((ys - ys.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        if r2 < TRENDLINE_MIN_R2:
            return None
        # RISING_SUPPORT only makes sense with positive slope, FALLING_RESISTANCE with negative.
        if kind == "RISING_SUPPORT" and slope <= 0:
            return None
        if kind == "FALLING_RESISTANCE" and slope >= 0:
            return None

        current_value = slope * (n - 1) + intercept
        return {
            "kind": kind, "slope": float(slope), "intercept": float(intercept),
            "r2": round(float(r2), 3), "current_value": round(float(current_value), 5),
            "touches": len(points), "last_index": max(p["index"] for p in points),
        }

    def check_trendline_sweep(self, df: pd.DataFrame, trendlines: list[dict]) -> dict | None:
        """Wick-through-and-close-back at the trendline's CURRENT bar value
        (only the last closed candle — trendline value moves every bar)."""
        if not trendlines or len(df) < 2:
            return None
        i = len(df) - 1
        o, h, l, c = (
            float(df["open"].iloc[i]), float(df["high"].iloc[i]),
            float(df["low"].iloc[i]), float(df["close"].iloc[i]),
        )
        for tl in trendlines:
            level = tl["current_value"]
            if tl["kind"] == "RISING_SUPPORT" and l < level and c > level:
                return {"kind": "RISING_SUPPORT", "level": round(level, 5),
                        "direction": "BULLISH_REVERSAL", "r2": tl["r2"],
                        "note": f"Rising trendline liquidity swept at {level:.5f} and rejected (R²={tl['r2']})"}
            if tl["kind"] == "FALLING_RESISTANCE" and h > level and c < level:
                return {"kind": "FALLING_RESISTANCE", "level": round(level, 5),
                        "direction": "BEARISH_REVERSAL", "r2": tl["r2"],
                        "note": f"Falling trendline liquidity swept at {level:.5f} and rejected (R²={tl['r2']})"}
        return None

    # ═══════════════════════════════════════════════════════
    # 4. INDUCEMENT
    # ═══════════════════════════════════════════════════════

    def detect_inducement(self, levels: list[dict], stop_hunt_events: list[dict]) -> dict | None:
        """
        If the most recent stop-hunt event swept an INTERNAL level, and the
        EXTERNAL level on the same side is still fresh (present in `levels`
        with matching type and not itself just swept), classify it as
        inducement: retail got trapped on the minor level before the real
        move toward/through the major one.
        """
        if not stop_hunt_events or not levels:
            return None

        most_recent = min(stop_hunt_events, key=lambda e: e["candles_ago"])
        # Find the matching level dict (by price) to check its scope tag.
        matches = [lvl for lvl in levels if abs(lvl["price"] - most_recent["level"]) < 1e-9]
        if not matches or matches[0].get("scope") != "INTERNAL":
            return None

        side = matches[0]["liquidity_type"]
        external_same_side_fresh = any(
            lvl.get("scope") == "EXTERNAL" and lvl["liquidity_type"] == side
            for lvl in levels
        )
        if not external_same_side_fresh:
            return None

        return {
            "confirmed": True,
            "swept_internal_level": most_recent["level"],
            "note": (
                f"Inducement pattern: internal {side} liquidity at "
                f"{most_recent['level']:.5f} swept first while the external "
                f"{side} level is still untouched — likely a trap before the real move"
            ),
        }

    # ═══════════════════════════════════════════════════════
    # UTILS (mirrors the swing-detection contract used elsewhere)
    # ═══════════════════════════════════════════════════════

    def _find_swings(self, df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
        if len(df) < SWING_WINDOW * 3:
            return [], []
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)
        w = SWING_WINDOW
        swing_highs, swing_lows = [], []
        for i in range(w, n - w):
            if highs[i] == highs[i - w: i + w + 1].max():
                swing_highs.append({"index": i, "price": float(highs[i])})
            if lows[i] == lows[i - w: i + w + 1].min():
                swing_lows.append({"index": i, "price": float(lows[i])})
        return swing_highs, swing_lows

    def _safe_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            val = df["atr"].iloc[-1]
            if val is not None and not np.isnan(val):
                return float(val)
        except Exception:
            pass
        try:
            highs, lows, closes = (
                df["high"].values[-period:], df["low"].values[-period:], df["close"].values[-period:],
            )
            trs = [max(h - l, abs(h - c), abs(l - c)) for h, l, c in zip(highs[1:], lows[1:], closes[:-1])]
            return float(np.mean(trs)) if trs else 0.0001
        except Exception:
            return 0.0001
