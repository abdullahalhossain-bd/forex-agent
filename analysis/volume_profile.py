# analysis/volume_profile.py  —  Day 86 | Volume Profile Engine
# ============================================================
# Forex-এ real volume নেই, কিন্তু tick volume আছে যেটা
# real volume এর সাথে 90%+ correlated। এই module সেটা ব্যবহার করে।
#
# Volume Profile মানে: price level অনুযায়ী volume distribute করা।
# কোন price level-এ সবচেয়ে বেশি trading হয়েছে সেটা দেখায়।
#
# ৩টা key level detect করে:
#
#   1. POC (Point of Control)
#      সবচেয়ে বেশি volume যে price level-এ traded হয়েছে।
#      এটা "fair value" — price এখানে ফিরে আসার প্রবণতা রাখে।
#
#   2. HVN (High Volume Node)
#      উঁচু volume এর অঞ্চল — strong S/R zones।
#      Price এখানে অনেকক্ষণ stuck থাকে, slow movement।
#
#   3. LVN (Low Volume Node)
#      কম volume এর অঞ্চল — price এখান দিয়ে দ্রুত যায়।
#      "Vacuum" — যেহেতু কম interest, পরের S/R এ দ্রুত reach করে।
#
# Output:
#   {
#     "poc":            float,    # Point of Control price
#     "poc_volume":     float,
#     "hvn_zones":      [{"price_low", "price_high", "volume", "strength"}, ...],
#     "lvn_zones":      [...],
#     "value_area_high": float,   # 70% volume zone upper
#     "value_area_low":  float,   # 70% volume zone lower
#     "price_position": "ABOVE_VA"|"INSIDE_VA"|"BELOW_VA"|"AT_POC",
#     "bias":           "BULLISH"|"BEARISH"|"NEUTRAL",
#     "signal":         "BUY"|"SELL"|"WAIT",
#     "note":           str
#   }
# ============================================================

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("volume_profile_engine")


class VolumeProfileEngine:
    """
    Price-binned volume distribution analyzer।

    Usage:
        engine = VolumeProfileEngine(num_bins=50)
        result = engine.analyze(df, volume_col='tick_volume')
    """

    def __init__(
        self,
        num_bins:          int = 50,
        value_area_pct:    float = 0.70,    # 70% Value Area
        hvn_percentile:    float = 75.0,    # top 25% bins = HVN
        lvn_percentile:    float = 25.0,    # bottom 25% bins = LVN
        min_zone_width:    int = 1,         # adjacent bins merge
    ):
        self.num_bins          = num_bins
        self.value_area_pct    = value_area_pct
        self.hvn_percentile    = hvn_percentile
        self.lvn_percentile    = lvn_percentile
        self.min_zone_width    = min_zone_width

    # ═══════════════════════════════════════════════════════
    # MAIN ENTRY
    # ═══════════════════════════════════════════════════════

    def analyze(
        self,
        df:         pd.DataFrame,
        volume_col: str = "tick_volume",
    ) -> Dict[str, Any]:
        """
        df-এ high/low/close + volume_col লাগবে।
        যদি tick_volume না থাকে, 'volume' column খোঁজে।
        """
        if df is None or len(df) < 30:
            return self._empty_result("Insufficient data")

        # Volume column detect
        if volume_col not in df.columns:
            for alt in ("volume", "tick_volume", "tickvol"):
                if alt in df.columns:
                    volume_col = alt
                    break
            else:
                # Fall back to uniform volume = 1 (still works, just less accurate)
                df = df.copy()
                df["volume"] = 1.0
                volume_col = "volume"

        df = df.copy()
        if volume_col not in df.columns:
            return self._empty_result("No volume column available")

        # ── Step 1: Build price-volume distribution ──
        profile = self._build_profile(df, volume_col)
        if profile is None or len(profile) == 0:
            return self._empty_result("Could not build volume profile")

        # ── Step 2: POC ──
        poc_idx = profile["volume"].idxmax()
        poc_price = float(profile.loc[poc_idx, "price_mid"])
        poc_volume = float(profile.loc[poc_idx, "volume"])

        # ── Step 3: Value Area (70%) ──
        va_high, va_low = self._value_area(profile, poc_idx)

        # ── Step 4: HVN / LVN zones ──
        hvn_zones = self._find_zones(profile, "high")
        lvn_zones = self._find_zones(profile, "low")

        # ── Step 5: Price position ──
        close = float(df["close"].iloc[-1])
        price_position = self._price_position(close, poc_price, va_high, va_low)

        # ── Step 6: Bias ──
        bias = self._bias(close, poc_price, va_high, va_low, price_position)

        # ── Step 7: Signal ──
        signal, note = self._signal(bias, price_position, close, poc_price, va_high, va_low)

        result = {
            "valid":           True,
            "poc":             round(poc_price, 5),
            "poc_volume":      round(poc_volume, 0),
            "hvn_zones":       hvn_zones,
            "lvn_zones":       lvn_zones,
            "value_area_high": round(va_high, 5),
            "value_area_low":  round(va_low, 5),
            "price_position":  price_position,
            "bias":            bias,
            "signal":          signal,
            "note":            note,
            "close":           round(close, 5),
            "num_bins":        len(profile),
        }

        log.info(
            f"[VolumeProfile] POC={poc_price:.5f} | VA=[{va_low:.5f}-{va_high:.5f}] | "
            f"pos={price_position} | bias={bias} | signal={signal}"
        )
        return result

    # ═══════════════════════════════════════════════════════
    # PROFILE BUILDING
    # ═══════════════════════════════════════════════════════

    def _build_profile(self, df: pd.DataFrame, volume_col: str) -> pd.DataFrame:
        """
        Price range-কে num_bins-এ ভাগ করো, প্রতিটা bin-এ যত candle
        পড়ে তার volume sum করো।
        """
        price_min = float(df[["open", "high", "low", "close"]].min().min())
        price_max = float(df[["open", "high", "low", "close"]].max().max())

        if price_max <= price_min:
            return None

        bin_edges = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_mids  = (bin_edges[:-1] + bin_edges[1:]) / 2

        # প্রতিটা candle-এর volume কোন bin-এ পড়ে সেটা distribute করো।
        # Simplification: use candle body range to assign volume across bins
        # it touches (more accurate than just close).
        volumes = np.zeros(self.num_bins)

        opens   = df["open"].values
        highs   = df["high"].values
        lows    = df["low"].values
        closes  = df["close"].values
        vols    = df[volume_col].fillna(1.0).values

        for i in range(len(df)):
            c_low  = min(opens[i], closes[i], lows[i])
            c_high = max(opens[i], closes[i], highs[i])

            # Find bins this candle's range touches
            for b in range(self.num_bins):
                if c_low <= bin_edges[b + 1] and c_high >= bin_edges[b]:
                    # Approximate: distribute volume equally across touched bins
                    volumes[b] += vols[i]

        profile = pd.DataFrame({
            "price_low":  bin_edges[:-1],
            "price_high": bin_edges[1:],
            "price_mid":  bin_mids,
            "volume":     volumes,
        })
        return profile

    def _value_area(self, profile: pd.DataFrame, poc_idx: int) -> tuple[float, float]:
        """
        Value Area: POC থেকে শুরু করে উপরে ও নিচে বিস্তার করে 70%
        total volume কভার করে এমন price range।
        """
        total_vol = profile["volume"].sum()
        if total_vol == 0:
            return 0.0, 0.0

        target = total_vol * self.value_area_pct
        current_vol = float(profile.loc[poc_idx, "volume"])
        upper_idx = poc_idx
        lower_idx = poc_idx

        while current_vol < target and (upper_idx < len(profile) - 1 or lower_idx > 0):
            # Compare next upper vs next lower, pick higher volume
            upper_vol = float(profile.loc[upper_idx + 1, "volume"]) if upper_idx + 1 < len(profile) else 0
            lower_vol = float(profile.loc[lower_idx - 1, "volume"]) if lower_idx - 1 >= 0 else 0

            if upper_vol >= lower_vol and upper_idx + 1 < len(profile):
                upper_idx += 1
                current_vol += upper_vol
            elif lower_idx - 1 >= 0:
                lower_idx -= 1
                current_vol += lower_vol
            else:
                break

        va_high = float(profile.loc[upper_idx, "price_high"])
        va_low  = float(profile.loc[lower_idx, "price_low"])
        return va_high, va_low

    def _find_zones(self, profile: pd.DataFrame, kind: str) -> List[Dict[str, Any]]:
        """
        HVN (kind='high'): top 25% volume bins
        LVN (kind='low'):  bottom 25% volume bins

        Adjacent qualifying bins merge করা হয়।
        """
        if len(profile) == 0:
            return []

        if kind == "high":
            threshold = float(np.percentile(profile["volume"], self.hvn_percentile))
            mask = profile["volume"] >= threshold
        else:
            threshold = float(np.percentile(profile["volume"], self.lvn_percentile))
            mask = profile["volume"] <= threshold

        zones: List[Dict[str, Any]] = []
        current_zone = None

        for idx, row in profile.iterrows():
            qualifies = bool(mask.loc[idx])
            if qualifies:
                if current_zone is None:
                    current_zone = {
                        "price_low":  float(row["price_low"]),
                        "price_high": float(row["price_high"]),
                        "volume":     float(row["volume"]),
                        "bin_count":  1,
                    }
                else:
                    current_zone["price_high"] = float(row["price_high"])
                    current_zone["volume"]    += float(row["volume"])
                    current_zone["bin_count"] += 1
            else:
                if current_zone is not None and current_zone["bin_count"] >= self.min_zone_width:
                    zones.append(self._format_zone(current_zone, kind, profile))
                current_zone = None

        # Last zone
        if current_zone is not None and current_zone["bin_count"] >= self.min_zone_width:
            zones.append(self._format_zone(current_zone, kind, profile))

        return zones

    def _format_zone(self, zone: dict, kind: str, profile: pd.DataFrame) -> Dict[str, Any]:
        max_vol = float(profile["volume"].max())
        avg_vol = float(profile["volume"].mean())
        if kind == "high":
            strength = round(zone["volume"] / max_vol * 100, 1) if max_vol > 0 else 0
        else:
            strength = round((1 - zone["volume"] / avg_vol) * 100, 1) if avg_vol > 0 else 0

        return {
            "price_low":   round(zone["price_low"], 5),
            "price_high":  round(zone["price_high"], 5),
            "price_mid":   round((zone["price_low"] + zone["price_high"]) / 2, 5),
            "volume":      round(zone["volume"], 0),
            "bin_count":   zone["bin_count"],
            "strength":    max(0, min(100, strength)),
        }

    # ═══════════════════════════════════════════════════════
    # PRICE POSITION + BIAS
    # ═══════════════════════════════════════════════════════

    def _price_position(
        self, close: float, poc: float, va_high: float, va_low: float
    ) -> str:
        if abs(close - poc) / poc < 0.0005:
            return "AT_POC"
        if close > va_high:   return "ABOVE_VA"
        if close < va_low:    return "BELOW_VA"
        return "INSIDE_VA"

    def _bias(
        self, close: float, poc: float, va_high: float, va_low: float, position: str
    ) -> str:
        """
        - Price ABOVE VA → bullish (acceptance above value)
        - Price BELOW VA → bearish
        - INSIDE VA / AT POC → neutral (range-bound)
        """
        if position == "ABOVE_VA":   return "BULLISH"
        if position == "BELOW_VA":   return "BEARISH"
        return "NEUTRAL"

    def _signal(
        self, bias: str, position: str, close: float,
        poc: float, va_high: float, va_low: float,
    ) -> tuple[str, str]:
        if bias == "BULLISH" and position == "ABOVE_VA":
            return "BUY", f"Price above Value Area — bullish acceptance, POC={poc:.5f}"
        if bias == "BEARISH" and position == "BELOW_VA":
            return "SELL", f"Price below Value Area — bearish acceptance, POC={poc:.5f}"
        if position == "AT_POC":
            return "WAIT", f"Price at POC — fair value, range-bound likely"
        return "WAIT", f"Price inside Value Area — wait for VA break"

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT
    # ═════════════════════════════:═════════════════════════
    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not result.get("valid"):
            return {
                "volume_profile_valid":  False,
                "vp_poc":                None,
                "vp_bias":               "NEUTRAL",
                "vp_signal":             "WAIT",
                "vp_price_position":     "UNKNOWN",
            }

        return {
            "volume_profile_valid":  True,
            "vp_poc":                result.get("poc"),
            "vp_value_area_high":    result.get("value_area_high"),
            "vp_value_area_low":     result.get("value_area_low"),
            "vp_price_position":     result.get("price_position"),
            "vp_bias":               result.get("bias"),
            "vp_signal":             result.get("signal"),
            "vp_hvn_count":          len(result.get("hvn_zones", [])),
            "vp_lvn_count":          len(result.get("lvn_zones", [])),
        }

    # ═══════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "valid":           False,
            "reason":          reason,
            "poc":             None,
            "hvn_zones":       [],
            "lvn_zones":       [],
            "value_area_high": None,
            "value_area_low":  None,
            "price_position":  "UNKNOWN",
            "bias":            "NEUTRAL",
            "signal":          "WAIT",
            "note":            reason,
        }

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 56
        log.info(bar)
        log.info("  📊  VOLUME PROFILE  (Day 86)")
        log.info(bar)

        if not result.get("valid"):
            log.info(f"  ⚠️  {result.get('reason', 'No analysis')}")
            log.info(bar)
            return

        log.info(f"  POC            : {result['poc']}  (vol {result['poc_volume']})")
        log.info(f"  Value Area     : {result['value_area_low']}  →  {result['value_area_high']}")
        log.info(f"  Price Position : {result['price_position']}")
        log.info(f"  Bias           : {result['bias']}")
        log.info(f"  Signal         : {result['signal']}")

        hvn = result.get("hvn_zones", [])
        if hvn:
            log.info(f"  HVN Zones ({len(hvn)}):")
            for z in hvn[:3]:
                log.info(f"    • {z['price_low']}-{z['price_high']}  str={z['strength']}")

        lvn = result.get("lvn_zones", [])
        if lvn:
            log.info(f"  LVN Zones ({len(lvn)}):")
            for z in lvn[:3]:
                log.info(f"    • {z['price_low']}-{z['price_high']}  str={z['strength']}")

        log.info(f"  Note           : {result['note']}")
        log.info(bar)


# ═══════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    n = 300
    # Price oscillates with a clear favorite zone around 1.1050
    prices = 1.1000 + np.sin(np.linspace(0, 8*np.pi, n)) * 0.003 + np.random.randn(n) * 0.0005
    # Most volume near middle
    tick_vol = 100 + np.abs(np.sin(np.linspace(0, 8*np.pi, n))) * 500 + np.random.randn(n) * 20

    df = pd.DataFrame({
        "open":        prices,
        "high":        prices + 0.0005,
        "low":         prices - 0.0005,
        "close":       prices,
        "tick_volume": tick_vol,
    })

    engine = VolumeProfileEngine(num_bins=30)
    result = engine.analyze(df)
    engine.print_summary(result)

    ctx = engine.get_ai_context(result)
    print("\nAI Context:")
    for k, v in ctx.items():
        print(f"  {k:<28}: {v}")
