# analysis/liquidity.py  —  Day 61 | Liquidity Analysis Engine
# ============================================================
# Institutional players retail trader-দের stop-loss/pending orders
# ("liquidity") hunt করে এন্ট্রি নেয়। এই engine বুঝবে:
#
#   1. Liquidity Pool Detection  — Equal Highs / Equal Lows
#   2. Liquidity Sweep Detector  — pool নেওয়া হয়েছে কিনা (stop hunt)
#   3. Premium / Discount Zone   — range-এর 50% থেকে কোথায় price আছে
#
# Note: Day 38 mtf_analyzer._detect_liquidity_sweep simple wick-reject
# logic ব্যবহার করে একটা single recent level-এর জন্য। এই module আরও
# গভীরে যায় — multiple equal-high/low pools track করে, এবং প্রতিটা
# pool-এর জন্য আলাদাভাবে sweep status বলে।
# ============================================================

import numpy as np
import pandas as pd
from utils.logger import get_logger

# BUGFIX (institutional review): this logger was previously named
# "liquidity_engine", identical to the logger in analysis/liquidity_engine.py
# (a different module with a same-named LiquidityEngine class — see the audit
# report). That made it impossible to tell, from the log stream alone, which
# of the two modules emitted a given line. Renamed to "liquidity" to match
# this file. Note: the class name collision itself (two distinct
# `LiquidityEngine` classes with different constructors/signatures/return
# schemas in analysis.liquidity and analysis.liquidity_engine) is NOT fixed
# here — renaming a class requires updating every call site across the whole
# project, and only the Decision Layer files were provided for this review.
# See "CRITICAL — Duplicate Class Name" in the audit report.
log = get_logger("liquidity")


class LiquidityEngine:
    """
    Usage:
        engine = LiquidityEngine()
        result = engine.analyze(df)
        ctx    = engine.get_ai_context(result)
    """

    EQUAL_LEVEL_TOLERANCE_ATR = 0.15   # এর মধ্যে দুটো high/low হলে "equal" ধরা হবে
    SWEEP_LOOKAHEAD           = 5      # pool তৈরি হওয়ার কত candle পরে sweep চেক করবে

    def __init__(self, swing_window: int = 5):
        self.swing_window = swing_window

    # ═══════════════════════════════════════════════════════
    # MAIN METHOD
    # ═══════════════════════════════════════════════════════

    def analyze(self, df: pd.DataFrame) -> dict:
        if len(df) < self.swing_window * 4 + 10:
            return self._empty_result("Insufficient data")

        atr = self._atr_value(df)
        swing_highs, swing_lows = self._find_swings(df)

        equal_highs = self._find_equal_levels(swing_highs, atr, kind="high")
        equal_lows  = self._find_equal_levels(swing_lows, atr, kind="low")

        pools = self._build_pools(df, equal_highs, equal_lows, atr)

        current_price = float(df["close"].iloc[-1])
        above, below  = self._nearest_pools(pools, current_price)

        premium_discount = self._premium_discount_zone(df, swing_highs, swing_lows, current_price)

        recent_sweep = self._most_recent_sweep(pools)

        result = {
            "valid":            True,
            "pools":            pools,
            "liquidity_above":  above,
            "liquidity_below":  below,
            "premium_discount": premium_discount,
            "recent_sweep":     recent_sweep,
            "current_price":    current_price,
        }

        log.info(
            f"[Liquidity] Pools={len(pools)} | "
            f"Above={above['price'] if above else None} | "
            f"Below={below['price'] if below else None} | "
            f"Zone={premium_discount['zone']}"
        )
        return result

    # ═══════════════════════════════════════════════════════
    # STEP 1: SWING HIGHS/LOWS (for pool building)
    # ═══════════════════════════════════════════════════════

    def _find_swings(self, df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
        highs = df["high"].values
        lows  = df["low"].values
        n     = len(df)
        w     = self.swing_window

        swing_highs, swing_lows = [], []
        for i in range(w, n - w):
            if highs[i] == highs[i - w: i + w + 1].max():
                swing_highs.append({"index": i, "price": float(highs[i])})
            if lows[i] == lows[i - w: i + w + 1].min():
                swing_lows.append({"index": i, "price": float(lows[i])})

        return swing_highs, swing_lows

    # ═══════════════════════════════════════════════════════
    # STEP 2: LIQUIDITY POOL — EQUAL HIGHS / EQUAL LOWS
    # ═══════════════════════════════════════════════════════

    def _find_equal_levels(self, swings: list[dict], atr: float, kind: str) -> list[dict]:
        """
        কাছাকাছি (ATR tolerance-এর মধ্যে) দুই বা ততোধিক swing point থাকলে
        সেটাকে একটা liquidity pool ধরো — কারণ retail trader-রা এখানে
        equal high/low দেখে stop/pending order রাখে।
        """
        if len(swings) < 2:
            return []

        tolerance = atr * self.EQUAL_LEVEL_TOLERANCE_ATR
        pools = []
        used = set()

        for i in range(len(swings)):
            if i in used:
                continue
            cluster = [swings[i]]
            for j in range(i + 1, len(swings)):
                if j in used:
                    continue
                if abs(swings[j]["price"] - swings[i]["price"]) <= tolerance:
                    cluster.append(swings[j])
                    used.add(j)

            if len(cluster) >= 2:
                avg_price = float(np.mean([c["price"] for c in cluster]))
                pools.append({
                    "kind":    kind,
                    "price":   round(avg_price, 5),
                    "touches": len(cluster),
                    "indices": [c["index"] for c in cluster],
                    "last_index": max(c["index"] for c in cluster),
                })

        return pools

    # ═══════════════════════════════════════════════════════
    # STEP 3: BUILD POOLS + SWEEP STATUS
    # ═══════════════════════════════════════════════════════

    def _build_pools(self, df: pd.DataFrame, equal_highs: list[dict], equal_lows: list[dict], atr: float) -> list[dict]:
        """প্রতিটা pool-এর জন্য sweep হয়েছে কিনা চেক করো এবং merge করো।"""
        all_pools = equal_highs + equal_lows
        n = len(df)
        highs = df["high"].values
        lows  = df["low"].values
        closes = df["close"].values

        enriched = []
        for pool in all_pools:
            sweep = self._check_sweep(pool, highs, lows, closes, n, atr)
            enriched.append({
                **pool,
                "swept":     sweep["swept"],
                "sweep_note": sweep["note"],
                "fresh":     not sweep["swept"],
            })

        enriched.sort(key=lambda p: p["last_index"], reverse=True)
        return enriched

    def _check_sweep(self, pool: dict, highs, lows, closes, n: int, atr: float) -> dict:
        """
        Sweep logic: pool level তৈরি হওয়ার পরে কোনো candle সেই level-এর
        ওপাশে wick করে কিন্তু close ফিরে আসে (stop hunt + rejection)।
        """
        start = pool["last_index"] + 1
        level = pool["price"]
        kind  = pool["kind"]

        for i in range(start, n):
            if kind == "high":
                wicked_above = highs[i] > level
                closed_back  = closes[i] < level
                if wicked_above and closed_back:
                    return {"swept": True, "note": f"Swept at index {i} (wick above, close back below)"}
            else:
                wicked_below = lows[i] < level
                closed_back  = closes[i] > level
                if wicked_below and closed_back:
                    return {"swept": True, "note": f"Swept at index {i} (wick below, close back above)"}

        return {"swept": False, "note": "Not yet swept"}

    # ═══════════════════════════════════════════════════════
    # STEP 4: NEAREST POOLS ABOVE/BELOW CURRENT PRICE
    # ═══════════════════════════════════════════════════════

    def _nearest_pools(self, pools: list[dict], current_price: float) -> tuple[dict | None, dict | None]:
        """Fresh (un-swept) pool গুলোর মধ্যে current price-এর সবচেয়ে কাছের above/below।"""
        fresh = [p for p in pools if p["fresh"]]

        above_candidates = [p for p in fresh if p["price"] > current_price]
        below_candidates = [p for p in fresh if p["price"] < current_price]

        above = min(above_candidates, key=lambda p: p["price"] - current_price) if above_candidates else None
        below = max(below_candidates, key=lambda p: p["price"]) if below_candidates else None

        return above, below

    def _most_recent_sweep(self, pools: list[dict]) -> dict | None:
        swept = [p for p in pools if p["swept"]]
        if not swept:
            return None
        most_recent = max(swept, key=lambda p: p["last_index"])
        direction = "BULLISH_REVERSAL_LIKELY" if most_recent["kind"] == "low" else "BEARISH_REVERSAL_LIKELY"
        return {
            "price":      most_recent["price"],
            "kind":       most_recent["kind"],
            "implication": direction,
            "note": (
                f"Liquidity below {most_recent['price']:.5f} was swept — possible bullish reversal"
                if most_recent["kind"] == "low" else
                f"Liquidity above {most_recent['price']:.5f} was swept — possible bearish reversal"
            ),
        }

    # ═══════════════════════════════════════════════════════
    # STEP 5: PREMIUM / DISCOUNT ZONE
    # ═══════════════════════════════════════════════════════

    def _premium_discount_zone(self, df: pd.DataFrame, swing_highs: list[dict], swing_lows: list[dict], current_price: float) -> dict:
        """
        সাম্প্রতিক range (most recent significant swing high/low)-এর mid-point
        দিয়ে Premium (উপরের অর্ধেক) / Discount (নিচের অর্ধেক) বলো।

        Discount zone -> institutional BUY zone
        Premium zone  -> institutional SELL zone
        """
        if not swing_highs or not swing_lows:
            return {"zone": "UNKNOWN", "mid": None, "entry_preference": "WAIT"}

        recent_high = max(swing_highs[-5:], key=lambda p: p["price"])["price"]
        recent_low  = min(swing_lows[-5:], key=lambda p: p["price"])["price"]

        if recent_high <= recent_low:
            return {"zone": "UNKNOWN", "mid": None, "entry_preference": "WAIT"}

        mid = (recent_high + recent_low) / 2
        ratio = (current_price - recent_low) / (recent_high - recent_low)

        if current_price >= mid:
            zone = "PREMIUM"
            preference = "SELL"
        else:
            zone = "DISCOUNT"
            preference = "BUY"

        # Deep zone tags
        if ratio >= 0.79:
            depth = "DEEP_PREMIUM"
        elif ratio >= 0.5:
            depth = "PREMIUM"
        elif ratio >= 0.21:
            depth = "DISCOUNT"
        else:
            depth = "DEEP_DISCOUNT"

        return {
            "zone":              zone,
            "depth":             depth,
            "mid":               round(mid, 5),
            "range_high":        round(recent_high, 5),
            "range_low":         round(recent_low, 5),
            "ratio":             round(ratio, 4),
            "entry_preference":  preference,
        }

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT
    # ═══════════════════════════════════════════════════════

    def get_ai_context(self, result: dict) -> dict:
        if not result.get("valid"):
            return {
                "liquidity_valid":       False,
                "liquidity_above":       None,
                "liquidity_below":       None,
                "liquidity_zone":        "UNKNOWN",
                "liquidity_entry_pref":  "WAIT",
                "recent_sweep":          None,
                "recent_sweep_implication": None,
            }

        above = result.get("liquidity_above")
        below = result.get("liquidity_below")
        pd_zone = result.get("premium_discount", {})
        sweep   = result.get("recent_sweep")

        return {
            "liquidity_valid":      True,
            "liquidity_above":      above["price"] if above else None,
            "liquidity_above_touches": above["touches"] if above else 0,
            "liquidity_below":      below["price"] if below else None,
            "liquidity_below_touches": below["touches"] if below else 0,
            "liquidity_zone":       pd_zone.get("zone", "UNKNOWN"),
            "liquidity_zone_depth": pd_zone.get("depth", "UNKNOWN"),
            "liquidity_entry_pref": pd_zone.get("entry_preference", "WAIT"),
            "liquidity_range_high": pd_zone.get("range_high"),
            "liquidity_range_low":  pd_zone.get("range_low"),
            "recent_sweep":         sweep["price"] if sweep else None,
            "recent_sweep_kind":    sweep["kind"] if sweep else None,
            "recent_sweep_implication": sweep["implication"] if sweep else None,
        }

    # ═══════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════

    def _atr_value(self, df: pd.DataFrame, period: int = 14) -> float:
        if "atr" in df.columns:
            val = df["atr"].iloc[-1]
            if not np.isnan(val):
                return float(val)
        highs  = df["high"].values[-period:]
        lows   = df["low"].values[-period:]
        closes = df["close"].values[-period:]
        trs = [
            max(h - l, abs(h - c), abs(l - c))
            for h, l, c in zip(highs[1:], lows[1:], closes[:-1])
        ]
        return float(np.mean(trs)) if trs else 0.0001

    def _empty_result(self, reason: str) -> dict:
        return {
            "valid": False, "reason": reason, "pools": [],
            "liquidity_above": None, "liquidity_below": None,
            "premium_discount": {"zone": "UNKNOWN", "entry_preference": "WAIT"},
            "recent_sweep": None, "current_price": None,
        }

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, result: dict) -> None:
        bar = "═" * 56
        log.info(bar)
        log.info("  💧  LIQUIDITY ANALYSIS ENGINE  (Day 61)")
        log.info(bar)

        if not result.get("valid"):
            log.info(f"  ⚠️  {result.get('reason')}")
            log.info(bar)
            return

        above = result["liquidity_above"]
        below = result["liquidity_below"]
        pd_zone = result["premium_discount"]
        sweep   = result["recent_sweep"]

        above_touch_str = f" (touches: {above['touches']})" if above else ""
        below_touch_str = f" (touches: {below['touches']})" if below else ""
        log.info(f"  Current Price  : {result['current_price']:.5f}")
        log.info(f"  Liquidity Above: {above['price'] if above else 'None'}{above_touch_str}")
        log.info(f"  Liquidity Below: {below['price'] if below else 'None'}{below_touch_str}")
        log.info("")
        log.info(f"  Zone           : {pd_zone.get('zone')} ({pd_zone.get('depth')})")
        log.info(f"  Range          : {pd_zone.get('range_low')} - {pd_zone.get('range_high')}")
        log.info(f"  Entry Pref     : {pd_zone.get('entry_preference')}")

        if sweep:
            log.info("")
            log.info(f"  Recent Sweep   : ✅ {sweep['note']}")
        else:
            log.info("")
            log.info("  Recent Sweep   : ❌ None detected")

        log.info(bar)


# ═══════════════════════════════════════════════════════════
# QUICK RUN — Direct test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data.fetcher import DataFetcher
    from data.indicators import Indicators

    fetcher = DataFetcher()
    ind     = Indicators()

    df = fetcher.fetch_ohlcv("EURUSD", "1h", limit=200)
    if df is not None:
        df = ind.add_all(df)

        engine = LiquidityEngine(swing_window=5)
        result = engine.analyze(df)
        engine.print_summary(result)

        ctx = engine.get_ai_context(result)
        print("\nAI Context:")
        for k, v in ctx.items():
            print(f"  {k:<28}: {v}")