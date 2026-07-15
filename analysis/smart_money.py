# analysis/smart_money.py  —  Day 61 | Smart Money Concepts Master Engine
# ============================================================
# এই module Day 61-এর সব building block একসাথে জোড়া দেয়:
#
#   structure.py      -> Market Structure (HH/HL/LH/LL, BOS, CHoCH, Displacement)
#   liquidity.py       -> Liquidity Pools, Sweep, Premium/Discount
#   order_block.py     -> (Day 44, reused) Order Block zones
#   fvg_detector.py    -> (Day 44, reused) Fair Value Gap zones
#
# এবং doc-এ চাওয়া অতিরিক্ত 10/10 feature যোগ করে:
#   ⭐ Multi-Timeframe SMC   (H4 bias -> H1 structure -> M15 entry)
#   ⭐ Liquidity Sweep        (liquidity.py থেকে)
#   ⭐ Displacement           (structure.py থেকে)
#   ⭐ Kill Zone Analysis     (London/NY/London-Close session tracking)
#   ⭐ SMC Confidence Score   (BOS+OB+FVG+Sweep+HTF bias = 100)
#
# Single-timeframe quick call: SmartMoneyEngine.analyze_single(df)
# Full MTF pipeline:           SmartMoneyEngine.analyze(symbol)
# ============================================================

from datetime import datetime, timezone

import pandas as pd
from utils.logger import get_logger

from analysis.structure import MarketStructureEngine
from analysis.liquidity import LiquidityEngine
from analysis.order_block import OrderBlockDetector
from analysis.fvg_detector import FVGDetector

log = get_logger("smart_money")

# ── Confidence score weights (total = 100) ──────────────────
SMC_WEIGHTS = {
    "bos":              25,
    "order_block":      25,
    "fvg":              20,
    "liquidity_sweep":  20,
    "htf_bias":         10,
}

# ── Kill zones (UTC hours) — ICT session concept ────────────
KILL_ZONES = {
    "LONDON_OPEN":    (7, 10),    # 07:00-10:00 UTC
    "NEW_YORK_OPEN":  (12, 15),   # 12:00-15:00 UTC
    "LONDON_CLOSE":   (15, 17),   # 15:00-17:00 UTC
}

# ── Per-timeframe swing window (passed to structure/liquidity engines) ──
TF_SWING_WINDOW = {
    "5m": 3, "15m": 5, "30m": 6, "1h": 7, "4h": 10, "1d": 14,
}


class SmartMoneyEngine:
    """
    Usage (single timeframe — fast, for use inside another pipeline):
        sm = SmartMoneyEngine()
        result = sm.analyze_single(df, timeframe="1h")
        ctx = sm.get_ai_context(result)

    Usage (full multi-timeframe SMC, professional top-down):
        sm = SmartMoneyEngine(symbol="EURUSD")
        result = sm.analyze()                 # D1 bias -> H4 structure -> H1 OB -> M15 entry
        sm.print_summary(result)
        ctx = sm.get_ai_context(result)
    """

    def __init__(self, symbol: str = "EURUSD"):
        self.symbol = symbol
        self.ob_detector  = OrderBlockDetector()
        self.fvg_detector = FVGDetector()

    # ═══════════════════════════════════════════════════════
    # SINGLE-TIMEFRAME ANALYSIS
    # ═══════════════════════════════════════════════════════

    def analyze_single(self, df: pd.DataFrame, timeframe: str = "1h") -> dict:
        """
        একটা single timeframe-এর জন্য সম্পূর্ণ SMC snapshot।
        MTF pipeline না চালিয়ে এটা সরাসরি AnalysisAgent-এ inject করা যায়।
        """
        if df is None or len(df) < 30 or "atr" not in df.columns:
            return self._empty_result("Insufficient data or missing ATR column")

        swing_window = TF_SWING_WINDOW.get(timeframe, 5)

        structure_engine = MarketStructureEngine(swing_window=swing_window)
        liquidity_engine  = LiquidityEngine(swing_window=swing_window)

        structure_result = structure_engine.analyze(df)
        liquidity_result = liquidity_engine.analyze(df)

        current_price = float(df["close"].iloc[-1])
        atr           = float(df["atr"].iloc[-1]) if not df["atr"].isna().iloc[-1] else None

        order_blocks = self.ob_detector.detect(df)
        fvgs         = self.fvg_detector.detect(df)

        nearest_ob  = self.ob_detector.nearest_active(order_blocks, current_price, atr=atr)
        nearest_fvg = self.fvg_detector.nearest_active(fvgs, current_price, atr=atr)

        kill_zone = self._current_kill_zone()

        score, factors, direction = self._score_confluence(
            structure_result, liquidity_result, nearest_ob, nearest_fvg, htf_bias=None
        )

        bias = direction if score >= 50 else "NEUTRAL"

        result = {
            "valid":            True,
            "symbol":           self.symbol,
            "timeframe":        timeframe,
            "current_price":    current_price,
            "structure":        structure_result,
            "liquidity":        liquidity_result,
            "order_blocks":     order_blocks,
            "fvgs":             fvgs,
            "nearest_ob":       nearest_ob,
            "nearest_fvg":      nearest_fvg,
            "kill_zone":        kill_zone,
            "confidence_score": score,
            "confidence_factors": factors,
            "bias":             bias,
        }

        log.info(
            f"[SmartMoney:{timeframe}] {self.symbol} | Bias={bias} | "
            f"Score={score}/100 | Structure={structure_result.get('structure')} | "
            f"KillZone={kill_zone['active']}"
        )
        return result

    # ═══════════════════════════════════════════════════════
    # ⭐ MULTI-TIMEFRAME SMC PIPELINE
    # ═══════════════════════════════════════════════════════

    def analyze(self) -> dict:
        """
        Professional top-down SMC workflow:
            D1  -> Overall bias
            H4  -> Structure (BOS/CHoCH)
            H1  -> Order Block + FVG zones
            M15 -> Entry timing (liquidity sweep + structure confirmation)

        Needs DataFetcher/Indicators — imported lazily so this module
        also works standalone (analyze_single) without network access.
        """
        from data.fetcher import DataFetcher
        from data.indicators import Indicators

        fetcher = DataFetcher()
        ind     = Indicators()

        tf_map = {"D1": "1d", "H4": "4h", "H1": "1h", "M15": "15m"}
        dfs = {}
        for label, code in tf_map.items():
            df = fetcher.fetch_ohlcv(self.symbol, code, limit=200)
            if df is None or df.empty:
                log.warning(f"[SmartMoney] Could not fetch {label} ({code})")
                continue
            dfs[label] = ind.add_all(df)

        if "H4" not in dfs or "M15" not in dfs:
            return self._empty_result("Missing required H4/M15 data for MTF SMC")

        # ── D1: Overall directional bias ───────────────────
        d1_bias = "NEUTRAL"
        if "D1" in dfs:
            d1_structure = MarketStructureEngine(swing_window=TF_SWING_WINDOW["1d"]).analyze(dfs["D1"])
            d1_bias = d1_structure.get("structure", "NEUTRAL") if d1_structure.get("valid") else "NEUTRAL"

        # ── H4: Structure (BOS/CHoCH) ───────────────────────
        h4_structure_engine = MarketStructureEngine(swing_window=TF_SWING_WINDOW["4h"])
        h4_liquidity_engine = LiquidityEngine(swing_window=TF_SWING_WINDOW["4h"])
        h4_structure = h4_structure_engine.analyze(dfs["H4"])
        h4_liquidity = h4_liquidity_engine.analyze(dfs["H4"])

        h4_price = float(dfs["H4"]["close"].iloc[-1])
        h4_atr   = float(dfs["H4"]["atr"].iloc[-1]) if not dfs["H4"]["atr"].isna().iloc[-1] else None

        # ── H1: Order Block + FVG zones ─────────────────────
        h1_obs, h1_fvgs, h1_nearest_ob, h1_nearest_fvg = {}, {}, None, None
        if "H1" in dfs:
            h1_price = float(dfs["H1"]["close"].iloc[-1])
            h1_atr   = float(dfs["H1"]["atr"].iloc[-1]) if not dfs["H1"]["atr"].isna().iloc[-1] else None
            h1_obs  = self.ob_detector.detect(dfs["H1"])
            h1_fvgs = self.fvg_detector.detect(dfs["H1"])
            h1_nearest_ob  = self.ob_detector.nearest_active(h1_obs, h1_price, atr=h1_atr)
            h1_nearest_fvg = self.fvg_detector.nearest_active(h1_fvgs, h1_price, atr=h1_atr)
        else:
            # fallback to H4 zones if H1 unavailable
            h1_obs  = self.ob_detector.detect(dfs["H4"])
            h1_fvgs = self.fvg_detector.detect(dfs["H4"])
            h1_nearest_ob  = self.ob_detector.nearest_active(h1_obs, h4_price, atr=h4_atr)
            h1_nearest_fvg = self.fvg_detector.nearest_active(h1_fvgs, h4_price, atr=h4_atr)

        # ── M15: Entry timing (liquidity sweep + structure) ─
        m15_structure_engine = MarketStructureEngine(swing_window=TF_SWING_WINDOW["15m"])
        m15_liquidity_engine  = LiquidityEngine(swing_window=TF_SWING_WINDOW["15m"])
        m15_structure = m15_structure_engine.analyze(dfs["M15"])
        m15_liquidity = m15_liquidity_engine.analyze(dfs["M15"])

        kill_zone = self._current_kill_zone()

        # ── Confluence scoring (H4 structure/liquidity drive bias,
        #    H1 zones add confluence, M15 confirms entry timing) ──
        score, factors, direction = self._score_confluence(
            h4_structure, h4_liquidity, h1_nearest_ob, h1_nearest_fvg,
            htf_bias=d1_bias,
        )

        # M15 entry confirmation bonus/penalty
        m15_aligned = (
            (direction == "BUY" and m15_structure.get("structure") == "BULLISH") or
            (direction == "SELL" and m15_structure.get("structure") == "BEARISH")
        )
        if m15_aligned:
            score = min(100, score + 5)
        signal = direction if score >= 60 and direction != "NEUTRAL" else "WAIT"

        result = {
            "valid":   True,
            "symbol":  self.symbol,
            "current_price": h4_price,
            "d1_bias": d1_bias,
            "h4": {
                "structure":   h4_structure,
                "liquidity":   h4_liquidity,
            },
            "h1": {
                "order_blocks": h1_obs,
                "fvgs":         h1_fvgs,
                "nearest_ob":   h1_nearest_ob,
                "nearest_fvg":  h1_nearest_fvg,
            },
            "m15": {
                "structure": m15_structure,
                "liquidity": m15_liquidity,
                "aligned":   m15_aligned,
            },
            "kill_zone":         kill_zone,
            "confluence_score":  score,
            "confluence_factors": factors,
            "direction":         direction,
            "signal":            signal,
            "explanation": self._build_explanation(
                d1_bias, h4_structure, h4_liquidity, h1_nearest_ob,
                h1_nearest_fvg, factors, signal,
            ),
        }

        log.info(
            f"[SmartMoney:MTF] {self.symbol} | Signal={signal} | Direction={direction} | "
            f"Score={score}/100 | D1={d1_bias} | H4={h4_structure.get('structure')} | "
            f"KillZone={kill_zone['active']}"
        )
        return result

    # ═══════════════════════════════════════════════════════
    # ⭐ SMC CONFIDENCE SCORE  (BOS25 + OB25 + FVG20 + Sweep20 + HTF10)
    # ═══════════════════════════════════════════════════════

    def _score_confluence(
        self, structure_result: dict, liquidity_result: dict,
        nearest_ob: dict | None, nearest_fvg: dict | None,
        htf_bias: str | None,
    ) -> tuple[int, dict, str]:

        score = 0
        factors = {
            "bos": False, "order_block": False, "fvg": False,
            "liquidity_sweep": False, "htf_bias": False,
        }
        # FIX (institutional review, Finding H-1): direction used to be
        # decided by unweighted vote-counting (each factor = 1 vote),
        # independent of the weighted `score` computed below. That let
        # direction and score contradict each other — e.g. a 25-point
        # bullish BOS + 10-point bullish HTF bias (2 votes) could be
        # outvoted by OB+FVG+sweep bearish (3 votes, 65 points), reporting
        # SELL with a high score even though the vote tally and the score
        # disagreed on which side actually had more evidence weight.
        # bull_weight/bear_weight now accumulate the SAME SMC_WEIGHTS used
        # for `score`, so direction is derived from the identical weighted
        # evidence — the two can no longer disagree.
        bull_weight, bear_weight = 0, 0

        # BOS
        bos = structure_result.get("bos", {}) if structure_result.get("valid") else {}
        if bos.get("event") == "BULLISH_BOS":
            factors["bos"] = True
            score += SMC_WEIGHTS["bos"]
            bull_weight += SMC_WEIGHTS["bos"]
        elif bos.get("event") == "BEARISH_BOS":
            factors["bos"] = True
            score += SMC_WEIGHTS["bos"]
            bear_weight += SMC_WEIGHTS["bos"]

        # Order Block (active/near)
        if nearest_ob and nearest_ob.get("in_zone"):
            factors["order_block"] = True
            score += SMC_WEIGHTS["order_block"]
            if nearest_ob["direction"] == "BULLISH":
                bull_weight += SMC_WEIGHTS["order_block"]
            else:
                bear_weight += SMC_WEIGHTS["order_block"]

        # FVG (active/near)
        if nearest_fvg and nearest_fvg.get("in_zone"):
            factors["fvg"] = True
            score += SMC_WEIGHTS["fvg"]
            if nearest_fvg["direction"] == "BULLISH":
                bull_weight += SMC_WEIGHTS["fvg"]
            else:
                bear_weight += SMC_WEIGHTS["fvg"]

        # Liquidity sweep
        sweep = liquidity_result.get("recent_sweep") if liquidity_result.get("valid") else None
        if sweep:
            factors["liquidity_sweep"] = True
            score += SMC_WEIGHTS["liquidity_sweep"]
            if sweep["implication"] == "BULLISH_REVERSAL_LIKELY":
                bull_weight += SMC_WEIGHTS["liquidity_sweep"]
            else:
                bear_weight += SMC_WEIGHTS["liquidity_sweep"]

        # HTF (D1) bias agreement
        if htf_bias in ("BULLISH", "BEARISH"):
            factors["htf_bias"] = True
            score += SMC_WEIGHTS["htf_bias"]
            if htf_bias == "BULLISH":
                bull_weight += SMC_WEIGHTS["htf_bias"]
            else:
                bear_weight += SMC_WEIGHTS["htf_bias"]

        if bull_weight > bear_weight:
            direction = "BUY"
        elif bear_weight > bull_weight:
            direction = "SELL"
        else:
            direction = "NEUTRAL"

        return min(100, score), factors, direction

    # ═══════════════════════════════════════════════════════
    # ⭐ KILL ZONE ANALYSIS
    # ═══════════════════════════════════════════════════════

    def _current_kill_zone(self) -> dict:
        """
        বর্তমান UTC সময় কোন ICT kill zone-এর মধ্যে পড়ে কিনা বলো।
        SMC setup এই session-গুলোতে বেশি reliable ধরা হয়।
        """
        now = datetime.now(timezone.utc)
        hour = now.hour

        for name, (start, end) in KILL_ZONES.items():
            if start <= hour < end:
                return {
                    "active": True,
                    "zone":   name,
                    "utc_hour": hour,
                    "note": f"Inside {name.replace('_', ' ').title()} kill zone — higher SMC reliability",
                }

        return {
            "active": False,
            "zone":   "NONE",
            "utc_hour": hour,
            "note": "Outside major kill zones — SMC setups less reliable",
        }

    # ═══════════════════════════════════════════════════════
    # EXPLANATION BUILDER
    # ═══════════════════════════════════════════════════════

    def _build_explanation(
        self, d1_bias, h4_structure, h4_liquidity,
        nearest_ob, nearest_fvg, factors, signal,
    ) -> str:
        parts = []

        if d1_bias != "NEUTRAL":
            parts.append(f"D1 overall bias is {d1_bias}")

        if factors.get("bos"):
            bos = h4_structure.get("bos", {})
            parts.append(f"H4 structure shows {bos.get('event')}")

        if factors.get("liquidity_sweep"):
            sweep = h4_liquidity.get("recent_sweep", {})
            parts.append(sweep.get("note", "Liquidity sweep detected"))

        if factors.get("order_block") and nearest_ob:
            parts.append(
                f"Price sitting in {nearest_ob['direction'].lower()} order block "
                f"({nearest_ob['zone_bottom']}-{nearest_ob['zone_top']})"
            )

        if factors.get("fvg") and nearest_fvg:
            parts.append(
                f"Price reacting to {nearest_fvg['direction'].lower()} FVG "
                f"({nearest_fvg['zone_bottom']}-{nearest_fvg['zone_top']})"
            )

        if not parts:
            return "No significant institutional footprint detected — WAIT."

        prefix = f"[{signal}] " if signal != "WAIT" else "[WAIT] "
        return prefix + ". ".join(parts) + "."

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT
    # ═══════════════════════════════════════════════════════

    def get_ai_context(self, result: dict) -> dict:
        if not result.get("valid"):
            return {
                "smc_valid":   False,
                "smc_signal":  "WAIT",
                "smc_direction": "NEUTRAL",
                "smc_score":   0,
            }

        # Single-timeframe shape
        if "bias" in result:
            structure = result.get("structure", {})
            liquidity = result.get("liquidity", {})
            nearest_ob  = result.get("nearest_ob")
            nearest_fvg = result.get("nearest_fvg")
            kill_zone   = result.get("kill_zone", {})

            return {
                "smc_valid":      True,
                "smc_bias":       result.get("bias"),
                "smc_score":      result.get("confidence_score", 0),
                "smc_factors":    result.get("confidence_factors", {}),
                "smc_structure":  structure.get("structure", "NEUTRAL"),
                "smc_bos":        structure.get("bos", {}).get("event", "NONE"),
                "smc_choch":      structure.get("choch", {}).get("event", "NONE"),
                "smc_displacement": structure.get("displacement", {}).get("detected", False),
                "smc_liquidity_zone": liquidity.get("premium_discount", {}).get("zone", "UNKNOWN"),
                "smc_ob_zone": (
                    f"{nearest_ob['zone_bottom']}-{nearest_ob['zone_top']}" if nearest_ob else None
                ),
                "smc_fvg_zone": (
                    f"{nearest_fvg['zone_bottom']}-{nearest_fvg['zone_top']}" if nearest_fvg else None
                ),
                "smc_kill_zone_active": kill_zone.get("active", False),
                "smc_kill_zone_name":   kill_zone.get("zone", "NONE"),
            }

        # Multi-timeframe shape
        h4 = result.get("h4", {})
        h1 = result.get("h1", {})
        m15 = result.get("m15", {})
        kill_zone = result.get("kill_zone", {})

        h4_structure = h4.get("structure", {})
        h4_liquidity = h4.get("liquidity", {})
        nearest_ob   = h1.get("nearest_ob")
        nearest_fvg  = h1.get("nearest_fvg")

        return {
            "smc_valid":       True,
            "smc_signal":      result.get("signal", "WAIT"),
            "smc_direction":   result.get("direction", "NEUTRAL"),
            "smc_score":       result.get("confluence_score", 0),
            "smc_factors":     result.get("confluence_factors", {}),
            "smc_d1_bias":     result.get("d1_bias", "NEUTRAL"),
            "smc_h4_structure": h4_structure.get("structure", "NEUTRAL"),
            "smc_h4_bos":      h4_structure.get("bos", {}).get("event", "NONE"),
            "smc_h4_choch":    h4_structure.get("choch", {}).get("event", "NONE"),
            "smc_h4_displacement": h4_structure.get("displacement", {}).get("detected", False),
            "smc_h4_liquidity_zone": h4_liquidity.get("premium_discount", {}).get("zone", "UNKNOWN"),
            "smc_h1_ob_zone": (
                f"{nearest_ob['zone_bottom']}-{nearest_ob['zone_top']}" if nearest_ob else None
            ),
            "smc_h1_fvg_zone": (
                f"{nearest_fvg['zone_bottom']}-{nearest_fvg['zone_top']}" if nearest_fvg else None
            ),
            "smc_m15_aligned": m15.get("aligned", False),
            "smc_kill_zone_active": kill_zone.get("active", False),
            "smc_kill_zone_name":   kill_zone.get("zone", "NONE"),
            "smc_explanation":      result.get("explanation", ""),
        }

    # ═══════════════════════════════════════════════════════
    # FALLBACK
    # ═══════════════════════════════════════════════════════

    def _empty_result(self, reason: str) -> dict:
        return {
            "valid": False, "reason": reason, "symbol": self.symbol,
            "confluence_score": 0, "confidence_score": 0,
            "direction": "NEUTRAL", "bias": "NEUTRAL", "signal": "WAIT",
        }

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, result: dict) -> None:
        bar = "═" * 58
        log.info(bar)
        log.info("  🧠  SMART MONEY ENGINE  (Day 61)")
        log.info(bar)

        if not result.get("valid"):
            log.info(f"  ⚠️  {result.get('reason')}")
            log.info(bar)
            return

        # Single-TF
        if "bias" in result:
            log.info(f"  Pair/TF      : {result['symbol']} {result['timeframe']}")
            log.info(f"  Bias         : {result['bias']}")
            log.info(f"  Score        : {result['confidence_score']}/100")
            kz = result["kill_zone"]
            log.info(f"  Kill Zone    : {'✅ ' + kz['zone'] if kz['active'] else '❌ none'}")
            log.info("")
            for name, weight in SMC_WEIGHTS.items():
                mark = "✅" if result["confidence_factors"].get(name) else "❌"
                log.info(f"  {mark} {name:<18} (+{weight})")
            log.info(bar)
            return

        # MTF
        log.info(f"  Pair         : {result['symbol']}")
        log.info(f"  Signal       : {result['signal']}  ({result['direction']})")
        log.info(f"  Score        : {result['confluence_score']}/100")
        log.info(f"  D1 Bias      : {result['d1_bias']}")
        log.info(f"  H4 Structure : {result['h4']['structure'].get('structure')}")
        kz = result["kill_zone"]
        log.info(f"  Kill Zone    : {'✅ ' + kz['zone'] if kz['active'] else '❌ none'}")
        log.info("")
        for name, weight in SMC_WEIGHTS.items():
            mark = "✅" if result["confluence_factors"].get(name) else "❌"
            log.info(f"  {mark} {name:<18} (+{weight})")
        log.info("")
        log.info(f"  Explanation  : {result['explanation']}")
        log.info(bar)


# ═══════════════════════════════════════════════════════════
# QUICK RUN — Direct test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    sm = SmartMoneyEngine(symbol="EURUSD")
    result = sm.analyze()
    sm.print_summary(result)

    ctx = sm.get_ai_context(result)
    print("\nAI Context (for MasterAnalyst):")
    for k, v in ctx.items():
        print(f"  {k:<28}: {v}")