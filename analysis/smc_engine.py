# analysis/smc_engine.py  —  Day 44 | Smart Money Concepts (SMC) Engine
# ============================================================
# Combines:
#   H4  → Order Block + FVG + BOS/CHoCH + Liquidity Sweep   (bias + zones)
#   M15 → Liquidity Sweep + BOS + Confirmation Candle        (entry timing)
#
# BOS / CHoCH / Liquidity Sweep detection duplicate করা হয়নি —
# analysis/mtf_analyzer.py (Day 38)-এর _detect_bos / _detect_choch /
# _detect_liquidity_sweep reuse করা হয়েছে (এগুলো pure df-in, dict-out
# helper, self-state ব্যবহার করে না)।
#
# Confluence scoring (doc অনুযায়ী, total 100):
#   Liquidity sweep      +20
#   Order block (active) +25
#   FVG (active)         +15
#   BOS                  +25
#   Confirmation candle  +15
# ============================================================

from data.fetcher import get_data_fetcher
from data.indicators import Indicators
from analysis.order_block import OrderBlockDetector
from analysis.fvg_detector import FVGDetector
from analysis.mtf_analyzer import MTFAnalyzer
from analysis.patterns import PatternDetector
from utils.logger import get_logger

log = get_logger("smc_engine")

SCORE_WEIGHTS = {
    "liquidity_sweep":     20,
    "order_block":         25,
    "fvg":                 15,
    "bos":                 25,
    "confirmation_candle": 15,
}

MIN_TRADE_SCORE = 45   # Soft floor (informational only) — score flows downstream as
                       # confidence rather than being hard-blocked here.
                       # The old code used this as a hard "signal=WAIT" cliff
                       # that discarded all directional evidence below 45.


class SMCEngine:
    """
    Usage:
        smc = SMCEngine("EURUSD")
        result = smc.analyze()
        smc.print_summary(result)
        ctx = smc.get_ai_context(result)   # MasterAnalyst-এ pass করো
    """

    def __init__(self, symbol: str = "EURUSD"):
        self.symbol       = symbol
        self.fetcher      = get_data_fetcher()
        self.ind          = Indicators()
        self.ob_detector  = OrderBlockDetector()
        self.fvg_detector = FVGDetector()
        self.mtf          = MTFAnalyzer(symbol)   # শুধু _detect_bos/_detect_choch/_detect_liquidity_sweep reuse-এর জন্য
        self.pat_detector = PatternDetector()

    # ═══════════════════════════════════════════════════════
    # MAIN METHOD
    # ═══════════════════════════════════════════════════════

    def analyze(self) -> dict:
        h4_df  = self._fetch_with_atr("4h", limit=150)
        m15_df = self._fetch_with_atr("15m", limit=150)

        if h4_df is None or m15_df is None:
            return self._empty_result("Could not fetch H4/M15 data")

        current_price = float(m15_df['close'].iloc[-1])
        m15_atr       = float(m15_df['atr'].iloc[-1]) if not m15_df['atr'].isna().iloc[-1] else None
        h4_atr        = float(h4_df['atr'].iloc[-1]) if not h4_df['atr'].isna().iloc[-1] else None

        # ── H4: Zones + Structure (bias) ──────────────────────
        h4_obs   = self.ob_detector.detect(h4_df)
        h4_fvgs  = self.fvg_detector.detect(h4_df)
        h4_bos   = self.mtf._detect_bos(h4_df)
        h4_choch = self.mtf._detect_choch(h4_df)
        h4_sweep = self.mtf._detect_liquidity_sweep(h4_df)

        nearest_ob  = self.ob_detector.nearest_active(h4_obs, current_price, atr=h4_atr)
        nearest_fvg = self.fvg_detector.nearest_active(h4_fvgs, current_price, atr=h4_atr)

        # ── M15: Entry timing ─────────────────────────────────
        m15_sweep = self.mtf._detect_liquidity_sweep(m15_df)
        m15_bos   = self.mtf._detect_bos(m15_df)

        # ── Round-10 audit fix: deduplicate candlestick pattern detection ──
        # Previously: smc_engine called self.pat_detector.run_full_detection(m15_df)
        # at line 88, which RE-RAN the entire 15+ pattern detection suite
        # (doji, hammer, engulfing, etc.) on the same df that
        # analysis_agent.py:235 had ALREADY detected on.
        #
        # This caused:
        #   1. Duplicate "🕯️ CANDLESTICK PATTERNS" log blocks per cycle
        #   2. ~2× CPU time on pattern detection (15+ functions × 150 bars)
        #   3. Confusing duplicate output in the operator's log
        #
        # Now: we check if the df ALREADY has pattern columns (set by
        # a previous run_full_detection call). If so, skip the redundant
        # detection and just call get_ai_pattern_context directly.
        # This is safe because PatternDetector.run_full_detection is
        # idempotent — it adds columns but doesn't overwrite existing ones.
        _PATTERN_CACHE_FLAG = "_smc_patterns_detected"
        if not m15_df.attrs.get(_PATTERN_CACHE_FLAG, False):
            m15_df = self.pat_detector.run_full_detection(m15_df)
            m15_df.attrs[_PATTERN_CACHE_FLAG] = True
            log.debug(
                f"[SMCEngine] {self.symbol} M15 pattern detection ran "
                f"(was not cached)"
            )
        else:
            log.debug(
                f"[SMCEngine] {self.symbol} M15 pattern detection SKIPPED "
                f"(already detected by analysis_agent)"
            )
        m15_pat   = self.pat_detector.get_ai_pattern_context(m15_df, lookback=3)

        # ── Confluence scoring ─────────────────────────────────
        score, factors, direction = self._score_confluence(
            h4_sweep, h4_bos, h4_choch, nearest_ob, nearest_fvg,
            m15_sweep, m15_bos, m15_pat,
        )
        # Compute grade from score + factors (informational only —
        # not used as a hard gate anywhere downstream).
        grade = self._rank_zone(score, factors)

        # Confidence-pipeline simplification: score flows downstream as
        # continuous confidence, not a hard WAIT cliff.
        # Previously: `tradeable = score >= 45` (or a secondary path),
        # `signal = direction if tradeable else "WAIT"` — this discarded
        # directional evidence whenever score < 45.
        # Now: signal always follows direction. The score becomes
        # `smc_score` in the AI context, and downstream consumers
        # (decision_score, signal_validator, decision_agent) decide
        # whether the overall confluence is sufficient.
        signal = direction if direction != "NEUTRAL" else "WAIT"

        from utils.confidence_trace import confidence_trace
        confidence_trace.record(
            module="smc_engine",
            before=score,
            after=score,
            reason=f"direction={direction}, score={score}/100, grade={grade} (no hard cutoff, MIN_TRADE_SCORE={MIN_TRADE_SCORE} is informational)",
        )

        result = {
            "symbol":        self.symbol,
            "current_price": current_price,
            "h4": {
                "order_blocks": h4_obs,
                "fvgs":         h4_fvgs,
                "bos":          h4_bos,
                "choch":        h4_choch,
                "liquidity_sweep": h4_sweep,
                "nearest_ob":   nearest_ob,
                "nearest_fvg":  nearest_fvg,
            },
            "m15": {
                "liquidity_sweep": m15_sweep,
                "bos":             m15_bos,
                "pattern":         m15_pat,
            },
            "confluence_score":  score,
            "confluence_factors": factors,
            "direction":         direction,
            "grade":             grade,
            "signal":            signal,
            "analysis": self._build_explanation(
                direction, h4_sweep, nearest_ob, nearest_fvg, h4_bos, factors
            ),
        }

        log.info(
            f"[SMCEngine] {self.symbol} | Signal: {signal} | "
            f"Direction: {direction} | Score: {score}/100 | Grade: {grade}"
        )
        return result

    # ═══════════════════════════════════════════════════════
    # DATA FETCH HELPER
    # ═══════════════════════════════════════════════════════

    def _fetch_with_atr(self, timeframe: str, limit: int):
        df = self.fetcher.fetch_ohlcv(self.symbol, timeframe, limit=limit)
        if df is None or df.empty:
            log.warning(f"[SMCEngine] No data for {self.symbol} {timeframe}")
            return None
        return self.ind.add_atr(df)

    # ═══════════════════════════════════════════════════════
    # CONFLUENCE SCORING  ⭐⭐⭐⭐⭐
    # ═══════════════════════════════════════════════════════

    def _score_confluence(
        self, h4_sweep, h4_bos, h4_choch, nearest_ob, nearest_fvg,
        m15_sweep, m15_bos, m15_pat,
    ) -> tuple[int, dict, str]:

        score   = 0
        factors = {
            "liquidity_sweep":     False,
            "order_block":         False,
            "fvg":                 False,
            "bos":                 False,
            "confirmation_candle": False,
        }
        # FIX (institutional review, Finding H-1): see smart_money.py for
        # the full rationale — direction is now derived from the SAME
        # SCORE_WEIGHTS used for `score`, instead of an unweighted vote
        # tally that could contradict the weighted score.
        bull_weight = 0
        bear_weight = 0

        # ── Liquidity Sweep (H4 preferred, M15 fallback) ──────
        sweep = h4_sweep if h4_sweep.get("type") != "NONE" else m15_sweep
        if sweep.get("type") == "BULLISH_SWEEP":
            factors["liquidity_sweep"] = True
            score += SCORE_WEIGHTS["liquidity_sweep"]
            bull_weight += SCORE_WEIGHTS["liquidity_sweep"]
        elif sweep.get("type") == "BEARISH_SWEEP":
            factors["liquidity_sweep"] = True
            score += SCORE_WEIGHTS["liquidity_sweep"]
            bear_weight += SCORE_WEIGHTS["liquidity_sweep"]

        # ── Order Block (active/near zone) ────────────────────
        if nearest_ob and nearest_ob.get("in_zone"):
            factors["order_block"] = True
            score += SCORE_WEIGHTS["order_block"]
            if nearest_ob["direction"] == "BULLISH":
                bull_weight += SCORE_WEIGHTS["order_block"]
            else:
                bear_weight += SCORE_WEIGHTS["order_block"]

        # ── FVG (active/near zone) ─────────────────────────────
        if nearest_fvg and nearest_fvg.get("in_zone"):
            factors["fvg"] = True
            score += SCORE_WEIGHTS["fvg"]
            if nearest_fvg["direction"] == "BULLISH":
                bull_weight += SCORE_WEIGHTS["fvg"]
            else:
                bear_weight += SCORE_WEIGHTS["fvg"]

        # ── BOS (H4 preferred, M15 as confirmation) ────────────
        bos = h4_bos if h4_bos.get("type") != "NONE" else m15_bos
        if bos.get("type") == "BULLISH_BOS":
            factors["bos"] = True
            score += SCORE_WEIGHTS["bos"]
            bull_weight += SCORE_WEIGHTS["bos"]
        elif bos.get("type") == "BEARISH_BOS":
            factors["bos"] = True
            score += SCORE_WEIGHTS["bos"]
            bear_weight += SCORE_WEIGHTS["bos"]

        # ── Confirmation candle (M15 candlestick pattern) ──────
        pat_signal = m15_pat.get("pattern_signal", "")
        if "Bullish" in pat_signal:
            factors["confirmation_candle"] = True
            score += SCORE_WEIGHTS["confirmation_candle"]
            bull_weight += SCORE_WEIGHTS["confirmation_candle"]
        elif "Bearish" in pat_signal:
            factors["confirmation_candle"] = True
            score += SCORE_WEIGHTS["confirmation_candle"]
            bear_weight += SCORE_WEIGHTS["confirmation_candle"]

        if bull_weight > bear_weight:
            direction = "BUY"
        elif bear_weight > bull_weight:
            direction = "SELL"
        else:
            direction = "NEUTRAL"

        return min(100, score), factors, direction

    # ═══════════════════════════════════════════════════════
    # ZONE RANKING  (A+ / A / B / Invalid)
    # ═══════════════════════════════════════════════════════

    def _rank_zone(self, score: int, factors: dict) -> str:
        true_count = sum(1 for v in factors.values() if v)
        has_ob_or_fvg = factors["order_block"] or factors["fvg"]

        if score >= 85 and true_count >= 4 and has_ob_or_fvg:
            return "A+"
        if score >= 65 and true_count >= 3 and has_ob_or_fvg:
            return "A"
        if score >= MIN_TRADE_SCORE:
            return "B"
        return "C"  # was "INVALID" — now just a weak label, not a block

    # ═══════════════════════════════════════════════════════
    # EXPLANATION BUILDER
    # ═══════════════════════════════════════════════════════

    def _build_explanation(self, direction, h4_sweep, nearest_ob, nearest_fvg, h4_bos, factors) -> str:
        parts = []
        if factors["liquidity_sweep"]:
            side = "sell-side" if h4_sweep.get("type") == "BULLISH_SWEEP" else "buy-side"
            parts.append(f"Price swept {side} liquidity")
        if factors["order_block"] and nearest_ob:
            parts.append(f"{nearest_ob['direction'].title()} order block respected at "
                         f"{nearest_ob['zone_bottom']}-{nearest_ob['zone_top']}")
        if factors["fvg"] and nearest_fvg:
            parts.append(f"{nearest_fvg['direction'].title()} FVG reacted at "
                         f"{nearest_fvg['zone_bottom']}-{nearest_fvg['zone_top']}")
        if factors["bos"]:
            parts.append(f"Market structure shifted {direction.lower()}" if direction != "NEUTRAL"
                         else "Break of structure detected")
        if factors["confirmation_candle"]:
            parts.append("Confirmation candle present on M15")

        if not parts:
            return "No significant SMC confluence found — no clear institutional footprint."
        return ". ".join(parts) + "."

    # ═══════════════════════════════════════════════════════
    # FALLBACK
    # ═══════════════════════════════════════════════════════

    def _empty_result(self, reason: str) -> dict:
        return {
            "symbol": self.symbol, "current_price": None,
            "h4": {}, "m15": {},
            "confluence_score": 0, "confluence_factors": {},
            "direction": "NEUTRAL", "grade": "INVALID", "signal": "WAIT",
            "analysis": reason,
        }

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT  (MasterAnalyst handoff)
    # ═══════════════════════════════════════════════════════

    def get_ai_context(self, result: dict) -> dict:
        h4 = result.get("h4", {})
        nearest_ob  = h4.get("nearest_ob")
        nearest_fvg = h4.get("nearest_fvg")

        return {
            "smc_signal":      result.get("signal", "WAIT"),
            "smc_direction":   result.get("direction", "NEUTRAL"),
            "smc_score":       result.get("confluence_score", 0),
            "smc_grade":       result.get("grade", "INVALID"),
            "smc_factors":     result.get("confluence_factors", {}),
            "smc_analysis":    result.get("analysis", ""),
            "smc_h4_ob_zone":  (
                f"{nearest_ob['zone_bottom']}-{nearest_ob['zone_top']}" if nearest_ob else None
            ),
            "smc_h4_fvg_zone": (
                f"{nearest_fvg['zone_bottom']}-{nearest_fvg['zone_top']}" if nearest_fvg else None
            ),
            "smc_h4_bos":      h4.get("bos", {}).get("type", "NONE"),
            "smc_h4_choch":    h4.get("choch", {}).get("type", "NONE"),
        }

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, result: dict) -> None:
        icon = {"BUY": "🟢", "SELL": "🔴", "WAIT": "🟡"}.get(result.get("signal"), "⚪")
        bar  = "═" * 56
        log.info(bar)
        log.info("  🧠  SMC ENGINE  (Day 44)")
        log.info(bar)
        log.info(f"  Pair         : {result['symbol']}")
        log.info(f"  Signal       : {icon} {result.get('signal')}")
        log.info(f"  Direction    : {result.get('direction')}")
        log.info(f"  Score        : {result.get('confluence_score')}/100")
        log.info(f"  Grade        : {result.get('grade')}")
        log.info("")
        factors = result.get("confluence_factors", {})
        for name, weight in SCORE_WEIGHTS.items():
            mark = "✅" if factors.get(name) else "❌"
            log.info(f"  {mark} {name:<22} (+{weight})")
        log.info("")
        log.info(f"  Analysis     : {result.get('analysis')}")
        log.info(bar)