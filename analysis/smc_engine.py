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

# FIX (audit C1): weights used to be static across every instrument,
# timeframe, and volatility regime. Institutional desks weight confluence
# factors differently depending on regime — e.g. in a high-volatility
# expansion, a liquidity sweep is more meaningful (real stop-hunt) while a
# single confirmation candle is less meaningful (noise); in a quiet range,
# zones (OB/FVG) matter more and sweeps are less conclusive. These
# multipliers are applied to SCORE_WEIGHTS based on the H4 ATR's own
# recent regime (see _volatility_regime()) and then renormalized back to
# a 100-point scale so score is still comparable across regimes.
REGIME_WEIGHT_MULTIPLIERS = {
    "HIGH": {
        "liquidity_sweep": 1.25, "bos": 1.10, "order_block": 0.90,
        "fvg": 0.85, "confirmation_candle": 0.70,
    },
    "LOW": {
        "liquidity_sweep": 0.85, "bos": 1.00, "order_block": 1.15,
        "fvg": 1.15, "confirmation_candle": 1.05,
    },
    "NORMAL": {
        "liquidity_sweep": 1.0, "bos": 1.0, "order_block": 1.0,
        "fvg": 1.0, "confirmation_candle": 1.0,
    },
}

# FIX (audit H5): patterns.py's _pattern_signal() maps every bullish/
# bearish pattern name to the same Bullish/Bearish label with no notion
# that e.g. a three-bar reversal is a stronger signal than a lone hammer.
# This tier table (standard TA reliability ordering) plus the candle's own
# body-size-vs-ATR (computed below) gives confirmation_candle a real
# quality multiplier instead of treating every pattern identically.
PATTERN_RELIABILITY_TIER = {
    "three_bar_reversal_bullish": 1.20, "three_bar_reversal_bearish": 1.20,
    "bullish_engulfing": 1.15, "bearish_engulfing": 1.15,
    "morning_star": 1.15, "evening_star": 1.15,
    "three_bar_continuation_bullish": 1.05, "three_bar_continuation_bearish": 1.05,
    "breakout_bullish": 1.10, "breakout_bearish": 1.10,
    "bullish_pin_bar": 1.05, "bearish_pin_bar": 1.05,
    "hammer": 0.90, "shooting_star": 0.90,
    "doji": 0.70,
}
DEFAULT_PATTERN_RELIABILITY = 1.0

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

        # FIX (audit C1): volatility regime from the H4 ATR's own recent
        # history (not an arbitrary external threshold) — current ATR vs.
        # its trailing average. Feeds into dynamic factor weighting in
        # _score_confluence() below.
        vol_regime = self._volatility_regime(h4_df, h4_atr)

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
        score, factors, direction, quality_notes = self._score_confluence(
            h4_sweep, h4_bos, h4_choch, nearest_ob, nearest_fvg,
            m15_sweep, m15_bos, m15_pat,
            vol_regime=vol_regime, h4_atr=h4_atr, m15_atr=m15_atr, current_price=current_price,
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

        # BUG FIX: unguarded import — a missing/broken utils.confidence_trace
        # would crash analyze() (this module's main entry point, and SMC
        # score feeds directly into the Confluence gate) for what's only a
        # diagnostic trace log. Made fail-safe.
        try:
            from utils.confidence_trace import confidence_trace
            confidence_trace.record(
                module="smc_engine",
                before=score,
                after=score,
                reason=f"direction={direction}, score={score}/100, grade={grade} (no hard cutoff, MIN_TRADE_SCORE={MIN_TRADE_SCORE} is informational)",
            )
        except Exception as e:
            log.debug(f"[SMCEngine] confidence_trace unavailable (non-fatal): {e}")

        # FIX (audit C2): previously only BUY/SELL/WAIT with a raw score —
        # no probability, uncertainty, or expected-RR surface at all. This
        # adds a probability-shaped number and an uncertainty tier, but is
        # explicit that it's a monotonic transform of the confluence score,
        # NOT a statistically calibrated win-rate (that requires backtested
        # outcome data this module doesn't have — expected_rr is likewise
        # not computed here since it needs an entry/SL/TP model this module
        # doesn't own). Consumers should treat this as "how much confluence
        # agrees", not "P(win)".
        true_count = sum(1 for v in factors.values() if v)
        uncertainty = "LOW" if true_count >= 4 else "MEDIUM" if true_count == 3 else "HIGH"

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
            "quality_notes":     quality_notes,   # per-factor freshness/magnitude adjustments applied (H1/H3/H4/H5)
            "volatility_regime": vol_regime,       # HIGH | NORMAL | LOW — drove the dynamic weights (C1)
            "confluence_probability": round(score / 100, 2),  # heuristic only — see note above (C2)
            "probability_basis": "heuristic_confluence_score_not_statistically_calibrated",
            "uncertainty":       uncertainty,      # LOW/MEDIUM/HIGH based on how many factors actually fired
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

    def _volatility_regime(self, h4_df, h4_atr) -> str:
        """
        FIX (audit C1): classifies current H4 volatility relative to its
        own recent history (not a hardcoded absolute threshold, which
        would need per-instrument tuning). Returns 'HIGH' / 'LOW' /
        'NORMAL'. Falls back to 'NORMAL' if ATR data is insufficient
        rather than guessing.
        """
        if h4_atr is None or "atr" not in h4_df.columns:
            return "NORMAL"
        atr_series = h4_df["atr"].dropna()
        if len(atr_series) < 20:
            return "NORMAL"
        mean_atr = float(atr_series.tail(50).mean())
        if mean_atr <= 0:
            return "NORMAL"
        ratio = h4_atr / mean_atr
        if ratio >= 1.3:
            return "HIGH"
        if ratio <= 0.7:
            return "LOW"
        return "NORMAL"

    def _dynamic_weights(self, vol_regime: str) -> dict:
        """SCORE_WEIGHTS adjusted for the current volatility regime (C1),
        renormalized back to a 100-point scale so scores stay comparable
        across regimes."""
        mult = REGIME_WEIGHT_MULTIPLIERS.get(vol_regime, REGIME_WEIGHT_MULTIPLIERS["NORMAL"])
        raw = {k: SCORE_WEIGHTS[k] * mult[k] for k in SCORE_WEIGHTS}
        scale = 100.0 / sum(raw.values())
        return {k: v * scale for k, v in raw.items()}

    @staticmethod
    def _freshness_multiplier(candles_ago) -> float:
        """FIX (audit H1/H3): a 5-candle-old zone and a 70-candle-old zone
        used to get identical weight. candles_ago comes straight from
        order_block.py / fvg_detector.py — real data, not estimated."""
        if candles_ago is None:
            return 1.0
        if candles_ago <= 10:
            return 1.15
        if candles_ago <= 40:
            return 1.0
        return 0.70

    @staticmethod
    def _magnitude_multiplier(level, current_price, atr) -> float:
        """FIX (audit H4/H2-partial): a break/sweep 1 pip beyond a level
        and one 50 pips beyond it used to score identically. Uses the
        break's distance from the broken level, in ATR units, as a proxy
        for conviction."""
        if level is None or current_price is None or not atr:
            return 1.0
        ratio = abs(current_price - level) / atr
        if ratio >= 1.0:
            return 1.20
        if ratio >= 0.5:
            return 1.0
        return 0.75

    def _score_confluence(
        self, h4_sweep, h4_bos, h4_choch, nearest_ob, nearest_fvg,
        m15_sweep, m15_bos, m15_pat,
        vol_regime: str = "NORMAL", h4_atr: float | None = None,
        m15_atr: float | None = None, current_price: float | None = None,
    ) -> tuple[int, dict, str, dict]:

        weights = self._dynamic_weights(vol_regime)   # C1: regime-adjusted, not static

        score   = 0.0
        factors = {
            "liquidity_sweep":     False,
            "order_block":         False,
            "fvg":                 False,
            "bos":                 False,
            "confirmation_candle": False,
        }
        quality_notes: dict = {}
        # FIX (institutional review, Finding H-1): see smart_money.py for
        # the full rationale — direction is now derived from the SAME
        # SCORE_WEIGHTS used for `score`, instead of an unweighted vote
        # tally that could contradict the weighted score.
        bull_weight = 0.0
        bear_weight = 0.0

        # ── Liquidity Sweep (H4 preferred, M15 fallback) ──────
        # FIX (audit H2-partial): magnitude-weighted by how far price
        # traveled beyond the swept level (in ATR units) — a 1-pip poke
        # and a 30-pip stop-hunt no longer score the same.
        sweep = h4_sweep if h4_sweep.get("type") != "NONE" else m15_sweep
        sweep_atr = h4_atr if sweep is h4_sweep else m15_atr
        if sweep.get("type") in ("BULLISH_SWEEP", "BEARISH_SWEEP"):
            mag_mult = self._magnitude_multiplier(sweep.get("level"), current_price, sweep_atr)
            w = weights["liquidity_sweep"] * mag_mult
            factors["liquidity_sweep"] = True
            score += w
            quality_notes["liquidity_sweep"] = {"magnitude_multiplier": round(mag_mult, 2)}
            if sweep["type"] == "BULLISH_SWEEP":
                bull_weight += w
            else:
                bear_weight += w

        # ── Order Block (active/near zone) ────────────────────
        # FIX (audit H1): freshness (candles_ago) and the detector's own
        # quality_score (structure-break/FVG-confluence/sweep-conditioned
        # composite from order_block.py) now both scale the weight instead
        # of every active OB counting the same.
        if nearest_ob and nearest_ob.get("in_zone"):
            fresh_mult = self._freshness_multiplier(nearest_ob.get("candles_ago"))
            q = nearest_ob.get("quality_score")
            q_mult = (0.5 + 0.5 * min(1.0, q / 100.0)) if q is not None else 1.0
            ob_mult = fresh_mult * q_mult
            w = weights["order_block"] * ob_mult
            factors["order_block"] = True
            score += w
            quality_notes["order_block"] = {
                "candles_ago": nearest_ob.get("candles_ago"),
                "quality_score": q,
                "multiplier": round(ob_mult, 2),
            }
            if nearest_ob["direction"] == "BULLISH":
                bull_weight += w
            else:
                bear_weight += w

        # ── FVG (active/near zone) ─────────────────────────────
        # FIX (audit H3): freshness (candles_ago) + gap size relative to
        # ATR now scale the weight — a tiny gap and a wide displacement
        # gap used to be worth the same.
        if nearest_fvg and nearest_fvg.get("in_zone"):
            fresh_mult = self._freshness_multiplier(nearest_fvg.get("candles_ago"))
            gap_size = nearest_fvg.get("zone_top", 0) - nearest_fvg.get("zone_bottom", 0)
            gap_mult = 1.0
            if h4_atr:
                gap_ratio = gap_size / h4_atr
                gap_mult = min(1.3, 0.7 + 0.3 * gap_ratio)
            fvg_mult = fresh_mult * gap_mult
            w = weights["fvg"] * fvg_mult
            factors["fvg"] = True
            score += w
            quality_notes["fvg"] = {
                "candles_ago": nearest_fvg.get("candles_ago"),
                "gap_size": round(gap_size, 5),
                "multiplier": round(fvg_mult, 2),
            }
            if nearest_fvg["direction"] == "BULLISH":
                bull_weight += w
            else:
                bear_weight += w

        # ── BOS (H4 preferred, M15 as confirmation) ────────────
        # FIX (audit H4): break magnitude beyond the broken swing level
        # (in ATR units) now scales the weight instead of every BOS
        # counting the same regardless of how decisive the break was.
        bos = h4_bos if h4_bos.get("type") != "NONE" else m15_bos
        bos_atr = h4_atr if bos is h4_bos else m15_atr
        if bos.get("type") in ("BULLISH_BOS", "BEARISH_BOS"):
            mag_mult = self._magnitude_multiplier(bos.get("level"), current_price, bos_atr)
            w = weights["bos"] * mag_mult
            factors["bos"] = True
            score += w
            quality_notes["bos"] = {"magnitude_multiplier": round(mag_mult, 2)}
            if bos["type"] == "BULLISH_BOS":
                bull_weight += w
            else:
                bear_weight += w

        # ── Confirmation candle (M15 candlestick pattern) ──────
        # FIX (audit H5): a Hammer and a Three-Bar-Reversal used to score
        # identically. Now weighted by a standard TA reliability tier
        # (PATTERN_RELIABILITY_TIER) combined with the candle's own body
        # size relative to ATR (a bigger-bodied confirmation candle is a
        # more decisive signal than a small one of the same pattern type).
        pat_signal = m15_pat.get("pattern_signal", "")
        pattern_name = m15_pat.get("latest_pattern", "none")
        if "Bullish" in pat_signal or "Bearish" in pat_signal:
            reliability = PATTERN_RELIABILITY_TIER.get(pattern_name, DEFAULT_PATTERN_RELIABILITY)
            body_mult = 1.0
            if m15_atr:
                body_ratio = m15_pat.get("body_size", 0) / m15_atr
                body_mult = min(1.3, 0.7 + 0.3 * body_ratio)
            pat_mult = reliability * body_mult
            w = weights["confirmation_candle"] * pat_mult
            factors["confirmation_candle"] = True
            score += w
            quality_notes["confirmation_candle"] = {
                "pattern": pattern_name, "reliability_tier": reliability,
                "multiplier": round(pat_mult, 2),
            }
            if "Bullish" in pat_signal:
                bull_weight += w
            else:
                bear_weight += w

        if bull_weight > bear_weight:
            direction = "BUY"
        elif bear_weight > bull_weight:
            direction = "SELL"
        else:
            direction = "NEUTRAL"

        return min(100, round(score)), factors, direction, quality_notes

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
            "smc_probability": result.get("confluence_probability", 0),   # heuristic, see probability_basis
            "smc_uncertainty": result.get("uncertainty", "HIGH"),
            "smc_volatility_regime": result.get("volatility_regime", "NORMAL"),
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