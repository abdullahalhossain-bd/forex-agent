# agents/analysis_agent.py  (Day 63 Session Intelligence + Day 65 Intermarket Update)
# ============================================================
# Day 47 pipeline-এর সাথে Day 63 Session Intelligence এবং Day 65
# Intermarket Analysis Engine যোগ হয়েছে।
#
# নতুন Step (Day 63):
#   Step 0: SessionAnalyzer — time-aware strategy selection
#
# নতুন Step (Day 65):
#   Step 8.5: IntermarketEngine — global macro context (DXY, Gold,
#             Oil, US10Y, S&P500, VIX) + Risk-On/Off regime + Macro+SMC
#             fusion। SMC engine (step 8) আর Session re-run-এর পরে
#             বসানো হয়েছে, কারণ fusion-এর জন্য smc_ctx ও session_ctx
#             দুটোই দরকার।
#
# Session Intelligence inject হয় সব module-এর আগে, তাই সব context
# session-aware হয়ে যায়। Dead zone-এ trade block হয়। Session-specific
# pair priority দেওয়া হয়। Intermarket context MasterAnalyst-কে global
# macro picture দেয়, যাতে AI শুধু chart না দেখে গোটা market দেখে।
# ============================================================

from typing import Dict, Any, List, Optional

from analysis.patterns import PatternDetector
from analysis.support_resistance import SupportResistance
from analysis.market_bias import MarketBiasEngine
from analysis.advanced_patterns import AdvancedPatternDetector
from analysis.fibonacci import FibonacciEngine
from analysis.sentiment import SentimentEngine
from analysis.smc_engine import SMCEngine
from analysis.sentiment_data import SentimentDataProvider
from analysis.session_analyzer import SessionAnalyzer   # ← Day 63
from analysis.intermarket import IntermarketEngine       # ← Day 65
from analysis.currency_strength import CurrencyStrengthEngine  # ← Day 64, wired 2026-07-22
# Day 90 — Six new analyzers
from analysis.divergence import DivergenceEngine
from analysis.ichimoku import IchimokuEngine
from analysis.volatility import VolatilityEngine
from analysis.volume_profile import VolumeProfileEngine
from analysis.smc_advanced import SMCAdvancedEngine
from analysis.structure_mtf import MTFStructureEngine
from analysis.structure import MarketStructureEngine
# Day 92 — NewsAPI.org provider (real-time news sentiment)
from analysis.news_api_provider import get_news_api_provider
# Day 94 — Institutional grade: economic calendar + FRED + retail sentiment
from fundamental.economic_calendar_api import EconomicCalendarAPI
from fundamental.fred_data import get_fred_api
from analysis.retail_sentiment import get_retail_sentiment_api
# Day 96 — Correlation + Volatility + Institutional Flow + Economic Surprise
from analysis.correlation_engine import CorrelationEngine
from analysis.institutional_flow import InstitutionalFlowEngine
from fundamental.economic_surprise import EconomicSurpriseEngine
# Day 97 — Microstructure + Network Monitor + Forecast Engine
from analysis.microstructure import get_microstructure_engine
from system.network_monitor import get_network_monitor
from ml.forecast_engine import get_forecast_engine
from strategy.selector import StrategySelector            # ← Day 90
from fundamental.news_filter import NewsFilter
from ai.ai_analyst import AIAnalyst
from agents.master_analyst import MasterAnalyst
from strategy.signal_engine import SignalEngine
from utils.logger import get_logger

log = get_logger("analysis_agent")



# P0 fix (audit C6): helper to track master_confidence overwrites.
def _track_confidence(master_ctx, stage, value):
    """Append (stage, value) to master_ctx['confidence_chain']."""
    try:
        chain = master_ctx.setdefault("confidence_chain", [])
        chain.append((stage, value))
    except Exception:
        pass


def _apply_confidence_penalty(signal_result: dict, amount: float, reason: str, source: str = "analysis") -> None:
    """Lower confidence by a small amount instead of hard-blocking the trade."""
    try:
        if not isinstance(signal_result, dict):
            return
        current_conf = float(signal_result.get("confidence", 0) or 0)
        # Confidence Calibration (audit Rule 6): cap below 100% — real
        # markets always carry uncertainty. Uses the same ceiling as
        # EntrySafetyFilters.calibrate_confidence() for consistency.
        try:
            from core.entry_safety_filters import EntrySafetyFilters
            new_conf = EntrySafetyFilters.calibrate_confidence(current_conf - amount)
        except Exception:
            new_conf = max(0, min(96, current_conf - amount))
        signal_result["confidence"] = new_conf
        penalties = signal_result.setdefault("confidence_penalties", [])
        penalties.append({
            "source": source,
            "reason": reason,
            "amount": amount,
            "confidence": new_conf,
        })
    except Exception:
        pass


class AnalysisAgent:
    """
    Day 65 Unified Pipeline:
      Session Intelligence (Day 63)
      -> Patterns -> S/R -> Advanced Patterns -> Fibonacci -> Bias -> Signal
      -> Sentiment -> SMC -> Session re-run -> Intermarket (Day 65)
      -> News -> Classic LLM -> Vision AI -> MasterAnalyst
    """

    def __init__(self, chart_reader=None):
        self.chart_reader       = chart_reader
        self.session_analyzer   = SessionAnalyzer()      # Day 63
        self.intermarket_engine = IntermarketEngine()     # Day 65
        # Day 90 — six new analyzers
        self.divergence_engine   = DivergenceEngine()
        self.ichimoku_engine     = IchimokuEngine()
        self.volatility_engine   = VolatilityEngine()
        self.volume_profile_eng  = VolumeProfileEngine()
        self.smc_advanced_engine = SMCAdvancedEngine()
        self.mtf_structure_eng   = MTFStructureEngine()
        self.structure_engine    = MarketStructureEngine()
        self.strategy_selector   = StrategySelector(conservative=False)
        # FIX: add fetcher for H4 data (reusable across calls)
        from data.fetcher import get_data_fetcher
        self._h4_fetcher         = get_data_fetcher()
        # PERFORMANCE FIX (execution-parity audit follow-up, backtest smoke
        # test, 2026-07-19): SentimentDataProvider and InstitutionalFlowEngine
        # both implement their own instance-level `self._cache` dict (5min /
        # 6h TTL respectively) but run() used to construct a BRAND NEW
        # instance of each on every single call — i.e. every bar in a
        # backtest, every cycle live. A fresh instance means a fresh empty
        # `self._cache = {}`, so the cache never actually hit; every bar
        # re-fetched yfinance/COT data over the network, which is what made
        # a multi-hundred-bar backtest take hours instead of minutes. Fix:
        # construct both ONCE here (matching self.intermarket_engine above,
        # which already got this right — its internal MacroDataProvider's
        # cache DOES work because IntermarketEngine itself is a single
        # instance reused for AnalysisAgent's whole lifetime). No behavior
        # change to WHAT is computed, no change to signal-generation logic —
        # this is purely fixing where these objects' caches actually live so
        # the caching code that was already written and intended actually
        # takes effect, in both backtest and live.
        self.sentiment_data_provider  = SentimentDataProvider()
        self.institutional_flow_engine = InstitutionalFlowEngine()
        # 2026-07-22 dead-file audit fix: Day-64 CurrencyStrengthEngine was
        # fully implemented (28-pair MT5 matrix, momentum, ranking,
        # opportunity finder) but never wired — the sentiment step below
        # was silently falling back to a crude 1-day yfinance %-change
        # proxy for "currency strength" (see analysis/sentiment_data.py
        # get_currency_strengths()). Constructed once here, same reasoning
        # as institutional_flow_engine/intermarket_engine above, so its
        # internal 5-min strength-matrix cache actually persists across
        # cycles instead of being rebuilt (and re-fetching 28 pairs) every
        # single call.
        self.currency_strength_engine = CurrencyStrengthEngine(timeframe="1h")

    def run(self, market_output: dict, memory_ctx: dict = None) -> dict:
        if "error" in market_output:
            return {"error": market_output["error"]}

        df        = market_output["df"]
        ind_ctx   = market_output["ind_ctx"]
        regime    = market_output["regime"]
        mtf_bias  = market_output["mtf_bias"]
        # FIX (execution-parity audit — found via backtest smoke test):
        # mtf_bias arrives as a dict on the success path (MarketAgent's
        # mtf.print_summary() -> get_bias() -> {"bias":..., "confidence":...})
        # but as a bare string ("NEUTRAL") on the failure-default path —
        # both in agents/market_agent.py (MTF analysis raised) and in
        # backtest/unified_engine.py's historical replay (MTF isn't
        # computed at all there). Several downstream consumers
        # (MarketBiasEngine.analyze, SignalEngine.generate,
        # MasterAnalyst.analyze) call `mtf_bias.get(...)` unconditionally
        # and crashed with AttributeError whenever the string form reached
        # them. A few call sites in this file already had their own
        # one-off `{"bias": mtf_bias} if isinstance(mtf_bias, str) else
        # mtf_bias` guard, but not all of them — normalizing once, here,
        # at the point mtf_bias enters this function, closes the gap for
        # every consumer instead of requiring each new call site to
        # remember to guard it individually.
        if isinstance(mtf_bias, str):
            mtf_bias = {"bias": mtf_bias, "confidence": "LOW"}
        symbol    = market_output["symbol"]
        timeframe = market_output.get("timeframe", "15m")

        log.info(
            f"[AnalysisAgent] Running Day 65 pipeline for {symbol} ({timeframe})"
        )

        # ── 0. SESSION INTELLIGENCE (Day 63) ─────────────────
        # SMC context এখনো নেই, তাই খালি dict দিয়ে শুরু, পরে update
        # compute_fusion=False — SMC data not available yet; fusion will be
        # computed in the second analyze() call at line ~520 after SMC runs.
        # This avoids the misleading "Fusion: ❌ (0/100)" log line.
        session_result = self.session_analyzer.analyze(
            pair           = symbol,
            smc_ctx        = {},
            signal         = "NO TRADE",
            signal_conf    = 0,
            compute_fusion = False,
        )
        session_ctx = self.session_analyzer.get_ai_context(session_result)

        # Dead zone guard — trade block
        # Day 81+ hotfix: in TEST_MODE, we bypass the dead zone so the
        # bot actually places trades during off-hours for MT5 verification.
        # Production should keep this block — dead zones are real risk.
        _skip_dead_zone = False
        try:
            from config import TEST_MODE
            _skip_dead_zone = bool(TEST_MODE)
        except Exception:
            pass

        if _skip_dead_zone and session_result["session_info"]["is_dead_zone"]:
            log.info(
                f"[AnalysisAgent] ⚠️ DEAD ZONE at {session_ctx['gmt_time']} — "
                f"BYPASSED (TEST_MODE=true). Pipeline continues for MT5 testing."
            )
            # Mark session_ctx so downstream consumers know we're in dead-zone-bypass
            session_ctx["dead_zone_bypassed"] = True
        elif session_result["session_info"]["is_dead_zone"]:
            log.info(f"[AnalysisAgent] ⛔ DEAD ZONE at {session_ctx['gmt_time']} — pipeline paused")
            # Day 81+ hotfix: include ALL downstream keys with safe defaults
            # so trader.py's `analysis_out["signal"].get("entry")` doesn't
            # raise KeyError. Every key that the success-path return at
            # the bottom of this function includes must also be present
            # here — otherwise trader.py crashes when it tries to read
            # them off the dead-zone dict.
            return {
                "df":                df,
                "pat_ctx":           {},
                "advanced_patterns": {},
                "advanced_pat_ctx":  {},
                "sr_result":         {"support_zones": [], "resistance_zones": []},
                "sr_ctx":            {},
                "liquidity_ctx":     {},
                "fib_result":        {},
                "fib_ctx":           {},
                "bias_result":       {},
                "bias_ctx":          {},
                # signal_result normally has shape {signal, confidence, entry, ...}
                # Provide minimal safe defaults so callers don't crash.
                "signal":            {"signal": "NO TRADE", "confidence": 0, "entry": None},
                "signal_ctx":        {},
                "llm":               {"signal": "WAIT", "confidence": 0},
                "llm_ctx":           {},
                "news":              {"trade_allowed": True, "news_reason": "dead zone"},
                "news_ctx":          {"news_trade_allowed": True, "news_reason": "dead zone"},
                "sentiment":         {},
                "sentiment_ctx":     {"sentiment_bias": "NEUTRAL", "sentiment_score": 0},
                "conflict":          {"has_conflict": False, "confidence_adjustment": 0},
                "smc":               {},
                "smc_ctx":           {},
                "vision":            {},
                "vision_ctx":        {},
                "vision_fusion":     {},
                "session":           session_result,
                "session_ctx":       session_ctx,
                "intermarket":       {},
                "intermarket_ctx":   {},
                "currency_strength_ctx": {},
                "macro_fusion":      {},
                "master":            {},
                "master_ctx":        {"master_signal": "WAIT", "master_confidence": 0,
                     # P0 fix (audit C6): audit trail for the 5-stage
                     # master_confidence overwrite chain.
                     "confidence_chain": [("init", 0)]},
                "news_intelligence": {},
                "confluence":        {},
                "feature_vector":    {},
                "ml_prediction":     {},
                "ensemble":          {},
                "rl_agent":          {},
                "master_decision":   {},
                # Day 90 — safe defaults for new contexts
                "structure":          {},
                "structure_ctx":      {"structure_valid": False, "structure_bias": "NEUTRAL"},
                "divergence":         {},
                "divergence_ctx":     {"divergence_valid": False, "divergence_signal": "NONE"},
                "ichimoku":           {},
                "ichimoku_ctx":       {"ichimoku_valid": False, "ichimoku_signal": "WAIT"},
                "volatility":         {},
                "volatility_ctx":     {"volatility_valid": False, "volatility_signal": "WAIT"},
                "volume_profile":     {},
                "volume_profile_ctx": {"volume_profile_valid": False, "vp_signal": "WAIT"},
                "smc_advanced":       {},
                "smc_advanced_ctx":   {"smc_adv_valid": False, "smc_adv_signal": "WAIT"},
                "news_api":            {},
                "news_api_ctx":        {"newsapi_bias":"NEUTRAL","newsapi_score":0,"newsapi_source":"unknown"},
                "econ_calendar":       {},
                "econ_calendar_ctx":   {"econcal_source":"none","econcal_trade_block":False},
                "fred_macro":          {},
                "fred_ctx":            {"fred_source":"none","fred_yield_curve":"unknown"},
                "retail_sentiment":    {},
                "retail_sentiment_ctx":{"sentiment_source":"fallback","sentiment_bias":"NEUTRAL"},
                "mtf_structure":      {},
                "mtf_structure_ctx":  {"mtf_structure_valid": False, "mtf_trade_permission": "NO_TRADE"},
                "strategy":           {"strategy": "WAIT", "confidence": 0, "risk_mult": 0.0, "active_modules": []},
                "final_signal":      "NO TRADE",
                # Dead-zone-specific metadata (consumed by trader.py for
                # reject_reason formatting)
                "dead_zone":         True,
                "dead_zone_reason":  "Low liquidity dead zone — no trades",
            }

        # ── 0.5 Default contexts (Day 81+ hotfix) ────────────────
        # These 7 ctx dicts are produced by analysis stages that run LATER
        # in the pipeline (Day 66 NewsIntelligence, Day 67 Confluence,
        # Day 68 FeatureStore, Day 69 ModelPredictor, Day 70 Ensemble,
        # Day 71 RL Agent, Day 72 MasterDecision). However, the TEST_MODE
        # aggressive fast-path below (line ~390) returns early and
        # references them. Without these defaults, the first early return
        # raises UnboundLocalError → supervisor catches → restart → crash
        # again → infinite restart loop.  Initialize ALL of them to {} so
        # every return path has a consistent schema.
        news_intel_ctx      = {}
        confluence_ctx      = {}
        feature_vector_ctx  = {}
        ml_prediction_ctx   = {}
        ensemble_ctx        = {}
        rl_ctx              = {}
        master_decision_ctx = {}

        # ── 1. Candlestick Patterns ───────────────────────────
        # Round-10 audit fix: mark df with a cache flag so downstream
        # modules (e.g. smc_engine.py) can skip redundant detection.
        detector = PatternDetector()
        df       = detector.run_full_detection(df)
        df.attrs["_smc_patterns_detected"] = True  # Round-10 dedup flag
        detector.get_latest_patterns(df, lookback=5)
        pat_ctx  = detector.get_ai_pattern_context(df)

        # ── 2. Support & Resistance ───────────────────────────
        sr      = SupportResistance()
        sr_res  = sr.analyze(df)
        sr.get_summary(sr_res)
        sr_ctx  = sr.get_ai_context(sr_res)

        # ── 2b. Liquidity (equal highs/lows + sweep detection) ─
        # Feeds the entry-safety Liquidity Sweep Detector (audit fix) —
        # without this, the override gate below has no sweep data and
        # defaults to "safe", which is not a fail-closed default we want
        # for something this cheap to compute.
        liquidity_ctx = {}
        try:
            from analysis.liquidity import LiquidityEngine
            liq_engine = LiquidityEngine()
            liq_res    = liq_engine.analyze(df)
            liquidity_ctx = liq_engine.get_ai_context(liq_res)
        except Exception as e:
            log.debug(f"[AnalysisAgent] Liquidity engine unavailable: {e}")
            liquidity_ctx = {}

        # ── 3. Advanced Patterns ─────────────────────────────
        advanced_pat_ctx = {}
        adv_patterns     = {}
        try:
            adv_detector = AdvancedPatternDetector(lookback=100)
            adv_patterns = adv_detector.detect_all(df)
            adv_patterns = adv_detector.boost_confidence(
                adv_patterns,
                ind_ctx    = ind_ctx,
                sr_ctx     = sr_ctx,
                regime_ctx = regime,
                pat_ctx    = pat_ctx,
            )
            adv_patterns = adv_detector.filter_false_patterns(
                adv_patterns,
                regime_ctx = regime,
                ind_ctx    = ind_ctx,
            )
            adv_detector.print_summary(adv_patterns)
            # Round-5 audit fix: pass the already-computed adv_patterns
            # list into get_ai_context() so it doesn't re-run
            # detect_all() + boost + filter on the same df. Previously
            # the operator saw TWO identical "Advanced patterns
            # detected: N" log lines per cycle (the second one fired
            # inside get_ai_context()'s old full-pipeline path).
            advanced_pat_ctx = adv_detector.get_ai_context(
                df,
                ind_ctx    = ind_ctx,
                sr_ctx     = sr_ctx,
                regime_ctx = regime,
                pat_ctx    = pat_ctx,
                patterns   = adv_patterns,  # ← reuse, don't recompute
            )
        except Exception as e:
            log.warning(f"[AnalysisAgent] Advanced Patterns error: {e}")

        # ── 4. Fibonacci Engine ──────────────────────────────
        fib_ctx    = {}
        fib_result = {}
        try:
            fib_engine = FibonacciEngine(timeframe=timeframe)
            fib_result = fib_engine.analyze(df, sr_ctx=sr_ctx, ind_ctx=ind_ctx)
            fib_engine.print_summary(fib_result)
            fib_ctx    = fib_engine.get_ai_context(fib_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] Fibonacci Engine error: {e}")

        # ── 5. Market Bias ────────────────────────────────────
        # Day 133 fix: fib_ctx was already computed above (step 4) and
        # passed to SignalEngine below, but NOT to MarketBiasEngine —
        # even after the Day 101+ fix added the fib_ctx parameter to
        # MarketBiasEngine.analyze(), this call site was never updated
        # to actually pass it. Production logs kept showing SignalEngine
        # correctly counting a Fib GOLDEN_ZONE BUY signal (bull score +4)
        # while MarketBiasEngine, looking at the same market state, never
        # saw it at all (Fib entry simply absent from its breakdown) —
        # because it was still being called with the old 4-argument form.
        bias_engine = MarketBiasEngine()
        bias_result = bias_engine.analyze(ind_ctx, pat_ctx, sr_ctx, mtf_bias, fib_ctx)
        bias_engine.print_summary(bias_result)
        bias_ctx    = bias_engine.get_ai_context(bias_result)

        # ── 6. Rule-based Signal ──────────────────────────────
        # 17-module integration pass: pull in votes from the
        # previously imported-only modules that only need `df`
        # (andean_oscillator, supertrend, utbot_alerts,
        # nadaraya_watson_envelope, daily_high_low,
        # auction_market_theory, candlestick_patterns_ml).
        # breaker_block/flip_zones/curve_mtf need order-block and
        # zone data that isn't ready until step 8 (SMC Engine) —
        # see the merge_zone_votes_into_signal() call below.
        extended_ctx = {}
        try:
            from analysis.extended_modules_adapter import get_extended_votes
            extended_ctx = get_extended_votes(df, symbol=symbol)
        except Exception as e:
            log.debug(f"[AnalysisAgent] Extended modules adapter error: {e}")

        signal_engine = SignalEngine()
        signal_result = signal_engine.generate(
            ind_ctx          = ind_ctx,
            pat_ctx          = pat_ctx,
            sr_ctx           = sr_ctx,
            regime           = regime,
            mtf_bias         = mtf_bias,
            advanced_pat_ctx = advanced_pat_ctx,
            fib_ctx          = fib_ctx,
            extended_ctx     = extended_ctx,
        )
        signal_engine.print_summary(signal_result)
        signal_ctx = signal_engine.get_ai_context(signal_result)

        # ── 6b. Day 97+ Book Rules: Trendline, Supply/Demand, Volume Confirm, Oscillator Gate ─
        trendline_ctx = {}
        try:
            from analysis.trendline_engine import TrendlineEngine
            te = TrendlineEngine()
            trendline_result = te.analyze(df, pair=symbol)
            trendline_ctx = {
                "trendline": trendline_result,
                "trendline_signals": trendline_result.get("signals", []),
            }
        except Exception as e:
            log.debug(f"[AnalysisAgent] Trendline engine error: {e}")

        supply_demand_ctx = {}
        try:
            from analysis.supply_demand_zones import SupplyDemandZones
            sd = SupplyDemandZones()
            sd_result = sd.detect(df)
            supply_demand_ctx = {
                "supply_demand": sd_result,
                "nearest_demand": sd_result.get("nearest_demand"),
                "nearest_supply": sd_result.get("nearest_supply"),
            }
        except Exception as e:
            log.debug(f"[AnalysisAgent] Supply/Demand zones error: {e}")

        volume_confirm_ctx = {}
        try:
            from analysis.volume_confirmation import VolumeConfirmation
            vc = VolumeConfirmation()
            vc_trend = vc.check_trend_confirmation(df)
            vc_context = vc.get_volume_context(df)
            volume_confirm_ctx = {
                "trend_confirmation": vc_trend,
                "volume_context": vc_context,
            }
        except Exception as e:
            log.debug(f"[AnalysisAgent] Volume confirmation error: {e}")

        oscillator_gate_ctx = {}
        try:
            from analysis.oscillator_regime_gate import OscillatorRegimeGate
            gate = OscillatorRegimeGate()
            rsi_val = float(ind_ctx.get("rsi", 50))
            adx_val = float(ind_ctx.get("adx", 0))
            gate_result = gate.adjust_signal(
                signal=signal_result.get("signal", "WAIT"),
                source="RSI",
                rsi=rsi_val,
                adx=adx_val,
                trend=regime.get("regime", "UNKNOWN"),
                volatility=regime.get("volatility", "NORMAL"),
            )
            rsi_adjusted = gate.get_rsi_signal(rsi_val, regime.get("regime", "UNKNOWN"))
            oscillator_gate_ctx = {
                "gate": gate_result,
                "rsi_adjusted": rsi_adjusted,
            }
        except Exception as e:
            log.debug(f"[AnalysisAgent] Oscillator gate error: {e}")

        # ── 6.5 Currency Strength Matrix (Day 64, wired 2026-07-22) ──
        # Runs BEFORE the sentiment step below so its (possibly richer,
        # possibly stale-cache, possibly failed) strengths dict is ready
        # to hand to SentimentEngine.final_sentiment_score() in place of
        # the crude yfinance 1-day-change proxy it used to get. Kept as
        # its own try/except so a currency-strength failure can never
        # take down the sentiment step — falls back to the old proxy.
        currency_strength_result = {}
        currency_strength_ctx    = {}
        try:
            currency_strength_result = self.currency_strength_engine.analyze()
            self.currency_strength_engine.print_summary(currency_strength_result)
            currency_strength_ctx = self.currency_strength_engine.get_ai_context(
                currency_strength_result
            )
        except Exception as e:
            log.warning(f"[AnalysisAgent] Currency Strength Engine error: {e}")

        # ── 7. Sentiment Engine ─────────────────────────────
        sentiment_ctx    = {}
        sentiment_result = {}
        conflict_result  = {}
        try:
            sent_provider    = self.sentiment_data_provider
            sent_data        = sent_provider.get_all(symbol)
            sent_provider.print_summary(sent_data)

            # Prefer the Day-64 MT5 multi-timeframe matrix (28 cross pairs,
            # real candles) over sent_data's crude yfinance 1-day %-change
            # proxy. Only fall back to the proxy if the matrix engine
            # failed above or came back with no usable strengths (e.g. all
            # 28 fetches failed) — never silently trade on an empty dict.
            currency_strengths = (
                currency_strength_ctx.get("currency_strengths")
                or sent_data["currency_strengths"]
            )

            sent_engine      = SentimentEngine()
            sentiment_result = sent_engine.final_sentiment_score(
                pair               = sent_data["pair"],
                retail_long_pct    = sent_data["retail_long_pct"],
                fg_index           = sent_data["fg_index"],
                currency_strengths = currency_strengths,
                dxy_trend          = sent_data["dxy_trend"],
                dxy_change_pct     = sent_data["dxy_change_pct"],
            )
            sent_engine.print_summary(sentiment_result)
            sentiment_ctx = sent_engine.get_ai_context(sentiment_result)
            sentiment_ctx["currency_strength_source"] = (
                "day64_mt5_matrix" if currency_strength_ctx.get("currency_strengths")
                else "yfinance_1d_proxy_fallback"
            )

            conflict_result = sent_engine.detect_conflict(
                technical_signal = signal_result.get("signal", "NO TRADE"),
                sentiment_result = sentiment_result,
            )
        except Exception as e:
            log.warning(f"[AnalysisAgent] Sentiment error: {e}")

        # ── 8. SMC Engine ───────────────────────────────────
        smc_result = {}
        smc_ctx    = {}
        try:
            smc        = SMCEngine(symbol)
            smc_result = smc.analyze()
            smc.print_summary(smc_result)
            smc_ctx    = smc.get_ai_context(smc_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] SMC Engine error: {e}")

        # ── 8.05 Zone-dependent extended modules (17-module pass, cont'd) ──
        # breaker_block / flip_zones / curve_mtf need order blocks and
        # nearest demand/supply zones, which only exist now (SMC +
        # Supply/Demand step above). Fold their votes into the
        # already-computed signal_result/signal_ctx rather than
        # re-running the whole SignalEngine.
        try:
            from analysis.extended_modules_adapter import (
                get_zone_dependent_votes, merge_zone_votes_into_signal,
            )
            zone_order_blocks = smc_result.get("h4", {}).get("order_blocks", [])
            zone_votes_ctx = get_zone_dependent_votes(
                df,
                order_blocks   = zone_order_blocks,
                nearest_demand = supply_demand_ctx.get("nearest_demand"),
                nearest_supply = supply_demand_ctx.get("nearest_supply"),
            )
            if zone_votes_ctx.get("votes"):
                signal_result = merge_zone_votes_into_signal(
                    signal_result, zone_votes_ctx["votes"]
                )
                signal_ctx = signal_engine.get_ai_context(signal_result)
        except Exception as e:
            log.debug(f"[AnalysisAgent] Zone-dependent extended modules error: {e}")

        # ── 8.1 Market Structure (Day 61 engine) ─────────────
        # Provides BOS/CHoCH/displacement context for downstream
        # MTF structure (8.95) and strategy selector.
        structure_result = {}
        structure_ctx    = {}
        try:
            structure_result = self.structure_engine.analyze(df)
            self.structure_engine.print_summary(structure_result)
            structure_ctx    = self.structure_engine.get_ai_context(structure_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] Market Structure error: {e}")

        # ── 8.2 Divergence Engine (Day 83) ───────────────────
        # RSI/MACD divergence — false-breakout filter.
        divergence_result = {}
        divergence_ctx    = {}
        try:
            divergence_result = self.divergence_engine.detect(df, indicator="rsi")
            self.divergence_engine.print_summary(divergence_result)
            divergence_ctx    = self.divergence_engine.get_ai_context(divergence_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] Divergence Engine error: {e}")

        # ── 8.3 Ichimoku Engine (Day 84) ─────────────────────
        # Cloud-based trend bias + strength.
        ichimoku_result = {}
        ichimoku_ctx    = {}
        try:
            ichimoku_result = self.ichimoku_engine.analyze(df)
            self.ichimoku_engine.print_summary(ichimoku_result)
            ichimoku_ctx    = self.ichimoku_engine.get_ai_context(ichimoku_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] Ichimoku Engine error: {e}")

        # ── 8.4 Volatility / Bollinger Squeeze (Day 85) ──────
        # Detects compression phases ahead of breakouts.
        volatility_result = {}
        volatility_ctx    = {}
        try:
            volatility_result = self.volatility_engine.analyze(df)
            self.volatility_engine.print_summary(volatility_result)
            volatility_ctx    = self.volatility_engine.get_ai_context(volatility_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] Volatility Engine error: {e}")

        # ── 8.5 Volume Profile (Day 86) ──────────────────────
        # POC / HVN / LVN + Value Area from tick volume.
        volume_profile_result = {}
        volume_profile_ctx    = {}
        try:
            volume_profile_result = self.volume_profile_eng.analyze(df)
            self.volume_profile_eng.print_summary(volume_profile_result)
            volume_profile_ctx    = self.volume_profile_eng.get_ai_context(volume_profile_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] Volume Profile error: {e}")

        # ── 8.6 SMC Advanced (Day 87) ────────────────────────
        # Mitigation Block + Inducement — ICT-style reversal signals.
        smc_advanced_result = {}
        smc_advanced_ctx    = {}
        try:
            smc_advanced_result = self.smc_advanced_engine.analyze(df)
            self.smc_advanced_engine.print_summary(smc_advanced_result)
            smc_advanced_ctx    = self.smc_advanced_engine.get_ai_context(smc_advanced_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] SMC Advanced error: {e}")

        # ── 8.65 NewsAPI Sentiment (Day 92) ──────────────────
        # Real-time financial news from Bloomberg/Reuters/etc via
        # NewsAPI.org. Adds breaking-news sentiment to complement
        # the scheduled-event awareness from Forex Factory scraper.
        news_api_result = {}
        news_api_ctx    = {}
        try:
            news_api_provider = get_news_api_provider()
            if news_api_provider.available:
                news_api_result = news_api_provider.fetch_headlines_for_pair(symbol)
                news_api_provider.print_summary(news_api_result)
                news_api_ctx    = news_api_provider.get_ai_context(news_api_result)
                # If news sentiment is very bearish, surface it as a warning
                if news_api_result.get("news_score", 0) < -40:
                    log.warning(
                        f"[AnalysisAgent] Day 92 NewsAPI: strong bearish sentiment "
                        f"on {symbol} (score={news_api_result['news_score']}) — "
                        f"AI should be cautious on longs"
                    )
            else:
                log.debug("[AnalysisAgent] NewsAPI key not set — skipping")
        except Exception as e:
            log.warning(f"[AnalysisAgent] NewsAPI provider error: {e}")

        # ── Day 63: Re-run Session with SMC context ───────────
        session_result = self.session_analyzer.analyze(
            pair        = symbol,
            smc_ctx     = smc_ctx,
            signal      = signal_result.get("signal", "NO TRADE"),
            signal_conf = signal_result.get("confidence", 0),
        )
        session_ctx = self.session_analyzer.get_ai_context(session_result)
        self.session_analyzer.print_summary(session_result)

        # ── 8.5 Intermarket / Global Macro Analysis (Day 65) ─
        intermarket_result = {}
        intermarket_ctx    = {}
        macro_fusion        = {}
        try:
            intermarket_result = self.intermarket_engine.analyze(symbol)
            self.intermarket_engine.print_summary(intermarket_result)
            intermarket_ctx = self.intermarket_engine.get_ai_context(intermarket_result)

            macro_fusion = self.intermarket_engine.fuse_with_smc(
                intermarket_result, smc_ctx=smc_ctx, session_ctx=session_ctx
            )
        except Exception as e:
            log.warning(f"[AnalysisAgent] Intermarket Engine error: {e}")

        # ── 8.95 MTF Structure (Day 88) — Internal vs External ──
        # Uses df as the "internal" timeframe. The external (HTF) tier is
        # fetched via self._h4_fetcher (reusable DataFetcher instance).
        # The MTF engine produces combined bias, alignment, conflict and
        # trade_permission fields that downstream consumers rely on.
        mtf_structure_result = {}
        mtf_structure_ctx    = {}
        try:
            df_h4 = None
            try:
                df_h4 = self._h4_fetcher.fetch_ohlcv(symbol, "H4", limit=150)
                if df_h4 is None or len(df_h4) <= 10:
                    df_h4 = None
                else:
                    log.debug(f"[AnalysisAgent] MTF H4 fetched: {len(df_h4)} candles")
            except Exception as _h4_err:
                log.debug(f"[AnalysisAgent] H4 fetch for MTF failed: {_h4_err}")
                df_h4 = None

            mtf_structure_result = self.mtf_structure_eng.analyze(
                df_external=df_h4,   # H4 data (None = fallback to internal approximation)
                df_internal=df,
            )
            self.mtf_structure_eng.print_summary(mtf_structure_result)
            mtf_structure_ctx = self.mtf_structure_eng.get_ai_context(mtf_structure_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] MTF Structure error: {e}")

        # ── 8.85 Economic Calendar (Day 96 — Trading Economics / Tradermade /
        # Finnhub all dropped; source chain is whatever EconomicCalendarAPI
        # itself falls back through, e.g. Investing/DailyFX RSS → FF scraper).
        # Blocks trades if high-impact event within ±30min.
        #
        # ── Round-5 audit fix (kept): single DECISION log line ─────────
        # Previously this block tried TradingEconomicsCalendar first and
        # EconomicCalendarAPI as fallback, each with its own log line,
        # which could look contradictory in the same cycle. Now there's
        # one source and one explicit decision line the operator can
        # grep for via "[EconCalDecision]".
        econ_calendar_result = {}
        econ_calendar_ctx    = {}
        try:
            econ_cal = EconomicCalendarAPI()
            currencies = list({symbol[:3], symbol[3:6]}) if len(symbol) >= 6 else ["USD"]
            econ_calendar_result = econ_cal.get_calendar(currencies=currencies, hours_ahead=24)
            econ_cal.print_summary(econ_calendar_result)
            econ_calendar_ctx = econ_cal.get_ai_context(econ_calendar_result)
            log.info(
                f"[EconCalDecision] {symbol} | "
                f"final_source={econ_calendar_result.get('source','?')} | "
                f"events={len(econ_calendar_result.get('events',[]) or [])} | "
                f"trade_block={econ_calendar_result.get('trade_block', False)}"
            )
        except Exception as e:
            log.warning(f"[AnalysisAgent] Economic Calendar error: {e}")
            log.warning(
                f"[EconCalDecision] {symbol} | FAILED — calendar source "
                f"down. Defaulting to conservative trade_block=True."
            )

        # ── 8.87 FRED Macro Data (Day 94 — central bank data) ─────
        # CPI, Unemployment, Treasury Yields, Fed Funds Rate, VIX.
        # Free unlimited API from St. Louis Fed.
        fred_result = {}
        fred_ctx    = {}
        try:
            fred = get_fred_api()
            if fred.available:
                fred_result = fred.get_macro_snapshot()
                fred.print_summary(fred_result)
                fred_ctx = fred.get_ai_context(fred_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] FRED macro data error: {e}")

        # ── 8.92 Retail Sentiment (Day 94/95 — OANDA → Myfxbook → synthetic) ──
        # Contrarian indicator: when 80%+ retail is long, smart money is short.
        # Day 95: now passes df for synthetic RSI fallback when both OANDA + Myfxbook fail.
        retail_sentiment_result = {}
        retail_sentiment_ctx    = {}
        try:
            sent_api = get_retail_sentiment_api()
            # Pass df for synthetic fallback (RSI-based sentiment computation)
            retail_sentiment_result = sent_api.get_sentiment(symbol, df=df)
            # Use the appropriate print/context method based on source
            src = retail_sentiment_result.get("source", "fallback")
            if "myfxbook" in src or "synthetic" in src:
                from analysis.myfxbook_sentiment import get_myfxbook_sentiment
                mfb = get_myfxbook_sentiment()
                mfb.print_summary(retail_sentiment_result)
                retail_sentiment_ctx = mfb.get_ai_context(retail_sentiment_result)
            else:
                sent_api.print_summary(retail_sentiment_result)
                retail_sentiment_ctx = sent_api.get_ai_context(retail_sentiment_result)
            # Strong contrarian signal — surface as warning
            if (retail_sentiment_result.get("contrarian_strength") == "STRONG"
                and retail_sentiment_result.get("confidence", 0) >= 70):
                log.warning(
                    f"[AnalysisAgent] Day 95 Retail: STRONG contrarian "
                    f"{retail_sentiment_result['contrarian_signal']} on {symbol} "
                    f"(retail {retail_sentiment_result['long_pct']:.0f}%L/"
                    f"{retail_sentiment_result['short_pct']:.0f}%S) "
                    f"[source={src}]"
                )
        except Exception as e:
            log.warning(f"[AnalysisAgent] Retail sentiment error: {e}")

        # ── 8.95 Correlation + Volatility Engine (Day 96) ──────────
        # Detects correlated exposure + ATR spikes → adjusts position size.
        # Day 97 fix: ensure atr column exists before passing df to engine.
        correlation_result = {}
        correlation_ctx    = {}
        try:
            corr_engine = CorrelationEngine()
            # Get live open pairs from PaperTrader
            open_pairs = []
            try:
                from execution.paper_trader import PaperTrader
                pt = PaperTrader()
                open_pairs = [t.get("pair") for t in pt.get_open_positions() if t.get("pair")]
            except Exception:
                pass
            # FIX: atr column নিশ্চিত করুন — missing হলে on-the-fly compute
            if "atr" not in df.columns and all(c in df.columns for c in ["high", "low", "close"]):
                try:
                    df["atr"] = CorrelationEngine._compute_atr(df)
                    log.debug("[AnalysisAgent] ATR computed on-the-fly for correlation engine")
                except Exception as _atr_err:
                    log.debug(f"[AnalysisAgent] ATR compute failed: {_atr_err}")
            correlation_result = corr_engine.analyze(symbol, df, open_pairs=open_pairs)
            if correlation_result.get("pair"):  # valid result guard
                corr_engine.print_summary(correlation_result)
            correlation_ctx = corr_engine.get_ai_context(correlation_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] Correlation engine error: {e}")

        # ── 8.96 Institutional Flow (Day 96 — COT + displacement) ──
        # Detects institutional direction vs retail → divergence signal.
        institutional_result = {}
        institutional_ctx    = {}
        try:
            inst_engine = self.institutional_flow_engine
            retail_long = retail_sentiment_result.get("long_pct", 50.0)
            institutional_result = inst_engine.analyze(symbol, retail_long_pct=retail_long, df=df)
            inst_engine.print_summary(institutional_result)
            institutional_ctx = inst_engine.get_ai_context(institutional_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] Institutional flow error: {e}")

        # ── 8.97 Economic Surprise Index (Day 96) ───────────────────
        # Actual vs Forecast comparison → detects market-moving surprises.
        surprise_result = {}
        surprise_ctx    = {}
        try:
            surprise_engine = EconomicSurpriseEngine()
            currency = symbol[:3] if len(symbol) >= 3 else "USD"
            surprise_result = surprise_engine.analyze(currency)
            if surprise_result.get("event_count", 0) > 0:
                surprise_engine.print_summary(surprise_result)
            surprise_ctx = surprise_engine.get_ai_context(surprise_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] Economic surprise error: {e}")

        # ── 8.975 Microstructure Engine (Day 97 — MT5 tick analysis) ──
        # Tick speed + spread expansion + volume burst + price acceleration.
        # Detects liquidity events → AI should avoid entry.
        microstructure_result = {}
        microstructure_ctx    = {}
        try:
            micro_engine = get_microstructure_engine()
            microstructure_result = micro_engine.analyze(symbol)
            micro_engine.print_summary(microstructure_result)
            microstructure_ctx = micro_engine.get_ai_context(microstructure_result)
            # Liquidity event → surface as warning
            if microstructure_result.get("liquidity_event"):
                log.warning(
                    f"[AnalysisAgent] Day 97 Microstructure: LIQUIDITY EVENT on {symbol} "
                    f"(spread={microstructure_result.get('spread_state')}, "
                    f"ticks={microstructure_result.get('tick_speed_state')}) — "
                    f"recommendation={microstructure_result.get('recommendation')}"
                )
        except Exception as e:
            log.warning(f"[AnalysisAgent] Microstructure error: {e}")

        # ── 8.978 Network Monitor (Day 97 — latency check) ──────────
        # Checks internet ping + MT5 ping. If latency > 500ms →
        # scalping disabled, only swing allowed.
        network_result = {}
        network_ctx    = {}
        try:
            net_mon = get_network_monitor()
            network_result = net_mon.check_now()
            network_ctx = net_mon.get_ai_context()
            # Only log if not GOOD (avoid log spam)
            if network_result.get("status") not in ("GOOD", "UNKNOWN"):
                net_mon.print_summary(network_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] Network monitor error: {e}")

        # ── 8.979 Forecast Engine (Day 97 — conservative extra vote) ──
        # EMA + RSI + candle body composite forecast. Weight = 10%
        # in decision fusion. NEVER overrides primary signals.
        forecast_result = {}
        forecast_ctx    = {}
        try:
            forecast_engine = get_forecast_engine()
            forecast_result = forecast_engine.forecast(df)
            forecast_engine.print_summary(forecast_result)
            forecast_ctx = forecast_engine.get_ai_context(forecast_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] Forecast engine error: {e}")

        # ── 8.97 Strategy Selector (Day 90) ──────────────────
        # Now that we have regime + mtf_bias + structure_ctx, ask the
        # selector to pick an active strategy family. The choice is
        # passed downstream to MasterAnalyst (for awareness) and to
        # MasterDecisionEngine (for position sizing + WAIT override).
        strategy_choice = {}
        try:
            strategy_choice = self.strategy_selector.select(
                regime    = regime if isinstance(regime, dict) else {},
                mtf_bias  = {"bias": mtf_bias} if isinstance(mtf_bias, str) else mtf_bias,
                structure = structure_ctx,
            )
            self.strategy_selector.print_summary(strategy_choice)
        except Exception as e:
            log.warning(f"[AnalysisAgent] Strategy Selector error: {e}")
            strategy_choice = {
                "strategy":       "WAIT",
                "active_modules": [],
                "avoid":          ["*"],
                "risk_mult":      0.0,
                "position_mult":  0.0,
                "reason":         f"selector error: {e}",
                "confidence":     0,
            }

        # ── 9. News Filter ───────────────────────────────────
        news_filter = NewsFilter()
        news_result = news_filter.check(symbol)
        news_filter.print_summary(news_result)
        news_ctx    = news_filter.get_ai_context(news_result)

        # ── 10. Classic LLM Analyst ──────────────────────────
        # BUGFIX: previously instantiated AIAnalyst() three separate times
        # for analyze()/print_summary()/get_ai_context() — one call chain
        # should use one instance, both to avoid 3x init cost (API client
        # setup, config load, etc.) and because print_summary/get_ai_context
        # were operating on throwaway objects rather than the one that
        # actually produced llm_result.
        ai_analyst = AIAnalyst()
        llm_result = ai_analyst.analyze(
            ind_ctx          = ind_ctx,
            pat_ctx          = pat_ctx,
            sr_ctx           = sr_ctx,
            regime           = regime,
            signal           = signal_result,
            mtf_bias         = mtf_bias,
            advanced_pat_ctx = advanced_pat_ctx,
            fib_ctx          = fib_ctx,
            symbol           = symbol,
        )
        ai_analyst.print_summary(llm_result)
        llm_ctx = ai_analyst.get_ai_context(llm_result)

        # ── 11. VISION AI (Day 47) ────────────────────────────
        vision_result = {}
        vision_ctx    = {}
        fusion_result = {}
        try:
            if self.chart_reader:
                log.info(f"[AnalysisAgent] 👁️ Running Vision AI for {symbol} {timeframe}")
                vision_result = self.chart_reader.capture_and_analyze(
                    symbol=symbol,
                    timeframe=timeframe,
                    quant_ctx=ind_ctx,
                )
                vision_ctx = vision_result.get("vision_ctx", {})

                fusion_result = self.chart_reader.fuse_with_quant(
                    vision_result=vision_result,
                    analysis_output={
                        "final_signal": signal_result.get("signal", "NO TRADE"),
                        "signal":       signal_result,
                        "ind_ctx":      ind_ctx,
                    }
                )
        except Exception as e:
            log.warning(f"[AnalysisAgent] Vision AI error (non-critical): {e}")

        # ── 12. MASTER ANALYST BRAIN ─────────────────────────
        master_result = {}
        master_ctx    = {}
        try:
            master = MasterAnalyst()
            master_result = master.analyze(
                symbol           = symbol,
                timeframe        = timeframe,
                ind_ctx          = ind_ctx,
                pat_ctx          = pat_ctx,
                sr_ctx           = sr_ctx,
                regime           = regime,
                mtf_bias         = mtf_bias,
                signal           = signal_result,
                sentiment_ctx    = sentiment_ctx,
                news_ctx         = news_ctx,
                memory_ctx       = memory_ctx or {},
                bias_ctx         = bias_ctx,
                smc_ctx          = smc_ctx,
                fib_ctx          = fib_ctx,
                advanced_pat_ctx = advanced_pat_ctx,
                vision_ctx       = vision_ctx,
                session_ctx      = session_ctx,        # ← Day 63
                intermarket_ctx  = intermarket_ctx,    # ← Day 65
                # Round-13: feed the classic LLM analyst's verdict in so
                # MasterAnalyst reconciles with it instead of silently
                # producing an unexplained second, possibly-conflicting
                # opinion from an independent LLM call (see llm_ctx above).
                classic_llm_ctx  = llm_ctx,
                # Day 90 — six new analyzers + strategy selector
                divergence_ctx     = divergence_ctx,
                ichimoku_ctx       = ichimoku_ctx,
                volatility_ctx     = volatility_ctx,
                volume_profile_ctx = volume_profile_ctx,
                smc_advanced_ctx   = smc_advanced_ctx,
                mtf_structure_ctx  = mtf_structure_ctx,
                strategy_ctx       = strategy_choice,
                # Day 92 — NewsAPI real-time news sentiment
                news_api_ctx       = news_api_ctx,
                # Day 94 — Institutional grade APIs
                econ_calendar_ctx  = econ_calendar_ctx,
                fred_ctx           = fred_ctx,
                retail_sentiment_ctx = retail_sentiment_ctx,
            )
            master.print_summary(master_result)
            master_ctx = master.get_ai_context(master_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] MasterAnalyst error: {e}")
            # Day 81+ hotfix (Barrier 2): when MasterAnalyst raises (LLM
            # unavailable, rate-limited, JSON parse error, etc.), master_ctx
            # stayed as {} which downstream code reads as master_signal=None
            # → final_signal never gets overridden from rule signal.  Populate
            # a safe default using the rule-engine signal so downstream
            # DecisionAgent + trader.py see a real signal.
            _rule_sig = (signal_result or {}).get("signal", "WAIT")
            _rule_conf = (signal_result or {}).get("confidence", 0)
            master_ctx = {
                "master_signal":     _rule_sig,
                "master_confidence": _rule_conf,
                "master_entry":      (signal_result or {}).get("entry"),
                "master_sl":         (signal_result or {}).get("sl"),
                "master_tp1":        (signal_result or {}).get("tp"),
                "master_tp2":        None,
                "master_story":      f"LLM unavailable — rule engine fallback {_rule_sig} ({_rule_conf}%)",
                "master_risks":      ["LLM analysis unavailable — rule engine signal only"],
                "master_critique":   "",
                # P0 fix (audit C6): audit trail.
                "confidence_chain":  [("init", 0), ("master_analyst_fallback", _rule_conf)],
                # P0 fix (audit C7): flag so decision_agent zeros the master vote.
                "_llm_unavailable":  True,
            }
            log.info(
                f"[AnalysisAgent] MasterAnalyst fallback → master_signal={_rule_sig} "
                f"conf={_rule_conf}% (rule-engine signal used as master)"
            )

        # ── Final Signal Resolution ───────────────────────────
        # Day 81+ hotfix: signal_result could be None if SignalEngine
        # raised inside its try block — use defensive .get() to avoid
        # 'NoneType' object is not subscriptable crash.
        if not isinstance(signal_result, dict):
            log.error(f"[AnalysisAgent] signal_result is {type(signal_result).__name__}, expected dict — using NO TRADE")
            signal_result = {"signal": "NO TRADE", "confidence": 0}
        final_signal = signal_result.get("signal", "NO TRADE")

        # ── ARCHITECTURAL FIX (institutional refactor) ───────────────
        # `execution_filters` records EVERY execution-layer gate verdict
        # (news, session, fusion, risk, spread, broker) WITHOUT touching
        # the analysis-layer `final_signal`. Previously, news/session
        # blocks would set `final_signal = "NO TRADE"`, destroying the
        # analysis verdict and causing downstream consumers to see
        # "WAIT 0%" instead of "BUY 79% (analysis) → BLOCKED by news".
        #
        # The analysis layer now ALWAYS produces its verdict (BUY/SELL/WAIT).
        # Execution gates record their verdict here. The TradePermission
        # layer is the SINGLE authority on whether to actually execute.
        # ──────────────────────────────────────────────────────────────
        execution_filters: Dict[str, Any] = {
            # Each gate adds: {"blocked": bool, "reason": str, "details": ...}
        }

        # Day 81+ AGGRESSIVE TEST_MODE: If TEST_MODE is true and the rule engine
        # has a tradeable signal (BUY/SELL/STRONG_BUY/STRONG_SELL with conf >= 10),
        # USE IT DIRECTLY. Skip ALL gates - MasterAnalyst/news/session/conflict.
        # This is the "just trade something" mode for verifying MT5 execution.
        # Day 81+ hotfix #2: lowered threshold from 30 → 10 so weak BUY/SELL
        # signals also flow through.  Without this, the bot stays in WAIT
        # when market is choppy and rule engine confidence is 15-25%.
        _test_mode = False
        try:
            from config import TEST_MODE
            _test_mode = bool(TEST_MODE)
        except Exception:
            pass

        rule_sig_raw = signal_result.get("signal", "WAIT")
        rule_conf = signal_result.get("confidence", 0)
        rule_sig_normalized = rule_sig_raw
        if "STRONG_BUY" in str(rule_sig_raw):
            rule_sig_normalized = "BUY"
        elif "STRONG_SELL" in str(rule_sig_raw):
            rule_sig_normalized = "SELL"

        if _test_mode and rule_sig_normalized in ("BUY", "SELL") and rule_conf >= 10:
            final_signal = rule_sig_normalized
            log.info(
                f"[AnalysisAgent] -> {final_signal} "
                f"(TEST_MODE AGGRESSIVE: Rule={rule_sig_raw} {rule_conf}% — "
                f"BYPASSING all gates for MT5 verification)"
            )

            # ── /DEBUG ──────────────────────────────────────────────

            # Skip ALL remaining gates - go straight to return
            # Build minimal context needed downstream
            return {
                "df":                df,
                "pat_ctx":           pat_ctx,
                "advanced_patterns": adv_patterns,
                "advanced_pat_ctx":  advanced_pat_ctx,
                "sr_result":         sr_res,
                "sr_ctx":            sr_ctx,
                "liquidity_ctx":     liquidity_ctx,
                "fib_ctx":           fib_ctx,
                "bias_result":       bias_result,
                "bias_ctx":          bias_ctx,
                "signal":            signal_result,
                "signal_ctx":        signal_ctx,
                "trendline_ctx":     trendline_ctx,
                "supply_demand_ctx": supply_demand_ctx,
                "volume_confirm_ctx":volume_confirm_ctx,
                "oscillator_gate_ctx":oscillator_gate_ctx,
                "llm":               llm_result,
                "llm_ctx":           llm_ctx,
                "news":              news_result,
                "news_ctx":          news_ctx,
                "sentiment":         sentiment_result,
                "sentiment_ctx":     sentiment_ctx,
                "conflict":          conflict_result,
                "smc":               smc_result,
                "smc_ctx":           smc_ctx,
                "vision":            vision_result,
                "vision_ctx":        vision_ctx,
                "vision_fusion":     fusion_result,
                "session":           session_result,
                "session_ctx":       session_ctx,
                "intermarket":       intermarket_result,
                "intermarket_ctx":   intermarket_ctx,
                "currency_strength_ctx": currency_strength_ctx,
                "macro_fusion":      macro_fusion,
                "master":            master_result,
                "master_ctx":        master_ctx,
                "news_intelligence": news_intel_ctx,
                "confluence":        confluence_ctx,
                "feature_vector":    feature_vector_ctx,
                "ml_prediction":     ml_prediction_ctx,
                "ensemble":          ensemble_ctx,
                "rl_agent":          rl_ctx,
                "master_decision":   master_decision_ctx,
                # Day 90 — six new analyzer contexts + strategy + structure
                "structure":          structure_result,
                "structure_ctx":      structure_ctx,
                "divergence":         divergence_result,
                "divergence_ctx":     divergence_ctx,
                "ichimoku":           ichimoku_result,
                "ichimoku_ctx":       ichimoku_ctx,
                "volatility":         volatility_result,
                "volatility_ctx":     volatility_ctx,
                "volume_profile":     volume_profile_result,
                "volume_profile_ctx": volume_profile_ctx,
                "smc_advanced":       smc_advanced_result,
                "smc_advanced_ctx":   smc_advanced_ctx,
                "news_api":           news_api_result,
                "news_api_ctx":       news_api_ctx,
                "econ_calendar":      econ_calendar_result,
                "econ_calendar_ctx":  econ_calendar_ctx,
                "fred_macro":         fred_result,
                "fred_ctx":           fred_ctx,
                "retail_sentiment":   retail_sentiment_result,
                "retail_sentiment_ctx": retail_sentiment_ctx,
                "mtf_structure":      mtf_structure_result,
                "mtf_structure_ctx":  mtf_structure_ctx,
                # Round-22 audit fix: wire 6 dead engine outputs into return dict.
                # Previously these were computed every cycle (real API calls,
                # real CPU cost) but their results were silently discarded —
                # never reached the return dict, never reached decision_agent
                # or master_analyst. This was a multi-generation bug (present
                # since at least the hotfix_archive_20260627 version).
                "correlation_ctx":       correlation_ctx,
                "institutional_ctx":     institutional_ctx,
                "surprise_ctx":          surprise_ctx,
                "microstructure_ctx":    microstructure_ctx,
                "network_ctx":           network_ctx,
                "forecast_ctx":          forecast_ctx,
                "strategy":           strategy_choice,
                "final_signal":      final_signal,
                "execution_filters": {},  # TEST_MODE bypass — no execution gates applied
                "test_mode_bypass":  True,  # Flag for downstream debugging
            }

        # Day 63: Session dead zone / strategy gate
        # ARCHITECTURAL FIX: session is an EXECUTION gate. Don't overwrite
        # `final_signal` — record the block in `execution_filters` and let
        # the analysis verdict flow through. The TradePermission layer
        # already enforces session quality (see risk/trade_permission.py
        # L139-152 + L210-242), so this early block was redundant AND
        # destroyed the analysis verdict.
        elif not session_result.get("session_trade_allowed", True):
            execution_filters["session"] = {
                "blocked": True,
                "reason": (
                    f"Session gate: {session_ctx['current_session']} — "
                    f"{session_ctx['session_strategy']}"
                ),
                "session": session_ctx.get("current_session"),
                "strategy": session_ctx.get("session_strategy"),
            }
            log.info(
                f"[AnalysisAgent] Execution filter: session blocked "
                f"({session_ctx['current_session']}) — analysis verdict "
                f"{final_signal} PRESERVED, will be gated by TradePermission"
            )

        elif not session_ctx.get("fusion_allowed", True):
            execution_filters["fusion"] = {
                "blocked": True,
                "reason": (
                    f"Fusion gate: SMC fusion rejected for "
                    f"{session_ctx.get('current_session', '?')} "
                    f"(score={session_ctx.get('fusion_score', 0)}/100, "
                    f"grade={session_ctx.get('fusion_grade', 'N/A')})"
                ),
                "fusion_score": session_ctx.get("fusion_score", 0),
                "fusion_grade": session_ctx.get("fusion_grade"),
            }
            log.info(
                f"[AnalysisAgent] Execution filter: fusion blocked "
                f"({session_ctx.get('current_session', '?')}) — analysis verdict "
                f"{final_signal} PRESERVED, will be gated by TradePermission"
            )

        # ARCHITECTURAL FIX: news is an EXECUTION gate. Same treatment.
        # TradePermission L111-122 already enforces this; the early
        # `final_signal = "NO TRADE"` was redundant.
        elif not news_result["trade_allowed"]:
            execution_filters["news"] = {
                "blocked": True,
                "reason": f"News block: {news_result.get('reason', 'unknown')}",
                "risk_level": news_result.get("risk_level"),
                "flagged_events": news_result.get("flagged_events", []),
            }
            log.info(
                f"[AnalysisAgent] Execution filter: news blocked "
                f"({news_result.get('reason', '?')}) — analysis verdict "
                f"{final_signal} PRESERVED, will be gated by TradePermission"
            )

        elif conflict_result.get("has_conflict") and sentiment_result.get("confidence", 0) >= 70:
            _apply_confidence_penalty(signal_result, 8, "sentiment_conflict", "analysis")
            log.info(
                "[AnalysisAgent] Confidence penalty: -8% due to sentiment conflict; "
                f"preserving {final_signal} for downstream evaluation"
            )

        elif fusion_result.get("has_conflict") and fusion_result.get("adjusted_conf", 100) < 45:
            _apply_confidence_penalty(signal_result, 10, "fusion_conflict", "analysis")
            log.info(
                "[AnalysisAgent] Confidence penalty: -10% due to fusion conflict; "
                f"preserving {final_signal} for downstream evaluation"
            )

        elif master_ctx.get("master_signal") in ("BUY", "SELL", "WAIT", "STRONG_BUY", "STRONG_SELL"):
            ma_signal    = master_ctx["master_signal"]
            # CRITICAL FIX (audit Priority 1 — Consensus Lock):
            # A single RuleEngine signal used to be able to override a
            # consensus WAIT (SignalFusion/MasterDecision) on its own
            # whenever confidence >= 30%:
            #     SignalFusion consensus = WAIT -> Confidence Override -> BUY
            # That is a critical-severity bug: a multi-layer consensus
            # WAIT should never be overridable by one layer acting alone.
            # It is now gated by EntrySafetyFilters.evaluate_override_gate(),
            # which only allows the override when ALL of these hold at once:
            #   - rule confidence clears a stricter 60% floor (was 30%)
            #   - HTF bias agrees with the rule signal's direction
            #   - the breakout level is confirmed by a CLOSED candle
            #   - no unresolved liquidity-sweep risk sits in the way
            #   - price isn't sitting 10-20 pips from the opposing S/R level
            #   - risk manager has approved the trade
            rule_sig = signal_result.get("signal", "WAIT")
            rule_conf = signal_result.get("confidence", 0)
            # Normalize STRONG_BUY → BUY, STRONG_SELL → SELL for the final signal
            rule_sig_normalized = rule_sig
            if "STRONG_BUY" in str(rule_sig):
                rule_sig_normalized = "BUY"
            elif "STRONG_SELL" in str(rule_sig):
                rule_sig_normalized = "SELL"

            if ma_signal == "WAIT" and rule_sig_normalized in ("BUY", "SELL"):
                from core.entry_safety_filters import get_entry_safety_filters
                safety = get_entry_safety_filters()
                gate = safety.evaluate_override_gate(
                    master_signal=ma_signal,
                    rule_signal=rule_sig_normalized,
                    rule_confidence=rule_conf,
                    mtf_bias={"bias": mtf_bias} if isinstance(mtf_bias, str) else mtf_bias,
                    sr_ctx=sr_ctx,
                    liquidity_ctx=liquidity_ctx,
                    df=df,
                    risk_approved=True,  # final risk gate still runs downstream in AITrader
                )

                if gate.allowed:
                    final_signal = rule_sig_normalized
                    # Trend-exhaustion penalty (Rule 5) still applies even
                    # when the override is otherwise allowed.
                    exhaustion_penalty = gate.details.get("exhaustion_penalty", 0.0)
                    if exhaustion_penalty:
                        _apply_confidence_penalty(signal_result, exhaustion_penalty, "trend_exhaustion", "analysis")
                    log.info(
                        f"[AnalysisAgent] -> {final_signal} (Rule signal: {rule_sig} {rule_conf}% conf, "
                        f"master WAIT — consensus-lock override APPROVED: {gate.reason})"
                    )
                else:
                    # Consensus lock holds — WAIT stands. Rule signal alone
                    # is not enough to force a trade against the 4-layer
                    # consensus.
                    final_signal = "WAIT"
                    log.info(
                        f"[AnalysisAgent] -> WAIT (Rule signal: {rule_sig} {rule_conf}% conf, "
                        f"master WAIT — consensus-lock override BLOCKED: {gate.reason})"
                    )
            else:
                if ma_signal == "WAIT":
                    if final_signal in ("BUY", "SELL"):
                        _apply_confidence_penalty(signal_result, 6, "master_wait", "analysis")
                        log.info(
                            f"[AnalysisAgent] MasterAnalyst WAIT; keeping {final_signal} with -6% penalty"
                        )
                    else:
                        final_signal = "WAIT"
                        log.info("[AnalysisAgent] -> WAIT (MasterAnalyst override with no directional fallback)")
                else:
                    final_signal = ma_signal
                    log.info(f"[AnalysisAgent] -> {final_signal} (MasterAnalyst override)")


        # ── Day 66: News Intelligence integration ────────────────────
        # After MasterAnalyst decides, run NewsIntelligence to:
        #   1. BLOCK the trade if pair is in a high-impact event window
        #   2. ADJUST confidence based on news bias alignment
        #
        # Day 81+ hotfix: previously TEST_MODE alone was enough to skip
        # the news block. The news intelligence module fetches
        # central-bank events from a hardcoded schedule which can produce
        # false-positive "CPI in 0min" blocks even when the actual
        # ForexFactory calendar is empty. This was blocking every trade
        # during certain GMT hours.
        #
        # Day 137 safety fix (real-money loss postmortem — GBPCAD
        # 2026-07-20): `_skip_news_block` was being OR'd together with
        # `_is_false_block` below, so TEST_MODE bypassed CONFIRMED real
        # news-window blocks too, not just false positives — this session
        # bypassed a genuine "CPI 11min ago — post-event block window"
        # block 11 times and one of those trades (GBPCAD) went on to lose
        # money in exactly the post-CPI volatility spike the block existed
        # to prevent. TEST_MODE is for MT5 connectivity/integration
        # testing friction, not for overriding a real detected news event.
        # It no longer participates in the block/bypass decision at all —
        # only `_is_false_block` (the live-calendar-confirmed false
        # positive check a few lines below) can bypass a news block now.
        news_intel_ctx = {}

        try:
            from intelligence.news_ai import get_news_intelligence
            # Use the symbol passed in market_output (or fallback to EURUSD)
            symbol = market_output.get("symbol", "EURUSD") if isinstance(market_output, dict) else "EURUSD"
            news_ai = get_news_intelligence()
            # Refresh pair universe if needed
            try:
                from config import SYMBOLS
                news_ai.set_pairs(list(SYMBOLS))
            except Exception:
                pass

            # 1. Block check
            block_check = news_ai.should_block_trade(symbol)

            # ── HOTFIX Day37: hardcoded_fallback false positive ──────────────
            # Live calendar clear হলে hardcoded schedule-এর block ignore করো
            #
            # Day 101 hotfix: this read `econ_calendar_result.get('trade_blocked', ...)`
            # but EconomicCalendarAPI.get_calendar() actually returns the key
            # "trade_block" (no "ed") — see economic_calendar_api.py:205, and
            # every other read site in this same file (lines ~619/640/655)
            # already used the correct spelling. Because the key never matched,
            # `.get()` silently fell back to its default (False) on every single
            # call — `_live_cal_blocked` was ALWAYS False, regardless of what the
            # live calendar actually said. That broke the intent of this hotfix:
            # instead of "bypass news_ai's hardcoded block ONLY IF the live
            # calendar confirms nothing is actually near", it became "bypass
            # news_ai's hardcoded block whenever its source is 'hardcoded'" —
            # the live-calendar safety check was a no-op. This is a separate,
            # standing bug from the TEST_MODE bypass a few lines below (which
            # is intentional and logged); this one silently fired even in live
            # trading whenever `intelligence.news_ai` fell back to its hardcoded
            # schedule, regardless of whether the real calendar agreed.
            _cal_source = econ_calendar_result.get('source', '') if isinstance(econ_calendar_result, dict) else ''
            _live_cal_blocked = econ_calendar_result.get('trade_block', False) if isinstance(econ_calendar_result, dict) else False
            _is_false_block = (
                block_check.get('blocked', False)
                and 'hardcoded' in _cal_source
                and not _live_cal_blocked
            )
            # ──────────────────────────────────────────────────────────────────

            if block_check["blocked"] and final_signal in ("BUY", "SELL"):
                if _is_false_block:
                    log.warning(
                        f"[AnalysisAgent] News block detected ({block_check['reason']}) — "
                        f"BYPASSED (live calendar confirms no real event; hardcoded "
                        f"schedule false positive). Trade continues."
                    )
                    news_intel_ctx = {
                        "blocked": False,
                        "block_reason": f"{block_check['reason']} (confirmed false positive, bypassed)",
                    }
                else:
                    # ARCHITECTURAL FIX: Don't overwrite final_signal.
                    # Day 66 NewsIntelligence is an EXECUTION gate.
                    # Record in execution_filters; analysis verdict preserved.
                    log.warning(
                        f"[AnalysisAgent] Execution filter: Day 66 News block "
                        f"({block_check['reason']}) — analysis verdict "
                        f"{final_signal} PRESERVED, will be gated by TradePermission"
                    )
                    execution_filters["news_intelligence"] = {
                        "blocked": True,
                        "reason": block_check["reason"],
                        "source": "day66_news_intelligence",
                    }
                    news_intel_ctx = {
                        "blocked": True,
                        "block_reason": block_check["reason"],
                    }
            else:
                # 2. Confidence adjustment
                if final_signal in ("BUY", "SELL"):
                    # Get base confidence from master_ctx
                    base_conf = float(master_ctx.get("master_confidence", 50) or 50)
                    adjustment = news_ai.adjust_confidence(symbol, base_conf, final_signal)
                    news_intel_ctx = {
                        "blocked": False,
                        "news_bias": adjustment["news_bias"],
                        "confidence_change": adjustment["change"],
                        "adjustment_reason": adjustment["reason"],
                        "adjusted_confidence": adjustment["adjusted_confidence"],
                    }
                    if adjustment["change"] != 0:
                        log.info(
                            f"[AnalysisAgent] Day 66 news confidence adjustment: "
                            f"{adjustment['change']:+.0f} ({adjustment['reason']})"
                        )
                        # Update master_ctx confidence so downstream DecisionAgent sees it
                        try:
                            master_ctx["master_confidence"] = adjustment["adjusted_confidence"]
                            _track_confidence(master_ctx, "news_ai", adjustment["adjusted_confidence"])
                        except Exception:
                            pass
                else:
                    news_intel_ctx = {"blocked": False, "news_bias": "N/A"}

            # Attach full report for dashboard / journal
            try:
                latest = news_ai.latest_report()
                if latest is not None:
                    news_intel_ctx["next_high_impact_event"] = latest.next_high_impact_event
                    news_intel_ctx["sentiment_summary"] = latest.sentiment_summary
                    news_intel_ctx["pair_biases"] = latest.pair_biases
                    news_intel_ctx["blocked_pairs"] = latest.blocked_pairs
            except Exception:
                pass
        except Exception as e:
            log.warning(f"[AnalysisAgent] Day 66 NewsIntelligence failed: {e}")
            news_intel_ctx = {"error": str(e)}

        # ── Day 67: Multi-Factor Confluence Engine ────────────────────
        # Run the confluence engine over ALL 7 analysis factors. This produces
        # a weighted score, runs validation gates (5+ factor rule, contradiction
        # detector, news block, etc.), and produces a final calibrated decision.
        confluence_ctx = {}
        try:
            from intelligence.confluence_engine import get_confluence_engine
            symbol = market_output.get("symbol", "EURUSD") if isinstance(market_output, dict) else "EURUSD"
            timeframe = market_output.get("timeframe", "15m") if isinstance(market_output, dict) else "15m"

            # Build a unified analysis dict for the confluence engine
            unified_analysis = {
                "smc_ctx": smc_ctx,
                "session_ctx": session_ctx,
                "intermarket_ctx": intermarket_ctx,
                "currency_strength_ctx": currency_strength_ctx,
                "sentiment_ctx": sentiment_ctx,
                "news_intelligence": news_intel_ctx,
                "signal": signal_result,
                "bias_ctx": bias_ctx,
            }

            # ── /DEBUG ──────────────────────────────────────────────────────

            engine = get_confluence_engine()
            # Pull news-blocked pairs for the validator
            news_blocked = {}
            try:
                latest = news_ai.latest_report() if 'news_ai' in dir() else None
                if latest is not None:
                    news_blocked = latest.blocked_pairs
            except Exception:
                pass

            decision = engine.evaluate(
                pair=symbol,
                timeframe=timeframe,
                analysis_out=unified_analysis,
                news_blocked_pairs=news_blocked,
                risk_approved=True,  # risk check happens downstream in AITrader
                correlation_blocked=False,  # same
            )

            confluence_ctx = decision.to_dict()

            # Day 67 override: only block if confluence says AVOID (not B or higher)
            # Made more permissive: B quality trades are now allowed through.
            #
            # Day 137 safety fix (real-money loss postmortem — GBPCAD
            # 2026-07-20): quality=AVOID used to be treated as just another
            # soft penalty (-12%), same tier as a routine quality demotion.
            # That let a trade the Confluence engine had ITSELF explicitly
            # flagged as "do not take" go on to execute, because a single
            # downstream layer (MasterAnalyst) separately cleared the
            # confidence floor and overrode the rest of the pipeline.
            # AVOID is not a quality demotion — it's the engine's own verdict
            # that this setup should not be traded. It must hard-block, and
            # unlike softer quality tiers below, TEST_MODE must NOT bypass it:
            # TEST_MODE exists to unblock false-positive/off-hours friction
            # for MT5 connectivity testing, not to override an explicit
            # do-not-trade verdict from the confluence engine itself.
            if not decision.should_trade and final_signal in ("BUY", "SELL"):
                if decision.setup_quality == "AVOID":
                    final_signal = "NO TRADE"
                    execution_filters["confluence_avoid"] = {
                        "blocked": True,
                        "reason": decision.block_reason or "confluence quality AVOID",
                    }
                    log.warning(
                        f"[AnalysisAgent] Day 67 Confluence: HARD BLOCK — quality=AVOID "
                        f"({decision.block_reason or 'failed validation'}) — signal downgraded "
                        f"to NO TRADE (not bypassed by TEST_MODE)"
                    )
                elif _test_mode:
                    log.info(
                        f"[AnalysisAgent] Day 67 Confluence: {final_signal} quality={decision.setup_quality} — "
                        f"BYPASSED (TEST_MODE=true)"
                    )
                else:
                    _apply_confidence_penalty(signal_result, 6, "confluence_quality", "analysis")
                    log.info(
                        f"[AnalysisAgent] Day 67 Confluence: {final_signal} allowed with -6% penalty "
                        f"(quality={decision.setup_quality}, factors={decision.aligned_factors}/{decision.total_factors})"
                    )
            elif decision.should_trade and decision.direction in ("BUY", "SELL"):
                # Confluence confirms — use its calibrated confidence
                final_signal = decision.direction
                log.info(
                    f"[AnalysisAgent] Day 67 Confluence confirms {decision.direction} "
                    f"| Quality={decision.setup_quality} | Conf={decision.confidence:.0f}% | "
                    f"Factors={decision.aligned_factors}/{decision.total_factors} | "
                    f"Net={decision.net_score:+.1f}"
                )
                try:
                    master_ctx["master_confidence"] = decision.confidence
                    _track_confidence(master_ctx, "confluence", decision.confidence)
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"[AnalysisAgent] Day 67 ConfluenceEngine failed: {e}")
            confluence_ctx = {"error": str(e)}

        # ── Day 68: Feature Engineering Layer ─────────────────────────
        # Build a ~110-feature vector from the current market state + all
        # analysis contexts. Persist to the FeatureStore for ML training.
        # The feature vector is attached to the output for downstream ML
        # inference (Day 69+).
        feature_vector_ctx: Dict[str, Any] = {}
        full_feature_vector: Dict[str, float] = {}
        try:
            from ml.feature_engineer import get_feature_engineer
            from ml.feature_store import get_feature_store
            symbol = market_output.get("symbol", "EURUSD") if isinstance(market_output, dict) else "EURUSD"
            timeframe = market_output.get("timeframe", "15m") if isinstance(market_output, dict) else "15m"

            engineer = get_feature_engineer()
            # LEAKAGE FIX: this dict used to include "signal" (signal_result),
            # "confluence" (confluence_ctx), and "master_ctx" — but the label
            # saved a few lines below is derived from `final_signal`, which by
            # THIS point in the pipeline already IS the master/confluence
            # decision layered on top of signal_result. Training on features
            # built from the same decision that produced the label is
            # textbook label leakage: any model/RL agent trained on this
            # feature store will show inflated backtest accuracy that will
            # not reproduce live, because at live-inference time the
            # equivalent feature won't yet encode the answer.
            # Kept only genuinely pre-decision, market-state context.
            # Also dropped the old `master_ctx.get("llm", {})` line — nothing
            # in this file ever writes master_ctx["llm"], so it always
            # evaluated to `{}` (dead code).
            unified_for_features = {
                "smc_ctx": smc_ctx,
                "session_ctx": session_ctx,
                "intermarket_ctx": intermarket_ctx,
                "currency_strength_ctx": currency_strength_ctx,
                "sentiment_ctx": sentiment_ctx,
                "news_intelligence": news_intel_ctx,
                "bias_ctx": bias_ctx,
                "fib_ctx": fib_ctx,
                "sr_ctx": sr_ctx,
                "advanced_pat_ctx": advanced_pat_ctx,
                "mtf_bias": market_output.get("mtf_bias") if isinstance(market_output, dict) else None,
            }
            full_feature_vector = engineer.build_feature_vector(
                df=df, analysis_out=unified_for_features, pair=symbol, timeframe=timeframe,
            )
            feature_vector_ctx = {
                "feature_count": len(full_feature_vector),
                "features_preview": dict(list(full_feature_vector.items())[:10]),
                "pair": symbol,
                "timeframe": timeframe,
            }
            log.info(
                f"[AnalysisAgent] Day 68 Feature Engineering: {len(full_feature_vector)} features generated for {symbol} {timeframe}"
            )

            # Persist to feature store (for ML training later)
            try:
                store = get_feature_store()
                label = None
                if final_signal == "BUY":
                    label = 1
                elif final_signal == "SELL":
                    label = 0
                store.save_features(
                    pair=symbol, timeframe=timeframe, features=full_feature_vector, label=label,
                )
            except Exception as e:
                log.debug(f"[Day 68] feature store save failed: {e}")
        except Exception as e:
            log.warning(f"[AnalysisAgent] Day 68 FeatureEngineering failed: {e}")
            feature_vector_ctx = {"error": str(e)}

        # ── Day 69: ML Model Prediction (Ensemble) ────────────────────
        # Run the ML predictor on the feature vector. If models are trained,
        # this produces a BUY/SELL/WAIT probability that adjusts the final
        # confidence. If no models are trained yet, it returns NOT_READY
        # and the agent falls back to rule-based logic.
        ml_prediction_ctx: Dict[str, Any] = {}
        try:
            from ml.model_predictor import get_model_predictor
            predictor = get_model_predictor()
            ml_pred = predictor.predict(
                features=full_feature_vector, pair=symbol, timeframe=timeframe,
            )
            ml_prediction_ctx = ml_pred

            if ml_pred.get("prediction") != "NOT_READY" and ml_pred.get("models_used", 0) > 0:
                ml_dir = ml_pred["prediction"]
                ml_proba = ml_pred["probability"]
                agreement = ml_pred.get("model_agreement", "0/0")

                log.info(
                    f"[AnalysisAgent] Day 69 ML ensemble: {ml_dir} "
                    f"| prob={ml_proba:.2f} | agreement={agreement} | "
                    f"models={ml_pred['models_used']}"
                )
        except Exception as e:
            log.warning(f"[AnalysisAgent] Day 69 ML prediction failed: {e}")
            ml_prediction_ctx = {"error": str(e)}

        # ── Day 70: AI Brain Fusion Layer (Ensemble Engine) ───────────
        # The culmination of Days 60-69. Fuses ALL intelligence layers:
        #   - XGBoost + RandomForest + LSTM (Day 69 ML models)
        #   - Rule Engine signal (Day 67 Confluence)
        #   - MasterAnalyst LLM (Day 42)
        # into a single institutional-grade decision with:
        #   - Voting (4/4=FULL, 3/4=HALF, 2/4=WAIT, <2=NO_TRADE)
        #   - Weighted confidence fusion (regime + performance adjusted)
        #   - Conflict detection + abstain capability
        #   - Position size multiplier
        ensemble_ctx: Dict[str, Any] = {}
        try:
            from ml.ensemble import get_ensemble_engine
            engine = get_ensemble_engine()

            # Gather inputs for the ensemble
            # Normalize STRONG_BUY/STRONG_SELL → BUY/SELL
            _fs = final_signal
            if "STRONG_BUY" in str(_fs):
                _fs = "BUY"
            elif "STRONG_SELL" in str(_fs):
                _fs = "SELL"
            rule_sig = _fs if _fs in ("BUY", "SELL") else "WAIT"
            # Use master_confidence if available, otherwise use signal confidence, otherwise 50
            rule_conf = float(master_ctx.get("master_confidence", 0) or 0)
            if rule_conf <= 0:
                rule_conf = float(signal_result.get("confidence", 0) or 0)
            if rule_conf <= 0 and rule_sig in ("BUY", "SELL"):
                rule_conf = 50.0  # minimum viable confidence
            master_sig = (master_ctx.get("master_signal") or "WAIT") if isinstance(master_ctx, dict) else "WAIT"
            master_conf = float(master_ctx.get("master_confidence", 50) or 50) if isinstance(master_ctx, dict) else 50.0
            # FIX: this used to reassign `regime` (the MarketRegimeDetector
            # dict from earlier in this function) to a plain macro-regime
            # STRING for this one ensemble call. That silently turned
            # `regime` into a non-dict for the rest of the function,
            # including the later MasterDecisionEngine call ~180 lines
            # down, which then always saw `isinstance(regime, dict) ==
            # False` and passed regime=None. Using a distinct name keeps
            # the original regime dict intact for everything downstream.
            macro_regime_label = (intermarket_ctx.get("macro_regime") or "UNKNOWN") if isinstance(intermarket_ctx, dict) else "UNKNOWN"

            # Run the ensemble engine
            ensemble_decision = engine.decide(
                pair=symbol,
                timeframe=timeframe,
                ml_prediction=ml_prediction_ctx,
                rule_signal=rule_sig,
                rule_confidence=rule_conf,
                master_signal=master_sig,
                master_confidence=master_conf,
                regime=macro_regime_label,
            )
            ensemble_ctx = ensemble_decision.to_dict()

            # ── Day 70 override: the ensemble is the FINAL decision ──
            # Made more permissive: only block on ABSTAIN, not on WAIT.
            # WAIT from ensemble now allows the original signal to proceed
            # if it had decent confidence from MasterAnalyst.
            if ensemble_decision.abstained:
                if _test_mode:
                    log.info(
                        f"[AnalysisAgent] Day 70 Ensemble ABSTAINED: "
                        f"{ensemble_decision.abstain_reason} — "
                        f"BYPASSED (TEST_MODE=true), keeping {final_signal}"
                    )
                else:
                    log.warning(
                        f"[AnalysisAgent] Day 70 Ensemble ABSTAINED: "
                        f"{ensemble_decision.abstain_reason}"
                    )
                    if final_signal in ("BUY", "SELL"):
                        _apply_confidence_penalty(signal_result, 10, "ensemble_abstain", "analysis")
                        log.info(
                            f"[AnalysisAgent] Ensemble abstain; keeping {final_signal} with -10% penalty"
                        )
            elif ensemble_decision.decision == "WAIT":
                # Don't automatically block — only block if confidence is very low
                # Day 81+ hotfix: In TEST_MODE, never let Ensemble WAIT block a trade
                if ensemble_decision.confidence < 40 and not _test_mode:
                    if final_signal in ("BUY", "SELL"):
                        _apply_confidence_penalty(signal_result, 8, "ensemble_wait", "analysis")
                        log.info(
                            f"[AnalysisAgent] Day 70 Ensemble → WAIT with -8% penalty "
                            f"(conf {ensemble_decision.confidence:.0f}% < 40%)"
                        )
                else:
                    # WAIT with decent confidence OR TEST_MODE — let the original signal pass
                    log.info(
                        f"[AnalysisAgent] Day 70 Ensemble WAIT but conf={ensemble_decision.confidence:.0f}% — "
                        f"allowing original signal {final_signal}"
                        + (" (TEST_MODE)" if _test_mode else "")
                    )
            elif ensemble_decision.decision in ("BUY", "SELL"):
                # Ensemble confirms a trade — use its fused confidence
                final_signal = ensemble_decision.decision
                # Update master confidence to the ensemble's fused confidence
                try:
                    master_ctx["master_confidence"] = ensemble_decision.confidence
                    _track_confidence(master_ctx, "ensemble", ensemble_decision.confidence)
                    master_ctx["ensemble_position_size"] = ensemble_decision.position_size
                    master_ctx["ensemble_position_multiplier"] = ensemble_decision.position_multiplier
                except Exception:
                    pass
                log.info(
                    f"[AnalysisAgent] Day 70 Ensemble DECISION: {final_signal} "
                    f"| conf={ensemble_decision.confidence:.0f}% | "
                    f"agreement={ensemble_decision.agreement} | "
                    f"position={ensemble_decision.position_size} "
                    f"({'conflict!' if ensemble_decision.has_conflict else 'clean'})"
                )
        except Exception as e:
            log.warning(f"[AnalysisAgent] Day 70 EnsembleEngine failed: {e}")
            ensemble_ctx = {"error": str(e)}

        # ── Day 71: Reinforcement Learning Agent (Final Wisdom Filter) ──
        # The RL agent acts as the FINAL filter on top of the Day 70 Ensemble.
        # It asks: "In similar past situations, did this type of trade work?"
        # If the RL agent says HOLD (action 0), the trade is blocked — even if
        # the ensemble agreed. This is the "knowing when NOT to trade" layer.
        rl_ctx: Dict[str, Any] = {}
        try:
            from ml.rl_agent import get_rl_agent
            import numpy as np

            agent = get_rl_agent()
            # FIX (RL observation-shape mismatch, recurring bug): a prior
            # fix hardcoded RL_OBSERVATION_SIZE based on inspecting the
            # model at one point in time (first 16, then 24). Each time
            # the model gets retrained with a different feature count,
            # that hardcoded number goes stale and every predict() call
            # silently falls back to the heuristic path again — which is
            # exactly what was happening here (model now expects 167).
            # Fix: ask the loaded model for its real size every time, so
            # this can never drift out of sync again. Falls back to the
            # feature vector's own length when no model is loaded (pure
            # heuristic mode, where the size doesn't matter).
            rl_obs_size = agent.expected_observation_size() or len(full_feature_vector) or 1
            state = np.array(
                list(full_feature_vector.values())[:rl_obs_size],
                dtype=np.float32,
            )
            if len(state) < rl_obs_size:
                state = np.pad(state, (0, rl_obs_size - len(state)))
            elif len(state) > rl_obs_size:
                state = state[:rl_obs_size]
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

            # Get ensemble signal for the RL agent to evaluate
            ensemble_signal = ensemble_ctx.get("decision", "WAIT") if isinstance(ensemble_ctx, dict) else "WAIT"
            ensemble_conf = ensemble_ctx.get("confidence", 0.0) if isinstance(ensemble_ctx, dict) else 0.0

            rl_action = agent.predict(state, ensemble_signal=ensemble_signal, ensemble_confidence=ensemble_conf)
            rl_ctx = rl_action.to_dict()

            log.info(
                f"[AnalysisAgent] Day 71 RL Agent: {rl_action.action_name} "
                f"| source={rl_action.source} | conf={rl_action.confidence:.2f} | "
                f"reason={rl_action.reason[:60]}"
            )

            # ── RL override logic ────────────────────────────────────
            # The RL agent can VETO a trade — but only if confidence is very low (< 40%)
            # Day 81+ hotfix: In TEST_MODE, never let RL VETO block a trade.
            if final_signal in ("BUY", "SELL") and rl_action.action_name == "HOLD":
                if _test_mode:
                    log.info(
                        f"[AnalysisAgent] Day 71 RL suggests HOLD — "
                        f"BYPASSED (TEST_MODE=true), keeping {final_signal}"
                    )
                elif ensemble_conf < 40:
                    _apply_confidence_penalty(signal_result, 10, "rl_hold", "analysis")
                    log.warning(
                        f"[AnalysisAgent] Day 71 RL penalty: Ensemble said {final_signal} "
                        f"but conf={ensemble_conf:.0f}% < 40% — {rl_action.reason[:80]}"
                    )
                else:
                    log.info(
                        f"[AnalysisAgent] Day 71 RL suggests HOLD but conf={ensemble_conf:.0f}% — "
                        f"allowing trade with caution"
                    )
            elif final_signal in ("BUY", "SELL") and rl_action.action_name == "CLOSE":
                log.warning(
                    f"[AnalysisAgent] Day 71 RL CLOSE: RL agent suggests closing position"
                )
                # Note: actual close happens in AITrader, not here — this is just a signal
        except Exception as e:
            log.warning(f"[AnalysisAgent] Day 71 RL Agent failed: {e}")
            rl_ctx = {"error": str(e)}

        # ── Day 73: Master Decision Engine (Central Brain) ────────────
        # The culmination of Days 60-72. Collects ALL intelligence layer
        # signals and fuses them into one final master decision with
        # dynamic weights, conflict resolution, and validation.
        master_decision_ctx: Dict[str, Any] = {}
        try:
            from core.master_decision import get_master_decision_engine
            engine = get_master_decision_engine()

            # Gather all 4 layer signals
            _rule_sig = final_signal if final_signal in ("BUY", "SELL") else "WAIT"
            _rule_conf = float(master_ctx.get("master_confidence", 0) or 0)
            if _rule_conf <= 0:
                _rule_conf = float(signal_result.get("confidence", 0) or 0)
            if _rule_conf <= 0 and _rule_sig in ("BUY", "SELL"):
                _rule_conf = 50.0

            _ml_sig = "WAIT"
            _ml_conf = 0.0
            if isinstance(ml_prediction_ctx, dict) and ml_prediction_ctx.get("prediction") != "NOT_READY":
                _ml_sig = ml_prediction_ctx.get("prediction", "WAIT")
                _ml_conf = float(ml_prediction_ctx.get("probability", 0.5)) * 100

            _rl_sig = rl_ctx.get("action_name", "HOLD") if isinstance(rl_ctx, dict) else "HOLD"
            # BUGFIX: rl_ctx["confidence"] is a fractional 0-1 value (see the
            # `.2f` no-percent-sign log format used for it elsewhere in this
            # file, vs. `.0f` + "%" used for every 0-100-scale confidence).
            # The old fallback default of 50 was on the WRONG scale for this
            # multiplication: 50 * 100 = 5000, an impossible confidence that
            # would corrupt MasterDecisionEngine's weighting any time rl_ctx
            # was missing the key (e.g. RL agent failure). Default must be
            # 0.5 (i.e. 50 after scaling), matching every other layer's
            # "unknown = 50% confidence" convention.
            _rl_conf = float(rl_ctx.get("confidence", 0.5) or 0.5) * 100 if isinstance(rl_ctx, dict) else 50.0

            _llm_sig = (master_ctx.get("master_signal") or "WAIT") if isinstance(master_ctx, dict) else "WAIT"
            _llm_conf = float(master_ctx.get("master_confidence", 0) or 0) if isinstance(master_ctx, dict) else 0.0

            master_decision = engine.decide(
                pair=symbol,
                timeframe=timeframe,
                rule_signal=_rule_sig,
                rule_confidence=_rule_conf,
                ml_signal=_ml_sig,
                ml_confidence=_ml_conf,
                rl_signal=_rl_sig,
                rl_confidence=_rl_conf,
                llm_signal=_llm_sig,
                llm_confidence=_llm_conf,
                rule_reasoning=str(signal_result.get("reasons", ""))[:100],
                ml_reasoning=str(ml_prediction_ctx.get("important_features", ""))[:100] if isinstance(ml_prediction_ctx, dict) else "",
                rl_reasoning=str(rl_ctx.get("reason", ""))[:100] if isinstance(rl_ctx, dict) else "",
                llm_reasoning=str(master_ctx.get("master_story", ""))[:100] if isinstance(master_ctx, dict) else "",
                # Day 90 — pass regime + structure + strategy context
                regime=regime if isinstance(regime, dict) else None,
                mtf_bias={"bias": mtf_bias} if isinstance(mtf_bias, str) else mtf_bias,
                structure=structure_ctx,
                strategy_context=strategy_choice,
            )
            master_decision_ctx = master_decision.to_dict()

            # Day 73 override: the master decision is the FINAL signal
            # Day 81+ hotfix: In TEST_MODE, don't let MasterDecisionEngine
            # override a BUY/SELL signal that was already set by the
            # AGGRESSIVE TEST_MODE path. The whole point of TEST_MODE is
            # to force trades through for MT5 verification — MasterDecision
            # (which aggregates rule+ML+RL+LLM) will almost always say WAIT
            # because LLM is rate-limited and ML models aren't trained yet.
            if master_decision.final_signal in ("BUY", "SELL"):
                final_signal = master_decision.final_signal
                try:
                    master_ctx["master_confidence"] = master_decision.master_confidence
                    _track_confidence(master_ctx, "master_decision", master_decision.master_confidence)
                    master_ctx["master_position_size"] = master_decision.position_size
                    master_ctx["master_position_multiplier"] = master_decision.position_multiplier
                except Exception:
                    pass
                log.info(
                    f"[AnalysisAgent] Day 73 Master Decision: {final_signal} "
                    f"| conf={master_decision.master_confidence:.0f}% | "
                    f"agreement={master_decision.agreement} | "
                    f"position={master_decision.position_size}"
                    f"{' | CONFLICT' if master_decision.has_conflict else ''}"
                    f"{' | OVERRIDE: ' + master_decision.override_reason if master_decision.override_reason else ''}"
                )
            elif master_decision.final_signal == "WAIT" and final_signal in ("BUY", "SELL"):
                if _test_mode:
                    log.info(
                        f"[AnalysisAgent] Day 73 Master Decision → WAIT "
                        f"(agreement {master_decision.agreement}) — "
                        f"BUT TEST_MODE=true, keeping {final_signal}"
                    )
                    # Don't override — keep the BUY/SELL from AGGRESSIVE TEST_MODE
                else:
                    log.info(
                        f"[AnalysisAgent] Day 73 Master Decision → WAIT "
                        f"(agreement {master_decision.agreement}, conf {master_decision.master_confidence:.0f}%)"
                    )
                    if master_decision.override_reason and final_signal in ("BUY", "SELL"):
                        _apply_confidence_penalty(signal_result, 8, "master_decision_wait", "analysis")
                        log.info(
                            f"[AnalysisAgent] MasterDecision WAIT; keeping {final_signal} with -8% penalty"
                        )
        except Exception as e:
            log.warning(f"[AnalysisAgent] Day 73 MasterDecisionEngine failed: {e}")
            master_decision_ctx = {"error": str(e)}

        # P2 FIX: "Master:" used to show master_ctx['master_signal'] which is
        # the LLM MasterAnalyst signal ONLY — NOT the MasterDecisionEngine's
        # fused 4-layer decision.  Downstream operators reading this log assumed
        # "Master" meant the authoritative fused decision, leading to confusion
        # when the log said "Master: BUY" but the actual final_signal was WAIT
        # (because the 4-layer fusion overrode the LLM).  Now we show both:
        #   LLM:   = the LLM MasterAnalyst's raw signal (for LLM-layer audit)
        #   Fused: = MasterDecisionEngine's 4-layer fused signal + agreement
        #   Final: = the signal that actually gets returned (may differ from
        #            Fused if TEST_MODE override or later engine applies)
        try:
            _llm_raw = master_ctx.get('master_signal', 'N/A') if isinstance(master_ctx, dict) else 'N/A'
            _fused_sig = master_decision.final_signal
            _fused_agr = master_decision.agreement
            _fused_conf = master_decision.master_confidence
        except Exception:
            _llm_raw = master_ctx.get('master_signal', 'N/A') if isinstance(master_ctx, dict) else 'N/A'
            _fused_sig = 'N/A'
            _fused_agr = 'N/A'
            _fused_conf = 0.0

        log.info(
            f"[AnalysisAgent] Complete — "
            f"Session: {session_ctx['current_session']} ({session_ctx['gmt_time']}) | "
            f"Strategy: {session_ctx['session_strategy']} | "
            f"Macro Regime: {intermarket_ctx.get('macro_regime', 'N/A')} | "
            f"Macro Score: {intermarket_ctx.get('macro_score', 'N/A')} | "
            f"Rule: {signal_result['signal']} | "
            f"LLM: {_llm_raw} | "
            f"Fused: {_fused_sig} ({_fused_agr}, conf {_fused_conf:.0f}%) | "
            f"Final: {final_signal}"
        )

        # ── Unified Signal Engine (Day 100+) ─────────────────
        # Runs the spec-compliant 5-engine stack (S/R zones + StopHunt +
        # ICT/AMD + Multi-Strategy PA + High-Reliability Patterns) and
        # produces a consensus signal via weighted voting.
        # This is OPTIONAL — failure does not break the main pipeline.
        unified_signal_ctx = {}
        try:
            from analysis.unified_signal_engine import UnifiedSignalEngine
            # Normalize timeframe to uppercase format the engine expects
            tf_norm = timeframe.upper() if timeframe else "4H"
            # Map common MT5 timeframes to engine's expected format
            tf_map = {"M15": "1H", "M30": "1H", "H1": "1H", "H4": "4H", "D1": "1D"}
            tf_for_engine = tf_map.get(tf_norm, "4H")

            unified_engine = UnifiedSignalEngine(timeframe=tf_for_engine)
            unified_signal_ctx = unified_engine.analyze(
                df, symbol=symbol, lower_tf_df=None  # lower TF not always available
            )
            consensus = unified_signal_ctx.get("consensus", {})
            log.info(
                f"[AnalysisAgent] Unified Signal Engine: "
                f"consensus={consensus.get('action', 'N/A')} "
                f"(BUY={consensus.get('buy_score', 0)}, SELL={consensus.get('sell_score', 0)}) "
                f"| patterns={len(unified_signal_ctx.get('detected_patterns', []))}"
            )
        except Exception as e:
            log.warning(f"[AnalysisAgent] Unified Signal Engine failed: {e}")
            unified_signal_ctx = {"error": str(e), "consensus": {"action": "NO_TRADE"}}

        # ── Adaptive Decision Engine (backtest-calibrated) ────
        # Replaces the rigid "all engines must agree" gate with a soft
        # confluence scoring system that learns from backtest results.
        # Solves the "multiple mandatory strategies = no trades" problem.
        try:
            from analysis.decision_bridge import make_adaptive_decision
            current_price = float(df["close"].iloc[-1]) if len(df) > 0 else None
            adaptive_decision = make_adaptive_decision(
                unified_signal_ctx, current_price=current_price,
                mode="confluence",  # soft scoring; any strategy can trade solo
            )
            unified_signal_ctx["adaptive_decision"] = adaptive_decision
            log.info(
                f"[AnalysisAgent] Adaptive Decision: "
                f"action={adaptive_decision.get('action', 'N/A')} "
                f"conf={adaptive_decision.get('confidence', 'N/A')} "
                f"score={adaptive_decision.get('score', 0):.2f} "
                f"source={adaptive_decision.get('source', 'N/A')}"
            )
        except Exception as e:
            log.warning(f"[AnalysisAgent] Adaptive Decision Engine failed: {e}")
            unified_signal_ctx["adaptive_decision"] = {
                "action": unified_signal_ctx.get("consensus", {}).get("action", "NO_TRADE"),
                "source": "legacy_fallback", "reason": f"Adaptive failed: {e}",
            }

        # ── Odd Enhancers Zone Scoring (Book 5 Chapter 6) ──────
        # CRITICAL FIX: Wire odd_enhancers into the LIVE pipeline.
        # Previously only used in backtest → live/backtest mismatch.
        # Now scores zones detected by the unified engine, producing
        # a zone_score that the decision system can use.
        try:
            from analysis.odd_enhancers import OddEnhancerScorer
            scorer = OddEnhancerScorer()
            sd_zones = unified_signal_ctx.get("sd_zones_result", {})
            zone_score_data = None
            if sd_zones and sd_zones.get("demand_zones"):
                nearest_demand = sd_zones["demand_zones"][0]
                zone_score_data = scorer.score_zone(
                    nearest_demand, df, current_price=current_price)
            elif sd_zones and sd_zones.get("supply_zones"):
                nearest_supply = sd_zones["supply_zones"][0]
                zone_score_data = scorer.score_zone(
                    nearest_supply, df, current_price=current_price)
            if zone_score_data:
                unified_signal_ctx["zone_score"] = {
                    "total_score": zone_score_data.total_score,
                    "max_score": zone_score_data.max_score,
                    "tier": zone_score_data.tier,
                    "entry_method": zone_score_data.entry_method,
                    "reason": zone_score_data.reason,
                }
                log.info(
                    f"[AnalysisAgent] Zone Score: "
                    f"{zone_score_data.total_score}/{zone_score_data.max_score:.0f} "
                    f"tier={zone_score_data.tier} method={zone_score_data.entry_method}"
                )
        except Exception as e:
            log.warning(f"[AnalysisAgent] Odd Enhancers scoring failed: {e}")
            unified_signal_ctx["zone_score"] = None

        return {
            "df":                df,
            "pat_ctx":           pat_ctx,
            "advanced_patterns": adv_patterns,
            "advanced_pat_ctx":  advanced_pat_ctx,
            "sr_result":         sr_res,
            "sr_ctx":            sr_ctx,
            "liquidity_ctx":     liquidity_ctx,
            "fib_result":        fib_result,
            "fib_ctx":           fib_ctx,
            "bias_result":       bias_result,
            "bias_ctx":          bias_ctx,
            "signal":            signal_result,
            "signal_ctx":        signal_ctx,
            "llm":               llm_result,
            "llm_ctx":           llm_ctx,
            "news":              news_result,
            "news_ctx":          news_ctx,
            "sentiment":         sentiment_result,
            "sentiment_ctx":     sentiment_ctx,
            "conflict":          conflict_result,
            "smc":               smc_result,
            "smc_ctx":           smc_ctx,
            # Day 47
            "vision":            vision_result,
            "vision_ctx":        vision_ctx,
            "vision_fusion":     fusion_result,
            # Day 63
            "session":           session_result,
            "session_ctx":       session_ctx,
            # Day 65
            "intermarket":       intermarket_result,
            "intermarket_ctx":   intermarket_ctx,
            "currency_strength_ctx": currency_strength_ctx,
            "macro_fusion":      macro_fusion,
            # Master
            "master":            master_result,
            "master_ctx":        master_ctx,
            # Day 66 — News Intelligence
            "news_intelligence": news_intel_ctx,
            # Day 67 — Confluence Engine
            "confluence":        confluence_ctx,
            # Day 68 — Feature Engineering
            "feature_vector":    feature_vector_ctx,
            # Day 69 — ML Prediction
            "ml_prediction":     ml_prediction_ctx,
            # Day 70 — Ensemble Brain Fusion
            "ensemble":          ensemble_ctx,
            # Day 71 — RL Agent (Final Wisdom Filter)
            "rl_agent":          rl_ctx,
            # Day 73 — Master Decision Engine
            "master_decision":   master_decision_ctx,
            # Day 90 — Six new analyzers + structure + strategy
            "structure":          structure_result,
            "structure_ctx":      structure_ctx,
            "divergence":         divergence_result,
            "divergence_ctx":     divergence_ctx,
            "ichimoku":           ichimoku_result,
            "ichimoku_ctx":       ichimoku_ctx,
            "volatility":         volatility_result,
            "volatility_ctx":     volatility_ctx,
            "volume_profile":     volume_profile_result,
            "volume_profile_ctx": volume_profile_ctx,
            "smc_advanced":       smc_advanced_result,
            "smc_advanced_ctx":   smc_advanced_ctx,
            "news_api":           news_api_result,
            "news_api_ctx":       news_api_ctx,
            "econ_calendar":      econ_calendar_result,
            "econ_calendar_ctx":  econ_calendar_ctx,
            "fred_macro":         fred_result,
            "fred_ctx":           fred_ctx,
            "retail_sentiment":   retail_sentiment_result,
            "retail_sentiment_ctx": retail_sentiment_ctx,
            "mtf_structure":      mtf_structure_result,
            "mtf_structure_ctx":  mtf_structure_ctx,
            # Round-22 audit fix: wire 6 dead engine outputs into return dict.
            # Previously these were computed every cycle but their results
            # were silently discarded — never reached the return dict.
            "correlation_ctx":       correlation_ctx,
            "institutional_ctx":     institutional_ctx,
            "surprise_ctx":          surprise_ctx,
            "microstructure_ctx":    microstructure_ctx,
            "network_ctx":           network_ctx,
            "forecast_ctx":          forecast_ctx,
            "strategy":           strategy_choice,
            "final_signal":      final_signal,
            # ARCHITECTURAL FIX: new field — records execution-gate verdicts
            # WITHOUT touching the analysis-layer `final_signal`. Consumed
            # by TradePermission, learning agent, and the audit trail.
            "execution_filters": execution_filters,
            # Day 100+ — Unified Signal Engine (5-engine consensus)
            "unified_signal":    unified_signal_ctx,
        }