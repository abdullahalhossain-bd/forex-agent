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
        from data.fetcher import DataFetcher
        self._h4_fetcher         = DataFetcher()

    def run(self, market_output: dict, memory_ctx: dict = None) -> dict:
        if "error" in market_output:
            return {"error": market_output["error"]}

        df        = market_output["df"]
        ind_ctx   = market_output["ind_ctx"]
        regime    = market_output["regime"]
        mtf_bias  = market_output["mtf_bias"]
        symbol    = market_output["symbol"]
        timeframe = market_output.get("timeframe", "15m")

        log.info(
            f"[AnalysisAgent] Running Day 65 pipeline for {symbol} ({timeframe})"
        )

        # ── 0. SESSION INTELLIGENCE (Day 63) ─────────────────
        # SMC context এখনো নেই, তাই খালি dict দিয়ে শুরু, পরে update
        session_result = self.session_analyzer.analyze(
            pair        = symbol,
            smc_ctx     = {},
            signal      = "NO TRADE",
            signal_conf = 0,
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
                "macro_fusion":      {},
                "master":            {},
                "master_ctx":        {"master_signal": "WAIT", "master_confidence": 0},
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
        detector = PatternDetector()
        df       = detector.run_full_detection(df)
        detector.get_latest_patterns(df, lookback=5)
        pat_ctx  = detector.get_ai_pattern_context(df)

        # ── 2. Support & Resistance ───────────────────────────
        sr      = SupportResistance()
        sr_res  = sr.analyze(df)
        sr.get_summary(sr_res)
        sr_ctx  = sr.get_ai_context(sr_res)

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
            advanced_pat_ctx = adv_detector.get_ai_context(
                df,
                ind_ctx    = ind_ctx,
                sr_ctx     = sr_ctx,
                regime_ctx = regime,
                pat_ctx    = pat_ctx,
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
        bias_engine = MarketBiasEngine()
        bias_result = bias_engine.analyze(ind_ctx, pat_ctx, sr_ctx, mtf_bias)
        bias_engine.print_summary(bias_result)
        bias_ctx    = bias_engine.get_ai_context(bias_result)

        # ── 6. Rule-based Signal ──────────────────────────────
        signal_engine = SignalEngine()
        signal_result = signal_engine.generate(
            ind_ctx          = ind_ctx,
            pat_ctx          = pat_ctx,
            sr_ctx           = sr_ctx,
            regime           = regime,
            mtf_bias         = mtf_bias,
            advanced_pat_ctx = advanced_pat_ctx,
            fib_ctx          = fib_ctx,
        )
        signal_engine.print_summary(signal_result)
        signal_ctx = signal_engine.get_ai_context(signal_result)

        # ── 7. Sentiment Engine ─────────────────────────────
        sentiment_ctx    = {}
        sentiment_result = {}
        conflict_result  = {}
        try:
            sent_provider    = SentimentDataProvider()
            sent_data        = sent_provider.get_all(symbol)
            sent_provider.print_summary(sent_data)

            sent_engine      = SentimentEngine()
            sentiment_result = sent_engine.final_sentiment_score(
                pair               = sent_data["pair"],
                retail_long_pct    = sent_data["retail_long_pct"],
                fg_index           = sent_data["fg_index"],
                currency_strengths = sent_data["currency_strengths"],
                dxy_trend          = sent_data["dxy_trend"],
                dxy_change_pct     = sent_data["dxy_change_pct"],
            )
            sent_engine.print_summary(sentiment_result)
            sentiment_ctx = sent_engine.get_ai_context(sentiment_result)

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

        # ── 8.85 Economic Calendar (Day 94/95 — institutional grade) ──
        # Multi-source calendar: Trading Economics → Investing RSS → DailyFX RSS
        # → Tradermade → Finnhub → Fxstreet → FF scraper.
        # Blocks trades if high-impact event within ±30min.
        econ_calendar_result = {}
        econ_calendar_ctx    = {}
        try:
            # Day 95: try Trading Economics calendar first (no Tradermade needed)
            from fundamental.trading_economics_calendar import TradingEconomicsCalendar
            te_cal = TradingEconomicsCalendar()
            currencies = list({symbol[:3], symbol[3:6]}) if len(symbol) >= 6 else ["USD"]
            econ_calendar_result = te_cal.get_calendar(currencies=currencies, hours_ahead=24)
            if econ_calendar_result.get("source") == "none":
                # Fall back to original EconomicCalendarAPI (Tradermade/FF chain)
                econ_cal = EconomicCalendarAPI()
                econ_calendar_result = econ_cal.get_calendar(currencies=currencies, hours_ahead=24)
                econ_cal.print_summary(econ_calendar_result)
                econ_calendar_ctx = econ_cal.get_ai_context(econ_calendar_result)
            else:
                te_cal.print_summary(econ_calendar_result)
                econ_calendar_ctx = te_cal.get_ai_context(econ_calendar_result)
        except Exception as e:
            log.warning(f"[AnalysisAgent] Economic Calendar error: {e}")
            # Last-resort: try original EconomicCalendarAPI
            try:
                econ_cal = EconomicCalendarAPI()
                currencies = list({symbol[:3], symbol[3:6]}) if len(symbol) >= 6 else ["USD"]
                econ_calendar_result = econ_cal.get_calendar(currencies=currencies, hours_ahead=24)
                econ_calendar_ctx = econ_cal.get_ai_context(econ_calendar_result)
            except Exception as e2:
                log.warning(f"[AnalysisAgent] Economic Calendar fallback also failed: {e2}")

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
            inst_engine = InstitutionalFlowEngine()
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
        llm_result = AIAnalyst().analyze(
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
        AIAnalyst().print_summary(llm_result)
        llm_ctx = AIAnalyst().get_ai_context(llm_result)

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
                "vision":            vision_result,
                "vision_ctx":        vision_ctx,
                "vision_fusion":     fusion_result,
                "session":           session_result,
                "session_ctx":       session_ctx,
                "intermarket":       intermarket_result,
                "intermarket_ctx":   intermarket_ctx,
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
                "strategy":           strategy_choice,
                "final_signal":      final_signal,
                "test_mode_bypass":  True,  # Flag for downstream debugging
            }

        # Day 63: Session dead zone / strategy gate
        elif not session_result["trade_allowed"]:
            final_signal = "NO TRADE"
            log.info(
                f"[AnalysisAgent] -> NO TRADE "
                f"(Session gate: {session_ctx['current_session']} — "
                f"{session_ctx['session_strategy']})"
            )

        elif not news_result["trade_allowed"]:
            final_signal = "NO TRADE"
            log.info("[AnalysisAgent] -> NO TRADE (news block override)")

        elif conflict_result.get("has_conflict") and sentiment_result.get("confidence", 0) >= 70:
            final_signal = "NO TRADE"
            log.info("[AnalysisAgent] -> NO TRADE (high-confidence sentiment conflict)")

        elif fusion_result.get("has_conflict") and fusion_result.get("adjusted_conf", 100) < 45:
            final_signal = "NO TRADE"
            log.info("[AnalysisAgent] -> NO TRADE (vision/quant conflict — low confidence)")

        elif master_ctx.get("master_signal") in ("BUY", "SELL", "WAIT", "STRONG_BUY", "STRONG_SELL"):
            ma_signal    = master_ctx["master_signal"]
            # If master returns WAIT but rule signal is BUY/SELL/STRONG_BUY/STRONG_SELL
            # with good confidence, use rule signal instead of letting master override.
            # Day 81+ hotfix: previously only "BUY"/"SELL" were checked, but the rule
            # engine also returns "STRONG_BUY"/"STRONG_SELL" — those were falling through
            # to the else branch and getting overridden by master's WAIT.
            rule_sig = signal_result.get("signal", "WAIT")
            rule_conf = signal_result.get("confidence", 0)
            # Normalize STRONG_BUY → BUY, STRONG_SELL → SELL for the final signal
            rule_sig_normalized = rule_sig
            if "STRONG_BUY" in str(rule_sig):
                rule_sig_normalized = "BUY"
            elif "STRONG_SELL" in str(rule_sig):
                rule_sig_normalized = "SELL"

            if ma_signal == "WAIT" and rule_sig_normalized in ("BUY", "SELL") and rule_conf >= 30:
                final_signal = rule_sig_normalized
                log.info(f"[AnalysisAgent] -> {final_signal} (Rule signal: {rule_sig} {rule_conf}% conf, master WAIT — rule override)")
            else:
                final_signal = "NO TRADE" if ma_signal == "WAIT" else ma_signal
                log.info(f"[AnalysisAgent] -> {final_signal} (MasterAnalyst override)")


        # ── Day 66: News Intelligence integration ────────────────────
        # After MasterAnalyst decides, run NewsIntelligence to:
        #   1. BLOCK the trade if pair is in a high-impact event window
        #   2. ADJUST confidence based on news bias alignment
        #
        # Day 81+ hotfix: In TEST_MODE, skip the news block (but still
        # log it as a warning). The news intelligence module fetches
        # central-bank events from a hardcoded schedule which can produce
        # false-positive "CPI in 0min" blocks even when the actual
        # ForexFactory calendar is empty. This was blocking every trade
        # during certain GMT hours.
        news_intel_ctx = {}
        _skip_news_block = False
        try:
            from config import TEST_MODE
            _skip_news_block = bool(TEST_MODE)
        except Exception:
            pass

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
            if block_check["blocked"] and final_signal in ("BUY", "SELL"):
                if _skip_news_block:
                    log.warning(
                        f"[AnalysisAgent] News block detected ({block_check['reason']}) — "
                        f"BYPASSED (TEST_MODE=true). Trade continues."
                    )
                    news_intel_ctx = {
                        "blocked": False,
                        "block_reason": f"{block_check['reason']} (TEST_MODE bypassed)",
                    }
                else:
                    log.warning(
                        f"[AnalysisAgent] -> NO TRADE (Day 66 News block: {block_check['reason']})"
                    )
                    final_signal = "NO TRADE"
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
            # Day 81+ hotfix: In TEST_MODE, don't let Confluence AVOID block trades.
            if not decision.should_trade and final_signal in ("BUY", "SELL"):
                if _test_mode:
                    log.info(
                        f"[AnalysisAgent] Day 67 Confluence: {final_signal} quality={decision.setup_quality} — "
                        f"BYPASSED (TEST_MODE=true)"
                    )
                elif decision.setup_quality == "AVOID":
                    log.info(
                        f"[AnalysisAgent] Day 67 Confluence: {final_signal} → NO TRADE "
                        f"(quality=AVOID, {decision.block_reason or 'failed validation'})"
                    )
                    final_signal = "NO TRADE"
                else:
                    log.info(
                        f"[AnalysisAgent] Day 67 Confluence: {final_signal} allowed "
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
            unified_for_features = {
                "smc_ctx": smc_ctx,
                "session_ctx": session_ctx,
                "intermarket_ctx": intermarket_ctx,
                "sentiment_ctx": sentiment_ctx,
                "news_intelligence": news_intel_ctx,
                "signal": signal_result,
                "bias_ctx": bias_ctx,
                "fib_ctx": fib_ctx,
                "sr_ctx": sr_ctx,
                "advanced_pat_ctx": advanced_pat_ctx,
                "mtf_bias": market_output.get("mtf_bias") if isinstance(market_output, dict) else None,
                "confluence": confluence_ctx,
                "master_ctx": master_ctx,
                "llm": (master_ctx or {}).get("llm", {}) if isinstance(master_ctx, dict) else {},
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
            regime = (intermarket_ctx.get("macro_regime") or "UNKNOWN") if isinstance(intermarket_ctx, dict) else "UNKNOWN"

            # Run the ensemble engine
            ensemble_decision = engine.decide(
                pair=symbol,
                timeframe=timeframe,
                ml_prediction=ml_prediction_ctx,
                rule_signal=rule_sig,
                rule_confidence=rule_conf,
                master_signal=master_sig,
                master_confidence=master_conf,
                regime=regime,
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
                    final_signal = "NO TRADE"
            elif ensemble_decision.decision == "WAIT":
                # Don't automatically block — only block if confidence is very low
                # Day 81+ hotfix: In TEST_MODE, never let Ensemble WAIT block a trade
                if ensemble_decision.confidence < 40 and not _test_mode:
                    if final_signal in ("BUY", "SELL"):
                        log.info(
                            f"[AnalysisAgent] Day 70 Ensemble → WAIT "
                            f"(conf {ensemble_decision.confidence:.0f}% < 40%)"
                        )
                        final_signal = "NO TRADE"
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
            # Build state vector from the feature vector
            state = np.array(list(full_feature_vector.values())[:160], dtype=np.float32)
            # Pad to consistent size
            if len(state) < 160:
                state = np.pad(state, (0, 160 - len(state)))
            elif len(state) > 160:
                state = state[:160]
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
                    log.warning(
                        f"[AnalysisAgent] Day 71 RL VETO: Ensemble said {final_signal} "
                        f"but conf={ensemble_conf:.0f}% < 40% — {rl_action.reason[:80]}"
                    )
                    final_signal = "NO TRADE"
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
            _rl_conf = float(rl_ctx.get("confidence", 50) or 50) * 100 if isinstance(rl_ctx, dict) else 50.0

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
                    if master_decision.override_reason:
                        final_signal = "NO TRADE"
        except Exception as e:
            log.warning(f"[AnalysisAgent] Day 73 MasterDecisionEngine failed: {e}")
            master_decision_ctx = {"error": str(e)}

        log.info(
            f"[AnalysisAgent] Complete — "
            f"Session: {session_ctx['current_session']} ({session_ctx['gmt_time']}) | "
            f"Strategy: {session_ctx['session_strategy']} | "
            f"Macro Regime: {intermarket_ctx.get('macro_regime', 'N/A')} | "
            f"Macro Score: {intermarket_ctx.get('macro_score', 'N/A')} | "
            f"Rule: {signal_result['signal']} | "
            f"Master: {master_ctx.get('master_signal', 'N/A')} | "
            f"Final: {final_signal}"
        )

        return {
            "df":                df,
            "pat_ctx":           pat_ctx,
            "advanced_patterns": adv_patterns,
            "advanced_pat_ctx":  advanced_pat_ctx,
            "sr_result":         sr_res,
            "sr_ctx":            sr_ctx,
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
            "strategy":           strategy_choice,
            "final_signal":      final_signal,
        }