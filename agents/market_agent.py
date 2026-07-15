# agents/market_agent.py  —  Day 12 | Market Data Agent
# Day 93 update: now uses DataOrchestrator which prefers MT5 for
# candles/account/positions, and falls back to API (Twelve Data,
# yfinance) only when MT5 is unavailable.

from data.fetcher import get_data_fetcher
from data.data_orchestrator import get_data_orchestrator
from data.validator import DataValidator
from data.indicators import Indicators
from analysis.timeframe import MultiTimeframeAnalyzer
from analysis.market_regime import MarketRegimeDetector
from utils.logger import get_logger

log = get_logger("market_agent")


class MarketAgent:
    """
    Market data collect, validate, indicator calculate, regime detect।
    Pipeline এর প্রথম agent।

    Day 93: Uses DataOrchestrator instead of DataFetcher directly,
    so candle data comes from MT5 when available (Windows + MT5
    terminal running) and falls back to API when not (Linux VPS).
    """

    def __init__(self, symbol: str, timeframe: str = "15m"):
        self.symbol    = symbol
        self.timeframe = timeframe
        # Day 93 — orchestrator handles MT5-vs-API choice
        self._orchestrator = get_data_orchestrator()
        # Keep DataFetcher for backward-compat (some MTF calls still use it)
        self._fetcher = get_data_fetcher()

    def run(self) -> dict:
        log.info(f"[MarketAgent] Running for {self.symbol} {self.timeframe}")

        # MTF — wrap in try/except so MTF failure doesn't kill the cycle
        mtf_bias = "NEUTRAL"
        try:
            mtf      = MultiTimeframeAnalyzer(self.symbol)
            mtf_data = mtf.analyze(["1d", "4h", "1h", "15m"])
            mtf_bias = mtf.print_summary(mtf_data)
        except Exception as e:
            log.warning(f"[MarketAgent] MTF analysis failed (non-critical): {e}")

        # ── Day 93 — Fetch via Orchestrator (MT5 first, API fallback) ──
        df = self._orchestrator.get_candles(self.symbol, self.timeframe, limit=300)
        if df is None:
            log.error(f"[MarketAgent] Data fetch failed for {self.symbol} (MT5 + API both unavailable)")
            return {"error": "fetch_failed"}

        # Validate
        if not DataValidator().validate(df, self.symbol, self.timeframe):
            log.error("Validation failed")
            return {"error": "validation_failed"}

        # Indicators — Co-founder fix: prefer the canonical indicator registry
        # (single source of truth, delegates to ExtendedIndicators/pandas-ta).
        # Falls back to direct ExtendedIndicators, then to legacy Indicators.
        try:
            from data.indicator_registry import add_canonical_indicators, get_ai_context as _get_ctx
            df = add_canonical_indicators(df, include_patterns=True)
            ind_ctx = _get_ctx(df)
            log.info(f"[MarketAgent] Used canonical indicator_registry ({len(df.columns)} cols)")
        except Exception as e_registry:
            log.warning(f"[MarketAgent] indicator_registry failed ({e_registry}) — falling back to ExtendedIndicators")
            try:
                from data.indicators_ext import ExtendedIndicators
                ind_ext = ExtendedIndicators()
                df = ind_ext.add_all(df, include_patterns=True)
                ind_ctx = ind_ext.get_ai_context(df)
                ind_ext.print_summary(df)
                log.info(f"[MarketAgent] Used ExtendedIndicators (pandas-ta, {len(df.columns)} cols)")
            except Exception as e:
                log.warning(f"[MarketAgent] ExtendedIndicators failed ({e}) — falling back to legacy Indicators")
                ind    = Indicators()
                df     = ind.add_all(df)
                ind_ctx = ind.get_ai_context(df)

        # Regime
        regime_detector = MarketRegimeDetector()
        regime_result   = regime_detector.detect(df)
        regime_detector.print_summary(regime_result)
        regime_ctx = regime_detector.get_ai_context(regime_result)

        log.info(
            f"[MarketAgent] Done — "
            f"Source: {self._orchestrator.last_source} | "
            f"Price: {ind_ctx.get('price')} | "
            f"Trend: {ind_ctx.get('trend')} | "
            f"Regime: {regime_result.get('regime')}"
        )

        return {
            "df":          df,
            "ind_ctx":     ind_ctx,
            "regime":      regime_result,
            "regime_ctx":  regime_ctx,
            "mtf_bias":    mtf_bias,
            "symbol":      self.symbol,
            "timeframe":   self.timeframe,
            "data_source": self._orchestrator.last_source,  # Day 93
        }