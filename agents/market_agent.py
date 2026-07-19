# agents/market_agent.py — Day 12 | Market Data Agent
# Day 93 update: uses DataOrchestrator, which prefers MT5 for
# candles/account/positions and falls back to API (Twelve Data,
# yfinance) only when MT5 is unavailable.
# Institutional hardening pass (this review): see README_market_agent.md
# for the full audit — summary of behavioral changes at the bottom of
# this docstring block.
"""
MarketAgent — first stage of the trading pipeline.

Collects candle data (via DataOrchestrator: MT5 first, API fallback),
validates it, computes indicators (three-tier fallback: canonical
indicator_registry -> ExtendedIndicators -> legacy Indicators), and
detects the current market regime. Also computes a multi-timeframe (MTF)
bias as supplementary context.

Design intent: this agent should never raise out of `run()` for a
routine failure (bad symbol, feed hiccup, indicator library bug). Every
external call is wrapped and failure degrades to a structured
`{"error": ..., "detail": ...}` dict so the pipeline can skip the cycle
for that symbol instead of crashing the whole process. Callers should
always check `result.get("error")` before touching `result["df"]`.

Known residual risk (documented, not silently hidden): `DataOrchestrator`
is a shared singleton exposing `last_source` as mutable instance state
rather than returning `(df, source)` per call. If multiple MarketAgent
instances run concurrently (one per symbol, common in multi-symbol
bots), there's a narrow race window where `self._orchestrator.last_source`
could reflect a different symbol's fetch by the time it's read here. See
README "Concurrency" section for the recommended orchestrator-level fix.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, TypedDict

from data.data_orchestrator import get_data_orchestrator
from data.validator import DataValidator
from data.indicators import Indicators
from analysis.timeframe import MultiTimeframeAnalyzer
from analysis.market_regime import MarketRegimeDetector
from utils.logger import get_logger

log = get_logger("market_agent")


class MarketAgentResult(TypedDict, total=False):
    error: str
    detail: str
    skipped: bool
    df: "object"  # pandas.DataFrame — left untyped to avoid a hard pandas import here
    ind_ctx: dict
    regime: dict
    regime_ctx: dict
    mtf_bias: str
    symbol: str
    timeframe: str
    data_source: str


# ----------------------------------------------------------------------
# Module-level state, made explicit and thread-safe (see Critical Issues
# in the audit: the previous `_legacy_fallback_count` was a single global
# int mutated via `global` with no lock, shared across every symbol —
# both a race condition under concurrent MarketAgent instances and a
# rate-limiter that could suppress a *new* symbol's first-ever legacy
# fallback warning just because other symbols had already tripped it.)
# ----------------------------------------------------------------------
_legacy_fallback_lock = threading.Lock()
_legacy_fallback_counts: dict[str, int] = {}
_symbol_check_import_warning_logged = False


def _note_legacy_fallback(symbol: str) -> int:
    """Thread-safe, per-symbol legacy-indicator-fallback counter.
    Returns the new count for `symbol`."""
    with _legacy_fallback_lock:
        count = _legacy_fallback_counts.get(symbol, 0) + 1
        _legacy_fallback_counts[symbol] = count
        return count


class MarketAgent:
    """
    Market data collect, validate, indicator calculate, regime detect।
    Pipeline এর প্রথম agent। (First agent in the pipeline: collects
    market data, validates it, computes indicators, detects regime.)

    Day 93: Uses DataOrchestrator instead of DataFetcher directly, so
    candle data comes from MT5 when available (Windows + MT5 terminal
    running) and falls back to API when not (Linux VPS).
    """

    def __init__(self, symbol: str, timeframe: str = "15m", verbose: bool = True) -> None:
        """
        Args:
            symbol: broker symbol, e.g. "EURUSD".
            timeframe: candle timeframe for the primary indicator/regime
                pass, e.g. "15m". The MTF pass always uses a fixed
                ["1d", "4h", "1h", "15m"] ladder regardless of this value.
            verbose: if True, call downstream `print_summary()` helpers
                (regime detector) in addition to structured logging.
                Set False in production/Docker to keep stdout clean and
                rely on `log` (which can be routed to a file/aggregator)
                instead. Does not affect `mtf.print_summary()`, whose
                return value (the bias string) this agent depends on —
                see README "Known Issues" for why that call can't be
                gated without changing the MTF analyzer's API.
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.verbose = verbose
        # Day 93 — orchestrator handles MT5-vs-API choice.
        self._orchestrator = get_data_orchestrator()

    # ------------------------------------------------------------------
    # Symbol-unavailable short-circuit
    # ------------------------------------------------------------------
    @staticmethod
    def _is_symbol_unavailable(symbol: str) -> bool:
        """
        Best-effort check against data.fetcher's known-unavailable-symbol
        cache. Prevents ~30 non-existent symbols (USOUSD, BTCUSD, etc.)
        from wasting MTF fetch time and triggering recovery pauses every
        cycle.

        Imported lazily (not at module load) to avoid a circular import
        between data.fetcher and this module. Unlike the previous
        version, an import failure is logged once (not silently
        swallowed on every single call) so a broken import doesn't
        silently disable this optimization forever without a trace.
        """
        global _symbol_check_import_warning_logged
        try:
            from data.fetcher import is_symbol_unavailable
        except ImportError as exc:
            if not _symbol_check_import_warning_logged:
                log.error(
                    f"[MarketAgent] Could not import is_symbol_unavailable — "
                    f"unavailable-symbol short-circuit disabled for this process: {exc}"
                )
                _symbol_check_import_warning_logged = True
            return False

        try:
            return bool(is_symbol_unavailable(symbol))
        except Exception as exc:
            log.warning(f"[MarketAgent] is_symbol_unavailable({symbol}) check failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Main pipeline step
    # ------------------------------------------------------------------
    def run(self) -> MarketAgentResult:
        if self._is_symbol_unavailable(self.symbol):
            log.debug(f"[MarketAgent] {self.symbol} — skipped (not on broker)")
            return {"error": "symbol_unavailable", "skipped": True}

        log.info(f"[MarketAgent] Running for {self.symbol} {self.timeframe}")

        # ── MTF bias — wrapped so MTF failure doesn't kill the cycle ──
        mtf_bias = "NEUTRAL"
        try:
            mtf = MultiTimeframeAnalyzer(self.symbol)
            mtf_data = mtf.analyze(["1d", "4h", "1h", "15m"])
            # NOTE: `print_summary` both prints AND returns the bias string
            # in the upstream MultiTimeframeAnalyzer API — a naming smell
            # inherited from that module (a "print" function shouldn't be
            # the sole source of a value this agent depends on). Flagged
            # in README as a recommended upstream fix; not changed here
            # since altering MultiTimeframeAnalyzer is out of this file's
            # scope and doing so blind risks breaking its contract.
            mtf_bias = mtf.print_summary(mtf_data)
        except Exception as e:
            log.warning(f"[MarketAgent] MTF analysis failed (non-critical): {e}")

        # ── Day 93 — Fetch via Orchestrator (MT5 first, API fallback) ──
        try:
            fetch_started = time.perf_counter()
            df = self._orchestrator.get_candles(self.symbol, self.timeframe, limit=300)
            fetch_elapsed_ms = round((time.perf_counter() - fetch_started) * 1000, 1)
        except Exception as e:
            log.error(f"[MarketAgent] get_candles raised for {self.symbol}: {e}")
            return {"error": "fetch_failed", "detail": str(e)}

        if df is None:
            # Distinguish "symbol doesn't exist" from "connection lost".
            # fetcher.mark_symbol_unavailable() was already called for code=-1.
            if self._is_symbol_unavailable(self.symbol):
                return {"error": "symbol_unavailable", "skipped": True}
            log.error(f"[MarketAgent] Data fetch failed for {self.symbol} (MT5 + API both unavailable)")
            return {"error": "fetch_failed"}

        # Capture the source immediately after the call to minimize (not
        # eliminate) the race window against concurrent MarketAgent
        # instances sharing the orchestrator singleton — see module
        # docstring "Known residual risk".
        data_source = self._orchestrator.last_source
        if fetch_elapsed_ms > 2000:
            log.warning(f"[MarketAgent] Slow candle fetch for {self.symbol}: {fetch_elapsed_ms}ms via {data_source}")

        # ── Validate ──
        try:
            is_valid = DataValidator().validate(df, self.symbol, self.timeframe)
        except Exception as e:
            log.error(f"[MarketAgent] Validator raised for {self.symbol}: {e}")
            return {"error": "validation_failed", "detail": str(e)}
        if not is_valid:
            log.error(f"[MarketAgent] Validation failed for {self.symbol}")
            return {"error": "validation_failed"}

        # ── Indicators — three-tier fallback ──
        # Co-founder fix: prefer the canonical indicator registry (single
        # source of truth, delegates to ExtendedIndicators/pandas-ta).
        # Falls back to direct ExtendedIndicators, then to legacy
        # Indicators. If ALL THREE fail, return a structured error
        # instead of letting the exception propagate out of run() —
        # the previous version had no handler around the final legacy
        # tier, so a total indicator failure would crash the caller
        # instead of degrading gracefully like every other failure mode
        # in this function.
        ind_ctx: dict
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
                if self.verbose:
                    ind_ext.print_summary(df)
                log.info(f"[MarketAgent] Used ExtendedIndicators (pandas-ta, {len(df.columns)} cols)")
            except Exception as e_ext:
                count = _note_legacy_fallback(self.symbol)
                # Rate-limit: only WARN on the first occurrence for this
                # symbol and every 50th thereafter; DEBUG in between.
                if count <= 1 or count % 50 == 1:
                    log.warning(
                        f"[MarketAgent] ⚠️ LEGACY INDICATOR FALLBACK for {self.symbol} — "
                        f"registry error: {e_registry} | ext error: {e_ext} | "
                        f"occurrences for this symbol: {count} (rate-limited log)"
                    )
                else:
                    log.debug(f"[MarketAgent] Legacy indicator fallback for {self.symbol} (#{count})")
                try:
                    ind = Indicators()
                    df = ind.add_all(df)
                    ind_ctx = ind.get_ai_context(df)
                except Exception as e_legacy:
                    log.error(
                        f"[MarketAgent] All three indicator tiers failed for {self.symbol}: "
                        f"registry={e_registry} | ext={e_ext} | legacy={e_legacy}"
                    )
                    return {"error": "indicator_calculation_failed", "detail": str(e_legacy)}

        # ── Regime — wrapped for the same reason as the fetch/validate
        # steps above; the previous version left this unguarded, breaking
        # the file's own established pattern of never letting a single
        # downstream library raise out of run(). ──
        try:
            regime_detector = MarketRegimeDetector()
            regime_result = regime_detector.detect(df)
            if self.verbose:
                regime_detector.print_summary(regime_result)
            regime_ctx = regime_detector.get_ai_context(regime_result)
        except Exception as e:
            log.error(f"[MarketAgent] Regime detection failed for {self.symbol}: {e}")
            return {"error": "regime_detection_failed", "detail": str(e)}

        log.info(
            f"[MarketAgent] Done — "
            f"Source: {data_source} | "
            f"Price: {ind_ctx.get('price')} | "
            f"Trend: {ind_ctx.get('trend')} | "
            f"Regime: {regime_result.get('regime')} | "
            f"FetchMs: {fetch_elapsed_ms}"
        )

        return {
            "df": df,
            "ind_ctx": ind_ctx,
            "regime": regime_result,
            "regime_ctx": regime_ctx,
            "mtf_bias": mtf_bias,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "data_source": data_source,  # Day 93
        }