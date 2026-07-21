"""
core/data_provider.py — DataProvider abstraction (execution-parity refactor).

Per the mode-agnostic engine mandate: the trading engine (AITrader.evaluate_
decision_core) must never know whether its market_out dict came from a live
MT5 tick or a historical bar. This module is the ONLY place that boundary
is allowed to live on the *input* side.

DataProvider.get_market_out(symbol, timeframe) -> MarketAgentResult-shaped
dict is the single contract. Nothing downstream inspects `mode`.

- LiveMT5Provider   wraps agents.market_agent.MarketAgent.run() (the exact
  object AITrader already builds live) — zero new logic, pure wrapper.
- HistoricalMT5Provider wraps the indicator-registry chain unified_engine.py
  already uses (data.indicator_registry -> indicators_ext -> indicators),
  moved here verbatim so it has one home instead of living inline in the
  backtest loop.

Both return the same dict shape. Building a NEW analysis/indicator path for
either mode is a parity violation — don't do it. If you need a new field on
market_out, add it to both providers, not one.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

log = logging.getLogger("data_provider")


class DataProvider(ABC):
    """Contract every provider must satisfy. Return shape must match
    agents/market_agent.py's MarketAgentResult dict exactly:
    {df, ind_ctx, regime, regime_ctx, mtf_bias, symbol, timeframe,
     data_source}.
    """

    @abstractmethod
    def get_market_out(self, symbol: str, timeframe: str) -> dict:
        ...

    @abstractmethod
    def current_time(self):
        """Broker-time timestamp of the last bar this provider has seen.
        Live: real wall-clock-ish broker time. Historical: the replay
        cursor's bar timestamp. Callers (session filters, news filters)
        must ask the provider for "now" instead of calling datetime.now()
        directly, or historical replay silently gets today's session/news
        state applied to a 2023 bar."""
        ...


class LiveMT5Provider(DataProvider):
    """Thin wrapper around the existing, already-live-tested MarketAgent.
    Does not reimplement indicator computation, regime detection, or MTF
    bias — it calls the exact same MarketAgent instance AITrader already
    constructs in __init__."""

    def __init__(self, market_agent):
        self._market_agent = market_agent

    def get_market_out(self, symbol: str, timeframe: str) -> dict:
        # Pass symbol/timeframe through so the agent analyses the
        # requested pair instead of whatever it was constructed with.
        self._market_agent.symbol = symbol
        self._market_agent.timeframe = timeframe
        return self._market_agent.run()

    def current_time(self):
        import datetime
        return datetime.datetime.utcnow()


class HistoricalMT5Provider(DataProvider):
    """Replays a pre-fetched historical MT5 candle DataFrame one bar at a
    time. Uses the SAME canonical indicator chain MarketAgent uses live
    (indicator_registry -> ExtendedIndicators -> legacy Indicators) so
    indicator values match bar-for-bar — this function is moved verbatim
    from backtest/unified_engine.py._build_market_out, not rewritten.

    KNOWN PARITY GAP (flagged, not hidden): `mtf_bias` is returned as a
    static NEUTRAL/LOW placeholder because computing a true historical MTF
    bias needs synchronized higher-timeframe slices at every bar — that is
    tracked follow-up work (see PARITY_GAPS.md), not silently papered over.
    Any confidence number that depends on MTF bias will NOT match what
    Demo would have produced on the same historical timestamp until this
    is closed.
    """

    def __init__(self, df, symbol: str, timeframe: str):
        self._df = df
        self._symbol = symbol
        self._timeframe = timeframe
        self._cursor = 0  # index of the last closed bar included

    def advance_to(self, bar_index: int) -> None:
        """Move the replay cursor. Caller (the replay loop) is responsible
        for only ever advancing forward — this class does not protect
        against look-ahead misuse by the caller, only against building
        market_out from bars beyond the cursor."""
        self._cursor = bar_index

    def current_time(self):
        return self._df.index[self._cursor]

    def get_market_out(self, symbol: str, timeframe: str) -> dict:
        df_slice = self._df.iloc[: self._cursor + 1].copy()
        ind_ctx = {}
        try:
            from data.indicator_registry import add_canonical_indicators, get_ai_context as _get_ctx
            df_slice = add_canonical_indicators(df_slice, include_patterns=True)
            ind_ctx = _get_ctx(df_slice)
        except Exception as e_registry:
            log.warning(f"[HistoricalMT5Provider] indicator_registry unavailable "
                        f"({e_registry}) — falling back to ExtendedIndicators, "
                        f"then legacy Indicators")
            try:
                from data.indicators_ext import ExtendedIndicators
                ind_ext = ExtendedIndicators()
                df_slice = ind_ext.add_all(df_slice, include_patterns=True)
                ind_ctx = ind_ext.get_ai_context(df_slice)
            except Exception:
                from data.indicators import Indicators
                ind = Indicators()
                df_slice = ind.add_all(df_slice)
                ind_ctx = ind.get_ai_context(df_slice)

        try:
            from analysis.market_regime import MarketRegimeDetector
            regime_detector = MarketRegimeDetector()
            regime_result = regime_detector.detect(df_slice)
            regime_ctx = regime_detector.get_ai_context(regime_result)
        except Exception as e:
            log.debug(f"[HistoricalMT5Provider] regime detection unavailable: {e}")
            regime_result, regime_ctx = {}, {}

        return {
            "df": df_slice,
            "ind_ctx": ind_ctx,
            "regime": regime_result,
            "regime_ctx": regime_ctx,
            # PARITY GAP — see class docstring.
            "mtf_bias": {"bias": "NEUTRAL", "confidence": "LOW"},
            "symbol": symbol,
            "timeframe": timeframe,
            "data_source": "historical_replay",
        }
