"""
core/data_provider.py — DataProvider abstraction (execution-parity refactor).

Per the mode-agnostic engine mandate: the trading engine (AITrader.evaluate_
decision_core) must never know whether its market_out dict came from a live
MT5 tick or a historical bar. This module is the ONLY place that boundary
is allowed to live on the *input* side.

DataProvider.get_market_out(symbol, timeframe) -> MarketAgentResult-shaped
dict is the single contract. Nothing downstream inspects `mode`.

- LiveMT5Provider         wraps agents.market_agent.MarketAgent.run() (the exact
  object AITrader already builds live) — zero new logic, pure wrapper.
- HistoricalMT5Provider   wraps the indicator-registry chain unified_engine.py
  already uses (data.indicator_registry -> indicators_ext -> indicators),
  moved here verbatim so it has one home instead of living inline in the
  backtest loop.
- HistoricalCSVProvider   lives in core/csv_data_provider.py — multi-timeframe
  CSV-based provider for when MT5 isn't available. Loads local CSVs once,
  serves per-bar market_out dicts with the SAME shape. This is the
  preferred backtest provider when running on a machine without MT5
  (CI, audit envs) — but requires CSVs to be downloaded first via
  scripts/download_historical_data.py on the production VPS.

All three return the same dict shape. Building a NEW analysis/indicator path
for any mode is a parity violation — don't do it. If you need a new field on
market_out, add it to all providers, not one.
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
        # P1-A R4 FIX: must return tz-aware UTC for parity with HistoricalCSVProvider.
        # Previously returned datetime.utcnow() (NAIVE) which crashes callers that
        # compare to tz-aware bar timestamps.
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)


class HistoricalMT5Provider(DataProvider):
    """Replays a pre-fetched historical MT5 candle DataFrame one bar at a
    time. Uses the SAME canonical indicator chain MarketAgent uses live
    (indicator_registry -> ExtendedIndicators -> legacy Indicators) so
    indicator values match bar-for-bar — this function is moved verbatim
    from backtest/unified_engine.py._build_market_out, not rewritten.

    PARITY (Phase 2 fix): `mtf_bias` is now computed via causal H1→4H/1D
    resampling (see `_compute_mtf_bias`). Previously this returned a
    static NEUTRAL/LOW placeholder, which caused SignalEngine.generate()
    to never add the MTF bias vote it adds in live trading. The computed
    bias uses the same EMA-trend agreement logic live
    MultiTimeframeAnalyzer uses; only the data source differs (resampled
    H1 vs live-fetched 4H/1D bars).
    """

    def __init__(self, df, symbol: str, timeframe: str):
        self._df = df
        self._symbol = symbol
        self._timeframe = timeframe
        self._cursor = 0  # index of the last closed bar included
        # Iteration-3: register M15 + resampled H1/H4 for SMCEngine via
        # DataFetcher backtest cache (same path live uses for multi-TF).
        try:
            from data.backtest_ohlcv_cache import register_from_m15, register_series
            tf_u = (timeframe or "").upper().replace(" ", "")
            if tf_u in ("M15", "15M", "15"):
                register_from_m15(symbol, df, also=("H1", "H4"))
            else:
                register_series(symbol, timeframe, df)
        except Exception as e:
            log.debug(f"[HistoricalMT5Provider] OHLCV cache register skipped: {e}")

    def advance_to(self, bar_index: int) -> None:
        """Move the replay cursor. Caller (the replay loop) is responsible
        for only ever advancing forward — this class does not protect
        against look-ahead misuse by the caller, only against building
        market_out from bars beyond the cursor."""
        self._cursor = bar_index
        try:
            from data.backtest_ohlcv_cache import set_asof
            if self._df is not None and 0 <= self._cursor < len(self._df):
                set_asof(self._df.index[self._cursor])
        except Exception:
            pass

    def current_time(self):
        return self._df.index[self._cursor]

    def get_market_out(self, symbol: str, timeframe: str) -> dict:
        # P4c FIX: guard against empty DataFrame or out-of-bounds cursor
        if self._df is None or len(self._df) == 0:
            log.warning("[HistoricalMT5Provider] Empty DataFrame — returning error dict")
            return {"error": "empty_data", "df": None, "symbol": symbol, "timeframe": timeframe}
        if self._cursor < 0 or self._cursor >= len(self._df):
            log.warning(f"[HistoricalMT5Provider] cursor {self._cursor} out of bounds (len={len(self._df)})")
            self._cursor = max(0, min(self._cursor, len(self._df) - 1))

        # ── P5 FIX: Bounded rolling window for O(n²)→O(n) backtest perf ─────
        # ORIGINAL CODE (O(n²) complexity):
        #   df_slice = self._df.iloc[: self._cursor + 1].copy()
        # The full history-to-date slice grew with each bar, so indicator
        # computation (139 indicators + pattern detection) ran on an
        # ever-growing dataset: 1st bar→1 row, 2nd→2 rows, ..., nth→n rows.
        # Total work = sum(1..n) = O(n²). A 5,000-bar backtest was 2,500x
        # slower than a 100-bar one.
        #
        # FIX: Bound the slice to a fixed rolling window (300 bars).
        # Indicator cost is now constant per bar → O(n) total.
        # Window size = 300 matches live trading's limit=300 in
        # agents/market_agent.py for execution parity.
        # Early bars (when cursor < 300) naturally clamp to start via max(0, ...).
        LOOKBACK_BARS = 300
        start = max(0, self._cursor - LOOKBACK_BARS)
        df_slice = self._df.iloc[start : self._cursor + 1].copy()
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

        # ── PARITY FIX (audit Issue #8 / #9): real mtf_bias from causal resample.
        # Previously this returned a hardcoded NEUTRAL/LOW placeholder, which
        # caused SignalEngine.generate() in backtest to NEVER add the MTF bias
        # vote (1-2 points of bull/bear score) it adds in live trading —
        # a real parity violation affecting final signal direction.
        #
        # Fix: compute a real MTF bias from the H1 df_slice by causally
        # resampling to 4H and 1D (no look-ahead — each resampled bar at
        # time T only uses H1 bars with timestamp <= T), then derive a
        # bias using the same EMA-trend logic live MultiTimeframeAnalyzer
        # uses (just on a different timeframe's EMA stack).
        #
        # If resampling fails (e.g. insufficient bars), we fall back to
        # the previous NEUTRAL/LOW placeholder — same as live's failure
        # path in MarketAgent when MTF fetch returns None.
        mtf_bias = self._compute_mtf_bias(df_slice, symbol, timeframe)

        # FIX (2026-08-19 audit Bug 4): expose bar_time so downstream
        # regime_suppression.can_suppress() can use the historical bar's
        # timestamp instead of wall-clock time when running in backtest.
        # In live mode, this is just the latest bar's timestamp (same as
        # what the live cycle would pass).
        _bar_time = None
        try:
            if len(df_slice) > 0:
                _bar_time = df_slice.index[-1]
        except Exception:
            pass

        return {
            "df": df_slice,
            "ind_ctx": ind_ctx,
            "regime": regime_result,
            "regime_ctx": regime_ctx,
            "mtf_bias": mtf_bias,
            "symbol": symbol,
            "timeframe": timeframe,
            "data_source": "historical_replay",
            "bar_time": _bar_time,
            "current_time": _bar_time,
        }

    def _compute_mtf_bias(self, df_slice: "pd.DataFrame", symbol: str, timeframe: str) -> dict:
        """Compute a real MTF bias from the H1 df_slice via causal resampling.

        Mirrors what live `MultiTimeframeAnalyzer.analyze()` +
        `get_bias()` produce: a {'bias': BULLISH/BEARISH/NEUTRAL,
        'confidence': HIGH/MEDIUM/LOW} dict derived from EMA-trend
        agreement across higher timeframes.

        Causal: at cursor time T, the 4H and 1D bars we look at are
        those whose OPEN time is <= T. We never look at a 4H/1D bar
        that opens after T. (A 4H bar opening at T-2h and closing at
        T+2h would be partially forming in live — here we use its
        last-known state, which is exactly what live trading does
        too: live evaluates mid-bar.)

        Falls back to {"bias":"NEUTRAL","confidence":"LOW"} if there
        isn't enough data to compute EMAs (matches live's failure path).
        """
        try:
            if df_slice is None or len(df_slice) < 50:
                return {"bias": "NEUTRAL", "confidence": "LOW"}
            # Resample H1 → 4H and 1D, causal (closed bars only at each step).
            # Use label='left' closed='left' so each 4H bar covers [t, t+4h)
            # and we only use bars whose open is <= cursor time.
            current_time = df_slice.index[-1]
            ohlc = df_slice[["open", "high", "low", "close"]].copy()
            # Ensure volume column exists (resample needs it for some ops)
            if "volume" not in ohlc.columns:
                ohlc["volume"] = 0.0

            bias_votes = []  # list of (bias, confidence) per tf

            for tf_label, rule in [("4h", "4h"), ("1d", "1D")]:
                try:
                    resampled = ohlc.resample(rule, label="left", closed="left").agg({
                        "open": "first", "high": "max", "low": "min",
                        "close": "last", "volume": "sum",
                    }).dropna(subset=["open", "high", "low", "close"])
                    # Causal: only 4H/1D bars that have CLOSED by current_time
                    # (i.e. their open was at least 4h/1d ago).
                    if tf_label == "4h":
                        cutoff = current_time - pd.Timedelta(hours=4)
                    else:
                        cutoff = current_time - pd.Timedelta(days=1)
                    resampled = resampled[resampled.index <= cutoff]
                    if len(resampled) < 50:
                        continue
                    # EMA trend: 20 vs 50 vs 200 (or however many bars we have)
                    ema_fast = resampled["close"].ewm(span=20, adjust=False).mean()
                    ema_slow = resampled["close"].ewm(span=50, adjust=False).mean()
                    last_close = float(resampled["close"].iloc[-1])
                    last_fast = float(ema_fast.iloc[-1])
                    last_slow = float(ema_slow.iloc[-1])
                    if last_close > last_fast > last_slow:
                        bias_votes.append(("BULLISH", "HIGH"))
                    elif last_close > last_fast:
                        bias_votes.append(("BULLISH", "MEDIUM"))
                    elif last_close < last_fast < last_slow:
                        bias_votes.append(("BEARISH", "HIGH"))
                    elif last_close < last_fast:
                        bias_votes.append(("BEARISH", "MEDIUM"))
                    else:
                        bias_votes.append(("NEUTRAL", "LOW"))
                except Exception:
                    continue

            if not bias_votes:
                return {"bias": "NEUTRAL", "confidence": "LOW"}

            # Aggregate: ≥75% agreement → HIGH, ≥50% → MEDIUM, else NEUTRAL/LOW
            bullish = sum(1 for b, _ in bias_votes if b == "BULLISH")
            bearish = sum(1 for b, _ in bias_votes if b == "BEARISH")
            total = len(bias_votes)
            if bullish >= 0.75 * total:
                return {"bias": "BULLISH", "confidence": "HIGH" if bullish == total else "MEDIUM"}
            if bearish >= 0.75 * total:
                return {"bias": "BEARISH", "confidence": "HIGH" if bearish == total else "MEDIUM"}
            if bullish > bearish:
                return {"bias": "BULLISH", "confidence": "LOW"}
            if bearish > bullish:
                return {"bias": "BEARISH", "confidence": "LOW"}
            return {"bias": "NEUTRAL", "confidence": "LOW"}
        except Exception as e:
            log.debug(f"[HistoricalMT5Provider] MTF bias computation failed: {e}")
            return {"bias": "NEUTRAL", "confidence": "LOW"}