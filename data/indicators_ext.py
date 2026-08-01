"""
data/indicators_ext.py — Day 93 Extended Indicator Library (pandas-ta)
=====================================================================
Wraps the `pandas-ta` library to compute 60+ technical indicators
on OHLCV data. This is the COMPREHENSIVE replacement for the minimal
`data/indicators.py` (which uses the smaller `ta` package with only
~6 indicators).

Why pandas-ta over `ta`:
  - 200+ indicators vs ~30 in `ta`
  - Active maintenance, larger community
  - Single import: `import pandas_ta as ta` then `df.ta.<indicator>()`
  - Returns DataFrame columns directly (no need to call multiple methods)

MT5 indicator integration:
  MT5 doesn't natively expose indicator values via Python — it only
  provides raw OHLCV. So indicator computation ALWAYS happens in
  Python (this module), regardless of whether the OHLCV came from
  MT5 or from an external API (Twelve Data, yfinance, etc.). This is
  by design: it keeps indicator logic in one place + lets us switch
  data sources without re-implementing indicators.

Indicators covered (60+):
  Trend        : SMA, EMA, WMA, HMA, VWMA, MACD, ADX, Aroon, CCI, Ichimoku
  Momentum     : RSI, Stochastic, Williams %R, ROC, MFI, TSI, Ultimate
  Volatility   : ATR, Bollinger, Keltner, Donchian, StdDev
  Volume       : OBV, VWAP, CMF, MFI, A/D Line
  S/R Levels   : Pivot Points (Classic/Fibonacci/Camarilla)
  Candlestick  : Doji, Hammer, Engulfing, Shooting Star, Marubozu, etc.
  Patterns     : 30+ candlestick patterns via pandas_ta.cdl_pattern()

Usage:
    from data.indicators_ext import ExtendedIndicators
    ind = ExtendedIndicators()
    df_with_indicators = ind.add_all(df)
    # All indicators are now columns in df_with_indicators

    # Or compute individually:
    df = ind.add_ichimoku(df)
    df = ind.add_pivots(df)
    df = ind.add_candlestick_patterns(df)

The output DataFrame is fully compatible with the existing AnalysisAgent
pipeline — it adds columns rather than removing any.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("indicators_ext")

# Suppress pandas-ta's harmless runtime warnings (NaN on warmup periods)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas_ta")


def _ensure_ordered_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee df has a sorted, de-duplicated DatetimeIndex.

    Fixes: 'VWAP requires an ordered DatetimeIndex' and similar errors
    raised by pandas-ta indicators (vwap, session-anchored indicators,
    etc.) that assume a proper time index.

    - If the index is already a sorted DatetimeIndex, this is a no-op.
    - If the index is a DatetimeIndex but unsorted, it gets sorted.
    - If the index is NOT a DatetimeIndex, we look for a common
      timestamp column and build the DatetimeIndex from it.
    - Duplicate timestamps are dropped (keep last) since pandas-ta's
      vwap() and other cumulative/rolling calcs misbehave on ties.

    Day 102+ hotfix: expanded the column-name list. MT5's
    copy_rates_from() returns the timestamp in a column literally named
    'time', but other data sources (CSV exports, REST APIs, parquet
    snapshots) use a wider variety of names — and the original 4-entry
    list missed all of them, causing every VWAP call to hit the manual
    cumulative fallback. We now also try (case-insensitively):
      bar_time, candle_time, date_time, gmt_time, utc, utc_time,
      time_utc, ts, epoch, datetime_utc, bar_timestamp
    Plus a final heuristic: the FIRST column whose dtype is datetime64.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]
        return df

    # Day 102+: expanded candidate list — order matters (most likely first)
    _CANDIDATES = (
        "time", "timestamp", "datetime", "date",
        "bar_time", "candle_time", "date_time", "gmt_time",
        "utc", "utc_time", "time_utc", "ts", "epoch",
        "datetime_utc", "bar_timestamp",
    )

    # Pass 1: exact-name match (case-sensitive, then case-insensitive)
    for col in _CANDIDATES:
        if col in df.columns:
            converted = _try_build_datetime_index(df, col)
            if converted is not None:
                return converted
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in _CANDIDATES:
        if cand in lower_cols:
            converted = _try_build_datetime_index(df, lower_cols[cand])
            if converted is not None:
                return converted

    # Pass 2: dtype-based heuristic — first datetime64 column
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            converted = _try_build_datetime_index(df, col)
            if converted is not None:
                return converted

    # No usable timestamp info — leave df as-is; callers that need a
    # real DatetimeIndex (e.g. vwap) will fall back gracefully.
    return df


def _try_build_datetime_index(df: pd.DataFrame, col: str) -> pd.DataFrame | None:
    """Attempt to build a sorted, de-duplicated DatetimeIndex from `col`.

    Returns the rebuilt DataFrame on success, or None if the column
    can't be parsed as datetime (e.g. all-null, all-NaT, or non-time
    strings). Caller should try the next candidate on None.
    """
    try:
        idx = pd.to_datetime(df[col], utc=False, errors="coerce")
        # Require at least 50% non-null to accept — a column of mostly
        # NaNs isn't a real timestamp.
        if idx.notna().mean() < 0.5:
            return None
        # Drop rows where the timestamp didn't parse (rare but possible)
        df_clean = df.loc[idx.notna()].copy()
        df_clean = df_clean.set_index(pd.DatetimeIndex(idx[idx.notna()]))
        df_clean = df_clean.sort_index()
        if df_clean.index.has_duplicates:
            df_clean = df_clean[~df_clean.index.duplicated(keep="last")]
        return df_clean
    except Exception:
        return None


def _safe_set(df: pd.DataFrame, key: str, value) -> None:
    """Safely assign a pandas-ta result to a single column.

    pandas-ta sometimes returns a DataFrame (multiple columns) instead
    of a Series — e.g. ta.tsi() returns a DataFrame with one column
    named 'TSI_13_25'. This helper extracts the first column when
    that happens, so df[key] = value works without raising
    'Cannot set a DataFrame with multiple columns to a single column'.
    """
    if isinstance(value, pd.DataFrame):
        if len(value.columns) == 1:
            df[key] = value.iloc[:, 0]
        else:
            # Multiple columns — keep only the first
            df[key] = value.iloc[:, 0]
    elif isinstance(value, pd.Series):
        df[key] = value
    else:
        try:
            df[key] = value
        except Exception:
            pass


class ExtendedIndicators:
    """Comprehensive indicator layer built on pandas-ta.

    All methods follow the convention:
        - Take a DataFrame with columns: open, high, low, close, volume
        - Return the SAME DataFrame with new indicator columns appended
        - Never drop or rename existing columns
        - Idempotent: calling twice is safe (overwrites prior values)
    """

    # ─────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────

    def add_all(
        self,
        df: pd.DataFrame,
        include_patterns: bool = True,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compute ALL indicators + append to df.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            include_patterns: if True, also compute 30+ candlestick patterns
                              (slightly slower — skip for tight loops)
            symbol, timeframe: optional cache-key hints. add_all() is called
                from 25+ places across the codebase (market_agent.py,
                analysis/timeframe.py, structure.py, liquidity.py,
                fibonacci.py, smart_money.py, currency_strength.py, etc.),
                frequently on the EXACT SAME candle data within a single
                analysis cycle. Passing symbol/timeframe here lets
                core/indicator_cache.py (previously wired nowhere in the
                codebase — zero importers) skip recomputation when a
                caller asks for indicators on data it already has cached.
                Omitting them still works (backward compatible with every
                existing call site) — the cache key then falls back to the
                data hash alone, which still catches same-cycle duplicate
                calls, just without the extra symbol/timeframe namespacing.

        Returns: df with 60+ new indicator columns.
        """
        if df is None or len(df) < 30:
            log.warning(f"[IndicatorsExt] Insufficient data ({len(df) if df is not None else 0} rows) — need 30+")
            return df

        cache_key_parts = None
        try:
            from core.indicator_cache import global_indicator_cache
            # Hash the OHLCV block actually consumed below, not the whole
            # df (extra pre-existing columns from a prior partial pass
            # shouldn't bust the cache).
            _cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            data_hash = str(pd.util.hash_pandas_object(df[_cols], index=True).sum())
            cache_key_parts = (symbol or "_any", timeframe or "_any", "add_all",
                               {"include_patterns": include_patterns, "rows": len(df)}, data_hash)
            cached = global_indicator_cache.get(*cache_key_parts)
            if cached is not None:
                log.debug(f"[IndicatorsExt] add_all cache HIT ({symbol or '?'}/{timeframe or '?'}, {len(df)} rows)")
                return cached.copy()
        except Exception as e:
            log.debug(f"[IndicatorsExt] indicator_cache unavailable, computing uncached: {e}")
            cache_key_parts = None

        df = df.copy()

        # Fix: 'VWAP requires an ordered DatetimeIndex' — guarantee a sorted,
        # de-duplicated DatetimeIndex BEFORE any indicator (vwap, ichimoku,
        # session/pivot calcs) runs, instead of letting each one crash later.
        df = _ensure_ordered_datetime_index(df)

        # Make sure we have a volume column (pandas-ta needs it for some indicators)
        if "volume" not in df.columns:
            df["volume"] = 0.0

        # Add indicators in dependency order
        df = self.add_moving_averages(df)
        df = self.add_momentum(df)
        df = self.add_volatility(df)
        df = self.add_volume_indicators(df)
        df = self.add_volume_rsi(df)       # Day 97+ Book Page 49-50
        df = self.add_trend_strength(df)
        df = self.add_ichimoku(df)
        df = self.add_pivots(df)
        if include_patterns:
            df = self.add_candlestick_patterns(df)
        df = self.add_support_resistance(df)
        df = self.add_trend_signals(df)

        log.info(
            f"[IndicatorsExt] {len(df.columns)} columns after add_all "
            f"({len(df)} rows)"
        )

        if cache_key_parts is not None:
            try:
                from core.indicator_cache import global_indicator_cache
                symbol_k, timeframe_k, name_k, params_k, hash_k = cache_key_parts
                global_indicator_cache.set(symbol_k, timeframe_k, name_k, params_k, hash_k, df.copy())
            except Exception as e:
                log.debug(f"[IndicatorsExt] indicator_cache store failed (non-fatal): {e}")

        return df

    # ─────────────────────────────────────────────────────────
    # TREND: Moving Averages
    # ─────────────────────────────────────────────────────────

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """SMA, EMA, WMA, HMA, VWMA across multiple periods."""
        import pandas_ta as ta

        close = df["close"]
        # SMA — Simple Moving Average
        for period in (10, 20, 50, 100, 200):
            col = f"sma_{period}"
            if len(close) >= period:
                df[col] = ta.sma(close, length=period)

        # EMA — Exponential Moving Average
        for period in (5, 9, 13, 21, 34, 55, 89):
            col = f"ema_{period}"
            if len(close) >= period:
                df[col] = ta.ema(close, length=period)

        # WMA — Weighted Moving Average
        if len(close) >= 20:
            df["wma_20"] = ta.wma(close, length=20)

        # HMA — Hull Moving Average (low lag)
        if len(close) >= 16:
            df["hma_20"] = ta.hma(close, length=20)

        # VWMA — Volume Weighted Moving Average
        if "volume" in df.columns and len(close) >= 20:
            try:
                df["vwma_20"] = ta.vwma(close, df["volume"], length=20)
            except Exception:
                pass

        return df

    # ─────────────────────────────────────────────────────────
    # MOMENTUM
    # ─────────────────────────────────────────────────────────

    def add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI, Stochastic, Williams %R, ROC, MFI, TSI, Ultimate Oscillator."""
        import pandas_ta as ta

        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # RSI (multiple periods for cross-confirmation)
        for period in (7, 14, 21):
            col = "rsi" if period == 14 else f"rsi_{period}"
            df[col] = ta.rsi(close, length=period)

        # RSI signal zones (overbought/oversold/neutral)
        if "rsi" in df.columns:
            df["rsi_signal"] = df["rsi"].apply(self._rsi_zone)

        # Stochastic Oscillator
        try:
            stoch = ta.stoch(high, low, close, k=14, d=3)
            if stoch is not None and len(stoch.columns) >= 2:
                df["stoch_k"] = stoch.iloc[:, 0]
                df["stoch_d"] = stoch.iloc[:, 1]
        except Exception as e:
            log.debug(f"[IndicatorsExt] Stoch failed: {e}")

        # Williams %R
        try:
            _safe_set(df, "willr", ta.willr(high, low, close, length=14))
        except Exception:
            pass

        # Rate of Change (ROC)
        try:
            _safe_set(df, "roc_10", ta.roc(close, length=10))
            _safe_set(df, "roc_20", ta.roc(close, length=20))
        except Exception:
            pass

        # MFI — Money Flow Index (volume-weighted RSI)
        if "volume" in df.columns:
            try:
                df["mfi"] = ta.mfi(high, low, close, df["volume"], length=14)
            except Exception:
                pass

        # TSI — True Strength Index (returns Series)
        try:
            tsi_result = ta.tsi(close, fast=13, slow=25)
            # pandas_ta.tsi returns a Series, but just in case it's a
            # DataFrame with one column, extract the first column.
            if isinstance(tsi_result, pd.DataFrame):
                df["tsi"] = tsi_result.iloc[:, 0]
            else:
                df["tsi"] = tsi_result
        except Exception as e:
            log.debug(f"[IndicatorsExt] TSI failed: {e}")

        # Ultimate Oscillator
        try:
            uo_result = ta.uo(high, low, close)
            _safe_set(df, "uo", uo_result)
        except Exception as e:
            log.debug(f"[IndicatorsExt] UO failed: {e}")

        return df

    @staticmethod
    def _rsi_zone(rsi: float) -> str:
        """Classify RSI value into a zone label.

        Cross-file consistency fix: labels must match
        data/indicator_registry.py's canonical `_rsi_zone()` exactly
        ("bullish_zone"/"bearish_zone", not "bullish"/"bearish").
        Previously these two functions used different label strings
        for the same zones. Since ExtendedIndicators.add_all() writes
        'rsi_signal' directly, indicator_registry.add_canonical_indicators()
        (which only filled 'rsi_signal' when the column was absent)
        never overrode it — any downstream code checking for the
        registry's documented "bullish_zone"/"bearish_zone" labels would
        silently never match once data had passed through ExtendedIndicators.
        """
        if pd.isna(rsi):
            return "neutral"
        if rsi >= 70:
            return "overbought"
        if rsi <= 30:
            return "oversold"
        if rsi >= 55:
            return "bullish_zone"
        if rsi <= 45:
            return "bearish_zone"
        return "neutral"

    # ─────────────────────────────────────────────────────────
    # VOLATILITY
    # ─────────────────────────────────────────────────────────

    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR, Bollinger Bands, Keltner Channel, Donchian Channel, StdDev."""
        import pandas_ta as ta

        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        # ATR (multiple periods)
        for period in (7, 14, 21):
            col = "atr" if period == 14 else f"atr_{period}"
            df[col] = ta.atr(high, low, close, length=period)

        # Bollinger Bands
        try:
            bb = ta.bbands(close, length=20, std=2)
            if bb is not None and len(bb) > 0:
                # Column names can vary between pandas-ta versions —
                # use positional indexing instead of name-based.
                # Expected order: lower, mid, upper, bandwidth, %B
                cols = list(bb.columns)
                if len(cols) >= 5:
                    df["bb_lower"]  = bb.iloc[:, 0]
                    df["bb_middle"] = bb.iloc[:, 1]
                    df["bb_upper"]  = bb.iloc[:, 2]
                    df["bb_width"]  = bb.iloc[:, 3]
                    df["bb_pct"]    = bb.iloc[:, 4]
                elif len(cols) >= 3:
                    df["bb_lower"]  = bb.iloc[:, 0]
                    df["bb_middle"] = bb.iloc[:, 1]
                    df["bb_upper"]  = bb.iloc[:, 2]
        except Exception as e:
            log.debug(f"[IndicatorsExt] BBands failed: {e}")

        # Keltner Channel
        try:
            kc = ta.kc(high, low, close, length=20, scalar=2)
            if kc is not None and len(kc.columns) >= 3:
                df["kc_lower"]  = kc.iloc[:, 0]
                df["kc_middle"] = kc.iloc[:, 1]
                df["kc_upper"]  = kc.iloc[:, 2]
        except Exception as e:
            log.debug(f"[IndicatorsExt] KC failed: {e}")

        # Donchian Channel
        try:
            dc = ta.donchian(high, low, lower_length=20, upper_length=20)
            if dc is not None and len(dc.columns) >= 3:
                df["dc_lower"] = dc.iloc[:, 0]
                df["dc_upper"] = dc.iloc[:, 2]
        except Exception as e:
            log.debug(f"[IndicatorsExt] DC failed: {e}")

        # Standard Deviation
        try:
            _safe_set(df, "std_20", ta.stdev(close, length=20))
        except Exception:
            pass

        return df

    # ─────────────────────────────────────────────────────────
    # VOLUME INDICATORS
    # ─────────────────────────────────────────────────────────

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """OBV, VWAP, CMF, Accumulation/Distribution Line."""
        import pandas_ta as ta

        if "volume" not in df.columns:
            return df

        high = df["high"]; low = df["low"]
        close = df["close"]; vol = df["volume"]

        # On-Balance Volume
        df["obv"] = ta.obv(close, vol)

        # VWAP (anchored to start of session — useful for intraday)
        # Root-cause fix: pandas-ta's vwap() requires a sorted DatetimeIndex.
        # add_all() already normalizes the index via
        # _ensure_ordered_datetime_index(), but add_volume_indicators() can
        # also be called standalone, so we defensively re-check here instead
        # of silently swallowing the error like before.
        try:
            vwap_df = df
            if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing:
                vwap_df = _ensure_ordered_datetime_index(df)

            if isinstance(vwap_df.index, pd.DatetimeIndex):
                vwap_series = ta.vwap(
                    vwap_df["high"], vwap_df["low"], vwap_df["close"], vwap_df["volume"]
                )
                # Re-align back onto the original df's index/order in case
                # _ensure_ordered_datetime_index() re-sorted or de-duped rows.
                df["vwap"] = vwap_series.reindex(df.index)
            else:
                # Still no usable DatetimeIndex (e.g. no timestamp column at
                # all) — fall back to a manual cumulative-typical-price VWAP
                # so downstream code gets a real value instead of a missing
                # column.
                typical_price = (high + low + close) / 3.0
                cum_vol = vol.cumsum().replace(0, pd.NA)
                df["vwap"] = (typical_price * vol).cumsum() / cum_vol
                log.warning(
                    "[IndicatorsExt] No DatetimeIndex/timestamp column available — "
                    "used manual cumulative VWAP fallback instead of pandas-ta"
                )
        except Exception as e:
            log.warning(f"[IndicatorsExt] VWAP failed even after index fix: {e}")

        # Accumulation/Distribution Line
        df["ad"] = ta.ad(high, low, close, vol)

        # CMF — Chaikin Money Flow
        try:
            df["cmf"] = ta.cmf(high, low, close, vol, length=20)
        except Exception:
            pass

        return df

    # ─────────────────────────────────────────────────────────
    # TREND STRENGTH
    # ─────────────────────────────────────────────────────────

    def add_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD, ADX, Aroon, CCI."""
        import pandas_ta as ta

        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        # MACD
        try:
            macd = ta.macd(close, fast=12, slow=26, signal=9)
            if macd is not None and len(macd.columns) >= 3:
                df["macd"]        = macd.iloc[:, 0]
                df["macd_hist"]   = macd.iloc[:, 1]
                df["macd_signal"] = macd.iloc[:, 2]
                df["macd_cross"] = df.apply(
                    lambda r: "bullish_cross" if r.get("macd", 0) > r.get("macd_signal", 0)
                              else "bearish_cross",
                    axis=1
                )
        except Exception as e:
            log.debug(f"[IndicatorsExt] MACD failed: {e}")

        # ADX — Average Directional Index
        try:
            adx = ta.adx(high, low, close, length=14)
            if adx is not None and len(adx.columns) >= 3:
                df["adx"]      = adx.iloc[:, 0]
                df["di_plus"]  = adx.iloc[:, 1]
                df["di_minus"] = adx.iloc[:, 2]
        except Exception as e:
            log.debug(f"[IndicatorsExt] ADX failed: {e}")

        # Aroon
        try:
            aroon = ta.aroon(high, low, length=25)
            if aroon is not None and len(aroon.columns) >= 3:
                df["aroon_down"] = aroon.iloc[:, 0]
                df["aroon_up"]   = aroon.iloc[:, 1]
                df["aroon_osc"]  = aroon.iloc[:, 2]
        except Exception as e:
            log.debug(f"[IndicatorsExt] Aroon failed: {e}")

        # CCI — Commodity Channel Index
        try:
            _safe_set(df, "cci", ta.cci(high, low, close, length=20))
        except Exception as e:
            log.debug(f"[IndicatorsExt] CCI failed: {e}")

        return df

    # ─────────────────────────────────────────────────────────
    # VOLUME RSI (Book Page 49-50)
    # ─────────────────────────────────────────────────────────

    def add_volume_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Day 97+ Book Rule (Page 49-50): Volume RSI.

        Like RSI but uses volume instead of price:
          - Up-day volume = volume on days where close > prev close
          - Down-day volume = volume on days where close < prev close
          - Volume RS = sum(up_vol) / sum(down_vol)
          - Volume RSI = 100 - (100 / (1 + Volume RS))

        Signal (Page 50):
          - Crosses above 50% → bullish (buy)
          - Crosses below 50% → bearish (sell)
        """
        if "volume" not in df.columns or "close" not in df.columns:
            return df
        if len(df) < period + 1:
            return df

        try:
            close = pd.to_numeric(df["close"], errors="coerce").fillna(0)
            vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

            # Calculate up-volume and down-volume
            price_change = close.diff()
            up_vol = vol.where(price_change > 0, 0.0)
            down_vol = vol.where(price_change < 0, 0.0)

            # Rolling sums
            up_sum = up_vol.rolling(window=period, min_periods=period).sum()
            down_sum = down_vol.rolling(window=period, min_periods=period).sum()

            # Volume RS and RSI
            vol_rs = up_sum / down_sum.replace(0, 1e-10)  # avoid div by zero
            df["vol_rsi"] = (100 - (100 / (1 + vol_rs))).fillna(50)

            # 50% crossover signal (Page 50)
            df["vol_rsi_signal"] = df["vol_rsi"].apply(
                lambda x: "BUY" if x > 50 else "SELL" if x < 50 else "NEUTRAL"
            )

            log.debug(f"[IndicatorsExt] Volume RSI added (period={period})")
        except Exception as e:
            log.debug(f"[IndicatorsExt] Volume RSI failed: {e}")

        return df

    # ─────────────────────────────────────────────────────────
    # ICHIMOKU CLOUD
    # ─────────────────────────────────────────────────────────

    def add_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tenkan-sen, Kijun-sen, Senkou A/B, Chikou Span."""
        import pandas_ta as ta

        try:
            ich = ta.ichimoku(df["high"], df["low"], df["close"],
                              tenkan=9, kijun=26, senkou=52)
            # pandas_ta returns (dataframe, displaced dataframe) tuple
            if isinstance(ich, tuple) and len(ich) >= 1:
                ich_df = ich[0]
                for col in ich_df.columns:
                    df[f"ich_{col.lower()}"] = ich_df[col].values
        except Exception as e:
            log.debug(f"[IndicatorsExt] Ichimoku failed: {e}")
        return df

    # ─────────────────────────────────────────────────────────
    # PIVOT POINTS (Support/Resistance)
    # ─────────────────────────────────────────────────────────

    def add_pivots(self, df: pd.DataFrame, method: str = "classic") -> pd.DataFrame:
        """Pivot Points (Classic, Fibonacci, or Camarilla).

        Computed from the most recent completed candle. The pivot levels
        act as future support/resistance for the current session.
        """
        if len(df) < 2:
            return df

        # Use last completed candle (not the in-progress one)
        last = df.iloc[-2]
        h, l, c = float(last["high"]), float(last["low"]), float(last["close"])

        if method == "fibonacci":
            pivot = (h + l + c) / 3
            r1 = pivot + 0.382 * (h - l)
            r2 = pivot + 0.618 * (h - l)
            r3 = pivot + 1.000 * (h - l)
            s1 = pivot - 0.382 * (h - l)
            s2 = pivot - 0.618 * (h - l)
            s3 = pivot - 1.000 * (h - l)
        elif method == "camarilla":
            pivot = c
            r1 = c + (h - l) * 1.1 / 12
            r2 = c + (h - l) * 1.1 / 6
            r3 = c + (h - l) * 1.1 / 4
            s1 = c - (h - l) * 1.1 / 12
            s2 = c - (h - l) * 1.1 / 6
            s3 = c - (h - l) * 1.1 / 4
        else:  # classic
            pivot = (h + l + c) / 3
            r1 = 2 * pivot - l
            r2 = pivot + (h - l)
            r3 = h + 2 * (pivot - l)
            s1 = 2 * pivot - h
            s2 = pivot - (h - l)
            s3 = l - 2 * (h - pivot)

        # Fill forward — same pivot levels apply to all future candles
        # until the next day's pivot computation
        df["pivot_p"] = pivot
        df["pivot_r1"] = r1
        df["pivot_r2"] = r2
        df["pivot_r3"] = r3
        df["pivot_s1"] = s1
        df["pivot_s2"] = s2
        df["pivot_s3"] = s3
        return df

    # ─────────────────────────────────────────────────────────
    # CANDLESTICK PATTERNS
    # ─────────────────────────────────────────────────────────

    def add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """30+ candlestick patterns via pandas_ta.cdl_pattern().

        Each pattern gets its own column with values:
            100 = bullish pattern detected
            -100 = bearish pattern detected
            0 = no pattern
        """
        import pandas_ta as ta

        try:
            # All patterns at once (returns DataFrame with multiple columns)
            patterns = ta.cdl_pattern(
                df["open"], df["high"], df["low"], df["close"], name="all"
            )
            if patterns is not None and len(patterns) > 0:
                # Fix: Use single pd.concat instead of loop to avoid DataFrame
                # fragmentation (PerformanceWarning). Adding 30+ columns one-by-one
                # in a loop causes pandas to rebuild the internal block manager
                # repeatedly, leading to fragmentation + slower performance.
                patterns_renamed = patterns.rename(columns=lambda x: f"cdl_{x.lower()}")
                df = pd.concat([df, patterns_renamed], axis=1)
        except Exception as e:
            log.debug(f"[IndicatorsExt] Candlestick patterns failed: {e}")
        return df

    # ─────────────────────────────────────────────────────────
    # SUPPORT / RESISTANCE (Fractal-based)
    # ─────────────────────────────────────────────────────────

    def add_support_resistance(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Detect swing highs/lows (fractal pivots) — basic S/R levels.

        A swing high is a candle whose `window` candles on each side
        all have lower highs. Mark with 1.0. A swing low is the
        mirror image, marked with -1.0.
        """
        df = df.copy()
        df["sr_signal"] = 0.0

        # Swing highs
        high = df["high"].values
        for i in range(window, len(df) - window):
            window_high = high[i - window:i + window + 1]
            if high[i] == window_high.max() and list(high).count(high[i]) == 1:
                df.iloc[i, df.columns.get_loc("sr_signal")] = 1.0

        # Swing lows
        low = df["low"].values
        for i in range(window, len(df) - window):
            window_low = low[i - window:i + window + 1]
            if low[i] == window_low.min() and list(low).count(low[i]) == 1:
                df.iloc[i, df.columns.get_loc("sr_signal")] = -1.0

        return df

    # ─────────────────────────────────────────────────────────
    # TREND SIGNALS (composite)
    # ─────────────────────────────────────────────────────────

    def add_trend_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite trend signal from EMA alignment + ADX.

        trend_signal values:
            'STRONG_UP'    : EMA9 > EMA21 > SMA50 + ADX >= 25
            'UP'           : EMA9 > EMA21 (no ADX confirm)
            'STRONG_DOWN'  : EMA9 < EMA21 < SMA50 + ADX >= 25
            'DOWN'         : EMA9 < EMA21 (no ADX confirm)
            'RANGING'      : ADX < 20 (no clear trend)
            'NEUTRAL'      : default
        """
        def classify(row):
            ema9 = row.get("ema_9")
            ema21 = row.get("ema_21")
            sma50 = row.get("sma_50")
            adx = row.get("adx")
            if any(pd.isna(v) for v in (ema9, ema21)):
                return "NEUTRAL"
            bullish_align = ema9 > ema21 and (sma50 is None or pd.isna(sma50) or ema21 > sma50)
            bearish_align = ema9 < ema21 and (sma50 is None or pd.isna(sma50) or ema21 < sma50)
            strong = adx is not None and not pd.isna(adx) and adx >= 25
            ranging = adx is not None and not pd.isna(adx) and adx < 20
            if ranging:
                return "RANGING"
            if bullish_align and strong:
                return "STRONG_UP"
            if bearish_align and strong:
                return "STRONG_DOWN"
            if bullish_align:
                return "UP"
            if bearish_align:
                return "DOWN"
            return "NEUTRAL"

        df["trend_signal"] = df.apply(classify, axis=1)
        return df

    # ─────────────────────────────────────────────────────────
    # AI CONTEXT (compact summary for LLM prompt)
    # ─────────────────────────────────────────────────────────

    def get_ai_context(self, df: pd.DataFrame) -> dict:
        """Compact summary of latest indicator values for LLM prompt."""
        if df is None or len(df) == 0:
            return {}
        last = df.iloc[-1]

        def safe(key, default=None, round_to=None):
            v = last.get(key, default)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return default
            if round_to and isinstance(v, (int, float)):
                return round(float(v), round_to)
            return v

        return {
            "price":          safe("close", round_to=5),
            "trend":          safe("trend_signal"),
            "rsi":            safe("rsi", round_to=1),
            "rsi_signal":     safe("rsi_signal"),
            "macd":           safe("macd", round_to=5),
            "macd_signal":    safe("macd_signal", round_to=5),
            "macd_cross":     safe("macd_cross"),
            "adx":            safe("adx", round_to=1),
            "atr":            safe("atr", round_to=5),
            "bb_pct":         safe("bb_pct", round_to=2),
            "bb_width":       safe("bb_width", round_to=2),
            "stoch_k":        safe("stoch_k", round_to=1),
            "stoch_d":        safe("stoch_d", round_to=1),
            "ema_9":          safe("ema_9", round_to=5),
            "ema_21":         safe("ema_21", round_to=5),
            "sma_50":         safe("sma_50", round_to=5),
            "sma_200":        safe("sma_200", round_to=5),
            "cci":            safe("cci", round_to=1),
            "obv":            safe("obv"),
            "pivot_p":        safe("pivot_p", round_to=5),
            "pivot_r1":       safe("pivot_r1", round_to=5),
            "pivot_s1":       safe("pivot_s1", round_to=5),
        }

    def print_summary(self, df: pd.DataFrame) -> None:
        """Print a one-line summary of the latest indicator state."""
        ctx = self.get_ai_context(df)
        if not ctx:
            return
        log.info(
            f"[IndicatorsExt] trend={ctx.get('trend','?')} | "
            f"RSI={ctx.get('rsi','?')} ({ctx.get('rsi_signal','?')}) | "
            f"MACD cross={ctx.get('macd_cross','?')} | "
            f"ADX={ctx.get('adx','?')} | "
            f"ATR={ctx.get('atr','?')} | "
            f"BB%={ctx.get('bb_pct','?')} | "
            f"Stoch K/D={ctx.get('stoch_k','?')}/{ctx.get('stoch_d','?')}"
        )