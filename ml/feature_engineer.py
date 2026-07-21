"""
ml/feature_engineer.py — Feature Engineering Layer (Day 68)
=============================================================

Transforms raw market data + analysis contexts into a structured
100+ feature vector for ML training and inference.

Feature groups (all in a single flat dict):
  - Price features       (15): change_1/5/20, distance_ma20/50/200, dist_sr, ...
  - Indicator features   (25): rsi7/14/21, macd_line/signal/hist, bb_position,
                              atr_pct, volume_ratio, ema_distances, ...
  - Pattern features     (20): hammer, engulfing, doji, double_top/bottom,
                              head_shoulders, triangle, flag, wedge, fib distances
  - Context features     (40): session one-hot, hour, day_of_week, news_risk,
                              currency_strength (EUR/USD/GBP/JPY), dxy, gold, vix,
                              sp500, us10y, days_to_news, hours_to_news
  - Multi-timeframe     (10): mtf_bias, h1_trend, h4_trend, d1_trend alignment

Total: ~110 features (depending on availability of analysis contexts).

CRITICAL: No data leakage. Every feature uses ONLY information available at
the time of the trade decision. Future candles are never touched.

Public API:
    FeatureEngineer().build_feature_vector(df, analysis_out, pair, timeframe)
        -> dict with ~110 numeric features
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("feature_engineer")


# ── Helpers ─────────────────────────────────────────────────────────

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _pips(price_diff: float, pair: str = "EURUSD") -> float:
    """Convert a price difference to pips (handles JPY pairs)."""
    pair = pair.upper()
    pip_size = 0.01 if pair.endswith("JPY") else 0.0001
    return price_diff / pip_size if pip_size > 0 else 0.0


# ── Feature groups ─────────────────────────────────────────────────

class FeatureEngineer:
    """Generates a flat ~110-feature dict from market data + analysis contexts."""

    def __init__(self):
        self.last_feature_count = 0

    def build_feature_vector(
        self,
        df: pd.DataFrame,
        analysis_out: Optional[Dict[str, Any]] = None,
        pair: str = "EURUSD",
        timeframe: str = "15m",
    ) -> Dict[str, float]:
        """Build a single-row feature vector for the LATEST candle.

        Args:
            df: OHLCV dataframe with indicators already added (from MarketAgent).
            analysis_out: The full AnalysisAgent output dict (optional — used for
                          context features: session, intermarket, news, etc.).
            pair: Symbol (e.g. "EURUSD").
            timeframe: Timeframe label (e.g. "15m").

        Returns:
            Dict[str, float] with ~110 features. All values are numeric (or 0.0
            if data unavailable).
        """
        features: Dict[str, float] = {}
        analysis_out = analysis_out or {}

        if df is None or len(df) == 0:
            log.warning("[FeatureEngineer] empty df — returning zero feature vector")
            return features

        last = df.iloc[-1]

        # ── 1. Price features ──────────────────────────────────────
        features.update(self._price_features(df, last, pair))

        # ── 2. Indicator features ──────────────────────────────────
        features.update(self._indicator_features(df, last, pair))

        # ── 3. Pattern features ────────────────────────────────────
        features.update(self._pattern_features(df, last, analysis_out))

        # ── 4. Context features ────────────────────────────────────
        features.update(self._context_features(analysis_out, pair, timeframe))

        # ── 5. Multi-timeframe features ────────────────────────────
        features.update(self._mtf_features(analysis_out))

        # ── 6. SMC + Liquidity features ────────────────────────────
        features.update(self._smc_liquidity_features(analysis_out, df))

        # ── 7. Confluence + sentiment (Day 66/67) ──────────────────
        features.update(self._confluence_features(analysis_out))

        self.last_feature_count = len(features)
        return features

    # ── 1. Price features ─────────────────────────────────────────

    def _price_features(self, df: pd.DataFrame, last: pd.Series, pair: str) -> Dict[str, float]:
        f: Dict[str, float] = {}
        close = _safe_float(last.get("close"))
        open_ = _safe_float(last.get("open"))
        high = _safe_float(last.get("high"))
        low = _safe_float(last.get("low"))
        volume = _safe_float(last.get("volume"))

        # OHLCV raw
        f["price_open"] = open_
        f["price_high"] = high
        f["price_low"] = low
        f["price_close"] = close
        f["price_volume"] = volume

        # Candle body + range
        f["candle_body"] = close - open_
        f["candle_range"] = high - low if high > low else 0.0
        f["candle_body_ratio"] = (abs(close - open_) / (high - low)) if (high - low) > 0 else 0.0
        f["candle_upper_wick"] = high - max(open_, close)
        f["candle_lower_wick"] = min(open_, close) - low

        # Price momentum (change over N candles)
        for n in (1, 3, 5, 10, 20):
            if len(df) > n:
                prev_close = _safe_float(df.iloc[-n - 1].get("close"))
                if prev_close > 0:
                    f[f"change_{n}"] = (close - prev_close) / prev_close
                else:
                    f[f"change_{n}"] = 0.0
            else:
                f[f"change_{n}"] = 0.0

        # Moving average distances (price vs MA in %)
        for n in (9, 20, 50, 200):
            col = f"sma_{n}" if f"sma_{n}" in df.columns else f"ema_{n}" if f"ema_{n}" in df.columns else None
            if col:
                ma_val = _safe_float(last.get(col))
                if ma_val > 0:
                    f[f"distance_ma{n}"] = (close - ma_val) / ma_val
                else:
                    f[f"distance_ma{n}"] = 0.0
            else:
                f[f"distance_ma{n}"] = 0.0

        # High/Low range over N candles
        for n in (5, 20, 50):
            if len(df) >= n:
                window = df.tail(n)
                f[f"high_{n}"] = _safe_float(window["high"].max())
                f[f"low_{n}"] = _safe_float(window["low"].min())
                f[f"range_{n}_pips"] = _pips(f[f"high_{n}"] - f[f"low_{n}"], pair)
            else:
                f[f"high_{n}"] = 0.0
                f[f"low_{n}"] = 0.0
                f[f"range_{n}_pips"] = 0.0

        return f

    # ── 2. Indicator features ─────────────────────────────────────

    def _indicator_features(self, df: pd.DataFrame, last: pd.Series, pair: str) -> Dict[str, float]:
        f: Dict[str, float] = {}

        # RSI at multiple windows
        # AUDIT FIX: for n==14 the old fallback set col="rsi" unconditionally
        # (a truthy string) without checking whether a "rsi" column actually
        # existed on df — so on raw historical OHLCV (no precomputed "rsi"
        # column) it read a nonexistent column via last.get("rsi") -> None
        # -> _safe_float default 0.0, SKIPPING the on-the-fly computation
        # below entirely. rsi_14 was therefore also constant 0.0 for the
        # whole training set, exactly like macd/atr/ema were.
        for n in (7, 14, 21):
            if f"rsi_{n}" in df.columns:
                col = f"rsi_{n}"
            elif n == 14 and "rsi" in df.columns:
                col = "rsi"
            else:
                col = None
            if col:
                f[f"rsi_{n}"] = _safe_float(last.get(col))
            else:
                # Compute on the fly if missing
                try:
                    delta = df["close"].diff()
                    gain = delta.clip(lower=0).rolling(n).mean()
                    loss = (-delta.clip(upper=0)).rolling(n).mean()
                    rs = gain / loss.replace(0, np.nan)
                    rsi = 100 - (100 / (1 + rs))
                    f[f"rsi_{n}"] = _safe_float(rsi.iloc[-1], 50.0)
                except Exception:
                    f[f"rsi_{n}"] = 50.0

        # RSI extremes
        f["rsi_overbought"] = 1.0 if f.get("rsi_14", 50) > 70 else 0.0
        f["rsi_oversold"] = 1.0 if f.get("rsi_14", 50) < 30 else 0.0

        # MACD
        # AUDIT FIX: previously this block only *read* macd/macd_signal/
        # macd_diff/macd_hist columns if a caller had already precomputed
        # them onto `df`. During historical training (add_features() in
        # scripts/train_models_quick.py calls build_feature_vector() on raw
        # OHLCV with no precomputed indicator columns), none of these
        # columns exist, so every row silently got macd=0.0 — a constant
        # column that then gets dropped as "zero-variance" and the model
        # never sees MACD at all. Compute it directly from close price (the
        # same fallback pattern already used for RSI above) whenever the
        # precomputed columns aren't present, so MACD always carries real
        # signal regardless of the caller.
        if all(c in df.columns for c in ("macd", "macd_signal")):
            macd_line = df["macd"]
            macd_signal_line = df["macd_signal"]
        else:
            ema_12 = df["close"].ewm(span=12, adjust=False).mean()
            ema_26 = df["close"].ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist_line = macd_line - macd_signal_line

        f["macd"] = _safe_float(macd_line.iloc[-1])
        f["macd_signal"] = _safe_float(macd_signal_line.iloc[-1])
        f["macd_diff"] = _safe_float(macd_hist_line.iloc[-1])
        f["macd_hist"] = _safe_float(macd_hist_line.iloc[-1])

        f["macd_cross_up"] = 1.0 if (len(df) >= 2 and
                                      _safe_float(macd_line.iloc[-1]) > _safe_float(macd_signal_line.iloc[-1]) and
                                      _safe_float(macd_line.iloc[-2]) <= _safe_float(macd_signal_line.iloc[-2])) else 0.0
        f["macd_cross_down"] = 1.0 if (len(df) >= 2 and
                                        _safe_float(macd_line.iloc[-1]) < _safe_float(macd_signal_line.iloc[-1]) and
                                        _safe_float(macd_line.iloc[-2]) >= _safe_float(macd_signal_line.iloc[-2])) else 0.0

        # Bollinger Bands position (0 = lower, 0.5 = middle, 1 = upper)
        if "bb_high" in df.columns and "bb_low" in df.columns:
            bb_high = _safe_float(last.get("bb_high"))
            bb_low = _safe_float(last.get("bb_low"))
            close = _safe_float(last.get("close"))
            f["bb_position"] = ((close - bb_low) / (bb_high - bb_low)) if (bb_high - bb_low) > 0 else 0.5
        else:
            f["bb_position"] = 0.5

        # ATR + ATR as % of price
        # AUDIT FIX: same class of bug as MACD above — `atr`/`atr_14` are
        # only present if a caller precomputed them. Compute a 14-period
        # ATR on the fly from high/low/close whenever neither column exists,
        # instead of silently defaulting to a constant 0.0.
        atr_col = "atr" if "atr" in df.columns else ("atr_14" if "atr_14" in df.columns else None)
        if atr_col is not None:
            atr = _safe_float(last.get(atr_col))
        else:
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ], axis=1).max(axis=1)
            atr = _safe_float(tr.rolling(14).mean().iloc[-1])
        f["atr"] = atr
        close = _safe_float(last.get("close"))
        f["atr_percentage"] = (atr / close) if close > 0 else 0.0
        f["atr_pips"] = _pips(atr, pair)

        # Volume ratio (current vs 20-period average)
        if "volume" in df.columns and len(df) >= 20:
            vol_avg = _safe_float(df["volume"].tail(20).mean())
            f["volume_ratio"] = (_safe_float(last.get("volume")) / vol_avg) if vol_avg > 0 else 1.0
        else:
            f["volume_ratio"] = 1.0

        # EMA distances + EMA alignment (trend strength)
        # AUDIT FIX: same class of bug as MACD/ATR above — ema_9/ema_20/
        # ema_50 are only present if a caller precomputed them, so on raw
        # historical OHLCV every ema_*_distance and both alignment flags
        # silently collapsed to a constant 0.0 (dropped as zero-variance,
        # even though EMA alignment is one of the highest-value trend
        # features for a tree model). Compute EMAs on the fly from close
        # whenever the precomputed columns aren't present.
        ema_vals = {}
        for n in (9, 20, 50):
            col = f"ema_{n}"
            if col in df.columns:
                ema_vals[n] = _safe_float(last.get(col))
            else:
                ema_vals[n] = _safe_float(df["close"].ewm(span=n, adjust=False).mean().iloc[-1])
            ema_val = ema_vals[n]
            f[f"ema_{n}_distance"] = ((close - ema_val) / ema_val) if ema_val > 0 else 0.0

        e9, e20, e50 = ema_vals[9], ema_vals[20], ema_vals[50]
        f["ema_bullish_alignment"] = 1.0 if (e9 > e20 > e50) else 0.0
        f["ema_bearish_alignment"] = 1.0 if (e9 < e20 < e50) else 0.0

        return f

    # ── 3. Pattern features ───────────────────────────────────────

    def _pattern_features(self, df: pd.DataFrame, last: pd.Series, analysis_out: Dict) -> Dict[str, float]:
        f: Dict[str, float] = {}

        # Candlestick patterns (one-hot from pattern columns)
        candle_patterns = ["doji", "hammer", "shooting_star", "pin_bar",
                          "bullish_engulfing", "bearish_engulfing",
                          "morning_star", "evening_star"]
        for p in candle_patterns:
            col = p
            if col in df.columns:
                val = last.get(col, "")
                f[f"pat_{p}"] = 1.0 if (val and val != "none" and str(val).lower() != "nan") else 0.0
            else:
                f[f"pat_{p}"] = 0.0

        # Advanced chart patterns from AnalysisAgent output
        adv = analysis_out.get("advanced_pat_ctx") or {}
        if isinstance(adv, dict):
            chart_patterns = ["head_and_shoulders", "inverse_head_and_shoulders",
                              "double_top", "double_bottom",
                              "ascending_triangle", "descending_triangle", "symmetrical_triangle",
                              "bull_flag", "bear_flag",
                              "rising_wedge", "falling_wedge",
                              "cup_and_handle"]
            recent = adv.get("recent_patterns", []) if isinstance(adv.get("recent_patterns"), list) else []
            recent_names = [str(p.get("pattern", "")).lower() for p in recent if isinstance(p, dict)]
            for p in chart_patterns:
                f[f"adv_{p}"] = 1.0 if any(p in name for name in recent_names) else 0.0
        else:
            for p in ["head_and_shoulders", "inverse_head_and_shoulders",
                      "double_top", "double_bottom",
                      "ascending_triangle", "descending_triangle", "symmetrical_triangle",
                      "bull_flag", "bear_flag",
                      "rising_wedge", "falling_wedge",
                      "cup_and_handle"]:
                f[f"adv_{p}"] = 0.0

        # Fibonacci proximity
        fib = analysis_out.get("fib_ctx") or {}
        if isinstance(fib, dict):
            close = _safe_float(last.get("close"))
            for level_name, level_key in [("236", "23.6"), ("382", "38.2"),
                                          ("500", "50.0"), ("618", "61.8"),
                                          ("786", "78.6")]:
                retracements = fib.get("retracements") or {}
                level_val = _safe_float(retracements.get(level_key) if isinstance(retracements, dict) else 0)
                if level_val > 0 and close > 0:
                    f[f"fib_{level_name}_distance_pips"] = _pips(abs(close - level_val), "EURUSD")
                else:
                    f[f"fib_{level_name}_distance_pips"] = 0.0
            # In fib zone (within 10 pips of any key level)
            fib_distances = [f.get(f"fib_{lvl}_distance_pips", 999) for lvl in ("382", "500", "618", "786")]
            f["in_fib_zone"] = 1.0 if min(fib_distances) <= 10 else 0.0
            f["fib_zone"] = f["in_fib_zone"]
        else:
            for lvl in ("236", "382", "500", "618", "786"):
                f[f"fib_{lvl}_distance_pips"] = 0.0
            f["in_fib_zone"] = 0.0
            f["fib_zone"] = 0.0

        return f

    # ── 4. Context features ───────────────────────────────────────

    def _context_features(self, a: Dict, pair: str, timeframe: str) -> Dict[str, float]:
        f: Dict[str, float] = {}

        # Session one-hot encoding
        session_ctx = a.get("session_ctx") or {}
        current_session = (session_ctx.get("current_session") or "BETWEEN_SESSIONS").upper() if isinstance(session_ctx, dict) else "BETWEEN_SESSIONS"
        for s in ("LONDON", "NEW_YORK", "TOKYO", "SYDNEY", "ASIAN", "BETWEEN_SESSIONS", "DEAD_ZONE"):
            f[f"session_{s.lower()}"] = 1.0 if s in current_session else 0.0
        f["session_overlap"] = 1.0 if "OVERLAP" in current_session else 0.0
        f["session_trade_quality"] = {"BEST": 1.0, "GOOD": 0.7, "CAUTION": 0.3, "LOW": 0.1}.get(
            (session_ctx.get("trade_quality") or "").upper() if isinstance(session_ctx, dict) else "", 0.5
        )

        # Time features
        # Timezone policy: use pd.Timestamp.now(tz="UTC") everywhere instead
        # of datetime.now(timezone.utc)/datetime.utcnow() so every "now" value
        # in the pipeline is the same tz-aware UTC type as DataFrame indices.
        now = pd.Timestamp.now(tz="UTC")
        f["hour_utc"] = float(now.hour)
        f["day_of_week"] = float(now.weekday())  # 0=Mon, 6=Sun
        f["is_weekend"] = 1.0 if now.weekday() >= 5 else 0.0
        f["is_monday_open"] = 1.0 if now.weekday() == 0 and now.hour < 12 else 0.0
        f["is_friday_close"] = 1.0 if now.weekday() == 4 and now.hour >= 20 else 0.0

        # News context (Day 66)
        news = a.get("news_intelligence") or {}
        if isinstance(news, dict):
            f["news_blocked"] = 1.0 if news.get("blocked") else 0.0
            f["news_confidence_change"] = _safe_float(news.get("confidence_change"))
            next_ev = news.get("next_high_impact_event") or {}
            if isinstance(next_ev, dict):
                mins_until = _safe_float(next_ev.get("minutes_until"), 9999)
                f["hours_to_news"] = max(0.0, mins_until / 60.0) if mins_until > 0 else 0.0
                f["high_impact_nearby"] = 1.0 if 0 < mins_until <= 60 else 0.0
            else:
                f["hours_to_news"] = 24.0
                f["high_impact_nearby"] = 0.0
            # News bias one-hot
            bias = (news.get("news_bias") or "NEUTRAL").upper()
            f["news_bullish"] = 1.0 if bias == "BULLISH" else 0.0
            f["news_bearish"] = 1.0 if bias == "BEARISH" else 0.0
            f["news_neutral"] = 1.0 if bias == "NEUTRAL" else 0.0
        else:
            f["news_blocked"] = 0.0
            f["news_confidence_change"] = 0.0
            f["hours_to_news"] = 24.0
            f["high_impact_nearby"] = 0.0
            f["news_bullish"] = 0.0
            f["news_bearish"] = 0.0
            f["news_neutral"] = 1.0

        # Currency strength (Day 64) — extract from intermarket_ctx or sentiment_ctx
        inter = a.get("intermarket_ctx") or {}
        if isinstance(inter, dict):
            base_cur = pair[:3].upper()
            quote_cur = pair[3:6].upper()
            # Try to extract per-currency bias
            macro_bias = (inter.get("macro_pair_bias") or "NEUTRAL").upper()
            f["currency_strength_base"] = 50.0  # default neutral
            f["currency_strength_quote"] = 50.0
            f["currency_strength_gap"] = 0.0
            if macro_bias == "BULLISH":
                f["currency_strength_base"] = 70.0
                f["currency_strength_quote"] = 35.0
                f["currency_strength_gap"] = 35.0
            elif macro_bias == "BEARISH":
                f["currency_strength_base"] = 35.0
                f["currency_strength_quote"] = 70.0
                f["currency_strength_gap"] = -35.0
            # Direct keys (EUR_strength, USD_strength) if available
            for cur in ("EUR", "USD", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"):
                val = inter.get(f"{cur.lower()}_strength") or inter.get(f"{cur}_strength")
                f[f"{cur.lower()}_strength"] = _safe_float(val, 50.0)
        else:
            f["currency_strength_base"] = 50.0
            f["currency_strength_quote"] = 50.0
            f["currency_strength_gap"] = 0.0
            for cur in ("EUR", "USD", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"):
                f[f"{cur.lower()}_strength"] = 50.0

        # Intermarket (Day 65) — DXY, Gold, VIX, SP500, US10Y
        if isinstance(inter, dict):
            f["dxy_trend"] = {"UP": 1.0, "DOWN": -1.0, "FLAT": 0.0}.get(
                (inter.get("dxy_trend") or "FLAT").upper(), 0.0)
            f["gold_trend"] = {"UP": 1.0, "DOWN": -1.0, "FLAT": 0.0}.get(
                (inter.get("gold_trend") or "FLAT").upper(), 0.0)
            f["vix_level"] = _safe_float(inter.get("vix_value"), 18.0)
            f["vix_fear_elevated"] = 1.0 if f["vix_level"] >= 25 else 0.0
            f["sp500_trend"] = {"UP": 1.0, "DOWN": -1.0, "FLAT": 0.0}.get(
                (inter.get("sp500_trend") or "FLAT").upper(), 0.0)
            f["us10y_trend"] = {"UP": 1.0, "DOWN": -1.0, "FLAT": 0.0}.get(
                (inter.get("us10y_trend") or "FLAT").upper(), 0.0)
            f["macro_score"] = _safe_float(inter.get("macro_score"))
            f["macro_regime_risk_on"] = 1.0 if (inter.get("macro_regime") or "").upper() == "RISK_ON" else 0.0
            f["macro_regime_risk_off"] = 1.0 if (inter.get("macro_regime") or "").upper() == "RISK_OFF" else 0.0
            f["cross_asset_confirmed"] = 1.0 if inter.get("cross_asset_confirmed") else 0.0
        else:
            for k in ("dxy_trend", "gold_trend", "sp500_trend", "us10y_trend"):
                f[k] = 0.0
            f["vix_level"] = 18.0
            f["vix_fear_elevated"] = 0.0
            f["macro_score"] = 0.0
            f["macro_regime_risk_on"] = 0.0
            f["macro_regime_risk_off"] = 0.0
            f["cross_asset_confirmed"] = 0.0

        # S/R proximity from analysis
        sr = a.get("sr_ctx") or {}
        if isinstance(sr, dict):
            f["sr_location"] = {"ABOVE_RESISTANCE": 1.0, "AT_RESISTANCE": 0.8,
                                "BETWEEN": 0.5, "AT_SUPPORT": 0.2,
                                "BELOW_SUPPORT": 0.0}.get(
                (sr.get("location") or "BETWEEN").upper(), 0.5)
            f["near_support"] = 1.0 if "SUPPORT" in (sr.get("location") or "").upper() else 0.0
            f["near_resistance"] = 1.0 if "RESISTANCE" in (sr.get("location") or "").upper() else 0.0
        else:
            f["sr_location"] = 0.5
            f["near_support"] = 0.0
            f["near_resistance"] = 0.0

        return f

    # ── 5. Multi-timeframe features ───────────────────────────────

    def _mtf_features(self, a: Dict) -> Dict[str, float]:
        f: Dict[str, float] = {}
        mtf = a.get("mtf_bias") or a.get("market_ctx", {}).get("mtf_bias") if isinstance(a.get("market_ctx"), dict) else None
        if isinstance(mtf, str):
            mtf_upper = mtf.upper()
            f["mtf_bullish"] = 1.0 if "BULL" in mtf_upper else 0.0
            f["mtf_bearish"] = 1.0 if "BEAR" in mtf_upper else 0.0
            f["mtf_neutral"] = 1.0 if "NEUTRAL" in mtf_upper else 0.0
        else:
            f["mtf_bullish"] = 0.0
            f["mtf_bearish"] = 0.0
            f["mtf_neutral"] = 1.0
        # HTF trend alignment (from smc_ctx if available)
        smc = a.get("smc_ctx") or {}
        if isinstance(smc, dict):
            f["smc_trend_aligned"] = 1.0 if smc.get("trend_aligned") else 0.0
            f["smc_grade_a_plus"] = 1.0 if (smc.get("grade") or "").upper() == "A+" else 0.0
            f["smc_grade_a"] = 1.0 if (smc.get("grade") or "").upper() == "A" else 0.0
        else:
            f["smc_trend_aligned"] = 0.0
            f["smc_grade_a_plus"] = 0.0
            f["smc_grade_a"] = 0.0
        return f

    # ── 6. SMC + Liquidity features ───────────────────────────────

    def _smc_liquidity_features(self, a: Dict, df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        f: Dict[str, float] = {}
        smc = a.get("smc_ctx") or {}
        if isinstance(smc, dict) and smc:
            signal = (smc.get("signal") or "NEUTRAL").upper()
            f["smc_buy"] = 1.0 if signal in ("BUY", "BULLISH") else 0.0
            f["smc_sell"] = 1.0 if signal in ("SELL", "BEARISH") else 0.0
            f["smc_neutral"] = 1.0 if signal in ("NEUTRAL", "WAIT", "") else 0.0
            f["smc_confluence_score"] = _safe_float(smc.get("confluence_score") or smc.get("score"))
            f["bos_detected"] = 1.0 if smc.get("bos") else 0.0
            f["choch_detected"] = 1.0 if smc.get("choch") else 0.0
            f["order_block_tap"] = 1.0 if smc.get("order_block") else 0.0
            f["fvg_detected"] = 1.0 if smc.get("fvg") else 0.0
            # Liquidity sweep
            sweep = smc.get("liquidity_sweep") or smc.get("sweep") or {}
            if isinstance(sweep, dict):
                f["liquidity_sweep"] = 1.0 if sweep.get("swept") else 0.0
                sweep_dir = (sweep.get("direction") or "").upper()
                f["liquidity_sweep_bullish"] = 1.0 if sweep_dir in ("BUY", "BULLISH") else 0.0
                f["liquidity_sweep_bearish"] = 1.0 if sweep_dir in ("SELL", "BEARISH") else 0.0
            else:
                f["liquidity_sweep"] = 0.0
                f["liquidity_sweep_bullish"] = 0.0
                f["liquidity_sweep_bearish"] = 0.0
            return f

        # AUDIT FIX: the live smc_ctx dict only exists when AnalysisAgent
        # runs in real time. During historical/offline training,
        # add_features() calls build_feature_vector(analysis_out={}) for
        # every bar, so bos_detected/order_block_tap/fvg_detected/etc were
        # ALL constant 0.0 for the entire dataset — dropped as
        # zero-variance and invisible to the model. Fall back to a
        # lightweight price-action derivation of the same concepts
        # straight from OHLC, using only data available at/ before the
        # current bar (no look-ahead), so these remain real, varying
        # signals during training even without a live analysis_out.
        return self._smc_from_price_action(df) if df is not None else self._smc_zero()

    def _smc_zero(self) -> Dict[str, float]:
        f: Dict[str, float] = {}
        for k in ("smc_buy", "smc_sell", "smc_neutral", "smc_confluence_score",
                  "bos_detected", "choch_detected", "order_block_tap", "fvg_detected",
                  "liquidity_sweep", "liquidity_sweep_bullish", "liquidity_sweep_bearish"):
            f[k] = 0.0
        f["smc_neutral"] = 1.0
        return f

    def _smc_from_price_action(self, df: pd.DataFrame, swing_lookback: int = 20) -> Dict[str, float]:
        """Lightweight, look-ahead-safe Smart Money Concepts approximation
        computed directly from OHLC — used as the training-time fallback
        for _smc_liquidity_features() when no live analysis_out is
        available.

        - BOS (Break of Structure): close breaks beyond the highest high /
          lowest low of the prior `swing_lookback` candles (excluding the
          current one) -> bullish/bearish structure break.
        - FVG (Fair Value Gap): classic 3-candle imbalance — candle[-3]'s
          high/low doesn't overlap candle[-1]'s low/high, leaving a price
          gap the market tends to revisit.
        - Order block: the last opposite-colored candle before an
          impulsive move (a same-direction candle with range >= 1.5x the
          recent average true range).
        All lookups use only bars up to and including the current index —
        no future data.
        """
        f = self._smc_zero()
        if df is None or len(df) < max(swing_lookback + 2, 5):
            return f
        if not all(c in df.columns for c in ("open", "high", "low", "close")):
            return f

        window = df.tail(swing_lookback + 1)
        prior = window.iloc[:-1]  # excludes current bar -> no look-ahead
        cur = window.iloc[-1]

        prior_high = _safe_float(prior["high"].max())
        prior_low = _safe_float(prior["low"].min())
        close = _safe_float(cur["close"])

        bullish_bos = close > prior_high > 0
        bearish_bos = close < prior_low and prior_low > 0
        f["bos_detected"] = 1.0 if (bullish_bos or bearish_bos) else 0.0
        f["smc_buy"] = 1.0 if bullish_bos else 0.0
        f["smc_sell"] = 1.0 if bearish_bos else 0.0
        f["smc_neutral"] = 0.0 if (bullish_bos or bearish_bos) else 1.0

        # Fair Value Gap: compare candle 3 bars back to the current candle
        if len(df) >= 3:
            c0, c2 = df.iloc[-3], df.iloc[-1]
            bull_gap = _safe_float(c2["low"]) > _safe_float(c0["high"]) > 0
            bear_gap = _safe_float(c2["high"]) < _safe_float(c0["low"]) and _safe_float(c0["low"]) > 0
            f["fvg_detected"] = 1.0 if (bull_gap or bear_gap) else 0.0

        # Order block: last opposite-colored candle before an impulsive
        # (range >= 1.5x recent average range) same-direction move.
        if len(df) >= swing_lookback:
            ranges = (df["high"] - df["low"]).tail(swing_lookback)
            avg_range = _safe_float(ranges.mean())
            cur_range = _safe_float(cur["high"] - cur["low"])
            impulsive = avg_range > 0 and cur_range >= 1.5 * avg_range
            if impulsive and len(df) >= 2:
                cur_bullish = _safe_float(cur["close"]) > _safe_float(cur["open"])
                prev = df.iloc[-2]
                prev_bearish = _safe_float(prev["close"]) < _safe_float(prev["open"])
                prev_bullish = _safe_float(prev["close"]) > _safe_float(prev["open"])
                f["order_block_tap"] = 1.0 if ((cur_bullish and prev_bearish) or
                                                (not cur_bullish and prev_bullish)) else 0.0

        # Liquidity sweep: current bar wicks beyond the prior swing extreme
        # but closes back inside it (stop-hunt pattern).
        cur_high = _safe_float(cur["high"])
        cur_low = _safe_float(cur["low"])
        sweep_high = cur_high > prior_high > 0 and close < prior_high
        sweep_low = cur_low < prior_low and prior_low > 0 and close > prior_low
        f["liquidity_sweep"] = 1.0 if (sweep_high or sweep_low) else 0.0
        f["liquidity_sweep_bearish"] = 1.0 if sweep_high else 0.0
        f["liquidity_sweep_bullish"] = 1.0 if sweep_low else 0.0

        f["smc_confluence_score"] = float(
            f["bos_detected"] + f["fvg_detected"] + f["order_block_tap"] + f["liquidity_sweep"]
        ) / 4.0

        return f

    # ── 7. Confluence + sentiment (Day 66/67) ─────────────────────

    def _confluence_features(self, a: Dict) -> Dict[str, float]:
        f: Dict[str, float] = {}
        # Sentiment
        sent = a.get("sentiment_ctx") or {}
        if isinstance(sent, dict):
            f["sentiment_final_score"] = _safe_float(sent.get("final_score"))
            sent_bias = (sent.get("bias") or "NEUTRAL").upper()
            f["sentiment_bullish"] = 1.0 if "BULL" in sent_bias else 0.0
            f["sentiment_bearish"] = 1.0 if "BEAR" in sent_bias else 0.0
        else:
            f["sentiment_final_score"] = 0.0
            f["sentiment_bullish"] = 0.0
            f["sentiment_bearish"] = 0.0

        # Confluence (Day 67)
        conf = a.get("confluence") or {}
        if isinstance(conf, dict):
            f["confluence_buy_score"] = _safe_float(conf.get("buy_score"))
            f["confluence_sell_score"] = _safe_float(conf.get("sell_score"))
            f["confluence_net_score"] = _safe_float(conf.get("net_score"))
            f["confluence_aligned_factors"] = _safe_float(conf.get("aligned_factors"))
            f["confluence_total_factors"] = _safe_float(conf.get("total_factors"))
            f["confluence_confidence"] = _safe_float(conf.get("confidence"))
            quality = (conf.get("setup_quality") or "AVOID").upper()
            f["quality_a_plus"] = 1.0 if quality == "A+" else 0.0
            f["quality_a"] = 1.0 if quality == "A" else 0.0
            f["quality_b"] = 1.0 if quality == "B" else 0.0
            f["quality_avoid"] = 1.0 if quality == "AVOID" else 0.0
        else:
            for k in ("confluence_buy_score", "confluence_sell_score", "confluence_net_score",
                      "confluence_aligned_factors", "confluence_total_factors", "confluence_confidence",
                      "quality_a_plus", "quality_a", "quality_b", "quality_avoid"):
                f[k] = 0.0

        # Master analyst signal
        master = a.get("master_ctx") or {}
        if isinstance(master, dict):
            ma_signal = (master.get("master_signal") or "WAIT").upper()
            f["master_buy"] = 1.0 if ma_signal == "BUY" else 0.0
            f["master_sell"] = 1.0 if ma_signal == "SELL" else 0.0
            f["master_wait"] = 1.0 if ma_signal == "WAIT" else 0.0
            f["master_confidence"] = _safe_float(master.get("master_confidence"))
        else:
            f["master_buy"] = 0.0
            f["master_sell"] = 0.0
            f["master_wait"] = 1.0
            f["master_confidence"] = 0.0

        # LLM signal
        llm = a.get("llm") or {}
        if isinstance(llm, dict):
            llm_sig = (llm.get("signal") or "WAIT").upper()
            f["llm_buy"] = 1.0 if llm_sig == "BUY" else 0.0
            f["llm_sell"] = 1.0 if llm_sig == "SELL" else 0.0
            f["llm_confidence"] = _safe_float(llm.get("confidence"))
        else:
            f["llm_buy"] = 0.0
            f["llm_sell"] = 0.0
            f["llm_confidence"] = 0.0

        # Rule signal
        sig = a.get("signal") or {}
        if isinstance(sig, dict):
            rule_sig = (sig.get("signal") or "WAIT").upper()
            f["rule_buy"] = 1.0 if rule_sig == "BUY" else 0.0
            f["rule_sell"] = 1.0 if rule_sig == "SELL" else 0.0
            f["rule_confidence"] = _safe_float(sig.get("confidence"))
        else:
            f["rule_buy"] = 0.0
            f["rule_sell"] = 0.0
            f["rule_confidence"] = 0.0

        return f


# ── Singleton ───────────────────────────────────────────────────────

_ENGINE: Optional[FeatureEngineer] = None


def get_feature_engineer() -> FeatureEngineer:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = FeatureEngineer()
    return _ENGINE
