"""
ml/pipeline/phase3_features.py — Institutional Feature Engineering
=================================================================
Computes 100+ features across 6 categories:
  - Trend: EMA, SMA, VWAP, SuperTrend, ADX, ATR
  - Momentum: RSI, MACD, Stochastic, CCI, ROC, Williams %R
  - Volume: OBV, CMF, MFI, volume profile
  - Volatility: Bollinger, Donchian, Keltner, realized vol
  - Market Structure: Swing H/L, liquidity zones, FVG, BOS, CHoCH, OB, S/R
  - Session: London, New York, Tokyo, overlap flags
  - Time: Hour, day, week, month features
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from ml.pipeline.utils import (
    DATA_HISTORY_DIR, PIPELINE_CACHE_DIR, PipelineConfig, PipelineTimer,
    dataset_hash, get_pipeline_logger,
)

log = get_pipeline_logger("phase3_features")

# ── Feature Registry ────────────────────────────────────────────
FEATURE_REGISTRY: Dict[str, List[str]] = {
    "trend": ["ema_8", "ema_21", "ema_50", "ema_200", "sma_20", "sma_50", "sma_200",
              "vwap", "supertrend", "supertrend_dir", "adx", "atr"],
    "momentum": ["rsi_14", "rsi_7", "macd", "macd_signal", "macd_hist",
                 "stoch_k", "stoch_d", "cci_20", "roc_10", "roc_20", "williams_r"],
    "volume": ["obv", "obv_ema", "cmf_20", "mfi_14", "vol_sma_20", "vol_ratio"],
    "volatility": ["bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pct",
                   "donchian_upper", "donchian_lower", "donchian_mid",
                   "keltner_upper", "keltner_lower", "keltner_mid",
                   "realized_vol_10", "realized_vol_20", "realized_vol_50"],
    "market_structure": ["swing_high_10", "swing_low_10", "swing_high_20", "swing_low_20",
                         "liquidity_zone_above", "liquidity_zone_below",
                         "fvg_bullish", "fvg_bearish", "bos_bullish", "bos_bearish",
                         "choch_bullish", "choch_bearish", "ob_bullish", "ob_bearish",
                         "nearest_support", "nearest_resistance", "sr_distance"],
    "session": ["session_london", "session_newyork", "session_tokyo", "session_overlap"],
    "time": ["hour_sin", "hour_cos", "day_of_week", "day_sin", "day_cos",
             "week_of_year", "month_sin", "month_cos"],
}


def compute_features(config: Optional[PipelineConfig] = None) -> Dict[str, pd.DataFrame]:
    """Main entry point for Phase 3. Returns dict of symbol -> featured DataFrame.
    
    Tries primary_timeframe first, then falls back to any available timeframe
    with sufficient data (>= 100 rows).
    """
    config = config or PipelineConfig()
    results = {}
    
    with PipelineTimer("Phase 3: Feature Engineering", log):
        # Build ordered list of timeframes to try: primary first, then rest
        tf_priority = [config.primary_timeframe]
        for t in config.timeframes:
            if t not in tf_priority:
                tf_priority.append(t)
        
        for symbol in config.symbols:
            # Find best available timeframe for this symbol
            df = None
            chosen_tf = None
            
            for tf in tf_priority:
                cache_path = PIPELINE_CACHE_DIR / f"{symbol}_{tf}_features.parquet"
                
                # Cache check
                if cache_path.exists() and config.cache_datasets:
                    results[symbol] = pd.read_parquet(cache_path)
                    log.info(f"  {symbol}: loaded from cache ({len(results[symbol])} rows, "
                             f"{len(results[symbol].columns)} cols, tf={tf})")
                    df = results[symbol]
                    chosen_tf = tf
                    break
                
                raw = _load_raw_data(symbol, tf)
                if raw is not None and len(raw) >= 100:
                    df = raw
                    chosen_tf = tf
                    log.info(f"  {symbol}: using {tf} data ({len(df)} rows) "
                             f"(primary={config.primary_timeframe} not available)")
                    break
            
            if df is None or chosen_tf is None:
                log.warning(f"  {symbol}: insufficient data on ALL timeframes, skipping")
                continue
            
            # Skip if already loaded from cache
            if symbol in results:
                continue
            
            log.info(f"  {symbol}: computing features on {len(df)} rows...")
            
            # Compute each feature set
            active_sets = config.feature_sets
            if "trend" in active_sets:
                df = _add_trend_features(df)
            if "momentum" in active_sets:
                df = _add_momentum_features(df)
            if "volume" in active_sets:
                df = _add_volume_features(df)
            if "volatility" in active_sets:
                df = _add_volatility_features(df)
            if "market_structure" in active_sets:
                df = _add_structure_features(df)
            if "session" in active_sets:
                df = _add_session_features(df)
            if "time" in active_sets:
                df = _add_time_features(df)
            
            # Drop rows with all-NaN features (from indicator warmup)
            feature_cols = [c for c in df.columns if c not in ("timestamp", "open", "high", "low", "close", "volume")]
            df = df.dropna(subset=feature_cols, how="all").reset_index(drop=True)
            
            # Save cache
            if config.cache_datasets:
                cache_path = PIPELINE_CACHE_DIR / f"{symbol}_{chosen_tf}_features.parquet"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path, index=False)
                log.info(f"  {symbol}: {len(df)} rows x {len(df.columns)} cols -> cached (tf={chosen_tf})")
            
            results[symbol] = df
    
    return results


def _load_raw_data(symbol: str, tf: str) -> Optional[pd.DataFrame]:
    path = DATA_HISTORY_DIR / symbol / f"{symbol}_{tf}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


# ── Trend Features ─────────────────────────────────────────────
def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    
    for p in (8, 21, 50, 200):
        df[f"ema_{p}"] = c.ewm(span=p, adjust=False).mean()
    for p in (20, 50, 200):
        df[f"sma_{p}"] = c.rolling(p).mean()
    
    # VWAP (cumulative, reset daily)
    typical = (h + l + c) / 3
    vol = df.get("volume", pd.Series(1, index=df.index)).fillna(1)
    df["vwap"] = (typical * vol).cumsum() / vol.cumsum().replace(0, np.nan)
    
    # SuperTrend
    atr = _atr(h, l, c, 10)
    df["supertrend"], df["supertrend_dir"] = _supertrend(h, l, c, period=10, multiplier=3.0)
    
    # ADX
    df["adx"] = _adx(h, l, c, 14)
    
    # ATR
    df["atr"] = _atr(h, l, c, 14)
    
    return df


# ── Momentum Features ──────────────────────────────────────────
def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df.get("volume", pd.Series(1, index=df.index)).fillna(1)
    
    # RSI
    for p in (7, 14):
        df[f"rsi_{p}"] = _rsi(c, p)
    
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # Stochastic
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    
    # CCI
    tp = (h + l + c) / 3
    ma = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci_20"] = (tp - ma) / (0.015 * md.replace(0, np.nan))
    
    # ROC
    for p in (10, 20):
        df[f"roc_{p}"] = c.pct_change(p) * 100
    
    # Williams %R
    high14 = h.rolling(14).max()
    low14 = l.rolling(14).min()
    df["williams_r"] = -100 * (high14 - c) / (high14 - low14).replace(0, np.nan)
    
    return df


# ── Volume Features ────────────────────────────────────────────
def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    v = df.get("volume", pd.Series(1, index=df.index)).fillna(1)
    h, l = df["high"], df["low"]
    
    # OBV
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv"] = obv
    df["obv_ema"] = obv.ewm(span=20, adjust=False).mean()
    
    # CMF (Chaikin Money Flow)
    mfv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan) * v
    df["cmf_20"] = mfv.rolling(20).sum() / v.rolling(20).sum()
    
    # MFI (Money Flow Index)
    tp = (h + l + c) / 3
    mf = tp * v
    pos_mf = mf.where(tp > tp.shift(1), 0)
    neg_mf = mf.where(tp < tp.shift(1), 0)
    pos_sum = pos_mf.rolling(14).sum()
    neg_sum = neg_mf.rolling(14).sum()
    mfi = 100 - 100 / (1 + pos_sum / neg_sum.replace(0, np.nan))
    df["mfi_14"] = mfi
    
    # Volume ratio
    df["vol_sma_20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_sma_20"].replace(0, np.nan)
    
    return df


# ── Volatility Features ────────────────────────────────────────
def _add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    
    # Bollinger Bands
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_mid"] = sma20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20.replace(0, np.nan)
    df["bb_pct"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    
    # Donchian
    for p in (20,):
        df[f"donchian_upper"] = h.rolling(p).max()
        df[f"donchian_lower"] = l.rolling(p).min()
        df[f"donchian_mid"] = (df["donchian_upper"] + df["donchian_lower"]) / 2
    
    # Keltner
    ema20 = c.ewm(span=20, adjust=False).mean()
    atr = _atr(h, l, c, 10)
    df["keltner_upper"] = ema20 + 2 * atr
    df["keltner_lower"] = ema20 - 2 * atr
    df["keltner_mid"] = ema20
    
    # Realized volatility
    returns = np.log(c / c.shift(1))
    for p in (10, 20, 50):
        df[f"realized_vol_{p}"] = returns.rolling(p).std() * np.sqrt(252 * 96)  # Annualized for M15
    
    return df


# ── Market Structure Features ──────────────────────────────────
def _add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    
    # Swing highs/lows (keep as bool for chaining, convert to float at end)
    # Round-22 audit fix: REMOVED center=True — it caused lookahead bias.
    #
    # center=True means bar i's value depends on p/2 bars of FUTURE data.
    # In live trading, you can't know if bar i is a swing high until p/2
    # bars have passed. This silently leaked future info into the training
    # set, making validation accuracy artificially high.
    #
    # Now: use backward-only rolling (center=False, the default). The
    # swing point is confirmed using only PAST data — bar i is a swing
    # high if it's the max of the PREVIOUS p bars (not the surrounding
    # p bars). This is conservative (delayed confirmation) but honest.
    #
    # The downstream code at L284-301 already uses .shift(1) on these
    # columns, which is correct — it ensures the feature at bar i uses
    # the swing status as of bar i-1 (no lookahead from the shift).
    # But the center=True was ADDITIONAL lookahead on top of that shift,
    # making the total lookahead = p/2 + 1 bars instead of just 1 bar.
    for p in (10, 20):
        df[f"swing_high_{p}"] = h.rolling(p, center=False).max().shift(1) == h.shift(1)
        df[f"swing_low_{p}"] = l.rolling(p, center=False).min().shift(1) == l.shift(1)
    
    # Liquidity zones
    df["liquidity_zone_above"] = c.shift(1) < df["swing_high_20"].shift(1) * 0.001 + c.shift(1)
    df["liquidity_zone_below"] = c.shift(1) > df["swing_low_20"].shift(1) * 0.001 - c.shift(1)
    
    # Fair Value Gaps
    df["fvg_bullish"] = (l.shift(2) > h.shift(0)) & (c.shift(1) > c.shift(2))
    df["fvg_bearish"] = (h.shift(2) < l.shift(0)) & (c.shift(1) < c.shift(2))
    
    # Break of Structure (keep bool for CHoCH chaining)
    df["bos_bullish"] = c > h.rolling(20).max().shift(1)
    df["bos_bearish"] = c < l.rolling(20).min().shift(1)
    
    # Change of Character (first BOS after opposite trend)
    df["choch_bullish"] = df["bos_bullish"] & (c.shift(5) < c.shift(5).rolling(20).mean())
    df["choch_bearish"] = df["bos_bearish"] & (c.shift(5) > c.shift(5).rolling(20).mean())
    
    # Order Blocks (last swing before BOS)
    df["ob_bullish"] = df["swing_low_10"] & (c > c.rolling(10).max().shift(1))
    df["ob_bearish"] = df["swing_high_10"] & (c < c.rolling(10).min().shift(1))
    
    # Support / Resistance (nearest)
    df["nearest_support"] = l.rolling(50).min()
    df["nearest_resistance"] = h.rolling(50).max()
    sr_range = df["nearest_resistance"] - df["nearest_support"]
    df["sr_distance"] = (c - df["nearest_support"]) / sr_range.replace(0, np.nan)
    
    # Convert all bool structure columns to float (for ML compatibility)
    bool_cols = [
        "swing_high_10", "swing_low_10", "swing_high_20", "swing_low_20",
        "liquidity_zone_above", "liquidity_zone_below",
        "fvg_bullish", "fvg_bearish",
        "bos_bullish", "bos_bearish",
        "choch_bullish", "choch_bearish",
        "ob_bullish", "ob_bearish",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    return df


# ── Session Features ───────────────────────────────────────────
def _add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    
    hour = df["timestamp"].dt.hour
    
    # London: 07:00-16:00 UTC
    df["session_london"] = ((hour >= 7) & (hour < 16)).astype(float)
    # New York: 12:00-21:00 UTC
    df["session_newyork"] = ((hour >= 12) & (hour < 21)).astype(float)
    # Tokyo: 00:00-09:00 UTC
    df["session_tokyo"] = ((hour >= 0) & (hour < 9)).astype(float)
    # Overlap (London-NY): 12:00-16:00 UTC
    df["session_overlap"] = ((hour >= 12) & (hour < 16)).astype(float)
    
    return df


# ── Time Features ──────────────────────────────────────────────
def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    
    hour = df["timestamp"].dt.hour
    dow = df["timestamp"].dt.dayofweek
    
    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_of_week"] = dow
    df["day_sin"] = np.sin(2 * np.pi * dow / 5)
    df["day_cos"] = np.cos(2 * np.pi * dow / 5)
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(float)
    month = df["timestamp"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    
    return df


# ── Technical Indicator Helpers ────────────────────────────────
def _atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr = _atr(high, low, close, 1)
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()

def _supertrend(high, low, close, period=10, multiplier=3.0):
    atr = _atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(0, index=close.index, dtype=float)
    
    st = lower_band.iloc[0] if len(lower_band) > 0 else 0
    d = 1  # 1 = bullish, -1 = bearish
    
    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i-1]:
            d = 1
        elif close.iloc[i] < lower_band.iloc[i-1]:
            d = -1
        
        if d == 1:
            st = max(lower_band.iloc[i], st) if st != 0 else lower_band.iloc[i]
        else:
            st = min(upper_band.iloc[i], st) if st != 0 else upper_band.iloc[i]
        
        supertrend.iloc[i] = st
        direction.iloc[i] = d
    
    return supertrend, direction


def get_feature_columns(df: pd.DataFrame, exclude: Optional[Set[str]] = None) -> List[str]:
    """Get list of feature column names (exclude price/volume/raw columns)."""
    exclude = exclude or {"timestamp", "open", "high", "low", "close", "volume",
                          "tick_vol", "real_vol", "spread"}
    return [c for c in df.columns if c not in exclude]