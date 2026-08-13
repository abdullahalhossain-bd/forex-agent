#!/usr/bin/env python3
"""Debug script: print the SignalEngine output on a specific bar."""
import os, sys
sys.path.insert(0, "/home/z/my-project/repos/forex-agent")
os.environ["BACKTEST_MODE"] = "1"

import pandas as pd
from data.indicators import Indicators
from strategy.signal_engine import SignalEngine

# Load EURUSD H1
df = pd.read_csv("/home/z/my-project/repos/forex-agent/data/EURUSD_H1.csv", encoding="utf-8-sig")
df = df.rename(columns={"datetime_utc": "time"})
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.set_index("time").sort_index()
for c in ["open","high","low","close","tick_volume"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["volume"] = df["tick_volume"]

# Indicators
ind = Indicators()
df = ind.add_moving_averages(df)
df = ind.add_rsi(df)
df = ind.add_macd(df)
df = ind.add_bollinger_bands(df)
df = ind.add_atr(df)
df = ind.add_adx(df)
df = ind.add_stochastic(df)
df = ind.add_trend_signals(df)
df["rolling_resistance_20"] = df["high"].shift(1).rolling(20).max()
df["rolling_support_20"] = df["low"].shift(1).rolling(20).min()
df["volume_ratio"] = df["tick_volume"] / df["tick_volume"].shift(1).rolling(20).mean()

print(f"Loaded {len(df)} bars")
print(f"Sample row at index 250:")
row = df.iloc[250]
print(f"  close={row['close']:.5f} ema_50={row.get('ema_50',0):.5f} ema_200={row.get('ema_200',0):.5f}")
print(f"  adx={row.get('adx',0):.2f} rsi={row.get('rsi',50):.2f} macd_hist={row.get('macd_hist',0):.5f}")
print(f"  trend={row.get('trend','')}")

# Build context
ind_ctx = {
    "price": float(row["close"]),
    "open": float(row["open"]),
    "high": float(row["high"]),
    "low": float(row["low"]),
    "close": float(row["close"]),
    "sma_20": float(row.get("sma_20", 0) or 0),
    "sma_50": float(row.get("sma_50", 0) or 0),
    "sma_200": float(row.get("sma_200", 0) or 0),
    "ema_9": float(row.get("ema_9", 0) or 0),
    "ema_21": float(row.get("ema_21", 0) or 0),
    "ema_50": float(row.get("ema_50", 0) or 0),
    "ema_200": float(row.get("ema_200", 0) or 0),
    "rsi": float(row.get("rsi", 50) or 50),
    "rsi_signal": row.get("rsi_signal", "neutral"),
    "macd": float(row.get("macd", 0) or 0),
    "macd_signal": float(row.get("macd_signal", 0) or 0),
    "macd_hist": float(row.get("macd_hist", 0) or 0),
    "macd_cross": row.get("macd_cross", ""),
    "adx": float(row.get("adx", 0) or 0),
    "stoch_k": float(row.get("stoch_k", 50) or 50),
    "stoch_d": float(row.get("stoch_d", 50) or 50),
    "bb_upper": float(row.get("bb_upper", 0) or 0),
    "bb_middle": float(row.get("bb_middle", 0) or 0),
    "bb_lower": float(row.get("bb_lower", 0) or 0),
    "bb_width": float(row.get("bb_width", 0) or 0),
    "atr": float(row.get("atr", 0) or 0),
    "trend": row.get("trend", "sideways"),
    "tick_volume": float(row.get("tick_volume", 0) or 0),
    "volume_ratio": float(row.get("volume_ratio", 1.0) or 1.0),
}

atr = ind_ctx["atr"]
sr_ctx = {
    "support": float(row.get("rolling_support_20", ind_ctx["close"]) or ind_ctx["close"]),
    "resistance": float(row.get("rolling_resistance_20", ind_ctx["close"]) or ind_ctx["close"]),
    "location": "near_support" if abs(ind_ctx["close"] - float(row.get("rolling_support_20", ind_ctx["close"]) or ind_ctx["close"])) <= atr else "mid_range",
    "nearest_support": float(row.get("rolling_support_20", ind_ctx["close"]) or ind_ctx["close"]),
    "nearest_resistance": float(row.get("rolling_resistance_20", ind_ctx["close"]) or ind_ctx["close"]),
}

pat_ctx = {"pattern": "none", "engulfing": "none", "star_pattern": "none",
           "trend": ind_ctx["trend"], "has_pattern": False,
           "pattern_direction": "NEUTRAL", "pattern_confidence": 0}

regime = {
    "regime": "TRENDING" if ind_ctx["adx"] >= 22 else "RANGING",
    "market_direction": "BULLISH" if "bull" in ind_ctx["trend"] else ("BEARISH" if "bear" in ind_ctx["trend"] else "NEUTRAL"),
    "strategy_type": "TREND_FOLLOW" if ind_ctx["adx"] >= 25 else ("RANGE_TRADE" if ind_ctx["adx"] < 18 else "WAIT"),
    "strength": "STRONG" if ind_ctx["adx"] >= 30 else ("MODERATE" if ind_ctx["adx"] >= 22 else "WEAK"),
}

engine = SignalEngine()
result = engine.generate(
    ind_ctx=ind_ctx, pat_ctx=pat_ctx, sr_ctx=sr_ctx,
    regime=regime,
    mtf_bias={"bias": "NEUTRAL", "confidence": "LOW"},
    advanced_pat_ctx=None, fib_ctx=None, extended_ctx=None,
)
print("\nSignalEngine output:")
for k, v in result.items():
    if k != "signals":
        print(f"  {k}: {v}")
print(f"  signals ({len(result.get('signals',[]))}):")
for s in result.get("signals", [])[:10]:
    print(f"    {s}")

# Now scan many bars
print("\nScanning bars 250-500 for BUY/SELL signals:")
engine = SignalEngine()
buy_count = sell_count = wait_count = 0
sample_signals = []
for i in range(250, min(500, len(df))):
    row = df.iloc[i]
    atr = float(row.get("atr", 0) or 0)
    if atr <= 0:
        continue
    ind_ctx = {
        "price": float(row["close"]), "open": float(row["open"]),
        "high": float(row["high"]), "low": float(row["low"]),
        "close": float(row["close"]),
        "sma_20": float(row.get("sma_20",0) or 0),
        "sma_50": float(row.get("sma_50",0) or 0),
        "sma_200": float(row.get("sma_200",0) or 0),
        "ema_9": float(row.get("ema_9",0) or 0),
        "ema_21": float(row.get("ema_21",0) or 0),
        "ema_50": float(row.get("ema_50",0) or 0),
        "ema_200": float(row.get("ema_200",0) or 0),
        "rsi": float(row.get("rsi",50) or 50),
        "rsi_signal": row.get("rsi_signal","neutral"),
        "macd": float(row.get("macd",0) or 0),
        "macd_signal": float(row.get("macd_signal",0) or 0),
        "macd_hist": float(row.get("macd_hist",0) or 0),
        "macd_cross": row.get("macd_cross",""),
        "adx": float(row.get("adx",0) or 0),
        "stoch_k": float(row.get("stoch_k",50) or 50),
        "stoch_d": float(row.get("stoch_d",50) or 50),
        "bb_upper": float(row.get("bb_upper",0) or 0),
        "bb_middle": float(row.get("bb_middle",0) or 0),
        "bb_lower": float(row.get("bb_lower",0) or 0),
        "bb_width": float(row.get("bb_width",0) or 0),
        "atr": atr,
        "trend": row.get("trend","sideways"),
        "tick_volume": float(row.get("tick_volume",0) or 0),
        "volume_ratio": float(row.get("volume_ratio",1.0) or 1.0),
    }
    sr_ctx = {
        "support": float(row.get("rolling_support_20",ind_ctx["close"]) or ind_ctx["close"]),
        "resistance": float(row.get("rolling_resistance_20",ind_ctx["close"]) or ind_ctx["close"]),
        "location": "mid_range",
        "nearest_support": float(row.get("rolling_support_20",ind_ctx["close"]) or ind_ctx["close"]),
        "nearest_resistance": float(row.get("rolling_resistance_20",ind_ctx["close"]) or ind_ctx["close"]),
    }
    pat_ctx = {"pattern":"none","engulfing":"none","star_pattern":"none",
               "trend":ind_ctx["trend"],"has_pattern":False,
               "pattern_direction":"NEUTRAL","pattern_confidence":0}
    regime = {
        "regime": "TRENDING" if ind_ctx["adx"] >= 22 else "RANGING",
        "market_direction": "BULLISH" if "bull" in ind_ctx["trend"] else ("BEARISH" if "bear" in ind_ctx["trend"] else "NEUTRAL"),
        "strategy_type": "TREND_FOLLOW" if ind_ctx["adx"] >= 25 else ("RANGE_TRADE" if ind_ctx["adx"] < 18 else "WAIT"),
        "strength": "STRONG" if ind_ctx["adx"] >= 30 else ("MODERATE" if ind_ctx["adx"] >= 22 else "WEAK"),
    }
    try:
        result = engine.generate(ind_ctx=ind_ctx, pat_ctx=pat_ctx, sr_ctx=sr_ctx,
                                   regime=regime, mtf_bias={"bias":"NEUTRAL","confidence":"LOW"},
                                   advanced_pat_ctx=None, fib_ctx=None, extended_ctx=None)
    except Exception as e:
        continue
    sig = result.get("signal","WAIT")
    if "BUY" in sig:
        buy_count += 1
        if len(sample_signals) < 5:
            sample_signals.append((i, sig, result.get("confidence",0),
                                    result.get("bull_score",0), result.get("bear_score",0),
                                    result.get("bull_factors",0), result.get("bear_factors",0),
                                    ind_ctx["adx"]))
    elif "SELL" in sig:
        sell_count += 1
    else:
        wait_count += 1

print(f"  BUY/STRONG_BUY: {buy_count}")
print(f"  SELL/STRONG_SELL: {sell_count}")
print(f"  WAIT: {wait_count}")
print(f"\nSample BUY signals:")
for s in sample_signals:
    print(f"  bar {s[0]}: {s[1]} conf={s[2]} bull={s[3]}/{s[5]}f bear={s[4]}/{s[6]}f adx={s[7]:.1f}")
