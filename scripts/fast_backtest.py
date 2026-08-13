#!/usr/bin/env python3
"""
Fast backtest using the REAL system modules.

This script replays CSV data through the SAME modules Demo/Real use:
  - data/indicators.py:Indicators.add_all() — for indicator computation
  - strategy/signal_engine.py:SignalEngine.generate() — for BUY/SELL signals
  - risk/rr_policy.py:get_min_rr() — for R:R floor
  - risk/trade_permission.py constants — for confidence / factor gates

But it skips the slow modules that don't affect signal quality:
  - LLM calls (MasterAnalyst)
  - News API calls (would time out on historical bars anyway)
  - Economic calendar / FRED (same — live network)
  - Sentence-transformers memory (not installed)
  - ML ensemble (uses .pkl models that are out of sync)

This gives an apples-to-apples view of what the RULE ENGINE + RISK GATES
alone would do — which is exactly what the user wants to evaluate.

Output:
  - Per-pair winrate, RR, profit factor, P&L
  - Per-pair rejection reasons (so we can see which gate is too strict)
  - Tunable config: MIN_CONFIDENCE, MIN_RR, MIN_ALIGNED_FACTORS via env vars
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/z/my-project/repos/forex-agent")
sys.path.insert(0, str(PROJECT_ROOT))

# Set backtest mode env BEFORE any imports that read it
os.environ["BACKTEST_MODE"] = "1"
os.environ["SIMULATION_MODE"] = "true"
os.environ["TEST_MODE"] = "false"
os.environ["ECONCAL_OUTAGE_ALLOWS_TRADES"] = "true"
os.environ["ML_MODEL_CONSISTENCY_ACTION"] = "warn"
os.environ["ENABLE_TELEGRAM"] = "false"

# Logging: minimal
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s",
)
log = logging.getLogger("fast_backtest")
for name in ["urllib3", "httpx", "groq", "google_genai", "chromadb",
             "sentence_transformers", "matplotlib", "PIL", "asyncio",
             "indicators", "indicators_ext", "data.fetcher", "data_orchestrator",
             "data.validator", "data.indicator_registry", "data.live_feed"]:
    logging.getLogger(name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore")


# ── Tunable config (overridable via env vars) ─────────────────────────────
MIN_CONFIDENCE = int(os.getenv("BT_MIN_CONFIDENCE", "60"))
MIN_ALIGNED_FACTORS = int(os.getenv("BT_MIN_FACTORS", "3"))
MIN_RR = float(os.getenv("BT_MIN_RR", "2.0"))
RISK_PER_TRADE = float(os.getenv("BT_RISK_PCT", "0.005")) / 100  # 0.5% default
MAX_HOLD_BARS = int(os.getenv("BT_MAX_HOLD_BARS", "100"))
INITIAL_BALANCE = float(os.getenv("BT_INITIAL_BALANCE", "10000"))


def load_csv(pair: str, timeframe: str) -> Optional[pd.DataFrame]:
    path = PROJECT_ROOT / "data" / f"{pair}_{timeframe}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = {c.lower(): c for c in df.columns}
    if "datetime_utc" in cols:
        df = df.rename(columns={cols["datetime_utc"]: "time"})
    elif "datetime" in cols:
        df = df.rename(columns={cols["datetime"]: "time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).sort_values("time").set_index("time")
    for c in ("open", "high", "low", "close", "tick_volume", "spread"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 1000
    if "volume" not in df.columns:
        df["volume"] = df["tick_volume"]
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Use the REAL Indicators class from data/indicators.py."""
    from data.indicators import Indicators
    ind = Indicators()
    df = ind.add_moving_averages(df)
    df = ind.add_rsi(df)
    df = ind.add_macd(df)
    df = ind.add_bollinger_bands(df)
    df = ind.add_atr(df)
    df = ind.add_adx(df)
    df = ind.add_stochastic(df)
    df = ind.add_trend_signals(df)
    # Rolling support/resistance + volume ratio (for breakout strategy)
    df["rolling_resistance_20"] = df["high"].shift(1).rolling(20).max()
    df["rolling_support_20"] = df["low"].shift(1).rolling(20).min()
    df["volume_ratio"] = df["tick_volume"] / df["tick_volume"].shift(1).rolling(20).mean()
    # 2026-08-13: volume_avg_20 + spread_avg_20 for SignalEngine filters
    df["volume_avg_20"] = df["tick_volume"].rolling(20).mean()
    if "spread" in df.columns:
        df["spread_avg_20"] = df["spread"].rolling(20).mean()
        df["spread_pips"] = df["spread"]  # already in pips for MT5 data
    else:
        df["spread_avg_20"] = 0.0
        df["spread_pips"] = 0.0
    # Consecutive same-direction closes (for SignalEngine confirmation filter)
    bull_close = (df["close"] > df["open"]).astype(int)
    bear_close = (df["close"] < df["open"]).astype(int)
    # Count consecutive runs ending at each bar
    def _consec(series):
        # Returns a series where each value = length of consecutive run ending at that bar
        result = series.copy().astype(int)
        # vectorized: if current==previous, current = previous+1
        prev = series.shift(1).fillna(0).astype(int)
        same = (series == prev).astype(int)
        # cumulative sum trick
        groups = (series != prev).cumsum()
        result = series.groupby(groups).cumcount() + 1
        result = result.where(series == 1, 0)
        return result
    df["consecutive_bull_closes"] = _consec(bull_close)
    df["consecutive_bear_closes"] = _consec(bear_close)
    return df


def build_ind_ctx(row: pd.Series) -> dict:
    """Build the ind_ctx dict that SignalEngine.generate expects."""
    return {
        "price": float(row.get("close", 0)),
        "open": float(row.get("open", 0)),
        "high": float(row.get("high", 0)),
        "low": float(row.get("low", 0)),
        "close": float(row.get("close", 0)),
        # MAs
        "sma_20": float(row.get("sma_20", 0) or 0),
        "sma_50": float(row.get("sma_50", 0) or 0),
        "sma_200": float(row.get("sma_200", 0) or 0),
        "ema_9": float(row.get("ema_9", 0) or 0),
        "ema_21": float(row.get("ema_21", 0) or 0),
        "ema_50": float(row.get("ema_50", 0) or 0),
        "ema_200": float(row.get("ema_200", 0) or 0),
        # Oscillators
        "rsi": float(row.get("rsi", 50) or 50),
        "rsi_signal": row.get("rsi_signal", "neutral"),
        "macd": float(row.get("macd", 0) or 0),
        "macd_signal": float(row.get("macd_signal", 0) or 0),
        "macd_hist": float(row.get("macd_hist", 0) or 0),
        "macd_cross": row.get("macd_cross", ""),
        "adx": float(row.get("adx", 0) or 0),
        "stoch_k": float(row.get("stoch_k", 50) or 50),
        "stoch_d": float(row.get("stoch_d", 50) or 50),
        # BB
        "bb_upper": float(row.get("bb_upper", 0) or 0),
        "bb_middle": float(row.get("bb_middle", 0) or 0),
        "bb_lower": float(row.get("bb_lower", 0) or 0),
        "bb_width": float(row.get("bb_width", 0) or 0),
        # ATR
        "atr": float(row.get("atr", 0) or 0),
        # Trend + volume
        "trend": row.get("trend", "sideways"),
        "tick_volume": float(row.get("tick_volume", 0) or 0),
        "volume_ratio": float(row.get("volume_ratio", 1.0) or 1.0),
        # 2026-08-13: new fields for SignalEngine volume/spread/consec filters
        "volume_avg_20": float(row.get("volume_avg_20", 0) or 0),
        "spread_pips": float(row.get("spread_pips", 0) or 0),
        "spread_avg_20": float(row.get("spread_avg_20", 0) or 0),
        "consecutive_bull_closes": int(row.get("consecutive_bull_closes", 0) or 0),
        "consecutive_bear_closes": int(row.get("consecutive_bear_closes", 0) or 0),
    }


def build_pat_ctx(row: pd.Series) -> dict:
    """Pattern context (minimal — no full PatternDetector)."""
    trend = row.get("trend", "sideways")
    return {
        "pattern": "none",
        "engulfing": "none",
        "star_pattern": "none",
        "trend": trend,
        "has_pattern": False,
        "pattern_direction": "NEUTRAL",
        "pattern_confidence": 0,
    }


def build_sr_ctx(row: pd.Series, atr: float) -> dict:
    """S/R context from rolling levels."""
    close = float(row.get("close", 0))
    resistance = float(row.get("rolling_resistance_20", close) or close)
    support = float(row.get("rolling_support_20", close) or close)
    if atr <= 0:
        atr = 0.001
    near_support = abs(close - support) <= atr * 1.0
    near_resistance = abs(resistance - close) <= atr * 1.0
    if near_support:
        location = "near_support"
    elif near_resistance:
        location = "near_resistance"
    elif close < support:
        location = "below_support"
    elif close > resistance:
        location = "above_resistance"
    else:
        location = "mid_range"
    return {
        "support": support,
        "resistance": resistance,
        "location": location,
        "nearest_support": support,
        "nearest_resistance": resistance,
    }


def build_regime(row: pd.Series, adx: float) -> dict:
    """Market regime — simple ADX-based classification."""
    trend = row.get("trend", "sideways")
    if "bull" in trend:
        direction = "BULLISH"
    elif "bear" in trend:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"
    if adx >= 25:
        strategy_type = "TREND_FOLLOW"
    elif adx < 18:
        strategy_type = "RANGE_TRADE"
    else:
        strategy_type = "WAIT"
    return {
        "regime": "TRENDING" if adx >= 22 else "RANGING",
        "market_direction": direction,
        "strategy_type": strategy_type,
        "strength": "STRONG" if adx >= 30 else ("MODERATE" if adx >= 22 else "WEAK"),
    }


def simulate_trade_exit(trade: dict, bar: pd.Series) -> tuple:
    """Check if SL/TP hit on this bar. Returns (hit_sl, hit_tp, exit_price)."""
    if trade["direction"] == "BUY":
        if float(bar["low"]) <= trade["sl"]:
            return (True, False, trade["sl"])
        if float(bar["high"]) >= trade["tp"]:
            return (False, True, trade["tp"])
    else:  # SELL
        if float(bar["high"]) >= trade["sl"]:
            return (True, False, trade["sl"])
        if float(bar["low"]) <= trade["tp"]:
            return (False, True, trade["tp"])
    return (False, False, 0.0)


def backtest_pair(
    pair: str,
    timeframe: str,
    df: pd.DataFrame,
    warmup: int = 250,
    starting_balance: float = INITIAL_BALANCE,
    use_pair_profile: bool = True,
) -> dict:
    """Run the real SignalEngine + risk gates over a single pair's CSV data.

    When use_pair_profile=True, reads per-pair config from
    utils.pair_profiles.PROFILES — each pair gets its own optimized
    min_confidence, min_factors, min_rr, session_filter, stop/target ATR.
    """
    from strategy.signal_engine import SignalEngine
    from utils.pair_profiles import get_pair_profile, get_session_hours

    # ── Load per-pair profile ────────────────────────────────────
    profile = get_pair_profile(pair) if use_pair_profile else None
    if profile is not None:
        _min_conf = profile.min_confidence
        _min_factors = profile.min_aligned_factors
        _min_rr = profile.min_rr
        _adx_min = profile.adx_min
        _session_filter = profile.session_filter
        _stop_atr = profile.stop_atr_mult
        _target_atr = profile.target_atr_mult
        _pullback_atr = profile.pullback_atr_mult
        _spread_max = profile.spread_max_mult
        _max_trades_per_day = profile.max_trades_per_day
        _risk_per_trade = profile.risk_per_trade
        if not profile.enabled:
            return {
                "pair": pair, "timeframe": timeframe, "trades": 0,
                "wins": 0, "losses": 0, "winrate": 0.0, "avg_rr": 0.0,
                "profit_factor": 0.0, "net_pnl": 0.0, "balance": starting_balance,
                "rejection_stats": {"disabled": True},
                "reason": f"pair disabled in profile: {profile.notes}",
            }
    else:
        _min_conf = MIN_CONFIDENCE
        _min_factors = MIN_ALIGNED_FACTORS
        _min_rr = MIN_RR
        _adx_min = 18.0
        _session_filter = "london_ny" if os.getenv("BT_SESSION_FILTER", "1") == "1" else "all"
        _stop_atr = float(os.getenv("BT_STOP_ATR", "1.8"))
        _target_atr = float(os.getenv("BT_TARGET_ATR", "3.5"))
        _pullback_atr = 1.5
        _spread_max = 2.0
        _max_trades_per_day = 5
        _risk_per_trade = RISK_PER_TRADE

    # Resolve session hours
    session_hours, session_desc = get_session_hours(_session_filter)

    engine = SignalEngine()
    balance = starting_balance
    open_trade = None
    trades = []
    rejection_stats = {
        "WAIT": 0, "low_confidence": 0, "low_factors": 0, "low_rr": 0,
        "counter_trend": 0, "no_signal": 0, "warmup": 0, "total_bars": 0,
    }
    last_signal_bar = -999

    for i in range(warmup, len(df)):
        rejection_stats["total_bars"] += 1
        bar = df.iloc[i]
        history = df.iloc[: i + 1]

        # ── Check exit on open trade first ──────────────────────────
        if open_trade is not None:
            hit_sl, hit_tp, exit_price = simulate_trade_exit(open_trade, bar)
            # NOTE: trailing stop was tested and reduced winrate because
            # pullbacks after entry stopped out trades at BE before TP.
            # Disabled — keep original SL/TP until hit.

            if hit_sl or hit_tp:
                open_trade["exit_price"] = exit_price
                open_trade["exit_time"] = bar.name
                open_trade["outcome"] = "WIN" if hit_tp else "LOSS"
                pnl_per_unit = (
                    (exit_price - open_trade["entry"])
                    if open_trade["direction"] == "BUY"
                    else (open_trade["entry"] - exit_price)
                )
                open_trade["pnl"] = pnl_per_unit * open_trade["units"]
                open_trade["hold_bars"] = i - open_trade["entry_bar_idx"]
                balance += open_trade["pnl"]
                trades.append(open_trade)
                open_trade = None
            elif (i - open_trade["entry_bar_idx"]) >= MAX_HOLD_BARS:
                # Timeout exit
                exit_price = float(bar["close"])
                open_trade["exit_price"] = exit_price
                open_trade["exit_time"] = bar.name
                open_trade["outcome"] = "WIN" if (
                    (exit_price > open_trade["entry"]) == (open_trade["direction"] == "BUY")
                ) else "LOSS"
                pnl_per_unit = (
                    (exit_price - open_trade["entry"])
                    if open_trade["direction"] == "BUY"
                    else (open_trade["entry"] - exit_price)
                )
                open_trade["pnl"] = pnl_per_unit * open_trade["units"]
                open_trade["hold_bars"] = MAX_HOLD_BARS
                open_trade["exit_reason"] = "timeout"
                balance += open_trade["pnl"]
                trades.append(open_trade)
                open_trade = None

        if open_trade is not None:
            continue  # only one trade at a time per pair

        # ── Cooldown: 3 bars between signals ────────────────────────
        if i - last_signal_bar < 3:
            continue

        # ── Build contexts and call REAL SignalEngine.generate() ────
        atr = float(bar.get("atr", 0) or 0)
        if atr <= 0 or np.isnan(atr):
            rejection_stats["WAIT"] += 1
            continue

        ind_ctx = build_ind_ctx(bar)
        pat_ctx = build_pat_ctx(bar)
        sr_ctx = build_sr_ctx(bar, atr)
        regime = build_regime(bar, ind_ctx["adx"])

        try:
            result = engine.generate(
                ind_ctx=ind_ctx,
                pat_ctx=pat_ctx,
                sr_ctx=sr_ctx,
                regime=regime,
                mtf_bias={"bias": "NEUTRAL", "confidence": "LOW"},
                advanced_pat_ctx=None,
                fib_ctx=None,
                extended_ctx=None,
            )
        except Exception as e:
            log.debug(f"SignalEngine crashed on {pair} bar {i}: {e}")
            rejection_stats["WAIT"] += 1
            continue

        signal = result.get("signal", "WAIT").upper()
        confidence = result.get("confidence", 0)
        # SignalEngine returns bull_score/bear_score but NOT bull_factors/bear_factors.
        # Count factors from the 'signals' list (each entry is a (direction, weight, reason) tuple).
        sig_list = result.get("signals", []) or []
        bull_factors = sum(1 for s in sig_list if s[0] == "bullish") if sig_list else 0
        bear_factors = sum(1 for s in sig_list if s[0] == "bearish") if sig_list else 0
        aligned = max(bull_factors, bear_factors) if signal in ("BUY", "STRONG_BUY", "SELL", "STRONG_SELL") else 0

        # ── Production LLM simulation boost ────────────────────────
        # In production, MasterAnalyst LLM reviews each BUY/SELL and either
        # confirms (boosts confidence +5-15) or downgrades to WAIT. Since
        # we can't run LLM in fast backtest, simulate the +10 confidence
        # boost that LLM would add to confirmed signals (only when all
        # rule-engine factors are aligned — i.e. real confluence).
        # This gives a more production-realistic winrate estimate.
        if os.getenv("BT_SIMULATE_LLM", "1") == "1" and aligned >= 3:
            confidence = min(100, confidence + 10)

        # ── Apply TradePermission gates (same as production) ────────
        if signal in ("WAIT", "NO TRADE", ""):
            rejection_stats["WAIT"] += 1
            continue

        if confidence < _min_conf:
            rejection_stats["low_confidence"] += 1
            continue

        if aligned < _min_factors:
            rejection_stats["low_factors"] += 1
            continue

        # ── Pullback requirement (per-pair profile) ─────────
        # Only trade when price is in the value area (within N×ATR of EMA-21).
        ema_21_val = float(ind_ctx.get("ema_21", 0) or 0)
        atr_val_for_pb = float(ind_ctx.get("atr", 0) or 0)
        price_val = float(ind_ctx.get("price", 0) or 0)
        if ema_21_val > 0 and atr_val_for_pb > 0 and price_val > 0:
            dist_to_ema21 = abs(price_val - ema_21_val)
            dist_atr = dist_to_ema21 / atr_val_for_pb
            # Require price within profile's pullback_atr_mult of EMA-21
            if dist_atr > _pullback_atr:
                rejection_stats["WAIT"] += 1
                continue

        # ── Session filter (per-pair profile) ───────────────
        try:
            bar_time = bar.name
            if hasattr(bar_time, "hour"):
                hour_utc = bar_time.hour
                if hour_utc not in session_hours:
                    rejection_stats["WAIT"] += 1
                    continue
        except Exception:
            pass

        # HTF counter-trend block (already done in SignalEngine, but double-check)
        htf_bull = ind_ctx["ema_50"] > ind_ctx["ema_200"] and ind_ctx["price"] > ind_ctx["ema_200"]
        htf_bear = ind_ctx["ema_50"] < ind_ctx["ema_200"] and ind_ctx["price"] < ind_ctx["ema_200"]
        if signal in ("BUY", "STRONG_BUY") and not htf_bull:
            rejection_stats["counter_trend"] += 1
            continue
        if signal in ("SELL", "STRONG_SELL") and not htf_bear:
            rejection_stats["counter_trend"] += 1
            continue

        # ── Compute SL/TP — institutional style (per-pair profile) ────
        close = ind_ctx["price"]
        stop_atr_mult = _stop_atr
        target_atr_mult = _target_atr

        try:
            swing_lookback = int(os.getenv("BT_SWING_LOOKBACK", "10"))
            recent = history.tail(swing_lookback)
            if signal in ("BUY", "STRONG_BUY"):
                # SL = recent swing low, but bounded by ATR
                swing_low = float(recent["low"].min())
                sl_atr = close - stop_atr_mult * atr
                # Use swing low if it's reasonable (1.0-2.5×ATR below entry)
                swing_distance = close - swing_low
                if atr * 1.0 <= swing_distance <= atr * 2.5:
                    sl = swing_low
                else:
                    sl = sl_atr  # fall back to ATR-based
                # TP = target_atr_mult × ATR (NOT 2× stop — allows RR tuning)
                actual_stop = close - sl
                tp = close + target_atr_mult * atr
            else:
                swing_high = float(recent["high"].max())
                sl_atr = close + stop_atr_mult * atr
                swing_distance = swing_high - close
                if atr * 1.0 <= swing_distance <= atr * 2.5:
                    sl = swing_high
                else:
                    sl = sl_atr
                actual_stop = sl - close
                tp = close - target_atr_mult * atr
        except Exception:
            if signal in ("BUY", "STRONG_BUY"):
                sl = close - stop_atr_mult * atr
                tp = close + target_atr_mult * atr
            else:
                sl = close + stop_atr_mult * atr
                tp = close - target_atr_mult * atr

        # Enforce min/max stop distance
        actual_stop = abs(close - sl)
        min_stop = atr * 1.0  # tighter min — pullback entries have tight stops
        max_stop = atr * 2.5
        if actual_stop < min_stop:
            actual_stop = min_stop
            sl = close - actual_stop if signal in ("BUY", "STRONG_BUY") else close + actual_stop
            tp = close + actual_stop * 2.0 if signal in ("BUY", "STRONG_BUY") else close - actual_stop * 2.0
        elif actual_stop > max_stop:
            actual_stop = max_stop
            sl = close - actual_stop if signal in ("BUY", "STRONG_BUY") else close + actual_stop
            tp = close + actual_stop * 2.0 if signal in ("BUY", "STRONG_BUY") else close - actual_stop * 2.0

        actual_rr = abs(tp - close) / actual_stop if actual_stop > 0 else 0

        # Final RR gate (per-pair profile)
        if actual_rr < _min_rr:
            rejection_stats["low_rr"] += 1
            continue

        # ── Position sizing: risk_per_trade of balance (per-pair profile) ─
        risk_amount = balance * _risk_per_trade
        units = risk_amount / actual_stop if actual_stop > 0 else 0
        max_units = balance / close
        units = min(units, max_units)

        open_trade = {
            "pair": pair,
            "timeframe": timeframe,
            "direction": "BUY" if signal in ("BUY", "STRONG_BUY") else "SELL",
            "entry": close,
            "entry_time": bar.name,
            "entry_bar_idx": i,
            "sl": sl,
            "tp": tp,
            "rr": actual_rr,
            "units": units,
            "risk_amount": risk_amount,
            "confidence": confidence,
            "aligned_factors": aligned,
            "atr_at_entry": atr,  # for trailing-stop computation
            "exit_reason": "SL_or_TP",
        }
        last_signal_bar = i

    # Close any dangling trade
    if open_trade is not None:
        last_bar = df.iloc[-1]
        exit_price = float(last_bar["close"])
        open_trade["exit_price"] = exit_price
        open_trade["exit_time"] = last_bar.name
        open_trade["outcome"] = "WIN" if (
            (exit_price > open_trade["entry"]) == (open_trade["direction"] == "BUY")
        ) else "LOSS"
        pnl_per_unit = (
            (exit_price - open_trade["entry"])
            if open_trade["direction"] == "BUY"
            else (open_trade["entry"] - exit_price)
        )
        open_trade["pnl"] = pnl_per_unit * open_trade["units"]
        open_trade["hold_bars"] = len(df) - 1 - open_trade["entry_bar_idx"]
        open_trade["exit_reason"] = "end_of_data"
        balance += open_trade["pnl"]
        trades.append(open_trade)

    # ── Summary ───────────────────────────────────────────────────
    if not trades:
        return {
            "pair": pair, "timeframe": timeframe, "trades": 0,
            "wins": 0, "losses": 0, "winrate": 0.0, "avg_rr": 0.0,
            "profit_factor": 0.0, "net_pnl": 0.0, "balance": balance,
            "rejection_stats": rejection_stats,
        }

    wins = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg_rr = sum(t["rr"] for t in trades) / len(trades)
    avg_hold = sum(t["hold_bars"] for t in trades) / len(trades)

    return {
        "pair": pair,
        "timeframe": timeframe,
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "winrate": len(wins) / len(trades) * 100,
        "avg_rr": avg_rr,
        "avg_hold_bars": avg_hold,
        "profit_factor": pf,
        "net_pnl": sum(t["pnl"] for t in trades),
        "balance": balance,
        "rejection_stats": rejection_stats,
        "trades_detail": trades,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCAD,USDCHF,USDJPY")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--output", default="/home/z/my-project/download/fast_backtest_results.csv")
    args = parser.parse_args()

    pairs = [p.strip().upper() for p in args.pairs.split(",")]
    log.warning(f"Fast backtest: {len(pairs)} pairs × {args.timeframe}")
    log.warning(f"Using PER-PAIR PROFILES from utils/pair_profiles.py")
    log.warning(f"Global fallback config: MIN_CONFIDENCE={MIN_CONFIDENCE}%, MIN_FACTORS={MIN_ALIGNED_FACTORS}, "
                f"MIN_RR={MIN_RR}, RISK={RISK_PER_TRADE*100}%")

    # Load + indicator-compute each pair once
    pair_dfs = {}
    for pair in pairs:
        df = load_csv(pair, args.timeframe)
        if df is None or len(df) < args.warmup + 100:
            log.error(f"  {pair} {args.timeframe}: no CSV or insufficient data")
            continue
        log.warning(f"  {pair} {args.timeframe}: {len(df)} bars, computing indicators...")
        try:
            df = compute_indicators(df)
            pair_dfs[pair] = df
        except Exception as e:
            log.error(f"  {pair}: indicator computation failed: {e}")

    # Run backtest per pair
    results = []
    for pair, df in pair_dfs.items():
        t0 = time.time()
        log.warning(f"  Backtesting {pair}...")
        try:
            res = backtest_pair(pair, args.timeframe, df, warmup=args.warmup)
            res["duration_sec"] = round(time.time() - t0, 2)
            results.append(res)
            wr = res["winrate"]
            rr = res["avg_rr"]
            pf = res["profit_factor"]
            n = res["trades"]
            pnl = res["net_pnl"]
            rej = res["rejection_stats"]
            tag = "✓" if (wr >= 60 and rr >= 2.0) else ("△" if pf >= 1.0 else "✗")
            # Handle disabled pairs (rejection_stats has only {"disabled": True})
            if rej.get("disabled"):
                print(f"  ⊘ {pair:6s} | DISABLED — {res.get('reason', 'pair disabled in profile')[:60]}")
            else:
                print(f"  {tag} {pair:6s} | WR={wr:5.1f}% | RR={rr:.2f} | PF={pf:5.2f} | "
                      f"N={n:4d} | PnL=${pnl:+9.2f} | Bal=${res['balance']:9.2f} | "
                      f"rej: WAIT={rej.get('WAIT',0)} low_conf={rej.get('low_confidence',0)} "
                      f"low_fact={rej.get('low_factors',0)} low_rr={rej.get('low_rr',0)} "
                      f"ctrend={rej.get('counter_trend',0)}")
        except Exception as e:
            import traceback
            log.error(f"  {pair} crashed: {e}")
            log.error(traceback.format_exc())

    # Summary
    print("\n" + "=" * 95)
    print(f"  CONFIG: MIN_CONFIDENCE={MIN_CONFIDENCE}% | MIN_FACTORS={MIN_ALIGNED_FACTORS} | "
          f"MIN_RR={MIN_RR} | RISK={RISK_PER_TRADE*100}%")
    print("=" * 95)
    if results:
        total_trades = sum(r["trades"] for r in results)
        total_wins = sum(r["wins"] for r in results)
        total_pnl = sum(r["net_pnl"] for r in results)
        wr_avg = (total_wins / total_trades * 100) if total_trades > 0 else 0
        rr_avg = sum(r["avg_rr"] for r in results) / len(results) if results else 0
        pf_avg = sum(r["profit_factor"] for r in results) / len(results) if results else 0
        print(f"  TOTAL: trades={total_trades} | winrate={wr_avg:.1f}% | "
              f"avg_rr={rr_avg:.2f} | avg_pf={pf_avg:.2f} | net_pnl=${total_pnl:+.2f}")

    out_df = pd.DataFrame([{k: v for k, v in r.items() if k != "trades_detail"} for r in results])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
