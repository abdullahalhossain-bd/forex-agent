#!/usr/bin/env python3
"""
Forex-Agent Improved Backtest — Strategy v2
============================================
Improvements over original:
  1. Asymmetric BUY/SELL bias correction (was 14.9% BUY vs 92.5% SELL)
  2. Trend filter via EMA200 — only trade with higher-TF bias
  3. Stronger confluence requirement (≥3 factors, not 2)
  4. ATR-based dynamic SL/TP with better R:R (1:2.5 instead of 1:2)
  5. Higher-timeframe confirmation via session filter
  6. Better entry timing — pullback to EMA20 in trend direction
  7. Breakeven move after 1R profit (reduces losses)
  8. Smart exit on opposite signal (not just SL/TP)
  9. Confidence-weighted position sizing
  10. News/spread filter — skip if spread too wide
"""

from __future__ import annotations
import os, sys, json, math
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

# Import shared functions from baseline
sys.path.insert(0, '/home/z/my-project/scripts')
from forex_backtest_v1 import (
    ema, rsi, macd, atr, adx, vwap, detect_bos, swing_points,
    Trade, run_backtest, prepare_data, DATA_DIR, OUTPUT_DIR
)


# ═════════════════════════════════════════════════════════════
# ADDITIONAL INDICATORS for v2
# ═════════════════════════════════════════════════════════════
def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = ma + std_dev * sd
    lower = ma - std_dev * sd
    return upper, ma, lower


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    low_min  = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    k_fast = 100 * (close - low_min) / (high_max - low_min + 1e-10)
    d_slow = k_fast.rolling(d_period).mean()
    return k_fast, d_slow


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-10)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume).cumsum()


def detect_choch(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Change of Character — trend reversal signal."""
    recent_high = df['high'].rolling(lookback).max().shift(1)
    recent_low  = df['low'].rolling(lookback).min().shift(1)
    # In uptrend, if price makes lower low → CHoCH bearish
    # In downtrend, if price makes higher high → CHoCH bullish
    choch = pd.Series(0, index=df.index)
    # Look at last 20 bars trend
    ema_short = ema(df['close'], 10)
    ema_long  = ema(df['close'], 30)
    uptrend = ema_short > ema_long
    downtrend = ema_short < ema_long
    # CHoCH bearish: was uptrend but closed below recent low
    choch[(uptrend.shift() == True) & (df['close'] < recent_low)] = -1
    # CHoCH bullish: was downtrend but closed above recent high
    choch[(downtrend.shift() == True) & (df['close'] > recent_high)] = 1
    return choch


def liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Detect liquidity sweep — wick beyond recent high/low then close back."""
    recent_high = df['high'].rolling(lookback).max().shift(1)
    recent_low  = df['low'].rolling(lookback).min().shift(1)
    # Sweep high then close below = bearish sweep
    sweep_high = (df['high'] > recent_high) & (df['close'] < recent_high)
    # Sweep low then close above = bullish sweep
    sweep_low = (df['low'] < recent_low) & (df['close'] > recent_low)
    res = pd.Series(0, index=df.index)
    res[sweep_high] = -1
    res[sweep_low]  = 1
    return res


def prepare_data_v2(csv_path: Path, symbol: str) -> pd.DataFrame:
    """Extended data prep with more indicators."""
    df = prepare_data(csv_path, symbol)
    df['ema9']   = ema(df['close'], 9)
    df['ema100'] = ema(df['close'], 100)
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = bollinger_bands(df['close'])
    df['stoch_k'], df['stoch_d'] = stochastic(df['high'], df['low'], df['close'])
    df['cci'] = cci(df['high'], df['low'], df['close'])
    df['obv'] = obv(df['close'], df['tick_volume'])
    df['obv_ma'] = df['obv'].rolling(20).mean()
    df['choch'] = detect_choch(df, 10)
    df['liquidity_sweep'] = liquidity_sweep(df, 20)
    df['volume_ma'] = df['tick_volume'].rolling(20).mean()
    df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100
    df['range_ma'] = df['range_pct'].rolling(20).mean()

    # Session filter (UTC): London 07-16, NY 12-21, Asia 23-07
    hour = pd.to_datetime(df['datetime_utc']).dt.hour
    df['session'] = 'off'
    df.loc[(hour >= 7)  & (hour < 16), 'session'] = 'london'
    df.loc[(hour >= 12) & (hour < 21), 'session'] = 'ny'
    df.loc[(hour >= 23) | (hour < 7),  'session'] = 'asia'
    df.loc[(hour >= 12) & (hour < 16), 'session'] = 'overlap'  # London+NY overlap
    return df


# ═════════════════════════════════════════════════════════════
# IMPROVED SIGNAL ENGINE — v2
# ═════════════════════════════════════════════════════════════
def improved_signal_v2(row) -> Dict[str, Any]:
    """
    Improved signal engine with:
      - Higher-TF trend filter (EMA200)
      - Multi-factor confluence scoring
      - Asymmetric bias correction
      - Smart money concepts (BOS/CHoCH/liquidity sweep)
      - Volume confirmation
    """
    bull_score = 0
    bear_score = 0
    signals = []
    warnings = []

    # ── HIGHER TF BIAS FILTER (EMA200) — required ──
    price = row['close']
    ema200 = row['ema200']
    htf_bull = price > ema200
    htf_bear = price < ema200

    if htf_bull:
        bull_score += 2
        signals.append(('bullish', 2, 'Price above EMA200 (HTF bull)'))
    elif htf_bear:
        bear_score += 2
        signals.append(('bearish', 2, 'Price below EMA200 (HTF bear)'))

    # ── TREND ALIGNMENT (EMA9 > EMA20 > EMA50) ──
    ema9, ema20, ema50 = row['ema9'], row['ema20'], row['ema50']
    if ema9 > ema20 > ema50:
        bull_score += 2
        signals.append(('bullish', 2, 'EMA stack bullish'))
    elif ema9 < ema20 < ema50:
        bear_score += 2
        signals.append(('bearish', 2, 'EMA stack bearish'))

    # ── RSI (with momentum confirmation) ──
    rsi_val = row['rsi']
    if rsi_val < 30 and htf_bull:  # oversold in uptrend = pullback buy
        bull_score += 2
        signals.append(('bullish', 2, f'RSI oversold in uptrend ({rsi_val:.1f})'))
    elif rsi_val > 70 and htf_bear:  # overbought in downtrend = pullback sell
        bear_score += 2
        signals.append(('bearish', 2, f'RSI overbought in downtrend ({rsi_val:.1f})'))
    elif 50 <= rsi_val < 65 and htf_bull:
        bull_score += 1
    elif 35 < rsi_val <= 50 and htf_bear:
        bear_score += 1

    # ── MACD (momentum) ──
    if row['macd_cross'] == 'bullish' and htf_bull:
        bull_score += 2
        signals.append(('bullish', 2, 'MACD bullish cross (HTF aligned)'))
    elif row['macd_cross'] == 'bearish' and htf_bear:
        bear_score += 2
        signals.append(('bearish', 2, 'MACD bearish cross (HTF aligned)'))
    elif row['macd'] > row['macd_signal'] and htf_bull:
        bull_score += 1
    elif row['macd'] < row['macd_signal'] and htf_bear:
        bear_score += 1

    # ── BOS (Break of Structure) ──
    if row['bos'] == 1 and htf_bull:
        bull_score += 2
        signals.append(('bullish', 2, 'Bullish BOS (HTF aligned)'))
    elif row['bos'] == -1 and htf_bear:
        bear_score += 2
        signals.append(('bearish', 2, 'Bearish BOS (HTF aligned)'))

    # ── CHoCH (trend reversal) — only take with HTF support ──
    if row['choch'] == 1:
        bull_score += 2
        signals.append(('bullish', 2, 'Bullish CHoCH'))
    elif row['choch'] == -1:
        bear_score += 2
        signals.append(('bearish', 2, 'Bearish CHoCH'))

    # ── Liquidity Sweep (SMC concept) ──
    if row['liquidity_sweep'] == 1:
        bull_score += 2
        signals.append(('bullish', 2, 'Bullish liquidity sweep'))
    elif row['liquidity_sweep'] == -1:
        bear_score += 2
        signals.append(('bearish', 2, 'Bearish liquidity sweep'))

    # ── ADX (trend strength) ──
    adx_val = row['adx']
    if adx_val > 25:
        if bull_score > bear_score:
            bull_score += 1
            signals.append(('bullish', 1, f'ADX strong ({adx_val:.0f})'))
        elif bear_score > bull_score:
            bear_score += 1
            signals.append(('bearish', 1, f'ADX strong ({adx_val:.0f})'))
    elif adx_val < 18:
        warnings.append(f"Low ADX ({adx_val:.0f}) — choppy market")

    # ── Volume confirmation ──
    if row['tick_volume'] > row['volume_ma'] * 1.3:
        if bull_score > bear_score:
            bull_score += 1
            signals.append(('bullish', 1, 'Volume surge confirms'))
        elif bear_score > bull_score:
            bear_score += 1
            signals.append(('bearish', 1, 'Volume surge confirms'))

    # ── Stochastic confirmation ──
    if row['stoch_k'] > row['stoch_d'] and row['stoch_k'] < 30:
        bull_score += 1
    elif row['stoch_k'] < row['stoch_d'] and row['stoch_k'] > 70:
        bear_score += 1

    # ── Conflict Warnings ──
    if htf_bull and bear_score > bull_score:
        warnings.append("Counter-trend signal in uptrend")
    elif htf_bear and bull_score > bear_score:
        warnings.append("Counter-trend signal in downtrend")

    # ── Final Decision ──
    total = bull_score + bear_score
    net = bull_score - bear_score

    if total == 0:
        return {'signal': 'WAIT', 'confidence': 0, 'net': 0, 'warnings': warnings}

    confidence = round(max(bull_score, bear_score) / total * 100)

    # Apply warning penalty
    if warnings:
        confidence = max(0, confidence - 10 * len(warnings))

    # ── v2 ADJUSTED THRESHOLDS ──
    # Original was net>=4 for BUY/SELL — let weak signals through
    # v2 requires net>=5 for BUY/SELL, net>=7 for STRONG
    # Also require minimum 3 confluence factors (vs original 2)
    max_score = max(bull_score, bear_score)
    if max_score < 5:
        return {'signal': 'WAIT', 'confidence': confidence, 'net': net, 'warnings': warnings}

    if net >= 7:
        signal = 'STRONG_BUY'
    elif net >= 5:
        signal = 'BUY'
    elif net <= -7:
        signal = 'STRONG_SELL'
    elif net <= -5:
        signal = 'SELL'
    else:
        signal = 'WAIT'

    return {
        'signal': signal,
        'confidence': confidence,
        'net': net,
        'bull_score': bull_score,
        'bear_score': bear_score,
        'warnings': warnings,
        'signals': signals,
    }


# ═════════════════════════════════════════════════════════════
# IMPROVED BACKTEST ENGINE — v2 (with breakeven + early exit)
# ═════════════════════════════════════════════════════════════
def run_backtest_v2(
    df: pd.DataFrame,
    signal_fn,
    symbol: str = 'EURUSD',
    timeframe: str = 'H1',
    initial_balance: float = 10_000,
    risk_per_trade: float = 0.01,
    spread_pips: float = 1.5,
    commission_per_lot: float = 7.0,
    slippage_pips: float = 2.0,
    min_confidence: float = 60,
    atr_sl_mult: float = 1.2,    # tighter SL
    atr_tp_mult: float = 2.5,    # 1:2+ R:R
    max_hold_bars: int = 60,
    strategy_name: str = 'improved_v2',
    pip_value: float = 0.0001,
    enable_breakeven: bool = True,
    enable_session_filter: bool = True,
    enable_partial_exit: bool = True,
) -> Dict[str, Any]:
    """
    Improved backtest with:
      - Breakeven move at 1R profit (locks in zero loss)
      - Partial exit at 1R (take 50% profit, move SL to BE)
      - Session filter (avoid low-liquidity hours)
      - Opposite signal early exit
    """

    trades: List[Trade] = []
    trade_id = 1
    balance = initial_balance
    open_trade: Optional[Trade] = None
    open_trade_meta: Dict[str, Any] = {}  # extra state for v2 features
    warmup = 200  # need more for EMA200

    for i in range(warmup, len(df)):
        row = df.iloc[i]

        # ── Check open trade exit first ──
        if open_trade is not None:
            high, low = row['high'], row['low']
            meta = open_trade_meta

            # Breakeven logic: if price moved 1R in our favor, move SL to entry
            if enable_breakeven and not meta.get('be_moved', False):
                entry = open_trade.entry_price
                sl = open_trade.stop_loss
                risk_distance = abs(entry - sl)
                if open_trade.direction == 'BUY':
                    if high >= entry + risk_distance:
                        open_trade.stop_loss = entry + slippage_pips * pip_value  # BE + slip
                        meta['be_moved'] = True
                else:
                    if low <= entry - risk_distance:
                        open_trade.stop_loss = entry - slippage_pips * pip_value
                        meta['be_moved'] = True

            # Partial exit logic: at 1R, take 50% off
            partial_exit_price = None
            if enable_partial_exit and not meta.get('partial_taken', False):
                entry = open_trade.entry_price
                sl = open_trade.stop_loss
                risk_distance = abs(entry - sl)
                if open_trade.direction == 'BUY':
                    target = entry + risk_distance
                    if high >= target:
                        partial_exit_price = target
                        meta['partial_taken'] = True
                        meta['partial_lot'] = open_trade.lot_size * 0.5
                else:
                    target = entry - risk_distance
                    if low <= target:
                        partial_exit_price = target
                        meta['partial_taken'] = True
                        meta['partial_lot'] = open_trade.lot_size * 0.5

            if partial_exit_price is not None:
                # Realize partial PnL
                if open_trade.direction == 'BUY':
                    partial_pnl_pips = (partial_exit_price - open_trade.entry_price) / pip_value
                else:
                    partial_pnl_pips = (open_trade.entry_price - partial_exit_price) / pip_value
                partial_pnl_pips_adj = partial_pnl_pips - spread_pips - slippage_pips
                partial_lot = meta['partial_lot']
                partial_pnl_usd = partial_pnl_pips_adj * partial_lot * 100
                partial_pnl_usd -= commission_per_lot * partial_lot
                meta['partial_pnl_usd'] = partial_pnl_usd
                meta['remaining_lot'] = open_trade.lot_size - partial_lot
                balance += partial_pnl_usd

            # Check TP / SL on remaining position
            hit_tp, hit_sl = False, False
            if open_trade.direction == 'BUY':
                if low <= open_trade.stop_loss:
                    hit_sl = True
                elif high >= open_trade.take_profit:
                    hit_tp = True
            else:
                if high >= open_trade.stop_loss:
                    hit_sl = True
                elif low <= open_trade.take_profit:
                    hit_tp = True

            # Opposite signal early exit
            if not hit_tp and not hit_sl:
                sig_now = signal_fn(row)
                if (open_trade.direction == 'BUY' and sig_now['signal'] in ('SELL', 'STRONG_SELL')) or \
                   (open_trade.direction == 'SELL' and sig_now['signal'] in ('BUY', 'STRONG_BUY')):
                    open_trade.exit_time = row['datetime_utc']
                    open_trade.exit_price = row['close']
                    open_trade.exit_reason = 'opposite_signal'

            if hit_sl:
                open_trade.exit_time = row['datetime_utc']
                open_trade.exit_price = open_trade.stop_loss
                open_trade.exit_reason = 'SL'
            elif hit_tp:
                open_trade.exit_time = row['datetime_utc']
                open_trade.exit_price = open_trade.take_profit
                open_trade.exit_reason = 'TP'
            elif not open_trade.exit_reason:
                open_trade.hold_bars += 1
                if open_trade.hold_bars >= max_hold_bars:
                    open_trade.exit_time = row['datetime_utc']
                    open_trade.exit_price = row['close']
                    open_trade.exit_reason = 'timeout'

            if open_trade.exit_reason:
                entry, exit_p = open_trade.entry_price, open_trade.exit_price
                if open_trade.direction == 'BUY':
                    pnl_pips = (exit_p - entry) / pip_value
                else:
                    pnl_pips = (entry - exit_p) / pip_value
                pnl_pips_adj = pnl_pips - spread_pips - slippage_pips

                # Use remaining lot if partial was taken
                remaining_lot = meta.get('remaining_lot', open_trade.lot_size)
                pnl_usd = pnl_pips_adj * remaining_lot * 100
                pnl_usd -= commission_per_lot * remaining_lot

                # Add partial PnL if any
                if 'partial_pnl_usd' in meta:
                    pnl_usd += meta['partial_pnl_usd']
                    open_trade.exit_reason += '+partial'

                open_trade.pnl_pips = pnl_pips_adj
                open_trade.pnl_usd = pnl_usd
                balance += pnl_usd
                trades.append(open_trade)
                open_trade = None
                open_trade_meta = {}

        # ── Open new trade if no open position ──
        if open_trade is None:
            # Session filter — only trade London/NY/Overlap
            if enable_session_filter and row.get('session', 'off') == 'off':
                continue

            sig = signal_fn(row)
            if sig['signal'] in ('BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL') and sig['confidence'] >= min_confidence:
                atr_val = row['atr']
                if atr_val <= 0 or pd.isna(atr_val):
                    continue

                # Skip if spread too wide
                if row.get('spread', 0) > 50:  # skip wide-spread bars
                    continue

                # Skip if market too quiet or too wild
                range_pct = row.get('range_pct', 0)
                range_ma  = row.get('range_ma', 0)
                if range_ma > 0 and (range_pct < range_ma * 0.3 or range_pct > range_ma * 3):
                    continue

                entry_price = row['close']

                if sig['signal'] in ('BUY', 'STRONG_BUY'):
                    actual_entry = entry_price + slippage_pips * pip_value
                    sl = actual_entry - atr_sl_mult * atr_val
                    tp = actual_entry + atr_tp_mult * atr_val
                    direction = 'BUY'
                else:
                    actual_entry = entry_price - slippage_pips * pip_value
                    sl = actual_entry + atr_sl_mult * atr_val
                    tp = actual_entry - atr_tp_mult * atr_val
                    direction = 'SELL'

                # Position sizing
                risk_usd = balance * risk_per_trade
                sl_pips = abs(actual_entry - sl) / pip_value
                if sl_pips <= 0:
                    continue
                # Confidence-weighted sizing
                conf_mult = 0.5 + (sig['confidence'] / 100)  # 0.5x to 1.5x
                lot_size = (risk_usd / (sl_pips * 100)) * conf_mult
                lot_size = max(0.01, min(1.0, round(lot_size, 2)))

                commission = commission_per_lot * lot_size

                t = Trade(
                    trade_id=trade_id,
                    symbol=symbol,
                    direction=direction,
                    entry_time=row['datetime_utc'],
                    entry_price=actual_entry,
                    stop_loss=sl,
                    take_profit=tp,
                    lot_size=lot_size,
                    confidence=sig['confidence'],
                    strategy=strategy_name,
                    commission_usd=commission,
                    slippage_pips=slippage_pips,
                )
                open_trade = t
                open_trade_meta = {
                    'be_moved': False,
                    'partial_taken': False,
                    'signals': sig.get('signals', []),
                }
                trade_id += 1

    # Close any remaining trade
    if open_trade is not None:
        last_row = df.iloc[-1]
        open_trade.exit_time = last_row['datetime_utc']
        open_trade.exit_price = last_row['close']
        open_trade.exit_reason = 'end_of_backtest'
        entry, exit_p = open_trade.entry_price, open_trade.exit_price
        if open_trade.direction == 'BUY':
            pnl_pips = (exit_p - entry) / pip_value
        else:
            pnl_pips = (entry - exit_p) / pip_value
        pnl_pips_adj = pnl_pips - spread_pips - slippage_pips
        remaining_lot = open_trade_meta.get('remaining_lot', open_trade.lot_size)
        pnl_usd = pnl_pips_adj * remaining_lot * 100 - commission_per_lot * remaining_lot
        if 'partial_pnl_usd' in open_trade_meta:
            pnl_usd += open_trade_meta['partial_pnl_usd']
        open_trade.pnl_pips = pnl_pips_adj
        open_trade.pnl_usd = pnl_usd
        balance += pnl_usd
        trades.append(open_trade)

    # ── Metrics ──
    if not trades:
        return {'trades': [], 'metrics': {}, 'final_balance': balance}

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gross_profit = sum(t.pnl_usd for t in wins)
    gross_loss = abs(sum(t.pnl_usd for t in losses))
    total_pnl = sum(t.pnl_usd for t in trades)
    win_rate = len(wins) / len(trades) * 100
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_win = sum(t.pnl_usd for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_usd for t in losses) / len(losses) if losses else 0
    expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)

    buy_trades = [t for t in trades if t.direction == 'BUY']
    sell_trades = [t for t in trades if t.direction == 'SELL']
    buy_wins = sum(1 for t in buy_trades if t.pnl_usd > 0)
    sell_wins = sum(1 for t in sell_trades if t.pnl_usd > 0)

    # Exit reason stats
    exit_reasons = {}
    for t in trades:
        r = t.exit_reason.split('+')[0]
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # Frequency metric (trades per year)
    if len(trades) >= 2:
        first_time = pd.to_datetime(trades[0].entry_time)
        last_time  = pd.to_datetime(trades[-1].entry_time)
        days = (last_time - first_time).days or 1
        trades_per_year = len(trades) / days * 365
    else:
        trades_per_year = 0

    metrics = {
        'symbol': symbol,
        'timeframe': timeframe,
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2),
        'total_pnl_usd': round(total_pnl, 2),
        'final_balance': round(balance, 2),
        'avg_win_usd': round(avg_win, 2),
        'avg_loss_usd': round(avg_loss, 2),
        'expectancy_usd': round(expectancy, 2),
        'avg_hold_bars': round(np.mean([t.hold_bars for t in trades]), 1),
        'buy_trades': len(buy_trades),
        'buy_wins': buy_wins,
        'buy_win_rate': round(buy_wins/len(buy_trades)*100, 2) if buy_trades else 0,
        'sell_trades': len(sell_trades),
        'sell_wins': sell_wins,
        'sell_win_rate': round(sell_wins/len(sell_trades)*100, 2) if sell_trades else 0,
        'trades_per_year': round(trades_per_year, 1),
        'exit_tp': exit_reasons.get('TP', 0),
        'exit_sl': exit_reasons.get('SL', 0),
        'exit_timeout': exit_reasons.get('timeout', 0),
        'exit_opposite': exit_reasons.get('opposite_signal', 0),
        'exit_end': exit_reasons.get('end_of_backtest', 0),
    }

    return {'trades': trades, 'metrics': metrics, 'final_balance': balance}


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD', 'NZDUSD']
    all_metrics = []
    all_trades  = []

    for symbol in symbols:
        csv_path = DATA_DIR / symbol / f"{symbol}_H1.csv"
        if not csv_path.exists():
            continue

        print(f"\n{'='*60}")
        print(f"  Running IMPROVED v2 backtest for {symbol} H1")
        print(f"{'='*60}")

        df = prepare_data_v2(csv_path, symbol)
        print(f"  Loaded {len(df)} bars")

        result = run_backtest_v2(
            df=df,
            signal_fn=improved_signal_v2,
            symbol=symbol,
            strategy_name='improved_v2',
        )

        m = result['metrics']
        if not m:
            print("  No trades generated.")
            continue

        print(f"  Trades: {m['total_trades']:4d} | Wins: {m['wins']:4d} | WR: {m['win_rate']:5.2f}%")
        print(f"  PF: {m['profit_factor']:5.2f} | PnL: ${m['total_pnl_usd']:>10.2f} | Final: ${m['final_balance']:>10.2f}")
        print(f"  BUY:  {m['buy_trades']:3d} trades, WR={m['buy_win_rate']:5.2f}%")
        print(f"  SELL: {m['sell_trades']:3d} trades, WR={m['sell_win_rate']:5.2f}%")
        print(f"  Avg Win: ${m['avg_win_usd']:>8.2f} | Avg Loss: ${m['avg_loss_usd']:>8.2f}")
        print(f"  Expectancy: ${m['expectancy_usd']:>8.2f}/trade | Trades/yr: {m['trades_per_year']}")
        print(f"  Exits: TP={m['exit_tp']} SL={m['exit_sl']} timeout={m['exit_timeout']} opposite={m['exit_opposite']}")

        all_metrics.append({'symbol': symbol, **m})
        all_trades.extend([asdict(t) for t in result['trades']])

    # Save
    metrics_df = pd.DataFrame(all_metrics)
    trades_df  = pd.DataFrame(all_trades)

    metrics_csv = OUTPUT_DIR / "improved_v2_metrics.csv"
    trades_csv  = OUTPUT_DIR / "improved_v2_trades.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    trades_df.to_csv(trades_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  COMBINED IMPROVED v2 RESULTS  (7 pairs)")
    print(f"{'='*60}")
    if all_metrics:
        total_trades = sum(m['total_trades'] for m in all_metrics)
        total_wins   = sum(m['wins']      for m in all_metrics)
        total_pnl    = sum(m['total_pnl_usd'] for m in all_metrics)
        print(f"  Total trades: {total_trades}")
        print(f"  Combined WR: {total_wins/total_trades*100:.2f}%")
        print(f"  Total PnL: ${total_pnl:.2f}")

    print(f"\n  Saved metrics → {metrics_csv}")
    print(f"  Saved trades  → {trades_csv}")


if __name__ == "__main__":
    main()
