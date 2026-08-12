#!/usr/bin/env python3
"""
Forex-Agent Standalone Backtest — Original Strategy Recreation
==============================================================
Recreates the original signal_engine.py logic on data/history CSVs
to establish a baseline win rate / frequency.

Inputs:  data/history/{SYMBOL}/{SYMBOL}_H1.csv
Outputs: baseline metrics + trade list CSV
"""

from __future__ import annotations
import os
import sys
import json
import math
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/z/my-project/download/forex-agent")
DATA_DIR     = PROJECT_ROOT / "data" / "history"
OUTPUT_DIR   = Path("/home/z/my-project/download")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════
# INDICATOR COMPUTATIONS
# ═════════════════════════════════════════════════════════════
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_pc  = (df['high'] - df['close'].shift()).abs()
    low_pc   = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    plus_dm  = (high.diff()).where((high.diff() > -low.diff()) & (high.diff() > 0), 0.0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.rolling(period).mean()
    plus_di  = 100 * (plus_dm.rolling(period).mean() / (atr_ + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr_ + 1e-10))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean()


def vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    return (typical_price * df['tick_volume']).rolling(window).sum() / df['tick_volume'].rolling(window).sum()


def detect_bos(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Break of structure — price closes beyond recent swing high/low."""
    recent_high = df['high'].rolling(lookback).max().shift(1)
    recent_low  = df['low'].rolling(lookback).min().shift(1)
    bos_bull = (df['close'] > recent_high).astype(int)
    bos_bear = (df['close'] < recent_low).astype(int)
    return bos_bull - bos_bear  # +1 bull BOS, -1 bear BOS, 0 none


def swing_points(df: pd.DataFrame, lookback: int = 5):
    """Detect swing highs/lows for S/R."""
    swing_high = df['high'].rolling(lookback*2+1, center=True).max()
    swing_low  = df['low'].rolling(lookback*2+1, center=True).min()
    is_swing_high = (df['high'] == swing_high).astype(bool)
    is_swing_low  = (df['low']  == swing_low).astype(bool)
    return is_swing_high, is_swing_low


# ═════════════════════════════════════════════════════════════
# ORIGINAL SIGNAL ENGINE (recreated from strategy/signal_engine.py)
# ═════════════════════════════════════════════════════════════
def original_signal(row) -> Dict[str, Any]:
    """Recreate original signal engine scoring logic."""
    bull_score = 0
    bear_score = 0
    signals = []

    # Trend (EMA20 vs EMA50)
    if row['ema20'] > row['ema50'] * 1.001:
        bull_score += 2
        signals.append(('bullish', 2, 'Bullish trend'))
    elif row['ema20'] < row['ema50'] * 0.999:
        bear_score += 2
        signals.append(('bearish', 2, 'Bearish trend'))

    # RSI
    rsi_val = row['rsi']
    if rsi_val < 30:
        bull_score += 2
        signals.append(('bullish', 2, f'RSI oversold ({rsi_val:.1f})'))
    elif rsi_val > 70:
        bear_score += 2
        signals.append(('bearish', 2, f'RSI overbought ({rsi_val:.1f})'))
    elif 50 <= rsi_val < 70:
        bull_score += 1
        signals.append(('bullish', 1, f'RSI bullish zone ({rsi_val:.1f})'))
    elif 30 < rsi_val <= 50:
        bear_score += 1
        signals.append(('bearish', 1, f'RSI bearish zone ({rsi_val:.1f})'))

    # MACD
    if row['macd_cross'] == 'bullish':
        bull_score += 1
    elif row['macd_cross'] == 'bearish':
        bear_score += 1

    # BOS (structure)
    if row['bos'] == 1:
        bull_score += 2
        signals.append(('bullish', 2, 'Bullish BOS'))
    elif row['bos'] == -1:
        bear_score += 2
        signals.append(('bearish', 2, 'Bearish BOS'))

    # ADX (trend strength confirmation)
    if row['adx'] > 25:
        if bull_score > bear_score:
            bull_score += 1
        elif bear_score > bull_score:
            bear_score += 1

    total = bull_score + bear_score
    net = bull_score - bear_score

    if total == 0:
        return {'signal': 'WAIT', 'confidence': 0, 'net': 0}

    confidence = round(max(bull_score, bear_score) / total * 100)

    # Original thresholds
    if net >= 6:
        signal = 'STRONG_BUY'
    elif net >= 4:
        signal = 'BUY'
    elif net <= -6:
        signal = 'STRONG_SELL'
    elif net <= -4:
        signal = 'SELL'
    else:
        signal = 'WAIT'

    return {'signal': signal, 'confidence': confidence, 'net': net,
            'bull_score': bull_score, 'bear_score': bear_score}


# ═════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════
@dataclass
class Trade:
    trade_id: int
    symbol: str
    direction: str
    entry_time: str
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    confidence: float
    strategy: str
    exit_time: str = ''
    exit_price: float = 0.0
    exit_reason: str = ''
    pnl_pips: float = 0.0
    pnl_usd: float = 0.0
    commission_usd: float = 0.0
    slippage_pips: float = 0.0
    hold_bars: int = 0


def run_backtest(
    df: pd.DataFrame,
    signal_fn,
    symbol: str = 'EURUSD',
    timeframe: str = 'H1',
    initial_balance: float = 10_000,
    risk_per_trade: float = 0.01,
    spread_pips: float = 1.5,
    commission_per_lot: float = 7.0,
    slippage_pips: float = 2.0,
    min_confidence: float = 55,
    atr_sl_mult: float = 1.5,
    atr_tp_mult: float = 3.0,
    max_hold_bars: int = 100,
    strategy_name: str = 'original',
    pip_value: float = 0.0001,  # 1 pip = 0.0001 for non-JPY pairs
) -> Dict[str, Any]:
    """Run backtest on prepared dataframe."""

    trades: List[Trade] = []
    trade_id = 1
    balance = initial_balance
    open_trade: Optional[Trade] = None
    warmup = 60

    for i in range(warmup, len(df)):
        row = df.iloc[i]

        # ── Check open trade exit first ──
        if open_trade is not None:
            high, low = row['high'], row['low']
            hit_tp, hit_sl = False, False
            if open_trade.direction == 'BUY':
                if low <= open_trade.stop_loss:
                    hit_sl = True
                elif high >= open_trade.take_profit:
                    hit_tp = True
            else:  # SELL
                if high >= open_trade.stop_loss:
                    hit_sl = True
                elif low <= open_trade.take_profit:
                    hit_tp = True

            if hit_sl:
                open_trade.exit_time = row['datetime_utc']
                open_trade.exit_price = open_trade.stop_loss
                open_trade.exit_reason = 'SL'
            elif hit_tp:
                open_trade.exit_time = row['datetime_utc']
                open_trade.exit_price = open_trade.take_profit
                open_trade.exit_reason = 'TP'
            else:
                open_trade.hold_bars += 1
                if open_trade.hold_bars >= max_hold_bars:
                    open_trade.exit_time = row['datetime_utc']
                    open_trade.exit_price = row['close']
                    open_trade.exit_reason = 'timeout'

            if open_trade.exit_reason:
                # Calculate PnL
                entry, exit_p = open_trade.entry_price, open_trade.exit_price
                if open_trade.direction == 'BUY':
                    pnl_pips = (exit_p - entry) / pip_value
                else:
                    pnl_pips = (entry - exit_p) / pip_value
                # subtract spread + slippage
                pnl_pips_adj = pnl_pips - spread_pips - slippage_pips
                pnl_usd = pnl_pips_adj * open_trade.lot_size * 100  # $1/pip/0.01 lot
                pnl_usd -= open_trade.commission_usd
                open_trade.pnl_pips = pnl_pips_adj
                open_trade.pnl_usd = pnl_usd
                balance += pnl_usd
                trades.append(open_trade)
                open_trade = None

        # ── Open new trade if no open position ──
        if open_trade is None:
            sig = signal_fn(row)
            if sig['signal'] in ('BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL') and sig['confidence'] >= min_confidence:
                atr_val = row['atr']
                if atr_val <= 0 or pd.isna(atr_val):
                    continue
                entry_price = row['close']

                # Apply slippage on entry
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
                lot_size = risk_usd / (sl_pips * 100)
                lot_size = max(0.01, min(1.0, round(lot_size, 2)))

                # Spread cost on entry
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
        pnl_usd = pnl_pips_adj * open_trade.lot_size * 100 - open_trade.commission_usd
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

    # Direction stats
    buy_trades = [t for t in trades if t.direction == 'BUY']
    sell_trades = [t for t in trades if t.direction == 'SELL']
    buy_wins = sum(1 for t in buy_trades if t.pnl_usd > 0)
    sell_wins = sum(1 for t in sell_trades if t.pnl_usd > 0)

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
    }

    return {'trades': trades, 'metrics': metrics, 'final_balance': balance}


# ═════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═════════════════════════════════════════════════════════════
def prepare_data(csv_path: Path, symbol: str) -> pd.DataFrame:
    """Load CSV, compute indicators."""
    df = pd.read_csv(csv_path)

    # Normalize column names
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'datetime_utc'})

    # Determine pip value
    pip_value = 0.01 if 'JPY' in symbol else 0.0001

    # Compute indicators
    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['ema200'] = ema(df['close'], 200)
    df['rsi'] = rsi(df['close'], 14)
    macd_line, signal_line = macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_cross'] = 'none'
    df.loc[(macd_line > signal_line) & (macd_line.shift() <= signal_line.shift()), 'macd_cross'] = 'bullish'
    df.loc[(macd_line < signal_line) & (macd_line.shift() >= signal_line.shift()), 'macd_cross'] = 'bearish'
    df['atr'] = atr(df, 14)
    df['adx'] = adx(df, 14)
    df['vwap'] = vwap(df, 20)
    df['bos'] = detect_bos(df, 20)
    df['pip_value'] = pip_value

    return df


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
            print(f"[SKIP] {csv_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Running BASELINE backtest for {symbol} H1")
        print(f"{'='*60}")

        df = prepare_data(csv_path, symbol)
        print(f"  Loaded {len(df)} bars | {df.iloc[0]['datetime_utc']} → {df.iloc[-1]['datetime_utc']}")

        result = run_backtest(
            df=df,
            signal_fn=original_signal,
            symbol=symbol,
            strategy_name='original_v1',
        )

        m = result['metrics']
        if not m:
            print("  No trades generated.")
            continue

        print(f"  Trades: {m['total_trades']:4d} | Wins: {m['wins']:4d} | WR: {m['win_rate']:5.2f}%")
        print(f"  PF: {m['profit_factor']:5.2f} | PnL: ${m['total_pnl_usd']:>10.2f} | Final Bal: ${m['final_balance']:>10.2f}")
        print(f"  BUY:  {m['buy_trades']:3d} trades, WR={m['buy_win_rate']:5.2f}%")
        print(f"  SELL: {m['sell_trades']:3d} trades, WR={m['sell_win_rate']:5.2f}%")
        print(f"  Avg Win: ${m['avg_win_usd']:>8.2f} | Avg Loss: ${m['avg_loss_usd']:>8.2f}")
        print(f"  Expectancy: ${m['expectancy_usd']:>8.2f}/trade | Avg Hold: {m['avg_hold_bars']} bars")

        all_metrics.append({'symbol': symbol, **m})
        all_trades.extend([asdict(t) for t in result['trades']])

    # ── Save combined results ──
    metrics_df = pd.DataFrame(all_metrics)
    trades_df  = pd.DataFrame(all_trades)

    metrics_csv = OUTPUT_DIR / "baseline_metrics.csv"
    trades_csv  = OUTPUT_DIR / "baseline_trades.csv"

    metrics_df.to_csv(metrics_csv, index=False)
    trades_df.to_csv(trades_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  COMBINED BASELINE RESULTS  (7 pairs)")
    print(f"{'='*60}")
    if all_metrics:
        total_trades = sum(m['total_trades'] for m in all_metrics)
        total_wins   = sum(m['wins']      for m in all_metrics)
        total_pnl    = sum(m['total_pnl_usd'] for m in all_metrics)
        print(f"  Total trades across all pairs: {total_trades}")
        print(f"  Combined win rate: {total_wins/total_trades*100:.2f}%")
        print(f"  Total PnL: ${total_pnl:.2f}")

    print(f"\n  Saved metrics → {metrics_csv}")
    print(f"  Saved trades  → {trades_csv}")


if __name__ == "__main__":
    main()
