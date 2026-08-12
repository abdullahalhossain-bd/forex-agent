#!/usr/bin/env python3
"""
Production-Level Backtest with Full Pipeline
=============================================
Uses ACTUAL production modules:
  - data/indicators.py (Indicators class — EMA50/200, ADX, Stochastic)
  - strategy/signal_engine.py (SignalEngine with HTF gate)
  - Pattern detection (engulfing, hammer, shooting star, strong candles)
  - Liquidity sweep detection
  - BOS (Break of Structure) detection
  - ATR-based SL/TP with swing refinement
  - Breakeven + profit lock
  - TradePermission min_confidence=70

This simulates the full production decision pipeline.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

PROJECT = Path("/home/z/my-project/download/forex-agent")
sys.path.insert(0, str(PROJECT))

# Config shim
import types
config = types.ModuleType('config')
config.MAX_LOT = 0.2
config.APPROVAL_MODE = 3
config.MAX_OPEN_TRADES = 20
config.EXECUTION_MODE = 'paper'
config.INITIAL_BALANCE = 10000.0
config.SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD', 'NZDUSD']
config.DEFAULT_TIMEFRAME = 'H1'
config.ENABLE_TELEGRAM = False
config.DAILY_LOSS_LIMIT_PCT = 3.0
config.WEEKLY_LOSS_LIMIT_PCT = 7.0
config.MAX_DRAWDOWN_PCT = 15.0
config.MIN_RR_PROD = 2.0
config.MIN_RR_TEST = 1.0
config.TEST_MODE = False
config.SIMULATION_MODE = True
config.USE_SCANNER = False
config.LOOP_INTERVAL_SEC = 180
config.BACKUP_INTERVAL_MIN = 30
config.RECOVERY_COOLDOWN_MIN = 5
config.MAX_LLM_CALLS_PER_CYCLE = 8
sys.modules['config'] = config

from data.indicators import Indicators
from strategy.signal_engine import SignalEngine

OUTPUT = Path("/home/z/my-project/download")


def get_pip_value(symbol):
    return 0.01 if 'JPY' in symbol else 0.0001


def get_pip_usd(symbol, price):
    if symbol.endswith('USD'):
        return 10.0
    elif symbol.startswith('USD'):
        return 1000.0 / price if 'JPY' in symbol else 10.0
    else:
        return 10.0 / price * 100 if 'JPY' in symbol else 10.0


def detect_pattern(df, i):
    """Enhanced candlestick pattern detection."""
    if i < 3:
        return {'pattern_signal': '', 'latest_pattern': 'none'}
    row = df.iloc[i]
    prev = df.iloc[i-1]
    prev2 = df.iloc[i-2]
    body = abs(row['close'] - row['open'])
    rng = row['high'] - row['low']
    if rng == 0:
        return {'pattern_signal': '', 'latest_pattern': 'none'}
    bp = body / rng
    upper_wick = row['high'] - max(row['open'], row['close'])
    lower_wick = min(row['open'], row['close']) - row['low']

    # Bullish engulfing
    if (prev['close'] < prev['open'] and row['close'] > row['open']
        and row['close'] >= prev['open'] and row['open'] <= prev['close'] and bp > 0.6):
        return {'pattern_signal': 'Bullish', 'latest_pattern': 'engulfing_bull'}
    # Bearish engulfing
    if (prev['close'] > prev['open'] and row['close'] < row['open']
        and row['close'] <= prev['open'] and row['open'] >= prev['close'] and bp > 0.6):
        return {'pattern_signal': 'Bearish', 'latest_pattern': 'engulfing_bear'}
    # Hammer
    if bp < 0.3 and lower_wick > 2 * body:
        return {'pattern_signal': 'Bullish', 'latest_pattern': 'hammer'}
    # Shooting star
    if bp < 0.3 and upper_wick > 2 * body:
        return {'pattern_signal': 'Bearish', 'latest_pattern': 'shooting_star'}
    # Strong bullish candle
    if bp > 0.7 and row['close'] > row['open']:
        return {'pattern_signal': 'Bullish', 'latest_pattern': 'strong_bull'}
    # Strong bearish candle
    if bp > 0.7 and row['close'] < row['open']:
        return {'pattern_signal': 'Bearish', 'latest_pattern': 'strong_bear'}
    # Doji
    if bp < 0.1:
        return {'pattern_signal': '', 'latest_pattern': 'doji'}
    # Morning star (3-candle bullish reversal)
    if (prev2['close'] < prev2['open'] and prev['close'] < prev['open']
        and row['close'] > row['open'] and row['close'] > prev2['open']):
        return {'pattern_signal': 'Bullish', 'latest_pattern': 'morning_star'}
    # Evening star (3-candle bearish reversal)
    if (prev2['close'] > prev2['open'] and prev['close'] > prev['open']
        and row['close'] < row['open'] and row['close'] < prev2['open']):
        return {'pattern_signal': 'Bearish', 'latest_pattern': 'evening_star'}
    return {'pattern_signal': '', 'latest_pattern': 'none'}


def detect_liquidity_sweep(df, i, lookback=20):
    """Detect liquidity sweep — wick beyond recent swing then close back."""
    if i < lookback + 1:
        return 0
    rh = df.iloc[i-lookback:i]['high'].max()
    rl = df.iloc[i-lookback:i]['low'].min()
    row = df.iloc[i]
    if row['high'] > rh and row['close'] < rh:
        return -1
    if row['low'] < rl and row['close'] > rl:
        return 1
    return 0


def detect_bos(df, i, lookback=20):
    """Break of structure."""
    if i < lookback + 1:
        return 0
    rh = df.iloc[i-lookback:i]['high'].max()
    rl = df.iloc[i-lookback:i]['low'].min()
    row = df.iloc[i]
    if row['close'] > rh:
        return 1
    if row['close'] < rl:
        return -1
    return 0


def find_swing_stop(df, i, direction, lookback=10):
    """Find recent swing low (BUY) or swing high (SELL) for SL placement."""
    if i < lookback:
        return None
    start = max(0, i - lookback)
    window = df.iloc[start:i]
    if direction == 'BUY':
        return window['low'].min()
    else:
        return window['high'].max()


def run_production_backtest(symbol, min_confidence=70):
    """Run backtest using production SignalEngine + full context."""
    csv_path = PROJECT / 'data' / 'history' / symbol / f'{symbol}_H1.csv'
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'datetime_utc'})

    ind = Indicators()
    df = ind.add_all(df)

    se = SignalEngine()
    pip_value = get_pip_value(symbol)
    balance = 10000.0
    initial_balance = balance
    trades = []
    open_trade = None
    open_meta = {}
    warmup = 250
    last_close_bar = -999
    signal_counts = {'STRONG_BUY': 0, 'BUY': 0, 'STRONG_SELL': 0, 'SELL': 0, 'WAIT': 0}
    rejection_reasons = {}

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        high, low, close = row['high'], row['low'], row['close']

        # Check exit
        if open_trade is not None:
            meta = open_meta
            entry = open_trade['entry_price']
            orig_sl = meta.get('orig_sl', open_trade['stop_loss'])
            risk_dist = abs(entry - orig_sl)

            # BE at 0.7R
            if not meta.get('be_moved') and risk_dist > 0:
                if open_trade['direction'] == 'BUY':
                    if high >= entry + risk_dist * 0.7:
                        open_trade['stop_loss'] = entry + 3.5 * pip_value
                        meta['be_moved'] = True
                else:
                    if low <= entry - risk_dist * 0.7:
                        open_trade['stop_loss'] = entry - 3.5 * pip_value
                        meta['be_moved'] = True

            # Profit lock at 1.2R
            if meta.get('be_moved') and not meta.get('profit_locked') and risk_dist > 0:
                if open_trade['direction'] == 'BUY':
                    if high >= entry + risk_dist * 1.2:
                        open_trade['stop_loss'] = entry + risk_dist * 0.5
                        meta['profit_locked'] = True
                else:
                    if low <= entry - risk_dist * 1.2:
                        open_trade['stop_loss'] = entry - risk_dist * 0.5
                        meta['profit_locked'] = True

            # Check TP/SL
            hit_tp, hit_sl = False, False
            if open_trade['direction'] == 'BUY':
                if low <= open_trade['stop_loss']:
                    hit_sl = True
                elif high >= open_trade['take_profit']:
                    hit_tp = True
            else:
                if high >= open_trade['stop_loss']:
                    hit_sl = True
                elif low <= open_trade['take_profit']:
                    hit_tp = True

            if hit_sl:
                exit_price = open_trade['stop_loss']
                exit_reason = 'SL_PROFIT' if meta.get('profit_locked') else ('SL_BE' if meta.get('be_moved') else 'SL')
            elif hit_tp:
                exit_price = open_trade['take_profit']
                exit_reason = 'TP'
            else:
                open_trade['hold_bars'] += 1
                if open_trade['hold_bars'] >= 40:
                    exit_price = close
                    exit_reason = 'timeout'
                else:
                    exit_price = None

            if exit_price is not None:
                entry_p = open_trade['entry_price']
                if open_trade['direction'] == 'BUY':
                    pnl_pips = (exit_price - entry_p) / pip_value
                else:
                    pnl_pips = (entry_p - exit_price) / pip_value
                pnl_pips_adj = pnl_pips - 3.5
                pip_usd = get_pip_usd(symbol, entry_p)
                pnl_usd = pnl_pips_adj * open_trade['lot_size'] * pip_usd
                pnl_usd -= 7.0 * open_trade['lot_size']

                trades.append({
                    'symbol': symbol,
                    'direction': open_trade['direction'],
                    'entry_price': entry_p,
                    'exit_price': exit_price,
                    'confidence': open_trade['confidence'],
                    'exit_reason': exit_reason,
                    'pnl_pips': pnl_pips_adj,
                    'pnl_usd': pnl_usd,
                    'hold_bars': open_trade['hold_bars'],
                    'strategy': open_trade.get('strategy', 'signal_engine'),
                })
                balance += pnl_usd
                last_close_bar = i
                open_trade = None
                open_meta = {}

        # Open new trade
        if open_trade is None:
            if i - last_close_bar < 3:  # cooldown 3 bars
                continue

            ind_ctx = ind.get_ai_context(df.iloc[:i+1])
            pat_ctx = detect_pattern(df, i)
            sr_ctx = {'price_location': ''}
            adx_val = ind_ctx.get('adx', 0)
            trend = ind_ctx.get('trend', '')

            # Build regime
            if 'bullish' in trend and adx_val > 25:
                regime = {'strategy_type': 'TREND', 'market_direction': 'BULLISH'}
            elif 'bearish' in trend and adx_val > 25:
                regime = {'strategy_type': 'TREND', 'market_direction': 'BEARISH'}
            else:
                regime = {'strategy_type': 'WAIT', 'market_direction': 'NEUTRAL'}

            mtf_bias = {
                'bias': 'BULLISH' if 'bullish' in trend else ('BEARISH' if 'bearish' in trend else 'NEUTRAL'),
                'confidence': 'HIGH' if adx_val > 30 else 'LOW',
            }

            sweep = detect_liquidity_sweep(df, i)
            bos = detect_bos(df, i)
            advanced_pat_ctx = {
                'has_pattern': bos != 0 or sweep != 0,
                'pattern_direction': 'BULLISH' if (bos == 1 or sweep == 1) else ('BEARISH' if (bos == -1 or sweep == -1) else 'NEUTRAL'),
                'pattern_confidence': 80 if (bos != 0 and sweep != 0) else 65,
                'advanced_pattern': 'BOS+Sweep' if bos != 0 and sweep != 0 else ('BOS' if bos != 0 else 'Sweep'),
            }

            try:
                result = se.generate(
                    ind_ctx=ind_ctx,
                    pat_ctx=pat_ctx,
                    sr_ctx=sr_ctx,
                    regime=regime,
                    mtf_bias=mtf_bias,
                    advanced_pat_ctx=advanced_pat_ctx,
                )
            except:
                continue

            sig = result['signal']
            conf = result['confidence']
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

            if sig not in ('BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL'):
                rejection_reasons['signal_wait'] = rejection_reasons.get('signal_wait', 0) + 1
                continue
            if conf < min_confidence:
                rejection_reasons['low_confidence'] = rejection_reasons.get('low_confidence', 0) + 1
                continue

            atr_val = ind_ctx.get('atr', 0)
            if not atr_val or atr_val <= 0:
                continue

            if sig in ('BUY', 'STRONG_BUY'):
                direction = 'BUY'
                entry = close + 2 * pip_value
                # Swing-based SL with ATR cap
                swing_low = find_swing_stop(df, i, 'BUY', lookback=10)
                if swing_low and swing_low < entry:
                    sl = swing_low - 0.2 * atr_val
                    if entry - sl > 1.5 * atr_val:
                        sl = entry - 1.5 * atr_val
                    if entry - sl < 0.5 * atr_val:
                        sl = entry - 0.5 * atr_val
                else:
                    sl = entry - 1.5 * atr_val
                tp = entry + 3.0 * atr_val  # 1:2 R:R
            else:
                direction = 'SELL'
                entry = close - 2 * pip_value
                swing_high = find_swing_stop(df, i, 'SELL', lookback=10)
                if swing_high and swing_high > entry:
                    sl = swing_high + 0.2 * atr_val
                    if sl - entry > 1.5 * atr_val:
                        sl = entry + 1.5 * atr_val
                    if sl - entry < 0.5 * atr_val:
                        sl = entry + 0.5 * atr_val
                else:
                    sl = entry + 1.5 * atr_val
                tp = entry - 3.0 * atr_val

            sl_pips = abs(entry - sl) / pip_value
            if sl_pips <= 0:
                continue
            risk_usd = balance * 0.01
            pip_usd = get_pip_usd(symbol, entry)
            conf_mult = 0.7 + (conf / 100) * 0.5
            lot = (risk_usd / (sl_pips * pip_usd)) * conf_mult
            lot = max(0.01, min(1.0, round(lot, 2)))

            open_trade = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry,
                'stop_loss': sl,
                'take_profit': tp,
                'lot_size': lot,
                'confidence': conf,
                'hold_bars': 0,
                'strategy': 'signal_engine_v3',
            }
            open_meta = {'be_moved': False, 'profit_locked': False, 'orig_sl': sl}

    # Close remaining
    if open_trade is not None:
        last = df.iloc[-1]
        entry_p = open_trade['entry_price']
        exit_p = last['close']
        if open_trade['direction'] == 'BUY':
            pnl_pips = (exit_p - entry_p) / pip_value
        else:
            pnl_pips = (entry_p - exit_p) / pip_value
        pnl_pips_adj = pnl_pips - 3.5
        pip_usd = get_pip_usd(symbol, entry_p)
        pnl_usd = pnl_pips_adj * open_trade['lot_size'] * pip_usd - 7.0 * open_trade['lot_size']
        trades.append({
            'symbol': symbol,
            'direction': open_trade['direction'],
            'entry_price': entry_p,
            'exit_price': exit_p,
            'confidence': open_trade['confidence'],
            'exit_reason': 'end_of_backtest',
            'pnl_pips': pnl_pips_adj,
            'pnl_usd': pnl_usd,
            'hold_bars': open_trade['hold_bars'],
            'strategy': open_trade.get('strategy', 'signal_engine'),
        })
        balance += pnl_usd

    return {
        'trades': trades,
        'final_balance': balance,
        'initial_balance': initial_balance,
        'signal_counts': signal_counts,
        'rejection_reasons': rejection_reasons,
        'total_bars': len(df),
    }


def main():
    print("=" * 80)
    print("  PRODUCTION-LEVEL BACKTEST (Full Pipeline)")
    print("  V3 Code + Pattern Detection + SMC + Swing Stops")
    print("=" * 80)

    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD', 'NZDUSD']
    all_results = []
    all_trades = []

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  {symbol} H1")
        print(f"{'='*60}")

        result = run_production_backtest(symbol, min_confidence=70)
        if result is None:
            print(f"  No data")
            continue

        trades = result['trades']
        if not trades:
            print(f"  No trades generated")
            print(f"  Signals: {result['signal_counts']}")
            print(f"  Rejections: {result['rejection_reasons']}")
            continue

        df = pd.DataFrame(trades)
        wins = (df['pnl_usd'] > 0).sum()
        losses = (df['pnl_usd'] <= 0).sum()
        total = len(df)
        wr = wins / total * 100 if total > 0 else 0

        gross_profit = df[df['pnl_usd'] > 0]['pnl_usd'].sum()
        gross_loss = abs(df[df['pnl_usd'] < 0]['pnl_usd'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        total_pnl = df['pnl_usd'].sum()
        final_balance = result['final_balance']
        return_pct = (final_balance / result['initial_balance'] - 1) * 100

        buy_trades = df[df['direction'] == 'BUY']
        sell_trades = df[df['direction'] == 'SELL']
        buy_wins = (buy_trades['pnl_usd'] > 0).sum() if len(buy_trades) > 0 else 0
        sell_wins = (sell_trades['pnl_usd'] > 0).sum() if len(sell_trades) > 0 else 0

        print(f"  Total bars: {result['total_bars']}")
        print(f"  Signals: {result['signal_counts']}")
        print(f"  Total trades: {total}")
        print(f"  *** WINRATE: {wr:.2f}% ***")
        print(f"  *** PROFIT: ${total_pnl:.2f} ***")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Wins: {wins}, Losses: {losses}")
        print(f"  Final Balance: ${final_balance:.2f}")
        print(f"  Return: {return_pct:+.2f}%")

        if len(buy_trades) > 0:
            print(f"  BUY:  {len(buy_trades):3d} trades, WR={buy_wins/len(buy_trades)*100:.1f}%, PnL=${buy_trades['pnl_usd'].sum():.2f}")
        if len(sell_trades) > 0:
            print(f"  SELL: {len(sell_trades):3d} trades, WR={sell_wins/len(sell_trades)*100:.1f}%, PnL=${sell_trades['pnl_usd'].sum():.2f}")

        # Exit reasons
        exit_reasons = df['exit_reason'].value_counts()
        print(f"\n  Exit Reasons:")
        for r, c in exit_reasons.items():
            sub = df[df['exit_reason'] == r]
            rw = (sub['pnl_usd'] > 0).sum()
            print(f"    {r:15s}: {c:3d} trades, WR={rw/c*100:.1f}%, PnL=${sub['pnl_usd'].sum():.2f}")

        all_results.append({
            'symbol': symbol,
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': round(wr, 2),
            'profit_factor': round(pf, 2),
            'total_pnl_usd': round(total_pnl, 2),
            'final_balance': round(final_balance, 2),
            'return_pct': round(return_pct, 2),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'buy_win_rate': round(buy_wins/len(buy_trades)*100, 2) if len(buy_trades) > 0 else 0,
            'sell_win_rate': round(sell_wins/len(sell_trades)*100, 2) if len(sell_trades) > 0 else 0,
        })
        all_trades.extend(trades)

    # Summary
    print(f"\n\n{'='*80}")
    print(f"  COMBINED SUMMARY (7 pairs)")
    print(f"{'='*80}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        total_trades = results_df['total_trades'].sum()
        total_wins = results_df['wins'].sum()
        total_pnl = results_df['total_pnl_usd'].sum()
        combined_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

        print(f"\n  Total trades: {total_trades}")
        print(f"  Total wins: {total_wins}")
        print(f"  *** COMBINED WINRATE: {combined_wr:.2f}% ***")
        print(f"  *** TOTAL PROFIT: ${total_pnl:.2f} ***")
        print(f"  Avg Profit Factor: {results_df['profit_factor'].mean():.2f}")

        print(f"\n  Per-Pair Results:")
        print(f"  {'Symbol':<10} {'Trades':>7} {'WR':>7} {'PF':>6} {'PnL':>10} {'Return':>8} {'BUY WR':>7} {'SELL WR':>8}")
        print(f"  {'-'*70}")
        for _, row in results_df.iterrows():
            print(f"  {row['symbol']:<10} {row['total_trades']:>7} {row['win_rate']:>6.1f}% {row['profit_factor']:>6.2f} ${row['total_pnl_usd']:>9.2f} {row['return_pct']:>+7.1f}% {row['buy_win_rate']:>6.1f}% {row['sell_win_rate']:>7.1f}%")

        # Save
        results_df.to_csv(OUTPUT / 'production_winrate_results.csv', index=False)
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(OUTPUT / 'production_winrate_trades.csv', index=False)
        print(f"\n  Results saved to:")
        print(f"    {OUTPUT / 'production_winrate_results.csv'}")
        print(f"    {OUTPUT / 'production_winrate_trades.csv'}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
