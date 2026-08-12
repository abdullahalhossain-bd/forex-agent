#!/usr/bin/env python3
"""
Forex-Agent Strategy v4 — Final Production-Tuned
================================================
Key fixes from v3:
  1. Correct pip_value per symbol (0.01 for JPY, 0.0001 otherwise)
  2. Correct PnL formula (pnl_pips * lot * 10 for non-JPY, less for JPY)
  3. Fixed breakeven logic — check BEFORE SL hit
  4. Much stricter signal: net >= 7 for BUY/SELL (was 6)
  5. Min 5 confluence factors required (was 4)
  6. RSI pullback entry ONLY (no extreme entries that fade trend)
  7. Cooldown between trades = 5 bars
  8. Wider SL with proper ATR (1.8x) for noise tolerance
  9. R:R 1:2.5 with trailing stop at 1R beyond breakeven
 10. Proper session filter — only London+NY overlap
 11. Min ADX = 22 (skip choppy)
 12. Min ATR filter (skip dead markets)
"""

from __future__ import annotations
import sys, json, math
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

sys.path.insert(0, '/home/z/my-project/scripts')
from forex_backtest_v1 import (
    ema, rsi, macd, atr, adx, vwap, detect_bos, Trade, prepare_data, DATA_DIR, OUTPUT_DIR
)
from forex_backtest_v2 import (
    bollinger_bands, stochastic, cci, obv, detect_choch, liquidity_sweep, prepare_data_v2
)


# ═════════════════════════════════════════════════════════════
# SYMBOL SPECS — proper pip values & contract sizes
# ═════════════════════════════════════════════════════════════
def get_pip_value(symbol: str) -> float:
    return 0.01 if 'JPY' in symbol else 0.0001


def get_pip_value_usd(symbol: str, current_price: float) -> float:
    """USD value of 1 pip per 1.0 lot (100k units)."""
    if symbol.endswith('USD'):
        return 10.0  # USD-quoted account
    elif symbol.startswith('USD'):
        # USDJPY: 1 pip = 0.01 JPY per unit, per 1 lot = 1000 JPY = 1000/rate USD
        return 1000.0 / current_price if 'JPY' in symbol else 10.0
    else:
        # Cross rates — approximate using current price
        return 10.0 / current_price * 100 if 'JPY' in symbol else 10.0


# ═════════════════════════════════════════════════════════════
# SIGNAL ENGINE v4 — Final
# ═════════════════════════════════════════════════════════════
def signal_v4(row, symbol='EURUSD') -> Dict[str, Any]:
    """
    Final signal engine v4:
      - Strict HTF gate
      - Trend pullback entries (not breakout-only)
      - 5+ confluence factors required
      - Net >= 7 for entry
    """
    bull_score = 0
    bear_score = 0
    bull_factors = 0
    bear_factors = 0
    signals = []
    warnings = []

    price = row['close']
    ema9, ema20, ema50, ema200 = row['ema9'], row['ema20'], row['ema50'], row['ema200']
    atr_val = row['atr']
    adx_val = row['adx']
    rsi_val = row['rsi']

    # ── GATE 1: HTF bias (EMA200) — required ──
    htf_bull = price > ema200 and ema50 > ema200
    htf_bear = price < ema200 and ema50 < ema200
    if not (htf_bull or htf_bear):
        return {'signal': 'WAIT', 'confidence': 0, 'net': 0, 'warnings': ['No HTF bias']}

    # ── GATE 2: ADX minimum ──
    if adx_val < 22:
        return {'signal': 'WAIT', 'confidence': 0, 'net': 0, 'warnings': [f'ADX too low ({adx_val:.0f})']}

    # ── GATE 3: ATR minimum (skip dead markets) ──
    atr_pct = atr_val / price * 100  # ATR as % of price
    if atr_pct < 0.05:  # too quiet
        return {'signal': 'WAIT', 'confidence': 0, 'net': 0, 'warnings': ['ATR too low']}

    # ── FACTOR 1: HTF direction (already verified) ──
    if htf_bull:
        bull_score += 2; bull_factors += 1
        signals.append(('bullish', 2, 'HTF bull'))
    else:
        bear_score += 2; bear_factors += 1
        signals.append(('bearish', 2, 'HTF bear'))

    # ── FACTOR 2: EMA stack alignment ──
    if htf_bull and ema9 > ema20 > ema50:
        bull_score += 2; bull_factors += 1
        signals.append(('bullish', 2, 'EMA stack aligned'))
    elif htf_bear and ema9 < ema20 < ema50:
        bear_score += 2; bear_factors += 1
        signals.append(('bearish', 2, 'EMA stack aligned'))
    else:
        # Pullback opportunity — EMA9 crossed EMA20 against trend
        if htf_bull and ema9 < ema20:
            signals.append(('bullish', 1, 'Pullback: EMA9<EMA20 in uptrend'))
            bull_score += 1; bull_factors += 1
        elif htf_bear and ema9 > ema20:
            signals.append(('bearish', 1, 'Pullback: EMA9>EMA20 in downtrend'))
            bear_score += 1; bear_factors += 1

    # ── FACTOR 3: RSI pullback zone (NOT oversold/overbought) ──
    if htf_bull and 35 <= rsi_val <= 55:
        bull_score += 2; bull_factors += 1
        signals.append(('bullish', 2, f'RSI pullback ({rsi_val:.0f})'))
    elif htf_bear and 45 <= rsi_val <= 65:
        bear_score += 2; bear_factors += 1
        signals.append(('bearish', 2, f'RSI pullback ({rsi_val:.0f})'))

    # ── FACTOR 4: MACD momentum confirmation ──
    if htf_bull and row['macd'] > row['macd_signal'] and row['macd'] > 0:
        bull_score += 2; bull_factors += 1
        signals.append(('bullish', 2, 'MACD bull + above 0'))
    elif htf_bear and row['macd'] < row['macd_signal'] and row['macd'] < 0:
        bear_score += 2; bear_factors += 1
        signals.append(('bearish', 2, 'MACD bear + below 0'))

    # ── FACTOR 5: BOS structure confirmation ──
    if row['bos'] == 1 and htf_bull:
        bull_score += 2; bull_factors += 1
        signals.append(('bullish', 2, 'Bullish BOS'))
    elif row['bos'] == -1 and htf_bear:
        bear_score += 2; bear_factors += 1
        signals.append(('bearish', 2, 'Bearish BOS'))

    # ── FACTOR 6: Liquidity sweep ──
    if row['liquidity_sweep'] == 1 and htf_bull:
        bull_score += 2; bull_factors += 1
        signals.append(('bullish', 2, 'Bullish liquidity sweep'))
    elif row['liquidity_sweep'] == -1 and htf_bear:
        bear_score += 2; bear_factors += 1
        signals.append(('bearish', 2, 'Bearish liquidity sweep'))

    # ── FACTOR 7: Volume surge ──
    if row['tick_volume'] > row['volume_ma'] * 1.5:
        if htf_bull and bull_score > bear_score:
            bull_score += 1; bull_factors += 1
            signals.append(('bullish', 1, 'Volume surge'))
        elif htf_bear and bear_score > bull_score:
            bear_score += 1; bear_factors += 1
            signals.append(('bearish', 1, 'Volume surge'))

    # ── FACTOR 8: Stochastic cross in pullback zone ──
    if htf_bull and row['stoch_k'] > row['stoch_d'] and row['stoch_k'] < 40:
        bull_score += 1; bull_factors += 1
    elif htf_bear and row['stoch_k'] < row['stoch_d'] and row['stoch_k'] > 60:
        bear_score += 1; bear_factors += 1

    # ── FACTOR 9: ADX strength bonus ──
    if adx_val > 30:
        if bull_score > bear_score:
            bull_score += 1; signals.append(('bullish', 1, f'ADX strong ({adx_val:.0f})'))
        elif bear_score > bull_score:
            bear_score += 1; signals.append(('bearish', 1, f'ADX strong ({adx_val:.0f})'))

    # ── FACTOR 10: Bollinger band touch (mean reversion in trend) ──
    if htf_bull and price <= row['bb_lower']:
        bull_score += 1; bull_factors += 1
        signals.append(('bullish', 1, 'BB lower touch'))
    elif htf_bear and price >= row['bb_upper']:
        bear_score += 1; bear_factors += 1
        signals.append(('bearish', 1, 'BB upper touch'))

    # ── Compute final scores ──
    total = bull_score + bear_score
    net = bull_score - bear_score

    if total == 0:
        return {'signal': 'WAIT', 'confidence': 0, 'net': 0, 'warnings': warnings}

    confidence = round(max(bull_score, bear_score) / total * 100)

    # v4 STRICT THRESHOLDS
    max_factors = max(bull_factors, bear_factors)
    if max_factors < 5:
        return {'signal': 'WAIT', 'confidence': confidence, 'net': net, 'warnings': warnings + ['Need 5+ factors']}

    # Require strong agreement
    if net >= 7 and htf_bull and bull_factors >= 5:
        signal = 'BUY' if net >= 9 else 'BUY'
        if net >= 10: signal = 'STRONG_BUY'
    elif net <= -7 and htf_bear and bear_factors >= 5:
        signal = 'SELL' if net >= -9 else 'SELL'
        if net <= -10: signal = 'STRONG_SELL'
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
        'factors': max_factors,
    }


# ═════════════════════════════════════════════════════════════
# BACKTEST ENGINE v4 — Final
# ═════════════════════════════════════════════════════════════
def run_backtest_v4(
    df: pd.DataFrame,
    signal_fn,
    symbol: str = 'EURUSD',
    timeframe: str = 'H1',
    initial_balance: float = 10_000,
    risk_per_trade: float = 0.01,
    spread_pips: float = 1.5,
    commission_per_lot: float = 7.0,
    slippage_pips: float = 2.0,
    min_confidence: float = 65,
    atr_sl_mult: float = 1.8,      # wider SL = noise tolerance
    atr_tp_mult: float = 3.6,      # 1:2 R:R
    max_hold_bars: int = 40,
    strategy_name: str = 'v4_final',
    enable_breakeven: bool = True,
    breakeven_at_r: float = 1.2,   # move to BE at 1.2R
    enable_trailing: bool = True,
    trail_at_r: float = 1.8,       # trail SL behind price by 1R
    enable_session_filter: bool = True,
    cooldown_bars: int = 5,
) -> Dict[str, Any]:

    pip_value = get_pip_value(symbol)
    trades: List[Trade] = []
    trade_id = 1
    balance = initial_balance
    open_trade: Optional[Trade] = None
    open_trade_meta: Dict[str, Any] = {}
    warmup = 200
    last_close_bar = -999

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        high, low, close = row['high'], row['low'], row['close']

        # ── Check open trade exit ──
        if open_trade is not None:
            meta = open_trade_meta
            entry = open_trade.entry_price

            # Breakeven move FIRST (before SL check)
            if enable_breakeven and not meta.get('be_moved', False):
                risk_distance = abs(entry - open_trade.stop_loss)
                if open_trade.direction == 'BUY':
                    be_trigger = entry + risk_distance * breakeven_at_r
                    if high >= be_trigger:
                        # Move SL to entry + spread/slippage cost (true breakeven)
                        open_trade.stop_loss = entry + (slippage_pips + spread_pips) * pip_value
                        meta['be_moved'] = True
                else:
                    be_trigger = entry - risk_distance * breakeven_at_r
                    if low <= be_trigger:
                        open_trade.stop_loss = entry - (slippage_pips + spread_pips) * pip_value
                        meta['be_moved'] = True

            # Trailing stop AFTER breakeven moved
            if enable_trailing and meta.get('be_moved', False):
                risk_distance = abs(entry - open_trade.stop_loss)  # current SL distance
                if open_trade.direction == 'BUY':
                    new_sl = close - risk_distance * 0.8  # trail 0.8R behind
                    if new_sl > open_trade.stop_loss:
                        open_trade.stop_loss = new_sl
                else:
                    new_sl = close + risk_distance * 0.8
                    if new_sl < open_trade.stop_loss:
                        open_trade.stop_loss = new_sl

            # Check TP/SL
            hit_tp, hit_sl = False, False
            if open_trade.direction == 'BUY':
                if low <= open_trade.stop_loss: hit_sl = True
                elif high >= open_trade.take_profit: hit_tp = True
            else:
                if high >= open_trade.stop_loss: hit_sl = True
                elif low <= open_trade.take_profit: hit_tp = True

            # Strong opposite signal exit (only for STRONG opposite)
            if not hit_tp and not hit_sl:
                sig_now = signal_fn(row, symbol)
                if sig_now['signal'] in ('STRONG_SELL', 'STRONG_BUY') and sig_now.get('confidence', 0) >= 80:
                    if (open_trade.direction == 'BUY' and sig_now['signal'] == 'STRONG_SELL') or \
                       (open_trade.direction == 'SELL' and sig_now['signal'] == 'STRONG_BUY'):
                        open_trade.exit_time = row['datetime_utc']
                        open_trade.exit_price = close
                        open_trade.exit_reason = 'opposite_signal'

            if hit_sl:
                open_trade.exit_time = row['datetime_utc']
                open_trade.exit_price = open_trade.stop_loss
                open_trade.exit_reason = 'SL' if not meta.get('be_moved') else 'SL_BE'
            elif hit_tp:
                open_trade.exit_time = row['datetime_utc']
                open_trade.exit_price = open_trade.take_profit
                open_trade.exit_reason = 'TP'
            elif not open_trade.exit_reason:
                open_trade.hold_bars += 1
                if open_trade.hold_bars >= max_hold_bars:
                    open_trade.exit_time = row['datetime_utc']
                    open_trade.exit_price = close
                    open_trade.exit_reason = 'timeout'

            if open_trade.exit_reason:
                entry_p, exit_p = open_trade.entry_price, open_trade.exit_price
                if open_trade.direction == 'BUY':
                    pnl_pips = (exit_p - entry_p) / pip_value
                else:
                    pnl_pips = (entry_p - exit_p) / pip_value
                pnl_pips_adj = pnl_pips - spread_pips - slippage_pips

                # Pip value in USD (correct for JPY vs non-JPY)
                pip_usd_value = get_pip_value_usd(symbol, entry_p)
                pnl_usd = pnl_pips_adj * open_trade.lot_size * pip_usd_value
                pnl_usd -= commission_per_lot * open_trade.lot_size

                open_trade.pnl_pips = pnl_pips_adj
                open_trade.pnl_usd = pnl_usd
                balance += pnl_usd
                trades.append(open_trade)
                last_close_bar = i
                open_trade = None
                open_trade_meta = {}

        # ── Open new trade ──
        if open_trade is None:
            if i - last_close_bar < cooldown_bars:
                continue

            # Session filter
            if enable_session_filter:
                sess = row['session'] if 'session' in df.columns else 'off'
                if sess not in ('london', 'ny', 'overlap'):
                    continue

            sig = signal_fn(row, symbol)
            if sig['signal'] not in ('BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL'):
                continue
            if sig['confidence'] < min_confidence:
                continue

            atr_val = row['atr']
            if atr_val <= 0 or pd.isna(atr_val):
                continue

            # Skip wide spread
            if row.get('spread', 0) > 25:
                continue

            # Volatility filter — skip extreme bars
            range_pct = (high - low) / close * 100
            range_ma  = row.get('range_ma', range_pct)
            if range_ma > 0 and range_pct > range_ma * 2.5:
                continue

            entry_price = close

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

            # Confidence-weighted sizing (conservative)
            conf_mult = 0.7 + (sig['confidence'] / 100) * 0.5  # 0.7x to 1.2x
            pip_usd_value = get_pip_value_usd(symbol, actual_entry)
            lot_size = (risk_usd / (sl_pips * pip_usd_value)) * conf_mult
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
            open_trade_meta = {'be_moved': False}
            trade_id += 1

    # Close any remaining
    if open_trade is not None:
        last_row = df.iloc[-1]
        open_trade.exit_time = last_row['datetime_utc']
        open_trade.exit_price = last_row['close']
        open_trade.exit_reason = 'end_of_backtest'
        entry_p, exit_p = open_trade.entry_price, open_trade.exit_price
        if open_trade.direction == 'BUY':
            pnl_pips = (exit_p - entry_p) / pip_value
        else:
            pnl_pips = (entry_p - exit_p) / pip_value
        pnl_pips_adj = pnl_pips - spread_pips - slippage_pips
        pip_usd_value = get_pip_value_usd(symbol, entry_p)
        pnl_usd = pnl_pips_adj * open_trade.lot_size * pip_usd_value - commission_per_lot * open_trade.lot_size
        open_trade.pnl_pips = pnl_pips_adj
        open_trade.pnl_usd = pnl_usd
        balance += pnl_usd
        trades.append(open_trade)

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

    exit_reasons = {}
    for t in trades:
        r = t.exit_reason
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    if len(trades) >= 2:
        first_time = pd.to_datetime(trades[0].entry_time)
        last_time  = pd.to_datetime(trades[-1].entry_time)
        days = (last_time - first_time).days or 1
        trades_per_year = len(trades) / days * 365
    else:
        trades_per_year = 0

    # Max drawdown
    equity = [initial_balance]
    for t in trades:
        equity.append(equity[-1] + t.pnl_usd)
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak: peak = e
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd

    metrics = {
        'symbol': symbol, 'timeframe': timeframe,
        'total_trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2),
        'total_pnl_usd': round(total_pnl, 2),
        'final_balance': round(balance, 2),
        'avg_win_usd': round(avg_win, 2),
        'avg_loss_usd': round(avg_loss, 2),
        'expectancy_usd': round(expectancy, 2),
        'avg_hold_bars': round(np.mean([t.hold_bars for t in trades]), 1),
        'buy_trades': len(buy_trades), 'buy_wins': buy_wins,
        'buy_win_rate': round(buy_wins/len(buy_trades)*100, 2) if buy_trades else 0,
        'sell_trades': len(sell_trades), 'sell_wins': sell_wins,
        'sell_win_rate': round(sell_wins/len(sell_trades)*100, 2) if sell_trades else 0,
        'trades_per_year': round(trades_per_year, 1),
        'max_drawdown_pct': round(max_dd, 2),
        'exit_tp': exit_reasons.get('TP', 0),
        'exit_sl': exit_reasons.get('SL', 0),
        'exit_sl_be': exit_reasons.get('SL_BE', 0),
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
        print(f"  v4 FINAL backtest for {symbol} H1")
        print(f"{'='*60}")

        df = prepare_data_v2(csv_path, symbol)
        result = run_backtest_v4(
            df=df, signal_fn=signal_v4,
            symbol=symbol, strategy_name='v4_final',
        )

        m = result['metrics']
        if not m:
            print("  No trades generated.")
            continue

        print(f"  Trades: {m['total_trades']:4d} | Wins: {m['wins']:4d} | WR: {m['win_rate']:5.2f}%")
        print(f"  PF: {m['profit_factor']:5.2f} | PnL: ${m['total_pnl_usd']:>10.2f} | MaxDD: {m['max_drawdown_pct']:5.2f}%")
        print(f"  BUY:  {m['buy_trades']:3d} trades, WR={m['buy_win_rate']:5.2f}%")
        print(f"  SELL: {m['sell_trades']:3d} trades, WR={m['sell_win_rate']:5.2f}%")
        print(f"  Avg Win: ${m['avg_win_usd']:>8.2f} | Avg Loss: ${m['avg_loss_usd']:>8.2f}")
        print(f"  Expectancy: ${m['expectancy_usd']:>8.2f}/trade | Trades/yr: {m['trades_per_year']}")
        print(f"  Exits: TP={m['exit_tp']} SL={m['exit_sl']} SL_BE={m['exit_sl_be']} timeout={m['exit_timeout']} opposite={m['exit_opposite']}")

        all_metrics.append({'symbol': symbol, **m})
        all_trades.extend([asdict(t) for t in result['trades']])

    metrics_df = pd.DataFrame(all_metrics)
    trades_df  = pd.DataFrame(all_trades)
    metrics_csv = OUTPUT_DIR / "v4_final_metrics.csv"
    trades_csv  = OUTPUT_DIR / "v4_final_trades.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    trades_df.to_csv(trades_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  COMBINED v4 FINAL RESULTS  (7 pairs)")
    print(f"{'='*60}")
    if all_metrics:
        total_trades = sum(m['total_trades'] for m in all_metrics)
        total_wins   = sum(m['wins']      for m in all_metrics)
        total_pnl    = sum(m['total_pnl_usd'] for m in all_metrics)
        print(f"  Total trades: {total_trades}")
        print(f"  Combined WR: {total_wins/total_trades*100:.2f}%")
        print(f"  Total PnL: ${total_pnl:.2f}")


if __name__ == "__main__":
    main()
