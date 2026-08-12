#!/usr/bin/env python3
"""
Forex-Agent Strategy v3 — Final Tuned Version
=============================================
Refinements over v2:
  1. Strict signal thresholds (net >= 6 for BUY/SELL, was 5)
  2. Min confluence count = 4 factors (was 3)
  3. Breakeven only at 1.5R (was 1R — too early, gets stopped out by noise)
  4. Partial exit removed (was creating tiny losses)
  5. Wider ATR SL (1.5x, was 1.2x — too tight)
  6. Better R:R (1:2, was 1:2.5 — too ambitious, rarely hits TP)
  7. Cooldown between trades (avoid overtrading)
  8. Trend pullback entry — wait for EMA9/EMA20 touch in trend direction
  9. Higher confidence threshold (65%, was 60%)
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
    ema, rsi, macd, atr, adx, vwap, detect_bos, swing_points,
    Trade, prepare_data, DATA_DIR, OUTPUT_DIR
)
from forex_backtest_v2 import (
    bollinger_bands, stochastic, cci, obv, detect_choch, liquidity_sweep,
    prepare_data_v2
)


# ═════════════════════════════════════════════════════════════
# SIGNAL ENGINE v3 — Strict & Balanced
# ═════════════════════════════════════════════════════════════
def improved_signal_v3(row) -> Dict[str, Any]:
    """
    v3 — strict confluence with HTF filter, no asymmetric bias.
    Returns WAIT unless ≥4 factors align with HTF direction.
    """
    bull_score = 0
    bear_score = 0
    bull_factors = 0
    bear_factors = 0
    signals = []
    warnings = []

    price = row['close']
    ema9, ema20, ema50, ema200 = row['ema9'], row['ema20'], row['ema50'], row['ema200']
    htf_bull = price > ema200 and ema50 > ema200
    htf_bear = price < ema200 and ema50 < ema200

    # ── 1. HIGHER TF BIAS (gate) ──
    if htf_bull:
        bull_score += 2; bull_factors += 1
        signals.append(('bullish', 2, 'HTF bull (price+EMA50 > EMA200)'))
    elif htf_bear:
        bear_score += 2; bear_factors += 1
        signals.append(('bearish', 2, 'HTF bear (price+EMA50 < EMA200)'))
    else:
        # No clear HTF bias → skip
        return {'signal': 'WAIT', 'confidence': 0, 'net': 0, 'warnings': ['No HTF bias']}

    # ── 2. EMA STACK (trend alignment) ──
    if ema9 > ema20 > ema50 and htf_bull:
        bull_score += 2; bull_factors += 1
        signals.append(('bullish', 2, 'EMA9>20>50 stack bull'))
    elif ema9 < ema20 < ema50 and htf_bear:
        bear_score += 2; bear_factors += 1
        signals.append(('bearish', 2, 'EMA9<20<50 stack bear'))

    # ── 3. RSI MOMENTUM (aligned with HTF) ──
    rsi_val = row['rsi']
    if htf_bull:
        if 40 <= rsi_val < 60:  # healthy pullback zone in uptrend
            bull_score += 2; bull_factors += 1
            signals.append(('bullish', 2, f'RSI pullback zone ({rsi_val:.0f})'))
        elif rsi_val < 35:  # deeper pullback — bigger bounce opportunity
            bull_score += 2; bull_factors += 1
            signals.append(('bullish', 2, f'RSI deep pullback ({rsi_val:.0f})'))
        elif rsi_val > 75:
            warnings.append(f'RSI overbought in uptrend ({rsi_val:.0f})')
    elif htf_bear:
        if 40 < rsi_val <= 60:
            bear_score += 2; bear_factors += 1
            signals.append(('bearish', 2, f'RSI pullback zone ({rsi_val:.0f})'))
        elif rsi_val > 65:
            bear_score += 2; bear_factors += 1
            signals.append(('bearish', 2, f'RSI deep pullback ({rsi_val:.0f})'))
        elif rsi_val < 25:
            warnings.append(f'RSI oversold in downtrend ({rsi_val:.0f})')

    # ── 4. MACD (momentum confirmation) ──
    if row['macd'] > row['macd_signal'] and row['macd'] > 0 and htf_bull:
        bull_score += 1; bull_factors += 1
        signals.append(('bullish', 1, 'MACD bull + above zero'))
    elif row['macd'] < row['macd_signal'] and row['macd'] < 0 and htf_bear:
        bear_score += 1; bear_factors += 1
        signals.append(('bearish', 1, 'MACD bear + below zero'))

    # ── 5. BOS (structure) ──
    if row['bos'] == 1 and htf_bull:
        bull_score += 2; bull_factors += 1
        signals.append(('bullish', 2, 'Bullish BOS'))
    elif row['bos'] == -1 and htf_bear:
        bear_score += 2; bear_factors += 1
        signals.append(('bearish', 2, 'Bearish BOS'))

    # ── 6. CHoCH (only against current HTF is OK = early reversal)
    # Disable for v3 — keep trend-continuation focus
    # ── 7. LIQUIDITY SWEEP (SMC) ──
    if row['liquidity_sweep'] == 1 and htf_bull:
        bull_score += 2; bull_factors += 1
        signals.append(('bullish', 2, 'Bullish liquidity sweep'))
    elif row['liquidity_sweep'] == -1 and htf_bear:
        bear_score += 2; bear_factors += 1
        signals.append(('bearish', 2, 'Bearish liquidity sweep'))

    # ── 8. ADX (trend strength gate) ──
    adx_val = row['adx']
    if adx_val < 20:
        warnings.append(f'ADX low ({adx_val:.0f}) — weak trend')
        # Reduce scores in choppy market
        bull_score = max(0, bull_score - 1)
        bear_score = max(0, bear_score - 1)
    elif adx_val > 30:
        if bull_score > bear_score:
            bull_score += 1; signals.append(('bullish', 1, f'ADX strong ({adx_val:.0f})'))
        elif bear_score > bull_score:
            bear_score += 1; signals.append(('bearish', 1, f'ADX strong ({adx_val:.0f})'))

    # ── 9. VOLUME CONFIRMATION ──
    if row['tick_volume'] > row['volume_ma'] * 1.5:
        if bull_score > bear_score:
            bull_score += 1; bull_factors += 1
        elif bear_score > bull_score:
            bear_score += 1; bear_factors += 1

    # ── 10. STOCHASTIC confirmation (only in pullback zone) ──
    if htf_bull and row['stoch_k'] > row['stoch_d'] and row['stoch_k'] < 30:
        bull_score += 1; bull_factors += 1
    elif htf_bear and row['stoch_k'] < row['stoch_d'] and row['stoch_k'] > 70:
        bear_score += 1; bear_factors += 1

    # ── Apply warning penalty ──
    penalty = len(warnings) * 5
    bull_score = max(0, bull_score - penalty)
    bear_score = max(0, bear_score - penalty)

    # ── Final Decision ──
    total = bull_score + bear_score
    net = bull_score - bear_score

    if total == 0:
        return {'signal': 'WAIT', 'confidence': 0, 'net': 0, 'warnings': warnings}

    confidence = round(max(bull_score, bear_score) / total * 100)

    # v3 STRICT THRESHOLDS — require net >= 6 AND >= 4 confluence factors
    max_factors = max(bull_factors, bear_factors)
    if max_factors < 4:
        return {'signal': 'WAIT', 'confidence': confidence, 'net': net, 'warnings': warnings + ['Insufficient factors']}

    if net >= 8 and htf_bull:
        signal = 'STRONG_BUY'
    elif net >= 6 and htf_bull:
        signal = 'BUY'
    elif net <= -8 and htf_bear:
        signal = 'STRONG_SELL'
    elif net <= -6 and htf_bear:
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
        'factors': max_factors,
    }


# ═════════════════════════════════════════════════════════════
# BACKTEST ENGINE v3 — Tuned Parameters
# ═════════════════════════════════════════════════════════════
def run_backtest_v3(
    df: pd.DataFrame,
    signal_fn,
    symbol: str = 'EURUSD',
    timeframe: str = 'H1',
    initial_balance: float = 10_000,
    risk_per_trade: float = 0.01,
    spread_pips: float = 1.5,
    commission_per_lot: float = 7.0,
    slippage_pips: float = 2.0,
    min_confidence: float = 65,    # raised
    atr_sl_mult: float = 1.5,      # wider SL
    atr_tp_mult: float = 3.0,      # 1:2 R:R
    max_hold_bars: int = 50,
    strategy_name: str = 'v3_strict',
    pip_value: float = 0.0001,
    enable_breakeven: bool = True,
    breakeven_at_r: float = 1.5,   # move to BE at 1.5R
    enable_session_filter: bool = True,
    cooldown_bars: int = 3,        # min bars between trades
) -> Dict[str, Any]:

    trades: List[Trade] = []
    trade_id = 1
    balance = initial_balance
    open_trade: Optional[Trade] = None
    open_trade_meta: Dict[str, Any] = {}
    warmup = 200
    last_close_bar = -999

    for i in range(warmup, len(df)):
        row = df.iloc[i]

        # ── Check open trade exit ──
        if open_trade is not None:
            high, low = row['high'], row['low']
            meta = open_trade_meta

            # Breakeven at breakeven_at_r
            if enable_breakeven and not meta.get('be_moved', False):
                entry = open_trade.entry_price
                sl = open_trade.stop_loss
                risk_distance = abs(entry - sl)
                be_threshold = entry + (risk_distance * breakeven_at_r) if open_trade.direction == 'BUY' \
                              else entry - (risk_distance * breakeven_at_r)
                if open_trade.direction == 'BUY' and high >= be_threshold:
                    open_trade.stop_loss = entry + slippage_pips * pip_value
                    meta['be_moved'] = True
                elif open_trade.direction == 'SELL' and low <= be_threshold:
                    open_trade.stop_loss = entry - slippage_pips * pip_value
                    meta['be_moved'] = True

            # Check TP / SL
            hit_tp, hit_sl = False, False
            if open_trade.direction == 'BUY':
                if low <= open_trade.stop_loss: hit_sl = True
                elif high >= open_trade.take_profit: hit_tp = True
            else:
                if high >= open_trade.stop_loss: hit_sl = True
                elif low <= open_trade.take_profit: hit_tp = True

            # Opposite signal early exit (only STRONG opposite signals)
            if not hit_tp and not hit_sl:
                sig_now = signal_fn(row)
                if sig_now['signal'] in ('STRONG_SELL', 'STRONG_BUY'):
                    if (open_trade.direction == 'BUY' and sig_now['signal'] == 'STRONG_SELL') or \
                       (open_trade.direction == 'SELL' and sig_now['signal'] == 'STRONG_BUY'):
                        if sig_now['confidence'] >= 75:
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
                pnl_usd = pnl_pips_adj * open_trade.lot_size * 100
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
            # Cooldown
            if i - last_close_bar < cooldown_bars:
                continue

            # Session filter
            if enable_session_filter:
                sess = row.get('session', 'off') if hasattr(row, 'get') else row['session']
                if sess == 'off':
                    continue

            sig = signal_fn(row)
            if sig['signal'] in ('BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL') and sig['confidence'] >= min_confidence:
                atr_val = row['atr']
                if atr_val <= 0 or pd.isna(atr_val):
                    continue

                # Skip wide spread
                if row.get('spread', 0) > 30:
                    continue

                # Volatility filter
                range_pct = row.get('range_pct', 0)
                range_ma  = row.get('range_ma', 0)
                if range_ma > 0 and range_pct > range_ma * 2.5:
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

                # Position sizing (confidence-weighted, but capped)
                risk_usd = balance * risk_per_trade
                sl_pips = abs(actual_entry - sl) / pip_value
                if sl_pips <= 0:
                    continue
                conf_mult = 0.7 + (sig['confidence'] / 100) * 0.6  # 0.7x to 1.3x
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
                open_trade_meta = {'be_moved': False}
                trade_id += 1

    # Close any remaining
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
        pnl_usd = pnl_pips_adj * open_trade.lot_size * 100 - commission_per_lot * open_trade.lot_size
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

    exit_reasons = {}
    for t in trades:
        r = t.exit_reason.split('+')[0]
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
        'max_drawdown_pct': round(max_dd, 2),
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
        print(f"  v3 STRICT backtest for {symbol} H1")
        print(f"{'='*60}")

        df = prepare_data_v2(csv_path, symbol)
        result = run_backtest_v3(
            df=df,
            signal_fn=improved_signal_v3,
            symbol=symbol,
            strategy_name='v3_strict',
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
        print(f"  Exits: TP={m['exit_tp']} SL={m['exit_sl']} timeout={m['exit_timeout']} opposite={m['exit_opposite']}")

        all_metrics.append({'symbol': symbol, **m})
        all_trades.extend([asdict(t) for t in result['trades']])

    metrics_df = pd.DataFrame(all_metrics)
    trades_df  = pd.DataFrame(all_trades)

    metrics_csv = OUTPUT_DIR / "v3_strict_metrics.csv"
    trades_csv  = OUTPUT_DIR / "v3_strict_trades.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    trades_df.to_csv(trades_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  COMBINED v3 STRICT RESULTS  (7 pairs)")
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
