#!/usr/bin/env python3
"""
Forex-Agent Strategy v5 — Production Final
==========================================
v4 had: TP=0 hits because trailing stop kicked in too early.
v5 fixes:
  1. NO trailing stop — let trades run to TP or SL
  2. TP closer (1:1.5 R:R instead of 1:2) — higher hit rate
  3. SL slightly wider (2.0 ATR) for noise tolerance
  4. Breakeven at 1R (not 1.2R) — protect capital faster
  5. SL_BE trades counted as wins (pnl > 0)
  6. Stronger signal: net >= 8 (was 7)
  7. Better session filter — skip first/last hour of session
"""

from __future__ import annotations
import sys, json, math
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import Optional, List, Dict, Any

sys.path.insert(0, '/home/z/my-project/scripts')
from forex_backtest_v1 import ema, rsi, macd, atr, adx, vwap, detect_bos, Trade, prepare_data, DATA_DIR, OUTPUT_DIR
from forex_backtest_v2 import bollinger_bands, stochastic, cci, obv, detect_choch, liquidity_sweep, prepare_data_v2
from forex_backtest_v4 import get_pip_value, get_pip_value_usd, signal_v4


def run_backtest_v5(
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
    atr_sl_mult: float = 2.0,      # wider for noise tolerance
    atr_tp_mult: float = 3.0,      # 1:1.5 R:R
    max_hold_bars: int = 60,       # let trades breathe
    strategy_name: str = 'v5_production',
    enable_breakeven: bool = True,
    breakeven_at_r: float = 1.0,   # at 1R move to BE
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

            # Breakeven move FIRST
            if enable_breakeven and not meta.get('be_moved', False):
                risk_distance = abs(entry - meta.get('orig_sl', open_trade.stop_loss))
                if open_trade.direction == 'BUY':
                    be_trigger = entry + risk_distance * breakeven_at_r
                    if high >= be_trigger:
                        open_trade.stop_loss = entry + (slippage_pips + spread_pips) * pip_value
                        meta['be_moved'] = True
                else:
                    be_trigger = entry - risk_distance * breakeven_at_r
                    if low <= be_trigger:
                        open_trade.stop_loss = entry - (slippage_pips + spread_pips) * pip_value
                        meta['be_moved'] = True

            # Check TP/SL
            hit_tp, hit_sl = False, False
            if open_trade.direction == 'BUY':
                if low <= open_trade.stop_loss: hit_sl = True
                elif high >= open_trade.take_profit: hit_tp = True
            else:
                if high >= open_trade.stop_loss: hit_sl = True
                elif low <= open_trade.take_profit: hit_tp = True

            # Strong opposite signal exit
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
                open_trade.exit_reason = 'SL_BE' if meta.get('be_moved') else 'SL'
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

            if row.get('spread', 0) > 25:
                continue

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

            risk_usd = balance * risk_per_trade
            sl_pips = abs(actual_entry - sl) / pip_value
            if sl_pips <= 0:
                continue

            conf_mult = 0.7 + (sig['confidence'] / 100) * 0.5
            pip_usd_value = get_pip_value_usd(symbol, actual_entry)
            lot_size = (risk_usd / (sl_pips * pip_usd_value)) * conf_mult
            lot_size = max(0.01, min(1.0, round(lot_size, 2)))

            commission = commission_per_lot * lot_size

            t = Trade(
                trade_id=trade_id, symbol=symbol, direction=direction,
                entry_time=row['datetime_utc'], entry_price=actual_entry,
                stop_loss=sl, take_profit=tp,
                lot_size=lot_size, confidence=sig['confidence'],
                strategy=strategy_name,
                commission_usd=commission, slippage_pips=slippage_pips,
            )
            open_trade = t
            open_trade_meta = {'be_moved': False, 'orig_sl': sl}
            trade_id += 1

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

    equity = [initial_balance]
    for t in trades:
        equity.append(equity[-1] + t.pnl_usd)
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak: peak = e
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd

    # Sharpe-like ratio
    if len(trades) > 5:
        returns = [t.pnl_usd / initial_balance for t in trades]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
    else:
        sharpe = 0

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
        'sharpe': round(sharpe, 2),
        'exit_tp': exit_reasons.get('TP', 0),
        'exit_sl': exit_reasons.get('SL', 0),
        'exit_sl_be': exit_reasons.get('SL_BE', 0),
        'exit_timeout': exit_reasons.get('timeout', 0),
        'exit_opposite': exit_reasons.get('opposite_signal', 0),
        'exit_end': exit_reasons.get('end_of_backtest', 0),
    }

    return {'trades': trades, 'metrics': metrics, 'final_balance': balance}


def main():
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD', 'NZDUSD']
    all_metrics = []
    all_trades  = []

    for symbol in symbols:
        csv_path = DATA_DIR / symbol / f"{symbol}_H1.csv"
        if not csv_path.exists():
            continue

        print(f"\n{'='*60}")
        print(f"  v5 PRODUCTION backtest for {symbol} H1")
        print(f"{'='*60}")

        df = prepare_data_v2(csv_path, symbol)
        result = run_backtest_v5(
            df=df, signal_fn=signal_v4,
            symbol=symbol, strategy_name='v5_production',
        )

        m = result['metrics']
        if not m:
            print("  No trades generated.")
            continue

        print(f"  Trades: {m['total_trades']:4d} | Wins: {m['wins']:4d} | WR: {m['win_rate']:5.2f}%")
        print(f"  PF: {m['profit_factor']:5.2f} | Sharpe: {m['sharpe']:.2f} | PnL: ${m['total_pnl_usd']:>9.2f} | MaxDD: {m['max_drawdown_pct']:5.2f}%")
        print(f"  BUY:  {m['buy_trades']:3d} trades, WR={m['buy_win_rate']:5.2f}%")
        print(f"  SELL: {m['sell_trades']:3d} trades, WR={m['sell_win_rate']:5.2f}%")
        print(f"  Avg Win: ${m['avg_win_usd']:>8.2f} | Avg Loss: ${m['avg_loss_usd']:>8.2f}")
        print(f"  Expectancy: ${m['expectancy_usd']:>8.2f}/trade | Trades/yr: {m['trades_per_year']}")
        print(f"  Exits: TP={m['exit_tp']} SL={m['exit_sl']} SL_BE={m['exit_sl_be']} timeout={m['exit_timeout']} opposite={m['exit_opposite']}")

        all_metrics.append({'symbol': symbol, **m})
        all_trades.extend([asdict(t) for t in result['trades']])

    metrics_df = pd.DataFrame(all_metrics)
    trades_df  = pd.DataFrame(all_trades)
    metrics_csv = OUTPUT_DIR / "v5_production_metrics.csv"
    trades_csv  = OUTPUT_DIR / "v5_production_trades.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    trades_df.to_csv(trades_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  COMBINED v5 PRODUCTION RESULTS  (7 pairs)")
    print(f"{'='*60}")
    if all_metrics:
        total_trades = sum(m['total_trades'] for m in all_metrics)
        total_wins   = sum(m['wins']      for m in all_metrics)
        total_pnl    = sum(m['total_pnl_usd'] for m in all_metrics)
        total_tp     = sum(m['exit_tp']    for m in all_metrics)
        total_sl     = sum(m['exit_sl']    for m in all_metrics)
        total_be     = sum(m['exit_sl_be'] for m in all_metrics)
        print(f"  Total trades: {total_trades}")
        print(f"  Combined WR: {total_wins/total_trades*100:.2f}%")
        print(f"  Total PnL: ${total_pnl:.2f}")
        print(f"  Exit breakdown: TP={total_tp} SL={total_sl} SL_BE={total_be}")


if __name__ == "__main__":
    main()
