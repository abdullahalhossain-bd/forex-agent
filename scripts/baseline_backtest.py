"""
Lightweight baseline backtest — bypasses the slow _enrich pattern detection
and the LLM-dependent MasterAnalyst. Runs in seconds, not minutes.

This is NOT a substitute for the full unified_engine backtest. It exists to
establish a baseline that can be re-run after fixes to detect regressions.
The full unified_engine is documented in the audit report as having a
performance bug (_enrich takes 60+ seconds per load).
"""
import sys, os, time, json
sys.path.insert(0, '/home/z/my-project/forex-agent')
os.chdir('/home/z/my-project/forex-agent')
os.environ['TEST_MODE'] = 'false'
os.environ['SIMULATION_MODE'] = 'true'
import logging
logging.getLogger().setLevel(logging.ERROR)

import pandas as pd
import numpy as np


def load_csv_raw(path):
    """Raw load — skip _enrich to avoid 60s+ pattern detection."""
    df = pd.read_csv(path, parse_dates=['datetime_utc'])
    df = df.set_index('datetime_utc')
    df = df.rename(columns={'tick_volume': 'volume'})
    return df


def compute_minimal_indicators(df, period=20):
    """SMA, RSI, ATR — enough for a basic signal."""
    df = df.copy()
    df['sma'] = df['close'].rolling(period).mean()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close']
    return df


def simple_signal(row, prev_row):
    """Simple long/short signal: SMA cross + RSI confirmation.
    BUY when close > SMA and RSI > 50 and rising
    SELL when close < SMA and RSI < 50 and falling
    """
    if pd.isna(row['sma']) or pd.isna(row['rsi']):
        return 'WAIT'
    if prev_row is None or pd.isna(prev_row['sma']):
        return 'WAIT'
    # Cross up
    if prev_row['close'] <= prev_row['sma'] and row['close'] > row['sma'] and row['rsi'] > 50:
        return 'BUY'
    # Cross down
    if prev_row['close'] >= prev_row['sma'] and row['close'] < row['sma'] and row['rsi'] < 50:
        return 'SELL'
    return 'WAIT'


def backtest_pair(df, pair_name='EURUSD', starting_balance=10000.0, risk_pct=0.01, rr=2.0,
                  atr_sl_mult=1.5, spread_pips=1.0, max_open=3, max_hold=100):
    """Run a simple backtest on one df.
    Returns dict with trades list + metrics.
    """
    balance = starting_balance
    trades = []
    open_trades = []
    pip_size = 0.0001 if 'JPY' not in pair_name else 0.01

    for i in range(50, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        # Manage open trades — check SL/TP hits
        for t in open_trades[:]:
            bars_held = i - t['entry_idx']
            # SL / TP check (use high/low)
            if t['direction'] == 'BUY':
                if row['low'] <= t['sl']:
                    t['exit_price'] = t['sl']
                    t['exit_time'] = row.name
                    t['pnl_pips'] = (t['sl'] - t['entry']) / pip_size
                    t['result'] = 'LOSS'
                    t['reason'] = 'SL'
                    balance += t['pnl_usd']
                    open_trades.remove(t)
                elif row['high'] >= t['tp']:
                    t['exit_price'] = t['tp']
                    t['exit_time'] = row.name
                    t['pnl_pips'] = (t['tp'] - t['entry']) / pip_size
                    t['result'] = 'WIN'
                    t['reason'] = 'TP'
                    balance += t['pnl_usd']
                    open_trades.remove(t)
                elif bars_held >= max_hold:
                    t['exit_price'] = row['close']
                    t['exit_time'] = row.name
                    t['pnl_pips'] = (row['close'] - t['entry']) / pip_size
                    t['result'] = 'WIN' if t['pnl_pips'] > 0 else 'LOSS'
                    t['reason'] = 'TIMEOUT'
                    balance += t['pnl_usd']
                    open_trades.remove(t)
            else:  # SELL
                if row['high'] >= t['sl']:
                    t['exit_price'] = t['sl']
                    t['exit_time'] = row.name
                    t['pnl_pips'] = (t['entry'] - t['sl']) / pip_size
                    t['result'] = 'LOSS'
                    t['reason'] = 'SL'
                    balance += t['pnl_usd']
                    open_trades.remove(t)
                elif row['low'] <= t['tp']:
                    t['exit_price'] = t['tp']
                    t['exit_time'] = row.name
                    t['pnl_pips'] = (t['entry'] - t['tp']) / pip_size
                    t['result'] = 'WIN'
                    t['reason'] = 'TP'
                    balance += t['pnl_usd']
                    open_trades.remove(t)
                elif bars_held >= max_hold:
                    t['exit_price'] = row['close']
                    t['exit_time'] = row.name
                    t['pnl_pips'] = (t['entry'] - row['close']) / pip_size
                    t['result'] = 'WIN' if t['pnl_pips'] > 0 else 'LOSS'
                    t['reason'] = 'TIMEOUT'
                    balance += t['pnl_usd']
                    open_trades.remove(t)

        # Look for new entry (only if slots available)
        if len(open_trades) >= max_open:
            continue
        # Don't open same-direction on same symbol if already open
        sig = simple_signal(row, prev)
        if sig in ('BUY', 'SELL'):
            # Skip if already open in same direction
            if any(t['direction'] == sig for t in open_trades):
                continue
            entry = row['close']
            atr = row['atr'] if not pd.isna(row['atr']) else entry * 0.005
            sl_dist = atr * atr_sl_mult
            if sig == 'BUY':
                sl = entry - sl_dist
                tp = entry + sl_dist * rr
            else:
                sl = entry + sl_dist
                tp = entry - sl_dist * rr
            # Subtract half spread from entry (cost)
            entry_adj = entry - (spread_pips * pip_size / 2) if sig == 'BUY' else entry + (spread_pips * pip_size / 2)
            risk_usd = balance * risk_pct
            pip_value_per_lot = 10.0  # $10 per pip per 1.0 lot for EURUSD
            lot = risk_usd / (sl_dist / pip_size * pip_value_per_lot) if sl_dist > 0 else 0
            lot = max(0.01, round(lot, 2))
            # Adjust SL/TP for spread
            if sig == 'BUY':
                sl_actual = sl
                tp_actual = tp
            else:
                sl_actual = sl
                tp_actual = tp
            t = {
                'entry_time': row.name,
                'entry_idx': i,
                'direction': sig,
                'entry': entry_adj,
                'sl': sl_actual,
                'tp': tp_actual,
                'lot': lot,
                'pnl_pips': 0,
                'pnl_usd': 0,
            }
            # Pre-compute pnl_usd
            t['_pnl_per_pip'] = lot * pip_value_per_lot
            # patch pnl_usd calc later
            open_trades.append(t)

    # Close any remaining trades at last close
    last_row = df.iloc[-1]
    for t in open_trades:
        if t['direction'] == 'BUY':
            t['exit_price'] = last_row['close']
            t['pnl_pips'] = (last_row['close'] - t['entry']) / pip_size
        else:
            t['exit_price'] = last_row['close']
            t['pnl_pips'] = (t['entry'] - last_row['close']) / pip_size
        t['exit_time'] = last_row.name
        t['result'] = 'WIN' if t['pnl_pips'] > 0 else 'LOSS'
        t['reason'] = 'EOD'
        t['pnl_usd'] = t['pnl_pips'] * t['_pnl_per_pip']
        balance += t['pnl_usd']
        trades.append(t)

    # Finalize all closed trades' pnl_usd
    for t in [tr for tr in trades if 'pnl_usd' not in tr or tr['pnl_usd'] == 0]:
        t['pnl_usd'] = t['pnl_pips'] * t.get('_pnl_per_pip', 0.01 * 10)
    trades = [t for t in trades if 'exit_price' in t]

    return {'trades': trades, 'final_balance': balance}


def compute_metrics(trades, starting_balance=10000.0):
    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0, 'net_pnl': 0,
                'avg_win': 0, 'avg_loss': 0, 'max_drawdown': 0, 'expectancy': 0,
                'buy_wr': 0, 'sell_wr': 0}
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]
    gross_profit = sum(t['pnl_usd'] for t in wins)
    gross_loss = abs(sum(t['pnl_usd'] for t in losses))
    net_pnl = sum(t['pnl_usd'] for t in trades)
    # Equity curve for drawdown
    eq = [starting_balance]
    for t in sorted(trades, key=lambda x: x.get('exit_time', x.get('entry_time'))):
        eq.append(eq[-1] + t['pnl_usd'])
    peak = eq[0]
    max_dd = 0
    for v in eq:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd
    buy_trades = [t for t in trades if t['direction'] == 'BUY']
    sell_trades = [t for t in trades if t['direction'] == 'SELL']
    buy_wins = [t for t in buy_trades if t['pnl_usd'] > 0]
    sell_wins = [t for t in sell_trades if t['pnl_usd'] > 0]
    return {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades) * 100,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'net_pnl': net_pnl,
        'avg_win': gross_profit / len(wins) if wins else 0,
        'avg_loss': gross_loss / len(losses) if losses else 0,
        'expectancy': net_pnl / len(trades),
        'max_drawdown': max_dd,
        'buy_wr': len(buy_wins) / len(buy_trades) * 100 if buy_trades else 0,
        'sell_wr': len(sell_wins) / len(sell_trades) * 100 if sell_trades else 0,
        'buy_count': len(buy_trades),
        'sell_count': len(sell_trades),
    }


def main():
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF']
    timeframes = ['H1', 'H4', 'M15']
    results = {}

    for tf in timeframes:
        for pair in pairs:
            path = f'data/{pair}_{tf}.csv'
            if not os.path.exists(path):
                continue
            print(f'\n=== {pair} {tf} ===', flush=True)
            t0 = time.time()
            df = load_csv_raw(path)
            df = compute_minimal_indicators(df)
            # Use last 1 year of data
            df = df.tail(min(len(df), 5000))
            t_load = time.time() - t0
            print(f'  loaded {len(df)} bars in {t_load:.2f}s', flush=True)

            t0 = time.time()
            r = backtest_pair(df, pair_name=pair)
            t_run = time.time() - t0
            m = compute_metrics(r['trades'])
            print(f'  backtest {t_run:.2f}s | trades={m["total_trades"]} WR={m["win_rate"]:.1f}% '
                  f'PF={m["profit_factor"]:.2f} net=${m["net_pnl"]:.2f} maxDD={m["max_drawdown"]:.1f}%', flush=True)
            results[f'{pair}_{tf}'] = {
                'metrics': m,
                'final_balance': r['final_balance'],
                'bars': len(df),
            }

    # Summary by pair and timeframe
    print(f'\n\n{"="*80}')
    print(f'  BASELINE BACKTEST SUMMARY (simple SMA+RSI strategy)')
    print(f'{"="*80}')
    total_trades = sum(r['metrics']['total_trades'] for r in results.values())
    total_pnl = sum(r['metrics']['net_pnl'] for r in results.values())
    total_wins = sum(r['metrics'].get('wins', 0) for r in results.values())
    print(f'  Total trades: {total_trades}')
    print(f'  Total P&L: ${total_pnl:.2f}')
    if total_trades:
        print(f'  Overall WR: {total_wins/total_trades*100:.1f}%')
    print(f'\n  Per-pair breakdown:')
    print(f'  {"Pair_TF":<15} {"Trades":>8} {"WR%":>7} {"PF":>6} {"Net$":>10} {"MaxDD%":>7}')
    for k, v in sorted(results.items()):
        m = v['metrics']
        print(f'  {k:<15} {m["total_trades"]:>8} {m["win_rate"]:>6.1f}% {m["profit_factor"]:>6.2f} {m["net_pnl"]:>9.2f} {m["max_drawdown"]:>6.1f}%')

    # Save
    os.makedirs('/home/z/my-project/download', exist_ok=True)
    with open('/home/z/my-project/download/baseline_backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Saved: /home/z/my-project/download/baseline_backtest_results.json')


if __name__ == '__main__':
    main()
