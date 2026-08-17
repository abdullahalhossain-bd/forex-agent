"""Lightweight unified backtest engine — runs AITrader.evaluate_decision_core
on each bar, but with explicit gc.collect() and minimal state retention.

Designed to run within a 4GB / no-swap container.
"""
import sys, os, time, json, gc, traceback
sys.path.insert(0, '/home/z/my-project/forex-agent')
os.chdir('/home/z/my-project/forex-agent')
os.environ['TEST_MODE'] = 'false'
os.environ['SIMULATION_MODE'] = 'true'
import logging
logging.getLogger().setLevel(logging.CRITICAL)

# Suppress all stdout (print_summary calls)
devnull = open(os.devnull, 'w')
old_stdout = sys.stdout

import pandas as pd
import numpy as np
from backtest.data_loader import HistoricalDataLoader
from core.trader import AITrader
from backtest.broker_sim import BrokerSimulator

# Lightweight BrokerSimulator instance (reuse the one from broker_sim.py)
# but we'll drive it manually to avoid unified_engine's overhead


def run_lightweight_backtest(symbol='EURUSD', timeframe='H1', n_bars=100,
                             starting_balance=10000.0, max_open_trades=3,
                             max_hold_bars=50, warmup_bars=20):
    """Run a minimal backtest that calls evaluate_decision_core per bar."""
    loader = HistoricalDataLoader()
    df = loader.load_csv(file_path=f'data/{symbol}_{timeframe}.csv',
                         pair=symbol, timeframe=timeframe)
    df = df.tail(n_bars)
    sys.stderr.write(f'Loaded {len(df)} bars for {symbol} {timeframe}\n')
    sys.stderr.flush()

    # Construct trader
    sys.stdout = devnull
    try:
        trader = AITrader(symbol=symbol, timeframe=timeframe, execution_mode='backtest')
    finally:
        sys.stdout = old_stdout

    # Use the HistoricalCSVProvider to build proper market_out
    from core.provider_factory import make_backtest_provider
    provider = make_backtest_provider(df=df, symbol=symbol, timeframe=timeframe)
    primary_df = getattr(provider, "primary_df", df)

    # Construct broker sim
    broker = BrokerSimulator(starting_balance=starting_balance)

    # Stats
    rejection_stats = {
        'WAIT': 0, 'NO_TRADE_ANALYSIS': 0, 'risk_rejected': 0,
        'permission_blocked': 0, 'engine_error': 0, 'max_trades': 0,
        'broker_rejected': 0, 'executed': 0, 'total_bars': 0,
    }
    trades = []
    open_trades = []  # list of (trade_obj, entry_bar_idx)
    equity_curve = []

    for i in range(warmup_bars, len(primary_df)):
        rejection_stats['total_bars'] += 1
        current_time = primary_df.index[i]
        bar = primary_df.iloc[i]

        # Check exits on open trades
        still_open = []
        for trade, entry_idx in open_trades:
            hold_bars = i - entry_idx
            exit_trade = broker.check_exit(trade, bar['high'], bar['low'], bar['close'], current_time)
            if exit_trade is not None:
                trades.append(exit_trade)
            elif hold_bars >= max_hold_bars:
                closed = broker.close_trade(trade, float(bar['close']), current_time, "timeout")
                closed.hold_bars = hold_bars
                trades.append(closed)
            else:
                trade.hold_bars = hold_bars
                still_open.append((trade, entry_idx))
        open_trades = still_open

        if len(open_trades) >= max_open_trades:
            rejection_stats['max_trades'] += 1
            equity_curve.append(broker.get_balance())
            continue

        # Build market_out using the provider
        sys.stdout = devnull
        try:
            provider.advance_to(i)
            market_out = provider.get_market_out(symbol, timeframe)
            session_ctx = {'current_session': 'BACKTEST', 'gmt_time': str(current_time),
                          'session_strategy': 'n/a'}
            core = trader.evaluate_decision_core(market_out, session_ctx, bypass_checks=[])
        except Exception as e:
            rejection_stats['engine_error'] += 1
            sys.stdout = old_stdout
            sys.stderr.write(f'  bar {i} ({current_time}) error: {type(e).__name__}: {e}\n')
            sys.stderr.flush()
            sys.stdout = devnull
            equity_curve.append(broker.get_balance())
            gc.collect()
            continue
        finally:
            sys.stdout = old_stdout

        analysis_out = core.get('analysis_out', {})
        dec_out = core.get('dec_out', {})
        risk_out = core.get('risk_out', {})
        perm_out = core.get('perm_out', {})

        if 'error' in analysis_out:
            rejection_stats['NO_TRADE_ANALYSIS'] += 1
            equity_curve.append(broker.get_balance())
            gc.collect()
            continue

        action = dec_out.get('decision', 'WAIT')
        if action not in ('BUY', 'SELL'):
            rejection_stats['WAIT'] += 1
            equity_curve.append(broker.get_balance())
            gc.collect()
            continue

        if not risk_out.get('approved'):
            rejection_stats['risk_rejected'] += 1
            equity_curve.append(broker.get_balance())
            gc.collect()
            continue

        if not perm_out.get('allowed'):
            rejection_stats['permission_blocked'] += 1
            equity_curve.append(broker.get_balance())
            gc.collect()
            continue

        # Execute trade
        entry = float(risk_out.get('entry', bar['close']))
        sl = float(risk_out.get('sl_price', 0) or 0)
        tp = float(risk_out.get('tp_price', 0) or 0)
        lot = float(risk_out.get('lot', 0.01))

        trade = broker.open_trade(
            symbol=symbol, direction=action, entry_price=entry,
            sl=sl, tp=tp, lot=lot, bar_time=current_time,
            spread_pips=float(bar.get('spread', 0) or 1.5),
            confidence=int(perm_out.get('confidence_pre_penalty', 50)),
            strategy=dec_out.get('strategy', ''),
        )
        if trade is None:
            rejection_stats['broker_rejected'] += 1
        else:
            rejection_stats['executed'] += 1
            open_trades.append((trade, i))

        equity_curve.append(broker.get_balance())
        gc.collect()

    # Close any remaining open trades at last close
    last_bar = df.iloc[-1]
    for trade, entry_idx in open_trades:
        closed = broker.close_trade(trade, float(last_bar['close']), df.index[-1], "EOD")
        closed.hold_bars = len(df) - 1 - entry_idx
        trades.append(closed)

    # Compute metrics
    total_trades = len(trades)
    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gross_profit = sum(t.pnl_usd for t in wins)
    gross_loss = abs(sum(t.pnl_usd for t in losses))
    net_pnl = sum(t.pnl_usd for t in trades)
    win_rate = len(wins) / total_trades * 100 if total_trades else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    avg_win = gross_profit / len(wins) if wins else 0
    avg_loss = gross_loss / len(losses) if losses else 0
    expectancy = net_pnl / total_trades if total_trades else 0

    # Max drawdown
    eq = [starting_balance] + [b for b in equity_curve]
    peak = eq[0]
    max_dd = 0
    for v in eq:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # BUY/SELL breakdown
    buy_trades = [t for t in trades if t.direction == 'BUY']
    sell_trades = [t for t in trades if t.direction == 'SELL']
    buy_wins = [t for t in buy_trades if t.pnl_usd > 0]
    sell_wins = [t for t in sell_trades if t.pnl_usd > 0]

    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'bars_processed': rejection_stats['total_bars'],
        'rejection_stats': rejection_stats,
        'metrics': {
            'total_trades': total_trades,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 3),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'net_pnl': round(net_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'expectancy': round(expectancy, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'buy_count': len(buy_trades),
            'sell_count': len(sell_trades),
            'buy_wr': round(len(buy_wins)/len(buy_trades)*100, 2) if buy_trades else 0,
            'sell_wr': round(len(sell_wins)/len(sell_trades)*100, 2) if sell_trades else 0,
        },
        'final_balance': round(broker.get_balance(), 2),
    }


def main():
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF']
    timeframes = ['H1']
    all_results = {}

    for pair in pairs:
        for tf in timeframes:
            path = f'data/{pair}_{tf}.csv'
            if not os.path.exists(path):
                continue
            sys.stderr.write(f'\n=== {pair} {tf} ===\n')
            sys.stderr.flush()
            t0 = time.time()
            try:
                result = run_lightweight_backtest(
                    symbol=pair, timeframe=tf, n_bars=100,
                    starting_balance=10000.0, max_open_trades=3,
                    max_hold_bars=50, warmup_bars=20,
                )
                elapsed = time.time() - t0
                sys.stderr.write(f'  done in {elapsed:.1f}s\n')
                m = result['metrics']
                sys.stderr.write(f'  trades={m["total_trades"]} WR={m["win_rate"]}% '
                                f'PF={m["profit_factor"]} net=${m["net_pnl"]} '
                                f'maxDD={m["max_drawdown_pct"]}%\n')
                sys.stderr.write(f'  rejection: {result["rejection_stats"]}\n')
                sys.stderr.flush()
                all_results[f'{pair}_{tf}'] = result
                # Save incrementally
                os.makedirs('/home/z/my-project/download', exist_ok=True)
                with open('/home/z/my-project/download/lightweight_backtest_results.json', 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)
            except Exception as e:
                sys.stderr.write(f'  EXCEPTION: {e}\n')
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()

    # Summary
    sys.stderr.write(f'\n\n{"="*80}\n')
    sys.stderr.write(f'  LIGHTWEIGHT BACKTEST SUMMARY\n')
    sys.stderr.write(f'{"="*80}\n')
    total_trades = sum(r['metrics']['total_trades'] for r in all_results.values())
    total_pnl = sum(r['metrics']['net_pnl'] for r in all_results.values())
    total_wins = sum(r['metrics']['wins'] for r in all_results.values())
    sys.stderr.write(f'  Total trades: {total_trades}\n')
    sys.stderr.write(f'  Total P&L: ${total_pnl:.2f}\n')
    if total_trades:
        sys.stderr.write(f'  Overall WR: {total_wins/total_trades*100:.1f}%\n')
    sys.stderr.write(f'\n  Per-pair breakdown:\n')
    sys.stderr.write(f'  {"Pair_TF":<15} {"Trades":>8} {"WR%":>7} {"PF":>6} {"Net$":>10} {"MaxDD%":>7}\n')
    for k, v in sorted(all_results.items()):
        m = v['metrics']
        sys.stderr.write(f'  {k:<15} {m["total_trades"]:>8} {m["win_rate"]:>6.1f}% '
                         f'{m["profit_factor"]:>6.2f} {m["net_pnl"]:>9.2f} '
                         f'{m["max_drawdown_pct"]:>6.1f}%\n')
    sys.stderr.write(f'{"="*80}\n')
    sys.stderr.write(f'\nSaved: /home/z/my-project/download/lightweight_backtest_results.json\n')


if __name__ == '__main__':
    main()
