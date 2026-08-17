"""Run unified backtest with stdout redirected to /dev/null for speed."""
import sys, os, time, json, traceback, contextlib
sys.path.insert(0, '/home/z/my-project/forex-agent')
os.chdir('/home/z/my-project/forex-agent')
os.environ['TEST_MODE'] = 'false'
os.environ['SIMULATION_MODE'] = 'true'
import logging
logging.getLogger().setLevel(logging.CRITICAL)

# Redirect all stdout (the 50+ print_summary calls per bar) to /dev/null
# This is the key perf fix — the prints themselves are fast but the I/O
# blocking on terminal output was the bottleneck.
devnull = open(os.devnull, 'w')

from backtest.unified_engine import run_unified_backtest
from backtest.data_loader import HistoricalDataLoader

loader = HistoricalDataLoader()
df = loader.load_csv(file_path='data/EURUSD_H1.csv', pair='EURUSD', timeframe='H1')
sys.stderr.write(f'loaded {len(df)} bars\n')
sys.stderr.flush()

# Use last 1000 bars for a meaningful backtest
df_test = df.tail(1000)
sys.stderr.write(f'testing with last {len(df_test)} bars\n')
sys.stderr.flush()

t0 = time.time()
try:
    # Redirect stdout during the entire backtest
    old_stdout = sys.stdout
    sys.stdout = devnull
    try:
        result = run_unified_backtest(
            symbol='EURUSD', df=df_test, timeframe='H1',
            starting_balance=10000.0, warmup_bars=50,
            max_open_trades=3, max_hold_bars=100,
            db_path='backtest/phase3_EURUSD_H1_1000.db',
            verbose=False, bypass_checks=[],
        )
    finally:
        sys.stdout = old_stdout
    elapsed = time.time() - t0
    sys.stderr.write(f'backtest done in {elapsed:.1f}s ({elapsed/max(1,len(df_test)-50):.2f}s per bar)\n')
    sys.stderr.write(f'trades={result.metrics.total_trades} bars={result.bars}\n')
    if result.error:
        sys.stderr.write(f'ERROR: {result.error}\n')
    sys.stderr.write(f'rejection_stats:\n')
    for k,v in result.rejection_stats.items():
        sys.stderr.write(f'  {k:<30} {v}\n')
    if result.metrics.total_trades > 0:
        m = result.metrics
        sys.stderr.write(f'\nMetrics:\n')
        sys.stderr.write(f'  WR={m.win_rate:.1f}% PF={m.profit_factor:.2f}\n')
        sys.stderr.write(f'  net=${m.total_pnl_usd:.2f} maxDD={m.max_drawdown_pct:.1f}%\n')
        sys.stderr.write(f'  sharpe={m.sharpe_ratio:.2f} sortino={m.sortino_ratio:.2f}\n')
        sys.stderr.write(f'  avg_win=${m.avg_win_usd:.2f} avg_loss=${m.avg_loss_usd:.2f}\n')
    # Save full result
    out = {
        'symbol': 'EURUSD',
        'timeframe': 'H1',
        'bars': result.bars,
        'elapsed_seconds': elapsed,
        'sec_per_bar': elapsed / max(1, len(df_test) - 50),
        'metrics': {
            'total_trades': result.metrics.total_trades,
            'win_rate': result.metrics.win_rate,
            'profit_factor': result.metrics.profit_factor,
            'total_pnl_usd': result.metrics.total_pnl_usd,
            'max_drawdown_pct': result.metrics.max_drawdown_pct,
            'sharpe_ratio': result.metrics.sharpe_ratio,
            'sortino_ratio': result.metrics.sortino_ratio,
            'avg_win_usd': result.metrics.avg_win_usd,
            'avg_loss_usd': result.metrics.avg_loss_usd,
        },
        'rejection_stats': dict(result.rejection_stats),
    }
    os.makedirs('/home/z/my-project/download', exist_ok=True)
    with open('/home/z/my-project/download/unified_backtest_1000bars.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    sys.stderr.write(f'\nSaved: /home/z/my-project/download/unified_backtest_1000bars.json\n')
except Exception as e:
    elapsed = time.time() - t0
    sys.stderr.write(f'EXCEPTION after {elapsed:.1f}s: {e}\n')
    traceback.print_exc(file=sys.stderr)
