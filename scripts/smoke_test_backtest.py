#!/usr/bin/env python3
"""Smoke test: time a single evaluate_decision_core call."""
import sys, os, logging, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["BYPASS_NEWS_GATE"] = "true"
os.environ["BYPASS_FUSION_GATE"] = "true"
os.environ["TEST_MODE"] = "false"

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)

import pandas as pd
from core.trader import AITrader
from database.db import TraderDB
from execution.paper_trader import PaperTrader
from backtest.broker_sim import BrokerSimulator
from backtest.metrics import calculate_metrics
from core.data_provider import HistoricalMT5Provider
from core.execution_adapter import HistoricalExecutionAdapter
from core.constants import set_backtest_mode, get_total_cost_pips

# Load data
df = pd.read_csv('data/EURUSD_H1.csv')
df = df.rename(columns={'datetime_utc': 'time'})
if 'volume' not in df.columns:
    df['volume'] = df.get('tick_volume', 0)
df['time'] = pd.to_datetime(df['time'], utc=True)
df = df.set_index('time')
for c in ['open', 'high', 'low', 'close', 'volume']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['open', 'high', 'low', 'close'])

# Use last 200 bars
N = 200
df_test = df.iloc[-N:].copy()
print(f'Data: {len(df_test)} bars, {df_test.index[50]} to {df_test.index[-1]}')

set_backtest_mode(True)

db = TraderDB(db_path='backtest/backtest_smoke_test.db')
paper = PaperTrader(starting_balance=10000.0, db=db)
trader = AITrader(balance=10000, symbol='EURUSD', timeframe='H1',
                   paper_balance=10000, execution_mode='backtest',
                   paper_trader=paper, db=db)

broker = BrokerSimulator(starting_balance=10000.0)
adapter = HistoricalExecutionAdapter(broker)
provider = HistoricalMT5Provider(df_test, 'EURUSD', 'H1')

import random as _random
_random.seed(42)
import numpy as _np
_np.random.seed(42)

print(f'EURUSD total cost: {get_total_cost_pips("EURUSD")} pips')
print(f'\nRunning {N-50} bars...')

total_t = 0
closed_trades = []
open_trades = []
entry_bar = {}
rejection = {"WAIT": 0, "NO_TRADE_ANALYSIS": 0, "risk_rejected": 0,
               "permission_blocked": 0, "engine_error": 0, "max_trades": 0}

for i in range(50, N):
    t0 = time.time()
    current_time = df_test.index[i]
    
    # Check exits
    still_open = []
    for trade in open_trades:
        result = broker.check_exit(trade, float(df_test.iloc[i]['high']),
                                    float(df_test.iloc[i]['low']), float(df_test.iloc[i]['close']),
                                    current_time)
        if result:
            closed_trades.append(result)
            entry_bar.pop(trade.trade_id, None)
        else:
            still_open.append(trade)
    open_trades = still_open
    
    if len(open_trades) >= 3:
        rejection['max_trades'] += 1
        continue
    
    provider.advance_to(i)
    try:
        market_out = provider.get_market_out('EURUSD', 'H1')
    except Exception as e:
        rejection['engine_error'] += 1
        continue
    
    session_ctx = {'current_session': 'BACKTEST', 'gmt_time': str(current_time), 'session_strategy': 'n/a'}
    try:
        core = trader.evaluate_decision_core(market_out, session_ctx)
    except Exception as e:
        rejection['engine_error'] += 1
        if i < 55 or i > N-5:
            print(f'  [{current_time}] Error: {str(e)[:100]}')
        total_t += time.time() - t0
        continue
    
    dt = time.time() - t0
    total_t += dt
    
    analysis_out = core['analysis_out']
    dec_out = core['dec_out']
    risk_out = core['risk_out']
    perm_out = core['perm_out']
    
    if 'error' in analysis_out:
        rejection['NO_TRADE_ANALYSIS'] += 1
        continue
    
    action = dec_out.get('decision', 'WAIT')
    if action not in ('BUY', 'SELL'):
        rejection['WAIT'] += 1
        continue
    
    if not risk_out.get('approved'):
        rejection['risk_rejected'] += 1
        continue
    
    if not perm_out.get('allowed'):
        rejection['permission_blocked'] += 1
        continue
    
    entry = dec_out.get('entry') or float(df_test.iloc[i]['close'])
    sl = risk_out.get('sl_price')
    tp = risk_out.get('tp_price')
    lot = risk_out.get('lot') or 0.01
    confidence = dec_out.get('confidence', 0)
    
    if not sl or not tp:
        rejection['engine_error'] += 1
        continue
    
    trade = adapter.open_trade(symbol='EURUSD', direction=action, entry_price=entry,
                                sl=sl, tp=tp, lot=lot, bar_time=current_time,
                                confidence=int(confidence) if confidence else 0,
                                strategy='unified_decision_core',
                                confluence_factors=0, quality_grade='B')
    entry_bar[trade.trade_id] = i
    open_trades.append(trade)
    print(f'  [{current_time}] OPEN {action} @ {entry:.5f} SL={sl:.5f} TP={tp:.5f} lot={lot} conf={confidence:.0f}% ({dt:.1f}s)')

# Force close remaining
for trade in open_trades:
    closed_trades.append(broker.close_trade(trade, float(df_test.iloc[-1]['close']), df_test.index[-1], 'end_of_backtest'))

metrics = calculate_metrics(trades=closed_trades, starting_balance=10000, ending_balance=broker.get_balance())
print(f'\n=== RESULTS ({N} bars, {len(df_test)-50} evaluated) ===')
print(f'Trades: {metrics.total_trades}')
print(f'Win rate: {metrics.win_rate:.1f}%')
print(f'PF: {metrics.profit_factor:.2f}')
print(f'PnL: ${metrics.total_pnl_usd:.2f}')
print(f'Max DD: {metrics.max_drawdown_pct:.1f}%')
print(f'Avg bar time: {total_t/max(1, N-50):.2f}s')
print(f'Rejection stats: {json.dumps(rejection)}')
