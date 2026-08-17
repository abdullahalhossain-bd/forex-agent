"""
Module ablation: vary key parameters to see how baseline responds.
NOT optimization — just sensitivity analysis.
"""
import sys, os, json
sys.path.insert(0, '/home/z/my-project/forex-agent')
os.chdir('/home/z/my-project/forex-agent')
os.environ['TEST_MODE'] = 'false'
os.environ['SIMULATION_MODE'] = 'true'
import logging
logging.getLogger().setLevel(logging.ERROR)

# Re-use the baseline script's functions
sys.path.insert(0, '/home/z/my-project/scripts')
from baseline_backtest import load_csv_raw, compute_minimal_indicators, backtest_pair, compute_metrics
import pandas as pd

pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF']
timeframes = ['H1']  # focus on H1 for speed

scenarios = [
    {'name': 'baseline_atr_1.5_rr_2.0', 'atr_sl_mult': 1.5, 'rr': 2.0, 'risk_pct': 0.01},
    {'name': 'tight_atr_1.0', 'atr_sl_mult': 1.0, 'rr': 2.0, 'risk_pct': 0.01},
    {'name': 'wide_atr_2.5', 'atr_sl_mult': 2.5, 'rr': 2.0, 'risk_pct': 0.01},
    {'name': 'low_rr_1.5', 'atr_sl_mult': 1.5, 'rr': 1.5, 'risk_pct': 0.01},
    {'name': 'high_rr_3.0', 'atr_sl_mult': 1.5, 'rr': 3.0, 'risk_pct': 0.01},
    {'name': 'low_risk_0.5pct', 'atr_sl_mult': 1.5, 'rr': 2.0, 'risk_pct': 0.005},
    {'name': 'high_risk_2pct', 'atr_sl_mult': 1.5, 'rr': 2.0, 'risk_pct': 0.02},
]

all_results = {}
for scenario in scenarios:
    name = scenario['name']
    print(f'\n=== Scenario: {name} ===', flush=True)
    total_trades = 0
    total_pnl = 0
    total_wins = 0
    pair_results = {}
    for pair in pairs:
        for tf in timeframes:
            path = f'data/{pair}_{tf}.csv'
            if not os.path.exists(path):
                continue
            df = load_csv_raw(path)
            df = compute_minimal_indicators(df)
            df = df.tail(min(len(df), 5000))
            r = backtest_pair(
                df, pair_name=pair,
                risk_pct=scenario['risk_pct'], rr=scenario['rr'],
                atr_sl_mult=scenario['atr_sl_mult'],
            )
            m = compute_metrics(r['trades'])
            total_trades += m['total_trades']
            total_pnl += m['net_pnl']
            total_wins += m.get('wins', 0)
            pair_results[f'{pair}_{tf}'] = m['total_trades']
    overall_wr = total_wins / total_trades * 100 if total_trades else 0
    print(f'  Trades={total_trades} WR={overall_wr:.1f}% P&L=${total_pnl:.2f}', flush=True)
    all_results[name] = {
        'trades': total_trades,
        'win_rate': overall_wr,
        'net_pnl': total_pnl,
        'wins': total_wins,
        'per_pair': pair_results,
        'params': scenario,
    }

print(f'\n\n{"="*80}')
print(f'  ABLATION SUMMARY (sensitivity analysis, NOT optimization)')
print(f'{"="*80}')
print(f'  {"Scenario":<30} {"Trades":>8} {"WR%":>7} {"Net$":>10}')
for name, r in all_results.items():
    print(f'  {name:<30} {r["trades"]:>8} {r["win_rate"]:>6.1f}% {r["net_pnl"]:>9.2f}')

with open('/home/z/my-project/download/ablation_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f'\nSaved: /home/z/my-project/download/ablation_results.json')
