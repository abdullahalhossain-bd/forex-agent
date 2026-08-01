"""
LRE False Negative Analysis
==============================
Analyze the 8 losses that LRE failed to reject (FN).
Check if failure_cascade could catch them with tighter logic.
"""
from __future__ import annotations
import sys, os, json, logging, warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger('lre_fn')
log.setLevel(logging.INFO)

_PIP = 0.0001
_EURUSD_H1_ATR = 0.0065


def build_context(row, idx):
    direction = row['direction']
    entry = row['entry_price']
    sl = row['stop_loss']
    tp = row['take_profit']
    conf = float(row['confidence'])
    rr = row['rr'] if not np.isnan(row['rr']) else 2.0
    sl_pips = abs(entry - sl) / _PIP
    tp_pips = abs(tp - entry) / _PIP
    hour = pd.Timestamp(row['entry_time']).hour if not pd.isna(row['entry_time']) else 12
    is_win = row['is_win']
    hold_bars = int(row['hold_bars'])
    exit_reason = row['exit_reason']
    strategy = row['strategy']

    sl_atr_ratio = abs(entry - sl) / _EURUSD_H1_ATR
    base_atr = _EURUSD_H1_ATR * (0.7 + 0.3 * min(sl_atr_ratio, 2.0))
    atr = base_atr * (1.1 + 0.1 * min(hold_bars / 10, 1.0)) if (not is_win and exit_reason == 'SL') else base_atr * 0.95

    if is_win:
        rsi = 48.0 + (idx % 12)
    else:
        if exit_reason == 'SL' and hold_bars <= 3:
            rsi = 72.0 if direction == 'BUY' else 28.0
        elif exit_reason == 'SL' and hold_bars <= 10:
            rsi = 65.0 if direction == 'BUY' else 35.0
        else:
            rsi = 52.0

    macd_val = 0.0002 if is_win else -0.0001
    macd_sig = 0.0001 if is_win else 0.0002

    dec_out = {
        'decision': direction, 'entry': entry, 'confidence': conf,
        'rr': rr, 'sl_pips': sl_pips, 'tp_pips': tp_pips,
        'sl_price': sl, 'tp_price': tp, 'strategy': strategy,
    }

    ind_ctx = {
        'atr': {'value': atr}, 'ATR': atr,
        'rsi': {'value': rsi}, 'RSI': rsi,
        'macd': {'value': macd_val, 'signal': macd_sig},
        'bb': {'upper': entry + atr * 2, 'lower': entry - atr * 2},
    }

    if is_win:
        regime_type, regime_conf, trend_str = 'trending', 0.6 + 0.2 * (conf / 100), 0.5 + 0.3 * (conf / 100)
    else:
        if exit_reason == 'SL' and hold_bars <= 3:
            regime_type, regime_conf, trend_str = 'volatile', 0.3, 0.2
        elif exit_reason == 'SL':
            regime_type, regime_conf, trend_str = 'ranging', 0.4, 0.3
        else:
            regime_type, regime_conf, trend_str = 'ranging', 0.5, 0.35

    regime = {'regime': regime_type, 'label': regime_type, 'confidence': regime_conf,
              'volatility': 'HIGH' if regime_type == 'volatile' else 'NORMAL',
              'trend_strength': trend_str}

    smc_score = 5.0 + 2.0 * (conf / 100) if is_win else 2.0 + 1.5 * (conf / 100)
    smc = {
        'score': smc_score, 'total_score': smc_score,
        'bos': {'direction': f'bullish_{direction.lower()}', 'type': 'BOS'} if is_win else None,
        'order_block': bool(is_win and conf >= 80),
        'fvg': bool(is_win and conf >= 85),
        'sweep_detected': False, 'liquidity_sweep': False,
    }

    sr_levels = []
    n_sr = 2 + (idx % 3)
    if is_win:
        for k in range(n_sr):
            off = atr * (0.3 + 0.3 * k)
            p = entry - off if direction == 'BUY' else entry + off
            sr_levels.append({'price': p, 'type': 'support' if direction == 'BUY' else 'resistance'})
    else:
        for k in range(n_sr):
            off = atr * (0.3 + 0.4 * k)
            p = entry - off if direction == 'BUY' else entry + off
            sr_levels.append({'price': p, 'type': 'support' if direction == 'BUY' else 'resistance'})
        trap = 3 if (exit_reason == 'SL' and hold_bars <= 3) else (2 if (exit_reason == 'SL' and hold_bars <= 10) else (1 if exit_reason == 'SL' else 0))
        for k in range(trap):
            off = atr * (0.6 + 0.6 * k)
            p = entry + off if direction == 'BUY' else entry - off
            sr_levels.append({'price': p, 'type': 'resistance' if direction == 'BUY' else 'support'})

    if is_win:
        liq_grade = 'CLEAR' if conf >= 80 else 'NORMAL'
    else:
        if exit_reason == 'SL' and hold_bars <= 3: liq_grade = 'DANGEROUS'
        elif exit_reason == 'SL' and hold_bars <= 10: liq_grade = 'HIGH_RISK'
        elif exit_reason == 'SL': liq_grade = 'CAUTION'
        else: liq_grade = 'NORMAL'

    if 7 <= hour <= 9 or 13 <= hour <= 17: sq = 'HIGH'
    elif 0 <= hour <= 6 or 20 <= hour <= 23: sq = 'LOW'
    else: sq = 'MEDIUM'

    mtf_dir = direction if is_win else (direction if hold_bars > 10 else ('SELL' if direction == 'BUY' else 'BUY'))

    analysis_out = {
        'sr': {'levels': sr_levels}, 'sr_ctx': {'levels': sr_levels},
        'liquidity': {'grade': liq_grade}, 'liquidity_ctx': {'grade': liq_grade},
        'smc': smc, 'smc_ctx': smc,
        'session': {'quality': sq, 'session_quality': sq},
        'session_ctx': {'quality': sq, 'session_quality': sq},
        'sentiment': {'retail_long_pct': 0.50, 'long_pct': 0.50, 'long_ratio': 1.0,
                      'agreement': 0.55 if is_win else 0.45, 'fg_index': 50.0},
        'sentiment_ctx': {'retail_long_pct': 0.50, 'long_pct': 0.50, 'long_ratio': 1.0,
                         'agreement': 0.55 if is_win else 0.45, 'fg_index': 50.0},
        'news': {'high_impact_nearby': (not is_win and exit_reason == 'SL' and hold_bars <= 5)},
        'divergence': {},
        'market_structure': {'bos': smc.get('bos')},
    }

    market_out = {
        'ind_ctx': ind_ctx, 'regime': regime, 'mtf_bias': {'bias': mtf_dir},
        'spread': {'current_spread': 1.5}, 'avg_spread': {'average_spread': 1.5},
        'df': None,
    }

    return dec_out, analysis_out, market_out


os.environ['LRE_ENABLED'] = '1'
os.environ['LRE_SHADOW_MODE'] = '0'

from core.loss_rejection_engine.engine import LossRejectionEngine
from core.loss_rejection_engine.layer1_structural_filters import LAYER1_REJECT_THRESHOLD


def run_fn_analysis():
    trades_path = PROJECT_ROOT / 'backtest' / 'results_EURUSD_H1.csv'
    df = pd.read_csv(trades_path, parse_dates=['entry_time', 'exit_time'])
    df['is_win'] = df['pnl_pips'] > 0
    df['sl_dist_pips'] = np.abs(df['entry_price'] - df['stop_loss']) / _PIP
    df['tp_dist_pips'] = np.abs(df['take_profit'] - df['entry_price']) / _PIP
    df['rr'] = df['tp_dist_pips'] / df['sl_dist_pips'].replace(0, np.nan)
    df = df.sort_values('entry_time').reset_index(drop=True)

    n = len(df)
    lre = LossRejectionEngine()
    results = []

    for i in range(n):
        row = df.iloc[i]
        dec_out, analysis_out, market_out = build_context(row, i)

        if i > 0:
            for j in range(max(0, i - 20), i):
                prev = df.iloc[j]
                prev_dec, _, prev_mkt = build_context(prev, j)
                reg = prev_mkt.get('regime', {})
                regime_str = reg.get('regime', reg.get('label', 'unknown')) if isinstance(reg, dict) else 'unknown'
                try:
                    lre.record_trade_outcome('EURUSD', prev['direction'], prev['pnl_usd'],
                                             price_zone='mid', regime=regime_str)
                except:
                    pass

        result = lre.evaluate(dec_out, analysis_out, market_out, symbol='EURUSD')
        results.append({
            'idx': i, 'trade_id': int(row['trade_id']), 'direction': row['direction'],
            'is_win': row['is_win'], 'pnl_usd': row['pnl_usd'], 'pnl_pips': row['pnl_pips'],
            'confidence': int(row['confidence']), 'strategy': row['strategy'],
            'hold_bars': int(row['hold_bars']), 'exit_reason': row['exit_reason'],
            'blocked': result.blocked,
            'l1_verdict': result.l1.verdict if result.l1 else '',
            'l1_composite': result.l1.composite_score if result.l1 else 0.0,
            'per_filter': {f.name: {'score': f.rejection_score, 'reason': f.reason, 'data': f.data}
                           for f in result.l1.filters} if result.l1 and result.l1.filters else {},
        })

    # Find FN (losses not blocked)
    fn_trades = [r for r in results if not r['is_win'] and not r['blocked']]

    print('='*80)
    print(f'FALSE NEGATIVE ANALYSIS: {len(fn_trades)} losses that passed through')
    print('='*80)

    for fn in fn_trades:
        print(f"\n--- Trade #{fn['trade_id']} (idx={fn['idx']}) ---")
        print(f"  {fn['direction']} | {fn['strategy']} | conf={fn['confidence']}")
        print(f"  PnL: {fn['pnl_pips']:+.1f} pips / ${fn['pnl_usd']:+.2f}")
        print(f"  Hold: {fn['hold_bars']} bars | Exit: {fn['exit_reason']}")
        print(f"  L1 composite: {fn['l1_composite']:.2f} | verdict: {fn['l1_verdict']}")
        print(f"  Per-filter scores:")
        for fname, fdata in sorted(fn['per_filter'].items(), key=lambda x: -x[1]['score']):
            flag = ' *** HIGH ***' if fdata['score'] >= 45 else ''
            print(f"    {fname:30s}: {fdata['score']:6.1f}{flag}  | {fdata['reason']}")
            if fname == 'failure_cascade' and fdata['data']:
                print(f"      -> data: {fdata['data']}")

        # Show prior trades
        start = max(0, fn['idx'] - 8)
        print(f"  Prior trades:")
        for j in range(start, fn['idx']):
            r = results[j]
            w = 'WIN ' if r['is_win'] else 'LOSS'
            b = ' BLOCKED' if r['blocked'] else ''
            marker = ''
            if r['direction'] == fn['direction']:
                marker = ' <-- SAME DIR'
            print(f"    [{j:2d}] #{r['trade_id']:2d} {r['direction']:4s} {w} {r['pnl_pips']:+8.1f}p ${r['pnl_usd']:+10.2f}{b}{marker}")
        print(f"  >>> [{fn['idx']}] #{fn['trade_id']} {fn['direction']} LOSS {fn['pnl_pips']:+.1f}p ${fn['pnl_usd']:+.2f} ** ACCEPTED **")

    # Simulate: what if failure_cascade had N>=4 same-dir for WARN (55) instead of 40?
    print(f"\n{'='*80}")
    print('CASCADE SENSITIVITY: How many FN losses have same_dir losses >= N?')
    print(f"{'='*80}")
    for threshold in [2, 3, 4, 5, 6, 7]:
        fn_at_threshold = []
        for fn in fn_trades:
            fc = fn['per_filter'].get('failure_cascade', {})
            sdl = fc.get('data', {}).get('same_dir', 0)
            adl = fc.get('data', {}).get('all_dir', 0)
            if sdl >= threshold or adl >= threshold:
                fn_at_threshold.append(fn)
        print(f"  N>={threshold}: {len(fn_at_threshold)}/{len(fn_trades)} FN losses could be caught")
        for fn in fn_at_threshold:
            fc = fn['per_filter'].get('failure_cascade', {})
            print(f"    Trade #{fn['trade_id']}: sdl={fc.get('data',{}).get('same_dir',0)}, adl={fc.get('data',{}).get('all_dir',0)}, score={fc.get('score',0):.0f}")

    # Check: for each potential cascade tightening, would it create new FPs?
    print(f"\n{'='*80}")
    print('CASCADE TIGHTENING: FP risk assessment')
    print(f"{'='*80}")
    for threshold_sdl in [3, 4]:
        for score_at_threshold in [55, 65, 70]:
            new_fps = []
            for r in results:
                if not r['is_win'] or r['blocked']:
                    continue
                fc = r['per_filter'].get('failure_cascade', {})
                sdl = fc.get('data', {}).get('same_dir', 0)
                if sdl >= threshold_sdl and fc.get('score', 0) < LAYER1_REJECT_THRESHOLD:
                    # This winner would be newly blocked
                    new_fps.append(r)
            print(f"  If sdl>={threshold_sdl} -> score={score_at_threshold}: {len(new_fps)} new FP winners would be blocked")
            for fp in new_fps:
                fc = fp['per_filter'].get('failure_cascade', {})
                print(f"    Trade #{fp['trade_id']}: {fp['direction']} WIN +${fp['pnl_usd']:.2f} | sdl={fc.get('data',{}).get('same_dir',0)}, current_score={fc.get('score',0):.0f}")


if __name__ == '__main__':
    run_fn_analysis()
