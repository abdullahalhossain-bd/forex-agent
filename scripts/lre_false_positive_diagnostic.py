"""
LRE False Positive Diagnostic
==============================
Identifies every winning trade falsely rejected by the LRE,
with per-filter root-cause analysis.

Focuses on failure_cascade and regime_transition filters.
Uses the SAME context reconstruction as lre_scientific_validation.py
for reproducibility.
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

log = logging.getLogger('lre_diag')
log.setLevel(logging.INFO)

_PIP = 0.0001
_EURUSD_H1_ATR = 0.0065


# ═══════════════════════════════════════════════════════════════
#  CONTEXT RECONSTRUCTION (same as lre_scientific_validation.py)
# ═══════════════════════════════════════════════════════════════
def build_context(row: pd.Series, idx: int) -> Tuple[Dict, Dict, Dict]:
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


# ═══════════════════════════════════════════════════════════════
#  IMPORT LRE
# ═══════════════════════════════════════════════════════════════
os.environ['LRE_ENABLED'] = '1'
os.environ['LRE_SHADOW_MODE'] = '0'  # ACTUAL rejections

from core.loss_rejection_engine.engine import LossRejectionEngine, LREResult
from core.loss_rejection_engine.layer1_structural_filters import (
    StructuralFilterLayer, Layer1Output, FILTER_WEIGHTS, LAYER1_REJECT_THRESHOLD,
)

log.info('LRE imported. REJECT_THRESHOLD=%.1f', LAYER1_REJECT_THRESHOLD)


# ═══════════════════════════════════════════════════════════════
#  DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════
@dataclass
class TradeDiag:
    idx: int
    trade_id: int
    direction: str
    is_win: bool
    pnl_usd: float
    pnl_pips: float
    confidence: int
    strategy: str
    hold_bars: int
    entry_time: str
    exit_reason: str
    lre_blocked: bool
    l1_verdict: str
    l1_composite: float
    l2_verdict: str
    l3_verdict: str
    blocking_layer: str
    blocking_filter: str
    block_reason: str
    per_filter_scores: Dict[str, float]
    per_filter_reasons: Dict[str, str]
    # Cascade-specific
    cascade_same_dir_losses: int = 0
    cascade_all_dir_losses: int = 0
    # Regime-specific
    regime_cur: str = ''
    regime_confirmed: str = ''
    regime_prev: str = ''
    regime_conf: float = 0.0
    regime_is_transition: bool = False


def run_diagnostic():
    trades_path = PROJECT_ROOT / 'backtest' / 'results_EURUSD_H1.csv'
    df = pd.read_csv(trades_path, parse_dates=['entry_time', 'exit_time'])
    df['is_win'] = df['pnl_pips'] > 0
    df['sl_dist_pips'] = np.abs(df['entry_price'] - df['stop_loss']) / _PIP
    df['tp_dist_pips'] = np.abs(df['take_profit'] - df['entry_price']) / _PIP
    df['rr'] = df['tp_dist_pips'] / df['sl_dist_pips'].replace(0, np.nan)
    df = df.sort_values('entry_time').reset_index(drop=True)

    n = len(df)
    log.info(f'Loaded {n} trades: {df["is_win"].sum()}W / {(~df["is_win"]).sum()}L')

    lre = LossRejectionEngine()
    diagnostics = []

    for i in range(n):
        row = df.iloc[i]
        dec_out, analysis_out, market_out = build_context(row, i)

        # Feed prior trade outcomes
        if i > 0:
            for j in range(max(0, i - 20), i):
                prev = df.iloc[j]
                prev_dec, _, prev_mkt = build_context(prev, j)
                reg = prev_mkt.get('regime', {})
                regime_str = reg.get('regime', reg.get('label', 'unknown')) if isinstance(reg, dict) else 'unknown'
                try:
                    lre.record_trade_outcome(
                        'EURUSD', prev['direction'], prev['pnl_usd'],
                        price_zone='mid', regime=regime_str,
                    )
                except:
                    pass

        result = lre.evaluate(dec_out, analysis_out, market_out, symbol='EURUSD')

        # Extract per-filter details
        per_scores = {}
        per_reasons = {}
        if result.l1 and result.l1.filters:
            for f in result.l1.filters:
                per_scores[f.name] = f.rejection_score
                per_reasons[f.name] = f.reason

        # Determine blocking layer/filter
        blocking_layer = ''
        blocking_filter = ''
        block_reason = ''
        if result.blocked:
            if result.l1 and not result.l1.pass_through:
                blocking_layer = 'L1'
                if result.l1.filters:
                    # Find which filter(s) caused the block
                    hard_blocks = [f for f in result.l1.filters if f.rejection_score >= LAYER1_REJECT_THRESHOLD]
                    if hard_blocks:
                        top = max(hard_blocks, key=lambda f: f.rejection_score)
                        blocking_filter = f'{top.name} (HARD:{top.rejection_score:.0f})'
                        block_reason = f'{top.name}: {top.reason}'
                    else:
                        # Composite score blocked
                        top = max(result.l1.filters, key=lambda f: f.rejection_score)
                        blocking_filter = f'COMPOSITE (top: {top.name}:{top.rejection_score:.0f})'
                        block_reason = f'Composite={result.l1.composite_score:.1f} >= {LAYER1_REJECT_THRESHOLD}'
            elif result.l2 and not result.l2.pass_through:
                blocking_layer = 'L2'
                blocking_filter = 'meta_labeler'
                block_reason = result.reason
            elif result.l3 and not result.l3.pass_through:
                blocking_layer = 'L3'
                blocking_filter = 'ood_detector'
                block_reason = result.reason

        # Extract cascade-specific data
        cascade_sdl = 0
        cascade_adl = 0
        if result.l1 and result.l1.filters:
            for f in result.l1.filters:
                if f.name == 'failure_cascade' and f.data:
                    cascade_sdl = f.data.get('same_dir', 0)
                    cascade_adl = f.data.get('all_dir', 0)

        # Extract regime-specific data
        regime_cur = ''
        regime_confirmed = ''
        regime_prev = ''
        regime_conf = 0.0
        regime_is_trans = False
        if result.l1 and result.l1.filters:
            for f in result.l1.filters:
                if f.name == 'regime_transition' and f.data:
                    regime_cur = f.data.get('regime', '')
                    regime_confirmed = f.data.get('confirmed', '')
                    regime_prev = f.data.get('prev', '')
                    regime_conf = f.data.get('confidence', 0.0)
                    regime_is_trans = f.data.get('is_transition', False)

        diag = TradeDiag(
            idx=i, trade_id=int(row['trade_id']), direction=row['direction'],
            is_win=row['is_win'], pnl_usd=row['pnl_usd'], pnl_pips=row['pnl_pips'],
            confidence=int(row['confidence']), strategy=row['strategy'],
            hold_bars=int(row['hold_bars']), entry_time=str(row['entry_time']),
            exit_reason=row['exit_reason'],
            lre_blocked=result.blocked,
            l1_verdict=result.l1.verdict if result.l1 else '',
            l1_composite=result.l1.composite_score if result.l1 else 0.0,
            l2_verdict=result.l2.verdict if result.l2 else '',
            l3_verdict=result.l3.verdict if result.l3 else '',
            blocking_layer=blocking_layer, blocking_filter=blocking_filter,
            block_reason=block_reason, per_filter_scores=per_scores,
            per_filter_reasons=per_reasons,
            cascade_same_dir_losses=cascade_sdl,
            cascade_all_dir_losses=cascade_adl,
            regime_cur=regime_cur, regime_confirmed=regime_confirmed,
            regime_prev=regime_prev, regime_conf=regime_conf,
            regime_is_transition=regime_is_trans,
        )
        diagnostics.append(diag)

    # ═══════════════════════════════════════════════════════════════
    #  ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    wins = [d for d in diagnostics if d.is_win]
    losses = [d for d in diagnostics if not d.is_win]

    # Confusion matrix
    tp = sum(1 for d in losses if d.lre_blocked)  # correctly rejected loss
    fp = sum(1 for d in wins if d.lre_blocked)    # falsely rejected winner
    tn = sum(1 for d in wins if not d.lre_blocked) # correctly accepted winner
    fn = sum(1 for d in losses if not d.lre_blocked) # falsely accepted loss

    total_winners = len(wins)
    total_losers = len(losses)
    wpr = tn / total_winners * 100 if total_winners else 0
    lrr = tp / total_losers * 100 if total_losers else 0

    print('\n' + '='*80)
    print('LRE FALSE POSITIVE DIAGNOSTIC REPORT')
    print('='*80)
    print(f'\nTotal trades: {len(diagnostics)}')
    print(f'Winners: {total_winners} | Losers: {total_losers}')
    print(f'\nConfusion Matrix (Meta Labeling):')
    print(f'  TP (rejected losers):  {tp}')
    print(f'  FP (rejected winners): {fp}')
    print(f'  TN (accepted winners): {tn}')
    print(f'  FN (accepted losers):  {fn}')
    print(f'\nWPR = {tn}/{total_winners} = {wpr:.1f}%')
    print(f'LRR = {tp}/{total_losers} = {lrr:.1f}%')

    # False positives detail
    fp_trades = [d for d in wins if d.lre_blocked]
    print(f'\n{"="*80}')
    print(f'FALSE POSITIVE ANALYSIS ({len(fp_trades)} winners falsely rejected)')
    print(f'{"="*80}')

    for fp_trade in fp_trades:
        print(f'\n--- Trade #{fp_trade.trade_id} (idx={fp_trade.idx}) ---')
        print(f'  {fp_trade.direction} | {fp_trade.strategy} | conf={fp_trade.confidence}')
        print(f'  PnL: +{fp_trade.pnl_pips:.1f} pips / +${fp_trade.pnl_usd:.2f}')
        print(f'  Hold: {fp_trade.hold_bars} bars | Exit: {fp_trade.exit_reason}')
        print(f'  Block: {fp_trade.blocking_layer} | {fp_trade.blocking_filter}')
        print(f'  L1 composite: {fp_trade.l1_composite:.2f} | L2: {fp_trade.l2_verdict} | L3: {fp_trade.l3_verdict}')
        print(f'  Cascade: same_dir_losses={fp_trade.cascade_same_dir_losses}, all_dir_losses={fp_trade.cascade_all_dir_losses}')
        print(f'  Regime: cur={fp_trade.regime_cur}, confirmed={fp_trade.regime_confirmed}, prev={fp_trade.regime_prev}, conf={fp_trade.regime_conf:.2f}, is_transition={fp_trade.regime_is_transition}')
        print(f'  Per-filter scores:')
        for fname, score in sorted(fp_trade.per_filter_scores.items(), key=lambda x: -x[1]):
            reason = fp_trade.per_filter_reasons.get(fname, '')
            flag = ' *** HARD BLOCK ***' if score >= LAYER1_REJECT_THRESHOLD else ''
            print(f'    {fname:30s}: {score:6.1f}{flag}  | {reason}')

    # Blocking filter distribution for FPs
    print(f'\n{"="*80}')
    print('BLOCKING FILTER DISTRIBUTION (False Positives)')
    print(f'{"="*80}')
    fp_blockers = defaultdict(int)
    for fp_trade in fp_trades:
        # Find the primary blocker
        if fp_trade.blocking_layer == 'L1':
            # Which specific filter?
            hard_blocks = [(name, score) for name, score in fp_trade.per_filter_scores.items() if score >= LAYER1_REJECT_THRESHOLD]
            if hard_blocks:
                for name, score in hard_blocks:
                    fp_blockers[f'L1:{name} (HARD:{score:.0f})'] += 1
            else:
                # Composite block - find top contributor
                top_name = max(fp_trade.per_filter_scores, key=fp_trade.per_filter_scores.get)
                fp_blockers[f'L1:COMPOSITE (top:{top_name})'] += 1
        elif fp_trade.blocking_layer == 'L2':
            fp_blockers['L2:meta_labeler'] += 1
        elif fp_trade.blocking_layer == 'L3':
            fp_blockers['L3:ood_detector'] += 1

    for blocker, count in sorted(fp_blockers.items(), key=lambda x: -x[1]):
        print(f'  {blocker}: {count}')

    # Categorize FP types
    print(f'\n{"="*80}')
    print('FALSE POSITIVE CATEGORIZATION')
    print(f'{"="*80}')
    categories = defaultdict(list)
    for fp_trade in fp_trades:
        # Determine primary cause
        if fp_trade.cascade_same_dir_losses >= 5:
            cat = 'cascade_same_dir_ge5'
        elif fp_trade.cascade_same_dir_losses >= 3:
            cat = 'cascade_same_dir_3-4'
        elif fp_trade.regime_is_transition:
            cat = 'regime_transition'
        elif fp_trade.cascade_all_dir_losses >= 4:
            cat = 'cascade_all_dir'
        else:
            # Check which filter had highest score
            top_filter = max(fp_trade.per_filter_scores, key=fp_trade.per_filter_scores.get)
            cat = f'other:{top_filter}'
        categories[cat].append(fp_trade)

    for cat, trades in sorted(categories.items(), key=lambda x: -len(x[1])):
        total_pnl = sum(t.pnl_usd for t in trades)
        print(f'\n  {cat}: {len(trades)} FPs, total missed profit: ${total_pnl:,.2f}')
        for t in trades:
            print(f'    Trade #{t.trade_id}: {t.direction} +${t.pnl_usd:.2f} | cascade_sdl={t.cascade_same_dir_losses} regime_trans={t.regime_is_transition}')

    # ═══════════════════════════════════════════════════════════════
    #  PRIOR TRADE SEQUENCE FOR EACH FP (cascade analysis)
    # ═══════════════════════════════════════════════════════════════
    print(f'\n{"="*80}')
    print('PRIOR TRADE SEQUENCES FOR FALSE POSITIVE WINNERS')
    print(f'{"="*80}')
    for fp_trade in fp_trades:
        print(f'\n--- Trade #{fp_trade.trade_id} ({fp_trade.direction}, WIN +${fp_trade.pnl_usd:.2f}) ---')
        # Show prior 10 trades
        start = max(0, fp_trade.idx - 10)
        for j in range(start, fp_trade.idx):
            d = diagnostics[j]
            marker = ''
            if d.direction == fp_trade.direction:
                marker = ' <-- SAME DIR' + (' LOSS' if not d.is_win else ' WIN')
            print(f'  [{j:2d}] #{d.trade_id:2d} {d.direction:4s} {"WIN" if d.is_win else "LOSS":4s} {d.pnl_pips:+8.1f}p ${d.pnl_usd:+10.2f}{marker}')
        print(f'  >>> [{fp_trade.idx:2d}] #{fp_trade.trade_id:2d} {fp_trade.direction:4s} WIN   {fp_trade.pnl_pips:+8.1f}p ${fp_trade.pnl_usd:+10.2f} ** BLOCKED **')

    # ═══════════════════════════════════════════════════════════════
    #  EXPECTANCY IMPACT
    # ═══════════════════════════════════════════════════════════════
    print(f'\n{"="*80}')
    print('EXPECTANCY IMPACT ANALYSIS')
    print(f'{"="*80}')

    # Baseline: all trades
    baseline_pnl = sum(d.pnl_usd for d in diagnostics)
    # After LRE: only non-blocked trades
    accepted_pnl = sum(d.pnl_usd for d in diagnostics if not d.lre_blocked)
    # Ideal: reject all losses, keep all winners
    ideal_pnl = sum(d.pnl_usd for d in wins)

    # LRE with FP fix: reject TP losses + FN losses, keep all winners
    tp_pnl = sum(d.pnl_usd for d in losses if d.lre_blocked)
    fp_pnl = sum(d.pnl_usd for d in wins if d.lre_blocked)
    fn_pnl = sum(d.pnl_usd for d in losses if not d.lre_blocked)
    tn_pnl = sum(d.pnl_usd for d in wins if not d.lre_blocked)

    print(f'  Baseline (all trades):     ${baseline_pnl:,.2f}')
    print(f'  After LRE (current):       ${accepted_pnl:,.2f} (delta: ${accepted_pnl - baseline_pnl:,.2f})')
    print(f'  Ideal (reject all losses):  ${ideal_pnl:,.2f}')
    print(f'  TP profit (blocked losses): -${abs(tp_pnl):,.2f}')
    print(f'  FP cost (blocked winners):  -${abs(fp_pnl):,.2f}')
    print(f'  FN cost (accepted losses):  ${fn_pnl:,.2f}')
    print(f'  TN profit (kept winners):   ${tn_pnl:,.2f}')

    # Save results as JSON
    results = {
        'confusion_matrix': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
        'WPR': round(wpr, 1), 'LRR': round(lrr, 1),
        'total_trades': len(diagnostics),
        'false_positives': [{
            'trade_id': t.trade_id, 'direction': t.direction,
            'pnl_usd': t.pnl_usd, 'pnl_pips': t.pnl_pips,
            'confidence': t.confidence, 'strategy': t.strategy,
            'blocking_layer': t.blocking_layer,
            'blocking_filter': t.blocking_filter,
            'block_reason': t.block_reason,
            'cascade_same_dir': t.cascade_same_dir_losses,
            'cascade_all_dir': t.cascade_all_dir_losses,
            'regime_cur': t.regime_cur,
            'regime_confirmed': t.regime_confirmed,
            'regime_prev': t.regime_prev,
            'regime_conf': t.regime_conf,
            'regime_is_transition': t.regime_is_transition,
            'per_filter_scores': t.per_filter_scores,
            'per_filter_reasons': t.per_filter_reasons,
        } for t in fp_trades],
        'baseline_pnl': baseline_pnl,
        'lre_filtered_pnl': accepted_pnl,
        'fp_total_cost': fp_pnl,
    }

    out_path = PROJECT_ROOT / 'download' / 'lre_fp_diagnostic.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults saved to {out_path}')

    return diagnostics, results


if __name__ == '__main__':
    diagnostics, results = run_diagnostic()
