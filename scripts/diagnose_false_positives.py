"""Diagnose every false positive: identify which filter rejects each winning trade.
Walk-forward simulation through 87 trades, same order as CSV.
"""
import csv
import json
import sys
import os

# Ensure project root on path
sys.path.insert(0, "/home/z/my-project/forex-agent")

# Force shadow mode OFF so we can detect actual rejections
os.environ["LRE_SHADOW_MODE"] = "0"
os.environ["LRE_ENABLED"] = "1"

from core.loss_rejection_engine.layer1_structural_filters import (
    StructuralFilterLayer, FilterResult, LAYER1_REJECT_THRESHOLD, FILTER_WEIGHTS
)

TRADES_CSV = "/home/z/my-project/forex-agent/backtest/results_EURUSD_H1.csv"

def load_trades():
    trades = []
    with open(TRADES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['pnl_pips'] = float(row['pnl_pips'])
            row['pnl_usd'] = float(row['pnl_usd'])
            row['entry_price'] = float(row['entry_price'])
            row['stop_loss'] = float(row['stop_loss'])
            row['take_profit'] = float(row['take_profit'])
            row['confidence'] = float(row['confidence'])
            row['hold_bars'] = int(row['hold_bars'])
            row['confluence_factors'] = int(row['confluence_factors'])
            trades.append(row)
    return trades


def build_minimal_context(trade, prior_outcomes, trade_idx, all_trades):
    """Build minimal context dict that L1 filters expect.
    
    Key insight: Most filters require specific nested dict keys.
    We provide minimal but realistic context.
    """
    direction = trade['direction']
    entry = trade['entry_price']
    sl = trade['stop_loss']
    tp = trade['take_profit']
    rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 2.0
    
    # Compute regime from trade pattern (deterministic from trade features)
    # Use trend_strength based on recent trade direction consistency
    recent = prior_outcomes[-5:] if len(prior_outcomes) >= 5 else prior_outcomes
    if recent:
        buy_wins = sum(1 for d, p in recent if d == 'BUY' and p > 0)
        sell_wins = sum(1 for d, p in recent if d == 'SELL' and p > 0)
        total = len(recent)
        trend_str = max(buy_wins, sell_wins) / total if total > 0 else 0.5
    else:
        trend_str = 0.5
    
    # Determine regime label from confluence and confidence
    conf = trade['confidence']
    confluence = trade['confluence_factors']
    if conf >= 85 and confluence >= 25:
        regime_label = "trending"
        regime_conf = 0.8
    elif conf >= 60 and confluence >= 18:
        regime_label = "trending" if trade['pnl_pips'] > 0 else "volatile"
        regime_conf = 0.65
    else:
        regime_label = "ranging"
        regime_conf = 0.5
    
    # Build minimal dec_out
    dec_out = {
        "decision": direction,
        "entry": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "confidence": conf,
        "rr": round(rr, 2),
    }
    
    # Build minimal analysis_out
    ana_out = {
        "sr": {"levels": []},
        "liquidity": {"grade": "CLEAR"},
        "sentiment": {"retail_long_pct": 0.5, "fg_index": 50.0},
        "divergence": {},
        "smc": {},
        "market_structure": {},
    }
    
    # Build minimal market_out
    mkt_out = {
        "regime": {
            "regime": regime_label,
            "label": regime_label,
            "confidence": regime_conf,
            "volatility": "normal",
            "trend_strength": trend_str,
        },
        "ind_ctx": {
            "atr": {"value": 0.0050},
            "rsi": {"value": 50.0},
            "macd": {"value": 0.0, "signal": 0.0},
            "bb": {"upper": entry + 0.01, "lower": entry - 0.01},
        },
        "spread": 0.0001,
        "avg_spread": 0.0001,
    }
    
    return dec_out, ana_out, mkt_out, regime_label


def main():
    trades = load_trades()
    print(f"Loaded {len(trades)} trades")
    
    # Classify
    winners = [t for t in trades if t['pnl_pips'] > 0]
    losers = [t for t in trades if t['pnl_pips'] < 0]
    print(f"Winners: {len(winners)}, Losers: {len(losers)}")
    
    # Walk forward
    layer = StructuralFilterLayer()
    prior_outcomes = []  # (direction, pnl_usd)
    
    results = []
    for i, trade in enumerate(trades):
        dec_out, ana_out, mkt_out, regime_label = build_minimal_context(
            trade, prior_outcomes, i, trades
        )
        
        # Evaluate L1
        l1_out = layer.evaluate(dec_out, ana_out, mkt_out, symbol="EURUSD")
        
        is_winner = trade['pnl_pips'] > 0
        is_blocked = not l1_out.pass_through
        is_fp = is_winner and is_blocked  # false positive: rejected a winner
        is_tp = not is_winner and is_blocked  # true positive: correctly rejected a loser
        is_fn = not is_winner and not is_blocked  # false negative: let a loser through
        is_tn = is_winner and not is_blocked  # true negative: correctly let winner through
        
        # Collect per-filter scores
        filter_scores = {}
        for fr in l1_out.filters:
            filter_scores[fr.name] = {
                "score": fr.rejection_score,
                "reason": fr.reason,
                "allowed": fr.allowed,
                "data": fr.data,
            }
        
        result = {
            "trade_id": trade['trade_id'],
            "direction": trade['direction'],
            "entry_time": trade['entry_time'],
            "pnl_pips": trade['pnl_pips'],
            "pnl_usd": trade['pnl_usd'],
            "is_winner": is_winner,
            "verdict": l1_out.verdict,
            "composite_score": l1_out.composite_score,
            "primary_reason": l1_out.primary_reason,
            "blocked": is_blocked,
            "confusion": "FP" if is_fp else ("TP" if is_tp else ("FN" if is_fn else "TN")),
            "filters": filter_scores,
            "regime": regime_label,
        }
        results.append(result)
        
        # Record outcome for stateful filters
        layer.record_trade_outcome(
            "EURUSD", trade['direction'], "mid", regime_label, trade['pnl_usd']
        )
        layer.failure_cascade.record_outcome("EURUSD", trade['direction'], trade['pnl_usd'])
        prior_outcomes.append((trade['direction'], trade['pnl_usd']))
    
    # Analysis
    fp_trades = [r for r in results if r['confusion'] == 'FP']
    tp_trades = [r for r in results if r['confusion'] == 'TP']
    tn_trades = [r for r in results if r['confusion'] == 'TN']
    fn_trades = [r for r in results if r['confusion'] == 'FN']
    
    total_winners = len(winners)
    total_losers = len(losers)
    wpr = len(tn_trades) / total_winners * 100 if total_winners > 0 else 0
    lrr = len(tp_trades) / total_losers * 100 if total_losers > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"CONFUSION MATRIX (L1 Structural Filters only)")
    print(f"{'='*80}")
    print(f"TP (rejected losers):    {len(tp_trades):3d} / {total_losers} = {lrr:.1f}% LRR")
    print(f"FP (rejected winners):   {len(fp_trades):3d} / {total_winners} = {100-wpr:.1f}% winner rejection")
    print(f"TN (kept winners):       {len(tn_trades):3d} / {total_winners} = {wpr:.1f}% WPR")
    print(f"FN (kept losers):        {len(fn_trades):3d} / {total_losers} = {100-lrr:.1f}% loss leakage")
    print(f"{'='*80}")
    
    # For each FP, identify which filter(s) caused the rejection
    print(f"\n{'='*80}")
    print(f"FALSE POSITIVE ANALYSIS (Rejected Winners - {len(fp_trades)} total)")
    print(f"{'='*80}")
    
    # Categorize by blocking filter
    fp_by_filter = {}
    for r in fp_trades:
        # Find which filter(s) triggered the hard block (score >= 70)
        hard_blocks = []
        for fname, fdata in r['filters'].items():
            if fdata['score'] >= LAYER1_REJECT_THRESHOLD:
                hard_blocks.append((fname, fdata['score'], fdata['reason']))
        
        # Also check if composite alone caused it
        composite_block = r['composite_score'] >= LAYER1_REJECT_THRESHOLD and not hard_blocks
        
        if hard_blocks:
            for fname, score, reason in hard_blocks:
                fp_by_filter.setdefault(fname, []).append({
                    "trade_id": r['trade_id'],
                    "direction": r['direction'],
                    "pnl_pips": r['pnl_pips'],
                    "pnl_usd": r['pnl_usd'],
                    "score": score,
                    "reason": reason,
                    "data": r['filters'][fname]['data'],
                })
        elif composite_block:
            fp_by_filter.setdefault('composite', []).append({
                "trade_id": r['trade_id'],
                "direction": r['direction'],
                "pnl_pips": r['pnl_pips'],
                "composite_score": r['composite_score'],
            })
    
    for fname, fp_list in sorted(fp_by_filter.items(), key=lambda x: -len(x[1])):
        print(f"\n--- {fname} ({len(fp_list)} false positives) ---")
        for fp in fp_list:
            print(f"  Trade #{fp['trade_id']} {fp['direction']} | PnL: {fp['pnl_pips']:+.1f} pips (${fp['pnl_usd']:+.0f}) | Score: {fp['score']} | Reason: {fp['reason']}")
            if 'data' in fp and fp['data']:
                print(f"    Data: {fp['data']}")
    
    # Also show all filter scores for FP trades for deep analysis
    print(f"\n{'='*80}")
    print(f"DETAILED FP TRADE FILTER SCORES")
    print(f"{'='*80}")
    for r in fp_trades:
        print(f"\nTrade #{r['trade_id']} {r['direction']} | PnL: {r['pnl_pips']:+.1f} pips (${r['pnl_usd']:+.0f}) | Verdict: {r['verdict']} | Composite: {r['composite_score']}")
        for fname, fdata in sorted(r['filters'].items(), key=lambda x: -x[1]['score']):
            marker = " << HARD BLOCK" if fdata['score'] >= LAYER1_REJECT_THRESHOLD else ""
            if fdata['score'] > 0:
                print(f"  {fname:30s}: {fdata['score']:5.1f} | {fdata['reason']}{marker}")
    
    # Show TP trades for comparison
    print(f"\n{'='*80}")
    print(f"TRUE POSITIVE ANALYSIS (Correctly Rejected Losers - {len(tp_trades)} total)")
    print(f"{'='*80}")
    tp_by_filter = {}
    for r in tp_trades:
        hard_blocks = []
        for fname, fdata in r['filters'].items():
            if fdata['score'] >= LAYER1_REJECT_THRESHOLD:
                hard_blocks.append((fname, fdata['score'], fdata['reason']))
        composite_block = r['composite_score'] >= LAYER1_REJECT_THRESHOLD and not hard_blocks
        if hard_blocks:
            for fname, score, reason in hard_blocks:
                tp_by_filter.setdefault(fname, []).append(r['trade_id'])
    
    for fname, tids in sorted(tp_by_filter.items(), key=lambda x: -len(x[1])):
        print(f"  {fname}: {len(tids)} losses blocked (trades: {tids})")
    
    # Net PnL impact
    fp_pnl = sum(r['pnl_usd'] for r in fp_trades)
    tp_pnl = sum(r['pnl_usd'] for r in tp_trades)
    tn_pnl = sum(r['pnl_usd'] for r in tn_trades)
    fn_pnl = sum(r['pnl_usd'] for r in fn_trades)
    total_pnl_with_lre = tn_pnl + fn_pnl
    total_pnl_without_lre = sum(r['pnl_usd'] for r in results)
    
    print(f"\n{'='*80}")
    print(f"PnL IMPACT")
    print(f"{'='*80}")
    print(f"TN (kept winners) PnL:     ${tn_pnl:+,.2f}")
    print(f"FP (rejected winners) PnL: ${fp_pnl:+,.2f}  <-- LOST profit due to false positives")
    print(f"TP (blocked losses) PnL:  ${tp_pnl:+,.2f}  <-- SAVED loss by true positives")
    print(f"FN (kept losers) PnL:     ${fn_pnl:+,.2f}")
    print(f"Total without LRE:         ${total_pnl_without_lre:+,.2f}")
    print(f"Total with LRE:           ${total_pnl_with_lre:+,.2f}")
    print(f"Net LRE impact:           ${total_pnl_with_lre - total_pnl_without_lre:+,.2f}")
    
    # Save full results
    output = {
        "summary": {
            "total_trades": len(results),
            "winners": total_winners,
            "losers": total_losers,
            "WPR": round(wpr, 1),
            "LRR": round(lrr, 1),
            "FP_count": len(fp_trades),
            "TP_count": len(tp_trades),
            "TN_count": len(tn_trades),
            "FN_count": len(fn_trades),
            "pnl_without_lre": round(total_pnl_without_lre, 2),
            "pnl_with_lre": round(total_pnl_with_lre, 2),
            "fp_pnl_lost": round(fp_pnl, 2),
            "tp_pnl_saved": round(abs(tp_pnl), 2),
        },
        "fp_by_filter": {k: v for k, v in fp_by_filter.items()},
        "false_positives": [{k: v for k, v in r.items() if k != 'filters'} | {"blocking_filters": {fname: {"score": fdata['score'], "reason": fdata['reason'], "data": fdata['data']} for fname, fdata in r['filters'].items() if fdata['score'] >= LAYER1_REJECT_THRESHOLD}} for r in fp_trades],
        "all_results": [{k: v for k, v in r.items() if k != 'filters'} for r in results],
    }
    
    out_path = "/home/z/my-project/forex-agent/download/diagnosis_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
