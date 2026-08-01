"""Debug market_memory state at each trade to find why it's not triggering."""
import csv, sys, os
from collections import defaultdict, deque

sys.path.insert(0, "/home/z/my-project/forex-agent")
os.environ["LRE_SHADOW_MODE"] = "0"
os.environ["LRE_ENABLED"] = "1"

TRADES_CSV = "/home/z/my-project/forex-agent/backtest/results_EURUSD_H1.csv"

from core.loss_rejection_engine.layer1_structural_filters import (
    StructuralFilterLayer, LAYER1_REJECT_THRESHOLD
)

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
    direction = trade['direction']
    entry = trade['entry_price']
    sl = trade['stop_loss']
    tp = trade['take_profit']
    rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 2.0
    
    recent = prior_outcomes[-5:] if len(prior_outcomes) >= 5 else prior_outcomes
    if recent:
        buy_wins = sum(1 for d, p in recent if d == 'BUY' and p > 0)
        sell_wins = sum(1 for d, p in recent if d == 'SELL' and p > 0)
        total = len(recent)
        trend_str = max(buy_wins, sell_wins) / total if total > 0 else 0.5
    else:
        trend_str = 0.5
    
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
    
    dec_out = {
        "decision": direction,
        "entry": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "confidence": conf,
        "rr": round(rr, 2),
    }
    
    ana_out = {
        "sr": {"levels": []},
        "liquidity": {"grade": "CLEAR"},
        "sentiment": {"retail_long_pct": 0.5, "fg_index": 50.0},
        "divergence": {},
        "smc": {},
        "market_structure": {},
    }
    
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


trades = load_trades()
layer = StructuralFilterLayer()
prior_outcomes = []

# Print first 10 trades to see market_memory state
for i, trade in enumerate(trades[:15]):
    dec_out, ana_out, mkt_out, regime_label = build_minimal_context(
        trade, prior_outcomes, i, trades
    )
    
    l1_out = layer.evaluate(dec_out, ana_out, mkt_out, symbol="EURUSD")
    
    # Get market_memory filter result
    mm_result = next((f for f in l1_out.filters if f.name == "market_memory"), None)
    
    is_winner = trade['pnl_pips'] > 0
    mm_db = layer.market_memory._db
    
    # Check what keys exist
    buy_keys = [k for k in mm_db if ":BUY:" in k]
    
    print(f"Trade #{trade['trade_id']:>3s} {trade['direction']:4s} | pips={trade['pnl_pips']:+7.1f} | usd={trade['pnl_usd']:+9.2f} | regime={regime_label:10s} | MM_score={mm_result.rejection_score if mm_result else 'N/A':>5} | MM_reason={mm_result.reason if mm_result else 'N/A'}")
    
    # Show active MM keys for this direction
    d = trade['direction']
    for k, v in mm_db.items():
        if f":{d}:" in k:
            wins = sum(v)
            total = len(v)
            wr = wins/total if total > 0 else 0
            print(f"    DB key: {k} -> N={total}, W={wins}, WR={wr:.0%}, last5={list(v)[-5:]}")
    
    layer.record_trade_outcome(
        "EURUSD", trade['direction'], "mid", regime_label, trade['pnl_usd']
    )
    prior_outcomes.append((trade['direction'], trade['pnl_usd']))
