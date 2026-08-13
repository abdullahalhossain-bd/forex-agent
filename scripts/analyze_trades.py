#!/usr/bin/env python3
"""Detailed analysis: for each pair, dump the first 10 trades to see entry/exit."""
import os, sys
sys.path.insert(0, "/home/z/my-project/repos/forex-agent")
os.environ["BACKTEST_MODE"] = "1"

import pandas as pd
sys.path.insert(0, "/home/z/my-project/scripts")
from fast_backtest import load_csv, compute_indicators, backtest_pair

for pair in ["EURUSD", "NZDUSD"]:
    print(f"\n{'='*70}")
    print(f"  {pair} — first 5 trades")
    print(f"{'='*70}")
    df = load_csv(pair, "H1")
    if df is None:
        continue
    df = compute_indicators(df)
    os.environ["BT_MIN_CONFIDENCE"] = "55"
    os.environ["BT_MIN_FACTORS"] = "3"
    os.environ["BT_MIN_RR"] = "2.0"
    os.environ["BT_SESSION_FILTER"] = "1"
    result = backtest_pair(pair, "H1", df, warmup=250)
    trades = result.get("trades_detail", [])
    print(f"Total trades: {len(trades)} | Wins: {result['wins']} | Losses: {result['losses']}")
    print(f"Winrate: {result['winrate']:.1f}% | Avg RR: {result['avg_rr']:.2f} | PF: {result['profit_factor']:.2f}")
    print(f"\nFirst 5 trades:")
    for i, t in enumerate(trades[:5]):
        print(f"  #{i+1} {t['direction']:4s} entry={t['entry']:.5f} sl={t['sl']:.5f} tp={t['tp']:.5f} "
              f"exit={t['exit_price']:.5f} outcome={t['outcome']:4s} pnl=${t['pnl']:+.2f} "
              f"hold={t['hold_bars']}bars rr={t['rr']:.2f}")
    print(f"\nLast 5 trades:")
    for i, t in enumerate(trades[-5:]):
        idx = len(trades) - 5 + i + 1
        print(f"  #{idx} {t['direction']:4s} entry={t['entry']:.5f} sl={t['sl']:.5f} tp={t['tp']:.5f} "
              f"exit={t['exit_price']:.5f} outcome={t['outcome']:4s} pnl=${t['pnl']:+.2f} "
              f"hold={t['hold_bars']}bars rr={t['rr']:.2f}")

    # Analyze why losses happened
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    wins = [t for t in trades if t["outcome"] == "WIN"]
    print(f"\nLoss analysis ({len(losses)} losses):")
    avg_loss_hold = sum(t["hold_bars"] for t in losses) / len(losses) if losses else 0
    print(f"  Avg hold bars (losses): {avg_loss_hold:.1f}")
    # How many losses hit SL within 1-2 bars?
    quick_losses = sum(1 for t in losses if t["hold_bars"] <= 2)
    print(f"  Quick losses (<=2 bars): {quick_losses}/{len(losses)} ({quick_losses/len(losses)*100:.0f}%)")

    print(f"\nWin analysis ({len(wins)} wins):")
    avg_win_hold = sum(t["hold_bars"] for t in wins) / len(wins) if wins else 0
    print(f"  Avg hold bars (wins): {avg_win_hold:.1f}")
