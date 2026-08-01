"""Deep analysis of failure_cascade and regime_transition false positives.

Analyzes the actual predictive value of cascade signals in the trade data
to find optimal thresholds.
"""
from __future__ import annotations
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CSV_PATH = PROJECT_ROOT / "backtest" / "results_EURUSD_H1.csv"

def load_trades():
    df = pd.read_csv(str(CSV_PATH), parse_dates=["entry_time", "exit_time"])
    df["is_win"] = df["pnl_pips"] > 0
    df = df.sort_values("entry_time").reset_index(drop=True)
    return df

def analyze_cascade_predictive_power(df):
    """For each N consecutive same-dir losses, what is the win rate of the NEXT trade?"""
    print("\n" + "="*80)
    print("CASCADING PATTERN ANALYSIS: Win rate after N consecutive same-dir losses")
    print("="*80)
    
    symbol = "EURUSD"
    # Track per-direction consecutive losses
    buy_consec = 0
    sell_consec = 0
    all_consec = 0
    
    results = {"same_dir": {}, "all_dir": {}}
    
    for idx, row in df.iterrows():
        d = row["direction"]
        is_win = row["is_win"]
        trade_id = row["trade_id"]
        pnl = row["pnl_usd"]
        
        # Record stats for this N
        if d == "BUY":
            n = buy_consec
            buy_consec = 0 if is_win else buy_consec + 1
        else:
            n = sell_consec
            sell_consec = 0 if is_win else sell_consec + 1
        
        key = f"{n}_same_dir_{d}"
        results["same_dir"].setdefault(n, {"wins": 0, "losses": 0, "trades": [], "directions": []})
        if is_win:
            results["same_dir"][n]["wins"] += 1
        else:
            results["same_dir"][n]["losses"] += 1
        results["same_dir"][n]["trades"].append({"id": trade_id, "d": d, "win": is_win, "pnl": pnl})
        results["same_dir"][n]["directions"].append(d)
        
        # All direction
        n_all = all_consec
        all_consec = 0 if is_win else all_consec + 1
        
        results["all_dir"].setdefault(n_all, {"wins": 0, "losses": 0, "trades": []})
        if is_win:
            results["all_dir"][n_all]["wins"] += 1
        else:
            results["all_dir"][n_all]["losses"] += 1
        results["all_dir"][n_all]["trades"].append({"id": trade_id, "d": d, "win": is_win, "pnl": pnl})
    
    # Print summary
    for category in ["same_dir", "all_dir"]:
        print(f"\n--- {category.upper()} ---")
        print(f"{'N consec':>10s} {'Total':>6s} {'Wins':>6s} {'Losses':>6s} {'WR':>8s} {'Avg PnL':>10s} {'Trade IDs'}")
        print("-" * 80)
        for n in sorted(results[category].keys()):
            r = results[category][n]
            total = r["wins"] + r["losses"]
            wr = r["wins"] / total * 100 if total > 0 else 0
            avg_pnl = np.mean([t["pnl"] for t in r["trades"]]) if r["trades"] else 0
            ids = [str(t["id"]) + ("W" if t["win"] else "L") for t in r["trades"]]
            print(f"{n:>10d} {total:>6d} {r['wins']:>6d} {r['losses']:>6d} {wr:>7.1f}% ${avg_pnl:>9.2f}  {', '.join(ids)}")
    
    return results


def analyze_trade_sequences_around_fps(df):
    """Show the full trade sequence around each blocked winner."""
    print("\n" + "="*80)
    print("TRADE SEQUENCES AROUND BLOCKED WINNERS")
    print("="*80)
    
    # Simulate original failure_cascade tracking
    same_dir = {"BUY": 0, "SELL": 0}
    all_dir = 0
    global_losses = 0
    
    for idx, row in df.iterrows():
        d = row["direction"]
        is_win = row["is_win"]
        
        # Check if this trade would be blocked by original thresholds
        blocked = False
        block_reason = ""
        
        if same_dir[d] >= 3:
            blocked = True
            block_reason = f"{same_dir[d]} same-dir {d} losses"
        if all_dir >= 4:
            blocked = True
            block_reason = f"{all_dir} total consec losses"
        if global_losses >= 4:
            blocked = True
            block_reason = f"{global_losses} global consec losses"
        
        if blocked and is_win:
            # Print context: 5 trades before, this trade, 2 after
            start = max(0, idx - 5)
            end = min(len(df), idx + 3)
            print(f"\n>>> BLOCKED WINNER: Trade #{row['trade_id']} | {d} | PnL=${row['pnl_usd']:.2f} | Reason: {block_reason}")
            print(f"    Strategy: {row['strategy']} | Confidence: {row['confidence']}% | Hold: {row['hold_bars']} bars")
            print(f"    --- Preceding trades ---")
            for i in range(start, idx):
                r = df.iloc[i]
                marker = "*" if not r["is_win"] else " "
                print(f"    {marker} #{r['trade_id']:>3d} {r['direction']:4s} {r['pnl_pips']:>8.1f} pips ${r['pnl_usd']:>10.2f} {r['strategy']:>10s} hold={r['hold_bars']:>3d} {'LOSS' if not r['is_win'] else 'WIN '} conf={r['confidence']}%")
            print(f"    >>> #{row['trade_id']:>3d} {d:4s} {row['pnl_pips']:>8.1f} pips ${row['pnl_usd']:>10.2f} {row['strategy']:>10s} hold={row['hold_bars']:>3d} WIN  conf={row['confidence']}% [BLOCKED]")
            for i in range(idx + 1, end):
                r = df.iloc[i]
                marker = "*" if not r["is_win"] else " "
                print(f"    {marker} #{r['trade_id']:>3d} {r['direction']:4s} {r['pnl_pips']:>8.1f} pips ${r['pnl_usd']:>10.2f} {r['strategy']:>10s} hold={r['hold_bars']:>3d} {'LOSS' if not r['is_win'] else 'WIN '} conf={r['confidence']}%")
        
        # Update state
        if is_win:
            same_dir[d] = 0
            all_dir = 0
            global_losses = 0
        else:
            same_dir[d] += 1
            all_dir += 1
            global_losses += 1


def analyze_regime_transitions(df):
    """Analyze what happens at regime transitions in the data."""
    print("\n" + "="*80)
    print("REGIME TRANSITION ANALYSIS (using trade parameters as regime proxy)")
    print("="*80)
    
    # Use SL width / confidence as regime proxy (no lookahead)
    _PIP = 0.0001
    _ATR = 0.0065
    
    prev_regime = None
    transitions = []
    
    for idx, row in df.iterrows():
        sl_pips = abs(row["entry_price"] - row["stop_loss"]) / _PIP
        conf = row["confidence"]
        sl_atr = sl_pips / (_ATR * 10000)
        
        if sl_atr > 2.5:
            regime = "volatile"
        elif conf >= 80 and sl_atr < 1.5:
            regime = "trending"
        elif conf >= 60:
            regime = "trending"  # high confidence trades assumed to be in trends
        else:
            regime = "ranging"
        
        if prev_regime and prev_regime != regime:
            transitions.append({
                "idx": idx, "trade_id": row["trade_id"],
                "from": prev_regime, "to": regime,
                "is_win": row["is_win"], "pnl": row["pnl_usd"],
                "direction": row["direction"],
            })
        
        prev_regime = regime
    
    print(f"\nTotal regime transitions detected: {len(transitions)}")
    for t in transitions:
        marker = "WIN" if t["is_win"] else "LOSS"
        print(f"  Trade #{t['trade_id']:>3d}: {t['from']:>10s} -> {t['to']:<10s} | {t['direction']:4s} | {marker} | ${t['pnl']:>10.2f}")
    
    if transitions:
        win_at_transition = sum(1 for t in transitions if t["is_win"])
        avg_pnl_trans = np.mean([t["pnl"] for t in transitions])
        print(f"\nWin rate AT transition: {win_at_transition}/{len(transitions)} = {win_at_transition/len(transitions)*100:.1f}%")
        print(f"Average PnL AT transition: ${avg_pnl_trans:.2f}")
    
    return transitions


def find_optimal_thresholds(df):
    """Sweep thresholds to find optimal WPR>=95% with max LRR."""
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION: Find max LRR with WPR >= 95%")
    print("="*80)
    
    best = {"lrr": -1, "wpr": 0, "config": None, "metrics": None}
    
    # Test different same-dir cascade thresholds
    print(f"\n{'SameDirMin':>12s} {'AllDirMin':>11s} {'GlobalMin':>11s} {'WPR':>8s} {'LRR':>8s} {'PF':>8s} {'Exp':>8s} {'Trades':>7s}")
    print("-" * 80)
    
    for sd_min in [2, 3, 4, 5, 6, 7, 8]:
        for ad_min in [3, 4, 5, 6, 7]:
            for gl_min in [4, 5, 6, 7, 8]:
                # Simulate
                same_dir = {"BUY": 0, "SELL": 0}
                all_dir = 0
                global_losses = 0
                
                tp = fp = tn = fn = 0
                
                for idx, row in df.iterrows():
                    d = row["direction"]
                    is_win = row["is_win"]
                    
                    blocked = False
                    if same_dir[d] >= sd_min:
                        blocked = True
                    if all_dir >= ad_min:
                        blocked = True
                    if global_losses >= gl_min:
                        blocked = True
                    
                    if blocked and not is_win:
                        tp += 1
                    elif blocked and is_win:
                        fp += 1
                    elif not blocked and is_win:
                        tn += 1
                    else:
                        fn += 1
                    
                    if is_win:
                        same_dir[d] = 0
                        all_dir = 0
                        global_losses = 0
                    else:
                        same_dir[d] += 1
                        all_dir += 1
                        global_losses += 1
                
                n_win = tp + tn + fp  # total winners = correctly rejected + correctly accepted + falsely rejected
                # Wait, that's wrong. Let me recalculate.
                # Winners in dataset: tn + fp (accepted winners + rejected winners)
                # Losers in dataset: fn + tp (accepted losers + rejected losers)
                total_winners = tn + fp
                total_losers = fn + tp
                
                wpr = tn / total_winners * 100 if total_winners > 0 else 100
                lrr = tp / total_losers * 100 if total_losers > 0 else 0
                
                if wpr >= 95.0 and lrr > best["lrr"]:
                    # Calculate PF and expectancy
                    post_trades_pnl = []
                    same_dir_s = {"BUY": 0, "SELL": 0}
                    all_dir_s = 0
                    global_losses_s = 0
                    
                    for idx, row in df.iterrows():
                        d = row["direction"]
                        is_win = row["is_win"]
                        blocked = False
                        if same_dir_s[d] >= sd_min:
                            blocked = True
                        if all_dir_s >= ad_min:
                            blocked = True
                        if global_losses_s >= gl_min:
                            blocked = True
                        
                        if not blocked:
                            post_trades_pnl.append(row["pnl_usd"])
                        
                        if is_win:
                            same_dir_s[d] = 0
                            all_dir_s = 0
                            global_losses_s = 0
                        else:
                            same_dir_s[d] += 1
                            all_dir_s += 1
                            global_losses_s += 1
                    
                    wins_pnl = [p for p in post_trades_pnl if p > 0]
                    losses_pnl = [p for p in post_trades_pnl if p <= 0]
                    gp = sum(wins_pnl)
                    gl = abs(sum(losses_pnl)) if losses_pnl else 1
                    pf = gp / gl if gl > 0 else 999
                    wr = len(wins_pnl) / len(post_trades_pnl) * 100 if post_trades_pnl else 0
                    avg_w = np.mean(wins_pnl) if wins_pnl else 0
                    avg_l = np.mean(losses_pnl) if losses_pnl else 0
                    exp = (wr/100 * avg_w) - ((1-wr/100) * avg_l)
                    
                    best = {
                        "lrr": lrr, "wpr": wpr,
                        "config": {"same_dir": sd_min, "all_dir": ad_min, "global": gl_min},
                        "metrics": {"pf": pf, "expectancy": exp, "trades": len(post_trades_pnl),
                                     "tp": tp, "fp": fp, "tn": tn, "fn": fn}
                    }
                
                # Print interesting configs
                if wpr >= 95.0 and lrr >= 10.0:
                    post_trades_count = tn + fn
                    print(f"{sd_min:>12d} {ad_min:>11d} {gl_min:>11d} {wpr:>7.1f}% {lrr:>7.1f}% {best.get('metrics', {}).get('pf', 0):>8.2f} {best.get('metrics', {}).get('expectancy', 0):>8.2f} {post_trades_count:>7d}")
    
    if best["config"]:
        print(f"\n*** OPTIMAL CONFIG ***")
        print(f"  Same-dir consecutive losses >= {best['config']['same_dir']} -> REJECT")
        print(f"  All-dir consecutive losses >= {best['config']['all_dir']} -> REJECT")
        print(f"  Global consecutive losses >= {best['config']['global']} -> REJECT")
        print(f"  WPR: {best['wpr']:.1f}% | LRR: {best['lrr']:.1f}%")
        print(f"  PF: {best['metrics']['pf']:.2f} | Expectancy: ${best['metrics']['expectancy']:.2f}")
        print(f"  Trades: {best['metrics']['trades']} | TP={best['metrics']['tp']} FP={best['metrics']['fp']} TN={best['metrics']['tn']} FN={best['metrics']['fn']}")
    
    return best


def main():
    df = load_trades()
    print(f"Loaded {len(df)} trades: {df['is_win'].sum()}W / {(~df['is_win']).sum()}L")
    
    # 1. Cascade predictive power
    cascade_results = analyze_cascade_predictive_power(df)
    
    # 2. Trade sequences around blocked winners
    analyze_trade_sequences_around_fps(df)
    
    # 3. Regime transition analysis
    transitions = analyze_regime_transitions(df)
    
    # 4. Threshold optimization
    best = find_optimal_thresholds(df)
    
    # Save results
    output = {
        "cascade_predictive_power": {str(k): {"wins": v["wins"], "losses": v["losses"]} for k, v in cascade_results["same_dir"].items()},
        "optimal_config": best["config"] if best["config"] else None,
        "optimal_metrics": best["metrics"] if best["metrics"] else None,
        "regime_transitions": transitions,
    }
    out_path = PROJECT_ROOT / "download" / "lre_deep_analysis.json"
    with open(str(out_path), "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nAnalysis saved to {out_path}")


if __name__ == "__main__":
    main()
