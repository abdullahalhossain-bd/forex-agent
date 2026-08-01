"""Analyze the 27 TP trades from market_memory to find optimal threshold.
Reads the pre-fix diagnosis to understand what WR patterns caused rejections.
"""
import json, csv
from collections import defaultdict

TRADES_CSV = "/home/z/my-project/forex-agent/backtest/results_EURUSD_H1.csv"

# Load pre-fix diagnosis
with open("/home/z/my-project/forex-agent/download/diagnosis_results.json") as f:
    diag = json.load(f)

# Load trades for context
trades = {}
with open(TRADES_CSV) as f:
    for row in csv.DictReader(f):
        trades[row['trade_id']] = row

# The 27 TP trades blocked by market_memory (from before fix)
tp_ids = ['5', '6', '11', '13', '16', '19', '29', '33', '35', '37', '39', '40', '41', '43', '46', '59', '60', '65', '66', '68', '69', '72', '73', '76', '81', '82', '87']
fp_ids = ['44', '49', '74', '85']

print("="*80)
print("MARKET MEMORY TRADE ANALYSIS")
print("="*80)

print(f"\nTRUE POSITIVES (correctly rejected losers): {len(tp_ids)}")
tp_pnl = 0
for tid in tp_ids:
    t = trades[tid]
    pnl = float(t['pnl_usd'])
    tp_pnl += pnl
    d = t['direction']
    print(f"  #{tid:>3s} {d:4s} | {float(t['pnl_pips']):+7.1f} pips | ${pnl:+8.2f} | conf={t['confidence']} | confact={t['confluence_factors']} | {t['strategy']}")
print(f"  TOTAL PnL of TP trades: ${tp_pnl:+,.2f}")

print(f"\nFALSE POSITIVES (incorrectly rejected winners): {len(fp_ids)}")
fp_pnl = 0
for tid in fp_ids:
    t = trades[tid]
    pnl = float(t['pnl_usd'])
    fp_pnl += pnl
    d = t['direction']
    print(f"  #{tid:>3s} {d:4s} | {float(t['pnl_pips']):+7.1f} pips | ${pnl:+8.2f} | conf={t['confidence']} | confact={t['confluence_factors']} | {t['strategy']}")
print(f"  TOTAL PnL of FP trades: ${fp_pnl:+,.2f}")

# Now simulate market_memory state to understand the WR at each TP/FP trade
print(f"\n{'='*80}")
print("SIMULATING MARKET MEMORY STATE AT EACH TRADE")
print("{'='*80}")

# Walk through all trades in CSV order, tracking market_memory state
from collections import deque

mm_db = defaultdict(lambda: deque(maxlen=50))
results = []

with open(TRADES_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        tid = row['trade_id']
        d = row['direction']
        pnl = float(row['pnl_usd'])
        pips = float(row['pnl_pips'])
        is_winner = pips > 0
        
        # Determine price zone and regime (same logic as diagnosis script)
        conf = int(row['confidence'])
        confluence = int(row['confluence_factors'])
        if conf >= 85 and confluence >= 25:
            regime = "trending"
        elif conf >= 60 and confluence >= 18:
            regime = "trending" if is_winner else "volatile"
        else:
            regime = "ranging"
        pz = "mid"  # no SR levels in minimal context
        
        key = f"EURUSD:{d}:{pz}:{regime}"
        hist = mm_db[key]
        
        n = len(hist)
        wins = sum(hist)
        wr = wins / n if n > 0 else 0.5
        
        # Consecutive losses
        cl = 0
        for o in reversed(hist):
            if o == 0: cl += 1
            else: break
        
        # Original v3 scoring
        if n >= 3:
            s_orig = 90 if wr<.2 else (75 if wr<.3 else (55 if wr<.4 else (35 if wr<.5 else max(0,20-wr*30))))
            if cl >= 3: s_orig = min(100, s_orig+20)
            elif cl >= 2: s_orig = min(100, s_orig+10)
        else:
            s_orig = 0
        
        label = ""
        if tid in tp_ids: label = "TP"
        elif tid in fp_ids: label = "FP"
        elif is_winner: label = "TN"
        else: label = "FN"
        
        results.append({
            'id': tid, 'dir': d, 'pnl': pnl, 'pips': pips, 'is_winner': is_winner,
            'key': key, 'n': n, 'wr': wr, 'cl': cl, 'score_orig': s_orig, 'label': label
        })
        
        # Record outcome
        mm_db[key].append(1 if is_winner else 0)

# Analyze TP vs FP by WR and N
print(f"\n{'ID':>5s} {'DIR':>4s} {'PnL':>9s} {'N':>4s} {'WR':>6s} {'CL':>3s} {'Score':>6s} {'Label':>4s}")
print("-"*60)
for r in results:
    if r['label'] in ('TP', 'FP'):
        print(f"{r['id']:>5s} {r['dir']:>4s} {r['pnl']:+9.2f} {r['n']:>4d} {r['wr']:>5.0%} {r['cl']:>3d} {r['score_orig']:>6.0f} {r['label']:>4s}")

# Statistical analysis: find threshold that maximizes TP while minimizing FP
print(f"\n{'='*80}")
print("THRESHOLD ANALYSIS: Finding optimal WR/N cutoff")
print("{'='*80}")

for min_n in [3, 5, 6, 8, 10]:
    for wr_thresh in [0.10, 0.15, 0.20, 0.25, 0.30]:
        tp = fp = tn = fn = 0
        for r in results:
            blocked = r['n'] >= min_n and r['wr'] < wr_thresh
            if blocked and not r['is_winner']: tp += 1
            elif blocked and r['is_winner']: fp += 1
            elif not blocked and r['is_winner']: tn += 1
            else: fn += 1
        
        total_winners = sum(1 for r in results if r['is_winner'])
        total_losers = sum(1 for r in results if not r['is_winner'])
        wpr = tn / total_winners * 100 if total_winners > 0 else 0
        lrr = tp / total_losers * 100 if total_losers > 0 else 0
        
        if fp <= 2 and tp >= 10:  # Show promising configs
            print(f"  min_N={min_n}, WR<{wr_thresh:.0%}: TP={tp:2d} FP={fp} | WPR={wpr:.1f}% LRR={lrr:.1f}%")

# Best config
print(f"\nBEST CONFIG WITH FP<=1:")
for min_n in [3, 5, 6, 8, 10, 12]:
    for wr_thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        tp = fp = tn = fn = 0
        for r in results:
            blocked = r['n'] >= min_n and r['wr'] < wr_thresh
            if blocked and not r['is_winner']: tp += 1
            elif blocked and r['is_winner']: fp += 1
            elif not blocked and r['is_winner']: tn += 1
            else: fn += 1
        
        total_winners = sum(1 for r in results if r['is_winner'])
        total_losers = sum(1 for r in results if not r['is_winner'])
        wpr = tn / total_winners * 100 if total_winners > 0 else 0
        lrr = tp / total_losers * 100 if total_losers > 0 else 0
        
        if fp <= 1 and tp >= 5:
            print(f"  min_N={min_n:2d}, WR<{wr_thresh:.0%}: TP={tp:2d} FP={fp} | WPR={wpr:.1f}% LRR={lrr:.1f}%")
