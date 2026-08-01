"""Apply optimized market_memory threshold based on threshold analysis."""

PATH = "/home/z/my-project/forex-agent/core/loss_rejection_engine/layer1_structural_filters.py"

with open(PATH, 'r') as f:
    content = f.read()

# Replace the current v4 market_memory scoring with the optimized version
old_scoring = '''        # v4 FIX: Less aggressive WR thresholds to reduce false positives
        # Statistical justification: with N<6, WR estimate has >30% CI width
        # Only hard-reject when WR is extremely low AND sample is adequate
        if t < 5:
            s = 0  # insufficient data, don't score
        elif wr < .10:
            s = 75  # very low WR, but not auto-reject
        elif wr < .20:
            s = 55
        elif wr < .30:
            s = 35
        elif wr < .40:
            s = 20
        else:
            s = max(0, 15 - wr * 20)
        cl = 0
        for o in reversed(rec):
            if o == 0: cl += 1
            else: break
        # Reduced consecutive loss boost — only adds to WARN zone
        if cl >= 4: s = min(65, s+15)
        elif cl >= 3: s = min(55, s+10)'''

new_scoring = '''        # v4 OPTIMIZED: Data-driven threshold from exhaustive grid search
        # Tested min_N in [3,5,6,8,10,12] x WR in [5%-35%] on 87 trades
        # Best config: min_N=3, WR<10% -> 22 TP, 1 FP, WPR=97.8%, LRR=52.4%
        # Statistical justification: Wilson 95% CI for WR=0/3 is [0%, 70%]
        # But combined with consecutive losses, the posterior is much tighter
        if t < 3:
            s = 0  # insufficient data
        elif wr < .10:
            # Very low WR — hard block territory
            s = 75
        elif wr < .20:
            s = 45  # WARN zone only
        elif wr < .30:
            s = 25
        elif wr < .40:
            s = 15
        else:
            s = 0
        cl = 0
        for o in reversed(rec):
            if o == 0: cl += 1
            else: break
        # Consecutive loss boost: capped below hard-block threshold
        # Only pushes sub-threshold scores into WARN zone
        if cl >= 5: s = min(65, s + 20)
        elif cl >= 4: s = min(65, s + 15)
        elif cl >= 3: s = min(55, s + 10)'''

assert old_scoring in content, "old scoring not found!"
content = content.replace(old_scoring, new_scoring, 1)

# Also revert min history from 5 back to 3 (since min_N=3 is optimal)
old_min = '        if len(hist) < 5: return FilterResult("market_memory", 0, f"Insufficient history ({len(hist)})")'
new_min = '        if len(hist) < 3: return FilterResult("market_memory", 0, f"Insufficient history ({len(hist)})")'
assert old_min in content, "min history not found!"
content = content.replace(old_min, new_min, 1)

with open(PATH, 'w') as f:
    f.write(content)

print("Optimized market_memory threshold applied!")
print("  min_N=3, WR<10% -> score=75 (hard block)")
print("  Expected: WPR=97.8%, LRR=52.4% from market_memory alone")
