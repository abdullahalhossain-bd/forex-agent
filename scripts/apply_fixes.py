"""Apply fixes to layer1_structural_filters.py:
1. Fix failure_cascade: store PnL in 3-tuple, fix magnitude/recovery, add cluster detection
2. Fix market_memory: less aggressive WR thresholds (statistically proven harmful: 4 FP)
3. Regime transition: already optimal (capped at 40), no changes needed
"""
import re

PATH = "/home/z/my-project/forex-agent/core/loss_rejection_engine/layer1_structural_filters.py"

with open(PATH, 'r') as f:
    content = f.read()

# ═══ FIX 1: failure_cascade ═══
# Replace record_outcome to store 3-tuple
old_record = '''    def record_outcome(self, sym, d, pnl):
        self._sh.setdefault(sym, deque(maxlen=20)).append((d, 1 if pnl>0 else 0))
        self._gh.append((sym, d, 1 if pnl>0 else 0))'''

new_record = '''    def record_outcome(self, sym, d, pnl):
        # v4 FIX: store (direction, outcome, pnl) 3-tuple
        # Enables magnitude-adaptive scoring and recovery grace
        outcome = 1 if pnl > 0 else 0
        self._sh.setdefault(sym, deque(maxlen=30)).append((d, outcome, pnl))
        self._gh.append((sym, d, outcome, pnl))'''

assert old_record in content, "record_outcome not found!"
content = content.replace(old_record, new_record, 1)

# Replace __init__ to increase deque sizes
old_init = '        self._sh: Dict[str, deque] = {}; self._gh: deque = deque(maxlen=30)'
new_init = '        self._sh: Dict[str, deque] = {}; self._gh: deque = deque(maxlen=50)'
assert old_init in content, "__init__ not found!"
content = content.replace(old_init, new_init, 1)

# Replace same-direction consecutive loss counting
old_sdl = '''        # ── Same-direction consecutive losses ──
        sdl = 0; total_loss_pnl = 0.0
        for dr, o, *_ in reversed(sh):
            if dr == d and o == 0: sdl += 1
            elif dr == d: break'''

new_sdl = '''        # ── Same-direction consecutive losses (with PnL tracking) ──
        sdl = 0; total_loss_pnl = 0.0
        for item in reversed(sh):
            dr, o = item[0], item[1]
            pnl_val = item[2] if len(item) > 2 else 0.0
            if dr == d and o == 0:
                sdl += 1
                total_loss_pnl += abs(pnl_val)
            elif dr == d:
                break'''

assert old_sdl in content, "sdl counting not found!"
content = content.replace(old_sdl, new_sdl, 1)

# Replace recovery grace
old_recovery = '''        # Recovery grace: large opposite-direction win indicates regime rotation
        opp = "SELL" if d == "BUY" else "BUY"
        recent_opp_win = 0.0
        for dr, o, *_ in reversed(sh):
            if dr == opp and o == 1:
                recent_opp_win = 1
                break
        if recent_opp_win > 0 and sdl >= 5:
            # Find the actual pnl of the most recent opp win
            for dr, o, pnl_val, *_ in reversed(sh):
                if dr == opp and o == 1:
                    if abs(pnl_val) > 500:
                        s = max(0, s - 15)
                        reasons.append(f"large {opp} win (${abs(pnl_val):.0f}) reduces cascade")
                    break'''

new_recovery = '''        # v4 FIX: Recovery grace — correctly reads stored pnl from 3-tuple
        opp = "SELL" if d == "BUY" else "BUY"
        if sdl >= 4:
            for item in reversed(sh):
                dr, o, pnl_val = item[0], item[1], (item[2] if len(item) > 2 else 0.0)
                if dr == opp and o == 1:
                    if pnl_val > 300:
                        s = max(0, s - 15)
                        reasons.append(f"large {opp} win (${pnl_val:.0f}) reduces cascade")
                    break'''

assert old_recovery in content, "recovery grace not found!"
content = content.replace(old_recovery, new_recovery, 1)

# Replace extreme streak threshold
old_extreme = '''        # Extreme streak mean reversion: N>=10 converts REJECT to WARN
        if sdl >= 10:'''
new_extreme = '''        # Extreme streak mean reversion: N>=8 converts REJECT to WARN
        if sdl >= 8:'''
assert old_extreme in content, "extreme streak not found!"
content = content.replace(old_extreme, new_extreme, 1)

# Replace all-direction loss counting
old_adl = '''        # ── All-direction consecutive losses ──
        adl = 0
        for _, o, *_ in reversed(sh):
            if o == 0: adl += 1
            else: break'''
new_adl = '''        # ── Rolling window cluster: loss-dense periods ──
        window = min(len(sh), 8)
        if window >= 5:
            recent = list(sh)[-window:]
            same_dir_losses = sum(1 for item in recent if item[0] == d and item[1] == 0)
            same_dir_total = sum(1 for item in recent if item[0] == d)
            if same_dir_total >= 3 and same_dir_losses / same_dir_total >= 0.75:
                cluster_score = 20
                reasons.append(f"loss cluster: {same_dir_losses}/{same_dir_total} {d} in last {window}")
                s = max(s, cluster_score)

        # ── All-direction consecutive losses ──
        adl = 0
        for item in reversed(sh):
            o = item[1]
            if o == 0: adl += 1
            else: break'''
assert old_adl in content, "all-direction not found!"
content = content.replace(old_adl, new_adl, 1)

# Replace global loss counting
old_global = '''        # ── Global consecutive losses ──
        gl = 0; gl_symbols = set()
        for sym_g, _, o in reversed(self._gh):
            if o == 0: gl += 1; gl_symbols.add(sym_g)
            else: break'''
new_global = '''        # ── Global consecutive losses ──
        gl = 0; gl_symbols = set()
        for item in reversed(self._gh):
            sym_g, o = item[0], item[2]
            if o == 0: gl += 1; gl_symbols.add(sym_g)
            else: break'''
assert old_global in content, "global loss counting not found!"
content = content.replace(old_global, new_global, 1)

# ═══ FIX 2: market_memory ═══
# Fix the aggressive WR scoring — require more history, less aggressive thresholds
old_mm_scoring = '''        rec = list(hist)[-10:]; wins = sum(rec); t = len(rec); wr = wins/t
        s = 90 if wr<.2 else (75 if wr<.3 else (55 if wr<.4 else (35 if wr<.5 else max(0,20-wr*30))))
        cl = 0
        for o in reversed(rec):
            if o == 0: cl += 1
            else: break
        if cl >= 3: s = min(100, s+20)
        elif cl >= 2: s = min(100, s+10)'''

new_mm_scoring = '''        rec = list(hist)[-10:]; wins = sum(rec); t = len(rec); wr = wins/t
        # v4 FIX: Less aggressive WR thresholds to reduce false positives
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

assert old_mm_scoring in content, "market_memory scoring not found!"
content = content.replace(old_mm_scoring, new_mm_scoring, 1)

# Also update market_memory min history from 3 to 5
old_mm_min = '        if len(hist) < 3: return FilterResult("market_memory", 0, f"Insufficient history ({len(hist)})")'
new_mm_min = '        if len(hist) < 5: return FilterResult("market_memory", 0, f"Insufficient history ({len(hist)})")'
assert old_mm_min in content, "market_memory min history not found!"
content = content.replace(old_mm_min, new_mm_min, 1)

# Update docstring for FailureCascadeDetector
old_docstring_start = '    """Detects consecutive loss patterns on same symbol or globally.'
new_docstring_start = '    """Detects consecutive and clustered loss patterns on same symbol or globally.'
assert old_docstring_start in content, "failure_cascade docstring not found!"
content = content.replace(old_docstring_start, new_docstring_start, 1)

with open(PATH, 'w') as f:
    f.write(content)

print("All fixes applied successfully!")
print("  1. failure_cascade: 3-tuple storage, PnL tracking, cluster detection")
print("  2. market_memory: less aggressive WR thresholds, min 5 history")
print("  3. regime_transition: unchanged (already optimal at max 40)")
