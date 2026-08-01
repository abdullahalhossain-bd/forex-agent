"""Analyze robustness study results and produce final report."""
import json, time, numpy as np
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / 'download'

def st(v):
    if not v: return {'mean':0,'median':0,'std':0,'min':0,'max':0,'count':0}
    a=np.array(v)
    return {'mean':round(float(np.mean(a)),4),'median':round(float(np.median(a)),4),
        'std':round(float(np.std(a)),4),'min':round(float(np.min(a)),4),
        'max':round(float(np.max(a)),4),'count':len(v)}

with open(OUT_DIR / 'lre_robustness_raw.json') as f:
    R = json.load(f)

print(f'Loaded {len(R)} results')

# Separate meaningful (>=5 trades, >=1 W, >=1 L) and active (LRE blocked >0)
mn = [r for r in R if r['t']>=5 and r['W']>=1 and r['L']>=1]
ac = [r for r in mn if r['blk']>0]

wv = [r['wpr'] for r in ac if r['W']>0]
lv = [r['lrr'] for r in ac if r['L']>0]
pvf = [r['lpf'] for r in mn if r['lpf']!=999]
pvb = [r['bpf'] for r in mn if r['bpf']!=999]
nv = [r['lnp'] for r in mn]
nb = [r['bnp'] for r in mn]
dv = [r['ldd'] for r in mn]
db = [r['bdd'] for r in mn]
tc = [r['t'] for r in mn]

# Rankings
wpr_rank = sorted(ac, key=lambda r: r['wpr'], reverse=True)
lrr_rank = sorted(ac, key=lambda r: r['lrr'], reverse=True)
np_rank = sorted(mn, key=lambda r: r['lnp'], reverse=True)

# Failures
wf = [r for r in ac if r['wpr'] < 0.80]
wc = [r for r in ac if r['wpr'] < 0.50]
ws = [r for r in mn if r['lnp'] < r['bnp']]

# FP/TP
tfp = sum(r['blkW'] for r in R)
ttp = sum(r['blkL'] for r in R)
t1fp = sum(r['l1w'] for r in R)
t2fp = sum(r['l2w'] for r in R)
t3fp = sum(r['l3w'] for r in R)
t1tp = sum(r['l1b']-r['l1w'] for r in R)
t2tp = sum(r['l2b']-r['l2w'] for r in R)
t3tp = sum(r['l3b']-r['l3w'] for r in R)

# Timeframe analysis
tfa = {}
for tf in ['M15','H1','H4']:
    tr = [r for r in ac if r['tf']==tf]
    tfa[tf] = {'count': len(tr),
        'wpr': st([r['wpr'] for r in tr if r['W']>0]),
        'lrr': st([r['lrr'] for r in tr if r['L']>0])}

# Category analysis
cat_map = {}
for r in R:
    s = r['sym']
    if 'JPY' in s: c='JPY'
    elif any(m in s for m in ['XAU','XAG','XPT','XPD']): c='metals'
    elif s.endswith('USD'): c='USD_majors'
    elif s.startswith('EUR'): c='EUR'
    elif s.startswith('GBP'): c='GBP'
    elif s.startswith('AUD'): c='AUD'
    elif s.startswith('NZD'): c='NZD'
    elif s.startswith('CAD'): c='CAD'
    elif s.startswith('CHF'): c='CHF'
    elif any(x in s for x in ['HKD','CNH','THB','TRY','MXN','ZAR','SEK','NOK','SGD']): c='exotic_emerging'
    else: c='other'
    cat_map.setdefault(c, []).append(r)
can = {}
for c, cr in sorted(cat_map.items()):
    ca = [r for r in cr if r['blk']>0 and r['W']>0 and r['L']>0]
    cm = [r for r in cr if r['t']>=5]
    cpf = [r['lpf'] for r in cm if r['lpf']!=999]
    can[c] = {'tested': len(cr), 'active': len(ca),
        'wpr': st([r['wpr'] for r in ca]),
        'lrr': st([r['lrr'] for r in ca if r['L']>0]),
        'avg_t': round(np.mean([r['t'] for r in cm]),1) if cm else 0,
        'pf': st(cpf)}

# Production readiness criteria
wm=st(wv)['mean']; wme=st(wv)['median']; lm=st(lv)['mean'];
wsd=st(wv)['std']; lsd=st(lv)['std']

trivial = (len(ac)==0 and len(mn)>0)
crit = {
    'wpr_mean>=80%': wm >= 0.80,
    'wpr_median>=80%': wme >= 0.80,
    'wpr_std<15%': wsd < 0.15,
    'lrr_mean>=20%': lm >= 0.20,
    'no_critical_failures': len(wc)==0,
    'wpr_fail_rate<20%': len(wf)/max(len(ac),1)<0.20,
    'net_profit_not_worse_avg': (np.mean(nv)>=np.mean(nb)) if nv and nb else True,
}
met = sum(crit.values()); tot = len(crit)
if met>=tot*0.8: v,vd='PRODUCTION READY',f'{met}/{tot} met. Acceptable generalization.'
elif met>=tot*0.6: v,vd='CONDITIONALLY READY',f'{met}/{tot} met. Market-specific safeguards needed.'
elif met>=tot*0.4: v,vd='NOT READY - REQUIRES WORK',f'{met}/{tot} met. Significant gaps.'
else: v,vd='NOT READY - OVERFITTED',f'{met}/{tot} met. Likely overfitted to EURUSD H1.'

# Root causes
rc = []
if wf:
    cats=set(); tfs=defaultdict(int); layers=defaultdict(int)
    for r in wf:
        s=r['sym']
        if 'JPY' in s: cats.add('JPY')
        elif any(m in s for m in ['XAU','XAG']): cats.add('metals')
        elif 'USD' in s: cats.add('USD_majors')
        else: cats.add('crosses')
        tfs[r['tf']]+=1
        # Which layer blocked the most wins
        if r['l1w']>0: layers['L1']+=r['l1w']
        if r['l2w']>0: layers['L2']+=r['l2w']
        if r['l3w']>0: layers['L3']+=r['l3w']
    if cats: rc.append(f'WPR failures in: {sorted(cats)}')
    if tfs: rc.append(f'By timeframe: {dict(tfs)}')
    if layers: rc.append(f'FP by layer: {dict(layers)}')
if trivial:
    rc.append('CRITICAL: LRE blocked ZERO trades on most combos. Filter is inert outside EURUSD H1.')

# ═══════════════════════════════════════════════════════════════
#  GENERATE REPORT
# ═══════════════════════════════════════════════════════════════
L = []; a = L.append
a('='*80); a('  LRE ROBUSTNESS STUDY — CROSS-MARKET GENERALIZATION ANALYSIS'); a('='*80)
a(f'  Parameters: FROZEN (no changes from EURUSD H1 optimization)')
a(f'  Combos discovered: 145 | Evaluated: {len(R)} | Meaningful: {len(mn)} | LRE-active: {len(ac)}')
a(f'  Strategy: donchian_breakout + ma_cross')
if trivial: a('  *** WARNING: LRE is trivially passing (never fires on most combos) ***')
a('')
a('-'*80); a('  AGGREGATE STATISTICS (across LRE-active combos where filter fired)'); a('-'*80)
a(f'  {"Metric":<28} {"Mean":>10} {"Median":>10} {"Std":>10} {"Min":>10} {"Max":>10} {"N":>6}')
for k,lbl in [('wpr','Winner Preservation Rate'),('lrr','Loss Rejection Rate'),
    ('lpf','LRE Profit Factor'),('bpf','Baseline Profit Factor'),
    ('lnp','LRE Net Profit ($)'),('bnp','Baseline Net Profit ($)'),
    ('ldd','LRE Max Drawdown ($)'),('bdd','Baseline Max Drawdown ($)'),
    ('t','Total Trade Count')]:
    s=st({'wpr':wv,'lrr':lv,'lpf':pvf,'bpf':pvb,'lnp':nv,'bnp':nb,'ldd':dv,'bdd':db,'t':tc}[k])
    a(f'  {lbl:<28} {s["mean"]:>10.4f} {s["median"]:>10.4f} {s["std"]:>10.4f} {s["min"]:>10.4f} {s["max"]:>10.4f} {s["count"]:>6}')
a('')
a('-'*80); a('  TOP 15 BY WINNER PRESERVATION RATE'); a('-'*80)
a(f'  {"#":<3} {"Symbol":<12} {"TF":<5} {"WPR":>8} {"Trades":>7} {"Wins":>6} {"BlkWins":>8} {"BlkTotal":>9}')
for i,r in enumerate(wpr_rank[:15],1):
    a(f'  {i:<3} {r["sym"]:<12} {r["tf"]:<5} {r["wpr"]:>8.1%} {r["t"]:>7} {r["W"]:>6} {r["blkW"]:>8} {r["blk"]:>9}')
a('')
a('-'*80); a('  WORST 15 BY WPR (WHERE FILTER FAILS)'); a('-'*80)
if ac:
    worst = sorted(ac, key=lambda r: r['wpr'])
    a(f'  {"#":<3} {"Symbol":<12} {"TF":<5} {"WPR":>8} {"BlkWins":>8} {"TotWins":>8}')
    for i,r in enumerate(worst[:15],1):
        a(f'  {i:<3} {r["sym"]:<12} {r["tf"]:<5} {r["wpr"]:>8.1%} {r["blkW"]:>8} {r["W"]:>8}')
a('')
a('-'*80); a('  TOP 15 BY LOSS REJECTION RATE'); a('-'*80)
a(f'  {"#":<3} {"Symbol":<12} {"TF":<5} {"LRR":>8} {"Trades":>7} {"Losses":>7} {"BlkLoss":>8}')
for i,r in enumerate(lrr_rank[:15],1):
    a(f'  {i:<3} {r["sym"]:<12} {r["tf"]:<5} {r["lrr"]:>8.1%} {r["t"]:>7} {r["L"]:>7} {r["blkL"]:>8}')
a('')
a('-'*80); a('  TOP 15 BY LRE NET PROFIT'); a('-'*80)
a(f'  {"#":<3} {"Symbol":<12} {"TF":<5} {"LRE_NP":>10} {"Base_NP":>10} {"Delta":>10}')
for i,r in enumerate(np_rank[:15],1):
    delta = r['lnp']-r['bnp']
    a(f'  {i:<3} {r["sym"]:<12} {r["tf"]:<5} {r["lnp"]:>10.2f} {r["bnp"]:>10.2f} {delta:>+10.2f}')
a('')
a('-'*80); a('  TIMEFRAME ANALYSIS'); a('-'*80)
a(f'  {"TF":<5} {"Active":>7} {"WPR Mean":>10} {"WPR Med":>10} {"WPR Std":>10} {"LRR Mean":>10} {"LRR Med":>10}')
for tf,d in tfa.items():
    a(f'  {tf:<5} {d["count"]:>7} {d["wpr"]["mean"]:>10.4f} {d["wpr"]["median"]:>10.4f} {d["wpr"]["std"]:>10.4f} {d["lrr"]["mean"]:>10.4f} {d["lrr"]["median"]:>10.4f}')
a('')
a('-'*80); a('  SYMBOL CATEGORY ANALYSIS'); a('-'*80)
a(f'  {"Category":<18} {"Combos":>7} {"Active":>7} {"WPR avg":>9} {"LRR avg":>9} {"AvgTrades":>10} {"PF avg":>9}')
for c,d in sorted(can.items()):
    a(f'  {c:<18} {d["tested"]:>7} {d["active"]:>7} {d["wpr"]["mean"]:>9.4f} {d["lrr"]["mean"]:>9.4f} {d["avg_t"]:>10.1f} {d["pf"]["mean"]:>9.4f}')
a('')
a('-'*80); a('  FAILURE ANALYSIS'); a('-'*80)
a(f'  WPR < 80%:   {len(wf)} combos ({len(wf)/max(len(ac),1):.1%} of active)')
a(f'  WPR < 50%:   {len(wc)} combos (CRITICAL)')
a(f'  LRE worse:   {len(ws)} combos ({len(ws)/max(len(mn),1):.1%})')
a(f'  False positives (wins blocked): L1={t1fp} L2={t2fp} L3={t3fp} Total={tfp}')
a(f'  True positives (losses blocked): L1={t1tp} L2={t2tp} L3={t3tp} Total={ttp}')
if tfp+ttp>0:
    prec = ttp/(tfp+ttp)
    a(f'  Overall Precision (TP/(TP+FP)): {prec:.4f}')
    recall = ttp/max(ttp+sum(r['L']-r['blkL'] for r in R if r['L']>0),1)
    a(f'  Overall Recall (blocked losses / total losses): {recall:.4f}')
a('')
if rc:
    a('-'*80); a('  ROOT CAUSE ANALYSIS'); a('-'*80)
    for r in rc: a(f'  >> {r}')
    a('')

# Additional analysis: how many combos had ZERO blocks?
zero_block = [r for r in mn if r['blk']==0]
a('-'*80); a('  LRE ENGAGEMENT ANALYSIS'); a('-'*80)
a(f'  Total meaningful combos: {len(mn)}')
a(f'  Combos where LRE fired (blocked >0): {len(ac)}')
a(f'  Combos where LRE was INERT (blocked 0): {len(zero_block)}')
a(f'  Engagement rate: {len(ac)/max(len(mn),1):.1%}')
if len(zero_block) > 0:
    zero_cats = set()
    for r in zero_block:
        s=r['sym']
        if 'JPY' in s: zero_cats.add('JPY')
        elif any(m in s for m in ['XAU','XAG']): zero_cats.add('metals')
        elif 'USD' in s: zero_cats.add('USD_majors')
        else: zero_cats.add('crosses')
    a(f'  Inert combos in: {sorted(zero_cats)}')
a('')

a('='*80); a(f'  PRODUCTION READINESS VERDICT: {v}'); a('='*80)
a(f'  {vd}'); a('')
a('  Criteria Checklist:')
for c,met2 in crit.items():
    a(f'    [{"PASS" if met2 else "FAIL"}] {c}')
a('')
a('='*80); a('  FINAL ANSWER: Is the Loss Rejection Engine robust enough for production?'); a('='*80)
if 'PRODUCTION' in v:
    a(f'  YES.')
    a(f'  Statistical evidence: WPR mean={wm:.4f}, median={wme:.4f}')
    a(f'  LRR mean={lm:.4f}. Acceptable across {len(ac)} active combinations.')
elif 'CONDITIONALLY' in v:
    a(f'  CONDITIONALLY YES — with significant caveats.')
    a(f'  WPR mean={wm:.4f} (std={wsd:.4f})')
    a(f'  LRR mean={lm:.4f}.')
    if trivial:
        a(f'  CRITICAL CONCERN: LRE was inert on {len(zero_block)}/{len(mn)} combos (never blocked any trades).')
        a(f'  The high WPR is an artifact of non-engagement, not true generalization.')
        a(f'  On the {len(ac)} combos where LRE did fire, WPR was {wm:.4f}.')
    else:
        a(f'  Deploy with market-specific monitoring and circuit breakers.')
else:
    a(f'  NO.')
    a(f'  WPR mean={wm:.4f} (target >= 0.80). {len(wf)}/{len(ac)} active combos fail WPR<80%.')
    if trivial:
        a(f'  CRITICAL: LRE was inert on {len(zero_block)}/{len(mn)} combos.')
        a(f'  The filter does not generalize — it simply does not engage outside its training domain.')
    if rc:
        a(f'  Root causes: {rc}')
a('')
a('='*80)

# Save JSON report
report = {
    'aggregate': {k: st({'wpr':wv,'lrr':lv,'lpf':pvf,'bpf':pvb,'lnp':nv,'bnp':nb,'ldd':dv,'bdd':db,'t':tc}[k])
        for k in ['wpr','lrr','lpf','bpf','lnp','bnp','ldd','bdd','t']},
    'meta': {'total': len(R), 'meaningful': len(mn), 'active': len(ac),
        'zero_block': len(zero_block), 'engagement_rate': round(len(ac)/max(len(mn),1),4),
        'trivial_pass': trivial},
    'failures': {'wpr80': len(wf), 'wpr50': len(wc),
        'wpr_fail_rate': round(len(wf)/max(len(ac),1),4),
        'worse': len(ws), 'worse_rate': round(len(ws)/max(len(mn),1),4),
        'fp_total': tfp, 'tp_total': ttp,
        'fp_l1': t1fp, 'fp_l2': t2fp, 'fp_l3': t3fp,
        'tp_l1': t1tp, 'tp_l2': t2tp, 'tp_l3': t3tp,
        'precision': round(ttp/(tfp+ttp),4) if tfp+ttp>0 else 0,
        'recall': round(ttp/max(ttp+sum(r['L']-r['blkL'] for r in R if r['L']>0),1),4)},
    'timeframe': tfa, 'category': can,
    'verdict': {'status': v, 'detail': vd, 'criteria': {k:bool(v2) for k,v2 in crit.items()}, 'met': f'{met}/{tot}'},
    'root_causes': rc,
    'rankings': {
        'top_wpr': [{'sym':r['sym'],'tf':r['tf'],'wpr':round(r['wpr'],4),'t':r['t'],'W':r['W'],'blkW':r['blkW']} for r in wpr_rank[:15]],
        'worst_wpr': [{'sym':r['sym'],'tf':r['tf'],'wpr':round(r['wpr'],4),'blkW':r['blkW'],'W':r['W']} for r in sorted(ac, key=lambda r:r['wpr'])[:15]],
        'top_lrr': [{'sym':r['sym'],'tf':r['tf'],'lrr':round(r['lrr'],4),'t':r['t'],'L':r['L'],'blkL':r['blkL']} for r in lrr_rank[:15]],
        'top_np': [{'sym':r['sym'],'tf':r['tf'],'lnp':round(r['lnp'],2),'bnp':round(r['bnp'],2)} for r in np_rank[:15]],
    },
    'all_results': R,
}

with open(OUT_DIR / 'lre_robustness_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

txt = '\n'.join(L)
with open(OUT_DIR / 'lre_robustness_report.txt', 'w') as f:
    f.write(txt)

print(txt)
print(f'\nSaved JSON: {OUT_DIR / "lre_robustness_report.json"}')
print(f'Saved TXT: {OUT_DIR / "lre_robustness_report.txt"}')
