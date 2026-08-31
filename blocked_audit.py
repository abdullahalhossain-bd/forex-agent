import json, collections, datetime
from pathlib import Path
path = Path('logs') / 'execution.log'
blocked = []
with path.open('r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get('event') == 'permission.checked' and obj.get('allowed') is False:
            blocked.append(obj)
N = len(blocked)
print('TOTAL_BLOCKED', N)
fields = ['ts', 'symbol', 'allowed', 'decision', 'raw_signal', 'confidence_pre_penalty', 'confidence_post_penalty', 'passed', 'total', 'failed_checks']
for k in fields:
    print('FIELD', k, sum(1 for b in blocked if k in b))
failed = collections.Counter()
sym = collections.defaultdict(lambda: {'count': 0, 'sum_pre': 0.0, 'sum_post': 0.0, 'checks': collections.Counter()})
buckets = collections.Counter()
combo = collections.Counter()
byday = collections.Counter()
byhour = collections.Counter()
for b in blocked:
    for c in b.get('failed_checks', []):
        failed[c] += 1
    s = b.get('symbol', 'UNKNOWN')
    sym[s]['count'] += 1
    sym[s]['sum_pre'] += float(b.get('confidence_pre_penalty') or 0)
    sym[s]['sum_post'] += float(b.get('confidence_post_penalty') or 0)
    sym[s]['checks'].update(b.get('failed_checks', []))
    p = float(b.get('confidence_pre_penalty') or 0)
    if p >= 90:
        buckets['90-100'] += 1
    elif p >= 80:
        buckets['80-89'] += 1
    elif p >= 70:
        buckets['70-79'] += 1
    elif p >= 60:
        buckets['60-69'] += 1
    else:
        buckets['Below 60'] += 1
    c = tuple(sorted(set(b.get('failed_checks', []))))
    if len(c) > 1:
        combo[c] += 1
    try:
        dt = datetime.datetime.fromisoformat(b.get('ts').replace('Z', '+00:00'))
        byday[dt.date().isoformat()] += 1
        byhour[f'{dt.date().isoformat()} {dt.hour:02d}:00'] += 1
    except Exception:
        pass
print('FAILED_COUNTS')
for c, n in failed.most_common():
    print(c, n)
print('PARETO')
cum = 0
for c, n in failed.most_common():
    pct = n * 100 / N
    cum += pct
    print(c, n, f'{pct:.1f}', f'{cum:.1f}')
print('SYMBOL_STATS')
for s, data in sorted(sym.items(), key=lambda x: -x[1]['count']):
    print(s, data['count'], f'{data['sum_pre']/data['count']:.2f}', f'{data['sum_post']/data['count']:.2f}', ';'.join(f'{c}({n})' for c, n in data['checks'].most_common(3)))
print('BUY_SELL')
buy = sum(1 for b in blocked if 'BUY' in str(b.get('decision', '')).upper() or 'BUY' in str(b.get('raw_signal', '')).upper())
sell = sum(1 for b in blocked if 'SELL' in str(b.get('decision', '')).upper() or 'SELL' in str(b.get('raw_signal', '')).upper())
print('BUY', buy, 'SELL', sell, 'TOTAL', N)
for cat in ['BUY', 'SELL']:
    cnt = 0
    fc = collections.Counter()
    for b in blocked:
        if cat in str(b.get('raw_signal', '')).upper() or cat in str(b.get('decision', '')).upper():
            cnt += 1
            fc.update(b.get('failed_checks', []))
    print(cat, cnt)
    for c, n in fc.most_common(5):
        print(' ', c, n)
print('BUCKETS')
for name in ['90-100', '80-89', '70-79', '60-69', 'Below 60']:
    print(name, buckets[name], f'{buckets[name] * 100 / N:.1f}')
print('COMBOS')
for items, n in combo.most_common(20):
    print(' + '.join(items), n)
eq_penalty_by_check = collections.Counter()
eq_penalty_sum = 0.0
eq_penalty_n = 0
for b in blocked:
    for flag in (b.get('entry_quality_failed_checks') or []):
        eq_penalty_by_check[flag] += 1
    if b.get('entry_quality_penalty') is not None:
        eq_penalty_sum += float(b['entry_quality_penalty'])
        eq_penalty_n += 1
print('ENTRY_QUALITY_BREAKDOWN (only present on entries logged after the diagnostic fix)')
if eq_penalty_n:
    print('avg entry_quality_penalty', f'{eq_penalty_sum/eq_penalty_n:.2f}', 'over', eq_penalty_n, 'entries')
    for c, n in eq_penalty_by_check.most_common():
        print(' ', c, n)
else:
    print('  (no entries yet -- run again after new trades are logged)')
print('DAYS')
for day, n in sorted(byday.items()):
    print(day, n)
print('HOURS')
for hour, n in sorted(byhour.items()):
    print(hour, n)