import json
from pathlib import Path
from collections import Counter
path = Path('logs') / 'execution.log'
blocked = []
with path.open('r', encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        try:
            o=json.loads(line)
        except Exception:
            continue
        if o.get('event') == 'permission.checked' and o.get('allowed') is False:
            blocked.append(o)
first = Counter()
second = Counter()
third = Counter()
redundant_by_gate = Counter()
redundant_trades = 0
for b in blocked:
    fcs = b.get('failed_checks', [])
    if fcs:
        first[fcs[0]] += 1
    if len(fcs) > 1:
        redundant_trades += 1
        for c in fcs[1:]:
            redundant_by_gate[c] += 1
        if len(fcs) >= 2:
            second[fcs[1]] += 1
        if len(fcs) >= 3:
            third[fcs[2]] += 1
print('TOTAL_BLOCKED', len(blocked))
print('FIRST_FAILURE_COUNTS')
for gate, count in first.most_common():
    print(gate, count, f'{count/len(blocked)*100:.1f}%')
print('SECOND_FAILURE_COUNTS')
for gate, count in second.most_common():
    print(gate, count, f'{count/len(blocked)*100:.1f}%')
print('THIRD_FAILURE_COUNTS')
for gate, count in third.most_common():
    print(gate, count, f'{count/len(blocked)*100:.1f}%')
print('REDUNDANT_TRADES', redundant_trades)
print('REDUNDANT_BY_GATE')
for gate, count in redundant_by_gate.most_common():
    print(gate, count, f'{count/len(blocked)*100:.1f}%')
