import json
from pathlib import Path
path = Path('logs') / 'execution.log'
blocked = []
with path.open('r', encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get('event') == 'permission.checked' and obj.get('allowed') is False:
            blocked.append(obj)
modules = {
    'Risk Engine': ['Risk approved'],
    'Signal Validation': ['Valid signal'],
    'Confidence Gate': ['Min confidence'],
    'Session Filter': ['Session quality'],
    'Confluence Engine': ['Confluence quality'],
    'S/R Zone Filter': ['S/R zone alignment'],
    'Regime Filter': ['Trend alignment (regime)'],
    'Zone Management': ['Zone cooldown (duplicate entry)'],
    'MTF Structure Filter': ['Execution filter: mtf_structure_no_trade'],
    'Confluence Avoid Filter': ['Execution filter: confluence_avoid'],
    'News Filter': ['Execution filter: news_intelligence'],
    'Signal Persistence Filter': ['Signal persistence'],
}
module_trade_counts = {m: 0 for m in modules}
module_check_occurrences = {m: 0 for m in modules}
for b in blocked:
    fcs = b.get('failed_checks', [])
    for m, checks in modules.items():
        if any(c in fcs for c in checks):
            module_trade_counts[m] += 1
        module_check_occurrences[m] += sum(1 for c in fcs if c in checks)
print('TOTAL_BLOCKED', len(blocked))
print('MODULE_COUNTS')
for m in modules:
    print(m, module_trade_counts[m], module_check_occurrences[m], f'{module_trade_counts[m]/len(blocked)*100:.1f}%')
# counts by check
print('CHECK_COUNTS')
check_counts = {}
for b in blocked:
    for c in b.get('failed_checks', []):
        check_counts[c] = check_counts.get(c, 0) + 1
for c,n in sorted(check_counts.items(), key=lambda x:-x[1]):
    print(c, n)
