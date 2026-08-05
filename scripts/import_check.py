import sys
from pathlib import Path

# Ensure project root is on sys.path when running from `scripts/` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

print('python', sys.version)
errors = []
try:
    import typed_config
    print('import typed_config OK')
except Exception as e:
    errors.append(('typed_config', str(e)))
try:
    import orchestrator.trading_sessions as ts
    print('import orchestrator.trading_sessions OK')
    s = ts.Sessions()
    print('Sessions OK:', isinstance(s, ts.Sessions))
except Exception as e:
    errors.append(('orchestrator.trading_sessions', str(e)))
try:
    from strategy import retest
    print('import strategy.retest OK')
    # instantiate class
    r = retest.RetestStrategy()
    print('RetestStrategy OK')
except Exception as e:
    errors.append(('strategy.retest', str(e)))
if errors:
    print('ERRORS:')
    for e in errors:
        print(e[0], e[1])
    sys.exit(2)
print('ALL CHECKS PASSED')
