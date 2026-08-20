import inspect
import core.trader as t
src = inspect.getsource(t)
print('entry fix present:', 'entry=risk_out.get("entry")' in src)
print('risk.finalized fix present:', 'log_event("risk.finalized", **_payload)' in src)
