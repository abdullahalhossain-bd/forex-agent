import os, sys
# Ensure project root on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
	sys.path.insert(0, _PROJECT_ROOT)
from risk.trade_permission import TradePermission
import json
perm=TradePermission()
decision_out={"decision":"BUY","confidence":70,"aligned_factors":2,"setup_quality":"B","consecutive_losses":3}
risk_out={"approved":True,"entry":1.0850,"sl_price":1.0830,"tp_price":1.0890,"lot":0.05,"rr_ratio":2.0}
news_ctx={"news_trade_allowed":True,"news_reason":"clear"}
res=perm.check(decision_out=decision_out,risk_out=risk_out,news_ctx=news_ctx,session_ctx=None)
print(json.dumps(res, indent=2))
