import unittest

from agents.master_analyst import MasterAnalyst
from risk.trade_permission import TradePermission


class TestAdxAndExecutionFilterTuning(unittest.TestCase):
    def test_master_analyst_adx_context_is_neutral(self):
        analyst = MasterAnalyst()
        base_conf = analyst._calculate_final_confidence(
            llm_conf=80,
            technical_conf=80,
            sentiment_conf=50,
            memory_ctx={},
            smc_ctx={},
            session_ctx={},
            intermarket_ctx={},
            sentiment_ctx={},
            adx_ctx={"adx": 60, "direction": "bullish", "_signal_direction": "bullish"},
        )
        self.assertEqual(base_conf, 80)

    def test_trade_permission_softens_hard_execution_filter_blocks(self):
        tp = TradePermission()
        decision_out = {
            "decision": "BUY",
            "confidence": 75,
            "direct_lane": False,
            "sr_ctx": {},
            "regime": {},
            "_symbol": "EURUSD",
        }
        risk_out = {"approved": True, "entry": 1.1000, "sl_price": 1.0900, "tp_price": 1.1200, "lot": 0.01, "rr_ratio": 2.0}
        execution_filters = {
            "mtf_structure_no_trade": {"blocked": True, "reason": "MTF structure: NO_TRADE"},
        }

        result = tp.check(decision_out, risk_out, news_ctx={"news_trade_allowed": True, "news_reason": "ok"}, execution_filters=execution_filters)

        self.assertTrue(result["allowed"])
        self.assertEqual(result["execution_action"], "BUY")
        self.assertTrue(any(check["check"] == "News safe" for check in result["checks"]))
