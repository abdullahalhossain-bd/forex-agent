from __future__ import annotations

import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agents.decision_agent import DecisionAgent
from risk.trade_permission import TradePermission


def make_synthetic_market_output() -> dict:
    return {
        "symbol": "EURUSD",
        "timeframe": "15m",
        "df": None,
        "ind_ctx": {"close": 1.0850, "price": 1.0850},
        "regime": {"regime_name": "NORMAL"},
        "mtf_bias": {},
    }


def make_synthetic_analysis_out(
    confidence: float = 72.0,
    llm_signal: str = "BUY",
    llm_confidence: float = 80.0,
) -> dict:
    return {
        "final_signal": "BUY",
        "signal": {
            "signal": "BUY",
            "confidence": confidence,
            "entry": 1.0850,
            "sl_pips": 20,
            "tp_pips": 40,
        },
        "llm": {"signal": llm_signal, "confidence": llm_confidence},
        "llm_ctx": {},
        "news": {"trade_allowed": True, "news_reason": "clear"},
        "news_ctx": {"news_trade_allowed": True, "news_reason": "clear"},
        "master_ctx": {"master_signal": "WAIT", "master_confidence": 0.0, "confidence_chain": [("init", 0)]},
        "session_ctx": {"quality": "LOW", "current_session": "TOKYO", "session_strategy": "LOW_VOL", "fusion_allowed": True},
        "execution_filters": {},
        "confidence": confidence,
        "sentiment": {"sentiment_bias": "NEUTRAL", "confidence": 0},
    }


def make_synthetic_risk_out() -> dict:
    return {
        "approved": True,
        "entry": 1.0850,
        "sl_price": 1.0830,
        "tp_price": 1.0890,
        "lot": 0.05,
        "rr_ratio": 2.0,
        "reject_reason": "OK",
    }


class TestWholeDecisionSystem(unittest.TestCase):
    def test_full_decision_permission_flow_allows_high_confidence_low_session(self):
        market_out = make_synthetic_market_output()
        analysis_out = make_synthetic_analysis_out(confidence=72.0)
        risk_out = make_synthetic_risk_out()

        decision = DecisionAgent().decide(market_out, analysis_out, risk_out)

        self.assertEqual(decision["decision"], "BUY")
        self.assertGreaterEqual(decision["confidence"], 55.0)

        permission = TradePermission().check(
            decision_out=decision,
            risk_out=risk_out,
            news_ctx=analysis_out["news_ctx"],
            session_ctx=analysis_out["session_ctx"],
            execution_filters=analysis_out["execution_filters"],
        )

        self.assertTrue(permission["execution_allowed"], permission)
        self.assertEqual(permission["final_action"], "BUY")
        self.assertEqual(permission["blocked_reason"], None)

    def test_full_decision_permission_flow_blocks_low_confidence_low_session(self):
        market_out = make_synthetic_market_output()
        analysis_out = make_synthetic_analysis_out(confidence=45.0)
        risk_out = make_synthetic_risk_out()

        decision = DecisionAgent().decide(market_out, analysis_out, risk_out)

        self.assertEqual(decision["decision"], "BUY")
        self.assertLess(decision["confidence"], 70.0)

        permission = TradePermission().check(
            decision_out=decision,
            risk_out=risk_out,
            news_ctx=analysis_out["news_ctx"],
            session_ctx=analysis_out["session_ctx"],
            execution_filters=analysis_out["execution_filters"],
        )

        self.assertFalse(permission["execution_allowed"])
        self.assertEqual(permission["final_action"], "NO TRADE")
        self.assertIsNotNone(permission["blocked_reason"])

    def test_full_decision_permission_flow_blocks_when_session_fusion_filter_applies(self):
        market_out = make_synthetic_market_output()
        analysis_out = make_synthetic_analysis_out(confidence=80.0)
        risk_out = make_synthetic_risk_out()

        # Simulate a hard execution filter from session fusion
        analysis_out["execution_filters"] = {
            "fusion": {
                "blocked": True,
                "reason": "Fusion gate: SMC fusion rejected",
            }
        }

        decision = DecisionAgent().decide(market_out, analysis_out, risk_out)
        self.assertEqual(decision["decision"], "BUY")

        permission = TradePermission().check(
            decision_out=decision,
            risk_out=risk_out,
            news_ctx=analysis_out["news_ctx"],
            session_ctx=analysis_out["session_ctx"],
            execution_filters=analysis_out["execution_filters"],
        )

        self.assertFalse(permission["execution_allowed"])
        self.assertEqual(permission["final_action"], "NO TRADE")
        self.assertIn("fusion", permission["failed_checks"][0]["check"].lower())
        self.assertIn("fusion gate", permission["blocked_reason"].lower())


if __name__ == "__main__":
    unittest.main()
