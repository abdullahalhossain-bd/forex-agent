from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestSessionOverlapGate(unittest.TestCase):
    def test_overlap_session_allows_trade_even_when_fusion_is_rejected(self):
        from analysis.session_analyzer import SessionAnalyzer

        analyzer = SessionAnalyzer()

        def fake_fusion(session, smc_ctx, signal):
            return {
                "fusion_allowed": False,
                "fusion_score": 45,
                "fusion_grade": "C",
                "issues": ["low SMC confluence"],
                "reason": "too weak",
            }

        with patch.object(analyzer, "session_smc_fusion", side_effect=fake_fusion):
            result = analyzer.analyze(
                pair="EURUSD",
                smc_ctx={"smc_score": 45, "smc_factors": {"bos": False}},
                signal_conf=80,
                signal="BUY",
                compute_fusion=True,
                dt=datetime(2026, 7, 14, 14, 30, tzinfo=timezone.utc),
            )

        self.assertEqual(result["session"], "LONDON_NY_OVERLAP")
        self.assertTrue(result["session_trade_allowed"])
        self.assertTrue(result["trade_allowed"])
        self.assertFalse(result["fusion"]["fusion_allowed"])


class TestNewsFilterWindow(unittest.TestCase):
    def test_news_filter_allows_trade_after_default_post_event_window(self):
        from fundamental.news_filter import NewsFilter

        filt = NewsFilter()
        event_time = datetime.now(timezone.utc) - __import__("datetime").timedelta(minutes=40)
        filt._fetch_events = lambda: ([{
            "currency": "USD",
            "high_impact": True,
            "title": "Fed Chair Testifies",
            "time": event_time,
        }], "test")

        result = filt.check("EURUSD")

        self.assertTrue(result["trade_allowed"])
        self.assertEqual(result["reason"], "No high impact news in window")


class TestAdaptiveDecisionCoercion(unittest.TestCase):
    def test_master_decision_coerces_adaptive_labels_to_numeric_values(self):
        from core.master_decision import MasterDecisionEngine

        engine = MasterDecisionEngine()
        engine.strategy_selector = None

        def fake_adaptive(unified_result, current_price=None, mode="confluence"):
            return {
                "action": "BUY",
                "confidence": "High",
                "score": "Medium",
                "source": "mock",
            }

        with patch("core.master_decision.make_adaptive_decision", side_effect=fake_adaptive):
            decision = engine.decide(
                pair="EURUSD",
                timeframe="15m",
                rule_signal="BUY",
                rule_confidence=70.0,
                ml_signal="WAIT",
                ml_confidence=0.0,
                rl_signal="WAIT",
                rl_confidence=0.0,
                llm_signal="WAIT",
                llm_confidence=0.0,
            )

        self.assertEqual(decision.adaptive_action, "BUY")
        self.assertEqual(decision.adaptive_confidence, 80.0)
        self.assertEqual(decision.adaptive_score, 50.0)


if __name__ == "__main__":
    unittest.main()
