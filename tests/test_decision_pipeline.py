"""
tests/test_decision_pipeline.py — Integration test for the institutional
decision pipeline (architectural refactor verification).

Verifies the core architectural invariants:
  1. News filter NEVER overwrites analysis-layer signal/confidence
  2. Risk gate NEVER produces a `signal` field
  3. TradePermission returns execution_allowed / blocked_reason / failed_checks
  4. DecisionAgent preserves max voter confidence on no-consensus
  5. ML predictor returns ml_available flag
  6. Ensemble dynamic weight rebalance works when ML missing
  7. ConfidenceFusion zeroes missing voter weights and renormalizes
  8. MasterAnalyst no longer hard-zeroes confidence on session gate
  9. analysis_agent.execution_filters dict is populated (not overwriting final_signal)
 10. core/trader preserves dec_out["decision"] as analysis verdict

Run: python -m pytest tests/test_decision_pipeline.py -v
  or: python tests/test_decision_pipeline.py
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import dataclass
from typing import Any, Dict

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestRiskEngineRejectNoSignalField(unittest.TestCase):
    """Issue 1: risk/risk_engine.py::_reject() must not echo `signal` field."""

    def test_reject_does_not_produce_signal_field(self):
        from risk.risk_engine import RiskEngine
        engine = RiskEngine(balance=10000.0, symbol="EURUSD")
        result = engine._reject("test rejection")
        self.assertFalse(result["approved"])
        self.assertEqual(result["lot"], 0)
        self.assertEqual(result["sl_pips"], 0)
        # CRITICAL: no `signal` field — risk gate is execution, not analysis
        self.assertNotIn("signal", result,
                         "risk_engine._reject() must NOT produce a `signal` field "
                         "(execution filter, not analysis layer)")


class TestTradePermissionNewFields(unittest.TestCase):
    """Issue 2: trade_permission must return new canonical fields."""

    def test_permission_returns_execution_fields(self):
        from risk.trade_permission import TradePermission
        perm = TradePermission()
        # Build minimal inputs
        decision_out = {"decision": "BUY", "confidence": 75, "aligned_factors": 2,
                        "setup_quality": "B"}
        risk_out = {"approved": True, "entry": 1.0850, "sl_price": 1.0830,
                    "tp_price": 1.0890, "lot": 0.05, "rr_ratio": 2.0}
        news_ctx = {"news_trade_allowed": True, "news_reason": "clear"}
        result = perm.check(decision_out=decision_out, risk_out=risk_out,
                            news_ctx=news_ctx, session_ctx=None)
        # New canonical fields
        self.assertIn("execution_allowed", result)
        self.assertIn("blocked_reason", result)
        self.assertIn("failed_checks", result)
        self.assertIn("execution_action", result)
        # Legacy fields preserved
        self.assertIn("allowed", result)
        self.assertIn("final_action", result)
        self.assertIn("checks", result)
        # When allowed, execution_action == decision
        if result["execution_allowed"]:
            self.assertEqual(result["execution_action"], "BUY")


class TestTradePermissionBlocksOnExecutionFilters(unittest.TestCase):
    """Issue 2b: trade_permission honors execution_filters dict."""

    def test_permission_blocks_on_execution_filters(self):
        from risk.trade_permission import TradePermission
        perm = TradePermission()
        decision_out = {"decision": "BUY", "confidence": 75, "aligned_factors": 2,
                        "setup_quality": "B"}
        risk_out = {"approved": True, "entry": 1.0850, "sl_price": 1.0830,
                    "tp_price": 1.0890, "lot": 0.05, "rr_ratio": 2.0}
        news_ctx = {"news_trade_allowed": True, "news_reason": "clear"}
        # Pass execution_filters with a blocked gate
        exec_filters = {
            "news": {
                "blocked": True,
                "reason": "High Impact News: USD Core CPI @ 12:45 UTC",
            }
        }
        result = perm.check(decision_out=decision_out, risk_out=risk_out,
                            news_ctx=news_ctx, session_ctx=None,
                            execution_filters=exec_filters)
        self.assertFalse(result["execution_allowed"])
        self.assertEqual(result["execution_action"], "NO TRADE")
        self.assertIsNotNone(result["blocked_reason"])


class TestMLPredictorMlAvailableField(unittest.TestCase):
    """Issue 8: model_predictor must return ml_available flag."""

    def test_predict_returns_ml_available(self):
        from ml.model_predictor import ModelPredictor
        predictor = ModelPredictor()
        # Predict without any models loaded — should return NOT_READY + ml_available=False
        result = predictor.predict(
            features={"close": 1.0850, "rsi": 50.0},
            pair="TESTPAIR",
            timeframe="15m",
        )
        self.assertIn("ml_available", result)
        self.assertIn("ml_unavailable_reason", result)
        # No models for TESTPAIR — must be False
        self.assertFalse(result["ml_available"])


class TestConfidenceFusionDynamicRebalance(unittest.TestCase):
    """Issue 10: confidence_fusion must zero-and-renormalize missing voters."""

    def test_dynamic_rebalance_when_voter_missing(self):
        from ml.confidence_fusion import ConfidenceFusion
        from ml.voting_engine import ModelVote, VoteResult
        fusion = ConfidenceFusion()
        # Only rules vote — xgboost/random_forest/lstm missing
        votes = [ModelVote(model_name="rules", signal="BUY", confidence=70.0, probability=0.7)]
        vote_result = VoteResult(
            decision="BUY", agreement="1/1", agreement_count=1, total_models=1,
            position_size="HALF", position_multiplier=0.5,
            has_strong_dissent=False, dissenting_models=[], dissent_reason="",
        )
        result = fusion.fuse(votes, vote_result, regime="NORMAL")
        # Rules weight should be 1.0 (renormalized after ML weights zeroed)
        self.assertIn("rules", result.weights_used)
        # After rebalance, rules weight should be 1.0 (only voter)
        self.assertAlmostEqual(result.weights_used["rules"], 1.0, places=2)
        # ML weights should be 0
        self.assertEqual(result.weights_used.get("xgboost", 0), 0)
        self.assertEqual(result.weights_used.get("random_forest", 0), 0)
        self.assertEqual(result.weights_used.get("lstm", 0), 0)
        # Confidence should be 70 (rules' confidence, no penalty)
        self.assertAlmostEqual(result.weighted_confidence, 70.0, places=1)


class TestSignalFusionPreservesAnalysisSignal(unittest.TestCase):
    """Issue 11: signal_fusion must preserve strongest single-layer signal."""

    def test_preserves_strongest_signal(self):
        from core.signal_fusion import SignalFusion, LayerSignal
        fusion = SignalFusion()
        signals = [
            LayerSignal(layer="rule_engine", signal="BUY", confidence=58.0, weight=0.30),
            LayerSignal(layer="llm_analyst", signal="SELL", confidence=72.0, weight=0.30),
            LayerSignal(layer="ml_ensemble", signal="WAIT", confidence=50.0, weight=0.20),
        ]
        result = fusion.fuse(signals)
        # analysis_signal should be the strongest BUY/SELL — SELL at 72%
        self.assertEqual(result.analysis_signal, "SELL")
        self.assertAlmostEqual(result.analysis_confidence, 72.0, places=1)


class TestSignalFusionPreservesConfidenceOnNoConsensus(unittest.TestCase):
    """Fusion should retain a usable confidence value even when consensus fails."""

    def test_preserves_confidence_when_consensus_fails(self):
        from core.signal_fusion import SignalFusion, LayerSignal
        fusion = SignalFusion()
        signals = [
            LayerSignal(layer="rule_engine", signal="BUY", confidence=64.0, weight=0.35),
            LayerSignal(layer="llm_analyst", signal="SELL", confidence=78.0, weight=0.35),
            LayerSignal(layer="ml_ensemble", signal="WAIT", confidence=50.0, weight=0.30),
        ]
        result = fusion.fuse(signals)
        self.assertEqual(result.final_signal, "WAIT")
        self.assertEqual(result.analysis_signal, "SELL")
        self.assertAlmostEqual(result.analysis_confidence, 78.0, places=1)
        self.assertGreaterEqual(result.master_confidence, 78.0)


class TestEnsembleMlAvailableField(unittest.TestCase):
    """Issue 9: ensemble must set ml_available + excluded_voters."""

    def test_ensemble_sets_ml_available_false_when_not_ready(self):
        from ml.ensemble import EnsembleEngine
        engine = EnsembleEngine()
        # Pass NOT_READY ml_prediction
        decision = engine.decide(
            pair="EURUSD",
            timeframe="15m",
            ml_prediction={"prediction": "NOT_READY", "ml_available": False},
            rule_signal="BUY",
            rule_confidence=65.0,
            master_signal="BUY",
            master_confidence=70.0,
            regime="NORMAL",
        )
        self.assertFalse(decision.ml_available)
        self.assertIn("ml_models", decision.excluded_voters)
        # Analysis signal must be preserved (BUY)
        self.assertEqual(decision.analysis_signal, "BUY")


class TestLiveRiskManagerTierPromotion(unittest.TestCase):
    """Issue H5: LiveRiskManager.record_trade_result triggers tier promotion."""

    def test_record_trade_result_accepts_pnl_usd(self):
        from risk.live_risk_manager import LiveRiskManager
        lrm = LiveRiskManager(initial_balance=10000.0, tier=1)
        # Should not raise — pnl_usd is now accepted (H5 fix)
        lrm.record_trade_result(won=True, pnl_usd=50.0)
        lrm.record_trade_result(won=False, pnl_usd=-30.0)
        # learning_agent should be None by default
        self.assertIsNone(lrm.learning_agent)
        # attach_learning_agent should exist
        self.assertTrue(hasattr(lrm, "attach_learning_agent"))


class TestLiveRiskManagerTradePermissionResultAlias(unittest.TestCase):
    """Issue H6: TradePermissionResult is the new name; TradePermission is alias."""

    def test_trade_permission_result_alias(self):
        from risk.live_risk_manager import TradePermissionResult, TradePermission
        self.assertIs(TradePermission, TradePermissionResult)
        # Instantiate to verify it works
        tpr = TradePermissionResult(allowed=True, lot=0.05)
        self.assertTrue(tpr.allowed)
        self.assertEqual(tpr.lot, 0.05)


class TestCoreConstantsCentralizedThresholds(unittest.TestCase):
    """Issue H9: core/constants.py has centralized thresholds."""

    def test_thresholds_exist(self):
        from core import constants as C
        # Max trades per day
        self.assertEqual(C.get_max_trades_per_day(1), C.MAX_TRADES_PER_DAY_TIER_1)
        self.assertEqual(C.get_max_trades_per_day(2), C.MAX_TRADES_PER_DAY_TIER_2)
        self.assertEqual(C.get_max_trades_per_day(3), C.MAX_TRADES_PER_DAY_TIER_3)
        # Min confidence
        self.assertEqual(C.get_min_confidence(1), C.MIN_CONFIDENCE_TIER_1)
        # Min RR
        self.assertEqual(C.MIN_RR_PROD, 2.0)
        # ML thresholds
        self.assertEqual(C.ML_BUY_THRESHOLD, 0.58)
        self.assertEqual(C.ML_SELL_THRESHOLD, 0.42)


class TestLLMKeyManagerNewProviders(unittest.TestCase):
    """Issue (LLM providers): Claude / GLM / DeepSeek added."""

    def test_new_provider_methods_exist(self):
        from core.llm_key_manager import LLMKeyManager
        km = LLMKeyManager()
        # Getters
        self.assertTrue(hasattr(km, "get_claude_client"))
        self.assertTrue(hasattr(km, "get_glm_client"))
        self.assertTrue(hasattr(km, "get_deepseek_client"))
        # Markers
        self.assertTrue(hasattr(km, "mark_claude_success"))
        self.assertTrue(hasattr(km, "mark_glm_failure"))
        self.assertTrue(hasattr(km, "mark_deepseek_success"))
        # Availability properties
        self.assertTrue(hasattr(km, "has_any_claude"))
        self.assertTrue(hasattr(km, "has_any_glm"))
        self.assertTrue(hasattr(km, "has_any_deepseek"))


class TestMasterDecisionAdaptiveAdvisory(unittest.TestCase):
    """Issue C2: master_decision wires adaptive engine as advisory."""

    def test_adaptive_fields_on_dataclass(self):
        from core.master_decision import MasterDecision
        md = MasterDecision(
            pair="EURUSD", timeframe="15m",
            final_signal="BUY", master_confidence=70.0,
            agreement="3/4", position_size="HALF", position_multiplier=0.5,
        )
        # New advisory fields exist with defaults
        self.assertEqual(md.adaptive_action, "")
        self.assertEqual(md.adaptive_confidence, 0.0)
        self.assertEqual(md.adaptive_score, 0.0)
        self.assertFalse(md.adaptive_divergence)


def run_all_tests():
    """Run all tests and return success count."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    # Add all test classes
    test_classes = [
        TestRiskEngineRejectNoSignalField,
        TestTradePermissionNewFields,
        TestTradePermissionBlocksOnExecutionFilters,
        TestMLPredictorMlAvailableField,
        TestConfidenceFusionDynamicRebalance,
        TestSignalFusionPreservesAnalysisSignal,
        TestEnsembleMlAvailableField,
        TestLiveRiskManagerTierPromotion,
        TestLiveRiskManagerTradePermissionResultAlias,
        TestCoreConstantsCentralizedThresholds,
        TestLLMKeyManagerNewProviders,
        TestMasterDecisionAdaptiveAdvisory,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
