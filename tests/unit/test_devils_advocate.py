import pytest

from core.devils_advocate import DevilsAdvocateGate


class TestDevilsAdvocateGate:
    def test_defaults_to_execute_when_disabled(self):
        gate = DevilsAdvocateGate(enabled=False, fail_mode="fail_open")
        result = gate.review(
            trade_context={"symbol": "EURUSD", "decision": "BUY", "confidence": 72},
            signal="BUY",
            risk_out={"approved": True, "entry": 1.1000, "sl_price": 1.0900, "tp_price": 1.1200},
            decision_out={"decision": "BUY", "confidence": 72},
        )

        assert result["decision"] == "EXECUTE"
        assert result["confidence"] >= 0
        assert result["reasons_for_concern"] == []
        assert result["risk_summary"]
        assert result["evidence"] == []

    def test_fail_open_uses_execute_when_provider_unavailable(self, monkeypatch):
        gate = DevilsAdvocateGate(enabled=True, fail_mode="fail_open")

        def _raise(*args, **kwargs):
            raise RuntimeError("provider unavailable")

        monkeypatch.setattr(gate, "_call_provider", _raise)

        result = gate.review(
            trade_context={"symbol": "EURUSD"},
            signal="SELL",
            risk_out={"approved": True},
            decision_out={"decision": "SELL", "confidence": 68},
        )

        assert result["decision"] == "EXECUTE"
        assert result["reasons_for_concern"] == []

    def test_vetoes_when_model_finds_strong_concern(self, monkeypatch):
        gate = DevilsAdvocateGate(enabled=True, fail_mode="fail_open")

        def _stub(*args, **kwargs):
            return {
                "decision": "VETO",
                "confidence": 88,
                "reasons_for_concern": ["Late entry into a stretched move"],
                "risk_summary": "High risk: poor reward-to-risk and volatility spike",
                "evidence": ["ATR expanded 2.5x", "Entry is 18 pips from prior swing high"],
            }

        monkeypatch.setattr(gate, "_call_provider", _stub)

        result = gate.review(
            trade_context={"symbol": "EURUSD", "market_context": {"volatility": "high"}},
            signal="BUY",
            risk_out={"approved": True, "rr_ratio": 1.1},
            decision_out={"decision": "BUY", "confidence": 70},
        )

        assert result["decision"] == "VETO"
        assert result["confidence"] >= 80
        assert result["reasons_for_concern"] == ["Late entry into a stretched move"]
        assert "VETO" not in result["risk_summary"].upper()
