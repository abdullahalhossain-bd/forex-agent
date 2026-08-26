import json

import pytest

import config
from core.devils_advocate import DevilsAdvocateGate, _DevilsAdvocateAuditLog


class TestDevilsAdvocateGate:
    def test_defaults_to_take_when_disabled(self):
        gate = DevilsAdvocateGate(enabled=False, fail_mode="fail_open")
        result = gate.review(
            trade_context={"symbol": "EURUSD", "decision": "BUY", "confidence": 72},
            signal="BUY",
            risk_out={"approved": True, "entry": 1.1000, "sl_price": 1.0900, "tp_price": 1.1200},
            decision_out={"decision": "BUY", "confidence": 72},
        )

        assert result["decision"] == "TAKE"
        assert result["confidence"] >= 0
        assert result["reasons_for_concern"] == []
        assert result["risk_summary"]
        assert result["evidence"] == []

    def test_live_mode_fail_open_defaults_to_reject_on_uncertainty(self, monkeypatch):
        # 2026-08-11 review fix: reviewer outages used to fail-open straight
        # to EXECUTE. That silently let "no opinion" masquerade as approval.
        # The new default policy is conservative: unless an operator opts
        # into the old behavior via DEVILS_ADVOCATE_UNCERTAIN_POLICY=take,
        # a provider outage resolves to REJECT even in fail_open live mode.
        gate = DevilsAdvocateGate(enabled=True, fail_mode="fail_open", mode="live")

        def _raise(*args, **kwargs):
            raise RuntimeError("provider unavailable")

        monkeypatch.setattr(gate, "_call_provider", _raise)

        result = gate.review(
            trade_context={"symbol": "EURUSD"},
            signal="SELL",
            risk_out={"approved": True},
            decision_out={"decision": "SELL", "confidence": 68},
        )

        assert result["decision"] == "REJECT"
        assert result["raw_decision"] == "UNCERTAIN"
        assert result["data_quality"] == "poor"
        assert result["critical_failure"]

    def test_explicit_opt_in_to_take_on_failure(self, monkeypatch):
        gate = DevilsAdvocateGate(
            enabled=True, fail_mode="fail_open", mode="live", uncertain_policy="take"
        )

        def _raise(*args, **kwargs):
            raise RuntimeError("provider unavailable")

        monkeypatch.setattr(gate, "_call_provider", _raise)

        result = gate.review(
            trade_context={"symbol": "EURUSD"},
            signal="SELL",
            risk_out={"approved": True},
            decision_out={"decision": "SELL", "confidence": 68},
        )

        assert result["decision"] == "TAKE"

    def test_research_mode_never_fails_open_even_with_opt_in(self, monkeypatch):
        # Research/backtest mode must never resolve a reviewer outage to
        # TAKE, regardless of uncertain_policy, to avoid contaminating
        # expectancy statistics with "no opinion" masquerading as approval.
        gate = DevilsAdvocateGate(
            enabled=True, fail_mode="fail_open", mode="research", uncertain_policy="take"
        )

        def _raise(*args, **kwargs):
            raise RuntimeError("provider unavailable")

        monkeypatch.setattr(gate, "_call_provider", _raise)

        result = gate.review(
            trade_context={"symbol": "EURUSD"},
            signal="SELL",
            risk_out={"approved": True},
            decision_out={"decision": "SELL", "confidence": 68},
        )

        assert result["decision"] == "REJECT"
        assert result["data_quality"] == "poor"

    def test_backtest_mode_never_fails_open_even_with_opt_in(self, monkeypatch):
        gate = DevilsAdvocateGate(
            enabled=True, fail_mode="fail_open", mode="backtest", uncertain_policy="take"
        )
        monkeypatch.setattr(gate, "_call_provider", lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("provider unavailable")
        ))

        result = gate.review(
            trade_context={"symbol": "EURUSD"},
            signal="BUY",
            risk_out={"approved": True},
            decision_out={"decision": "BUY", "confidence": 68},
        )

        assert result["decision"] == "REJECT"

    def test_finalized_result_preserves_normalized_pillars(self):
        gate = DevilsAdvocateGate(enabled=False)
        result = gate._finalize(
            {
                "decision": "REJECT",
                "confidence": 88,
                "thesis_quality": 0.3,
                "counter_evidence_strength": 0.9,
                "expected_edge": "negative",
                "pillars": {
                    "structure": {"verdict": "FAIL", "reason": "Opposing HTF"},
                    "location": {"verdict": "PASS", "reason": "Room to target"},
                },
                "critical_failure": "model called this a system failure",
                "reasons_for_rejection": ["Opposing HTF"],
            },
            data_quality="good",
        )

        assert result["pillars"]["structure"] == {"verdict": "FAIL", "reason": "Opposing HTF"}
        assert result["pillars"]["location"]["verdict"] == "PASS"
        assert result["fail_pillar_count"] == 1
        assert result["critical_failure"] is None
        assert result["reasons_for_rejection"] == ["Opposing HTF"]

    def test_evidence_contains_distances_and_entry_quality_details(self):
        gate = DevilsAdvocateGate(enabled=False)
        evidence = gate._build_evidence(
            {"symbol": "EURUSD", "market_context": {"session": "London"}, "perm_out": {
                "entry_quality_detail": {
                    "passed_count": 11,
                    "total_count": 13,
                    "results": [{
                        "flag_name": "rejection_psychology",
                        "passed": False,
                        "reason": "No zone-anchored rejection wick found",
                        "details": {"has_rejection_wick": False, "wick_body_ratio": 0.4, "near_zone": False},
                    }],
                }
            }},
            "BUY",
            {"approved": True, "entry": 1.1000, "sl_price": 1.0950, "tp_price": 1.1100},
            {"decision": "BUY", "mtf_trends": {"4h": "sideways", "1h": "unknown"},
             "structure_ctx": {"bos": "BULLISH", "choch": "NONE", "m15": "sideways"},
             "sr_ctx": {"pdh": 1.1050, "pdl": 1.0950, "nearest_resistance": 1.1020}},
        )

        assert evidence["location"]["distance_to_pdh_pips"] == 50.0
        assert evidence["location"]["distance_to_nearest_resistance_pips"] == 20.0
        assert evidence["entry_quality"]["failed_check_details"][0]["reason"] == "No zone-anchored rejection wick found"
        assert evidence["entry_quality"]["rejection_wick_present"] is False
        assert evidence["context"]["htf_conflict"] is False

    def test_audit_row_persists_nested_evidence_and_pillars(self, tmp_path):
        audit = _DevilsAdvocateAuditLog()
        audit.path = tmp_path / "audit.jsonl"
        audit.record(
            "trade-1", {"symbol": "EURUSD"}, "BUY", {"entry": 1.1}, {},
            {"decision": "TAKE", "raw_decision": "TAKE", "pillars": {"structure": {"verdict": "PASS"}},
             "fail_pillar_count": 0, "supporting_evidence": [], "contradicting_evidence": [],
             "reasons_for_rejection": [], "critical_failure": None},
            evidence={"context": {"htf_conflict": False}},
            thesis={"claims": ["Aligned structure"]},
        )

        row = json.loads(audit.path.read_text(encoding="utf-8"))
        assert row["evidence"]["context"]["htf_conflict"] is False
        assert row["pillars"]["structure"]["verdict"] == "PASS"
        assert row["future_outcome"] is None

    def test_rejects_when_model_finds_strong_contradicting_evidence(self, monkeypatch):
        def _stub(*args, **kwargs):
            return {
                "decision": "REJECT",
                "confidence": 88,
                "thesis_quality": 0.3,
                "counter_evidence_strength": 0.9,
                "expected_edge": "negative",
                "risk_level": "high",
                "supporting_evidence": ["H4 trend aligned"],
                "contradicting_evidence": ["ATR expanded 2.5x", "Entry is 18 pips from prior swing high"],
                "reasons_for_rejection": ["Late entry into a stretched move"],
                "critical_failure": None,
            }

        gate = DevilsAdvocateGate(enabled=True, fail_mode="fail_open")
        monkeypatch.setattr(gate, "_call_provider", _stub)

        result = gate.review(
            trade_context={"symbol": "EURUSD", "market_context": {"volatility": "high"}},
            signal="BUY",
            risk_out={"approved": True, "rr_ratio": 1.1},
            decision_out={"decision": "BUY", "confidence": 70},
        )

        assert result["decision"] == "REJECT"
        assert result["raw_decision"] == "REJECT"
        assert result["confidence"] >= 80
        assert result["reasons_for_concern"] == ["Late entry into a stretched move"]
        assert result["counter_evidence_strength"] == 0.9
        assert "REJECT" not in result["risk_summary"].upper() or "reasoning" not in result["risk_summary"]

    def test_takes_when_thesis_survives_review(self, monkeypatch):
        def _stub(*args, **kwargs):
            return {
                "decision": "TAKE",
                "confidence": 74,
                "thesis_quality": 0.8,
                "counter_evidence_strength": 0.2,
                "expected_edge": "positive",
                "risk_level": "low",
                "supporting_evidence": ["H4 and H1 aligned", "Clean displacement"],
                "contradicting_evidence": [],
                "reasons_for_rejection": [],
                "critical_failure": None,
            }

        gate = DevilsAdvocateGate(enabled=True, fail_mode="fail_open")
        monkeypatch.setattr(gate, "_call_provider", _stub)

        result = gate.review(
            trade_context={"symbol": "EURUSD"},
            signal="BUY",
            risk_out={"approved": True, "rr_ratio": 2.0},
            decision_out={"decision": "BUY", "confidence": 80},
        )

        assert result["decision"] == "TAKE"
        assert result["raw_decision"] == "TAKE"

    def test_model_uncertain_resolves_conservatively_by_default(self, monkeypatch):
        def _stub(*args, **kwargs):
            return {
                "decision": "UNCERTAIN",
                "confidence": 40,
                "thesis_quality": 0.5,
                "counter_evidence_strength": 0.5,
                "expected_edge": "unknown",
                "risk_level": "medium",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "reasons_for_rejection": [],
                "critical_failure": None,
            }

        gate = DevilsAdvocateGate(enabled=True, fail_mode="fail_open")
        monkeypatch.setattr(gate, "_call_provider", _stub)

        result = gate.review(
            trade_context={"symbol": "EURUSD"},
            signal="BUY",
            risk_out={"approved": True, "rr_ratio": 1.5},
            decision_out={"decision": "BUY", "confidence": 60},
        )

        # UNCERTAIN must never silently pass through as TAKE.
        assert result["decision"] == "REJECT"
        assert result["raw_decision"] == "UNCERTAIN"

    def test_not_reviewable_trade_defaults_to_take(self):
        gate = DevilsAdvocateGate(enabled=True, fail_mode="fail_open")
        result = gate.review(
            trade_context={"symbol": "EURUSD"},
            signal="WAIT",
            risk_out={"approved": True},
            decision_out={"decision": "BUY", "confidence": 60},
        )

        assert result["decision"] == "TAKE"
        assert result["risk_summary"] == "Not a reviewable trade"

    def test_cloud_review_bypasses_ollama_when_local_backend_is_enabled(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_LOCAL", True)
        gate = DevilsAdvocateGate(enabled=True)

        class _Client:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return type("Response", (), {
                            "choices": [type("Choice", (), {
                                "message": type("Message", (), {
                                    "content": '{"decision":"TAKE"}'
                                })()
                            })()]
                        })()

        class _Manager:
            def get_groq_client(self, **kwargs):
                assert kwargs == {"allow_local": True}
                return _Client()

            def mark_groq_success(self, **kwargs):
                pass

        def _ollama_must_not_run(*args, **kwargs):
            raise AssertionError("Devil's Advocate must not call Ollama")

        monkeypatch.setattr("core.llm_gateway.call_remote_ollama", _ollama_must_not_run)
        result = gate._call_one_provider(_Manager(), "groq", "Return JSON", 9999999999.0)

        assert result == '{"decision":"TAKE"}'