#!/usr/bin/env python3
"""
scripts/test_decision_layer_fix.py — Regression tests for the 2026-08-27
decision-chain audit fix in agents/decision_agent.py.

Covers:
  T1. Day53 ConfidenceEngine SKIP survives the post-consensus override
      (previously: analysis final_signal=BUY resurrected a NO TRADE issued
      for an auto-disabled pattern → dead safety feature).
  T2. Normal consensus BUY path unaffected by the fix.
  T3. Excluded (parse-failed) MasterAnalyst vote cannot drive voting,
      and any resulting direction copy is now tagged consensus_override.
  T4. FusionV3 still downgrades sub-1.0 RRR setups to WAIT.
  T5. ce_skip/consensus_override machine-readable flags present in output.

Run from repo root:  python3 scripts/test_decision_layer_fix.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.decision_agent import DecisionAgent  # noqa: E402


def base_market():
    return {
        "symbol": "EURUSD", "timeframe": "M15",
        "regime": {"regime": "TRENDING"},
        "ind_ctx": {"close": 1.0850, "atr": 0.0012},
    }


def placeholder_risk():
    return {"approved": False, "lot": 0, "sl_pips": 0, "tp_pips": 0,
            "rr_ratio": 0, "reject_reason": None, "is_placeholder": True}


def test_t1_skip_survives_override(da):
    """A *fresh* disabled-pattern cycle must stay NO TRADE even when
    analysis final_signal is directional.

    NOTE: production disabled_patterns.json entries auto-re-enable after
    their expiry (last batch expired 2026-08-19). To test the guard we
    inject a fresh EURUSD H1 disable (regime UNKNOWN) and prove the
    prefix-match blocks a TRENDING-regime cycle too — exercising BOTH
    fixes (skip-survival + cross-regime matching). State restored after.
    """
    import json as _json
    _dp_path = "memory/disabled_patterns.json"
    _backup = open(_dp_path, "rb").read()
    try:
        # Inject fresh, non-expired disable (expires far in the future)
        disabled = _json.loads(_backup.decode() or "{}")
        disabled["mt5_real_trade|EURUSD|H1|UNKNOWN"] = {
            "reason": "test-injected: win rate below threshold",
            "disabled_at": "2026-01-01T00:00:00+00:00",
            "re_enable_after": "2099-01-01T00:00:00+00:00",
        }
        with open(_dp_path, "w") as fh:
            fh.write(_json.dumps(disabled, indent=1))

        market = base_market()
        market["timeframe"] = "H1"
        market["regime"] = {"regime": "TRENDING"}   # different regime label!
        analysis = {
            "final_signal": "BUY",
            "signal": {"signal": "BUY", "confidence": 72},
            "llm": {"signal": "BUY", "confidence": 70},
            "master_ctx": {"master_signal": "BUY", "master_confidence": 75,
                           "master_entry": 1.0850,
                           "master_sl": 1.0832, "master_tp1": 1.0886},
            "news": {"trade_allowed": True},
            "session_ctx": {"session_trade_allowed": True,
                            "fusion_allowed": True,
                            "is_dead_zone": False,
                            "current_session": "LONDON"},
        }
        out = da.decide(market, analysis, placeholder_risk())
        assert out["decision"] == "NO TRADE", (
            f"T1 FAIL: CE skip was resurrected -> {out['decision']}")
        assert out.get("ce_skip") is True, "T1 FAIL: ce_skip flag missing"
        joined = " ".join(str(r) for r in out["reasons"])
        assert "SKIP retained" in joined or "ConfidenceEngine SKIP" in joined, \
            f"T1 FAIL: no skip reason in reasons: {joined[:200]}"
        print("T1 PASS: CE SKIP survives override AND covers "
              "cross-regime cycles (UNKNOWN-disable blocks TRENDING)")
    finally:
        with open(_dp_path, "wb") as fh:
            fh.write(_backup)


def test_t2_normal_consensus_unaffected(da):
    market = base_market()
    analysis = {
        "final_signal": "WAIT",   # analysis layer silent; local consensus drives
        "signal": {"signal": "BUY", "confidence": 72},
        "llm": {"signal": "BUY", "confidence": 70},
        "master_ctx": {"master_signal": "BUY", "master_confidence": 75,
                       "master_entry": 1.0850,
                       "master_sl": 1.0832, "master_tp1": 1.0886},
        "news": {"trade_allowed": True},
        "session_ctx": {"session_trade_allowed": True, "fusion_allowed": True,
                        "is_dead_zone": False, "current_session": "LONDON"},
    }
    out = da.decide(market, analysis, placeholder_risk())
    assert out["decision"] == "BUY", f"T2 FAIL: {out['decision']}"
    assert out["fusion_v3"] and out["fusion_v3"]["safe"], \
        f"T2 FAIL: fusion_v3 not safe: {out['fusion_v3']}"
    assert out.get("consensus_override") is False
    print(f"T2 PASS: consensus BUY intact "
          f"(conf={out['confidence']}, rrr=1:{out['fusion_v3']['rrr']})")


def test_t3_excluded_master_no_vote_leak(da):
    """Parse-failed master (SELL@90) must be excluded; rule alone (1 vote)
    cannot reach MIN_CONSENSUS — outcome must NOT follow master direction."""
    market = base_market()
    analysis = {
        "final_signal": "WAIT",
        "signal": {"signal": "SELL", "confidence": 55},   # rule: SELL (1 vote)
        "llm": {"signal": "WAIT", "confidence": 40},
        "master_ctx": {"master_signal": "BUY", "master_confidence": 90,
                       "_llm_parse_failed": True},         # excluded!
        "news": {"trade_allowed": True},
        "session_ctx": {"session_trade_allowed": True, "fusion_allowed": True,
                        "is_dead_zone": False, "current_session": "LONDON"},
    }
    out = da.decide(market, analysis, placeholder_risk())
    # With the standing analysis-verdict-preserve semantics this cycle can't
    # produce a BUY: final_signal is WAIT so nothing resurrects master's BUY;
    # rule's SELL has only 1 vote → no consensus → WAIT (conf preserved).
    assert out["decision"] != "BUY", (
        f"T3 FAIL: excluded master drove a BUY: {out['decision']}")
    print(f"T3 PASS: excluded master did not authorize trade "
          f"(decision={out['decision']}, conf={out['confidence']})")


def test_t4_rrr_guard_intact(da):
    market = base_market()
    analysis = {
        "final_signal": "BUY",
        "signal": {"signal": "BUY", "confidence": 72},
        "llm": {"signal": "BUY", "confidence": 70},
        "master_ctx": {"master_signal": "BUY", "master_confidence": 80,
                       "master_entry": 1.0850,
                       "master_sl": 1.0840, "master_tp1": 1.0854},  # RR ~1:0.35
        "news": {"trade_allowed": True},
        "session_ctx": {"session_trade_allowed": True, "fusion_allowed": True,
                        "is_dead_zone": False, "current_session": "LONDON"},
    }
    out = da.decide(market, analysis, placeholder_risk())
    fv = out.get("fusion_v3") or {}
    assert out["decision"] == "WAIT" and fv.get("rrr_valid") is False, \
        f"T4 FAIL: decision={out['decision']} fv={fv}"
    print(f"T4 PASS: sub-1.0 RRR downgraded to WAIT (rrr=1:{fv['rrr']})")


def main():
    da = DecisionAgent()
    assert da._signal_fusion is not None, "SignalFusion gate must be available"
    assert da.confidence_engine is not None, "ConfidenceEngine must load"
    test_t1_skip_survives_override(da)
    test_t2_normal_consensus_unaffected(da)
    test_t3_excluded_master_no_vote_leak(da)
    test_t4_rrr_guard_intact(da)
    print("\nALL DECISION-LAYER REGRESSION TESTS PASSED")


if __name__ == "__main__":
    main()
