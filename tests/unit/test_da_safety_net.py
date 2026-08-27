"""
tests/unit/test_da_safety_net.py — 2026-08-27 six-point deterministic safety net.

Covers each check independently plus the end-to-end short-circuit inside
DevilsAdvocateGate.review() (a veto must return REJECT without ever
reaching the LLM provider call).
"""

import pytest

from core.da_safety_net import (
    DASafetyNet,
    VERDICT_PASS,
    VERDICT_SKIP,
    VERDICT_VETO,
    VERDICT_WARN,
)


def _mk(**overrides):
    """Build a clean, fully-passing-ish input bundle."""
    trade_context = {
        "symbol": "EURUSD",
        "market_context": {
            "sessions_active": ["london"],
            "h4_trend": "bullish",
        },
    }
    risk_out = {
        "approved": True,
        "entry": 1.1000,
        "sl_price": 1.0970,   # 30 pips
        "sl_pips": 30,
        "balance": 1000.0,
        "risk_pc": 1.0,       # $10 risk -> correct lot = 10/(30*10) = 0.0333
        "lot": 0.03,          # within 25% of 0.0333
    }
    decision_out = {
        "decision": "BUY",
        "ind_ctx": {"atr": 0.0015, "spread_pips": 1.2},
        "sr_ctx": {},          # no structure levels by default
    }
    for k, v in overrides.pop("tc", {}).items():
        trade_context[k] = v
    for k, v in overrides.pop("mc", {}).items():
        trade_context["market_context"][k] = v
    risk_out.update(overrides.pop("risk", {}))
    decision_out.update(overrides.pop("dec", {}))
    assert not overrides, f"unused override keys: {list(overrides)}"
    return trade_context, risk_out, decision_out


def _verdict_of(checks, name):
    for c in checks:
        if c["check"] == name:
            return c["verdict"]
    raise AssertionError(f"check {name} missing from {checks}")


class TestSessionFilter:
    def test_major_session_passes(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(mc={"sessions_active": ["london", "tokyo"]})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "session_filter") == VERDICT_PASS

    def test_asian_chop_vetoes_non_asian_pair(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(mc={"sessions_active": ["tokyo", "sydney"]})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "session_filter") == VERDICT_VETO

    def test_asian_session_warns_for_jpy_pair(self):
        net = DASafetyNet()
        tc, risk, dec = _mk()
        tc["symbol"] = "USDJPY"
        tc["market_context"]["sessions_active"] = ["tokyo"]
        out = net.run(tc, "SELL", risk, dec)
        assert _verdict_of(out["checks"], "session_filter") == VERDICT_WARN

    def test_empty_sessions_is_market_closed_veto(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(mc={"sessions_active": []})
        out = net.run(tc, "BUY", risk, dec)
        v = next(c for c in out["checks"] if c["check"] == "session_filter")
        assert v["verdict"] == VERDICT_VETO and "closed" in v["reason"].lower()

    def test_no_data_skips_unless_wallclock_authorized(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(mc={"sessions_active": None})
        tc["market_context"].pop("sessions_active")
        tc["market_context"]["session"] = None
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "session_filter") == VERDICT_SKIP

    def test_wallclock_path_used_when_authorized(self, monkeypatch):
        from utils.session import SessionAnalyzer
        monkeypatch.setattr(
            SessionAnalyzer, "get_current_session",
            lambda self: {"active_sessions": ["sydney"], "trade_quality": "x"},
        )
        net = DASafetyNet()
        tc, risk, dec = _mk(mc={"sessions_active": None})
        tc["market_context"].pop("sessions_active")
        tc["market_context"].pop("session", None)
        tc["da_allow_wallclock_session"] = True
        out = net.run(tc, "BUY", risk, dec)
        # sydney-only + EURUSD => asian chop veto via live clock path
        assert _verdict_of(out["checks"], "session_filter") == VERDICT_VETO

    def test_quality_string_via_legacy_key_skips_not_vetoes(self):
        # regression guard: legacy "session" key holding a trade-QUALITY
        # string must never be parsed as a session name and vetoed
        net = DASafetyNet()
        tc, risk, dec = _mk(mc={"sessions_active": None})
        tc["market_context"].pop("sessions_active")
        tc["market_context"]["session"] = "\U0001F7E2 BEST \u2014 highest liquidity"
        out = net.run(tc, "BUY", risk, dec)
        v = next(c for c in out["checks"] if c["check"] == "session_filter")
        assert v["verdict"] == VERDICT_SKIP

    def test_unrecognized_sessions_active_skips_not_market_closed(self):
        # caller passes junk names — must SKIP, not false "market closed" veto
        net = DASafetyNet()
        tc, risk, dec = _mk(mc={"sessions_active": ["asia", "europe"]})
        out = net.run(tc, "BUY", risk, dec)
        v = next(c for c in out["checks"] if c["check"] == "session_filter")
        assert v["verdict"] == VERDICT_SKIP

    def test_spaced_session_name_normalized(self):
        # "London/New York" style strings split AND space-normalize correctly
        net = DASafetyNet()
        tc, risk, dec = _mk()
        tc["market_context"].pop("sessions_active")
        tc["market_context"]["session"] = "London/New York"
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "session_filter") == VERDICT_PASS


class TestTrendFilter:
    def test_h4_opposing_buy_vetoed(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(mc={"h4_trend": "downtrend"})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "h4_trend_filter") == VERDICT_VETO

    def test_h4_aligned_passes(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(mc={"h4_trend": "bullish"})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "h4_trend_filter") == VERDICT_PASS

    def test_sideways_neutral_does_not_veto(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(mc={"h4_trend": "sideways"})
        out = net.run(tc, "SELL", risk, dec)
        assert _verdict_of(out["checks"], "h4_trend_filter") == VERDICT_PASS

    def test_mtf_trends_fallback(self):
        net = DASafetyNet()
        tc, risk, dec = _mk()
        tc["market_context"].pop("h4_trend")
        dec["mtf_trends"] = {"4h": "bearish"}
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "h4_trend_filter") == VERDICT_VETO


class TestSpreadCheck:
    def test_spread_within_limit_passes(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(dec={"ind_ctx": {"atr": 0.0015, "spread_pips": 1.2}})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "spread_check") == VERDICT_PASS

    def test_spread_over_limit_vetoed(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(dec={"ind_ctx": {"atr": 0.0015, "spread_pips": 3.5}})
        out = net.run(tc, "BUY", risk, dec)  # EURUSD max 2.0
        assert _verdict_of(out["checks"], "spread_check") == VERDICT_VETO

    def test_spread_eating_atr_vetoed_even_under_pip_limit(self):
        # XAUUSD allows 5 pips but a 3-pip spread vs tiny ATR destroys edge
        net = DASafetyNet()
        tc, risk, dec = _mk()
        tc["symbol"] = "XAUUSD"
        dec["ind_ctx"] = {"atr": 0.05, "spread_pips": 3.0}  # atr=500p ok ratio; use tighter
        dec["ind_ctx"] = {"atr": 0.006, "spread_pips": 3.0}  # atr=60p, ratio .05 passable
        dec["ind_ctx"] = {"atr": 0.0006, "spread_pips": 3.0}  # broken pip math guard
        # instead craft directly: gold pip size 0.01, ATR 8 pips, spread 3 → 37% pass;
        # make spread 4 → 50% veto (>35%) while under absolute limit of 5
        dec["ind_ctx"] = {"atr": 0.08, "spread_pips": 4.0}
        out = net.run(tc, "SELL", risk, dec)
        c = next(c for c in out["checks"] if c["check"] == "spread_check")
        assert c["verdict"] == VERDICT_VETO and "ATR" in c["reason"]

    def test_spread_near_limit_warns(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(dec={"ind_ctx": {"atr": 0.010, "spread_pips": 1.8}})
        out = net.run(tc, "BUY", risk, dec)  # 1.8 > 0.7*2.0, under it
        assert _verdict_of(out["checks"], "spread_check") == VERDICT_WARN

    def test_news_window_halves_limit(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(dec={"ind_ctx": {"atr": 0.010, "spread_pips": 1.2}})
        tc["analysis_out"] = {"news_ctx": {"risk_level": "high"}}
        out = net.run(tc, "BUY", risk, dec)  # news max = 1.0 → 1.2 > 1.0 veto
        assert _verdict_of(out["checks"], "spread_check") == VERDICT_VETO


class TestAtrRegime:
    def test_news_spike_candle_vetoed(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(dec={"ind_ctx": {"atr": 0.0015, "last_candle_range": 0.006}})
        out = net.run(tc, "BUY", risk, dec)  # 4x ATR > 2.5x
        v = next(c for c in out["checks"] if c["check"] == "atr_regime_check")
        assert v["verdict"] == VERDICT_VETO and "spike" in v["reason"]

    def test_atr_collapse_via_atr_pips(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(dec={"ind_ctx": {"atr": 0.0002, "atr_pips": 2.0}})
        out = net.run(tc, "BUY", risk, dec)
        v = next(c for c in out["checks"] if c["check"] == "atr_regime_check")
        assert v["verdict"] == VERDICT_VETO and "collapse" in v["reason"]

    def test_low_vol_regime_warns_without_atr_pips(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(dec={
            "ind_ctx": {"atr": 0.0012},
            "regime": {"volatility": "LOW_VOLATILITY"},
        })
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "atr_regime_check") == VERDICT_WARN

    def test_missing_atr_skips(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(dec={"ind_ctx": {}})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "atr_regime_check") == VERDICT_SKIP


class TestStructureSL:
    def test_sl_inside_structure_noise_vetoed_for_buy(self):
        # entry 1.1000, support 1.0980 (within 2x SL distance), SL 1.0975+?
        # SL 1.0990 sits ABOVE support-2pips(1.0978) -> inside noise
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"sl_price": 1.0990, "sl_pips": 10})
        tc["market_context"]["nearest_support"] = 1.0980
        out = net.run(tc, "BUY", risk, dec)
        v = next(c for c in out["checks"] if c["check"] == "structure_sl_check")
        assert v["verdict"] == VERDICT_VETO and "noise" in v["reason"]

    def test_sl_beyond_structure_passes(self):
        net = DASafetyNet()
        tc, risk, dec = _mk()  # SL 1.0970 < support 1.0980 - 2 pips
        tc["market_context"]["nearest_support"] = 1.0980
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "structure_sl_check") == VERDICT_PASS

    def test_far_structure_skipped(self):
        # support 80 pips below entry = 2.67x SL distance -> irrelevant horizon
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"sl_price": 1.0990, "sl_pips": 10})
        tc["market_context"]["nearest_support"] = 1.0920
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "structure_sl_check") == VERDICT_SKIP

    def test_sell_structure_mirror(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"entry": 1.1000, "sl_price": 1.1010, "sl_pips": 10})
        tc["market_context"]["nearest_resistance"] = 1.1020
        # resistance-adjacent SELL: SL 1.1010 is BELOW 1.1020+2pips=1.1022? yes 1.1010<1.1022
        # -> inside noise (stop under the level gets wicked before rejection)
        out = net.run(tc, "SELL", risk, dec)
        v = next(c for c in out["checks"] if c["check"] == "structure_sl_check")
        assert v["verdict"] == VERDICT_VETO

    def test_min_sl_floor_vetoed(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"sl_price": 1.0996, "sl_pips": 4})
        out = net.run(tc, "BUY", risk, dec)  # 4 pips < 6 floor, no structure needed
        v = next(c for c in out["checks"] if c["check"] == "structure_sl_check")
        assert v["verdict"] == VERDICT_VETO and "floor" in v["reason"]


class TestLotSizing:
    def test_consistent_lot_passes(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"lot": 0.033})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "lot_sizing_check") == VERDICT_PASS

    def test_lot_mismatch_vetoed(self):
        # intended: balance*1% / (30 * 10) = 0.0333 lot; booked 1.0 = ~2900% off
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"lot": 1.00})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "lot_sizing_check") == VERDICT_VETO

    def test_jpy_pair_size_consistency(self):
        # USDJPY: sl 25 pips, pip value 6.5 -> correct lot = 10/(25*6.5)=0.0615
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"lot": 0.06})
        tc["symbol"] = "USDJPY"
        risk.update({"sl_pips": 25})
        out = net.run(tc, "SELL", risk, dec)
        assert _verdict_of(out["checks"], "lot_sizing_check") == VERDICT_PASS

    def test_gold_wrong_pip_value_detected(self):
        # XAUUSD pip val $1/lot: correct lot = 10/(30*1)=0.33 — booking with
        # FX-style assumption (pip_val 10 -> lot 0.03) is 90% off -> veto
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"lot": 0.03})
        tc["symbol"] = "XAUUSD"
        out = net.run(tc, "BUY", risk, dec)
        v = next(c for c in out["checks"] if c["check"] == "lot_sizing_check")
        assert v["verdict"] == VERDICT_VETO and "mismatch" in v["reason"]

    def test_unmapped_symbol_warns_on_fallback_table(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"lot": 0.033})
        tc["symbol"] = "USDMXN"   # not in PIP_VALUE_USD -> DEFAULT 10 used
        out = net.run(tc, "BUY", risk, dec)
        # numbers still consistent w/ fallback (10) -> warn not veto
        v = next(c for c in out["checks"] if c["check"] == "lot_sizing_check")
        assert v["verdict"] == VERDICT_WARN

    def test_incomplete_fields_skip(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"balance": None})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "lot_sizing_check") == VERDICT_SKIP


class TestGateIntegration:
    def _gate(self, monkeypatch, captured=None):
        from core.devils_advocate import DevilsAdvocateGate
        gate = DevilsAdvocateGate(enabled=True, fail_mode="fail_open", mode="live")

        def _fake_provider(payload):
            if captured is not None:
                captured.append(payload)
            return {
                "decision": "TAKE",
                "confidence": 80,
                "thesis_quality": 8,
                "counter_evidence_strength": 2,
                "expected_edge": "positive",
                "pillars": {},
            }

        monkeypatch.setattr(gate, "_call_provider", _fake_provider)
        return gate

    def test_veto_short_circuits_before_llm_call(self, monkeypatch):
        captured = []
        gate = self._gate(monkeypatch, captured)
        tc, risk, dec = _mk(mc={"sessions_active": ["tokyo"]})  # asian chop EURUSD
        result = gate.review(trade_context=tc, signal="BUY", risk_out=risk, decision_out=dec)
        assert result["decision"] == "REJECT"
        assert "safety_net_veto" in str(result.get("critical_failure"))
        assert not captured, "LLM must NOT be called when safety net vetoes"

    def test_clean_trade_reaches_llm_with_warnings_attached(self, monkeypatch):
        captured = []
        gate = self._gate(monkeypatch, captured)
        tc, risk, dec = _mk()  # all pass
        result = gate.review(trade_context=tc, signal="BUY", risk_out=risk, decision_out=dec)
        assert result["decision"] == "TAKE"
        assert len(captured) == 1

    def test_warn_only_mode_downgrades_veto_to_warning(self, monkeypatch):
        captured = []
        gate = self._gate(monkeypatch, captured)
        gate._safety_net.mode = "warn_only"
        tc, risk, dec = _mk(mc={"h4_trend": "bearish"})  # would veto
        result = gate.review(trade_context=tc, signal="BUY", risk_out=risk, decision_out=dec)
        assert result["decision"] == "TAKE"
        sn = gate._safety_net.run(tc, "BUY", risk, dec)
        assert all(c["verdict"] != VERDICT_VETO for c in sn["vetoes"])
