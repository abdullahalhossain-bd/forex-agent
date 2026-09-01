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

    def test_asian_chop_warns_non_asian_pair(self):
        # 2026-09 audit: downgraded VETO -> WARN (unvalidated C-type
        # empirical rule; see DA_SAFETY_NET_AUDIT_PHASE1-20.md)
        net = DASafetyNet()
        tc, risk, dec = _mk(mc={"sessions_active": ["tokyo", "sydney"]})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "session_filter") == VERDICT_WARN

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
        # sydney-only + EURUSD => asian low-liquidity WARN via live clock path
        # (2026-09 audit: downgraded from VETO)
        assert _verdict_of(out["checks"], "session_filter") == VERDICT_WARN

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
    def test_h4_opposing_buy_warns_by_default(self):
        # 2026-09 audit (Phase 10): counter-trend is WARN unless an
        # explicit policy (strategy_charter.counter_trend_prohibited or
        # DA_H4_COUNTERTREND_HARD_VETO=true) opts into a hard VETO.
        net = DASafetyNet()
        tc, risk, dec = _mk(mc={"h4_trend": "downtrend"})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "h4_trend_filter") == VERDICT_WARN

    def test_h4_opposing_buy_vetoed_with_explicit_charter_policy(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(mc={"h4_trend": "downtrend"})
        tc["strategy_charter"] = {"counter_trend_prohibited": True}
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "h4_trend_filter") == VERDICT_VETO

    def test_h4_opposing_buy_vetoed_with_env_policy(self, monkeypatch):
        net = DASafetyNet()
        monkeypatch.setenv("DA_H4_COUNTERTREND_HARD_VETO", "true")
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
        # WARN by default (2026-09 audit); no explicit counter-trend policy set
        assert _verdict_of(out["checks"], "h4_trend_filter") == VERDICT_WARN


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
        # XAUUSD authoritative absolute limit is 400 pips (core.spread_policy)
        # so this is purely a spread/ATR ratio VETO, not an absolute-pip one.
        # gold pip size 0.01, ATR 8 pips (atr=0.08), spread 4.5 -> ratio 0.5625
        # (> DA_MAX_SPREAD_RATIO_VETO default 0.50) while nowhere near the
        # absolute 400-pip limit.
        net = DASafetyNet()
        tc, risk, dec = _mk()
        tc["symbol"] = "XAUUSD"
        dec["ind_ctx"] = {"atr": 0.08, "spread_pips": 4.5}
        out = net.run(tc, "SELL", risk, dec)
        c = next(c for c in out["checks"] if c["check"] == "spread_check")
        assert c["verdict"] == VERDICT_VETO and "ATR" in c["reason"]

    def test_spread_near_limit_warns(self):
        # EURUSD authoritative max (core.spread_policy) = 3.0 pips;
        # 2.5 is between the 0.7x-of-limit WARN band (2.1) and the limit,
        # and its ATR ratio (0.010 price -> 100 pips ATR) stays low so the
        # ratio branch doesn't fire first.
        net = DASafetyNet()
        tc, risk, dec = _mk(dec={"ind_ctx": {"atr": 0.010, "spread_pips": 2.5}})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "spread_check") == VERDICT_WARN

    def test_news_window_halves_limit(self):
        # EURUSD authoritative max = 3.0 -> halved to 1.5 under a news
        # window; 1.8 pips breaches that.
        net = DASafetyNet()
        tc, risk, dec = _mk(dec={"ind_ctx": {"atr": 0.010, "spread_pips": 1.8}})
        tc["analysis_out"] = {"news_ctx": {"risk_level": "high"}}
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "spread_check") == VERDICT_VETO


class TestAtrRegime:
    def test_news_spike_candle_skipped_without_explicit_policy(self):
        # 2026-09 audit: DA_NEWS_SPIKE_ATR_MULT no longer has a baked-in
        # default. Without an explicit operator value, the spike sub-check
        # does not fire (overall verdict falls through to PASS).
        net = DASafetyNet()
        tc, risk, dec = _mk(dec={"ind_ctx": {"atr": 0.0015, "last_candle_range": 0.006}})
        out = net.run(tc, "BUY", risk, dec)
        v = next(c for c in out["checks"] if c["check"] == "atr_regime_check")
        assert v["verdict"] == VERDICT_PASS

    def test_news_spike_candle_vetoed_with_explicit_policy(self, monkeypatch):
        net = DASafetyNet()
        monkeypatch.setenv("DA_NEWS_SPIKE_ATR_MULT", "2.5")
        tc, risk, dec = _mk(dec={"ind_ctx": {"atr": 0.0015, "last_candle_range": 0.006}})
        out = net.run(tc, "BUY", risk, dec)  # 4x ATR > 2.5x
        v = next(c for c in out["checks"] if c["check"] == "atr_regime_check")
        assert v["verdict"] == VERDICT_VETO and "spike" in v["reason"]

    def test_atr_collapse_no_longer_flat_floor_vetoed(self):
        # 2026-09 audit (Phase 8): the flat, symbol-blind DA_MIN_ATR_PIPS
        # floor was removed entirely (no cross-symbol ATR-pip comparison
        # without a validated percentile/regime source). A low atr_pips
        # value alone no longer VETOes.
        net = DASafetyNet()
        tc, risk, dec = _mk(dec={"ind_ctx": {"atr": 0.0002, "atr_pips": 2.0}})
        out = net.run(tc, "BUY", risk, dec)
        v = next(c for c in out["checks"] if c["check"] == "atr_regime_check")
        assert v["verdict"] == VERDICT_PASS

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


def _with_swing_result(tc, *, passed, severity="WARNING", reason="", details=None):
    """Attach a fake upstream entry_quality_guardrails.sl_swing_anchor
    result to trade_context, the way risk/trade_permission.py does in the
    real pipeline (trade_context["perm_out"]["entry_quality_detail"]).
    2026-09 audit: structure_sl_check now VALIDATES this instead of
    recomputing its own independent swing-anchor judgment (finding F5)."""
    tc["perm_out"] = {
        "entry_quality_detail": {
            "results": [
                {
                    "flag_name": "sl_swing_anchor",
                    "passed": passed,
                    "severity": severity,
                    "reason": reason,
                    "details": details or {},
                }
            ]
        }
    }
    return tc


class TestStructureSL:
    def test_no_upstream_result_skips(self):
        # 2026-09 audit (finding F5): no independent re-derivation anymore.
        # Without the upstream sl_swing_anchor result on
        # trade_context.perm_out, this SKIPs rather than invent its own
        # swing-detection/buffer judgment.
        net = DASafetyNet()
        tc, risk, dec = _mk()
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "structure_sl_check") == VERDICT_SKIP

    def test_upstream_block_severity_mirrored_as_veto(self):
        net = DASafetyNet()
        tc, risk, dec = _mk()
        _with_swing_result(tc, passed=False, severity="BLOCK",
                            reason="SL is on the wrong side of entry")
        out = net.run(tc, "BUY", risk, dec)
        v = next(c for c in out["checks"] if c["check"] == "structure_sl_check")
        assert v["verdict"] == VERDICT_VETO

    def test_upstream_warning_severity_mirrored_as_warn(self):
        net = DASafetyNet()
        tc, risk, dec = _mk()
        _with_swing_result(tc, passed=False, severity="WARNING",
                            reason="SL is 12.0 pips (2.1x ATR) from nearest swing")
        out = net.run(tc, "BUY", risk, dec)
        v = next(c for c in out["checks"] if c["check"] == "structure_sl_check")
        assert v["verdict"] == VERDICT_WARN

    def test_upstream_pass_mirrored_as_pass(self):
        net = DASafetyNet()
        tc, risk, dec = _mk()
        _with_swing_result(tc, passed=True,
                            reason="SL anchored to swing 1.0980 (10.0 pips / 0.8x ATR)")
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "structure_sl_check") == VERDICT_PASS

    def test_sell_direction_also_validates_upstream(self):
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"entry": 1.1000, "sl_price": 1.1010, "sl_pips": 10})
        _with_swing_result(tc, passed=False, severity="BLOCK",
                            reason="SL wrong side for SELL")
        out = net.run(tc, "SELL", risk, dec)
        v = next(c for c in out["checks"] if c["check"] == "structure_sl_check")
        assert v["verdict"] == VERDICT_VETO


class TestLotSizing:
    @staticmethod
    def _mock_live(monkeypatch, pip_val):
        """Mock an available MT5 connection + live pip value, the way
        DASafetyNet._check_lot_sizing now requires before it will VETO on
        a mismatch (2026-09 audit finding F4)."""
        fake_conn = object()
        monkeypatch.setattr(
            "core.service_registry.get_registry",
            lambda: type("R", (), {"try_resolve": staticmethod(lambda name: fake_conn)})(),
        )
        monkeypatch.setattr(
            "core.constants.get_live_pip_value_per_lot",
            lambda symbol, mt5_conn=None: pip_val,
        )

    def test_consistent_lot_warns_without_live_verification(self):
        # 2026-09 audit: no MT5 connection in this test env -> pip value
        # is a static-table approximation -> always WARN (informational,
        # "sizing not independently verified"), never a silent PASS on
        # unverified data.
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"lot": 0.033})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "lot_sizing_check") == VERDICT_WARN

    def test_consistent_lot_passes_when_live_verified(self, monkeypatch):
        self._mock_live(monkeypatch, pip_val=10.0)
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"lot": 0.033})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "lot_sizing_check") == VERDICT_PASS

    def test_lot_mismatch_warns_without_live_verification(self):
        # intended: balance*1% / (30 * 10) = 0.0333 lot; booked 1.0 = ~2900% off,
        # but with no live-verified pip value + no explicit
        # DA_LOT_MISMATCH_TOL policy set, this is WARN via the risk-overflow
        # branch, not a silent PASS and not a VETO on approximated data.
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"lot": 1.00})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "lot_sizing_check") == VERDICT_WARN

    def test_lot_mismatch_vetoed_when_live_verified(self, monkeypatch):
        self._mock_live(monkeypatch, pip_val=10.0)
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"lot": 1.00})
        out = net.run(tc, "BUY", risk, dec)
        assert _verdict_of(out["checks"], "lot_sizing_check") == VERDICT_VETO

    def test_jpy_pair_size_consistency_when_live_verified(self, monkeypatch):
        # USDJPY: sl 25 pips, pip value 6.5 -> correct lot = 10/(25*6.5)=0.0615
        self._mock_live(monkeypatch, pip_val=6.5)
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"lot": 0.06})
        tc["symbol"] = "USDJPY"
        risk.update({"sl_pips": 25})
        out = net.run(tc, "SELL", risk, dec)
        assert _verdict_of(out["checks"], "lot_sizing_check") == VERDICT_PASS

    def test_gold_wrong_pip_value_detected_but_capped_at_warn_without_live_data(self):
        # 2026-09 audit fix (finding F4): with no live MT5 connection
        # available in this test environment, get_live_pip_value_per_lot()
        # falls back to the static table. Per the "never VETO on
        # non-authoritative/approximated data" policy, a mismatch computed
        # from the non-live fallback is now capped at WARN, not VETO — a
        # VETO here requires the pip value to actually come from live
        # broker data (see next test for the live-verified case).
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"lot": 0.03}, dec={"ind_ctx": {"atr": 0.6, "spread_pips": 4.0}})
        tc["symbol"] = "XAUUSD"
        out = net.run(tc, "BUY", risk, dec)
        v = next(c for c in out["checks"] if c["check"] == "lot_sizing_check")
        assert v["verdict"] == VERDICT_WARN

    def test_gold_wrong_pip_value_vetoed_when_live_verified(self, monkeypatch):
        self._mock_live(monkeypatch, pip_val=1.0)  # true XAUUSD pip value
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"lot": 0.03}, dec={"ind_ctx": {"atr": 0.6, "spread_pips": 4.0}})
        tc["symbol"] = "XAUUSD"
        tc["market_context"] = tc.get("market_context", {})
        import os
        os.environ["DA_LOT_MISMATCH_TOL"] = "0.25"
        try:
            out = net.run(tc, "BUY", risk, dec)
        finally:
            os.environ.pop("DA_LOT_MISMATCH_TOL", None)
        v = next(c for c in out["checks"] if c["check"] == "lot_sizing_check")
        assert v["verdict"] == VERDICT_VETO and "mismatch" in v["reason"]

    def test_unmapped_symbol_skips_rather_than_guess(self):
        # 2026-09 audit fix: no live MT5 connection AND the symbol isn't in
        # the static PIP_VALUE_USD table either -> SKIP, not a silent
        # generic-default guess (get_pip_value_usd's own internal 10.0
        # fallback is exactly the kind of default this check must not
        # rely on for a symbol nobody has actually mapped).
        net = DASafetyNet()
        tc, risk, dec = _mk(risk={"lot": 0.033})
        tc["symbol"] = "USDMXN"
        out = net.run(tc, "BUY", risk, dec)
        v = next(c for c in out["checks"] if c["check"] == "lot_sizing_check")
        assert v["verdict"] == VERDICT_SKIP

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
        tc, risk, dec = _mk(mc={"sessions_active": []})  # market closed -> hard VETO
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
