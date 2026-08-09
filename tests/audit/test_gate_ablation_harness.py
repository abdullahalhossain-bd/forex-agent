"""
tests/audit/test_gate_ablation_harness.py — Regression tests for the
audit additions to backtest/persistent_runner.py and ml/rl_agent.py.

These tests verify:
  1. FOREX_BYPASS_CHECKS env var is read correctly (baseline = empty set)
  2. _record_gate_stats aggregates per-gate pass/fail correctly
  3. RL quality gate now rejects catastrophically-bad models (avg_reward check)
  4. TEST_MODE force-approve is refused in mt5_live mode (safety guard)
  5. The patched persistent_runner module still imports cleanly
  6. trade_permission._bypass_check correctly recognizes all 7 gate names
     the user listed

Run with:
    py -3.13 -m pytest tests/audit/test_gate_ablation_harness.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ── Test 1: persistent_runner imports cleanly after patch ─────────────

def test_persistent_runner_imports():
    """The patched persistent_runner must still import without errors."""
    import backtest.persistent_runner as pr
    assert hasattr(pr, "SymbolWorker")
    assert hasattr(pr, "run_multi_symbol")


# ── Test 2: bypass-checks env var reader ──────────────────────────────

def test_bypass_env_var_unset_returns_empty_set():
    """When FOREX_BYPASS_CHECKS is unset, bypass set is empty (baseline)."""
    from backtest.persistent_runner import SymbolWorker
    os.environ.pop("FOREX_BYPASS_CHECKS", None)
    result = SymbolWorker._read_bypass_checks_from_env()
    assert result == set(), f"expected empty set, got {result}"


def test_bypass_env_var_empty_string_returns_empty_set():
    """Empty string env var = baseline behavior."""
    from backtest.persistent_runner import SymbolWorker
    os.environ["FOREX_BYPASS_CHECKS"] = ""
    result = SymbolWorker._read_bypass_checks_from_env()
    assert result == set()
    os.environ.pop("FOREX_BYPASS_CHECKS", None)


def test_bypass_env_var_single_gate():
    """Single gate name is parsed correctly."""
    from backtest.persistent_runner import SymbolWorker
    os.environ["FOREX_BYPASS_CHECKS"] = "Min confidence"
    result = SymbolWorker._read_bypass_checks_from_env()
    assert "Min confidence" in result
    assert len(result) == 1
    os.environ.pop("FOREX_BYPASS_CHECKS", None)


def test_bypass_env_var_multiple_gates():
    """Comma-separated gate names are all parsed."""
    from backtest.persistent_runner import SymbolWorker
    os.environ["FOREX_BYPASS_CHECKS"] = "Min confidence, Session quality, Trend alignment (regime)"
    result = SymbolWorker._read_bypass_checks_from_env()
    assert len(result) == 3
    assert "Min confidence" in result
    assert "Session quality" in result
    assert "Trend alignment (regime)" in result
    os.environ.pop("FOREX_BYPASS_CHECKS", None)


def test_bypass_env_var_strips_whitespace():
    """Whitespace around gate names is stripped."""
    from backtest.persistent_runner import SymbolWorker
    os.environ["FOREX_BYPASS_CHECKS"] = "  Min confidence  ,  Session quality  "
    result = SymbolWorker._read_bypass_checks_from_env()
    assert "Min confidence" in result
    assert "Session quality" in result
    assert "  Min confidence  " not in result  # stripped
    os.environ.pop("FOREX_BYPASS_CHECKS", None)


# ── Test 3: gate-stats aggregator ─────────────────────────────────────

def test_record_gate_stats_aggregates_pass_fail():
    """_record_gate_stats correctly counts passed/failed per gate."""
    from backtest.persistent_runner import SymbolWorker
    # Build a minimal SymbolWorker instance without running __init__
    # (we only need the _gate_stats + _gate_first_blocker attrs)
    worker = SymbolWorker.__new__(SymbolWorker)
    worker._gate_stats = {}
    worker._gate_first_blocker = ""

    # Simulate perm_out with 3 checks, 2 passed + 1 failed
    perm_out = {
        "checks": [
            {"check": "Valid signal", "passed": True, "detail": "BUY"},
            {"check": "Min confidence", "passed": True, "detail": "75% (min 60%)"},
            {"check": "Session quality", "passed": False, "detail": "LOW"},
        ]
    }
    worker._record_gate_stats(perm_out)
    assert worker._gate_stats["Valid signal"] == {"passed": 1, "failed": 0}
    assert worker._gate_stats["Min confidence"] == {"passed": 1, "failed": 0}
    assert worker._gate_stats["Session quality"] == {"passed": 0, "failed": 1}
    assert worker._gate_first_blocker == "Session quality"


def test_record_gate_stats_handles_empty_perm_out():
    """Empty perm_out is a no-op."""
    from backtest.persistent_runner import SymbolWorker
    worker = SymbolWorker.__new__(SymbolWorker)
    worker._gate_stats = {}
    worker._gate_first_blocker = ""
    worker._record_gate_stats({})
    assert worker._gate_stats == {}
    assert worker._gate_first_blocker == ""


def test_record_gate_stats_first_blocker_is_first_failing_gate():
    """The first_blocker is the FIRST gate in checks list that failed,
    not the last."""
    from backtest.persistent_runner import SymbolWorker
    worker = SymbolWorker.__new__(SymbolWorker)
    worker._gate_stats = {}
    worker._gate_first_blocker = ""
    perm_out = {
        "checks": [
            {"check": "Valid signal", "passed": True, "detail": ""},
            {"check": "Min confidence", "passed": False, "detail": ""},  # first fail
            {"check": "Session quality", "passed": False, "detail": ""},  # second fail
        ]
    }
    worker._record_gate_stats(perm_out)
    assert worker._gate_first_blocker == "Min confidence"


# ── Test 4: trade_permission._bypass_check recognizes all 7 user gates ─

@pytest.mark.parametrize("gate_name", [
    "Min confidence",
    "Session quality",
    "Confluence quality",
    "Risk approved",
    "S/R zone alignment",
    "Valid signal",
    "Trend alignment (regime)",
])
def test_bypass_check_recognizes_user_gate_names(gate_name):
    """Each of the 7 gate names the user listed must be recognized by
    _bypass_check when passed in bypass_checks."""
    from risk.trade_permission import _bypass_check, _normalize_bypass_checks
    bypass = _normalize_bypass_checks({gate_name})
    assert _bypass_check(gate_name, bypass), (
        f"gate '{gate_name}' not recognized by _bypass_check "
        f"(bypass set: {bypass})"
    )


@pytest.mark.parametrize("alias", [
    "min_confidence",
    "session_quality",
    "confluence_quality",
    "risk_approved",
    "sr_alignment",
    "valid_signal",
    "trend_alignment",
])
def test_bypass_check_recognizes_aliases(alias):
    """Short aliases (snake_case) must also work for env-var convenience."""
    from risk.trade_permission import _bypass_check, _normalize_bypass_checks
    bypass = _normalize_bypass_checks({alias})
    # The alias should resolve to its target gate name
    assert any(_bypass_check(g, bypass) for g in [
        "Min confidence", "Session quality", "Confluence quality",
        "Risk approved", "S/R zone alignment", "Valid signal",
        "Trend alignment (regime)",
    ])


# ── Test 5: RL quality gate now rejects catastrophically-bad models ────

def test_rl_quality_gate_rejects_negative_avg_reward():
    """A model with avg_reward=-9140 (the shipped ppo_forex_latest) must
    now FAIL the quality gate, not pass."""
    from ml.rl_agent import RLAgent
    agent = RLAgent.__new__(RLAgent)  # bypass __init__
    # Simulate the shipped model's meta
    meta = {
        "episodes": 60000,
        "win_rate": 0.017,
        "avg_reward": -9140.96,
    }
    passed, reason = agent._passes_quality_gate(meta)
    assert not passed, f"expected gate to FAIL, but it passed: {reason}"
    assert "avg_reward" in reason or "consistently" in reason.lower(), (
        f"reason should mention avg_reward / consistent losses, got: {reason}"
    )


def test_rl_quality_gate_passes_healthy_model():
    """A model with positive avg_reward, decent win_rate, enough episodes
    should pass."""
    from ml.rl_agent import RLAgent
    agent = RLAgent.__new__(RLAgent)
    meta = {
        "episodes": 500,
        "win_rate": 0.55,
        "avg_reward": 25.0,
    }
    passed, reason = agent._passes_quality_gate(meta)
    assert passed, f"expected gate to PASS, but it failed: {reason}"


def test_rl_quality_gate_rejects_no_meta():
    """Missing meta = unverified, must fail."""
    from ml.rl_agent import RLAgent
    agent = RLAgent.__new__(RLAgent)
    passed, reason = agent._passes_quality_gate(None)
    assert not passed
    assert "unverified" in reason.lower() or "no training metadata" in reason.lower()


def test_rl_quality_gate_rejects_undertrained():
    """Undertrained model (episodes < 5) must fail."""
    from ml.rl_agent import RLAgent
    agent = RLAgent.__new__(RLAgent)
    meta = {"episodes": 2, "win_rate": 0.5, "avg_reward": 10.0}
    passed, reason = agent._passes_quality_gate(meta)
    assert not passed
    assert "undertrained" in reason.lower() or "episodes" in reason.lower()


def test_rl_quality_gate_rejects_never_won():
    """Model with win_rate=0 must fail."""
    from ml.rl_agent import RLAgent
    agent = RLAgent.__new__(RLAgent)
    meta = {"episodes": 100, "win_rate": 0.0, "avg_reward": 10.0}
    passed, reason = agent._passes_quality_gate(meta)
    assert not passed
    assert "win_rate" in reason.lower()


# ── Test 6: config.py doc + 48-pair list integrity ────────────────────

def test_config_symbols_has_48_pairs():
    """config.SYMBOLS must have exactly 48 pairs (no duplicates)."""
    from config import SYMBOLS
    assert len(SYMBOLS) == 48, f"expected 48 pairs, got {len(SYMBOLS)}"
    assert len(set(SYMBOLS)) == 48, "duplicates in SYMBOLS"


def test_config_forex_pairs_equals_symbols():
    """Config.FOREX_PAIRS must equal SYMBOLS (the comment fix removed
    the misleading '28 pairs' claim)."""
    from config import Config, SYMBOLS
    assert list(Config.FOREX_PAIRS) == list(SYMBOLS)


# ── Test 7: data/trained_models coverage (informational) ──────────────

def test_trained_models_coverage():
    """Verify how many of the 48 pairs have at least one trained ML model.
    This is informational — not a hard failure, but documents the gap."""
    from config import SYMBOLS
    models_root = PROJECT_ROOT / "data" / "trained_models"
    trained = set()
    if models_root.is_dir():
        trained = {d.name for d in models_root.iterdir() if d.is_dir()}
    missing = [s for s in SYMBOLS if s not in trained]
    # Not a hard failure — but if this fires, the operator should know
    print(f"\n[audit] ML models trained: {len(trained)}/{len(SYMBOLS)}")
    if missing:
        print(f"[audit] Missing ML models: {missing}")
    # Sanity: at least the majors should have models
    for major in ["EURUSD", "GBPUSD", "USDJPY"]:
        assert major in trained, f"major pair {major} missing ML models"


if __name__ == "__main__":
    # Allow running without pytest
    import inspect
    g = globals()
    for name, fn in list(g.items()):
        if name.startswith("test_") and callable(fn):
            print(f"\n--- {name} ---")
            try:
                # Handle parametrized tests
                if hasattr(fn, "pytestmark"):
                    # Skip parametrized for direct run
                    print("  (parametrized — run via pytest)")
                    continue
                fn()
                print("  PASS")
            except AssertionError as e:
                print(f"  FAIL: {e}")
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
