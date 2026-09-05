"""Architecture contracts for the unified historical replay entry point."""
from pathlib import Path


def test_unified_engine_is_only_a_compatibility_facade():
    source = Path("backtest/unified_engine.py").read_text(encoding="utf-8")
    assert "BrokerSimulator" not in source
    assert "check_exit(" not in source
    assert "next-bar-open" not in source.lower()
    assert "run_live_mirroring_replay" in source
    assert "CanonicalHistorical" not in source or "live_mirroring_runner" in source


def test_strict_runner_owns_canonical_execution_lifecycle():
    source = Path("backtest/live_mirroring_runner.py").read_text(encoding="utf-8")
    assert "CanonicalHistoricalExecutionAdapter" in source
    assert "HistoricalPositionMonitor" in source
    assert "LiveMirroringExecutionBridge" in source
    assert "evaluate_decision_core" in source
    assert "df.iloc[: i + 1]" in source
    assert "iloc[i + 1]" not in source


def test_canonical_execution_has_no_random_fill_path():
    source = Path("backtest/canonical_execution.py").read_text(encoding="utf-8")
    assert "random" not in source.lower()
    assert "historical_ask" in source
    assert "AMBIGUOUS_INTRABAR" in source
