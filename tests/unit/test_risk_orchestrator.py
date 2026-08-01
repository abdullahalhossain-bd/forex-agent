# tests/unit/test_risk_orchestrator.py
# ============================================================
# Unit tests for the AdvancedRiskOrchestrator.
# Verifies Kelly sizing, daily loss limit, and correlation
# guard compose correctly without touching real capital.
# ============================================================

import pytest


@pytest.mark.smoke
def test_risk_orchestrator_importable():
    from risk.advanced_risk_orchestrator import AdvancedRiskOrchestrator  # noqa: F401


@pytest.mark.unit
def test_kelly_sizing_capped_at_max_lot():
    """Kelly fraction is capped — even with a great edge, never exceed max_lot."""
    from risk.advanced_risk_orchestrator import AdvancedRiskOrchestrator

    orch = AdvancedRiskOrchestrator(
        account_balance=10_000.0,
        max_lot_size=0.50,
        risk_per_trade_pct=2.0,
        use_kelly_sizing=True,
    )

    size = orch.position_size(
        win_rate=0.65,        # 65% win rate
        avg_win=2.0,          # avg win = 2R
        avg_loss=1.0,         # avg loss = 1R
        stop_distance_pips=20,
        pip_value_per_lot=10.0,
    )

    assert 0 < size <= 0.50, f"Position size {size} must be in (0, max_lot]"


@pytest.mark.unit
def test_kelly_sizing_zero_edge_returns_zero():
    """If win_rate is 50% and avg_win == avg_loss, edge is 0 → no trade."""
    from risk.advanced_risk_orchestrator import AdvancedRiskOrchestrator

    orch = AdvancedRiskOrchestrator(
        account_balance=10_000.0,
        max_lot_size=0.50,
        risk_per_trade_pct=2.0,
        use_kelly_sizing=True,
    )

    size = orch.position_size(
        win_rate=0.50,
        avg_win=1.0,
        avg_loss=1.0,
        stop_distance_pips=20,
        pip_value_per_lot=10.0,
    )
    assert size == 0.0, "Zero edge should produce zero size"


@pytest.mark.unit
def test_daily_loss_limit_blocks_new_trades():
    """Once daily loss limit is hit, can_trade() must return False."""
    from risk.advanced_risk_orchestrator import AdvancedRiskOrchestrator

    orch = AdvancedRiskOrchestrator(
        account_balance=10_000.0,
        max_lot_size=0.50,
        risk_per_trade_pct=2.0,
        daily_loss_limit_pct=3.0,  # $300 daily loss limit
    )

    # Record a $350 loss — over the $300 limit
    orch.record_trade_result(pnl_usd=-350.0)
    assert orch.can_trade() is False, "Daily loss limit breached — must stop"


@pytest.mark.unit
def test_correlation_guard_blocks_redundant_pairs():
    """If EURUSD is already open, GBPUSD (highly correlated) must be blocked."""
    from risk.advanced_risk_orchestrator import AdvancedRiskOrchestrator

    orch = AdvancedRiskOrchestrator(
        account_balance=10_000.0,
        max_lot_size=0.50,
        risk_per_trade_pct=2.0,
    )
    orch.set_open_positions([{"pair": "EURUSD", "side": "BUY"}])

    allowed = orch.is_correlation_safe("GBPUSD", "BUY")
    assert allowed is False, "GBPUSD long is highly correlated with EURUSD long"


@pytest.mark.unit
def test_correlation_guard_allows_uncorrelated_pair():
    """EURUSD open does not block USDJPY (low correlation)."""
    from risk.advanced_risk_orchestrator import AdvancedRiskOrchestrator

    orch = AdvancedRiskOrchestrator(
        account_balance=10_000.0,
        max_lot_size=0.50,
        risk_per_trade_pct=2.0,
    )
    orch.set_open_positions([{"pair": "EURUSD", "side": "BUY"}])

    allowed = orch.is_correlation_safe("USDJPY", "BUY")
    assert allowed is True
