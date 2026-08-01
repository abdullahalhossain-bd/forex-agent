# tests/unit/test_backtest_metrics.py
# ============================================================
# Unit tests for the quick_metrics pure function.
# Verifies the metric math is correct without needing market data.
# ============================================================

import pytest


@pytest.mark.unit
def test_quick_metrics_zero_trades():
    from backtest.quick_backtest import quick_metrics

    m = quick_metrics(trades=[], equity_curve=[(0, 10_000.0)], initial_capital=10_000.0)
    assert m["total_trades"] == 0
    assert m["total_return_pct"] == 0.0
    assert m["win_rate"] == 0.0


@pytest.mark.unit
def test_quick_metrics_all_wins():
    from backtest.quick_backtest import quick_metrics

    trades = [{"pnl": 100}, {"pnl": 200}, {"pnl": 150}]
    eq = [(0, 10_000), (1, 10_100), (2, 10_300), (3, 10_450)]
    m = quick_metrics(trades=trades, equity_curve=eq, initial_capital=10_000.0)

    assert m["total_trades"] == 3
    assert m["win_rate"] == 100.0
    assert m["total_return_pct"] == 4.5
    assert m["profit_factor"] == float("inf")  # no losses
    assert m["max_drawdown_pct"] == 0.0


@pytest.mark.unit
def test_quick_metrics_mixed():
    from backtest.quick_backtest import quick_metrics

    trades = [{"pnl": 200}, {"pnl": -100}, {"pnl": 150}, {"pnl": -50}]
    eq = [(0, 10_000), (1, 10_200), (2, 10_100), (3, 10_250), (4, 10_200)]
    m = quick_metrics(trades=trades, equity_curve=eq, initial_capital=10_000.0)

    assert m["total_trades"] == 4
    assert m["win_rate"] == 50.0
    assert m["total_return_pct"] == 2.0
    assert m["profit_factor"] == round((200 + 150) / (100 + 50), 2)  # 2.33
    assert m["max_drawdown_pct"] > 0  # we did dip from 10_250 to 10_200


@pytest.mark.unit
def test_quick_metrics_drawdown_calc():
    """Drawdown is measured from peak, not from start."""
    from backtest.quick_backtest import quick_metrics

    # Equity goes 10000 → 12000 (peak) → 9000 (trough)
    trades = [{"pnl": 2000}, {"pnl": -3000}]
    eq = [(0, 10_000), (1, 12_000), (2, 9_000)]
    m = quick_metrics(trades=trades, equity_curve=eq, initial_capital=10_000.0)

    # Drawdown from peak 12000 to trough 9000 = 25%
    assert m["max_drawdown_pct"] == 25.0
