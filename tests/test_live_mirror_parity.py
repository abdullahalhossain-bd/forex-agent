from datetime import datetime, timezone

import pytest

from backtest.historical_execution_router import HistoricalExecutionRouter
from backtest.position_state import CanonicalPositionState
from backtest.broker_sim import BrokerSimulator
from core.clock import ReplayClock


def _clock():
    return ReplayClock(datetime(2026, 1, 2, 10, 0, tzinfo=timezone.utc))


def test_market_order_never_fills_on_signal_bar():
    broker = BrokerSimulator(partial_fill_prob=0.0, slippage_pips=0.0, commission_per_lot=0.0)
    router = HistoricalExecutionRouter(broker, max_lot=1.0)
    result = router.submit(symbol="EURUSD", direction="BUY", entry_price=1.1000,
                           sl=1.0950, tp=1.1100, lot=0.1, confidence=90,
                           bar_index=10, bar_time=_clock().now())
    assert result["status"] == "QUEUED"
    assert router.advance(bar_index=10, bar_open=1.1010, bar_high=1.1020,
                          bar_low=1.0990, bar_close=1.1005, bar_time=_clock().now()) == []
    fills = router.advance(bar_index=11, bar_open=1.1030, bar_high=1.1040,
                           bar_low=1.1020, bar_close=1.1035, bar_time=_clock().now())
    assert len(fills) == 1
    assert fills[0].requested_entry == 1.1030


def test_limit_order_waits_then_fills_when_touched():
    broker = BrokerSimulator(partial_fill_prob=0.0, slippage_pips=0.0, commission_per_lot=0.0)
    router = HistoricalExecutionRouter(broker, max_lot=1.0, default_limit_expiry_bars=3)
    result = router.submit(symbol="EURUSD", direction="BUY", entry_price=1.1000,
                           sl=1.0950, tp=1.1100, lot=0.1, confidence=90,
                           bar_index=10, bar_time=_clock().now(), order_type="BUY_LIMIT")
    assert result["status"] == "PENDING"
    assert router.advance(bar_index=11, bar_open=1.1050, bar_high=1.1060,
                          bar_low=1.1010, bar_close=1.1040, bar_time=_clock().now()) == []
    fills = router.advance(bar_index=12, bar_open=1.1030, bar_high=1.1040,
                           bar_low=1.0990, bar_close=1.1010, bar_time=_clock().now())
    assert len(fills) == 1
    assert fills[0].requested_entry == 1.1000
    assert not router.pending_orders


def test_limit_order_expires_without_fill():
    broker = BrokerSimulator(partial_fill_prob=0.0, slippage_pips=0.0, commission_per_lot=0.0)
    router = HistoricalExecutionRouter(broker, default_limit_expiry_bars=1)
    router.submit(symbol="EURUSD", direction="BUY", entry_price=1.1000,
                  sl=1.0950, tp=1.1100, lot=0.1, confidence=90,
                  bar_index=10, bar_time=_clock().now(), order_type="LIMIT")
    router.advance(bar_index=11, bar_open=1.1050, bar_high=1.1060,
                   bar_low=1.1020, bar_close=1.1040, bar_time=_clock().now())
    router.advance(bar_index=12, bar_open=1.1050, bar_high=1.1060,
                   bar_low=1.1020, bar_close=1.1040, bar_time=_clock().now())
    assert not router.pending_orders
    assert any(e["event"] == "LIMIT_EXPIRED" for e in router.telemetry)


def test_replay_clock_is_deterministic_and_monotonic():
    clock = _clock()
    assert clock.current_session() == "LONDON"
    clock.advance(datetime(2026, 1, 3, 2, 0, tzinfo=timezone.utc))
    assert clock.current_date().isoformat() == "2026-01-03"
    assert clock.current_session() == "ASIA"
    with pytest.raises(ValueError):
        clock.advance(datetime(2026, 1, 2, 23, 0, tzinfo=timezone.utc))


def test_canonical_position_state_tracks_open_and_close():
    class T:
        trade_id = 7
        symbol = "EURUSD"
        direction = "BUY"
        lot_size = 0.2
        entry_price = 1.1
        entry_time = "2026-01-02T10:00:00+00:00"

    state = CanonicalPositionState()
    state.register(T())
    assert state.count() == 1
    assert state.count_symbol("eurusd") == 1
    assert state.has_symbol_direction("EURUSD", "BUY")
    state.close(7, reason="SL")
    assert state.count() == 0
