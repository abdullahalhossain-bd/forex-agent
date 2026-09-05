from datetime import datetime, timezone

from core.clock import ReplayClock
from risk.trade_frequency import TradeFrequencyController


def test_replay_clock_drives_trade_frequency_day_boundary():
    clock = ReplayClock(datetime(2026, 1, 1, 23, 59, tzinfo=timezone.utc))
    gate = TradeFrequencyController(clock=clock, min_daily=0, max_daily=50)
    gate.record_trade("EURUSD", "BUY")
    assert gate.trade_count_today() == 1

    clock.advance(datetime(2026, 1, 2, 0, 1, tzinfo=timezone.utc))
    assert gate.trade_count_today() == 0


def test_replay_clock_rejects_backward_time():
    clock = ReplayClock(datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc))
    try:
        clock.advance(datetime(2026, 1, 1, 23, 59, tzinfo=timezone.utc))
    except ValueError as exc:
        assert "backwards" in str(exc)
    else:
        raise AssertionError("ReplayClock accepted backward time")
