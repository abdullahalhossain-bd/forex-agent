from datetime import datetime, timezone

import pytest

from core.clock import ReplayClock
from core.data_provider import HistoricalMT5Provider
import pandas as pd


def test_replay_clock_normalizes_naive_timestamp_to_utc():
    clock = ReplayClock(datetime(2024, 1, 2, 10, 30))

    assert clock.now() == datetime(2024, 1, 2, 10, 30, tzinfo=timezone.utc)
    assert clock.current_date().isoformat() == "2024-01-02"
    assert clock.current_session() == "LONDON"


def test_replay_clock_advances_deterministically():
    clock = ReplayClock()

    clock.advance(datetime(2024, 1, 2, 7, tzinfo=timezone.utc))
    clock.advance(datetime(2024, 1, 2, 14, tzinfo=timezone.utc))

    assert clock.now().hour == 14
    assert clock.current_session() == "NEW_YORK"


def test_replay_clock_rejects_backward_movement():
    clock = ReplayClock(datetime(2024, 1, 2, 10, tzinfo=timezone.utc))

    with pytest.raises(ValueError, match="cannot move backwards"):
        clock.advance(datetime(2024, 1, 2, 9, tzinfo=timezone.utc))


def test_uninitialized_replay_clock_requires_explicit_timestamp():
    with pytest.raises(RuntimeError, match="not been initialized"):
        ReplayClock().now()


def test_historical_provider_advances_the_shared_replay_clock():
    index = pd.date_range("2024-01-02 08:00", periods=2, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.1, 1.2],
            "volume": [10, 11],
        },
        index=index,
    )
    clock = ReplayClock()
    provider = HistoricalMT5Provider(frame, "EURUSD", "H1", clock=clock)

    provider.advance_to(1)

    assert provider.current_time() == index[1]
    assert clock.now() == index[1]