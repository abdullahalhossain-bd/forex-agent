"""Unit tests for the strict live-trading-mirror boundary."""
from __future__ import annotations

import pandas as pd
import pytest

from backtest.live_mirror import validate_historical_ohlcv


def _df(index):
    return pd.DataFrame(
        {
            "open": [1.1000] * len(index),
            "high": [1.1010] * len(index),
            "low": [1.0990] * len(index),
            "close": [1.1005] * len(index),
        },
        index=index,
    )


def test_validation_accepts_utc_monotonic_ohlc():
    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    result = validate_historical_ohlcv(_df(idx))
    assert result.rows == 3
    assert result.start.tz is not None
    assert result.end.tz is not None


def test_validation_rejects_naive_timestamps():
    idx = pd.date_range("2026-01-01", periods=3, freq="h")
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        validate_historical_ohlcv(_df(idx))


def test_validation_rejects_duplicate_timestamps():
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-01-01", tz="UTC"),
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_historical_ohlcv(_df(idx))


def test_validation_rejects_invalid_high():
    idx = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    df = _df(idx)
    df.loc[idx[0], "high"] = 1.0995
    with pytest.raises(ValueError, match="high below"):
        validate_historical_ohlcv(df)


def test_validation_rejects_non_monotonic_input():
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-01 01:00", tz="UTC"),
            pd.Timestamp("2026-01-01 00:00", tz="UTC"),
        ]
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_historical_ohlcv(_df(idx))
