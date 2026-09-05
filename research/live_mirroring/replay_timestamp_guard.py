"""P0 timestamp-boundary helpers for historical replay.

The helpers are intentionally independent of the live trading path. They are
used by replay adapters/tests to prove that a decision context contains no
future rows and that higher-timeframe candles are closed before use.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


def ensure_not_future(source_timestamp: Any, replay_timestamp: Any, *, field: str = "source") -> None:
    """Raise if a decision input occurs after the replay decision timestamp."""
    if source_timestamp is None or replay_timestamp is None:
        raise ValueError(f"missing timestamp for {field}")
    src = _as_datetime(source_timestamp)
    cutoff = _as_datetime(replay_timestamp)
    if src > cutoff:
        raise ValueError(
            f"look-ahead detected: {field} timestamp {src.isoformat()} "
            f"> replay timestamp {cutoff.isoformat()}"
        )


def closed_candle_cutoff(candle_open: Any, timeframe_seconds: int) -> datetime:
    """Return the instant at which a candle becomes eligible for replay."""
    if timeframe_seconds <= 0:
        raise ValueError("timeframe_seconds must be positive")
    return _as_datetime(candle_open).timestamp() and datetime.fromtimestamp(
        _as_datetime(candle_open).timestamp() + timeframe_seconds,
        tz=_as_datetime(candle_open).tzinfo,
    )


def ensure_candle_closed(candle_open: Any, timeframe_seconds: int, replay_timestamp: Any) -> None:
    """Raise unless the complete candle is closed by replay time."""
    close_time = closed_candle_cutoff(candle_open, timeframe_seconds)
    cutoff = _as_datetime(replay_timestamp)
    if close_time > cutoff:
        raise ValueError(
            f"future/unclosed candle: close={close_time.isoformat()} "
            f"> replay={cutoff.isoformat()}"
        )


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        v = value.replace("Z", "+00:00")
        return datetime.fromisoformat(v)
    raise TypeError(f"unsupported timestamp type: {type(value)!r}")
