# tests/conftest.py — shared pytest fixtures
# ============================================================
# Provides reusable fixtures so individual test files stay short.
# Heavily mocks external I/O (network, DB, filesystem) so tests
# run hermetically under CI without API keys or live data.
# ============================================================

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure test-mode env visible during collection and for unittest.TestCase
# tests which don't receive pytest fixtures; keeps behavior consistent.
os.environ.setdefault("FOREX_TEST_MODE", "1")

# Make sure repo root is importable so `from analysis.xxx import ...`
# works no matter where pytest is invoked from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Environment defaults — keep tests hermetic ────────────────
# Tests should NEVER hit the real network or read real .env files.
# We set safe stubs before any module under test is imported.
@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Force offline-mode env vars for every test."""
    monkeypatch.setenv("OANDA_API_KEY", "")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "")
    monkeypatch.setenv("OPENAI_API_KEY", "test-stub-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-stub-key")
    monkeypatch.setenv("TRADING_MODE", "paper")
    # Tests run against a temp DB so the real trader.db is never touched.
    monkeypatch.setenv("FOREX_TEST_MODE", "1")


# ── Temp DB fixture ───────────────────────────────────────────
@pytest.fixture
def temp_db_path(tmp_path):
    """A fresh, isolated SQLite path per test."""
    return str(tmp_path / "test_trader.db")


# ── Sample OHLCV DataFrame fixture ────────────────────────────
@pytest.fixture
def sample_ohlcv_df():
    """A small synthetic OHLCV DataFrame usable by indicator/backtest tests."""
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(seed=42)
    n = 200
    base = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    df = pd.DataFrame({
        "Open":   base + rng.normal(0, 0.0002, n),
        "High":   base + abs(rng.normal(0.0003, 0.0001, n)),
        "Low":    base - abs(rng.normal(0.0003, 0.0001, n)),
        "Close":  base,
        "Volume": rng.integers(100, 1000, n),
    }, index=pd.date_range("2025-01-01", periods=n, freq="15min"))
    df.index.name = "timestamp"
    return df


# ── Mock sentiment data fixture ───────────────────────────────
@pytest.fixture
def mock_sentiment_data():
    """A deterministic sentiment payload for SentimentEngine tests."""
    return {
        "pair":               "EURUSD",
        "retail_long_pct":    62.5,
        "retail_source":      "synthetic_rsi",
        "fg_index":           72.0,
        "fg_source":          "fx_native",
        "currency_strengths": {"USD": 68, "EUR": 52, "GBP": 60, "JPY": 45},
        "strength_source":    "engine",
        "dxy_trend":          "BULLISH",
        "dxy_change_pct":     0.32,
        "dxy_source":         "yfinance",
    }


# ── Mock broker fixture ───────────────────────────────────────
@pytest.fixture
def mock_broker():
    """A MagicMock broker that returns realistic-looking quotes."""
    broker = MagicMock()
    broker.get_quote.return_value = {"bid": 1.0998, "ask": 1.1000, "spread": 0.0002}
    broker.place_order.return_value = {"order_id": "TEST-001", "status": "filled"}
    broker.get_balance.return_value = 10000.0
    broker.get_positions.return_value = []
    return broker


# ── Stub time fixture for deterministic backtests ─────────────
@pytest.fixture
def fixed_now(monkeypatch):
    """Freeze 'now' to 2025-06-15 12:00 UTC for time-sensitive tests."""
    from datetime import datetime, timezone
    fixed = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    class _FixedDatetime:
        @classmethod
        def now(cls, tz=None):
            return fixed if tz is None else fixed.astimezone(tz)

        @classmethod
        def utcnow(cls):
            return fixed

    monkeypatch.setattr("datetime.datetime", _FixedDatetime)
    return fixed
