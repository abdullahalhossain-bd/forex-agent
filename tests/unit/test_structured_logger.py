# tests/unit/test_structured_logger.py
# ============================================================
# Unit tests for the structured JSON logger.
# Captures stdout to verify JSON output is well-formed.
# ============================================================

import io
import json
import logging
import pytest


@pytest.fixture
def json_logger(monkeypatch):
    """Return a logger configured for JSON output, capturing stdout."""
    monkeypatch.setenv("FOREX_LOG_FORMAT", "json")

    # Reset logging state so the new format takes effect
    import importlib
    import utils.structured_logger as mod
    importlib.reload(mod)

    logger = mod.get_logger("test_json_logger")
    # Replace its handler's stream with our capture
    capture = io.StringIO()
    for h in logger.handlers:
        h.stream = capture
    return logger, capture, mod


@pytest.mark.unit
def test_json_logger_emits_valid_json(json_logger):
    logger, capture, _ = json_logger
    logger.info("hello world")
    line = capture.getvalue().strip()
    payload = json.loads(line)  # must not raise
    assert payload["msg"] == "hello world"
    assert payload["level"] == "info"
    assert "ts" in payload
    assert payload["logger"] == "test_json_logger"


@pytest.mark.unit
def test_json_logger_includes_extras(json_logger):
    logger, capture, _ = json_logger
    logger.info("trade_opened", extra={"pair": "EURUSD", "lot": 0.05})
    payload = json.loads(capture.getvalue().strip())
    assert payload["pair"] == "EURUSD"
    assert payload["lot"] == 0.05


@pytest.mark.unit
def test_log_event_helper(json_logger):
    logger, capture, mod = json_logger
    mod.log_event(logger, "info", "trade_closed", pair="EURUSD", pnl=150.0)
    payload = json.loads(capture.getvalue().strip())
    assert payload["msg"] == "trade_closed"
    assert payload["pair"] == "EURUSD"
    assert payload["pnl"] == 150.0


@pytest.mark.unit
def test_text_mode_fallback(monkeypatch):
    """When FOREX_LOG_FORMAT=text (or unset), output is human-readable, not JSON."""
    monkeypatch.setenv("FOREX_LOG_FORMAT", "text")
    import importlib
    import utils.structured_logger as mod
    importlib.reload(mod)

    logger = mod.get_logger("test_text_logger")
    capture = io.StringIO()
    for h in logger.handlers:
        h.stream = capture

    logger.info("hello world")
    out = capture.getvalue()
    # Should NOT be valid JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.strip())
    # Should contain the message text
    assert "hello world" in out
