# tests/integration/test_config_loader.py
# ============================================================
# Integration test: Pydantic config loader reads from env + .env file.
# Uses tmp_path to avoid touching the real .env.
# ============================================================

import os
import pytest


@pytest.mark.integration
def test_load_config_from_env(monkeypatch):
    """Config values set via env var should be picked up by the loader."""
    monkeypatch.setenv("FOREX_TRADING_MODE", "paper")
    monkeypatch.setenv("FOREX_TRADING_RISK_PER_TRADE_PCT", "1.5")
    monkeypatch.setenv("FOREX_TRADING_MAX_LOT_SIZE", "0.30")

    from typed_config.loader import load_trading_config
    cfg = load_trading_config()

    assert cfg.mode == "paper"
    assert cfg.risk_per_trade_pct == 1.5
    assert cfg.max_lot_size == 0.30


@pytest.mark.integration
def test_load_config_falls_back_to_defaults(monkeypatch):
    """If env vars are missing, sensible defaults are used."""
    # Clear any env that might leak from the host
    for var in ("FOREX_TRADING_MODE", "FOREX_TRADING_RISK_PER_TRADE_PCT", "FOREX_TRADING_MAX_LOT_SIZE"):
        monkeypatch.delenv(var, raising=False)

    from typed_config.loader import load_trading_config
    cfg = load_trading_config()

    assert cfg.mode == "paper"
    assert 0 < cfg.risk_per_trade_pct <= 5.0
    assert cfg.max_lot_size > 0
