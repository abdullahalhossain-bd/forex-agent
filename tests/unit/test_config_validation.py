# tests/unit/test_config_validation.py
# ============================================================
# Unit tests for the Pydantic config schema.
# Verifies that bad config values are rejected loudly
# instead of silently corrupting trading behavior.
# ============================================================

import os
import pytest


@pytest.mark.smoke
def test_config_module_importable():
    """config.schemas must be importable."""
    from typed_config import schemas  # noqa: F401


@pytest.mark.unit
def test_valid_trading_config_loads():
    """A well-formed trading config should pass validation."""
    from typed_config.schemas import TradingConfig

    cfg = TradingConfig(
        mode="paper",
        default_lot_size=0.01,
        max_lot_size=0.5,
        max_concurrent_positions=3,
        risk_per_trade_pct=1.0,
    )
    assert cfg.mode == "paper"
    assert cfg.risk_per_trade_pct == 1.0


@pytest.mark.unit
def test_negative_lot_size_rejected():
    """Lot size must be positive — negative values must raise."""
    from typed_config.schemas import TradingConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TradingConfig(
            mode="paper",
            default_lot_size=-0.01,
            max_lot_size=0.5,
            max_concurrent_positions=3,
            risk_per_trade_pct=1.0,
        )


@pytest.mark.unit
def test_risk_per_trade_pct_capped():
    """Risk per trade > 5% is reckless for retail — must be rejected."""
    from typed_config.schemas import TradingConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TradingConfig(
            mode="paper",
            default_lot_size=0.01,
            max_lot_size=0.5,
            max_concurrent_positions=3,
            risk_per_trade_pct=10.0,  # too high
        )


@pytest.mark.unit
def test_invalid_mode_rejected():
    """Trading mode must be one of paper/live/demo."""
    from typed_config.schemas import TradingConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TradingConfig(
            mode="prod",  # typo — should be 'live'
            default_lot_size=0.01,
            max_lot_size=0.5,
            max_concurrent_positions=3,
            risk_per_trade_pct=1.0,
        )


@pytest.mark.unit
def test_secrets_redacted_on_repr():
    """API keys must not appear in the __repr__ output."""
    from typed_config.schemas import SecretsConfig

    secrets = SecretsConfig(
        oanda_api_key="super-secret-key-12345",
        oanda_account_id="001-001-12345-001",
    )
    repr_str = repr(secrets)
    assert "super-secret-key-12345" not in repr_str, (
        "API key leaked in repr — would be exposed in logs/errors"
    )
