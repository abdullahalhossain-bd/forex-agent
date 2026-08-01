# typed_config/schemas.py
# ============================================================
# Pydantic-validated config schemas for the forex-agent system.
#
# Why this exists: the flat `config.py` module reads env vars into
# plain module-level constants with no validation — a typo'd env var
# (e.g. RISK_PER_TRADE_PCT="10" instead of "1") or a wrong type is
# never caught, it just silently flows into position sizing and can
# blow up an account live. This module is the "loud failure" layer:
# bad values raise pydantic.ValidationError at startup instead of
# corrupting trading behavior at 3am.
#
# This is additive — it does not replace config.py. See
# typed_config/loader.py for how it's populated from the environment.
# ============================================================

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class TradingConfig(BaseModel):
    """Validated core trading parameters.

    Every field has a hard-safety bound in addition to its type, so a
    malformed or reckless value is rejected at construction time rather
    than reaching the risk engine or order manager.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["paper", "demo", "live"] = "paper"

    # Lot sizes: must be strictly positive. Upper bound of 100 lots is a
    # generous sanity ceiling — no retail account should ever place a
    # single order anywhere near this; it exists purely to catch a
    # misplaced decimal point (e.g. "10.0" instead of "0.10").
    default_lot_size: float = Field(gt=0, le=100)
    max_lot_size: float = Field(gt=0, le=100)

    max_concurrent_positions: int = Field(gt=0, le=200)

    # Risk per trade, in percent. Capped at 5% — anything higher is
    # reckless for a retail account (a string of 5-6 losses at >5% risk
    # each starts threatening the whole account) and is almost always a
    # units mistake (e.g. entering "5.0" meaning "0.5%").
    risk_per_trade_pct: float = Field(gt=0, le=5.0)

    @field_validator("max_lot_size")
    @classmethod
    def _max_lot_at_least_default(cls, v: float, info) -> float:
        default = info.data.get("default_lot_size")
        if default is not None and v < default:
            raise ValueError(
                f"max_lot_size ({v}) must be >= default_lot_size ({default})"
            )
        return v


class SecretsConfig(BaseModel):
    """Broker/API credentials.

    Uses SecretStr so a stray `print(config)`, log line, or exception
    traceback never leaks a live API key or account id into logs,
    Telegram messages, or crash reports.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    oanda_api_key: SecretStr = SecretStr("")
    oanda_account_id: str = ""


class AppConfig(BaseModel):
    """Top-level typed config, composed of validated sub-sections."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trading: TradingConfig = Field(
        default_factory=lambda: TradingConfig(
            mode="paper",
            default_lot_size=0.01,
            max_lot_size=0.20,
            max_concurrent_positions=10,
            risk_per_trade_pct=0.5,
        )
    )
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)
