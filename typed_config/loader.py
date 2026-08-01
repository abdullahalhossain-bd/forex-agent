# typed_config/loader.py
# ============================================================
# Reads FOREX_TRADING_* / OANDA_* env vars (and a .env file, if
# python-dotenv is available) into validated typed_config.schemas
# objects. Falls back to conservative, safe defaults for anything
# unset — it never guesses in the direction of "more risk".
# ============================================================

import os
from typing import Optional

from .schemas import AppConfig, SecretsConfig, TradingConfig

_ENV_PREFIX = "FOREX_TRADING_"

# Load a .env file into os.environ if python-dotenv is installed.
# Never overrides variables the process/shell already set.
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(override=False)
except ImportError:
    pass


def _env_str(name: str, default: str) -> str:
    val = os.getenv(_ENV_PREFIX + name)
    return default if val is None or val == "" else val


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(_ENV_PREFIX + name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {_ENV_PREFIX}{name}={raw!r} is not a valid number"
        )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(_ENV_PREFIX + name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {_ENV_PREFIX}{name}={raw!r} is not a valid integer"
        )


def load_trading_config() -> TradingConfig:
    """Build a validated TradingConfig from FOREX_TRADING_* env vars.

    Defaults (used for anything unset) mirror the conservative values in
    the flat config.py: 0.01 default lot, 0.20 max lot, 10 concurrent
    positions, 0.5% risk per trade.
    """
    return TradingConfig(
        mode=_env_str("MODE", "paper"),
        default_lot_size=_env_float("DEFAULT_LOT_SIZE", 0.01),
        max_lot_size=_env_float("MAX_LOT_SIZE", 0.20),
        max_concurrent_positions=_env_int("MAX_CONCURRENT_POSITIONS", 10),
        risk_per_trade_pct=_env_float("RISK_PER_TRADE_PCT", 0.5),
    )


def load_secrets_config() -> SecretsConfig:
    """Build SecretsConfig from OANDA_* env vars (legacy naming, matches
    the keys already read directly in config.py)."""
    return SecretsConfig(
        oanda_api_key=os.environ.get("OANDA_API_KEY", ""),
        oanda_account_id=os.environ.get("OANDA_ACCOUNT_ID", ""),
    )


def load_app_config() -> AppConfig:
    """Build the full validated AppConfig from the environment."""
    return AppConfig(trading=load_trading_config(), secrets=load_secrets_config())
