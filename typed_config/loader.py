"""typed_config.loader — load Pydantic-configured settings from env/.env"""
import os
from .schemas import TradingConfig

_ENV_MAP = {
    "FOREX_TRADING_MODE": ("mode", str),
    "FOREX_TRADING_RISK_PER_TRADE_PCT": ("risk_per_trade_pct", float),
    "FOREX_TRADING_DEFAULT_LOT_SIZE": ("default_lot_size", float),
    "FOREX_TRADING_MAX_LOT_SIZE": ("max_lot_size", float),
    "FOREX_TRADING_MAX_CONCURRENT_POSITIONS": ("max_concurrent_positions", int),
}


def load_trading_config() -> TradingConfig:
    """Read env vars and return a validated `TradingConfig` instance."""
    data = {}
    for env_var, (field, cast) in _ENV_MAP.items():
        val = os.getenv(env_var)
        if val is not None and val != "":
            try:
                data[field] = cast(val)
            except Exception:
                # let pydantic raise a clearer ValidationError later
                data[field] = val
    # Provide safe defaults via TradingConfig defaults
    return TradingConfig(**data)
