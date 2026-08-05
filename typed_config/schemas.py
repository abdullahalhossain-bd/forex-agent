from pydantic import BaseModel, SecretStr, Field, validator
from typing import Optional

class TradingConfig(BaseModel):
    mode: str = Field("paper", description="Trading mode: paper/live/demo")
    default_lot_size: float = Field(0.01, gt=0)
    max_lot_size: float = Field(0.5, gt=0)
    max_concurrent_positions: int = Field(5, ge=1)
    risk_per_trade_pct: float = Field(0.5, gt=0)

    @validator("mode")
    def validate_mode(cls, v):
        if v not in ("paper", "live", "demo"):
            raise ValueError("mode must be one of paper/live/demo")
        return v

    @validator("risk_per_trade_pct")
    def cap_risk(cls, v):
        if v > 5.0:
            raise ValueError("risk_per_trade_pct must be <= 5.0")
        return v


class SecretsConfig(BaseModel):
    oanda_api_key: Optional[SecretStr] = None
    oanda_account_id: Optional[str] = None

    class Config:
        # Pydantic hides SecretStr contents by default; keep repr safe
        keep_untouched = (SecretStr,)


class AppConfig(BaseModel):
    trading: TradingConfig = TradingConfig()
    secrets: Optional[SecretsConfig] = None


# module-level exports for tests
__all__ = ["TradingConfig", "SecretsConfig", "AppConfig"]