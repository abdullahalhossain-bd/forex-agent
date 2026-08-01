"""
risk/exposure_manager.py — Exposure & Correlation Manager (Day 75)
====================================================================

Prevents overexposure to a single currency or correlation group.
Tracks open positions and blocks new trades that would create
excessive risk.

Rules:
  1. Max same-direction positions in one correlation group: 2
  2. Max total open positions: 5 (configurable)
  3. Max single-currency exposure: 30% of balance
  4. No hedging same pair (BUY + SELL same pair)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("exposure_manager")

# Reuse correlation groups from scanner config
try:
    from scanner.config import CORRELATION_GROUPS
except Exception as e:
    CORRELATION_GROUPS = [
        {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"},
        {"USDCHF", "USDJPY", "USDCAD"},
        {"XAUUSD", "XAGUSD"},
    ]


@dataclass
class ExposureCheck:
    """Result of exposure check."""
    allowed: bool
    reason: str
    same_group_count: int = 0
    total_open: int = 0
    currency_exposure: Dict[str, float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExposureManager:
    """Manages portfolio exposure and correlation risk."""

    MAX_SAME_GROUP = 2        # max same-direction in one correlation group
    MAX_TOTAL_OPEN = 5        # max concurrent positions
    MAX_CURRENCY_PCT = 0.30   # max 30% balance risked on one currency

    def __init__(self):
        self._open_positions: List[Dict[str, Any]] = []

    def update_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Update the current open positions list."""
        self._open_positions = positions or []

    def check(
        self,
        pair: str,
        direction: str,
        lot: float,
        risk_usd: float,
        balance: float,
    ) -> ExposureCheck:
        """Check if a new trade would violate exposure limits.

        Args:
            pair: Trading pair (e.g. "EURUSD").
            direction: BUY or SELL.
            lot: Lot size.
            risk_usd: Risk amount in USD.
            balance: Account balance.

        Returns:
            ExposureCheck with allowed bool + reason.
        """
        pair = pair.upper()
        direction = direction.upper()
        total_open = len(self._open_positions)

        # Check 1: Total open positions
        if total_open >= self.MAX_TOTAL_OPEN:
            return ExposureCheck(
                allowed=False,
                reason=f"Max open positions reached ({total_open}/{self.MAX_TOTAL_OPEN})",
                total_open=total_open,
            )

        # Check 2: Same pair hedging
        for pos in self._open_positions:
            if pos.get("pair", "").upper() == pair:
                return ExposureCheck(
                    allowed=False,
                    reason=f"Already have open position on {pair}",
                    total_open=total_open,
                )

        # Check 3: Correlation group exposure
        group = self._find_group(pair)
        same_group_count = 0
        if group:
            for pos in self._open_positions:
                pos_pair = pos.get("pair", "").upper()
                pos_dir = pos.get("direction", "").upper()
                if pos_pair in group and pos_dir == direction:
                    same_group_count += 1

            if same_group_count >= self.MAX_SAME_GROUP:
                return ExposureCheck(
                    allowed=False,
                    reason=f"Correlation group limit: {same_group_count} same-direction in group {sorted(group)[:3]}",
                    same_group_count=same_group_count,
                    total_open=total_open,
                )

        # Check 4: Single currency exposure
        base_currency = pair[:3]
        quote_currency = pair[3:6]
        currency_exposure: Dict[str, float] = {}
        for pos in self._open_positions:
            pos_pair = pos.get("pair", "").upper()
            pos_risk = float(pos.get("risk_usd", 0))
            if len(pos_pair) >= 6:
                currency_exposure[pos_pair[:3]] = currency_exposure.get(pos_pair[:3], 0) + pos_risk
                currency_exposure[pos_pair[3:6]] = currency_exposure.get(pos_pair[3:6], 0) + pos_risk

        for curr in (base_currency, quote_currency):
            total_curr_risk = currency_exposure.get(curr, 0) + risk_usd
            if balance > 0 and total_curr_risk / balance > self.MAX_CURRENCY_PCT:
                return ExposureCheck(
                    allowed=False,
                    reason=f"Currency {curr} exposure {total_curr_risk/balance:.0%} > {self.MAX_CURRENCY_PCT:.0%} limit",
                    same_group_count=same_group_count,
                    total_open=total_open,
                    currency_exposure=currency_exposure,
                )

        return ExposureCheck(
            allowed=True,
            reason="Exposure OK",
            same_group_count=same_group_count,
            total_open=total_open,
            currency_exposure=currency_exposure,
        )

    def _find_group(self, pair: str) -> Optional[set]:
        """Find which correlation group a pair belongs to."""
        for group in CORRELATION_GROUPS:
            if pair in group:
                return group
        return None

    def status(self) -> Dict[str, Any]:
        return {
            "open_positions": len(self._open_positions),
            "max_total": self.MAX_TOTAL_OPEN,
            "max_same_group": self.MAX_SAME_GROUP,
            "positions": [{"pair": p.get("pair"), "direction": p.get("direction")} for p in self._open_positions],
        }


# ── Singleton ───────────────────────────────────────────────────────

_MANAGER: Optional[ExposureManager] = None


def get_exposure_manager() -> ExposureManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ExposureManager()
    return _MANAGER
