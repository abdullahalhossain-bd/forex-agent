"""
risk/correlation_manager.py — Correlation-Adjusted Position Sizing (Day 76)
=============================================================================

Reduces or blocks position size based on correlation with existing
open positions. Prevents overexposure to a single currency or
correlation group.

Rules:
  - 0 correlated open positions → 1.0× (no adjustment)
  - 1 correlated same-direction → 0.7× (reduce 30%)
  - 2 correlated same-direction → 0.4× (reduce 60%)
  - 3+ correlated same-direction → 0.0× (block — too much exposure)

Also calculates portfolio heat (total risk across all open positions).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("correlation_manager")

try:
    from scanner.config import CORRELATION_GROUPS
except Exception as e:
    CORRELATION_GROUPS = [
        {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"},
        {"USDCHF", "USDJPY", "USDCAD"},
        {"XAUUSD", "XAGUSD"},
    ]

MAX_PORTFOLIO_HEAT = 0.05  # max 5% total risk across all open positions


@dataclass
class CorrelationResult:
    """Output of correlation adjustment."""
    factor: float                  # multiplier (0.0 to 1.0)
    correlated_count: int          # same-direction correlated positions
    correlation_group: List[str]   # which pairs are correlated
    portfolio_heat_pct: float      # total risk % across all positions
    heat_exceeded: bool            # portfolio heat limit exceeded
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CorrelationManager:
    """Correlation-aware position sizing + portfolio heat tracking."""

    # Multipliers based on correlated position count
    CORR_MULTIPLIERS = {
        0: 1.0,
        1: 0.7,
        2: 0.4,
    }

    def adjust(
        self,
        pair: str,
        direction: str,
        open_positions: List[Dict[str, Any]],
        balance: float,
        proposed_risk_usd: float,
    ) -> CorrelationResult:
        """Calculate correlation adjustment + portfolio heat.

        Args:
            pair: New trade pair.
            direction: BUY or SELL.
            open_positions: List of open positions with pair, direction, risk_usd.
            balance: Account balance.
            proposed_risk_usd: Risk amount for the new trade.

        Returns:
            CorrelationResult with multiplier + portfolio heat info.
        """
        pair = pair.upper()
        direction = direction.upper()

        # Find correlation group
        group = self._find_group(pair)
        group_pairs = sorted(group) if group else []

        # Count correlated same-direction positions
        correlated = 0
        if group:
            for pos in open_positions:
                pos_pair = pos.get("pair", "").upper()
                pos_dir = pos.get("direction", "").upper()
                if pos_pair in group and pos_pair != pair and pos_dir == direction:
                    correlated += 1

        # Get multiplier
        if correlated >= 3:
            return CorrelationResult(
                factor=0.0, correlated_count=correlated,
                correlation_group=group_pairs,
                portfolio_heat_pct=self._calc_heat(open_positions, balance),
                heat_exceeded=False,
                reason=f"{correlated} correlated same-direction positions → BLOCKED",
            )

        mult = self.CORR_MULTIPLIERS.get(correlated, 0.0)

        # Calculate portfolio heat
        current_heat = self._calc_heat(open_positions, balance)
        new_heat = current_heat + (proposed_risk_usd / balance if balance > 0 else 0)
        heat_exceeded = new_heat > MAX_PORTFOLIO_HEAT

        if heat_exceeded:
            return CorrelationResult(
                factor=0.0, correlated_count=correlated,
                correlation_group=group_pairs,
                portfolio_heat_pct=new_heat,
                heat_exceeded=True,
                reason=f"Portfolio heat {new_heat:.1%} > {MAX_PORTFOLIO_HEAT:.0%} limit → BLOCKED",
            )

        reason = f"{correlated} correlated → ×{mult}" if correlated > 0 else "No correlation risk"
        return CorrelationResult(
            factor=mult,
            correlated_count=correlated,
            correlation_group=group_pairs,
            portfolio_heat_pct=new_heat,
            heat_exceeded=False,
            reason=reason,
        )

    def _find_group(self, pair: str) -> Optional[set]:
        """Find correlation group for a pair."""
        for group in CORRELATION_GROUPS:
            if pair in group:
                return group
        return None

    def _calc_heat(self, positions: List[Dict], balance: float) -> float:
        """Calculate total portfolio heat (% risk across all open positions)."""
        if balance <= 0:
            return 0.0
        total_risk = sum(float(p.get("risk_usd", 0)) for p in positions)
        return total_risk / balance


# ── Singleton ───────────────────────────────────────────────────────

_MGR: Optional[CorrelationManager] = None


def get_correlation_manager() -> CorrelationManager:
    global _MGR
    if _MGR is None:
        _MGR = CorrelationManager()
    return _MGR
