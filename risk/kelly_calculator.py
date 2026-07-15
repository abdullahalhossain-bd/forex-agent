"""
risk/kelly_calculator.py — Kelly Criterion Calculator (Day 76)
================================================================

Calculates optimal position size using the Kelly Criterion, adjusted
for forex trading safety.

Kelly Formula:
  Kelly % = W - ((1 - W) / R)

Where:
  W = Win probability (historical win rate)
  R = Average win / average loss (in R multiples)

Safety measures:
  1. Half-Kelly (use 50% of full Kelly — standard professional practice)
  2. Hard cap at MAX_KELLY_RISK (default 2% — never exceed)
  3. Minimum sample size required (20 trades)
  4. If Kelly is negative → no trade (edge is negative)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from utils.logger import get_logger

log = get_logger("kelly_calculator")

MAX_KELLY_RISK = 0.02        # never risk more than 2% based on Kelly
HALF_KELLY_DIVISOR = 2.0     # use half-Kelly for safety
MIN_SAMPLES = 20             # need at least 20 trades for Kelly
DEFAULT_WIN_RATE = 0.50      # if no history, assume 50%
DEFAULT_RR_RATIO = 1.5       # if no history, assume 1:1.5 R:R


@dataclass
class KellyResult:
    """Output of Kelly calculation."""
    kelly_pct: float          # full Kelly % (0-1)
    half_kelly_pct: float     # half Kelly % (0-1)
    final_risk_pct: float     # capped final risk % (0-1)
    win_rate: float           # historical win rate used
    rr_ratio: float           # win/loss ratio used
    sample_size: int          # trades used for calculation
    is_valid: bool            # False if insufficient data or negative Kelly
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KellyCalculator:
    """Kelly Criterion position sizing with safety caps."""

    def calculate(
        self,
        win_rate: float = None,
        avg_win_r: float = None,
        avg_loss_r: float = None,
        trade_count: int = 0,
        confidence: float = 70.0,
    ) -> KellyResult:
        """Calculate Kelly-based risk percentage.

        Args:
            win_rate: Historical win rate (0-1). If None, uses default.
            avg_win_r: Average win in R multiples (e.g. 2.0 = 2R win).
            avg_loss_r: Average loss in R multiples (e.g. 1.0 = 1R loss).
            trade_count: Number of historical trades.
            confidence: Current trade confidence (0-100) — adjusts Kelly weight.

        Returns:
            KellyResult with full/half/capped Kelly percentages.
        """
        # Use defaults if no history
        w = win_rate if win_rate is not None else DEFAULT_WIN_RATE
        r_ratio = (avg_win_r / avg_loss_r) if (avg_win_r and avg_loss_r and avg_loss_r > 0) else DEFAULT_RR_RATIO

        # Check minimum sample size
        if trade_count < MIN_SAMPLES:
            result = KellyResult(
                kelly_pct=0.0, half_kelly_pct=0.0, final_risk_pct=0.0,
                win_rate=w, rr_ratio=r_ratio, sample_size=trade_count,
                is_valid=False,
                reason=f"Insufficient samples ({trade_count} < {MIN_SAMPLES}) — using default risk",
            )
            # Return a safe default risk
            result.final_risk_pct = 0.01  # default 1%
            return result

        # Kelly Formula: K% = W - ((1-W) / R)
        kelly = w - ((1 - w) / r_ratio)

        if kelly <= 0:
            return KellyResult(
                kelly_pct=kelly, half_kelly_pct=0, final_risk_pct=0,
                win_rate=w, rr_ratio=r_ratio, sample_size=trade_count,
                is_valid=False,
                reason=f"Kelly negative ({kelly:.3f}) — no statistical edge. Skip trade.",
            )

        # Half-Kelly (standard professional practice)
        half_kelly = kelly / HALF_KELLY_DIVISOR

        # Confidence adjustment: higher confidence → closer to full Kelly
        conf_mult = 0.5 + (confidence / 100.0) * 0.5  # 0.5 at 0% conf, 1.0 at 100%
        adjusted_kelly = half_kelly * conf_mult

        # Hard cap
        final_risk = min(adjusted_kelly, MAX_KELLY_RISK)

        return KellyResult(
            kelly_pct=round(kelly, 4),
            half_kelly_pct=round(half_kelly, 4),
            final_risk_pct=round(final_risk, 4),
            win_rate=w,
            rr_ratio=r_ratio,
            sample_size=trade_count,
            is_valid=True,
            reason=f"Kelly={kelly:.1%} → Half={half_kelly:.1%} → ConfAdj={adjusted_kelly:.1%} → Capped={final_risk:.1%}",
        )


# ── Singleton ───────────────────────────────────────────────────────

_CALC: Optional[KellyCalculator] = None


def get_kelly_calculator() -> KellyCalculator:
    global _CALC
    if _CALC is None:
        _CALC = KellyCalculator()
    return _CALC
