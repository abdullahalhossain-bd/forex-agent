"""
risk/volatility_adjuster.py — Volatility-Based Position Adjustment (Day 76)
=============================================================================

Adjusts position size based on current market volatility (ATR).

Rules:
  - ATR < 0.7× median  → LOW vol  → 1.2× (slight boost, small moves)
  - ATR 0.7-1.3× median → NORMAL  → 1.0× (normal size)
  - ATR 1.3-1.8× median → HIGH    → 0.6× (reduce, bigger SL needed)
  - ATR 1.8-2.5× median → ELEVATED → 0.4× (significant reduction)
  - ATR > 2.5× median   → EXTREME → 0.2× (minimal size, news event)
  - ATR > 4.0× median   → BLOCKED → 0.0× (no trade, too dangerous)

Also checks if we're in a news window (high volatility expected).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from utils.logger import get_logger

log = get_logger("volatility_adjuster")


@dataclass
class VolatilityResult:
    """Output of volatility adjustment."""
    factor: float           # multiplier (0.0 to 1.2)
    level: str               # LOW / NORMAL / HIGH / ELEVATED / EXTREME / BLOCKED
    atr_ratio: float         # current ATR / median ATR
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VolatilityAdjuster:
    """ATR-based volatility position adjustment."""

    # Volatility thresholds (ratio of current ATR to median ATR)
    LOW_VOL_THRESHOLD = 0.7
    HIGH_VOL_THRESHOLD = 1.3
    ELEVATED_VOL_THRESHOLD = 1.8
    EXTREME_VOL_THRESHOLD = 2.5
    BLOCK_VOL_THRESHOLD = 4.0

    # Multipliers
    LOW_VOL_MULT = 1.2
    NORMAL_VOL_MULT = 1.0
    HIGH_VOL_MULT = 0.6
    ELEVATED_VOL_MULT = 0.4
    EXTREME_VOL_MULT = 0.2
    BLOCK_MULT = 0.0

    def adjust(
        self,
        atr: float,
        atr_median: float,
        news_active: bool = False,
    ) -> VolatilityResult:
        """Calculate volatility adjustment factor.

        Args:
            atr: Current ATR value.
            atr_median: Median ATR (historical baseline).
            news_active: Whether high-impact news is active.

        Returns:
            VolatilityResult with multiplier + level.
        """
        if atr_median <= 0:
            return VolatilityResult(
                factor=self.NORMAL_VOL_MULT, level="NORMAL",
                atr_ratio=1.0, reason="No ATR baseline — using normal",
            )

        ratio = atr / atr_median

        # News override — if news is active, cap at extreme
        if news_active:
            return VolatilityResult(
                factor=self.EXTREME_VOL_MULT, level="EXTREME",
                atr_ratio=ratio,
                reason=f"News active — volatility capped at {self.EXTREME_VOL_MULT}×",
            )

        if ratio >= self.BLOCK_VOL_THRESHOLD:
            return VolatilityResult(
                factor=self.BLOCK_MULT, level="BLOCKED",
                atr_ratio=ratio,
                reason=f"ATR {ratio:.1f}× median ≥ {self.BLOCK_VOL_THRESHOLD}× — too volatile",
            )
        elif ratio >= self.EXTREME_VOL_THRESHOLD:
            return VolatilityResult(
                factor=self.EXTREME_VOL_MULT, level="EXTREME",
                atr_ratio=ratio,
                reason=f"ATR {ratio:.1f}× — extreme volatility, size ×{self.EXTREME_VOL_MULT}",
            )
        elif ratio >= self.ELEVATED_VOL_THRESHOLD:
            return VolatilityResult(
                factor=self.ELEVATED_VOL_MULT, level="ELEVATED",
                atr_ratio=ratio,
                reason=f"ATR {ratio:.1f}× — elevated volatility, size ×{self.ELEVATED_VOL_MULT}",
            )
        elif ratio >= self.HIGH_VOL_THRESHOLD:
            return VolatilityResult(
                factor=self.HIGH_VOL_MULT, level="HIGH",
                atr_ratio=ratio,
                reason=f"ATR {ratio:.1f}× — high volatility, size ×{self.HIGH_VOL_MULT}",
            )
        elif ratio < self.LOW_VOL_THRESHOLD:
            return VolatilityResult(
                factor=self.LOW_VOL_MULT, level="LOW",
                atr_ratio=ratio,
                reason=f"ATR {ratio:.1f}× — low volatility, size ×{self.LOW_VOL_MULT}",
            )
        else:
            return VolatilityResult(
                factor=self.NORMAL_VOL_MULT, level="NORMAL",
                atr_ratio=ratio,
                reason=f"ATR {ratio:.1f}× — normal volatility",
            )


# ── Singleton ───────────────────────────────────────────────────────

_ADJ: Optional[VolatilityAdjuster] = None


def get_volatility_adjuster() -> VolatilityAdjuster:
    global _ADJ
    if _ADJ is None:
        _ADJ = VolatilityAdjuster()
    return _ADJ
