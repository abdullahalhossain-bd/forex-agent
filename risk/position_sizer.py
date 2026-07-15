"""
risk/position_sizer.py — Advanced Position Sizing Engine (Day 76)
===================================================================

The ultimate position sizing system that combines 5 adjustment factors:

  1. Base Risk % (from capital tier)
  2. Kelly Criterion (historical performance edge)
  3. Volatility Adjustment (ATR-based)
  4. Confidence Scaling (Master Decision confidence)
  5. Correlation Adjustment (portfolio exposure)

Plus additional safety features:
  - Drawdown Adaptive Sizing (reduce risk when drawdown is high)
  - Profit Protection Mode (reduce risk at new equity highs)
  - Loss Streak Penalty (reduce after consecutive losses)
  - Portfolio Heat Check (total risk across all positions)
  - Hard caps (never exceed max risk)

Final Formula:
  final_lot = base_lot × kelly × volatility × confidence × correlation × drawdown × streak

All factors are clamped to prevent extreme values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

from risk.kelly_calculator import KellyCalculator, KellyResult, get_kelly_calculator
from risk.volatility_adjuster import VolatilityAdjuster, VolatilityResult, get_volatility_adjuster
from risk.confidence_scaler import ConfidenceScaler, ConfidenceResult, get_confidence_scaler
from risk.correlation_manager import CorrelationManager, CorrelationResult, get_correlation_manager

log = get_logger("position_sizer")

# Absolute hard caps
MAX_RISK_PCT = 0.02        # never risk more than 2% per trade
MIN_LOT = 0.01             # minimum lot size

# Day 81+ hotfix: MAX_LOT was 50.0 — way too high.  On a $10k account
# with 1% risk, a 15-pip SL gives lot=0.67, but Kelly × vol × conf ×
# corr multipliers can compound to 2-3x, producing lot=1.8-2.5 which
# is 2.7-4% risk per trade.  This caused the $435 loss the user saw.
# Load from config so it's overridable via .env without code change.
try:
    from config import MAX_LOT as _CFG_MAX_LOT
    MAX_LOT = float(_CFG_MAX_LOT)
except Exception as e:
    # P1 fix (audit §6): this used to fail silently to a hardcoded 0.20
    # with only a code comment marking it — no log line — so a renamed
    # config.MAX_LOT or import-order change would silently change the
    # system's real position-size cap with zero runtime visibility.
    MAX_LOT = 0.20  # safe default — $10k account
    log.warning(
        f"[position_sizer] config.MAX_LOT unavailable ({e}) — "
        f"falling back to hardcoded MAX_LOT={MAX_LOT}. If this is "
        f"unexpected, check config.py for a renamed/missing MAX_LOT."
    )


@dataclass
class AdvancedPositionResult:
    """Complete output of the advanced position sizing engine."""
    lot: float
    risk_amount_usd: float
    risk_pct: float
    base_lot: float
    kelly_result: Dict[str, Any]
    volatility_result: Dict[str, Any]
    confidence_result: Dict[str, Any]
    correlation_result: Dict[str, Any]
    drawdown_mult: float
    streak_mult: float
    profit_protection_mult: float
    tier_mult: float
    final_mult: float          # combined multiplier
    explanation: List[str]
    approved: bool
    reject_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def reason(self) -> str:
        """Backward-compat: live_risk_manager uses sizing.reason for reject text."""
        if not self.approved:
            return self.reject_reason
        # Approved — return the last explanation line as the summary reason
        return self.explanation[-1] if self.explanation else ""


class PositionSizer:
    """Advanced position sizing with 5-factor adjustment."""

    def __init__(self):
        self.kelly = get_kelly_calculator()
        self.volatility = get_volatility_adjuster()
        self.confidence = get_confidence_scaler()
        self.correlation = get_correlation_manager()

    def calculate(
        self,
        balance: float,
        risk_pct: float,
        sl_pips: float,
        pip_value_per_lot: float = 10.0,
        confidence: float = 70.0,
        atr: float = 0.001,
        atr_median: float = 0.001,
        consecutive_losses: int = 0,
        tier_mult: float = 0.5,
        # Kelly inputs
        win_rate: float = None,
        avg_win_r: float = None,
        avg_loss_r: float = None,
        trade_count: int = 0,
        # Correlation inputs
        pair: str = "EURUSD",
        direction: str = "BUY",
        open_positions: List[Dict] = None,
        # Drawdown inputs
        current_drawdown_pct: float = 0.0,
        # Profit protection
        is_new_equity_high: bool = False,
        # News
        news_active: bool = False,
    ) -> AdvancedPositionResult:
        """Calculate the optimal position size using all 5 factors.

        This is the MAIN entry point for position sizing.

        Returns:
            AdvancedPositionResult with lot + full breakdown + explanation.
        """
        explanation: List[str] = []
        open_positions = open_positions or []

        if sl_pips <= 0 or pip_value_per_lot <= 0 or balance <= 0:
            return AdvancedPositionResult(
                lot=0.0, risk_amount_usd=0.0, risk_pct=0.0, base_lot=0.0,
                kelly_result={}, volatility_result={}, confidence_result={},
                correlation_result={}, drawdown_mult=0.0, streak_mult=0.0,
                profit_protection_mult=0.0, tier_mult=0.0, final_mult=0.0,
                explanation=["Invalid inputs (SL, pip value, or balance ≤ 0)"],
                approved=False, reject_reason="Invalid inputs",
            )

        # ── 1. Base risk amount ─────────────────────────────────────
        base_risk = balance * risk_pct
        base_lot = base_risk / (sl_pips * pip_value_per_lot)
        base_lot = max(MIN_LOT, min(base_lot, MAX_LOT))
        explanation.append(f"Base: ${base_risk:.0f} risk → lot {base_lot:.2f}")

        # ── 2. Kelly Criterion ──────────────────────────────────────
        kelly_res = self.kelly.calculate(
            win_rate=win_rate,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            trade_count=trade_count,
            confidence=confidence,
        )
        kelly_mult = kelly_res.final_risk_pct / risk_pct if risk_pct > 0 else 0
        kelly_mult = max(0.0, min(kelly_mult, 1.5))  # cap Kelly multiplier
        explanation.append(f"Kelly: {kelly_res.reason}")

        if not kelly_res.is_valid and kelly_res.final_risk_pct == 0:
            return AdvancedPositionResult(
                lot=0.0, risk_amount_usd=0.0, risk_pct=0.0, base_lot=base_lot,
                kelly_result=kelly_res.to_dict(), volatility_result={},
                confidence_result={}, correlation_result={},
                drawdown_mult=0.0, streak_mult=0.0, profit_protection_mult=0.0,
                tier_mult=tier_mult, final_mult=0.0,
                explanation=explanation, approved=False,
                reject_reason=kelly_res.reason,
            )

        # ── 3. Volatility adjustment ────────────────────────────────
        vol_res = self.volatility.adjust(atr, atr_median, news_active)
        explanation.append(f"Volatility: {vol_res.reason}")

        if vol_res.factor == 0.0:
            return AdvancedPositionResult(
                lot=0.0, risk_amount_usd=0.0, risk_pct=0.0, base_lot=base_lot,
                kelly_result=kelly_res.to_dict(),
                volatility_result=vol_res.to_dict(),
                confidence_result={}, correlation_result={},
                drawdown_mult=0.0, streak_mult=0.0, profit_protection_mult=0.0,
                tier_mult=tier_mult, final_mult=0.0,
                explanation=explanation, approved=False,
                reject_reason=vol_res.reason,
            )

        # ── 4. Confidence scaling ───────────────────────────────────
        conf_res = self.confidence.scale(confidence)
        explanation.append(f"Confidence: {conf_res.reason}")

        if conf_res.factor == 0.0:
            return AdvancedPositionResult(
                lot=0.0, risk_amount_usd=0.0, risk_pct=0.0, base_lot=base_lot,
                kelly_result=kelly_res.to_dict(),
                volatility_result=vol_res.to_dict(),
                confidence_result=conf_res.to_dict(),
                correlation_result={},
                drawdown_mult=0.0, streak_mult=0.0, profit_protection_mult=0.0,
                tier_mult=tier_mult, final_mult=0.0,
                explanation=explanation, approved=False,
                reject_reason=conf_res.reason,
            )

        # ── 5. Correlation adjustment ───────────────────────────────
        est_risk = base_risk * kelly_mult * vol_res.factor * conf_res.factor
        corr_res = self.correlation.adjust(
            pair=pair, direction=direction,
            open_positions=open_positions,
            balance=balance, proposed_risk_usd=est_risk,
        )
        explanation.append(f"Correlation: {corr_res.reason}")

        if corr_res.factor == 0.0:
            return AdvancedPositionResult(
                lot=0.0, risk_amount_usd=0.0, risk_pct=0.0, base_lot=base_lot,
                kelly_result=kelly_res.to_dict(),
                volatility_result=vol_res.to_dict(),
                confidence_result=conf_res.to_dict(),
                correlation_result=corr_res.to_dict(),
                drawdown_mult=0.0, streak_mult=0.0, profit_protection_mult=0.0,
                tier_mult=tier_mult, final_mult=0.0,
                explanation=explanation, approved=False,
                reject_reason=corr_res.reason,
            )

        # ── 6. Drawdown adaptive sizing ─────────────────────────────
        dd_mult = self._drawdown_mult(current_drawdown_pct)
        explanation.append(f"Drawdown: {current_drawdown_pct:.1%} → ×{dd_mult}")

        # ── 7. Loss streak penalty ──────────────────────────────────
        streak_mult = self._streak_mult(consecutive_losses)
        explanation.append(f"Streak: {consecutive_losses} losses → ×{streak_mult}")

        if streak_mult == 0.0:
            return AdvancedPositionResult(
                lot=0.0, risk_amount_usd=0.0, risk_pct=0.0, base_lot=base_lot,
                kelly_result=kelly_res.to_dict(),
                volatility_result=vol_res.to_dict(),
                confidence_result=conf_res.to_dict(),
                correlation_result=corr_res.to_dict(),
                drawdown_mult=dd_mult, streak_mult=0.0,
                profit_protection_mult=0.0, tier_mult=tier_mult, final_mult=0.0,
                explanation=explanation, approved=False,
                reject_reason=f"Loss streak {consecutive_losses} — trading halted",
            )

        # ── 8. Profit protection mode ───────────────────────────────
        profit_mult = 0.7 if is_new_equity_high else 1.0
        if is_new_equity_high:
            explanation.append(f"Profit protection: new equity high → ×{profit_mult}")

        # ── Final calculation ───────────────────────────────────────
        final_mult = (
            kelly_mult *
            vol_res.factor *
            conf_res.factor *
            corr_res.factor *
            dd_mult *
            streak_mult *
            profit_mult *
            tier_mult
        )

        final_lot = base_lot * final_mult
        final_lot = max(MIN_LOT, min(round(final_lot, 2), MAX_LOT))

        # Calculate actual risk
        actual_risk = final_lot * sl_pips * pip_value_per_lot
        actual_risk_pct = actual_risk / balance if balance > 0 else 0

        # Hard cap on risk %
        if actual_risk_pct > MAX_RISK_PCT:
            capped_risk = balance * MAX_RISK_PCT
            final_lot = max(MIN_LOT, min(round(capped_risk / (sl_pips * pip_value_per_lot), 2), MAX_LOT))
            actual_risk = final_lot * sl_pips * pip_value_per_lot
            actual_risk_pct = actual_risk / balance
            explanation.append(f"Hard cap: risk capped at {MAX_RISK_PCT:.0%} → lot {final_lot:.2f}")

        explanation.append(f"→ FINAL: lot={final_lot:.2f} | risk=${actual_risk:.0f} ({actual_risk_pct:.2%})")

        return AdvancedPositionResult(
            lot=final_lot,
            risk_amount_usd=round(actual_risk, 2),
            risk_pct=round(actual_risk_pct, 4),
            base_lot=round(base_lot, 2),
            kelly_result=kelly_res.to_dict(),
            volatility_result=vol_res.to_dict(),
            confidence_result=conf_res.to_dict(),
            correlation_result=corr_res.to_dict(),
            drawdown_mult=dd_mult,
            streak_mult=streak_mult,
            profit_protection_mult=profit_mult,
            tier_mult=tier_mult,
            final_mult=round(final_mult, 4),
            explanation=explanation,
            approved=True,
        )

    def _drawdown_mult(self, drawdown_pct: float) -> float:
        """Drawdown adaptive sizing."""
        if drawdown_pct >= 0.15:
            return 0.0   # emergency — no trades
        elif drawdown_pct >= 0.12:
            return 0.25  # protective
        elif drawdown_pct >= 0.08:
            return 0.5   # defensive
        elif drawdown_pct >= 0.05:
            return 0.7   # cautious
        else:
            return 1.0   # normal

    def _streak_mult(self, consecutive_losses: int) -> float:
        """Loss streak penalty.

        P1 fix (audit §4.3): the halt threshold here (previously a
        separate hardcoded 5) used to be dead code beyond loss #2 in
        practice, because CircuitBreaker.MAX_CONSECUTIVE_LOSSES=3 halts
        trading before a 4th or 5th loss could ever reach this method —
        and the two thresholds could be tuned independently and silently
        drift apart. Halt threshold is now derived from
        CircuitBreaker.MAX_CONSECUTIVE_LOSSES (falling back to the
        original hardcoded 5 only if CircuitBreaker can't be imported),
        so there's exactly one number that determines "how many losses
        before trading stops" — the intermediate discount steps below it
        are unaffected and still provide graduated de-risking.
        """
        try:
            from risk.circuit_breaker import CircuitBreaker
            halt_at = CircuitBreaker.MAX_CONSECUTIVE_LOSSES
        except Exception:
            halt_at = 5  # fallback if CircuitBreaker unavailable

        if consecutive_losses >= halt_at:
            return 0.0   # halt — matches CircuitBreaker's own threshold
        elif consecutive_losses >= max(halt_at - 1, 1):
            return 0.4
        elif consecutive_losses >= max(halt_at - 2, 1):
            return 0.7
        elif consecutive_losses >= 2:
            return 0.85
        else:
            return 1.0


# ── Backward-compat alias ────────────────────────────────────────────
# Earlier Day 75 code (live_risk_manager.py) imports `PositionSizeResult`.
# Day 76 renames it to `AdvancedPositionResult`. Keep both names working.
PositionSizeResult = AdvancedPositionResult


# ── Singleton ───────────────────────────────────────────────────────

_SIZER: Optional[PositionSizer] = None


def get_position_sizer() -> PositionSizer:
    global _SIZER
    if _SIZER is None:
        _SIZER = PositionSizer()
    return _SIZER