# risk/position_allocator.py — Day 58 | Position Sizing & Kelly Criterion
# ============================================================
# Advanced position sizing using Kelly Criterion, with dynamic
# adjustment based on confidence, risk mode, and market conditions.
#
# Kelly Criterion:
#   Optimal Risk = (W * R - L) / R
#   Where:
#     W = Win probability
#     L = Loss probability (1 - W)
#     R = Average win / Average loss (Reward/Risk ratio)
#
# Safety: We use Fractional Kelly (25%) to reduce variance.
# ============================================================

import math
from utils.logger import get_logger

log = get_logger("position_allocator")


class PositionAllocator:
    """
    Position Sizing Engine with Kelly Criterion.

    AI-এর Position Sizing Brain.
    Determines optimal lot size for each trade using:

      1. Kelly Criterion (mathematical optimal sizing)
      2. Fractional Kelly (25% for safety)
      3. Confidence-based adjustment
      4. Risk mode scaling
      5. Dynamic minimum R:R requirements

    The Kelly Criterion tells us the mathematically optimal fraction
    of our bankroll to risk on each trade. Full Kelly maximizes
    long-term growth but has enormous variance. We use 25% Kelly
    (quarter Kelly) for much smoother equity curves.

    Usage:
        pa = PositionAllocator(balance=10000, kelly_fraction=0.25)
        risk = pa.calculate_kelly_risk(
            win_rate=0.60,
            avg_win=2.0,    # average winning trade RR
            avg_loss=1.0,    # average losing trade RR
        )
        # risk = 0.10 → 10% of bankroll (full Kelly)
        # With 25% fraction → 2.5% risk per trade
    """

    def __init__(
        self,
        balance: float = 10000.0,
        kelly_fraction: float = 0.25,   # Use 25% of full Kelly
        min_rr: float = 1.5,            # Minimum R:R for any trade
    ):
        self.balance = balance
        self.kelly_fraction = kelly_fraction
        self.min_rr = min_rr

        # Risk caps by mode
        self._mode_rr_requirements = {
            "AGGRESSIVE": 1.5,    # Min RR: 1:1.5 in aggressive
            "NORMAL": 2.0,        # Min RR: 1:2 in normal
            "DEFENSIVE": 2.5,      # Min RR: 1:2.5 in defensive
            "EMERGENCY": 999.0,    # Effectively no trades allowed
        }

        log.info(
            f"[PositionAllocator] Initialized | "
            f"Balance: ${balance:,.2f} | "
            f"Kelly Fraction: {kelly_fraction*100:.0f}% | "
            f"Min RR: 1:{min_rr}"
        )

    # ═══════════════════════════════════════════════════════
    # KELLY CRITERION CALCULATION
    # ═══════════════════════════════════════════════════════

    def calculate_kelly_risk(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate Kelly Criterion optimal risk fraction.

        Formula:
            Kelly % = (W * R - L) / R

        Where:
            W = Win probability (e.g., 0.60)
            L = Loss probability (1 - W)
            R = Reward/Risk ratio (avg_win / avg_loss)

        Example:
            Win rate: 60%, Avg win: $200, Avg loss: $100
            R = 200/100 = 2.0
            Kelly = (0.60 * 2.0 - 0.40) / 2.0 = 0.40 (40%)

            With 25% fractional Kelly → 0.40 * 0.25 = 10% risk

        Args:
            win_rate: Win probability (0.0 - 1.0)
            avg_win: Average winning trade amount or RR ratio
            avg_loss: Average losing trade amount or RR ratio

        Returns:
            Risk fraction (0.0 - 1.0). Already includes fractional Kelly.
        """
        if win_rate <= 0 or win_rate >= 1:
            return 0.0

        if avg_loss <= 0:
            return 0.0

        W = win_rate
        L = 1.0 - W
        R = avg_win / avg_loss  # Reward/Risk ratio

        if R <= 0:
            return 0.0

        # Full Kelly
        kelly_full = (W * R - L) / R

        # If Kelly is negative, don't trade (negative edge)
        if kelly_full <= 0:
            log.info(
                f"[PositionAllocator] Negative Kelly ({kelly_full:.3f}) — "
                f"no edge detected. WR: {win_rate:.1%}, RR: 1:{R:.2f}"
            )
            return 0.0

        # Apply fractional Kelly for safety
        kelly_fractional = kelly_full * self.kelly_fraction

        # Cap at reasonable maximum (never risk more than 5% even with Kelly)
        max_risk = 0.05
        kelly_capped = min(kelly_fractional, max_risk)

        log.debug(
            f"[PositionAllocator] Kelly: Full={kelly_full:.3f} | "
            f"Fractional({self.kelly_fraction*100:.0f}%)={kelly_fractional:.3f} | "
            f"Capped={kelly_capped:.3f}"
        )

        return round(kelly_capped, 4)

    def calculate_lot_size(
        self,
        risk_fraction: float,
        sl_pips: float,
        symbol: str = "EURUSD",
    ) -> float:
        """
        Convert risk fraction to lot size.

        Args:
            risk_fraction: Risk as fraction of balance (e.g., 0.01 = 1%)
            sl_pips: Stop loss distance in pips
            symbol: Currency pair

        Returns:
            Lot size (rounded to 0.01)
        """
        from core.constants import get_pip_value_usd

        if sl_pips <= 0:
            return 0.01

        risk_usd = self.balance * risk_fraction
        pip_val = get_pip_value_usd(symbol)

        lot = risk_usd / (sl_pips * pip_val) if pip_val > 0 else 0.01
        lot = max(0.01, min(round(lot, 2), 100.0))

        return lot

    # ═══════════════════════════════════════════════════════
    # DYNAMIC R:R REQUIREMENTS
    # ═══════════════════════════════════════════════════════

    def get_minimum_rr(self, confidence: float, mode: str) -> float:
        """
        Get dynamic minimum R:R based on confidence and risk mode.

        Higher confidence → accept lower R:R (the setup is strong)
        Lower confidence  → require higher R:R (need more margin of safety)

        Risk mode adjustment:
          AGGRESSIVE: lower RR threshold
          NORMAL: standard RR threshold
          DEFENSIVE: higher RR threshold (only take excellent trades)
        """
        base_rr = self._mode_rr_requirements.get(mode, 2.0)

        # Confidence adjustment: reduce RR requirement for high confidence
        if confidence >= 85:
            confidence_discount = 0.8   # 20% lower RR requirement
        elif confidence >= 70:
            confidence_discount = 1.0   # standard
        elif confidence >= 55:
            confidence_discount = 1.2   # 20% higher RR requirement
        else:
            confidence_discount = 1.5   # 50% higher RR requirement

        adjusted_rr = base_rr * confidence_discount
        return round(adjusted_rr, 1)

    # ═══════════════════════════════════════════════════════
    # CONFIDENCE-ADJUSTED SIZING
    # ═══════════════════════════════════════════════════════

    def adjust_for_confidence(
        self,
        base_risk: float,
        confidence: float,
    ) -> float:
        """
        Adjust position size based on trade confidence.

        Scale factor:
          95+ confidence: 1.2x (boost by 20%)
          80-95 confidence: 1.0x (standard)
          65-80 confidence: 0.8x (reduce by 20%)
          50-65 confidence: 0.5x (reduce by 50%)
          < 50 confidence:  0.0 (no trade)
        """
        if confidence >= 95:
            scale = 1.2
        elif confidence >= 80:
            scale = 1.0
        elif confidence >= 65:
            scale = 0.8
        elif confidence >= 50:
            scale = 0.5
        else:
            return 0.0

        adjusted = base_risk * scale
        return round(min(adjusted, 0.05), 4)  # never exceed 5%

    # ═══════════════════════════════════════════════════════
    # KELLY ANALYSIS & REPORTING
    # ═══════════════════════════════════════════════════════

    def analyze_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> dict:
        """
        Complete Kelly Criterion analysis.

        Returns detailed breakdown of Kelly calculation and recommendations.
        """
        if avg_loss <= 0:
            return {
                "kelly_full": 0, "kelly_fractional": 0,
                "recommended_risk_pct": 0, "edge": "None (no loss data)",
            }

        W = win_rate
        L = 1 - W
        R = avg_win / avg_loss

        kelly_full = (W * R - L) / R if R > 0 else 0
        kelly_fractional = kelly_full * self.kelly_fraction

        # Edge analysis
        edge = W * avg_win - L * avg_loss  # Expected value per trade
        if edge > 0:
            edge_desc = f"Positive edge: +{edge:.2f} per unit risked"
        else:
            edge_desc = f"Negative edge: {edge:.2f} — do not trade"

        # Optimal risk in dollars
        recommended_risk_usd = self.balance * kelly_fractional

        return {
            "win_rate": round(W * 100, 1),
            "loss_rate": round(L * 100, 1),
            "reward_risk_ratio": round(R, 2),
            "kelly_full_pct": round(kelly_full * 100, 2),
            "kelly_fractional_pct": round(kelly_fractional * 100, 2),
            "kelly_fraction_used": self.kelly_fraction,
            "recommended_risk_pct": round(kelly_fractional * 100, 2),
            "recommended_risk_usd": round(recommended_risk_usd, 2),
            "edge": edge_desc,
            "edge_value": round(edge, 4),
            "has_edge": edge > 0,
            "action": "TRADE" if kelly_fractional > 0 else "NO TRADE",
        }

    def print_kelly_analysis(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> None:
        """Print formatted Kelly analysis."""
        analysis = self.analyze_kelly(win_rate, avg_win, avg_loss)
        bar = "=" * 48
        print(f"\n{bar}")
        print("  KELLY CRITERION ANALYSIS")
        print(bar)
        print(f"  Win Rate         : {analysis['win_rate']}%")
        print(f"  Loss Rate        : {analysis['loss_rate']}%")
        print(f"  Reward:Risk      : 1:{analysis['reward_risk_ratio']}")
        print(f"  Full Kelly       : {analysis['kelly_full_pct']}%")
        print(f"  Fractional Kelly : {analysis['kelly_fractional_pct']}% ({self.kelly_fraction*100:.0f}%)")
        print(f"  Recommended Risk : {analysis['recommended_risk_pct']}%")
        print(f"  Risk in $        : ${analysis['recommended_risk_usd']:,.2f}")
        print(f"  Edge             : {analysis['edge']}")
        print(f"  Action           : {analysis['action']}")
        print(bar + "\n")
