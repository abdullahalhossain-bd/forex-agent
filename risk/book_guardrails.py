# risk/book_guardrails.py
# ============================================================
# Book Pages 136-151 — Final Risk Management Guardrails
# ============================================================
# Implements the last 3 deterministic rules from
# "The Only Technical Analysis Book You Will Ever Need":
#
# 1. Correlation-based exposure limit (Page 136)
#    Avoid stacking correlated FX pairs. If proposed position would
#    push portfolio correlation/concentration above threshold → reject.
#
# 2. Anti-revenge-trading guardrail (Pages 138-139)
#    "Never chase losses" — block oversized trades after a loss streak.
#    If recent_losses_streak >= N AND next_trade_size > normal_size → BLOCK.
#
# 3. Cost-aware expected-value gate (Page 138)
#    "Don't ignore fees/commissions" — net EV must be > 0.
#    net_ev = expected_pnl - (spread_cost + commission + slippage)
#    If net_ev <= 0 → REJECT trade.
#
# These are designed as drop-in guardrails for TradePermission.
# Each returns a GuardrailResult with pass/fail + reason.
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np

log = logging.getLogger(__name__)


# ─── Dataclass ────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """Result of a single guardrail check."""
    rule_name: str
    passed: bool
    reason: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "passed":    bool(self.passed),
            "reason":    self.reason,
            "details":   self.details,
        }


# ─── Constants ────────────────────────────────────────────────
DEFAULT_CORRELATION_THRESHOLD = 0.70   # Page 136: avoid > 0.70 correlation
DEFAULT_LOSS_STREAK_THRESHOLD = 3      # Page 138: 3+ consecutive losses
DEFAULT_POSITION_ESCALATION_MULT = 1.25  # 25% above normal = "escalation"
DEFAULT_MIN_NET_EV_PIPS = 1.0           # Net EV must be ≥ 1 pip after costs
DEFAULT_SPREAD_PIPS = {
    "EURUSD": 1.0, "USDJPY": 1.2, "GBPUSD": 1.5, "USDCHF": 1.8,
    "AUDUSD": 1.5, "USDCAD": 2.0, "NZDUSD": 1.8,
    "XAUUSD": 25.0,  # gold spread in pips (0.1 = 1 pip)
}
DEFAULT_COMMISSION_PIPS = 0.7   # round-trip commission in pips
DEFAULT_SLIPPAGE_PIPS = 0.5     # estimated slippage in pips


# ═════════════════════════════════════════════════════════════
# GUARDRAIL 1: Correlation-based Exposure Limit (Page 136)
# ═════════════════════════════════════════════════════════════

def check_correlation_exposure(
    proposed_pair: str,
    proposed_direction: str,
    open_positions: List[dict],
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> GuardrailResult:
    """
    Book Page 136 — Diversification rule.
    "Diversify across genuinely uncorrelated assets/sectors.
     Avoid pairing automobile + auto-ancillary sectors (they move together)."

    For FX: avoid stacking multiple highly-correlated USD pairs
    (e.g., long EUR/USD + long GBP/USD simultaneously without awareness).

    Args:
        proposed_pair: e.g., "EURUSD"
        proposed_direction: "BUY" or "SELL"
        open_positions: list of dicts with keys {"pair": str, "direction": str}
        correlation_matrix: {pair_a: {pair_b: corr_coef}} — if None, uses
                           simple USD-base heuristic
        threshold: max allowed correlation (default 0.70)

    Returns:
        GuardrailResult with passed=True if portfolio stays diversified.
    """
    if not open_positions:
        return GuardrailResult(
            rule_name="correlation_exposure",
            passed=True,
            reason="No open positions — no correlation risk",
            details={"open_position_count": 0},
        )

    # Build correlation lookup if not provided
    if correlation_matrix is None:
        correlation_matrix = _default_fx_correlation_matrix()

    proposed_pair = proposed_pair.upper()
    violations = []
    max_corr = 0.0

    for pos in open_positions:
        existing_pair = pos.get("pair", "").upper()
        existing_dir = pos.get("direction", "").upper()
        if existing_pair == proposed_pair:
            # Same pair — that's doubling up, definitely flag
            violations.append({
                "existing_pair": existing_pair,
                "existing_direction": existing_dir,
                "correlation": 1.0,
                "issue": "same_pair_overlap",
            })
            max_corr = 1.0
            continue

        corr = _lookup_correlation(
            correlation_matrix, proposed_pair, existing_pair
        )
        max_corr = max(max_corr, abs(corr))

        if abs(corr) > threshold:
            # If same direction + high correlation → overconcentrated
            if existing_dir.upper() == proposed_direction.upper():
                violations.append({
                    "existing_pair": existing_pair,
                    "existing_direction": existing_dir,
                    "correlation": round(corr, 3),
                    "issue": "correlated_same_direction",
                })
            else:
                # Opposite direction on correlated pair = partial hedge (less risky)
                violations.append({
                    "existing_pair": existing_pair,
                    "existing_direction": existing_dir,
                    "correlation": round(corr, 3),
                    "issue": "correlated_opposite_direction (partial hedge)",
                })

    if violations:
        same_dir_count = sum(1 for v in violations if "same_direction" in v["issue"])
        if same_dir_count > 0:
            return GuardrailResult(
                rule_name="correlation_exposure",
                passed=False,
                reason=(
                    f"Overconcentrated exposure — proposed {proposed_pair} {proposed_direction} "
                    f"would stack with {same_dir_count} correlated position(s) "
                    f"(max_corr={max_corr:.2f} > threshold={threshold}). "
                    f"Book Page 136: avoid false diversification."
                ),
                details={
                    "violations": violations,
                    "max_correlation": round(max_corr, 3),
                    "threshold": threshold,
                },
            )

    return GuardrailResult(
        rule_name="correlation_exposure",
        passed=True,
        reason=(
            f"Portfolio diversified — proposed {proposed_pair} "
            f"max_corr={max_corr:.2f} ≤ threshold={threshold}"
        ),
        details={
            "max_correlation": round(max_corr, 3),
            "threshold": threshold,
            "open_position_count": len(open_positions),
        },
    )


def _default_fx_correlation_matrix() -> Dict[str, Dict[str, float]]:
    """Default approximate correlation matrix for major FX pairs.
    Based on typical daily-return correlations (approximate, industry-standard)."""
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "XAUUSD"]
    # Symmetric matrix — values are typical correlations
    corr_data = {
        "EURUSD": {"GBPUSD": 0.85, "USDJPY": -0.40, "USDCHF": -0.85, "AUDUSD": 0.65, "USDCAD": -0.50, "NZDUSD": 0.60, "XAUUSD": 0.40},
        "GBPUSD": {"EURUSD": 0.85, "USDJPY": -0.35, "USDCHF": -0.75, "AUDUSD": 0.60, "USDCAD": -0.40, "NZDUSD": 0.55, "XAUUSD": 0.35},
        "USDJPY": {"EURUSD": -0.40, "GBPUSD": -0.35, "USDCHF": 0.50, "AUDUSD": -0.25, "USDCAD": 0.40, "NZDUSD": -0.20, "XAUUSD": -0.30},
        "USDCHF": {"EURUSD": -0.85, "GBPUSD": -0.75, "USDJPY": 0.50, "AUDUSD": -0.55, "USDCAD": 0.45, "NZDUSD": -0.50, "XAUUSD": -0.35},
        "AUDUSD": {"EURUSD": 0.65, "GBPUSD": 0.60, "USDJPY": -0.25, "USDCHF": -0.55, "USDCAD": -0.50, "NZDUSD": 0.75, "XAUUSD": 0.55},
        "USDCAD": {"EURUSD": -0.50, "GBPUSD": -0.40, "USDJPY": 0.40, "USDCHF": 0.45, "AUDUSD": -0.50, "NZDUSD": -0.40, "XAUUSD": -0.30},
        "NZDUSD": {"EURUSD": 0.60, "GBPUSD": 0.55, "USDJPY": -0.20, "USDCHF": -0.50, "AUDUSD": 0.75, "USDCAD": -0.40, "XAUUSD": 0.45},
        "XAUUSD": {"EURUSD": 0.40, "GBPUSD": 0.35, "USDJPY": -0.30, "USDCHF": -0.35, "AUDUSD": 0.55, "USDCAD": -0.30, "NZDUSD": 0.45},
    }
    return corr_data


def _lookup_correlation(matrix, pair_a, pair_b):
    """Lookup correlation with fallback to 0."""
    try:
        if pair_a in matrix and pair_b in matrix[pair_a]:
            return matrix[pair_a][pair_b]
        if pair_b in matrix and pair_a in matrix[pair_b]:
            return matrix[pair_b][pair_a]
    except Exception as e:
        import logging; logging.warning(f"Suppressed at risk/book_guardrails.py:209: {e}")
        pass
    return 0.0


# ═════════════════════════════════════════════════════════════
# GUARDRAIL 2: Anti-Revenge-Trading (Pages 138-139)
# ═════════════════════════════════════════════════════════════

def check_anti_revenge_trading(
    proposed_lot_size: float,
    normal_lot_size: float,
    consecutive_losses: int,
    loss_streak_threshold: int = DEFAULT_LOSS_STREAK_THRESHOLD,
    escalation_mult: float = DEFAULT_POSITION_ESCALATION_MULT,
) -> GuardrailResult:
    """
    Book Pages 138-139 — "Don't chase losses" guardrail.
    "Never chase losses or over-trade to recover them."

    Pseudocode (from book):
      IF recent_losses_streak >= N AND next_trade_size > normal_size:
          BLOCK trade  # prevent revenge-trading position escalation

    Args:
        proposed_lot_size: lot size of the proposed trade
        normal_lot_size: the trader's normal/risk-based lot size
        consecutive_losses: current consecutive loss count (from circuit_breaker)
        loss_streak_threshold: N (default 3)
        escalation_mult: above this multiple of normal = "escalation" (default 1.25)

    Returns:
        GuardrailResult with passed=False if revenge-trading detected.
    """
    if normal_lot_size <= 0:
        return GuardrailResult(
            rule_name="anti_revenge_trading",
            passed=True,
            reason="No normal lot size reference — skipping",
            details={"proposed_lot": proposed_lot_size},
        )

    is_oversized = proposed_lot_size > normal_lot_size * escalation_mult
    is_loss_streak = consecutive_losses >= loss_streak_threshold

    if is_loss_streak and is_oversized:
        return GuardrailResult(
            rule_name="anti_revenge_trading",
            passed=False,
            reason=(
                f"REVENGE TRADING DETECTED — {consecutive_losses} consecutive losses "
                f"AND proposed lot {proposed_lot_size:.2f} > normal {normal_lot_size:.2f} "
                f"× {escalation_mult} = {normal_lot_size * escalation_mult:.2f}. "
                f"Book Page 138: 'Never chase losses'. BLOCKED."
            ),
            details={
                "consecutive_losses": consecutive_losses,
                "loss_streak_threshold": loss_streak_threshold,
                "proposed_lot": proposed_lot_size,
                "normal_lot": normal_lot_size,
                "escalation_multiple": escalation_mult,
                "oversized": is_oversized,
                "loss_streak": is_loss_streak,
            },
        )

    if is_loss_streak:
        return GuardrailResult(
            rule_name="anti_revenge_trading",
            passed=True,
            reason=(
                f"Loss streak ({consecutive_losses}) active but position NOT oversized "
                f"({proposed_lot_size:.2f} ≤ {normal_lot_size * escalation_mult:.2f}). "
                f"Trade allowed — stay disciplined."
            ),
            details={
                "consecutive_losses": consecutive_losses,
                "proposed_lot": proposed_lot_size,
                "normal_lot": normal_lot_size,
            },
        )

    return GuardrailResult(
        rule_name="anti_revenge_trading",
        passed=True,
        reason=(
            f"No loss streak ({consecutive_losses} < {loss_streak_threshold}). "
            f"Trade allowed."
        ),
        details={
            "consecutive_losses": consecutive_losses,
            "proposed_lot": proposed_lot_size,
            "normal_lot": normal_lot_size,
        },
    )


# ═════════════════════════════════════════════════════════════
# GUARDRAIL 3: Cost-Aware Expected Value (Page 138)
# ═════════════════════════════════════════════════════════════

def check_cost_aware_ev(
    expected_pnl_pips: float,
    pair: str,
    win_probability: float = 0.5,
    sl_pips: float = 20.0,
    tp_pips: float = 40.0,
    spread_pips: Optional[float] = None,
    commission_pips: float = DEFAULT_COMMISSION_PIPS,
    slippage_pips: float = DEFAULT_SLIPPAGE_PIPS,
    min_net_ev_pips: float = DEFAULT_MIN_NET_EV_PIPS,
) -> GuardrailResult:
    """
    Book Page 138 — "Don't ignore fees/commissions" guardrail.
    "Don't ignore fees/commissions (they compound against frequent/small-account traders)."

    Pseudocode (from book):
      net_expected_value = expected_pnl - (spread_cost + commission + slippage_estimate)
      IF net_expected_value <= 0:
          REJECT trade

    Args:
        expected_pnl_pips: pre-cost expected PnL in pips (if None, computed from win_prob + SL/TP)
        pair: e.g., "EURUSD"
        win_probability: 0..1 (default 0.5)
        sl_pips: stop-loss distance in pips
        tp_pips: take-profit distance in pips
        spread_pips: bid-ask spread (auto-looked-up if None)
        commission_pips: round-trip commission in pips
        slippage_pips: estimated slippage in pips
        min_net_ev_pips: minimum net EV to accept trade (default 1 pip)

    Returns:
        GuardrailResult with passed=False if net EV ≤ 0.
    """
    pair = pair.upper()

    # Lookup spread if not provided
    if spread_pips is None:
        spread_pips = DEFAULT_SPREAD_PIPS.get(pair, 2.0)

    # Compute expected PnL if not provided
    if expected_pnl_pips is None:
        expected_pnl_pips = (win_probability * tp_pips) - ((1 - win_probability) * sl_pips)

    # Total transaction costs
    total_costs = spread_pips + commission_pips + slippage_pips
    net_ev = expected_pnl_pips - total_costs

    details = {
        "pair":              pair,
        "expected_pnl_pips": round(expected_pnl_pips, 2),
        "spread_pips":       spread_pips,
        "commission_pips":   commission_pips,
        "slippage_pips":     slippage_pips,
        "total_costs_pips":  round(total_costs, 2),
        "net_ev_pips":       round(net_ev, 2),
        "min_required_ev":   min_net_ev_pips,
        "win_probability":   round(win_probability, 3),
        "sl_pips":           sl_pips,
        "tp_pips":           tp_pips,
    }

    if net_ev <= 0:
        return GuardrailResult(
            rule_name="cost_aware_ev",
            passed=False,
            reason=(
                f"Net EV after costs is {net_ev:.2f} pips (≤ 0). "
                f"Expected {expected_pnl_pips:.2f} - costs {total_costs:.2f} "
                f"(spread={spread_pips}, comm={commission_pips}, slip={slippage_pips}). "
                f"Book Page 138: 'Don't ignore fees'. REJECTED."
            ),
            details=details,
        )

    if net_ev < min_net_ev_pips:
        return GuardrailResult(
            rule_name="cost_aware_ev",
            passed=False,
            reason=(
                f"Net EV {net_ev:.2f} pips is positive but below minimum {min_net_ev_pips:.2f}. "
                f"Trade not worthwhile after costs. REJECTED."
            ),
            details=details,
        )

    return GuardrailResult(
        rule_name="cost_aware_ev",
        passed=True,
        reason=(
            f"Net EV {net_ev:.2f} pips (expected {expected_pnl_pips:.2f} - costs {total_costs:.2f}). "
            f"Trade is profitable after fees."
        ),
        details=details,
    )


# ═════════════════════════════════════════════════════════════
# AGGREGATE: Run All 3 Guardrails
# ═════════════════════════════════════════════════════════════

def run_all_guardrails(
    proposed_pair: str,
    proposed_direction: str,
    proposed_lot_size: float,
    normal_lot_size: float,
    consecutive_losses: int,
    open_positions: List[dict],
    expected_pnl_pips: Optional[float] = None,
    win_probability: float = 0.5,
    sl_pips: float = 20.0,
    tp_pips: float = 40.0,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    loss_streak_threshold: int = DEFAULT_LOSS_STREAK_THRESHOLD,
    min_net_ev_pips: float = DEFAULT_MIN_NET_EV_PIPS,
) -> dict:
    """
    Run all 3 book guardrails + return aggregate result.

    Returns:
        {
            "all_passed": bool,
            "passed_count": int,  # 0..3
            "results": [GuardrailResult.to_dict(), ...],
            "block_reason": str | None,
        }
    """
    results = []

    # 1. Correlation exposure
    results.append(check_correlation_exposure(
        proposed_pair=proposed_pair,
        proposed_direction=proposed_direction,
        open_positions=open_positions,
        threshold=correlation_threshold,
    ))

    # 2. Anti-revenge-trading
    results.append(check_anti_revenge_trading(
        proposed_lot_size=proposed_lot_size,
        normal_lot_size=normal_lot_size,
        consecutive_losses=consecutive_losses,
        loss_streak_threshold=loss_streak_threshold,
    ))

    # 3. Cost-aware EV
    results.append(check_cost_aware_ev(
        expected_pnl_pips=expected_pnl_pips,
        pair=proposed_pair,
        win_probability=win_probability,
        sl_pips=sl_pips,
        tp_pips=tp_pips,
        min_net_ev_pips=min_net_ev_pips,
    ))

    passed_count = sum(1 for r in results if r.passed)
    all_passed = passed_count == len(results)
    block_reason = None if all_passed else next(
        (r.reason for r in results if not r.passed), None
    )

    return {
        "all_passed":    bool(all_passed),
        "passed_count":  int(passed_count),
        "total_count":   len(results),
        "results":       [r.to_dict() for r in results],
        "block_reason":  block_reason,
    }


# ═════════════════════════════════════════════════════════════
# CLI entry
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Book Guardrails Demo ===\n")

    # Scenario 1: Correlated positions (EURUSD long + GBPUSD long)
    print("--- Scenario 1: Correlated EURUSD + GBPUSD (both long) ---")
    r1 = check_correlation_exposure(
        proposed_pair="EURUSD",
        proposed_direction="BUY",
        open_positions=[{"pair": "GBPUSD", "direction": "BUY"}],
    )
    print(f"Passed: {r1.passed}")
    print(f"Reason: {r1.reason}\n")

    # Scenario 2: Revenge trading
    print("--- Scenario 2: Loss streak + oversized lot ---")
    r2 = check_anti_revenge_trading(
        proposed_lot_size=0.50,
        normal_lot_size=0.20,
        consecutive_losses=3,
    )
    print(f"Passed: {r2.passed}")
    print(f"Reason: {r2.reason}\n")

    # Scenario 3: Cost-aware EV
    print("--- Scenario 3: Low expected PnL after costs ---")
    r3 = check_cost_aware_ev(
        expected_pnl_pips=2.0,  # only 2 pips expected
        pair="EURUSD",
    )
    print(f"Passed: {r3.passed}")
    print(f"Reason: {r3.reason}\n")

    # Aggregate
    print("--- Aggregate: All 3 guardrails ---")
    agg = run_all_guardrails(
        proposed_pair="EURUSD",
        proposed_direction="BUY",
        proposed_lot_size=0.50,
        normal_lot_size=0.20,
        consecutive_losses=3,
        open_positions=[{"pair": "GBPUSD", "direction": "BUY"}],
        expected_pnl_pips=2.0,
        pair_alt=None,
    ) if False else run_all_guardrails(
        proposed_pair="EURUSD",
        proposed_direction="BUY",
        proposed_lot_size=0.50,
        normal_lot_size=0.20,
        consecutive_losses=3,
        open_positions=[{"pair": "GBPUSD", "direction": "BUY"}],
        expected_pnl_pips=2.0,
    )
    print(f"All passed: {agg['all_passed']} ({agg['passed_count']}/{agg['total_count']})")
    if agg["block_reason"]:
        print(f"Block reason: {agg['block_reason']}")
