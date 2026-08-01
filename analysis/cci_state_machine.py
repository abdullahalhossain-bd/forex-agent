"""
analysis/cci_state_machine.py — Book 5 (Frank Miller S&D) Chapter 11 CCI Module
================================================================================

Pages 120-125 implement a complete CCI (Commodity Channel Index) state
machine for entry / add-to-position / exit decisions, used as a
CONFLUENCE layer on top of supply/demand zones (NOT as a standalone
signal — Book P125 explicitly warns against standalone CCI use).

  ── ENTRY RULES (Book P120) ────────────────────────────────────
    Long  entry at demand zone : CCI < -100 (oversold extreme)
    Short entry at supply zone : CCI > +100 (overbought extreme)

  ── EXIT RULES (Book P125) ─────────────────────────────────────
    Exit long  : CCI retraces back below +100 (momentum fading)
    Exit short : CCI retraces back above -100

  ── ADD-TO-POSITION RULES (Book P125) ──────────────────────────
    Add long  : CCI > 0  (uptrend context confirmed)
    Add short : CCI < 0  (downtrend context confirmed)

  ── AMBIGUOUS ZONE (Book P125) ─────────────────────────────────
    CCI near zero = uncertain state (correction vs full reversal
    not distinguishable) → no action recommended.

  ── CONFLUENCE STACK (Book P124) ───────────────────────────────
    Fully optimized setup = trend line + supply/demand zone + CCI extreme.
    Even losing trades under full optimization shouldn't cause regret
    (qualitative risk-management principle, P124).

Default CCI parameters: length=20, constant=0.015 (industry standard,
matches `ta.cci()` already used in data/indicators_ext.py).

Usage:
    from analysis.cci_state_machine import CCIStateMachine
    sm = CCIStateMachine()
    signal = sm.evaluate(cci_value=-150, position="long", zone_type="demand")
    # → {"action": "ENTER", "reason": "CCI -150 < -100 at demand zone", ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from utils.logger import get_logger

log = get_logger("cci_sm")


# ════════════════════════════════════════════════════════════════
#  CONSTANTS — Book P120, P125 thresholds
# ════════════════════════════════════════════════════════════════

# Entry thresholds (Book P120)
CCI_LONG_ENTRY_MAX  = -100.0   # CCI < -100 at demand zone = long entry
CCI_SHORT_ENTRY_MIN = +100.0   # CCI > +100 at supply zone = short entry

# Exit thresholds (Book P125) — momentum-fade signals
CCI_LONG_EXIT_BELOW  = +100.0  # CCI retraces below +100 → exit long
CCI_SHORT_EXIT_ABOVE = -100.0  # CCI retraces above -100 → exit short

# Add-to-position thresholds (Book P125)
CCI_ADD_LONG_MIN  = 0.0   # CCI > 0 → may add to long
CCI_ADD_SHORT_MAX = 0.0   # CCI < 0 → may add to short

# Ambiguous zone (Book P125)
CCI_AMBIGUOUS_BAND = 20.0  # |CCI| < 20 = near zero = uncertain

# Standard CCI parameters
CCI_DEFAULT_LENGTH  = 20
CCI_DEFAULT_CONST   = 0.015


# ════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

@dataclass
class CCISignal:
    """Single CCI evaluation result."""
    action: str           # "ENTER" | "EXIT" | "ADD" | "HOLD" | "NO_TRADE"
    direction: str        # "long" | "short" | "neutral"
    cci_value: float
    reason: str
    confluence_score: int = 0   # 0-3 (trend+zone+CCI = 3 = fully optimized)
    ambiguous: bool = False


# ════════════════════════════════════════════════════════════════
#  STATE MACHINE
# ════════════════════════════════════════════════════════════════

class CCIStateMachine:
    """
    Book 5 Chapter 11 — CCI entry/add/exit state machine.

    CCI is a CONFLUENCE layer, not a standalone signal (Book P125).
    All entry signals require a coincident supply/demand zone context.

    State inputs:
      • cci_value   : current CCI reading (typically from ta.cci)
      • zone_type   : "demand" | "supply" | None
      • position    : "long" | "short" | None (current open position)
      • trend_align : bool — does trend agree with proposed direction?
      • at_zone     : bool — is price currently at a zone?

    State outputs:
      CCISignal with action ∈ {ENTER, EXIT, ADD, HOLD, NO_TRADE}
    """

    def __init__(
        self,
        long_entry_max: float = CCI_LONG_ENTRY_MAX,
        short_entry_min: float = CCI_SHORT_ENTRY_MIN,
        long_exit_below: float = CCI_LONG_EXIT_BELOW,
        short_exit_above: float = CCI_SHORT_EXIT_ABOVE,
        ambiguous_band: float = CCI_AMBIGUOUS_BAND,
    ):
        self.long_entry_max = long_entry_max
        self.short_entry_min = short_entry_min
        self.long_exit_below = long_exit_below
        self.short_exit_above = short_exit_above
        self.ambiguous_band = ambiguous_band

    # ══════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════

    def evaluate(
        self,
        cci_value: float,
        zone_type: Optional[str] = None,
        position: Optional[str] = None,
        trend_align: bool = True,
        at_zone: bool = True,
        peak_cci_since_entry: Optional[float] = None,
    ) -> CCISignal:
        """
        Evaluate the CCI state machine.

        Args:
            cci_value   : current CCI reading
            zone_type   : "demand" | "supply" | None (zone price is at)
            position    : "long" | "short" | None (currently open position)
            trend_align : True if trend agrees with proposed direction
            at_zone     : True if price is currently at a zone
            peak_cci_since_entry : the most extreme CCI reading recorded since
                the current position was opened (max value while long, min
                value while short). REQUIRED for correct exit signals — see
                `_check_exit_long` / `_check_exit_short` docstrings for why.
                If omitted, exit-on-retrace is skipped (never fires) and a
                warning is logged, rather than firing incorrectly.

        Returns:
            CCISignal
        """
        # ── Ambiguous zone check (Book P125) ──────────────────
        if abs(cci_value) < self.ambiguous_band:
            return CCISignal(
                action="HOLD",
                direction="neutral",
                cci_value=cci_value,
                reason=f"CCI {cci_value:.0f} near zero — ambiguous (correction vs reversal unclear)",
                ambiguous=True,
            )

        # ── EXIT logic takes priority if a position is open ──
        if position == "long":
            exit_sig = self._check_exit_long(cci_value, peak_cci_since_entry)
            if exit_sig:
                return exit_sig
        elif position == "short":
            exit_sig = self._check_exit_short(cci_value, peak_cci_since_entry)
            if exit_sig:
                return exit_sig

        # ── ADD-to-position logic (Book P125) ────────────────
        if position == "long" and cci_value > CCI_ADD_LONG_MIN:
            return CCISignal(
                action="ADD",
                direction="long",
                cci_value=cci_value,
                reason=f"CCI {cci_value:.0f} > 0 — may add to long (uptrend confirmed)",
                confluence_score=self._score_confluence(trend_align, at_zone, zone_type == "demand"),
            )
        if position == "short" and cci_value < CCI_ADD_SHORT_MAX:
            return CCISignal(
                action="ADD",
                direction="short",
                cci_value=cci_value,
                reason=f"CCI {cci_value:.0f} < 0 — may add to short (downtrend confirmed)",
                confluence_score=self._score_confluence(trend_align, at_zone, zone_type == "supply"),
            )

        # ── ENTRY logic (Book P120) ──────────────────────────
        # Entries only apply when flat. Without this guard, an open long
        # sitting at CCI < -100 in a demand zone would re-emit ENTER on every
        # bar (on top of / instead of ADD), which a caller could easily
        # mistake for a fresh position and double up exposure.
        # Long entry: CCI < -100 at demand zone
        if (position is None and cci_value < self.long_entry_max and
                zone_type == "demand" and at_zone):
            conf = self._score_confluence(trend_align, True, True)
            return CCISignal(
                action="ENTER",
                direction="long",
                cci_value=cci_value,
                reason=f"CCI {cci_value:.0f} < {self.long_entry_max:.0f} at demand zone — long entry",
                confluence_score=conf,
            )

        # Short entry: CCI > +100 at supply zone
        if (position is None and cci_value > self.short_entry_min and
                zone_type == "supply" and at_zone):
            conf = self._score_confluence(trend_align, True, True)
            return CCISignal(
                action="ENTER",
                direction="short",
                cci_value=cci_value,
                reason=f"CCI {cci_value:.0f} > {self.short_entry_min:.0f} at supply zone — short entry",
                confluence_score=conf,
            )

        # ── No actionable signal ─────────────────────────────
        return CCISignal(
            action="HOLD",
            direction="neutral",
            cci_value=cci_value,
            reason=f"CCI {cci_value:.0f} — no entry/add/exit trigger (zone={zone_type}, pos={position})",
        )

    # ══════════════════════════════════════════════════════════
    #  CONFLUENCE SCORING (Book P124 — "fully optimized setup")
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _score_confluence(trend_align: bool, at_zone: bool, cci_extreme: bool) -> int:
        """
        Book P124: fully optimized setup = trend line + zone + CCI extreme.
        Score 0-3 (each component contributes 1 point).
        """
        return int(trend_align) + int(at_zone) + int(cci_extreme)

    # ══════════════════════════════════════════════════════════
    #  EXIT CHECKS (Book P125)
    # ══════════════════════════════════════════════════════════

    def _check_exit_long(
        self, cci_value: float, peak_cci_since_entry: Optional[float]
    ) -> Optional[CCISignal]:
        """
        Book P125: exit long when CCI RETRACES back below +100 — i.e. CCI
        must have previously risen to/above +100 during this trade, and has
        now fallen back under it. This is a crossing condition, not a bare
        "cci_value < 100" check.

        BUG FIXED: the original implementation checked only
        `cci_value < self.long_exit_below` (100). Because long entries fire
        at CCI < -100 (see Book P120), that condition is already true the
        instant a long is opened, so the very next evaluation would exit the
        trade immediately — before any add-to-position logic ever runs. This
        made the module functionally close every long trade on the bar after
        entry regardless of outcome.
        """
        if peak_cci_since_entry is None:
            log.warning(
                "_check_exit_long called without peak_cci_since_entry — "
                "cannot safely evaluate the retrace condition, skipping exit "
                "check. Callers must track the max CCI seen while long and "
                "pass it in, or this signal will never fire."
            )
            return None
        if peak_cci_since_entry >= self.long_exit_below and cci_value < self.long_exit_below:
            return CCISignal(
                action="EXIT",
                direction="long",
                cci_value=cci_value,
                reason=(
                    f"CCI peaked at {peak_cci_since_entry:.0f} (≥ +{self.long_exit_below:.0f}) "
                    f"and has now retraced to {cci_value:.0f} — exit long (momentum fading)"
                ),
            )
        return None

    def _check_exit_short(
        self, cci_value: float, peak_cci_since_entry: Optional[float]
    ) -> Optional[CCISignal]:
        """
        Book P125: exit short when CCI retraces back above -100 — i.e. CCI
        must have previously fallen to/below -100 during this trade, and has
        now risen back above it. See `_check_exit_long` for the bug this
        mirrors and fixes.
        """
        if peak_cci_since_entry is None:
            log.warning(
                "_check_exit_short called without peak_cci_since_entry — "
                "cannot safely evaluate the retrace condition, skipping exit "
                "check. Callers must track the min CCI seen while short and "
                "pass it in, or this signal will never fire."
            )
            return None
        if peak_cci_since_entry <= self.short_exit_above and cci_value > self.short_exit_above:
            return CCISignal(
                action="EXIT",
                direction="short",
                cci_value=cci_value,
                reason=(
                    f"CCI bottomed at {peak_cci_since_entry:.0f} (≤ {self.short_exit_above:.0f}) "
                    f"and has now retraced to {cci_value:.0f} — exit short (momentum fading)"
                ),
            )
        return None

    # ══════════════════════════════════════════════════════════
    #  ZONE-FAILURE DIAGNOSTIC (Book P122-123)
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def diagnose_zone_failure(
        test_count: int,
        trend_line_broken: bool,
        departure_strength: str,
        cci_at_retest: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Book P122-123: retrospective failure-analysis method.
        The same enhancer criteria that score a zone pre-entry can
        also explain why a zone failed post-entry.

        Args:
            test_count          : number of times the zone was retested
            trend_line_broken   : did the relevant trend line break?
            departure_strength  : "strong" | "moderate" | "weak"
            cci_at_retest       : CCI value at the retest (optional)

        Returns:
            {failure_reasons: [...], severity: "high"|"medium"|"low"}
        """
        reasons = []
        if test_count >= 2:
            reasons.append(f"Not fresh — tested {test_count} times (≥2 = stale, Book P68)")
        if trend_line_broken:
            reasons.append("Trend line broke — primary trend threatened (Book P122)")
        if departure_strength == "weak":
            reasons.append("Weak departure — indecision-dominated (Book P65)")
        if cci_at_retest is not None:
            if cci_at_retest > 0 and departure_strength == "weak":
                reasons.append(f"CCI {cci_at_retest:.0f} > 0 with weak departure — "
                               f"opposing zone likely forming (Book P122)")

        severity = "high" if len(reasons) >= 3 else "medium" if len(reasons) >= 2 else "low"
        return {"failure_reasons": reasons, "severity": severity, "reason_count": len(reasons)}


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY (smoke test)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("=" * 64)
    print("  CCI STATE MACHINE — Book 5 Chapter 11 (Pages 120-125)")
    print("=" * 64)

    sm = CCIStateMachine()

    # Case 1: Long entry at demand zone, CCI = -150
    sig = sm.evaluate(cci_value=-150, zone_type="demand", position=None)
    print(f"\n[1] CCI=-150 at demand zone, no position:")
    print(f"    action={sig.action} dir={sig.direction} conf={sig.confluence_score}/3")
    print(f"    {sig.reason}")

    # Case 2: Short entry at supply zone, CCI = +200
    sig = sm.evaluate(cci_value=200, zone_type="supply", position=None)
    print(f"\n[2] CCI=+200 at supply zone, no position:")
    print(f"    action={sig.action} dir={sig.direction} conf={sig.confluence_score}/3")
    print(f"    {sig.reason}")

    # Case 3a: REGRESSION CHECK for the fixed bug — a long entered at
    # CCI=-150 must NOT be exited on the very next bar just because
    # -140 < +100. Without peak tracking supplied, the exit check must
    # no-op (not silently exit the trade).
    sig = sm.evaluate(cci_value=-140, zone_type="demand", position="long")
    print(f"\n[3a] Regression check — CCI=-140 the bar after entry, long open, "
          f"no peak tracked:")
    print(f"    action={sig.action} dir={sig.direction}")
    print(f"    {sig.reason}")
    assert sig.action != "EXIT", "REGRESSION: long exited immediately after entry"
    assert sig.action != "ENTER", "REGRESSION: re-entered a position that's already open"

    # Case 3b: Exit long — CCI genuinely ran to +250 then retraced to +50.
    # This is what "exit on retrace" is supposed to detect.
    sig = sm.evaluate(cci_value=50, zone_type="demand", position="long",
                      peak_cci_since_entry=250)
    print(f"\n[3b] CCI peaked +250, now +50, long position open:")
    print(f"    action={sig.action} dir={sig.direction}")
    print(f"    {sig.reason}")
    assert sig.action == "EXIT"

    # Case 4: Add to long — CCI > 0, hasn't retraced from any peak yet
    sig = sm.evaluate(cci_value=80, zone_type="demand", position="long",
                      peak_cci_since_entry=80)
    print(f"\n[4] CCI=+80, long position open:")
    print(f"    action={sig.action} dir={sig.direction} conf={sig.confluence_score}/3")
    print(f"    {sig.reason}")

    # Case 5: Ambiguous — CCI near zero
    sig = sm.evaluate(cci_value=10, zone_type="demand", position=None)
    print(f"\n[5] CCI=+10, no position:")
    print(f"    action={sig.action} ambiguous={sig.ambiguous}")
    print(f"    {sig.reason}")

    # Case 6: Zone failure diagnosis (Book P122)
    print(f"\n[6] Zone failure diagnosis (Book P122):")
    diag = CCIStateMachine.diagnose_zone_failure(
        test_count=2,
        trend_line_broken=True,
        departure_strength="weak",
        cci_at_retest=50,
    )
    print(f"    severity: {diag['severity']} ({diag['reason_count']} reasons)")
    for r in diag["failure_reasons"]:
        print(f"    - {r}")

    print("\n" + "=" * 64)
    print("  CCI state machine smoke test complete.")
    print("=" * 64)