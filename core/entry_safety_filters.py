"""
core/entry_safety_filters.py — Entry Safety Filter Layer
============================================================
Hardening layer added after a code-review audit flagged six gaps in
the decision pipeline. Priority 1 (Critical severity) was this bug:

    SignalFusion consensus = WAIT
              |
    Confidence Override
              |
             BUY

A single RuleEngine signal was able to override a WAIT verdict that
the 4-layer consensus (SignalFusion / MasterDecision) had already
reached. That is now fixed via `consensus_lock()` /
`evaluate_override_gate()` below: a lone RuleEngine signal can never
force a trade out of consensus WAIT. It may only do so when several
independent, strict conditions are satisfied simultaneously — HTF
alignment, confirmed breakout, liquidity safety, and risk approval.

The six rules implemented here:

1. Consensus Lock            — see above.
2. Resistance Distance Filter — block BUY/SELL entries that sit
                                 10-20 pips from the nearest opposing
                                 S/R level (the "about to get rejected"
                                 zone).
3. Breakout Confirmation      — a breakout only counts once the last
   Filter                       CLOSED candle has cleared the level,
                                 never an in-progress candle.
4. Liquidity Sweep Detector   — reject entries that are either riding
                                 a reversal-implying sweep against
                                 their own direction, or running
                                 straight at an untouched liquidity
                                 pool (classic stop-hunt setup).
5. Trend Exhaustion Filter    — a long run of same-direction candles
                                 is scored as exhaustion risk and
                                 penalizes the entry score instead of
                                 being read as extra confirmation.
6. Confidence Calibration     — usable/displayed confidence is hard-
                                 capped below 100% (default ceiling
                                 96%). Real markets always carry
                                 uncertainty; letting confidence hit
                                 100% makes the system overconfident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas should always be present
    pd = None

from utils.logger import get_logger

log = get_logger("entry_safety_filters")

# ── Tunables ──────────────────────────────────────────────────────
RESISTANCE_BLOCK_MIN_PIPS = 10.0
RESISTANCE_BLOCK_MAX_PIPS = 20.0
TREND_EXHAUSTION_LOOKBACK = 6
TREND_EXHAUSTION_MIN_RUN = 5          # 5+ consecutive same-direction candles
TREND_EXHAUSTION_PENALTY = 15.0       # points shaved off entry score
CONFIDENCE_CEILING = 96.0             # never let usable confidence hit 100%
RULE_OVERRIDE_MIN_CONFIDENCE = 60.0   # stricter than the old 30% floor it replaces

# TP-side resistance (2nd trade-audit fix): TP sitting right under an
# opposing level is a "won't get there" setup, not just an entry issue.
TP_RESISTANCE_WARN_PIPS = 20.0
TP_RESISTANCE_PENALTY = 6.0

# Pullback confirmation (2nd trade-audit fix): don't buy mid-pullback.
PULLBACK_LOOKBACK = 8

# Breakout age / ATR extension (RuleEngine calibration, 2nd audit fix)
BREAKOUT_STALE_CANDLES = 8            # breakout older than this = chasing
BREAKOUT_AGE_PENALTY = 5.0
ATR_EXTENSION_RATIO_WARN = 2.5        # candle range vs ATR — overextended
ATR_EXTENSION_PENALTY = 6.0

# Signal-strength downgrade — how many risk flags trigger STRONG_BUY -> BUY
STRONG_SIGNAL_DOWNGRADE_MIN_FLAGS = 1

# LLM disagreement — keyword-weighted penalty instead of a flat number
LLM_DISAGREEMENT_BASE_PENALTY = 5.0
LLM_DISAGREEMENT_KEYWORD_PENALTY = 2.0
LLM_DISAGREEMENT_KEYWORDS = {
    "resistance": ["resistance", "supply", "overhead"],
    "momentum": ["momentum", "weak momentum", "slowing"],
    "liquidity": ["liquidity", "sweep", "stop hunt", "stop-hunt"],
    "exhaustion": ["exhaustion", "exhausted", "overextended", "extended"],
}


@dataclass
class SafetyCheckResult:
    allowed: bool
    reason: str = ""
    penalty: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class EntrySafetyFilters:
    """Stateless checks. Use `evaluate_override_gate()` for the
    specific "should a lone RuleEngine signal be allowed to override a
    consensus WAIT" decision, or call the individual rules directly
    wherever else they're needed in the pipeline."""

    # ── Rule 1: Consensus Lock ──────────────────────────────────────
    @staticmethod
    def consensus_lock(
        master_signal: str,
        rule_signal: str,
        rule_confidence: float,
        htf_aligned: bool,
        breakout_confirmed: bool,
        liquidity_safe: bool,
        risk_approved: bool,
    ) -> SafetyCheckResult:
        """
        If the multi-layer consensus (SignalFusion / MasterDecision)
        says WAIT, a single RuleEngine signal is NOT allowed to force a
        trade on its own. It may only override consensus when ALL of
        the following gates fire at once:
          - HTF bias agrees with the rule signal's direction
          - The breakout level has been confirmed by a CLOSED candle
          - No unresolved liquidity-sweep risk sits in the way
          - Risk manager has separately approved the trade
        Failing ANY of these keeps the WAIT.
        """
        if master_signal != "WAIT":
            return SafetyCheckResult(allowed=True, reason="Consensus is not WAIT — lock not applicable")

        if rule_signal not in ("BUY", "SELL"):
            return SafetyCheckResult(allowed=False, reason="No directional rule signal")

        if rule_confidence < RULE_OVERRIDE_MIN_CONFIDENCE:
            return SafetyCheckResult(
                allowed=False,
                reason=(
                    f"Consensus WAIT — rule confidence {rule_confidence:.0f}% below "
                    f"{RULE_OVERRIDE_MIN_CONFIDENCE:.0f}% override floor"
                ),
            )

        gates = {
            "htf_aligned": htf_aligned,
            "breakout_confirmed": breakout_confirmed,
            "liquidity_safe": liquidity_safe,
            "risk_approved": risk_approved,
        }
        failed = [name for name, ok in gates.items() if not ok]
        if failed:
            return SafetyCheckResult(
                allowed=False,
                reason=f"Consensus WAIT — RuleEngine cannot override alone; failed gates: {failed}",
                details=gates,
            )

        return SafetyCheckResult(
            allowed=True,
            reason=(
                "Consensus WAIT overridden — all strict gates satisfied "
                "(HTF aligned, breakout confirmed, liquidity safe, risk approved)"
            ),
            details=gates,
        )

    # ── Rule 2: Resistance / Support Distance Filter ────────────────
    @staticmethod
    def distance_filter(
        direction: str,
        dist_to_resistance_pips: Optional[float],
        dist_to_support_pips: Optional[float],
    ) -> SafetyCheckResult:
        if direction == "BUY" and dist_to_resistance_pips is not None:
            if RESISTANCE_BLOCK_MIN_PIPS <= dist_to_resistance_pips <= RESISTANCE_BLOCK_MAX_PIPS:
                return SafetyCheckResult(
                    allowed=False,
                    reason=(
                        f"BUY blocked — resistance only {dist_to_resistance_pips:.1f} pips away "
                        f"({RESISTANCE_BLOCK_MIN_PIPS:.0f}-{RESISTANCE_BLOCK_MAX_PIPS:.0f} pip danger zone)"
                    ),
                )
        if direction == "SELL" and dist_to_support_pips is not None:
            if RESISTANCE_BLOCK_MIN_PIPS <= dist_to_support_pips <= RESISTANCE_BLOCK_MAX_PIPS:
                return SafetyCheckResult(
                    allowed=False,
                    reason=(
                        f"SELL blocked — support only {dist_to_support_pips:.1f} pips away "
                        f"({RESISTANCE_BLOCK_MIN_PIPS:.0f}-{RESISTANCE_BLOCK_MAX_PIPS:.0f} pip danger zone)"
                    ),
                )
        return SafetyCheckResult(allowed=True, reason="Clear of near-term S/R danger zone")

    # ── Rule 3: Breakout Confirmation Filter ─────────────────────────
    @staticmethod
    def breakout_confirmation(df, level: Optional[float], direction: str) -> SafetyCheckResult:
        """
        A breakout only "counts" once the last FULLY CLOSED candle has
        closed beyond the level — never a still-forming/live candle.
        Callers must pass a `df` whose last row is the most recent
        CLOSED candle (never an in-progress bar).
        """
        if level is None or df is None or len(df) < 2:
            return SafetyCheckResult(allowed=True, reason="No breakout level to confirm")

        try:
            last_close = float(df["close"].iloc[-1])
        except Exception:
            return SafetyCheckResult(allowed=True, reason="No close data available")

        if direction == "BUY":
            confirmed = last_close > level
        elif direction == "SELL":
            confirmed = last_close < level
        else:
            return SafetyCheckResult(allowed=True, reason="No directional breakout to confirm")

        if not confirmed:
            return SafetyCheckResult(
                allowed=False,
                reason=(
                    f"Breakout not confirmed — last closed candle ({last_close}) "
                    f"has not cleared level {level}"
                ),
            )
        return SafetyCheckResult(allowed=True, reason="Breakout confirmed by closed candle")

    # ── Rule 4: Liquidity Sweep Detector ─────────────────────────────
    @staticmethod
    def liquidity_sweep_filter(direction: str, liquidity_ctx: Optional[Dict[str, Any]]) -> SafetyCheckResult:
        """
        Rejects the trade when price is riding a sweep that implies a
        reversal against our direction, or when it's running straight
        at an untouched multi-touch liquidity pool — both are classic
        false-breakout / stop-hunt setups.
        """
        liquidity_ctx = liquidity_ctx or {}
        if not liquidity_ctx.get("liquidity_valid"):
            return SafetyCheckResult(allowed=True, reason="No liquidity context available")

        sweep_kind = liquidity_ctx.get("recent_sweep_kind")
        implication = str(liquidity_ctx.get("recent_sweep_implication") or "").upper()

        if sweep_kind and implication and "REVERSAL" in implication:
            adverse = (
                (direction == "BUY" and sweep_kind == "high") or
                (direction == "SELL" and sweep_kind == "low")
            )
            if adverse:
                return SafetyCheckResult(
                    allowed=False,
                    reason=(
                        f"Liquidity sweep risk — recent {sweep_kind} sweep implies reversal, "
                        f"false-breakout probability elevated"
                    ),
                )

        pool_ahead = (
            liquidity_ctx.get("liquidity_above") if direction == "BUY"
            else liquidity_ctx.get("liquidity_below")
        )
        touches = liquidity_ctx.get(
            "liquidity_above_touches" if direction == "BUY" else "liquidity_below_touches", 0
        ) or 0
        if pool_ahead is not None and touches >= 3 and not sweep_kind:
            return SafetyCheckResult(
                allowed=False,
                reason=(
                    f"Liquidity sweep risk — untouched {touches}-touch pool sits directly ahead, "
                    f"high false-breakout probability"
                ),
            )

        return SafetyCheckResult(allowed=True, reason="No adverse liquidity sweep risk detected")

    # ── Rule 5: Trend Exhaustion Filter ──────────────────────────────
    @staticmethod
    def trend_exhaustion(df, direction: str, lookback: int = TREND_EXHAUSTION_LOOKBACK) -> SafetyCheckResult:
        """
        Counts the current run of same-direction candles. A long run
        (>= TREND_EXHAUSTION_MIN_RUN) in the trade's own direction is
        scored as exhaustion risk — it shaves points off the entry
        score rather than being read as extra confirmation.
        """
        if df is None or len(df) < 2 or direction not in ("BUY", "SELL"):
            return SafetyCheckResult(allowed=True, reason="Insufficient data for exhaustion check")

        try:
            closes = df["close"].values
            opens = df["open"].values
        except Exception:
            return SafetyCheckResult(allowed=True, reason="No OHLC data available")

        n = min(lookback, len(df))
        run = 0
        for i in range(1, n + 1):
            idx = -i
            bullish = closes[idx] > opens[idx]
            if direction == "BUY" and bullish:
                run += 1
            elif direction == "SELL" and not bullish:
                run += 1
            else:
                break

        if run >= TREND_EXHAUSTION_MIN_RUN:
            return SafetyCheckResult(
                allowed=True,
                reason=(
                    f"Trend exhaustion — {run} consecutive {direction.lower()}-side candles, "
                    f"entry score penalized rather than boosted"
                ),
                penalty=TREND_EXHAUSTION_PENALTY,
                details={"consecutive_run": run},
            )
        return SafetyCheckResult(
            allowed=True,
            reason=f"No exhaustion detected ({run}-candle run)",
            details={"consecutive_run": run},
        )

    # ── Rule 6: Confidence Calibration ───────────────────────────────
    @staticmethod
    def calibrate_confidence(confidence: float, ceiling: float = CONFIDENCE_CEILING) -> float:
        """
        Never let usable/displayed confidence reach 100%. Real markets
        always carry uncertainty — a system that reaches 100%
        confidence becomes overconfident and over-trades. Hard-caps at
        `ceiling` (default 96%) and floors at 0%.
        """
        try:
            c = float(confidence)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(c, ceiling))

    # ── Convenience: full override-gate chain for rules 1-5 ──────────
    @classmethod
    def evaluate_override_gate(
        cls,
        *,
        master_signal: str,
        rule_signal: str,
        rule_confidence: float,
        mtf_bias: Optional[Dict[str, Any]] = None,
        sr_ctx: Optional[Dict[str, Any]] = None,
        liquidity_ctx: Optional[Dict[str, Any]] = None,
        breakout_level: Optional[float] = None,
        df=None,
        risk_approved: bool = True,
    ) -> SafetyCheckResult:
        """
        One-call helper wiring rules 1-5 together for the specific
        "RuleEngine wants to override a consensus WAIT" decision point.
        Returns a single SafetyCheckResult; `.allowed` tells the caller
        whether the override may proceed, `.details` carries the
        individual gate reasons + the trend-exhaustion penalty for
        downstream confidence adjustment.
        """
        sr_ctx = sr_ctx or {}
        mtf_bias = mtf_bias or {}

        direction = rule_signal if rule_signal in ("BUY", "SELL") else None
        if direction is None:
            return SafetyCheckResult(allowed=False, reason="No directional rule signal to evaluate")

        # HTF alignment
        htf_bias_dir = str(mtf_bias.get("bias", "")).upper()
        htf_aligned = (
            (direction == "BUY" and "BULL" in htf_bias_dir) or
            (direction == "SELL" and "BEAR" in htf_bias_dir)
        )

        dist_check = cls.distance_filter(
            direction,
            sr_ctx.get("dist_to_resistance_pips"),
            sr_ctx.get("dist_to_support_pips"),
        )

        level = breakout_level
        if level is None:
            level = sr_ctx.get("nearest_resistance") if direction == "BUY" else sr_ctx.get("nearest_support")
        breakout_check = cls.breakout_confirmation(df, level, direction)

        liquidity_check = cls.liquidity_sweep_filter(direction, liquidity_ctx)

        exhaustion_check = cls.trend_exhaustion(df, direction)

        gate = cls.consensus_lock(
            master_signal=master_signal,
            rule_signal=rule_signal,
            rule_confidence=rule_confidence,
            htf_aligned=htf_aligned,
            breakout_confirmed=breakout_check.allowed,
            liquidity_safe=liquidity_check.allowed,
            risk_approved=risk_approved,
        )

        # Distance + liquidity are hard blocks in their own right, even
        # if the four consensus_lock gates all pass.
        if gate.allowed and not dist_check.allowed:
            gate = SafetyCheckResult(allowed=False, reason=dist_check.reason)
        if gate.allowed and not liquidity_check.allowed:
            gate = SafetyCheckResult(allowed=False, reason=liquidity_check.reason)

        gate.details = {
            "htf_aligned": htf_aligned,
            "distance_check": dist_check.reason,
            "breakout_check": breakout_check.reason,
            "liquidity_check": liquidity_check.reason,
            "exhaustion_check": exhaustion_check.reason,
            "exhaustion_penalty": exhaustion_check.penalty,
        }
        return gate


# ── Module-level singleton-style convenience access ──────────────────
_FILTERS: Optional[EntrySafetyFilters] = None


def get_entry_safety_filters() -> EntrySafetyFilters:
    global _FILTERS
    if _FILTERS is None:
        _FILTERS = EntrySafetyFilters()
    return _FILTERS