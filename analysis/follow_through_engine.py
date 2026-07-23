# analysis/follow_through_engine.py — Follow Through Engine
# ============================================================
# Gap identified against Al Brooks price-action framework: a breakout by
# itself is not a trade signal. What matters is what the market does in
# the 2-3 bars AFTER the breakout — does it continue (trend bars, higher
# highs/higher closes, small pullback) or does it fail (a strong opposite
# bar, a close back inside the broken level -> trap)?
#
# This engine consumes a breakout/BOS event (e.g. from
# analysis/structure.py::MarketStructureEngine._detect_bos, or any other
# caller that knows a level + direction + the bar index where it broke)
# and scores how well subsequent bars confirm it.
#
# LOOK-AHEAD SAFETY: evaluate() only ever reads df.iloc[breakout_index+1 :]
# up to and including the LAST row of whatever `df` slice the caller
# passes in. It never looks past "now". Call it again as each new bar
# closes to progressively update the score — do not pre-compute it once
# against a full historical df and expect it to reflect what a live bot
# would have known bar-by-bar.
#
# NOT WIRED INTO THE LIVE DECISION PIPELINE YET. Per the audit checklist
# (look-ahead bias, repaint, SignalFusion conflict, backtest-only
# assumptions), new signal sources should run in shadow/logging mode for
# a demo period before being given any weight in SignalFusion/RiskEngine.
# This module is intentionally standalone (no import from agents/ or
# core/) so it can be wired in deliberately, behind a flag, rather than
# by accident.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger("follow_through_engine")

VALID_DIRECTIONS = ("BULLISH", "BEARISH")


@dataclass
class FollowThroughResult:
    valid: bool
    direction: str
    breakout_level: float
    breakout_index: int
    bars_since_breakout: int
    status: str  # "AWAITING_CONFIRMATION" | "CONFIRMED" | "FAILED" | "INVALID"
    score: int  # 0-100
    failed_breakout: bool
    trap_bar_index: Optional[int]
    reasons: List[str] = field(default_factory=list)

    def get_ai_context(self) -> dict:
        """Small dict form for feeding into a context/probability engine."""
        return {
            "follow_through_status": self.status,
            "follow_through_score": self.score,
            "follow_through_failed": self.failed_breakout,
        }


class FollowThroughEngine:
    """
    Usage:
        engine = FollowThroughEngine()
        # `bos` is the dict shape returned by
        # MarketStructureEngine._detect_bos(): {"event", "level", "confidence"}
        result = engine.evaluate_from_bos(df, bos, breakout_index=len(df) - 1)

        # or directly:
        result = engine.evaluate(df, breakout_level=1.0850, direction="BULLISH",
                                  breakout_index=42)
    """

    def __init__(
        self,
        confirm_bars: int = 3,
        strong_body_ratio: float = 0.55,
        max_pullback_ratio: float = 0.5,
        confirm_score_threshold: int = 60,
    ):
        """
        confirm_bars: how many bars after the breakout to evaluate before
            declaring CONFIRMED/FAILED if neither has happened yet.
        strong_body_ratio: candle body / candle range needed to call a bar
            a "trend bar" (matches the book's Trend Bar Detector concept).
        max_pullback_ratio: max retracement (as a fraction of the
            breakout bar's range) before a pullback is considered too deep
            to still call this a clean continuation.
        confirm_score_threshold: score (0-100) at/above which status
            becomes CONFIRMED once at least one confirming bar exists.
        """
        self.confirm_bars = confirm_bars
        self.strong_body_ratio = strong_body_ratio
        self.max_pullback_ratio = max_pullback_ratio
        self.confirm_score_threshold = confirm_score_threshold

    # ── Public API ──────────────────────────────────────────────

    def evaluate_from_bos(
        self, df: pd.DataFrame, bos: dict, breakout_index: Optional[int] = None,
    ) -> FollowThroughResult:
        """Convenience wrapper around evaluate() for
        MarketStructureEngine._detect_bos()-shaped dicts."""
        event = (bos or {}).get("event", "NONE")
        level = (bos or {}).get("level")
        if event in (None, "NONE") or level is None:
            return self._invalid("No breakout event to evaluate")

        direction = "BULLISH" if "BULL" in event.upper() else "BEARISH"
        if breakout_index is None:
            breakout_index = len(df) - 1
        return self.evaluate(df, breakout_level=float(level), direction=direction,
                              breakout_index=breakout_index)

    def evaluate(
        self,
        df: pd.DataFrame,
        breakout_level: float,
        direction: str,
        breakout_index: int,
    ) -> FollowThroughResult:
        direction = direction.upper()
        if direction not in VALID_DIRECTIONS:
            return self._invalid(f"Unknown direction: {direction}")
        if df is None or len(df) == 0:
            return self._invalid("Empty dataframe")
        if not all(c in df.columns for c in ("open", "high", "low", "close")):
            return self._invalid("Missing OHLC columns")
        if breakout_index < 0 or breakout_index >= len(df):
            return self._invalid("breakout_index out of range")

        last_index = len(df) - 1
        bars_since = last_index - breakout_index

        if bars_since < 1:
            # Breakout bar itself hasn't been followed by anything yet —
            # nothing to confirm/deny. Caller should re-evaluate on the
            # next bar close.
            return FollowThroughResult(
                valid=True, direction=direction, breakout_level=breakout_level,
                breakout_index=breakout_index, bars_since_breakout=0,
                status="AWAITING_CONFIRMATION", score=0, failed_breakout=False,
                trap_bar_index=None,
                reasons=["Breakout bar just formed — awaiting next bar(s)"],
            )

        breakout_bar = df.iloc[breakout_index]
        breakout_range = float(breakout_bar["high"] - breakout_bar["low"])
        confirming = df.iloc[breakout_index + 1: last_index + 1]

        score = 0
        reasons: List[str] = []
        failed_breakout = False
        trap_bar_index: Optional[int] = None
        deepest_pullback = 0.0

        for offset, (idx, bar) in enumerate(confirming.iterrows()):
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])
            open_ = float(bar["open"])
            body = abs(close - open_)
            rng = max(high - low, 1e-12)
            body_ratio = body / rng

            if direction == "BULLISH":
                # Trap: closes back below the broken level -> failed breakout
                if close < breakout_level:
                    failed_breakout = True
                    trap_bar_index = idx
                    reasons.append(
                        f"Bar {offset+1} after breakout closed back below "
                        f"the broken level ({close:.5f} < {breakout_level:.5f}) "
                        f"— failed breakout / trap"
                    )
                    break
                prev_close = float(df.iloc[breakout_index + offset]["close"])
                higher_high = high > float(df.iloc[breakout_index + offset]["high"])
                higher_close = close > prev_close
                is_bull_bar = close > open_
                pullback = max(0.0, float(breakout_bar["high"]) - low) / breakout_range \
                    if breakout_range > 0 else 0.0
            else:  # BEARISH
                if close > breakout_level:
                    failed_breakout = True
                    trap_bar_index = idx
                    reasons.append(
                        f"Bar {offset+1} after breakout closed back above "
                        f"the broken level ({close:.5f} > {breakout_level:.5f}) "
                        f"— failed breakout / trap"
                    )
                    break
                prev_close = float(df.iloc[breakout_index + offset]["close"])
                higher_high = low < float(df.iloc[breakout_index + offset]["low"])  # "lower low"
                higher_close = close < prev_close  # "lower close"
                is_bull_bar = close < open_  # bearish bar here
                pullback = max(0.0, high - float(breakout_bar["low"])) / breakout_range \
                    if breakout_range > 0 else 0.0

            deepest_pullback = max(deepest_pullback, pullback)

            if offset == 0:
                if higher_high and higher_close:
                    score += 40
                    reasons.append("1st confirming bar made a new high/low and closed beyond prior close")
                elif higher_close:
                    score += 20
                    reasons.append("1st confirming bar closed favorably but without a fresh extreme")
            elif offset == 1:
                if higher_high and higher_close:
                    score += 30
                    reasons.append("2nd confirming bar continued the move")
                elif higher_close:
                    score += 15
            else:
                if higher_close:
                    score += 5

            if is_bull_bar and body_ratio >= self.strong_body_ratio:
                score += 10 if offset == 0 else 5
                reasons.append(f"Bar {offset+1} is a strong trend bar (body ratio {body_ratio:.2f})")

            if offset >= self.confirm_bars - 1:
                break

        if not failed_breakout and deepest_pullback > self.max_pullback_ratio:
            score = max(0, score - 20)
            reasons.append(
                f"Pullback retraced {deepest_pullback:.0%} of the breakout bar's "
                f"range (> {self.max_pullback_ratio:.0%} threshold) — weaker continuation"
            )

        score = max(0, min(100, score))

        if failed_breakout:
            status = "FAILED"
        elif score >= self.confirm_score_threshold:
            status = "CONFIRMED"
        elif bars_since >= self.confirm_bars:
            status = "FAILED"
            reasons.append(
                f"No confirmation reached score threshold ({score} < "
                f"{self.confirm_score_threshold}) after {bars_since} bars"
            )
        else:
            status = "AWAITING_CONFIRMATION"

        result = FollowThroughResult(
            valid=True, direction=direction, breakout_level=breakout_level,
            breakout_index=breakout_index, bars_since_breakout=bars_since,
            status=status, score=score, failed_breakout=failed_breakout,
            trap_bar_index=trap_bar_index, reasons=reasons,
        )
        log.info(
            f"[FollowThrough] dir={direction} level={breakout_level:.5f} "
            f"bars_since={bars_since} status={status} score={score}"
        )
        return result

    # ── Internal ────────────────────────────────────────────────

    def _invalid(self, reason: str) -> FollowThroughResult:
        return FollowThroughResult(
            valid=False, direction="NONE", breakout_level=0.0, breakout_index=-1,
            bars_since_breakout=0, status="INVALID", score=0, failed_breakout=False,
            trap_bar_index=None, reasons=[reason],
        )


_engine_instance: Optional[FollowThroughEngine] = None


def get_follow_through_engine() -> FollowThroughEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = FollowThroughEngine()
    return _engine_instance