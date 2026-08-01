# analysis/follow_through_engine.py — Follow Through Engine
# ============================================================
# Gap identified against Al Brooks price-action framework: a breakout by
# itself is not a trade signal. What matters is what the market does in
# the bars AFTER the breakout — does it continue (trend bars, higher
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
#
# 2026-07-23 revision — external review feedback incorporated:
#   1. EXPIRED status: running out of the confirmation window without a
#      trap is now distinct from FAILED. FAILED means the market actively
#      closed back through the level (a trap). EXPIRED means it simply
#      never built enough follow-through in time — a weaker, less
#      alarming signal that SignalFusion should probably weight
#      differently than an active trap.
#   2. Five-state status model instead of three: PENDING (early, not
#      enough data), WEAK (some confirmation but below threshold),
#      CONFIRMED, FAILED (trap), EXPIRED (ran out of bars).
#   3. ATR normalization: "strong trend bar" now uses body/ATR when an
#      ATR value is available (per-bar `atr` column, matching
#      data/indicators_ext.py's column naming, or a scalar override) —
#      not just body/range — so a EURUSD candle and an XAUUSD candle are
#      judged on a comparable scale instead of raw range.
#   4. Session awareness: an optional `session` label (or auto-derived
#      from a bar timestamp if the df carries one) applies a session
#      weight to the score, since a London breakout and an Asian-session
#      breakout don't have the same follow-through probability.
#   5. Tick volume: if the df carries a `volume` column (this codebase's
#      convention for MT5 tick_volume — see data/fetcher.py), relative
#      volume on the confirming bars vs. their recent average nudges the
#      score, since a breakout on above-average participation is more
#      credible than one on a quiet tape.
#   6. Confidence curve: each evaluate() call now also returns the
#      running score after each confirming bar processed in that call
#      (not just the final number), so callers can see how confidence
#      built bar-by-bar instead of just the endpoint.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger("follow_through_engine")

VALID_DIRECTIONS = ("BULLISH", "BEARISH")
VALID_STATUSES = ("PENDING", "WEAK", "CONFIRMED", "FAILED", "EXPIRED", "INVALID")

# Session weight defaults. These are deliberately mild multipliers, not
# hard gates — a London breakout isn't guaranteed to work and an Asian
# one isn't guaranteed to fail, this just nudges the score. Tune once
# shadow-mode data shows the real per-session follow-through rate for
# each pair; until then these are a reasonable, literature-informed
# starting point (London/NY overlap = highest liquidity & follow-through,
# Asian session = typically choppier / more range-bound).
DEFAULT_SESSION_WEIGHTS = {
    "ASIAN": 0.90,
    "LONDON": 1.05,
    "NEWYORK": 1.05,
    "LONDON_NY_OVERLAP": 1.15,
}


def _infer_session(ts: datetime) -> str:
    """Classify a UTC timestamp into a trading session bucket.

    Deliberately self-contained (no import from orchestrator/) to keep
    this module standalone per the class docstring. Boundaries are the
    conventional UTC session hours:
        Asian:            00:00–07:00
        London:           07:00–16:00
        London/NY overlap: 12:00–16:00
        New York:         12:00–21:00
        (21:00–00:00 falls back to "ASIAN" as the next session opens)
    """
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc)
    hour = ts.hour
    if 12 <= hour < 16:
        return "LONDON_NY_OVERLAP"
    if 7 <= hour < 12:
        return "LONDON"
    if 16 <= hour < 21:
        return "NEWYORK"
    return "ASIAN"


@dataclass
class FollowThroughResult:
    valid: bool
    direction: str
    breakout_level: float
    breakout_index: int
    bars_since_breakout: int
    status: str  # "PENDING" | "WEAK" | "CONFIRMED" | "FAILED" | "EXPIRED" | "INVALID"
    score: int  # 0-100, session-weighted
    raw_score: int  # 0-100, before session weighting — useful for debugging/audit
    failed_breakout: bool
    expired: bool
    trap_bar_index: Optional[int]
    session: Optional[str]
    atr_normalized: bool  # True if ATR was actually available and used
    confidence_curve: List[int] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def get_ai_context(self) -> dict:
        """Small dict form for feeding into a context/probability engine."""
        return {
            "follow_through_status": self.status,
            "follow_through_score": self.score,
            "follow_through_failed": self.failed_breakout,
            "follow_through_expired": self.expired,
            "follow_through_session": self.session,
            "follow_through_curve": self.confidence_curve,
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

    `df` may optionally carry an `atr` column (matches
    data/indicators_ext.py naming) and/or a `volume` column (matches
    data/fetcher.py's tick-volume-as-volume convention) — both are used
    if present and silently skipped if not, so this keeps working against
    a bare OHLC frame.
    """

    def __init__(
        self,
        confirm_bars: int = 3,
        strong_body_ratio: float = 0.55,
        strong_body_atr_ratio: float = 0.70,
        max_pullback_ratio: float = 0.5,
        confirm_score_threshold: int = 60,
        weak_score_threshold: int = 25,
        volume_lookback: int = 10,
        session_weights: Optional[dict] = None,
    ):
        """
        confirm_bars: how many bars after the breakout to evaluate before
            declaring CONFIRMED/EXPIRED if neither FAILED nor CONFIRMED
            has happened yet.
        strong_body_ratio: candle body / candle range needed to call a bar
            a "trend bar" when no ATR is available (fallback).
        strong_body_atr_ratio: candle body / ATR needed to call a bar a
            "trend bar" when ATR IS available (preferred — comparable
            across pairs/instruments, unlike a raw range ratio).
        max_pullback_ratio: max retracement (as a fraction of the
            breakout bar's range) before a pullback is considered too deep
            to still call this a clean continuation.
        confirm_score_threshold: score (0-100) at/above which status
            becomes CONFIRMED once at least one confirming bar exists.
        weak_score_threshold: score (0-100) at/above which a still-pending
            setup is labeled WEAK instead of PENDING — i.e. "something is
            happening but not enough yet" vs. "nothing meaningful yet".
        volume_lookback: how many bars of preceding volume to average
            when judging whether a confirming bar's volume is "relatively
            high". Ignored if the df has no `volume` column.
        session_weights: override the default session score multipliers
            (see DEFAULT_SESSION_WEIGHTS).
        """
        self.confirm_bars = confirm_bars
        self.strong_body_ratio = strong_body_ratio
        self.strong_body_atr_ratio = strong_body_atr_ratio
        self.max_pullback_ratio = max_pullback_ratio
        self.confirm_score_threshold = confirm_score_threshold
        self.weak_score_threshold = weak_score_threshold
        self.volume_lookback = volume_lookback
        self.session_weights = session_weights or dict(DEFAULT_SESSION_WEIGHTS)

    # ── Public API ──────────────────────────────────────────────

    def evaluate_from_bos(
        self, df: pd.DataFrame, bos: dict, breakout_index: Optional[int] = None,
        session: Optional[str] = None,
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
                              breakout_index=breakout_index, session=session)

    def evaluate(
        self,
        df: pd.DataFrame,
        breakout_level: float,
        direction: str,
        breakout_index: int,
        atr: Optional[float] = None,
        session: Optional[str] = None,
    ) -> FollowThroughResult:
        """
        atr: optional scalar ATR override, used only if the df has no
            per-bar `atr` column. If neither is available, falls back to
            the raw body/range ratio (strong_body_ratio).
        session: optional explicit session label ("ASIAN" | "LONDON" |
            "NEWYORK" | "LONDON_NY_OVERLAP"). If not given, it's inferred
            from the breakout bar's own timestamp (index or a
            "time"/"timestamp" column) when available; otherwise no
            session weighting is applied (weight = 1.0).
        """
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

        resolved_session = session or self._resolve_session(df, breakout_index)
        session_weight = self.session_weights.get(resolved_session, 1.0) if resolved_session else 1.0

        if bars_since < 1:
            # Breakout bar itself hasn't been followed by anything yet —
            # nothing to confirm/deny. Caller should re-evaluate on the
            # next bar close.
            return FollowThroughResult(
                valid=True, direction=direction, breakout_level=breakout_level,
                breakout_index=breakout_index, bars_since_breakout=0,
                status="PENDING", score=0, raw_score=0, failed_breakout=False,
                expired=False, trap_bar_index=None, session=resolved_session,
                atr_normalized=False, confidence_curve=[],
                reasons=["Breakout bar just formed — awaiting next bar(s)"],
            )

        breakout_bar = df.iloc[breakout_index]
        breakout_range = float(breakout_bar["high"] - breakout_bar["low"])
        confirming = df.iloc[breakout_index + 1: last_index + 1]

        has_atr_col = "atr" in df.columns
        atr_normalized = has_atr_col or atr is not None
        has_volume_col = "volume" in df.columns
        avg_volume = None
        if has_volume_col and breakout_index > 0:
            lookback_start = max(0, breakout_index - self.volume_lookback)
            prior_volume = df.iloc[lookback_start:breakout_index]["volume"]
            if len(prior_volume) > 0 and prior_volume.mean() > 0:
                avg_volume = float(prior_volume.mean())

        score = 0
        reasons: List[str] = []
        failed_breakout = False
        trap_bar_index: Optional[int] = None
        deepest_pullback = 0.0
        confidence_curve: List[int] = []

        for offset, (idx, bar) in enumerate(confirming.iterrows()):
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])
            open_ = float(bar["open"])
            body = abs(close - open_)
            rng = max(high - low, 1e-12)

            bar_atr = float(bar["atr"]) if has_atr_col and pd.notna(bar.get("atr")) else atr
            if bar_atr and bar_atr > 0:
                body_ratio = body / bar_atr
                strong_threshold = self.strong_body_atr_ratio
            else:
                body_ratio = body / rng
                strong_threshold = self.strong_body_ratio

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
                    confidence_curve.append(0)
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
                    confidence_curve.append(0)
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

            if is_bull_bar and body_ratio >= strong_threshold:
                score += 10 if offset == 0 else 5
                metric = "ATR" if (bar_atr and bar_atr > 0) else "range"
                reasons.append(
                    f"Bar {offset+1} is a strong trend bar (body/{metric} ratio {body_ratio:.2f})"
                )

            if has_volume_col and avg_volume:
                bar_volume = float(bar.get("volume", 0) or 0)
                if bar_volume >= 1.2 * avg_volume:
                    score += 5
                    reasons.append(
                        f"Bar {offset+1} volume {bar_volume:.0f} is "
                        f">=1.2x the {self.volume_lookback}-bar average "
                        f"({avg_volume:.0f}) — real participation, not a thin tape"
                    )

            confidence_curve.append(max(0, min(100, score)))

            if offset >= self.confirm_bars - 1:
                break

        if not failed_breakout and deepest_pullback > self.max_pullback_ratio:
            score = max(0, score - 20)
            reasons.append(
                f"Pullback retraced {deepest_pullback:.0%} of the breakout bar's "
                f"range (> {self.max_pullback_ratio:.0%} threshold) — weaker continuation"
            )
            if confidence_curve:
                confidence_curve[-1] = max(0, min(100, score))

        raw_score = max(0, min(100, score))

        expired = False
        if failed_breakout:
            status = "FAILED"
            weighted_score = 0
        else:
            weighted_score = max(0, min(100, round(raw_score * session_weight)))
            if weighted_score >= self.confirm_score_threshold:
                status = "CONFIRMED"
            elif bars_since >= self.confirm_bars:
                # Ran out of the confirmation window without an active
                # trap AND without reaching the confirm threshold. This is
                # a weaker, less alarming outcome than FAILED — the market
                # just didn't build enough momentum in time, it didn't
                # actively reverse through the level.
                status = "EXPIRED"
                expired = True
                reasons.append(
                    f"No confirmation reached score threshold ({weighted_score} < "
                    f"{self.confirm_score_threshold}) after {bars_since} bars "
                    f"— window expired without an active trap"
                )
            elif weighted_score >= self.weak_score_threshold:
                status = "WEAK"
            else:
                status = "PENDING"

        if resolved_session and session_weight != 1.0 and not failed_breakout:
            reasons.append(
                f"Session={resolved_session} weight={session_weight:.2f} applied "
                f"(raw score {raw_score} -> {weighted_score})"
            )

        result = FollowThroughResult(
            valid=True, direction=direction, breakout_level=breakout_level,
            breakout_index=breakout_index, bars_since_breakout=bars_since,
            status=status, score=weighted_score, raw_score=raw_score,
            failed_breakout=failed_breakout, expired=expired,
            trap_bar_index=trap_bar_index, session=resolved_session,
            atr_normalized=atr_normalized, confidence_curve=confidence_curve,
            reasons=reasons,
        )
        log.info(
            f"[FollowThrough] dir={direction} level={breakout_level:.5f} "
            f"bars_since={bars_since} status={status} score={weighted_score} "
            f"(raw={raw_score}, session={resolved_session}) curve={confidence_curve}"
        )
        return result

    # ── Internal ────────────────────────────────────────────────

    def _resolve_session(self, df: pd.DataFrame, breakout_index: int) -> Optional[str]:
        """Best-effort session inference from the breakout bar's own
        timestamp. Returns None (no weighting applied) if no timestamp
        is available anywhere — this must never raise or block scoring,
        session-awareness is a nice-to-have refinement, not a dependency.
        """
        try:
            bar = df.iloc[breakout_index]
            ts = None
            if "time" in df.columns:
                ts = bar["time"]
            elif "timestamp" in df.columns:
                ts = bar["timestamp"]
            elif isinstance(df.index, pd.DatetimeIndex):
                ts = df.index[breakout_index]

            if ts is None:
                return None
            if not isinstance(ts, datetime):
                ts = pd.Timestamp(ts).to_pydatetime()
            return _infer_session(ts)
        except Exception as e:
            log.debug(f"[FollowThrough] session inference skipped: {e}")
            return None

    def _invalid(self, reason: str) -> FollowThroughResult:
        return FollowThroughResult(
            valid=False, direction="NONE", breakout_level=0.0, breakout_index=-1,
            bars_since_breakout=0, status="INVALID", score=0, raw_score=0,
            failed_breakout=False, expired=False, trap_bar_index=None,
            session=None, atr_normalized=False, confidence_curve=[],
            reasons=[reason],
        )


_engine_instance: Optional[FollowThroughEngine] = None


def get_follow_through_engine() -> FollowThroughEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = FollowThroughEngine()
    return _engine_instance
