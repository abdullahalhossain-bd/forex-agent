"""
core/signal_scorer.py — Cumulative Score-Based Decision (Day 81+)
====================================================================

WHY THIS EXISTS:
    The current pipeline requires EVERY layer to say YES for a trade to
    pass. If SMC says BUY but LLM says WAIT, the trade dies. With 9+
    layers, the probability that ALL of them align simultaneously is
    very low — that's why 24h runs produce 0 trades.

    Score-based decision is different: each layer CONTRIBUTES points
    to a cumulative score. If the score crosses the threshold, we trade
    — even if 1-2 layers were neutral.

SCORING RUBRIC (tunable via env):

    Layer                     Points  Condition
    ─────────────────────────────────────────────────────────
    smc_signal                +20     BUY or SELL signal from SMC
    liquidity_detected        +15     Liquidity sweep / OB detected
    ml_confidence             +20     ML ensemble conf >= 70%
                                      (+10 if 50-70%, 0 if <50%)
    llm_signal                +15     LLM agrees with direction
    trend_confirmation        +15     MTF bias aligns with signal
    session_quality           +10     HIGH/MEDIUM session
    news_clear                +10     No high-impact news in window
    risk_approved             +10     RiskEngine approved (lot > 0)
    ─────────────────────────────────────────────────────────
    MAX POSSIBLE             115

TRADE THRESHOLDS (per TRADING_MODE):

    TEST        30  (very permissive — verify MT5 works)
    AUTONOMOUS  60  (balanced — production)
    SAFE        80  (high conviction only)

ADAPTIVE THRESHOLD (Day 85):
    If no trades for 6 hours → threshold drops by 10
    If no trades for 12 hours → threshold drops by another 5
    If 3+ trades in last hour → threshold rises by 10 (don't over-trade)

    This way the system self-tunes: during quiet markets it gets more
    permissive, during active markets it gets stricter.

⚠️  STATUS (institutional review, see audit report): NOT CURRENTLY WIRED
    IN. Despite the problem statement above being the exact reason this
    module exists, nothing in trader.py's run_cycle() (or anywhere else
    in the uploaded codebase) calls SignalScorer. The "0 trades in 24h"
    failure mode this file was built to fix may therefore still be
    present in whatever decision logic actually runs (core/agents/
    decision_agent.py, not part of this review). Activating this
    requires replacing or augmenting that decision logic's all-agree
    requirement with SignalScorer.decide() — not done here because
    decision_agent.py wasn't provided for review, and swapping the core
    decision rule blind, without being able to test it against the real
    dependency, is exactly the kind of change that should be verified in
    shadow mode first rather than pushed straight to trader.py.

USAGE:
    from core.signal_scorer import SignalScorer

    scorer = SignalScorer()
    scorer.add("smc",        20, "BOS confirmed")
    scorer.add("liquidity",  15, "Sweep at 1.0820")
    scorer.add("ml",         20, "72% confidence")
    scorer.add("llm",        0,  "WAIT — neutral")  # no contribution
    scorer.add("trend",      15, "MTF bullish")
    scorer.add("session",    10, "London HIGH")
    scorer.add("news",       10, "clear")
    scorer.add("risk",       10, "approved lot=0.01")

    decision = scorer.decide(direction="BUY", pair="EURUSD")
    # → {signal: "BUY", score: 100, threshold: 60, trade: True}
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

log = get_logger("signal_scorer")


# ── Score components ──────────────────────────────────────────

@dataclass
class ScoreComponent:
    layer: str
    points: int       # contribution to the score (can be 0 or negative)
    max_possible: int  # maximum this layer could have contributed
    detail: str = ""


# ── Adaptive threshold tracker ────────────────────────────────

class AdaptiveThreshold:
    """
    Tracks recent trade timestamps and adjusts the threshold up/down
    based on how active the market has been.

    Rules:
      - No trades in last 6h  →  threshold -= 10
      - No trades in last 12h →  threshold -= 15 (cumulative)
      - 3+ trades in last 1h  →  threshold += 10 (avoid over-trading)
      - Else                  →  threshold = base
    """

    def __init__(self, base: int):
        self.base = base
        self._trade_times: List[float] = []

    def record_trade(self, ts: float = None) -> None:
        self._trade_times.append(ts or time.time())
        # Keep only last 24h of trades
        cutoff = time.time() - 86400
        self._trade_times = [t for t in self._trade_times if t >= cutoff]

    def current(self) -> int:
        now = time.time()
        # Trades in last 1h
        trades_1h = sum(1 for t in self._trade_times if t >= now - 3600)
        # Hours since last trade
        if self._trade_times:
            hours_since = (now - max(self._trade_times)) / 3600
        else:
            hours_since = 999  # no trades ever

        adj = 0
        if hours_since >= 12:
            adj = -15
        elif hours_since >= 6:
            adj = -10
        if trades_1h >= 3:
            adj += 10  # over-trading guard stacks on top

        adjusted = max(20, self.base + adj)  # never go below 20
        if adj != 0:
            log.info(
                f"[AdaptiveThreshold] base={self.base} adj={adj:+d} "
                f"→ threshold={adjusted} (hours_since_trade={hours_since:.1f}, "
                f"trades_1h={trades_1h})"
            )
        return adjusted


# ── Signal scorer ─────────────────────────────────────────────

class SignalScorer:
    """
    Accumulates scores from all pipeline layers and decides whether
    to trade based on a (possibly adaptive) threshold.
    """

    # Default max-points per layer (used for reporting % of max)
    #
    # BUG FIX (2026-08-05): this dict used to have keys "smc" and "trend"
    # that NOTHING ever calls — the real caller, core/orphan_consumers.py
    # apply_signal_scoring(), calls scorer.add() with the layer names
    # "rule_pa", "ml", "llm", "mtf", "strength", "liquidity". Because
    # add() fell back to `self.LAYER_MAX.get(layer, points)`, any of
    # those unregistered names (rule_pa, mtf, strength) got their "max"
    # silently set equal to whatever points they scored *that cycle* —
    # which (a) made a perfect layer always look like 100% coverage
    # (max == its own score) and (b) turned negative contrarian scores
    # (e.g. mtf contradicting direction, -10) into a NEGATIVE max_possible,
    # corrupting the score/threshold math for the whole decision. On top
    # of that, "session"/"news"/"risk" (30 pts of budget) were NEVER
    # populated by the real caller at all, so the achievable ceiling was
    # nowhere near what the 60-65 threshold assumed — meaning almost
    # every signal was mathematically unable to clear the bar regardless
    # of true quality. This is the reason execution.log showed 53/53
    # NO_TRADE even at 95-99% aggregate confidence.
    #
    # Caps below now match the actual layer names AND the actual point
    # ranges computed in orphan_consumers.apply_signal_scoring():
    #   rule_pa   : up to 30 (patterns) + 10 (STRONG bonus) = 40
    #   ml        : up to 25 (min(25, ml_conf * 0.25))
    #   llm       : up to 30 (min(30, llm_conf * 0.30)) — was capped at 15,
    #               silently halving the LLM layer's real weight
    #   mtf       : ±20 (matches direction / contradicts)
    #   strength  : ±10
    #   liquidity : up to 20 (A+ grade) — was capped at 15, clipping the
    #               engine's own top grade
    #   session/news/risk kept for any caller that still uses the older
    #   layer names (e.g. the SignalScorer docstring example at the top
    #   of this file).
    LAYER_MAX = {
        "rule_pa":   40,
        "ml":        25,
        "llm":       30,
        "mtf":       20,
        "strength":  10,
        "liquidity": 20,
        "smc":       20,
        "trend":     15,
        "session":   10,
        "news":      10,
        "risk":      10,
    }

    # Base threshold per TRADING_MODE
    BASE_THRESHOLDS = {
        "TEST":       30,
        "AUTONOMOUS": 60,
        "SAFE":       80,
    }

    def __init__(self):
        self._components: List[ScoreComponent] = []
        self._threshold: Optional[AdaptiveThreshold] = None
        self._init_threshold()

    def _init_threshold(self) -> None:
        try:
            from config import TRADING_MODE, TEST_MODE
            mode = "TEST" if TEST_MODE else TRADING_MODE
        except Exception as e:
            mode = "AUTONOMOUS"
        base = self.BASE_THRESHOLDS.get(mode, 60)
        self._threshold = AdaptiveThreshold(base)
        log.info(f"[SignalScorer] base threshold = {base} (mode={mode})")

    # ── Recording ──────────────────────────────────────────────

    def add(self, layer: str, points: int, detail: str = "") -> None:
        """Record one layer's contribution.

        BUG FIX (2026-08-05): previously `max_p = self.LAYER_MAX.get(layer,
        points)` used the passed-in `points` as the fallback max whenever
        `layer` wasn't a registered key. That silently broke max_possible()
        for any unregistered layer (see LAYER_MAX comment above) and could
        even make max_possible negative when points was negative. Now an
        unregistered layer is loud (logged) and contributes 0 rather than
        corrupting the score/threshold math, and negative (contradiction)
        points are floored at -max_p instead of 0 so a real contradiction
        still counts as a penalty instead of being clamped away.
        """
        if layer not in self.LAYER_MAX:
            log.warning(
                f"[SignalScorer] add() called with unregistered layer "
                f"'{layer}' (points={points}) — no LAYER_MAX entry exists, "
                f"so this layer contributes 0 instead of corrupting "
                f"max_possible(). Register '{layer}' in LAYER_MAX."
            )
            self._components.append(ScoreComponent(
                layer=layer, points=0, max_possible=0, detail=detail,
            ))
            return
        max_p = self.LAYER_MAX[layer]
        # Clamp contribution to [-max_p, max_p] — positive contributions
        # capped at the layer's ceiling, negative (contrarian) contributions
        # floored symmetrically so one bad layer can't swing the score more
        # than a perfect layer could swing it the other way.
        points = max(-max_p, min(points, max_p))
        self._components.append(ScoreComponent(
            layer=layer, points=points, max_possible=max_p, detail=detail,
        ))

    def reset(self) -> None:
        self._components = []

    # ── Decision ───────────────────────────────────────────────

    def total_score(self) -> int:
        return sum(c.points for c in self._components)

    def max_possible(self) -> int:
        return sum(c.max_possible for c in self._components)

    def current_threshold(self) -> int:
        return self._threshold.current() if self._threshold else 60

    def decide(self, direction: str, pair: str = "") -> Dict:
        """Decide whether to trade based on accumulated score.

        Args:
            direction: "BUY" / "SELL" / "WAIT" — comes from rule/ML/LLM consensus
            pair:      symbol being evaluated (for logging)

        Returns:
            {
                "signal":     "BUY" | "SELL" | "WAIT",
                "score":      85,
                "threshold":  60,
                "max":        115,
                "trade":      True,
                "coverage":   0.74,  # score / max
                "components": [{layer, points, max, detail}, ...],
                "reason":     "Score 85 >= threshold 60"
            }
        """
        score = self.total_score()
        threshold = self.current_threshold()
        max_p = self.max_possible() or 1
        coverage = round(score / max_p, 2)

        if direction not in ("BUY", "SELL"):
            return self._result("WAIT", score, threshold, max_p, coverage,
                                reason="No clear direction from agents")

        if score >= threshold:
            # NOTE (known issue): record_trade() is called here at scoring
            # time, BEFORE actual execution confirmation. If the trade later
            # fails at the execution layer (rejection, slippage, etc.), the
            # adaptive threshold has already recorded it as a real trade,
            # slightly over-tightening the threshold. Ideally this call should
            # be moved to after execution confirmation in the caller, but that
            # requires changes outside this module.
            self._threshold.record_trade()
            return self._result(direction, score, threshold, max_p, coverage,
                                trade=True,
                                reason=f"Score {score} >= threshold {threshold}")

        return self._result("WAIT", score, threshold, max_p, coverage,
                            reason=f"Score {score} < threshold {threshold}")

    def _result(self, signal: str, score: int, threshold: int,
                max_p: int, coverage: float, trade: bool = False,
                reason: str = "") -> Dict:
        return {
            "signal":     signal,
            "score":      score,
            "threshold":  threshold,
            "max":        max_p,
            "coverage":   coverage,
            "trade":      trade,
            "components": [
                {"layer": c.layer, "points": c.points,
                 "max": c.max_possible, "detail": c.detail}
                for c in self._components
            ],
            "reason":     reason,
        }

    # ── Reporting ──────────────────────────────────────────────

    def summary_block(self) -> str:
        """ASCII one-box summary for the debugger log."""
        lines = ["┌─ Signal Score Breakdown ──────────────────────┐"]
        for c in self._components:
            pct = (c.points / c.max_possible * 100) if c.max_possible else 0
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            lines.append(f"│ {c.layer:<10} {bar} {c.points:>2}/{c.max_possible:<2} {c.detail[:18]:<18} │")
        score = self.total_score()
        threshold = self.current_threshold()
        max_p = self.max_possible() or 1
        lines.append("│ ──────────────────────────────────────────────│")
        lines.append(f"│ TOTAL      {score:>3} / {max_p:<3}    threshold={threshold:<3}      │")
        if score >= threshold:
            lines.append(f"│ DECISION   ✅ TRADE (score >= threshold)       │")
        else:
            lines.append(f"│ DECISION   ⏳ WAIT  (score < threshold)        │")
        lines.append("└──────────────────────────────────────────────┘")
        return "\n".join(lines)