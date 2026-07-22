"""
core/confidence_breakdown.py — Transparent confidence scorecard
============================================================
Builds the itemized confidence breakdown shown on Telegram trade
alerts, e.g.:

    Rule        +18
    Trend       +15
    Momentum    +10
    Sentiment    +8
    LLM         -12
    Liquidity    -6
    Resistance   -5
    ------------------
    Total        68%

Every line is a REAL contribution pulled from data the pipeline
already computed — including the filters added for the Consensus
Lock fix (`core/entry_safety_filters.py`). Liquidity and Resistance
show up here even on trades that were NOT hard-blocked by those
filters, so the operator can see the risk they carried instead of it
disappearing silently once a trade is filtered out.

`Total` is always the exact sum of the line items (then calibrated
via `EntrySafetyFilters.calibrate_confidence`, so it can never read
100%) — the scorecard is meant to be auditable, not decorative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from core.entry_safety_filters import EntrySafetyFilters

log = get_logger("confidence_breakdown")

# ── Point weights (tunable) ──────────────────────────────────────────
RULE_WEIGHT = 0.30          # mirrors SignalFusion's rule_engine layer weight
LLM_WEIGHT = 0.20           # mirrors SignalFusion's llm_analyst layer weight
TREND_STRONG_POINTS = 15.0  # trend_signal STRONG_UP/STRONG_DOWN aligned
TREND_WEAK_POINTS = 8.0     # trend_signal UP/DOWN aligned
MOMENTUM_RSI_POINTS = 5.0   # rsi_signal aligned with direction
MOMENTUM_MACD_POINTS = 5.0  # macd_cross aligned with direction
LIQUIDITY_PENALTY = 6.0     # matches EntrySafetyFilters liquidity_sweep_filter risk
RESISTANCE_PENALTY = 5.0    # matches EntrySafetyFilters distance_filter danger zone


@dataclass
class ConfidenceComponent:
    label: str
    points: float
    detail: str = ""


@dataclass
class ConfidenceBreakdown:
    direction: str
    components: List[ConfidenceComponent] = field(default_factory=list)
    raw_total: float = 0.0
    total: float = 0.0  # calibrated, never 100%

    def to_telegram_lines(self) -> List[str]:
        lines = []
        for c in self.components:
            sign = "+" if c.points >= 0 else ""
            lines.append(f"{c.label}: {sign}{c.points:.0f}")
        lines.append(f"Total = {self.total:.0f}%")
        return lines

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "components": [{"label": c.label, "points": c.points, "detail": c.detail} for c in self.components],
            "raw_total": self.raw_total,
            "total": self.total,
        }


def _trend_points(direction: str, trend_signal: Optional[str]) -> ConfidenceComponent:
    trend_signal = str(trend_signal or "").upper()
    bullish_strong = trend_signal == "STRONG_UP"
    bullish_weak = trend_signal == "UP"
    bearish_strong = trend_signal == "STRONG_DOWN"
    bearish_weak = trend_signal == "DOWN"

    if direction == "BUY":
        if bullish_strong:
            return ConfidenceComponent("Trend", TREND_STRONG_POINTS, trend_signal)
        if bullish_weak:
            return ConfidenceComponent("Trend", TREND_WEAK_POINTS, trend_signal)
        if bearish_strong:
            return ConfidenceComponent("Trend", -TREND_STRONG_POINTS, trend_signal)
        if bearish_weak:
            return ConfidenceComponent("Trend", -TREND_WEAK_POINTS, trend_signal)
    elif direction == "SELL":
        if bearish_strong:
            return ConfidenceComponent("Trend", TREND_STRONG_POINTS, trend_signal)
        if bearish_weak:
            return ConfidenceComponent("Trend", TREND_WEAK_POINTS, trend_signal)
        if bullish_strong:
            return ConfidenceComponent("Trend", -TREND_STRONG_POINTS, trend_signal)
        if bullish_weak:
            return ConfidenceComponent("Trend", -TREND_WEAK_POINTS, trend_signal)
    return ConfidenceComponent("Trend", 0.0, trend_signal or "NEUTRAL")


def _momentum_points(direction: str, rsi_signal: Optional[str], macd_cross: Optional[str]) -> ConfidenceComponent:
    rsi_signal = str(rsi_signal or "").upper()
    macd_cross = str(macd_cross or "").upper()
    points = 0.0
    details = []

    rsi_bull = "BULL" in rsi_signal or "OVERSOLD" in rsi_signal
    rsi_bear = "BEAR" in rsi_signal or "OVERBOUGHT" in rsi_signal
    macd_bull = "BULL" in macd_cross or macd_cross == "UP"
    macd_bear = "BEAR" in macd_cross or macd_cross == "DOWN"

    if direction == "BUY":
        if rsi_bull:
            points += MOMENTUM_RSI_POINTS; details.append(f"rsi={rsi_signal}")
        elif rsi_bear:
            points -= MOMENTUM_RSI_POINTS; details.append(f"rsi={rsi_signal}")
        if macd_bull:
            points += MOMENTUM_MACD_POINTS; details.append(f"macd={macd_cross}")
        elif macd_bear:
            points -= MOMENTUM_MACD_POINTS; details.append(f"macd={macd_cross}")
    elif direction == "SELL":
        if rsi_bear:
            points += MOMENTUM_RSI_POINTS; details.append(f"rsi={rsi_signal}")
        elif rsi_bull:
            points -= MOMENTUM_RSI_POINTS; details.append(f"rsi={rsi_signal}")
        if macd_bear:
            points += MOMENTUM_MACD_POINTS; details.append(f"macd={macd_cross}")
        elif macd_bull:
            points -= MOMENTUM_MACD_POINTS; details.append(f"macd={macd_cross}")

    return ConfidenceComponent("Momentum", points, ", ".join(details))


def build_confidence_breakdown(
    *,
    direction: str,
    rule_confidence: float = 0.0,
    llm_signal: str = "WAIT",
    llm_confidence: float = 0.0,
    sentiment_boost: float = 0.0,
    ind_ctx: Optional[Dict[str, Any]] = None,
    sr_ctx: Optional[Dict[str, Any]] = None,
    liquidity_ctx: Optional[Dict[str, Any]] = None,
) -> ConfidenceBreakdown:
    """
    Build the full 7-line scorecard: Rule, Trend, Momentum, Sentiment,
    LLM, Liquidity, Resistance -> Total.

    Liquidity and Resistance are ALWAYS evaluated and shown here, even
    on trades where EntrySafetyFilters didn't hard-block the trade —
    the point is visibility into risk that was filtered/penalized, not
    just a pass/fail gate.
    """
    ind_ctx = ind_ctx or {}
    sr_ctx = sr_ctx or {}
    liquidity_ctx = liquidity_ctx or {}
    direction = direction if direction in ("BUY", "SELL") else "BUY"

    components: List[ConfidenceComponent] = []

    # 1. Rule
    components.append(ConfidenceComponent("Rule", round(rule_confidence * RULE_WEIGHT), f"{rule_confidence:.0f}% conf"))

    # 2. Trend
    components.append(_trend_points(direction, ind_ctx.get("trend")))

    # 3. Momentum
    components.append(_momentum_points(direction, ind_ctx.get("rsi_signal"), ind_ctx.get("macd_cross")))

    # 4. Sentiment (already computed upstream as a signed +/- adjustment)
    components.append(ConfidenceComponent("Sentiment", round(sentiment_boost), ""))

    # 5. LLM
    llm_norm = "BUY" if "BUY" in str(llm_signal).upper() else ("SELL" if "SELL" in str(llm_signal).upper() else "WAIT")
    llm_pts = round(llm_confidence * LLM_WEIGHT)
    if llm_norm == direction:
        components.append(ConfidenceComponent("LLM", llm_pts, llm_signal))
    elif llm_norm in ("BUY", "SELL"):
        components.append(ConfidenceComponent("LLM", -llm_pts, llm_signal))
    else:
        components.append(ConfidenceComponent("LLM", 0.0, llm_signal))

    # 6. Liquidity (audit Rule 4 — shown even when not the deciding block)
    liq_check = EntrySafetyFilters.liquidity_sweep_filter(direction, liquidity_ctx)
    liq_pts = -LIQUIDITY_PENALTY if not liq_check.allowed else 0.0
    components.append(ConfidenceComponent("Liquidity", liq_pts, liq_check.reason))

    # 7. Resistance / Support distance (audit Rule 2 — shown even when
    # not the deciding block)
    dist_check = EntrySafetyFilters.distance_filter(
        direction, sr_ctx.get("dist_to_resistance_pips"), sr_ctx.get("dist_to_support_pips")
    )
    dist_pts = -RESISTANCE_PENALTY if not dist_check.allowed else 0.0
    label = "Resistance" if direction == "BUY" else "Support"
    components.append(ConfidenceComponent(label, dist_pts, dist_check.reason))

    raw_total = sum(c.points for c in components)
    total = EntrySafetyFilters.calibrate_confidence(raw_total)

    return ConfidenceBreakdown(direction=direction, components=components, raw_total=raw_total, total=total)
