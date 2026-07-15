"""
intelligence/confluence_engine.py — Multi-Factor Confluence Engine
====================================================================

Day 67 — The brain of the professional decision-making system.

Collects ALL analysis outputs from the AnalysisAgent's 12-step pipeline
and computes a single weighted confluence score. Then runs validation
gates (5+ factor rule, contradiction detector, news block, etc.) and
produces a final decision:

    A+ Setup  →  6-7 factors aligned, net score ≥ 50  → trade with full size
    A Setup   →  5-6 factors aligned, net score ≥ 35  → trade with normal size
    B Setup   →  5 factors aligned, net score ≥ 20    → trade with reduced size
    AVOID     →  <5 factors, contradiction, or news block → WAIT

Pipeline:
    AnalysisAgent outputs (12 contexts)
        ↓
    ConfluenceEngine.collect_factors()
        ↓
    DecisionScorer.score()  →  ConfluenceScore
        ↓
    ConfidenceCalibrator.calibrate()  →  calibrated confidence
        ↓
    SignalValidator.validate_all()  →  final pass/block
        ↓
    Final decision (BUY/SELL/WAIT + quality + factors breakdown)
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

try:
    from intelligence.decision_score import (
        ConfluenceScore, FactorScore, DecisionScorer, get_scorer,
    )
except ImportError:
    # Dummy classes/functions so code doesn't crash if decision_score unavailable
    class ConfluenceScore: pass
    class FactorScore: pass
    class DecisionScorer: pass
    def get_scorer(): return None

try:
    from intelligence.signal_validator import SignalValidator, get_signal_validator
except ImportError:
    class SignalValidator: pass
    def get_signal_validator(): return None

try:
    from intelligence.confidence_calibrator import get_calibrator
except ImportError:
    def get_calibrator(): return None

log = get_logger("confluence_engine")

DECISION_HISTORY_PATH = Path("memory/decision_history.jsonl")


@dataclass
class ConfluenceDecision:
    """The final output of the ConfluenceEngine."""
    pair: str
    timeframe: str
    direction: str                # BUY / SELL / NEUTRAL / WAIT
    confidence: float             # 0-100 (calibrated)
    setup_quality: str            # A+ / A / B / AVOID
    aligned_factors: int
    total_factors: int
    buy_score: float
    sell_score: float
    net_score: float
    factors: List[Dict[str, Any]] = field(default_factory=list)
    market_story: str = ""
    risks: List[str] = field(default_factory=list)
    validation_gates: List[Dict[str, Any]] = field(default_factory=list)
    should_trade: bool = False
    block_reason: str = ""
    calibration: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_telegram_alert(self) -> Optional[str]:
        """Format a Telegram signal alert. Returns None if not tradeable."""
        if not self.should_trade:
            return None
        quality_emoji = {"A+": "🌟", "A": "✅", "B": "⚠️"}.get(self.setup_quality, "❓")
        dir_emoji = "🟢" if self.direction == "BUY" else "🔴"
        factors_str = "\n".join(
            f"  {'✅' if f['direction'] == self.direction else '❌'} {f['name']}: {f['direction']} ({f['strength']:.0f}%)"
            for f in self.factors
        )
        return (
            f"{dir_emoji} FOREX AI SIGNAL — {self.setup_quality}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Pair: {self.pair} ({self.timeframe})\n"
            f"Direction: {self.direction}\n"
            f"Confidence: {self.confidence:.0f}%\n"
            f"Aligned Factors: {self.aligned_factors}/{self.total_factors}\n"
            f"Quality: {quality_emoji} {self.setup_quality}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Factors:\n{factors_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Story: {self.market_story[:200]}\n"
            f"━━━━━━━━━━━━━━━━━━━━━"
        )


class ConfluenceEngine:
    """Collects all analyses → computes confluence → validates → decides."""

    def __init__(self):
        self.scorer = get_scorer()
        self.validator = get_signal_validator()
        self.calibrator = get_calibrator()
        self._lock = threading.RLock()

    def evaluate(
        self,
        pair: str,
        timeframe: str,
        analysis_out: Dict[str, Any],
        news_blocked_pairs: Optional[Dict[str, str]] = None,
        risk_approved: bool = True,
        correlation_blocked: bool = False,
    ) -> ConfluenceDecision:
        """Run the full pipeline on a single pair's analysis output."""
        factors = self._collect_factors(analysis_out)
        score = self.scorer.score(factors)

        # Calibrate confidence based on historical accuracy
        calibration = self.calibrator.calibrate(score.confidence)
        calibrated_conf = calibration["calibrated"]
        score.confidence = calibrated_conf

        # Run validation gates
        validation = self.validator.validate_all(
            score=score,
            pair=pair,
            news_blocked_pairs=news_blocked_pairs or {},
            correlation_blocked=correlation_blocked,
            risk_approved=risk_approved,
        )

        # Build final decision
        should_trade = validation["should_trade"]
        direction = score.final_direction if should_trade else "WAIT"
        if score.final_direction == "NEUTRAL":
            direction = "WAIT"

        # Build market story (top 2-3 factors)
        top_factors = sorted(
            [f for f in score.factors if f.direction == score.final_direction and f.is_meaningful],
            key=lambda x: x.weighted_score, reverse=True,
        )[:3]
        story_parts = []
        for f in top_factors:
            story_parts.append(f"{f.name} {f.direction.lower()} ({f.reasoning[:40]})")
        market_story = " | ".join(story_parts) if story_parts else "No strong confluence"

        # Risks (factors opposing the decision)
        risks = []
        for f in score.factors:
            if f.direction != score.final_direction and f.direction != "NEUTRAL" and f.is_meaningful:
                risks.append(f"{f.name} {f.direction.lower()} ({f.reasoning[:40]})")

        decision = ConfluenceDecision(
            pair=pair,
            timeframe=timeframe,
            direction=direction,
            confidence=calibrated_conf,
            setup_quality=score.setup_quality if should_trade else "AVOID",
            aligned_factors=score.aligned_factors,
            total_factors=score.total_factors,
            buy_score=score.buy_score,
            sell_score=score.sell_score,
            net_score=score.net_score,
            factors=[f.to_dict() for f in score.factors],
            market_story=market_story,
            risks=risks[:3],
            validation_gates=validation["gates"],
            should_trade=should_trade,
            block_reason=validation["block_reason"],
            calibration=calibration,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        # Record to decision history
        self._record_decision(decision)

        return decision

    # ── Factor collectors ───────────────────────────────────────────

    def _collect_factors(self, analysis_out: Dict[str, Any]) -> List[FactorScore]:
        """Extract 7 factor scores from the AnalysisAgent's output dict."""
        factors: List[FactorScore] = []
        factors.append(self._smc_factor(analysis_out))
        factors.append(self._liquidity_factor(analysis_out))
        factors.append(self._session_factor(analysis_out))
        factors.append(self._currency_strength_factor(analysis_out))
        factors.append(self._intermarket_factor(analysis_out))
        factors.append(self._news_factor(analysis_out))
        factors.append(self._technical_factor(analysis_out))
        return [f for f in factors if f is not None]

    def _smc_factor(self, a: Dict[str, Any]) -> FactorScore:
        """Factor 1: Market Structure (SMC) — BOS / CHoCH / OB / FVG."""
        smc = a.get("smc_ctx") or a.get("smc") or {}
        signal = (smc.get("signal") or "").upper()
        conf = float(smc.get("confluence_score") or smc.get("score") or 0)
        grade = (smc.get("grade") or "").upper()

        direction = "NEUTRAL"
        strength = conf
        if signal in ("BUY", "BULLISH"):
            direction = "BUY"
        elif signal in ("SELL", "BEARISH"):
            direction = "SELL"

        # Grade bonus
        if grade in ("A+", "A"):
            strength = min(100, strength + 15)

        reasoning = smc.get("explanation") or smc.get("reason") or f"SMC {signal} {grade}"
        return FactorScore(
            name="smc", direction=direction, strength=min(100, strength),
            confidence=70, weight=25, reasoning=str(reasoning)[:100],
            details={"signal": signal, "grade": grade, "score": conf},
        )

    def _liquidity_factor(self, a: Dict[str, Any]) -> FactorScore:
        """Factor 2: Liquidity — sweep / equal highs-lows / stop hunt."""
        # Liquidity may come from session_ctx.fusion or smc_ctx.sweep
        smc = a.get("smc_ctx") or {}
        session = a.get("session_ctx") or {}
        fusion = session.get("fusion") if isinstance(session, dict) else {}
        sweep = smc.get("liquidity_sweep") or smc.get("sweep") or {}

        direction = "NEUTRAL"
        strength = 30.0
        reasoning = "no clear liquidity signal"

        # If sweep detected with reversal, that's a strong signal
        if isinstance(sweep, dict) and sweep.get("swept"):
            sweep_dir = (sweep.get("direction") or "").upper()
            if sweep_dir in ("BUY", "BULLISH"):
                direction = "BUY"
                strength = 70
                reasoning = f"sell-side liquidity sweep → bullish reversal"
            elif sweep_dir in ("SELL", "BEARISH"):
                direction = "SELL"
                strength = 70
                reasoning = f"buy-side liquidity sweep → bearish reversal"

        # Session fusion can add liquidity context
        if isinstance(fusion, dict) and fusion.get("fusion_score", 0) >= 60:
            strength = min(100, strength + 10)

        return FactorScore(
            name="liquidity", direction=direction, strength=strength,
            confidence=60, weight=20, reasoning=reasoning,
            details={"sweep": bool(sweep.get("swept")) if isinstance(sweep, dict) else False},
        )

    def _session_factor(self, a: Dict[str, Any]) -> FactorScore:
        """Factor 3: Session — London/NY = strong, Asian = range, dead zone = avoid."""
        session = a.get("session_ctx") or {}
        current = (session.get("current_session") or "").upper()
        trade_quality = (session.get("trade_quality") or "").upper()
        trade_allowed = session.get("trade_allowed", True)

        direction = "NEUTRAL"
        strength = 50
        reasoning = f"session={current} quality={trade_quality}"

        if not trade_allowed or current == "DEAD_ZONE":
            strength = 0
            reasoning = "dead zone — no trades"
        elif "OVERLAP" in current or "BEST" in trade_quality:
            strength = 80
            reasoning = f"{current} overlap — premium liquidity"
        elif "GOOD" in trade_quality:
            strength = 60
        elif "CAUTION" in trade_quality:
            strength = 35
        else:
            strength = 25

        # Session is direction-neutral — it boosts the OTHER factors' weight
        # We mark it NEUTRAL with a strength that affects validation only
        return FactorScore(
            name="session", direction=direction, strength=strength,
            confidence=70, weight=5, reasoning=reasoning,
            details={"session": current, "quality": trade_quality},
        )

    def _currency_strength_factor(self, a: Dict[str, Any]) -> FactorScore:
        """Factor 4: Currency Strength — relative strength model."""
        # May be in intermarket_ctx or sentiment_ctx
        inter = a.get("intermarket_ctx") or {}
        sent = a.get("sentiment_ctx") or {}
        macro_pair_bias = (inter.get("macro_pair_bias") or "").upper() if isinstance(inter, dict) else ""

        direction = "NEUTRAL"
        strength = 30
        reasoning = "no currency strength data"

        if macro_pair_bias in ("BULLISH", "BUY"):
            direction = "BUY"
            strength = 60
            reasoning = "macro_pair_bias BULLISH"
        elif macro_pair_bias in ("BEARISH", "SELL"):
            direction = "SELL"
            strength = 60
            reasoning = "macro_pair_bias BEARISH"

        # Sentiment can refine
        if isinstance(sent, dict):
            sent_score = sent.get("final_score", 0)
            if sent_score > 20:
                direction = "BUY"
                strength = min(80, strength + 20)
                reasoning += f" | sentiment +{sent_score}"
            elif sent_score < -20:
                direction = "SELL"
                strength = min(80, strength + 20)
                reasoning += f" | sentiment {sent_score}"

        return FactorScore(
            name="currency_strength", direction=direction, strength=strength,
            confidence=60, weight=15, reasoning=reasoning[:100],
            details={"macro_pair_bias": macro_pair_bias},
        )

    def _intermarket_factor(self, a: Dict[str, Any]) -> FactorScore:
        """Factor 5: Intermarket — DXY / Gold / VIX / US10Y / SP500."""
        inter = a.get("intermarket_ctx") or {}
        if not isinstance(inter, dict):
            return FactorScore(name="intermarket", direction="NEUTRAL", strength=0,
                               confidence=0, weight=15, reasoning="no intermarket data")

        macro_score = float(inter.get("macro_score") or 0)
        macro_regime = (inter.get("macro_regime") or "").upper()
        cross_confirmed = inter.get("cross_asset_confirmed", False)

        direction = "NEUTRAL"
        strength = abs(macro_score) * 0.8
        reasoning = f"macro_score={macro_score} regime={macro_regime}"

        if macro_score > 20:
            direction = "BUY"
        elif macro_score < -20:
            direction = "SELL"

        if cross_confirmed:
            strength = min(100, strength + 15)
            reasoning += " | cross-asset confirmed"

        return FactorScore(
            name="intermarket", direction=direction, strength=min(100, strength),
            confidence=65, weight=15, reasoning=reasoning[:100],
            details={"macro_score": macro_score, "regime": macro_regime,
                     "cross_confirmed": cross_confirmed},
        )

    def _news_factor(self, a: Dict[str, Any]) -> FactorScore:
        """Factor 6: News Intelligence — Day 66 bias."""
        news_intel = a.get("news_intelligence") or {}
        if not isinstance(news_intel, dict) or news_intel.get("blocked"):
            return FactorScore(
                name="news", direction="NEUTRAL", strength=0, confidence=0, weight=10,
                reasoning="news blocked" if news_intel.get("blocked") else "no news data",
                details=news_intel if isinstance(news_intel, dict) else {},
            )

        news_bias = (news_intel.get("news_bias") or "NEUTRAL").upper()
        adj = news_intel.get("confidence_change", 0)

        direction = "NEUTRAL"
        strength = 30
        reasoning = news_intel.get("adjustment_reason") or "neutral news"

        if news_bias == "BULLISH":
            direction = "BUY"
            strength = 50 + abs(adj) * 2
        elif news_bias == "BEARISH":
            direction = "SELL"
            strength = 50 + abs(adj) * 2

        return FactorScore(
            name="news", direction=direction, strength=min(100, strength),
            confidence=70, weight=10, reasoning=reasoning[:100],
            details={"news_bias": news_bias, "adjustment": adj},
        )

    def _technical_factor(self, a: Dict[str, Any]) -> FactorScore:
        """Factor 7: Technical — RSI / MACD / EMA / Pattern."""
        signal = a.get("signal") or {}
        bias = a.get("bias_ctx") or {}
        bias_direction = (bias.get("bias") or signal.get("signal") or "NEUTRAL").upper()
        rule_conf = float(signal.get("confidence") or 0)

        direction = "NEUTRAL"
        if bias_direction in ("BUY", "BULLISH"):
            direction = "BUY"
        elif bias_direction in ("SELL", "BEARISH"):
            direction = "SELL"

        strength = rule_conf
        reasoning = signal.get("reasons") or f"rule signal {bias_direction}"
        if isinstance(reasoning, list):
            reasoning = "; ".join(str(r) for r in reasoning[:2])

        return FactorScore(
            name="technical", direction=direction, strength=min(100, strength),
            confidence=65, weight=10, reasoning=str(reasoning)[:100],
            details={"rule_signal": bias_direction, "rule_confidence": rule_conf},
        )

    # ── Decision history persistence ────────────────────────────────

    def _record_decision(self, decision: ConfluenceDecision) -> None:
        """Append the decision to memory/decision_history.jsonl for later learning."""
        try:
            DECISION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": decision.generated_at,
                "pair": decision.pair,
                "timeframe": decision.timeframe,
                "direction": decision.direction,
                "confidence": decision.confidence,
                "setup_quality": decision.setup_quality,
                "aligned_factors": decision.aligned_factors,
                "total_factors": decision.total_factors,
                "buy_score": decision.buy_score,
                "sell_score": decision.sell_score,
                "net_score": decision.net_score,
                "should_trade": decision.should_trade,
                "block_reason": decision.block_reason,
                "factors": [
                    {"name": f["name"], "direction": f["direction"], "strength": f["strength"]}
                    for f in decision.factors
                ],
            }
            with DECISION_HISTORY_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            log.debug(f"[Confluence] decision history write failed: {e}")

    def record_outcome(self, pair: str, direction: str, confidence: float, won: bool) -> None:
        """Record a closed-trade outcome for confidence calibration."""
        try:
            self.calibrator.record_outcome(predicted_confidence=confidence, won=won)
        except Exception as e:
            log.debug(f"[Confluence] outcome record failed: {e}")

    def stats(self) -> Dict[str, Any]:
        """Return confluence + calibration stats."""
        return {
            "calibration": self.calibrator.status(),
        }


# ── singleton ───────────────────────────────────────────────────────
_ENGINE: Optional[ConfluenceEngine] = None


def get_confluence_engine() -> ConfluenceEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ConfluenceEngine()
    return _ENGINE
