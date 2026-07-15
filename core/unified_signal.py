"""
core/unified_signal.py — Unified Signal Object (Day 81+)
=========================================================

Single canonical signal dataclass that ALL agents (Market, SMC, ML, LLM,
Risk, Execution) speak.  This eliminates the "three parallel voting
systems return different dict shapes" problem.

Why this matters:
  Before: decision_agent returns {decision, confidence, entry, sl, tp, ...}
          confluence_engine returns {pair, direction, confidence, factors, ...}
          signal_fusion returns {signal, fused_confidence, layer_outputs, ...}
          → Each downstream consumer had to know which producer's schema to read.

  After:  Every agent emits UnifiedSignal.  One schema, one parser,
          one validator.  Downstream code never has to guess field names.

Pipeline (matches user's vision):

    MT5 Data  →  Market Analyst Agent  ─┐
                                       ├─→  Decision Engine  →  UnifiedSignal  →  Risk  →  Execution
    SMC Agent  ──────────────────────────┤
    ML Agent   ──────────────────────────┤
    LLM Agent  ──────────────────────────┤
    Risk Agent ──────────────────────────┘

Usage:
    sig = UnifiedSignal(
        pair="EURUSD",
        timeframe="M15",
        signal="BUY",
        confidence=82,
        reasons=["BOS confirmed", "Liquidity swept", "Order block detected"],
        entry=1.0850,
        sl=1.0820,
        tp=[1.0900, 1.0950],
        risk_percent=0.5,
        source_agents=["smc", "ml", "llm"],
        agent_votes={"smc": 82, "ml": 71, "llm": 78},
    )

    if sig.is_tradeable:
        execution_router.execute(sig.to_execution_dict())

⚠️  STATUS (institutional review, see audit report): NOT CURRENTLY WIRED
    IN. No agent, decision engine, or downstream consumer in the uploaded
    codebase constructs or reads a UnifiedSignal — trader.py's run_cycle()
    still passes around the older per-producer dict shapes this class was
    designed to replace. Ready to adopt; adoption requires updating each
    producing agent (not part of this review) to emit UnifiedSignal
    instead of its current dict.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ── Constants ─────────────────────────────────────────────────

VALID_SIGNALS = {"BUY", "SELL", "WAIT", "NO_TRADE"}

# Agent identifiers — keep this list authoritative so we can validate
# source_agent / agent_votes keys against it.
KNOWN_AGENTS = {
    "market",     # Market Analyst Agent (price/indicators/regime)
    "smc",        # Smart Money Concepts agent (BOS/CHOCH/order blocks)
    "ml",         # ML ensemble agent
    "llm",        # LLM analyst agent (Groq/Gemini)
    "rl",         # Reinforcement Learning agent
    "risk",       # Risk Officer Agent
    "confluence", # Confluence engine (7-factor scorer)
    "rule",       # Rule-based strategy agent
}


# ── Unified Signal ────────────────────────────────────────────

@dataclass
class UnifiedSignal:
    """
    Canonical signal object that flows through the entire pipeline:
        Agent → DecisionEngine → Risk → Execution → Memory → Learning

    All fields are typed so downstream code can rely on them without
    defensive `.get()` calls.
    """

    # ── Identity ──
    pair: str                                    # e.g. "EURUSD"
    timeframe: str                               # e.g. "M15"
    signal: str                                  # BUY / SELL / WAIT / NO_TRADE
    confidence: float                            # 0..100

    # ── Trade parameters (None for WAIT / NO_TRADE) ──
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: List[float] = field(default_factory=list)  # multi-TP support
    lot: Optional[float] = None
    risk_percent: Optional[float] = None         # e.g. 0.5 = 0.5% of balance

    # ── Provenance — which agents contributed to this decision ──
    source_agents: List[str] = field(default_factory=list)
    agent_votes: Dict[str, float] = field(default_factory=dict)  # {agent: confidence 0..100}

    # ── Human-readable reasoning (for Telegram / dashboard / journal) ──
    reasons: List[str] = field(default_factory=list)
    market_story: str = ""                       # 1-2 sentence narrative

    # ── Context that produced this signal ──
    market_bias: Optional[str] = None            # BULLISH / BEARISH / NEUTRAL
    regime: Optional[str] = None                 # TRENDING_UP / RANGING / VOLATILE / ...
    session: Optional[str] = None                # LONDON / NY / TOKYO / SYDNEY / OVERLAP / CLOSED

    # ── Risk metadata ──
    risk_warnings: List[str] = field(default_factory=list)
    news_safe: bool = True
    spread_pips: Optional[float] = None

    # ── Timestamps ──
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ── Metadata (any extra fields an agent wants to attach) ──
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Validation ────────────────────────────────────────────────

    def __post_init__(self):
        self.signal = self.signal.upper().replace("-", "_").replace(" ", "_")
        if self.signal == "NO":
            self.signal = "NO_TRADE"
        if self.signal not in VALID_SIGNALS:
            raise ValueError(f"UnifiedSignal.signal must be one of {VALID_SIGNALS}, got '{self.signal}'")

        if not 0 <= self.confidence <= 100:
            raise ValueError(f"UnifiedSignal.confidence must be 0..100, got {self.confidence}")

        # Validate agent identifiers
        for agent in self.source_agents:
            if agent not in KNOWN_AGENTS:
                # Don't fail hard — just log via metadata for forward-compat
                self.metadata.setdefault("unknown_agents", []).append(agent)

    # ── Convenience properties ────────────────────────────────────

    @property
    def is_tradeable(self) -> bool:
        """True if this signal represents an actionable trade (BUY/SELL)."""
        return self.signal in ("BUY", "SELL")

    @property
    def is_wait(self) -> bool:
        return self.signal == "WAIT"

    @property
    def is_block(self) -> bool:
        return self.signal == "NO_TRADE"

    @property
    def direction(self) -> Optional[str]:
        """Returns 'long' / 'short' / None — useful for risk engine."""
        if self.signal == "BUY":
            return "long"
        if self.signal == "SELL":
            return "short"
        return None

    @property
    def rr_ratio(self) -> Optional[float]:
        """Reward-to-risk ratio using first TP.  None if SL/TP/entry missing."""
        if not self.is_tradeable:
            return None
        if self.entry is None or self.sl is None or not self.tp:
            return None
        risk = abs(self.entry - self.sl)
        if risk == 0:
            return None
        reward = abs(self.tp[0] - self.entry)
        return round(reward / risk, 2)

    @property
    def consensus_level(self) -> str:
        """How many agents agreed — STRONG (>=3), MODERATE (2), WEAK (1), NONE (0)."""
        n = len([v for v in self.agent_votes.values() if v >= 50])
        if n >= 3:
            return "STRONG"
        if n == 2:
            return "MODERATE"
        if n == 1:
            return "WEAK"
        return "NONE"

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict — for EventBus, JSON logging, dashboard API."""
        d = asdict(self)
        # Normalize empty list → [] (not None) so downstream parsers stay simple
        for k in ("tp", "source_agents", "reasons", "risk_warnings"):
            if d.get(k) is None:
                d[k] = []
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_execution_dict(self) -> Dict[str, Any]:
        """Compact dict for ExecutionRouter.execute() — matches the existing
        router's expected schema so we don't have to rewrite the router."""
        if not self.is_tradeable:
            return {"decision": self.signal, "symbol": self.pair}
        return {
            "decision":      self.signal,
            "symbol":        self.pair,
            "entry":         self.entry,
            "sl":            self.sl,
            "tp":            self.tp[0] if self.tp else None,
            "lot":           self.lot,
            "confidence":    self.confidence,
            "rr":            self.rr_ratio or 0,
            "timeframe":     self.timeframe,
            "source_agents": self.source_agents,
            "reasons":       self.reasons[:3],  # keep Telegram message short
        }

    def to_telegram_message(self) -> str:
        """Pre-formatted Telegram alert message."""
        if not self.is_tradeable:
            return (
                f"⏳ {self.pair} {self.timeframe}\n"
                f"Signal: {self.signal} ({self.confidence:.0f}%)\n"
                f"Reason: {self.reasons[0] if self.reasons else 'n/a'}"
            )

        arrow = "🟢" if self.signal == "BUY" else "🔴"
        lines = [
            f"{arrow} {self.signal} {self.pair} ({self.timeframe})",
            f"━━━━━━━━━━━━━━━━━━━━━",
            f"Confidence : {self.confidence:.0f}%  ({self.consensus_level})",
            f"Entry      : {self.entry}",
            f"SL         : {self.sl}",
        ]
        if self.tp:
            for i, tp in enumerate(self.tp, 1):
                lines.append(f"TP{i}        : {tp}")
        if self.rr_ratio:
            lines.append(f"R:R        : 1:{self.rr_ratio}")
        if self.lot:
            lines.append(f"Lot        : {self.lot}")
        if self.risk_percent:
            lines.append(f"Risk       : {self.risk_percent}% of balance")
        if self.source_agents:
            lines.append(f"Agents     : {' + '.join(self.source_agents)}")
        if self.reasons:
            lines.append("━━━━━━━━━━━━━━━━━━━━━")
            lines.append("Why:")
            for r in self.reasons[:4]:
                lines.append(f"  • {r}")
        if self.risk_warnings:
            lines.append("⚠️ Warnings:")
            for w in self.risk_warnings[:2]:
                lines.append(f"  • {w}")
        return "\n".join(lines)

    # ── Class methods for construction ────────────────────────────

    @classmethod
    def wait(cls, pair: str, timeframe: str, reason: str = "") -> "UnifiedSignal":
        """Construct a WAIT signal — used when no tradeable setup found."""
        return cls(
            pair=pair,
            timeframe=timeframe,
            signal="WAIT",
            confidence=0,
            reasons=[reason] if reason else [],
        )

    @classmethod
    def block(cls, pair: str, timeframe: str, reason: str) -> "UnifiedSignal":
        """Construct a NO_TRADE signal — used when risk/news/session blocks."""
        return cls(
            pair=pair,
            timeframe=timeframe,
            signal="NO_TRADE",
            confidence=0,
            reasons=[reason],
            risk_warnings=[reason],
        )

    @classmethod
    def from_agent_votes(
        cls,
        pair: str,
        timeframe: str,
        agent_votes: Dict[str, float],
        reasons: List[str] = None,
        **kwargs,
    ) -> "UnifiedSignal":
        """
        Build a UnifiedSignal by aggregating votes from multiple agents.

        Voting rule:
          - Each agent votes BUY/SELL/WAIT with a confidence 0..100.
          - Agents with confidence < 50 are treated as WAIT.
          - The signal with the highest summed confidence wins.
          - Final confidence = weighted average of agreeing agents.
        """
        # NOTE: agent_votes here is {agent_name: (signal, confidence)}.
        # The simpler {agent_name: confidence} form is for storage only.
        buy_score = 0.0
        sell_score = 0.0
        buy_agents = []
        sell_agents = []
        for agent, vote in agent_votes.items():
            if isinstance(vote, (tuple, list)) and len(vote) == 2:
                vote_signal, vote_conf = vote[0].upper(), float(vote[1])
            else:
                # Just a confidence number — assume direction was BUY
                vote_signal, vote_conf = "BUY", float(vote)
            if vote_signal == "BUY" and vote_conf >= 50:
                buy_score += vote_conf
                buy_agents.append(agent)
            elif vote_signal == "SELL" and vote_conf >= 50:
                sell_score += vote_conf
                sell_agents.append(agent)

        if buy_score > sell_score and buy_agents:
            winning_signal = "BUY"
            agreeing = buy_agents
            agreeing_confs = [agent_votes[a][1] if isinstance(agent_votes[a], (tuple, list)) else agent_votes[a]
                              for a in buy_agents]
        elif sell_score > buy_score and sell_agents:
            winning_signal = "SELL"
            agreeing = sell_agents
            agreeing_confs = [agent_votes[a][1] if isinstance(agent_votes[a], (tuple, list)) else agent_votes[a]
                              for a in sell_agents]
        else:
            return cls.wait(pair, timeframe, reason="No agent consensus")

        final_conf = sum(agreeing_confs) / len(agreeing_confs)
        return cls(
            pair=pair,
            timeframe=timeframe,
            signal=winning_signal,
            confidence=round(final_conf, 1),
            source_agents=agreeing,
            agent_votes={a: (agent_votes[a][1] if isinstance(agent_votes[a], (tuple, list))
                              else agent_votes[a]) for a in agreeing},
            reasons=reasons or [],
            **kwargs,
        )


# ── Convenience helpers ───────────────────────────────────────

def merge_signals(signals: List[UnifiedSignal], pair: str, timeframe: str) -> UnifiedSignal:
    """Merge multiple agent-emitted UnifiedSignals into one canonical signal.

    Used by the Decision Engine to combine SMC + ML + LLM + Rule outputs
    into the single signal that flows to Risk + Execution.
    """
    if not signals:
        return UnifiedSignal.wait(pair, timeframe, reason="No agent signals")

    buy_signals = [s for s in signals if s.signal == "BUY"]
    sell_signals = [s for s in signals if s.signal == "SELL"]

    if not buy_signals and not sell_signals:
        return UnifiedSignal.wait(pair, timeframe, reason="All agents voted WAIT/NO_TRADE")

    # Pick the side with more agents (tie → higher total confidence)
    if len(buy_signals) > len(sell_signals) or (
        len(buy_signals) == len(sell_signals)
        and sum(s.confidence for s in buy_signals) >= sum(s.confidence for s in sell_signals)
    ):
        winners = buy_signals
        winning_signal = "BUY"
    else:
        winners = sell_signals
        winning_signal = "SELL"

    # Aggregate
    all_reasons = []
    for s in winners:
        all_reasons.extend(s.reasons)
    all_reasons = list(dict.fromkeys(all_reasons))  # dedupe, preserve order

    final_conf = sum(s.confidence for s in winners) / len(winners)
    agent_votes = {}
    for s in winners:
        # Use the first source_agent name, or fall back to a synthetic key
        agent_name = s.source_agents[0] if s.source_agents else f"agent_{id(s) % 1000}"
        agent_votes[agent_name] = s.confidence

    # Use the highest-confidence winner's trade params (entry/sl/tp)
    best = max(winners, key=lambda s: s.confidence)

    return UnifiedSignal(
        pair=pair,
        timeframe=timeframe,
        signal=winning_signal,
        confidence=round(final_conf, 1),
        entry=best.entry,
        sl=best.sl,
        tp=best.tp,
        lot=best.lot,
        risk_percent=best.risk_percent,
        source_agents=list(agent_votes.keys()),
        agent_votes=agent_votes,
        reasons=all_reasons[:6],  # cap at 6 reasons for readability
        market_story=best.market_story,
        market_bias=best.market_bias,
        regime=best.regime,
        session=best.session,
        risk_warnings=[w for s in winners for w in s.risk_warnings],
        news_safe=all(s.news_safe for s in winners),
        spread_pips=best.spread_pips,
        metadata={"merged_from": len(signals), "winners": len(winners)},
    )