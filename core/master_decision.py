"""
core/master_decision.py — Master Decision Engine (Day 73)
============================================================

The central brain coordination layer. Collects signals from ALL
intelligence layers, fuses them with dynamic weights, validates
the decision, and produces the FINAL BUY/SELL/WAIT signal.

Pipeline:
  1. Collect signals from 4 layers:
     - Rule Engine (Confluence Day 67)
     - ML Ensemble (Day 69-70)
     - RL Agent (Day 71)
     - LLM Analyst (MasterAnalyst Day 42+)
  2. SignalFusion → weighted confidence + conflict detection
  3. DecisionValidator → emergency checks + reasonableness
  4. ConfidenceManager → dynamic weight adjustment from outcomes
  5. Final output: BUY/SELL/WAIT + confidence + position size + explanation

This replaces the fragmented Day 66-72 integration in AnalysisAgent
with a single clean coordination layer.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

from core.signal_fusion import SignalFusion, LayerSignal, FusionResult, get_signal_fusion
from core.decision_validator import DecisionValidator, ValidationResult, get_decision_validator
from core.confidence_manager import ConfidenceManager, get_confidence_manager
from core.constants import MEMORY_DIR

# C2 ARCHITECTURAL FIX — wire AdaptiveDecisionEngine as advisory layer.
# This module is the backtest-calibrated soft scoring system that learns
# from historical trade outcomes. We import it DEFENSIVELY so the engine
# still works if the adaptive layer is unavailable (e.g. weights file
# missing on a fresh deployment).
try:
    from analysis.decision_bridge import make_adaptive_decision
    _ADAPTIVE_AVAILABLE = True
except Exception as _e_adaptive:
    make_adaptive_decision = None
    _ADAPTIVE_AVAILABLE = False
    log = get_logger("master_decision")
    log.debug(f"[MasterDecision] AdaptiveDecisionEngine unavailable (advisory layer disabled): {_e_adaptive}")

log = get_logger("master_decision")

DB_PATH = MEMORY_DIR / "master_decisions.db"

_ADAPTIVE_LABEL_TO_NUMERIC = {
    "high": 80.0,
    "medium": 50.0,
    "low": 0.0,
}


def _coerce_adaptive_numeric(value, default: float = 0.0) -> float:
    """Map adaptive engine label strings (e.g. 'Low') to numeric values."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return _ADAPTIVE_LABEL_TO_NUMERIC.get(str(value).strip().lower(), default)


@dataclass
class MasterDecision:
    """The final output of the Master Decision Engine."""
    pair: str
    timeframe: str
    final_signal: str          # BUY / SELL / WAIT / NO_TRADE
    master_confidence: float   # 0-100
    agreement: str             # "4/4"
    position_size: str         # FULL / HALF / REDUCED / WAIT / NO_TRADE
    position_multiplier: float
    layer_signals: Dict[str, str] = field(default_factory=dict)  # {"rule_engine": "BUY 71%"}
    has_conflict: bool = False
    conflict_reason: str = ""
    override_reason: str = ""
    explanation: List[str] = field(default_factory=list)
    weights_used: Dict[str, float] = field(default_factory=dict)
    validation_checks: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: str = ""
    decision_id: Optional[int] = None
    # Day 90 — Strategy Selector context
    strategy: str = "WAIT"                              # active strategy family
    strategy_confidence: int = 0                        # selector's confidence
    strategy_risk_mult: float = 0.0                     # selector's risk multiplier
    strategy_active_modules: List[str] = field(default_factory=list)
    strategy_avoid: List[str] = field(default_factory=list)
    strategy_reason: str = ""
    # C2 ARCHITECTURAL FIX — Adaptive Decision (backtest-calibrated) advisory
    # layer. The AdaptiveDecisionEngine is a backtest-calibrated soft scoring
    # system that learns from historical trade outcomes. Its verdict is
    # ADVISORY only — it does NOT override the MasterDecision's final_signal,
    # but its score/confidence are recorded so the operator can compare
    # the two pipelines and eventually promote it to authoritative.
    adaptive_action: str = ""            # BUY/SELL/NO_TRADE from AdaptiveDecisionEngine
    adaptive_confidence: float = 0.0     # 0-100
    adaptive_score: float = 0.0          # raw confluence score
    adaptive_source: str = ""            # which strategies agreed
    adaptive_divergence: bool = False    # True if adaptive disagrees with master

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_telegram_alert(self) -> Optional[str]:
        """Format a high-conviction Telegram alert."""
        if self.final_signal not in ("BUY", "SELL"):
            return None
        dir_emoji = "🟢" if self.final_signal == "BUY" else "🔴"
        lines = [
            f"{dir_emoji} FOREX AI MASTER SIGNAL",
            f"━━━━━━━━━━━━━━━━━━━━━",
            f"Pair: {self.pair} ({self.timeframe})",
            f"Direction: {self.final_signal}",
            f"4-Layer Agreement: {self.agreement}",
            f"Master Confidence: {self.master_confidence:.0f}%",
            f"Position: {self.position_size}",
            f"━━━━━━━━━━━━━━━━━━━━━",
            f"Layers:",
        ]
        for layer, info in self.layer_signals.items():
            lines.append(f"  {info}")
        lines.append(f"━━━━━━━━━━━━━━━━━━━━━")
        if self.has_conflict:
            lines.append(f"⚠️ {self.conflict_reason[:100]}")
        if self.explanation:
            lines.append("Why:")
            for exp in self.explanation[:3]:
                lines.append(f"  {exp}")
        return "\n".join(lines)


class MasterDecisionEngine:
    """Central brain — collects, fuses, validates, and outputs the final decision."""

    def __init__(self):
        self.fusion = get_signal_fusion()
        self.validator = get_decision_validator()
        self.confidence_mgr = get_confidence_manager()
        # Day 90 — Strategy Selector (lazy import to avoid circular dep)
        try:
            from strategy.selector import StrategySelector
            self.strategy_selector = StrategySelector(conservative=False)
        except Exception as e:
            log.debug(f"[MasterDecision] StrategySelector unavailable: {e}")
            self.strategy_selector = None
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(DB_PATH)) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS master_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    timeframe TEXT,
                    rule_signal TEXT,
                    ml_signal TEXT,
                    rl_signal TEXT,
                    llm_signal TEXT,
                    agreement TEXT,
                    confidence REAL,
                    final_signal TEXT,
                    position_size TEXT,
                    has_conflict INTEGER,
                    override_reason TEXT,
                    timestamp TEXT NOT NULL,
                    actual_result TEXT,
                    pnl_usd REAL
                )
            """)
            # Day 90 migration — add strategy columns (idempotent)
            try:
                existing_cols = {row[1] for row in c.execute("PRAGMA table_info(master_decisions)").fetchall()}
                new_cols = {
                    "strategy":              "TEXT",
                    "strategy_confidence":   "INTEGER",
                    "strategy_risk_mult":    "REAL",
                    "strategy_reason":       "TEXT",
                }
                for col, col_type in new_cols.items():
                    if col not in existing_cols:
                        c.execute(f"ALTER TABLE master_decisions ADD COLUMN {col} {col_type}")
            except Exception as e:
                log.debug(f"[MasterDecision] DB migration: {e}")
            c.commit()

    def decide(
        self,
        pair: str,
        timeframe: str,
        rule_signal: str = "WAIT",
        rule_confidence: float = 0.0,
        ml_signal: str = "WAIT",
        ml_confidence: float = 0.0,
        rl_signal: str = "HOLD",
        rl_confidence: float = 50.0,
        llm_signal: str = "WAIT",
        llm_confidence: float = 0.0,
        rule_reasoning: str = "",
        ml_reasoning: str = "",
        rl_reasoning: str = "",
        llm_reasoning: str = "",
        # Day 90 — Strategy Selector inputs
        regime: Optional[Dict[str, Any]] = None,
        mtf_bias: Optional[Dict[str, Any]] = None,
        structure: Optional[Dict[str, Any]] = None,
        strategy_context: Optional[Dict[str, Any]] = None,
        # Ollama Qwen3:4B institutional validation
        market_data: Optional[Dict[str, Any]] = None,
        entry_price: float = 0.0,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        risk_reward: float = 0.0,
    ) -> MasterDecision:
        """Run the full master decision pipeline.

        Args:
            pair: Trading pair (e.g. "EURUSD").
            timeframe: Timeframe label.
            rule_signal/confidence: From Day 67 Confluence Engine.
            ml_signal/confidence: From Day 69-70 ML Ensemble.
            rl_signal/confidence: From Day 71 RL Agent.
            llm_signal/confidence: From Day 42+ MasterAnalyst.
            regime: Optional MarketRegimeDetector.detect() result. If provided
                    AND strategy_context is None, the engine will run the
                    StrategySelector itself to derive the strategy family.
            mtf_bias: Optional MultiTimeframeAnalyzer bias dict (for selector).
            structure: Optional MarketStructureEngine ai_context (for selector).
            strategy_context: Pre-built StrategySelector.select() result.
                              If provided, takes precedence over regime-based
                              internal selection.

        Returns:
            MasterDecision with the final signal + full breakdown.
        """
        # ── Day 90 — Strategy selection ────────────────────────
        # Either use a pre-built strategy_context (passed by AnalysisAgent
        # which has all the right inputs) or build it here from regime.
        # FIX: `strategy_choice = {}` used to mean two different things —
        # "the selector genuinely chose WAIT" and "we never got strategy
        # data at all" — and both looked identical below (strategy_family
        # == "WAIT"), so a missing/failed selector call silently overrode
        # validated BUY/SELL signals as if it were a deliberate stand-aside.
        # `strategy_available` distinguishes the two so a missing selector
        # can no longer masquerade as a real decision.
        strategy_choice: Dict[str, Any] = {}
        strategy_available = False
        if strategy_context is not None:
            strategy_choice = strategy_context
            strategy_available = True
        elif self.strategy_selector is not None and regime:
            try:
                strategy_choice = self.strategy_selector.select(
                    regime=regime,
                    mtf_bias=mtf_bias,
                    structure=structure,
                )
                strategy_available = True
            except Exception as e:
                log.warning(f"[MasterDecision] StrategySelector failed: {e}")
                strategy_choice = {}
                strategy_available = False
        else:
            log.debug(
                "[MasterDecision] No regime/strategy_context supplied — "
                "skipping strategy-based override for this decision"
            )

        # Strategy WAIT means "stand aside" — the engine may still proceed
        # but position_multiplier will be 0.
        strategy_family = strategy_choice.get("strategy", "WAIT")
        strategy_conf   = int(strategy_choice.get("confidence", 0))
        # FIX: default to 1.0 (no adjustment), not 0.0. A real WAIT choice
        # from the selector always carries its own risk_mult of 0.0 (set
        # explicitly in selector.py's `if strategy == STRATEGY_WAIT:
        # final_risk = 0.0`), so this default only ever applies when
        # strategy_choice is the missing-data {} — and missing data should
        # leave position sizing to the validator, not silently zero it.
        strategy_risk   = float(strategy_choice.get("risk_mult", 1.0))
        strategy_mods   = list(strategy_choice.get("active_modules", []))
        strategy_avoid  = list(strategy_choice.get("avoid", []))
        strategy_reason = str(strategy_choice.get("reason", ""))
        # Get dynamic weights
        weights = self.confidence_mgr.get_weights()

        # ── Step 1: Collect signals from all 4 layers ─────────────
        signals: List[LayerSignal] = []

        # Normalize signals
        def _norm(s):
            if "STRONG_BUY" in str(s): return "BUY"
            if "STRONG_SELL" in str(s): return "SELL"
            return str(s).upper() if s else "WAIT"

        rule_sig = _norm(rule_signal)
        ml_sig = _norm(ml_signal)
        rl_sig = _norm(rl_signal) if rl_signal != "HOLD" else "WAIT"
        llm_sig = _norm(llm_signal)

        # Only include layers with meaningful signals
        if rule_sig in ("BUY", "SELL") or rule_confidence > 0:
            signals.append(LayerSignal(
                layer="rule_engine", signal=rule_sig,
                confidence=max(rule_confidence, 50.0 if rule_sig in ("BUY", "SELL") else 0),
                weight=weights.get("rule_engine", 0.30),
                reasoning=rule_reasoning[:100],
            ))

        if ml_sig in ("BUY", "SELL") or ml_confidence > 0:
            signals.append(LayerSignal(
                layer="ml_ensemble", signal=ml_sig,
                confidence=max(ml_confidence, 50.0 if ml_sig in ("BUY", "SELL") else 0),
                weight=weights.get("ml_ensemble", 0.30),
                reasoning=ml_reasoning[:100],
            ))

        if rl_sig in ("BUY", "SELL"):
            signals.append(LayerSignal(
                layer="rl_agent", signal=rl_sig,
                confidence=max(rl_confidence, 50.0),
                weight=weights.get("rl_agent", 0.20),
                reasoning=rl_reasoning[:100],
            ))

        if llm_sig in ("BUY", "SELL") or llm_confidence > 0:
            signals.append(LayerSignal(
                layer="llm_analyst", signal=llm_sig,
                confidence=max(llm_confidence, 50.0 if llm_sig in ("BUY", "SELL") else 0),
                weight=weights.get("llm_analyst", 0.20),
                reasoning=llm_reasoning[:100],
            ))

        # If no signals at all → WAIT
        if not signals:
            return MasterDecision(
                pair=pair.upper(), timeframe=timeframe,
                final_signal="WAIT", master_confidence=0.0,
                agreement="0/4", position_size="NO_TRADE", position_multiplier=0.0,
                explanation=["No intelligence layers produced a signal"],
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                strategy=strategy_family,
                strategy_confidence=strategy_conf,
                strategy_risk_mult=strategy_risk,
                strategy_active_modules=strategy_mods,
                strategy_avoid=strategy_avoid,
                strategy_reason=strategy_reason,
            )

        # ── Step 2: Fuse signals ───────────────────────────────────
        fusion_result = self.fusion.fuse(signals)

        # ── Step 3: Validate ───────────────────────────────────────
        validation = self.validator.validate(
            fusion_result, signals,
            market_data=market_data,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
        )

        # ── Day 90 — Strategy-aware position sizing ─────────────────
        # The validator produces a position_multiplier based on agreement
        # and confidence. The strategy selector also produces a risk_mult
        # based on regime + volatility. We MULTIPLY them so that:
        #   - A WAIT strategy → 0.0 (no trade regardless of validation)
        #   - A high-volatility regime → reduced size
        #   - Both factors agree → full size
        # If strategy_family == WAIT, force final_signal to WAIT (unless
        # it's already NO_TRADE) — the selector said "stand aside".
        validator_mult = validation.position_multiplier
        strategy_adjusted_mult = round(validator_mult * strategy_risk, 3)
        # FIX: only treat this as a deliberate "selector says stand aside"
        # when the selector actually ran (strategy_available). If regime/
        # strategy_context was never supplied, strategy_family defaults to
        # "WAIT" with no meaning behind it — that must NOT override a
        # validated BUY/SELL from the four intelligence layers.
        if strategy_available and strategy_family == "WAIT":
            strategy_adjusted_mult = 0.0
            # If validation said BUY/SELL but strategy said WAIT, downgrade
            if validation.final_signal in ("BUY", "SELL"):
                log.info(
                    f"[MasterDecision] Strategy WAIT override — "
                    f"validation said {validation.final_signal} but selector says stand aside"
                )
                final_signal = "WAIT"
                override_reason = (
                    f"Strategy WAIT ({strategy_reason}) — standing aside despite layer agreement"
                )
            else:
                final_signal = validation.final_signal
                override_reason = validation.override_reason
            position_size = "WAIT" if final_signal != "NO_TRADE" else "NO_TRADE"
            master_confidence = validation.confidence * 0.5  # halve confidence on WAIT
        else:
            final_signal = validation.final_signal
            override_reason = validation.override_reason
            position_size = validation.position_size
            master_confidence = validation.confidence
            # If strategy_adjusted_mult is much lower than validator_mult,
            # downgrade position size label
            if strategy_adjusted_mult > 0 and strategy_adjusted_mult < validator_mult * 0.6:
                if position_size == "FULL":
                    position_size = "HALF"
                elif position_size == "HALF":
                    position_size = "REDUCED"

        # ── Step 4: Build final decision ───────────────────────────
        # Build layer display dict
        layer_display: Dict[str, str] = {}
        for s in signals:
            layer_display[s.layer] = f"{s.signal} {s.confidence:.0f}%"

        decision = MasterDecision(
            pair=pair.upper(),
            timeframe=timeframe,
            final_signal=final_signal,
            master_confidence=master_confidence,
            agreement=fusion_result.agreement,
            position_size=position_size,
            position_multiplier=strategy_adjusted_mult,
            layer_signals=layer_display,
            has_conflict=fusion_result.has_conflict,
            conflict_reason=fusion_result.conflict_reason,
            override_reason=override_reason,
            explanation=fusion_result.explanation,
            weights_used={k: round(v, 3) for k, v in weights.items()},
            validation_checks=validation.checks or [],
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # Day 90 — Strategy Selector context
            strategy=strategy_family,
            strategy_confidence=strategy_conf,
            strategy_risk_mult=strategy_risk,
            strategy_active_modules=strategy_mods,
            strategy_avoid=strategy_avoid,
            strategy_reason=strategy_reason,
        )

        # ── C2 ARCHITECTURAL FIX — Adaptive Decision Advisory Layer ──────
        # Run the AdaptiveDecisionEngine as an ADVISORY cross-check.
        # It does NOT override `final_signal` — its verdict is recorded
        # so the operator can compare the two pipelines and decide
        # whether to promote it to authoritative in the future.
        #
        # The adaptive engine needs a UnifiedSignalEngine output (5-engine
        # consensus). Since MasterDecision doesn't have direct access to
        # that, we build a minimal pseudo-unified context from the 4
        # layer signals we already have. The adaptive engine will fall
        # back to legacy consensus mode if the input doesn't have the
        # full 5-engine structure.
        if _ADAPTIVE_AVAILABLE and make_adaptive_decision is not None:
            try:
                _pseudo_unified = {
                    "consensus": {
                        "action": final_signal if final_signal in ("BUY", "SELL") else "NO_TRADE",
                        "confidence": master_confidence,
                        "buy_score": 1 if final_signal == "BUY" else 0,
                        "sell_score": 1 if final_signal == "SELL" else 0,
                    },
                    "engines": {
                        "rule_engine": {"signal": rule_sig, "confidence": rule_confidence},
                        "ml_ensemble": {"signal": ml_sig, "confidence": ml_confidence},
                        "llm_analyst": {"signal": llm_sig, "confidence": llm_confidence},
                    },
                }
                _adaptive = make_adaptive_decision(
                    _pseudo_unified,
                    current_price=None,
                    mode="confluence",
                )
                decision.adaptive_action = _adaptive.get("action", "NO_TRADE")
                decision.adaptive_confidence = _coerce_adaptive_numeric(
                    _adaptive.get("confidence", 0)
                )
                decision.adaptive_score = _coerce_adaptive_numeric(
                    _adaptive.get("score", 0)
                )
                decision.adaptive_source = _adaptive.get("source", "unknown")
                # Divergence flag — operator can monitor how often the two
                # pipelines disagree. High divergence rate → investigate
                # which one is more accurate via backtest.
                _adaptive_action_norm = decision.adaptive_action.upper()
                if _adaptive_action_norm in ("BUY", "SELL") and final_signal in ("BUY", "SELL"):
                    decision.adaptive_divergence = (_adaptive_action_norm != final_signal)
                if decision.adaptive_divergence:
                    log.info(
                        f"[MasterDecision] ADAPTIVE DIVERGENCE — master={final_signal} "
                        f"vs adaptive={decision.adaptive_action} "
                        f"(conf={decision.adaptive_confidence:.0f}%, "
                        f"score={decision.adaptive_score:.2f}, "
                        f"source={decision.adaptive_source})"
                    )
            except Exception as _e_adaptive_run:
                log.debug(f"[MasterDecision] Adaptive advisory skipped: {_e_adaptive_run}")

        # ── Step 5: Persist to DB ──────────────────────────────────
        try:
            with sqlite3.connect(str(DB_PATH)) as c:
                cur = c.execute("""
                    INSERT INTO master_decisions
                    (pair, timeframe, rule_signal, ml_signal, rl_signal, llm_signal,
                     agreement, confidence, final_signal, position_size, has_conflict,
                     override_reason, timestamp,
                     strategy, strategy_confidence, strategy_risk_mult, strategy_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    decision.pair, decision.timeframe,
                    layer_display.get("rule_engine", ""),
                    layer_display.get("ml_ensemble", ""),
                    layer_display.get("rl_agent", ""),
                    layer_display.get("llm_analyst", ""),
                    decision.agreement, decision.master_confidence,
                    decision.final_signal, decision.position_size,
                    1 if decision.has_conflict else 0,
                    decision.override_reason,
                    decision.generated_at,
                    decision.strategy, decision.strategy_confidence,
                    decision.strategy_risk_mult, decision.strategy_reason,
                ))
                c.commit()
                decision.decision_id = cur.lastrowid
        except Exception as e:
            log.debug(f"[MasterDecision] DB save failed: {e}")

        log.info(
            f"[MasterDecision] {pair} {timeframe} → {decision.final_signal} "
            f"| conf={decision.master_confidence:.0f}% | agreement={decision.agreement} "
            f"| position={decision.position_size} (mult={decision.position_multiplier:.2f})"
            f"{' | CONFLICT' if decision.has_conflict else ''}"
            f"{' | OVERRIDE: ' + decision.override_reason if decision.override_reason else ''}"
            f" | strategy={decision.strategy}@{decision.strategy_confidence}%"
        )

        return decision

    def record_outcome(self, decision_id: int, result: str, pnl_usd: float,
                       layer_predictions: Optional[Dict[str, str]] = None) -> None:
        """Record the actual trade outcome and update layer accuracy."""
        try:
            with sqlite3.connect(str(DB_PATH)) as c:
                c.execute(
                    "UPDATE master_decisions SET actual_result = ?, pnl_usd = ? WHERE id = ?",
                    (result, float(pnl_usd), decision_id),
                )
                c.commit()
        except Exception as e:
            log.warning(f"Suppressed exception at line 422: {e}")
            pass

        # Update each layer's accuracy
        if layer_predictions:
            for layer, predicted in layer_predictions.items():
                self.confidence_mgr.record_outcome(layer, predicted, result)

    def stats(self) -> Dict[str, Any]:
        """Return master decision engine stats."""
        try:
            with sqlite3.connect(str(DB_PATH)) as c:
                total = c.execute("SELECT COUNT(*) FROM master_decisions").fetchone()[0]
                with_result = c.execute("SELECT COUNT(*) FROM master_decisions WHERE actual_result IS NOT NULL").fetchone()[0]
                wins = c.execute("SELECT COUNT(*) FROM master_decisions WHERE actual_result = 'WIN'").fetchone()[0]
                losses = c.execute("SELECT COUNT(*) FROM master_decisions WHERE actual_result = 'LOSS'").fetchone()[0]
        except Exception as e:
            total = with_result = wins = losses = 0
        return {
            "total_decisions": total,
            "closed_with_result": with_result,
            "wins": wins,
            "losses": losses,
            "win_rate": round((wins / (wins + losses) * 100) if (wins + losses) else 0, 1),
            "confidence_manager": self.confidence_mgr.status(),
        }


# ── Singleton ───────────────────────────────────────────────────────

_ENGINE: Optional[MasterDecisionEngine] = None


def get_master_decision_engine() -> MasterDecisionEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = MasterDecisionEngine()
    return _ENGINE