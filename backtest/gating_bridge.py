"""
backtest/gating_bridge.py — Bridge live gating pipeline into backtest engine
=============================================================================

Round-8 audit fix: the backtest engine (engine.py, simulator.py,
honest_backtest_engine.py, per_strategy_tester.py) was completely
disconnected from the live gating pipeline. Backtest results showed
win-rates for a world where EVERY signal becomes a trade — but live
trading blocks most signals via:

  - ConfidenceEngine.calculate() → final_confidence
  - LiveRiskManager.check_trade_permission() → tier gate (80%/70%/55%)
  - LiveRiskManager tier progression (Tier 1 → 2 → 3 as trades close)
  - Confluence quality gate (AVOID grade blocks trade)
  - ConfidenceEngine sample-size progression (record_outcome on close)

This bridge module wraps those live modules so the backtest engine
can call them WITHOUT needing the full live orchestrator wired up.
It's a thin adapter — the actual logic lives in the live modules
(learning/confidence_engine.py, risk/live_risk_manager.py).

Usage in backtest engine:
    from backtest.gating_bridge import BacktestGate, BacktestGateResult

    gate = BacktestGate(symbol="EURUSD", timeframe="M15", initial_balance=10000)

    # Before opening a trade:
    gate_result = gate.check(
        signal=signal_dict,
        confidence_base=signal.get("confidence", 70),
        pattern=signal.get("pattern", "unknown"),
        regime=signal.get("regime", "UNKNOWN"),
        balance=current_balance,
        atr=signal.get("atr", 0.001),
        sl_pips=signal.get("stop_pips", 15),
        tp_pips=signal.get("stop_pips", 15) * signal.get("rr_ratio", 2.0),
    )
    if gate_result.allowed:
        position = simulator.open_position(...)
    else:
        # Log as BLOCKED — this is the key metric that was missing
        blocked_stats.record(gate_result.reject_reason)

    # After a trade closes:
    gate.record_outcome(
        pattern=signal.get("pattern", "unknown"),
        pair="EURUSD",
        timeframe="M15",
        regime=signal.get("regime", "UNKNOWN"),
        won=(closed_trade["result"] == "WIN"),
    )
    # This updates ConfidenceEngine's sample_size + win_rate, so
    # future signals on the same pattern get realistic confidence
    # adjustments instead of staying at the flat prior.

The gate is OPTIONAL — if the live modules can't be imported (e.g.
missing dependencies), the gate degrades to allow-all mode and logs
a warning. This keeps the backtest runnable in minimal environments.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("backtest.gating_bridge")


# ── Lazy imports — degrade gracefully if live modules unavailable ──

_CONFIDENCE_ENGINE = None
_LIVE_RISK_MANAGER = None
_LRM_SINGLETON = None


def _try_import_live_modules():
    """Import live gating modules. Returns (engine_cls, lrm_cls) or (None, None)."""
    global _CONFIDENCE_ENGINE, _LIVE_RISK_MANAGER
    if _CONFIDENCE_ENGINE is not None and _LIVE_RISK_MANAGER is not None:
        return _CONFIDENCE_ENGINE, _LIVE_RISK_MANAGER
    try:
        from learning.confidence_engine import ConfidenceEngine
        _CONFIDENCE_ENGINE = ConfidenceEngine
    except Exception as e:
        log.warning(
            f"[BacktestGate] ConfidenceEngine import failed — gate will "
            f"run in allow-all mode: {e}"
        )
    try:
        from risk.live_risk_manager import LiveRiskManager, get_live_risk_manager
        _LIVE_RISK_MANAGER = LiveRiskManager
        # Get the shared singleton so tier progression persists across
        # multiple BacktestGate instances (e.g. one per strategy in
        # per_strategy_tester). The singleton is created with default
        # tier=1 (Round-7 fix).
        global _LRM_SINGLETON
        _LRM_SINGLETON = get_live_risk_manager()
    except Exception as e:
        log.warning(
            f"[BacktestGate] LiveRiskManager import failed — gate will "
            f"run in allow-all mode: {e}"
        )
    return _CONFIDENCE_ENGINE, _LIVE_RISK_MANAGER


@dataclass
class BacktestGateResult:
    """Result of a backtest gating check."""
    allowed: bool = False
    final_confidence: float = 0.0
    reject_reason: str = ""
    reject_stage: str = ""  # "confidence_engine" | "lrm" | "confluence" | ""
    tier: int = 1
    tier_name: str = "Initial Live"
    min_confidence_required: float = 80.0
    components: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "final_confidence": round(self.final_confidence, 1),
            "reject_reason": self.reject_reason,
            "reject_stage": self.reject_stage,
            "tier": self.tier,
            "tier_name": self.tier_name,
            "min_confidence_required": self.min_confidence_required,
        }


class BacktestGate:
    """Bridge between backtest engine and live gating pipeline.

    Wraps ConfidenceEngine + LiveRiskManager so the backtest can:
      1. Check if a signal would pass live gating (before opening)
      2. Record outcomes to build up sample_size (after closing)
      3. Auto-promote/demote tier as trades close

    This makes backtest results reflect what would ACTUALLY happen
    in live trading, instead of an idealized "every signal trades" world.
    """

    def __init__(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "M15",
        initial_balance: float = 10000.0,
        enable_confidence_engine: bool = True,
        enable_lrm: bool = True,
        enable_confluence_gate: bool = True,
        confluence_avoid_blocks: bool = True,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.enable_confidence_engine = enable_confidence_engine
        self.enable_lrm = enable_lrm
        self.enable_confluence_gate = enable_confluence_gate
        self.confluence_avoid_blocks = confluence_avoid_blocks

        # Stats for reporting
        self._blocked_count = 0
        self._allowed_count = 0
        self._block_reasons: Dict[str, int] = {}

        # Lazy-init live modules
        engine_cls, lrm_cls = _try_import_live_modules()
        self._ce: Optional[Any] = engine_cls() if engine_cls else None
        self._lrm: Optional[Any] = _LRM_SINGLETON if lrm_cls else None

        if self._ce is None or self._lrm is None:
            log.warning(
                f"[BacktestGate] {symbol} {timeframe} running in DEGRADED mode "
                f"(allow-all). Live gating modules unavailable. Backtest "
                f"results will NOT reflect live trade-blocking behavior."
            )

    def check(
        self,
        signal: Dict[str, Any],
        confidence_base: float = 70.0,
        pattern: str = "unknown",
        regime: str = "UNKNOWN",
        balance: Optional[float] = None,
        atr: float = 0.001,
        atr_median: float = 0.001,
        sl_pips: float = 15.0,
        tp_pips: float = 30.0,
        spread_pips: float = 1.5,
        open_positions: Optional[List[Dict]] = None,
        daily_pnl: float = 0.0,
        weekly_pnl: float = 0.0,
    ) -> BacktestGateResult:
        """Run the live gating pipeline on a backtest signal.

        Returns a BacktestGateResult. If `allowed=False`, the backtest
        engine should NOT open the trade — it should log the block
        reason instead.

        If live modules are unavailable, returns allowed=True (degraded
        allow-all mode) so the backtest still runs.
        """
        # Degraded mode — allow all
        if self._ce is None or self._lrm is None:
            return BacktestGateResult(
                allowed=True,
                final_confidence=confidence_base,
                reject_reason="",
                reject_stage="",
                components={"degraded_mode": True},
            )

        bal = balance if balance is not None else self.initial_balance

        # ── Stage 1: ConfidenceEngine ──────────────────────────────
        final_conf = confidence_base
        ce_components = {}
        if self.enable_confidence_engine:
            try:
                ce_result = self._ce.calculate(
                    pattern=pattern,
                    pair=self.symbol,
                    timeframe=self.timeframe,
                    regime=regime,
                    base_confidence=int(confidence_base),
                )
                final_conf = float(ce_result.get("final_confidence", confidence_base))
                ce_components = {
                    "historical": ce_result.get("historical_score"),
                    "recent": ce_result.get("recent_score"),
                    "regime_score": ce_result.get("regime_score"),
                    "bayesian_penalty": ce_result.get("bayesian_penalty"),
                    "sample_size": ce_result.get("sample_size"),
                    "should_skip": ce_result.get("should_skip"),
                    "skip_reason": ce_result.get("skip_reason"),
                }
                # If ConfidenceEngine says skip (win rate < 30%), block
                if ce_result.get("should_skip"):
                    self._record_block("confidence_engine_skip")
                    return BacktestGateResult(
                        allowed=False,
                        final_confidence=final_conf,
                        reject_reason=f"ConfidenceEngine skip: {ce_result.get('skip_reason')}",
                        reject_stage="confidence_engine",
                        components=ce_components,
                    )
            except Exception as e:
                log.debug(f"[BacktestGate] ConfidenceEngine error (non-fatal): {e}")
                final_conf = confidence_base

        # ── Stage 2: Confluence quality gate ───────────────────────
        # In live, trade_permission.py checks setup_quality against
        # BLOCKED_SETUP_QUALITIES. We approximate this in backtest by
        # checking the signal's quality_grade / setup_quality field.
        if self.enable_confluence_gate and self.confluence_avoid_blocks:
            quality = str(signal.get("setup_quality", signal.get("quality_grade", ""))).upper()
            if quality in ("AVOID", "F", "POOR", "BAD"):
                self._record_block("confluence_avoid")
                return BacktestGateResult(
                    allowed=False,
                    final_confidence=final_conf,
                    reject_reason=f"Confluence quality AVOID ({quality})",
                    reject_stage="confluence",
                    components=ce_components,
                )

        # ── Stage 3: LiveRiskManager ───────────────────────────────
        if self.enable_lrm:
            try:
                direction = str(signal.get("signal", signal.get("direction", "HOLD"))).upper()
                if direction in ("LONG",):
                    direction = "BUY"
                elif direction in ("SHORT",):
                    direction = "SELL"

                lrm_result = self._lrm.check_trade_permission(
                    pair=self.symbol,
                    direction=direction,
                    confidence=final_conf,
                    sl_pips=sl_pips,
                    tp_pips=tp_pips,
                    balance=bal,
                    atr=atr,
                    atr_median=atr_median,
                    spread_pips=spread_pips,
                    open_positions=open_positions or [],
                    daily_pnl=daily_pnl,
                    weekly_pnl=weekly_pnl,
                )

                # Normalize result (TradePermission dataclass or dict)
                if hasattr(lrm_result, "allowed"):
                    allowed = bool(lrm_result.allowed)
                    reject_reason = getattr(lrm_result, "reject_reason", "")
                    tier = getattr(lrm_result, "tier", self._lrm.current_tier.tier)
                elif isinstance(lrm_result, dict):
                    allowed = bool(lrm_result.get("allowed", False))
                    reject_reason = lrm_result.get("reject_reason", lrm_result.get("reason", ""))
                    tier = lrm_result.get("tier", self._lrm.current_tier.tier)
                else:
                    allowed = True
                    reject_reason = ""
                    tier = self._lrm.current_tier.tier

                if not allowed:
                    self._record_block("lrm")
                    return BacktestGateResult(
                        allowed=False,
                        final_confidence=final_conf,
                        reject_reason=f"LRM: {reject_reason}",
                        reject_stage="lrm",
                        tier=tier,
                        tier_name=self._lrm.current_tier.name,
                        min_confidence_required=self._lrm.current_tier.min_confidence,
                        components=ce_components,
                    )

                self._allowed_count += 1
                return BacktestGateResult(
                    allowed=True,
                    final_confidence=final_conf,
                    reject_reason="",
                    reject_stage="",
                    tier=tier,
                    tier_name=self._lrm.current_tier.name,
                    min_confidence_required=self._lrm.current_tier.min_confidence,
                    components=ce_components,
                )
            except Exception as e:
                log.debug(f"[BacktestGate] LRM error (non-fatal, allowing): {e}")
                # Fail-open in backtest (don't block the whole backtest
                # on an LRM internal error). Live mode would fail-closed.
                self._allowed_count += 1
                return BacktestGateResult(
                    allowed=True,
                    final_confidence=final_conf,
                    components={**ce_components, "lrm_error": str(e)},
                )

        # LRM disabled — allow
        self._allowed_count += 1
        return BacktestGateResult(
            allowed=True,
            final_confidence=final_conf,
            components=ce_components,
        )

    def record_outcome(
        self,
        pattern: str,
        pair: str,
        timeframe: str,
        regime: str,
        won: bool,
    ) -> None:
        """Record a trade outcome to build up ConfidenceEngine sample_size.

        Call this AFTER a trade closes. This is what makes the backtest
        simulate the real-world bootstrap period: the first few trades
        have no historical data (flat prior), but as trades close and
        outcomes are recorded, the ConfidenceEngine starts adjusting
        confidence based on actual performance.
        """
        if self._ce is None:
            return
        try:
            # ConfidenceEngine may have a record_outcome / record_result method
            if hasattr(self._ce, "record_outcome"):
                self._ce.record_outcome(
                    pattern=pattern,
                    pair=pair,
                    timeframe=timeframe,
                    regime=regime,
                    won=won,
                )
            elif hasattr(self._ce, "record_result"):
                self._ce.record_result(pattern, pair, timeframe, regime, won)
            elif hasattr(self._ce, "update_stats"):
                self._ce.update_stats(pattern, pair, timeframe, regime, won)
        except Exception as e:
            log.debug(f"[BacktestGate] record_outcome error (non-fatal): {e}")

        # Also feed LiveRiskManager's trade-result tracker so tier
        # auto-promotion can fire.
        if self._lrm is not None:
            try:
                if hasattr(self._lrm, "record_trade_result"):
                    self._lrm.record_trade_result(won=won)
            except Exception as e:
                log.debug(f"[BacktestGate] LRM record_trade_result error: {e}")

    def maybe_promote_tier(self, total_closed_trades: int, win_rate: float) -> bool:
        """Check if the tier should be promoted/demoted based on closed-trade stats.

        Returns True if tier changed. Call this periodically (e.g. after
        each trade close, or once per backtest cycle).
        """
        if self._lrm is None:
            return False
        try:
            return self._lrm.maybe_promote_tier(total_closed_trades, win_rate)
        except Exception as e:
            log.debug(f"[BacktestGate] maybe_promote_tier error: {e}")
            return False

    def _record_block(self, stage: str) -> None:
        self._blocked_count += 1
        self._block_reasons[stage] = self._block_reasons.get(stage, 0) + 1

    def stats(self) -> Dict[str, Any]:
        """Return gating statistics for the backtest report."""
        total = self._allowed_count + self._blocked_count
        return {
            "total_signals": total,
            "allowed": self._allowed_count,
            "blocked": self._blocked_count,
            "block_rate": round(self._blocked_count / max(1, total), 3),
            "block_reasons": dict(self._block_reasons),
            "current_tier": self._lrm.current_tier.tier if self._lrm else None,
            "tier_name": self._lrm.current_tier.name if self._lrm else None,
            "degraded_mode": self._ce is None or self._lrm is None,
        }
