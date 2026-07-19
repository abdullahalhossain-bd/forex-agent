"""
core/production_trading_system.py — Unified Production Entry Point

⚠️  STATUS (institutional review, see audit report): NOT CURRENTLY WIRED IN.
    Despite the "ONLY entry point you should use for live trading" claim
    below, nothing in runtime.py / trader.py / trading_engine.py imports
    or calls ProductionTradingSystem. The actual live path registered by
    runtime.py's boot_runtime_phase() is core.trading_engine.TradingEngine
    (which extends AutonomousTraderSystem in core/trader.py).
    Additionally, execute_trade() below still contains a placeholder
    comment ("In real implementation: result = mt5.order_send(...)") and
    simulates a fill — it does not currently submit real broker orders
    even if this class were wired in.
    This file is left in place (not deleted) because it contains real,
    possibly-wanted logic (data quality validation, cognitive-bias
    defenses, adversarial defenses) that may be worth merging into the
    live TradingEngine path in a future change. Until that merge happens,
    treat this module as a design reference / work-in-progress, not a
    production system.
==================================================================

Wires together ALL defenses into a single production-ready trading system:

  LAYER 1 — Data Quality:
    DataQualityValidator (bad tick rejection, spike filter)

  LAYER 2 — Cognitive Bias Defenses:
    PreRegistrationFramework (confirmation bias)
    StrategyGraveyard (survivorship bias)
    CalibrationTracker (overconfidence)
    SelectionAuditLog (selection bias)

  LAYER 3 — Adversarial Defenses:
    BrokerExecutionGuard (last-look protection)
    NewsEventBlackout (news spread widening)
    CrashRecoveryManager (orphaned positions)
    StrategyDegradationMonitor (strategy decay)
    VolatilityScaledSizer (volatility clustering)
    OrderReconciler (state desync)

  LAYER 4 — Risk Management:
    StrictRiskManager (0.5% risk, correlation control, daily/weekly limits)

  LAYER 5 — Decision Engine:
    AdaptiveDecisionEngine (calibrated weights from backtest)
    DecisionBridge (unified → adaptive)

  LAYER 6 — Infrastructure:
    GracefulShutdownManager (signal handlers, cleanup)

Usage:
    from core.production_trading_system import ProductionTradingSystem

    system = ProductionTradingSystem(
        account_equity=10_000,
        is_beginner=True,
        mode="confluence",
    )

    # Main loop
    while system.is_running():
        for pair in system.get_pairs():
            df = system.fetch_data(pair, "H1")
            signal = system.evaluate(pair, df)
            if signal and signal["action"] in ("BUY", "SELL"):
                system.execute_trade(pair, signal)
        system.sleep_until_next_bar()

    # Graceful shutdown (Ctrl+C or SIGTERM)
    system.shutdown()

This is the ONLY entry point you should use for live trading.
"""

from __future__ import annotations

import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger("production")


class ProductionTradingSystem:
    """
    Unified production trading system with all defenses wired.

    This is the single entry point for live trading. It integrates:
      - Data quality validation
      - Cognitive bias defenses (pre-registration, graveyard, calibration)
      - Adversarial defenses (broker guard, news blackout, crash recovery)
      - Strict risk management (0.5% risk, correlation control)
      - Adaptive decision engine (calibrated weights)
      - Graceful shutdown (signal handlers)
    """

    def __init__(
        self,
        account_equity: float = 10_000,
        is_beginner: bool = True,
        mode: str = "confluence",
        state_dir: str = "state",
        pairs: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
    ):
        self.account_equity = account_equity
        self.is_beginner = is_beginner
        self.mode = mode
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.pairs = pairs or ["EURUSD", "GBPUSD", "XAUUSD"]
        self.timeframes = timeframes or ["H1", "H4"]

        self._running = False
        self._lock = threading.RLock()

        # Initialize all defense layers
        self._init_layers()

    def _init_layers(self):
        """Initialize all defense layers."""
        log.info("[Production] Initializing defense layers...")

        # Layer 6: Graceful Shutdown (first, so it can catch everything)
        from core.graceful_shutdown import GracefulShutdownManager
        self.shutdown_mgr = GracefulShutdownManager(timeout_seconds=30.0)

        # Layer 1: Data Quality
        from risk.adversarial_defenses import DataQualityValidator
        self.data_validator = DataQualityValidator(
            spike_atr_mult=5.0,
            volume_floor_pct=0.10,
            max_gap_bars=3,
        )

        # Layer 2: Cognitive Bias Defenses
        from risk.cognitive_bias_defenses import (
            PreRegistrationFramework, StrategyGraveyard,
            CalibrationTracker, SelectionAuditLog,
        )
        self.prereg = PreRegistrationFramework(
            storage_path=str(self.state_dir / "pre_registrations.json"))
        self.graveyard = StrategyGraveyard(
            storage_path=str(self.state_dir / "strategy_graveyard.json"))
        self.calibration = CalibrationTracker()
        self.selection_audit = SelectionAuditLog(
            storage_path=str(self.state_dir / "selection_audit.json"))

        # Layer 3: Adversarial Defenses
        from risk.adversarial_defenses import (
            BrokerExecutionGuard, NewsEventBlackout, CrashRecoveryManager,
            StrategyDegradationMonitor, VolatilityScaledSizer,
            OrderReconciler,
        )
        self.exec_guard = BrokerExecutionGuard(
            max_rejections_per_hour=5,
            max_slippage_pips=5.0,
        )
        self.news_blackout = NewsEventBlackout(
            calendar_path=str(self.state_dir / "economic_calendar.json"))
        self.crash_recovery = CrashRecoveryManager(
            state_dir=str(self.state_dir))
        self.degradation_monitor = StrategyDegradationMonitor(
            rolling_window=50,
            wr_drop_threshold=0.10,
        )
        self.vol_sizer = VolatilityScaledSizer(
            base_risk_pct=0.5,
            crisis_atr_mult=2.5,
        )
        self.reconciler = OrderReconciler(
            poll_interval_seconds=60,
            max_mismatches_before_halt=1,
        )

        # Layer 4: Strict Risk Manager
        from risk.strict_risk_manager import StrictRiskManager
        self.risk_manager = StrictRiskManager(
            account_equity=self.account_equity,
            is_beginner=self.is_beginner,
        )

        # Layer 5: Decision Engine (loaded lazily — requires calibrated_weights.json)
        self._decision_engine = None  # initialized on first use

        # Register cleanup callbacks (LIFO order — last registered runs first)
        self.shutdown_mgr.register_cleanup_callback(self._cleanup_reconciler)
        self.shutdown_mgr.register_cleanup_callback(self._cleanup_state)
        self.shutdown_mgr.register_cleanup_callback(self._cleanup_mt5)

        log.info("[Production] All defense layers initialized")
        log.info(f"  Pairs: {self.pairs}")
        log.info(f"  Timeframes: {self.timeframes}")
        log.info(f"  Mode: {self.mode}")
        log.info(f"  Beginner: {self.is_beginner}")

    # ══════════════════════════════════════════════════════════
    #  MAIN LOOP API
    # ══════════════════════════════════════════════════════════

    def is_running(self) -> bool:
        """Check if the system should keep running."""
        return not self.shutdown_mgr.is_shutting_down()

    def get_pairs(self) -> List[str]:
        """Get the list of pairs to trade."""
        return self.pairs

    def fetch_data(self, pair: str, timeframe: str, n_candles: int = 500) -> Optional[pd.DataFrame]:
        """Fetch and validate OHLCV data for a pair."""
        from backtest.mt5_bulk_fetcher import MT5BulkFetcher
        fetcher = MT5BulkFetcher()
        result = fetcher.fetch(pair, timeframe, n_candles=n_candles)

        if result.df is None or len(result.df) < 50:
            log.warning(f"[Production] Insufficient data for {pair} {timeframe}")
            return None

        # Validate data quality
        summary = self.data_validator.validate_dataframe(result.df, pair)
        if summary["validity_rate"] < 0.95:
            log.warning(f"[Production] Poor data quality for {pair}: "
                       f"{summary['validity_rate']*100:.1f}% valid bars")
            return None

        return result.df

    def evaluate(
        self,
        pair: str,
        df: pd.DataFrame,
        strategy_signals: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate whether to trade. Runs all defense layers.

        Args:
            pair: trading pair
            df: OHLCV dataframe (recent bars)
            strategy_signals: list of signals from strategy engines

        Returns:
            Decision dict with action, confidence, entry, stop, tp
            or None if blocked
        """
        if not self.is_running():
            return None

        # ── LAYER 1: Data Quality ──────────────────────────
        is_valid, reason = self.data_validator.validate_bar(
            df, len(df) - 1, pair)
        if not is_valid:
            log.info(f"[Production] {pair}: blocked by data quality — {reason}")
            return None

        # ── LAYER 3a: News Blackout ────────────────────────
        can_trade, reason = self.news_blackout.can_trade(pair)
        if not can_trade:
            log.info(f"[Production] {pair}: blocked by news blackout — {reason}")
            return None

        # ── LAYER 3b: Broker Health ────────────────────────
        can_submit, reason = self.exec_guard.can_submit()
        if not can_submit:
            log.warning(f"[Production] {pair}: blocked by broker health — {reason}")
            return None

        # ── LAYER 3c: Reconciler Health ────────────────────
        can_trade, reason = self.reconciler.can_trade()
        if not can_trade:
            log.critical(f"[Production] {pair}: blocked by reconciler — {reason}")
            return None

        # ── LAYER 3d: Volatility-Scaled Risk ───────────────
        risk_pct, vol_reason = self.vol_sizer.calculate_risk_pct(df)
        if risk_pct == 0:
            log.warning(f"[Production] {pair}: blocked by volatility — {vol_reason}")
            return None

        # ── LAYER 4: Risk Manager ──────────────────────────
        if not strategy_signals:
            return None

        # Convert strategy signals to internal format
        from analysis.adaptive_decision_engine import StrategySignal
        signals = []
        for s in strategy_signals:
            signals.append(StrategySignal(
                strategy=s.get("strategy", "unknown"),
                action=s.get("action", "NO_TRADE"),
                confidence=s.get("confidence", "Medium"),
                entry_price=s.get("entry_price"),
                stop_loss=s.get("stop_loss"),
                take_profit=s.get("take_profit"),
                r_multiple=s.get("r_rr", 2.0),
            ))

        # ── LAYER 5: Adaptive Decision ─────────────────────
        from analysis.decision_bridge import make_adaptive_decision
        unified_result = {"detected_patterns": [], "consensus": {}}
        for s in strategy_signals:
            key = s.get("strategy", "unknown") + "_result"
            unified_result[key] = {"signal": s}

        current_price = float(df["close"].iloc[-1])
        decision = make_adaptive_decision(
            unified_result, current_price=current_price, mode=self.mode)

        if decision["action"] not in ("BUY", "SELL"):
            log.info(f"[Production] {pair}: decision={decision['action']} — "
                    f"{decision.get('reason', '')}")
            return None

        # Direction
        direction = "long" if decision["action"] == "BUY" else "short"

        # ── LAYER 4 (again): Risk Manager can_open_trade ──
        check = self.risk_manager.can_open_trade(pair, direction)
        if not check.allowed:
            log.info(f"[Production] {pair}: blocked by risk manager — {check.reason}")
            return None

        # ── LAYER 3e: Strategy Degradation ─────────────────
        for sig in signals:
            if sig.action in ("BUY", "SELL"):
                enabled, reason = self.degradation_monitor.is_strategy_enabled(sig.strategy)
                if not enabled:
                    log.info(f"[Production] {pair}: strategy {sig.strategy} disabled — {reason}")
                    return None

        # ── Compute position size ──────────────────────────
        entry = decision.get("entry_price") or current_price
        stop = decision.get("stop_loss")
        if not stop:
            log.warning(f"[Production] {pair}: no stop_loss in decision")
            return None

        # Volatility-scaled risk amount
        risk_amount = self.account_equity * (risk_pct / 100.0)
        # Adjust for current risk_pct (vol-scaled)
        base_risk = self.risk_manager.position_size(entry, stop, pair)
        # Use the more conservative of the two
        actual_risk = min(risk_amount, base_risk)

        log.info(f"[Production] {pair} {direction} APPROVED — "
                f"risk=${actual_risk:.2f} ({risk_pct:.2f}%), "
                f"score={decision.get('score', 0):.2f}, "
                f"agreeing={decision.get('agreeing_strategies', [])}")

        return {
            "action": decision["action"],
            "direction": direction,
            "entry_price": entry,
            "stop_loss": stop,
            "take_profit": decision.get("take_profit"),
            "risk_amount": actual_risk,
            "risk_pct": risk_pct,
            "confidence": decision.get("confidence", "Medium"),
            "score": decision.get("score", 0),
            "agreeing_strategies": decision.get("agreeing_strategies", []),
            "reason": decision.get("reason", ""),
        }

    def execute_trade(self, pair: str, decision: Dict[str, Any]) -> Optional[str]:
        """Execute a trade with all defenses active."""
        if not self.is_running():
            return None

        direction = decision["direction"]
        entry = decision["entry_price"]
        stop = decision["stop_loss"]
        tp = decision.get("take_profit")
        risk_amount = decision["risk_amount"]

        # Compute lot size from risk amount
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            log.error(f"[Production] Invalid stop distance for {pair}")
            return None

        # Simple lot size: risk_amount / stop_distance / contract_size
        # (Real implementation would use PositionSizer)
        lot_size = risk_amount / stop_distance / 100_000  # crude approximation
        lot_size = max(0.01, round(lot_size, 2))  # min 0.01, round to step

        # Crash recovery: log intent BEFORE submitting
        order_id = f"{pair}_{direction}_{int(time.time())}"
        self.crash_recovery.log_order_intent(
            order_id, pair, direction, entry, stop, tp or 0.0, lot_size)

        # Submit to broker (placeholder — real implementation calls MT5)
        log.info(f"[Production] Submitting order {order_id}: "
                f"{pair} {direction} {lot_size} lots @ {entry}")

        # In real implementation:
        # result = mt5.order_send({...})
        # For now, simulate a successful fill
        fill_price = entry  # would be result.price

        # Record execution
        from risk.adversarial_defenses import OrderAttempt
        attempt = OrderAttempt(
            order_id=order_id,
            pair=pair,
            direction=direction,
            intended_price=entry,
            submitted_at=datetime.now(timezone.utc),
            status="filled",
            fill_price=fill_price,
            slippage_pips=abs(fill_price - entry) / 0.0001,
        )
        self.exec_guard.record_attempt(attempt)

        # Crash recovery: confirm fill
        self.crash_recovery.confirm_order_filled(order_id, fill_price)

        # Register with risk manager
        self.risk_manager.register_trade(
            pair, direction, entry, stop, risk_amount)

        # Register with reconciler
        self.reconciler.register_local_position(
            pair, direction, lot_size, entry, stop, tp or 0.0)

        return order_id

    def close_trade(
        self,
        pair: str,
        direction: str,
        pnl_dollars: float,
        pnl_pips: float,
        strategy_name: str = "unknown",
        win: bool = False,
        confidence_at_entry: float = 0.5,
    ):
        """Close a trade and update all tracking systems."""
        # Risk manager
        self.risk_manager.close_trade(pair, direction, pnl_dollars, pnl_pips)

        # Reconciler
        self.reconciler.remove_local_position(pair, direction)

        # Degradation monitor
        r_multiple = pnl_pips / 10.0  # rough R calc
        self.degradation_monitor.record_trade(strategy_name, win, r_multiple)

        # Calibration tracker
        self.calibration.record_prediction(strategy_name, confidence_at_entry, win)

        # If lost and strategy is bad, bury it
        if not win and pnl_dollars < -50:
            # Check if this strategy should be buried
            stats = self.degradation_monitor.get_status()
            if strategy_name in stats:
                strat_stat = stats[strategy_name]
                if strat_stat["rolling_wr"] < 0.35 and strat_stat["rolling_trades"] >= 20:
                    self.graveyard.bury(
                        strategy_name=strategy_name,
                        pair=pair,
                        timeframe="unknown",
                        n_trades=strat_stat["rolling_trades"],
                        win_rate=strat_stat["rolling_wr"],
                        avg_r=strat_stat["rolling_avg_r"],
                        failure_reason=f"WR {strat_stat['rolling_wr']*100:.1f}% < 35%",
                        lessons="Auto-buried after poor rolling performance",
                    )

    # ══════════════════════════════════════════════════════════
    #  CLEANUP CALLBACKS
    # ══════════════════════════════════════════════════════════

    def _cleanup_mt5(self):
        """Close MT5 connection."""
        try:
            from backtest.mt5_bulk_fetcher import MT5_AVAILABLE
            if MT5_AVAILABLE:
                import MetaTrader5 as mt5
                mt5.shutdown()
                log.info("[Production] MT5 connection closed")
        except Exception as e:
            log.error(f"[Production] MT5 cleanup failed: {e}")

    def _cleanup_state(self):
        """Save system state."""
        try:
            state = {
                "account_equity": self.risk_manager.account_equity,
                "peak_equity": self.risk_manager.peak_equity,
                "trade_count": self.risk_manager.trade_count,
                "open_positions": len(self.risk_manager.open_positions),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            self.crash_recovery.save_system_state(state)
            log.info("[Production] State saved")
        except Exception as e:
            log.error(f"[Production] State save failed: {e}")

    def _cleanup_reconciler(self):
        """Stop reconciler thread."""
        try:
            self.reconciler.stop()
            log.info("[Production] Reconciler stopped")
        except Exception as e:
            log.error(f"[Production] Reconciler cleanup failed: {e}")

    # ══════════════════════════════════════════════════════════
    #  STARTUP / SHUTDOWN
    # ══════════════════════════════════════════════════════════

    def startup(self):
        """Called at system startup — reconcile state with broker."""
        log.info("[Production] Starting up...")

        # Load saved state
        saved = self.crash_recovery.load_system_state()
        if saved:
            log.info(f"[Production] Loaded saved state from {saved.get('saved_at')}")
            # Reconcile with broker
            # broker_positions = mt5.positions_get()
            # report = self.crash_recovery.reconcile_on_startup(broker_positions)
            # if report["orphaned_positions"]:
            #     log.critical("ORPHANED POSITIONS — manual intervention required")
            #     return False

        # Start reconciler background thread
        # self.reconciler.set_broker_query_function(mt5.positions_get)
        # self.reconciler.start()

        self._running = True
        log.info("[Production] Startup complete")
        return True

    def graceful_shutdown(self):
        """Graceful shutdown."""
        log.info("[Production] Shutting down...")
        self._running = False
        self.shutdown_mgr.request_shutdown("manual")
        self.shutdown_mgr.wait_for_completion()
        log.info("[Production] Shutdown complete")

    def shutdown(self):
        """Alias for graceful_shutdown (backward compat)."""
        self.graceful_shutdown()

    def get_status(self) -> Dict[str, Any]:
        """Get full system status for monitoring."""
        return {
            "running": self._running,
            "shutdown_state": self.shutdown_mgr.get_state(),
            "risk_manager": {
                "equity": self.risk_manager.account_equity,
                "drawdown_pct": self.risk_manager._drawdown_pct(),
                "open_positions": len(self.risk_manager.open_positions),
                "day_pnl": self.risk_manager.day_pnl,
                "week_pnl": self.risk_manager.week_pnl,
            },
            "exec_guard": self.exec_guard.get_stats(),
            "degradation_monitor": self.degradation_monitor.get_status(),
            "graveyard": self.graveyard.get_summary(),
            "calibration": self.calibration.get_calibration(),
            "prereg_confirmation_rate": self.prereg.get_confirmation_rate(),
        }


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY (smoke test)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("=" * 70)
    print("  PRODUCTION TRADING SYSTEM — Smoke Test")
    print("=" * 70)

    system = ProductionTradingSystem(
        account_equity=10_000,
        is_beginner=True,
        mode="confluence",
        state_dir="/tmp/test_production_state",
        pairs=["EURUSD", "GBPUSD"],
        timeframes=["H1"],
    )

    print("\n  System initialized with all defense layers")
    print(f"  Status: {system.get_status()}")

    # Simulate a trade evaluation
    print("\n  Simulating trade evaluation...")
    import numpy as np
    import pandas as pd
    np.random.seed(42)
    n = 100
    # Use realistic prices around 1.0850 (no negative)
    base = 1.0850
    closes = base + np.cumsum(np.random.randn(n) * 0.0005)
    df = pd.DataFrame({
        "open": closes - np.random.randn(n) * 0.0002,
        "high": closes + np.abs(np.random.randn(n)) * 0.0008,
        "low": closes - np.abs(np.random.randn(n)) * 0.0008,
        "close": closes,
        "volume": np.random.randint(100, 1000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

    # Simulate a strategy signal
    signals = [{
        "strategy": "stop_hunt",
        "action": "BUY",
        "confidence": "High",
        "entry_price": 1.0850,
        "stop_loss": 1.0820,
        "take_profit": 1.0910,
        "r_rr": 2.0,
    }]

    decision = system.evaluate("EURUSD", df, signals)
    if decision:
        print(f"\n  Decision: {decision['action']} {decision['direction']}")
        print(f"  Entry: {decision['entry_price']}")
        print(f"  Stop: {decision['stop_loss']}")
        print(f"  Risk: ${decision['risk_amount']:.2f} ({decision['risk_pct']:.2f}%)")
        print(f"  Confidence: {decision['confidence']}")
        print(f"  Score: {decision['score']:.2f}")

        # Execute (simulated)
        order_id = system.execute_trade("EURUSD", decision)
        print(f"\n  Order executed: {order_id}")

        # Close trade (simulated win)
        system.close_trade(
            "EURUSD", decision["direction"],
            pnl_dollars=30.0, pnl_pips=3.0,
            strategy_name="stop_hunt", win=True,
            confidence_at_entry=0.8,
        )
        print(f"  Trade closed: +$30.00 (win)")
    else:
        print("\n  No trade decision made")

    print(f"\n  Final status: {system.get_status()}")

    # Graceful shutdown
    print("\n  Shutting down...")
    system.shutdown()

    print("\n" + "=" * 70)
    print("  Production system test complete.")
    print("=" * 70)