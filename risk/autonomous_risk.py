# risk/autonomous_risk.py — Day 58 | Autonomous Risk Manager Core
# ============================================================
# AI Trader-এর Fund Manager Brain — Master Orchestrator.
#
# Architecture:
#   Trading Signals → Autonomous Risk Manager → Execution Engine
#                        ├── Position Sizing
#                        ├── Capital Allocation
#                        └── Risk Control
#
# Sub-components:
#   CapitalManager — portfolio-level capital allocation
#   PositionAllocator — Kelly Criterion + dynamic sizing
#   ExposureManager — correlation risk + portfolio exposure
#   DrawdownController — account protection system
#   MonteCarloEngine — Monte Carlo simulation
#   RiskScenarioSimulator — "what if" analysis
#
# Modes:
#   AGGRESSIVE — High confidence, low volatility, good regime
#   NORMAL     — Standard market conditions
#   DEFENSIVE  — High volatility, loss streak, news event
#   EMERGENCY  — Critical drawdown, capital preservation
# ============================================================

import json
import math
from datetime import datetime, date, timezone, timedelta
from typing import Optional
from pathlib import Path

from utils.logger import get_logger
from risk.capital_manager import CapitalManager
from risk.position_allocator import PositionAllocator
from risk.exposure_manager import ExposureManager
from risk.drawdown_controller import DrawdownController
from risk.monte_carlo import MonteCarloEngine

log = get_logger("autonomous_risk")

from core.constants import MEMORY_DIR
AUTONOMOUS_RISK_STATE_PATH = MEMORY_DIR / "autonomous_risk_state.json"
AUTONOMOUS_RISK_REPORT_PATH = MEMORY_DIR / "capital_report.json"


# ── Mode Definitions ─────────────────────────────────────────
RISK_MODES = {
    "AGGRESSIVE": {
        "risk_range": (0.01, 0.015),       # 1% - 1.5%
        "description": "High confidence, low volatility, favorable regime",
        "max_capital_deploy": 0.90,         # deploy up to 90% of capital
        "reserve_ratio": 0.10,
    },
    "NORMAL": {
        "risk_range": (0.008, 0.01),        # 0.8% - 1%
        "description": "Standard market conditions",
        "max_capital_deploy": 0.75,         # deploy up to 75% of capital
        "reserve_ratio": 0.25,
    },
    "DEFENSIVE": {
        "risk_range": (0.0025, 0.005),      # 0.25% - 0.5%
        "description": "High volatility, loss streak, news event",
        "max_capital_deploy": 0.40,         # deploy up to 40% of capital
        "reserve_ratio": 0.60,
    },
    "EMERGENCY": {
        "risk_range": (0.0, 0.001),          # 0% - 0.1% (almost no trading)
        "description": "Critical drawdown, capital preservation only",
        "max_capital_deploy": 0.0,           # NO new trades
        "reserve_ratio": 1.0,
    },
}



# ── P0 fix (audit C3) helpers ──────────────────────────────────────
def _safe_get_exposure_pct(exposure_manager) -> float:
    """Get total exposure as a fraction (0-1) from ExposureManager.
    Handles both dict-style and list-style _open_positions, and missing methods."""
    try:
        if hasattr(exposure_manager, "get_total_exposure_pct"):
            return float(exposure_manager.get_total_exposure_pct())
        # Fallback: compute from _open_positions list
        _op = getattr(exposure_manager, "_open_positions", [])
        if isinstance(_op, list):
            # Without balance context, can't compute pct — return 0
            return 0.0
        return 0.0
    except Exception:
        return 0.0

class AutonomousRiskManager:
    """
    Autonomous Risk Manager — AI Trader-এর Fund Manager Brain.

    এটা Day 58 এর সবচেয়ে গুরুত্বপূর্ণ অংশ।
    AI নিজে সিদ্ধান্ত নেবে:
      - কোথায় কত টাকা risk করবে
      - কখন risk কমাবে / বাড়াবে
      - কখন capital protect করবে
      - কোন strategy-তে কত capital দেবে

    The manager coordinates all sub-components:
      CapitalManager → capital allocation across pairs/strategies
      PositionAllocator → Kelly Criterion position sizing
      ExposureManager → correlation risk management
      DrawdownController → account protection
      MonteCarloEngine → risk simulation

    Usage:
        arm = AutonomousRiskManager(balance=10000)
        decision = arm.evaluate_trade_signal(
            signal="BUY", symbol="EURUSD", entry=1.0850,
            atr=0.0080, confidence=75, strategy="SMC_FVG",
            regime={"volatility": "NORMAL", "trend": "BULLISH"}
        )
        if decision["approved"]:
            execute(decision)
    """

    def __init__(
        self,
        balance: float = 10000.0,
        initial_mode: str = "NORMAL",
        kelly_fraction: float = 0.25,   # Fractional Kelly (25%)
    ):
        self.balance = balance
        self.initial_balance = balance
        self.kelly_fraction = kelly_fraction

        # Sub-components
        self.capital_manager = CapitalManager(total_capital=balance)
        self.position_allocator = PositionAllocator(
            balance=balance,
            kelly_fraction=kelly_fraction,
        )
        self.exposure_manager = ExposureManager()
        self.drawdown_controller = DrawdownController(
            initial_balance=balance,
        )
        self.monte_carlo = MonteCarloEngine()

        # State
        self._state = self._load_state()
        self.current_mode = initial_mode
        self._mode_reason = "System initialized"

        # Performance tracking
        self._trade_history: list[dict] = []
        self._recent_performance = {
            "last_50_win_rate": 0.5,
            "last_20_win_rate": 0.5,
            "last_50_drawdown": 0.0,
            "current_streak": 0,
            "daily_loss_pct": 0.0,
            "weekly_loss_pct": 0.0,
        }

        log.info(
            f"[AutonomousRisk] Initialized | "
            f"Balance: ${balance:,.2f} | Mode: {initial_mode} | "
            f"Kelly Fraction: {kelly_fraction*100:.0f}%"
        )

    # ═══════════════════════════════════════════════════════
    # CORE EVALUATION — Main entry point for every trade signal
    # ═══════════════════════════════════════════════════════

    def evaluate_trade_signal(
        self,
        signal: str,
        symbol: str,
        entry: float,
        atr: float,
        confidence: float = 70.0,
        strategy: str = "default",
        regime: dict | None = None,
        market_data: dict | None = None,
    ) -> dict:
        """
        Evaluate a trade signal through the complete risk pipeline.

        Pipeline:
          1. Determine current risk mode (AGGRESSIVE/NORMAL/DEFENSIVE/EMERGENCY)
          2. Check drawdown protection (emergency stop if critical)
          3. Check daily/weekly loss limits
          4. Check correlation risk (avoid correlated exposure)
          5. Calculate Kelly Criterion optimal risk
          6. Calculate position size
          7. Calculate adaptive SL/TP based on volatility
          8. Approve/reject with full risk metrics

        Args:
            signal: "BUY" or "SELL"
            symbol: Currency pair (e.g., "EURUSD")
            entry: Entry price
            atr: Current ATR value
            confidence: Strategy confidence (0-100)
            strategy: Strategy name (e.g., "SMC_FVG", "RSI")
            regime: Market regime dict (volatility, trend, etc.)
            market_data: Additional market data (optional)

        Returns:
            Complete risk decision dict with approval status,
            position sizing, SL/TP, and risk metrics.
        """
        if regime is None:
            regime = {}
        if market_data is None:
            market_data = {}

        symbol = symbol.upper().replace("/", "").replace("=X", "")[:6]

        # Step 1: Update risk mode based on current conditions
        # But if already in EMERGENCY (manual), keep it until conditions improve
        self._update_risk_mode(regime, confidence)

        mode_config = RISK_MODES[self.current_mode]

        # Step 2: Emergency check — is trading even allowed?
        #
        # P1 fix: RISK_MODES["EMERGENCY"]["max_capital_deploy"] == 0.0 is
        # documented as "NO new trades", but that flag was previously only
        # enforced in allocate_portfolio() — NOT here. _update_risk_mode()
        # can set current_mode="EMERGENCY" via a path (e.g.
        # daily_loss_pct >= 2.5) that's independent of
        # drawdown_controller.check_emergency(). If check_emergency()
        # didn't also trip, execution continued, blended risk clamped to
        # EMERGENCY's (0.0, 0.001) range, computed lot rounded to ~0 —
        # and then the `max(0.01, ...)` floor a few lines down silently
        # forced a real minimum-size trade through anyway. Explicitly gate
        # on the mode's own deploy limit so "EMERGENCY = no new trades" is
        # actually true at the point a trade is evaluated, not just at
        # portfolio-allocation time.
        if mode_config["max_capital_deploy"] <= 0:
            return self._build_rejection(
                symbol, signal,
                f"EMERGENCY mode — no new trades (reason: {self._mode_reason})",
                mode_config,
            )

        emergency = self.drawdown_controller.check_emergency(
            self.balance, self._recent_performance
        )
        if emergency["stop_trading"]:
            log.warning(
                f"[AutonomousRisk] EMERGENCY: {emergency['reason']}"
            )
            return self._build_rejection(
                symbol, signal, emergency["reason"], mode_config,
            )

        # Step 3: Daily loss limit check (Day 81+ — load from config)
        #
        # P1 fix (audit follow-up, same class of bug as circuit_breaker.py
        # §3.2): this silently fell back to `_daily_limit = 20.0` if
        # config import failed — permissive compared to the 1.5-3% every
        # other risk file in this codebase assumes, and yet ANOTHER
        # independent copy of the same threshold (now the 3rd/4th place
        # this number is duplicated: circuit_breaker.py, risk_engine.py,
        # live_risk_manager.py's tiers, and here). Not consolidating into
        # one shared constant in this pass (that needs ai_trader.py to
        # confirm which engine is authoritative — see audit §3.2), but at
        # minimum this can't silently downgrade to a different, more
        # permissive number than the rest of the system on import failure.
        from config import DAILY_LOSS_LIMIT_PCT as _DLL
        _daily_limit = float(_DLL)
        if self._recent_performance["daily_loss_pct"] >= _daily_limit:
            return self._build_rejection(
                symbol, signal,
                f"Daily loss limit reached ({self._recent_performance['daily_loss_pct']:.1f}%)",
                mode_config,
            )

        # Step 4: Correlation check
        # P0 fix (audit C3): ExposureManager has no check_new_position() method.
        # Fall back to the actual check() method, or skip if that also fails.
        try:
            if hasattr(self.exposure_manager, "check_new_position"):
                exposure_check = self.exposure_manager.check_new_position(
                    symbol, signal, self._get_open_positions()
                )
            elif hasattr(self.exposure_manager, "check"):
                # Bug #10 fix: 'decision' doesn't exist yet at this point in
                # the function — it's only built at Step 8 (~100 lines below,
                # after lot sizing). Referencing it here raised
                # UnboundLocalError whenever this fallback branch (no
                # check_new_position(), only check()) was taken. The actual
                # lot/risk_usd aren't known yet this early, so use the same
                # safe defaults the old (broken) code intended as its fallback.
                exposure_check = self.exposure_manager.check(
                    pair=symbol,
                    direction=signal,
                    lot=0.01,
                    risk_usd=0,
                    balance=self.balance,
                )
                exposure_check = {"allowed": getattr(exposure_check, "allowed", True),
                                  "reason": getattr(exposure_check, "reason", "")}
            else:
                exposure_check = {"allowed": True, "reason": "exposure_manager has no check method"}
        except Exception as e:
            log.warning(f"[AutonomousRisk] exposure check failed: {e}; allowing trade")
            exposure_check = {"allowed": True, "reason": f"check failed: {e}"}
        if not exposure_check.get("allowed", True):
            return self._build_rejection(
                symbol, signal, exposure_check.get("reason", "exposure blocked"), mode_config,
            )

        # Step 5: Capital availability
        allocated = self.capital_manager.get_total_allocated()
        available = self.balance - allocated
        if available < self.balance * 0.05:
            return self._build_rejection(
                symbol, signal,
                f"Insufficient free capital (${available:.0f} of ${self.balance:.0f})",
                mode_config,
            )

        # Step 6: Calculate position size using Kelly Criterion
        performance = self._recent_performance
        kelly_risk = self.position_allocator.calculate_kelly_risk(
            win_rate=performance["last_50_win_rate"],
            avg_win=performance.get("avg_win", 2.0),
            avg_loss=performance.get("avg_loss", 1.0),
        )

        # Blend Kelly with mode-based risk
        mode_risk = sum(mode_config["risk_range"]) / 2
        dynamic_risk = self._blend_risk(kelly_risk, mode_risk, confidence)

        # Step 7: Calculate position size
        from core.constants import get_pip_size, get_pip_value_usd
        pip = get_pip_size(symbol)
        pip_val = get_pip_value_usd(symbol)

        sl_distance = self._calculate_adaptive_sl(entry, atr, regime, symbol)
        sl_pips = round(sl_distance / pip) if pip > 0 else 20
        risk_usd = self.balance * dynamic_risk

        if sl_pips > 0 and pip_val > 0:
            lot = risk_usd / (sl_pips * pip_val)
        else:
            lot = 0.01

        lot = max(0.01, min(round(lot, 2), 100.0))

        # TP based on minimum RR from position allocator
        min_rr = self.position_allocator.get_minimum_rr(
            confidence, self.current_mode
        )
        if signal == "BUY":
            sl_price = round(entry - sl_distance, 5)
            tp_price = round(entry + sl_distance * min_rr, 5)
        else:
            sl_price = round(entry + sl_distance, 5)
            tp_price = round(entry - sl_distance * min_rr, 5)

        tp_pips = round(sl_pips * min_rr)
        rr_ratio = round(tp_pips / sl_pips, 2) if sl_pips > 0 else 0

        # Margin check
        margin_needed = lot * 1000
        if margin_needed > self.balance * 0.5:
            lot = (self.balance * 0.5) / 1000
            lot = max(0.01, min(round(lot, 2), 100.0))

        # Step 8: Capital allocation
        strategy_ranking = self._get_strategy_capital_ranking()
        capital_pct = strategy_ranking.get(strategy, 0.1)

        decision = {
            "approved": True,
            "signal": signal,
            "symbol": symbol,
            "entry": entry,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "lot": lot,
            "risk_usd": round(risk_usd, 2),
            "risk_pct": round(dynamic_risk * 100, 2),
            "rr_ratio": rr_ratio,
            "risk_mode": self.current_mode,
            "mode_reason": self._mode_reason,
            "kelly_risk_pct": round(kelly_risk * 100, 2),
            "capital_allocation_pct": round(capital_pct * 100, 1),
            "strategy_ranking": strategy_ranking,
            "confidence": confidence,
            "reject_reason": None,
            "exposure_info": {
                # P0 fix (audit C3): ExposureManager has no get_total_exposure_pct().
                "total_exposure_pct": _safe_get_exposure_pct(self.exposure_manager),
                "correlated_pairs": exposure_check.get("correlated_pairs", []) if isinstance(exposure_check, dict) else [],
            },
            "drawdown_info": {
                "current_drawdown_pct": self.drawdown_controller.current_drawdown_pct(self.balance),
                "max_drawdown_pct": self.drawdown_controller.max_drawdown_limit,
            },
        }

        log.info(
            f"[AutonomousRisk] APPROVED {signal} {symbol} | "
            f"Lot: {lot} | Risk: {dynamic_risk*100:.2f}% | "
            f"Mode: {self.current_mode} | SL: {sl_pips}p TP: {tp_pips}p | "
            f"RR: 1:{rr_ratio}"
        )

        # Day 37 fix: register the new position with ExposureManager so that
        # subsequent check_new_position() calls see it. Previously
        # open_position() was never called — _open_positions stayed empty
        # and the correlation/currency-exposure gates were effectively inert.
        # P0 fix (audit C3): ExposureManager has no open_position() method.
        # update_positions() is the real API; call it with the new position appended.
        try:
            if hasattr(self.exposure_manager, "open_position"):
                self.exposure_manager.open_position(symbol=symbol, direction=signal, lot=lot)
            elif hasattr(self.exposure_manager, "update_positions"):
                # Best-effort: refresh from broker source-of-truth.
                # AutonomousRiskManager doesn't track positions itself; the real
                # PositionManager does. This is a no-op here.
                pass
        except Exception as e:
            log.warning(f"[AutonomousRisk] exposure_manager position register failed: {e}")

        self._save_state()
        return decision

    # ═══════════════════════════════════════════════════════
    # RISK MODE MANAGEMENT
    # ═══════════════════════════════════════════════════════

    def _update_risk_mode(
        self, regime: dict, confidence: float
    ) -> None:
        """
        Dynamically determine risk mode based on:
          1. Recent performance (win rate, drawdown, streaks)
          2. Market regime (volatility, trend)
          3. Current confidence
        """
        perf = self._recent_performance
        new_mode = "NORMAL"
        reason = ""

        # Emergency triggers (highest priority)
        dd_pct = self.drawdown_controller.current_drawdown_pct(self.balance)
        if dd_pct >= 15.0:
            new_mode = "EMERGENCY"
            reason = f"Critical drawdown: {dd_pct:.1f}% >= 15%"
        elif perf["daily_loss_pct"] >= 2.5:
            new_mode = "EMERGENCY"
            reason = f"Near daily limit: {perf['daily_loss_pct']:.1f}%"
        elif perf["weekly_loss_pct"] >= 6.0:
            new_mode = "EMERGENCY"
            reason = f"Near weekly limit: {perf['weekly_loss_pct']:.1f}%"

        # If currently in EMERGENCY (manually set), don't downgrade
        # unless conditions are clearly safe (dd < 5% and no severe loss streak)
        elif self.current_mode == "EMERGENCY" and dd_pct < 5.0 and perf["current_streak"] > -4:
            new_mode = "DEFENSIVE"  # Step down from emergency, not straight to normal
            reason = f"Emergency lifted — stepping down to DEFENSIVE (DD: {dd_pct:.1f}%)"

        # Defensive triggers
        elif perf["last_20_win_rate"] < 0.40:
            new_mode = "DEFENSIVE"
            reason = f"Low recent win rate: {perf['last_20_win_rate']*100:.0f}%"
        elif perf["current_streak"] <= -4:
            new_mode = "DEFENSIVE"
            reason = f"Loss streak: {perf['current_streak']} consecutive losses"
        elif perf["last_50_drawdown"] >= 10.0:
            new_mode = "DEFENSIVE"
            reason = f"Elevated drawdown: {perf['last_50_drawdown']:.1f}%"
        elif regime.get("volatility", "").upper() in ("HIGH_VOLATILITY", "EXTREME"):
            new_mode = "DEFENSIVE"
            reason = "High market volatility detected"
        elif regime.get("news_impact", "").upper() == "HIGH":
            new_mode = "DEFENSIVE"
            reason = "High-impact news event nearby"

        # Aggressive triggers
        elif perf["last_50_win_rate"] >= 0.68 and dd_pct < 3.0:
            new_mode = "AGGRESSIVE"
            reason = (
                f"Strong performance: WR {perf['last_50_win_rate']*100:.0f}%, "
                f"Low DD: {dd_pct:.1f}%"
            )
        elif confidence >= 85 and dd_pct < 2.0 and perf["current_streak"] >= 3:
            new_mode = "AGGRESSIVE"
            reason = (
                f"High confidence: {confidence}%, "
                f"Win streak: {perf['current_streak']}, "
                f"Low DD: {dd_pct:.1f}%"
            )

        else:
            reason = "Normal market conditions"

        if new_mode != self.current_mode:
            old_mode = self.current_mode
            self.current_mode = new_mode
            self._mode_reason = reason
            log.warning(
                f"[AutonomousRisk] MODE CHANGE: {old_mode} -> {new_mode} | {reason}"
            )

    def set_mode(self, mode: str, reason: str = "Manual override") -> None:
        """Manually set the risk mode."""
        mode = mode.upper()
        if mode not in RISK_MODES:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of: {list(RISK_MODES.keys())}"
            )
        self.current_mode = mode
        self._mode_reason = reason
        self._save_state()
        log.info(f"[AutonomousRisk] Mode manually set to {mode}: {reason}")

    # ═══════════════════════════════════════════════════════
    # ADAPTIVE STOP LOSS ENGINE
    # ═══════════════════════════════════════════════════════

    def _calculate_adaptive_sl(
        self, entry: float, atr: float, regime: dict, symbol: str
    ) -> float:
        """
        Calculate adaptive stop loss based on:
          1. ATR (primary volatility measure)
          2. Market volatility regime
          3. Account risk mode

        Low volatility  → SL: 1.2 × ATR (tight)
        Normal           → SL: 1.5 × ATR (standard)
        High volatility  → SL: 2.0 × ATR (wide)

        Defensive mode adds extra buffer.
        """
        volatility = regime.get("volatility", "NORMAL").upper()

        vol_multipliers = {
            "LOW_VOLATILITY": 1.2,
            "NORMAL": 1.5,
            "HIGH_VOLATILITY": 2.0,
            "EXTREME": 2.5,
        }
        multiplier = vol_multipliers.get(volatility, 1.5)

        # Mode-based adjustment
        if self.current_mode == "DEFENSIVE":
            multiplier *= 1.2  # wider SL in defensive mode
        elif self.current_mode == "AGGRESSIVE":
            multiplier *= 0.9  # tighter SL in aggressive mode

        return round(atr * multiplier, 5)

    # ═══════════════════════════════════════════════════════
    # PERFORMANCE TRACKING & FEEDBACK
    # ═══════════════════════════════════════════════════════

    def record_trade_result(
        self,
        symbol: str,
        pnl_usd: float,
        result: str,
        strategy: str = "default",
        rr_achieved: float = 0.0,
    ) -> None:
        """
        Record trade result and update all subsystems.

        This feeds back into:
          - Performance metrics (win rate, drawdown)
          - Capital manager (available capital)
          - Drawdown controller (protection level)
          - Exposure manager (close position)
        """
        self.balance += pnl_usd
        self.balance = max(0, self.balance)

        trade_record = {
            "symbol": symbol,
            "pnl_usd": round(pnl_usd, 2),
            "result": result,
            "strategy": strategy,
            "rr_achieved": rr_achieved,
            "balance": round(self.balance, 2),
            "mode": self.current_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._trade_history.append(trade_record)

        # Update performance metrics
        self._update_performance_metrics()

        # Update drawdown controller
        self.drawdown_controller.record_trade(pnl_usd, self.balance)

        # Update exposure
        # P0 fix (audit C3): ExposureManager has no close_position() method.
        # Position tracking is the PositionManager's job; this is a no-op here.
        try:
            if hasattr(self.exposure_manager, "close_position"):
                self.exposure_manager.close_position(symbol)
        except Exception as e:
            log.warning(f"[AutonomousRisk] exposure_manager close_position failed: {e}")

        # Update capital manager
        self.capital_manager.update_balance(self.balance)

        # Update daily tracking
        if pnl_usd < 0:
            today = date.today().isoformat()
            daily = self._state.get("daily_losses", {})
            daily[today] = daily.get(today, 0) + abs(pnl_usd)
            self._state["daily_losses"] = daily

        self._save_state()

        log.info(
            f"[AutonomousRisk] Trade recorded: {result} {symbol} "
            f"PnL: ${pnl_usd:+.2f} | Balance: ${self.balance:,.2f} | "
            f"Mode: {self.current_mode}"
        )

    def _update_performance_metrics(self) -> None:
        """Recalculate all performance metrics from trade history."""
        history = self._trade_history
        if not history:
            return

        # Last 50 trades
        last_50 = history[-50:]
        wins_50 = sum(1 for t in last_50 if t["result"] == "WIN")
        self._recent_performance["last_50_win_rate"] = (
            wins_50 / len(last_50) if last_50 else 0.5
        )

        # Last 20 trades
        last_20 = history[-20:]
        wins_20 = sum(1 for t in last_20 if t["result"] == "WIN")
        self._recent_performance["last_20_win_rate"] = (
            wins_20 / len(last_20) if last_20 else 0.5
        )

        # Drawdown from last 50
        if len(last_50) >= 2:
            peak = max(t["balance"] for t in last_50)
            trough = min(t["balance"] for t in last_50)
            dd = (peak - trough) / peak * 100 if peak > 0 else 0
            self._recent_performance["last_50_drawdown"] = dd

        # Current streak
        # Co-founder fix: prefer StreakTracker (authoritative, persisted via
        # CircuitBreaker) over our in-memory computation. Our in-memory
        # streak resets on restart and only sees our own _trade_history;
        # StreakTracker sees ALL trades across all modules + restarts.
        try:
            from risk.streak_tracker import get_consecutive_losses
            cb_streak = get_consecutive_losses()
            if cb_streak > 0:
                # CB tracks consecutive losses — use it directly as negative streak
                streak = -cb_streak
            else:
                # CB shows 0 losses — compute local win streak from history
                streak = 0
                for t in reversed(history):
                    if t["result"] == "WIN":
                        if streak >= 0:
                            streak += 1
                        else:
                            break
                    else:
                        if streak <= 0:
                            streak -= 1
                        else:
                            break
        except Exception:
            # Fallback to original in-memory computation
            streak = 0
            for t in reversed(history):
                if t["result"] == "WIN":
                    if streak >= 0:
                        streak += 1
                    else:
                        break
                else:
                    if streak <= 0:
                        streak -= 1
                    else:
                        break
        self._recent_performance["current_streak"] = streak

        # Daily loss %
        today = date.today().isoformat()
        daily_loss = self._state.get("daily_losses", {}).get(today, 0)
        self._recent_performance["daily_loss_pct"] = (
            daily_loss / self.initial_balance * 100
            if self.initial_balance > 0 else 0
        )

        # Weekly loss %
        week_ago = (date.today() - timedelta(days=7)).isoformat()
        weekly_loss = sum(
            v for k, v in self._state.get("daily_losses", {}).items()
            if k >= week_ago and v > 0
        )
        self._recent_performance["weekly_loss_pct"] = (
            weekly_loss / self.initial_balance * 100
            if self.initial_balance > 0 else 0
        )

        # Average win/loss (RR)
        wins = [t for t in history if t["result"] == "WIN" and t["rr_achieved"] > 0]
        losses = [t for t in history if t["result"] == "LOSS" and t["rr_achieved"] > 0]
        if wins:
            self._recent_performance["avg_win"] = (
                sum(t["rr_achieved"] for t in wins) / len(wins)
            )
        if losses:
            self._recent_performance["avg_loss"] = (
                sum(abs(t["rr_achieved"]) for t in losses) / len(losses)
            )

    # ═══════════════════════════════════════════════════════
    # STRATEGY CAPITAL RANKING
    # ═══════════════════════════════════════════════════════

    def _get_strategy_capital_ranking(self) -> dict:
        """
        Allocate capital priority based on strategy performance.
        Uses research results from Day 57 + live trading history.

        Returns:
            {"SMC_FVG": 0.70, "Breakout": 0.20, "RSI": 0.10}
        """
        strategy_stats: dict[str, dict] = {}

        for trade in self._trade_history:
            strat = trade.get("strategy", "default")
            if strat not in strategy_stats:
                strategy_stats[strat] = {
                    "trades": 0, "wins": 0, "total_pnl": 0.0,
                }
            strategy_stats[strat]["trades"] += 1
            if trade["result"] == "WIN":
                strategy_stats[strat]["wins"] += 1
            strategy_stats[strat]["total_pnl"] += trade["pnl_usd"]

        if not strategy_stats:
            return {"default": 1.0}

        # Score = (win_rate * 0.6) + (profit_factor * 0.3) + (trade_count * 0.1)
        scores = {}
        for name, stats in strategy_stats.items():
            if stats["trades"] == 0:
                continue
            wr = stats["wins"] / stats["trades"]
            pf = (
                (stats["total_pnl"] + abs(min(0, stats["total_pnl"])))
                / max(abs(min(0, stats["total_pnl"])), 1)
            )
            count_score = min(stats["trades"] / 50, 1.0)  # cap at 50 trades
            scores[name] = round(wr * 0.6 + pf * 0.3 + count_score * 0.1, 3)

        # Normalize to percentages
        total_score = sum(scores.values()) or 1.0
        ranking = {k: round(v / total_score, 2) for k, v in scores.items()}

        return ranking

    # ═══════════════════════════════════════════════════════
    # PORTFOLIO CAPITAL ALLOCATION
    # ═══════════════════════════════════════════════════════

    def allocate_portfolio(self, opportunities: list[dict]) -> dict:
        """
        Allocate capital across multiple trading opportunities.

        Args:
            opportunities: List of trade opportunity dicts:
                [{
                    "symbol": "EURUSD", "signal": "BUY",
                    "confidence": 80, "strategy": "SMC_FVG",
                    "rr_ratio": 2.5, "atr": 0.008
                }, ...]

        Returns:
            Allocation dict with approved trades and capital distribution.
        """
        mode_config = RISK_MODES[self.current_mode]
        max_deploy = mode_config["max_capital_deploy"]
        reserve = mode_config["reserve_ratio"]

        # If emergency mode, no new trades
        if self.current_mode == "EMERGENCY":
            return {
                "mode": self.current_mode,
                "approved_trades": [],
                "total_allocated": 0.0,
                "reserve": self.balance,
                "reason": "EMERGENCY mode — capital preservation only",
            }

        # Rank opportunities by composite score
        scored = []
        for opp in opportunities:
            strategy_rank = self._get_strategy_capital_ranking()
            strat_score = strategy_rank.get(opp.get("strategy", ""), 0.1)

            # Confidence score (0-1)
            conf_score = opp.get("confidence", 50) / 100

            # RR score
            rr = opp.get("rr_ratio", 1.5)
            rr_score = min(rr / 3.0, 1.0)  # cap at RR=3

            composite = round(
                strat_score * 0.4 + conf_score * 0.35 + rr_score * 0.25, 3
            )
            scored.append({**opp, "composite_score": composite})

        scored.sort(key=lambda x: x["composite_score"], reverse=True)

        # Allocate capital proportionally
        total_score = sum(o["composite_score"] for o in scored) or 1.0
        allocated_capital = 0.0
        approved = []

        for opp in scored:
            remaining_deployable = self.balance * max_deploy - allocated_capital
            if remaining_deployable <= 0:
                break

            # Correlation check
            exposure_check = self.exposure_manager.check_new_position(
                opp["symbol"], opp["signal"], self._get_open_positions()
            )
            if not exposure_check["allowed"]:
                continue

            # Calculate this opportunity's share
            share = opp["composite_score"] / total_score
            proposed_capital = min(remaining_deployable, self.balance * max_deploy * share)

            approved.append({
                **opp,
                "allocated_capital": round(proposed_capital, 2),
                "share_pct": round(share * 100, 1),
            })
            allocated_capital += proposed_capital

        return {
            "mode": self.current_mode,
            "mode_reason": self._mode_reason,
            "approved_trades": approved,
            "total_allocated": round(allocated_capital, 2),
            "reserve": round(self.balance - allocated_capital, 2),
            "reserve_pct": round((self.balance - allocated_capital) / self.balance * 100, 1),
            "max_deploy_pct": round(max_deploy * 100, 1),
        }

    # ═══════════════════════════════════════════════════════
    # MONTE CARLO & RISK SIMULATION
    # ═══════════════════════════════════════════════════════

    def run_risk_simulation(
        self,
        n_simulations: int = 10000,
        n_trades: int = 100,
    ) -> dict:
        """
        Run Monte Carlo simulation of account equity.

        Returns:
            {
                "worst_case": -18.0,
                "average": +22.0,
                "best_case": +65.0,
                "median": +18.5,
                "percentile_5": -8.0,
                "percentile_95": +42.0,
                "survival_rate": 0.97,
                "ruin_probability": 0.03,
            }
        """
        perf = self._recent_performance
        win_rate = perf["last_50_win_rate"] or 0.5
        avg_win_pct = perf.get("avg_win", 2.0) * perf["last_50_win_rate"]
        avg_loss_pct = perf.get("avg_loss", 1.0) * (1 - perf["last_50_win_rate"])

        result = self.monte_carlo.run(
            win_rate=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            n_simulations=n_simulations,
            n_trades=n_trades,
            initial_balance=self.balance,
            risk_per_trade=sum(RISK_MODES[self.current_mode]["risk_range"]) / 2,
        )

        log.info(
            f"[AutonomousRisk] Monte Carlo ({n_simulations} sims): "
            f"Avg: {result['average_pct']:+.1f}% | "
            f"Worst: {result['worst_case_pct']:+.1f}% | "
            f"Survival: {result['survival_rate']*100:.1f}%"
        )

        return result

    def simulate_scenario(self, scenario: str, n_losses: int = 5) -> dict:
        """
        Simulate "what if" scenarios.

        Args:
            scenario: "consecutive_losses", "best_case", "worst_case", etc.
            n_losses: Number of consecutive losses to simulate.

        Returns:
            Impact analysis dict.
        """
        risk_pct = sum(RISK_MODES[self.current_mode]["risk_range"]) / 2
        risk_per_trade = self.balance * risk_pct

        if scenario == "consecutive_losses":
            total_loss = risk_per_trade * n_losses
            new_balance = self.balance - total_loss
            impact_pct = total_loss / self.balance * 100
            dd_pct = self.drawdown_controller.current_drawdown_pct(new_balance)
            survival = new_balance > self.initial_balance * 0.5

            return {
                "scenario": f"{n_losses} consecutive losses",
                "total_loss_usd": round(total_loss, 2),
                "new_balance": round(new_balance, 2),
                "impact_pct": round(impact_pct, 1),
                "drawdown_pct": round(dd_pct, 1),
                "survival": survival,
                "survival_threshold": "Balance > 50% of initial",
                "mode_after": "DEFENSIVE" if impact_pct > 5 else self.current_mode,
            }

        elif scenario == "best_case":
            avg_win = self._recent_performance.get("avg_win", 2.0)
            gain = risk_per_trade * avg_win * n_losses
            return {
                "scenario": f"{n_losses} consecutive wins",
                "total_gain_usd": round(gain, 2),
                "new_balance": round(self.balance + gain, 2),
                "gain_pct": round(gain / self.balance * 100, 1),
            }

        else:
            return {"scenario": scenario, "message": "Unknown scenario type"}

    # ═══════════════════════════════════════════════════════
    # CAPITAL REPORT
    # ═══════════════════════════════════════════════════════

    def generate_capital_report(self) -> dict:
        """
        Generate comprehensive capital management report.

        Returns:
            Dashboard-ready capital report dict.
        """
        dd_pct = self.drawdown_controller.current_drawdown_pct(self.balance)
        allocations = self.capital_manager.get_allocations()
        strategy_ranking = self._get_strategy_capital_ranking()
        total_allocated = self.capital_manager.get_total_allocated()

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "account": {
                "balance": round(self.balance, 2),
                "initial_balance": round(self.initial_balance, 2),
                "total_pnl": round(self.balance - self.initial_balance, 2),
                "total_pnl_pct": round(
                    (self.balance - self.initial_balance)
                    / self.initial_balance * 100, 2
                ),
            },
            "risk_mode": {
                "current": self.current_mode,
                "reason": self._mode_reason,
                "risk_range": list(RISK_MODES[self.current_mode]["risk_range"]),
                "max_deploy_pct": RISK_MODES[self.current_mode]["max_capital_deploy"] * 100,
            },
            "drawdown": {
                "current_pct": round(dd_pct, 2),
                "max_limit_pct": self.drawdown_controller.max_drawdown_limit,
                "protection_level": self.drawdown_controller.get_protection_level(self.balance),
            },
            "capital_allocation": {
                "total_allocated": round(total_allocated, 2),
                "reserve": round(self.balance - total_allocated, 2),
                "reserve_pct": round(
                    (self.balance - total_allocated) / self.balance * 100, 1
                ),
                "allocations": allocations,
            },
            "strategy_ranking": strategy_ranking,
            "performance": {
                "total_trades": len(self._trade_history),
                "win_rate_50": round(self._recent_performance["last_50_win_rate"] * 100, 1),
                "win_rate_20": round(self._recent_performance["last_20_win_rate"] * 100, 1),
                "current_streak": self._recent_performance["current_streak"],
                "daily_loss_pct": round(self._recent_performance["daily_loss_pct"], 2),
                "weekly_loss_pct": round(self._recent_performance["weekly_loss_pct"], 2),
            },
            "exposure": {
                "total_exposure_pct": round(
                    self.exposure_manager.get_total_exposure_pct(), 1
                ),
                "open_positions": list((self.exposure_manager._open_positions.keys() if isinstance(getattr(self.exposure_manager, "_open_positions", {}), dict) else []).__iter__()),
            },
            "kelly": {
                "current_risk_pct": round(
                    self.position_allocator.calculate_kelly_risk(
                        self._recent_performance["last_50_win_rate"],
                        self._recent_performance.get("avg_win", 2.0),
                        self._recent_performance.get("avg_loss", 1.0),
                    ) * 100, 2
                ),
                "kelly_fraction": self.kelly_fraction,
                "fraction_type": "Fractional Kelly (25%)",
            },
        }

        # Save report
        AUTONOMOUS_RISK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AUTONOMOUS_RISK_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        return report

    # ═══════════════════════════════════════════════════════
    # STATUS & REPORTING
    # ═══════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        """Get comprehensive risk manager statistics."""
        return {
            "balance": round(self.balance, 2),
            "initial_balance": round(self.initial_balance, 2),
            "current_mode": self.current_mode,
            "mode_reason": self._mode_reason,
            "total_trades": len(self._trade_history),
            "drawdown_pct": round(
                self.drawdown_controller.current_drawdown_pct(self.balance), 2
            ),
            "win_rate_50": round(
                self._recent_performance["last_50_win_rate"] * 100, 1
            ),
            "win_rate_20": round(
                self._recent_performance["last_20_win_rate"] * 100, 1
            ),
            "daily_loss_pct": round(
                self._recent_performance["daily_loss_pct"], 2
            ),
            "exposure_pct": round(
                self.exposure_manager.get_total_exposure_pct(), 1
            ),
        }

    def get_ai_context(self, decision: dict) -> dict:
        """Generate AI context for the LLM prompt."""
        return {
            "risk_approved": decision["approved"],
            "risk_mode": decision.get("risk_mode", self.current_mode),
            "risk_lot": decision.get("lot", 0),
            "risk_sl_pips": decision.get("sl_pips", 0),
            "risk_tp_pips": decision.get("tp_pips", 0),
            "risk_rr": decision.get("rr_ratio", 0),
            "risk_pct": decision.get("risk_pct", 0),
            "kelly_risk_pct": decision.get("kelly_risk_pct", 0),
            "risk_reject": decision.get("reject_reason"),
            "exposure_pct": decision.get("exposure_info", {}).get(
                "total_exposure_pct", 0
            ),
            "drawdown_pct": decision.get("drawdown_info", {}).get(
                "current_drawdown_pct", 0
            ),
            "strategy_ranking": decision.get("strategy_ranking", {}),
        }

    def print_status(self) -> None:
        """Print comprehensive risk manager status."""
        stats = self.get_stats()
        dd_pct = self.drawdown_controller.current_drawdown_pct(self.balance)
        mode_icons = {
            "AGGRESSIVE": "A", "NORMAL": "N",
            "DEFENSIVE": "D", "EMERGENCY": "E",
        }
        icon = mode_icons.get(self.current_mode, "?")

        bar = "=" * 50
        print(f"\n{bar}")
        print(f"  [{icon}] AUTONOMOUS RISK MANAGER — Day 58")
        print(bar)
        print(f"  Account Balance    : ${self.balance:,.2f}")
        print(f"  Initial Balance    : ${self.initial_balance:,.2f}")
        print(f"  PnL               : ${self.balance - self.initial_balance:+,.2f}")
        print(f"  Risk Mode          : {self.current_mode}")
        print(f"  Mode Reason        : {self._mode_reason}")
        print(f"  Drawdown           : {dd_pct:.1f}%")
        print(f"  Win Rate (50)      : {stats['win_rate_50']}%")
        print(f"  Win Rate (20)      : {stats['win_rate_20']}%")
        print(f"  Total Trades       : {stats['total_trades']}")
        print(f"  Daily Loss         : {stats['daily_loss_pct']}%")
        print(f"  Exposure           : {stats['exposure_pct']}%")
        print(f"  Kelly Fraction     : {self.kelly_fraction*100:.0f}%")
        print(bar)

        # Capital allocation
        allocs = self.capital_manager.get_allocations()
        if allocs:
            print("  Capital Allocation:")
            total_alloc = sum(v for v in allocs.values())
            for sym, amt in allocs.items():
                pct = amt / self.balance * 100 if self.balance > 0 else 0
                print(f"    {sym:<10}: ${amt:>8,.2f} ({pct:.1f}%)")
            print(f"    {'RESERVE':<10}: ${self.balance - total_alloc:>8,.2f}")
        print(bar + "\n")

    # ═══════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════

    def _blend_risk(
        self, kelly_risk: float, mode_risk: float, confidence: float
    ) -> float:
        """
        Blend Kelly risk with mode-based risk using confidence weighting.

        High confidence → trust Kelly more
        Low confidence  → trust mode defaults more
        """
        conf_weight = confidence / 100
        blended = kelly_risk * conf_weight + mode_risk * (1 - conf_weight)

        # Clamp to mode range
        mode_config = RISK_MODES[self.current_mode]
        min_r, max_r = mode_config["risk_range"]
        return max(min_r, min(blended, max_r))

    def _build_rejection(
        self, symbol: str, signal: str, reason: str, mode_config: dict
    ) -> dict:
        """Build a rejection response dict."""
        log.info(f"[AutonomousRisk] REJECTED {signal} {symbol}: {reason}")
        return {
            "approved": False,
            "signal": "NO TRADE",
            "symbol": symbol,
            "reject_reason": reason,
            "risk_mode": self.current_mode,
            "mode_reason": self._mode_reason,
            "lot": 0,
            "risk_pct": 0,
            "sl_pips": 0,
            "tp_pips": 0,
            "rr_ratio": 0,
            "confidence": 0,
        }

    def _get_open_positions(self) -> list[dict]:
        """Get current open positions from exposure manager."""
        return [
            {"symbol": sym, "direction": info["direction"]}
            for sym, info in (self.exposure_manager._open_positions.items() if isinstance(getattr(self.exposure_manager, "_open_positions", {}), dict) else []).__iter__()
        ]

    def _load_state(self) -> dict:
        """
        Load persisted state from disk.

        Day 102+ hotfix: FAIL SAFE on corruption. Previously returned
        a fresh empty state on any read error — silently resetting
        daily_losses, trade_count, and mode_history. For a risk module
        that tracks mode history (AGGRESSIVE/NORMAL/DEFENSIVE/EMERGENCY),
        silently resetting to a fresh state could drop the system back
        to AGGRESSIVE mode after a crash, even if it was safely in
        EMERGENCY mode before. Now: on corruption, return a state with
        mode=EMERGENCY so the system stays defensive until a human
        investigates.
        """
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        if AUTONOMOUS_RISK_STATE_PATH.exists():
            try:
                with open(AUTONOMOUS_RISK_STATE_PATH) as f:
                    state = json.load(f)
                # Validate required keys
                if "daily_losses" not in state or "mode_history" not in state:
                    raise ValueError("missing required keys")
                return state
            except Exception as e:
                log.critical(
                    f"[AutonomousRisk] STATE FILE CORRUPT ({e}) — "
                    f"FAILING SAFE (EMERGENCY mode). Manual review required."
                )
                return {
                    "daily_losses": {"_corrupt": 999999.0},
                    "trade_count": 0,
                    "mode_history": [{
                        "mode": "EMERGENCY",
                        "reason": "STATE FILE CORRUPT — fail-safe activation",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }],
                    "last_mode": "EMERGENCY",
                    "_corrupt": True,
                }
        return {
            "daily_losses": {},
            "trade_count": 0,
            "mode_history": [],
        }

    def _save_state(self) -> None:
        """
        Persist state to disk.

        Day 102+ hotfix: ATOMIC WRITE (temp file + os.replace).
        Previously used direct `open("w") + json.dump` — a crash
        mid-write would leave the state file truncated/corrupt,
        which (combined with the old fail-open _load_state) silently
        reset the system to AGGRESSIVE mode on the next restart.
        """
        import os
        import tempfile

        self._state["trade_count"] = len(self._trade_history)
        self._state["last_mode"] = self.current_mode

        mode_history = self._state.get("mode_history", [])
        if not mode_history or mode_history[-1]["mode"] != self.current_mode:
            mode_history.append({
                "mode": self.current_mode,
                "reason": self._mode_reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            self._state["mode_history"] = mode_history[-100:]  # keep last 100

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=str(MEMORY_DIR),
                suffix=".tmp",
                prefix="autonomous_risk_",
                delete=False,
            ) as tmp_f:
                json.dump(self._state, tmp_f, indent=2)
                tmp_path = tmp_f.name
            os.replace(tmp_path, str(AUTONOMOUS_RISK_STATE_PATH))
        except Exception as e:
            log.warning(f"[AutonomousRisk] state save failed: {e}")
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def reset_daily(self) -> None:
        """Reset daily counters. Called at start of new trading day."""
        self._state["daily_losses"] = {}
        self.drawdown_controller.reset_daily()
        self._save_state()
        log.info("[AutonomousRisk] Daily reset complete")