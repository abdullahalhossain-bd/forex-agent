# core/trading_engine.py  —  Day 37 | Autonomous Forex AI Trader, Brain Controller
# ============================================================
# This is the file the Day 37 doc calls for: "Create core/trading_engine.py
# — AI Trader-এর main brain controller."
#
# All the actual logic — Market Scanner → Analysis/Decision/Risk → Safety
# Guard (RiskEngine + TradePermission + CorrelationFilter) → Approval Mode
# → Execution Router (Paper/MT5 Demo) → Position Manager (PaperTrader price
# monitoring) → Telegram + DB + Learning Memory — already lives in
# core/trader.py's AITrader and AutonomousTraderSystem. This module is the
# named entry point + startup banner described in the Day 37 doc's
# "Step 1 — Start System" terminal output, so `main.py` can import
# `TradingEngine` instead of reaching into core/trader.py directly.
#
# Day 37+ runtime unification: TradingEngine now accepts an optional
# `registry` parameter (a ServiceRegistry) so the new boot_runtime() flow
# in core/runtime.py can construct it with all shared services wired in.
# ============================================================

from config import EXECUTION_MODE
from core.trader import AITrader, AutonomousTraderSystem  # noqa: F401
from utils.logger import get_logger

log = get_logger("trading_engine")


class TradingEngine(AutonomousTraderSystem):
    """
    Thin composition root on top of AutonomousTraderSystem — adds the
    Day 37 startup banner and a couple of introspection helpers. No new
    trading logic lives here on purpose: every gate (circuit breaker,
    approval mode, safety guard, execution router) is already wired inside
    core/trader.py so it works the same whether you drive it from here,
    from a script, or from server/signal_pipeline.py's webhook path.
    """

    def __init__(self, *args, registry=None, **kwargs):
        super().__init__(*args, registry=registry, **kwargs)

    def run(self) -> dict:
        self._print_banner()
        return super().run()

    def _print_banner(self) -> None:
        bar = "=" * 33
        print(bar)
        print("🤖 AUTONOMOUS FOREX AI TRADER")
        print()
        print("Mode:")
        print(self.execution_mode.upper())
        print()
        print("Scanner:")
        print("ON" if self.use_scanner else "OFF")
        print()
        print("Approval:")
        print(self.approval.mode_name)
        print()
        print("Registry:")
        print("yes" if self._registry else "no")
        print()
        print("Status:")
        print("STARTING...")
        print(bar)

    def pending_approvals(self) -> list[dict]:
        """Mode 2 (SUPERVISED) — trades waiting on a human approve()/reject()."""
        return self.approval.get_pending()

    def approve(self, pending_id: int) -> dict:
        return self.approval.approve(pending_id)

    def reject(self, pending_id: int, reason: str = "") -> dict:
        return self.approval.reject(pending_id, reason=reason)

    def circuit_breaker_status(self) -> dict:
        return self.circuit_breaker.get_status()

    def resume_trading(self, reason: str = "Manual override") -> dict:
        """Manual kill-switch reset — same as the doc's Mode 3 'AUTO MODE' resume."""
        return self.circuit_breaker.manual_resume(reason=reason)

    def health(self) -> dict:
        """Expose AutonomousTraderSystem.health_status() via the engine too."""
        return self.health_status()