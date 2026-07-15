# broker/safety_guard.py — Unified Safety Guard for Pre-Trade Checks
# ============================================================
# Wraps TradePermission + CorrelationFilter into a single safety gate
# called by AITrader before any trade execution.
# ============================================================

from utils.logger import get_logger
from risk.trade_permission import TradePermission
from scanner.correlation_filter import CorrelationFilter

log = get_logger("safety_guard")


class SafetyGuard:
    """
    Unified pre-trade safety check combining:
      1. TradePermission (news, confidence, session, duplicate)
      2. CorrelationFilter (correlated pair overexposure)
    
    Usage:
        guard = SafetyGuard()
        result = guard.check(decision_out, risk_out, news_ctx, session_ctx)
        if result["allowed"]:
            ... execute trade ...
    """

    def __init__(self, paper_trader=None):
        self._perm = TradePermission()
        self._corr = CorrelationFilter()
        self._paper_trader = paper_trader

    def check(
        self,
        decision_out: dict,
        risk_out: dict,
        news_ctx: dict,
        session_ctx: dict | None = None,
        symbol: str = "",
        final_action: str = "",
        execution_filters: dict | None = None,
    ) -> dict:
        """Run all safety checks. Returns dict with 'allowed' boolean."""

        # 1. TradePermission checks
        perm_out = self._perm.check(
            decision_out=decision_out,
            risk_out=risk_out,
            news_ctx=news_ctx,
            session_ctx=session_ctx,
            execution_filters=execution_filters,
        )

        # 2. Duplicate trade check
        if self._paper_trader and final_action:
            if self._paper_trader.has_open_position(symbol, final_action):
                perm_out["allowed"] = False
                perm_out["execution_allowed"] = False
                perm_out["final_action"] = "NO TRADE"
                perm_out["execution_action"] = "NO TRADE"
                perm_out["blocked_reason"] = f"{symbol} {final_action} already open"
                perm_out["checks"].append({
                    "check": "Duplicate trade",
                    "passed": False,
                    "detail": f"{symbol} {final_action} already open",
                })

        # 3. Correlation filter
        if perm_out["allowed"] and final_action:
            open_pairs = []
            if self._paper_trader:
                open_pairs = [t.get("pair") for t in self._paper_trader.get_open_positions()]
            self._corr.sync_open(open_pairs)
            still_allowed = self._corr.allow(
                [{"symbol": symbol, "signal": final_action}]
            )
            if not still_allowed:
                perm_out["allowed"] = False
                perm_out["execution_allowed"] = False
                perm_out["final_action"] = "NO TRADE"
                perm_out["execution_action"] = "NO TRADE"
                perm_out["blocked_reason"] = "Correlated pair group already has an open position"
                perm_out["checks"].append({
                    "check": "Correlation filter",
                    "passed": False,
                    "detail": "Correlated pair group already has an open position",
                })

        self._perm.print_summary(perm_out)
        return perm_out

    def get_status(self) -> dict:
        """Return current safety guard state for dashboard."""
        try:
            from scanner.config import CORRELATION_GROUPS
            groups = CORRELATION_GROUPS
        except Exception as e:
            groups = {}
        return {
            "trade_permission_active": True,
            "correlation_filter_active": True,
            "open_correlated_groups": groups,
        }
