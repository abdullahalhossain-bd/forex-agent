# scanner/correlation_filter.py  —  Day 36 Part 5 | Correlation Risk Filter
# ============================================================
# একই underlying currency-র একাধিক trade = hidden concentration risk।
# Example: EURUSD BUY + GBPUSD BUY + AUDUSD BUY = তিনটাই USD weakness bet।
# এই filter সেটা detect করে block করে।
# ============================================================

from utils.logger import get_logger
from scanner.config import CORRELATION_GROUPS

log = get_logger("correlation_filter")


class CorrelationFilter:
    """
    Usage:
        cf = CorrelationFilter()
        cf.register_open("EURUSD")   # trade open হলে register করো
        ok = cf.allow(opportunities)  # list filter করে correlated skip করে
    """

    def __init__(self):
        # Currently open positions (symbol set) — RiskEngine থেকে sync করা উচিত
        self._open_symbols: set[str] = set()

    def sync_open(self, open_symbols: list[str]) -> None:
        """RiskEngine বা PositionManager-এর current open positions sync করো।"""
        self._open_symbols = {s.upper()[:6] for s in open_symbols}

    def register_open(self, symbol: str) -> None:
        self._open_symbols.add(symbol.upper()[:6])

    def register_close(self, symbol: str) -> None:
        self._open_symbols.discard(symbol.upper()[:6])

    def allow(self, opportunities: list[dict]) -> list[dict]:
        """
        opportunities = [{"symbol": "EURUSD", "signal": "BUY", ...}, ...]
        শুধু open positions এর সাথে correlated হলে block করবে।
        Same scan cycle এর অন্য pair block করবে না।
        """
        allowed = []
        blocked_groups: set[int] = set()

        # শুধু open positions থেকে blocked groups তৈরি করো
        for i, group in enumerate(CORRELATION_GROUPS):
            if self._open_symbols & group:
                blocked_groups.add(i)
                log.info(
                    f"[CorrelationFilter] Group {i} blocked by open position: "
                    f"{self._open_symbols & group}"
                )

        for opp in opportunities:
            sym = opp.get("symbol", "").upper()[:6]
            group_idx = self._find_group(sym)

            if group_idx is not None and group_idx in blocked_groups:
                log.warning(
                    f"[CorrelationFilter] ⚠️ {sym} skipped — "
                    f"correlated group {group_idx} already active"
                )
                opp = dict(opp)
                opp["correlation_blocked"] = True
                opp["correlation_reason"] = (
                    f"Correlated with open position in group {group_idx}: "
                    f"{self._open_symbols & CORRELATION_GROUPS[group_idx]}"
                )
                continue

            # ✅ Same-cycle blocking সরানো হয়েছে — শুধু open positions block করবে
            opp = dict(opp)
            opp["correlation_blocked"] = False
            allowed.append(opp)

        return allowed

    def _find_group(self, symbol: str) -> int | None:
        for i, group in enumerate(CORRELATION_GROUPS):
            if symbol in group:
                return i
        return None

    def print_status(self) -> None:
        log.info(f"[CorrelationFilter] Open symbols: {self._open_symbols}")