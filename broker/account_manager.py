# broker/account_manager.py  —  Day 31 Part 2 & 6 | Account State + Execution Safety
# ============================================================
# Account info read করা ও "এখন trade করা নিরাপদ কিনা" — এই দুটো
# দায়িত্ব এই module-এর।
#
# DAY 82+ FIX:
#   - trading_permission() এ max_spread_pips আর hardcode 3.0 নয়।
#     symbol-specific threshold ব্যবহার হয় (XAUUSD=50, DEFAULT=10)।
#   - TEST_MODE=true হলে spread_ok সবসময় True — শুধু trade_mode চেক।
#   - market_status() এ spread=0 কে "closed" না ধরে "off-hours" হিসেবে দেখে।
# ============================================================

import os
from datetime import datetime, timezone
from utils.logger import get_logger
from broker.mt5_connection import MT5_AVAILABLE

log = get_logger("account_manager")

if MT5_AVAILABLE:
    import MetaTrader5 as mt5


# ── Per-symbol max spread (pips) for production trading ───────
SPREAD_LIMITS_PIPS = {
    "EURUSD": 3.0,
    "GBPUSD": 4.0,
    "USDJPY": 3.0,
    "AUDUSD": 3.0,
    "USDCAD": 4.0,
    "XAUUSD": 50.0,
    "XAGUSD": 10.0,
    "DEFAULT": 10.0,
}


def _spread_limit(symbol: str) -> float:
    return SPREAD_LIMITS_PIPS.get(symbol.upper(), SPREAD_LIMITS_PIPS["DEFAULT"])


def _test_mode() -> bool:
    return os.getenv("TEST_MODE", "false").lower() == "true"


class AccountManager:
    """
    MT5 account state বোঝে এবং trade নেওয়ার আগে safety check করে।

    Usage:
        am = AccountManager(connection)
        info = am.get_account_snapshot()
        perm = am.trading_permission(symbol="EURUSD", risk_engine_ok=True)
        if perm["allowed"]:
            ... place order ...
    """

    MIN_FREE_MARGIN_PCT = 20.0     # equity-র কমপক্ষে কত % free margin থাকা উচিত
    MIN_MARGIN_LEVEL_PCT = 200.0   # margin level এর নিচে গেলে risky

    def __init__(self, connection):
        self.connection = connection

    # ─────────────────────────────────────────────
    # ACCOUNT SNAPSHOT  (Day 31 Part 2)
    # ─────────────────────────────────────────────

    def get_account_snapshot(self) -> dict | None:
        if not MT5_AVAILABLE or not self.connection.connected:
            log.warning("[AccountManager] MT5 connected নয়")
            return None

        account = mt5.account_info()
        if account is None:
            log.error(f"[AccountManager] account_info() failed: {mt5.last_error()}")
            return None

        free_margin_pct = (
            round(account.margin_free / account.equity * 100, 1)
            if account.equity else 0.0
        )

        snapshot = {
            "balance":         account.balance,
            "equity":          account.equity,
            "margin":          account.margin,
            "free_margin":     account.margin_free,
            "free_margin_pct": free_margin_pct,
            "margin_level":    account.margin_level,
            "currency":        account.currency,
            "trade_allowed":   account.trade_allowed,
            # MT5 User Guide Page 6 — account type (netting vs hedging)
            # Critical for position management: netting merges same-symbol orders,
            # hedging keeps them as independent positions.
            "margin_mode":     self._get_margin_mode(account),
            "account_type":    self._get_account_type_label(account),
            "checked_at":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        snapshot["status"] = self._classify_health(snapshot)
        return snapshot

    def _get_margin_mode(self, account) -> str:
        """MT5 User Guide Page 6 — detect netting vs hedging account type.
        Returns: 'netting' | 'hedging' | 'unknown'
        """
        try:
            # MT5 account_info has 'margin_mode' field:
            #   0 = ACCOUNT_MARGIN_MODE_RETAIL_NETTING
            #   1 = ACCOUNT_MARGIN_MODE_RETAIL_HEDGING
            #   2 = ACCOUNT_MARGIN_MODE_EXCHANGE
            mode = getattr(account, "margin_mode", None)
            if mode is None:
                return "unknown"
            if mode == 0:
                return "netting"
            elif mode == 1:
                return "hedging"
            elif mode == 2:
                return "exchange"
            return f"unknown({mode})"
        except Exception as e:
            return "unknown"

    def _get_account_type_label(self, account) -> str:
        """Human-readable account type label for logging."""
        mode = self._get_margin_mode(account)
        if mode == "netting":
            return "Netting (orders merge into single position per symbol)"
        elif mode == "hedging":
            return "Hedging (independent positions, can hold opposite directions)"
        elif mode == "exchange":
            return "Exchange (exchange-traded instruments)"
        return f"Unknown ({mode})"

    def _classify_health(self, snap: dict) -> str:
        if not snap["trade_allowed"]:
            return "BLOCKED"
        if snap["margin_level"] and 0 < snap["margin_level"] < self.MIN_MARGIN_LEVEL_PCT:
            return "AT_RISK"
        if snap["free_margin_pct"] < self.MIN_FREE_MARGIN_PCT:
            return "LOW_MARGIN"
        return "HEALTHY"

    def print_snapshot(self) -> None:
        s = self.get_account_snapshot()
        if not s:
            return
        icon = {
            "HEALTHY": "🟢", "LOW_MARGIN": "🟡",
            "AT_RISK": "🟠", "BLOCKED": "🔴",
        }.get(s["status"], "⚪")
        bar = "═" * 44
        log.info(bar)
        log.info(f"  {icon}  ACCOUNT STATUS — {s['status']}")
        log.info(bar)
        log.info(f"  Balance       : ${s['balance']:.2f}")
        log.info(f"  Equity        : ${s['equity']:.2f}")
        log.info(f"  Free Margin   : ${s['free_margin']:.2f} ({s['free_margin_pct']}%)")
        log.info(f"  Margin Level  : {s['margin_level']:.1f}%")
        log.info(bar)

    # ─────────────────────────────────────────────
    # SYMBOL VALIDATION  (Bonus 1)
    # ─────────────────────────────────────────────

    def resolve_symbol(self, requested_symbol: str) -> str | None:
        """
        AI চায় 'EURUSD' কিন্তু broker এর কাছে 'EURUSD.m' বা 'EURUSDm'
        থাকতে পারে। mt5.symbols_get() দিয়ে actual matching symbol খুঁজে দেয়।
        """
        if not MT5_AVAILABLE:
            return None

        requested = requested_symbol.upper().strip()

        exact_match = mt5.symbols_get(f"*{requested}*")
        if exact_match:
            for s in exact_match:
                if s.name.upper() == requested:
                    return s.name

        all_symbols = mt5.symbols_get()
        if not all_symbols:
            log.error("[AccountManager] symbols_get() থেকে কিছু পাওয়া যায়নি")
            return None

        for s in all_symbols:
            name_upper = s.name.upper()
            cleaned = (
                name_upper.replace(".", "").replace("M", "", 1)
                if name_upper.endswith("M") else name_upper
            )
            if cleaned == requested or requested in name_upper:
                log.info(f"[AccountManager] Symbol resolved: '{requested}' → '{s.name}'")
                return s.name

        log.error(f"[AccountManager] কোনো matching symbol পাওয়া যায়নি: {requested}")
        return None

    # ─────────────────────────────────────────────
    # MARKET STATUS CHECK  (Bonus 2)
    # ─────────────────────────────────────────────

    def market_status(self, broker_symbol: str) -> dict:
        """
        Market open কিনা, spread acceptable কিনা, trading allowed কিনা চেক করে।

        DAY 82+ FIX:
          tick.bid == 0 এবং tick.ask == 0 হলেই "no data"।
          spread=0 alone মানে closed না — off-hours MT5 cached এ spread=0
          আসতে পারে কিন্তু trade_mode=4 থাকলে market আসলে open।
        """
        if not MT5_AVAILABLE:
            return {"ok": False, "reason": "MT5 unavailable"}

        info = mt5.symbol_info(broker_symbol)
        if info is None:
            return {"ok": False, "reason": f"Symbol info না পাওয়া গেছে: {broker_symbol}"}

        if not info.visible:
            mt5.symbol_select(broker_symbol, True)
            info = mt5.symbol_info(broker_symbol)

        tick = mt5.symbol_info_tick(broker_symbol)
        if tick is None or (tick.bid == 0 and tick.ask == 0):
            return {"ok": False, "reason": "কোনো live tick data নেই — market বন্ধ থাকতে পারে"}

        spread_points = info.spread
        spread_pips = (
            spread_points / 10 if info.digits in (3, 5) else spread_points
        )

        result = {
            "ok":         True,
            "symbol":     broker_symbol,
            "trade_mode": info.trade_mode,
            "spread_pips": round(spread_pips, 1),
            "bid":        tick.bid,
            "ask":        tick.ask,
            "digits":     info.digits,
        }

        if info.trade_mode == 0:
            result["ok"] = False
            result["reason"] = "এই symbol-এ trading disabled আছে এই broker-এ"
        elif info.trade_mode == 3:
            result["ok"] = False
            result["reason"] = "এই symbol-এ শুধুমাত্র পজিশন ক্লোজ করা যাবে, নতুন ট্রেড নয়"

        return result

    # ─────────────────────────────────────────────
    # EXECUTION SAFETY CHECK  (Day 31 Part 6)
    # ─────────────────────────────────────────────

    def trading_permission(
        self,
        symbol: str,
        risk_engine_ok: bool = True,
        max_spread_pips: float = None,  # None → symbol-specific threshold
    ) -> dict:
        """
        ট্রেড এক্সিকিউট করার আগে সব সেফটি লেয়ার পাস করছে কিনা যাচাই করে।

        DAY 82+ FIX:
          - max_spread_pips=None হলে SPREAD_LIMITS_PIPS থেকে নেওয়া হয়।
          - TEST_MODE=true হলে spread_ok সবসময় True (শুধু trade_mode চেক)।
          - market_status() এ spread=0 কে closed না ধরা হয় (tick.bid/ask চেক)।
        """
        checks = {}

        snapshot = self.get_account_snapshot()
        checks["account_ok"] = bool(
            snapshot and snapshot["status"] in ("HEALTHY", "LOW_MARGIN")
        )

        broker_symbol = self.resolve_symbol(symbol)
        checks["symbol_ok"] = broker_symbol is not None

        market = (
            self.market_status(broker_symbol)
            if broker_symbol
            else {"ok": False, "reason": "symbol unresolved"}
        )
        checks["market_ok"] = market.get("ok", False)

        # Spread check — TEST_MODE ছাড়া symbol-specific limit দিয়ে
        if _test_mode():
            checks["spread_ok"] = True
            spread_detail = "TEST_MODE — spread check skipped"
        else:
            limit = max_spread_pips if max_spread_pips is not None else _spread_limit(symbol)
            actual_spread = market.get("spread_pips", 999)
            checks["spread_ok"] = actual_spread <= limit if checks["market_ok"] else False
            spread_detail = f"{actual_spread} pips (limit {limit})"

        checks["risk_ok"] = risk_engine_ok

        allowed = all(checks.values())
        failed = [k for k, v in checks.items() if not v]

        if failed:
            log.warning(
                f"[AccountManager] trading_permission DENIED for {symbol} "
                f"— failed: {failed}"
            )
        else:
            log.info(f"[AccountManager] trading_permission ALLOWED for {symbol}")

        return {
            "allowed":        allowed,
            "broker_symbol":  broker_symbol,
            "checks":         checks,
            "failed_checks":  failed,
            "market_info":    market,
            "account_status": snapshot["status"] if snapshot else "UNKNOWN",
        }