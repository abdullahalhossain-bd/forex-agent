# broker/spread_monitor.py  —  Day 32 Bonus 2 | Spread Monitor
# ============================================================
# XAUUSD আর news time-এ spread অনেক widen হয় — সেই সময় trade নিলে
# entry-ই অনেক খারাপ জায়গায় হয়ে যায়। এই monitor সেটা block করে।
#
# account_manager.py-এর market_status() ইতিমধ্যে spread_pips রিটার্ন
# করে — এই module শুধু per-symbol threshold নিয়ে pass/fail decide করে,
# duplicate spread-fetch logic লেখা হয়নি।
# ============================================================

from utils.logger import get_logger

log = get_logger("spread_monitor")

# Pair-ভিত্তিক max acceptable spread (pips) — normal market condition-এর জন্য
MAX_SPREAD_PIPS = {
    "EURUSD": 2.0, "GBPUSD": 2.5,
    "USDJPY": 2.0, "USDCHF": 3.0,
    "AUDUSD": 2.5, "USDCAD": 3.0,
    "XAUUSD": 5.0,    # গোল্ড স্বাভাবিকভাবেই বেশি spread রাখে
    "DEFAULT": 3.0,
}

# News window-এর সময় threshold আরও কড়া হওয়া উচিত (spread spike থামাতে)
NEWS_WINDOW_MULTIPLIER = 0.5   # news চলাকালীন allowed spread অর্ধেক করে দেয়


class SpreadMonitor:
    """
    Usage:
        monitor = SpreadMonitor()
        check = monitor.check(symbol="EURUSD", current_spread_pips=1.8, news_active=False)
        if not check["allowed"]:
            ... trade skip করো ...
    """

    def check(
        self, symbol: str, current_spread_pips: float, news_active: bool = False
    ) -> dict:
        clean_symbol = symbol.upper()[:6]
        max_allowed = MAX_SPREAD_PIPS.get(clean_symbol, MAX_SPREAD_PIPS["DEFAULT"])

        if news_active:
            max_allowed *= NEWS_WINDOW_MULTIPLIER

        allowed = current_spread_pips <= max_allowed
        reason = (
            "OK"
            if allowed
            else f"Spread {current_spread_pips} pips > max {max_allowed} pips"
                 f"{' (news window — stricter limit)' if news_active else ''}"
        )

        if not allowed:
            log.warning(f"[SpreadMonitor] {clean_symbol} blocked — {reason}")

        return {
            "allowed": allowed,
            "current_spread_pips": current_spread_pips,
            "max_allowed_pips": round(max_allowed, 2),
            "reason": reason,
        }