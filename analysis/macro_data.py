# analysis/macro_data.py  —  Day 65 | Macro Data Provider
# ============================================================
# IntermarketEngine-এর জন্য global market data সংগ্রহ করে।
#
# Sources (yfinance):
#   DXY     -> "DX-Y.NYB"   (US Dollar Index)
#   Gold    -> "GC=F"       (Gold futures)
#   Oil     -> "CL=F"       (WTI Crude futures)
#   US10Y   -> "^TNX"       (10-Year Treasury Yield, x10 scale)
#   SP500   -> "^GSPC"
#   VIX     -> "^VIX"
#
# প্রতিটা asset-এর জন্য: current value, % change, trend (BULLISH/
# BEARISH/NEUTRAL) — ছোট, consistent dict ফরম্যাটে রিটার্ন করে, ঠিক
# analysis/sentiment_data.py-এর প্যাটার্নে (5 min cache + fallback)।
# ============================================================

import time
from utils.logger import get_logger

log = get_logger("macro_data")

# ── yfinance tickers for each tracked global asset ─────────────
# Day 82 fix: Removed GC=F (Gold futures delisted from yfinance).
# Alternative: use XAUUSD pair directly from forex data instead.
GLOBAL_SYMBOLS = {
    "DXY":   "DX-Y.NYB",
    # "GOLD":  "GC=F",      # DELISTED — yfinance no longer provides this data
    "OIL":   "CL=F",
    "US10Y": "^TNX",
    "SP500": "^GSPC",
    "VIX":   "^VIX",
}

TREND_THRESHOLD_PCT = 0.15   # এর বেশি change হলে BULLISH/BEARISH, নাহলে NEUTRAL


class MacroDataProvider:
    """
    Usage:
        provider = MacroDataProvider()
        data = provider.get_all()
        provider.print_summary(data)
    """

    def __init__(self, cache_ttl: int = 300):
        self._cache: dict = {}
        self._cache_time: float = 0
        self._cache_ttl = cache_ttl

    # ═══════════════════════════════════════════════════════════
    # MAIN METHOD
    # ═══════════════════════════════════════════════════════════

    def get_all(self) -> dict:
        """
        সব global asset একসাথে fetch করো (5 min cache)।

        Returns:
            {
                "dxy":   {"value": 104.5, "change_pct": 0.32, "trend": "BULLISH"},
                "gold":  {...},
                "oil":   {...},
                "us10y": {...},
                "sp500": {...},
                "vix":   {...},
                "source": "yfinance" | "cache" | "fallback",
            }
        """
        if self._cache and (time.time() - self._cache_time) < self._cache_ttl:
            log.info("[MacroData] Using cached global market data")
            return {**self._cache, "source": "cache"}

        try:
            import yfinance as yf

            symbols = list(GLOBAL_SYMBOLS.values())
            raw    = yf.download(symbols, period="5d", interval="1d", progress=False)
            closes = raw.get("Close") if raw is not None else None

            if closes is None or closes.empty:
                return self._fallback()

            result = {}
            for label, sym in GLOBAL_SYMBOLS.items():
                result[label.lower()] = self._compute_asset(closes, sym, label)

            result["source"] = "yfinance"
            self._cache      = {k: v for k, v in result.items() if k != "source"}
            self._cache_time = time.time()

            log.info(
                f"[MacroData] DXY={result['dxy']['trend']} "
                f"Gold={result['gold']['trend']} "
                f"VIX={result['vix']['trend']} "
                f"SP500={result['sp500']['trend']}"
            )
            return result

        except Exception as e:
            log.warning(f"[MacroData] Fetch error: {e} — using fallback")
            return self._fallback()

    def _compute_asset(self, closes, sym: str, label: str) -> dict:
        try:
            col = closes[sym].dropna()
            if len(col) < 2:
                return self._fallback_asset()

            prev    = float(col.iloc[-2])
            current = float(col.iloc[-1])
            change  = round((current - prev) / prev * 100, 3) if prev else 0.0

            # ^TNX yfinance-এ already %×10 scale-এ আসে (e.g. 42.5 = 4.25%)
            display_value = round(current / 10, 3) if label == "US10Y" else round(current, 3)

            return {
                "value":      display_value,
                "change_pct": change,
                "trend":      self._classify_trend(change),
            }
        except Exception as e:
            return self._fallback_asset()

    def _classify_trend(self, change_pct: float) -> str:
        if change_pct >= TREND_THRESHOLD_PCT:
            return "BULLISH"
        if change_pct <= -TREND_THRESHOLD_PCT:
            return "BEARISH"
        return "NEUTRAL"

    def _fallback_asset(self) -> dict:
        return {"value": None, "change_pct": 0.0, "trend": "NEUTRAL"}

    def _fallback(self) -> dict:
        result = {label.lower(): self._fallback_asset() for label in GLOBAL_SYMBOLS}
        result["source"] = "fallback"
        return result

    # ═══════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════════

    def print_summary(self, data: dict) -> None:
        bar = "─" * 48
        print(f"\n{bar}")
        print("  🌎  GLOBAL MARKET DATA  (Day 65)")
        print(bar)
        icons = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}
        for label in GLOBAL_SYMBOLS:
            key  = label.lower()
            d    = data.get(key, {})
            icon = icons.get(d.get("trend"), "⚪")
            val  = d.get("value")
            chg  = d.get("change_pct", 0.0) or 0.0
            val_str = f"{val:.3f}" if val is not None else "N/A"
            print(f"  {label:<6} {icon} {d.get('trend', 'NEUTRAL'):<8}  {val_str:>10}  ({chg:+.2f}%)")
        print(f"  [{data.get('source', 'unknown')}]")
        print(bar + "\n")