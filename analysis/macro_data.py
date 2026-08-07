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

        # Phase 2.5 (backtest infra optimization): skip yfinance entirely
        # during historical replay — today's DXY/Gold/Oil/VIX value has no
        # relevance to a historical bar, and profiling showed this call
        # (5 tickers, each with its own timeout) was a major per-bar cost.
        from core.constants import is_backtest_mode
        if is_backtest_mode():
            return self._fallback()

        try:
            import yfinance as yf

            # ── FIX: yfinance's bulk download() has a known bug where it
            # returns "possibly delisted; no price data found" for some
            # symbols in a multi-symbol batch (especially ^TNX, ^GSPC, ^VIX,
            # CL=F, DX-Y.NYB during weekends / pre-market windows). The bulk
            # call marks ALL of them as failed even when individual fetches
            # would succeed. Fetching each symbol individually with a short
            # timeout and graceful per-symbol fallback fixes this.
            result = {}
            successful = 0
            failed = []
            for label, sym in GLOBAL_SYMBOLS.items():
                asset_data = self._fetch_single_symbol(yf, sym, label)
                if asset_data is not None:
                    result[label.lower()] = asset_data
                    successful += 1
                else:
                    result[label.lower()] = self._fallback_asset()
                    failed.append(sym)

            if successful == 0:
                log.warning(
                    f"[MacroData] All {len(GLOBAL_SYMBOLS)} symbols failed — "
                    f"using fallback. Failed: {failed}"
                )
                return self._fallback()

            if failed:
                log.info(
                    f"[MacroData] {successful}/{len(GLOBAL_SYMBOLS)} symbols fetched OK; "
                    f"{len(failed)} failed (using fallback for those): {failed}"
                )

            result["source"] = "yfinance"
            self._cache      = {k: v for k, v in result.items() if k != "source"}
            self._cache_time = time.time()

            log.info(
                f"[MacroData] DXY={result['dxy']['trend']} "
                f"OIL={result['oil']['trend']} "
                f"VIX={result['vix']['trend']} "
                f"SP500={result['sp500']['trend']}"
            )
            return result

        except Exception as e:
            log.warning(f"[MacroData] Fetch error: {e} — using fallback")
            return self._fallback()

    def _fetch_single_symbol(self, yf_module, sym: str, label: str) -> dict | None:
        """Fetch one symbol individually — returns None on failure.

        Per-symbol fetches avoid yfinance's bulk-download bug where one
        failing symbol in a batch marks the entire batch as "possibly
        delisted". Also includes a short period-extension retry: if 5d
        returns empty (common on Monday pre-market), try 1mo.
        """
        for period in ("5d", "1mo"):
            try:
                raw = yf_module.download(
                    sym, period=period, interval="1d",
                    progress=False, auto_adjust=False, actions=False,
                    threads=False,
                )
                if raw is None or raw.empty:
                    continue
                # Single-symbol download returns a flat DataFrame
                # (no symbol in columns). Try both shapes.
                close_col = None
                if "Close" in raw.columns:
                    close_col = raw["Close"]
                elif "close" in raw.columns:
                    close_col = raw["close"]
                else:
                    # Maybe a MultiIndex — try first Close level
                    try:
                        close_col = raw.xs("Close", axis=1, level=-1)
                    except Exception:
                        close_col = None

                if close_col is None or close_col is None or len(close_col) < 2:
                    continue

                # FIX: newer yfinance versions sometimes return MultiIndex
                # columns even for a single-symbol download, so raw["Close"]
                # (or the xs() fallback above) can come back as a one-column
                # DataFrame instead of a Series. In that shape, .iloc[-2] /
                # .iloc[-1] each return a one-element Series rather than a
                # scalar, and float(series) is what triggers "Calling float
                # on a single element Series is deprecated". squeeze()
                # collapses that one-column DataFrame down to a plain Series
                # (no-op if it's already a Series), so the .iloc[...] below
                # always yields a real scalar.
                if hasattr(close_col, "columns"):
                    close_col = close_col.squeeze(axis=1)

                # Drop NaN values
                close_col = close_col.dropna()
                if len(close_col) < 2:
                    continue

                prev    = float(close_col.iloc[-2])
                current = float(close_col.iloc[-1])
                if prev <= 0:
                    return self._fallback_asset()
                change  = round((current - prev) / prev * 100, 3)

                # ^TNX yfinance-এ already %×10 scale-এ আসে (e.g. 42.5 = 4.25%)
                display_value = round(current / 10, 3) if label == "US10Y" else round(current, 3)

                return {
                    "value":      display_value,
                    "change_pct": change,
                    "trend":      self._classify_trend(change),
                }
            except Exception as e:
                log.debug(f"[MacroData] {sym} period={period} failed: {e}")
                continue
        return None

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