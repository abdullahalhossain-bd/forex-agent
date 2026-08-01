# broker/mt5_data.py  —  Day 32 Part 1, 2, 6 | Tick + Multi-Timeframe Candle Data
# ============================================================
# একজন professional trader শুধু একটা timeframe দেখে না — D1 দিয়ে
# direction, H4 দিয়ে structure, M15 দিয়ে entry timing বোঝে। এই
# module সেই multi-timeframe candle fetch + live tick + disk-এ
# save করার দায়িত্ব নেয়।
#
# account_manager.py-এর resolve_symbol()/market_status() থেকে
# duplicate করা হয়নি — broker symbol resolve করার জন্য AccountManager
# কেই call করো, এই module শুধু candle/tick data নিয়ে কাজ করে।
#
# FIX (Day 37 hotfix): lowercase timeframe aliases যোগ হয়েছে।
# data.fetcher "15m", "1h", "4h", "1d" পাঠায়, কিন্তু এই module
# আগে শুধু "M15", "H1", "H4", "D1" চিনত — Unknown timeframe error
# আসত। এখন উভয় format-ই চলে।
# ============================================================

import os
import csv
from datetime import datetime, timezone
from utils.logger import get_logger
from broker.mt5_connection import MT5_AVAILABLE

log = get_logger("mt5_data")

if MT5_AVAILABLE:
    import MetaTrader5 as mt5

LIVE_DATA_DIR = "data/live"
os.makedirs(LIVE_DATA_DIR, exist_ok=True)

# ── Timeframe map ─────────────────────────────────────────────
# Uppercase (MT5 standard): M1, M5, M15, H1, H4, D1
# Lowercase aliases:        1m, 5m, 15m, 1h, 4h, 1d
# data.fetcher, analysis_agent, DataFetcher — সবাই lowercase পাঠায়।
# MT5DataFeed এখন দুটোই accept করে।
TIMEFRAMES = {
    # ── Standard uppercase ────────────────────────────────────
    "M1":  "TIMEFRAME_M1",
    "M5":  "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1":  "TIMEFRAME_H1",
    "H4":  "TIMEFRAME_H4",
    "D1":  "TIMEFRAME_D1",
    "W1":  "TIMEFRAME_W1",
    "MN1": "TIMEFRAME_MN1",
    # ── Lowercase aliases (data.fetcher format) ───────────────
    "1m":  "TIMEFRAME_M1",
    "5m":  "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1h":  "TIMEFRAME_H1",
    "4h":  "TIMEFRAME_H4",
    "1d":  "TIMEFRAME_D1",
    "1w":  "TIMEFRAME_W1",
    # ── Alternative spellings ─────────────────────────────────
    "m1":  "TIMEFRAME_M1",
    "m5":  "TIMEFRAME_M5",
    "m15": "TIMEFRAME_M15",
    "m30": "TIMEFRAME_M30",
    "h1":  "TIMEFRAME_H1",
    "h4":  "TIMEFRAME_H4",
    "d1":  "TIMEFRAME_D1",
}


def _mt5_timeframe(label: str):
    """
    Label string ('M15', '15m', 'm15') কে actual mt5.TIMEFRAME_*
    constant-এ map করে।

    তিনটা format support করে:
      1. MT5 standard uppercase: "M15", "H4", "D1"
      2. data.fetcher lowercase: "15m", "4h", "1d"
      3. Alternative lowercase:  "m15", "h4", "d1"

    Returns:
        mt5.TIMEFRAME_* constant, অথবা None যদি অচেনা হয়।
    """
    if not MT5_AVAILABLE:
        return None

    # Direct lookup (covers all three formats via TIMEFRAMES dict)
    attr = TIMEFRAMES.get(label)
    if not attr:
        # শেষ চেষ্টা: uppercase করে দেখো ("15m" → "15M" কাজ করবে না,
        # কিন্তু "m15" → "M15" কাজ করতে পারে)
        attr = TIMEFRAMES.get(label.upper())

    if not attr:
        log.error(f"[MT5DataFeed] Unknown timeframe: {label!r} — "
                  f"supported: {sorted(TIMEFRAMES.keys())}")
        return None

    tf_const = getattr(mt5, attr, None)
    if tf_const is None:
        log.error(f"[MT5DataFeed] mt5.{attr} constant not found — "
                  f"MetaTrader5 version may be too old")
    return tf_const


def normalize_timeframe(label: str) -> str:
    """
    যেকোনো timeframe string-কে MT5 standard uppercase-এ convert করে।

    Examples:
        "15m" → "M15"
        "1h"  → "H1"
        "4h"  → "H4"
        "1d"  → "D1"
        "M15" → "M15"  (already correct)

    Useful for logging, file naming, DB keys।
    """
    _reverse = {v_attr: k_upper for k_upper, v_attr in TIMEFRAMES.items()
                if k_upper == k_upper.upper() and len(k_upper) <= 3}
    attr = TIMEFRAMES.get(label) or TIMEFRAMES.get(label.upper())
    if attr:
        # Find the canonical uppercase key for this attr
        for k, v in TIMEFRAMES.items():
            if v == attr and k == k.upper():
                return k
    return label.upper()


class MT5DataFeed:
    """
    MT5 থেকে tick + multi-timeframe candle data নেয়।

    Usage:
        feed = MT5DataFeed()
        tick = feed.get_tick("EURUSD")

        # উভয় format কাজ করে:
        candles = feed.get_candles("EURUSD", "M15", count=500)
        candles = feed.get_candles("EURUSD", "15m", count=500)  # ← same result

        all_tf = feed.get_multi_timeframe("EURUSD")
    """

    DEFAULT_CANDLE_COUNT = 500

    # ─────────────────────────────────────────────
    # TICK DATA  (Part 1)
    # ─────────────────────────────────────────────

    def get_tick(self, broker_symbol: str) -> dict | None:
        """
        Live bid/ask/spread নেয়। broker_symbol অবশ্যই
        AccountManager.resolve_symbol()-এর exact output হতে হবে।
        """
        if not MT5_AVAILABLE:
            log.error("[MT5DataFeed] MetaTrader5 package নেই")
            return None

        tick = mt5.symbol_info_tick(broker_symbol)
        if tick is None or tick.time == 0:
            log.warning(f"[MT5DataFeed] কোনো tick data নেই: {broker_symbol}")
            return None

        info = mt5.symbol_info(broker_symbol)
        digits = info.digits if info else 5
        spread_points = (tick.ask - tick.bid)
        spread_pips = round(spread_points * (10 ** (digits - 1)), 1) if digits else 0

        return {
            "symbol":      broker_symbol,
            "bid":         tick.bid,
            "ask":         tick.ask,
            "spread_pips": spread_pips,
            "time":        datetime.fromtimestamp(tick.time, tz=timezone.utc).isoformat(),
        }

    def get_tick_stream(self, broker_symbols: list[str]) -> dict[str, dict]:
        """একসাথে একাধিক pair-এর tick নেয় — multi-pair scanner-এর basic building block।"""
        stream = {}
        for sym in broker_symbols:
            tick = self.get_tick(sym)
            if tick:
                stream[sym] = tick
        return stream

    def print_tick_stream(self, broker_symbols: list[str]) -> None:
        stream = self.get_tick_stream(broker_symbols)
        bar = "═" * 36
        log.info(bar)
        log.info("  📡  LIVE TICK STREAM")
        log.info(bar)
        for sym, tick in stream.items():
            log.info(f"  {sym:<8} Bid: {tick['bid']}  Ask: {tick['ask']}  ({tick['spread_pips']} pips)")
        log.info(bar)

    # ─────────────────────────────────────────────
    # CANDLE DATA — single + multi timeframe  (Part 2)
    # ─────────────────────────────────────────────

    def get_candles(
        self, broker_symbol: str, timeframe: str, count: int = None
    ) -> list[dict]:
        """
        একটা timeframe-এর candles নেয়। Returns list of dicts:
        {time, open, high, low, close, volume, spread}

        timeframe: "M15", "15m", "m15" — সবই চলে।
        """
        if not MT5_AVAILABLE:
            log.error("[MT5DataFeed] MetaTrader5 package নেই")
            return []

        tf_const = _mt5_timeframe(timeframe)
        if tf_const is None:
            return []  # error already logged in _mt5_timeframe

        count = count or self.DEFAULT_CANDLE_COUNT
        rates = mt5.copy_rates_from_pos(broker_symbol, tf_const, 0, count)
        if rates is None or len(rates) == 0:
            log.warning(f"[MT5DataFeed] কোনো candle পাওয়া যায়নি: {broker_symbol} {timeframe}")
            return []

        candles = [
            {
                "time":   datetime.fromtimestamp(int(r["time"]), tz=timezone.utc).isoformat(),
                "open":   float(r["open"]),
                "high":   float(r["high"]),
                "low":    float(r["low"]),
                "close":  float(r["close"]),
                "volume": float(r["tick_volume"]),
                "spread": int(r["spread"]),
            }
            for r in rates
        ]
        return candles

    def get_multi_timeframe(
        self, broker_symbol: str, timeframes: list[str] = None, count: int = None
    ) -> dict[str, list[dict]]:
        """
        D1 → H4 → H1 → M15 → M5 → M1 — professional trader-এর মতো
        একসাথে সব timeframe নেয়। Direction (D1/H4) আর entry timing
        (M15/M5) দুটোই context-এ থাকবে।

        timeframes list-এ uppercase বা lowercase দুটোই দেওয়া যায়।
        """
        # Default: standard uppercase keys
        timeframes = timeframes or list(dict.fromkeys(
            k for k in TIMEFRAMES if k == k.upper() and not k.startswith("MN")
        ))
        result = {}
        for tf in timeframes:
            candles = self.get_candles(broker_symbol, tf, count=count)
            result[tf] = candles
        return result

    def print_multi_timeframe_status(
        self, broker_symbol: str, timeframes: list[str] = None
    ) -> None:
        timeframes = timeframes or ["D1", "H4", "H1", "M15", "M5", "M1"]
        data = self.get_multi_timeframe(broker_symbol, timeframes)
        log.info(f"  {broker_symbol}")
        for tf in timeframes:
            n = len(data.get(tf, []))
            status = "✅" if n > 0 else "❌"
            log.info(f"    {tf:<4} {status}  ({n} candles)")

    # ─────────────────────────────────────────────
    # STORAGE  (Part 6)
    # ─────────────────────────────────────────────

    def save_live_csv(self, broker_symbol: str, timeframe: str, candles: list[dict]) -> str:
        """
        data/live/{SYMBOL}_{TIMEFRAME}_live.csv-এ save করে। DB-তে নয় —
        কারণ db.py-এর `candles` table আলাদা পরিসরের জন্য আছে এবং এখানে
        existing schema পরিবর্তন না করেই raw live snapshot রাখা হচ্ছে।

        timeframe normalize হয়ে uppercase file name পাবে।
        """
        tf_norm = normalize_timeframe(timeframe)
        path = os.path.join(LIVE_DATA_DIR, f"{broker_symbol}_{tf_norm}_live.csv")
        if not candles:
            return path

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(candles[0].keys()))
            writer.writeheader()
            writer.writerows(candles)

        log.info(f"[MT5DataFeed] Saved {len(candles)} candles → {path}")
        return path