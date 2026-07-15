# broker/market_data_manager.py  —  Day 32 Part 7 | Central Data Manager
# ============================================================
# MT5 → Data Manager → Market Agent → Technical Agent → Risk Agent
#
# এই module-টা MT5DataFeed + DataValidator + SymbolManager একসাথে
# জড়ো করে একটা single entry point দেয়, যাতে rule engine, AIAnalyst,
# SignalPipeline সবাই একই clean, validated, multi-timeframe data পায় —
# প্রতিটা module আলাদা ভাবে MT5 call করলে data inconsistent হতে পারত।
# ============================================================

from utils.logger import get_logger
from broker.mt5_data import MT5DataFeed
from broker.data_validator import DataValidator
from broker.symbol_manager import SymbolManager, DEFAULT_SYMBOLS

log = get_logger("market_data_manager")


class MarketDataManager:
    """
    সব downstream agent (rule engine, AIAnalyst, RiskEngine) এই class
    থেকে data নেবে — সরাসরি MT5DataFeed call করবে না।

    Usage:
        mdm = MarketDataManager(connection, account_manager)
        bundle = mdm.get_clean_bundle("EURUSD")
        # bundle = {
        #   "broker_symbol": "EURUSD",
        #   "tick": {...},
        #   "timeframes": {"M15": [...], "H1": [...], ...},
        #   "quality": {"M15": {...report...}, ...},
        # }
    """

    def __init__(self, connection, account_manager):
        self.connection = connection
        self.account_manager = account_manager
        self.feed = MT5DataFeed()
        self.validator = DataValidator(data_feed=self.feed)
        self.symbol_manager = SymbolManager(account_manager, data_feed=self.feed)

    # ─────────────────────────────────────────────
    # SINGLE-SYMBOL CLEAN BUNDLE
    # ─────────────────────────────────────────────

    def get_clean_bundle(
        self, symbol: str, timeframes: list[str] = None, count: int = None
    ) -> dict | None:
        """
        একটা symbol-এর জন্য tick + multi-timeframe candles, validated ও
        gap-filled। rule engine/AIAnalyst-কে এই bundle পাঠানো হবে।
        """
        broker_symbol = self.account_manager.resolve_symbol(symbol)
        if not broker_symbol:
            log.error(f"[MarketDataManager] Symbol resolve ব্যর্থ: {symbol}")
            return None

        market = self.account_manager.market_status(broker_symbol)
        if not market.get("ok"):
            log.warning(f"[MarketDataManager] Market not OK for {broker_symbol}: {market.get('reason')}")
            return None

        tick = self.feed.get_tick(broker_symbol)
        raw_timeframes = self.feed.get_multi_timeframe(broker_symbol, timeframes, count)

        clean_timeframes = {}
        quality_reports = {}
        for tf, candles in raw_timeframes.items():
            clean, report = self.validator.validate_and_fill(candles, broker_symbol, tf)
            clean_timeframes[tf] = clean
            quality_reports[tf] = report
            if report["quality_score"] < 70:
                log.warning(
                    f"[MarketDataManager] Low quality data: {broker_symbol} {tf} "
                    f"({report['quality_score']}/100)"
                )

        return {
            "broker_symbol": broker_symbol,
            "tick": tick,
            "spread_pips": market.get("spread_pips"),
            "timeframes": clean_timeframes,
            "quality": quality_reports,
        }

    # ─────────────────────────────────────────────
    # MULTI-SYMBOL SCAN  (Part 3 wiring)
    # ─────────────────────────────────────────────

    def scan_market(self, symbols: list[str] = None) -> dict:
        symbols = symbols or DEFAULT_SYMBOLS
        resolved = self.symbol_manager.resolve_all(symbols)
        broker_symbols = [v for v in resolved.values() if v]
        return self.symbol_manager.scan(broker_symbols)

    # ─────────────────────────────────────────────
    # STATUS REPORT  (doc-এর "Final Output Example"-এর real version)
    # ─────────────────────────────────────────────

    def print_status_report(self, symbols: list[str] = None, timeframes: list[str] = None) -> None:
        symbols = symbols or DEFAULT_SYMBOLS
        bar = "═" * 44
        log.info(bar)
        log.info("  🤖  AI MARKET DATA ENGINE")
        log.info(bar)
        log.info(f"  Connection : {'✅ Connected' if self.connection.connected else '🔴 Disconnected'}")
        log.info("")
        for sym in symbols:
            broker_symbol = self.account_manager.resolve_symbol(sym)
            if not broker_symbol:
                log.info(f"  {sym}  ❌ symbol not found")
                continue
            log.info(f"  {sym}")
            self.feed.print_multi_timeframe_status(broker_symbol, timeframes)
        log.info(bar)