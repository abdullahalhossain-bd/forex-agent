# execution/paper_trader.py  —  Day 17 | Paper Trading Engine
# ============================================================
# Full virtual trade lifecycle simulation:
#   Signal → Virtual Order → Price Update → SL/TP/Timeout → Close → DB Save
#
# Realism features:
#   1. Slippage simulation
#   2. Spread simulation
#   3. Commission
#   4. Trade timeout (48h)
#   5. Trade context snapshot (trend/RSI/pattern/session at entry)
#   6. Balance restoration from DB on restart
#   7. Multiple TP levels (partial close)
#   8. Breakeven and trailing stop support
# ============================================================

from datetime import datetime, timedelta, timezone
from utils.logger import get_logger
from database.db import TraderDB
from core.constants import PIP_SIZE, get_pip_size, get_pip_value_usd, clean_symbol

log = get_logger("paper_trader")

# Per-pair spread in pips (typical retail broker average)
SPREAD_PIPS = {
    "EURUSD": 1.2, "GBPUSD": 1.5,
    "USDJPY": 1.3, "USDCHF": 1.8,
    "AUDUSD": 1.4, "USDCAD": 1.7,
    "DEFAULT": 1.5,
}


class PaperTrader:
    """
    Virtual trading account. AITrader.get_signal()-এর output (a `result` dict)
    নিয়ে virtual order open করে, candle-by-candle price update-এ
    SL/TP/timeout চেক করে, এবং close হলে DB-তে save করে।

    এটা TradeMemory (Day 16, vector/lesson memory)-এর replacement না —
    paper_trader আসল trade lifecycle (P&L, SL/TP hit, broker realism) চালায়;
    TradeMemory pattern-lesson মনে রাখে। দুটো পাশাপাশি চলে।
    """

    STARTING_BALANCE = 10000.0
    COMMISSION_PER_LOT = 7.0      # round-trip, per standard lot ($)
    SLIPPAGE_PIPS_MAX = 0.8        # worst-case slippage on market fill
    TIMEOUT_HOURS = 48

    def __init__(self, starting_balance: float = None, db: TraderDB = None):
        # Unify balance default: use config.INITIAL_BALANCE if no explicit
        # value is passed, falling back to the class constant. This prevents
        # the three disconnected $10,000 defaults (class constant, config.py,
        # main.py CLI arg) from drifting out of sync.
        try:
            from config import INITIAL_BALANCE
            _config_default = INITIAL_BALANCE
        except Exception:
            _config_default = self.STARTING_BALANCE

        self.starting_balance = starting_balance if starting_balance is not None else _config_default
        self.db = db or TraderDB()
        self.open_positions: list[dict] = []
        self._restore_open_positions()
        self._restore_balance_from_db()
        log.info(f"PaperTrader ready | Balance: ${self.balance:.2f} | "
                 f"Open positions restored: {len(self.open_positions)}")

    # ─────────────────────────────────────────────
    # 1. OPEN TRADE  (from AITrader result dict)
    # ─────────────────────────────────────────────

    def open_trade_from_signal(self, result: dict) -> dict | None:
        """
        AITrader.get_signal()-এর `result` dict থেকে সরাসরি trade open করো।
        শুধুমাত্র BUY/SELL এবং trade_allowed=True হলে open হবে।
        """
        if result.get("final_action") not in ("BUY", "SELL"):
            log.info(f"[PaperTrader] Skipped — final_action={result.get('final_action')}")
            return None

        if not result.get("entry"):
            log.warning("[PaperTrader] Skipped — no entry price in result")
            return None

        symbol = self._clean_symbol(result["symbol"])
        signal_type = result["final_action"]

        if self.has_open_position(symbol, signal_type):
            log.warning(
                f"[PaperTrader] Duplicate prevented — {symbol} {signal_type} already open"
            )
            return None

        # Bonus 1 — Slippage simulation
        slippage_pips = self._simulate_slippage()
        slippage_price = self._pips_to_price(symbol, slippage_pips)
        filled_entry = (
            result["entry"] + slippage_price if signal_type == "BUY"
            else result["entry"] - slippage_price
        )

        # Bonus 2 — Spread simulation (pay half-spread crossing the book on entry)
        half_spread_price = self._pips_to_price(symbol, SPREAD_PIPS.get(symbol, SPREAD_PIPS["DEFAULT"]) / 2)
        filled_entry = (
            filled_entry + half_spread_price if signal_type == "BUY"
            else filled_entry - half_spread_price
        )

        trade = self._build_trade_record(result, symbol, signal_type, filled_entry, slippage_pips)
        trade_id = self.db.save_trade_open(trade)
        trade["id"] = trade_id
        self.open_positions.append(trade)

        log.info(
            f"[PaperTrader] OPEN #{trade_id} {signal_type} {symbol} "
            f"@ {filled_entry:.5f} (requested {result['entry']:.5f}, "
            f"slip {slippage_pips:.1f}p) | SL {trade['sl']} | TP {trade['tp']} | Lot {trade['lot']}"
        )
        return trade

    def _build_trade_record(self, result: dict, symbol: str, signal_type: str,
                             filled_entry: float, slippage_pips: float) -> dict:
        """Bonus 5 — context snapshot (trend/RSI/pattern/regime/session) trade-এর সাথে save হয়।"""
        return {
            "pair":       symbol,
            "timeframe":  result.get("timeframe"),
            "type":       signal_type,
            "entry":      round(filled_entry, 5),
            "sl":         result.get("sl"),
            "tp":         result.get("tp"),
            "lot":        result.get("lot", 0.01),
            "confidence": result.get("confidence"),
            "open_time":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pattern":    result.get("pattern_context", {}).get("pattern")
                          or result.get("rule_signal"),
            "regime":     result.get("regime"),
            "trend":      result.get("trend"),
            "rsi":        result.get("rsi"),
            "session":    result.get("session"),
            "slippage_pips_at_open": slippage_pips,
            "context": {
                "trend":      result.get("trend"),
                "rsi":        result.get("rsi"),
                "regime":     result.get("regime"),
                "volatility": result.get("volatility"),
                "mtf_bias":   result.get("mtf_bias"),
                "llm_signal": result.get("llm_signal"),
                "decision_confidence": result.get("confidence"),
                "memory_trade_id": result.get("trade_id"),
                "rr_ratio": result.get("rr"),
            },
        }

    # ─────────────────────────────────────────────
    # 2. PRICE UPDATE  (call every new candle)
    # ─────────────────────────────────────────────

    def update_price(self, pair: str, price: float, now: datetime = None,
                     high: float = None, low: float = None) -> list[dict]:
        """
        নতুন candle/price আসলে call করো। সব matching open trades চেক করে
        SL hit / TP hit / timeout — যেটা আগে met হয় সেটায় close করে।
        Returns list of closed trade records (এই call-এ যা close হলো)।

        Day 102+ CRITICAL hotfix: INTRA-CANDLE SL/TP DETECTION.
        Previously this method received only `price` (the candle close)
        and tested `price >= tp` / `price <= sl`. This missed all
        intra-candle hits — e.g. a BUY with TP=1.1050, candle high=1.1055,
        close=1.1045 — TP was hit during the candle but the close finished
        below TP, so the trade stayed open. This produced optimistic paper
        P&L (trades ran longer than they should have) and corrupted the
        WIN/LOSS labels fed to the ML learning engine.

        Now: caller can pass `high` and `low` (the candle's full range).
        If provided, we test `high >= tp` / `low <= sl` for BUY (mirror
        for SELL) — matching how a real broker would fill at the TP/SL
        level. If high/low are not provided, we fall back to the old
        close-only behavior (backward compatible with monitor_open_trades
        which is called with a single tick price).
        """
        now = now or datetime.now(timezone.utc)
        symbol = self._clean_symbol(pair)
        closed = []

        # If caller didn't pass high/low, fall back to using `price` for
        # both — preserves the old close-only behavior.
        if high is None:
            high = price
        if low is None:
            low = price

        # iterate over a copy — close_trade() mutates self.open_positions
        for trade in list(self.open_positions):
            if trade["pair"] != symbol:
                continue

            # Bonus 4 — timeout check first (doesn't depend on price)
            opened_at = datetime.fromisoformat(trade["open_time"])
            if now - opened_at >= timedelta(hours=self.TIMEOUT_HOURS):
                closed.append(self.close_trade(trade, "TIMEOUT", price, now))
                continue

            # Day 102+: use candle high/low for SL/TP detection.
            # A real broker fills at the SL/TP level if the candle's
            # range touches it, regardless of where the close finishes.
            # Without this, paper P&L is optimistic and ML labels are
            # corrupted — trades that should have closed at TP stay
            # open and may close at a worse price later.
            if trade["type"] == "BUY":
                # TP hit if candle high reached TP
                if trade["tp"] and high >= trade["tp"]:
                    closed.append(self.close_trade(trade, "TP HIT", trade["tp"], now))
                # SL hit if candle low reached SL
                elif trade["sl"] and low <= trade["sl"]:
                    closed.append(self.close_trade(trade, "SL HIT", trade["sl"], now))
            else:  # SELL
                # TP hit if candle low reached TP (sells profit when price falls)
                if trade["tp"] and low <= trade["tp"]:
                    closed.append(self.close_trade(trade, "TP HIT", trade["tp"], now))
                # SL hit if candle high reached SL
                elif trade["sl"] and high >= trade["sl"]:
                    closed.append(self.close_trade(trade, "SL HIT", trade["sl"], now))

        return closed

    # ─────────────────────────────────────────────
    # 3. CLOSE TRADE
    # ─────────────────────────────────────────────

    def close_trade(self, trade: dict, reason: str, exit_price: float, now: datetime = None) -> dict:
        now = now or datetime.now(timezone.utc)
        symbol = trade["pair"]

        # Bonus 2 — pay the other half-spread on exit
        half_spread_price = self._pips_to_price(symbol, SPREAD_PIPS.get(symbol, SPREAD_PIPS["DEFAULT"]) / 2)
        filled_exit = (
            exit_price - half_spread_price if trade["type"] == "BUY"
            else exit_price + half_spread_price
        )

        pnl_pips, gross_pnl = self._calculate_pnl(trade, filled_exit)

        # Bonus 3 — commission (round trip, scaled by lot size; 1.0 lot = standard)
        commission = round(self.COMMISSION_PER_LOT * trade["lot"], 2)
        net_pnl = round(gross_pnl - commission, 2)

        result = "WIN" if net_pnl > 0 else ("LOSS" if net_pnl < 0 else "BREAKEVEN")

        self.balance = round(self.balance + net_pnl, 2)

        close_data = {
            "close_time":  now.isoformat(timespec="seconds"),
            "exit_price":  round(filled_exit, 5),
            "result":      result,
            "pnl":         net_pnl,
            "pnl_pips":    round(pnl_pips, 1),
            "spread_cost": self._spread_cost_usd(symbol, trade["lot"]),
            "commission":  commission,
            "slippage":    trade.get("slippage_pips_at_open", 0),
        }

        self.db.save_trade_close(trade["id"], close_data)

        if trade in self.open_positions:
            self.open_positions.remove(trade)

        icon = {"TP HIT": "✅", "SL HIT": "🛑", "TIMEOUT": "⏰"}.get(reason, "•")
        log.info(
            f"[PaperTrader] CLOSE #{trade['id']} {icon} {reason} | "
            f"{symbol} {trade['type']} | Exit {filled_exit:.5f} | "
            f"PnL ${net_pnl} ({pnl_pips:+.1f} pips) | Balance: ${self.balance:.2f}"
        )

        return {**trade, **close_data, "close_reason": reason}

    # ─────────────────────────────────────────────
    # 4. P&L CALCULATION
    # ─────────────────────────────────────────────

    def _calculate_pnl(self, trade: dict, exit_price: float) -> tuple[float, float]:
        """
        Returns (pnl_pips, gross_pnl_usd).

        Standard FX convention: pip value scales with lot size.
        """
        symbol = trade["pair"]
        pip = get_pip_size(symbol)

        if trade["type"] == "BUY":
            diff = exit_price - trade["entry"]
        else:
            diff = trade["entry"] - exit_price

        pnl_pips = diff / pip
        pip_value_per_lot = get_pip_value_usd(symbol)
        gross_pnl = round(pnl_pips * pip_value_per_lot * trade["lot"], 2)
        return pnl_pips, gross_pnl

    # ─────────────────────────────────────────────
    # 5. ACCOUNT DASHBOARD
    # ─────────────────────────────────────────────

    def get_dashboard(self) -> dict:
        stats = self.db.get_account_stats(starting_balance=self.starting_balance)
        return {
            **stats,
            "open_positions": len(self.open_positions),
        }

    def print_dashboard(self) -> None:
        d = self.get_dashboard()
        bar = "═" * 44
        log.info(bar)
        log.info("  💹  AI PAPER ACCOUNT")
        log.info(bar)
        log.info(f"  Balance        : ${d['balance']}")
        log.info(f"  Total Trades   : {d['total_trades']}")
        log.info(f"  Win Rate       : {d['win_rate']}%")
        log.info(f"  Net Profit     : ${d['total_pnl']}")
        log.info(f"  Open Positions : {d['open_positions']}")
        log.info(bar)

    # ─────────────────────────────────────────────
    # UTILS
    # ─────────────────────────────────────────────

    def _restore_open_positions(self) -> None:
        """Restore open trades from DB so they survive app restarts."""
        df = self.db.get_open_trades()
        for _, row in df.iterrows():
            self.open_positions.append(row.to_dict())

    def _restore_balance_from_db(self) -> None:
        """Restore balance from DB trade history instead of resetting to starting_balance."""
        try:
            stats = self.db.get_account_stats(starting_balance=self.starting_balance)
            db_balance = stats.get("balance", self.starting_balance)
            # Only use DB balance if there are closed trades (otherwise it's just the starting value)
            if stats.get("total_trades", 0) > 0:
                self.balance = round(db_balance, 2)
                log.info(f"[PaperTrader] Balance restored from DB: ${self.balance:.2f} "
                         f"(based on {stats.get('total_trades', 0)} closed trades)")
            else:
                self.balance = self.starting_balance
                log.info(f"[PaperTrader] No closed trades in DB — using starting balance: "
                         f"${self.starting_balance:.2f} (set INITIAL_BALANCE_USD env to override)")
        except Exception as e:
            log.warning(f"[PaperTrader] Could not restore balance from DB: {e}. "
                        f"Using starting balance ${self.starting_balance:.2f}")
            self.balance = self.starting_balance

    def has_open_position(self, pair: str, trade_type: str | None = None) -> bool:
        symbol = self._clean_symbol(pair)
        for trade in self.open_positions:
            if trade["pair"] != symbol:
                continue
            if trade_type and trade["type"] != trade_type:
                continue
            return True
        return False

    def get_open_positions(self, pair: str | None = None) -> list[dict]:
        if not pair:
            return list(self.open_positions)
        symbol = self._clean_symbol(pair)
        return [trade for trade in self.open_positions if trade["pair"] == symbol]

    def _simulate_slippage(self) -> float:
        """Bonus 1 — random slippage 0 থেকে SLIPPAGE_PIPS_MAX এর মধ্যে।"""
        import random
        return round(random.uniform(0, self.SLIPPAGE_PIPS_MAX), 2)

    def _pips_to_price(self, symbol: str, pips: float) -> float:
        return pips * get_pip_size(symbol)

    def _spread_cost_usd(self, symbol: str, lot: float) -> float:
        """Full round-trip spread cost in USD for this trade's lot size."""
        spread_pips = SPREAD_PIPS.get(symbol, SPREAD_PIPS["DEFAULT"])
        pip_value_per_lot = get_pip_value_usd(symbol)
        return round(spread_pips * pip_value_per_lot * lot, 2)

    def _clean_symbol(self, symbol: str) -> str:
        return clean_symbol(symbol)

    # ─────────────────────────────────────────────
    # 6. ADVANCED TRADE MANAGEMENT
    # ─────────────────────────────────────────────

    def apply_breakeven(self, trade: dict, breakeven_price: float) -> bool:
        """Move SL to entry (breakeven) for a given trade."""
        if trade not in self.open_positions:
            return False
        trade["sl"] = round(breakeven_price, 5)
        log.info(f"[PaperTrader] Breakeven applied #{trade['id']} {trade['pair']} SL→{breakeven_price:.5f}")
        return True

    def apply_trailing_stop(self, trade: dict, new_sl: float) -> bool:
        """Trail SL to a new level (only moves in favorable direction)."""
        if trade not in self.open_positions:
            return False
        current_sl = trade.get("sl", 0)
        if trade["type"] == "BUY":
            if new_sl > current_sl:
                trade["sl"] = round(new_sl, 5)
                log.info(f"[PaperTrader] Trailing SL #{trade['id']} BUY SL→{new_sl:.5f}")
                return True
        elif trade["type"] == "SELL":
            if new_sl < current_sl or current_sl == 0:
                trade["sl"] = round(new_sl, 5)
                log.info(f"[PaperTrader] Trailing SL #{trade['id']} SELL SL→{new_sl:.5f}")
                return True
        return False

    def partial_close(self, trade: dict, close_percent: float, price: float) -> dict | None:
        """Partially close a trade (for multiple TP levels)."""
        if trade not in self.open_positions:
            return None
        if close_percent <= 0 or close_percent > 100:
            return None

        original_lot = trade["lot"]
        close_lot = round(original_lot * close_percent / 100, 2)
        if close_lot < 0.01:
            close_lot = 0.01

        # Calculate PnL for the closed portion
        pnl_pips, gross_pnl = self._calculate_pnl(trade, price)
        proportion = close_lot / original_lot if original_lot > 0 else 1.0
        partial_pnl = round(gross_pnl * proportion, 2)
        commission = round(self.COMMISSION_PER_LOT * close_lot, 2)
        net_pnl = round(partial_pnl - commission, 2)

        # Update trade lot
        remaining_lot = round(original_lot - close_lot, 2)
        if remaining_lot <= 0:
            # Close entire trade
            return self.close_trade(trade, "PARTIAL_FULL", price)

        trade["lot"] = remaining_lot
        self.balance = round(self.balance + net_pnl, 2)

        log.info(
            f"[PaperTrader] Partial close #{trade['id']} {trade['pair']} "
            f"{close_percent}% ({close_lot} lots) @ {price:.5f} | "
            f"PnL: ${net_pnl} | Remaining: {remaining_lot} lots"
        )

        return {
            "trade_id": trade["id"],
            "pair": trade["pair"],
            "type": trade["type"],
            "close_percent": close_percent,
            "close_lot": close_lot,
            "remaining_lot": remaining_lot,
            "pnl": net_pnl,
            "exit_price": price,
        }
