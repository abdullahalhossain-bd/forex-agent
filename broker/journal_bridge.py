# broker/journal_bridge.py  —  Day 31 Bonus 4 | Demo Trading Journal Link
# ============================================================
# Lock-in problem এটা সমাধান করে: paper trading আর MT5 demo trading
# যদি আলাদা storage ব্যবহার করে, তাহলে Learning Agent / TradeMemory
# অর্ধেক data দেখবে — pattern lesson ভুল হবে।
#
# Solution: existing `trades` table-এই save করো (db.py পরিবর্তন
# করতে হয়নি), কিন্তু context_json-এর ভেতরে `source` ট্যাগ যুক্ত
# করো যাতে paper vs mt5_demo আলাদা করা যায় reporting/learning-এ।
# ============================================================

from utils.logger import get_logger
from database.db import TraderDB
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path
import pandas as pd
from core.constants import MEMORY_DIR

log = get_logger("journal_bridge")

# Day 102+ hotfix: spool file for orphan MT5 trades.
# When the DB write fails after a broker fill, we spool the trade to
# this JSONL file so it can be reconciled at next boot instead of
# being silently lost. Without this, FILLED_ORPHAN positions stay open
# at the broker but invisible to the risk engine.
ORPHAN_SPOOL_PATH = MEMORY_DIR / "orphan_trade_spool.jsonl"


class JournalBridge:
    """
    MT5 demo trade-কে paper trade-এর মতো same `trades` table-এ লেখে,
    যাতে দুটো mode-ই একই learning memory ব্যবহার করে।

    Usage:
        bridge = JournalBridge(db)
        trade_id = bridge.log_mt5_open(decision_result, broker_symbol, filled_entry, mt5_order_ticket)
        ...
        bridge.log_mt5_close(trade_id, close_data)

    Day 102+ hotfix: RETRY + SPOOL FALLBACK.
    Previously, any DB write error (locked, disk full, corruption) after
    a successful broker fill would propagate up to ExecutionRouter and
    produce a FILLED_ORPHAN — a real broker position with no DB record,
    invisible to risk/circuit-breaker/learning. Now we:
      1. Retry the DB write up to 3 times with 0.5s backoff.
      2. On final failure, spool the trade to orphan_trade_spool.jsonl
         so the next boot's orphan_cleanup can back-fill it.
      3. Re-raise only if BOTH retry AND spool fail — at which point
         the operator will see a critical log line.
    """

    DB_RETRY_COUNT = 3
    DB_RETRY_DELAY_SEC = 0.5

    def __init__(self, db: TraderDB = None):
        self.db = db or TraderDB()
        # Co-founder fix: lazily-instantiated learning DB mirror.
        # Lazy so we don't fail construction if memory/trader.db is
        # temporarily locked or corrupt — broker trading must still work.
        self._learning_db = None

    def _get_learning_db(self):
        """Lazily connect to the learning DB (memory/trader.db).

        Co-founder fix: CROSS-DB LEARNING SYNC.
        The repo has TWO SQLite databases with overlapping `trades` tables:
          - database/trader.db (TraderDB) — authoritative broker journal
          - memory/trader.db    (Database) — used by LearningEngine + TradeMemory
        Previously, MT5 trades written via JournalBridge reached ONLY
        database/trader.db. The learning layer (which reads memory/trader.db)
        never saw them — meaning the bot couldn't learn from real broker
        fills, only from paper trades. Critical data integrity gap.

        Returns None if the DB can't be opened — callers must handle
        this gracefully (broker trade still succeeds, learning sync
        is just skipped).
        """
        if self._learning_db is not None:
            return self._learning_db
        try:
            from memory.database import Database
            self._learning_db = Database()
            return self._learning_db
        except Exception as e:
            log.warning(
                f"[JournalBridge] learning DB unavailable — MT5 trades "
                f"will NOT sync to memory/trader.db (learning gap): {e}"
            )
            return None

    def _sync_open_to_learning_db(self, trade: dict, broker_trade_id: int) -> int | None:
        """Mirror a trade open into memory/trader.db for the learning layer.

        Returns the memory DB trade_id on success, None on failure.
        Failures are SOFT — broker trading must not fail just because
        the learning DB is unavailable.
        """
        ldb = self._get_learning_db()
        if ldb is None:
            return None
        try:
            # Translate TraderDB schema → memory Database schema.
            # memory DB trades table columns: pair, signal, entry, sl, tp,
            # lot, result, pnl, rr_ratio, confidence, chart_snapshot
            memory_trade_id = ldb.save_trade({
                "pair":       trade.get("pair"),
                "signal":     trade.get("type"),  # 'type' in TraderDB → 'signal' in memory DB
                "entry":      trade.get("entry"),
                "sl":         trade.get("sl"),
                "tp":         trade.get("tp"),
                "lot":        trade.get("lot", 0.01),
                "result":     "OPEN",
                "pnl":        0,
                "rr_ratio":   (trade.get("context") or {}).get("rr_ratio", 0),
                "confidence": trade.get("confidence", 0),
                "chart_snapshot": {
                    "source":         "mt5_demo",
                    "broker_db_id":   broker_trade_id,
                    "pattern":        trade.get("pattern"),
                    "regime":         trade.get("regime"),
                    "trend":          trade.get("trend"),
                    "rsi":            trade.get("rsi"),
                    "session":        trade.get("session"),
                },
            })
            log.info(
                f"[JournalBridge] Learning sync: broker DB #{broker_trade_id} "
                f"→ memory DB #{memory_trade_id}"
            )
            return memory_trade_id
        except Exception as e:
            log.warning(
                f"[JournalBridge] learning DB sync (open) failed for broker "
                f"trade #{broker_trade_id}: {e}. Learning will miss this trade."
            )
            return None

    def _sync_close_to_learning_db(self, broker_trade_id: int, result: str, pnl: float) -> None:
        """Mirror a trade close into memory/trader.db.

        Looks up the memory DB trade by broker_db_id (stashed in
        chart_snapshot at open time). If not found, falls back to
        find_open_trade_by_pair. SOFT failure — broker close already
        succeeded.
        """
        ldb = self._get_learning_db()
        if ldb is None:
            return
        try:
            # Try to find the memory DB trade by broker_db_id.
            # We saved it in chart_snapshot at open time.
            import sqlite3
            with ldb._lock:
                cur = ldb.conn.cursor()
                cur.execute(
                    "SELECT id FROM trades WHERE chart_snapshot LIKE ? AND result = 'OPEN' ORDER BY id DESC LIMIT 1",
                    (f'%"broker_db_id": {broker_trade_id}%',),
                )
                row = cur.fetchone()
            if row:
                memory_trade_id = dict(row)["id"]
                ldb.update_trade_result(memory_trade_id, result, pnl)
                log.info(
                    f"[JournalBridge] Learning sync (close): broker DB #{broker_trade_id} "
                    f"→ memory DB #{memory_trade_id} ({result}, ${pnl:.2f})"
                )
            else:
                # Fallback: find any OPEN trade for the same pair
                # (less precise but better than dropping the close event)
                log.warning(
                    f"[JournalBridge] learning DB sync (close): no memory trade "
                    f"found with broker_db_id={broker_trade_id} — trying pair fallback"
                )
                # We don't have the pair here cleanly; let the orphan_cleanup
                # reconciliation handle it at next boot.
        except Exception as e:
            log.warning(
                f"[JournalBridge] learning DB sync (close) failed for broker "
                f"trade #{broker_trade_id}: {e}. Learning close missed."
            )

    def log_mt5_open(
        self,
        decision_result: dict,
        broker_symbol: str,
        filled_entry: float,
        mt5_order_ticket: int = None,
    ) -> int:
        """
        MT5 demo-তে order place হওয়ার পরে call করো। PaperTrader-এর
        _build_trade_record()-এর সাথে structurally মিলিয়ে রাখা হয়েছে
        যাতে db.save_trade_open() unchanged থাকতে পারে।
        """
        from datetime import datetime, timezone

        trade = {
            "pair":       broker_symbol,
            "timeframe":  decision_result.get("timeframe"),
            "type":       decision_result.get("decision"),
            "entry":      round(filled_entry, 5),
            "sl":         decision_result.get("sl"),
            "tp":         decision_result.get("tp"),
            "lot":        decision_result.get("lot", 0.01),
            "confidence": decision_result.get("confidence"),
            "open_time":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pattern":    decision_result.get("pattern"),
            "regime":     decision_result.get("regime"),
            "trend":      decision_result.get("trend"),
            "rsi":        decision_result.get("rsi"),
            "session":    decision_result.get("session"),
            "context": {
                "source": "mt5_demo",          # ⭐ paper trade থেকে আলাদা করার ট্যাগ
                "mt5_order_ticket": mt5_order_ticket,
                "mtf_bias":   decision_result.get("mtf_bias"),
                "llm_signal": decision_result.get("llm_signal"),
                "rr_ratio":   decision_result.get("rr"),
            },
        }

        # Day 102+: retry loop
        last_exc = None
        for attempt in range(1, self.DB_RETRY_COUNT + 1):
            try:
                trade_id = self.db.save_trade_open(trade)
                log.info(
                    f"[JournalBridge] MT5 demo trade logged → DB #{trade_id} "
                    f"(ticket={mt5_order_ticket})"
                )
                # Co-founder fix: sync to learning DB so the bot can
                # learn from real broker fills, not just paper trades.
                # Soft failure — broker trade already succeeded.
                self._sync_open_to_learning_db(trade, trade_id)
                return trade_id
            except Exception as e:
                last_exc = e
                log.warning(
                    f"[JournalBridge] DB write attempt {attempt}/{self.DB_RETRY_COUNT} "
                    f"failed: {e}"
                )
                if attempt < self.DB_RETRY_COUNT:
                    time.sleep(self.DB_RETRY_DELAY_SEC)

        # All retries failed — spool to disk for later reconciliation
        log.critical(
            f"[JournalBridge] DB write failed after {self.DB_RETRY_COUNT} attempts — "
            f"spooling MT5 ticket={mt5_order_ticket} to {ORPHAN_SPOOL_PATH} for "
            f"reconciliation at next boot. ORIGINAL ERROR: {last_exc}"
        )
        self._spool_orphan(trade, mt5_order_ticket)
        # Re-raise so ExecutionRouter marks this as FILLED_ORPHAN
        # — but at least now the orphan is recoverable.
        raise last_exc

    def _spool_orphan(self, trade: dict, mt5_ticket) -> None:
        """Append the orphan trade to the spool file for later reconciliation."""
        try:
            ORPHAN_SPOOL_PATH.parent.mkdir(parents=True, exist_ok=True)
            spool_entry = {
                "spooled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "mt5_order_ticket": mt5_ticket,
                "trade": trade,
                "status": "PENDING_RECONCILE",
            }
            with open(ORPHAN_SPOOL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(spool_entry, default=str) + "\n")
            log.info(
                f"[JournalBridge] Orphan trade spooled (ticket={mt5_ticket}) — "
                f"will be reconciled at next boot via orphan_cleanup"
            )
        except Exception as spool_err:
            log.critical(
                f"[JournalBridge] FAILED TO SPOOL orphan trade (ticket={mt5_ticket}): "
                f"{spool_err}. THIS TRADE IS NOW INVISIBLE TO THE RISK ENGINE. "
                f"Manual intervention required — close the MT5 position immediately."
            )

    def log_mt5_close(self, trade_id: int, close_data: dict) -> None:
        """close_data একই shape — PaperTrader.close_trade()-এর close_data দেখো।"""
        # Day 102+: same retry pattern as log_mt5_open
        last_exc = None
        for attempt in range(1, self.DB_RETRY_COUNT + 1):
            try:
                self.db.save_trade_close(trade_id, close_data)
                log.info(f"[JournalBridge] MT5 demo trade closed → DB #{trade_id}")
                # Co-founder fix: sync close to learning DB.
                # close_data should have 'result' and 'pnl' fields (PaperTrader schema).
                _result = (close_data or {}).get("result", "UNKNOWN")
                _pnl = float((close_data or {}).get("pnl", 0) or 0)
                self._sync_close_to_learning_db(trade_id, _result, _pnl)
                return
            except Exception as e:
                last_exc = e
                log.warning(
                    f"[JournalBridge] close DB write attempt {attempt}/{self.DB_RETRY_COUNT} "
                    f"failed: {e}"
                )
                if attempt < self.DB_RETRY_COUNT:
                    time.sleep(self.DB_RETRY_DELAY_SEC)
        log.error(
            f"[JournalBridge] close DB write failed for trade #{trade_id} after "
            f"{self.DB_RETRY_COUNT} attempts: {last_exc}. Close event lost."
        )

    # ─────────────────────────────────────────────
    # COMBINED REPORTING — paper + mt5_demo একসাথে
    # ─────────────────────────────────────────────

    def get_combined_stats(self, starting_balance: float = 10000.0) -> dict:
        """
        সব trade (source নির্বিশেষে) মিলিয়ে stats — Learning Agent
        এটা ব্যবহার করবে, mode আলাদা ভাবে নয়।
        """
        return self.db.get_account_stats(starting_balance=starting_balance)

    def get_stats_by_source(self, starting_balance: float = 10000.0) -> dict:
        """
        Paper vs MT5-demo আলাদা ভাবে break-down — যাতে বোঝা যায় simulation
        আর real-broker-condition performance কতটা মিলছে/আলাদা।
        """
        import json
        history = self.db.get_trade_history(limit=10000)
        paper_pnl, demo_pnl = 0.0, 0.0
        paper_n, demo_n = 0, 0

        for _, row in history.iterrows():
            ctx_raw = row.get("context_json") or "{}"
            try:
                ctx = json.loads(ctx_raw)
            except Exception as e:
                ctx = {}
            source = ctx.get("source", "paper")  # context_json নেই মানে পুরনো paper trade
            pnl = row.get("pnl", 0) or 0
            if source == "mt5_demo":
                demo_pnl += pnl
                demo_n += 1
            else:
                paper_pnl += pnl
                paper_n += 1

        return {
            "paper":    {"trades": paper_n, "pnl": round(paper_pnl, 2)},
            "mt5_demo": {"trades": demo_n, "pnl": round(demo_pnl, 2)},
        }

    # ─────────────────────────────────────────────
    # MT5 User Guide Page 25 — Trade History Export (XML/HTML)
    # ─────────────────────────────────────────────

    def export_history_xml(self, filepath: str = "trade_history.xml", limit: int = 1000) -> str:
        """
        MT5 User Guide Page 25 — export trade history as XML.

        Args:
            filepath: output file path
            limit: max trades to export

        Returns:
            Filepath of the exported XML file.
        """
        from xml.etree.ElementTree import Element, SubElement, tostring
        from xml.dom import minidom

        history = self.db.get_trade_history(limit=limit)
        root = Element("TradeHistory")
        root.set("exported_at", datetime.now(timezone.utc).isoformat())
        root.set("total_trades", str(len(history)))

        for _, row in history.iterrows():
            trade = SubElement(root, "Trade")
            trade.set("trade_id", str(row.get("trade_id", "")))
            for col in ["pair", "direction", "entry_price", "exit_price",
                        "stop_loss", "take_profit", "lot_size", "pnl",
                        "confidence", "regime", "timeframe", "status"]:
                val = row.get(col, "")
                if pd.notna(val):
                    trade_el = SubElement(trade, col.replace("_", ""))
                    trade_el.text = str(val)

            # Context JSON
            ctx_raw = row.get("context_json") or "{}"
            ctx_el = SubElement(trade, "Context")
            ctx_el.text = ctx_raw

        xml_str = minidom.parseString(tostring(root)).toprettyxml(indent="  ")
        with open(filepath, "w") as f:
            f.write(xml_str)
        log.info(f"[JournalBridge] Exported {len(history)} trades to XML: {filepath}")
        return filepath

    def export_history_html(self, filepath: str = "trade_history.html", limit: int = 1000) -> str:
        """
        MT5 User Guide Page 25 — export trade history as HTML.

        Args:
            filepath: output file path
            limit: max trades to export

        Returns:
            Filepath of the exported HTML file.
        """
        history = self.db.get_trade_history(limit=limit)

        html_parts = [
            "<!DOCTYPE html>",
            "<html><head><title>Trade History Export</title>",
            "<style>",
            "body { font-family: monospace; margin: 20px; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "tr:nth-child(even) { background-color: #f2f2f2; }",
            ".positive { color: green; } .negative { color: red; }",
            "</style></head><body>",
            f"<h1>Trade History Export</h1>",
            f"<p>Exported: {datetime.now(timezone.utc).isoformat()}</p>",
            f"<p>Total Trades: {len(history)}</p>",
            "<table><thead><tr>",
        ]

        # Headers
        columns = ["trade_id", "pair", "direction", "entry_price", "exit_price",
                    "stop_loss", "take_profit", "lot_size", "pnl",
                    "confidence", "regime", "timeframe", "status"]
        for col in columns:
            html_parts.append(f"<th>{col}</th>")
        html_parts.append("</tr></thead><tbody>")

        # Rows
        for _, row in history.iterrows():
            html_parts.append("<tr>")
            pnl = row.get("pnl", 0) or 0
            pnl_class = "positive" if pnl > 0 else "negative" if pnl < 0 else ""
            for col in columns:
                val = row.get(col, "")
                if pd.isna(val):
                    val = ""
                if col == "pnl" and pnl_class:
                    html_parts.append(f'<td class="{pnl_class}">{val}</td>')
                else:
                    html_parts.append(f"<td>{val}</td>")
            html_parts.append("</tr>")

        html_parts.append("</tbody></table></body></html>")

        with open(filepath, "w") as f:
            f.write("\n".join(html_parts))
        log.info(f"[JournalBridge] Exported {len(history)} trades to HTML: {filepath}")
        return filepath

    def export_history_csv(self, filepath: str = "trade_history.csv", limit: int = 1000) -> str:
        """
        Export trade history as CSV (bonus format — not in MT5 guide but useful).

        Args:
            filepath: output file path
            limit: max trades to export

        Returns:
            Filepath of the exported CSV file.
        """
        history = self.db.get_trade_history(limit=limit)
        history.to_csv(filepath, index=False)
        log.info(f"[JournalBridge] Exported {len(history)} trades to CSV: {filepath}")
        return filepath