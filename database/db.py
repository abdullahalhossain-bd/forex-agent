# database/db.py
# ============================================================
# AI Trader — SQLite Database
# CSV এর চেয়ে fast, structured, queryable
# পরে PostgreSQL-এ migrate করা সহজ হবে
#
# Day 43 addition: `economic_history` table — news event + actual
# market reaction memory, যাতে FundamentalSentimentScore module
# পরে এই history থেকে currency bias বের করতে পারে।
# ============================================================

import sqlite3
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timezone
from utils.logger import get_logger

log = get_logger(__name__)

from config import PROJECT_ROOT
DB_PATH = str(PROJECT_ROOT / "database" / "trader.db")
(PROJECT_ROOT / "database").mkdir(parents=True, exist_ok=True)


# ── JSON encoder that handles numpy types ───────────────────────────
# pandas/numpy produce np.int64, np.float64, np.bool_ etc. which the
# standard json.dumps() can't serialize.  This encoder converts them
# to native Python types so save_analysis() / save_trade_open() never
# crash with "Object of type bool is not JSON serializable".
class _NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        # pd.isna() crashes on dicts/lists — only call on scalars.
        if isinstance(obj, (int, float, str)) or obj is None:
            try:
                if pd.isna(obj):
                    return None
            except Exception as e:
                log.warning(f"Suppressed exception at line 48: {e}")
                pass
        return super().default(obj)


def _safe_json_dumps(obj):
    """json.dumps that never crashes — converts numpy types + falls back to None."""
    try:
        return json.dumps(obj, cls=_NumpySafeEncoder, default=str)
    except Exception as e:
        log.warning("_safe_json_dumps failed to serialize object: %s", e)
        return None


class TraderDB:
    """
    AI Trader-এর central database।

    Tables:
        candles            — OHLCV data
        indicators         — calculated indicator values
        patterns           — detected patterns
        analysis           — full AI context per run
        trades             — paper/demo trade journal
        economic_history   — (Day 43) news event + market reaction memory
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_tables()
        log.info(f"Database ready: {db_path}")

    def _connect(self):
        """FIX: Enable WAL mode for better concurrent read/write performance
        and reduced lock contention. Also set timeout for busy waits.
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        # WAL mode allows concurrent reads during writes — critical for
        # a trading bot that reads positions while writing new trades.
        conn.execute("PRAGMA journal_mode=WAL")
        # Normal sync is safer — don't use OFF in production
        conn.execute("PRAGMA synchronous=NORMAL")
        # 5-second busy timeout before giving up on locked DB
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_tables(self):
        """Tables তৈরি করো (already exists হলে skip)"""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS candles (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol    TEXT    NOT NULL,
                    timeframe TEXT    NOT NULL,
                    time      TEXT    NOT NULL,
                    open      REAL,
                    high      REAL,
                    low       REAL,
                    close     REAL,
                    volume    REAL,
                    UNIQUE(symbol, timeframe, time)
                );

                CREATE TABLE IF NOT EXISTS indicators (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol    TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    time      TEXT NOT NULL,
                    rsi       REAL,
                    macd      REAL,
                    macd_sig  REAL,
                    sma_20    REAL,
                    sma_50    REAL,
                    sma_200   REAL,
                    atr       REAL,
                    bb_upper  REAL,
                    bb_lower  REAL,
                    trend     TEXT,
                    UNIQUE(symbol, timeframe, time)
                );

                CREATE TABLE IF NOT EXISTS patterns (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol    TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    time      TEXT NOT NULL,
                    pattern   TEXT,
                    engulfing TEXT,
                    star      TEXT,
                    signal    TEXT
                );

                CREATE TABLE IF NOT EXISTS analysis (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_time    TEXT NOT NULL,
                    symbol      TEXT NOT NULL,
                    timeframe   TEXT NOT NULL,
                    bias_score  INTEGER,
                    bias_label  TEXT,
                    context_json TEXT
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair            TEXT NOT NULL,
                    timeframe       TEXT,
                    type            TEXT NOT NULL,
                    entry           REAL,
                    sl              REAL,
                    tp              REAL,
                    lot             REAL,
                    confidence      INTEGER,
                    open_time       TEXT NOT NULL,
                    close_time      TEXT,
                    exit_price      REAL,
                    result          TEXT,
                    pnl             REAL,
                    pnl_pips        REAL,
                    spread_cost     REAL,
                    commission      REAL,
                    slippage        REAL,
                    swap            REAL DEFAULT 0,  -- P0-4: swap/rollover cost tracking
                    error_message   TEXT,              -- P0-4: store execution errors
                    pattern         TEXT,
                    regime          TEXT,
                    trend           TEXT,
                    rsi             REAL,
                    session         TEXT,
                    status          TEXT DEFAULT 'OPEN',
                    context_json    TEXT
                );

                CREATE TABLE IF NOT EXISTS economic_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    event           TEXT NOT NULL,
                    currency        TEXT NOT NULL,
                    impact          TEXT,
                    event_time      TEXT NOT NULL,
                    expected        TEXT,
                    actual          TEXT,
                    market_reaction TEXT,
                    pips_moved      REAL,
                    lesson          TEXT,
                    created_at      TEXT NOT NULL
                );
            """)
        # P0-4: Migration — add swap and error_message columns to existing databases
        self._migrate_trades_table()

    def _migrate_trades_table(self):
        """Add missing columns to existing trades table (backward-compatible migration)."""
        with self._connect() as conn:
            # Check existing columns
            cursor = conn.execute("PRAGMA table_info(trades)")
            existing_cols = {row[1] for row in cursor.fetchall()}

            if "swap" not in existing_cols:
                conn.execute("ALTER TABLE trades ADD COLUMN swap REAL DEFAULT 0")
                log.info("[DB] Migration: added 'swap' column to trades table")
            if "error_message" not in existing_cols:
                conn.execute("ALTER TABLE trades ADD COLUMN error_message TEXT")
                log.info("[DB] Migration: added 'error_message' column to trades table")
            if "mt5_ticket" not in existing_cols:
                # BUGFIX: the MT5 order ticket was already being captured
                # (buried inside context_json['mt5_order_ticket']) but never
                # written to its own column. core/orphan_cleanup.py has
                # ticket-based matching logic that reads row["mt5_ticket"]
                # specifically to avoid ambiguous pair+type matching (which
                # can wrongly close a still-open trade — e.g. when two
                # positions share the same pair/direction — losing its real
                # exit_price/pnl forever). Without this column that safer
                # path was permanently dead code. Add it so ticket-based
                # reconciliation actually runs.
                conn.execute("ALTER TABLE trades ADD COLUMN mt5_ticket INTEGER")
                log.info("[DB] Migration: added 'mt5_ticket' column to trades table")

    # ─────────────────────────────────────────────
    # SAVE METHODS
    # ─────────────────────────────────────────────

    def save_candles(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """OHLCV data save করো — duplicate হলে skip"""
        rows = []
        for ts, row in df.iterrows():
            rows.append((
                symbol, timeframe, str(ts),
                row.get('open'), row.get('high'),
                row.get('low'),  row.get('close'), row.get('volume'),
            ))
        with self._connect() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO candles
                (symbol, timeframe, time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
        log.info(f"Candles saved: {symbol} {timeframe} | {len(rows)} rows")

    def save_indicators(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Indicator values save করো"""
        rows = []
        for ts, row in df.iterrows():
            rows.append((
                symbol, timeframe, str(ts),
                _safe(row, 'rsi'),       _safe(row, 'macd'),
                _safe(row, 'macd_signal'), _safe(row, 'sma_20'),
                _safe(row, 'sma_50'),    _safe(row, 'sma_200'),
                _safe(row, 'atr'),       _safe(row, 'bb_upper'),
                _safe(row, 'bb_lower'),  row.get('trend', ''),
            ))
        with self._connect() as conn:
            # BUGFIX: market_agent.py re-fetches only the latest N candles
            # (limit=300) every cycle and recomputes indicators fresh on
            # that window. Indicators like RSI(14)/SMA(200) need warm-up,
            # so the oldest rows *within each window* come out NaN. As the
            # window slides forward, a timestamp's last appearance is near
            # that warm-up edge right before it rolls out of the window —
            # and the old "INSERT OR REPLACE" clobbered its previously
            # correct value with NULL at that moment, permanently (it's
            # never recomputed again once outside the window). Observed as
            # scattered multi-row NULL blocks throughout indicators.rsi
            # (not just an initial warm-up block). Fix: upsert with
            # COALESCE so a NULL/NaN newly computed value never overwrites
            # a previously stored real value.
            conn.executemany("""
                INSERT INTO indicators
                (symbol, timeframe, time, rsi, macd, macd_sig,
                 sma_20, sma_50, sma_200, atr, bb_upper, bb_lower, trend)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe, time) DO UPDATE SET
                    rsi      = COALESCE(excluded.rsi, indicators.rsi),
                    macd     = COALESCE(excluded.macd, indicators.macd),
                    macd_sig = COALESCE(excluded.macd_sig, indicators.macd_sig),
                    sma_20   = COALESCE(excluded.sma_20, indicators.sma_20),
                    sma_50   = COALESCE(excluded.sma_50, indicators.sma_50),
                    sma_200  = COALESCE(excluded.sma_200, indicators.sma_200),
                    atr      = COALESCE(excluded.atr, indicators.atr),
                    bb_upper = COALESCE(excluded.bb_upper, indicators.bb_upper),
                    bb_lower = COALESCE(excluded.bb_lower, indicators.bb_lower),
                    trend    = CASE WHEN excluded.trend IS NOT NULL AND excluded.trend != ''
                                    THEN excluded.trend ELSE indicators.trend END
            """, rows)
        log.info(f"Indicators saved: {symbol} {timeframe}")

    def save_patterns(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Detected patterns save করো"""
        rows = []
        for ts, row in df.iterrows():
            pat = row.get('pattern', 'none')
            eng = row.get('engulfing', 'none')
            star = row.get('star_pattern', 'none')
            if pat == 'none' and eng == 'none' and star == 'none':
                continue
            rows.append((symbol, timeframe, str(ts), pat, eng, star, ''))
        if rows:
            with self._connect() as conn:
                conn.executemany("""
                    INSERT INTO patterns
                    (symbol, timeframe, time, pattern, engulfing, star, signal)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, rows)
            log.info(f"Patterns saved: {len(rows)} patterns")

    def save_analysis(self, symbol: str, timeframe: str,
                      bias_score: int, bias_label: str, context: dict):
        """Full analysis result save করো"""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO analysis
                (run_time, symbol, timeframe, bias_score, bias_label, context_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                symbol, timeframe,
                bias_score, bias_label,
                _safe_json_dumps(context),
            ))
        log.info(f"Analysis saved: {symbol} bias={bias_score} ({bias_label})")

    # ─────────────────────────────────────────────
    # TRADES  (Day 17 — Paper Trading)
    # ─────────────────────────────────────────────

    def save_trade_open(self, trade: dict) -> int:
        """
        নতুন trade open হলে save করো। Returns the new trade's row id.
        `trade` dict-টা PaperTrader._build_trade_record() থেকে আসে।
        """
        context = trade.get("context", {}) or {}
        # BUGFIX: the ticket was already present in context (as
        # 'mt5_order_ticket' or 'ticket') but never surfaced to its own
        # column — see mt5_ticket migration note above.
        mt5_ticket = (
            context.get("mt5_order_ticket")
            or context.get("mt5_ticket")
            or context.get("ticket")
            or trade.get("mt5_ticket")
        )
        with self._connect() as conn:
            cur = conn.execute("""
                INSERT INTO trades
                (pair, timeframe, type, entry, sl, tp, lot, confidence,
                 open_time, pattern, regime, trend, rsi, session,
                 status, context_json, mt5_ticket)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
            """, (
                trade["pair"], trade.get("timeframe"), trade["type"],
                trade["entry"], trade["sl"], trade["tp"], trade["lot"],
                trade.get("confidence"), trade["open_time"],
                trade.get("pattern"), trade.get("regime"),
                trade.get("trend"), trade.get("rsi"), trade.get("session"),
                _safe_json_dumps(context),
                int(mt5_ticket) if mt5_ticket is not None else None,
            ))
            trade_id = cur.lastrowid
        log.info(f"Trade OPEN saved: #{trade_id} {trade['pair']} {trade['type']} @ {trade['entry']}")
        return trade_id

    def save_trade_close(self, trade_id: int, close_data: dict) -> None:
        """
        Trade close হলে update করো (WIN/LOSS/BREAKEVEN + pnl + costs)।
        close_data keys: close_time, exit_price, result, pnl, pnl_pips,
                          spread_cost, commission, slippage, swap, error_message
        """
        with self._connect() as conn:
            conn.execute("""
                UPDATE trades
                SET close_time = ?, exit_price = ?, result = ?,
                    pnl = ?, pnl_pips = ?, spread_cost = ?,
                    commission = ?, slippage = ?,
                    swap = ?, error_message = ?, status = 'CLOSED'
                WHERE id = ?
            """, (
                close_data["close_time"], close_data["exit_price"],
                close_data["result"], close_data["pnl"], close_data.get("pnl_pips"),
                close_data.get("spread_cost", 0), close_data.get("commission", 0),
                close_data.get("slippage", 0),
                close_data.get("swap", 0),  # P0-4: swap cost
                close_data.get("error_message"),  # P0-4: execution error
                trade_id,
            ))
        log.info(f"Trade CLOSE saved: #{trade_id} {close_data['result']} | PnL: ${close_data['pnl']}")

    def get_open_trades(self, pair: str = None) -> pd.DataFrame:
        """বর্তমান open trades দেখো (price update loop-এর জন্য)"""
        query  = "SELECT * FROM trades WHERE status = 'OPEN'"
        params = ()
        if pair:
            query += " AND pair = ?"
            params = (pair,)
        with self._connect() as conn:
            return pd.read_sql(query, conn, params=params)

    def close_orphaned_open_trade(
        self,
        pair: str,
        result: str,
        pnl: float,
        exit_price: float | None = None,
        close_time: str | None = None,
    ) -> int | None:
        """Day 99+ V4 FIX (Audit Issue #4): close an orphaned OPEN trade
        for the given pair when the broker confirms a close but we don't
        have a `memory_trade_id` to reference (e.g. process restarted
        mid-trade, MT5 position closed externally, etc.).

        Picks the most-recently-opened OPEN trade for `pair` and marks
        it CLOSED with the supplied `result` / `pnl`. Other close fields
        (exit_price, close_time, etc.) are filled in from whatever is
        available — broker-supplied values take priority, sensible
        defaults are used otherwise.

        Args:
            pair:        trade pair symbol (e.g. "EURUSD")
            result:      "WIN" / "LOSS" / "BREAKEVEN"
            pnl:         realized PnL in account currency
            exit_price:  optional exit price (None → use entry as fallback)
            close_time:  optional ISO timestamp (None → now UTC)

        Returns:
            The trade_id of the closed trade, or None if no OPEN trade
            was found for this pair.
        """
        from datetime import datetime, timezone
        try:
            close_time = close_time or datetime.now(timezone.utc).isoformat(timespec="seconds")
            with self._connect() as conn:
                # Find the most recent OPEN trade for this pair.
                row = conn.execute(
                    "SELECT id, entry FROM trades "
                    "WHERE status = 'OPEN' AND pair = ? "
                    "ORDER BY open_time DESC LIMIT 1",
                    (pair,),
                ).fetchone()
                if row is None:
                    return None
                orphan_id, entry = row
                # Fall back to entry price if exit price not supplied —
                # this gives PnL≈0 for BREAKEVEN cases and at least
                # produces a valid close record instead of leaving the
                # trade stuck at OPEN forever.
                _exit = exit_price if exit_price is not None else entry
                conn.execute("""
                    UPDATE trades
                    SET close_time = ?, exit_price = ?, result = ?,
                        pnl = ?, status = 'CLOSED'
                    WHERE id = ?
                """, (close_time, _exit, result, pnl, orphan_id))
            log.info(
                f"Orphan trade closed: #{orphan_id} {pair} {result} | PnL: ${pnl:.2f}"
            )
            return orphan_id
        except Exception as e:
            log.warning(f"close_orphaned_open_trade failed for {pair}: {e}")
            return None

    def has_open_trade(self, pair: str, trade_type: str | None = None) -> bool:
        """Duplicate trade protection-এর জন্য open position আছে কিনা দেখো।"""
        query = "SELECT COUNT(*) FROM trades WHERE status = 'OPEN' AND pair = ?"
        params: list = [pair]
        if trade_type:
            query += " AND type = ?"
            params.append(trade_type)
        with self._connect() as conn:
            count = conn.execute(query, tuple(params)).fetchone()[0]
        return count > 0

    def get_trade_history(self, pair: str = None, limit: int = 50) -> pd.DataFrame:
        """Closed trades history দেখো"""
        query  = "SELECT * FROM trades WHERE status = 'CLOSED'"
        params = []
        if pair:
            query += " AND pair = ?"
            params.append(pair)
        query += " ORDER BY close_time DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return pd.read_sql(query, conn, params=params)

    def get_account_stats(self, starting_balance: float = 10000.0) -> dict:
        """Dashboard summary — Day 17 doc-এর 'AI PAPER ACCOUNT' output

        2026-07-20 fix: trades closed via core/orphan_cleanup.py (when a DB
        'OPEN' trade has no matching live MT5/paper position) get
        result='AUTO_CLOSED' with pnl left NULL — the real fill/close price
        was never available, so no P&L could be computed. These were
        previously counted in `total` (status='CLOSED') without being a WIN
        or a LOSS, which silently understated win_rate (extra trade in the
        denominator that never counts toward wins), while SUM(pnl) quietly
        ignores their NULL pnl — meaning if that position actually gained or
        lost real money at the broker, it's missing from total_pnl/balance
        with no indication anything is off. Now excluded from win_rate's
        denominator and reported separately as `unresolved_trades` so the
        gap is visible instead of silently baked into the numbers.
        """
        with self._connect() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) as losses,
                       SUM(CASE WHEN result='AUTO_CLOSED' THEN 1 ELSE 0 END) as unresolved,
                       SUM(pnl) as total_pnl
                FROM trades WHERE status = 'CLOSED'
            """).fetchone()
        total, wins, losses, unresolved, total_pnl = row
        total      = total or 0
        wins       = wins or 0
        losses     = losses or 0
        unresolved = unresolved or 0
        total_pnl  = total_pnl or 0.0
        resolved   = total - unresolved
        win_rate   = round(wins / resolved * 100, 1) if resolved else 0.0
        return {
            "balance":           round(starting_balance + total_pnl, 2),
            "total_trades":      total,
            "wins":              wins,
            "losses":            losses,
            "unresolved_trades": unresolved,
            "win_rate":          win_rate,
            "total_pnl":         round(total_pnl, 2),
        }

    def get_overall_stats(self, starting_balance: float = 10000.0) -> dict:
        """Telegram status/reporting-এর জন্য all-time paper account stats।"""
        base_stats = self.get_account_stats(starting_balance=starting_balance)
        with self._connect() as conn:
            open_trades = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE status = 'OPEN'"
            ).fetchone()[0]
        return {
            "total": base_stats["total_trades"],
            "wins": base_stats["wins"],
            "losses": base_stats["losses"],
            "win_rate": base_stats["win_rate"],
            "total_pnl": base_stats["total_pnl"],
            "balance": base_stats["balance"],
            "open_trades": open_trades or 0,
        }

    # ─────────────────────────────────────────────
    # ECONOMIC HISTORY  (Day 43 — News Memory System)
    # ─────────────────────────────────────────────

    def save_economic_event(self, event: dict) -> int:
        """
        একটা economic event + (জানা থাকলে) তার actual market reaction
        save করো। `event` dict-এর সম্ভাব্য keys:
            event, currency, impact, event_time, expected, actual,
            market_reaction ("BULLISH"/"BEARISH"/"NEUTRAL"), pips_moved, lesson
        """
        with self._connect() as conn:
            cur = conn.execute("""
                INSERT INTO economic_history
                (event, currency, impact, event_time, expected, actual,
                 market_reaction, pips_moved, lesson, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.get("event", ""),
                event.get("currency", "").upper(),
                event.get("impact", "HIGH"),
                event.get("event_time", datetime.utcnow().isoformat()),
                event.get("expected"),
                event.get("actual"),
                event.get("market_reaction"),
                event.get("pips_moved"),
                event.get("lesson"),
                datetime.utcnow().isoformat(),
            ))
            event_id = cur.lastrowid
        log.info(
            f"Economic event saved: #{event_id} {event.get('currency')} "
            f"{event.get('event')} → {event.get('market_reaction', 'unknown')}"
        )
        return event_id

    def get_economic_history(self, currency: str = None, limit: int = 50) -> pd.DataFrame:
        """Recent economic events (lesson/reaction সহ) দেখো — currency filter optional।"""
        query  = "SELECT * FROM economic_history"
        params = []
        if currency:
            query += " WHERE currency = ?"
            params.append(currency.upper())
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return pd.read_sql(query, conn, params=params)

    def get_currency_fundamental_bias(self, currency: str, lookback: int = 10) -> dict:
        """
        একটা currency-র সাম্প্রতিক economic_history entries দেখে
        bullish/bearish reaction count থেকে একটা সরল fundamental bias বের করে।
        FundamentalSentimentScore module এটাকেই raw input হিসেবে ব্যবহার করবে।
        """
        history = self.get_economic_history(currency=currency, limit=lookback)
        if history.empty:
            return {
                "currency": currency.upper(), "sample_size": 0,
                "bullish_count": 0, "bearish_count": 0, "neutral_count": 0,
                "raw_score": 0,
            }

        reactions = history["market_reaction"].fillna("NEUTRAL").str.upper()
        bullish   = int((reactions == "BULLISH").sum())
        bearish   = int((reactions == "BEARISH").sum())
        neutral   = int((reactions == "NEUTRAL").sum())
        raw_score = bullish - bearish   # সরল net score

        return {
            "currency":      currency.upper(),
            "sample_size":   len(history),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "raw_score":     raw_score,
        }

    # ─────────────────────────────────────────────
    # QUERY METHODS
    # ─────────────────────────────────────────────

    def get_latest_analysis(self, symbol: str, limit: int = 5) -> pd.DataFrame:
        """সর্বশেষ N analysis result দেখো"""
        with self._connect() as conn:
            return pd.read_sql("""
                SELECT run_time, timeframe, bias_score, bias_label
                FROM analysis
                WHERE symbol = ?
                ORDER BY run_time DESC
                LIMIT ?
            """, conn, params=(symbol, limit))

    def get_pattern_history(self, symbol: str, limit: int = 20) -> pd.DataFrame:
        """Recent patterns দেখো"""
        with self._connect() as conn:
            return pd.read_sql("""
                SELECT time, pattern, engulfing, star
                FROM patterns
                WHERE symbol = ?
                  AND (pattern != 'none' OR engulfing != 'none')
                ORDER BY time DESC
                LIMIT ?
            """, conn, params=(symbol, limit))

    def stats(self):
        """Database stats দেখো"""
        with self._connect() as conn:
            for table in ['candles', 'indicators', 'patterns', 'analysis', 'trades', 'economic_history']:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                log.info(f"  {table:<18}: {count} rows")


def _safe(row, col):
    """NaN → None (SQLite-এর জন্য)"""
    import math
    val = row.get(col)
    if val is None:
        return None
    try:
        return None if math.isnan(float(val)) else float(val)
    except Exception as e:
        log.warning(f"Suppressed exception at line 519: {e}")
        return None