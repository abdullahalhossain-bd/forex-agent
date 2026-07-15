# analytics/strategy_tracker.py  —  Day 54 | Strategy Performance Tracker
# ============================================================
# প্রতিটি trade-এর পূর্ণ environment data collect করে।
#
# Track করে:
#   - Pair, Timeframe, Session, Day of Week
#   - Pattern, Market Regime
#   - Win/Loss, Profit (pips), R:R
#
# Walk-Forward:  last 7 days / last 30 days আলাদা করে দেখা যায়।
# Strategy Version Control: কোন version-এ trade হয়েছিল সেটা save হয়।
# Auto-Disable: win rate < 35% এবং negative expectancy হলে setup disable।
# ============================================================

import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional
from utils.logger import get_logger

log = get_logger("strategy_tracker")

DB_PATH = "memory/strategy_tracker.db"
STRATEGY_VERSION = "v1.0"   # Day 54 শুরু


# ════════════════════════════════════════════════════════════
# SESSION DETECTOR
# ════════════════════════════════════════════════════════════

def detect_session(utc_hour: int) -> str:
    """
    UTC hour থেকে trading session বের করো।
    Asian:     00-08 UTC
    London:    08-16 UTC
    New York:  13-21 UTC (overlap 13-16 London+NY)
    """
    if 8 <= utc_hour < 13:
        return "London"
    elif 13 <= utc_hour < 16:
        return "London_NY_Overlap"
    elif 16 <= utc_hour < 21:
        return "New_York"
    else:
        return "Asian"


# ════════════════════════════════════════════════════════════
# STRATEGY TRACKER
# ════════════════════════════════════════════════════════════

class StrategyTracker:
    """
    Day 54 Performance Intelligence Layer।

    সব trade-এর environment data SQLite-এ সংরক্ষণ করে।
    DecisionAgent-এর output + outcome পরে আলাদাভাবে update করা যায়।

    Usage:
        tracker = StrategyTracker()

        # Trade decision নেওয়ার সময়:
        trade_id = tracker.record_trade(decision_out, analysis_out, market_out)

        # Trade close হলে outcome update:
        tracker.update_outcome(trade_id, result="WIN", profit_pips=45, rr_actual=2.1)
    """

    def __init__(self, db_path: str = DB_PATH, strategy_version: str = STRATEGY_VERSION):
        self.db_path = db_path
        self.strategy_version = strategy_version
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    # ─────────────────────────────────────────────
    # DATABASE SETUP
    # ─────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_uuid      TEXT UNIQUE,
                    timestamp       TEXT,
                    strategy_ver    TEXT,

                    -- Environment
                    pair            TEXT,
                    timeframe       TEXT,
                    session         TEXT,
                    day_of_week     TEXT,
                    utc_hour        INTEGER,

                    -- Setup
                    pattern         TEXT,
                    regime          TEXT,
                    signal          TEXT,
                    confidence      REAL,
                    entry           REAL,
                    sl              REAL,
                    tp              REAL,
                    lot             REAL,
                    planned_rr      REAL,

                    -- Outcome (পরে update হবে)
                    result          TEXT,    -- WIN / LOSS / BE / NULL
                    profit_pips     REAL,
                    rr_actual       REAL,
                    closed_at       TEXT,

                    -- Meta
                    is_disabled     INTEGER DEFAULT 0,
                    notes           TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS disabled_setups (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    setup_key   TEXT UNIQUE,   -- e.g. "EURUSD|H1|London|TRENDING|Hammer"
                    reason      TEXT,
                    disabled_at TEXT,
                    re_check_at TEXT           -- কখন আবার check করবে
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pair ON trades(pair);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session ON trades(session);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pattern ON trades(pattern);
            """)
        log.info(f"[StrategyTracker] DB ready: {self.db_path}")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ─────────────────────────────────────────────
    # RECORD TRADE  (Decision-time)
    # ─────────────────────────────────────────────

    def record_trade(
        self,
        decision_out: dict,
        analysis_out: dict,
        market_out:   dict,
    ) -> Optional[str]:
        """
        Trade decision নেওয়ার মুহূর্তে environment সহ DB-তে save করো।
        Returns: trade_uuid (outcome update-এর জন্য রাখো)
        """
        import uuid

        signal = decision_out.get("decision", "NO TRADE")
        if signal not in ("BUY", "SELL"):
            return None   # NO TRADE / WAIT track করার দরকার নেই

        now      = datetime.now(timezone.utc)
        trade_id = str(uuid.uuid4())[:8]

        pair      = market_out.get("symbol", "UNKNOWN")
        timeframe = market_out.get("timeframe", "UNKNOWN")
        session   = detect_session(now.hour)
        dow       = now.strftime("%A")   # Monday, Tuesday…

        pattern = decision_out.get("pattern") or \
                  analysis_out.get("advanced_pat_ctx", {}).get("top_pattern") or \
                  analysis_out.get("pat_ctx", {}).get("latest_pattern", "Unknown")

        regime    = decision_out.get("regime") or \
                    market_out.get("regime", {}).get("regime", "UNKNOWN")

        confidence = decision_out.get("confidence", 0)
        entry      = decision_out.get("entry")
        sl         = decision_out.get("sl")
        tp         = decision_out.get("tp")
        lot        = decision_out.get("lot", 0)
        planned_rr = decision_out.get("rr", 0)

        row = (
            trade_id, now.isoformat(), self.strategy_version,
            pair, timeframe, session, dow, now.hour,
            pattern, regime, signal, confidence,
            entry, sl, tp, lot, planned_rr,
            None, None, None, None,   # outcome fields — পরে update
            0, None                   # is_disabled, notes
        )

        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO trades (
                        trade_uuid, timestamp, strategy_ver,
                        pair, timeframe, session, day_of_week, utc_hour,
                        pattern, regime, signal, confidence,
                        entry, sl, tp, lot, planned_rr,
                        result, profit_pips, rr_actual, closed_at,
                        is_disabled, notes
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, row)
            log.info(
                f"[StrategyTracker] #{trade_id} recorded — "
                f"{pair} {timeframe} {session} {dow} | "
                f"{signal} | Pattern: {pattern} | Regime: {regime} | "
                f"Conf: {confidence}%"
            )
            return trade_id
        except Exception as e:
            log.error(f"[StrategyTracker] DB write error: {e}")
            return None

    # ─────────────────────────────────────────────
    # UPDATE OUTCOME  (Trade close-time)
    # ─────────────────────────────────────────────

    def update_outcome(
        self,
        trade_uuid:  str,
        result:      str,        # "WIN" / "LOSS" / "BE"
        profit_pips: float = 0,
        rr_actual:   float = 0,
    ) -> bool:
        """
        Trade বন্ধ হওয়ার পর outcome update করো।
        ConfidenceEngine.record_outcome() এর পর call করো।
        """
        closed_at = datetime.now(timezone.utc).isoformat()
        try:
            with self._conn() as conn:
                conn.execute("""
                    UPDATE trades
                    SET result=?, profit_pips=?, rr_actual=?, closed_at=?
                    WHERE trade_uuid=?
                """, (result.upper(), profit_pips, rr_actual, closed_at, trade_uuid))
            log.info(
                f"[StrategyTracker] #{trade_uuid} outcome → "
                f"{result} | {profit_pips:+.1f} pips | RR: {rr_actual}"
            )
            # Auto-disable check চালাও
            row = self._get_trade(trade_uuid)
            if row:
                self._check_auto_disable(
                    pair=row["pair"], timeframe=row["timeframe"],
                    session=row["session"], regime=row["regime"],
                    pattern=row["pattern"],
                )
            return True
        except Exception as e:
            log.error(f"[StrategyTracker] outcome update error: {e}")
            return False

    def _get_trade(self, trade_uuid: str) -> Optional[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM trades WHERE trade_uuid=?", (trade_uuid,)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # ─────────────────────────────────────────────
    # PAIR PERFORMANCE
    # ─────────────────────────────────────────────

    def pair_performance(self, days: int = 90) -> dict:
        """কোন pair সবচেয়ে ভালো / খারাপ করছে।"""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT pair,
                       COUNT(*) as trades,
                       SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                       SUM(profit_pips) as total_pips
                FROM trades
                WHERE result IS NOT NULL AND timestamp >= ?
                GROUP BY pair
                ORDER BY wins * 1.0 / COUNT(*) DESC
            """, (since,)).fetchall()

        result = {}
        for r in rows:
            trades = r["trades"]
            wins   = r["wins"] or 0
            result[r["pair"]] = {
                "trades":      trades,
                "wins":        wins,
                "win_rate":    round(wins / trades * 100, 1) if trades else 0,
                "total_pips":  round(r["total_pips"] or 0, 1),
            }
        return result

    # ─────────────────────────────────────────────
    # SESSION PERFORMANCE
    # ─────────────────────────────────────────────

    def session_performance(self, pair: str = None, days: int = 90) -> dict:
        """কোন session-এ সবচেয়ে ভালো কাজ করে।"""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        query = """
            SELECT session,
                   COUNT(*) as trades,
                   SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                   SUM(profit_pips) as total_pips
            FROM trades
            WHERE result IS NOT NULL AND timestamp >= ?
        """
        params = [since]
        if pair:
            query += " AND pair=?"
            params.append(pair)
        query += " GROUP BY session ORDER BY wins * 1.0 / COUNT(*) DESC"

        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        result = {}
        for r in rows:
            trades = r["trades"]
            wins   = r["wins"] or 0
            result[r["session"]] = {
                "trades":     trades,
                "wins":       wins,
                "win_rate":   round(wins / trades * 100, 1) if trades else 0,
                "total_pips": round(r["total_pips"] or 0, 1),
            }
        return result

    # ─────────────────────────────────────────────
    # DAY OF WEEK PERFORMANCE
    # ─────────────────────────────────────────────

    def day_of_week_performance(self, days: int = 90) -> dict:
        """কোন weekday-এ সবচেয়ে ভালো।"""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT day_of_week,
                       COUNT(*) as trades,
                       SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                       SUM(profit_pips) as total_pips
                FROM trades
                WHERE result IS NOT NULL AND timestamp >= ?
                GROUP BY day_of_week
                ORDER BY wins * 1.0 / COUNT(*) DESC
            """, (since,)).fetchall()

        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        result = {}
        for r in rows:
            trades = r["trades"]
            wins   = r["wins"] or 0
            result[r["day_of_week"]] = {
                "trades":     trades,
                "wins":       wins,
                "win_rate":   round(wins / trades * 100, 1) if trades else 0,
                "total_pips": round(r["total_pips"] or 0, 1),
            }
        # weekday order অনুযায়ী sort করো
        return {k: result[k] for k in order if k in result}

    # ─────────────────────────────────────────────
    # PATTERN PERFORMANCE MATRIX
    # ─────────────────────────────────────────────

    def pattern_performance_matrix(self, min_trades: int = 5) -> dict:
        """
        Pattern × Pair × Regime — তিন-মাত্রার performance table।
        ConfidenceEngine-এর historical data-র complement।
        """
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT pattern, pair, regime,
                       COUNT(*) as trades,
                       SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                       AVG(rr_actual) as avg_rr
                FROM trades
                WHERE result IS NOT NULL
                GROUP BY pattern, pair, regime
                HAVING COUNT(*) >= ?
                ORDER BY wins * 1.0 / COUNT(*) DESC
            """, (min_trades,)).fetchall()

        matrix = {}
        for r in rows:
            key    = f"{r['pattern']}|{r['pair']}|{r['regime']}"
            trades = r["trades"]
            wins   = r["wins"] or 0
            matrix[key] = {
                "pattern":  r["pattern"],
                "pair":     r["pair"],
                "regime":   r["regime"],
                "trades":   trades,
                "wins":     wins,
                "win_rate": round(wins / trades * 100, 1) if trades else 0,
                "avg_rr":   round(r["avg_rr"] or 0, 2),
            }
        return matrix

    # ─────────────────────────────────────────────
    # REGIME PERFORMANCE
    # ─────────────────────────────────────────────

    def regime_performance(self, days: int = 90) -> dict:
        """TRENDING vs RANGING vs HIGH_VOLATILITY performance।"""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT regime,
                       COUNT(*) as trades,
                       SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                       SUM(profit_pips) as total_pips,
                       AVG(rr_actual) as avg_rr
                FROM trades
                WHERE result IS NOT NULL AND timestamp >= ?
                GROUP BY regime
                ORDER BY wins * 1.0 / COUNT(*) DESC
            """, (since,)).fetchall()

        result = {}
        for r in rows:
            trades = r["trades"]
            wins   = r["wins"] or 0
            result[r["regime"]] = {
                "trades":     trades,
                "wins":       wins,
                "win_rate":   round(wins / trades * 100, 1) if trades else 0,
                "total_pips": round(r["total_pips"] or 0, 1),
                "avg_rr":     round(r["avg_rr"] or 0, 2),
            }
        return result

    # ─────────────────────────────────────────────
    # WALK-FORWARD  ⭐
    # ─────────────────────────────────────────────

    def walk_forward_stats(self) -> dict:
        """
        Lifetime / Last 30d / Last 7d আলাদাভাবে দেখো।
        বর্তমান market condition বদলেছে কিনা বোঝা যায়।
        """
        def _stats(since_iso: Optional[str]) -> dict:
            query = "SELECT result, profit_pips FROM trades WHERE result IS NOT NULL"
            params = []
            if since_iso:
                query += " AND timestamp >= ?"
                params.append(since_iso)

            with self._conn() as conn:
                rows = conn.execute(query, params).fetchall()

            if not rows:
                return {"trades": 0, "win_rate": 0, "total_pips": 0}

            total  = len(rows)
            wins   = sum(1 for r in rows if r[0] == "WIN")
            pips   = sum((r[1] or 0) for r in rows)
            return {
                "trades":     total,
                "wins":       wins,
                "win_rate":   round(wins / total * 100, 1),
                "total_pips": round(pips, 1),
            }

        now = datetime.now(timezone.utc)
        return {
            "lifetime":    _stats(None),
            "last_30_days": _stats((now - timedelta(days=30)).isoformat()),
            "last_7_days":  _stats((now - timedelta(days=7)).isoformat()),
        }

    # ─────────────────────────────────────────────
    # AUTO-DISABLE  ⭐
    # ─────────────────────────────────────────────

    def _check_auto_disable(
        self,
        pair: str, timeframe: str, session: str,
        regime: str, pattern: str,
        lookback: int = 50,
        min_trades: int = 10,
        disable_threshold: float = 35.0,
    ) -> None:
        """
        Last `lookback` trades-এ win rate < 35% এবং negative expectancy হলে
        setup টা temporarily disable করো।
        """
        setup_key = f"{pair}|{timeframe}|{session}|{regime}|{pattern}"

        with self._conn() as conn:
            rows = conn.execute("""
                SELECT result, profit_pips FROM trades
                WHERE pair=? AND timeframe=? AND session=? AND regime=? AND pattern=?
                  AND result IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT ?
            """, (pair, timeframe, session, regime, pattern, lookback)).fetchall()

        if len(rows) < min_trades:
            return   # পর্যাপ্ত data নেই, সিদ্ধান্ত নেওয়া যাবে না

        wins       = sum(1 for r in rows if r[0] == "WIN")
        win_rate   = wins / len(rows) * 100
        total_pips = sum((r[1] or 0) for r in rows)
        expectancy = total_pips / len(rows)

        if win_rate < disable_threshold and expectancy < 0:
            re_check = (datetime.now(timezone.utc) + timedelta(days=14)).isoformat()
            with self._conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO disabled_setups
                    (setup_key, reason, disabled_at, re_check_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    setup_key,
                    f"Win rate {win_rate:.1f}% < 35% AND expectancy {expectancy:.1f} pips (negative)",
                    datetime.now(timezone.utc).isoformat(),
                    re_check,
                ))
            log.warning(
                f"[StrategyTracker] ⛔ AUTO-DISABLED: {setup_key} | "
                f"Win rate: {win_rate:.1f}% | Expectancy: {expectancy:.1f} pips"
            )

    def is_setup_disabled(
        self,
        pair: str, timeframe: str, session: str,
        regime: str, pattern: str,
    ) -> tuple[bool, str]:
        """
        DecisionAgent call করবে trade নেওয়ার আগে।
        Returns: (is_disabled, reason)
        """
        setup_key = f"{pair}|{timeframe}|{session}|{regime}|{pattern}"
        now = datetime.now(timezone.utc).isoformat()

        with self._conn() as conn:
            row = conn.execute("""
                SELECT reason, re_check_at FROM disabled_setups
                WHERE setup_key=? AND re_check_at > ?
            """, (setup_key, now)).fetchone()

        if row:
            return True, row[0]
        return False, ""

    def get_disabled_setups(self) -> list:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM disabled_setups
                WHERE re_check_at > ?
                ORDER BY disabled_at DESC
            """, (datetime.now(timezone.utc).isoformat(),)).fetchall()
        return [dict(r) for r in rows]

    # ─────────────────────────────────────────────
    # BEST / WORST SETUP SUMMARY
    # ─────────────────────────────────────────────

    def best_worst_setups(self, min_trades: int = 5) -> dict:
        """Rank করা best ও worst setup।"""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT pair, timeframe, session, pattern, regime,
                       COUNT(*) as trades,
                       SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                       AVG(rr_actual) as avg_rr,
                       SUM(profit_pips) as total_pips
                FROM trades
                WHERE result IS NOT NULL
                GROUP BY pair, timeframe, session, pattern, regime
                HAVING COUNT(*) >= ?
                ORDER BY wins * 1.0 / COUNT(*) DESC
            """, (min_trades,)).fetchall()

        setups = []
        for r in rows:
            trades = r["trades"]
            wins   = r["wins"] or 0
            setups.append({
                "pair":       r["pair"],
                "timeframe":  r["timeframe"],
                "session":    r["session"],
                "pattern":    r["pattern"],
                "regime":     r["regime"],
                "trades":     trades,
                "win_rate":   round(wins / trades * 100, 1) if trades else 0,
                "avg_rr":     round(r["avg_rr"] or 0, 2),
                "total_pips": round(r["total_pips"] or 0, 1),
            })

        best  = setups[:3]   if setups else []
        worst = setups[-3:][::-1] if len(setups) >= 3 else setups[::-1]
        return {"best": best, "worst": worst}

    # ─────────────────────────────────────────────
    # PREFERRED TIMEFRAME PER PAIR
    # ─────────────────────────────────────────────

    def preferred_timeframe(self, min_trades: int = 5) -> dict:
        """প্রতিটি pair-এর জন্য কোন timeframe সবচেয়ে ভালো।"""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT pair, timeframe,
                       COUNT(*) as trades,
                       SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins
                FROM trades
                WHERE result IS NOT NULL
                GROUP BY pair, timeframe
                HAVING COUNT(*) >= ?
                ORDER BY pair, wins * 1.0 / COUNT(*) DESC
            """, (min_trades,)).fetchall()

        result = {}
        for r in rows:
            pair = r["pair"]
            if pair not in result:  # প্রতিটি pair-এ সেরাটা রাখো (already sorted DESC)
                trades = r["trades"]
                wins   = r["wins"] or 0
                result[pair] = {
                    "timeframe": r["timeframe"],
                    "win_rate":  round(wins / trades * 100, 1) if trades else 0,
                    "trades":    trades,
                }
        return result

    # ─────────────────────────────────────────────
    # PRINT SUMMARY
    # ─────────────────────────────────────────────

    def print_summary(self) -> None:
        bar = "═" * 60
        wf  = self.walk_forward_stats()
        bw  = self.best_worst_setups()
        pr  = self.preferred_timeframe()
        dis = self.get_disabled_setups()

        print(f"\n{bar}")
        print("  📊  STRATEGY TRACKER SUMMARY  (Day 54)")
        print(bar)

        # Walk-forward
        print("\n  ── Walk-Forward Performance ──")
        for period, stats in wf.items():
            print(
                f"  {period:15s}: {stats['trades']:4d} trades | "
                f"WR: {stats['win_rate']:5.1f}% | "
                f"Pips: {stats['total_pips']:+.1f}"
            )

        # Pair performance
        pair_perf = self.pair_performance()
        if pair_perf:
            print("\n  ── Pair Performance ──")
            for pair, s in pair_perf.items():
                print(
                    f"  {pair:10s}: {s['trades']:4d} trades | "
                    f"WR: {s['win_rate']:5.1f}% | "
                    f"Pips: {s['total_pips']:+.1f}"
                )

        # Session performance
        sess_perf = self.session_performance()
        if sess_perf:
            print("\n  ── Session Performance ──")
            for sess, s in sess_perf.items():
                print(
                    f"  {sess:22s}: WR: {s['win_rate']:5.1f}% | "
                    f"Trades: {s['trades']}"
                )

        # Preferred timeframe
        if pr:
            print("\n  ── Preferred Timeframe per Pair ──")
            for pair, info in pr.items():
                print(
                    f"  {pair:10s} → {info['timeframe']:5s} "
                    f"(WR: {info['win_rate']:.1f}%, {info['trades']} trades)"
                )

        # Best setups
        if bw["best"]:
            print("\n  ── Best Setups ──")
            for s in bw["best"]:
                print(
                    f"  ✅ {s['pair']} {s['timeframe']} {s['session']} | "
                    f"{s['pattern']} | WR: {s['win_rate']:.1f}% | "
                    f"RR: {s['avg_rr']} | {s['trades']} trades"
                )

        # Worst setups
        if bw["worst"]:
            print("\n  ── Worst Setups ──")
            for s in bw["worst"]:
                print(
                    f"  ⚠  {s['pair']} {s['timeframe']} {s['session']} | "
                    f"{s['pattern']} | WR: {s['win_rate']:.1f}% | "
                    f"{s['trades']} trades"
                )

        # Disabled setups
        if dis:
            print("\n  ── Auto-Disabled Setups ──")
            for d in dis:
                print(f"  ⛔ {d['setup_key']} | {d['reason'][:60]}")

        print(f"\n{bar}\n")