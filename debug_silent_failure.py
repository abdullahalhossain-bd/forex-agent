"""
debug_silent_failure.py — Diagnose silent trade failures.

Tests 5 hypotheses the user identified:
  H1: Agent deadlock / silent failure (threads stuck, no progress)
  H2: MT5 connection but no data flowing (test vs reality gap)
  H3: Signal generated but silently rejected by risk
  H4: Timezone mismatch (MT5 server time ≠ local time)
  H5: Database writes failing (learning loop broken)

Each test prints [PASS] / [FAIL] / [WARN] with diagnostic detail.
Exits 0 if all critical tests pass, 1 otherwise.

Run from project root:
    python debug_silent_failure.py

This script does NOT place any orders.  It only reads state and runs
diagnostic checks.
"""
from __future__ import annotations

import json
import os
import sys
import sqlite3
import time
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass


class Report:
    def __init__(self):
        self.results = []

    def pass_(self, name, detail=""):
        self.results.append(("PASS", name, detail))

    def fail(self, name, detail=""):
        self.results.append(("FAIL", name, detail))

    def warn(self, name, detail=""):
        self.results.append(("WARN", name, detail))

    def section(self, title):
        self.results.append(("-", f"\n── {title} " + "─" * max(0, 60 - len(title)), ""))

    def print_(self):
        print()
        print("=" * 70)
        print("  SILENT FAILURE DIAGNOSTIC — forex_ai")
        print("=" * 70)
        for status, name, detail in self.results:
            if status == "-":
                print(name)
                continue
            line = f"  [{status}] {name}"
            if detail:
                line += f" — {detail}"
            print(line)
        print()
        print("=" * 70)
        fails = sum(1 for s, _, _ in self.results if s == "FAIL")
        warns = sum(1 for s, _, _ in self.results if s == "WARN")
        passes = sum(1 for s, _, _ in self.results if s == "PASS")
        print(f"  Result: {passes} PASS, {warns} WARN, {fails} FAIL")
        print("=" * 70)


r = Report()


# ──────────────────────────────────────────────────────────────────
# H1: Agent deadlock / silent failure
# ──────────────────────────────────────────────────────────────────

r.section("H1: Agent Deadlock / Silent Failure")

# Check 1: are there any threads stuck?
main_thread = threading.main_thread()
active_threads = threading.enumerate()
r.pass_("threads enumerate", f"{len(active_threads)} active (main={main_thread.name})")
for t in active_threads:
    if t.name != "MainThread":
        r.pass_(f"thread {t.name}", f"alive={t.is_alive()} daemon={t.daemon}")

# Check 2: signal_debug.jsonl — how many cycles in last hour?
debug_path = ROOT / "memory" / "signal_debug.jsonl"
if debug_path.exists():
    try:
        lines = debug_path.read_text().strip().split("\n")
        recent_cycles = 0
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        for line in lines[-50:]:
            try:
                rec = json.loads(line)
                ts = rec.get("started_at", "")
                if ts:
                    rec_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if rec_time > one_hour_ago:
                        recent_cycles += 1
            except Exception:
                pass
        if recent_cycles == 0:
            r.warn("recent cycles", f"0 cycles in last hour ({len(lines)} total) — bot may be stalled")
        else:
            r.pass_("recent cycles", f"{recent_cycles} cycles in last hour")
    except Exception as e:
        r.fail("signal_debug.jsonl parse", str(e))
else:
    r.warn("signal_debug.jsonl", "not found — no cycle history to analyze")

# Check 3: any 'Trader exited unexpectedly' in trader.log?
trader_log = ROOT / "logs" / "trader.log"
if trader_log.exists():
    try:
        content = trader_log.read_text(errors="ignore")
        restart_count = content.count("Trader exited unexpectedly")
        crash_count = content.count("Symbol cycle failed")
        if restart_count > 0:
            r.fail("trader restarts", f"{restart_count} unexpected exits in log — process is crashing")
        else:
            r.pass_("trader restarts", "0 unexpected exits")
        if crash_count > 0:
            r.warn("cycle crashes", f"{crash_count} 'Symbol cycle failed' — per-symbol crashes (non-fatal)")
        else:
            r.pass_("cycle crashes", "0 symbol-cycle crashes")
    except Exception as e:
        r.fail("trader.log read", str(e))
else:
    r.warn("trader.log", "not found")


# ──────────────────────────────────────────────────────────────────
# H2: MT5 connection but no data flowing
# ──────────────────────────────────────────────────────────────────

r.section("H2: MT5 Connection vs Data Flow")

try:
    import MetaTrader5 as mt5
    r.pass_("MetaTrader5 package", "imported")
except ImportError:
    r.warn("MetaTrader5 package", "not installed — cannot test MT5 data flow on this machine")
    mt5 = None

if mt5 is not None:
    # Initialize
    try:
        from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH
        init_kwargs = {}
        if MT5_PATH: init_kwargs["path"] = MT5_PATH
        if MT5_LOGIN: init_kwargs["login"] = int(MT5_LOGIN)
        if MT5_PASSWORD: init_kwargs["password"] = MT5_PASSWORD
        if MT5_SERVER: init_kwargs["server"] = MT5_SERVER

        if not mt5.initialize(**init_kwargs):
            r.fail("mt5.initialize()", f"last_error={mt5.last_error()}")
        else:
            r.pass_("mt5.initialize()", "connected")

            # Test 1: account_info
            acc = mt5.account_info()
            if acc is None:
                r.fail("mt5.account_info()", f"last_error={mt5.last_error()}")
            else:
                r.pass_("account_info", f"balance=${acc.balance:.2f} trade_allowed={acc.trade_allowed}")

            # Test 2: terminal_info
            term = mt5.terminal_info()
            if term is None:
                r.fail("terminal_info()", f"last_error={mt5.last_error()}")
            else:
                r.pass_("terminal_info", f"connected={term.connected} trade_allowed={term.trade_allowed}")

            # Test 3: symbol_info_tick for each configured symbol
            from config import SYMBOLS
            for sym in SYMBOLS:
                # Try exact + suffix variants
                broker_sym = sym
                info = mt5.symbol_info(broker_sym)
                if info is None:
                    matches = mt5.symbols_get(f"*{sym}*")
                    if matches:
                        broker_sym = matches[0].name
                        info = mt5.symbol_info(broker_sym)

                if info is None:
                    r.fail(f"symbol {sym}", "not found at broker")
                    continue

                if not info.visible:
                    mt5.symbol_select(broker_sym, True)

                tick = mt5.symbol_info_tick(broker_sym)
                if tick is None or tick.time == 0:
                    r.fail(f"tick {sym}", f"no live tick — market closed? broker_sym={broker_sym}")
                else:
                    tick_age = time.time() - tick.time
                    if tick_age > 300:
                        r.warn(f"tick {sym}", f"stale tick ({tick_age:.0f}s old) — broker_sym={broker_sym}")
                    else:
                        r.pass_(f"tick {sym}", f"bid={tick.bid} ask={tick.ask} age={tick_age:.0f}s")

            # Test 4: copy_rates — actual candle data
            rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, 0, 5)
            if rates is None or len(rates) == 0:
                r.fail("copy_rates EURUSD M15", f"no candle data — last_error={mt5.last_error()}")
            else:
                last_candle = rates[-1]
                candle_age = time.time() - last_candle["time"]
                r.pass_("copy_rates EURUSD M15",
                        f"{len(rates)} candles, last close={last_candle['close']:.5f}, age={candle_age:.0f}s")
                if candle_age > 3600:
                    r.warn("candle freshness", f"last candle is {candle_age/3600:.1f}h old — market may be closed")

            mt5.shutdown()
    except Exception as e:
        r.fail("MT5 test", str(e))


# ──────────────────────────────────────────────────────────────────
# H3: Signal generated but silently rejected by risk
# ──────────────────────────────────────────────────────────────────

r.section("H3: Silent Risk Rejection")

# Test 1: signal_debug evidence
if debug_path.exists():
    try:
        lines = debug_path.read_text().strip().split("\n")
        no_trade_count = 0
        buy_sell_count = 0
        risk_reject_count = 0
        for line in lines:
            try:
                rec = json.loads(line)
                if rec.get("final_action") in ("BUY", "SELL"):
                    buy_sell_count += 1
                elif rec.get("final_action") == "NO TRADE":
                    no_trade_count += 1
                layers = rec.get("layers", [])
                for l in layers:
                    if l.get("layer") == "risk" and l.get("status") == "REJECT":
                        risk_reject_count += 1
            except Exception:
                pass
        r.pass_("signal_debug summary",
                f"BUY/SELL={buy_sell_count}, NO TRADE={no_trade_count}, risk REJECT={risk_reject_count} (of {len(lines)} cycles)")
        if no_trade_count > 0 and buy_sell_count == 0:
            r.fail("silent rejection pattern",
                   "ALL cycles ended in NO TRADE — signals are being silently rejected")
    except Exception as e:
        r.fail("signal_debug analysis", str(e))

# Test 2: actually run RiskEngine with a sample BUY signal
try:
    from risk.risk_engine import RiskEngine
    re = RiskEngine(balance=10000, symbol="EURUSD")
    result = re.evaluate(signal="BUY", entry=1.0850, atr=0.0010, regime={"volatility": "NORMAL"})
    if result["approved"]:
        r.pass_("RiskEngine sample BUY",
                f"approved=True lot={result['lot']} risk_usd=${result['risk_usd']} sl_pips={result['sl_pips']}")
        # Check lot against MAX_LOT cap
        from config import MAX_LOT
        if result["lot"] > MAX_LOT:
            r.fail("lot exceeds MAX_LOT", f"lot={result['lot']} > MAX_LOT={MAX_LOT} — would be capped")
        else:
            r.pass_("lot within cap", f"lot={result['lot']} <= MAX_LOT={MAX_LOT}")
    else:
        r.fail("RiskEngine sample BUY",
               f"approved=False reject_reason={result.get('reject_reason')}")
except Exception as e:
    r.fail("RiskEngine test", str(e))

# Test 3: TradePermission threshold
try:
    from risk.trade_permission import TradePermission
    tp = TradePermission()
    r.pass_("TradePermission.MIN_CONFIDENCE", f"={tp.MIN_CONFIDENCE} (TEST_MODE check)")
    sample_dec = {"decision": "BUY", "confidence": 50}
    sample_risk = {"approved": True}
    perm = tp.check(sample_dec, sample_risk, {"news_trade_allowed": True}, None)
    if perm["allowed"]:
        r.pass_("TradePermission sample BUY 50%", "allowed=True")
    else:
        failed = [c["check"] for c in perm.get("checks", []) if not c.get("passed")]
        r.fail("TradePermission sample BUY 50%", f"BLOCKED — failed: {failed}")
except Exception as e:
    r.fail("TradePermission test", str(e))

# Test 4: CircuitBreaker state
try:
    from risk.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker()
    status = cb.allow_trade()
    if status["allowed"]:
        r.pass_("CircuitBreaker", f"mode={status['mode']} reason={status['reason']}")
    else:
        r.fail("CircuitBreaker", f"BLOCKED mode={status['mode']} reason={status['reason']}")
except Exception as e:
    r.fail("CircuitBreaker test", str(e))


# ──────────────────────────────────────────────────────────────────
# H4: Timezone mismatch
# ──────────────────────────────────────────────────────────────────

r.section("H4: Timezone Mismatch")

local_now = datetime.now()
utc_now = datetime.now(timezone.utc)
local_offset = local_now - utc_now.replace(tzinfo=None)
r.pass_("local time", f"{local_now.isoformat()} (offset={local_offset.total_seconds()/3600:+.1f}h)")
r.pass_("UTC time", utc_now.isoformat())

# Check daily_risk.json date vs local date
dr_path = ROOT / "memory" / "daily_risk.json"
if dr_path.exists():
    try:
        dr = json.loads(dr_path.read_text())
        dr_date = dr.get("date")
        today_utc = datetime.utcnow().strftime("%Y-%m-%d")
        today_local = local_now.strftime("%Y-%m-%d")
        if dr_date == today_utc:
            r.pass_("daily_risk.json date", f"date={dr_date} (matches UTC today)")
        elif dr_date == today_local:
            r.pass_("daily_risk.json date", f"date={dr_date} (matches local today)")
        else:
            r.warn("daily_risk.json date",
                   f"date={dr_date} — neither UTC today ({today_utc}) nor local today ({today_local}). "
                   "Timezone mismatch could cause daily loss limit to never reset.")
    except Exception as e:
        r.fail("daily_risk.json parse", str(e))

# Check circuit_breaker_state.json date
cb_path = ROOT / "memory" / "circuit_breaker_state.json"
if cb_path.exists():
    try:
        cb_state = json.loads(cb_path.read_text())
        cb_date = cb_state.get("date")
        if cb_date:
            today_utc = datetime.utcnow().strftime("%Y-%m-%d")
            if cb_date == today_utc:
                r.pass_("CB state date", f"date={cb_date} (matches today)")
            else:
                r.warn("CB state date",
                       f"date={cb_date} — STALE (today={today_utc}). CB._load_state() will reset "
                       "daily_loss_usd on next read, but mode/consecutive_losses persist across days.")
    except Exception as e:
        r.fail("CB state parse", str(e))

# Check session_analyzer timezone assumption
try:
    from analysis.session_analyzer import SessionAnalyzer
    sa = SessionAnalyzer()
    if hasattr(sa, "analyze"):
        session_result = sa.analyze(pair="EURUSD", smc_ctx={}, signal="NO TRADE", signal_conf=0)
        if hasattr(sa, "get_ai_context"):
            sess_ctx = sa.get_ai_context(session_result)
            gmt_time = sess_ctx.get("gmt_time", "?")
            current_session = sess_ctx.get("current_session", "?")
            r.pass_("session analyzer",
                    f"gmt_time={gmt_time} session={current_session}")
            # If local time is e.g. 3am but session says "LONDON", timezone mismatch
            local_hour = local_now.hour
            if "LONDON" in str(current_session).upper() and not (7 <= local_hour <= 16):
                r.warn("session vs local time",
                       f"local hour={local_hour} but session=LONDON (usually 7-16 GMT) — possible TZ mismatch")
        else:
            r.warn("session analyzer", "no get_ai_context method — API mismatch")
    else:
        r.warn("session analyzer", "no analyze method — API mismatch (uses detect_london_manipulation?)")
except Exception as e:
    r.warn("session analyzer test", str(e))


# ──────────────────────────────────────────────────────────────────
# H5: Database writes failing
# ──────────────────────────────────────────────────────────────────

r.section("H5: Database Writes")

db_path = ROOT / "database" / "trader.db"
if not db_path.exists():
    r.warn("trader.db", "not found — DB not initialized yet")
else:
    r.pass_("trader.db exists", f"size={db_path.stat().st_size} bytes")

    # Test 1: can we open the DB?
    try:
        conn = sqlite3.connect(str(db_path), timeout=2.0)
        cur = conn.cursor()

        # Test 2: list tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        r.pass_("DB tables", f"{len(tables)} tables: {tables[:5]}{'...' if len(tables) > 5 else ''}")

        # Test 3: count rows in trades table
        if "trades" in tables:
            cur.execute("SELECT COUNT(*) FROM trades")
            trade_count = cur.fetchone()[0]
            r.pass_("trades table", f"{trade_count} rows")

            # Sample most recent 3 trades
            cur.execute("SELECT pair, type, status, open_time FROM trades ORDER BY rowid DESC LIMIT 3")
            for row in cur.fetchall():
                r.pass_("  recent trade", f"pair={row[0]} type={row[1]} status={row[2]} ts={row[3]}")
        else:
            r.fail("trades table", "missing — DB schema not initialized")

        # Test 4: can we write? (insert + rollback)
        if "trades" in tables:
            try:
                # Get actual column names from PRAGMA
                cur.execute("PRAGMA table_info(trades)")
                cols = [row[1] for row in cur.fetchall()]
                r.pass_("trades schema", f"columns: {', '.join(cols[:8])}{'...' if len(cols) > 8 else ''}")

                # Use only columns that exist
                cur.execute("BEGIN")
                # Schema uses 'type' not 'signal', 'open_time' not 'created_at'
                insert_cols = ["pair", "type", "status", "lot", "entry", "open_time"]
                insert_vals = ("TEST", "BUY", "TEST_WRITE", 0.01, 1.0, datetime.now(timezone.utc).isoformat())
                placeholders = ", ".join(["?"] * len(insert_cols))
                col_str = ", ".join(insert_cols)
                cur.execute(
                    f"INSERT INTO trades ({col_str}) VALUES ({placeholders})",
                    insert_vals
                )
                cur.execute("ROLLBACK")
                r.pass_("DB write test", "insert succeeded (rolled back)")
            except Exception as e:
                r.fail("DB write test", f"write failed: {e} — DB may be locked or schema drifted")
                try:
                    cur.execute("ROLLBACK")
                except:
                    pass

        # Test 5: check for learning tables
        learning_tables = [t for t in tables if "learn" in t.lower() or "memory" in t.lower() or "decision" in t.lower()]
        if learning_tables:
            for lt in learning_tables:
                cur.execute(f"SELECT COUNT(*) FROM {lt}")
                cnt = cur.fetchone()[0]
                r.pass_(f"  {lt}", f"{cnt} rows")
        else:
            r.warn("learning tables", "no learning/memory/decision tables found — learning loop may be broken")

        conn.close()
    except sqlite3.OperationalError as e:
        r.fail("DB operational", str(e))
    except Exception as e:
        r.fail("DB test", str(e))

# Check decision_history.jsonl for learning loop activity
dh_path = ROOT / "memory" / "decision_history.jsonl"
if dh_path.exists():
    try:
        lines = dh_path.read_text().strip().split("\n")
        r.pass_("decision_history.jsonl", f"{len(lines)} decisions logged")
        if len(lines) > 0:
            try:
                last = json.loads(lines[-1])
                # Field name is 'direction' (ConfluenceDecision), not 'decision'
                direction = last.get("direction") or last.get("decision") or last.get("signal") or "?"
                ts = last.get("ts") or last.get("timestamp") or "?"
                r.pass_("last decision",
                        f"pair={last.get('pair')} direction={direction} conf={last.get('confidence')}% ts={str(ts)[:19]}")
            except Exception as e:
                r.warn("last decision parse", str(e))
    except:
        pass
else:
    r.warn("decision_history.jsonl", "not found — learning loop not recording decisions")


# ──────────────────────────────────────────────────────────────────
# Print final report
# ──────────────────────────────────────────────────────────────────

r.print_()

# Exit code based on failures
fails = sum(1 for s, _, _ in r.results if s == "FAIL")
sys.exit(0 if fails == 0 else 1)
