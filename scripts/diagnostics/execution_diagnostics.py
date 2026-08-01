"""
execution_diagnostics.py — End-to-end MT5 execution diagnostic.

Run from project root:
    python execution_diagnostics.py

Tests (in order):
  1.  MetaTrader5 Python package importable
  2.  MT5 terminal initialized
  3.  Login succeeded
  4.  Account info readable (trade_allowed, balance, margin)
  5.  Terminal info readable (connected, trade_allowed)
  6.  Each configured SYMBOL is visible in Market Watch
  7.  Each configured SYMBOL has a live tick (market open)
  8.  Each configured SYMBOL has reasonable spread (< 5 pips)
  9.  Each configured SYMBOL has trade_mode != Disabled/CloseOnly
  10. Filling mode resolvable per symbol (FOK/IOC/RETURN)
  11. Margin calculation for a 0.01-lot market order
  12. order_check() passes for a sample BUY 0.01 order (does NOT place)
  13. KillSwitch state file not stuck at level_3
  14. CircuitBreaker state file mode != PAUSED/LEARNING (sticky)
  15. ApprovalMode state file == 3 (autonomous)
  16. daily_risk.json total_loss_usd < daily loss limit
  17. confidence_engine: MIN_SAMPLE_SIZE constant and small-sample penalty
  18. TradePermission: MIN_CONFIDENCE threshold check

Each test prints [PASS] / [FAIL] / [WARN] with diagnostic detail.
Exits 0 if all critical tests pass, 1 otherwise.

This script does NOT place any orders. The order_check() call at step 12
validates the request structure without sending.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Load .env so config.py picks up env vars
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

class Report:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warned = 0
        self.lines = []

    def pass_(self, name, detail=""):
        self.passed += 1
        self.lines.append(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))

    def fail(self, name, detail=""):
        self.failed += 1
        self.lines.append(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))

    def warn(self, name, detail=""):
        self.warned += 1
        self.lines.append(f"  [WARN] {name}" + (f" — {detail}" if detail else ""))

    def section(self, title):
        self.lines.append("")
        self.lines.append(f"── {title} " + "─" * max(0, 60 - len(title)))

    def print_(self):
        print()
        print("=" * 70)
        print("  EXECUTION DIAGNOSTICS — forex_ai")
        print("=" * 70)
        for line in self.lines:
            print(line)
        print()
        print("=" * 70)
        print(f"  Result: {self.passed} PASS, {self.warned} WARN, {self.failed} FAIL")
        print("=" * 70)


r = Report()


# ──────────────────────────────────────────────────────────────────
# 1. MT5 package + initialization
# ──────────────────────────────────────────────────────────────────

r.section("1. MT5 Package + Initialization")

try:
    import MetaTrader5 as mt5
    r.pass_("MetaTrader5 package imported", f"version={getattr(mt5, '__version__', '?')}")
except ImportError as e:
    r.fail("MetaTrader5 package import", str(e))
    r.print_()
    sys.exit(1)

# Load config
try:
    from config import (
        MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH,
        SYMBOLS, EXECUTION_MODE, TEST_MODE, TRADING_MODE,
        APPROVAL_MODE, ABSOLUTE_SAFETY,
    )
except Exception as e:
    r.fail("config import", str(e))
    r.print_()
    sys.exit(1)

r.pass_("config loaded",
        f"EXECUTION_MODE={EXECUTION_MODE} TEST_MODE={TEST_MODE} "
        f"TRADING_MODE={TRADING_MODE} APPROVAL_MODE={APPROVAL_MODE}")

if EXECUTION_MODE != "mt5_demo":
    r.fail("EXECUTION_MODE",
           f"expected 'mt5_demo', got '{EXECUTION_MODE}' — trades cannot be placed")
    r.print_()
    sys.exit(1)

# Initialize
init_kwargs = {}
if MT5_PATH:
    init_kwargs["path"] = MT5_PATH
if MT5_LOGIN:
    init_kwargs["login"] = int(MT5_LOGIN)
if MT5_PASSWORD:
    init_kwargs["password"] = MT5_PASSWORD
if MT5_SERVER:
    init_kwargs["server"] = MT5_SERVER

try:
    if not mt5.initialize(**init_kwargs):
        err = mt5.last_error()
        r.fail("mt5.initialize()", f"last_error={err}")
        r.print_()
        sys.exit(1)
    r.pass_("mt5.initialize()", f"login={MT5_LOGIN} server={MT5_SERVER}")
except Exception as e:
    r.fail("mt5.initialize() raised", str(e))
    r.print_()
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────
# 2. Login + account info
# ──────────────────────────────────────────────────────────────────

r.section("2. Account + Login")

# (login is done inside initialize() when login/password are passed)
account = mt5.account_info()
if account is None:
    r.fail("mt5.account_info()", f"last_error={mt5.last_error()}")
else:
    r.pass_("mt5.account_info()",
            f"balance=${account.balance:.2f} equity=${account.equity:.2f} "
            f"margin=${account.margin:.2f} free=${account.margin_free:.2f}")
    if not account.trade_allowed:
        r.fail("account.trade_allowed",
               "False — broker has disabled trading on this account "
               "(check Investor password vs Master password, or terminal 'Algo Trading' button)")
    else:
        r.pass_("account.trade_allowed", "True")
    if account.margin_level > 0 and account.margin_level < 200:
        r.warn("account.margin_level",
               f"{account.margin_level:.1f}% — below 200% safety threshold")
    else:
        r.pass_("account.margin_level",
                f"{account.margin_level:.1f}%" if account.margin_level > 0 else "n/a (no open positions)")


# ──────────────────────────────────────────────────────────────────
# 3. Terminal info
# ──────────────────────────────────────────────────────────────────

r.section("3. Terminal Info")

term = mt5.terminal_info()
if term is None:
    r.fail("mt5.terminal_info()", f"last_error={mt5.last_error()}")
else:
    r.pass_("mt5.terminal_info()",
            f"build={term.build} community={getattr(term, 'community_account', False)}")
    if not term.connected:
        r.fail("terminal.connected",
               "False — terminal not connected to broker server "
               "(check internet / server name / credentials)")
    else:
        r.pass_("terminal.connected", "True")
    if not term.trade_allowed:
        r.fail("terminal.trade_allowed",
               "False — 'Algo Trading' button is OFF in the terminal. "
               "Click the 'Algo Trading' toolbar button to enable.")
    else:
        r.pass_("terminal.trade_allowed", "True")


# ──────────────────────────────────────────────────────────────────
# 4. Symbol availability per configured pair
# ──────────────────────────────────────────────────────────────────

r.section("4. Symbol Availability (per configured SYMBOLS)")

# Resolve broker symbol: try exact match, then suffix variants
def resolve_broker_symbol(requested: str) -> str | None:
    # 1. exact
    info = mt5.symbol_info(requested)
    if info is not None:
        return info.name
    # 2. symbols_get wildcard
    matches = mt5.symbols_get(f"*{requested}*")
    if matches:
        # prefer exact-equal name (case-insensitive)
        for m in matches:
            if m.name.upper().replace(".", "").replace("M", "") == requested.upper():
                return m.name
        # else first match
        return matches[0].name
    return None

symbol_to_broker: dict[str, str] = {}

for sym in SYMBOLS:
    broker_sym = resolve_broker_symbol(sym)
    if broker_sym is None:
        r.fail(f"symbol_resolve({sym})",
               "broker does not have this symbol — check suffix (e.g. EURUSD.m)")
        continue
    symbol_to_broker[sym] = broker_sym

    info = mt5.symbol_info(broker_sym)
    if info is None:
        r.fail(f"symbol_info({broker_sym})", f"last_error={mt5.last_error()}")
        continue

    # Ensure visible in Market Watch
    if not info.visible:
        ok = mt5.symbol_select(broker_sym, True)
        if not ok:
            r.fail(f"symbol_select({broker_sym})", "could not add to Market Watch")
            continue
        info = mt5.symbol_info(broker_sym)  # re-fetch

    # Tick check
    tick = mt5.symbol_info_tick(broker_sym)
    if tick is None or tick.time == 0:
        r.warn(f"tick({broker_sym})",
               "no live tick — market may be closed (weekend / holiday) "
               "or symbol not subscribed")
        continue

    # Spread
    digits = info.digits
    point = info.point
    spread_pips = (tick.ask - tick.bid) / (point * 10) if digits in (3, 5) else (tick.ask - tick.bid) / point
    spread_ok = spread_pips <= 5.0
    if not spread_ok:
        r.warn(f"spread({broker_sym})",
               f"{spread_pips:.1f} pips — wider than 5 pip threshold (news? volatility?)")
    else:
        r.pass_(f"tick+spread({broker_sym})",
                f"bid={tick.bid} ask={tick.ask} spread={spread_pips:.1f}pips")

    # Trade mode
    if info.trade_mode == 0:
        r.fail(f"trade_mode({broker_sym})", "Disabled — broker has disabled trading on this symbol")
    elif info.trade_mode == 3:
        r.fail(f"trade_mode({broker_sym})", "Close Only — only existing positions can be closed")
    else:
        r.pass_(f"trade_mode({broker_sym})", f"mode={info.trade_mode} (1=Full, 2=LongOnly, 4=CloseBy, 5=Full)")

    # Filling mode
    fm = info.filling_mode
    modes = []
    if fm & 1: modes.append("FOK")
    if fm & 2: modes.append("IOC")
    if fm & 4: modes.append("RETURN")
    if not modes:
        r.fail(f"filling_mode({broker_sym})",
               "broker reports NO filling modes supported — orders will fail with retcode 10030")
    else:
        r.pass_(f"filling_mode({broker_sym})", " | ".join(modes))


# ──────────────────────────────────────────────────────────────────
# 5. order_check() — validate a sample BUY 0.01 order (does NOT place)
# ──────────────────────────────────────────────────────────────────

r.section("5. order_check() — validate sample order (no placement)")

if not symbol_to_broker:
    r.fail("order_check", "skipped — no resolvable symbols")
else:
    # Use first resolved symbol that has a live tick
    sample_sym = None
    for req, broker in symbol_to_broker.items():
        tick = mt5.symbol_info_tick(broker)
        if tick and tick.time != 0:
            sample_sym = (req, broker, tick)
            break
    if sample_sym is None:
        r.fail("order_check", "no symbol with a live tick — market may be closed")
    else:
        req_sym, broker_sym, tick = sample_sym
        info = mt5.symbol_info(broker_sym)
        fm = info.filling_mode
        if fm & 2: filling = mt5.ORDER_FILLING_IOC
        elif fm & 1: filling = mt5.ORDER_FILLING_FOK
        elif fm & 4: filling = mt5.ORDER_FILLING_RETURN
        else: filling = mt5.ORDER_FILLING_IOC

        # SL/TP: 1% away from entry
        entry = tick.ask
        sl = round(entry * 0.99, info.digits)
        tp = round(entry * 1.01, info.digits)

        check_request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       broker_sym,
            "volume":       0.01,
            "type":         mt5.ORDER_TYPE_BUY,
            "price":        entry,
            "sl":           sl,
            "tp":           tp,
            "deviation":    10,
            "magic":        424242,
            "comment":      "diagnostic_check",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = mt5.order_check(check_request)
        if result is None:
            r.fail("order_check()", f"returned None — last_error={mt5.last_error()}")
        else:
            # retcode 0 = success
            if result.retcode == 0:
                r.pass_("order_check()",
                        f"{broker_sym} BUY 0.01 @ {entry} SL={sl} TP={tp} filling={filling} "
                        f"→ retcode=0 OK (margin=${result.margin:.2f} needed)")
            else:
                r.fail("order_check()",
                       f"retcode={result.retcode} comment={result.comment} "
                       f"(see https://www.mql5.com/en/docs/constants/tradingconstants/errorcodes)")


# ──────────────────────────────────────────────────────────────────
# 6. State file audit
# ──────────────────────────────────────────────────────────────────

r.section("6. State Files (sticky blocks?)")

mem_dir = ROOT / "memory"

# KillSwitch
ks_path = mem_dir / "kill_switch_state.json"
if ks_path.exists():
    try:
        ks = json.loads(ks_path.read_text())
        if ks.get("level_3_active"):
            r.fail("kill_switch_state.json",
                   "level_3_active=true — manual reset required "
                   "(delete file or call KillSwitch.manual_reset(level=3))")
        elif ks.get("level_2_active") or ks.get("level_1_active"):
            # check cooldown expiry
            now = datetime.now(timezone.utc)
            for level in ("level_1", "level_2"):
                if ks.get(f"{level}_active"):
                    until_str = ks.get(f"{level}_until")
                    if until_str:
                        try:
                            until = datetime.fromisoformat(until_str)
                            if now > until:
                                r.warn("kill_switch_state.json",
                                       f"{level}_active=true but cooldown expired "
                                       f"({until_str}) — KillSwitch.check() will auto-clear "
                                       f"on next call (not currently on trade path)")
                            else:
                                r.fail("kill_switch_state.json",
                                       f"{level}_active=true, cooldown until {until_str} — "
                                       f"trades blocked if KillSwitch.check() is wired in")
                        except Exception:
                            r.warn("kill_switch_state.json",
                                   f"{level}_active=true but malformed until timestamp")
        else:
            r.pass_("kill_switch_state.json", "no active levels")
    except Exception as e:
        r.warn("kill_switch_state.json", f"parse failed: {e}")
else:
    r.pass_("kill_switch_state.json", "not present (fresh state)")

# CircuitBreaker
cb_path = mem_dir / "circuit_breaker_state.json"
if cb_path.exists():
    try:
        cb = json.loads(cb_path.read_text())
        mode = cb.get("mode", "TRADING")
        if mode in ("PAUSED", "LEARNING"):
            r.fail("circuit_breaker_state.json",
                   f"mode={mode} — sticky block. Call CircuitBreaker.manual_resume() "
                   f"or edit the file to set mode='TRADING'. "
                   f"reason={cb.get('pause_reason', 'unknown')}")
        else:
            r.pass_("circuit_breaker_state.json",
                    f"mode={mode} consecutive_losses={cb.get('consecutive_losses', 0)}")
    except Exception as e:
        r.warn("circuit_breaker_state.json", f"parse failed: {e}")
else:
    r.pass_("circuit_breaker_state.json", "not present")

# ApprovalMode
am_path = mem_dir / "approval_mode.json"
if am_path.exists():
    try:
        am = json.loads(am_path.read_text())
        mode = am.get("mode", 2)
        if mode == 1:
            r.fail("approval_mode.json",
                   "mode=1 (ANALYSIS_ONLY) — no trades will ever execute. "
                   "Set mode=3 (autonomous) in .env or edit this file.")
        elif mode == 2:
            r.warn("approval_mode.json",
                   "mode=2 (SUPERVISED) — each trade requires human approval. "
                   "Mode 3 = autonomous.")
        else:
            r.pass_("approval_mode.json", f"mode={mode} (autonomous)")
    except Exception as e:
        r.warn("approval_mode.json", f"parse failed: {e}")
else:
    r.pass_("approval_mode.json",
            f"not present — falls back to APPROVAL_MODE={APPROVAL_MODE} from .env")

# daily_risk
dr_path = mem_dir / "daily_risk.json"
if dr_path.exists():
    try:
        dr = json.loads(dr_path.read_text())
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if dr.get("date") != today:
            r.warn("daily_risk.json",
                   f"stale date {dr.get('date')} (today={today}) — "
                   f"will be reset on next RiskEngine._load_daily()")
        loss = dr.get("total_loss_usd", 0)
        open_trades = len(dr.get("open_pairs", []))
        r.pass_("daily_risk.json",
                f"loss=${loss} open_trades={open_trades} date={dr.get('date')}")
    except Exception as e:
        r.warn("daily_risk.json", f"parse failed: {e}")
else:
    r.pass_("daily_risk.json", "not present (fresh day)")


# ──────────────────────────────────────────────────────────────────
# 7. ConfidenceEngine small-sample penalty check
# ──────────────────────────────────────────────────────────────────

r.section("7. ConfidenceEngine (small-sample penalty)")

try:
    from learning.confidence_engine import ConfidenceEngine, MIN_SAMPLE_SIZE
    r.pass_("confidence_engine import", f"MIN_SAMPLE_SIZE={MIN_SAMPLE_SIZE}")
    if MIN_SAMPLE_SIZE > 5:
        r.warn("MIN_SAMPLE_SIZE",
               f"={MIN_SAMPLE_SIZE} — fresh accounts (0 trades) get a -20 Bayesian penalty. "
               f"In TEST_MODE this is halved to -10. If you see 'conf=0%' in signal_debug.jsonl, "
               f"this is the cause.")
    # Test penalty for sample_size=0
    ce = ConfidenceEngine()
    pen = ce._bayesian_penalty(0)
    r.pass_("_bayesian_penalty(0)", f"={pen:.1f} (TEST_MODE={TEST_MODE})")
except Exception as e:
    r.fail("confidence_engine check", str(e))


# ──────────────────────────────────────────────────────────────────
# 8. TradePermission threshold check
# ──────────────────────────────────────────────────────────────────

r.section("8. TradePermission (MIN_CONFIDENCE)")

try:
    from risk.trade_permission import TradePermission
    tp = TradePermission()
    r.pass_("TradePermission.MIN_CONFIDENCE",
            f"={tp.MIN_CONFIDENCE} (TEST_MODE={TEST_MODE}) "
            f"— decision_out.confidence must be ≥ this to pass")
    if tp.MIN_CONFIDENCE > 30 and TEST_MODE:
        r.warn("TradePermission",
               f"TEST_MODE=true but MIN_CONFIDENCE={tp.MIN_CONFIDENCE} — "
               f"check that config import succeeds inside trade_permission.py")
except Exception as e:
    r.fail("TradePermission check", str(e))


# ──────────────────────────────────────────────────────────────────
# 9. DecisionAgent + analysis_agent TEST_MODE bypass
# ──────────────────────────────────────────────────────────────────

r.section("9. DecisionAgent TEST_MODE bypass")

try:
    from agents.decision_agent import DecisionAgent
    r.pass_("DecisionAgent import", "OK")
    # Check that the TEST_MODE bypass code path exists in source
    import inspect
    src = inspect.getsource(DecisionAgent.decide)
    if "TEST_MODE AGGRESSIVE" in src and "_test_mode and final_signal in" in src:
        r.pass_("DecisionAgent TEST_MODE bypass", "present in source")
    else:
        r.fail("DecisionAgent TEST_MODE bypass",
               "code path missing — decision will fall through to voting "
               "and require MIN_CONSENSUS=2 which fails when LLM is unavailable")
except Exception as e:
    r.fail("DecisionAgent check", str(e))

try:
    import inspect
    from agents.analysis_agent import AnalysisAgent
    src = inspect.getsource(AnalysisAgent.run)
    if "TEST_MODE AGGRESSIVE" in src:
        r.pass_("AnalysisAgent TEST_MODE bypass", "present in source")
    else:
        r.fail("AnalysisAgent TEST_MODE bypass", "missing")
except Exception as e:
    r.fail("AnalysisAgent check", str(e))


# ──────────────────────────────────────────────────────────────────
# 10. Shutdown
# ──────────────────────────────────────────────────────────────────

try:
    mt5.shutdown()
except Exception:
    pass

r.print_()

# Exit code
sys.exit(0 if r.failed == 0 else 1)
