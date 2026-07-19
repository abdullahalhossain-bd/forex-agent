# execution/execution_router.py  —  Day 31 | Paper vs MT5 Demo Switch
# ============================================================
# Day 9X fix (this patch): MT5 reconnect হওয়ার পরেও router পুরনো
# connection state ধরে রাখছিল, shutdown()-এ health monitor thread
# properly stop হতো না, আর order execute করার আগে connection alive
# কিনা সেটাও verify করা হতো না। এই patch-এ:
#   - _reconnect_lock add হলো (race-free reconnect)
#   - _ensure_mt5_connected() add হলো — প্রতিটা order-এর আগে call হয়
#   - HealthMonitor.start()/.stop() ব্যবহার করা হলো (double-thread bug fix)
#   - shutdown() এখন health monitor + connection দুটোই safely বন্ধ করে
#   - order placement-এর আগে lot/direction/broker_symbol guard add হলো
#   - entry price None হলে warn/strict-block করার logic add হলো
#
# Day 9X+1 fix:
#   - _ensure_mt5_connected() এ retry loop যোগ করা হয়েছে।
#     আগে HealthMonitor "waiting 10s" থাকার সময় execute() call হলে
#     router নিজে একবার reconnect try করে fail করতো। এখন নিজেই
#     MAX_RECONNECT_WAIT সেকেন্ড পর্যন্ত poll করে — HealthMonitor
#     যখন reconnect করবে, router সেটা ধরে ফেলবে এবং trade যাবে।
#
# Day 90+ fix (MT5 disconnect-flapping bugfix):
#   - ExecutionRouter আগে নিজের নতুন MT5Connection() বানাতো, যেটা
#     core/runtime.py-এর registry-তে আগে থেকেই register হওয়া shared
#     "mt5_connection" instance-এর পাশাপাশি আরেকটা সম্পূর্ণ আলাদা MT5
#     session খুলতো (নিজের mt5.initialize()/mt5.shutdown() সহ)।
#     MetaTrader5 package প্রসেস-লেভেলে single session ধরে, তাই দুটো
#     MT5Connection object একসাথে initialize/shutdown করায় একে অপরের
#     session ভেঙে দিচ্ছিল — এটাই বারবার "MT5 connection lost" flapping
#     এর root cause ছিল (প্রতি ৫-১০ মিনিটে disconnect/reconnect)।
#
#     Fix: constructor এখন ঐচ্ছিক `mt5_conn` parameter নেয়। যদি একটা
#     already-connected MT5Connection ইনজেক্ট করা হয় (যেমন registry থেকে
#     resolve করে runtime.py পাস করবে), router সেটাই reuse করবে — নিজে
#     নতুন বানাবে না। parameter না দিলে আগের মতোই behave করবে (backward
#     compatible), যাতে অন্য কোথাও সরাসরি ExecutionRouter() বানালে ভেঙে
#     না যায়।
#
#     এছাড়া: shared connection ব্যবহার হলে router নিজের local
#     HealthMonitor thread চালু করে না (কারণ সেটাও duplicate
#     polling/contention তৈরি করতো) — central health monitor-এর উপর
#     নির্ভর করে।
# ============================================================

import threading
import time

from utils.logger import get_logger
from config import validate_mt5_config

log = get_logger("execution_router")

ENTRY_PRICE_STRICT_MODE = True

# Day 9X+1: _ensure_mt5_connected() retry config.
# HealthMonitor "waiting 10s" থাকার সময় trade আসলে এই window-এ
# MT5 poll করবো। 30s মানে HealthMonitor-এর 10s wait + 1-2 reconnect
# attempt-এর জন্য যথেষ্ট।
_RECONNECT_POLL_INTERVAL = 2.0   # seconds between each poll
_RECONNECT_MAX_WAIT      = 30.0  # max total wait before giving up


def _log_event(event: str, **fields):
    try:
        from core.execution_logger import log_event
        log_event(event, **fields)
    except Exception:
        pass


def _check_absolute_safety(symbol: str) -> tuple[bool, str]:
    try:
        from config import ABSOLUTE_SAFETY, TEST_MODE
        if not ABSOLUTE_SAFETY:
            return True, "ABSOLUTE_SAFETY disabled"
    except Exception:
        pass

    try:
        from data.live_feed import get_live_feed
        feed = get_live_feed()
        safe, reason = feed.is_safe_to_trade(symbol)
        if not safe:
            log.warning(f"[ABSOLUTE_SAFETY] BLOCKED {symbol}: {reason}")
        return safe, reason
    except Exception as e:
        log.error(
            f"[ABSOLUTE_SAFETY] check itself failed — FAIL-CLOSED: {e}",
            exc_info=True,
        )
        return False, f"safety check unavailable: {e}"


class ExecutionRouter:
    def __init__(self, mode: str = None, db=None, paper_trader=None, mt5_conn=None):
        """
        Args:
            mt5_conn: Day 90+ hotfix — ঐচ্ছিক, ইতিমধ্যে-connected
                MT5Connection instance (সাধারণত core.runtime-এর registry
                থেকে resolve করে পাস করা হয়)। দেওয়া হলে router এটাই
                ব্যবহার করবে এবং নিজের নতুন connection বানাবে না — এতে
                একই process-এ একাধিক mt5.initialize() session খোলা আটকায়।
                None দিলে আগের মতোই নিজে connection বানাবে (backward
                compatible)।
        """
        # BUG FIX (execution-parity audit): this used to hardcode
        # `self.mode = "mt5_demo"` regardless of the `mode` argument the
        # caller passed in — meaning AITrader(execution_mode=...) could
        # never actually select anything else, and "Real" money execution
        # had no way to be reached even once implemented below. The `mode`
        # parameter is now honored.
        self.mode = (mode or "mt5_demo").lower()
        self._mt5_executor = None
        self._db = db
        self._simulation_mode = False
        self._reconnect_lock = threading.Lock()
        self._owns_mt5_conn = False

        try:
            from config import SIMULATION_MODE
            self._simulation_mode = bool(SIMULATION_MODE)
        except Exception:
            pass

        # Co-founder fix: MT5 fallback to simulation
        try:
            from config import MT5_FALLBACK_TO_SIMULATION
            self._mt5_fallback_to_sim = bool(MT5_FALLBACK_TO_SIMULATION)
        except Exception:
            self._mt5_fallback_to_sim = True

        if self._simulation_mode:
            self._init_simulation_mode()
            return

        # Backtest mode never places broker orders — backtest.unified_engine
        # calls AITrader.evaluate_decision_core() directly and routes fills
        # through backtest.broker_sim.BrokerSimulator, never through this
        # router. This branch exists only so constructing an AITrader with
        # execution_mode="backtest" doesn't hit the `else: raise ValueError`
        # below if something does touch self._router.
        if self.mode == "backtest":
            self._init_simulation_mode()
            log.info("[ExecutionRouter] Mode: BACKTEST (router inert — "
                     "backtest.unified_engine drives fills directly)")
            return

        if self.mode in ("mt5_demo", "mt5_live"):
            if self.mode == "mt5_live":
                # ── REAL MONEY — explicit, non-bypassable safety gate ──
                # Per the execution-parity audit: "Real" execution did not
                # exist as a code path before this fix. It is deliberately
                # NOT a one-line config flip. Two independent conditions
                # must both be true, and neither is the default:
                #   1. ALLOW_REAL_MONEY_TRADING=true in the environment
                #      (not .env-committed by default — see config.py).
                #   2. Separate MT5_REAL_* credentials configured — real
                #      and demo credentials are never allowed to be the
                #      same variable, so a stale/default demo login can't
                #      silently end up trading a real account.
                # Failure here raises, never silently falls back to demo
                # or simulation — going quiet about "I couldn't get you
                # into a real account" is fine; going quiet about "I put
                # your real-money trade into a demo/sim account instead"
                # would hide a correctness bug behind an apparently-working
                # system, which is worse.
                from config import ALLOW_REAL_MONEY_TRADING, MT5_REAL_LOGIN, MT5_REAL_PASSWORD, MT5_REAL_SERVER
                if not ALLOW_REAL_MONEY_TRADING:
                    raise RuntimeError(
                        "EXECUTION_MODE=mt5_live requested but ALLOW_REAL_MONEY_TRADING "
                        "is not set to true. Real-money execution requires an explicit, "
                        "separate opt-in — set ALLOW_REAL_MONEY_TRADING=true AND "
                        "MT5_REAL_LOGIN/MT5_REAL_PASSWORD/MT5_REAL_SERVER in your "
                        "environment (not just MT5_LOGIN/PASSWORD/SERVER, which remain "
                        "demo-only) before this mode will initialize."
                    )
                if not (MT5_REAL_LOGIN and MT5_REAL_PASSWORD and MT5_REAL_SERVER):
                    raise RuntimeError(
                        "ALLOW_REAL_MONEY_TRADING=true but MT5_REAL_LOGIN/"
                        "MT5_REAL_PASSWORD/MT5_REAL_SERVER are not fully set — "
                        "refusing to initialize real-money execution with incomplete "
                        "or fallback credentials."
                    )
                _login, _password, _server = MT5_REAL_LOGIN, MT5_REAL_PASSWORD, MT5_REAL_SERVER
                _mt5_fallback_to_sim_effective = False  # a real-money order NEVER silently
                                                          # becomes a simulated one on failure
            try:
                from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH
                from broker.mt5_connection import get_mt5_connection
                from broker.health_monitor import HealthMonitor
                from broker.account_manager import AccountManager
                from broker.order_manager import OrderManager
                from broker.journal_bridge import JournalBridge
            except Exception as e:
                if self.mode == "mt5_demo" and self._mt5_fallback_to_sim:
                    log.warning(f"[ExecutionRouter] MT5 imports failed ({e}) — falling back to SIMULATION")
                    self._init_simulation_mode()
                    return
                raise

            if self.mode == "mt5_demo":
                if not self._mt5_fallback_to_sim:
                    validate_mt5_config()
            else:
                self._mt5_fallback_to_sim = False  # real money: see above, no silent fallback

            if self.mode == "mt5_live":
                MT5_LOGIN, MT5_PASSWORD, MT5_SERVER = _login, _password, _server

            if mt5_conn is not None:
                self._mt5_conn = mt5_conn
                if not getattr(self._mt5_conn, "connected", False):
                    if not self._mt5_conn.connect():
                        msg = "Injected shared MT5 connection failed to connect."
                        if self._mt5_fallback_to_sim:
                            log.warning(f"[ExecutionRouter] {msg} FALLING BACK TO SIMULATION")
                            self._init_simulation_mode()
                            return
                        raise RuntimeError(msg)
                log.info("[ExecutionRouter] Using shared/injected MT5Connection")
            else:
                if self._mt5_fallback_to_sim and not MT5_LOGIN:
                    log.warning("[ExecutionRouter] MT5_LOGIN not set — FALLING BACK TO SIMULATION")
                    self._init_simulation_mode()
                    return
                # Bug fix: was `MT5Connection(...)` — built yet another
                # independent session instead of reusing whichever one
                # core/runtime.py (or data/fetcher.py / data_orchestrator.py)
                # already has open for this same (login, server). Route
                # through the singleton factory so they all share one.
                self._mt5_conn = get_mt5_connection(
                    login=MT5_LOGIN, password=MT5_PASSWORD,
                    server=MT5_SERVER, path=MT5_PATH or None,
                    auto_connect=True,
                )
                if not self._mt5_conn.connected:
                    msg = "MT5 demo connection failed."
                    if self._mt5_fallback_to_sim:
                        log.warning(f"[ExecutionRouter] {msg} FALLING BACK TO SIMULATION")
                        self._init_simulation_mode()
                        return
                    raise RuntimeError(msg)
                self._owns_mt5_conn = True

            self._account_manager = AccountManager(self._mt5_conn)
            self._health_monitor = HealthMonitor(
                self._mt5_conn,
                on_disconnect=lambda msg: (
                    log.warning(f"[Router] MT5 connection lost — trading paused"),
                    _log_event("router.mt5.disconnect", reason=msg)
                ),
                on_reconnect=lambda msg: (
                    log.info(f"[Router] MT5 connection restored — trading resumed"),
                    _log_event("router.mt5.reconnect")
                ),
                on_fatal=lambda msg: log.error(f"[Router] {msg}"),
            )
            # Day 90+ hotfix: শেয়ার্ড connection হলে নিজের local
            # HealthMonitor thread চালু করি না — এতে dual polling
            # contention (false-positive disconnect flapping) কমে।
            # শুধু নিজে owned connection বানালেই local monitor চালাও।
            if self._owns_mt5_conn:
                try:
                    if hasattr(self._health_monitor, "start"):
                        self._health_monitor.start()
                        log.info("[ExecutionRouter] HealthMonitor started")
                    else:
                        self._health_thread = threading.Thread(
                            target=self._health_monitor.run_loop,
                            name="mt5_health_monitor",
                            daemon=True,
                        )
                        self._health_thread.start()
                        log.warning(
                            "[ExecutionRouter] HealthMonitor has no start() — "
                            "falling back to manual thread"
                        )
                except Exception as e:
                    log.warning(f"[ExecutionRouter] Could not start HealthMonitor: {e}")
            else:
                log.info(
                    "[ExecutionRouter] Shared MT5 connection in use — skipping local "
                    "HealthMonitor thread (relying on the central health monitor instead)"
                )
                self._health_monitor = None

            self._order_manager  = OrderManager(self._mt5_conn, self._account_manager)
            self._journal_bridge = JournalBridge(db=self._db)
            log.info("[ExecutionRouter] Mode: MT5_DEMO (real broker, demo account)")

        else:
            raise ValueError(f"Unknown EXECUTION_MODE: {self.mode}")

    def _init_simulation_mode(self) -> None:
        """Initialize the router in SIMULATION mode."""
        from execution.simulated_executor import SimulatedExecutor
        self._simulation_mode = True
        self._order_manager   = SimulatedExecutor(db=self._db)
        self._journal_bridge  = None
        self._mt5_conn        = None
        self._account_manager = None
        self._health_monitor  = None
        self._owns_mt5_conn   = False
        log.info("[ExecutionRouter] Mode: SIMULATION (no broker contact — dry run)")

    # ─────────────────────────────────────────────
    # CONNECTION HEALTH
    # ─────────────────────────────────────────────

    def _ensure_mt5_connected(self) -> bool:
        """
        Day 9X+1 fix: order execute করার আগে MT5 connection alive কিনা
        নিশ্চিত করে। HealthMonitor "waiting 10s" থাকার সময় trade আসলে
        আগে একবার try করে fail করতো। এখন _RECONNECT_MAX_WAIT সেকেন্ড
        পর্যন্ত poll করে — HealthMonitor যখন reconnect করবে, এই loop
        সেটা ধরে ফেলবে এবং trade proceed করবে।

        Flow:
          1. mt5.account_info() → alive? → return True immediately
          2. না হলে নিজে reconnect try করো (lock দিয়ে race-safe)
          3. success হলে → True
          4. না হলে poll করতে থাকো (_RECONNECT_POLL_INTERVAL interval-এ)
             যতক্ষণ না _RECONNECT_MAX_WAIT শেষ হয় বা connection ফিরে আসে
          5. timeout হলে → False (trade block)
        """
        if self._simulation_mode:
            return True

        try:
            import MetaTrader5 as mt5

            # ── Fast path: already connected ──
            if mt5.account_info() is not None:
                return True

            log.warning(
                "[ExecutionRouter] MT5 connection lost before execution "
                "— attempting reconnect..."
            )

            # ── Try immediate reconnect (lock-protected) ──
            with self._reconnect_lock:
                if mt5.account_info() is not None:
                    return True  # another thread already fixed it

                try:
                    self._mt5_conn.disconnect()
                except Exception:
                    pass

                if self._mt5_conn.connect():
                    log.info("[ExecutionRouter] MT5 reconnected successfully (immediate)")
                    return True

            # ── Poll loop: wait for HealthMonitor to reconnect ──
            # HealthMonitor "waiting 10s" শেষ হলে reconnect করবে।
            # আমরা সেটার জন্য অপেক্ষা করবো।
            deadline = time.monotonic() + _RECONNECT_MAX_WAIT
            poll_count = 0
            while time.monotonic() < deadline:
                time.sleep(_RECONNECT_POLL_INTERVAL)
                poll_count += 1

                if mt5.account_info() is not None:
                    log.info(
                        f"[ExecutionRouter] MT5 connection detected after "
                        f"{poll_count * _RECONNECT_POLL_INTERVAL:.0f}s poll "
                        f"(HealthMonitor reconnected) — proceeding with trade"
                    )
                    return True

                remaining = deadline - time.monotonic()
                log.debug(
                    f"[ExecutionRouter] Waiting for MT5 reconnect... "
                    f"({remaining:.0f}s remaining)"
                )

            log.error(
                f"[ExecutionRouter] MT5 reconnect failed after "
                f"{_RECONNECT_MAX_WAIT:.0f}s — trade aborted"
            )
            return False

        except Exception as e:
            log.error(
                f"[ExecutionRouter] Connection check failed: {e}",
                exc_info=True,
            )
            return False

    # ─────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ─────────────────────────────────────────────

    def execute(self, decision_result: dict) -> dict | None:
        # ── HARD GATE 1: decision must be BUY/SELL ──
        if decision_result.get("decision") not in ("BUY", "SELL"):
            log.info(f"[ExecutionRouter] No action — decision={decision_result.get('decision')}")
            return None

        # ── HARD GATE 2: explicit permission flag ──
        if "trade_allowed" in decision_result and not decision_result["trade_allowed"]:
            log.error(
                f"[ExecutionRouter] ⛔ HARD BLOCK — trade_allowed=False for "
                f"{decision_result.get('symbol')} {decision_result.get('decision')}. "
                f"Refusing to execute."
            )
            _log_event("router.execute.fail",
                       symbol=decision_result.get("symbol", "?"),
                       reason="hard_gate: trade_allowed=False",
                       stage="permission_bypass_blocked")
            return None

        # ── HARD GATE 3: lot cap ──
        lot = decision_result.get("lot", 0.01)
        try:
            from config import MAX_LOT
        except Exception:
            MAX_LOT = 0.20
        if lot > MAX_LOT:
            log.warning(
                f"[ExecutionRouter] ⛔ LOT CAP — lot={lot} exceeds MAX_LOT={MAX_LOT}. "
                f"Capping to {MAX_LOT}."
            )
            decision_result = {**decision_result, "lot": MAX_LOT}
            _log_event("router.execute.lot_capped",
                       symbol=decision_result.get("symbol", "?"),
                       original_lot=lot, capped_lot=MAX_LOT)

        # ── HARD GATE 4: lot must be positive ──
        if lot <= 0:
            log.error(f"[ExecutionRouter] ⛔ INVALID LOT — lot={lot}. Refusing to execute.")
            _log_event("router.execute.fail",
                       symbol=decision_result.get("symbol", "?"),
                       reason=f"invalid lot={lot}",
                       stage="lot_validation")
            return None

        return self._execute_mt5_demo(decision_result)

    def _execute_mt5_demo(self, decision_result: dict) -> dict | None:
        symbol    = decision_result.get("symbol", "EURUSD")
        direction = decision_result.get("decision")
        lot       = decision_result.get("lot", 0.01)
        sl        = decision_result.get("sl")
        tp        = decision_result.get("tp")

        _log_event("router.execute.start", symbol=symbol, decision=direction,
                   lot=lot, sl=sl, tp=tp, simulation=self._simulation_mode)

        # ── Day 9X+1: connection-alive check with poll loop ──────
        if not self._ensure_mt5_connected():
            log.error(
                f"[ExecutionRouter] MT5 not connected after {_RECONNECT_MAX_WAIT:.0f}s "
                f"— refusing to execute {symbol} {direction}"
            )
            _log_event("router.execute.fail", symbol=symbol,
                       reason="mt5 disconnected after poll timeout",
                       stage="connection_check")
            return None

        # ── ABSOLUTE_SAFETY hard gate ─────────────────────────────
        safe, reason = _check_absolute_safety(symbol)
        if not safe:
            log.warning(
                f"[ExecutionRouter] ABSOLUTE_SAFETY blocked trade — "
                f"{symbol} {direction}: {reason}"
            )
            _log_event("router.execute.fail", symbol=symbol, reason=reason,
                       stage="absolute_safety")
            return None

        # ── direction guard ───────────────────────────────────────
        if direction not in ("BUY", "SELL"):
            log.error(f"[ExecutionRouter] ⛔ INVALID DIRECTION — direction={direction}.")
            _log_event("router.execute.fail", symbol=symbol,
                       reason=f"invalid direction={direction}",
                       stage="direction_validation")
            return None

        # ── lot guard ─────────────────────────────────────────────
        if lot <= 0:
            log.error(f"[ExecutionRouter] ⛔ INVALID LOT — lot={lot}.")
            _log_event("router.execute.fail", symbol=symbol,
                       reason=f"invalid lot={lot}",
                       stage="lot_validation")
            return None

        # ── entry price validation ────────────────────────────────
        if decision_result.get("entry") is None:
            log.warning(f"[ExecutionRouter] No entry price for {symbol}")
            if ENTRY_PRICE_STRICT_MODE:
                _log_event("router.execute.fail", symbol=symbol,
                           reason="entry price is None",
                           stage="entry_price_validation")
                return None

        # ── broker permission check ───────────────────────────────
        if self._simulation_mode:
            broker_symbol = symbol
        else:
            perm = self._account_manager.trading_permission(
                symbol=symbol,
                risk_engine_ok=True,
            )
            if not perm["allowed"]:
                log.warning(
                    f"[ExecutionRouter] MT5 demo — trade blocked: {perm['failed_checks']}"
                )
                _log_event("router.execute.fail", symbol=symbol,
                           reason=f"trading_permission: {perm['failed_checks']}",
                           stage="trading_permission")
                return None
            broker_symbol = perm["broker_symbol"]

        # ── broker_symbol guard ───────────────────────────────────
        if broker_symbol is None:
            log.error("[ExecutionRouter] broker_symbol is None — refusing to execute")
            _log_event("router.execute.fail", symbol=symbol,
                       reason="broker_symbol is None",
                       stage="broker_symbol_validation")
            return None

        # ── place order ───────────────────────────────────────────
        try:
            order_result = self._order_manager.place_market_order(
                symbol=broker_symbol,
                direction=direction,
                lot=lot,
                sl=sl,
                tp=tp,
                comment="ai_trader_demo" if not self._simulation_mode else "ai_trader_sim",
            )
        except Exception as e:
            log.error(
                f"[ExecutionRouter] place_market_order raised: {e}",
                exc_info=True,
            )
            _log_event("router.execute.fail", symbol=symbol,
                       reason=f"place_market_order raised: {e}",
                       stage="order_send")
            return None

        if not order_result.get("success"):
            log.error(
                f"[ExecutionRouter] MT5 demo — order failed: {order_result.get('reason')}"
            )
            _log_event("router.execute.fail", symbol=symbol,
                       reason=order_result.get('reason', 'unknown'),
                       stage="order_result",
                       retcode=order_result.get('retcode'))
            return None

        filled_entry = order_result.get("price", decision_result.get("entry"))
        ticket       = order_result.get("ticket")
        # Audit fix (EX-1): use the ACTUAL filled volume (order_result may
        # differ from the requested `lot` on a partial fill), not the
        # originally requested lot — downstream position sizing / risk
        # tracking must reflect real exposure, not intent.
        filled_lot   = order_result.get("volume", lot)
        if order_result.get("partial_fill"):
            log.warning(
                f"[ExecutionRouter] ⚠️  PARTIAL FILL propagated — {symbol} "
                f"requested={lot} filled={filled_lot} "
                f"remaining={order_result.get('remaining_volume')} ticket={ticket}"
            )
            _log_event("router.execute.partial_fill", symbol=symbol, ticket=ticket,
                       requested_lot=lot, filled_lot=filled_lot,
                       remaining_lot=order_result.get("remaining_volume"))

        # ── simulation: skip DB journal ───────────────────────────
        if self._simulation_mode or self._journal_bridge is None:
            log.info(
                f"[ExecutionRouter] ✅ SIMULATED order FILLED — {direction} {broker_symbol} "
                f"lot={filled_lot} ticket={ticket}"
            )
            _log_event("router.execute.success", symbol=symbol, ticket=ticket,
                       price=filled_entry, lot=filled_lot, simulation=True)
            return {
                "id": None, "status": "SIMULATED",
                "broker_symbol": broker_symbol, "ticket": ticket,
                "entry": filled_entry, "sl": sl, "tp": tp,
                "lot": filled_lot, "type": direction, "pair": broker_symbol,
            }

        # ── DB journal ────────────────────────────────────────────
        # Audit fix (EX-1): journal must record the ACTUAL filled lot on a
        # partial fill, not the originally requested lot — otherwise the
        # DB/journal overstates real exposure and P&L math downstream
        # (position sizing, correlation, drawdown) is computed on a lot
        # size that was never actually filled.
        journal_decision_result = (
            {**decision_result, "lot": filled_lot}
            if filled_lot != lot else decision_result
        )
        trade_id       = None
        journal_failed = False
        try:
            trade_id = self._journal_bridge.log_mt5_open(
                decision_result  = journal_decision_result,
                broker_symbol    = broker_symbol,
                filled_entry     = filled_entry,
                mt5_order_ticket = ticket,
            )
        except Exception as e:
            journal_failed = True
            log.error(
                f"[ExecutionRouter] ⚠️  ORPHAN POSITION — broker filled {ticket} "
                f"({broker_symbol} {direction} lot={lot}) but DB journal failed: {e}",
                exc_info=True,
            )
            _log_event("orphan.position", symbol=symbol, ticket=ticket,
                       reason=f"journal failed after broker fill: {e}",
                       broker_symbol=broker_symbol, direction=direction, lot=lot)

        log.info(
            f"[ExecutionRouter] ✅ MT5 demo order FILLED — {direction} {broker_symbol} "
            f"lot={filled_lot} ticket={ticket} → DB #{trade_id}"
            + ("  ⚠️  ORPHAN (DB write failed)" if journal_failed else "")
        )
        _log_event("router.execute.success", symbol=symbol, ticket=ticket,
                   price=filled_entry, lot=filled_lot, trade_id=trade_id,
                   orphan=journal_failed)

        # Round-30: Register the MT5 ticket → DB trade_id mapping so
        # PositionManager can track this position and detect closes.
        # Previously register_open() was never called from the router,
        # leaving PositionManager._ticket_to_db_id empty for all MT5 trades.
        if trade_id is not None and ticket is not None:
            try:
                # PositionManager is per-AITrader, not in registry.
                # The AITrader that owns this router will call
                # position_manager.register_open() itself — but we also
                # store the mapping in the execution result so the caller
                # can use it.
                pass  # AITrader handles register_open() in its execute path
            except Exception:
                pass

        return {
            "id":            trade_id,
            "status":        "FILLED" if not journal_failed else "FILLED_ORPHAN",
            "broker_symbol": broker_symbol,
            "ticket":        ticket,
            "entry":         filled_entry,
            "sl":            sl,
            "tp":            tp,
            "lot":           filled_lot,
            "type":          direction,
            "pair":          broker_symbol,
        }

    def shutdown(self) -> None:
        try:
            # Day 90+ hotfix: শুধু নিজের owned health monitor/connection
            # বন্ধ করো। শেয়ার্ড connection হলে অন্য components এখনো
            # সেটা ব্যবহার করতে পারে — তাই shutdown স্কিপ করি, সেটার
            # lifecycle যে রেজিস্ট্রি/runtime এটার মালিকানা নিয়েছে সে-ই
            # সামলাবে।
            if hasattr(self, "_health_monitor") and self._health_monitor:
                try:
                    if hasattr(self._health_monitor, "stop"):
                        self._health_monitor.stop()
                    else:
                        log.warning(
                            "[ExecutionRouter] HealthMonitor has no stop() — "
                            "thread will exit only when the process exits"
                        )
                except Exception as e:
                    log.warning(f"[ExecutionRouter] HealthMonitor.stop() failed: {e}")

            if hasattr(self, "_health_thread") and self._health_thread:
                try:
                    self._health_thread.join(timeout=2.0)
                except Exception:
                    pass

            if self._owns_mt5_conn and hasattr(self, "_mt5_conn") and self._mt5_conn:
                try:
                    self._mt5_conn.disconnect()
                except Exception as e:
                    log.warning(f"[ExecutionRouter] MT5 disconnect failed: {e}")
            elif hasattr(self, "_mt5_conn") and self._mt5_conn:
                log.info(
                    "[ExecutionRouter] Shared MT5 connection in use — skipping "
                    "disconnect on this router's shutdown (owned elsewhere)"
                )

            log.info("[ExecutionRouter] Shutdown complete")

        except Exception as e:
            log.error(f"[ExecutionRouter] Shutdown failed: {e}", exc_info=True)