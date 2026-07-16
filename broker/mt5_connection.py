# broker/mt5_connection.py

import time
from datetime import datetime
from threading import Lock
from utils.logger import get_logger

log = get_logger("mt5_connection")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    log.warning(
        "MetaTrader5 package not found. Install with: pip install MetaTrader5"
    )


def get_mt5_connection(
    login: int,
    password: str,
    server: str,
    path: str = None,
    auto_connect: bool = True,
) -> "MT5Connection":
    """Module-level shortcut for MT5Connection.get_instance().

    Use this everywhere an MT5 connection is needed instead of calling
    MT5Connection(...) directly, so the whole process shares one real
    terminal session per (login, server) instead of each caller silently
    creating (and re-logging-in) its own.
    """
    return MT5Connection.get_instance(
        login=login, password=password, server=server,
        path=path, auto_connect=auto_connect,
    )


class MT5Connection:
    MAX_RETRIES = 3
    RETRY_DELAY_SEC = 5

    # Day 90+ hotfix: health-check tuning.
    # Previously is_alive() called BOTH terminal_info() AND account_info()
    # under the same lock on every check, and marked the connection dead
    # on the very first None result. With multiple AITrader instances
    # (one per pair) all hammering the same MT5 terminal through this
    # shared lock, terminal_info()/account_info() would occasionally
    # return None just from momentary contention/latency — not a real
    # disconnect. That caused the "MT5 connection lost" flapping seen
    # in logs every 5-10 minutes even though MT5 was actually fine.
    # Day 90+ hotfix tuning (kept as-is):
    HEALTH_CHECK_RETRIES = 2          # extra in-place retries before declaring dead
    HEALTH_CHECK_RETRY_DELAY = 0.5    # seconds between in-place retries

    # Round-5 audit fix: cache is_alive() result for a few seconds.
    # The operator's log showed "[MT5Connection] Health check failed
    # after 3 attempts" firing on EVERY cycle, followed immediately by
    # auto-reconnect SUCCESS — meaning the connection was actually
    # fine, the health check was just too aggressive (3 attempts × 0.5s
    # = 1.5s of lock contention per cycle, ×6 pairs = 9s wasted per
    # cycle on health checks alone).
    #
    # Caching the result for 5s means:
    #   - First MT5 op in any 5s window: full health check runs
    #   - Subsequent ops in the same 5s window: cached True (no lock,
    #     no terminal_info() call)
    #   - If the cached result is False, the cache is bypassed and a
    #     fresh check runs (so reconnect logic still fires promptly)
    HEALTH_CHECK_CACHE_SEC = 5.0
    _last_health_check_ts: float = 0.0
    _last_health_check_result: bool = True

    # P1 fix (audit §4.1): terminal_info() only proves the terminal app is
    # running, not that the account session is still authenticated — a
    # broker-side session invalidation can leave terminal_info() healthy
    # while trading is actually blocked. Run a cheap account_info() check
    # every Nth health check (not every call, to avoid reintroducing the
    # Day 90 false-positive-flapping problem) so auth-level failures don't
    # stay invisible for an entire session.
    AUTH_CHECK_EVERY_N_HEALTH_CHECKS = 20

    MT5_LOCK = Lock()

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: str = None
    ):
        self.login = login
        self.password = password
        self.server = server
        self.path = path

        self.connected = False
        self.connected_at = None
        self.last_ping = None

        # Day 90+ hotfix: track consecutive health-check failures so we
        # can distinguish "one flaky call" from "actually disconnected".
        self._consecutive_failures = 0
        # P1 fix (audit §4.1): counts total is_alive() calls so we know
        # when to run the periodic account_info() auth check.
        self._health_check_count = 0

    # ==========================================================
    # SINGLETON FACTORY
    # ==========================================================
    # Bug fix: multiple modules (core/runtime.py, data/fetcher.py,
    # data/data_orchestrator.py, execution/execution_router.py) each called
    # `MT5Connection(login=..., password=..., server=...)` directly whenever
    # no shared instance had been explicitly injected into them. Every one
    # of those calls is a brand-new object that independently runs
    # mt5.shutdown() + mt5.initialize() + mt5.login() against the SAME
    # underlying MT5 terminal, which is what produced the duplicate
    # "MT5 CONNECTION" banners seen back-to-back in the logs (e.g.
    # 16:58:37 and 16:58:47) — two full re-logins a few seconds apart,
    # each one silently invalidating the other's session.
    #
    # get_mt5_connection() below is the fix: it's the one place that should
    # be used to obtain an MT5Connection anywhere in the codebase. The same
    # (login, server) pair always returns the exact same already-connected
    # instance instead of building + logging in again.
    _instances: dict[tuple, "MT5Connection"] = {}
    _instances_lock = Lock()

    @classmethod
    def get_instance(
        cls,
        login: int,
        password: str,
        server: str,
        path: str = None,
        auto_connect: bool = True,
    ) -> "MT5Connection":
        """Return the shared MT5Connection for this (login, server), creating
        and connecting it on first use. Subsequent calls with the same
        (login, server) reuse the existing instance — no duplicate
        mt5.initialize()/mt5.login() calls, no duplicate connection banners.
        """
        key = (login, server)
        with cls._instances_lock:
            inst = cls._instances.get(key)
            if inst is None:
                inst = cls(login=login, password=password, server=server, path=path)
                cls._instances[key] = inst
            elif password and inst.password != password:
                # Credentials changed for this login/server — update them so
                # the next reconnect uses the fresh password.
                inst.password = password

        if auto_connect and not inst.connected:
            inst.connect()
        return inst

    # ==========================================================
    # CONNECT
    # ==========================================================

    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            log.error("MetaTrader5 package not installed")
            return False

        for attempt in range(1, self.MAX_RETRIES + 1):

            if self._try_connect():
                return True

            log.warning(
                f"[MT5Connection] Attempt "
                f"{attempt}/{self.MAX_RETRIES} failed "
                f"retrying in {self.RETRY_DELAY_SEC}s"
            )

            time.sleep(self.RETRY_DELAY_SEC)

        log.error("[MT5Connection] Connection failed")
        return False

    def _try_connect(self) -> bool:
        try:
            with self.MT5_LOCK:
                mt5.shutdown()
                time.sleep(1)

                init_kwargs = {}

                if self.path:
                    init_kwargs["path"] = self.path

                if not mt5.initialize(**init_kwargs):
                    err = mt5.last_error()
                    log.error(
                        f"[MT5Connection] initialize failed: {err}"
                    )
                    return False

                authorized = mt5.login(
                    self.login,
                    password=self.password,
                    server=self.server
                )

                if not authorized:
                    err = mt5.last_error()
                    log.error(
                        f"[MT5Connection] Login failed: {err}"
                    )
                    mt5.shutdown()
                    return False

            self.connected = True
            self.connected_at = datetime.utcnow()
            self.last_ping = datetime.utcnow()
            self._consecutive_failures = 0

            self._print_connected_banner()
            return True

        except Exception as e:
            log.exception(
                f"[MT5Connection] Connect exception: {e}"
            )
            return False

    # ==========================================================
    # DISCONNECT
    # ==========================================================

    def disconnect(self):
        try:
            with self.MT5_LOCK:
                if MT5_AVAILABLE:
                    mt5.shutdown()
        except Exception as e:
            log.warning(f"[MT5Connection] Suppressed exception in disconnect(): {e}")
            pass

        self.connected = False
        self.connected_at = None
        self.last_ping = None

        log.info("[MT5Connection] Disconnected")

    # ==========================================================
    # HEALTH CHECK
    # ==========================================================

    def is_alive(self) -> bool:
        """Day 90+ hotfix: lenient health check.

        Old behavior: grabbed the lock, called terminal_info() AND
        account_info() back-to-back, and immediately marked the
        connection dead if either returned None — even on a single
        transient hiccup. With several AITrader instances sharing one
        MT5 terminal via MT5_LOCK, that produced frequent false-positive
        "disconnect" events (visible in logs every 5-10 min) even though
        the terminal was actually fine moments later.

        New behavior:
          - Only call terminal_info() (account_info() is checked
            separately, on demand, by get_account_info() — no need to
            pay for both calls on every health check).
          - On a None result, retry in-place up to
            HEALTH_CHECK_RETRIES times with a short delay before
            declaring the connection dead.
          - Track consecutive failures for visibility/debugging.

        Round-5 audit fix: cache the result for HEALTH_CHECK_CACHE_SEC
        (default 5s). When the cache holds a True result and the
        5s window hasn't expired, skip the terminal_info() call
        entirely — this avoids hammering MT5 with health checks on
        every single tick fetch across 6 pairs. If the cached result
        is False (or the cache is stale), the full check runs.
        """
        if not MT5_AVAILABLE:
            return False

        # ── Round-5: cache check ──────────────────────────────────
        import time as _time
        now = _time.time()
        cache_age = now - self._last_health_check_ts
        if (
            self._last_health_check_result  # only cache positive results
            and cache_age < self.HEALTH_CHECK_CACHE_SEC
        ):
            # Cache hit — skip the terminal_info() call entirely.
            # This is the fast path for the common case where the
            # connection is healthy and 6 pairs are all calling
            # is_alive() within the same 5s window.
            return True

        attempts = 1 + self.HEALTH_CHECK_RETRIES

        for attempt in range(1, attempts + 1):
            try:
                with self.MT5_LOCK:
                    terminal = mt5.terminal_info()

                if terminal is not None:
                    self.last_ping = datetime.utcnow()
                    self._consecutive_failures = 0

                    # P1 fix (audit §4.1): periodic auth-level check.
                    # terminal_info() alone can't detect a broker-side
                    # session invalidation (re-login elsewhere, token
                    # expiry) — only account_info() does. Running it on
                    # every health check reintroduces the Day 90 flapping
                    # problem, so it's gated to every Nth check instead.
                    self._health_check_count += 1
                    if self._health_check_count % self.AUTH_CHECK_EVERY_N_HEALTH_CHECKS == 0:
                        try:
                            with self.MT5_LOCK:
                                account = mt5.account_info()
                            if account is None:
                                log.warning(
                                    "[MT5Connection] Periodic auth check failed — "
                                    "terminal is up but account_info() is None "
                                    "(session may be invalidated server-side)"
                                )
                                self.connected = False
                                # Round-5: cache negative result too (short TTL)
                                self._last_health_check_ts = _time.time()
                                self._last_health_check_result = False
                                return False
                        except Exception as e:
                            log.warning(f"[MT5Connection] Periodic auth check error: {e}")

                    # Round-5: cache the positive result
                    self._last_health_check_ts = _time.time()
                    self._last_health_check_result = True
                    return True

                if attempt < attempts:
                    time.sleep(self.HEALTH_CHECK_RETRY_DELAY)

            except Exception as e:
                log.warning(
                    f"[MT5Connection] Health check error "
                    f"(attempt {attempt}/{attempts}): {e}"
                )
                if attempt < attempts:
                    time.sleep(self.HEALTH_CHECK_RETRY_DELAY)

        # All attempts exhausted — genuinely consider it down
        self._consecutive_failures += 1
        self.connected = False
        # Round-5: do NOT cache negative results for the full TTL —
        # we want the NEXT is_alive() call to retry immediately so
        # reconnect logic fires promptly. Just record the timestamp
        # for diagnostics; the cache-hit guard above only fires on
        # positive results.
        self._last_health_check_ts = _time.time()
        self._last_health_check_result = False
        log.warning(
            f"[MT5Connection] Health check failed after {attempts} "
            f"attempts (consecutive_failures={self._consecutive_failures})"
        )
        return False

    # ==========================================================
    # ACCOUNT INFO
    # ==========================================================

    def get_account_info(self):
        if not self._require_connected():
            return None

        try:
            with self.MT5_LOCK:
                account = mt5.account_info()

            if account is None:
                log.error(
                    f"account_info failed: {mt5.last_error()}"
                )
                return None

            return {
                "login": account.login,
                "balance": account.balance,
                "equity": account.equity,
                "margin": account.margin,
                "free_margin": account.margin_free,
                "margin_level": account.margin_level,
                "currency": account.currency,
                "leverage": account.leverage,
                "server": account.server,
                "trade_allowed": account.trade_allowed,
            }

        except Exception as e:
            log.exception(
                f"[MT5Connection] account info error: {e}"
            )
            return None

    # ==========================================================
    # INTERNAL
    # ==========================================================

    def _require_connected(self):
        if not self.connected:
            # Bug fix: try auto-reconnect if previously connected
            if self.login and self.password and self.server:
                log.info("[MT5Connection] Not connected — attempting auto-reconnect...")
                return self.reconnect()
            return False

        if not self.is_alive():
            self.connected = False
            # Bug fix: auto-reconnect on health check failure
            log.info("[MT5Connection] Connection lost — attempting auto-reconnect...")
            return self.reconnect()

        return True

    def reconnect(self) -> bool:
        """Reconnect to MT5 after disconnection (Bug fix: auto-reconnect)."""
        try:
            self.disconnect()
            time.sleep(2)  # brief pause before retry
            success = self.connect()
            if success:
                log.info("[MT5Connection] Auto-reconnect SUCCESS")
            else:
                log.error("[MT5Connection] Auto-reconnect FAILED")
            return success
        except Exception as e:
            log.error(f"[MT5Connection] Reconnect exception: {e}")
            return False

    # ==========================================================
    # TICK SAFE
    # ==========================================================

    def get_tick(self, symbol):
        if not self._require_connected():
            return None

        try:
            with self.MT5_LOCK:
                tick = mt5.symbol_info_tick(symbol)

            if tick is None:
                log.warning(
                    f"[MT5Connection] No tick for {symbol}"
                )
                return None

            return tick

        except Exception as e:
            log.exception(
                f"[MT5Connection] Tick error: {e}"
            )
            return None

    # ==========================================================
    # POSITIONS
    # ==========================================================

    def positions_get(self, **kwargs):
        """Thread-safe wrapper around mt5.positions_get().

        Day 102: added so consumers (e.g. AITrader._get_live_open_pairs)
        can query positions through the shared connection instead of
        calling mt5.initialize()/shutdown() independently and killing
        the shared session.
        """
        if not self._require_connected():
            return None

        try:
            with self.MT5_LOCK:
                return mt5.positions_get(**kwargs)
        except Exception as e:
            log.exception(f"[MT5Connection] positions_get error: {e}")
            return None

    # ==========================================================
    # CANDLES / SYMBOLS  (P1 fix — audit §3.1)
    # ==========================================================
    # Added so DataFetcher._fetch_mt5() no longer needs to call
    # mt5.initialize()/mt5.symbol_select()/mt5.copy_rates_from_pos()
    # directly against the global module. Every MT5 touch — data or
    # execution — now goes through this one locked, session-owning
    # object.

    def ensure_connected(self) -> bool:
        """Public wrapper around _require_connected() for external callers
        (e.g. DataFetcher) that need to guarantee a live session without
        calling mt5.initialize() themselves."""
        return self._require_connected()

    def symbol_select(self, symbol: str, enable: bool = True):
        """Thread-safe wrapper around mt5.symbol_select()."""
        if not self._require_connected():
            return False
        try:
            with self.MT5_LOCK:
                return mt5.symbol_select(symbol, enable)
        except Exception as e:
            log.exception(f"[MT5Connection] symbol_select error: {e}")
            return False

    def copy_rates_from_pos(self, symbol: str, timeframe, start_pos: int, count: int):
        """Thread-safe wrapper around mt5.copy_rates_from_pos()."""
        if not self._require_connected():
            return None
        try:
            with self.MT5_LOCK:
                return mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
        except Exception as e:
            log.exception(f"[MT5Connection] copy_rates_from_pos error: {e}")
            return None

    # ==========================================================
    # RECONNECT
    # ==========================================================
    # Day 102+ CRITICAL hotfix: duplicate reconnect() removed.
    #
    # Previously there were TWO `def reconnect` definitions in this
    # class — lines 264-277 (with try/except + bool return + logging)
    # and lines 331-340 (a dumber version: no error handling, returns
    # connect() result directly, no log on failure). Python silently
    # kept the SECOND definition, so the safer version was dead code.
    # Any caller of reconnect() — including _require_connected() at
    # lines 253/260 — would hit the dumb version, which would propagate
    # raw exceptions from connect() instead of returning False on
    # failure. This silently killed MT5 sessions on transient errors.
    #
    # Fix: removed the second definition. The first (safer) version
    # at lines 264-277 is now the only one. No behavioral change for
    # callers — they still get a bool back.

    # ==========================================================
    # BANNER
    # ==========================================================

    def _print_connected_banner(self):
        try:
            with self.MT5_LOCK:
                account = mt5.account_info()

            bar = "═" * 44

            log.info(bar)
            log.info(
                "  🤖  AI TRADER — MT5 CONNECTION"
            )
            log.info(bar)
            log.info(
                f"  Connected : {self.server}"
            )
            log.info(
                f"  Account   : {self.login}"
            )
            log.info(
                "  Status    : ✅ Ready"
            )

            if account:
                log.info(
                    f"  Balance   : ${account.balance:.2f}"
                )

            log.info(bar)

        except Exception as e:
            log.warning(f"[MT5Connection] Suppressed exception in _print_connected_banner(): {e}")
            pass