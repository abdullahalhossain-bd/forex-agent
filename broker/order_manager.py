# broker/order_manager.py  —  Day 33 | MT5 Order Execution Engine
# ============================================================
# AI এখন শুধু "BUY EURUSD" বলে না — এই module আসলে MT5 demo
# account-এ order পাঠায়। ৭টা function (doc অনুযায়ী) + ৩টা
# bonus safety layer (pre-trade validation, retry, confirmation)।
#
# Execution Logger ও Paper/Demo router আলাদা module-এ আছে
# (broker/journal_bridge.py এবং execution/execution_router.py) —
# duplicate করা হয়নি, এখানে শুধু order placement logic।
# ============================================================

import time
from datetime import datetime, timezone
from utils.logger import get_logger
from broker.mt5_connection import MT5_AVAILABLE

log = get_logger("order_manager")

if MT5_AVAILABLE:
    import MetaTrader5 as mt5

# retcode গুলোর human-readable meaning — confirmation check-ের জন্য
RETCODE_SUCCESS = {10008, 10009}   # TRADE_RETCODE_PLACED, TRADE_RETCODE_DONE

# Day 99+ FIX (Issue #4): retcodes that indicate the order was placed
# but is still PENDING execution on the broker side. These are not
# failures — the broker accepted the order — but the position may not
# appear in mt5.positions_get() for a few hundred milliseconds. Without
# a confirmation poll, the caller (ExecutionRouter) could log
# "PENDING_EXECUTOR" and the order status would hang indefinitely
# (router waits for confirmation, broker has already filled, neither
# side polls the other).
RETCODE_PENDING = {10008}  # TRADE_RETCODE_PLACED — order accepted but not filled yet

# How long to poll for the position to appear in mt5.positions_get()
# after a successful order_send. 2 seconds is plenty — MetaQuotes
# documents sub-second latency in normal conditions, but demo servers
# under load can take longer.
POSITION_CONFIRM_TIMEOUT_SEC = 2.0
POSITION_CONFIRM_POLL_INTERVAL_SEC = 0.2


def _confirm_position_appeared(
    broker_symbol: str,
    ticket: int,
    timeout: float = POSITION_CONFIRM_TIMEOUT_SEC,
    poll_interval: float = POSITION_CONFIRM_POLL_INTERVAL_SEC,
) -> bool:
    """Day 99+ FIX (Issue #4): poll mt5.positions_get() until the
    freshly-placed ticket appears, or until `timeout` seconds elapse.

    This closes the gap between `order_send()` returning retcode=10009
    (DONE) and the position actually being queryable via positions_get.
    Without this poll, the ExecutionRouter could see the order succeed
    but then immediately try to query/modify the position and find
    nothing — leading to the "PENDING_EXECUTOR" hang the operator
    reported.

    Returns:
        True if the position was confirmed (ticket appears in
        positions_get for this symbol).
        False if the position never appeared within the timeout.
    """
    import time as _time
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        try:
            positions = _mt5_positions_get(symbol=broker_symbol)
            if positions:
                for p in positions:
                    # ticket may be the order ticket or the deal ticket;
                    # MT5 exposes both via .ticket on the position object.
                    if getattr(p, "ticket", None) == ticket or \
                       getattr(p, "identifier", None) == ticket:
                        return True
        except Exception:
            pass
        _time.sleep(poll_interval)
    return False


def _mt5_positions_get(retries: int = 2, delay: float = 0.3, **kwargs):
    """Call mt5.positions_get() with retry logic.
    
    MT5 can return None intermittently. This helper retries
    a few times before giving up, reducing false negatives.
    """
    import MetaTrader5 as mt5_lib
    for attempt in range(retries + 1):
        try:
            result = mt5_lib.positions_get(**kwargs) if kwargs else mt5_lib.positions_get()
            if result is not None:
                return result
        except Exception:
            pass
        if attempt < retries:
            time.sleep(delay)
    return None


def _resolve_filling_mode(broker_symbol: str):
    """Pick the most permissive filling mode the broker supports.

    Different brokers accept different `type_filling` values. The
    MetaQuotes-Demo server, ICMarkets, and many others reject
    ORDER_FILLING_FOK outright with retcode=10030 ("Unsupported filling
    mode").  We probe `mt5.symbol_info(symbol).filling_mode` (a bitmask)
    and pick the first supported mode in this priority order:

        1. ORDER_FILLING_IOC  (Immediate-or-Cancel — most permissive, supported by almost all brokers)
        2. ORDER_FILLING_FOK  (Fill-or-Kill — stricter, some brokers reject)
        3. ORDER_FILLING_RETURN (Return — used by some ECN brokers)

    Falls back to IOC if the probe fails — IOC works on >95% of brokers.
    """
    if not MT5_AVAILABLE:
        return None
    try:
        info = mt5.symbol_info(broker_symbol)
        if info is None:
            return mt5.ORDER_FILLING_IOC
        mode = info.filling_mode  # bitmask: bit0=FOK, bit1=IOC, bit2=RETURN
        if mode & 2:  # IOC supported
            return mt5.ORDER_FILLING_IOC
        if mode & 1:  # FOK supported
            return mt5.ORDER_FILLING_FOK
        if mode & 4:  # RETURN supported
            return mt5.ORDER_FILLING_RETURN
        return mt5.ORDER_FILLING_IOC  # safest default
    except Exception as e:
        # Day 81+ hotfix: was silent `return mt5.ORDER_FILLING_IOC`.
        # If symbol_info raises (disconnect, symbol not in Market Watch),
        # silently defaulting to IOC meant the next order_send would fail
        # with retcode=10030 and the operator had no idea why.  Now we
        # log the exception so the root cause is visible.
        log.warning(
            f"[OrderManager] _resolve_filling_mode({broker_symbol}) raised: {e} "
            f"— falling back to ORDER_FILLING_IOC"
        )
        try:
            from core.execution_logger import log_broker_last_error
            log_broker_last_error(symbol=broker_symbol, error=e,
                                  stage="resolve_filling_mode")
        except Exception as e:
            log.warning(f"Suppressed exception at line 69: {e}")
            pass
        return mt5.ORDER_FILLING_IOC


class OrderManager:
    """
    MT5-এ actual order পাঠায়, modify করে, close করে।

    Usage:
        om = OrderManager(connection, account_manager)
        result = om.place_market_order("EURUSD", "BUY", lot=0.01, sl=1.0825, tp=1.0900)
        if result["success"]:
            ticket = result["ticket"]
            om.modify_order(ticket, new_sl=1.0855)
            ...
            om.close_order(ticket)
    """

    MAX_RETRIES = 3
    RETRY_DELAY_SEC = 2
    MAX_LOT = 10.0   # sanity ceiling — risk engine আগেই size করে, এটা শুধু hard backstop

    def __init__(self, connection, account_manager):
        self.connection = connection
        self.account_manager = account_manager

    # ─────────────────────────────────────────────
    # FUNCTION 1 — MARKET ORDER
    # ─────────────────────────────────────────────

    def place_market_order(
        self, symbol: str, direction: str, lot: float, sl: float = None, tp: float = None,
        comment: str = "ai_trader",
    ) -> dict:
        """BUY/SELL instantly বর্তমান market price-এ। Pre-trade validation + retry সহ।

        CRITICAL FIX: Duplicate order prevention.
        Before each retry, check if a position was already opened by a
        previous attempt (broker filled but response was lost). This prevents
        double-position risk when network/timeout causes retry.
        """
        validation = self._pre_trade_validate(symbol, direction, lot, sl, tp)
        if not validation["ok"]:
            log.warning(f"[OrderManager] Pre-trade validation failed: {validation['reason']}")
            return {"success": False, "reason": validation["reason"]}

        broker_symbol = validation["broker_symbol"]

        # ── Day 99+ V3 FIX (Master List Issue #4 — Precision bug) ──
        # Use the precision-normalized values returned by _pre_trade_validate.
        # These have SL/TP rounded to symbol.digits and lot snapped to
        # symbol.volume_step, preventing retcode=10015 (INVALID_PRICE) and
        # 10014 (INVALID_VOLUME) rejections.
        lot = validation.get("lot_normalized", lot)
        sl = validation.get("sl_normalized", sl)
        tp = validation.get("tp_normalized", tp)
        digits = validation.get("digits", 5)

        # RED TEAM FIX: Spread check before placing any order.
        # If spread is abnormally wide (e.g., during news), reject the order.
        # This prevents entering at terrible prices during volatility spikes.
        try:
            tick = mt5.symbol_info_tick(broker_symbol)
            if tick:
                # Compute pip size from symbol digits
                info = mt5.symbol_info(broker_symbol)
                if info:
                    # Day 102+ CRITICAL hotfix: include 5 in the digit test.
                    # Previously: `10 if info.digits == 3 else 1` — missed
                    # the 5-digit case (EURUSD/GBPUSD/AUDUSD/etc. all have
                    # digits=5). For those pairs, info.point is a pipette
                    # (0.00001), so a normal 1.5-pip spread computed as
                    # spread_pips=15 — exceeding the >10.0 threshold at
                    # line 152, silently rejecting virtually every FX order.
                    pip_size = info.point * (10 if info.digits in (3, 5) else 1)
                else:
                    pip_size = 0.0001  # fallback for 5-digit pairs
                spread_pips = (tick.ask - tick.bid) / pip_size
                # Reject if spread > 10 pips (abnormally wide)
                if spread_pips > 10.0:
                    log.warning(
                        f"[OrderManager] SPREAD REJECTED: {symbol} spread={spread_pips:.1f} pips "
                        f"(>10 pips threshold) — likely news/volatility. Order rejected."
                    )
                    return {"success": False, "reason": f"Spread too wide ({spread_pips:.1f} pips)"}
        except Exception as e:
            log.warning(f"[OrderManager] Spread check failed (proceeding): {e}")

        # PREMORTEM FIX: Margin check before placing order.
        # If free margin is too low, the order will be rejected by broker
        # anyway — but by then we've already wasted time. Check upfront.
        try:
            account_info = mt5.account_info()
            if account_info:
                free_margin = account_info.margin_free
                # Estimate required margin (rough: lot * 1000 for 1:100 leverage)
                estimated_margin = lot * 1000
                if free_margin < estimated_margin * 2:  # 2x safety buffer
                    log.warning(
                        f"[OrderManager] MARGIN REJECTED: free_margin=${free_margin:.0f} "
                        f"< 2x estimated ${estimated_margin:.0f} for {lot} lots {symbol}"
                    )
                    return {"success": False, "reason": f"Insufficient free margin (${free_margin:.0f})"}
        except Exception as e:
            log.warning(f"[OrderManager] Margin check failed (proceeding): {e}")

        # Record positions BEFORE we start, so we can detect new ones
        try:
            pre_positions = _mt5_positions_get(symbol=broker_symbol) or []
            pre_tickets = {p.ticket for p in pre_positions}
        except Exception as e:
            # P1 fix: was `pre_tickets = set()` which silently disabled
            # duplicate-order detection. Now fail-closed: abort the order
            # rather than risk a double-position on retry.
            log.error(f"[OrderManager] pre_positions fetch failed: {e}; aborting order "
                      f"to prevent potential double-position")
            return {
                "success": False,
                "reason": f"pre_positions fetch failed: {e}",
                "retcode": -1,
                "ticket": None,
            }

        for attempt in range(1, self.MAX_RETRIES + 1):
            # DUPLICATE ORDER PREVENTION: before retrying, check if the
            # previous attempt actually filled (broker filled but we
            # didn't get the response due to timeout/network)
            if attempt > 1:
                try:
                    current_positions = _mt5_positions_get(symbol=broker_symbol) or []
                    new_positions = [p for p in current_positions
                                     if p.ticket not in pre_tickets]
                    if new_positions:
                        # A position appeared that wasn't there before —
                        # our previous order DID fill! Don't retry.
                        log.warning(
                            f"[OrderManager] DUPLICATE PREVENTED: {len(new_positions)} "
                            f"new position(s) appeared on retry attempt {attempt} — "
                            f"previous order likely filled. Tickets: "
                            f"{[p.ticket for p in new_positions]}"
                        )
                        pos = new_positions[0]
                        return {
                            "success": True,
                            "ticket": pos.ticket,
                            "price": pos.price_open,
                            "reason": f"Filled on previous attempt (detected on retry {attempt})",
                            "duplicate_prevented": True,
                        }
                except Exception as e:
                    log.warning(f"[OrderManager] Duplicate check failed: {e}")

            tick = mt5.symbol_info_tick(broker_symbol)
            if tick is None:
                self._wait_retry(attempt, "no tick data")
                continue

            # BLACK SWAN FIX: Tick sanity validation.
            # MT5 can return garbage ticks during:
            # - Broker connectivity issues (ask=0, bid=0)
            # - Symbol delisting (tick exists but price=0)
            # - Data feed corruption (ask < bid = impossible)
            # - Extreme volatility (spread > 100 pips = likely error)
            try:
                ask = float(tick.ask)
                bid = float(tick.bid)
                if ask <= 0 or bid <= 0:
                    log.warning(f"[OrderManager] GARBAGE TICK: {symbol} ask={ask} bid={bid} — skipping")
                    self._wait_retry(attempt, "garbage tick (zero/negative price)")
                    continue
                if ask < bid:
                    log.warning(f"[OrderManager] INVERTED TICK: {symbol} ask={ask} < bid={bid} — skipping")
                    self._wait_retry(attempt, "inverted tick (ask < bid)")
                    continue
                # Reject if spread > 100 pips (data error, not real market)
                info = mt5.symbol_info(broker_symbol)
                if info:
                    # Day 102+ CRITICAL hotfix: same fix as line 154 — include
                    # 5-digit pairs (EURUSD/GBPUSD/etc.) so pip_size isn't a
                    # pipette and spread_check isn't 10× too large.
                    pip_size = info.point * (10 if info.digits in (3, 5) else 1)
                    spread_check = (ask - bid) / pip_size
                    if spread_check > 100:
                        log.warning(f"[OrderManager] ABNORMAL TICK: {symbol} spread={spread_check:.0f} pips — likely data error, skipping")
                        self._wait_retry(attempt, f"abnormal spread {spread_check:.0f} pips")
                        continue
            except Exception as e:
                log.warning(f"[OrderManager] Tick validation failed: {e}")

            price = tick.ask if direction == "BUY" else tick.bid
            # Day 99+ V3 FIX (Master List Issue #4 — Precision bug): round
            # the price to the symbol's declared digits. MT5 rejects orders
            # with retcode=10015 (INVALID_PRICE) if the price has more
            # decimal places than the symbol allows (e.g. tick.ask =
            # 1.0825478 on EURUSD with digits=5 → must become 1.08255).
            try:
                price = round(float(price), int(digits))
            except (TypeError, ValueError):
                pass  # leave price as-is if rounding fails
            order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL

            # Auto-detect the broker's supported filling mode — this is the
            # #1 cause of "Unsupported filling mode" (retcode 10030) rejections
            # on demo accounts (MetaQuotes-Demo, ICMarkets, etc.).
            filling_mode = _resolve_filling_mode(broker_symbol)

            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       broker_symbol,
                "volume":       lot,
                "type":         order_type,
                "price":        price,
                "sl":           sl or 0.0,
                "tp":           tp or 0.0,
                "deviation":    10,         # max acceptable slippage (points)
                "magic":        424242,
                "comment":      comment,
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }

            result = mt5.order_send(request)
            # BLACK SWAN FIX: Validate API response before accessing attributes
            if result is None or not hasattr(result, 'retcode'):
                log.warning(f"[OrderManager] order_send returned invalid response — MT5 API issue")
                self._wait_retry(attempt, "invalid API response")
                continue
            outcome = self._check_confirmation(result, attempt, requested_volume=lot, symbol=broker_symbol)
            if outcome["success"]:
                log.info(
                    f"[OrderManager] ✅ ORDER FILLED — {direction} {broker_symbol} "
                    f"lot={lot} ticket={outcome['ticket']}"
                )

                # Day 99+ FIX (Issue #4): confirm the position actually
                # appears in mt5.positions_get() before returning success.
                # Without this poll, the broker can return retcode=10009
                # (DONE) but the position isn't queryable for ~200-500ms
                # afterward — ExecutionRouter would then log
                # "PENDING_EXECUTOR" and hang indefinitely. Polling for
                # up to 2s guarantees the position is visible to the
                # rest of the pipeline before we declare success.
                _ticket = outcome.get("ticket")
                if _ticket is not None:
                    try:
                        confirmed = _confirm_position_appeared(
                            broker_symbol=broker_symbol,
                            ticket=_ticket,
                        )
                        if confirmed:
                            log.debug(
                                f"[OrderManager] position confirmed in "
                                f"positions_get (ticket={_ticket})"
                            )
                        else:
                            # Not fatal — the order DID succeed (retcode
                            # 10009). Position may simply be slow to
                            # propagate on this broker, or the broker
                            # closes the position immediately (e.g. on
                            # certain CFD symbols). Log a warning so the
                            # operator knows PositionManager may not see
                            # it on the first query.
                            log.warning(
                                f"[OrderManager] position ticket={_ticket} "
                                f"did not appear in positions_get within "
                                f"{POSITION_CONFIRM_TIMEOUT_SEC:.1f}s — "
                                f"order succeeded but position tracking "
                                f"may lag. Watch for orphan positions."
                            )
                            outcome["position_confirm_lag"] = True
                    except Exception as e:
                        log.warning(
                            f"[OrderManager] position confirm poll raised: {e} "
                            f"— order still considered successful (retcode ok)"
                        )

                # EX-1 fix: on partial fill, try ONCE to fill the remainder
                # with a fresh order rather than silently under-reporting
                # exposure. We deliberately do NOT loop this through the
                # main retry loop (which has its own duplicate-order
                # detection tuned for "did my last attempt actually fill"),
                # since here we already KNOW attempt 1 filled — we just need
                # top-up volume, tracked as a separate ticket.
                if outcome.get("partial_fill"):
                    outcome = self._attempt_fill_remainder(
                        broker_symbol=broker_symbol, direction=direction,
                        sl=sl, tp=tp, comment=comment,
                        first_outcome=outcome,
                    )
                # Day 97+ Book Page 11: Execution quality monitoring
                try:
                    from monitoring.execution_quality import get_execution_monitor
                    eqm = get_execution_monitor()
                    eqm.record_order(
                        symbol=broker_symbol, direction=direction,
                        expected_entry=request.get("price", 0),
                        actual_entry=getattr(result, "price", 0),
                        lot=lot, success=True,
                        ticket=outcome.get("ticket"),
                    )
                except Exception as e:
                    log.warning(f"Suppressed exception at line 201: {e}")
                    pass
                return outcome

            if not outcome.get("retryable", True):
                return outcome   # permanent rejection (যেমন invalid lot) — retry করার মানে নেই

            self._wait_retry(attempt, outcome["reason"])

        log.error(f"[OrderManager] ⛔ Order failed after {self.MAX_RETRIES} retries — {symbol} {direction}")
        return {"success": False, "reason": f"Failed after {self.MAX_RETRIES} retries"}

    # ─────────────────────────────────────────────
    # FUNCTION 2 — LIMIT ORDER
    # ─────────────────────────────────────────────

    def place_limit_order(
        self, symbol: str, price: float, direction: str, lot: float,
        sl: float = None, tp: float = None, comment: str = "ai_trader_limit",
    ) -> dict:
        """Pullback/support/breakout-retest entry-র জন্য — future price-এ pending order।"""
        validation = self._pre_trade_validate(symbol, direction, lot, sl, tp)
        if not validation["ok"]:
            return {"success": False, "reason": validation["reason"]}

        broker_symbol = validation["broker_symbol"]
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT

        # Use broker-supported filling mode (auto-detected).
        filling_mode = _resolve_filling_mode(broker_symbol)

        request = {
            "action":       mt5.TRADE_ACTION_PENDING,
            "symbol":       broker_symbol,
            "volume":       lot,
            "type":         order_type,
            "price":        price,
            "sl":           sl or 0.0,
            "tp":           tp or 0.0,
            "magic":        424242,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        for attempt in range(1, self.MAX_RETRIES + 1):
            result = mt5.order_send(request)
            # BLACK SWAN FIX: Validate API response before accessing attributes
            if result is None or not hasattr(result, 'retcode'):
                log.warning(f"[OrderManager] order_send returned invalid response — MT5 API issue")
                self._wait_retry(attempt, "invalid API response")
                continue
            outcome = self._check_confirmation(result, attempt)
            if outcome["success"]:
                log.info(f"[OrderManager] ✅ LIMIT ORDER PLACED — {direction} {broker_symbol} @ {price}")
                return outcome
            if not outcome.get("retryable", True):
                return outcome
            self._wait_retry(attempt, outcome["reason"])

        return {"success": False, "reason": f"Limit order failed after {self.MAX_RETRIES} retries"}

    # ─────────────────────────────────────────────
    # FUNCTION 2b — PLACE STOP ORDER (Buy Stop / Sell Stop)
    # MT5 User Guide Page 15 — breakout entry via pending stop orders
    # ─────────────────────────────────────────────

    def place_stop_order(
        self, symbol: str, price: float, direction: str, lot: float,
        sl: float = None, tp: float = None, comment: str = "ai_trader_stop",
    ) -> dict:
        """
        MT5 User Guide Page 15 — Buy Stop / Sell Stop pending order.

        Buy Stop : price ABOVE market → buy on breakout (resistance break)
        Sell Stop: price BELOW market → sell on breakdown (support break)

        Book rules:
          - Price target ABOVE market + intent to BUY (breakout) → Buy Stop
          - Price target BELOW market + intent to SELL (breakdown) → Sell Stop
        """
        validation = self._pre_trade_validate(symbol, direction, lot, sl, tp)
        if not validation["ok"]:
            return {"success": False, "reason": validation["reason"]}

        broker_symbol = validation["broker_symbol"]

        # Determine stop order type based on direction
        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY_STOP
            # Buy Stop: price must be ABOVE current market
            tick = mt5.symbol_info_tick(broker_symbol)
            if tick and price <= tick.ask:
                return {"success": False,
                        "reason": f"Buy Stop price {price} must be ABOVE current ask {tick.ask}"}
        else:  # SELL
            order_type = mt5.ORDER_TYPE_SELL_STOP
            # Sell Stop: price must be BELOW current market
            tick = mt5.symbol_info_tick(broker_symbol)
            if tick and price >= tick.bid:
                return {"success": False,
                        "reason": f"Sell Stop price {price} must be BELOW current bid {tick.bid}"}

        filling_mode = _resolve_filling_mode(broker_symbol)

        request = {
            "action":       mt5.TRADE_ACTION_PENDING,
            "symbol":       broker_symbol,
            "volume":       lot,
            "type":         order_type,
            "price":        price,
            "sl":           sl or 0.0,
            "tp":           tp or 0.0,
            "magic":        424242,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        for attempt in range(1, self.MAX_RETRIES + 1):
            result = mt5.order_send(request)
            outcome = self._check_confirmation(result, attempt=attempt)
            if outcome.get("success"):
                return outcome
            if attempt < self.MAX_RETRIES:
                import time
                time.sleep(self.RETRY_DELAY_SEC)

        return {"success": False, "reason": f"Stop order failed after {self.MAX_RETRIES} retries"}

    # ─────────────────────────────────────────────
    # FUNCTION 2c — PLACE STOP LIMIT ORDER (Buy Stop Limit / Sell Stop Limit)
    # MT5 User Guide Page 15 — advanced pending order type
    # ─────────────────────────────────────────────

    def place_stop_limit_order(
        self, symbol: str, stop_price: float, limit_price: float,
        direction: str, lot: float,
        sl: float = None, tp: float = None, comment: str = "ai_trader_stoplimit",
    ) -> dict:
        """
        MT5 User Guide Page 15 — Buy Stop Limit / Sell Stop Limit.

        Buy Stop Limit : when price rises to stop_price, a Buy Limit order
                         is placed at limit_price (below stop_price).
        Sell Stop Limit: when price falls to stop_price, a Sell Limit order
                         is placed at limit_price (above stop_price).
        """
        validation = self._pre_trade_validate(symbol, direction, lot, sl, tp)
        if not validation["ok"]:
            return {"success": False, "reason": validation["reason"]}

        broker_symbol = validation["broker_symbol"]

        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY_STOP_LIMIT
        else:
            order_type = mt5.ORDER_TYPE_SELL_STOP_LIMIT

        filling_mode = _resolve_filling_mode(broker_symbol)

        request = {
            "action":       mt5.TRADE_ACTION_PENDING,
            "symbol":       broker_symbol,
            "volume":       lot,
            "type":         order_type,
            "price":        stop_price,       # trigger price
            "stoplimit":    limit_price,      # limit order price after trigger
            "sl":           sl or 0.0,
            "tp":           tp or 0.0,
            "magic":        424242,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        for attempt in range(1, self.MAX_RETRIES + 1):
            result = mt5.order_send(request)
            outcome = self._check_confirmation(result, attempt=attempt)
            if outcome.get("success"):
                return outcome
            if attempt < self.MAX_RETRIES:
                import time
                time.sleep(self.RETRY_DELAY_SEC)

        return {"success": False, "reason": f"Stop-Limit order failed after {self.MAX_RETRIES} retries"}

    # ─────────────────────────────────────────────
    # FUNCTION 3 — MODIFY ORDER  (SL/TP move, break-even, trailing)
    # ─────────────────────────────────────────────

    def modify_order(self, ticket: int, new_sl: float = None, new_tp: float = None) -> dict:
        position = self._get_position(ticket)
        if position is None:
            return {"success": False, "reason": f"Position not found: {ticket}"}

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol":   position.symbol,
            "sl":       new_sl if new_sl is not None else position.sl,
            "tp":       new_tp if new_tp is not None else position.tp,
        }

        result = mt5.order_send(request)
        outcome = self._check_confirmation(result, attempt=1)
        if outcome["success"]:
            log.info(f"[OrderManager] SL/TP updated — ticket {ticket} → SL {new_sl} TP {new_tp}")
        return outcome

    # ─────────────────────────────────────────────
    # FUNCTION 4 — CLOSE ORDER
    # ─────────────────────────────────────────────

    def close_order(self, ticket: int, comment: str = "manual_close") -> dict:
        position = self._get_position(ticket)
        if position is None:
            return {"success": False, "reason": f"Position not found: {ticket}"}

        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            return {"success": False, "reason": "No tick data — cannot close"}

        is_buy = position.type == mt5.ORDER_TYPE_BUY
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        price = tick.bid if is_buy else tick.ask

        # Use broker-supported filling mode for close orders too.
        filling_mode = _resolve_filling_mode(position.symbol)

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       position.symbol,
            "volume":       position.volume,
            "type":         close_type,
            "position":     ticket,
            "price":        price,
            "deviation":    10,
            "magic":        424242,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        result = mt5.order_send(request)
        outcome = self._check_confirmation(result, attempt=1)
        if outcome["success"]:
            profit = position.profit
            log.info(f"[OrderManager] ✅ Position closed — ticket {ticket} | Profit: ${profit:.2f}")
            outcome["profit"] = profit
        return outcome

    # ─────────────────────────────────────────────
    # FUNCTION 5 — CLOSE ALL  (kill switch / emergency)
    # ─────────────────────────────────────────────

    def close_all_orders(self, reason: str = "Emergency close") -> list[dict]:
        log.warning(f"[OrderManager] 🚨 EMERGENCY — closing all positions: {reason}")
        positions = self.get_open_positions()
        results = []
        for pos in positions:
            outcome = self.close_order(pos["ticket"], comment=f"emergency:{reason}"[:31])
            results.append(outcome)
        log.warning(f"[OrderManager] {len(results)} positions processed for emergency close")
        return results

    # ─────────────────────────────────────────────
    # FUNCTION 6 — OPEN POSITIONS
    # ─────────────────────────────────────────────

    def get_open_positions(self, symbol: str = None, magic: int = 424242) -> list[dict]:
        """Get open positions from MT5.

        RED TEAM FIX: Filter by magic number to exclude manual trades.
        Without this, the bot would try to manage positions opened
        manually by the trader, causing conflicts.
        """
        if not MT5_AVAILABLE:
            return []
        positions = _mt5_positions_get(symbol=symbol) if symbol else _mt5_positions_get()
        if positions is None:
            return []
        # FIX: Filter by magic number — only manage OUR trades
        return [
            {
                "ticket":   p.ticket,
                "symbol":   p.symbol,
                "type":     "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume":   p.volume,
                "price_open": p.price_open,
                "sl":       p.sl,
                "tp":       p.tp,
                "profit":   p.profit,
                "open_time": datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
                "magic":    p.magic,
            }
            for p in positions
            if p.magic == magic  # Only our trades
        ]

    def print_open_positions(self) -> None:
        positions = self.get_open_positions()
        bar = "═" * 40
        log.info(bar)
        log.info("  📊  OPEN POSITIONS")
        log.info(bar)
        if not positions:
            log.info("  (none)")
        for p in positions:
            icon = "🟢" if p["profit"] >= 0 else "🔴"
            log.info(f"  {icon} {p['symbol']} {p['type']} | Lot {p['volume']} | Profit ${p['profit']:.2f}")
        log.info(bar)

    # ─────────────────────────────────────────────
    # FUNCTION 7 — TRADE HISTORY
    # ─────────────────────────────────────────────

    def get_order_history(self, days_back: int = 7) -> list[dict]:
        if not MT5_AVAILABLE:
            return []
        from datetime import timedelta
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days_back)
        deals = mt5.history_deals_get(start, end)
        if deals is None:
            return []
        return [
            {
                "ticket":      d.ticket,
                "position_id": d.position_id,
                "symbol":      d.symbol,
                "type":        "BUY" if d.type == mt5.DEAL_TYPE_BUY else "SELL",
                "volume":      d.volume,
                "price":       d.price,
                "profit":      d.profit,
                "time":        datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat(),
            }
            for d in deals
        ]

    # ─────────────────────────────────────────────
    # BONUS 1 — PRE-TRADE VALIDATION
    # ─────────────────────────────────────────────

    def _pre_trade_validate(
        self, symbol: str, direction: str, lot: float, sl: float, tp: float
    ) -> dict:
        if not MT5_AVAILABLE or not self.connection.connected:
            return {"ok": False, "reason": "MT5 not connected"}

        if direction not in ("BUY", "SELL"):
            return {"ok": False, "reason": f"Invalid direction: {direction}"}

        if lot <= 0 or lot > self.MAX_LOT:
            return {"ok": False, "reason": f"Invalid lot size: {lot} (max {self.MAX_LOT})"}

        # ── Day 99+ V3 FIX (Master List Issue #4 — Algo Trading check) ──
        # Proactively check that the MT5 terminal's "Algo Trading" button
        # is enabled BEFORE attempting any order_send. Without this, every
        # order fails with retcode=10027 (TRADE_RETCODE_CLIENT_AUTOTRAADING_DISABLED),
        # classified as permanent (non-retryable), and the operator has to
        # discover it from per-order rejection logs. Now we surface it as
        # a clear pre-trade validation failure with an actionable message.
        try:
            term_info = mt5.terminal_info()
            if term_info is not None and not getattr(term_info, "trade_allowed", True):
                return {
                    "ok": False,
                    "reason": (
                        "MT5 terminal 'Algo Trading' button is OFF — "
                        "every order would fail with retcode=10027. "
                        "Enable Algo Trading in the MT5 terminal toolbar "
                        "(the green play button) and retry."
                    ),
                }
        except Exception as e:
            # Don't fail validation if the terminal_info call itself errors
            # — best-effort check only. The reactive 10027 path will still
            # catch it if the check is silently broken.
            log.debug(f"[OrderManager] terminal_info() check skipped: {e}")

        perm = self.account_manager.trading_permission(symbol=symbol, risk_engine_ok=True)
        if not perm["allowed"]:
            return {"ok": False, "reason": f"Trading not permitted: {perm['failed_checks']}"}

        broker_symbol = perm["broker_symbol"]

        # ── Day 99+ V3 FIX (Master List Issue #4 — Market Watch bug) ──
        # Ensure the symbol is in the terminal's Market Watch before
        # calling symbol_info_tick / symbol_info on it. Without this,
        # mt5.symbol_info_tick() returns None for any symbol not in
        # Market Watch, and the order fails with "no tick data" after
        # MAX_RETRIES — the root cause is invisible to the operator.
        try:
            info_check = mt5.symbol_info(broker_symbol)
            if info_check is None:
                return {
                    "ok": False,
                    "reason": (
                        f"symbol_info({broker_symbol}) returned None — "
                        f"symbol not found on this broker. Check the "
                        f"symbol name (e.g. 'XAUUSD' vs 'GOLD') and "
                        f"that your broker offers it."
                    ),
                }
            if not getattr(info_check, "visible", False):
                log.info(
                    f"[OrderManager] Symbol {broker_symbol} not in Market "
                    f"Watch — calling symbol_select({broker_symbol}, True)"
                )
                if not mt5.symbol_select(broker_symbol, True):
                    return {
                        "ok": False,
                        "reason": (
                            f"Failed to add {broker_symbol} to Market Watch "
                            f"via symbol_select() — manual add required in "
                            f"the MT5 terminal (right-click Market Watch → "
                            f"Show All)."
                        ),
                    }
                # Re-fetch info after adding to Market Watch.
                info_check = mt5.symbol_info(broker_symbol)
        except Exception as e:
            log.warning(
                f"[OrderManager] symbol_select/visible check failed: {e} "
                f"— proceeding anyway (best-effort)"
            )
            info_check = None

        # ── Day 99+ V3 FIX (Master List Issue #4 — Precision bug) ──
        # Round SL, TP, lot, and (later) price to the symbol's declared
        # digits / volume_step. Without this, MT5 rejects orders with
        # retcode=10015 (INVALID_PRICE) or 10014 (INVALID_VOLUME) when
        # Python sends more decimal places than the symbol allows
        # (e.g. EURUSD with digits=5 rejecting sl=1.082547).
        # We return the normalized values in the validation result so
        # place_market_order / place_limit_order / etc. can use them
        # directly when building the request dict.
        normalized = self._normalize_order_params(broker_symbol, lot, sl, tp)
        if normalized is None:
            return {
                "ok": False,
                "reason": (
                    f"Failed to normalize order params for {broker_symbol} "
                    f"(lot={lot}, sl={sl}, tp={tp}) — symbol_info may be "
                    f"unavailable."
                ),
            }

        lot_n, sl_n, tp_n, digits, volume_step = normalized

        # Re-validate SL/TP sanity using the normalized values.
        if info_check and sl_n and tp_n:
            tick = mt5.symbol_info_tick(broker_symbol)
            if tick is not None:
                ref_price = tick.ask if direction == "BUY" else tick.bid
                if direction == "BUY" and not (sl_n < ref_price < tp_n):
                    return {"ok": False, "reason": f"Invalid SL/TP for BUY: SL={sl_n} price={ref_price} TP={tp_n}"}
                if direction == "SELL" and not (tp_n < ref_price < sl_n):
                    return {"ok": False, "reason": f"Invalid SL/TP for SELL: TP={tp_n} price={ref_price} SL={sl_n}"}

        return {
            "ok": True,
            "broker_symbol": broker_symbol,
            # Day 99+ V3 FIX: pass normalized values back to the caller
            # so the request dict uses precision-correct numbers.
            "lot_normalized": lot_n,
            "sl_normalized": sl_n,
            "tp_normalized": tp_n,
            "digits": digits,
            "volume_step": volume_step,
        }

    @staticmethod
    def _normalize_order_params(
        broker_symbol: str,
        lot: float,
        sl: float,
        tp: float,
    ):
        """Day 99+ V3 FIX (Master List Issue #4 — Precision bug 10015/10014).

        Round SL/TP to the symbol's declared `digits` precision and
        snap lot to the symbol's `volume_step` (clamped to volume_min /
        volume_max). Without this, MT5 rejects orders with
        TRADE_RETCODE_INVALID_PRICE (10015) or TRADE_RETCODE_INVALID_VOLUME
        (10014) when Python sends more decimal places than the symbol
        allows.

        Returns:
            (lot, sl, tp, digits, volume_step) tuple, or None if
            symbol_info() failed.
        """
        try:
            info = mt5.symbol_info(broker_symbol)
            if info is None:
                return None
            digits = int(getattr(info, "digits", 5))
            volume_step = float(getattr(info, "volume_step", 0.01) or 0.01)
            volume_min = float(getattr(info, "volume_min", 0.01) or 0.01)
            volume_max = float(getattr(info, "volume_max", 100.0) or 100.0)

            # Round SL/TP to digits precision. None/0 stay as-is
            # (MT5 accepts 0.0 for "no SL/TP").
            sl_n = round(float(sl), digits) if sl else 0.0
            tp_n = round(float(tp), digits) if tp else 0.0

            # Snap lot to volume_step: round to nearest step, then clamp
            # to [volume_min, volume_max].
            lot_f = float(lot)
            # Snap to grid: round(lot / step) * step
            lot_snapped = round(lot_f / volume_step) * volume_step
            # Round to 2 decimals to avoid float noise (e.g. 0.01000000001)
            lot_snapped = round(lot_snapped, 2)
            # Clamp to broker's min/max
            if lot_snapped < volume_min:
                log.warning(
                    f"[OrderManager] lot {lot} below volume_min {volume_min} "
                    f"for {broker_symbol} — clamping up to {volume_min}"
                )
                lot_snapped = volume_min
            elif lot_snapped > volume_max:
                log.warning(
                    f"[OrderManager] lot {lot} above volume_max {volume_max} "
                    f"for {broker_symbol} — clamping down to {volume_max}"
                )
                lot_snapped = volume_max

            return (lot_snapped, sl_n, tp_n, digits, volume_step)
        except Exception as e:
            log.warning(
                f"[OrderManager] _normalize_order_params failed for "
                f"{broker_symbol}: {e} — using raw values (may cause 10015/10014)"
            )
            return None

    # ─────────────────────────────────────────────
    # BONUS 2 + 3 — RETRY + CONFIRMATION
    # ─────────────────────────────────────────────

    def _check_confirmation(self, result, attempt: int, requested_volume: float = None, symbol: str = "?") -> dict:
        """mt5.order_send()-এর result.retcode চেক করে success/failure ঠিক করে।

        Audit fix (EX-1 / X-3): a "successful" retcode (10008/10009) only
        means the broker accepted and executed the order — it does NOT
        guarantee `result.volume` equals what we requested. On thin
        liquidity / fast markets, brokers can partially fill a market
        order (e.g. requested 1.00 lot, filled 0.60). Previously this
        method returned `"volume": result.volume` with no comparison
        against the requested size, so callers (ExecutionRouter, PaperTrader
        sync, position sizing) silently assumed full fill — understating
        real exposure risk if a huge lot was requested and only partially
        filled, or leaving the "missing" volume completely untracked.

        Now: when `requested_volume` is supplied, we compare it against
        `result.volume` and flag `partial_fill=True` + `remaining_volume`
        whenever the filled amount is materially less than requested, so
        callers can decide how to handle the shortfall (log/alert, size
        down downstream risk tracking, or place a follow-up order) instead
        of silently misreporting the fill as complete.
        """
        if result is None:
            # Day 81+ hotfix: log mt5.last_error() so the operator can
            # see WHY order_send returned None (terminal disconnected,
            # IPC pipe broken, terminal not running, etc.).  Previously
            # this was silent — only "order_send returned None" was
            # logged, with no MT5-side diagnostic.
            try:
                last_err = mt5.last_error()
            except Exception as e:
                last_err = "(last_error() itself failed)"
            log.error(
                f"[OrderManager] order_send returned None on attempt {attempt} — "
                f"mt5.last_error()={last_err}"
            )
            try:
                from core.execution_logger import log_broker_last_error
                log_broker_last_error(symbol=symbol, error=last_err,
                                      attempt=attempt, stage="order_send_none")
            except Exception as e:
                log.warning(f"Suppressed exception at line 585: {e}")
                pass
            return {
                "success": False,
                "reason": f"order_send returned None (last_error={last_err})",
                "retryable": True,
            }

        if result.retcode in RETCODE_SUCCESS:
            filled_volume = float(result.volume or 0)
            # Partial-fill detection: tolerate tiny float noise (0.001 lot).
            partial_fill = (
                requested_volume is not None
                and filled_volume > 0
                and filled_volume < (float(requested_volume) - 0.001)
            )
            remaining_volume = (
                round(float(requested_volume) - filled_volume, 3)
                if partial_fill else 0.0
            )

            # Day 99+ FIX (Issue #4): distinguish PLACED (10008, pending)
            # from DONE (10009, fully filled). Both are technically
            # "success" but the router / position manager need to know
            # whether to expect the position immediately or poll for it.
            is_pending = result.retcode in RETCODE_PENDING
            if is_pending:
                log.info(
                    f"[OrderManager] order PLACED (retcode=10008) — broker "
                    f"accepted but position may not be queryable yet. "
                    f"Will poll positions_get to confirm."
                )

            if partial_fill:
                log.warning(
                    f"[OrderManager] ⚠️  PARTIAL FILL — {symbol} requested="
                    f"{requested_volume} filled={filled_volume} "
                    f"remaining={remaining_volume} (attempt {attempt})"
                )
                try:
                    from core.execution_logger import log_event
                    log_event(
                        "order.partial_fill", symbol=symbol,
                        requested_volume=requested_volume,
                        filled_volume=filled_volume,
                        remaining_volume=remaining_volume,
                        ticket=result.order or result.deal,
                        attempt=attempt,
                    )
                except Exception as e:
                    log.warning(f"Suppressed exception logging partial_fill event: {e}")

            # Day 81+ hotfix: log every successful order_send to logs/execution.log
            try:
                from core.execution_logger import log_broker_order_send
                log_broker_order_send(
                    symbol=symbol,
                    retcode=result.retcode,
                    comment=getattr(result, "comment", None),
                    price=result.price,
                    volume=result.volume,
                    ticket=result.order or result.deal,
                    attempt=attempt,
                )
            except Exception as e:
                log.warning(f"Suppressed exception at line 607: {e}")
                pass
            return {
                "success": True,
                "ticket": result.order or result.deal,
                "retcode": result.retcode,
                "price": result.price,
                "volume": filled_volume,
                "requested_volume": requested_volume,
                "partial_fill": partial_fill,
                "remaining_volume": remaining_volume,
                # Day 99+ FIX (Issue #4): expose pending state so the
                # caller can decide whether to poll for the position
                # or treat it as already filled.
                "pending": is_pending,
            }

        # Permanent rejection reasons — retry-এর মানে নেই
        permanent_codes = {
            10013,  # TRADE_RETCODE_INVALID — invalid request
            10014,  # invalid volume
            10015,  # invalid price
            10016,  # invalid stops
            10019,  # no money
            10027,  # autotrading disabled (client side) — broker পরিবর্তন ছাড়া retry futile
        }
        # 10030 (Unsupported filling mode) is RETRYABLE because _resolve_filling_mode
        # will pick a different mode on the next attempt. This was the #1 cause
        # of "trades silently fail" on MetaQuotes-Demo and ICMarkets demo servers
        # — they reject ORDER_FILLING_FOK outright.
        retryable = result.retcode not in permanent_codes

        log.warning(
            f"[OrderManager] Attempt {attempt} rejected — retcode={result.retcode} "
            f"comment={getattr(result, 'comment', '')}"
        )
        return {
            "success": False,
            "reason": f"retcode={result.retcode} ({getattr(result, 'comment', 'no comment')})",
            "retryable": retryable,
        }

    def _attempt_fill_remainder(
        self, broker_symbol: str, direction: str, sl, tp, comment: str,
        first_outcome: dict,
    ) -> dict:
        """One follow-up order attempt for the volume MT5 didn't fill on the
        primary order (EX-1). Best-effort: if this also fails or comes back
        partial, we log clearly and return the ACCURATE combined totals —
        never silently claim more volume was filled than actually was.

        Returns `first_outcome` merged with:
          - "volume": combined filled volume across both tickets
          - "tickets": [primary_ticket, followup_ticket] (followup omitted if it failed)
          - "partial_fill": whether there's STILL unfilled volume after this attempt
          - "remaining_volume": whatever is still outstanding
        """
        remaining = first_outcome.get("remaining_volume", 0.0)
        primary_ticket = first_outcome.get("ticket")
        if remaining <= 0:
            return first_outcome

        log.info(
            f"[OrderManager] Attempting follow-up fill for remaining "
            f"{remaining} lot on {broker_symbol} (primary ticket={primary_ticket})"
        )
        try:
            tick = mt5.symbol_info_tick(broker_symbol)
            if tick is None:
                raise RuntimeError("no tick data for follow-up order")
            price = tick.ask if direction == "BUY" else tick.bid
            order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
            filling_mode = _resolve_filling_mode(broker_symbol)
            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       broker_symbol,
                "volume":       remaining,
                "type":         order_type,
                "price":        price,
                "sl":           sl or 0.0,
                "tp":           tp or 0.0,
                "deviation":    10,
                "magic":        424242,
                "comment":      f"{comment}_topup",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            result = mt5.order_send(request)
            if result is None or not hasattr(result, "retcode"):
                raise RuntimeError("follow-up order_send returned invalid response")

            followup = self._check_confirmation(result, attempt=1, requested_volume=remaining, symbol=broker_symbol)
        except Exception as e:
            log.warning(
                f"[OrderManager] Follow-up fill attempt failed for "
                f"{broker_symbol}: {e} — reporting actual filled volume only"
            )
            first_outcome["tickets"] = [primary_ticket]
            return first_outcome

        combined_volume = float(first_outcome.get("volume", 0)) + float(followup.get("volume", 0) or 0)
        still_remaining = followup.get("remaining_volume", remaining) if followup.get("success") else remaining

        if followup.get("success"):
            log.info(
                f"[OrderManager] ✅ Follow-up filled {followup.get('volume')} lot "
                f"(ticket={followup.get('ticket')}) — combined volume={combined_volume}"
            )
        else:
            log.warning(
                f"[OrderManager] ⚠️  Follow-up fill FAILED for {broker_symbol}: "
                f"{followup.get('reason')} — still short {still_remaining} lot "
                f"(primary ticket={primary_ticket} filled {first_outcome.get('volume')} lot only)"
            )

        first_outcome["volume"] = combined_volume
        first_outcome["partial_fill"] = still_remaining > 0.001
        first_outcome["remaining_volume"] = round(still_remaining, 3) if still_remaining > 0.001 else 0.0
        first_outcome["tickets"] = [primary_ticket] + (
            [followup.get("ticket")] if followup.get("success") else []
        )
        return first_outcome

    def _wait_retry(self, attempt: int, reason: str) -> None:
        log.warning(f"[OrderManager] Retry {attempt}/{self.MAX_RETRIES} — {reason}")
        time.sleep(self.RETRY_DELAY_SEC)

    # ─────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────

    def _get_position(self, ticket: int):
        if not MT5_AVAILABLE:
            return None
        positions = _mt5_positions_get(ticket=ticket)
        if not positions:
            return None
        return positions[0]