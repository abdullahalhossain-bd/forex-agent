# broker/symbol_manager.py  —  Day 32 Part 3 | Multi-Pair Scanner
# ============================================================
# একসাথে একাধিক pair-এর basic trend/volatility snapshot নেয়।
# এটা rule_engine.py-এর replacement না — সেটা uploaded হয়নি, তাই
# এখানে শুধু একটা lightweight heuristic (EMA slope + ATR-based
# volatility bucket) দেখানো হলো scanner output demonstrate করার
# জন্য। আসল rule_engine.py এলে এই heuristic বদলে দেওয়া উচিত।
# ============================================================

from utils.logger import get_logger
from broker.mt5_data import MT5DataFeed

log = get_logger("symbol_manager")

DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]


class SymbolManager:
    """
    Multi-pair scanner — broker symbol resolve করা AccountManager-এর
    দায়িত্ব, এই class শুধু resolved symbol list নিয়ে scan করে।

    Usage:
        sm = SymbolManager(account_manager)
        broker_symbols = sm.resolve_all(["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"])
        snapshot = sm.scan(broker_symbols)
    """

    def __init__(self, account_manager, data_feed: MT5DataFeed = None):
        self.account_manager = account_manager
        self.feed = data_feed or MT5DataFeed()

    def resolve_all(self, requested_symbols: list[str]) -> dict[str, str | None]:
        """প্রতিটা requested symbol-কে broker-এর exact name-এ resolve করে।"""
        resolved = {}
        for sym in requested_symbols:
            resolved[sym] = self.account_manager.resolve_symbol(sym)
        return resolved

    def scan(self, broker_symbols: list[str]) -> dict[str, dict]:
        """
        প্রতিটা pair-এর জন্য basic trend/volatility snapshot বানায়।
        rule_engine.py uploaded হলে এই snapshot-টা সেই engine-এর input
        হতে পারে — এখন শুধু M15 candle থেকে heuristic বের করা হচ্ছে।
        """
        results = {}
        for sym in broker_symbols:
            candles = self.feed.get_candles(sym, "M15", count=50)
            if not candles:
                results[sym] = {"status": "NO_DATA"}
                continue
            results[sym] = self._classify(candles)
        return results

    def _classify(self, candles: list[dict]) -> dict:
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]

        recent_avg = sum(closes[-10:]) / 10
        older_avg = sum(closes[-30:-10]) / 20 if len(closes) >= 30 else sum(closes[:-10]) / max(1, len(closes) - 10)

        slope = recent_avg - older_avg
        avg_range = sum(h - l for h, l in zip(highs[-20:], lows[-20:])) / min(20, len(highs))
        price_ref = closes[-1] or 1
        range_pct = (avg_range / price_ref) * 100 if price_ref else 0

        if abs(slope) < avg_range * 0.3:
            trend = "RANGE"
        elif slope > 0:
            trend = "BULLISH"
        else:
            trend = "BEARISH"

        volatility = "HIGH_VOLATILITY" if range_pct > 0.15 else "NORMAL"

        return {
            "status": "OK",
            "trend": trend,
            "volatility": volatility,
            "last_close": closes[-1],
        }

    def print_scan(self, broker_symbols: list[str]) -> None:
        results = self.scan(broker_symbols)
        bar = "═" * 36
        log.info(bar)
        log.info("  🔍  MARKET SCANNER")
        log.info(bar)
        for sym, r in results.items():
            if r["status"] != "OK":
                log.info(f"  {sym:<8} ❌ {r['status']}")
                continue
            tag = r["volatility"] if r["volatility"] == "HIGH_VOLATILITY" else r["trend"]
            log.info(f"  {sym:<8} {tag}")
        log.info(bar)

    # ─────────────────────────────────────────────
    # MT5 User Guide Page 8 — Full Symbol Specification
    # ─────────────────────────────────────────────

    def get_symbol_specification(self, symbol: str) -> dict | None:
        """
        MT5 User Guide Page 8 — query full symbol specification.
        Returns all fields needed for order sizing + risk management:
          - digits, contract_size, point, spread
          - stops_level (min SL/TP distance from market)
          - volume_min, volume_max, volume_step
          - swap_mode, swap_long, swap_short
          - session hours (trade_start, trade_end)
          - profit_mode, margin_mode
        """
        try:
            import MetaTrader5 as mt5
        except ImportError:
            log.warning("[SymbolManager] MetaTrader5 not available")
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            log.warning(f"[SymbolManager] symbol_info() failed for {symbol}: {mt5.last_error()}")
            return None

        spec = {
            "symbol":           info.name,
            "digits":           info.digits,
            "point":            info.point,
            "spread":           info.spread,
            "spread_float":     info.spread_float,
            "contract_size":    info.trade_contract_size,
            "stops_level":      info.trade_stops_level,
            "freeze_level":     info.trade_freeze_level,
            "volume_min":       info.volume_min,
            "volume_max":       info.volume_max,
            "volume_step":      info.volume_step,
            "swap_mode":        info.swap_mode,
            "swap_long":        info.swap_long,
            "swap_short":       info.swap_short,
            "margin_currency":  getattr(info, "margin_currency", ""),
            "profit_currency":  getattr(info, "profit_currency", ""),
            "margin_mode":      getattr(info, "margin_mode", None),
            "profit_mode":      getattr(info, "profit_mode", None),
            # Session info (may be None for some symbols)
            "session_deals":    getattr(info, "session_deals", None),
            "session_quotes":   getattr(info, "session_quotes", None),
            # Visibility
            "visible":          info.visible,
            "selected":         info.selected,
        }

        # Add human-readable swap mode label
        swap_modes = {
            0: "Disabled",
            1: "Points",
            2: "Symbol Currency Interest Rate",
            3: "Margin Currency Interest Rate",
            4: "Percentage of Price",
            5: "Percentage of Open Price",
        }
        spec["swap_mode_label"] = swap_modes.get(info.swap_mode, f"Unknown({info.swap_mode})")

        log.info(
            f"[SymbolManager] {symbol} spec: digits={info.digits}, "
            f"contract={info.trade_contract_size}, stops_level={info.trade_stops_level}, "
            f"vol_min={info.volume_min}, vol_step={info.volume_step}, "
            f"swap={spec['swap_mode_label']}"
        )
        return spec

    def validate_order_against_spec(
        self, symbol: str, price: float, sl: float, tp: float, volume: float
    ) -> dict:
        """
        MT5 User Guide Page 8 — validate order parameters against symbol spec.
        Checks: stops_level (SL/TP distance), volume_min/max/step.
        Returns: {"valid": bool, "violations": [str]}
        """
        spec = self.get_symbol_specification(symbol)
        if spec is None:
            return {"valid": False, "violations": ["Cannot retrieve symbol specification"]}

        violations = []

        # Check stops_level — SL and TP must be at least stops_level points from price
        stops_level_price = spec["stops_level"] * spec["point"]
        if stops_level_price > 0:
            if sl and abs(price - sl) < stops_level_price:
                violations.append(
                    f"SL {sl} is only {abs(price-sl)/spec['point']:.0f} points from price, "
                    f"min required: {spec['stops_level']} (stops_level)"
                )
            if tp and abs(price - tp) < stops_level_price:
                violations.append(
                    f"TP {tp} is only {abs(price-tp)/spec['point']:.0f} points from price, "
                    f"min required: {spec['stops_level']} (stops_level)"
                )

        # Check volume bounds
        if volume < spec["volume_min"]:
            violations.append(
                f"Volume {volume} < min {spec['volume_min']}"
            )
        if volume > spec["volume_max"]:
            violations.append(
                f"Volume {volume} > max {spec['volume_max']}"
            )
        # Check volume step alignment
        if spec["volume_step"] > 0:
            steps = round(volume / spec["volume_step"])
            expected_vol = steps * spec["volume_step"]
            if abs(volume - expected_vol) > spec["volume_step"] * 0.01:
                violations.append(
                    f"Volume {volume} not aligned to step {spec['volume_step']} "
                    f"(nearest: {expected_vol})"
                )

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "spec": spec,
        }

    # ─────────────────────────────────────────────
    # MT5 User Guide Page 39 — Volume Limit Check
    # ─────────────────────────────────────────────

    def check_volume_limit(
        self, symbol: str, direction: str, proposed_volume: float,
        open_positions: list = None, pending_orders: list = None,
    ) -> dict:
        """
        MT5 User Guide Page 39 — Volume limit validation.

        Rule: total same-direction volume (open positions + pending orders)
        cannot exceed the symbol's Volume limit. Opposite-direction orders
        are exempt from this cap.

        Worked example (from book):
          If limit is 5 lots and trader holds 5-lot Buy:
            - CAN place 5-lot Sell Limit (opposite direction, exempt)
            - CANNOT place additional Buy Limit (same direction, would exceed)
            - CANNOT place Sell Limit exceeding 5 lots

        Args:
            symbol: e.g., "EURUSD"
            direction: "BUY" or "SELL" (proposed new order)
            proposed_volume: lot size of proposed order
            open_positions: list of {"symbol": str, "direction": str, "volume": float}
            pending_orders: list of {"symbol": str, "direction": str, "volume": float}

        Returns:
            {"valid": bool, "violations": [str], "current_exposure": float, "limit": float}
        """
        spec = self.get_symbol_specification(symbol)
        if spec is None:
            return {"valid": False, "violations": ["Cannot retrieve spec for volume limit check"]}

        # Volume limit is not always exposed via symbol_info — try to get it
        volume_limit = getattr(spec, "volume_limit", None)
        if volume_limit is None or volume_limit <= 0:
            # If no explicit volume_limit field, use volume_max as fallback
            volume_limit = spec.get("volume_max", 0)

        if volume_limit <= 0:
            return {"valid": True, "violations": [], "current_exposure": 0, "limit": 0,
                    "note": "No volume limit configured for this symbol"}

        open_positions = open_positions or []
        pending_orders = pending_orders or []
        dir_upper = direction.upper()

        # Calculate same-direction aggregate exposure
        same_direction_volume = proposed_volume

        for pos in open_positions:
            if pos.get("symbol", "").upper() == symbol.upper():
                if pos.get("direction", "").upper() == dir_upper:
                    same_direction_volume += float(pos.get("volume", 0))

        for order in pending_orders:
            if order.get("symbol", "").upper() == symbol.upper():
                if order.get("direction", "").upper() == dir_upper:
                    same_direction_volume += float(order.get("volume", 0))

        violations = []
        if same_direction_volume > volume_limit:
            violations.append(
                f"Total same-direction {dir_upper} volume {same_direction_volume:.2f} "
                f"(proposed {proposed_volume:.2f} + existing "
                f"{same_direction_volume - proposed_volume:.2f}) exceeds "
                f"Volume limit {volume_limit:.2f} for {symbol}"
            )

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "current_same_direction_exposure": round(same_direction_volume, 2),
            "proposed_volume": proposed_volume,
            "volume_limit": volume_limit,
            "direction": dir_upper,
        }

    # ─────────────────────────────────────────────
    # MT5 User Guide Page 38 — Trade Mode Permission Check
    # ─────────────────────────────────────────────

    def check_trade_mode(self, symbol: str) -> dict:
        """
        MT5 User Guide Page 38 — check if symbol allows trading.

        Trade mode values:
          0 = TRADE_MODE_DISABLED (no trading)
          1 = TRADE_MODE_LONGONLY (only long positions)
          2 = TRADE_MODE_SHORTONLY (only short positions)
          3 = TRADE_MODE_CLOSEONLY (only close existing positions)
          4 = TRADE_MODE_FULL (full trading, default)

        Returns:
            {"allowed": bool, "mode": str, "restrictions": [str]}
        """
        spec = self.get_symbol_specification(symbol)
        if spec is None:
            return {"allowed": False, "mode": "unknown",
                    "restrictions": ["Cannot retrieve symbol spec"]}

        try:
            import MetaTrader5 as mt5
            info = mt5.symbol_info(symbol)
            if info is None:
                return {"allowed": False, "mode": "unknown",
                        "restrictions": ["symbol_info() returned None"]}

            mode = info.trade_mode
            mode_labels = {
                0: "DISABLED",
                1: "LONGONLY",
                2: "SHORTONLY",
                3: "CLOSEONLY",
                4: "FULL",
            }
            mode_label = mode_labels.get(mode, f"UNKNOWN({mode})")

            restrictions = []
            allowed = True

            if mode == 0:
                allowed = False
                restrictions.append("Trading is DISABLED for this symbol")
            elif mode == 1:
                restrictions.append("Only LONG (BUY) positions allowed")
            elif mode == 2:
                restrictions.append("Only SHORT (SELL) positions allowed")
            elif mode == 3:
                restrictions.append("Only CLOSE of existing positions allowed (no new entries)")
            elif mode == 4:
                pass  # Full trading — no restrictions

            return {
                "allowed": allowed,
                "mode": mode_label,
                "mode_code": mode,
                "restrictions": restrictions,
            }
        except Exception as e:
            return {"allowed": False, "mode": "error",
                    "restrictions": [str(e)]}

    # ─────────────────────────────────────────────
    # MT5 User Guide Page 40 — Triple Swap Day Calculation
    # ─────────────────────────────────────────────

    def calculate_swap_cost(
        self, symbol: str, direction: str, volume: float,
        holding_days: int = 1, current_price: float = None,
    ) -> dict:
        """
        MT5 User Guide Page 40 — calculate overnight swap (rollover) cost.

        Handles 8 swap types from book Page 39:
          0: Disabled (no swap)
          1: In points
          2: In base currency
          3: In margin currency
          4: In deposit currency
          5: As percentage of current price
          6: As percentage of open price
          7: Re-open at Close price
          8: Re-open at Bid price

        Triple swap day (Page 40):
          On Wednesday (for FX), 3× normal swap is charged to cover weekend.
          This is essential for accurate overnight cost modeling.

        Args:
            symbol: e.g., "EURUSD"
            direction: "BUY" or "SELL"
            volume: lot size
            holding_days: number of days held (default 1)
            current_price: current market price (for percentage-based swap)

        Returns:
            {"swap_cost": float, "swap_type": str, "triple_swap_applied": bool}
        """
        spec = self.get_symbol_specification(symbol)
        if spec is None:
            return {"swap_cost": 0, "swap_type": "unknown", "triple_swap_applied": False}

        swap_mode = spec.get("swap_mode", 0)
        swap_long = spec.get("swap_long", 0)
        swap_short = spec.get("swap_short", 0)
        swap_rate = swap_long if direction.upper() == "BUY" else swap_short
        contract_size = spec.get("contract_size", 1)
        point = spec.get("point", 0.0001)

        # Check if today is triple swap day (Wednesday for FX)
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc)
        is_triple_day = today.weekday() == 2  # Wednesday
        multiplier = 3 if is_triple_day else 1

        swap_cost = 0.0
        swap_type_label = spec.get("swap_mode_label", "Unknown")

        if swap_mode == 0:
            # Disabled
            swap_cost = 0.0
        elif swap_mode == 1:
            # In points — swap_rate is in points
            swap_cost = swap_rate * multiplier * volume * point * contract_size
        elif swap_mode in (2, 3, 4):
            # In base/margin/deposit currency — swap_rate is a fixed amount
            swap_cost = swap_rate * multiplier * volume
        elif swap_mode == 5:
            # As percentage of current price
            if current_price:
                swap_cost = (swap_rate / 100) * current_price * multiplier * volume * contract_size
        elif swap_mode == 6:
            # As percentage of open price (need open price — use current as fallback)
            if current_price:
                swap_cost = (swap_rate / 100) * current_price * multiplier * volume * contract_size
        elif swap_mode in (7, 8):
            # Re-open at Close/Bid price — complex, approximate as points-based
            swap_cost = swap_rate * multiplier * volume * point * contract_size
        else:
            swap_cost = 0.0

        # Multiply by holding days (if > 1 day, each day incurs swap)
        # But triple swap only applies once (on Wednesday)
        if holding_days > 1:
            extra_days = holding_days - 1
            swap_cost += swap_rate * extra_days * volume * point * contract_size

        return {
            "swap_cost": round(swap_cost, 5),
            "swap_type": swap_type_label,
            "swap_mode_code": swap_mode,
            "swap_rate": swap_rate,
            "volume": volume,
            "holding_days": holding_days,
            "triple_swap_applied": is_triple_day,
            "triple_multiplier": multiplier,
            "triple_swap_day": "Wednesday" if is_triple_day else "No",
        }