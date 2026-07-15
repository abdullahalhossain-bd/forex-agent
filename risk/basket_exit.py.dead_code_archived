# risk/basket_exit.py — Average-price basket exit system
# =============================================================================
# Inspired by: https://github.com/smartedgetrading/SmartEdge-EA
# SmartEdge docs: "Average-price-based exit system" and
# "Managing position baskets as a group"
#
# When using grid-scaling (multiple positions on the same symbol in the same
# direction), the exit logic must consider the BASKET as a whole, not
# individual positions. The exit trigger is based on the volume-weighted
# average entry price, not any single position's entry.
#
# This module computes:
#   - Average entry price = sum(price_i * lot_i) / sum(lot_i)
#   - Basket break-even price (same as average for forex, no commission)
#   - Basket TP = average + tp_pips * pip_size (for BUY)
#   - Basket SL = average - sl_pips * pip_size (for BUY)
#
# And provides a `check_exit()` method that returns:
#   - "TP" if current price hits the basket take-profit
#   - "SL" if current price hits the basket stop-loss
#   - "BE" if current price is at break-even (optional)
#   - None if no exit condition is met
#
# Usage
# -----
#     exit_mgr = BasketExitManager(tp_pips=30, sl_pips=50, pip_size=0.0001)
#
#     # Add positions (manually or from ControlledGridScaler)
#     exit_mgr.add_position("EURUSD", "BUY", price=1.0850, lot=0.1, ticket=1)
#     exit_mgr.add_position("EURUSD", "BUY", price=1.0825, lot=0.1, ticket=2)
#
#     # Check exit on each tick
#     signal = exit_mgr.check_exit("EURUSD", current_price=1.0860)
#     if signal == "TP":
#         # Close entire basket
#         for ticket in exit_mgr.get_tickets("EURUSD"):
#             broker.close_position(ticket)
#         exit_mgr.clear("EURUSD")
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from utils.logger import get_logger

log = get_logger("basket_exit")


@dataclass
class _BasketPosition:
    """A single position within a basket."""
    ticket: int
    direction: str       # "BUY" or "SELL"
    price: float
    lot: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class BasketExitManager:
    """
    Manages basket-level exit logic for grid-scaled positions.

    Computes volume-weighted average entry price, then checks TP/SL against
    that average (not individual position entries).
    """

    def __init__(
        self,
        tp_pips: float = 30.0,
        sl_pips: float = 50.0,
        pip_size: float = 0.0001,
        break_even_buffer_pips: float = 0.0,
    ):
        """
        Parameters
        ----------
        tp_pips : take-profit distance from average price, in pips.
        sl_pips : stop-loss distance from average price, in pips.
        pip_size : pip size (0.0001 for USD pairs, 0.01 for JPY pairs).
        break_even_buffer_pips : if > 0, a "BE" (break-even) signal is
            emitted when the basket is within this many pips of the average.
            Useful for trailing stops. Set to 0 to disable.
        """
        if tp_pips <= 0:
            raise ValueError("tp_pips must be > 0")
        if sl_pips <= 0:
            raise ValueError("sl_pips must be > 0")
        if pip_size <= 0:
            raise ValueError("pip_size must be > 0")

        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        self.pip_size = pip_size
        self.be_buffer = break_even_buffer_pips * pip_size

        # symbol → list of _BasketPosition
        self._baskets: dict[str, list[_BasketPosition]] = {}

    # ── Position management ──────────────────────────────────────────────────

    def add_position(self, symbol: str, direction: str, price: float,
                     lot: float, ticket: int) -> None:
        """Add a position to the basket for `symbol`."""
        direction = direction.upper()
        if direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY or SELL, got {direction!r}")

        # Direction consistency check
        existing = self._baskets.get(symbol, [])
        if existing and existing[0].direction != direction:
            raise ValueError(
                f"Basket for {symbol} is {existing[0].direction}, "
                f"can't add {direction}"
            )

        pos = _BasketPosition(ticket=ticket, direction=direction,
                               price=price, lot=lot)
        self._baskets.setdefault(symbol, []).append(pos)
        log.info(f"BasketExit: added {direction} {symbol} "
                 f"@ {price:.5f} lot={lot} ticket={ticket} "
                 f"(basket size: {len(self._baskets[symbol])})")

    def remove_position(self, symbol: str, ticket: int) -> None:
        """Remove a single position (after it closes)."""
        existing = self._baskets.get(symbol, [])
        before = len(existing)
        self._baskets[symbol] = [p for p in existing if p.ticket != ticket]
        after = len(self._baskets[symbol])
        if after == 0:
            del self._baskets[symbol]
        if before == after:
            log.warning(f"BasketExit: ticket {ticket} not found for {symbol}")

    def clear(self, symbol: str) -> None:
        """Clear the entire basket for `symbol` (after closing all positions)."""
        if symbol in self._baskets:
            n = len(self._baskets[symbol])
            del self._baskets[symbol]
            log.info(f"BasketExit: cleared {n} positions for {symbol}")

    # ── Basket calculations ──────────────────────────────────────────────────

    def get_average_price(self, symbol: str) -> Optional[float]:
        """Volume-weighted average entry price. None if basket is empty."""
        existing = self._baskets.get(symbol, [])
        if not existing:
            return None
        total_value = sum(p.price * p.lot for p in existing)
        total_lots = sum(p.lot for p in existing)
        return total_value / total_lots if total_lots > 0 else None

    def get_total_lots(self, symbol: str) -> float:
        """Total lot size across all positions in the basket."""
        return sum(p.lot for p in self._baskets.get(symbol, []))

    def get_direction(self, symbol: str) -> Optional[str]:
        """Direction of the basket (BUY or SELL). None if empty."""
        existing = self._baskets.get(symbol, [])
        return existing[0].direction if existing else None

    def get_tickets(self, symbol: str) -> list[int]:
        """List of all ticket numbers in the basket."""
        return [p.ticket for p in self._baskets.get(symbol, [])]

    def get_basket_size(self, symbol: str) -> int:
        """Number of positions in the basket."""
        return len(self._baskets.get(symbol, []))

    # ── Exit level calculations ──────────────────────────────────────────────

    def get_tp_price(self, symbol: str) -> Optional[float]:
        """Basket take-profit price. None if basket is empty."""
        avg = self.get_average_price(symbol)
        direction = self.get_direction(symbol)
        if avg is None or direction is None:
            return None
        tp_distance = self.tp_pips * self.pip_size
        if direction == "BUY":
            return avg + tp_distance
        else:
            return avg - tp_distance

    def get_sl_price(self, symbol: str) -> Optional[float]:
        """Basket stop-loss price. None if basket is empty."""
        avg = self.get_average_price(symbol)
        direction = self.get_direction(symbol)
        if avg is None or direction is None:
            return None
        sl_distance = self.sl_pips * self.pip_size
        if direction == "BUY":
            return avg - sl_distance
        else:
            return avg + sl_distance

    def get_be_price(self, symbol: str) -> Optional[float]:
        """Break-even price (= average price). None if basket is empty."""
        return self.get_average_price(symbol)

    # ── Exit check ───────────────────────────────────────────────────────────

    def check_exit(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if the basket should be closed based on `current_price`.

        Returns:
          - "TP" if take-profit is hit
          - "SL" if stop-loss is hit
          - "BE" if break-even buffer is hit (only if break_even_buffer_pips > 0)
          - None if no exit condition is met

        TP/SL logic:
          For BUY basket: TP when price ≥ avg + tp_pips; SL when price ≤ avg - sl_pips
          For SELL basket: TP when price ≤ avg - tp_pips; SL when price ≥ avg + sl_pips
        """
        avg = self.get_average_price(symbol)
        direction = self.get_direction(symbol)
        if avg is None or direction is None:
            return None

        tp = self.get_tp_price(symbol)
        sl = self.get_sl_price(symbol)

        if direction == "BUY":
            if current_price >= tp:
                log.info(f"BasketExit: {symbol} TP hit — price {current_price:.5f} "
                         f"≥ {tp:.5f} (avg {avg:.5f} + {self.tp_pips} pips)")
                return "TP"
            if current_price <= sl:
                log.info(f"BasketExit: {symbol} SL hit — price {current_price:.5f} "
                         f"≤ {sl:.5f} (avg {avg:.5f} - {self.sl_pips} pips)")
                return "SL"
        else:  # SELL
            if current_price <= tp:
                log.info(f"BasketExit: {symbol} TP hit — price {current_price:.5f} "
                         f"≤ {tp:.5f} (avg {avg:.5f} - {self.tp_pips} pips)")
                return "TP"
            if current_price >= sl:
                log.info(f"BasketExit: {symbol} SL hit — price {current_price:.5f} "
                         f"≥ {sl:.5f} (avg {avg:.5f} + {self.sl_pips} pips)")
                return "SL"

        # Break-even check (optional)
        if self.be_buffer > 0:
            if direction == "BUY" and avg <= current_price <= avg + self.be_buffer:
                return "BE"
            if direction == "SELL" and avg - self.be_buffer <= current_price <= avg:
                return "BE"

        return None

    # ── Full state ───────────────────────────────────────────────────────────

    def get_basket_state(self, symbol: str) -> dict:
        """Return full state of the basket for debugging/audit."""
        existing = self._baskets.get(symbol, [])
        avg = self.get_average_price(symbol)
        return {
            "symbol": symbol,
            "direction": self.get_direction(symbol),
            "size": len(existing),
            "average_price": avg,
            "total_lots": self.get_total_lots(symbol),
            "tp_price": self.get_tp_price(symbol),
            "sl_price": self.get_sl_price(symbol),
            "be_price": avg,
            "tp_pips": self.tp_pips,
            "sl_pips": self.sl_pips,
            "positions": [
                {"ticket": p.ticket, "price": p.price, "lot": p.lot,
                 "time": p.timestamp}
                for p in existing
            ],
        }

    def get_all_symbols(self) -> list[str]:
        """Return all symbols with active baskets."""
        return list(self._baskets.keys())


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mgr = BasketExitManager(tp_pips=30, sl_pips=50, pip_size=0.0001)

    # Build a BUY basket: 0.1 @ 1.0850, 0.1 @ 1.0825
    mgr.add_position("EURUSD", "BUY", price=1.0850, lot=0.1, ticket=1)
    mgr.add_position("EURUSD", "BUY", price=1.0825, lot=0.1, ticket=2)

    # Average = (1.0850*0.1 + 1.0825*0.1) / 0.2 = 1.08375
    avg = mgr.get_average_price("EURUSD")
    assert abs(avg - 1.08375) < 1e-6, f"got {avg}"
    print(f"Average price: {avg:.5f}")

    # TP = 1.08375 + 30*0.0001 = 1.08375 + 0.003 = 1.08675
    tp = mgr.get_tp_price("EURUSD")
    assert abs(tp - 1.08675) < 1e-6, f"got {tp}"
    print(f"TP price: {tp:.5f}")

    # SL = 1.08375 - 50*0.0001 = 1.08375 - 0.005 = 1.07875
    sl = mgr.get_sl_price("EURUSD")
    assert abs(sl - 1.07875) < 1e-6, f"got {sl}"
    print(f"SL price: {sl:.5f}")

    # Price at 1.0860 — no exit
    assert mgr.check_exit("EURUSD", 1.0860) is None

    # Price hits TP
    assert mgr.check_exit("EURUSD", 1.08700) == "TP"

    # Price hits SL
    assert mgr.check_exit("EURUSD", 1.07800) == "SL"

    # Test SELL basket
    mgr2 = BasketExitManager(tp_pips=30, sl_pips=50, pip_size=0.01)  # JPY pair
    mgr2.add_position("USDJPY", "SELL", price=150.00, lot=0.1, ticket=10)
    mgr2.add_position("USDJPY", "SELL", price=150.50, lot=0.1, ticket=11)
    # avg = (150.00 + 150.50) / 2 = 150.25
    # TP = 150.25 - 30*0.01 = 150.25 - 0.30 = 149.95
    # SL = 150.25 + 50*0.01 = 150.25 + 0.50 = 150.75
    assert abs(mgr2.get_average_price("USDJPY") - 150.25) < 1e-6
    assert abs(mgr2.get_tp_price("USDJPY") - 149.95) < 1e-6
    assert abs(mgr2.get_sl_price("USDJPY") - 150.75) < 1e-6
    assert mgr2.check_exit("USDJPY", 149.90) == "TP"
    assert mgr2.check_exit("USDJPY", 150.80) == "SL"
    print(f"USDJPY SELL basket: avg={mgr2.get_average_price('USDJPY'):.2f}, "
          f"TP={mgr2.get_tp_price('USDJPY'):.2f}, SL={mgr2.get_sl_price('USDJPY'):.2f}")

    # Test direction mismatch
    try:
        mgr.add_position("EURUSD", "SELL", 1.0800, 0.1, 99)
        assert False, "should have raised"
    except ValueError:
        pass

    # Test empty basket
    assert mgr.get_average_price("GBPUSD") is None
    assert mgr.check_exit("GBPUSD", 1.2500) is None

    print(f"\nBasket state: {mgr.get_basket_state('EURUSD')}")
    print("\nBasketExitManager smoke test passed.")
