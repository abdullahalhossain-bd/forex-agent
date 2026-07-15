# risk/controlled_grid_scaler.py — Controlled (non-martingale) grid scaling
# =============================================================================
# Inspired by: https://github.com/smartedgetrading/SmartEdge-EA
# SmartEdge docs: "Controlled scaling... Scaling is limited, risk-aware,
# and never martingale-based" and "Grid-based position scaling with
# 2-step lot sizing"
#
# Controlled grid scaling adds positions to an existing trade as price moves
# AGAINST the entry, but with STRICT limits to avoid the blow-up risk of
# traditional martingale grids:
#
#   1. FIXED lot sizing (not doubling) — each scale uses the same lot size
#      as the initial entry, or a configurable multiplier capped at 1.0.
#   2. MAX LEVELS — hard ceiling on the number of scale-ins (default 3).
#   3. MIN SPACING — minimum price distance between scales (in ATR multiples
#      or pips) to prevent clustering.
#   4. MAX BASKET LOSS — if the basket's unrealized loss exceeds a threshold
#      (in USD or % of equity), no more scales are allowed.
#   5. DIRECTION LOCK — scales must be in the SAME direction as the initial
#      entry (no reversing).
#
# This is fundamentally different from martingale:
#   Martingale: 0.1 → 0.2 → 0.4 → 0.8 → 1.6 → 3.2 ... (exponential, blows up)
#   Controlled: 0.1 → 0.1 → 0.1 (linear, capped at 3 levels)
#
# Usage
# -----
#     scaler = ControlledGridScaler(
#         max_levels=3,
#         lot_size=0.1,
#         min_spacing_pips=20,
#         max_basket_loss_usd=100.0,
#     )
#
#     # Initial entry
#     if scaler.can_scale("EURUSD", "BUY", current_price, current_loss=0):
#         scaler.on_scale("EURUSD", "BUY", current_price, ticket=1001)
#
#     # Price drops 20 pips — check if we can scale in
#     if scaler.can_scale("EURUSD", "BUY", new_price, current_loss=-15.0):
#         scaler.on_scale("EURUSD", "BUY", new_price, ticket=1002)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from utils.logger import get_logger

log = get_logger("controlled_grid_scaler")


@dataclass
class _ScaleRecord:
    """Record of one scale-in event."""
    level: int                 # 1 = initial, 2 = first scale, etc.
    direction: str             # "BUY" or "SELL"
    price: float
    lot_size: float
    ticket: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class ControlledGridScaler:
    """
    Controlled grid-scaling guard. Tracks scale-in levels per symbol and
    enforces: max levels, min spacing, max basket loss, same-direction-only.
    """

    def __init__(
        self,
        max_levels: int = 3,
        lot_size: float = 0.1,
        lot_multiplier: float = 1.0,
        min_spacing_pips: float = 20.0,
        pip_size: float = 0.0001,
        max_basket_loss_usd: float = 100.0,
    ):
        """
        Parameters
        ----------
        max_levels : int
            Maximum number of positions per symbol (initial + scales).
            Default 3 (1 initial + 2 scales).
        lot_size : float
            Base lot size for all positions.
        lot_multiplier : float
            Multiplier applied per level. 1.0 = fixed (no martingale).
            Must be ≤ 1.5 to prevent exponential growth. If higher, raises.
        min_spacing_pips : float
            Minimum price distance between scales, in pips.
        pip_size : float
            Pip size for the symbol (0.0001 for USD pairs, 0.01 for JPY pairs).
        max_basket_loss_usd : float
            Maximum unrealized loss (in USD) before scaling is blocked.
            Set to 0 to disable this check.
        """
        if max_levels < 1:
            raise ValueError("max_levels must be >= 1")
        if lot_size <= 0:
            raise ValueError("lot_size must be > 0")
        if lot_multiplier > 1.5:
            raise ValueError(
                f"lot_multiplier {lot_multiplier} > 1.5 — this approaches "
                f"martingale. Use ≤ 1.5 for controlled scaling."
            )
        if min_spacing_pips < 0:
            raise ValueError("min_spacing_pips must be >= 0")

        self.max_levels = max_levels
        self.base_lot = lot_size
        self.lot_multiplier = lot_multiplier
        self.min_spacing_pips = min_spacing_pips
        self.pip_size = pip_size
        self.max_basket_loss_usd = max_basket_loss_usd

        # symbol → list of _ScaleRecord
        self._scales: dict[str, list[_ScaleRecord]] = {}

    # ── Pre-scale check ──────────────────────────────────────────────────────

    def can_scale(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        current_loss: float = 0.0,
    ) -> bool:
        """
        Check whether a new scale-in is allowed.

        Parameters
        ----------
        symbol : trading symbol.
        direction : "BUY" or "SELL" — must match the initial entry's direction.
        current_price : current market price.
        current_loss : current unrealized P&L of the basket in USD
            (negative = loss). Used for max_basket_loss_usd check.

        Returns True if all checks pass:
          - Under max_levels
          - Direction matches initial entry (if one exists)
          - Price has moved at least min_spacing_pips against the entry
          - Basket loss is under max_basket_loss_usd
        """
        direction = direction.upper()
        if direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY or SELL, got {direction!r}")

        existing = self._scales.get(symbol, [])

        # Check 1: max levels
        if len(existing) >= self.max_levels:
            log.debug(f"GridScaler: {symbol} at max levels ({self.max_levels}) — blocked")
            return False

        # Check 2: basket loss limit
        if self.max_basket_loss_usd > 0 and current_loss <= -self.max_basket_loss_usd:
            log.info(f"GridScaler: {symbol} basket loss ${current_loss:.2f} "
                     f"exceeds max ${self.max_basket_loss_usd:.2f} — blocked")
            return False

        # Check 3: direction match (if positions already exist)
        if existing:
            initial_direction = existing[0].direction
            if direction != initial_direction:
                log.debug(f"GridScaler: {symbol} initial dir={initial_direction}, "
                          f"requested={direction} — blocked (direction mismatch)")
                return False

            # Check 4: min spacing — price must have moved AGAINST the entry
            last_price = existing[-1].price
            min_distance = self.min_spacing_pips * self.pip_size

            if direction == "BUY":
                # For a BUY basket, price must have DROPPED to scale in lower
                price_movement = last_price - current_price
            else:
                # For a SELL basket, price must have RISEN to scale in higher
                price_movement = current_price - last_price

            if price_movement < min_distance:
                log.debug(f"GridScaler: {symbol} spacing insufficient — "
                          f"need {min_distance:.5f}, got {price_movement:.5f}")
                return False

        return True

    # ── State updates ────────────────────────────────────────────────────────

    def on_scale(self, symbol: str, direction: str, price: float, ticket: int) -> int:
        """
        Record a scale-in. Returns the level number (1-based).
        Raises RuntimeError if can_scale() would return False.
        """
        if not self.can_scale(symbol, direction, price):
            raise RuntimeError(
                f"GridScaler: cannot scale {direction} on {symbol} — checks failed"
            )

        existing = self._scales.get(symbol, [])
        level = len(existing) + 1
        lot = self.base_lot * (self.lot_multiplier ** (level - 1))

        record = _ScaleRecord(
            level=level, direction=direction.upper(),
            price=price, lot_size=lot, ticket=ticket,
        )
        self._scales.setdefault(symbol, []).append(record)
        log.info(f"GridScaler: {symbol} scaled {direction} level {level}/{self.max_levels} "
                 f"@ {price:.5f} lot={lot:.2f} ticket={ticket}")
        return level

    def on_close(self, symbol: str, ticket: int) -> None:
        """Remove a closed position from tracking."""
        existing = self._scales.get(symbol, [])
        before = len(existing)
        self._scales[symbol] = [s for s in existing if s.ticket != ticket]
        after = len(self._scales[symbol])
        if after == 0:
            del self._scales[symbol]
        if before == after:
            log.warning(f"GridScaler: ticket {ticket} not found for {symbol}")
        else:
            log.info(f"GridScaler: closed {symbol} ticket={ticket} ({after} remaining)")

    def close_all(self, symbol: str) -> None:
        """Clear all tracking for a symbol (after closing the entire basket)."""
        if symbol in self._scales:
            n = len(self._scales[symbol])
            del self._scales[symbol]
            log.info(f"GridScaler: cleared {n} positions for {symbol}")

    # ── Introspection ────────────────────────────────────────────────────────

    def get_levels(self, symbol: str) -> int:
        """Return the current number of scale levels for `symbol`."""
        return len(self._scales.get(symbol, []))

    def get_remaining_levels(self, symbol: str) -> int:
        """Return how many more scale-ins are allowed."""
        return self.max_levels - self.get_levels(symbol)

    def get_average_price(self, symbol: str) -> Optional[float]:
        """
        Return the volume-weighted average entry price for the basket.
        None if no positions open.
        """
        existing = self._scales.get(symbol, [])
        if not existing:
            return None
        total_value = sum(s.price * s.lot_size for s in existing)
        total_lots = sum(s.lot_size for s in existing)
        return total_value / total_lots if total_lots > 0 else None

    def get_total_lots(self, symbol: str) -> float:
        """Return the total lot size across all scale levels."""
        return sum(s.lot_size for s in self._scales.get(symbol, []))

    def get_basket_state(self, symbol: str) -> dict:
        """Return full state of the basket on `symbol`."""
        existing = self._scales.get(symbol, [])
        return {
            "symbol": symbol,
            "direction": existing[0].direction if existing else None,
            "levels": len(existing),
            "max_levels": self.max_levels,
            "remaining": self.max_levels - len(existing),
            "average_price": self.get_average_price(symbol),
            "total_lots": self.get_total_lots(symbol),
            "positions": [
                {"level": s.level, "price": s.price, "lot": s.lot_size,
                 "ticket": s.ticket, "time": s.timestamp}
                for s in existing
            ],
        }

    def get_all_symbols(self) -> list[str]:
        """Return all symbols with active grid baskets."""
        return list(self._scales.keys())


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scaler = ControlledGridScaler(
        max_levels=3, lot_size=0.1, lot_multiplier=1.0,
        min_spacing_pips=20, pip_size=0.0001, max_basket_loss_usd=100.0,
    )

    # Initial entry at 1.0850
    assert scaler.can_scale("EURUSD", "BUY", 1.0850, current_loss=0)
    level = scaler.on_scale("EURUSD", "BUY", 1.0850, ticket=1)
    assert level == 1
    assert scaler.get_levels("EURUSD") == 1

    # Price only drops 5 pips — not enough spacing (need 20)
    assert not scaler.can_scale("EURUSD", "BUY", 1.0845, current_loss=-5)

    # Price drops 25 pips — now can scale
    assert scaler.can_scale("EURUSD", "BUY", 1.0825, current_loss=-15)
    level = scaler.on_scale("EURUSD", "BUY", 1.0825, ticket=2)
    assert level == 2
    assert scaler.get_levels("EURUSD") == 2

    # Average price should be (1.0850 + 1.0825) / 2 = 1.08375
    avg = scaler.get_average_price("EURUSD")
    assert abs(avg - 1.08375) < 1e-6, f"got {avg}"

    # Try opposite direction — blocked
    assert not scaler.can_scale("EURUSD", "SELL", 1.0820, current_loss=-20)

    # Basket loss exceeds limit — blocked
    assert not scaler.can_scale("EURUSD", "BUY", 1.0800, current_loss=-105)

    # Close one position
    scaler.on_close("EURUSD", ticket=1)
    assert scaler.get_levels("EURUSD") == 1

    # Close all
    scaler.close_all("EURUSD")
    assert scaler.get_levels("EURUSD") == 0
    assert scaler.get_average_price("EURUSD") is None

    # Test lot_multiplier (capped at 1.5)
    scaler2 = ControlledGridScaler(max_levels=3, lot_size=0.1, lot_multiplier=1.2)
    scaler2.on_scale("EURUSD", "BUY", 1.0850, ticket=1)
    scaler2.on_scale("EURUSD", "BUY", 1.0820, ticket=2)  # 30 pips drop
    state = scaler2.get_basket_state("EURUSD")
    assert state["positions"][0]["lot"] == 0.1
    assert abs(state["positions"][1]["lot"] - 0.12) < 1e-6  # 0.1 * 1.2
    print(f"Multiplier test: lots = {[p['lot'] for p in state['positions']]}")

    # Test multiplier cap
    try:
        ControlledGridScaler(lot_multiplier=2.0)
        assert False, "should have raised"
    except ValueError:
        pass

    print(f"\nBasket state example: {scaler2.get_basket_state('EURUSD')}")
    print("\nControlledGridScaler smoke test passed.")
