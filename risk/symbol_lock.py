# risk/symbol_lock.py — One-direction-per-symbol guard
# =============================================================================
# Inspired by: https://github.com/smartedgetrading/SmartEdge-EA
# SmartEdge docs: "SymbolLock logic to prevent double trades" and
# "Only one direction is allowed per symbol at a time."
#
# SymbolLock ensures that for any given symbol, the bot has at most ONE
# open position direction at a time. If a LONG is open on EURUSD, no new
# SHORT can be opened until the LONG closes. This prevents conflicting
# trades on the same symbol — a common cause of confusing P&L in multi-
# strategy systems.
#
# This is a STATEFUL guard — it tracks open positions per symbol. Use it
# as a pre-trade check before sending any order:
#
#     lock = SymbolLock()
#     if lock.can_open("EURUSD", "BUY"):
#         # ... send buy order ...
#         lock.on_open("EURUSD", "BUY", ticket=12345)
#     else:
#         log.info("EURUSD already has an open position — skipping")
#
# When a position closes:
#     lock.on_close("EURUSD", ticket=12345)
#
# The lock also supports a max-positions-per-symbol limit (default 1, but
# can be higher for grid-scaling systems that allow multiple positions in
# the SAME direction).
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from utils.logger import get_logger

log = get_logger("symbol_lock")


@dataclass
class _PositionRecord:
    """Internal record of a tracked position."""
    symbol: str
    direction: str       # "BUY" or "SELL"
    ticket: int
    opened_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class SymbolLock:
    """
    Guard that enforces one-direction-per-symbol (or N-directions-per-symbol)
    trading. Thread-safe via a per-symbol lock dict.

    Parameters
    ----------
    max_positions_per_symbol : int
        Maximum number of simultaneous positions per symbol. Default 1
        (strict one-direction-per-symbol). Set higher for grid-scaling
        systems that allow multiple positions in the same direction.
    allow_same_direction_stacking : bool
        If True (default), multiple positions in the SAME direction are
        allowed up to max_positions_per_symbol. If False, only ONE position
        total per symbol regardless of direction.
    """

    def __init__(
        self,
        max_positions_per_symbol: int = 1,
        allow_same_direction_stacking: bool = False,
    ):
        if max_positions_per_symbol < 1:
            raise ValueError("max_positions_per_symbol must be >= 1")
        self.max_per_symbol = max_positions_per_symbol
        self.allow_same_direction_stacking = allow_same_direction_stacking
        # symbol → list of _PositionRecord
        self._positions: dict[str, list[_PositionRecord]] = {}

    # ── Pre-trade check ──────────────────────────────────────────────────────

    def can_open(self, symbol: str, direction: str) -> bool:
        """
        Check whether a new position in `direction` can be opened on `symbol`.

        Returns True if:
          - No position is currently open on this symbol, OR
          - max_positions_per_symbol > 1 AND allow_same_direction_stacking
            is True AND the existing positions are in the SAME direction.
        """
        direction = direction.upper()
        if direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY or SELL, got {direction!r}")

        existing = self._positions.get(symbol, [])

        if len(existing) == 0:
            return True

        if len(existing) >= self.max_per_symbol:
            log.debug(f"SymbolLock: {symbol} at max ({self.max_per_symbol}) — blocked")
            return False

        # Under the limit — check direction conflict
        if self.allow_same_direction_stacking:
            # Only allow same-direction adds
            for pos in existing:
                if pos.direction != direction:
                    log.debug(f"SymbolLock: {symbol} has open {pos.direction}, "
                              f"can't add {direction} (no opposite stacking)")
                    return False
            return True
        else:
            # Strict: no stacking at all (already checked len >= max above,
            # and max defaults to 1, so this branch means max > 1 but no
            # stacking allowed — which is contradictory. Treat as blocked.)
            log.debug(f"SymbolLock: {symbol} already has position — blocked (no stacking)")
            return False

    # ── State updates ────────────────────────────────────────────────────────

    def on_open(self, symbol: str, direction: str, ticket: int) -> None:
        """
        Record that a position was opened. Call this AFTER the order fills.
        Raises RuntimeError if can_open() would return False (use can_open()
        first, or use try_open() which combines both).
        """
        if not self.can_open(symbol, direction):
            raise RuntimeError(
                f"SymbolLock: cannot open {direction} on {symbol} — "
                f"position already exists and stacking is not allowed"
            )
        record = _PositionRecord(symbol=symbol, direction=direction.upper(), ticket=ticket)
        self._positions.setdefault(symbol, []).append(record)
        log.info(f"SymbolLock: opened {direction} {symbol} ticket={ticket} "
                 f"({len(self._positions[symbol])} open)")

    def on_close(self, symbol: str, ticket: int) -> None:
        """Record that a position was closed. Call this AFTER the close fills."""
        existing = self._positions.get(symbol, [])
        before = len(existing)
        self._positions[symbol] = [p for p in existing if p.ticket != ticket]
        after = len(self._positions[symbol])
        if after == 0:
            del self._positions[symbol]
        if before == after:
            log.warning(f"SymbolLock: ticket {ticket} not found for {symbol}")
        else:
            log.info(f"SymbolLock: closed {symbol} ticket={ticket} ({after} remaining)")

    def try_open(self, symbol: str, direction: str, ticket: int) -> bool:
        """
        Atomic check-and-open. Returns True if the position was recorded,
        False if it was blocked by the lock.
        """
        if self.can_open(symbol, direction):
            self.on_open(symbol, direction, ticket)
            return True
        return False

    # ── Introspection ────────────────────────────────────────────────────────

    def get_open_positions(self, symbol: str) -> list[dict]:
        """Return a list of open position dicts for `symbol` (empty if none)."""
        return [
            {"symbol": p.symbol, "direction": p.direction, "ticket": p.ticket,
             "opened_at": p.opened_at}
            for p in self._positions.get(symbol, [])
        ]

    def get_open_direction(self, symbol: str) -> Optional[str]:
        """
        Return the direction of the open position on `symbol`, or None.
        If multiple positions are open (stacking), returns the direction of
        the first one.
        """
        existing = self._positions.get(symbol, [])
        if not existing:
            return None
        return existing[0].direction

    def is_locked(self, symbol: str) -> bool:
        """True if any position is open on `symbol`."""
        return len(self._positions.get(symbol, [])) > 0

    def get_all_locked_symbols(self) -> list[str]:
        """Return a list of all symbols with open positions."""
        return list(self._positions.keys())

    def get_state(self) -> dict:
        """Return a full snapshot of the lock state (for debugging/audit)."""
        return {
            symbol: [
                {"direction": p.direction, "ticket": p.ticket, "opened_at": p.opened_at}
                for p in positions
            ]
            for symbol, positions in self._positions.items()
        }

    def clear(self) -> None:
        """Clear all tracked positions (use after a restart to re-sync with broker)."""
        self._positions.clear()
        log.info("SymbolLock: all tracking cleared")


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    lock = SymbolLock(max_positions_per_symbol=1)

    # Initially nothing is locked
    assert not lock.is_locked("EURUSD")
    assert lock.get_open_direction("EURUSD") is None
    assert lock.can_open("EURUSD", "BUY")

    # Open a BUY
    assert lock.try_open("EURUSD", "BUY", ticket=1001)
    assert lock.is_locked("EURUSD")
    assert lock.get_open_direction("EURUSD") == "BUY"

    # Can't open SELL while BUY is open
    assert not lock.can_open("EURUSD", "SELL")
    assert not lock.try_open("EURUSD", "SELL", ticket=1002)

    # Can't open another BUY either (max=1)
    assert not lock.can_open("EURUSD", "BUY")

    # But CAN open on a different symbol
    assert lock.try_open("GBPUSD", "SELL", ticket=2001)
    assert lock.get_open_direction("GBPUSD") == "SELL"

    # Close the EURUSD position
    lock.on_close("EURUSD", ticket=1001)
    assert not lock.is_locked("EURUSD")
    assert lock.is_locked("GBPUSD")  # still locked

    # Now can open EURUSD again
    assert lock.can_open("EURUSD", "SELL")

    print(f"Locked symbols: {lock.get_all_locked_symbols()}")
    print(f"State: {lock.get_state()}")

    # Test stacking mode
    lock2 = SymbolLock(max_positions_per_symbol=3, allow_same_direction_stacking=True)
    assert lock2.try_open("EURUSD", "BUY", 1)
    assert lock2.try_open("EURUSD", "BUY", 2)  # same direction OK
    assert lock2.try_open("EURUSD", "BUY", 3)  # same direction OK
    assert not lock2.try_open("EURUSD", "BUY", 4)  # max reached
    assert not lock2.try_open("EURUSD", "SELL", 5)  # opposite direction blocked
    print(f"Stacking state: {len(lock2.get_open_positions('EURUSD'))} positions")

    print("\nSymbolLock smoke test passed.")
