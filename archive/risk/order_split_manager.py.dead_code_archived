# risk/order_split_manager.py — Order splitting + TP2 trailing stop manager
# =============================================================================
# Ported from: https://github.com/codedpro/mt5-trade-split-manager
# Original: bulk-add-signals.mq5 (order splitting + trailing) + server.py (API)
# Original author: codedpro — MIT license
#
# Splits a single trade signal into multiple positions with configurable volume
# distribution and take-profit levels. When TP2 is reached, automatically
# moves the stop-loss to TP1 (breakeven) for all remaining positions.
#
# Example: BUY 1.0 lot EURUSD with 5 TPs at 30-pip intervals
#   → 5 positions: 0.6, 0.1, 0.1, 0.1, 0.1 lots
#   → TP1=1.0880, TP2=1.0910, TP3=1.0940, TP4=1.0970, TP5=1.1000
#   → When TP2 fills: SL for positions 3,4,5 moves to TP1 (1.0880)
#
# Default volume split for N levels:
#   N=1 → [1.0]
#   N≥2 → [0.60, 0.40/(N-1), 0.40/(N-1), ...] (e.g., 60/10/10/10/10 for N=5)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import json
from pathlib import Path

from utils.logger import get_logger

log = get_logger("order_split_manager")


def default_volume_split(n: int) -> list[float]:
    """
    Default volume split for N TP levels.
    N=1 → [1.0]
    N≥2 → TP1=0.60, each remaining = 0.40/(N-1)
    For N=5 this gives the original 60/10/10/10/10.
    """
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    return [0.60] + [0.40 / (n - 1)] * (n - 1)


def validate_volume_split(split: list[float], n_tps: int) -> bool:
    """
    Validate a volume split array.
    - Length must match n_tps
    - All entries >= 0
    - At least one entry > 0
    - Sum ≈ 1.0 (within 0.99..1.01)
    """
    if len(split) != n_tps:
        return False
    if any(x < 0 for x in split):
        return False
    if not any(x > 0 for x in split):
        return False
    total = sum(split)
    return 0.99 <= total <= 1.01


@dataclass
class SplitPosition:
    """A single position within a split order group."""
    ticket: int = 0               # broker ticket (0 = not yet placed)
    tp_level: int = 0             # which TP level (1-indexed: 1, 2, 3, ...)
    tp_price: float = 0.0         # the TP price for this position
    volume: float = 0.0           # lot size for this position
    is_filled: bool = False       # has this pending order been filled?
    is_closed: bool = False       # has this position been closed (TP hit)?


@dataclass
class OrderGroup:
    """A group of positions sharing a single entry signal."""
    group_id: str
    symbol: str
    direction: str                # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    lot_size: float               # total lot size
    positions: list[SplitPosition] = field(default_factory=list)
    tp2_reached: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def is_all_closed(self) -> bool:
        """True when every position in the group is closed."""
        return all(p.is_closed for p in self.positions) and len(self.positions) > 0

    def get_position_by_tp(self, tp_level: int) -> Optional[SplitPosition]:
        for p in self.positions:
            if p.tp_level == tp_level:
                return p
        return None


class OrderSplitManager:
    """
    Manages order splitting and TP2 trailing stop logic.

    Usage:
        manager = OrderSplitManager()

        # Create a split order
        group = manager.create_split_order(
            symbol="EURUSD", direction="BUY", entry_price=1.0850,
            stop_loss=1.0820, lot_size=1.0,
            tp_levels=[1.0880, 1.0910, 1.0940, 1.0970, 1.1000],
            volume_split=None,  # auto: 60/10/10/10/10
        )

        # Simulate fills (in live trading, broker fills these)
        manager.mark_position_filled(group.group_id, tp_level=1, ticket=1001)
        manager.mark_position_filled(group.group_id, tp_level=2, ticket=1002)

        # When TP2 closes:
        manager.mark_position_closed(group.group_id, tp_level=2)
        # → SL for TP3, TP4, TP5 automatically moves to TP1 price

        # Check trailing
        trailing = manager.get_trailing_actions(group.group_id)
        # → [{"ticket": 1003, "new_sl": 1.0880}, ...]
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._groups: dict[str, OrderGroup] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self.load()

    def create_split_order(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        lot_size: float,
        tp_levels: list[float],
        volume_split: Optional[list[float]] = None,
    ) -> OrderGroup:
        """
        Create a split order group.

        Parameters
        ----------
        symbol : trading symbol.
        direction : "BUY" or "SELL".
        entry_price : entry price (for reference).
        stop_loss : initial stop-loss price.
        lot_size : total lot size to split across positions.
        tp_levels : list of 1-10 take-profit prices.
        volume_split : optional volume weights (must sum to ~1.0).
            If None, uses default_volume_split().

        Returns
        -------
        OrderGroup with SplitPosition entries.
        """
        direction = direction.upper()
        if direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY or SELL, got {direction!r}")
        if not 1 <= len(tp_levels) <= 10:
            raise ValueError(f"tp_levels must contain 1-10 prices, got {len(tp_levels)}")

        if volume_split is None:
            volume_split = default_volume_split(len(tp_levels))
        elif not validate_volume_split(volume_split, len(tp_levels)):
            raise ValueError(
                f"Invalid volume_split: must have {len(tp_levels)} entries, "
                f"all >= 0, at least one > 0, sum ~1.0"
            )

        group_id = f"GRP_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{len(self._groups)}"
        positions = []
        for i, (tp_price, vol_frac) in enumerate(zip(tp_levels, volume_split)):
            pos = SplitPosition(
                tp_level=i + 1,
                tp_price=tp_price,
                volume=round(lot_size * vol_frac, 2),
            )
            positions.append(pos)

        group = OrderGroup(
            group_id=group_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            lot_size=lot_size,
            positions=positions,
        )
        self._groups[group_id] = group
        log.info(f"Created split order {group_id}: {direction} {symbol} "
                 f"{lot_size} lots → {len(positions)} positions, SL={stop_loss}")
        self._save_if_needed()
        return group

    def mark_position_filled(self, group_id: str, tp_level: int, ticket: int) -> None:
        """Record that a pending order was filled (position opened)."""
        group = self._groups.get(group_id)
        if not group:
            raise KeyError(f"Group {group_id} not found")
        pos = group.get_position_by_tp(tp_level)
        if not pos:
            raise KeyError(f"TP level {tp_level} not found in group {group_id}")
        pos.ticket = ticket
        pos.is_filled = True
        log.info(f"Position filled: {group_id} TP{tp_level} ticket={ticket}")
        self._save_if_needed()

    def mark_position_closed(self, group_id: str, tp_level: int) -> list[dict]:
        """
        Mark a position as closed (TP hit). If TP2 is closed, triggers
        trailing stop: moves SL to TP1 for all remaining open positions.

        Returns list of trailing actions: [{"ticket": int, "new_sl": float}, ...]
        """
        group = self._groups.get(group_id)
        if not group:
            raise KeyError(f"Group {group_id} not found")
        pos = group.get_position_by_tp(tp_level)
        if not pos:
            raise KeyError(f"TP level {tp_level} not found in group {group_id}")

        pos.is_closed = True
        log.info(f"Position closed: {group_id} TP{tp_level}")

        trailing_actions = []

        # TP2 trailing logic: when TP2 closes, move SL to TP1 for remaining
        if tp_level == 2 and not group.tp2_reached:
            group.tp2_reached = True
            tp1_price = group.positions[0].tp_price
            new_sl = tp1_price if tp1_price > 0 else group.entry_price

            for p in group.positions:
                if p.tp_level > 2 and p.is_filled and not p.is_closed and p.ticket > 0:
                    trailing_actions.append({
                        "ticket": p.ticket,
                        "new_sl": new_sl,
                        "tp_level": p.tp_level,
                        "old_sl": group.stop_loss,
                    })
                    log.info(f"Trailing SL → TP1 for {group_id} TP{p.tp_level} "
                             f"ticket={p.ticket}: SL {group.stop_loss} → {new_sl}")

        # Clean up if all closed
        if group.is_all_closed():
            log.info(f"All positions closed for group {group_id}")
            # Don't delete — keep for audit. Just mark.

        self._save_if_needed()
        return trailing_actions

    def get_trailing_actions(self, group_id: str) -> list[dict]:
        """Get pending trailing actions (if TP2 was reached but SL not yet moved)."""
        group = self._groups.get(group_id)
        if not group or not group.tp2_reached:
            return []

        tp1_price = group.positions[0].tp_price
        new_sl = tp1_price if tp1_price > 0 else group.entry_price
        actions = []
        for p in group.positions:
            if p.tp_level > 2 and p.is_filled and not p.is_closed and p.ticket > 0:
                actions.append({
                    "ticket": p.ticket,
                    "new_sl": new_sl,
                    "tp_level": p.tp_level,
                })
        return actions

    def get_group(self, group_id: str) -> Optional[OrderGroup]:
        return self._groups.get(group_id)

    def get_all_groups(self) -> list[OrderGroup]:
        return list(self._groups.values())

    def get_active_groups(self) -> list[OrderGroup]:
        """Groups with at least one open position."""
        return [g for g in self._groups.values()
                if any(p.is_filled and not p.is_closed for p in g.positions)]

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_if_needed(self) -> None:
        if self._persist_path:
            self.save()

    def save(self) -> None:
        """Save group state to JSON file."""
        if not self._persist_path:
            return
        data = []
        for g in self._groups.values():
            data.append({
                "group_id": g.group_id,
                "symbol": g.symbol,
                "direction": g.direction,
                "entry_price": g.entry_price,
                "stop_loss": g.stop_loss,
                "lot_size": g.lot_size,
                "tp2_reached": g.tp2_reached,
                "created_at": g.created_at,
                "positions": [
                    {"ticket": p.ticket, "tp_level": p.tp_level,
                     "tp_price": p.tp_price, "volume": p.volume,
                     "is_filled": p.is_filled, "is_closed": p.is_closed}
                    for p in g.positions
                ],
            })
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(json.dumps(data, indent=2))
        log.debug(f"Saved {len(data)} groups to {self._persist_path}")

    def load(self) -> None:
        """Load group state from JSON file."""
        if not self._persist_path or not self._persist_path.exists():
            return
        data = json.loads(self._persist_path.read_text())
        for g_data in data:
            positions = [SplitPosition(**p) for p in g_data.pop("positions", [])]
            group = OrderGroup(**g_data, positions=positions)
            self._groups[group.group_id] = group
        log.info(f"Loaded {len(self._groups)} groups from {self._persist_path}")


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    manager = OrderSplitManager()

    # Create a 5-TP split order
    group = manager.create_split_order(
        symbol="EURUSD", direction="BUY", entry_price=1.0850,
        stop_loss=1.0820, lot_size=1.0,
        tp_levels=[1.0880, 1.0910, 1.0940, 1.0970, 1.1000],
    )

    print(f"Group: {group.group_id}")
    print(f"Positions ({len(group.positions)}):")
    for p in group.positions:
        print(f"  TP{p.tp_level}: price={p.tp_price:.5f} vol={p.volume:.2f}")

    # Check default split (60/10/10/10/10)
    assert group.positions[0].volume == 0.60
    assert all(group.positions[i].volume == 0.10 for i in range(1, 5))

    # Simulate fills
    for i, p in enumerate(group.positions, 1):
        manager.mark_position_filled(group.group_id, i, ticket=1000+i)

    # Close TP1
    manager.mark_position_closed(group.group_id, 1)
    assert not group.tp2_reached

    # Close TP2 → triggers trailing
    actions = manager.mark_position_closed(group.group_id, 2)
    assert group.tp2_reached
    assert len(actions) == 3  # TP3, TP4, TP5 get trailing
    assert actions[0]["new_sl"] == 1.0880  # SL moves to TP1

    print(f"\nTrailing actions after TP2:")
    for a in actions:
        print(f"  Ticket {a['ticket']} (TP{a['tp_level']}): SL → {a['new_sl']:.5f}")

    # Custom volume split
    group2 = manager.create_split_order(
        symbol="XAUUSD", direction="SELL", entry_price=4100,
        stop_loss=4150, lot_size=2.0,
        tp_levels=[4080, 4060, 4040],
        volume_split=[0.5, 0.3, 0.2],
    )
    assert group2.positions[0].volume == 1.0  # 2.0 * 0.5
    assert group2.positions[1].volume == 0.6  # 2.0 * 0.3
    assert group2.positions[2].volume == 0.4  # 2.0 * 0.2

    # Single TP
    group3 = manager.create_split_order(
        symbol="GBPUSD", direction="BUY", entry_price=1.2700,
        stop_loss=1.2650, lot_size=0.5,
        tp_levels=[1.2750],
    )
    assert len(group3.positions) == 1
    assert group3.positions[0].volume == 0.5

    # Validation
    try:
        manager.create_split_order("EURUSD", "BUY", 1.0, 0.9, 1.0, [])
        assert False
    except ValueError:
        pass

    print(f"\nActive groups: {len(manager.get_active_groups())}")
    print(f"Total groups: {len(manager.get_all_groups())}")

    print("\nOrder split manager smoke test passed.")
