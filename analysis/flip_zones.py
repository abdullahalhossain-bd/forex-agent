"""
analysis/flip_zones.py — Book 5 (Frank Miller S&D) Chapter 8 Flip Zones
========================================================================

Pages 89-90 introduce the **flip zone** concept, adapting classic
S/R role-reversal to the supply/demand framework:

  • When price BREAKS DOWN through a demand zone (support), the demand
    zone is INVALIDATED and reclassified as a new SUPPLY zone (resistance).
    Future retests of this level are now selling opportunities.

  • When price BREAKS UP through a supply zone (resistance), the supply
    zone is INVALIDATED and reclassified as a new DEMAND zone (support).
    Future retests of this level are now buying opportunities.

This is a state-transition rule: `zone.type` flips on confirmed break.
A "confirmed break" = a candle CLOSE beyond the zone's distal line
(not just a wick pierce).

The module exposes a `FlipZoneDetector` class that:
  • Maintains a registry of active zones with state (active / flipped / invalidated)
  • Scans new candles for confirmed breaks
  • On break, emits a FlipZoneEvent and reclassifies the zone
  • Returns the updated zone list for downstream consumers

Usage:
    from analysis.flip_zones import FlipZoneDetector
    detector = FlipZoneDetector()
    detector.register_zone(zone_dict)
    events = detector.update(df)  # process new candles
    # events: [{"zone_id":..., "old_type":"demand", "new_type":"supply",
    #           "break_idx":..., "break_price":...}, ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("flip_zones")


# ════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

@dataclass
class FlipZoneEvent:
    """Emitted when a zone's type flips due to a confirmed break."""
    zone_id: str
    old_type: str           # "demand" | "supply"
    new_type: str           # "supply" | "demand"
    break_idx: int          # candle index that confirmed the break
    break_price: float      # close price of the breaking candle
    break_direction: str    # "up" (broke resistance) | "down" (broke support)
    break_time: str = ""    # ISO timestamp if available


@dataclass
class ZoneState:
    """Tracks a zone's lifecycle state."""
    zone_id: str
    zone_type: str          # "demand" | "supply"
    proximal: float
    distal: float
    zone_low: float
    zone_high: float
    sd_pattern: str = ""
    state: str = "active"   # "active" | "flipped" | "invalidated"
    flip_count: int = 0     # how many times this zone has flipped
    registered_at_idx: int = 0
    flipped_at_idx: int = -1
    flipped_to_type: str = ""
    original_type: str = "" # preserved across flips

    def __post_init__(self):
        if not self.original_type:
            self.original_type = self.zone_type


# ════════════════════════════════════════════════════════════════
#  DETECTOR
# ════════════════════════════════════════════════════════════════

class FlipZoneDetector:
    """
    Book 5 Chapter 8 — Flip Zone detector.

    Maintains a registry of zones, scans OHLCV data for confirmed breaks,
    and emits FlipZoneEvent objects when a zone flips type.

    Confirmation rule (Book P89-90):
      A break is CONFIRMED when a candle CLOSES beyond the zone's distal line:
        - Demand zone (support): break confirmed when close < distal (lower edge)
        - Supply zone (resistance): break confirmed when close > distal (upper edge)
      A mere wick pierce is NOT a confirmed break — only the candle's close counts.
    """

    # A flipped zone becomes a NEW zone of the opposite type at the SAME price level.
    # The original zone is marked "flipped" (no longer tradeable as original type).
    # Optional: invalidate zones after N flips (default: keep flipping, the book
    # doesn't specify a maximum flip count).
    MAX_FLIPS_PER_ZONE = 4  # safety cap to prevent infinite flipping

    def __init__(self):
        self.zones: Dict[str, ZoneState] = {}
        self.events: List[FlipZoneEvent] = []

    # ══════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════

    def register_zone(
        self,
        zone: Dict[str, Any],
        current_idx: int = 0,
    ) -> str:
        """
        Add a zone to the flip-zone tracker.

        Args:
            zone       : zone dict with keys: zone_low, zone_high, distal,
                         proximal, sd_pattern (used to infer type)
            current_idx: candle index at which the zone was first detected

        Returns:
            zone_id (str)
        """
        zone_id = zone.get("zone_id") or str(uuid.uuid4())[:8]
        zone_type = self._infer_zone_type(zone)
        zs = ZoneState(
            zone_id=zone_id,
            zone_type=zone_type,
            proximal=float(zone.get("proximal", zone.get("zone_high", 0))),
            distal=float(zone.get("distal", zone.get("zone_low", 0))),
            zone_low=float(zone.get("zone_low", 0)),
            zone_high=float(zone.get("zone_high", 0)),
            sd_pattern=str(zone.get("sd_pattern", "")),
            registered_at_idx=current_idx,
        )
        self.zones[zone_id] = zs
        return zone_id

    def update(self, df: pd.DataFrame) -> List[FlipZoneEvent]:
        """
        Scan all candles in df for confirmed zone breaks.

        Idempotent: each zone can flip at most once per call (zones that
        flip are re-registered as the new type and continue to be tracked).
        Already-flipped zones are not re-flipped by the same candle.

        Args:
            df : OHLCV DataFrame

        Returns:
            List of FlipZoneEvent objects emitted during this scan.
        """
        if len(df) < 2:
            return []

        new_events: List[FlipZoneEvent] = []
        closes = df["close"].astype(float).values
        n = len(df)

        for zone_id, zs in list(self.zones.items()):
            if zs.state != "active":
                continue
            if zs.flip_count >= self.MAX_FLIPS_PER_ZONE:
                zs.state = "invalidated"
                continue

            # Determine the breakout price level
            # Demand zone (support) → break DOWN through distal (lower edge)
            # Supply zone (resistance) → break UP through distal (upper edge)
            if zs.zone_type == "demand":
                break_level = min(zs.distal, zs.proximal)
                break_dir = "down"
                new_type = "supply"
            else:  # supply
                break_level = max(zs.distal, zs.proximal)
                break_dir = "up"
                new_type = "demand"

            # Scan from registration index onwards for a CONFIRMED break
            # (candle CLOSE beyond the break level)
            for i in range(zs.registered_at_idx, n):
                c = float(closes[i])
                if break_dir == "down" and c < break_level:
                    # Demand zone broken down → flip to supply
                    evt = FlipZoneEvent(
                        zone_id=zone_id,
                        old_type=zs.zone_type,
                        new_type=new_type,
                        break_idx=i,
                        break_price=c,
                        break_direction=break_dir,
                        break_time=str(df.index[i]) if hasattr(df.index[i], "isoformat") else "",
                    )
                    new_events.append(evt)
                    # Update zone state
                    zs.state = "flipped"
                    zs.flipped_at_idx = i
                    zs.flipped_to_type = new_type
                    zs.flip_count += 1
                    # Register a NEW zone of the opposite type at the same level
                    flipped_zone_dict = {
                        "zone_id": f"{zone_id}-flip{zs.flip_count}",
                        "zone_low": zs.zone_low,
                        "zone_high": zs.zone_high,
                        "distal": zs.distal,
                        "proximal": zs.proximal,
                        "sd_pattern": f"Flip-{zs.sd_pattern}" if zs.sd_pattern else "Flip",
                        "type": new_type,
                    }
                    self.register_zone(flipped_zone_dict, current_idx=i)
                    # Mark the new zone's type explicitly (since pattern may be ambiguous)
                    self.zones[flipped_zone_dict["zone_id"]].zone_type = new_type
                    break  # only one flip per zone per scan
                elif break_dir == "up" and c > break_level:
                    # Supply zone broken up → flip to demand
                    evt = FlipZoneEvent(
                        zone_id=zone_id,
                        old_type=zs.zone_type,
                        new_type=new_type,
                        break_idx=i,
                        break_price=c,
                        break_direction=break_dir,
                        break_time=str(df.index[i]) if hasattr(df.index[i], "isoformat") else "",
                    )
                    new_events.append(evt)
                    zs.state = "flipped"
                    zs.flipped_at_idx = i
                    zs.flipped_to_type = new_type
                    zs.flip_count += 1
                    flipped_zone_dict = {
                        "zone_id": f"{zone_id}-flip{zs.flip_count}",
                        "zone_low": zs.zone_low,
                        "zone_high": zs.zone_high,
                        "distal": zs.distal,
                        "proximal": zs.proximal,
                        "sd_pattern": f"Flip-{zs.sd_pattern}" if zs.sd_pattern else "Flip",
                        "type": new_type,
                    }
                    self.register_zone(flipped_zone_dict, current_idx=i)
                    self.zones[flipped_zone_dict["zone_id"]].zone_type = new_type
                    break

        self.events.extend(new_events)
        if new_events:
            log.info(
                f"[FlipZones] {len(new_events)} flip event(s) detected — "
                f"total zones tracked: {len(self.zones)}"
            )
        return new_events

    def get_active_zones(self, zone_type: Optional[str] = None) -> List[ZoneState]:
        """Return all currently-active zones (optionally filtered by type)."""
        result = [zs for zs in self.zones.values() if zs.state == "active"]
        if zone_type:
            result = [zs for zs in result if zs.zone_type == zone_type]
        return result

    def get_flipped_zones(self) -> List[ZoneState]:
        """Return all zones that have flipped at least once."""
        return [zs for zs in self.zones.values() if zs.state == "flipped"]

    def get_events(self) -> List[FlipZoneEvent]:
        """Return the complete event history."""
        return list(self.events)

    def clear(self):
        """Reset the detector."""
        self.zones.clear()
        self.events.clear()

    # ══════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _infer_zone_type(zone: Dict[str, Any]) -> str:
        """
        Infer 'demand' or 'supply' from zone dict.
        Same logic as odd_enhancers._is_supply_zone but returns the string.
        """
        # Explicit type field wins
        zone_type = str(zone.get("type", "")).lower()
        if "supply" in zone_type:
            return "supply"
        if "demand" in zone_type:
            return "demand"

        # Pattern-based: check SUFFIX (what happens AFTER the base)
        sd_pattern = str(zone.get("sd_pattern", "")).lower()
        if sd_pattern.endswith("drop"):
            return "supply"
        if sd_pattern.endswith("rally"):
            return "demand"

        # Fallback: presence-based
        if "supply" in sd_pattern:
            return "supply"
        if "demand" in sd_pattern:
            return "demand"

        # Default to demand (arbitrary but consistent)
        return "demand"


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY (smoke test)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("=" * 64)
    print("  FLIP ZONE DETECTOR — Book 5 Chapter 8")
    print("=" * 64)

    # Build synthetic OHLCV
    np.random.seed(42)
    n = 30
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.linspace(1.0850, 1.0870, n) + np.random.randn(n) * 0.0003
    high = close + np.abs(np.random.randn(n)) * 0.0005
    low = close - np.abs(np.random.randn(n)) * 0.0005
    opn = close + np.random.randn(n) * 0.0002
    df = pd.DataFrame({"open": opn, "high": high, "low": low, "close": close},
                      index=dates)

    # Register a DEMAND zone at 1.0840-1.0850
    # Wait — current price starts at 1.0850, so the zone is right at the start.
    # Let's make price BREAK DOWN through 1.0840 at candle 15.
    for i in range(15, 25):
        df.iloc[i, df.columns.get_loc("close")] = 1.0830 - (i - 15) * 0.0005
        df.iloc[i, df.columns.get_loc("open")] = 1.0845 - (i - 15) * 0.0003
        df.iloc[i, df.columns.get_loc("high")] = 1.0850 - (i - 15) * 0.0002
        df.iloc[i, df.columns.get_loc("low")] = 1.0828 - (i - 15) * 0.0005

    detector = FlipZoneDetector()
    zone_id = detector.register_zone(
        {
            "zone_id": "test-demand-1",
            "zone_low": 1.0840,
            "zone_high": 1.0850,
            "distal": 1.0840,   # lower edge for demand
            "proximal": 1.0850,
            "sd_pattern": "Drop-Base-Rally",
        },
        current_idx=0,
    )
    print(f"\nRegistered demand zone {zone_id} at 1.0840-1.0850")

    events = detector.update(df)
    print(f"\nFlip events detected: {len(events)}")
    for evt in events:
        print(f"  Zone {evt.zone_id}: {evt.old_type} → {evt.new_type} "
              f"(broke {evt.break_direction} at idx {evt.break_idx}, "
              f"price {evt.break_price:.5f})")

    active = detector.get_active_zones()
    print(f"\nActive zones: {len(active)}")
    for zs in active:
        print(f"  {zs.zone_id}: {zs.zone_type} at {zs.zone_low}-{zs.zone_high} "
              f"(state={zs.state}, flips={zs.flip_count})")

    print("\n" + "=" * 64)
    print("  Flip zone detector smoke test complete.")
    print("=" * 64)