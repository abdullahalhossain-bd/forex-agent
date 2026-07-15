"""
analysis/microstructure.py — Day 97 Tick Microstructure Engine
================================================================
Analyzes tick-level market data from MT5 to detect:

  1. Tick speed anomalies (sudden burst = institutional activity)
  2. Spread expansion (widening = low liquidity / news)
  3. Volume burst (unusual tick volume = big orders)
  4. Price acceleration (rapid directional move = displacement)
"""
from __future__ import annotations

import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, Optional

from utils.logger import get_logger

log = get_logger("microstructure")


class MicrostructureEngine:
    """Tick-level market microstructure analyzer (MT5 native)."""

    TICK_SPEED_NORMAL_MIN = 0.5
    TICK_SPEED_BURST_MULT = 3.0
    SPREAD_WIDE_MULT = 2.0
    SPREAD_EXTREME_MULT = 5.0
    VOLUME_BURST_MULT = 3.0
    BASELINE_TICKS = 100

    def __init__(self):
        self._tick_cache: Dict[str, Deque] = {}

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze tick microstructure for a symbol."""
        tick_data = self._fetch_ticks(symbol)

        if tick_data is None:
            return self._fallback_result(symbol, "MT5 unavailable for tick data")

        tick_speed_state = self._analyze_tick_speed(tick_data)
        spread_state = self._analyze_spread(tick_data, symbol)
        volume_state = self._analyze_volume(tick_data)
        acceleration = self._analyze_acceleration(tick_data, symbol)

        anomalies = sum([
            tick_speed_state == "BURST",
            spread_state in ("WIDE", "EXTREME"),
            volume_state == "BURST",
        ])
        liquidity_event = anomalies >= 2

        if liquidity_event or spread_state == "EXTREME":
            recommendation = "AVOID"
        elif anomalies >= 1 or tick_speed_state == "DEAD":
            recommendation = "CAUTION"
        else:
            recommendation = "PROCEED"

        result = {
            "symbol":             symbol,
            "source":             "mt5_ticks",
            "tick_speed_state":   tick_speed_state,
            "spread_state":       spread_state,
            "volume_state":       volume_state,
            "acceleration_pips":  round(acceleration, 2),
            "liquidity_event":    liquidity_event,
            "anomaly_count":      anomalies,
            "recommendation":     recommendation,
            "timestamp":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        log.info(
            f"[Microstructure] {symbol} | ticks={tick_speed_state} "
            f"spread={spread_state} vol={volume_state} accel={acceleration:.1f}p/s | "
            f"anomalies={anomalies} → {recommendation}"
        )
        return result

    # ─────────────────────────────────────────────────────────
    # TICK DATA FETCH — FIX: numpy.void has no .time attribute
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_tick_field(t, field_name: str, index: int, default=0):
        """Safely extract a field from a tick — handles both numpy.void and dict."""
        # Try attribute access first (structured numpy array row)
        try:
            val = getattr(t, field_name, None)
            if val is not None:
                return val
        except Exception as e:
            log.debug(f"[microstructure] suppressed: {e}")
            pass
        # Try dict-style access
        try:
            if hasattr(t, "__getitem__"):
                return t[field_name]
        except Exception as e:
            log.debug(f"[microstructure] suppressed: {e}")
            pass
        # Try positional index (numpy.void indexed by position)
        try:
            if hasattr(t, "__len__") and len(t) > index:
                return t[index]
        except Exception as e:
            log.debug(f"[microstructure] suppressed: {e}")
            pass
        return default

    @staticmethod
    def _fetch_ticks(symbol: str) -> Optional[list]:
        """Fetch recent ticks from MT5.

        FIX: MT5 copy_ticks_range returns numpy structured array.
        numpy.void rows don't support attribute access like t.time —
        must use t['time'] or index-based access.
        """
        try:
            from broker.mt5_connection import MT5_AVAILABLE
            if not MT5_AVAILABLE:
                return None

            import MetaTrader5 as mt5
            # CRITICAL FIX: Do NOT call mt5.shutdown() here!
            # mt5.shutdown() kills the MT5 connection for the ENTIRE process.
            # The main trading loop also uses mt5 — if this function runs during
            # a trading cycle, it will disconnect the broker and all subsequent
            # order operations will fail.
            # Instead, just use the existing connection (mt5.initialize() is
            # idempotent — if already initialized, it returns True without
            # re-initializing).
            if not mt5.initialize():
                return None

            utc_to = datetime.now(timezone.utc)
            utc_from = utc_to - timedelta(seconds=60)
            ticks = mt5.copy_ticks_range(symbol, utc_from, utc_to, mt5.COPY_TICKS_ALL)
            # DO NOT CALL mt5.shutdown() — leave connection alive for main loop

            if ticks is None or len(ticks) == 0:
                return None

            result = []
            for t in ticks:
                try:
                    # FIX: use dict-style access for numpy structured array rows
                    # numpy.void supports t['field'] but NOT t.field reliably
                    tick_dict = {
                        "time":   int(t["time"]),
                        "bid":    float(t["bid"]),
                        "ask":    float(t["ask"]),
                        "last":   float(t["last"]) if "last" in t.dtype.names else 0.0,
                        "volume": float(t["volume_real"]) if "volume_real" in t.dtype.names
                                  else float(t["volume"]) if "volume" in t.dtype.names else 0.0,
                        "flags":  int(t["flags"]) if "flags" in t.dtype.names else 0,
                    }
                    result.append(tick_dict)
                except Exception as e:
                    # Skip malformed ticks rather than crash
                    log.debug(f"[Microstructure] skipping malformed tick: {e}")
                    continue

            return result if result else None

        except Exception as e:
            log.debug(f"[Microstructure] tick fetch failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    # ANALYZERS
    # ─────────────────────────────────────────────────────────

    def _analyze_tick_speed(self, ticks: list) -> str:
        """Classify tick arrival speed."""
        if len(ticks) < 5:
            return "DEAD"

        time_span = ticks[-1]["time"] - ticks[0]["time"]
        if time_span <= 0:
            return "NORMAL"

        tps = len(ticks) / time_span

        if tps < self.TICK_SPEED_NORMAL_MIN:
            return "DEAD"

        midpoint = len(ticks) // 2
        if midpoint > 0:
            baseline_span = ticks[midpoint]["time"] - ticks[0]["time"]
            if baseline_span > 0:
                baseline_tps = midpoint / baseline_span
                if tps > baseline_tps * self.TICK_SPEED_BURST_MULT:
                    return "BURST"

        return "NORMAL"

    def _analyze_spread(self, ticks: list, symbol: str) -> str:
        """Classify spread state."""
        if len(ticks) < 5:
            return "NORMAL"

        spreads = [(t["ask"] - t["bid"]) for t in ticks if t["ask"] > 0 and t["bid"] > 0]
        if not spreads:
            return "NORMAL"

        current_spread = spreads[-1]
        avg_spread = sum(spreads) / len(spreads)

        if avg_spread == 0:
            return "NORMAL"

        ratio = current_spread / avg_spread

        if ratio >= self.SPREAD_EXTREME_MULT:
            return "EXTREME"
        if ratio >= self.SPREAD_WIDE_MULT:
            return "WIDE"
        return "NORMAL"

    def _analyze_volume(self, ticks: list) -> str:
        """Classify tick volume state."""
        volumes = [t.get("volume", 0) for t in ticks if t.get("volume", 0) > 0]
        if len(volumes) < 5:
            return "NORMAL"

        current_vol = volumes[-1]
        avg_vol = sum(volumes) / len(volumes)

        if avg_vol == 0:
            return "NORMAL"

        ratio = current_vol / avg_vol

        if ratio >= self.VOLUME_BURST_MULT:
            return "BURST"
        return "NORMAL"

    @staticmethod
    def _analyze_acceleration(ticks: list, symbol: str) -> float:
        """Calculate price acceleration in pips per second."""
        if len(ticks) < 2:
            return 0.0

        first = ticks[0]
        last = ticks[-1]

        first_mid = (first["bid"] + first["ask"]) / 2
        last_mid = (last["bid"] + last["ask"]) / 2

        time_diff = last["time"] - first["time"]
        if time_diff <= 0:
            return 0.0

        price_change = last_mid - first_mid
        pip_size = 0.01 if "JPY" in symbol else 0.0001
        pips_per_sec = (price_change / pip_size) / time_diff

        return pips_per_sec

    # ─────────────────────────────────────────────────────────
    # FALLBACK
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_result(symbol: str, reason: str) -> Dict[str, Any]:
        """When MT5 unavailable — return neutral (don't block trades)."""
        return {
            "symbol":             symbol,
            "source":             "fallback",
            "tick_speed_state":   "UNKNOWN",
            "spread_state":       "UNKNOWN",
            "volume_state":       "UNKNOWN",
            "acceleration_pips":  0.0,
            "liquidity_event":    False,
            "anomaly_count":      0,
            "recommendation":     "PROCEED",
            "reason":             reason,
            "timestamp":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # ─────────────────────────────────────────────────────────
    # AI CONTEXT
    # ─────────────────────────────────────────────────────────

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "micro_tick_state":      result.get("tick_speed_state", "UNKNOWN"),
            "micro_spread_state":    result.get("spread_state", "UNKNOWN"),
            "micro_volume_state":    result.get("volume_state", "UNKNOWN"),
            "micro_acceleration":    result.get("acceleration_pips", 0),
            "micro_liquidity_event": result.get("liquidity_event", False),
            "micro_recommendation":  result.get("recommendation", "PROCEED"),
        }

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 50
        log.info(bar)
        log.info("  🔬  MICROSTRUCTURE  (Day 97)")
        log.info(bar)
        log.info(f"  Symbol           : {result.get('symbol','?')}")
        log.info(f"  Source           : {result.get('source','?')}")
        log.info(f"  Tick speed       : {result.get('tick_speed_state','?')}")
        log.info(f"  Spread           : {result.get('spread_state','?')}")
        log.info(f"  Volume           : {result.get('volume_state','?')}")
        log.info(f"  Acceleration     : {result.get('acceleration_pips',0):.1f} pips/sec")
        log.info(f"  Liquidity event  : {'⛔ YES' if result.get('liquidity_event') else '✅ no'}")
        log.info(f"  Recommendation   : {result.get('recommendation','?')}")
        log.info(bar)


# ── Singleton ────────────────────────────────────────────────────

_INSTANCE: Optional[MicrostructureEngine] = None


def get_microstructure_engine() -> MicrostructureEngine:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = MicrostructureEngine()
    return _INSTANCE