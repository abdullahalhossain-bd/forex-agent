"""
analysis/breaker_block.py — Breaker Block Detector (Day 81+)
==============================================================

Masterclass concept (ICT):
  A Breaker Block is a failed Order Block that flips from support to
  resistance (or vice versa). When price breaks through an order block
  instead of respecting it, that block becomes a Breaker Block — a
  strong reversal signal.

Bullish Breaker Block:
  1. Price was in downtrend
  2. Last bullish order block (support) gets broken (price closes below)
  3. Price reverses up and retests that broken block
  4. The broken support now acts as resistance → but wait, for BULLISH
     breaker, we need: bearish OB breaks UP, then acts as support on retest

Bearish Breaker Block:
  1. Price was in uptrend
  2. Last bearish order block (resistance) gets broken (price closes above)
  3. Price reverses down and retests that broken block
  4. The broken resistance now acts as support

Simplified detection logic:
  - Find the most recent Order Block that was BROKEN (price closed beyond it)
  - If price has since returned to retest it → Breaker Block active
  - Bullish breaker: broken bearish OB now acting as support
  - Bearish breaker: broken bullish OB now acting as resistance

Usage:
    from analysis.breaker_block import BreakerBlockDetector

    detector = BreakerBlockDetector()
    breakers = detector.detect(df, order_blocks)
    # → [{"type": "bullish_breaker", "zone_top": ..., "zone_bottom": ..., "active": True}]
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from utils.logger import get_logger

log = get_logger("breaker_block")


class BreakerBlockDetector:
    """
    Detects Breaker Blocks from failed Order Blocks.

    A Breaker Block is the ICT concept where a broken order block
    flips its role — support becomes resistance, or resistance becomes
    support. This is a powerful reversal signal.
    """

    # How close price needs to be to the broken OB zone to count as "retest"
    RETEST_TOLERANCE_PCT = 0.3  # 0.3% of price

    def detect(self, df: pd.DataFrame, order_blocks: List[Dict]) -> List[Dict]:
        """
        Detect breaker blocks from a list of order blocks.

        Args:
            df:            OHLCV DataFrame
            order_blocks:  List of order block dicts with keys:
                           direction, zone_top, zone_bottom, timestamp

        Returns:
            List of breaker block dicts:
            [{
                "type": "bullish_breaker" | "bearish_breaker",
                "zone_top": float,
                "zone_bottom": float,
                "broken_at": str (timestamp),
                "retest_at": str | None,
                "active": bool,
            }]
        """
        if df is None or df.empty or not order_blocks:
            return []

        breakers: List[Dict] = []
        current_price = float(df["close"].iloc[-1])

        for ob in order_blocks:
            try:
                breaker = self._check_breaker(df, ob, current_price)
                if breaker:
                    breakers.append(breaker)
            except Exception as e:
                log.debug(f"Breaker check failed for OB {ob}: {e}")

        if breakers:
            log.info(f"[BreakerBlock] Detected {len(breakers)} breaker block(s)")

        return breakers

    def _check_breaker(
        self, df: pd.DataFrame, ob: Dict, current_price: float
    ) -> Dict | None:
        """Check if a single order block has become a breaker block."""
        ob_direction = ob.get("direction", "").upper()
        zone_top = float(ob.get("zone_top", 0))
        zone_bottom = float(ob.get("zone_bottom", 0))

        if zone_top <= 0 or zone_bottom <= 0:
            return None

        # Find candles AFTER the order block was formed
        ob_time = ob.get("timestamp") or ob.get("time")
        if ob_time:
            try:
                # Filter df to candles after OB formation
                if ob_time in df.index:
                    after_ob = df.loc[df.index > ob_time]
                else:
                    after_ob = df  # fallback: use all candles
            except Exception:
                after_ob = df
        else:
            after_ob = df

        if len(after_ob) < 3:
            return None

        # ── Bullish Breaker: bearish OB (resistance) broken UP ──────
        # A bearish OB was resistance. If price broke ABOVE it, then
        # came back down to retest it, it's now support → bullish breaker.
        if "BEARISH" in ob_direction:
            # Check if price broke above the OB zone
            broke_above = any(after_ob["close"] > zone_top)
            if not broke_above:
                return None

            # Find the break candle
            break_candles = after_ob[after_ob["close"] > zone_top]
            if break_candles.empty:
                return None

            break_time = str(break_candles.index[0])

            # Check if price has retested the zone (came back near it)
            tolerance = current_price * (self.RETEST_TOLERANCE_PCT / 100)
            retest_candles = after_ob[
                (after_ob["low"] <= zone_top + tolerance)
                & (after_ob["low"] >= zone_bottom - tolerance)
            ]

            is_active = bool(
                abs(current_price - zone_top) <= tolerance
                or abs(current_price - zone_bottom) <= tolerance
            )

            return {
                "type": "bullish_breaker",
                "zone_top": zone_top,
                "zone_bottom": zone_bottom,
                "broken_at": break_time,
                "retest_at": str(retest_candles.index[-1]) if not retest_candles.empty else None,
                "active": is_active,
                "original_ob_direction": ob_direction,
            }

        # ── Bearish Breaker: bullish OB (support) broken DOWN ──────
        # A bullish OB was support. If price broke BELOW it, then
        # came back up to retest it, it's now resistance → bearish breaker.
        if "BULLISH" in ob_direction:
            # Check if price broke below the OB zone
            broke_below = any(after_ob["close"] < zone_bottom)
            if not broke_below:
                return None

            break_candles = after_ob[after_ob["close"] < zone_bottom]
            if break_candles.empty:
                return None

            break_time = str(break_candles.index[0])

            # Check retest
            tolerance = current_price * (self.RETEST_TOLERANCE_PCT / 100)
            retest_candles = after_ob[
                (after_ob["high"] >= zone_bottom - tolerance)
                & (after_ob["high"] <= zone_top + tolerance)
            ]

            is_active = bool(
                abs(current_price - zone_top) <= tolerance
                or abs(current_price - zone_bottom) <= tolerance
            )

            return {
                "type": "bearish_breaker",
                "zone_top": zone_top,
                "zone_bottom": zone_bottom,
                "broken_at": break_time,
                "retest_at": str(retest_candles.index[-1]) if not retest_candles.empty else None,
                "active": is_active,
                "original_ob_direction": ob_direction,
            }

        return None

    def get_ai_context(self, breakers: List[Dict]) -> Dict:
        """Convert breaker blocks to AI context dict for MasterAnalyst."""
        if not breakers:
            return {"has_breaker": False, "breaker_type": "NONE"}

        active_breakers = [b for b in breakers if b.get("active")]
        if not active_breakers:
            return {"has_breaker": False, "breaker_type": "NONE"}

        # Use the most recent active breaker
        latest = active_breakers[-1]
        return {
            "has_breaker": True,
            "breaker_type": latest["type"],
            "breaker_zone_top": latest["zone_top"],
            "breaker_zone_bottom": latest["zone_bottom"],
            "breaker_broken_at": latest.get("broken_at"),
            "breaker_retest_at": latest.get("retest_at"),
            "breaker_signal": "BUY" if "bullish" in latest["type"] else "SELL",
        }
