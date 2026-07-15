# analysis/stop_hunt_signal_engine.py
# ============================================================
# Zone-Based Stop Hunt Detector + Trade Signal Generator
# ============================================================
# Builds on top of analysis/support_resistance.py (v2 zones).
#
# Spec compliance (Steps 2 & 3 from user spec):
#
# STEP 2 — Stop Hunt / Liquidity Grab Detection:
#   A zone is flagged "Stop Hunt Confirmed" if ALL of:
#     1. Price pierced the zone with a WICK but the candle BODY
#        closed INSIDE or NEAR (within 0.3× zone width) the zone.
#     2. Wick is at least 1.5× – 2× the body.
#     3. Within the next 1–3 candles, price reversed back in the
#        opposite direction (close past zone boundary in reverse).
#     4. The spike occurred near a round number / equal high/low /
#        obvious swing point (sanity-checked, boosts confidence).
#
# STEP 3 — Trade Signal:
#   - Entry  : open of the candle AFTER stop-hunt confirmation,
#              in the reversal direction.
#   - SL     : just beyond the stop-hunt candle's wick extreme
#              (wick_extreme ± buffer).
#   - TP     : opposite-side nearest zone OR 1:2 R:R minimum —
#              whichever is hit first.
#   - If zone is broken but stop-hunt confirmation fails
#     (body closes outside zone) → NO_TRADE / "Wait for Retest".
#   - Insufficient data → NO_TRADE / "Insufficient data".
#   - Risk management priority: when unsure → NO_TRADE.
#
# Output JSON (exact spec schema):
#   {
#     "resistance_zones": [{zone_top, zone_bottom, touches, strength}],
#     "support_zones":    [{zone_top, zone_bottom, touches, strength}],
#     "stop_hunt_detected": bool,
#     "stop_hunt_zone": "support|resistance|null",
#     "signal": {
#       "action": "BUY|SELL|NO_TRADE",
#       "entry_price": float|null,
#       "stop_loss": float|null,
#       "take_profit": float|null,
#       "reason": str,
#       "confidence": "Low|Medium|High"
#     }
#   }
# ============================================================

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from analysis.support_resistance import SupportResistance, _classify_strength

log = logging.getLogger(__name__)


# ─── Tunable constants ────────────────────────────────────────
WICK_BODY_RATIO_MIN     = 1.5   # spec: wick >= 1.5x body
WICK_BODY_RATIO_STRONG  = 2.0   # spec: 2x = strong signal
REVERSAL_LOOKBACK       = 3     # spec: 1–3 candles after break
ZONE_NEAR_FACTOR        = 0.3   # body "near" zone = within 0.3× zone width
SL_BUFFER_ATR_MULT      = 0.15  # SL = wick_extreme ± 0.15×ATR buffer
MIN_RR_RATIO            = 2.0   # spec: minimum 1:2 R:R
ROUND_NUMBER_PIPS_FX    = 50    # round number = multiples of 50 pips (0.0050)
MIN_CANDLES_REQUIRED    = 30    # spec: insufficient data threshold


# ─── Helpers (imported from shared _engine_utils) ─────────────
# Eliminates 5-way duplication of ATR / pip_value / is_round_number.
from analysis._engine_utils import (
    atr_series as _atr,
    pip_value as _pip_value,
    is_round_number as _is_round_number,
    no_trade_signal as _no_trade_signal_shared,
)


def _strength_to_confidence(strength: str, has_round_number: bool,
                            wick_body_ratio: float) -> str:
    """Map strength + confluence factors → confidence Low/Medium/High."""
    score = 0
    if strength == "Strong":
        score += 2
    elif strength == "Medium":
        score += 1
    if has_round_number:
        score += 1
    if wick_body_ratio >= WICK_BODY_RATIO_STRONG:
        score += 1
    if score >= 3:
        return "High"
    if score == 2:
        return "Medium"
    return "Low"


# ─── Dataclass for stop-hunt events ───────────────────────────

@dataclass
class StopHuntEvent:
    """A single confirmed stop-hunt event on a zone."""
    detected: bool = False
    zone_role: str = "null"           # "support" | "resistance" | "null"
    zone_top: float = 0.0
    zone_bottom: float = 0.0
    break_index: int = -1             # candle index that wicked through zone
    confirm_index: int = -1           # candle index that confirmed reversal
    wick_extreme: float = 0.0         # the wick's extreme price (for SL placement)
    wick_body_ratio: float = 0.0
    reversal_direction: str = "NONE"  # "BUY" (bullish reversal) | "SELL" (bearish)
    zone_strength: str = "Weak"       # S/R zone strength
    has_round_number: bool = False
    has_equal_highs_lows: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "detected":              self.detected,
            "zone_role":             self.zone_role,
            "zone_top":              self.zone_top,
            "zone_bottom":           self.zone_bottom,
            "break_index":           self.break_index,
            "confirm_index":         self.confirm_index,
            "wick_extreme":          self.wick_extreme,
            "wick_body_ratio":       self.wick_body_ratio,
            "reversal_direction":    self.reversal_direction,
            "zone_strength":         self.zone_strength,
            "has_round_number":      self.has_round_number,
            "has_equal_highs_lows":  self.has_equal_highs_lows,
            "note":                  self.note,
        }


# ─── Main Engine ──────────────────────────────────────────────

class StopHuntSignalEngine:
    """
    Combines S/R Zone detection + Stop Hunt detection + Trade signal
    generation into a single spec-compliant engine.

    Usage:
        engine = StopHuntSignalEngine(timeframe="H1")
        result = engine.analyze(df, symbol="EURUSD")
        print(json.dumps(result, indent=2))
    """

    def __init__(
        self,
        timeframe: str = "H1",
        swing_window: Optional[int] = None,
        cluster_threshold_pct: Optional[float] = None,
        min_touches: int = 2,
        wick_body_ratio: float = WICK_BODY_RATIO_MIN,
        reversal_lookback: int = REVERSAL_LOOKBACK,
        max_zones_per_side: int = 3,
    ):
        self.timeframe = timeframe
        self.wick_body_ratio = wick_body_ratio
        self.reversal_lookback = reversal_lookback

        self.sr_engine = SupportResistance(
            timeframe=timeframe,
            swing_window=swing_window,
            cluster_threshold_pct=cluster_threshold_pct,
            min_touches=min_touches,
            wick_body_ratio=wick_body_ratio,
            max_zones_per_side=max_zones_per_side,
        )

    # ═══════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> dict:
        """
        Full pipeline:
          1. Detect S/R zones via SupportResistance.analyze()
          2. Scan recent candles for stop-hunt events on each zone
          3. Pick the strongest recent stop-hunt event
          4. Generate trade signal (entry/SL/TP/RR)
          5. Return spec-compliant JSON

        Returns Python dict — caller can json.dumps() it.
        """
        # ── Edge case: insufficient data ──
        if df is None or len(df) < MIN_CANDLES_REQUIRED:
            return self._no_trade_result(
                reason=f"Insufficient data ({len(df) if df is not None else 0} candles, "
                       f"need ≥{MIN_CANDLES_REQUIRED})",
                sr_zones={"resistance_zones": [], "support_zones": []},
            )

        # ── Step 1: S/R Zones ──
        try:
            sr_result = self.sr_engine.analyze(df, symbol=symbol)
        except Exception as e:
            log.error(f"[StopHuntEngine] S/R analyze failed: {e}")
            return self._no_trade_result(
                reason=f"S/R analysis failed: {e}",
                sr_zones={"resistance_zones": [], "support_zones": []},
            )

        # Slim zones for output schema (spec rule 4)
        resistance_zones_slim = [
            {
                "zone_top":    z["zone_top"],
                "zone_bottom": z["zone_bottom"],
                "touches":     z["touches"],
                "strength":    z["strength"],
            }
            for z in sr_result.get("resistance_zones", [])
        ]
        support_zones_slim = [
            {
                "zone_top":    z["zone_top"],
                "zone_bottom": z["zone_bottom"],
                "touches":     z["touches"],
                "strength":    z["strength"],
            }
            for z in sr_result.get("support_zones", [])
        ]

        # ── Step 2: Stop Hunt Detection ──
        event = self._detect_stop_hunt(df, sr_result, symbol)

        # ── Step 3: Trade Signal Generation ──
        signal = self._generate_signal(df, event, sr_result, symbol)

        return {
            "resistance_zones":   resistance_zones_slim,
            "support_zones":      support_zones_slim,
            "stop_hunt_detected": event.detected,
            "stop_hunt_zone":     event.zone_role if event.detected else "null",
            "signal":             signal,
        }

    # ═══════════════════════════════════════════════════════════
    # STEP 2: STOP HUNT DETECTION
    # ═══════════════════════════════════════════════════════════

    def _detect_stop_hunt(
        self,
        df: pd.DataFrame,
        sr_result: dict,
        symbol: str,
    ) -> StopHuntEvent:
        """
        Scan recent candles for stop-hunt signatures on each zone.

        Returns the most recent + strongest StopHuntEvent, or an
        empty (detected=False) event if none found.
        """
        n = len(df)
        # Only scan the last 20 candles for stop hunts
        scan_start = max(0, n - 20)

        all_zones = []
        for z in sr_result.get("resistance_zones", []):
            all_zones.append({**z, "_role": "resistance"})
        for z in sr_result.get("support_zones", []):
            all_zones.append({**z, "_role": "support"})

        if not all_zones:
            return StopHuntEvent()

        candidate_events: List[StopHuntEvent] = []

        for zone in all_zones:
            ev = self._check_zone_for_stop_hunt(df, zone, symbol, scan_start, n)
            if ev.detected:
                candidate_events.append(ev)

        if not candidate_events:
            return StopHuntEvent()

        # Pick the most recent (highest confirm_index), tie-break by zone strength
        strength_rank = {"Strong": 3, "Medium": 2, "Weak": 1}
        candidate_events.sort(
            key=lambda e: (
                e.confirm_index,
                strength_rank.get(e.zone_strength, 0),
                e.wick_body_ratio,
            ),
            reverse=True,
        )
        return candidate_events[0]

    def _check_zone_for_stop_hunt(
        self,
        df: pd.DataFrame,
        zone: dict,
        symbol: str,
        scan_start: int,
        n: int,
    ) -> StopHuntEvent:
        """
        Check a single zone for stop-hunt pattern.

        For RESISTANCE zone:
          - Stop hunt = wick pierces ABOVE zone_top, but body closes
            inside/below zone, then 1–3 candles later price closes
            BELOW zone_bottom (bearish reversal).
          - Reversal direction = SELL.

        For SUPPORT zone:
          - Stop hunt = wick pierces BELOW zone_bottom, but body closes
            inside/above zone, then 1–3 candles later price closes
            ABOVE zone_top (bullish reversal).
          - Reversal direction = BUY.

        "Inside or near zone" body close is checked using ATR-based band
        so it works for both tight (2-pip) and wide (50-pip) zones.
        """
        zone_top    = float(zone["zone_top"])
        zone_bottom = float(zone["zone_bottom"])
        zone_width  = max(zone_top - zone_bottom, 1e-9)
        role        = zone["_role"]
        zone_str    = zone.get("strength", "Weak")

        highs  = df["high"].values
        lows   = df["low"].values
        opens  = df["open"].values
        closes = df["close"].values

        # Compute ATR once for the "near" band
        try:
            atr_series = _atr(df, period=14)
            atr_val = float(atr_series.iloc[-1])
            if not np.isfinite(atr_val) or atr_val <= 0:
                atr_val = float(closes[-1]) * 0.001
        except Exception as e:
            atr_val = float(closes[-1]) * 0.001

        # Body close "near" zone = within max(0.5×zone_width, 0.5×ATR) of boundary
        # This handles both tight (2-pip) and wide (50-pip) zones sanely
        near_band = max(zone_width * 0.5, atr_val * 0.5)

        for i in range(scan_start, n - 1):  # leave room for confirmation candles
            o, h, l, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
            body = abs(c - o)
            if body < 1e-9:
                continue  # skip doji with zero body

            if role == "resistance":
                # Wick pierced above zone_top?
                if h <= zone_top:
                    continue
                wick_extreme = h
                upper_wick = h - max(o, c)
                wick_body_ratio = upper_wick / body
                # Body must close INSIDE zone (zone_bottom ≤ close ≤ zone_top)
                # OR "near" zone (close within near_band below zone_bottom)
                body_closes_inside_or_near = (
                    zone_bottom <= c <= zone_top
                ) or (
                    zone_bottom - near_band <= c < zone_bottom
                )
                if not body_closes_inside_or_near:
                    continue
                # Wick must be ≥ 1.5× body
                if wick_body_ratio < self.wick_body_ratio:
                    continue
                # Look for bearish reversal in next 1–3 candles
                # → price closes BELOW zone_bottom
                confirm_idx = self._find_reversal_confirmation(
                    closes, i + 1, n, "below", zone_bottom
                )
                if confirm_idx is None:
                    continue
                reversal_dir = "SELL"
                wick_extreme_final = h
            else:  # support
                # Wick pierced below zone_bottom?
                if l >= zone_bottom:
                    continue
                wick_extreme = l
                lower_wick = min(o, c) - l
                wick_body_ratio = lower_wick / body
                # Body must close INSIDE zone OR near (above zone_top)
                body_closes_inside_or_near = (
                    zone_bottom <= c <= zone_top
                ) or (
                    zone_top < c <= zone_top + near_band
                )
                if not body_closes_inside_or_near:
                    continue
                # Wick ≥ 1.5× body
                if wick_body_ratio < self.wick_body_ratio:
                    continue
                # Look for bullish reversal in next 1–3 candles
                # → price closes ABOVE zone_top
                confirm_idx = self._find_reversal_confirmation(
                    closes, i + 1, n, "above", zone_top
                )
                if confirm_idx is None:
                    continue
                reversal_dir = "BUY"
                wick_extreme_final = l

            # ─── Stop hunt confirmed ───
            # Round number / equal highs/lows check (Step 2 rule 4)
            has_rn = _is_round_number(wick_extreme_final, symbol)
            has_ehl = self._check_equal_highs_lows(df, i, wick_extreme_final, role)

            note = (
                f"Stop hunt at {role} zone [{zone_bottom:.5f}–{zone_top:.5f}] "
                f"| wick_extreme={wick_extreme_final:.5f} "
                f"| wick/body={wick_body_ratio:.2f} "
                f"| confirmed at candle {confirm_idx} "
                f"| reversal={reversal_dir}"
                f"{' | near round number' if has_rn else ''}"
                f"{' | equal highs/lows' if has_ehl else ''}"
            )

            return StopHuntEvent(
                detected=True,
                zone_role=role,
                zone_top=zone_top,
                zone_bottom=zone_bottom,
                break_index=i,
                confirm_index=confirm_idx,
                wick_extreme=wick_extreme_final,
                wick_body_ratio=wick_body_ratio,
                reversal_direction=reversal_dir,
                zone_strength=zone_str,
                has_round_number=has_rn,
                has_equal_highs_lows=has_ehl,
                note=note,
            )

        return StopHuntEvent()

    def _find_reversal_confirmation(
        self,
        closes: np.ndarray,
        start_idx: int,
        n: int,
        direction: str,
        threshold: float,
    ) -> Optional[int]:
        """
        Within `reversal_lookback` candles starting at `start_idx`,
        find the first candle whose close is on the reversal side
        of `threshold`.

        direction="above" → look for close > threshold
        direction="below" → look for close < threshold
        """
        end_idx = min(n, start_idx + self.reversal_lookback)
        for j in range(start_idx, end_idx):
            c = float(closes[j])
            if direction == "above" and c > threshold:
                return j
            if direction == "below" and c < threshold:
                return j
        return None

    def _check_equal_highs_lows(
        self,
        df: pd.DataFrame,
        break_idx: int,
        wick_extreme: float,
        role: str,
        lookback: int = 20,
        tolerance_pct: float = 0.001,
    ) -> bool:
        """
        Check if the wick_extreme is near (within tolerance_pct) any
        other swing high/low in the past `lookback` candles.
        This identifies "equal highs" / "equal lows" — classic
        liquidity pools where stops rest.
        """
        try:
            start = max(0, break_idx - lookback)
            end = break_idx
            if end - start < 3:
                return False
            window = df.iloc[start:end]
            if role == "resistance":
                # Compare wick_extreme to other swing highs
                ref_levels = window["high"].values
            else:
                ref_levels = window["low"].values
            # Exclude the break candle itself
            ref_levels = ref_levels[:-1] if len(ref_levels) > 0 else ref_levels
            for lv in ref_levels:
                if lv > 0 and abs(lv - wick_extreme) / lv <= tolerance_pct:
                    return True
            return False
        except Exception as e:
            log.debug(f"[stop_hunt_signal_engine.py] suppressed: {e}")
            return False

    # ═══════════════════════════════════════════════════════════
    # STEP 3: TRADE SIGNAL GENERATION
    # ═══════════════════════════════════════════════════════════

    def _generate_signal(
        self,
        df: pd.DataFrame,
        event: StopHuntEvent,
        sr_result: dict,
        symbol: str,
    ) -> dict:
        """Generate the spec-compliant signal dict."""
        # ── Case A: No stop hunt detected → NO_TRADE ──
        if not event.detected:
            return self._no_trade_signal(
                reason="No stop hunt confirmed — wait for setup"
            )

        # ── Case B: Stop hunt confirmed → compute entry/SL/TP ──
        atr_series = _atr(df, period=14)
        atr_val = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
        if not np.isfinite(atr_val) or atr_val <= 0:
            atr_val = float(df["close"].iloc[-1]) * 0.001  # fallback 0.1% of price

        # Entry = open of the candle AFTER confirmation
        # (or current close if confirmation is the last candle)
        confirm_idx = event.confirm_index
        if confirm_idx + 1 < len(df):
            entry_price = float(df["open"].iloc[confirm_idx + 1])
        else:
            entry_price = float(df["close"].iloc[-1])

        # SL = just beyond wick extreme
        sl_buffer = atr_val * SL_BUFFER_ATR_MULT
        if event.reversal_direction == "BUY":
            stop_loss = event.wick_extreme - sl_buffer
            # TP = opposite-side nearest zone OR 1:2 R:R
            take_profit = self._compute_tp(
                entry_price, stop_loss, sr_result.get("resistance_zones", []),
                direction="BUY", min_rr=MIN_RR_RATIO,
            )
            # Sanity: TP must be above entry, SL below entry
            if not (stop_loss < entry_price < take_profit):
                return self._no_trade_signal(
                    reason="Stop hunt confirmed but SL/TP geometry invalid — wait for retest"
                )
        else:  # SELL
            stop_loss = event.wick_extreme + sl_buffer
            take_profit = self._compute_tp(
                entry_price, stop_loss, sr_result.get("support_zones", []),
                direction="SELL", min_rr=MIN_RR_RATIO,
            )
            if not (take_profit < entry_price < stop_loss):
                return self._no_trade_signal(
                    reason="Stop hunt confirmed but SL/TP geometry invalid — wait for retest"
                )

        # R:R check
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr = reward / risk if risk > 0 else 0.0
        if rr < MIN_RR_RATIO:
            # Fall back to 1:2 R:R if zone-based TP is too close
            if event.reversal_direction == "BUY":
                take_profit = entry_price + risk * MIN_RR_RATIO
            else:
                take_profit = entry_price - risk * MIN_RR_RATIO
            rr = MIN_RR_RATIO
            tp_note = f" | TP set to 1:{MIN_RR_RATIO:.0f} R:R (zone TP too close)"
        else:
            tp_note = ""

        # Confidence scoring
        confidence = _strength_to_confidence(
            event.zone_strength,
            event.has_round_number,
            event.wick_body_ratio,
        )

        reason = (
            f"Stop hunt confirmed at {event.zone_role} zone "
            f"[{event.zone_bottom:.5f}–{event.zone_top:.5f}] "
            f"({event.zone_strength}). "
            f"Wick/body={event.wick_body_ratio:.2f}, "
            f"reversal={event.reversal_direction}, R:R=1:{rr:.1f}."
            f"{' Round number confluence.' if event.has_round_number else ''}"
            f"{' Equal highs/lows confluence.' if event.has_equal_highs_lows else ''}"
            f"{tp_note}"
        )

        return {
            "action":      event.reversal_direction,    # BUY or SELL
            "entry_price": round(entry_price, 5),
            "stop_loss":   round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "reason":      reason,
            "confidence":  confidence,
        }

    def _compute_tp(
        self,
        entry_price: float,
        stop_loss: float,
        opposite_zones: list,
        direction: str,
        min_rr: float = 2.0,
    ) -> float:
        """
        Compute take-profit as the nearest opposite-side zone boundary.
        If no zone available, use 1:min_rr R:R.
        """
        risk = abs(entry_price - stop_loss)
        if risk <= 0:
            risk = abs(entry_price) * 0.005  # fallback 0.5%

        # Default = min R:R
        if direction == "BUY":
            default_tp = entry_price + risk * min_rr
            # Try to find a resistance zone above entry
            candidates = [
                z["zone_bottom"] for z in opposite_zones
                if z.get("zone_bottom", 0) > entry_price
            ]
            if candidates:
                zone_tp = min(candidates)
                # If zone is at least 1:1, use it; otherwise use default
                if (zone_tp - entry_price) >= risk:
                    return zone_tp
            return default_tp
        else:  # SELL
            default_tp = entry_price - risk * min_rr
            candidates = [
                z["zone_top"] for z in opposite_zones
                if z.get("zone_top", 0) < entry_price and z.get("zone_top", 0) > 0
            ]
            if candidates:
                zone_tp = max(candidates)
                if (entry_price - zone_tp) >= risk:
                    return zone_tp
            return default_tp

    # ═══════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _no_trade_signal(reason: str) -> dict:
        return {
            "action":      "NO_TRADE",
            "entry_price": None,
            "stop_loss":   None,
            "take_profit": None,
            "reason":      reason,
            "confidence":  "Low",
        }

    @staticmethod
    def _no_trade_result(reason: str, sr_zones: dict) -> dict:
        return {
            "resistance_zones":   sr_zones.get("resistance_zones", []),
            "support_zones":      sr_zones.get("support_zones", []),
            "stop_hunt_detected": False,
            "stop_hunt_zone":     "null",
            "signal":             StopHuntSignalEngine._no_trade_signal(reason),
        }

    # ═══════════════════════════════════════════════════════════
    # LLM-FRIENDLY OUTPUT
    # ═══════════════════════════════════════════════════════════

    def analyze_to_json(self, df: pd.DataFrame, symbol: str = "") -> str:
        """Run full analyze() and return spec-compliant JSON string."""
        result = self.analyze(df, symbol=symbol)
        return json.dumps(result, ensure_ascii=False, indent=2)

    def to_prompt_text(self, result: dict) -> str:
        """Plain-text rendering for embedding into LLM prompts."""
        lines = [
            f"=== STOP HUNT SIGNAL ===",
            f"Stop Hunt Detected : {result['stop_hunt_detected']}",
            f"Stop Hunt Zone     : {result['stop_hunt_zone']}",
            "",
            "-- Resistance Zones --",
        ]
        if not result.get("resistance_zones"):
            lines.append("  (none)")
        else:
            for z in result["resistance_zones"]:
                lines.append(
                    f"  R: {z['zone_bottom']:.5f} → {z['zone_top']:.5f}  "
                    f"| touches={z['touches']} | {z['strength']}"
                )
        lines.append("")
        lines.append("-- Support Zones --")
        if not result.get("support_zones"):
            lines.append("  (none)")
        else:
            for z in result["support_zones"]:
                lines.append(
                    f"  S: {z['zone_bottom']:.5f} → {z['zone_top']:.5f}  "
                    f"| touches={z['touches']} | {z['strength']}"
                )
        lines.append("")
        sig = result.get("signal", {})
        lines.append("-- Signal --")
        lines.append(f"  Action     : {sig.get('action')}")
        lines.append(f"  Entry      : {sig.get('entry_price')}")
        lines.append(f"  Stop Loss  : {sig.get('stop_loss')}")
        lines.append(f"  Take Profit: {sig.get('take_profit')}")
        lines.append(f"  Confidence : {sig.get('confidence')}")
        lines.append(f"  Reason     : {sig.get('reason')}")
        lines.append("=" * 50)
        return "\n".join(lines)


# ============================================================
# Convenience: one-shot helper for LLM Agent integration
# ============================================================

def detect_stop_hunt_signal(
    df: pd.DataFrame,
    symbol: str = "",
    timeframe: str = "H1",
    **kwargs,
) -> str:
    """
    One-shot helper — pass OHLC df, get spec-compliant JSON back.

    Returns JSON string with the EXACT schema from the spec:
      {
        "resistance_zones": [...],
        "support_zones": [...],
        "stop_hunt_detected": bool,
        "stop_hunt_zone": "support|resistance|null",
        "signal": {action, entry_price, stop_loss, take_profit, reason, confidence}
      }
    """
    engine = StopHuntSignalEngine(timeframe=timeframe, **kwargs)
    return engine.analyze_to_json(df, symbol=symbol)


# ============================================================
# CLI entry — quick smoke test
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    base = 1.0850
    close = base + np.cumsum(np.random.randn(n) * 0.0005)
    df = pd.DataFrame({
        "open":  close + np.random.randn(n) * 0.0002,
        "high":  close + abs(np.random.randn(n)) * 0.0008,
        "low":   close - abs(np.random.randn(n)) * 0.0008,
        "close": close,
    }, index=dates)

    engine = StopHuntSignalEngine(timeframe="H1")
    result = engine.analyze(df, symbol="EURUSD")
    print(json.dumps(result, indent=2))
