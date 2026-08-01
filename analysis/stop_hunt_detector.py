# analysis/stop_hunt_detector.py  —  Day 62 | Stop Hunt Detector
# ============================================================
# AI Trader-এর সবচেয়ে গুরুত্বপূর্ণ liquidity module।
#
# Retail view  : "Support ভেঙেছে → SELL"
# Institutional view : "Support-এর নিচে liquidity ছিল → stop loss
#                        নেওয়া হয়েছে → reversal হতে পারে"
#
# Detection logic:
#   1. Price একটা known liquidity level (equal high/low, PDH/PDL,
#      Asian high/low) break করে।
#   2. Break-এর পরের candle(s)-এ strong rejection দেখা যায়
#      (wick + close back inside, opposite-direction momentum)।
#   3. শর্ত পূরণ হলে → stop_hunt = True + reversal direction + target।
# ============================================================

import warnings

import numpy as np
import pandas as pd
from utils.logger import get_logger

# ── Pip-size helper (institutional review, Finding C-3) ─────────────────
# map_liquidity_target() below previously hardcoded * 10000 (i.e. pip =
# 0.0001) for distance_pips, which is ~100x wrong on JPY-quoted pairs.
try:
    from analysis._engine_utils import pip_value as _pip_value
except ImportError:  # pragma: no cover - fallback if _engine_utils is unavailable
    def _pip_value(symbol: str = "", price: float | None = None) -> float:
        """Local fallback pip-size resolver (JPY = 0.01, else 0.0001)."""
        s = (symbol or "").upper()
        if "JPY" in s:
            return 0.01
        if price is not None and price > 20:
            return 0.01
        return 0.0001

log = get_logger("stop_hunt_detector")


class StopHuntDetector:
    """
    Usage:
        detector = StopHuntDetector()
        events = detector.detect(df, liquidity_levels)
        best   = detector.best_signal(events)

    DEPRECATION NOTICE (institutional review, Finding H-2):
    This codebase currently contains two independent stop-hunt engines —
    this one (level-list driven, Day 62) and the newer, more rigorous
    `stop_hunt_signal_engine.py` (zone-driven, ATR-adaptive rejection
    scoring, explicit R:R sanity checks, structured NO_TRADE fallback).
    They can disagree on the same market event with no documented
    arbitration rule between them.

    No behavior has been changed here beyond the Finding C-3 pip-size fix
    — removing/merging this class outright risks breaking callers outside
    this review's scope. New integrations should prefer
    `stop_hunt_signal_engine.py`; this class emits a DeprecationWarning on
    instantiation as a visible, non-breaking pointer in that direction.
    """

    REJECTION_WICK_RATIO  = 1.5   # rejection wick, body-এর কমপক্ষে এই গুণ হতে হবে
    REJECTION_LOOKBACK    = 3     # break-এর পরে কত candle-এর মধ্যে rejection খুঁজবে
    MIN_PENETRATION_ATR   = 0.05  # level-এর কতটা বাইরে গেলে "swept" ধরা হবে (ATR fraction)
    MAX_RESULTS           = 6

    def __init__(self):
        warnings.warn(
            "StopHuntDetector is superseded by StopHuntSignalEngine "
            "(analysis/stop_hunt_signal_engine.py), which has stronger "
            "rejection scoring, R:R sanity checks, and explicit NO_TRADE "
            "fallback discipline. See institutional review Finding H-2. "
            "This class still works unchanged and is not being removed "
            "automatically to avoid breaking existing callers.",
            DeprecationWarning,
            stacklevel=2,
        )

    # ═══════════════════════════════════════════════════════
    # MAIN METHOD
    # ═══════════════════════════════════════════════════════

    def detect(self, df: pd.DataFrame, liquidity_levels: list[dict]) -> list[dict]:
        """
        Args:
            df: OHLC + 'atr' column
            liquidity_levels: list of dicts, each needs at least:
                {'price': float, 'liquidity_type': 'BUY_SIDE'|'SELL_SIDE', 'label': str}
                (EQUAL_HIGH/EQUAL_LOW থেকে আসা dict, বা PDH/PDL থেকে বানানো dict)

        Returns: list of stop-hunt event dicts, most-recent-first.
        """
        if len(df) < 10 or 'atr' not in df.columns or not liquidity_levels:
            return []

        highs  = df['high'].values
        lows   = df['low'].values
        opens  = df['open'].values
        closes = df['close'].values
        atrs   = df['atr'].values
        n      = len(df)

        events = []

        for level in liquidity_levels:
            price = level.get('price')
            ltype = level.get('liquidity_type', 'SELL_SIDE')   # what kind of stops rest there
            if price is None:
                continue

            event = self._check_level_for_sweep(
                price, ltype, highs, lows, opens, closes, atrs, n, level
            )
            if event:
                events.append(event)

        events.sort(key=lambda e: e['candles_ago'])
        log.info(f"[StopHuntDetector] {len(events)} stop-hunt event(s) detected")
        return events[: self.MAX_RESULTS]

    # ═══════════════════════════════════════════════════════
    # CORE SWEEP + REJECTION LOGIC
    # ═══════════════════════════════════════════════════════

    def _check_level_for_sweep(
        self, price, liquidity_type, highs, lows, opens, closes, atrs, n, level
    ) -> dict | None:
        """
        একটা single liquidity level-এর জন্য sweep + rejection check করো।

        SELL_SIDE liquidity (নিচে sell-stops) ভাঙলে আমরা bullish reversal খুঁজি।
        BUY_SIDE liquidity (উপরে buy-stops) ভাঙলে আমরা bearish reversal খুঁজি।
        """
        # সাম্প্রতিক candle গুলোর মধ্যে কোনটা level break করেছে খুঁজো
        search_window = min(n, 30)
        start_idx     = n - search_window

        for i in range(start_idx, n):
            atr = atrs[i]
            if np.isnan(atr) or atr == 0:
                continue
            penetration_min = atr * self.MIN_PENETRATION_ATR

            if liquidity_type == 'SELL_SIDE':
                # নিচের দিকে break — wick নিচে গিয়ে level-এর নিচে গেলে
                broke = lows[i] < (price - penetration_min)
            else:  # BUY_SIDE
                broke = highs[i] > (price + penetration_min)

            if not broke:
                continue

            # Break হয়েছে index i-তে — এখন rejection খুঁজো i থেকে কয়েক candle-এর মধ্যে
            rejection = self._find_rejection(
                i, price, liquidity_type, highs, lows, opens, closes, atrs, n
            )
            if rejection:
                direction = 'BULLISH_REVERSAL' if liquidity_type == 'SELL_SIDE' else 'BEARISH_REVERSAL'

                return {
                    'stop_hunt':     True,
                    'level':         round(float(price), 5),
                    'level_label':   level.get('label', level.get('type', 'LEVEL')),
                    'liquidity_type': liquidity_type,
                    'break_index':   i,
                    'rejection_index': rejection['index'],
                    'candles_ago':   n - 1 - rejection['index'],
                    'direction':     direction,
                    'rejection_strength': rejection['strength'],
                    'confirmation': rejection['confirmation'],
                    'note': (
                        f"Stop hunt at {price:.5f} ({level.get('label','level')}) — "
                        f"{liquidity_type} liquidity swept, {direction} signal"
                    ),
                }

        return None

    def _find_rejection(
        self, break_idx, price, liquidity_type, highs, lows, opens, closes, atrs, n
    ) -> dict | None:
        """
        Break candle-এর পর থেকে REJECTION_LOOKBACK candle-এর মধ্যে strong
        rejection (close back inside + বড় opposite wick) আছে কিনা দেখো।
        """
        end_idx = min(n, break_idx + 1 + self.REJECTION_LOOKBACK)

        for j in range(break_idx, end_idx):
            body = abs(closes[j] - opens[j])
            if body == 0:
                body = atrs[j] * 0.05 if atrs[j] else 1e-6

            upper_wick = highs[j] - max(opens[j], closes[j])
            lower_wick = min(opens[j], closes[j]) - lows[j]

            if liquidity_type == 'SELL_SIDE':
                # নিচে sweep হয়েছে → bullish rejection চাই: close back above level,
                # বড় lower wick, candle bullish বা strong rejection wick
                closed_back_above = closes[j] > price
                strong_wick        = lower_wick > body * self.REJECTION_WICK_RATIO
                bullish_close      = closes[j] > opens[j]

                if closed_back_above and (strong_wick or bullish_close):
                    confirmations = []
                    if strong_wick:
                        confirmations.append("Long lower wick rejection")
                    if bullish_close:
                        confirmations.append("Bullish close")
                    if closed_back_above:
                        confirmations.append("Closed back above swept level")

                    strength = self._score_rejection(strong_wick, bullish_close, closed_back_above)
                    return {'index': j, 'strength': strength, 'confirmation': confirmations}

            else:  # BUY_SIDE
                closed_back_below = closes[j] < price
                strong_wick         = upper_wick > body * self.REJECTION_WICK_RATIO
                bearish_close       = closes[j] < opens[j]

                if closed_back_below and (strong_wick or bearish_close):
                    confirmations = []
                    if strong_wick:
                        confirmations.append("Long upper wick rejection")
                    if bearish_close:
                        confirmations.append("Bearish close")
                    if closed_back_below:
                        confirmations.append("Closed back below swept level")

                    strength = self._score_rejection(strong_wick, bearish_close, closed_back_below)
                    return {'index': j, 'strength': strength, 'confirmation': confirmations}

        return None

    def _score_rejection(self, strong_wick: bool, directional_close: bool, closed_back: bool) -> str:
        score = sum([strong_wick, directional_close, closed_back])
        if score >= 3:
            return 'STRONG'
        if score == 2:
            return 'MODERATE'
        return 'WEAK'

    # ═══════════════════════════════════════════════════════
    # BEST SIGNAL SELECTOR
    # ═══════════════════════════════════════════════════════

    def best_signal(self, events: list[dict]) -> dict | None:
        """
        সবচেয়ে strong + সবচেয়ে recent stop-hunt event বেছে নাও।
        """
        if not events:
            return None

        strength_rank = {'STRONG': 3, 'MODERATE': 2, 'WEAK': 1}
        sorted_events = sorted(
            events,
            key=lambda e: (strength_rank.get(e['rejection_strength'], 0), -e['candles_ago']),
            reverse=True,
        )
        return sorted_events[0]

    # ═══════════════════════════════════════════════════════
    # LIQUIDITY TARGET MAPPING
    # ═══════════════════════════════════════════════════════

    def map_liquidity_target(
        self,
        direction:        str,
        current_price:    float,
        liquidity_levels: list[dict],
        symbol:            str = "",
    ) -> dict | None:
        """
        Stop hunt confirm হওয়ার পর — পরবর্তী liquidity target কোথায় সেটা বলো।

        BULLISH_REVERSAL → উপরে নিকটতম BUY_SIDE liquidity (equal high / PDH) খুঁজো
        BEARISH_REVERSAL → নিচে নিকটতম SELL_SIDE liquidity (equal low / PDL) খুঁজো

        Args:
            symbol: optional instrument symbol (e.g. "USDJPY"), used only to
                resolve the correct pip size for distance_pips. Defaults to
                "" for backward compatibility with existing callers; falls
                back to a price-magnitude heuristic when omitted.
        """
        if direction == 'BULLISH_REVERSAL':
            candidates = [
                lv for lv in liquidity_levels
                if lv.get('liquidity_type') == 'BUY_SIDE' and lv.get('price', 0) > current_price
            ]
            if not candidates:
                return None
            target = min(candidates, key=lambda lv: lv['price'])

        elif direction == 'BEARISH_REVERSAL':
            candidates = [
                lv for lv in liquidity_levels
                if lv.get('liquidity_type') == 'SELL_SIDE' and lv.get('price', 0) < current_price
            ]
            if not candidates:
                return None
            target = max(candidates, key=lambda lv: lv['price'])

        else:
            return None

        # FIX (C-3): was hardcoded * 10000 (pip = 0.0001), ~100x wrong on JPY pairs.
        pip_size = _pip_value(symbol, current_price)
        return {
            'target_liquidity': round(float(target['price']), 5),
            'target_label':     target.get('label', target.get('type', 'LEVEL')),
            'distance_pips':    round(abs(target['price'] - current_price) / pip_size, 1),
        }

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT
    # ═══════════════════════════════════════════════════════

    def get_ai_context(self, best_event: dict | None, target: dict | None) -> dict:
        if not best_event:
            return {
                'stop_hunt_detected': False,
                'stop_hunt_direction': 'NONE',
                'stop_hunt_level': None,
                'stop_hunt_strength': 'NONE',
                'stop_hunt_target': None,
                'stop_hunt_note': 'No stop hunt detected',
            }

        return {
            'stop_hunt_detected':  True,
            'stop_hunt_direction': best_event['direction'],
            'stop_hunt_level':     best_event['level'],
            'stop_hunt_level_label': best_event['level_label'],
            'stop_hunt_strength':  best_event['rejection_strength'],
            'stop_hunt_candles_ago': best_event['candles_ago'],
            'stop_hunt_confirmation': best_event['confirmation'],
            'stop_hunt_target':     target.get('target_liquidity') if target else None,
            'stop_hunt_target_label': target.get('target_label') if target else None,
            'stop_hunt_note':       best_event['note'],
        }

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, events: list[dict]) -> None:
        bar = "═" * 56
        log.info(bar)
        log.info("  🎯  STOP HUNT DETECTOR  (Day 62)")
        log.info(bar)
        if not events:
            log.info("  No stop hunts detected.")
            log.info(bar)
            return

        for e in events:
            icon = "🟢" if e['direction'] == 'BULLISH_REVERSAL' else "🔴"
            log.info(
                f"  {icon} {e['direction']}  level={e['level']} ({e['level_label']})  "
                f"strength={e['rejection_strength']}  ({e['candles_ago']} candles ago)"
            )
            for c in e['confirmation']:
                log.info(f"      • {c}")
        log.info(bar)