# analysis/market_structure.py  —  Day 45 | Market Structure (Swings / BOS / CHoCH / Sweep)
#
# ⚠️ NEW MODULE — did not exist in what was uploaded. Built specifically to gate
# OrderBlockDetector per the video-research audit (every source requires a
# structure break to validate an OB; none of your existing files did this).
# If you already have a market-structure / BOS-CHoCH module in forex-agent,
# tell me and I will re-wire order_block.py against that one instead — don't
# let this become a second, drifting implementation of the same concept.
#
# CAUSALITY CONTRACT (read this before touching the file):
#   A "fractal" swing high/low at index i can only be confirmed once
#   `strength` bars have closed AFTER i. Until then it does not exist yet —
#   using it earlier is look-ahead bias. Every method below only ever
#   references swings that are already confirmed as of the bar being
#   evaluated. Do not "peek" at self.swings for indices beyond what the
#   caller has processed.
import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("market_structure")


class MarketStructure:
    """
    Usage:
        ms = MarketStructure(strength=2)
        state = ms.analyze(df)   # df needs only open/high/low/close, causal, closed bars only

    state = {
        'swings':        [ {index, type: HIGH|LOW, price, confirmed_at} ... ],
        'trend':         'UP' | 'DOWN' | 'UNDEFINED',
        'events':        [ {index, type: BOS|CHOCH, direction, broke_index, broke_price} ... ],
        'last_swing_high': {...} | None,
        'last_swing_low':  {...} | None,
    }
    """

    def analyze(self, df: pd.DataFrame, strength: int = 2) -> dict:
        if len(df) < (strength * 2 + 5):
            log.warning("[MarketStructure] Insufficient data")
            return {'swings': [], 'trend': 'UNDEFINED', 'events': [],
                    'last_swing_high': None, 'last_swing_low': None}

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        n = len(df)

        swings = self._detect_fractals(highs, lows, strength, n)

        trend = 'UNDEFINED'
        events = []
        confirmed_highs = []   # chronological list of confirmed swing-high dicts
        confirmed_lows = []
        last_broken_high_idx = None
        last_broken_low_idx = None

        # Walk bar-by-bar so BOS/CHoCH is only ever evaluated against swings
        # that were ALREADY confirmed strictly before the current bar's close.
        swings_by_confirm_idx = {}
        for s in swings:
            swings_by_confirm_idx.setdefault(s['confirmed_at'], []).append(s)

        for i in range(n):
            # Register any swing that becomes confirmed exactly at this bar
            for s in swings_by_confirm_idx.get(i, []):
                if s['type'] == 'HIGH':
                    confirmed_highs.append(s)
                else:
                    confirmed_lows.append(s)

            if not confirmed_highs or not confirmed_lows:
                continue

            c = closes[i]
            last_high = confirmed_highs[-1]
            last_low = confirmed_lows[-1]

            # BOS / CHoCH detection: a bar CLOSING beyond the last confirmed
            # swing (not a wick) is what actually breaks structure — see
            # video-research finding re: close vs wick ambiguity.
            if last_high['broken_idx'] is None and c > last_high['price']:
                last_high['broken_idx'] = i
                direction = 'UP'
                event_type = 'BOS' if trend in ('UP', 'UNDEFINED') else 'CHOCH'
                events.append({'index': i, 'type': event_type, 'direction': direction,
                                'broke_index': last_high['index'], 'broke_price': last_high['price']})
                trend = 'UP'

            if last_low['broken_idx'] is None and c < last_low['price']:
                last_low['broken_idx'] = i
                direction = 'DOWN'
                event_type = 'BOS' if trend in ('DOWN', 'UNDEFINED') else 'CHOCH'
                events.append({'index': i, 'type': event_type, 'direction': direction,
                                'broke_index': last_low['index'], 'broke_price': last_low['price']})
                trend = 'DOWN'

        log.info(f"[MarketStructure] {len(swings)} swings, {len(events)} BOS/CHoCH events, trend={trend}")
        return {
            'swings': swings,
            'trend': trend,
            'events': events,
            'last_swing_high': confirmed_highs[-1] if confirmed_highs else None,
            'last_swing_low': confirmed_lows[-1] if confirmed_lows else None,
        }

    def _detect_fractals(self, highs, lows, strength: int, n: int) -> list[dict]:
        swings = []
        for i in range(strength, n - strength):
            window_h = highs[i - strength: i + strength + 1]
            if highs[i] == window_h.max() and np.argmax(window_h) == strength:
                swings.append({'index': i, 'type': 'HIGH', 'price': float(highs[i]),
                                'confirmed_at': i + strength, 'broken_idx': None})
            window_l = lows[i - strength: i + strength + 1]
            if lows[i] == window_l.min() and np.argmin(window_l) == strength:
                swings.append({'index': i, 'type': 'LOW', 'price': float(lows[i]),
                                'confirmed_at': i + strength, 'broken_idx': None})
        swings.sort(key=lambda s: s['confirmed_at'])
        return swings

    def structure_break_between(self, state: dict, start_idx: int, end_idx: int) -> dict | None:
        """Most recent BOS/CHoCH event with index in (start_idx, end_idx]. Used by
        OrderBlockDetector to check whether an impulse leg actually broke structure."""
        candidates = [e for e in state['events'] if start_idx < e['index'] <= end_idx]
        return candidates[-1] if candidates else None

    def liquidity_sweep_before(self, df: pd.DataFrame, state: dict, idx: int,
                                lookback: int = 15) -> dict | None:
        """
        Checks whether, within `lookback` bars before idx, price wicked beyond a
        confirmed swing high/low and then CLOSED back inside range (the classic
        sweep-then-reverse signature — best-corroborated concept across the
        video research: videos 4, 5, and 6 all independently flag sweep-conditioned
        order blocks as the highest-confidence type).
        """
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        recent_swings = [s for s in state['swings']
                          if s['confirmed_at'] <= idx and idx - lookback <= s['index'] < idx]
        for s in sorted(recent_swings, key=lambda s: s['index'], reverse=True):
            for k in range(s['index'] + 1, min(idx + 1, len(df))):
                if s['type'] == 'HIGH' and highs[k] > s['price'] and closes[k] < s['price']:
                    return {'swept_index': s['index'], 'swept_price': s['price'],
                            'sweep_index': k, 'type': 'HIGH_SWEEP'}
                if s['type'] == 'LOW' and lows[k] < s['price'] and closes[k] > s['price']:
                    return {'swept_index': s['index'], 'swept_price': s['price'],
                            'sweep_index': k, 'type': 'LOW_SWEEP'}
        return None