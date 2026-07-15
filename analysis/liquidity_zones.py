# analysis/liquidity_zones.py  —  Day 62 | Liquidity Zone Mapping
# ============================================================
# Retail trader stop loss কোথায় জমা হয় সেটা খুঁজে বের করে।
#
# Covers:
#   ✅ Equal Highs / Equal Lows   (buy-side / sell-side liquidity)
#   ✅ Previous Day High/Low      (PDH / PDL)
#   ✅ Previous Week High/Low     (PWH / PWL)
#   ✅ Asian Session Range        (00:00–08:00 UTC approx)
#
# এই module শুধু "কোথায় liquidity আছে" বলে — stop hunt confirm করা
# stop_hunt_detector.py-এর কাজ।
# ============================================================

import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("liquidity_zones")


class LiquidityZoneMapper:
    """
    Usage:
        mapper = LiquidityZoneMapper()
        eq_highs = mapper.find_equal_highs(df)
        eq_lows  = mapper.find_equal_lows(df)
        levels   = mapper.calculate_previous_levels(df)
        asian    = mapper.asian_session_range(df)
    """

    EQUAL_TOLERANCE_ATR_MULT = 0.15   # equal high/low মেলানোর tolerance (ATR fraction)
    MIN_TOUCHES              = 2      # কমপক্ষে কতবার একই zone-এ touch হলে "equal" ধরা হবে
    SWING_WINDOW             = 5
    MAX_RESULTS              = 8

    # ═══════════════════════════════════════════════════════
    # 1. EQUAL HIGH / EQUAL LOW
    # ═══════════════════════════════════════════════════════

    def find_equal_highs(self, df: pd.DataFrame) -> list[dict]:
        """
        Swing highs যেগুলো প্রায় একই price-এ বারবার touch হয়েছে।
        এর উপরে buy-stop liquidity (retail short SL + breakout buyers) জমা থাকে।
        """
        if len(df) < self.SWING_WINDOW * 3 or 'atr' not in df.columns:
            return []

        highs = df['high'].values
        atr   = self._safe_atr(df)
        n     = len(df)
        w     = self.SWING_WINDOW

        swing_points = [
            (i, highs[i]) for i in range(w, n - w)
            if highs[i] == max(highs[i - w: i + w + 1])
        ]
        if len(swing_points) < self.MIN_TOUCHES:
            return []

        tolerance = atr * self.EQUAL_TOLERANCE_ATR_MULT
        clusters  = self._cluster_points(swing_points, tolerance)

        results = []
        for cluster in clusters:
            if len(cluster) < self.MIN_TOUCHES:
                continue
            prices    = [p for _, p in cluster]
            avg_price = round(float(np.mean(prices)), 5)
            last_idx  = max(i for i, _ in cluster)

            results.append({
                'type':        'EQUAL_HIGH',
                'price':       avg_price,
                'touches':     len(cluster),
                'liquidity':   'ABOVE',
                'liquidity_type': 'BUY_SIDE',     # buy-stops resting above
                'last_index':  last_idx,
                'candles_ago': n - 1 - last_idx,
                'note': (
                    f"Equal High at {avg_price:.5f} ({len(cluster)} touches) — "
                    f"buy-side liquidity resting above"
                ),
            })

        results.sort(key=lambda r: (r['touches'], -r['candles_ago']), reverse=True)
        return results[: self.MAX_RESULTS]

    def find_equal_lows(self, df: pd.DataFrame) -> list[dict]:
        """
        Swing lows যেগুলো প্রায় একই price-এ বারবার touch হয়েছে।
        এর নিচে sell-stop liquidity (retail long SL + breakdown sellers) জমা থাকে।
        """
        if len(df) < self.SWING_WINDOW * 3 or 'atr' not in df.columns:
            return []

        lows = df['low'].values
        atr  = self._safe_atr(df)
        n    = len(df)
        w    = self.SWING_WINDOW

        swing_points = [
            (i, lows[i]) for i in range(w, n - w)
            if lows[i] == min(lows[i - w: i + w + 1])
        ]
        if len(swing_points) < self.MIN_TOUCHES:
            return []

        tolerance = atr * self.EQUAL_TOLERANCE_ATR_MULT
        clusters  = self._cluster_points(swing_points, tolerance)

        results = []
        for cluster in clusters:
            if len(cluster) < self.MIN_TOUCHES:
                continue
            prices    = [p for _, p in cluster]
            avg_price = round(float(np.mean(prices)), 5)
            last_idx  = max(i for i, _ in cluster)

            results.append({
                'type':        'EQUAL_LOW',
                'price':       avg_price,
                'touches':     len(cluster),
                'liquidity':   'BELOW',
                'liquidity_type': 'SELL_SIDE',    # sell-stops resting below
                'last_index':  last_idx,
                'candles_ago': n - 1 - last_idx,
                'note': (
                    f"Equal Low at {avg_price:.5f} ({len(cluster)} touches) — "
                    f"sell-side liquidity resting below"
                ),
            })

        results.sort(key=lambda r: (r['touches'], -r['candles_ago']), reverse=True)
        return results[: self.MAX_RESULTS]

    def _cluster_points(self, points: list[tuple[int, float]], tolerance: float) -> list[list[tuple[int, float]]]:
        """Price-এর কাছাকাছি swing points-গুলো একই cluster-এ গ্রুপ করো।"""
        if not points:
            return []
        sorted_pts = sorted(points, key=lambda p: p[1])
        clusters: list[list[tuple[int, float]]] = [[sorted_pts[0]]]

        for pt in sorted_pts[1:]:
            last_cluster_prices = [p for _, p in clusters[-1]]
            if abs(pt[1] - np.mean(last_cluster_prices)) <= tolerance:
                clusters[-1].append(pt)
            else:
                clusters.append([pt])
        return clusters

    # ═══════════════════════════════════════════════════════
    # 2. PREVIOUS DAY / WEEK HIGH-LOW
    # ═══════════════════════════════════════════════════════

    def calculate_previous_levels(self, df: pd.DataFrame) -> dict:
        """
        PDH/PDL এবং PWH/PWL — df-এর index একটা DatetimeIndex হতে হবে।
        Intraday (M5–H1) timeframe-এর জন্য designed।
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            log.warning("[LiquidityZones] DataFrame index is not DatetimeIndex — PDH/PDL skipped")
            return self._empty_previous_levels()

        result = {}
        result.update(self._previous_day_levels(df))
        result.update(self._previous_week_levels(df))
        return result

    def _previous_day_levels(self, df: pd.DataFrame) -> dict:
        dates = df.index.normalize()
        unique_days = sorted(set(dates))

        if len(unique_days) < 2:
            return {'PDH': None, 'PDL': None, 'pdh_note': 'Insufficient daily history'}

        prev_day   = unique_days[-2]
        prev_slice = df[dates == prev_day]
        if prev_slice.empty:
            return {'PDH': None, 'PDL': None, 'pdh_note': 'No data for previous day'}

        pdh = round(float(prev_slice['high'].max()), 5)
        pdl = round(float(prev_slice['low'].min()), 5)

        return {
            'PDH': pdh,
            'PDL': pdl,
            'pdh_note': f"Previous Day High={pdh:.5f} — major resistance / buy-side liquidity",
            'pdl_note': f"Previous Day Low={pdl:.5f} — major support / sell-side liquidity",
        }

    def _previous_week_levels(self, df: pd.DataFrame) -> dict:
        iso = df.index.to_series().apply(lambda d: (d.isocalendar()[0], d.isocalendar()[1]))
        unique_weeks = sorted(set(iso))

        if len(unique_weeks) < 2:
            return {'PWH': None, 'PWL': None, 'pwh_note': 'Insufficient weekly history'}

        prev_week   = unique_weeks[-2]
        mask        = iso == prev_week
        prev_slice  = df[mask.values]
        if prev_slice.empty:
            return {'PWH': None, 'PWL': None, 'pwh_note': 'No data for previous week'}

        pwh = round(float(prev_slice['high'].max()), 5)
        pwl = round(float(prev_slice['low'].min()), 5)

        return {
            'PWH': pwh,
            'PWL': pwl,
            'pwh_note': f"Previous Week High={pwh:.5f} — major weekly resistance liquidity",
            'pwl_note': f"Previous Week Low={pwl:.5f} — major weekly support liquidity",
        }

    def _empty_previous_levels(self) -> dict:
        return {
            'PDH': None, 'PDL': None, 'PWH': None, 'PWL': None,
            'pdh_note': 'Unavailable', 'pdl_note': 'Unavailable',
            'pwh_note': 'Unavailable', 'pwl_note': 'Unavailable',
        }

    # ═══════════════════════════════════════════════════════
    # 3. ASIAN SESSION RANGE
    # ═══════════════════════════════════════════════════════

    def asian_session_range(
        self,
        df: pd.DataFrame,
        start_hour: int = 0,
        end_hour:   int = 8,
    ) -> dict:
        """
        Asian session (default 00:00–08:00 UTC) range বের করো — সবচেয়ে
        সাম্প্রতিক session-এর high/low। London open manipulation
        detect করার জন্য foundational data।
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return {'valid': False, 'reason': 'Index not datetime'}

        recent = df.tail(200)
        hours  = recent.index.hour
        mask   = (hours >= start_hour) & (hours < end_hour)
        session_df = recent[mask]

        if session_df.empty:
            return {'valid': False, 'reason': 'No Asian session candles found'}

        # সর্বশেষ calendar day-এর session নাও
        last_day      = session_df.index.normalize().max()
        last_session  = session_df[session_df.index.normalize() == last_day]
        if last_session.empty:
            return {'valid': False, 'reason': 'No recent Asian session'}

        high = round(float(last_session['high'].max()), 5)
        low  = round(float(last_session['low'].min()), 5)

        return {
            'valid':  True,
            'high':   high,
            'low':    low,
            'range_pips': round((high - low) * 10000, 1),
            'session_date': str(last_day.date()),
            'note': f"Asian range {low:.5f}-{high:.5f} ({round((high-low)*10000,1)} pips)",
        }

    # ═══════════════════════════════════════════════════════
    # UTILS
    # ═══════════════════════════════════════════════════════

    def _safe_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            val = df['atr'].iloc[-1]
            if val and not np.isnan(val):
                return float(val)
        except Exception:
            pass
        try:
            highs, lows, closes = (
                df['high'].values[-period:],
                df['low'].values[-period:],
                df['close'].values[-period:],
            )
            trs = [max(h - l, abs(h - c), abs(l - c))
                   for h, l, c in zip(highs[1:], lows[1:], closes[:-1])]
            return float(np.mean(trs)) if trs else 0.0001
        except Exception:
            return 0.0001

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, eq_highs, eq_lows, prev_levels, asian) -> None:
        bar = "═" * 52
        log.info(bar)
        log.info("  💧  LIQUIDITY ZONE MAPPER  (Day 62)")
        log.info(bar)

        log.info("  ── Equal Highs (Buy-Side Liquidity) ──")
        if eq_highs:
            for h in eq_highs[:3]:
                log.info(f"  🔴 {h['price']}  touches={h['touches']}  ({h['candles_ago']} candles ago)")
        else:
            log.info("  None detected")

        log.info("  ── Equal Lows (Sell-Side Liquidity) ──")
        if eq_lows:
            for l in eq_lows[:3]:
                log.info(f"  🟢 {l['price']}  touches={l['touches']}  ({l['candles_ago']} candles ago)")
        else:
            log.info("  None detected")

        log.info("  ── Previous Day/Week Levels ──")
        log.info(f"  PDH={prev_levels.get('PDH')}  PDL={prev_levels.get('PDL')}")
        log.info(f"  PWH={prev_levels.get('PWH')}  PWL={prev_levels.get('PWL')}")

        log.info("  ── Asian Session Range ──")
        if asian.get('valid'):
            log.info(f"  High={asian['high']}  Low={asian['low']}  Range={asian['range_pips']} pips")
        else:
            log.info(f"  Unavailable: {asian.get('reason')}")

        log.info(bar)