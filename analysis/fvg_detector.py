# analysis/fvg_detector.py  —  Day 44 | Fair Value Gap (FVG) Detection
# ============================================================
# Fair Value Gap = ৩-candle imbalance pattern। দ্রুত move-এর কারণে
# candle 1 আর candle 3-এর মধ্যে একটা "ফাঁকা" zone থেকে যায়, যেটা price
# পরে এসে fill করতে পারে (mean-reversion magnet)।
#
#   Bullish FVG : candle3.low  > candle1.high   → gap [c1.high, c3.low]
#   Bearish FVG : candle3.high < candle1.low     → gap [c3.high, c1.low]
# ============================================================

import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("fvg_detector")


class FVGDetector:
    """
    Usage:
        detector = FVGDetector()
        fvgs = detector.detect(df)   # df-এ আগে থেকে 'atr' column থাকতে হবে
        nearest = detector.nearest_active(fvgs, current_price)
    """

    MIN_GAP_ATR_MULT = 0.10   # noise filter — gap অন্তত এই ATR fraction-এর সমান হতে হবে
    MAX_RESULTS       = 10
    PROXIMITY_ATR      = 0.3

    def detect(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < 10 or 'atr' not in df.columns:
            log.warning("[FVG] Insufficient data or missing ATR column")
            return []

        highs = df['high'].values
        lows  = df['low'].values
        atrs  = df['atr'].values
        n     = len(df)

        results = []

        for i in range(2, n):
            atr = atrs[i]
            if np.isnan(atr) or atr == 0:
                continue

            c1_high, c1_low = highs[i - 2], lows[i - 2]
            c3_high, c3_low = highs[i], lows[i]

            # Bullish FVG
            if c3_low > c1_high:
                gap = c3_low - c1_high
                if gap >= atr * self.MIN_GAP_ATR_MULT:
                    results.append(self._build(df, i, 'bullish', c1_high, c3_low))

            # Bearish FVG
            if c3_high < c1_low:
                gap = c1_low - c3_high
                if gap >= atr * self.MIN_GAP_ATR_MULT:
                    results.append(self._build(df, i, 'bearish', c3_high, c1_low))

        deduped = sorted(results, key=lambda r: r['index'], reverse=True)
        log.info(f"[FVG] Detected {len(deduped)} fair value gaps")
        return deduped[: self.MAX_RESULTS]

    def _build(self, df: pd.DataFrame, i: int, direction: str,
               zone_bottom: float, zone_top: float) -> dict:
        highs = df['high'].values
        lows  = df['low'].values
        n     = len(df)

        filled   = False
        fill_pct = 0.0
        for k in range(i + 1, n):
            if lows[k] <= zone_top and highs[k] >= zone_bottom:
                filled  = True
                overlap = min(highs[k], zone_top) - max(lows[k], zone_bottom)
                fill_pct = max(
                    fill_pct,
                    round(min(1.0, overlap / max(zone_top - zone_bottom, 1e-9)), 2),
                )

        return {
            'type':        'FVG',
            'direction':   direction.upper(),
            'index':       i,
            'zone_top':    round(float(zone_top), 5),
            'zone_bottom': round(float(zone_bottom), 5),
            'filled':      filled,
            'fresh':       not filled,
            'fill_pct':    fill_pct,
            'candles_ago': n - 1 - i,
        }

    # ─────────────────────────────────────────────
    # NEAREST ACTIVE GAP
    # ─────────────────────────────────────────────

    def nearest_active(self, fvgs: list[dict], current_price: float, atr: float = None) -> dict | None:
        fresh = [g for g in fvgs if g['fresh']]
        if not fresh:
            return None

        best      = None
        best_dist = float('inf')
        tolerance = (atr * self.PROXIMITY_ATR) if atr else 0.0

        for g in fresh:
            if g['zone_bottom'] <= current_price <= g['zone_top']:
                dist, in_zone = 0.0, True
            else:
                dist = min(
                    abs(current_price - g['zone_top']),
                    abs(current_price - g['zone_bottom']),
                )
                in_zone = dist <= tolerance

            if dist < best_dist:
                best_dist = dist
                best = {**g, 'distance': round(dist, 5), 'in_zone': in_zone}

        return best

    # ─────────────────────────────────────────────
    # PRINT SUMMARY
    # ─────────────────────────────────────────────

    def print_summary(self, fvgs: list[dict]) -> None:
        bar = "═" * 48
        log.info(bar)
        log.info("  🌀  FAIR VALUE GAP DETECTION  (Day 44)")
        log.info(bar)
        if not fvgs:
            log.info("  No fair value gaps detected.")
        for g in fvgs[:5]:
            icon = "🟢" if g['direction'] == 'BULLISH' else "🔴"
            tag  = "open" if g['fresh'] else f"filled {int(g['fill_pct']*100)}%"
            log.info(
                f"  {icon} FVG  [{g['zone_bottom']} - {g['zone_top']}]  "
                f"{tag}  ({g['candles_ago']} candles ago)"
            )
        log.info(bar)