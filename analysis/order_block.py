# analysis/order_block.py  —  Day 44 | Order Block Detection
import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("order_block")


class OrderBlockDetector:
    IMPULSE_ATR_MULT = 1.8
    LOOKBACK_FOR_OB  = 3
    MAX_RESULTS      = 10
    PROXIMITY_ATR    = 0.3

    def detect(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < 20 or 'atr' not in df.columns:
            log.warning("[OrderBlock] Insufficient data or missing ATR column")
            return []

        opens  = df['open'].values
        closes = df['close'].values
        highs  = df['high'].values
        lows   = df['low'].values
        atrs   = df['atr'].values
        n      = len(df)

        results = []

        for i in range(5, n):
            atr = atrs[i]
            if np.isnan(atr) or atr == 0:
                continue

            body = closes[i] - opens[i]
            if abs(body) < atr * self.IMPULSE_ATR_MULT:
                continue

            is_bullish_impulse = body > 0

            ob_idx = None
            for j in range(i - 1, max(i - 1 - self.LOOKBACK_FOR_OB, -1), -1):
                c_body = closes[j] - opens[j]
                if is_bullish_impulse and c_body < 0:
                    ob_idx = j
                    break
                if not is_bullish_impulse and c_body > 0:
                    ob_idx = j
                    break

            if ob_idx is None:
                continue

            zone_top    = float(highs[ob_idx])
            zone_bottom = float(lows[ob_idx])
            ob_type     = 'BULLISH_ORDER_BLOCK' if is_bullish_impulse else 'BEARISH_ORDER_BLOCK'

            mitigated = False
            for k in range(i + 1, n):
                if lows[k] <= zone_top and highs[k] >= zone_bottom:
                    mitigated = True
                    break

            results.append({
                'type':          ob_type,
                'direction':     'BULLISH' if is_bullish_impulse else 'BEARISH',
                'index':         ob_idx,
                'impulse_index': i,
                'zone_top':      round(zone_top, 5),
                'zone_bottom':   round(zone_bottom, 5),
                'mitigated':     mitigated,
                'fresh':         not mitigated,
                'candles_ago':   n - 1 - ob_idx,
            })

        seen_idx = set()
        deduped  = []
        for r in sorted(results, key=lambda r: r['index'], reverse=True):
            if r['index'] in seen_idx:
                continue
            seen_idx.add(r['index'])
            deduped.append(r)

        log.info(f"[OrderBlock] Detected {len(deduped)} order blocks")
        return deduped[: self.MAX_RESULTS]

    def nearest_active(self, order_blocks, current_price, atr=None):
        fresh = [ob for ob in order_blocks if ob['fresh']]
        if not fresh:
            return None

        best       = None
        best_dist  = float('inf')
        tolerance  = (atr * self.PROXIMITY_ATR) if atr else 0.0

        for ob in fresh:
            if ob['zone_bottom'] <= current_price <= ob['zone_top']:
                dist = 0.0
                in_zone = True
            else:
                dist = min(
                    abs(current_price - ob['zone_top']),
                    abs(current_price - ob['zone_bottom']),
                )
                in_zone = dist <= tolerance

            if dist < best_dist:
                best_dist = dist
                best = {**ob, 'distance': round(dist, 5), 'in_zone': in_zone}

        return best

    def print_summary(self, order_blocks):
        bar = "═" * 48
        log.info(bar)
        log.info("  🧱  ORDER BLOCK DETECTION  (Day 44)")
        log.info(bar)
        if not order_blocks:
            log.info("  No order blocks detected.")
        for ob in order_blocks[:5]:
            icon = "🟢" if ob['direction'] == 'BULLISH' else "🔴"
            tag  = "FRESH" if ob['fresh'] else "mitigated"
            log.info(
                f"  {icon} {ob['type']}  [{ob['zone_bottom']} - {ob['zone_top']}]  "
                f"{tag}  ({ob['candles_ago']} candles ago)"
            )
        log.info(bar)