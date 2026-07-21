# analysis/fibonacci.py
# ============================================================
# Day 40 — Fibonacci Engine (AI Price Zone Intelligence)
#
# Features:
#   ✅ Auto Swing High / Low Detection (Dynamic, TF-aware)
#   ✅ Fibonacci Retracement (23.6, 38.2, 50, 61.8, 78.6)
#   ✅ Fibonacci Extension (127.2, 161.8, 261.8)
#   ✅ Confluence Analysis (Fib + S/R + Structure)
#   ✅ Fibonacci Signal Integration (AI context)
#   ✅ Fibonacci Failure Detection
#   ✅ Memory-ready output (fib_history table)
# ============================================================
import os
import pandas as pd
import numpy as np
from utils.logger import get_logger
from utils.pip_utils import pip_size, price_to_pips

log = get_logger(__name__)

# ── Standard Fibonacci levels ──────────────────────────────────
FIB_RETRACEMENT_LEVELS = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
FIB_EXTENSION_LEVELS   = [1.0, 1.272, 1.618, 2.0, 2.618]

# ── Confluence tolerance (pips equivalent) ─────────────────────
# FIX (institutional review, item #1): this used to be a fixed 0.0015
# (i.e. "15 pips") regardless of instrument, which is correct for 5-decimal
# pairs (EURUSD, GBPUSD, ...) but wrong by ~100x for JPY crosses and metals.
# CONFLUENCE_TOLERANCE_PIPS is now the pip COUNT (instrument-independent);
# it gets converted to a price distance via pip_size(symbol) at call time.
# CONFLUENCE_TOLERANCE is kept as the old literal value for backward
# compatibility with any external code that imports this constant directly —
# it is no longer used internally by this module.
CONFLUENCE_TOLERANCE_PIPS = 15
CONFLUENCE_TOLERANCE = 0.0015   # deprecated — kept only for backward compat

# ── Dynamic swing window per timeframe ─────────────────────────
TF_SWING_WINDOW = {
    '1m':  3,
    '5m':  5,
    'M5':  5,
    '15m': 7,
    'M15': 7,
    '30m': 8,
    '1h':  10,
    'H1':  10,
    '4h':  14,
    'H4':  14,
    '1d':  20,
    'D1':  20,
}
DEFAULT_SWING_WINDOW = 7


class FibonacciEngine:
    """
    AI-powered Fibonacci analysis engine।

    Auto swing detect করে Fibonacci levels calculate করে,
    S/R confluence খোঁজে এবং trade signal দেয়।

    Usage:
        fib = FibonacciEngine(timeframe='H1')
        result = fib.analyze(df, sr_ctx=sr_ctx)
        ctx = fib.get_ai_context(result)
    """

    def __init__(self, timeframe: str = '15m', symbol: str = None):
        """
        symbol : optional instrument symbol (e.g. "EURUSD", "USDJPY",
            "XAUUSD"). Used only to pick the correct pip size for
            confluence/distance calculations. If omitted, pip math falls
            back to the standard 0.0001 pip size (previous behavior) —
            existing callers that don't pass a symbol are unaffected.
        """
        self.timeframe    = timeframe
        self.symbol       = symbol
        self.swing_window = TF_SWING_WINDOW.get(timeframe, DEFAULT_SWING_WINDOW)
        log.info(f"FibonacciEngine ready | TF={timeframe} | symbol={symbol} | swing_window={self.swing_window}")

    # ═══════════════════════════════════════════════════════════
    # MAIN ANALYSIS METHOD
    # ═══════════════════════════════════════════════════════════

    def analyze(
        self,
        df:       pd.DataFrame,
        sr_ctx:   dict = None,
        ind_ctx:  dict = None,
    ) -> dict:
        """
        Full Fibonacci analysis pipeline।

        Steps:
          1. Swing detect
          2. Retracement levels
          3. Extension targets
          4. Confluence zones
          5. Current price position
          6. Signal
        """
        if len(df) < self.swing_window * 3:
            return self._empty_result("Insufficient data")

        # Step 1: Auto swing detection
        swings = self.find_swing_points(df)
        if not swings['valid']:
            return self._empty_result("No significant swing detected")

        swing_high = swings['high']
        swing_low  = swings['low']
        trend      = swings['trend']

        # Step 2: Retracement levels
        retracements = self.calculate_retracement(swing_high, swing_low, trend)

        # Step 3: Extension targets
        extensions = self.calculate_extension(swing_high, swing_low, trend)

        # Step 4: Current price position
        curr_price = float(df['close'].iloc[-1])
        position   = self._price_position(curr_price, retracements, trend)

        # Step 5: Confluence with S/R
        confluence_zones = self.find_confluence(
            retracements, extensions, sr_ctx, curr_price
        )

        # Step 6: Signal
        signal = self._generate_signal(
            curr_price, position, confluence_zones, trend, ind_ctx
        )

        # Step 7: Failure detection
        failure_risk = self._detect_failure_risk(df, position, ind_ctx)

        result = {
            'swing':           swings,
            'swing_high':      swing_high,
            'swing_low':       swing_low,
            'trend':           trend,
            'range_pips':      round(price_to_pips(abs(swing_high - swing_low), self.symbol), 1),
            'retracements':    retracements,
            'extensions':      extensions,
            'curr_price':      curr_price,
            'position':        position,
            'confluence':      confluence_zones,
            'signal':          signal,
            'failure_risk':    failure_risk,
        }

        log.info(
            f"Fib Analysis | Swing: {swing_low:.5f}→{swing_high:.5f} "
            f"| Trend: {trend} | Position: {position.get('nearest_level')} "
            f"| Signal: {signal.get('bias')}"
        )
        return result

    # ═══════════════════════════════════════════════════════════
    # STEP 1: AUTO SWING DETECTION
    # ═══════════════════════════════════════════════════════════

    def find_swing_points(self, df: pd.DataFrame) -> dict:
        """
        Auto swing high / low detect করো।

        Dynamic window — timeframe অনুযায়ী আলাদা।
        Significant swing = ATR-এর কমপক্ষে 2x range।

        Returns both the most recent significant swing pair.
        """
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        n      = len(df)
        w      = self.swing_window

        # ATR for minimum size filter
        atr = self._calc_atr(df)

        # Find all local swing highs and lows
        swing_highs = []
        swing_lows  = []

        for i in range(w, n - w):
            # Swing High: highest point in window
            if highs[i] == max(highs[i - w: i + w + 1]):
                swing_highs.append((i, highs[i]))

            # Swing Low: lowest point in window
            if lows[i] == min(lows[i - w: i + w + 1]):
                swing_lows.append((i, lows[i]))

        if not swing_highs or not swing_lows:
            return {'valid': False, 'reason': 'No swings found'}

        # Pick most recent significant swing pair
        # Recent: within last 50% of data
        recent_cutoff = n // 2

        recent_highs = [(i, v) for i, v in swing_highs if i >= recent_cutoff]
        recent_lows  = [(i, v) for i, v in swing_lows  if i >= recent_cutoff]

        if not recent_highs:
            recent_highs = swing_highs
        if not recent_lows:
            recent_lows = swing_lows

        best_high_idx, best_high = max(recent_highs, key=lambda x: x[1])
        best_low_idx,  best_low  = min(recent_lows,  key=lambda x: x[1])

        # Minimum range: 2× ATR
        swing_range = best_high - best_low
        if swing_range < atr * 2:
            return {
                'valid':  False,
                'reason': f'Swing range {swing_range:.5f} too small (ATR={atr:.5f})'
            }

        # Trend: which came first?
        if best_low_idx < best_high_idx:
            trend = 'BULLISH'   # Low first → price moved up
        else:
            trend = 'BEARISH'   # High first → price moved down

        return {
            'valid':          True,
            'high':           round(best_high, 5),
            'low':            round(best_low, 5),
            'high_idx':       best_high_idx,
            'low_idx':        best_low_idx,
            'trend':          trend,
            'range':          round(swing_range, 5),
            'range_pips':     round(price_to_pips(swing_range, self.symbol), 1),
            'atr':            round(atr, 5),
            'all_highs':      [(i, round(v, 5)) for i, v in swing_highs],
            'all_lows':       [(i, round(v, 5)) for i, v in swing_lows],
        }

    # ═══════════════════════════════════════════════════════════
    # STEP 2: RETRACEMENT LEVELS
    # ═══════════════════════════════════════════════════════════

    def calculate_retracement(
        self,
        high:  float,
        low:   float,
        trend: str = 'BULLISH',
    ) -> dict:
        """
        Fibonacci Retracement levels calculate করো।

        Bullish retracement: high থেকে নিচে (pullback levels)
        Bearish retracement: low থেকে উপরে (bounce levels)

        Formula (Bullish):
            level = high - (high - low) × ratio
        Formula (Bearish):
            level = low + (high - low) × ratio
        """
        diff   = high - low
        levels = {}

        for ratio in FIB_RETRACEMENT_LEVELS:
            label = f"{ratio * 100:.1f}%"
            if trend == 'BULLISH':
                # Price fell from high — retracement levels going down
                price = high - diff * ratio
            else:
                # Price rose from low — retracement levels going up
                price = low + diff * ratio
            levels[label] = round(price, 5)

        return {
            'trend':   trend,
            'high':    high,
            'low':     low,
            'diff':    round(diff, 5),
            'levels':  levels,
            # Key levels shortcut
            '23.6':    levels['23.6%'],
            '38.2':    levels['38.2%'],
            '50.0':    levels['50.0%'],
            '61.8':    levels['61.8%'],
            '78.6':    levels['78.6%'],
        }

    # ═══════════════════════════════════════════════════════════
    # STEP 3: EXTENSION LEVELS
    # ═══════════════════════════════════════════════════════════

    def calculate_extension(
        self,
        high:  float,
        low:   float,
        trend: str = 'BULLISH',
    ) -> dict:
        """
        Fibonacci Extension — target levels বের করো।

        Bullish extension: swing low থেকে উপরে
        Bearish extension: swing high থেকে নিচে

        Formula (Bullish):
            level = low + diff × ratio
        Formula (Bearish):
            level = high - diff × ratio
        """
        diff   = high - low
        levels = {}

        for ratio in FIB_EXTENSION_LEVELS:
            label = f"{ratio * 100:.1f}%"
            if trend == 'BULLISH':
                price = low + diff * ratio
            else:
                price = high - diff * ratio
            levels[label] = round(price, 5)

        return {
            'trend':   trend,
            'levels':  levels,
            # Key targets shortcut
            'TP1':     levels['127.2%'],
            'TP2':     levels['161.8%'],
            'TP3':     levels['261.8%'],
        }

    # ═══════════════════════════════════════════════════════════
    # STEP 4: CONFLUENCE ANALYSIS
    # ═══════════════════════════════════════════════════════════

    def find_confluence(
        self,
        retracements: dict,
        extensions:   dict,
        sr_ctx:       dict = None,
        curr_price:   float = None,
    ) -> list[dict]:
        """
        Fib levels + S/R levels কাছাকাছি হলে → Confluence zone।

        Stronger confluence = more reasons → higher strength score.
        """
        confluence_zones = []
        tolerance        = CONFLUENCE_TOLERANCE_PIPS * pip_size(self.symbol)

        # All Fib levels (retracement + extension)
        all_fib = {}
        for label, price in retracements['levels'].items():
            all_fib[f"Fib {label}"] = price
        for label, price in extensions['levels'].items():
            all_fib[f"Ext {label}"] = price

        # S/R levels from context
        sr_levels = {}
        if sr_ctx:
            if sr_ctx.get('nearest_support'):
                sr_levels['Support']   = sr_ctx['nearest_support']
            if sr_ctx.get('nearest_resistance'):
                sr_levels['Resistance'] = sr_ctx['nearest_resistance']
            if sr_ctx.get('pivot'):
                sr_levels['Pivot']     = sr_ctx['pivot']
            if sr_ctx.get('R1'):
                sr_levels['R1']        = sr_ctx['R1']
            if sr_ctx.get('S1'):
                sr_levels['S1']        = sr_ctx['S1']

        # Find confluences
        for fib_name, fib_price in all_fib.items():
            reasons  = [fib_name]
            strength = self._fib_base_strength(fib_name)

            # Check S/R proximity
            for sr_name, sr_price in sr_levels.items():
                if abs(fib_price - sr_price) <= tolerance:
                    reasons.append(sr_name)
                    strength += 20

            # Check proximity to current price
            near_curr = False
            if curr_price:
                dist_pips = price_to_pips(abs(fib_price - curr_price), self.symbol)
                if dist_pips <= CONFLUENCE_TOLERANCE_PIPS:
                    near_curr = True
                    strength += 10

            # Only report if multiple reasons OR very strong level
            if len(reasons) >= 2 or strength >= 75:
                trend    = retracements['trend']
                zone_type = self._zone_type(fib_name, trend)

                confluence_zones.append({
                    'price':      fib_price,
                    'reasons':    reasons,
                    'strength':   min(99, strength),
                    'zone_type':  zone_type,
                    'near_price': near_curr,
                    'dist_pips':  round(price_to_pips(abs(fib_price - curr_price), self.symbol), 1) if curr_price else None,
                    'note':       (
                        f"{zone_type} at {fib_price:.5f} — "
                        f"{' + '.join(reasons)} (strength: {strength})"
                    ),
                })

        # Sort by strength descending
        confluence_zones.sort(key=lambda z: z['strength'], reverse=True)
        return confluence_zones

    def _fib_base_strength(self, fib_name: str) -> int:
        """Fibonacci level-এর base strength (কোনটা বেশি react করে)"""
        strength_map = {
            '61.8': 80,   # Golden ratio — most important
            '50.0': 70,
            '38.2': 65,
            '78.6': 60,
            '23.6': 50,
            '127.2': 65,  # Extension targets
            '161.8': 75,  # Golden ratio extension
            '261.8': 55,
            '100.0': 55,
            '0.0':   45,
        }
        for key, val in strength_map.items():
            if key in fib_name:
                return val
        return 40

    def _zone_type(self, fib_name: str, trend: str) -> str:
        """Fib level + trend দিয়ে zone type বলো"""
        retracement_labels = ['23.6', '38.2', '50.0', '61.8', '78.6']
        extension_labels   = ['127.2', '161.8', '261.8']

        is_retracement = any(l in fib_name for l in retracement_labels)
        is_extension   = any(l in fib_name for l in extension_labels)

        if is_retracement:
            return 'BUY_ZONE' if trend == 'BULLISH' else 'SELL_ZONE'
        if is_extension:
            return 'TARGET_ZONE'
        return 'ZONE'

    # ═══════════════════════════════════════════════════════════
    # STEP 5: PRICE POSITION
    # ═══════════════════════════════════════════════════════════

    def _price_position(
        self,
        curr_price:   float,
        retracements: dict,
        trend:        str,
    ) -> dict:
        """
        Current price কোন Fib level-এর কাছে আছে বলো।
        """
        levels = retracements['levels']
        nearest_label = None
        nearest_price = None
        nearest_dist  = float('inf')

        for label, price in levels.items():
            dist = abs(curr_price - price)
            if dist < nearest_dist:
                nearest_dist  = dist
                nearest_label = label
                nearest_price = price

        nearest_pips = round(price_to_pips(nearest_dist, self.symbol), 1)

        # Which zone is price in?
        high = retracements['high']
        low  = retracements['low']
        diff = high - low

        if diff == 0:
            ratio = 0.5
        else:
            if trend == 'BULLISH':
                ratio = (high - curr_price) / diff
            else:
                ratio = (curr_price - low) / diff

        # Zone categorization
        if ratio <= 0.0:
            zone = 'ABOVE_HIGH'
        elif ratio <= 0.236:
            zone = 'SHALLOW_RETRACEMENT'
        elif ratio <= 0.382:
            zone = 'MINOR_RETRACEMENT'
        elif ratio <= 0.500:
            zone = 'MODERATE_RETRACEMENT'
        elif ratio <= 0.618:
            zone = 'GOLDEN_ZONE'          # 50-61.8 is the golden zone
        elif ratio <= 0.786:
            zone = 'OTE_ZONE'             # Day 81+ — OTE (Optimal Trade Entry) 61.8-78.6%
        elif ratio <= 1.0:
            zone = 'NEAR_SWING_LOW'
        else:
            zone = 'BELOW_LOW'

        return {
            'nearest_level': nearest_label,
            'nearest_price': nearest_price,
            'nearest_pips':  nearest_pips,
            'ratio':         round(ratio, 4),
            'zone':          zone,
            'in_golden_zone': 0.500 <= ratio <= 0.618,
            # Day 81+ — OTE (Optimal Trade Entry) zone: 61.8%-78.6%
            # This is the ICT "sweet spot" for sniper entries.
            'in_ote_zone':   0.618 <= ratio <= 0.786,
        }

    # ═══════════════════════════════════════════════════════════
    # STEP 6: SIGNAL GENERATION
    # ═══════════════════════════════════════════════════════════

    def _generate_signal(
        self,
        curr_price:       float,
        position:         dict,
        confluence_zones: list,
        trend:            str,
        ind_ctx:          dict = None,
    ) -> dict:
        """
        Fib position + confluence + indicator দেখে signal দাও।
        """
        zone     = position.get('zone', '')
        in_gold  = position.get('in_golden_zone', False)
        ratio    = position.get('ratio', 0.5)

        # Base bias from Fibonacci position
        if trend == 'BULLISH':
            if 0.382 <= ratio <= 0.786:
                bias = 'BUY'
                conf = 65
            elif ratio > 0.786:
                bias = 'WAIT'   # Too deep — swing may be invalid
                conf = 40
            elif ratio < 0.236:
                bias = 'WAIT'   # Barely retraced — wait for more pullback
                conf = 45
            else:
                bias = 'BUY'
                conf = 55
        else:  # BEARISH
            if 0.382 <= ratio <= 0.786:
                bias = 'SELL'
                conf = 65
            elif ratio > 0.786:
                bias = 'WAIT'
                conf = 40
            elif ratio < 0.236:
                bias = 'WAIT'
                conf = 45
            else:
                bias = 'SELL'
                conf = 55

        # Golden zone bonus
        if in_gold:
            conf += 12

        # Confluence bonus
        top_confluence = confluence_zones[0] if confluence_zones else None
        if top_confluence:
            if top_confluence.get('near_price'):
                conf += top_confluence['strength'] // 10

        # Indicator alignment bonus
        if ind_ctx:
            rsi    = ind_ctx.get('rsi', 50)
            trend_ = ind_ctx.get('trend', '')
            macd_c = ind_ctx.get('macd_cross', '')

            if bias == 'BUY':
                if rsi < 50 and 'bullish' in trend_:   conf += 8
                if 'bullish_cross' in macd_c:           conf += 6
                if rsi > 70:                            conf -= 10  # overbought
            elif bias == 'SELL':
                if rsi > 50 and 'bearish' in trend_:   conf += 8
                if 'bearish_cross' in macd_c:           conf += 6
                if rsi < 30:                            conf -= 10  # oversold

        conf = max(0, min(99, conf))

        # Entry / SL / TP suggestion
        entry = curr_price
        atr   = ind_ctx.get('atr', 0.0010) if ind_ctx else 0.0010

        if bias == 'BUY':
            sl     = round(position['nearest_price'] - atr * 1.5, 5)
            tp1    = None   # Extension targets used instead
            reason = f"Price in Fib {zone} zone ({ratio*100:.1f}%) — bullish retracement"
        elif bias == 'SELL':
            sl     = round(position['nearest_price'] + atr * 1.5, 5)
            tp1    = None
            reason = f"Price in Fib {zone} zone ({ratio*100:.1f}%) — bearish retracement"
        else:
            sl     = None
            tp1    = None
            reason = f"Fib zone {zone} — wait for better position"

        return {
            'bias':            bias,
            'confidence':      conf,
            'zone':            zone,
            'in_golden_zone':  in_gold,
            'entry':           round(entry, 5),
            'sl':              sl,
            'reason':          reason,
            'top_confluence':  top_confluence,
        }

    # ═══════════════════════════════════════════════════════════
    # STEP 7: FAILURE RISK DETECTION
    # ═══════════════════════════════════════════════════════════

    def _detect_failure_risk(
        self,
        df:       pd.DataFrame,
        position: dict,
        ind_ctx:  dict = None,
    ) -> dict:
        """
        Fibonacci level fail করার risk detect করো।

        High risk conditions:
          - High volatility (large ATR)
          - RSI extreme opposite direction
          - Price below 78.6% (very deep retracement)
          - News event (external — caller must inject)
        """
        risks   = []
        risk_score = 0

        ratio = position.get('ratio', 0.5)

        # Deep retracement risk
        if ratio > 0.786:
            risks.append("Price below 78.6% — swing may be invalidated")
            risk_score += 30

        # Volatility check
        if ind_ctx:
            atr   = ind_ctx.get('atr', 0)
            price = ind_ctx.get('price', 1)
            atr_pct = atr / max(price, 1e-5) * 100

            if atr_pct > 0.15:
                risks.append(f"High volatility (ATR={atr_pct:.2f}%) — Fib levels less reliable")
                risk_score += 20

            rsi = ind_ctx.get('rsi', 50)
            trend = ind_ctx.get('trend', '')

            if ratio < 0.618 and 'strong_bearish' in trend:
                risks.append("Shallow retracement in strong bearish trend — likely to break lower")
                risk_score += 15

            if ratio < 0.618 and 'strong_bullish' in trend:
                risks.append("Shallow retracement in strong bullish trend — may not retrace deeper")
                risk_score += 10

        risk_level = 'HIGH' if risk_score >= 40 else ('MEDIUM' if risk_score >= 20 else 'LOW')

        return {
            'risk_score':  risk_score,
            'risk_level':  risk_level,
            'risks':       risks,
            'note':        f"Fib failure risk: {risk_level} ({risk_score}/100)",
        }

    # ═══════════════════════════════════════════════════════════
    # AI CONTEXT — Integration
    # ═══════════════════════════════════════════════════════════

    def get_ai_context(self, result: dict) -> dict:
        """
        DecisionAgent, SignalEngine, MarketBiasEngine-এর জন্য
        Fibonacci context।
        """
        if not result.get('swing', {}).get('valid'):
            return {
                'fib_valid':        False,
                'fib_bias':         'NEUTRAL',
                'fib_confidence':   0,
                'fib_zone':         'NONE',
                'fib_level_near':   None,
                'fib_in_golden':    False,
                'fib_confluence':   0,
                'fib_confluence_strength': 0,
                'fib_signal':       'WAIT',
                'fib_failure_risk': 'UNKNOWN',
                'fib_swing_high':   None,
                'fib_swing_low':    None,
                'fib_trend':        'NEUTRAL',
                'fib_61_8':         None,
                'fib_50_0':         None,
                'fib_38_2':         None,
                'fib_tp1':          None,
                'fib_tp2':          None,
            }

        signal   = result.get('signal', {})
        position = result.get('position', {})
        retrace  = result.get('retracements', {})
        ext      = result.get('extensions', {})
        conf_z   = result.get('confluence', [])
        failure  = result.get('failure_risk', {})

        top_conf = conf_z[0] if conf_z else {}

        return {
            'fib_valid':              True,
            'fib_bias':               signal.get('bias', 'WAIT'),
            'fib_confidence':         signal.get('confidence', 0),
            'fib_zone':               position.get('zone', 'UNKNOWN'),
            'fib_level_near':         position.get('nearest_level'),
            'fib_level_near_price':   position.get('nearest_price'),
            'fib_level_near_pips':    position.get('nearest_pips'),
            'fib_in_golden':          position.get('in_golden_zone', False),
            # Day 81+ — OTE zone for ICT sniper entries
            'fib_in_ote':             position.get('in_ote_zone', False),
            'fib_confluence':         len(conf_z),
            'fib_confluence_strength': top_conf.get('strength', 0),
            'fib_confluence_note':    top_conf.get('note', ''),
            'fib_signal':             signal.get('bias', 'WAIT'),
            'fib_signal_reason':      signal.get('reason', ''),
            'fib_failure_risk':       failure.get('risk_level', 'LOW'),
            'fib_failure_score':      failure.get('risk_score', 0),
            # Swing info
            'fib_swing_high':         result.get('swing_high'),
            'fib_swing_low':          result.get('swing_low'),
            'fib_trend':              result.get('trend', 'NEUTRAL'),
            'fib_range_pips':         result.get('range_pips', 0),
            # Key levels
            'fib_61_8':               retrace.get('61.8'),
            'fib_50_0':               retrace.get('50.0'),
            'fib_38_2':               retrace.get('38.2'),
            'fib_78_6':               retrace.get('78.6'),
            'fib_23_6':               retrace.get('23.6'),
            # Targets
            'fib_tp1':                ext.get('TP1'),
            'fib_tp2':                ext.get('TP2'),
            'fib_tp3':                ext.get('TP3'),
        }

    # ═══════════════════════════════════════════════════════════
    # MEMORY — fib_history table format
    # ═══════════════════════════════════════════════════════════

    def get_memory_record(
        self,
        result:    dict,
        pair:      str,
        outcome:   str = None,   # 'WIN' / 'LOSS' / None (open)
        profit_pips: float = None,
    ) -> dict:
        """
        Database-এ save করার জন্য fib_history record।

        Day 52-53 Memory Integration-এ use হবে।
        """
        position = result.get('position', {})
        signal   = result.get('signal', {})

        return {
            'pair':          pair,
            'timeframe':     self.timeframe,
            'swing_high':    result.get('swing_high'),
            'swing_low':     result.get('swing_low'),
            'fib_trend':     result.get('trend'),
            'fib_level':     position.get('nearest_level'),
            'fib_zone':      position.get('zone'),
            'in_golden':     position.get('in_golden_zone', False),
            'confluence':    len(result.get('confluence', [])),
            'conf_strength': result.get('confluence', [{}])[0].get('strength', 0)
                             if result.get('confluence') else 0,
            'signal':        signal.get('bias'),
            'confidence':    signal.get('confidence'),
            'failure_risk':  result.get('failure_risk', {}).get('risk_level'),
            'outcome':       outcome,
            'profit_pips':   profit_pips,
        }

    # ═══════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════════

    def print_summary(self, result: dict):
        if not result.get('swing', {}).get('valid'):
            print("\n  ⚠️  Fibonacci: No valid swing detected.\n")
            return

        swing  = result['swing']
        retrace = result['retracements']
        ext    = result['extensions']
        pos    = result['position']
        sig    = result['signal']
        conf_z = result['confluence']
        fail   = result['failure_risk']
        price  = result['curr_price']

        trend_icon = '▲' if result['trend'] == 'BULLISH' else '▼'

        print("\n" + "═" * 58)
        print("  📐  FIBONACCI ENGINE  (Day 40)")
        print("═" * 58)
        print(f"  Pair/TF       :  {self.timeframe}")
        print(f"  Swing         :  {trend_icon} {result['trend']}  "
              f"| H={swing['high']:.5f}  L={swing['low']:.5f}  "
              f"| Range={swing['range_pips']:.1f} pips")
        print()

        # Retracement levels
        print("  ── Retracement Levels ──")
        levels_sorted = sorted(
            retrace['levels'].items(),
            key=lambda x: x[1],
            reverse=(result['trend'] == 'BULLISH')
        )
        for label, price_lvl in levels_sorted:
            dist   = price_to_pips(abs(price - price_lvl), self.symbol)
            marker = ' ◄ PRICE' if dist < 5 else (f'  ({dist:.1f}p away)' if dist < 30 else '')
            bold   = ' ⭐' if '61.8' in label or '50.0' in label else ''
            print(f"  {label:<8}  {price_lvl:.5f}{bold}{marker}")

        print()
        print("  ── Extension Targets ──")
        for label, price_lvl in ext['levels'].items():
            if float(label.replace('%', '')) > 100:
                tag = ' (TP1)' if '127' in label else (' (TP2)' if '161' in label else ' (TP3)')
                print(f"  Ext {label:<6}  {price_lvl:.5f}{tag}")

        # Position
        print()
        print("  ── Current Position ──")
        golden_tag = '  🌟 GOLDEN ZONE' if pos.get('in_golden_zone') else ''
        print(f"  Price         :  {price:.5f}")
        print(f"  Zone          :  {pos['zone']}{golden_tag}")
        print(f"  Nearest Fib   :  {pos['nearest_level']} ({pos['nearest_pips']:.1f} pips away)")

        # Confluence zones
        if conf_z:
            print()
            print("  ── Confluence Zones ──")
            for z in conf_z[:4]:
                icon = '🔥' if z['strength'] >= 80 else ('⚡' if z['strength'] >= 65 else '💡')
                near = ' ◄ NEAR' if z.get('near_price') else ''
                print(f"  {icon}  {z['price']:.5f}  str={z['strength']}  "
                      f"{' + '.join(z['reasons'][:3])}{near}")

        # Failure risk
        if fail['risks']:
            print()
            print("  ── Failure Risks ──")
            for r in fail['risks']:
                print(f"  ⚠️  {r}")

        # Signal
        print()
        bias_icon = {'BUY': '🟢', 'SELL': '🔴', 'WAIT': '🟡'}.get(sig['bias'], '⬜')
        print(f"  ┌──────────────────────────────────────────────────┐")
        print(f"  │  {bias_icon} {sig['bias']:<6}  |  Confidence: {sig['confidence']}%              │")
        print(f"  │  {sig['reason'][:52]:<52}│")
        if sig.get('sl'):
            print(f"  │  SL: {sig['sl']:<50}│")
        print(f"  │  Failure Risk: {fail['risk_level']:<37}│")
        print(f"  └──────────────────────────────────────────────────┘")
        print("═" * 58 + "\n")

    # ═══════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════

    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """ATR calculate করো — column থাকলে নেও, না থাকলে calculate করো।"""
        if 'atr' in df.columns:
            val = df['atr'].iloc[-1]
            if not np.isnan(val):
                return float(val)

        highs  = df['high'].values[-period:]
        lows   = df['low'].values[-period:]
        closes = df['close'].values[-period:]
        trs = [
            max(h - l, abs(h - c), abs(l - c))
            for h, l, c in zip(highs[1:], lows[1:], closes[:-1])
        ]
        return float(np.mean(trs)) if trs else 0.0001

    def _empty_result(self, reason: str) -> dict:
        return {
            'swing':        {'valid': False, 'reason': reason},
            'retracements': {},
            'extensions':   {},
            'position':     {},
            'confluence':   [],
            'signal':       {'bias': 'WAIT', 'confidence': 0, 'reason': reason},
            'failure_risk': {'risk_level': 'UNKNOWN', 'risk_score': 0, 'risks': []},
        }


# ═══════════════════════════════════════════════════════════════
# QUICK RUN — Direct test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data.fetcher import DataFetcher
    from data.indicators import Indicators
    from analysis.support_resistance import SupportResistance

    fetcher = DataFetcher()
    ind     = Indicators()
    sr_eng  = SupportResistance()

    df = fetcher.fetch_ohlcv("EURUSD", "1h", limit=200)
    if df is not None:
        df      = ind.add_all(df)
        ind_ctx = ind.get_ai_context(df)
        sr_res  = sr_eng.analyze(df)
        sr_ctx  = sr_eng.get_ai_context(sr_res)

        fib    = FibonacciEngine(timeframe='1h')
        result = fib.analyze(df, sr_ctx=sr_ctx, ind_ctx=ind_ctx)
        fib.print_summary(result)

        ctx = fib.get_ai_context(result)
        print("AI Context (for DecisionAgent):")
        for k, v in ctx.items():
            print(f"  {k:<30}: {v}")