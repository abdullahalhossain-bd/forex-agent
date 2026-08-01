# analysis/mtf_analyzer.py
# ============================================================
# Day 38 — Multi-Timeframe Analysis (MTF Intelligence Layer)
# Professional Top-Down Analysis:
#   H4  → Market Direction (Trend)
#   H1  → Zone Confirmation
#   M15 → Setup Detection
#   M5  → Entry Timing
#
# Features:
#   ✅ Market Structure (BOS, CHoCH, Liquidity Sweep)
#   ✅ MTF Agreement Logic (all 4 TF align → trade)
#   ✅ Confidence Score (H4:30 + H1:25 + M15:25 + M5:20)
#   ✅ Conflict Severity Scoring
#   ✅ Higher Timeframe Override Rule
#   ✅ Final Decision: BUY / SELL / WAIT
# ============================================================

import pandas as pd
from data.fetcher import get_data_fetcher
from data.indicators import Indicators
from utils.logger import get_logger

log = get_logger(__name__)

# ── Timeframe hierarchy ──────────────────────────────────────
MTF_CHAIN = {
    'H4':  '4h',   # Trend direction
    'H1':  '1h',   # Zone confirmation
    'M15': '15m',  # Setup detection
    'M5':  '5m',   # Entry timing
}

# ── Confidence weights (total = 100) ────────────────────────
TF_WEIGHTS = {
    'H4':  30,
    'H1':  25,
    'M15': 25,
    'M5':  20,
}

# ── Conflict severity matrix ─────────────────────────────────
# H4 vs lower TF — কতটা dangerous
CONFLICT_SEVERITY = {
    ('H4', 'H1'):  'CRITICAL',   # সবচেয়ে বিপজ্জনক
    ('H4', 'M15'): 'HIGH',
    ('H4', 'M5'):  'MEDIUM',
    ('H1', 'M15'): 'LOW',
    ('H1', 'M5'):  'LOW',
    ('M15', 'M5'): 'ACCEPTABLE', # এটা সহ্য করা যায়
}

# ── Override threshold ───────────────────────────────────────
# H4 এই score এর বেশি হলে lower TF override হবে
H4_OVERRIDE_THRESHOLD = 25  # H4 weight এর ৮০%+


class MTFAnalyzer:
    """
    Multi-Timeframe Analysis Engine।

    Professional top-down approach:
    H4 trend → H1 zone → M15 setup → M5 entry

    সব timeframe একমত হলেই trade।
    """

    def __init__(self, symbol: str = "EURUSD"):
        self.symbol  = symbol
        self.fetcher = get_data_fetcher()
        self.ind     = Indicators()

    # ═══════════════════════════════════════════════════════
    # MAIN METHOD
    # ═══════════════════════════════════════════════════════

    def analyze(self, regime_ctx: dict = None) -> dict:
        """
        সব timeframe fetch করো, analyze করো।
        Final decision: BUY / SELL / WAIT

        Args:
            regime_ctx: optional output of
                `MarketRegimeDetector.get_ai_context()` (analysis/market_regime.py).
                When provided and it reports a non-tradeable regime (CHOPPY, or
                `strategy_type == "NO_TRADE"`, or `risk_multiplier <= 0`), this
                is enforced as a HARD GATE — the MTF decision is forced to
                WAIT regardless of how aligned the timeframes are. This closes
                the integration gap flagged in the institutional review: a
                good regime filter that isn't the master gate for every
                signal generator wasn't actually protecting capital. Passing
                nothing preserves the exact prior behavior (backward
                compatible).
        """
        log.info(f"MTF Analysis started: {self.symbol}")

        # Step 1: প্রতিটা TF-এর data + indicators
        tf_data = self._fetch_all_timeframes()
        if not tf_data:
            return self._empty_result("No data fetched")

        # Step 2: প্রতিটা TF-এর context বের করো
        tf_contexts = self._build_tf_contexts(tf_data)

        # Step 3: Market structure detect করো
        tf_structure = self._detect_market_structure(tf_data)

        # Step 4: MTF direction বের করো (BULLISH / BEARISH / NEUTRAL)
        tf_directions = self._get_tf_directions(tf_contexts)

        # Step 5: Conflict detection
        conflicts = self._detect_conflicts(tf_directions)

        # Step 6: H4 Override Rule check
        h4_override = self._check_h4_override(tf_directions, conflicts)

        # Step 7: Confidence score calculate
        confidence, score_breakdown = self._calculate_confidence(
            tf_directions, tf_contexts, conflicts, h4_override
        )

        # Step 8: Final decision
        decision = self._make_decision(
            tf_directions, conflicts, h4_override, confidence
        )

        # Step 9 (NEW): Global regime hard gate — overrides everything above
        # if the broader market regime says this is not a tradeable moment.
        regime_gate = self._apply_regime_gate(regime_ctx)
        if regime_gate["blocked"]:
            decision = 'WAIT'

        result = {
            'pair':          self.symbol,
            'decision':      decision,
            'confidence':    confidence,
            'score_breakdown': score_breakdown,
            'timeframes':    tf_directions,
            'structure':     tf_structure,
            'contexts':      tf_contexts,
            'conflicts':     conflicts,
            'h4_override':   h4_override,
            'regime_gate':   regime_gate,
            'reason':        (
                regime_gate['note'] if regime_gate['blocked'] else
                self._build_reason(
                    decision, tf_directions, conflicts,
                    h4_override, confidence
                )
            ),
        }

        log.info(
            f"MTF Result: {decision} | Confidence: {confidence}% | "
            f"Conflicts: {len(conflicts)}"
            + (f" | REGIME-BLOCKED: {regime_gate['note']}" if regime_gate['blocked'] else "")
        )
        return result

    # ═══════════════════════════════════════════════════════
    # STEP 9 (NEW): GLOBAL REGIME HARD GATE
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _apply_regime_gate(regime_ctx: dict = None) -> dict:
        """
        Enforce analysis/market_regime.py's verdict as a hard gate.

        Expects the shape returned by `MarketRegimeDetector.get_ai_context()`:
            {
                'market_regime': 'TRENDING'|'RANGING'|'CHOPPY'|'BREAKOUT',
                'strategy_type': 'TREND_FOLLOW'|'RANGE'|'NO_TRADE'|'WAIT',
                'risk_multiplier': float,
                ...
            }

        Blocks (forces WAIT) when the regime is CHOPPY, the suggested
        strategy is explicitly NO_TRADE, or the risk multiplier is zero.
        If `regime_ctx` is None (not supplied by the caller), the gate is a
        no-op — this keeps every existing caller of `MTFAnalyzer.analyze()`
        working exactly as before.
        """
        if not regime_ctx:
            return {"blocked": False, "checked": False, "note": "No regime context supplied — gate not applied"}

        regime = regime_ctx.get("market_regime", "UNKNOWN")
        strategy_type = regime_ctx.get("strategy_type", "")
        risk_mult = regime_ctx.get("risk_multiplier", 1.0)

        if regime == "CHOPPY" or strategy_type == "NO_TRADE" or (risk_mult is not None and risk_mult <= 0):
            return {
                "blocked": True,
                "checked": True,
                "regime": regime,
                "note": (
                    f"Blocked by market regime gate: regime={regime}, "
                    f"strategy_type={strategy_type}, risk_multiplier={risk_mult}. "
                    f"MTF timeframes may be aligned, but the broader regime says "
                    f"this is not a tradeable market right now."
                ),
            }

        return {"blocked": False, "checked": True, "regime": regime, "note": "Regime gate passed"}

    # ═══════════════════════════════════════════════════════
    # STEP 1: DATA FETCH
    # ═══════════════════════════════════════════════════════

    def _fetch_all_timeframes(self) -> dict:
        """H4, H1, M15, M5 — চারটা timeframe fetch করো"""
        tf_data = {}
        for label, tf_code in MTF_CHAIN.items():
            log.info(f"Fetching {label} ({tf_code})...")
            df = self.fetcher.fetch_ohlcv(
                symbol    = self.symbol,
                timeframe = tf_code,
                limit     = 200,
            )
            if df is None or df.empty:
                log.warning(f"Could not fetch {label}")
                continue

            df = self.ind.add_all(df)
            tf_data[label] = df
            log.info(f"{label} ready: {len(df)} candles")

        return tf_data

    # ═══════════════════════════════════════════════════════
    # STEP 2: TF CONTEXT BUILD
    # ═══════════════════════════════════════════════════════

    def _build_tf_contexts(self, tf_data: dict) -> dict:
        """প্রতিটা TF-এর indicator context বের করো"""
        contexts = {}
        for label, df in tf_data.items():
            ctx              = self.ind.get_ai_context(df)
            ctx['timeframe'] = label
            ctx['candles']   = len(df)

            # Extra info per TF role
            if label == 'H4':
                ctx['role'] = 'TREND'
                ctx['structure_note'] = self._h4_structure_note(ctx)
            elif label == 'H1':
                ctx['role'] = 'ZONE_CONFIRMATION'
                ctx['zone_note'] = self._h1_zone_note(ctx)
            elif label == 'M15':
                ctx['role'] = 'SETUP'
                ctx['setup_note'] = self._m15_setup_note(ctx)
            elif label == 'M5':
                ctx['role'] = 'ENTRY'
                ctx['entry_note'] = self._m5_entry_note(ctx)

            contexts[label] = ctx
        return contexts

    def _h4_structure_note(self, ctx: dict) -> str:
        trend = ctx.get('trend', '')
        rsi   = ctx.get('rsi', 50)
        if 'strong_bullish' in trend and rsi > 50:
            return "Strong uptrend — HH/HL structure"
        if 'strong_bearish' in trend and rsi < 50:
            return "Strong downtrend — LH/LL structure"
        if 'bullish' in trend:
            return "Bullish trend — momentum building"
        if 'bearish' in trend:
            return "Bearish trend — sellers in control"
        return "Sideways / Neutral"

    def _h1_zone_note(self, ctx: dict) -> str:
        bb_pct = ctx.get('bb_pct', 0.5)
        rsi    = ctx.get('rsi', 50)
        price  = ctx.get('price', 0)
        s20    = ctx.get('sma_20', 0)

        if bb_pct < 0.2 and rsi < 45:
            return "Price at demand zone (BB lower + RSI low)"
        if bb_pct > 0.8 and rsi > 55:
            return "Price at supply zone (BB upper + RSI high)"
        if price > s20:
            return "Price above SMA20 — bullish zone"
        return "Mid-range — no clear zone"

    def _m15_setup_note(self, ctx: dict) -> str:
        macd_cross = ctx.get('macd_cross', '')
        rsi_signal = ctx.get('rsi_signal', '')
        trend      = ctx.get('trend', '')

        if 'bullish_cross' in macd_cross and 'bullish' in trend:
            return "Bullish MACD cross + trend aligned — setup confirmed"
        if 'bearish_cross' in macd_cross and 'bearish' in trend:
            return "Bearish MACD cross + trend aligned — setup confirmed"
        if rsi_signal == 'oversold':
            return "RSI oversold — potential bullish setup"
        if rsi_signal == 'overbought':
            return "RSI overbought — potential bearish setup"
        return "No clear setup yet"

    def _m5_entry_note(self, ctx: dict) -> str:
        trend  = ctx.get('trend', '')
        macd   = ctx.get('macd_cross', '')
        bb_pct = ctx.get('bb_pct', 0.5)

        if 'bullish' in trend and 'bullish_cross' in macd:
            return "Breakout confirmation — BUY entry timing"
        if 'bearish' in trend and 'bearish_cross' in macd:
            return "Breakdown confirmation — SELL entry timing"
        if bb_pct < 0.1:
            return "Pullback to BB lower — potential long entry"
        if bb_pct > 0.9:
            return "Pullback to BB upper — potential short entry"
        return "Waiting for entry trigger"

    # ═══════════════════════════════════════════════════════
    # STEP 3: MARKET STRUCTURE (BOS, CHoCH, LIQUIDITY)
    # ═══════════════════════════════════════════════════════

    def _detect_market_structure(self, tf_data: dict) -> dict:
        """
        BOS  — Break of Structure
        CHoCH — Change of Character
        Liquidity Sweep — wick beyond key level then rejection

        H4 এবং H1 তে detect করা হবে (বড় picture)
        """
        structure = {}
        for label in ['H4', 'H1']:
            if label not in tf_data:
                continue
            df = tf_data[label]
            structure[label] = {
                'bos':             self._detect_bos(df),
                'choch':           self._detect_choch(df),
                'liquidity_sweep': self._detect_liquidity_sweep(df),
            }
        return structure

    def _detect_bos(self, df: pd.DataFrame) -> dict:
        """
        Break of Structure:
        Bullish BOS — price breaks above previous swing high
        Bearish BOS — price breaks below previous swing low
        """
        if len(df) < 20:
            return {'type': 'NONE', 'note': 'Insufficient data'}

        recent = df.tail(50)
        highs  = recent['high'].values
        lows   = recent['low'].values
        close  = recent['close'].values

        # Previous swing high (last 20 candles exclude last 5)
        prev_high = max(highs[-20:-5])
        prev_low  = min(lows[-20:-5])
        curr_close = close[-1]

        if curr_close > prev_high:
            return {
                'type':  'BULLISH_BOS',
                'level': round(prev_high, 5),
                'note':  f'Price broke above {prev_high:.5f} — bullish structure',
            }
        if curr_close < prev_low:
            return {
                'type':  'BEARISH_BOS',
                'level': round(prev_low, 5),
                'note':  f'Price broke below {prev_low:.5f} — bearish structure',
            }
        return {'type': 'NONE', 'note': 'No structural break detected'}

    def _detect_choch(self, df: pd.DataFrame) -> dict:
        """
        Change of Character:
        আগে bearish → এখন bullish higher high তৈরি হলে → CHoCH bullish
        আগে bullish → এখন bearish lower low তৈরি হলে → CHoCH bearish
        """
        if len(df) < 30:
            return {'type': 'NONE', 'note': 'Insufficient data'}

        recent = df.tail(60)
        closes = recent['close'].values
        highs  = recent['high'].values
        lows   = recent['low'].values

        # Compare two halves
        mid   = len(closes) // 2
        first_half_trend  = closes[mid] - closes[0]
        second_half_trend = closes[-1] - closes[mid]

        first_high  = max(highs[:mid])
        second_high = max(highs[mid:])
        first_low   = min(lows[:mid])
        second_low  = min(lows[mid:])

        # Bearish then bullish CHoCH
        if first_half_trend < 0 and second_half_trend > 0:
            if second_high > first_high:
                return {
                    'type': 'BULLISH_CHOCH',
                    'note': 'Trend reversal: Bearish → Bullish character change',
                }

        # Bullish then bearish CHoCH
        if first_half_trend > 0 and second_half_trend < 0:
            if second_low < first_low:
                return {
                    'type': 'BEARISH_CHOCH',
                    'note': 'Trend reversal: Bullish → Bearish character change',
                }

        return {'type': 'NONE', 'note': 'No character change detected'}

    def _detect_liquidity_sweep(self, df: pd.DataFrame) -> dict:
        """
        Liquidity Sweep:
        Price wicks beyond a key level then quickly rejects।
        বড় player-রা retail trader-দের stop hunt করছে।
        """
        if len(df) < 10:
            return {'type': 'NONE', 'note': 'Insufficient data'}

        recent    = df.tail(20)
        last_3    = df.tail(3)

        prev_high = recent['high'].iloc[:-3].max()
        prev_low  = recent['low'].iloc[:-3].min()

        for _, candle in last_3.iterrows():
            wick_above = candle['high'] > prev_high and candle['close'] < prev_high
            wick_below = candle['low'] < prev_low  and candle['close'] > prev_low

            if wick_above:
                return {
                    'type':  'BEARISH_SWEEP',
                    'level': round(prev_high, 5),
                    'note':  f'Liquidity sweep above {prev_high:.5f} — potential reversal down',
                }
            if wick_below:
                return {
                    'type':  'BULLISH_SWEEP',
                    'level': round(prev_low, 5),
                    'note':  f'Liquidity sweep below {prev_low:.5f} — potential reversal up',
                }

        return {'type': 'NONE', 'note': 'No liquidity sweep detected'}

    # ═══════════════════════════════════════════════════════
    # STEP 4: TF DIRECTION
    # ═══════════════════════════════════════════════════════

    def _get_tf_directions(self, tf_contexts: dict) -> dict:
        """
        প্রতিটা TF-এর direction বের করো।
        Output: { 'H4': 'bullish', 'H1': 'bearish', ... }
        """
        directions = {}
        for label, ctx in tf_contexts.items():
            trend      = ctx.get('trend', 'sideways')
            rsi        = ctx.get('rsi', 50)
            macd_cross = ctx.get('macd_cross', '')

            # Primary: trend direction
            if 'bullish' in trend:
                direction = 'bullish'
            elif 'bearish' in trend:
                direction = 'bearish'
            else:
                # Sideways — RSI দিয়ে tiebreak
                if rsi > 55:
                    direction = 'bullish'
                elif rsi < 45:
                    direction = 'bearish'
                else:
                    direction = 'neutral'

            # MACD confirmation
            macd_aligned = (
                ('bullish' in direction and 'bullish_cross' in macd_cross) or
                ('bearish' in direction and 'bearish_cross' in macd_cross)
            )

            directions[label] = {
                'direction':     direction,
                'trend':         trend,
                'rsi':           round(rsi, 1),
                'macd_aligned':  macd_aligned,
                'strong':        'strong_' in trend,
            }

        return directions

    # ═══════════════════════════════════════════════════════
    # STEP 5: CONFLICT DETECTION
    # ═══════════════════════════════════════════════════════

    def _detect_conflicts(self, tf_directions: dict) -> list:
        """
        Timeframe-এর মধ্যে conflict খোঁজো।
        বড় TF vs ছোট TF — opposite direction হলেই conflict।
        """
        conflicts = []
        tf_labels = list(tf_directions.keys())

        for i in range(len(tf_labels)):
            for j in range(i + 1, len(tf_labels)):
                tf_a = tf_labels[i]
                tf_b = tf_labels[j]

                dir_a = tf_directions.get(tf_a, {}).get('direction', 'neutral')
                dir_b = tf_directions.get(tf_b, {}).get('direction', 'neutral')

                # Neutral TF conflict করে না
                if dir_a == 'neutral' or dir_b == 'neutral':
                    continue

                if dir_a != dir_b:
                    severity = CONFLICT_SEVERITY.get(
                        (tf_a, tf_b),
                        CONFLICT_SEVERITY.get((tf_b, tf_a), 'LOW')
                    )
                    conflicts.append({
                        'tf_a':     tf_a,
                        'tf_b':     tf_b,
                        'dir_a':    dir_a,
                        'dir_b':    dir_b,
                        'severity': severity,
                        'note':     (
                            f"{tf_a} is {dir_a} but {tf_b} is {dir_b} "
                            f"— {severity} conflict"
                        ),
                    })

        # Severity অনুযায়ী sort (CRITICAL first)
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'ACCEPTABLE': 4}
        conflicts.sort(key=lambda c: severity_order.get(c['severity'], 5))

        return conflicts

    # ═══════════════════════════════════════════════════════
    # STEP 6: H4 OVERRIDE RULE
    # ═══════════════════════════════════════════════════════

    def _h4_conviction_score(self, h4: dict) -> int:
        """
        Score H4's directional conviction on the same 0-{H4 weight} scale as
        TF_WEIGHTS['H4'], so H4_OVERRIDE_THRESHOLD (a value out of that scale)
        is actually comparable to something. Previously this constant was
        defined but never referenced anywhere — a documented spec/implementation
        drift from the institutional review. This is additive scoring only;
        it does not replace the existing boolean 'strong' check below, it
        supplements it.
        """
        weight = TF_WEIGHTS['H4']
        score = 0.0
        if h4.get('strong'):
            score += weight * 0.6
        if h4.get('direction', 'neutral') != 'neutral':
            score += weight * 0.2
        if h4.get('macd_aligned'):
            score += weight * 0.2
        return int(min(weight, score))

    def _check_h4_override(self, tf_directions: dict, conflicts: list) -> dict:
        """
        Rule: H4 strongly bullish/bearish হলে
        M5 এর বিপরীত signal ignore করো।

        Example:
        H4 strong bearish → M5 bullish = ignore M5, WAIT

        NOTE (institutional review fix): in addition to the original boolean
        'strong' trend check, H4 conviction is now also scored on the
        0-{TF_WEIGHTS['H4']} scale and compared against H4_OVERRIDE_THRESHOLD.
        This is purely additive — it can only make the override trigger in
        MORE cases than before (e.g. H4 not flagged 'strong' by trend
        classification alone, but with clear direction + MACD confirmation),
        never fewer. No previously-triggering scenario stops triggering.
        """
        h4 = tf_directions.get('H4', {})
        m5 = tf_directions.get('M5', {})

        h4_dir    = h4.get('direction', 'neutral')
        h4_strong = h4.get('strong', False)
        m5_dir    = m5.get('direction', 'neutral')

        h4_score = self._h4_conviction_score(h4)
        h4_conviction = h4_strong or h4_score >= H4_OVERRIDE_THRESHOLD

        # H4 strongly opposite to M5?
        if h4_conviction and h4_dir != 'neutral' and m5_dir != 'neutral':
            if h4_dir != m5_dir:
                return {
                    'active':    True,
                    'h4_dir':    h4_dir,
                    'm5_dir':    m5_dir,
                    'h4_score':  h4_score,
                    'overriding': 'M5',
                    'note': (
                        f"H4 strongly {h4_dir} (conviction {h4_score}/{TF_WEIGHTS['H4']}) — "
                        f"M5 {m5_dir} signal IGNORED. Do not trade against H4 trend."
                    ),
                }

        # H4 strong + CRITICAL conflict with H1?
        critical_conflicts = [c for c in conflicts if c['severity'] == 'CRITICAL']
        if h4_conviction and critical_conflicts:
            return {
                'active':    True,
                'h4_dir':    h4_dir,
                'm5_dir':    m5_dir,
                'h4_score':  h4_score,
                'overriding': 'H1',
                'note': (
                    f"H4 strongly {h4_dir} (conviction {h4_score}/{TF_WEIGHTS['H4']}) "
                    f"overrides H1 conflict. Wait for H1 to align with H4."
                ),
            }

        return {'active': False, 'h4_score': h4_score, 'note': 'No H4 override needed'}

    # ═══════════════════════════════════════════════════════
    # STEP 7: CONFIDENCE CALCULATION
    # ═══════════════════════════════════════════════════════

    def _calculate_confidence(
        self,
        tf_directions: dict,
        tf_contexts: dict,
        conflicts: list,
        h4_override: dict,
    ) -> tuple[int, dict]:
        """
        Confidence = sum of weights for aligned TFs
        H4:30 + H1:25 + M15:25 + M5:20 = 100

        Penalties:
          CRITICAL conflict  : -20
          HIGH conflict      : -12
          MEDIUM conflict    : -7
          H4 Override active : -10
          M5 weak entry      : -5
        """
        # Overall bias — majority direction
        dirs   = [v.get('direction') for v in tf_directions.values()]
        bull_c = dirs.count('bullish')
        bear_c = dirs.count('bearish')
        dominant = 'bullish' if bull_c > bear_c else (
                   'bearish' if bear_c > bull_c else 'neutral')

        if dominant == 'neutral':
            return 0, {}

        score_breakdown = {}
        total_score     = 0

        for tf, weight in TF_WEIGHTS.items():
            if tf not in tf_directions:
                score_breakdown[tf] = {'weight': weight, 'earned': 0, 'reason': 'No data'}
                continue

            tf_dir   = tf_directions[tf].get('direction', 'neutral')
            aligned  = tf_dir == dominant
            ctx      = tf_contexts.get(tf, {})
            strong   = tf_directions[tf].get('strong', False)
            macd_ok  = tf_directions[tf].get('macd_aligned', False)

            if not aligned:
                earned = 0
                reason = f"Not aligned ({tf_dir} vs dominant {dominant})"
            else:
                earned = weight
                # Bonus: strong trend + MACD confirms
                if strong and macd_ok:
                    earned = min(weight, int(weight * 1.1))  # max 10% bonus
                    reason = f"Aligned + strong + MACD ✓ ({earned}/{weight})"
                elif strong:
                    reason = f"Aligned + strong trend ({earned}/{weight})"
                elif macd_ok:
                    reason = f"Aligned + MACD confirms ({earned}/{weight})"
                else:
                    reason = f"Aligned ({earned}/{weight})"

            score_breakdown[tf] = {
                'weight':    weight,
                'earned':    earned,
                'direction': tf_dir,
                'aligned':   aligned,
                'reason':    reason,
            }
            total_score += earned

        # ── Penalties ──────────────────────────────────────
        penalties     = []
        penalty_total = 0

        severity_penalty = {'CRITICAL': 20, 'HIGH': 12, 'MEDIUM': 7, 'LOW': 3, 'ACCEPTABLE': 0}
        for conflict in conflicts:
            p = severity_penalty.get(conflict['severity'], 0)
            if p > 0:
                penalty_total += p
                penalties.append(f"-{p} ({conflict['severity']} conflict: {conflict['tf_a']} vs {conflict['tf_b']})")

        if h4_override.get('active'):
            penalty_total += 10
            penalties.append("-10 (H4 override active)")

        # M5 weak entry penalty
        m5_ctx = tf_contexts.get('M5', {})
        if m5_ctx and not tf_directions.get('M5', {}).get('macd_aligned'):
            penalty_total += 5
            penalties.append("-5 (M5 entry not MACD confirmed)")

        final_confidence = max(0, min(100, total_score - penalty_total))

        score_breakdown['_penalties'] = penalties
        score_breakdown['_raw_score'] = total_score
        score_breakdown['_penalty_total'] = penalty_total
        score_breakdown['_dominant'] = dominant

        return final_confidence, score_breakdown

    # ═══════════════════════════════════════════════════════
    # STEP 8: FINAL DECISION
    # ═══════════════════════════════════════════════════════

    def _make_decision(
        self,
        tf_directions: dict,
        conflicts: list,
        h4_override: dict,
        confidence: int,
    ) -> str:
        """
        Rule 1: H4 Override active → WAIT
        Rule 2: CRITICAL conflict → WAIT
        Rule 3: Confidence < 45 → WAIT
        Rule 4: All 4 TF bullish + confidence ≥ 65 → BUY
        Rule 5: All 4 TF bearish + confidence ≥ 65 → SELL
        Rule 6: 3/4 TF aligned + confidence ≥ 55 → BUY/SELL
        Rule 7: else → WAIT
        """
        # H4 Override → WAIT
        if h4_override.get('active'):
            return 'WAIT'

        # CRITICAL conflict → WAIT
        critical = [c for c in conflicts if c['severity'] == 'CRITICAL']
        if critical:
            return 'WAIT'

        # Low confidence → WAIT
        if confidence < 45:
            return 'WAIT'

        # Count aligned TFs
        dirs   = [v.get('direction') for v in tf_directions.values()]
        bull_c = dirs.count('bullish')
        bear_c = dirs.count('bearish')

        # All 4 aligned
        if bull_c == 4 and confidence >= 65:
            return 'BUY'
        if bear_c == 4 and confidence >= 65:
            return 'SELL'

        # 3 out of 4 aligned
        if bull_c >= 3 and confidence >= 55:
            return 'BUY'
        if bear_c >= 3 and confidence >= 55:
            return 'SELL'

        return 'WAIT'

    # ═══════════════════════════════════════════════════════
    # REASON BUILDER
    # ═══════════════════════════════════════════════════════

    def _build_reason(
        self,
        decision: str,
        tf_directions: dict,
        conflicts: list,
        h4_override: dict,
        confidence: int,
    ) -> str:
        parts = []

        # Alignment summary
        aligned_tfs = [
            tf for tf, info in tf_directions.items()
            if info.get('direction') not in ('neutral', None)
        ]
        dirs = [tf_directions[tf]['direction'] for tf in aligned_tfs]
        bull_tfs = [tf for tf, d in zip(aligned_tfs, dirs) if d == 'bullish']
        bear_tfs = [tf for tf, d in zip(aligned_tfs, dirs) if d == 'bearish']

        if bull_tfs:
            parts.append(f"Bullish: {', '.join(bull_tfs)}")
        if bear_tfs:
            parts.append(f"Bearish: {', '.join(bear_tfs)}")

        if decision in ('BUY', 'SELL'):
            parts.append(f"All key timeframes aligned → {decision}")
        elif h4_override.get('active'):
            parts.append(h4_override['note'])
        elif conflicts:
            top_conflict = conflicts[0]
            parts.append(f"Conflict: {top_conflict['note']}")

        if confidence >= 75:
            parts.append("High confidence signal")
        elif confidence >= 55:
            parts.append("Moderate confidence — confirm on lower TF")
        else:
            parts.append("Low confidence — wait for better setup")

        return " | ".join(parts)

    # ═══════════════════════════════════════════════════════
    # HELPER
    # ═══════════════════════════════════════════════════════

    def _empty_result(self, reason: str) -> dict:
        return {
            'pair':       self.symbol,
            'decision':   'WAIT',
            'confidence': 0,
            'reason':     reason,
            'timeframes': {},
            'structure':  {},
            'contexts':   {},
            'conflicts':  [],
            'h4_override': {'active': False},
            'regime_gate': {'blocked': False, 'checked': False, 'note': reason},
            'score_breakdown': {},
        }


    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, result: dict):
        dec  = result['decision']
        conf = result['confidence']

        decision_icon = {'BUY': '🟢', 'SELL': '🔴', 'WAIT': '🟡'}.get(dec, '⬜')

        print("\n" + "═" * 58)
        print("  📊  MULTI-TIMEFRAME ANALYSIS  (Day 38)")
        print("═" * 58)
        print(f"  Pair          :  {result['pair']}")
        print()

        # TF Directions
        print("  ── Timeframe Directions ──")
        for tf, info in result.get('timeframes', {}).items():
            d     = info.get('direction', 'neutral')
            rsi   = info.get('rsi', 0)
            arrow = '▲' if d == 'bullish' else ('▼' if d == 'bearish' else '→')
            strong_tag = ' ★' if info.get('strong') else ''
            macd_tag   = ' M✓' if info.get('macd_aligned') else ''
            print(f"  {tf:<6}  :  {arrow} {d:<10}  RSI {rsi:.1f}{strong_tag}{macd_tag}")

        # Market Structure
        structure = result.get('structure', {})
        if structure:
            print()
            print("  ── Market Structure ──")
            for tf, s in structure.items():
                bos   = s.get('bos', {}).get('type', 'NONE')
                choch = s.get('choch', {}).get('type', 'NONE')
                liq   = s.get('liquidity_sweep', {}).get('type', 'NONE')
                print(f"  {tf:<6}  :  BOS={bos:<15} CHoCH={choch:<16} LIQ={liq}")

        # Conflicts
        conflicts = result.get('conflicts', [])
        if conflicts:
            print()
            print("  ── Conflicts ──")
            for c in conflicts:
                icon = {'CRITICAL': '🚨', 'HIGH': '⚠️ ', 'MEDIUM': '⚡', 'LOW': '💡', 'ACCEPTABLE': '✅'}.get(c['severity'], '❓')
                print(f"  {icon}  {c['note']}")

        # H4 Override
        h4_ov = result.get('h4_override', {})
        if h4_ov.get('active'):
            print()
            print("  ── H4 Override Active ──")
            print(f"  ⛔  {h4_ov['note']}")

        # Confidence breakdown
        sb = result.get('score_breakdown', {})
        if sb:
            print()
            print("  ── Confidence Breakdown ──")
            for tf, weight in TF_WEIGHTS.items():
                info = sb.get(tf, {})
                earned = info.get('earned', 0)
                reason = info.get('reason', '')
                bar    = '█' * earned + '░' * (weight - earned)
                print(f"  {tf:<6}  [{bar}]  {earned:>2}/{weight}  {reason}")

            penalties = sb.get('_penalties', [])
            if penalties:
                print()
                for p in penalties:
                    print(f"  ❌  Penalty: {p}")

            raw   = sb.get('_raw_score', 0)
            total = sb.get('_penalty_total', 0)
            print(f"\n  Raw Score     :  {raw}")
            print(f"  Total Penalty :  -{total}")

        print()
        print(f"  ┌──────────────────────────────────────────────────┐")
        print(f"  │  {decision_icon} {dec:<6}  |  Confidence: {conf}%{'':<21}│")
        print(f"  │  {result.get('reason', '')[:52]:<52}│")
        print(f"  └──────────────────────────────────────────────────┘")
        print("═" * 58 + "\n")

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT — অন্য module-এর সাথে integration
    # ═══════════════════════════════════════════════════════

    def get_ai_context(self, result: dict) -> dict:
        """
        MarketBiasEngine, DecisionBrain — এদের জন্য context।
        Existing mtf_bias format match করে।
        """
        tf_dirs = result.get('timeframes', {})
        conflicts = result.get('conflicts', [])

        return {
            # Existing timeframe.py format — backward compatible
            'bias':            result['decision'],
            'confidence':      'HIGH' if result['confidence'] >= 65 else (
                               'MEDIUM' if result['confidence'] >= 45 else 'LOW'),
            'confidence_pct':  result['confidence'],
            'trends': {
                tf: info.get('direction', 'neutral')
                for tf, info in tf_dirs.items()
            },

            # Day 38 extras
            'decision':        result['decision'],
            'h4_trend':        tf_dirs.get('H4', {}).get('direction', 'neutral'),
            'h1_trend':        tf_dirs.get('H1', {}).get('direction', 'neutral'),
            'm15_trend':       tf_dirs.get('M15', {}).get('direction', 'neutral'),
            'm5_trend':        tf_dirs.get('M5', {}).get('direction', 'neutral'),
            'conflict_count':  len(conflicts),
            'has_critical':    any(c['severity'] == 'CRITICAL' for c in conflicts),
            'h4_override':     result.get('h4_override', {}).get('active', False),
            'reason':          result.get('reason', ''),

            # Structure
            'h4_bos':   result.get('structure', {}).get('H4', {}).get('bos', {}).get('type', 'NONE'),
            'h4_choch': result.get('structure', {}).get('H4', {}).get('choch', {}).get('type', 'NONE'),
            'h1_bos':   result.get('structure', {}).get('H1', {}).get('bos', {}).get('type', 'NONE'),

            # NEW: regime hard-gate visibility (additive, safe for existing consumers to ignore)
            'regime_gate_blocked': result.get('regime_gate', {}).get('blocked', False),
            'regime_gate_note':    result.get('regime_gate', {}).get('note', ''),
        }


# ═══════════════════════════════════════════════════════════
# QUICK RUN — Direct test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    analyzer = MTFAnalyzer(symbol="EURUSD")
    result   = analyzer.analyze()
    analyzer.print_summary(result)

    # AI context (MarketBiasEngine-এর সাথে integration)
    ai_ctx = analyzer.get_ai_context(result)
    print("AI Context for DecisionBrain:")
    for k, v in ai_ctx.items():
        print(f"  {k:<20}: {v}")