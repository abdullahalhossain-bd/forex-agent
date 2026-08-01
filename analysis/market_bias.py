# analysis/market_bias.py
# ============================================================
# Market Bias Engine — Confidence Score + Conflict Detection
# "SELL" এর বদলে "SELL 62% — but support nearby, wait"
# ============================================================

from utils.logger import get_logger
log = get_logger(__name__)


class MarketBiasEngine:
    """
    সব indicator + pattern + S/R + MTF একসাথে দেখে:
    1. Bias (bullish/bearish/neutral)
    2. Confidence score (0-100%)
    3. Conflict warnings
    4. Final recommendation
    """

    def analyze(
        self,
        ind_ctx:  dict,
        pat_ctx:  dict,
        sr_ctx:   dict,
        mtf_bias: dict = None,
        fib_ctx:  dict = None,
    ) -> dict:

        signals  = []
        warnings = []

        # ── Trend (weight: 2) ──────────────────────────────
        trend = ind_ctx.get('trend', '')
        if 'strong_bullish' in trend:
            signals.append(('bullish', 2, 'Strong bullish trend (MA alignment)'))
        elif 'bullish' in trend:
            signals.append(('bullish', 1, 'Bullish trend'))
        elif 'strong_bearish' in trend:
            signals.append(('bearish', 2, 'Strong bearish trend (MA alignment)'))
        elif 'bearish' in trend:
            signals.append(('bearish', 1, 'Bearish trend'))

        # ── RSI (weight: 1-2) ──────────────────────────────
        rsi     = ind_ctx.get('rsi', 50)
        rsi_sig = ind_ctx.get('rsi_signal', '')
        if rsi_sig == 'oversold':
            signals.append(('bullish', 2, f'RSI oversold ({rsi:.1f}) — bounce likely'))
        elif rsi_sig == 'overbought':
            signals.append(('bearish', 2, f'RSI overbought ({rsi:.1f}) — drop likely'))
        elif rsi_sig == 'bullish_zone':
            signals.append(('bullish', 1, f'RSI in bullish zone ({rsi:.1f})'))
        elif rsi_sig == 'bearish_zone':
            signals.append(('bearish', 1, f'RSI in bearish zone ({rsi:.1f})'))

        # ── MACD (weight: 1) ───────────────────────────────
        macd_cross = ind_ctx.get('macd_cross', '')
        macd_val   = ind_ctx.get('macd', 0)
        if macd_cross == 'bullish_cross':
            signals.append(('bullish', 1, 'MACD bullish crossover'))
        elif macd_cross == 'bearish_cross':
            signals.append(('bearish', 1, 'MACD bearish crossover'))

        # ── Pattern (weight: 2) ────────────────────────────
        pat_sig = pat_ctx.get('pattern_signal', '')
        pat_name = pat_ctx.get('latest_pattern', 'none')
        if 'Bullish' in pat_sig and pat_name != 'none':
            signals.append(('bullish', 2, f'Bullish pattern: {pat_name}'))
        elif 'Bearish' in pat_sig and pat_name != 'none':
            signals.append(('bearish', 2, f'Bearish pattern: {pat_name}'))

        # ── Location / S/R (weight: 1) ─────────────────────
        location = sr_ctx.get('price_location', '')
        dist_sup = sr_ctx.get('dist_to_support_pips') or 0
        dist_res = sr_ctx.get('dist_to_resistance_pips') or 0

        if location == 'near_support':
            signals.append(('bullish', 1, f'Price near support ({dist_sup} pips away)'))
        elif location == 'near_resistance':
            signals.append(('bearish', 1, f'Price near resistance ({dist_res} pips away)'))

        # ── MTF Bias (weight: 1) ───────────────────────────
        if mtf_bias:
            mtf_dir  = mtf_bias.get('bias', 'NEUTRAL')
            mtf_conf = mtf_bias.get('confidence', 'LOW')
            w = 2 if mtf_conf == 'HIGH' else 1
            if mtf_dir == 'BULLISH':
                signals.append(('bullish', w, f'MTF bias: BULLISH ({mtf_conf} confidence)'))
            elif mtf_dir == 'BEARISH':
                signals.append(('bearish', w, f'MTF bias: BEARISH ({mtf_conf} confidence)'))

        # ── Fibonacci Zone (weight: 1-4) ────────────────────
        # Day 101+ FIX: this engine never received Fibonacci data, so a
        # strong Fib golden-zone + confluence BUY/SELL signal (which
        # SignalEngine already factors in with weight up to 4) was
        # silently invisible here. That made MarketBiasEngine report
        # "Conflicts: 0" / 100% confidence in cases where SignalEngine,
        # looking at the same market state, saw a real bull/bear
        # disagreement driven by the Fib zone. Wiring fib_ctx in fixes
        # both the scoring gap and the conflict-detection gap below.
        fib_zone      = None
        fib_dir       = None
        if fib_ctx:
            fib_signal    = fib_ctx.get('signal', 'WAIT')
            fib_zone      = fib_ctx.get('zone', '')
            confluence    = fib_ctx.get('confluence_strength', 0) or 0
            is_golden     = fib_zone == 'GOLDEN_ZONE'
            weight        = 1
            if is_golden:
                weight += 1
            if confluence >= 90:
                weight += 2
            elif confluence >= 70:
                weight += 1

            if fib_signal == 'BUY':
                fib_dir = 'bullish'
                zone_label = fib_zone.replace('_', ' ').title() if fib_zone else 'Fib zone'
                signals.append(('bullish', weight, f'Fib BUY zone ({fib_zone}) — {zone_label} + Confluence (str={confluence})'))
            elif fib_signal == 'SELL':
                fib_dir = 'bearish'
                zone_label = fib_zone.replace('_', ' ').title() if fib_zone else 'Fib zone'
                signals.append(('bearish', weight, f'Fib SELL zone ({fib_zone}) — {zone_label} + Confluence (str={confluence})'))

        # ── Conflict Detection ─────────────────────────────
        bull_score = sum(w for d, w, _ in signals if d == 'bullish')
        bear_score = sum(w for d, w, _ in signals if d == 'bearish')

        # Conflict 1: bearish trend + near support
        if 'bearish' in trend and location == 'near_support':
            warnings.append(
                "⚠️  CONFLICT: Bearish trend but price near support. "
                "Avoid chasing sell — wait for support break confirmation."
            )
            bear_score = max(0, bear_score - 1)   # penalty

        # Conflict 2: bullish trend + near resistance
        if 'bullish' in trend and location == 'near_resistance':
            warnings.append(
                "⚠️  CONFLICT: Bullish trend but price near resistance. "
                "Avoid chasing buy — wait for resistance break confirmation."
            )
            bull_score = max(0, bull_score - 1)

        # Conflict 3: RSI extreme vs trend
        if rsi_sig == 'oversold' and 'bearish' in trend:
            warnings.append(
                "⚠️  CONFLICT: Bearish trend but RSI oversold. "
                "Possible short-term bounce. Trade carefully."
            )
        if rsi_sig == 'overbought' and 'bullish' in trend:
            warnings.append(
                "⚠️  CONFLICT: Bullish trend but RSI overbought. "
                "Possible pullback before continuation."
            )

        # Conflict 4: trend direction vs Fibonacci zone signal
        if fib_dir == 'bullish' and 'bearish' in trend:
            warnings.append(
                f"⚠️  CONFLICT: Bearish trend but price in Fib {fib_zone or 'retracement'} "
                "zone with a BUY signal. Possible reversal/bounce — trade carefully."
            )
        elif fib_dir == 'bearish' and 'bullish' in trend:
            warnings.append(
                f"⚠️  CONFLICT: Bullish trend but price in Fib {fib_zone or 'retracement'} "
                "zone with a SELL signal. Possible reversal/pullback — trade carefully."
            )

        # ── Final Bias ─────────────────────────────────────
        total    = bull_score + bear_score
        net      = bull_score - bear_score

        if total == 0:
            bias, confidence = 'NEUTRAL', 0
        else:
            confidence = round(max(bull_score, bear_score) / total * 100)
            if net >= 3:    bias = 'STRONG_BUY'
            elif net >= 1:  bias = 'BUY'
            elif net <= -3: bias = 'STRONG_SELL'
            elif net <= -1: bias = 'SELL'
            else:           bias = 'NEUTRAL'

        # Confidence-pipeline simplification: replace flat -15 × N penalty
        # with diminishing-returns per conflict, capped so minor conflicts
        # can no longer stack confidence to zero.
        #
        # Old:  confidence = max(0, confidence - 15 * len(warnings))
        #   1 conflict → -15, 2 → -30, 3 → -45, 4 → -60 (crushed to 0)
        #
        # New:  each successive conflict deducts less than the previous.
        #   1st conflict → -8, 2nd → -6, 3rd → -4, 4th+ → -3 each.
        #   Total cap: confidence cannot be reduced below 25 by conflicts.
        if warnings:
            deduction_per_conflict = [8, 6, 4]  # diminishing
            total_deduction = 0
            for i in range(len(warnings)):
                if i < len(deduction_per_conflict):
                    total_deduction += deduction_per_conflict[i]
                else:
                    total_deduction += 3  # floor for extra conflicts
            confidence = max(25, confidence - total_deduction)

            from utils.confidence_trace import confidence_trace
            confidence_trace.record(
                module="market_bias",
                before=min(100, confidence + total_deduction),
                after=confidence,
                reason=f"{len(warnings)} conflict(s), diminishing deduction -{total_deduction}, floored at 25",
            )

        # ── Recommendation ─────────────────────────────────
        recommendation = self._recommendation(bias, confidence, warnings)

        result = {
            'bias':           bias,
            'confidence':     confidence,
            'bull_score':     bull_score,
            'bear_score':     bear_score,
            'net_score':      net,
            'signals':        signals,
            'warnings':       warnings,
            'recommendation': recommendation,
        }

        log.info(f"Bias: {bias} | Confidence: {confidence}% | "
                 f"Conflicts: {len(warnings)}")
        return result

    def _recommendation(self, bias, confidence, warnings) -> str:
        if warnings and confidence < 60:
            return "🟡 WAIT — Conflicting signals. Wait for confirmation."
        if bias == 'STRONG_BUY'  and confidence >= 70:
            return "🟢 STRONG BUY — High confidence. Look for entry."
        # Round-13 audit fix: moderate-confidence STRONG_BUY (55-69%)
        # was falling through to "NEUTRAL — No clear edge" because only
        # the ≥70% branch existed. This was misleading: a STRONG_BUY
        # at 63% is NOT "no clear edge" — it's a moderate setup that
        # just needs confirmation. Now shows the correct label.
        if bias == 'STRONG_BUY'  and confidence >= 55:
            return "🟢 BUY BIAS — Moderate setup. Confirm on lower TF."
        if bias == 'BUY'         and confidence >= 55:
            return "🟢 BUY BIAS — Moderate setup. Confirm on lower TF."
        if bias == 'STRONG_SELL' and confidence >= 70:
            return "🔴 STRONG SELL — High confidence. Look for entry."
        # Round-13: same fix for STRONG_SELL moderate confidence.
        if bias == 'STRONG_SELL' and confidence >= 55:
            return "🔴 SELL BIAS — Moderate setup. Confirm on lower TF."
        if bias == 'SELL'        and confidence >= 55:
            return "🔴 SELL BIAS — Moderate setup. Confirm on lower TF."
        return "🟡 NEUTRAL — No clear edge. Stay out."

    def print_summary(self, result: dict):
        print("\n" + "═" * 52)
        print("  🧠  MARKET BIAS ENGINE")
        print("═" * 52)
        print(f"  Bias          :  {result['bias']}")
        print(f"  Confidence    :  {result['confidence']}%")
        print(f"  Bull Score    :  {result['bull_score']}")
        print(f"  Bear Score    :  {result['bear_score']}")
        print()

        print("  ── Signal Breakdown ──")
        for direction, weight, reason in result['signals']:
            arrow = '▲' if direction == 'bullish' else '▼'
            print(f"  {arrow} [{weight}]  {reason}")

        if result['warnings']:
            print()
            print("  ── Conflicts ──")
            for w in result['warnings']:
                print(f"  {w}")

        print()
        print(f"  ┌──────────────────────────────────────────┐")
        print(f"  │  {result['recommendation']:<42}│")
        print(f"  └──────────────────────────────────────────┘")
        print("═" * 52 + "\n")

    def get_ai_context(self, result: dict) -> dict:
        return {
            'bias':           result['bias'],
            'confidence_pct': result['confidence'],
            'recommendation': result['recommendation'],
            'has_conflict':   len(result['warnings']) > 0,
            'conflict_count': len(result['warnings']),
        }