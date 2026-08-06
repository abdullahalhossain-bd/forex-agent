# analysis/market_bias.py
# ============================================================
# Market Bias Engine — Confidence Score + Conflict Detection
# "SELL" এর বদলে "SELL 62% — but support nearby, wait"
# ============================================================
#
# CHANGELOG — institutional review fix (architecture-mismatch flag: this
# engine was RSI/MACD/MA-driven while the rest of the project moved to a
# structure-first (Order Block / FVG / Curve / DNA / Regime / AMT)
# architecture; the review's own scored table put this module at 7.4/10,
# the weakest in its batch, for exactly that reason):
#
#   [REVIEW-1] RSI/MACD/MA ("trend") are now SECONDARY signals — their
#       weights are halved from the original 1-2 scale, and they now only
#       break ties / add confirmation on top of a structural read rather
#       than driving the bias outright. This is additive-safe: an
#       analyze() call with none of the new *_ctx params behaves like the
#       old engine, just with lower indicator weights (see NOTE below on
#       why that's not a silent behavior-preserving change).
#   [REVIEW-2] Added structural signals as PRIMARY: `smc_ctx` (SMCEngine's
#       fused Structure+OrderBlock+FVG confluence — reused, not
#       reimplemented, since that fusion already exists and this project's
#       own convention is "one source of truth per concern"), `curve_ctx`
#       (curve_mtf.py's Book-P130-135 HTF curve bias + confidence),
#       `dna_ctx` (market_dna.py's regime cluster direction), `amt_ctx`
#       (auction_market_theory.py's value-area position).
#   [REVIEW-3] Weights are confidence-scaled, not flat: a structural
#       signal's vote is `max_weight * (its own confidence / 100)`, so a
#       low-confidence curve read contributes less than a high-confidence
#       one instead of the old flat weight-per-category scheme.
#   [REVIEW-4] `regime_ctx` (market_regime.py) is NOT a vote — it's a
#       confidence modifier + conflict source, same role it plays in
#       mtf_analyzer.py's regime gate: CHOPPY regime discounts confidence
#       and raises a conflict warning rather than blocking outright (this
#       engine reports a bias, it doesn't itself execute trades, so a hard
#       gate belongs upstream — see mtf_analyzer._apply_regime_gate).
#
# NOTE ON PRODUCTION STATUS (2026-08-06): agents/analysis_agent.py disabled
# this engine on 2026-07-31 after a win-rate audit measured it at <40% WR
# — with the OLD RSI/MACD/MA-primary design. That's the design this
# changelog replaces. This rebuild is NOT re-wired back into
# analysis_agent.py by the same change that produced it: re-enabling a
# kill-switch that was flipped off on real trading-outcome data is a
# risk-management call that needs a fresh win-rate audit against this new
# version first, not a code-review pass. Whoever re-enables it should
# treat that audit as a precondition, not a formality.
# ============================================================

from utils.logger import get_logger
log = get_logger(__name__)


# Max weight for each structural (primary) signal, before confidence-scaling.
# Curve and SMC get the highest ceiling — both are themselves fusions of
# multiple confluences (SMC = structure+OB+FVG; curve = HTF S&D zones), so
# a high-confidence read from either is worth more than a single-factor
# signal like AMT's value-area position.
STRUCTURAL_MAX_WEIGHT = {
    'smc':   4.0,   # SMCEngine: structure + order block + FVG confluence
    'curve': 4.0,   # curve_mtf.py: Book P130-135 HTF curve bias
    'dna':   3.0,   # market_dna.py: regime cluster direction
    'amt':   2.0,   # auction_market_theory.py: value-area position
}

# Legacy indicator weights (REVIEW-1: halved from the pre-review 1-2 scale,
# rounded to one decimal rather than truncated to 0, so a lone indicator
# read still contributes something when no structural context is supplied
# — this engine has to degrade gracefully, not go silent, for any caller
# that hasn't been updated to pass the new *_ctx params yet).
_TREND_WEIGHT_STRONG = 1.0   # was 2
_TREND_WEIGHT_WEAK   = 0.5   # was 1
_RSI_WEIGHT_EXTREME  = 1.0   # was 2 (oversold/overbought)
_RSI_WEIGHT_ZONE     = 0.5   # was 1 (bullish_zone/bearish_zone)
_MACD_WEIGHT         = 0.5   # was 1
_PATTERN_WEIGHT      = 2.0   # unchanged — not an indicator the review flagged
_SR_WEIGHT           = 1.0   # unchanged
_REGIME_CHOPPY_CONFIDENCE_PENALTY = 15


class MarketBiasEngine:
    """
    সব structural signal (SMC confluence, curve, DNA regime, AMT) + legacy
    indicator/pattern/S/R/MTF/Fib signal একসাথে দেখে:
    1. Bias (bullish/bearish/neutral)
    2. Confidence score (0-100%)
    3. Conflict warnings
    4. Final recommendation

    Structural signals (REVIEW-2) are the PRIMARY drivers when present;
    RSI/MACD/MA (REVIEW-1) are secondary confirmation/tie-break, matching
    the rest of this project's structure-first architecture (order_block,
    fvg_detector, curve_mtf, mtf_analyzer all made the same shift).
    """

    def analyze(
        self,
        ind_ctx:      dict,
        pat_ctx:      dict,
        sr_ctx:       dict,
        mtf_bias:     dict = None,
        fib_ctx:      dict = None,
        smc_ctx:      dict = None,
        curve_ctx:    dict = None,
        dna_ctx:      dict = None,
        amt_ctx:      dict = None,
        regime_ctx:   dict = None,
    ) -> dict:

        signals  = []
        warnings = []

        # ════════════════════════════════════════════════════
        # PRIMARY: STRUCTURAL SIGNALS (REVIEW-2, REVIEW-3)
        # ════════════════════════════════════════════════════

        structural_directions = {}  # for conflict checks below

        # ── SMC (Structure + Order Block + FVG confluence) ──
        if smc_ctx:
            smc_dir   = str(smc_ctx.get('smc_direction', 'NEUTRAL')).upper()
            smc_score = smc_ctx.get('smc_score', 0) or 0
            if smc_dir in ('BULLISH', 'BEARISH') and smc_score > 0:
                direction = 'bullish' if smc_dir == 'BULLISH' else 'bearish'
                weight = round(STRUCTURAL_MAX_WEIGHT['smc'] * min(100, smc_score) / 100, 2)
                bos = smc_ctx.get('smc_h4_bos', 'NONE')
                signals.append((direction, weight, f'SMC confluence: {smc_dir} (score={smc_score}, H4 BOS={bos})'))
                structural_directions['smc'] = direction

        # ── Curve (curve_mtf.py — Book P130-135) ──
        if curve_ctx:
            curve_bias = curve_ctx.get('bias')
            curve_conf = curve_ctx.get('confidence', curve_ctx.get('curve_confidence', 0)) or 0
            direction = None
            if curve_bias == 'BUY_ONLY':
                direction = 'bullish'
            elif curve_bias == 'SELL_ONLY':
                direction = 'bearish'
            # TREND_FOLLOW_OR_NO_TRADE (equilibrium) contributes no vote —
            # that's the book's own rule (P133): equilibrium has no bias.
            if direction:
                weight = round(STRUCTURAL_MAX_WEIGHT['curve'] * min(100, curve_conf) / 100, 2)
                reason = curve_ctx.get('reason', f'Curve bias: {curve_bias} ({curve_conf}% confidence)')
                signals.append((direction, weight, reason))
                structural_directions['curve'] = direction

        # ── DNA (market_dna.py — regime cluster) ──
        if dna_ctx:
            dna_dir  = str(dna_ctx.get('direction', 'neutral')).lower()
            dna_conf = dna_ctx.get('confidence', 0) or 0
            if dna_dir in ('bullish', 'bearish') and dna_conf > 0:
                weight = round(STRUCTURAL_MAX_WEIGHT['dna'] * min(100, dna_conf) / 100, 2)
                label = dna_ctx.get('cluster_label', dna_ctx.get('cluster_id', 'unlabeled'))
                signals.append((dna_dir, weight, f'DNA regime cluster {label}: {dna_dir} ({dna_conf}% confidence)'))
                structural_directions['dna'] = dna_dir

        # ── AMT (auction_market_theory.py — value area) ──
        if amt_ctx:
            amt_dir = str(amt_ctx.get('direction', amt_ctx.get('position', ''))).lower()
            if 'above' in amt_dir or amt_dir == 'bullish':
                signals.append(('bullish', STRUCTURAL_MAX_WEIGHT['amt'],
                                 f'AMT: price above value area — {amt_ctx.get("note", "seeking acceptance higher")}'))
                structural_directions['amt'] = 'bullish'
            elif 'below' in amt_dir or amt_dir == 'bearish':
                signals.append(('bearish', STRUCTURAL_MAX_WEIGHT['amt'],
                                 f'AMT: price below value area — {amt_ctx.get("note", "seeking acceptance lower")}'))
                structural_directions['amt'] = 'bearish'

        # ════════════════════════════════════════════════════
        # SECONDARY: LEGACY INDICATOR SIGNALS (REVIEW-1)
        # ════════════════════════════════════════════════════

        # ── Trend (MA alignment) ───────────────────────────
        trend = ind_ctx.get('trend', '')
        if 'strong_bullish' in trend:
            signals.append(('bullish', _TREND_WEIGHT_STRONG, 'Strong bullish trend (MA alignment)'))
        elif 'bullish' in trend:
            signals.append(('bullish', _TREND_WEIGHT_WEAK, 'Bullish trend'))
        elif 'strong_bearish' in trend:
            signals.append(('bearish', _TREND_WEIGHT_STRONG, 'Strong bearish trend (MA alignment)'))
        elif 'bearish' in trend:
            signals.append(('bearish', _TREND_WEIGHT_WEAK, 'Bearish trend'))

        # ── RSI ─────────────────────────────────────────────
        rsi     = ind_ctx.get('rsi', 50)
        rsi_sig = ind_ctx.get('rsi_signal', '')
        if rsi_sig == 'oversold':
            signals.append(('bullish', _RSI_WEIGHT_EXTREME, f'RSI oversold ({rsi:.1f}) — bounce likely'))
        elif rsi_sig == 'overbought':
            signals.append(('bearish', _RSI_WEIGHT_EXTREME, f'RSI overbought ({rsi:.1f}) — drop likely'))
        elif rsi_sig == 'bullish_zone':
            signals.append(('bullish', _RSI_WEIGHT_ZONE, f'RSI in bullish zone ({rsi:.1f})'))
        elif rsi_sig == 'bearish_zone':
            signals.append(('bearish', _RSI_WEIGHT_ZONE, f'RSI in bearish zone ({rsi:.1f})'))

        # ── MACD ────────────────────────────────────────────
        macd_cross = ind_ctx.get('macd_cross', '')
        if macd_cross == 'bullish_cross':
            signals.append(('bullish', _MACD_WEIGHT, 'MACD bullish crossover'))
        elif macd_cross == 'bearish_cross':
            signals.append(('bearish', _MACD_WEIGHT, 'MACD bearish crossover'))

        # ── Pattern (unchanged weight — not flagged by the review) ──
        pat_sig  = pat_ctx.get('pattern_signal', '')
        pat_name = pat_ctx.get('latest_pattern', 'none')
        if 'Bullish' in pat_sig and pat_name != 'none':
            signals.append(('bullish', _PATTERN_WEIGHT, f'Bullish pattern: {pat_name}'))
        elif 'Bearish' in pat_sig and pat_name != 'none':
            signals.append(('bearish', _PATTERN_WEIGHT, f'Bearish pattern: {pat_name}'))

        # ── Location / S/R (unchanged weight) ──────────────
        location = sr_ctx.get('price_location', '')
        dist_sup = sr_ctx.get('dist_to_support_pips') or 0
        dist_res = sr_ctx.get('dist_to_resistance_pips') or 0

        if location == 'near_support':
            signals.append(('bullish', _SR_WEIGHT, f'Price near support ({dist_sup} pips away)'))
        elif location == 'near_resistance':
            signals.append(('bearish', _SR_WEIGHT, f'Price near resistance ({dist_res} pips away)'))

        # ── MTF Bias (unchanged weight) ─────────────────────
        if mtf_bias:
            mtf_dir  = mtf_bias.get('bias', 'NEUTRAL')
            mtf_conf = mtf_bias.get('confidence', 'LOW')
            w = 2 if mtf_conf == 'HIGH' else 1
            if mtf_dir == 'BULLISH':
                signals.append(('bullish', w, f'MTF bias: BULLISH ({mtf_conf} confidence)'))
            elif mtf_dir == 'BEARISH':
                signals.append(('bearish', w, f'MTF bias: BEARISH ({mtf_conf} confidence)'))

        # ── Fibonacci Zone (unchanged weight) ───────────────
        fib_zone = None
        fib_dir  = None
        if fib_ctx:
            fib_signal = fib_ctx.get('signal', 'WAIT')
            fib_zone   = fib_ctx.get('zone', '')
            confluence = fib_ctx.get('confluence_strength', 0) or 0
            is_golden  = fib_zone == 'GOLDEN_ZONE'
            weight     = 1
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

        # Conflict 5 (NEW, REVIEW-2): structural signals disagreeing with
        # each other (e.g. SMC bullish but curve bearish). This is a
        # sharper warning than an indicator conflict — two structure-level
        # reads disagreeing usually means the setup isn't clean yet.
        distinct_structural = set(structural_directions.values())
        if len(distinct_structural) > 1:
            detail = ", ".join(f"{k}={v}" for k, v in structural_directions.items())
            warnings.append(
                f"⚠️  CONFLICT: Structural signals disagree ({detail}). "
                "Two structure-level reads pointing opposite ways — wait for them to align."
            )

        # Conflict 6 (NEW, REVIEW-4): CHOPPY regime — confidence modifier,
        # not a vote (see class docstring / changelog).
        regime_choppy = bool(regime_ctx and str(regime_ctx.get('regime', regime_ctx.get('market_regime', ''))).upper() == 'CHOPPY')
        if regime_choppy:
            warnings.append(
                "⚠️  CONFLICT: Market regime is CHOPPY — structural and indicator "
                "signals are less reliable in this regime. Confidence discounted."
            )

        # ── Final Bias ─────────────────────────────────────
        total = bull_score + bear_score
        net   = bull_score - bear_score

        if total == 0:
            bias, confidence = 'NEUTRAL', 0
        else:
            confidence = round(max(bull_score, bear_score) / total * 100)
            if net >= 3:    bias = 'STRONG_BUY'
            elif net >= 1:  bias = 'BUY'
            elif net <= -3: bias = 'STRONG_SELL'
            elif net <= -1: bias = 'SELL'
            else:           bias = 'NEUTRAL'

        # Confidence-pipeline: diminishing-returns per conflict, capped so
        # minor conflicts can no longer stack confidence to zero.
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

            try:
                from utils.confidence_trace import confidence_trace
                confidence_trace.record(
                    module="market_bias",
                    before=min(100, confidence + total_deduction),
                    after=confidence,
                    reason=f"{len(warnings)} conflict(s), diminishing deduction -{total_deduction}, floored at 25",
                )
            except Exception as e:
                log.debug(f"[MarketBias] confidence_trace unavailable (non-fatal): {e}")

        # REVIEW-4: additional flat discount for CHOPPY regime, applied
        # after the conflict deduction above (separate mechanism — regime
        # is a market-state fact, not a "signals disagree" conflict, so it
        # isn't subject to the diminishing-returns cap).
        if regime_choppy and total > 0:
            confidence = max(0, confidence - _REGIME_CHOPPY_CONFIDENCE_PENALTY)

        # ── Recommendation ─────────────────────────────────
        recommendation = self._recommendation(bias, confidence, warnings)

        result = {
            'bias':                  bias,
            'confidence':            confidence,
            'bull_score':            bull_score,
            'bear_score':            bear_score,
            'net_score':             net,
            'signals':               signals,
            'warnings':              warnings,
            'recommendation':        recommendation,
            'structural_directions': structural_directions,  # NEW: which structural signals fired, for auditing
        }

        log.info(f"Bias: {bias} | Confidence: {confidence}% | "
                 f"Conflicts: {len(warnings)} | Structural: {structural_directions}")
        return result

    def _recommendation(self, bias, confidence, warnings) -> str:
        if warnings and confidence < 60:
            return "🟡 WAIT — Conflicting signals. Wait for confirmation."
        if bias == 'STRONG_BUY'  and confidence >= 70:
            return "🟢 STRONG BUY — High confidence. Look for entry."
        if bias == 'STRONG_BUY'  and confidence >= 55:
            return "🟢 BUY BIAS — Moderate setup. Confirm on lower TF."
        if bias == 'BUY'         and confidence >= 55:
            return "🟢 BUY BIAS — Moderate setup. Confirm on lower TF."
        if bias == 'STRONG_SELL' and confidence >= 70:
            return "🔴 STRONG SELL — High confidence. Look for entry."
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

        if result.get('structural_directions'):
            print()
            print("  ── Structural Signals ──")
            for source, direction in result['structural_directions'].items():
                arrow = '▲' if direction == 'bullish' else '▼'
                print(f"  {arrow}  {source}: {direction}")

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
            'bias':                  result['bias'],
            'confidence_pct':        result['confidence'],
            'recommendation':        result['recommendation'],
            'has_conflict':          len(result['warnings']) > 0,
            'conflict_count':        len(result['warnings']),
            'structural_directions': result.get('structural_directions', {}),
        }