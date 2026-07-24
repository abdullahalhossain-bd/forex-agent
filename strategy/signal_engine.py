# strategy/signal_engine.py — Production Signal Engine
# ============================================================
# This IS the production signal engine used by core/runtime.py.
# It generates BUY/SELL signals based on indicator scoring + Fibonacci levels.
#
# Originally written as a "patch" to a pre-existing SignalEngine class,
# it has since become the standalone production module.
# ============================================================


# ── generate() method signature update ────────────────────────
#
# def generate(
#     self,
#     ind_ctx:          dict,
#     pat_ctx:          dict,
#     sr_ctx:           dict,
#     regime:           dict = None,
#     mtf_bias:         dict = None,
#     advanced_pat_ctx: dict = None,
#     fib_ctx:          dict = None,    # ⭐ Day 40 — add this
# ) -> dict:


# ── Fibonacci scoring block (paste inside generate()) ─────────
#
# Existing score variables: bull_score, bear_score, signals, warnings
# এর পরে এই block যোগ করো:

def _apply_fib_scoring(
    fib_ctx: dict,
    bull_score: int,
    bear_score: int,
    signals:    list,
    warnings:   list,
) -> tuple[int, int]:
    """
    Module-level backward-compat wrapper — delegates to the static method
    on SignalEngine.  Kept so any external code that imports
    `_apply_fib_scoring` directly still works.
    """
    return SignalEngine._apply_fib_scoring(
        fib_ctx, bull_score, bear_score, signals, warnings
    )


# ══════════════════════════════════════════════════════════════
# FULL generate() METHOD — Complete updated version
# তোমার existing generate() কে এটা দিয়ে replace করো।
# (existing logic সব রেখে শুধু fib block যোগ হয়েছে)
# ══════════════════════════════════════════════════════════════

class SignalEngine:
    """
    Mixin — existing SignalEngine-এ যোগ করো।

    class SignalEngine(SignalEngineDay40Mixin):
        ...
    """

    @staticmethod
    def _apply_fib_scoring(
        fib_ctx: dict,
        bull_score: int,
        bear_score: int,
        signals:    list,
        warnings:   list,
    ) -> tuple[int, int]:
        """
        Fibonacci context দেখে bull/bear score adjust করো।

        Scoring:
          BUY  signal  + in golden zone  → +3
          BUY  signal  + confluence high → +2
          SELL signal  + in golden zone  → +3
          SELL signal  + confluence high → +2
          Fib failure risk HIGH          → -2 (both directions)
          Fib zone conflict with trend   → warning
        """
        if not fib_ctx or not fib_ctx.get('fib_valid'):
            return bull_score, bear_score

        fib_bias      = fib_ctx.get('fib_bias', 'WAIT')
        fib_conf      = fib_ctx.get('fib_confidence', 0)
        in_golden     = fib_ctx.get('fib_in_golden', False)
        conf_strength = fib_ctx.get('fib_confluence_strength', 0)
        failure_risk  = fib_ctx.get('fib_failure_risk', 'LOW')
        fib_zone      = fib_ctx.get('fib_zone', '')
        fib_level     = fib_ctx.get('fib_level_near', '')

        # ── BUY signal from Fibonacci ──────────────────────────────
        if fib_bias == 'BUY' and fib_conf >= 55:
            weight = 2
            reason = f"Fib BUY zone ({fib_zone})"

            if in_golden:
                weight += 1
                reason += " — Golden Zone (50-61.8%)"

            if conf_strength >= 70:
                weight += 1
                reason += f" + Confluence (str={conf_strength})"

            bull_score += weight
            signals.append(('bullish', weight, reason))

        # ── SELL signal from Fibonacci ─────────────────────────────
        elif fib_bias == 'SELL' and fib_conf >= 55:
            weight = 2
            reason = f"Fib SELL zone ({fib_zone})"

            if in_golden:
                weight += 1
                reason += " — Golden Zone (50-61.8%)"

            if conf_strength >= 70:
                weight += 1
                reason += f" + Confluence (str={conf_strength})"

            bear_score += weight
            signals.append(('bearish', weight, reason))

        # ── Near key level (informational) ────────────────────────
        elif fib_bias == 'WAIT' and fib_level:
            signals.append(('neutral', 0, f"Fib: Price near {fib_level} — wait for reaction"))

        # ── Failure risk penalty ───────────────────────────────────
        if failure_risk == 'HIGH':
            bull_score = max(0, bull_score - 2)
            bear_score = max(0, bear_score - 2)
            warnings.append(
                f"⚠️  Fib failure risk HIGH — levels less reliable. "
                f"Reduce position size."
            )
        elif failure_risk == 'MEDIUM':
            warnings.append(
                f"💡 Fib failure risk MEDIUM — confirm with other signals."
            )

        return bull_score, bear_score

    def generate(
        self,
        ind_ctx:          dict,
        pat_ctx:          dict,
        sr_ctx:           dict,
        regime:           dict = None,
        mtf_bias:         dict = None,
        advanced_pat_ctx: dict = None,
        fib_ctx:          dict = None,    # ⭐ Day 40
        extended_ctx:     dict = None,    # 17-module integration pass
    ) -> dict:
        """
        Rule-based signal generation।
        সব context দেখে BUY / SELL / WAIT / NO TRADE।
        """
        signals  = []
        warnings = []
        bull_score = 0
        bear_score = 0

        # ── Trend ─────────────────────────────────────────────
        trend = ind_ctx.get('trend', '')
        if 'strong_bullish' in trend:
            bull_score += 2
            signals.append(('bullish', 2, 'Strong bullish trend'))
        elif 'bullish' in trend:
            bull_score += 1
            signals.append(('bullish', 1, 'Bullish trend'))
        elif 'strong_bearish' in trend:
            bear_score += 2
            signals.append(('bearish', 2, 'Strong bearish trend'))
        elif 'bearish' in trend:
            bear_score += 1
            signals.append(('bearish', 1, 'Bearish trend'))

        # ── RSI ───────────────────────────────────────────────
        rsi_sig = ind_ctx.get('rsi_signal', '')
        rsi     = ind_ctx.get('rsi', 50)
        if rsi_sig == 'oversold':
            bull_score += 2
            signals.append(('bullish', 2, f'RSI oversold ({rsi:.1f})'))
        elif rsi_sig == 'overbought':
            bear_score += 2
            signals.append(('bearish', 2, f'RSI overbought ({rsi:.1f})'))
        elif rsi_sig == 'bullish_zone':
            bull_score += 1
            signals.append(('bullish', 1, f'RSI bullish zone ({rsi:.1f})'))
        elif rsi_sig == 'bearish_zone':
            bear_score += 1
            signals.append(('bearish', 1, f'RSI bearish zone ({rsi:.1f})'))

        # ── MACD ──────────────────────────────────────────────
        macd_cross = ind_ctx.get('macd_cross', '')
        if macd_cross == 'bullish_cross':
            bull_score += 1
            signals.append(('bullish', 1, 'MACD bullish cross'))
        elif macd_cross == 'bearish_cross':
            bear_score += 1
            signals.append(('bearish', 1, 'MACD bearish cross'))

        # ── Candlestick Pattern ───────────────────────────────
        pat_sig  = pat_ctx.get('pattern_signal', '')
        pat_name = pat_ctx.get('latest_pattern', 'none')
        if 'Bullish' in pat_sig and pat_name != 'none':
            bull_score += 2
            signals.append(('bullish', 2, f'Bullish pattern: {pat_name}'))
        elif 'Bearish' in pat_sig and pat_name != 'none':
            bear_score += 2
            signals.append(('bearish', 2, f'Bearish pattern: {pat_name}'))

        # ── S/R Location ──────────────────────────────────────
        location = sr_ctx.get('price_location', '')
        if location == 'near_support':
            bull_score += 1
            signals.append(('bullish', 1, 'Price near support'))
        elif location == 'near_resistance':
            bear_score += 1
            signals.append(('bearish', 1, 'Price near resistance'))

        # ── MTF Bias ──────────────────────────────────────────
        if mtf_bias:
            mtf_dir  = mtf_bias.get('bias', 'NEUTRAL')
            mtf_conf = mtf_bias.get('confidence', 'LOW')
            w = 2 if mtf_conf == 'HIGH' else 1
            if mtf_dir == 'BULLISH':
                bull_score += w
                signals.append(('bullish', w, f'MTF bias BULLISH ({mtf_conf})'))
            elif mtf_dir == 'BEARISH':
                bear_score += w
                signals.append(('bearish', w, f'MTF bias BEARISH ({mtf_conf})'))

        # ── Advanced Pattern (Day 39) ─────────────────────────
        if advanced_pat_ctx and advanced_pat_ctx.get('has_pattern'):
            adv_dir  = advanced_pat_ctx.get('pattern_direction', 'NEUTRAL')
            adv_conf = advanced_pat_ctx.get('pattern_confidence', 0)
            adv_name = advanced_pat_ctx.get('advanced_pattern', '')
            if adv_dir == 'BULLISH' and adv_conf >= 60:
                w = 2 if adv_conf >= 75 else 1
                bull_score += w
                signals.append(('bullish', w, f'Advanced pattern: {adv_name} ({adv_conf}%)'))
            elif adv_dir == 'BEARISH' and adv_conf >= 60:
                w = 2 if adv_conf >= 75 else 1
                bear_score += w
                signals.append(('bearish', w, f'Advanced pattern: {adv_name} ({adv_conf}%)'))

        # ── Fibonacci (Day 40) ⭐ ─────────────────────────────
        bull_score, bear_score = self._apply_fib_scoring(
            fib_ctx, bull_score, bear_score, signals, warnings
        )

        # ── Extended modules (17-module integration pass) ──────
        # Votes from previously imported-only modules: andean_oscillator,
        # supertrend, utbot_alerts, nadaraya_watson_envelope,
        # daily_high_low, auction_market_theory, candlestick_patterns_ml,
        # breaker_block, flip_zones, curve_mtf. See
        # analysis/extended_modules_adapter.py for what's wired and why.
        if extended_ctx and extended_ctx.get('votes'):
            from analysis.extended_modules_adapter import apply_extended_votes
            bull_score, bear_score = apply_extended_votes(
                extended_ctx['votes'], bull_score, bear_score, signals
            )

        # ── Conflict Warnings ─────────────────────────────────
        if 'bearish' in trend and location == 'near_support':
            warnings.append("⚠️  Bearish trend + near support — wait for break")
            bear_score = max(0, bear_score - 1)

        if 'bullish' in trend and location == 'near_resistance':
            warnings.append("⚠️  Bullish trend + near resistance — wait for break")
            bull_score = max(0, bull_score - 1)

        if rsi_sig == 'oversold' and 'bearish' in trend:
            warnings.append("⚠️  RSI oversold in bearish trend — short-term bounce only")

        if rsi_sig == 'overbought' and 'bullish' in trend:
            warnings.append("⚠️  RSI overbought in bullish trend — pullback possible")

        # ── Fibonacci vs Trend conflict ───────────────────────
        if fib_ctx and fib_ctx.get('fib_valid'):
            fib_bias = fib_ctx.get('fib_bias', 'WAIT')
            if fib_bias == 'BUY' and 'strong_bearish' in trend:
                warnings.append("⚠️  Fib BUY zone but strong bearish trend — counter-trend risk")
            elif fib_bias == 'SELL' and 'strong_bullish' in trend:
                warnings.append("⚠️  Fib SELL zone but strong bullish trend — counter-trend risk")

        # ── Final Decision ────────────────────────────────────
        total  = bull_score + bear_score
        net    = bull_score - bear_score

        if total == 0:
            signal, confidence = 'WAIT', 0
        else:
            if total == 0:
                confidence = 0
            else:
                confidence = round(max(bull_score, bear_score) / total * 100)
            if warnings:
                confidence = max(0, confidence - 10 * len(warnings))

            # Day 96 bugfix: net>=2 let a single oversold/overbought RSI
            # reading (weight +2) trigger BUY/SELL on its own, with no
            # second confirming factor. Raised to 3 so at least two
            # agreeing signals (or one strong + one weak) are required.
            # Standalone-backtest audit fix (2026-07): an 8-day/14-pair
            # sanity-check backtest of this exact engine found it firing on
            # ~40-45% of bars at the old net>=3 threshold — far too
            # trigger-happy for live trading once spread/slippage are
            # accounted for — and that plain BUY/SELL (net==3, one strong +
            # one weak factor) was the weakest-performing signal tier
            # (+0.079R avg) versus STRONG_BUY (+0.196R). Raised by one point
            # per tier so BUY/SELL now needs two solid confirming factors
            # (e.g. trend+2 AND rsi+2, not trend+2 alone) rather than one.
            # This will reduce trade frequency — that is the intended
            # effect, not a bug; re-run the backtest after changing this if
            # you want to confirm the new frequency/expectancy trade-off.
            if net >= 6:    signal = 'STRONG_BUY'
            elif net >= 4:  signal = 'BUY'
            elif net <= -6: signal = 'STRONG_SELL'
            elif net <= -4: signal = 'SELL'
            else:           signal = 'WAIT'

        # Regime filter
        if regime:
            reg_type  = regime.get('strategy_type', '')
            reg_dir   = regime.get('market_direction', '')
            if reg_type == 'WAIT':
                signal     = 'WAIT'
                confidence = max(0, confidence - 20)
                warnings.append("⚠️  Market regime says WAIT — no strong trend")

            # Round-22 audit fix: directional conflict check.
            # Previously: reg_dir was computed but never used. A BUY signal
            # from scoring would pass even if the regime said market_direction
            # = BEARISH — a dangerous counter-trend trade with no warning.
            # selector.py handles this via _detect_conflict(), but signal_engine
            # had no equivalent check. Now: if the signal direction conflicts
            # with the regime direction, add a warning + confidence penalty.
            if reg_dir and signal in ('BUY', 'STRONG_BUY') and 'BEAR' in reg_dir.upper():
                warnings.append(f"⚠️  Signal BUY conflicts with regime direction {reg_dir} — counter-trend risk")
                confidence = max(0, confidence - 15)
            elif reg_dir and signal in ('SELL', 'STRONG_SELL') and 'BULL' in reg_dir.upper():
                warnings.append(f"⚠️  Signal SELL conflicts with regime direction {reg_dir} — counter-trend risk")
                confidence = max(0, confidence - 15)

        recommendation = self._signal_recommendation(signal, confidence, warnings)

        return {
            'signal':         signal,
            'confidence':     confidence,
            'bull_score':     bull_score,
            'bear_score':     bear_score,
            'net_score':      net,
            'signals':        signals,
            'warnings':       warnings,
            'recommendation': recommendation,
            # Fib details passthrough for DecisionAgent
            'fib_zone':       fib_ctx.get('fib_zone') if fib_ctx else None,
            'fib_level':      fib_ctx.get('fib_level_near') if fib_ctx else None,
            'fib_in_golden':  fib_ctx.get('fib_in_golden') if fib_ctx else False,
            'fib_tp1':        fib_ctx.get('fib_tp1') if fib_ctx else None,
            'fib_tp2':        fib_ctx.get('fib_tp2') if fib_ctx else None,
        }

    def _signal_recommendation(self, signal, confidence, warnings) -> str:
        if warnings and confidence < 55:
            return "🟡 WAIT — Conflicting signals. Wait for confluence."
        if signal == 'STRONG_BUY'  and confidence >= 70:
            return "🟢 STRONG BUY — High confidence. Look for entry."
        # Round-13 audit fix: moderate-confidence STRONG_BUY (55-69%)
        # was falling through to "WAIT — No clear edge" because only
        # the ≥70% branch existed. Now shows the correct moderate label.
        if signal == 'STRONG_BUY'  and confidence >= 55:
            return "🟢 BUY — Moderate setup. Confirm entry."
        if signal == 'BUY'         and confidence >= 55:
            return "🟢 BUY — Moderate setup. Confirm entry."
        if signal == 'STRONG_SELL' and confidence >= 70:
            return "🔴 STRONG SELL — High confidence. Look for entry."
        # Round-13: same fix for STRONG_SELL moderate confidence.
        if signal == 'STRONG_SELL' and confidence >= 55:
            return "🔴 SELL — Moderate setup. Confirm entry."
        if signal == 'SELL'        and confidence >= 55:
            return "🔴 SELL — Moderate setup. Confirm entry."
        return "🟡 WAIT — No clear edge. Stay out."

    def get_ai_context(self, result: dict) -> dict:
        return {
            'signal':         result['signal'],
            'confidence':     result['confidence'],
            'recommendation': result['recommendation'],
            'has_conflict':   len(result['warnings']) > 0,
            'fib_in_golden':  result.get('fib_in_golden', False),
            'fib_tp1':        result.get('fib_tp1'),
            'fib_tp2':        result.get('fib_tp2'),
        }

    def print_summary(self, result: dict):
        print("\n" + "═" * 52)
        print("  🎯  SIGNAL ENGINE  (Day 40)")
        print("═" * 52)
        print(f"  Signal        :  {result['signal']}")
        print(f"  Confidence    :  {result['confidence']}%")
        print(f"  Bull / Bear   :  {result['bull_score']} / {result['bear_score']}")
        if result.get('fib_zone'):
            golden = " 🌟" if result.get('fib_in_golden') else ""
            print(f"  Fib Zone      :  {result['fib_zone']} ({result.get('fib_level', '')}){golden}")
        if result.get('fib_tp1'):
            print(f"  Fib Targets   :  TP1={result['fib_tp1']}  TP2={result.get('fib_tp2', 'N/A')}")
        print()
        print("  ── Signals ──")
        for direction, weight, reason in result['signals']:
            arrow = '▲' if direction == 'bullish' else ('▼' if direction == 'bearish' else '→')
            print(f"  {arrow} [{weight}]  {reason}")
        if result['warnings']:
            print()
            print("  ── Warnings ──")
            for w in result['warnings']:
                print(f"  {w}")
        print()
        print(f"  ┌──────────────────────────────────────────┐")
        print(f"  │  {result['recommendation']:<42}│")
        print(f"  └──────────────────────────────────────────┘")
        print("═" * 52 + "\n")