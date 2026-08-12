# strategy/signal_engine.py — Production Signal Engine
# ============================================================
# This IS the production signal engine used by core/runtime.py.
# It generates BUY/SELL signals based on indicator scoring (Trend, RSI,
# MACD, Candlestick, S/R, MTF bias, Advanced pattern, Extended votes).
#
# Fibonacci scoring was REMOVED on 2026-08-04 (final audit) — the
# static method had been disabled since 2026-07-29 (win-rate audit
# measured it at 35.9%, below the 40% retention bar) and was retained
# only as dead code. fib_ctx is still accepted as a generate() arg and
# returned in the dict for informational display by DecisionAgent /
# print_summary, but it no longer has any scoring path attached.
# ============================================================


# ══════════════════════════════════════════════════════════════
# Production SignalEngine — generate() method
# (Fibonacci scoring block removed 2026-08-04 final audit)
# ══════════════════════════════════════════════════════════════

class SignalEngine:
    """
    Mixin — existing SignalEngine-এ যোগ করো।

    class SignalEngine(SignalEngineDay40Mixin):
        ...
    """

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
        Rule-based signal generation with HTF trend gate (2026-08-12 winrate audit).
        সব context দেখে BUY / SELL / WAIT / NO TRADE।
        """
        signals  = []
        warnings = []
        bull_score = 0
        bear_score = 0
        bull_factors = 0
        bear_factors = 0

        # ── HTF TREND GATE (2026-08-12 winrate audit) ────────
        # Hard requirement: price must be on the correct side of EMA200
        # AND EMA50 must be aligned with EMA200. This prevents the
        # counter-trend trades that caused the original BUY 14.9% vs
        # SELL 92.5% asymmetry on EURUSD H1 backtests.
        price   = ind_ctx.get('price', 0)
        ema_50  = ind_ctx.get('ema_50') or ind_ctx.get('sma_50')
        ema_200 = ind_ctx.get('ema_200') or ind_ctx.get('sma_200')
        adx_val = ind_ctx.get('adx', 0) or 0

        htf_bull = False
        htf_bear = False
        if price and ema_50 and ema_200:
            try:
                htf_bull = float(price) > float(ema_200) and float(ema_50) > float(ema_200)
                htf_bear = float(price) < float(ema_200) and float(ema_50) < float(ema_200)
            except (TypeError, ValueError):
                pass

        # ── ADX GATE (2026-08-12) ─────────────────────────────
        # Skip choppy markets — ADX < 22 means no clear trend.
        try:
            adx_val = float(adx_val)
        except (TypeError, ValueError):
            adx_val = 0.0

        if adx_val < 22 and adx_val > 0:
            return {
                'signal': 'WAIT', 'confidence': 0, 'bull_score': 0,
                'bear_score': 0, 'net_score': 0, 'signals': [],
                'warnings': [f"ADX too low ({adx_val:.0f}) — choppy market"],
                'recommendation': "🟡 WAIT — ADX below threshold, no clear trend",
                'fib_zone': None, 'fib_level': None, 'fib_in_golden': False,
                'fib_tp1': None, 'fib_tp2': None,
            }

        # ── Trend ─────────────────────────────────────────────
        trend = ind_ctx.get('trend', '')
        if 'strong_bullish' in trend:
            bull_score += 2; bull_factors += 1
            signals.append(('bullish', 2, 'Strong bullish trend'))
        elif 'bullish' in trend:
            bull_score += 1; bull_factors += 1
            signals.append(('bullish', 1, 'Bullish trend'))
        elif 'strong_bearish' in trend:
            bear_score += 2; bear_factors += 1
            signals.append(('bearish', 2, 'Strong bearish trend'))
        elif 'bearish' in trend:
            bear_score += 1; bear_factors += 1
            signals.append(('bearish', 1, 'Bearish trend'))

        # ── HTF Bias vote (2026-08-12) ────────────────────────
        if htf_bull:
            bull_score += 2; bull_factors += 1
            signals.append(('bullish', 2, 'HTF bull (price+EMA50 > EMA200)'))
        elif htf_bear:
            bear_score += 2; bear_factors += 1
            signals.append(('bearish', 2, 'HTF bear (price+EMA50 < EMA200)'))

        # ── RSI ───────────────────────────────────────────────
        rsi_sig = ind_ctx.get('rsi_signal', '')
        rsi     = ind_ctx.get('rsi', 50)
        if rsi_sig == 'oversold' and htf_bull:
            bull_score += 2; bull_factors += 1
            signals.append(('bullish', 2, f'RSI oversold in uptrend ({rsi:.1f})'))
        elif rsi_sig == 'overbought' and htf_bear:
            bear_score += 2; bear_factors += 1
            signals.append(('bearish', 2, f'RSI overbought in downtrend ({rsi:.1f})'))
        elif rsi_sig == 'bullish_zone' and (htf_bull or not htf_bear):
            bull_score += 1; bull_factors += 1
            signals.append(('bullish', 1, f'RSI bullish zone ({rsi:.1f})'))
        elif rsi_sig == 'bearish_zone' and (htf_bear or not htf_bull):
            bear_score += 1; bear_factors += 1
            signals.append(('bearish', 1, f'RSI bearish zone ({rsi:.1f})'))

        # ── MACD ──────────────────────────────────────────────
        macd_cross = ind_ctx.get('macd_cross', '')
        macd_val   = ind_ctx.get('macd', 0)
        macd_sig   = ind_ctx.get('macd_signal', 0)
        if macd_cross == 'bullish_cross' and (htf_bull or not htf_bear):
            bull_score += 2; bull_factors += 1
            signals.append(('bullish', 2, 'MACD bullish cross'))
        elif macd_cross == 'bearish_cross' and (htf_bear or not htf_bull):
            bear_score += 2; bear_factors += 1
            signals.append(('bearish', 2, 'MACD bearish cross'))
        elif macd_val and macd_sig and macd_val > macd_sig and macd_val > 0 and htf_bull:
            bull_score += 1; bull_factors += 1
            signals.append(('bullish', 1, 'MACD above signal + above zero'))
        elif macd_val and macd_sig and macd_val < macd_sig and macd_val < 0 and htf_bear:
            bear_score += 1; bear_factors += 1
            signals.append(('bearish', 1, 'MACD below signal + below zero'))

        # ── Stochastic confirmation (2026-08-12) ──────────────
        stoch_k = ind_ctx.get('stoch_k', 50)
        stoch_d = ind_ctx.get('stoch_d', 50)
        try:
            stoch_k = float(stoch_k) if stoch_k else 50.0
            stoch_d = float(stoch_d) if stoch_d else 50.0
        except (TypeError, ValueError):
            stoch_k, stoch_d = 50.0, 50.0
        if stoch_k > stoch_d and stoch_k < 35 and htf_bull:
            bull_score += 1; bull_factors += 1
            signals.append(('bullish', 1, f'Stoch bull cross in pullback ({stoch_k:.0f})'))
        elif stoch_k < stoch_d and stoch_k > 65 and htf_bear:
            bear_score += 1; bear_factors += 1
            signals.append(('bearish', 1, f'Stoch bear cross in pullback ({stoch_k:.0f})'))

        # ── ADX strength bonus (2026-08-12) ───────────────────
        if adx_val > 30:
            if bull_score > bear_score:
                bull_score += 1
                signals.append(('bullish', 1, f'ADX strong ({adx_val:.0f})'))
            elif bear_score > bull_score:
                bear_score += 1
                signals.append(('bearish', 1, f'ADX strong ({adx_val:.0f})'))

        # ── Candlestick Pattern ───────────────────────────────
        pat_sig  = pat_ctx.get('pattern_signal', '')
        pat_name = pat_ctx.get('latest_pattern', 'none')
        if 'Bullish' in pat_sig and pat_name != 'none' and (htf_bull or not htf_bear):
            bull_score += 2; bull_factors += 1
            signals.append(('bullish', 2, f'Bullish pattern: {pat_name}'))
        elif 'Bearish' in pat_sig and pat_name != 'none' and (htf_bear or not htf_bull):
            bear_score += 2; bear_factors += 1
            signals.append(('bearish', 2, f'Bearish pattern: {pat_name}'))

        # ── S/R Location — DISABLED 2026-07-30 ─────────────────
        # Win-rate audit measured sr_zones at ~34.5%, below the same 40%
        # retention bar that got Fibonacci disabled here on 2026-07-29 (see
        # below). location is still read further down for the trend-vs-S/R
        # conflict warnings (those are risk warnings, not directional votes,
        # so they're left as-is) — it just no longer moves bull_score/
        # bear_score on its own.
        # location = sr_ctx.get('price_location', '')
        # if location == 'near_support':
        #     bull_score += 1
        #     signals.append(('bullish', 1, 'Price near support'))
        # elif location == 'near_resistance':
        #     bear_score += 1
        #     signals.append(('bearish', 1, 'Price near resistance'))
        location = sr_ctx.get('price_location', '')

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

        # ── Fibonacci scoring: REMOVED 2026-08-04 (final audit) ──
        # Was disabled 2026-07-29 (win-rate 35.9% < 40% retention bar).
        # The dead _apply_fib_scoring() static method was deleted from
        # this module. fib_ctx is still accepted as an arg and returned
        # in the dict below for informational display only — it no
        # longer contributes to bull_score / bear_score in any form.

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

        # ── Fibonacci vs Trend conflict: REMOVED 2026-08-04 (final audit) ─
        # Was part of the Fibonacci scoring path that was disabled on
        # 2026-07-29 and removed on 2026-08-04. No fib_bias-driven
        # warning is emitted anymore.

        # ── Final Decision (2026-08-12 winrate audit — stricter) ────
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

            # 2026-08-12 winrate audit: raised thresholds from net>=4/6 to
            # net>=6/8 AND added minimum factor count requirement (4+ for
            # BUY/SELL, 6+ for STRONG). This eliminates single-source signals
            # (e.g. trend+2 alone) that produced 14.9% BUY winrate on EURUSD
            # and forces true multi-factor confluence. Also requires HTF
            # alignment — no BUY if htf_bear is True, no SELL if htf_bull
            # is True (hard counter-trend block).
            max_factors = max(bull_factors, bear_factors)

            # Hard counter-trend block (2026-08-12)
            if htf_bear and bull_score > bear_score and net >= 6:
                # Block BUY signals against HTF bear trend
                signal = 'WAIT'
                confidence = max(0, confidence - 30)
                warnings.append("⚠️  BUY blocked by HTF bear trend — counter-trend")
            elif htf_bull and bear_score > bull_score and net <= -6:
                # Block SELL signals against HTF bull trend
                signal = 'WAIT'
                confidence = max(0, confidence - 30)
                warnings.append("⚠️  SELL blocked by HTF bull trend — counter-trend")
            elif max_factors < 3:
                # Need at least 3 confluence factors
                signal = 'WAIT'
            else:
                if net >= 8 and max_factors >= 5:
                    signal = 'STRONG_BUY'
                elif net >= 6 and max_factors >= 4 and htf_bull:
                    signal = 'BUY'
                elif net <= -8 and max_factors >= 5:
                    signal = 'STRONG_SELL'
                elif net <= -6 and max_factors >= 4 and htf_bear:
                    signal = 'SELL'
                else:
                    signal = 'WAIT'

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
        # 2026-08-12: raised all confidence thresholds from 55 → 70 per
        # operator request ("minimum confidence 60 → 70"). This filters
        # out moderate-confidence setups that produced 33% WR on conf=60.
        if warnings and confidence < 70:
            return "🟡 WAIT — Conflicting signals. Wait for confluence."
        if signal == 'STRONG_BUY'  and confidence >= 80:
            return "🟢 STRONG BUY — High confidence. Look for entry."
        if signal == 'STRONG_BUY'  and confidence >= 70:
            return "🟢 BUY — Strong setup. Confirm entry."
        if signal == 'BUY'         and confidence >= 70:
            return "🟢 BUY — Confirmed setup. Take entry."
        if signal == 'STRONG_SELL' and confidence >= 80:
            return "🔴 STRONG SELL — High confidence. Look for entry."
        if signal == 'STRONG_SELL' and confidence >= 70:
            return "🔴 SELL — Strong setup. Confirm entry."
        if signal == 'SELL'        and confidence >= 70:
            return "🔴 SELL — Confirmed setup. Take entry."
        return "🟡 WAIT — Confidence below 70% threshold. Stay out."

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