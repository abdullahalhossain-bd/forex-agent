# analysis/liquidity_engine.py  —  Day 62 | Liquidity Engine (Core)
# ============================================================
# Institutional Liquidity Intelligence — Day 62 এর মূল entry point।
#
# Combines:
#   liquidity_zones.py      → Equal High/Low, PDH/PDL, PWH/PWL, Asian range
#   session_analysis.py     → London open manipulation detection
#   stop_hunt_detector.py   → Stop hunt confirm + reversal direction + target
#
# Output: AnalysisAgent / MasterAnalyst-এ inject করার জন্য একটা
#         clean, structured liquidity_ctx dict + bias label।
#
# এই engine BOS/CHoCH/Order Block/FVG detect করে না — ওগুলো আগের
# day (44)-এর order_block.py / fvg_detector.py / mtf_analyzer.py-এর
# কাজ। এখানে শুধু liquidity-specific reasoning যুক্ত হয়েছে, এবং
# (ঐচ্ছিকভাবে) সেই module-গুলোর output confluence হিসেবে নেওয়া যায়।
# ============================================================

import pandas as pd
from analysis.liquidity_zones import LiquidityZoneMapper
# Round-18 audit fix: use the canonical class name (LondonManipulationDetector)
# instead of the deprecated SessionAnalyzer alias. The alias still works
# but emits a DeprecationWarning every cycle, and the rename's purpose
# (removing ambiguous names from the codebase) was left half-done.
from analysis.session_analysis import LondonManipulationDetector
from analysis.stop_hunt_detector import StopHuntDetector
# Concepts added this pass — internal/external scope, high/low-resistance
# (failure swing) classification, trendline liquidity, inducement — none of
# these were in the original 3-file pipeline (see README changelog).
from analysis.liquidity_structure import LiquidityStructureAnalyzer
from analysis.fvg_detector import FVGDetector
from utils.logger import get_logger

log = get_logger("liquidity_engine")

MIN_LIQUIDITY_SCORE = 55   # এর নিচে হলে liquidity bias = NEUTRAL/WAIT


class LiquidityEngine:
    """
    Day 62 — সব liquidity sub-module একসাথে চালিয়ে একটা unified
    liquidity_bias + confidence score বের করে।

    Usage:
        engine = LiquidityEngine()
        result = engine.analyze(df, smc_ctx=smc_ctx)   # smc_ctx ঐচ্ছিক confluence
        engine.print_summary(result)
        ctx = engine.get_ai_context(result)            # MasterAnalyst-এ pass করো
    """

    def __init__(self):
        self.zone_mapper       = LiquidityZoneMapper()
        # Round-18: use canonical name, not deprecated alias
        self.session_analyzer  = LondonManipulationDetector()
        self.stop_hunt_detector = StopHuntDetector()
        self.structure_analyzer = LiquidityStructureAnalyzer()
        self.fvg_detector       = FVGDetector()

    # ═══════════════════════════════════════════════════════
    # MAIN METHOD
    # ═══════════════════════════════════════════════════════

    def analyze(self, df: pd.DataFrame, smc_ctx: dict = None, symbol: str = "") -> dict:
        """
        Args:
            df      : OHLCV + 'atr' column, DatetimeIndex (Indicators.add_all() এর পরে)
            smc_ctx : (ঐচ্ছিক) Day 44 SMCEngine.get_ai_context() output — confluence boost-এর জন্য

        Returns:
            {
                'liquidity_levels': [...],     # সব known liquidity levels একসাথে
                'equal_highs': [...],
                'equal_lows':  [...],
                'previous_levels': {...},      # PDH/PDL/PWH/PWL
                'asian_range': {...},
                'session': {...},               # London manipulation
                'stop_hunt_events': [...],
                'best_stop_hunt': {...} | None,
                'target': {...} | None,
                'bias': 'BULLISH'|'BEARISH'|'NEUTRAL',
                'score': int (0-100),
                'grade': 'A+'|'A'|'B'|'INVALID',
                'analysis': str,
            }
        """
        if df is None or len(df) < 20 or 'atr' not in df.columns:
            return self._empty_result("Insufficient data or missing ATR column")

        current_price = float(df['close'].iloc[-1])

        # ── Step 1: Equal highs / lows ────────────────────────
        eq_highs = self.zone_mapper.find_equal_highs(df)
        eq_lows  = self.zone_mapper.find_equal_lows(df)

        # ── Step 2: PDH/PDL/PWH/PWL ────────────────────────────
        prev_levels = self.zone_mapper.calculate_previous_levels(df)

        # ── Step 3: Asian range + London manipulation ──────────
        asian_range = self.zone_mapper.asian_session_range(df)
        session     = self.session_analyzer.detect_london_manipulation(df, asian_range)

        # ── Step 4: Build unified liquidity level list ─────────
        liquidity_levels = self._build_liquidity_levels(eq_highs, eq_lows, prev_levels, asian_range)
        # NEW: tag each level internal/external (video 3/4 concept)
        liquidity_levels = self.structure_analyzer.classify_internal_external(liquidity_levels, df)

        # ── Step 5: Stop hunt detection ─────────────────────────
        stop_hunt_events = self.stop_hunt_detector.detect(df, liquidity_levels)
        best_hunt        = self.stop_hunt_detector.best_signal(stop_hunt_events)
        if best_hunt:
            # The real stop_hunt_detector.py doesn't classify SWEEP vs GRAB
            # (that distinction — video 4's "slow deceptive test" vs "fast
            # obvious wick" — was never in its scope). Computed here instead
            # of patching that file, using its 'break_index' field.
            best_hunt['pattern'] = self._classify_sweep_pattern(df, best_hunt)

        # ── Step 6: Liquidity target mapping ────────────────────
        target = None
        if best_hunt:
            target = self.stop_hunt_detector.map_liquidity_target(
                best_hunt['direction'], current_price, liquidity_levels, symbol=symbol,
            )

        # ── Step 7 (NEW): structure extensions ──────────────────
        resistance_map = self.structure_analyzer.classify_resistance(df)
        trendlines      = self.structure_analyzer.detect_trendlines(df)
        trendline_sweep = self.structure_analyzer.check_trendline_sweep(df, trendlines)
        inducement      = self.structure_analyzer.detect_inducement(liquidity_levels, stop_hunt_events)
        fvgs            = self.fvg_detector.detect(df)
        fvg_match       = self._fvg_confluence(fvgs, best_hunt, current_price, atr=self._atr_for(df))

        # ── Step 8: Confluence score + bias ─────────────────────
        score, bias, grade, factors = self._score_liquidity_bias(
            best_hunt, session, smc_ctx or {}, liquidity_levels,
            resistance_map, trendline_sweep, inducement, fvg_match,
        )

        result = {
            'current_price':     current_price,
            'liquidity_levels':  liquidity_levels,
            'equal_highs':       eq_highs,
            'equal_lows':        eq_lows,
            'previous_levels':   prev_levels,
            'asian_range':       asian_range,
            'session':           session,
            'stop_hunt_events':  stop_hunt_events,
            'best_stop_hunt':    best_hunt,
            'target':            target,
            'resistance_map':    resistance_map,
            'trendlines':        trendlines,
            'trendline_sweep':   trendline_sweep,
            'inducement':        inducement,
            'fvgs':              fvgs,
            'fvg_match':         fvg_match,
            'bias':              bias,
            'score':             score,
            'grade':             grade,
            'factors':           factors,
            'analysis':          self._build_explanation(best_hunt, session, target, bias, inducement, trendline_sweep, fvg_match),
        }

        log.info(
            f"[LiquidityEngine] Bias: {bias} | Score: {score}/100 | Grade: {grade} | "
            f"StopHunt: {best_hunt['direction'] if best_hunt else 'NONE'}"
        )
        return result

    def _classify_sweep_pattern(self, df: pd.DataFrame, hunt: dict) -> str:
        """
        SWEEP = price already tested this level (within LEVEL_TOUCH_ATR) at
        least once in the lookback before the candle that finally broke and
        rejected it — the "slow, deceptive" version. GRAB = the break candle
        IS the first real test — "fast, obvious". Backtest evidence (see
        FINDINGS.md) shows this distinction has real, OOS-robust predictive
        value (SWEEP notably outperforms GRAB), so it's worth computing even
        though the detector itself doesn't track it.
        """
        LEVEL_TOUCH_ATR = 0.10
        LOOKBACK = 15
        break_idx = hunt.get('break_index')
        if break_idx is None:
            return 'UNKNOWN'

        atr = self._atr_for(df)
        tol = atr * LEVEL_TOUCH_ATR
        level = hunt['level']
        kind  = hunt['liquidity_type']
        highs = df['high'].values
        lows  = df['low'].values

        start = max(0, break_idx - LOOKBACK)
        for j in range(start, break_idx):
            if kind == 'BUY_SIDE' and abs(highs[j] - level) <= tol:
                return 'SWEEP'
            if kind == 'SELL_SIDE' and abs(lows[j] - level) <= tol:
                return 'SWEEP'
        return 'GRAB'

    def _atr_for(self, df: pd.DataFrame) -> float:
        try:
            val = float(df['atr'].iloc[-1])
            return val if val and val == val else 0.0001  # NaN check via val==val
        except Exception:
            return 0.0001

    def _fvg_confluence(self, fvgs: list[dict], best_hunt: dict | None, current_price: float, atr: float) -> dict | None:
        """
        A fresh (unfilled) FVG in the SAME direction as the stop-hunt
        reversal, reasonably close to current price, is treated as
        confluence — the videos describe FVGs as a mean-reversion magnet;
        a hunt reversing toward one nearby adds independent evidence.
        """
        if not fvgs or not best_hunt:
            return None
        want_direction = 'BULLISH' if best_hunt['direction'] == 'BULLISH_REVERSAL' else 'BEARISH'
        match = self.fvg_detector.nearest_active(
            [g for g in fvgs if g['direction'] == want_direction], current_price, atr=atr,
        )
        if match and match['distance'] <= atr * 2:
            return match
        return None

    # ═══════════════════════════════════════════════════════
    # BUILD UNIFIED LEVEL LIST
    # ═══════════════════════════════════════════════════════

    def _build_liquidity_levels(self, eq_highs, eq_lows, prev_levels, asian_range) -> list[dict]:
        """
        সব liquidity source একটা common schema-তে আনো:
            {'price': float, 'liquidity_type': 'BUY_SIDE'|'SELL_SIDE', 'label': str}
        """
        levels = []

        for h in eq_highs:
            levels.append({'price': h['price'], 'liquidity_type': 'BUY_SIDE', 'label': 'EQUAL_HIGH', 'touches': h.get('touches', 1)})
        for l in eq_lows:
            levels.append({'price': l['price'], 'liquidity_type': 'SELL_SIDE', 'label': 'EQUAL_LOW', 'touches': l.get('touches', 1)})

        if prev_levels.get('PDH'):
            levels.append({'price': prev_levels['PDH'], 'liquidity_type': 'BUY_SIDE', 'label': 'PDH', 'touches': 1})
        if prev_levels.get('PDL'):
            levels.append({'price': prev_levels['PDL'], 'liquidity_type': 'SELL_SIDE', 'label': 'PDL', 'touches': 1})
        if prev_levels.get('PWH'):
            levels.append({'price': prev_levels['PWH'], 'liquidity_type': 'BUY_SIDE', 'label': 'PWH', 'touches': 1})
        if prev_levels.get('PWL'):
            levels.append({'price': prev_levels['PWL'], 'liquidity_type': 'SELL_SIDE', 'label': 'PWL', 'touches': 1})

        if asian_range.get('valid'):
            levels.append({'price': asian_range['high'], 'liquidity_type': 'BUY_SIDE', 'label': 'ASIAN_HIGH', 'touches': 1})
            levels.append({'price': asian_range['low'], 'liquidity_type': 'SELL_SIDE', 'label': 'ASIAN_LOW', 'touches': 1})

        return levels

    # ═══════════════════════════════════════════════════════
    # SCORING  (doc-অনুযায়ী Liquidity Probability Score)
    # ═══════════════════════════════════════════════════════

    def _score_liquidity_bias(self, best_hunt, session, smc_ctx, liquidity_levels,
                               resistance_map, trendline_sweep, inducement, fvg_match) -> tuple[int, str, str, dict]:
        """
        Score breakdown — RECALIBRATED from backtest evidence (see below).

        Original weights (theoretical, un-validated):
            stop_hunt +25, rejection_strength +15, institutional_level +10,
            session_alignment +15, smc_confluence +10, external_level +10,
            stacking_depth +5, high_resistance +5, inducement +5,
            trendline_confluence +5, fvg_confluence +10.

        EVIDENCE (10 real CSVs — AUDCAD/AUDCHF/AUDJPY/EURUSD across
        M15/H1/H4 — 6887 pooled walk-forward trades, chronologically
        split in half per dataset to check in-sample vs out-of-sample
        robustness, delta-R = avg R with factor minus avg R without):

            factor                  IS deltaR   OOS deltaR   verdict
            session_alignment         +0.143      +0.090     robust positive -> weight kept/nudged up
            SWEEP vs GRAB pattern      +0.019      +0.119     robust positive -> now scored (was not scored at all)
            trendline_confluence      -0.376      -0.233     robust NEGATIVE -> flipped to a penalty
            fvg_confluence            -0.123      -0.148     robust NEGATIVE -> flipped to a small penalty
            inducement                -0.102      -0.028     consistently negative, weaker OOS -> reduced to a small penalty
            external_level            +0.091      +0.011     positive but fades OOS -> weight reduced, not removed
            high_resistance           +0.057      +0.004     positive but fades OOS -> weight reduced, not removed
            rejection_strength        -0.028      +0.104     sign FLIPS between halves -> treated as noise, weight reduced
            stacking_depth            -0.104      +0.023     sign FLIPS between halves -> treated as noise, weight reduced

        IMPORTANT CAVEAT: this recalibration is fit to the same 10 datasets
        it's evidence from — even with the IS/OOS split, it is not a
        held-out validation on genuinely unseen data or other instruments.
        Treat this as "best evidence so far", re-run the OOS split
        (backtest.py trades.csv + a fresh chronological split) as new data
        comes in, and don't be surprised if some of these weights need
        revisiting. This is exactly the kind of in-sample tuning that
        overstates live performance if treated as final.
        """
        factors = {
            'stop_hunt':           False,
            'rejection_strength':  False,
            'institutional_level': False,
            'session_alignment':   False,
            'smc_confluence':      False,
            'external_level':      False,
            'stacking_depth':      False,
            'high_resistance':     False,
            'inducement':          False,       # now a PENALTY factor, kept in the dict for visibility
            'trendline_confluence': False,      # now a PENALTY factor
            'fvg_confluence':      False,       # now a PENALTY factor
            'sweep_pattern':       False,       # NEW — was tracked but never scored before
        }
        score = 0

        if not best_hunt:
            return 0, 'NEUTRAL', 'INVALID', factors

        factors['stop_hunt'] = True
        score += 25

        strength_score = {'STRONG': 10, 'MODERATE': 5, 'WEAK': 0}  # de-weighted: sign flips OOS
        score += strength_score.get(best_hunt['rejection_strength'], 0)
        if best_hunt['rejection_strength'] in ('STRONG', 'MODERATE'):
            factors['rejection_strength'] = True

        if best_hunt['level_label'] in ('PDH', 'PDL', 'PWH', 'PWL', 'EQUAL_HIGH', 'EQUAL_LOW', 'ASIAN_HIGH', 'ASIAN_LOW'):
            factors['institutional_level'] = True
            score += 10

        if best_hunt.get('pattern') == 'SWEEP':
            factors['sweep_pattern'] = True
            score += 10  # robust OOS positive — now actually scored

        hunt_dir   = 'BULLISH' if best_hunt['direction'] == 'BULLISH_REVERSAL' else 'BEARISH'
        sess_dir   = session.get('direction', 'NEUTRAL')
        if session.get('is_manipulation') and sess_dir == hunt_dir:
            factors['session_alignment'] = True
            score += 18  # robust OOS positive — nudged up from 15

        smc_dir = smc_ctx.get('smc_direction', 'NEUTRAL')
        smc_sig = smc_ctx.get('smc_signal', 'WAIT')
        if smc_sig in ('BUY', 'SELL'):
            smc_dir_norm = 'BULLISH' if smc_sig == 'BUY' else 'BEARISH'
            if smc_dir_norm == hunt_dir:
                factors['smc_confluence'] = True
                score += 10

        # ── NEW factors (recalibrated) ──
        swept_level = next((lvl for lvl in liquidity_levels if abs(lvl['price'] - best_hunt['level']) < 1e-9), None)
        if swept_level and swept_level.get('scope') == 'EXTERNAL':
            factors['external_level'] = True
            score += 5  # de-weighted: fades OOS
        if swept_level and swept_level.get('touches', 1) >= 3:
            factors['stacking_depth'] = True
            score += 2  # de-weighted: sign flips OOS, treated as weak/noisy

        side_key = 'buy_side' if swept_level and swept_level['liquidity_type'] == 'BUY_SIDE' else 'sell_side'
        if resistance_map.get(side_key) == 'HIGH_RESISTANCE':
            factors['high_resistance'] = True
            score += 3  # de-weighted: fades OOS

        if inducement and inducement.get('confirmed'):
            factors['inducement'] = True
            score -= 5  # FLIPPED to penalty: robustly negative in both IS and OOS halves

        if trendline_sweep and trendline_sweep.get('direction') == best_hunt['direction']:
            factors['trendline_confluence'] = True
            score -= 10  # FLIPPED to penalty: strongly and robustly negative (-0.38R / -0.23R)

        if fvg_match:
            factors['fvg_confluence'] = True
            score -= 5  # FLIPPED to penalty: robustly negative in both halves

        score = max(0, min(100, score))
        bias  = hunt_dir if score >= MIN_LIQUIDITY_SCORE else 'NEUTRAL'

        grade = self._rank_grade(score, factors)
        return score, bias, grade, factors

    def _rank_grade(self, score: int, factors: dict) -> str:
        # Penalty factors (inducement/trendline/fvg) should never count TOWARD
        # a higher grade even though they're in the same dict — only count
        # the factors that are genuine positive evidence.
        positive_factor_keys = [
            'stop_hunt', 'rejection_strength', 'institutional_level', 'session_alignment',
            'smc_confluence', 'external_level', 'stacking_depth', 'high_resistance', 'sweep_pattern',
        ]
        true_count = sum(1 for k in positive_factor_keys if factors.get(k))
        if score >= 70 and true_count >= 6:
            return 'A+'
        if score >= 55 and true_count >= 4:
            return 'A'
        if score >= MIN_LIQUIDITY_SCORE:
            return 'B'
        return 'INVALID'

    # ═══════════════════════════════════════════════════════
    # EXPLANATION
    # ═══════════════════════════════════════════════════════

    def _build_explanation(self, best_hunt, session, target, bias, inducement=None, trendline_sweep=None, fvg_match=None) -> str:
        if not best_hunt:
            return "No liquidity sweep / stop hunt detected — no institutional footprint found."

        parts = [best_hunt['note']]
        if best_hunt.get('confirmation'):
            parts.append(", ".join(best_hunt['confirmation']))
        if inducement and inducement.get('confirmed'):
            parts.append(inducement['note'])
        if trendline_sweep:
            parts.append(trendline_sweep['note'])
        if fvg_match:
            parts.append(f"Confluence with fresh {fvg_match['direction']} FVG at [{fvg_match['zone_bottom']}-{fvg_match['zone_top']}]")
        if session.get('is_manipulation'):
            parts.append(session['note'])
        if target:
            parts.append(
                f"Target liquidity at {target['target_liquidity']} "
                f"({target['target_label']}, {target['distance_pips']} pips away)"
            )
        return ". ".join(parts) + "."

    # ═══════════════════════════════════════════════════════
    # FALLBACK
    # ═══════════════════════════════════════════════════════

    def _empty_result(self, reason: str) -> dict:
        return {
            'current_price': None, 'liquidity_levels': [], 'equal_highs': [], 'equal_lows': [],
            'previous_levels': {}, 'asian_range': {'valid': False}, 'session': {'valid': False},
            'stop_hunt_events': [], 'best_stop_hunt': None, 'target': None,
            'resistance_map': {}, 'trendlines': [], 'trendline_sweep': None,
            'inducement': None, 'fvgs': [], 'fvg_match': None,
            'bias': 'NEUTRAL', 'score': 0, 'grade': 'INVALID', 'factors': {},
            'analysis': reason,
        }

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT  (MasterAnalyst / DecisionAgent handoff)
    # ═══════════════════════════════════════════════════════

    def get_ai_context(self, result: dict) -> dict:
        best   = result.get('best_stop_hunt')
        target = result.get('target')
        inducement = result.get('inducement')
        trendline_sweep = result.get('trendline_sweep')
        fvg_match = result.get('fvg_match')
        bias = result.get('bias', 'NEUTRAL')

        # ── Legacy (Day-61 LiquidityPoolAnalyzer) contract ──────────
        # core/entry_safety_filters.py::liquidity_sweep_filter(),
        # risk/trade_permission.py and risk/institutional_entry_framework.py
        # were all written against the OLD schema (liquidity_valid,
        # recent_sweep_kind, recent_sweep_implication, liquidity_above/
        # liquidity_below + *_touches) and never migrated when this Day-62
        # engine replaced Day-61's LiquidityPoolAnalyzer. Because none of
        # those keys existed here, every one of those consumers silently
        # read None/{} forever and fell through their own "no liquidity
        # context available" no-op path -- no exception, no log, nothing
        # to notice. Emitting the legacy keys HERE (derived from this
        # engine's own data, not recomputed elsewhere) fixes every
        # consumer in one place instead of duplicating the mapping in
        # three separate files, and stops this exact failure mode from
        # quietly reappearing if a fourth consumer gets added later.
        #
        # Mapping notes:
        #   - recent_sweep_kind / recent_sweep_implication are only
        #     populated once `bias` has cleared the engine's own score
        #     threshold (MIN_LIQUIDITY_SCORE) -- i.e. bias != NEUTRAL.
        #     A best_stop_hunt can exist with bias still NEUTRAL (score
        #     below threshold); surfacing that as an actionable sweep
        #     would make the legacy filter trip on noise the engine
        #     itself already decided not to trust.
        #   - BEARISH bias means the swept level was BUY_SIDE liquidity
        #     (stops resting above a high) -> "high" sweep, mirroring
        #     the Day-61 kind label. BULLISH -> "low" sweep.
        #   - liquidity_above/liquidity_below are the nearest UNTOUCHED
        #     pool strictly above/below current price, independent of
        #     whether a stop hunt has already happened -- this is a
        #     distinct question ("is there bait sitting directly ahead
        #     of this trade") from best_stop_hunt ("did a sweep already
        #     occur"). See _nearest_untouched_pool().
        recent_sweep_kind = None
        recent_sweep_implication = None
        if best is not None and bias in ('BULLISH', 'BEARISH'):
            recent_sweep_kind = 'high' if bias == 'BEARISH' else 'low'
            recent_sweep_implication = 'REVERSAL'

        pool_above, above_touches = self._nearest_untouched_pool(result, 'above')
        pool_below, below_touches = self._nearest_untouched_pool(result, 'below')

        return {
            # ---- Day-62 native schema ----
            'liquidity_bias':        bias,
            'liquidity_score':       result.get('score', 0),
            'liquidity_grade':       result.get('grade', 'INVALID'),
            'liquidity_factors':     result.get('factors', {}),
            'liquidity_stop_hunt':   best is not None,
            'liquidity_swept_level': best.get('level') if best else None,
            'liquidity_swept_type':  best.get('level_label') if best else None,
            'liquidity_swept_pattern': best.get('pattern') if best else None,
            'liquidity_direction':   best.get('direction') if best else 'NONE',
            'liquidity_target':      target.get('target_liquidity') if target else None,
            'liquidity_target_label': target.get('target_label') if target else None,
            'liquidity_session_event': result.get('session', {}).get('event', 'NONE'),
            'liquidity_inducement':  bool(inducement and inducement.get('confirmed')),
            'liquidity_trendline_confluence': trendline_sweep is not None,
            'liquidity_fvg_confluence': fvg_match is not None,
            'liquidity_analysis':    result.get('analysis', ''),

            # ---- Legacy (Day-61) contract, derived from the data above ----
            'liquidity_valid':          result.get('current_price') is not None,
            'recent_sweep_kind':        recent_sweep_kind,
            'recent_sweep_implication': recent_sweep_implication,
            'liquidity_above':          pool_above,
            'liquidity_above_touches':  above_touches,
            'liquidity_below':          pool_below,
            'liquidity_below_touches':  below_touches,
            'equal_highs':              [h['price'] for h in result.get('equal_highs', [])],
            'equal_lows':               [l['price'] for l in result.get('equal_lows', [])],
        }

    def _nearest_untouched_pool(self, result: dict, direction: str):
        """
        Legacy-schema helper. Nearest UNTOUCHED-by-a-hunt liquidity pool
        strictly above (direction='above') or below (direction='below')
        current price, plus its touch count -- what Day-61's
        LiquidityPoolAnalyzer used to expose as liquidity_above /
        liquidity_above_touches (and the _below equivalents).

        Returns (price | None, touches: int).
        """
        current_price = result.get('current_price')
        levels = result.get('liquidity_levels') or []
        if current_price is None or not levels:
            return None, 0

        want_type = 'BUY_SIDE' if direction == 'above' else 'SELL_SIDE'
        candidates = [
            lvl for lvl in levels
            if lvl.get('liquidity_type') == want_type
            and lvl.get('price') is not None
            and (
                (direction == 'above' and lvl['price'] > current_price)
                or (direction == 'below' and lvl['price'] < current_price)
            )
        ]
        if not candidates:
            return None, 0

        nearest = min(candidates, key=lambda lvl: abs(lvl['price'] - current_price))
        return nearest['price'], nearest.get('touches', 1)

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, result: dict) -> None:
        bar  = "═" * 58
        icon = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '🟡'}.get(result.get('bias'), '⚪')

        log.info(bar)
        log.info("  💧  LIQUIDITY ENGINE  (Day 62 — Liquidity Hunter)")
        log.info(bar)
        log.info(f"  Bias         : {icon} {result.get('bias')}")
        log.info(f"  Score        : {result.get('score')}/100")
        log.info(f"  Grade        : {result.get('grade')}")
        log.info("")

        factors = result.get('factors', {})
        for name, val in factors.items():
            mark = "✅" if val else "❌"
            log.info(f"  {mark} {name}")

        log.info("")
        best = result.get('best_stop_hunt')
        if best:
            log.info(f"  Stop Hunt    : {best['direction']} at {best['level']} ({best['level_label']})")
            log.info(f"  Strength     : {best['rejection_strength']}")

        target = result.get('target')
        if target:
            log.info(f"  Target       : {target['target_liquidity']} ({target['target_label']}, "
                      f"{target['distance_pips']} pips)")

        log.info("")
        log.info(f"  Analysis     : {result.get('analysis')}")
        log.info(bar)


# ═══════════════════════════════════════════════════════════════
# QUICK RUN — Direct test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data.fetcher import DataFetcher
    from data.indicators import Indicators

    fetcher = DataFetcher()
    ind     = Indicators()

    df = fetcher.fetch_ohlcv("EURUSD", "15m", limit=300)
    if df is not None:
        df = ind.add_all(df)

        engine = LiquidityEngine()
        result = engine.analyze(df)
        engine.print_summary(result)

        ctx = engine.get_ai_context(result)
        print("\nAI Context (for MasterAnalyst):")
        for k, v in ctx.items():
            print(f"  {k:<26}: {v}")