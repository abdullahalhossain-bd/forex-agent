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

    # ═══════════════════════════════════════════════════════
    # MAIN METHOD
    # ═══════════════════════════════════════════════════════

    def analyze(self, df: pd.DataFrame, smc_ctx: dict = None) -> dict:
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

        # ── Step 5: Stop hunt detection ─────────────────────────
        stop_hunt_events = self.stop_hunt_detector.detect(df, liquidity_levels)
        best_hunt        = self.stop_hunt_detector.best_signal(stop_hunt_events)

        # ── Step 6: Liquidity target mapping ────────────────────
        target = None
        if best_hunt:
            target = self.stop_hunt_detector.map_liquidity_target(
                best_hunt['direction'], current_price, liquidity_levels
            )

        # ── Step 7: Confluence score + bias ─────────────────────
        score, bias, grade, factors = self._score_liquidity_bias(
            best_hunt, session, smc_ctx or {}
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
            'bias':              bias,
            'score':             score,
            'grade':             grade,
            'factors':           factors,
            'analysis':          self._build_explanation(best_hunt, session, target, bias),
        }

        log.info(
            f"[LiquidityEngine] Bias: {bias} | Score: {score}/100 | Grade: {grade} | "
            f"StopHunt: {best_hunt['direction'] if best_hunt else 'NONE'}"
        )
        return result

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
            levels.append({'price': h['price'], 'liquidity_type': 'BUY_SIDE', 'label': 'EQUAL_HIGH'})
        for l in eq_lows:
            levels.append({'price': l['price'], 'liquidity_type': 'SELL_SIDE', 'label': 'EQUAL_LOW'})

        if prev_levels.get('PDH'):
            levels.append({'price': prev_levels['PDH'], 'liquidity_type': 'BUY_SIDE', 'label': 'PDH'})
        if prev_levels.get('PDL'):
            levels.append({'price': prev_levels['PDL'], 'liquidity_type': 'SELL_SIDE', 'label': 'PDL'})
        if prev_levels.get('PWH'):
            levels.append({'price': prev_levels['PWH'], 'liquidity_type': 'BUY_SIDE', 'label': 'PWH'})
        if prev_levels.get('PWL'):
            levels.append({'price': prev_levels['PWL'], 'liquidity_type': 'SELL_SIDE', 'label': 'PWL'})

        if asian_range.get('valid'):
            levels.append({'price': asian_range['high'], 'liquidity_type': 'BUY_SIDE', 'label': 'ASIAN_HIGH'})
            levels.append({'price': asian_range['low'], 'liquidity_type': 'SELL_SIDE', 'label': 'ASIAN_LOW'})

        return levels

    # ═══════════════════════════════════════════════════════
    # SCORING  (doc-অনুযায়ী Liquidity Probability Score)
    # ═══════════════════════════════════════════════════════

    def _score_liquidity_bias(self, best_hunt, session, smc_ctx) -> tuple[int, str, str, dict]:
        """
        Score breakdown (total 100):
            Stop hunt confirmed (any strength) : +30
            Rejection strength STRONG           : +20  (MODERATE +10, WEAK +0)
            Level type PDH/PDL/PWH/PWL/Equal    : +15  (institutional-grade level)
            London session manipulation aligned : +20
            SMC confluence (BOS/OB/FVG agree)   : +15
        """
        factors = {
            'stop_hunt':        False,
            'rejection_strength': False,
            'institutional_level': False,
            'session_alignment': False,
            'smc_confluence':    False,
        }
        score = 0
        bias  = 'NEUTRAL'

        if not best_hunt:
            return 0, 'NEUTRAL', 'INVALID', factors

        factors['stop_hunt'] = True
        score += 30

        strength_score = {'STRONG': 20, 'MODERATE': 10, 'WEAK': 0}
        score += strength_score.get(best_hunt['rejection_strength'], 0)
        if best_hunt['rejection_strength'] in ('STRONG', 'MODERATE'):
            factors['rejection_strength'] = True

        if best_hunt['level_label'] in ('PDH', 'PDL', 'PWH', 'PWL', 'EQUAL_HIGH', 'EQUAL_LOW', 'ASIAN_HIGH', 'ASIAN_LOW'):
            factors['institutional_level'] = True
            score += 15

        hunt_dir   = 'BULLISH' if best_hunt['direction'] == 'BULLISH_REVERSAL' else 'BEARISH'
        sess_dir   = session.get('direction', 'NEUTRAL')
        if session.get('is_manipulation') and sess_dir == hunt_dir:
            factors['session_alignment'] = True
            score += 20

        smc_dir = smc_ctx.get('smc_direction', 'NEUTRAL')
        smc_sig = smc_ctx.get('smc_signal', 'WAIT')
        if smc_sig in ('BUY', 'SELL'):
            smc_dir_norm = 'BULLISH' if smc_sig == 'BUY' else 'BEARISH'
            if smc_dir_norm == hunt_dir:
                factors['smc_confluence'] = True
                score += 15

        score = min(100, score)
        bias  = hunt_dir if score >= MIN_LIQUIDITY_SCORE else 'NEUTRAL'

        grade = self._rank_grade(score, factors)
        return score, bias, grade, factors

    def _rank_grade(self, score: int, factors: dict) -> str:
        true_count = sum(1 for v in factors.values() if v)
        if score >= 85 and true_count >= 4:
            return 'A+'
        if score >= 65 and true_count >= 3:
            return 'A'
        if score >= MIN_LIQUIDITY_SCORE:
            return 'B'
        return 'INVALID'

    # ═══════════════════════════════════════════════════════
    # EXPLANATION
    # ═══════════════════════════════════════════════════════

    def _build_explanation(self, best_hunt, session, target, bias) -> str:
        if not best_hunt:
            return "No liquidity sweep / stop hunt detected — no institutional footprint found."

        parts = [best_hunt['note']]
        if best_hunt.get('confirmation'):
            parts.append(", ".join(best_hunt['confirmation']))
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
            'bias': 'NEUTRAL', 'score': 0, 'grade': 'INVALID', 'factors': {},
            'analysis': reason,
        }

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT  (MasterAnalyst / DecisionAgent handoff)
    # ═══════════════════════════════════════════════════════

    def get_ai_context(self, result: dict) -> dict:
        best   = result.get('best_stop_hunt')
        target = result.get('target')

        return {
            'liquidity_bias':        result.get('bias', 'NEUTRAL'),
            'liquidity_score':       result.get('score', 0),
            'liquidity_grade':       result.get('grade', 'INVALID'),
            'liquidity_factors':     result.get('factors', {}),
            'liquidity_stop_hunt':   best is not None,
            'liquidity_swept_level': best.get('level') if best else None,
            'liquidity_swept_type':  best.get('level_label') if best else None,
            'liquidity_direction':   best.get('direction') if best else 'NONE',
            'liquidity_target':      target.get('target_liquidity') if target else None,
            'liquidity_target_label': target.get('target_label') if target else None,
            'liquidity_session_event': result.get('session', {}).get('event', 'NONE'),
            'liquidity_analysis':    result.get('analysis', ''),
        }

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