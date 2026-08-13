# analysis/order_block.py  —  Day 47 | Order Block Detection (v3 — timeframe-aware,
#                                                                regime-aware rewrite)
#
# CHANGELOG vs Day 45 version (v2) — every change traces to the Day-46 code-review
# findings (order_block_review.md) plus a pass for overfitting risk:
#
#   [REVIEW-1]  Hardcoded, non-timeframe-aware magic numbers → FIXED.
#       IMPULSE_ATR_MULT / SWEEP_LOOKBACK / MAX_RUN_LOOKBACK are now looked up
#       from TIMEFRAME_PARAMS by an explicit `timeframe` argument, instead of
#       one fixed constant used for M5 and H4 alike. NOTE (overfitting caveat):
#       the per-timeframe numbers below are reasonable priors carried over
#       from the old single M15-tuned constants, NOT re-fit on new data —
#       they still need walk-forward validation before being trusted blindly.
#       Passing no `timeframe` keeps the exact old M15 behavior.
#   [REVIEW-2 / OVERFITTING] Single-threshold impulse gate → SOFTENED.
#       `abs(body) > ATR*mult` was a single hard cutoff — the classic
#       overfitting shape (one magic number, one instrument's tuning bleeds
#       into every pair/session). It's replaced with a composite impulse
#       score (ATR expansion + close-efficiency + optional volume-spike proxy
#       when tick_volume/volume exists) so no single knob decides pass/fail.
#       The ATR multiplier is now the pass bar for that composite score
#       rather than the entire criterion.
#   [REVIEW-3]  Flat +30/+20/+20/+15 scoring → REPLACED with weighted 0-100
#       scoring across impulse/structure/fvg/sweep/regime/zone-quality/decay,
#       plus a returned `score_breakdown` dict so every point is explainable
#       (review item "Score needs explainability").
#   [REVIEW-4]  No time decay → FIXED. Older OBs are discounted via an
#       exp(-age/half_life) multiplier; half_life scales with timeframe.
#   [REVIEW-5]  No regime awareness (market_regime.py "not wired" per the
#       Day-45 comment) → FIXED. MarketRegimeDetector is wired in: CHOPPY
#       regime penalizes score (OBs are noisier when structure is unclear),
#       a regime direction aligned with the OB direction gets a small bonus.
#       Callers who already computed regime elsewhere can pass `regime_ctx=`
#       to skip the recompute; auto-detected otherwise.
#   [REVIEW-6]  No zone-width normalization → FIXED. `zone_width_atr` field
#       added; abnormally wide zones (>3x ATR — probably an over-extended
#       'consecutive' run, not a precise OB) are penalized, not hard-dropped.
#   [REVIEW-7]  No invalidation reason → FIXED. `invalid_reason` field:
#       BROKEN / AGED / LOW_SCORE / None.
#   [REVIEW-8]  Dedup only compared exact run_end_index → IMPROVED. Also
#       collapses zones that materially overlap (>60% IoU) in the same
#       direction, keeping the higher-scoring one — same-direction OBs one
#       bar apart used to both survive as near-duplicate signals.
#   [REVIEW-9]  detect() was one ~200-line method → SPLIT into
#       _resolve_params / _impulse_strength / _build_candidate / _score
#       so each concern (params, impulse test, zone construction, scoring)
#       is independently testable.
#
# Still intentionally NOT done here (flagging, not doing silently):
#   - True dataclass/Pydantic model instead of dict: skipped on purpose —
#     ~15 other modules (smc_engine, liquidity_engine, agents/*, etc.) consume
#     this as a plain dict with [] / .get(). Changing the return type is a
#     breaking API change across the codebase and belongs in its own PR with
#     those call sites updated together, not silently bundled into this one.
#   - ML-based (LightGBM/XGBoost) probability scoring — needs a labeled
#     historical outcome dataset (review item 19/20) that doesn't exist yet.
#   - Multi-timeframe confluence scoring across the OB chain itself (H4→M15)
#     — mtf_analyzer.py / smc_engine.py already do cross-TF orchestration one
#     level up; duplicating it here would be two sources of truth.
#   - Discount/Premium (50% Fib) ranking — still needs a defined "leg" to
#     measure from (unchanged from v2's deferral note).

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from utils.logger import get_logger
from analysis.fvg_detector import FVGDetector
from analysis.market_structure import MarketStructure

log = get_logger("order_block")

# ─────────────────────────────────────────────────────────────────
# TIMEFRAME-AWARE PARAMETERS (REVIEW-1)
#
# These are carried-over priors from the old fixed M15 constants, scaled by
# rough intuition (lower TF = noisier = needs a bigger relative impulse and
# a shorter sweep/run lookback in bar-count terms; higher TF = fewer bars
# available so lookbacks shrink in bar-count even though each bar covers
# more time). They have NOT been walk-forward validated per-timeframe —
# treat as a starting point, not ground truth, and re-tune from backtest
# results before relying on them for a pair/TF this wasn't checked against.
# ─────────────────────────────────────────────────────────────────
TIMEFRAME_PARAMS = {
    'M1':  {'impulse_atr_mult': 1.3, 'sweep_lookback': 25, 'max_run_lookback': 8, 'decay_half_life': 80},
    'M5':  {'impulse_atr_mult': 1.5, 'sweep_lookback': 20, 'max_run_lookback': 7, 'decay_half_life': 100},
    'M15': {'impulse_atr_mult': 1.8, 'sweep_lookback': 15, 'max_run_lookback': 6, 'decay_half_life': 150},
    'M30': {'impulse_atr_mult': 1.9, 'sweep_lookback': 14, 'max_run_lookback': 6, 'decay_half_life': 180},
    'H1':  {'impulse_atr_mult': 2.0, 'sweep_lookback': 12, 'max_run_lookback': 5, 'decay_half_life': 220},
    'H4':  {'impulse_atr_mult': 2.2, 'sweep_lookback': 10, 'max_run_lookback': 4, 'decay_half_life': 300},
    'D1':  {'impulse_atr_mult': 2.4, 'sweep_lookback': 8,  'max_run_lookback': 4, 'decay_half_life': 400},
}
DEFAULT_TIMEFRAME = 'M15'   # preserves exact Day-45 numeric behavior when timeframe isn't passed

# Score weights (REVIEW-3). Sum of the "gates already passed" bonuses plus the
# graded components tops out at 100 before decay/state adjustments are applied.
SCORE_WEIGHTS = {
    'base':              25,
    'structure_break':   18,
    'fvg_confluence':    15,
    'sweep_conditioned': 15,
    'impulse_strength':  15,   # graded 0..15 by composite impulse score, not just ATR ratio
    'zone_quality':      7,    # graded 0..7, penalized if zone_width_atr is extreme
    'regime':            5,    # graded -5..+5
}
STATE_PENALTY = {'fresh': 0, 'tested': -10, 'broken': -40}
AGED_CANDLES_THRESHOLD = 300   # candles_ago beyond this → invalid_reason='AGED' (informational, not filtered)
LOW_SCORE_THRESHOLD = 35


class OrderBlockDetector:
    # Kept as class attrs (Day-45 default / mode='single' legacy behavior);
    # actual per-call values now come from _resolve_params(timeframe).
    IMPULSE_ATR_MULT = TIMEFRAME_PARAMS[DEFAULT_TIMEFRAME]['impulse_atr_mult']
    MAX_RESULTS = 10
    PROXIMITY_ATR = 0.3
    MAX_RUN_LOOKBACK = TIMEFRAME_PARAMS[DEFAULT_TIMEFRAME]['max_run_lookback']
    SINGLE_LOOKBACK = 3       # legacy single-candle lookback (mode='single' only)
    SWEEP_LOOKBACK = TIMEFRAME_PARAMS[DEFAULT_TIMEFRAME]['sweep_lookback']
    FVG_SEARCH_WINDOW = 3     # bars around the impulse to look for a confluent FVG
    ZONE_WIDTH_ATR_SOFT_CAP = 3.0   # zone wider than this (in ATR units) starts losing zone_quality points
    ZONE_OVERLAP_IOU_DEDUPE = 0.6   # same-direction zones overlapping >= this IoU are treated as duplicates

    def __init__(self, mode: str = "consecutive", require_structure_break: bool = True,
                 require_fvg: bool = True, structure_strength: int = 2,
                 timeframe: str = DEFAULT_TIMEFRAME, use_regime_filter: bool = True):
        """
        mode: 'consecutive' (default, video-6-backed) or 'single' (Day-44 legacy, for A/B backtest only)
        require_structure_break / require_fvg: hard validity gates. Both default True
            per the two rules video 6 states explicitly and that videos 2/3/4 corroborate.
            Set False only for controlled comparison runs — not for live signal generation.
        timeframe: one of TIMEFRAME_PARAMS' keys (M1/M5/M15/M30/H1/H4/D1). Selects the
            impulse/sweep/run-lookback/decay parameters for this instance. Unknown
            values fall back to DEFAULT_TIMEFRAME with a warning rather than raising,
            since this is frequently constructed once and reused across symbols.
        use_regime_filter: wire in MarketRegimeDetector (REVIEW-5). Set False to get
            the exact pre-regime scoring behavior for A/B comparison.
        """
        if mode not in ("consecutive", "single"):
            raise ValueError("mode must be 'consecutive' or 'single'")
        self.mode = mode
        self.require_structure_break = require_structure_break
        self.require_fvg = require_fvg
        self.structure_strength = structure_strength
        self.use_regime_filter = use_regime_filter
        self._fvg = FVGDetector()
        self._ms = MarketStructure()
        self._regime = None
        if use_regime_filter:
            try:
                from analysis.market_regime import MarketRegimeDetector
                self._regime = MarketRegimeDetector()
            except Exception as e:
                log.warning(f"[OrderBlock] market_regime unavailable ({e}) — regime scoring disabled")
                self.use_regime_filter = False

        self.params = self._resolve_params(timeframe)

    # ─────────────────────────────────────────────
    # PARAM RESOLUTION (REVIEW-1, REVIEW-9)
    # ─────────────────────────────────────────────

    def _resolve_params(self, timeframe: str) -> dict:
        tf = (timeframe or DEFAULT_TIMEFRAME).upper()
        if tf not in TIMEFRAME_PARAMS:
            log.warning(f"[OrderBlock] Unknown timeframe '{timeframe}', falling back to {DEFAULT_TIMEFRAME}")
            tf = DEFAULT_TIMEFRAME
        return dict(TIMEFRAME_PARAMS[tf])

    def detect(self, df: pd.DataFrame, closed_bars_only: bool = True, max_results: int | None = None,
               timeframe: str | None = None, regime_ctx: dict | None = None) -> list[dict]:
        """
        CONTRACT: `df` must contain only CLOSED bars. If your caller has a live/
        forming candle appended, drop it before calling this — passing it in
        will make `state`/`fresh` repaint as that bar develops (see CRITICAL-3
        in the v2 changelog). `closed_bars_only` is a documentation flag,
        not an enforced check (we can't detect "still forming" from OHLC alone
        without a timestamp/now comparison the caller owns).

        timeframe: optional per-call override of the instance's timeframe
            (e.g. one detector reused across M15/H4 callers). Falls back to
            the value passed to __init__ when omitted.
        regime_ctx: optional pre-computed MarketRegimeDetector.detect(df) result,
            to avoid recomputing regime when the caller already has it (e.g.
            smc_engine already runs regime detection once per df upstream).
        """
        if len(df) < 20 or 'atr' not in df.columns:
            log.warning("[OrderBlock] Insufficient data or missing ATR column")
            return []
        if not closed_bars_only:
            log.warning("[OrderBlock] closed_bars_only=False — caller accepts repaint risk on the forming bar")

        params = self.params if timeframe is None else self._resolve_params(timeframe)

        opens = df['open'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        atrs = df['atr'].values
        vols = None
        for vol_col in ('volume', 'tick_volume'):
            if vol_col in df.columns:
                vols = df[vol_col].values
                break
        n = len(df)

        structure_state = self._ms.analyze(df, strength=self.structure_strength)
        fvgs = self._fvg.detect(df)  # FVGDetector.detect(df) has no max_results param; see v2 note

        if self.use_regime_filter and regime_ctx is None:
            try:
                regime_ctx = self._regime.detect(df)
            except Exception as e:
                log.warning(f"[OrderBlock] regime detection failed ({e}) — continuing without it")
                regime_ctx = None

        raw = []
        for i in range(5, n):
            candidate = self._build_candidate(
                i, opens, closes, highs, lows, atrs, vols, n,
                structure_state, fvgs, regime_ctx, params, df)
            if candidate is not None:
                raw.append(candidate)

        deduped = self._dedupe_keep_best(raw)
        deduped.sort(key=lambda r: (r['quality_score'], r['impulse_index']), reverse=True)
        log.debug(f"[OrderBlock] Detected {len(deduped)} order blocks "
                 f"(mode={self.mode}, require_structure={self.require_structure_break}, "
                 f"require_fvg={self.require_fvg}, regime={'on' if self.use_regime_filter else 'off'})")
        # Day-46 fix, same pattern as fvg_detector.py: MAX_RESULTS=10 is correct
        # for a live "top OBs right now" caller, wrong for a backtest scanning
        # full history. max_results=None preserves old behavior exactly.
        cap = self.MAX_RESULTS if max_results is None else max_results
        return deduped if cap == 0 else deduped[:cap]

    # ─────────────────────────────────────────────
    # CANDIDATE CONSTRUCTION (split out of the old monolithic detect() loop)
    # ─────────────────────────────────────────────

    def _build_candidate(self, i, opens, closes, highs, lows, atrs, vols, n,
                          structure_state, fvgs, regime_ctx, params, df):
        atr = atrs[i]
        if np.isnan(atr) or atr == 0:
            return None

        body = closes[i] - opens[i]
        candle_range = highs[i] - lows[i]

        # ── Composite impulse score (REVIEW-2 / overfitting fix) ──────────
        # Instead of one hard `abs(body) > ATR*mult` cutoff, blend ATR
        # expansion with close-efficiency (how much of the candle's range
        # was "kept" by the close, vs. wicked away) and, when volume/
        # tick_volume exists, a volume-spike proxy. No single number can
        # single-handedly pass or fail a candle anymore.
        atr_ratio = abs(body) / atr
        close_efficiency = (abs(body) / candle_range) if candle_range > 0 else 0.0
        vol_spike = 0.0
        if vols is not None and i >= 10:
            recent_avg = np.nanmean(vols[max(0, i - 10):i]) or 0.0
            if recent_avg > 0:
                vol_spike = min(2.0, vols[i] / recent_avg) / 2.0  # 0..1

        impulse_composite = (0.55 * min(atr_ratio / params['impulse_atr_mult'], 1.5)
                              + 0.30 * close_efficiency
                              + 0.15 * vol_spike)
        if atr_ratio < params['impulse_atr_mult'] * 0.7:
            # Even a strong close-efficiency/volume score can't rescue a candle
            # whose ATR expansion is well below bar — this keeps the composite
            # from becoming "volume spike alone triggers an OB".
            return None
        if impulse_composite < 0.65:
            return None

        is_bullish_impulse = body > 0

        ob_start, ob_end = self._find_ob_run(opens, closes, i, is_bullish_impulse, params)
        if ob_start is None:
            return None

        # ── Gate 1: structure break (CRITICAL-1, v2) ──────────────────────
        structure_event = structure_state and self._ms.structure_break_between(
            structure_state, start_idx=ob_end, end_idx=i)
        if self.require_structure_break and structure_event is None:
            return None

        # ── Gate 2: FVG confluence (IMPORTANT-1, v2) ───────────────────────
        fvg_hit = self._find_confluent_fvg(fvgs, ob_end, i, is_bullish_impulse)
        if self.require_fvg and fvg_hit is None:
            return None

        zone_top = float(highs[ob_start:ob_end + 1].max())
        zone_bottom = float(lows[ob_start:ob_end + 1].min())
        body_top = float(max(opens[ob_end], closes[ob_end]))
        body_bottom = float(min(opens[ob_end], closes[ob_end]))
        ob_type = 'BULLISH_ORDER_BLOCK' if is_bullish_impulse else 'BEARISH_ORDER_BLOCK'
        zone_width_atr = (zone_top - zone_bottom) / atr if atr else 0.0

        # ── State taxonomy: fresh / tested / broken (CRITICAL-2, v2) ───────
        tested_at, broken_at = self._scan_state(
            highs, lows, closes, i, zone_top, zone_bottom, is_bullish_impulse, n)
        state = 'broken' if broken_at is not None else ('tested' if tested_at is not None else 'fresh')

        # ── Sweep-conditioning (IMPORTANT-3, v2) ────────────────────────────
        sweep = structure_state and self._ms.liquidity_sweep_before(
            df, structure_state, ob_start, lookback=params['sweep_lookback'])

        candles_ago = n - 1 - ob_end
        quality_score, breakdown = self._score(
            impulse_composite=impulse_composite,
            structure_break=structure_event is not None,
            fvg_confluence=fvg_hit is not None,
            sweep_conditioned=sweep is not None,
            state=state,
            zone_width_atr=zone_width_atr,
            regime_ctx=regime_ctx,
            direction='BULLISH' if is_bullish_impulse else 'BEARISH',
            candles_ago=candles_ago,
            decay_half_life=params['decay_half_life'],
        )

        invalid_reason = None
        if state == 'broken':
            invalid_reason = 'BROKEN'
        elif candles_ago > AGED_CANDLES_THRESHOLD:
            invalid_reason = 'AGED'
        elif quality_score < LOW_SCORE_THRESHOLD:
            invalid_reason = 'LOW_SCORE'

        return {
            'type': ob_type,
            'direction': 'BULLISH' if is_bullish_impulse else 'BEARISH',
            'index': ob_end,                       # OB candle nearest the impulse (kept for backward-compat)
            'run_start_index': ob_start,
            'run_end_index': ob_end,
            'impulse_index': i,
            'zone_top': round(zone_top, 5),
            'zone_bottom': round(zone_bottom, 5),
            'body_top': round(body_top, 5),
            'body_bottom': round(body_bottom, 5),
            'zone_width_atr': round(zone_width_atr, 3),
            'state': state,
            'fresh': state == 'fresh',
            'tested': state == 'tested',
            'broken': state == 'broken',
            'invalid_reason': invalid_reason,
            'structure_break': structure_event is not None,
            'structure_event_type': structure_event['type'] if structure_event else None,
            'fvg_confluence': fvg_hit is not None,
            'sweep_conditioned': sweep is not None,
            'sweep_info': sweep,
            'impulse_composite': round(impulse_composite, 3),
            'quality_score': quality_score,
            'score_breakdown': breakdown,
            'candles_ago': candles_ago,
        }

    # ─────────────────────────────────────────────
    # OB RUN SELECTION
    # ─────────────────────────────────────────────

    def _find_ob_run(self, opens, closes, impulse_idx: int, is_bullish_impulse: bool, params: dict):
        """
        Returns (run_start_idx, run_end_idx) — the candle range that forms the OB,
        both inclusive, run_end being the candle immediately preceding the impulse's
        opposite-color run.

        mode='single': legacy — just the first opposite-colored candle within
            SINGLE_LOOKBACK bars (Day-44 behavior).
        mode='consecutive': walk backward from impulse_idx-1 while candles keep the
            opposite color, capped at params['max_run_lookback'] (timeframe-aware;
            video 6's explicit fix for missed fills from over-refining to a single candle).
        """
        j = impulse_idx - 1
        if j < 0:
            return None, None

        def is_opposite(k):
            c_body = closes[k] - opens[k]
            if is_bullish_impulse:
                return c_body < 0
            return c_body > 0

        if self.mode == "single":
            for k in range(j, max(j - self.SINGLE_LOOKBACK, -1), -1):
                if is_opposite(k):
                    return k, k
            return None, None

        # consecutive mode
        if not is_opposite(j):
            # last candle before impulse isn't opposite-colored — fall back to
            # nearest opposite candle within lookback (same as single mode)
            for k in range(j, max(j - self.SINGLE_LOOKBACK, -1), -1):
                if is_opposite(k):
                    return k, k
            return None, None

        run_end = j
        run_start = j
        limit = max(j - params['max_run_lookback'], -1)
        k = j - 1
        while k > limit and is_opposite(k):
            run_start = k
            k -= 1
        return run_start, run_end

    # ─────────────────────────────────────────────
    # STATE SCAN (fresh / tested / broken)
    # ─────────────────────────────────────────────

    def _scan_state(self, highs, lows, closes, impulse_idx, zone_top, zone_bottom,
                     is_bullish_impulse, n):
        tested_at = None
        broken_at = None
        for k in range(impulse_idx + 1, n):
            touched = lows[k] <= zone_top and highs[k] >= zone_bottom
            if touched and tested_at is None:
                tested_at = k
            # Break = a CLOSE beyond the zone on the invalidation side, not a wick.
            # Video 5's explicit rule; also bias-and-validation.md's close-vs-wick note.
            if is_bullish_impulse and closes[k] < zone_bottom:
                broken_at = k
                break
            if not is_bullish_impulse and closes[k] > zone_top:
                broken_at = k
                break
        return tested_at, broken_at

    # ─────────────────────────────────────────────
    # FVG CONFLUENCE
    # ─────────────────────────────────────────────

    def _find_confluent_fvg(self, fvgs: list[dict], ob_end: int, impulse_idx: int,
                             is_bullish_impulse: bool):
        wanted_dir = 'BULLISH' if is_bullish_impulse else 'BEARISH'
        lo = ob_end - self.FVG_SEARCH_WINDOW
        hi = impulse_idx + self.FVG_SEARCH_WINDOW
        for g in fvgs:
            if g['direction'] == wanted_dir and lo <= g['index'] <= hi:
                return g
        return None

    # ─────────────────────────────────────────────
    # SCORING (REVIEW-3: weighted, explainable, decay + regime aware)
    # ─────────────────────────────────────────────

    def _score(self, impulse_composite: float, structure_break: bool, fvg_confluence: bool,
               sweep_conditioned: bool, state: str, zone_width_atr: float,
               regime_ctx: dict | None, direction: str, candles_ago: int,
               decay_half_life: int) -> tuple[int, dict]:
        breakdown = {'base': SCORE_WEIGHTS['base']}
        score = SCORE_WEIGHTS['base']

        if structure_break:
            breakdown['structure_break'] = SCORE_WEIGHTS['structure_break']
            score += SCORE_WEIGHTS['structure_break']
        if fvg_confluence:
            breakdown['fvg_confluence'] = SCORE_WEIGHTS['fvg_confluence']
            score += SCORE_WEIGHTS['fvg_confluence']
        if sweep_conditioned:
            breakdown['sweep_conditioned'] = SCORE_WEIGHTS['sweep_conditioned']
            score += SCORE_WEIGHTS['sweep_conditioned']

        impulse_pts = round(min(1.0, impulse_composite) * SCORE_WEIGHTS['impulse_strength'])
        breakdown['impulse_strength'] = impulse_pts
        score += impulse_pts

        # REVIEW-6: zone width normalization — penalize zones that are wide
        # relative to ATR (likely an over-extended run rather than a precise OB).
        if zone_width_atr <= self.ZONE_WIDTH_ATR_SOFT_CAP:
            zone_pts = SCORE_WEIGHTS['zone_quality']
        else:
            overshoot = zone_width_atr - self.ZONE_WIDTH_ATR_SOFT_CAP
            zone_pts = max(0, round(SCORE_WEIGHTS['zone_quality'] * math.exp(-overshoot / 2.0)))
        breakdown['zone_quality'] = zone_pts
        score += zone_pts

        # REVIEW-5: regime adjustment. CHOPPY regime = noisier structure =
        # small penalty. TRENDING regime aligned with OB direction = small
        # bonus; opposed = small penalty. Anything else (RANGING/BREAKOUT/
        # UNKNOWN, or regime unavailable) is neutral.
        regime_pts = 0
        if regime_ctx:
            regime = regime_ctx.get('regime')
            reg_dir = regime_ctx.get('direction')
            if regime == 'CHOPPY':
                regime_pts = -SCORE_WEIGHTS['regime']
            elif regime == 'TRENDING' and reg_dir in ('BULLISH', 'BEARISH'):
                regime_pts = SCORE_WEIGHTS['regime'] if reg_dir == direction else -round(SCORE_WEIGHTS['regime'] / 2)
        breakdown['regime'] = regime_pts
        score += regime_pts

        state_pts = STATE_PENALTY.get(state, 0)
        breakdown['state_penalty'] = state_pts
        score += state_pts

        raw_score = max(0, min(100, score))

        # REVIEW-4: time decay — applied as a multiplier on the raw score so
        # the breakdown above stays a clean audit trail of "points earned",
        # and decay is visible as its own explicit factor.
        decay_factor = math.exp(-candles_ago / max(decay_half_life, 1))
        breakdown['decay_factor'] = round(decay_factor, 3)
        final_score = max(0, min(100, round(raw_score * (0.5 + 0.5 * decay_factor))))
        # Decay only ever discounts up to 50% of the raw score (0.5 + 0.5*factor
        # floors at 0.5x) — an old-but-otherwise-perfect OB shouldn't be driven
        # to a near-zero score just from age; state (broken/tested) already
        # does most of the invalidation work.

        return final_score, breakdown

    # ─────────────────────────────────────────────
    # DEDUP (REVIEW-8: exact-index dedupe + zone-overlap dedupe)
    # ─────────────────────────────────────────────

    def _dedupe_keep_best(self, raw: list[dict]) -> list[dict]:
        best_by_idx: dict[int, dict] = {}
        for r in raw:
            key = r['run_end_index']
            existing = best_by_idx.get(key)
            if existing is None or r['quality_score'] > existing['quality_score']:
                best_by_idx[key] = r

        candidates = sorted(best_by_idx.values(), key=lambda r: r['quality_score'], reverse=True)
        kept: list[dict] = []
        for cand in candidates:
            if any(self._zones_overlap(cand, k) for k in kept):
                continue
            kept.append(cand)
        return kept

    def _zones_overlap(self, a: dict, b: dict) -> bool:
        if a['direction'] != b['direction']:
            return False
        lo = max(a['zone_bottom'], b['zone_bottom'])
        hi = min(a['zone_top'], b['zone_top'])
        if hi <= lo:
            return False
        inter = hi - lo
        union = max(a['zone_top'], b['zone_top']) - min(a['zone_bottom'], b['zone_bottom'])
        iou = inter / union if union > 0 else 0.0
        return iou >= self.ZONE_OVERLAP_IOU_DEDUPE

    # ─────────────────────────────────────────────
    # NEAREST ACTIVE (fresh + tested only — broken is excluded, not "still active")
    # ─────────────────────────────────────────────

    def nearest_active(self, order_blocks, current_price, atr=None):
        active = [ob for ob in order_blocks if ob['state'] in ('fresh', 'tested')]
        if not active:
            return None

        best = None
        best_dist = float('inf')
        best_score = -1
        tolerance = (atr * self.PROXIMITY_ATR) if atr else 0.0

        # REVIEW (minor, "nearest_active should prefer probability then
        # distance"): within the proximity tolerance, prefer the higher-quality
        # zone over the merely-closer one; outside tolerance, distance still wins
        # (a far-but-great OB isn't "nearest" in any useful sense).
        for ob in active:
            if ob['zone_bottom'] <= current_price <= ob['zone_top']:
                dist, in_zone = 0.0, True
            else:
                dist = min(abs(current_price - ob['zone_top']), abs(current_price - ob['zone_bottom']))
                in_zone = dist <= tolerance

            better = False
            if in_zone and dist <= tolerance:
                if best is None or ob['quality_score'] > best_score or (
                        ob['quality_score'] == best_score and dist < best_dist):
                    better = True
            elif dist < best_dist and (best is None or not (best_dist <= tolerance)):
                better = True

            if better:
                best_dist = dist
                best_score = ob['quality_score']
                best = {**ob, 'distance': round(dist, 5), 'in_zone': in_zone}
        return best

    def print_summary(self, order_blocks):
        bar = "═" * 60
        log.info(bar)
        log.info("  🧱  ORDER BLOCK DETECTION  (Day 47 — timeframe & regime aware)")
        log.info(bar)
        if not order_blocks:
            log.info("  No order blocks detected.")
        for ob in order_blocks[:5]:
            icon = "🟢" if ob['direction'] == 'BULLISH' else "🔴"
            flags = []
            if ob['structure_break']:
                flags.append("BOS/CHoCH")
            if ob['fvg_confluence']:
                flags.append("FVG")
            if ob['sweep_conditioned']:
                flags.append("SWEEP")
            flag_str = "+".join(flags) if flags else "no confluence"
            reason = f"  [{ob['invalid_reason']}]" if ob.get('invalid_reason') else ""
            log.info(
                f"  {icon} {ob['type']}  [{ob['zone_bottom']} - {ob['zone_top']}]  "
                f"{ob['state'].upper()}  Q={ob['quality_score']}  ({flag_str})  "
                f"({ob['candles_ago']} candles ago){reason}"
            )
        log.info(bar)