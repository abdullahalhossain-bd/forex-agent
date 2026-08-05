# analysis/fvg_detector.py  —  Day 46 | Fair Value Gap (FVG) Detection (v2 — audited rewrite)
# ============================================================
# Fair Value Gap = ৩-candle imbalance pattern। দ্রুত move-এর কারণে
# candle 1 আর candle 3-এর মধ্যে একটা "ফাঁকা" zone থেকে যায়, যেটা price
# পরে এসে fill করতে পারে (mean-reversion magnet)।
#
#   Bullish FVG : candle3.low  > candle1.high   → gap [c1.high, c3.low]
#   Bearish FVG : candle3.high < candle1.low     → gap [c3.high, c1.low]
#
# CHANGELOG vs Day 44 (v1) — traces to fvg_detector_review.md findings,
# plus the same overfitting pass applied to order_block.py:
#
#   [REVIEW-1 / OVERFITTING] `MIN_GAP_ATR_MULT = 0.10` was one static,
#       non-timeframe-aware magic number gating every candidate → SOFTENED.
#       Replaced with a per-timeframe base threshold (TIMEFRAME_PARAMS,
#       same honesty caveat as order_block.py: these are carried-over
#       priors, not walk-forward validated) PLUS a composite gap_score
#       (gap-as-ATR% + middle-candle displacement + close-efficiency +
#       optional volume-spike) so no single ratio alone decides pass/fail.
#   [REVIEW-2] Middle candle (the displacement candle) was completely
#       ignored → FIXED. Its body%, ATR expansion and close-efficiency now
#       feed the composite gap_score — a 3-bar pattern with a weak middle
#       candle is a much lower-confidence imbalance than one with a strong
#       displacement candle, even at the same raw gap size.
#   [REVIEW-3] `filled` was a single boolean flipped by ANY touch, and
#       `fill_pct` took the max of any single touch's overlap (not
#       cumulative) → FIXED. State is now FRESH / PARTIAL / FILLED / BROKEN,
#       and fill_pct is the union of all covered sub-intervals across every
#       touch (interval-merge), so two 30%-overlap touches on different
#       sides of the zone correctly read as ~60% filled, not 30%.
#   [REVIEW-4] No time decay → FIXED, same exp(-age/half_life) pattern as
#       order_block.py, timeframe-scaled, capped at 50% max discount.
#   [REVIEW-5] No regime awareness → FIXED. Optional MarketRegimeDetector
#       wiring (CHOPPY regime penalizes; pass `regime_ctx=` to reuse an
#       already-computed regime instead of recomputing).
#   [REVIEW-6] No stacked-FVG awareness → PARTIALLY FIXED. Adjacent/
#       overlapping same-direction gaps are now flagged with `stack_count`
#       / `is_stacked` (a lightweight signal, not a full nested-gap merge
#       engine — see deferred list below).
#   [REVIEW-7] No invalidation reason / no explainable score → FIXED.
#       `invalid_reason` (BROKEN/AGED/LOW_SCORE) and `score_breakdown` added,
#       same convention as order_block.py so downstream code that already
#       reads one detector's breakdown format can read both.
#   [REVIEW-8] detect() mixed scanning + fill-simulation in one flow →
#       split into _resolve_params / _scan_gap / _scan_fill / _score.
#
# Deferred (flagging, not doing silently — each is a meaningfully-sized
# feature that deserves its own design pass, not a bolt-on here):
#   - IFVG (Inverse FVG) — needs its own invalidation-then-reversal state
#     machine distinct from plain BROKEN; would double the state surface.
#   - Balanced Price Range (BPR) — needs opposing bullish+bearish gap
#     pairing logic across two impulses, not a single-gap concept.
#   - Full nested/stacked gap MERGE (collapsing overlapping gaps into one
#     combined zone) — `stack_count` flags the situation; actually merging
#     zones changes zone_top/zone_bottom semantics for every consumer
#     (order_block.py's _find_confluent_fvg, liquidity_engine.py, etc.)
#     and should ship together with those call-site updates.
#   - Multi-timeframe FVG alignment — mtf_analyzer.py/smc_engine.py already
#     own cross-TF orchestration; duplicating it here is a second source
#     of truth.
#   - dict → dataclass — same cross-module breaking-change reasoning as
#     order_block.py's deferral (smc_engine, smart_money, liquidity_engine,
#     fibonacci.py, agents/* all consume this as a plain dict with []/.get()).
# ============================================================

import math
import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("fvg_detector")

# ─────────────────────────────────────────────────────────────────
# TIMEFRAME-AWARE PARAMETERS (REVIEW-1). Same caveat as order_block.py:
# these are scaled-by-intuition starting points carried over from the old
# fixed 0.10 constant, NOT independently walk-forward validated per pair/TF.
# ─────────────────────────────────────────────────────────────────
TIMEFRAME_PARAMS = {
    'M1':  {'min_gap_atr_mult': 0.08, 'decay_half_life': 60},
    'M5':  {'min_gap_atr_mult': 0.09, 'decay_half_life': 90},
    'M15': {'min_gap_atr_mult': 0.10, 'decay_half_life': 150},
    'M30': {'min_gap_atr_mult': 0.11, 'decay_half_life': 180},
    'H1':  {'min_gap_atr_mult': 0.12, 'decay_half_life': 220},
    'H4':  {'min_gap_atr_mult': 0.14, 'decay_half_life': 300},
    'D1':  {'min_gap_atr_mult': 0.16, 'decay_half_life': 400},
}
DEFAULT_TIMEFRAME = 'M15'   # preserves exact Day-44 numeric behavior when timeframe isn't passed

SCORE_WEIGHTS = {
    'base':        30,
    'gap_strength': 40,   # graded 0..40 by the composite gap_score
    'stacked':      10,   # graded 0..10, bonus for adjacent same-direction gaps
    'regime':        5,   # graded -5..+5
}
STATE_PENALTY = {'fresh': 0, 'partial': -8, 'filled': -25, 'broken': -45}
AGED_CANDLES_THRESHOLD = 300
LOW_SCORE_THRESHOLD = 35
STACK_ZONE_GAP_ATR = 1.0   # same-direction gaps within this many ATRs of each other count as "stacked"
FILLED_THRESHOLD = 0.92    # cumulative fill_pct at/above this is treated as fully FILLED, not PARTIAL


class FVGDetector:
    """
    Usage:
        detector = FVGDetector(timeframe='M15')
        fvgs = detector.detect(df)   # df-এ আগে থেকে 'atr' column থাকতে হবে
        nearest = detector.nearest_active(fvgs, current_price)
    """

    # Legacy class attrs kept for anything referencing them directly;
    # actual per-call values come from _resolve_params(timeframe).
    MIN_GAP_ATR_MULT = TIMEFRAME_PARAMS[DEFAULT_TIMEFRAME]['min_gap_atr_mult']
    MAX_RESULTS       = 10
    PROXIMITY_ATR      = 0.3

    def __init__(self, timeframe: str = DEFAULT_TIMEFRAME, use_regime_filter: bool = True):
        self.params = self._resolve_params(timeframe)
        self.use_regime_filter = use_regime_filter
        self._regime = None
        if use_regime_filter:
            try:
                from analysis.market_regime import MarketRegimeDetector
                self._regime = MarketRegimeDetector()
            except Exception as e:
                log.warning(f"[FVG] market_regime unavailable ({e}) — regime scoring disabled")
                self.use_regime_filter = False

    def _resolve_params(self, timeframe: str) -> dict:
        tf = (timeframe or DEFAULT_TIMEFRAME).upper()
        if tf not in TIMEFRAME_PARAMS:
            log.warning(f"[FVG] Unknown timeframe '{timeframe}', falling back to {DEFAULT_TIMEFRAME}")
            tf = DEFAULT_TIMEFRAME
        return dict(TIMEFRAME_PARAMS[tf])

    def detect(self, df: pd.DataFrame, max_results: int | None = None,
               timeframe: str | None = None, regime_ctx: dict | None = None) -> list[dict]:
        """
        timeframe: optional per-call override of the instance's timeframe.
        regime_ctx: optional pre-computed MarketRegimeDetector.detect(df) result
            to avoid recomputing regime when the caller already has it.
        """
        if len(df) < 10 or 'atr' not in df.columns:
            log.warning("[FVG] Insufficient data or missing ATR column")
            return []

        params = self.params if timeframe is None else self._resolve_params(timeframe)

        opens = df['open'].values if 'open' in df.columns else None
        highs = df['high'].values
        lows  = df['low'].values
        closes = df['close'].values
        atrs  = df['atr'].values
        vols = None
        for vol_col in ('volume', 'tick_volume'):
            if vol_col in df.columns:
                vols = df[vol_col].values
                break
        n     = len(df)

        if self.use_regime_filter and regime_ctx is None:
            try:
                regime_ctx = self._regime.detect(df)
            except Exception as e:
                log.warning(f"[FVG] regime detection failed ({e}) — continuing without it")
                regime_ctx = None

        raw = []
        for i in range(2, n):
            cand = self._scan_gap(i, opens, highs, lows, closes, atrs, vols, n, params)
            if cand is not None:
                raw.append(cand)

        self._flag_stacked(raw, atrs)
        for r in raw:
            r['quality_score'], r['score_breakdown'] = self._score(r, regime_ctx, params)
            r['invalid_reason'] = self._invalid_reason(r)
            r.pop('_atr', None)  # internal-only, used by _flag_stacked

        deduped = sorted(raw, key=lambda r: (r['quality_score'], r['index']), reverse=True)
        log.info(f"[FVG] Detected {len(deduped)} fair value gaps "
                 f"(regime={'on' if self.use_regime_filter else 'off'})")
        cap = self.MAX_RESULTS if max_results is None else max_results
        return deduped if not cap else deduped[:cap]

    # ─────────────────────────────────────────────
    # GAP SCAN (REVIEW-1, REVIEW-2, REVIEW-8)
    # ─────────────────────────────────────────────

    def _scan_gap(self, i, opens, highs, lows, closes, atrs, vols, n, params):
        atr = atrs[i]
        if np.isnan(atr) or atr == 0:
            return None

        c1_high, c1_low = highs[i - 2], lows[i - 2]
        c3_high, c3_low = highs[i], lows[i]

        direction = None
        zone_bottom = zone_top = None
        if c3_low > c1_high:
            direction, zone_bottom, zone_top = 'BULLISH', c1_high, c3_low
        elif c3_high < c1_low:
            direction, zone_bottom, zone_top = 'BEARISH', c3_high, c1_low
        if direction is None:
            return None

        gap = zone_top - zone_bottom
        gap_atr_ratio = gap / atr
        if gap_atr_ratio < params['min_gap_atr_mult'] * 0.7:
            # Same "no single number rescues a fail" pattern as order_block.py:
            # this hard floor is well below the nominal threshold, so a strong
            # displacement/volume score can still lift a borderline gap over
            # the line, but a genuinely tiny gap can't be talked into passing.
            return None

        # ── Middle-candle displacement (REVIEW-2) ──────────────────────────
        mid = i - 1
        displacement_score = 0.0
        if opens is not None:
            body = closes[mid] - opens[mid]
            candle_range = highs[mid] - lows[mid]
            body_atr_ratio = abs(body) / atr
            close_efficiency = (abs(body) / candle_range) if candle_range > 0 else 0.0
            same_dir = (direction == 'BULLISH' and body > 0) or (direction == 'BEARISH' and body < 0)
            displacement_score = (0.6 * min(body_atr_ratio / 1.2, 1.5) + 0.4 * close_efficiency)
            if not same_dir:
                displacement_score *= 0.4  # displacement candle fighting the gap direction is a weak signal

        vol_spike = 0.0
        if vols is not None and i >= 10:
            recent_avg = np.nanmean(vols[max(0, i - 10):i]) or 0.0
            if recent_avg > 0:
                vol_spike = min(2.0, vols[i - 1] / recent_avg) / 2.0  # 0..1, keyed to the middle/displacement candle

        gap_score = (0.5 * min(gap_atr_ratio / params['min_gap_atr_mult'], 2.0) / 2.0
                     + 0.35 * min(displacement_score, 1.0)
                     + 0.15 * vol_spike)
        if gap_atr_ratio < params['min_gap_atr_mult'] and gap_score < 0.55:
            # Below the nominal size AND a weak composite — reject.
            return None

        tested_at, broken_at, fill_pct = self._scan_fill(
            highs, lows, closes, i, zone_top, zone_bottom, direction, n)

        if broken_at is not None:
            state = 'broken'
        elif fill_pct >= FILLED_THRESHOLD:
            state = 'filled'
        elif tested_at is not None:
            state = 'partial'
        else:
            state = 'fresh'

        return {
            'type':          'FVG',
            'direction':     direction,
            'index':         i,
            'mid_index':     mid,
            'zone_top':      round(float(zone_top), 5),
            'zone_bottom':   round(float(zone_bottom), 5),
            'gap_atr_ratio': round(float(gap_atr_ratio), 3),
            'gap_score':     round(float(min(1.0, gap_score)), 3),
            'state':         state,
            'filled':        state in ('filled', 'broken'),
            'fresh':         state == 'fresh',
            'fill_pct':      fill_pct,
            'stack_count':   0,        # filled in by _flag_stacked()
            'is_stacked':    False,
            'candles_ago':   n - 1 - i,
            '_atr':          float(atr),
        }

    def _scan_fill(self, highs, lows, closes, i, zone_top, zone_bottom, direction, n):
        """
        Cumulative fill (REVIEW-3): union of every touch's covered
        sub-interval, not the max of any single touch. BROKEN = a candle
        CLOSES fully beyond the zone on the side that invalidates the gap's
        expected reaction (below zone_bottom for a bullish/support gap,
        above zone_top for a bearish/resistance gap) — close-based, not
        wick-based, mirroring order_block.py's state rule.
        """
        tested_at = None
        broken_at = None
        covered = []
        zone_height = zone_top - zone_bottom
        for k in range(i + 1, n):
            lo = max(lows[k], zone_bottom)
            hi = min(highs[k], zone_top)
            if hi > lo:
                if tested_at is None:
                    tested_at = k
                covered.append((lo, hi))
            if direction == 'BULLISH' and closes[k] < zone_bottom:
                broken_at = k
                break
            if direction == 'BEARISH' and closes[k] > zone_top:
                broken_at = k
                break

        fill_pct = 0.0
        if covered and zone_height > 0:
            covered.sort()
            merged = [covered[0]]
            for lo, hi in covered[1:]:
                if lo <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
                else:
                    merged.append((lo, hi))
            union = sum(hi - lo for lo, hi in merged)
            fill_pct = round(min(1.0, union / zone_height), 2)

        return tested_at, broken_at, fill_pct

    # ─────────────────────────────────────────────
    # STACKED / ADJACENT GAP FLAGGING (REVIEW-6, lightweight — see deferred
    # list at the top for the full nested-merge version)
    # ─────────────────────────────────────────────

    def _flag_stacked(self, gaps: list[dict], atrs):
        by_dir: dict[str, list[dict]] = {}
        for g in gaps:
            by_dir.setdefault(g['direction'], []).append(g)
        for direction, group in by_dir.items():
            group.sort(key=lambda g: g['index'])
            for a in group:
                count = 0
                for b in group:
                    if a is b:
                        continue
                    atr = a['_atr'] or 1e-9
                    # "adjacent/overlapping" = zones within STACK_ZONE_GAP_ATR
                    # of each other AND formed within a handful of bars.
                    zone_gap = max(0.0, max(a['zone_bottom'], b['zone_bottom'])
                                    - min(a['zone_top'], b['zone_top']))
                    if zone_gap / atr <= STACK_ZONE_GAP_ATR and abs(a['index'] - b['index']) <= 5:
                        count += 1
                a['stack_count'] = count
                a['is_stacked'] = count >= 1

    # ─────────────────────────────────────────────
    # SCORING (REVIEW-7: weighted, explainable, decay + regime aware)
    # ─────────────────────────────────────────────

    def _score(self, r: dict, regime_ctx: dict | None, params: dict) -> tuple[int, dict]:
        breakdown = {'base': SCORE_WEIGHTS['base']}
        score = SCORE_WEIGHTS['base']

        gap_pts = round(r['gap_score'] * SCORE_WEIGHTS['gap_strength'])
        breakdown['gap_strength'] = gap_pts
        score += gap_pts

        stacked_pts = min(SCORE_WEIGHTS['stacked'], r['stack_count'] * 5)
        breakdown['stacked'] = stacked_pts
        score += stacked_pts

        regime_pts = 0
        if regime_ctx:
            regime = regime_ctx.get('regime')
            reg_dir = regime_ctx.get('direction')
            if regime == 'CHOPPY':
                regime_pts = -SCORE_WEIGHTS['regime']
            elif regime == 'TRENDING' and reg_dir in ('BULLISH', 'BEARISH'):
                regime_pts = SCORE_WEIGHTS['regime'] if reg_dir == r['direction'] else -round(SCORE_WEIGHTS['regime'] / 2)
        breakdown['regime'] = regime_pts
        score += regime_pts

        state_pts = STATE_PENALTY.get(r['state'], 0)
        breakdown['state_penalty'] = state_pts
        score += state_pts

        raw_score = max(0, min(100, score))

        decay_factor = math.exp(-r['candles_ago'] / max(params['decay_half_life'], 1))
        breakdown['decay_factor'] = round(decay_factor, 3)
        final_score = max(0, min(100, round(raw_score * (0.5 + 0.5 * decay_factor))))

        return final_score, breakdown

    def _invalid_reason(self, r: dict) -> str | None:
        if r['state'] == 'broken':
            return 'BROKEN'
        if r['candles_ago'] > AGED_CANDLES_THRESHOLD:
            return 'AGED'
        if r['quality_score'] < LOW_SCORE_THRESHOLD:
            return 'LOW_SCORE'
        return None

    # ─────────────────────────────────────────────
    # NEAREST ACTIVE GAP
    # ─────────────────────────────────────────────

    def nearest_active(self, fvgs: list[dict], current_price: float, atr: float = None) -> dict | None:
        # 'fresh' key kept for backward compat but active now also allows
        # PARTIAL (a gap that's ~40% filled is still a live magnet, not gone).
        active = [g for g in fvgs if g.get('state', 'fresh' if g.get('fresh') else 'filled') in ('fresh', 'partial')]
        if not active:
            return None

        best = None
        best_dist = float('inf')
        best_score = -1
        tolerance = (atr * self.PROXIMITY_ATR) if atr else 0.0

        for g in active:
            if g['zone_bottom'] <= current_price <= g['zone_top']:
                dist, in_zone = 0.0, True
            else:
                dist = min(
                    abs(current_price - g['zone_top']),
                    abs(current_price - g['zone_bottom']),
                )
                in_zone = dist <= tolerance

            score = g.get('quality_score', 0)
            better = False
            if in_zone and dist <= tolerance:
                if best is None or score > best_score or (score == best_score and dist < best_dist):
                    better = True
            elif dist < best_dist and (best is None or not (best_dist <= tolerance)):
                better = True

            if better:
                best_dist = dist
                best_score = score
                best = {**g, 'distance': round(dist, 5), 'in_zone': in_zone}

        return best

    # ─────────────────────────────────────────────
    # PRINT SUMMARY
    # ─────────────────────────────────────────────

    def print_summary(self, fvgs: list[dict]) -> None:
        bar = "═" * 48
        log.info(bar)
        log.info("  🌀  FAIR VALUE GAP DETECTION  (Day 46 — regime & decay aware)")
        log.info(bar)
        if not fvgs:
            log.info("  No fair value gaps detected.")
        for g in fvgs[:5]:
            icon = "🟢" if g['direction'] == 'BULLISH' else "🔴"
            tag  = g.get('state', 'fresh' if g.get('fresh') else 'filled').upper()
            stack = f"  stacked x{g['stack_count']}" if g.get('is_stacked') else ""
            reason = f"  [{g['invalid_reason']}]" if g.get('invalid_reason') else ""
            log.info(
                f"  {icon} FVG  [{g['zone_bottom']} - {g['zone_top']}]  "
                f"{tag} ({int(g['fill_pct']*100)}%)  Q={g.get('quality_score', '-')}  "
                f"({g['candles_ago']} candles ago){stack}{reason}"
            )
        log.info(bar)