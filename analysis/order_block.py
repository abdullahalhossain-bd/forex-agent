# analysis/order_block.py  —  Day 45 | Order Block Detection (v2 — research-audited rewrite)
#
# CHANGELOG vs Day 44 version — every change traces to a specific audit finding:
#
#   [CRITICAL-1] No structure-break gate  → FIXED. Every impulse now requires a
#       confirmed BOS/CHoCH (from market_structure.py) between the OB candle
#       and the impulse candle. Sources: videos 2,3,4,6 all require this;
#       video 6 states it as one of only two hard validity rules.
#   [CRITICAL-2] mitigated == "any wick touch" → FIXED. State is now a
#       3-way taxonomy: fresh / tested (wick touch, no close-through) /
#       broken (a candle CLOSED beyond the zone). Wick-touch used to
#       silently kill 'fresh' forever; that's wrong per video 5's explicit
#       close-vs-wick rule and bias-and-validation.md.
#   [CRITICAL-3] Repaint risk on forming bar → FIXED. detect() now takes
#       `closed_bars_only=True` as an explicit, documented contract; caller
#       is responsible for not passing an unclosed final bar. See docstring.
#   [IMPORTANT-1] No FVG gate → FIXED. FVGDetector is wired in; validity by
#       default requires an overlapping imbalance in the same displacement
#       leg (video 6's 2nd hard rule). Toggle via require_fvg=False if you
#       want to A/B test against the old, looser behavior.
#   [IMPORTANT-2] Single-candle-only OB definition → FIXED. Default mode is
#       now 'consecutive' (take the whole run of same-direction candles,
#       per video 6's explicit "avoid over-refining" argument). mode='single'
#       preserves the old Day-44 behavior for backtest comparison.
#   [IMPORTANT-3] No sweep-conditioning → FIXED. Best-corroborated concept
#       across the research (videos 4, 5, 6 independently agree): an OB
#       formed right after a liquidity sweep of a prior swing is the
#       highest-confidence type. Wired via MarketStructure.liquidity_sweep_before().
#   [IMPORTANT-4] Dedup kept earliest duplicate arbitrarily → FIXED. Now
#       keeps the highest quality_score among duplicates at the same ob_idx.
#
# NOT changed / explicitly deferred (ask before adding):
#   - Discount/Premium (50% Fib) ranking between competing same-direction OBs
#     (video 2) — not wired, needs a defined "leg" to measure the Fib from.
#   - Kill-zone/session timing quality bonus (video 4) — single-source concept,
#     needs session boundaries; not wired.
#   - market_regime.py (uploaded, ADX/ATR-based) is NOT wired into scoring yet
#     — could feed a "don't trust OBs in CHOPPY regime" penalty. Flagging as
#     optional, not doing it silently.

from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import get_logger
from analysis.fvg_detector import FVGDetector
from analysis.market_structure import MarketStructure

log = get_logger("order_block")


class OrderBlockDetector:
    IMPULSE_ATR_MULT = 1.8
    MAX_RESULTS = 10
    PROXIMITY_ATR = 0.3
    MAX_RUN_LOOKBACK = 6      # cap on how far back a consecutive same-direction run can extend
    SINGLE_LOOKBACK = 3       # legacy single-candle lookback (mode='single' only)
    SWEEP_LOOKBACK = 15       # bars searched for a pre-formation liquidity sweep
    FVG_SEARCH_WINDOW = 3     # bars around the impulse to look for a confluent FVG

    def __init__(self, mode: str = "consecutive", require_structure_break: bool = True,
                 require_fvg: bool = True, structure_strength: int = 2):
        """
        mode: 'consecutive' (default, video-6-backed) or 'single' (Day-44 legacy, for A/B backtest only)
        require_structure_break / require_fvg: hard validity gates. Both default True
            per the two rules video 6 states explicitly and that videos 2/3/4 corroborate.
            Set False only for controlled comparison runs — not for live signal generation.
        """
        if mode not in ("consecutive", "single"):
            raise ValueError("mode must be 'consecutive' or 'single'")
        self.mode = mode
        self.require_structure_break = require_structure_break
        self.require_fvg = require_fvg
        self.structure_strength = structure_strength
        self._fvg = FVGDetector()
        self._ms = MarketStructure()

    def detect(self, df: pd.DataFrame, closed_bars_only: bool = True, max_results: int | None = None) -> list[dict]:
        """
        CONTRACT: `df` must contain only CLOSED bars. If your caller has a live/
        forming candle appended, drop it before calling this — passing it in
        will make `state`/`fresh` repaint as that bar develops (see CRITICAL-3
        in the changelog above). `closed_bars_only` is a documentation flag,
        not an enforced check (we can't detect "still forming" from OHLC alone
        without a timestamp/now comparison the caller owns).
        """
        if len(df) < 20 or 'atr' not in df.columns:
            log.warning("[OrderBlock] Insufficient data or missing ATR column")
            return []
        if not closed_bars_only:
            log.warning("[OrderBlock] closed_bars_only=False — caller accepts repaint risk on the forming bar")

        opens = df['open'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        atrs = df['atr'].values
        n = len(df)

        structure_state = self._ms.analyze(df, strength=self.structure_strength)
        # max_results=0 → uncapped. OrderBlockDetector scans a full window looking
        # for historical confluence, unlike a live "what's active now" caller —
        # see Day-45 fix note in fvg_detector.py. Using the default-capped call
        # here would silently starve require_fvg=True of matches on any df longer
        # than ~10 FVGs' worth of history.
        # Round-14 fix: FVGDetector.detect() takes only `df` — it doesn't
        # accept max_results at all (always internally caps at its own
        # MAX_RESULTS=10). Calling it with max_results=0 raised a
        # TypeError on every single invocation, meaning order-block
        # detection was completely non-functional before this fix.
        fvgs = self._fvg.detect(df)

        raw = []
        for i in range(5, n):
            atr = atrs[i]
            if np.isnan(atr) or atr == 0:
                continue

            body = closes[i] - opens[i]
            if abs(body) < atr * self.IMPULSE_ATR_MULT:
                continue

            is_bullish_impulse = body > 0

            ob_start, ob_end = self._find_ob_run(opens, closes, i, is_bullish_impulse)
            if ob_start is None:
                continue

            # ── Gate 1: structure break (CRITICAL-1) ──────────────────────
            structure_event = structure_state and self._ms.structure_break_between(
                structure_state, start_idx=ob_end, end_idx=i)
            if self.require_structure_break and structure_event is None:
                continue

            # ── Gate 2: FVG confluence (IMPORTANT-1) ──────────────────────
            fvg_hit = self._find_confluent_fvg(fvgs, ob_end, i, is_bullish_impulse)
            if self.require_fvg and fvg_hit is None:
                continue

            zone_top = float(highs[ob_start:ob_end + 1].max())
            zone_bottom = float(lows[ob_start:ob_end + 1].min())
            body_top = float(max(opens[ob_end], closes[ob_end]))
            body_bottom = float(min(opens[ob_end], closes[ob_end]))
            ob_type = 'BULLISH_ORDER_BLOCK' if is_bullish_impulse else 'BEARISH_ORDER_BLOCK'

            # ── State taxonomy: fresh / tested / broken (CRITICAL-2) ──────
            tested_at, broken_at = self._scan_state(
                highs, lows, closes, i, zone_top, zone_bottom, is_bullish_impulse, n)
            state = 'broken' if broken_at is not None else ('tested' if tested_at is not None else 'fresh')

            # ── Sweep-conditioning (IMPORTANT-3) ───────────────────────────
            sweep = structure_state and self._ms.liquidity_sweep_before(
                df, structure_state, ob_start, lookback=self.SWEEP_LOOKBACK)

            quality_score = self._score(
                atr_ratio=abs(body) / atr,
                structure_break=structure_event is not None,
                fvg_confluence=fvg_hit is not None,
                sweep_conditioned=sweep is not None,
                state=state,
            )

            raw.append({
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
                'state': state,
                'fresh': state == 'fresh',
                'tested': state == 'tested',
                'broken': state == 'broken',
                'structure_break': structure_event is not None,
                'structure_event_type': structure_event['type'] if structure_event else None,
                'fvg_confluence': fvg_hit is not None,
                'sweep_conditioned': sweep is not None,
                'sweep_info': sweep,
                'quality_score': quality_score,
                'candles_ago': n - 1 - ob_end,
            })

        deduped = self._dedupe_keep_best(raw)
        deduped.sort(key=lambda r: (r['quality_score'], r['impulse_index']), reverse=True)
        log.info(f"[OrderBlock] Detected {len(deduped)} order blocks "
                 f"(mode={self.mode}, require_structure={self.require_structure_break}, require_fvg={self.require_fvg})")
        # Day-46 fix, same pattern as fvg_detector.py: MAX_RESULTS=10 is correct
        # for a live "top OBs right now" caller, wrong for a backtest scanning
        # full history. max_results=None preserves old behavior exactly.
        cap = self.MAX_RESULTS if max_results is None else max_results
        return deduped if cap == 0 else deduped[:cap]

    # ─────────────────────────────────────────────
    # OB RUN SELECTION
    # ─────────────────────────────────────────────

    def _find_ob_run(self, opens, closes, impulse_idx: int, is_bullish_impulse: bool):
        """
        Returns (run_start_idx, run_end_idx) — the candle range that forms the OB,
        both inclusive, run_end being the candle immediately preceding the impulse's
        opposite-color run.

        mode='single': legacy — just the first opposite-colored candle within
            SINGLE_LOOKBACK bars (Day-44 behavior).
        mode='consecutive': walk backward from impulse_idx-1 while candles keep the
            opposite color, capped at MAX_RUN_LOOKBACK (video 6's explicit fix for
            missed fills from over-refining to a single candle).
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
        limit = max(j - self.MAX_RUN_LOOKBACK, -1)
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
    # SCORING
    # ─────────────────────────────────────────────

    def _score(self, atr_ratio: float, structure_break: bool, fvg_confluence: bool,
               sweep_conditioned: bool, state: str) -> int:
        score = 30
        if structure_break:
            score += 20
        if fvg_confluence:
            score += 20
        if sweep_conditioned:
            score += 15   # best-corroborated concept in the research — weighted accordingly
        score += min(15, round(atr_ratio * 5))  # impulse strength, capped
        if state == 'tested':
            score -= 10
        elif state == 'broken':
            score -= 40
        return max(0, min(100, score))

    # ─────────────────────────────────────────────
    # DEDUP
    # ─────────────────────────────────────────────

    def _dedupe_keep_best(self, raw: list[dict]) -> list[dict]:
        best_by_idx: dict[int, dict] = {}
        for r in raw:
            key = r['run_end_index']
            existing = best_by_idx.get(key)
            if existing is None or r['quality_score'] > existing['quality_score']:
                best_by_idx[key] = r
        return list(best_by_idx.values())

    # ─────────────────────────────────────────────
    # NEAREST ACTIVE (fresh + tested only — broken is excluded, not "still active")
    # ─────────────────────────────────────────────

    def nearest_active(self, order_blocks, current_price, atr=None):
        active = [ob for ob in order_blocks if ob['state'] in ('fresh', 'tested')]
        if not active:
            return None

        best = None
        best_dist = float('inf')
        tolerance = (atr * self.PROXIMITY_ATR) if atr else 0.0

        for ob in active:
            if ob['zone_bottom'] <= current_price <= ob['zone_top']:
                dist, in_zone = 0.0, True
            else:
                dist = min(abs(current_price - ob['zone_top']), abs(current_price - ob['zone_bottom']))
                in_zone = dist <= tolerance

            if dist < best_dist:
                best_dist = dist
                best = {**ob, 'distance': round(dist, 5), 'in_zone': in_zone}
        return best

    def print_summary(self, order_blocks):
        bar = "═" * 60
        log.info(bar)
        log.info("  🧱  ORDER BLOCK DETECTION  (Day 45 — research-audited)")
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
            log.info(
                f"  {icon} {ob['type']}  [{ob['zone_bottom']} - {ob['zone_top']}]  "
                f"{ob['state'].upper()}  Q={ob['quality_score']}  ({flag_str})  "
                f"({ob['candles_ago']} candles ago)"
            )
        log.info(bar)