# analysis/fibonacci.py
# ============================================================
# Day 40 — Fibonacci Engine (AI Price Zone Intelligence)
#
# Features:
#   ✅ Auto Swing High / Low Detection (Dynamic, TF-aware)
#   ✅ Fibonacci Retracement (23.6, 38.2, 50, 61.8, 78.6)
#   ✅ Fibonacci Extension (127.2, 161.8, 261.8)
#   ✅ Confluence Analysis (Fib + S/R + Structure + optional EMA/VWAP)
#   ✅ Fibonacci Signal Integration (AI context)
#   ✅ Fibonacci Failure Detection
#   ✅ Memory-ready output (fib_history table)
#   ✅ Liquidity-sweep trigger (wick through 61.8%, close back inside)
#   ✅ Higher-timeframe trend alignment (soft bonus, optional hard gate)
#   ✅ Configurable level-set presets ('standard' / 'conservative')
#
# Update pass (reviewed against the uploaded strategy transcripts/articles):
# most of the "smart money" framing in that material (liquidity magnets,
# institutional traps, etc.) is narrative rather than mechanism — but two
# concrete, implementable, repeatedly-corroborated ideas were missing from
# this engine and are added below:
#   1. Liquidity sweep as a trigger pattern, distinct from a plain pin bar
#      (wick clears the 61.8% boundary, close reclaims it same bar).
#   2. Higher-timeframe trend context as a confidence input (and optional
#      gate) — every source that discussed multi-timeframe use agreed a
#      Fib setup against the HTF trend performs worse, matching
#      trading-concepts.md's top-down MTA guidance.
# Also added: an optional 'conservative' level-set preset matching the
# standalone settings article, and confluence support for arbitrary extra
# levels (EMA/anchored VWAP) beyond S/R. Fibonacci Time Zones and the
# 0.65 "golden pocket" from the other sources were deliberately NOT added
# — both are acknowledged even within that source material as having no
# mathematical/geometric basis (arbitrary projections / arbitrary ratio),
# so they'd add configuration surface without a defensible edge.
#
# Day 43 — 5-year walk-forward validation (EURUSD H1/H4/M15,
# 2021-07-27 to 2026-07-24, no look-ahead: rolling 250-bar decision
# window, outcomes checked only on subsequent bars):
#   - require_engulfing_only default flipped True->see its docstring
#     for the full before/after numbers. This is the ONLY default
#     changed by this pass — every other Day-42 gate (require_trend_ma,
#     min_adx, max_atr_multiple, min_confluence_strength, golden-zone
#     block) was re-tested and NOT changed:
#       * require_trend_ma=True cut trades to n=8 and expectancy stayed
#         negative (-0.134R) — not statistically significant either way,
#         left at its existing default (False).
#       * min_confluence_strength=65 produced zero effect in this
#         harness because no external sr_ctx (S/R) was wired in, so
#         Fib-only confluence already scores >=65 by construction —
#         this gate needs re-validation once called with real sr_ctx;
#         inconclusive here, so left unchanged.
#   - Independent level-touch study (n=923-2829 touches per level,
#     H1, hold = no close beyond level by 0.5xATR within 10 bars):
#     23.6% held only 32.3% of touches (weakest), 50.0% held 43.1%
#     (strongest), 38.2% had the highest reaction rate (79.3%). This
#     is consistent with — and now evidence-backed support for — the
#     existing 'conservative' preset's decision to drop 23.6% entirely.
#     No code change made here since 'conservative' already exists as
#     an opt-in preset; this is corroborating evidence, not a new fix.
#   - Directional asymmetry observed (BUY 50.0% WR vs SELL 22.2% WR
#     under require_engulfing_only, this sample): NOT implemented as a
#     long-only filter — 5 years of one pair is one historical regime
#     (EUR broadly weak for much of this window), and hard-coding a
#     directional bias off one sample's regime is the overfitting this
#     review is supposed to guard against. Flagged for monitoring, not
#     acted on.
# ============================================================
import os
import pandas as pd
import numpy as np
from utils.logger import get_logger
from utils.pip_utils import pip_size, price_to_pips

log = get_logger(__name__)

# ── Standard Fibonacci levels ──────────────────────────────────
FIB_RETRACEMENT_LEVELS = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
FIB_EXTENSION_LEVELS   = [1.0, 1.272, 1.618, 2.0, 2.618]

# ── ABC Expansion ratios (Phase 1) ──────────────────────────────
# Distinct mechanism from the 2-point extension above: extension assumes
# the next leg starts back at B (i.e. C == A after a full retrace);
# expansion instead uses the ACTUAL correction low/high (point C) as the
# projection anchor, which is where price is really trading from. Ratios
# are the standard ABC set used for TP projection (61.8% = shallow target,
# 100% = AB=CD equal-leg target, 161.8% = golden extension, 261.8% = deep).
EXPANSION_RATIOS = [0.618, 1.0, 1.272, 1.618, 2.618]

# ── Level-set presets ───────────────────────────────────────────
# The reference material reviewed doesn't agree on one "correct" grid:
# some sources keep 23.6% as a valid shallow-pullback level and use
# 127.2/161.8/261.8% extension targets (what's coded above); a separate
# "best settings" article explicitly recommends dropping 23.6% and
# 2.0/2.618% extensions for a tighter 0/38.2/50/61.8/78.6/100%
# retracement grid with 1.272/1.414/1.618% extension targets. Neither
# is objectively "more correct" — it's a settings choice, not a math
# correction — so both are exposed as presets instead of one silently
# overwriting the other. Default ('standard') preserves prior behavior
# exactly; existing callers/backtests are unaffected.
LEVEL_SET_PRESETS = {
    'standard': {
        'retracement': [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0],
        'extension':   [1.0, 1.272, 1.618, 2.0, 2.618],
    },
    # Matches the "best Fibonacci retracement/extension levels for
    # Forex & Gold" article: keep 0/38.2/50/61.8/78.6/1.0 for
    # retracement, and 1.272/1.414/1.618 (no 2.0/2.618) for extension.
    'conservative': {
        'retracement': [0.0, 0.382, 0.500, 0.618, 0.786, 1.0],
        'extension':   [1.0, 1.272, 1.414, 1.618],
    },
}

# ── Confluence tolerance (pips equivalent) ─────────────────────
# FIX (institutional review, item #1): this used to be a fixed 0.0015
# (i.e. "15 pips") regardless of instrument, which is correct for 5-decimal
# pairs (EURUSD, GBPUSD, ...) but wrong by ~100x for JPY crosses and metals.
# CONFLUENCE_TOLERANCE_PIPS is now the pip COUNT (instrument-independent);
# it gets converted to a price distance via pip_size(symbol) at call time.
# CONFLUENCE_TOLERANCE is kept as the old literal value for backward
# compatibility with any external code that imports this constant directly —
# it is no longer used internally by this module.
CONFLUENCE_TOLERANCE_PIPS = 15
CONFLUENCE_TOLERANCE = 0.0015   # deprecated — kept only for backward compat

# ── Dynamic swing window per timeframe ─────────────────────────
TF_SWING_WINDOW = {
    '1m':  3,
    '5m':  5,
    'M5':  5,
    '15m': 7,
    'M15': 7,
    '30m': 8,
    '1h':  10,
    'H1':  10,
    '4h':  14,
    'H4':  14,
    '1d':  20,
    'D1':  20,
}
DEFAULT_SWING_WINDOW = 7


class FibonacciEngine:
    """
    AI-powered Fibonacci analysis engine।

    Auto swing detect করে Fibonacci levels calculate করে,
    S/R confluence খোঁজে এবং trade signal দেয়।

    Usage:
        fib = FibonacciEngine(timeframe='H1')
        result = fib.analyze(df, sr_ctx=sr_ctx)
        ctx = fib.get_ai_context(result)
    """

    def __init__(self, timeframe: str = '15m', symbol: str = None,
                 require_trigger_candle: bool = True, min_rr: float = 1.5,
                 level_set: str = 'standard', require_htf_alignment: bool = False,
                 min_confluence_strength: int = 0,
                 require_engulfing_only: bool = True,
                 require_trend_ma: bool = False, trend_ma_period: int = 200,
                 min_adx: float = 0.0,
                 max_atr_multiple: float = 0.0,
                 allow_golden_zone: bool = False,
                 require_cluster_for_golden: bool = True,
                 allow_cluster_touch_entry: bool = False,
                 min_rr_dynamic: bool = False,
                 max_swing_age_bars: int = 0,
                 require_volume_confirmation: bool = False,
                 volume_multiple: float = 1.5,
                 require_liquidity_alignment: bool = False,
                 min_liquidity_score: int = 0):
        """
        symbol : optional instrument symbol (e.g. "EURUSD", "USDJPY",
            "XAUUSD"). Used only to pick the correct pip size for
            confluence/distance calculations. If omitted, pip math falls
            back to the standard 0.0001 pip size (previous behavior) —
            existing callers that don't pass a symbol are unaffected.

        require_trigger_candle : if True (recommended — this is what every
            source you shared agrees on, including the video transcript),
            a BUY/SELL bias is only emitted when the last CLOSED candle
            shows an actual reversal/momentum trigger at the Fib zone —
            not just because price happened to touch the zone. Without
            this, "price is in the golden zone" alone fires a signal even
            while price is still free-falling through it.

        min_rr : minimum reward:risk (TP distance / SL distance) required
            for a BUY/SELL to stand; otherwise downgraded to WAIT. The
            strategies you shared all specify a minimum (2:1 for shallow
            pullbacks, ~1.5-2:1 for golden zone with stepped targets) —
            1.5 is used as a single conservative default across zones.

        level_set : 'standard' (default, unchanged prior behavior) or
            'conservative' (the leaner 0/38.2/50/61.8/78.6/100% retracement
            + 1.272/1.414/1.618% extension grid from the "best settings"
            article). See LEVEL_SET_PRESETS. Purely a display/measurement
            grid choice — does not change swing detection or signal logic.

        require_htf_alignment : if True, a higher-timeframe trend context
            passed into analyze(htf_ctx=...) becomes a hard GATE — a
            candidate BUY/SELL is downgraded to WAIT when htf_ctx['trend']
            conflicts with this timeframe's trend, matching the repeated
            claim across the shared material that "when they disagree,
            your setup fails even if it looks perfect." Default False
            (soft confidence adjustment only) because most callers won't
            have htf_ctx wired up yet — turning this on with no htf_ctx
            supplied is a no-op, so it's safe to flip on incrementally.

        --- Backtest-driven additions (Day 42) ---
        The following params exist because a live backtest (EURUSD H1, 22
        trades, 36.4% win rate) surfaced specific, evidence-backed
        weaknesses. All default to preserving EXACT prior behavior — every
        one of them is opt-in. Two of them (allow_golden_zone,
        allow_cluster_touch_entry) pull in the OPPOSITE direction from the
        others (they loosen instead of tighten) — see their docstrings for
        why both exist and are still individually defensible.

        min_confluence_strength : hard-gates a candidate unless the
            strongest confluence zone's `strength` score is >= this value
            (0 disables the gate — prior behavior). The backtest showed
            confidence (the blended heuristic score) did NOT predict
            winners — 80+ confidence signals had a 0% win rate on n=2 —
            but confluence *strength* (S/R + multi-level agreement) is a
            more mechanistic, less gameable proxy. Suggested value: 60-75.

        require_engulfing_only : if True, only 'bullish_engulfing' /
            'bearish_engulfing' (plus 'liquidity_sweep' and
            'cluster_touch', which are independently structurally
            validated, not generic candle shapes) count as a confirmed
            trigger — pin bars and momentum breakouts are rejected.
            DEFAULT FLIPPED TO True (institutional review, 5yr walk-forward
            backtest, EURUSD H1, 2021-07 to 2026-07, no look-ahead —
            each decision used only a rolling 250-bar window ending at
            the closed bar, outcomes checked only on later bars):
              - All trades (default gates otherwise unchanged): n=40,
                WR 27.5%, PF 0.93, expectancy -0.053R, net -2.14R,
                max DD 13.34R (losing).
              - require_engulfing_only=True: n=30, WR 33.3%, PF 1.21,
                expectancy +0.137R, net +4.1R, max DD 6.34R (profitable,
                lower drawdown, fewer but better trades).
              - Trigger-level breakdown on the default-gate sample:
                bullish_engulfing 54.5% WR (n=11), bearish_engulfing
                20.0% WR (n=15), bullish_pin_bar 12.5% WR (n=8),
                bearish_pin_bar 16.7% WR (n=6) — pin bars underperformed
                in both directions, corroborating the original n=22
                smoke-test finding this parameter was already built
                around. Two independent samples now agree in direction.
              Caveat: n=30 over 5 years (~6 trades/yr) is still a modest
              sample for one pair/timeframe — directionally consistent
              across two samples, but treat as a validated prior, not a
              guarantee; re-check if live/expanded-universe results
              diverge. Set False to restore the original (pre-review)
              behavior and re-admit pin-bar/momentum triggers.

        require_trend_ma / trend_ma_period : if require_trend_ma is True,
            a BUY is only allowed when curr_price > the trend_ma_period
            SMA of closes, and a SELL only when curr_price < it (computed
            from `df` inside analyze()). A simple, widely-used
            trend-alignment filter — default period 200. Requires at
            least trend_ma_period closed bars; silently a no-op (not an
            error) if the caller's df is shorter.

        min_adx : if > 0, a candidate is gated to WAIT unless
            ind_ctx['adx'] >= min_adx. The backtest's bullish trades
            (18.2% win rate) badly underperformed bearish (54.5%) in what
            was a bearish-trending window — consistent with Fib
            retracement setups needing a genuine trend to retrace FROM,
            not a ranging market. Suggested value: ~22-25. No-op if
            ind_ctx doesn't carry an 'adx' key (rather than silently
            passing — see the reason string when this blocks a trade).

        max_atr_multiple : if > 0, gates out setups where the current ATR
            exceeds max_atr_multiple × the trailing average ATR (see
            _calc_atr_avg) — i.e. avoid entries during volatility spikes
            prone to false breakouts/whipsaw. Suggested value: ~1.5-2.0.

        allow_golden_zone / require_cluster_for_golden : the backtest
            showed the 50-61.8% "golden zone" band had a 0% win rate in
            this sample (n was small — treat as a strong prior, not
            certainty) even though the broader 0.382-0.786 candidate band
            below structurally includes it. allow_golden_zone=False
            (default) explicitly excludes GOLDEN_ZONE entries regardless
            of the ratio band. Setting it True re-admits them, but ONLY
            when require_cluster_for_golden is also True (default) AND a
            multi-swing cluster (see detect_clusters) sits at the current
            price — the idea being a bare golden-zone touch performed
            worst, but independent-swing-scale agreement at that same
            nominal level is a materially different, more corroborated
            signal. Set require_cluster_for_golden=False to admit ANY
            golden-zone touch once allow_golden_zone=True — not
            recommended given the backtest evidence, but exposed for
            explicit A/B testing.

        allow_cluster_touch_entry : if True, a multi-swing cluster
            (>= 2 independent swing scales agreeing, see detect_clusters)
            within CONFLUENCE_TOLERANCE_PIPS of curr_price counts as a
            confirmed trigger ('cluster_touch') even when
            _detect_trigger_candle found no qualifying candle. This is a
            genuine loosening (raises frequency) — justified only because
            a cluster is independently-corroborated structure, not a bare
            zone touch. Combine with require_engulfing_only freely; a
            cluster_touch trigger is exempt from that restriction (it's
            not a candle-shape claim, so "engulfing-only" doesn't apply
            to it).

        min_rr_dynamic : if True, the effective min_rr required is
            LOWERED (by 0.5, floored at 1.0) when current ATR is below
            80% of its trailing average, i.e. in demonstrably calmer
            conditions where a lower R:R target is plausibly reachable
            without added noise-driven stop-outs. Never raises the
            requirement above self.min_rr. Default False (prior fixed
            min_rr behavior unchanged).

        max_swing_age_bars : if > 0, a swing whose extreme (the later of
            high_idx/low_idx) is more than this many CLOSED bars old is
            rejected outright in analyze() (returns _empty_result — no
            signal at all, not just a WAIT candidate) rather than having
            a Fib grid drawn from it. Rationale: a 50+ bar old swing's
            structural relevance decays as newer price action overwrites
            it. 0 (default) disables — matches prior behavior exactly.

        require_volume_confirmation / volume_multiple : if True, a
            trigger-candle pattern is only confirmed when the trigger
            bar's volume is >= volume_multiple x the mean of the prior 5
            closed bars' volume — guards against low-participation
            fakeout candles. No-op (never blocks) if `df` has no 'volume'
            column, since not every data source supplies real volume for
            FX (MT5 tick_volume is a proxy, not true volume — treat this
            as a heuristic filter, not a hard truth).

        --- Liquidity engine integration (Day 63) ---
        Wired the same way as htf_ctx: this module does NOT import or run
        the liquidity engine itself (analysis.liquidity_engine.LiquidityEngine)
        — the caller runs it separately (it needs its own OHLCV+ATR+
        DatetimeIndex prep) and passes the result into analyze(liquidity_ctx=...).
        Keeps this module decoupled from that pipeline's own dependencies
        (session_analysis, stop_hunt_detector, fvg_detector, etc.) — the
        same reason htf_ctx works this way rather than this class computing
        HTF trend itself.

        require_liquidity_alignment : if True, a candidate BUY/SELL is
            downgraded to WAIT only when liquidity_ctx['bias'] ACTIVELY
            opposes it (e.g. candidate BUY but liquidity bias BEARISH) —
            NEUTRAL liquidity (no strong read either way) never blocks a
            trade. Tested on EURUSD H1, 5yr, existing validated trade set
            (n=32): dropped exactly 1 conflicting trade (a loser), n=31,
            WR 34.4%->35.5%, PF 1.22->1.28. That is a one-trade difference —
            not statistically significant on its own, and the underlying
            reason is structural, not noise: the liquidity engine's score
            gate (MIN_LIQUIDITY_SCORE=55 for a non-NEUTRAL bias) means it
            has an opinion at only ~15% of fib entries, so this can only
            ever be a rare, low-frequency veto, never a primary edge
            source. Default False — enable it if you want the (currently
            unproven-at-scale but directionally clean, never-hurts-in-
            testing-so-far) extra veto; re-validate as more liquidity_ctx-
            tagged trade history accumulates.

        min_liquidity_score : if >0, additionally requires liquidity_ctx
            ['score'] >= this value (not just bias agreement) before the
            alignment check above can pass — i.e. a stricter version of
            require_liquidity_alignment. NOT recommended as currently
            tested: raising this toward the engine's own MIN_LIQUIDITY_SCORE
            (55) cut the EURUSD H1 sample from 32 trades to 4 — far too few
            to trust a win rate from. Left at 0 (off) by default; only
            raise this once you have a large enough liquidity_ctx-tagged
            sample to validate it independently, the same rule this whole
            review has followed for every other gate.
        """
        if level_set not in LEVEL_SET_PRESETS:
            raise ValueError(f"Unknown level_set '{level_set}'. Valid: {sorted(LEVEL_SET_PRESETS)}")

        self.timeframe    = timeframe
        self.symbol       = symbol
        self.swing_window = TF_SWING_WINDOW.get(timeframe, DEFAULT_SWING_WINDOW)
        self.require_trigger_candle = require_trigger_candle
        self.min_rr       = min_rr
        self.level_set     = level_set
        self.retracement_levels = LEVEL_SET_PRESETS[level_set]['retracement']
        self.extension_levels   = LEVEL_SET_PRESETS[level_set]['extension']
        self.require_htf_alignment = require_htf_alignment

        # Day 42 additions — see __init__ docstring for full rationale.
        self.min_confluence_strength    = min_confluence_strength
        self.require_engulfing_only     = require_engulfing_only
        self.require_trend_ma           = require_trend_ma
        self.trend_ma_period            = trend_ma_period
        self.min_adx                    = min_adx
        self.max_atr_multiple           = max_atr_multiple
        self.allow_golden_zone          = allow_golden_zone
        self.require_cluster_for_golden = require_cluster_for_golden
        self.allow_cluster_touch_entry  = allow_cluster_touch_entry
        self.min_rr_dynamic             = min_rr_dynamic
        self.max_swing_age_bars         = max_swing_age_bars
        self.require_volume_confirmation = require_volume_confirmation
        self.volume_multiple            = volume_multiple
        self.require_liquidity_alignment = require_liquidity_alignment
        self.min_liquidity_score        = min_liquidity_score

        log.info(f"FibonacciEngine ready | TF={timeframe} | symbol={symbol} | swing_window={self.swing_window} "
                 f"| require_trigger_candle={require_trigger_candle} | min_rr={min_rr} "
                 f"| level_set={level_set} | require_htf_alignment={require_htf_alignment} "
                 f"| min_confluence_strength={min_confluence_strength} | require_engulfing_only={require_engulfing_only} "
                 f"| require_trend_ma={require_trend_ma}({trend_ma_period}) | min_adx={min_adx} "
                 f"| max_atr_multiple={max_atr_multiple} | allow_golden_zone={allow_golden_zone} "
                 f"| allow_cluster_touch_entry={allow_cluster_touch_entry} | min_rr_dynamic={min_rr_dynamic} "
                 f"| max_swing_age_bars={max_swing_age_bars} | require_volume_confirmation={require_volume_confirmation} "
                 f"| require_liquidity_alignment={require_liquidity_alignment} | min_liquidity_score={min_liquidity_score}")

    # ═══════════════════════════════════════════════════════════
    # MAIN ANALYSIS METHOD
    # ═══════════════════════════════════════════════════════════

    def analyze(
        self,
        df:       pd.DataFrame,
        sr_ctx:   dict = None,
        ind_ctx:  dict = None,
        htf_ctx:  dict = None,
        indicator_levels: dict = None,
        liquidity_ctx: dict = None,
    ) -> dict:
        """
        Full Fibonacci analysis pipeline।

        Steps:
          1. Swing detect
          2. Retracement levels
          3. Extension targets (+ 3b: ABC expansion)
          4. Current price position
          5. Confluence zones (+ 5b: multi-swing cluster detection)
          6. Signal

        IMPORTANT — closed-bar contract: `df` must contain only fully CLOSED
        candles. `curr_price` is taken from `df['close'].iloc[-1]`; if the
        caller's fetch includes the still-forming bar (common with MT5's
        `copy_rates_from_pos(symbol, tf, 0, n)`, which returns the live bar
        at index 0), every level/zone/signal computed here will repaint as
        that bar's close keeps moving. Drop the forming bar before calling
        this (e.g. `df.iloc[:-1]` when the fetch includes it).

        htf_ctx : optional higher-timeframe trend context, see
            _generate_signal's docstring. Purely additive — omit it and
            behavior is identical to before this parameter existed.

        indicator_levels : optional {label: price} dict of extra
            price-based levels to check for confluence beyond S/R —
            e.g. {'EMA50': 1.0842, 'AnchoredVWAP': 1.0851}. Several of
            the shared references treat a moving average or an anchored
            VWAP coinciding with the Fib zone the same way they treat
            S/R confluence; this lets the caller supply those without
            the engine having to compute EMA/VWAP itself (that belongs
            in the indicators module, not here).

        liquidity_ctx : optional output of
            analysis.liquidity_engine.LiquidityEngine.analyze() (or any
            dict with the same 'bias'/'score' keys), computed by the
            CALLER on the same bar this call sees — this module never
            imports or runs that engine itself. Purely additive: omit it
            and behavior is identical to before this parameter existed;
            it only becomes a gate when require_liquidity_alignment=True
            (see __init__ docstring for the evidence behind that default).
        """
        if len(df) < self.swing_window * 3:
            return self._empty_result("Insufficient data")

        # Step 1: Auto swing detection
        swings = self.find_swing_points(df)
        if not swings['valid']:
            return self._empty_result("No significant swing detected")

        swing_high = swings['high']
        swing_low  = swings['low']
        trend      = swings['trend']

        # Step 1b: Swing-age gate (Day 42 — opt-in, see __init__ docstring).
        # Bars since the swing's later extreme formed. Computed unconditionally
        # (cheap) so it's available in the result dict for diagnostics even
        # when max_swing_age_bars=0 leaves the gate itself disabled.
        swing_age_bars = (len(df) - 1) - max(swings['high_idx'], swings['low_idx'])
        if self.max_swing_age_bars > 0 and swing_age_bars > self.max_swing_age_bars:
            return self._empty_result(
                f"Swing too old ({swing_age_bars} bars > max_swing_age_bars={self.max_swing_age_bars}) "
                f"— structure likely decayed"
            )

        # Step 2: Retracement levels
        retracements = self.calculate_retracement(swing_high, swing_low, trend)

        # Step 2b: Flip Zones (Phase 3) — support/resistance polarity flip.
        # Needs df (closed-bar history since the swing formed) and swings
        # (for the B-point index), so it's computed right after the
        # retracement grid exists and before anything downstream uses it.
        flip_zones = self.detect_flip_zones(df, swings, retracements)

        # Step 3: Extension targets (2-point: A implicitly = B retest)
        extensions = self.calculate_extension(swing_high, swing_low, trend)

        # Step 3b: ABC Expansion (3-point, Phase 1) — only valid once a
        # real correction (point C) has formed after the swing extreme (B).
        # Returns {'valid': False, ...} rather than raising when C hasn't
        # formed yet or has invalidated the swing (broken past A) — this
        # is expected/normal on a fresh swing, not an error condition.
        expansion = self.calculate_expansion(swings)

        # Step 4: Current price position
        curr_price = float(df['close'].iloc[-1])
        position   = self._price_position(curr_price, retracements, trend)

        # Step 5: Confluence with S/R (+ optional EMA/VWAP/etc. levels)
        confluence_zones = self.find_confluence(
            retracements, extensions, sr_ctx, curr_price,
            indicator_levels=indicator_levels, expansion=expansion,
        )

        # Step 5b: Multi-Swing + Cluster Detection (Phase 2) — overlay
        # Fib grids from major/minor/internal swing scales and report
        # where INDEPENDENT swings agree on a price zone. This is a
        # different signal from Step 5's confluence: that checks one
        # swing's grid against the chart's own S/R; this checks whether
        # multiple swing scales' own grids agree with EACH OTHER, which
        # single-swing confluence structurally cannot see.
        multi_swings = self.find_multi_swings(df)
        clusters     = self.detect_clusters(multi_swings)

        # Step 5c: Trend MA / average-ATR context (Day 42) — only computed
        # when a gate that actually needs them is enabled, to avoid paying
        # for a rolling mean over the whole window on every call by default.
        ma = None
        if self.require_trend_ma and len(df) >= self.trend_ma_period:
            ma = float(df['close'].rolling(self.trend_ma_period).mean().iloc[-1])

        atr_avg = None
        if self.max_atr_multiple > 0 or self.min_rr_dynamic:
            atr_avg = self._calc_atr_avg(df)

        # Step 6: Signal
        signal = self._generate_signal(
            curr_price, position, confluence_zones, trend, ind_ctx,
            df=df, extensions=extensions, retracements=retracements,
            htf_ctx=htf_ctx, flip_zones=flip_zones,
            ma=ma, atr_avg=atr_avg, clusters=clusters,
            swing_age_bars=swing_age_bars, liquidity_ctx=liquidity_ctx,
        )

        # Step 7: Failure detection
        failure_risk = self._detect_failure_risk(df, position, ind_ctx)

        result = {
            'swing':           swings,
            'swing_high':      swing_high,
            'swing_low':       swing_low,
            'trend':           trend,
            'range_pips':      round(price_to_pips(abs(swing_high - swing_low), self.symbol), 1),
            'swing_age_bars':  swing_age_bars,
            'retracements':    retracements,
            'flip_zones':      flip_zones,
            'extensions':      extensions,
            'expansion':       expansion,
            'curr_price':      curr_price,
            'position':        position,
            'confluence':      confluence_zones,
            'multi_swings':    multi_swings,
            'clusters':        clusters,
            'signal':          signal,
            'failure_risk':    failure_risk,
        }

        log.info(
            f"Fib Analysis | Swing: {swing_low:.5f}→{swing_high:.5f} "
            f"| Trend: {trend} | Position: {position.get('nearest_level')} "
            f"| Signal: {signal.get('bias')}"
        )
        return result

    # ═══════════════════════════════════════════════════════════
    # STEP 1: AUTO SWING DETECTION
    # ═══════════════════════════════════════════════════════════

    def find_swing_points(self, df: pd.DataFrame) -> dict:
        """
        Auto swing high / low detect করো।

        Dynamic window — timeframe অনুযায়ী আলাদা।
        Significant swing = ATR-এর কমপক্ষে 2x range।

        Returns both the most recent significant swing pair.
        """
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        n      = len(df)
        w      = self.swing_window

        # ATR for minimum size filter
        atr = self._calc_atr(df)

        # Find all local swing highs and lows
        swing_highs = []
        swing_lows  = []

        for i in range(w, n - w):
            # Swing High: highest point in window
            if highs[i] == max(highs[i - w: i + w + 1]):
                swing_highs.append((i, highs[i]))

            # Swing Low: lowest point in window
            if lows[i] == min(lows[i - w: i + w + 1]):
                swing_lows.append((i, lows[i]))

        if not swing_highs or not swing_lows:
            return {'valid': False, 'reason': 'No swings found'}

        # Pick most recent significant swing pair
        # Recent: within last 50% of data
        recent_cutoff = n // 2

        recent_highs = [(i, v) for i, v in swing_highs if i >= recent_cutoff]
        recent_lows  = [(i, v) for i, v in swing_lows  if i >= recent_cutoff]

        if not recent_highs:
            recent_highs = swing_highs
        if not recent_lows:
            recent_lows = swing_lows

        best_high_idx, best_high = max(recent_highs, key=lambda x: x[1])
        best_low_idx,  best_low  = min(recent_lows,  key=lambda x: x[1])

        # Minimum range: 2× ATR
        swing_range = best_high - best_low
        if swing_range < atr * 2:
            return {
                'valid':  False,
                'reason': f'Swing range {swing_range:.5f} too small (ATR={atr:.5f})'
            }

        # Trend: which came first?
        if best_low_idx < best_high_idx:
            trend = 'BULLISH'   # Low first → price moved up
        else:
            trend = 'BEARISH'   # High first → price moved down

        return {
            'valid':          True,
            'high':           round(best_high, 5),
            'low':            round(best_low, 5),
            'high_idx':       best_high_idx,
            'low_idx':        best_low_idx,
            'trend':          trend,
            'range':          round(swing_range, 5),
            'range_pips':     round(price_to_pips(swing_range, self.symbol), 1),
            'atr':            round(atr, 5),
            'all_highs':      [(i, round(v, 5)) for i, v in swing_highs],
            'all_lows':       [(i, round(v, 5)) for i, v in swing_lows],
        }

    # ═══════════════════════════════════════════════════════════
    # STEP 2: RETRACEMENT LEVELS
    # ═══════════════════════════════════════════════════════════

    def calculate_retracement(
        self,
        high:  float,
        low:   float,
        trend: str = 'BULLISH',
    ) -> dict:
        """
        Fibonacci Retracement levels calculate করো।

        Bullish retracement: high থেকে নিচে (pullback levels)
        Bearish retracement: low থেকে উপরে (bounce levels)

        Formula (Bullish):
            level = high - (high - low) × ratio
        Formula (Bearish):
            level = low + (high - low) × ratio
        """
        diff   = high - low
        levels = {}

        for ratio in self.retracement_levels:
            label = f"{ratio * 100:.1f}%"
            if trend == 'BULLISH':
                # Price fell from high — retracement levels going down
                price = high - diff * ratio
            else:
                # Price rose from low — retracement levels going up
                price = low + diff * ratio
            levels[label] = round(price, 5)

        return {
            'trend':   trend,
            'high':    high,
            'low':     low,
            'diff':    round(diff, 5),
            'levels':  levels,
            # Key levels shortcut — .get() because the 'conservative'
            # level_set preset drops 23.6%, so it won't always be present.
            '23.6':    levels.get('23.6%'),
            '38.2':    levels.get('38.2%'),
            '50.0':    levels.get('50.0%'),
            '61.8':    levels.get('61.8%'),
            '78.6':    levels.get('78.6%'),
        }

    # ═══════════════════════════════════════════════════════════
    # STEP 2b: FLIP ZONE DETECTION (Phase 3)
    # ═══════════════════════════════════════════════════════════

    def detect_flip_zones(self, df: pd.DataFrame, swings: dict, retracements: dict) -> list[dict]:
        """
        Fibonacci Flip Zone: support/resistance polarity flip.

        Mechanism (same polarity-flip logic already standard for plain
        S/R levels, applied here to Fib retracement levels specifically):
        once price CLOSES beyond a retracement level in the direction
        that breaks its original role, that level's role flips —
        BULLISH: level was support (price retracing down onto it from
            above) -> a close BELOW it breaks that support -> the level
            becomes RESISTANCE on any later retest from below.
        BEARISH: level was resistance (price retracing up onto it from
            below) -> a close ABOVE it breaks that resistance -> the
            level becomes SUPPORT on any later retest from above.

        Only checked on bars from the swing's B-point onward — the grid
        doesn't exist before B forms, so price sitting on either side of
        a level before that has no meaning here. 0%/100% (the swing
        extremes themselves) are skipped: a "break" of those means a NEW
        swing has formed, not a flip of an interior level.

        A level only counts as broken on a CLOSE beyond it, not a wick —
        same closed-bar-only philosophy as the rest of this engine (a
        wick poking through and closing back inside is the existing
        liquidity-sweep pattern, a different feature).

        status:
          'FLIPPED'   — broken, and price is STILL on the broken side as
                        of the latest closed bar (the flip is currently
                        active / tradeable on a retest).
          'RECLAIMED' — broke at some point but has since closed back on
                        the original side (a failed break / whipsaw) —
                        reported as context, original role still holds;
                        _generate_signal only acts on 'FLIPPED'.
        """
        if df is None or len(df) < 2 or not swings.get('valid'):
            return []

        trend  = retracements.get('trend')
        levels = retracements.get('levels', {})
        if not levels:
            return []

        b_idx = max(swings['high_idx'], swings['low_idx'])
        closes = df['close'].values[b_idx:]
        if len(closes) < 1:
            return []

        curr_price = float(closes[-1])
        flips = []

        for label, price in levels.items():
            if label in ('0.0%', '100.0%'):
                continue

            if trend == 'BULLISH':
                broken_mask = closes < price
                original_role, flipped_role = 'SUPPORT', 'RESISTANCE'
            else:
                broken_mask = closes > price
                original_role, flipped_role = 'RESISTANCE', 'SUPPORT'

            if not broken_mask.any():
                continue  # never broken — original role holds, nothing to report

            currently_broken = bool(broken_mask[-1])
            status = 'FLIPPED' if currently_broken else 'RECLAIMED'
            role   = flipped_role if currently_broken else original_role

            dist_pips = round(price_to_pips(abs(curr_price - price), self.symbol), 1)
            flips.append({
                'label':         label,
                'price':         price,
                'status':        status,
                'role':          role,
                'original_role': original_role,
                'near_price':    dist_pips <= CONFLUENCE_TOLERANCE_PIPS,
                'dist_pips':     dist_pips,
            })

        flips.sort(key=lambda f: f['dist_pips'])
        return flips

    # ═══════════════════════════════════════════════════════════
    # STEP 3: EXTENSION LEVELS
    # ═══════════════════════════════════════════════════════════

    def calculate_extension(
        self,
        high:  float,
        low:   float,
        trend: str = 'BULLISH',
    ) -> dict:
        """
        Fibonacci Extension — target levels বের করো।

        Bullish extension: swing low থেকে উপরে
        Bearish extension: swing high থেকে নিচে

        Formula (Bullish):
            level = low + diff × ratio
        Formula (Bearish):
            level = high - diff × ratio
        """
        diff   = high - low
        levels = {}

        for ratio in self.extension_levels:
            label = f"{ratio * 100:.1f}%"
            if trend == 'BULLISH':
                price = low + diff * ratio
            else:
                price = high - diff * ratio
            levels[label] = round(price, 5)

        # Key targets shortcut. The 'conservative' level_set preset uses
        # 127.2/141.4/161.8% instead of 127.2/161.8/261.8%, so TP1/TP2/TP3
        # are resolved by nearest-available-ratio rather than hardcoded
        # keys, falling back gracefully if a level isn't in the grid.
        def _nearest(target_ratio):
            available = self.extension_levels
            closest = min(available, key=lambda r: abs(r - target_ratio))
            return levels.get(f"{closest * 100:.1f}%")

        return {
            'trend':   trend,
            'levels':  levels,
            'TP1':     _nearest(1.272),
            'TP2':     _nearest(1.618),
            'TP3':     _nearest(2.618),
        }

    # ═══════════════════════════════════════════════════════════
    # STEP 3b: ABC EXPANSION (3-point, Phase 1)
    # ═══════════════════════════════════════════════════════════

    def calculate_expansion(self, swings: dict) -> dict:
        """
        ABC 3-point Fibonacci Expansion.

        A = swing origin, B = swing extreme (both already found by
        find_swing_points — this is the same A/B used for retracement).
        C = the correction extreme that formed AFTER B, in the opposite
        direction — i.e. the pullback low (bullish) / pullback high
        (bearish) price is currently correcting from.

        Expansion projects the AB leg length forward from C:
            Bullish : level = C + (B - A) × ratio
            Bearish : level = C - (A - B) × ratio

        This is mechanically different from calculate_extension(), which
        implicitly assumes the next leg starts at B (full retrace to B
        before continuation). Expansion uses the REAL correction point,
        so targets move as the pullback develops — this is what most
        traders mean by "Fibonacci Expansion" / TP projection tooling.

        Returns {'valid': False, 'reason': ...} — not a raised exception —
        when point C hasn't formed yet (swing too fresh) or when the
        correction has broken past A (swing invalidated, C is no longer
        a valid ABC point). Both are normal/expected states, not errors;
        callers should check ['valid'] before reading levels, same
        pattern as find_swing_points().
        """
        if not swings.get('valid'):
            return {'valid': False, 'reason': 'No base A-B swing'}

        trend       = swings['trend']
        high_idx    = swings['high_idx']
        low_idx     = swings['low_idx']

        if trend == 'BULLISH':
            a_price, b_price = swings['low'], swings['high']
            b_idx = high_idx
            c_candidates = [(i, v) for i, v in swings.get('all_lows', []) if i > b_idx]
        else:
            a_price, b_price = swings['high'], swings['low']
            b_idx = low_idx
            c_candidates = [(i, v) for i, v in swings.get('all_highs', []) if i > b_idx]

        if not c_candidates:
            return {'valid': False, 'reason': 'Point C not formed yet (no correction after B)'}

        # Most recent correction swing = current point C.
        c_idx, c_price = c_candidates[-1]

        # Invalidation check: if the correction has traded past A, this is
        # no longer a valid ABC structure (the original swing is broken) —
        # report invalid rather than silently projecting from a bad C.
        if trend == 'BULLISH' and c_price <= a_price:
            return {'valid': False, 'reason': 'Point C broke past A — ABC structure invalidated'}
        if trend == 'BEARISH' and c_price >= a_price:
            return {'valid': False, 'reason': 'Point C broke past A — ABC structure invalidated'}

        ab_range = abs(b_price - a_price)
        levels   = {}
        for ratio in EXPANSION_RATIOS:
            label = f"{ratio * 100:.1f}%"
            if trend == 'BULLISH':
                price = c_price + ab_range * ratio
            else:
                price = c_price - ab_range * ratio
            levels[label] = round(price, 5)

        def _nearest(target_ratio):
            closest = min(EXPANSION_RATIOS, key=lambda r: abs(r - target_ratio))
            return levels.get(f"{closest * 100:.1f}%")

        return {
            'valid':      True,
            'trend':      trend,
            'a':          round(a_price, 5),
            'b':          round(b_price, 5),
            'c':          round(c_price, 5),
            'c_idx':      c_idx,
            'ab_range':   round(ab_range, 5),
            'levels':     levels,
            'TP1':        _nearest(0.618),
            'TP2':        _nearest(1.0),
            'TP3':        _nearest(1.618),
        }

    # ═══════════════════════════════════════════════════════════
    # STEP 4: CONFLUENCE ANALYSIS
    # ═══════════════════════════════════════════════════════════

    def find_confluence(
        self,
        retracements: dict,
        extensions:   dict,
        sr_ctx:       dict = None,
        curr_price:   float = None,
        indicator_levels: dict = None,
        expansion:    dict = None,
    ) -> list[dict]:
        """
        Fib levels + S/R (+ optional other indicator) levels কাছাকাছি হলে
        → Confluence zone।

        Stronger confluence = more reasons → higher strength score.

        indicator_levels : optional {label: price}, e.g. moving-average or
            anchored-VWAP levels — several of the shared references treat
            these the same way as S/R for confluence purposes ("the golden
            zone and the anchored VWAP coincide... this area should act as
            a strong level of support"). Merged in alongside sr_ctx; caller
            computes the actual EMA/VWAP value, this just checks proximity.

        expansion : optional output of calculate_expansion(). Only merged
            in when expansion['valid'] is True (point C hasn't formed on
            every swing) — an ABC target lining up with a retracement/S/R
            level is a genuinely independent confirmation (different swing
            data: A/B/C vs just A/B), so it's treated the same as an
            extension target rather than a separate category.
        """
        confluence_zones = []
        tolerance        = CONFLUENCE_TOLERANCE_PIPS * pip_size(self.symbol)

        # All Fib levels (retracement + extension + ABC expansion)
        all_fib = {}
        for label, price in retracements['levels'].items():
            all_fib[f"Fib {label}"] = price
        for label, price in extensions['levels'].items():
            all_fib[f"Ext {label}"] = price
        if expansion and expansion.get('valid'):
            for label, price in expansion['levels'].items():
                all_fib[f"ABC {label}"] = price

        # S/R levels from context
        sr_levels = {}
        if sr_ctx:
            if sr_ctx.get('nearest_support'):
                sr_levels['Support']   = sr_ctx['nearest_support']
            if sr_ctx.get('nearest_resistance'):
                sr_levels['Resistance'] = sr_ctx['nearest_resistance']
            if sr_ctx.get('pivot'):
                sr_levels['Pivot']     = sr_ctx['pivot']
            if sr_ctx.get('R1'):
                sr_levels['R1']        = sr_ctx['R1']
            if sr_ctx.get('S1'):
                sr_levels['S1']        = sr_ctx['S1']

        # Extra caller-supplied levels (EMA, anchored VWAP, ...)
        if indicator_levels:
            for label, price in indicator_levels.items():
                if price is not None:
                    sr_levels[label] = price

        # Find confluences
        for fib_name, fib_price in all_fib.items():
            reasons  = [fib_name]
            strength = self._fib_base_strength(fib_name)

            # Check S/R proximity
            for sr_name, sr_price in sr_levels.items():
                if abs(fib_price - sr_price) <= tolerance:
                    reasons.append(sr_name)
                    strength += 20

            # Check proximity to current price
            near_curr = False
            if curr_price:
                dist_pips = price_to_pips(abs(fib_price - curr_price), self.symbol)
                if dist_pips <= CONFLUENCE_TOLERANCE_PIPS:
                    near_curr = True
                    strength += 10

            # Only report if multiple reasons OR very strong level
            if len(reasons) >= 2 or strength >= 75:
                trend    = retracements['trend']
                zone_type = self._zone_type(fib_name, trend)

                confluence_zones.append({
                    'price':      fib_price,
                    'reasons':    reasons,
                    'strength':   min(99, strength),
                    'zone_type':  zone_type,
                    'near_price': near_curr,
                    'dist_pips':  round(price_to_pips(abs(fib_price - curr_price), self.symbol), 1) if curr_price else None,
                    'note':       (
                        f"{zone_type} at {fib_price:.5f} — "
                        f"{' + '.join(reasons)} (strength: {strength})"
                    ),
                })

        # Sort by strength descending
        confluence_zones.sort(key=lambda z: z['strength'], reverse=True)
        return confluence_zones

    def _fib_base_strength(self, fib_name: str) -> int:
        """
        Fibonacci level-এর base strength (কোনটা বেশি react করে)।

        FIX (institutional review): this used to match via `key in
        fib_name` substring checks against string keys. '61.8' is a
        substring of '161.8' and '261.8' (e.g. "261.8"[1:5] == "61.8"),
        so those levels were silently scored as golden-ratio strength (80)
        instead of their own mapped values (75 / 55) — checked first in
        dict-iteration order, the bug always won silently. Confluence
        strength — and therefore signal confidence — was inflated for
        every 161.8%/261.8% extension or ABC-expansion touch. Fixed by
        parsing the exact numeric ratio out of the label instead of
        substring matching.
        """
        strength_map = {
            61.8:  80,   # Golden ratio — most important
            50.0:  70,
            38.2:  65,
            78.6:  60,
            23.6:  50,
            127.2: 65,   # Extension targets
            161.8: 75,   # Golden ratio extension
            261.8: 55,
            100.0: 55,
            0.0:   45,
        }
        try:
            ratio = float(fib_name.split()[-1].rstrip('%'))
        except (ValueError, IndexError):
            return 40
        return strength_map.get(ratio, 40)

    def _zone_type(self, fib_name: str, trend: str) -> str:
        """Fib level + trend দিয়ে zone type বলো"""
        # ABC expansion levels reuse some retracement-looking ratios (e.g.
        # 61.8%) but are targets, not entry zones — classify by prefix
        # before falling through to the retracement/extension label match.
        if fib_name.startswith('ABC '):
            return 'TARGET_ZONE'

        retracement_labels = ['23.6', '38.2', '50.0', '61.8', '78.6']
        extension_labels   = ['127.2', '161.8', '261.8']

        is_retracement = any(l in fib_name for l in retracement_labels)
        is_extension   = any(l in fib_name for l in extension_labels)

        if is_retracement:
            return 'BUY_ZONE' if trend == 'BULLISH' else 'SELL_ZONE'
        if is_extension:
            return 'TARGET_ZONE'
        return 'ZONE'

    # ═══════════════════════════════════════════════════════════
    # STEP 5b: MULTI-SWING + CLUSTER DETECTION (Phase 2)
    # ═══════════════════════════════════════════════════════════

    def find_multi_swings(self, df: pd.DataFrame) -> dict:
        """
        Same underlying pivot set as find_swing_points() (no new
        pivot-detection logic — reuses its 'all_highs'/'all_lows'), but
        picks THREE different high/low PAIRS out of that pivot set
        instead of one, at different structural scales:

          'minor'    : exactly what find_swing_points() already returns
                       (the swing retracement/extension/ABC all use) —
                       most recent SIGNIFICANT swing, restricted to the
                       recent half of the lookback.
          'major'    : the single largest-range (high, low) pair across
                       the ENTIRE lookback window, not just the recent
                       half — the dominant structure the chart is inside.
          'internal' : the most recently COMPLETED leg (the last two
                       pivots, whichever types they are) — current
                       micro-structure, often smaller than 'minor'.

        Each entry has the same shape find_swing_points() returns (or
        {'valid': False, 'reason': ...}), so calculate_retracement() /
        calculate_extension() work on any of them unchanged.

        Caveat (state honestly, don't oversell): 'major' is O(n²) over
        the pivot lists (all highs × all lows) to find the true global
        max-range pair. Pivot counts are small (one pivot per swing_window
        bars) so this is cheap in practice, but it's a brute-force
        argmax, not a smarter O(n log n) approach — fine at typical
        200-500 bar lookbacks, would need revisiting for much longer ones.
        """
        base = self.find_swing_points(df)
        if not base['valid']:
            return {
                'minor':    {'valid': False, 'reason': base.get('reason', 'invalid')},
                'major':    {'valid': False, 'reason': 'base swing invalid'},
                'internal': {'valid': False, 'reason': 'base swing invalid'},
            }

        all_highs = base['all_highs']
        all_lows  = base['all_lows']
        atr       = base['atr']
        swings    = {'minor': base}

        # MAJOR: largest-range pair over the FULL pivot set (find_swing_points
        # restricts to the recent half when possible; this deliberately does not).
        if all_highs and all_lows:
            hi, hv = max(all_highs, key=lambda x: x[1])
            li, lv = min(all_lows,  key=lambda x: x[1])
            major_range = hv - lv
            if major_range >= atr * 2:
                trend = 'BULLISH' if li < hi else 'BEARISH'
                swings['major'] = {
                    'valid': True, 'high': round(hv, 5), 'low': round(lv, 5),
                    'high_idx': hi, 'low_idx': li, 'trend': trend,
                    'range': round(major_range, 5),
                    'range_pips': round(price_to_pips(major_range, self.symbol), 1),
                    'atr': atr, 'all_highs': all_highs, 'all_lows': all_lows,
                }
            else:
                swings['major'] = {'valid': False, 'reason': 'No major swing meeting 2×ATR filter'}
        else:
            swings['major'] = {'valid': False, 'reason': 'Insufficient pivots'}

        # INTERNAL: most recently completed leg — last two pivots of
        # DIFFERENT types (scanning backward; the raw local-max/min pivot
        # list isn't guaranteed to strictly alternate high/low/high/low).
        merged = sorted(
            [(i, v, 'high') for i, v in all_highs] + [(i, v, 'low') for i, v in all_lows],
            key=lambda x: x[0],
        )
        internal_pair = None
        for j in range(len(merged) - 1, 0, -1):
            a, b = merged[j], merged[j - 1]
            if a[2] != b[2]:
                internal_pair = (a, b)
                break

        if internal_pair:
            a, b = internal_pair
            hi, hv = (a[0], a[1]) if a[2] == 'high' else (b[0], b[1])
            li, lv = (a[0], a[1]) if a[2] == 'low'  else (b[0], b[1])
            internal_range = hv - lv
            if internal_range >= atr * 2:
                trend = 'BULLISH' if li < hi else 'BEARISH'
                swings['internal'] = {
                    'valid': True, 'high': round(hv, 5), 'low': round(lv, 5),
                    'high_idx': hi, 'low_idx': li, 'trend': trend,
                    'range': round(internal_range, 5),
                    'range_pips': round(price_to_pips(internal_range, self.symbol), 1),
                    'atr': atr, 'all_highs': all_highs, 'all_lows': all_lows,
                }
            else:
                swings['internal'] = {'valid': False, 'reason': 'Latest leg too small (< 2×ATR)'}
        else:
            swings['internal'] = {'valid': False, 'reason': 'No alternating pivot pair found'}

        return swings

    def detect_clusters(self, multi_swings: dict) -> list[dict]:
        """
        Overlay retracement + extension grids from every valid swing in
        multi_swings and report price zones where levels from >= 2
        DISTINCT swings land within tolerance of each other.

        This is deliberately NOT a single-swing self-check — one swing's
        own 50%/61.8%/78.6% levels are naturally spread across the range
        by construction, so requiring >=2 distinct swings is what makes a
        "cluster" mean something (independent structures agreeing),
        rather than just re-reporting one grid's own levels back.

        Clustering method: sort all (price, swing, level) points, then
        single-linkage chain — merge a point into the current cluster if
        it's within tolerance of the PREVIOUS point. This is a known
        simplification (a dense run of points can chain into one cluster
        even if the first and last are far apart) — acceptable at typical
        Fib-grid point counts (~20-30 points across 3 swings) where levels
        are naturally sparse, but worth knowing if this is ever fed a much
        denser point set.

        'strength' here is the same kind of hand-set heuristic already
        used by find_confluence() (not a new methodology) — weighted by
        distinct-swing count first (the thing that actually matters) and
        level count second. This is explicitly NOT the data-driven
        Matrix/Confidence score from the roadmap (Phase 5) — that requires
        historical bounce-statistics to fit real weights; this is a
        same-bar structural observation only.
        """
        tolerance = CONFLUENCE_TOLERANCE_PIPS * pip_size(self.symbol)

        points = []
        for swing_label, swing in multi_swings.items():
            if not swing.get('valid'):
                continue
            retr = self.calculate_retracement(swing['high'], swing['low'], swing['trend'])
            ext  = self.calculate_extension(swing['high'], swing['low'], swing['trend'])
            for lvl_label, price in retr['levels'].items():
                points.append((price, swing_label, f"Fib {lvl_label}"))
            for lvl_label, price in ext['levels'].items():
                points.append((price, swing_label, f"Ext {lvl_label}"))

        if len(points) < 2:
            return []

        points.sort(key=lambda p: p[0])

        raw_clusters = []
        current = [points[0]]
        for p in points[1:]:
            if p[0] - current[-1][0] <= tolerance:
                current.append(p)
            else:
                raw_clusters.append(current)
                current = [p]
        raw_clusters.append(current)

        results = []
        for c in raw_clusters:
            distinct_swings = sorted(set(m[1] for m in c))
            if len(distinct_swings) < 2:
                continue  # single-swing self-overlap — not a real cluster
            avg_price = sum(m[0] for m in c) / len(c)
            results.append({
                'price':        round(avg_price, 5),
                'swing_count':  len(distinct_swings),
                'level_count':  len(c),
                'swings':       distinct_swings,
                'members':      [f"{m[1]}:{m[2]}" for m in c],
                'strength':     min(99, 50 + 20 * len(distinct_swings) + 5 * len(c)),
            })

        results.sort(key=lambda z: z['strength'], reverse=True)
        return results

    # ═══════════════════════════════════════════════════════════
    # STEP 5: PRICE POSITION
    # ═══════════════════════════════════════════════════════════

    def _price_position(
        self,
        curr_price:   float,
        retracements: dict,
        trend:        str,
    ) -> dict:
        """
        Current price কোন Fib level-এর কাছে আছে বলো।
        """
        levels = retracements['levels']
        nearest_label = None
        nearest_price = None
        nearest_dist  = float('inf')

        for label, price in levels.items():
            dist = abs(curr_price - price)
            if dist < nearest_dist:
                nearest_dist  = dist
                nearest_label = label
                nearest_price = price

        nearest_pips = round(price_to_pips(nearest_dist, self.symbol), 1)

        # Which zone is price in?
        high = retracements['high']
        low  = retracements['low']
        diff = high - low

        if diff == 0:
            ratio = 0.5
        else:
            if trend == 'BULLISH':
                ratio = (high - curr_price) / diff
            else:
                ratio = (curr_price - low) / diff

        # Zone categorization
        if ratio <= 0.0:
            zone = 'ABOVE_HIGH'
        elif ratio <= 0.236:
            zone = 'SHALLOW_RETRACEMENT'
        elif ratio <= 0.382:
            zone = 'MINOR_RETRACEMENT'
        elif ratio <= 0.500:
            zone = 'MODERATE_RETRACEMENT'
        elif ratio <= 0.618:
            zone = 'GOLDEN_ZONE'          # 50-61.8 is the golden zone
        elif ratio <= 0.786:
            zone = 'OTE_ZONE'             # Day 81+ — OTE (Optimal Trade Entry) 61.8-78.6%
        elif ratio <= 1.0:
            zone = 'NEAR_SWING_LOW'
        else:
            zone = 'BELOW_LOW'

        return {
            'nearest_level': nearest_label,
            'nearest_price': nearest_price,
            'nearest_pips':  nearest_pips,
            'ratio':         round(ratio, 4),
            'zone':          zone,
            'in_golden_zone': 0.500 <= ratio <= 0.618,
            # Day 81+ — OTE (Optimal Trade Entry) zone: 61.8%-78.6%
            # This is the ICT "sweet spot" for sniper entries.
            'in_ote_zone':   0.618 <= ratio <= 0.786,
        }

    # ═══════════════════════════════════════════════════════════
    # STEP 6: SIGNAL GENERATION
    # ═══════════════════════════════════════════════════════════

    def _generate_signal(
        self,
        curr_price:       float,
        position:         dict,
        confluence_zones: list,
        trend:            str,
        ind_ctx:          dict = None,
        df:               pd.DataFrame = None,
        extensions:       dict = None,
        retracements:     dict = None,
        htf_ctx:          dict = None,
        flip_zones:       list = None,
        ma:               float = None,
        atr_avg:          float = None,
        clusters:         list = None,
        swing_age_bars:   int = None,
        liquidity_ctx:    dict = None,
    ) -> dict:
        """
        Fib position + confluence + indicator দেখে signal দাও।

        Assumes `curr_price` and every value in `position`/`confluence_zones`
        were derived from a fully CLOSED bar (see analyze() docstring). If a
        forming/live bar leaks in here, SL/TP and confidence will repaint as
        that bar's high/low/close keep moving intra-bar.

        Per the strategy references reviewed (ThinkMarkets/Dukascopy-style
        rule sets + the video walkthrough): touching a Fib zone is the
        SETUP, not the trigger. This method treats the raw zone/ratio bias
        below as a candidate only — it's gated by a trigger-candle check
        and a minimum R:R before becoming an actual BUY/SELL.

        htf_ctx : optional {'trend': 'BULLISH'|'BEARISH'|'NEUTRAL', ...}
            describing the prevailing trend on a HIGHER timeframe than
            self.timeframe (caller's responsibility to compute — e.g. run
            a second FibonacciEngine or a simple structure check on H4
            while trading M15). Multiple sources reviewed independently
            converge on the same rule: "Fibonacci is a micro tool, trend
            direction is a macro truth" — a technically perfect Fib setup
            against the higher-timeframe trend has a materially worse
            hit rate than one that agrees with it. When supplied:
              - aligned  -> confidence bonus
              - conflict -> confidence penalty, and (if
                self.require_htf_alignment) the candidate is gated to WAIT
              - not supplied -> no effect at all (fully backward compatible)

        flip_zones : optional output of detect_flip_zones() (Phase 3).
            When the price is currently sitting near ('near_price') a
            level whose status is 'FLIPPED', this OVERRIDES the ordinary
            ratio-based candidate below — not just adjusts its confidence.
            Reasoning: the ratio-based candidate above assumes the zone
            still holds its original role (e.g. "bullish trend + 50-61.8%
            retracement = expect a bounce, go BUY"). If that exact level
            has already broken and flipped to resistance, "expect a
            bounce" is now the WRONG read of the same price location —
            the mechanistically correct expectation is rejection on
            retest, i.e. a SELL, regardless of what the raw ratio would
            otherwise have suggested. Symmetric for a flipped-to-support
            level in a bearish trend -> BUY. Not supplied (None/[]) ->
            no effect, fully backward compatible.

        ma, atr_avg, clusters, swing_age_bars : optional Day 42 context —
            see __init__ docstring for the gates that consume them
            (require_trend_ma, max_atr_multiple/min_rr_dynamic,
            allow_cluster_touch_entry/require_cluster_for_golden). All
            None by default -> those gates are no-ops -> fully backward
            compatible with existing callers that don't supply them.
        """
        zone     = position.get('zone', '')
        in_gold  = position.get('in_golden_zone', False)
        ratio    = position.get('ratio', 0.5)

        # ── Golden-zone gate (Day 42, opt-in — see __init__ docstring) ──
        # Backtest evidence: 50-61.8% retracement entries had a 0% win rate
        # in the reviewed sample. The ratio-based candidate_bias band below
        # (0.382-0.786) structurally still includes this zone, so it's
        # excluded here explicitly rather than by narrowing that band —
        # narrowing the band would also silently exclude MODERATE_RETRACEMENT
        # (0.382-0.5), which the evidence does NOT indict.
        golden_zone_blocked = False
        if zone == 'GOLDEN_ZONE' and not self.allow_golden_zone:
            golden_zone_blocked = True
        elif zone == 'GOLDEN_ZONE' and self.allow_golden_zone and self.require_cluster_for_golden:
            tol = CONFLUENCE_TOLERANCE_PIPS * pip_size(self.symbol)
            has_golden_cluster = any(
                abs(curr_price - c.get('price', curr_price)) <= tol and c.get('swing_count', 0) >= 2
                for c in (clusters or [])
            )
            if not has_golden_cluster:
                golden_zone_blocked = True

        # Base bias from Fibonacci position — this is a CANDIDATE only.
        # Whether it survives to become an actual BUY/SELL happens further
        # down, after the trigger-candle and min-R:R checks.
        if trend == 'BULLISH':
            if 0.382 <= ratio <= 0.786:
                candidate_bias = 'BUY'
                conf = 65
            elif ratio > 0.786:
                candidate_bias = 'WAIT'   # Too deep — swing may be invalid
                conf = 40
            elif ratio < 0.236:
                candidate_bias = 'WAIT'   # Barely retraced — wait for more pullback
                conf = 45
            else:
                candidate_bias = 'BUY'
                conf = 55
        else:  # BEARISH
            if 0.382 <= ratio <= 0.786:
                candidate_bias = 'SELL'
                conf = 65
            elif ratio > 0.786:
                candidate_bias = 'WAIT'
                conf = 40
            elif ratio < 0.236:
                candidate_bias = 'WAIT'
                conf = 45
            else:
                candidate_bias = 'SELL'
                conf = 55

        if golden_zone_blocked and candidate_bias in ('BUY', 'SELL'):
            candidate_bias = 'WAIT'
            conf = 30

        # ── Flip-zone retest override (Phase 3) ──────────────────────────
        # Deliberately placed BEFORE the golden-zone/confluence/indicator
        # bonuses below, so those still apply on top of a flip-retest
        # candidate exactly as they would any other candidate — a flip
        # setup with strong confluence should still score higher than one
        # without. This REPLACES candidate_bias/conf, it doesn't blend
        # with the ratio-based value above (the two readings of "what
        # should happen here" are contradictory, not additive).
        is_flip_retest = False
        active_flip = None
        if flip_zones:
            active_flip = next(
                (f for f in flip_zones if f['status'] == 'FLIPPED' and f['near_price']),
                None,
            )
        if active_flip:
            is_flip_retest = True
            candidate_bias  = 'SELL' if active_flip['role'] == 'RESISTANCE' else 'BUY'
            conf = 60  # base confidence for a flip-retest setup, independent of the ratio score above

        # Golden zone bonus
        if in_gold:
            conf += 12

        # Confluence bonus
        top_confluence = confluence_zones[0] if confluence_zones else None
        if top_confluence:
            if top_confluence.get('near_price'):
                conf += top_confluence['strength'] // 10

        # Indicator alignment bonus
        if ind_ctx:
            rsi    = ind_ctx.get('rsi', 50)
            trend_ = ind_ctx.get('trend', '')
            macd_c = ind_ctx.get('macd_cross', '')

            if candidate_bias == 'BUY':
                if rsi < 50 and 'bullish' in trend_:   conf += 8
                if 'bullish_cross' in macd_c:           conf += 6
                if rsi > 70:                            conf -= 10  # overbought
            elif candidate_bias == 'SELL':
                if rsi > 50 and 'bearish' in trend_:   conf += 8
                if 'bearish_cross' in macd_c:           conf += 6
                if rsi < 30:                            conf -= 10  # oversold

        # ── Higher-timeframe alignment ──────────────────────────────────
        htf_conflict = False
        htf_trend = (htf_ctx or {}).get('trend')
        if htf_trend in ('BULLISH', 'BEARISH'):
            wants_bullish = candidate_bias == 'BUY'
            htf_bullish   = htf_trend == 'BULLISH'
            if wants_bullish == htf_bullish:
                conf += 10
            else:
                htf_conflict = True
                conf -= 20

        conf = max(0, min(99, conf))

        # Entry / SL / TP suggestion
        entry = curr_price

        # FIX (institutional review, item #2): this used to fall back to a
        # hardcoded 0.0010, which is correct order-of-magnitude for 5-decimal
        # majors (EURUSD, GBPUSD) but wrong by ~10-100x for JPY crosses and
        # by ~1000x for metals (XAUUSD). If the caller didn't supply ind_ctx
        # with a real ATR, compute one directly from the price data instead
        # of guessing — this module already has _calc_atr for exactly this.
        if ind_ctx and ind_ctx.get('atr'):
            atr = ind_ctx['atr']
        elif df is not None and len(df) >= 2:
            atr = self._calc_atr(df)
        else:
            # Last-resort fallback only fires with no df and no ind_ctx at
            # all (e.g. someone calling _generate_signal directly in a unit
            # test) — scale it off price rather than a fixed constant so it
            # is at least the right order of magnitude for the instrument.
            atr = abs(curr_price) * 0.001 if curr_price else 0.0010

        # FIX (institutional review, item #3): tp1/tp used to always be
        # None even though extensions were already computed in analyze() —
        # the signal a downstream DecisionAgent actually consumes never
        # carried a target. Wire the nearest extension level in the trade's
        # direction as TP1, and expose R:R using the SL distance.
        # Day 41 — CRITICAL FIX (found via backtest smoke-test, confirmed
        # live-affecting): `extensions` is always computed off the SWING's
        # own trend direction (calculate_extension(swing_high, swing_low,
        # trend) — see that method: BULLISH extends upward from swing_low,
        # BEARISH extends downward from swing_high). That's correct for a
        # trend-aligned candidate (BUY in a BULLISH swing, SELL in a
        # BEARISH one) — the extension continues the same direction the
        # trade is going, so `candidates` (extension levels beyond entry
        # in the trade's direction) is always non-empty.
        #
        # But candidate_bias can be flipped OPPOSITE the swing's trend by
        # the flip-zone-retest override above (a SELL inside a BULLISH
        # swing, betting on rejection from a flipped resistance). In that
        # case every extension level is a bullish continuation target —
        # i.e. on the WRONG side of entry for a SELL — so `candidates`
        # (filtered to p < entry) comes back EMPTY, and the old code fell
        # back to `extensions.get('TP1')` UNCONDITIONALLY: the nearest
        # bullish extension, above entry, handed to a SELL as its take-
        # profit. That TP sits on the losing side of the trade — the
        # backtest's SL/TP check (`hit_tp: lo <= trade.tp`) then fires
        # almost immediately since the "target" is trivially easy to
        # touch, mislabeling a losing trade WIN. This is not backtest-only:
        # the same tp1 value is what a live DecisionAgent would receive.
        #
        # Fix: never hand back a TP on the wrong side of entry. If there's
        # no valid extension-based target in the trade's actual direction
        # (which only happens for a counter-trend flip-zone candidate),
        # tp1 stays None — the R:R gate below then fails (rr is None ->
        # rr_ok False) and the signal correctly downgrades to WAIT instead
        # of shipping a structurally-wrong target. Trend-aligned trades are
        # completely unaffected (candidates is essentially always non-empty
        # for them, so behavior there is unchanged).
        tp1 = None
        if extensions and candidate_bias in ('BUY', 'SELL'):
            ext_levels = extensions.get('levels', {})
            if candidate_bias == 'BUY':
                candidates = [p for p in ext_levels.values() if p > entry]
                tp1 = min(candidates) if candidates else None
            else:
                candidates = [p for p in ext_levels.values() if p < entry]
                tp1 = max(candidates) if candidates else None

        # tp2 (Day 42) — next extension level beyond tp1 in the same
        # direction, e.g. 127.2% -> 161.8%. Exists purely so a caller (e.g.
        # the backtester) can implement scaled/partial exits (take partial
        # size at tp1, move SL to breakeven, let the remainder run to tp2)
        # instead of all-in/all-out at a single target. None if there's no
        # further extension level beyond tp1 — callers should treat that as
        # "no runner target available", not an error.
        tp2 = None
        if extensions and candidate_bias in ('BUY', 'SELL') and tp1 is not None:
            ext_levels = extensions.get('levels', {})
            if candidate_bias == 'BUY':
                further = [p for p in ext_levels.values() if p > tp1]
                tp2 = min(further) if further else None
            else:
                further = [p for p in ext_levels.values() if p < tp1]
                tp2 = max(further) if further else None

        # FIX (strategy-reference review): SL used to be placed 1.5x ATR
        # from the NEAREST FIB LEVEL — which, when price is already sitting
        # right on that level, can be only a few pips away. Every reference
        # you shared places the stop beyond the SWING POINT (the anchor the
        # whole grid was drawn from), with a small ATR buffer for noise —
        # not tight against the level being traded. `sl` below is now that
        # structural stop; `sl_tight` keeps the old fib-level-based distance
        # for comparison/logging only — don't trade off sl_tight, it stops
        # out on ordinary noise before the setup has a chance to work.
        sl, sl_tight, reason = None, None, f"Fib zone {zone} — wait for better position"
        swing_high = retracements.get('high') if retracements else None
        swing_low  = retracements.get('low')  if retracements else None

        if candidate_bias == 'BUY' and swing_low is not None:
            sl        = round(swing_low - atr * 0.5, 5)
            sl_tight  = round(position['nearest_price'] - atr * 1.5, 5)
            reason    = f"Price in Fib {zone} zone ({ratio*100:.1f}%) — bullish retracement"
        elif candidate_bias == 'SELL' and swing_high is not None:
            sl        = round(swing_high + atr * 0.5, 5)
            sl_tight  = round(position['nearest_price'] + atr * 1.5, 5)
            reason    = f"Price in Fib {zone} zone ({ratio*100:.1f}%) — bearish retracement"

        rr = None
        if sl is not None and tp1 is not None:
            risk   = abs(entry - sl)
            reward = abs(tp1 - entry)
            rr     = round(reward / risk, 2) if risk > 1e-9 else None

        # ── Gate 1: trigger candle ─────────────────────────────────────
        # All three strategy write-ups you shared (and the video) agree:
        # price sitting in a zone is the setup, not the signal. Require an
        # actual reversal/momentum candle on the last CLOSED bar before
        # promoting the candidate to a real BUY/SELL.
        trigger = {'confirmed': not self.require_trigger_candle, 'pattern': None}
        if self.require_trigger_candle and candidate_bias in ('BUY', 'SELL') and df is not None:
            trigger = self._detect_trigger_candle(df, candidate_bias, zone, retracements)

            # Day 42 (opt-in, loosens frequency) — a multi-swing cluster at
            # the current price is an independently-corroborated structural
            # signal, not a candle-shape claim. If the candle-based check
            # above didn't confirm, accept a cluster touch instead of
            # rejecting outright. Deliberately exempt from
            # require_engulfing_only below — that restriction is about
            # candle SHAPE quality, which doesn't apply to this trigger.
            if not trigger['confirmed'] and self.allow_cluster_touch_entry and clusters:
                tol = CONFLUENCE_TOLERANCE_PIPS * pip_size(self.symbol)
                near_cluster = next(
                    (c for c in clusters
                     if abs(curr_price - c.get('price', curr_price)) <= tol and c.get('swing_count', 0) >= 2),
                    None,
                )
                if near_cluster:
                    trigger = {'confirmed': True, 'pattern': 'cluster_touch'}

            # Day 42 (opt-in, tightens frequency) — backtest evidence: the
            # only reliably profitable trigger pattern was engulfing; pin
            # bars and momentum breakouts underperformed. liquidity_sweep
            # and cluster_touch are independently structurally validated
            # (not generic candle shapes), so they're not restricted here.
            if (self.require_engulfing_only and trigger['confirmed']
                    and trigger.get('pattern') not in ('bullish_engulfing', 'bearish_engulfing',
                                                        'liquidity_sweep', 'cluster_touch')):
                trigger = {'confirmed': False, 'pattern': trigger.get('pattern')}

        # ── Gate 2: minimum reward:risk (optionally dynamic, Day 42) ────
        # min_rr_dynamic (opt-in): lower the bar (never raise it) when
        # current ATR is demonstrably below its trailing average — calmer
        # conditions where a smaller R:R target is more plausibly reachable
        # without extra noise-driven stop-outs. Floored at 1.0 regardless.
        effective_min_rr = self.min_rr
        if self.min_rr_dynamic and atr_avg and atr_avg > 0 and atr < atr_avg * 0.8:
            effective_min_rr = max(1.0, self.min_rr - 0.5)
        rr_ok = (rr is not None and rr >= effective_min_rr)

        # ── Gate 3: higher-timeframe alignment (opt-in) ──────────────────
        # Off by default (see __init__ docstring) so callers without
        # htf_ctx wired up are completely unaffected. When on, a conflict
        # kills the trade outright rather than just denting confidence —
        # matching "when they disagree, your setup fails, even if it
        # looks perfect" from the shared material.
        htf_ok = not (self.require_htf_alignment and htf_conflict)

        # ── Gate 4: minimum confluence strength (opt-in, Day 42) ─────────
        # Backtest evidence: the blended "confidence" score did NOT predict
        # winners (80+ confidence had 0% win rate, n=2); confluence strength
        # (S/R + multi-level agreement) is a more mechanistic proxy.
        confluence_ok = True
        if self.min_confluence_strength > 0:
            confluence_ok = bool(top_confluence) and top_confluence.get('strength', 0) >= self.min_confluence_strength

        # ── Gate 5: trend-MA alignment (opt-in, Day 42) ──────────────────
        trend_ma_ok = True
        if self.require_trend_ma and ma is not None:
            if candidate_bias == 'BUY' and curr_price < ma:
                trend_ma_ok = False
            elif candidate_bias == 'SELL' and curr_price > ma:
                trend_ma_ok = False

        # ── Gate 6: minimum ADX / trend strength (opt-in, Day 42) ────────
        # Backtest evidence: bearish trades (54.5% WR) badly outperformed
        # bullish (18.2% WR) in what was a bearish-trending window —
        # consistent with Fib retracement setups needing genuine trend
        # strength to retrace FROM, not a ranging/choppy market.
        adx_val = (ind_ctx or {}).get('adx')
        adx_ok = True
        if self.min_adx > 0:
            adx_ok = adx_val is not None and adx_val >= self.min_adx

        # ── Gate 7: volatility ceiling (opt-in, Day 42) ──────────────────
        # Avoid entries during a volatility spike relative to recent norms
        # (more prone to false breakouts/whipsaw than the SL/TP math, which
        # is sized off a SINGLE current ATR reading, accounts for).
        atr_ok = True
        if self.max_atr_multiple > 0 and atr_avg and atr_avg > 0:
            atr_ok = atr <= atr_avg * self.max_atr_multiple

        # ── Gate 8: liquidity-engine alignment (opt-in, Day 63) ──────────
        # Off by default — see __init__ docstring for the evidence (n=32
        # EURUSD H1 sample: dropped 1 conflicting trade, WR 34.4%->35.5%,
        # PF 1.22->1.28 — real but too thin to trust as a primary edge).
        # Informational fields (liquidity_bias/score) are populated below
        # regardless of whether this gate is enabled, so callers can see
        # what the liquidity engine said even when it isn't blocking.
        liq_bias = (liquidity_ctx or {}).get('bias')
        liq_score = (liquidity_ctx or {}).get('score')
        liquidity_conflict = False
        if liq_bias in ('BULLISH', 'BEARISH'):
            wants_bullish = candidate_bias == 'BUY'
            liq_bullish = liq_bias == 'BULLISH'
            liquidity_conflict = (wants_bullish != liq_bullish)
        liquidity_ok = True
        if self.require_liquidity_alignment:
            liquidity_ok = not liquidity_conflict
            if liquidity_ok and self.min_liquidity_score > 0:
                liquidity_ok = (liq_bias in ('BULLISH', 'BEARISH')
                                 and (liq_score or 0) >= self.min_liquidity_score
                                 and not liquidity_conflict)

        all_gates_ok = (trigger['confirmed'] and rr_ok and htf_ok and confluence_ok
                        and trend_ma_ok and adx_ok and atr_ok and liquidity_ok
                        and not golden_zone_blocked)

        if candidate_bias in ('BUY', 'SELL') and all_gates_ok:
            bias = candidate_bias
        else:
            bias = 'WAIT'
            if candidate_bias in ('BUY', 'SELL') and golden_zone_blocked:
                reason = ("Golden zone (50-61.8%) entries disabled by default — backtest evidence "
                          "showed a 0% win rate for this band; set allow_golden_zone=True "
                          "(with require_cluster_for_golden) to re-enable")
            elif candidate_bias in ('BUY', 'SELL') and not confluence_ok:
                reason = (f"Confluence strength {top_confluence.get('strength', 0) if top_confluence else 0} "
                          f"below required min_confluence_strength={self.min_confluence_strength}")
            elif candidate_bias in ('BUY', 'SELL') and not trend_ma_ok:
                reason = f"Price not aligned with {self.trend_ma_period}-period trend MA filter"
            elif candidate_bias in ('BUY', 'SELL') and not adx_ok:
                reason = (f"ADX {adx_val if adx_val is not None else 'unavailable'} below "
                          f"min_adx={self.min_adx} — trend too weak/ranging")
            elif candidate_bias in ('BUY', 'SELL') and not atr_ok:
                reason = (f"ATR {atr:.5f} exceeds max_atr_multiple={self.max_atr_multiple}x "
                          f"average ({atr_avg:.5f}) — volatility too high")
            elif candidate_bias in ('BUY', 'SELL') and not htf_ok:
                reason = (f"Fib {zone} setup confirmed but conflicts with higher-timeframe "
                          f"trend ({htf_trend}) — skipping (require_htf_alignment=True)")
            elif candidate_bias in ('BUY', 'SELL') and not trigger['confirmed']:
                reason = (f"Fib {zone} zone reached ({ratio*100:.1f}%) but no confirmation "
                          f"candle yet — waiting for reversal/momentum trigger before "
                          f"{candidate_bias.lower()}")
            elif candidate_bias in ('BUY', 'SELL') and not rr_ok:
                reason = (f"Fib {zone} setup confirmed but R:R {rr} is below the "
                          f"{effective_min_rr} minimum — skipping")
            elif candidate_bias in ('BUY', 'SELL') and not liquidity_ok:
                reason = (f"Fib {zone} setup confirmed but liquidity engine bias "
                          f"({liq_bias}, score={liq_score}) conflicts — skipping "
                          f"(require_liquidity_alignment=True)")
            sl, tp1, tp2, rr = None, None, None, None

        return {
            'bias':            bias,
            'candidate_bias':  candidate_bias,
            'liquidity_bias':  liq_bias,
            'liquidity_score': liq_score,
            'liquidity_conflict': liquidity_conflict,
            'confidence':      conf,
            'zone':            zone,
            'strategy_type':   self._strategy_type(zone),
            'in_golden_zone':  in_gold,
            'entry':           round(entry, 5),
            'sl':              sl,
            'sl_tight':        sl_tight,
            'tp1':             round(tp1, 5) if tp1 is not None else None,
            'tp2':             round(tp2, 5) if tp2 is not None else None,
            'rr':              rr,
            'trigger_pattern': trigger.get('pattern'),
            'reason':          reason,
            'top_confluence':  top_confluence,
            'htf_trend':       htf_trend,
            'htf_conflict':    htf_conflict,
            'swing_age_bars':  swing_age_bars,
        }

    def _strategy_type(self, zone: str) -> str:
        """Map a Fib zone to the named strategy it corresponds to in the
        reviewed strategy references — purely a reporting label, doesn't
        change any math."""
        if zone in ('SHALLOW_RETRACEMENT', 'MINOR_RETRACEMENT'):
            return 'SHALLOW_CONTINUATION'          # 23.6%-38.2% pullback
        if zone in ('MODERATE_RETRACEMENT', 'GOLDEN_ZONE'):
            return 'GOLDEN_ZONE'                    # 50%-61.8% kill zone
        if zone == 'OTE_ZONE':
            return 'DEEP_REVERSAL'                  # 61.8%-78.6%, needs stronger confirmation
        if zone in ('NEAR_SWING_LOW', 'BELOW_LOW', 'ABOVE_HIGH'):
            return 'BREAKOUT_RETEST_WATCH'          # beyond 100% — different playbook, not traded here
        return 'UNKNOWN'

    def _detect_trigger_candle(self, df: pd.DataFrame, direction: str, zone: str,
                                retracements: dict = None) -> dict:
        """
        Thin wrapper around _detect_trigger_candle_pattern() that additionally
        applies the Day 42 opt-in volume-confirmation gate (see __init__
        docstring: require_volume_confirmation / volume_multiple). Kept as a
        separate method rather than inlined so the pattern-detection logic
        itself stays testable/readable without the volume concern mixed in.
        """
        result = self._detect_trigger_candle_pattern(df, direction, zone, retracements)
        if result['confirmed'] and not self._volume_confirmed(df):
            return {'confirmed': False, 'pattern': result['pattern']}
        return result

    def _volume_confirmed(self, df: pd.DataFrame) -> bool:
        """
        Day 42 (opt-in) — guards against low-participation fakeout candles.
        Requires the trigger bar's volume >= volume_multiple x the mean of
        the prior 5 closed bars. Returns True (never blocks) when the gate
        is disabled, or when volume data isn't available/usable — MT5's
        'volume' is tick_volume (a proxy, not true traded volume), and not
        every data source supplies it, so this is a heuristic filter, not
        a hard truth; missing data should not silently kill every signal.
        """
        if not self.require_volume_confirmation:
            return True
        if df is None or 'volume' not in df.columns or len(df) < 6:
            return True
        avg_vol = df['volume'].iloc[-6:-1].mean()
        if not avg_vol or avg_vol <= 0:
            return True
        return bool(df['volume'].iloc[-1] >= avg_vol * self.volume_multiple)

    def _detect_trigger_candle_pattern(self, df: pd.DataFrame, direction: str, zone: str,
                                        retracements: dict = None) -> dict:
        """
        Confirmation-candle check on the last CLOSED bar (df.iloc[-1]).

        Zone-dependent, matching the reviewed strategy references:
          - Shallow zones (23.6-38.2%): momentum/breakout candle in the
            trend direction — these pullbacks are shallow because momentum
            is strong, so the trigger is continuation strength, not reversal.
          - Golden/deep zones (50%+): a liquidity sweep (see
            _detect_liquidity_sweep — checked FIRST, it's the more specific
            and more frequently cited pattern across the shared material)
            or, failing that, a plain reversal candle (engulfing/pin bar).

        Returns {'confirmed': bool, 'pattern': str | None}. Never uses
        df.iloc[-1] to mean anything other than the most recent CLOSED bar —
        see the closed-bar contract in analyze()'s docstring.
        """
        if df is None or len(df) < 2:
            return {'confirmed': False, 'pattern': None}

        c0 = df.iloc[-1]   # trigger bar
        c1 = df.iloc[-2]   # prior bar, for engulfing comparison
        rng0 = c0['high'] - c0['low']
        if rng0 <= 0:
            return {'confirmed': False, 'pattern': None}
        body0 = abs(c0['close'] - c0['open'])
        shallow_zone = zone in ('SHALLOW_RETRACEMENT', 'MINOR_RETRACEMENT')
        deep_zone     = zone in ('MODERATE_RETRACEMENT', 'GOLDEN_ZONE', 'OTE_ZONE')

        # Liquidity sweep takes priority in deep/golden zones — repeatedly
        # described across the shared material as the actual entry trigger
        # ("the trader failed to understand the intention behind the
        # retracement... it was a liquidity grab"), distinct from and
        # more specific than a generic pin bar.
        if deep_zone and not shallow_zone:
            sweep = self._detect_liquidity_sweep(df, direction, retracements)
            if sweep['swept']:
                return {'confirmed': True, 'pattern': 'liquidity_sweep'}

        if direction == 'BUY':
            bullish_engulf = (c1['close'] < c1['open'] and c0['close'] > c0['open']
                               and c0['close'] >= c1['open'] and c0['open'] <= c1['close'])
            lower_wick = min(c0['close'], c0['open']) - c0['low']
            pin_bar = (c0['close'] > c0['open'] and lower_wick >= rng0 * 0.6
                       and body0 <= rng0 * 0.35)
            momentum = (c0['close'] > c0['open'] and body0 >= rng0 * 0.6
                        and c0['close'] > c1['high'])
            if shallow_zone:
                return {'confirmed': momentum, 'pattern': 'momentum_breakout' if momentum else None}
            if bullish_engulf:
                return {'confirmed': True, 'pattern': 'bullish_engulfing'}
            if pin_bar:
                return {'confirmed': True, 'pattern': 'bullish_pin_bar'}
            return {'confirmed': False, 'pattern': None}

        else:  # SELL
            bearish_engulf = (c1['close'] > c1['open'] and c0['close'] < c0['open']
                               and c0['close'] <= c1['open'] and c0['open'] >= c1['close'])
            upper_wick = c0['high'] - max(c0['close'], c0['open'])
            pin_bar = (c0['close'] < c0['open'] and upper_wick >= rng0 * 0.6
                       and body0 <= rng0 * 0.35)
            momentum = (c0['close'] < c0['open'] and body0 >= rng0 * 0.6
                        and c0['close'] < c1['low'])
            if shallow_zone:
                return {'confirmed': momentum, 'pattern': 'momentum_breakout' if momentum else None}
            if bearish_engulf:
                return {'confirmed': True, 'pattern': 'bearish_engulfing'}
            if pin_bar:
                return {'confirmed': True, 'pattern': 'bearish_pin_bar'}
            return {'confirmed': False, 'pattern': None}

    def _detect_liquidity_sweep(self, df: pd.DataFrame, direction: str,
                                 retracements: dict = None) -> dict:
        """
        "Internal liquidity sweep" pattern, repeatedly described across the
        shared material as the actual entry trigger for Fib golden-zone
        trades (as opposed to a plain touch or a generic pin bar):

          BUY  (bullish retracement, price pulling back down into the
                zone): the last CLOSED bar wicks BELOW the zone's outer
                boundary (the 61.8% retracement level — the deeper edge
                of the golden/OTE zone) then CLOSES back above it. The
                wick clears out stops resting below the level; the close
                back inside shows the sweep was rejected, not accepted.
          SELL (bearish retracement, price pulling back up into the
                zone): mirror image — wicks ABOVE 61.8%, closes back
                below it.

        This deliberately checks against the 61.8% level specifically
        (not the raw swing point) because that's the boundary the source
        material consistently describes stops clustering around, and
        because it fires earlier/more often than waiting for a full
        swing-point sweep while still requiring a genuine wick-then-
        reclaim, not just "price is inside the zone."

        Returns {'swept': bool, 'boundary': float | None}. Silently
        returns not-swept (never raises) if retracements/levels are
        missing — this is an additive signal, not a required one.
        """
        if df is None or len(df) < 1 or not retracements:
            return {'swept': False, 'boundary': None}

        levels = retracements.get('levels', {})
        boundary = levels.get('61.8%')
        if boundary is None:
            return {'swept': False, 'boundary': None}

        c0 = df.iloc[-1]
        if direction == 'BUY':
            swept = bool(c0['low'] < boundary and c0['close'] > boundary)
        else:
            swept = bool(c0['high'] > boundary and c0['close'] < boundary)

        return {'swept': swept, 'boundary': boundary}

    # ═══════════════════════════════════════════════════════════
    # STEP 7: FAILURE RISK DETECTION
    # ═══════════════════════════════════════════════════════════

    def _detect_failure_risk(
        self,
        df:       pd.DataFrame,
        position: dict,
        ind_ctx:  dict = None,
    ) -> dict:
        """
        Fibonacci level fail করার risk detect করো।

        High risk conditions:
          - High volatility (large ATR)
          - RSI extreme opposite direction
          - Price below 78.6% (very deep retracement)
          - News event (external — caller must inject)
        """
        risks   = []
        risk_score = 0

        ratio = position.get('ratio', 0.5)

        # Deep retracement risk
        if ratio > 0.786:
            risks.append("Price below 78.6% — swing may be invalidated")
            risk_score += 30

        # Volatility check
        if ind_ctx:
            atr   = ind_ctx.get('atr', 0)
            price = ind_ctx.get('price', 1)
            atr_pct = atr / max(price, 1e-5) * 100

            if atr_pct > 0.15:
                risks.append(f"High volatility (ATR={atr_pct:.2f}%) — Fib levels less reliable")
                risk_score += 20

            rsi = ind_ctx.get('rsi', 50)
            trend = ind_ctx.get('trend', '')

            if ratio < 0.618 and 'strong_bearish' in trend:
                risks.append("Shallow retracement in strong bearish trend — likely to break lower")
                risk_score += 15

            if ratio < 0.618 and 'strong_bullish' in trend:
                risks.append("Shallow retracement in strong bullish trend — may not retrace deeper")
                risk_score += 10

        risk_level = 'HIGH' if risk_score >= 40 else ('MEDIUM' if risk_score >= 20 else 'LOW')

        return {
            'risk_score':  risk_score,
            'risk_level':  risk_level,
            'risks':       risks,
            'note':        f"Fib failure risk: {risk_level} ({risk_score}/100)",
        }

    # ═══════════════════════════════════════════════════════════
    # AI CONTEXT — Integration
    # ═══════════════════════════════════════════════════════════

    def get_ai_context(self, result: dict) -> dict:
        """
        DecisionAgent, SignalEngine, MarketBiasEngine-এর জন্য
        Fibonacci context।
        """
        if not result.get('swing', {}).get('valid'):
            return {
                'fib_valid':        False,
                'fib_bias':         'NEUTRAL',
                'fib_confidence':   0,
                'fib_zone':         'NONE',
                'fib_level_near':   None,
                'fib_in_golden':    False,
                'fib_confluence':   0,
                'fib_confluence_strength': 0,
                'fib_signal':       'WAIT',
                'fib_failure_risk': 'UNKNOWN',
                'fib_swing_high':   None,
                'fib_swing_low':    None,
                'fib_trend':        'NEUTRAL',
                'fib_61_8':         None,
                'fib_50_0':         None,
                'fib_38_2':         None,
                'fib_tp1':          None,
                'fib_tp2':          None,
                'fib_tp3':          None,
                'fib_htf_trend':    None,
                'fib_htf_conflict': False,
            }

        signal   = result.get('signal', {})
        position = result.get('position', {})
        retrace  = result.get('retracements', {})
        ext      = result.get('extensions', {})
        conf_z   = result.get('confluence', [])
        failure  = result.get('failure_risk', {})

        top_conf = conf_z[0] if conf_z else {}

        return {
            'fib_valid':              True,
            'fib_bias':               signal.get('bias', 'WAIT'),
            'fib_confidence':         signal.get('confidence', 0),
            'fib_zone':               position.get('zone', 'UNKNOWN'),
            'fib_level_near':         position.get('nearest_level'),
            'fib_level_near_price':   position.get('nearest_price'),
            'fib_level_near_pips':    position.get('nearest_pips'),
            'fib_in_golden':          position.get('in_golden_zone', False),
            # Day 81+ — OTE zone for ICT sniper entries
            'fib_in_ote':             position.get('in_ote_zone', False),
            'fib_confluence':         len(conf_z),
            'fib_confluence_strength': top_conf.get('strength', 0),
            'fib_confluence_note':    top_conf.get('note', ''),
            'fib_signal':             signal.get('bias', 'WAIT'),
            'fib_signal_reason':      signal.get('reason', ''),
            'fib_failure_risk':       failure.get('risk_level', 'LOW'),
            'fib_failure_score':      failure.get('risk_score', 0),
            # Swing info
            'fib_swing_high':         result.get('swing_high'),
            'fib_swing_low':          result.get('swing_low'),
            'fib_trend':              result.get('trend', 'NEUTRAL'),
            'fib_range_pips':         result.get('range_pips', 0),
            # Key levels
            'fib_61_8':               retrace.get('61.8'),
            'fib_50_0':               retrace.get('50.0'),
            'fib_38_2':               retrace.get('38.2'),
            'fib_78_6':               retrace.get('78.6'),
            'fib_23_6':               retrace.get('23.6'),
            # Targets
            'fib_tp1':                ext.get('TP1'),
            'fib_tp2':                ext.get('TP2'),
            'fib_tp3':                ext.get('TP3'),
            # Higher-timeframe alignment (None/False when htf_ctx wasn't supplied)
            'fib_htf_trend':          signal.get('htf_trend'),
            'fib_htf_conflict':       signal.get('htf_conflict', False),
        }

    # ═══════════════════════════════════════════════════════════
    # MEMORY — fib_history table format
    # ═══════════════════════════════════════════════════════════

    def get_memory_record(
        self,
        result:    dict,
        pair:      str,
        outcome:   str = None,   # 'WIN' / 'LOSS' / None (open)
        profit_pips: float = None,
    ) -> dict:
        """
        Database-এ save করার জন্য fib_history record।

        Day 52-53 Memory Integration-এ use হবে।
        """
        position = result.get('position', {})
        signal   = result.get('signal', {})

        return {
            'pair':          pair,
            'timeframe':     self.timeframe,
            'swing_high':    result.get('swing_high'),
            'swing_low':     result.get('swing_low'),
            'fib_trend':     result.get('trend'),
            'fib_level':     position.get('nearest_level'),
            'fib_zone':      position.get('zone'),
            'in_golden':     position.get('in_golden_zone', False),
            'confluence':    len(result.get('confluence', [])),
            'conf_strength': result.get('confluence', [{}])[0].get('strength', 0)
                             if result.get('confluence') else 0,
            'signal':        signal.get('bias'),
            'confidence':    signal.get('confidence'),
            'failure_risk':  result.get('failure_risk', {}).get('risk_level'),
            'abc_valid':     result.get('expansion', {}).get('valid', False),
            'abc_c':         result.get('expansion', {}).get('c'),
            'abc_tp1':       result.get('expansion', {}).get('TP1'),
            'cluster_count': len(result.get('clusters', [])),
            'top_cluster':   result.get('clusters', [{}])[0].get('price') if result.get('clusters') else None,
            'outcome':       outcome,
            'profit_pips':   profit_pips,
        }

    # ═══════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════════

    def print_summary(self, result: dict):
        if not result.get('swing', {}).get('valid'):
            print("\n  ⚠️  Fibonacci: No valid swing detected.\n")
            return

        swing  = result['swing']
        retrace = result['retracements']
        ext    = result['extensions']
        pos    = result['position']
        sig    = result['signal']
        conf_z = result['confluence']
        fail   = result['failure_risk']
        price  = result['curr_price']

        trend_icon = '▲' if result['trend'] == 'BULLISH' else '▼'

        print("\n" + "═" * 58)
        print("  📐  FIBONACCI ENGINE  (Day 40)")
        print("═" * 58)
        print(f"  Pair/TF       :  {self.timeframe}")
        print(f"  Swing         :  {trend_icon} {result['trend']}  "
              f"| H={swing['high']:.5f}  L={swing['low']:.5f}  "
              f"| Range={swing['range_pips']:.1f} pips")
        print()

        # Retracement levels
        print("  ── Retracement Levels ──")
        levels_sorted = sorted(
            retrace['levels'].items(),
            key=lambda x: x[1],
            reverse=(result['trend'] == 'BULLISH')
        )
        for label, price_lvl in levels_sorted:
            dist   = price_to_pips(abs(price - price_lvl), self.symbol)
            marker = ' ◄ PRICE' if dist < 5 else (f'  ({dist:.1f}p away)' if dist < 30 else '')
            bold   = ' ⭐' if '61.8' in label or '50.0' in label else ''
            print(f"  {label:<8}  {price_lvl:.5f}{bold}{marker}")

        print()
        print("  ── Extension Targets ──")
        for label, price_lvl in ext['levels'].items():
            if float(label.replace('%', '')) > 100:
                tag = ' (TP1)' if '127' in label else (' (TP2)' if '161' in label else ' (TP3)')
                print(f"  Ext {label:<6}  {price_lvl:.5f}{tag}")

        # ABC Expansion (only once point C has formed)
        expansion = result.get('expansion', {})
        if expansion.get('valid'):
            print()
            print(f"  ── ABC Expansion ──  (A={expansion['a']:.5f}  B={expansion['b']:.5f}  C={expansion['c']:.5f})")
            for label, price_lvl in expansion['levels'].items():
                tag = ' (TP1)' if label == '61.8%' else (' (TP2)' if label == '100.0%' else (' (TP3)' if label == '161.8%' else ''))
                print(f"  ABC {label:<6}  {price_lvl:.5f}{tag}")

        # Position
        print()
        print("  ── Current Position ──")
        golden_tag = '  🌟 GOLDEN ZONE' if pos.get('in_golden_zone') else ''
        print(f"  Price         :  {price:.5f}")
        print(f"  Zone          :  {pos['zone']}{golden_tag}")
        print(f"  Nearest Fib   :  {pos['nearest_level']} ({pos['nearest_pips']:.1f} pips away)")

        # Confluence zones
        if conf_z:
            print()
            print("  ── Confluence Zones ──")
            for z in conf_z[:4]:
                icon = '🔥' if z['strength'] >= 80 else ('⚡' if z['strength'] >= 65 else '💡')
                near = ' ◄ NEAR' if z.get('near_price') else ''
                print(f"  {icon}  {z['price']:.5f}  str={z['strength']}  "
                      f"{' + '.join(z['reasons'][:3])}{near}")

        # Multi-swing clusters (Phase 2) — independent swing grids agreeing
        clusters = result.get('clusters', [])
        if clusters:
            print()
            print("  ── Multi-Swing Clusters ──")
            for z in clusters[:3]:
                print(f"  🧲  {z['price']:.5f}  str={z['strength']}  "
                      f"swings={'+'.join(z['swings'])} ({z['level_count']} levels)")

        # Failure risk
        if fail['risks']:
            print()
            print("  ── Failure Risks ──")
            for r in fail['risks']:
                print(f"  ⚠️  {r}")

        # Signal
        print()
        bias_icon = {'BUY': '🟢', 'SELL': '🔴', 'WAIT': '🟡'}.get(sig['bias'], '⬜')
        print(f"  ┌──────────────────────────────────────────────────┐")
        print(f"  │  {bias_icon} {sig['bias']:<6}  |  Confidence: {sig['confidence']}%              │")
        print(f"  │  {sig['reason'][:52]:<52}│")
        if sig.get('sl'):
            print(f"  │  SL: {sig['sl']:<50}│")
        if sig.get('tp1'):
            rr_txt = f"  (R:R {sig['rr']})" if sig.get('rr') else ''
            print(f"  │  TP1: {str(sig['tp1']) + rr_txt:<49}│")
        print(f"  │  Failure Risk: {fail['risk_level']:<37}│")
        print(f"  └──────────────────────────────────────────────────┘")
        print("═" * 58 + "\n")

    # ═══════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════

    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """ATR calculate করো — column থাকলে নেও, না থাকলে calculate করো।"""
        if 'atr' in df.columns:
            val = df['atr'].iloc[-1]
            if not np.isnan(val):
                return float(val)

        highs  = df['high'].values[-period:]
        lows   = df['low'].values[-period:]
        closes = df['close'].values[-period:]
        trs = [
            max(h - l, abs(h - c), abs(l - c))
            for h, l, c in zip(highs[1:], lows[1:], closes[:-1])
        ]
        return float(np.mean(trs)) if trs else 0.0001

    def _calc_atr_avg(self, df: pd.DataFrame, atr_period: int = 14, avg_window: int = 20) -> float:
        """
        Day 42 — trailing AVERAGE of a rolling ATR series, for the
        volatility gate (max_atr_multiple) and dynamic R:R
        (min_rr_dynamic). Distinct from _calc_atr(), which returns a
        single current-point ATR value; this needs a baseline to compare
        that point against. Falls back to the single-point ATR if there
        isn't enough history for a meaningful rolling average, rather than
        returning 0/None and silently disabling whichever gate called it.
        """
        if df is None or len(df) < atr_period + 2:
            return self._calc_atr(df, atr_period) if df is not None else 0.0001

        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        trs = np.maximum.reduce([
            highs[1:] - lows[1:],
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ])
        atr_series = pd.Series(trs).rolling(atr_period).mean().dropna()
        if len(atr_series) == 0:
            return self._calc_atr(df, atr_period)
        window = atr_series.iloc[-avg_window:]
        return float(window.mean())

    def _empty_result(self, reason: str) -> dict:
        return {
            'swing':        {'valid': False, 'reason': reason},
            'retracements': {},
            'flip_zones':   [],
            'extensions':   {},
            'expansion':    {'valid': False, 'reason': reason},
            'position':     {},
            'confluence':   [],
            'multi_swings': {},
            'clusters':     [],
            'signal':       {'bias': 'WAIT', 'confidence': 0, 'reason': reason},
            'failure_risk': {'risk_level': 'UNKNOWN', 'risk_score': 0, 'risks': []},
        }


# ═══════════════════════════════════════════════════════════════
# QUICK RUN — Direct test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data.fetcher import DataFetcher
    from data.indicators import Indicators
    from analysis.support_resistance import SupportResistance

    fetcher = DataFetcher()
    ind     = Indicators()
    # FIX (Finding #8, S/R correctness audit): this demo fetches "1h"
    # data; the engine's default timeframe ("H1") already matched by
    # coincidence, but made explicit here so it can't silently drift out
    # of sync if the fetch timeframe above is ever changed.
    sr_eng  = SupportResistance(timeframe="H1")

    df = fetcher.fetch_ohlcv("EURUSD", "1h", limit=200)
    if df is not None:
        df      = ind.add_all(df)
        ind_ctx = ind.get_ai_context(df)
        sr_res  = sr_eng.analyze(df)
        sr_ctx  = sr_eng.get_ai_context(sr_res)

        fib    = FibonacciEngine(timeframe='1h')
        result = fib.analyze(df, sr_ctx=sr_ctx, ind_ctx=ind_ctx)
        fib.print_summary(result)

        ctx = fib.get_ai_context(result)
        print("AI Context (for DecisionAgent):")
        for k, v in ctx.items():
            print(f"  {k:<30}: {v}")