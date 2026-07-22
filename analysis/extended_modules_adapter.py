"""
analysis/extended_modules_adapter.py
=====================================
Adapter layer that wires the previously "imported-only" analysis modules
into the live signal-fusion pipeline as standardized directional votes.

Background
----------
An audit found 17 modules under analysis/ that were imported somewhere
in the codebase (scripts/, intelligence/ported_indicators_registry.py,
core/obsolete.py, etc.) but whose calculate/analyze/compute functions
were never actually *called* as part of a live trading decision — so
their contribution to signal generation was 0%, despite the modules
being fully implemented.

This adapter closes that gap for the subset of those modules that are
genuinely per-candle directional signal generators. Each wrapped module
produces a vote of the form:

    (direction: "bullish" | "bearish", weight: int, reason: str)

which is exactly the tuple shape strategy/signal_engine.py already uses
internally for its `signals` list — so these votes plug directly into
the existing bull_score / bear_score accumulation with no change to the
scoring model itself. See `apply_extended_votes()` at the bottom, which
is the single entry point analysis_agent.py / signal_engine.py calls.

Modules wired here (17)
------------------------
  andean_oscillator, supertrend, utbot_alerts, nadaraya_watson_envelope,
  daily_high_low, auction_market_theory, candlestick_patterns_ml,
  breaker_block, flip_zones, curve_mtf
  (10 from the original 17-module "imported-only" audit)

  candlestick_patterns_br, candlestick_patterns_mw, supermao_ichimoku
  (3 more added in a follow-up pass — these were in the SEPARATE "fully
  dead, zero importers anywhere" bucket, not the imported-only 17.
  core/obsolete.py had marked candlestick_patterns_br/_mw as "superseded
  by candlestick_patterns_ml.py", but a direct read of all three files
  shows they are NOT duplicates: _ml.py is an 8-pattern boolean detector,
  _mw.py is an independent 33-pattern scanner with its own trend filter,
  and _br.py is an 11-pattern Brazilian-book scanner with trend/volume/
  next-bar confirmation filtering that _ml.py doesn't have. supermao_ichimoku
  is a distinct Ichimoku implementation from the live analysis/ichimoku.py,
  not a copy of it. Since none of the three actually duplicate live logic
  and all three are working per-candle directional generators, they were
  wired in here rather than deleted. core/obsolete.py has been updated to
  match — see that file's entries for these three paths.)

  vw_macd, supermao_bands
  (2 more added in the 2026-07-22 dead-file audit — both were in the
  fully-dead, zero-importers bucket. Both already expose a clean +1/-1/0
  directional column (vwmacd_cross, sm_signal) — same contract as
  supertrend/andean_oscillator — so no new scoring logic was needed,
  just a wrapper. supermao_bands also computes sm_tp/sm_sl, but only the
  directional signal is used here; TP/SL sizing is deliberately left to
  the risk engine, not duplicated in the fusion layer.)

  golden_death_cross (crossover_signals.py)
  (Added in the Tier 2 pass, 2026-07-22. crossover_signals.py had zero
  importers anywhere in the codebase. Its golden_cross/death_cross
  helpers (sma_50/sma_200, confirmation=True to reject fake crosses)
  don't duplicate any existing live vote — grep confirmed the only other
  "crossover" usages in the codebase are research/strategy_generator.py's
  config dict for an unrelated research tool and phase5_regime.py's
  ema20 crossover used for *regime classification*, not a directional
  fusion vote. Wired directly into get_extended_votes.)

  cci_state_machine (Book 5 Ch.11, added Tier 2 pass 2026-07-22)
  (This one is intentionally wired into get_zone_dependent_votes, NOT
  get_extended_votes, because its own docstring explicitly states CCI
  is "a CONFLUENCE layer, not a standalone signal" and requires a
  coincident supply/demand zone to mean anything (Book P125 explicitly
  warns against standalone CCI use). It needs the same nearest_demand/
  nearest_supply zone data breaker_block/flip_zones/curve_mtf already
  depend on, so it slots into that existing second-pass mechanism
  instead of being forced into the first-pass list. Only the "ENTER"
  action produces a vote (long at demand zone with CCI < -100, short at
  supply zone with CCI > +100); ADD/EXIT/HOLD/NO_TRADE states are
  position-management concepts this adapter has no position context
  for, so they're not translated into votes. trend_align defaults to
  True (matching the module's own default) since this adapter doesn't
  have an independent trend proxy to feed it without duplicating trend
  signals other wired modules already contribute.)

Modules deliberately NOT wired here (12) — with reasons
--------------------------------------------------------
  - atr_sl_finder.py     -> stop-loss sizing tool, not a directional
                            signal. Belongs in the risk/execution layer
                            (position sizing / SL placement), not fusion.
  - chandelier_exit.py   -> trailing-exit tool, not an entry signal.
                            Same reasoning as atr_sl_finder.
  - book_rules_index.py  -> static rules knowledge base (Book 5 chapter
                            index). No market computation to vote with.
  - research_domains.py  -> options/futures/correlation research helpers
                            that need external data feeds per symbol;
                            not a per-candle directional vote.
  - quantitative_factors.py -> statistical regime/factor toolkit (Hurst
                            exponent, Kalman filter, HMM regime, Bayesian
                            win-probability). These describe *how
                            reliable* a regime is, not *which direction*
                            to trade -- a confidence multiplier, not a
                            BUY/SELL vote. Left for a future
                            confidence-layer integration pass; wiring it
                            in here as a fake directional vote would
                            misrepresent what it measures.
  - phd_frontier.py      -> experimental research toolkit (information
                            theory, chaos dynamics, game theory, anomaly
                            detection, knowledge graphs, federated
                            learning). No defined signal contract; needs
                            its own design pass, not a shoehorned vote.
  - risk_management.py   -> position sizing / margin-call / drawdown
                            simulation. This duplicates the live risk
                            engine's job (risk/capital_manager.py etc.)
                            and must never be treated as a trade-entry
                            vote.
  - adx_trend_filter.py  -> outputs adx_filter_pass (should we trade at
                            all given trend strength) plus adx_direction,
                            but its job is gating, not casting a vote of
                            its own. Faking a bullish/bearish vote out of
                            a strength-only filter would double-count
                            trend strength that other modules already
                            price in. Candidate for a future confluence
                            *gate* (multiply confidence, don't add score),
                            not this vote list.
  - adx_filters.py       -> same reasoning as adx_trend_filter.py:
                            bishop_exit/adx_rising are filter helpers
                            (should we exit / is momentum still valid),
                            not directional votes.

Tier 2 pass (2026-07-22) — additional exclusions, with reasons
----------------------------------------------------------------
  - engulfing_bar_strategy.py -> genuinely richer than a bare pattern
                            flag (Nison counter-trend requirement, MA/
                            Fib/SR confluence scoring, quality grading,
                            entry/SL/TP), but engulfing detection itself
                            is NOT new: candlestick_patterns_ml.py,
                            _br.py, and _mw.py are all already wired
                            above and each independently flags bullish/
                            bearish engulfing. A 4th independent vote on
                            the same candle shape would double-count one
                            pattern as up to 4 votes. Its unique value
                            (the confluence/quality layer) belongs as a
                            confidence multiplier on the *existing*
                            engulfing votes, not a 4th additive vote —
                            that redesign is out of scope for this pass.
  - pin_bar_strategy.py  -> same double-count problem as above: a pin
                            bar is a Hammer/Shooting Star/Inverted
                            Hammer by shape, and candlestick_patterns_mw.py
                            (already wired) explicitly scans for exactly
                            those patterns. Wiring this as a 2nd vote on
                            the same candle shape was rejected for the
                            same reason as engulfing_bar_strategy.py.
  - trend_level_signal.py -> not a raw signal at all; its own header
                            says it synthesizes market_regime.py (Trend)
                            + support_resistance.py (Level) +
                            high_reliability_patterns.py (Signal) into
                            one decision. All three inputs already feed
                            the live pipeline independently elsewhere;
                            adding this as a 4th vote would double-count
                            all three simultaneously, not just once.
  - amd_strategy.py      -> confirmed dead code, not a Tier 2 candidate
                            at all: core/obsolete.py already documents
                            it as superseded by ict_amd_signal_engine.py
                            ("stricter spec"), which is live. Verified
                            the file is still present on disk but has no
                            live consumer — a Tier 3-style duplicate that
                            happened to not be listed with the other
                            Tier 3 files in the original audit.
  - megaphone_pennant.py -> by design does NOT produce a directional
                            vote. Its own docstring is explicit: MEGAPHONE
                            means "no trade possible" and PENNANT means
                            "bracket both sides" (buy-stop above + sell-
                            stop below simultaneously) — neither is a
                            bullish/bearish call, and the module says a
                            directional trend should be evaluated by
                            trend-following logic elsewhere. Forcing a
                            vote out of it would misrepresent what it
                            classifies, same reasoning as
                            quantitative_factors.py above.
  - crossover_signals.py -> the golden_cross/death_cross convenience
                            functions ARE wired (see golden_death_cross
                            above); cross_above/cross_below/Cruzamentos
                            remain unwired as they're generic helpers
                            other modules can import directly, not
                            signal generators of their own.

Note on risk_management.py (re-audited during the Tier 2 pass): its
MarginCallDetector/DrawdownSimulator are static Book-formula
calculators (loss% x leverage margin-call math, compounding-loss
simulation) — checked against risk/kill_switch.py and
risk/drawdown_controller.py and confirmed these do NOT duplicate that
math; the risk/ engine tracks live balance/peak-equity state, while
risk_management.py computes hypothetical/reference scenarios. So it is
not a redundant duplicate of the live risk engine as originally
assumed — but the original exclusion reason still holds unchanged: it
is sizing/scenario math, not a directional entry signal, and must never
be wired as a trade-entry vote.

Every wrapper below is defensive: any exception is caught and logged,
and that module's vote is simply omitted. This mirrors the existing
analysis_agent.py convention of one try/except per module, so adding
this adapter can never take down the rest of the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)

Vote = Tuple[str, int, str]  # (direction, weight, reason)


# ──────────────────────────────────────────────────────────────────
# Individual wrappers — one per module. Each returns a Vote or None.
# ──────────────────────────────────────────────────────────────────

def _vote_andean_oscillator(df: pd.DataFrame) -> Optional[Vote]:
    from analysis.andean_oscillator import compute
    out = compute(df)
    phase = int(out["ao_phase"].iloc[-1])
    if phase == 1:
        return ("bullish", 1, "Andean Oscillator: uptrend phase")
    if phase == -1:
        return ("bearish", 1, "Andean Oscillator: downtrend phase")
    return None


def _vote_supertrend(df: pd.DataFrame) -> Optional[Vote]:
    from analysis.supertrend import compute
    out = compute(df)
    trend = int(out["st_trend"].iloc[-1])
    if trend == 1:
        return ("bullish", 2, "SuperTrend: bullish (price above trend line)")
    if trend == -1:
        return ("bearish", 2, "SuperTrend: bearish (price below trend line)")
    return None


def _vote_utbot_alerts(df: pd.DataFrame) -> Optional[Vote]:
    from analysis.utbot_alerts import compute
    out = compute(df)
    bull = float(out["ut_bull_arrow"].iloc[-1])
    bear = float(out["ut_bear_arrow"].iloc[-1])
    if bull != 0:
        return ("bullish", 2, "UT Bot Alert: fresh bull arrow")
    if bear != 0:
        return ("bearish", 2, "UT Bot Alert: fresh bear arrow")
    return None


def _vote_nadaraya_watson(df: pd.DataFrame) -> Optional[Vote]:
    from analysis.nadaraya_watson_envelope import compute
    out = compute(df)
    pos = int(out["nwe_pos"].iloc[-1])
    if pos == 1:
        return ("bullish", 1, "Nadaraya-Watson Envelope: close above upper band (breakout)")
    if pos == -1:
        return ("bearish", 1, "Nadaraya-Watson Envelope: close below lower band (breakdown)")
    return None


def _vote_daily_high_low(df: pd.DataFrame) -> Optional[Vote]:
    from analysis.daily_high_low import compute
    out = compute(df)
    row = out.iloc[-1]
    close = float(row["close"])
    dhl_high = row.get("dhl_high")
    dhl_low = row.get("dhl_low")
    if dhl_high is None or dhl_low is None or pd.isna(dhl_high) or pd.isna(dhl_low):
        return None
    if close > float(dhl_high):
        return ("bullish", 1, "Price broke above previous day's high")
    if close < float(dhl_low):
        return ("bearish", 1, "Price broke below previous day's low")
    return None


def _vote_auction_market_theory(df: pd.DataFrame) -> Optional[Vote]:
    from analysis.auction_market_theory import analyze_auction_market
    result = analyze_auction_market(df)
    signal = str(result.get("signal", "NEUTRAL")).upper()
    score = int(result.get("score", 0))
    reason = result.get("reason", "Auction Market Theory")
    if signal == "BUY" and score > 0:
        return ("bullish", max(1, round(score / 25)), f"Auction Market Theory: {reason}")
    if signal == "SELL" and score > 0:
        return ("bearish", max(1, round(score / 25)), f"Auction Market Theory: {reason}")
    return None


def _vote_candlestick_patterns_ml(df: pd.DataFrame) -> Optional[Vote]:
    """Aggregate the 8 boolean pattern detectors over the last 2 candles
    into a single bullish/bearish vote (bullish patterns minus bearish
    patterns detected on the most recent bar)."""
    from analysis.candlestick_patterns_ml import CandleStickPatterns as C

    if len(df) < 3:
        return None
    cur, prev = df.iloc[-1], df.iloc[-2]

    bullish_hits = 0
    bearish_hits = 0

    try:
        if C.is_hammer(cur["open"], cur["close"], cur["high"], cur["low"]):
            bullish_hits += 1
        if C.is_inverted_hammer(cur["open"], cur["close"], cur["high"], cur["low"]):
            bearish_hits += 1  # shooting-star reading at a high
        if C.is_dragonfly_doji(cur["open"], cur["close"], cur["high"], cur["low"]):
            bullish_hits += 1
        if C.is_bullish_engulfing(cur["open"], cur["close"], prev["open"], prev["close"]):
            bullish_hits += 1
        if C.is_bullish_harami(cur["open"], cur["close"], prev["open"], prev["close"]):
            bullish_hits += 1
        if C.is_piercing_pattern(cur["open"], cur["close"], prev["open"], prev["close"]):
            bullish_hits += 1
    except Exception as e:
        log.debug(f"[ExtendedSignals] candlestick_patterns_ml partial failure: {e}")

    if bullish_hits > bearish_hits:
        return ("bullish", min(2, bullish_hits), f"{bullish_hits} bullish candlestick pattern(s)")
    if bearish_hits > bullish_hits:
        return ("bearish", min(2, bearish_hits), f"{bearish_hits} bearish candlestick pattern(s)")
    return None


_BR_BULLISH_PATTERNS = {
    "Hammer", "Inverted Hammer", "Bullish Engulfing", "Bullish Harami",
    "Piercing Line", "Tweezer Bottom", "Bullish Kicking",
}


def _vote_candlestick_patterns_br(df: pd.DataFrame) -> Optional[Vote]:
    """Brazilian-book scanner (analysis/candlestick_patterns_br.py). This
    module only defines a bullish-reversal pattern set plus 4 direction-
    less "ambivalent" patterns (Marubozu, Doji, Spinning Top, Star) — it
    has no bearish counterpart list, so this wrapper only ever casts
    bullish votes. Distinct from candlestick_patterns_ml.py: this one adds
    trend/volume/next-bar confirmation filters, so it fires less often but
    with higher quality. NOT a duplicate despite core/obsolete.py's old
    "superseded by candlestick_patterns_ml" note — see that file's
    docstring for the correction."""
    from analysis.candlestick_patterns_br import detect_all, PATTERN_FUNCTIONS

    if len(df) < 3:
        return None
    try:
        out = detect_all(df)
        last = out.iloc[-1]
    except Exception as e:
        log.debug(f"[ExtendedSignals] candlestick_patterns_br failed: {e}")
        return None

    fired = [
        name for name in PATTERN_FUNCTIONS
        if name in _BR_BULLISH_PATTERNS and bool(last.get(name, False))
    ]
    if not fired:
        return None
    return ("bullish", min(2, len(fired)), f"Brazilian-book pattern(s): {', '.join(fired)}")


def _vote_candlestick_patterns_mw(df: pd.DataFrame) -> Optional[Vote]:
    """MotiveWave-style 33-pattern scanner (analysis/candlestick_patterns_mw.py).
    Independent implementation from candlestick_patterns_ml.py/patterns_br.py —
    broader pattern set (1/2/3-bar), each pre-classified bullish/bearish/neutral
    via csp_signal. NOT a duplicate of candlestick_patterns_ml.py despite
    core/obsolete.py's old note — see that file's docstring for the
    correction."""
    from analysis.candlestick_patterns_mw import compute

    if len(df) < 5:
        return None
    try:
        out = compute(df)
        sig = out["csp_signal"].iloc[-1]
    except Exception as e:
        log.debug(f"[ExtendedSignals] candlestick_patterns_mw failed: {e}")
        return None
    if pd.isna(sig) or int(sig) == 0:
        return None
    pattern = out["csp_pattern"].iloc[-1]
    if int(sig) == 1:
        return ("bullish", 1, f"MotiveWave scanner: {pattern}")
    return ("bearish", 1, f"MotiveWave scanner: {pattern}")


def _vote_supermao_ichimoku(df: pd.DataFrame) -> Optional[Vote]:
    """Alternate Ichimoku cloud strategy (analysis/supermao_ichimoku.py),
    separate from the live analysis/ichimoku.py. Needs enough bars for the
    52-period Senkou Span B plus the 26-bar forward displacement, so this
    is skipped on short history. TK-cross + cloud-position entry rule
    ported from the original MQL4 EA."""
    from analysis.supermao_ichimoku import compute

    if len(df) < 90:
        return None
    try:
        out = compute(df)
        sig = int(out["smi_signal"].iloc[-1])
    except Exception as e:
        log.debug(f"[ExtendedSignals] supermao_ichimoku failed: {e}")
        return None
    if sig == 1:
        return ("bullish", 2, "SuperMao Ichimoku: bullish TK-cross above cloud")
    if sig == -1:
        return ("bearish", 2, "SuperMao Ichimoku: bearish TK-cross below cloud")
    return None


def _vote_breaker_block(df: pd.DataFrame, order_blocks: Optional[List[Dict]]) -> Optional[Vote]:
    """Requires the order-block list already produced elsewhere in the
    pipeline (analysis/order_block.py, called by smc_engine). If it
    isn't available this cycle, the vote is simply skipped."""
    if not order_blocks:
        return None
    from analysis.breaker_block import BreakerBlockDetector
    detector = BreakerBlockDetector()
    breakers = detector.detect(df, order_blocks)
    active = [b for b in breakers if b.get("active")]
    if not active:
        return None
    # Most recent active breaker wins
    latest = active[-1]
    if latest.get("type") == "bullish_breaker":
        return ("bullish", 2, "Breaker Block: failed bearish OB flipped bullish, retested")
    if latest.get("type") == "bearish_breaker":
        return ("bearish", 2, "Breaker Block: failed bullish OB flipped bearish, retested")
    return None


def _vote_flip_zones(
    df: pd.DataFrame,
    nearest_demand: Optional[Dict],
    nearest_supply: Optional[Dict],
) -> Optional[Vote]:
    """Lightweight, stateless-per-call use of FlipZoneDetector: register
    the nearest demand/supply zones already computed by
    supply_demand_zones.py this cycle, then scan df for a confirmed
    flip. This does not persist zone state across cycles (that would
    require a long-lived detector instance owned by analysis_agent);
    it still correctly catches a flip that completes within the
    lookback window handed to it."""
    if not nearest_demand and not nearest_supply:
        return None
    from analysis.flip_zones import FlipZoneDetector

    detector = FlipZoneDetector()
    if nearest_demand:
        z = dict(nearest_demand)
        z.setdefault("sd_pattern", "demand")
        detector.register_zone(z, current_idx=0)
    if nearest_supply:
        z = dict(nearest_supply)
        z.setdefault("sd_pattern", "supply")
        detector.register_zone(z, current_idx=0)

    events = detector.update(df)
    if not events:
        return None
    latest = events[-1]
    new_type = getattr(latest, "new_type", None) or getattr(latest, "zone_type", None)
    if new_type and "supply" in str(new_type).lower():
        return ("bearish", 2, "Flip Zone: demand zone flipped to supply (confirmed break)")
    if new_type and "demand" in str(new_type).lower():
        return ("bullish", 2, "Flip Zone: supply zone flipped to demand (confirmed break)")
    return None


def _vote_curve_mtf(
    current_price: float,
    nearest_demand: Optional[Dict],
    nearest_supply: Optional[Dict],
) -> Optional[Vote]:
    if not nearest_demand or not nearest_supply:
        return None
    from analysis.curve_mtf import CurveMTF, DirectionalBias

    curve = CurveMTF.from_zones(nearest_demand, nearest_supply, current_price)
    bias = CurveMTF.get_bias(curve, current_price)
    if bias == DirectionalBias.BUY_ONLY:
        return ("bullish", 1, "Curve MTF: HTF curve position permits BUY only")
    if bias == DirectionalBias.SELL_ONLY:
        return ("bearish", 1, "Curve MTF: HTF curve position permits SELL only")
    return None


def _vote_vw_macd(df: pd.DataFrame) -> Optional[Vote]:
    """Volume-Weighted MACD (analysis/vw_macd.py). Was fully dead (zero
    importers) — audit fix wires it in here. `vwmacd_cross` is already a
    clean +1/-1/0 crossover signal (institutional review already fixed
    the real_volume/tick_volume column-priority bug and confirmed the
    calc is causal), same shape as the already-wired supertrend/andean
    votes. Needs tick_volume or real_volume in df; if neither is present
    compute() raises and this wrapper just skips the vote."""
    from analysis.vw_macd import compute
    try:
        out = compute(df)
    except Exception as e:
        log.debug(f"[ExtendedSignals] vw_macd failed: {e}")
        return None
    cross = out["vwmacd_cross"].iloc[-1]
    if pd.isna(cross) or int(cross) == 0:
        return None
    if int(cross) == 1:
        return ("bullish", 2, "VW-MACD: bullish volume-weighted crossover")
    return ("bearish", 2, "VW-MACD: bearish volume-weighted crossover")


def _vote_supermao_bands(df: pd.DataFrame) -> Optional[Vote]:
    """SuperMao multi-band Bollinger + MACD strategy (analysis/supermao_bands.py).
    Was fully dead (zero importers) — audit fix wires it in here.
    `sm_signal` is a clean +1/-1/0 entry signal (the module also computes
    sm_tp/sm_sl for a full trade plan, but only the directional signal is
    used for the fusion vote — SL/TP sizing stays the risk engine's job).
    Needs avg_period=50 warm-up bars; short history returns NaN and the
    vote is skipped."""
    from analysis.supermao_bands import compute
    try:
        out = compute(df)
    except Exception as e:
        log.debug(f"[ExtendedSignals] supermao_bands failed: {e}")
        return None
    sig = out["sm_signal"].iloc[-1]
    if pd.isna(sig) or int(sig) == 0:
        return None
    if int(sig) == 1:
        return ("bullish", 2, "SuperMao Bands: long signal (band + MACD confluence)")
    return ("bearish", 2, "SuperMao Bands: short signal (band + MACD confluence)")


def _vote_golden_death_cross(df: pd.DataFrame) -> Optional[Vote]:
    """SMA-50/SMA-200 golden/death cross (analysis/crossover_signals.py).
    Was fully dead (zero importers). Uses the sma_50/sma_200 columns
    data/indicators_ext.py already computes; confirmation=True so a
    cross that immediately reverses doesn't fire. Needs at least 3 bars
    of both columns present (2 lookback + current)."""
    if "sma_50" not in df.columns or "sma_200" not in df.columns:
        return None
    from analysis.crossover_signals import golden_cross, death_cross

    fast, slow = df["sma_50"], df["sma_200"]
    if fast.isna().iloc[-3:].any() or slow.isna().iloc[-3:].any():
        return None
    if bool(golden_cross(fast, slow, confirmation=True).iloc[-1]):
        return ("bullish", 2, "Golden Cross: SMA50 crossed above SMA200 (confirmed)")
    if bool(death_cross(fast, slow, confirmation=True).iloc[-1]):
        return ("bearish", 2, "Death Cross: SMA50 crossed below SMA200 (confirmed)")
    return None


def _vote_cci_state_machine(
    df: pd.DataFrame,
    nearest_demand: Optional[Dict],
    nearest_supply: Optional[Dict],
) -> Optional[Vote]:
    """Book 5 Ch.11 CCI confluence layer (analysis/cci_state_machine.py).
    Explicitly not a standalone signal per the book -- only fires an
    ENTER vote when CCI is at an extreme AND price is at the matching
    zone type, mirroring the same nearest_demand/nearest_supply contract
    breaker_block/flip_zones/curve_mtf already use. Uses df['cci'] if
    data/indicators_ext.py already computed it this cycle, otherwise
    computes it locally with the same length=20 default so the module's
    own thresholds (calibrated to that period) stay valid."""
    if df is None or len(df) < 20:
        return None
    if "cci" in df.columns and not pd.isna(df["cci"].iloc[-1]):
        cci_value = float(df["cci"].iloc[-1])
    else:
        try:
            import pandas_ta as ta
            cci_series = ta.cci(df["high"], df["low"], df["close"], length=20)
            if cci_series is None or pd.isna(cci_series.iloc[-1]):
                return None
            cci_value = float(cci_series.iloc[-1])
        except Exception as e:
            log.debug(f"[ExtendedSignals] cci_state_machine cci calc failed: {e}")
            return None

    # "At zone" proximity: reuse whichever zone dict is closer, same
    # small-pip-distance idea as the confluence checks elsewhere in this
    # file. distance_pips is produced by supply_demand_zones.py.
    AT_ZONE_PIPS = 15.0
    zone_type = None
    if nearest_demand and nearest_demand.get("distance_pips") is not None:
        if abs(float(nearest_demand["distance_pips"])) <= AT_ZONE_PIPS:
            zone_type = "demand"
    if zone_type is None and nearest_supply and nearest_supply.get("distance_pips") is not None:
        if abs(float(nearest_supply["distance_pips"])) <= AT_ZONE_PIPS:
            zone_type = "supply"
    if zone_type is None:
        return None

    from analysis.cci_state_machine import CCIStateMachine
    sm = CCIStateMachine()
    # No open-position tracking available at this layer, so position is
    # always None here -- this only ever evaluates the ENTER branch,
    # never ADD/EXIT (those require knowing about an existing trade).
    result = sm.evaluate(cci_value=cci_value, zone_type=zone_type, position=None, trend_align=True, at_zone=True)
    if result.action != "ENTER":
        return None
    if result.direction == "long":
        return ("bullish", 1, f"CCI State Machine: {result.reason}")
    if result.direction == "short":
        return ("bearish", 1, f"CCI State Machine: {result.reason}")
    return None


# ──────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────

def get_extended_votes(
    df: pd.DataFrame,
    *,
    order_blocks: Optional[List[Dict]] = None,
    nearest_demand: Optional[Dict] = None,
    nearest_supply: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run all wired extended modules and return their votes plus a
    per-module status breakdown (for logging / health-check surfacing,
    same idea as core/obsolete.py's registry but for what's actually
    firing this cycle).

    Returns
    -------
    {
        "votes": [(direction, weight, reason), ...],
        "module_status": {module_name: "voted" | "no_signal" | "error: ..."},
    }
    """
    if df is None or len(df) < 5:
        return {"votes": [], "module_status": {}}

    current_price = float(df["close"].iloc[-1])

    checks = [
        ("andean_oscillator", lambda: _vote_andean_oscillator(df)),
        ("supertrend", lambda: _vote_supertrend(df)),
        ("utbot_alerts", lambda: _vote_utbot_alerts(df)),
        ("nadaraya_watson_envelope", lambda: _vote_nadaraya_watson(df)),
        ("daily_high_low", lambda: _vote_daily_high_low(df)),
        ("auction_market_theory", lambda: _vote_auction_market_theory(df)),
        ("candlestick_patterns_ml", lambda: _vote_candlestick_patterns_ml(df)),
        ("candlestick_patterns_br", lambda: _vote_candlestick_patterns_br(df)),
        ("candlestick_patterns_mw", lambda: _vote_candlestick_patterns_mw(df)),
        ("supermao_ichimoku", lambda: _vote_supermao_ichimoku(df)),
        ("breaker_block", lambda: _vote_breaker_block(df, order_blocks)),
        ("flip_zones", lambda: _vote_flip_zones(df, nearest_demand, nearest_supply)),
        ("curve_mtf", lambda: _vote_curve_mtf(current_price, nearest_demand, nearest_supply)),
        ("vw_macd", lambda: _vote_vw_macd(df)),
        ("supermao_bands", lambda: _vote_supermao_bands(df)),
        ("golden_death_cross", lambda: _vote_golden_death_cross(df)),
    ]

    votes: List[Vote] = []
    module_status: Dict[str, str] = {}

    for name, fn in checks:
        try:
            vote = fn()
        except Exception as e:
            module_status[name] = f"error: {e}"
            log.debug(f"[ExtendedSignals] {name} raised: {e}")
            continue
        if vote is None:
            module_status[name] = "no_signal"
        else:
            votes.append(vote)
            module_status[name] = "voted"

    return {"votes": votes, "module_status": module_status}


def get_zone_dependent_votes(
    df: pd.DataFrame,
    *,
    order_blocks: Optional[List[Dict]] = None,
    nearest_demand: Optional[Dict] = None,
    nearest_supply: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    breaker_block, flip_zones, and curve_mtf all need order-block /
    demand-supply-zone data that analysis_agent.py only has *after* the
    SMC Engine (step 8) and Supply/Demand Zones step run — which is
    later than the first signal_engine.generate() call (step 6).
    Rather than reorder the whole pipeline, analysis_agent.py calls
    this separately once that data is available, then folds the result
    into the already-computed signal_result via
    `merge_zone_votes_into_signal()` below.
    """
    if df is None or len(df) < 5:
        return {"votes": [], "module_status": {}}

    current_price = float(df["close"].iloc[-1])

    checks = [
        ("breaker_block", lambda: _vote_breaker_block(df, order_blocks)),
        ("flip_zones", lambda: _vote_flip_zones(df, nearest_demand, nearest_supply)),
        ("curve_mtf", lambda: _vote_curve_mtf(current_price, nearest_demand, nearest_supply)),
        ("cci_state_machine", lambda: _vote_cci_state_machine(df, nearest_demand, nearest_supply)),
    ]

    votes: List[Vote] = []
    module_status: Dict[str, str] = {}
    for name, fn in checks:
        try:
            vote = fn()
        except Exception as e:
            module_status[name] = f"error: {e}"
            log.debug(f"[ExtendedSignals] {name} raised: {e}")
            continue
        module_status[name] = "voted" if vote else "no_signal"
        if vote:
            votes.append(vote)

    return {"votes": votes, "module_status": module_status}


def merge_zone_votes_into_signal(signal_result: Dict[str, Any], votes: List[Vote]) -> Dict[str, Any]:
    """
    Fold zone-dependent votes into an already-computed signal_result
    dict (the output of SignalEngine.generate()), recomputing
    signal/confidence with the exact same net-score thresholds
    signal_engine.py uses, so behaviour stays consistent whether a
    vote arrives in the first pass or this second pass.

    Mutates and returns signal_result. No-op if votes is empty.
    """
    if not votes:
        return signal_result

    bull_score = signal_result.get("bull_score", 0)
    bear_score = signal_result.get("bear_score", 0)
    signals = signal_result.get("signals", [])
    warnings = signal_result.get("warnings", [])

    for direction, weight, reason in votes:
        if direction == "bullish":
            bull_score += weight
        elif direction == "bearish":
            bear_score += weight
        signals.append((direction, weight, reason))

    total = bull_score + bear_score
    net = bull_score - bear_score
    if total == 0:
        signal, confidence = "WAIT", 0
    else:
        confidence = round(max(bull_score, bear_score) / total * 100)
        if warnings:
            confidence = max(0, confidence - 10 * len(warnings))
        if net >= 5:
            signal = "STRONG_BUY"
        elif net >= 3:
            signal = "BUY"
        elif net <= -5:
            signal = "STRONG_SELL"
        elif net <= -3:
            signal = "SELL"
        else:
            signal = "WAIT"

    signal_result["bull_score"] = bull_score
    signal_result["bear_score"] = bear_score
    signal_result["signals"] = signals
    signal_result["signal"] = signal
    signal_result["confidence"] = confidence
    signal_result["zone_votes_applied"] = [
        {"direction": d, "weight": w, "reason": r} for d, w, r in votes
    ]
    return signal_result


def apply_extended_votes(
    votes: List[Vote],
    bull_score: int,
    bear_score: int,
    signals: list,
) -> Tuple[int, int]:
    """
    Fold extended-module votes into the existing bull_score/bear_score
    + signals accumulators used by strategy/signal_engine.py's
    generate(). Call this exactly the same way _apply_fib_scoring() is
    called there — same tuple contract, no other changes needed to the
    scoring model.
    """
    for direction, weight, reason in votes:
        if direction == "bullish":
            bull_score += weight
        elif direction == "bearish":
            bear_score += weight
        signals.append((direction, weight, reason))
    return bull_score, bear_score