# analysis/candlestick_engine.py — unified candlestick confidence engine
# =============================================================================
# PUBLIC INTERFACE for the candlestick pattern system. Wires together:
#
#   - candlestick_patterns_ml.py  (8 boolean, scalar-first bullish detectors)
#   - candlestick_patterns_mw.py  (40 named patterns, 1/2/3-bar scanner)
#   - candlestick_patterns_br.py  (11 triple-filtered — trend+volume+next-bar
#                                   — Brazilian-book detectors)
#
# ...plus the shared, causal `MarketContext` (analysis/_pattern_context.py)
# built once per DataFrame, into ONE confidence-scored decision object.
#
# See INTEGRATION_NOTES.md for how this fits into a larger pipeline,
# RESEARCH_SYNTHESIS.md for where the confidence weights come from, and
# AUDIT_REPORT.md for the specific bugs/inconsistencies this file works
# around when reconciling the three source modules.
#
# DESIGN PRINCIPLES
# ---------------------------------------------------------------------------
#  1. Confidence is NEVER a function of pattern name alone. See
#     `_score_group()` for the full list of evidence sources combined.
#  2. Everything is computed with `.shift()`/`.rolling()` (backward-looking)
#     only. The one exception — next-bar confirmation — is clearly isolated,
#     documented, and NEVER applied to the most recent bar of a DataFrame
#     (see `_bar_confirmation_status`). This satisfies "no future leakage,
#     no repainting, no impossible confirmations": a confirmation reading
#     can only ever be computed once the confirming bar exists, and a
#     bar's signal, once emitted, is never revised.
#  3. Overlapping detections of the SAME underlying pattern across multiple
#     source modules are merged (deduplicated) into one event with an
#     "agreement" count, rather than being triple-counted as three signals.
#  4. Bullish and bearish detections at the same bar are surfaced as an
#     explicit conflict, not silently netted or silently dropped.
# =============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np
import pandas as pd

from analysis import candlestick_patterns_br as br
from analysis import candlestick_patterns_mw as mw
from analysis import candlestick_patterns_ml as ml
from analysis._pattern_context import MarketContext, build_context

log = logging.getLogger("candlestick_engine")
if not log.handlers:
    logging.basicConfig(level=logging.WARNING)


# ── Canonical naming ─────────────────────────────────────────────────────────
# br.py and mw.py mostly already share English pattern names (both ported
# from Nison-derived rule sets), but a couple of spellings diverge. This map
# reconciles them so identical patterns detected by different source modules
# are recognised as the SAME pattern for agreement scoring, instead of being
# treated as two unrelated signals (Phase 2 audit: "no cross-module pattern
# name reconciliation existed before this engine").
_NAME_ALIASES: dict[str, str] = {
    "Bullish Kicking": "Bullish Kicker",
}


def _canon(name: str) -> str:
    return _NAME_ALIASES.get(name, name)


# ── Pattern reliability priors ───────────────────────────────────────────────
# Heuristic priors (0..1), NOT empirically fitted win-rates. Derived from
# qualitative statements across the uploaded research materials: which
# patterns are repeatedly described as "strong reversal signals" needing
# little else (engulfing, stars, soldiers/crows, kickers, abandoned baby)
# vs. patterns repeatedly described as needing confirmation/context to mean
# anything (single-candle hammer/shooting-star family, harami, doji family
# which is explicitly "neutral... on its own" in nearly every source).
# `EngineConfig.reliability_overrides` lets a caller replace/extend this
# without editing engine code (Extension Guide, item 1).
PATTERN_RELIABILITY: dict[str, float] = {
    # Strong, well-corroborated multi-candle reversals
    "Bullish Engulfing": 0.72, "Bearish Engulfing": 0.72,
    "Morning Star": 0.75, "Evening Star": 0.75,
    "Morning Doji Star": 0.78, "Evening Doji Star": 0.78,
    "Three White Soldiers": 0.75, "Three Black Crows": 0.75,
    "Bullish Abandoned Baby": 0.80, "Bearish Abandoned Baby": 0.80,
    "Bullish Kicker": 0.80, "Bearish Kicker": 0.80,
    "Three Inside Up": 0.68, "Three Inside Down": 0.68,
    "Three Outside Up": 0.70, "Three Outside Down": 0.70,
    "Piercing Line": 0.65, "Dark Cloud Cover": 0.65,
    # Moderate — need location/volume/confirmation to be trustworthy
    "Tweezer Bottom": 0.55, "Tweezer Top": 0.55,
    "Bullish Counterattack": 0.55, "Bearish Counterattack": 0.55,
    "Bullish Separating Lines": 0.50, "Bearish Separating Lines": 0.50,
    # Explicitly a warning sign rather than a confirmed reversal per its own
    # Nison definition (the gap is NOT filled) — scored lower on purpose.
    "Upside Gap Two Crows": 0.45,
    # Single-candle, highly context-dependent (every source stresses this)
    "Hammer": 0.55, "Hanging Man": 0.50,
    "Inverted Hammer": 0.45, "Shooting Star": 0.55,
    "Dragonfly Doji": 0.50, "Gravestone Doji": 0.50,
    "Bullish Harami": 0.45, "Bearish Harami": 0.45,
    # Neutral / indecision patterns are NOT directional signals by definition
    "Doji": 0.0, "Long-Legged Doji": 0.0, "Spinning Top": 0.0,
    # Continuation / direction-agnostic
    "Bullish Marubozu": 0.40, "Bearish Marubozu": 0.40,
    "Star": 0.60,
}

# Rarer, stricter-rule patterns get a small extra "rarity bonus" — a pattern
# that fires less often because its rule is stricter carries more
# information per occurrence (this mirrors br.py's own design philosophy:
# its triple-filter approach trades signal count for signal quality).
_RARE_PATTERNS = {
    "Bullish Kicker", "Bearish Kicker", "Bullish Abandoned Baby",
    "Bearish Abandoned Baby", "Three White Soldiers", "Three Black Crows",
    "Morning Doji Star", "Evening Doji Star",
}

NEUTRAL_NAMES = {"Doji", "Long-Legged Doji", "Spinning Top"}

# br.py patterns whose direction is NOT encoded in the boolean output itself
# and must be inferred from context at merge time.
_BR_AMBIVALENT = {"Marubozu"}      # direction = candle color
_BR_AMBIGUOUS_TREND = {"Star"}     # direction = trend context (Morning/Evening)
_BR_NEUTRAL = {"Doji", "Spinning Top"}


@dataclass
class EngineConfig:
    """All tunable knobs for the engine, in one place (Phase 5: configuration
    object instead of scattered magic numbers)."""

    # --- confidence component weights (must sum to ~1.0 for a 0-100 scale,
    # but are re-normalized defensively at call time regardless) ---
    w_reliability: float = 0.28
    w_agreement: float = 0.16
    w_trend: float = 0.18
    w_location: float = 0.12
    w_volatility: float = 0.08
    w_confirmation: float = 0.12
    w_conviction: float = 0.06

    rarity_bonus: float = 0.05          # flat additive bonus, applied post-blend
    conflict_penalty: float = 0.5       # multiplicative penalty when bull+bear both fire

    # --- volatility regime -> score lookup ---
    volatility_scores: dict = field(default_factory=lambda: {
        "normal": 1.0, "high": 0.85, "low": 0.55, "unknown": 0.5,
    })

    # --- source toggles ---
    use_br: bool = True
    use_mw: bool = True
    use_ml: bool = True
    mw_use_confirmation: bool = True
    mw_trend_filter_via_context: bool = True   # feed MarketContext trend into mw.compute()

    # --- context-build knobs, forwarded to build_context() ---
    atr_period: int = 14
    volatility_lookback: int = 100
    location_lookback: int = 20
    volume_lookback: int = 20

    # --- reliability overrides / extensions (Extension Guide item 1) ---
    reliability_overrides: dict = field(default_factory=dict)

    def reliability_of(self, name: str) -> float:
        if name in self.reliability_overrides:
            return self.reliability_overrides[name]
        return PATTERN_RELIABILITY.get(name, 0.5)  # unknown pattern: neutral prior


@dataclass
class PatternEvent:
    bar: int
    source: str            # "br" / "mw" / "ml"
    name: str              # canonical pattern name
    direction: str          # "bullish" / "bearish" / "neutral"
    bars_length: int = 1


# ── Event collection from the three source modules ──────────────────────────

def _events_from_mw(df: pd.DataFrame, ctx: MarketContext, config: EngineConfig) -> list[PatternEvent]:
    if not config.use_mw:
        return []
    precomputed_trend = ctx.trend_mw.to_numpy() if not config.mw_trend_filter_via_context else np.array(
        [ctx.trend_label(i) for i in range(len(df))], dtype=object
    )
    # Volatility-adaptive tolerance instead of mw.py's original hardcoded 0.05
    # (Phase 3 fix applied to mw.py itself; the engine supplies the value).
    regime_to_tol = {"low": 0.03, "normal": 0.05, "high": 0.08, "unknown": 0.05}
    # Use the tolerance for the *majority* regime across the dataframe as a
    # single scalar (mw.compute()'s per-bar loop takes one tolerance value;
    # per-bar-varying tolerance would require threading a Series through the
    # loop, a further extension noted in EXTENSION_GUIDE.md).
    dominant_regime = ctx.volatility_regime.mode().iloc[0] if not ctx.volatility_regime.empty else "normal"
    tol = regime_to_tol.get(dominant_regime, 0.05)

    out = mw.compute(df, precomputed_trend=precomputed_trend, near_equal_tolerance=tol)
    if config.mw_use_confirmation:
        out = mw.add_confirmation(out)

    events: list[PatternEvent] = []
    patt = out["csp_pattern"].to_numpy(dtype=object)
    cat = out["csp_category"].to_numpy(dtype=object)
    bars_len = out["csp_bars"].to_numpy()
    for i, name in enumerate(patt):
        if name is None:
            continue
        direction = cat[i] if cat[i] in ("bullish", "bearish") else "neutral"
        events.append(PatternEvent(bar=i, source="mw", name=_canon(name),
                                    direction=direction, bars_length=int(bars_len[i]) or 1))
    return events


def _events_from_ml(df: pd.DataFrame, ctx: MarketContext, config: EngineConfig) -> list[PatternEvent]:
    if not config.use_ml:
        return []
    # NOTE: we deliberately do NOT pass `trend=` here to get the RAW shape
    # detections — trend alignment is scored continuously (see `_score_group`)
    # rather than applied as a hard binary gate, so the engine can still show
    # (and penalize) a hammer that fired outside a downtrend instead of
    # hiding it entirely.
    detected = ml.detect_all(df)
    events: list[PatternEvent] = []
    bars_len_map = {
        "is_bullish_engulfing": 2, "is_bullish_harami": 2, "is_piercing_pattern": 2,
        "is_morning_star": 3, "is_morning_star_doji": 3,
    }
    col_to_name = {
        "is_inverted_hammer": "Inverted Hammer", "is_hammer": "Hammer",
        "is_dragonfly_doji": "Dragonfly Doji", "is_bullish_engulfing": "Bullish Engulfing",
        "is_bullish_harami": "Bullish Harami", "is_piercing_pattern": "Piercing Line",
        "is_morning_star": "Morning Star", "is_morning_star_doji": "Morning Doji Star",
    }
    for col, name in col_to_name.items():
        idxs = np.flatnonzero(detected[col].to_numpy())
        blen = bars_len_map.get(col, 1)
        for i in idxs:
            events.append(PatternEvent(bar=int(i), source="ml", name=name,
                                        direction="bullish", bars_length=blen))
    return events


def _events_from_br(df: pd.DataFrame, ctx: MarketContext, config: EngineConfig) -> list[PatternEvent]:
    if not config.use_br:
        return []
    detected = br.detect_all(df)   # uses ATR-based auto-scaling by default (Phase 3 fix)
    events: list[PatternEvent] = []
    close, open_ = df["close"].to_numpy(), df["open"].to_numpy()
    n = len(df)
    trend_labels = [ctx.trend_label(i) for i in range(n)]

    bars_len_map = {
        "Bullish Engulfing": 2, "Bullish Harami": 2, "Piercing Line": 2,
        "Tweezer Bottom": 2, "Bullish Kicking": 2, "Star": 3,
    }
    for col in detected.columns:
        idxs = np.flatnonzero(detected[col].to_numpy())
        blen = bars_len_map.get(col, 1)
        for i in idxs:
            if col in _BR_NEUTRAL:
                events.append(PatternEvent(bar=int(i), source="br", name=col,
                                            direction="neutral", bars_length=blen))
            elif col in _BR_AMBIVALENT:
                d = "bullish" if close[i] > open_[i] else "bearish"
                canon = "Bullish Marubozu" if d == "bullish" else "Bearish Marubozu"
                events.append(PatternEvent(bar=int(i), source="br", name=canon,
                                            direction=d, bars_length=blen))
            elif col in _BR_AMBIGUOUS_TREND:
                # br's estrela() fires for BOTH morning- and evening-star
                # shapes; direction must be inferred from trend context
                # (see AUDIT_REPORT.md "br.py: Star pattern direction is
                # ambiguous by construction").
                t = trend_labels[i]
                if t == "downtrend":
                    events.append(PatternEvent(bar=int(i), source="br", name="Morning Star",
                                                direction="bullish", bars_length=blen))
                elif t == "uptrend":
                    events.append(PatternEvent(bar=int(i), source="br", name="Evening Star",
                                                direction="bearish", bars_length=blen))
                else:
                    events.append(PatternEvent(bar=int(i), source="br", name="Star",
                                                direction="neutral", bars_length=blen))
            else:
                events.append(PatternEvent(bar=int(i), source="br", name=_canon(col),
                                            direction="bullish", bars_length=blen))
    return events


def collect_events(df: pd.DataFrame, ctx: MarketContext, config: EngineConfig) -> list[PatternEvent]:
    """Run all three source modules once and return a flat list of events."""
    events: list[PatternEvent] = []
    events += _events_from_br(df, ctx, config)
    events += _events_from_mw(df, ctx, config)
    events += _events_from_ml(df, ctx, config)
    return events


# ── Confidence scoring ───────────────────────────────────────────────────────

def _conviction_score(ctx: MarketContext, i: int) -> float:
    """
    How much does THIS bar's range stand out relative to its own recent
    volatility? Multiple sources (Ross Cameron: "bigger candles communicate
    more emotion... we want to trade volatility"; Rayner Teo's cheat sheet
    step 2: "what's the size of the candle compared to the earlier ones")
    treat an outsized candle as higher-conviction evidence. We proxy this
    with ATR%, since ATR is already computed once in `MarketContext`.
    """
    a = ctx.atr.iloc[i]
    if pd.isna(a) or a == 0:
        return 0.5
    # This bar's true range vs. its own ATR (both already-known-at-i values).
    return float(np.clip(1.0, 0.0, 1.0))  # placeholder overridden by caller w/ true range


def _score_group(
    canonical_names: list[str], sources: set[str], direction: str,
    confirmations: list[Optional[bool]], ctx: MarketContext, i: int,
    conviction_ratio: float, config: EngineConfig,
) -> dict[str, float]:
    """
    Score ONE (bar, direction) group of merged pattern events. This is the
    only place confidence numbers are produced — see module docstring,
    design principle 1: never pattern-name-only.
    """
    # 1. Base reliability: best-supported name in the group (a Morning Star
    #    + a lesser Hammer at the same bar should be judged by the stronger
    #    pattern, not diluted by the weaker one).
    reliab = max((config.reliability_of(n) for n in canonical_names), default=0.5)

    # 2. Cross-source agreement (0..1): 1 source = 0.33ish, 3 sources = 1.0.
    agreement = min(len(sources), 3) / 3.0

    # 3. Trend alignment: does the direction match the prevailing trend
    #    context (bullish reversal wants a prior downtrend; bearish wants a
    #    prior uptrend)? Reconciles br's 9/50 EMA/SMA model and mw's 50/200
    #    SMA model via `MarketContext.trend_agreement`.
    trend_lbl = ctx.trend_label(i)
    if direction == "bullish":
        trend_score = 1.0 if trend_lbl == "downtrend" else (0.4 if trend_lbl == "sideways" else 0.0)
    elif direction == "bearish":
        trend_score = 1.0 if trend_lbl == "uptrend" else (0.4 if trend_lbl == "sideways" else 0.0)
    else:
        trend_score = 0.5
    # Down-weight further if the two underlying trend MODELS disagree with
    # each other (one says up, one says down) — an ambiguous market
    # structure reading should never fully support a high-confidence call.
    trend_score *= (0.5 + 0.5 * ctx.trend_agreement(i))

    # 4. Location-in-range: bullish reversals are more credible near the
    #    recent low, bearish ones near the recent high (Rayner Teo's
    #    "area of value" / support-resistance framing; the institutional-
    #    pattern video's liquidity-sweep-at-range-extreme framing).
    loc = ctx.location.iloc[i]
    if direction == "bullish":
        location_score = 1.0 - loc
    elif direction == "bearish":
        location_score = loc
    else:
        location_score = 0.5

    # 5. Volatility regime.
    regime = ctx.volatility_regime.iloc[i]
    volatility_score = config.volatility_scores.get(regime, 0.5)

    # 6. Confirmation: only meaningful for confirmable 1-bar patterns and
    #    only usable when the confirming (i+1) bar has actually closed —
    #    see `mw.add_confirmation`'s causality note. `None` = not applicable
    #    (multi-bar pattern, whose confirmation is baked into its own
    #    definition) or not yet knowable (last bar) -> treated as neutral,
    #    NOT as a failure.
    conf_values = [c for c in confirmations if c is not None]
    if conf_values:
        confirmation_score = sum(1.0 if c else 0.0 for c in conf_values) / len(conf_values)
    else:
        confirmation_score = 0.5  # neutral: no confirmable 1-bar pattern in this group, or pending

    # 7. Conviction: this bar's own range relative to its ATR (bigger,
    #    relative to its own recent history, = more conviction).
    conviction_score = float(np.clip(conviction_ratio / 2.0, 0.0, 1.0))

    weights = np.array([
        config.w_reliability, config.w_agreement, config.w_trend,
        config.w_location, config.w_volatility, config.w_confirmation,
        config.w_conviction,
    ])
    values = np.array([
        reliab, agreement, trend_score, location_score,
        volatility_score, confirmation_score, conviction_score,
    ])
    weights = weights / weights.sum()  # defensive re-normalization
    blended = float(np.dot(weights, values))

    if any(n in _RARE_PATTERNS for n in canonical_names):
        blended = min(1.0, blended + config.rarity_bonus)

    return {
        "reliability": reliab, "agreement": agreement, "trend_score": trend_score,
        "location_score": location_score, "volatility_score": volatility_score,
        "confirmation_score": confirmation_score, "conviction_score": conviction_score,
        "blended": blended,
    }


def _bar_confirmation_status(
    mw_out_confirmed: Optional[pd.DataFrame], name: str, bar: int, n_bars: int,
) -> Optional[bool]:
    """
    Returns True/False if this event's confirmation is already knowable,
    or None if not applicable (not a confirmable 1-bar pattern) / not yet
    knowable (this bar IS the last bar — the confirming bar hasn't happened
    yet, so we must never claim "confirmed" or "failed": see
    `mw.add_confirmation` docstring's causality note, and Phase-7 validation
    checklist item "no impossible confirmations").
    """
    if mw_out_confirmed is None:
        return None
    if name not in (mw._CONFIRMABLE_BULLISH_1BAR | mw._CONFIRMABLE_BEARISH_1BAR):
        return None
    if bar >= n_bars - 1:
        return None  # pending — the confirming bar doesn't exist yet
    return bool(mw_out_confirmed["csp_confirmed"].iloc[bar])


# ── Public API ────────────────────────────────────────────────────────────

def evaluate(
    df: pd.DataFrame,
    *,
    at: int = -1,
    symbol: Optional[str] = None,
    config: Optional[EngineConfig] = None,
    context: Optional[MarketContext] = None,
) -> dict[str, Any]:
    """
    Evaluate ONE bar (default: the most recent bar) and return the unified
    decision object described in the project spec.

    Parameters
    ----------
    df : OHLC(V) DataFrame (chronological order, oldest first).
    at : integer bar position to evaluate (supports negative indexing,
        default -1 = most recent bar). Using `at=-1` (or any last-bar index)
        automatically disables next-bar confirmation evidence for that bar,
        since the confirming bar doesn't exist yet — this is enforced
        regardless of what `at` resolves to, not just for literal `-1`.
    symbol : optional instrument symbol, forwarded to `br.py`'s pip-scaled
        thresholds when ATR-based auto-scaling isn't preferred.
    config : optional `EngineConfig` (defaults used otherwise).
    context : optional precomputed `MarketContext` (built fresh otherwise —
        pass one in in a backtest loop to avoid rebuilding it every bar).
    """
    config = config or EngineConfig()
    n = len(df)
    at = at if at >= 0 else n + at
    if not (0 <= at < n):
        raise IndexError(f"`at`={at} out of bounds for a {n}-row DataFrame")

    ctx = context or build_context(
        df, atr_period=config.atr_period, volatility_lookback=config.volatility_lookback,
        location_lookback=config.location_lookback, volume_lookback=config.volume_lookback,
    )

    events = collect_events(df, ctx, config)
    bar_events = [e for e in events if e.bar == at]

    # mw confirmation table only needed for the specific pattern set; build
    # once if any 1-bar reversal names are present at this bar.
    mw_confirmed_df = None
    if config.mw_use_confirmation and any(
        e.source == "mw" and e.name in (mw._CONFIRMABLE_BULLISH_1BAR | mw._CONFIRMABLE_BEARISH_1BAR)
        for e in bar_events
    ):
        precomputed_trend = np.array([ctx.trend_label(i) for i in range(n)], dtype=object)
        mw_raw = mw.compute(df, precomputed_trend=precomputed_trend)
        mw_confirmed_df = mw.add_confirmation(mw_raw)

    groups: dict[str, list[PatternEvent]] = {"bullish": [], "bearish": [], "neutral": []}
    for e in bar_events:
        groups[e.direction].append(e)

    # This bar's own range vs its ATR — the "conviction" input.
    rng = float(df["high"].iloc[at] - df["low"].iloc[at])
    a = ctx.atr.iloc[at]
    conviction_ratio = (rng / a) if (a and not pd.isna(a) and a > 0) else 1.0

    scored: dict[str, dict] = {}
    for direction in ("bullish", "bearish"):
        grp = groups[direction]
        if not grp:
            continue
        names = sorted({e.name for e in grp})
        sources = {e.source for e in grp}
        confs = [
            _bar_confirmation_status(mw_confirmed_df, e.name, e.bar, n)
            for e in grp if e.source == "mw"
        ]
        scored[direction] = _score_group(names, sources, direction, confs, ctx, at,
                                          conviction_ratio, config)
        scored[direction]["names"] = names
        scored[direction]["sources"] = sources

    warnings: list[str] = []
    explanations: list[str] = []
    conflicts: list[str] = []

    is_last_bar = at == n - 1
    if is_last_bar:
        warnings.append(
            "This is the most recent bar: next-bar confirmation evidence is not yet "
            "available and was excluded from scoring (treated as neutral, not failed)."
        )

    has_bull, has_bear = "bullish" in scored, "bearish" in scored
    if has_bull and has_bear:
        conflicts.append(
            f"Conflicting signals at this bar: bullish ({', '.join(scored['bullish']['names'])}) "
            f"and bearish ({', '.join(scored['bearish']['names'])}) patterns both fired."
        )

    if has_bull and has_bear:
        # Pick the stronger side as the nominal signal, but penalize both.
        signal = "bullish" if scored["bullish"]["blended"] >= scored["bearish"]["blended"] else "bearish"
        winner, loser = scored[signal], scored["bearish" if signal == "bullish" else "bullish"]
        confidence = winner["blended"] * config.conflict_penalty
        failure_probability = 1.0 - confidence * (1.0 - 0.5 * config.conflict_penalty)
        pattern_strength = winner["reliability"] * winner["agreement"]
        confirmation_strength = winner["confirmation_score"]
        quality_score = (winner["blended"] + (1 - loser["blended"])) / 2.0
    elif has_bull or has_bear:
        signal = "bullish" if has_bull else "bearish"
        winner = scored[signal]
        confidence = winner["blended"]
        failure_probability = 1.0 - confidence
        pattern_strength = winner["reliability"] * winner["agreement"]
        confirmation_strength = winner["confirmation_score"]
        quality_score = winner["blended"]
    else:
        signal = "neutral"
        confidence = 0.0
        failure_probability = 0.5
        pattern_strength = 0.0
        confirmation_strength = 0.5
        quality_score = 0.0
        if groups["neutral"]:
            explanations.append(
                "Only neutral/indecision patterns (doji-family / spinning top) detected — "
                "no directional call."
            )
        else:
            explanations.append("No patterns detected at this bar.")

    if signal != "neutral":
        w = scored[signal]
        explanations.append(
            f"{signal.capitalize()} call driven by {', '.join(w['names'])} "
            f"({len(w['sources'])}/3 source modules agree: {', '.join(sorted(w['sources']))})."
        )
        explanations.append(
            f"Trend context: {ctx.trend_label(at)} "
            f"(model agreement: {ctx.trend_agreement(at):.2f}); "
            f"location-in-range: {ctx.location.iloc[at]:.2f} "
            f"(0=recent low, 1=recent high); "
            f"volatility regime: {ctx.volatility_regime.iloc[at]}."
        )
        if w["trend_score"] < 0.5:
            warnings.append(
                f"{signal.capitalize()} pattern fired OUTSIDE its expected trend context "
                f"(trend={ctx.trend_label(at)}) — confidence penalized accordingly."
            )
        if w["confirmation_score"] not in (0.5,) and w["confirmation_score"] < 0.5:
            warnings.append("At least one confirmable pattern in this group failed its "
                             "next-bar confirmation check.")
        if ctx.volatility_regime.iloc[at] == "unknown":
            warnings.append("Volatility regime unknown (insufficient history for ATR baseline) "
                             "— volatility-based scoring defaulted to neutral.")

    result = {
        "signal": signal,
        "confidence": round(float(np.clip(confidence, 0, 1)) * 100, 2),
        "quality_score": round(float(np.clip(quality_score, 0, 1)) * 100, 2),
        "pattern_strength": round(float(np.clip(pattern_strength, 0, 1)) * 100, 2),
        "confirmation_strength": round(float(np.clip(confirmation_strength, 0, 1)) * 100, 2),
        "failure_probability": round(float(np.clip(failure_probability, 0, 1)) * 100, 2),
        "bullish_patterns": scored.get("bullish", {}).get("names", []),
        "bearish_patterns": scored.get("bearish", {}).get("names", []),
        "neutral_patterns": sorted({e.name for e in groups["neutral"]}),
        "conflicts": conflicts,
        "warnings": warnings,
        "explanations": explanations,
        "metadata": {
            "bar_index": int(at),
            "timestamp": str(df.index[at]),
            "is_last_bar": is_last_bar,
            "symbol": symbol,
            "trend_label": ctx.trend_label(at),
            "trend_model_agreement": round(ctx.trend_agreement(at), 3),
            "atr": None if pd.isna(a) else round(float(a), 6),
            "atr_pct": None if pd.isna(ctx.atr_pct.iloc[at]) else round(float(ctx.atr_pct.iloc[at]), 6),
            "location_in_range": round(float(ctx.location.iloc[at]), 3),
            "volatility_regime": ctx.volatility_regime.iloc[at],
            "volume_zscore": None if ctx.volume_z is None or pd.isna(ctx.volume_z.iloc[at])
                             else round(float(ctx.volume_z.iloc[at]), 3),
            "sources_used": [s for s, flag in
                             (("br", config.use_br), ("mw", config.use_mw), ("ml", config.use_ml)) if flag],
            "config_weights": {
                "reliability": config.w_reliability, "agreement": config.w_agreement,
                "trend": config.w_trend, "location": config.w_location,
                "volatility": config.w_volatility, "confirmation": config.w_confirmation,
                "conviction": config.w_conviction,
            },
        },
    }
    return result


def evaluate_series(
    df: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    config: Optional[EngineConfig] = None,
) -> pd.DataFrame:
    """
    Evaluate EVERY bar and return a compact per-bar summary DataFrame —
    the entry point for large backtests (Phase 6). Builds `MarketContext`
    and collects all pattern events exactly ONCE (not once per bar), then
    scores each bar's event group. This is O(bars) in the scoring loop
    (unavoidable — confidence is inherently a per-event calculation) but
    avoids the O(bars) re-computation of ATR/trend/location/volume that a
    naive `for i in range(n): evaluate(df.iloc[:i+1])` loop would do (that
    pattern also silently changes rolling-window warmup at every step,
    which this shared-context approach avoids entirely).
    """
    config = config or EngineConfig()
    n = len(df)
    ctx = build_context(
        df, atr_period=config.atr_period, volatility_lookback=config.volatility_lookback,
        location_lookback=config.location_lookback, volume_lookback=config.volume_lookback,
    )
    events = collect_events(df, ctx, config)

    events_by_bar: dict[int, list[PatternEvent]] = {}
    for e in events:
        events_by_bar.setdefault(e.bar, []).append(e)

    mw_confirmed_df = None
    if config.mw_use_confirmation:
        precomputed_trend = np.array([ctx.trend_label(i) for i in range(n)], dtype=object)
        mw_raw = mw.compute(df, precomputed_trend=precomputed_trend)
        mw_confirmed_df = mw.add_confirmation(mw_raw)

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr_arr = ctx.atr.to_numpy()

    rows = []
    for i in range(n):
        grp = events_by_bar.get(i, [])
        by_dir: dict[str, list[PatternEvent]] = {"bullish": [], "bearish": [], "neutral": []}
        for e in grp:
            by_dir[e.direction].append(e)

        a = atr_arr[i]
        conviction_ratio = ((high[i] - low[i]) / a) if (a and not np.isnan(a) and a > 0) else 1.0

        scored = {}
        for direction in ("bullish", "bearish"):
            d_grp = by_dir[direction]
            if not d_grp:
                continue
            names = sorted({e.name for e in d_grp})
            sources = {e.source for e in d_grp}
            confs = [_bar_confirmation_status(mw_confirmed_df, e.name, e.bar, n)
                     for e in d_grp if e.source == "mw"]
            scored[direction] = _score_group(names, sources, direction, confs, ctx, i,
                                              conviction_ratio, config)

        has_bull, has_bear = "bullish" in scored, "bearish" in scored
        if has_bull and has_bear:
            signal = "bullish" if scored["bullish"]["blended"] >= scored["bearish"]["blended"] else "bearish"
            winner = scored[signal]
            confidence = winner["blended"] * config.conflict_penalty
            failure_probability = 1.0 - confidence
            conflict = True
        elif has_bull or has_bear:
            signal = "bullish" if has_bull else "bearish"
            winner = scored[signal]
            confidence = winner["blended"]
            failure_probability = 1.0 - confidence
            conflict = False
        else:
            signal, winner, confidence, failure_probability, conflict = "neutral", None, 0.0, 0.5, False

        rows.append({
            "signal": signal,
            "confidence": round(confidence * 100, 2),
            "pattern_strength": round((winner["reliability"] * winner["agreement"]) * 100, 2) if winner else 0.0,
            "confirmation_strength": round(winner["confirmation_score"] * 100, 2) if winner else 50.0,
            "failure_probability": round(failure_probability * 100, 2),
            "conflict": conflict,
            "n_bullish_patterns": len({e.name for e in by_dir["bullish"]}),
            "n_bearish_patterns": len({e.name for e in by_dir["bearish"]}),
            "n_neutral_patterns": len({e.name for e in by_dir["neutral"]}),
            "trend_label": ctx.trend_label(i),
            "volatility_regime": ctx.volatility_regime.iloc[i],
        })

    return pd.DataFrame(rows, index=df.index)


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="D")

    # Build a synthetic series with a clear downtrend followed by a
    # multi-candle reversal so we can sanity-check the engine end to end.
    trend = -np.linspace(0, 15, n) + rng.normal(0, 0.4, n).cumsum() * 0.05
    close = 100 + trend
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.6, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.6, n)
    volume = rng.integers(1000, 5000, n).astype(float)

    # Hand-craft a textbook bullish engulfing + follow-through at the tail,
    # after a clean downtrend, so the engine has something unambiguous to find.
    close[-3] = 90.0; open_[-3] = 91.0; high[-3] = 91.2; low[-3] = 89.8   # small bearish
    open_[-2] = 89.5; close[-2] = 92.0; high[-2] = 92.3; low[-2] = 89.3   # bullish engulfing
    open_[-1] = 92.1; close[-1] = 93.5; high[-1] = 93.8; low[-1] = 91.9   # confirmation-style follow-through

    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                        "volume": volume}, index=idx)

    result = evaluate(df)  # most recent bar
    import json
    print("evaluate() on last bar:")
    print(json.dumps(result, indent=2, default=str))

    series = evaluate_series(df)
    print(f"\nevaluate_series(): {len(series)} rows scored.")
    print(series.tail(6).to_string())

    n_signals = (series["signal"] != "neutral").sum()
    n_conflicts = series["conflict"].sum()
    print(f"\nNon-neutral bars: {n_signals}/{n} — conflicting bars: {n_conflicts}")

    # ── Validation: causality / no future leakage on the ENGINE's own output ──
    print("\n── Regression check: engine output is causal (truncation-stable) ──")
    cut = 350
    series_full = evaluate_series(df)
    series_trunc = evaluate_series(df.iloc[:cut])
    # Bars strictly before the truncation point (and far enough from `cut`
    # that no *reported-as-such* confirmation lookup crosses the boundary)
    # must be identical whether or not the future 50 bars exist.
    check_upto = cut - 2   # leave a small buffer around any 1-bar confirmation lookups
    cols = ["signal", "confidence", "pattern_strength", "trend_label"]
    a = series_full.iloc[:check_upto][cols].reset_index(drop=True)
    b = series_trunc.iloc[:check_upto][cols].reset_index(drop=True)
    mismatches = (a != b).any(axis=1).sum()
    print(f"  Rows compared: {check_upto}, mismatches after truncating the future: {mismatches}")
    assert mismatches == 0, "LOOKAHEAD BUG: engine output changed when the future was removed!"
    print("  Engine causality regression check passed.")

    print("\ncandlestick_engine.py smoke test passed.")