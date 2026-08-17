# analysis/unified_signal_engine.py
# ============================================================
# Unified Signal Engine — Connects All Strategy Engines
# ============================================================
# Orchestrates 5+ engines into one coherent system:
#   1. SupportResistance (zone base — shared)
#   2. HighReliabilityPatternDetector (pattern library — shared, now VOTES)
#   3. StopHuntSignalEngine (stop hunt reversal)
#   4. ICTAMDSignalEngine (ICT/SMC AMD+FVG+MSS, 1:6 R:R) — regime-gated
#   5. MultiStrategyPAEngine (8-step PA, session filter, MTF)
#   6. LiquidityPoolAnalyzer (sweep-reversal)
#   7. CCIStateMachine (mean-reversion at zones)
#
# ------------------------------------------------------------------
# INSTITUTIONAL REVIEW — CHANGE LOG (this revision)
# ------------------------------------------------------------------
# Fixes applied in THIS file (all issues raised in the review that are
# fixable without touching engine internals we don't have source for):
#
#   #2  Engine weights are no longer bare magic numbers — externalized to
#       `EngineWeights` (a dataclass), overridable per-symbol, and each
#       vote's effective weight is additionally scaled by expected value
#       (R:R) and market regime instead of being a flat constant.
#   #3  Detected high-reliability patterns now VOTE in consensus (bounded,
#       recency- and zone-gated) instead of only being able to force WAIT
#       via the consolidation override.
#   #4  Confidence is now a blended, explainable composite score (margin,
#       vote count, pattern confluence, average R:R) instead of two bare
#       thresholds (`vote_count>=2` / `score>=3`). See `_score_confidence`.
#   #5  Every sub-engine result now carries `engine_status` ∈
#       {"healthy","disabled","failed"} plus timing, and a top-level
#       `engine_health` block + explicit consensus reason distinguishes
#       "no setup found" from "N engines crashed" (no more silent NO_TRADE).
#   #6  Liquidity: no longer blindly takes "the most recent sweep." Ranks
#       all swept pools by (touches, recency) and requires a minimum pool
#       strength, so a strong 3-touch pool isn't discarded in favor of a
#       weak 2-touch pool that merely swept later. See `_select_best_sweep`.
#   #7  CCI zone selection now scans ALL nearby zones and picks the
#       strongest (touches, then strength label) instead of stopping at
#       the first match in iteration order.
#   #8  Pip sizing is now symbol-aware (`_pip_size`) — JPY crosses, metals,
#       indices, and crypto get correct pip/point sizes instead of a blind
#       ATR/20 approximation. ATR/20 is kept ONLY as a logged, explicit
#       last-resort fallback for unrecognized symbols.
#   #9  Consensus no longer flips on a 0.1-point score edge. A winning
#       side must clear both an absolute and a relative margin over the
#       other side (`ConsensusConfig.min_margin_abs/min_margin_ratio`) or
#       the result is NO_TRADE with the margin shown in the reason.
#  #10  The full per-engine vote trail (why each engine voted or
#       abstained) is now returned in `consensus["vote_trail"]`, not just
#       logged — visible to dashboards/LLM callers/audits.
#  #11  Weights are overridable per symbol via `pair_weight_overrides`
#       (e.g. tighter ICT weight on indices vs. majors) instead of one
#       static table for every pair.
#  #12  A lightweight, explicit regime classifier (`_detect_regime`) now
#       runs before the engines and (a) is passed into ICTAMDSignalEngine
#       as `regime_ctx` — that engine already has a documented hard-gate
#       contract for exactly this (`market_regime`/`strategy_type`/
#       `risk_multiplier`) that was simply never being fed live — and
#       (b) rescales every other engine's vote weight by regime.
#  #13  Engine "confidence" strings are mapped to numeric priors and
#       blended into a calibrated 0–1 `consensus["calibrated_score"]`
#       alongside the existing Low/Medium/High label. This is a heuristic
#       blend, NOT a statistically calibrated probability (see caveat in
#       `_score_confidence` docstring) — true calibration needs a labeled
#       historical dataset (Platt/isotonic scaling against realized
#       outcomes), which belongs in the backtest/DNA pipeline, not here.
#  #15  Consensus scoring now factors expected value via R:R, not just
#       vote count — a BUY vote backed by 1:6 R:R contributes more than
#       an otherwise-identical vote backed by 1:1.5 R:R.
#
# ------------------------------------------------------------------
# KNOWN LIMITATION — NOT fixed in this file (issue #1 / #14, shared layer)
# ------------------------------------------------------------------
# StopHuntSignalEngine, MultiStrategyPAEngine, and SupportResistance are
# NOT included in this upload (only unified_signal_engine.py,
# ict_amd_signal_engine.py, liquidity.py, cci_state_machine.py, and
# high_reliability_patterns.py were provided). Those three engines each
# still run their own internal S/R computation instead of consuming the
# `sr_zones_tagged` this file already computes once — that duplication
# cannot be safely removed from here without their source, because
# guessing at undocumented constructor/analyze() kwargs and passing them
# blindly would silently swallow into the existing broad `except Exception`
# handlers and misreport as "engine failed" rather than a clean interface
# error. The concrete, minimal fix once those files are available:
#   1. Add `precomputed_zones: Optional[list] = None` to each of
#      `StopHuntSignalEngine.analyze()`, `MultiStrategyPAEngine.analyze()`,
#      and have them skip their internal `SupportResistance(...).analyze()`
#      call when it's supplied.
#   2. Here, pass `precomputed_zones=sr_zones_tagged` to all three calls.
# ICTAMDSignalEngine already builds its own zones too (still duplicated
# CPU work), but it DOES already expose the `regime_ctx` hook this
# revision now wires up, which was the higher-value gap (a documented,
# built, but never-called safety gate).
# ============================================================

import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import json

from analysis.support_resistance import SupportResistance
from analysis.stop_hunt_signal_engine import StopHuntSignalEngine
from analysis.ict_amd_signal_engine import ICTAMDSignalEngine
from analysis.multi_strategy_pa_engine import MultiStrategyPAEngine
from analysis.high_reliability_patterns import (
    HighReliabilityPatternDetector,
    DetectedPattern,
)
from analysis.liquidity import LiquidityPoolAnalyzer
from analysis.cci_state_machine import CCIStateMachine

log = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────
MIN_CANDLES_REQUIRED = 30
SLOW_ENGINE_WARN_SECONDS = 2.0  # soft-timeout: we log, we do not (cannot,
# synchronously, without threads/processes) actually preempt a hung call —
# see `_run_engine` docstring for why this is a *reported* timeout, not an
# enforced one, and what enforcing one for real would require.

ENGINE_STATUS_HEALTHY = "healthy"
ENGINE_STATUS_DISABLED = "disabled"
ENGINE_STATUS_FAILED = "failed"

VALID_ACTIONS = ("BUY", "SELL")

# Symbol → pip/point size. This is a best-effort table for a *generic*
# multi-symbol engine and is NOT a substitute for broker-reported
# `SYMBOL_POINT` / digits (see mt5-technical review notes) — inject the
# real broker values via `UnifiedSignalEngine(pip_size_overrides={...})`
# in production rather than relying on these defaults for anything with
# real money behind it.
_JPY_PIP = 0.01
_METAL_PIP = {"XAU": 0.01, "XAG": 0.001}
_INDEX_PIP = 1.0
_CRYPTO_PIP = 1.0
_DEFAULT_FX_PIP = 0.0001
_INDEX_TOKENS = ("US30", "US500", "NAS100", "NDX", "SPX", "GER40", "DE40",
                  "UK100", "JP225", "AUS200", "FRA40", "DJI")
_CRYPTO_TOKENS = ("BTC", "ETH", "SOL", "XRP", "LTC", "DOGE")


# ─── Helpers (shared with other engines via _engine_utils) ─────
from analysis._engine_utils import atr_value as _atr


def _zones_to_unified(sr_zones: list, sd_zones: list = None,
                       trendline_zones: list = None) -> list:
    """
    Merge zones from multiple engines into a unified list with consistent schema.
    Each zone: {"type": str, "zone_top": float, "zone_bottom": float, "touches": int, "strength": str, "source": str}
    """
    unified = []

    # S/R zones
    for z in sr_zones or []:
        unified.append({
            "type": z.get("type") or ("resistance" if z.get("role") == "resistance" else "support"),
            "zone_top": float(z.get("zone_top", 0)),
            "zone_bottom": float(z.get("zone_bottom", 0)),
            "touches": int(z.get("touches", 0)),
            "strength": z.get("strength", "Weak"),
            "source": "SR",
        })

    # S/D zones
    for z in sd_zones or []:
        unified.append({
            "type": z.get("type", "supply"),  # supply or demand
            "zone_top": float(z.get("zone_top", 0)),
            "zone_bottom": float(z.get("zone_bottom", 0)),
            "touches": 0,
            "strength": "Medium",  # S/D zones are institutional
            "source": "SD",
        })

    # Trendline zones (if provided)
    for z in trendline_zones or []:
        unified.append({
            "type": "Trendline",
            "zone_top": float(z.get("zone_top", 0)),
            "zone_bottom": float(z.get("zone_bottom", 0)),
            "touches": int(z.get("touches", 0)),
            "strength": z.get("strength", "Medium"),
            "source": "Trendline",
        })

    return unified


_STRENGTH_RANK = {"Strong": 3, "Medium": 2, "Weak": 1}


def _zone_strength_rank(zone: dict) -> int:
    return _STRENGTH_RANK.get(zone.get("strength", "Weak"), 0)


def _pip_size(symbol: str, overrides: Optional[Dict[str, float]] = None) -> Optional[float]:
    """
    Symbol-aware pip/point size lookup (institutional review fix #8).

    Returns None (rather than guessing) when the symbol doesn't match any
    known convention — callers MUST have an explicit fallback path and
    MUST log when they take it, so an unrecognized symbol fails loudly in
    logs instead of silently trading on a wrong pip size.
    """
    if not symbol:
        return None
    sym = symbol.upper().replace("/", "").replace("_", "").replace("-", "")

    if overrides:
        if symbol.upper() in overrides:
            return overrides[symbol.upper()]
        if sym in overrides:
            return overrides[sym]

    for token in _INDEX_TOKENS:
        if token in sym:
            return _INDEX_PIP
    for token in _CRYPTO_TOKENS:
        if token in sym:
            return _CRYPTO_PIP
    for metal, pip in _METAL_PIP.items():
        if metal in sym:
            return pip
    if sym.endswith("JPY") or "JPY" in sym:
        return _JPY_PIP
    if len(sym) == 6 and sym.isalpha():
        return _DEFAULT_FX_PIP
    return None


def _extract_rr(signal: dict) -> float:
    """Pull a risk:reward figure out of a per-engine signal dict, tolerant
    of the different field names each engine uses. Defaults to 1.0 (i.e.
    "no R:R information, don't boost or penalize") rather than 0, so a
    missing field doesn't zero out an otherwise valid vote."""
    for key in ("risk_reward", "r_rr", "rr", "rrr"):
        val = signal.get(key)
        if val is not None:
            try:
                v = float(val)
                if v > 0:
                    return v
            except (TypeError, ValueError):
                pass
    return 1.0


# ─── Config (institutional review fixes #2, #11) ───────────────

@dataclass
class EngineWeights:
    """Base per-engine consensus weights — calibrated from M15 walk-forward
    backtests (2025-07 → 2026-07, GBPUSD/USDCAD/EURAUD/GBPCAD).

    Empirical findings (n≈160 trades):
      - Liquidity alone drove ~100% of signals but only ~41% WR → cut weight.
      - ICT/AMD sparse but higher quality when it fires → keep high.
      - StopHunt High-conf historically ~52% (report); overall ~37% → moderate.
      - CCI ~50% in limited sample → modest weight, needs confluence.
      - PA engine does not support M15 → effectively disabled on M15.
      - Patterns highly mixed (Tweezer Top 57%, Hammer 18%, Hanging Man 17%)
        → low base weight + allowlist of proven patterns only.

    2026-08-14 frequency pass → daily-trade pass:
      Target ≥1 actionable signal/day across active pairs.
      Liquidity still cannot solo (1.0 < min_action 1.15 when alone after
      conf_scale), but Liquidity+CCI / Liquidity+Pattern / StopHunt High
      / ICT clear the bar easily.
    """
    stop_hunt: float = 2.2
    ict_amd: float = 3.2
    pa: float = 1.4
    liquidity: float = 1.0           # pair with any other engine → trade
    cci: float = 1.2
    pattern: float = 0.85            # allowlist patterns contribute meaningfully


@dataclass
class ConsensusConfig:
    """Tunable consensus-layer thresholds.

    2026-08-14 recalibration (winrate audit) + frequency → daily pass:
      Goal: daily trades without pure-solo-Liquidity spam.
      min_action_score 1.15 → Liquidity(1.0) alone still blocked after
      conf_scale (~0.7–0.9), but any 2-engine combo clears.
      High can come from ICT alone or StopHunt-High alone.
    """
    min_action_score: float = 1.15      # Liquidity alone blocked; 2 engines clear
    min_margin_abs: float = 0.20
    min_margin_ratio: float = 1.06
    high_confidence_score: float = 2.4
    medium_confidence_score: float = 1.30
    rr_ev_weight: float = 0.18
    rr_ev_cap: float = 2.2
    max_pattern_votes: int = 2
    min_engines_for_high: int = 1
    min_calibrated_for_high: float = 0.58
    min_calibrated_for_medium: float = 0.36


# Regime → per-engine multiplier (institutional review fix #12).
# Trend-following engines (ICT, PA) get a boost in a trending regime and a
# discount in range/choppy; mean-reversion engines (CCI, Liquidity) get the
# opposite. CHOPPY caps everything low (and separately hard-blocks ICT via
# `regime_ctx`).
_REGIME_WEIGHT_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    # Frequency pass: mild uplift vs previous ultra-discount so more
    # regimes can still clear min_action_score without re-opening pure
    # Liquidity spam.
    "TRENDING": {"stop_hunt": 1.10, "ict_amd": 1.25, "pa": 1.20, "liquidity": 0.70, "cci": 0.75, "pattern": 1.00},
    "RANGING":  {"stop_hunt": 1.15, "ict_amd": 0.90, "pa": 1.00, "liquidity": 0.85, "cci": 1.25, "pattern": 1.05},
    "VOLATILE": {"stop_hunt": 0.90, "ict_amd": 0.85, "pa": 0.85, "liquidity": 0.65, "cci": 0.75, "pattern": 0.85},
    # Softened further so mean-reversion can fire in chop without ICT.
    "CHOPPY":   {"stop_hunt": 0.70, "ict_amd": 0.30, "pa": 0.60, "liquidity": 0.55, "cci": 0.95, "pattern": 0.50},
}

# Patterns allowed to vote (others are blocked). Derived from M15 WR audit:
# keep only those with WR ≥ ~40% + classic high-reliability, or borderline
# that historically helped confluence (frequency pass expanded slightly).
_PATTERN_VOTE_ALLOWLIST = {
    "Tweezer Top", "Three Inside Up", "Three Inside Down",
    "Bearish Engulfing", "Bullish Engulfing",
    "Bearish Harami", "Bullish Harami",
    "Evening Star", "Morning Star",
    "Shooting Star", "Dark Cloud Cover",
    "Three White Soldiers", "Three Black Crows",
    "Doji",  # high frequency, modest WR — only votes under confluence
}
# Patterns that actively hurt (WR < 25% in audit) — hard block.
_PATTERN_VOTE_BLOCKLIST = {
    "Hammer", "Hanging Man", "Piercing Line", "Inverted Hammer",
    "Tweezer Bottom",  # 27.9% WR in audit — surprisingly weak on M15
}

_CONFIDENCE_PRIOR = {"High": 0.85, "Medium": 0.6, "Low": 0.35}


# ─── Main Unified Engine ──────────────────────────────────────

class UnifiedSignalEngine:
    """
    Master orchestrator — connects all strategy engines into one system.

    Usage:
        engine = UnifiedSignalEngine(timeframe="4H")
        result = engine.analyze(df, symbol="EURUSD", lower_tf_df=lower_df)
        print(json.dumps(result, indent=2))
    """

    def __init__(
        self,
        timeframe: str = "4H",
        swing_window: Optional[int] = None,
        cluster_threshold_pct: Optional[float] = None,
        min_touches: int = 2,
        # Strategy-specific config
        enable_stop_hunt: bool = True,
        enable_ict_amd: bool = True,
        enable_pa: bool = True,
        enable_patterns: bool = True,
        enable_liquidity: bool = True,
        enable_cci: bool = True,
        # R:R thresholds
        ict_min_rr: float = 6.0,
        pa_min_rr: float = 2.0,
        liquidity_min_rr: float = 2.0,
        cci_min_rr: float = 2.0,   # was 1.5 — raise for positive EV at ~45% WR
        # Pattern lookback
        pattern_lookback: int = 20,
        # ── Institutional review additions ──
        engine_weights: Optional[EngineWeights] = None,
        consensus_config: Optional[ConsensusConfig] = None,
        pair_weight_overrides: Optional[Dict[str, EngineWeights]] = None,
        pip_size_overrides: Optional[Dict[str, float]] = None,
        slow_engine_warn_seconds: float = SLOW_ENGINE_WARN_SECONDS,
    ):
        self.timeframe = timeframe.upper()
        self.enable_stop_hunt = enable_stop_hunt
        self.enable_ict_amd = enable_ict_amd
        self.enable_pa = enable_pa
        self.enable_patterns = enable_patterns
        self.enable_liquidity = enable_liquidity
        self.enable_cci = enable_cci
        self.liquidity_min_rr = liquidity_min_rr
        self.cci_min_rr = cci_min_rr

        self.weights = engine_weights or EngineWeights()
        self.consensus_config = consensus_config or ConsensusConfig()
        self.pair_weight_overrides = pair_weight_overrides or {}
        self.pip_size_overrides = pip_size_overrides or {}
        self.slow_engine_warn_seconds = slow_engine_warn_seconds

        # ── Shared S/R engine (base for all strategies) ──
        self.sr_engine = SupportResistance(
            timeframe=timeframe,
            swing_window=swing_window,
            cluster_threshold_pct=cluster_threshold_pct,
            min_touches=min_touches,
            wick_body_ratio=1.5,
            max_zones_per_side=10,
        )

        # ── Strategy engines ──
        self.stop_hunt_engine = StopHuntSignalEngine(
            timeframe=timeframe,
            swing_window=swing_window,
            cluster_threshold_pct=cluster_threshold_pct,
            min_touches=min_touches,
        )
        self.ict_engine = ICTAMDSignalEngine(
            timeframe=timeframe,
            swing_window=swing_window,
            cluster_threshold_pct=cluster_threshold_pct,
            min_touches=min_touches,
            min_rr_ratio=ict_min_rr,
        )
        self.pa_engine = MultiStrategyPAEngine(
            timeframe=timeframe,
            swing_window=swing_window,
            cluster_threshold_pct=cluster_threshold_pct,
            min_touches=min_touches,
        )
        self.liquidity_engine = LiquidityPoolAnalyzer()
        self.cci_engine = CCIStateMachine()
        self.pattern_detector = HighReliabilityPatternDetector(
            lookback=pattern_lookback,
        )

        # Detect once whether ICTAMDSignalEngine.analyze() accepts a
        # regime_ctx kwarg, rather than assuming a fixed version — this is
        # the *safe* version of the "try to pass shared context" pattern
        # described in the module docstring: we introspect the real,
        # available signature instead of guessing at undocumented kwargs
        # for engines we don't have source for.
        try:
            self._ict_accepts_regime_ctx = (
                "regime_ctx" in inspect.signature(self.ict_engine.analyze).parameters
            )
        except (TypeError, ValueError):
            self._ict_accepts_regime_ctx = False

    # ═══════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        lower_tf_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Run all enabled engines and produce unified output.

        Args:
            df: OHLC DataFrame (primary timeframe)
            symbol: e.g., "EURUSD"
            lower_tf_df: lower TF OHLC for MTF confirmation (H2 for 4H, M30 for 1H)

        Returns:
            Unified dict with all engine outputs + consensus signal.
        """
        # ── Edge case: insufficient data ──
        if df is None or len(df) < MIN_CANDLES_REQUIRED:
            return self._insufficient_data_result(symbol)

        sym = symbol.upper()
        engine_timings: Dict[str, float] = {}
        engine_statuses: Dict[str, str] = {}

        # ── SHARED LAYER: compute once, reuse ──
        atr_val = _atr(df, period=14)

        # S/R Zones (shared)
        t0 = time.monotonic()
        try:
            sr_result = self.sr_engine.analyze(df, symbol=sym)
            sr_zones_raw = sr_result.get("resistance_zones", []) + sr_result.get("support_zones", [])
            engine_statuses["support_resistance"] = ENGINE_STATUS_HEALTHY
        except Exception as e:
            log.error(f"[Unified] S/R analyze failed: {e}", exc_info=True)
            sr_result = {}
            sr_zones_raw = []
            engine_statuses["support_resistance"] = ENGINE_STATUS_FAILED
        engine_timings["support_resistance"] = time.monotonic() - t0

        # Build unified zone list (for pattern confluence + signal sharing)
        sr_zones_tagged = []
        for z in sr_result.get("resistance_zones", []):
            sr_zones_tagged.append({**z, "type": "resistance"})
        for z in sr_result.get("support_zones", []):
            sr_zones_tagged.append({**z, "type": "support"})
        unified_zones = _zones_to_unified(sr_zones_tagged)

        # ── REGIME (institutional review fix #12) ──
        regime_ctx = self._detect_regime(df, atr_val)
        regime_label = regime_ctx["market_regime"]
        regime_mult = _REGIME_WEIGHT_MULTIPLIERS.get(
            regime_label, _REGIME_WEIGHT_MULTIPLIERS["RANGING"]
        )

        # ── PATTERNS (shared) ──
        detected_patterns: List[DetectedPattern] = []
        pattern_dicts = []
        pattern_repetition = {"zone_strength_boosts": [], "momentum_sequence": None, "consolidation_detected": False}
        if self.enable_patterns:
            t0 = time.monotonic()
            try:
                detected_patterns = self.pattern_detector.detect(
                    df, zones=unified_zones, atr_value=atr_val
                )
                pattern_dicts = [p.to_spec_dict() for p in detected_patterns]
                pattern_repetition = self.pattern_detector.analyze_repetition(detected_patterns)
                engine_statuses["patterns"] = ENGINE_STATUS_HEALTHY
            except Exception as e:
                log.error(f"[Unified] Pattern detection failed: {e}", exc_info=True)
                engine_statuses["patterns"] = ENGINE_STATUS_FAILED
            engine_timings["patterns"] = time.monotonic() - t0
        else:
            engine_statuses["patterns"] = ENGINE_STATUS_DISABLED

        # ── STRATEGY ENGINES ──
        stop_hunt_result, engine_statuses["stop_hunt"], engine_timings["stop_hunt"] = self._run_engine(
            "StopHunt", self.enable_stop_hunt,
            lambda: self.stop_hunt_engine.analyze(df, symbol=sym),
            self._fallback_stop_hunt,
        )

        # ICT/AMD: now fed the live regime context it was built to accept
        # but was never being given (institutional review fix #12).
        def _run_ict():
            if self._ict_accepts_regime_ctx:
                return self.ict_engine.analyze(df, symbol=sym, regime_ctx=regime_ctx)
            return self.ict_engine.analyze(df, symbol=sym)

        ict_result, engine_statuses["ict_amd"], engine_timings["ict_amd"] = self._run_engine(
            "ICT/AMD", self.enable_ict_amd, _run_ict, self._fallback_ict,
        )

        pa_result, engine_statuses["pa"], engine_timings["pa"] = self._run_engine(
            "PA", self.enable_pa,
            lambda: self.pa_engine.analyze(df, symbol=sym, lower_tf_df=lower_tf_df),
            lambda reason="PA engine failed": self._fallback_pa(sym, reason=reason),
        )

        liquidity_result, engine_statuses["liquidity"], engine_timings["liquidity"] = self._run_engine(
            "Liquidity", self.enable_liquidity,
            lambda: self._analyze_liquidity(df, atr_val),
            self._fallback_liquidity,
        )

        cci_result, engine_statuses["cci"], engine_timings["cci"] = self._run_engine(
            "CCI", self.enable_cci,
            lambda: self._analyze_cci(df, sym, sr_zones_tagged, atr_val),
            self._fallback_cci,
        )

        # Tag every result with its own status (issue #5) so a consumer
        # reading `result["ict_amd"]` in isolation can still tell healthy
        # vs. disabled vs. failed without cross-referencing engine_health.
        for res, status in (
            (stop_hunt_result, engine_statuses["stop_hunt"]),
            (ict_result, engine_statuses["ict_amd"]),
            (pa_result, engine_statuses["pa"]),
            (liquidity_result, engine_statuses["liquidity"]),
            (cci_result, engine_statuses["cci"]),
        ):
            if isinstance(res, dict):
                res["engine_status"] = status

        # ── CONSENSUS SIGNAL ──
        weights = self._weights_for_symbol(sym)
        consensus = self._compute_consensus(
            stop_hunt_result, ict_result, pa_result,
            detected_patterns, pattern_repetition,
            liquidity_result=liquidity_result, cci_result=cci_result,
            weights=weights, regime_mult=regime_mult, regime_label=regime_label,
            engine_statuses=engine_statuses, df_len=len(df),
        )

        # ── BUILD UNIFIED OUTPUT ──
        return self._build_unified_result(
            symbol=sym,
            timeframe=self.timeframe,
            atr=atr_val,
            current_price=float(df["close"].iloc[-1]),
            sr_zones=sr_zones_tagged,
            unified_zones=unified_zones,
            detected_patterns=pattern_dicts,
            pattern_repetition=pattern_repetition,
            stop_hunt_result=stop_hunt_result,
            ict_result=ict_result,
            pa_result=pa_result,
            liquidity_result=liquidity_result,
            cci_result=cci_result,
            consensus=consensus,
            regime_ctx=regime_ctx,
            engine_statuses=engine_statuses,
            engine_timings=engine_timings,
        )

    # ═══════════════════════════════════════════════════════════
    # GENERIC ENGINE RUNNER (institutional review fix #5)
    # ═══════════════════════════════════════════════════════════

    def _run_engine(self, name: str, enabled: bool, fn, fallback_fn):
        """
        Uniformly run a sub-engine call with status tracking.

        Returns (result_dict, status, elapsed_seconds).

        NOTE on "timeout" (review issue #5's spirit, extended): this is a
        *soft* timeout — we measure elapsed wall time and log a WARNING if
        it exceeds `slow_engine_warn_seconds`, but a synchronous Python
        function call cannot be forcibly preempted without running it in a
        separate thread/process and killing that on a deadline. If a hard
        timeout is required (e.g. a sub-engine can hang on a network call),
        wrap `fn` with `concurrent.futures.ThreadPoolExecutor.submit(...).
        result(timeout=...)` at the call site — deliberately not done here
        so the (frequent) fast path doesn't pay thread-pool overhead.
        """
        if not enabled:
            return fallback_fn(reason=f"{name} engine disabled"), ENGINE_STATUS_DISABLED, 0.0

        t0 = time.monotonic()
        try:
            result = fn()
            elapsed = time.monotonic() - t0
            if elapsed > self.slow_engine_warn_seconds:
                log.warning(
                    f"[Unified] {name} engine took {elapsed:.2f}s "
                    f"(soft-timeout threshold {self.slow_engine_warn_seconds:.2f}s)"
                )
            return result, ENGINE_STATUS_HEALTHY, elapsed
        except Exception as e:
            elapsed = time.monotonic() - t0
            log.error(f"[Unified] {name} engine failed after {elapsed:.2f}s: {e}", exc_info=True)
            return (
                fallback_fn(reason=f"{name} engine raised {type(e).__name__}: {e}"),
                ENGINE_STATUS_FAILED,
                elapsed,
            )

    def _weights_for_symbol(self, symbol: str) -> EngineWeights:
        """Institutional review fix #11 — per-pair weight overrides."""
        return self.pair_weight_overrides.get(symbol, self.weights)

    # ═══════════════════════════════════════════════════════════
    # REGIME DETECTION (institutional review fix #12)
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _detect_regime(df: pd.DataFrame, atr_val: float, lookback: int = 30) -> dict:
        """
        Lightweight regime classifier producing the exact contract
        ICTAMDSignalEngine.analyze(regime_ctx=...) already expects:
        {"market_regime": ..., "strategy_type": ..., "risk_multiplier": ...}.

        This is a lookback linear-regression trend/range/volatility read,
        NOT the full `analysis/market_regime.py` module that
        ICTAMDSignalEngine's docstring references (that module was not
        included in this upload — the interface contract in its code is
        what's implemented here; swap this for the real
        `MarketRegimeDetector` when that file is available, since it may
        also fold in session/news context this simplified version doesn't).

        Classification:
          - atr_pct = ATR / price. High atr_pct + low trend R² => VOLATILE
            or CHOPPY (whipsaw, not tradeable) depending on how extreme.
          - Otherwise: strong linear fit (R² and slope) => TRENDING,
            weak fit => RANGING.
        """
        close = df["close"].astype(float)
        price = float(close.iloc[-1]) if len(close) else 0.0
        atr_pct = (atr_val / price) if price else 0.0

        n = min(lookback, len(close) - 1) if len(close) > 1 else 0
        r2 = 0.0
        slope_pct = 0.0
        if n >= 5:
            window = close.iloc[-n:].values
            x = np.arange(n)
            try:
                slope, intercept = np.polyfit(x, window, 1)
                fitted = slope * x + intercept
                ss_res = float(np.sum((window - fitted) ** 2))
                ss_tot = float(np.sum((window - window.mean()) ** 2))
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                slope_pct = (slope * n) / window.mean() if window.mean() else 0.0
            except (np.linalg.LinAlgError, ValueError):
                r2, slope_pct = 0.0, 0.0

        # Thresholds are heuristic starting points — tune against realized
        # forward volatility/whipsaw for the instruments actually traded.
        if atr_pct >= 0.010 and r2 < 0.35:
            regime, strategy_type, risk_mult = "CHOPPY", "NO_TRADE", 0.0
        elif atr_pct >= 0.010:
            regime, strategy_type, risk_mult = "VOLATILE", "MEAN_REVERSION", 0.6
        elif r2 >= 0.5 and abs(slope_pct) >= 0.003:
            regime, strategy_type, risk_mult = "TRENDING", "TREND_FOLLOWING", 1.0
        else:
            regime, strategy_type, risk_mult = "RANGING", "MEAN_REVERSION", 0.85

        return {
            "market_regime": regime,
            "strategy_type": strategy_type,
            "risk_multiplier": risk_mult,
            "atr_pct": round(atr_pct, 5),
            "trend_r2": round(r2, 3),
            "slope_pct": round(slope_pct, 5),
        }

    # ═══════════════════════════════════════════════════════════
    # LIVE LIQUIDITY + CCI ANALYSIS (single-bar, "now" only)
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _select_best_sweep(pools: list, recency_window: int = 15) -> Optional[dict]:
        """
        Institutional review fix #6.

        Previously: `LiquidityPoolAnalyzer._most_recent_sweep` (inside
        liquidity.py, not modified here) always returns the sweep with the
        highest `last_index`, ignoring pool strength (`touches`) entirely —
        a fresh weak 2-touch sweep would always beat a strong 4-touch sweep
        from a few bars earlier.

        Fix: rank all swept pools within a recency window by
        (touches desc, last_index desc) instead of recency alone, so a
        strong pool that swept slightly earlier still wins over a weak
        pool that swept most recently. Falls back to the single most
        recent sweep if nothing is within the recency window (keeps prior
        behavior as a floor rather than returning nothing).
        """
        swept = [p for p in pools if p.get("swept")]
        if not swept:
            return None

        max_index = max(p["last_index"] for p in swept)
        in_window = [p for p in swept if (max_index - p["last_index"]) <= recency_window]
        candidates = in_window or swept

        best = max(candidates, key=lambda p: (p.get("touches", 0), p["last_index"]))
        direction = "BULLISH_REVERSAL_LIKELY" if best["kind"] == "low" else "BEARISH_REVERSAL_LIKELY"
        return {
            "price": best["price"],
            "kind": best["kind"],
            "touches": best.get("touches", 0),
            "implication": direction,
            "note": (
                f"Strongest recent sweep: liquidity below {best['price']:.5f} "
                f"({best.get('touches', 0)} touches) was swept — possible bullish reversal"
                if best["kind"] == "low" else
                f"Strongest recent sweep: liquidity above {best['price']:.5f} "
                f"({best.get('touches', 0)} touches) was swept — possible bearish reversal"
            ),
        }

    def _analyze_liquidity(self, df: pd.DataFrame, atr_val: float) -> dict:
        """Live liquidity-sweep-reversal signal for the current bar, using
        the strongest recent sweep (fix #6) rather than merely the latest."""
        res = self.liquidity_engine.analyze(df)
        if not res.get("valid"):
            return self._fallback_liquidity(reason="No valid liquidity pools")

        sweep = self._select_best_sweep(res.get("pools", []))
        if not sweep:
            return self._fallback_liquidity(reason="No recent sweep detected")

        kind = sweep.get("kind")
        if kind == "high":
            action = "SELL"
        elif kind == "low":
            action = "BUY"
        else:
            return self._fallback_liquidity(reason=f"Unrecognized sweep kind: {kind}")

        if atr_val <= 0:
            return self._fallback_liquidity(reason="ATR unavailable")

        entry = float(df["close"].iloc[-1])
        pool_price = sweep.get("price", entry)
        if action == "BUY":
            stop = min(pool_price, entry) - 1.0 * atr_val
            tp = entry + abs(entry - stop) * self.liquidity_min_rr
        else:
            stop = max(pool_price, entry) + 1.0 * atr_val
            tp = entry - abs(entry - stop) * self.liquidity_min_rr

        # Profit-safe confidence (2026-08-15 audit):
        # Historical liquidity-driven signals ~41% WR. Never label High —
        # High is reserved for ICT / StopHunt-High which have proven edge.
        # Strong pools (4+ touches) → Medium; 2-3 → Low; <2 → abstain path.
        touches = int(sweep.get("touches", 0) or 0)
        if touches >= 4:
            confidence = "Medium"
        elif touches >= 2:
            confidence = "Low"
        else:
            return self._fallback_liquidity(
                reason=f"Sweep pool too weak ({touches} touches < 2)"
            )

        return {
            "valid": True,
            "recent_sweep": sweep,
            "signal": {
                "action": action, "entry_price": round(entry, 5),
                "stop_loss": round(stop, 5), "take_profit": round(tp, 5),
                "risk_reward": self.liquidity_min_rr, "r_rr": self.liquidity_min_rr,
                "reason": f"Liquidity sweep of {kind} at {pool_price} ({touches} touches)",
                "confidence": confidence,
            },
        }

    def _analyze_cci(self, df: pd.DataFrame, symbol: str, sr_zones_tagged: list, atr_val: float) -> dict:
        """Live CCI-state-machine signal for the current bar. Zone
        selection now scans all candidate zones and picks the strongest
        instead of stopping at the first match (fix #7); pip proximity now
        uses a symbol-aware pip size instead of a blind ATR/20 proxy
        (fix #8)."""
        cci = self._compute_cci(df, period=20)
        if cci is None or len(cci) == 0 or atr_val <= 0:
            return self._fallback_cci(reason="CCI unavailable")

        close = float(df["close"].iloc[-1])

        pip = _pip_size(symbol, self.pip_size_overrides)
        if pip is None:
            pip = atr_val / 20
            log.warning(
                f"[Unified] CCI: no known pip size for symbol='{symbol}', "
                f"falling back to ATR/20 approximation ({pip:.6f}). "
                f"Add this symbol to pip_size_overrides for accurate zone proximity."
            )
        proximity = 20 * pip

        # Scan ALL candidate zones, keep the strongest match (fix #7).
        best_zone = None
        best_zone_type = None
        for z in sr_zones_tagged:
            z_type = z.get("type")
            if z_type == "support" and abs(close - z.get("zone_top", 0)) < proximity:
                candidate_zone_type = "demand"
            elif z_type == "resistance" and abs(close - z.get("zone_bottom", 0)) < proximity:
                candidate_zone_type = "supply"
            else:
                continue
            if best_zone is None or _zone_strength_rank(z) > _zone_strength_rank(best_zone) or (
                _zone_strength_rank(z) == _zone_strength_rank(best_zone)
                and z.get("touches", 0) > best_zone.get("touches", 0)
            ):
                best_zone = z
                best_zone_type = candidate_zone_type

        if best_zone_type is None:
            return self._fallback_cci(reason="Price not near a scored S/R zone")

        cci_val = float(cci[-1])
        sig = self.cci_engine.evaluate(
            cci_value=cci_val, zone_type=best_zone_type,
            position=None, trend_align=True, at_zone=True,
        )
        if sig.action != "ENTER":
            return self._fallback_cci(reason=f"CCI state machine: {sig.action}")

        action = "BUY" if sig.direction == "long" else "SELL"
        stop = close - 1.5 * atr_val if action == "BUY" else close + 1.5 * atr_val
        tp = close + 1.5 * atr_val * self.cci_min_rr if action == "BUY" \
            else close - 1.5 * atr_val * self.cci_min_rr

        confluence = int(getattr(sig, "confluence_score", 0) or 0)
        zone_strength = str(best_zone.get("strength", "Weak") or "Weak")
        # Profit-safe CCI confidence (2026-08-15 audit):
        # Overall CCI WR ~35%. Only Strong zone + full confluence (3) may
        # claim High. Medium needs confluence≥2 at Strong/Medium zone.
        # Weak zones never vote above Low — they historically drag EV negative.
        if confluence >= 3 and zone_strength == "Strong":
            confidence = "High"
        elif confluence >= 2 and zone_strength in ("Strong", "Medium"):
            confidence = "Medium"
        elif confluence >= 1:
            confidence = "Low"
        else:
            return self._fallback_cci(reason="CCI confluence too weak")

        return {
            "valid": True,
            "cci_value": round(cci_val, 2),
            "zone_type": best_zone_type,
            "zone_strength": zone_strength,
            "signal": {
                "action": action, "entry_price": round(close, 5),
                "stop_loss": round(stop, 5), "take_profit": round(tp, 5),
                "risk_reward": self.cci_min_rr, "r_rr": self.cci_min_rr,
                "reason": f"CCI={cci_val:.1f} at {best_zone_type} zone "
                          f"({zone_strength}, {best_zone.get('touches', 0)} touches, conf={confluence}/3)",
                "confidence": confidence,
            },
        }

    @staticmethod
    def _compute_cci(df: pd.DataFrame, period: int = 20):
        """Compute CCI indicator."""
        try:
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)
            typical = (high + low + close) / 3
            sma = typical.rolling(period).mean()
            mad = typical.rolling(period).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            cci = (typical - sma) / (0.015 * mad)
            return cci.values
        except Exception as e:
            log.warning(f"[Unified] CCI computation failed: {e}")
            return None

    # ═══════════════════════════════════════════════════════════
    # PATTERN VOTES (institutional review fix #3)
    # ═══════════════════════════════════════════════════════════

    def _pattern_votes(
        self, detected_patterns: List[DetectedPattern], df_len: int, weights: EngineWeights,
        regime_mult: Dict[str, float],
    ) -> Tuple[List[Tuple[str, float, str, str]], List[str]]:
        """
        Turn high-confluence patterns into bounded consensus votes instead
        of letting pattern detection only ever suppress trades (via the
        consolidation override) and never confirm one.

        Gated deliberately narrow, since pattern-only "votes" are the
        weakest-evidence signal in the stack:
          - must be `near_zone` (pattern without S/R confluence is noise)
          - must have a directional `direction` (bullish/bearish; neutral
            patterns like Doji don't vote)
          - must be on the most recent 1-2 candles (a pattern from 15 bars
            ago in the lookback window is not "now")
          - capped at `max_pattern_votes` so a cluster of low-grade
            patterns can't out-vote the real strategy engines
        """
        trail = []
        votes = []
        reliability_mult = {"High": 1.0, "Medium": 0.55, "Low": 0.25}
        eligible = []
        for p in detected_patterns:
            name = getattr(p, "pattern_name", "") or ""
            if name in _PATTERN_VOTE_BLOCKLIST:
                trail.append(f"Pattern:{name}=blocked abstained: blocklisted (historically weak WR)")
                continue
            if name not in _PATTERN_VOTE_ALLOWLIST and getattr(p, "reliability", "") != "High":
                trail.append(f"Pattern:{name}=skipped abstained: not in allowlist and not High reliability")
                continue
            if not (p.near_zone and p.direction in ("bullish", "bearish") and p.candle_index >= df_len - 2):
                continue
            eligible.append(p)
        if not eligible:
            trail.append("Patterns=none_eligible abstained: no directional, zone-confluent, recent, allowed pattern")
            return votes, trail

        # Highest reliability first, cap count.
        eligible.sort(key=lambda p: reliability_mult.get(p.reliability, 0), reverse=True)
        for p in eligible[: self.consensus_config.max_pattern_votes]:
            action = "BUY" if p.direction == "bullish" else "SELL"
            w = weights.pattern * reliability_mult.get(p.reliability, 0.25) * regime_mult.get("pattern", 1.0)
            votes.append((action, w, p.reliability, f"Pattern:{p.pattern_name}"))
            trail.append(f"Pattern:{p.pattern_name}={action}({p.reliability}) voted weight={w:.2f}")
        return votes, trail

    # ═══════════════════════════════════════════════════════════
    # CONSENSUS SIGNAL (voting across engines)
    # ═══════════════════════════════════════════════════════════

    def _compute_consensus(
        self,
        stop_hunt_result: Optional[dict],
        ict_result: Optional[dict],
        pa_result: Optional[dict],
        detected_patterns: List[DetectedPattern],
        pattern_repetition: dict,
        liquidity_result: Optional[dict],
        cci_result: Optional[dict],
        weights: EngineWeights,
        regime_mult: Dict[str, float],
        regime_label: str,
        engine_statuses: Dict[str, str],
        df_len: int,
    ) -> dict:
        """
        Weighted, regime- and R:R-aware voting consensus across all engines.

        Rules:
          - Each engine that produces BUY/SELL casts a vote whose effective
            weight = base_weight * regime_multiplier * rr_multiplier
            (fixes #2, #12, #15 — flat hardcoded weights ignored regime
            and expected value entirely).
          - High-reliability, zone-confluent, recent patterns also vote,
            capped and down-weighted by reliability (fix #3).
          - NO_TRADE / WAIT does NOT vote (abstain), but IS logged with a
            reason in `vote_trail`, always returned (fix #10).
          - Consolidation (multi-Doji) still forces WAIT — that override is
            correct and unchanged.
          - A side must clear `min_action_score` AND beat the other side by
            `min_margin_abs`/`min_margin_ratio` to win — a 0.1-point edge no
            longer flips BUY vs SELL (fix #9).
          - Confidence is a blended composite, not two bare thresholds
            (fix #4/#13) — see `_score_confidence`.
        """
        cfg = self.consensus_config
        votes: List[Tuple[str, float, str, str]] = []
        _vote_trail: List[str] = []

        def _engine_vote(result: Optional[dict], base_weight: float, mult_key: str, name: str):
            if not result:
                _vote_trail.append(f"{name}=no_result abstained: engine returned nothing")
                return
            sig = result.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action not in VALID_ACTIONS:
                _vote_trail.append(
                    f"{name}={action} abstained: {str(sig.get('reason', 'no reason given'))[:100]}"
                )
                return
            eng_conf = sig.get("confidence", "Medium")
            # Historical audit: StopHunt Medium=10% / Low=0% WR.
            # Daily-trade pass: Low still hard-skip; Medium allowed at
            # heavily reduced weight so it can only help confluence, never solo.
            if name == "StopHunt" and eng_conf == "Low":
                _vote_trail.append(
                    f"{name}={action}({eng_conf}) abstained: StopHunt Low hist WR=0%"
                )
                return
            stop_hunt_med_penalty = 0.35 if (name == "StopHunt" and eng_conf == "Medium") else 1.0
            rr = _extract_rr(sig)
            rr_mult = 1.0 + min(cfg.rr_ev_weight * (rr - 1.0), cfg.rr_ev_cap - 1.0)
            rr_mult = max(rr_mult, 0.5)
            conf_mult = _CONFIDENCE_PRIOR.get(eng_conf, 0.5)
            # Blend: full weight only when engine itself is confident
            conf_scale = 0.55 + 0.45 * conf_mult  # High≈0.93, Medium≈0.82, Low≈0.71
            eff_weight = (
                base_weight * regime_mult.get(mult_key, 1.0) * rr_mult
                * conf_scale * stop_hunt_med_penalty
            )
            votes.append((action, eff_weight, eng_conf, name))
            _vote_trail.append(
                f"{name}={action}({eng_conf}) voted "
                f"weight={eff_weight:.2f} (base={base_weight}, regime_x{regime_mult.get(mult_key, 1.0):.2f}, "
                f"rr_x{rr_mult:.2f}, conf_x{conf_scale:.2f}, rr={rr:.1f})"
            )

        _engine_vote(stop_hunt_result, weights.stop_hunt, "stop_hunt", "StopHunt")
        _engine_vote(ict_result, weights.ict_amd, "ict_amd", "ICT/AMD")
        _engine_vote(pa_result, weights.pa, "pa", "PA")
        _engine_vote(liquidity_result, weights.liquidity, "liquidity", "Liquidity")
        _engine_vote(cci_result, weights.cci, "cci", "CCI")

        # df_len is the REAL bar count of the analyzed dataframe (fix —
        # previously this was derived from max(p.candle_index for p in
        # detected_patterns), i.e. "recent" was relative to whichever
        # pattern happened to be detected most recently, not the actual
        # live bar. If pattern detection found nothing near the current
        # bar but did find something several bars back, that stale
        # pattern was incorrectly treated as "now" and allowed to vote —
        # a plausible contributor to wrong-side consensus signals.
        pattern_votes, pattern_trail = self._pattern_votes(
            detected_patterns, df_len=df_len, weights=weights, regime_mult=regime_mult,
        )
        votes.extend(pattern_votes)
        _vote_trail.extend(pattern_trail)

        log.info("[Unified] Vote trail: " + " | ".join(_vote_trail))

        # Engine-health context (fix #5): distinguish "nothing set up" from
        # "several engines crashed" in the final reason.
        failed_engines = [k for k, v in engine_statuses.items() if v == ENGINE_STATUS_FAILED]
        degraded_note = f" [DEGRADED: {len(failed_engines)} engine(s) failed: {', '.join(failed_engines)}]" \
            if failed_engines else ""

        # Tally votes by direction
        buy_score = sum(w for a, w, c, e in votes if a == "BUY")
        sell_score = sum(w for a, w, c, e in votes if a == "SELL")
        total_score = buy_score + sell_score

        base_result = {"vote_trail": _vote_trail, "regime": regime_label}

        # Consolidation override
        if pattern_repetition.get("consolidation_detected", False):
            return {
                **base_result,
                "action": "WAIT",
                "confidence": "Medium",
                "calibrated_score": 0.5,
                "reason": "Consolidation detected (multiple Doji) — engines abstain, lean WAIT" + degraded_note,
                "voting_engines": [],
                "buy_score": 0.0,
                "sell_score": 0.0,
            }

        if total_score == 0:
            return {
                **base_result,
                "action": "NO_TRADE",
                "confidence": "Low",
                "calibrated_score": 0.0,
                "reason": (
                    f"All {len(failed_engines)} engine(s) failed — cannot determine a signal"
                    if failed_engines and len(failed_engines) == len(engine_statuses)
                    else "No engine produced BUY/SELL signal — all abstained"
                ) + degraded_note,
                "voting_engines": [],
                "buy_score": 0.0,
                "sell_score": 0.0,
            }

        # === Margin-gated consensus (fix #9) ===
        if buy_score >= sell_score:
            winner, winner_score, loser_score = "BUY", buy_score, sell_score
        else:
            winner, winner_score, loser_score = "SELL", sell_score, buy_score

        margin_abs_ok = (winner_score - loser_score) >= cfg.min_margin_abs
        margin_ratio_ok = (loser_score == 0) or (winner_score / loser_score >= cfg.min_margin_ratio)
        score_ok = winner_score >= cfg.min_action_score

        if not (score_ok and (margin_abs_ok or margin_ratio_ok)):
            return {
                **base_result,
                "action": "NO_TRADE",
                "confidence": "Low",
                "calibrated_score": 0.0,
                "reason": (
                    f"Insufficient consensus: BUY={buy_score:.2f} SELL={sell_score:.2f} "
                    f"(need score>={cfg.min_action_score}, margin>={cfg.min_margin_abs} abs "
                    f"or x{cfg.min_margin_ratio} ratio)"
                ) + degraded_note,
                "voting_engines": [],
                "buy_score": round(buy_score, 2),
                "sell_score": round(sell_score, 2),
            }

        consensus_action = winner
        consensus_score = winner_score
        winning_votes = [v for v in votes if v[0] == consensus_action]
        vote_count = len(winning_votes)

        confidence, calibrated_score = self._score_confidence(
            vote_count=vote_count,
            consensus_score=consensus_score,
            margin=winner_score - loser_score,
            winning_votes=winning_votes,
            cfg=cfg,
        )

        # ── Profit gate (2026-08-15) ──────────────────────────────────
        # Historical priors: ICT≈100%, StopHunt-High≈52%, Liquidity≈41%,
        # CCI≈35%, Patterns≈38%.
        # Rules:
        #  1. Edge engine present (ICT/StopHunt) → always allow.
        #  2. Weak-only: need ≥2 distinct real engines (not just patterns)
        #     e.g. Liquidity+CCI. Solo CCI / pattern-only / solo Liq → block.
        #  3. Weak-only + Low confidence → always block (negative EV).
        _win_blob = " ".join(e for _, _, _, e in winning_votes).lower()
        _has_edge = (
            "ict" in _win_blob
            or "stophunt" in _win_blob
            or "stop_hunt" in _win_blob
            or "stop hunt" in _win_blob
        )
        _real_engines = set()
        for _, _, _, e in winning_votes:
            el = e.lower()
            if el.startswith("pattern"):
                continue
            if "liquidity" in el:
                _real_engines.add("liquidity")
            elif "cci" in el:
                _real_engines.add("cci")
            elif "pa" in el:
                _real_engines.add("pa")
            else:
                _real_engines.add(el.split(":")[0].strip() or el)
        _weak_ok = len(_real_engines) >= 2

        if not _has_edge and (confidence == "Low" or not _weak_ok):
            return {
                **base_result,
                "action": "NO_TRADE",
                "confidence": confidence if confidence else "Low",
                "calibrated_score": calibrated_score,
                "reason": (
                    f"Profit gate: {consensus_action} weak-only stack "
                    f"(engines={sorted(_real_engines) or ['patterns-only']}, "
                    f"conf={confidence}) — need ICT/StopHunt or ≥2 real engines"
                ) + degraded_note,
                "voting_engines": [
                    {"engine": e, "action": a, "weight": round(w, 3), "confidence": c}
                    for a, w, c, e in winning_votes
                ],
                "buy_score": round(buy_score, 2),
                "sell_score": round(sell_score, 2),
            }

        reason = (
            f"Consensus {consensus_action} | {vote_count} engine(s)/pattern(s) agreed | "
            f"Score={consensus_score:.2f} vs {loser_score:.2f} | Regime={regime_label}"
        ) + degraded_note

        return {
            **base_result,
            "action": consensus_action,
            "confidence": confidence,
            "calibrated_score": calibrated_score,
            "reason": reason,
            "voting_engines": [
                {"engine": e, "action": a, "weight": round(w, 3), "confidence": c}
                for a, w, c, e in winning_votes
            ],
            "buy_score": round(buy_score, 2),
            "sell_score": round(sell_score, 2),
        }

    @staticmethod
    def _score_confidence(
        vote_count: int, consensus_score: float, margin: float,
        winning_votes: List[Tuple[str, float, str, str]], cfg: ConsensusConfig,
    ) -> Tuple[str, float]:
        """
        Institutional review fixes #4 / #13.

        Blends vote count, absolute score, margin over the losing side, and
        each voting engine's own reported confidence label into one
        composite 0-1 `calibrated_score`, then buckets it to Low/Medium/High
        for backward-compatible consumers.

        Caveat (explicitly, per skill SOP on backtest/live honesty): this is
        a HEURISTIC BLEND, not a statistically calibrated probability. Real
        calibration (Platt scaling / isotonic regression mapping raw score
        -> realized win rate) requires a labeled historical outcome dataset
        and belongs in the backtest/DNA pipeline (`dna_walkforward.py` /
        `dna_journal.py` if those track outcomes) — wire that mapping in
        here once it exists rather than trusting this number as a true P(win).
        """
        # Component 1: vote-count signal, saturating at 3 agreeing sources.
        # Strong engines (ICT, StopHunt) count more toward "quality".
        count_component = min(vote_count / 3.0, 1.0)
        # Component 2: absolute score vs. the "High" threshold.
        score_component = min(consensus_score / max(cfg.high_confidence_score, 1e-9), 1.0)
        # Component 3: margin over the losing side, saturating at 2x the min margin.
        margin_component = min(margin / max(cfg.min_margin_abs * 2, 1e-9), 1.0)
        # Component 4: mean of each voting engine's own confidence prior.
        priors = [_CONFIDENCE_PRIOR.get(c, 0.5) for _, _, c, _ in winning_votes]
        prior_component = float(np.mean(priors)) if priors else 0.5
        # Component 5: engine quality bonus/penalty from historical WR audit.
        # ICT (~100% tiny-n) and StopHunt-High (~52%) raise trust.
        # Liquidity-dominant / pure weak-engine stacks are penalized so they
        # cannot inflate calibrated_score into High and bleed the book.
        engine_names = " ".join(e for _, _, _, e in winning_votes).lower()
        quality_bonus = 0.0
        if "ict" in engine_names:
            quality_bonus += 0.14
        if "stophunt" in engine_names or "stop_hunt" in engine_names or "stop hunt" in engine_names:
            quality_bonus += 0.08
        if "liquidity" in engine_names and vote_count == 1:
            quality_bonus -= 0.22  # solo Liquidity ~41% WR — hard penalize
        if "liquidity" in engine_names and "ict" not in engine_names and "stophunt" not in engine_names and "stop_hunt" not in engine_names and "stop hunt" not in engine_names:
            # Liquidity without a proven edge engine → modest penalty
            quality_bonus -= 0.08
        if "cci" in engine_names and vote_count == 1:
            quality_bonus -= 0.12  # solo CCI overall ~35% WR
        if "pattern" in engine_names and vote_count == 1:
            quality_bonus -= 0.10  # pattern-only historically mixed

        calibrated = (
            0.28 * count_component + 0.28 * score_component
            + 0.18 * margin_component + 0.18 * prior_component
            + quality_bonus
        )
        calibrated = round(min(max(calibrated, 0.0), 1.0), 3)

        # Bucketing (profit-safe 2026-08-15):
        # High ONLY when a proven-edge engine (ICT or StopHunt) is in the
        # winning stack. Liquidity/CCI/Pattern confluence → Medium at best.
        # Keeps daily frequency via Medium while protecting book WR.
        min_cal_h = getattr(cfg, "min_calibrated_for_high", 0.58)
        min_cal_m = getattr(cfg, "min_calibrated_for_medium", 0.36)

        engine_blob = " ".join(e for _, _, _, e in winning_votes).lower()
        has_ict = "ict" in engine_blob
        has_stophunt = (
            "stophunt" in engine_blob
            or "stop_hunt" in engine_blob
            or "stop hunt" in engine_blob
        )
        has_edge = has_ict or has_stophunt

        if has_edge and calibrated >= min_cal_h and (
            consensus_score >= cfg.high_confidence_score * 0.85 or vote_count >= 2
        ):
            label = "High"
        elif calibrated >= min_cal_m or (
            vote_count >= 1 and consensus_score >= cfg.medium_confidence_score
        ):
            label = "Medium"
        else:
            label = "Low"
        return label, calibrated

    # ═══════════════════════════════════════════════════════════
    # FALLBACK EMPTY RESULTS (for engine failures)
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _fallback_stop_hunt(reason: str = "StopHunt engine failed") -> dict:
        return {
            "resistance_zones": [], "support_zones": [],
            "stop_hunt_detected": False, "stop_hunt_zone": "null",
            "signal": {
                "action": "NO_TRADE", "entry_price": None, "stop_loss": None,
                "take_profit": None, "reason": reason,
                "confidence": "Low",
            },
        }

    @staticmethod
    def _fallback_ict(reason: str = "ICT/AMD engine failed") -> dict:
        return {
            "zones": {"strongest_zone": None, "weakest_zone": None},
            "accumulation": {"valid": False, "range_high": None, "range_low": None},
            "manipulation": {"detected": False, "direction": "null",
                             "sweep_price": None, "zone_strength_used": "null"},
            "fvg": {"found": False, "type": "null", "top": None, "bottom": None, "midpoint": None},
            "mss_confirmed": False,
            "signal": {
                "action": "NO_TRADE", "entry_price": None, "stop_loss": None,
                "take_profit": None, "risk_reward": None,
                "reason": reason, "confidence": "Low",
            },
        }

    @staticmethod
    def _fallback_pa(symbol: str, reason: str = "PA engine failed") -> dict:
        return {
            "pair": symbol, "timeframe": "", "session_time_ok": False,
            "trend": {"structure": "sideways", "bos_detected": False, "choch_detected": False},
            "zones": {"support_resistance": [], "supply_demand": [], "strongest_confluence_zone": None},
            "shooting_star_setup": {"detected": False, "candle1_confirmed": False, "candle2_seller_pressure_confirmed": False},
            "multi_timeframe_confirmation": {"lower_tf_used": "null", "aligned": False},
            "confirmation_checklist": {
                "candlestick_pattern": False, "chart_pattern": False, "candle_behavior": False,
                "confluence_level": False, "trendline_confluence": False, "multi_tf_alignment": False,
                "total_confirmed": 0,
            },
            "signal": {
                "action": "NO_TRADE", "entry_price": None, "stop_loss": None,
                "take_profit_suggested": None, "risk_reward": None,
                "reason": reason, "confidence": "Low",
            },
        }

    @staticmethod
    def _fallback_liquidity(reason: str = "Liquidity engine failed") -> dict:
        return {
            "valid": False, "recent_sweep": None,
            "signal": {
                "action": "NO_TRADE", "entry_price": None, "stop_loss": None,
                "take_profit": None, "r_rr": None,
                "reason": reason, "confidence": "Low",
            },
        }

    @staticmethod
    def _fallback_cci(reason: str = "CCI engine failed") -> dict:
        return {
            "valid": False, "cci_value": None, "zone_type": None,
            "signal": {
                "action": "NO_TRADE", "entry_price": None, "stop_loss": None,
                "take_profit": None, "r_rr": None,
                "reason": reason, "confidence": "Low",
            },
        }

    @staticmethod
    def _insufficient_data_result(symbol: str) -> dict:
        return {
            "pair": symbol,
            "timeframe": "",
            "current_price": None,
            "atr": None,
            "zones": {"support_resistance": [], "unified_zones": []},
            "detected_patterns": [],
            "pattern_repetition": {"zone_strength_boosts": [], "momentum_sequence": None, "consolidation_detected": False},
            "stop_hunt": UnifiedSignalEngine._fallback_stop_hunt(),
            "ict_amd": UnifiedSignalEngine._fallback_ict(),
            "multi_strategy_pa": UnifiedSignalEngine._fallback_pa(symbol),
            "liquidity": UnifiedSignalEngine._fallback_liquidity(reason="Insufficient data"),
            "cci_state": UnifiedSignalEngine._fallback_cci(reason="Insufficient data"),
            "regime": {"market_regime": "UNKNOWN", "strategy_type": "NO_TRADE", "risk_multiplier": 0.0},
            "engine_health": {"status": "insufficient_data", "engines": {}},
            "consensus": {
                "action": "NO_TRADE", "confidence": "Low", "calibrated_score": 0.0,
                "reason": "Insufficient data",
                "voting_engines": [], "buy_score": 0.0, "sell_score": 0.0,
                "vote_trail": ["all engines abstained: insufficient candle data"],
            },
        }

    # ═══════════════════════════════════════════════════════════
    # BUILD UNIFIED OUTPUT
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _build_unified_result(
        symbol: str,
        timeframe: str,
        atr: float,
        current_price: float,
        sr_zones: list,
        unified_zones: list,
        detected_patterns: list,
        pattern_repetition: dict,
        stop_hunt_result: Optional[dict],
        ict_result: Optional[dict],
        pa_result: Optional[dict],
        consensus: dict,
        liquidity_result: Optional[dict] = None,
        cci_result: Optional[dict] = None,
        regime_ctx: Optional[dict] = None,
        engine_statuses: Optional[Dict[str, str]] = None,
        engine_timings: Optional[Dict[str, float]] = None,
    ) -> dict:
        """Build the unified output dict."""
        sr_zones_output = [
            {
                "type": z.get("type", "support"),
                "zone_top": round(float(z.get("zone_top", 0)), 5),
                "zone_bottom": round(float(z.get("zone_bottom", 0)), 5),
                "touches": int(z.get("touches", 0)),
                "strength": z.get("strength", "Weak"),
            }
            for z in sr_zones[:10]
        ]

        engine_statuses = engine_statuses or {}
        engine_timings = engine_timings or {}
        failed = [k for k, v in engine_statuses.items() if v == ENGINE_STATUS_FAILED]
        engine_health = {
            "status": "degraded" if failed else "healthy",
            "failed_engines": failed,
            "engines": {
                name: {
                    "status": engine_statuses.get(name, "unknown"),
                    "elapsed_seconds": round(engine_timings.get(name, 0.0), 3),
                }
                for name in engine_statuses
            },
        }

        return {
            "pair": symbol,
            "timeframe": timeframe,
            "current_price": round(current_price, 5),
            "atr": round(atr, 6),
            # ── SHARED LAYER ──
            "zones": {
                "support_resistance": sr_zones_output,
                "unified_zones": [
                    {
                        "type": z["type"],
                        "zone_top": round(z["zone_top"], 5),
                        "zone_bottom": round(z["zone_bottom"], 5),
                        "touches": z["touches"],
                        "strength": z["strength"],
                        "source": z["source"],
                    }
                    for z in unified_zones[:15]
                ],
            },
            "detected_patterns": detected_patterns,
            "pattern_repetition": pattern_repetition,
            "regime": regime_ctx or {},
            "engine_health": engine_health,
            # ── PER-ENGINE RESULTS ──
            "stop_hunt": stop_hunt_result,
            "ict_amd": ict_result,
            "multi_strategy_pa": pa_result,
            "liquidity": liquidity_result,
            "cci_state": cci_result,
            # ── CONSENSUS ──
            "consensus": consensus,
        }

    # ═══════════════════════════════════════════════════════════
    # LLM-FRIENDLY OUTPUT
    # ═══════════════════════════════════════════════════════════

    def analyze_to_json(
        self, df: pd.DataFrame, symbol: str, lower_tf_df: Optional[pd.DataFrame] = None
    ) -> str:
        return json.dumps(self.analyze(df, symbol, lower_tf_df), ensure_ascii=False, indent=2)

    def to_prompt_text(self, result: dict) -> str:
        """Plain-text rendering for LLM prompts."""
        lines = [
            f"=== UNIFIED SIGNAL ({result['pair']} {result['timeframe']}) ===",
            f"Current Price: {result.get('current_price')}",
            f"ATR: {result.get('atr')}",
            f"Regime: {result.get('regime', {}).get('market_regime', 'UNKNOWN')}",
            "",
            "-- Zones (S/R) --",
        ]
        for z in result["zones"].get("support_resistance", [])[:5]:
            lines.append(f"  {z['type']}: [{z['zone_bottom']} - {z['zone_top']}] touches={z['touches']} ({z['strength']})")

        lines.append("")
        lines.append("-- Detected Patterns --")
        patterns = result.get("detected_patterns", [])
        if not patterns:
            lines.append("  (none)")
        else:
            for p in patterns[:10]:
                emoji = "🟢" if p["reliability"] == "High" else "⚪"
                lines.append(
                    f"  {emoji} {p['pattern_name']} ({p['type']}) @ {p['candle_index_or_time']} "
                    f"| near_zone={p['near_zone']} ({p['zone_type']}) | {p['reliability']}"
                )

        rep = result.get("pattern_repetition", {})
        if rep.get("consolidation_detected"):
            lines.append("  ⚠ Consolidation detected (multiple Doji) → lean WAIT")
        if rep.get("momentum_sequence"):
            ms = rep["momentum_sequence"]
            lines.append(f"  📈 Momentum sequence: {ms['direction']} x{ms['count']}")

        lines.append("")
        lines.append("-- Engine Health --")
        eh = result.get("engine_health", {})
        lines.append(f"  Status: {eh.get('status', 'unknown')}")
        if eh.get("failed_engines"):
            lines.append(f"  Failed: {', '.join(eh['failed_engines'])}")

        lines.append("")
        lines.append("-- Engine Signals --")

        sh = result.get("stop_hunt", {})
        if sh:
            sh_sig = sh.get("signal", {})
            lines.append(f"  StopHunt: {sh_sig.get('action')} (detected={sh.get('stop_hunt_detected')}, status={sh.get('engine_status')})")

        ict = result.get("ict_amd", {})
        if ict:
            ict_sig = ict.get("signal", {})
            lines.append(
                f"  ICT/AMD: {ict_sig.get('action')} "
                f"(acc={ict.get('accumulation', {}).get('valid')}, "
                f"manip={ict.get('manipulation', {}).get('detected')}, "
                f"fvg={ict.get('fvg', {}).get('found')}, "
                f"mss={ict.get('mss_confirmed')}, status={ict.get('engine_status')})"
            )

        pa = result.get("multi_strategy_pa", {})
        if pa:
            pa_sig = pa.get("signal", {})
            lines.append(
                f"  PA: {pa_sig.get('action')} "
                f"(trend={pa.get('trend', {}).get('structure')}, "
                f"checklist={pa.get('confirmation_checklist', {}).get('total_confirmed')}/6, "
                f"session={pa.get('session_time_ok')}, status={pa.get('engine_status')})"
            )

        liq = result.get("liquidity", {})
        if liq:
            liq_sig = liq.get("signal", {})
            lines.append(f"  Liquidity: {liq_sig.get('action')} (status={liq.get('engine_status')})")

        cci = result.get("cci_state", {})
        if cci:
            cci_sig = cci.get("signal", {})
            lines.append(f"  CCI: {cci_sig.get('action')} (status={cci.get('engine_status')})")

        lines.append("")
        lines.append("-- Consensus --")
        con = result.get("consensus", {})
        lines.append(f"  Action: {con.get('action')}")
        lines.append(f"  Confidence: {con.get('confidence')} (calibrated={con.get('calibrated_score')})")
        lines.append(f"  Buy Score: {con.get('buy_score')} | Sell Score: {con.get('sell_score')}")
        lines.append(f"  Voting: {con.get('voting_engines')}")
        lines.append(f"  Reason: {con.get('reason')}")
        lines.append("  Vote Trail:")
        for line in con.get("vote_trail", []):
            lines.append(f"    - {line}")
        lines.append("=" * 50)
        return "\n".join(lines)


# ============================================================
# Convenience: one-shot helper
# ============================================================

def detect_unified_signal(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "4H",
    lower_tf_df: Optional[pd.DataFrame] = None,
    **kwargs,
) -> str:
    """One-shot helper — returns unified JSON."""
    engine = UnifiedSignalEngine(timeframe=timeframe, **kwargs)
    return engine.analyze_to_json(df, symbol, lower_tf_df)


# ============================================================
# CLI entry
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-06-03 06:00", periods=n, freq="4h")
    base = 1.0850
    close = base + np.cumsum(np.random.randn(n) * 0.0008)
    df = pd.DataFrame({
        "open":  close + np.random.randn(n) * 0.0003,
        "high":  close + abs(np.random.randn(n)) * 0.0012,
        "low":   close - abs(np.random.randn(n)) * 0.0012,
        "close": close,
    }, index=dates)

    engine = UnifiedSignalEngine(timeframe="4H")
    result = engine.analyze(df, symbol="EURUSD")
    print(engine.to_prompt_text(result))