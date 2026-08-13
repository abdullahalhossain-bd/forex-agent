"""
utils/pair_profiles.py — Per-Pair Configuration System

Each FX pair has its own "personality" — volatility, trendiness, session
sensitivity, noise level. A single global config cannot be optimal for all
pairs. This module provides per-pair overrides for:

  - min_confidence: minimum confidence to trade
  - min_aligned_factors: minimum confluence factors
  - min_rr: minimum reward:risk ratio
  - session_filter: which trading sessions to allow (london, ny, asian, all)
  - stop_atr_mult: ATR multiplier for stop-loss
  - target_atr_mult: ATR multiplier for take-profit
  - adx_min: minimum ADX to allow entry
  - pullback_atr_mult: how close to EMA-21 price must be (value area)
  - max_trades_per_day: per-pair daily cap
  - enabled: whether to trade this pair at all

Profiles were derived from a 10-config per-pair backtest sweep on H1 data
(July 2025 - July 2026, ~6200 bars per pair). See
/home/z/my-project/download/per_pair_config_sweep.json for the raw data.

Usage:
    from utils.pair_profiles import get_pair_profile, get_active_pairs
    profile = get_pair_profile("NZDUSD")
    # profile.min_confidence, profile.min_rr, etc.

    active_pairs = get_active_pairs()  # only pairs with enabled=True
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PairProfile:
    """Per-pair trading configuration."""
    symbol: str
    enabled: bool = True
    # 2026-08-13: per-pair STRATEGY SYSTEM (not just config)
    # Each pair uses a completely different entry logic
    strategy: str = "trend_follow"  # trend_follow, mean_reversion, range_trading, breakout
    # Signal quality gates
    min_confidence: int = 50
    min_aligned_factors: int = 2
    min_rr: float = 1.0
    adx_min: float = 18.0
    # Entry filters
    session_filter: str = "all"  # "london_ny", "asian", "all", "none"
    pullback_atr_mult: float = 1.5     # max distance from EMA-21 (in ATRs)
    spread_max_mult: float = 2.0       # max spread as multiple of 20-bar avg
    # SL/TP (used as fallback when strategy doesn't provide sl_price/tp_price)
    stop_atr_mult: float = 1.5
    target_atr_mult: float = 3.0
    # Risk
    max_trades_per_day: int = 3
    risk_per_trade: float = 0.005      # 0.5% default
    # Notes (for documentation)
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────
# PER-PAIR PROFILES (derived from backtest sweep)
# ─────────────────────────────────────────────────────────────────────────

_DEFAULT_PROFILE = PairProfile(
    symbol="DEFAULT",
    enabled=True,
    min_confidence=85,
    min_aligned_factors=5,
    min_rr=2.0,
    adx_min=18.0,
    session_filter="london_ny",
    pullback_atr_mult=1.5,
    spread_max_mult=2.0,
    stop_atr_mult=1.8,
    target_atr_mult=3.5,
    max_trades_per_day=3,
    risk_per_trade=0.005,
    notes="Default profile — used when pair not in PROFILES dict",
)

PROFILES: dict[str, PairProfile] = {

    # ─── EURUSD: MEAN REVERSION (56% WR, PF 1.33) ───
    # EURUSD is choppy on H1 — trend-following fails here.
    # Mean-reversion (RSI extreme + BB touch) wins 56% of the time.
    "EURUSD": PairProfile(
        symbol="EURUSD",
        enabled=True,
        strategy="mean_reversion",
        min_confidence=50,
        min_aligned_factors=2,
        min_rr=1.0,
        adx_min=0.0,  # mean-reversion WANTS low ADX
        session_filter="all",
        pullback_atr_mult=1.5,
        spread_max_mult=2.0,
        stop_atr_mult=1.5,
        target_atr_mult=1.5,
        max_trades_per_day=3,
        risk_per_trade=0.005,
        notes="Mean-reversion. RSI extreme + BB touch. 56% WR, PF 1.33. Best pair for this strategy.",
    ),

    # ─── GBPUSD: RANGE TRADING (PF 1.29) ───
    # GBPUSD is volatile — range trading at S/R edges works.
    "GBPUSD": PairProfile(
        symbol="GBPUSD",
        enabled=True,
        strategy="range_trading",
        min_confidence=50,
        min_aligned_factors=2,
        min_rr=1.0,
        adx_min=0.0,  # range trading WANTS low ADX
        session_filter="all",
        pullback_atr_mult=1.5,
        spread_max_mult=2.0,
        stop_atr_mult=2.0,
        target_atr_mult=2.0,
        max_trades_per_day=2,
        risk_per_trade=0.004,
        notes="Range trading. S/R edges + low ADX. PF 1.29. Volatile pair.",
    ),

    # ─── AUDUSD: RANGE TRADING (PF 1.11) ───
    # Commodity pair — range-bound behavior.
    "AUDUSD": PairProfile(
        symbol="AUDUSD",
        enabled=True,
        strategy="range_trading",
        min_confidence=50,
        min_aligned_factors=2,
        min_rr=1.0,
        adx_min=0.0,
        session_filter="all",
        pullback_atr_mult=1.5,
        spread_max_mult=2.0,
        stop_atr_mult=1.5,
        target_atr_mult=2.0,
        max_trades_per_day=3,
        risk_per_trade=0.005,
        notes="Range trading. Commodity pair, range-bound. PF 1.11.",
    ),

    # ─── NZDUSD: RANGE TRADING (PF 1.41 — BEST!) ───
    # Best performer — range trading with highest PF.
    "NZDUSD": PairProfile(
        symbol="NZDUSD",
        enabled=True,
        strategy="range_trading",
        min_confidence=50,
        min_aligned_factors=2,
        min_rr=1.0,
        adx_min=0.0,
        session_filter="all",
        pullback_atr_mult=1.5,
        spread_max_mult=2.0,
        stop_atr_mult=1.5,
        target_atr_mult=2.0,
        max_trades_per_day=4,
        risk_per_trade=0.005,
        notes="Range trading. BEST PF 1.41. All sessions. Highest profit factor.",
    ),

    # ─── USDCAD: TREND FOLLOWING (PF 1.15) ───
    # Low ATR% (0.082) — trends are clean when they happen.
    "USDCAD": PairProfile(
        symbol="USDCAD",
        enabled=True,
        strategy="trend_follow",
        min_confidence=50,
        min_aligned_factors=3,
        min_rr=1.0,
        adx_min=22.0,  # trend-follow wants higher ADX
        session_filter="all",
        pullback_atr_mult=0.8,
        spread_max_mult=2.0,
        stop_atr_mult=1.0,
        target_atr_mult=3.0,
        max_trades_per_day=3,
        risk_per_trade=0.005,
        notes="Trend following. Low volatility, clean trends. PF 1.15.",
    ),

    # ─── USDCHF: RANGE TRADING (PF 1.19) ───
    # Inverse EURUSD correlation — choppy, range trading works.
    "USDCHF": PairProfile(
        symbol="USDCHF",
        enabled=True,
        strategy="range_trading",
        min_confidence=50,
        min_aligned_factors=2,
        min_rr=1.0,
        adx_min=0.0,
        session_filter="all",
        pullback_atr_mult=1.5,
        spread_max_mult=2.0,
        stop_atr_mult=1.5,
        target_atr_mult=2.0,
        max_trades_per_day=2,
        risk_per_trade=0.004,
        notes="Range trading. Inverse EURUSD, choppy. PF 1.19.",
    ),

    # ─── USDJPY: RANGE TRADING (PF 1.06) ───
    # Strong bull bias but sharp reversals — range trading safer.
    "USDJPY": PairProfile(
        symbol="USDJPY",
        enabled=True,
        strategy="range_trading",
        min_confidence=50,
        min_aligned_factors=2,
        min_rr=1.0,
        adx_min=0.0,
        session_filter="all",
        pullback_atr_mult=1.5,
        spread_max_mult=2.0,
        stop_atr_mult=1.5,
        target_atr_mult=2.0,
        max_trades_per_day=2,
        risk_per_trade=0.004,
        notes="Range trading. Sharp reversals. PF 1.06. Conservative risk.",
    ),
}


def get_pair_profile(symbol: str) -> PairProfile:
    """Get the trading profile for a specific pair.

    Falls back to _DEFAULT_PROFILE if the pair is not in PROFILES.
    """
    sym = symbol.upper().strip()
    if sym in PROFILES:
        return PROFILES[sym]
    # Return default with the requested symbol name
    return PairProfile(
        symbol=sym,
        enabled=_DEFAULT_PROFILE.enabled,
        min_confidence=_DEFAULT_PROFILE.min_confidence,
        min_aligned_factors=_DEFAULT_PROFILE.min_aligned_factors,
        min_rr=_DEFAULT_PROFILE.min_rr,
        adx_min=_DEFAULT_PROFILE.adx_min,
        session_filter=_DEFAULT_PROFILE.session_filter,
        pullback_atr_mult=_DEFAULT_PROFILE.pullback_atr_mult,
        spread_max_mult=_DEFAULT_PROFILE.spread_max_mult,
        stop_atr_mult=_DEFAULT_PROFILE.stop_atr_mult,
        target_atr_mult=_DEFAULT_PROFILE.target_atr_mult,
        max_trades_per_day=_DEFAULT_PROFILE.max_trades_per_day,
        risk_per_trade=_DEFAULT_PROFILE.risk_per_trade,
        notes=_DEFAULT_PROFILE.notes,
    )


def get_active_pairs() -> list[str]:
    """Return list of pairs that are enabled (profile.enabled=True)."""
    return [sym for sym, prof in PROFILES.items() if prof.enabled]


def is_pair_enabled(symbol: str) -> bool:
    """Check if a pair is enabled for trading."""
    return get_pair_profile(symbol).enabled


def get_session_hours(session_filter: str) -> tuple[set, str]:
    """Convert session_filter string to set of UTC hours and description.

    Returns:
        (set_of_utc_hours, description_string)
    """
    session_filter = session_filter.lower().strip()
    if session_filter == "all" or session_filter == "none":
        return set(range(24)), "All sessions (24h)"
    if session_filter == "london_ny":
        # London 07-11 GMT + NY overlap 12-16 GMT
        return {7, 8, 9, 10, 12, 13, 14, 15}, "London (07-11) + NY overlap (12-16) GMT"
    if session_filter == "london":
        return {7, 8, 9, 10, 11}, "London (07-11 GMT)"
    if session_filter == "ny":
        return {12, 13, 14, 15, 16}, "NY (12-16 GMT)"
    if session_filter == "asian":
        # Asian session 00-06 GMT (Tokyo + Sydney)
        return {0, 1, 2, 3, 4, 5}, "Asian (00-06 GMT)"
    if session_filter == "asian_london":
        # Asian + London (skip NY — for volatile pairs that chop in NY)
        return {0, 1, 2, 3, 4, 5, 7, 8, 9, 10}, "Asian (00-06) + London (07-11) GMT"
    return {7, 8, 9, 10, 12, 13, 14, 15}, f"Default london_ny (unknown filter: {session_filter})"


# ─── Smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Per-Pair Profiles Summary")
    print("=" * 100)
    print(f"  {'Pair':8s} {'Enabled':8s} {'Conf':5s} {'Fact':5s} {'RR':4s} {'ADX':5s} "
          f"{'Session':12s} {'SL×ATR':7s} {'TP×ATR':7s} {'Notes'}")
    print("-" * 100)
    for sym, prof in PROFILES.items():
        print(f"  {sym:8s} {'YES' if prof.enabled else 'NO':8s} "
              f"{prof.min_confidence:5d} {prof.min_aligned_factors:5d} "
              f"{prof.min_rr:4.1f} {prof.adx_min:5.1f} "
              f"{prof.session_filter:12s} "
              f"{prof.stop_atr_mult:7.1f} {prof.target_atr_mult:7.1f} "
              f"{prof.notes[:40]}")
    print("=" * 100)
    print(f"\nActive pairs ({len(get_active_pairs())}): {get_active_pairs()}")
    print(f"\nSample profile NZDUSD:")
    p = get_pair_profile("NZDUSD")
    for field_name in ["enabled", "min_confidence", "min_aligned_factors",
                        "min_rr", "adx_min", "session_filter", "stop_atr_mult"]:
        print(f"  {field_name}: {getattr(p, field_name)}")
