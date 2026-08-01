# intelligence/ported_indicators_registry.py — Registry of ported MQL5 indicators
# =============================================================================
# Lightweight registry that documents the indicators ported from
# geraked/metatrader5 and provides a uniform interface for the confluence
# engine to call them.
#
# This module does NOT modify the existing ConfluenceEngine. Instead, it
# exposes a `compute_all(df)` function that runs every ported indicator
# and returns a single DataFrame with all the new columns merged. The
# confluence engine can then opt-in to consume specific columns.
#
# Integration path (RECOMMENDED — non-breaking):
#   1. ConfluenceEngine calls `compute_all(df)` once per analysis cycle.
#   2. It reads the columns it cares about (e.g., `st_trend`, `ut_signal`).
#   3. Each ported indicator's contribution to the confluence score is
#      added as a new FactorScore in `decision_score.py`.
#
# Integration path (FULL — modifies ConfluenceEngine):
#   Add new Factor entries in ConfluenceEngine.collect_factors() that call
#   the per-indicator compute functions directly. See CONTRIBUTING.md for
#   the recipe "Adding a new analysis module".
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger("ported_indicators")


# ── Indicator registry ───────────────────────────────────────────────────────

@dataclass
class PortedIndicator:
    """Metadata for a ported MQL5 indicator."""
    name: str                       # short name, e.g., "supertrend"
    display_name: str               # human-friendly, e.g., "SuperTrend"
    source_mql5: str                # path to original .mq5 in upstream repo
    compute_fn: Callable[..., pd.DataFrame]
    default_params: dict
    output_columns: List[str]
    confluence_signal_column: Optional[str] = None
    # ^ which column to read for a +1/-1/0 directional signal
    description: str = ""


_REGISTRY: Dict[str, PortedIndicator] = {}


def register(indicator: PortedIndicator) -> None:
    """Register a ported indicator. Idempotent."""
    _REGISTRY[indicator.name] = indicator
    log.debug(f"Registered ported indicator: {indicator.name}")


def get(name: str) -> Optional[PortedIndicator]:
    """Look up a registered indicator by name."""
    return _REGISTRY.get(name)


def list_indicators() -> List[PortedIndicator]:
    """Return all registered ported indicators."""
    return list(_REGISTRY.values())


# ── Compute helpers ──────────────────────────────────────────────────────────

def compute_one(name: str, df: pd.DataFrame, **overrides) -> pd.DataFrame:
    """
    Compute a single ported indicator on `df`. Returns df with the
    indicator's columns added. Unknown name → ValueError.
    """
    ind = get(name)
    if ind is None:
        raise ValueError(f"Unknown ported indicator: {name!r}. "
                         f"Known: {list(_REGISTRY.keys())}")
    params = {**ind.default_params, **overrides}
    return ind.compute_fn(df, **params)


def compute_all(df: pd.DataFrame, **overrides_per_indicator) -> pd.DataFrame:
    """
    Compute EVERY registered ported indicator on `df`. Returns a single
    DataFrame with all columns merged. Indicators are computed in
    registration order; later indicators see the columns of earlier ones
    (but they shouldn't depend on them — each is independent).

    `overrides_per_indicator` is an optional dict-of-dicts:
        {"supertrend": {"period": 14}, "utbot": {"atr_coef": 1.5}}
    """
    out = df.copy()
    for ind in _REGISTRY.values():
        try:
            params = {**ind.default_params,
                      **overrides_per_indicator.get(ind.name, {})}
            out = ind.compute_fn(out, **params)
            log.debug(f"Computed {ind.name}: added {ind.output_columns}")
        except Exception as e:
            log.warning(f"Failed to compute {ind.name}: {e}")
            # Continue with other indicators even if one fails
    return out


def directional_signals(df: pd.DataFrame) -> Dict[str, int]:
    """
    Read the directional signal column of every registered indicator
    from `df` and return a {indicator_name: +1/-1/0} dict.

    Use this for quick confluence: if ≥3 indicators agree on direction,
    that's a strong signal.
    """
    out = {}
    for ind in _REGISTRY.values():
        if ind.confluence_signal_column is None:
            continue
        if ind.confluence_signal_column not in df.columns:
            continue
        col = df[ind.confluence_signal_column]
        if col.empty:
            continue
        # Take the LAST value (most recent bar)
        last = col.iloc[-1]
        try:
            out[ind.name] = int(last) if not pd.isna(last) else 0
        except (ValueError, TypeError):
            out[ind.name] = 0
    return out


# ── Auto-registration of ported indicators ───────────────────────────────────
# Done at import time so `from intelligence.ported_indicators_registry import
# compute_all` "just works".

def _autoregister():
    """Import and register all ported indicators. Safe to call multiple times."""
    try:
        from analysis.supertrend import compute as st_compute
        register(PortedIndicator(
            name="supertrend",
            display_name="SuperTrend",
            source_mql5="Indicators/SuperTrend.mq5",
            compute_fn=st_compute,
            default_params={"period": 10, "multiplier": 3.0},
            output_columns=["supertrend", "st_trend", "st_color"],
            confluence_signal_column="st_trend",
            description="ATR-based trend-following line. +1 bull, -1 bear.",
        ))
    except ImportError as e:
        log.warning(f"Could not register supertrend: {e}")

    try:
        from analysis.utbot_alerts import compute as ut_compute
        register(PortedIndicator(
            name="utbot",
            display_name="UT Bot Alerts",
            source_mql5="Indicators/UTBot.mq5",
            compute_fn=ut_compute,
            default_params={"atr_coef": 2.0, "atr_len": 1},
            output_columns=["ut_trail", "ut_bull_arrow", "ut_bear_arrow", "ut_signal"],
            confluence_signal_column="ut_signal",
            description="ATR-trailing-stop breakout arrows. +1 bull, -1 bear, 0 no signal.",
        ))
    except ImportError as e:
        log.warning(f"Could not register utbot: {e}")

    try:
        from analysis.andean_oscillator import compute as ao_compute
        register(PortedIndicator(
            name="andean_oscillator",
            display_name="Andean Oscillator",
            source_mql5="Indicators/AndeanOscillator.mq5",
            compute_fn=ao_compute,
            default_params={"length": 50, "signal_length": 9},
            output_columns=["ao_bull", "ao_bear", "ao_signal", "ao_phase"],
            confluence_signal_column="ao_phase",
            description="Volatility-aware oscillator. +1 upside vol dominates (uptrend), -1 downside (downtrend).",
        ))
    except ImportError as e:
        log.warning(f"Could not register andean_oscillator: {e}")

    try:
        from analysis.nadaraya_watson_envelope import compute as nwe_compute
        register(PortedIndicator(
            name="nwe",
            display_name="Nadaraya-Watson Envelope",
            source_mql5="Indicators/NadarayaWatsonEnvelope.mq5",
            compute_fn=nwe_compute,
            default_params={"band_width": 8.0, "multiplier": 3.0, "window_size": 500},
            output_columns=["nwe_mid", "nwe_upper", "nwe_lower", "nwe_pos"],
            confluence_signal_column="nwe_pos",
            description="Non-parametric Gaussian-kernel envelope. +1 above upper, -1 below lower, 0 inside.",
        ))
    except ImportError as e:
        log.warning(f"Could not register nwe: {e}")

    try:
        from analysis.daily_high_low import compute as dhl_compute
        register(PortedIndicator(
            name="daily_high_low",
            display_name="Daily High/Low",
            source_mql5="Indicators/DailyHighLow.mq5",
            compute_fn=dhl_compute,
            default_params={"previous": True, "price_mode": "lowhigh"},
            output_columns=["dhl_high", "dhl_low"],
            confluence_signal_column=None,  # No directional signal — levels only
            description="Previous day's high/low as horizontal levels. Used for breakout strategies.",
        ))
    except ImportError as e:
        log.warning(f"Could not register daily_high_low: {e}")

    try:
        from analysis.chandelier_exit import compute as ce_compute
        register(PortedIndicator(
            name="chandelier_exit",
            display_name="Chandelier Exit",
            source_mql5="Indicators/ChandelierExit.mq5",
            compute_fn=ce_compute,
            default_params={"atr_period": 1, "atr_mult": 0.75},
            output_columns=["ce_long_stop", "ce_short_stop", "ce_dir",
                            "ce_buy_signal", "ce_sell_signal"],
            confluence_signal_column="ce_dir",
            description="Heikin-Ashi + ATR trailing exit. +1 long, -1 short. Buy/sell signals on flips.",
        ))
    except ImportError as e:
        log.warning(f"Could not register chandelier_exit: {e}")

    try:
        from analysis.atr_sl_finder import compute as asf_compute
        register(PortedIndicator(
            name="atr_sl_finder",
            display_name="ATR Stop-Loss Finder",
            source_mql5="Indicators/AtrSlFinder.mq5",
            compute_fn=asf_compute,
            default_params={"length": 14, "multiplier": 1.5, "causal": True},
            output_columns=["atr_sl_upper", "atr_sl_lower", "atr_sl_ma"],
            confluence_signal_column=None,  # SL/TP levels only
            description="Suggested SL levels above high / below low based on smoothed TR × multiplier.",
        ))
    except ImportError as e:
        log.warning(f"Could not register atr_sl_finder: {e}")


_autoregister()


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli():
    """Print registry status."""
    print("═══ Ported Indicators Registry ═══")
    for ind in list_indicators():
        sig = ind.confluence_signal_column or "(no directional signal)"
        print(f"  {ind.name:18s}  →  {ind.display_name:25s}  signal: {sig}")
        print(f"    source:  {ind.source_mql5}")
        print(f"    outputs: {', '.join(ind.output_columns)}")
        print(f"    desc:    {ind.description}")
        print()


if __name__ == "__main__":
    _cli()
