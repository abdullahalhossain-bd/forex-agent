"""
strategies/_common.py — Shared helpers for all strategy modules.

Extracted from breakout.py / mean_reversion.py, which independently fixed
the same NaN-handling bug. Per mean_reversion.py's own docstring:

    "Recommend extracting this helper into a shared strategies/_common.py
    used by every strategy module, so it's fixed once instead of re-broken
    independently in strategy #4, #5, etc."

That bug, confirmed present in momentum.py, pullback.py, range_trading.py,
retest.py, and reversal.py (grep for `.get(key, default) or default`):

    float(row.get("adx", 0) or 0)

`NaN or 0` evaluates to `NaN` in Python, because NaN is truthy. So a NaN
value from the data pipeline SURVIVES the "or default" fallback (only
falsy values like 0/None/"" get replaced) and silently propagates into
downstream comparisons and stop/target math. For a filter like
`adx >= min_adx`, a NaN adx makes the comparison False -- which looks like
"filter correctly blocked the trade" but is actually "filter silently did
nothing because the input was garbage."

Use `safe_float()` below instead of the `or default` pattern anywhere a
row/Series value is read.
"""
from __future__ import annotations

import math
from typing import Any, Optional


class MarketDataError(Exception):
    """Raised when a required field is missing, malformed, or NaN."""


def safe_float(row: Any, key: str, default: Optional[float] = None, required: bool = False) -> float:
    """
    NaN-safe field extraction from a pandas Series/dict-like row.

    Unlike `float(row.get(key, default) or default)`, this treats a missing
    key AND a NaN value as equivalent -- both fall back to `default`.

    Raises MarketDataError if `required` and the resolved value is missing/NaN.
    """
    if key not in row or row[key] is None:
        value = default
    else:
        try:
            value = float(row[key])
        except (TypeError, ValueError) as exc:
            raise MarketDataError(f"Field '{key}' is not numeric: {row[key]!r}") from exc
        if math.isnan(value):
            value = default

    if value is None:
        if required:
            raise MarketDataError(f"Required field '{key}' is missing or NaN")
        return float("nan")
    return float(value)


# Standard pip sizes (mirrors risk/atr_risk_manager.py's PIP_SIZES and
# trend_follow.py's local copy, kept in sync deliberately).
PIP_SIZES = {"JPY": 0.01, "XAU": 0.01, "XAG": 0.01, "DEFAULT": 0.0001}


def pip_size_for(pair: Optional[str]) -> float:
    """
    Pair-aware pip size. Falls back to the FX-major default (0.0001) if no
    pair is given -- this is a heuristic fallback, not a broker-verified
    value. Callers that know the traded symbol should always pass `pair`
    rather than rely on this default (see trend_follow.py's `_stop_pips`
    docstring for why guessing pip size from price magnitude is wrong).
    """
    if not pair:
        return PIP_SIZES["DEFAULT"]
    p = pair.upper()
    if "JPY" in p:
        return PIP_SIZES["JPY"]
    if p.startswith(("XAU", "XAG")):
        return PIP_SIZES["XAU"]
    return PIP_SIZES["DEFAULT"]