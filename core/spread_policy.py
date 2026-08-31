# core/spread_policy.py — Canonical spread measurement + limits
# ============================================================
# Single source of truth for:
#   - converting bid/ask → spread_pips
#   - symbol / asset-class max spread limits
#   - NORMAL / ELEVATED / EXTREME classification
#
# LiveRiskManager, OrderManager, SpreadMonitor, AccountManager,
# and DA safety net should prefer these helpers over hardcoded
# max_spread = 5.0.
# ============================================================

from __future__ import annotations

import os
from typing import Optional, Tuple

# Reuse project pip table when available
try:
    from core.constants import get_pip_size, strip_mt5_suffix, MT5_SYMBOL_SUFFIX
except Exception:  # pragma: no cover — standalone/tests
    MT5_SYMBOL_SUFFIX = os.getenv("MT5_SYMBOL_SUFFIX", "m")

    def strip_mt5_suffix(symbol: str) -> str:
        s = str(symbol or "")
        suf = MT5_SYMBOL_SUFFIX or ""
        if suf and s.lower().endswith(suf.lower()):
            return s[: -len(suf)]
        return s

    def get_pip_size(symbol: str) -> float:
        clean = strip_mt5_suffix(symbol).upper()
        if "XAU" in clean or "XPT" in clean or "XPD" in clean:
            return 0.01
        if "XAG" in clean:
            return 0.001
        if "JPY" in clean:
            return 0.01
        return 0.0001


def clean_symbol(symbol: str) -> str:
    """Bare uppercase pair, broker suffix stripped (EURUSDm → EURUSD)."""
    return strip_mt5_suffix(str(symbol or "")).upper()


def pip_size_for_symbol(symbol: str, digits: Optional[int] = None, point: Optional[float] = None) -> float:
    """Canonical pip size.

    Prefer the project PIP_SIZE table. If MT5 digits/point are supplied and
    the table only has DEFAULT, derive from MT5 convention:
      digits in (3, 5) → pip = 10 * point (pipette symbols)
      else            → pip = point
    """
    clean = clean_symbol(symbol)
    try:
        size = float(get_pip_size(clean) or 0.0)
    except Exception:
        size = 0.0

    # If table has a real entry (not only DEFAULT path unknown), trust it
    # get_pip_size always returns something; refine for unknown exotics via MT5
    if digits is not None and point is not None and point > 0:
        derived = float(point) * (10.0 if int(digits) in (3, 5) else 1.0)
        # Prefer table for known majors/metals; for DEFAULT-like 0.0001 on
        # non-standard names, prefer MT5-derived when it differs a lot.
        if clean not in (
            "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
            "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
            "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD",
        ):
            # Exotic: MT5 digits are authoritative when available
            if derived > 0:
                return derived
        if size <= 0:
            return derived

    if size > 0:
        return size
    if point is not None and point > 0 and digits is not None:
        return float(point) * (10.0 if int(digits) in (3, 5) else 1.0)
    return 0.0001


def spread_price_to_pips(
    bid: float,
    ask: float,
    symbol: str,
    digits: Optional[int] = None,
    point: Optional[float] = None,
) -> Optional[float]:
    """Convert raw bid/ask distance into pips. None if tick invalid."""
    try:
        b = float(bid)
        a = float(ask)
    except (TypeError, ValueError):
        return None
    if b <= 0 or a <= 0 or a < b:
        return None
    pip = pip_size_for_symbol(symbol, digits=digits, point=point)
    if pip <= 0:
        return None
    return (a - b) / pip


# ── Asset-class defaults (operational — not historical percentiles) ──
# Tuned from live Exness snapshot + industry norms. Override via
# SPREAD_LIMITS_PIPS symbol map or env SPREAD_EMERGENCY_MAX_PIPS.

ASSET_CLASS_MAX_SPREAD_PIPS = {
    "major_fx": 4.0,          # EURUSD, GBPUSD, … normal ~0.8–1.5
    "minor_fx": 6.0,          # EURGBP, AUDNZD, …
    "jpy_cross": 30.0,        # USDJPY live ~10; crosses higher
    "nordic_fx": 400.0,       # EURSEK/GBPSEK live 150–320
    "try_fx": 3000.0,         # EURTRY/GBPTRY live 1600–2600
    "zar_fx": 200.0,          # USDZAR live ~40–120; spike room
    "mxn_fx": 120.0,
    "sgd_hkd_cnh": 50.0,
    "metal_gold": 400.0,      # XAUUSD live ~260
    "metal_silver": 60.0,     # XAGUSD live ~30
    "metal_other": 900.0,     # XPT/XPD live 400–600
    "index": 200.0,
    "crypto": 2000.0,
    "default": 25.0,
}

# Explicit symbol overrides (bare names). Wins over asset class.
# Operational defaults — not statistically fitted from multi-day history.
SYMBOL_MAX_SPREAD_PIPS = {
    # Majors
    "EURUSD": 3.0, "GBPUSD": 3.5, "AUDUSD": 3.5, "NZDUSD": 4.0,
    "USDCAD": 4.0, "USDCHF": 4.0, "USDJPY": 20.0,
    # Common minors
    "EURGBP": 4.0, "EURJPY": 25.0, "GBPJPY": 30.0, "AUDJPY": 25.0,
    "EURCHF": 5.0, "EURAUD": 5.0, "EURCAD": 5.0, "EURNZD": 6.0,
    "GBPAUD": 6.0, "GBPCAD": 6.0, "GBPCHF": 5.0, "GBPNZD": 6.0,
    "AUDCAD": 5.0, "AUDCHF": 5.0, "AUDNZD": 5.0,
    "NZDCAD": 5.0, "NZDCHF": 5.0, "CADCHF": 5.0,
    "NZDJPY": 25.0, "CADJPY": 25.0, "CHFJPY": 30.0,
    # ZAR / TRY / SEK / NOK — wide retail quotes
    "USDZAR": 200.0, "EURZAR": 250.0, "GBPZAR": 250.0, "AUDZAR": 200.0,
    "USDTRY": 800.0, "EURTRY": 3000.0, "GBPTRY": 3500.0,
    "EURSEK": 400.0, "GBPSEK": 400.0, "USDSEK": 250.0,
    "EURNOK": 250.0, "GBPNOK": 400.0, "USDNOK": 250.0,
    "USDMXN": 100.0, "USDSGD": 25.0, "USDHKD": 50.0, "USDCNH": 50.0,
    # Metals
    "XAUUSD": 400.0, "XAGUSD": 60.0, "XPTUSD": 900.0, "XPDUSD": 900.0,
}

# Hard ceiling — nothing above this is treated as a real market quote
# for FX majors; exotics use max(symbol_limit * extreme_mult, this only
# as a secondary guard in classify).
EMERGENCY_MAX_SPREAD_PIPS = float(os.getenv("SPREAD_EMERGENCY_MAX_PIPS", "5000"))

# Elevated = above this fraction of the hard limit (warn / optional soft reject)
ELEVATED_FRACTION = float(os.getenv("SPREAD_ELEVATED_FRACTION", "0.70"))
# Extreme = above this fraction of the hard limit (always reject)
EXTREME_FRACTION = float(os.getenv("SPREAD_EXTREME_FRACTION", "1.00"))


def asset_class_for_symbol(symbol: str) -> str:
    s = clean_symbol(symbol)
    if s.startswith("XAU"):
        return "metal_gold"
    if s.startswith("XAG"):
        return "metal_silver"
    if s.startswith(("XPT", "XPD")):
        return "metal_other"
    if any(x in s for x in ("BTC", "ETH", "XRP", "SOL", "BNB")):
        return "crypto"
    if any(x in s for x in ("US30", "US500", "NAS", "USTEC", "DE30", "UK100", "JP225", "HK50", "STOXX")):
        return "index"
    if "TRY" in s:
        return "try_fx"
    if "ZAR" in s:
        return "zar_fx"
    if "MXN" in s:
        return "mxn_fx"
    if any(x in s for x in ("SEK", "NOK", "DKK", "PLN", "HUF", "CZK")):
        return "nordic_fx"
    if any(x in s for x in ("SGD", "HKD", "CNH", "CNY")):
        return "sgd_hkd_cnh"
    if "JPY" in s:
        return "jpy_cross"
    majors = {
        "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
    }
    if s in majors:
        return "major_fx"
    if len(s) == 6 and s.isalpha():
        return "minor_fx"
    return "default"


def get_max_spread_pips(symbol: str) -> float:
    """Hard max spread (pips) for risk gates.

    Resolution order:
      1. SYMBOL_MAX_SPREAD_PIPS override
      2. ASSET_CLASS_MAX_SPREAD_PIPS
      3. default
    Capped by EMERGENCY_MAX_SPREAD_PIPS.
    """
    clean = clean_symbol(symbol)
    if clean in SYMBOL_MAX_SPREAD_PIPS:
        limit = float(SYMBOL_MAX_SPREAD_PIPS[clean])
    else:
        cls = asset_class_for_symbol(clean)
        limit = float(ASSET_CLASS_MAX_SPREAD_PIPS.get(cls, ASSET_CLASS_MAX_SPREAD_PIPS["default"]))
    return min(limit, EMERGENCY_MAX_SPREAD_PIPS)


def classify_spread(spread_pips: float, symbol: str) -> Tuple[str, float]:
    """Return (NORMAL|ELEVATED|EXTREME|INVALID, max_allowed)."""
    max_allowed = get_max_spread_pips(symbol)
    if spread_pips is None or spread_pips < 0:
        return "INVALID", max_allowed
    if spread_pips > max_allowed * EXTREME_FRACTION:
        return "EXTREME", max_allowed
    if spread_pips > max_allowed * ELEVATED_FRACTION:
        return "ELEVATED", max_allowed
    return "NORMAL", max_allowed


def spread_allowed(
    spread_pips: Optional[float],
    symbol: str,
    *,
    reject_elevated: bool = False,
    news_active: bool = False,
    news_multiplier: float = 0.5,
) -> dict:
    """Canonical allow/reject decision for spread gates.

    Returns dict:
      allowed, level, spread_pips, max_allowed_pips, reason
    """
    if spread_pips is None:
        return {
            "allowed": False,
            "level": "INVALID",
            "spread_pips": None,
            "max_allowed_pips": get_max_spread_pips(symbol),
            "reason": "missing/invalid spread (no tick)",
        }
    try:
        sp = float(spread_pips)
    except (TypeError, ValueError):
        return {
            "allowed": False,
            "level": "INVALID",
            "spread_pips": None,
            "max_allowed_pips": get_max_spread_pips(symbol),
            "reason": "missing/invalid spread (no tick)",
        }
    if sp < 0:
        return {
            "allowed": False,
            "level": "INVALID",
            "spread_pips": sp,
            "max_allowed_pips": get_max_spread_pips(symbol),
            "reason": "negative spread",
        }

    max_allowed = get_max_spread_pips(symbol)
    if news_active:
        max_allowed = max_allowed * float(news_multiplier)

    level, _ = classify_spread(sp, symbol)
    # Re-classify against possibly tightened news limit
    if sp > max_allowed:
        level = "EXTREME"
        allowed = False
        reason = f"Spread too high: {sp:.1f} > {max_allowed:.1f} pips ({clean_symbol(symbol)})"
    elif reject_elevated and sp > max_allowed * ELEVATED_FRACTION:
        level = "ELEVATED"
        allowed = False
        reason = f"Spread elevated: {sp:.1f} > {max_allowed * ELEVATED_FRACTION:.1f} pips ({clean_symbol(symbol)})"
    else:
        if sp > max_allowed * ELEVATED_FRACTION:
            level = "ELEVATED"
        else:
            level = "NORMAL"
        allowed = True
        reason = f"{sp:.1f} pips (limit {max_allowed:.1f})"

    return {
        "allowed": allowed,
        "level": level,
        "spread_pips": sp,
        "max_allowed_pips": round(max_allowed, 2),
        "reason": reason,
    }


# Backward-compatible dict for callers expecting SPREAD_LIMITS_PIPS-style maps
def as_limits_dict() -> dict:
    out = dict(SYMBOL_MAX_SPREAD_PIPS)
    out["DEFAULT"] = ASSET_CLASS_MAX_SPREAD_PIPS["default"]
    return out
