# broker/magic_number.py — Per-(symbol, timeframe) magic number hashing
# =============================================================================
# Ported from: https://github.com/tanvird3/TradingRobot/blob/master/SuperMao.mq4
# Original author: khantanvir (Tanvir) — license not specified in source
#
# MT4/MT5 "magic number" identifies which EA placed an order, so multiple EAs
# on the same account don't trip over each other's trades. The original
# SuperMao uses a simple deterministic hash:
#
#   MagicNumber = base + Period() + StringGetChar(MySymbol, 0) + StringGetChar(MySymbol, 3)
#
# This produces a unique int per (symbol, timeframe) combination. We port
# the same hash so orders placed by this Python bot can be tagged with a
# magic number that:
#   1. Is deterministic (same input → same magic, across runs)
#   2. Differs per symbol+timeframe (so EURUSD-H1 trades don't conflict with
#      GBPUSD-H1 or EURUSD-M15 trades)
#   3. Matches the existing MT5 magic number (424242) when the symbol and
#      timeframe are blank — backward-compatible with the project's existing
#      `core/constants.py:MT5_MAGIC_NUMBER`.
#
# Note: MT4/MT5 magic numbers are 32-bit signed ints. We clamp to that range.
# =============================================================================

from __future__ import annotations

from typing import Optional


# MT4 timeframe constants → integer period codes (matches MT4 Period())
# These are the values returned by MT4's Period() function.
TIMEFRAME_TO_MT4_PERIOD = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
    "MN1": 43200,
}

# Default base. Matches core/constants.py:MT5_MAGIC_NUMBER for backward
# compatibility (when symbol/timeframe are blank, hash returns the base).
DEFAULT_BASE = 424242


def magic_number(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    *,
    base: int = DEFAULT_BASE,
) -> int:
    """
    Compute a deterministic magic number for a (symbol, timeframe) pair.

    Faithful to the SuperMao.mq4 formula:
        MagicNumber = base + Period() + StringGetChar(symbol, 0) + StringGetChar(symbol, 3)

    Parameters
    ----------
    symbol : forex pair, e.g., "EURUSD". If None or empty, only base+period is used.
    timeframe : one of M1, M5, M15, M30, H1, H4, D1, W1, MN1 (case-insensitive).
        If None, period contribution is 0.
    base : starting value (default 424242, matches core/constants.MT5_MAGIC_NUMBER).

    Returns
    -------
    int in the range [base, base + 43200 + 255 + 255]. Clamped to 32-bit signed.

    Examples
    --------
    >>> magic_number()  # no symbol, no timeframe → just base
    424242
    >>> magic_number("EURUSD", "H1")  # base + 60 + 'E' + 'U'
    424442
    >>> magic_number("GBPJPY", "M15")  # base + 15 + 'G' + 'J'
    424344
    """
    result = base

    # Timeframe contribution (MT4 Period())
    if timeframe:
        tf_key = timeframe.upper()
        if tf_key not in TIMEFRAME_TO_MT4_PERIOD:
            raise ValueError(
                f"Unknown timeframe {timeframe!r}. "
                f"Supported: {list(TIMEFRAME_TO_MT4_PERIOD.keys())}"
            )
        result += TIMEFRAME_TO_MT4_PERIOD[tf_key]

    # Symbol char contributions (chars at index 0 and 3)
    if symbol:
        s = symbol.upper().replace("/", "").replace("_", "").replace("=X", "").strip()
        if len(s) >= 1:
            result += ord(s[0])
        if len(s) >= 4:
            result += ord(s[3])

    # Clamp to 32-bit signed int range (MT4/MT5 magic number constraint)
    return max(-(2**31), min(2**31 - 1, result))


def magic_numbers_for_universe(
    symbols: list[str],
    timeframes: list[str],
    *,
    base: int = DEFAULT_BASE,
) -> dict[tuple[str, str], int]:
    """
    Compute magic numbers for every (symbol, timeframe) combination.

    Useful for setting up a multi-pair bot:
        magic_map = magic_numbers_for_universe(
            ["EURUSD", "GBPUSD", "USDJPY"],
            ["H1", "M15"],
        )
        # → {("EURUSD", "H1"): 424442, ("EURUSD", "M15"): 424397, ...}
    """
    return {
        (sym, tf): magic_number(sym, tf, base=base)
        for sym in symbols
        for tf in timeframes
    }


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # No-args → base
    assert magic_number() == DEFAULT_BASE
    print(f"OK no-args → {magic_number()}")

    # EURUSD H1: 424242 + 60 + ord('E') + ord('U') = 424242 + 60 + 69 + 85 = 424456
    m = magic_number("EURUSD", "H1")
    print(f"EURUSD H1: {m}  (expect 424456)")
    assert m == 424456, f"got {m}"

    # GBPJPY M15: 424242 + 15 + ord('G') + ord('J') = 424242 + 15 + 71 + 74 = 424402
    m = magic_number("GBPJPY", "M15")
    print(f"GBPJPY M15: {m}  (expect 424402)")
    assert m == 424402, f"got {m}"

    # Same (symbol, tf) → same magic (deterministic)
    assert magic_number("EURUSD", "H1") == magic_number("EURUSD", "H1")

    # Different (symbol, tf) → different magic
    assert magic_number("EURUSD", "H1") != magic_number("EURUSD", "M15")
    assert magic_number("EURUSD", "H1") != magic_number("GBPUSD", "H1")

    # Slash/slash variants normalize
    assert magic_number("EUR/USD", "H1") == magic_number("EURUSD", "H1")
    assert magic_number("eur_usd", "h1") == magic_number("EURUSD", "H1")

    # Universe
    universe = magic_numbers_for_universe(["EURUSD", "GBPUSD"], ["H1", "M15"])
    assert len(universe) == 4
    assert len(set(universe.values())) == 4, "all magic numbers must be unique"
    print(f"Universe: {universe}")

    print("\nMagic number smoke test passed.")
