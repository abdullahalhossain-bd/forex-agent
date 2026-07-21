# analysis/crossover_signals.py — Vectorized crossover detection
# =============================================================================
# Ported from: https://github.com/JimmyAreaFiscal/MercadoFinanceiro/blob/main/backtesting.py
# Original author: JimmyAreaFiscal — license not specified (public repo)
# Original: backtesting.py → Cruzamentos class
#
# Clean vectorized crossover detection between two pandas Series (e.g., fast
# MA vs slow MA, or price vs MA). Returns a boolean Series where True marks
# the bar where the crossover occurs.
#
# Two functions:
#   cross_above(series_1, series_2, confirmation=False)
#     → True on the bar where series_1 crosses ABOVE series_2
#   cross_below(series_1, series_2, confirmation=False)
#     → True on the bar where series_1 crosses BELOW series_2
#
# If confirmation=True, the function also requires that series_1 remains
# above (or below) series_2 on the CURRENT bar — i.e., the crossover
# happened on the previous bar AND the separation persists. This filters
# out "fake" crossovers that reverse immediately.
#
# Faithful to the original Portuguese-named functions:
#   cruzar_acima  → cross_above
#   cruzar_abaixo → cross_below
# =============================================================================

from __future__ import annotations

import pandas as pd
import os

def cross_above(series_1: pd.Series, series_2: pd.Series,
                confirmation: bool = False) -> pd.Series:
    """
    Detect where series_1 crosses ABOVE series_2.

    A crossover-above is defined as:
        - On bar (t-1): series_1 > series_2  (just crossed above)
        - On bar (t-2): series_1 < series_2  (was below before)

    If `confirmation=True`, additionally require:
        - On bar (t): series_1 > series_2  (still above — not a fake crossover)

    Parameters
    ----------
    series_1, series_2 : pd.Series
        The two series to compare (e.g., fast MA and slow MA).
    confirmation : bool
        If True, require the current bar to confirm the crossover.

    Returns
    -------
    pd.Series of bool — True on crossover bars.
    """
    cond = (series_1.shift(1) > series_2.shift(1)) & \
           (series_1.shift(2) < series_2.shift(2))
    if confirmation:
        cond = cond & (series_1 > series_2)
    return cond


def cross_below(series_1: pd.Series, series_2: pd.Series,
                confirmation: bool = False) -> pd.Series:
    """
    Detect where series_1 crosses BELOW series_2.

    A crossover-below is defined as:
        - On bar (t-1): series_1 < series_2  (just crossed below)
        - On bar (t-2): series_1 > series_2  (was above before)

    If `confirmation=True`, additionally require:
        - On bar (t): series_1 < series_2  (still below — not a fake crossover)
    """
    cond = (series_1.shift(1) < series_2.shift(1)) & \
           (series_1.shift(2) > series_2.shift(2))
    if confirmation:
        cond = cond & (series_1 < series_2)
    return cond


# ── Convenience: golden/death cross ───────────────────────────────────────────

def golden_cross(fast: pd.Series, slow: pd.Series,
                 confirmation: bool = True) -> pd.Series:
    """
    Golden cross: fast MA crosses ABOVE slow MA (bullish signal).
    Default confirmation=True to filter fake crossovers.
    """
    return cross_above(fast, slow, confirmation=confirmation)


def death_cross(fast: pd.Series, slow: pd.Series,
                confirmation: bool = True) -> pd.Series:
    """
    Death cross: fast MA crosses BELOW slow MA (bearish signal).
    Default confirmation=True to filter fake crossovers.
    """
    return cross_below(fast, slow, confirmation=confirmation)


# ── Backwards-compatible class API (matches original Cruzamentos) ─────────────

class Cruzamentos:
    """
    Backwards-compatible class wrapper matching the original Portuguese API.
    All methods are static — use as:
        Cruzamentos.cruzar_acima(s1, s2, confirmacao=False)
        Cruzamentos.cruzar_abaixo(s1, s2, confirmacao=False)
    """

    @staticmethod
    def cruzar_acima(series_1, series_2, confirmacao=False):
        """Portuguese alias for cross_above."""
        return cross_above(series_1, series_2, confirmation=confirmacao)

    @staticmethod
    def cruzar_abaixo(series_1, series_2, confirmacao=False):
        """Portuguese alias for cross_below."""
        return cross_below(series_1, series_2, confirmation=confirmacao)


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    # Build a series where fast crosses above slow at bar 5, below at bar 10
    n = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    slow = pd.Series(np.linspace(100, 100, n), index=idx)  # flat at 100
    fast = pd.Series(np.zeros(n), index=idx)
    fast.iloc[:5] = 95   # below
    fast.iloc[5:10] = 105  # above
    fast.iloc[10:] = 95    # below again

    above = cross_above(fast, slow)
    below = cross_below(fast, slow)
    above_conf = cross_above(fast, slow, confirmation=True)
    below_conf = cross_below(fast, slow, confirmation=True)

    print("Cross above bars:", list(above[above].index))
    print("Cross below bars:", list(below[below].index))
    print("Cross above (confirmed):", list(above_conf[above_conf].index))
    print("Cross below (confirmed):", list(below_conf[below_conf].index))

    # Expect cross_above at bar 6 (shift(1) is bar 5 which is first above, shift(2) is bar 4 which is below)
    assert above.sum() >= 1, "expected at least one cross-above"
    assert below.sum() >= 1, "expected at least one cross-below"

    # Test class API
    assert Cruzamentos.cruzar_acima(fast, slow).equals(above)
    assert Cruzamentos.cruzar_abaixo(fast, slow).equals(below)

    print("\nCrossover signals smoke test passed.")
