"""
utils/pip_utils.py

Compatibility helper.
Provides safe dependency checking and installation functions.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Iterable


def is_installed(package: str) -> bool:
    """Return True if package can be imported."""
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False


def install(package: str) -> bool:
    """Install a package using pip."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package]
        )
        return True
    except Exception:
        return False


def ensure(package: str) -> bool:
    """Ensure package is installed."""
    if is_installed(package):
        return True
    return install(package)


def ensure_many(packages: Iterable[str]) -> bool:
    """Ensure multiple packages are installed."""
    ok = True
    for pkg in packages:
        if not ensure(pkg):
            ok = False
    return ok


# ==========================================================
# Forex Pip Utilities
# ==========================================================

def pip_size(symbol: str | None = None) -> float:
    """
    Return the pip size for a trading symbol.
    """

    if not symbol:
        return 0.0001

    symbol = symbol.upper()

    # Forex JPY pairs
    if "JPY" in symbol:
        return 0.01

    # Gold
    if symbol.startswith(("XAU", "GOLD")):
        return 0.1

    # Silver
    if symbol.startswith(("XAG", "SILVER")):
        return 0.01

    # Crypto
    if symbol.startswith(("BTC", "ETH", "SOL", "BNB")):
        return 1.0

    # Indices
    if symbol.startswith(("US30", "NAS100", "SPX", "GER40")):
        return 1.0

    # Default Forex
    return 0.0001


def price_to_pips(price_difference: float, symbol: str) -> float:
    """
    Convert price difference into pips.
    """

    return price_difference / pip_size(symbol)


def pips_to_price(pips: float, symbol: str) -> float:
    """
    Convert pips into price movement.
    """

    return pips * pip_size(symbol)