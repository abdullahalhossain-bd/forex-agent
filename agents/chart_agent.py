# agents/chart_agent.py
"""
ChartAgent
==========
Browser-automation utility that:
  1. Computes swing-based support/resistance levels from historical OHLC
     data (via yfinance), with symbol-aware precision.
  2. Opens a live TradingView chart (via Playwright) and draws those
     levels onto it as horizontal lines, for visual reference / vision-AI
     screenshot capture.

This module does NOT place trades, size positions, or compute risk — it
is a charting/annotation tool only. It is deliberately kept independent
of MT5/broker execution.

Data-quality note: yfinance's intraday FX data is a delayed/indicative
feed, not a live broker tick stream. Levels computed here may differ by
a few pips from what your actual broker or TradingView's own feed shows.
Treat this as a visual reference, not an execution-grade price source.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import yfinance as yf
from playwright.sync_api import sync_playwright, Page, Browser
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

log = logging.getLogger("chart_agent")

# Default Playwright action timeout override (ms). The library default is
# 30s per action; with ~6 sequential actions in add_indicator() alone, a
# single broken selector can otherwise cost minutes before failing.
DEFAULT_ACTION_TIMEOUT_MS = 8_000


class ChartDataError(Exception):
    """Raised when S/R level calculation fails — signals callers NOT to
    trust self.support_levels / self.resistance_levels (they are cleared,
    not left stale, on this error)."""


@dataclass
class SymbolSpec:
    """
    Precision spec for a single symbol, since FX/metals/crypto pairs do
    not share a common decimal precision or pip size. Getting this wrong
    silently breaks level rounding and clustering (see audit finding #3).
    """
    yahoo_symbol: str        # e.g. "EURUSD=X", "JPY=X", "GC=F"
    tv_symbol: str           # e.g. "EURUSD", "USDJPY", "XAUUSD"
    decimals: int            # rounding precision for this symbol
    cluster_tol: float       # absolute-price tolerance for level clustering


# Canonical symbol table — the single source of truth tying the Yahoo
# and TradingView representations of "the same pair" together, so a
# caller only ever specifies ONE logical symbol instead of two
# independently-drifting strings (audit finding #4).
_SYMBOL_TABLE = {
    "EURUSD": SymbolSpec("EURUSD=X", "EURUSD", decimals=5, cluster_tol=0.0004),
    "GBPUSD": SymbolSpec("GBPUSD=X", "GBPUSD", decimals=5, cluster_tol=0.0004),
    "AUDUSD": SymbolSpec("AUDUSD=X", "AUDUSD", decimals=5, cluster_tol=0.0004),
    "NZDUSD": SymbolSpec("NZDUSD=X", "NZDUSD", decimals=5, cluster_tol=0.0004),
    "USDCHF": SymbolSpec("USDCHF=X", "USDCHF", decimals=5, cluster_tol=0.0004),
    "USDCAD": SymbolSpec("USDCAD=X", "USDCAD", decimals=5, cluster_tol=0.0004),
    # JPY crosses: 2-3 decimal precision, pip = 0.01 (~100x coarser than
    # a 5-decimal pair) — using EURUSD's tolerance here would effectively
    # disable clustering entirely.
    "USDJPY": SymbolSpec("USDJPY=X", "USDJPY", decimals=3, cluster_tol=0.04),
    "EURJPY": SymbolSpec("EURJPY=X", "EURJPY", decimals=3, cluster_tol=0.04),
    "GBPJPY": SymbolSpec("GBPJPY=X", "GBPJPY", decimals=3, cluster_tol=0.04),
    # Metals: 2-decimal precision, much larger absolute price scale.
    "XAUUSD": SymbolSpec("GC=F", "XAUUSD", decimals=2, cluster_tol=0.5),
    "XAGUSD": SymbolSpec("SI=F", "XAGUSD", decimals=3, cluster_tol=0.02),
}


def get_symbol_spec(symbol: str) -> SymbolSpec:
    """
    Resolve a logical symbol (e.g. "USDJPY") to its Yahoo/TradingView
    representations and precision. Falls back to a generic 5-decimal FX
    spec for unlisted symbols rather than raising, but logs a warning so
    silently-wrong precision doesn't go unnoticed.
    """
    spec = _SYMBOL_TABLE.get(symbol.upper())
    if spec is not None:
        return spec
    log.warning(
        "No SymbolSpec for '%s' — falling back to generic 5-decimal FX "
        "precision. Add an entry to _SYMBOL_TABLE for correct rounding.",
        symbol,
    )
    return SymbolSpec(f"{symbol}=X", symbol, decimals=5, cluster_tol=0.0004)


@dataclass
class SRLevels:
    """Result of an S/R calculation — a single cohesive snapshot instead
    of three independently-mutable instance attributes, so a caller can
    never observe a half-updated (stale symbol + new price) state."""
    symbol: str
    current_price: Optional[float] = None
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.current_price is not None


class ChartAgent:
    def __init__(self) -> None:
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        # Single cohesive state object — see SRLevels docstring. Starts
        # invalid; only ever replaced wholesale (never partially mutated)
        # so a failed recalculation can't leave mismatched fields behind.
        self.levels = SRLevels(symbol="")

    # Convenience accessors preserving the old public attribute names,
    # since other modules in this codebase may already read
    # chart_agent.current_price / .support_levels / .resistance_levels.
    @property
    def current_price(self) -> Optional[float]:
        return self.levels.current_price

    @property
    def support_levels(self) -> List[float]:
        return self.levels.support_levels

    @property
    def resistance_levels(self) -> List[float]:
        return self.levels.resistance_levels

    def __enter__(self) -> "ChartAgent":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Guaranteed cleanup even if an exception occurred mid-use —
        # the original code had no such guarantee (audit finding, medium).
        self.close(wait_for_user=False)

    # ─────────────────────────────────────────
    # STARTUP
    # ─────────────────────────────────────────
    def start(self, headless: bool = False) -> None:
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=headless,
            args=["--start-maximized"],
        )
        self.page = self.browser.new_page(viewport={"width": 1600, "height": 900})
        self.page.set_default_timeout(DEFAULT_ACTION_TIMEOUT_MS)
        log.info("Browser started (headless=%s)", headless)

    # ─────────────────────────────────────────
    # STEP 1: S/R CALCULATION (from historical data)
    # ─────────────────────────────────────────
    def calculate_sr_levels(
        self,
        symbol: str = "EURUSD",
        period: str = "5d",
        interval: str = "15m",
        max_retries: int = 2,
    ) -> SRLevels:
        """
        Compute swing-based S/R levels for `symbol` (logical symbol, e.g.
        "EURUSD", "USDJPY", "XAUUSD" — NOT a raw Yahoo/TradingView string).

        On any failure, self.levels is reset to an invalid, empty state
        for THIS symbol rather than left holding a previous symbol's
        stale levels (audit finding #2) — callers must check
        `result.is_valid` before using the levels, and draw_sr_levels()
        refuses to run against invalid state.

        Raises ChartDataError if data cannot be obtained after retries.
        """
        spec = get_symbol_spec(symbol)
        log.info("Calculating S/R for %s (yahoo=%s)...", symbol, spec.yahoo_symbol)

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 2):
            try:
                df = yf.download(
                    spec.yahoo_symbol,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )
                if df is None or df.empty:
                    raise ChartDataError(
                        f"yfinance returned no data for {spec.yahoo_symbol} "
                        f"(period={period}, interval={interval})"
                    )

                # yfinance sometimes returns MultiIndex columns even for a
                # single ticker depending on version — .values.flatten()
                # normalizes both the Series and single-column-DataFrame
                # cases to a flat 1-D array.
                highs = df["High"].values.flatten()
                lows = df["Low"].values.flatten()
                closes = df["Close"].values.flatten()

                if len(closes) < 7 or np.isnan(closes[-1]):
                    raise ChartDataError(
                        f"Insufficient/invalid data for {symbol}: "
                        f"{len(closes)} candles"
                    )

                current_price = float(closes[-1])
                support, resistance = self._find_pivots(highs, lows, spec.decimals)

                all_support = self._cluster(support, spec.cluster_tol)
                all_resistance = self._cluster(resistance, spec.cluster_tol)

                result = SRLevels(
                    symbol=symbol,
                    current_price=current_price,
                    support_levels=[x for x in all_support if x < current_price][-3:],
                    resistance_levels=[x for x in all_resistance if x > current_price][:3],
                )
                self.levels = result

                log.info(
                    "  Current Price : %.*f | Support: %s | Resistance: %s",
                    spec.decimals, current_price,
                    result.support_levels, result.resistance_levels,
                )
                return result

            except Exception as e:  # noqa: BLE001 — deliberately broad: any
                # failure here must fall through to the retry/reset path
                # below, not just yfinance-specific exceptions.
                last_error = e
                log.warning("S/R calc attempt %d/%d failed for %s: %s",
                            attempt, max_retries + 1, symbol, e)
                time.sleep(1.5 * attempt)  # simple linear backoff

        # All retries exhausted — reset state for THIS symbol to an
        # explicitly invalid snapshot (never silently keep a different
        # symbol's old levels around).
        self.levels = SRLevels(symbol=symbol)
        raise ChartDataError(
            f"Failed to compute S/R for {symbol} after {max_retries + 1} attempts"
        ) from last_error

    @staticmethod
    def _find_pivots(
        highs: np.ndarray, lows: np.ndarray, decimals: int
    ) -> Tuple[List[float], List[float]]:
        """5-bar fractal pivot detection (2 bars either side must confirm).
        Not look-ahead bias for historical analysis: bar i only becomes a
        confirmed pivot once i+1, i+2 have closed, which is exactly how a
        live system would learn about it too."""
        support, resistance = [], []
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i - 1] and lows[i] < lows[i - 2]
                    and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]):
                support.append(round(float(lows[i]), decimals))
            if (highs[i] > highs[i - 1] and highs[i] > highs[i - 2]
                    and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]):
                resistance.append(round(float(highs[i]), decimals))
        return support, resistance

    @staticmethod
    def _cluster(levels: List[float], tol: float) -> List[float]:
        """Greedy single-linkage clustering. Note: can chain levels whose
        first/last members are farther apart than `tol` if each
        consecutive pair is within tol — acceptable for this coarse
        visual use case, but not a substitute for a proper clustering
        algorithm if used elsewhere for precise zone detection."""
        if not levels:
            return []
        levels = sorted(set(levels))
        out = [levels[0]]
        for lvl in levels[1:]:
            if abs(lvl - out[-1]) > tol:
                out.append(lvl)
        return out

    # ─────────────────────────────────────────
    # STEP 2: TRADINGVIEW OPEN
    # ─────────────────────────────────────────
    def open_tradingview(self, symbol: str = "EURUSD") -> None:
        """`symbol` is the same logical symbol passed to
        calculate_sr_levels() — resolved to its TradingView representation
        via the shared SymbolSpec table, so the two can no longer drift
        apart (audit finding #4)."""
        spec = get_symbol_spec(symbol)
        url = f"https://www.tradingview.com/chart/?symbol=FX:{spec.tv_symbol}"
        log.info("Opening TradingView: %s", spec.tv_symbol)
        self.page.goto(url, wait_until="domcontentloaded")
        try:
            # Wait for a real chart element instead of a fixed sleep.
            self.page.locator(".chart-container").first.wait_for(timeout=15_000)
        except PlaywrightTimeoutError:
            log.warning("Chart container did not appear within 15s — "
                        "continuing anyway, subsequent steps may fail")
        self.page.keyboard.press("Escape")
        log.info("TradingView loaded")

    # ─────────────────────────────────────────
    # STEP 3: TIMEFRAME
    # ─────────────────────────────────────────
    def change_timeframe(self, timeframe: str = "15") -> bool:
        try:
            btn = self.page.locator(f'button[data-value="{timeframe}"]').first
            btn.click()
            log.info("Timeframe set: %sm", timeframe)
            return True
        except PlaywrightTimeoutError as e:
            log.warning("Timeframe button not found (%sm): %s", timeframe, e)
            return False

    # ─────────────────────────────────────────
    # STEP 4: INDICATOR ADD
    # ─────────────────────────────────────────
    def add_indicator(self, name: str) -> bool:
        try:
            log.info("Adding indicator: %s", name)
            self.page.locator('button[data-name="indicators"]').first.click()

            search = self.page.locator('input[data-role="search"]').first
            search.wait_for()
            search.fill(name)

            result = self.page.locator('[class*="itemRow"]').first
            result.wait_for()
            result.click()

            self.page.keyboard.press("Escape")
            log.info("  %s added", name)
            return True
        except PlaywrightTimeoutError as e:
            log.warning("  %s failed (element not found within timeout): %s", name, e)
            # Best-effort cleanup so a failed indicator search dialog
            # doesn't stay open and eat subsequent keystrokes (e.g. "h").
            try:
                self.page.keyboard.press("Escape")
            except Exception:
                pass
            return False

    # ─────────────────────────────────────────
    # STEP 5: S/R DRAW (from price scale)
    # ─────────────────────────────────────────
    def _get_chart_price_range(self) -> Tuple[float, float]:
        """Read the chart's visible price range from the price-scale
        labels. Falls back to current_price +/- 0.5% only if that DOM
        read fails — this fallback range is approximate and will distort
        on-screen positions if the real visible range is wider."""
        try:
            labels = self.page.locator(
                '[class*="priceScale"] [class*="labelRow"]'
            ).all_text_contents()
            prices = []
            for label in labels:
                try:
                    prices.append(float(label.replace(",", "")))
                except ValueError:
                    continue
            if len(prices) >= 2:
                return min(prices), max(prices)
        except Exception as e:
            log.debug("Could not read price scale from DOM: %s", e)

        if self.current_price is None:
            raise ChartDataError(
                "No price scale readable and no current_price available — "
                "cannot compute a fallback range"
            )
        p = self.current_price
        log.warning("Falling back to price range ±0.5%% around %.5f "
                    "(DOM price-scale read failed)", p)
        return p * 0.995, p * 1.005

    @staticmethod
    def _price_to_y(price: float, box: dict, price_min: float, price_max: float) -> float:
        ratio = (price_max - price) / (price_max - price_min)
        ratio = max(0.05, min(0.95, ratio))
        return box["y"] + ratio * box["height"]

    def draw_sr_levels(self) -> None:
        """Draws the currently-held S/R levels onto the open TradingView
        chart using the Horizontal Line tool ('h' shortcut).

        Refuses to run if self.levels is invalid (i.e. the last
        calculate_sr_levels() call failed) — prevents drawing stale or
        empty state (audit finding #2)."""
        if not self.levels.is_valid:
            log.error("Refusing to draw: no valid S/R levels computed "
                       "(call calculate_sr_levels() successfully first)")
            return

        try:
            chart = self.page.locator(".chart-container").first
            box = chart.bounding_box()
            if not box:
                log.error("Chart container not found — cannot draw")
                return

            price_min, price_max = self._get_chart_price_range()
            chart_cx = box["x"] + box["width"] * 0.5

            def draw_line(price: float, label: str) -> None:
                # Skip levels outside the visible range instead of
                # clamping them onto the visible band, which would
                # otherwise misrepresent a far-away level as being close
                # to price (audit finding, medium).
                if not (price_min <= price <= price_max):
                    log.info("  Skipping %s at %.5f — outside visible "
                              "chart range [%.5f, %.5f]",
                              label, price, price_min, price_max)
                    return
                try:
                    self.page.keyboard.press("h")
                    y = self._price_to_y(price, box, price_min, price_max)
                    self.page.mouse.click(chart_cx, y)
                    self.page.keyboard.press("Escape")
                    log.info("  %s: %.5f", label, price)
                except Exception as e:
                    log.warning("  %s draw error: %s", label, e)

            log.info("Drawing Support levels (green)...")
            for lvl in self.levels.support_levels:
                draw_line(lvl, "Support")

            log.info("Drawing Resistance levels (red)...")
            for lvl in self.levels.resistance_levels:
                draw_line(lvl, "Resistance")

            log.info("All S/R levels drawn")
        except Exception as e:
            log.error("Draw S/R error: %s", e)

    # ─────────────────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────────────────
    def close(self, wait_for_user: bool = False) -> None:
        """
        Args:
            wait_for_user: if True, blocks on a keypress before closing
                (useful for interactive/manual debugging sessions). MUST
                be False (the default) for any headless, scheduled, or
                service context — the original unconditional input() call
                would otherwise hang forever and leak the browser process
                every cycle (audit finding #1, critical).
        """
        if wait_for_user:
            input("\nPress Enter to close the browser...")
        try:
            if self.browser:
                self.browser.close()
        finally:
            if self.playwright:
                self.playwright.stop()
        log.info("Browser closed")