# backtest/vectorized_strategies.py — Vectorized strategy implementations
# =============================================================================
# Ported from FXBot (https://github.com/trentstauff/FXBot) backtesting/ folder:
#   - SMABacktest.py        → SMABacktest
#   - BollingerBandsBacktest.py → BollingerBandsBacktest
#   - MomentumBacktest.py   → MomentumBacktest
#   - ContrarianBacktest.py → ContrarianBacktest
# Original author: Trent Stauffner — MIT license (inferred)
#
# Each class:
#   - Inherits from backtest.vectorized_base.VectorizedBacktester
#   - Implements prepare_data(), test(), and (where useful) optimize()
#   - Returns (performance, out_performance) from test()
#   - Stores full results DataFrame in self._results for plot_results()
#
# Vectorized = fast. Use these for parameter optimization; use the event-driven
# backtest/engine.py for realistic simulation (slippage, partial fills, etc.).
# =============================================================================

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from backtest.vectorized_base import VectorizedBacktester
from utils.logger import get_logger

log = get_logger("vectorized_strategies")


# ── SMA Crossover ────────────────────────────────────────────────────────────

class SMABacktest(VectorizedBacktester):
    """Simple Moving Average crossover. BUY when SMAS > SMAL, SELL otherwise."""

    def __init__(self, *args, smas: int = 10, smal: int = 50, **kwargs):
        if smas >= smal:
            raise ValueError(f"smas ({smas}) must be < smal ({smal})")
        self.smas = smas
        self.smal = smal
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return (f"SMABacktest(symbol={self.symbol}, start={self.start}, "
                f"end={self.end}, smas={self.smas}, smal={self.smal}, "
                f"granularity={self.granularity}, trading_cost={self.trading_cost})")

    def prepare_data(self) -> pd.DataFrame:
        df = self._data.copy()
        df["smas"] = df["price"].rolling(self.smas).mean()
        df["smal"] = df["price"].rolling(self.smal).mean()
        return df

    def set_params(self, smas: int = None, smal: int = None):
        if smas is not None:
            self.smas = smas
            self._data["smas"] = self._data["price"].rolling(self.smas).mean()
        if smal is not None:
            self.smal = smal
            self._data["smal"] = self._data["price"].rolling(self.smal).mean()

    def test(self, mute: bool = False) -> Tuple[float, float]:
        if not mute:
            log.info(f"Testing SMA strategy with smas={self.smas}, smal={self.smal} ...")
        data = self._data.copy()
        data["position"] = np.where(data["smas"] > data["smal"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        if data.empty:
            log.warning("No data after dropna — check warmup period vs data length.")
            return 1.0, 0.0
        # Re-attach to self._data so _compute_performance has the position column
        self._data = data
        perf, outperf, n_trades = self._compute_performance(data)
        if not mute:
            log.info(f"Return: {(perf-1)*100:.2f}%, OutPerformance: {outperf*100:.2f}%, Trades: {n_trades}")
        return perf, outperf

    def optimize(self, smas_range=range(5, 50), smal_range=range(50, 200),
                 mute: bool = False) -> Tuple[float, int, int]:
        """Grid-search smas × smal. Returns (max_return, best_smas, best_smal)."""
        if not mute:
            log.info(f"Optimizing SMA over {len(smas_range) * len(smal_range)} combos ...")
        max_return = -np.inf
        best_smas = best_smal = -1
        total = len(smas_range) * len(smal_range)
        done = 0
        for smas in smas_range:
            for smal in smal_range:
                if smas >= smal:
                    continue
                self.set_params(smas=smas, smal=smal)
                ret = self.test(mute=True)[0]
                if ret > max_return:
                    max_return = ret
                    best_smas, best_smal = smas, smal
                done += 1
                if not mute and done % max(1, total // 10) == 0:
                    log.info(f"  {done}/{total} ({100*done/total:.0f}%) ...")
        self.set_params(smas=best_smas, smal=best_smal)
        self.test(mute=True)
        if not mute:
            log.info(f"Optimal: smas={best_smas}, smal={best_smal}, "
                     f"return={(max_return-1)*100:.2f}%")
        return max_return, best_smas, best_smal


# ── Bollinger Bands Reversion ────────────────────────────────────────────────

class BollingerBandsBacktest(VectorizedBacktester):
    """Bollinger Bands mean-reversion. BUY at lower band, SELL at upper band."""

    def __init__(self, *args, sma: int = 20, deviation: float = 2.0, **kwargs):
        self.sma = sma
        self.deviation = deviation
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return (f"BollingerBandsBacktest(symbol={self.symbol}, sma={self.sma}, "
                f"deviation={self.deviation}, granularity={self.granularity})")

    def prepare_data(self) -> pd.DataFrame:
        df = self._data.copy()
        df["sma"] = df["price"].rolling(self.sma).mean()
        df["std"] = df["price"].rolling(self.sma).std()
        df["lower"] = df["sma"] - df["std"] * self.deviation
        df["upper"] = df["sma"] + df["std"] * self.deviation
        return df

    def set_params(self, sma: int = None, deviation: float = None):
        if sma is not None:
            self.sma = sma
            self._data["sma"] = self._data["price"].rolling(self.sma).mean()
            self._data["std"] = self._data["price"].rolling(self.sma).std()
            self._data["lower"] = self._data["sma"] - self._data["std"] * self.deviation
            self._data["upper"] = self._data["sma"] + self._data["std"] * self.deviation
        if deviation is not None:
            self.deviation = deviation
            self._data["lower"] = self._data["sma"] - self._data["std"] * self.deviation
            self._data["upper"] = self._data["sma"] + self._data["std"] * self.deviation

    def test(self, mute: bool = False) -> Tuple[float, float]:
        if not mute:
            log.info(f"Testing BB strategy with sma={self.sma}, deviation={self.deviation} ...")
        data = self._data.copy()
        # Position: +1 when price < lower, -1 when price > upper, else hold previous
        # Round-21 audit fix: vectorized using np.select + ffill.
        # Previously: O(n) Python loop with chained .iloc[i] assignment
        # (triggers FutureWarning on pandas >= 2.1, error on 3.0).
        # Now: vectorized ~100× faster + no FutureWarning.
        signal = np.select(
            [data["price"] < data["lower"], data["price"] > data["upper"]],
            [1, -1],
            default=np.nan,
        )
        position = pd.Series(signal, index=data.index).ffill().fillna(0).astype(int)
        data["position"] = position
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        if data.empty:
            log.warning("No data after dropna — check warmup period.")
            return 1.0, 0.0
        self._data = data
        perf, outperf, n_trades = self._compute_performance(data)
        if not mute:
            log.info(f"Return: {(perf-1)*100:.2f}%, OutPerformance: {outperf*100:.2f}%, Trades: {n_trades}")
        return perf, outperf

    def optimize(self, sma_range=range(10, 50), dev_range=(1.5, 2.0, 2.5, 3.0),
                 mute: bool = False) -> Tuple[float, int, float]:
        max_return = -np.inf
        best_sma, best_dev = -1, -1
        for sma in sma_range:
            for dev in dev_range:
                self.set_params(sma=sma, deviation=dev)
                ret = self.test(mute=True)[0]
                if ret > max_return:
                    max_return = ret
                    best_sma, best_dev = sma, dev
        self.set_params(sma=best_sma, deviation=best_dev)
        self.test(mute=True)
        if not mute:
            log.info(f"Optimal: sma={best_sma}, dev={best_dev}, return={(max_return-1)*100:.2f}%")
        return max_return, best_sma, best_dev


# ── Momentum ─────────────────────────────────────────────────────────────────

class MomentumBacktest(VectorizedBacktester):
    """
    Momentum: BUY when recent returns are positive (trend continues),
    SELL when negative. Sensitive to window — combine with other strategies.
    """

    def __init__(self, *args, window: int = 3, **kwargs):
        self.window = window
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return (f"MomentumBacktest(symbol={self.symbol}, window={self.window}, "
                f"granularity={self.granularity})")

    def prepare_data(self) -> pd.DataFrame:
        df = self._data.copy()
        df["rolling_returns"] = df["returns"].rolling(self.window).mean()
        return df

    def set_params(self, window: int):
        self.window = window
        self._data["rolling_returns"] = self._data["returns"].rolling(self.window).mean()

    def test(self, mute: bool = False) -> Tuple[float, float]:
        if not mute:
            log.info(f"Testing Momentum strategy with window={self.window} ...")
        data = self._data.copy()
        data["position"] = np.where(data["rolling_returns"] > 0, 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        if data.empty:
            log.warning("No data after dropna — check warmup.")
            return 1.0, 0.0
        self._data = data
        perf, outperf, n_trades = self._compute_performance(data)
        if not mute:
            log.info(f"Return: {(perf-1)*100:.2f}%, OutPerformance: {outperf*100:.2f}%, Trades: {n_trades}")
        return perf, outperf

    def optimize(self, window_range=range(1, 20),
                 mute: bool = False) -> Tuple[float, int]:
        max_return = -np.inf
        best_window = -1
        for w in window_range:
            self.set_params(w)
            ret = self.test(mute=True)[0]
            if ret > max_return:
                max_return = ret
                best_window = w
        self.set_params(best_window)
        self.test(mute=True)
        if not mute:
            log.info(f"Optimal: window={best_window}, return={(max_return-1)*100:.2f}%")
        return max_return, best_window


# ── Contrarian ───────────────────────────────────────────────────────────────

class ContrarianBacktest(VectorizedBacktester):
    """
    Contrarian: BUY when recent returns are negative (expect reversal),
    SELL when positive. The mirror of Momentum.
    """

    def __init__(self, *args, window: int = 3, **kwargs):
        self.window = window
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return (f"ContrarianBacktest(symbol={self.symbol}, window={self.window}, "
                f"granularity={self.granularity})")

    def prepare_data(self) -> pd.DataFrame:
        df = self._data.copy()
        df["rolling_returns"] = df["returns"].rolling(self.window).mean()
        return df

    def set_params(self, window: int):
        self.window = window
        self._data["rolling_returns"] = self._data["returns"].rolling(self.window).mean()

    def test(self, mute: bool = False) -> Tuple[float, float]:
        if not mute:
            log.info(f"Testing Contrarian strategy with window={self.window} ...")
        data = self._data.copy()
        # Note: opposite sign of Momentum
        data["position"] = np.where(data["rolling_returns"] <= 0, 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        if data.empty:
            log.warning("No data after dropna — check warmup.")
            return 1.0, 0.0
        self._data = data
        perf, outperf, n_trades = self._compute_performance(data)
        if not mute:
            log.info(f"Return: {(perf-1)*100:.2f}%, OutPerformance: {outperf*100:.2f}%, Trades: {n_trades}")
        return perf, outperf

    def optimize(self, window_range=range(1, 20),
                 mute: bool = False) -> Tuple[float, int]:
        max_return = -np.inf
        best_window = -1
        for w in window_range:
            self.set_params(w)
            ret = self.test(mute=True)[0]
            if ret > max_return:
                max_return = ret
                best_window = w
        self.set_params(best_window)
        self.test(mute=True)
        if not mute:
            log.info(f"Optimal: window={best_window}, return={(max_return-1)*100:.2f}%")
        return max_return, best_window
