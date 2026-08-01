# backtest/ml_engine.py — Event-driven backtest engine with ML integration
# =============================================================================
# Ported from: https://github.com/MaxwellMendenhall/ml_backtest
# Original: ml_backtest/backtest/backtest.py + ml_backtest/interfaces/interface.py
# Original author: Maxwell Mendenhall — MIT license
#
# Event-driven backtesting engine that supports ML-predicted take-profit
# distances. The flow:
#
#   1. Run a Backtest with a Strategy that uses a FIXED take-profit.
#   2. Collect the resulting trades (entry, exit, PnL, target = high - entry).
#   3. Train a regressor (see ml/optimal_tp_predictor.py) on:
#        X = pattern features at entry bar (see ml/pattern_features.py)
#        y = "target" (how far price moved in favorable direction)
#   4. Re-run the Backtest with the trained model. The Strategy calls
#      strategy.predict(entry_time, cs_features) to get a per-trade TP
#      instead of using the fixed value.
#
# Differences from the original Mendenhall implementation:
#   - Decoupled from the Strategy base class — any class with init() and
#     on_data(index, low, high, close, open, dates) methods works.
#   - Strategy.buy()/sell() are exposed via the Strategy base class (same API).
#   - Dates can be any sortable type (DatetimeIndex, Unix timestamps, strings).
#     The original required Unix timestamps.
#   - The engine tracks max drawdown + profit factor (same as original).
#
# This is INDEPENDENT of backtest/engine.py (the existing event-driven engine).
# Use that for full realism (slippage, partial fills, our risk/ modules).
# Use this for the ML-optimized-TT workflow specifically.
# =============================================================================

from __future__ import annotations

from datetime import datetime, time as dt_time, timezone
from typing import Optional, List

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from utils.logger import get_logger

log = get_logger("ml_engine")


# ── Strategy base class (mirrors ml_backtest/interfaces/interface.py:Strategy) ─

class Strategy:
    """
    Base class for ML-integrated strategies. Subclass and override:
        init()        — set market hours, etc.
        on_data(...)  — called per bar; call self.buy() / self.sell() to enter

    When an ML model is loaded (via set_ml), call self.predict(entry_time,
    cs_features) to get a per-trade TP prediction instead of using a fixed TP.
    """

    def __init__(self):
        self.positions: list[dict] = []
        self.in_position: bool = False
        self.model = None
        self._columns: Optional[List[str]] = None
        self._rows: Optional[int] = None
        self._df: Optional[pd.DataFrame] = None
        self.market_open_time: Optional[dt_time] = None
        self.market_close_time: Optional[dt_time] = None
        self.cs_patterns: bool = False

    def init(self):
        raise NotImplementedError("Method 'init()' must be defined in the subclass.")

    def on_data(self, index, low, high, close, open, dates):
        raise NotImplementedError("Method 'on_data()' must be defined in the subclass.")

    # ── Order placement (final — do not override) ────────────────────────────

    def buy(self, price: float, take_profit: Optional[float] = None,
            stop_loss: Optional[float] = None, entry_time=None,
            metadata: Optional[dict] = None):
        if not self.in_position:
            position = {
                'type': 'long',
                'entry price': price,
                'take profit': take_profit,
                'stop loss': stop_loss,
                'entry time': entry_time,
            }
            if metadata is not None:
                position['metadata'] = metadata
            self.positions.append(position)
            self.in_position = True

    def sell(self, price: float, take_profit: Optional[float] = None,
             stop_loss: Optional[float] = None, entry_time=None,
             metadata: Optional[dict] = None):
        if not self.in_position:
            position = {
                'type': 'short',
                'entry price': price,
                'take profit': take_profit,
                'stop loss': stop_loss,
                'entry time': entry_time,
            }
            if metadata is not None:
                position['metadata'] = metadata
            self.positions.append(position)
            self.in_position = True

    # ── ML integration ───────────────────────────────────────────────────────

    def set_ml(self, model=None, columns=None, rows=None, df=None, cs_pattern=False):
        self.model = model
        self._columns = columns
        self._rows = rows
        self._df = df
        self.cs_patterns = cs_pattern

    def predict(self, current_entry_time, cs_features: Optional[np.ndarray] = None) -> float:
        """
        Predict the take-profit distance for a trade entered at `current_entry_time`.
        Uses the ML model + the lookback-window feature extraction from
        ml_backtest's DataProcessing.process_entries().
        """
        if self.model is None or self._columns is None or self._rows is None or self._df is None:
            raise ValueError("Model, columns, rows, or DataFrame not set")

        # Convert df to numpy with date as Unix timestamp in column 0
        df = self._df.copy()
        if 'date' in df.columns and not np.issubdtype(df['date'].dtype, np.integer):
            df['date'] = pd.to_datetime(df['date']).astype('int64') // 10**9
        df_np = df.to_numpy()
        column_indices = [df.columns.get_loc(c) for c in self._columns]
        entry_times = np.array([current_entry_time])

        # Find the index in df_np where date <= entry_time
        indices = np.where(df_np[:, 0] <= entry_times[0])[0]
        if indices.size == 0:
            return 0.0
        matching_index = indices[-1]
        start_index = max(0, matching_index - self._rows + 1)
        before_data = df_np[start_index:matching_index + 1, column_indices]
        single_row = before_data.flatten()

        if cs_features is not None:
            single_row = np.concatenate([single_row, cs_features])

        # Predict
        prediction = self.model.predict(single_row.reshape(1, -1))
        if prediction.size == 1:
            return float(prediction.item())
        return float(prediction[0])

    # ── Trading hours filter ─────────────────────────────────────────────────

    def trading_hours(self, date) -> bool:
        if self.market_open_time is None or self.market_close_time is None:
            return True  # no filter
        if isinstance(date, (np.integer, int)):
            dt_obj = datetime.fromtimestamp(int(date), tz=timezone.utc)
            current_time = dt_obj.time()
        elif isinstance(date, str):
            try:
                dt_obj = datetime.strptime(date, "%m/%d/%Y %I:%M:%S %p")
                current_time = dt_obj.time()
            except ValueError:
                try:
                    dt_obj = pd.to_datetime(date).to_pydatetime()
                    current_time = dt_obj.time()
                except Exception:
                    return False
        elif isinstance(date, (datetime, pd.Timestamp)):
            current_time = pd.Timestamp(date).time()
        else:
            return False
        return self.market_open_time <= current_time <= self.market_close_time


# ── Backtest engine ──────────────────────────────────────────────────────────

class MLBacktest:
    """
    Event-driven backtest with optional ML model for per-trade TP prediction.

    Usage
    -----
    >>> strategy = MyStrategy()
    >>> bt = MLBacktest(df, strategy)
    >>> print(bt.get_results())
    >>> trades = bt.get_trades()  # feed into ml.optimal_tp_predictor

    Then train a model and re-run:
    >>> strategy2 = MyStrategy()
    >>> bt2 = MLBacktest(df_with_features, strategy2, model=model,
    ...                  columns=['EMA_Diff', 'SMA_Diff', 'MACD_hist'],
    ...                  rows=10, cs_pattern=True)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        initial_cash: float = 100_000,
        model=None,
        columns: Optional[List[str]] = None,
        rows: Optional[int] = None,
        cs_pattern: bool = False,
        show_progress: bool = False,
    ):
        self._data = data
        self._strategy = strategy
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._equity_peak = initial_cash
        self._max_drawdown = 0.0
        self._open_positions: list[dict] = []
        self._completed_trades: list[dict] = []
        self._trade_counter = 0
        self._long_wins = self._short_wins = 0
        self._long_loses = self._short_loses = 0
        self._long_evens = self._short_evens = 0
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        self._start_time = None
        self._end_time = None
        self._model = model
        self._columns = columns
        self._rows = rows
        self._cs_pattern = cs_pattern
        self._results_df: Optional[pd.DataFrame] = None

        self._run(show_progress=show_progress)
        self._compute_results()

    # ── Main loop ────────────────────────────────────────────────────────────

    def _run(self, show_progress: bool = False):
        # Ensure 'date' column is in Unix-timestamp form for ML predict()
        df = self._data.copy()
        if 'date' not in df.columns:
            raise ValueError("DataFrame must have a 'date' column")
        # Check if date is already integer (Unix timestamp); if not, convert
        try:
            if not np.issubdtype(df['date'].dtype, np.integer):
                df['date'] = pd.to_datetime(df['date']).astype('int64') // 10**9
        except TypeError:
            # datetime64 with timezone — convert via Timestamp
            df['date'] = pd.to_datetime(df['date']).astype('int64') // 10**9

        dates = df['date'].values
        lows = df['low'].values
        highs = df['high'].values
        close = df['close'].values
        open_ = df['open'].values

        self._start_time = dates[0]
        self._end_time = dates[-1]

        self._strategy.init()

        if self._columns is not None and self._rows is not None and self._model is not None:
            self._strategy.set_ml(
                model=self._model, columns=self._columns, rows=self._rows,
                df=df, cs_pattern=self._cs_pattern,
            )

        iterator = range(len(dates))
        if show_progress and _HAS_TQDM:
            iterator = tqdm(iterator, total=len(dates))

        for index in iterator:
            self._strategy.on_data(
                index, lows[:index + 1], highs[:index + 1],
                close[:index + 1], open_[:index + 1], dates[:index + 1],
            )

            for position in list(self._strategy.positions):
                exit_price = None
                if 'highest high' not in position or highs[index] > position['highest high']:
                    position['highest high'] = highs[index]
                    position['target'] = position['highest high'] - position['entry price']

                if position['type'] == 'long':
                    if position.get('stop loss') is not None and lows[index] <= position['stop loss']:
                        exit_price = position['stop loss']
                    elif position.get('take profit') is not None and highs[index] >= position['take profit']:
                        exit_price = position['take profit']
                else:  # short
                    if position.get('stop loss') is not None and highs[index] >= position['stop loss']:
                        exit_price = position['stop loss']
                    elif position.get('take profit') is not None and lows[index] <= position['take profit']:
                        exit_price = position['take profit']

                if exit_price is not None:
                    position['exit price'] = exit_price
                    position['exit time'] = dates[index]
                    self._close_position(position, exit_price)
                    self._trade_counter += 1

    def _close_position(self, position: dict, current_price: float):
        # Round-21 audit fix: P/L dimensional bug.
        # Previously: self._cash += profit_loss where profit_loss was a
        # raw price difference (e.g. 0.0010 for 10 pips on EURUSD).
        # Adding 0.0010 to self._cash ($100,000) is dimensionally wrong.
        # Real P/L = price_diff × position_size × contract_size.
        # Default contract_size = 100,000 units per 1.0 lot (standard lot).
        # Default position_size = 1.0 lot (hardcoded below at L336).
        CONTRACT_SIZE = 100_000  # standard lot = 100,000 units
        position_size = position.get('size', 1.0)  # lot size

        if position['type'] == 'long':
            price_diff = current_price - position['entry price']
            profit_loss = price_diff * position_size * CONTRACT_SIZE
            if profit_loss > 0:
                self._long_wins += 1
                self._gross_profit += profit_loss
            elif profit_loss < 0:
                self._long_loses += 1
                self._gross_loss += profit_loss
            else:
                self._long_evens += 1
        else:
            price_diff = position['entry price'] - current_price
            profit_loss = price_diff * position_size * CONTRACT_SIZE
            if profit_loss > 0:
                self._short_wins += 1
                self._gross_profit += profit_loss
            elif profit_loss < 0:
                self._short_loses += 1
                self._gross_loss += profit_loss
            else:
                self._short_evens += 1

        self._cash += profit_loss

        if self._cash > self._equity_peak:
            self._equity_peak = self._cash
        else:
            drawdown = self._equity_peak - self._cash
            self._max_drawdown = max(self._max_drawdown, drawdown)

        self._strategy.positions.remove(position)
        position['exit price'] = current_price
        position['size'] = 1
        position['pnl'] = profit_loss
        self._completed_trades.append(position)
        self._strategy.in_position = False

    # ── Results ──────────────────────────────────────────────────────────────

    def _compute_results(self):
        wins = self._short_wins + self._long_wins
        loses = self._short_loses + self._long_loses
        total = wins + loses
        win_rate = (wins / total * 100) if total > 0 else 0.0
        profit_factor = (
            self._gross_profit / abs(self._gross_loss)
            if self._gross_loss != 0 else float('inf')
        )
        self._results_dict = [{
            'start time': self._start_time,
            'end time': self._end_time,
            '# of trades': self._trade_counter,
            '# of wins': wins,
            '# of loses': loses,
            'win rate': f"{round(win_rate, 2)}%",
            '# of long wins': self._long_wins,
            '# of long loses': self._long_loses,
            '# of long evens': self._long_evens,
            '# of short wins': self._short_wins,
            '# of short loses': self._short_loses,
            '# of short evens': self._short_evens,
            'net profit': round(self._gross_profit + self._gross_loss, 2),
            'max drawdown': f"-{round(self._max_drawdown, 2)}",
            'gross profit': round(self._gross_profit, 2),
            'gross loss': round(self._gross_loss, 2),
            'profit factor': round(profit_factor, 2) if profit_factor != float('inf') else 'inf',
            'final cash': round(self._cash, 2),
        }]
        self._results_df = pd.DataFrame(self._results_dict).T

    def get_trades(self) -> pd.DataFrame:
        """Return completed trades as a DataFrame. Use as ML training input."""
        return pd.DataFrame(self._completed_trades)

    def get_results(self) -> pd.DataFrame:
        """Return backtest summary stats as a transposed DataFrame."""
        return self._results_df
