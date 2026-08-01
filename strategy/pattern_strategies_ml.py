# strategies/pattern_strategies_ml.py — ML-integrated pattern strategies
# =============================================================================
# Ported from: https://github.com/MaxwellMendenhall/ml_backtest
# Original: ml_backtest/strategies/*.py
# Original author: Maxwell Mendenhall — MIT license
#
# Five candlestick-pattern strategies that integrate with the ML-optimized
# take-profit workflow. Each strategy:
#   1. Detects its pattern on every bar (using analysis.candlestick_patterns_ml).
#   2. If pattern fires and not in position → BUY with metadata for ML.
#   3. Take-profit is FIXED (entry + 50) when no ML model is loaded.
#   4. Take-profit is ML-PREDICTED when a model is loaded (entry + prediction).
#
# Workflow
# --------
#   # 1. Run a baseline backtest with fixed TP
#   strategy = HammerStrategy()
#   bt = MLBacktest(df, strategy)
#   trades = bt.get_trades()
#
#   # 2. Train an ML model on the trades
#   from ml.optimal_tp_predictor import train_tp_predictor_from_trades
#   predictor = train_tp_predictor_from_trades(trades, 'Hammer',
#                                              save_path='hammer_tp_model')
#
#   # 3. Re-run with the ML model
#   strategy2 = HammerStrategy()
#   bt2 = MLBacktest(df, strategy2, model=predictor.model,
#                    columns=['SMA_Diff', 'EMA_Diff', 'MACD_hist'], rows=10,
#                    cs_pattern=True)
#   print(bt2.get_results())  # → typically higher win rate + larger avg winner
# =============================================================================

from __future__ import annotations

from datetime import time as dt_time

import numpy as np

from backtest.ml_engine import Strategy
from analysis.candlestick_patterns_ml import CandleStickPatterns
from ml.pattern_features import (
    hammer_features, inverted_hammer_features, dragonfly_doji_features,
    engulfing_features, harami_features, piercing_pattern_features,
    morning_star_features, morning_star_doji_features,
)


# ── Default fixed TP/SL (matches original Mendenhall values) ─────────────────
# These are used when no ML model is loaded. They're in price units (not pips).
DEFAULT_TP_DISTANCE = 50.0   # entry + 50
DEFAULT_SL_DISTANCE = 37.0   # entry - 37


# ── 1-bar pattern strategies ─────────────────────────────────────────────────

class HammerStrategy(Strategy):
    """Long entry on Hammer pattern. Fixed or ML-predicted TP."""

    def init(self):
        # Market hours default: 24/5 (forex) — no filter
        self.market_open_time = dt_time(0, 0)
        self.market_close_time = dt_time(23, 59)

    def on_data(self, index, low, high, close, open, dates):
        if index < 1:
            return
        current_open = open[index]
        current_close = close[index]
        current_high = high[index]
        current_low = low[index]
        current_date = dates[index]

        if (CandleStickPatterns.is_hammer(current_open, current_close,
                                           current_high, current_low)
                and not self.in_position
                and self.trading_hours(current_date)):
            metadata = {
                'current_open': current_open,
                'current_close': current_close,
                'current_high': current_high,
                'current_low': current_low,
            }
            entry_price = current_close

            if self.model is not None:
                if self.cs_patterns:
                    cs_features = hammer_features(**metadata).reshape(1, -1)
                    prediction = self.predict(current_date, cs_features)
                else:
                    prediction = self.predict(current_date)
                take_profit = entry_price + prediction
            else:
                take_profit = entry_price + DEFAULT_TP_DISTANCE

            stop_loss = entry_price - DEFAULT_SL_DISTANCE
            self.buy(price=entry_price, take_profit=take_profit,
                     stop_loss=stop_loss, entry_time=current_date, metadata=metadata)


class InvertedHammerStrategy(Strategy):
    """Long entry on Inverted Hammer pattern."""

    def init(self):
        self.market_open_time = dt_time(0, 0)
        self.market_close_time = dt_time(23, 59)

    def on_data(self, index, low, high, close, open, dates):
        if index < 1:
            return
        current_open = open[index]
        current_close = close[index]
        current_high = high[index]
        current_low = low[index]
        current_date = dates[index]

        if (CandleStickPatterns.is_inverted_hammer(current_open, current_close,
                                                    current_high, current_low)
                and not self.in_position
                and self.trading_hours(current_date)):
            metadata = {
                'current_open': current_open,
                'current_close': current_close,
                'current_high': current_high,
                'current_low': current_low,
            }
            entry_price = current_close
            if self.model is not None:
                if self.cs_patterns:
                    cs_features = inverted_hammer_features(**metadata).reshape(1, -1)
                    prediction = self.predict(current_date, cs_features)
                else:
                    prediction = self.predict(current_date)
                take_profit = entry_price + prediction
            else:
                take_profit = entry_price + DEFAULT_TP_DISTANCE
            stop_loss = entry_price - DEFAULT_SL_DISTANCE
            self.buy(price=entry_price, take_profit=take_profit,
                     stop_loss=stop_loss, entry_time=current_date, metadata=metadata)


class DragonflyDojiStrategy(Strategy):
    """Long entry on Dragonfly Doji pattern."""

    def init(self):
        self.market_open_time = dt_time(0, 0)
        self.market_close_time = dt_time(23, 59)

    def on_data(self, index, low, high, close, open, dates):
        if index < 1:
            return
        current_open = open[index]
        current_close = close[index]
        current_high = high[index]
        current_low = low[index]
        current_date = dates[index]

        if (CandleStickPatterns.is_dragonfly_doji(current_open, current_close,
                                                   current_high, current_low)
                and not self.in_position
                and self.trading_hours(current_date)):
            metadata = {
                'current_open': current_open,
                'current_close': current_close,
                'current_high': current_high,
                'current_low': current_low,
            }
            entry_price = current_close
            if self.model is not None:
                if self.cs_patterns:
                    cs_features = dragonfly_doji_features(**metadata).reshape(1, -1)
                    prediction = self.predict(current_date, cs_features)
                else:
                    prediction = self.predict(current_date)
                take_profit = entry_price + prediction
            else:
                take_profit = entry_price + DEFAULT_TP_DISTANCE
            stop_loss = entry_price - DEFAULT_SL_DISTANCE
            self.buy(price=entry_price, take_profit=take_profit,
                     stop_loss=stop_loss, entry_time=current_date, metadata=metadata)


# ── 2-bar pattern strategies ─────────────────────────────────────────────────

class BullishEngulfingStrategy(Strategy):
    """Long entry on Bullish Engulfing pattern."""

    def init(self):
        self.market_open_time = dt_time(0, 0)
        self.market_close_time = dt_time(23, 59)

    def on_data(self, index, low, high, close, open, dates):
        if index < 1:
            return
        current_open = open[index]
        current_close = close[index]
        current_high = high[index]
        current_low = low[index]
        current_date = dates[index]
        prev_open = open[index - 1]
        prev_close = close[index - 1]

        if (CandleStickPatterns.is_bullish_engulfing(current_open, current_close,
                                                      prev_open, prev_close)
                and not self.in_position
                and self.trading_hours(current_date)):
            metadata = {
                'current_open': current_open,
                'current_close': current_close,
                'prev_open': prev_open,
                'prev_close': prev_close,
            }
            entry_price = current_close
            if self.model is not None:
                if self.cs_patterns:
                    cs_features = engulfing_features(**metadata).reshape(1, -1)
                    prediction = self.predict(current_date, cs_features)
                else:
                    prediction = self.predict(current_date)
                take_profit = entry_price + prediction
            else:
                take_profit = entry_price + DEFAULT_TP_DISTANCE
            stop_loss = entry_price - DEFAULT_SL_DISTANCE
            self.buy(price=entry_price, take_profit=take_profit,
                     stop_loss=stop_loss, entry_time=current_date, metadata=metadata)


class BullishHaramiStrategy(Strategy):
    """Long entry on Bullish Harami pattern."""

    def init(self):
        self.market_open_time = dt_time(0, 0)
        self.market_close_time = dt_time(23, 59)

    def on_data(self, index, low, high, close, open, dates):
        if index < 1:
            return
        current_open = open[index]
        current_close = close[index]
        current_high = high[index]
        current_low = low[index]
        current_date = dates[index]
        prev_open = open[index - 1]
        prev_close = close[index - 1]

        if (CandleStickPatterns.is_bullish_harami(current_open, current_close,
                                                   prev_open, prev_close)
                and not self.in_position
                and self.trading_hours(current_date)):
            metadata = {
                'current_open': current_open,
                'current_close': current_close,
                'prev_open': prev_open,
                'prev_close': prev_close,
            }
            entry_price = current_close
            if self.model is not None:
                if self.cs_patterns:
                    cs_features = harami_features(**metadata).reshape(1, -1)
                    prediction = self.predict(current_date, cs_features)
                else:
                    prediction = self.predict(current_date)
                take_profit = entry_price + prediction
            else:
                take_profit = entry_price + DEFAULT_TP_DISTANCE
            stop_loss = entry_price - DEFAULT_SL_DISTANCE
            self.buy(price=entry_price, take_profit=take_profit,
                     stop_loss=stop_loss, entry_time=current_date, metadata=metadata)


class PiercingPatternStrategy(Strategy):
    """Long entry on Piercing Line pattern."""

    def init(self):
        self.market_open_time = dt_time(0, 0)
        self.market_close_time = dt_time(23, 59)

    def on_data(self, index, low, high, close, open, dates):
        if index < 1:
            return
        current_open = open[index]
        current_close = close[index]
        current_high = high[index]
        current_low = low[index]
        current_date = dates[index]
        prev_open = open[index - 1]
        prev_close = close[index - 1]

        if (CandleStickPatterns.is_piercing_pattern(current_open, current_close,
                                                     prev_open, prev_close)
                and not self.in_position
                and self.trading_hours(current_date)):
            metadata = {
                'current_open': current_open,
                'current_close': current_close,
                'prev_open': prev_open,
                'prev_close': prev_close,
            }
            entry_price = current_close
            if self.model is not None:
                if self.cs_patterns:
                    cs_features = piercing_pattern_features(**metadata).reshape(1, -1)
                    prediction = self.predict(current_date, cs_features)
                else:
                    prediction = self.predict(current_date)
                take_profit = entry_price + prediction
            else:
                take_profit = entry_price + DEFAULT_TP_DISTANCE
            stop_loss = entry_price - DEFAULT_SL_DISTANCE
            self.buy(price=entry_price, take_profit=take_profit,
                     stop_loss=stop_loss, entry_time=current_date, metadata=metadata)


# ── 3-bar pattern strategies ─────────────────────────────────────────────────

class MorningStarStrategy(Strategy):
    """Long entry on Morning Star pattern."""

    def init(self):
        self.market_open_time = dt_time(0, 0)
        self.market_close_time = dt_time(23, 59)

    def on_data(self, index, low, high, close, open, dates):
        if index < 2:
            return
        current_open = open[index]
        current_close = close[index]
        current_high = high[index]
        current_low = low[index]
        current_date = dates[index]
        prev_open = open[index - 1]
        prev_close = close[index - 1]
        b_prev_open = open[index - 2]
        b_prev_close = close[index - 2]

        if (CandleStickPatterns.is_morning_star(b_prev_open, b_prev_close,
                                                 prev_open, prev_close,
                                                 current_open, current_close)
                and not self.in_position
                and self.trading_hours(current_date)):
            metadata = {
                'b_prev_open': b_prev_open,
                'b_prev_close': b_prev_close,
                'prev_open': prev_open,
                'prev_close': prev_close,
                'current_open': current_open,
                'current_close': current_close,
            }
            entry_price = current_close
            if self.model is not None:
                if self.cs_patterns:
                    cs_features = morning_star_features(**metadata).reshape(1, -1)
                    prediction = self.predict(current_date, cs_features)
                else:
                    prediction = self.predict(current_date)
                take_profit = entry_price + prediction
            else:
                take_profit = entry_price + DEFAULT_TP_DISTANCE
            stop_loss = entry_price - DEFAULT_SL_DISTANCE
            self.buy(price=entry_price, take_profit=take_profit,
                     stop_loss=stop_loss, entry_time=current_date, metadata=metadata)


class MorningStarDojiStrategy(Strategy):
    """Long entry on Morning Doji Star pattern."""

    def init(self):
        self.market_open_time = dt_time(0, 0)
        self.market_close_time = dt_time(23, 59)

    def on_data(self, index, low, high, close, open, dates):
        if index < 2:
            return
        current_open = open[index]
        current_close = close[index]
        current_high = high[index]
        current_low = low[index]
        current_date = dates[index]
        prev_open = open[index - 1]
        prev_close = close[index - 1]
        prev_high = high[index - 1]
        prev_low = low[index - 1]
        b_prev_open = open[index - 2]
        b_prev_close = close[index - 2]

        if (CandleStickPatterns.is_morning_star_doji(b_prev_open, b_prev_close,
                                                      prev_open, prev_close,
                                                      prev_high, prev_low,
                                                      current_open, current_close)
                and not self.in_position
                and self.trading_hours(current_date)):
            metadata = {
                'b_prev_open': b_prev_open,
                'b_prev_close': b_prev_close,
                'prev_high': prev_high,
                'prev_low': prev_low,
                'current_open': current_open,
                'current_close': current_close,
            }
            entry_price = current_close
            if self.model is not None:
                if self.cs_patterns:
                    cs_features = morning_star_doji_features(**metadata).reshape(1, -1)
                    prediction = self.predict(current_date, cs_features)
                else:
                    prediction = self.predict(current_date)
                take_profit = entry_price + prediction
            else:
                take_profit = entry_price + DEFAULT_TP_DISTANCE
            stop_loss = entry_price - DEFAULT_SL_DISTANCE
            self.buy(price=entry_price, take_profit=take_profit,
                     stop_loss=stop_loss, entry_time=current_date, metadata=metadata)


# ── Strategy registry ────────────────────────────────────────────────────────

PATTERN_STRATEGIES = {
    'Hammer':              HammerStrategy,
    'Inverted Hammer':     InvertedHammerStrategy,
    'Dragonfly Doji':      DragonflyDojiStrategy,
    'Bullish Engulfing':   BullishEngulfingStrategy,
    'Bullish Harami':      BullishHaramiStrategy,
    'Piercing Line':       PiercingPatternStrategy,
    'Morning Star':        MorningStarStrategy,
    'Morning Doji Star':   MorningStarDojiStrategy,
}
