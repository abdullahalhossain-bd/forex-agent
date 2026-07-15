"""
strategies/ema_rsi_combo.py — EMA-200 + RSI-50 Combo Strategy (Day 81+)
=========================================================================

Masterclass concept:
  - Price ABOVE EMA-200 = bullish bias (only take BUY trades)
  - Price BELOW EMA-200 = bearish bias (only take SELL trades)
  - RSI crosses above 50 = bullish momentum confirmation
  - RSI crosses below 50 = bearish momentum confirmation

Entry rules:
  BUY:  price > EMA-200  AND  RSI crosses above 50  AND  RSI < 70
  SELL: price < EMA-200  AND  RSI crosses below 50  AND  RSI > 30

This is a trend-following filter — it only trades in the direction
of the higher-timeframe trend (EMA-200) and enters on momentum
shifts (RSI crossing the 50 midline).

Usage:
    from strategies.ema_rsi_combo import EmaRsiComboStrategy

    strategy = EmaRsiComboStrategy()
    signal = strategy.analyze("EURUSD", "15m", df)

    if signal.is_tradeable:
        # send to risk engine → execution
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from core.unified_signal import UnifiedSignal
from utils.logger import get_logger

log = get_logger("ema_rsi_combo")


# ── Constants ──────────────────────────────────────────────────

EMA_PERIOD = 200
RSI_PERIOD = 14
RSI_MIDLINE = 50
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# SL/TP based on ATR
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0   # R:R = 1:2
MIN_LOT = 0.01


class EmaRsiComboStrategy:
    """
    EMA-200 trend filter + RSI-50 momentum trigger.

    Returns a UnifiedSignal so it integrates with the existing
    decision pipeline.
    """

    name = "ema_rsi_combo"
    timeframe_scope = ("M5", "M15", "M30", "H1")

    def analyze(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        session: str = "",
    ) -> UnifiedSignal:
        """Run the EMA-200 + RSI-50 combo check."""
        if timeframe.upper() not in self.timeframe_scope:
            return UnifiedSignal.wait(
                symbol, timeframe,
                reason=f"{timeframe} not in scope {self.timeframe_scope}"
            )

        if df is None or len(df) < EMA_PERIOD + 10:
            return UnifiedSignal.wait(
                symbol, timeframe,
                reason=f"Need at least {EMA_PERIOD + 10} candles (got {len(df) if df is not None else 0})"
            )

        # ── Compute indicators ────────────────────────────────────
        df = df.copy()
        df["ema_200"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # ATR for SL/TP
        df["tr"] = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = df["tr"].rolling(14).mean()

        # ── Get latest values ─────────────────────────────────────
        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last["close"])
        ema_200 = float(last["ema_200"])
        rsi = float(last["rsi"])
        prev_rsi = float(prev["rsi"])
        atr = float(last["atr"])

        if pd.isna(ema_200) or pd.isna(rsi) or pd.isna(atr):
            return UnifiedSignal.wait(symbol, timeframe, reason="Indicator NaN")

        # ── Determine bias ────────────────────────────────────────
        above_ema = close > ema_200
        below_ema = close < ema_200

        # RSI cross detection
        rsi_crossed_up = prev_rsi < RSI_MIDLINE and rsi >= RSI_MIDLINE
        rsi_crossed_down = prev_rsi > RSI_MIDLINE and rsi <= RSI_MIDLINE

        # ── Entry logic ───────────────────────────────────────────
        reasons = []
        direction = None

        if above_ema and rsi_crossed_up and rsi < RSI_OVERBOUGHT:
            direction = "BUY"
            reasons.append(f"Price {close:.5f} > EMA-200 {ema_200:.5f}")
            reasons.append(f"RSI crossed UP 50 ({prev_rsi:.0f} → {rsi:.0f})")
            reasons.append(f"RSI not overbought ({rsi:.0f} < {RSI_OVERBOUGHT})")

        elif below_ema and rsi_crossed_down and rsi > RSI_OVERSOLD:
            direction = "SELL"
            reasons.append(f"Price {close:.5f} < EMA-200 {ema_200:.5f}")
            reasons.append(f"RSI crossed DOWN 50 ({prev_rsi:.0f} → {rsi:.0f})")
            reasons.append(f"RSI not oversold ({rsi:.0f} > {RSI_OVERSOLD})")

        if direction is None:
            # Check if we're in a trend but no trigger yet
            if above_ema and rsi > RSI_MIDLINE:
                return UnifiedSignal.wait(
                    symbol, timeframe,
                    reason=f"Above EMA-200 + RSI {rsi:.0f}>50 but no cross (waiting for trigger)"
                )
            elif below_ema and rsi < RSI_MIDLINE:
                return UnifiedSignal.wait(
                    symbol, timeframe,
                    reason=f"Below EMA-200 + RSI {rsi:.0f}<50 but no cross (waiting for trigger)"
                )
            else:
                return UnifiedSignal.wait(
                    symbol, timeframe,
                    reason=f"No setup: price={'above' if above_ema else 'below'} EMA-200, RSI={rsi:.0f}"
                )

        # ── Build trade signal ────────────────────────────────────
        if direction == "BUY":
            sl = close - (atr * SL_ATR_MULT)
            tp = [close + (atr * TP_ATR_MULT)]
            confidence = min(85, 50 + (rsi - 50) * 2)  # higher RSI = more confidence
        else:
            sl = close + (atr * SL_ATR_MULT)
            tp = [close - (atr * TP_ATR_MULT)]
            confidence = min(85, 50 + (50 - rsi) * 2)

        return UnifiedSignal(
            pair=symbol,
            timeframe=timeframe,
            signal=direction,
            confidence=confidence,
            entry=close,
            sl=sl,
            tp=tp,
            lot=MIN_LOT,
            risk_percent=0.5,
            source_agents=["ema_rsi_combo"],
            agent_votes={"ema_rsi_combo": confidence},
            reasons=reasons,
            market_story=f"EMA-200 {'above' if above_ema else 'below'} + RSI {'crossed up' if rsi_crossed_up else 'crossed down'} 50",
            market_bias="BULLISH" if direction == "BUY" else "BEARISH",
            regime="TRENDING",
            session=session or None,
            metadata={
                "strategy": "ema_rsi_combo",
                "ema_200": ema_200,
                "rsi": rsi,
                "prev_rsi": prev_rsi,
                "atr": atr,
                "above_ema": above_ema,
                "rsi_crossed_up": rsi_crossed_up,
                "rsi_crossed_down": rsi_crossed_down,
            },
        )
