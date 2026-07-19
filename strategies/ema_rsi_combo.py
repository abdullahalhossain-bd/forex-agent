"""
strategies/ema_rsi_combo.py — EMA-200 + RSI-50 Combo Strategy (institutional rewrite)
=======================================================================================

Concept (unchanged from v1):
  - Price ABOVE EMA-200 = bullish bias (only take BUY trades)
  - Price BELOW EMA-200 = bearish bias (only take SELL trades)
  - RSI crosses above 50 = bullish momentum confirmation
  - RSI crosses below 50 = bearish momentum confirmation

CONTRACT (read before wiring this into a live loop):
- `df` MUST contain only fully CLOSED bars; `df.iloc[-1]` is treated as the
  most recent CLOSED bar. If the currently-forming bar is included as the
  last row, every RSI-cross detection is subject to repainting: the cross
  can appear to happen, then "un-happen" as the forming bar's close moves,
  which is exactly the failure mode described in bias-and-validation.md.
- EMA-200 is computed via `ewm(adjust=False)`, which is a full recursive
  average seeded from the FIRST row of whatever `df` you pass in. If the
  window of history passed to `analyze()` changes length/start point
  between calls (e.g. a rolling buffer of only the last N bars), the
  EMA-200 value will drift depending on where that window starts — it is
  NOT equivalent to a true 200-bar-anchored EMA unless a long, consistent
  lookback is always supplied. Recommend passing >= ~2000 bars (10x the EMA
  period) so the initialization transient is negligible, or persisting EMA
  state statefully across calls instead of recomputing from a variable
  window each time.
- Returns a UnifiedSignal. Position sizing (`lot`) is NOT computed here —
  see MIN_LOT note below. A real risk engine must size the position from
  account equity, risk %, and the SL distance; do not treat the `lot` field
  on this signal as authoritative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.unified_signal import UnifiedSignal
from utils.logger import get_logger

log = get_logger("ema_rsi_combo")


class MarketDataError(Exception):
    """Raised when required OHLC columns are missing/malformed. This is a
    data-pipeline failure and must propagate loudly, not be swallowed into
    a benign WAIT signal that looks like "no setup today"."""


# ── Constants ──────────────────────────────────────────────────

EMA_PERIOD = 200
RSI_PERIOD = 14
RSI_MIDLINE = 50.0
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0
ATR_PERIOD = 14

# Extra bars beyond EMA_PERIOD before we trust the indicator stack:
# RSI_PERIOD/ATR_PERIOD warm up far faster than EMA_PERIOD, so the binding
# constraint is EMA-200; +10 gives one extra bar of margin so `prev` (iloc[-2])
# is also drawn from a fully warmed-up row, not the first valid one.
WARMUP_BUFFER = 10

SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0  # R:R = 1:2

# NOT an authoritative position size. Many UnifiedSignal consumers require
# a non-null `lot`, so this is a schema-compatibility placeholder only —
# a downstream risk engine must override it using account equity, risk_percent,
# and the actual SL distance (see risk-engine reference: never trade a fixed
# lot size in a system that claims to do risk management).
PLACEHOLDER_LOT = 0.01


@dataclass(frozen=True)
class ComboConfig:
    """Validated strategy configuration."""

    sl_atr_mult: float = SL_ATR_MULT
    tp_atr_mult: float = TP_ATR_MULT
    rsi_overbought: float = RSI_OVERBOUGHT
    rsi_oversold: float = RSI_OVERSOLD
    # Bars required between two signals in the same symbol/timeframe to stop
    # RSI chop around the midline from firing repeatedly inside one trend leg.
    min_bars_between_signals: int = 5
    # Optional lightweight trend-strength gate: minimum EMA-200 slope over
    # the last `ema_slope_lookback` bars, expressed as a fraction of price
    # (e.g. 0.0005 = 0.05%). Default 0 = disabled (matches v1 behavior).
    # This exists because, unlike the breakout strategy, this module has no
    # ADX filter, so it will happily fire on RSI/EMA chop in a flat market
    # unless something measures trend strength.
    min_ema_slope_pct: float = 0.0
    ema_slope_lookback: int = 10

    def __post_init__(self) -> None:
        if self.sl_atr_mult <= 0:
            raise ValueError("sl_atr_mult must be > 0")
        if self.tp_atr_mult <= 0:
            raise ValueError("tp_atr_mult must be > 0")
        if not (0 < self.rsi_oversold < RSI_MIDLINE < self.rsi_overbought < 100):
            raise ValueError("RSI thresholds must satisfy 0 < oversold < 50 < overbought < 100")
        if self.min_bars_between_signals < 0:
            raise ValueError("min_bars_between_signals must be >= 0")
        if self.min_ema_slope_pct < 0:
            raise ValueError("min_ema_slope_pct must be >= 0")
        if self.ema_slope_lookback < 1:
            raise ValueError("ema_slope_lookback must be >= 1")


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Wilder's RSI (RMA-smoothed), matching the conventional/platform-standard
    definition (e.g. TradingView, MT5 default RSI).

    v1 used `delta.rolling(period).mean()` for average gain/loss — a SIMPLE
    moving average, not Wilder's recursive smoothing. This is a real
    mathematical mismatch, not just a style choice: with a plain rolling
    mean, a large historical gain/loss aging OUT of the window can flip the
    RSI value on a bar with no new price action to justify it (a "phantom"
    move caused purely by window mechanics). Wilder smoothing decays old
    observations gradually instead of dropping them off a cliff, and is
    what "RSI" conventionally refers to in institutional and retail
    platforms alike — using a different formula silently means this
    strategy's "RSI crosses 50" events won't line up with what the RSI
    value on a standard chart would show.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _true_range(df: pd.DataFrame) -> pd.Series:
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)


class EmaRsiComboStrategy:
    """
    EMA-200 trend filter + RSI-50 momentum trigger.

    Known gaps vs. an institutional trend-momentum system (documented, not
    silently assumed solved):
      - No liquidity-sweep / stop-hunt filter, no HTF bias confirmation.
      - No session/kill-zone gating despite carrying `session` through.
      - No spread/news-proximity gate — must be enforced by the caller.
      - No ADX-equivalent trend-strength filter by default; `min_ema_slope_pct`
        is an optional, off-by-default proxy — validate before relying on it.
      - No portfolio-level exposure/correlation check (belongs in risk engine).
    """

    name = "ema_rsi_combo"
    timeframe_scope = ("M5", "M15", "M30", "H1")

    def __init__(self, config: Optional[ComboConfig] = None) -> None:
        self.config = config or ComboConfig()

    def analyze(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        session: str = "",
        bars_since_last_signal: Optional[int] = None,
    ) -> UnifiedSignal:
        """Run the EMA-200 + RSI-50 combo check on the most recent closed bar.

        Args:
            bars_since_last_signal: bars elapsed since this strategy last
                fired on this symbol/timeframe, supplied by the caller (this
                module stays stateless/pure for testability). Enforces
                `min_bars_between_signals`. Pass None to disable the check.

        Raises:
            MarketDataError: if required OHLC columns are missing or the
                indicator stack can't be computed — propagated, not muted
                into a WAIT, since that indicates a broken data pipeline.
        """
        if timeframe.upper() not in self.timeframe_scope:
            return UnifiedSignal.wait(
                symbol, timeframe, reason=f"{timeframe} not in scope {self.timeframe_scope}"
            )

        min_bars = EMA_PERIOD + WARMUP_BUFFER
        if df is None or len(df) < min_bars:
            return UnifiedSignal.wait(
                symbol, timeframe,
                reason=f"Need at least {min_bars} candles (got {len(df) if df is not None else 0})",
            )

        try:
            df = self._compute_indicators(df)
        except KeyError as exc:
            raise MarketDataError(f"Missing required OHLC column: {exc}") from exc

        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last["close"])
        ema_200 = float(last["ema_200"])
        rsi = float(last["rsi"])
        prev_rsi = float(prev["rsi"])
        atr = float(last["atr"])

        if any(math.isnan(v) for v in (ema_200, rsi, prev_rsi, atr)):
            return UnifiedSignal.wait(symbol, timeframe, reason="Indicator NaN")

        if atr <= 0:
            # v1 had no guard here at all: a zero ATR silently produced
            # sl == entry (zero-distance stop), which most brokers reject
            # outright and which is meaningless risk-wise even if accepted.
            return UnifiedSignal.wait(symbol, timeframe, reason="ATR is zero or negative")

        if (
            bars_since_last_signal is not None
            and bars_since_last_signal < self.config.min_bars_between_signals
        ):
            return UnifiedSignal.wait(
                symbol, timeframe,
                reason=(
                    f"cooldown active: {bars_since_last_signal} < "
                    f"{self.config.min_bars_between_signals} bars"
                ),
            )

        if self.config.min_ema_slope_pct > 0:
            slope_ok, slope_pct = self._ema_slope_ok(df)
            if not slope_ok:
                return UnifiedSignal.wait(
                    symbol, timeframe,
                    reason=f"EMA-200 slope too flat ({slope_pct:.4%} < {self.config.min_ema_slope_pct:.4%})",
                )

        above_ema = close > ema_200
        below_ema = close < ema_200
        rsi_crossed_up = prev_rsi < RSI_MIDLINE <= rsi
        rsi_crossed_down = prev_rsi > RSI_MIDLINE >= rsi

        direction, reasons = self._decide_direction(
            above_ema, below_ema, rsi_crossed_up, rsi_crossed_down, close, ema_200, rsi, prev_rsi
        )

        if direction is None:
            if above_ema and rsi > RSI_MIDLINE:
                return UnifiedSignal.wait(
                    symbol, timeframe,
                    reason=f"Above EMA-200 + RSI {rsi:.0f}>50 but no cross (waiting for trigger)",
                )
            if below_ema and rsi < RSI_MIDLINE:
                return UnifiedSignal.wait(
                    symbol, timeframe,
                    reason=f"Below EMA-200 + RSI {rsi:.0f}<50 but no cross (waiting for trigger)",
                )
            return UnifiedSignal.wait(
                symbol, timeframe,
                reason=f"No setup: price={'above' if above_ema else 'below'} EMA-200, RSI={rsi:.0f}",
            )

        return self._build_signal(
            symbol, timeframe, session, direction, close, atr, rsi, prev_rsi,
            ema_200, above_ema, rsi_crossed_up, rsi_crossed_down, reasons,
        )

    # ── Helpers (split out for unit testing without UnifiedSignal wiring) ──

    @staticmethod
    def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low", "close"):
            if col not in df.columns:
                raise KeyError(col)
        df = df.copy()
        df["ema_200"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
        df["rsi"] = _wilder_rsi(df["close"], RSI_PERIOD)
        df["tr"] = _true_range(df)
        df["atr"] = df["tr"].rolling(ATR_PERIOD).mean()
        return df

    def _ema_slope_ok(self, df: pd.DataFrame) -> tuple[bool, float]:
        lookback = self.config.ema_slope_lookback
        if len(df) <= lookback:
            return False, 0.0
        recent = float(df["ema_200"].iloc[-1])
        past = float(df["ema_200"].iloc[-1 - lookback])
        if past == 0 or math.isnan(recent) or math.isnan(past):
            return False, 0.0
        slope_pct = abs(recent - past) / past
        return slope_pct >= self.config.min_ema_slope_pct, slope_pct

    @staticmethod
    def _decide_direction(
        above_ema: bool, below_ema: bool, rsi_crossed_up: bool, rsi_crossed_down: bool,
        close: float, ema_200: float, rsi: float, prev_rsi: float,
    ) -> tuple[Optional[str], list[str]]:
        reasons: list[str] = []
        if above_ema and rsi_crossed_up and rsi < RSI_OVERBOUGHT:
            reasons = [
                f"Price {close:.5f} > EMA-200 {ema_200:.5f}",
                f"RSI crossed UP 50 ({prev_rsi:.0f} -> {rsi:.0f})",
                f"RSI not overbought ({rsi:.0f} < {RSI_OVERBOUGHT:.0f})",
            ]
            return "BUY", reasons
        if below_ema and rsi_crossed_down and rsi > RSI_OVERSOLD:
            reasons = [
                f"Price {close:.5f} < EMA-200 {ema_200:.5f}",
                f"RSI crossed DOWN 50 ({prev_rsi:.0f} -> {rsi:.0f})",
                f"RSI not oversold ({rsi:.0f} > {RSI_OVERSOLD:.0f})",
            ]
            return "SELL", reasons
        return None, reasons

    def _build_signal(
        self, symbol, timeframe, session, direction, close, atr, rsi, prev_rsi,
        ema_200, above_ema, rsi_crossed_up, rsi_crossed_down, reasons,
    ) -> UnifiedSignal:
        cfg = self.config
        if direction == "BUY":
            sl = close - (atr * cfg.sl_atr_mult)
            tp = [close + (atr * cfg.tp_atr_mult)]
            # Confidence is a bounded HEURISTIC score, not a calibrated
            # probability of winning — do not feed into position sizing as
            # if it were a win-rate estimate.
            confidence = min(85.0, 50.0 + (rsi - RSI_MIDLINE) * 2.0)
        else:
            sl = close + (atr * cfg.sl_atr_mult)
            tp = [close - (atr * cfg.tp_atr_mult)]
            confidence = min(85.0, 50.0 + (RSI_MIDLINE - rsi) * 2.0)
        confidence = round(max(0.0, confidence), 1)

        log.info(
            "%s %s signal: entry=%.5f sl=%.5f tp=%.5f confidence=%.1f",
            symbol, direction, close, sl, tp[0], confidence,
        )

        return UnifiedSignal(
            pair=symbol,
            timeframe=timeframe,
            signal=direction,
            confidence=confidence,
            entry=close,
            sl=sl,
            tp=tp,
            lot=PLACEHOLDER_LOT,  # NOT authoritative — see module docstring.
            risk_percent=0.5,
            source_agents=["ema_rsi_combo"],
            agent_votes={"ema_rsi_combo": confidence},
            reasons=reasons,
            market_story=(
                f"EMA-200 {'above' if above_ema else 'below'} + "
                f"RSI {'crossed up' if rsi_crossed_up else 'crossed down'} 50"
            ),
            market_bias="BULLISH" if direction == "BUY" else "BEARISH",
            regime="TRENDING",
            session=session or None,
            metadata={
                "strategy": "ema_rsi_combo",
                "strategy_version": "v2",
                "ema_200": ema_200,
                "rsi": rsi,
                "prev_rsi": prev_rsi,
                "atr": atr,
                "above_ema": above_ema,
                "rsi_crossed_up": rsi_crossed_up,
                "rsi_crossed_down": rsi_crossed_down,
                "confidence_is_heuristic_not_calibrated_probability": True,
                "lot_is_placeholder_not_risk_sized": True,
            },
        )