# strategy/multi_strategy_set.py — Multi-strategy set system (Set1–Set11 pattern)
# =============================================================================
# Inspired by: https://github.com/smartedgetrading/SmartEdge-EA
# SmartEdge changelog v1.1.0: "Introduced multiple strategy sets (Set1–Set11)
# with independent indicator configs per strategy"
#
# A registry system for running multiple strategy "sets" in parallel, each
# with its own independent indicator configuration. This allows:
#
#   - Running RSI(14) + EMA(50) on one set, and RSI(21) + EMA(200) on another
#   - Each set produces its own signal, and the combiner aggregates them
#   - A "vote" or "weighted average" or "any/all" combiner decides the final
#     action
#
# This is the architecture SmartEdge uses internally — the "Master EA" runs
# multiple strategy sets and combines their signals before executing.
#
# Usage
# -----
#     registry = StrategySetRegistry()
#
#     # Define Set1: RSI(14) oversold + EMA(50) uptrend
#     registry.register("Set1", StrategySet(
#         name="RSI+EMA Trend Follow",
#         indicators={
#             "rsi": {"type": "rsi", "period": 14, "oversold": 30, "overbought": 70},
#             "ema": {"type": "ema", "period": 50},
#         },
#         entry_rule="rsi < oversold AND close > ema",  # BUY signal
#         weight=1.0,
#     ))
#
#     # Define Set2: MACD crossover
#     registry.register("Set2", StrategySet(
#         name="MACD Cross",
#         indicators={"macd": {"type": "macd", "fast": 12, "slow": 26, "signal": 9}},
#         entry_rule="macd_cross_above",  # BUY signal
#         weight=0.8,
#     ))
#
#     # Evaluate all sets on the current data
#     signals = registry.evaluate_all(df)
#     # → {"Set1": +1, "Set2": 0, ...}
#
#     # Combine signals
#     final = registry.combine(signals, method="weighted_vote")
#     # → +1 (BUY), -1 (SELL), or 0 (HOLD)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger("multi_strategy_set")


@dataclass
class StrategySet:
    """
    One strategy set: a named collection of indicators + an entry rule +
    a weight for signal combination.

    Attributes
    ----------
    name : human-readable name.
    indicators : dict of indicator configs. Each config has a "type" key
        (rsi, ema, macd, etc.) plus type-specific parameters.
    entry_rule : a string or callable that defines the BUY condition.
        If string: evaluated against the computed indicator columns
        (e.g., "rsi < 30 AND close > ema"). Uses pandas eval.
        If callable: called as `entry_rule(df) -> pd.Series[bool]`.
    exit_rule : optional, same shape as entry_rule but for SELL signals.
    weight : float weight for signal combination (default 1.0).
    enabled : if False, this set is skipped during evaluation.
    """
    name: str
    indicators: dict[str, dict] = field(default_factory=dict)
    entry_rule: str | Callable | None = None
    exit_rule: str | Callable | None = None
    weight: float = 1.0
    enabled: bool = True


# ── Built-in indicator calculators ───────────────────────────────────────────

def _calc_rsi(df: pd.DataFrame, period: int = 14,
              oversold: float = 30, overbought: float = 70,
              close_col: str = "close") -> pd.DataFrame:
    """Compute RSI and add `rsi`, `rsi_oversold`, `rsi_overbought` columns."""
    out = df.copy()
    delta = out[close_col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    out["rsi"] = 100 - (100 / (1 + rs))
    out["rsi_oversold"] = oversold
    out["rsi_overbought"] = overbought
    return out


def _calc_ema(df: pd.DataFrame, period: int = 50,
              close_col: str = "close") -> pd.DataFrame:
    """Compute EMA and add `ema` column."""
    out = df.copy()
    out["ema"] = out[close_col].ewm(span=period, adjust=False).mean()
    return out


def _calc_sma(df: pd.DataFrame, period: int = 200,
              close_col: str = "close") -> pd.DataFrame:
    """Compute SMA and add `sma` column."""
    out = df.copy()
    out["sma"] = out[close_col].rolling(period).mean()
    return out


def _calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
               signal: int = 9, close_col: str = "close") -> pd.DataFrame:
    """Compute MACD and add `macd`, `macd_signal`, `macd_hist`, `macd_cross` columns."""
    out = df.copy()
    ema_fast = out[close_col].ewm(span=fast, adjust=False).mean()
    ema_slow = out[close_col].ewm(span=slow, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=signal, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    # Cross signals: +1 when macd crosses above signal, -1 when below
    prev_diff = (out["macd"] - out["macd_signal"]).shift(1)
    curr_diff = out["macd"] - out["macd_signal"]
    out["macd_cross"] = 0
    out.loc[(prev_diff <= 0) & (curr_diff > 0), "macd_cross"] = 1
    out.loc[(prev_diff >= 0) & (curr_diff < 0), "macd_cross"] = -1
    return out


INDICATOR_CALCULATORS = {
    "rsi": _calc_rsi,
    "ema": _calc_ema,
    "sma": _calc_sma,
    "macd": _calc_macd,
}


# ── Registry ─────────────────────────────────────────────────────────────────

class StrategySetRegistry:
    """
    Registry of multiple strategy sets. Evaluate all sets on the same data,
    then combine their signals.
    """

    def __init__(self):
        self._sets: dict[str, StrategySet] = {}

    def register(self, key: str, strategy_set: StrategySet) -> None:
        """Register a strategy set under `key` (e.g., 'Set1')."""
        self._sets[key] = strategy_set
        log.info(f"Registered strategy set '{key}': {strategy_set.name}")

    def unregister(self, key: str) -> None:
        """Remove a strategy set."""
        if key in self._sets:
            del self._sets[key]
            log.info(f"Unregistered strategy set '{key}'")

    def get(self, key: str) -> Optional[StrategySet]:
        return self._sets.get(key)

    def list_sets(self) -> list[str]:
        return list(self._sets.keys())

    def enable(self, key: str) -> None:
        if key in self._sets:
            self._sets[key].enabled = True

    def disable(self, key: str) -> None:
        if key in self._sets:
            self._sets[key].enabled = False

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame, indicators: dict) -> pd.DataFrame:
        """Compute all indicators for a strategy set."""
        out = df.copy()
        for name, config in indicators.items():
            ind_type = config.get("type")
            if ind_type not in INDICATOR_CALCULATORS:
                log.warning(f"Unknown indicator type: {ind_type}")
                continue
            calc = INDICATOR_CALCULATORS[ind_type]
            params = {k: v for k, v in config.items() if k != "type"}
            out = calc(out, **params)
        return out

    def _eval_rule(self, df: pd.DataFrame, rule) -> pd.Series:
        """Evaluate an entry/exit rule. Returns a boolean Series."""
        if rule is None:
            return pd.Series(False, index=df.index)
        if callable(rule):
            return rule(df)
        # String rule — use pandas eval
        try:
            # Replace 'AND'/'OR' with '&'/'|' for pandas eval
            expr = rule.replace(" AND ", " & ").replace(" OR ", " | ")
            return df.eval(expr).astype(bool)
        except Exception as e:
            log.warning(f"Rule eval failed: {e}")
            return pd.Series(False, index=df.index)

    def evaluate(self, key: str, df: pd.DataFrame) -> dict:
        """
        Evaluate a single strategy set on `df`.
        Returns {entry_signal: bool, exit_signal: bool, data: df_with_indicators}.
        """
        s = self._sets.get(key)
        if s is None or not s.enabled:
            return {"entry_signal": False, "exit_signal": False, "data": df}

        data = self._compute_indicators(df, s.indicators)
        entry = self._eval_rule(data, s.entry_rule)
        exit_sig = self._eval_rule(data, s.exit_rule)

        # Take the LAST bar's signal
        last_entry = bool(entry.iloc[-1]) if len(entry) > 0 else False
        last_exit = bool(exit_sig.iloc[-1]) if len(exit_sig) > 0 else False

        return {
            "entry_signal": last_entry,
            "exit_signal": last_exit,
            "data": data,
            "weight": s.weight,
        }

    def evaluate_all(self, df: pd.DataFrame) -> dict[str, int]:
        """
        Evaluate all registered strategy sets on `df`.
        Returns {set_key: signal} where signal is:
            +1 = entry (BUY)
            -1 = exit (SELL)
             0 = no signal
        """
        results = {}
        for key, s in self._sets.items():
            if not s.enabled:
                results[key] = 0
                continue
            eval_result = self.evaluate(key, df)
            if eval_result["entry_signal"]:
                results[key] = 1
            elif eval_result["exit_signal"]:
                results[key] = -1
            else:
                results[key] = 0
        return results

    # ── Signal combination ───────────────────────────────────────────────────

    @staticmethod
    def combine(signals: dict[str, int], method: str = "majority_vote",
                weights: Optional[dict[str, float]] = None) -> int:
        """
        Combine signals from multiple strategy sets into a single action.

        Parameters
        ----------
        signals : {set_key: +1/-1/0} dict from evaluate_all().
        method : one of:
            - "majority_vote": +1 if more buys than sells, -1 if more sells,
              0 if tie or all zero.
            - "weighted_vote": same but weighted by each set's weight.
            - "unanimous_buy": +1 only if ALL non-zero signals are +1.
            - "unanimous_sell": -1 only if ALL non-zero signals are -1.
            - "any": +1 if any +1, -1 if any -1 (most aggressive).
            - "all": +1 only if all are +1, -1 only if all are -1 (strictest).
        weights : optional {set_key: weight} dict for weighted_vote.

        Returns +1, -1, or 0.
        """
        if not signals:
            return 0

        vals = list(signals.values())
        buys = sum(1 for v in vals if v > 0)
        sells = sum(1 for v in vals if v < 0)

        if method == "majority_vote":
            if buys > sells:
                return 1
            elif sells > buys:
                return -1
            return 0

        elif method == "weighted_vote":
            total_weight = 0.0
            weighted_sum = 0.0
            for key, sig in signals.items():
                w = (weights or {}).get(key, 1.0)
                if sig != 0:
                    total_weight += w
                    weighted_sum += sig * w
            if total_weight == 0:
                return 0
            avg = weighted_sum / total_weight
            if avg > 0.3:
                return 1
            elif avg < -0.3:
                return -1
            return 0

        elif method == "unanimous_buy":
            non_zero = [v for v in vals if v != 0]
            return 1 if non_zero and all(v > 0 for v in non_zero) else 0

        elif method == "unanimous_sell":
            non_zero = [v for v in vals if v != 0]
            return -1 if non_zero and all(v < 0 for v in non_zero) else 0

        elif method == "any":
            if buys > 0:
                return 1
            elif sells > 0:
                return -1
            return 0

        elif method == "all":
            if all(v > 0 for v in vals):
                return 1
            elif all(v < 0 for v in vals):
                return -1
            return 0

        else:
            raise ValueError(f"Unknown combination method: {method}")


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    # Build synthetic data
    n = 200
    rng = np.random.default_rng(42)
    close = 1.0850 + np.cumsum(rng.normal(0, 0.0005, n))
    df = pd.DataFrame({
        "open": close, "high": close + 0.0003, "low": close - 0.0003,
        "close": close,
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"))

    registry = StrategySetRegistry()

    # Set1: RSI oversold + price above EMA
    registry.register("Set1", StrategySet(
        name="RSI+EMA Trend Follow",
        indicators={
            "rsi": {"type": "rsi", "period": 14, "oversold": 30, "overbought": 70},
            "ema": {"type": "ema", "period": 50},
        },
        entry_rule="rsi < rsi_oversold AND close > ema",
        weight=1.0,
    ))

    # Set2: MACD bullish cross
    registry.register("Set2", StrategySet(
        name="MACD Bullish Cross",
        indicators={"macd": {"type": "macd", "fast": 12, "slow": 26, "signal": 9}},
        entry_rule="macd_cross == 1",
        weight=0.8,
    ))

    # Set3: Price above SMA (simple trend)
    registry.register("Set3", StrategySet(
        name="SMA Trend",
        indicators={"sma": {"type": "sma", "period": 50}},
        entry_rule="close > sma",
        weight=0.5,
    ))

    # Evaluate all
    signals = registry.evaluate_all(df)
    print(f"Signals: {signals}")

    # Combine with different methods
    for method in ["majority_vote", "weighted_vote", "any", "all",
                    "unanimous_buy"]:
        result = StrategySetRegistry.combine(signals, method=method)
        print(f"  {method:20s} → {result:+d}")

    # Test enable/disable
    registry.disable("Set3")
    signals2 = registry.evaluate_all(df)
    print(f"\nAfter disabling Set3: {signals2}")

    # Test custom callable rule
    registry.register("Set4", StrategySet(
        name="Custom Callable",
        indicators={"rsi": {"type": "rsi", "period": 14}},
        entry_rule=lambda df: (df["rsi"] < 25) & (df["close"] > df["close"].shift(1)),
        weight=1.0,
    ))
    result = registry.evaluate("Set4", df)
    print(f"Set4 entry signal: {result['entry_signal']}")

    print(f"\nRegistered sets: {registry.list_sets()}")
    print("\nMulti-strategy set system smoke test passed.")
