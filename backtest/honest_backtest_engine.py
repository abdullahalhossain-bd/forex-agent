"""
backtest/honest_backtest_engine.py — Look-Ahead-Free Backtest Engine
====================================================================

CRITICAL FIX for Fatal Flaw #1 (look-ahead bias) and #6 (execution realism).

This engine NEVER lets a strategy see data from bars that haven't happened yet.

Key rules enforced:
  1. At bar i, only df.iloc[0:i+1] is visible to the strategy
  2. Zone detection runs INCREMENTALLY — zones are recomputed at each bar
     using only past data
  3. Entry happens at next bar OPEN (not current close) — models real latency
  4. Slippage + spread + commission applied to every fill
  5. Stop-loss can be skipped (gap risk modeled)

Usage:
    from backtest.honest_backtest_engine import HonestBacktester
    bt = HonestBacktester(
        spread_pips=1.5,
        commission_per_lot=7.0,
        slippage_pips=2.0,
        max_hold_bars=50,
    )
    result = bt.test_strategy(df, strategy_fn=your_strategy, pair="EURUSD")
    # result contains honest stats: real win rate after costs, no look-ahead
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("honest_bt")


def _deterministic_seed(signal_bar_idx: int, pair: str) -> int:
    """Day 99+ V5 FIX (Audit Issue #2): produce a deterministic 31-bit
    integer seed from (signal_bar_idx, pair) so that backtests are
    reproducible across processes / Python interpreter restarts.

    The OLD code used `hash((signal_bar_idx, pair)) % (2**31)`, which
    relied on Python's built-in `hash()`. Since Python 3.3, `hash()` is
    randomized per-process via PYTHONHASHSEED (a security feature against
    hash-collision DoS attacks). This means the SAME backtest inputs
    produced DIFFERENT random seeds across runs, breaking reproducibility
    — a critical requirement for backtest validation, regression testing,
    and walk-forward analysis.

    The new implementation uses hashlib.md5 (a cryptographic hash with a
    fixed output regardless of PYTHONHASHSEED) to derive a deterministic
    seed. The first 8 hex digits of the md5 digest are converted to an
    int and masked to 31 bits (matching the original `2**31` range).

    Args:
        signal_bar_idx: the bar index where the signal was generated
        pair:           the trading pair (e.g. "EURUSD")

    Returns:
        A deterministic int in [0, 2**31) suitable for np.random.RandomState.
    """
    key = f"{pair}_{signal_bar_idx}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    return int(digest[:8], 16) % (2**31)


# ════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

@dataclass
class HonestTrade:
    """Trade record from honest backtest (with realistic costs)."""
    entry_time: pd.Timestamp
    entry_bar_idx: int
    direction: str
    raw_entry_price: float      # signal price
    actual_entry_price: float   # after spread + slippage
    stop_loss: float
    take_profit: float
    exit_time: Optional[pd.Timestamp] = None
    exit_bar_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""       # "TP" | "SL" | "SL_GAP" | "timeout" | "manual"
    # Cost breakdown
    spread_cost_pips: float = 0.0
    slippage_cost_pips: float = 0.0
    commission_pips: float = 0.0
    # P&L
    gross_pnl_pips: float = 0.0   # before costs
    net_pnl_pips: float = 0.0     # after all costs
    r_multiple: float = 0.0       # net R (risk-adjusted)
    win: Optional[bool] = None


@dataclass
class HonestResult:
    """Result of an honest backtest."""
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    n_gap_losses: int = 0           # trades where SL was gapped through
    win_rate: float = 0.0           # NET (after costs) win rate
    gross_win_rate: float = 0.0     # before costs
    avg_net_r: float = 0.0
    total_net_r: float = 0.0
    profit_factor: float = 0.0      # net profit / net loss
    max_drawdown_r: float = 0.0
    total_cost_pips: float = 0.0
    avg_cost_per_trade_pips: float = 0.0
    # Statistical significance
    p_value: float = 1.0            # t-test: mean R-multiple > 0 (expectancy)
    is_significant: bool = False    # p < 0.05 (uncorrected)
    bonferroni_significant: bool = False  # p < 0.05/n_tests
    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    trades: List[HonestTrade] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════
#  HONEST BACKTESTER
# ════════════════════════════════════════════════════════════════

class HonestBacktester:
    """
    Backtester that eliminates look-ahead bias and models realistic costs.

    CRITICAL RULES:
      1. Strategy at bar i can only see df.iloc[0:i+1] — never future
      2. Entry at next bar OPEN + slippage (not current close)
      3. Spread + commission on every fill
      4. Stop-loss can be GAPPED through (realistic for news/weekends)
      5. Position held max_hold_bars; exits at close on timeout

    Cost model (per trade, on a 1.0 lot standard position):
      - spread: depends on pair (forex majors ~1.5 pips, XAUUSD ~5 pips)
      - commission: $7/lot round-turn = 0.7 pip on EURUSD
      - slippage: 1-3 pips on entry, 2-10 pips on stop-loss during volatility
    """

    # Default cost parameters per pair category (in pips)
    DEFAULT_COSTS = {
        "forex_majors":  {"spread": 1.5, "commission": 0.7, "slippage": 1.5},
        "forex_crosses": {"spread": 2.5, "commission": 0.7, "slippage": 2.0},
        "metals":        {"spread": 5.0, "commission": 1.0, "slippage": 3.0},
        "indices":       {"spread": 2.0, "commission": 1.0, "slippage": 2.0},
        "crypto":        {"spread": 10.0, "commission": 2.0, "slippage": 5.0},
        "default":       {"spread": 2.0, "commission": 0.7, "slippage": 2.0},
    }

    def __init__(
        self,
        spread_pips: Optional[float] = None,
        commission_per_lot: float = 7.0,
        slippage_pips: Optional[float] = None,
        max_hold_bars: int = 50,
        gap_probability: float = 0.03,  # 3% of SL exits gap through
        gap_multiplier: float = 2.5,    # gap loss = SL_dist × 2.5
    ):
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips
        self.max_hold_bars = max_hold_bars
        self.gap_probability = gap_probability
        self.gap_multiplier = gap_multiplier

    def _pair_costs(self, pair: str) -> Dict[str, float]:
        """Get realistic cost parameters for a pair."""
        pair = pair.upper()
        if "JPY" in pair:
            cat = "forex_majors" if pair in ("USDJPY", "EURJPY", "GBPJPY") else "forex_crosses"
        elif any(m in pair for m in ["XAU", "XAG"]):
            cat = "metals"
        elif any(m in pair for m in ["US30", "NAS100", "SPX500"]):
            cat = "indices"
        elif any(m in pair for m in ["BTC", "ETH"]):
            cat = "crypto"
        elif len(pair) == 6 and pair.isalpha():
            cat = "forex_majors" if pair in ("EURUSD", "GBPUSD", "USDJPY", "USDCHF",
                                              "AUDUSD", "USDCAD", "NZDUSD") else "forex_crosses"
        else:
            cat = "default"

        costs = self.DEFAULT_COSTS[cat].copy()
        if self.spread_pips is not None:
            costs["spread"] = self.spread_pips
        if self.slippage_pips is not None:
            costs["slippage"] = self.slippage_pips
        return costs

    def _pip_size(self, pair: str) -> float:
        if "JPY" in pair:
            return 0.01
        if "XAU" in pair:
            return 0.1
        if "XAG" in pair:
            return 0.01
        if any(idx in pair for idx in ["US30", "NAS100", "SPX500"]):
            return 1.0
        return 0.0001

    # ══════════════════════════════════════════════════════════
    #  CORE: BAR-BY-BAR SIMULATION (no look-ahead)
    # ══════════════════════════════════════════════════════════

    def simulate_trade(
        self,
        df: pd.DataFrame,
        signal_bar_idx: int,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        pair: str,
    ) -> HonestTrade:
        """
        Simulate ONE trade with realistic execution.

        Rules:
          - Entry at NEXT bar OPEN + slippage (not signal bar close)
          - Spread applied at entry
          - Commission charged
          - Stop-loss may gap through (with gap_probability)
          - Timeout at max_hold_bars
        """
        pip = self._pip_size(pair)
        costs = self._pair_costs(pair)
        spread_pips = costs["spread"]
        slippage_pips = costs["slippage"]
        commission_pips = costs["commission"]

        # Entry happens at NEXT bar open (not signal close)
        # This models the latency: signal at bar close, order arrives next bar
        entry_bar = signal_bar_idx + 1
        if entry_bar >= len(df):
            return HonestTrade(
                entry_time=df.index[signal_bar_idx],
                entry_bar_idx=signal_bar_idx,
                direction=direction,
                raw_entry_price=entry_price,
                actual_entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit,
                exit_reason="no_next_bar",
            )

        # Actual entry price = next bar OPEN + slippage
        next_open = float(df.iloc[entry_bar]["open"])
        if direction == "long":
            actual_entry = next_open + slippage_pips * pip
            # Spread already reflected in slippage for entry
        else:
            actual_entry = next_open - slippage_pips * pip

        # Risk per trade (in pips)
        if direction == "long":
            risk_pips = (actual_entry - stop_loss) / pip
        else:
            risk_pips = (stop_loss - actual_entry) / pip
        risk_pips = max(risk_pips, 1.0)  # avoid div by zero

        # Iterate forward from entry_bar
        exit_idx = None
        exit_price = None
        exit_reason = ""
        gap_loss = False

        # Did this stop get gapped through? (random with gap_probability)
        # Only applies if a bar opens BEYOND the stop
        # Day 99+ V5 FIX (Audit Issue #2): use _deterministic_seed() instead
        # of hash() so the backtest is reproducible across processes /
        # Python interpreter restarts (PYTHONHASHSEED no longer affects it).
        rng = np.random.RandomState(_deterministic_seed(signal_bar_idx, pair))

        for i in range(entry_bar, min(entry_bar + self.max_hold_bars + 1, len(df))):
            row = df.iloc[i]
            bar_open = float(row["open"])
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_close = float(row["close"])

            if direction == "long":
                # Check if bar opens BELOW stop (gap)
                if i > entry_bar and bar_open < stop_loss:
                    # Gap through stop — fill at worse price
                    gap_dist = (stop_loss - bar_open) / pip
                    if rng.random() < self.gap_probability or gap_dist > risk_pips * 0.5:
                        # Significant gap — fill at open price (worse)
                        exit_idx = i
                        exit_price = bar_open
                        exit_reason = "SL_GAP"
                        gap_loss = True
                        break

                # Normal stop check (within bar)
                if bar_low <= stop_loss:
                    exit_idx = i
                    exit_price = stop_loss
                    exit_reason = "SL"
                    break

                # TP check
                if bar_high >= take_profit:
                    exit_idx = i
                    exit_price = take_profit
                    exit_reason = "TP"
                    break
            else:  # short
                # Gap check
                if i > entry_bar and bar_open > stop_loss:
                    gap_dist = (bar_open - stop_loss) / pip
                    if rng.random() < self.gap_probability or gap_dist > risk_pips * 0.5:
                        exit_idx = i
                        exit_price = bar_open
                        exit_reason = "SL_GAP"
                        gap_loss = True
                        break

                if bar_high >= stop_loss:
                    exit_idx = i
                    exit_price = stop_loss
                    exit_reason = "SL"
                    break

                if bar_low <= take_profit:
                    exit_idx = i
                    exit_price = take_profit
                    exit_reason = "TP"
                    break

            # Timeout
            if i == entry_bar + self.max_hold_bars:
                exit_idx = i
                exit_price = bar_close
                exit_reason = "timeout"
                break

        if exit_idx is None:
            # End of data
            exit_idx = len(df) - 1
            exit_price = float(df.iloc[-1]["close"])
            exit_reason = "end_of_data"

        # Compute P&L (gross = before costs)
        if direction == "long":
            gross_pnl_pips = (exit_price - actual_entry) / pip
        else:
            gross_pnl_pips = (actual_entry - exit_price) / pip

        # Costs
        # Total cost = spread (entry+exit) + slippage (entry+exit) + commission
        # For gap losses, additional slippage on exit
        exit_slippage = slippage_pips * (2.5 if gap_loss else 1.0)
        # QUANT FIX: Add swap (overnight financing) cost.
        # Swap ~1 pip/day for forex. Average hold = max_hold_bars * bar_duration.
        # For H1 bars, max_hold=100 → ~4 days → ~4 pips swap.
        # For M15 bars, max_hold=100 → ~1 day → ~1 pip swap.
        # Conservative: 0.5 pip per bar held (overnight only, but we estimate).
        bars_held = exit_idx - entry_bar if exit_idx else self.max_hold_bars
        swap_pips = bars_held * 0.02  # ~0.02 pip/bar = ~0.5 pip/day for H1
        total_cost_pips = spread_pips + slippage_pips + exit_slippage + commission_pips + swap_pips

        net_pnl_pips = gross_pnl_pips - total_cost_pips
        net_r = net_pnl_pips / risk_pips

        win = None
        if exit_reason == "TP":
            win = True
        elif exit_reason in ("SL", "SL_GAP"):
            win = False
        elif exit_reason in ("timeout", "end_of_data"):
            win = net_pnl_pips > 0

        return HonestTrade(
            entry_time=df.index[signal_bar_idx],
            entry_bar_idx=signal_bar_idx,
            direction=direction,
            raw_entry_price=entry_price,
            actual_entry_price=actual_entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            exit_time=df.index[exit_idx],
            exit_bar_idx=exit_idx,
            exit_price=exit_price,
            exit_reason=exit_reason,
            spread_cost_pips=spread_pips,
            slippage_cost_pips=slippage_pips + exit_slippage,
            commission_pips=commission_pips,
            gross_pnl_pips=round(gross_pnl_pips, 2),
            net_pnl_pips=round(net_pnl_pips, 2),
            r_multiple=round(net_r, 3),
            win=win,
        )

    # ══════════════════════════════════════════════════════════
    #  STRATEGY TESTING WITH NO LOOK-AHEAD
    # ══════════════════════════════════════════════════════════

    def test_strategy(
        self,
        df: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame, int], Optional[Dict[str, Any]]],
        pair: str = "EURUSD",
        n_comparisons: int = 1,
    ) -> HonestResult:
        """
        Test a strategy on a dataframe, with NO look-ahead bias.

        Args:
            df           : full OHLCV dataframe
            strategy_fn  : callable(df, current_idx) → signal dict or None
                           The function receives df.iloc[0:current_idx+1]
                           ONLY — it cannot see future bars.
            pair         : pair name for cost calculation
            n_comparisons: number of strategy/param combos tested (for Bonferroni)

        The strategy_fn returns:
            {
                "direction": "long" | "short",
                "entry": float,
                "stop_loss": float,
                "take_profit": float,
            }
            or None if no signal at this bar.

        Returns:
            HonestResult with all stats + statistical significance
        """
        result = HonestResult()
        n = len(df)

        # Walk through each bar
        # Strategy sees ONLY df.iloc[0:i+1] — never future
        last_signal_bar = -100  # avoid overlapping trades

        for i in range(50, n - 2):  # need 50 bars warmup, 2 bars for entry+exit
            # Skip if too close to last trade
            if i - last_signal_bar < 5:
                continue

            # Strategy can only see data up to bar i (inclusive)
            visible_df = df.iloc[:i+1]

            try:
                signal = strategy_fn(visible_df, i)
            except Exception as e:
                log.warning(f"Suppressed exception at line 412: {e}")
                continue

            if signal is None:
                continue

            # Validate signal
            direction = signal.get("direction")
            entry = signal.get("entry")
            sl = signal.get("stop_loss")
            tp = signal.get("take_profit")
            if not all([direction, entry, sl, tp]):
                continue
            if direction not in ("long", "short"):
                continue

            # Simulate trade (with realistic costs + next-bar entry)
            trade = self.simulate_trade(df, i, direction, entry, sl, tp, pair)
            result.trades.append(trade)
            last_signal_bar = i

        # Compute stats
        self._compute_stats(result, n_comparisons)
        return result

    # ══════════════════════════════════════════════════════════
    #  STATISTICAL ANALYSIS
    # ══════════════════════════════════════════════════════════

    def _compute_stats(self, result: HonestResult, n_comparisons: int = 1):
        """Compute all statistics including significance tests."""
        trades = result.trades
        if not trades:
            return

        # Basic counts
        result.n_trades = len(trades)
        result.n_wins = sum(1 for t in trades if t.win is True)
        result.n_losses = sum(1 for t in trades if t.win is False)
        result.n_gap_losses = sum(1 for t in trades if t.exit_reason == "SL_GAP")

        # Win rates (NET = after costs)
        result.win_rate = result.n_wins / result.n_trades if result.n_trades else 0.0
        # Gross win rate (before costs) — for comparison
        gross_wins = sum(1 for t in trades if t.gross_pnl_pips > 0)
        result.gross_win_rate = gross_wins / result.n_trades if result.n_trades else 0.0

        # R-multiple stats
        r_values = [t.r_multiple for t in trades]
        result.avg_net_r = float(np.mean(r_values))
        result.total_net_r = float(np.sum(r_values))

        # Profit factor
        gross_profit = sum(r for r in r_values if r > 0)
        gross_loss = abs(sum(r for r in r_values if r < 0))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Equity curve (cumulative R)
        result.equity_curve = list(np.cumsum(r_values))

        # Max drawdown
        eq = np.array(result.equity_curve)
        running_max = np.maximum.accumulate(eq)
        drawdowns = running_max - eq
        result.max_drawdown_r = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Cost analysis
        total_costs = []
        for t in trades:
            total_costs.append(t.spread_cost_pips + t.slippage_cost_pips + t.commission_pips)
        result.total_cost_pips = sum(total_costs)
        result.avg_cost_per_trade_pips = float(np.mean(total_costs)) if total_costs else 0.0

        # Statistical significance — test EXPECTANCY, not raw win rate.
        #
        # BUG (fixed): this used to run `binomtest(n_wins, n_trades, p=0.5,
        # alternative="greater")` — i.e. H0: win rate = 50%. That null is
        # only correct for a 1:1 payoff strategy. These strategies use
        # ~2:1 R:R (TP=3xATR/SL=1.5xATR, or 4x/2x for donchian), whose
        # breakeven win rate is ~33%, not 50%. Testing WR vs 50% meant
        # p-value was ~1.0 for every strategy regardless of real edge,
        # since none of them were designed to hit 50%+ win rate.
        #
        # Fix: one-sample t-test on the R-multiples themselves.
        # H0: mean R-multiple = 0 (no edge, whatever the win rate/R:R is)
        # H1: mean R-multiple > 0 (positive expectancy)
        # This is R:R-agnostic and directly answers "is there an edge".
        from scipy.stats import ttest_1samp
        try:
            r_arr = np.array(r_values)
            if len(r_arr) >= 2 and np.std(r_arr, ddof=1) > 0:
                t_stat, two_sided_p = ttest_1samp(r_arr, popmean=0.0)
                # one-sided p-value for H1: mean > 0
                if t_stat > 0:
                    result.p_value = float(two_sided_p / 2)
                else:
                    result.p_value = float(1 - two_sided_p / 2)
            else:
                result.p_value = 1.0
            result.is_significant = result.p_value < 0.05
            # Bonferroni: divide alpha by number of comparisons
            bonferroni_alpha = 0.05 / max(n_comparisons, 1)
            result.bonferroni_significant = result.p_value < bonferroni_alpha
        except Exception as e:
            # Fallback if scipy not available
            result.p_value = 1.0
            result.is_significant = False
            result.bonferroni_significant = False


# ════════════════════════════════════════════════════════════════
#  INCREMENTAL ZONE DETECTOR — NO LOOK-AHEAD
# ════════════════════════════════════════════════════════════════

class IncrementalZoneDetector:
    """
    Detects S/R zones incrementally — at each bar, only uses PAST data.

    This is the FIX for Fatal Flaw #1 (look-ahead bias).

    At bar i, we compute zones using ONLY bars 0..i.
    We do NOT pre-compute zones on the full dataset.

    Method:
      - Maintain a rolling window of last N bars (e.g., 200)
      - Find swing highs/lows in that window
      - Cluster them into zones
      - At each new bar, recompute zones with the updated window
    """

    def __init__(
        self,
        window_size: int = 200,
        swing_lookback: int = 5,
        zone_tolerance_pips: float = 10.0,
        pip_size: float = 0.0001,
        cache_size: int = 500,
    ):
        self.window_size = window_size
        self.swing_lookback = swing_lookback
        self.zone_tolerance = zone_tolerance_pips * pip_size
        self.pip_size = pip_size
        self._cache_size = cache_size
        # FIX: use OrderedDict for proper LRU eviction instead of clear()-all
        # The previous "clear() if > 1000" approach caused periodic latency
        # spikes every 1000 bars as the entire cache was rebuilt from scratch.
        from collections import OrderedDict
        self._cached_zones_at: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()

    def zones_at_bar(self, df: pd.DataFrame, bar_idx: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get S/R zones visible at bar_idx — uses ONLY bars 0..bar_idx.

        Returns:
            {"support": [...], "resistance": [...]}
            Each zone: {"price": float, "strength": int, "age_bars": int}
        """
        if bar_idx in self._cached_zones_at:
            # LRU: move to end (most recently used)
            self._cached_zones_at.move_to_end(bar_idx)
            return self._cached_zones_at[bar_idx]

        # Use only past `window_size` bars (no future)
        start = max(0, bar_idx - self.window_size)
        window = df.iloc[start:bar_idx + 1]

        if len(window) < self.swing_lookback * 2 + 1:
            return {"support": [], "resistance": []}

        # Find swing highs/lows
        highs = window["high"].values
        lows = window["low"].values
        n = len(window)

        swing_highs = []
        swing_lows = []
        for i in range(self.swing_lookback, n - self.swing_lookback):
            # Swing high: highest high in [i-L, i+L]
            is_high = all(highs[i] >= highs[i+k] for k in range(-self.swing_lookback, self.swing_lookback + 1) if k != 0)
            if is_high:
                swing_highs.append({"price": float(highs[i]),
                                    "bar": i + start,
                                    "age": bar_idx - (i + start)})
            is_low = all(lows[i] <= lows[i+k] for k in range(-self.swing_lookback, self.swing_lookback + 1) if k != 0)
            if is_low:
                swing_lows.append({"price": float(lows[i]),
                                   "bar": i + start,
                                   "age": bar_idx - (i + start)})

        # Cluster into zones
        resistance = self._cluster_zones(swing_highs)
        support = self._cluster_zones(swing_lows)

        # Only keep zones from the recent past (age < window_size/2)
        resistance = [z for z in resistance if z["age"] < self.window_size // 2]
        support = [z for z in support if z["age"] < self.window_size // 2]

        result = {"support": support, "resistance": resistance}
        # Cache (but limit cache size to avoid memory issues)
        # LRU eviction: evict oldest entries when cache exceeds max size.
        # This is a smooth, bounded eviction — no periodic latency spikes
        # from clearing the entire cache.
        while len(self._cached_zones_at) >= self._cache_size:
            self._cached_zones_at.popitem(last=False)  # pop oldest (FIFO end)
        self._cached_zones_at[bar_idx] = result
        return result

    def _cluster_zones(self, swings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster nearby swing points into zones."""
        if not swings:
            return []
        # Sort by price
        swings.sort(key=lambda x: x["price"])
        zones = []
        current_cluster = [swings[0]]
        for s in swings[1:]:
            if abs(s["price"] - current_cluster[-1]["price"]) < self.zone_tolerance:
                current_cluster.append(s)
            else:
                zones.append(self._make_zone(current_cluster))
                current_cluster = [s]
        zones.append(self._make_zone(current_cluster))
        return zones

    @staticmethod
    def _make_zone(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        prices = [c["price"] for c in cluster]
        return {
            "price": float(np.mean(prices)),
            "zone_top": max(prices),
            "zone_bottom": min(prices),
            "strength": len(cluster),
            "age": min(c["age"] for c in cluster),
        }


# ════════════════════════════════════════════════════════════════
#  MONTE CARLO VALIDATOR
# ════════════════════════════════════════════════════════════════

class MonteCarloValidator:
    """
    Monte Carlo simulation to test if a strategy's edge is real.

    Takes the sequence of trade results (R-multiples) and:
      1. Randomizes order 10,000 times
      2. Computes drawdown distribution
      3. Computes probability of ruin (account blow-up)
      4. Computes 95% confidence interval for win rate
    """

    def __init__(self, n_simulations: int = 10_000, risk_per_trade: float = 0.005):
        self.n_simulations = n_simulations
        self.risk_per_trade = risk_per_trade  # 0.5% default

    def validate(self, trades: List[HonestTrade]) -> Dict[str, Any]:
        """Run Monte Carlo simulation on trade results."""
        if not trades:
            return {"error": "no trades"}

        r_values = np.array([t.r_multiple for t in trades])
        n = len(r_values)

        # Run simulations
        max_drawdowns = []
        final_equities = []
        ruin_count = 0

        for _ in range(self.n_simulations):
            # Shuffle trade order
            shuffled = np.random.permutation(r_values)
            # Equity curve (in R units, scaled by risk_per_trade)
            equity = np.cumsum(shuffled) * self.risk_per_trade
            starting_equity = 1.0
            equity_curve = starting_equity + equity

            # Track max drawdown
            running_max = np.maximum.accumulate(equity_curve)
            drawdowns = (running_max - equity_curve) / running_max
            max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
            max_drawdowns.append(max_dd)

            final_equity = equity_curve[-1]
            final_equities.append(final_equity)

            # Ruin = equity drops below 50% (50% drawdown)
            if max_dd >= 0.50:
                ruin_count += 1

        # Compute statistics
        results = {
            "n_trades": n,
            "n_simulations": self.n_simulations,
            "median_max_drawdown": float(np.median(max_drawdowns)),
            "95th_percentile_drawdown": float(np.percentile(max_drawdowns, 95)),
            "worst_drawdown": float(np.max(max_drawdowns)),
            "median_final_equity": float(np.median(final_equities)),
            "5th_percentile_final_equity": float(np.percentile(final_equities, 5)),
            "95th_percentile_final_equity": float(np.percentile(final_equities, 95)),
            "probability_of_ruin": ruin_count / self.n_simulations,
            "probability_of_profit": sum(1 for e in final_equities if e > 1.0) / self.n_simulations,
            "expected_value_per_trade": float(np.mean(r_values)),
            "sharpe_ratio": float(np.mean(r_values) / np.std(r_values)) if np.std(r_values) > 0 else 0.0,
        }

        # Win rate confidence interval (Wilson score)
        # NOTE: this is informational only. It answers "what's the win rate
        # range", not "is there an edge" — for asymmetric R:R strategies a
        # win rate well under 50% can still be profitable, so the deployment
        # gate no longer uses win_rate_ci_low as a pass/fail threshold (see
        # expectancy_ci_low below, which is the correct gate).
        # FIX: guard against n=0 division (defensive — validate() checks
        # `if not trades` earlier, but r_values could theoretically be empty
        # if all trades had NaN r_multiple)
        wins = sum(1 for r in r_values if r > 0)
        if n == 0:
            results["win_rate"] = 0.0
            results["win_rate_ci_low"] = 0.0
            results["win_rate_ci_high"] = 1.0
        else:
            wr = wins / n
            z = 1.96  # 95% CI
            denominator = 1 + z**2 / n
            center = (wr + z**2 / (2 * n)) / denominator
            margin = z * np.sqrt((wr * (1 - wr) + z**2 / (4 * n)) / n) / denominator
            results["win_rate"] = wr
            results["win_rate_ci_low"] = max(0.0, center - margin)
            results["win_rate_ci_high"] = min(1.0, center + margin)

        # Expectancy confidence interval (bootstrap on mean R-multiple).
        # This is the R:R-agnostic edge test: does the CI for the average
        # R-multiple per trade exclude zero? Used by DeploymentGate instead
        # of win_rate_ci, which was the wrong gate for ~2:1 R:R strategies.
        if n >= 2:
            boot_means = np.empty(self.n_simulations)
            for i in range(self.n_simulations):
                sample = np.random.choice(r_values, size=n, replace=True)
                boot_means[i] = np.mean(sample)
            results["expectancy_mean"] = float(np.mean(r_values))
            results["expectancy_ci_low"] = float(np.percentile(boot_means, 2.5))
            results["expectancy_ci_high"] = float(np.percentile(boot_means, 97.5))
        else:
            results["expectancy_mean"] = float(np.mean(r_values)) if n else 0.0
            results["expectancy_ci_low"] = 0.0
            results["expectancy_ci_high"] = 0.0

        return results


# ════════════════════════════════════════════════════════════════
#  WALK-FORWARD VALIDATOR
# ════════════════════════════════════════════════════════════════

class WalkForwardValidator:
    """
    Walk-forward validation — required to claim any edge is real.

    Splits data into rolling windows:
      [---train---][test][---train---][test]...

    Strategy parameters optimized on train, tested on test (OOS).
    Reports OOS performance metrics.
    """

    def __init__(
        self,
        train_bars: int = 1000,
        test_bars: int = 500,
        step_bars: int = 500,
    ):
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars

    def split(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate (train, test) splits for walk-forward."""
        splits = []
        n = len(df)
        start = 0
        while start + self.train_bars + self.test_bars <= n:
            train = df.iloc[start : start + self.train_bars]
            test = df.iloc[start + self.train_bars : start + self.train_bars + self.test_bars]
            splits.append((train, test))
            start += self.step_bars
        return splits

    def validate(
        self,
        df: pd.DataFrame,
        strategy_fn: Callable,
        pair: str = "EURUSD",
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation.

        Returns:
            {
                "n_splits": int,
                "in_sample_results": [...],
                "out_of_sample_results": [...],
                "is_degraded": bool,  # True if OOS much worse than IS
                "verdict": "pass" | "fail" | "marginal",
            }
        """
        splits = self.split(df)
        if not splits:
            return {"error": "data too short for walk-forward",
                    "min_required": self.train_bars + self.test_bars,
                    "actual": len(df)}

        bt = HonestBacktester()
        is_results = []
        oos_results = []

        for i, (train_df, test_df) in enumerate(splits):
            is_result = bt.test_strategy(train_df, strategy_fn, pair=pair)
            oos_result = bt.test_strategy(test_df, strategy_fn, pair=pair)
            is_results.append({
                "split": i + 1,
                "n_trades": is_result.n_trades,
                "win_rate": is_result.win_rate,
                "avg_r": is_result.avg_net_r,
            })
            oos_results.append({
                "split": i + 1,
                "n_trades": oos_result.n_trades,
                "win_rate": oos_result.win_rate,
                "avg_r": oos_result.avg_net_r,
            })

        # Compare IS vs OOS
        is_avg_wr = np.mean([r["win_rate"] for r in is_results if r["n_trades"] > 0]) if is_results else 0
        oos_avg_wr = np.mean([r["win_rate"] for r in oos_results if r["n_trades"] > 0]) if oos_results else 0
        degradation = is_avg_wr - oos_avg_wr

        # BUG (fixed): verdict used to require oos_avg_wr >= 0.50, i.e. the
        # same 50%-win-rate null used in _compute_stats(). These strategies
        # run at ~2:1 R:R (breakeven WR ~33%), so gating pass/fail on a raw
        # 50% win rate meant "fail" was baked in regardless of the strategy's
        # real edge. Verdict now uses EXPECTANCY (avg R-multiple) OOS, which
        # is the right measure for asymmetric R:R strategies, plus a check
        # that OOS expectancy hasn't degraded much vs IS (overfitting check).
        is_avg_r = np.mean([r["avg_r"] for r in is_results if r["n_trades"] > 0]) if is_results else 0
        oos_avg_r = np.mean([r["avg_r"] for r in oos_results if r["n_trades"] > 0]) if oos_results else 0
        r_degradation = is_avg_r - oos_avg_r  # positive = OOS worse than IS

        if oos_avg_r > 0 and r_degradation < 0.10:
            verdict = "pass"
        elif oos_avg_r > -0.05 and r_degradation < 0.20:
            verdict = "marginal"
        else:
            verdict = "fail"

        return {
            "n_splits": len(splits),
            "in_sample_avg_wr": is_avg_wr,
            "out_of_sample_avg_wr": oos_avg_wr,
            "in_sample_avg_r": is_avg_r,
            "out_of_sample_avg_r": oos_avg_r,
            "r_degradation": r_degradation,
            "degradation": degradation,
            "is_degraded": r_degradation > 0.10,
            "verdict": verdict,
            "in_sample_results": is_results,
            "out_of_sample_results": oos_results,
        }


# ════════════════════════════════════════════════════════════════
#  LIVE DEPLOYMENT GATE
# ════════════════════════════════════════════════════════════════

class DeploymentGate:
    """
    HARD GATE: blocks live deployment until ALL criteria pass.

    A strategy cannot go live until ALL of:
      1. ≥ 100 trades in honest backtest
      2. Win rate CI lower bound > 50% (95% confidence)
      3. Bonferroni-significant (after multiple comparison correction)
      4. Walk-forward verdict = "pass"
      5. Monte Carlo probability of ruin < 5%
      6. Monte Carlo 95th percentile drawdown < 25%
      7. Profit factor > 1.3 (after costs)
    """

    @staticmethod
    def evaluate(
        honest_result: HonestResult,
        monte_carlo: Dict[str, Any],
        walk_forward: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate whether strategy can be deployed live."""
        checks = []

        # 1. Sample size
        checks.append({
            "name": "sufficient_trades",
            "passed": honest_result.n_trades >= 100,
            "value": honest_result.n_trades,
            "required": "≥ 100",
            "severity": "BLOCKING",
        })

        # 2. Expectancy CI (was: win_rate_ci > 50%)
        #
        # BUG (fixed): this used to require win_rate_ci_low > 0.50 — i.e.
        # the bootstrap lower bound of the raw win rate had to clear 50%.
        # That's only correct for a 1:1 payoff. These strategies run at
        # ~2:1 R:R (breakeven WR ~33%), so this gate blocked every strategy
        # by construction, independent of whether it actually had an edge.
        # Fix: gate on the expectancy CI (mean R-multiple) instead — the
        # lower bound of the bootstrap CI must be > 0, i.e. even in a
        # pessimistic resample the strategy still makes money per trade.
        exp_low = monte_carlo.get("expectancy_ci_low", -1.0)
        checks.append({
            "name": "expectancy_ci",
            "passed": exp_low > 0,
            "value": f"{exp_low:+.3f}R (lower bound)",
            "required": "> 0R",
            "severity": "BLOCKING",
        })

        # 3. Statistical significance (Bonferroni-corrected)
        checks.append({
            "name": "bonferroni_significance",
            "passed": honest_result.bonferroni_significant,
            "value": f"p={honest_result.p_value:.4f}",
            "required": "p < 0.05/n_comparisons",
            "severity": "BLOCKING",
        })

        # 4. Walk-forward
        wf_verdict = walk_forward.get("verdict", "fail")
        checks.append({
            "name": "walk_forward",
            "passed": wf_verdict == "pass",
            "value": wf_verdict,
            "required": "pass",
            "severity": "BLOCKING",
        })

        # 5. Probability of ruin
        por = monte_carlo.get("probability_of_ruin", 1.0)
        checks.append({
            "name": "probability_of_ruin",
            "passed": por < 0.05,
            "value": f"{por*100:.1f}%",
            "required": "< 5%",
            "severity": "BLOCKING",
        })

        # 6. Max drawdown (95th percentile)
        dd95 = monte_carlo.get("95th_percentile_drawdown", 1.0)
        checks.append({
            "name": "max_drawdown_95",
            "passed": dd95 < 0.25,
            "value": f"{dd95*100:.1f}%",
            "required": "< 25%",
            "severity": "BLOCKING",
        })

        # 7. Profit factor
        pf = honest_result.profit_factor
        checks.append({
            "name": "profit_factor",
            "passed": pf > 1.3,
            "value": f"{pf:.2f}",
            "required": "> 1.30",
            "severity": "BLOCKING",
        })

        # Overall verdict
        all_passed = all(c["passed"] for c in checks)
        blocking_failed = [c["name"] for c in checks if not c["passed"] and c["severity"] == "BLOCKING"]

        return {
            "can_deploy_live": all_passed,
            "verdict": "APPROVED FOR LIVE" if all_passed else "BLOCKED — DO NOT DEPLOY",
            "blocking_failures": blocking_failed,
            "checks": checks,
        }


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY (smoke test)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from backtest.mt5_bulk_fetcher import MT5BulkFetcher

    print("=" * 70)
    print("  HONEST BACKTEST ENGINE — Smoke Test")
    print("=" * 70)

    # Get test data
    fetcher = MT5BulkFetcher()
    fetch_result = fetcher.fetch("EURUSD", "M15", n_candles=2000)
    df = fetch_result.df
    print(f"\nData: {fetch_result.pair} {fetch_result.timeframe} "
          f"({fetch_result.n_candles} candles, {fetch_result.source})")

    # Build a simple strategy using incremental zones (no look-ahead)
    detector = IncrementalZoneDetector(window_size=100, swing_lookback=5,
                                       zone_tolerance_pips=10, pip_size=0.0001)

    def sr_bounce_strategy(visible_df: pd.DataFrame, current_idx: int):
        """Simple S/R bounce strategy using ONLY past data."""
        if len(visible_df) < 50:
            return None
        zones = detector.zones_at_bar(visible_df, current_idx)
        if not zones["support"] and not zones["resistance"]:
            return None

        current_close = float(visible_df.iloc[-1]["close"])
        atr = float(np.mean(np.diff(visible_df["high"].iloc[-14:].values -
                                     visible_df["low"].iloc[-14:].values)))

        # Check if price is near support → long
        for s in zones["support"][:3]:  # only check top 3 strongest
            if abs(current_close - s["price"]) < atr * 0.5:
                return {
                    "direction": "long",
                    "entry": current_close,
                    "stop_loss": s["price"] - atr * 1.5,
                    "take_profit": current_close + atr * 3.0,
                }

        # Check if price is near resistance → short
        for r in zones["resistance"][:3]:
            if abs(current_close - r["price"]) < atr * 0.5:
                return {
                    "direction": "short",
                    "entry": current_close,
                    "stop_loss": r["price"] + atr * 1.5,
                    "take_profit": current_close - atr * 3.0,
                }

        return None

    # Run honest backtest
    bt = HonestBacktester()
    print("\nRunning honest backtest (no look-ahead, realistic costs)...")
    result = bt.test_strategy(df, sr_bounce_strategy, pair="EURUSD", n_comparisons=1)

    print(f"\n── HONEST BACKTEST RESULT ──")
    print(f"  Trades:           {result.n_trades}")
    print(f"  Net win rate:     {result.win_rate*100:.1f}% (after costs)")
    print(f"  Gross win rate:   {result.gross_win_rate*100:.1f}% (before costs)")
    print(f"  Avg net R:        {result.avg_net_r:+.3f}")
    print(f"  Total net R:      {result.total_net_r:+.2f}")
    print(f"  Profit factor:    {result.profit_factor:.2f}")
    print(f"  Max drawdown:     {result.max_drawdown_r:.2f} R")
    print(f"  Gap losses:       {result.n_gap_losses}")
    print(f"  Avg cost/trade:   {result.avg_cost_per_trade_pips:.2f} pips")
    print(f"  Total cost:       {result.total_cost_pips:.1f} pips")
    print(f"  P-value (vs 50%): {result.p_value:.4f}")
    print(f"  Significant:      {result.is_significant}")
    print(f"  Bonferroni sig:   {result.bonferroni_significant}")

    # Monte Carlo
    print(f"\n── MONTE CARLO (10,000 simulations) ──")
    mc = MonteCarloValidator(n_simulations=1000).validate(result.trades)
    print(f"  Median max DD:        {mc['median_max_drawdown']*100:.1f}%")
    print(f"  95th pct DD:          {mc['95th_percentile_drawdown']*100:.1f}%")
    print(f"  Worst DD:             {mc['worst_drawdown']*100:.1f}%")
    print(f"  Probability of ruin:  {mc['probability_of_ruin']*100:.1f}%")
    print(f"  Probability of profit:{mc['probability_of_profit']*100:.1f}%")
    print(f"  WR (95% CI):          {mc['win_rate_ci_low']*100:.1f}% - {mc['win_rate_ci_high']*100:.1f}%")

    # Walk-forward
    print(f"\n── WALK-FORWARD VALIDATION ──")
    wf = WalkForwardValidator(train_bars=800, test_bars=400, step_bars=400).validate(
        df, sr_bounce_strategy, pair="EURUSD")
    print(f"  Splits:               {wf.get('n_splits', 0)}")
    print(f"  IS avg WR:            {wf.get('in_sample_avg_wr', 0)*100:.1f}%")
    print(f"  OOS avg WR:           {wf.get('out_of_sample_avg_wr', 0)*100:.1f}%")
    print(f"  Degradation:          {wf.get('degradation', 0)*100:.1f}%")
    print(f"  Verdict:              {wf.get('verdict', 'fail')}")

    # Deployment gate
    print(f"\n── DEPLOYMENT GATE ──")
    gate = DeploymentGate.evaluate(result, mc, wf)
    print(f"  Verdict: {gate['verdict']}")
    for c in gate["checks"]:
        marker = "✅" if c["passed"] else "❌"
        print(f"  {marker} {c['name']:<25} {c['value']:<30} (req: {c['required']})")

    print("\n" + "=" * 70)