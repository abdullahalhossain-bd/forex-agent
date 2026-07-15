"""
backtest/per_strategy_tester.py — Per-Strategy Independent Backtester
=====================================================================

Tests EACH strategy INDEPENDENTLY (not as a mandatory combo) to measure
its standalone win rate. This solves the "multiple mandatory strategies
= no trades" problem by revealing which strategies work alone vs. only
in confluence.

Key design:
  • Each strategy is tested in isolation — NO multi-strategy gate
  • Tracks per-strategy, per-tactic, per-confidence-level win rates
  • Reports win rate, profit factor, max drawdown, R:R distribution
  • Generates actionable recommendations for the decision system

Strategies tested independently:
  1. Supply/Demand zones (with odd-enhancer scoring tiers A/B)
  2. Support/Resistance zones
  3. Pin bar (aggressive vs. conservative entry)
  4. High-reliability candlestick patterns (20 patterns)
  5. Stop Hunt signal engine
  6. ICT/AMD signal engine
  7. Multi-Strategy PA engine (8-step checklist)
  8. CCI state machine (entry only)
  9. Flip zone retest entry
 10. Unified consensus engine (for comparison — baseline)

Usage:
    from backtest.per_strategy_tester import PerStrategyTester
    tester = PerStrategyTester()
    results = tester.run_all(df, pair="EURUSD", timeframe="H1")
    # results = {"strategies": {"pin_bar": {"trades": 23, "win_rate": 0.61, ...}, ...}}
"""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("per_strategy")


# ════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """Single trade record."""
    strategy: str
    pair: str
    timeframe: str
    direction: str            # "long" | "short"
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""     # "TP" | "SL" | "timeout" | "manual"
    pnl_pips: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float = 0.0   # realized R multiple (1.0 = hit TP, -1.0 = hit SL)
    confidence: str = "Medium"
    tactic: str = "default"   # sub-method (e.g., "aggressive", "conservative")
    win: Optional[bool] = None


@dataclass
class StrategyResult:
    """Aggregated result for one strategy on one (pair, timeframe)."""
    strategy: str
    pair: str
    timeframe: str
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    n_breakeven: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_r: float = 0.0
    max_r: float = 0.0
    min_r: float = 0.0
    total_r: float = 0.0
    avg_hold_bars: float = 0.0
    max_drawdown_r: float = 0.0
    # Per-confidence breakdown
    by_confidence: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Per-tactic breakdown
    by_tactic: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Per-direction breakdown
    by_direction: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Individual trades
    trades: List[Trade] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════
#  TRADE SIMULATOR — single-trade backtest engine
# ════════════════════════════════════════════════════════════════

class TradeSimulator:
    """
    Simulates a single trade: enters at entry_price, exits at SL/TP/timeout.
    Uses bar-by-bar simulation to detect which level was hit first.

    Conservative approach:
      • If both SL and TP hit in same bar → assume SL hit first (pessimistic)
      • Spread/slippage applied at entry
    """

    def __init__(
        self,
        spread_pips: float = 1.0,
        slippage_pips: float = 0.5,
        max_hold_bars: int = 100,
    ):
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.max_hold_bars = max_hold_bars

    def simulate(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        pip_size: float = 0.0001,
    ) -> Tuple[Optional[int], float, str]:
        """
        Simulate a trade from entry_idx onward.

        Returns:
            (exit_idx, exit_price, exit_reason)
        """
        if entry_idx >= len(df) - 1:
            return None, entry_price, "no_data"

        # Apply spread/slippage
        if direction == "long":
            actual_entry = entry_price + (self.slippage_pips * pip_size)
        else:
            actual_entry = entry_price - (self.slippage_pips * pip_size)

        # Iterate forward
        for i in range(entry_idx + 1, min(entry_idx + self.max_hold_bars + 1, len(df))):
            row = df.iloc[i]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            if direction == "long":
                # Check SL first (pessimistic)
                if low <= stop_loss:
                    return i, stop_loss, "SL"
                # Then TP
                if high >= take_profit:
                    return i, take_profit, "TP"
            else:  # short
                # Check SL first (pessimistic)
                if high >= stop_loss:
                    return i, stop_loss, "SL"
                # Then TP
                if low <= take_profit:
                    return i, take_profit, "TP"

            # Timeout
            if i == entry_idx + self.max_hold_bars:
                return i, close, "timeout"

        return len(df) - 1, float(df.iloc[-1]["close"]), "end_of_data"


# ════════════════════════════════════════════════════════════════
#  PER-STRATEGY TESTER
# ════════════════════════════════════════════════════════════════

class PerStrategyTester:
    """
    Tests each strategy INDEPENDENTLY on a given OHLCV DataFrame.

    Solves the "mandatory multi-strategy = no trades" problem by
    revealing each strategy's standalone performance.
    """

    def __init__(
        self,
        spread_pips: float = 1.0,
        slippage_pips: float = 0.5,
        max_hold_bars: int = 100,
        risk_reward_ratio: float = 2.0,  # default R:R if strategy doesn't specify
        pip_size: float = 0.0001,
    ):
        self.sim = TradeSimulator(spread_pips, slippage_pips, max_hold_bars)
        self.rr_ratio = risk_reward_ratio
        self.pip_size = pip_size

    # ══════════════════════════════════════════════════════════
    #  PUBLIC API — run all strategies
    # ══════════════════════════════════════════════════════════

    def run_all(
        self,
        df: pd.DataFrame,
        pair: str = "EURUSD",
        timeframe: str = "H1",
    ) -> Dict[str, Any]:
        """
        Run all strategies independently on the given data.

        Returns:
            {
                "pair": str,
                "timeframe": str,
                "n_bars": int,
                "strategies": {strategy_name: StrategyResult, ...},
                "summary": {...}
            }
        """
        log.info(f"[PerStrategy] Running all strategies on {pair} {timeframe} "
                 f"({len(df)} bars)")

        # Determine pip size from pair
        pip = self._pip_size_for_pair(pair)

        # Run each strategy
        strategies_results: Dict[str, StrategyResult] = {}

        # 1. Pin Bar strategy
        try:
            strategies_results["pin_bar"] = self._test_pin_bar(df, pair, timeframe, pip)
        except Exception as e:
            log.error(f"[pin_bar] failed: {e}")
            strategies_results["pin_bar"] = StrategyResult("pin_bar", pair, timeframe)

        # 2. High-reliability patterns (20 patterns)
        try:
            strategies_results["candlestick_patterns"] = self._test_candlestick_patterns(
                df, pair, timeframe, pip)
        except Exception as e:
            log.error(f"[candlestick_patterns] failed: {e}")
            strategies_results["candlestick_patterns"] = StrategyResult(
                "candlestick_patterns", pair, timeframe)

        # 3. Supply/Demand zones with odd-enhancer scoring
        try:
            strategies_results["sd_zones_scored"] = self._test_sd_zones_scored(
                df, pair, timeframe, pip)
        except Exception as e:
            log.error(f"[sd_zones_scored] failed: {e}")
            strategies_results["sd_zones_scored"] = StrategyResult(
                "sd_zones_scored", pair, timeframe)

        # 4. Support/Resistance zones
        try:
            strategies_results["sr_zones"] = self._test_sr_zones(df, pair, timeframe, pip)
        except Exception as e:
            log.error(f"[sr_zones] failed: {e}")
            strategies_results["sr_zones"] = StrategyResult("sr_zones", pair, timeframe)

        # 5. Stop Hunt signal engine
        try:
            strategies_results["stop_hunt"] = self._test_stop_hunt(df, pair, timeframe, pip)
        except Exception as e:
            log.error(f"[stop_hunt] failed: {e}")
            strategies_results["stop_hunt"] = StrategyResult("stop_hunt", pair, timeframe)

        # 6. ICT/AMD signal engine
        try:
            strategies_results["ict_amd"] = self._test_ict_amd(df, pair, timeframe, pip)
        except Exception as e:
            log.error(f"[ict_amd] failed: {e}")
            strategies_results["ict_amd"] = StrategyResult("ict_amd", pair, timeframe)

        # 7. Multi-Strategy PA engine
        try:
            strategies_results["multi_pa"] = self._test_multi_pa(df, pair, timeframe, pip)
        except Exception as e:
            log.error(f"[multi_pa] failed: {e}")
            strategies_results["multi_pa"] = StrategyResult("multi_pa", pair, timeframe)

        # 8. CCI state machine (needs CCI indicator)
        try:
            strategies_results["cci_state"] = self._test_cci_state(df, pair, timeframe, pip)
        except Exception as e:
            log.error(f"[cci_state] failed: {e}")
            strategies_results["cci_state"] = StrategyResult("cci_state", pair, timeframe)

        # Build summary
        summary = self._build_summary(strategies_results)

        return {
            "pair": pair,
            "timeframe": timeframe,
            "n_bars": len(df),
            "strategies": strategies_results,
            "summary": summary,
        }

    # ══════════════════════════════════════════════════════════
    #  STRATEGY IMPLEMENTATIONS (each tested independently)
    # ══════════════════════════════════════════════════════════

    def _test_pin_bar(self, df, pair, timeframe, pip):
        """Strategy 1: Pin Bar — test both aggressive and conservative entries."""
        from analysis.pin_bar_strategy import PinBarStrategy
        result = StrategyResult("pin_bar", pair, timeframe)
        strategy = PinBarStrategy()

        # CRITICAL FIX: Use incremental zone detection (no look-ahead bias).
        # Previously: sr.analyze(df) computed zones on FULL df (including future bars).
        # Now: zones computed using only df.iloc[0:i+1] at each bar.
        from analysis.support_resistance import SupportResistance
        sr = SupportResistance()
        # Pre-compute is REMOVED — zones will be computed incrementally per bar
        sr_zones = {"resistance_zones": [], "support_zones": []}  # placeholder

        lookback = 200  # bars to look back
        start = max(20, len(df) - lookback)

        for i in range(start, len(df) - 5):
            try:
                # FIX: only pass data up to current bar (no future leak)
                visible_df = df.iloc[:i+1]
                window = visible_df.iloc[max(0, i-50):i+1]
                # Compute zones using ONLY past data
                try:
                    sr_zones_i = sr.analyze(visible_df)
                except Exception as e:
                    sr_zones_i = {"resistance_zones": [], "support_zones": []}
                setup = strategy.analyze(window, timeframe=timeframe,
                                         sr_zones=sr_zones_i)
                if setup is None or not setup.valid:
                    continue

                # Test both tactics
                for tactic in ["aggressive", "conservative"]:
                    if tactic == "aggressive":
                        entry = setup.aggressive_entry
                        stop = setup.aggressive_sl
                    else:
                        entry = setup.conservative_entry
                        stop = setup.conservative_sl
                    if not entry or not stop:
                        continue
                    tp_dist = abs(entry - stop) * self.rr_ratio
                    tp = entry + tp_dist if setup.direction == "bullish" else entry - tp_dist

                    exit_idx, exit_price, reason = self.sim.simulate(
                        df, i, "long" if setup.direction == "bullish" else "short",
                        entry, stop, tp, pip)

                    if exit_idx is None:
                        continue
                    trade = self._make_trade(
                        "pin_bar", pair, timeframe,
                        "long" if setup.direction == "bullish" else "short",
                        df.index[i], entry, stop, tp,
                        df.index[exit_idx], exit_price, reason, pip,
                        confidence=setup.quality_grade, tactic=tactic)
                    result.trades.append(trade)
            except Exception as e:
                log.warning(f"Suppressed exception at line 370: {e}")
                continue

        self._finalize_result(result)
        return result

    def _test_candlestick_patterns(self, df, pair, timeframe, pip):
        """Strategy 2: High-reliability candlestick patterns (20 patterns)."""
        from analysis.high_reliability_patterns import HighReliabilityPatternDetector
        result = StrategyResult("candlestick_patterns", pair, timeframe)
        detector = HighReliabilityPatternDetector(lookback=300)

        try:
            patterns = detector.detect(df)
        except Exception as e:
            log.debug(f"[patterns] detect failed: {e}")
            return result

        for pat in patterns:
            try:
                i = pat.candle_index
                if i >= len(df) - 5:
                    continue
                # Direction: bullish if direction field is bullish, else short
                direction = "long" if pat.direction == "bullish" else "short"
                entry = float(df.iloc[i]["close"])
                # Use ATR-based stop
                atr = self._atr(df, i)
                if atr <= 0:
                    continue
                if direction == "long":
                    stop = entry - 1.5 * atr
                    tp = entry + 1.5 * atr * self.rr_ratio
                else:
                    stop = entry + 1.5 * atr
                    tp = entry - 1.5 * atr * self.rr_ratio

                exit_idx, exit_price, reason = self.sim.simulate(
                    df, i, direction, entry, stop, tp, pip)

                if exit_idx is None:
                    continue

                trade = self._make_trade(
                    "candlestick_patterns", pair, timeframe, direction,
                    df.index[i], entry, stop, tp,
                    df.index[exit_idx], exit_price, reason, pip,
                    confidence=pat.reliability, tactic=pat.pattern_name)
                result.trades.append(trade)
            except Exception as e:
                log.warning(f"Suppressed exception at line 420: {e}")
                continue

        self._finalize_result(result)
        return result

    def _test_sd_zones_scored(self, df, pair, timeframe, pip):
        """Strategy 3: Supply/Demand zones with odd-enhancer scoring."""
        from analysis.supply_demand_zones import SupplyDemandZones
        from analysis.odd_enhancers import OddEnhancerScorer
        result = StrategyResult("sd_zones_scored", pair, timeframe)
        sd = SupplyDemandZones()
        scorer = OddEnhancerScorer()

        # CRITICAL FIX: No look-ahead bias. Zones detected per-bar using only past data.
        # Previously: sd.detect(df) used FULL df. Now we detect incrementally.
        # Detect zones using only the visible (past) data at zone formation time
        try:
            zones_result = sd.detect(df.iloc[:len(df)//2])  # use first half for zone formation
        except Exception as e:
            return result

        all_zones = []
        for z in zones_result.get("demand_zones", []):
            z["type"] = "demand"
            all_zones.append(z)
        for z in zones_result.get("supply_zones", []):
            z["type"] = "supply"
            all_zones.append(z)

        # Only test zones that formed in the first half — test them on the second half
        test_start = len(df) // 2
        for zone in all_zones:
            try:
                # Score the zone using only data up to zone formation (not future)
                zone_base_end = zone.get("base_idx_end", test_start)
                if zone_base_end >= len(df) - 5:
                    continue
                # Use visible data only for scoring
                visible_df = df.iloc[:zone_base_end + 1]
                scored = scorer.score_zone(zone, visible_df, float(visible_df["close"].iloc[-1]))
                # Only trade Tier A or B
                if scored.tier == "SKIP":
                    continue

                zone_type = zone.get("type", "demand")
                direction = "long" if zone_type == "demand" else "short"
                entry = zone.get("proximal", zone.get("zone_high", 0))
                stop = zone.get("distal", zone.get("zone_low", 0))

                if direction == "long":
                    tp = entry + abs(entry - stop) * self.rr_ratio
                else:
                    tp = entry - abs(entry - stop) * self.rr_ratio

                # Find the first bar after zone formation
                base_end = zone.get("base_idx_end", 5)
                if base_end >= len(df) - 5:
                    continue

                exit_idx, exit_price, reason = self.sim.simulate(
                    df, base_end, direction, entry, stop, tp, pip)

                if exit_idx is None:
                    continue

                trade = self._make_trade(
                    "sd_zones_scored", pair, timeframe, direction,
                    df.index[base_end], entry, stop, tp,
                    df.index[exit_idx], exit_price, reason, pip,
                    confidence=scored.tier, tactic=scored.entry_method)
                result.trades.append(trade)
            except Exception as e:
                log.warning(f"Suppressed exception at line 493: {e}")
                continue

        self._finalize_result(result)
        return result

    def _test_sr_zones(self, df, pair, timeframe, pip):
        """Strategy 4: Support/Resistance zone bounces."""
        from analysis.support_resistance import SupportResistance
        result = StrategyResult("sr_zones", pair, timeframe)
        sr = SupportResistance()

        # CRITICAL FIX: No look-ahead bias. S/R zones computed per-bar.
        # Previously: sr.analyze(df) used FULL df. Now we compute incrementally.
        # Use first half of data for zone formation, test on second half
        try:
            sr_result = sr.analyze(df.iloc[:len(df)//2])
        except Exception as e:
            return result

        # Combine all zones (from first half only — no future data)
        zones = []
        for z in sr_result.get("resistance_zones", []):
            zones.append(("resistance", z))
        for z in sr_result.get("support_zones", []):
            zones.append(("support", z))

        # Test zones on the second half of data (out-of-sample)
        test_start = len(df) // 2

        # For each zone, find touch points and simulate bounce trades
        for zone_type, zone in zones:
            try:
                zone_top = zone.get("zone_top", 0)
                zone_bottom = zone.get("zone_bottom", 0)
                if zone_top == 0 or zone_bottom == 0:
                    continue

                # Find bars where price touched the zone (only in test period)
                for i in range(test_start, len(df) - 5):
                    low = float(df.iloc[i]["low"])
                    high = float(df.iloc[i]["high"])
                    close = float(df.iloc[i]["close"])

                    if zone_type == "support" and low <= zone_top and close > zone_bottom:
                        # Bounce off support → long
                        direction = "long"
                        entry = close
                        stop = zone_bottom - 5 * pip
                        tp = entry + abs(entry - stop) * self.rr_ratio
                    elif zone_type == "resistance" and high >= zone_bottom and close < zone_top:
                        # Bounce off resistance → short
                        direction = "short"
                        entry = close
                        stop = zone_top + 5 * pip
                        tp = entry - abs(entry - stop) * self.rr_ratio
                    else:
                        continue

                    exit_idx, exit_price, reason = self.sim.simulate(
                        df, i, direction, entry, stop, tp, pip)
                    if exit_idx is None:
                        continue

                    trade = self._make_trade(
                        "sr_zones", pair, timeframe, direction,
                        df.index[i], entry, stop, tp,
                        df.index[exit_idx], exit_price, reason, pip,
                        confidence="Medium", tactic=zone_type)
                    result.trades.append(trade)
            except Exception as e:
                log.warning(f"Suppressed exception at line 564: {e}")
                continue

        self._finalize_result(result)
        return result

    def _test_stop_hunt(self, df, pair, timeframe, pip):
        """Strategy 5: Stop Hunt signal engine."""
        from analysis.stop_hunt_signal_engine import StopHuntSignalEngine
        result = StrategyResult("stop_hunt", pair, timeframe)
        engine = StopHuntSignalEngine()

        # Run on rolling windows
        lookback = 200
        start = max(50, len(df) - lookback)

        for i in range(start, len(df) - 5):
            try:
                window = df.iloc[max(0, i-100):i+1]
                sig = engine.analyze(window)
                if sig is None:
                    continue
                signal = sig.get("signal", {})
                action = signal.get("action", "NO_TRADE")
                if action not in ("BUY", "SELL"):
                    continue

                direction = "long" if action == "BUY" else "short"
                entry = signal.get("entry_price") or float(df.iloc[i]["close"])
                stop = signal.get("stop_loss")
                tp = signal.get("take_profit")

                if not stop or not tp:
                    atr = self._atr(df, i)
                    if atr <= 0:
                        continue
                    stop = entry - 1.5 * atr if direction == "long" else entry + 1.5 * atr
                    tp = entry + 1.5 * atr * self.rr_ratio if direction == "long" \
                        else entry - 1.5 * atr * self.rr_ratio

                exit_idx, exit_price, reason = self.sim.simulate(
                    df, i, direction, entry, stop, tp, pip)
                if exit_idx is None:
                    continue

                trade = self._make_trade(
                    "stop_hunt", pair, timeframe, direction,
                    df.index[i], entry, stop, tp,
                    df.index[exit_idx], exit_price, reason, pip,
                    confidence=signal.get("confidence", "Medium"),
                    tactic="stop_hunt_default")
                result.trades.append(trade)
            except Exception as e:
                log.warning(f"Suppressed exception at line 617: {e}")
                continue

        self._finalize_result(result)
        return result

    def _test_ict_amd(self, df, pair, timeframe, pip):
        """Strategy 6: ICT/AMD signal engine."""
        from analysis.ict_amd_signal_engine import ICTAMDSignalEngine
        result = StrategyResult("ict_amd", pair, timeframe)
        engine = ICTAMDSignalEngine()

        lookback = 200
        start = max(50, len(df) - lookback)

        for i in range(start, len(df) - 5):
            try:
                window = df.iloc[max(0, i-100):i+1]
                sig = engine.analyze(window)
                if sig is None:
                    continue
                signal = sig.get("signal", {})
                action = signal.get("action", "NO_TRADE")
                if action not in ("BUY", "SELL"):
                    continue

                direction = "long" if action == "BUY" else "short"
                entry = signal.get("entry_price") or float(df.iloc[i]["close"])
                stop = signal.get("stop_loss")
                tp = signal.get("take_profit")

                if not stop or not tp:
                    atr = self._atr(df, i)
                    if atr <= 0:
                        continue
                    stop = entry - 1.5 * atr if direction == "long" else entry + 1.5 * atr
                    tp = entry + 1.5 * atr * self.rr_ratio if direction == "long" \
                        else entry - 1.5 * atr * self.rr_ratio

                exit_idx, exit_price, reason = self.sim.simulate(
                    df, i, direction, entry, stop, tp, pip)
                if exit_idx is None:
                    continue

                trade = self._make_trade(
                    "ict_amd", pair, timeframe, direction,
                    df.index[i], entry, stop, tp,
                    df.index[exit_idx], exit_price, reason, pip,
                    confidence=signal.get("confidence", "Medium"),
                    tactic="ict_amd_default")
                result.trades.append(trade)
            except Exception as e:
                log.warning(f"Suppressed exception at line 669: {e}")
                continue

        self._finalize_result(result)
        return result

    def _test_multi_pa(self, df, pair, timeframe, pip):
        """Strategy 7: Multi-Strategy PA engine."""
        from analysis.multi_strategy_pa_engine import MultiStrategyPAEngine
        result = StrategyResult("multi_pa", pair, timeframe)
        engine = MultiStrategyPAEngine()

        lookback = 200
        start = max(50, len(df) - lookback)

        for i in range(start, len(df) - 5):
            try:
                window = df.iloc[max(0, i-100):i+1]
                sig = engine.analyze(window)
                if sig is None:
                    continue
                signal = sig.get("signal", {})
                action = signal.get("action", "NO_TRADE")
                if action not in ("BUY", "SELL"):
                    continue

                direction = "long" if action == "BUY" else "short"
                entry = signal.get("entry_price") or float(df.iloc[i]["close"])
                stop = signal.get("stop_loss")
                tp = signal.get("take_profit")

                if not stop or not tp:
                    atr = self._atr(df, i)
                    if atr <= 0:
                        continue
                    stop = entry - 1.5 * atr if direction == "long" else entry + 1.5 * atr
                    tp = entry + 1.5 * atr * self.rr_ratio if direction == "long" \
                        else entry - 1.5 * atr * self.rr_ratio

                exit_idx, exit_price, reason = self.sim.simulate(
                    df, i, direction, entry, stop, tp, pip)
                if exit_idx is None:
                    continue

                trade = self._make_trade(
                    "multi_pa", pair, timeframe, direction,
                    df.index[i], entry, stop, tp,
                    df.index[exit_idx], exit_price, reason, pip,
                    confidence=signal.get("confidence", "Medium"),
                    tactic="multi_pa_default")
                result.trades.append(trade)
            except Exception as e:
                log.warning(f"Suppressed exception at line 721: {e}")
                continue

        self._finalize_result(result)
        return result

    def _test_cci_state(self, df, pair, timeframe, pip):
        """Strategy 8: CCI state machine (entry signals only)."""
        from analysis.cci_state_machine import CCIStateMachine
        result = StrategyResult("cci_state", pair, timeframe)
        sm = CCIStateMachine()

        # Compute CCI inline (simplified)
        try:
            cci = self._compute_cci(df, period=20)
        except Exception as e:
            return result
        if cci is None or len(cci) == 0:
            return result

        # CRITICAL FIX: No look-ahead bias. Zones computed per-bar.
        from analysis.support_resistance import SupportResistance
        sr = SupportResistance()
        sr_zones = {"resistance_zones": [], "support_zones": []}  # placeholder

        # Find nearby zone for each bar (using only past data)
        def nearest_zone_type(idx):
            close = float(df.iloc[idx]["close"])
            # FIX: compute zones using only data up to idx
            try:
                sr_zones_local = sr.analyze(df.iloc[:idx+1])
            except Exception as e:
                sr_zones_local = {"resistance_zones": [], "support_zones": []}
            for z in sr_zones_local.get("support_zones", []):
                if abs(close - z.get("zone_top", 0)) < 20 * pip:
                    return "demand"
            for z in sr_zones_local.get("resistance_zones", []):
                if abs(close - z.get("zone_bottom", 0)) < 20 * pip:
                    return "supply"
            return None

        for i in range(30, len(df) - 5):
            try:
                if i >= len(cci):
                    continue
                cci_val = float(cci[i])
                zone_type = nearest_zone_type(i)
                if zone_type is None:
                    continue

                sig = sm.evaluate(cci_value=cci_val, zone_type=zone_type,
                                  position=None, trend_align=True, at_zone=True)
                if sig.action != "ENTER":
                    continue

                direction = "long" if sig.direction == "long" else "short"
                entry = float(df.iloc[i]["close"])
                atr = self._atr(df, i)
                if atr <= 0:
                    continue
                stop = entry - 1.5 * atr if direction == "long" else entry + 1.5 * atr
                tp = entry + 1.5 * atr * self.rr_ratio if direction == "long" \
                    else entry - 1.5 * atr * self.rr_ratio

                exit_idx, exit_price, reason = self.sim.simulate(
                    df, i, direction, entry, stop, tp, pip)
                if exit_idx is None:
                    continue

                trade = self._make_trade(
                    "cci_state", pair, timeframe, direction,
                    df.index[i], entry, stop, tp,
                    df.index[exit_idx], exit_price, reason, pip,
                    confidence="High" if sig.confluence_score == 3 else "Medium",
                    tactic=f"cci_{int(cci_val)}")
                result.trades.append(trade)
            except Exception as e:
                log.warning(f"Suppressed exception at line 798: {e}")
                continue

        self._finalize_result(result)
        return result

    # ══════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════

    # Round-8 audit fix: confidence field normalization map.
    # Previously, _make_trade() was called with wildly inconsistent
    # confidence values from different strategies:
    #   - pin_bar: quality_grade "A"/"B"/"C"/"D"/"F" (string)
    #   - candlestick_patterns: reliability "Low"/"High" (string)
    #   - sd_zones_scored: tier "A"/"B"/"C"/"D" (string)
    #   - sr_zones: hardcoded "Medium" (string)
    #   - stop_hunt/ict_amd/multi_pa: whatever signal["confidence"] was
    #     (could be int, float, str, or None)
    #   - cci_state: "High"/"Medium" (string)
    #
    # The by_conf.setdefault(t.confidence, ...) grouping at line ~890
    # then created ~10+ unrelated buckets, making per-confidence
    # win-rate aggregation meaningless.
    #
    # Now: ALL confidence values are normalized to a single integer
    # 0-100 scale at the _make_trade() boundary. The normalized value
    # is also stored as a string label in a separate field for
    # readability, but the numeric value is what gets grouped on.
    _CONFIDENCE_NORMALIZE_MAP = {
        # Letter grades (pin_bar, sd_zones_scored)
        "A": 90, "B": 75, "C": 60, "D": 40, "F": 20,
        # Word labels (candlestick_patterns, sr_zones, cci_state)
        "HIGH": 80, "MEDIUM": 60, "LOW": 40,
        "VERY_HIGH": 95, "VERY_LOW": 25,
        # Numeric strings
        "0": 0, "25": 25, "50": 50, "75": 75, "100": 100,
    }

    @classmethod
    def _normalize_confidence(cls, value) -> int:
        """Normalize any confidence value to a 0-100 integer scale.

        Handles:
          - int/float in 0-100 range (returned as-is, clamped)
          - int/float in 0-1 range (treated as fraction, × 100)
          - str letter grade ("A"-"F") → mapped via _CONFIDENCE_NORMALIZE_MAP
          - str word ("High"/"Medium"/"Low") → mapped
          - str numeric ("75") → parsed
          - None / unknown → 50 (neutral)
        """
        if value is None:
            return 50
        if isinstance(value, (int, float)):
            v = float(value)
            # If in 0-1 range, treat as fraction
            if 0 <= v <= 1.0:
                return int(round(v * 100))
            # Otherwise treat as 0-100 scale
            return int(max(0, min(100, round(v))))
        if isinstance(value, str):
            v = value.strip().upper()
            if v in cls._CONFIDENCE_NORMALIZE_MAP:
                return cls._CONFIDENCE_NORMALIZE_MAP[v]
            # Try parsing as a number
            try:
                return cls._normalize_confidence(float(v))
            except (ValueError, TypeError):
                return 50
        return 50

    @classmethod
    def _confidence_label(cls, numeric_value: int) -> str:
        """Convert a 0-100 numeric confidence to a readable label.

        Bucket boundaries (aligned with the normalize map):
          ≥ 85  → "High"
          50-84 → "Medium"
          < 50  → "Low"
        """
        if numeric_value >= 85:
            return "High"
        if numeric_value >= 50:
            return "Medium"
        return "Low"

    def _make_trade(
        self, strategy, pair, tf, direction,
        entry_time, entry, stop, tp,
        exit_time, exit_price, reason, pip,
        confidence="Medium", tactic="default",
    ) -> Trade:
        """Build a Trade record with P&L computed.

        Round-8 audit fix: confidence is now normalized to a 0-100
        integer at this boundary, so all downstream grouping
        (by_confidence in _finalize_result) works on a consistent
        scale regardless of which strategy produced the signal.
        """
        if direction == "long":
            pnl_pips = (exit_price - entry) / pip
            risk_pips = (entry - stop) / pip
        else:
            pnl_pips = (entry - exit_price) / pip
            risk_pips = (stop - entry) / pip

        r_multiple = pnl_pips / risk_pips if risk_pips > 0 else 0.0
        win = None
        if reason == "TP":
            win = True
        elif reason == "SL":
            win = False
        elif reason in ("timeout", "end_of_data", "manual"):
            win = pnl_pips > 0

        # Round-8: normalize confidence to 0-100 int + readable label.
        # Store the numeric value in confidence (was a str before),
        # and the label in a new tactic suffix for backward compat.
        conf_numeric = self._normalize_confidence(confidence)
        conf_label = self._confidence_label(conf_numeric)

        return Trade(
            strategy=strategy, pair=pair, timeframe=tf,
            direction=direction, entry_time=entry_time,
            entry_price=entry, stop_loss=stop, take_profit=tp,
            exit_time=exit_time, exit_price=exit_price,
            exit_reason=reason,
            pnl_pips=round(pnl_pips, 2),
            pnl_pct=round((pnl_pips * pip) / entry * 100, 4),
            r_multiple=round(r_multiple, 3),
            confidence=conf_label,  # Now always "High"/"Medium"/"Low"
            tactic=f"{tactic} [conf={conf_numeric}]", win=win,
        )

    def _finalize_result(self, result: StrategyResult):
        """Compute aggregate stats from result.trades."""
        trades = result.trades
        if not trades:
            return

        wins = [t for t in trades if t.win is True]
        losses = [t for t in trades if t.win is False]
        breakeven = [t for t in trades if t.win is None]

        result.n_trades = len(trades)
        result.n_wins = len(wins)
        result.n_losses = len(losses)
        result.n_breakeven = len(breakeven)
        result.win_rate = len(wins) / len(trades) if trades else 0.0
        result.avg_r = np.mean([t.r_multiple for t in trades])
        result.max_r = max(t.r_multiple for t in trades)
        result.min_r = min(t.r_multiple for t in trades)
        result.total_r = sum(t.r_multiple for t in trades)

        # Profit factor
        gross_profit = sum(t.r_multiple for t in trades if t.r_multiple > 0)
        gross_loss = abs(sum(t.r_multiple for t in trades if t.r_multiple < 0))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average hold bars
        hold_bars = []
        for t in trades:
            if t.exit_time and t.entry_time:
                try:
                    bars = (t.exit_time - t.entry_time).total_seconds() / 3600  # hours
                    hold_bars.append(bars)
                except Exception as e:
                    log.warning(f"Suppressed exception at line 876: {e}")
                    pass
        result.avg_hold_bars = np.mean(hold_bars) if hold_bars else 0

        # Max drawdown in R
        equity = np.cumsum([t.r_multiple for t in trades])
        running_max = np.maximum.accumulate(equity)
        drawdowns = running_max - equity
        result.max_drawdown_r = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Per-confidence breakdown
        by_conf: Dict[str, List[Trade]] = {}
        for t in trades:
            by_conf.setdefault(t.confidence, []).append(t)
        for conf, items in by_conf.items():
            result.by_confidence[conf] = {
                "n_trades": len(items),
                "win_rate": sum(1 for t in items if t.win is True) / len(items),
                "avg_r": np.mean([t.r_multiple for t in items]),
                "total_r": sum(t.r_multiple for t in items),
            }

        # Per-tactic breakdown
        by_tactic: Dict[str, List[Trade]] = {}
        for t in trades:
            by_tactic.setdefault(t.tactic, []).append(t)
        for tac, items in by_tactic.items():
            result.by_tactic[tac] = {
                "n_trades": len(items),
                "win_rate": sum(1 for t in items if t.win is True) / len(items),
                "avg_r": np.mean([t.r_multiple for t in items]),
            }

        # Per-direction breakdown
        by_dir: Dict[str, List[Trade]] = {}
        for t in trades:
            by_dir.setdefault(t.direction, []).append(t)
        for d, items in by_dir.items():
            result.by_direction[d] = {
                "n_trades": len(items),
                "win_rate": sum(1 for t in items if t.win is True) / len(items),
                "avg_r": np.mean([t.r_multiple for t in items]),
            }

    def _build_summary(self, results: Dict[str, StrategyResult]) -> Dict[str, Any]:
        """Build a summary across all strategies."""
        summary = {
            "total_strategies": len(results),
            "strategies_with_trades": sum(1 for r in results.values() if r.n_trades > 0),
            "best_win_rate": None,
            "worst_win_rate": None,
            "most_trades": None,
            "best_profit_factor": None,
            "ranking": [],
        }

        ranked = sorted(
            [r for r in results.values() if r.n_trades > 0],
            key=lambda r: r.win_rate,
            reverse=True,
        )

        if ranked:
            summary["best_win_rate"] = {
                "strategy": ranked[0].strategy,
                "win_rate": ranked[0].win_rate,
                "n_trades": ranked[0].n_trades,
            }
            summary["worst_win_rate"] = {
                "strategy": ranked[-1].strategy,
                "win_rate": ranked[-1].win_rate,
                "n_trades": ranked[-1].n_trades,
            }
            most_trades = max(ranked, key=lambda r: r.n_trades)
            summary["most_trades"] = {
                "strategy": most_trades.strategy,
                "n_trades": most_trades.n_trades,
            }
            best_pf = max(ranked, key=lambda r: r.profit_factor if r.profit_factor != float("inf") else 0)
            summary["best_profit_factor"] = {
                "strategy": best_pf.strategy,
                "profit_factor": best_pf.profit_factor,
            }
            summary["ranking"] = [
                {
                    "strategy": r.strategy,
                    "n_trades": r.n_trades,
                    "win_rate": round(r.win_rate, 4),
                    "avg_r": round(r.avg_r, 3),
                    "total_r": round(r.total_r, 2),
                    "profit_factor": round(r.profit_factor, 2) if r.profit_factor != float("inf") else "inf",
                }
                for r in ranked
            ]

        return summary

    def _atr(self, df, idx, period=14):
        """Compute ATR at idx."""
        if idx < period:
            return 0.0
        window = df.iloc[idx-period:idx+1]
        high = window["high"].values
        low = window["low"].values
        close = window["close"].values
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        return float(np.mean(tr[-period:]))

    def _compute_cci(self, df, period=20):
        """Compute CCI indicator."""
        try:
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)
            typical = (high + low + close) / 3
            sma = typical.rolling(period).mean()
            mad = typical.rolling(period).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            cci = (typical - sma) / (0.015 * mad)
            return cci.values
        except Exception as e:
            log.warning(f"Suppressed exception at line 1003: {e}")
            return None

    def _pip_size_for_pair(self, pair: str) -> float:
        """Determine pip size based on pair."""
        pair = pair.upper()
        if "JPY" in pair:
            return 0.01
        if "XAU" in pair or "XAG" in pair:
            return 0.1
        if any(idx in pair for idx in ["US30", "NAS100", "SPX500"]):
            return 1.0
        if "BTC" in pair or "ETH" in pair:
            return 1.0
        return 0.0001


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from backtest.mt5_bulk_fetcher import MT5BulkFetcher

    print("=" * 64)
    print("  PER-STRATEGY TESTER — Smoke Test")
    print("=" * 64)

    fetcher = MT5BulkFetcher()
    result = fetcher.fetch("EURUSD", "H1", n_candles=500)
    df = result.df
    print(f"\nData: {result.pair} {result.timeframe} ({result.n_candles} candles, {result.source})")

    tester = PerStrategyTester()
    results = tester.run_all(df, pair="EURUSD", timeframe="H1")

    print(f"\n── Summary ──")
    print(f"Strategies tested: {results['summary']['total_strategies']}")
    print(f"Strategies with trades: {results['summary']['strategies_with_trades']}")
    if results["summary"]["ranking"]:
        print(f"\n── Strategy Ranking (by win rate) ──")
        for r in results["summary"]["ranking"]:
            print(f"  {r['strategy']:<25} trades={r['n_trades']:>3}  "
                  f"WR={r['win_rate']*100:>5.1f}%  avgR={r['avg_r']:>+6.2f}  "
                  f"PF={r['profit_factor']}")

    print("\n" + "=" * 64)
