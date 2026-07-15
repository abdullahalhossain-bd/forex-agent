"""
analysis/adaptive_decision_engine.py — Adaptive Decision Engine
================================================================

Solves the "multiple mandatory strategies = no trades" problem by
replacing the all-or-nothing consensus gate with a soft confluence
scoring system that learns from backtest results.

Key principles:
  1. NO mandatory multi-strategy requirement — each strategy can trade alone
  2. Strategies are weighted by their HISTORICAL win rate (not arbitrary weights)
  3. Confidence levels (High/Medium/Low) are calibrated against backtest data
  4. If a strategy has ≥50% win rate, it can trade independently
  5. Confluence bonuses are added when multiple strategies agree, but NOT required

Three modes:
  • "single"     — any single strategy with sufficient win rate can trade
  • "confluence" — soft confluence scoring (default; bonus for agreement)
  • "strict"     — require 2+ strategies (legacy mode, for comparison)

Usage:
    from analysis.adaptive_decision_engine import AdaptiveDecisionEngine
    engine = AdaptiveDecisionEngine(mode="confluence")
    engine.load_backtest_results("backtest_results.json")
    decision = engine.decide(signals=[...], current_price=1.0850)
    # → {"action": "BUY", "confidence": "High", "score": 8.5, "reason": ...}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

log = get_logger("adaptive_decision")


# ════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════

# Minimum win rate for a strategy to trade alone (in "single" mode)
MIN_WIN_RATE_SOLO = 0.45

# Minimum win rate for a strategy to participate in confluence
MIN_WIN_RATE_CONF = 0.35

# Default weights (overridden by backtest results)
DEFAULT_STRATEGY_WEIGHTS = {
    "pin_bar":               1.5,
    "candlestick_patterns":  1.0,
    "sd_zones_scored":       2.0,
    "sr_zones":              1.0,
    "stop_hunt":             2.0,
    "ict_amd":               3.0,
    "multi_pa":              1.5,
    "cci_state":             1.0,
}

# Confidence tier thresholds (confluence score → confidence)
CONFIDENCE_HIGH_MIN = 4.0
CONFIDENCE_MED_MIN = 2.0
# Below MED_MIN → Low confidence (still tradeable in "single" mode)


# ════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

@dataclass
class StrategySignal:
    """A signal from one strategy."""
    strategy: str               # e.g., "pin_bar"
    action: str                 # "BUY" | "SELL" | "NO_TRADE" | "WAIT"
    confidence: str = "Medium"  # "High" | "Medium" | "Low"
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    r_multiple: float = 2.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """Final trading decision."""
    action: str                 # "BUY" | "SELL" | "NO_TRADE" | "WAIT"
    confidence: str             # "High" | "Medium" | "Low"
    score: float                # confluence score
    reason: str
    agreeing_strategies: List[str] = field(default_factory=list)
    disagreeing_strategies: List[str] = field(default_factory=list)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    mode: str = "confluence"    # which mode produced this decision
    weighted_by: Dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyStats:
    """Statistics for one strategy (loaded from backtest)."""
    name: str
    win_rate: float = 0.0
    n_trades: int = 0
    avg_r: float = 0.0
    profit_factor: float = 0.0
    by_confidence: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_tactic: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    weight: float = 1.0  # derived from win rate


# ════════════════════════════════════════════════════════════════
#  ADAPTIVE DECISION ENGINE
# ════════════════════════════════════════════════════════════════

class AdaptiveDecisionEngine:
    """
    Adaptive decision engine that learns from backtest results.

    Replaces the rigid "all strategies must agree" gate with a soft
    confluence scoring system. Each strategy's vote is weighted by
    its historical win rate.

    Modes:
      • "single"     — any strategy with win_rate ≥ MIN_WIN_RATE_SOLO
                       can trade alone (no consensus required)
      • "confluence" — soft scoring; agreement adds bonus but isn't
                       required (default, recommended)
      • "strict"     — require 2+ agreeing strategies (legacy)
    """

    def __init__(
        self,
        mode: str = "confluence",
        strategy_weights: Optional[Dict[str, float]] = None,
    ):
        self.mode = mode
        self.strategy_weights = strategy_weights or DEFAULT_STRATEGY_WEIGHTS.copy()
        self.stats: Dict[str, StrategyStats] = {}
        self._calibrated = False

    # ══════════════════════════════════════════════════════════
    #  CALIBRATION — load backtest results
    # ══════════════════════════════════════════════════════════

    def load_backtest_results(self, results: Dict[str, Any]) -> int:
        """
        Load backtest results to calibrate strategy weights.

        Args:
            results: dict from per_strategy_tester (can be aggregated
                     across multiple pairs/timeframes)

        Returns:
            Number of strategies calibrated
        """
        count = 0

        # Handle aggregated results
        if "strategies" in results:
            strategies_dict = results["strategies"]
        elif "by_strategy" in results:
            strategies_dict = results["by_strategy"]
        else:
            strategies_dict = results

        for name, data in strategies_dict.items():
            if not isinstance(data, dict):
                continue
            stats = StrategyStats(
                name=name,
                win_rate=data.get("win_rate", 0.0),
                n_trades=data.get("n_trades", 0),
                avg_r=data.get("avg_r", 0.0),
                profit_factor=data.get("profit_factor", 0.0),
                by_confidence=data.get("by_confidence", {}),
                by_tactic=data.get("by_tactic", {}),
            )
            # Derive weight from win rate (higher WR = higher weight)
            # Weight = base_weight × (win_rate / 0.5)
            base = self.strategy_weights.get(name, 1.0)
            if stats.win_rate > 0 and stats.n_trades >= 5:
                # Calibrated weight
                stats.weight = base * (stats.win_rate / 0.5)
            else:
                # Insufficient data — keep default weight
                stats.weight = base

            self.stats[name] = stats
            count += 1

        self._calibrated = count > 0
        log.info(
            f"[Adaptive] Calibrated {count} strategies from backtest "
            f"(mode={self.mode})"
        )
        return count

    def load_from_file(self, path: str) -> int:
        """Load backtest results from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.load_backtest_results(data)

    # ══════════════════════════════════════════════════════════
    #  DECISION — main entry point
    # ══════════════════════════════════════════════════════════

    def decide(
        self,
        signals: List[StrategySignal],
        current_price: Optional[float] = None,
    ) -> Decision:
        """
        Make a trading decision from a list of strategy signals.

        Args:
            signals       : list of StrategySignal objects (one per strategy)
            current_price : latest close price

        Returns:
            Decision object
        """
        if not signals:
            return Decision(
                action="NO_TRADE", confidence="Low", score=0.0,
                reason="No signals provided", mode=self.mode,
            )

        # Separate actionable signals (BUY/SELL) from abstentions
        actionable = [s for s in signals if s.action in ("BUY", "SELL")]
        abstentions = [s for s in signals if s.action not in ("BUY", "SELL")]

        # ── Screen out strategies with a PROVEN poor edge ───────────
        # BUG FIXED: previously, a strategy with a backtested win rate below
        # MIN_WIN_RATE_CONF was only blocked from trading in the "single"
        # and "confluence" paths when it was the ONLY agreeing strategy.
        # If two or more such underperforming strategies agreed with each
        # other, there was no gate at all — they could combine to produce a
        # BUY/SELL decision (even "High" confidence, if enough of them
        # piled on) despite backtests showing each one loses money more
        # often than a coin flip. A strategy that has demonstrated a losing
        # edge shouldn't get a vote at all, whether it acts alone or in a
        # group. This filter applies before scoring, for every mode.
        screened_out = []
        actionable_filtered = []
        for sig in actionable:
            stats = self.stats.get(sig.strategy)
            if stats and stats.n_trades >= 5 and stats.win_rate < MIN_WIN_RATE_CONF:
                screened_out.append(sig)
            else:
                actionable_filtered.append(sig)

        if screened_out:
            log.info(
                "[Adaptive] Screened out %d signal(s) from strategies with "
                "backtested win rate below %.0f%%: %s",
                len(screened_out), MIN_WIN_RATE_CONF * 100,
                ", ".join(f"{s.strategy}({self.stats[s.strategy].win_rate*100:.1f}%)"
                          for s in screened_out),
            )
        actionable = actionable_filtered
        abstentions = abstentions + screened_out

        if not actionable:
            return Decision(
                action="NO_TRADE", confidence="Low", score=0.0,
                reason=f"All {len(signals)} strategies abstained "
                       f"({', '.join(s.strategy for s in abstentions)})",
                mode=self.mode,
            )

        # Tally BUY vs SELL scores
        buy_score, sell_score = 0.0, 0.0
        buy_strategies, sell_strategies = [], []
        weighted_by = {}

        for sig in actionable:
            weight = self._get_weight(sig.strategy, sig.confidence)
            weighted_by[sig.strategy] = weight

            if sig.action == "BUY":
                buy_score += weight
                buy_strategies.append(sig.strategy)
            else:
                sell_score += weight
                sell_strategies.append(sig.strategy)

        # Determine direction
        if buy_score > sell_score:
            direction = "BUY"
            score = buy_score
            agreeing = buy_strategies
            disagreeing = sell_strategies
        elif sell_score > buy_score:
            direction = "SELL"
            score = sell_score
            agreeing = sell_strategies
            disagreeing = buy_strategies
        else:
            return Decision(
                action="NO_TRADE", confidence="Low", score=buy_score,
                reason=f"Tie: BUY={buy_score:.2f}, SELL={sell_score:.2f}",
                agreeing_strategies=buy_strategies + sell_strategies,
                weighted_by=weighted_by, mode=self.mode,
            )

        # ── Apply mode-specific gating ─────────────────────
        if self.mode == "single":
            result = self._decide_single(direction, score, agreeing,
                                         disagreeing, actionable, current_price,
                                         weighted_by)
        elif self.mode == "strict":
            result = self._decide_strict(direction, score, agreeing,
                                         disagreeing, actionable, current_price,
                                         weighted_by)
        else:  # "confluence" (default)
            result = self._decide_confluence(direction, score, agreeing,
                                             disagreeing, actionable,
                                             current_price, weighted_by)

        return self._flag_stale_price(result, current_price)

    # ── Staleness guard ──────────────────────────────────────────
    # `current_price` was accepted as a parameter but never actually used
    # anywhere in the original decision logic — entry/SL/TP were pulled
    # straight from whichever upstream signal had the best R:R, with no
    # check that its entry price still matches the market. If an upstream
    # strategy's signal was computed a few bars ago (e.g. a stale cache, a
    # slow indicator pipeline, or a signal queued during a data gap), this
    # engine would happily approve the trade at a stale price. This does
    # not block the trade — flagging is deliberately conservative since we
    # don't have ATR/volatility context here to size a hard cutoff safely
    # — but it surfaces the discrepancy in `reason` so a human or an
    # upstream execution layer can apply its own tolerance before sending
    # the order.
    STALE_PRICE_WARN_PCT = 0.005  # 0.5% — deliberately wide; see note above

    def _flag_stale_price(self, result: "Decision", current_price: Optional[float]) -> "Decision":
        if (
            result.action in ("BUY", "SELL")
            and current_price
            and result.entry_price
            and current_price > 0
        ):
            drift = abs(result.entry_price - current_price) / current_price
            if drift > self.STALE_PRICE_WARN_PCT:
                result.reason += (
                    f" | WARNING: signal entry_price {result.entry_price} differs "
                    f"from current_price {current_price} by {drift*100:.2f}% "
                    f"(> {self.STALE_PRICE_WARN_PCT*100:.1f}%) — verify signal freshness "
                    f"before execution"
                )
        return result

    # ══════════════════════════════════════════════════════════
    #  MODE-SPECIFIC DECISION LOGIC
    # ══════════════════════════════════════════════════════════

    def _decide_single(
        self, direction, score, agreeing, disagreeing,
        actionable, current_price, weighted_by,
    ) -> Decision:
        """Mode 'single': any one strategy can trade alone if WR ≥ threshold."""
        # Find the strongest agreeing strategy
        strongest = max(actionable, key=lambda s: self._get_weight(s.strategy, s.confidence))
        stats = self.stats.get(strongest.strategy)

        # Check if this strategy can trade solo
        wr = stats.win_rate if stats else 0.0
        n = stats.n_trades if stats else 0

        # Allow solo trade if:
        #   (a) Strategy has ≥ MIN_WIN_RATE_SOLO win rate with ≥5 trades, OR
        #   (b) Strategy has no backtest data (uncalibrated — give benefit of doubt)
        can_trade_solo = (n < 5) or (wr >= MIN_WIN_RATE_SOLO)
        if not can_trade_solo:
            # Win rate too low for solo trading
            return Decision(
                action="NO_TRADE", confidence="Low", score=score,
                reason=f"Single mode: {strongest.strategy} WR={wr*100:.1f}% "
                       f"< {MIN_WIN_RATE_SOLO*100:.0f}% threshold (n={n})",
                agreeing_strategies=agreeing,
                disagreeing_strategies=disagreeing,
                weighted_by=weighted_by, mode=self.mode,
            )

        # Solo trade allowed
        confidence = self._confidence_from_score(score)
        entry, stop, tp = self._extract_levels(actionable, direction)

        return Decision(
            action=direction, confidence=confidence, score=score,
            reason=f"Single mode: {strongest.strategy} (WR={wr*100:.1f}%, "
                   f"n={n}) trading solo — score {score:.2f}",
            agreeing_strategies=agreeing,
            disagreeing_strategies=disagreeing,
            entry_price=entry, stop_loss=stop, take_profit=tp,
            weighted_by=weighted_by, mode=self.mode,
        )

    def _decide_confluence(
        self, direction, score, agreeing, disagreeing,
        actionable, current_price, weighted_by,
    ) -> Decision:
        """Mode 'confluence': soft scoring (default). Trade if score ≥ threshold."""
        # In confluence mode, we trade as long as score ≥ MIN_WIN_RATE_CONF-derived threshold
        # The score itself is the gate

        confidence = self._confidence_from_score(score)
        entry, stop, tp = self._extract_levels(actionable, direction)

        # Single strategy with low score — check if it can trade solo
        if len(agreeing) == 1:
            strat = agreeing[0]
            stats = self.stats.get(strat)
            wr = stats.win_rate if stats else 0.0
            n = stats.n_trades if stats else 0

            # Allow if uncalibrated (n<5) or has decent WR
            can_trade_solo = (n < 5) or (wr >= MIN_WIN_RATE_SOLO)
            if not can_trade_solo:
                return Decision(
                    action="NO_TRADE", confidence="Low", score=score,
                    reason=f"Confluence: {strat} alone has WR={wr*100:.1f}% "
                           f"< solo threshold {MIN_WIN_RATE_SOLO*100:.0f}% (n={n})",
                    agreeing_strategies=agreeing,
                    disagreeing_strategies=disagreeing,
                    weighted_by=weighted_by, mode=self.mode,
                )

        # Trade allowed
        n_agree = len(agreeing)
        n_disagree = len(disagreeing)
        return Decision(
            action=direction, confidence=confidence, score=score,
            reason=f"Confluence: {n_agree} agreeing, {n_disagree} disagreeing "
                   f"({', '.join(agreeing)}) — score {score:.2f}",
            agreeing_strategies=agreeing,
            disagreeing_strategies=disagreeing,
            entry_price=entry, stop_loss=stop, take_profit=tp,
            weighted_by=weighted_by, mode=self.mode,
        )

    def _decide_strict(
        self, direction, score, agreeing, disagreeing,
        actionable, current_price, weighted_by,
    ) -> Decision:
        """Mode 'strict': require 2+ agreeing strategies (legacy)."""
        if len(agreeing) < 2:
            return Decision(
                action="NO_TRADE", confidence="Low", score=score,
                reason=f"Strict mode: only {len(agreeing)} strategy agrees "
                       f"(need ≥2) — {', '.join(agreeing)}",
                agreeing_strategies=agreeing,
                disagreeing_strategies=disagreeing,
                weighted_by=weighted_by, mode=self.mode,
            )

        confidence = self._confidence_from_score(score)
        entry, stop, tp = self._extract_levels(actionable, direction)

        return Decision(
            action=direction, confidence=confidence, score=score,
            reason=f"Strict mode: {len(agreeing)} strategies agree "
                   f"({', '.join(agreeing)}) — score {score:.2f}",
            agreeing_strategies=agreeing,
            disagreeing_strategies=disagreeing,
            entry_price=entry, stop_loss=stop, take_profit=tp,
            weighted_by=weighted_by, mode=self.mode,
        )

    # ══════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════

    def _get_weight(self, strategy: str, confidence: str) -> float:
        """Get effective weight: base × confidence_mult × win_rate_mult."""
        base = self.strategy_weights.get(strategy, 1.0)
        # Confidence multiplier
        conf_mult = {"High": 1.5, "Medium": 1.0, "Low": 0.5}.get(confidence, 1.0)
        # Win-rate multiplier (from calibration)
        stats = self.stats.get(strategy)
        if stats and stats.n_trades >= 5 and stats.win_rate > 0:
            wr_mult = stats.win_rate / 0.5  # 50% WR → 1.0× multiplier
        else:
            wr_mult = 1.0
        return base * conf_mult * wr_mult

    @staticmethod
    def _confidence_from_score(score: float) -> str:
        """Map confluence score to confidence tier."""
        if score >= CONFIDENCE_HIGH_MIN:
            return "High"
        if score >= CONFIDENCE_MED_MIN:
            return "Medium"
        return "Low"

    @staticmethod
    def _extract_levels(
        actionable: List[StrategySignal], direction: str,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Extract entry/SL/TP from the strongest agreeing signal."""
        # Use the signal with highest R:R or first BUY/SELL in direction
        candidates = [s for s in actionable if s.action == direction]
        if not candidates:
            return None, None, None
        # Pick the one with the best r_multiple
        best = max(candidates, key=lambda s: s.r_multiple)
        return best.entry_price, best.stop_loss, best.take_profit

    # ══════════════════════════════════════════════════════════
    #  EXPORT — save calibrated weights for the live trading system
    # ══════════════════════════════════════════════════════════

    def export_weights(self, path: str) -> None:
        """Export calibrated weights to JSON for the live trading system."""
        output = {
            "mode": self.mode,
            "calibrated": self._calibrated,
            "min_win_rate_solo": MIN_WIN_RATE_SOLO,
            "min_win_rate_confluence": MIN_WIN_RATE_CONF,
            "confidence_thresholds": {
                "high": CONFIDENCE_HIGH_MIN,
                "medium": CONFIDENCE_MED_MIN,
            },
            "strategies": {},
        }
        for name, stats in self.stats.items():
            output["strategies"][name] = {
                "win_rate": round(stats.win_rate, 4),
                "n_trades": stats.n_trades,
                "avg_r": round(stats.avg_r, 3),
                "profit_factor": round(stats.profit_factor, 2),
                "calibrated_weight": round(stats.weight, 3),
                "by_confidence": stats.by_confidence,
                "by_tactic": stats.by_tactic,
            }
        # Also include uncalibrated defaults
        for name, weight in DEFAULT_STRATEGY_WEIGHTS.items():
            if name not in output["strategies"]:
                output["strategies"][name] = {
                    "win_rate": 0.0,
                    "n_trades": 0,
                    "avg_r": 0.0,
                    "profit_factor": 0.0,
                    "calibrated_weight": weight,
                    "note": "No backtest data — using default weight",
                }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        log.info(f"[Adaptive] Weights exported to {path}")


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("=" * 64)
    print("  ADAPTIVE DECISION ENGINE — Smoke Test")
    print("=" * 64)

    # Simulate backtest results
    fake_backtest = {
        "strategies": {
            "pin_bar":              {"win_rate": 0.55, "n_trades": 23, "avg_r": 0.4, "profit_factor": 1.8},
            "candlestick_patterns": {"win_rate": 0.52, "n_trades": 229, "avg_r": 0.5, "profit_factor": 2.2},
            "sd_zones_scored":      {"win_rate": 0.48, "n_trades": 15, "avg_r": 0.3, "profit_factor": 1.4},
            "sr_zones":             {"win_rate": 0.44, "n_trades": 77, "avg_r": 0.3, "profit_factor": 1.6},
            "stop_hunt":            {"win_rate": 0.60, "n_trades": 12, "avg_r": 0.6, "profit_factor": 2.5},
            "ict_amd":              {"win_rate": 0.58, "n_trades": 8, "avg_r": 0.7, "profit_factor": 2.8},
        }
    }

    for mode in ["single", "confluence", "strict"]:
        print(f"\n── Mode: {mode} ──")
        engine = AdaptiveDecisionEngine(mode=mode)
        engine.load_backtest_results(fake_backtest)

        # Test 1: single strategy signal
        signals1 = [
            StrategySignal("pin_bar", "BUY", "Medium", 1.0850, 1.0820, 1.0910),
        ]
        d1 = engine.decide(signals1, current_price=1.0850)
        print(f"  Single pin_bar BUY → {d1.action} ({d1.confidence}, score={d1.score:.2f})")

        # Test 2: 2 strategies agreeing
        signals2 = [
            StrategySignal("pin_bar", "BUY", "Medium", 1.0850, 1.0820, 1.0910),
            StrategySignal("stop_hunt", "BUY", "High", 1.0850, 1.0825, 1.0905),
        ]
        d2 = engine.decide(signals2, current_price=1.0850)
        print(f"  2 strategies agree → {d2.action} ({d2.confidence}, score={d2.score:.2f})")

        # Test 3: 3 strategies agreeing
        signals3 = [
            StrategySignal("pin_bar", "BUY", "Medium", 1.0850, 1.0820, 1.0910),
            StrategySignal("stop_hunt", "BUY", "High", 1.0850, 1.0825, 1.0905),
            StrategySignal("candlestick_patterns", "BUY", "High", 1.0850, 1.0830, 1.0895),
        ]
        d3 = engine.decide(signals3, current_price=1.0850)
        print(f"  3 strategies agree → {d3.action} ({d3.confidence}, score={d3.score:.2f})")

        # Test 4: disagreement
        signals4 = [
            StrategySignal("pin_bar", "BUY", "Medium", 1.0850, 1.0820, 1.0910),
            StrategySignal("ict_amd", "SELL", "High", 1.0850, 1.0880, 1.0790),
        ]
        d4 = engine.decide(signals4, current_price=1.0850)
        print(f"  Disagreement      → {d4.action} ({d4.confidence}, score={d4.score:.2f})")

    # ── Regression check for the multi-weak-strategy confluence bug ──
    # Two strategies with backtested win rates BELOW the confluence
    # threshold (0.35) should NOT be able to combine into a trade just
    # because they agree with each other.
    print("\n── Regression check: weak strategies should not combine into a trade ──")
    weak_backtest = {
        "strategies": {
            "sr_zones":  {"win_rate": 0.30, "n_trades": 40, "avg_r": 0.1, "profit_factor": 0.9},
            "multi_pa":  {"win_rate": 0.28, "n_trades": 35, "avg_r": 0.1, "profit_factor": 0.8},
        }
    }
    weak_engine = AdaptiveDecisionEngine(mode="confluence")
    weak_engine.load_backtest_results(weak_backtest)
    weak_signals = [
        StrategySignal("sr_zones", "BUY", "High", 1.0850, 1.0820, 1.0910),
        StrategySignal("multi_pa", "BUY", "High", 1.0850, 1.0825, 1.0905),
    ]
    d_weak = weak_engine.decide(weak_signals, current_price=1.0850)
    print(f"  2 backtested-losing strategies agree → {d_weak.action} "
          f"({d_weak.confidence}, score={d_weak.score:.2f})")
    print(f"  reason: {d_weak.reason}")
    assert d_weak.action == "NO_TRADE", "REGRESSION: weak strategies combined into a trade"

    # ── Stale price flag demo ──
    print("\n── Stale price flag demo ──")
    stale_engine = AdaptiveDecisionEngine(mode="single")
    stale_engine.load_backtest_results(fake_backtest)
    d_stale = stale_engine.decide(
        [StrategySignal("ict_amd", "BUY", "High", 1.0850, 1.0800, 1.0950)],
        current_price=1.1200,  # far from the signal's entry_price
    )
    print(f"  action={d_stale.action}  reason={d_stale.reason}")
    assert "WARNING" in d_stale.reason

    # Export weights
    engine.export_weights("/tmp/adaptive_weights.json")
    print(f"\n  Weights exported to /tmp/adaptive_weights.json")

    print("\n" + "=" * 64)