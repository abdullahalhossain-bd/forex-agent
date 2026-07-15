"""
analysis/market_state_memory.py — Market State Memory & Adaptive Strategy
=============================================================================
Course Module — Hedge Fund Layer #4: Market State Memory

CONCEPT:
The AI remembers what's working and what's not in the CURRENT market
environment. If fake breakouts are common today, it adjusts. If
trend-following is winning, it increases trend strategy weight.

This is NOT the same as the general trade journal — it's a ROLLING
memory of recent market behavior that influences CURRENT decisions.

FEATURES:
1. Tracks recent trade outcomes by strategy type (last 20 trades)
2. Detects market behavior patterns (fake breakout frequency, etc.)
3. Adjusts strategy weights based on recent performance
4. Remembers "today's market character" (choppy, trending, etc.)

USAGE:
    from analysis.market_state_memory import MarketStateMemory
    msm = MarketStateMemory()
    msm.record_trade(strategy="BREAKOUT", result="LOSS", reason="fake_breakout")
    msm.record_trade(strategy="TREND_FOLLOW", result="WIN", reason="trend_continuation")

    # Get current market character
    character = msm.get_market_character()
    # → {"character": "FAKE_BREAKOUT_HEAVY", "trend_follow_weight": 0.3, ...}

    # Get strategy recommendation
    rec = msm.get_strategy_recommendation()
    # → {"best_strategy": "MEAN_REVERSION", "weight": 1.5, "reason": "..."}
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from utils.logger import get_logger

log = get_logger("market_state_memory")

MEMORY_PATH = Path("memory/market_state_memory.json")
MAX_RECENT_TRADES = 50  # rolling window


@dataclass
class MarketCharacter:
    """Current market character — what's the market doing RIGHT NOW."""
    character: str = "UNKNOWN"  # TRENDING / RANGING / CHOPPY / FAKE_BREAKOUT_HEAVY / VOLATILE / QUIET
    fake_breakout_rate: float = 0.0  # 0-1, how often recent breakouts were fake
    trend_follow_win_rate: float = 0.5  # recent trend-follow win rate
    mean_revert_win_rate: float = 0.5  # recent mean-reversion win rate
    breakout_win_rate: float = 0.5  # recent breakout win rate
    pullback_win_rate: float = 0.5  # recent pullback win rate
    best_strategy: str = "TREND_FOLLOW"
    worst_strategy: str = "UNKNOWN"
    strategy_weights: dict = field(default_factory=dict)
    confidence: float = 0.0  # 0-1, how confident we are in this assessment
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "character": self.character,
            "fake_breakout_rate": round(self.fake_breakout_rate, 2),
            "trend_follow_win_rate": round(self.trend_follow_win_rate, 2),
            "mean_revert_win_rate": round(self.mean_revert_win_rate, 2),
            "breakout_win_rate": round(self.breakout_win_rate, 2),
            "pullback_win_rate": round(self.pullback_win_rate, 2),
            "best_strategy": self.best_strategy,
            "worst_strategy": self.worst_strategy,
            "strategy_weights": self.strategy_weights,
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
        }


class MarketStateMemory:
    """Rolling memory of market behavior that influences current decisions."""

    def __init__(self):
        self._trades: List[dict] = []
        self._load()

    def _load(self):
        """Load recent trades from disk."""
        try:
            if MEMORY_PATH.exists():
                data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
                self._trades = data.get("trades", [])[-MAX_RECENT_TRADES:]
        except Exception as e:
            log.debug(f"[MarketStateMemory] load failed: {e}")
            self._trades = []

    def _save(self):
        """Persist to disk (atomic write)."""
        try:
            import tempfile
            MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "trades": self._trades[-MAX_RECENT_TRADES:],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            with tempfile.NamedTemporaryFile(
                mode="w", dir=str(MEMORY_PATH.parent),
                suffix=".tmp", prefix="msm_", delete=False
            ) as f:
                json.dump(data, f, indent=2, default=str)
                tmp = f.name
            import os
            os.replace(tmp, str(MEMORY_PATH))
        except Exception as e:
            log.warning(f"[MarketStateMemory] save failed: {e}")

    def record_trade(
        self,
        strategy: str,
        result: str,  # "WIN" / "LOSS" / "BREAKEVEN"
        reason: str = "",
        pair: str = "",
        pnl: float = 0.0,
    ):
        """Record a completed trade outcome."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy.upper(),
            "result": result.upper(),
            "reason": reason.lower(),
            "pair": pair,
            "pnl": pnl,
        }
        self._trades.append(entry)
        self._trades = self._trades[-MAX_RECENT_TRADES:]
        self._save()

        log.info(
            f"[MarketStateMemory] Recorded: {strategy} {result} "
            f"(reason={reason[:40]}, pnl={pnl:.2f}) — total tracked: {len(self._trades)}"
        )

    def get_market_character(self) -> MarketCharacter:
        """Analyze recent trades to determine current market character.

        Uses last 20 trades to determine:
        - Fake breakout frequency
        - Which strategies are winning/losing
        - Overall market character
        """
        char = MarketCharacter()

        if len(self._trades) < 5:
            char.reason = f"Only {len(self._trades)} trades — insufficient for market character"
            char.confidence = 0.1
            return char

        recent = self._trades[-20:]  # last 20 trades

        # ── Compute per-strategy win rates ──
        strategy_results = defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0})
        fake_breakout_count = 0
        breakout_total = 0

        for t in recent:
            strat = t["strategy"]
            strategy_results[strat]["total"] += 1
            if t["result"] == "WIN":
                strategy_results[strat]["wins"] += 1
            elif t["result"] == "LOSS":
                strategy_results[strat]["losses"] += 1

            # Track fake breakouts
            if "breakout" in strat.lower() or "BREAKOUT" in strat:
                breakout_total += 1
                if "fake" in t.get("reason", "") or t["result"] == "LOSS":
                    fake_breakout_count += 1

        # Compute win rates
        for strat, counts in strategy_results.items():
            wr = counts["wins"] / counts["total"] if counts["total"] > 0 else 0.5
            if "TREND" in strat:
                char.trend_follow_win_rate = wr
            if "MEAN" in strat or "RANGE" in strat:
                char.mean_revert_win_rate = wr
            if "BREAKOUT" in strat:
                char.breakout_win_rate = wr
            if "PULLBACK" in strat:
                char.pullback_win_rate = wr

        # Fake breakout rate
        if breakout_total > 0:
            char.fake_breakout_rate = fake_breakout_count / breakout_total

        # ── Determine market character ──
        # High fake breakout rate = choppy/manipulation market
        if char.fake_breakout_rate > 0.5 and breakout_total >= 3:
            char.character = "FAKE_BREAKOUT_HEAVY"
            char.reason = f"Fake breakout rate {char.fake_breakout_rate:.0%} — market is chopping, avoid breakouts"
        # Trend-following winning = trending market
        elif char.trend_follow_win_rate > 0.6:
            char.character = "TRENDING"
            char.reason = f"Trend-following win rate {char.trend_follow_win_rate:.0%} — trend market, favor trend strategies"
        # Mean reversion winning = ranging market
        elif char.mean_revert_win_rate > 0.6:
            char.character = "RANGING"
            char.reason = f"Mean reversion win rate {char.mean_revert_win_rate:.0%} — range market, favor mean reversion"
        # Both losing = choppy/difficult
        elif char.trend_follow_win_rate < 0.35 and char.mean_revert_win_rate < 0.35:
            char.character = "CHOPPY"
            char.reason = "Both trend and mean reversion losing — choppy market, reduce trading"
        else:
            char.character = "MIXED"
            char.reason = "No clear edge in either direction — standard approach"

        # ── Compute strategy weights ──
        # Weight = win_rate × 2 (so 50% WR = 1.0, 75% WR = 1.5, 25% WR = 0.5)
        # Clamped to 0.2 - 2.0
        for strat, counts in strategy_results.items():
            wr = counts["wins"] / counts["total"] if counts["total"] > 0 else 0.5
            weight = max(0.2, min(wr * 2.0, 2.0))
            char.strategy_weights[strat] = round(weight, 2)

        # ── Best/worst strategy ──
        if strategy_results:
            best_strat = max(strategy_results.items(),
                           key=lambda x: x[1]["wins"] / max(x[1]["total"], 1))
            worst_strat = min(strategy_results.items(),
                            key=lambda x: x[1]["wins"] / max(x[1]["total"], 1))
            char.best_strategy = best_strat[0]
            char.worst_strategy = worst_strat[0]

        # Confidence based on sample size
        char.confidence = min(len(recent) / 20.0, 1.0)

        log.info(
            f"[MarketStateMemory] Character: {char.character} | "
            f"trend_WR={char.trend_follow_win_rate:.0%} mean_WR={char.mean_revert_win_rate:.0%} "
            f"breakout_WR={char.breakout_win_rate:.0%} fake_rate={char.fake_breakout_rate:.0%} | "
            f"best={char.best_strategy} worst={char.worst_strategy} | "
            f"conf={char.confidence:.0%}"
        )

        return char

    def get_strategy_recommendation(self) -> dict:
        """Get strategy recommendation based on recent market behavior.

        Returns:
            {
                "best_strategy": str,
                "weight": float,  # position size multiplier
                "avoid": list,    # strategies to avoid
                "reason": str,
            }
        """
        char = self.get_market_character()

        if char.confidence < 0.25:
            return {
                "best_strategy": "WAIT",
                "weight": 1.0,
                "avoid": [],
                "reason": f"Insufficient data ({char.confidence:.0%} confidence) — default approach",
            }

        avoid = []
        reason = char.reason

        # If fake breakouts are common, avoid breakout strategies
        if char.fake_breakout_rate > 0.5:
            avoid.append("BREAKOUT")
            avoid.append("RETEST")

        # If trend-following is losing badly, avoid it
        if char.trend_follow_win_rate < 0.35:
            avoid.append("TREND_FOLLOW")
            avoid.append("PULLBACK")

        # If mean reversion is losing badly, avoid it
        if char.mean_revert_win_rate < 0.35:
            avoid.append("MEAN_REVERSION")
            avoid.append("RANGE")

        # Best strategy weight
        weight = char.strategy_weights.get(char.best_strategy, 1.0)

        return {
            "best_strategy": char.best_strategy,
            "weight": weight,
            "avoid": avoid,
            "reason": reason,
            "character": char.character,
        }

    def should_skip_trading(self) -> tuple[bool, str]:
        """Check if the bot should skip trading entirely based on recent performance.

        Returns:
            (should_skip: bool, reason: str)
        """
        if len(self._trades) < 10:
            return False, "Not enough data to evaluate"

        recent = self._trades[-10:]
        wins = sum(1 for t in recent if t["result"] == "WIN")
        losses = sum(1 for t in recent if t["result"] == "LOSS")
        win_rate = wins / len(recent) if recent else 0.5

        if win_rate < 0.2 and losses >= 7:
            return True, (
                f"Win rate {win_rate:.0%} over last {len(recent)} trades "
                f"({wins}W/{losses}L) — market conditions unfavorable, skip trading"
            )

        return False, f"Win rate {win_rate:.0%} — trading allowed"


# Singleton
_msm: Optional[MarketStateMemory] = None


def get_market_state_memory() -> MarketStateMemory:
    global _msm
    if _msm is None:
        _msm = MarketStateMemory()
    return _msm


# ── Smoke test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    msm = MarketStateMemory()

    # Simulate trades — fake breakout heavy market
    for i in range(10):
        msm.record_trade("BREAKOUT", "LOSS", "fake_breakout", "EURUSD", -50)
    for i in range(5):
        msm.record_trade("MEAN_REVERSION", "WIN", "range_reversal", "EURUSD", 30)
    for i in range(3):
        msm.record_trade("TREND_FOLLOW", "LOSS", "counter_trend", "EURUSD", -40)

    char = msm.get_market_character()
    print(f"Character: {char.character}")
    print(f"Fake breakout rate: {char.fake_breakout_rate:.0%}")
    print(f"Trend WR: {char.trend_follow_win_rate:.0%}")
    print(f"Mean rev WR: {char.mean_revert_win_rate:.0%}")
    print(f"Breakout WR: {char.breakout_win_rate:.0%}")
    print(f"Best strategy: {char.best_strategy}")
    print(f"Reason: {char.reason}")

    rec = msm.get_strategy_recommendation()
    print(f"\nRecommendation: {rec}")

    skip, reason = msm.should_skip_trading()
    print(f"\nSkip trading: {skip} — {reason}")

    print("\nMarket state memory smoke test passed.")
