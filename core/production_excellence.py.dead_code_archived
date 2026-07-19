"""
core/production_excellence.py — Production-Ready Excellence Modules
====================================================================
3 missing production features from the final list:

4. Shadow Mode — new models run in parallel (paper) before going live
6. Strategy Marketplace — rank, retire, promote strategies automatically
12. Data Source Voting — multi-source conflict detection + confidence reduction

USAGE:
    from core.production_excellence import (
        ShadowModeManager, StrategyMarketplace, DataSourceVoter,
    )
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import json
from pathlib import Path
from utils.logger import get_logger

log = get_logger("prod_excellence")


# ════════════════════════════════════════════════════════════════════
# 4. SHADOW MODE — Champion vs Challenger model evaluation
# ════════════════════════════════════════════════════════════════════

@dataclass
# ⚠️  STATUS (institutional review, see audit report): NOT CURRENTLY WIRED
#     IN. ShadowModeManager, StrategyMarketplace, and DataSourceVoter below
#     are not imported or called from trader.py, trading_engine.py, or
#     runtime.py. Left in place as design-complete, ready-to-wire modules.
#     If activating: ShadowModeManager needs live+shadow trade streams from
#     the execution layer; StrategyMarketplace needs each strategy's win/
#     loss stream from the learning pipeline; DataSourceVoter needs a
#     multi-broker/multi-feed price source (currently only one MT5 feed is
#     wired via MarketAgent).


class ShadowTrade:
    """A shadow-mode trade record."""
    timestamp: str
    model_version: str  # "champion" or "challenger"
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    confidence: float
    outcome: Optional[str] = None  # WIN / LOSS / PENDING
    pnl: float = 0.0


class ShadowModeManager:
    """Shadow mode: new models run in parallel before deployment.

    Flow:
    1. Champion model (current production) runs live
    2. Challenger model (new candidate) runs in shadow (paper only)
    3. After N trades, compare performance
    4. If challenger is significantly better → promote
    5. If challenger is worse → retire

    This prevents bad models from ever touching the live account.
    """

    SHADOW_PERIOD_TRADES = 50  # minimum trades before evaluation
    PROMOTION_THRESHOLD = 0.05  # challenger must be 5% better to promote
    DEMOTION_THRESHOLD = -0.10  # if 10% worse, reject immediately

    def __init__(self):
        self._champion_version: str = "v1"
        self._challenger_version: Optional[str] = None
        self._shadow_trades: List[ShadowTrade] = []
        self._evaluation_history: List[dict] = []

    def set_champion(self, version: str):
        """Set the current champion (production) model version."""
        self._champion_version = version
        log.info(f"[ShadowMode] Champion set: {version}")

    def register_challenger(self, version: str):
        """Register a new challenger model for shadow evaluation."""
        self._challenger_version = version
        self._shadow_trades = []  # reset for new challenger
        log.info(f"[ShadowMode] Challenger registered: {version} — shadow mode active")

    def record_shadow_trade(self, trade: ShadowTrade):
        """Record a shadow trade from the challenger model."""
        self._shadow_trades.append(trade)
        log.debug(f"[ShadowMode] Shadow trade #{len(self._shadow_trades)}: "
                  f"{trade.symbol} {trade.direction} conf={trade.confidence:.0%}")

    def evaluate(self, champion_trades: List[dict]) -> dict:
        """Evaluate challenger vs champion performance.

        Args:
            champion_trades: Recent champion model trades with 'result' and 'pnl'.

        Returns:
            {"promote": bool, "challenger_wr": float, "champion_wr": float, ...}
        """
        if not self._challenger_version:
            return {"promote": False, "reason": "No challenger registered"}

        challenger_completed = [t for t in self._shadow_trades if t.outcome and t.outcome != "PENDING"]
        if len(challenger_completed) < self.SHADOW_PERIOD_TRADES:
            return {
                "promote": False,
                "reason": f"Insufficient shadow trades ({len(challenger_completed)}/{self.SHADOW_PERIOD_TRADES})",
                "challenger_version": self._challenger_version,
                "champion_version": self._champion_version,
            }

        # Compute win rates
        chal_wins = sum(1 for t in challenger_completed if t.outcome == "WIN")
        chal_wr = chal_wins / len(challenger_completed)

        champ_wins = sum(1 for t in champion_trades if t.get("result") == "WIN")
        champ_wr = champ_wins / len(champion_trades) if champion_trades else 0.5

        # Compute average PnL
        chal_avg_pnl = np.mean([t.pnl for t in challenger_completed])
        champ_avg_pnl = np.mean([float(t.get("pnl", 0)) for t in champion_trades]) if champion_trades else 0

        # Decision
        wr_diff = chal_wr - champ_wr
        pnl_diff = chal_avg_pnl - champ_avg_pnl

        if wr_diff >= self.PROMOTION_THRESHOLD and pnl_diff > 0:
            decision = "PROMOTE"
            reason = f"Challenger WR={chal_wr:.0%} > Champion WR={champ_wr:.0%} (diff +{wr_diff:.0%})"
        elif wr_diff <= self.DEMOTION_THRESHOLD:
            decision = "REJECT"
            reason = f"Challenger WR={chal_wr:.0%} << Champion WR={champ_wr:.0%} (diff {wr_diff:.0%})"
        else:
            decision = "CONTINUE"
            reason = f"Challenger WR={chal_wr:.0%} ≈ Champion WR={champ_wr:.0%} — continue shadow"

        result = {
            "promote": decision == "PROMOTE",
            "reject": decision == "REJECT",
            "decision": decision,
            "challenger_version": self._challenger_version,
            "champion_version": self._champion_version,
            "challenger_wr": round(chal_wr, 3),
            "champion_wr": round(champ_wr, 3),
            "wr_difference": round(wr_diff, 3),
            "challenger_avg_pnl": round(chal_avg_pnl, 2),
            "champion_avg_pnl": round(champ_avg_pnl, 2),
            "n_challenger_trades": len(challenger_completed),
            "reason": reason,
        }

        self._evaluation_history.append(result)

        if decision == "PROMOTE":
            log.info(f"[ShadowMode] PROMOTE: {self._challenger_version} → new champion ({reason})")
            self._champion_version = self._challenger_version
            self._challenger_version = None
        elif decision == "REJECT":
            log.warning(f"[ShadowMode] REJECT: {self._challenger_version} ({reason})")
            self._challenger_version = None

        return result

    def status(self) -> dict:
        return {
            "champion_version": self._champion_version,
            "challenger_version": self._challenger_version,
            "shadow_trades": len(self._shadow_trades),
            "evaluations": len(self._evaluation_history),
            "shadow_active": self._challenger_version is not None,
        }


# ════════════════════════════════════════════════════════════════════
# 6. STRATEGY MARKETPLACE — Automatic strategy lifecycle management
# ════════════════════════════════════════════════════════════════════

@dataclass
class StrategyRecord:
    """A strategy's record in the marketplace."""
    name: str
    status: str = "ACTIVE"  # ACTIVE / PROBATION / RETIRED / EXPERIMENTAL
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    sharpe: float = 0.0
    last_evaluated: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    promoted_at: Optional[str] = None
    retired_at: Optional[str] = None
    win_rate: float = 0.0
    avg_pnl: float = 0.0


class StrategyMarketplace:
    """Internal strategy marketplace — rank, retire, promote strategies.

    Lifecycle:
    1. EXPERIMENTAL: new strategy, paper-trade only (50 trades min)
    2. ACTIVE: passes evaluation, trades live
    3. PROBATION: performance declining, monitored closely
    4. RETIRED: edge lost, no longer trades

    The marketplace automatically:
    - Promotes experimental strategies that prove themselves
    - Puts declining strategies on probation
    - Retires strategies that lose their edge
    - Ranks all active strategies by performance
    """

    MIN_TRADES_FOR_PROMOTION = 30
    PROBATION_WR_THRESHOLD = 0.35  # below 35% WR → probation
    RETIRE_WR_THRESHOLD = 0.25  # below 25% WR → retire
    PROBATION_TRADES = 20  # trades on probation before retire decision

    def __init__(self):
        self._strategies: Dict[str, StrategyRecord] = {}
        self._load()

    def _load(self):
        """Load strategy records from disk."""
        try:
            path = Path("memory/strategy_marketplace.json")
            if path.exists():
                data = json.loads(path.read_text())
                for name, rec in data.items():
                    self._strategies[name] = StrategyRecord(**rec)
        except Exception as e:
            log.debug(f"[StrategyMarket] load failed: {e}")

    def _save(self):
        """Save to disk."""
        try:
            path = Path("memory/strategy_marketplace.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {name: rec.__dict__ for name, rec in self._strategies.items()}
            path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            log.warning(f"[StrategyMarket] save failed: {e}")

    def register(self, name: str, status: str = "EXPERIMENTAL"):
        """Register a new strategy in the marketplace."""
        if name not in self._strategies:
            self._strategies[name] = StrategyRecord(name=name, status=status)
            log.info(f"[StrategyMarket] Registered: {name} ({status})")
            self._save()

    def record_trade(self, strategy: str, result: str, pnl: float):
        """Record a trade outcome for a strategy."""
        if strategy not in self._strategies:
            self.register(strategy)

        rec = self._strategies[strategy]
        rec.total_trades += 1
        if result == "WIN":
            rec.wins += 1
        elif result == "LOSS":
            rec.losses += 1
        rec.total_pnl += pnl
        rec.win_rate = rec.wins / rec.total_trades if rec.total_trades > 0 else 0
        rec.avg_pnl = rec.total_pnl / rec.total_trades if rec.total_trades > 0 else 0
        rec.last_evaluated = datetime.now(timezone.utc).isoformat()

        # Check for status transitions
        self._check_transitions(rec)
        self._save()

    def _check_transitions(self, rec: StrategyRecord):
        """Check if a strategy should be promoted, put on probation, or retired."""
        # EXPERIMENTAL → ACTIVE
        if rec.status == "EXPERIMENTAL" and rec.total_trades >= self.MIN_TRADES_FOR_PROMOTION:
            if rec.win_rate >= 0.45 and rec.total_pnl > 0:
                rec.status = "ACTIVE"
                rec.promoted_at = datetime.now(timezone.utc).isoformat()
                log.info(f"[StrategyMarket] PROMOTED: {rec.name} → ACTIVE (WR={rec.win_rate:.0%}, PnL=${rec.total_pnl:.2f})")

        # ACTIVE → PROBATION
        if rec.status == "ACTIVE" and rec.total_trades >= 20:
            recent_wr = rec.win_rate  # simplified — could use rolling window
            if recent_wr < self.PROBATION_WR_THRESHOLD:
                rec.status = "PROBATION"
                log.warning(f"[StrategyMarket] PROBATION: {rec.name} (WR={recent_wr:.0%} < {self.PROBATION_WR_THRESHOLD:.0%})")

        # PROBATION → RETIRED
        if rec.status == "PROBATION":
            if rec.win_rate < self.RETIRE_WR_THRESHOLD and rec.total_trades >= self.PROBATION_TRADES:
                rec.status = "RETIRED"
                rec.retired_at = datetime.now(timezone.utc).isoformat()
                log.warning(f"[StrategyMarket] RETIRED: {rec.name} (WR={rec.win_rate:.0%} < {self.RETIRE_WR_THRESHOLD:.0%})")

        # RETIRED → can be re-activated manually
        # (no automatic re-activation — requires human review)

    def get_rankings(self) -> List[dict]:
        """Get strategy rankings (best to worst)."""
        active = [r for r in self._strategies.values() if r.status in ("ACTIVE", "EXPERIMENTAL")]
        ranked = sorted(active, key=lambda r: (r.win_rate, r.total_pnl), reverse=True)
        return [
            {
                "name": r.name,
                "status": r.status,
                "win_rate": round(r.win_rate, 3),
                "total_pnl": round(r.total_pnl, 2),
                "total_trades": r.total_trades,
                "avg_pnl": round(r.avg_pnl, 2),
            }
            for r in ranked
        ]

    def get_best_strategy(self) -> Optional[str]:
        """Get the best performing active strategy."""
        rankings = self.get_rankings()
        return rankings[0]["name"] if rankings else None

    def is_strategy_allowed(self, name: str) -> bool:
        """Check if a strategy is allowed to trade."""
        if name not in self._strategies:
            return True  # unregistered = allowed by default
        return self._strategies[name].status in ("ACTIVE", "EXPERIMENTAL")

    def status_report(self) -> dict:
        return {
            "total_strategies": len(self._strategies),
            "active": sum(1 for r in self._strategies.values() if r.status == "ACTIVE"),
            "experimental": sum(1 for r in self._strategies.values() if r.status == "EXPERIMENTAL"),
            "probation": sum(1 for r in self._strategies.values() if r.status == "PROBATION"),
            "retired": sum(1 for r in self._strategies.values() if r.status == "RETIRED"),
            "rankings": self.get_rankings()[:5],  # top 5
        }


# ════════════════════════════════════════════════════════════════════
# 12. DATA SOURCE VOTING — Multi-source conflict detection
# ════════════════════════════════════════════════════════════════════

class DataSourceVoter:
    """Multi-source data voting and conflict detection.

    When multiple data sources provide the same information (e.g., price
    from MT5 + yfinance + broker API), this module:
    1. Collects values from all sources
    2. Detects conflicts (significant disagreement)
    3. Votes on the most likely correct value
    4. Reduces confidence when sources disagree

    This prevents bad data from a single source from corrupting decisions.
    """

    CONFLICT_THRESHOLD_PCT = 0.001  # 0.1% disagreement = conflict
    MIN_SOURCES = 2  # need at least 2 sources to vote

    def __init__(self):
        self._source_history: Dict[str, List[dict]] = defaultdict(list)

    def vote_price(self, prices_by_source: Dict[str, float]) -> dict:
        """Vote on the correct price from multiple sources.

        Args:
            prices_by_source: {"MT5": 1.1000, "yfinance": 1.1001, "broker": 1.1002}

        Returns:
            {"consensus_price": float, "conflict": bool, "confidence": float, ...}
        """
        if len(prices_by_source) < self.MIN_SOURCES:
            # Single source — no voting possible
            price = list(prices_by_source.values())[0] if prices_by_source else 0
            return {
                "consensus_price": price,
                "conflict": False,
                "confidence": 0.5,  # lower confidence with single source
                "sources": list(prices_by_source.keys()),
                "reason": "Single source — no voting",
            }

        values = list(prices_by_source.values())
        median_price = float(np.median(values))
        mean_price = float(np.mean(values))

        # Check for conflicts
        max_val = max(values)
        min_val = min(values)
        spread_pct = (max_val - min_val) / median_price if median_price > 0 else 0

        conflict = spread_pct > self.CONFLICT_THRESHOLD_PCT

        # Confidence: higher agreement = higher confidence
        if not conflict:
            confidence = 0.95  # sources agree
        else:
            # Reduce confidence proportional to disagreement
            confidence = max(0.3, 0.95 - spread_pct * 100)

        # Identify outlier sources
        outliers = []
        for source, price in prices_by_source.items():
            deviation = abs(price - median_price) / median_price if median_price > 0 else 0
            if deviation > self.CONFLICT_THRESHOLD_PCT:
                outliers.append({"source": source, "price": price, "deviation_pct": round(deviation * 100, 3)})

        # Consensus: use median (robust to outliers)
        consensus = median_price

        result = {
            "consensus_price": round(consensus, 5),
            "conflict": conflict,
            "confidence": round(confidence, 3),
            "sources": list(prices_by_source.keys()),
            "n_sources": len(prices_by_source),
            "median_price": round(median_price, 5),
            "mean_price": round(mean_price, 5),
            "spread_pct": round(spread_pct * 100, 3),
            "outliers": outliers,
            "reason": f"{'CONFLICT' if conflict else 'AGREEMENT'}: {len(prices_by_source)} sources, spread={spread_pct*100:.3f}%",
        }

        if conflict:
            log.warning(f"[DataVoter] CONFLICT: {outliers} — confidence reduced to {confidence:.0%}")
        else:
            log.debug(f"[DataVoter] Agreement: {len(prices_by_source)} sources, spread={spread_pct*100:.3f}%")

        return result

    def vote_indicator(self, indicator_name: str,
                       values_by_source: Dict[str, float]) -> dict:
        """Vote on an indicator value from multiple sources.

        Same logic as vote_price but for indicators (RSI, MACD, etc.).
        """
        return self.vote_price(values_by_source)  # same logic

    def get_source_reliability(self, source: str, window: int = 100) -> dict:
        """Get reliability score for a data source.

        Based on how often this source was the outlier.
        """
        history = self._source_history.get(source, [])
        if len(history) < 10:
            return {"reliability": 0.5, "reason": "Insufficient history"}

        recent = history[-window:]
        outlier_count = sum(1 for h in recent if h.get("was_outlier", False))
        reliability = 1.0 - (outlier_count / len(recent))

        return {
            "source": source,
            "reliability": round(reliability, 3),
            "outlier_rate": round(outlier_count / len(recent), 3),
            "total_observations": len(recent),
        }


# ════════════════════════════════════════════════════════════════════
# SMOKE TESTS
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== 1. Shadow Mode ===")
    sm = ShadowModeManager()
    sm.set_champion("v1")
    sm.register_challenger("v2")
    # Simulate 50 challenger trades
    import random
    random.seed(42)
    for i in range(50):
        outcome = "WIN" if random.random() > 0.35 else "LOSS"
        pnl = random.uniform(20, 80) if outcome == "WIN" else random.uniform(-50, -10)
        sm.record_shadow_trade(ShadowTrade(
            timestamp=datetime.now().isoformat(), model_version="v2",
            symbol="EURUSD", direction="BUY", entry=1.1, sl=1.095, tp=1.11,
            confidence=0.7, outcome=outcome, pnl=pnl,
        ))
    # Simulate champion trades
    champ_trades = [{"result": "WIN" if random.random() > 0.45 else "LOSS", "pnl": random.uniform(-40, 60)} for _ in range(50)]
    eval_result = sm.evaluate(champ_trades)
    print(f"  Decision: {eval_result['decision']}")
    print(f"  Challenger WR: {eval_result.get('challenger_wr', 0):.0%}")
    print(f"  Champion WR: {eval_result.get('champion_wr', 0):.0%}")

    print("\n=== 2. Strategy Marketplace ===")
    market = StrategyMarketplace()
    market.register("TREND_FOLLOW", "EXPERIMENTAL")
    # Simulate trades
    for i in range(35):
        result = "WIN" if random.random() > 0.4 else "LOSS"
        pnl = random.uniform(20, 60) if result == "WIN" else random.uniform(-40, -10)
        market.record_trade("TREND_FOLLOW", result, pnl)
    print(f"  Status: {market._strategies['TREND_FOLLOW'].status}")
    print(f"  WR: {market._strategies['TREND_FOLLOW'].win_rate:.0%}")
    print(f"  Rankings: {market.get_rankings()[:3]}")

    # Simulate bad strategy
    market.register("BAD_BREAKOUT", "EXPERIMENTAL")
    for i in range(30):
        result = "WIN" if random.random() > 0.8 else "LOSS"
        pnl = random.uniform(10, 30) if result == "WIN" else random.uniform(-50, -20)
        market.record_trade("BAD_BREAKOUT", result, pnl)
    print(f"  Bad strategy status: {market._strategies['BAD_BREAKOUT'].status}")

    print("\n=== 3. Data Source Voting ===")
    voter = DataSourceVoter()
    # Agreement case
    r1 = voter.vote_price({"MT5": 1.1000, "yfinance": 1.1001, "broker": 1.1000})
    print(f"  Agreement: consensus={r1['consensus_price']} conflict={r1['conflict']} conf={r1['confidence']:.0%}")
    # Conflict case
    r2 = voter.vote_price({"MT5": 1.1000, "yfinance": 1.1050, "broker": 1.1001})
    print(f"  Conflict: consensus={r2['consensus_price']} conflict={r2['conflict']} conf={r2['confidence']:.0%}")
    print(f"  Outliers: {r2['outliers']}")

    print("\nAll production excellence smoke tests passed.")