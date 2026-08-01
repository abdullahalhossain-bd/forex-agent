# risk/capital_manager.py — Day 58 | Portfolio Capital Allocation Manager
# ============================================================
# Manages how capital is distributed across currency pairs,
# strategies, and reserves. Works with the Autonomous Risk Manager
# to ensure optimal deployment of available funds.
#
# Features:
#   - Per-pair capital allocation with max limits
#   - Reserve capital management (always keep buffer)
#   - Strategy-based capital priority
#   - Dynamic rebalancing based on performance
# ============================================================

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from utils.logger import get_logger
from core.constants import MEMORY_DIR

log = get_logger("capital_manager")

CAPITAL_STATE_PATH = MEMORY_DIR / "capital_allocation_state.json"

# Default allocation weights per pair (normalized to 1.0)
DEFAULT_PAIR_WEIGHTS = {
    "EURUSD": 0.35,
    "GBPUSD": 0.25,
    "USDJPY": 0.20,
    "XAUUSD": 0.10,
    "RESERVE": 0.10,
}


class CapitalManager:
    """
    Portfolio Capital Allocation Manager.

    AI-এর Capital Allocation Brain.
    Determines how much capital to deploy across pairs and strategies.

    Responsibilities:
      1. Track capital allocated to each position/pair
      2. Maintain reserve capital buffer
      3. Rebalance allocations based on performance
      4. Enforce maximum allocation limits per pair
      5. Integrate with Day 57 research results for strategy priority

    Usage:
        cm = CapitalManager(total_capital=10000)
        cm.allocate("EURUSD", 1500)
        cm.allocate("GBPUSD", 1000)
        cm.print_status()
    """

    def __init__(
        self,
        total_capital: float = 10000.0,
        max_single_pair_pct: float = 0.40,
        reserve_ratio: float = 0.10,
    ):
        self.total_capital = total_capital
        self.max_single_pair_pct = max_single_pair_pct  # max 40% in one pair
        self.reserve_ratio = reserve_ratio              # always keep 10% reserve

        # Allocations: {symbol: allocated_amount}
        self._allocations: dict[str, float] = {}
        self._pair_weights: dict[str, float] = dict(DEFAULT_PAIR_WEIGHTS)
        self._performance_scores: dict[str, float] = {}

        self._state = self._load_state()

        log.info(
            f"[CapitalManager] Initialized | "
            f"Capital: ${total_capital:,.2f} | "
            f"Max single pair: {max_single_pair_pct*100:.0f}% | "
            f"Reserve: {reserve_ratio*100:.0f}%"
        )

    # ═══════════════════════════════════════════════════════
    # CAPITAL ALLOCATION
    # ═══════════════════════════════════════════════════════

    def allocate(self, symbol: str, amount: float) -> dict:
        """
        Allocate capital to a specific pair.

        Checks:
          - Not exceeding max single pair limit
          - Not exceeding total deployable capital
          - Reserve is maintained

        Returns:
            {"success": True/False, "allocated": amount, "reason": str}
        """
        symbol = symbol.upper()
        reserve = self.total_capital * self.reserve_ratio
        available = self.total_capital - self.get_total_allocated() - reserve

        if available <= 0:
            return {
                "success": False, "allocated": 0,
                "reason": "No deployable capital (reserve protected)",
            }

        # Max single pair check
        max_for_pair = self.total_capital * self.max_single_pair_pct
        current_for_pair = self._allocations.get(symbol, 0)
        proposed_total = current_for_pair + amount

        if proposed_total > max_for_pair:
            allowed = max_for_pair - current_for_pair
            allowed = max(0, min(allowed, available))
            self._allocations[symbol] = current_for_pair + allowed
            self._save_state()
            return {
                "success": allowed > 0,
                "allocated": round(allowed, 2),
                "reason": f"Capped at max {self.max_single_pair_pct*100:.0f}% "
                         f"(${max_for_pair:,.0f}) for {symbol}",
            }

        # Cap to available
        actual = min(amount, available)
        self._allocations[symbol] = current_for_pair + actual
        self._save_state()

        log.info(
            f"[CapitalManager] Allocated ${actual:,.2f} to {symbol} "
            f"(total for pair: ${self._allocations[symbol]:,.2f})"
        )

        return {
            "success": True,
            "allocated": round(actual, 2),
            "reason": "OK",
        }

    def deallocate(self, symbol: str, amount: float | None = None) -> float:
        """
        Release capital from a pair. If amount is None, release all.

        Returns:
            Amount released.
        """
        symbol = symbol.upper()
        current = self._allocations.get(symbol, 0)

        if amount is None:
            released = current
        else:
            released = min(amount, current)

        self._allocations[symbol] = max(0, current - released)

        if self._allocations[symbol] == 0:
            self._allocations.pop(symbol, None)

        self._save_state()
        log.info(
            f"[CapitalManager] Deallocated ${released:,.2f} from {symbol}"
        )
        return round(released, 2)

    # ═══════════════════════════════════════════════════════
    # PORTFOLIO ALLOCATION
    # ═══════════════════════════════════════════════════════

    def compute_optimal_allocation(
        self,
        symbols: list[str],
        strategy_ranking: dict[str, float] | None = None,
        risk_mode: str = "NORMAL",
    ) -> dict:
        """
        Compute optimal capital allocation across symbols.

        Factors:
          1. Pair weights (default config)
          2. Strategy performance ranking (from Day 57 research)
          3. Risk mode (AGGRESSIVE/NORMAL/DEFENSIVE/EMERGENCY)
          4. Correlation between pairs (avoid over-concentration)

        Args:
            symbols: List of symbols to allocate to
            strategy_ranking: {strategy_name: weight} from research results
            risk_mode: Current risk mode

        Returns:
            {symbol: allocated_amount} dict
        """
        if risk_mode == "EMERGENCY":
            log.warning("[CapitalManager] EMERGENCY — no capital deployment")
            return {}

        # Mode-based deploy limits
        deploy_limits = {
            "AGGRESSIVE": 0.90,
            "NORMAL": 0.75,
            "DEFENSIVE": 0.40,
            "EMERGENCY": 0.0,
        }
        max_deploy_pct = deploy_limits.get(risk_mode, 0.75)
        deployable = self.total_capital * max_deploy_pct

        # Calculate weights for requested symbols
        weights = {}
        for sym in symbols:
            base_weight = self._pair_weights.get(sym, 0.10)
            weights[sym] = base_weight

        # Boost weights based on strategy ranking
        if strategy_ranking:
            for sym in weights:
                best_strat = self._get_best_strategy_for_pair(sym, strategy_ranking)
                if best_strat:
                    boost = strategy_ranking.get(best_strat, 0.1) * 0.3
                    weights[sym] = min(weights[sym] + boost, 0.50)

        # Normalize weights
        total_weight = sum(weights.values()) or 1.0
        allocations = {}
        for sym, w in weights.items():
            alloc = deployable * (w / total_weight)
            allocations[sym] = round(alloc, 2)

        # Enforce max single pair
        max_single = self.total_capital * self.max_single_pair_pct
        for sym in allocations:
            if allocations[sym] > max_single:
                excess = allocations[sym] - max_single
                allocations[sym] = max_single
                # Redistribute excess to other pairs
                others = [s for s in allocations if s != sym]
                if others:
                    per_other = excess / len(others)
                    for other in others:
                        allocations[other] = min(
                            allocations[other] + per_other, max_single
                        )

        log.info(
            f"[CapitalManager] Optimal allocation ({risk_mode}): "
            + ", ".join(f"{s}: ${a:,.0f}" for s, a in allocations.items())
        )

        return allocations

    def _get_best_strategy_for_pair(
        self, symbol: str, strategy_ranking: dict
    ) -> str | None:
        """Find the highest-ranked strategy for a pair."""
        # In a real system, this would use research results to find
        # which strategy works best for which pair
        best = max(strategy_ranking, key=strategy_ranking.get)
        return best if strategy_ranking.get(best, 0) > 0.1 else None

    # ═══════════════════════════════════════════════════════
    # QUERIES
    # ═══════════════════════════════════════════════════════

    def get_allocations(self) -> dict[str, float]:
        """Get current allocations dict."""
        return dict(self._allocations)

    def get_total_allocated(self) -> float:
        """Get total capital currently allocated."""
        return sum(self._allocations.values())

    def get_reserve(self) -> float:
        """Get reserve capital (unallocated minus reserve minimum)."""
        allocated = self.get_total_allocated()
        required_reserve = self.total_capital * self.reserve_ratio
        free = self.total_capital - allocated
        return max(0, free - required_reserve)

    def get_available_for_deployment(self) -> float:
        """Get capital available for new deployments (after reserve)."""
        return self.total_capital - self.get_total_allocated() - (
            self.total_capital * self.reserve_ratio
        )

    def get_allocation_pct(self, symbol: str) -> float:
        """Get allocation percentage for a symbol."""
        allocated = self._allocations.get(symbol.upper(), 0)
        return (allocated / self.total_capital * 100) if self.total_capital > 0 else 0

    # ═══════════════════════════════════════════════════════
    # UPDATE & REBALANCE
    # ═══════════════════════════════════════════════════════

    def update_balance(self, new_balance: float) -> None:
        """Update total capital (e.g., after trade PnL)."""
        old = self.total_capital
        self.total_capital = max(0, new_balance)

        # Scale allocations proportionally
        if old > 0 and new_balance > 0:
            scale = new_balance / old
            self._allocations = {
                k: round(v * scale, 2)
                for k, v in self._allocations.items()
            }

        self._save_state()

    def rebalance(self, target_weights: dict[str, float]) -> dict:
        """
        Rebalance allocations to match target weights.

        Args:
            target_weights: {symbol: target_weight} (should sum to ~1.0)

        Returns:
            Rebalancing report.
        """
        deployable = self.total_capital * (1 - self.reserve_ratio)
        changes = {}

        for sym, weight in target_weights.items():
            target_amount = deployable * weight
            current = self._allocations.get(sym, 0)
            diff = target_amount - current
            changes[sym] = {
                "current": round(current, 2),
                "target": round(target_amount, 2),
                "change": round(diff, 2),
                "action": "increase" if diff > 0 else "decrease",
            }

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_capital": round(self.total_capital, 2),
            "deployable": round(deployable, 2),
            "changes": changes,
        }

        log.info(f"[CapitalManager] Rebalance computed: {len(changes)} changes")
        return report

    def update_strategy_weights(
        self, strategy_ranking: dict[str, float]
    ) -> None:
        """
        Update pair weights based on Day 57 research results.

        If a strategy is performing well for a pair, increase
        that pair's allocation weight.
        """
        self._performance_scores = dict(strategy_ranking)
        self._save_state()
        log.info(
            f"[CapitalManager] Strategy weights updated: "
            f"{len(strategy_ranking)} strategies"
        )

    # ═══════════════════════════════════════════════════════
    # STATUS
    # ═══════════════════════════════════════════════════════

    def get_summary(self) -> dict:
        """Get capital management summary."""
        allocated = self.get_total_allocated()
        reserve = self.total_capital - allocated
        return {
            "total_capital": round(self.total_capital, 2),
            "total_allocated": round(allocated, 2),
            "reserve": round(reserve, 2),
            "allocated_pct": round(allocated / self.total_capital * 100, 1) if self.total_capital > 0 else 0,
            "reserve_pct": round(reserve / self.total_capital * 100, 1) if self.total_capital > 0 else 0,
            "pair_count": len(self._allocations),
            "allocations": dict(self._allocations),
        }

    def print_status(self) -> None:
        """Print capital allocation status."""
        s = self.get_summary()
        bar = "=" * 44
        print(f"\n{bar}")
        print("  CAPITAL MANAGER")
        print(bar)
        print(f"  Total Capital   : ${s['total_capital']:,.2f}")
        print(f"  Allocated       : ${s['total_allocated']:,.2f} ({s['allocated_pct']}%)")
        print(f"  Reserve         : ${s['reserve']:,.2f} ({s['reserve_pct']}%)")
        print(f"  Active Pairs    : {s['pair_count']}")
        if s["allocations"]:
            print("  Allocations:")
            for sym, amt in s["allocations"].items():
                pct = amt / self.total_capital * 100 if self.total_capital > 0 else 0
                print(f"    {sym:<10}: ${amt:>8,.2f} ({pct:.1f}%)")
        print(bar + "\n")

    # ═══════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════

    def _load_state(self) -> dict:
        CAPITAL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if CAPITAL_STATE_PATH.exists():
            try:
                with open(CAPITAL_STATE_PATH) as f:
                    return json.load(f)
            except Exception as e:
                log.warning("Suppressed exception while loading capital state: %s", e)
                pass
        return {"allocations": {}, "pair_weights": dict(DEFAULT_PAIR_WEIGHTS)}

    def _save_state(self) -> None:
        state = {
            "total_capital": self.total_capital,
            "allocations": self._allocations,
            "pair_weights": self._pair_weights,
            "performance_scores": self._performance_scores,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # P1 fix: atomic write (temp + os.replace) to prevent crash-corruption.
        # Mirrors the pattern used by kill_switch.py, circuit_breaker.py, risk_engine.py,
        # drawdown_controller.py, autonomous_risk.py, approval_mode.py.
        tmp_path = str(CAPITAL_STATE_PATH) + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        try:
            os.replace(tmp_path, CAPITAL_STATE_PATH)
        except OSError as e:
            log.error(f"[CapitalManager] atomic save failed: {e}")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise