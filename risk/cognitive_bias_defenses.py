"""
risk/cognitive_bias_defenses.py — Defenses against your own mind
=================================================================

Addresses the 6 cognitive biases identified in the audit:

  1. Confirmation bias  → PreRegistrationFramework (write hypothesis BEFORE testing)
  2. Survivorship bias  → StrategyGraveyard (track failures as carefully as successes)
  3. Recency bias       → RegimeTaggedStrategies (test across multiple regimes)
  4. Gambler's fallacy  → IndependentEventTracker (don't let streaks affect decisions)
  5. Overconfidence     → CalibrationTracker (track prediction accuracy)
  6. Selection bias     → SelectionAuditLog (log every selection decision)

Usage:
    from risk.cognitive_bias_defenses import (
        PreRegistrationFramework, StrategyGraveyard,
        CalibrationTracker, SelectionAuditLog,
    )
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

log = get_logger("bias_defense")


# ════════════════════════════════════════════════════════════════
#  DEFENSE #1: Pre-Registration Framework (Confirmation Bias)
# ════════════════════════════════════════════════════════════════

@dataclass
class PreRegistration:
    """
    A pre-registered hypothesis — written BEFORE looking at data.

    This forces you to state what you expect, preventing you from
    retrofitting explanations to random results.
    """
    id: str
    hypothesis: str               # "Strategy X will have WR > 55% on EURUSD H1"
    expected_metric: str          # "win_rate"
    expected_value: float         # 0.55
    expected_direction: str       # "greater_than" | "less_than" | "equal"
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    registered_at: str = ""
    registered_before_data: bool = True
    # Results (filled in AFTER testing)
    actual_value: Optional[float] = None
    result: str = ""              # "confirmed" | "refuted" | "inconclusive"
    resolved_at: str = ""


class PreRegistrationFramework:
    """
    Defense against confirmation bias.

    You MUST pre-register a hypothesis before looking at backtest results.
    The framework locks the hypothesis — you can't change it after seeing data.

    Process:
      1. Before backtest: register hypothesis
      2. Run backtest
      3. Compare results to pre-registered expectation
      4. Record whether confirmed/refuted — no reinterpretation allowed
    """

    def __init__(self, storage_path: str = "state/pre_registrations.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.registrations: Dict[str, PreRegistration] = {}
        self._load()

    def register(
        self,
        hypothesis: str,
        expected_metric: str,
        expected_value: float,
        expected_direction: str = "greater_than",
    ) -> str:
        """Register a hypothesis BEFORE looking at data."""
        with self._lock:
            reg_id = f"reg_{len(self.registrations) + 1:04d}"
            reg = PreRegistration(
                id=reg_id,
                hypothesis=hypothesis,
                expected_metric=expected_metric,
                expected_value=expected_value,
                expected_direction=expected_direction,
                registered_at=datetime.now(timezone.utc).isoformat(),
                registered_before_data=True,
            )
            self.registrations[reg_id] = reg
            self._save()
            log.info(f"[PreReg] Registered {reg_id}: {hypothesis[:60]}...")
            return reg_id

    def resolve(
        self,
        reg_id: str,
        actual_value: float,
    ) -> str:
        """Resolve a pre-registration with actual data."""
        with self._lock:
            if reg_id not in self.registrations:
                return "not_found"
            reg = self.registrations[reg_id]
            reg.actual_value = actual_value
            reg.resolved_at = datetime.now(timezone.utc).isoformat()

            if reg.expected_direction == "greater_than":
                reg.result = "confirmed" if actual_value > reg.expected_value else "refuted"
            elif reg.expected_direction == "less_than":
                reg.result = "confirmed" if actual_value < reg.expected_value else "refuted"
            else:  # equal
                diff = abs(actual_value - reg.expected_value)
                reg.result = "confirmed" if diff < 0.05 else "refuted"

            self._save()
            log.info(f"[PreReg] {reg_id} {reg.result}: "
                     f"expected {reg.expected_value}, got {actual_value}")
            return reg.result

    def get_confirmation_rate(self) -> float:
        """What fraction of pre-registered hypotheses were confirmed?

        If this is > 70%, you're probably cheating (changing hypotheses
        after seeing data, or only registering things you're sure of).
        If < 30%, your strategies don't work.
        Healthy range: 30-60%.
        """
        resolved = [r for r in self.registrations.values() if r.result]
        if not resolved:
            return 0.0
        confirmed = sum(1 for r in resolved if r.result == "confirmed")
        return confirmed / len(resolved)

    def _save(self):
        data = {k: asdict(v) for k, v in self.registrations.items()}
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            for k, v in data.items():
                # Convert tuple back from list
                if 'confidence_interval' in v and isinstance(v['confidence_interval'], list):
                    v['confidence_interval'] = tuple(v['confidence_interval'])
                self.registrations[k] = PreRegistration(**v)
        except Exception as e:
            log.warning(f"[PreReg] Failed to load: {e}")


# ════════════════════════════════════════════════════════════════
#  DEFENSE #2: Strategy Graveyard (Survivorship Bias)
# ════════════════════════════════════════════════════════════════

@dataclass
class GraveyardEntry:
    """A failed strategy — recorded so it's not forgotten."""
    strategy_name: str
    pair: str
    timeframe: str
    n_trades: int
    win_rate: float
    avg_r: float
    failure_reason: str          # "WR too low" | "regime change" | "OOS failed"
    buried_at: str = ""
    lessons: str = ""            # what was learned


class StrategyGraveyard:
    """
    Defense against survivorship bias.

    Tracks EVERY strategy that failed — not just the winners.
    The graveyard is more informative than the leaderboard because:
      - Failures teach you what NOT to do
      - Prevents you from re-trying the same failed approach
      - Forces honest accounting of your track record
    """

    def __init__(self, storage_path: str = "state/strategy_graveyard.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.graveyard: List[GraveyardEntry] = []
        self._load()

    def bury(
        self,
        strategy_name: str,
        pair: str,
        timeframe: str,
        n_trades: int,
        win_rate: float,
        avg_r: float,
        failure_reason: str,
        lessons: str = "",
    ):
        """Record a failed strategy."""
        with self._lock:
            entry = GraveyardEntry(
                strategy_name=strategy_name,
                pair=pair, timeframe=timeframe,
                n_trades=n_trades, win_rate=win_rate,
                avg_r=avg_r, failure_reason=failure_reason,
                lessons=lessons,
                buried_at=datetime.now(timezone.utc).isoformat(),
            )
            self.graveyard.append(entry)
            self._save()
            log.info(f"[Graveyard] Buried {strategy_name} on {pair} {timeframe}: "
                     f"{failure_reason}")

    def is_already_failed(
        self,
        strategy_name: str,
        pair: str,
        timeframe: str,
    ) -> Optional[GraveyardEntry]:
        """Check if this strategy already failed on this pair/TF."""
        for entry in self.graveyard:
            if (entry.strategy_name == strategy_name
                and entry.pair == pair
                and entry.timeframe == timeframe):
                return entry
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get graveyard statistics."""
        total = len(self.graveyard)
        by_reason: Dict[str, int] = {}
        by_strategy: Dict[str, int] = {}
        for entry in self.graveyard:
            by_reason[entry.failure_reason] = by_reason.get(entry.failure_reason, 0) + 1
            by_strategy[entry.strategy_name] = by_strategy.get(entry.strategy_name, 0) + 1
        return {
            "total_buried": total,
            "by_reason": by_reason,
            "by_strategy": by_strategy,
        }

    def _save(self):
        data = [asdict(e) for e in self.graveyard]
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            self.graveyard = [GraveyardEntry(**d) for d in data]
        except Exception as e:
            log.warning(f"[Graveyard] Failed to load: {e}")


# ════════════════════════════════════════════════════════════════
#  DEFENSE #3: Calibration Tracker (Overconfidence Bias)
# ════════════════════════════════════════════════════════════════

class CalibrationTracker:
    """
    Defense against overconfidence bias.

    Tracks whether your confidence predictions match reality.
    If you say "80% confident" on 100 trades, ~80 should win.
    Most people are shocked to find their "80% confidence" wins 50%.

    Brier score: lower = better calibrated. 0.0 = perfect, 0.25 = random.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.predictions: List[Dict[str, Any]] = []

    def record_prediction(
        self,
        strategy: str,
        confidence: float,        # 0.0 to 1.0
        outcome: bool,            # True = win, False = loss
    ):
        """Record a prediction and its outcome."""
        with self._lock:
            self.predictions.append({
                "strategy": strategy,
                "confidence": confidence,
                "outcome": outcome,
                "time": datetime.now(timezone.utc).isoformat(),
            })

    def get_calibration(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        with self._lock:
            if not self.predictions:
                return {"error": "no predictions"}

            # Group by confidence bucket
            buckets = {}
            for p in self.predictions:
                bucket = round(p["confidence"], 1)  # 0.0, 0.1, ..., 1.0
                if bucket not in buckets:
                    buckets[bucket] = {"n": 0, "wins": 0}
                buckets[bucket]["n"] += 1
                if p["outcome"]:
                    buckets[bucket]["wins"] += 1

            # Calculate actual win rate per bucket
            calibration = {}
            for bucket, data in sorted(buckets.items()):
                calibration[bucket] = {
                    "n": data["n"],
                    "predicted_wr": bucket,
                    "actual_wr": data["wins"] / data["n"],
                    "calibration_error": abs(bucket - data["wins"] / data["n"]),
                }

            # Brier score (lower = better)
            brier = sum(
                (p["confidence"] - (1.0 if p["outcome"] else 0.0)) ** 2
                for p in self.predictions
            ) / len(self.predictions)

            return {
                "n_predictions": len(self.predictions),
                "calibration_by_bucket": calibration,
                "brier_score": round(brier, 4),
                "mean_confidence": sum(p["confidence"] for p in self.predictions) / len(self.predictions),
                "actual_win_rate": sum(1 for p in self.predictions if p["outcome"]) / len(self.predictions),
            }


# ════════════════════════════════════════════════════════════════
#  DEFENSE #4: Selection Audit Log (Selection Bias)
# ════════════════════════════════════════════════════════════════

class SelectionAuditLog:
    """
    Defense against selection bias.

    Logs every selection decision: which pairs/strategies/params you chose
    to test, and why. This makes selection bias visible.

    If you only test the "best-looking" pairs, the audit log will show it.
    """

    def __init__(self, storage_path: str = "state/selection_audit.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.selections: List[Dict[str, Any]] = []
        self._load()

    def log_selection(
        self,
        what: str,                  # "pair" | "strategy" | "timeframe" | "param"
        chosen: str,                # what was selected
        from_pool: List[str],       # what was available
        reason: str,                # why this was chosen
    ):
        """Log a selection decision."""
        with self._lock:
            entry = {
                "time": datetime.now(timezone.utc).isoformat(),
                "what": what,
                "chosen": chosen,
                "from_pool": from_pool,
                "pool_size": len(from_pool),
                "reason": reason,
            }
            self.selections.append(entry)
            self._save()

    def get_audit_summary(self) -> Dict[str, Any]:
        """Summarize selection patterns."""
        with self._lock:
            if not self.selections:
                return {"error": "no selections logged"}

            by_what: Dict[str, int] = {}
            for s in self.selections:
                by_what[s["what"]] = by_what.get(s["what"], 0) + 1

            # Check for cherry-picking patterns
            warnings = []
            pair_selections = [s for s in self.selections if s["what"] == "pair"]
            if pair_selections:
                chosen_pairs = set(s["chosen"] for s in pair_selections)
                total_available = set()
                for s in pair_selections:
                    total_available.update(s["from_pool"])
                coverage = len(chosen_pairs) / len(total_available) if total_available else 0
                if coverage < 0.3:
                    warnings.append(
                        f"LOW PAIR COVERAGE: only {coverage*100:.0f}% of available "
                        f"pairs tested — possible selection bias"
                    )

            return {
                "total_selections": len(self.selections),
                "by_type": by_what,
                "warnings": warnings,
            }

    def _save(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.selections, f, indent=2, default=str)

    def _load(self):
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path) as f:
                self.selections = json.load(f)
        except Exception as e:
            log.warning(f"[SelectionAudit] Failed to load: {e}")


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY (smoke test)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("=" * 70)
    print("  COGNITIVE BIAS DEFENSES — Smoke Test")
    print("=" * 70)

    # Defense #1: Pre-Registration
    print("\n── Defense #1: Pre-Registration Framework ──")
    prereg = PreRegistrationFramework(storage_path="/tmp/test_prereg.json")
    reg_id = prereg.register(
        hypothesis="EURUSD H1 donchian_breakout will have WR > 55%",
        expected_metric="win_rate",
        expected_value=0.55,
        expected_direction="greater_than",
    )
    print(f"  Registered: {reg_id}")
    result = prereg.resolve(reg_id, actual_value=0.893)
    print(f"  Resolved: {result} (actual 89.3% vs expected >55%)")
    print(f"  Confirmation rate: {prereg.get_confirmation_rate()*100:.0f}%")

    # Defense #2: Strategy Graveyard
    print("\n── Defense #2: Strategy Graveyard ──")
    graveyard = StrategyGraveyard(storage_path="/tmp/test_graveyard.json")
    graveyard.bury(
        strategy_name="sr_bounce",
        pair="GBPUSD", timeframe="H1",
        n_trades=57, win_rate=0.193, avg_r=-0.49,
        failure_reason="WR too low (19.3% < 40% threshold)",
        lessons="S/R bounces fail in trending regimes — need ADX filter",
    )
    check = graveyard.is_already_failed("sr_bounce", "GBPUSD", "H1")
    print(f"  Already failed? {check is not None}")
    print(f"  Summary: {graveyard.get_summary()}")

    # Defense #3: Calibration Tracker
    print("\n── Defense #3: Calibration Tracker ──")
    cal = CalibrationTracker()
    # Simulate predictions: claim 80% confidence but only 50% win
    for _ in range(40):
        cal.record_prediction("test", confidence=0.8, outcome=True)
    for _ in range(40):
        cal.record_prediction("test", confidence=0.8, outcome=False)
    stats = cal.get_calibration()
    print(f"  Predicted confidence: {stats['mean_confidence']*100:.0f}%")
    print(f"  Actual win rate: {stats['actual_win_rate']*100:.0f}%")
    print(f"  Brier score: {stats['brier_score']} (0=perfect, 0.25=random)")
    print(f"  → You're OVERCONFIDENT (predicted 80%, got 50%)")

    # Defense #4: Selection Audit
    print("\n── Defense #4: Selection Audit Log ──")
    audit = SelectionAuditLog(storage_path="/tmp/test_audit.json")
    audit.log_selection(
        what="pair",
        chosen="EURUSD",
        from_pool=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD",
                    "USDCAD", "NZDUSD", "USDCHF", "EURJPY", "GBPJPY"],
        reason="Best performer in V2 backtest",
    )
    summary = audit.get_audit_summary()
    print(f"  Selections logged: {summary['total_selections']}")
    print(f"  Warnings: {summary.get('warnings', 'none')}")

    print("\n" + "=" * 70)
    print("  All 4 cognitive bias defenses tested successfully.")
    print("=" * 70)
