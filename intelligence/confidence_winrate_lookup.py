"""
intelligence/confidence_winrate_lookup.py
==========================================

Historical win-rate lookup keyed by (strategy, confidence) and optionally (strategy, tactic).

PURPOSE
-------
When the live decision system tags a signal with a confidence level (High/Medium/Low),
this module answers:

  * "What win rate has this strategy+confidence historically delivered?"
  * "Is the sample size large enough to trust that number?"
  * "Should we trade, reduce size, or skip?"
  * "What position-scale multiplier should we apply?"

The data is loaded from `confidence_winrate_data.json` (sibling file), which is generated
from the latest backtest run. Re-run the backtest and regenerate the JSON to refresh.

USAGE — Python
--------------
    from intelligence.confidence_winrate_lookup import ConfidenceWinrateLookup

    lw = ConfidenceWinrateLookup()                       # auto-loads JSON
    rec = lw.recommend(strategy="ict_amd", confidence="High")
    print(rec)
    # Recommendation(action='trust', expected_winrate=1.0, n=3,
    #                position_scale=1.0, tier='use',
    #                sample_reliability='low_n', notes='...')

    # In risk sizing:
    size *= rec.position_scale          # 0.0 = skip, 1.0 = full size
    if rec.action == 'skip':
        return None

    # Per-tactic lookup
    tw = lw.tactic_winrate("candlestick_patterns", "Three Black Crows")
    # {'trades': 5, 'win_rate': 0.6, 'found': True}

USAGE — CLI
-----------
    python intelligence/confidence_winrate_lookup.py --strategy ict_amd --confidence High
    python intelligence/confidence_winrate_lookup.py --strategy stop_hunt --confidence Medium
    python intelligence/confidence_winrate_lookup.py --list-strategies
    python intelligence/confidence_winrate_lookup.py --tactic-lookup candlestick_patterns "Three Black Crows"
    python intelligence/confidence_winrate_lookup.py --summary

FAIL-SAFE
---------
If the JSON is missing or unreadable, every lookup returns a safe "skip" recommendation
with position_scale=0.0. We fail CLOSED — never trade on missing historical data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("confidence_winrate_lookup")

# ---------------------------------------------------------------------------
# Path resolution — works whether invoked from project root or as a module.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_JSON_PATH = _THIS_DIR / "confidence_winrate_data.json"

# Allow override via env var (useful for tests / CI)
_JSON_PATH = Path(os.environ.get("CONFIDENCE_WINRATE_DATA", str(_DEFAULT_JSON_PATH)))


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Recommendation:
    """Result of a (strategy, confidence) lookup."""
    found: bool                          # True if we have historical data for this combo
    strategy: str
    confidence: str                      # "High" | "Medium" | "Low" | "any"
    expected_winrate: Optional[float]    # 0.0-1.0, or None if no data
    n_trades: int                        # historical sample size
    action: str                          # "trust" | "caution" | "skip"
    position_scale: float                # 0.0-1.0 — multiply into position sizing
    tier: str                            # "use" | "caution" | "disable" | "no_data"
    sample_reliability: str              # "high_n" | "med_n" | "low_n" | "no_data"
    notes: str                           # human-readable explanation
    source: str                          # JSON path or "fallback"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# ConfidenceWinrateLookup
# ---------------------------------------------------------------------------

class ConfidenceWinrateLookup:
    """Load once, query many times. Thread-safe for read-only access."""

    def __init__(self, json_path: Optional[Path] = None) -> None:
        self.json_path: Path = json_path or _JSON_PATH
        self._data: Dict[str, Any] = {}
        self._loaded: bool = False
        self._load()

    # ---- loading ---------------------------------------------------------

    def _load(self) -> None:
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            self._loaded = True
            log.debug("Loaded confidence winrate data from %s", self.json_path)
        except FileNotFoundError:
            log.warning("confidence_winrate_data.json not found at %s — failing closed.", self.json_path)
            self._data = {}
            self._loaded = False
        except json.JSONDecodeError as e:
            log.error("Invalid JSON in %s: %s — failing closed.", self.json_path, e)
            self._data = {}
            self._loaded = False

    # ---- introspection ---------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def list_strategies(self) -> List[str]:
        if not self._loaded:
            return []
        return [s["strategy"] for s in self._data.get("strategy_summary", [])]

    def list_confidences(self, strategy: str) -> List[str]:
        if not self._loaded:
            return []
        per = self._data.get("per_confidence_winrates", {}).get(strategy, {})
        return list(per.keys())

    def summary(self) -> Dict[str, Any]:
        """High-level overview of what's in the JSON."""
        if not self._loaded:
            return {"loaded": False, "json_path": str(self.json_path)}

        return {
            "loaded": True,
            "json_path": str(self.json_path),
            "source": self._data.get("metadata", {}).get("source", "unknown"),
            "generated_at": self._data.get("metadata", {}).get("generated_at", "unknown"),
            "pairs_tested": self._data.get("metadata", {}).get("pairs_tested", 0),
            "timeframes": self._data.get("metadata", {}).get("timeframes", []),
            "strategies": self.list_strategies(),
            "suggested_decision_mode": self._data.get("metadata", {}).get("suggested_decision_mode"),
            "global_thresholds": self._data.get("global_thresholds", {}),
        }

    # ---- core lookups ----------------------------------------------------

    def winrate(self, strategy: str, confidence: str) -> Optional[float]:
        """Return historical win rate for (strategy, confidence) or None."""
        if not self._loaded:
            return None
        per = self._data.get("per_confidence_winrates", {}).get(strategy, {})
        entry = per.get(confidence)
        if not entry:
            return None
        return entry.get("win_rate")

    def trades(self, strategy: str, confidence: str) -> int:
        if not self._loaded:
            return 0
        per = self._data.get("per_confidence_winrates", {}).get(strategy, {})
        entry = per.get(confidence)
        if not entry:
            return 0
        return int(entry.get("trades", 0))

    def tier(self, strategy: str, confidence: str) -> str:
        if not self._loaded:
            return "no_data"
        per = self._data.get("per_confidence_winrates", {}).get(strategy, {})
        entry = per.get(confidence)
        if not entry:
            return "no_data"
        return entry.get("tier", "no_data")

    def sample_reliability(self, strategy: str, confidence: str) -> str:
        if not self._loaded:
            return "no_data"
        per = self._data.get("per_confidence_winrates", {}).get(strategy, {})
        entry = per.get(confidence)
        if not entry:
            return "no_data"
        return entry.get("sample_reliability", "no_data")

    # ---- recommendation --------------------------------------------------

    def recommend(
        self,
        strategy: str,
        confidence: str,
        tactic: Optional[str] = None,
    ) -> Recommendation:
        """
        Build a single actionable recommendation for the live decision system.

        Decision logic:
          1. If we have a confidence_rules entry in confidence_to_action_map, use it.
          2. Else compute on the fly: use per_confidence_winrates + global_thresholds.
          3. If JSON not loaded, return fail-closed skip.
          4. If tactic provided AND has its own data, allow tactic-level override
             (tactic WR >= 50% with n >= 10 promotes 'skip' -> 'caution').
        """
        if not self._loaded:
            return _fail_closed(strategy, confidence, source="fallback",
                                notes="JSON not loaded — failing closed.")

        # 1. Explicit rule match (highest priority)
        rule = self._match_rule(strategy, confidence)
        if rule is not None:
            return Recommendation(
                found=True,
                strategy=strategy,
                confidence=confidence,
                expected_winrate=rule.get("expected_winrate"),
                n_trades=int(rule.get("n", 0)),
                action=rule.get("action", "skip"),
                position_scale=float(rule.get("position_scale", 0.0)),
                tier=_tier_from_action(rule.get("action", "skip")),
                sample_reliability=self._reliability_from_n(int(rule.get("n", 0))),
                notes=rule.get("notes", ""),
                source=str(self.json_path),
            )

        # 2. Compute on the fly from per_confidence_winrates
        wr = self.winrate(strategy, confidence)
        n = self.trades(strategy, confidence)
        if wr is None or n == 0:
            # 2b. Try strategy-level fallback (candlestick_patterns, sr_zones, cci_state
            #     all have only one confidence bucket — apply to any confidence.)
            fallback = self._strategy_level_fallback(strategy)
            if fallback is not None:
                wr, n = fallback
                confidence_used = "any"
            else:
                return _fail_closed(strategy, confidence, source=str(self.json_path),
                                    notes="No historical data for this combo.")
        else:
            confidence_used = confidence

        action, scale, tier, notes = self._compute_action(wr, n)
        sample_rel = self._reliability_from_n(n)

        # 3. Tactic override — a known-good tactic can rescue a borderline skip
        if tactic and action == "skip":
            tw = self.tactic_winrate(strategy, tactic)
            if tw["found"] and tw["trades"] >= 10 and tw["win_rate"] >= 0.50:
                action = "caution"
                scale = 0.5
                tier = "caution"
                notes = (notes + f" Tactic '{tactic}' rescued: WR={tw['win_rate']:.1%}, "
                                  f"n={tw['trades']}.").strip()

        return Recommendation(
            found=True,
            strategy=strategy,
            confidence=confidence_used,
            expected_winrate=wr,
            n_trades=n,
            action=action,
            position_scale=scale,
            tier=tier,
            sample_reliability=sample_rel,
            notes=notes,
            source=str(self.json_path),
        )

    # ---- tactic-level lookup --------------------------------------------

    def tactic_winrate(self, strategy: str, tactic: str) -> Dict[str, Any]:
        """
        Look up win rate for a specific tactic within a strategy.
        Returns {'found': bool, 'trades': int, 'win_rate': float|None}.
        """
        if not self._loaded:
            return {"found": False, "trades": 0, "win_rate": None}

        tactics = self._data.get("per_tactic_winrates_top", {}).get(strategy, [])
        if not tactics and strategy == "cci_state":
            # cci_state top tactics are stored separately
            top = self._data.get("cci_top_tactics_by_winrate", {}).get("top_25", [])
            tactics = top

        for entry in tactics:
            if entry.get("tactic") == tactic:
                return {
                    "found": True,
                    "trades": int(entry.get("trades", 0)),
                    "win_rate": float(entry.get("win_rate", 0.0)),
                }
        return {"found": False, "trades": 0, "win_rate": None}

    def top_tactics(self, strategy: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Return top N tactics by win_rate for a strategy (n >= 10 only)."""
        if not self._loaded:
            return []
        tactics = self._data.get("per_tactic_winrates_top", {}).get(strategy, [])
        if not tactics and strategy == "cci_state":
            tactics = self._data.get("cci_top_tactics_by_sample_size", {}).get("top_25", [])
        filtered = [t for t in tactics if int(t.get("trades", 0)) >= 10]
        filtered.sort(key=lambda t: t.get("win_rate", 0), reverse=True)
        return filtered[:limit]

    # ---- private helpers -------------------------------------------------

    def _match_rule(self, strategy: str, confidence: str) -> Optional[Dict[str, Any]]:
        rules = self._data.get("confidence_to_action_map", {}).get("rules", [])
        for rule in rules:
            if rule.get("if_strategy") != strategy:
                continue
            rc = rule.get("if_confidence")
            if rc == confidence or rc == "any":
                return rule
        return None

    def _strategy_level_fallback(self, strategy: str) -> Optional[Tuple[float, int]]:
        """For strategies with only one confidence bucket, apply that WR to any confidence."""
        if not self._loaded:
            return None
        per = self._data.get("per_confidence_winrates", {}).get(strategy, {})
        if not per:
            return None
        # If exactly one confidence entry, return it as fallback
        if len(per) == 1:
            entry = next(iter(per.values()))
            wr = entry.get("win_rate")
            n = int(entry.get("trades", 0))
            if wr is not None and n > 0:
                return (float(wr), n)
        return None

    def _compute_action(self, wr: float, n: int) -> Tuple[str, float, str, str]:
        """Apply global_thresholds to (win_rate, sample_size) -> (action, scale, tier, notes)."""
        thr = self._data.get("global_thresholds", {})
        min_n_reliable = int(thr.get("min_trades_for_reliability", 10))
        wr_use = float(thr.get("winrate_use_threshold", 0.50))
        wr_caution = float(thr.get("winrate_caution_threshold", 0.40))

        if n < min_n_reliable:
            # Tiny sample — even if WR looks good, treat as caution
            if wr >= wr_use:
                return ("caution", 0.5, "caution",
                        f"WR={wr:.1%} looks good but n={n}<{min_n_reliable} is too small to trust fully.")
            return ("skip", 0.0, "disable",
                    f"WR={wr:.1%} and n={n}<{min_n_reliable} — insufficient evidence.")

        if wr >= wr_use:
            scale = 1.0 if wr >= 0.60 else 0.7
            return ("trust", scale, "use",
                    f"WR={wr:.1%} with n={n} — safe to trade.")

        if wr >= wr_caution:
            return ("caution", 0.4, "caution",
                    f"WR={wr:.1%} with n={n} — borderline, reduce size.")

        return ("skip", 0.0, "disable",
                f"WR={wr:.1%} with n={n} — below {wr_caution:.0%} threshold, skip.")

    def _reliability_from_n(self, n: int) -> str:
        if n >= 30:
            return "high_n"
        if n >= 10:
            return "med_n"
        if n > 0:
            return "low_n"
        return "no_data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tier_from_action(action: str) -> str:
    return {
        "trust":   "use",
        "caution": "caution",
        "skip":    "disable",
    }.get(action, "no_data")


def _fail_closed(strategy: str, confidence: str, source: str, notes: str) -> Recommendation:
    return Recommendation(
        found=False,
        strategy=strategy,
        confidence=confidence,
        expected_winrate=None,
        n_trades=0,
        action="skip",
        position_scale=0.0,
        tier="no_data",
        sample_reliability="no_data",
        notes=notes,
        source=source,
    )


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_SINGLETON: Optional[ConfidenceWinrateLookup] = None


def get_lookup() -> ConfidenceWinrateLookup:
    """Return a process-wide singleton. Safe to call from anywhere."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = ConfidenceWinrateLookup()
    return _SINGLETON


def reload_lookup(json_path: Optional[Path] = None) -> ConfidenceWinrateLookup:
    """Force a fresh load (useful after regenerating the JSON)."""
    global _SINGLETON
    _SINGLETON = ConfidenceWinrateLookup(json_path=json_path)
    return _SINGLETON


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> int:
    p = argparse.ArgumentParser(
        prog="confidence_winrate_lookup",
        description="Look up historical win rate by strategy + confidence level.",
    )
    p.add_argument("--strategy", help="Strategy name (e.g. ict_amd)")
    p.add_argument("--confidence", choices=["High", "Medium", "Low", "any"],
                   help="Confidence level")
    p.add_argument("--tactic", help="Optional tactic name for tactic-level lookup")
    p.add_argument("--tactic-lookup", nargs=2, metavar=("STRATEGY", "TACTIC"),
                   help="Look up win rate for a specific tactic")
    p.add_argument("--list-strategies", action="store_true",
                   help="List all strategies in the JSON")
    p.add_argument("--top-tactics", metavar="STRATEGY", type=int, default=0,
                   help="Show top N tactics by win_rate for STRATEGY")
    p.add_argument("--summary", action="store_true",
                   help="Print JSON metadata + thresholds")
    p.add_argument("--all", action="store_true",
                   help="Print full per-confidence table for every strategy")
    args = p.parse_args()

    lw = get_lookup()

    if not lw.is_loaded:
        print(f"[ERROR] JSON not loaded. Expected at: {_DEFAULT_JSON_PATH}", file=sys.stderr)
        return 2

    if args.summary:
        print(json.dumps(lw.summary(), indent=2, ensure_ascii=False))
        return 0

    if args.list_strategies:
        for s in lw.list_strategies():
            print(s)
        return 0

    if args.tactic_lookup:
        strategy, tactic = args.tactic_lookup
        result = lw.tactic_winrate(strategy, tactic)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["found"] else 1

    if args.top_tactics:
        tactics = lw.top_tactics(args.top_tactics, limit=20)
        if not tactics:
            print(f"No tactics found for '{args.top_tactics}'", file=sys.stderr)
            return 1
        print(f"Top tactics for {args.top_tactics} (n >= 10):")
        print(f"  {'Tactic':<30} {'Trades':>8} {'WinRate':>10}")
        print(f"  {'-' * 30} {'-' * 8} {'-' * 10}")
        for t in tactics:
            print(f"  {t['tactic']:<30} {t['trades']:>8} {t['win_rate']:>10.1%}")
        return 0

    if args.all:
        print("\n=== Per-Confidence Win Rates (all strategies) ===\n")
        print(f"  {'Strategy':<22} {'Confidence':<12} {'Trades':>8} {'WinRate':>10} {'Tier':<10} {'Reliability':<12}")
        print(f"  {'-' * 22} {'-' * 12} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 12}")
        for strat in lw.list_strategies():
            for conf in lw.list_confidences(strat):
                wr = lw.winrate(strat, conf)
                n = lw.trades(strat, conf)
                tier = lw.tier(strat, conf)
                rel = lw.sample_reliability(strat, conf)
                wr_s = f"{wr:.1%}" if wr is not None else "—"
                print(f"  {strat:<22} {conf:<12} {n:>8} {wr_s:>10} {tier:<10} {rel:<12}")
        return 0

    if args.strategy and args.confidence:
        rec = lw.recommend(args.strategy, args.confidence, tactic=args.tactic)
        print(json.dumps(rec.to_dict(), indent=2, ensure_ascii=False))
        return 0 if rec.action != "skip" else 1

    # No args -> print help
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
