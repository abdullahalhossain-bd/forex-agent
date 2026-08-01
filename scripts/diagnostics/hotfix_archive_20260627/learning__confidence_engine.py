# learning/confidence_engine.py  —  Day 53 | Dynamic Confidence Scoring ⭐⭐⭐⭐⭐
# ============================================================
# Day 53 Core — AI এখন সব pattern-কে সমান গুরুত্ব দেবে না।
#
# আগে:  Hammer detected + RSI oversold → BUY (confidence 75)
# এখন:  Hammer detected
#           ↓
#        Historical: EURUSD M15 Hammer = 65% win
#           ↓
#        Recent 10:  30% win (market changed)
#           ↓
#        Bayesian:   small sample? uncertainty high
#           ↓
#        Time decay: 6mo old data = 20% weight
#           ↓
#        Final confidence: 47 → WAIT
#
# 10/10 Features:
#   ⭐ Bayesian Updating       — small sample = lower confidence
#   ⭐ Minimum Sample Rule     — 30 trades before pattern confidence updates
#   ⭐ Market Regime Memory    — Hammer TRENDING=72% vs RANGING=35%
#   ⭐ Confidence Decay        — old data loses weight over time
#   ⭐ Last-10 Trade Adjustment — recent performance overrides historical
#   ⭐ Pattern Skip System     — disable pattern if win rate < 30%
#   ⭐ Confidence Calibration  — stated 80% but actual 55%? rescale
#   ⭐ Pattern Weight Store    — persistent weights per pattern/pair/tf/regime
# ============================================================

import json
import math
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from utils.logger import get_logger

log = get_logger("learning.confidence_engine")


def _test_mode() -> bool:
    """Lazy check for TEST_MODE flag. Returns False if config isn't
    importable (e.g. during unit tests)."""
    try:
        from config import TEST_MODE
        return bool(TEST_MODE)
    except Exception:
        return False

# ── Storage ──────────────────────────────────────────────────
PATTERN_STATS_PATH   = "memory/pattern_stats.json"
CONFIDENCE_HIST_PATH = "memory/confidence_history.json"
DISABLED_PATTERNS_PATH = "memory/disabled_patterns.json"

# ── Constants ─────────────────────────────────────────────────
MIN_SAMPLE_SIZE       = 5     # Lowered from 30 to allow early trading (learning phase)
SKIP_THRESHOLD        = 30.0  # win rate এর নিচে গেলে pattern disable
RECENT_WINDOW         = 10    # last N trades
DECAY_HALF_LIFE_DAYS  = 90    # 90 দিন পুরনো data → 50% weight

# ── Weights for final confidence formula ──────────────────────
W_HISTORICAL = 0.50
W_RECENT     = 0.30
W_REGIME     = 0.20


class ConfidenceEngine:
    """
    Day 53 — Pattern + Context ভিত্তিক Dynamic Confidence Scoring।

    Usage:
        engine = ConfidenceEngine()

        # Trade analysis-এর আগে:
        score = engine.calculate(
            pattern="Hammer",
            pair="EURUSD",
            timeframe="M15",
            regime="TRENDING",
        )
        print(score["final_confidence"])  # 65
        print(score["should_skip"])       # False
        print(score["reason"])            # "Historical 72%, Recent 60%, Regime good"

        # Trade close হওয়ার পর record করো:
        engine.record_outcome(
            pattern="Hammer", pair="EURUSD", timeframe="M15",
            regime="TRENDING", outcome="WIN", confidence_used=65,
        )
    """

    def __init__(self):
        os.makedirs("memory", exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # MAIN: CALCULATE CONFIDENCE
    # ══════════════════════════════════════════════════════════

    def calculate(
        self,
        pattern: str,
        pair: str = "EURUSD",
        timeframe: str = "M15",
        regime: str = "UNKNOWN",
        base_confidence: int = 70,
    ) -> dict:
        """
        Pattern + Context থেকে final confidence score বের করো।

        Returns:
        {
            "final_confidence": 61,
            "historical_score": 70,
            "recent_score": 40,
            "regime_score": 80,
            "bayesian_penalty": -8,
            "decay_factor": 0.9,
            "sample_size": 45,
            "should_skip": False,
            "skip_reason": None,
            "adjustment": -14,
            "reason": "...",
            "components": {...}
        }
        """
        key = self._key(pattern, pair, timeframe, regime)
        stats = self._load_stats()
        entry = stats.get(key, {})

        # 1. Historical win rate (time-decay weighted)
        historical_score, decay_factor, sample_size = self._get_historical_score(entry)

        # 2. Recent 10 trades adjustment
        recent_score = self._get_recent_score(entry)

        # 3. Market regime score
        regime_score = self._get_regime_score(pattern, regime, stats)

        # 4. Bayesian penalty (small sample = lower confidence)
        bayesian_penalty = self._bayesian_penalty(sample_size)

        # 5. Final weighted formula
        raw_final = (
            historical_score * W_HISTORICAL
            + recent_score   * W_RECENT
            + regime_score   * W_REGIME
        )
        final_confidence = max(5, min(95, round(raw_final + bayesian_penalty)))

        # 6. Pattern skip check
        should_skip, skip_reason = self._check_skip(pattern, pair, timeframe, regime, entry, recent_score)

        # 7. Adjustment from base
        adjustment = final_confidence - base_confidence

        # 8. Build reason string
        reason = self._build_reason(
            historical_score, recent_score, regime_score,
            bayesian_penalty, sample_size, decay_factor, should_skip
        )

        result = {
            "pattern":           pattern,
            "pair":              pair,
            "timeframe":         timeframe,
            "regime":            regime,
            "base_confidence":   base_confidence,
            "final_confidence":  final_confidence,
            "adjustment":        adjustment,
            "historical_score":  round(historical_score, 1),
            "recent_score":      round(recent_score, 1),
            "regime_score":      round(regime_score, 1),
            "bayesian_penalty":  round(bayesian_penalty, 1),
            "decay_factor":      round(decay_factor, 2),
            "sample_size":       sample_size,
            "should_skip":       should_skip,
            "skip_reason":       skip_reason,
            "reason":            reason,
            "components": {
                "historical": f"{round(historical_score,1)} × {W_HISTORICAL}",
                "recent":     f"{round(recent_score,1)} × {W_RECENT}",
                "regime":     f"{round(regime_score,1)} × {W_REGIME}",
                "bayesian":   bayesian_penalty,
            },
        }

        # Log
        icon = "⛔" if should_skip else ("⚠️" if final_confidence < 50 else "✅")
        log.info(
            f"[ConfidenceEngine] {icon} {pattern} | {pair} {timeframe} {regime} | "
            f"hist={historical_score:.0f} recent={recent_score:.0f} regime={regime_score:.0f} "
            f"→ final={final_confidence:.1f} (adj={adjustment:+.1f})"
        )

        return result

    # ══════════════════════════════════════════════════════════
    # RECORD TRADE OUTCOME
    # ══════════════════════════════════════════════════════════

    def record_outcome(
        self,
        pattern: str,
        pair: str,
        timeframe: str,
        regime: str,
        outcome: str,         # "WIN" | "LOSS" | "BE"
        confidence_used: int = None,
        pnl: float = None,
    ) -> None:
        """
        Trade result record করো — confidence system এটা দিয়ে শিখবে।
        """
        stats = self._load_stats()
        key = self._key(pattern, pair, timeframe, regime)

        if key not in stats:
            stats[key] = self._empty_entry(pattern, pair, timeframe, regime)

        e = stats[key]
        e["total_trades"] += 1
        is_win = outcome == "WIN"
        if is_win:
            e["wins"] += 1
        elif outcome == "LOSS":
            e["losses"] += 1

        e["win_rate"] = round(e["wins"] / e["total_trades"] * 100, 1)
        e["last_updated"] = datetime.now(timezone.utc).isoformat()

        # Recent 10 trades ring buffer
        recent = e.get("recent_results", [])
        recent.append({
            "outcome":    outcome,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "confidence": confidence_used,
            "pnl":        pnl,
        })
        e["recent_results"] = recent[-RECENT_WINDOW:]

        # Pattern weight update (exponential moving average)
        alpha = 0.1
        e["weight"] = round(
            (1 - alpha) * e.get("weight", 0.5) + alpha * (1.0 if is_win else 0.0),
            4
        )

        stats[key] = e
        self._save_stats(stats)

        # Also save to confidence history
        self._record_history(pattern, pair, timeframe, regime, outcome, confidence_used, pnl)

        # Check if pattern should be disabled
        self._update_disabled_list(pattern, pair, timeframe, regime, e)

        log.info(
            f"[ConfidenceEngine] Recorded {outcome} | {key} | "
            f"win_rate={e['win_rate']}% ({e['wins']}/{e['total_trades']})"
        )

    # ══════════════════════════════════════════════════════════
    # COMPONENT CALCULATIONS
    # ══════════════════════════════════════════════════════════

    def _get_historical_score(self, entry: dict) -> tuple:
        """
        Time-decay weighted historical win rate。

        গত ৯০ দিনে যত ট্রেড, তত বেশি weight।
        পুরনো ট্রেড exponentially কম গুরুত্ব পায়।
        """
        if not entry or entry.get("total_trades", 0) == 0:
            return 50.0, 1.0, 0  # no data → neutral 50%

        total   = entry.get("total_trades", 0)
        raw_wr  = entry.get("win_rate", 50.0)

        # Time decay
        last_updated = entry.get("last_updated")
        decay = 1.0
        if last_updated:
            try:
                days_old = (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(last_updated)
                ).days
                # Exponential decay: weight = 0.5^(days/half_life)
                decay = max(0.2, 0.5 ** (days_old / DECAY_HALF_LIFE_DAYS))
            except Exception:
                decay = 1.0

        # Apply decay: blend toward 50% (neutral) as data ages
        decayed_wr = 50.0 + (raw_wr - 50.0) * decay

        return decayed_wr, decay, total

    def _get_recent_score(self, entry: dict) -> float:
        """
        Last 10 trades win rate — recent market behavior。

        W L L L W L L W L L → 30% → confidence drops significantly
        """
        recent = entry.get("recent_results", [])
        if not recent:
            return 50.0  # no recent data → neutral

        wins = sum(1 for r in recent if r.get("outcome") == "WIN")
        return round(wins / len(recent) * 100, 1)

    def _get_regime_score(self, pattern: str, regime: str, all_stats: dict) -> float:
        """
        এই pattern এই regime-এ কতটা ভালো perform করে?

        Hammer TRENDING = 72%, RANGING = 35%
        সব pair+tf-এর average নিয়ে regime-specific score।
        """
        regime_wins  = 0
        regime_total = 0

        for key, entry in all_stats.items():
            parts = key.split("|")
            if len(parts) == 4 and parts[0] == pattern and parts[3] == regime:
                regime_wins  += entry.get("wins", 0)
                regime_total += entry.get("total_trades", 0)

        if regime_total == 0:
            # Default scores per regime type
            defaults = {
                "TRENDING":    70.0,
                "RANGING":     40.0,
                "VOLATILE":    45.0,
                "HIGH_VOLATILITY": 35.0,
                "UNKNOWN":     50.0,
            }
            return defaults.get(regime.upper(), 50.0)

        return round(regime_wins / regime_total * 100, 1)

    def _bayesian_penalty(self, sample_size: int) -> float:
        """
        Bayesian uncertainty penalty — কম data = কম confidence।

        Logic: uncertainty ∝ 1/√(sample_size)
        At MIN_SAMPLE_SIZE (30): no penalty
        At 10 trades:  -10 penalty
        At 5 trades:   -15 penalty
        At 0 trades:   -20 penalty

        Day 81+ hotfix: in TEST_MODE, the penalty is halved so signals
        can actually pass through during the early-learning phase.
        Without this, a fresh bot with 0 trades gets -20 on every
        signal, dropping BUY 71% → 51% → below threshold → NO TRADE.
        The bot then never trades, never builds a sample, and stays
        stuck at -20 forever. Halving the penalty breaks that loop.
        """
        if sample_size >= MIN_SAMPLE_SIZE:
            return 0.0

        if sample_size == 0:
            base_penalty = -20.0
        else:
            base_penalty = -20.0 * (1 - math.sqrt(sample_size / MIN_SAMPLE_SIZE))

        # Day 81+ — halve the penalty in TEST_MODE
        try:
            from config import TEST_MODE
            if TEST_MODE:
                base_penalty = base_penalty / 2
        except Exception:
            pass

        return round(base_penalty, 1)

    def _check_skip(
        self, pattern: str, pair: str, timeframe: str, regime: str,
        entry: dict, recent_score: float
    ) -> tuple:
        """
        Pattern skip system — win rate < 30% হলে temporarily disable।
        """
        disabled = self._load_disabled()
        key = self._key(pattern, pair, timeframe, regime)

        # Already disabled?
        if key in disabled:
            d = disabled[key]
            # Auto-re-enable after 50 trades or 30 days
            re_enable_date = d.get("re_enable_after")
            if re_enable_date:
                try:
                    if datetime.now(timezone.utc).isoformat() > re_enable_date:
                        del disabled[key]
                        self._save_disabled(disabled)
                        log.info(f"[ConfidenceEngine] Pattern re-enabled: {key}")
                        return False, None
                except Exception:
                    pass
            return True, d.get("reason", "Pattern disabled due to poor performance")

        # Check current stats
        total    = entry.get("total_trades", 0)
        win_rate = entry.get("win_rate", 50.0)

        if total >= 20 and win_rate < SKIP_THRESHOLD:
            return True, f"Win rate {win_rate}% below threshold {SKIP_THRESHOLD}% over {total} trades"

        if total >= 10 and recent_score < 20:
            return True, f"Recent 10-trade win rate critically low: {recent_score}%"

        return False, None

    # ══════════════════════════════════════════════════════════
    # PATTERN SKIP MANAGEMENT
    # ══════════════════════════════════════════════════════════

    def disable_pattern(
        self, pattern: str, pair: str, timeframe: str, regime: str,
        reason: str = "Manual disable", days: int = 30
    ) -> dict:
        """Pattern manually disable করো।"""
        key = self._key(pattern, pair, timeframe, regime)
        disabled = self._load_disabled()

        re_enable = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
        disabled[key] = {
            "reason":           reason,
            "disabled_at":      datetime.now(timezone.utc).isoformat(),
            "re_enable_after":  re_enable,
        }
        self._save_disabled(disabled)
        log.warning(f"[ConfidenceEngine] ⛔ Pattern DISABLED: {key} | Reason: {reason}")
        return {"disabled": True, "key": key, "re_enable_after": re_enable}

    def enable_pattern(self, pattern: str, pair: str, timeframe: str, regime: str) -> dict:
        """Pattern manually re-enable করো।"""
        key = self._key(pattern, pair, timeframe, regime)
        disabled = self._load_disabled()
        if key in disabled:
            del disabled[key]
            self._save_disabled(disabled)
            log.info(f"[ConfidenceEngine] ✅ Pattern RE-ENABLED: {key}")
            return {"enabled": True, "key": key}
        return {"enabled": False, "reason": "Pattern was not disabled"}

    def get_disabled_patterns(self) -> dict:
        return self._load_disabled()

    def _update_disabled_list(
        self, pattern: str, pair: str, timeframe: str, regime: str, entry: dict
    ) -> None:
        """win rate দেখে auto-disable।"""
        total    = entry.get("total_trades", 0)
        win_rate = entry.get("win_rate", 50.0)

        if total >= 20 and win_rate < SKIP_THRESHOLD:
            key = self._key(pattern, pair, timeframe, regime)
            disabled = self._load_disabled()
            if key not in disabled:
                self.disable_pattern(
                    pattern, pair, timeframe, regime,
                    reason=f"Auto-disabled: win rate {win_rate}% < {SKIP_THRESHOLD}% threshold",
                    days=14
                )

    # ══════════════════════════════════════════════════════════
    # CONFIDENCE CALIBRATION
    # ══════════════════════════════════════════════════════════

    def calibrate(self, pattern: str = None, pair: str = None) -> dict:
        """
        Stated confidence vs actual win rate — gap check।

        Example:
            Stated 80% confidence → actual 55% win rate
            Gap: 25% → overconfident → rescale factor = 55/80 = 0.69
        """
        history = self._load_history()

        if pattern:
            history = [h for h in history if h.get("pattern") == pattern]
        if pair:
            history = [h for h in history if h.get("pair") == pair]

        if not history:
            return {"calibrated": False, "reason": "No history data"}

        # Group by confidence bucket (60-70, 70-80, etc.)
        buckets = {}
        for h in history:
            conf = h.get("confidence_used")
            if conf is None:
                continue
            bucket = f"{(conf // 10) * 10}-{(conf // 10) * 10 + 10}"
            if bucket not in buckets:
                buckets[bucket] = {"stated": 0, "wins": 0, "total": 0}
            buckets[bucket]["stated"] = (conf // 10) * 10 + 5
            buckets[bucket]["total"] += 1
            if h.get("outcome") == "WIN":
                buckets[bucket]["wins"] += 1

        # Calibration report
        calibration = {}
        for bucket, data in buckets.items():
            if data["total"] < 5:
                continue
            actual = round(data["wins"] / data["total"] * 100, 1)
            stated = data["stated"]
            gap    = stated - actual
            calibration[bucket] = {
                "stated_confidence": stated,
                "actual_win_rate":   actual,
                "gap":               gap,
                "sample":            data["total"],
                "status":            (
                    "OVERCONFIDENT" if gap > 15 else
                    "UNDERCONFIDENT" if gap < -10 else
                    "CALIBRATED"
                ),
            }

        overall_overconf = sum(
            1 for v in calibration.values() if v["status"] == "OVERCONFIDENT"
        )

        return {
            "calibrated":        True,
            "buckets":           calibration,
            "overall_bias":      "OVERCONFIDENT" if overall_overconf > len(calibration) / 2 else "OK",
            "recommendation":    "Reduce confidence scores by 15-20%" if overall_overconf > 1 else "Calibration acceptable",
            "pattern":           pattern,
            "total_trades_analyzed": len(history),
        }

    # ══════════════════════════════════════════════════════════
    # PATTERN STATS REPORT
    # ══════════════════════════════════════════════════════════

    def get_all_pattern_stats(self) -> dict:
        """Full pattern performance database।"""
        return self._load_stats()

    def get_pattern_summary(self, pattern: str = None) -> list:
        """Human-readable pattern stats।"""
        stats = self._load_stats()
        rows  = []
        for key, entry in stats.items():
            if pattern and not key.startswith(pattern):
                continue
            rows.append({
                "key":         key,
                "win_rate":    entry.get("win_rate", 0),
                "total":       entry.get("total_trades", 0),
                "weight":      entry.get("weight", 0.5),
                "recent_win":  self._get_recent_score(entry),
            })
        return sorted(rows, key=lambda x: x["win_rate"], reverse=True)

    # ══════════════════════════════════════════════════════════
    # DECISION AGENT INTEGRATION
    # ══════════════════════════════════════════════════════════

    def adjust_decision(
        self,
        signal: str,
        base_confidence: int,
        pattern: str,
        pair: str,
        timeframe: str,
        regime: str,
    ) -> dict:
        """
        DecisionAgent-এ call করার জন্য — signal + confidence adjust করে।

        Returns:
        {
            "signal": "BUY" | "SELL" | "NO TRADE",
            "base_confidence": 75,
            "pattern_adjustment": -15,
            "final_confidence": 60,
            "decision": "WAIT",   # if confidence drops below threshold
            "should_skip": False,
        }
        """
        score = self.calculate(pattern, pair, timeframe, regime, base_confidence)

        final_conf = score["final_confidence"]
        should_skip = score["should_skip"]

        # Decision logic
        # TEST_MODE: disable auto-skip entirely (so the system always emits
        # a tradeable signal), and lower the WAIT threshold to 10 so
        # confidence >= 10 flows through to MT5.
        # Production: should_skip respected, WAIT threshold = 25.
        if should_skip and not _test_mode():
            final_signal = "NO TRADE"
            decision = "SKIP"
        elif final_conf < (10 if _test_mode() else 25):
            final_signal = "WAIT"
            decision = "WAIT"
        else:
            final_signal = signal
            decision = signal

        return {
            "signal":             signal,
            "base_confidence":    base_confidence,
            "pattern_adjustment": score["adjustment"],
            "final_confidence":   final_conf,
            "final_signal":       final_signal,
            "decision":           decision,
            "should_skip":        should_skip,
            "skip_reason":        score.get("skip_reason"),
            "reason":             score["reason"],
            "components":         score["components"],
        }

    # ══════════════════════════════════════════════════════════
    # PRINT SUMMARIES
    # ══════════════════════════════════════════════════════════

    def print_score(self, result: dict) -> None:
        bar = "═" * 60
        icon = "⛔" if result.get("should_skip") else ("⚠️" if result["final_confidence"] < 50 else "✅")
        print(f"\n{bar}")
        print("  🎯  CONFIDENCE ENGINE  (Day 53)")
        print(bar)
        print(f"  {icon}  Pattern     : {result['pattern']}")
        print(f"     Pair/TF/Regime : {result['pair']} {result['timeframe']} {result['regime']}")
        print()
        print(f"  ── Score Components ──")
        print(f"  Historical ({W_HISTORICAL*100:.0f}%) : {result['historical_score']}%  (decay={result['decay_factor']}x, n={result['sample_size']})")
        print(f"  Recent-10  ({W_RECENT*100:.0f}%) : {result['recent_score']}%")
        print(f"  Regime     ({W_REGIME*100:.0f}%) : {result['regime_score']}%")
        print(f"  Bayesian penalty      : {result['bayesian_penalty']:+.1f}")
        print()
        print(f"  Base confidence  : {result['base_confidence']}")
        print(f"  Adjustment       : {result['adjustment']:+.1f}")
        print(f"  Final confidence : {result['final_confidence']}")
        if result.get("should_skip"):
            print(f"\n  ⛔ SKIP: {result['skip_reason']}")
        print(f"\n  {result['reason']}")
        print(bar + "\n")

    def print_all_stats(self) -> None:
        rows = self.get_pattern_summary()
        bar = "═" * 68
        print(f"\n{bar}")
        print("  📊  PATTERN CONFIDENCE DATABASE  (Day 53)")
        print(bar)
        print(f"  {'Key':<40} {'WinRate':>8} {'Total':>7} {'Recent':>8} {'Weight':>7}")
        print(f"  {'-'*40} {'-'*8} {'-'*7} {'-'*8} {'-'*7}")
        for r in rows:
            verdict = "✅" if r["win_rate"] >= 60 else "⚠️" if r["win_rate"] >= 45 else "⛔"
            print(f"  {verdict} {r['key'][:38]:<38} {r['win_rate']:>7.1f}% {r['total']:>7} {r['recent_win']:>7.1f}% {r['weight']:>7.3f}")
        print(bar + "\n")

    def print_calibration(self, result: dict) -> None:
        bar = "═" * 56
        print(f"\n{bar}")
        print("  🔬  CONFIDENCE CALIBRATION REPORT  (Day 53)")
        print(bar)
        if not result.get("calibrated"):
            print(f"  {result.get('reason', 'No data')}")
        else:
            print(f"  Overall bias : {result['overall_bias']}")
            print(f"  Trades analyzed: {result['total_trades_analyzed']}")
            print(f"  Recommendation: {result['recommendation']}")
            print()
            for bucket, data in result["buckets"].items():
                status_icon = "🔴" if data["status"] == "OVERCONFIDENT" else "🟢" if data["status"] == "CALIBRATED" else "🟡"
                print(f"  {status_icon} Conf {bucket}% → Actual {data['actual_win_rate']}% | Gap={data['gap']:+.1f}% | n={data['sample']}")
        print(bar + "\n")

    # ══════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════

    def _build_reason(
        self, hist: float, recent: float, regime: float,
        bayes: float, sample: int, decay: float, skip: bool
    ) -> str:
        parts = []

        if sample < MIN_SAMPLE_SIZE:
            parts.append(f"⚠️ Small sample ({sample}/{MIN_SAMPLE_SIZE}) — Bayesian penalty {bayes:.0f}")
        if decay < 0.7:
            parts.append(f"📅 Data aged ({decay:.2f}x decay)")
        if recent < 40:
            parts.append(f"📉 Recent 10 poor ({recent:.0f}%)")
        elif recent > 65:
            parts.append(f"📈 Recent 10 strong ({recent:.0f}%)")
        if regime < 40:
            parts.append(f"🚫 Regime unfavorable ({regime:.0f}%)")
        elif regime > 65:
            parts.append(f"✅ Regime favorable ({regime:.0f}%)")
        if skip:
            parts.append("⛔ Pattern disabled")

        if not parts:
            parts.append(f"Historical {hist:.0f}% | Recent {recent:.0f}% | Regime {regime:.0f}%")

        return " | ".join(parts)

    def _key(self, pattern: str, pair: str, timeframe: str, regime: str) -> str:
        return f"{pattern}|{pair}|{timeframe}|{regime}".replace(" ", "_")

    def _empty_entry(self, pattern, pair, timeframe, regime) -> dict:
        return {
            "pattern":        pattern,
            "pair":           pair,
            "timeframe":      timeframe,
            "market_regime":  regime,
            "total_trades":   0,
            "wins":           0,
            "losses":         0,
            "win_rate":       50.0,
            "weight":         0.5,
            "recent_results": [],
            "last_updated":   datetime.now(timezone.utc).isoformat(),
        }

    # ── Storage ───────────────────────────────────────────────

    def _load_stats(self) -> dict:
        if not os.path.exists(PATTERN_STATS_PATH):
            return {}
        try:
            with open(PATTERN_STATS_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_stats(self, data: dict) -> None:
        with open(PATTERN_STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_history(self) -> list:
        if not os.path.exists(CONFIDENCE_HIST_PATH):
            return []
        try:
            with open(CONFIDENCE_HIST_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _record_history(
        self, pattern, pair, timeframe, regime,
        outcome, confidence_used, pnl
    ) -> None:
        history = self._load_history()
        history.append({
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "pattern":         pattern,
            "pair":            pair,
            "timeframe":       timeframe,
            "regime":          regime,
            "outcome":         outcome,
            "confidence_used": confidence_used,
            "pnl":             pnl,
        })
        with open(CONFIDENCE_HIST_PATH, "w", encoding="utf-8") as f:
            json.dump(history[-2000:], f, indent=2, default=str)

    def _load_disabled(self) -> dict:
        if not os.path.exists(DISABLED_PATTERNS_PATH):
            return {}
        try:
            with open(DISABLED_PATTERNS_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_disabled(self, data: dict) -> None:
        with open(DISABLED_PATTERNS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)