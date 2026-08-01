# research/hypothesis_engine.py — Day 57 | Hypothesis Generator
# ============================================================
# AI নিজে প্রশ্ন তৈরি করে, testable hypothesis বানায়,
# এবং experiment design করে।
#
# Example:
#   Hypothesis: "ATR filter improves SMC strategy win rate"
#   Experiment: Strategy A (SMC only) vs Strategy B (SMC + ATR)
#   Result: +14% profit factor → CONFIRMED
# ============================================================

import random
from datetime import datetime, timezone
from typing import Optional

from utils.logger import get_logger

log = get_logger("research.hypothesis_engine")


# ── Hypothesis Templates ───────────────────────────────────────
HYPOTHESIS_TEMPLATES = [
    {
        "question": "Does adding {filter} improve {strategy} win rate?",
        "type": "filter_addition",
        "control": "base_strategy",
        "treatment": "base_with_filter",
    },
    {
        "question": "Does removing {filter} from {strategy} reduce drawdown?",
        "type": "filter_removal",
        "control": "full_strategy",
        "treatment": "without_filter",
    },
    {
        "question": "Is {pair} better for {strategy} than {pair2}?",
        "type": "pair_comparison",
        "control": "pair_a",
        "treatment": "pair_b",
    },
    {
        "question": "Does {session} session improve {strategy} expectancy?",
        "type": "session_filter",
        "control": "all_sessions",
        "treatment": "session_only",
    },
    {
        "question": "Does changing ATR period from {period_a} to {period_b} improve entries?",
        "type": "parameter_optimization",
        "control": "param_a",
        "treatment": "param_b",
    },
    {
        "question": "Does {timeframe} timeframe give better signals than {timeframe2}?",
        "type": "timeframe_comparison",
        "control": "tf_a",
        "treatment": "tf_b",
    },
    {
        "question": "Does combining {entry_a} with {entry_b} create synergy?",
        "type": "entry_combination",
        "control": "single_entry",
        "treatment": "combined_entries",
    },
    {
        "question": "Does widening SL to {sl_mult}x ATR improve profit factor?",
        "type": "sl_optimization",
        "control": "current_sl",
        "treatment": "wider_sl",
    },
]

# ── Variable Pools ───────────────────────────────────────────────
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
SESSIONS = ["London", "New York", "London-NY Overlap", "Tokyo", "Asian"]
TIMEFRAMES = ["M5", "M15", "H1", "H4"]
ATR_PERIODS = [10, 14, 20, 28]
SL_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5]


class Hypothesis:
    """
    Represents a single testable hypothesis.

    Attributes:
        id: Unique hypothesis identifier
        question: Human-readable question
        hypothesis_type: Type of hypothesis (filter_addition, parameter_optimization, etc.)
        control: Control group description
        treatment: Treatment group description
        variables: Dict of variables used
        created_at: Timestamp
    """

    _counter: int = 0

    def __init__(
        self,
        question: str,
        hypothesis_type: str,
        control: str,
        treatment: str,
        variables: Optional[dict] = None,
    ):
        Hypothesis._counter += 1
        self.id = f"HYP-{Hypothesis._counter:04d}"
        self.question = question
        self.hypothesis_type = hypothesis_type
        self.control = control
        self.treatment = treatment
        self.variables = variables or {}
        self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.result: Optional[dict] = None
        self.status = "PENDING"  # PENDING, TESTING, CONFIRMED, REJECTED, INCONCLUSIVE

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "hypothesis_type": self.hypothesis_type,
            "control": self.control,
            "treatment": self.treatment,
            "variables": self.variables,
            "created_at": self.created_at,
            "status": self.status,
            "result": self.result,
        }

    def set_result(self, result: dict) -> None:
        """Set the experiment result and update status."""
        self.result = result
        improvement = result.get("improvement_pct", 0)
        confidence = result.get("confidence", 0)
        p_value = result.get("p_value", 1.0)

        if improvement > 5 and confidence >= 70 and p_value < 0.05:
            self.status = "CONFIRMED"
        elif improvement > -5 and confidence < 60:
            self.status = "INCONCLUSIVE"
        else:
            self.status = "REJECTED"

    def __repr__(self) -> str:
        return f"Hypothesis({self.id}: {self.question[:50]}...)"


class HypothesisEngine:
    """
    AI Hypothesis Generation Engine.

    Usage:
        engine = HypothesisEngine()
        hypothesis = engine.generate(strategy_name="SMC_Basic_v1")
        hypothesis = engine.generate_from_market_observation(observation_dict)
        results = engine.evaluate_hypothesis(hypothesis, backtest_a, backtest_b)
    """

    MAX_HYPOTHESES_PER_CYCLE = 5

    def __init__(self):
        self._history: list[Hypothesis] = []

    # ═══════════════════════════════════════════════════════
    # HYPOTHESIS GENERATION
    # ═══════════════════════════════════════════════════════

    def generate(self, strategy_name: str = None) -> Hypothesis:
        """
        Generate a testable hypothesis based on a strategy or random template.
        """
        template = random.choice(HYPOTHESIS_TEMPLATES)
        variables = self._fill_variables(template)

        question = template["question"].format(**variables)

        hypothesis = Hypothesis(
            question=question,
            hypothesis_type=template["type"],
            control=template["control"],
            treatment=template["treatment"],
            variables={**variables, "strategy_name": strategy_name or "auto"},
        )

        self._history.append(hypothesis)
        log.info(
            f"[HypothesisEngine] Generated: {hypothesis.id} — {question[:80]}"
        )
        return hypothesis

    def generate_from_market_observation(self, observation: dict) -> Hypothesis:
        """
        Generate a hypothesis from a specific market observation.

        Example observation:
            {
                "pair": "EURUSD",
                "observation": "London session has higher directional movement",
                "evidence": {"avg_ATR_change": "+35%", "win_rate": "72%"}
            }
        """
        pair = observation.get("pair", "EURUSD")
        obs_text = observation.get("observation", "")
        evidence = observation.get("evidence", {})

        hypothesis_type = "market_observation"
        question = (
            f"Based on observation: '{obs_text}' for {pair}. "
            f"Can we create a profitable strategy leveraging this pattern?"
        )

        hypothesis = Hypothesis(
            question=question,
            hypothesis_type=hypothesis_type,
            control="no_strategy",
            treatment="observation_based_strategy",
            variables={
                "pair": pair,
                "observation": obs_text,
                "evidence": evidence,
            },
        )

        self._history.append(hypothesis)
        log.info(
            f"[HypothesisEngine] Observation-based: {hypothesis.id} — "
            f"{question[:80]}"
        )
        return hypothesis

    def generate_batch(self, n: int = 3, strategy_name: str = None) -> list[Hypothesis]:
        """Generate a batch of hypotheses."""
        batch = []
        for _ in range(min(n, self.MAX_HYPOTHESES_PER_CYCLE)):
            batch.append(self.generate(strategy_name=strategy_name))
        log.info(f"[HypothesisEngine] Generated {len(batch)} hypotheses")
        return batch

    # ═══════════════════════════════════════════════════════
    # HYPOTHESIS EVALUATION
    # ═══════════════════════════════════════════════════════

    def evaluate_hypothesis(
        self,
        hypothesis: Hypothesis,
        control_result: dict,
        treatment_result: dict,
    ) -> dict:
        """
        Evaluate a hypothesis by comparing control vs treatment backtest results.

        Args:
            hypothesis: The hypothesis to evaluate
            control_result: Backtest result for control group
            treatment_result: Backtest result for treatment group

        Returns:
            Evaluation result dict with improvement metrics.
        """
        ctrl_summary = control_result.get("summary", {})
        treat_summary = treatment_result.get("summary", {})

        ctrl_wr = ctrl_summary.get("win_rate", 0)
        treat_wr = treat_summary.get("win_rate", 0)
        ctrl_pf = ctrl_summary.get("profit_factor", 0)
        treat_pf = treat_summary.get("profit_factor", 0)
        ctrl_dd = ctrl_summary.get("max_drawdown", 0)
        treat_dd = treat_summary.get("max_drawdown", 0)
        ctrl_rr = ctrl_summary.get("average_rr", 0)
        treat_rr = treat_summary.get("average_rr", 0)
        ctrl_trades = ctrl_summary.get("trades", 0)
        treat_trades = treat_summary.get("trades", 0)

        wr_improvement = treat_wr - ctrl_wr if ctrl_wr > 0 else 0
        pf_improvement = ((treat_pf - ctrl_pf) / ctrl_pf * 100) if ctrl_pf > 0 else 0
        dd_change = treat_dd - ctrl_dd  # negative = improvement
        rr_improvement = treat_rr - ctrl_rr

        # Composite score
        improvement_pct = 0
        if ctrl_wr > 0:
            improvement_pct = round(pf_improvement * 0.4 + wr_improvement * 0.3 + (-dd_change) * 0.2 + rr_improvement * 10 * 0.1, 1)

        # Confidence based on sample size
        confidence = min(95, round(40 + min(ctrl_trades, treat_trades) * 0.15 + abs(pf_improvement) * 0.5))

        # Pseudo p-value (simplified)
        p_value = max(0.01, 1.0 - abs(improvement_pct) / 100)

        result = {
            "hypothesis_id": hypothesis.id,
            "control": {
                "win_rate": ctrl_wr,
                "profit_factor": ctrl_pf,
                "max_drawdown": ctrl_dd,
                "average_rr": ctrl_rr,
                "trades": ctrl_trades,
            },
            "treatment": {
                "win_rate": treat_wr,
                "profit_factor": treat_pf,
                "max_drawdown": treat_dd,
                "average_rr": treat_rr,
                "trades": treat_trades,
            },
            "improvement_pct": improvement_pct,
            "wr_improvement": wr_improvement,
            "pf_improvement": pf_improvement,
            "dd_change": dd_change,
            "rr_improvement": rr_improvement,
            "confidence": confidence,
            "p_value": round(p_value, 3),
            "verdict": "pending",
        }

        # Determine verdict
        if improvement_pct > 5 and confidence >= 70:
            result["verdict"] = "CONFIRMED"
        elif improvement_pct > -5 and confidence < 60:
            result["verdict"] = "INCONCLUSIVE"
        else:
            result["verdict"] = "REJECTED"

        hypothesis.set_result(result)

        log.info(
            f"[HypothesisEngine] {hypothesis.id} result: "
            f"improvement={improvement_pct:+.1f}% | "
            f"PF: {ctrl_pf:.2f} → {treat_pf:.2f} | "
            f"WR: {ctrl_wr:.1f}% → {treat_wr:.1f}% | "
            f"verdict={result['verdict']}"
        )
        return result

    # ═══════════════════════════════════════════════════════
    # MARKET BEHAVIOR RESEARCH
    # ═══════════════════════════════════════════════════════

    def analyze_market_behavior(self, pair: str, session_data: dict) -> dict:
        """
        Analyze market behavior for a specific pair + session combination.

        Args:
            pair: Currency pair (e.g., "EURUSD")
            session_data: Dict with volatility, liquidity, spread, win_rate, session movement data

        Returns:
            Research finding dict.
        """
        finding = {
            "pair": pair,
            "session": session_data.get("session", "unknown"),
            "finding": "",
            "evidence": {},
            "recommendation": "",
            "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        avg_atr = session_data.get("avg_ATR", 0)
        win_rate = session_data.get("win_rate", 0)
        spread = session_data.get("avg_spread", 0)
        movement = session_data.get("session_movement", 0)

        # Build finding
        if win_rate > 65 and avg_atr > 0:
            finding["finding"] = (
                f"{pair} during {session_data.get('session')} session shows "
                f"higher directional movement with good win rate."
            )
            finding["evidence"] = {
                "avg_ATR_change": f"+{avg_atr:.0f}%" if avg_atr > 0 else f"{avg_atr:.0f}%",
                "win_rate": f"{win_rate:.0f}%",
                "spread_avg": f"{spread:.1f} pips",
            }
            finding["recommendation"] = (
                f"Consider {session_data.get('session')} session filter for {pair} strategies."
            )
        elif win_rate < 45:
            finding["finding"] = (
                f"{pair} during {session_data.get('session')} session has "
                f"poor performance — likely choppy or low liquidity."
            )
            finding["evidence"] = {
                "avg_ATR_change": f"{avg_atr:.0f}%",
                "win_rate": f"{win_rate:.0f}%",
                "spread_avg": f"{spread:.1f} pips",
            }
            finding["recommendation"] = (
                f"Avoid {session_data.get('session')} session for {pair} or tighten filters."
            )
        else:
            finding["finding"] = (
                f"{pair} during {session_data.get('session')} session has neutral performance."
            )
            finding["evidence"] = session_data
            finding["recommendation"] = "No strong session bias detected — strategy-neutral."

        log.info(
            f"[HypothesisEngine] Market behavior: {pair} {session_data.get('session')} — "
            f"WR: {win_rate:.0f}%, ATR: {avg_atr:.0f}%"
        )
        return finding

    # ═══════════════════════════════════════════════════════
    # HISTORY & STATS
    # ═══════════════════════════════════════════════════════

    def get_history(self) -> list[dict]:
        """Return all hypotheses with their results."""
        return [h.to_dict() for h in self._history]

    def get_stats(self) -> dict:
        """Return hypothesis testing statistics."""
        total = len(self._history)
        confirmed = sum(1 for h in self._history if h.status == "CONFIRMED")
        rejected = sum(1 for h in self._history if h.status == "REJECTED")
        inconclusive = sum(1 for h in self._history if h.status == "INCONCLUSIVE")
        pending = sum(1 for h in self._history if h.status == "PENDING")

        return {
            "total_hypotheses": total,
            "confirmed": confirmed,
            "rejected": rejected,
            "inconclusive": inconclusive,
            "pending": pending,
            "confirmation_rate": round(confirmed / total * 100, 1) if total > 0 else 0,
        }

    # ═══════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════

    def _fill_variables(self, template: dict) -> dict:
        """Fill template variables with random values."""
        variables = {}

        if "{filter}" in template["question"]:
            from research.strategy_generator import FILTER_COMPONENTS
            variables["filter"] = random.choice(FILTER_COMPONENTS)
        if "{strategy}" in template["question"]:
            variables["strategy"] = random.choice(
                ["SMC strategy", "RSI strategy", "EMA Cross", "Breakout", "FVG strategy"]
            )
        if "{pair}" in template["question"]:
            variables["pair"] = random.choice(PAIRS)
        if "{pair2}" in template["question"]:
            remaining = [p for p in PAIRS if p != variables.get("pair")]
            variables["pair2"] = random.choice(remaining)
        if "{session}" in template["question"]:
            variables["session"] = random.choice(SESSIONS)
        if "{timeframe}" in template["question"]:
            variables["timeframe"] = random.choice(TIMEFRAMES)
        if "{timeframe2}" in template["question"]:
            remaining = [t for t in TIMEFRAMES if t != variables.get("timeframe")]
            variables["timeframe2"] = random.choice(remaining)
        if "{period_a}" in template["question"]:
            variables["period_a"] = random.choice(ATR_PERIODS)
        if "{period_b}" in template["question"]:
            remaining = [p for p in ATR_PERIODS if p != variables.get("period_a")]
            variables["period_b"] = random.choice(remaining)
        if "{sl_mult}" in template["question"]:
            variables["sl_mult"] = random.choice(SL_MULTIPLIERS)
        if "{entry_a}" in template["question"]:
            from research.strategy_generator import ENTRY_COMPONENTS
            variables["entry_a"] = random.choice(ENTRY_COMPONENTS)
        if "{entry_b}" in template["question"]:
            from research.strategy_generator import ENTRY_COMPONENTS
            variables["entry_b"] = random.choice(ENTRY_COMPONENTS)

        return variables
