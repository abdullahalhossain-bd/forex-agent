"""
backtest/performance_report.py — Performance Report Generator (Day 74)
=======================================================================

Generates human-readable performance reports and Telegram alerts
from backtest comparison results.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from backtest.ml_backtest import BacktestMetrics
from backtest.comparison_engine import ComparisonResult

log = get_logger("performance_report")


def generate_text_report(
    results: Dict[str, BacktestMetrics],
    comparison: ComparisonResult,
    pair: str = "EURUSD",
    timeframe: str = "15m",
) -> str:
    """Generate a human-readable performance comparison report."""
    lines = [
        "═══════════════════════════════════════════════════════",
        "  📊 FOREX AI BACKTEST REPORT — Day 74",
        "═══════════════════════════════════════════════════════",
        f"  Pair: {pair} | Timeframe: {timeframe}",
        "───────────────────────────────────────────────────────",
        f"  {'System':<20s} | {'Win Rate':>8s} | {'PF':>6s} | {'Drawdown':>8s} | {'Sharpe':>6s} | {'Trades':>6s} | {'PnL':>8s}",
        "───────────────────────────────────────────────────────",
    ]

    system_labels = {
        "rule_only": "Rule Only",
        "rule_intel": "Rule + Intelligence",
        "full_ai": "Full AI System",
    }

    for name in ("rule_only", "rule_intel", "full_ai"):
        m = results.get(name)
        if m:
            label = system_labels.get(name, name)
            lines.append(
                f"  {label:<20s} | {m.win_rate:>7.1f}% | {m.profit_factor:>6.2f} | "
                f"{m.max_drawdown_pct:>7.1f}% | {m.sharpe_ratio:>6.2f} | "
                f"{m.total_trades:>6d} | ${m.total_pnl_usd:>7.0f}"
            )

    lines.extend([
        "═══════════════════════════════════════════════════════",
        f"  Winner: {system_labels.get(comparison.winner, comparison.winner)} ✅",
        f"  ML Improved: {'YES ✅' if comparison.ml_improved else 'NO ❌'}",
        f"  Improvement: +{comparison.improvement_pct:.1f}% WR | +{comparison.improvement_pf:.2f} PF",
        f"  Recommendation: {comparison.recommendation}",
    ])

    if comparison.diagnosis:
        lines.append("  ── Diagnosis ──")
        for issue in comparison.diagnosis:
            lines.append(f"    • {issue}")

    lines.append("═══════════════════════════════════════════════════════")
    return "\n".join(lines)


def generate_telegram_report(
    results: Dict[str, BacktestMetrics],
    comparison: ComparisonResult,
    pair: str = "EURUSD",
) -> Optional[str]:
    """Generate a concise Telegram alert with backtest results."""
    if not results:
        return None

    rule = results.get("rule_only", BacktestMetrics(system="rule_only"))
    ai = results.get("full_ai", BacktestMetrics(system="full_ai"))

    status_emoji = "✅" if comparison.ml_improved else "❌"
    status_text = "ML ENABLED" if comparison.ml_improved else "ML NEEDS TUNING"

    return (
        f"📊 FOREX AI BACKTEST REPORT\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Pair: {pair}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Rule Only:\n"
        f"  WR: {rule.win_rate:.1f}% | PF: {rule.profit_factor:.2f}\n"
        f"  DD: {rule.max_drawdown_pct:.1f}% | Trades: {rule.total_trades}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Full AI:\n"
        f"  WR: {ai.win_rate:.1f}% | PF: {ai.profit_factor:.2f}\n"
        f"  DD: {ai.max_drawdown_pct:.1f}% | Trades: {ai.total_trades}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Improvement:\n"
        f"  +{comparison.improvement_pct:.1f}% WR\n"
        f"  +{comparison.improvement_pf:.2f} PF\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Decision: {status_emoji} {status_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━━"
    )


def generate_strategy_contribution(results: Dict[str, BacktestMetrics]) -> Dict[str, float]:
    """Analyze which intelligence layers contribute to profit.

    Returns: {"smc": 35.0, "liquidity": 25.0, "ml": 30.0, "news": 10.0}
    """
    rule_pnl = results.get("rule_only", BacktestMetrics(system="rule_only")).total_pnl_usd
    intel_pnl = results.get("rule_intel", BacktestMetrics(system="rule_intel")).total_pnl_usd
    ai_pnl = results.get("full_ai", BacktestMetrics(system="full_ai")).total_pnl_usd

    total = abs(ai_pnl) if ai_pnl != 0 else 1.0
    contributions = {}

    # Rule-based contribution
    rule_contrib = (abs(rule_pnl) / total) * 100 if rule_pnl else 0
    contributions["rule_based"] = round(rule_contrib, 1)

    # Intelligence improvement contribution
    intel_improvement = abs(intel_pnl - rule_pnl) if intel_pnl and rule_pnl else 0
    contributions["intelligence"] = round((intel_improvement / total) * 100, 1)

    # ML improvement contribution
    ml_improvement = abs(ai_pnl - intel_pnl) if ai_pnl and intel_pnl else 0
    contributions["ml_ensemble"] = round((ml_improvement / total) * 100, 1)

    return contributions
