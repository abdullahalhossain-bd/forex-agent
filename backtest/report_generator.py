"""
backtest/report_generator.py — Research-Grade Backtest Report Generator

Generates comprehensive, institutional-quality backtest reports covering:
  1. Executive Summary
  2. Pair Performance
  3. Strategy Performance
  4. Agent Performance
  5. Module Contribution
  6. Rejection Report
  7. Confidence Calibration
  8. Session Report
  9. Day-of-Week / Hourly Report
  10. Pattern Report
  11. Risk Report
  12. Equity Curve Data
  13. Monte Carlo Summary
  14. Walk-Forward Summary
  15. Trade Replay Summary
  16. Automatic Weakness Detection

Outputs: TXT report, JSON report, CSV trade log, CSV replay log
"""

from __future__ import annotations

import json
import csv
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import numpy as np

log = logging.getLogger("report_generator")


# ============================================================
# Data containers
# ============================================================

@dataclass
class RunMetadata:
    project_version: str = "1.0.0"
    git_commit: str = ""
    run_date: str = ""
    execution_time_sec: float = 0.0
    mode: str = "Historical Backtest"
    broker_model: str = "Simulated"
    spread_pips: float = 0.0
    commission_per_lot: float = 7.0
    slippage_pips: float = 2.0
    symbol: str = ""
    timeframe: str = ""
    total_bars: int = 0
    warmup_bars: int = 50
    starting_balance: float = 10000.0

    def to_dict(self):
        return asdict(self)


@dataclass
class ExecutiveSummary:
    ending_balance: float = 0.0
    net_profit: float = 0.0
    return_pct: float = 0.0
    profit_factor: float = 0.0
    recovery_factor: float = 0.0
    expectancy: float = 0.0
    win_rate: float = 0.0
    loss_rate: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0
    avg_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_hold_bars: float = 0.0
    avg_rr: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    rejected_trades: int = 0
    skipped_trades: int = 0
    avg_confidence: float = 0.0
    avg_position_size: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    payoff_ratio: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class ReportData:
    """Container for all report data."""
    metadata: RunMetadata = field(default_factory=RunMetadata)
    summary: ExecutiveSummary = field(default_factory=ExecutiveSummary)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    rejection_stats: Dict[str, int] = field(default_factory=dict)
    pair_breakdown: Dict[str, Dict] = field(default_factory=dict)
    strategy_breakdown: Dict[str, Dict] = field(default_factory=dict)
    agent_breakdown: Dict[str, Dict] = field(default_factory=dict)
    module_contribution: Dict[str, Dict] = field(default_factory=dict)
    session_breakdown: Dict[str, Dict] = field(default_factory=dict)
    day_of_week_breakdown: Dict[str, Dict] = field(default_factory=dict)
    hourly_breakdown: Dict[str, Dict] = field(default_factory=dict)
    pattern_breakdown: Dict[str, Dict] = field(default_factory=dict)
    risk_report: Dict[str, Any] = field(default_factory=dict)
    confidence_calibration: Dict[str, Dict] = field(default_factory=dict)
    monte_carlo: Dict[str, Any] = field(default_factory=dict)
    walk_forward: Dict[str, Any] = field(default_factory=dict)
    trade_replay_summary: Dict[str, Any] = field(default_factory=dict)
    weakness_detection: Dict[str, Any] = field(default_factory=dict)
    monthly_returns: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ============================================================
# Report Generator
# ============================================================

class BacktestReportGenerator:
    """Generates institutional-quality backtest reports."""

    def __init__(self, output_dir: str = "backtest/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def generate(self, data: ReportData) -> Dict[str, str]:
        """Generate all report outputs. Returns dict of {format: filepath}."""
        paths = {}
        paths["txt"] = self._write_txt(data)
        paths["json"] = self._write_json(data)
        paths["csv"] = self._write_csv(data)
        log.info(f"[ReportGenerator] Reports written to {self.output_dir}")
        return paths

    # ----------------------------------------------------------
    # TXT Report
    # ----------------------------------------------------------

    def _write_txt(self, data: ReportData) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("  FOREX AI — COMPLETE BACKTEST REPORT")
        lines.append("=" * 70)

        # 1. Metadata
        m = data.metadata
        lines.append("")
        lines.append("[RUN INFO]")
        lines.append(f"  Project Version    : {m.project_version}")
        lines.append(f"  Git Commit         : {m.git_commit or 'N/A'}")
        lines.append(f"  Run Date           : {m.run_date}")
        lines.append(f"  Execution Time     : {m.execution_time_sec:.1f}s")
        lines.append(f"  Mode               : {m.mode}")
        lines.append(f"  Broker Model       : {m.broker_model}")
        lines.append(f"  Spread             : {m.spread_pips} pips")
        lines.append(f"  Commission         : ${m.commission_per_lot}/lot")
        lines.append(f"  Slippage           : {m.slippage_pips} pips")
        lines.append(f"  Symbol             : {m.symbol}")
        lines.append(f"  Timeframe          : {m.timeframe}")
        lines.append(f"  Total Bars         : {m.total_bars}")
        lines.append(f"  Warmup Bars        : {m.warmup_bars}")

        # 2. Executive Summary
        lines.append("")
        lines.append("=" * 70)
        lines.append("  EXECUTIVE SUMMARY")
        lines.append("=" * 70)
        s = data.summary
        lines.append(f"  Starting Balance   : ${m.starting_balance:,.2f}")
        lines.append(f"  Ending Balance     : ${s.ending_balance:,.2f}")
        lines.append(f"  Net Profit         : ${s.net_profit:+,.2f}")
        lines.append(f"  Return %           : {s.return_pct:+.2f}%")
        lines.append(f"  Profit Factor      : {s.profit_factor:.2f}")
        lines.append(f"  Recovery Factor    : {s.recovery_factor:.2f}")
        lines.append(f"  Expectancy         : {s.expectancy:.2f}R")
        lines.append(f"  Win Rate           : {s.win_rate:.1f}%")
        lines.append(f"  Loss Rate          : {s.loss_rate:.1f}%")
        lines.append(f"  Sharpe Ratio       : {s.sharpe:.2f}")
        lines.append(f"  Sortino Ratio      : {s.sortino:.2f}")
        lines.append(f"  Calmar Ratio       : {s.calmar:.2f}")
        lines.append(f"  Max Drawdown       : {s.max_drawdown_pct:.2f}% (${s.max_drawdown_usd:,.2f})")
        lines.append(f"  Average Trade      : ${s.avg_trade:+,.2f}")
        lines.append(f"  Average Win        : ${s.avg_win:+,.2f} ({s.avg_win_pips:.1f} pips)")
        lines.append(f"  Average Loss       : ${s.avg_loss:+,.2f} ({s.avg_loss_pips:.1f} pips)")
        lines.append(f"  Average Hold       : {s.avg_hold_bars:.1f} bars")
        lines.append(f"  Average R:R        : 1:{s.avg_rr:.2f}")
        lines.append(f"  Largest Win        : ${s.largest_win:+,.2f}")
        lines.append(f"  Largest Loss       : ${s.largest_loss:+,.2f}")
        lines.append(f"  Payoff Ratio       : {s.payoff_ratio:.2f}")
        lines.append(f"  Gross Profit       : ${s.gross_profit:,.2f}")
        lines.append(f"  Gross Loss         : ${s.gross_loss:,.2f}")
        lines.append(f"  Max Consec Wins    : {s.max_consecutive_wins}")
        lines.append(f"  Max Consec Losses  : {s.max_consecutive_losses}")
        lines.append(f"  Total Trades       : {s.total_trades}")
        lines.append(f"    Winning          : {s.winning_trades}")
        lines.append(f"    Losing           : {s.losing_trades}")
        lines.append(f"  Rejected Trades    : {s.rejected_trades}")
        lines.append(f"  Skipped Bars       : {s.skipped_trades}")
        lines.append(f"  Avg Confidence     : {s.avg_confidence:.1f}%")
        lines.append(f"  Avg Position Size  : {s.avg_position_size:.4f} lots")

        # 3. Pair Performance
        if data.pair_breakdown:
            lines.append("")
            lines.append("=" * 70)
            lines.append("  PAIR PERFORMANCE")
            lines.append("=" * 70)
            lines.append(f"  {'Pair':<10} {'Trades':>7} {'Win%':>7} {'PF':>7} "
                          f"{'Net P&L':>12} {'DD%':>7} {'Avg RR':>8} {'Avg Hold':>9} {'Avg Spr':>8} {'Avg Slip':>9}")
            lines.append("-" * 70)
            for pair, d in sorted(data.pair_breakdown.items()):
                lines.append(
                    f"  {pair:<10} {d.get('trades',0):>7} {d.get('win_rate',0):>6.1f}% "
                    f"{d.get('profit_factor',0):>7.2f} {d.get('pnl_usd',0):>+11,.2f} "
                    f"{d.get('max_drawdown_pct',0):>6.1f}% {d.get('avg_rr',0):>7.2f} "
                    f"{d.get('avg_hold',0):>8.1f}b {d.get('avg_spread',0):>7.1f}p "
                    f"{d.get('avg_slippage',0):>8.1f}p"
                )

        # 4. Strategy Performance
        if data.strategy_breakdown:
            lines.append("")
            lines.append("=" * 70)
            lines.append("  STRATEGY PERFORMANCE")
            lines.append("=" * 70)
            lines.append(f"  {'Strategy':<20} {'Trades':>7} {'Win%':>7} {'PF':>7} "
                          f"{'Expect':>8} {'Contrib':>8} {'Sharpe':>8} {'Avg Conf':>9} {'Avg RR':>8}")
            lines.append("-" * 70)
            for strat, d in sorted(data.strategy_breakdown.items(), key=lambda x: x[1].get("pnl_usd", 0), reverse=True):
                lines.append(
                    f"  {strat:<20} {d.get('trades',0):>7} {d.get('win_rate',0):>6.1f}% "
                    f"{d.get('profit_factor',0):>7.2f} {d.get('expectancy',0):>7.2f}R "
                    f"{d.get('contribution',0):>7.1f}% {d.get('sharpe',0):>7.2f} "
                    f"{d.get('avg_confidence',0):>8.1f}% {d.get('avg_rr',0):>7.2f}"
                )

        # 5. Agent Performance
        if data.agent_breakdown:
            lines.append("")
            lines.append("=" * 70)
            lines.append("  AGENT PERFORMANCE")
            lines.append("=" * 70)
            lines.append(f"  {'Agent':<25} {'Votes':>7} {'Correct':>9} {'Wrong':>7} "
                          f"{'Abstain':>9} {'Avg Conf':>9} {'Contrib':>9}")
            lines.append("-" * 70)
            for agent, d in sorted(data.agent_breakdown.items()):
                lines.append(
                    f"  {agent:<25} {d.get('votes',0):>7} {d.get('correct_pct',0):>8.1f}% "
                    f"{d.get('wrong_pct',0):>6.1f}% {d.get('abstain_pct',0):>8.1f}% "
                    f"{d.get('avg_confidence',0):>8.1f}% {d.get('contribution',0):>8.1f}%"
                )

        # 6. Module Contribution
        if data.module_contribution:
            lines.append("")
            lines.append("=" * 70)
            lines.append("  MODULE CONTRIBUTION")
            lines.append("=" * 70)
            lines.append(f"  {'Module':<25} {'Enabled':>8} {'Trades':>7} {'Win%':>7} {'Impact':>8}")
            lines.append("-" * 70)
            for mod, d in sorted(data.module_contribution.items()):
                lines.append(
                    f"  {mod:<25} {'YES' if d.get('enabled', True) else 'NO':>8} "
                    f"{d.get('trades',0):>7} {d.get('win_rate',0):>6.1f}% {d.get('impact',0):>7.1f}%"
                )

        # 7. Rejection Report
        if data.rejection_stats:
            lines.append("")
            lines.append("=" * 70)
            lines.append("  REJECTION REPORT")
            lines.append("=" * 70)
            total_rej = sum(data.rejection_stats.values())
            for reason, count in sorted(data.rejection_stats.items(), key=lambda x: x[1], reverse=True):
                pct = count / total_rej * 100 if total_rej > 0 else 0
                lines.append(f"  {reason:<25} {count:>7} ({pct:.1f}%)")
            lines.append(f"  {'TOTAL':<25} {total_rej:>7}")

            # Rejection vs Acceptance
            accepted = s.total_trades
            lines.append("")
            lines.append(f"  Accepted           : {accepted}")
            lines.append(f"  Rejected           : {total_rej}")
            lines.append(f"  Acceptance %       : {accepted/(accepted+total_rej)*100:.1f}%" if (accepted+total_rej) > 0 else "  Acceptance %       : N/A")

        # 8. Confidence Calibration
        if data.confidence_calibration:
            lines.append("")
            lines.append("=" * 70)
            lines.append("  CONFIDENCE CALIBRATION")
            lines.append("=" * 70)
            lines.append(f"  {'Confidence':<15} {'Trades':>7} {'Actual Win%':>12}")
            lines.append("-" * 70)
            for bucket, d in sorted(data.confidence_calibration.items()):
                lines.append(f"  {bucket:<15} {d.get('trades',0):>7} {d.get('actual_win_pct',0):>11.1f}%")

        # 9. Session Report
        if data.session_breakdown:
            lines.append("")
            lines.append("=" * 70)
            lines.append("  SESSION REPORT")
            lines.append("=" * 70)
            lines.append(f"  {'Session':<20} {'Trades':>7} {'Win%':>7} {'PF':>7} {'P&L':>12} {'DD%':>7}")
            lines.append("-" * 70)
            for sess, d in sorted(data.session_breakdown.items(), key=lambda x: x[1].get("pnl_usd", 0), reverse=True):
                lines.append(
                    f"  {sess:<20} {d.get('trades',0):>7} {d.get('win_rate',0):>6.1f}% "
                    f"{d.get('profit_factor',0):>7.2f} {d.get('pnl_usd',0):>+11,.2f} "
                    f"{d.get('max_drawdown_pct',0):>6.1f}%"
                )

        # 10. Day-of-Week Report
        if data.day_of_week_breakdown:
            lines.append("")
            lines.append("=" * 70)
            lines.append("  DAY-OF-WEEK REPORT")
            lines.append("=" * 70)
            lines.append(f"  {'Day':<12} {'Trades':>7} {'Win%':>7} {'PF':>7} {'P&L':>12}")
            lines.append("-" * 70)
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            for day in day_order:
                d = data.day_of_week_breakdown.get(day, {})
                if d:
                    lines.append(
                        f"  {day:<12} {d.get('trades',0):>7} {d.get('win_rate',0):>6.1f}% "
                        f"{d.get('profit_factor',0):>7.2f} {d.get('pnl_usd',0):>+11,.2f}"
                    )

        # 11. Hourly Report
        if data.hourly_breakdown:
            lines.append("")
            lines.append("=" * 70)
            lines.append("  HOURLY REPORT (Top 10 by trade count)")
            lines.append("=" * 70)
            lines.append(f"  {'Hour':<8} {'Trades':>7} {'Win%':>7} {'PF':>7} {'P&L':>12}")
            lines.append("-" * 70)
            sorted_hours = sorted(data.hourly_breakdown.items(), key=lambda x: x[1].get("trades", 0), reverse=True)[:10]
            for hour, d in sorted_hours:
                lines.append(
                    f"  {hour:<8} {d.get('trades',0):>7} {d.get('win_rate',0):>6.1f}% "
                    f"{d.get('profit_factor',0):>7.2f} {d.get('pnl_usd',0):>+11,.2f}"
                )

        # 12. Pattern Report
        if data.pattern_breakdown:
            lines.append("")
            lines.append("=" * 70)
            lines.append("  PATTERN REPORT")
            lines.append("=" * 70)
            lines.append(f"  {'Pattern':<20} {'Trades':>7} {'Win':>6} {'Loss':>6} {'PF':>7} {'Avg RR':>8}")
            lines.append("-" * 70)
            for pat, d in sorted(data.pattern_breakdown.items(), key=lambda x: x[1].get("pnl_usd", 0), reverse=True):
                lines.append(
                    f"  {pat:<20} {d.get('trades',0):>7} {d.get('wins',0):>6} {d.get('losses',0):>6} "
                    f"{d.get('profit_factor',0):>7.2f} {d.get('avg_rr',0):>7.2f}"
                )

        # 13. Risk Report
        if data.risk_report:
            lines.append("")
            lines.append("=" * 70)
            lines.append("  RISK REPORT")
            lines.append("=" * 70)
            r = data.risk_report
            lines.append(f"  Average Risk       : {r.get('avg_risk_usd',0):,.2f} USD")
            lines.append(f"  Risk %             : {r.get('avg_risk_pct',0):.2f}%")
            lines.append(f"  Average Lot        : {r.get('avg_lot',0):.4f}")
            lines.append(f"  Maximum Lot        : {r.get('max_lot',0):.4f}")
            lines.append(f"  Average SL (pips)  : {r.get('avg_sl_pips',0):.1f}")
            lines.append(f"  Average TP (pips)  : {r.get('avg_tp_pips',0):.1f}")
            lines.append(f"  Average R:R        : 1:{r.get('avg_rr',0):.2f}")
            lines.append(f"  Risk Utilization   : {r.get('risk_utilization',0):.1f}%")

        # 14. Monte Carlo
        if data.monte_carlo:
            lines.append("")
            lines.append("=" * 70)
            lines.append("  MONTE CARLO SIMULATION")
            lines.append("=" * 70)
            mc = data.monte_carlo
            lines.append(f"  Simulations        : {mc.get('n_simulations', 0):,}")
            lines.append(f"  Trades per path    : {mc.get('n_trades', 0)}")
            lines.append(f"  Worst DD           : {mc.get('worst_max_drawdown_pct', 0):.1f}%")
            lines.append(f"  Median Return      : {mc.get('median_pct', 0):+.1f}%")
            lines.append(f"  5th Percentile     : {mc.get('percentile_5_pct', 0):+.1f}%")
            lines.append(f"  95th Percentile    : {mc.get('percentile_95_pct', 0):+.1f}%")
            lines.append(f"  Risk of Ruin       : {mc.get('risk_of_ruin', 0)*100:.1f}%")
            lines.append(f"  Survival Rate      : {mc.get('survival_rate', 0)*100:.1f}%")

        # 15. Walk-Forward
        if data.walk_forward:
            lines.append("")
            lines.append("=" * 70)
            lines.append("  WALK-FORWARD ANALYSIS")
            lines.append("=" * 70)
            wf = data.walk_forward
            lines.append(f"  Windows            : {wf.get('total_windows', 0)}")
            lines.append(f"  IS P&L             : ${wf.get('total_is_pnl', 0):,.2f}")
            lines.append(f"  OOS P&L            : ${wf.get('total_oos_pnl', 0):,.2f}")
            lines.append(f"  WFE                : {wf.get('overall_wfe', 0):.1%}")
            lines.append(f"  Result             : {'PASS' if wf.get('pass_min_wfe') else 'FAIL'}")
            if wf.get("windows"):
                lines.append("")
                lines.append(f"  {'Win':<6} {'IS Trades':>10} {'IS P&L':>11} {'OOS Trades':>11} {'OOS P&L':>11} {'WFE':>8}")
                lines.append("-" * 70)
                for w in wf["windows"]:
                    lines.append(
                        f"  {w['window']:<6} {w['is_trades']:>10} ${w['is_pnl']:>9,.2f} "
                        f"{w['oos_trades']:>11} ${w['oos_pnl']:>9,.2f} {w['wfe']:>7.1%}"
                    )

        # 16. Monthly Returns
        if data.monthly_returns:
            lines.append("")
            lines.append("=" * 70)
            lines.append("  MONTHLY RETURNS")
            lines.append("=" * 70)
            for month, ret in sorted(data.monthly_returns.items()):
                lines.append(f"  {month:<12} {'+'if ret>=0 else ''}{ret:.2f}%")

        # 17. Automatic Weakness Detection
        if data.weakness_detection:
            lines.append("")
            lines.append("=" * 70)
            lines.append("  AUTOMATIC WEAKNESS DETECTION")
            lines.append("=" * 70)
            wd = data.weakness_detection
            lines.append(f"  Overall Grade      : {wd.get('overall_grade', 'N/A')}")
            lines.append("")

            for item in wd.get("findings", []):
                lines.append(f"  [{item.get('severity', '?')}] {item.get('title', '')}")
                lines.append(f"      {item.get('detail', '')}")
                lines.append(f"      Win Rate: {item.get('win_rate', 'N/A')}")
                lines.append(f"      Recommendation: {item.get('recommendation', '')}")
                lines.append("")

            if wd.get("suggested_improvements"):
                lines.append("  SUGGESTED IMPROVEMENTS:")
                for imp in wd["suggested_improvements"]:
                    lines.append(f"    {imp}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("  END OF REPORT")
        lines.append("=" * 70)

        report_text = "\n".join(lines)
        filename = f"backtest_report_{data.metadata.symbol}_{data.metadata.timeframe}_{self._timestamp}.txt"
        path = self.output_dir / filename
        path.write_text(report_text, encoding="utf-8")
        return str(path)

    # ----------------------------------------------------------
    # JSON Report
    # ----------------------------------------------------------

    def _write_json(self, data: ReportData) -> str:
        filename = f"backtest_report_{data.metadata.symbol}_{data.metadata.timeframe}_{self._timestamp}.json"
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data.to_dict(), f, indent=2, default=str)
        return str(path)

    # ----------------------------------------------------------
    # CSV Trade Log
    # ----------------------------------------------------------

    def _write_csv(self, data: ReportData) -> str:
        """Write trade-level CSV from trade_replay_summary."""
        trades = data.trade_replay_summary.get("trades", [])
        if not trades:
            return ""
        filename = f"trades_{data.metadata.symbol}_{data.metadata.timeframe}_{self._timestamp}.csv"
        path = self.output_dir / filename
        if not trades:
            return ""

        fieldnames = list(trades[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for t in trades:
                writer.writerow({k: v if not isinstance(v, dict) else str(v) for k, v in t.items()})
        return str(path)


# ============================================================
# Analysis Helpers — Build ReportData from raw trade objects
# ============================================================

def analyze_trades(
    trades: list,
    starting_balance: float,
    ending_balance: float,
    equity_curve: list,
    rejection_stats: dict,
    metadata: RunMetadata,
) -> ReportData:
    """Build complete ReportData from raw trade objects + metrics."""
    data = ReportData(metadata=metadata, equity_curve=equity_curve)
    data.rejection_stats = rejection_stats

    if not trades:
        data.summary.ending_balance = ending_balance
        data.summary.starting_balance = starting_balance
        data.summary.net_profit = ending_balance - starting_balance
        data.summary.return_pct = (ending_balance - starting_balance) / starting_balance * 100 if starting_balance else 0
        return data

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd < 0]
    n = len(trades)

    s = data.summary
    s.starting_balance = starting_balance
    s.ending_balance = ending_balance
    s.total_trades = n
    s.winning_trades = len(wins)
    s.losing_trades = len(losses)
    s.win_rate = len(wins) / n * 100
    s.loss_rate = len(losses) / n * 100
    s.net_profit = ending_balance - starting_balance
    s.return_pct = s.net_profit / starting_balance * 100 if starting_balance else 0
    s.gross_profit = sum(t.pnl_usd for t in wins)
    s.gross_loss = abs(sum(t.pnl_usd for t in losses))
    s.profit_factor = s.gross_profit / s.gross_loss if s.gross_loss > 0 else 0
    s.avg_trade = s.net_profit / n
    s.avg_win = np.mean([t.pnl_usd for t in wins]) if wins else 0
    s.avg_loss = np.mean([t.pnl_usd for t in losses]) if losses else 0
    s.avg_win_pips = np.mean([t.pnl_pips for t in wins]) if wins else 0
    s.avg_loss_pips = np.mean([t.pnl_pips for t in losses]) if losses else 0
    s.largest_win = max((t.pnl_usd for t in wins), default=0)
    s.largest_loss = min((t.pnl_usd for t in losses), default=0)
    s.payoff_ratio = abs(s.avg_win / s.avg_loss) if s.avg_loss != 0 else 0
    s.avg_rr = abs(s.avg_win_pips / s.avg_loss_pips) if s.avg_loss_pips != 0 else 0
    s.avg_hold_bars = np.mean([t.hold_bars for t in trades])
    s.avg_confidence = np.mean([getattr(t, "confidence", 0) for t in trades])
    s.avg_position_size = np.mean([t.lot_size for t in trades])

    # Max consecutive wins/losses
    max_cw, max_cl, cw, cl = 0, 0, 0, 0
    for t in trades:
        if t.pnl_usd > 0:
            cw += 1; cl = 0
            max_cw = max(max_cw, cw)
        else:
            cl += 1; cw = 0
            max_cl = max(max_cl, cl)
    s.max_consecutive_wins = max_cw
    s.max_consecutive_losses = max_cl

    # Expectancy (in R multiples)
    avg_win_r = np.mean([abs(t.pnl_pips) for t in wins]) if wins else 0
    avg_loss_r = np.mean([abs(t.pnl_pips) for t in losses]) if losses else 0
    if avg_loss_r > 0:
        s.expectancy = (len(wins) / n * avg_win_r - len(losses) / n * avg_loss_r) / avg_loss_r
    else:
        s.expectancy = 0

    # Recovery Factor
    eq = np.array([starting_balance] + [starting_balance + sum(t.pnl_usd for t in trades[:i+1]) for i in range(n)])
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1) * 100
    s.max_drawdown_pct = float(np.max(dd))
    s.max_drawdown_usd = float(np.max(peak - eq))
    s.recovery_factor = s.return_pct / s.max_drawdown_pct if s.max_drawdown_pct > 0 else 0

    # Drawdown curve
    data.drawdown_curve = dd.tolist()

    # Sharpe / Sortino
    if n > 1:
        rets = np.array([t.pnl_usd / starting_balance for t in trades])
        ar = np.mean(rets)
        sr = np.std(rets, ddof=1)
        bars_per_day = {"M1": 960, "M5": 288, "M15": 96, "H1": 24, "H4": 6, "D1": 1}
        tpy = 252 * bars_per_day.get(metadata.timeframe, 96)
        if sr > 0:
            s.sharpe = (ar * tpy - 0.02) / sr * np.sqrt(tpy)
        ds = rets[rets < 0]
        if len(ds) > 0:
            dstd = np.std(ds, ddof=1)
            if dstd > 0:
                s.sortino = (ar * tpy - 0.02) / dstd * np.sqrt(tpy)
    s.calmar = s.return_pct / s.max_drawdown_pct if s.max_drawdown_pct > 0 else 0

    # Rejected/Skipped
    s.rejected_trades = sum(rejection_stats.values())
    s.skipped_trades = metadata.total_bars - metadata.warmup_bars - n - s.rejected_trades

    # --- Breakdowns ---
    _build_pair_breakdown(trades, data)
    _build_strategy_breakdown(trades, data)
    _build_session_breakdown(trades, data)
    _build_day_of_week_breakdown(trades, data)
    _build_hourly_breakdown(trades, data)
    _build_pattern_breakdown(trades, data)
    _build_risk_report(trades, data, starting_balance)
    _build_confidence_calibration(trades, data)
    _build_monthly_returns(trades, data, starting_balance)

    return data


def _calc_subgroup_stats(items):
    """Calculate stats for a subgroup of trades."""
    if not items:
        return {}
    n = len(items)
    w = [t for t in items if t.pnl_usd > 0]
    l = [t for t in items if t.pnl_usd <= 0]
    gp = sum(t.pnl_usd for t in w)
    gl = abs(sum(t.pnl_usd for t in l))
    return {
        "trades": n,
        "wins": len(w),
        "losses": len(l),
        "win_rate": round(len(w) / n * 100, 1),
        "pnl_usd": round(sum(t.pnl_usd for t in items), 2),
        "pnl_pips": round(sum(t.pnl_pips for t in items), 1),
        "profit_factor": round(gp / gl, 2) if gl > 0 else 0,
        "avg_pnl": round(np.mean([t.pnl_usd for t in items]), 2),
    }


def _build_pair_breakdown(trades, data):
    groups = defaultdict(list)
    for t in trades:
        groups[t.symbol].append(t)
    for pair, items in groups.items():
        d = _calc_subgroup_stats(items)
        d["avg_rr"] = round(abs(np.mean([t.pnl_pips for t in items if t.pnl_usd > 0] or [0]) /
                                  (np.mean([abs(t.pnl_pips) for t in items if t.pnl_usd < 0] or [1]))), 2)
        d["avg_hold"] = round(np.mean([t.hold_bars for t in items]), 1)
        d["avg_spread"] = 0
        d["avg_slippage"] = round(np.mean([t.slippage_pips for t in items]), 1)
        eq = np.cumsum([t.pnl_usd for t in items])
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.where(peak > 0, peak, 1) * 100
        d["max_drawdown_pct"] = round(float(np.max(dd)), 2) if len(dd) > 0 else 0
        data.pair_breakdown[pair] = d


def _build_strategy_breakdown(trades, data):
    groups = defaultdict(list)
    for t in trades:
        groups[t.strategy or "unknown"].append(t)
    total_pnl = sum(t.pnl_usd for t in trades) or 1
    for strat, items in groups.items():
        d = _calc_subgroup_stats(items)
        d["avg_rr"] = round(abs(np.mean([t.pnl_pips for t in items if t.pnl_usd > 0] or [0]) /
                                  (np.mean([abs(t.pnl_pips) for t in items if t.pnl_usd < 0] or [1]))), 2)
        d["avg_confidence"] = round(np.mean([t.confidence for t in items]), 1)
        d["contribution"] = round(sum(t.pnl_usd for t in items) / total_pnl * 100, 1)
        d["expectancy"] = 0
        d["sharpe"] = 0
        data.strategy_breakdown[strat] = d


def _build_session_breakdown(trades, data):
    groups = defaultdict(list)
    for t in trades:
        try:
            et = datetime.fromisoformat(t.entry_time) if isinstance(t.entry_time, str) else t.entry_time
            h = et.hour
            if 0 <= h < 7:
                sess = "Asian"
            elif 7 <= h < 13:
                sess = "London"
            elif 13 <= h < 16:
                sess = "London-NY Overlap"
            elif 16 <= h < 22:
                sess = "New York"
            else:
                sess = "Off-Hours"
            groups[sess].append(t)
        except Exception:
            groups["Unknown"].append(t)

    for sess, items in groups.items():
        d = _calc_subgroup_stats(items)
        eq = np.cumsum([t.pnl_usd for t in items])
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.where(peak > 0, peak, 1) * 100
        d["max_drawdown_pct"] = round(float(np.max(dd)), 2) if len(dd) > 0 else 0
        data.session_breakdown[sess] = d


def _build_day_of_week_breakdown(trades, data):
    groups = defaultdict(list)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for t in trades:
        try:
            et = datetime.fromisoformat(t.entry_time) if isinstance(t.entry_time, str) else t.entry_time
            groups[day_names[et.weekday()]].append(t)
        except Exception:
            pass
    for day, items in groups.items():
        d = _calc_subgroup_stats(items)
        data.day_of_week_breakdown[day] = d


def _build_hourly_breakdown(trades, data):
    groups = defaultdict(list)
    for t in trades:
        try:
            et = datetime.fromisoformat(t.entry_time) if isinstance(t.entry_time, str) else t.entry_time
            groups[f"Hour {et.hour}"].append(t)
        except Exception:
            pass
    for hour, items in groups.items():
        d = _calc_subgroup_stats(items)
        data.hourly_breakdown[hour] = d


def _build_pattern_breakdown(trades, data):
    # Use strategy field as pattern proxy (real implementation would use actual pattern tags)
    groups = defaultdict(list)
    for t in trades:
        pattern = getattr(t, "pattern", "") or t.strategy or "General"
        groups[pattern].append(t)
    for pat, items in groups.items():
        d = _calc_subgroup_stats(items)
        d["avg_rr"] = round(abs(np.mean([t.pnl_pips for t in items if t.pnl_usd > 0] or [0]) /
                                  (np.mean([abs(t.pnl_pips) for t in items if t.pnl_usd < 0] or [1]))), 2)
        data.pattern_breakdown[pat] = d


def _build_risk_report(trades, data, starting_balance):
    if not trades:
        return
    pip_vals = []
    for t in trades:
        sym = t.symbol.upper()
        if sym.endswith("JPY"):
            pip_vals.append(0.01)
        elif sym == "XAUUSD":
            pip_vals.append(0.1)
        else:
            pip_vals.append(0.0001)

    sl_pips_list = []
    tp_pips_list = []
    for t, pv in zip(trades, pip_vals):
        if t.direction == "BUY":
            sl_pips_list.append(abs(t.entry_price - t.stop_loss) / pv)
            tp_pips_list.append(abs(t.take_profit - t.entry_price) / pv)
        else:
            sl_pips_list.append(abs(t.stop_loss - t.entry_price) / pv)
            tp_pips_list.append(abs(t.entry_price - t.take_profit) / pv)

    data.risk_report = {
        "avg_risk_usd": round(np.mean([abs(t.pnl_usd) for t in trades if t.pnl_usd < 0] or [0]), 2),
        "avg_risk_pct": round(np.mean([abs(t.pnl_usd) for t in trades if t.pnl_usd < 0] or [0]) / starting_balance * 100, 2),
        "avg_lot": round(np.mean([t.lot_size for t in trades]), 4),
        "max_lot": round(max(t.lot_size for t in trades), 4),
        "avg_sl_pips": round(np.mean(sl_pips_list), 1),
        "avg_tp_pips": round(np.mean(tp_pips_list), 1),
        "avg_rr": round(np.mean(tp_pips_list) / np.mean(sl_pips_list), 2) if np.mean(sl_pips_list) > 0 else 0,
        "risk_utilization": round(np.mean([t.lot_size for t in trades]) / 0.20 * 100, 1),
    }


def _build_confidence_calibration(trades, data):
    buckets = {
        "90-100": (90, 100),
        "80-90": (80, 90),
        "70-80": (70, 80),
        "60-70": (60, 70),
        "50-60": (50, 60),
        "Below 50": (0, 50),
    }
    for label, (lo, hi) in buckets.items():
        matched = [t for t in trades if lo <= t.confidence < hi]
        if matched:
            wins = sum(1 for t in matched if t.pnl_usd > 0)
            data.confidence_calibration[label] = {
                "trades": len(matched),
                "actual_win_pct": round(wins / len(matched) * 100, 1),
            }


def _build_monthly_returns(trades, data, starting_balance):
    monthly = defaultdict(float)
    for t in trades:
        try:
            et = datetime.fromisoformat(t.entry_time) if isinstance(t.entry_time, str) else t.entry_time
            key = et.strftime("%Y-%m")
            monthly[key] += t.pnl_usd
        except Exception:
            pass
    data.monthly_returns = {k: round(v / starting_balance * 100, 2) for k, v in monthly.items()}
