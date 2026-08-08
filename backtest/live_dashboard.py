"""
backtest/live_dashboard.py — Live progress dashboard for backtest runs.

Continuously reads summary.json + checkpoint.json + llm_summary.json
from a run directory and displays a real-time dashboard in the terminal.

USAGE:
    # Show dashboard for a running backtest
    python -m backtest.live_dashboard --run-id 2026-08-08_153022

    # One-shot print (no refresh loop)
    python -m backtest.live_dashboard --run-id 2026-08-08_153022 --once
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.persistence import RunDir, read_jsonl_count


def _format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    if seconds < 0:
        return "—"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_pnl(usd: float) -> str:
    """Format USD P&L with sign."""
    if usd >= 0:
        return f"+${usd:,.2f}"
    return f"-${abs(usd):,.2f}"


def render_dashboard(run_dir: RunDir) -> str:
    """Build the dashboard text. Returns a single string."""
    config = run_dir.read_config() or {}
    summary = run_dir.read_summary() or {}
    checkpoint = run_dir.read_checkpoint() or {}

    run_id = run_dir.run_id
    symbols = config.get("symbols", [])
    timeframe = config.get("timeframe", "?")
    started_at = config.get("started_at", "")

    # Calculate progress
    total_bars_per_symbol = {}
    cursor_per_symbol = {}
    for sym in symbols:
        # Read cursor from checkpoint (only tracks the most-recent symbol)
        if checkpoint.get("symbol") == sym:
            cursor_per_symbol[sym] = checkpoint.get("cursor", 0)
        else:
            # For other symbols, check if their trade file exists and count
            # (proxy for "completed" — if trades file exists and cursor >= bars, done)
            cursor_per_symbol[sym] = 0  # unknown — show 0%

    # Try to read per-symbol completion from individual checkpoint history
    # (we only save one global checkpoint, so this is best-effort)
    current_symbol = checkpoint.get("symbol", "")
    current_cursor = checkpoint.get("cursor", 0)

    # Aggregate stats
    total_trades = summary.get("total_trades", 0)
    wins = summary.get("wins", 0)
    losses = summary.get("losses", 0)
    win_rate = summary.get("win_rate", 0.0)
    gross_profit = summary.get("gross_profit_usd", 0.0)
    gross_loss = summary.get("gross_loss_usd", 0.0)
    net_pnl = summary.get("net_pnl_usd", 0.0)
    equity = summary.get("current_equity_usd", 0.0)

    # LLM stats
    llm_summary_path = run_dir.llm_summary_path
    llm_summary = {}
    if llm_summary_path.exists():
        try:
            llm_summary = json.loads(llm_summary_path.read_text())
        except Exception:
            pass
    llm_queued = llm_summary.get("total_losses_queued", 0)
    llm_analyzed = llm_summary.get("analyzed", 0)
    llm_pending = llm_summary.get("pending", 0)
    llm_failed = llm_summary.get("failed", 0)

    # Per-symbol stats
    symbols_stats = summary.get("symbols", {})

    # Build the dashboard
    lines = []
    lines.append("═" * 70)
    lines.append(f"BACKTEST RUN: {run_id}")
    lines.append("═" * 70)
    lines.append(f"Timeframe: {timeframe} | Started: {started_at[:19]}")
    lines.append("")

    # Current activity
    if current_symbol:
        lines.append(f"Active symbol: {current_symbol} (cursor {current_cursor})")
    lines.append("")

    # Trades
    lines.append("─── Trades ───")
    lines.append(f"  Total:     {total_trades:>8,}")
    lines.append(f"  Wins:      {wins:>8,}")
    lines.append(f"  Losses:    {losses:>8,}")
    lines.append(f"  Win Rate:  {win_rate:>7.2f}%")
    lines.append("")

    # P&L
    lines.append("─── P&L ───")
    lines.append(f"  Gross Profit: {_format_pnl(gross_profit):>12}")
    lines.append(f"  Gross Loss:   {_format_pnl(gross_loss):>12}")
    lines.append(f"  Net P&L:      {_format_pnl(net_pnl):>12}")
    lines.append(f"  Equity:       ${equity:>10,.2f}")
    lines.append("")

    # Per-symbol
    lines.append("─── Per-Symbol ───")
    if symbols_stats:
        lines.append(f"  {'Symbol':<10} {'Trades':>7} {'Wins':>6} {'Losses':>7} {'WR':>7} {'Net P&L':>14}")
        lines.append(f"  {'-'*10} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*14}")
        for sym in sorted(symbols_stats.keys()):
            s = symbols_stats[sym]
            wr = 100.0 * s.get("wins", 0) / max(s.get("trades", 1), 1)
            lines.append(f"  {sym:<10} {s.get('trades',0):>7} {s.get('wins',0):>6} "
                          f"{s.get('losses',0):>7} {wr:>6.2f}% "
                          f"{_format_pnl(s.get('net_pnl', 0)):>14}")
    else:
        lines.append("  (no per-symbol stats yet)")
    lines.append("")

    # LLM
    lines.append("─── LLM Loss Analysis ───")
    lines.append(f"  Losses queued: {llm_queued:>6,}")
    lines.append(f"  Analyzed:      {llm_analyzed:>6,}")
    lines.append(f"  Pending:       {llm_pending:>6,}")
    lines.append(f"  Failed:        {llm_failed:>6,}")
    if llm_summary.get("categories"):
        lines.append("  Categories:")
        for cat, count in sorted(llm_summary["categories"].items(),
                                   key=lambda x: -x[1]):
            lines.append(f"    {cat:<30} {count:>5}")
    lines.append("")

    # Updated timestamp
    lines.append(f"  Updated: {summary.get('updated_at', '?')[:19]}")
    lines.append("═" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Live backtest dashboard")
    parser.add_argument("--run-id", required=True, help="Run ID to display")
    parser.add_argument("--once", action="store_true",
                        help="Print once and exit (no refresh loop)")
    parser.add_argument("--refresh", type=float, default=2.0,
                        help="Refresh interval in seconds (default: 2)")
    args = parser.parse_args()

    run_dir = RunDir(args.run_id)
    if not run_dir.root.exists():
        print(f"Run directory does not exist: {run_dir.root}")
        sys.exit(1)

    if args.once:
        print(render_dashboard(run_dir))
        return

    # Refresh loop
    try:
        while True:
            # Clear screen (ANSI escape)
            print("\033[2J\033[H", end="")
            print(render_dashboard(run_dir))
            print(f"\n(Refreshing every {args.refresh}s — Ctrl+C to exit)")
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nDashboard exited.")


if __name__ == "__main__":
    main()
