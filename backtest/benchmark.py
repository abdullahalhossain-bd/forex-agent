"""
backtest/benchmark.py — Main Benchmark Runner (Day 74)
========================================================

The main entry point for running performance benchmarks. Orchestrates:
  1. MLBacktest — run 3 system versions on historical data
  2. ComparisonEngine — compare results + detect ML improvement
  3. PerformanceReport — generate text report + Telegram alert
  4. MLBacktest diagnostic — feature importance analysis

Usage:
    from backtest.benchmark import run_benchmark
    result = run_benchmark(pair="EURUSD", timeframe="15m")

Or from CLI:
    python -m backtest.benchmark --pair EURUSD --timeframe 15m
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from utils.logger import get_logger

from backtest.ml_backtest import MLBacktest, BacktestMetrics
from backtest.comparison_engine import ComparisonEngine, ComparisonResult, get_comparison_engine
from backtest.performance_report import (
    generate_text_report,
    generate_telegram_report,
    generate_strategy_contribution,
)

log = get_logger("benchmark")


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    pair: str
    timeframe: str
    metrics: Dict[str, Dict[str, Any]]
    comparison: Dict[str, Any]
    strategy_contribution: Dict[str, float]
    text_report: str = ""
    telegram_report: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_benchmark(
    pair: str = "EURUSD",
    timeframe: str = "15m",
    limit: int = 2000,
    send_telegram: bool = True,
) -> BenchmarkResult:
    """Run a full performance benchmark.

    Args:
        pair: Trading pair to backtest.
        timeframe: Timeframe.
        limit: Number of historical candles.
        send_telegram: Whether to send Telegram alert with results.

    Returns:
        BenchmarkResult with all metrics + reports.
    """
    log.info(f"[Benchmark] Starting benchmark: {pair} {timeframe} | {limit} candles")

    # 1. Run backtests
    bt = MLBacktest()
    results = bt.run_backtest(pair=pair, timeframe=timeframe, limit=limit)

    if not results:
        log.error("[Benchmark] No backtest results — aborting")
        return BenchmarkResult(
            pair=pair, timeframe=timeframe,
            metrics={}, comparison={}, strategy_contribution={},
        )

    # 2. Compare systems
    comparator = get_comparison_engine()
    comparison = comparator.compare(results)

    # 3. Strategy contribution analysis
    contribution = generate_strategy_contribution(results)
    log.info(f"[Benchmark] Strategy contribution: {contribution}")

    # 4. Generate reports
    text_report = generate_text_report(results, comparison, pair, timeframe)
    telegram_report = generate_telegram_report(results, comparison, pair)

    print(text_report)

    # 5. Send Telegram alert
    if send_telegram and telegram_report:
        try:
            from core.service_registry import get_registry
            registry = get_registry()
            notifier = registry.try_resolve("telegram_notifier")
            if notifier:
                from utils.async_utils import run_coro_sync
                run_coro_sync(notifier.send_message(telegram_report))
                log.info("[Benchmark] Telegram report sent")
        except Exception as e:
            log.warning(f"[Benchmark] Telegram send failed: {e}")

    # 6. Build result
    result = BenchmarkResult(
        pair=pair,
        timeframe=timeframe,
        metrics={name: m.to_dict() for name, m in results.items()},
        comparison=comparison.to_dict(),
        strategy_contribution=contribution,
        text_report=text_report,
        telegram_report=telegram_report or "",
    )

    log.info(f"[Benchmark] Complete — winner: {comparison.winner} | ML improved: {comparison.ml_improved}")
    return result


def get_benchmark_history(limit: int = 20) -> list:
    """Get recent benchmark results from DB."""
    bt = MLBacktest()
    return bt.get_history(limit=limit)


# ── CLI entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FOREX AI performance benchmark")
    parser.add_argument("--pair", default="EURUSD", help="Trading pair")
    parser.add_argument("--timeframe", default="15m", help="Timeframe")
    parser.add_argument("--limit", type=int, default=2000, help="Number of candles")
    parser.add_argument("--no-telegram", action="store_true", help="Skip Telegram alert")
    args = parser.parse_args()

    result = run_benchmark(
        pair=args.pair,
        timeframe=args.timeframe,
        limit=args.limit,
        send_telegram=not args.no_telegram,
    )