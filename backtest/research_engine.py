"""
backtest/research_engine.py — Research-Grade Backtest Engine

Enhanced version of unified_engine.py that:
  - Tracks every module's contribution per bar
  - Records agent votes for every decision
  - Produces comprehensive trade replay data
  - Integrates Monte Carlo and Walk-Forward analysis
  - Runs automatic weakness detection
  - Generates institutional-quality reports (TXT + JSON + CSV)

This engine uses the EXACT same decision kernel as live trading:
  MarketAgent → AnalysisAgent → DecisionAgent → RiskEngine →
  PositionSizer → TradePermission → ExecutionSimulator
"""

from __future__ import annotations

import logging
import time
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("research_backtest_engine")


def run_research_backtest(
    symbol: str,
    df: "pd.DataFrame",
    timeframe: str = "H1",
    starting_balance: float = 10000.0,
    warmup_bars: int = 50,
    max_open_trades: int = 3,
    max_hold_bars: int = 100,
    spread_pips: Optional[float] = None,
    commission_per_lot: float = 7.0,
    slippage_pips: float = 2.0,
    db_path: str = "backtest/backtest_run.db",
    output_dir: str = "backtest/results",
    run_monte_carlo: bool = True,
    run_walk_forward: bool = True,
    mc_simulations: int = 1000,
    wf_windows: int = 5,
    verbose: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a full research-grade backtest with comprehensive reporting.

    Returns dict with:
      - report_txt: path to TXT report
      - report_json: path to JSON report
      - trades_csv: path to trades CSV
      - metrics: summary metrics dict
      - execution_time: seconds
    """
    start_time = time.time()

    # Seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    from backtest.broker_sim import BrokerSimulator, DEFAULT_SPREAD_PIPS, SimulatedTrade
    from backtest.trade_replay import TradeReplayLogger, TradeReplayEntry
    from backtest.report_generator import (
        BacktestReportGenerator, ReportData, RunMetadata,
        analyze_trades,
    )
    from backtest.weakness_detector import WeaknessDetector

    # ----------------------------------------------------------
    # Setup
    # ----------------------------------------------------------
    if spread_pips is None:
        spread_pips = DEFAULT_SPREAD_PIPS.get(symbol, 2.0)

    broker = BrokerSimulator(
        starting_balance=starting_balance,
        commission_per_lot=commission_per_lot,
        slippage_pips=slippage_pips,
    )

    replay = TradeReplayLogger(output_dir=output_dir)
    report_gen = BacktestReportGenerator(output_dir=output_dir)
    detector = WeaknessDetector(min_trades_for_analysis=5)

    open_trades: List[SimulatedTrade] = []
    closed_trades: List[SimulatedTrade] = []
    equity_curve: List[float] = [starting_balance]
    entry_bar: Dict[int, int] = {}

    rejection_stats: Dict[str, int] = {
        "WAIT": 0,
        "NO_TRADE_ANALYSIS": 0,
        "risk_rejected": 0,
        "permission_blocked": 0,
        "engine_error": 0,
        "max_trades": 0,
        "total_bars": 0,
        "low_confidence": 0,
        "news_blocked": 0,
        "session_blocked": 0,
        "correlation_blocked": 0,
        "duplicate_blocked": 0,
    }

    total_bars = len(df)
    log.info(f"[ResearchEngine] Starting: {symbol} {timeframe} | {total_bars} bars | "
             f"balance=${starting_balance} | seed={seed}")

    # ----------------------------------------------------------
    # Construct the shared AITrader (same as live)
    # ----------------------------------------------------------
    try:
        trader = _make_backtest_trader(symbol, timeframe, starting_balance, db_path)
    except Exception as e:
        log.error(f"[ResearchEngine] Could not construct AITrader: {e}", exc_info=True)
        execution_time = time.time() - start_time
        return {"error": str(e), "execution_time": execution_time}

    # ----------------------------------------------------------
    # Bar-by-bar replay
    # ----------------------------------------------------------
    for i in range(warmup_bars, total_bars):
        current_time = df.index[i]
        rejection_stats["total_bars"] += 1

        # --- Exits first ---
        still_open = []
        for trade in open_trades:
            opened_at = entry_bar.get(trade.trade_id, i)
            result = broker.check_exit(
                trade,
                float(df.iloc[i]["high"]),
                float(df.iloc[i]["low"]),
                float(df.iloc[i]["close"]),
                current_time,
            )
            if result:
                result.hold_bars = i - opened_at
                closed_trades.append(result)
                entry_bar.pop(trade.trade_id, None)

                # Log to replay
                replay.log_trade(TradeReplayEntry(
                    trade_id=trade.trade_id,
                    bar_index=i,
                    chart_time=str(current_time),
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=trade.direction,
                    entry_price=trade.entry_price,
                    exit_price=result.exit_price,
                    sl_price=trade.stop_loss,
                    tp_price=trade.take_profit,
                    lot_size=trade.lot_size,
                    pnl_pips=result.pnl_pips,
                    pnl_usd=result.pnl_usd,
                    exit_reason=result.exit_reason,
                    hold_bars=result.hold_bars,
                    commission_usd=trade.commission_usd,
                    slippage_pips=trade.slippage_pips,
                    confidence=trade.confidence,
                    strategy=trade.strategy,
                ))

                if verbose and i % 100 == 0:
                    log.info(f"  [{current_time}] {result.exit_reason} {trade.direction} "
                              f"{symbol} @ {result.exit_price:.5f} "
                              f"pnl=${result.pnl_usd:.2f}")
            else:
                trade.hold_bars = i - opened_at
                if trade.hold_bars > max_hold_bars:
                    closed = broker.close_trade(trade, float(df.iloc[i]["close"]), current_time, "timeout")
                    closed.hold_bars = trade.hold_bars
                    closed_trades.append(closed)
                    entry_bar.pop(trade.trade_id, None)
                else:
                    still_open.append(trade)
        open_trades = still_open

        # Max trades guard
        if len(open_trades) >= max_open_trades:
            rejection_stats["max_trades"] += 1
            equity_curve.append(broker.get_balance())
            continue

        # --- Build market_out ---
        try:
            market_out = _build_market_out(df, i, symbol, timeframe)
        except Exception as e:
            rejection_stats["engine_error"] += 1
            equity_curve.append(broker.get_balance())
            continue

        # --- Run shared decision core ---
        try:
            session_ctx = {
                "current_session": _detect_session(current_time),
                "gmt_time": str(current_time),
                "session_strategy": "n/a",
            }
            core = trader.evaluate_decision_core(market_out, session_ctx)
        except Exception as e:
            rejection_stats["engine_error"] += 1
            if verbose and i % 200 == 0:
                log.info(f"  [{current_time}] Decision core error: {str(e)[:120]}")
            equity_curve.append(broker.get_balance())
            continue

        analysis_out = core.get("analysis_out", {})
        dec_out = core.get("dec_out", {})
        risk_out = core.get("risk_out", {})
        perm_out = core.get("perm_out", {})

        # Extract agent votes for replay
        agent_votes = {}
        if "agent_votes" in dec_out:
            agent_votes = dec_out["agent_votes"]
        elif "vote_detail" in dec_out:
            agent_votes = dec_out["vote_detail"]

        # Analysis error
        if "error" in analysis_out:
            rejection_stats["NO_TRADE_ANALYSIS"] += 1
            equity_curve.append(broker.get_balance())
            continue

        # No trade decision
        action = dec_out.get("decision", "WAIT")
        if action not in ("BUY", "SELL"):
            rej_reason = "WAIT"
            if dec_out.get("reject_reason"):
                rej_reason = f"WAIT ({dec_out['reject_reason']})"
            rejection_stats["WAIT"] += 1
            replay.log_rejection(
                bar_index=i, chart_time=str(current_time),
                symbol=symbol, timeframe=timeframe,
                reason=rej_reason,
                confidence=dec_out.get("confidence", 0),
                agent_votes=agent_votes,
            )
            equity_curve.append(broker.get_balance())
            continue

        # Risk rejected
        if not risk_out.get("approved"):
            rej_reason = risk_out.get("reject_reason", "risk_rejected")
            rejection_stats["risk_rejected"] += 1
            replay.log_rejection(
                bar_index=i, chart_time=str(current_time),
                symbol=symbol, timeframe=timeframe,
                reason=f"Risk: {rej_reason}",
                confidence=dec_out.get("confidence", 0),
                direction=action,
                agent_votes=agent_votes,
            )
            equity_curve.append(broker.get_balance())
            continue

        # Permission blocked
        if not perm_out.get("allowed"):
            rej_reason = perm_out.get("reason", "permission_blocked")
            rejection_stats["permission_blocked"] += 1
            replay.log_rejection(
                bar_index=i, chart_time=str(current_time),
                symbol=symbol, timeframe=timeframe,
                reason=f"Permission: {rej_reason}",
                confidence=dec_out.get("confidence", 0),
                direction=action,
                agent_votes=agent_votes,
            )
            equity_curve.append(broker.get_balance())
            continue

        # --- Execute trade ---
        entry = dec_out.get("entry") or float(df.iloc[i]["close"])
        sl = risk_out.get("sl_price")
        tp = risk_out.get("tp_price")
        lot = risk_out.get("lot") or 0.01
        confidence = dec_out.get("confidence", 0)

        if not sl or not tp:
            rejection_stats["engine_error"] += 1
            equity_curve.append(broker.get_balance())
            continue

        trade = broker.open_trade(
            symbol=symbol,
            direction=action,
            entry_price=entry,
            sl=sl,
            tp=tp,
            lot=lot,
            bar_time=current_time,
            confidence=int(confidence) if confidence else 0,
            strategy=_extract_strategy(dec_out, analysis_out),
        )
        entry_bar[trade.trade_id] = i
        open_trades.append(trade)

        equity_curve.append(broker.get_balance())

        if verbose and (len(closed_trades) < 20 or len(closed_trades) % 10 == 0):
            log.info(f"  [{current_time}] OPEN {action} {symbol} @ {entry:.5f} "
                      f"lot={lot} conf={confidence:.0f}% balance=${broker.get_balance():.2f}")

    # Close remaining open trades
    last_close = float(df.iloc[-1]["close"])
    last_time = df.index[-1]
    for trade in open_trades:
        closed = broker.close_trade(trade, last_close, last_time, "end_of_backtest")
        closed.hold_bars = trade.hold_bars
        closed_trades.append(closed)

    # ----------------------------------------------------------
    # Build Report
    # ----------------------------------------------------------
    execution_time = time.time() - start_time
    ending_balance = broker.get_balance()

    # Get git commit
    git_commit = _get_git_commit()

    metadata = RunMetadata(
        run_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        execution_time_sec=round(execution_time, 1),
        git_commit=git_commit,
        symbol=symbol,
        timeframe=timeframe,
        total_bars=total_bars,
        warmup_bars=warmup_bars,
        starting_balance=starting_balance,
        spread_pips=spread_pips,
        commission_per_lot=commission_per_lot,
        slippage_pips=slippage_pips,
    )

    report_data = analyze_trades(
        trades=closed_trades,
        starting_balance=starting_balance,
        ending_balance=ending_balance,
        equity_curve=equity_curve,
        rejection_stats=rejection_stats,
        metadata=metadata,
    )

    # Monte Carlo
    if run_monte_carlo and len(closed_trades) >= 20:
        try:
            mc_result = _run_monte_carlo(closed_trades, starting_balance, mc_simulations)
            report_data.monte_carlo = mc_result
        except Exception as e:
            log.warning(f"[ResearchEngine] Monte Carlo failed: {e}")
    else:
        report_data.monte_carlo = {"n_simulations": 0, "detail": "Insufficient trades or disabled"}

    # Walk-Forward
    if run_walk_forward and len(closed_trades) >= 30:
        try:
            wf_result = _run_walk_forward(closed_trades, wf_windows)
            report_data.walk_forward = wf_result
        except Exception as e:
            log.warning(f"[ResearchEngine] Walk-Forward failed: {e}")
    else:
        report_data.walk_forward = {"total_windows": 0, "detail": "Insufficient trades or disabled"}

    # Weakness Detection
    try:
        wd_result = detector.analyze(report_data)
        report_data.weakness_detection = wd_result
    except Exception as e:
        log.warning(f"[ResearchEngine] Weakness detection failed: {e}")
        report_data.weakness_detection = {}

    # Trade replay summary
    report_data.trade_replay_summary = replay.summary()

    # Save replay files
    replay.save_csv(f"replay_{symbol}_{timeframe}.csv")
    replay.save_json(f"replay_{symbol}_{timeframe}.json")

    # Generate reports
    report_paths = report_gen.generate(report_data)

    # Print summary
    s = report_data.summary
    log.info(f"[ResearchEngine] Done: {symbol} | {len(closed_trades)} trades | "
             f"WR={s.win_rate:.1f}% | PF={s.profit_factor:.2f} | "
             f"P&L=${s.net_profit:+,.2f} | Sharpe={s.sharpe:.2f} | "
             f"MaxDD={s.max_drawdown_pct:.1f}% | Time={execution_time:.1f}s")

    if report_data.weakness_detection.get("overall_grade"):
        log.info(f"[ResearchEngine] System Grade: {report_data.weakness_detection['overall_grade']}")

    # Print key findings
    for finding in report_data.weakness_detection.get("findings", [])[:5]:
        log.info(f"  [{finding['severity']}] {finding['title']}")

    return {
        "report_txt": report_paths.get("txt", ""),
        "report_json": report_paths.get("json", ""),
        "trades_csv": report_paths.get("csv", ""),
        "metrics": s.to_dict(),
        "execution_time": execution_time,
        "total_trades": len(closed_trades),
        "win_rate": s.win_rate,
        "profit_factor": s.profit_factor,
        "net_profit": s.net_profit,
        "sharpe": s.sharpe,
        "max_drawdown_pct": s.max_drawdown_pct,
        "system_grade": report_data.weakness_detection.get("overall_grade", "N/A"),
    }


# ============================================================
# Helper Functions
# ============================================================

def _make_backtest_trader(symbol, timeframe, starting_balance, db_path):
    """Construct the SAME AITrader class Demo/Real use."""
    from core.trader import AITrader
    from database.db import TraderDB
    from execution.paper_trader import PaperTrader

    db = TraderDB(db_path=db_path)
    paper = PaperTrader(starting_balance=starting_balance, db=db)

    trader = AITrader(
        balance=starting_balance,
        symbol=symbol,
        timeframe=timeframe,
        paper_balance=starting_balance,
        execution_mode="backtest",
        paper_trader=paper,
        db=db,
    )
    return trader


def _build_market_out(df, i, symbol, timeframe):
    """Build market_out dict from historical data at bar i."""
    from core.data_provider import HistoricalMT5Provider
    provider = HistoricalMT5Provider(df, symbol, timeframe)
    provider.advance_to(i)
    return provider.get_market_out(symbol, timeframe)


def _detect_session(dt) -> str:
    """Detect trading session from datetime."""
    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        h = dt.hour if hasattr(dt, "hour") else 12
        if 0 <= h < 7:
            return "Asian"
        elif 7 <= h < 13:
            return "London"
        elif 13 <= h < 16:
            return "London-NY"
        elif 16 <= h < 22:
            return "New York"
        else:
            return "Off-Hours"
    except Exception:
        return "Unknown"


def _extract_strategy(dec_out, analysis_out):
    """Extract strategy name from decision/analysis output."""
    strategy = dec_out.get("strategy", "")
    if strategy:
        return strategy
    # Try analysis
    strat_ctx = analysis_out.get("strategy", {})
    if isinstance(strat_ctx, dict):
        return strat_ctx.get("strategy", "unified_decision_core")
    return "unified_decision_core"


def _get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd="/home/z/my-project/forex-agent",
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _run_monte_carlo(trades, starting_balance, n_sims=1000):
    """Run Monte Carlo simulation on trade results."""
    try:
        from risk.monte_carlo import MonteCarloEngine
    except ImportError:
        return {"n_simulations": 0, "detail": "MonteCarloEngine not available"}

    if not trades:
        return {"n_simulations": 0, "detail": "No trades"}

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd < 0]
    n = len(trades)
    wr = len(wins) / n

    avg_win_pct = (np.mean([t.pnl_usd for t in wins]) / starting_balance * 100) if wins else 1.0
    avg_loss_pct = (np.mean([abs(t.pnl_usd) for t in losses]) / starting_balance * 100) if losses else 1.0
    risk_pct = (np.mean([abs(t.pnl_usd) for t in losses]) / starting_balance * 100) if losses else 1.0

    mc = MonteCarloEngine(seed=42)
    result = mc.run(
        win_rate=wr,
        avg_win_pct=max(avg_win_pct, 0.1),
        avg_loss_pct=max(avg_loss_pct, 0.1),
        n_simulations=n_sims,
        n_trades=n,
        initial_balance=starting_balance,
        risk_per_trade=max(risk_pct / 100, 0.001),
    )
    return result


def _run_walk_forward(trades, n_windows=5):
    """Run walk-forward analysis on trade results."""
    try:
        from backtest.walk_forward import run_walk_forward
    except ImportError:
        return {"total_windows": 0, "detail": "walk_forward module not available"}

    result = run_walk_forward(trades, n_windows=n_windows)
    return result.to_dict()
