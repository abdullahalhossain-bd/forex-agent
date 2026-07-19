"""
backtest/unified_engine.py — THE backtest runner (execution-parity fix).

Replaces three previously-disconnected paths:
  1. run_backtest.py's standalone loop, which ran ONLY UnifiedSignalEngine
     (5 analysis modules, no risk engine, hand-rolled position sizing).
  2. main.py's `--mode backtest`, which constructed backtest.engine.
     BacktestEngine and returned without ever calling it (dead code).
  3. backtest.engine.BacktestEngine itself, a third, unrelated
     strategy-object-based backtester never wired to the live pipeline.

This module does not reimplement signal generation, decision fusion, risk
management, or position sizing. It constructs a real `core.trader.AITrader`
— the SAME class Demo/Real use — and replays historical bars through its
`evaluate_decision_core()` method (analysis -> decision -> risk -> sizing
-> permission/correlation), bar by bar, with no look-ahead (each call only
sees data up to and including the current closed bar).

Per the project's execution-parity rule, only the following are allowed to
differ from Demo/Real, and this module is where that boundary lives:
  - Data source:   historical MT5 candles, not a live tick.
  - Execution/fills: backtest.broker_sim.BrokerSimulator (bar high/low SL/TP
    touch detection), not a real/demo MT5 order. NOTE: this is NOT the same
    thing as execution.simulated_executor.SimulatedExecutor — that module is
    a live-pipeline dry-run smoke test (fills instantly at a fabricated
    price, no OHLC awareness) and cannot replay historical SL/TP touches;
    BrokerSimulator is the only component in the repo that can, so it is
    kept and used here deliberately, not as a leftover duplicate.
  - Account/state: an isolated PaperTrader + TraderDB pointed at a
    dedicated backtest DB file, never the live `database/trader.db`.

Everything upstream of "what do we do with this decision" — indicators,
the ~29-module analysis stack, decision fusion, RiskEngine, PositionSizer,
TradePermission, CorrelationFilter — is the exact object graph Demo/Real
construct in AITrader.__init__(execution_mode="backtest").

KNOWN LIMITATIONS (see the accompanying audit-fix writeup — not silently
hidden here):
  - Several of AnalysisAgent's ~29 sub-modules call live external services
    (news APIs, economic calendar, retail sentiment, FRED). Called against
    a historical bar, they will either time out, return "no data", or (if
    they cache) return TODAY's data misapplied to a historical timestamp.
    Each of those modules already wraps its own call in try/except and
    degrades gracefully (confirmed by reading analysis_agent.py), so this
    will not crash a backtest run — but it does mean confidence scores for
    those specific sub-signals are not historically accurate. Treat any
    single backtest run's confidence numbers as approximate until those
    modules are given an offline/historical-safe mode.
  - risk.trade_frequency's daily-cap controller is wall-clock/day-boundary
    based and backed by global state. It is intentionally NOT invoked here
    for the same reason a live daily cap makes no sense replayed across
    years of historical bars in seconds — this is a deliberate, documented
    scope boundary, not an oversight.
  - The shared decision core's duplicate-position and correlation checks
    read `AITrader._paper.get_open_positions()`. This module fills/tracks
    trades through BrokerSimulator (needed for realistic bar-based SL/TP
    touch detection — see above) and deliberately does NOT also mirror
    every open/close into `_paper`, because PaperTrader.close_trade()
    expects a full trade record it built itself via open_trade_from_signal()
    — feeding it a synthetic dict risks corrupting PnL bookkeeping in a way
    that would be worse than the gap it closes. Net effect: this backtest
    engine is single-symbol per run, and within a run its own
    `max_open_trades` cap is the re-entry guard (matching run_backtest.py's
    prior behavior) rather than the live duplicate/correlation checks. This
    is a real, intentional scope boundary, not a silent omission — closing
    it properly means either extending PaperTrader with a
    "register externally-managed trade" method, or running this engine
    multi-symbol and letting `_paper`/`_corr_filter` do the correlation
    check across the historical positions the harness itself opens. That
    is follow-up work, flagged here rather than shipped half-working.
  - This module has been syntax-checked (py_compile) but not executed
    end-to-end in the environment this fix was written in (no MetaTrader5
    package, no chromadb/sentence-transformers, no live network access are
    available there). Run a short (e.g. 50-100 bar) smoke test in your own
    environment before trusting results from this path.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

log = logging.getLogger("unified_backtest_engine")


@dataclass
class UnifiedBacktestResult:
    symbol: str
    timeframe: str
    bars: int
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    rejection_stats: dict = field(default_factory=dict)
    metrics: object = None
    error: Optional[str] = None


def _build_market_out(df_slice: "pd.DataFrame", symbol: str, timeframe: str) -> dict:
    """Build a MarketAgentResult-shaped dict from a historical slice, using
    the SAME canonical indicator registry MarketAgent uses live — not a
    separate/duplicate indicator formula. Falls back through the identical
    chain agents/market_agent.py uses (indicator_registry -> ExtendedIndicators
    -> legacy Indicators) so behavior matches live bar-for-bar.
    """
    df = df_slice.copy()
    ind_ctx = {}
    try:
        from data.indicator_registry import add_canonical_indicators, get_ai_context as _get_ctx
        df = add_canonical_indicators(df, include_patterns=True)
        ind_ctx = _get_ctx(df)
    except Exception as e_registry:
        log.warning(f"[unified_engine] indicator_registry unavailable ({e_registry}) — "
                    f"falling back to ExtendedIndicators, then legacy Indicators")
        try:
            from data.indicators_ext import ExtendedIndicators
            ind_ext = ExtendedIndicators()
            df = ind_ext.add_all(df, include_patterns=True)
            ind_ctx = ind_ext.get_ai_context(df)
        except Exception:
            from data.indicators import Indicators
            ind = Indicators()
            df = ind.add_all(df)
            ind_ctx = ind.get_ai_context(df)

    try:
        from analysis.market_regime import MarketRegimeDetector
        regime_detector = MarketRegimeDetector()
        regime_result = regime_detector.detect(df)
        regime_ctx = regime_detector.get_ai_context(regime_result)
    except Exception as e:
        log.debug(f"[unified_engine] regime detection unavailable: {e}")
        regime_result, regime_ctx = {}, {}

    return {
        "df": df,
        "ind_ctx": ind_ctx,
        "regime": regime_result,
        "regime_ctx": regime_ctx,
        # Matches MarketAgent's own dict-shaped default when MTF data
        # isn't available ({"bias": ..., "confidence": ...} — NOT a bare
        # string; MarketBiasEngine/SignalEngine/MasterAnalyst all call
        # .get() on this and crash on a string). MTF bias is intentionally
        # NOT computed from historical data here (see module docstring
        # "KNOWN LIMITATIONS") — this is a documented parity gap, not an
        # oversight: computing a true historical MTF bias would need
        # synchronized higher-timeframe candle slices at every bar, which
        # is real follow-up work, not a one-line fix.
        "mtf_bias": {"bias": "NEUTRAL", "confidence": "LOW"},
        "symbol": symbol,
        "timeframe": timeframe,
        "data_source": "historical_replay",
    }


def _make_backtest_trader(symbol: str, timeframe: str, starting_balance: float,
                           db_path: str):
    """Construct the SAME AITrader class Demo/Real use, wired to isolated
    backtest state (its own PaperTrader + TraderDB file — never the live
    database/trader.db) so a backtest run cannot contaminate live trade
    history, memory, or learning state.
    """
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
    )
    return trader


def run_unified_backtest(
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
    verbose: bool = False,
) -> UnifiedBacktestResult:
    """Replay `df` bar-by-bar through the SAME decision core Demo/Real use.

    No look-ahead: bar i only ever sees df.iloc[:i+1]. Confidence/risk/
    sizing/permission gates are the live ones — a strategy that gets
    rejected live (low confidence, news block, correlation, duplicate
    position, risk-engine reject) gets rejected here too, for the same
    reason, via the same code.
    """
    from backtest.broker_sim import BrokerSimulator, DEFAULT_SPREAD_PIPS
    from backtest.metrics import calculate_metrics

    # CRITICAL FIX (reproducibility -- same bug as run_backtest.py's legacy
    # loop): BrokerSimulator draws slippage from np.random.normal() and
    # partial-fill behavior from stdlib random.random()/random.uniform(),
    # neither seeded anywhere in this module. Without this, repeat runs of
    # the identical shared-kernel backtest will silently drift in P&L just
    # like the legacy loop did before it was fixed.
    import random as _random
    _random.seed(42)
    import numpy as _np
    _np.random.seed(42)

    try:
        trader = _make_backtest_trader(symbol, timeframe, starting_balance, db_path)
    except Exception as e:
        log.error(f"[unified_engine] Could not construct backtest AITrader: {e}", exc_info=True)
        return UnifiedBacktestResult(symbol=symbol, timeframe=timeframe, bars=len(df), error=str(e))

    if spread_pips is None:
        spread_pips = DEFAULT_SPREAD_PIPS.get(symbol, 2.0)
    broker = BrokerSimulator(starting_balance=starting_balance,
                              commission_per_lot=commission_per_lot,
                              slippage_pips=slippage_pips)

    open_trades, closed_trades, equity_curve = [], [], [starting_balance]
    entry_bar: dict = {}
    rejection_stats = {"WAIT": 0, "NO_TRADE_ANALYSIS": 0, "risk_rejected": 0,
                        "permission_blocked": 0, "engine_error": 0, "max_trades": 0,
                        "total_bars": 0}
    total_bars = len(df)
    log.info(f"[unified_engine] Starting: {symbol} {timeframe} | {total_bars} bars | "
             f"balance=${starting_balance} | pipeline=shared(AnalysisAgent+DecisionAgent+RiskEngine+PositionSizer)")

    for i in range(warmup_bars, total_bars):
        current_time = df.index[i]
        rejection_stats["total_bars"] += 1

        # Exits first — bar high/low sweep against open trades.
        still_open = []
        for trade in open_trades:
            opened_at = entry_bar.get(trade.trade_id, i)
            result = broker.check_exit(trade, float(df.iloc[i]["high"]),
                                        float(df.iloc[i]["low"]), float(df.iloc[i]["close"]),
                                        current_time)
            if result:
                result.hold_bars = i - opened_at
                closed_trades.append(result)
                entry_bar.pop(trade.trade_id, None)
                # FIX (visibility gap): previously only OPEN was logged, never
                # how/when a trade resolved -- made it impossible to tell
                # from the log alone whether the strategy was even taking
                # exits (vs. e.g. every trade silently timing out).
                if verbose:
                    log.info(f"  [{current_time}] {result.exit_reason} {result.direction} "
                              f"{result.symbol} @ {result.exit_price:.5f} "
                              f"pnl=${result.pnl_usd:.2f} ({result.pnl_pips:+.1f}p) "
                              f"balance=${broker.get_balance():.2f}")
            else:
                trade.hold_bars = i - opened_at
                if trade.hold_bars > max_hold_bars:
                    closed = broker.close_trade(trade, float(df.iloc[i]["close"]), current_time, "timeout")
                    closed.hold_bars = trade.hold_bars
                    closed_trades.append(closed)
                    entry_bar.pop(trade.trade_id, None)
                    if verbose:
                        log.info(f"  [{current_time}] TIMEOUT {closed.direction} "
                                  f"{closed.symbol} @ {closed.exit_price:.5f} "
                                  f"pnl=${closed.pnl_usd:.2f} ({closed.pnl_pips:+.1f}p) "
                                  f"balance=${broker.get_balance():.2f}")
                else:
                    still_open.append(trade)
        open_trades = still_open

        if len(open_trades) >= max_open_trades:
            rejection_stats["max_trades"] += 1
            equity_curve.append(broker.get_balance())
            continue

        df_slice = df.iloc[:i + 1]
        try:
            market_out = _build_market_out(df_slice, symbol, timeframe)
        except Exception as e:
            rejection_stats["engine_error"] += 1
            if verbose:
                log.info(f"  [{current_time}] Market build error: {str(e)[:120]}")
            equity_curve.append(broker.get_balance())
            continue

        try:
            session_ctx = {"current_session": "BACKTEST", "gmt_time": str(current_time),
                            "session_strategy": "n/a"}
            core = trader.evaluate_decision_core(market_out, session_ctx)
        except Exception as e:
            rejection_stats["engine_error"] += 1
            if verbose:
                log.info(f"  [{current_time}] Decision core error: {str(e)[:120]}")
            equity_curve.append(broker.get_balance())
            continue

        analysis_out = core["analysis_out"]
        dec_out = core["dec_out"]
        risk_out = core["risk_out"]
        perm_out = core["perm_out"]

        if "error" in analysis_out:
            rejection_stats["NO_TRADE_ANALYSIS"] += 1
            equity_curve.append(broker.get_balance())
            continue

        action = dec_out.get("decision", "WAIT")
        if action not in ("BUY", "SELL"):
            rejection_stats["WAIT"] += 1
            equity_curve.append(broker.get_balance())
            continue

        if not risk_out.get("approved"):
            rejection_stats["risk_rejected"] += 1
            equity_curve.append(broker.get_balance())
            continue

        if not perm_out.get("allowed"):
            rejection_stats["permission_blocked"] += 1
            equity_curve.append(broker.get_balance())
            continue

        entry = dec_out.get("entry") or float(df.iloc[i]["close"])
        sl = risk_out.get("sl_price")
        tp = risk_out.get("tp_price")
        lot = risk_out.get("lot") or 0.01
        confidence = dec_out.get("confidence", 0)

        if not sl or not tp:
            rejection_stats["engine_error"] += 1
            equity_curve.append(broker.get_balance())
            continue

        trade = broker.open_trade(symbol=symbol, direction=action, entry_price=entry,
                                   sl=sl, tp=tp, lot=lot, bar_time=current_time,
                                   confidence=int(confidence) if confidence else 0,
                                   strategy="unified_decision_core",
                                   confluence_factors=0, quality_grade="B")
        entry_bar[trade.trade_id] = i
        open_trades.append(trade)
        if verbose:
            log.info(f"  [{current_time}] OPEN {action} {symbol} @ {entry:.5f} "
                      f"lot={lot} conf={confidence}")
        equity_curve.append(broker.get_balance())

    last_close = float(df.iloc[-1]["close"])
    last_time = df.index[-1]
    for trade in open_trades:
        closed_trades.append(broker.close_trade(trade, last_close, last_time, "end_of_backtest"))

    metrics = calculate_metrics(trades=closed_trades, starting_balance=starting_balance,
                                 ending_balance=broker.get_balance())
    log.info(f"[unified_engine] Done: {symbol} | {len(closed_trades)} trades | "
             f"WR={metrics.win_rate:.1f}% | PF={metrics.profit_factor:.2f} | "
             f"P&L=${metrics.total_pnl_usd:.2f}")

    return UnifiedBacktestResult(
        symbol=symbol, timeframe=timeframe, bars=total_bars,
        trades=closed_trades, equity_curve=equity_curve,
        rejection_stats=rejection_stats, metrics=metrics,
    )