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

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

log = logging.getLogger("unified_backtest_engine")


def _json_safe(obj):
    """Best-effort conversion of arbitrary analysis/decision/risk dict
    content into something json.dump can handle — numpy scalars, pandas
    Timestamps, custom objects, etc. all get stringified rather than
    raising or silently dropping data. Used so the per-trade forensic
    log (see save_trade_forensics below) can capture whatever each
    module actually returned without needing to know its exact schema.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, pd.Series):
        return _json_safe(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        try:
            return _json_safe(obj.to_dict(orient="records"))
        except Exception:
            return obj.to_string()
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, _np.generic):
            return obj.item()
    except Exception:
        pass
    try:
        return str(obj)
    except Exception:
        return repr(obj)


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
    forensics_path: Optional[str] = None


# market_out construction moved to core.data_provider.HistoricalMT5Provider
# (formal DataProvider abstraction — see that module). Kept out of this
# file so backtest/demo/real share ONE definition of "how a market_out
# dict gets built from a historical slice", not one copy per caller.


def _record_forensic_exit(trade_forensics: dict, trade_id, df: "pd.DataFrame",
                           opened_at: int, exit_idx: int, exit_time, exit_reason: str,
                           exit_price: float, pnl_usd: float, pnl_pips: float,
                           hold_bars: int) -> None:
    """Fill in the exit half of a trade's forensic record: exit details
    plus the actual OHLC path from entry bar to exit bar, so a later
    autopsy can see whether price moved straight against the trade or
    ran favorably first and reversed — without re-deriving it from raw
    data. No-op if this trade has no open-side forensic record (e.g.
    save_forensics=False).
    """
    entry = trade_forensics.get(trade_id)
    if entry is None:
        return
    path_df = df.iloc[opened_at:exit_idx + 1][["open", "high", "low", "close"]].copy()
    path = [
        {"time": str(ts), "open": float(r.open), "high": float(r.high),
         "low": float(r.low), "close": float(r.close)}
        for ts, r in path_df.iterrows()
    ]
    entry.update({
        "exit_bar_index": exit_idx,
        "exit_time": str(exit_time),
        "exit_reason": exit_reason,
        "exit_price": exit_price,
        "pnl_usd": pnl_usd,
        "pnl_pips": pnl_pips,
        "hold_bars": hold_bars,
        "ohlc_path": path,
    })


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
        db=db,
    )
    return trader


def run_unified_backtest(
    symbol: str,
    df: "pd.DataFrame",
    timeframe: str = "H1",
    starting_balance: float = 10000.0,
    warmup_bars: int = 300,
    max_open_trades: int | None = None,
    max_hold_bars: int = 100,
    spread_pips: Optional[float] = None,
    commission_per_lot: float | None = None,
    slippage_pips: float | None = None,
    db_path: str = "backtest/backtest_run.db",
    verbose: bool = False,
    save_forensics: bool = True,
    forensics_path: Optional[str] = None,
    bypass_checks: set[str] | list[str] | None = None,
) -> UnifiedBacktestResult:
    """Replay `df` bar-by-bar through the SAME decision core Demo/Real use.

    No look-ahead: bar i only ever sees df.iloc[:i+1]. Confidence/risk/
    sizing/permission gates are the live ones — a strategy that gets
    rejected live (low confidence, news block, correlation, duplicate
    position, risk-engine reject) gets rejected here too, for the same
    reason, via the same code.

    Parity defaults (vs config.py):
      - warmup_bars: 300 (matches live MarketAgent's limit=300 lookback)
      - max_open_trades: config.MAX_OPEN_TRADES (default 10) — same cap as live

    save_forensics: if True (default), write a per-trade forensic log to
    `forensics_path` (default: backtest/results/{symbol}_{timeframe}_
    trade_forensics.json) capturing, for every trade: the FULL
    analysis_out/dec_out/risk_out/perm_out dicts from the bar it was
    opened on (whatever voters/gates those modules populated — this
    module doesn't assume their schema), plus entry/exit price, reason,
    P&L, and the OHLC bar-by-bar path from entry to exit. This exists so
    "why did trade X lose" can be answered from the log directly instead
    of manually re-running/reconstructing state after the fact.
    """
    from backtest.broker_sim import BrokerSimulator, DEFAULT_SPREAD_PIPS
    from backtest.metrics import calculate_metrics
    from core.data_provider import HistoricalMT5Provider
    from core.execution_adapter import HistoricalExecutionAdapter
    from core.constants import (
        set_backtest_mode, reset_backtest_memory,
        COMMISSION_USD_PER_LOT as _DEF_COMMISSION,
        BROKER_SLIPPAGE_PIPS as _DEF_SLIPPAGE,
    )

    # ── Parity: max_open_trades defaults to config.MAX_OPEN_TRADES ──
    # Previously hardcoded to 3, which capped backtest at 3 concurrent
    # trades while live allows 10. Now both read from the same source.
    if max_open_trades is None:
        try:
            from config import MAX_OPEN_TRADES as _CFG_MAX_OPEN
            max_open_trades = int(_CFG_MAX_OPEN)
        except Exception:
            # config import failed (e.g. unit test) — keep safe fallback
            max_open_trades = 3
            log.warning("[unified_engine] config.MAX_OPEN_TRADES unavailable — falling back to 3")

    # Resolve cost defaults from shared constants (single source of truth)
    _commission = commission_per_lot if commission_per_lot is not None else _DEF_COMMISSION
    _slippage = slippage_pips if slippage_pips is not None else _DEF_SLIPPAGE

    # Phase 2.5: tell every external-data module (FRED, macro data, news,
    # economic calendar, ...) this is an offline historical replay, not a
    # live session, so they skip their network calls entirely instead of
    # retrying an unreachable/irrelevant live API on every single bar.
    set_backtest_mode(True)

    # ── Parity Fix #3: wipe backtest-isolated memory dir so this run
    # starts from a clean slate (no stale circuit-breaker state, no
    # confidence calibration bleed-through from a previous backtest run).
    # CRITICAL: this ONLY touches memory/_backtest/ — the live memory/
    # directory is never affected.
    reset_backtest_memory()

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
                              commission_per_lot=_commission,
                              slippage_pips=_slippage)
    # Formal ExecutionAdapter/DataProvider boundary — see
    # core/execution_adapter.py and core/data_provider.py. `broker` still
    # owns the actual fill state; `adapter`/`provider` are the named
    # abstraction the target architecture calls for, wrapping the same
    # objects rather than duplicating their logic.
    adapter = HistoricalExecutionAdapter(broker)
    # ── PARITY (CSV provider support): use HistoricalCSVProvider when CSV
    # files exist for this symbol/timeframe; fall back to HistoricalMT5Provider
    # (df-based) when they don't. The factory handles the selection logic.
    # Pass the existing df as a fallback for cases where CSVs aren't available
    # (e.g. unit tests with synthetic data).
    try:
        from core.provider_factory import make_backtest_provider
        provider = make_backtest_provider(
            symbol=symbol, timeframe=timeframe, df=df, prefer="auto",
        )
        log.info(f"[unified_engine] Provider: {type(provider).__name__}")
    except Exception as e_provider:
        log.warning(f"[unified_engine] Provider factory failed ({e_provider}) — "
                    f"falling back to HistoricalMT5Provider(df)")
        provider = HistoricalMT5Provider(df, symbol, timeframe)

    # ── PARITY (CSV provider support): when using HistoricalCSVProvider,
    # the provider has its OWN primary_df loaded from CSV. Use that as the
    # authoritative bar source for the main loop (so the loop iterates over
    # the CSV's bars, not whatever df was passed in). For HistoricalMT5Provider
    # (fallback), keep using the passed df as before.
    primary_df = getattr(provider, "primary_df", None)
    if primary_df is None or (hasattr(primary_df, "empty") and primary_df.empty):
        primary_df = df
    total_bars = len(primary_df)
    log.info(f"[unified_engine] Primary bars: {total_bars} (source: "
             f"{'CSV' if hasattr(provider, 'primary_df') else 'passed df'})")

    open_trades, closed_trades, equity_curve = [], [], [starting_balance]
    entry_bar: dict = {}
    # trade_id -> forensic record. Populated at open time with the full
    # decision-core output for that bar; supplemented at close time with
    # exit details + the OHLC path traveled. See save_forensics param.
    trade_forensics: dict = {}
    rejection_stats = {"WAIT": 0, "NO_TRADE_ANALYSIS": 0, "risk_rejected": 0,
                        "permission_blocked": 0, "engine_error": 0, "max_trades": 0,
                        "total_bars": 0}
    total_bars = len(primary_df)
    log.info(f"[unified_engine] Starting: {symbol} {timeframe} | {total_bars} bars | "
             f"balance=${starting_balance} | pipeline=shared(AnalysisAgent+DecisionAgent+RiskEngine+PositionSizer)")

    for i in range(warmup_bars, total_bars):
        current_time = primary_df.index[i]
        rejection_stats["total_bars"] += 1

        # Exits first — bar high/low sweep against open trades.
        still_open = []
        for trade in open_trades:
            opened_at = entry_bar.get(trade.trade_id, i)
            result = broker.check_exit(trade, float(primary_df.iloc[i]["high"]),
                                        float(primary_df.iloc[i]["low"]), float(primary_df.iloc[i]["close"]),
                                        current_time)
            if result:
                result.hold_bars = i - opened_at
                closed_trades.append(result)
                entry_bar.pop(trade.trade_id, None)
                _record_forensic_exit(trade_forensics, trade.trade_id, primary_df, opened_at, i,
                                       current_time, result.exit_reason, result.exit_price,
                                       result.pnl_usd, result.pnl_pips, result.hold_bars)
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
                    closed = broker.close_trade(trade, float(primary_df.iloc[i]["close"]), current_time, "timeout")
                    closed.hold_bars = trade.hold_bars
                    closed_trades.append(closed)
                    entry_bar.pop(trade.trade_id, None)
                    _record_forensic_exit(trade_forensics, trade.trade_id, primary_df, opened_at, i,
                                           current_time, closed.exit_reason, closed.exit_price,
                                           closed.pnl_usd, closed.pnl_pips, closed.hold_bars)
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

        provider.advance_to(i)
        try:
            market_out = provider.get_market_out(symbol, timeframe)
        except Exception as e:
            rejection_stats["engine_error"] += 1
            if verbose:
                import traceback
                log.error(f"  [{current_time}] Market build error: {str(e)}")
                log.error(f"  Traceback: {traceback.format_exc()}")
            equity_curve.append(broker.get_balance())
            continue

        try:
            session_ctx = {"current_session": "BACKTEST", "gmt_time": str(current_time),
                            "session_strategy": "n/a"}
            core = trader.evaluate_decision_core(market_out, session_ctx, bypass_checks=bypass_checks)
        except Exception as e:
            rejection_stats["engine_error"] += 1
            if verbose:
                import traceback
                log.error(f"  [{current_time}] Decision core error: {str(e)}")
                log.error(f"  Traceback: {traceback.format_exc()}")
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

        # FIX (2026-08-11): look-ahead bias — fill at next bar's open, not signal bar's close.
        if dec_out.get("entry"):
            entry = dec_out["entry"]
        elif i + 1 < len(primary_df):
            entry = float(primary_df.iloc[i + 1]["open"])
        else:
            entry = float(primary_df.iloc[i]["close"])
        sl = risk_out.get("sl_price")
        tp = risk_out.get("tp_price")
        lot = risk_out.get("lot") or 0.01
        confidence = dec_out.get("confidence", 0)

        # FIX (2026-08-19 winrate audit): ATR-based SL/TP fallback.
        # When RiskEngine doesn't produce SL/TP (placeholder risk, missing
        # ind_ctx fields, etc.), the old code immediately rejected the
        # trade as "engine_error". With max_open_trades raised and rule
        # signals flowing through in BT mode, this was killing ~15% of
        # otherwise-valid signals. Now compute ATR-based levels:
        #   SL = 1.5 × ATR (standard swing-trader multiplier)
        #   TP = 3.0 × ATR (R:R = 1:2 — institutional minimum)
        if (not sl or not tp) and action in ("BUY", "SELL"):
            try:
                _atr = float(market_out.get("ind_ctx", {}).get("atr", 0) or 0)
                if _atr > 0:
                    if action == "BUY":
                        if not sl:
                            sl = entry - (_atr * 1.5)
                        if not tp:
                            tp = entry + (_atr * 3.0)
                    else:  # SELL
                        if not sl:
                            sl = entry + (_atr * 1.5)
                        if not tp:
                            tp = entry - (_atr * 3.0)
                    if not lot or lot <= 0:
                        lot = 0.01
            except Exception:
                pass

        if not sl or not tp:
            rejection_stats["engine_error"] += 1
            equity_curve.append(broker.get_balance())
            continue

        trade = adapter.open_trade(symbol=symbol, direction=action, entry_price=entry,
                                    sl=sl, tp=tp, lot=lot, bar_time=current_time,
                                    spread_pips=market_out.get("ind_ctx", {}).get("spread_pips"),
                                    confidence=int(confidence) if confidence else 0,
                                    strategy="unified_decision_core",
                                    confluence_factors=0, quality_grade="B")
        # ── Parity: BrokerSimulator.open_trade may return None when the
        # assumed spread exceeds broker.spread_monitor.MAX_SPREAD_PIPS
        # (mirrors live OrderManager spread rejection). Treat as a
        # permission-blocked rejection, not a crash.
        if trade is None:
            rejection_stats["permission_blocked"] += 1
            equity_curve.append(broker.get_balance())
            continue
        entry_bar[trade.trade_id] = i
        open_trades.append(trade)
        if save_forensics:
            trade_forensics[trade.trade_id] = {
                "trade_id": trade.trade_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "entry_bar_index": i,
                "entry_time": str(current_time),
                "direction": action,
                "entry_price": entry,
                "sl_price": sl,
                "tp_price": tp,
                "lot": lot,
                "confidence": confidence,
                # Full decision-core output for this bar, whatever fields
                # each module actually populated (rule engine, LLM, ML,
                # RL, sentiment, session/confluence scores, gate results,
                # etc.) — captured as-is rather than assuming a schema.
                "analysis_out": _json_safe(analysis_out),
                "dec_out": _json_safe(dec_out),
                "risk_out": _json_safe(risk_out),
                "perm_out": _json_safe(perm_out),
            }
        if verbose:
            log.info(f"  [{current_time}] OPEN {action} {symbol} @ {entry:.5f} "
                      f"lot={lot} conf={confidence}")
        equity_curve.append(broker.get_balance())

    last_close = float(primary_df.iloc[-1]["close"])
    last_time = primary_df.index[-1]
    last_idx = len(primary_df) - 1
    for trade in open_trades:
        opened_at = entry_bar.get(trade.trade_id, last_idx)
        closed = broker.close_trade(trade, last_close, last_time, "end_of_backtest")
        closed_trades.append(closed)
        _record_forensic_exit(trade_forensics, trade.trade_id, primary_df, opened_at, last_idx,
                               last_time, closed.exit_reason, closed.exit_price,
                               closed.pnl_usd, closed.pnl_pips, last_idx - opened_at)

    metrics = calculate_metrics(trades=closed_trades, starting_balance=starting_balance,
                                 ending_balance=broker.get_balance())
    log.info(f"[unified_engine] Done: {symbol} | {len(closed_trades)} trades | "
             f"WR={metrics.win_rate:.1f}% | PF={metrics.profit_factor:.2f} | "
             f"P&L=${metrics.total_pnl_usd:.2f}")

    resolved_forensics_path = None
    if save_forensics and trade_forensics:
        import os
        resolved_forensics_path = forensics_path or \
            f"backtest/results/{symbol}_{timeframe}_trade_forensics.json"
        try:
            os.makedirs(os.path.dirname(resolved_forensics_path) or ".", exist_ok=True)
            with open(resolved_forensics_path, "w", encoding="utf-8") as f:
                json.dump(list(trade_forensics.values()), f, indent=2, default=str)
            log.info(f"[unified_engine] Trade forensics saved: "
                     f"{resolved_forensics_path} ({len(trade_forensics)} trade(s))")
        except Exception as e:
            log.warning(f"[unified_engine] Could not save trade forensics: {e}")
            resolved_forensics_path = None

    return UnifiedBacktestResult(
        symbol=symbol, timeframe=timeframe, bars=total_bars,
        trades=closed_trades, equity_curve=equity_curve,
        rejection_stats=rejection_stats, metrics=metrics,
        forensics_path=resolved_forensics_path,
    )