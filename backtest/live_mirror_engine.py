"""Canonical historical replay engine for the live-trading mirror boundary.

Decision/risk/permission logic stays in AITrader.evaluate_decision_core().
This module owns only historical execution timing and replay exposure state.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from backtest.broker_sim import BrokerSimulator
from backtest.historical_execution_router import HistoricalExecutionRouter
from backtest.position_state import CanonicalPositionState
from core.clock import ReplayClock
from core.constants import COMMISSION_USD_PER_LOT, BROKER_SLIPPAGE_PIPS, reset_backtest_memory, set_backtest_mode
from core.data_provider import HistoricalMT5Provider
from backtest.metrics import calculate_metrics

@dataclass
class MirrorRunResult:
    symbol: str
    timeframe: str
    bars: int
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    rejection_stats: dict = field(default_factory=dict)
    metrics: object = None
    error: Optional[str] = None

def run_live_mirror_engine(*, symbol: str, df: pd.DataFrame, timeframe: str = "H1",
                            starting_balance: float = 10000.0, warmup_bars: int = 300,
                            max_open_trades: Optional[int] = None, max_hold_bars: int = 100,
                            spread_pips: Optional[float] = None,
                            commission_per_lot: Optional[float] = None,
                            slippage_pips: Optional[float] = None, clock=None,
                            verbose: bool = False) -> MirrorRunResult:
    """Replay the shared decision kernel with the canonical historical router."""
    from core.trader import AITrader
    from database.db import TraderDB
    from execution.paper_trader import PaperTrader
    import config

    replay_clock = clock or ReplayClock()
    if max_open_trades is None:
        max_open_trades = int(getattr(config, "MAX_OPEN_TRADES", 3))
    db = TraderDB(db_path="backtest/live_mirror_engine.db")
    paper = PaperTrader(starting_balance=starting_balance, db=db)
    trader = AITrader(balance=starting_balance, symbol=symbol, timeframe=timeframe,
                      paper_balance=starting_balance, execution_mode="backtest",
                      paper_trader=paper, db=db, clock=replay_clock)
    broker = BrokerSimulator(starting_balance=starting_balance,
        commission_per_lot=COMMISSION_USD_PER_LOT if commission_per_lot is None else commission_per_lot,
        slippage_pips=BROKER_SLIPPAGE_PIPS if slippage_pips is None else slippage_pips)
    state = CanonicalPositionState()
    router = HistoricalExecutionRouter(broker, max_open_positions=max_open_trades, position_state=state)
    provider = HistoricalMT5Provider(df.copy(deep=True), symbol, timeframe, clock=replay_clock)
    open_trades, closed_trades, entry_bar = [], [], {}
    equity = [starting_balance]
    stats = {"WAIT": 0, "NO_TRADE_ANALYSIS": 0, "risk_rejected": 0,
             "permission_blocked": 0, "engine_error": 0, "max_trades": 0,
             "execution_rejected": 0, "missing_levels": 0, "total_bars": 0}

    set_backtest_mode(True)
    reset_backtest_memory()
    import random, numpy as np
    random.seed(42); np.random.seed(42)

    for i in range(int(warmup_bars), len(df)):
        stats["total_bars"] += 1
        ts = df.index[i]
        replay_clock.advance(ts)

        # Existing positions are resolved against this bar only.
        still_open = []
        for trade in open_trades:
            opened = entry_bar[trade.trade_id]
            result = broker.check_exit(trade, float(df.iloc[i]["high"]), float(df.iloc[i]["low"]),
                                       float(df.iloc[i]["close"]), ts)
            if result is not None:
                result.hold_bars = i - opened
                closed_trades.append(result); router.on_close(trade, reason=result.exit_reason)
            elif i - opened > max_hold_bars:
                result = broker.close_trade(trade, float(df.iloc[i]["close"]), ts, "timeout")
                result.hold_bars = i - opened
                closed_trades.append(result); router.on_close(trade, reason="timeout")
            else:
                still_open.append(trade)
        open_trades = still_open

        # Fill orders submitted on earlier bars and evaluate pending limits.
        provider.advance_to(i)
        fills = router.advance(bar_index=i, bar_open=float(df.iloc[i]["open"]),
                                bar_high=float(df.iloc[i]["high"]), bar_low=float(df.iloc[i]["low"]),
                                bar_close=float(df.iloc[i]["close"]), bar_time=ts)
        for trade in fills:
            open_trades.append(trade); entry_bar[trade.trade_id] = i

        if state.count() >= int(max_open_trades):
            stats["max_trades"] += 1; equity.append(broker.get_balance()); continue

        try:
            market_out = provider.get_market_out(symbol, timeframe)
            session_ctx = {"current_session": replay_clock.current_session(),
                           "gmt_time": str(replay_clock.now()), "session_strategy": "n/a"}
            core = trader.evaluate_decision_core(market_out, session_ctx)
        except Exception as exc:
            stats["engine_error"] += 1
            if verbose: print(f"[{ts}] decision-core error: {exc}")
            equity.append(broker.get_balance()); continue

        analysis_out, dec_out = core["analysis_out"], core["dec_out"]
        risk_out, perm_out = core["risk_out"], core["perm_out"]
        if "error" in analysis_out:
            stats["NO_TRADE_ANALYSIS"] += 1; equity.append(broker.get_balance()); continue
        action = str(dec_out.get("decision", "WAIT")).upper()
        if action not in {"BUY", "SELL"}:
            stats["WAIT"] += 1; equity.append(broker.get_balance()); continue
        if not risk_out.get("approved"):
            stats["risk_rejected"] += 1; equity.append(broker.get_balance()); continue
        if not perm_out.get("allowed"):
            stats["permission_blocked"] += 1; equity.append(broker.get_balance()); continue

        sl, tp, lot = risk_out.get("sl_price"), risk_out.get("tp_price"), risk_out.get("lot")
        requested_entry = dec_out.get("entry") or dec_out.get("entry_price")
        if sl is None or tp is None or lot is None or float(lot) <= 0:
            stats["missing_levels"] += 1; equity.append(broker.get_balance()); continue
        if requested_entry is None: requested_entry = float(df.iloc[i]["close"])
        result = router.submit(symbol=symbol, direction=action, entry_price=float(requested_entry),
            sl=float(sl), tp=float(tp), lot=float(lot), confidence=int(dec_out.get("confidence", 0) or 0),
            bar_index=i, bar_time=ts,
            order_type=dec_out.get("order_type", dec_out.get("execution_type", "MARKET")))
        if result["status"] == "REJECTED": stats["execution_rejected"] += 1
        equity.append(broker.get_balance())

    router.cancel_all("end_of_backtest")
    last_ts, last_close = df.index[-1], float(df.iloc[-1]["close"])
    for trade in list(open_trades):
        result = broker.close_trade(trade, last_close, last_ts, "end_of_backtest")
        closed_trades.append(result); router.on_close(trade, reason="end_of_backtest")
    metrics = calculate_metrics(trades=closed_trades, starting_balance=starting_balance,
                                ending_balance=broker.get_balance())
    return MirrorRunResult(symbol=symbol, timeframe=timeframe, bars=len(df), trades=closed_trades,
                           equity_curve=equity, rejection_stats=stats, metrics=metrics)
