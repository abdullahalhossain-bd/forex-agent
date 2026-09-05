"""Research runner for the strict live-trading-mirroring replay path.

The decision path is still `AITrader.evaluate_decision_core()`; this runner
only supplies historical time/data and replaces the broker execution side
with the canonical adapter. It intentionally does not optimize or alter
strategy parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from backtest.canonical_execution import CanonicalHistoricalExecutionAdapter, FillPolicy
from backtest.canonical_position_monitor import Bar, HistoricalPositionMonitor
from backtest.live_mirroring_execution import LiveMirroringExecutionBridge
from core.clock import ReplayClock


@dataclass
class ReplayTradeRecord:
    lifecycle: dict
    analysis_out: dict
    decision_out: dict
    risk_out: dict
    permission_out: dict


@dataclass
class LiveMirroringReplayResult:
    symbol: str
    timeframe: str
    bars_seen: int
    trades: list[ReplayTradeRecord] = field(default_factory=list)
    open_trade_ids: list[int] = field(default_factory=list)


def run_live_mirroring_replay(
    *, symbol: str, timeframe: str, df: pd.DataFrame,
    starting_balance: float = 10000.0, warmup_bars: int = 300,
    pip_size: float = 0.0001, pnl_multiplier: float = 100000.0,
    spread_pips: float = 0.0, slippage_pips: float = 0.0,
    commission_per_lot: float = 0.0,
    intrabar_policy: str = "AMBIGUOUS_INTRABAR",
    db_path: str = "backtest/live_mirroring.db",
) -> LiveMirroringReplayResult:
    """Run a strict replay over an already-loaded historical OHLC dataframe.

    `df.iloc[:i+1]` is the only market window passed to the shared decision
    core at bar i. A position is monitored only with bars at/after its entry.
    No next-bar-open fallback, random fill, current external data, or wall
    clock is used by this runner.
    """
    if df is None or len(df) <= warmup_bars:
        raise ValueError("insufficient historical bars for replay")
    required = {"open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"historical dataframe missing columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("historical dataframe index must be a DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("historical dataframe index must be timezone-aware")

    from core.trader import AITrader
    from database.db import TraderDB
    from execution.paper_trader import PaperTrader
    from core.constants import set_backtest_mode, reset_backtest_memory

    set_backtest_mode(True)
    reset_backtest_memory()
    clock = ReplayClock()
    db = TraderDB(db_path=db_path)
    paper = PaperTrader(starting_balance=starting_balance, db=db)
    trader = AITrader(
        balance=starting_balance, symbol=symbol, timeframe=timeframe,
        paper_balance=starting_balance, execution_mode="backtest",
        paper_trader=paper, db=db, clock=clock,
    )

    adapter = CanonicalHistoricalExecutionAdapter(
        pip_size=pip_size,
        fill_policy=FillPolicy(
            spread_pips=spread_pips, slippage_pips=slippage_pips,
            commission_per_lot=commission_per_lot,
            intrabar_policy=intrabar_policy,
        ),
    )
    monitor = HistoricalPositionMonitor(
        pip_size=pip_size, intrabar_policy=intrabar_policy,
        commission_per_lot=commission_per_lot,
    )
    bridge = LiveMirroringExecutionBridge(adapter, monitor)
    result = LiveMirroringReplayResult(symbol=symbol, timeframe=timeframe,
                                       bars_seen=0)
    opened: dict[int, int] = {}

    for i in range(warmup_bars, len(df)):
        # Close detection consumes the current bar before a new decision.
        bar_time = df.index[i]
        bar = Bar(timestamp=bar_time.isoformat(), high=float(df.iloc[i]["high"]),
                  low=float(df.iloc[i]["low"]), close=float(df.iloc[i]["close"]))
        for trade_id in list(adapter.open_positions):
            closed = bridge.advance_position(trade_id, bar)
            if closed is not None:
                result.trades.append(ReplayTradeRecord(
                    lifecycle=closed.to_dict(), analysis_out={}, decision_out={},
                    risk_out={}, permission_out={},
                ))
                opened.pop(trade_id, None)

        clock.advance(bar_time)
        window = df.iloc[: i + 1].copy()
        # The provider/analysis layer is responsible for converting this
        # historical window to the exact market_out shape used by live code.
        from core.data_provider import HistoricalMT5Provider
        provider = HistoricalMT5Provider(window, symbol, timeframe, clock=clock)
        market_out = provider.get_market_out(symbol, timeframe)
        session_ctx = {
            "current_session": clock.current_session(),
            "gmt_time": clock.now().isoformat(),
            "session_strategy": "n/a",
        }
        core = trader.evaluate_decision_core(market_out, session_ctx)
        result.bars_seen += 1

        analysis_out = core["analysis_out"]
        decision_out = core["dec_out"]
        risk_out = core["risk_out"]
        permission_out = core["perm_out"]
        if analysis_out.get("error"):
            continue
        if decision_out.get("decision") not in {"BUY", "SELL"}:
            continue
        if not risk_out.get("approved") or not permission_out.get("allowed"):
            continue

        # Entry must come from the live decision/risk output. There is no
        # fabricated next-bar-open fallback in this strict runner.
        required = ("entry", "sl", "tp", "lot")
        if any(decision_out.get(k) is None for k in ("entry",)) or \
           any(risk_out.get(k) is None for k in ("sl_price", "tp_price", "lot")):
            continue
        decision_payload = dict(decision_out)
        decision_payload.update({
            "symbol": symbol,
            "sl": risk_out["sl_price"],
            "tp": risk_out["tp_price"],
            "lot": risk_out["lot"],
        })
        # Current closed bar's observed bid is the execution market input.
        # This is deliberately distinct from the live requested entry price.
        historical_bid = float(df.iloc[i]["close"])
        trade = bridge.execute_decision(
            decision_result=decision_payload,
            signal_time=clock.now().isoformat(),
            entry_time=clock.now().isoformat(),
            historical_bid=historical_bid,
            pnl_multiplier=pnl_multiplier,
        )
        opened[trade.trade_id] = i
        result.trades.append(ReplayTradeRecord(
            lifecycle=trade.to_dict(),
            analysis_out=analysis_out,
            decision_out=decision_out,
            risk_out=risk_out,
            permission_out=permission_out,
        ))

    result.open_trade_ids = sorted(adapter.open_positions)
    return result
