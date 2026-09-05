"""Strict research runner for live-trading-mirroring replay.

Historical replay reuses the live AITrader decision core and replaces only
broker execution with the canonical deterministic execution lifecycle.
No strategy optimization is performed here.
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
    rejection_stats: dict = field(default_factory=dict)


def _row_float(row, *names) -> Optional[float]:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return float(row[name])
    return None


def run_live_mirroring_replay(
    *, symbol: str, timeframe: str, df: pd.DataFrame,
    starting_balance: float = 10000.0, warmup_bars: int = 300,
    pip_size: float = 0.0001, pnl_multiplier: float = 100000.0,
    spread_pips: float = 0.0, slippage_pips: float = 0.0,
    commission_per_lot: float = 0.0,
    intrabar_policy: str = "AMBIGUOUS_INTRABAR",
    db_path: str = "backtest/live_mirroring.db",
    max_open_trades: int | None = None,
    max_hold_bars: int | None = None,
    clock: ReplayClock | None = None,
    bypass_checks: set[str] | list[str] | None = None,
) -> LiveMirroringReplayResult:
    """Run strict replay over historical OHLC data.

    Invariant: at decision bar i, only ``df.iloc[:i+1]`` is passed to the
    market provider. Open positions are checked only against the current
    historical bar. Entry uses historical bid/ask data, never requested
    future prices. The live decision/risk/permission outputs are not rebuilt
    or substituted by this runner.
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
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("historical dataframe index must be strictly increasing without duplicates")
    if max_open_trades is None:
        try:
            from config import MAX_OPEN_TRADES
            max_open_trades = int(MAX_OPEN_TRADES)
        except Exception:
            max_open_trades = 3
    if max_open_trades < 1:
        raise ValueError("max_open_trades must be >= 1")

    from core.trader import AITrader
    from database.db import TraderDB
    from execution.paper_trader import PaperTrader
    from core.constants import set_backtest_mode, reset_backtest_memory

    set_backtest_mode(True)
    reset_backtest_memory()
    replay_clock = clock or ReplayClock()
    db = TraderDB(db_path=db_path)
    paper = PaperTrader(starting_balance=starting_balance, db=db)
    trader = AITrader(
        balance=starting_balance, symbol=symbol, timeframe=timeframe,
        paper_balance=starting_balance, execution_mode="backtest",
        paper_trader=paper, db=db, clock=replay_clock,
    )

    adapter = CanonicalHistoricalExecutionAdapter(
        pip_size=pip_size,
        fill_policy=FillPolicy(
            spread_pips=float(spread_pips), slippage_pips=float(slippage_pips),
            commission_per_lot=float(commission_per_lot),
            intrabar_policy=intrabar_policy,
        ),
    )
    monitor = HistoricalPositionMonitor(
        pip_size=pip_size, intrabar_policy=intrabar_policy,
        commission_per_lot=commission_per_lot,
    )
    bridge = LiveMirroringExecutionBridge(adapter, monitor)
    result = LiveMirroringReplayResult(symbol=symbol, timeframe=timeframe, bars_seen=0)
    opened: dict[int, int] = {}
    record_by_trade: dict[int, ReplayTradeRecord] = {}
    stats = {
        "total_bars": 0, "WAIT": 0, "analysis_error": 0,
        "risk_rejected": 0, "permission_blocked": 0,
        "missing_execution_fields": 0, "max_open_trades": 0,
        "execution_rejected": 0, "intrabar_ambiguous": 0,
        "closed": 0, "forced_timeout": 0,
    }

    for i in range(warmup_bars, len(df)):
        bar_time = df.index[i]
        stats["total_bars"] += 1

        # Position lifecycle is advanced before evaluating a new signal.
        bar = Bar(
            timestamp=bar_time.isoformat(),
            high=float(df.iloc[i]["high"]),
            low=float(df.iloc[i]["low"]),
            close=float(df.iloc[i]["close"]),
            bid=_row_float(df.iloc[i], "bid", "Bid", "close"),
            spread_pips=_row_float(df.iloc[i], "spread_pips", "spread", "Spread"),
        )
        for trade_id in list(adapter.open_positions):
            opened_at = opened.get(trade_id, i)
            try:
                closed = bridge.advance_position(trade_id, bar)
            except ValueError as exc:
                if "AMBIGUOUS_INTRABAR" in str(exc):
                    stats["intrabar_ambiguous"] += 1
                raise
            if closed is not None:
                stats["closed"] += 1
                rec = record_by_trade.get(trade_id)
                if rec is not None:
                    rec.lifecycle = closed.to_dict()
                opened.pop(trade_id, None)

            elif max_hold_bars is not None and (i - opened_at) >= max_hold_bars:
                bid = bar.bid if bar.bid is not None else bar.close
                spread = bar.spread_pips if bar.spread_pips is not None else spread_pips
                closed = bridge.force_market_close(
                    trade_id, timestamp=bar_time.isoformat(), bid=bid,
                    spread_pips=float(spread), slippage_pips=slippage_pips,
                    reason="MAX_HOLD_TIMEOUT",
                )
                stats["closed"] += 1
                stats["forced_timeout"] += 1
                rec = record_by_trade.get(trade_id)
                if rec is not None:
                    rec.lifecycle = closed.to_dict()
                opened.pop(trade_id, None)

        replay_clock.advance(bar_time)

        if len(adapter.open_positions) >= max_open_trades:
            stats["max_open_trades"] += 1
            result.bars_seen += 1
            continue

        window = df.iloc[: i + 1].copy()
        from core.data_provider import HistoricalMT5Provider
        provider = HistoricalMT5Provider(window, symbol, timeframe, clock=replay_clock)
        market_out = provider.get_market_out(symbol, timeframe)
        session_ctx = {
            "current_session": replay_clock.current_session(),
            "gmt_time": replay_clock.now().isoformat(),
            "session_strategy": "n/a",
        }
        core = trader.evaluate_decision_core(
            market_out, session_ctx, bypass_checks=bypass_checks
        )
        result.bars_seen += 1

        analysis_out = core["analysis_out"]
        decision_out = core["dec_out"]
        risk_out = core["risk_out"]
        permission_out = core["perm_out"]
        if analysis_out.get("error"):
            stats["analysis_error"] += 1
            continue
        if decision_out.get("decision") not in {"BUY", "SELL"}:
            stats["WAIT"] += 1
            continue
        if not risk_out.get("approved"):
            stats["risk_rejected"] += 1
            continue
        if not permission_out.get("allowed"):
            stats["permission_blocked"] += 1
            continue
        if decision_out.get("entry") is None or any(
            risk_out.get(k) is None for k in ("sl_price", "tp_price", "lot")
        ):
            stats["missing_execution_fields"] += 1
            continue

        decision_payload = dict(decision_out)
        decision_payload.update({
            "symbol": symbol,
            "sl": risk_out["sl_price"],
            "tp": risk_out["tp_price"],
            "lot": risk_out["lot"],
        })
        historical_bid = bar.bid if bar.bid is not None else bar.close
        historical_ask = _row_float(df.iloc[i], "ask", "Ask")
        if historical_ask is None and bar.spread_pips is not None:
            historical_ask = historical_bid + bar.spread_pips * pip_size

        try:
            trade = bridge.execute_decision(
                decision_result=decision_payload,
                signal_time=replay_clock.now().isoformat(),
                entry_time=replay_clock.now().isoformat(),
                historical_bid=historical_bid,
                pnl_multiplier=pnl_multiplier,
            )
            # Use explicit ask when available by reopening through the canonical
            # adapter directly; this is the same execution boundary, not a new
            # strategy path. Avoid this branch when no ask is present.
            if historical_ask is not None:
                adapter.open_positions.pop(trade.trade_id, None)
                adapter._next_trade_id -= 1
                trade = adapter.open_trade(
                    decision_result=decision_payload,
                    signal_time=replay_clock.now().isoformat(),
                    entry_time=replay_clock.now().isoformat(),
                    historical_bid=historical_bid,
                    historical_ask=historical_ask,
                    pnl_multiplier=pnl_multiplier,
                )
        except ValueError:
            stats["execution_rejected"] += 1
            raise

        opened[trade.trade_id] = i
        record = ReplayTradeRecord(
            lifecycle=trade.to_dict(), analysis_out=analysis_out,
            decision_out=decision_out, risk_out=risk_out,
            permission_out=permission_out,
        )
        record_by_trade[trade.trade_id] = record
        result.trades.append(record)

    # End-of-dataset close is explicit and uses the final historical bid.
    if adapter.open_positions:
        final_time = df.index[-1]
        final_bid = _row_float(df.iloc[-1], "bid", "Bid", "close")
        final_ask = _row_float(df.iloc[-1], "ask", "Ask")
        if final_bid is None:
            final_bid = float(df.iloc[-1]["close"])
        final_spread = _row_float(df.iloc[-1], "spread_pips", "spread", "Spread")
        if final_spread is None:
            final_spread = spread_pips
        for trade_id in list(adapter.open_positions):
            closed = bridge.force_market_close(
                trade_id, timestamp=final_time.isoformat(), bid=final_bid,
                spread_pips=float(final_spread), slippage_pips=slippage_pips,
                reason="END_OF_REPLAY",
            )
            stats["closed"] += 1
            rec = record_by_trade.get(trade_id)
            if rec is not None:
                rec.lifecycle = closed.to_dict()

    result.open_trade_ids = sorted(adapter.open_positions)
    result.rejection_stats = stats
    return result
