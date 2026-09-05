from backtest.canonical_execution import CanonicalHistoricalExecutionAdapter, FillPolicy
from backtest.canonical_position_monitor import Bar, HistoricalPositionMonitor
from backtest.live_mirroring_execution import LiveMirroringExecutionBridge


BUY = {
    "symbol": "EURUSD",
    "decision": "BUY",
    "entry": 1.1000,
    "sl": 1.0990,
    "tp": 1.1020,
    "lot": 0.10,
    "confidence": 82,
}


def _bridge(policy="AMBIGUOUS_INTRABAR"):
    adapter = CanonicalHistoricalExecutionAdapter(
        pip_size=0.0001,
        fill_policy=FillPolicy(spread_pips=1.0, intrabar_policy=policy),
    )
    monitor = HistoricalPositionMonitor(pip_size=0.0001, intrabar_policy=policy)
    return LiveMirroringExecutionBridge(adapter, monitor)


def test_entry_is_timestamp_bound_and_payload_preserved():
    bridge = _bridge()
    trade = bridge.execute_decision(
        decision_result=BUY,
        signal_time="2026-01-02T10:00:00+00:00",
        entry_time="2026-01-02T10:15:00+00:00",
        historical_bid=1.1000,
        pnl_multiplier=100000.0,
    )
    assert trade.requested_entry == BUY["entry"]
    assert trade.requested_lot == BUY["lot"]
    assert trade.stop_loss == BUY["sl"]
    assert trade.take_profit == BUY["tp"]
    assert trade.filled_lot == BUY["lot"]
    assert trade.fill_price == 1.10005


def test_entry_before_signal_is_rejected():
    bridge = _bridge()
    try:
        bridge.execute_decision(
            decision_result=BUY,
            signal_time="2026-01-02T10:15:00+00:00",
            entry_time="2026-01-02T10:00:00+00:00",
            historical_bid=1.1000,
            pnl_multiplier=100000.0,
        )
    except ValueError as exc:
        assert "LOOK_AHEAD_EXECUTION" in str(exc)
    else:
        raise AssertionError("future-leaking entry was accepted")


def test_sl_close_and_realized_pnl():
    bridge = _bridge()
    trade = bridge.execute_decision(
        decision_result=BUY,
        signal_time="2026-01-02T10:00:00+00:00",
        entry_time="2026-01-02T10:15:00+00:00",
        historical_bid=1.1000,
        pnl_multiplier=100000.0,
    )
    closed = bridge.advance_position(trade.trade_id, Bar(
        timestamp="2026-01-02T10:30:00+00:00",
        high=1.1004, low=1.0988, close=1.0995,
    ))
    assert closed.status == "CLOSED"
    assert closed.exit_reason == "SL"
    assert closed.exit_price == BUY["sl"]
    assert closed.pnl_usd < 0


def test_both_sl_tp_touched_is_ambiguous():
    bridge = _bridge()
    trade = bridge.execute_decision(
        decision_result=BUY,
        signal_time="2026-01-02T10:00:00+00:00",
        entry_time="2026-01-02T10:15:00+00:00",
        historical_bid=1.1000,
        pnl_multiplier=100000.0,
    )
    try:
        bridge.advance_position(trade.trade_id, Bar(
            timestamp="2026-01-02T10:30:00+00:00",
            high=1.1030, low=1.0980, close=1.1005,
        ))
    except ValueError as exc:
        assert "AMBIGUOUS_INTRABAR" in str(exc)
    else:
        raise AssertionError("ambiguous OHLC bar was silently resolved")


def test_exit_cannot_precede_entry():
    bridge = _bridge()
    trade = bridge.execute_decision(
        decision_result=BUY,
        signal_time="2026-01-02T10:00:00+00:00",
        entry_time="2026-01-02T10:15:00+00:00",
        historical_bid=1.1000,
        pnl_multiplier=100000.0,
    )
    try:
        bridge.force_market_close(
            trade.trade_id,
            timestamp="2026-01-02T10:00:00+00:00",
            bid=1.1001,
        )
    except ValueError as exc:
        assert "EXIT_BEFORE_ENTRY" in str(exc)
    else:
        raise AssertionError("exit before entry was accepted")
