"""Contract tests for the canonical live-mirroring execution lifecycle."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backtest.canonical_execution import CanonicalHistoricalExecutionAdapter, FillPolicy
from backtest.canonical_position_monitor import Bar, HistoricalPositionMonitor
from backtest.live_mirroring_execution import LiveMirroringExecutionBridge


UTC = timezone.utc


def _ts(minute: int) -> str:
    return datetime(2026, 1, 2, 12, minute, tzinfo=UTC).isoformat()


def _bridge(policy="AMBIGUOUS_INTRABAR"):
    adapter = CanonicalHistoricalExecutionAdapter(
        pip_size=0.0001,
        fill_policy=FillPolicy(spread_pips=2.0, slippage_pips=0.5,
                               commission_per_lot=3.0,
                               intrabar_policy=policy),
    )
    monitor = HistoricalPositionMonitor(
        pip_size=0.0001, intrabar_policy=policy, commission_per_lot=3.0
    )
    return adapter, LiveMirroringExecutionBridge(adapter, monitor)


def _payload(direction="BUY"):
    return {
        "symbol": "EURUSD",
        "decision": direction,
        "entry": 1.1000,
        "sl": 1.0980 if direction == "BUY" else 1.1020,
        "tp": 1.1040 if direction == "BUY" else 1.0960,
        "lot": 0.10,
        "confidence": 82,
    }


def test_entry_uses_historical_bid_not_requested_entry():
    adapter, bridge = _bridge()
    trade = bridge.execute_decision(
        decision_result=_payload(),
        signal_time=_ts(0), entry_time=_ts(0),
        historical_bid=1.1010, pnl_multiplier=100000.0,
    )
    # BUY pays historical ask (2 pip spread) + 0.5 pip adverse slippage.
    assert trade.requested_entry == pytest.approx(1.1000)
    assert trade.fill_price == pytest.approx(1.10125)
    assert trade.requested_lot == pytest.approx(0.10)
    assert trade.filled_lot == pytest.approx(0.10)


def test_stop_loss_close_calculates_realized_pnl_and_commission():
    adapter, bridge = _bridge()
    trade = bridge.execute_decision(
        decision_result=_payload(), signal_time=_ts(0), entry_time=_ts(0),
        historical_bid=1.1000, pnl_multiplier=100000.0,
    )
    closed = bridge.advance_position(
        trade.trade_id,
        Bar(timestamp=_ts(1), high=1.1010, low=1.0970, close=1.0990),
    )
    assert closed is not None
    assert closed.exit_reason == "SL"
    assert closed.status == "CLOSED"
    assert closed.exit_price == pytest.approx(1.0980)
    assert closed.commission_usd == pytest.approx(0.30)
    assert closed.pnl_usd == pytest.approx((1.0980 - 1.10025) * 0.10 * 100000 - 0.30)
    assert trade.trade_id not in adapter.open_positions


def test_both_sl_and_tp_is_explicitly_ambiguous_by_default():
    adapter, bridge = _bridge()
    trade = bridge.execute_decision(
        decision_result=_payload(), signal_time=_ts(0), entry_time=_ts(0),
        historical_bid=1.1000, pnl_multiplier=100000.0,
    )
    with pytest.raises(ValueError, match="AMBIGUOUS_INTRABAR"):
        bridge.advance_position(
            trade.trade_id,
            Bar(timestamp=_ts(1), high=1.1050, low=1.0970, close=1.1000),
        )


def test_exit_before_entry_is_rejected():
    adapter, bridge = _bridge()
    trade = bridge.execute_decision(
        decision_result=_payload(), signal_time=_ts(1), entry_time=_ts(1),
        historical_bid=1.1000, pnl_multiplier=100000.0,
    )
    with pytest.raises(ValueError, match="EXIT_BEFORE_ENTRY"):
        bridge.force_market_close(
            trade.trade_id, timestamp=_ts(0), bid=1.0990,
            reason="INVALID_TEST_CLOSE",
        )


def test_sell_fill_uses_bid_and_adverse_slippage_only():
    adapter, bridge = _bridge()
    trade = bridge.execute_decision(
        decision_result=_payload("SELL"),
        signal_time=_ts(0), entry_time=_ts(0),
        historical_bid=1.1010, pnl_multiplier=100000.0,
    )
    assert trade.fill_price == pytest.approx(1.10095)


def test_market_close_is_historical_and_deterministic():
    adapter, bridge = _bridge()
    trade = bridge.execute_decision(
        decision_result=_payload(), signal_time=_ts(0), entry_time=_ts(0),
        historical_bid=1.1000, pnl_multiplier=100000.0,
    )
    closed = bridge.force_market_close(
        trade.trade_id, timestamp=_ts(2), bid=1.1020,
        spread_pips=2.0, slippage_pips=0.5, reason="END_OF_REPLAY",
    )
    # BUY closes at historical bid minus spread + adverse slippage.
    assert closed.exit_price == pytest.approx(1.10175)
    assert closed.exit_time == _ts(2)
    assert closed.exit_reason == "END_OF_REPLAY"
    assert closed.status == "CLOSED"
