from types import SimpleNamespace

from backtest.historical_execution_router import HistoricalExecutionRouter
from backtest.position_state import CanonicalPositionState


class FakeBroker:
    def __init__(self):
        self.next_id = 1
        self.opens = []

    def open_trade(self, **kwargs):
        trade = SimpleNamespace(
            trade_id=self.next_id,
            symbol=kwargs["symbol"],
            direction=kwargs["direction"],
            lot_size=kwargs["lot"],
            entry_price=kwargs["entry_price"],
            entry_time=kwargs.get("bar_time"),
        )
        self.next_id += 1
        self.opens.append(kwargs)
        return trade


def test_market_is_not_filled_on_submission_bar():
    broker = FakeBroker()
    state = CanonicalPositionState()
    router = HistoricalExecutionRouter(broker, position_state=state)

    result = router.submit(symbol="EURUSD", direction="BUY", entry_price=1.10,
                           sl=1.09, tp=1.12, lot=0.01, confidence=80,
                           bar_index=10, bar_time="t10")
    assert result["status"] == "QUEUED"
    assert router.advance(bar_index=10, bar_open=1.101, bar_high=1.102,
                          bar_low=1.099, bar_close=1.1005, bar_time="t10") == []
    assert state.count() == 0

    fills = router.advance(bar_index=11, bar_open=1.103, bar_high=1.104,
                           bar_low=1.101, bar_close=1.102, bar_time="t11")
    assert len(fills) == 1
    assert fills[0].entry_price == 1.103
    assert state.count() == 1


def test_pending_limit_waits_then_fills_on_touch():
    broker = FakeBroker()
    router = HistoricalExecutionRouter(broker, default_limit_expiry_bars=3)
    result = router.submit(symbol="EURUSD", direction="BUY", entry_price=1.1000,
                           sl=1.0900, tp=1.1200, lot=0.01, confidence=80,
                           bar_index=10, bar_time="t10", order_type="BUY_LIMIT")
    assert result["status"] == "PENDING"
    assert len(router.pending_orders) == 1

    assert router.advance(bar_index=11, bar_open=1.105, bar_high=1.106,
                          bar_low=1.101, bar_close=1.104, bar_time="t11") == []
    fills = router.advance(bar_index=12, bar_open=1.103, bar_high=1.104,
                           bar_low=1.099, bar_close=1.101, bar_time="t12")
    assert len(fills) == 1
    assert fills[0].entry_price == 1.1000
    assert not router.pending_orders


def test_pending_limit_expires_without_fill():
    broker = FakeBroker()
    router = HistoricalExecutionRouter(broker, default_limit_expiry_bars=1)
    router.submit(symbol="EURUSD", direction="SELL", entry_price=1.1000,
                  sl=1.1100, tp=1.0800, lot=0.01, confidence=80,
                  bar_index=10, bar_time="t10", order_type="SELL_LIMIT")
    router.advance(bar_index=11, bar_open=1.105, bar_high=1.106,
                   bar_low=1.101, bar_close=1.104, bar_time="t11")
    assert len(router.pending_orders) == 1
    router.advance(bar_index=12, bar_open=1.105, bar_high=1.106,
                   bar_low=1.101, bar_close=1.104, bar_time="t12")
    assert not router.pending_orders
    assert any(e["event"] == "LIMIT_EXPIRED" for e in router.telemetry)


def test_canonical_state_blocks_duplicate_direction():
    broker = FakeBroker()
    state = CanonicalPositionState()
    router = HistoricalExecutionRouter(broker, position_state=state)
    router.submit(symbol="EURUSD", direction="BUY", entry_price=1.1,
                  sl=1.09, tp=1.12, lot=0.01, confidence=80, bar_index=1)
    router.advance(bar_index=2, bar_open=1.1, bar_high=1.11,
                   bar_low=1.09, bar_close=1.105)
    rejected = router.submit(symbol="EURUSD", direction="BUY", entry_price=1.106,
                             sl=1.09, tp=1.12, lot=0.01, confidence=80, bar_index=2)
    assert rejected == {"status": "REJECTED", "reason": "duplicate_symbol_direction"}
