import pytest
from datetime import datetime

from backtest.broker_sim import BrokerSimulator
from backtest.symbol_specs import pip_value_usd_per_lot


def test_broker_sim_applies_half_spread_to_buy_and_sell_entries():
    sim = BrokerSimulator(
        spread_pips={"EURUSD": 1.6},
        commission_per_lot=0.0,
        slippage_pips=0.0,
        slippage_stdev=0.0,
        partial_fill_prob=0.0,
        enforce_spread_limit=False,
    )

    buy = sim.open_trade(
        symbol="EURUSD",
        direction="BUY",
        entry_price=1.10000,
        sl=1.09000,
        tp=1.11000,
        lot=1.0,
        bar_time=datetime(2025, 1, 1, 0, 0, 0),
        spread_pips=1.6,
        confidence=0,
    )

    assert buy is not None
    assert buy.entry_price == 1.10008
    assert buy.spread_pips == 1.6
    assert buy.spread_cost_usd == round(1.6 * pip_value_usd_per_lot("EURUSD") * 1.0, 2)

    sell = sim.open_trade(
        symbol="EURUSD",
        direction="SELL",
        entry_price=1.10000,
        sl=1.11000,
        tp=1.09000,
        lot=1.0,
        bar_time=datetime(2025, 1, 1, 0, 0, 0),
        spread_pips=1.6,
        confidence=0,
    )

    assert sell is not None
    assert sell.entry_price == 1.09992


def test_broker_sim_close_trade_applies_half_spread_on_exit():
    sim = BrokerSimulator(
        spread_pips={"EURUSD": 1.6},
        commission_per_lot=0.0,
        slippage_pips=0.0,
        slippage_stdev=0.0,
        partial_fill_prob=0.0,
        enforce_spread_limit=False,
    )

    trade = sim.open_trade(
        symbol="EURUSD",
        direction="BUY",
        entry_price=1.10000,
        sl=1.09000,
        tp=1.11000,
        lot=1.0,
        bar_time=datetime(2025, 1, 1, 0, 0, 0),
        spread_pips=1.6,
        confidence=0,
    )

    assert trade is not None

    closed = sim.close_trade(
        trade,
        close_price=1.11000,
        bar_time=datetime(2025, 1, 1, 1, 0, 0),
        reason="test",
    )

    assert closed.exit_price == 1.10992
    assert closed.pnl_pips == round((1.10992 - trade.entry_price) / 0.0001, 1)
    assert closed.pnl_usd == round(closed.pnl_pips * pip_value_usd_per_lot("EURUSD") * trade.lot_size, 2)
