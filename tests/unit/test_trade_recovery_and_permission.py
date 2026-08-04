import pytest

from execution.trade_recovery import _reconcile_closed_trades
from risk.trade_permission import TradePermission


class DummyLearningDB:
    def __init__(self):
        self.open_trades = []
        self.updated = []

    def get_open_trades(self):
        return self.open_trades

    def update_trade_result(self, trade_id, result, pnl):
        self.updated.append((trade_id, result, pnl))


class DummyOrderManager:
    def get_order_history(self, days_back=7):
        return []


def test_reconcile_closed_trades_handles_stub_db_without_crashing():
    class StubDB:
        pass

    db = StubDB()
    closed = _reconcile_closed_trades(
        learning_db=db,
        order_manager=DummyOrderManager(),
        mt5_open_tickets=set(),
        magic_number=424242,
        history_days=7,
    )

    assert closed == 0


def test_reconcile_closed_trades_updates_closed_trade_when_db_present():
    db = DummyLearningDB()
    db.open_trades = [{"id": 7, "pair": "EURUSD", "mt5_ticket": 101}]

    closed = _reconcile_closed_trades(
        learning_db=db,
        order_manager=DummyOrderManager(),
        mt5_open_tickets=set(),
        magic_number=424242,
        history_days=7,
    )

    assert closed == 1
    assert db.updated == [(7, "LOSS", -12.5)]


def test_trade_permission_allows_when_news_is_unavailable_but_bypass_enabled(monkeypatch):
    monkeypatch.setenv("BYPASS_NEWS_GATE", "true")
    tp = TradePermission()
    result = tp.check(
        decision_out={"decision": "BUY", "confidence": 80, "direct_lane": False},
        risk_out={"approved": True, "entry": 1.1, "sl_price": 1.09, "tp_price": 1.12, "rr_ratio": 2.0, "lot": 0.01},
        news_ctx={},
        session_ctx={"fusion": {"fusion_allowed": True, "fusion_score": 80, "fusion_grade": "A"}},
        execution_filters={},
    )
    assert result["allowed"] is True
