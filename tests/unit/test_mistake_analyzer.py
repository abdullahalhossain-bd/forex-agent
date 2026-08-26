from database.db import TraderDB
from learning.mistake_analyzer import AdvancedMistakeAnalyzer


def _open_trade(db, pair="EURUSD"):
    return db.save_trade_open({
        "pair": pair,
        "timeframe": "15m",
        "type": "BUY",
        "entry": 1.1000,
        "sl": 1.0900,
        "tp": 1.1200,
        "lot": 0.01,
        "confidence": 72,
        "open_time": "2026-08-26T10:00:00+00:00",
        "pattern": "breakout",
        "regime": "trend",
        "trend": "bullish",
        "rsi": 58,
        "session": "london",
        "context": {"rr_ratio": 2.0},
    })


def test_loss_trade_is_analyzed_and_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr("core.constants.MISTAKES_JSON_PATH", tmp_path / "mistakes.json")
    db = TraderDB(db_path=str(tmp_path / "trader.db"))
    trade_id = _open_trade(db)
    db.save_trade_close(trade_id, {
        "close_time": "2026-08-26T11:00:00+00:00",
        "exit_price": 1.0900,
        "result": "LOSS",
        "pnl": -10.0,
        "pnl_pips": -100.0,
    })

    analyzer = AdvancedMistakeAnalyzer(llm_client=False)
    analyzer.memory.db = db
    analyzer.analyze_closed_trade(trade_id)

    mistakes = db.get_mistakes()
    assert len(mistakes) == 1
    assert int(mistakes.iloc[0]["trade_id"]) == trade_id


def test_win_trade_does_not_create_mistake(tmp_path):
    db = TraderDB(db_path=str(tmp_path / "trader.db"))
    trade_id = _open_trade(db)
    db.save_trade_close(trade_id, {
        "close_time": "2026-08-26T11:00:00+00:00",
        "exit_price": 1.1200,
        "result": "WIN",
        "pnl": 20.0,
        "pnl_pips": 200.0,
    })

    analyzer = AdvancedMistakeAnalyzer(llm_client=False)
    analyzer.memory.db = db
    analyzer.analyze_closed_trade(trade_id)

    assert db.get_mistakes().empty


def test_mt5_ticket_resolves_trade_id(tmp_path):
    db = TraderDB(db_path=str(tmp_path / "trader.db"))
    trade_id = db.save_trade_open({
        "pair": "EURUSD",
        "timeframe": "15m",
        "type": "BUY",
        "entry": 1.1,
        "sl": 1.09,
        "tp": 1.12,
        "lot": 0.01,
        "open_time": "2026-08-26T10:00:00+00:00",
        "context": {"mt5_ticket": 12345},
    })

    assert db.get_trade_id_by_mt5_ticket(12345) == trade_id