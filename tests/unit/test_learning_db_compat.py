from memory.database import Database


def test_memory_database_shim_supports_learning_db_compat_methods(tmp_path):
    db_path = tmp_path / "compat.db"
    db = Database(db_path=str(db_path))

    trade_id = db.save_trade({
        "pair": "EURUSD",
        "signal": "BUY",
        "entry": 1.1000,
        "sl": 1.0900,
        "tp": 1.1200,
        "lot": 0.01,
        "result": "OPEN",
        "pnl": 0.0,
        "rr_ratio": 2.0,
        "confidence": 80,
        "chart_snapshot": {"source": "mt5_demo"},
    })

    assert trade_id is not None

    db.update_trade_result(trade_id, "WIN", 12.5)

    history = db.get_trade_history(limit=10)
    assert not history.empty
    assert history.iloc[0]["result"] == "WIN"
    assert history.iloc[0]["pnl"] == 12.5
