import pandas as pd

from agents.analysis_agent import AnalysisAgent
from core.constants import set_backtest_mode


def test_backtest_fast_mode_skips_heavy_analysis(monkeypatch):
    set_backtest_mode(True)

    class FailOnUse:
        def __init__(self, *args, **kwargs):
            raise AssertionError("heavy analysis should be skipped in backtest fast mode")

    monkeypatch.setattr("agents.analysis_agent.SupportResistance", FailOnUse)
    monkeypatch.setattr("agents.analysis_agent.AdvancedPatternDetector", FailOnUse)
    monkeypatch.setattr("agents.analysis_agent.SentimentDataProvider", FailOnUse)
    monkeypatch.setattr("agents.analysis_agent.SMCAdvancedEngine", FailOnUse)

    df = pd.DataFrame(
        {
            "open": [1.1000, 1.1010, 1.1020, 1.1030],
            "high": [1.1010, 1.1020, 1.1030, 1.1040],
            "low": [1.0990, 1.1000, 1.1010, 1.1020],
            "close": [1.1005, 1.1015, 1.1025, 1.1035],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="H"),
    )

    market_out = {
        "df": df,
        "ind_ctx": {"close": 1.1035, "rsi": 55, "trend": "up"},
        "regime": {"regime": "TRENDING"},
        "mtf_bias": {"bias": "BULLISH", "confidence": "HIGH"},
        "symbol": "EURUSD",
        "timeframe": "H1",
    }

    agent = AnalysisAgent(backtest_fast_mode=True)
    result = agent.run(market_out)

    assert result["signal"]["signal"] in {"BUY", "SELL", "WAIT"}
    assert result["final_signal"] in {"BUY", "SELL", "WAIT", "NO TRADE"}
