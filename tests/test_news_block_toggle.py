import os

from fundamental.news_filter import NewsFilter


def test_news_block_can_be_disabled(monkeypatch):
    monkeypatch.setenv("NEWS_BLOCK_ENABLED", "true")

    news_filter = NewsFilter()
    result = news_filter.check("EURUSD")

    assert result["trade_allowed"] is True
    assert result["source"] == "disabled"
    assert "disabled" in result["reason"].lower()
