# tests/unit/test_sentiment_logic.py
# ============================================================
# Unit tests for sentiment-engine scoring logic.
# Hermetic — no network, no DB. Uses pure-Python fakes.
# ============================================================

import pytest


# ── Smoke test: SentimentEngine can be imported and instantiated ──
@pytest.mark.smoke
def test_sentiment_engine_importable():
    """The SentimentEngine class is importable from analysis.sentiment."""
    from analysis.sentiment import SentimentEngine  # noqa: F401


@pytest.mark.unit
def test_fear_greed_extreme_greed_reduces_buy_score():
    """At index_value ≥ FG_EXTREME_GREED, score should be negative
    (penalize BUY entries) when source is fx_native."""
    from analysis.sentiment import SentimentEngine
    se = SentimentEngine()

    result = se.fear_greed(index_value=85, source="fx_native")

    assert result["label"] == "EXTREME_GREED"
    assert result["score"] < 0, "Extreme greed should penalize BUY (negative score)"
    assert result["data_quality"] == "verified"


@pytest.mark.unit
def test_fear_greed_extreme_fear_reduces_sell_score():
    """At index_value ≤ FG_EXTREME_FEAR, score should be positive
    (penalize SELL entries) when source is fx_native."""
    from analysis.sentiment import SentimentEngine
    se = SentimentEngine()

    result = se.fear_greed(index_value=12, source="fx_native")

    assert result["label"] == "EXTREME_FEAR"
    assert result["score"] > 0, "Extreme fear should penalize SELL (positive score)"
    assert result["data_quality"] == "verified"


@pytest.mark.unit
def test_fear_greed_crypto_proxy_is_halved():
    """When source is 'alternative.me' (crypto F&G proxy),
    the absolute score must be halved vs the fx_native score."""
    from analysis.sentiment import SentimentEngine
    se = SentimentEngine()

    native = se.fear_greed(index_value=85, source="fx_native")
    proxy  = se.fear_greed(index_value=85, source="alternative.me")

    assert proxy["data_quality"] == "crypto_proxy"
    # |proxy score| should be approximately half of |native score|
    assert abs(proxy["score"]) == pytest.approx(abs(native["score"]) / 2, abs=1)


@pytest.mark.unit
def test_fear_greed_fallback_is_halved_like_crypto_proxy():
    """When source is 'fallback', score is also halved (same penalty as
    crypto proxy) — both are treated as unvalidated, low-confidence sources."""
    from analysis.sentiment import SentimentEngine
    se = SentimentEngine()

    native  = se.fear_greed(index_value=85, source="fx_native")
    fb      = se.fear_greed(index_value=85, source="fallback")

    assert fb["data_quality"] == "no_data_fallback"
    assert abs(fb["score"]) == pytest.approx(abs(native["score"]) / 2, abs=1)


@pytest.mark.unit
def test_fear_greed_neutral_band():
    """Index value 46–54 should fall in the NEUTRAL band with score 0."""
    from analysis.sentiment import SentimentEngine
    se = SentimentEngine()

    result = se.fear_greed(index_value=50, source="fx_native")

    assert result["label"] == "NEUTRAL"
    assert result["score"] == 0


@pytest.mark.unit
def test_sentiment_context_exposes_data_quality_when_fallback_used():
    """Sentiment AI context should explicitly carry the data-quality label."""
    from analysis.sentiment import SentimentEngine

    se = SentimentEngine()
    ctx = se.get_ai_context({
        "sentiment_score": 0,
        "bias": "NEUTRAL",
        "confidence": 40,
        "data_quality": "degraded",
        "retail": {"retail_long_pct": 50},
        "fear_greed": {"label": "NEUTRAL"},
        "currency_strength": {"pair_bias": "NEUTRAL"},
        "dxy": {"dxy_trend": "NEUTRAL"},
        "reasons": [],
    })

    assert "sentiment_data_quality" in ctx
    assert ctx["sentiment_data_quality"] == "degraded"


@pytest.mark.unit
def test_confidence_breakdown_exposes_neutral_fallback_reason():
    """A zero-point sentiment line should explicitly explain fallback/neutral status."""
    from core.confidence_breakdown import build_confidence_breakdown

    breakdown = build_confidence_breakdown(
        direction="BUY",
        rule_confidence=23,
        llm_signal="BUY",
        llm_confidence=18,
        sentiment_boost=0,
        sentiment_quality="no_data_fallback",
        ind_ctx={"trend": "NEUTRAL", "rsi_signal": "NEUTRAL", "macd_cross": "NEUTRAL"},
        sr_ctx={},
        liquidity_ctx={},
    )

    sentiment_component = next(c for c in breakdown.components if c.label == "Sentiment")
    detail = sentiment_component.detail.lower()
    assert "neutral" in detail or "fallback" in detail
    assert "no_data_fallback" in detail
