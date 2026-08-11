# tests/unit/test_decision_score_neutral_weight.py
# ============================================================
# Regression test for audit fix (2026-08-11):
#
#   BUG: DecisionScorer.score() added a factor's full weight to the
#   normalization denominator (total_weight_used) even when that factor's
#   direction was NEUTRAL (i.e. it structurally could NOT contribute to
#   buy_weighted/sell_weighted). This silently deflated net_score/confidence
#   on every decision that included a NEUTRAL factor (e.g. "session", which
#   is *always* NEUTRAL by design, or "intermarket"/"news" whenever their
#   upstream data is missing).
#
# Evidence: confidence_counterfactual_summary.json / blocked_trade_analysis
#   showed 218 "Min confidence" blocks, 72.5% of which were "naturally low"
#   pre-penalty confidence -- consistent with confidence being mechanically
#   capped below its true value by wasted denominator weight.
#
# Fix: only add a factor's weight to total_weight_used when
#   f.direction in ("BUY", "SELL").
# ============================================================

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intelligence.decision_score import DecisionScorer, FactorScore, FACTOR_WEIGHTS


def _perfect_bullish_factors_with_neutral_session():
    """All directional factors maximally bullish; session (5% weight) is
    NEUTRAL, as it always is in production (see confluence_engine._session_factor).
    """
    return [
        FactorScore(name="smc", direction="BUY", strength=100, confidence=100, weight=0),
        FactorScore(name="liquidity", direction="BUY", strength=100, confidence=100, weight=0),
        FactorScore(name="currency_strength", direction="BUY", strength=100, confidence=100, weight=0),
        FactorScore(name="intermarket", direction="BUY", strength=100, confidence=100, weight=0),
        FactorScore(name="news", direction="BUY", strength=100, confidence=100, weight=0),
        FactorScore(name="technical", direction="BUY", strength=100, confidence=100, weight=0),
        # session is structurally NEUTRAL by design -- it should never be
        # able to cap the confidence of an otherwise-perfect setup.
        FactorScore(name="session", direction="NEUTRAL", strength=80, confidence=70, weight=0),
    ]


def test_neutral_session_factor_does_not_deflate_a_perfect_setup():
    scorer = DecisionScorer(weights=FACTOR_WEIGHTS.copy())
    result = scorer.score(_perfect_bullish_factors_with_neutral_session())

    assert result.final_direction == "BUY"
    # Every directional (non-session) factor is maxed out, so with the
    # NEUTRAL session factor correctly excluded from the denominator, the
    # buy_score should reach (approximately) the full 100 scale -- not be
    # capped at 95% (100 - session's 5% weight) as it was before the fix.
    assert result.buy_score >= 99.0, (
        f"buy_score={result.buy_score} — a NEUTRAL factor's weight is still "
        f"diluting the normalization denominator"
    )
    assert result.net_score >= 99.0
    assert result.confidence == 95.0  # confidence formula caps at 95


def test_missing_data_neutral_factor_does_not_deflate_confidence():
    """intermarket/news falling back to NEUTRAL (missing data) must not
    mechanically punish confidence versus the same setup without that
    factor present at all."""
    scorer = DecisionScorer(weights=FACTOR_WEIGHTS.copy())

    with_missing_intermarket = [
        FactorScore(name="smc", direction="BUY", strength=80, confidence=80, weight=0),
        FactorScore(name="liquidity", direction="BUY", strength=80, confidence=80, weight=0),
        # intermarket data unavailable -> NEUTRAL/0, as produced by
        # confluence_engine._intermarket_factor's "no intermarket data" path
        FactorScore(name="intermarket", direction="NEUTRAL", strength=0, confidence=0, weight=0),
    ]
    without_intermarket_factor_at_all = [
        FactorScore(name="smc", direction="BUY", strength=80, confidence=80, weight=0),
        FactorScore(name="liquidity", direction="BUY", strength=80, confidence=80, weight=0),
    ]

    r1 = scorer.score(with_missing_intermarket)
    r2 = scorer.score(without_intermarket_factor_at_all)

    # Presence of a missing-data NEUTRAL factor should not change the
    # buy_score computed from the exact same directional factors.
    assert abs(r1.buy_score - r2.buy_score) < 0.01


def test_all_neutral_factors_do_not_divide_by_zero():
    scorer = DecisionScorer(weights=FACTOR_WEIGHTS.copy())
    result = scorer.score([
        FactorScore(name="session", direction="NEUTRAL", strength=50, confidence=70, weight=0),
        FactorScore(name="news", direction="NEUTRAL", strength=0, confidence=0, weight=0),
    ])
    assert result.final_direction == "NEUTRAL"
    assert result.buy_score == 0.0
    assert result.sell_score == 0.0
