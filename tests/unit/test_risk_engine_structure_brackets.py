from unittest.mock import patch

import pandas as pd

from risk.risk_engine import RiskEngine
from risk.rr_policy import get_execution_min_rr, get_min_rr


def _candles():
    return pd.DataFrame({
        "open": [1.1000] * 30,
        "high": [1.1005] * 30,
        "low": [1.0995] * 30,
        "close": [1.1000] * 30,
    })


def _engine():
    engine = RiskEngine(balance=10000.0, symbol="EURUSD")
    engine.MAX_LOT = 10.0
    return engine


def test_tiered_rr_policy_keeps_two_r_preferred_and_one_point_five_as_floor():
    assert get_min_rr() == 2.0
    assert get_execution_min_rr() == 2.0
    assert get_execution_min_rr(strategy="stop_hunt") == 1.5


def test_structure_target_between_one_point_five_and_two_r_is_allowed():
    with patch("risk.risk_engine.get_live_pip_value_per_lot", return_value=10.0), \
         patch("risk.structure_stop.compute_structure_stop", return_value=1.0980), \
         patch("risk.entry_quality_guardrails._find_swing_highs", return_value=[1.1036]):
        result = _engine().evaluate(
            signal="BUY", entry=1.1000, atr=0.0010, df=_candles()
        )

    assert result["approved"] is True
    assert result["sl_price"] == 1.098
    assert result["tp_price"] == 1.1036
    assert result["sl_source"] == "fractal_swing_atr"
    assert result["tp_source"] == "structure"
    assert result["rr_ratio"] == 1.8


def test_structural_target_below_one_point_five_r_is_rejected():
    with patch("risk.risk_engine.get_live_pip_value_per_lot", return_value=10.0), \
         patch("risk.structure_stop.compute_structure_stop", return_value=1.0980), \
         patch("risk.entry_quality_guardrails._find_swing_highs", return_value=[1.1025]):
        result = _engine().evaluate(
            signal="BUY", entry=1.1000, atr=0.0010, df=_candles()
        )

    assert result["approved"] is False
    assert "below execution minimum 1.50" in result["reject_reason"]