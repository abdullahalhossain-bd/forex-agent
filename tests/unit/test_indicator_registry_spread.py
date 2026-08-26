import pandas as pd

from data.indicator_registry import get_ai_context


def test_mt5_usdjpy_candle_spread_points_convert_to_pips():
    frame = pd.DataFrame({"close": [159.292], "spread": [10]})
    frame.attrs["mt5_digits"] = 3

    context = get_ai_context(frame, spread_in_points=True)

    assert context["spread_pips"] == 1.0
    assert context["spread_avg_20"] == 1.0


def test_non_mt5_spread_remains_unchanged():
    frame = pd.DataFrame({"close": [1.1], "spread": [1.5]})

    context = get_ai_context(frame)

    assert context["spread_pips"] == 1.5
