import logging

import numpy as np
import pandas as pd
import pytest

from analysis._engine_utils import atr_series, atr_value


def test_atr_series_computes_standard_atr():
    df = pd.DataFrame(
        {
            "high": [1.10, 1.12, 1.11, 1.13],
            "low": [1.08, 1.09, 1.07, 1.10],
            "close": [1.09, 1.11, 1.10, 1.12],
        }
    )

    series = atr_series(df, period=2)

    assert len(series) == len(df)
    assert np.all(np.isfinite(series.iloc[1:]))
    assert series.iloc[-1] > 0


def test_atr_series_handles_nan_rows_without_raising(caplog):
    df = pd.DataFrame(
        {
            "high": [1.10, np.nan, 1.11, 1.13],
            "low": [1.08, 1.09, np.nan, 1.10],
            "close": [1.09, 1.11, 1.10, np.nan],
        }
    )

    caplog.set_level(logging.DEBUG)
    series = atr_series(df, period=2)

    assert len(series) == len(df)
    assert np.isnan(series.iloc[1]) or np.isnan(series.iloc[2]) or np.isnan(series.iloc[3])
    assert "ATR series contains" in caplog.text


def test_atr_series_missing_columns_returns_nan_series(caplog):
    df = pd.DataFrame({"open": [1.0, 1.1], "close": [1.05, 1.08]})

    caplog.set_level(logging.WARNING)
    series = atr_series(df, period=2)

    assert len(series) == len(df)
    assert series.isna().all()
    assert "missing required columns" in caplog.text


def test_atr_value_falls_back_to_close_and_logs(caplog):
    df = pd.DataFrame(
        {
            "high": [1.10, np.nan],
            "low": [1.08, 1.09],
            "close": [1.09, 1.11],
        }
    )

    caplog.set_level(logging.WARNING)
    value = atr_value(df, period=2)

    assert value == pytest.approx(1.11 * 0.001)
    assert "ATR invalid/non-finite" in caplog.text
