# data/validator.py
# ============================================================
# Day 7+ — Data Quality Check System
# ভুল data → ভুল analysis → ভুল trade
# এই module সেটা prevent করে
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.logger import get_logger

log = get_logger(__name__)


class DataValidator:
    """
    OHLCV data fetch করার পরে এই class দিয়ে validate করো।
    সমস্যা থাকলে warning দেবে, critical হলে False return করবে।
    """

    def validate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Run all data quality checks.
        Return True  → data OK, proceed
        Return False → critical issue, don't proceed
        """
        log.debug(f"Validating data: {symbol} {timeframe} | rows={len(df)}")
        passed = True

        passed &= self._check_empty(df)
        if not passed:
            return False  # Short-circuit: no point checking columns on empty df

        passed &= self._check_columns(df)
        if not passed:
            return False  # Short-circuit: missing required columns

        passed &= self._check_missing_values(df)
        passed &= self._check_duplicates(df)
        passed &= self._check_price_sanity(df)
        passed &= self._check_ohlc_logic(df)
        self._check_gaps(df, timeframe)

        if passed:
            log.debug("Data validation passed")
        else:
            log.error("Data validation FAILED — check warnings above")

        return passed

    # ─────────────────────────────────────────────
    # CHECKS
    # ─────────────────────────────────────────────

    def _check_empty(self, df):
        if df is None or len(df) == 0:
            log.error("DataFrame is empty")
            return False
        if len(df) < 50:
            log.warning(f"Very few candles: {len(df)} (need 200+ for reliable indicators)")
        return True

    def _check_columns(self, df):
        required = {'open', 'high', 'low', 'close'}
        missing  = required - set(df.columns)
        if missing:
            log.error(f"Missing columns: {missing}")
            return False
        return True

    def _check_missing_values(self, df):
        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                continue
            n = int(df[col].isna().sum())
            if n > 0:
                log.warning(f"Missing values in '{col}': {n} rows")
                return False
        return True

    def _check_duplicates(self, df):
        dupes = int(df.index.duplicated().sum())
        if dupes > 0:
            log.warning(f"Duplicate timestamps: {dupes}")
            return False
        return True

    def _check_price_sanity(self, df):
        """Negative price বা extreme spike detect করো"""
        ok = True
        for col in ['open', 'high', 'low', 'close']:
            non_positive = int((df[col] <= 0).sum())
            if non_positive > 0:
                log.error(f"Non-positive price in '{col}': {non_positive} rows")
                ok = False
            pct_change = df[col].pct_change().abs()
            spikes = int((pct_change > 0.05).sum())
            if spikes > 0:
                log.warning(f"Price spike (>5%) in '{col}': {spikes} occurrences")
        return ok

    def _check_ohlc_logic(self, df):
        """High সবচেয়ে বড়, Low সবচেয়ে ছোট হওয়া উচিত"""
        bad = int((
            (df['high'] < df['low'])
            | (df['high'] < df['open'])
            | (df['high'] < df['close'])
            | (df['low']  > df['open'])
            | (df['low']  > df['close'])
        ).sum())
        if bad > 0:
            log.warning(f"OHLC logic violation: {bad} candles")
            return False
        return True

    def _check_gaps(self, df, timeframe):
        """Expected timeframe onujayi missing candle ache kina - informational
        only. Kokhono validation fail kore na; weekend/holiday closure normal."""
        tf_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, 'm15': 15,
            '30m': 30, '1h': 60, 'h1': 60, '4h': 240, 'h4': 240,
            '1d': 1440, 'd1': 1440,
            'm1': 1, 'm5': 5, 'm30': 30,
        }
        normalized = str(timeframe).strip().lower()
        mins = tf_minutes.get(normalized)
        if not mins or len(df) < 2:
            return True

        try:
            expected_delta = pd.Timedelta(minutes=mins)
            idx = pd.to_datetime(df.index)
            if hasattr(idx, 'tz') and idx.tz is not None:
                idx = idx.tz_convert('UTC')
            actual_deltas = pd.to_timedelta(pd.Series(idx).diff().dropna())
            gaps = actual_deltas[actual_deltas > expected_delta * 1.5]

            if len(gaps) > 0:
                log.debug(f"Time gaps detected: {len(gaps)} gap(s) "
                          f"(market closed periods or missing data) - informational only")
                for ts, delta in gaps.head(3).items():
                    log.debug(f"  Gap at {ts}: {delta}")
        except Exception as e:
            log.warning(f"Gap check skipped (non-critical): {e}")
        return True