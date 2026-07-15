# data/validator.py
# ============================================================
# Day 7+ — Data Quality Check System
# ভুল data → ভুল analysis → ভুল trade
# এই module সেটা prevent করে
# ============================================================

import pandas as pd
import numpy as np
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
        log.info(f"Validating data: {symbol} {timeframe} | rows={len(df)}")
        passed = True

        passed &= self._check_empty(df)
        if not passed:
            return False  # Short-circuit: no point checking columns on empty df

        passed &= self._check_columns(df)
        if not passed:
            return False  # Short-circuit: missing required columns

        self._check_missing_values(df)
        self._check_duplicates(df)
        self._check_price_sanity(df)
        self._check_ohlc_logic(df)
        self._check_gaps(df, timeframe)

        if passed:
            log.info("Data validation passed")
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
        required = {'open', 'high', 'low', 'close', 'volume'}
        missing  = required - set(df.columns)
        if missing:
            log.error(f"Missing columns: {missing}")
            return False
        return True

    def _check_missing_values(self, df):
        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                continue
            n = df[col].isna().sum()
            if n > 0:
                log.warning(f"Missing values in '{col}': {n} rows")

    def _check_duplicates(self, df):
        dupes = df.index.duplicated().sum()
        if dupes > 0:
            log.warning(f"Duplicate timestamps: {dupes}")

    def _check_price_sanity(self, df):
        """Negative price বা extreme spike detect করো"""
        for col in ['open', 'high', 'low', 'close']:
            if (df[col] <= 0).any():
                log.error(f"Non-positive price in '{col}'")
            # Spike: ১ candle-এ ৫% এর বেশি move
            pct_change = df[col].pct_change().abs()
            spikes = (pct_change > 0.05).sum()
            if spikes > 0:
                log.warning(f"Price spike (>5%) in '{col}': {spikes} occurrences")

    def _check_ohlc_logic(self, df):
        """High সবচেয়ে বড়, Low সবচেয়ে ছোট হওয়া উচিত"""
        bad = (
            (df['high'] < df['low'])
            | (df['high'] < df['open'])
            | (df['high'] < df['close'])
            | (df['low']  > df['open'])
            | (df['low']  > df['close'])
        ).sum()
        if bad > 0:
            log.warning(f"OHLC logic violation: {bad} candles")

    def _check_gaps(self, df, timeframe):
        """Expected timeframe অনুযায়ী missing candle আছে কিনা"""
        tf_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15,
            '30m': 30, '1h': 60, '4h': 240, '1d': 1440,
        }
        mins = tf_minutes.get(timeframe)
        if not mins or len(df) < 2:
            return

        try:
            expected_delta = pd.Timedelta(minutes=mins)
            # Bug fix: ensure index is datetime before diff
            idx = pd.to_datetime(df.index)
            actual_deltas = pd.Series(idx).diff().dropna()
            # Bug fix: use pd.to_timedelta to ensure consistent type
            actual_deltas = pd.to_timedelta(actual_deltas)
            gaps = actual_deltas[actual_deltas > expected_delta * 1.5]

            if len(gaps) > 0:
                log.warning(f"Time gaps detected: {len(gaps)} gaps "
                            f"(market closed periods or missing data)")
                for ts, delta in gaps.head(3).items():
                    log.debug(f"  Gap at {ts}: {delta}")
        except Exception as e:
            log.warning(f"Gap check skipped (non-critical): {e}")