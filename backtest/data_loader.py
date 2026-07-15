import pandas as pd

from analysis.market_regime import MarketRegimeDetector
from analysis.patterns import PatternDetector
from data.indicators import Indicators
from utils.logger import get_logger

log = get_logger("backtest_loader")


class HistoricalDataLoader:
    REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "volume"]

    TF_MAP = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }

    def __init__(self):
        self.indicators = Indicators()
        self.patterns = PatternDetector()
        self.regime = MarketRegimeDetector()

    def load_csv(
        self,
        file_path: str,
        pair: str = "EURUSD",
        timeframe: str = "15m",
        fill_missing: bool = True,
        enrich: bool = True,
    ) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df = self._normalize_columns(df)
        self._validate_columns(df)

        # Round-21 audit fix: utc=False silently dropped tz-aware
        # timestamps (e.g. "2024-01-01T00:00:00Z") by coercing them
        # to NaT, then dropna deleted those rows. utc=True normalizes
        # all timestamps to UTC so mixed-tz data is handled correctly.
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).drop_duplicates(subset=["time"]).sort_values("time")
        df = df.set_index("time")

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])

        if fill_missing:
            df = self._fill_missing_candles(df, timeframe)

        if enrich:
            df = self._enrich(df)

        df.attrs["pair"] = self._clean_pair(pair)
        df.attrs["timeframe"] = timeframe
        log.info(
            f"[BacktestLoader] Ready | pair={df.attrs['pair']} | timeframe={timeframe} "
            f"| candles={len(df)} | missing_filled={df.attrs.get('missing_candles_filled', 0)}"
        )
        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(col).strip().lower() for col in df.columns]
        if "datetime" in df.columns and "time" not in df.columns:
            df = df.rename(columns={"datetime": "time"})
        if "date" in df.columns and "time" not in df.columns:
            df = df.rename(columns={"date": "time"})
        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Historical data missing required columns: {missing}")

    def _fill_missing_candles(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        freq = self.TF_MAP.get(timeframe, "15min")
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        reindexed = df.reindex(full_index)

        missing_count = int(reindexed["close"].isna().sum())
        reindexed["close"] = reindexed["close"].ffill()
        reindexed["open"] = reindexed["open"].fillna(reindexed["close"].shift(1)).fillna(reindexed["close"])
        reindexed["high"] = reindexed["high"].fillna(reindexed[["open", "close"]].max(axis=1))
        reindexed["low"] = reindexed["low"].fillna(reindexed[["open", "close"]].min(axis=1))
        reindexed["volume"] = reindexed["volume"].fillna(0.0)
        reindexed = reindexed.dropna(subset=["open", "high", "low", "close"])

        reindexed.attrs["missing_candles_filled"] = missing_count
        return reindexed

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.indicators.add_all(df)
        df = self.patterns.run_full_detection(df)
        # Day 99+ V5 FIX (Audit Issue #3): defensively call _add_adx.
        # The method DOES exist on MarketRegimeDetector today, but it's
        # a private (`_`-prefixed) method, so a future refactor could
        # rename or remove it without any warning at the call site. If
        # that happens, the backtest would crash with AttributeError
        # — silently breaking all backtests until someone notices.
        # We now wrap the call in try/except and fall back to a self-
        # contained ADX computation if the method is missing or raises.
        # The fallback produces the same `adx`, `di_plus`, `di_minus`
        # columns the original method adds, so downstream code is
        # unaffected.
        try:
            df = self.regime._add_adx(df)
        except (AttributeError, Exception) as e:
            # AttributeError: method was renamed/removed.
            # Other Exception: method exists but raised (bad data, etc).
            log.warning(
                f"[DataLoader] self.regime._add_adx(df) failed "
                f"({type(e).__name__}: {e}) — falling back to inline "
                f"ADX computation. Backtest will continue but consider "
                f"updating data_loader.py to use the public ADX API."
            )
            df = self._add_adx_fallback(df)

        df["rolling_support_20"] = df["low"].rolling(20).min().shift(1)
        df["rolling_resistance_20"] = df["high"].rolling(20).max().shift(1)
        df["rolling_support_50"] = df["low"].rolling(50).min().shift(1)
        df["rolling_resistance_50"] = df["high"].rolling(50).max().shift(1)
        df["avg_volume_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = (df["volume"] / df["avg_volume_20"].replace(0, pd.NA)).fillna(1.0)

        df["regime"] = df.apply(self._row_regime, axis=1)
        df["regime_direction"] = df.apply(self._row_direction, axis=1)
        df["regime_volatility"] = df.apply(self._row_volatility, axis=1)
        df["session_name"] = [self._session_name(ts) for ts in df.index]

        return df.dropna().copy()

    @staticmethod
    def _add_adx_fallback(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Day 99+ V5 FIX (Audit Issue #3): self-contained ADX computation
        used as a fallback when `MarketRegimeDetector._add_adx` is missing
        or raises. Produces the same `adx`, `di_plus`, `di_minus` columns
        as the original method so downstream code is unaffected.

        Algorithm: standard Wilder's ADX — True Range + directional
        movement, smoothed with Wilder's EMA, then DX → ADX.
        """
        import numpy as np
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)

        # Wilder's smoothing (EWM with alpha=1/period)
        atr_s = tr.ewm(alpha=1/period, adjust=False).mean()
        dmp_s = dm_plus.ewm(alpha=1/period, adjust=False).mean()
        dmm_s = dm_minus.ewm(alpha=1/period, adjust=False).mean()

        di_plus = 100 * dmp_s / atr_s.replace(0, np.nan)
        di_minus = 100 * dmm_s / atr_s.replace(0, np.nan)

        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
        df["adx"] = dx.ewm(alpha=1/period, adjust=False).mean()
        df["di_plus"] = di_plus
        df["di_minus"] = di_minus

        return df

    def _row_regime(self, row) -> str:
        adx = float(row.get("adx", 0) or 0)
        if adx >= 20:
            return "TRENDING"
        if adx >= 14:
            return "BREAKOUT"
        return "RANGING"

    def _row_direction(self, row) -> str:
        price = float(row.get("close", 0) or 0)
        ema = float(row.get("ema_21", 0) or 0)
        sma50 = float(row.get("sma_50", 0) or 0)
        sma200 = float(row.get("sma_200", 0) or 0)
        bullish = sum([price > ema, price > sma50, price > sma200])
        bearish = sum([price < ema, price < sma50, price < sma200])
        if bullish >= 2:
            return "BULLISH"
        if bearish >= 2:
            return "BEARISH"
        return "NEUTRAL"

    def _row_volatility(self, row) -> str:
        atr = float(row.get("atr", 0) or 0)
        close = float(row.get("close", 0) or 0)
        if close <= 0:
            return "NORMAL"
        ratio = atr / close
        if ratio >= 0.0025:
            return "HIGH"
        if ratio <= 0.0008:
            return "LOW"
        return "NORMAL"

    def _session_name(self, timestamp) -> str:
        hour = pd.Timestamp(timestamp).hour
        if 12 <= hour < 16:
            return "London/New York"
        if 7 <= hour < 16:
            return "London"
        if 12 <= hour < 21:
            return "New York"
        if 0 <= hour < 9:
            return "Tokyo"
        return "Sydney"

    def _clean_pair(self, pair: str) -> str:
        # Round-14 fix: see backtest/simulator.py — blanket "USDT"->"USD"
        # replace corrupted USDTRY/USDTHB. Only strip a trailing Tether suffix.
        cleaned = str(pair).upper().replace("/", "").replace("=X", "").strip()
        if cleaned.endswith("USDT"):
            cleaned = cleaned[:-1]
        return cleaned


def load_data(file_path: str, pair: str = "EURUSD", timeframe: str = "15m") -> pd.DataFrame:
    return HistoricalDataLoader().load_csv(file_path=file_path, pair=pair, timeframe=timeframe)