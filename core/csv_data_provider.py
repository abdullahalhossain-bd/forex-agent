"""
core/csv_data_provider.py — HistoricalCSVDataProvider (multi-timeframe CSV backtest source).

PARITY GOAL
-----------
Replaces HistoricalMT5Provider as the primary backtest data source. Loads
historical bars from local CSV files (downloaded once via
scripts/download_historical_data.py), exposes the SAME `DataProvider`
interface as `LiveMT5Provider`, and produces market_out dicts that match
what `MarketAgent.run()` produces live.

ARCHITECTURE
------------
```
LIVE:     LiveMT5Provider     → wraps MarketAgent → MT5 ticks (real-time)
BACKTEST: HistoricalCSVDataProvider → loads CSVs → simulated bars
                  ↓
            DataProvider.get_market_out(symbol, tf) → market_out dict
                  ↓
            AITrader.evaluate_decision_core → SAME analysis/risk/permission
```

The CSV provider supports TWO file layouts:
  - Nested (preferred): data/historical/{SYMBOL}/{TF}.csv
  - Flat (legacy):      data/{SYMBOL}_{TF}.csv

ANTI-LOOK-AHEAD CONTRACT
------------------------
For every decision timestamp `T`:
  - Primary TF slice: only rows with `datetime <= T`
  - Higher TF (H1, H4, D1) slices: only rows whose bar OPENED at or before
    `T - tf_interval` (i.e. the bar has CLOSED by T)
  - mtf_bias: computed from the closed higher-TF bars only

This is tested explicitly in tests/parity/test_csv_provider_lookahead.py.

PARITY NOTES
------------
- Phase 2 fixes preserved:
  - Symbol-specific pip size / digits / contract size (via backtest.symbol_specs)
  - Spread-limit enforcement (via broker.spread_monitor.MAX_SPREAD_PIPS)
  - config.MAX_OPEN_TRADES default
  - warmup=300 default
  - Memory isolation (memory/_backtest/)
  - Explicit backtest news bypass
  - Real MTF bias computation
  - TEST_MODE fix (institutional_backtest.py)

- The provider REUSES the same indicator chain (add_canonical_indicators →
  ExtendedIndicators → Indicators) as MarketAgent and HistoricalMT5Provider.
  No new indicator code is introduced.

- tick_volume is required by volume-weighted indicators (OBV, VWAP, CMF, MFI,
  VWMA). If the CSV doesn't have `tick_volume`, the provider fills with 0
  (matching DataFetcher's behavior — those indicators neutralize).

- `spread` from CSV is exposed via `market_out["ind_ctx"]["spread_pips"]`
  so downstream gates (cost-aware EV, spread-limit) can use the historical
  spread instead of the DEFAULT_SPREAD_PIPS fallback. If the CSV's spread
  column is missing or all-zero, the provider falls back to the static
  DEFAULT_SPREAD_PIPS table (and logs this once).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from core.data_provider import DataProvider
from backtest.symbol_specs import get_pip_size

log = logging.getLogger("csv_data_provider")


# ── CSV file location helpers ─────────────────────────────────────────────

def _find_csv(symbol: str, timeframe: str, data_dir: Optional[Path] = None) -> Path | None:
    """Find the CSV file for (symbol, timeframe) in either layout.

    Returns the path if found, else None. Tries:
      1. data/historical/{SYMBOL}/{TF}.csv  (nested, preferred)
      2. data/{SYMBOL}_{TF}.csv             (flat, legacy)
    """
    if data_dir is None:
        try:
            from config import PROJECT_ROOT
            data_dir = Path(PROJECT_ROOT) / "data"
        except Exception:
            data_dir = Path(__file__).resolve().parents[1] / "data"

    sym = symbol.upper()
    tf = timeframe.upper()

    # Try nested first
    nested = data_dir / "historical" / sym / f"{tf}.csv"
    if nested.exists():
        return nested

    # Then flat
    flat = data_dir / f"{sym}_{tf}.csv"
    if flat.exists():
        return flat

    return None


def _normalize_timeframe(tf: str) -> str:
    """Normalize timeframe labels: '15m' → 'M15', 'h1' → 'H1', '4h' → 'H4'."""
    t = tf.strip().upper()
    # Handle "15M", "M15", "1H", "H1", "4H", "H4", "D1", "1D"
    if t in ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"):
        return t
    # Try to parse "15M" / "1H" / "4H" / "1D"
    if t.endswith("M") and t[:-1].isdigit():
        return f"M{t[:-1]}"
    if t.endswith("H") and t[:-1].isdigit():
        return f"H{t[:-1]}"
    if t.endswith("D") and t[:-1].isdigit():
        return "D1"
    return t


def _tf_to_minutes(tf: str) -> int:
    """Return the interval in minutes for a timeframe label."""
    t = _normalize_timeframe(tf)
    return {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440, "W1": 10080,
    }.get(t, 60)


# ── CSV loader ────────────────────────────────────────────────────────────

def _load_csv(filepath: Path, symbol: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with normalized columns.

    Returns df with:
      - DatetimeIndex (UTC), name="time"
      - Columns: open, high, low, close, volume (renamed from tick_volume if needed)
      - Optional: spread, real_volume (kept if present)
      - Sorted ascending, deduplicated
    """
    df = pd.read_csv(filepath, encoding="utf-8-sig")

    # Find timestamp column
    ts_col = None
    for candidate in ("datetime_utc", "datetime", "time", "timestamp", "date"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        raise ValueError(f"CSV {filepath} has no timestamp column (looked for: datetime_utc, datetime, time, timestamp, date)")

    # Parse as UTC
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.rename(columns={ts_col: "time"})
    df = df.set_index("time").sort_index()

    # Drop duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]

    # Normalize volume column name
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})
    if "volume" not in df.columns:
        if "tickvol" in df.columns:
            df = df.rename(columns={"tickvol": "volume"})
        else:
            df["volume"] = 0.0

    # Ensure OHLC are numeric
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    if "spread" in df.columns:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(0)

    return df


# ── Provider ──────────────────────────────────────────────────────────────

class HistoricalCSVDataProvider(DataProvider):
    """Multi-timeframe CSV-based historical data provider.

    Loads ALL needed timeframes upfront (one CSV per TF), then on each
    `advance_to(bar_index)` call, builds a market_out dict containing:
      - Primary TF slice (300-bar rolling window ending at cursor)
      - Indicators computed on that slice (same chain as live)
      - Regime detection on that slice
      - Real MTF bias computed from HIGHER TF slices (H1, H4, D1) that
        CLOSED at or before the cursor's timestamp

    The cursor moves through the PRIMARY timeframe's bars. Higher TF
    slices are aligned to the cursor's timestamp.

    Args:
        symbol: e.g. "EURUSD"
        primary_timeframe: e.g. "M15" or "H1"
        data_dir: where to look for CSVs (default: PROJECT_ROOT/data)
        mtf_timeframes: higher TFs to load for MTF bias (default: ["H1","H4","D1"]
            — uses primary TF as the entry signal, higher TFs only for bias)
        lookback_bars: rolling window size for primary TF slice (default: 300,
            matches live MarketAgent's limit=300)
    """

    def __init__(
        self,
        symbol: str,
        primary_timeframe: str = "H1",
        data_dir: Optional[Path] = None,
        mtf_timeframes: Optional[list[str]] = None,
        lookback_bars: int = 300,
    ):
        self._symbol = symbol.upper()
        self._primary_tf = _normalize_timeframe(primary_timeframe)
        self._lookback = lookback_bars
        self._mtf_tfs = mtf_timeframes or ["H1", "H4", "D1"]
        self._data_dir = data_dir

        # Load primary TF CSV
        primary_path = _find_csv(self._symbol, self._primary_tf, data_dir)
        if primary_path is None:
            raise FileNotFoundError(
                f"No CSV found for {self._symbol} {self._primary_tf}. "
                f"Looked in data/historical/{self._symbol}/{self._primary_tf}.csv "
                f"and data/{self._symbol}_{self._primary_tf}.csv. "
                f"Run: python scripts/download_historical_data.py --symbols {self._symbol} "
                f"--timeframes {self._primary_tf} --start 2025-07-01 --end 2026-08-08"
            )
        log.info(f"[CSVProvider] Loading primary {self._symbol} {self._primary_tf} from {primary_path}")
        self._primary_df = _load_csv(primary_path, self._symbol)
        if len(self._primary_df) == 0:
            raise ValueError(f"Primary CSV {primary_path} is empty")

        # Load higher TF CSVs for MTF bias (each optional — if missing, that TF is skipped)
        self._mtf_dfs: dict[str, pd.DataFrame] = {}
        for tf in self._mtf_tfs:
            tf_norm = _normalize_timeframe(tf)
            if tf_norm == self._primary_tf:
                continue  # don't double-load primary
            path = _find_csv(self._symbol, tf_norm, data_dir)
            if path is None:
                log.warning(f"[CSVProvider] No CSV for {self._symbol} {tf_norm} — MTF bias will skip this TF")
                continue
            try:
                self._mtf_dfs[tf_norm] = _load_csv(path, self._symbol)
                log.info(f"[CSVProvider] Loaded MTF {self._symbol} {tf_norm} ({len(self._mtf_dfs[tf_norm])} bars)")
            except Exception as e:
                log.warning(f"[CSVProvider] Failed to load {tf_norm} CSV: {e}")

        self._cursor = 0
        self._warned_default_spread = False

        # Decide ONCE which indicator chain is available, instead of
        # retrying the (possibly failing) import on every single bar.
        # This also stops the per-bar log spam when pandas_ta is missing.
        self._indicator_mode = self._resolve_indicator_mode()

    # ── DataProvider interface ────────────────────────────────────────────

    def _resolve_indicator_mode(self) -> str:
        """Probe once (at construction time) which indicator chain is
        importable, log the outcome a single time, and remember it so
        get_market_out() never has to retry a failing import per bar.

        Returns one of: "registry", "extended", "legacy".
        """
        try:
            import data.indicator_registry  # noqa: F401
            # `indicator_registry` itself doesn't import pandas_ta at module
            # level — it lazily imports `data.indicators_ext` (which DOES
            # need pandas_ta) inside add_canonical_indicators(), which runs
            # once PER BAR. So a bare import of indicator_registry can
            # succeed even when pandas_ta is missing, which used to cause
            # "registry" mode to be selected and then fail (and log an
            # ERROR) on every single bar. Probe the real dependency here,
            # once, so we fall back correctly instead of spamming.
            import data.indicators_ext  # noqa: F401
            return "registry"
        except Exception as e_registry:
            log.warning(f"[CSVProvider] indicator_registry unavailable ({e_registry}) — "
                        f"falling back to ExtendedIndicators, then legacy Indicators "
                        f"(this is logged once per run)")
            try:
                import data.indicators_ext  # noqa: F401
                return "extended"
            except Exception as e_ext:
                log.warning(f"[CSVProvider] ExtendedIndicators unavailable ({e_ext}) — "
                            f"falling back to legacy Indicators")
                return "legacy"

    def advance_to(self, bar_index: int) -> None:
        """Move the cursor to bar_index in the primary TF."""
        if bar_index < 0:
            bar_index = 0
        if bar_index >= len(self._primary_df):
            bar_index = len(self._primary_df) - 1
        self._cursor = bar_index

    def current_time(self):
        """Return the timestamp of the bar at the cursor."""
        return self._primary_df.index[self._cursor]

    def get_market_out(self, symbol: str, timeframe: str) -> dict:
        """Build the market_out dict for the cursor position.

        Returns the same dict shape as LiveMT5Provider.get_market_out
        and HistoricalMT5Provider.get_market_out:
          {df, ind_ctx, regime, regime_ctx, mtf_bias, symbol, timeframe, data_source}
        """
        if self._cursor < 0 or self._cursor >= len(self._primary_df):
            log.warning(f"[CSVProvider] cursor {self._cursor} out of bounds (len={len(self._primary_df)})")
            self._cursor = max(0, min(self._cursor, len(self._primary_df) - 1))

        # 1. Primary TF slice (causal — only bars up to and including cursor)
        start = max(0, self._cursor - self._lookback)
        df_slice = self._primary_df.iloc[start : self._cursor + 1].copy()

        # 2. Compute indicators (same chain as live MarketAgent).
        # Which chain to use was decided ONCE in __init__ (self._indicator_mode)
        # — no more per-bar import retries / log spam.
        ind_ctx = {}
        try:
            if self._indicator_mode == "registry":
                from data.indicator_registry import add_canonical_indicators, get_ai_context as _get_ctx
                df_slice = add_canonical_indicators(df_slice, include_patterns=True)
                ind_ctx = _get_ctx(df_slice)
            elif self._indicator_mode == "extended":
                from data.indicators_ext import ExtendedIndicators
                ind_ext = ExtendedIndicators()
                df_slice = ind_ext.add_all(df_slice, include_patterns=True)
                ind_ctx = ind_ext.get_ai_context(df_slice)
            else:
                from data.indicators import Indicators
                ind = Indicators()
                df_slice = ind.add_all(df_slice)
                ind_ctx = ind.get_ai_context(df_slice)
        except Exception as e:
            # A failure HERE (mid-run, on a specific bar) is a real bug and
            # must not be silently swallowed the way the old per-bar
            # try/except did — surface it instead of returning empty ind_ctx.
            log.error(f"[CSVProvider] indicator computation failed on bar "
                      f"{self._cursor} ({self._indicator_mode} mode): {e}")
            raise

        # 3. Regime detection (same as live)
        try:
            from analysis.market_regime import MarketRegimeDetector
            regime_detector = MarketRegimeDetector()
            regime_result = regime_detector.detect(df_slice)
            regime_ctx = regime_detector.get_ai_context(regime_result)
        except Exception as e:
            log.debug(f"[CSVProvider] regime detection unavailable: {e}")
            regime_result, regime_ctx = {}, {}

        # 4. Spread from CSV (if available), else fallback to DEFAULT_SPREAD_PIPS
        spread_pips = self._get_spread_pips()
        if spread_pips is not None:
            ind_ctx["spread_pips"] = spread_pips

        # 5. Real MTF bias from higher TFs (causal — only closed higher-TF bars)
        mtf_bias = self._compute_mtf_bias_from_csvs()

        return {
            "df": df_slice,
            "ind_ctx": ind_ctx,
            "regime": regime_result,
            "regime_ctx": regime_ctx,
            "mtf_bias": mtf_bias,
            "symbol": symbol,
            "timeframe": timeframe,
            "data_source": "historical_csv",
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_spread_pips(self) -> float | None:
        """Return the spread (in pips) for the cursor's bar, from CSV.

        Returns None if the CSV doesn't have a usable spread column.
        """
        if "spread" not in self._primary_df.columns:
            if not self._warned_default_spread:
                log.warning(f"[CSVProvider] No 'spread' column in {self._symbol} CSV — "
                            f"falling back to DEFAULT_SPREAD_PIPS table")
                self._warned_default_spread = True
            return None

        # MT5 spread is in POINTS, not pips. Convert: spread_pips = spread_points × point / pip
        # For 5-digit FX: point = 0.00001, pip = 0.0001 → spread_pips = spread_points / 10
        # For 3-digit JPY: point = 0.001, pip = 0.01 → spread_pips = spread_points / 10
        # For XAUUSD (2 digits): point = 0.01, pip = 0.01 → spread_pips = spread_points
        # For indices (1 digit): point = 1.0, pip = 1.0 → spread_pips = spread_points
        spread_points = float(self._primary_df.iloc[self._cursor].get("spread", 0))
        if spread_points == 0:
            # Use the column's recent mean as fallback for this bar
            recent = self._primary_df["spread"].iloc[max(0, self._cursor - 50):self._cursor + 1]
            recent_nonzero = recent[recent > 0]
            if len(recent_nonzero) > 0:
                spread_points = float(recent_nonzero.mean())
            else:
                return None  # truly no spread data

        pip = get_pip_size(self._symbol)
        # Heuristic: if pip is 0.0001 or 0.01 (i.e. pip = 10 × point), spread_pips = points / 10
        # If pip == point (e.g. XAUUSD pip=0.01, point=0.01), spread_pips = points
        # For our purposes, MT5 returns spread in points, and 1 pip = 10 points for 5-digit/3-digit FX
        # So spread_pips = spread_points / 10 for FX, spread_points for XAU/indices
        if pip in (0.0001, 0.01):  # 5-digit FX or 3-digit JPY
            return spread_points / 10.0
        # XAUUSD pip=0.01 == point=0.01, indices pip=1.0 == point=1.0
        return spread_points

    def _compute_mtf_bias_from_csvs(self) -> dict:
        """Compute a real MTF bias from the loaded higher-TF CSVs.

        Uses EMA-trend agreement across H1, H4, D1 (whichever are loaded).
        At cursor time T, only uses higher-TF bars that have CLOSED by T
        (i.e. whose open_time <= T - tf_interval).

        Returns {"bias": BULLISH/BEARISH/NEUTRAL, "confidence": HIGH/MEDIUM/LOW}.

        Falls back to NEUTRAL/LOW if no higher TF CSVs are available or if
        there's insufficient data (matches live's failure path when MTF
        fetch returns None).
        """
        try:
            current_time = self.current_time()
            bias_votes = []  # list of (bias, confidence) per TF

            for tf, df in self._mtf_dfs.items():
                if df is None or len(df) < 50:
                    continue

                # Causal: only bars that CLOSED by current_time.
                # A bar opened at time t closes at t + tf_interval. We want
                # bars where t + tf_interval <= current_time, i.e. t <= current_time - tf_interval.
                tf_minutes = _tf_to_minutes(tf)
                cutoff = current_time - pd.Timedelta(minutes=tf_minutes)
                causal = df[df.index <= cutoff]
                if len(causal) < 50:
                    continue

                # EMA trend: 20 vs 50 (matches MultiTimeframeAnalyzer logic)
                ema_fast = causal["close"].ewm(span=20, adjust=False).mean()
                ema_slow = causal["close"].ewm(span=50, adjust=False).mean()
                last_close = float(causal["close"].iloc[-1])
                last_fast = float(ema_fast.iloc[-1])
                last_slow = float(ema_slow.iloc[-1])
                if last_close > last_fast > last_slow:
                    bias_votes.append(("BULLISH", "HIGH"))
                elif last_close > last_fast:
                    bias_votes.append(("BULLISH", "MEDIUM"))
                elif last_close < last_fast < last_slow:
                    bias_votes.append(("BEARISH", "HIGH"))
                elif last_close < last_fast:
                    bias_votes.append(("BEARISH", "MEDIUM"))
                else:
                    bias_votes.append(("NEUTRAL", "LOW"))

            if not bias_votes:
                return {"bias": "NEUTRAL", "confidence": "LOW"}

            # Aggregate
            bullish = sum(1 for b, _ in bias_votes if b == "BULLISH")
            bearish = sum(1 for b, _ in bias_votes if b == "BEARISH")
            total = len(bias_votes)
            if bullish >= 0.75 * total:
                return {"bias": "BULLISH", "confidence": "HIGH" if bullish == total else "MEDIUM"}
            if bearish >= 0.75 * total:
                return {"bias": "BEARISH", "confidence": "HIGH" if bearish == total else "MEDIUM"}
            if bullish > bearish:
                return {"bias": "BULLISH", "confidence": "LOW"}
            if bearish > bullish:
                return {"bias": "BEARISH", "confidence": "LOW"}
            return {"bias": "NEUTRAL", "confidence": "LOW"}
        except Exception as e:
            log.debug(f"[CSVProvider] MTF bias computation failed: {e}")
            return {"bias": "NEUTRAL", "confidence": "LOW"}

    # ── Public properties for tests ───────────────────────────────────────

    @property
    def primary_df(self) -> pd.DataFrame:
        """Access to the primary TF DataFrame (for tests)."""
        return self._primary_df

    @property
    def mtf_dfs(self) -> dict[str, pd.DataFrame]:
        """Access to the loaded higher-TF DataFrames (for tests)."""
        return self._mtf_dfs

    @property
    def n_bars(self) -> int:
        """Number of bars in the primary TF."""
        return len(self._primary_df)