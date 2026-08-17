"""
data/backtest_ohlcv_cache.py — Point-in-time OHLCV for backtest MTF/SMC.

Problem
-------
In backtest mode DataFetcher.fetch_ohlcv() returns None (no live MT5).
SMCEngine then gets empty H4/M15 → smc_score=0 → fusion gate blocks 100%.

Solution
--------
Register full historical series (primary M15 + causally resampled H1/H4).
On each bar, set_asof(cursor_time). fetch_ohlcv returns only *completed*
higher-TF bars whose close time <= asof (no look-ahead).

Look-ahead contract
-------------------
- Resample with label='left', closed='left' (bar at T covers [T, T+freq)).
- Serve HTF bar at T only when asof >= T + freq (bar has fully closed).
- Primary TF bars: index <= asof.
"""
from __future__ import annotations

from typing import Optional
import threading
import pandas as pd

_lock = threading.RLock()
_series: dict[tuple[str, str], pd.DataFrame] = {}
_asof: Optional[pd.Timestamp] = None

# pandas offset aliases for completion check
_TF_DELTA = {
    "M1": pd.Timedelta(minutes=1),
    "M5": pd.Timedelta(minutes=5),
    "M15": pd.Timedelta(minutes=15),
    "M30": pd.Timedelta(minutes=30),
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
}

_RESAMPLE_RULE = {
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


def _norm_sym(symbol: str) -> str:
    return symbol.replace("/", "").replace("=", "").upper()


def _norm_tf(tf: str) -> str:
    t = (tf or "").strip().upper().replace(" ", "")
    mapping = {
        "1M": "M1", "5M": "M5", "15M": "M15", "30M": "M30",
        "1H": "H1", "60M": "H1", "4H": "H4", "240M": "H4",
        "1D": "D1", "D": "D1", "DAILY": "D1",
    }
    return mapping.get(t, t)


def clear() -> None:
    with _lock:
        _series.clear()
        global _asof
        _asof = None


def set_asof(ts) -> None:
    """Call once per backtest bar before any fetch_ohlcv for that bar."""
    global _asof
    with _lock:
        _asof = pd.Timestamp(ts)
        if _asof.tzinfo is not None:
            _asof = _asof.tz_convert("UTC").tz_localize(None)


def register_series(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    """Register a full OHLCV series (datetime index, open/high/low/close[/volume])."""
    if df is None or df.empty:
        return
    sym, tf = _norm_sym(symbol), _norm_tf(timeframe)
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        for col in ("datetime_utc", "time", "datetime", "date"):
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
                out = out.set_index(col)
                break
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out = out.sort_index()
    # normalize columns
    rename = {}
    for c in out.columns:
        cl = str(c).lower()
        if cl in ("open", "high", "low", "close", "volume", "tick_volume", "spread"):
            rename[c] = "volume" if cl == "tick_volume" else cl
    if rename:
        out = out.rename(columns=rename)
    if "volume" not in out.columns:
        out["volume"] = 0.0
    keep = [c for c in ("open", "high", "low", "close", "volume", "spread") if c in out.columns]
    out = out[keep].astype(float, errors="ignore")
    with _lock:
        _series[(sym, tf)] = out


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Causal OHLC resample: open=first, high=max, low=min, close=last, volume=sum."""
    tf = _norm_tf(target_tf)
    rule = _RESAMPLE_RULE.get(tf)
    if rule is None:
        raise ValueError(f"unsupported resample target {target_tf}")
    if df is None or df.empty:
        return pd.DataFrame()
    ohlc = df.copy()
    if ohlc.index.tz is not None:
        ohlc.index = ohlc.index.tz_convert("UTC").tz_localize(None)
    if "volume" not in ohlc.columns:
        ohlc["volume"] = 0.0
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "spread" in ohlc.columns:
        agg["spread"] = "mean"
    res = ohlc.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open", "high", "low", "close"])
    return res


def register_from_m15(symbol: str, m15_df: pd.DataFrame, also: tuple[str, ...] = ("H1", "H4")) -> None:
    """Register M15 plus resampled higher TFs (completed-bar semantics at fetch time)."""
    register_series(symbol, "M15", m15_df)
    for tf in also:
        try:
            htf = resample_ohlcv(m15_df, tf)
            if not htf.empty:
                register_series(symbol, tf, htf)
        except Exception:
            pass


def get_ohlcv(symbol: str, timeframe: str, limit: int = 300) -> Optional[pd.DataFrame]:
    """
    Return last `limit` bars with no look-ahead relative to current asof.

    Primary TF: index <= asof
    Higher TF: only bars whose period has fully closed by asof
               (bar_open + tf_delta <= asof)
    """
    sym, tf = _norm_sym(symbol), _norm_tf(timeframe)
    with _lock:
        df = _series.get((sym, tf))
        asof = _asof
    if df is None or df.empty:
        return None
    if asof is None:
        # No cursor yet — refuse to serve (forces callers to set_asof)
        return None

    delta = _TF_DELTA.get(tf)
    if delta is None:
        visible = df.loc[df.index <= asof]
    else:
        # completed bars only
        # bar at T covers [T, T+delta); closed when asof >= T+delta
        closed_mask = (df.index + delta) <= asof
        visible = df.loc[closed_mask]
        # also never include bars that opened after asof
        visible = visible.loc[visible.index <= asof]

    if visible.empty:
        return None
    if limit and len(visible) > limit:
        visible = visible.iloc[-limit:]
    return visible.copy()


def lookahead_self_check(m15_df: pd.DataFrame, symbol: str = "TEST") -> dict:
    """
    Verify completed-H4 bars at asof T never depend on M15 after T.

    Method: for several asof points, rebuild H4 from m15.loc[:asof] only and
    compare to precomputed H4 filtered to completed bars — max abs diff on OHLC
    must be ~0 for all fully closed bars.
    """
    clear()
    register_from_m15(symbol, m15_df, also=("H4",))
    failures = []
    # sample every ~N bars near the end
    idxs = list(range(max(0, len(m15_df) - 500), len(m15_df), 50))
    for i in idxs:
        asof = m15_df.index[i]
        set_asof(asof)
        served = get_ohlcv(symbol, "H4", limit=500)
        # rebuild from only visible m15
        visible_m15 = m15_df.loc[:asof]
        rebuilt = resample_ohlcv(visible_m15, "H4")
        # only completed
        delta = _TF_DELTA["H4"]
        rebuilt = rebuilt.loc[(rebuilt.index + delta) <= asof]
        if served is None or served.empty:
            if not rebuilt.empty:
                failures.append({"asof": str(asof), "reason": "served empty, rebuilt not"})
            continue
        # align
        common = served.index.intersection(rebuilt.index)
        if len(common) == 0:
            continue
        a = served.loc[common, ["open", "high", "low", "close"]]
        b = rebuilt.loc[common, ["open", "high", "low", "close"]]
        diff = (a - b).abs().max().max()
        if diff > 1e-9:
            failures.append({"asof": str(asof), "max_diff": float(diff), "n": len(common)})
    clear()
    return {"ok": len(failures) == 0, "checked_points": len(idxs), "failures": failures[:5]}
