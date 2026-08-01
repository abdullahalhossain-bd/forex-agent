# tests/backtest_fibonacci.py
# ============================================================
# Walk-forward backtester for FibonacciEngine, driven off MT5 data.
#
# MOVED (was analysis/backtest_fibonacci.py -> now tests/backtest_fibonacci.py).
# This is a test/verification script, not analysis-pipeline code, so it
# belongs in tests/ next to the other test scripts. No import changes were
# needed for the move itself: `from analysis.fibonacci import FibonacciEngine`
# is a project-root-relative import and works the same regardless of which
# subfolder this file physically lives in, as long as it's run with the
# project root on sys.path (e.g. `python -m tests.backtest_fibonacci ...`
# or `python tests/backtest_fibonacci.py ...` from the project root).
#
# Design goals (institutional review checklist):
#   - No look-ahead: at bar i, the engine only ever sees df.iloc[:i+1]
#     (bars 0..i, all closed). Entry fills happen at bar i+1's OPEN,
#     never at bar i's close — you can't trade a signal at the same
#     price you used to generate it. Same no-look-ahead rule now also
#     applies to the HTF grid (see "NEW" section below): at bar i, the
#     HTF slice only ever includes HTF bars that were fully CLOSED by
#     bar i's timestamp.
#   - Spread + slippage modeled on every fill (both legs), not just
#     the tightest historical spread.
#   - SL/TP checked against each subsequent bar's high/low, not close,
#     so intrabar stop-outs aren't missed.
#   - Ambiguous same-bar SL+TP touch resolved conservatively (SL wins),
#     since we don't have tick data to know which came first.
#   - Reports trades AND signal quality (confidence buckets, zone,
#     trend, and now HTF confirmation / time-cluster / secondary-swing
#     cluster) so you can see not just "did it make money" but "which
#     signals — and which specific FEATURE of a signal — were actually
#     good."
#
# NEW in this version (catching this file up to fibonacci.py):
#   fibonacci.py's analyze() grew three features this file never
#   exercised because it always called `engine.analyze(window)` with no
#   extra arguments:
#     1. htf_df / htf_timeframe   -> multi-timeframe Fib confluence
#     2. secondary swing / "Fibonacci cluster" -> computed automatically
#        inside analyze(), but the result fields were never captured
#     3. time projections / time-cluster ("price+time cluster")        -> same,
#        computed automatically but never captured
#   (1) required real wiring (fetching + no-look-ahead-slicing a second,
#   higher-timeframe series). (2) and (3) just required reading fields
#   that were already in the result dict and reporting on them.
#   All three are now wired in and broken out in the report below, plus
#   the new multi-pair "recent signals" scan (--scan-all-pairs) has them
#   as columns too.
#
# NEW in this version (the actual feature ask): --scan-all-pairs mode.
#   Runs the same walk-forward backtest across every symbol in
#   config.SYMBOLS (or a --symbols-file), keeps only the signals whose
#   ENTRY fell in the last --recent-days days of each pair's data, and
#   reports: which pair fired which signal on which day, whether it
#   would have won or lost, and — aggregated across all pairs — which
#   zone / strategy_type / confidence bucket / HTF-confirmation / time-
#   cluster status tends to sit on the winning side vs the losing side.
#
# Requires: pip install MetaTrader5 pandas numpy --break-system-packages
# Must run on a machine with an MT5 terminal installed and logged in
# (the MetaTrader5 python package talks to the local terminal, not a
# remote server — this will NOT work on a headless Linux box with no
# terminal installed).
# ============================================================

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# FIX (path robustness): the previous version relied on sys.path.insert(0, ".")
# in the fallback branch, which only works if the CURRENT WORKING DIRECTORY
# happens to be the project root when you launch python. Running the script
# the ordinary way — `python tests/backtest_fibonacci.py` — puts this
# script's OWN folder (tests/) on sys.path[0], not the project root, and "."
# is whatever directory your shell happened to be in, not necessarily where
# fibonacci.py lives. That caused "ModuleNotFoundError: No module named
# 'analysis'" / "'fibonacci'" even though the project layout was fine.
# Fix: derive the project root from THIS FILE's own location (works no
# matter what your cwd is or how you invoke the script), and add it to
# sys.path explicitly before importing.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)   # parent of tests/ == project root
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from utils.logger import get_logger
    log = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger(__name__)

try:
    from analysis.fibonacci import FibonacciEngine, HTF_HIERARCHY, TF_SWING_WINDOW, DEFAULT_SWING_WINDOW
except ImportError:
    # Standalone fallback: fibonacci.py sitting directly next to THIS file
    # (not inside analysis/) — e.g. testing outside the full project tree.
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)
    from fibonacci import FibonacciEngine, HTF_HIERARCHY, TF_SWING_WINDOW, DEFAULT_SWING_WINDOW

try:
    from config import SYMBOLS as CONFIG_SYMBOLS
except ImportError:
    CONFIG_SYMBOLS = []  # --scan-all-pairs falls back to --symbols-file if this is empty

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # only fail loudly when connect() is actually called



# ── Timeframe string -> MT5 constant ───────────────────────────
_TF_MAP = {
    "1m": "TIMEFRAME_M1", "M1": "TIMEFRAME_M1",
    "5m": "TIMEFRAME_M5", "M5": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15", "M15": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30", "M30": "TIMEFRAME_M30",
    "1h": "TIMEFRAME_H1", "H1": "TIMEFRAME_H1",
    "4h": "TIMEFRAME_H4", "H4": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1", "D1": "TIMEFRAME_D1",
}

# ── Timeframe string -> bar duration in minutes ────────────────
# Used to (a) know when an HTF bar is actually CLOSED relative to a
# primary-timeframe bar's timestamp, and (b) size how many HTF/primary
# bars to fetch to cover N calendar days.
_TF_MINUTES = {
    "1m": 1, "M1": 1,
    "5m": 5, "M5": 5,
    "15m": 15, "M15": 15,
    "30m": 30, "M30": 30,
    "1h": 60, "H1": 60,
    "4h": 240, "H4": 240,
    "1d": 1440, "D1": 1440,
}


@dataclass
class BacktestConfig:
    """
    Externalized config — no magic numbers buried in the loop.

    symbol            : broker symbol string, e.g. "EURUSD". Note this is
                         broker-specific (some brokers suffix "EURUSD.a" /
                         "EURUSDm") — pass the EXACT string your terminal
                         shows in Market Watch.
    timeframe         : one of the keys in _TF_MAP.
    bars              : how many closed bars of history to pull.
    spread_pips       : round-trip cost assumption. Don't leave at 0 —
                         that silently makes marginal strategies look
                         profitable. Check your broker's typical spread
                         for this symbol and use a realistic (not best-case)
                         figure.
    slippage_pips     : additional adverse fill assumption on entry.
    min_confidence    : signals below this confidence are ignored (treated
                         as WAIT) — lets you test "only take A-grade setups."
    max_holding_bars  : force-close a trade at this bar's close if neither
                         SL nor TP was hit (avoids an unrealistic
                         infinite-hold assumption).
    risk_per_trade_pct: for equity-curve simulation only (position sizing is
                         NOT sent to a broker here — this is analysis, not
                         execution).
    use_htf_confluence: NEW — if True, auto-fetch the next-higher timeframe
                         (via fibonacci.HTF_HIERARCHY) and feed it into
                         engine.analyze() as htf_df/htf_timeframe, exactly
                         like fibonacci.py supports but this file never
                         exercised before. No-look-ahead: at each bar, only
                         HTF bars fully closed by that bar's time are used.
    htf_timeframe     : NEW — override the auto-picked HTF (must be a valid
                         key in _TF_MAP). Leave None to use HTF_HIERARCHY.
    """
    symbol: str
    timeframe: str = "H1"
    bars: int = 3000
    spread_pips: float = 1.2
    slippage_pips: float = 0.5
    min_confidence: int = 55
    max_holding_bars: int = 48
    risk_per_trade_pct: float = 1.0
    starting_equity: float = 10_000.0
    require_trigger_candle: bool = True
    min_rr: float = 1.5
    use_htf_confluence: bool = True
    htf_timeframe: Optional[str] = None
    # FIX (critical — backtest/live parity): the engine must see the SAME
    # size lookback window in backtest as it will in live trading (the
    # live caller does fetcher.fetch_ohlcv(symbol, tf, limit=N) — a FIXED
    # window, not "everything since the start of history"). Previously
    # this file fed engine.analyze() an ever-GROWING window (df.iloc[:i+1],
    # unbounded), so find_swing_points()'s `recent_cutoff = n // 2` search
    # covered a completely different (much larger, older) span late in a
    # long backtest than a live 200-bar fetch ever would — meaning the
    # backtest was trading a different Fibonacci grid than production ever
    # sees. lookback_bars now bounds the window the same way in both
    # places. 200 matches the example live call in fibonacci.py's
    # __main__; override with --lookback-bars if your live fetcher uses
    # a different limit.
    lookback_bars: int = 200



@dataclass
class Trade:
    entry_idx: int
    entry_time: pd.Timestamp
    direction: str          # 'BUY' / 'SELL'
    entry_price: float
    sl: float
    tp: float
    confidence: int
    zone: str
    trend: str
    confluence_strength: int
    strategy_type: str = "UNKNOWN"
    trigger_pattern: Optional[str] = None
    # NEW — fields that already existed in fibonacci.py's result dict but
    # were never captured here before this version:
    htf_grid_used: bool = False       # an HTF grid was actually available/used this bar
    htf_confirmed: bool = False       # top confluence zone includes an "HTF ..." reason
    in_time_cluster: bool = False     # current bar landed inside a Fib time projection window
    price_time_cluster: bool = False  # price confluence AND time-cluster coincided (strongest per engine)
    secondary_swing_used: bool = False  # a same-TF "Fibonacci cluster" leg was found and checked
    exit_idx: Optional[int] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None   # 'WIN' / 'LOSS' / 'TIMEOUT'
    still_open: bool = False        # True only if closed via end-of-data mark-to-market, not a real exit
    pips: float = 0.0
    r_multiple: float = 0.0


# ═══════════════════════════════════════════════════════════════
# MT5 DATA LAYER
# ═══════════════════════════════════════════════════════════════

def connect_mt5() -> None:
    """Initialize connection to the locally running, logged-in MT5 terminal."""
    if mt5 is None:
        raise RuntimeError(
            "MetaTrader5 package not installed. Run: "
            "pip install MetaTrader5 --break-system-packages "
            "(and this must run on Windows with an MT5 terminal installed/logged in)."
        )
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    log.info("MT5 connected | terminal=%s", mt5.terminal_info())


def fetch_closed_bars(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    """
    Pull `bars` candles from MT5 and return only fully CLOSED bars.

    copy_rates_from_pos(symbol, tf, 0, n) returns the CURRENTLY FORMING
    bar at position 0 — including it would feed the engine a candle whose
    close is still moving, which is a look-ahead/repaint bug. We always
    fetch one extra bar and drop the most recent one.
    """
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed.")

    tf_const = getattr(mt5, _TF_MAP.get(timeframe, ""), None)
    if tf_const is None:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Valid: {sorted(_TF_MAP)}")

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Symbol '{symbol}' not found/enabled in Market Watch — "
                            f"check the exact broker symbol name (suffixes like .a/.m/.raw vary).")

    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, bars + 1)
    if rates is None or len(rates) < 2:
        raise RuntimeError(f"copy_rates_from_pos returned no data for {symbol} {timeframe}: "
                            f"{mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["time", "open", "high", "low", "close", "volume"]]

    df = df.iloc[:-1].reset_index(drop=True)   # drop forming bar
    log.info("Fetched %d closed bars | %s %s | %s -> %s",
             len(df), symbol, timeframe, df["time"].iloc[0], df["time"].iloc[-1])
    return df


def resolve_htf(timeframe: str, override: Optional[str]) -> Optional[str]:
    """Pick the higher timeframe to use for multi-timeframe confluence."""
    if override:
        return override
    return HTF_HIERARCHY.get(timeframe)


def htf_bars_needed(primary_bars: int, primary_tf: str, htf_tf: str,
                     min_bars: int = 500) -> int:
    """
    How many HTF bars to fetch so the HTF series covers roughly the same
    calendar span as `primary_bars` bars of `primary_tf`, plus a buffer
    for the HTF engine's own swing-detection warmup.
    """
    primary_min = _TF_MINUTES.get(primary_tf, 60)
    htf_min = _TF_MINUTES.get(htf_tf, 240)
    ratio = max(1, htf_min // max(primary_min, 1))
    warmup = TF_SWING_WINDOW.get(htf_tf, DEFAULT_SWING_WINDOW) * 3 + 50
    return max(min_bars, primary_bars // ratio + warmup)


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

class FibonacciBacktester:
    def __init__(self, cfg: BacktestConfig, htf_df: pd.DataFrame = None,
                 htf_timeframe: str = None):
        self.cfg = cfg
        self.engine = FibonacciEngine(timeframe=cfg.timeframe, symbol=cfg.symbol,
                                       require_trigger_candle=cfg.require_trigger_candle,
                                       min_rr=cfg.min_rr)
        self._pip_size = self._resolve_pip_size(cfg.symbol)

        # NEW — optional HTF series for multi-timeframe confluence, sliced
        # per-bar with no look-ahead (see _htf_slice below).
        self.htf_df = htf_df
        self.htf_timeframe = htf_timeframe
        self._htf_minutes = _TF_MINUTES.get(htf_timeframe, 0) if htf_timeframe else 0
        # FIX (crash): fetch_closed_bars() always returns tz-aware UTC
        # timestamps. Calling .to_numpy() directly on a tz-aware datetime
        # column does NOT give a clean numpy datetime64 array — raw
        # datetime64 has no tz concept, so pandas silently falls back to an
        # object array of tz-aware Timestamp objects. np.searchsorted then
        # tries to compare those against a tz-NAIVE np.datetime64 `cutoff`
        # (built below) and raises "Cannot compare tz-naive and tz-aware
        # timestamps". Fix: strip the tz label on both sides consistently
        # (values already represent the same UTC instant either way, so no
        # actual time shift happens) before either becomes a numpy array.
        self._htf_times = (
            htf_df["time"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
            if htf_df is not None else None
        )
        self._htf_min_bars = (TF_SWING_WINDOW.get(htf_timeframe, DEFAULT_SWING_WINDOW) * 3
                               if htf_timeframe else 0)

    @staticmethod
    def _resolve_pip_size(symbol: str) -> float:
        try:
            from utils.pip_utils import pip_size
            return pip_size(symbol)
        except ImportError:
            # Fallback only used if utils.pip_utils isn't importable in this
            # environment — JPY pairs use a 100x larger pip than other FX.
            return 0.01 if "JPY" in symbol.upper() else 0.0001

    def _htf_slice(self, t_i: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Return only the HTF bars fully CLOSED as of primary-bar time `t_i`
        (no look-ahead). An HTF bar with open time T is closed once
        T + htf_bar_duration <= t_i, i.e. T <= t_i - htf_bar_duration.
        """
        if self.htf_df is None or self._htf_times is None:
            return None
        # Same tz-naive-UTC normalization as self._htf_times above — t_i
        # comes from the primary df's tz-aware "time" column, so it must be
        # stripped the same way before it can be compared against
        # self._htf_times via np.searchsorted.
        t_i_naive = pd.Timestamp(t_i)
        if t_i_naive.tzinfo is not None:
            t_i_naive = t_i_naive.tz_convert("UTC").tz_localize(None)
        cutoff = np.datetime64(t_i_naive - pd.Timedelta(minutes=self._htf_minutes))
        idx = int(np.searchsorted(self._htf_times, cutoff, side="right"))
        if idx < max(self._htf_min_bars, 3):
            return None
        return self.htf_df.iloc[:idx]

    def run(self, df: pd.DataFrame) -> list[Trade]:
        trades: list[Trade] = []
        open_trade: Optional[Trade] = None

        warmup = self.engine.swing_window * 3 + 5
        n = len(df)

        for i in range(warmup, n - 1):
            # FIX (critical): bounded, FIXED-SIZE window — bars
            # [max(0, i+1-lookback_bars) .. i] — matching what the live
            # fetcher actually hands the engine (a fixed `limit=N` pull),
            # instead of the previous df.iloc[:i+1] which grew every bar
            # and silently changed which swing find_swing_points() picks
            # as the backtest progressed. Still strictly no-look-ahead:
            # every bar in the window is <= i.
            start = max(0, i + 1 - self.cfg.lookback_bars)
            window = df.iloc[start : i + 1]

            # ── manage an already-open trade using bar i (now closed) ──
            if open_trade is not None:
                bar = df.iloc[i]
                open_trade = self._check_exit(open_trade, i, bar)
                if open_trade.outcome is not None:
                    trades.append(open_trade)
                    open_trade = None
                    continue  # don't open a new trade same bar we just closed one

            # ── force-timeout check ──
            if open_trade is not None and (i - open_trade.entry_idx) >= self.cfg.max_holding_bars:
                bar = df.iloc[i]
                open_trade.exit_idx = i
                open_trade.exit_time = bar["time"]
                open_trade.exit_price = bar["close"]
                open_trade.outcome = "TIMEOUT"
                self._finalize_pnl(open_trade)
                trades.append(open_trade)
                open_trade = None

            if open_trade is not None:
                continue  # still in a position, don't evaluate new entries

            # ── evaluate a fresh signal off bars 0..i ──
            bar_time = df.iloc[i]["time"]
            htf_window = self._htf_slice(bar_time) if self.htf_timeframe else None
            result = self.engine.analyze(window, htf_df=htf_window, htf_timeframe=self.htf_timeframe)
            signal = result.get("signal", {})
            bias = signal.get("bias", "WAIT")
            conf = signal.get("confidence", 0)

            if bias not in ("BUY", "SELL") or conf < self.cfg.min_confidence:
                continue
            if signal.get("sl") is None or signal.get("tp1") is None:
                continue  # incomplete signal, skip rather than guess

            # Entry fills at NEXT bar's open (i+1) — the earliest price
            # actually tradable after the decision made at close of bar i.
            next_bar = df.iloc[i + 1]
            raw_open = next_bar["open"]
            cost = (self.cfg.spread_pips + self.cfg.slippage_pips) * self._pip_size

            if bias == "BUY":
                entry_price = raw_open + cost
            else:
                entry_price = raw_open - cost

            # NEW — pull the HTF/time-cluster/secondary-swing fields that
            # analyze() already computes, so the report can break trades
            # down by these features (previously discarded entirely).
            conf_zones = result.get("confluence", []) or []
            htf_confirmed = any(
                any(r.startswith("HTF ") for r in z.get("reasons", []))
                for z in conf_zones
            )
            time_cluster = result.get("time_cluster", {}) or {}

            open_trade = Trade(
                entry_idx=i + 1,
                entry_time=next_bar["time"],
                direction=bias,
                entry_price=entry_price,
                sl=signal["sl"],
                tp=signal["tp1"],
                confidence=conf,
                zone=signal.get("zone", "UNKNOWN"),
                trend=result.get("trend", "UNKNOWN"),
                confluence_strength=(conf_zones or [{}])[0].get("strength", 0),
                strategy_type=signal.get("strategy_type", "UNKNOWN"),
                trigger_pattern=signal.get("trigger_pattern"),
                htf_grid_used=bool(result.get("htf_grid_used", False)),
                htf_confirmed=htf_confirmed,
                in_time_cluster=bool(time_cluster.get("in_cluster", False)),
                price_time_cluster=bool(signal.get("price_time_cluster", False)),
                secondary_swing_used=result.get("secondary_swing") is not None,
            )

        # If a trade is still open at the end of data, close it at last close
        # (mark-to-market, not a real exit) so it's still counted in stats.
        if open_trade is not None:
            last = df.iloc[-1]
            open_trade.exit_idx = n - 1
            open_trade.exit_time = last["time"]
            open_trade.exit_price = last["close"]
            open_trade.outcome = "TIMEOUT"
            open_trade.still_open = True
            self._finalize_pnl(open_trade)
            trades.append(open_trade)

        return trades

    def _check_exit(self, trade: Trade, i: int, bar: pd.Series) -> Trade:
        """
        Check whether bar i's high/low touched SL or TP. If both were
        touched within the same bar, we can't know which happened first
        without tick data — resolve conservatively by assuming SL hit
        first (never assume the best case).
        """
        hi, lo = bar["high"], bar["low"]

        if trade.direction == "BUY":
            hit_tp = hi >= trade.tp
            hit_sl = lo <= trade.sl
        else:
            hit_tp = lo <= trade.tp
            hit_sl = hi >= trade.sl

        # FIX (important — cost realism): the module docstring claims costs
        # are modeled "on every fill (both legs)", but only entry ever got
        # spread+slippage applied — SL/TP exits filled at the exact stop/
        # target price, i.e. zero exit slippage. Real stop-outs, especially
        # during the fast moves that actually trigger a Fib-invalidation
        # SL, routinely fill WORSE than the stated level (the broker can't
        # guarantee your exact stop price in a gapping/fast market). TP
        # fills are left exact (limit-style — those DO fill at-or-better,
        # so no adverse adjustment is applied there). This makes reported
        # losses slightly larger (more realistic) without touching wins.
        sl_slip = self.cfg.slippage_pips * self._pip_size
        if trade.direction == "BUY":
            adverse_sl_price = trade.sl - sl_slip
        else:
            adverse_sl_price = trade.sl + sl_slip

        if hit_sl and hit_tp:
            trade.outcome, trade.exit_price = "LOSS", adverse_sl_price  # conservative
        elif hit_sl:
            trade.outcome, trade.exit_price = "LOSS", adverse_sl_price
        elif hit_tp:
            trade.outcome, trade.exit_price = "WIN", trade.tp

        if trade.outcome is not None:
            trade.exit_idx = i
            trade.exit_time = bar["time"]
            self._finalize_pnl(trade)

        return trade

    def _finalize_pnl(self, trade: Trade) -> None:
        sign = 1 if trade.direction == "BUY" else -1
        price_delta = (trade.exit_price - trade.entry_price) * sign
        trade.pips = round(price_delta / self._pip_size, 1)

        risk = abs(trade.entry_price - trade.sl)
        trade.r_multiple = round(price_delta / risk, 3) if risk > 1e-9 else 0.0
        if trade.outcome == "TIMEOUT" and trade.pips > 0:
            trade.outcome = "WIN"
        elif trade.outcome == "TIMEOUT":
            trade.outcome = "LOSS"


# ═══════════════════════════════════════════════════════════════
# REPORTING (single symbol)
# ═══════════════════════════════════════════════════════════════

def _feature_breakdown(df: pd.DataFrame, col: str, label: str) -> list[str]:
    lines = [f"  ── By {label} ──"]
    for name, grp in df.groupby(col, dropna=False):
        wr = (grp["outcome"] == "WIN").mean() * 100
        lines.append(f"    {str(name):<20} n={len(grp):<4} win_rate={wr:.1f}%  "
                      f"avg_R={grp['r_multiple'].mean():.2f}  net_pips={grp['pips'].sum():.1f}")
    lines.append("")
    return lines


def build_report(trades: list[Trade], cfg: BacktestConfig) -> str:
    if not trades:
        return "No trades were generated — try lowering min_confidence or increasing bars."

    df = pd.DataFrame([t.__dict__ for t in trades])
    wins = df[df["outcome"] == "WIN"]
    losses = df[df["outcome"] == "LOSS"]

    win_rate = len(wins) / len(df) * 100
    gross_win = wins["pips"].sum()
    gross_loss = abs(losses["pips"].sum())
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")
    expectancy_r = round(df["r_multiple"].mean(), 3)
    net_pips = round(df["pips"].sum(), 1)

    equity_r = df["r_multiple"].cumsum()
    running_max = equity_r.cummax()
    max_dd_r = round((equity_r - running_max).min(), 2)

    lines = []
    lines.append("=" * 60)
    lines.append(f"  FIBONACCI ENGINE BACKTEST — {cfg.symbol} {cfg.timeframe}")
    lines.append("=" * 60)
    lines.append(f"  Bars tested        : {cfg.bars}")
    lines.append(f"  Cost per trade      : {cfg.spread_pips + cfg.slippage_pips} pips "
                 f"(spread {cfg.spread_pips} + slippage {cfg.slippage_pips})")
    lines.append(f"  Min confidence      : {cfg.min_confidence}%")
    lines.append(f"  HTF confluence      : "
                 f"{'ON (' + str(cfg.htf_timeframe or resolve_htf(cfg.timeframe, None)) + ')' if cfg.use_htf_confluence else 'OFF'}")
    lines.append("")
    lines.append(f"  Total trades        : {len(df)}")
    lines.append(f"  Wins / Losses       : {len(wins)} / {len(losses)}")
    lines.append(f"  Win rate            : {win_rate:.1f}%")
    lines.append(f"  Profit factor       : {profit_factor}")
    lines.append(f"  Net pips            : {net_pips}")
    lines.append(f"  Expectancy (R/trade): {expectancy_r}")
    lines.append(f"  Max drawdown (R)    : {max_dd_r}")
    lines.append("")

    lines += _feature_breakdown(df, "trend", "trend")

    lines.append("  ── By confidence bucket ──")
    df["conf_bucket"] = pd.cut(df["confidence"], bins=[0, 60, 70, 80, 100],
                                labels=["55-60", "60-70", "70-80", "80+"])
    for bucket, grp in df.groupby("conf_bucket", observed=True):
        if len(grp) == 0:
            continue
        wr = (grp["outcome"] == "WIN").mean() * 100
        lines.append(f"    {str(bucket):<8} n={len(grp):<4} win_rate={wr:.1f}%  "
                      f"avg_R={grp['r_multiple'].mean():.2f}")
    lines.append("")

    lines += _feature_breakdown(df, "zone", "Fib zone")
    lines += _feature_breakdown(df, "strategy_type", "strategy type")
    lines += _feature_breakdown(df, "trigger_pattern", "trigger pattern")
    # NEW — previously-uncaptured features, now broken out same as the rest:
    lines += _feature_breakdown(df, "htf_confirmed", "HTF confirmation (multi-timeframe agreement)")
    lines += _feature_breakdown(df, "in_time_cluster", "Fib time-cluster (bar landed on a time projection)")
    lines += _feature_breakdown(df, "price_time_cluster", "price+time cluster (strongest combo per engine)")
    lines += _feature_breakdown(df, "secondary_swing_used", "secondary swing / Fib cluster leg present")

    lines.append("=" * 60)
    lines.append("  CAVEAT: this is backtest performance only. Expect meaningfully")
    lines.append("  worse live results — real spread/slippage vary with volatility")
    lines.append("  and news (this uses a fixed assumption), execution isn't")
    lines.append("  instant, and market regime will differ from this sample.")
    lines.append("  This is not investment advice or a signal to trade live.")
    lines.append("=" * 60)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# NEW — MULTI-PAIR "RECENT SIGNALS" SCAN
# ═══════════════════════════════════════════════════════════════
# Answers exactly: "across all pairs, what signal fired in the last N
# days, which ones would have won vs lost, and which feature (zone /
# strategy / HTF confirmation / time-cluster / confidence) tends to sit
# on the winning side."
#
# Mechanically this is the SAME walk-forward engine above (so results are
# still no-look-ahead, next-bar-open fills, etc.) — it just runs it once
# per symbol and keeps only the tail of trades whose ENTRY fell inside the
# requested recent window.

def scan_recent_pairs(
    symbols: list[str],
    timeframe: str,
    recent_days: int,
    base_cfg: BacktestConfig,
) -> list[Trade]:
    """
    Run the backtest for every symbol, keep only trades entered within the
    last `recent_days` days of EACH symbol's own fetched data (so a symbol
    with a data gap doesn't get penalized against another's clock), and
    return the combined trade list (each trade still tagged with .symbol
    via a dynamically-added attribute — see below).
    """
    minutes_per_bar = _TF_MINUTES.get(timeframe, 60)
    bars_per_day = max(1, (24 * 60) // minutes_per_bar)
    warmup_bars = TF_SWING_WINDOW.get(timeframe, DEFAULT_SWING_WINDOW) * 3 + 5
    needed_bars = max(base_cfg.bars,
                       warmup_bars + bars_per_day * recent_days + base_cfg.max_holding_bars + 20)

    htf_tf = resolve_htf(timeframe, base_cfg.htf_timeframe) if base_cfg.use_htf_confluence else None

    all_trades: list[Trade] = []
    for sym in symbols:
        try:
            df = fetch_closed_bars(sym, timeframe, needed_bars)
        except (RuntimeError, ValueError) as e:
            log.warning("Skipping %s: %s", sym, e)
            continue
        if len(df) < warmup_bars + 5:
            log.warning("Skipping %s: not enough history returned (%d bars)", sym, len(df))
            continue

        htf_df = None
        if htf_tf:
            try:
                htf_df = fetch_closed_bars(sym, htf_tf, htf_bars_needed(needed_bars, timeframe, htf_tf))
            except (RuntimeError, ValueError) as e:
                log.warning("%s: HTF fetch failed (%s) — continuing without multi-timeframe confluence: %s",
                            sym, htf_tf, e)

        cfg = replace(base_cfg, symbol=sym, timeframe=timeframe, bars=needed_bars)
        backtester = FibonacciBacktester(cfg, htf_df=htf_df, htf_timeframe=htf_tf if htf_df is not None else None)
        trades = backtester.run(df)

        cutoff = df["time"].iloc[-1] - pd.Timedelta(days=recent_days)
        recent = [t for t in trades if t.entry_time >= cutoff]
        for t in recent:
            t.symbol = sym  # dynamic attribute, only used for this report
        all_trades.extend(recent)
        log.info("%s: %d trade(s) in the last %d day(s)", sym, len(recent), recent_days)

    return all_trades


def build_recent_scan_report(trades: list[Trade], recent_days: int, timeframe: str) -> str:
    if not trades:
        return (f"No signals fired on any scanned pair in the last {recent_days} day(s) "
                f"at {timeframe} — try more --recent-days, a lower --min-confidence, or check "
                f"config.SYMBOLS / --symbols-file.")

    df = pd.DataFrame([{**t.__dict__, "symbol": getattr(t, "symbol", "?")} for t in trades])
    df = df.sort_values(["symbol", "entry_time"])

    lines = []
    lines.append("=" * 72)
    lines.append(f"  MULTI-PAIR SIGNAL SCAN — last {recent_days} day(s) @ {timeframe}")
    lines.append("=" * 72)
    lines.append(f"  Pairs with a signal : {df['symbol'].nunique()}")
    lines.append(f"  Total signals       : {len(df)}")
    win_rate = (df["outcome"] == "WIN").mean() * 100
    lines.append(f"  Overall win rate    : {win_rate:.1f}%  "
                 f"({(df['outcome']=='WIN').sum()}W / {(df['outcome']=='LOSS').sum()}L)")
    lines.append(f"  Net pips (all pairs): {df['pips'].sum():.1f}")
    lines.append("")

    # ── per pair, per signal detail ──
    lines.append("  ── Signals by pair (chronological) ──")
    for sym, grp in df.groupby("symbol"):
        wr = (grp["outcome"] == "WIN").mean() * 100
        lines.append(f"\n  {sym}  ({len(grp)} signal(s), win_rate={wr:.1f}%)")
        for _, r in grp.iterrows():
            open_tag = "  [STILL OPEN — mark-to-market]" if r["still_open"] else ""
            feat_tags = []
            if r["htf_confirmed"]:
                feat_tags.append("HTF")
            if r["in_time_cluster"]:
                feat_tags.append("time-cluster")
            if r["price_time_cluster"]:
                feat_tags.append("price+time")
            if r["secondary_swing_used"]:
                feat_tags.append("cluster-leg")
            feat_str = f"  [{', '.join(feat_tags)}]" if feat_tags else ""
            lines.append(
                f"    {str(r['entry_time'])[:16]}  {r['direction']:<4} "
                f"{r['zone']:<20} {r['strategy_type']:<20} conf={r['confidence']:<3} "
                f"-> {r['outcome']:<5} {r['pips']:>7.1f}p  R={r['r_multiple']:>5.2f}"
                f"{feat_str}{open_tag}"
            )
    lines.append("")

    # ── which feature wins vs loses, aggregated across ALL pairs ──
    lines.append("  ── Which feature is winning vs losing (all pairs combined) ──")
    lines += _feature_breakdown(df, "zone", "Fib zone")
    lines += _feature_breakdown(df, "strategy_type", "strategy type")
    lines += _feature_breakdown(df, "direction", "direction")
    df["conf_bucket"] = pd.cut(df["confidence"], bins=[0, 60, 70, 80, 100],
                                labels=["55-60", "60-70", "70-80", "80+"])
    lines += _feature_breakdown(df, "conf_bucket", "confidence bucket")
    lines += _feature_breakdown(df, "htf_confirmed", "HTF confirmation")
    lines += _feature_breakdown(df, "in_time_cluster", "Fib time-cluster")
    lines += _feature_breakdown(df, "secondary_swing_used", "secondary swing / cluster leg")

    lines.append("=" * 72)
    lines.append("  CAVEAT: small-sample warning — a few days across N pairs is a thin")
    lines.append("  sample per feature; treat win-rate splits here as a lead to dig")
    lines.append("  into with a longer --bars backtest per pair, not a final verdict.")
    lines.append("  Backtest performance only — not investment advice.")
    lines.append("=" * 72)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward MT5 backtest for FibonacciEngine")
    parser.add_argument("--symbol", default=None, help="Exact broker symbol, e.g. EURUSD "
                         "(required unless --scan-all-pairs)")
    parser.add_argument("--timeframe", default="H1", choices=sorted(_TF_MAP))
    parser.add_argument("--bars", type=int, default=3000)
    parser.add_argument("--spread-pips", type=float, default=1.2)
    parser.add_argument("--slippage-pips", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=int, default=55)
    parser.add_argument("--max-holding-bars", type=int, default=48)
    parser.add_argument("--min-rr", type=float, default=1.5,
                         help="Minimum reward:risk required to take a trade")
    parser.add_argument("--lookback-bars", type=int, default=200,
                         help="Fixed-size window handed to the engine at each step — MUST match "
                              "whatever `limit=N` your live fetcher uses, or backtest results won't "
                              "reflect what the live bot actually sees (default 200).")
    parser.add_argument("--no-trigger-candle", action="store_true",
                         help="Disable the confirmation-candle requirement (fire on zone-touch alone — "
                              "not recommended, kept for A/B comparison against the gated version)")
    parser.add_argument("--no-htf-confluence", action="store_true",
                         help="Disable auto multi-timeframe confluence (HTF grid via HTF_HIERARCHY)")
    parser.add_argument("--htf-timeframe", default=None, choices=sorted(_TF_MAP),
                         help="Override the auto-picked higher timeframe for confluence")
    parser.add_argument("--csv-out", default=None, help="Optional path to dump per-trade CSV")

    # NEW — multi-pair recent-signal scan
    parser.add_argument("--scan-all-pairs", action="store_true",
                         help="Scan every symbol in config.SYMBOLS (or --symbols-file) instead of "
                              "a single --symbol, and report only signals from the last --recent-days.")
    parser.add_argument("--recent-days", type=int, default=5,
                         help="Used with --scan-all-pairs: how many days back to report signals for.")
    parser.add_argument("--symbols-file", default=None,
                         help="Optional text file, one symbol per line, overrides config.SYMBOLS.")

    args = parser.parse_args()

    if not args.scan_all_pairs and not args.symbol:
        parser.error("--symbol is required unless --scan-all-pairs is given")

    base_cfg = BacktestConfig(
        symbol=args.symbol or "<multi>",
        timeframe=args.timeframe,
        bars=args.bars,
        spread_pips=args.spread_pips,
        slippage_pips=args.slippage_pips,
        min_confidence=args.min_confidence,
        max_holding_bars=args.max_holding_bars,
        min_rr=args.min_rr,
        require_trigger_candle=not args.no_trigger_candle,
        use_htf_confluence=not args.no_htf_confluence,
        htf_timeframe=args.htf_timeframe,
        lookback_bars=args.lookback_bars,
    )

    if args.scan_all_pairs:
        symbols = CONFIG_SYMBOLS
        if args.symbols_file:
            with open(args.symbols_file) as f:
                symbols = [s.strip() for s in f if s.strip()]
        if not symbols:
            log.error("No symbols to scan — pass --symbols-file, or make sure config.SYMBOLS "
                      "is importable from this working directory.")
            sys.exit(1)

        try:
            connect_mt5()
            trades = scan_recent_pairs(symbols, args.timeframe, args.recent_days, base_cfg)
        except RuntimeError as e:
            log.error("Scan aborted: %s", e)
            sys.exit(1)
        finally:
            if mt5 is not None:
                mt5.shutdown()

        report = build_recent_scan_report(trades, args.recent_days, args.timeframe)
        print(report)

        if args.csv_out and trades:
            pd.DataFrame([{**t.__dict__, "symbol": getattr(t, "symbol", "?")} for t in trades]) \
                .to_csv(args.csv_out, index=False)
            log.info("Per-trade detail written to %s", args.csv_out)
        return

    # ── single-symbol backtest (original behavior, now HTF-aware) ──
    htf_tf = resolve_htf(args.timeframe, args.htf_timeframe) if base_cfg.use_htf_confluence else None
    try:
        connect_mt5()
        df = fetch_closed_bars(base_cfg.symbol, base_cfg.timeframe, base_cfg.bars)
        htf_df = None
        if htf_tf:
            try:
                htf_df = fetch_closed_bars(base_cfg.symbol, htf_tf,
                                            htf_bars_needed(base_cfg.bars, base_cfg.timeframe, htf_tf))
            except (RuntimeError, ValueError) as e:
                log.warning("HTF fetch failed (%s) — continuing without multi-timeframe confluence: %s",
                            htf_tf, e)
                htf_tf = None
    except RuntimeError as e:
        log.error("Backtest aborted: %s", e)
        sys.exit(1)
    finally:
        if mt5 is not None:
            mt5.shutdown()

    backtester = FibonacciBacktester(base_cfg, htf_df=htf_df, htf_timeframe=htf_tf)
    trades = backtester.run(df)
    report = build_report(trades, base_cfg)
    print(report)

    if args.csv_out and trades:
        pd.DataFrame([t.__dict__ for t in trades]).to_csv(args.csv_out, index=False)
        log.info("Per-trade detail written to %s", args.csv_out)


if __name__ == "__main__":
    main()