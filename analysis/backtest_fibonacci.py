# analysis/backtest_fibonacci.py
# ============================================================
# Walk-forward backtester for FibonacciEngine, driven off MT5 data.
#
# Design goals (institutional review checklist):
#   - No look-ahead: at bar i, the engine only ever sees df.iloc[:i+1]
#     (bars 0..i, all closed). Entry fills happen at bar i+1's OPEN,
#     never at bar i's close — you can't trade a signal at the same
#     price you used to generate it.
#   - Spread + slippage modeled on every fill (both legs), not just
#     the tightest historical spread.
#   - SL/TP checked against each subsequent bar's high/low, not close,
#     so intrabar stop-outs aren't missed.
#   - Ambiguous same-bar SL+TP touch resolved conservatively (SL wins),
#     since we don't have tick data to know which came first.
#   - Reports trades AND signal quality (confidence buckets, zone,
#     trend) so you can see not just "did it make money" but "which
#     signals were actually good."
#
# Day 41 — COMPLETED (was Fib-only, did not match live signal path):
#   - Wires in the same Indicators + SupportResistance engines the live
#     pipeline uses (fibonacci.py's own __main__ quick-run does this),
#     rebuilt per-bar from a bounded trailing window so entries actually
#     get confluence-scored the way they do live, not just bare Fib math.
#     Graceful fallback to the old Fib-only path if those modules aren't
#     importable, or per-bar if they raise — a run never crashes because
#     of this, it just quietly loses confluence for that bar/run.
#   - Bounded context_window_bars (default 300) instead of the entire
#     0..i history, for both performance (was O(n^2) over a full run) and
#     fidelity (live fetches a bounded lookback, not the whole history).
#   - Fixed an inconsistency where a force-TIMEOUT close allowed same-bar
#     re-entry while an SL/TP close explicitly didn't — both now defer to
#     the next bar, matching the "don't reopen same bar we just closed"
#     rule the SL/TP path already stated.
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
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

try:
    from utils.logger import get_logger
    log = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger(__name__)

try:
    from analysis.fibonacci import FibonacciEngine
except ImportError:
    # Allow running this file standalone (e.g. next to fibonacci.py
    # instead of inside the full analysis/ package) without breaking
    # the normal project import path.
    sys.path.insert(0, ".")
    from fibonacci import FibonacciEngine

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # only fail loudly when connect() is actually called

# Day 41 — FIX (institutional review): the previous version of this backtester
# called self.engine.analyze(window) with NO sr_ctx / ind_ctx, meaning every
# backtested signal was Fib-only — no S/R confluence, no indicator-based
# ATR/context, no confluence-driven confidence boost. Live trading (see
# fibonacci.py's own __main__ quick-run block) always builds sr_ctx/ind_ctx
# and passes them in. A backtest that never sees confluence is not testing
# the strategy that actually runs live — it's testing a strictly weaker one,
# and any win-rate/expectancy numbers from the old version are not
# representative. This import wires in the same two engines fibonacci.py's
# own quick-run uses, with a graceful fallback (old Fib-only behavior) if
# they aren't importable in this environment, so the file still runs
# standalone next to fibonacci.py alone if that's all you have.
try:
    from data.indicators import Indicators
    from analysis.support_resistance import SupportResistance
    _CONTEXT_ENGINES_AVAILABLE = True
except ImportError:
    Indicators = None
    SupportResistance = None
    _CONTEXT_ENGINES_AVAILABLE = False

# NEW (this pass) — same graceful-fallback pattern as the context engines
# above: FibonacciEngine now accepts an optional liquidity_ctx (Day 63,
# see fibonacci.py's __init__ docstring). This backtester needs to build
# and pass one to actually exercise require_liquidity_alignment /
# min_liquidity_score, or those flags are silent no-ops here even when
# the live pipeline has liquidity_ctx wired up.
try:
    from analysis.liquidity_engine import LiquidityEngine
    _LIQUIDITY_ENGINE_AVAILABLE = True
except ImportError:
    LiquidityEngine = None
    _LIQUIDITY_ENGINE_AVAILABLE = False


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

# ── Day 42: timeframe -> the next timeframe UP, for htf_ctx ────────────
# One rung up the standard MT5 ladder. Deliberately doesn't map "1d"/"D1" —
# there's no sensible "higher" timeframe to auto-derive for a daily chart
# in this backtester; require_htf_alignment is a no-op there (logged).
_HTF_MAP = {
    "1m": "5m", "M1": "5m",
    "5m": "15m", "M5": "15m",
    "15m": "1h", "M15": "1h",
    "30m": "4h", "M30": "4h",
    "1h": "4h", "H1": "4h",
    "4h": "1d", "H4": "1d",
}

# pandas resample rule for each timeframe key this module works with.
_RESAMPLE_RULE = {
    "1m": "1min", "M1": "1min",
    "5m": "5min", "M5": "5min",
    "15m": "15min", "M15": "15min",
    "30m": "30min", "M30": "30min",
    "1h": "1h", "H1": "1h",
    "4h": "4h", "H4": "4h",
    "1d": "1D", "D1": "1D",
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

    # Day 41 additions — see import-block comment above for why these exist.
    use_context_engines: bool = True
    # context_window_bars : how many trailing CLOSED bars are handed to the
    #     Indicators/SupportResistance/Fib engines at each step, instead of
    #     the ever-growing df.iloc[:i+1]. Two reasons this matters, not one:
    #       1. Performance — recomputing S/R + every indicator over an
    #          ever-growing window is O(n^2) across a 3000-bar run.
    #       2. Fidelity — live main.py fetches a BOUNDED lookback (e.g.
    #          limit=200 in fibonacci.py's own quick-run example), not the
    #          bot's entire trading history. Feeding the engine unbounded
    #          history here would test a version of the strategy that never
    #          actually runs live (more S/R/trend context than it has in
    #          production). Default 300 comfortably covers swing_window*3
    #          for every timeframe in TF_SWING_WINDOW plus headroom for
    #          slower indicators (e.g. EMA200/ATR14).
    context_window_bars: int = 300

    # Day 42 additions — mirror FibonacciEngine's new opt-in params 1:1 so
    # this backtester can actually exercise them (see fibonacci.py's
    # __init__ docstring for full rationale on each).
    require_htf_alignment: bool = False
    min_confluence_strength: int = 0
    # UPDATED (this pass): these two used to default False here, matching
    # FibonacciEngine's OLD defaults. fibonacci.py's own defaults have since
    # been flipped to True (5yr EURUSD H1 walk-forward validation — see its
    # __init__ docstring: require_engulfing_only PF 0.93->1.21, allow_golden_zone
    # +cluster PF 1.21->1.22). This dataclass was passing its OWN default
    # explicitly into FibonacciEngine's constructor either way (see __init__
    # below), so leaving these False HERE was silently overriding the
    # engine's validated default back to the old, worse-performing behavior
    # — a real bug, not a style nit. Now matches.
    require_engulfing_only: bool = True
    require_trend_ma: bool = False
    trend_ma_period: int = 200
    min_adx: float = 0.0
    max_atr_multiple: float = 0.0
    allow_golden_zone: bool = True
    require_cluster_for_golden: bool = True
    allow_cluster_touch_entry: bool = False
    min_rr_dynamic: bool = False
    max_swing_age_bars: int = 0
    require_volume_confirmation: bool = False
    volume_multiple: float = 1.5

    # NEW (this pass) — mirrors FibonacciEngine's Day-63 liquidity params
    # 1:1, same reasoning as every other Day-42 param mirrored above: this
    # backtester needs to be able to actually exercise what the engine
    # supports, or its numbers stop being representative of live behavior.
    # Both default False/0 (off) — see fibonacci.py's __init__ docstring
    # for the evidence behind that default (n=32 EURUSD H1: soft veto
    # dropped 1 trade, WR 34.4%->35.5%, PF 1.22->1.28 — real but too thin
    # to enable by default yet).
    require_liquidity_alignment: bool = False
    min_liquidity_score: int = 0

    # Day 42 — trade-management additions (separate from the signal-gating
    # params above; these change how an OPEN trade is managed, not whether
    # one opens). Both default False/no-op — exact prior all-in/all-out
    # behavior is unchanged unless explicitly enabled.
    use_partial_tp: bool = False
    # partial_tp_fraction : fraction of position size closed at tp1 when
    #     use_partial_tp is True; the remainder runs to tp2 (or is force-
    #     closed at max_holding_bars / SL as usual). Requires the engine to
    #     have supplied tp2 (see fibonacci.py) — if tp2 is None for a given
    #     trade, this silently falls back to all-in/all-out at tp1 for that
    #     trade only, rather than skipping it or erroring.
    partial_tp_fraction: float = 0.5
    use_breakeven_stop: bool = False
    # breakeven_trigger_r : move SL to entry price once price has moved
    #     this many R in the trade's favor (checked independently of
    #     use_partial_tp — either can be enabled alone).
    breakeven_trigger_r: float = 1.0


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
    exit_idx: Optional[int] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None   # 'WIN' / 'LOSS' / 'TIMEOUT'
    pips: float = 0.0
    r_multiple: float = 0.0

    # Day 42 — scaled-exit / breakeven-stop bookkeeping (see
    # BacktestConfig.use_partial_tp / use_breakeven_stop). All inert
    # (fields unused) unless those flags are on, so a plain Trade is
    # exactly as before when neither is enabled.
    tp2: Optional[float] = None
    original_sl: Optional[float] = None   # kept for R-multiple math even
                                           # after sl gets moved to breakeven
    filled_pct: float = 1.0               # 1.0 = full size still open
    partial_tp_hit: bool = False
    partial_realized_pips: float = 0.0    # weighted PnL banked from the
                                           # partial close, folded into the
                                           # final pips/r_multiple at exit
    breakeven_moved: bool = False


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


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

class FibonacciBacktester:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.engine = FibonacciEngine(
            timeframe=cfg.timeframe, symbol=cfg.symbol,
            require_trigger_candle=cfg.require_trigger_candle,
            min_rr=cfg.min_rr,
            require_htf_alignment=cfg.require_htf_alignment,
            min_confluence_strength=cfg.min_confluence_strength,
            require_engulfing_only=cfg.require_engulfing_only,
            require_trend_ma=cfg.require_trend_ma,
            trend_ma_period=cfg.trend_ma_period,
            min_adx=cfg.min_adx,
            max_atr_multiple=cfg.max_atr_multiple,
            allow_golden_zone=cfg.allow_golden_zone,
            require_cluster_for_golden=cfg.require_cluster_for_golden,
            allow_cluster_touch_entry=cfg.allow_cluster_touch_entry,
            min_rr_dynamic=cfg.min_rr_dynamic,
            max_swing_age_bars=cfg.max_swing_age_bars,
            require_volume_confirmation=cfg.require_volume_confirmation,
            volume_multiple=cfg.volume_multiple,
            require_liquidity_alignment=cfg.require_liquidity_alignment,
            min_liquidity_score=cfg.min_liquidity_score,
        )
        self._pip_size = self._resolve_pip_size(cfg.symbol)

        # Day 42 — HTF (higher-timeframe) context. Only built when the
        # caller actually wants alignment gated/scored (require_htf_alignment
        # True, or the engine will use it as a soft confidence bonus even
        # when False — see fibonacci.py's htf_ctx docstring), to avoid the
        # extra resample+engine cost on every bar for callers who don't
        # care. Uses its OWN FibonacciEngine instance on a HIGHER timeframe
        # (see _HTF_MAP) purely for trend direction via find_swing_points —
        # deliberately not a second full analyze() call, just the swing/trend
        # read, which is all htf_ctx needs.
        self._htf_timeframe = _HTF_MAP.get(cfg.timeframe)
        self._htf_engine = (
            FibonacciEngine(timeframe=self._htf_timeframe, symbol=cfg.symbol)
            if self._htf_timeframe else None
        )
        if cfg.require_htf_alignment and self._htf_engine is None:
            log.warning(
                "require_htf_alignment=True but no higher timeframe is mapped for '%s' "
                "(see _HTF_MAP) — htf_ctx will never be supplied, making this a no-op.",
                cfg.timeframe,
            )

        # Day 41 — see module-level import comment. use_context_engines is
        # requested AND the modules actually imported; if either is false we
        # fall back to Fib-only (old behavior) rather than crashing.
        self._use_context = cfg.use_context_engines and _CONTEXT_ENGINES_AVAILABLE
        if cfg.use_context_engines and not _CONTEXT_ENGINES_AVAILABLE:
            log.warning(
                "use_context_engines=True but data.indicators / "
                "analysis.support_resistance are not importable here — "
                "running Fib-only (no S/R confluence, no indicator context). "
                "Results will NOT match live signal quality/confidence. "
                "Run this from inside the project (analysis/, data/ on "
                "sys.path) to get a live-representative backtest."
            )
        self._ind = Indicators() if self._use_context else None
        self._sr = SupportResistance() if self._use_context else None

        # NEW (this pass) — only instantiate when actually needed (either
        # gate flag on), same lazy-cost reasoning as the HTF engine above.
        # LiquidityEngine itself needs an 'atr' column + DatetimeIndex,
        # which _build_liquidity_ctx() below constructs from ctx_df (the
        # already-indicator-enriched window from _build_context, so 'atr'
        # is already present when context engines are on) — falls back to
        # computing its own ATR if context engines are off.
        self._use_liquidity = (
            (cfg.require_liquidity_alignment or cfg.min_liquidity_score > 0)
            and _LIQUIDITY_ENGINE_AVAILABLE
        )
        if (cfg.require_liquidity_alignment or cfg.min_liquidity_score > 0) and not _LIQUIDITY_ENGINE_AVAILABLE:
            log.warning(
                "require_liquidity_alignment/min_liquidity_score set but "
                "analysis.liquidity_engine is not importable here — "
                "liquidity_ctx will never be supplied, making these a no-op. "
                "Run this from inside the project (analysis/ on sys.path, "
                "with liquidity_zones.py / liquidity_structure.py / "
                "session_analysis.py / stop_hunt_detector.py / "
                "fvg_detector.py all present) to exercise them."
            )
        self._liquidity_engine = LiquidityEngine() if self._use_liquidity else None

        # Day 41 — diagnostic: categorize WHY every evaluated bar didn't
        # produce a trade. Without this, "0 trades" and "0 trades because
        # every candidate failed the trigger-candle gate" look identical
        # from the outside — and they call for completely different next
        # steps (former: nothing wrong, market didn't offer setups;
        # latter: a gate is mis-tuned or broken). Populated in run().
        self.rejection_counts: Counter = Counter()

    @staticmethod
    def _resolve_pip_size(symbol: str) -> float:
        try:
            from utils.pip_utils import pip_size
            return pip_size(symbol)
        except ImportError:
            # Fallback only used if utils.pip_utils isn't importable in this
            # environment — JPY pairs use a 100x larger pip than other FX.
            return 0.01 if "JPY" in symbol.upper() else 0.0001

    def _build_context(self, window: pd.DataFrame) -> tuple[pd.DataFrame, Optional[dict], Optional[dict]]:
        """
        Day 41 — mirrors fibonacci.py's own __main__ quick-run pattern:
            df      = ind.add_all(df)
            ind_ctx = ind.get_ai_context(df)
            sr_res  = sr_eng.analyze(df)
            sr_ctx  = sr_eng.get_ai_context(sr_res)
        Returns (df_with_indicators, ind_ctx, sr_ctx). If context engines are
        disabled/unavailable, returns (window, None, None) unchanged — the
        Fib engine's analyze() already treats sr_ctx/ind_ctx=None as "no
        confluence sources beyond Fib itself", so this degrades gracefully
        rather than raising.

        Any exception from these third-party-to-this-file engines is caught
        and logged once per occurrence, then treated as "no context this
        bar" rather than aborting the whole backtest — a single bad bar of
        S/R/indicator data shouldn't kill a 3000-bar run, and the Fib engine
        alone is still a valid (if weaker) signal source.
        """
        if not self._use_context:
            return window, None, None
        try:
            df_ind = self._ind.add_all(window.copy())
            ind_ctx = self._ind.get_ai_context(df_ind)
            sr_res = self._sr.analyze(df_ind)
            sr_ctx = self._sr.get_ai_context(sr_res)
            return df_ind, ind_ctx, sr_ctx
        except Exception as e:
            log.warning("Context engines failed on this window (%s) — "
                        "falling back to Fib-only for this bar.", e)
            return window, None, None

    def _build_htf_ctx(self, window: pd.DataFrame) -> Optional[dict]:
        """
        Day 42 — was flagged as a known gap: "the backtester currently does
        not pass htf_ctx" even though FibonacciEngine supports it. Builds a
        higher-timeframe trend read by resampling this bounded trailing
        `window` (already restricted to closed bars 0..i, see run()'s
        invariant comment) up one rung on _HTF_MAP, then reusing
        find_swing_points() on the resampled bars — no separate
        pivot-detection logic to maintain.

        No-look-ahead note: the LAST resampled HTF bar is very likely built
        from an incomplete set of the lower-TF bars in `window` (e.g. only
        2 of 4 H1 bars available for the current H4 candle) — i.e. it's
        the HTF-equivalent of a still-forming bar. It's unconditionally
        dropped before trend detection, mirroring fetch_closed_bars()'s own
        "always drop the forming bar" rule at the base timeframe.

        Returns {'trend': 'BULLISH'|'BEARISH'} or None (not enough data /
        no swing found / resample failed) — None is a legitimate, expected
        outcome early in a run and is handled by FibonacciEngine as
        "no htf_ctx supplied", not an error.
        """
        if self._htf_engine is None or window is None or len(window) < 20:
            return None
        rule = _RESAMPLE_RULE.get(self.cfg.timeframe)
        if rule is None:
            return None
        try:
            htf = (
                window.set_index("time")
                      .resample(rule)
                      .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
                      .dropna()
            )
            htf = htf.iloc[:-1]  # drop the likely-still-forming HTF bar
            if len(htf) < self._htf_engine.swing_window * 3:
                return None
            htf = htf.reset_index()
            swings = self._htf_engine.find_swing_points(htf)
            if not swings.get("valid"):
                return None
            return {"trend": swings["trend"]}
        except Exception as e:
            log.warning("HTF context build failed (%s) — proceeding without htf_ctx for this bar.", e)
            return None

    def _build_liquidity_ctx(self, ctx_df: pd.DataFrame) -> Optional[dict]:
        """
        NEW (this pass) — mirrors _build_htf_ctx's structure: build the
        optional context dict fibonacci.py's analyze(liquidity_ctx=...)
        accepts, purely additive (returns None -> engine behaves exactly
        as if this parameter didn't exist).

        LiquidityEngine.analyze() requires a DatetimeIndex and an 'atr'
        column. `ctx_df` already has 'atr' when context engines are on
        (Indicators.add_all() adds it); if context engines are off, we
        compute a minimal ATR here rather than skipping liquidity
        entirely, so require_liquidity_alignment still works standalone
        with --no-context-engines.

        No-look-ahead: `ctx_df` is already bounded to bars 0..i (see run()'s
        invariant comment) — same window the Fib engine itself just saw.
        """
        if not self._use_liquidity or ctx_df is None or len(ctx_df) < 20:
            return None
        try:
            work = ctx_df.copy()
            if "atr" not in work.columns:
                h, l, c = work["high"].values, work["low"].values, work["close"].values
                tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
                atr_series = pd.Series(tr).rolling(14).mean()
                work = work.iloc[1:].reset_index(drop=True)
                work["atr"] = atr_series.values
            work = work.set_index("time")
            result = self._liquidity_engine.analyze(work, symbol=self.cfg.symbol)
            if not result or result.get("bias") is None:
                return None
            return {"bias": result["bias"], "score": result.get("score")}
        except Exception as e:
            log.warning("Liquidity context build failed (%s) — proceeding without liquidity_ctx for this bar.", e)
            return None

    def run(self, df: pd.DataFrame) -> list[Trade]:
        trades: list[Trade] = []
        open_trade: Optional[Trade] = None

        warmup = self.engine.swing_window * 3 + 5
        n = len(df)

        for i in range(warmup, n - 1):
            # Invariant preserved throughout this loop: only bars 0..i (all
            # CLOSED) ever inform a decision made "at" bar i. The bounded
            # context window built below (Day 41) is a sub-slice of that —
            # still 0..i, just capped in length — so this invariant holds.

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
                # Day 41 — FIX: this branch used to fall straight through
                # into signal evaluation below, opening a NEW trade on the
                # SAME bar a position was just force-closed on. The SL/TP
                # exit path above explicitly `continue`s to avoid exactly
                # this (see its comment); the timeout path was missing the
                # same guard, so timeout-closed trades got a free same-bar
                # re-entry that SL/TP-closed trades never got — an
                # inconsistency that biases the timeout bucket's stats.
                continue

            if open_trade is not None:
                continue  # still in a position, don't evaluate new entries

            # ── evaluate a fresh signal off bars 0..i, bounded to a ──
            # ── live-representative trailing window (Day 41)        ──
            ctx_start = max(0, i + 1 - self.cfg.context_window_bars)
            ctx_window = df.iloc[ctx_start: i + 1]
            ctx_df, ind_ctx, sr_ctx = self._build_context(ctx_window)
            htf_ctx = self._build_htf_ctx(ctx_window)
            liquidity_ctx = self._build_liquidity_ctx(ctx_df)
            result = self.engine.analyze(ctx_df, sr_ctx=sr_ctx, ind_ctx=ind_ctx, htf_ctx=htf_ctx,
                                          liquidity_ctx=liquidity_ctx)
            signal = result.get("signal", {})
            bias = signal.get("bias", "WAIT")
            conf = signal.get("confidence", 0)
            candidate_bias = signal.get("candidate_bias", "WAIT")

            # Day 41 — diagnostic: bucket EVERY non-trade bar by why it
            # didn't trade, using the same gate order fibonacci.py itself
            # applies (see _generate_signal). This turns "0 trades" from
            # a mystery into a number per gate — e.g. 2800 bars rejected
            # at "fib_position_wait" (no valid setup at all — market
            # never pulled back into the zone) vs 3 rejected at
            # "trigger_candle" tells you completely different things
            # about whether this is a market-condition result or a
            # mis-tuned/broken gate.
            if bias not in ("BUY", "SELL"):
                reason_txt = signal.get("reason", "")
                if candidate_bias not in ("BUY", "SELL"):
                    self.rejection_counts["fib_position_wait"] += 1
                elif "Golden zone" in reason_txt:
                    self.rejection_counts["golden_zone_gate"] += 1
                elif "Confluence strength" in reason_txt:
                    self.rejection_counts["confluence_strength_gate"] += 1
                elif "trend MA filter" in reason_txt:
                    self.rejection_counts["trend_ma_gate"] += 1
                elif "ADX" in reason_txt:
                    self.rejection_counts["adx_gate"] += 1
                elif "volatility too high" in reason_txt:
                    self.rejection_counts["volatility_gate"] += 1
                elif "confirmation candle" in reason_txt:
                    self.rejection_counts["trigger_candle_gate"] += 1
                elif "R:R" in reason_txt:
                    self.rejection_counts["min_rr_gate"] += 1
                elif "higher-timeframe" in reason_txt:
                    self.rejection_counts["htf_gate"] += 1
                else:
                    self.rejection_counts["other_wait"] += 1
            elif conf < self.cfg.min_confidence:
                self.rejection_counts["min_confidence_gate"] += 1
            else:
                self.rejection_counts["passed_all_gates"] += 1

            if bias not in ("BUY", "SELL") or conf < self.cfg.min_confidence:
                continue
            if signal.get("sl") is None or signal.get("tp1") is None:
                self.rejection_counts["incomplete_sl_tp"] += 1
                continue  # incomplete signal, skip rather than guess

            # Day 41 — defensive gate, independent of the fibonacci.py fix
            # for the flip-zone TP bug: never accept a signal whose SL/TP
            # aren't on the structurally correct side of the (pre-cost)
            # entry reference (curr_price). BUY must have tp > entry > sl;
            # SELL must have sl > entry > tp. This is a second line of
            # defense — if any future change to the engine (or a different
            # engine swapped in later) ever emits an inverted level again,
            # the backtest skips the trade instead of silently mislabeling
            # a loss as a WIN.
            ref_price = signal.get("entry", ctx_df["close"].iloc[-1])
            sl_val, tp_val = signal["sl"], signal["tp1"]
            if bias == "BUY" and not (tp_val > ref_price > sl_val):
                log.warning("Skipping inverted BUY signal at bar %d: sl=%s entry=%s tp=%s",
                            i, sl_val, ref_price, tp_val)
                continue
            if bias == "SELL" and not (sl_val > ref_price > tp_val):
                log.warning("Skipping inverted SELL signal at bar %d: sl=%s entry=%s tp=%s",
                            i, sl_val, ref_price, tp_val)
                continue

            # Entry fills at NEXT bar's open (i+1) — the earliest price
            # actually tradable after the decision made at close of bar i.
            next_bar = df.iloc[i + 1]
            raw_open = next_bar["open"]
            cost = (self.cfg.spread_pips + self.cfg.slippage_pips) * self._pip_size

            if bias == "BUY":
                entry_price = raw_open + cost
            else:
                entry_price = raw_open - cost

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
                confluence_strength=(result.get("confluence") or [{}])[0].get("strength", 0),
                strategy_type=signal.get("strategy_type", "UNKNOWN"),
                trigger_pattern=signal.get("trigger_pattern"),
                tp2=signal.get("tp2"),
                original_sl=signal["sl"],
            )

        # If a trade is still open at the end of data, close it at last close
        # (mark-to-market, not a real exit) so it's still counted in stats.
        if open_trade is not None:
            last = df.iloc[-1]
            open_trade.exit_idx = n - 1
            open_trade.exit_time = last["time"]
            open_trade.exit_price = last["close"]
            open_trade.outcome = "TIMEOUT"
            self._finalize_pnl(open_trade)
            trades.append(open_trade)

        return trades

    def _check_exit(self, trade: Trade, i: int, bar: pd.Series) -> Trade:
        """
        Check whether bar i's high/low touched SL or TP. If both were
        touched within the same bar, we can't know which happened first
        without tick data — resolve conservatively by assuming SL hit
        first (never assume the best case). This conservative SL-first
        rule now also takes priority over a would-be partial-TP promotion
        for the same reason: we don't know if the partial-TP level was
        actually touched before the stop within the bar.

        Day 42 — two opt-in trade-management additions layered on top of
        the original all-in/all-out check (see BacktestConfig docstrings):
          - use_breakeven_stop: move SL to entry once price has moved
            breakeven_trigger_r in the trade's favor.
          - use_partial_tp: on first touch of tp (still tp1 at that point),
            bank partial_tp_fraction of size, move SL to breakeven for the
            remainder, and re-target tp2 for the runner instead of closing
            the whole position. Falls back to a normal full close if tp2
            isn't available for this trade.
        """
        hi, lo = bar["high"], bar["low"]

        if self.cfg.use_breakeven_stop and not trade.breakeven_moved:
            orig_risk = abs(trade.entry_price - (trade.original_sl if trade.original_sl is not None else trade.sl))
            if orig_risk > 1e-9:
                trigger_dist = orig_risk * self.cfg.breakeven_trigger_r
                moved_favorably = (
                    (trade.direction == "BUY" and hi >= trade.entry_price + trigger_dist) or
                    (trade.direction == "SELL" and lo <= trade.entry_price - trigger_dist)
                )
                if moved_favorably:
                    trade.sl = trade.entry_price
                    trade.breakeven_moved = True

        if trade.direction == "BUY":
            hit_tp = hi >= trade.tp
            hit_sl = lo <= trade.sl
        else:
            hit_tp = lo <= trade.tp
            hit_sl = hi >= trade.sl

        if hit_sl:
            trade.outcome, trade.exit_price = "LOSS", trade.sl       # conservative
        elif (self.cfg.use_partial_tp and hit_tp and not trade.partial_tp_hit
                and trade.tp2 is not None):
            sign = 1 if trade.direction == "BUY" else -1
            partial_delta = (trade.tp - trade.entry_price) * sign
            trade.partial_realized_pips = round(
                partial_delta / self._pip_size * self.cfg.partial_tp_fraction, 2
            )
            trade.partial_tp_hit = True
            trade.filled_pct = round(1.0 - self.cfg.partial_tp_fraction, 4)
            trade.sl = trade.entry_price       # breakeven for the runner
            trade.breakeven_moved = True
            trade.tp = trade.tp2               # remaining size now targets tp2
            # Don't finalize — the runner keeps trading against the new
            # sl/tp starting next bar. Deliberately NOT re-checking tp2
            # against this same bar's high/low right after tp1 fired —
            # that would be an unrealistic same-bar double-target fill.
            return trade
        elif hit_tp:
            trade.outcome, trade.exit_price = "WIN", trade.tp

        if trade.outcome is not None:
            trade.exit_idx = i
            trade.exit_time = bar["time"]
            self._finalize_pnl(trade)

        return trade

    def _finalize_pnl(self, trade: Trade) -> None:
        sign = 1 if trade.direction == "BUY" else -1
        remaining_delta_pips = (trade.exit_price - trade.entry_price) * sign / self._pip_size
        # Day 42 — blend any partial-TP pips already banked with the
        # remainder's PnL at its own filled_pct weight (1.0 if partial
        # exit never engaged for this trade).
        trade.pips = round(trade.partial_realized_pips + remaining_delta_pips * trade.filled_pct, 1)

        # Risk for R-multiple math is always the ORIGINAL stop distance,
        # even if SL was later moved to breakeven — otherwise a
        # breakeven-moved trade's R would be computed off a near-zero
        # risk distance and be meaningless.
        risk_ref_sl = trade.original_sl if trade.original_sl is not None else trade.sl
        risk = abs(trade.entry_price - risk_ref_sl)
        trade.r_multiple = round((trade.pips * self._pip_size) / risk, 3) if risk > 1e-9 else 0.0

        if trade.outcome == "TIMEOUT" and trade.pips > 0:
            trade.outcome = "WIN"
        elif trade.outcome == "TIMEOUT":
            trade.outcome = "LOSS"
        elif trade.outcome == "LOSS" and trade.partial_tp_hit and trade.pips > 0:
            # Day 42 — a trade that banked a partial win at tp1 and then
            # got stopped out at BREAKEVEN on the runner nets a small win
            # overall, not a loss — relabel so win-rate reflects the whole
            # trade rather than just the runner leg's own outcome.
            trade.outcome = "WIN"


# ═══════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════

def build_report(trades: list[Trade], cfg: BacktestConfig, rejection_counts: Counter = None) -> str:
    if not trades:
        lines = ["No trades were generated — try lowering min_confidence or increasing bars."]
        if rejection_counts:
            total = sum(rejection_counts.values())
            lines.append("")
            lines.append(f"  ── Why no trades fired (Day 41 diagnostic, {total} bars evaluated) ──")
            for reason, count in rejection_counts.most_common():
                pct = count / total * 100 if total else 0
                lines.append(f"    {reason:<22} {count:>6}  ({pct:.1f}%)")
            lines.append("")
            if rejection_counts.get("fib_position_wait", 0) / max(total, 1) > 0.8:
                lines.append("  Dominant reason is 'fib_position_wait' — the Fib ratio itself never")
                lines.append("  sat in a valid zone (0.236-0.786) on a closed bar. This usually means")
                lines.append("  the market didn't offer meaningful pullbacks in this window (a strong")
                lines.append("  trend with shallow retracements), not a broken gate. Try a longer")
                lines.append("  --bars window or a different symbol/timeframe to confirm.")
            elif rejection_counts.get("trigger_candle_gate", 0) / max(total, 1) > 0.5:
                lines.append("  Dominant reason is 'trigger_candle_gate' — plenty of valid Fib setups,")
                lines.append("  but no confirmation candle on the closed bar. Try --no-trigger-candle")
                lines.append("  to A/B whether that gate specifically is costing you all your trades.")
            elif rejection_counts.get("min_rr_gate", 0) / max(total, 1) > 0.5:
                lines.append("  Dominant reason is 'min_rr_gate' — setups and triggers are firing,")
                lines.append("  but R:R never clears --min-rr. Try lowering it to see how many would")
                lines.append("  clear a lower bar (don't just lower it blindly and trade that live).")
        return "\n".join(lines)

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
    lines.append("")
    lines.append(f"  Total trades        : {len(df)}")
    lines.append(f"  Wins / Losses       : {len(wins)} / {len(losses)}")
    lines.append(f"  Win rate            : {win_rate:.1f}%")
    lines.append(f"  Profit factor       : {profit_factor}")
    lines.append(f"  Net pips            : {net_pips}")
    lines.append(f"  Expectancy (R/trade): {expectancy_r}")
    lines.append(f"  Max drawdown (R)    : {max_dd_r}")
    if cfg.use_partial_tp and "partial_tp_hit" in df.columns:
        n_partial = int(df["partial_tp_hit"].sum())
        lines.append(f"  Partial-TP hits     : {n_partial}/{len(df)} "
                     f"({n_partial / len(df) * 100:.1f}%) — fraction {cfg.partial_tp_fraction}, "
                     f"remainder to tp2")
    if cfg.use_breakeven_stop and "breakeven_moved" in df.columns:
        n_be = int(df["breakeven_moved"].sum())
        lines.append(f"  Breakeven-stop moves: {n_be}/{len(df)} ({n_be / len(df) * 100:.1f}%)")
    lines.append("")
    lines.append("  ── By trend ──")
    for trend_name, grp in df.groupby("trend"):
        wr = (grp["outcome"] == "WIN").mean() * 100
        lines.append(f"    {trend_name:<10} n={len(grp):<4} win_rate={wr:.1f}%  "
                      f"avg_R={grp['r_multiple'].mean():.2f}")
    lines.append("")
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
    lines.append("  ── By Fib zone ──")
    for zone_name, grp in df.groupby("zone"):
        wr = (grp["outcome"] == "WIN").mean() * 100
        lines.append(f"    {zone_name:<20} n={len(grp):<4} win_rate={wr:.1f}%  "
                      f"avg_R={grp['r_multiple'].mean():.2f}")
    lines.append("")
    lines.append("  ── By strategy type ──")
    for strat_name, grp in df.groupby("strategy_type"):
        wr = (grp["outcome"] == "WIN").mean() * 100
        lines.append(f"    {strat_name:<24} n={len(grp):<4} win_rate={wr:.1f}%  "
                      f"avg_R={grp['r_multiple'].mean():.2f}")
    lines.append("")
    lines.append("  ── By trigger pattern ──")
    for pattern_name, grp in df.groupby("trigger_pattern", dropna=False):
        wr = (grp["outcome"] == "WIN").mean() * 100
        lines.append(f"    {str(pattern_name):<20} n={len(grp):<4} win_rate={wr:.1f}%  "
                      f"avg_R={grp['r_multiple'].mean():.2f}")
    lines.append("=" * 60)
    if rejection_counts:
        total_evaluated = sum(rejection_counts.values())
        lines.append("")
        lines.append(f"  ── Signal funnel (Day 41 diagnostic, {total_evaluated} bars evaluated) ──")
        for reason, count in rejection_counts.most_common():
            pct = count / total_evaluated * 100 if total_evaluated else 0
            lines.append(f"    {reason:<22} {count:>6}  ({pct:.1f}%)")
    lines.append("  CAVEAT: this is backtest performance only. Expect meaningfully")
    lines.append("  worse live results — real spread/slippage vary with volatility")
    lines.append("  and news (this uses a fixed assumption), execution isn't")
    lines.append("  instant, and market regime will differ from this sample.")
    lines.append("  This is not investment advice or a signal to trade live.")
    lines.append("=" * 60)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward MT5 backtest for FibonacciEngine")
    parser.add_argument("--symbol", required=True, help="Exact broker symbol, e.g. EURUSD")
    parser.add_argument("--timeframe", default="H1", choices=sorted(_TF_MAP))
    parser.add_argument("--bars", type=int, default=3000)
    parser.add_argument("--spread-pips", type=float, default=1.2)
    parser.add_argument("--slippage-pips", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=int, default=55)
    parser.add_argument("--max-holding-bars", type=int, default=48)
    parser.add_argument("--min-rr", type=float, default=1.5,
                         help="Minimum reward:risk required to take a trade")
    parser.add_argument("--no-trigger-candle", action="store_true",
                         help="Disable the confirmation-candle requirement (fire on zone-touch alone — "
                              "not recommended, kept for A/B comparison against the gated version)")
    parser.add_argument("--no-context-engines", action="store_true",
                         help="Day 41: disable S/R + indicator context (Fib-only signals, old behavior). "
                              "Useful for A/B'ing how much confluence actually contributes.")
    parser.add_argument("--context-window-bars", type=int, default=300,
                         help="Day 41: trailing bars fed to the context/Fib engines each step "
                              "(bounded, live-representative — see BacktestConfig docstring)")

    # Day 42 — signal-gating additions (all opt-in, off by default; see
    # fibonacci.py's __init__ docstring for the backtest evidence behind each).
    parser.add_argument("--require-htf-alignment", action="store_true",
                         help="Hard-gate out signals that conflict with the higher-timeframe trend "
                              "(auto-derived via resample — see _HTF_MAP)")
    parser.add_argument("--min-confluence-strength", type=int, default=0,
                         help="Require top confluence-zone strength >= this (0 disables). Suggested: 60-75")
    parser.add_argument("--require-engulfing-only", action="store_true",
                         help="Only accept engulfing / liquidity_sweep / cluster_touch triggers "
                              "(rejects pin bars and momentum breakouts)")
    parser.add_argument("--require-trend-ma", action="store_true",
                         help="Require price above/below a trend MA for BUY/SELL respectively")
    parser.add_argument("--trend-ma-period", type=int, default=200)
    parser.add_argument("--min-adx", type=float, default=0.0,
                         help="Require ind_ctx['adx'] >= this (0 disables). Suggested: ~22-25")
    parser.add_argument("--max-atr-multiple", type=float, default=0.0,
                         help="Gate out entries where current ATR > this x trailing average ATR "
                              "(0 disables). Suggested: ~1.5-2.0")
    parser.add_argument("--allow-golden-zone", action="store_true",
                         help="Re-admit 50-61.8% retracement entries (excluded by default — "
                              "backtest evidence showed 0%% win rate for this band)")
    parser.add_argument("--no-cluster-for-golden", action="store_true",
                         help="With --allow-golden-zone, admit ANY golden-zone touch instead of "
                              "requiring multi-swing cluster corroboration (not recommended)")
    parser.add_argument("--allow-cluster-touch-entry", action="store_true",
                         help="Accept a multi-swing cluster at price as a trigger even with no "
                              "qualifying candle pattern (raises frequency)")
    parser.add_argument("--min-rr-dynamic", action="store_true",
                         help="Lower (never raise) the min-R:R bar in demonstrably low-volatility conditions")
    parser.add_argument("--max-swing-age-bars", type=int, default=0,
                         help="Reject swings whose extreme is older than this many closed bars (0 disables)")
    parser.add_argument("--require-volume-confirmation", action="store_true",
                         help="Require trigger-bar volume >= --volume-multiple x recent average")
    parser.add_argument("--volume-multiple", type=float, default=1.5)

    # NEW (this pass) — mirrors fibonacci.py's Day-63 liquidity params.
    parser.add_argument("--require-liquidity-alignment", action="store_true",
                         help="Hard-gate out signals whose direction actively conflicts with the "
                              "liquidity engine's bias (NEUTRAL never blocks). See fibonacci.py's "
                              "__init__ docstring for the evidence behind this defaulting off.")
    parser.add_argument("--min-liquidity-score", type=int, default=0,
                         help="Additionally require liquidity_ctx['score'] >= this (0 disables). "
                              "NOT recommended above ~0 yet — see fibonacci.py docstring, this cut "
                              "the validated EURUSD H1 sample from 32 trades to 4 when tested near "
                              "the liquidity engine's own MIN_LIQUIDITY_SCORE=55.")

    # Day 42 — trade-management additions (change how an OPEN trade is
    # managed, independent of the signal-gating flags above).
    parser.add_argument("--use-partial-tp", action="store_true",
                         help="Bank --partial-tp-fraction of size at tp1, move remainder's SL to "
                              "breakeven, and let it run to tp2 instead of closing fully at tp1")
    parser.add_argument("--partial-tp-fraction", type=float, default=0.5)
    parser.add_argument("--use-breakeven-stop", action="store_true",
                         help="Move SL to entry once price has moved --breakeven-trigger-r in favor")
    parser.add_argument("--breakeven-trigger-r", type=float, default=1.0)

    parser.add_argument("--csv-out", default=None, help="Optional path to dump per-trade CSV")
    args = parser.parse_args()

    cfg = BacktestConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        bars=args.bars,
        spread_pips=args.spread_pips,
        slippage_pips=args.slippage_pips,
        min_confidence=args.min_confidence,
        max_holding_bars=args.max_holding_bars,
        min_rr=args.min_rr,
        require_trigger_candle=not args.no_trigger_candle,
        use_context_engines=not args.no_context_engines,
        context_window_bars=args.context_window_bars,
        require_htf_alignment=args.require_htf_alignment,
        min_confluence_strength=args.min_confluence_strength,
        require_engulfing_only=args.require_engulfing_only,
        require_trend_ma=args.require_trend_ma,
        trend_ma_period=args.trend_ma_period,
        min_adx=args.min_adx,
        max_atr_multiple=args.max_atr_multiple,
        allow_golden_zone=args.allow_golden_zone,
        require_cluster_for_golden=not args.no_cluster_for_golden,
        allow_cluster_touch_entry=args.allow_cluster_touch_entry,
        min_rr_dynamic=args.min_rr_dynamic,
        max_swing_age_bars=args.max_swing_age_bars,
        require_volume_confirmation=args.require_volume_confirmation,
        volume_multiple=args.volume_multiple,
        require_liquidity_alignment=args.require_liquidity_alignment,
        min_liquidity_score=args.min_liquidity_score,
        use_partial_tp=args.use_partial_tp,
        partial_tp_fraction=args.partial_tp_fraction,
        use_breakeven_stop=args.use_breakeven_stop,
        breakeven_trigger_r=args.breakeven_trigger_r,
    )

    try:
        connect_mt5()
        df = fetch_closed_bars(cfg.symbol, cfg.timeframe, cfg.bars)
    except RuntimeError as e:
        log.error("Backtest aborted: %s", e)
        sys.exit(1)
    finally:
        if mt5 is not None:
            mt5.shutdown()

    backtester = FibonacciBacktester(cfg)
    trades = backtester.run(df)
    report = build_report(trades, cfg, backtester.rejection_counts)
    print(report)

    if args.csv_out and trades:
        pd.DataFrame([t.__dict__ for t in trades]).to_csv(args.csv_out, index=False)
        log.info("Per-trade detail written to %s", args.csv_out)


if __name__ == "__main__":
    main()