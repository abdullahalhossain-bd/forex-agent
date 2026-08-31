# analysis/candlestick_patterns_br.py — Brazilian-book candlestick patterns
# =============================================================================
# Ported from: https://github.com/JimmyAreaFiscal/MercadoFinanceiro/blob/main/livro_candlesticks.ipynb
# Original author: JimmyAreaFiscal — license not specified (public repo)
# Source book: "Candlestick: um método para ampliar lucros na bolsa de valores"
#
# 11 candlestick pattern detectors from a Brazilian trading book. Unlike the
# basic boolean detectors in candlestick_patterns_ml.py or the 33-pattern
# MotiveWave scanner, these patterns add THREE layers of filtering:
#
#   1. TREND CONFIRMATION — uses 9-EMA vs 50-SMA crossover with a rolling
#      percentage threshold (e.g., 80% of last 5 bars must be in downtrend
#      for a bullish reversal to fire).
#   2. VOLUME CONFIRMATION — volume must exceed the recent max/average
#      (configurable period and ratio).
#   3. NEXT-BAR CONFIRMATION — some patterns (e.g., Inverted Hammer) wait
#      1 bar and require the next bar to confirm (close above the pattern's
#      high) before firing.
#
# This "triple filter" approach significantly reduces false signals compared
# to raw pattern detection. The trade-off: fewer signals, but higher quality.
#
# All functions return a boolean pd.Series (True on bars where the pattern
# fires). They use vectorized pandas operations — no per-bar loops.
#
# Differences from the original notebook:
#   - Volume column is optional (defaults to a synthetic constant if 'real_volume'
#     or 'volume' column is missing — volume filter is skipped).
#   - Function names translated to English (encontrar_martelo → hammer).
#   - Portuguese aliases preserved for backwards compatibility.
#   - Each function takes a single `df` argument (the original took `dados`).
# =============================================================================

from __future__ import annotations

import inspect
import weakref
from typing import Optional

import pandas as pd
import numpy as np

from utils.logger import get_logger
from analysis._engine_utils import pip_value

log = get_logger("candlestick_patterns_br")

# Tracks whether the FX-unsafe-default warning has already been logged this
# process, so a hot loop calling doji()/spinning_top() per-bar without a
# `symbol` doesn't spam the log.
_warned_no_symbol: set[str] = set()

# Private, module-level trend cache used by `_compute_trend()` below. Keyed
# by (id(df), fast_span, slow_period, id(close_values)) and evicted via
# weakref.finalize when the source DataFrame is garbage-collected. NEVER
# stored on `df.attrs` — mutating a caller's DataFrame as a side channel
# broke `pd.concat()` for any other code touching the same df downstream
# (see the CACHING FIX note in `_compute_trend()`).
_TREND_CACHE: dict = {}


def _resolve_absolute_threshold(
    default_for_stocks: float, pips: float, symbol: Optional[str], fn_name: str,
    df: Optional[pd.DataFrame] = None, atr_fraction: float = 0.15,
) -> float:
    """
    Resolve an absolute price threshold in an asset-aware way.

    `doji()` and `spinning_top()` compare candle body/shadow size directly
    against an absolute price threshold (not a ratio of the bar's range).
    That is fine for stock/index prices (dollars) but silently meaningless
    on 5-decimal FX quotes — e.g. a body of 0.0010 (10 pips on EURUSD) would
    never exceed a stock-scaled default of 0.01, so on FX data every candle
    would qualify as a Doji regardless of the intended filter.

    Resolution order (Phase 3 improvement — see AUDIT_REPORT.md
    "br.py: FX-unsafe absolute thresholds"):

      1. Explicit `symbol` given → pip-scaled threshold via `pip_value()`.
         Most precise when the caller knows the instrument.
      2. No `symbol`, but `df` given and long enough for ATR(14) → a
         threshold derived from `atr_fraction * ATR`. This is the general
         fix: it is asset-agnostic (works for stocks, FX, crypto, indices
         alike) because it scales with the instrument's OWN recent
         volatility instead of a hardcoded constant, and doesn't require a
         symbol/pip lookup table at all. This is now the DEFAULT path.
      3. Neither available (e.g. < atr_period bars, cold start) → the
         original stock-scaled default, with the original one-time warning
         (fully backward compatible with pre-fix behaviour).
    """
    if symbol:
        return pip_value(symbol) * pips
    if df is not None and len(df) >= 14:
        try:
            from analysis._pattern_context import atr as _atr_fn
            a = _atr_fn(df, period=14)
            last_valid = a.dropna()
            if len(last_valid) > 0:
                return float(last_valid.iloc[-1]) * atr_fraction
        except Exception:
            pass  # fall through to the legacy stock-scaled default below
    if fn_name not in _warned_no_symbol:
        log.warning(
            f"{fn_name}() called without `symbol` or a usable `df` for ATR "
            f"scaling — using the stock-scaled default ({default_for_stocks}). "
            f"On FX data (5-decimal quotes) this threshold is far too large "
            f"and the filter will rarely, if ever, reject a candle. Pass "
            f"`symbol=...` (e.g. 'EURUSD') or ensure >=14 bars are available "
            f"for automatic ATR-based scaling."
        )
        _warned_no_symbol.add(fn_name)
    return default_for_stocks


# ── Helper: get volume column (real_volume or volume) ────────────────────────

def _get_volume(df: pd.DataFrame) -> pd.Series | None:
    """Return the volume Series, or None if no volume column exists."""
    if "real_volume" in df.columns:
        return df["real_volume"]
    if "tick_volume" in df.columns:
        return df["tick_volume"]
    if "volume" in df.columns:
        return df["volume"]
    return None


def _compute_trend(df: pd.DataFrame, fast_span: int = 9, slow_period: int = 50):
    """
    Compute 9-EMA and 50-SMA trend indicators.
    Returns (mm_9, mm_50) — both pd.Series.

    PERFORMANCE FIX (Phase 2 audit: "br.py recomputes trend from scratch in
    every one of its 8 trend-gated pattern functions"): every public
    pattern function in this module (martelo, martelo_invertido,
    engolfo_de_alta, harami_fundo, linha_perfuracao, pinca_fundo,
    chute_alta, estrela) calls `_trend_down_confirmed`/`_trend_up_confirmed`
    independently, and each of those calls `_compute_trend`. On a single
    `detect_all(df)` call that's the same 9-EMA/50-SMA recomputed ~8 times
    over identical data.

    CACHING FIX (candlestick architecture audit, 2026-08-31): this used to
    cache the result on `df.attrs["_br_trend_cache"]` — a dict pandas
    already carries on every DataFrame. That silently MUTATED the caller's
    shared DataFrame with array-valued entries. Once any Series derived
    from that DataFrame (via ordinary column selection, which inherits
    `.attrs`) was later passed through `pd.concat()` anywhere else in the
    pipeline — e.g. `_pattern_context.atr()`, which every candlestick_engine
    call runs right after this module — pandas' `__finalize__` attrs-
    equality check tries to compare dicts containing numpy arrays/Series
    and raises `ValueError: The truth value of an array... is ambiguous`.
    This was reproduced directly: `evaluate()` (single bar) worked, but
    `evaluate_series()` (which calls `br.detect_all()` and then later
    rebuilds ATR via `pd.concat` on the same df) crashed immediately after.
    A "performance cache" must never have an observable side effect on the
    caller's object. The cache now lives in a private, module-level,
    id-keyed dict instead of on `df.attrs`, with a `weakref.finalize` hook
    that evicts the entry once the DataFrame is garbage-collected — so
    there is no memory leak and no mutation of anything the caller can see.
    """
    cache_key = (id(df), fast_span, slow_period, id(df["close"].values))
    cached = _TREND_CACHE.get(cache_key)
    if cached is not None:
        return cached
    mm_9 = df["close"].ewm(span=fast_span, adjust=False).mean()
    mm_50 = df["close"].rolling(slow_period).mean()
    _TREND_CACHE[cache_key] = (mm_9, mm_50)
    # Evict this entry once `df` itself is garbage-collected, so the cache
    # can never outlive (or leak) the DataFrame it was computed from. If
    # `df` isn't weak-referenceable for some reason, fail open (no caching
    # for that call) rather than raise — correctness over the optimization.
    try:
        weakref.finalize(df, _TREND_CACHE.pop, cache_key, None)
    except TypeError:
        pass
    return mm_9, mm_50


def _trend_down_confirmed(df: pd.DataFrame, qtde_candle_tendencia: int = 5,
                          perc_confirmacao_tendencia: float = 0.8,
                          fast_span: int = 9, slow_period: int = 50) -> pd.Series:
    """
    Downtrend confirmation: 9-EMA < 50-SMA for at least
    perc_confirmacao_tendencia (e.g., 80%) of the last
    qtde_candle_tendencia bars.
    """
    mm_9, mm_50 = _compute_trend(df, fast_span, slow_period)
    downtrend = (mm_9 < mm_50)
    return downtrend.rolling(qtde_candle_tendencia).mean() >= perc_confirmacao_tendencia


def _trend_up_confirmed(df: pd.DataFrame, qtde_candle_tendencia: int = 5,
                        perc_confirmacao_tendencia: float = 0.8,
                        fast_span: int = 9, slow_period: int = 50) -> pd.Series:
    """Uptrend confirmation (mirror of _trend_down_confirmed)."""
    mm_9, mm_50 = _compute_trend(df, fast_span, slow_period)
    uptrend = (mm_9 > mm_50)
    return uptrend.rolling(qtde_candle_tendencia).mean() >= perc_confirmacao_tendencia


# ── 1. Ambivalent patterns ───────────────────────────────────────────────────

def marobozu(df: pd.DataFrame, sombra_max: float = 0.01) -> pd.Series:
    """
    Marubozu: candle with negligible shadows (≤ 1% of body).
    Ambivalent — direction-agnostic.
    """
    corpo = abs(df["close"] - df["open"])
    sombra_sup = df["high"] - df[["close", "open"]].max(axis=1)
    sombra_inf = df[["close", "open"]].min(axis=1) - df["low"]
    # Guard against zero-body candles
    corpo_safe = corpo.replace(0, np.nan)
    signal = (sombra_sup <= sombra_max * corpo_safe) & \
             (sombra_inf <= sombra_max * corpo_safe)
    return signal.fillna(False)


def doji(df: pd.DataFrame, corpo_max: Optional[float] = None,
        symbol: Optional[str] = None) -> pd.Series:
    """
    Doji: candle with body ≤ corpo_max (absolute value, not percentage).
    NOTE: the original uses an absolute threshold, not a percentage of range.
    For forex, use a small value like 0.0001 (1 pip). For stocks, use 0.01.

    If `corpo_max` is not given, it's resolved automatically: from
    `symbol` via `_engine_utils.pip_value()` when `symbol` is provided
    (recommended for FX call sites), or the original stock-scaled 0.01
    default otherwise (logged as a one-time warning, since 0.01 is far too
    large a body threshold for 5-decimal FX quotes). Passing an explicit
    `corpo_max` always wins and skips this resolution — fully backward
    compatible with existing call sites.
    """
    if corpo_max is None:
        corpo_max = _resolve_absolute_threshold(0.01, pips=2.0, symbol=symbol, fn_name="doji", df=df)
    corpo = abs(df["close"] - df["open"])
    return corpo <= corpo_max


def spinning_top(df: pd.DataFrame,
                 qtde_candles_proximos: int = 5,
                 percentual_diff_candles_proximos: float = 0.3,
                 diff_sombras: Optional[float] = None,
                 corpo_min: Optional[float] = None,
                 symbol: Optional[str] = None) -> pd.Series:
    """
    Spinning Top: small body (≤ 30% of recent min body) with roughly equal
    upper and lower shadows (difference ≤ diff_sombras), and a body no
    smaller than `corpo_min` (a noise floor — without it, a run of
    near-zero-body candles would all qualify).

    `diff_sombras` and `corpo_min` are absolute price thresholds (like
    `doji`'s `corpo_max`) and share the same asset-scale caveat: the
    original stock-scaled defaults are far too large for 5-decimal FX
    quotes. Pass `symbol` (e.g. "EURUSD") to derive both automatically via
    `_engine_utils.pip_value()`; explicit values always win and are fully
    backward compatible with existing call sites.
    """
    if diff_sombras is None:
        diff_sombras = _resolve_absolute_threshold(0.01, pips=1.0, symbol=symbol, fn_name="spinning_top(diff_sombras)", df=df)
    if corpo_min is None:
        corpo_min = _resolve_absolute_threshold(3, pips=0.5, symbol=symbol, fn_name="spinning_top(corpo_min)", df=df, atr_fraction=0.05)
    corpo = abs(df["close"] - df["open"])
    media_corpo = corpo.rolling(qtde_candles_proximos).min().shift(1)
    sinal_corpo_pequeno = (corpo <= media_corpo * percentual_diff_candles_proximos) & \
                          (corpo >= corpo_min)
    sombra_sup = df["high"] - df[["close", "open"]].max(axis=1)
    sombra_inf = df[["close", "open"]].min(axis=1) - df["low"]
    sinal_sombras = abs(sombra_sup - sombra_inf) <= diff_sombras
    return sinal_sombras & sinal_corpo_pequeno


def estrela(df: pd.DataFrame,
            qtde_candles_proximos: int = 5,
            percentual_diff_candles_proximos: float = 0.3) -> pd.Series:
    """
    Star (Estrela): small candle that gaps from the previous candle, with a
    trend context. Detects both Morning Star (in downtrend) and Evening Star
    (in uptrend) variants via gap direction + trend.
    """
    candles_total = abs(df["high"] - df["low"])
    media_candles_total = candles_total.rolling(qtde_candles_proximos).mean().shift(1)
    sinal_candle_pequeno = candles_total <= media_candles_total * percentual_diff_candles_proximos

    gap_baixa = (df["open"] < df[["close", "open"]].min(axis=1).shift(1)) & \
                (df["close"] < df[["close", "open"]].min(axis=1).shift(1))
    gap_alta = (df["open"] > df[["close", "open"]].max(axis=1).shift(1)) & \
               (df["close"] > df[["close", "open"]].max(axis=1).shift(1))

    mm_50 = df["close"].rolling(50).mean()
    mm_9 = df["close"].ewm(span=9, adjust=False).mean()
    tendencia_alta = mm_9 > mm_50
    tendencia_baixa = mm_9 < mm_50
    candle_alta = df["close"] > df["open"]
    candle_baixa = df["close"] < df["open"]

    # Morning star: gap down on prev, gap up on current, was in downtrend 2 bars ago
    sinal_gaps_estrela_manha = (gap_baixa.shift(1)) & (gap_alta) & \
                               (tendencia_baixa.shift(2) & candle_baixa.shift(2))
    # Evening star: gap up on prev, gap down on current, was in uptrend 2 bars ago
    sinal_gaps_estrela_tarde = (gap_alta.shift(1)) & (gap_baixa) & \
                               (tendencia_alta.shift(2) & candle_alta.shift(2))

    return (sinal_gaps_estrela_tarde | sinal_gaps_estrela_manha) & \
           (sinal_candle_pequeno.shift(1))


# ── 2. Bullish patterns ─────────────────────────────────────────────────────

def martelo(df: pd.DataFrame,
            relacao_sombra_corpo: float = 2.5,
            relacao_sombra_sup_sombra_inf: float = 0.1,
            qtde_candle_tendencia: int = 5,
            perc_confirmacao_tendencia: float = 0.8,
            require_confirmation: bool = False) -> pd.Series:
    """
    Hammer (Martelo): long lower shadow (≥ 2.5x body), small/no upper shadow,
    in a confirmed downtrend, with volume increase.

    `require_confirmation` (Phase 3 fix — see AUDIT_REPORT.md "br.py:
    inconsistent confirmation logic between Hammer and Inverted Hammer"):
    `martelo_invertido()` (below) already requires the NEXT bar to close
    above the pattern's high before firing, because multiple research
    sources (IG article: "the next day must be bullish to confirm this
    reversal pattern"; Ross Cameron transcripts: "we look for confirmation,
    the first candle to make a new high") say single-bar reversal
    candles are meaningfully more reliable once confirmed — yet the plain
    Hammer had no equivalent option, an unjustified inconsistency between
    two sibling patterns. Default is `False` (fires on the hammer bar
    itself, preserving the original, backward-compatible behaviour);
    set `True` for confirmed (one-bar-delayed, lower false-positive) signals.
    """
    # Long lower shadow
    candle_martelo = df[["close", "open"]].min(axis=1) - df["low"] >= \
                     relacao_sombra_corpo * abs(df["close"] - df["open"])
    # Small upper shadow
    sem_sombra_sup = df["high"] - df[["close", "open"]].max(axis=1) <= \
                     relacao_sombra_sup_sombra_inf * \
                     (df[["close", "open"]].min(axis=1) - df["low"])
    candle_martelo = candle_martelo & sem_sombra_sup

    # Trend confirmation
    tendencia_baixa = _trend_down_confirmed(df, qtde_candle_tendencia,
                                            perc_confirmacao_tendencia)
    # Volume confirmation
    vol = _get_volume(df)
    if vol is not None:
        aumento_volume = vol >= vol.rolling(qtde_candle_tendencia).max()
        sinal = candle_martelo & tendencia_baixa & aumento_volume
    else:
        sinal = candle_martelo & tendencia_baixa

    if not require_confirmation:
        return sinal

    # Next-bar confirmation: the bar AFTER the hammer must close above the
    # hammer's high. This shifts the signal itself forward by one bar (it
    # now fires on the confirmation bar, not the hammer bar) — no lookahead,
    # since we only ever use `.shift(1)` (past data relative to the firing bar).
    confirmacao = df["close"] > df["high"].shift(1)
    return sinal.shift(1).fillna(False) & confirmacao


def martelo_invertido(df: pd.DataFrame,
                      relacao_sombra_corpo: float = 2.5,
                      relacao_sombra_inf_sombra_sup: float = 0.1,
                      qtde_candle_tendencia: int = 5,
                      perc_confirmacao_tendencia: float = 0.8) -> pd.Series:
    """
    Inverted Hammer (Martelo Invertido): long upper shadow, small/no lower
    shadow, in downtrend, with volume + NEXT-BAR CONFIRMATION (the next bar
    must close above the pattern's high).
    """
    candle_martelo_inv = df["high"] - df[["close", "open"]].max(axis=1) >= \
                         relacao_sombra_corpo * abs(df["close"] - df["open"])
    sem_sombra_inf = df[["close", "open"]].min(axis=1) - df["low"] <= \
                     relacao_sombra_inf_sombra_sup * \
                     (df["high"] - df[["close", "open"]].max(axis=1))
    candle_martelo_inv = candle_martelo_inv & sem_sombra_inf

    tendencia_baixa = _trend_down_confirmed(df, qtde_candle_tendencia,
                                            perc_confirmacao_tendencia)
    vol = _get_volume(df)
    if vol is not None:
        aumento_volume = vol >= vol.rolling(qtde_candle_tendencia).max()
        sinal = candle_martelo_inv & tendencia_baixa & aumento_volume
    else:
        sinal = candle_martelo_inv & tendencia_baixa

    # Next-bar confirmation: next bar's close or open > pattern's high
    confirmacao = (df["close"] > df[["close", "open"]].max(axis=1).shift(1)) | \
                  (df["open"] > df[["close", "open"]].max(axis=1).shift(1))
    return (sinal.shift(1)) & confirmacao


def engolfo_de_alta(df: pd.DataFrame,
                    relacao_entre_corpos: float = 2.5,
                    perc_bem_distante: float = 0.1,
                    qtde_candle_tendencia: int = 5,
                    perc_confirmacao_tendencia: float = 0.8) -> pd.Series:
    """
    Bullish Engulfing (Engolfo de Alta): current candle's body is ≥ 2.5x prev
    body, engulfs prev body, prev is bearish, current is bullish, in downtrend,
    with volume increase.
    """
    corpo = abs(df["close"] - df["open"])
    candle_corpo_grande = corpo >= relacao_entre_corpos * corpo.shift(1)
    candle_engolfante = (df["open"] < (1 - perc_bem_distante) * df["close"].shift(1)) & \
                        (df["close"] > (1 + perc_bem_distante) * df["open"].shift(1))
    candle_de_baixa = df["close"] < df["open"]
    candle_de_alta = df["close"] > df["open"]
    tendencia_baixa = _trend_down_confirmed(df, qtde_candle_tendencia,
                                            perc_confirmacao_tendencia)

    sinal = candle_corpo_grande & candle_engolfante & \
            candle_de_baixa.shift(1) & candle_de_alta & tendencia_baixa

    vol = _get_volume(df)
    if vol is not None:
        aumento_volume = vol >= vol.rolling(qtde_candle_tendencia).max().shift(1)
        sinal = sinal & aumento_volume
    return sinal


def harami_fundo(df: pd.DataFrame,
                 relacao_entre_corpos: float = 3,
                 sombra_dentro_corpo: bool = True,
                 qtde_candle_tendencia: int = 5,
                 perc_confirmacao_tendencia: float = 0.8) -> pd.Series:
    """
    Bullish Harami (Harami de Fundo): prev candle has large body (≥ 3x current),
    current candle is contained within prev body, opposite directions, in downtrend.
    """
    corpo = abs(df["close"] - df["open"])
    candle_corpo_grande = corpo.shift(1) >= relacao_entre_corpos * corpo

    if sombra_dentro_corpo:
        candle_dentro = (df["low"] >= df[["open", "close"]].min(axis=1).shift(1)) & \
                        (df["high"] <= df[["open", "close"]].max(axis=1).shift(1))
    else:
        candle_dentro = (df[["open", "close"]].max(axis=1) <= df[["open", "close"]].max(axis=1).shift(1)) & \
                        (df[["open", "close"]].min(axis=1) >= df[["open", "close"]].min(axis=1).shift(1))

    dir_candle = (df["close"] > df["open"]) & (df["close"].shift(1) < df["open"].shift(1))
    tendencia_baixa = _trend_down_confirmed(df, qtde_candle_tendencia,
                                            perc_confirmacao_tendencia)
    return candle_corpo_grande & candle_dentro & dir_candle & tendencia_baixa


def linha_perfuracao(df: pd.DataFrame,
                     tamanho_gap: float = 0.01,
                     percentual_avanco: float = 0.5,
                     qtde_candle_tendencia: int = 5,
                     perc_confirmacao_tendencia: float = 0.8,
                     require_gap: bool = True) -> pd.Series:
    """
    Piercing Line (Linha de Perfuração): prev bearish, current bullish, current
    opens below prev low (if `require_gap`), closes above prev midpoint
    (>= 50% advance into prev body), in downtrend, with advance filter.

    `require_gap` (Phase 3 fix — see AUDIT_REPORT.md "br.py: hard-coded gap
    requirement contradicts 24h-market evidence"): the original notebook
    hard-requires `open < prev_low`, matching the Nison textbook definition.
    But per the Rayner Teo video transcript (uploaded source), FX and crypto
    trade ~24h/day and essentially never gap the way single-session equities
    do, so this hard requirement makes the pattern nearly impossible to
    detect on that data — exactly the same class of asset-agnosticism bug
    already present in `engolfo_de_alta`'s configurable `perc_bem_distante`.
    Set `require_gap=False` for continuous (FX/crypto) markets; the default
    (`True`) preserves the original, fully backward-compatible behaviour for
    single-session equities where the textbook gap is meaningful.
    """
    cond_1 = (df["open"].shift(1) > df["close"].shift(1)) & (df["close"] > df["open"])
    cond_2 = (df["open"] < df["low"].shift(1)) if require_gap else pd.Series(True, index=df.index)
    cond_3 = df["close"] >= df[["close", "open"]].mean(axis=1).shift(1)
    cond_4 = _trend_down_confirmed(df, qtde_candle_tendencia, perc_confirmacao_tendencia)

    if require_gap:
        filtro_1 = df["low"].shift(1) - df["open"] >= \
                   tamanho_gap * abs(df["close"].shift(1) - df["open"].shift(1))
    else:
        filtro_1 = pd.Series(True, index=df.index)
    filtro_2 = df["close"] - df[["close", "open"]].min(axis=1).shift(1) >= \
               abs(df["close"] - df["open"]).shift(1) * percentual_avanco

    return cond_1 & cond_2 & cond_3 & cond_4 & filtro_1 & filtro_2


def pinca_fundo(df: pd.DataFrame,
                max_diferenca_minimos: float = 0.001,
                relacao_min_corpo_total_1_candle: float = 0.5,
                relacao_max_corpo_total_2_candle: float = 0.25,
                qtde_candle_tendencia: int = 5,
                perc_confirmacao_tendencia: float = 0.8,
                buscar_reforcado: bool = False,
                reforco_superior_topo_anterior: float = 0.1) -> pd.Series:
    """
    Tweezer Bottom (Pinça de Fundo): two candles with similar lows, first is
    bearish with large body, second has small body, in downtrend.
    Optionally search for reinforced variant (3-candle tweezer).
    """
    cond_1 = abs(df["low"] - df["low"].shift(1)) <= max_diferenca_minimos * df["low"].shift(1)
    cond_2 = (df["open"].shift(1) - df["close"].shift(1)) >= \
             relacao_min_corpo_total_1_candle * (df["high"].shift(1) - df["low"].shift(1))
    cond_3 = abs(df["close"] - df["open"]) <= \
             relacao_max_corpo_total_2_candle * (df["high"] - df["low"])
    cond_4 = _trend_down_confirmed(df, qtde_candle_tendencia, perc_confirmacao_tendencia)

    if buscar_reforcado:
        filtro_1 = abs(df["low"].shift(1) - df["low"].shift(2)) <= \
                   max_diferenca_minimos * df["low"].shift(2)
        filtro_2 = df[["open", "close"]].max(axis=1) - \
                   df[["open", "close"]].max(axis=1).shift(1) >= \
                   reforco_superior_topo_anterior * df[["open", "close"]].max(axis=1).shift(1)
        return cond_1 & cond_2 & cond_3 & cond_4 & filtro_1 & filtro_2
    return cond_1 & cond_2 & cond_3 & cond_4


def chute_alta(df: pd.DataFrame,
               max_relacao_sombras_corpo: float = 0.01,
               qtde_candle_tendencia: int = 5,
               perc_confirmacao_tendencia: float = 0.8,
               filtro_volume: bool = True,
               periodo_historico_volume: int = 10,
               prop_min_entre_volumes: float = 1.1) -> pd.Series:
    """
    Bullish Kicking (Chute de Alta): prev is bearish Marubozu (no shadows),
    current is bullish Marubozu, current gaps above prev high, in downtrend.
    Optional volume filter: current volume > recent average AND > prev volume.
    """
    corpo = abs(df["close"] - df["open"])
    sombra_sup = df["high"] - df[["close", "open"]].max(axis=1)
    sombra_inf = df[["close", "open"]].min(axis=1) - df["low"]

    cond_1 = (df["close"].shift(1) < df["open"].shift(1)) & \
             (sombra_sup.shift(1) < corpo.shift(1) * max_relacao_sombras_corpo) & \
             (sombra_inf.shift(1) < corpo.shift(1) * max_relacao_sombras_corpo)
    cond_2 = (df["close"] > df["open"]) & \
             (sombra_sup < corpo * max_relacao_sombras_corpo) & \
             (sombra_inf < corpo * max_relacao_sombras_corpo)
    cond_3 = df["low"] > df["high"].shift(1)
    cond_4 = _trend_down_confirmed(df, qtde_candle_tendencia, perc_confirmacao_tendencia)

    sinal = cond_1 & cond_2 & cond_3 & cond_4
    if filtro_volume:
        vol = _get_volume(df)
        if vol is not None:
            filtro = (vol > vol.rolling(periodo_historico_volume).mean()) & \
                     (vol > vol.shift() * prop_min_entre_volumes)
            sinal = sinal & filtro
        else:
            log.debug("Volume filter requested but no volume column — skipping")
    return sinal


# ── Registry ─────────────────────────────────────────────────────────────────

PATTERN_FUNCTIONS = {
    # Ambivalent
    'Marubozu':       marobozu,
    'Doji':           doji,
    'Spinning Top':   spinning_top,
    'Star':           estrela,
    # Bullish
    'Hammer':                 martelo,
    'Inverted Hammer':        martelo_invertido,
    'Bullish Engulfing':      engolfo_de_alta,
    'Bullish Harami':         harami_fundo,
    'Piercing Line':          linha_perfuracao,
    'Tweezer Bottom':         pinca_fundo,
    'Bullish Kicking':        chute_alta,
}


# ── Portuguese aliases (backwards compatibility with original notebook) ─────

encontrar_marobozus = marobozu
encontrar_doji = doji
encontrar_spinning_top = spinning_top
encontrar_estrela = estrela
encontrar_martelo = martelo
encontrar_martelo_invertido = martelo_invertido
encontrar_engolfo_de_alta = engolfo_de_alta
encontrar_harami_fundo = harami_fundo
encontrar_linha_perfuracao = linha_perfuracao
encontrar_pinca_fundo = pinca_fundo
encontrar_chute_alta = chute_alta


def detect_all(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Run all 11 pattern detectors on `df` and return a DataFrame of bool columns.

    Any kwargs are passed to each pattern function that accepts them; a
    function that doesn't declare a given kwarg simply doesn't receive it
    (rather than erroring). This is what lets a single
    `detect_all(df, symbol="EURUSD")` call correctly pip-scale `doji()` and
    `spinning_top()`'s absolute thresholds while leaving ratio-based
    detectors (which don't take `symbol`) untouched.

    BUG FIXED: previously this function accepted **kwargs in its signature
    but never actually forwarded them to `fn(df)` — every pattern was
    always called with defaults only, silently contradicting this
    docstring. Any caller relying on e.g. `detect_all(df, corpo_max=...)`
    to reach `doji()` was being ignored with no error or warning.
    """
    results = pd.DataFrame(index=df.index)
    for name, fn in PATTERN_FUNCTIONS.items():
        try:
            accepted = set(inspect.signature(fn).parameters.keys())
            fn_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
            results[name] = fn(df, **fn_kwargs)
        except Exception as e:
            log.warning(f"Pattern {name} failed: {e}")
            results[name] = False
    return results


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    # Synthetic data with enough bars for 50-SMA + trend filters
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    # Downtrending data (so bullish patterns can fire)
    close = 100.0 - np.cumsum(rng.normal(0.05, 0.5, n))
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.5, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.5, n)
    vol = rng.integers(1000, 10000, n).astype(float)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "real_volume": vol,
    }, index=idx)

    # Run all patterns
    results = detect_all(df)
    print("Pattern detection results:")
    for col in results.columns:
        n_signals = int(results[col].sum())
        if n_signals > 0:
            print(f"  {col:25s}: {n_signals} signals")

    # Test individual patterns
    print(f"\nMarubozu: {int(marobozu(df).sum())} signals")
    print(f"Doji:     {int(doji(df, corpo_max=0.01).sum())} signals")
    print(f"Hammer:   {int(martelo(df).sum())} signals")

    # Test without volume column
    df_no_vol = df.drop(columns=['real_volume'])
    results_nv = detect_all(df_no_vol)
    print(f"\nWithout volume column:")
    for col in results_nv.columns:
        n_signals = int(results_nv[col].sum())
        if n_signals > 0:
            print(f"  {col:25s}: {n_signals} signals")

    print("\nBrazilian candlestick patterns smoke test passed.")

    # ── Regression check: FX-scale asset-aware thresholds ──────────────
    print("\n── Regression check: FX-scale doji/spinning_top thresholds ──")
    fx_idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    fx_close = 1.0850 + np.cumsum(rng.normal(0, 0.0003, n))
    fx_open = fx_close + rng.normal(0, 0.0006, n)  # bodies spread ~0-15 pips, FX-scale
    fx_high = np.maximum(fx_open, fx_close) + rng.uniform(0.0001, 0.0003, n)
    fx_low = np.minimum(fx_open, fx_close) - rng.uniform(0.0001, 0.0003, n)
    fx_df = pd.DataFrame({"open": fx_open, "high": fx_high, "low": fx_low,
                          "close": fx_close}, index=fx_idx)

    # OLD failure mode (pre-Phase-3-fix), reproduced by forcing the legacy
    # stock-scaled constant via a too-short DataFrame (ATR needs >=14 bars,
    # so this is the one remaining path that still hits the 0.01 default):
    doji_legacy_default = doji(fx_df.iloc[:10])
    n_legacy = int(doji_legacy_default.sum())
    print(f"  Doji signals, legacy stock-scaled fallback (<14 bars, no symbol): "
          f"{n_legacy}/10 bars")
    assert n_legacy > 8, "expected the legacy stock-scaled default to over-flag FX-scale bodies"

    # NEW default path (Phase 3 fix): no symbol given, but enough bars for
    # ATR(14) — the threshold now auto-scales to the instrument's own
    # volatility instead of silently using a stock-sized constant.
    doji_atr_scaled = doji(fx_df)
    n_atr_scaled = int(doji_atr_scaled.sum())

    # With symbol="EURUSD", the threshold is pip-scaled (2 pips ~= 0.0002).
    doji_fx_scaled = doji(fx_df, symbol="EURUSD")
    n_fx_scaled = int(doji_fx_scaled.sum())

    print(f"  Doji signals, NEW ATR-based default (no symbol, full df): {n_atr_scaled}/{n} bars")
    print(f"  Doji signals with symbol='EURUSD' (pip-scaled):           {n_fx_scaled}/{n} bars")
    # Both selective paths should reject the vast majority of bars (unlike
    # the legacy no-symbol/no-df constant, which flagged almost everything).
    assert n_atr_scaled < n * 0.5, "expected ATR-based default to be selective, not over-flag"
    assert n_fx_scaled < n * 0.5, "expected symbol-scaled threshold to be selective, not over-flag"
    print("  FX-scale threshold regression check passed.")

    # detect_all() must forward `symbol` correctly (this was previously a
    # dead kwarg — detect_all(df, **kwargs) never actually called fn(df, **kwargs))
    results_fx = detect_all(fx_df, symbol="EURUSD")
    assert int(results_fx["Doji"].sum()) == n_fx_scaled, (
        "detect_all() did not correctly forward `symbol` to doji()"
    )
    print("  detect_all() kwarg-forwarding regression check passed.")

    # ── Regression check: new optional confirmation / gap params ───────
    print("\n── Regression check: Hammer confirmation & Piercing Line gap toggle ──")
    n_hammer_raw = int(martelo(df).sum())
    n_hammer_confirmed = int(martelo(df, require_confirmation=True).sum())
    assert n_hammer_confirmed <= n_hammer_raw, \
        "confirmed hammer signals should never exceed raw hammer signals"
    print(f"  Hammer (raw):        {n_hammer_raw} signals")
    print(f"  Hammer (confirmed):  {n_hammer_confirmed} signals (subset of raw, one bar later)")

    n_piercing_gap = int(linha_perfuracao(df).sum())
    n_piercing_nogap = int(linha_perfuracao(df, require_gap=False).sum())
    assert n_piercing_nogap >= n_piercing_gap, \
        "dropping the hard gap requirement should never reduce signal count"
    print(f"  Piercing Line (require_gap=True):  {n_piercing_gap} signals")
    print(f"  Piercing Line (require_gap=False): {n_piercing_nogap} signals (>= gapped count)")
    print("  Confirmation/gap-toggle regression check passed.")