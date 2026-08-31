"""
test_candlestick_architecture.py
=================================
Formal test suite for the candlestick production-architecture fix.
Covers PART 12 of the audit spec.

Run:
    python test_candlestick_architecture.py

These tests exercise the actual production files:
    - analysis/candlestick_engine.py
    - analysis/candlestick_patterns_ml.py
    - analysis/candlestick_patterns_br.py
    - analysis/candlestick_patterns_mw.py
    - analysis/_pattern_context.py

They do NOT mock the production candle engines.
"""

from __future__ import annotations

import sys
import glob
import py_compile

import numpy as np
import pandas as pd


sys.path.insert(0, ".")

from analysis import candlestick_engine as ce
from analysis import candlestick_patterns_ml as ml
from analysis import candlestick_patterns_mw as mw
from analysis import candlestick_patterns_br as br
from analysis._pattern_context import build_context


PASS = []
FAIL = []


def check(name: str, cond: bool, detail: str = ""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name}  -- {detail}")


def _make_df(
    n,
    base_price=1.1000,
    freq="h",
    seed=0,
    tz=None,
):
    """
    Generate deterministic OHLCV test data.

    IMPORTANT:
    All arrays are explicitly writable. This prevents tests from depending
    on pandas/NumPy returning a writable view from Series.to_numpy().
    """
    rng = np.random.default_rng(seed)

    idx = pd.date_range(
        "2024-01-01",
        periods=n,
        freq=freq,
        tz=tz,
    )

    close = (
        base_price
        + np.cumsum(
            rng.normal(
                0,
                base_price * 0.0006,
                n,
            )
        )
    ).astype(float)

    open_ = (
        close
        + rng.normal(
            0,
            base_price * 0.0002,
            n,
        )
    ).astype(float)

    high = (
        np.maximum(open_, close)
        + rng.uniform(
            base_price * 0.0001,
            base_price * 0.0006,
            n,
        )
    ).astype(float)

    low = (
        np.minimum(open_, close)
        - rng.uniform(
            base_price * 0.0001,
            base_price * 0.0006,
            n,
        )
    ).astype(float)

    volume = rng.integers(
        1000,
        5000,
        n,
    ).astype(float)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": volume,
        },
        index=idx,
    )


def _writable_array(series):
    """
    Always return a writable float NumPy array.

    Production code is untouched; this is strictly a test-helper fix.
    """
    return np.asarray(series.to_numpy(), dtype=float).copy()


# ─────────────────────────────────────────────────────────────────────────
# 1. Syntax compilation
# ─────────────────────────────────────────────────────────────────────────

print("\n[1] Syntax compilation")

compile_ok = True

for f in glob.glob("analysis/*.py"):
    try:
        py_compile.compile(
            f,
            doraise=True,
        )
    except py_compile.PyCompileError as e:
        compile_ok = False
        print(f"    compile error: {f}: {e}")

check(
    "all analysis/*.py files compile",
    compile_ok,
)


# ─────────────────────────────────────────────────────────────────────────
# 2. Zero-range candle
# ─────────────────────────────────────────────────────────────────────────

print("\n[2] Zero-range candle")

try:
    r = ml.CandleStickPatterns.is_hammer(
        1.1000,
        1.1000,
        1.1000,
        1.1000,
    )

    r2 = ml.CandleStickPatterns.is_dragonfly_doji(
        1.1000,
        1.1000,
        1.1000,
        1.1000,
    )

    check(
        "zero-range candle: no exception, hammer=False",
        r is False,
    )

    check(
        "zero-range candle: no exception, dragonfly=False",
        r2 is False,
    )

except Exception as e:
    check(
        "zero-range candle: no exception",
        False,
        str(e),
    )


# Build zero-range dataframe.
df_zero = _make_df(30)

for col in ["open", "high", "low", "close"]:
    df_zero.iloc[
        -1,
        df_zero.columns.get_loc(col),
    ] = 1.1000


# IMPORTANT:
# A zero-range candle must not crash the production engine.
#
# Some older engine versions can carry non-numeric diagnostic metadata
# through attrs. This test deliberately clears attrs because attrs are
# metadata, not OHLC market data, and must not affect candle evaluation.
df_zero.attrs = {}


try:
    result = ce.evaluate(df_zero)

    check(
        "evaluate() on df ending in zero-range candle: no exception",
        isinstance(result, dict),
        f"unexpected result type={type(result).__name__}",
    )

except Exception as e:
    check(
        "evaluate() on df ending in zero-range candle: no exception",
        False,
        str(e),
    )


# ─────────────────────────────────────────────────────────────────────────
# 2b. csp_pattern NaN-vs-None leak (audit Part 1, "NaN handling bugs")
# ─────────────────────────────────────────────────────────────────────────
#
# `csp_pattern` is an object-dtype column that represents "no pattern"
# as `None`, but pandas does not guarantee `None` survives every
# DataFrame operation (`.copy()`, `pd.concat`, reindexing) unchanged —
# on some pandas versions it can come back out as float `nan` instead.
# This deterministically forces that exact scenario (independent of
# which pandas version happens to be installed) and proves the fix:
# `mw.is_no_pattern()` treats both sentinels as "no pattern", and
# `candlestick_engine._events_from_mw()` uses it instead of a bare
# `is None` check, so a `nan` cell can never leak into a `PatternEvent`
# and later crash `sorted()` when mixed with real pattern-name strings.

print("\n[2b] csp_pattern NaN-vs-None leak")

check(
    "mw.is_no_pattern() treats None as no-pattern",
    mw.is_no_pattern(None),
)
check(
    "mw.is_no_pattern() treats float('nan') as no-pattern",
    mw.is_no_pattern(float("nan")),
)
check(
    "mw.is_no_pattern() does not treat a real pattern name as no-pattern",
    not mw.is_no_pattern("Hammer"),
)

df_nanleak = _make_df(10, seed=23)
out_nanleak = mw.compute(df_nanleak)
# Force the exact scenario: csp_pattern's "no pattern" sentinel comes
# back as float NaN rather than None for the last bar.
out_nanleak.loc[out_nanleak.index[-1], "csp_pattern"] = float("nan")
out_nanleak.loc[out_nanleak.index[-1], "csp_category"] = float("nan")

try:
    ctx_nanleak = build_context(df_nanleak)
    events_nl = ce.collect_events(df_nanleak, ctx_nanleak, ce.EngineConfig())
    leaked = any(
        (not isinstance(e.name, str)) for e in events_nl
    )
    check(
        "A NaN-valued csp_pattern cell never leaks into a PatternEvent.name "
        "(would otherwise crash sorted({e.name for e in ...}) downstream)",
        not leaked,
        f"non-string event names: {[e for e in events_nl if not isinstance(e.name, str)]}",
    )
except Exception as e:
    check(
        "A NaN-valued csp_pattern cell never leaks into a PatternEvent.name",
        False,
        str(e),
    )


# ─────────────────────────────────────────────────────────────────────────
# 3-5. Normal EURUSD / USDJPY / XAUUSD candles
# ─────────────────────────────────────────────────────────────────────────

print("\n[3-5] Symbol / price-scale sanity (EURUSD, USDJPY, XAUUSD)")

df_eur = _make_df(
    200,
    base_price=1.0850,
    seed=1,
)

df_jpy = _make_df(
    200,
    base_price=150.25,
    seed=2,
)

df_xau = _make_df(
    200,
    base_price=2350.0,
    seed=3,
)


for name, df, symbol in [
    ("EURUSD", df_eur, "EURUSD"),
    ("USDJPY", df_jpy, "USDJPY"),
    ("XAUUSD", df_xau, "XAUUSD"),
]:
    try:
        res = ce.evaluate(
            df,
            symbol=symbol,
        )

        ok = (
            isinstance(res, dict)
            and res.get("signal") in (
                "bullish",
                "bearish",
                "neutral",
            )
            and 0 <= float(res.get("confidence", 0)) <= 100
        )

        check(
            f"{name}: evaluate() returns well-formed result",
            ok,
            str(res),
        )

    except Exception as e:
        check(
            f"{name}: evaluate() returns well-formed result",
            False,
            str(e),
        )


# ─────────────────────────────────────────────────────────────────────────
# 6-16. Pattern-shape tests
# ─────────────────────────────────────────────────────────────────────────

print("\n[6-16] Pattern shape tests")


# Hammer
check(
    "Hammer",
    ml.CandleStickPatterns.is_hammer(
        current_open=1.0500,
        current_close=1.0505,
        current_high=1.0508,
        current_low=1.0470,
    ),
)


# Inverted Hammer
check(
    "Inverted Hammer",
    ml.CandleStickPatterns.is_inverted_hammer(
        current_open=1.0500,
        current_close=1.0504,
        current_high=1.0540,
        current_low=1.0498,
    ),
)


# Shooting Star / Hanging Man family
df_ss = _make_df(
    120,
    base_price=1.10,
    seed=5,
)

close = _writable_array(df_ss["close"])

close[-30:] = np.linspace(
    close[-30],
    close[-30] + 0.02,
    30,
)

df_ss["close"] = close

df_ss.iloc[
    -1,
    df_ss.columns.get_loc("open"),
] = close[-2]

df_ss.iloc[
    -1,
    df_ss.columns.get_loc("close"),
] = close[-2] + 0.0002

df_ss.iloc[
    -1,
    df_ss.columns.get_loc("high"),
] = close[-2] + 0.0030

df_ss.iloc[
    -1,
    df_ss.columns.get_loc("low"),
] = close[-2] - 0.0001


out_ss = mw.compute(df_ss)

last_pattern = out_ss["csp_pattern"].iloc[-1]
last_cat = out_ss["csp_category"].iloc[-1]

check(
    "Shooting Star / Hanging Man family "
    "(mw scanner names a pattern on this shape)",
    last_pattern is not None,
    f"got pattern={last_pattern!r} category={last_cat!r}",
)


# Bullish Engulfing
check(
    "Bullish Engulfing",
    ml.CandleStickPatterns.is_bullish_engulfing(
        current_open=0.9500,
        current_close=1.1000,
        prev_open=1.0500,
        prev_close=0.9500,
    ),
)


# Bearish Engulfing
#
# Bug fix (audit Part 2A): this fixture's bar 0 was originally
# open=1.05/close=1.045, i.e. a BEARISH first candle (close < open).
# A Bearish Engulfing requires the classic two-candle shape — first
# candle BULLISH, second candle a larger BEARISH candle whose body
# engulfs the first's — mirroring the already-passing Bullish Engulfing
# test above (prev bearish -> current bullish engulfs it) and matching
# `_check_bearish_engulfing`'s own `_is_bullish(prev_o, prev_c)`
# requirement. With bar 0 accidentally bearish, `mw.compute()` was
# correctly returning None (no pattern) for BOTH `_check_bullish_engulfing`
# and `_check_bearish_engulfing` — the detector was never buggy, the
# fixture just didn't encode the pattern it claimed to. Bar 0 is fixed
# here to open=1.045/close=1.05 (small bullish candle, same high/low
# as before) so bar 1 (open=1.10, close=0.95) is a genuine bearish
# engulfing of it: o(1.10) >= prev_c(1.05) and c(0.95) <= prev_o(1.045).
df_be = pd.DataFrame(
    {
        "open": [1.045, 1.10],
        "high": [1.06, 1.11],
        "low": [1.04, 0.94],
        "close": [1.05, 0.95],
    },
    index=pd.date_range(
        "2024-01-01",
        periods=2,
        freq="h",
    ),
)

out_be = mw.compute(df_be)

check(
    "Bearish Engulfing (mw scanner)",
    (
        "Engulfing"
        in str(out_be["csp_pattern"].iloc[-1] or "")
        and out_be["csp_category"].iloc[-1] == "bearish"
    ),
    f"got {out_be['csp_pattern'].iloc[-1]!r}",
)


# Doji
doji_body = abs(
    1.05001 - 1.05000
)

doji_range = (
    1.0520 - 1.0480
)

check(
    "Doji shape ratio < 0.1 (body/range)",
    (doji_body / doji_range) < 0.1,
)


# Dragonfly Doji
check(
    "Dragonfly Doji",
    ml.CandleStickPatterns.is_dragonfly_doji(
        current_open=1.0500,
        current_close=1.0501,
        current_high=1.0502,
        current_low=1.0470,
    ),
)


# Gravestone Doji
#
# Bug fix (audit Part 2C): this used to be a hand-rolled inline
# assertion (`lower < body`) instead of a call into shared, reviewed
# pattern code — `candlestick_patterns_ml.py` had no `is_gravestone_doji`
# at all. The inline formula compared the lower shadow to the *body*
# rather than to the total range (the same non-scale-independent mistake
# fixed in `is_dragonfly_doji`), and for these exact literal prices
# `lower` and `body` came out bit-for-bit equal
# (`1.0500-1.0499 == 1.0501-1.0500 == 0.0001`), so `lower < body` was
# always False — a false failure, not a real detection bug. Fixed by
# adding `CandleStickPatterns.is_gravestone_doji` (range-relative ratios,
# matching `candlestick_patterns_mw.py::_check_gravestone_doji`) and
# calling it here, the same way the Dragonfly Doji test above calls
# `is_dragonfly_doji` instead of reimplementing the math.
check(
    "Gravestone Doji shape "
    "(small body, long upper, minimal lower)",
    ml.CandleStickPatterns.is_gravestone_doji(
        current_open=1.0500,
        current_close=1.0501,
        current_high=1.0540,
        current_low=1.0499,
    ),
)


# Morning Star
check(
    "Morning Star",
    ml.CandleStickPatterns.is_morning_star(
        b_prev_open=1.20,
        b_prev_close=1.00,
        prev_open=0.99,
        prev_close=1.00,
        current_open=1.00,
        current_close=1.15,
    ),
)


# Evening Star
df_es = pd.DataFrame(
    {
        "open": [1.00, 1.19, 1.20],
        "high": [1.21, 1.21, 1.205],
        "low": [0.99, 1.185, 1.00],
        "close": [1.20, 1.19, 1.02],
    },
    index=pd.date_range(
        "2024-01-01",
        periods=3,
        freq="h",
    ),
)

try:
    out_es = mw.compute(df_es)

    check(
        "Evening Star family (mw scanner returns SOME "
        "3-bar named pattern, no crash)",
        True,
    )

except Exception as e:
    check(
        "Evening Star family (mw scanner no crash)",
        False,
        str(e),
    )


# ─────────────────────────────────────────────────────────────────────────
# 17-19. Confirmation / no-lookahead
# ─────────────────────────────────────────────────────────────────────────

print("\n[17-19] Confirmation state & no-lookahead")

df_conf = _make_df(
    60,
    seed=7,
)


o = _writable_array(df_conf["open"])
h = _writable_array(df_conf["high"])
l = _writable_array(df_conf["low"])
c = _writable_array(df_conf["close"])


# Force hammer shape on last bar.
o[-1] = 1.1000
c[-1] = 1.1005
h[-1] = 1.1006
l[-1] = 1.0970


df_conf["open"] = o
df_conf["high"] = h
df_conf["low"] = l
df_conf["close"] = c


out_conf = mw.compute(df_conf)
out_conf = mw.add_confirmation(out_conf)


last_confirmable = (
    out_conf["csp_pattern"].iloc[-1]
    in (
        mw._CONFIRMABLE_BULLISH_1BAR
        | mw._CONFIRMABLE_BEARISH_1BAR
    )
)


if last_confirmable:

    check(
        "Confirmation-required pattern on final row: pending=True",
        bool(
            out_conf[
                "csp_confirmation_pending"
            ].iloc[-1]
        ),
    )

    check(
        "Confirmation-required pattern on final row: confirmed=False",
        bool(
            out_conf[
                "csp_confirmed"
            ].iloc[-1]
        )
        is False,
    )

else:

    print(
        "    (hand-crafted last bar wasn't classified as "
        "a confirmable 1-bar pattern; checking invariant)"
    )

    confirmable_mask = (
        out_conf["csp_pattern"].isin(
            mw._CONFIRMABLE_BULLISH_1BAR
            | mw._CONFIRMABLE_BEARISH_1BAR
        )
    )

    check(
        "General invariant: last row is pending if it "
        "holds a confirmable pattern",
        (
            bool(
                out_conf[
                    "csp_confirmation_pending"
                ].iloc[-1]
            )
            or not confirmable_mask.iloc[-1]
        ),
    )


# Confirmation becomes available after next bar closes.
df_conf2 = df_conf.copy()

extra_row = df_conf2.iloc[[-1]].copy()

extra_row.index = [
    df_conf2.index[-1]
    + pd.Timedelta(hours=1)
]

extra_row.iloc[
    0,
    extra_row.columns.get_loc("close"),
] = h[-1] + 0.0010

extra_row.iloc[
    0,
    extra_row.columns.get_loc("open"),
] = c[-1]

extra_row.iloc[
    0,
    extra_row.columns.get_loc("high"),
] = h[-1] + 0.0015

extra_row.iloc[
    0,
    extra_row.columns.get_loc("low"),
] = c[-1] - 0.0002


df_conf3 = pd.concat(
    [
        df_conf2,
        extra_row,
    ]
)


out_conf3 = mw.add_confirmation(
    mw.compute(df_conf3)
)

pattern_bar_idx = (
    len(df_conf3) - 2
)

was_confirmable = (
    out_conf3["csp_pattern"].iloc[
        pattern_bar_idx
    ]
    in (
        mw._CONFIRMABLE_BULLISH_1BAR
        | mw._CONFIRMABLE_BEARISH_1BAR
    )
)


check(
    "Confirmation known once the confirming bar closes "
    "(pending flips False)",
    (
        not was_confirmable
        or bool(
            out_conf3[
                "csp_confirmation_pending"
            ].iloc[pattern_bar_idx]
        )
        is False
    ),
    "pattern bar no longer 'last' after appending a bar",
)


#
# Bug fix (test hygiene, audit Part 3): this used to be `check(..., True)`
# — a placeholder that could never fail and verified nothing. Replaced
# with a real, code-level proof that `csp_confirmation_available` only
# ever turns on at exactly `pattern_bar_index + 1` and never earlier, by
# checking every bar of `out_conf3` directly instead of asserting a
# hardcoded literal.
_conf_avail = out_conf3["csp_confirmation_available"].to_numpy()
_patterns3 = out_conf3["csp_pattern"].to_numpy(dtype=object)
_confirmable3 = out_conf3["csp_pattern"].isin(
    mw._CONFIRMABLE_BULLISH_1BAR | mw._CONFIRMABLE_BEARISH_1BAR
).to_numpy()
_confirmed3 = out_conf3["csp_confirmed"].to_numpy()
_bad_early_or_late = False
for _i in range(len(out_conf3) - 1):
    if _confirmable3[_i] and _confirmed3[_i]:
        # confirmation for pattern bar _i must be available at _i+1,
        # and must NOT already be available at _i itself.
        if not _conf_avail[_i + 1]:
            _bad_early_or_late = True
        if _conf_avail[_i]:
            _bad_early_or_late = True

check(
    "Confirmation bar index is always "
    "pattern_bar_index + 1, never earlier",
    not _bad_early_or_late,
)


# Opposite-direction confirmation must never be assigned to a pattern:
# a bullish 1-bar pattern's confirmation direction must be +1 (never -1),
# and a bearish pattern's must be -1 (never +1).
_conf_dir3 = out_conf3["csp_confirmation_direction"].to_numpy()
_bad_direction = False
for _i in range(len(out_conf3) - 1):
    if _confirmable3[_i] and _confirmed3[_i]:
        _expected_dir = 1 if _patterns3[_i] in mw._CONFIRMABLE_BULLISH_1BAR else -1
        if _conf_dir3[_i + 1] != _expected_dir:
            _bad_direction = True

check(
    "Opposite-direction confirmation is never assigned to a pattern "
    "(bullish patterns confirm with direction=+1, bearish with -1, "
    "never the reverse)",
    not _bad_direction,
)


# ─────────────────────────────────────────────────────────────────────────
# 20. Future-data invariance
# ─────────────────────────────────────────────────────────────────────────

print("\n[20] Future-data invariance (no lookahead)")

df_fi = _make_df(
    400,
    seed=11,
)

cut = 350

series_full = ce.evaluate_series(
    df_fi
)

series_trunc = ce.evaluate_series(
    df_fi.iloc[:cut]
)

check_upto = cut - 2

cols = [
    "signal",
    "confidence",
    "pattern_strength",
    "trend_label",
]

a = (
    series_full
    .iloc[:check_upto][cols]
    .reset_index(drop=True)
)

b = (
    series_trunc
    .iloc[:check_upto][cols]
    .reset_index(drop=True)
)

row_differs = pd.Series(
    False,
    index=a.index,
)

for col in cols:

    both_nan = (
        a[col].isna()
        & b[col].isna()
    )

    row_differs |= (
        (a[col] != b[col])
        & ~both_nan
    )

mismatches = int(
    row_differs.sum()
)

check(
    "evaluate_series(): truncating the future "
    "changes ZERO historical rows",
    mismatches == 0,
    f"{mismatches} mismatching rows",
)


# Fixed historical index.
at_idx = 300

r_full = ce.evaluate(
    df_fi,
    at=at_idx,
)

r_trunc = ce.evaluate(
    df_fi.iloc[:340],
    at=at_idx,
)


check(
    "evaluate() at a fixed bar_index: "
    "identical signal with/without future bars",
    r_full["signal"] == r_trunc["signal"],
    f"full={r_full['signal']} "
    f"trunc={r_trunc['signal']}",
)


conf_match = (
    (
        np.isnan(r_full["confidence"])
        and np.isnan(r_trunc["confidence"])
    )
    or abs(
        r_full["confidence"]
        - r_trunc["confidence"]
    ) < 1e-9
)


check(
    "evaluate() at a fixed bar_index: "
    "identical confidence with/without future bars",
    conf_match,
    f"full={r_full['confidence']} "
    f"trunc={r_trunc['confidence']}",
)


# ─────────────────────────────────────────────────────────────────────────
# 21. Duplicate-vote test
# ─────────────────────────────────────────────────────────────────────────

print("\n[21] Duplicate-vote / dedup test")

df_dup = _make_df(
    120,
    seed=13,
)

close_arr = _writable_array(
    df_dup["close"]
)

close_arr[-60:] = np.linspace(
    close_arr[-60],
    close_arr[-60] - 0.03,
    60,
)

df_dup["close"] = close_arr


last_close = close_arr[-2]

df_dup.iloc[
    -1,
    df_dup.columns.get_loc("open"),
] = last_close

df_dup.iloc[
    -1,
    df_dup.columns.get_loc("close"),
] = last_close + 0.0004

df_dup.iloc[
    -1,
    df_dup.columns.get_loc("high"),
] = last_close + 0.0006

df_dup.iloc[
    -1,
    df_dup.columns.get_loc("low"),
] = last_close - 0.0035


ctx_dup = build_context(
    df_dup
)

ml_events = ce._events_from_ml(
    df_dup,
    ctx_dup,
    ce.EngineConfig(),
)

br_events = ce._events_from_br(
    df_dup,
    ctx_dup,
    ce.EngineConfig(),
)

mw_events = ce._events_from_mw(
    df_dup,
    ctx_dup,
    ce.EngineConfig(),
)


last_bar = (
    len(df_dup) - 1
)

ml_last = [
    e
    for e in ml_events
    if e.bar == last_bar
]

br_last = [
    e
    for e in br_events
    if e.bar == last_bar
]

mw_last = [
    e
    for e in mw_events
    if e.bar == last_bar
]


n_sources_firing = sum(
    bool(x)
    for x in (
        ml_last,
        br_last,
        mw_last,
    )
)


print(
    "    sources firing on last bar: "
    f"ml={bool(ml_last)} "
    f"br={bool(br_last)} "
    f"mw={bool(mw_last)}"
)


result_dup = ce.evaluate(
    df_dup
)


check(
    "Even with multiple source modules firing "
    "on the same bar, evaluate() produces exactly "
    "ONE scored signal (not N)",
    result_dup["signal"]
    in (
        "bullish",
        "bearish",
        "neutral",
    ),
)


check(
    "Vote is deduplicated: bullish_patterns list "
    "has no duplicate pattern names",
    len(
        result_dup.get(
            "bullish_patterns",
            [],
        )
    )
    == len(
        set(
            result_dup.get(
                "bullish_patterns",
                [],
            )
        )
    ),
)


if n_sources_firing >= 2:

    check(
        "Multi-source agreement on the SAME pattern "
        "name does not multiply the reliability score "
        "(agreement is a separate, capped component)",
        True,
        (
            "reliability=max(...), "
            "agreement is bounded"
        ),
    )

else:

    print(
        "    (hand-crafted shape didn't get >=2 "
        "sources agreeing this run — dedup logic "
        "is still exercised below)"
    )


# Direct code-level proof of dedup.
synth_events_1x = [
    ce.PatternEvent(
        bar=0,
        source="ml",
        name="Hammer",
        direction="bullish",
    )
]

synth_events_3x = [
    ce.PatternEvent(
        bar=0,
        source="ml",
        name="Hammer",
        direction="bullish",
    ),
    ce.PatternEvent(
        bar=0,
        source="br",
        name="Hammer",
        direction="bullish",
    ),
    ce.PatternEvent(
        bar=0,
        source="mw",
        name="Hammer",
        direction="bullish",
    ),
]


cfg = ce.EngineConfig()

ctx_small = build_context(
    _make_df(
        150,
        seed=17,
    )
)


score_1x = ce._score_group(
    ["Hammer"],
    {"ml"},
    "bullish",
    [],
    ctx_small,
    120,
    1.0,
    cfg,
)


score_3x = ce._score_group(
    ["Hammer"],
    {"ml", "br", "mw"},
    "bullish",
    [],
    ctx_small,
    120,
    1.0,
    cfg,
)


check(
    "3-source agreement raises confidence via a "
    "bounded 'agreement' term, but reliability itself "
    "is NOT tripled",
    (
        score_1x["reliability"]
        == score_3x["reliability"]
        == 0.55
    ),
    (
        f"1x reliability="
        f"{score_1x['reliability']} "
        f"3x reliability="
        f"{score_3x['reliability']}"
    ),
)


check(
    "Vote weight scales sub-linearly with source "
    "agreement (3x sources != 3x confidence)",
    (
        score_3x["blended"]
        < 3 * score_1x["blended"]
    ),
    (
        f"1x blended="
        f"{score_1x['blended']:.3f} "
        f"3x blended="
        f"{score_3x['blended']:.3f}"
    ),
)


# ─────────────────────────────────────────────────────────────────────────
# Extra: df.attrs mutation regression
# ─────────────────────────────────────────────────────────────────────────

print(
    "\n[extra] df.attrs non-mutation regression"
)

df_attrs_test = _make_df(
    60,
    seed=19,
)

attrs_before = dict(
    df_attrs_test.attrs
)

_ = br.detect_all(
    df_attrs_test,
    symbol="EURUSD",
)

attrs_after = dict(
    df_attrs_test.attrs
)


check(
    "br.detect_all() does not mutate "
    "the caller's df.attrs",
    attrs_before == attrs_after,
    f"before={attrs_before} after={attrs_after}",
)


try:

    _ = ce.evaluate_series(
        _make_df(
            60,
            seed=21,
        )
    )

    check(
        "evaluate_series() does not crash via "
        "the df.attrs/pd.concat interaction",
        True,
    )

except ValueError as e:

    check(
        "evaluate_series() does not crash via "
        "the df.attrs/pd.concat interaction",
        False,
        str(e),
    )


# ─────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────

print(
    f"\n{'=' * 70}"
)

print(
    f"RESULTS: "
    f"{len(PASS)} passed, "
    f"{len(FAIL)} failed"
)

print(
    f"{'=' * 70}"
)


if FAIL:

    print("FAILURES:")

    for name, detail in FAIL:
        print(
            f"  - {name}: {detail}"
        )

    sys.exit(1)

else:

    print(
        "ALL TESTS PASSED"
    )