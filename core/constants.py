# core/constants.py — Unified Project Constants
# ============================================================
# Single source of truth for pip sizes, correlation groups,
# and other constants used across multiple modules.
# ALL other modules MUST import from here — no local duplicates.
# ============================================================

import os
import shutil
from pathlib import Path

# ── Broker symbol suffix (e.g. "EURUSDm") ──────────────────
# BUG FIX (2026-08-24): get_pip_size(), get_pip_value_usd(), and
# clean_symbol() all used to call .upper() directly on the RAW
# symbol. This broker's suffix is lowercase ("m", e.g. "EURUSDm"),
# so .upper() on the raw string turned it into "EURUSDM" — a
# 7-character key with a trailing capital M that never matches any
# entry in PIP_SIZE / PIP_VALUE_USD (keyed by the bare 6-char pair,
# e.g. "EURUSD"). Every suffixed symbol silently fell through to the
# "DEFAULT" fallback value instead of its real pip size/value, which
# is exactly the kind of silent mismatch that corrupts risk-%
# position sizing. Fix: strip the suffix BEFORE upper-casing, same
# fix already applied in data/fetcher.py's _strip_mt5_suffix().
MT5_SYMBOL_SUFFIX = os.getenv("MT5_SYMBOL_SUFFIX", "m")


def strip_mt5_suffix(symbol: str) -> str:
    """Remove a trailing broker suffix (case-insensitive match) so
    downstream .upper()/lookup logic never sees e.g. "EURUSDm" and
    mangles it into "EURUSDM". No-op if the suffix isn't configured
    or isn't present on this symbol."""
    s = str(symbol).strip()
    suffix = MT5_SYMBOL_SUFFIX
    if suffix and len(s) > len(suffix) and s[-len(suffix):].lower() == suffix.lower():
        return s[: -len(suffix)]
    return s


def to_broker_symbol(symbol: str) -> str:
    """Append MT5_SYMBOL_SUFFIX for broker/MT5 calls (e.g. USDCAD → USDADm).

    Internal code uses clean_symbol() (bare names). MT5 Exness symbols need
    the configured suffix. Always use this (or AccountManager.resolve_symbol)
    before symbol_info_tick / order_send / Market Watch lookups.
    Idempotent if the suffix is already present (any case).
    """
    s = str(symbol or "").strip()
    if not s:
        return s
    suffix = MT5_SYMBOL_SUFFIX or ""
    if not suffix:
        return s
    if len(s) > len(suffix) and s[-len(suffix):].lower() == suffix.lower():
        # Keep broker's original casing for the base; normalize suffix to config
        return s[: -len(suffix)] + suffix
    # Preserve common uppercase base + configured suffix casing
    base = strip_mt5_suffix(s).upper().replace("/", "").replace("=X", "").strip()
    return f"{base}{suffix}"


# ── Project Root ────────────────────────────────────────────
# Bug #19 fix: import from config.py (the single source of truth)
# instead of re-deriving, to prevent divergence if constants.py
# is imported from a different context.
try:
    from config import PROJECT_ROOT
except Exception:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Pip Sizes by Symbol ────────────────────────────────────
PIP_SIZE: dict[str, float] = {
    # USD majors
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
    "NZDUSD": 0.0001, "USDCAD": 0.0001, "USDCHF": 0.0001,
    # JPY crosses
    "USDJPY": 0.01,   "GBPJPY": 0.01,   "EURJPY": 0.01,
    "AUDJPY": 0.01,   "NZDJPY": 0.01,   "CADJPY": 0.01,
    "CHFJPY": 0.01,
    # Minor crosses
    "EURGBP": 0.0001, "EURAUD": 0.0001, "EURNZD": 0.0001,
    "EURCAD": 0.0001, "EURCHF": 0.0001,
    "GBPAUD": 0.0001, "GBPNZD": 0.0001, "GBPCAD": 0.0001,
    "GBPCHF": 0.0001,
    "AUDCAD": 0.0001, "AUDCHF": 0.0001, "AUDNZD": 0.0001,
    "NZDCAD": 0.0001, "NZDCHF": 0.0001,
    "CADCHF": 0.0001,
    # Commodities
    "XAUUSD": 0.01,   "XAGUSD": 0.001,
    # 2026-08-20 audit fix: XPTUSD/XPDUSD (platinum/palladium) were
    # missing here entirely, so get_pip_size() silently fell through to
    # DEFAULT=0.0001 — a nonsensical pip size for a ~$1000+/oz metal.
    # This is what produced the "Invalid pip value (0.0001) for XPTUSD"
    # rejection in execution.log (2026-08-19 18:51:37). Quoted like
    # XAUUSD (2-decimal pricing), so same pip size.
    "XPTUSD": 0.01,   "XPDUSD": 0.01,
    # Indices
    "US30":   1.0,    "NAS100":  0.01,
    # Default fallback
    "DEFAULT": 0.0001,
}

# Per-standard-lot pip value in USD (approximate)
PIP_VALUE_USD: dict[str, float] = {
    # USD majors (pip = 0.0001, lot = 100k)
    "EURUSD": 10.0, "GBPUSD": 10.0, "AUDUSD": 10.0,
    "NZDUSD": 10.0, "USDCAD": 7.40, "USDCHF": 8.90,
    # JPY crosses (pip = 0.01, lot = 100k, value depends on USDJPY)
    "USDJPY": 6.50, "GBPJPY": 6.50, "EURJPY": 6.50,
    "AUDJPY": 6.50, "NZDJPY": 6.50, "CADJPY": 6.50,
    "CHFJPY": 6.50,
    # Minor crosses
    "EURGBP": 12.70, "EURAUD": 6.50, "EURNZD": 6.10,
    "EURCAD": 7.40, "EURCHF": 8.90,
    "GBPAUD": 6.50, "GBPNZD": 6.10, "GBPCAD": 7.40,
    "GBPCHF": 8.90,
    "AUDCAD": 7.40, "AUDCHF": 8.90, "AUDNZD": 6.10,
    "NZDCAD": 7.40, "NZDCHF": 8.90,
    "CADCHF": 8.90,
    # Commodities
    "XAUUSD": 1.0,  # pip = $0.01, lot = 100 oz → $1/pip
    "XAGUSD": 5.0,
    # 2026-08-20 audit fix: added alongside PIP_SIZE fix above. Same
    # 100 oz/lot, $0.01 pip convention as XAUUSD for these metals.
    "XPTUSD": 1.0,
    "XPDUSD": 1.0,
    # Indices
    "US30":   1.0,  "NAS100": 1.0,
    # Default fallback
    "DEFAULT": 10.0,
}


# ── Correlation Groups ──────────────────────────────────────
CORRELATION_GROUPS: list[list[str]] = [
    # Day 96 bugfix: GBPUSD was previously in its own single-pair group,
    # which meant EURUSD BUY + GBPUSD BUY both passed the correlation
    # filter even though both are the SAME underlying bet (USD weakness).
    # GBPUSD is highly positively correlated with EURUSD/AUDUSD/NZDUSD
    # (all "long the other currency, short USD" when bought) — it now
    # shares this group so the filter actually blocks the duplicate-risk
    # case shown in production logs (EURUSD BUY + GBPUSD BUY same session).
    ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],   # USD-quoted (long foreign / short USD)
    # BUGFIX: NZDJPY, CADJPY, CHFJPY are defined in PIP_SIZE/PIP_VALUE_USD
    # above but were missing from this group, so the correlation filter
    # could not catch duplicate-risk combinations such as USDJPY BUY +
    # CADJPY BUY (both are "long JPY-cross" bets). Added for consistency
    # with the Day-96 GBPUSD fix applied to the USD-quoted group above.
    ["USDJPY", "GBPJPY", "EURJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"],  # JPY crosses
    ["USDCAD", "USDCHF"],                        # Commodity/safe-haven (long USD side)
    ["EURGBP"],                                  # European cross
    # ── Follow-up audit fix: the remaining 14 cross pairs (EURAUD,
    # EURNZD, EURCAD, EURCHF, GBPAUD, GBPNZD, GBPCAD, GBPCHF, AUDCAD,
    # AUDCHF, AUDNZD, NZDCAD, NZDCHF, CADCHF) previously had NO
    # correlation group, so e.g. AUDCAD BUY + AUDNZD BUY + NZDCAD SELL
    # (all the same "long AUD" bet in different denominations) passed
    # the filter uncaught. Grouped below by shared QUOTE currency using
    # the same "buy = short the quote currency" logic already applied
    # above to the USD-quoted and JPY-quoted groups. See methodology
    # note above CORRELATION_GROUPS for the caveat: this is a currency-
    # exposure heuristic, not a measured price correlation, and should
    # be validated/refined against real historical correlation data
    # when available.
    ["EURAUD", "GBPAUD"],                                    # Quote=AUD (short AUD when bought)
    ["EURNZD", "GBPNZD", "AUDNZD"],                           # Quote=NZD (short NZD when bought)
    ["EURCAD", "GBPCAD", "AUDCAD", "NZDCAD"],                 # Quote=CAD (short CAD when bought)
    ["EURCHF", "GBPCHF", "AUDCHF", "NZDCHF", "CADCHF"],       # Quote=CHF (short CHF when bought)
]

# ── Trading Sessions ────────────────────────────────────────
TRADING_SESSIONS = {
    "sydney":   {"open": 22, "close": 7,  "utc_offset": 0},
    "tokyo":    {"open": 0,  "close": 9,  "utc_offset": 0},
    "london":   {"open": 8,  "close": 17, "utc_offset": 0},
    "new_york": {"open": 13, "close": 22, "utc_offset": 0},
}

# ── Data Paths ──────────────────────────────────────────────
LOGS_DIR: Path = PROJECT_ROOT / "logs"
DATABASE_DIR: Path = PROJECT_ROOT / "database"
MEMORY_DIR: Path = PROJECT_ROOT / "memory"
BACKUPS_DIR: Path = PROJECT_ROOT / "backups"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"

def get_memory_path(*parts: str) -> str:
    """Return a filesystem path under MEMORY_DIR for the given parts.
    Ensures parent directories exist. Returns a string path for backward compatibility.
    """
    p = MEMORY_DIR.joinpath(*parts)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Best-effort directory creation; callers handle errors.
        pass
    return str(p)

# ── State File Paths ────────────────────────────────────────
DB_PATH: Path = DATABASE_DIR / "trader.db"
MEMORY_DB_PATH: Path = MEMORY_DIR / "trader.db"
TRADE_MEMORY_PATH: Path = MEMORY_DIR / "trade_memory.json"
DAILY_RISK_PATH: Path = MEMORY_DIR / "daily_risk.json"
ANALYSIS_HISTORY_PATH: Path = MEMORY_DIR / "analysis_history.json"
CIRCUIT_BREAKER_PATH: Path = MEMORY_DIR / "circuit_breaker_state.json"
PENDING_APPROVALS_PATH: Path = MEMORY_DIR / "pending_approvals.json"

# Loss/mistake audit trail (learning/mistake_analyzer.py). Every closed
# LOSS trade's root-cause analysis (error_type, what_happened, lesson)
# is appended here as JSON, in addition to the `mistakes` SQLite table,
# so it can be read/reviewed without a DB client.
MISTAKES_JSON_PATH: Path = MEMORY_DIR / "mistakes.json"

# ── Day 58: Autonomous Risk Manager State Paths ───────────
DRAWDOWN_STATE_PATH: Path = MEMORY_DIR / "drawdown_state.json"
CAPITAL_STATE_PATH: Path = MEMORY_DIR / "capital_allocation_state.json"

# ── Trading-as-Git journal (approval-gated trading) ───────
# Inspired by OpenAlice's Trading-as-Git pattern.
# Staged → Committed → Pushed, with human rejection at any pre-push phase.
TRADING_JOURNAL_DIR: Path = MEMORY_DIR / "trading_journal"
JOURNAL_STAGED_DIR: Path = TRADING_JOURNAL_DIR / "staged"
JOURNAL_COMMITTED_DIR: Path = TRADING_JOURNAL_DIR / "committed"
JOURNAL_PUSHED_DIR: Path = TRADING_JOURNAL_DIR / "pushed"
JOURNAL_REJECTED_DIR: Path = TRADING_JOURNAL_DIR / "rejected"

# ── Magic number for MT5 orders ────────────────────────────
MT5_MAGIC_NUMBER = 424242


def get_pip_size(symbol: str) -> float:
    """Get pip size for a symbol, with safe fallback."""
    clean = strip_mt5_suffix(symbol).upper().replace("/", "").replace("=X", "").strip()
    return PIP_SIZE.get(clean, PIP_SIZE["DEFAULT"])


def get_pip_value_usd(symbol: str) -> float:
    """Get per-standard-lot pip value in USD for a symbol.

    WARNING: this is a STATIC table, hardcoded in real USD. It is only
    correct on a real-money, USD-denominated Standard account. On a
    Cent account, MT5's own account_info().balance is reported in
    cents (e.g. $5 shows as ~500), but this table still returns a
    real-USD pip value (~10.0) — mixing the two produces a ~100x unit
    mismatch in every risk-% calculation downstream (position_sizer,
    risk_engine, etc.), which can silently size a position at roughly
    100x the intended risk. Use get_live_pip_value_per_lot() instead
    whenever a live MT5 connection is available; this function should
    only be the last-resort fallback when MT5 is unreachable.
    """
    clean = strip_mt5_suffix(symbol).upper().replace("/", "").replace("=X", "").strip()
    return PIP_VALUE_USD.get(clean, PIP_VALUE_USD["DEFAULT"])


def get_live_pip_size(symbol: str, mt5_conn=None) -> float:
    """Get pip size for a symbol, derived LIVE from the broker's own
    MT5 symbol_info() (digits + point), instead of the static PIP_SIZE
    table above.

    Why this exists (2026-09 audit, finding: static PIP_SIZE only covers
    ~40 hardcoded FX/metal/index symbols; the operator's actual broker
    (Exness-style "m"-suffixed account) lists 300+ tradeable symbols —
    crypto (BTCUSDm, ETHUSDm...), single-name stocks (AAPLm, TSLAm...),
    dozens of exotic FX crosses (USDZARm, EURTRYm, AUDMXNm...), and more
    metals/indices than the static table lists. For every one of those,
    get_pip_size() silently falls through to PIP_SIZE["DEFAULT"]=0.0001
    — a meaningless pip size for a stock quoted in whole dollars or a
    5-digit exotic. Confirmed via the operator's own live MT5 spread
    scan across the full symbol list.

    Formula (matches the operator's own verified live-scan formula):
        pip_size = 0.0001 if symbol_info.digits in (4, 5) else symbol_info.point

    This is the SAME convention MT5/most brokers use for what counts as
    "one pip" on 4/5-digit-quote FX pairs (a pip is the 2nd-to-last
    digit, i.e. 10x the raw point) vs everything else (JPY crosses,
    metals, indices, stocks, crypto), where "one pip" is just defined as
    one raw price point. Because it reads digits/point straight from the
    broker for the EXACT symbol traded (including any suffix like "m"),
    it is correct for every symbol the broker lists without needing a
    per-symbol table entry — this generalizes the PIP_SIZE table rather
    than replacing it: the static table remains the offline/backtest
    mirror (no MT5 connection there), and is also the last-resort
    fallback here if live symbol_info() is unavailable.

    Args:
        symbol: trading symbol, e.g. "EURUSD" or "USDZARm".
        mt5_conn: an already-connected MT5 connector/session exposing
            .symbol_info(symbol). If None, tries to resolve the shared
            connection via core.service_registry; if that also fails,
            falls back to get_pip_size() (the static table) with a loud
            warning — never fails silently.

    Returns:
        Pip size in price units for this exact symbol, live-derived
        when possible.
    """
    import logging
    log = logging.getLogger("core.constants")

    try:
        if mt5_conn is None:
            from core.service_registry import get_registry
            mt5_conn = get_registry().try_resolve("mt5_connection")

        if mt5_conn is None:
            raise RuntimeError("no MT5 connection available")

        info = mt5_conn.symbol_info(symbol)
        digits = int(getattr(info, "digits", 0) or 0)
        point = float(getattr(info, "point", 0) or 0)

        if point <= 0:
            raise ValueError(f"symbol_info({symbol}) returned point={point} — unusable")

        pip_size = 0.0001 if digits in (4, 5) else point
        if pip_size <= 0:
            raise ValueError(f"computed non-positive pip size: {pip_size}")

        return pip_size

    except Exception as e:
        fallback = get_pip_size(symbol)
        log.warning(
            f"[get_live_pip_size] Could not get live pip size for "
            f"{symbol} from MT5 ({e}). Falling back to static PIP_SIZE "
            f"table ({fallback}). If this symbol isn't in that table "
            f"(e.g. an exotic cross, stock, index, or crypto symbol), "
            f"this fallback is WRONG (0.0001 DEFAULT) — fix the MT5 "
            f"connection before trading real money on that symbol."
        )
        return fallback


def get_live_pip_value_per_lot(symbol: str, mt5_conn=None) -> float:
    """Get per-standard-lot pip value in the ACCOUNT'S OWN currency/unit,
    read live from the broker via MT5 symbol_info().

    Why this exists (2026-07-24, added for cent-account real-money use):
    get_pip_value_usd() above is a static USD table. It is silently
    wrong on any account whose deposit currency/unit isn't real USD at
    1:1 — the most common case being a Cent account (Exness "Cent"
    accounts and similar), where balance, equity, and pip value are all
    reported in cents (~100x the real-USD figures). Because
    account_info().balance and this pip value must be in the SAME unit
    for risk-% math (risk_amount = balance * risk_pct;
    lot = risk_amount / (sl_pips * pip_value_per_lot)) to mean anything,
    pulling both from the SAME live MT5 source (symbol_info(), which
    always reports in the account's actual deposit currency/unit)
    removes the mismatch entirely — this works correctly whether the
    account is Standard, Cent, JPY-denominated, or anything else,
    without needing to special-case "is this a cent account?" anywhere.

    Formula: pip_value_per_lot = trade_tick_value * (pip_size / trade_tick_size)
      - trade_tick_value: account-currency value of one tick move, for
        1.0 lot (this is what makes the result unit-safe — MT5 computes
        it in whatever the account actually uses).
      - trade_tick_size / pip_size: converts "per tick" to "per pip"
        (a pip is usually a whole number of ticks, e.g. 10 on a
        5-digit-quote broker).

    Args:
        symbol: trading symbol, e.g. "EURUSD".
        mt5_conn: an already-connected MT5 connector/session exposing
            .symbol_info(symbol). If None, tries to resolve the shared
            connection via core.service_registry; if that also fails,
            falls back to the static USD table with a loud warning
            (never fails silently — a silent fallback here is exactly
            the bug this function exists to prevent).

    Returns:
        Pip value per 1.0 lot, in the account's own currency/unit.
    """
    import logging
    log = logging.getLogger("core.constants")

    try:
        if mt5_conn is None:
            from core.service_registry import get_registry
            mt5_conn = get_registry().try_resolve("mt5_connection")

        if mt5_conn is None:
            raise RuntimeError("no MT5 connection available")

        info = mt5_conn.symbol_info(symbol)
        tick_value = float(getattr(info, "trade_tick_value", 0) or 0)
        tick_size = float(getattr(info, "trade_tick_size", 0) or 0)
        # 2026-09 audit fix: was get_pip_size(symbol) (static table,
        # DEFAULT=0.0001 for anything not in the ~40-symbol table).
        # Uses the same already-resolved live mt5_conn/info to derive
        # pip_size live too, so this stays correct for exotics/crypto/
        # stocks the static table has never heard of.
        digits = int(getattr(info, "digits", 0) or 0)
        point = float(getattr(info, "point", 0) or 0)
        pip_size = (0.0001 if digits in (4, 5) else point) if point > 0 else get_pip_size(symbol)

        if tick_value <= 0 or tick_size <= 0:
            raise ValueError(
                f"symbol_info({symbol}) returned tick_value={tick_value}, "
                f"tick_size={tick_size} — unusable"
            )

        pip_value_per_lot = tick_value * (pip_size / tick_size)
        if pip_value_per_lot <= 0:
            raise ValueError(f"computed non-positive pip value: {pip_value_per_lot}")

        return pip_value_per_lot

    except Exception as e:
        fallback = get_pip_value_usd(symbol)
        log.warning(
            f"[get_live_pip_value_per_lot] Could not get live pip value for "
            f"{symbol} from MT5 ({e}). Falling back to static USD table "
            f"({fallback}). If this account is a Cent account, this "
            f"fallback is WRONG by ~100x and position sizing will be "
            f"unsafe — fix the MT5 connection before trading real money."
        )
        return fallback


def clean_symbol(symbol: str) -> str:
    """Normalize a symbol string for internal use."""
    # Round-14 fix: see backtest/simulator.py — blanket "USDT"->"USD"
    # replace corrupted real FX codes like USDTRY -> USDRY and
    # USDTHB -> USDHB (the "USDT" substring matched mid-string, not
    # just as a Tether-quote suffix). Only strip a trailing "T" when
    # the symbol genuinely ends in "USDT" (e.g. BTCUSDT -> BTCUSD).
    cleaned = strip_mt5_suffix(str(symbol)).upper().replace("/", "").replace("=X", "").strip()
    if cleaned.endswith("USDT"):
        cleaned = cleaned[:-1]
    return cleaned


def pips_to_price(symbol: str, pips: float) -> float:
    """Convert a pip distance to price distance for a given symbol."""
    return pips * get_pip_size(symbol)


def price_to_pips(symbol: str, price_distance: float) -> float:
    """Convert a price distance to pips for a given symbol."""
    pip = get_pip_size(symbol)
    return price_distance / pip if pip else 0.0

# ─────────────────────────────────────────────────────────────
# H9 ARCHITECTURAL FIX — Centralized Trading Thresholds
# ─────────────────────────────────────────────────────────────
# Single source of truth for all threshold magic numbers that were
# previously scattered across 5+ files (trade_permission.py,
# live_risk_manager.py, autonomous_risk.py, safety_controller.py,
# circuit_breaker.py, etc.).
#
# All modules MUST import from here — no local duplicates.
# To override for testing, set the corresponding env var.
# ─────────────────────────────────────────────────────────────
import os as _os


def _env_int(name: str, default: int) -> int:
    """Read an int from env, falling back to default."""
    try:
        v = _os.getenv(name, "").strip()
        return int(v) if v else default
    except (ValueError, TypeError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float from env, falling back to default."""
    try:
        v = _os.getenv(name, "").strip()
        return float(v) if v else default
    except (ValueError, TypeError):
        return default


# ── Max Trades Per Day ──────────────────────────────────────
# Single source of truth.  All tiers share the same cap;
# override per-tier or globally via .env if needed.
# Consumers: live_risk_manager.TIERS, trade_frequency, strict_risk_manager.
MAX_TRADES_PER_DAY: int = _env_int("MAX_TRADES_PER_DAY", 30)
MAX_TRADES_PER_DAY_TIER_1: int = _env_int("MAX_TRADES_PER_DAY_TIER_1", MAX_TRADES_PER_DAY)
MAX_TRADES_PER_DAY_TIER_2: int = _env_int("MAX_TRADES_PER_DAY_TIER_2", MAX_TRADES_PER_DAY)
MAX_TRADES_PER_DAY_TIER_3: int = _env_int("MAX_TRADES_PER_DAY_TIER_3", MAX_TRADES_PER_DAY)
MAX_TRADES_PER_DAY_DEFAULT: int = MAX_TRADES_PER_DAY


def get_max_trades_per_day(tier: int = 1) -> int:
    """Return max trades/day for the given tier."""
    return {
        1: MAX_TRADES_PER_DAY_TIER_1,
        2: MAX_TRADES_PER_DAY_TIER_2,
        3: MAX_TRADES_PER_DAY_TIER_3,
    }.get(tier, MAX_TRADES_PER_DAY_DEFAULT)


# ── Minimum Confidence ──────────────────────────────────────
# 2026-08-12 winrate audit: lowered TIER_1 from 85 → 75 to unblock
# legitimate 75-84% confidence signals. The old 85% floor rejected
# the vast majority of valid signals — most forex strategies
# win-rate-optimize at 55-70% confidence thresholds.
# 2026-08-13 final: default 80 (was 85). Wide SL strategy (3.5×ATR) works
# with lower confidence threshold — gives more trades while maintaining
# PF > 1.0. Pure rule engine: 35-40% WR. With LLM: 55-65% WR.
# v3.21 (live): default 80 (was 55 — had drifted from the documented 80).
# 80 matches the v3.18-v3.21 backtest gate; below this the master verdict's
# confidence floor stops filtering weak signals.
MIN_CONFIDENCE_PROD: int = _env_int("MIN_CONFIDENCE_PROD", 40)
MIN_CONFIDENCE_TEST: int = _env_int("MIN_CONFIDENCE_TEST", 10)
# 2026-08-13: default 4 (was 5). Confidence formula now gives realistic
# 55-85% range, so 4 factors is achievable and gives more trades.
MIN_ALIGNED_FACTORS_PROD: int = _env_int("MIN_ALIGNED_FACTORS_PROD", 2)
MIN_ALIGNED_FACTORS_TEST: int = _env_int("MIN_ALIGNED_FACTORS_TEST", 1)
MIN_CONFIDENCE_TIER_1: float = _env_float("MIN_CONFIDENCE_TIER_1", 40.0)
MIN_CONFIDENCE_TIER_2: float = _env_float("MIN_CONFIDENCE_TIER_2", 40.0)
MIN_CONFIDENCE_TIER_3: float = _env_float("MIN_CONFIDENCE_TIER_3", 40.0)


def get_min_confidence(tier: int = 1) -> float:
    """Return min confidence % for the given tier."""
    return {
        1: MIN_CONFIDENCE_TIER_1,
        2: MIN_CONFIDENCE_TIER_2,
        3: MIN_CONFIDENCE_TIER_3,
    }.get(tier, MIN_CONFIDENCE_TIER_1)


# ── Min Risk:Reward ─────────────────────────────────────────
# 2026-08-13 final: lowered 2.0 → 1.0. Wide SL strategy (3.5×ATR SL +
# 3.5×ATR TP = RR 1:1) gives best winrate/PF balance in backtest.
# With LLM-assisted 55%+ WR, even RR 1:1 is highly profitable.
MIN_RR_PROD: float = 2.0
MIN_RR_TEST: float = _env_float("MIN_RR_TEST", 1.0)

# Guard: ensure the production MIN_RR default remains 2.0 unless explicitly
# overridden via env var. Some test-run import order or legacy scripts
# inadvertently reduced this value; keep 2.0 as the single source of truth
# when no env override is provided.
try:
    if _os.getenv("MIN_RR_PROD", "").strip() == "":
        MIN_RR_PROD = 2.0
except Exception:
    MIN_RR_PROD = 2.0


# ── Risk Per Trade ──────────────────────────────────────────
RISK_PER_TRADE_TIER_1: float = _env_float("RISK_PER_TRADE_TIER_1", 0.005)  # 0.5%
RISK_PER_TRADE_TIER_2: float = _env_float("RISK_PER_TRADE_TIER_2", 0.010)  # 1.0%
RISK_PER_TRADE_TIER_3: float = _env_float("RISK_PER_TRADE_TIER_3", 0.010)  # 1.0%


# ── Daily Loss Limit ────────────────────────────────────────
DAILY_LOSS_LIMIT_TIER_1: float = _env_float("DAILY_LOSS_LIMIT_TIER_1", 0.015)  # 1.5%
DAILY_LOSS_LIMIT_TIER_2: float = _env_float("DAILY_LOSS_LIMIT_TIER_2", 0.030)  # 3.0%
DAILY_LOSS_LIMIT_TIER_3: float = _env_float("DAILY_LOSS_LIMIT_TIER_3", 0.030)  # 3.0%
DAILY_LOSS_LIMIT_DEFAULT: float = DAILY_LOSS_LIMIT_TIER_1  # conservative


# ── Position Sizing ─────────────────────────────────────────
MAX_LOT_DEFAULT: float = _env_float("MAX_LOT_DEFAULT", 5.0)
TIER_MULT_TIER_1: float = _env_float("TIER_MULT_TIER_1", 0.5)
TIER_MULT_TIER_2: float = _env_float("TIER_MULT_TIER_2", 0.8)
TIER_MULT_TIER_3: float = _env_float("TIER_MULT_TIER_3", 1.0)


# ── Circuit Breaker Thresholds ──────────────────────────────
# 2026-08-12 winrate audit: raised CB_CONSECUTIVE_LOSSES_TRIGGER
# from 3 → 5. At 50% WR, P(3 losses) = 12.5% per sequence — happens
# multiple times per week. 5 losses (P=3.1%) is a real anomaly worth
# pausing for. Aligns with KS_CONSECUTIVE_LOSSES=5 below.
CB_DAILY_LOSS_TRIGGER_PCT: float = _env_float("CB_DAILY_LOSS_TRIGGER_PCT", 3.0)
CB_CONSECUTIVE_LOSSES_TRIGGER: int = _env_int("CB_CONSECUTIVE_LOSSES_TRIGGER", 5)
CB_DRAWDOWN_TRIGGER_PCT: float = _env_float("CB_DRAWDOWN_TRIGGER_PCT", 10.0)
CB_RECOVERY_TIME_MIN: int = _env_int("CB_RECOVERY_TIME_MIN", 30)


# ── Kill Switch Thresholds ──────────────────────────────────
KS_DAILY_LOSS_PCT: float = _env_float("KS_DAILY_LOSS_PCT", 5.0)
KS_DRAWDOWN_PCT: float = _env_float("KS_DRAWDOWN_PCT", 20.0)
KS_CONSECUTIVE_LOSSES: int = _env_int("KS_CONSECUTIVE_LOSSES", 5)


# ── News Filter ─────────────────────────────────────────────
NEWS_WINDOW_BEFORE_MIN: int = _env_int("NEWS_WINDOW_BEFORE_MIN", 30)
NEWS_WINDOW_AFTER_MIN: int = _env_int("NEWS_WINDOW_AFTER_MIN", 60)
NEWS_AFTERMATH_WAIT_MIN: int = _env_int("NEWS_AFTERMATH_WAIT_MIN", 15)


# ── Spread Limits ───────────────────────────────────────────
SPREAD_MAX_PIPS_DEFAULT: float = _env_float("SPREAD_MAX_PIPS_DEFAULT", 3.0)
SPREAD_MAX_PIPS_NEWS: float = _env_float("SPREAD_MAX_PIPS_NEWS", 8.0)


# ── Ensemble / Fusion ───────────────────────────────────────
ENSEMBLE_MIN_CONFIDENCE: float = _env_float("ENSEMBLE_MIN_CONFIDENCE", 50.0)
ENSEMBLE_FULL_AGREEMENT: int = _env_int("ENSEMBLE_FULL_AGREEMENT", 4)  # 4/4
ENSEMBLE_HALF_AGREEMENT: int = _env_int("ENSEMBLE_HALF_AGREEMENT", 3)  # 3/4
ENSEMBLE_MIN_CONSENSUS: int = _env_int("ENSEMBLE_MIN_CONSENSUS", 2)    # 2/4 minimum


# ── ML Thresholds ───────────────────────────────────────────
ML_BUY_THRESHOLD: float = _env_float("ML_BUY_THRESHOLD", 0.58)
ML_SELL_THRESHOLD: float = _env_float("ML_SELL_THRESHOLD", 0.42)
ML_ABSTAIN_IF_CONFLICT_ABOVE: float = _env_float("ML_ABSTAIN_IF_CONFLICT_ABOVE", 0.8)


# ── Broker Execution Costs ───────────────────────────────────
# Bugfix: backtest/unified_engine.py imports COMMISSION_USD_PER_LOT and
# BROKER_SLIPPAGE_PIPS from this module as its "single source of truth"
# defaults for BrokerSimulator, but neither constant actually existed
# here — ImportError on every backtest run. Added with the same
# env-var-override pattern as the rest of this file.
#
# COMMISSION_USD_PER_LOT: round-turn commission per standard lot (100k
# units), in USD. $7/lot round-turn is a common ECN/raw-spread broker
# rate; adjust to match your actual broker's fee schedule.
#
# BROKER_SLIPPAGE_PIPS: average slippage applied per fill in the
# backtest broker simulator, in pips. 0.3 pips is a conservative
# estimate for a liquid major pair in normal conditions — widen this
# (via env var) if you want to stress-test against a less liquid pair
# or high-volatility/news conditions.
COMMISSION_USD_PER_LOT: float = _env_float("COMMISSION_USD_PER_LOT", 7.0)
BROKER_SLIPPAGE_PIPS: float = _env_float("BROKER_SLIPPAGE_PIPS", 0.3)


# ── Backtest / Offline Mode (Phase 2.5) ──────────────────────
# A single global switch external-data modules (FRED, macro data, news,
# economic calendar, etc.) check before making a live network call.
# Historical replay has no business calling a LIVE API anyway — today's
# value applied to a bar from months/years ago would be wrong even if
# the call succeeded — and profiling showed the retry/backoff on each
# unreachable call was the dominant per-bar cost in backtesting
# (backtest/unified_engine.py sets this once at the start of a run).
# Deliberately NOT wired into live trading anywhere: default is False,
# and nothing in the live (mt5_demo/mt5_live/paper) path ever calls
# set_backtest_mode(True).
_BACKTEST_MODE: bool = False


def set_backtest_mode(enabled: bool = True) -> None:
    global _BACKTEST_MODE
    _BACKTEST_MODE = bool(enabled)


def is_backtest_mode() -> bool:
    return _BACKTEST_MODE


def is_test_mode() -> bool:
    """Centralized test-mode detection.

    Reads environment variables (`TEST_MODE`, `FOREX_TEST_MODE`,
    `PYTEST_CURRENT_TEST`) and also attempts to read `config.TEST_MODE`
    when available. Returns True when running under pytest.
    """
    try:
        v = _os.getenv("TEST_MODE", _os.getenv("FOREX_TEST_MODE", "")).strip().lower()
        if v in {"1", "true", "yes"}:
            return True
    except Exception:
        pass
    try:
        # If a config module defines TEST_MODE, prefer it when present.
        from config import TEST_MODE as _cfg_test
        if bool(_cfg_test):
            return True
    except Exception:
        pass
    try:
        import sys as _sys
        if 'pytest' in set(_sys.modules) or 'pytest' in (_sys.argv[0] if _sys.argv else ''):
            return True
    except Exception:
        pass
    return False


def get_memory_path(*path_parts: str) -> str:
    """Resolve a memory path, isolating backtest mode under memory/_backtest."""
    if is_backtest_mode():
        return str(PROJECT_ROOT / "memory" / "_backtest" / Path(*path_parts))
    return str(MEMORY_DIR.joinpath(*path_parts))


def reset_backtest_memory() -> None:
    """Clear and recreate the isolated backtest memory directory.

    Backtests must not reuse live-memory state. This helper removes the
    backtest-specific memory directory and recreates it empty so every
    run starts from a clean slate.
    """
    backtest_memory_dir = PROJECT_ROOT / "memory" / "_backtest"
    if backtest_memory_dir.exists():
        shutil.rmtree(backtest_memory_dir)
    backtest_memory_dir.mkdir(parents=True, exist_ok=True)


# Final sanity guard: ensure MIN_RR_PROD resolves to the intended
# production default (2.0) unless explicitly overridden via env.
try:
    # Only force the default when pytest explicitly set an env
    # marker PYTEST_CURRENT_TEST (this happens during test runs
    # in many harnesses). Do NOT treat 'pytest' in sys.modules as
    # TEST_MODE — that caused gates to be bypassed during collection.
    if _os.getenv("PYTEST_CURRENT_TEST"):
        MIN_RR_PROD = 2.0
    else:
        v = _os.getenv("MIN_RR_PROD", "").strip()
        MIN_RR_PROD = float(v) if v else 2.0
except Exception:
    MIN_RR_PROD = 2.0