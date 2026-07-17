# core/constants.py — Unified Project Constants
# ============================================================
# Single source of truth for pip sizes, correlation groups,
# and other constants used across multiple modules.
# ALL other modules MUST import from here — no local duplicates.
# ============================================================

from pathlib import Path

# ── Project Root ────────────────────────────────────────────
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

# ── State File Paths ────────────────────────────────────────
DB_PATH: Path = DATABASE_DIR / "trader.db"
MEMORY_DB_PATH: Path = MEMORY_DIR / "trader.db"
TRADE_MEMORY_PATH: Path = MEMORY_DIR / "trade_memory.json"
DAILY_RISK_PATH: Path = MEMORY_DIR / "daily_risk.json"
ANALYSIS_HISTORY_PATH: Path = MEMORY_DIR / "analysis_history.json"
CIRCUIT_BREAKER_PATH: Path = MEMORY_DIR / "circuit_breaker_state.json"
PENDING_APPROVALS_PATH: Path = MEMORY_DIR / "pending_approvals.json"

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
    clean = symbol.upper().replace("/", "").replace("=X", "").strip()[:6]
    return PIP_SIZE.get(clean, PIP_SIZE["DEFAULT"])


def get_pip_value_usd(symbol: str) -> float:
    """Get per-standard-lot pip value in USD for a symbol."""
    clean = symbol.upper().replace("/", "").replace("=X", "").strip()[:6]
    return PIP_VALUE_USD.get(clean, PIP_VALUE_USD["DEFAULT"])


def clean_symbol(symbol: str) -> str:
    """Normalize a symbol string for internal use."""
    # Round-14 fix: see backtest/simulator.py — blanket "USDT"->"USD"
    # replace corrupted real FX codes like USDTRY -> USDRY and
    # USDTHB -> USDHB (the "USDT" substring matched mid-string, not
    # just as a Tether-quote suffix). Only strip a trailing "T" when
    # the symbol genuinely ends in "USDT" (e.g. BTCUSDT -> BTCUSD).
    cleaned = str(symbol).upper().replace("/", "").replace("=X", "").strip()
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
MAX_TRADES_PER_DAY: int = _env_int("MAX_TRADES_PER_DAY", 20)
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
# Was duplicated in: trade_permission.MIN_CONFIDENCE_PROD (40),
# live_risk_manager.TIERS.min_confidence (80/70/55),
# autonomous_risk (50).
MIN_CONFIDENCE_PROD: int = _env_int("MIN_CONFIDENCE_PROD", 40)
MIN_CONFIDENCE_TEST: int = _env_int("MIN_CONFIDENCE_TEST", 10)
MIN_CONFIDENCE_TIER_1: float = _env_float("MIN_CONFIDENCE_TIER_1", 80.0)
MIN_CONFIDENCE_TIER_2: float = _env_float("MIN_CONFIDENCE_TIER_2", 70.0)
MIN_CONFIDENCE_TIER_3: float = _env_float("MIN_CONFIDENCE_TIER_3", 55.0)


def get_min_confidence(tier: int = 1) -> float:
    """Return min confidence % for the given tier."""
    return {
        1: MIN_CONFIDENCE_TIER_1,
        2: MIN_CONFIDENCE_TIER_2,
        3: MIN_CONFIDENCE_TIER_3,
    }.get(tier, MIN_CONFIDENCE_TIER_1)


# ── Min Risk:Reward ─────────────────────────────────────────
MIN_RR_PROD: float = _env_float("MIN_RR_PROD", 2.0)
MIN_RR_TEST: float = _env_float("MIN_RR_TEST", 1.0)


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
MAX_LOT_DEFAULT: float = _env_float("MAX_LOT_DEFAULT", 0.20)
TIER_MULT_TIER_1: float = _env_float("TIER_MULT_TIER_1", 0.5)
TIER_MULT_TIER_2: float = _env_float("TIER_MULT_TIER_2", 0.8)
TIER_MULT_TIER_3: float = _env_float("TIER_MULT_TIER_3", 1.0)


# ── Circuit Breaker Thresholds ──────────────────────────────
CB_DAILY_LOSS_TRIGGER_PCT: float = _env_float("CB_DAILY_LOSS_TRIGGER_PCT", 3.0)
CB_CONSECUTIVE_LOSSES_TRIGGER: int = _env_int("CB_CONSECUTIVE_LOSSES_TRIGGER", 3)
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
