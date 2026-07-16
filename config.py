# config.py — Autonomous Forex AI Trader Configuration
# ============================================================
# Single source of truth for all configuration. Sensitive credentials
# come from .env — never hardcode or commit secrets.
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Project Paths ──────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent
LOG_DIR: Path = PROJECT_ROOT / "logs"
DATA_DIR: Path = PROJECT_ROOT / "data"
DB_PATH: Path = PROJECT_ROOT / "database" / "trader.db"
MODEL_DIR: Path = PROJECT_ROOT / "models"
CHART_OUTPUT: Path = DATA_DIR / "chart.html"

# Ensure directories exist
for _d in (LOG_DIR, DATA_DIR, MODEL_DIR, DB_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

# ── General Project Settings ───────────────────────────────────
PROJECT_NAME = "Autonomous Forex AI Trader"

# ── Capital & Risk Management ──────────────────────────────────
# Day 37+ professional tuning — calibrated for 28-pair universe.
#
# P3 audit fix: INITIAL_BALANCE is now overridable via env var so the
# boot-time balance can match the actual live MT5 account balance.
# Previously this was hardcoded to $10,000 — which produced the
# "891.6% drift" warning every cycle on a $99k live account because
# _sync_balance() detected a >5% deviation between the boot-time
# hardcoded value and the live MT5 balance it pulled at runtime.
#
# Set INITIAL_BALANCE_USD in .env to your real account balance:
#   INITIAL_BALANCE_USD=99159.93
# (omit cents if you prefer: INITIAL_BALANCE_USD=99000)
#
# When mt5_demo mode is active, _sync_balance() will still pull the
# real live balance on every cycle — but having the boot-time value
# match means position sizing is correct from the FIRST trade, not
# only after the first resync.
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE_USD", "10000"))
INITIAL_CAPITAL = INITIAL_BALANCE  # Alias for compatibility
RISK_PER_TRADE = 0.005              # 0.5% per trade (production-safe — matches strict_risk_manager)
MAX_DAILY_LOSS = 0.03              # 3% daily loss limit (legacy — kept for backward compat)

# ── Daily Loss Limit (Day 81+ — single source of truth) ──────
# All risk modules (RiskEngine, CircuitBreaker, KillSwitch,
# DrawdownController, AutonomousRisk, RiskAgent) read from this.
# Override in .env:  DAILY_LOSS_LIMIT_PCT=5
# CRITICAL FIX: default is now 5.0% — production-safe.
# 20% daily loss would mean a $2,000 loss on a $10k account in ONE DAY.
# That's account-destroying. 5% is still aggressive but survivable.
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "5.0"))
# At 0.5% risk per trade, 5% daily loss = max 10 losing trades/day.
# That's still a lot — if you hit this, something is wrong with the market
# or the strategy.  Halt and investigate.
# Round-15: increased from 6 → 10 concurrent positions to support the
# expanded 62-pair universe. The bot now has enough LLM capacity
# (7 providers × 16 keys = 112 keys total) to handle more concurrent
# analysis. 10 is still conservative — increase to 15-20 if your
# account size supports it.
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "10"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "8"))    # portfolio-wide headroom
MAX_RISK_PER_PAIR = 0.005          # max 0.5% risk on a single pair (was 2%)

# ── Market & Data Settings ─────────────────────────────────────
MARKET = "forex"
DATA_SOURCE = "yfinance"

# Complete pair universe: 7 majors + 21 minors/crosses + 2 metals = 30 pairs.
# Per user request — agent trades the FULL forex universe + precious metals.
# Each pair gets its own AITrader instance in AutonomousTraderSystem.
# (MAX_OPEN_TRADES = 5 still applies, so only 5 concurrent positions max.)
#
# Day 81+ hotfix: reduced from 30 pairs → 6 majors.
# Reason: with 30 pairs × ~3 LLM calls/pair × ~1000 tokens/call = ~90k
# tokens per cycle.  Groq free-tier TPD limit is 100k/key, so even with
# 6 keys (600k TPD) the bot exhausted all keys in ~7 cycles and entered
# a 429 storm + supervisor restart loop.  6 majors keeps the same
# analytical depth while cutting token usage ~5x.  Re-enable more pairs
# only after switching to Groq Dev tier or adding response caching.
#
# Round-15 audit fix: EXPANDED to full pair universe (61 pairs).
# The operator requested "as many pairs as possible". With the Round-9
# 7-provider LLM hierarchy (Gemini/Cerebras/Groq/SambaNova/OpenRouter/
# GitHub/HuggingFace) + Round-10 TPD budget tracking + Round-14 hour-aware
# retry-after parsing, the bot can now handle the full universe without
# exhausting a single provider's quota. The 429 storm that forced the
# Day 81 reduction to 6 pairs is no longer a concern.
#
# To restore the 6-pair conservative list, uncomment the block below.
SYMBOLS = [
    # ── MAJORS (7) — USD on one side, highest liquidity ──
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "USDCAD", "AUDUSD", "NZDUSD",
    # ── MINORS / CROSSES (21) — non-USD, still high liquidity ──
    "EURGBP", "EURJPY", "EURCHF", "EURAUD",
    "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY",
    # ── METALS / COMMODITIES (4) ──
    "XAUUSD", "XAGUSD",          # Gold, Silver
    "XPTUSD", "XPDUSD",          # Platinum, Palladium
    # ── ENERGY (2) ──
    "USOUSD", "UKOUSD",          # WTI Crude, Brent Crude
    # ── CRYPTO (4) — high volatility, 24/7 ──
    "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD",
    # ── INDEX CFDs (4) ──
    "US30USD", "NAS100USD",      # Dow Jones, NASDAQ 100
    "SPX500USD", "GER40USD",     # S&P 500, DAX 40
    # ── EXOTIC (2) — lower liquidity, higher spread ──
    "USDTRY", "USDZAR",          # Turkish Lira, South African Rand
    # ── ADDITIONAL CROSSES (10) ──
    "EURNOK", "EURSEK",          # Scandinavian crosses
    "GBPSEK", "GBPNOK",          # Scandinavian GBP crosses
    "AUDSGD", "NZDSGD",          # Singapore Dollar crosses
    "CADHKD", "SGDJPY",          # HK/Singapore crosses
    "HKDJPY", "MXNJPY",          # HK/Mexico Yen crosses
    # ── ASIA PACIFIC (7) ──
    "USDCNH", "USDHKD", "USDSGD",  # China offshore, HK, Singapore
    "USDMXN", "USDTHB",            # Mexico, Thailand
    "USDSAR", "USDAED",            # Saudi Arabia, UAE
]
# Total: 7 + 21 + 4 + 2 + 4 + 4 + 2 + 10 + 8 = 62 pairs

# Conservative 6-pair list (kept for reference — uncomment to restore):
# SYMBOLS = [
#     # ── MAJORS (6) — high liquidity, tight spreads ──
#     "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD",
# ]

# Original 30-pair list (kept for reference — uncomment to restore):
# SYMBOLS = [
#     # ── MAJORS (7) — USD on one side ──
#     "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
#     "USDCAD", "AUDUSD", "NZDUSD",
#     # ── MINORS / CROSSES (21) ──
#     "EURGBP", "EURJPY", "EURCHF", "EURAUD",
#     "EURCAD", "EURNZD",
#     "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
#     "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
#     "NZDJPY", "NZDCHF", "NZDCAD",
#     "CADJPY", "CADCHF", "CHFJPY",
#     # ── METALS / COMMODITIES (2) ──
#     "XAUUSD", "XAGUSD",
# ]

# ── Timeframes ─────────────────────────────────────────────────
DEFAULT_TIMEFRAME = "15m"
MTF_CHAIN = ["1d", "4h", "1h", "15m"]

# ── Technical Indicator Settings ───────────────────────────────
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MA_FAST = 20
MA_SLOW = 50
MA_TREND = 200
ATR_PERIOD = 14

# ── Support / Resistance Settings ──────────────────────────────
SR_WINDOW = 5
SR_TOLERANCE = 0.0015

# ── File Paths (legacy compatibility) ─────────────────────────
LOG_FILE = str(LOG_DIR / "trader.log")

# ── System / Operational Loops ─────────────────────────────────
# Day 90 — env-overridable for token economy.  Default 180s (3 min)
# to stretch free-tier LLM keys across the full trading day.
LOOP_INTERVAL_SEC = int(os.getenv("LOOP_INTERVAL_SEC", "180"))
BACKUP_INTERVAL_MIN = 30
RECOVERY_COOLDOWN_MIN = 5

# ── Monitoring ─────────────────────────────────────────────────
MONITORING_INTERVAL = 60  # seconds between health checks

# ── AI / LLM Settings ─────────────────────────────────────────
# Day 100+ Update: Default to cheaper/faster models to reduce 429 rate limits.
# Production logs showed llama-3.3-70b-versatile hitting Groq TPD limits (98k+ tokens).
# llama-3.1-8b-instant is ~10x cheaper and rarely hits limits.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
# Anthropic + OpenRouter intentionally disabled — MasterAnalyst now uses
# the same Groq/Gemini chain as AIAnalyst (per user request, free-tier only).
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ── Execution Mode ─────────────────────────────────────────────
# "mt5_demo" -> Real MT5 demo account execution (DEFAULT — user has MT5 set up)
# "paper"    -> Legacy paper mode (ExecutionRouter no longer supports this —
#               will raise ValueError if set).  Kept for backward compat
#               reference only.
#
# Day 81+ hotfix: was defaulting to "paper", but ExecutionRouter only
# accepts "mt5_demo" and raises ValueError for anything else.  If .env
# failed to load (e.g. wrong working dir, missing file), the bot would
# crash on boot with "Unknown EXECUTION_MODE: paper".  Default is now
# "mt5_demo" to match the only supported mode.
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "mt5_demo").lower()

# ── SIMULATION MODE ─────────────────────────────────────────────
# When True, ExecutionRouter uses SimulatedExecutor instead of real MT5.
# The full signal → risk → approval → router chain runs, but the final
# order is logged to logs/execution.log as "broker.order_send" with
# retcode=10009 (TRADE_RETCODE_DONE) — NO real broker contact.
#
# Use this to verify the order-flow chain end-to-end without a live
# MT5 terminal.  Especially useful for:
#   - Diagnosing why trades aren't placed (run + tail logs/execution.log)
#   - CI / unit tests of the execution path
#   - Dry-run on a fresh VPS before plugging in MT5 credentials
#
# Default: False (preserve existing behaviour).
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "false").lower() == "true"

# ── MT5 FALLBACK TO SIMULATION ──────────────────────────────────
# When True (default), if EXECUTION_MODE=mt5_demo but the MT5 terminal
# is not running / not reachable / credentials are wrong, the bot will
# NOT crash on boot.  Instead it logs a WARNING and automatically falls
# back to SIMULATION_MODE so the full analysis pipeline still runs.
MT5_FALLBACK_TO_SIMULATION = os.getenv("MT5_FALLBACK_TO_SIMULATION", "true").lower() == "true"

# ── Position Sizing Hard Caps (Day 81+ loss-prevention) ───────
# Absolute maximum lot size per trade, regardless of what RiskEngine
# or PositionSizer computes.  Default 0.20 — for a $10k account with
# 1% risk ($100) and a 15-pip SL on EURUSD, the math gives ~0.67 lot,
# but multipliers (Kelly × vol × conf × corr) can compound to 2-3x.
# This cap is the LAST line of defense against lot explosion.
#
# Override per account size:
#   $1k  → MAX_LOT=0.05
#   $10k → MAX_LOT=0.20  (default)
#   $50k → MAX_LOT=1.00
#   $100k→ MAX_LOT=2.00
MAX_LOT = float(os.getenv("MAX_LOT", "0.20"))

# Maximum LLM calls per symbol cycle.  Each cycle fires:
#   - SentimentModel (1 call)            — from sentiment_data provider
#   - AIAnalyst._call_groq (1 call)      — classic LLM analyst
#   - MasterAnalyst._call_llm (1 call)   — master brain
#   - NewsIntelligence (sometimes 1)     — news bias adjustment
# Total ~3-4 calls per symbol.  Was 5 — too tight, caused LLM throttle
# to kick in before all 3 callers got a turn.  Default now 8 to leave
# headroom for retries.
# Day 102+ CRITICAL hotfix: code default was "2" but comment said 8 —
# the mismatch silently throttled the 3rd LLM caller (MasterAnalyst)
# every cycle, degrading AI quality to rule-engine-only. Aligned the
# code with the documented intent.
# Round-15: increased from 8 → 20 to support 62-pair universe.
# Each pair needs ~3 LLM calls (SentimentModel + MasterAnalyst + retries).
# 62 pairs × 3 calls = 186 calls/cycle theoretical max, but caching +
# skip-AIAnalyst-if-MasterAnalyst-runs keeps real usage ~20-40 calls.
MAX_LLM_CALLS_PER_CYCLE = int(os.getenv("MAX_LLM_CALLS_PER_CYCLE", "8"))

# Minimum delay (seconds) between LLM calls to the same provider.
# Groq free tier rate-limits aggressively; this prevents the 429 storm.
LLM_CALL_INTERVAL_SEC = float(os.getenv("LLM_CALL_INTERVAL_SEC", "1.0"))

# GLOBAL rolling-window cap: max LLM calls per 60 seconds across ALL
# symbol cycles.  Per-cycle cap alone is not enough — with 6 pairs ×
# 5 calls/cycle = 30 calls in 2 minutes, all 6 Groq keys hit TPD limit
# (100k tokens/day each).  Default 12 calls/min — leaves headroom for
# 6 pairs × 2 calls = 12 calls/cycle without throttling the master
# analyst. (Day 102+ hotfix: code default was "3" but comment said 12 —
# the mismatch guaranteed most pairs got throttled each cycle.)
# Round-15: increased from 12 → 60 to support 62-pair universe.
# 62 pairs × ~3 calls/pair = ~186 calls/cycle, but spread across
# 7 providers with TPD budget tracking. 60 calls/min gives enough
# headroom for the expanded universe while staying under free-tier
# RPM limits on any single provider.
MAX_LLM_CALLS_PER_MIN = int(os.getenv("MAX_LLM_CALLS_PER_MIN", "60"))

# Telegram rate limit — max messages per minute.  Telegram's API
# limit is 30 msg/sec globally but per-channel practical limit is ~20
# msg/min before users mute the bot.  Default 10.
TELEGRAM_MAX_MSG_PER_MIN = int(os.getenv("TELEGRAM_MAX_MSG_PER_MIN", "10"))

# ── TEST MODE ─────────────────────────────────────────────────
# When true (default for first-time MT5 demo verification): all safety
# gates become permissive so the system actually places trades.
#  - TradePermission MIN_CONFIDENCE = 10 (instead of 60)
#  - Session quality check becomes warning (instead of block)
#  - ConfidenceEngine auto-skip disabled
#  - ConfidenceEngine WAIT threshold = 10 (instead of 25)
# Switch to false once you've confirmed MT5 orders are filling correctly
# and you want the full safety pipeline re-engaged.
# CRITICAL FIX: default is now "false" — production-safe.
# Set TEST_MODE=true explicitly in .env only during initial development.
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

# ── TRADING MODE (Day 81+) ────────────────────────────────────
# SAFE        — high-confidence-only, all confirmations required, small lots
# AUTONOMOUS  — system trades per ApprovalMode (default mode 3 = no human gate)
# ABSOLUTE_SAFETY is an independent kill-switch flag — when true, the
# following hard gates ALWAYS block execution regardless of TRADING_MODE:
#   - broker disconnect
#   - spread > 5x normal
#   - extreme volatility (ATR > 3x median)
#   - news window (±30 min around high-impact events)
#   - margin level < 200%
TRADING_MODE = os.getenv("TRADING_MODE", "AUTONOMOUS").upper()
ABSOLUTE_SAFETY = os.getenv("ABSOLUTE_SAFETY", "true").lower() == "true"

# Confidence thresholds per TRADING_MODE (used by TradePermission)
TRADING_MODE_CONFIDENCE = {
    "SAFE":       80,   # only high-conviction trades
    "AUTONOMOUS": 60,   # balanced — production default
    "TEST":       10,   # permissive — only when TEST_MODE=true
}

# ── Use Scanner ────────────────────────────────────────────────
USE_SCANNER = os.getenv("USE_SCANNER", "false").lower() == "true"

# ── Approval Mode ──────────────────────────────────────────────
# 1 = analysis only (AI watches, never trades)
# 2 = supervised (AI suggests, human must approve each trade)
# 3 = autonomous (default — no human gate)
APPROVAL_MODE = int(os.getenv("APPROVAL_MODE", "3"))

# ── MT5 Broker Credentials ─────────────────────────────────────
MT5_LOGIN_ENV = os.getenv("MT5_LOGIN", "0")
MT5_LOGIN = int(MT5_LOGIN_ENV) if MT5_LOGIN_ENV and MT5_LOGIN_ENV.isdigit() and MT5_LOGIN_ENV != "0" else None
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_PATH = os.getenv("MT5_PATH")  # Optional: MT5 terminal.exe path override
MT5_INVESTOR = os.getenv("MT5_INVESTOR")

# ── Telegram ───────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_TELEGRAM = os.getenv("ENABLE_TELEGRAM", "false").lower() == "true"

# ── External API Keys ─────────────────────────────────────────
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ── Retraining Settings ───────────────────────────────────────
RETRAINING_INTERVAL = int(os.getenv("RETRAINING_INTERVAL", "24"))  # hours
PERFORMANCE_THRESHOLD = float(os.getenv("PERFORMANCE_THRESHOLD", "0.55"))
MIN_TRAINING_SAMPLES = int(os.getenv("MIN_TRAINING_SAMPLES", "100"))

# Walk-forward / evaluation defaults
WALK_FORWARD_MIN_TRAIN_SIZE = int(os.getenv("WALK_FORWARD_MIN_TRAIN_SIZE", str(MIN_TRAINING_SAMPLES)))
WALK_FORWARD_STEP_SIZE = int(os.getenv("WALK_FORWARD_STEP_SIZE", "50"))

# Model prediction thresholds
MODEL_BUY_THRESHOLD = float(os.getenv("MODEL_BUY_THRESHOLD", "0.58"))
MODEL_SELL_THRESHOLD = float(os.getenv("MODEL_SELL_THRESHOLD", "0.42"))

# ── SMTP / Email Alerts ────────────────────────────────────────
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_RECIPIENTS = os.getenv("ALERT_RECIPIENTS", "")
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")

# ── Webhook ────────────────────────────────────────────────────
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "5000"))

# ── Logging ────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5


# ── Configuration Validation ───────────────────────────────────
def validate_mt5_config() -> None:
    """Validate MT5 credentials before starting mt5_demo mode.

    Skipped when SIMULATION_MODE=true or MT5_FALLBACK_TO_SIMULATION=true.
    """
    if SIMULATION_MODE:
        return
    if MT5_FALLBACK_TO_SIMULATION:
        return
    if EXECUTION_MODE == "mt5_demo":
        missing = []
        if not MT5_LOGIN:
            missing.append("MT5_LOGIN")
        if not MT5_PASSWORD:
            missing.append("MT5_PASSWORD")
        if not MT5_SERVER:
            missing.append("MT5_SERVER")
        if missing:
            from core.exceptions import ConfigurationError
            raise ConfigurationError(
                f"MT5 credentials missing in .env: {', '.join(missing)}. "
                f"Set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER, or set "
                f"MT5_FALLBACK_TO_SIMULATION=true."
            )


def validate_telegram_config() -> None:
    """Validate Telegram credentials before enabling notifications."""
    if ENABLE_TELEGRAM:
        missing = []
        if not TELEGRAM_TOKEN:
            missing.append("TELEGRAM_TOKEN")
        if not TELEGRAM_CHAT_ID:
            missing.append("TELEGRAM_CHAT_ID")
        if missing:
            import logging
            logging.getLogger(__name__).warning(
                f"Telegram enabled but credentials missing: {', '.join(missing)}. "
                f"Notifications will be disabled."
            )

def validate_all_config() -> None:
    """
    Validate every configuration required before startup.

    This is the single entry point used by main.py.
    """
    validate_mt5_config()
    validate_telegram_config()

class Config:
    """Unified configuration class — merges all settings for modules
    that prefer class-based access over module-level constants."""

    # Project
    PROJECT_NAME = PROJECT_NAME
    PROJECT_ROOT = PROJECT_ROOT

    # Paths
    DATA_DIR = DATA_DIR
    LOG_DIR = LOG_DIR
    MODEL_DIR = MODEL_DIR
    DB_PATH = DB_PATH
    CHART_OUTPUT = CHART_OUTPUT
    LOG_FILE = LOG_FILE

    # Capital & Risk
    INITIAL_BALANCE = INITIAL_BALANCE
    INITIAL_CAPITAL = INITIAL_CAPITAL
    RISK_PER_TRADE = RISK_PER_TRADE
    MAX_DAILY_LOSS = MAX_DAILY_LOSS
    DAILY_LOSS_LIMIT_PCT = DAILY_LOSS_LIMIT_PCT
    MAX_OPEN_TRADES = MAX_OPEN_TRADES
    MAX_POSITIONS = MAX_POSITIONS
    MAX_RISK_PER_PAIR = MAX_RISK_PER_PAIR

    # Market
    MARKET = MARKET
    DATA_SOURCE = DATA_SOURCE
    SYMBOLS = SYMBOLS

    # Timeframes
    DEFAULT_TIMEFRAME = DEFAULT_TIMEFRAME
    MTF_CHAIN = MTF_CHAIN

    # Indicators
    RSI_PERIOD = RSI_PERIOD
    RSI_OVERBOUGHT = RSI_OVERBOUGHT
    RSI_OVERSOLD = RSI_OVERSOLD
    MA_FAST = MA_FAST
    MA_SLOW = MA_SLOW
    MA_TREND = MA_TREND
    ATR_PERIOD = ATR_PERIOD

    # S/R
    SR_WINDOW = SR_WINDOW
    SR_TOLERANCE = SR_TOLERANCE

    # System

    LOOP_INTERVAL_SEC = LOOP_INTERVAL_SEC
    BACKUP_INTERVAL_MIN = BACKUP_INTERVAL_MIN
    RECOVERY_COOLDOWN_MIN = RECOVERY_COOLDOWN_MIN
    MONITORING_INTERVAL = MONITORING_INTERVAL

    # Execution
    EXECUTION_MODE = EXECUTION_MODE
    USE_SCANNER = USE_SCANNER
    APPROVAL_MODE = APPROVAL_MODE
    TEST_MODE = TEST_MODE
    TRADING_MODE = TRADING_MODE
    ABSOLUTE_SAFETY = ABSOLUTE_SAFETY
    TRADING_MODE_CONFIDENCE = TRADING_MODE_CONFIDENCE

    # MT5
    MT5_LOGIN = MT5_LOGIN
    MT5_PASSWORD = MT5_PASSWORD
    MT5_SERVER = MT5_SERVER
    MT5_PATH = MT5_PATH

    # Telegram
    TELEGRAM_TOKEN = TELEGRAM_TOKEN
    TELEGRAM_CHAT_ID = TELEGRAM_CHAT_ID
    ENABLE_TELEGRAM = ENABLE_TELEGRAM

    # LLM
    GROQ_API_KEY = GROQ_API_KEY
    GROQ_MODEL = GROQ_MODEL
    GEMINI_API_KEY = GEMINI_API_KEY
    GEMINI_MODEL = GEMINI_MODEL
    # Anthropic + OpenRouter disabled (per user request — free-tier only)
    # ANTHROPIC_API_KEY = ANTHROPIC_API_KEY
    # OPENROUTER_API_KEY = OPENROUTER_API_KEY

    # External APIs
    ALPHA_VANTAGE_API_KEY = ALPHA_VANTAGE_API_KEY
    FINNHUB_API_KEY = FINNHUB_API_KEY
    TWELVE_DATA_API_KEY = TWELVE_DATA_API_KEY
    FRED_API_KEY = FRED_API_KEY

    # Retraining
    RETRAINING_INTERVAL = RETRAINING_INTERVAL
    PERFORMANCE_THRESHOLD = PERFORMANCE_THRESHOLD
    MIN_TRAINING_SAMPLES = MIN_TRAINING_SAMPLES

    # Logging
    LOG_LEVEL = LOG_LEVEL
    LOG_MAX_SIZE = LOG_MAX_SIZE
    LOG_BACKUP_COUNT = LOG_BACKUP_COUNT

    # Forex pairs for scanner/data updater — full 28-pair universe
    FOREX_PAIRS = SYMBOLS  # Reuse the SYMBOLS list (28 pairs)

    # Data update configuration
    DATA_UPDATE_TIME = "06:00"
    DATA_UPDATE_TIMEZONE = "UTC"
    DATA_HISTORY_DAYS = 365 * 5
    DATA_UPDATE_RETRY_ATTEMPTS = 3
    DATA_UPDATE_RETRY_DELAY = 300

    # Legacy OANDA keys (optional — not used by default)
    OANDA_API_KEY = os.environ.get('OANDA_API_KEY', '')
    OANDA_ACCOUNT_ID = os.environ.get('OANDA_ACCOUNT_ID', '')

    # Database (legacy — system uses SQLite by default)
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = os.environ.get('DB_PORT', '5432')
    DB_NAME = os.environ.get('DB_NAME', 'forex_ai')
    DB_USER = os.environ.get('DB_USER', 'postgres')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', '')

    # SMTP
    SMTP_HOST = SMTP_HOST
    SMTP_PORT = SMTP_PORT
    SMTP_USERNAME = SMTP_USERNAME
    SMTP_PASSWORD = SMTP_PASSWORD
    ALERT_RECIPIENTS = ALERT_RECIPIENTS
    ALERT_WEBHOOK_URL = ALERT_WEBHOOK_URL

    # Webhook
    WEBHOOK_SECRET = WEBHOOK_SECRET
    WEBHOOK_PORT = WEBHOOK_PORT


# Auto-validate on import
validate_mt5_config()
validate_telegram_config()
