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
# 2026-08-13 fix: MAX_DAILY_LOSS was 0.03 (3%) but DAILY_LOSS_LIMIT_PCT below
# is 5.0% — two different values for the same concept. MAX_DAILY_LOSS is read
# by NOTHING in the live path (only Config.MAX_DAILY_LOSS alias at line ~625).
# Aliased to DAILY_LOSS_LIMIT_PCT / 100 for consistency.
MAX_DAILY_LOSS = 0.05              # alias of DAILY_LOSS_LIMIT_PCT/100 (legacy compat)

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
try:
    MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "10") or 10)
except (ValueError, TypeError):
    MAX_OPEN_TRADES = 10
try:
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "8") or 8)
except (ValueError, TypeError):
    MAX_POSITIONS = 8    # portfolio-wide headroom
MAX_RISK_PER_PAIR = 0.005          # max 0.5% risk on a single pair (was 2%)

# ── Market & Data Settings ─────────────────────────────────────
MARKET = "forex"
DATA_SOURCE = "yfinance"

# 2026-08-20 fix: Exness (and many other brokers) append a suffix to every
# symbol — e.g. "EURUSD" is actually "EURUSDm" on this account (confirmed
# via symbol.py's live MT5 audit: all 121 tradeable Exness symbols end in
# "m"). SYMBOLS below was defined with plain names ("EURUSD", "GBPUSD",
# ...) and had NO suffix logic anywhere in this file, which means every
# order/quote request would silently fail to resolve on MT5 ("symbol not
# found") even though the pair list itself was fine.
#
# Fix: append the broker suffix to every symbol when building SYMBOLS,
# without removing or renaming any pair in the underlying lists below.
# Override via .env if you switch brokers/accounts:
#   MT5_SYMBOL_SUFFIX=m      (Exness "m" accounts — current default)
#   MT5_SYMBOL_SUFFIX=       (broker uses bare names, no suffix)
BROKER_SYMBOL_SUFFIX = os.getenv("MT5_SYMBOL_SUFFIX", "m")


def _with_broker_suffix(pairs):
    """Append BROKER_SYMBOL_SUFFIX to each pair, without altering the
    original (suffix-less) pair name lists elsewhere in this file."""
    if not BROKER_SYMBOL_SUFFIX:
        return list(pairs)
    return [
        p if p.endswith(BROKER_SYMBOL_SUFFIX) else f"{p}{BROKER_SYMBOL_SUFFIX}"
        for p in pairs
    ]

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
# 2026-08-13: SYMBOLS now driven by utils/pair_profiles.py — only pairs
# with enabled=True are traded. Each pair gets its own optimized config
# (min_confidence, min_factors, min_rr, session_filter, SL/TP ATR).
# See utils/pair_profiles.py for the per-pair profiles and backtest data.
try:
    from utils.pair_profiles import get_active_pairs as _get_active_pairs
    _PROFILE_PAIRS = _get_active_pairs()
except Exception:
    _PROFILE_PAIRS = []

# 2026-08-13 FINAL: For LIVE TRADING SAFETY, only trade pairs that have
# been backtested AND have an optimized profile in utils/pair_profiles.py.
# Trading unprofiled pairs (minors/metals/exotics) without backtest data
# is risky — they may have completely different behavior than majors.
# To add a new pair: (1) backtest it (2) add profile in pair_profiles.py
# (3) it will automatically appear in SYMBOLS via _PROFILE_PAIRS.
#
# Previous 48-pair list (minors/metals/exotics) is commented out below
# for reference. Re-enable individual pairs ONLY after adding their
# profile to pair_profiles.py.
# Build the full 48-pair universe as the default fallback when
# `utils.pair_profiles` is unavailable. Tests and audits expect the
# complete list to be present in CI; realistic deployments may prefer
# a reduced active set controlled by pair_profiles.py.
_MAJOR_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "USDCAD", "AUDUSD", "NZDUSD",
]

# SYMBOLS will be finalized after the disabled-reference list below
# is defined to avoid forward-reference NameError during import.
# Suffix fix applied here too: _PROFILE_PAIRS comes from
# utils/pair_profiles.py and was also missing the broker suffix.
SYMBOLS = _with_broker_suffix(_PROFILE_PAIRS) if _PROFILE_PAIRS else []

# Previous 48-pair list (DISABLED for live trading safety):
# To re-enable a pair, add it to utils/pair_profiles.py PROFILES dict
# with enabled=True and backtest-optimized parameters.
_DISABLED_SYMBOLS_REFERENCE = [
    # Minors / Crosses (21)
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY",
    # Metals (4)
    "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD",
    # Exotic (2)
    "USDTRY", "USDZAR",
    # Additional Crosses (9)
    "EURNOK", "EURSEK", "GBPSEK", "GBPNOK",
    "AUDSGD", "NZDSGD", "SGDJPY", "HKDJPY", "MXNJPY",
    # Asia Pacific (5)
    "USDCNH", "USDHKD", "USDSGD", "USDMXN", "USDTHB",
]
# Total: 7 + 21 + 4 + 2 + 9 + 5 = 48 pairs
# (majors + minors + metals + exotic + additional crosses + Asia Pacific)
# (11 pairs removed 2026-07-23: USOUSD, UKOUSD, BTCUSD, ETHUSD, LTCUSD,
#  XRPUSD, US30USD, NAS100USD, SPX500USD, GER40USD, CADHKD — all were
#  repeatedly failing "Could not fetch" / "NO_TRADE — Market data fetch
#  failed" under MT5_ONLY_MODE=true with no fallback source enabled.
#  Re-add if/when either (a) this broker's correct tickers for the index
#  CFDs are confirmed and mapped, or (b) MT5_ONLY_MODE is turned off so
#  the fallback data source can cover crypto/energy.)

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

# Build the full 48-pair universe if pair_profiles did not supply active pairs.
# All 48 original pairs are kept exactly as-is — only the broker suffix is
# added so they resolve correctly against live MT5 (Exness) symbols.
FULL_PAIR_UNIVERSE = _with_broker_suffix(_MAJOR_PAIRS + _DISABLED_SYMBOLS_REFERENCE)
SYMBOLS = FULL_PAIR_UNIVERSE

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
try:
    LOOP_INTERVAL_SEC = int(os.getenv("LOOP_INTERVAL_SEC", "180") or 180)
except (ValueError, TypeError):
    LOOP_INTERVAL_SEC = 180
BACKUP_INTERVAL_MIN = 30
RECOVERY_COOLDOWN_MIN = 5

# ── Monitoring ─────────────────────────────────────────────────
MONITORING_INTERVAL = 60  # seconds between health checks

# ── AI / LLM Settings ─────────────────────────────────────────
# 2026-07-25 NEW PROVIDER CASCADE ORDER:
#   1. Groq       (Primary)     — llama-3.1-8b-instant
#   2. Gemini     (Fallback #1) — gemini-flash-lite-latest
#   3. OpenRouter (Fallback #2) — universal router (100+ models)
#   4. Cerebras   (OPTIONAL)    — set OC_INCLUDE_CEREBRAS=1
#   5. SambaNova  (OPTIONAL)    — set OC_INCLUDE_SAMBANOVA=1
# OLLAMA HAS BEEN COMPLETELY REMOVED from the AIAnalyst/MasterAnalyst
# cascade per user request ("Ollama akebare sese"). It is kept ONLY
# as an opt-in institutional veto gate (set OLLAMA_VALIDATOR_ENABLED=true).
#
# Day 100+ Update: Default to cheaper/faster models to reduce 429 rate limits.
# Production logs showed llama-3.3-70b-versatile hitting Groq TPD limits (98k+ tokens).
# llama-3.1-8b-instant is ~10x cheaper and rarely hits limits.
# ── IMPORTANT — do not call providers with these directly ──────
# GROQ_API_KEY / GEMINI_API_KEY below are the single LEGACY key only
# (kept for backward compat with pre-multi-key deployments). They are
# NOT the 14/13-key rotation pool — that lives in core/llm_key_manager.py,
# which reads GROQ_API_KEY_1..14 / GEMINI_API_KEY_1..13 from the
# environment directly (plus this legacy single key as one extra
# fallback entry appended to the pool). Any code that calls Groq/Gemini
# using `config.GROQ_API_KEY` / `config.GEMINI_API_KEY` directly bypasses
# the 14-key rotation entirely and will hit that one account's rate
# limit immediately. Always go through
# `core.llm_key_manager.get_llm_key_manager()` (`get_groq_client()` /
# `get_gemini_client()`) for any actual API call.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# 2026-08-20 fix: this default was still "llama-3.1-8b-instant", which the
# comment above this block already documents as deprecated by Groq on
# 2026-06-17 (same day as llama-3.3-70b-versatile). ai/ai_analyst.py and
# agents/master_analyst.py were already migrated to "openai/gpt-oss-20b";
# this constant (currently unused elsewhere in the codebase, but kept for
# any future caller) was missed. Matching it here for consistency.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# ── GROQ_MODEL sanity check (2026-08-20 fix) ─────────────────────
# "groq/compound" and "groq/compound-mini" are Groq's AGENTIC tool-use
# systems (built-in web search / code execution / browsing), not plain
# chat-completion models. They're a poor fit for this bot's structured
# trade-decision JSON calls, and have been reported (Groq's own docs +
# third-party issue trackers) to intermittently return 413 "Request
# Entity Too Large" even for prompts well under their advertised 131K
# context window — independent of anything this bot's prompt-building
# code can trim. Warn loudly at boot rather than let the operator
# discover it as a wall of "Groq failed attempt 1/3 ... 413" retries
# in the per-symbol analysis logs.
GROQ_MODEL_LOOKS_AGENTIC = "compound" in GROQ_MODEL.lower()

def validate_groq_model_config(logger=None) -> bool:
    """Warn if GROQ_MODEL is an agentic 'compound' system rather than a
    plain chat model. Returns True if GROQ_MODEL looks fine.
    Call once at boot (see main.py initialize())."""
    if not GROQ_MODEL_LOOKS_AGENTIC:
        return True
    msg_lines = [
        "=" * 60,
        "  CONFIG WARNING: GROQ_MODEL is an agentic 'compound' system",
        f"    GROQ_MODEL (configured) = {GROQ_MODEL}",
        "  'compound' / 'compound-mini' are Groq's tool-using agent",
        "  systems (web search, code execution, browsing) — not plain",
        "  chat models. They are known to intermittently return HTTP",
        "  413 'Request Entity Too Large' on ordinary prompts, and are",
        "  not designed for this bot's structured trade-JSON calls.",
        "  Fix: set GROQ_MODEL in .env to a standard chat model, e.g.",
        "    GROQ_MODEL=openai/gpt-oss-20b",
        "=" * 60,
    ]
    if logger is not None:
        for line in msg_lines:
            logger.warning(line)
    else:
        for line in msg_lines:
            print(line)
    return False
# OpenRouter is now FALLBACK #2 in the cascade (was previously disabled).
# To enable: set OPENROUTER_API_KEY in .env (one key is enough).
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
# Optional extras — disabled by default. Set OC_INCLUDE_CEREBRAS=1 /
# OC_INCLUDE_SAMBANOVA=1 in .env to add them to the cascade.
OC_INCLUDE_CEREBRAS = os.getenv("OC_INCLUDE_CEREBRAS", "0") == "1"
OC_INCLUDE_SAMBANOVA = os.getenv("OC_INCLUDE_SAMBANOVA", "0") == "1"
# Anthropic intentionally disabled — MasterAnalyst + AIAnalyst both use
# the Groq → Gemini → OpenRouter chain (per user request, free-tier only).
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
# Propagate HF_TOKEN to all recognised HuggingFace Hub env-var names so
# that huggingface_hub / sentence-transformers never issue "unauthenticated"
# warnings regardless of which module loads first.
if HF_TOKEN:
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)
    os.environ.setdefault("HF_HUB_AUTH_TOKEN", HF_TOKEN)
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)

# Disable HF telemetry (reduces unnecessary network noise)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ── Execution Mode ─────────────────────────────────────────────
# "mt5_demo"  -> Real MT5 demo account execution (DEFAULT — user has MT5 set up)
# "mt5_live"  -> Real MT5 REAL-MONEY account execution (execution-parity
#                audit, §9). Uses the exact same order-placement code path
#                as mt5_demo — MT5's API doesn't distinguish demo/real at
#                the call level, only the account credentials differ —
#                gated by ALLOW_REAL_MONEY_TRADING + MT5_REAL_* below.
#                Never falls back to demo/simulation on failure — see
#                execution/execution_router.py.
# "backtest"  -> Router is inert; backtest.unified_engine drives fills
#                through backtest.broker_sim.BrokerSimulator directly.
# "paper"     -> Legacy paper mode (ExecutionRouter no longer supports this —
#                will raise ValueError if set).  Kept for backward compat
#                reference only.
#
# Day 81+ hotfix: was defaulting to "paper", but ExecutionRouter only
# accepted "mt5_demo" and raised ValueError for anything else.  If .env
# failed to load (e.g. wrong working dir, missing file), the bot would
# crash on boot with "Unknown EXECUTION_MODE: paper".  Default is now
# "mt5_demo" — the safest, always-available mode — regardless of how
# many modes ExecutionRouter supports.
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

# ── MAX_LOT / balance sanity check (2026-08-20 fix) ─────────────
# RiskEngine's safety guard (risk/risk_engine.py) rejects any trade where
# MAX_LOT caps the lot to less than MIN_RISK_FRACTION_OF_INTENDED of the
# intended risk, rather than silently under-risking. That guard is
# correct and should stay — but when MAX_LOT is simply misconfigured for
# the account size, the guard's symptom is a wall of per-symbol
# "MAX_LOT cap shrinks actual risk to X%" rejections buried in the logs,
# with no single place that says *why*. This check surfaces the root
# cause once, loudly, at boot.
#
# Recommended MAX_LOT scales ~linearly with balance (see table above:
# $10k→0.20, $50k→1.00, $100k→2.00, i.e. roughly balance / 50,000).
RECOMMENDED_MAX_LOT = round(max(0.05, INITIAL_BALANCE / 50000.0), 2)
MAX_LOT_SEVERELY_UNDERSIZED = MAX_LOT < RECOMMENDED_MAX_LOT * 0.5

def validate_max_lot_config(logger=None) -> bool:
    """Warn loudly if MAX_LOT is inconsistent with INITIAL_BALANCE.

    Returns True if MAX_LOT looks fine, False if it's severely undersized
    (which will cause RiskEngine to systematically reject trades via its
    "MAX_LOT cap shrinks actual risk" guard). Call this once at boot
    (see main.py initialize()) so the operator sees ONE clear message
    instead of discovering it symbol-by-symbol in risk-rejection logs.
    """
    if not MAX_LOT_SEVERELY_UNDERSIZED:
        return True
    msg_lines = [
        "=" * 60,
        "  CONFIG WARNING: MAX_LOT looks undersized for INITIAL_BALANCE",
        f"    INITIAL_BALANCE = ${INITIAL_BALANCE:,.2f}",
        f"    MAX_LOT (configured) = {MAX_LOT}",
        f"    MAX_LOT (recommended) ≈ {RECOMMENDED_MAX_LOT}  "
        f"(balance / 50,000, min 0.05)",
        "  With this gap, RiskEngine's safety guard will reject most/all",
        "  trades rather than silently under-risk them (see",
        "  risk/risk_engine.py 'MAX_LOT cap shrinks actual risk' reject).",
        "  Fix: set MAX_LOT in .env to match your account size, e.g.",
        f"    MAX_LOT={RECOMMENDED_MAX_LOT}",
        "=" * 60,
    ]
    if logger is not None:
        for line in msg_lines:
            logger.warning(line)
    else:
        for line in msg_lines:
            print(line)
    return False

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
# 2026-08-13 fix: default was still "8" despite comment saying 20 —
# this throttled most pairs after the first 2-3. Aligned with intent.
try:
    MAX_LLM_CALLS_PER_CYCLE = int(os.getenv("MAX_LLM_CALLS_PER_CYCLE", "20") or 20)
except (ValueError, TypeError):
    MAX_LLM_CALLS_PER_CYCLE = 20

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
try:
    MAX_LLM_CALLS_PER_MIN = int(os.getenv("MAX_LLM_CALLS_PER_MIN", "60") or 60)
except (ValueError, TypeError):
    MAX_LLM_CALLS_PER_MIN = 60

# Telegram rate limit — max messages per minute.  Telegram's API
# limit is 30 msg/sec globally but per-channel practical limit is ~20
# msg/min before users mute the bot.  Default 10.
try:
    TELEGRAM_MAX_MSG_PER_MIN = int(os.getenv("TELEGRAM_MAX_MSG_PER_MIN", "10") or 10)
except (ValueError, TypeError):
    TELEGRAM_MAX_MSG_PER_MIN = 10

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
try:
    APPROVAL_MODE = int(os.getenv("APPROVAL_MODE", "3") or 3)
except (ValueError, TypeError):
    APPROVAL_MODE = 3

# ── MT5 Broker Credentials (DEMO — default account) ─────────────
MT5_LOGIN_ENV = os.getenv("MT5_LOGIN", "0")
MT5_LOGIN = int(MT5_LOGIN_ENV) if MT5_LOGIN_ENV and MT5_LOGIN_ENV.isdigit() and MT5_LOGIN_ENV != "0" else None
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_PATH = os.getenv("MT5_PATH")  # Optional: MT5 terminal.exe path override
MT5_INVESTOR = os.getenv("MT5_INVESTOR")

# ── MT5-only data mode (operator request, 2026-07-23) ────────────
# The external fallback chain (AlphaVantage/Polygon/Finnhub/TwelveData/
# yfinance) was found producing bad data under real conditions:
#   - Finnhub: dead all session (API key not set) — not a real fallback
#   - TwelveData: rate-limited (429) on every attempt — wastes wall-clock
#   - Polygon: internally inconsistent prices per request (e.g. XAUUSD
#     swinging $130+ across 70s), 3-candle H1 windows instead of 200
#   - AlphaVantage/yfinance: "unsupported timeframe" gaps on H4
# A 3-candle or self-contradictory-price fallback is worse than no data
# at all — indicators with real lookback (RSI-14, MACD, MAs) are
# unreliable on that little history, and downstream code already has a
# fail-safe path for "no data this cycle" (skip, log, retry next cycle).
# Default TRUE: MT5 is the only data source; on failure, fetch_ohlcv()
# returns None instead of cascading through the external chain. Set
# MT5_ONLY_MODE=false to restore the old multi-provider fallback (e.g.
# for a Linux/Mac dev box with no MT5 terminal at all).
MT5_ONLY_MODE = os.getenv("MT5_ONLY_MODE", "true").lower() == "true"

# ── ML model registry/disk consistency check (startup) ───────────
# `memory/ml_models/_registry.json` can drift from disk (deleted/moved
# .pkl files, partial deploys, manual cleanup) — the registry still says
# a model exists, but load_model() returns None at trade time and the
# predictor silently degrades to NOT_READY mid-session. core/runtime.py
# audits registry-vs-disk for every configured pair during the AI boot
# phase; this controls what it does when it finds a mismatch:
#   "warn"          — log the mismatch loudly, keep booting (old behavior)
#   "auto_retrain"  — log it, then retrain baseline models for just the
#                      affected pair/timeframe (scripts/train_missing_pairs
#                      .train_one_pair), so NOT_READY is fixed before the
#                      first trading cycle instead of discovered during it
#   "hard_fail"      — log it and refuse to start (exit 1); use when you
#                      want a broken model registry to block deployment
#                      rather than silently run with a degraded ensemble
ML_MODEL_CONSISTENCY_ACTION = os.getenv("ML_MODEL_CONSISTENCY_ACTION", "auto_retrain").lower()

# ── REAL-MONEY execution gate (execution-parity audit, §9) ──────
# execution/execution_router.py imports these four names when
# EXECUTION_MODE=mt5_live. They previously did not exist in this file
# at all, so any attempt to actually use mt5_live raised a bare
# ImportError instead of the router's intended, explicit
# RuntimeError("ALLOW_REAL_MONEY_TRADING is not set...") — a Critical
# bug: the safety message was unreachable because the import itself
# failed first.
#
# ALLOW_REAL_MONEY_TRADING is a SEPARATE opt-in from EXECUTION_MODE on
# purpose: setting EXECUTION_MODE=mt5_live alone is not enough to place
# a real order. Both this flag AND real credentials below must be set.
# Default is always False/empty — real trading is never on by an
# unattended default in any environment (dev, staging, or prod).
ALLOW_REAL_MONEY_TRADING = os.getenv("ALLOW_REAL_MONEY_TRADING", "false").lower() == "true"

# Deliberately separate variable names from MT5_LOGIN/PASSWORD/SERVER
# above (never aliased to them) so a real account can't be reached by
# accident just because demo credentials happen to be set.
MT5_REAL_LOGIN_ENV = os.getenv("MT5_REAL_LOGIN", "0")
MT5_REAL_LOGIN = (
    int(MT5_REAL_LOGIN_ENV)
    if MT5_REAL_LOGIN_ENV and MT5_REAL_LOGIN_ENV.isdigit() and MT5_REAL_LOGIN_ENV != "0"
    else None
)
MT5_REAL_PASSWORD = os.getenv("MT5_REAL_PASSWORD")
MT5_REAL_SERVER = os.getenv("MT5_REAL_SERVER")

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
try:
    RETRAINING_INTERVAL = int(os.getenv("RETRAINING_INTERVAL", "24") or 24)
except (ValueError, TypeError):
    RETRAINING_INTERVAL = 24  # hours
PERFORMANCE_THRESHOLD = float(os.getenv("PERFORMANCE_THRESHOLD", "0.55"))
try:
    MIN_TRAINING_SAMPLES = int(os.getenv("MIN_TRAINING_SAMPLES", "100") or 100)
except (ValueError, TypeError):
    MIN_TRAINING_SAMPLES = 100

# Walk-forward / evaluation defaults
try:
    WALK_FORWARD_MIN_TRAIN_SIZE = int(os.getenv("WALK_FORWARD_MIN_TRAIN_SIZE", str(MIN_TRAINING_SAMPLES)) or MIN_TRAINING_SAMPLES)
except (ValueError, TypeError):
    WALK_FORWARD_MIN_TRAIN_SIZE = MIN_TRAINING_SAMPLES
try:
    WALK_FORWARD_STEP_SIZE = int(os.getenv("WALK_FORWARD_STEP_SIZE", "50") or 50)
except (ValueError, TypeError):
    WALK_FORWARD_STEP_SIZE = 50

# Model prediction thresholds
MODEL_BUY_THRESHOLD = float(os.getenv("MODEL_BUY_THRESHOLD", "0.58"))
MODEL_SELL_THRESHOLD = float(os.getenv("MODEL_SELL_THRESHOLD", "0.42"))

# ── SMTP / Email Alerts ────────────────────────────────────────
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
try:
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587") or 587)
except (ValueError, TypeError):
    SMTP_PORT = 587
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_RECIPIENTS = os.getenv("ALERT_RECIPIENTS", "")
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")

# ── Webhook ────────────────────────────────────────────────────
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
try:
    WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "5000") or 5000)
except (ValueError, TypeError):
    WEBHOOK_PORT = 5000

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

    # MT5 (demo)
    MT5_LOGIN = MT5_LOGIN
    MT5_PASSWORD = MT5_PASSWORD
    MT5_SERVER = MT5_SERVER
    MT5_PATH = MT5_PATH

    # MT5 (real money — execution-parity audit §9)
    ALLOW_REAL_MONEY_TRADING = ALLOW_REAL_MONEY_TRADING
    MT5_REAL_LOGIN = MT5_REAL_LOGIN
    MT5_REAL_PASSWORD = MT5_REAL_PASSWORD
    MT5_REAL_SERVER = MT5_REAL_SERVER

    # Telegram
    TELEGRAM_TOKEN = TELEGRAM_TOKEN
    TELEGRAM_CHAT_ID = TELEGRAM_CHAT_ID
    ENABLE_TELEGRAM = ENABLE_TELEGRAM

    # LLM
    GROQ_API_KEY = GROQ_API_KEY
    GROQ_MODEL = GROQ_MODEL
    GEMINI_API_KEY = GEMINI_API_KEY
    GEMINI_MODEL = GEMINI_MODEL
    # OpenRouter is now FALLBACK #2 in the cascade (2026-07-25).
    OPENROUTER_API_KEY = OPENROUTER_API_KEY
    OPENROUTER_MODEL = OPENROUTER_MODEL
    # Optional extras — disabled by default
    OC_INCLUDE_CEREBRAS = OC_INCLUDE_CEREBRAS
    OC_INCLUDE_SAMBANOVA = OC_INCLUDE_SAMBANOVA
    # Anthropic intentionally disabled (free-tier-only policy)
    # ANTHROPIC_API_KEY = ANTHROPIC_API_KEY

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

    # Forex pairs for scanner/data updater — full 48-pair universe
    # (SYMBOLS is defined at module top: 7 majors + 21 minors + 4 metals
    #  + 2 exotic + 9 additional crosses + 5 Asia Pacific = 48)
    FOREX_PAIRS = SYMBOLS  # Reuse the SYMBOLS list (48 pairs)

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


# Validation is called explicitly from main.py via validate_all_config(),
# NOT on import — to avoid side-effects when config is imported as a
# dependency (e.g. from tests, docs generation, or IDE tooling).