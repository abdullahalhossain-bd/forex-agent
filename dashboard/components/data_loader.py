# dashboard/components/data_loader.py  —  Day 56 | Shared Data Access Layer
#                                          + Day 65 Global Market State
# ============================================================
# সব dashboard page এই module দিয়ে memory/*.json ফাইল পড়ে।
# Day 52-55-এর learning modules (confidence_engine, deep_analyzer,
# lesson_memory, performance_feedback, rule_updater, auto_optimizer,
# strategy_config) যেসব path-এ write করে, এখানে সেই একই path ব্যবহার
# করা হয়েছে — যাতে dashboard সরাসরি live data দেখাতে পারে।
#
# Day 65: IntermarketEngine memory/intermarket_history.json-এ যা লেখে
# (analysis/intermarket.py দেখুন), সেখান থেকে সর্বশেষ entry পড়ে
# dashboard-এর "🌎 Global Market State" প্যানেল দেখায়।
#
# কোনো ফাইল না থাকলে (এখনো কোনো trade/analysis হয়নি), ছোট demo data
# fallback করে — dashboard কখনো খালি/ভাঙা দেখাবে না।
# ============================================================

import json
import os
import random
from datetime import datetime, timedelta, timezone

# ── Paths (Day 52-55 modules-এর সাথে identical) ─────────────────
PATTERN_STATS_PATH       = "memory/pattern_stats.json"
DISABLED_PATTERNS_PATH   = "memory/disabled_patterns.json"
LESSON_MEMORY_PATH       = "memory/lesson_memory.json"
PATTERN_RULES_PATH       = "memory/pattern_rules.json"
PENDING_RULE_APPROVALS   = "memory/pending_rule_approvals.json"
DEEP_ANALYSIS_LOG_PATH   = "memory/deep_analysis_log.json"
PERFORMANCE_FEEDBACK_PATH = "memory/performance_feedback.json"
STRATEGY_CONFIG_PATH     = "memory/strategy_config.json"
STRATEGY_VERSIONS_DIR    = "memory/strategy_versions"
PENDING_OPTIMIZER_PATH   = "memory/pending_optimizer_approvals.json"
WEEKLY_REPORTS_PATH      = "memory/weekly_reports.json"
OPTIMIZER_RUN_LOG_PATH   = "memory/optimizer_run_log.json"
SYSTEM_CONTROL_PATH      = "memory/system_control.json"
# Day 65
INTERMARKET_HISTORY_PATH = "memory/intermarket_history.json"


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════
# SYSTEM CONTROL (Emergency Panel reads/writes this)
# ══════════════════════════════════════════════════════════════

def get_system_control() -> dict:
    return load_json(SYSTEM_CONTROL_PATH, {
        "trading_enabled": True,
        "mode": "DEMO",
        "last_changed_by": "system",
        "last_changed_at": None,
    })


def set_system_control(**kwargs) -> dict:
    ctrl = get_system_control()
    ctrl.update(kwargs)
    ctrl["last_changed_at"] = datetime.now(timezone.utc).isoformat()
    save_json(SYSTEM_CONTROL_PATH, ctrl)
    return ctrl


# ══════════════════════════════════════════════════════════════
# LIVE ROOM DATA  (real engine না থাকলে demo data)
# ══════════════════════════════════════════════════════════════

def get_live_signals() -> list:
    """বাস্তব scanner থাকলে memory/live_signals.json থেকে আসবে; নাহলে demo।"""
    data = load_json("memory/live_signals.json", None)
    if data:
        return data
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    demo = []
    for p in pairs:
        signal = random.choice(["BUY", "SELL", "WAIT"])
        entry = round(random.uniform(1.05, 1.30), 4) if "JPY" not in p else round(random.uniform(140, 160), 2)
        demo.append({
            "pair": p,
            "signal": signal,
            "confidence": random.randint(45, 92),
            "entry": entry,
            "sl": round(entry * 0.997, 4),
            "tp": round(entry * 1.006, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    return demo


def get_open_positions() -> list:
    data = load_json("memory/open_positions.json", None)
    if data:
        return data
    return [
        {"pair": "EURUSD", "direction": "BUY", "entry": 1.0850, "current": 1.0870, "pnl": 20.0, "lots": 0.10},
    ]


def get_todays_pnl() -> dict:
    trades = load_json(PERFORMANCE_FEEDBACK_PATH, [])
    today = datetime.now(timezone.utc).date()
    todays = [
        t for t in trades
        if t.get("timestamp") and _safe_date(t["timestamp"]) == today
    ]
    if not todays:
        return {"pnl": 0.0, "win_rate": None, "trades": 0}
    wins = sum(1 for t in todays if t.get("win"))
    pnl = round(sum(t.get("pnl", 0) or 0 for t in todays), 2)
    return {
        "pnl": pnl,
        "win_rate": round(wins / len(todays) * 100, 1),
        "trades": len(todays),
    }


def _safe_date(ts: str):
    try:
        return datetime.fromisoformat(ts).date()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
# AI BRAIN DATA
# ══════════════════════════════════════════════════════════════

def get_ai_brain_state() -> dict:
    """বাস্তব engine থাকলে memory/ai_brain_state.json থেকে আসবে; নাহলে demo।"""
    data = load_json("memory/ai_brain_state.json", None)
    if data:
        return data
    return {
        "pair": "EURUSD",
        "timeframe": "M15",
        "market_regime": "TRENDING_BULLISH",
        "structure": "HH + HL",
        "liquidity": "Sell-side liquidity swept",
        "smc": "Bullish Order Block detected",
        "reasoning": (
            "Price rejected demand zone. BOS confirmed. "
            "FVG filled. Risk acceptable."
        ),
        "decision": "BUY",
        "confidence": 86,
        "confidence_breakdown": {
            "Technical":   85,
            "SMC":         90,
            "Sentiment":   70,
            "News Filter": 100,
            "Risk":        95,
        },
    }


def get_decision_timeline() -> list:
    """⭐ Bonus 1 — AI Decision Timeline।"""
    data = load_json("memory/decision_timeline.json", None)
    if data:
        return data
    now = datetime.now(timezone.utc)
    return [
        {"time": (now - timedelta(minutes=6)).strftime("%H:%M"), "event": "Detected liquidity sweep"},
        {"time": (now - timedelta(minutes=4)).strftime("%H:%M"), "event": "BOS confirmed"},
        {"time": (now - timedelta(minutes=2)).strftime("%H:%M"), "event": "Risk approved"},
        {"time": (now - timedelta(minutes=1)).strftime("%H:%M"), "event": "BUY executed"},
    ]


# ══════════════════════════════════════════════════════════════
# LEARNING CENTER DATA
# ══════════════════════════════════════════════════════════════

def get_recent_mistakes(limit: int = 10) -> list:
    log_entries = load_json(DEEP_ANALYSIS_LOG_PATH, [])
    if not log_entries:
        lessons = load_json(LESSON_MEMORY_PATH, [])
        return [
            {
                "id": l.get("lesson_id", l.get("id")),
                "reason": l.get("mistake", l.get("error_type", "Unknown")),
                "market": l.get("market_condition", l.get("regime_at_time", "UNKNOWN")),
                "lesson": l.get("new_rule", l.get("lesson", "")),
            }
            for l in lessons[-limit:][::-1]
        ]
    out = []
    for entry in log_entries[-limit:][::-1]:
        analysis = entry.get("analysis", {})
        out.append({
            "id": entry.get("timestamp", "")[:19],
            "reason": analysis.get("loss_reason", analysis.get("error_type", "Unknown")),
            "market": analysis.get("regime_at_time", "UNKNOWN"),
            "lesson": analysis.get("lesson", ""),
        })
    return out


def get_pattern_performance() -> list:
    stats = load_json(PATTERN_STATS_PATH, {})
    if not stats:
        return [
            {"key": "Hammer|EURUSD|H1|TRENDING", "win_rate": 68, "total": 54},
            {"key": "Engulfing|GBPUSD|M15|RANGING", "win_rate": 41, "total": 39},
        ]
    rows = []
    for key, e in stats.items():
        rows.append({
            "key": key,
            "win_rate": e.get("win_rate", 0),
            "total": e.get("total_trades", 0),
        })
    return sorted(rows, key=lambda r: r["win_rate"], reverse=True)


def get_learned_rules() -> list:
    rules = load_json(PATTERN_RULES_PATH, {})
    if not rules:
        return [
            {"summary": "Avoid GBPUSD M5 Asian session"},
            {"summary": "Prefer BOS + FVG setups"},
            {"summary": "Reduce risk during high ATR"},
        ]
    out = []
    for key, r in rules.items():
        out.append({
            "summary": f"{r.get('pattern')} in {r.get('condition')} → confidence {r.get('confidence')}%",
            "lesson": r.get("lesson", ""),
        })
    return out


def get_trade_replay_list(limit: int = 20) -> list:
    """⭐ Bonus 2 — Trade Replay System list of past trades."""
    trades = load_json(PERFORMANCE_FEEDBACK_PATH, [])
    return trades[-limit:][::-1]


# ══════════════════════════════════════════════════════════════
# STRATEGY LAB DATA
# ══════════════════════════════════════════════════════════════

def get_strategy_versions() -> list:
    if not os.path.isdir(STRATEGY_VERSIONS_DIR):
        return []
    versions = []
    for fname in sorted(os.listdir(STRATEGY_VERSIONS_DIR)):
        if fname.endswith(".json"):
            v = load_json(os.path.join(STRATEGY_VERSIONS_DIR, fname), None)
            if v:
                versions.append(v)
    versions.sort(key=lambda v: v.get("created_at", ""))
    return versions


def get_strategy_config() -> dict:
    return load_json(STRATEGY_CONFIG_PATH, {
        "version": "1.0",
        "active_pairs": ["EURUSD", "USDJPY"],
        "disabled_pairs": {},
        "session_preference": {},
        "risk_percent": 1.0,
    })


def get_backtest_results() -> list:
    """বাস্তব backtest engine থাকলে memory/backtest_results.json থেকে আসবে।"""
    data = load_json("memory/backtest_results.json", None)
    if data:
        return data
    return [
        {"strategy": "RSI", "win_rate": 45, "profit_factor": 1.1, "max_dd": 18},
        {"strategy": "SMC", "win_rate": 62, "profit_factor": 2.0, "max_dd": 11},
        {"strategy": "SMC + FVG", "win_rate": 68, "profit_factor": 2.4, "max_dd": 8},
    ]


def get_pending_optimizer_suggestions() -> list:
    return load_json(PENDING_OPTIMIZER_PATH, [])


# ══════════════════════════════════════════════════════════════
# RISK MONITOR DATA
# ══════════════════════════════════════════════════════════════

def get_equity_curve() -> list:
    data = load_json("memory/equity_curve.json", None)
    if data:
        return data
    trades = load_json(PERFORMANCE_FEEDBACK_PATH, [])
    if not trades:
        # demo random-walk equity curve
        equity, curve = 10000.0, []
        for i in range(40):
            equity += random.uniform(-80, 100)
            curve.append({"step": i, "equity": round(equity, 2)})
        return curve
    equity, curve = 10000.0, []
    for i, t in enumerate(trades[-60:]):
        equity += t.get("pnl", 0) or 0
        curve.append({"step": i, "equity": round(equity, 2)})
    return curve


def get_risk_status() -> dict:
    cfg = get_strategy_config()
    daily_limit = load_json("memory/daily_risk_limit.json", {"limit": 300, "used": 80})
    return {
        "current_risk_pct": cfg.get("risk_percent", 1.0),
        "max_allowed_pct":  1.0,
        "daily_limit":      daily_limit.get("limit", 300),
        "daily_used":       daily_limit.get("used", 0),
    }


def get_system_health() -> dict:
    """⭐ Bonus 3 — System Health Panel."""
    data = load_json("memory/system_health.json", None)
    if data:
        return data
    return {
        "mt5": "CONNECTED",
        "database": "OK",
        "vision_ai": "OK",
        "last_restart": "2 days ago",
    }


# ══════════════════════════════════════════════════════════════
# Day 65 — GLOBAL MARKET STATE  (🌎 Intermarket Dashboard Panel)
# ══════════════════════════════════════════════════════════════

def get_global_market_state() -> dict:
    """
    analysis/intermarket.py-এর IntermarketEngine._save_memory() যা লেখে
    (memory/intermarket_history.json), তার সর্বশেষ entry পড়ে dashboard
    panel-এর জন্য ready ফরম্যাটে ফেরত দেয়। কোনো history না থাকলে demo
    data fallback করে — doc-এর mockup অনুযায়ী।
    """
    history = load_json(INTERMARKET_HISTORY_PATH, [])
    if history:
        latest = history[-1]
        trend = lambda v: "UP" if (v or 0) > 0 else ("DOWN" if (v or 0) < 0 else "FLAT")
        return {
            "pair":          latest.get("pair", "—"),
            "dxy_value":     latest.get("dxy_value"),
            "dxy_trend":     latest.get("dxy_trend", "NEUTRAL"),
            "gold_trend":    trend(latest.get("gold_change")),
            "oil_trend":     trend(latest.get("oil_change")),
            "bond_yield":    latest.get("bond_yield"),
            "sp500_trend":   trend(latest.get("sp500_change")),
            "vix_value":     latest.get("vix_value"),
            "macro_regime":  latest.get("regime", "NEUTRAL"),
            "macro_score":   latest.get("macro_score", 0),
            "pair_bias":     latest.get("pair_bias", "NEUTRAL"),
            "timestamp":     latest.get("timestamp"),
            "source":        "live",
        }

    # Demo fallback (matches the Day 65 doc's dashboard mockup)
    return {
        "pair":         "EURUSD",
        "dxy_value":    104.3,
        "dxy_trend":    "BULLISH",
        "gold_trend":   "DOWN",
        "oil_trend":    "FLAT",
        "bond_yield":   4.28,
        "sp500_trend":  "DOWN",
        "vix_value":    24.5,
        "macro_regime": "RISK_OFF",
        "macro_score":  70,
        "pair_bias":    "SELL",
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "source":       "demo",
    }