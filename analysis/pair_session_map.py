# analysis/pair_session_map.py  —  Day 63 | Pair-Session Specialization Engine
# ============================================================
# কোন session-এ কোন pair সবচেয়ে active ও reliable।
# AI এই map দেখে pair priority ঠিক করবে।
# ============================================================

# ── Primary pair recommendations per session ──────────────────
PAIR_SESSION_MAP = {
    "SYDNEY": {
        "preferred":    ["AUDUSD", "NZDUSD", "AUDNZD", "AUDJPY"],
        "avoid":        ["EURUSD", "GBPUSD"],
        "note":         "AUD/NZD pairs most active. Low spread window.",
    },
    "TOKYO": {
        "preferred":    ["USDJPY", "AUDJPY", "EURJPY", "GBPJPY", "CADJPY"],
        "avoid":        ["GBPUSD", "EURCAD"],
        "note":         "JPY crosses dominate. Tokyo BOJ influence.",
    },
    "LONDON": {
        "preferred":    ["EURUSD", "GBPUSD", "EURGBP", "GBPJPY", "EURCAD"],
        "avoid":        ["AUDUSD", "NZDUSD"],
        "note":         "EUR/GBP pairs most liquid. Highest volume globally.",
    },
    "NEW_YORK": {
        "preferred":    ["USDCAD", "USDCHF", "EURUSD", "GBPUSD", "USDJPY"],
        "avoid":        ["AUDJPY", "NZDUSD"],
        "note":         "USD pairs dominant. NY session drives USD volatility.",
    },
    "LONDON_NY_OVERLAP": {
        "preferred":    ["EURUSD", "GBPUSD", "USDJPY", "USDCAD"],
        "avoid":        [],
        "note":         "All major pairs active. Best spreads of the day.",
    },
}

# ── Always Monitor (any session) ──────────────────────────────
ALWAYS_MONITOR = ["XAUUSD"]

# ── Session Priority Score per pair ──────────────────────────
# Higher = more appropriate for that session (0-100)
PAIR_PRIORITY = {
    "EURUSD": {
        "SYDNEY":            20,
        "TOKYO":             35,
        "LONDON":            95,
        "NEW_YORK":          85,
        "LONDON_NY_OVERLAP": 100,
    },
    "GBPUSD": {
        "SYDNEY":            15,
        "TOKYO":             30,
        "LONDON":            95,
        "NEW_YORK":          80,
        "LONDON_NY_OVERLAP": 98,
    },
    "USDJPY": {
        "SYDNEY":            40,
        "TOKYO":             90,
        "LONDON":            70,
        "NEW_YORK":          85,
        "LONDON_NY_OVERLAP": 88,
    },
    "AUDUSD": {
        "SYDNEY":            90,
        "TOKYO":             75,
        "LONDON":            50,
        "NEW_YORK":          55,
        "LONDON_NY_OVERLAP": 60,
    },
    "NZDUSD": {
        "SYDNEY":            85,
        "TOKYO":             70,
        "LONDON":            45,
        "NEW_YORK":          50,
        "LONDON_NY_OVERLAP": 55,
    },
    "USDCAD": {
        "SYDNEY":            30,
        "TOKYO":             35,
        "LONDON":            65,
        "NEW_YORK":          90,
        "LONDON_NY_OVERLAP": 85,
    },
    "USDCHF": {
        "SYDNEY":            25,
        "TOKYO":             40,
        "LONDON":            70,
        "NEW_YORK":          88,
        "LONDON_NY_OVERLAP": 85,
    },
    "XAUUSD": {
        "SYDNEY":            55,
        "TOKYO":             60,
        "LONDON":            80,
        "NEW_YORK":          88,
        "LONDON_NY_OVERLAP": 92,
    },
    "EURGBP": {
        "SYDNEY":            20,
        "TOKYO":             30,
        "LONDON":            90,
        "NEW_YORK":          65,
        "LONDON_NY_OVERLAP": 80,
    },
    "EURJPY": {
        "SYDNEY":            30,
        "TOKYO":             85,
        "LONDON":            75,
        "NEW_YORK":          70,
        "LONDON_NY_OVERLAP": 80,
    },
    "GBPJPY": {
        "SYDNEY":            25,
        "TOKYO":             80,
        "LONDON":            88,
        "NEW_YORK":          75,
        "LONDON_NY_OVERLAP": 85,
    },
    "AUDJPY": {
        "SYDNEY":            85,
        "TOKYO":             90,
        "LONDON":            55,
        "NEW_YORK":          50,
        "LONDON_NY_OVERLAP": 60,
    },
    "EURCAD": {
        "SYDNEY":            20,
        "TOKYO":             25,
        "LONDON":            78,
        "NEW_YORK":          70,
        "LONDON_NY_OVERLAP": 75,
    },
}

# Default priority for unknown pairs
DEFAULT_PRIORITY = 50


def get_pair_priority(pair: str, session: str) -> int:
    """Return priority score (0-100) for a pair in a given session."""
    pair_clean = pair.upper().replace("/", "").replace("=X", "")[:6]
    session_map = PAIR_PRIORITY.get(pair_clean, {})
    return session_map.get(session, DEFAULT_PRIORITY)


def get_preferred_pairs(session: str) -> list[str]:
    """Return list of preferred pairs for a session."""
    if session in PAIR_SESSION_MAP:
        return PAIR_SESSION_MAP[session]["preferred"] + ALWAYS_MONITOR
    return ALWAYS_MONITOR


def get_pair_session_recommendation(pair: str, session: str) -> dict:
    """
    Single pair + session combination का recommendation।
    Returns priority level and a label.
    """
    score = get_pair_priority(pair, session)
    pair_clean = pair.upper().replace("/", "").replace("=X", "")[:6]

    session_info = PAIR_SESSION_MAP.get(session, {})
    preferred    = session_info.get("preferred", [])
    avoid        = session_info.get("avoid", [])

    if pair_clean in avoid:
        label = "AVOID"
    elif score >= 85:
        label = "EXCELLENT"
    elif score >= 70:
        label = "GOOD"
    elif score >= 55:
        label = "FAIR"
    else:
        label = "POOR"

    return {
        "pair":    pair,
        "session": session,
        "priority": score,
        "label":   label,
        "note":    session_info.get("note", ""),
    }