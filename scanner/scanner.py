# scanner/config.py  —  Day 36 Part 1 | Pair Universe + Session Config
# ============================================================

# ── Full pair universe ──
FOREX_PAIRS = [
    # Majors
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "USDCAD", "USDCHF", "NZDUSD",
    # Crosses
    "EURGBP", "EURJPY", "GBPJPY", "EURAUD",
    "EURCAD", "GBPAUD", "GBPCAD", "AUDCAD",
    "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY",
    # Commodity
    "XAUUSD",
]

# ── Default scan subset (faster, fewer API calls) ──
DEFAULT_SCAN_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "USDCAD", "XAUUSD",
]

# ── Correlation groups (same underlying risk) ──
CORRELATION_GROUPS = [
    {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "EURAUD", "GBPAUD"},
    {"USDCHF", "USDJPY", "USDCAD", "CADJPY", "CHFJPY"},
    {"XAUUSD"},   # standalone — correlated with USD weakness but treat separately
]

# ── Trading sessions (UTC hours) ──
SESSIONS = {
    "ASIAN":   {"start": 0,  "end": 9},
    "LONDON":  {"start": 7,  "end": 16},
    "NEW_YORK": {"start": 12, "end": 21},
}

# ── Pairs most active per session ──
SESSION_PAIRS = {
    "ASIAN":    ["USDJPY", "AUDUSD", "NZDUSD", "AUDJPY", "CADJPY"],
    "LONDON":   ["EURUSD", "GBPUSD", "EURGBP", "EURJPY", "GBPJPY"],
    "NEW_YORK": ["EURUSD", "GBPUSD", "USDCAD", "XAUUSD", "USDJPY"],
}

# ── Opportunity ranking weights ──
RANK_WEIGHTS = {
    "technical_strength": 0.30,
    "mtf_alignment":      0.25,
    "rr_ratio":           0.20,
    "news_safety":        0.15,
    "liquidity":          0.10,
}

# ── Minimum score to surface an opportunity ──
MIN_OPPORTUNITY_SCORE = 60

# ── Max opportunities to return ──
TOP_N = 3