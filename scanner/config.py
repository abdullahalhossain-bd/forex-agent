# scanner/config.py  —  Day 36 Part 1 | Pair Universe + Session Config
# ============================================================
# Updated: 30 pairs (7 majors + 21 crosses + 2 metals)
# ============================================================

# ── Full pair universe (30: 7 majors + 21 crosses + 2 metals) ──
# Per user request — agent scans ALL major, minor, exotic + metals.
FOREX_PAIRS = [
    # Majors (7)
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "USDCAD", "AUDUSD", "NZDUSD",
    # EUR crosses (6)
    "EURGBP", "EURJPY", "EURCHF", "EURAUD",
    "EURCAD", "EURNZD",
    # GBP crosses (5)
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    # AUD crosses (4)
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    # NZD crosses (3)
    "NZDJPY", "NZDCHF", "NZDCAD",
    # CAD/CHF crosses (3)
    "CADJPY", "CADCHF", "CHFJPY",
    # Metals (2)
    "XAUUSD",  # Gold
    "XAGUSD",  # Silver
]

# ── Default scan subset — scans ALL 30 pairs every cycle ──
DEFAULT_SCAN_PAIRS = list(FOREX_PAIRS)

# ── Correlation groups (same underlying risk) ──
# Updated for 30 pairs — correlated groups are blocked from
# having same-direction positions open simultaneously.
CORRELATION_GROUPS = [
    # USD-quoted majors — EUR/GBP আলাদা করা হয়েছে
    {"EURUSD", "AUDUSD", "NZDUSD"},
    {"GBPUSD"},                              # আলাদা group
    {"USDCHF", "USDJPY", "USDCAD"},
    # EUR group
    {"EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD"},
    # GBP group
    {"GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD"},
    # JPY group
    {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"},
    # AUD group
    {"AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD"},
    # CAD group
    {"USDCAD", "EURCAD", "GBPCAD", "AUDCAD", "CADJPY", "CADCHF", "NZDCAD"},
    # CHF group
    {"USDCHF", "EURCHF", "GBPCHF", "AUDCHF", "CADCHF", "CHFJPY", "NZDCHF"},
    # NZD group
    {"NZDUSD", "EURNZD", "GBPNZD", "AUDNZD", "NZDJPY", "NZDCHF", "NZDCAD"},
    # Metals
    {"XAUUSD", "XAGUSD"},
]
# ── Trading sessions (UTC hours) ──
SESSIONS = {
    "ASIAN":   {"start": 0,  "end": 9},
    "LONDON":  {"start": 7,  "end": 16},
    "NEW_YORK": {"start": 12, "end": 21},
}

# ── Pairs most active per session ──
SESSION_PAIRS = {
    "ASIAN":    ["USDJPY", "AUDJPY", "NZDJPY", "AUDUSD", "NZDUSD", "AUDNZD", "CADJPY", "CHFJPY", "XAUUSD"],
    "LONDON":   ["EURUSD", "GBPUSD", "EURGBP", "EURJPY", "GBPJPY", "EURCHF", "GBPCHF", "EURAUD", "GBPAUD", "XAUUSD"],
    "NEW_YORK": ["EURUSD", "GBPUSD", "USDCAD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "EURJPY", "GBPJPY", "XAUUSD", "XAGUSD"],
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
TOP_N = 5
