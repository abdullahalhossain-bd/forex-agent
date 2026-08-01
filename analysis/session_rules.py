# analysis/session_rules.py  —  Day 63 | Session Rules & Strategy Definitions
# ============================================================
# Forex market session time windows (GMT), characteristics,
# strategy modes, dead zones, and DST awareness.
# ============================================================

from datetime import timezone

# ── Session Time Windows (GMT, start inclusive, end exclusive) ─
SESSION_WINDOWS = {
    "SYDNEY": {
        "start": 22,   # 22:00 GMT
        "end":   7,    # 07:00 GMT next day (crosses midnight)
        "crosses_midnight": True,
    },
    "TOKYO": {
        "start": 0,    # 00:00 GMT
        "end":   9,    # 09:00 GMT
        "crosses_midnight": False,
    },
    "LONDON": {
        "start": 8,    # 08:00 GMT
        "end":   17,   # 17:00 GMT
        "crosses_midnight": False,
    },
    "NEW_YORK": {
        "start": 13,   # 13:00 GMT
        "end":   22,   # 22:00 GMT
        "crosses_midnight": False,
    },
    "LONDON_NY_OVERLAP": {
        "start": 13,   # 13:00 GMT
        "end":   17,   # 17:00 GMT
        "crosses_midnight": False,
    },
}

# ── Dead Zones (avoid trading) ────────────────────────────────
# RESTORED (institutional review, Finding C-2): the Day 81+ hotfix emptied
# this list entirely, which removed ALL session/liquidity-based trade
# gating — the bot was left trading 24/7 with no liquidity-window
# protection whatsoever, including through the lowest-liquidity hours
# (Sydney open / early Tokyo) it was explicitly designed to avoid.
#
# The original hotfix was solving a real operational problem — the bot's
# run window happened to fully overlap 22:00-02:00 GMT, so the hard block
# meant zero trading opportunity for that operator. Deleting the safety
# feature for everyone was the wrong fix for that problem. The correct
# fix is DEAD_ZONES_ENABLED below: a single, visible, explicit switch.
# Anyone who genuinely needs to trade through this window can set it to
# False in their own deployment config — a conscious, auditable decision
# — rather than the protection being silently absent for all users.
DEAD_ZONES_ENABLED = True

DEAD_ZONES = [
    {"start": 22, "end": 24, "reason": "Sydney open — very low liquidity"},
    {"start": 0,  "end": 2,  "reason": "Early Tokyo — low volume"},
]

# ── Session Characteristics ───────────────────────────────────
SESSION_CHARACTERISTICS = {
    "SYDNEY": {
        "volatility":    "LOW",
        "behavior":      "RANGING",
        "description":   "Low volatility range formation. Avoid breakout trades.",
        "risk_level":    "LOW",
    },
    "TOKYO": {
        "volatility":    "MEDIUM",
        "behavior":      "RANGING",
        "description":   "JPY movement dominant. Consolidation. Suited for range strategies.",
        "risk_level":    "LOW",
    },
    "LONDON": {
        "volatility":    "HIGH",
        "behavior":      "TRENDING",
        "description":   "Highest liquidity globally. Breakouts and liquidity sweeps common.",
        "risk_level":    "MEDIUM",
    },
    "NEW_YORK": {
        "volatility":    "HIGH",
        "behavior":      "TRENDING",
        "description":   "USD volatility. Trend continuation from London direction.",
        "risk_level":    "MEDIUM",
    },
    "LONDON_NY_OVERLAP": {
        "volatility":    "VERY_HIGH",
        "behavior":      "INSTITUTIONAL",
        "description":   "Maximum volume. Best setups. Only A+ trades allowed.",
        "risk_level":    "LOW",   # risk is low because setups are highest quality
    },
    "DEAD_ZONE": {
        "volatility":    "VERY_LOW",
        "behavior":      "NO_TRADE",
        "description":   "Low liquidity. Spreads widen. No trading recommended.",
        "risk_level":    "VERY_HIGH",
    },
    "BETWEEN_SESSIONS": {
        "volatility":    "LOW",
        "behavior":      "WAIT",
        "description":   "Between sessions. Wait for next session open.",
        "risk_level":    "HIGH",
    },
}

# ── Confidence floor — ONE knob instead of six ────────────────
# Previously every session repeated its own "min_confidence": 60 with a
# different "was 65/70/72/85" history comment. Same number, six places,
# six comment threads — hard to see at a glance that they'd all already
# converged. Consolidated here. Change this ONE value to retune every
# tradeable session at once. DEAD_ZONE is intentionally NOT driven by
# this constant (see below) — it's a full session block, not a
# confidence level, and mixing the two here would make an accidental
# edit able to silently reopen trading in the illiquid dead-zone hours.
BASE_MIN_CONFIDENCE = 60

# ── Strategy Modes Per Session ────────────────────────────────
SESSION_STRATEGIES = {
    "SYDNEY": {
        "strategy":          "RANGE_TRADING",
        "action":            "Buy near range low, Sell near range high",
        "avoid":             "Breakout trades — false signals likely",
        "min_confidence":    BASE_MIN_CONFIDENCE,
        "risk_multiplier":   0.7,
        "note":              "Low volatility. Tight SL. Small targets.",
    },
    "TOKYO": {
        "strategy":          "RANGE_TRADING",
        "action":            "JPY pairs: fade extremes. Range-bound entries.",
        "avoid":             "Trending breakouts — consolidation phase",
        "min_confidence":    BASE_MIN_CONFIDENCE,
        "risk_multiplier":   0.8,
        "note":              "JPY dominates. USDJPY, EURJPY best suited.",
    },
    "LONDON": {
        "strategy":          "LONDON_BREAKOUT",
        "action":            "Asian range breakout. Liquidity sweep + BOS entry.",
        "avoid":             "Counter-trend during strong London moves",
        "min_confidence":    BASE_MIN_CONFIDENCE,
        "risk_multiplier":   1.0,
        "note":              "Check Asian high/low for liquidity sweep direction.",
    },
    "NEW_YORK": {
        "strategy":          "TREND_CONTINUATION",
        "action":            "Continue London trend. USD news-driven moves.",
        "avoid":             "Reversals without strong SMC confirmation",
        "min_confidence":    BASE_MIN_CONFIDENCE,
        "risk_multiplier":   1.0,
        "note":              "Follow London direction. Check order flow.",
    },
    "LONDON_NY_OVERLAP": {
        "strategy":          "A_PLUS_ONLY",
        "action":            "Full SMC confluence required. Institutional setups only.",
        "avoid":             "Anything below A+ grade",
        "min_confidence":    BASE_MIN_CONFIDENCE,
        "risk_multiplier":   1.2,
        "note":              "Best trading window. Wait for perfect setup.",
    },
    "DEAD_ZONE": {
        "strategy":          "NO_TRADE",
        "action":            "Do nothing. Prepare for next session.",
        "avoid":             "All trades",
        "min_confidence":    999,  # impossible to meet — intentional full
        # session block (illiquid, wide spreads), not a confidence knob.
        # Not tied to BASE_MIN_CONFIDENCE on purpose (see comment above).
        "risk_multiplier":   0.0,
        "note":              "Low liquidity. Spreads wide. High slippage risk.",
    },
    "BETWEEN_SESSIONS": {
        "strategy":          "WAIT",
        "action":            "Wait for next session to open.",
        "avoid":             "Forcing trades",
        "min_confidence":    BASE_MIN_CONFIDENCE,
        "risk_multiplier":   0.6,
        "note":              "Transitioning between sessions. Low participation.",
    },
}

# ── London Open Window (first 2 hours — best manipulation window) ─
LONDON_OPEN_WINDOW = {"start": 8, "end": 10}

# ── Minimum SMC Requirements per session ─────────────────────
# Lowered min_smc_score (50/55 → 40/45): 50 was rejecting a lot of
# genuinely tradeable setups whose SMC score landed in the 30-45 range
# (e.g. a real trending move without every single SMC factor present —
# see fusion_score computation in session_analyzer.session_smc_fusion(),
# which is 60% smc_score + 40% "requirements met" bonus, so a score in
# the 30s can still represent a reasonably structured move). require_bos
# is left as-is for LONDON/NEW_YORK/OVERLAP — BOS is a real structural
# confirmation, not an arbitrary number, so it stays a hard requirement.
#
# Round-31 fix (log-driven recalibration, 2026-07-17): a 24h production
# log (execution.log / trader.log) showed confidence 60-75% BUY signals
# with good R:R (1:2) getting BLOCKED by this gate alone — 82 straight
# "Fusion gate: SMC fusion rejected for NEW_YORK" rejections, with the
# actual smc_score distribution mostly landing at 0-35 (median ~15-20),
# only occasionally touching 30-35. The 30 requirement for NEW_YORK was
# therefore rejecting almost everything, not filtering out genuinely bad
# setups — the achievable range and the requirement barely overlapped.
# Lowered min_smc_score another notch (30/40 → 20/30) so setups that
# clear roughly the middle of the observed range can pass. require_bos
# stays untouched — that's a real structural check, not a tunable score.
# Base SMC score floor — one knob, mirrors BASE_MIN_CONFIDENCE above.
# LONDON keeps a slightly higher bar (BASE_MIN_SMC_SCORE + 10) since it's
# the breakout session where a real SMC read matters most; everything
# else shares the base value. require_bos stays per-session — that's a
# structural confirmation (did price actually break structure), not a
# tunable score, so it is NOT touched by this simplification.
BASE_MIN_SMC_SCORE = 12

SMC_REQUIREMENTS = {
    "SYDNEY":            {"min_smc_score": BASE_MIN_SMC_SCORE,      "require_bos": False, "require_ob": False},
    "TOKYO":             {"min_smc_score": BASE_MIN_SMC_SCORE,      "require_bos": False, "require_ob": False},
    "LONDON":            {"min_smc_score": BASE_MIN_SMC_SCORE + 8, "require_bos": True,  "require_ob": False},
    "NEW_YORK":          {"min_smc_score": BASE_MIN_SMC_SCORE,      "require_bos": True,  "require_ob": False},
    "LONDON_NY_OVERLAP": {"min_smc_score": BASE_MIN_SMC_SCORE + 3, "require_bos": True,  "require_ob": False},
    "DEAD_ZONE":         {"min_smc_score": 999,                     "require_bos": True,  "require_ob": True},
    "BETWEEN_SESSIONS":  {"min_smc_score": BASE_MIN_SMC_SCORE,      "require_bos": False, "require_ob": False},
}