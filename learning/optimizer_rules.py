# learning/optimizer_rules.py  —  Day 55 | Optimizer Safety Rules ⭐⭐⭐⭐⭐
# ============================================================
# AutoOptimizer যেন নিজের performance দেখে impulsively, overfit হয়ে
# trading system পরিবর্তন না করে — সেটার জন্য এই central "Safety Layer"।
#
# Rule (spec অনুযায়ী):
#   No update unless:
#     - Minimum 50 trades
#     - AND statistically significant result
#     - AND backtest confirmation
#
# এই ফাইলটা কোনো state রাখে না — pure constants + validator functions।
# auto_optimizer.py এবং strategy_config.py এটা import করে ব্যবহার করে।
# ============================================================

import math
from dataclasses import dataclass
from typing import Optional

# ── Sample-size thresholds ─────────────────────────────────────
MIN_TRADES_FOR_PAIR_DECISION    = 50    # pair remove/keep সিদ্ধান্তের জন্য
MIN_TRADES_FOR_PATTERN_UPDATE   = 50    # pattern confidence rewrite-এর জন্য
MIN_TRADES_FOR_SESSION_DECISION = 20    # session preference-এর জন্য
MIN_TRADES_FOR_RISK_CHANGE      = 30    # risk % পরিবর্তনের জন্য

# ── Performance thresholds ─────────────────────────────────────
PAIR_REMOVE_WIN_RATE      = 40.0   # এর নিচে win rate হলে pair candidate for removal
PAIR_REMOVE_PROFIT_FACTOR = 1.0    # PF < 1.0 মানে net negative expectancy
PATTERN_LOW_WIN_RATE      = 45.0   # এর নিচে গেলে pattern confidence কমানো হবে
SESSION_GOOD_WIN_RATE     = 60.0
SESSION_BAD_WIN_RATE      = 40.0

# ── Statistical significance (one-sided z-test vs 50% baseline) ─
Z_CRITICAL_95 = 1.645   # 95% confidence, one-sided

# ── Risk engine ─────────────────────────────────────────────────
DEFAULT_BASE_RISK_PCT   = 1.0
MIN_RISK_PCT            = 0.25
MAX_RISK_PCT            = 2.0
RISK_STEP_PCT           = 0.2     # একবারে সর্বোচ্চ কতটা বদলানো যাবে
LOW_VOLATILITY_FACTOR   = 0.8     # risk = base / factor  → boosts risk slightly
HIGH_VOLATILITY_FACTOR  = 2.0     # risk = base / factor  → halves risk

# ── Version rollback ─────────────────────────────────────────────
ROLLBACK_DEGRADATION_PCT = 10.0   # নতুন version পুরনোর চেয়ে ১০%+ খারাপ হলে rollback
ROLLBACK_MIN_TRADES      = 20     # rollback decision নেওয়ার আগে নতুন version-এ ন্যূনতম ট্রেড

# ── Autonomy ───────────────────────────────────────────────────
HUMAN_APPROVAL_MODE_DEFAULT = True   # শুরুতে সব change human approval লাগবে


@dataclass
class ValidationResult:
    ok: bool
    reason: str
    sample_size: int
    significant: bool
    z_score: Optional[float] = None


def is_statistically_significant(
    wins: int, total: int, baseline: float = 0.5, z_critical: float = Z_CRITICAL_95
) -> tuple:
    """
    One-sided z-test: observed win rate বেসলাইন (default 50%) থেকে
    সত্যিই আলাদা কিনা, নাকি স্রেফ random variance।

    Returns: (is_significant: bool, z_score: float)
    """
    if total <= 0:
        return False, 0.0

    p_hat = wins / total
    se = math.sqrt(baseline * (1 - baseline) / total)
    if se == 0:
        return False, 0.0

    z = (p_hat - baseline) / se
    return abs(z) >= z_critical, round(z, 2)


def validate_change(
    sample_size: int,
    wins: int,
    min_sample: int = MIN_TRADES_FOR_PATTERN_UPDATE,
    backtest_passed: bool = True,
    baseline: float = 0.5,
) -> ValidationResult:
    """
    Safety Layer-এর কেন্দ্রীয় gate। Day 55 spec-এর তিনটা শর্ত মিলিয়ে
    চেক করে — কোনো একটাও fail করলে change প্রত্যাখ্যাত।

    Usage:
        v = validate_change(sample_size=80, wins=30, min_sample=50)
        if not v.ok:
            print(v.reason)  # explain why blocked
    """
    if sample_size < min_sample:
        return ValidationResult(
            ok=False,
            reason=f"Sample size {sample_size} < minimum required {min_sample}",
            sample_size=sample_size,
            significant=False,
        )

    significant, z = is_statistically_significant(wins, sample_size, baseline=baseline)
    if not significant:
        return ValidationResult(
            ok=False,
            reason=f"Not statistically significant (z={z}, need |z|>={Z_CRITICAL_95})",
            sample_size=sample_size,
            significant=False,
            z_score=z,
        )

    if not backtest_passed:
        return ValidationResult(
            ok=False,
            reason="Backtest confirmation failed",
            sample_size=sample_size,
            significant=True,
            z_score=z,
        )

    return ValidationResult(
        ok=True,
        reason=f"Validated — n={sample_size}, z={z}, backtest confirmed",
        sample_size=sample_size,
        significant=True,
        z_score=z,
    )


def volatility_to_risk(base_risk: float, volatility_factor: float) -> float:
    """
    Day 55 formula:  risk = base_risk / volatility_factor

    volatility_factor উদাহরণ:
        0.8  → low volatility  → risk বাড়বে সামান্য
        1.0  → normal          → risk অপরিবর্তিত
        2.0  → high volatility → risk অর্ধেক হবে
    """
    volatility_factor = max(0.1, volatility_factor)
    risk = base_risk / volatility_factor
    return round(max(MIN_RISK_PCT, min(MAX_RISK_PCT, risk)), 2)


def clamp_risk_step(old_risk: float, new_risk: float, max_step: float = RISK_STEP_PCT) -> float:
    """একবারে risk যেন খুব বেশি লাফ না দেয় — gradual change।"""
    delta = new_risk - old_risk
    if abs(delta) > max_step:
        new_risk = old_risk + math.copysign(max_step, delta)
    return round(max(MIN_RISK_PCT, min(MAX_RISK_PCT, new_risk)), 2)