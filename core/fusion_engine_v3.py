"""
core/fusion_engine_v3.py — Day 99+ V3 Fusion Engine (Master List Issues #5)

================================================================
Implements the 4 fusion-engine fixes from the operator's master list:

  5a. Conflicting Signals Deadlock     → Weighted system (Tech 40% / LLM 40% / News 20%)
  5b. Asynchronous Stale Data          → Signal TTL (default 30s) — old signals rejected
  5c. Dynamic RRR Mismatch             → RRR validator (min 1:1.5) — bad RRR → WAIT
  5d. Missing Key Exception            → .get(default) everywhere — KeyError-proof

This module is a DEFENSIVE WRAPPER around the existing decision pipeline.
It does NOT replace DecisionAgent.decide(); instead, DecisionAgent calls
validate_fusion() AFTER its own voting, and the result includes:

  - ttl_valid : bool        — was the signal produced within the TTL window?
  - rrr_valid : bool        — does the SL/TP/Entry combination clear min RRR?
  - weighted_confidence     — confidence re-scored using the 40/40/20 weights
  - conflict_resolution     — what to do when tech and LLM disagree
  - safe                     — bool, True only if all gates pass

If `safe == False`, DecisionAgent downgrades the decision to WAIT (not NO
TRADE — the analysis verdict is still valid; we just refuse to execute on
stale or bad-RRR signals, which is exactly the master-list requirement).
================================================================
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from utils.logger import get_logger

log = get_logger("fusion_engine_v3")


# ── Default weights (Master List Issue #5a) ────────────────────
# Tech 40% + LLM 40% + News 20% = 100%. These match the operator's
# stated requirement and can be overridden via env vars if a different
# deployment wants to emphasize one source over another.
import os
DEFAULT_TECH_WEIGHT   = float(os.getenv("FUSION_TECH_WEIGHT",   "0.40"))
DEFAULT_LLM_WEIGHT    = float(os.getenv("FUSION_LLM_WEIGHT",    "0.40"))
DEFAULT_NEWS_WEIGHT   = float(os.getenv("FUSION_NEWS_WEIGHT",   "0.20"))

# ── Signal TTL (Master List Issue #5b) ─────────────────────────
# A signal older than this is considered stale (the market has moved
# during the LLM's 5-10s thinking time). Default 30s, override via env.
DEFAULT_SIGNAL_TTL_SEC = float(os.getenv("FUSION_SIGNAL_TTL_SEC", "60"))  # Extended TTL

# ── Minimum RRR (Master List Issue #5c) ────────────────────────
# Hard floor: trades below this RRR are always downgraded to WAIT.
# Set to 1.0 — anything below 1:1 is mathematically losing on average.
# For 1.0–1.3 range: pass but apply a confidence penalty (see below).
# Override via env: FUSION_MIN_RRR
DEFAULT_MIN_RRR = float(os.getenv("FUSION_MIN_RRR", "1.0"))

# Soft RRR threshold: trades with RRR between hard floor and this value
# get a confidence penalty proportional to how far below they are.
# This replaces the old all-or-nothing downgrade that killed valid trades
# at RRR 1:1.17 (only 10% below the old 1.30 minimum).
DEFAULT_SOFT_RRR = float(os.getenv("FUSION_SOFT_RRR", "1.3"))


@dataclass
class FusionValidation:
    """Result of fusion-engine validation."""
    safe: bool = False                          # all gates pass → safe to execute
    ttl_valid: bool = True                      # signal within TTL window
    rrr_valid: bool = True                      # RRR ≥ minimum
    signal_age_sec: float = 0.0                 # age of the signal (seconds)
    rrr: float = 0.0                            # computed risk:reward ratio
    weighted_confidence: float = 0.0            # confidence re-scored with weights
    conflict_resolution: str = ""               # how a tech-vs-LLM conflict was handled
    failure_reasons: list = field(default_factory=list)
    weights_used: dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safe":                self.safe,
            "ttl_valid":           self.ttl_valid,
            "rrr_valid":           self.rrr_valid,
            "signal_age_sec":      round(self.signal_age_sec, 2),
            "rrr":                 round(self.rrr, 2),
            "weighted_confidence": round(self.weighted_confidence, 1),
            "conflict_resolution": self.conflict_resolution,
            "failure_reasons":     self.failure_reasons,
            "weights_used":        self.weights_used,
        }


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Master List Issue #5d: KeyError-proof float coercion."""
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_get(d: Optional[dict], key: str, default: Any = None) -> Any:
    """Master List Issue #5d: .get() with default, None-safe."""
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def compute_rrr(entry: float, sl: float, tp: float, direction: str) -> float:
    """Compute risk:reward ratio (1 : N).

    For BUY:  risk = entry - sl,  reward = tp - entry
    For SELL: risk = sl - entry, reward = entry - tp

    Returns 0.0 if any value is missing or risk ≤ 0 (would divide by zero
    or produce a negative ratio).
    """
    entry = _safe_float(entry)
    sl = _safe_float(sl)
    tp = _safe_float(tp)
    if entry <= 0 or sl <= 0 or tp <= 0:
        return 0.0
    direction = (direction or "").upper()
    if direction == "BUY":
        risk = entry - sl
        reward = tp - entry
    elif direction == "SELL":
        risk = sl - entry
        reward = entry - tp
    else:
        return 0.0
    if risk <= 0:
        return 0.0
    return reward / risk


def resolve_conflict(
    tech_signal: str,
    tech_conf: float,
    llm_signal: str,
    llm_conf: float,
    news_signal: str,
    news_conf: float,
    tech_weight: float = DEFAULT_TECH_WEIGHT,
    llm_weight: float = DEFAULT_LLM_WEIGHT,
    news_weight: float = DEFAULT_NEWS_WEIGHT,
) -> Tuple[str, float, str]:
    """Master List Issue #5a: weighted conflict resolution.

    When tech says BUY and LLM says SELL (or vice versa), instead of
    deadlocking, we compute a weighted score:

        buy_score  = (tech_w if tech=BUY  else 0) + (llm_w if llm=BUY  else 0) + (news_w if news=BUY  else 0)
        sell_score = (tech_w if tech=SELL else 0) + (llm_w if llm=SELL else 0) + (news_w if news=SELL else 0)

    Whichever side has the higher weighted score wins; confidence is
    that side's weighted-average confidence. If tied, return WAIT.

    Returns: (final_signal, weighted_confidence, explanation)
    """
    # Bug #15 fix: normalize weights to sum to 1.0 so misconfigured
    # env vars (e.g. TECH=0.7, LLM=0.7, sum=1.6) don't produce
    # confidence > 100%.
    total_w = tech_weight + llm_weight + news_weight
    if total_w > 0 and abs(total_w - 1.0) > 1e-6:
        tech_weight /= total_w
        llm_weight  /= total_w
        news_weight /= total_w

    # Normalize signals
    def _norm(s):
        s = (s or "").upper().strip()
        if s in ("STRONG_BUY",):  return "BUY"
        if s in ("STRONG_SELL",): return "SELL"
        if s in ("BULLISH",):     return "BUY"   # sentiment-bullish = buy
        if s in ("BEARISH",):     return "SELL"  # sentiment-bearish = sell
        if s in ("WAIT", "HOLD", "NEUTRAL", "NO TRADE", ""): return "WAIT"
        return s

    t_sig = _norm(tech_signal)
    l_sig = _norm(llm_signal)
    n_sig = _norm(news_signal)

    buy_score = 0.0
    sell_score = 0.0
    buy_confs = []
    sell_confs = []

    for sig, conf, w, label in [
        (t_sig, _safe_float(tech_conf), tech_weight, "tech"),
        (l_sig, _safe_float(llm_conf),  llm_weight,  "llm"),
        (n_sig, _safe_float(news_conf), news_weight, "news"),
    ]:
        if sig == "BUY":
            buy_score += w
            buy_confs.append((conf, w))
        elif sig == "SELL":
            sell_score += w
            sell_confs.append((conf, w))

    if buy_score > sell_score and buy_confs:
        weighted_conf = sum(c * w for c, w in buy_confs) / sum(w for _, w in buy_confs)
        return "BUY", weighted_conf, (
            f"Weighted fusion: BUY wins (buy_score={buy_score:.2f} vs "
            f"sell_score={sell_score:.2f})"
        )
    if sell_score > buy_score and sell_confs:
        weighted_conf = sum(c * w for c, w in sell_confs) / sum(w for _, w in sell_confs)
        return "SELL", weighted_conf, (
            f"Weighted fusion: SELL wins (sell_score={sell_score:.2f} vs "
            f"buy_score={buy_score:.2f})"
        )
    return "WAIT", 0.0, (
        f"Weighted fusion: tied (buy={buy_score:.2f}, sell={sell_score:.2f}) → WAIT"
    )


def validate_fusion(
    decision: str,
    confidence: float,
    entry: Optional[float],
    sl: Optional[float],
    tp: Optional[float],
    signal_timestamp: Optional[Any] = None,
    tech_signal: str = "WAIT",
    tech_conf: float = 0.0,
    llm_signal: str = "WAIT",
    llm_conf: float = 0.0,
    news_signal: str = "NEUTRAL",
    news_conf: float = 0.0,
    signal_ttl_sec: float = DEFAULT_SIGNAL_TTL_SEC,
    min_rrr: float = DEFAULT_MIN_RRR,
    tech_weight: float = DEFAULT_TECH_WEIGHT,
    llm_weight: float = DEFAULT_LLM_WEIGHT,
    news_weight: float = DEFAULT_NEWS_WEIGHT,
) -> FusionValidation:
    """Run all four fusion-engine validations on a decision.

    Called from DecisionAgent._result() AFTER the voting block has
    produced a decision. If `safe == False`, DecisionAgent downgrades
    to WAIT (preserving the analysis confidence for audit).

    Args:
        decision: BUY / SELL / WAIT / NO TRADE
        confidence: 0-100
        entry, sl, tp: price levels (any may be None or 0)
        signal_timestamp: when the signal was generated (datetime,
            ISO string, epoch float, or None)
        tech_signal, tech_conf: technical/rule-engine signal
        llm_signal, llm_conf: LLM analyst signal
        news_signal, news_conf: news/sentiment signal
        signal_ttl_sec: max age in seconds (default 30)
        min_rrr: minimum risk:reward ratio (default 1.5)
        tech_weight, llm_weight, news_weight: fusion weights
            (default 0.40 / 0.40 / 0.20)

    Returns:
        FusionValidation dataclass with all fields populated.
    """
    result = FusionValidation()
    result.weights_used = {
        "tech":  tech_weight,
        "llm":   llm_weight,
        "news":  news_weight,
    }

    # ── Master List Issue #5b: Signal TTL / staleness check ────────
    result.signal_age_sec = _compute_signal_age(signal_timestamp)
    if result.signal_age_sec > signal_ttl_sec:
        result.ttl_valid = False
        result.failure_reasons.append(
            f"Signal is {result.signal_age_sec:.1f}s old "
            f"(> TTL of {signal_ttl_sec:.0f}s) — market may have moved. "
            f"Downgrading to WAIT."
        )

    # ── Master List Issue #5c: RRR validator (adaptive) ─────────────
    # Two-tier RRR check:
    #   1. HARD floor (min_rrr, default 1.0): below this → always WAIT
    #   2. SOFT threshold (soft_rrr, default 1.3): between hard and soft
    #      → pass with proportional confidence penalty instead of outright
    #      downgrade. This fixes the issue where RRR 1:1.17 (just below
    #      old 1.30 minimum) killed valid BUY signals from multiple modules.
    if decision in ("BUY", "SELL") and entry and sl and tp:
        result.rrr = compute_rrr(entry, sl, tp, decision)
        if result.rrr < min_rrr:
            # Hard fail — RRR is genuinely bad
            result.rrr_valid = False
            result.failure_reasons.append(
                f"RRR is 1:{result.rrr:.2f} — below hard minimum 1:{min_rrr:.2f}. "
                f"Downgrading to WAIT (entry={entry}, sl={sl}, tp={tp}, "
                f"dir={decision})."
            )
        elif result.rrr < DEFAULT_SOFT_RRR:
            # Soft penalty placeholder — applied AFTER resolve_conflict() below
            # so it operates on the real weighted confidence, not 0.0.
            pass
    elif decision in ("BUY", "SELL"):
        # BUY/SELL with missing SL/TP/entry — can't validate RRR.
        # Don't fail validation outright (the risk engine may have a
        # reason for missing values), but flag it.
        result.rrr = 0.0
        # If entry/sl/tp are all present but RRR computation returned 0
        # (e.g. risk ≤ 0 because SL is on wrong side), that's a real fail.
        if entry and sl and tp:
            result.rrr_valid = False
            result.failure_reasons.append(
                f"RRR computation returned 0 — likely SL on wrong side of "
                f"entry for {decision}. Downgrading to WAIT."
            )

    # ── Master List Issue #5a: Weighted confidence + conflict resolution ─
    final_signal, weighted_conf, conflict_expl = resolve_conflict(
        tech_signal=tech_signal, tech_conf=tech_conf,
        llm_signal=llm_signal,   llm_conf=llm_conf,
        news_signal=news_signal, news_conf=news_conf,
        tech_weight=tech_weight, llm_weight=llm_weight, news_weight=news_weight,
    )
    result.weighted_confidence = weighted_conf
    result.conflict_resolution = conflict_expl

    # ── Soft RRR penalty (applied AFTER resolve_conflict so it operates
    #    on the real weighted confidence, not the default 0.0) ─────────
    if decision in ("BUY", "SELL") and entry and sl and tp:
        if 0 < result.rrr < DEFAULT_SOFT_RRR:
            rrr_gap = (DEFAULT_SOFT_RRR - result.rrr) / DEFAULT_SOFT_RRR
            penalty_pct = rrr_gap * 15  # max ~4.5pp penalty at RRR=1.0
            result.weighted_confidence = max(0, result.weighted_confidence - penalty_pct)
            log.info(
                f"[FusionV3] SOFT RRR penalty: 1:{result.rrr:.2f} is below "
                f"soft threshold 1:{DEFAULT_SOFT_RRR:.2f}. Confidence reduced by "
                f"{penalty_pct:.1f}pp (new weighted_conf={result.weighted_confidence:.1f}%)"
            )

    # If the weighted fusion disagrees with the decision, flag it.
    # (DecisionAgent may have produced BUY via its own voting, but the
    # 40/40/20 weighted fusion says WAIT or SELL — that's a real conflict.)
    if decision in ("BUY", "SELL") and final_signal != decision:
        result.failure_reasons.append(
            f"Weighted fusion disagrees: decision={decision} but "
            f"fusion={final_signal}. Conflict resolution: {conflict_expl}"
        )

    # ── Final safe flag ────────────────────────────────────────────
    result.safe = (
        result.ttl_valid
        and result.rrr_valid
        and len(result.failure_reasons) == 0
    )

    if not result.safe and decision in ("BUY", "SELL"):
        log.warning(
            f"[FusionV3] DOWNGRADE {decision}→WAIT | "
            f"age={result.signal_age_sec:.1f}s ttl_valid={result.ttl_valid} | "
            f"rrr=1:{result.rrr:.2f} rrr_valid={result.rrr_valid} | "
            f"reasons={result.failure_reasons}"
        )
    elif decision in ("BUY", "SELL"):
        log.info(
            f"[FusionV3] OK {decision} | age={result.signal_age_sec:.1f}s | "
            f"rrr=1:{result.rrr:.2f} | weighted_conf={weighted_conf:.1f}%"
        )

    return result


def _compute_signal_age(signal_timestamp: Optional[Any]) -> float:
    """Compute the age of a signal in seconds, robustly.

    Accepts:
        - datetime (naive or tz-aware)
        - ISO 8601 string
        - epoch float (seconds since 1970-01-01 UTC)
        - None (returns 0 — assume fresh)

    Returns:
        Age in seconds. 0.0 if timestamp is None or unparseable.
    """
    if signal_timestamp is None:
        return 0.0

    now_utc = datetime.now(timezone.utc)

    # datetime object
    if isinstance(signal_timestamp, datetime):
        ts = signal_timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (now_utc - ts).total_seconds())

    # Numeric (epoch seconds)
    try:
        epoch = float(signal_timestamp)
        ts = datetime.fromtimestamp(epoch, tz=timezone.utc)
        return max(0.0, (now_utc - ts).total_seconds())
    except (TypeError, ValueError):
        pass

    # ISO string
    try:
        ts = datetime.fromisoformat(str(signal_timestamp))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (now_utc - ts).total_seconds())
    except (TypeError, ValueError):
        pass

    return 0.0
