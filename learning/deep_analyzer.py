# learning/deep_analyzer.py  —  Day 52 | Advanced Mistake Analyzer (Self-Learning Intelligence Layer)
# ============================================================
# Day 52 Core File — AI এখন শুধু trade নেবে না, হারলে বুঝবে কেন,
# rule update করবে, এবং ভবিষ্যতে একই ভুল কমাবে।
#
# Updated Learning Loop:
#   Trade Executed → Trade Result → Win/Loss Detection
#   → Deep Mistake Analyzer → Root Cause Finding
#   → Lesson Generation → Strategy Adjustment
#   → Future Decision Improvement
#
# 10/10 Features:
#   ⭐ Loss Context Collection     — full market context at time of loss
#   ⭐ Root Cause Analysis         — pattern/timeframe/SL analysis
#   ⭐ Statistical Validation      — minimum 5 same mistakes before rule update
#   ⭐ Counterfactual Analysis     — "what if I waited / took H1 confirmation?"
#   ⭐ A/B Strategy Testing        — compare filtered vs unfiltered strategy
#   ⭐ Confidence Calibration      — adjust if stated 80% but win rate 55%
#   ⭐ Human Approval Gate         — suggests rule change, you approve first
# ============================================================

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

from utils.logger import get_logger

log = get_logger("learning.deep_analyzer")

# ── LLM client init — Groq (primary) + Gemini (fallback) via KeyManager ──
LLM_AVAILABLE = False
_groq_client = None
_gemini_client = None
_key_manager = None
MODEL = ""
MAX_TOK = 1200

import os as _os
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()

try:
    from core.llm_key_manager import get_llm_key_manager
    _key_manager = get_llm_key_manager()
    _groq_client = _key_manager.get_groq_client()
    if _groq_client is not None:
        MODEL = _os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        LLM_AVAILABLE = True
        log.info(f"[DeepAnalyzer] Groq client initialized | model={MODEL}")
    if not LLM_AVAILABLE:
        _gemini_client = _key_manager.get_gemini_client()
        if _gemini_client is not None:
            MODEL = _os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            LLM_AVAILABLE = True
            log.info(f"[DeepAnalyzer] Gemini client initialized (fallback) | model={MODEL}")
except Exception as e:
    log.warning(f"[DeepAnalyzer] LLMKeyManager init failed: {e} — trying single-key")
    _groq_key = _os.getenv("GROQ_API_KEY_1") or _os.getenv("GROQ_API_KEY", "")
    if _groq_key:
        try:
            from groq import Groq as _Groq
            _groq_client = _Groq(api_key=_groq_key)
            MODEL = _os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            LLM_AVAILABLE = True
            log.info(f"[DeepAnalyzer] Groq client initialized (single-key) | model={MODEL}")
        except Exception as _e:
            log.warning(f"[DeepAnalyzer] Groq init failed: {_e}")
    if not LLM_AVAILABLE:
        _gemini_key = _os.getenv("GEMINI_API_KEY_1") or _os.getenv("GEMINI_API_KEY", "")
        if _gemini_key:
            try:
                from google import genai as _google_genai
                _gemini_client = _google_genai.Client(api_key=_gemini_key)
                MODEL = _os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
                LLM_AVAILABLE = True
                log.info(f"[DeepAnalyzer] Gemini client initialized (single-key) | model={MODEL}")
            except Exception as _e:
                log.warning(f"[DeepAnalyzer] Gemini init failed: {_e}")

if not LLM_AVAILABLE:
    log.warning("[DeepAnalyzer] No LLM available — heuristic fallback active")

# ── Storage Paths ─────────────────────────────────────────────
LESSON_MEMORY_PATH     = "memory/lesson_memory.json"
RULE_STORE_PATH        = "memory/pattern_rules.json"
PENDING_APPROVALS_PATH = "memory/pending_rule_approvals.json"
ANALYSIS_LOG_PATH      = "memory/deep_analysis_log.json"

MISTAKE_THRESHOLD = 5   # একই ভুল কতবার হলে rule update হবে


# ═══════════════════════════════════════════════════════════════
# DEEP MISTAKE ANALYZER
# ═══════════════════════════════════════════════════════════════

class DeepMistakeAnalyzer:
    """
    Day 52 Main Class — AI-এর Self-Learning Intelligence Layer।

    একটা সাধারণ loss analyzer না — এটা প্রতিটা loss-এর পরে:
      1. Full context সংগ্রহ করে
      2. Root cause বের করে (pattern/timeframe/SL)
      3. Counterfactual চিন্তা করে ("যদি WAIT করতাম?")
      4. Statistically validate করে (5+ same mistake?)
      5. Confidence calibrate করে
      6. Rule change suggest করে (human approval দরকার)
      7. Future decision-এ inject করে

    Usage:
        analyzer = DeepMistakeAnalyzer()

        # Trade close হওয়ার পর:
        result = analyzer.analyze_loss(trade_context)

        # Decision নেওয়ার আগে past lessons check করো:
        lessons = analyzer.get_relevant_lessons(pattern="Bullish Engulfing", regime="RANGING")

        # Pending rule approvals দেখো:
        analyzer.get_pending_approvals()
    """

    _SYSTEM = """You are an elite forex trading post-mortem analyst and self-learning AI system.

Your job: analyze a losing trade deeply, find the ROOT CAUSE, and generate an actionable lesson.

Think like a 20-year veteran trader reviewing their trading journal — honest, specific, data-driven.

Rules:
1. Be brutally honest. Vague analysis like "market was unpredictable" is USELESS.
2. Identify the SPECIFIC structural mistake: wrong market regime, timeframe conflict, SL too tight, pattern misread.
3. Generate a CONCRETE rule, not a platitude.
4. Suggest confidence adjustment only if the mistake is structural (not random noise).
5. Flag if this same mistake has appeared before — pattern repetition is critical.

Output ONLY valid JSON. No markdown, no extra text.

JSON schema:
{
  "loss_reason": "One clear sentence explaining why this trade failed",
  "error_type": "REGIME_MISMATCH | TIMEFRAME_CONFLICT | SL_TOO_TIGHT | PATTERN_FAILURE | NEWS_SURPRISE | SPREAD_ISSUE | OVERCONFIDENCE",
  "pattern_failed": "pattern name or null",
  "regime_at_time": "TRENDING | RANGING | VOLATILE | UNKNOWN",
  "timeframe_conflict": "description or null (e.g. H4 bearish vs M15 bullish)",
  "risk_issue": "SL/RR problem description or null",
  "what_happened": "2-3 sentence detailed explanation of the failure mechanism",
  "lesson": "A strict, actionable rule for the future",
  "counterfactual": {
    "what_if_waited": "outcome if WAIT was chosen",
    "what_if_h1_confirm": "outcome if H1 confirmation was required",
    "better_action": "WAIT | SMALLER_SIZE | DIFFERENT_ENTRY | SKIP"
  },
  "rule_update": {
    "pattern": "pattern this applies to",
    "condition": "market condition to avoid",
    "confidence_adjustment": -20
  },
  "severity": "HIGH | MEDIUM | LOW"
}"""

    def __init__(self):
        os.makedirs("memory", exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # 1. MAIN: ANALYZE A LOSS TRADE
    # ──────────────────────────────────────────────────────────

    def analyze_loss(self, trade_context: dict) -> dict:
        """
        Loss trade-এর পরে Deep Analysis চালাও।

        trade_context example:
        {
            "pair": "EURUSD",
            "timeframe": "M15",
            "entry": 1.0850,
            "exit": 1.0820,
            "pattern": "Bullish Engulfing",
            "rsi": 35,
            "macd": "bullish_cross",
            "market_regime": "RANGING",
            "news": "none",
            "spread": 1.5,
            "atr": 0.0035,
            "sl_pips": 20,
            "h4_trend": "bearish",
            "confidence_at_entry": 75,
            "pnl": -25.0
        }
        """
        log.info(f"[DeepAnalyzer] Analyzing LOSS — {trade_context.get('pair')} {trade_context.get('timeframe')}")

        # Step 1: Context collect করো
        context = self._collect_loss_context(trade_context)

        # Step 2: LLM root cause analysis
        analysis = self._run_llm_analysis(context)

        # Step 3: Counterfactual analysis
        counterfactual = self._build_counterfactual(trade_context, analysis)
        analysis["counterfactual"] = counterfactual

        # Step 4: Statistical validation — এই ভুল কতবার হয়েছে?
        validation = self._validate_statistically(analysis)
        analysis["statistical_validation"] = validation

        # Step 5: Confidence calibration
        calibration = self._calibrate_confidence(
            stated_confidence=trade_context.get("confidence_at_entry", 70),
            error_type=analysis.get("error_type", "UNKNOWN"),
            pattern=analysis.get("pattern_failed"),
        )
        analysis["confidence_calibration"] = calibration

        # Step 6: Rule update (human approval gate)
        if validation.get("should_update_rule"):
            rule_proposal = self._propose_rule_update(analysis, validation)
            analysis["rule_proposal"] = rule_proposal
            self._queue_for_approval(rule_proposal, analysis)
            log.info(f"[DeepAnalyzer] ⏸️ Rule update queued for human approval: {rule_proposal}")

        # Step 7: Lesson save
        self._save_lesson(trade_context, analysis)
        self._log_analysis(trade_context, analysis)

        self._print_analysis_summary(analysis)
        return analysis

    # ──────────────────────────────────────────────────────────
    # 2. LOSS CONTEXT COLLECTION
    # ──────────────────────────────────────────────────────────

    def _collect_loss_context(self, trade: dict) -> dict:
        """
        Trade-এর সম্পূর্ণ context সংগ্রহ করো।
        Doc-এর JSON format অনুযায়ী।
        """
        atr     = trade.get("atr", 0.0035)
        sl_pips = trade.get("sl_pips", 20)
        pip_val = 0.0001 if "JPY" not in trade.get("pair", "") else 0.01
        atr_pips = round(atr / pip_val)

        sl_analysis = "SL too tight" if sl_pips < atr_pips * 1.2 else "SL adequate"

        # Past lessons for this pattern
        past = self._get_past_lessons_for_pattern(
            pattern=trade.get("pattern", ""),
            regime=trade.get("market_regime", ""),
        )

        context = {
            "trade_snapshot": {
                "pair":          trade.get("pair"),
                "timeframe":     trade.get("timeframe"),
                "entry":         trade.get("entry"),
                "exit":          trade.get("exit"),
                "pnl":           trade.get("pnl"),
                "pattern":       trade.get("pattern"),
                "rsi":           trade.get("rsi"),
                "macd":          trade.get("macd"),
                "market_regime": trade.get("market_regime"),
                "news":          trade.get("news", "none"),
                "spread":        trade.get("spread"),
                "confidence":    trade.get("confidence_at_entry", 70),
            },
            "sl_analysis": {
                "sl_pips":   sl_pips,
                "atr_pips":  atr_pips,
                "verdict":   sl_analysis,
            },
            "timeframe_context": {
                "entry_tf": trade.get("timeframe"),
                "h4_trend": trade.get("h4_trend", "unknown"),
                "h1_trend": trade.get("h1_trend", "unknown"),
                "conflict":  (
                    trade.get("h4_trend", "") != "" and
                    trade.get("h4_trend", "neutral") != "neutral"
                ),
            },
            "past_similar_losses": past[:3],
        }

        return context

    # ──────────────────────────────────────────────────────────
    # 3. LLM ROOT CAUSE ANALYSIS
    # ──────────────────────────────────────────────────────────

    def _run_llm_analysis(self, context: dict) -> dict:
        """LLM দিয়ে root cause analysis করো।"""
        if not LLM_AVAILABLE:
            return self._heuristic_analysis(context)

        prompt = (
            "Analyze this losing trade and find the root cause:\n\n"
            f"{json.dumps(context, indent=2, default=str)}\n\n"
            "Return your analysis as JSON following the schema exactly."
        )

        try:
            # Primary: Groq
            if _groq_client is not None:
                resp = _groq_client.chat.completions.create(
                    model=MODEL,
                    max_tokens=MAX_TOK,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": self._SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw = resp.choices[0].message.content.strip()
            # Fallback: Gemini
            elif _gemini_client is not None:
                full_prompt = f"{self._SYSTEM}\n\n{prompt}"
                resp = _gemini_client.models.generate_content(model=MODEL, contents=full_prompt)
                raw = resp.text.strip()
            else:
                return self._heuristic_analysis(context)

            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw).strip()
            result = json.loads(raw)
            result["llm_analyzed"] = True
            return result
        except Exception as e:
            log.error(f"[DeepAnalyzer] LLM analysis failed: {e}")
            return self._heuristic_analysis(context)

    def _heuristic_analysis(self, context: dict) -> dict:
        """LLM ছাড়া rule-based heuristic analysis।"""
        snap = context.get("trade_snapshot", {})
        sl   = context.get("sl_analysis", {})
        tf   = context.get("timeframe_context", {})

        # Simple rule-based root cause
        if tf.get("conflict") and tf.get("h4_trend") in ("bearish", "BEARISH"):
            error_type = "TIMEFRAME_CONFLICT"
            reason = f"M15 bullish signal against H4 bearish trend"
            lesson = f"Avoid {snap.get('pattern')} reversal signals when H4 trend is opposing"
        elif snap.get("market_regime") in ("RANGING", "ranging"):
            error_type = "REGIME_MISMATCH"
            reason = "Reversal pattern in ranging market — low probability setup"
            lesson = f"Avoid {snap.get('pattern')} in RANGING market conditions"
        elif sl.get("verdict") == "SL too tight":
            error_type = "SL_TOO_TIGHT"
            reason = f"SL {sl.get('sl_pips')}p below ATR requirement {sl.get('atr_pips')}p"
            lesson = "Always set SL >= 1.5x ATR. Never trade with SL below market noise"
        else:
            error_type = "PATTERN_FAILURE"
            reason = "Pattern failed due to market variance"
            lesson = "Review confluence requirements before entry"

        return {
            "loss_reason":        reason,
            "error_type":         error_type,
            "pattern_failed":     snap.get("pattern"),
            "regime_at_time":     snap.get("market_regime"),
            "timeframe_conflict": f"H4 {tf.get('h4_trend')} vs {snap.get('timeframe')} bullish" if tf.get("conflict") else None,
            "risk_issue":         sl.get("verdict") if sl.get("verdict") != "SL adequate" else None,
            "what_happened":      reason,
            "lesson":             lesson,
            "rule_update": {
                "pattern":              snap.get("pattern"),
                "condition":            snap.get("market_regime"),
                "confidence_adjustment": -15,
            },
            "severity":       "MEDIUM",
            "llm_analyzed":   False,
        }

    # ──────────────────────────────────────────────────────────
    # 4. COUNTERFACTUAL ANALYSIS  ⭐⭐⭐⭐⭐
    # ──────────────────────────────────────────────────────────

    def _build_counterfactual(self, trade: dict, analysis: dict) -> dict:
        """
        "যদি অন্য সিদ্ধান্ত নিতাম?" — AI নিজেই এই প্রশ্ন করে।

        Example:
          BUY নিলাম → Loss
          যদি WAIT করতাম?        → "Loss avoided"
          যদি H1 confirm নিতাম?  → "H1 showed bearish, would have skipped"
        """
        error = analysis.get("error_type", "")
        regime = trade.get("market_regime", "")
        h4 = trade.get("h4_trend", "neutral")
        pnl = trade.get("pnl", -10)

        # Scenario 1: WAIT করলে কী হতো?
        if error in ("REGIME_MISMATCH", "PATTERN_FAILURE"):
            wait_outcome = f"Loss of {abs(pnl):.1f} avoided — WAIT was the right call in {regime} market"
        elif error == "TIMEFRAME_CONFLICT":
            wait_outcome = f"Loss avoided — H4 {h4} trend signaled caution"
        else:
            wait_outcome = "Outcome uncertain — loss may have been unavoidable"

        # Scenario 2: H1 confirmation নিলে কী হতো?
        if h4 in ("bearish", "BEARISH"):
            h1_outcome = "H1 timeframe likely showed bearish continuation — would have skipped entry"
        elif regime in ("RANGING", "ranging"):
            h1_outcome = "H1 would show no clear trend — confirmation would have prevented entry"
        else:
            h1_outcome = "H1 confirmation uncertain — might have still entered"

        # Scenario 3: Smaller size নিলে?
        smaller_outcome = f"Loss reduced to ~{abs(pnl) * 0.5:.1f} with 50% position size"

        # Best alternative
        if error == "TIMEFRAME_CONFLICT":
            better_action = "WAIT"
        elif error == "SL_TOO_TIGHT":
            better_action = "DIFFERENT_ENTRY"
        elif error == "REGIME_MISMATCH":
            better_action = "SKIP"
        else:
            better_action = "SMALLER_SIZE"

        return {
            "what_if_waited":      wait_outcome,
            "what_if_h1_confirm":  h1_outcome,
            "what_if_smaller_size": smaller_outcome,
            "better_action":       better_action,
            "conclusion": (
                f"Best alternative was to {better_action}. "
                f"Estimated outcome: {wait_outcome}"
            ),
        }

    # ──────────────────────────────────────────────────────────
    # 5. STATISTICAL VALIDATION  ⭐⭐⭐⭐⭐
    # ──────────────────────────────────────────────────────────

    def _validate_statistically(self, analysis: dict) -> dict:
        """
        এই একই ভুল কতবার হয়েছে? MISTAKE_THRESHOLD (5) পার হলে rule update।

        Overfitting এড়ানোর জন্য single trade-এ rule change হয় না।
        """
        pattern  = analysis.get("pattern_failed", "unknown")
        error    = analysis.get("error_type", "unknown")
        regime   = analysis.get("regime_at_time", "unknown")

        lessons = self._load_lessons()
        same_mistakes = [
            l for l in lessons
            if l.get("error_type") == error
            and l.get("pattern_failed") == pattern
            and l.get("regime_at_time") == regime
        ]

        count = len(same_mistakes)
        should_update = count >= MISTAKE_THRESHOLD

        # Loss rate for this pattern+regime
        all_with_pattern = [l for l in lessons if l.get("pattern_failed") == pattern]
        loss_rate = round(count / max(len(all_with_pattern), 1) * 100, 1)

        return {
            "same_mistake_count":  count,
            "threshold":           MISTAKE_THRESHOLD,
            "should_update_rule":  should_update,
            "loss_rate_pct":       loss_rate,
            "pattern_regime_combo": f"{pattern} in {regime}",
            "verdict": (
                f"Rule update triggered ({count}/{MISTAKE_THRESHOLD} occurrences)"
                if should_update else
                f"Monitoring — {count}/{MISTAKE_THRESHOLD} occurrences, not yet threshold"
            ),
        }

    # ──────────────────────────────────────────────────────────
    # 6. CONFIDENCE CALIBRATION  ⭐⭐⭐⭐⭐
    # ──────────────────────────────────────────────────────────

    def _calibrate_confidence(
        self,
        stated_confidence: int,
        error_type: str,
        pattern: Optional[str],
    ) -> dict:
        """
        AI stated confidence 80% কিন্তু win rate 55%?
        → Confidence over-estimated, adjust downward.

        Example:
          Stated: 80%
          Actual win rate for this pattern: 55%
          → Gap: 25% → AI was overconfident
          → Future confidence for this setup: 55%
        """
        lessons = self._load_lessons()

        # Calculate actual win rate for this pattern
        pattern_lessons = [l for l in lessons if l.get("pattern_failed") == pattern]
        if not pattern_lessons:
            actual_win_rate = stated_confidence  # no data, assume stated
        else:
            wins = [l for l in pattern_lessons if l.get("outcome") == "WIN"]
            actual_win_rate = round(len(wins) / len(pattern_lessons) * 100, 1)

        gap = stated_confidence - actual_win_rate
        overconfident = gap > 15

        if overconfident:
            recommended_conf = max(30, actual_win_rate + 5)
            verdict = f"OVERCONFIDENT — stated {stated_confidence}% but actual {actual_win_rate}%. Reduce to {recommended_conf}%"
        elif gap < -10:
            recommended_conf = min(90, actual_win_rate - 5)
            verdict = f"UNDERCONFIDENT — stated {stated_confidence}% but actual {actual_win_rate}%"
        else:
            recommended_conf = stated_confidence
            verdict = f"CALIBRATED — stated {stated_confidence}% close to actual {actual_win_rate}%"

        return {
            "stated_confidence":     stated_confidence,
            "actual_win_rate":       actual_win_rate,
            "gap":                   gap,
            "is_overconfident":      overconfident,
            "recommended_confidence": recommended_conf,
            "verdict":               verdict,
            "data_points":           len(pattern_lessons),
        }

    # ──────────────────────────────────────────────────────────
    # 7. RULE UPDATE + HUMAN APPROVAL  ⭐⭐⭐⭐⭐
    # ──────────────────────────────────────────────────────────

    def _propose_rule_update(self, analysis: dict, validation: dict) -> dict:
        """Rule change proposal তৈরি করো (human approve করার আগে active হবে না)।"""
        current_rules = self._load_rules()
        pattern = analysis.get("pattern_failed", "unknown")
        regime  = analysis.get("regime_at_time", "unknown")

        # Current confidence for this pattern
        key = f"{pattern}_{regime}"
        current_conf = current_rules.get(key, {}).get("confidence", 75)
        new_conf = max(20, current_conf + analysis.get("rule_update", {}).get("confidence_adjustment", -15))

        return {
            "proposal_id":         f"rule_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "pattern":             pattern,
            "condition":           regime,
            "rule_key":            key,
            "current_confidence":  current_conf,
            "proposed_confidence": new_conf,
            "change":              new_conf - current_conf,
            "based_on":            validation.get("verdict"),
            "lesson":              analysis.get("lesson"),
            "status":              "PENDING_APPROVAL",
            "created_at":          datetime.now(timezone.utc).isoformat(),
        }

    def _queue_for_approval(self, proposal: dict, analysis: dict) -> None:
        """Human approval-এর জন্য queue করো।"""
        pending = self._load_json(PENDING_APPROVALS_PATH, [])
        pending.append({
            "proposal":  proposal,
            "analysis":  {k: v for k, v in analysis.items() if k != "llm_raw"},
            "queued_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save_json(PENDING_APPROVALS_PATH, pending)

    def approve_rule_change(self, proposal_id: str) -> dict:
        """
        Human approval দিলে rule activate হবে।

        Usage:
            analyzer.approve_rule_change("rule_20240120_143022")
        """
        pending = self._load_json(PENDING_APPROVALS_PATH, [])
        approved = None

        remaining = []
        for item in pending:
            if item["proposal"]["proposal_id"] == proposal_id:
                approved = item["proposal"]
            else:
                remaining.append(item)

        if not approved:
            return {"success": False, "reason": "Proposal not found"}

        # Rule activate করো
        rules = self._load_rules()
        rules[approved["rule_key"]] = {
            "confidence":   approved["proposed_confidence"],
            "pattern":      approved["pattern"],
            "condition":    approved["condition"],
            "lesson":       approved.get("lesson", ""),
            "activated_at": datetime.now(timezone.utc).isoformat(),
        }

        self._save_json(RULE_STORE_PATH, rules)
        self._save_json(PENDING_APPROVALS_PATH, remaining)

        log.info(f"[DeepAnalyzer] ✅ Rule APPROVED and activated: {approved['rule_key']} → conf={approved['proposed_confidence']}%")
        return {"success": True, "rule": rules[approved["rule_key"]]}

    def reject_rule_change(self, proposal_id: str) -> dict:
        """Human rule change reject করলে queue থেকে সরিয়ে দাও।"""
        pending = self._load_json(PENDING_APPROVALS_PATH, [])
        remaining = [p for p in pending if p["proposal"]["proposal_id"] != proposal_id]
        self._save_json(PENDING_APPROVALS_PATH, remaining)
        log.info(f"[DeepAnalyzer] ❌ Rule REJECTED: {proposal_id}")
        return {"success": True}

    # ──────────────────────────────────────────────────────────
    # 8. A/B STRATEGY TESTING  ⭐⭐⭐⭐⭐
    # ──────────────────────────────────────────────────────────

    def run_ab_test(self, pattern: str, regime: str) -> dict:
        """
        Strategy A (without filter) vs Strategy B (with new rule) compare।

        Returns backtest-style comparison from lesson memory.
        """
        lessons = self._load_lessons()

        # Strategy A: all trades with this pattern
        strategy_a = [l for l in lessons if l.get("pattern_failed") == pattern]

        # Strategy B: trades with this pattern but EXCLUDING the bad regime
        strategy_b = [
            l for l in lessons
            if l.get("pattern_failed") == pattern
            and l.get("regime_at_time") != regime
        ]

        def stats(trades):
            if not trades:
                return {"trades": 0, "win_rate": 0, "avg_pnl": 0}
            wins = [t for t in trades if t.get("outcome") == "WIN"]
            pnls = [t.get("pnl", 0) for t in trades if t.get("pnl") is not None]
            return {
                "trades":   len(trades),
                "win_rate": round(len(wins) / len(trades) * 100, 1),
                "avg_pnl":  round(sum(pnls) / len(pnls), 2) if pnls else 0,
            }

        a_stats = stats(strategy_a)
        b_stats = stats(strategy_b)

        improvement = b_stats["win_rate"] - a_stats["win_rate"]
        verdict = (
            f"✅ Filter HELPS — win rate improves from {a_stats['win_rate']}% to {b_stats['win_rate']}% (+{improvement:.1f}%)"
            if improvement > 5 else
            f"⚠️ Filter NEUTRAL — marginal difference ({improvement:.1f}%)"
            if improvement >= 0 else
            f"❌ Filter HURTS — win rate drops {abs(improvement):.1f}%"
        )

        result = {
            "pattern":     pattern,
            "filter":      f"Exclude {regime} regime",
            "strategy_a":  {"description": "No filter", **a_stats},
            "strategy_b":  {"description": f"Filter out {regime}", **b_stats},
            "improvement": improvement,
            "verdict":     verdict,
            "recommendation": "APPLY_FILTER" if improvement > 5 else "KEEP_MONITORING",
        }

        log.info(f"[DeepAnalyzer] A/B Test | {verdict}")
        return result

    # ──────────────────────────────────────────────────────────
    # 9. QUERY LESSONS FOR DECISION AGENT
    # ──────────────────────────────────────────────────────────

    def get_relevant_lessons(self, pattern: str = None, regime: str = None, limit: int = 5) -> list:
        """
        নতুন setup দেখলে AI জিজ্ঞেস করবে: এই পরিস্থিতিতে আগে কী হয়েছিল?

        Returns relevant lessons sorted by recency.
        """
        lessons = self._load_lessons()

        filtered = []
        for l in lessons:
            match = True
            if pattern and l.get("pattern_failed") != pattern:
                match = False
            if regime and l.get("regime_at_time") != regime:
                match = False
            if match:
                filtered.append(l)

        # Sort by date desc
        filtered.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return filtered[:limit]

    def get_rule_for_setup(self, pattern: str, regime: str) -> dict:
        """
        Current setup-এর জন্য rule আছে কিনা দেখো।
        DecisionAgent এটা দিয়ে confidence adjust করতে পারবে।
        """
        rules = self._load_rules()
        key = f"{pattern}_{regime}"
        rule = rules.get(key)

        if rule:
            log.info(f"[DeepAnalyzer] Rule found for {key}: conf={rule.get('confidence')}%")
            return {"has_rule": True, "rule": rule, "key": key}
        return {"has_rule": False, "key": key}

    def get_memory_context_for_decision(self, pattern: str, regime: str) -> dict:
        """
        DecisionAgent-এ inject করার জন্য memory context।

        Example output:
            "গত 10 বার ranging + engulfing = 7 loss → Avoid"
        """
        lessons = self.get_relevant_lessons(pattern=pattern, regime=regime, limit=10)
        rule = self.get_rule_for_setup(pattern, regime)

        losses = [l for l in lessons if l.get("outcome") == "LOSS"]
        wins   = [l for l in lessons if l.get("outcome") == "WIN"]

        if not lessons:
            return {"has_memory": False}

        summary = (
            f"গত {len(lessons)} বার {regime} + {pattern} = {len(losses)} loss, {len(wins)} win"
        )
        recommendation = "AVOID" if len(losses) > len(wins) * 1.5 else "CAUTION"

        return {
            "has_memory":     True,
            "pattern":        pattern,
            "regime":         regime,
            "total_trades":   len(lessons),
            "losses":         len(losses),
            "wins":           len(wins),
            "loss_rate":      round(len(losses) / len(lessons) * 100, 1),
            "summary":        summary,
            "recommendation": recommendation,
            "rule":           rule.get("rule") if rule.get("has_rule") else None,
            "lessons":        [l.get("lesson", "") for l in lessons[:3]],
        }

    # ──────────────────────────────────────────────────────────
    # 10. PENDING APPROVALS VIEWER
    # ──────────────────────────────────────────────────────────

    def get_pending_approvals(self) -> list:
        """Pending rule changes দেখো।"""
        pending = self._load_json(PENDING_APPROVALS_PATH, [])
        return pending

    def print_pending_approvals(self) -> None:
        """Human-readable format-এ pending approvals দেখাও।"""
        pending = self.get_pending_approvals()
        bar = "═" * 60
        print(f"\n{bar}")
        print("  📋  PENDING RULE APPROVALS  (Day 52)")
        print(bar)

        if not pending:
            print("  ✅ No pending approvals — all rules up to date")
        else:
            for i, item in enumerate(pending, 1):
                p = item["proposal"]
                print(f"\n  [{i}] ID: {p['proposal_id']}")
                print(f"      Pattern : {p['pattern']} in {p['condition']}")
                print(f"      Change  : {p['current_confidence']}% → {p['proposed_confidence']}% ({p['change']:+d}%)")
                print(f"      Reason  : {p['based_on']}")
                print(f"      Lesson  : {p['lesson']}")
                print(f"      Created : {p['created_at'][:19]}")
                print()
                print(f"      To APPROVE: analyzer.approve_rule_change('{p['proposal_id']}')")
                print(f"      To REJECT : analyzer.reject_rule_change('{p['proposal_id']}')")

        print(bar + "\n")

    # ──────────────────────────────────────────────────────────
    # STORAGE HELPERS
    # ──────────────────────────────────────────────────────────

    def _save_lesson(self, trade: dict, analysis: dict) -> None:
        """Lesson memory-তে save করো।"""
        lessons = self._load_lessons()
        lessons.append({
            "id":             len(lessons) + 1,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "pair":           trade.get("pair"),
            "timeframe":      trade.get("timeframe"),
            "pattern_failed": analysis.get("pattern_failed"),
            "error_type":     analysis.get("error_type"),
            "regime_at_time": analysis.get("regime_at_time"),
            "lesson":         analysis.get("lesson"),
            "outcome":        "LOSS",
            "pnl":            trade.get("pnl"),
        })
        self._save_json(LESSON_MEMORY_PATH, lessons[-1000:])  # শেষ 1000 রাখো

    def _log_analysis(self, trade: dict, analysis: dict) -> None:
        """Full analysis log করো।"""
        log_entries = self._load_json(ANALYSIS_LOG_PATH, [])
        safe_analysis = {k: v for k, v in analysis.items() if k != "llm_raw"}
        log_entries.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trade":     {k: v for k, v in trade.items()},
            "analysis":  safe_analysis,
        })
        self._save_json(ANALYSIS_LOG_PATH, log_entries[-500:])

    def _get_past_lessons_for_pattern(self, pattern: str, regime: str) -> list:
        lessons = self._load_lessons()
        return [
            l for l in lessons
            if l.get("pattern_failed") == pattern and l.get("regime_at_time") == regime
        ][-5:]

    def _load_lessons(self) -> list:
        return self._load_json(LESSON_MEMORY_PATH, [])

    def _load_rules(self) -> dict:
        return self._load_json(RULE_STORE_PATH, {})

    def _load_json(self, path: str, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _save_json(self, path: str, data) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            log.error(f"[DeepAnalyzer] Save error ({path}): {e}")

    # ──────────────────────────────────────────────────────────
    # PRINT SUMMARY
    # ──────────────────────────────────────────────────────────

    def _print_analysis_summary(self, analysis: dict) -> None:
        bar = "═" * 62
        severity_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(
            analysis.get("severity", ""), "⚪"
        )
        print(f"\n{bar}")
        print("  🔍  DEEP MISTAKE ANALYZER  (Day 52)")
        print(bar)
        print(f"  {severity_icon}  Severity      : {analysis.get('severity')}")
        print(f"  Error Type    : {analysis.get('error_type')}")
        print(f"  Pattern       : {analysis.get('pattern_failed', 'N/A')}")
        print(f"  Regime        : {analysis.get('regime_at_time')}")
        if analysis.get("timeframe_conflict"):
            print(f"  TF Conflict   : {analysis.get('timeframe_conflict')}")
        if analysis.get("risk_issue"):
            print(f"  Risk Issue    : {analysis.get('risk_issue')}")
        print()
        print(f"  ── Root Cause ──")
        print(f"  {analysis.get('loss_reason', 'N/A')}")
        print()
        print(f"  ── Lesson ──")
        print(f"  {analysis.get('lesson', 'N/A')}")
        print()

        # Counterfactual
        cf = analysis.get("counterfactual", {})
        if cf:
            print(f"  ── Counterfactual Analysis ──")
            print(f"  If WAIT    : {cf.get('what_if_waited', 'N/A')}")
            print(f"  If H1 conf : {cf.get('what_if_h1_confirm', 'N/A')}")
            print(f"  Best action: {cf.get('better_action', 'N/A')}")
            print()

        # Statistical validation
        sv = analysis.get("statistical_validation", {})
        if sv:
            print(f"  ── Statistical Validation ──")
            print(f"  {sv.get('verdict', '')}")
            if sv.get("should_update_rule"):
                print(f"  ⚡ RULE UPDATE QUEUED FOR HUMAN APPROVAL")
            print()

        # Confidence calibration
        cc = analysis.get("confidence_calibration", {})
        if cc:
            print(f"  ── Confidence Calibration ──")
            print(f"  {cc.get('verdict', '')}")

        print(bar + "\n")