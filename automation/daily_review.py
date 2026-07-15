# automation/daily_review.py  —  Day 51 Bonus #3 | Daily Self Review ⭐
# ============================================================
# Doc:
#     আজ কয়টি decision নিয়েছি? কোন decision ভুল ছিল? কেন ভুল ছিল?
#     আগামীকাল rule পরিবর্তন দরকার?
#
# এই AI কোনো toy bot না — একজন autonomous professional human trader-এর
# মতো কাজ করার কথা। তাই দিন শেষের review পরিসংখ্যান report না, বরং
# agents/master_analyst.py-এর LLM-reasoning pattern অনুসরণ করে আসলে
# *কেন* ভুল হলো তা বুঝে রুল-পরিবর্তনের suggestion দেওয়া — ঠিক যেভাবে
# একজন professional trader দিনের শেষে নিজের journal লেখে।
#
# Input data: LearningAgent.get_performance_stats() + closed trades থেকে
# (নতুন storage বানানো হয়নি, existing memory/trade_memory.json reuse)।
# mistake_analyzer.py (Day 19)-এর per-trade analysis-এর উপরের স্তর —
# এটা প্রতিটা trade আলাদা না, পুরো দিনটা একসাথে দেখে pattern বের করে।
# ============================================================

import json
import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()
log = get_logger("daily_review")

# ── LLM client init — Groq (primary) + Gemini (fallback) via KeyManager ──
LLM_AVAILABLE = False
_groq_client = None
_gemini_client = None
_key_manager = None
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS = 1200
REVIEW_LOG_DIR = "memory/daily_reviews"

try:
    from core.llm_key_manager import get_llm_key_manager
    _key_manager = get_llm_key_manager()
    _groq_client = _key_manager.get_groq_client()
    if _groq_client is not None:
        LLM_AVAILABLE = True
        log.info(f"[DailyReview] Groq client initialized | model={MODEL}")
    if not LLM_AVAILABLE:
        _gemini_client = _key_manager.get_gemini_client()
        if _gemini_client is not None:
            MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            LLM_AVAILABLE = True
            log.info(f"[DailyReview] Gemini client initialized (fallback) | model={MODEL}")
except Exception as e:
    log.warning(f"[DailyReview] LLMKeyManager init failed: {e} — trying single-key")
    groq_key = os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY", "")
    if groq_key:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=groq_key)
            LLM_AVAILABLE = True
        except Exception as e2:
            log.warning(f"[DailyReview] Groq init failed: {e2}")
    if not LLM_AVAILABLE:
        gemini_key = os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY", "")
        if gemini_key:
            try:
                from google import genai as google_genai
                _gemini_client = google_genai.Client(api_key=gemini_key)
                MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
                LLM_AVAILABLE = True
            except Exception as e2:
                log.warning(f"[DailyReview] Gemini init failed: {e2}")

if not LLM_AVAILABLE:
    log.warning("[DailyReview] No LLM available — daily review will use heuristic fallback")

_SYSTEM = """You are an elite professional forex trader doing your end-of-day self-review,
exactly the way a disciplined human trader journals after a trading session.

You will receive a summary of today's decisions, executed trades, and outcomes from an
autonomous AI trading system you are responsible for. Your job is not to recite statistics
back — you already have the numbers. Your job is to find the REASONING pattern behind
mistakes, the way an experienced trader reviews their own psychology and process, not just
their P&L.

Rules:
1. Be honest and specific — vague praise or vague blame is useless for tomorrow's trading.
2. For each mistake, identify the root cause: was it a bad signal, bad timing, ignored
   conflict, bad risk sizing, or an external shock (news/spread)?
3. Suggest CONCRETE rule changes only when the evidence supports it — do not suggest
   changes after a single unlucky trade; look for repeated patterns.
4. If today was clean (no real mistakes), say so plainly — false self-criticism is also
   a discipline failure for a professional trader.

Output ONLY valid JSON. No markdown, no extra text, no code blocks.

JSON schema:
{
  "decisions_taken": integer,
  "mistakes_found": ["mistake 1", "mistake 2"],
  "root_causes": ["root cause 1", "root cause 2"],
  "what_went_well": ["thing 1", "thing 2"],
  "suggested_rule_changes": ["concrete suggestion 1", "concrete suggestion 2"],
  "overall_assessment": "2-3 sentence honest summary of today's trading discipline",
  "confidence_in_tomorrow": "HIGH" | "MEDIUM" | "LOW"
}"""


class DailyReview:
    """
    Usage:
        review = DailyReview()
        result = review.run(
            cycles=metrics._cycles,          # RuntimeMetrics-এর cycle log
            closed_trades=learning_agent._load(),   # বা LearningAgent থেকে
            performance_stats=learning_agent.get_performance_stats(),
            shadow_stats=execution_router.get_shadow_stats(),
        )
        review.save(result)
        review.print_summary(result)
    """

    def run(
        self,
        cycles: list,
        closed_trades: list = None,
        performance_stats: dict = None,
        shadow_stats: dict = None,
        error_summary: dict = None,
    ) -> dict:
        context = self._build_context(
            cycles, closed_trades or [], performance_stats or {},
            shadow_stats or {}, error_summary or {},
        )

        if not LLM_AVAILABLE:
            return self._fallback_review(cycles, closed_trades or [])

        try:
            raw = self._call_llm(context)
            parsed = self._parse_response(raw)
            parsed["llm_raw"] = raw
            parsed["error"] = None
        except Exception as e:
            log.error(f"[DailyReview] LLM error: {e}")
            parsed = self._fallback_review(cycles, closed_trades or [])
            parsed["error"] = str(e)

        parsed["generated_at"] = datetime.now(timezone.utc).isoformat()
        log.info(
            f"[DailyReview] Review complete — {len(parsed.get('mistakes_found', []))} mistakes, "
            f"assessment: {parsed.get('overall_assessment', '')[:60]}"
        )
        return parsed

    # ─────────────────────────────────────────────
    # CONTEXT BUILDER
    # ─────────────────────────────────────────────

    def _build_context(self, cycles, closed_trades, performance_stats, shadow_stats, error_summary) -> str:
        outcome_breakdown = {}
        for c in cycles:
            outcome_breakdown[c["outcome"]] = outcome_breakdown.get(c["outcome"], 0) + 1

        losing_trades = [t for t in closed_trades if t.get("result") == "LOSS"]
        winning_trades = [t for t in closed_trades if t.get("result") == "WIN"]

        ctx = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "total_cycles": len(cycles),
            "outcome_breakdown": outcome_breakdown,
            "closed_trades_count": len(closed_trades),
            "wins": len(winning_trades),
            "losses": len(losing_trades),
            "losing_trade_details": [
                {
                    "pair": t.get("symbol"), "confidence": t.get("confidence"),
                    "reasons": t.get("reasons", [])[:2],
                    "regime": t.get("regime"), "rsi": t.get("rsi"),
                }
                for t in losing_trades[:10]   # বেশি হলে context overload এড়াতে cap
            ],
            "performance_stats": performance_stats,
            "shadow_mode_stats": shadow_stats,
            "error_summary": error_summary,
        }
        return json.dumps(ctx, indent=2, default=str)

    def _call_llm(self, context: str) -> str:
        user_prompt = (
            "এখানে আজকের সম্পূর্ণ trading session-এর data:\n\n"
            f"{context}\n\n"
            "একজন professional trader-এর মতো honest self-review করো এবং JSON ফেরত দাও।"
        )
        # Primary: Groq
        if _groq_client is not None:
            try:
                resp = _groq_client.chat.completions.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                log.warning(f"[DailyReview] Groq call failed: {e} — trying Gemini")
        # Fallback: Gemini
        if _gemini_client is not None:
            try:
                full_prompt = f"{_SYSTEM}\n\n{user_prompt}"
                resp = _gemini_client.models.generate_content(model=MODEL, contents=full_prompt)
                return resp.text.strip()
            except Exception as e:
                log.error(f"[DailyReview] Gemini call failed: {e}")
                raise
        raise RuntimeError("[DailyReview] No LLM available")

    def _parse_response(self, raw: str) -> dict:
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        text = re.sub(r"\s*```$", "", text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise

        data.setdefault("decisions_taken", 0)
        data.setdefault("mistakes_found", [])
        data.setdefault("root_causes", [])
        data.setdefault("what_went_well", [])
        data.setdefault("suggested_rule_changes", [])
        data.setdefault("overall_assessment", "")
        data.setdefault("confidence_in_tomorrow", "MEDIUM")
        return data

    def _fallback_review(self, cycles: list, closed_trades: list) -> dict:
        """LLM unavailable হলে rule-based bare-minimum (no fake reasoning claimed)।"""
        losses = [t for t in closed_trades if t.get("result") == "LOSS"]
        return {
            "decisions_taken": len(cycles),
            "mistakes_found": [f"{len(losses)} losing trades (LLM unavailable — no root-cause analysis)"],
            "root_causes": [],
            "what_went_well": [],
            "suggested_rule_changes": [],
            "overall_assessment": "LLM review unavailable — raw stats only, no qualitative judgment possible.",
            "confidence_in_tomorrow": "MEDIUM",
            "llm_raw": "",
        }

    # ─────────────────────────────────────────────
    # SAVE / PRINT
    # ─────────────────────────────────────────────

    def save(self, result: dict) -> str:
        os.makedirs(REVIEW_LOG_DIR, exist_ok=True)
        date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = os.path.join(REVIEW_LOG_DIR, f"review_{date_tag}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            log.info(f"[DailyReview] Saved → {path}")
        except Exception as e:
            log.error(f"[DailyReview] Save error: {e}")
        return path

    def print_summary(self, result: dict) -> None:
        bar = "═" * 56
        print(f"\n{bar}")
        print("  📓  DAILY SELF REVIEW  (Day 51)")
        print(bar)
        print(f"  Decisions taken : {result.get('decisions_taken')}")
        print(f"  Confidence (tmrw): {result.get('confidence_in_tomorrow')}")
        print()
        if result.get("mistakes_found"):
            print("  ── Mistakes ──")
            for m in result["mistakes_found"]:
                print(f"  ⚠  {m}")
        if result.get("root_causes"):
            print("\n  ── Root Causes ──")
            for r in result["root_causes"]:
                print(f"  →  {r}")
        if result.get("what_went_well"):
            print("\n  ── What Went Well ──")
            for w in result["what_went_well"]:
                print(f"  ✅ {w}")
        if result.get("suggested_rule_changes"):
            print("\n  ── Suggested Rule Changes ──")
            for s in result["suggested_rule_changes"]:
                print(f"  🔧 {s}")
        print(f"\n  Overall: {result.get('overall_assessment')}")
        if result.get("error"):
            print(f"\n  ⚠ Error: {result['error']}")
        print(bar + "\n")