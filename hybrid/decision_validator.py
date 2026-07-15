# hybrid/decision_validator.py  —  Day 49 | Decision Conflict Handler ⭐⭐⭐⭐⭐
# ============================================================
# Doc Step 5 (Decision Fusion) + Bonus #1 (Decision Conflict Handler)।
#
# এটা DecisionAgent (Day 42)-এর replacement না — DecisionAgent এর
# নিজের vote-counting logic আছে (Master/LLM/Rule weighted votes +
# sentiment conflict)। এই module তার আগে বসে একটা narrower, harder
# question-এর উত্তর দেয়:
#
#       "Quant engine (rule+LLM+Master মিলিয়ে) যা বলছে, আর Vision AI
#        (Day 47 ChartReader) চার্ট দেখে যা বলছে — এই দুটো কি
#        fundamentally disagree করছে?"
#
# Doc-এর rule অক্ষরে অক্ষরে:
#       API = BUY, Vision = SELL  →  No trade. কখনো force trade নয়।
#
# আর confluence হলে doc-এর weighted formula:
#       final_score = quant_score * 0.6 + vision_score * 0.4
#
# Day 47-এর ChartReader.fuse_with_quant() এর সাথে এর তুলনা:
#   fuse_with_quant()   → confidence ADJUST করে (conflict→ -15%, agree→ +8%)
#   DecisionValidator    → conflict severe হলে HARD BLOCK করে (doc-এর
#                          "no trade, কখনো force না" rule অনুযায়ী),
#                          নাহলে doc-এর 60/40 weighted score বের করে
#
# দুটো একসাথে ব্যবহার করাই উদ্দেশ্য — fuse_with_quant() soft adjustment,
# DecisionValidator hard gate। FlowController উভয়ই call করে।
# ============================================================

from utils.logger import get_logger

log = get_logger("decision_validator")

# Doc formula weight
QUANT_WEIGHT = 0.6
VISION_WEIGHT = 0.4

# Vision confidence এই থ্রেশহোল্ডের নিচে থাকলে vision opinion-কে
# "unreliable" ধরা হয় (যেমন vision API fail করলে fallback 0% confidence
# আসে চার্ট_reader.py-এর _fallback() থেকে) — তখন conflict-check skip করে
# শুধু quant-এর উপর নির্ভর করা হয়, কিন্তু confidence-এ penalty দেওয়া হয়।
MIN_RELIABLE_VISION_CONF = 30


class DecisionValidator:
    """
    Usage:
        validator = DecisionValidator()
        verdict = validator.validate(
            quant_signal="BUY", quant_confidence=80,
            vision_signal="BUY", vision_confidence=85,
        )
        if verdict["final_signal"] == "NO TRADE":
            ... block ...
        else:
            final_conf = verdict["final_score"]
    """

    def validate(
        self,
        quant_signal: str,
        quant_confidence: int,
        vision_signal: str,
        vision_confidence: int,
        vision_available: bool = True,
    ) -> dict:
        """
        quant_signal / vision_signal: "BUY" | "SELL" | "WAIT" | "NO TRADE"

        Returns:
            {
                "final_signal": "BUY"|"SELL"|"WAIT"|"NO TRADE",
                "final_score": int (0-99),
                "has_hard_conflict": bool,
                "reason": str,
                "quant_signal": ..., "vision_signal": ...,
            }
        """
        quant_norm = self._normalize(quant_signal)
        vision_norm = self._normalize(vision_signal)

        # H-C5/H-M2 fix: quant_confidence/vision_confidence may arrive as
        # None (e.g. vision API failure before a fallback fills it in) or
        # out-of-range (>100). Comparing None to MIN_RELIABLE_VISION_CONF
        # below would raise TypeError; an out-of-range value would silently
        # skew the weighted fusion score. Coerce + clamp both to 0-100.
        quant_confidence = self._coerce_confidence(quant_confidence, "quant_confidence")
        vision_confidence = self._coerce_confidence(vision_confidence, "vision_confidence")

        # ── Vision অনুপলব্ধ/অবিশ্বাস্য হলে quant-only decision, confidence penalty সহ ──
        if not vision_available or vision_confidence < MIN_RELIABLE_VISION_CONF:
            penalized_conf = max(0, round(quant_confidence * 0.85))
            result = {
                "final_signal": quant_norm if quant_norm in ("BUY", "SELL") else "NO TRADE",
                "final_score": penalized_conf,
                "has_hard_conflict": False,
                "reason": "Vision unavailable/unreliable — quant-only decision (penalized)",
                "quant_signal": quant_norm,
                "vision_signal": vision_norm,
            }
            log.info(
                f"[DecisionValidator] Vision unreliable (conf={vision_confidence}) — "
                f"quant-only → {result['final_signal']} ({penalized_conf}%)"
            )
            return result

        # ── Doc rule: hard opposite directions → কখনো force trade নয় ──
        if self._is_hard_conflict(quant_norm, vision_norm):
            result = {
                "final_signal": "NO TRADE",
                "final_score": 0,
                "has_hard_conflict": True,
                "reason": (
                    f"Quant says {quant_norm} but Vision says {vision_norm} — "
                    f"direct conflict, blocking trade (no force trade rule)"
                ),
                "quant_signal": quant_norm,
                "vision_signal": vision_norm,
            }
            log.warning(
                f"[DecisionValidator] 🚫 HARD CONFLICT — Quant={quant_norm} "
                f"Vision={vision_norm} → NO TRADE"
            )
            return result

        # ── No hard conflict → direction প্রথমে ঠিক করো, তারপর সেই direction-এর
        #    জন্য doc-এর weighted fusion score বের করো (vision-এর irrelevant/WAIT
        #    score দিয়ে winning side-কে drag-down করা ভুল — তাই score সবসময়
        #    "যে direction জিতেছে" তার দুই sidescore-এর উপর ভিত্তি করে বের হয়) ──
        if quant_norm in ("BUY", "SELL"):
            # quant primary source (API দ্রুত/accurate — doc-এর core thesis)
            final_signal = quant_norm
            quant_score = quant_confidence
            # vision একই direction-এ agree করলেই তার confidence ব্লেন্ডে যোগ হয়;
            # vision disagree (hard conflict) আগেই block হয়ে গেছে, আর vision
            # neutral/WAIT হলে তার score blend-এ না নিয়ে শুধু quant-এর confidence-ই
            # final score (vision থাকলে শুধু supporting bonus হিসেবে কিছুটা যুক্ত হয়)
            if vision_norm == quant_norm:
                vision_score = vision_confidence
                final_score = round(quant_score * QUANT_WEIGHT + vision_score * VISION_WEIGHT)
            else:
                final_score = quant_score   # vision neutral — pure quant confidence
        elif vision_norm in ("BUY", "SELL") and vision_confidence >= 70:
            final_signal = vision_norm   # quant indecisive, vision strongly confirms a direction
            final_score = vision_confidence
        else:
            final_signal = "NO TRADE"
            final_score = 0

        result = {
            "final_signal": final_signal,
            "final_score": final_score,
            "has_hard_conflict": False,
            "reason": (
                f"Quant {quant_norm} ({quant_confidence}%) + Vision {vision_norm} "
                f"({vision_confidence}%) → fused {final_score}%"
            ),
            "quant_signal": quant_norm,
            "vision_signal": vision_norm,
        }
        log.info(
            f"[DecisionValidator] ✅ Fusion | Quant={quant_norm}({quant_confidence}%) "
            f"Vision={vision_norm}({vision_confidence}%) → {final_signal} ({final_score}%)"
        )
        return result

    # ─────────────────────────────────────────────
    # CONFLICT DEFINITION
    # ─────────────────────────────────────────────

    def _is_hard_conflict(self, quant: str, vision: str) -> bool:
        """শুধু সরাসরি opposite direction-কে hard conflict ধরা হয় — WAIT vs BUY
        conflict নয় (vision uncertain হতেই পারে), কিন্তু BUY vs SELL conflict।"""
        opposite_pairs = {("BUY", "SELL"), ("SELL", "BUY")}
        return (quant, vision) in opposite_pairs

    def _normalize(self, signal: str) -> str:
        s = (signal or "WAIT").upper().strip()
        # H-H12 fix: AnalysisAgent/MasterAnalyst can emit graded signals like
        # "STRONG_BUY"/"STRONG_SELL" (not just plain BUY/SELL) — these were
        # previously falling through to the catch-all "WAIT", silently
        # discarding a valid directional signal.
        if s in ("BUY", "STRONG_BUY", "STRONG BUY"):
            return "BUY"
        if s in ("SELL", "STRONG_SELL", "STRONG SELL"):
            return "SELL"
        if s in ("WAIT", "NO TRADE", "WAIT_FOR_CONFIRMATION", "HOLD"):
            return "WAIT"
        return "WAIT"

    def _coerce_confidence(self, value, field_name: str) -> int:
        """H-C5/H-M2 fix: coerce to int and clamp to 0-100. Handles None
        (e.g. vision unavailable before a fallback fills it in) and
        out-of-range values from upstream agents."""
        if value is None:
            return 0
        try:
            value = int(round(float(value)))
        except (TypeError, ValueError):
            log.warning(f"[DecisionValidator] Invalid {field_name}={value!r} — defaulting to 0")
            return 0
        return max(0, min(100, value))

    # ─────────────────────────────────────────────
    # PRINT SUMMARY
    # ─────────────────────────────────────────────

    def print_summary(self, verdict: dict) -> None:
        bar = "═" * 50
        icon = {"BUY": "🟢", "SELL": "🔴", "WAIT": "🟡", "NO TRADE": "⛔"}.get(
            verdict["final_signal"], "⚪"
        )
        print(f"\n{bar}")
        print("  🔀  DECISION VALIDATOR  (Day 49)")
        print(bar)
        print(f"  Quant Signal   : {verdict['quant_signal']}")
        print(f"  Vision Signal  : {verdict['vision_signal']}")
        print(f"  Hard Conflict  : {'⚠️ YES' if verdict['has_hard_conflict'] else '✅ NO'}")
        print(f"  Final          : {icon} {verdict['final_signal']} ({verdict['final_score']}%)")
        print(f"  Reason         : {verdict['reason']}")
        print(bar + "\n")