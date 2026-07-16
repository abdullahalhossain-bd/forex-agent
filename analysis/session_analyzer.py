# analysis/session_analyzer.py  —  Day 63 | Session-Based Intelligence Engine
# ============================================================
# AI-এর Market Time Intelligence Layer।
#
# Features:
#   ✅ Current session detection (GMT-aware)
#   ✅ DST adjustment (US & EU daylight saving)
#   ✅ Session transition detector ⭐
#   ✅ Strategy mode auto-switcher ⭐⭐⭐⭐⭐
#   ✅ Pair specialization (priority scoring)
#   ✅ Session confidence score
#   ✅ Dead zone protection
#   ✅ London manipulation window detection
#   ✅ Session performance memory hook
#   ✅ SMC + Session fusion
# ============================================================

from datetime import datetime, timezone, timedelta
from analysis.session_rules import (
    SESSION_WINDOWS,
    SESSION_CHARACTERISTICS,
    SESSION_STRATEGIES,
    SMC_REQUIREMENTS,
    DEAD_ZONES,
    DEAD_ZONES_ENABLED,
    LONDON_OPEN_WINDOW,
)
from analysis.pair_session_map import (
    get_pair_priority,
    get_preferred_pairs,
    get_pair_session_recommendation,
)
from utils.logger import get_logger

log = get_logger("session_analyzer")


class SessionAnalyzer:
    """
    Day 63 — Session-Based Market Intelligence।

    AI এখন জানবে:
    - কোন session চলছে
    - সেই session-এ কী strategy ব্যবহার করবে
    - কোন pair সবচেয়ে ভালো
    - কখন trade করবে না
    - London manipulation window কি active
    - SMC + Session fusion score

    Usage:
        analyzer = SessionAnalyzer()
        result   = analyzer.analyze(pair="EURUSD", smc_ctx={})
        ctx      = analyzer.get_ai_context(result)
    """

    def __init__(self):
        self._session_performance: dict = {}   # memory hook — future learning

    # ═══════════════════════════════════════════════════════════
    # STEP 1: CURRENT SESSION DETECTION
    # ═══════════════════════════════════════════════════════════

    def get_current_session(self, dt: datetime = None) -> dict:
        """
        Current GMT time দেখে active session(s) detect করো।

        DST note:
          US DST  → NY session shifts 13:00-22:00 → 12:00-21:00
          EU DST  → London session shifts 08:00-17:00 → 07:00-16:00
          This implementation uses a pytz-free approximation.

        Returns:
            {
                "primary_session": "LONDON",
                "active_sessions": ["LONDON"],
                "gmt_hour": 9,
                "is_overlap": False,
                "is_dead_zone": False,
                "london_open_window": False,
            }
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        gmt_hour    = dt.hour
        gmt_minute  = dt.minute
        gmt_decimal = gmt_hour + gmt_minute / 60.0

        # ── DST adjustment (Round-17: zoneinfo-based, was fixed-date approx) ──
        # Previously: _is_us_dst()/_is_eu_dst() used fixed-date heuristics
        # (e.g. "March >= 8", "November < 7") which could be off by ~1 hour
        # for 4-8 days per year around DST transitions (floating Sundays).
        #
        # Now: use zoneinfo to get the ACTUAL UTC offset for America/New_York
        # and Europe/London at the current moment. This is 100% accurate —
        # the OS timezone database tracks the exact DST transition rules.
        # zoneinfo is in the Python 3.9+ standard library.
        us_dst, eu_dst = self._get_dst_flags(dt)

        # Adjust session windows if DST active
        ny_start = 12 if us_dst else 13
        ny_end   = 21 if us_dst else 22
        ld_start = 7  if eu_dst  else 8
        ld_end   = 16 if eu_dst  else 17
        ov_start = max(ny_start, ld_start)   # overlap start
        ov_end   = min(ny_end,   ld_end)     # overlap end

        active_sessions = []

        # Check overlap first (most important)
        if ov_start <= gmt_hour < ov_end:
            active_sessions.append("LONDON_NY_OVERLAP")

        # Round-17: simplified dead if/elif branches.
        # Previously had confusing if/elif where both branches did the same
        # append. Now: single condition — append if in window and not already present.
        if ld_start <= gmt_hour < ld_end and "LONDON" not in active_sessions:
            active_sessions.append("LONDON")

        if ny_start <= gmt_hour < ny_end and "NEW_YORK" not in active_sessions:
            active_sessions.append("NEW_YORK")

        # Tokyo (00:00–09:00)
        if 0 <= gmt_hour < 9:
            active_sessions.append("TOKYO")

        # Sydney (22:00–07:00, crosses midnight)
        if gmt_hour >= 22 or gmt_hour < 7:
            active_sessions.append("SYDNEY")

        # Dead zone check
        is_dead = self._is_dead_zone(gmt_hour)

        # London open manipulation window (08:00–10:00 or 07:00-09:00 DST)
        london_open = ld_start <= gmt_hour < (ld_start + 2)

        # Primary session priority: OVERLAP > LONDON > NEW_YORK > TOKYO > SYDNEY
        priority = ["LONDON_NY_OVERLAP", "LONDON", "NEW_YORK", "TOKYO", "SYDNEY"]
        primary  = next((s for s in priority if s in active_sessions), "BETWEEN_SESSIONS")

        if is_dead:
            primary = "DEAD_ZONE"

        return {
            "primary_session":      primary,
            "active_sessions":      active_sessions,
            "gmt_hour":             gmt_hour,
            "gmt_minute":           gmt_minute,
            "gmt_time":             dt.strftime("%H:%M GMT"),
            "is_overlap":           "LONDON_NY_OVERLAP" in active_sessions,
            "is_dead_zone":         is_dead,
            "london_open_window":   london_open,
            "us_dst_active":        us_dst,
            "eu_dst_active":        eu_dst,
        }

    # ═══════════════════════════════════════════════════════════
    # STEP 2: SESSION BEHAVIOR ANALYSIS
    # ═══════════════════════════════════════════════════════════

    def analyze_session_behavior(self, session: str) -> dict:
        """
        Session-এর volatility, behavior, characteristics return করো।
        """
        char = SESSION_CHARACTERISTICS.get(session, SESSION_CHARACTERISTICS["BETWEEN_SESSIONS"])
        return {
            "session":     session,
            "volatility":  char["volatility"],
            "behavior":    char["behavior"],
            "description": char["description"],
            "risk_level":  char["risk_level"],
        }

    # ═══════════════════════════════════════════════════════════
    # STEP 3: STRATEGY MODE SELECTOR  ⭐⭐⭐⭐⭐
    # ═══════════════════════════════════════════════════════════

    def get_strategy_mode(self, session: str, gmt_hour: int = None) -> dict:
        """
        Session অনুযায়ী strategy automatically select করো।

        Returns:
            {
                "strategy": "LONDON_BREAKOUT",
                "action":   "Asian range breakout...",
                "avoid":    "Counter-trend...",
                "min_confidence": 70,
                "risk_multiplier": 1.0,
                "trade_allowed": True,
            }
        """
        strat = SESSION_STRATEGIES.get(session, SESSION_STRATEGIES["BETWEEN_SESSIONS"]).copy()

        # Extra: London open sub-window (first 2h) → check Asian range
        if session == "LONDON" and gmt_hour is not None:
            if LONDON_OPEN_WINDOW["start"] <= gmt_hour < LONDON_OPEN_WINDOW["end"]:
                strat["sub_strategy"] = "LONDON_MANIPULATION_WINDOW"
                strat["action"]       = (
                    "Monitor Asian high/low sweep. Wait for liquidity grab + BOS before entry."
                )
                strat["alert"]        = "⚠️  London manipulation window active — wait for sweep then entry"

        strat["trade_allowed"] = (session != "DEAD_ZONE")
        strat["session"]       = session
        return strat

    # ═══════════════════════════════════════════════════════════
    # STEP 4: PAIR PREFERENCE
    # ═══════════════════════════════════════════════════════════

    def get_pair_preference(self, pair: str, session: str) -> dict:
        """
        Pair + Session combination-এর priority ও recommendation।
        """
        rec = get_pair_session_recommendation(pair, session)
        preferred = get_preferred_pairs(session)
        rec["preferred_pairs"] = preferred
        rec["is_preferred"]    = pair.upper().replace("/", "")[:6] in [
            p.replace("/", "")[:6] for p in preferred
        ]
        return rec

    # ═══════════════════════════════════════════════════════════
    # STEP 5: SESSION TRANSITION DETECTOR  ⭐
    # ═══════════════════════════════════════════════════════════

    def detect_session_transition(self, gmt_hour: int) -> dict:
        """
        Session transition point কাছে কিনা detect করো।

        London open (08:00): Asian range formed → manipulation likely
        NY open (13:00):     London trend established → continuation
        Session close windows: liquidity drain possible

        Returns:
            {
                "in_transition": True,
                "transition_type": "LONDON_OPEN",
                "minutes_away": 0,
                "alert": "London manipulation window active",
                "action": "Wait for Asian range sweep then enter"
            }
        """
        transitions = [
            {
                "name":   "LONDON_OPEN",
                "hour":   8,
                "window": 1,   # ±1 hour
                "alert":  "London open — Asian range liquidity sweep imminent",
                "action": "Watch Asian high/low. Enter after sweep + BOS confirmation.",
            },
            {
                "name":   "NY_OPEN",
                "hour":   13,
                "window": 1,
                "alert":  "New York open — high volatility, trend acceleration possible",
                "action": "Confirm London trend direction. Look for pullback entry.",
            },
            {
                "name":   "LONDON_CLOSE",
                "hour":   17,
                "window": 1,
                "alert":  "London close — liquidity may drain. Reduce position size.",
                "action": "Close or trail existing trades. Avoid new entries.",
            },
            {
                "name":   "TOKYO_OPEN",
                "hour":   0,
                "window": 1,
                "alert":  "Tokyo open — range formation begins. Wait for range to establish.",
                "action": "Observe first 30–60 min. Range trade after formation.",
            },
        ]

        for t in transitions:
            dist = abs(gmt_hour - t["hour"])
            # Handle midnight wrap
            dist = min(dist, 24 - dist)
            if dist <= t["window"]:
                return {
                    "in_transition":   True,
                    "transition_type": t["name"],
                    "hours_away":      dist,
                    "alert":           t["alert"],
                    "action":          t["action"],
                }

        return {"in_transition": False, "transition_type": None}

    # ═══════════════════════════════════════════════════════════
    # STEP 6: SESSION CONFIDENCE SCORE
    # ═══════════════════════════════════════════════════════════

    def calculate_session_confidence(
        self,
        session:      str,
        pair:         str,
        smc_ctx:      dict = None,
        signal_conf:  int  = 0,
    ) -> dict:
        """
        Session + Pair + SMC + Signal diye final session confidence দাও।

        Scoring:
            London + preferred pair  : +20
            Liquidity sweep present  : +25
            BOS confirmed            : +25
            FVG present              : +15
            Signal confidence        : +15 (scaled)

        Total possible: 100
        """
        smc_ctx = smc_ctx or {}
        score   = 0
        reasons = []

        # ── Session + Pair alignment ──────────────────────────
        pair_priority = get_pair_priority(pair, session)
        pair_score    = round(pair_priority * 0.20)   # max 20
        score += pair_score
        reasons.append(f"+{pair_score} Session-Pair match ({pair} in {session})")

        # ── SMC factors ───────────────────────────────────────
        if smc_ctx.get("smc_factors", {}).get("liquidity_sweep"):
            score += 25
            reasons.append("+25 Liquidity sweep present")

        if smc_ctx.get("smc_factors", {}).get("bos"):
            score += 25
            reasons.append("+25 BOS confirmed")

        if smc_ctx.get("smc_factors", {}).get("fvg"):
            score += 15
            reasons.append("+15 FVG active")

        # ── Signal confidence ─────────────────────────────────
        sig_contribution = round(min(signal_conf, 100) * 0.15)
        score += sig_contribution
        reasons.append(f"+{sig_contribution} Signal confidence ({signal_conf}%)")

        score = min(100, score)

        # ── Grade ─────────────────────────────────────────────
        if score >= 85:     grade = "A+"
        elif score >= 70:   grade = "A"
        elif score >= 55:   grade = "B"
        else:               grade = "C"

        return {
            "session_score":   score,
            "session_grade":   grade,
            "score_reasons":   reasons,
            "session":         session,
            "pair":            pair,
        }

    # ═══════════════════════════════════════════════════════════
    # STEP 7: SMC + SESSION FUSION  ⭐⭐⭐⭐⭐
    # ═══════════════════════════════════════════════════════════

    def session_smc_fusion(
        self,
        session:  str,
        smc_ctx:  dict,
        signal:   str,
    ) -> dict:
        """
        Session requirements + SMC score fusion — simplified.

        Plain-language rule: does this session allow trading right now,
        and if so, does the SMC setup clear that session's bar?

          1. No SMC data yet            → PENDING, don't block.
          2. SMC score < session minimum → blocked.
          3. Session requires BOS but it's missing → blocked.
          4. Otherwise → allowed, and fusion_score IS the SMC score
             (no extra formula — what you see is what SMC actually found).

        Note: the old version also checked `require_ob` (order block
        required), but that flag is False for every session except
        DEAD_ZONE — and DEAD_ZONE already blocks everything via
        min_smc_score=999. So it never changed an outcome; removed as
        dead logic rather than kept as a check that can never fail.

        Returns:
            {
                "fusion_allowed": True,
                "fusion_score": 72,
                "fusion_grade": "A",
                "issues": [],
                "reason": "...",
            }
        """
        if not smc_ctx or "smc_score" not in smc_ctx:
            return {
                "fusion_allowed": True,   # don't block while waiting for SMC
                "fusion_score":   None,
                "fusion_grade":   "PENDING",
                "issues":         ["SMC context not populated — fusion deferred"],
                "reason":         "SMC analysis not yet available",
            }

        reqs      = SMC_REQUIREMENTS.get(session, SMC_REQUIREMENTS["BETWEEN_SESSIONS"])
        smc_score = smc_ctx.get("smc_score", 0)
        has_bos   = smc_ctx.get("smc_factors", {}).get("bos", False)

        issues = []
        if smc_score < reqs["min_smc_score"]:
            issues.append(
                f"SMC score {smc_score} < required {reqs['min_smc_score']} for {session}"
            )
        if reqs["require_bos"] and not has_bos:
            issues.append(f"BOS required for {session} but not detected")

        allowed      = not issues
        fusion_score = smc_score   # score IS the SMC score — no hidden math

        if fusion_score >= 80:
            grade = "A+"
        elif fusion_score >= 60:
            grade = "A"
        elif fusion_score >= reqs["min_smc_score"]:
            grade = "B"
        else:
            grade = "INVALID"

        return {
            "fusion_allowed": allowed,
            "fusion_score":   fusion_score,
            "fusion_grade":   grade,
            "issues":         issues,
            "reason":         " | ".join(issues) if issues else f"All {session} requirements met",
        }

    # ═══════════════════════════════════════════════════════════
    # MAIN ANALYZE METHOD
    # ═══════════════════════════════════════════════════════════

    def analyze(
        self,
        pair:           str  = "EURUSD",
        smc_ctx:        dict = None,
        signal:         str  = "NO TRADE",
        signal_conf:    int  = 0,
        dt:             datetime = None,
        compute_fusion: bool = True,
    ) -> dict:
        """
        Full session analysis pipeline।

        Returns complete session intelligence package.

        Parameters:
            compute_fusion: If True, compute session+SMC fusion score.
                If False, skip fusion (returns fusion_score=None).
                Use False for the initial early-session call when SMC
                context is not yet available — avoids misleading
                "Fusion: ❌ (0/100)" log lines.
        """
        smc_ctx = smc_ctx or {}

        # Step 1: Detect session
        session_info = self.get_current_session(dt)
        session      = session_info["primary_session"]
        gmt_hour     = session_info["gmt_hour"]

        # Step 2: Session behavior
        behavior = self.analyze_session_behavior(session)

        # Step 3: Strategy mode
        strategy = self.get_strategy_mode(session, gmt_hour)

        # Step 4: Pair preference
        pair_pref = self.get_pair_preference(pair, session)

        # Step 5: Transition detection
        transition = self.detect_session_transition(gmt_hour)

        # Step 6: Session confidence
        session_conf = self.calculate_session_confidence(session, pair, smc_ctx, signal_conf)

        # Step 7: SMC fusion
        # Skip fusion computation when smc_ctx is empty (early pipeline call
        # before SMC engine has run). The real fusion is computed in the
        # second analyze() call after SMC context is available.
        if compute_fusion and smc_ctx:
            fusion = self.session_smc_fusion(session, smc_ctx, signal)
        else:
            # Pending — will be computed when SMC context is available
            fusion = {
                "fusion_allowed": True,   # don't block on pending
                "fusion_score":   None,   # None = not yet computed
                "fusion_grade":   "PENDING",
                "issues":         ["SMC context not yet available — fusion deferred"],
                "reason":         "Fusion pending SMC analysis",
            }

        # ── Final trade gate ──────────────────────────────────
        # Day 37+ fix: this intentionally does NOT hard-block on general
        # session score dropping below (min_confidence - 20) — that decision
        # is delegated to the downstream Risk Engine + TradePermission,
        # which can weigh session_score/fusion_score alongside other risk
        # factors this module doesn't see. That part is unchanged here.
        #
        # What IS restored (institutional review, Finding C-2): the
        # DEAD_ZONE veto below only works if a session can actually be
        # classified DEAD_ZONE, which requires DEAD_ZONES (session_rules.py)
        # to be non-empty and DEAD_ZONES_ENABLED to be True. Previously
        # DEAD_ZONES was empty, so `session != "DEAD_ZONE"` was always True
        # and this gate was a permanent no-op — the lowest-liquidity hours
        # (Sydney open / early Tokyo) were no longer blocked at all.
        # Round-10 audit fix: include fusion_allowed in trade_allowed.
        # Previously: `trade_allowed = strategy["trade_allowed"] and session != "DEAD_ZONE"`
        # — completely ignored fusion["fusion_allowed"]. The operator's
        # audit caught this: "Fusion Allowed ❌ (SMC score 55 < 65)"
        # would show in the log, then immediately below "✅ TRADE ALLOWED".
        #
        # Now: trade_allowed ALSO requires fusion["fusion_allowed"].
        # This makes the session_analyzer's own banner consistent with
        # the downstream TradePermission gate (which was already fixed
        # in Round-5 + Round-10).
        fusion_allowed = fusion.get("fusion_allowed", True)
        fusion_score = fusion.get("fusion_score", 0)
        fusion_grade = fusion.get("fusion_grade", "")

        # Session gate: strategy + dead zone only. Fusion is enforced separately
        # by decision_agent and TradePermission — bundling fusion here caused
        # misleading "Session gate: LONDON_NY_OVERLAP (strategy=A_PLUS_ONLY)" blocks.
        session_trade_allowed = (
            strategy["trade_allowed"]
            and session != "DEAD_ZONE"
        )

        # LONDON_NY_OVERLAP / A_PLUS_ONLY: honour A and A+ graded setups even when
        # a minor SMC checklist item fails (e.g. score 54 vs 55 threshold).
        if session == "LONDON_NY_OVERLAP" and fusion_grade in ("A", "A+"):
            fusion_allowed = True
            if not fusion.get("fusion_allowed", True):
                fusion = {
                    **fusion,
                    "fusion_allowed": True,
                    "reason": f"A/A+ grade ({fusion_grade}) — overlap fusion gate passed",
                }

        # Banner/log trade flag = session only; fusion blocks downstream.
        trade_allowed = session_trade_allowed

        # Log line: show "⏳ pending" when fusion is deferred, not "❌ (0/100)"
        if fusion_score is None:
            fusion_str = "⏳ pending"
        else:
            fusion_str = f"{'✅' if fusion_allowed else '❌'} ({fusion_score}/100)"

        log.info(
            f"[SessionAnalyzer] {session} | {pair} | "
            f"Strategy: {strategy['strategy']} | "
            f"Score: {session_conf['session_score']} | "
            f"Fusion: {fusion_str} | "
            f"Trade: {'✅' if trade_allowed else '❌'}"
        )

        return {
            "session_info":    session_info,
            "session":         session,
            "behavior":        behavior,
            "strategy":        strategy,
            "pair_preference": pair_pref,
            "transition":      transition,
            "session_conf":    session_conf,
            "fusion":              fusion,
            "session_trade_allowed": session_trade_allowed,
            "trade_allowed":       trade_allowed,
            "pair":                pair,
            "gmt_time":        session_info["gmt_time"],
        }

    # ═══════════════════════════════════════════════════════════
    # AI CONTEXT (AnalysisAgent / MasterAnalyst handoff)
    # ═══════════════════════════════════════════════════════════

    def get_ai_context(self, result: dict) -> dict:
        """
        MasterAnalyst + DecisionAgent-এ inject করার জন্য।
        """
        # fusion_score may be None when compute_fusion=False (early pipeline call).
        # Convert to 0 for downstream consumers that do numeric comparisons
        # (e.g. confluence_engine.py:269 `fusion_score >= 60`).
        _fusion_score = result["fusion"]["fusion_score"]
        if _fusion_score is None:
            _fusion_score = 0

        return {
            "current_session":        result["session"],
            "session_volatility":     result["behavior"]["volatility"],
            "session_behavior":       result["behavior"]["behavior"],
            "session_strategy":       result["strategy"]["strategy"],
            "session_trade_allowed":  result.get(
                "session_trade_allowed", result["trade_allowed"]
            ),
            "session_min_confidence": result["strategy"]["min_confidence"],
            "session_risk_mult":      result["strategy"]["risk_multiplier"],
            "pair_session_priority":  result["pair_preference"]["priority"],
            "pair_session_label":     result["pair_preference"]["label"],
            "is_overlap":             result["session_info"]["is_overlap"],
            "is_dead_zone":           result["session_info"]["is_dead_zone"],
            "london_open_window":     result["session_info"]["london_open_window"],
            "in_session_transition":  result["transition"]["in_transition"],
            "transition_type":        result["transition"].get("transition_type"),
            "transition_alert":       result["transition"].get("alert"),
            "session_score":          result["session_conf"]["session_score"],
            "session_grade":          result["session_conf"]["session_grade"],
            "fusion_allowed":         result["fusion"]["fusion_allowed"],
            "fusion_score":           _fusion_score,
            "fusion_grade":           result["fusion"]["fusion_grade"],
            "preferred_pairs":        result["pair_preference"]["preferred_pairs"],
            "gmt_time":               result["gmt_time"],
        }

    # ═══════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════════

    def print_summary(self, result: dict) -> None:
        bar = "═" * 58
        session   = result["session"]
        trade_ok  = result["trade_allowed"]
        strat     = result["strategy"]
        sess_conf = result["session_conf"]
        fusion    = result["fusion"]
        trans     = result["transition"]
        pair_pref = result["pair_preference"]

        icons = {
            "LONDON_NY_OVERLAP": "🔥",
            "LONDON":            "🇬🇧",
            "NEW_YORK":          "🗽",
            "TOKYO":             "🗼",
            "SYDNEY":            "🦘",
            "DEAD_ZONE":         "💤",
            "BETWEEN_SESSIONS":  "⏳",
        }
        icon = icons.get(session, "🌐")

        print(f"\n{bar}")
        print(f"  {icon}  SESSION INTELLIGENCE  (Day 63)")
        print(bar)
        print(f"  Time            :  {result['gmt_time']}")
        print(f"  Session         :  {session}")
        print(f"  Pair            :  {result['pair']}")
        print(f"  Volatility      :  {result['behavior']['volatility']}")
        print(f"  Behavior        :  {result['behavior']['behavior']}")
        print()

        print(f"  ── Strategy Mode ──")
        print(f"  Strategy        :  {strat['strategy']}")
        print(f"  Action          :  {strat['action'][:55]}")
        print(f"  Avoid           :  {strat['avoid'][:55]}")
        print(f"  Min Confidence  :  {strat['min_confidence']}%")
        print(f"  Risk Multiplier :  {strat['risk_multiplier']}x")
        if strat.get("alert"):
            print(f"  ⚠️  {strat['alert']}")
        print()

        print(f"  ── Pair Preference ──")
        print(f"  Priority        :  {pair_pref['priority']}/100  [{pair_pref['label']}]")
        print(f"  Preferred Pairs :  {', '.join(pair_pref['preferred_pairs'][:5])}")
        print()

        if trans["in_transition"]:
            print(f"  ── 🔔 Session Transition Alert ──")
            print(f"  Type            :  {trans['transition_type']}")
            print(f"  Alert           :  {trans['alert']}")
            print(f"  Action          :  {trans['action']}")
            print()

        print(f"  ── Session Score ──")
        print(f"  Score           :  {sess_conf['session_score']}/100  [{sess_conf['session_grade']}]")
        for reason in sess_conf["score_reasons"]:
            print(f"    {reason}")
        print()

        print(f"  ── SMC + Session Fusion ──")
        print(f"  Fusion Allowed  :  {'✅' if fusion['fusion_allowed'] else '❌'}")
        _fs = fusion['fusion_score']
        if _fs is None:
            print(f"  Fusion Score    :  ⏳ pending (SMC not yet available)  [{fusion['fusion_grade']}]")
        else:
            print(f"  Fusion Score    :  {_fs}/100  [{fusion['fusion_grade']}]")
        if fusion["issues"]:
            for issue in fusion["issues"]:
                print(f"  ⚠️  {issue}")
        print()

        trade_icon = "✅ TRADE ALLOWED" if trade_ok else "⛔ NO TRADE"
        print(f"  ┌──────────────────────────────────────────────────┐")
        print(f"  │  {trade_icon:<51}│")
        print(f"  │  {result['behavior']['description'][:51]:<51}│")
        print(f"  └──────────────────────────────────────────────────┘")
        print(bar + "\n")

    # ═══════════════════════════════════════════════════════════
    # DST HELPERS
    # ═══════════════════════════════════════════════════════════

    def _get_dst_flags(self, dt: datetime) -> tuple:
        """Round-17: get accurate DST flags using zoneinfo.

        Uses the OS timezone database (via zoneinfo, Python 3.9+ stdlib)
        to determine whether US Eastern (America/New_York) and UK
        (Europe/London) are currently observing DST at the given UTC
        datetime. This is 100% accurate — no fixed-date approximation.

        Falls back to the old _is_us_dst()/_is_eu_dst() heuristics if
        zoneinfo is unavailable (e.g. Python < 3.9 or missing tzdata
        package on some Linux distros).

        Returns:
            tuple: (us_dst: bool, eu_dst: bool)
        """
        try:
            from zoneinfo import ZoneInfo
            # Convert UTC datetime to local timezones
            # If dt is naive, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            ny_time = dt.astimezone(ZoneInfo("America/New_York"))
            ld_time = dt.astimezone(ZoneInfo("Europe/London"))

            # DST is active if the UTC offset includes the summer adjustment.
            # America/New_York: UTC-5 (EST) in winter, UTC-4 (EDT) in summer
            # Europe/London: UTC+0 (GMT) in winter, UTC+1 (BST) in summer
            us_dst = ny_time.dst() is not None and ny_time.dst().total_seconds() > 0
            eu_dst = ld_time.dst() is not None and ld_time.dst().total_seconds() > 0

            return us_dst, eu_dst
        except Exception:
            # Fallback to old fixed-date approximation
            return self._is_us_dst(dt), self._is_eu_dst(dt)

    def _is_us_dst(self, dt: datetime) -> bool:
        """
        US DST: 2nd Sunday in March → 1st Sunday in November (approx).
        """
        month = dt.month
        day   = dt.day
        if month < 3 or month > 11:
            return False
        if month > 3 and month < 11:
            return True
        if month == 3:
            # 2nd Sunday in March ≈ day 8-14
            return day >= 8
        if month == 11:
            return day < 7

    def _is_eu_dst(self, dt: datetime) -> bool:
        """
        EU DST: Last Sunday in March → Last Sunday in October (approx).
        """
        month = dt.month
        day   = dt.day
        if month < 3 or month > 10:
            return False
        if month > 3 and month < 10:
            return True
        if month == 3:
            return day >= 25
        if month == 10:
            return day < 25

    def _is_dead_zone(self, gmt_hour: int) -> bool:
        # DEAD_ZONES is imported at module level (see top of file) — no
        # need to re-import here. The previous local import shadowed the
        # module-level binding and confused linters.
        #
        # RESTORED (institutional review, Finding C-2): this check used to
        # be a no-op because DEAD_ZONES was emptied upstream in
        # session_rules.py, meaning trade_allowed could never be False due
        # to session/liquidity risk — the bot would trade in any session,
        # at any hour, with the dead-zone gate no longer able to fire.
        # DEAD_ZONES_ENABLED is a single, explicit switch for anyone who
        # deliberately needs to disable this (e.g. run-window conflicts) —
        # see session_rules.py for the full rationale.
        if not DEAD_ZONES_ENABLED:
            return False
        for zone in DEAD_ZONES:
            if zone["start"] <= gmt_hour < zone["end"]:
                return True
        return False

    # ═══════════════════════════════════════════════════════════
    # SESSION PERFORMANCE MEMORY (hook for Day 64+)
    # ═══════════════════════════════════════════════════════════

    def record_trade_outcome(
        self,
        pair:     str,
        session:  str,
        strategy: str,
        outcome:  str,    # "WIN" | "LOSS" | "BE"
        pnl_pips: float = 0,
    ) -> None:
        """
        session_performance memory-তে trade result save করো।
        Future: database-এ persist করা হবে।

        Example DB table:
            session_performance(id, pair, session, strategy, wins, losses, profit_factor, date)
        """
        key = f"{pair}_{session}_{strategy}"
        if key not in self._session_performance:
            self._session_performance[key] = {"wins": 0, "losses": 0, "pnl": 0}

        entry = self._session_performance[key]
        if outcome == "WIN":
            entry["wins"] += 1
        elif outcome == "LOSS":
            entry["losses"] += 1
        entry["pnl"] += pnl_pips

        log.info(
            f"[SessionMemory] {pair} {session} {strategy} | "
            f"{outcome} {pnl_pips:+.1f}p | "
            f"W:{entry['wins']} L:{entry['losses']}"
        )

    def get_session_performance(self, pair: str, session: str) -> dict:
        """Pair + Session combination-এর historical performance।"""
        results = {}
        for key, data in self._session_performance.items():
            if key.startswith(f"{pair}_{session}"):
                total = data["wins"] + data["losses"]
                wr    = round(data["wins"] / total * 100, 1) if total > 0 else 0
                results[key] = {**data, "total": total, "win_rate": wr}
        return results


# ═══════════════════════════════════════════════════════════════
# QUICK RUN — Direct test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    analyzer = SessionAnalyzer()

    # Test with current time
    result = analyzer.analyze(
        pair        = "EURUSD",
        smc_ctx     = {
            "smc_score": 72,
            "smc_factors": {
                "liquidity_sweep": True,
                "order_block":     True,
                "fvg":             False,
                "bos":             True,
                "confirmation_candle": False,
            },
        },
        signal      = "BUY",
        signal_conf = 75,
    )
    analyzer.print_summary(result)

    ctx = analyzer.get_ai_context(result)
    print("AI Context:")
    for k, v in ctx.items():
        print(f"  {k:<30}: {v}")