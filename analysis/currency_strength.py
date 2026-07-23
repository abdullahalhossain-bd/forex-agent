# analysis/currency_strength.py  —  Day 64 | Currency Strength Matrix Engine
# ============================================================
# AI-কে শেখায়: "কোন currency আসলে শক্তিশালী, আর কোন currency দুর্বল?"
#
# Day 61-63 পর্যন্ত AI শিখেছে market structure, SMC, liquidity, session
# intelligence — কিন্তু সবকিছুই single-pair view (যেমন শুধু EURUSD)।
# Day 64-এ AI পুরো forex market scan করে relative currency strength বের
# করবে, এবং strongest-vs-weakest currency দিয়ে best pair বাছবে।
#
# Pipeline:
#   1. সব major cross pair fetch + indicator calculate
#   2. প্রতিটা pair থেকে base/quote currency-তে strength contribution
#   3. Normalize (0-100), momentum/acceleration track করো (history থেকে)
#   4. Rank করো, strong-vs-weak pair বের করো, correlation filter করো
#   5. Heatmap + cycle detection + (ঐচ্ছিক) multi-timeframe confluence
#
# Reference: Day 64 — Currency Strength Matrix Engine (Relative Currency
# Intelligence) doc।
# ============================================================

import os
import time
import json
from datetime import datetime

from data.fetcher import get_data_fetcher
from data.indicators import Indicators
from analysis.strength_calculator import StrengthCalculator
from analysis.currency_ranker import CurrencyRanker
from utils.logger import get_logger
from core.constants import MEMORY_DIR

log = get_logger("currency_strength")

# 2026-07-22 wiring fix: calculate_strength() fetches all 28 CROSS_PAIRS
# every call. Before this engine was wired into the live pipeline that
# cost was never paid (module was dead). Now that AnalysisAgent calls
# analyze() once per cycle, an unconditional 28-pair fetch every cycle
# would multiply MT5/broker call volume ~28x versus the single-pair
# analysis the rest of the pipeline does — same class of problem as the
# already-known Groq dual-call rate-limit issue. Cache the strength
# matrix across cycles; a currency's relative strength doesn't need
# recomputing every 15m-bar-close anyway.
_STRENGTH_CACHE_TTL_SECONDS = 300  # 5 min, matches sentiment_data.py's convention

# ── Tracked currencies ──────────────────────────────────────────
MAJOR_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

# ── Cross pairs — প্রতিটা currency ঠিক ৭টা pair-এ আছে (full coverage,
#    C(8,2) = 28 unique pairs, যেমনটা doc-এর "USD: 7 pairs" উদাহরণে আছে) ──
CROSS_PAIRS = [
    ("EURUSD", "EUR", "USD"), ("GBPUSD", "GBP", "USD"),
    ("AUDUSD", "AUD", "USD"), ("NZDUSD", "NZD", "USD"),
    ("USDCAD", "USD", "CAD"), ("USDCHF", "USD", "CHF"),
    ("USDJPY", "USD", "JPY"),
    ("EURGBP", "EUR", "GBP"), ("EURJPY", "EUR", "JPY"),
    ("EURAUD", "EUR", "AUD"), ("EURCAD", "EUR", "CAD"),
    ("EURCHF", "EUR", "CHF"), ("EURNZD", "EUR", "NZD"),
    ("GBPJPY", "GBP", "JPY"), ("GBPCAD", "GBP", "CAD"),
    ("GBPCHF", "GBP", "CHF"), ("GBPAUD", "GBP", "AUD"),
    ("GBPNZD", "GBP", "NZD"),
    ("AUDJPY", "AUD", "JPY"), ("AUDCAD", "AUD", "CAD"),
    ("AUDNZD", "AUD", "NZD"), ("AUDCHF", "AUD", "CHF"),
    ("NZDJPY", "NZD", "JPY"), ("NZDCAD", "NZD", "CAD"),
    ("NZDCHF", "NZD", "CHF"),
    ("CADJPY", "CAD", "JPY"), ("CADCHF", "CAD", "CHF"),
    ("CHFJPY", "CHF", "JPY"),
]

# ── Memory (currency_strength_history table-এর JSON-ভিত্তিক সংস্করণ) ──
HISTORY_PATH         = str(MEMORY_DIR / "currency_strength_history.json")
TRADE_HISTORY_PATH   = str(MEMORY_DIR / "currency_strength_trades.json")
HISTORY_MAX_ENTRIES  = 2000
MOMENTUM_LOOKBACK    = 5
MOMENTUM_THRESHOLD   = 1.5   # এর কম change হলে FLAT ধরা হবে


class CurrencyStrengthEngine:
    """
    Day 64 — Global Currency Intelligence Engine।

    Usage:
        engine = CurrencyStrengthEngine(timeframe="1h")
        result = engine.analyze(min_diff=40)
        engine.print_summary(result)
        ctx = engine.get_ai_context(result)   # MasterAnalyst-এ pass করো
    """

    def __init__(self, timeframe: str = "1h", candle_limit: int = 100):
        self.timeframe    = timeframe
        self.candle_limit = candle_limit
        self.fetcher      = get_data_fetcher()
        self.ind          = Indicators()
        self.calculator   = StrengthCalculator()
        self.ranker       = CurrencyRanker()
        self._strength_cache      = None
        self._strength_cache_time = 0.0

    # ═══════════════════════════════════════════════════════
    # STEP 1: CURRENCY STRENGTH CALCULATION
    # ═══════════════════════════════════════════════════════

    def calculate_strength(self) -> dict:
        """
        সব CROSS_PAIRS fetch করে প্রতিটা currency-র raw contribution
        accumulate করো, তারপর normalize (0-100) করো।

        Cached for _STRENGTH_CACHE_TTL_SECONDS — see module docstring
        for why (28-pair fetch is too heavy to repeat every cycle).
        """
        now = time.time()
        if self._strength_cache and (now - self._strength_cache_time) < _STRENGTH_CACHE_TTL_SECONDS:
            log.debug("[CurrencyStrength] Using cached strength matrix")
            return self._strength_cache

        raw_scores     = {c: 0.0 for c in MAJOR_CURRENCIES}
        counts         = {c: 0   for c in MAJOR_CURRENCIES}
        pair_details   = {}
        fetch_failures = []

        for symbol, base, quote in CROSS_PAIRS:
            df = self.fetcher.fetch_ohlcv(symbol, self.timeframe, limit=self.candle_limit)
            if df is None or df.empty:
                fetch_failures.append(symbol)
                continue

            df      = self.ind.add_all(df)
            ind_ctx = self.ind.get_ai_context(df)

            pair_score = self.calculator.compute_pair_score(df, ind_ctx)
            total      = pair_score["total"]

            raw_scores[base]  += total
            raw_scores[quote] -= total
            counts[base]  += 1
            counts[quote] += 1
            pair_details[symbol] = pair_score

        if fetch_failures:
            log.warning(f"[CurrencyStrength] Could not fetch: {fetch_failures}")

        avg_scores = {
            c: round(raw_scores[c] / counts[c], 3) if counts[c] > 0 else 0.0
            for c in MAJOR_CURRENCIES
        }
        normalized = self.calculator.normalize_scores(avg_scores)

        log.info(f"[CurrencyStrength] Normalized strengths: {normalized}")

        result = {
            "strengths":    normalized,
            "raw_scores":   avg_scores,
            "pair_details": pair_details,
            "timeframe":    self.timeframe,
            "pairs_used":   len(pair_details),
            "pairs_failed": fetch_failures,
            "timestamp":    datetime.utcnow().isoformat(),
        }
        self._strength_cache      = result
        self._strength_cache_time = now
        return result

    # ═══════════════════════════════════════════════════════
    # STEP 2: MOMENTUM DETECTION  ⭐⭐⭐⭐⭐
    # ═══════════════════════════════════════════════════════

    def calculate_momentum(self, strengths: dict) -> dict:
        """
        শুধু "এখন কত শক্তিশালী" না — history-এর সাথে তুলনা করে
        "শক্তি বাড়ছে নাকি কমছে" বলো (UP / DOWN / FLAT) + acceleration।
        """
        history  = self._load_history()
        momentum = {}

        for cur, score in strengths.items():
            past = [
                h["strength_score"] for h in history
                if h["currency"] == cur
            ][-MOMENTUM_LOOKBACK:]

            delta = round(score - past[-1], 2) if past else 0.0

            if len(past) >= 2:
                prev_delta   = past[-1] - past[-2]
                acceleration = round(delta - prev_delta, 2)
            else:
                acceleration = 0.0

            if delta > MOMENTUM_THRESHOLD:
                direction = "UP"
            elif delta < -MOMENTUM_THRESHOLD:
                direction = "DOWN"
            else:
                direction = "FLAT"

            momentum[cur] = {
                "score":          score,
                "delta":          delta,
                "acceleration":   acceleration,
                "momentum":       direction,
                "accelerating":   abs(acceleration) >= 2.0,
                "history_points": len(past),
            }

        return momentum

    # ═══════════════════════════════════════════════════════
    # STEP 3: RANKING  (delegates to CurrencyRanker)
    # ═══════════════════════════════════════════════════════

    def rank_currencies(self, strengths: dict) -> dict:
        return self.ranker.rank(strengths)

    # ═══════════════════════════════════════════════════════
    # STEP 4: STRONG vs WEAK PAIR FINDER
    # ═══════════════════════════════════════════════════════

    def find_best_pairs(self, strengths: dict, min_diff: int = 40) -> list[dict]:
        opportunities = self.ranker.find_best_pairs(strengths, min_diff=min_diff)
        return self.ranker.detect_correlation_risk(opportunities)

    # ═══════════════════════════════════════════════════════
    # ⭐ STRENGTH + SMC CONFIRMATION  (extra feature #3)
    # ═══════════════════════════════════════════════════════

    def evaluate_setup(
        self,
        opportunity: dict,
        smc_ctx:     dict = None,
        session_ctx: dict = None,
    ) -> dict:
        """
        Currency-strength opportunity + SMC confluence + session
        intelligence একসাথে মিলিয়ে "A+ setup" কিনা বলো।

        Example (doc):
            GBP strongest + JPY weakest + GBPJPY bullish BOS +
            liquidity sweep + London session = A+ setup
        """
        smc_ctx     = smc_ctx or {}
        session_ctx = session_ctx or {}

        reasons = [
            f"{opportunity['strong_currency']} strongest currency "
            f"({opportunity['strong_strength']})",
            f"{opportunity['weak_currency']} weakest currency "
            f"({opportunity['weak_strength']})",
        ]
        score = 40 if opportunity.get("trade_quality") in ("HIGH", "VERY_HIGH") else 20

        bos_event = str(smc_ctx.get("smc_h4_bos") or smc_ctx.get("smc_bos") or "")
        if smc_ctx.get("smc_factors", {}).get("bos") or "BOS" in bos_event:
            reasons.append("BOS (Break of Structure) confirmed")
            score += 25

        if smc_ctx.get("smc_factors", {}).get("liquidity_sweep"):
            reasons.append("Liquidity sweep detected")
            score += 20

        if session_ctx.get("current_session") in ("LONDON", "NEW_YORK", "LONDON_NY_OVERLAP"):
            reasons.append(f"{session_ctx.get('current_session')} session — high liquidity")
            score += 15

        if opportunity.get("correlated"):
            score -= 10
            reasons.append("⚠️ Correlated with another active opportunity — reduce size")

        score = max(0, min(100, score))
        grade = "A+" if score >= 85 else ("A" if score >= 65 else ("B" if score >= 45 else "C"))

        return {
            "pair":        opportunity["pair"],
            "direction":   opportunity["direction"],
            "setup_score": score,
            "setup_grade": grade,
            "is_a_plus":   grade == "A+",
            "reasons":     reasons,
        }

    # ═══════════════════════════════════════════════════════
    # ⭐ MULTI-TIMEFRAME STRENGTH  (doc #13)
    # ═══════════════════════════════════════════════════════

    def multi_timeframe_strength(self, timeframes: tuple = ("15m", "1h", "1d")) -> dict:
        """
        একই currency একাধিক timeframe-এ strong/weak কিনা চেক করো —
        যত বেশি timeframe align করে, confidence তত বাড়ে।

        Note: প্রতিটা TF-এর জন্য পুরো CROSS_PAIRS আবার fetch হয়, তাই এটা
        ব্যয়বহুল (heavy)। Scanner-এ প্রতি cycle-এ না চালিয়ে, শুধু
        high-quality opportunity confirm করার আগে call করাই ভালো।
        """
        tf_strengths = {}
        for tf in timeframes:
            engine_tf = CurrencyStrengthEngine(timeframe=tf, candle_limit=self.candle_limit)
            tf_result = engine_tf.calculate_strength()
            tf_strengths[tf] = tf_result["strengths"]

        combined = {}
        for cur in MAJOR_CURRENCIES:
            scores = [tf_strengths[tf].get(cur, 50.0) for tf in timeframes]
            avg    = round(sum(scores) / len(scores), 1)

            directions = ["STRONG" if s >= 60 else ("WEAK" if s <= 40 else "NEUTRAL") for s in scores]
            aligned    = directions.count(directions[0]) == len(directions) and directions[0] != "NEUTRAL"

            combined[cur] = {
                "avg_strength":     avg,
                "per_tf":           dict(zip(timeframes, scores)),
                "aligned":          aligned,
                "confidence_boost": 15 if aligned else 0,
            }

        log.info(
            f"[CurrencyStrength] MTF combined: "
            f"{ {k: v['avg_strength'] for k, v in combined.items()} }"
        )
        return {"timeframes": timeframes, "per_currency": combined}

    # ═══════════════════════════════════════════════════════
    # FULL PIPELINE
    # ═══════════════════════════════════════════════════════

    def analyze(self, min_diff: int = 40, save_history: bool = True) -> dict:
        """
        Day 64-এর পুরো pipeline একসাথে:
        calculate_strength -> momentum -> rank -> best pairs ->
        correlation filter -> heatmap -> cycle detection -> memory save
        """
        strength_result = self.calculate_strength()
        strengths       = strength_result["strengths"]

        momentum      = self.calculate_momentum(strengths)
        ranking       = self.rank_currencies(strengths)
        opportunities = self.find_best_pairs(strengths, min_diff=min_diff)
        heatmap       = self.ranker.build_heatmap(strengths)

        if save_history:
            self._save_history(strengths, momentum)

        cycle = self.ranker.detect_cycle(self._load_history())

        result = {
            "strengths":     strengths,
            "raw_scores":    strength_result["raw_scores"],
            "momentum":      momentum,
            "ranking":       ranking,
            "opportunities": opportunities,
            "heatmap":       heatmap,
            "cycle":         cycle,
            "pairs_used":    strength_result["pairs_used"],
            "pairs_failed":  strength_result["pairs_failed"],
            "timeframe":     self.timeframe,
            "timestamp":     strength_result["timestamp"],
        }

        log.info(
            f"[CurrencyStrength] Strongest: {ranking['strongest']} | "
            f"Weakest: {ranking['weakest']} | "
            f"Opportunities (diff>={min_diff}): {len(opportunities)}"
        )
        return result

    # ═══════════════════════════════════════════════════════
    # MEMORY — currency_strength_history (doc #12)
    # ═══════════════════════════════════════════════════════

    def _load_history(self) -> list:
        if not os.path.exists(HISTORY_PATH):
            return []  # normal: no history yet, not an error
        try:
            with open(HISTORY_PATH) as f:
                return json.load(f)
        except Exception as e:
            # FIX (institutional review, item #4): this used to log a
            # meaningless "Suppressed exception at line 348" and silently
            # return []. A file that EXISTS but fails to parse (corrupted
            # JSON, partial write, permissions issue) is a real data
            # problem — silently resetting to an empty history here means
            # momentum/cycle detection loses its memory without anyone
            # being told why. Log it clearly instead of masking it.
            log.warning(
                f"[CurrencyStrength] History file at {HISTORY_PATH} exists but "
                f"failed to load ({type(e).__name__}: {e}) — falling back to "
                f"empty history. Momentum/cycle detection will restart from "
                f"scratch until this is investigated."
            )
            return []

    def _save_history(self, strengths: dict, momentum: dict) -> None:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        history = self._load_history()
        ts      = datetime.utcnow().isoformat()

        for cur, score in strengths.items():
            history.append({
                "id":             len(history) + 1,
                "currency":       cur,
                "strength_score": score,
                "momentum":       momentum.get(cur, {}).get("momentum", "FLAT"),
                "timeframe":      self.timeframe,
                "timestamp":      ts,
            })

        with open(HISTORY_PATH, "w") as f:
            json.dump(history[-HISTORY_MAX_ENTRIES:], f, indent=2)

        log.info(f"[CurrencyStrength] Saved {len(strengths)} snapshot(s) to history")

    def record_trade_outcome(self, pair: str, outcome: str, pnl_pips: float = 0) -> None:
        """
        Currency-strength-based trade-র result track করো — future-এ
        learning loop-এ feed করা যাবে (কোন strong/weak combo সবচেয়ে
        reliable সেটা শিখতে)। session_analyzer.record_trade_outcome()-এর
        মতোই হালকা memory hook।
        """
        os.makedirs(os.path.dirname(TRADE_HISTORY_PATH), exist_ok=True)
        trades = []
        if os.path.exists(TRADE_HISTORY_PATH):
            try:
                with open(TRADE_HISTORY_PATH) as f:
                    trades = json.load(f)
            except Exception as e:
                # FIX (institutional review, item #4): this previously
                # swallowed the exception with no log line at all. Same
                # reasoning as _load_history above — a corrupted trade
                # history file failing silently means outcome tracking
                # quietly resets without anyone knowing.
                log.warning(
                    f"[CurrencyStrength] Trade history file at {TRADE_HISTORY_PATH} "
                    f"exists but failed to load ({type(e).__name__}: {e}) — "
                    f"falling back to empty trade history."
                )
                trades = []

        trades.append({
            "id":        len(trades) + 1,
            "pair":      pair,
            "outcome":   outcome,
            "pnl_pips":  pnl_pips,
            "timestamp": datetime.utcnow().isoformat(),
        })

        with open(TRADE_HISTORY_PATH, "w") as f:
            json.dump(trades[-1000:], f, indent=2)

        log.info(f"[CurrencyStrength] Trade outcome recorded — {pair} {outcome} {pnl_pips:+.1f}p")

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT  (MasterAnalyst / DecisionAgent handoff)
    # ═══════════════════════════════════════════════════════

    def get_ai_context(self, result: dict) -> dict:
        ranking       = result.get("ranking", {})
        opportunities = result.get("opportunities", [])
        cycle         = result.get("cycle", {})
        best          = opportunities[0] if opportunities else None

        return {
            "currency_strengths":         result.get("strengths", {}),
            "currency_strongest":         ranking.get("strongest", []),
            "currency_weakest":           ranking.get("weakest", []),
            "currency_momentum":          result.get("momentum", {}),
            "currency_cycle":             cycle,
            "currency_best_pair":         best.get("pair") if best else None,
            "currency_best_direction":    best.get("direction") if best else None,
            "currency_best_diff":         best.get("strength_difference") if best else 0,
            "currency_best_quality":      best.get("trade_quality") if best else "NONE",
            "currency_trade_recommended": best is not None,
            "currency_opportunities":     opportunities,
            "currency_pairs_used":        result.get("pairs_used", 0),
        }

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY  (Real-Time Currency Map — doc #9)
    # ═══════════════════════════════════════════════════════

    def print_summary(self, result: dict) -> None:
        bar = "═" * 58
        log.info(bar)
        log.info("  🌍  CURRENCY STRENGTH MATRIX ENGINE  (Day 64)")
        log.info(bar)
        log.info(f"  Timeframe   : {result.get('timeframe')}")
        log.info(
            f"  Pairs used  : {result.get('pairs_used')} "
            f"(failed: {len(result.get('pairs_failed', []))})"
        )
        log.info("")

        ranking   = result.get("ranking", {})
        momentum  = result.get("momentum", {})

        log.info("  ── Currency Strength Map ──")
        for cur, score in ranking.get("ranked", []):
            mom       = momentum.get(cur, {})
            arrow     = {"UP": "↑↑", "DOWN": "↓↓", "FLAT": "→"}.get(mom.get("momentum"), "→")
            bar_len   = int(score / 5)
            strength_bar = "█" * bar_len + "░" * (20 - bar_len)
            log.info(f"  {cur}  {strength_bar}  {score:>5.1f}  {arrow}")

        log.info("")
        log.info(f"  Strongest   : {', '.join(ranking.get('strongest', []))}")
        log.info(f"  Weakest     : {', '.join(ranking.get('weakest', []))}")

        log.info("")
        log.info("  ── 🔥 Best Opportunities ──")
        opportunities = result.get("opportunities", [])
        if not opportunities:
            log.info("  None — no pair meets the strength-difference threshold")
        for i, opp in enumerate(opportunities[:5], 1):
            corr = "  ⚠️ correlated" if opp.get("correlated") else ""
            log.info(
                f"  {i}. {opp['pair']:<7} {opp['direction']:<4} "
                f"diff={opp['strength_difference']:<5} [{opp['trade_quality']}]{corr}"
            )

        cycle = result.get("cycle", {})
        if cycle:
            log.info("")
            log.info("  ── Currency Cycles ──")
            for cur, info in cycle.items():
                log.info(f"  {cur}: {info.get('cycle')} (slope {info.get('slope'):+.2f})")

        log.info(bar)

    def print_heatmap(self, result: dict) -> None:
        self.ranker.print_heatmap(result.get("heatmap", {}))


# ═══════════════════════════════════════════════════════════════
# QUICK RUN — Direct test
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# SHARED SINGLETON — single source of truth
# ------------------------------------------------------------
# 2026-07-22 fix: sentiment_data.py used to run its own private
# yfinance-based recalculation of currency strength (12 sample pairs,
# 1-day % change, no indicators) alongside this engine's real
# calculation (28 pairs, RSI/momentum via StrengthCalculator, 1h
# candles). Same currency, two different numbers, neither aware of
# the other's cache — architecture-level duplication, not a bug you
# could patch locally. Any caller that wants currency strength
# (sentiment_data.py included) must now go through this singleton so
# there is exactly one calculation and one cache.
# ═══════════════════════════════════════════════════════════════
_engine_singleton: "CurrencyStrengthEngine | None" = None


def get_currency_strength_engine(timeframe: str = "1h", candle_limit: int = 100) -> "CurrencyStrengthEngine":
    """
    Process-wide singleton accessor. Returns the same CurrencyStrengthEngine
    instance (and therefore the same _strength_cache) on every call, so
    every consumer sees identical, consistently-cached strength numbers.

    First caller's timeframe/candle_limit wins for the lifetime of the
    process; later calls with different args are ignored (logged once)
    rather than silently spinning up a second, divergent engine.
    """
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = CurrencyStrengthEngine(timeframe=timeframe, candle_limit=candle_limit)
        log.info(
            f"[CurrencyStrength] Singleton engine created "
            f"(timeframe={timeframe}, candle_limit={candle_limit})"
        )
    elif timeframe != _engine_singleton.timeframe or candle_limit != _engine_singleton.candle_limit:
        log.warning(
            f"[CurrencyStrength] get_currency_strength_engine() called with "
            f"(timeframe={timeframe}, candle_limit={candle_limit}) but singleton "
            f"already exists with (timeframe={_engine_singleton.timeframe}, "
            f"candle_limit={_engine_singleton.candle_limit}) — ignoring new args, "
            f"reusing existing engine so cache/consistency isn't broken."
        )
    return _engine_singleton


if __name__ == "__main__":
    engine = CurrencyStrengthEngine(timeframe="1h")
    result = engine.analyze(min_diff=40)
    engine.print_summary(result)
    engine.print_heatmap(result)

    ctx = engine.get_ai_context(result)
    print("\nAI Context (for MasterAnalyst):")
    for k, v in ctx.items():
        print(f"  {k:<30}: {v}")