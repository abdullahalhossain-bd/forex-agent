# analysis/intermarket.py  —  Day 65 | Intermarket Analysis Engine ⭐
# ============================================================
# Global Market Intelligence — Forex-কে isolated market হিসেবে না
# দেখে, DXY + Gold + Oil + Bond Yield + S&P500 + VIX একসাথে দেখে
# macro regime বের করে এবং currency bias adjust করে।
#
#                     Global Markets
#                          ↓
#               Intermarket Engine ⭐
#      ┌─────────┬─────────┬─────────┬─────────┐
#     DXY      GOLD      YIELD     SP500     VIX
#      └─────────┴─────────┴─────────┴─────────┘
#                          ↓
#               Macro Environment
#                          ↓
#          Currency Strength Adjustment
#                          ↓
#               Trade Decision
#
# Sub-modules:
#   macro_data.py        -> raw global asset data (DXY/Gold/Oil/US10Y/SP500/VIX)
#   correlation_engine.py -> pair-vs-asset correlation matrix
#   risk_sentiment.py     -> Risk-On / Risk-Off + VIX fear classification
#
# Fix (Day 96 bug):
#   calculate_correlations() was passing a single string to
#   CorrelationEngine.build_matrix() which expects List[str].
#   Python would iterate the string chars ["E","U","R","U","S","D"]
#   producing an empty/garbage matrix and an empty result dict,
#   causing print_summary({}) to log "Pair: ?" on the first call.
#   Fixed by building a proper list of common pairs and wrapping
#   the matrix result in {"matrix": ..., "pair": pair}.
#
#   Also removed the stray self.corr_engine.print_summary() call
#   inside IntermarketEngine.print_summary() — that method expects
#   a full analyze() result dict, not the {"matrix":...} dict we
#   build here, so it always printed "Pair: ?" on the first cycle.
# ============================================================

import json
import os
from datetime import datetime, timezone

from analysis.macro_data import MacroDataProvider
from analysis.correlation_engine import CorrelationEngine
from analysis.risk_sentiment import RiskSentimentEngine
from utils.logger import get_logger
from core.constants import MEMORY_DIR

log = get_logger("intermarket_engine")

MACRO_MEMORY_PATH = str(MEMORY_DIR / "intermarket_history.json")

# Macro Score weights (total 100, doc অনুযায়ী)
MACRO_SCORE_WEIGHTS = {
    "dxy":   20,
    "yield": 20,
    "gold":  15,
    "vix":   20,
    "sp500": 15,
    "oil":   10,
}

MIN_FUSION_SCORE = 60   # Macro+SMC fusion-এ এর নিচে হলে confluence weak ধরা হবে

# Common pairs used when building the correlation matrix.
# The requested pair is always included; these provide context.
_MATRIX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "USDCAD", "USDCHF", "NZDUSD", "XAUUSD",
]


class IntermarketEngine:
    """
    Usage:
        engine = IntermarketEngine()
        result = engine.analyze("GBPUSD")
        engine.print_summary(result)
        ctx = engine.get_ai_context(result)        # MasterAnalyst-এ pass করো
        fusion = engine.fuse_with_smc(result, smc_ctx, session_ctx)
    """

    def __init__(self):
        self.macro_provider = MacroDataProvider()
        self.corr_engine    = CorrelationEngine()
        self.risk_engine    = RiskSentimentEngine()

    # ═══════════════════════════════════════════════════════════
    # MAIN METHOD
    # ═══════════════════════════════════════════════════════════

    def analyze(self, pair: str = "EURUSD", news_ctx: dict = None) -> dict:
        """
        Full intermarket pipeline:
          1. fetch_global_data
          2. calculate_correlations
          3. detect_market_regime   (risk-on/off)
          4. generate_macro_bias    (per-currency + per-pair bias)
          5. macro score
          6. cross-asset confirmation
          7. event risk integration
          8. historical macro memory save
        """
        macro_data    = self.fetch_global_data()
        corr_result   = self.calculate_correlations(pair)
        regime_result = self.detect_market_regime(macro_data)
        bias_result   = self.generate_macro_bias(pair, macro_data, regime_result)

        macro_score, score_breakdown = self._calculate_macro_score(macro_data, pair)
        confirmation = self._cross_asset_confirmation(macro_data, bias_result)
        event_risk   = self._event_risk_integration(news_ctx or {})

        result = {
            "pair":            pair,
            "macro_data":      macro_data,
            "correlations":    corr_result,
            "regime":          regime_result,
            "bias":            bias_result,
            "macro_score":     macro_score,
            "score_breakdown": score_breakdown,
            "confirmation":    confirmation,
            "event_risk":      event_risk,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }

        self._save_memory(result)

        log.info(
            f"[IntermarketEngine] {pair} | Regime: {regime_result['environment']} | "
            f"Macro Score: {macro_score}/100 | Pair Bias: {bias_result['pair_bias']}"
        )
        return result

    # ═══════════════════════════════════════════════════════════
    # STEP 1: FETCH GLOBAL DATA
    # ═══════════════════════════════════════════════════════════

    def fetch_global_data(self) -> dict:
        return self.macro_provider.get_all()

    # ═══════════════════════════════════════════════════════════
    # STEP 2: CORRELATIONS
    # FIX: build_matrix() expects List[str], not a single string.
    # Passing "EURUSD" directly caused Python to iterate over the
    # characters ["E","U","R","U","S","D"] and produce a garbage
    # matrix + empty result dict → "Pair: ?" in the first log line.
    # ═══════════════════════════════════════════════════════════

    def calculate_correlations(self, pair: str) -> dict:
        """Build a correlation matrix for the requested pair vs common pairs.

        Returns a dict with keys:
            "pair"   — the primary pair being analyzed
            "matrix" — nested dict {pair_a: {pair_b: float}}
        """
        clean_pair = pair.upper().replace("/", "").replace("=X", "")
        # Always include the requested pair; deduplicate with the common set.
        pairs = list(dict.fromkeys([clean_pair] + _MATRIX_PAIRS))
        matrix = self.corr_engine.build_matrix(pairs)
        return {"pair": clean_pair, "matrix": matrix}

    # ═══════════════════════════════════════════════════════════
    # STEP 3: MARKET REGIME  (Risk-On / Risk-Off)
    # ═══════════════════════════════════════════════════════════

    def detect_market_regime(self, macro_data: dict) -> dict:
        return self.risk_engine.analyze(macro_data)

    # ═══════════════════════════════════════════════════════════
    # STEP 4: MACRO BIAS PER PAIR  ⭐⭐⭐⭐⭐
    # ═══════════════════════════════════════════════════════════

    def generate_macro_bias(self, pair: str, macro_data: dict, regime_result: dict) -> dict:
        """
        Doc অনুযায়ী per-currency BUY/SELL bias বের করো:
          DXY ↑    -> USD strength ↑  -> USD pair-quote SELL bias (e.g. EURUSD↓)
          Gold ↓   -> confirms USD strength (DXY↑ + Gold↓ = confirmed)
          US10Y ↑  -> capital attraction -> USD support
          Risk-Off -> USD/JPY/CHF BUY bias, AUD/NZD/GBP/CAD SELL bias
        """
        clean_pair = pair.upper().replace("/", "").replace("=X", "")[:6]
        base  = clean_pair[:3]
        quote = clean_pair[3:6] if len(clean_pair) >= 6 else clean_pair[3:]

        dxy    = macro_data.get("dxy", {})
        gold   = macro_data.get("gold", {})
        yield_ = macro_data.get("us10y", {})

        environment = regime_result.get("environment", "NEUTRAL")
        preferred   = regime_result.get("preferred_assets", [])
        avoid       = regime_result.get("avoid_assets", [])

        usd_bias = "NEUTRAL"
        usd_confirmations = []

        if dxy.get("trend") == "BULLISH":
            usd_bias = "STRONG" if gold.get("trend") == "BEARISH" else "MODERATE"
            usd_confirmations.append("DXY bullish")
            if gold.get("trend") == "BEARISH":
                usd_confirmations.append("Gold bearish — confirms USD strength")
            if yield_.get("trend") == "BULLISH":
                usd_confirmations.append("US10Y yield rising — supports USD")

        elif dxy.get("trend") == "BEARISH":
            usd_bias = "STRONG" if gold.get("trend") == "BULLISH" else "MODERATE"
            usd_confirmations.append("DXY bearish")
            if gold.get("trend") == "BULLISH":
                usd_confirmations.append("Gold bullish — confirms USD weakness")
            if yield_.get("trend") == "BEARISH":
                usd_confirmations.append("US10Y yield falling — pressures USD")

        # Per-currency bias map
        currency_bias = {
            cur: self._single_currency_bias(cur, usd_bias, dxy.get("trend"), preferred, avoid)
            for cur in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
        }

        base_bias  = currency_bias.get(base, "NEUTRAL")
        quote_bias = currency_bias.get(quote, "NEUTRAL")
        pair_bias  = self._resolve_pair_bias(base_bias, quote_bias)

        return {
            "pair":              clean_pair,
            "base":              base,
            "quote":             quote,
            "usd_bias":          usd_bias,
            "usd_confirmations": usd_confirmations,
            "currency_bias":     currency_bias,
            "pair_bias":         pair_bias,
            "macro_regime":      environment,
        }

    def _single_currency_bias(self, currency, usd_bias, dxy_trend, preferred, avoid) -> str:
        # USD নিজে — DXY থেকে সরাসরি
        if currency == "USD":
            if dxy_trend == "BULLISH":
                return "BUY"
            if dxy_trend == "BEARISH":
                return "SELL"
            return "NEUTRAL"

        # Risk environment override (AUD/NZD/GBP/CAD vs JPY/CHF)
        if currency in preferred:
            return "BUY"
        if currency in avoid:
            return "SELL"

        # USD-counter currencies: USD strong -> these weak, and vice versa
        if currency in ("EUR", "GBP", "AUD", "NZD"):
            if usd_bias in ("STRONG", "MODERATE") and dxy_trend == "BULLISH":
                return "SELL"
            if usd_bias in ("STRONG", "MODERATE") and dxy_trend == "BEARISH":
                return "BUY"

        return "NEUTRAL"

    def _resolve_pair_bias(self, base_bias: str, quote_bias: str) -> str:
        if base_bias == "BUY" and quote_bias != "BUY":
            return "BUY"
        if base_bias == "SELL" and quote_bias != "SELL":
            return "SELL"
        if quote_bias == "SELL" and base_bias != "SELL":
            return "BUY"
        if quote_bias == "BUY" and base_bias != "BUY":
            return "SELL"
        return "NEUTRAL"

    # ═══════════════════════════════════════════════════════════
    # ⭐ MACRO SCORE  (DXY20 + Yield20 + Gold15 + VIX20 + SP500 15 + Oil10)
    # ═══════════════════════════════════════════════════════════

    def _calculate_macro_score(self, macro_data: dict, pair: str) -> tuple[int, dict]:
        """
        Doc-এর উদাহরণ অনুযায়ী একটা unified Macro Score (0-100) — কয়টা
        macro factor একটা স্পষ্ট (non-neutral) সংকেত দিচ্ছে তার ভিত্তিতে।
        Oil শুধু CAD-related pair-এ count হয়, কারণ Oil মূলত CAD-এর সাথে
        সম্পর্কিত (doc Section 4)।
        """
        clean_pair = pair.upper().replace("/", "").replace("=X", "")[:6]
        is_cad_pair = "CAD" in clean_pair

        factors = {
            "dxy":   macro_data.get("dxy", {}).get("trend"),
            "yield": macro_data.get("us10y", {}).get("trend"),
            "gold":  macro_data.get("gold", {}).get("trend"),
            "vix":   macro_data.get("vix", {}).get("trend"),
            "sp500": macro_data.get("sp500", {}).get("trend"),
        }
        if is_cad_pair:
            factors["oil"] = macro_data.get("oil", {}).get("trend")

        breakdown = {}
        score = 0
        for name, trend in factors.items():
            if trend in ("BULLISH", "BEARISH"):
                breakdown[name] = MACRO_SCORE_WEIGHTS[name]
                score += MACRO_SCORE_WEIGHTS[name]

        return min(100, score), breakdown

    # ═══════════════════════════════════════════════════════════
    # ⭐ CROSS-ASSET CONFIRMATION
    # ═══════════════════════════════════════════════════════════

    def _cross_asset_confirmation(self, macro_data: dict, bias_result: dict) -> dict:
        """
        USD BUY নিতে হলে doc অনুযায়ী confluence দরকার:
            DXY ↑ + Yield ↑ + Gold ↓ + USD strength ↑
        এই মেথড সেই confluence count করে confirmed/unconfirmed বলে —
        false signal কমানোর জন্য।
        """
        dxy    = macro_data.get("dxy", {}).get("trend")
        yield_ = macro_data.get("us10y", {}).get("trend")
        gold   = macro_data.get("gold", {}).get("trend")
        usd_bias = bias_result.get("usd_bias", "NEUTRAL")

        if usd_bias in ("STRONG", "MODERATE") and dxy == "BULLISH":
            checks = {
                "dxy_bullish":   True,
                "yield_bullish": yield_ == "BULLISH",
                "gold_bearish":  gold == "BEARISH",
            }
        elif usd_bias in ("STRONG", "MODERATE") and dxy == "BEARISH":
            checks = {
                "dxy_bearish":   True,
                "yield_bearish": yield_ == "BEARISH",
                "gold_bullish":  gold == "BULLISH",
            }
        else:
            checks = {}

        confirmed_count = sum(1 for v in checks.values() if v)
        total = len(checks) or 1
        confirmed = bool(checks) and confirmed_count >= 2

        return {
            "checks":          checks,
            "confirmed_count": confirmed_count,
            "total_checks":    total,
            "confirmed":       confirmed,
            "note": (
                f"{confirmed_count}/{total} cross-asset factors confirm — "
                f"{'high confidence' if confirmed else 'weak confluence, reduce confidence'}"
                if checks else "No directional USD bias to confirm"
            ),
        }

    # ═══════════════════════════════════════════════════════════
    # ⭐ EVENT RISK INTEGRATION
    # ═══════════════════════════════════════════════════════════

    def _event_risk_integration(self, news_ctx: dict) -> dict:
        """
        FOMC / high-impact USD news চলাকালীন macro confidence কমিয়ে
        দাও — doc-এর উদাহরণ অনুযায়ী ("US10Y volatility high -> reduce
        USD trade confidence")। news_ctx আসে NewsFilter.get_ai_context()
        থেকে (analysis_agent.py-তে আগে থেকেই উপলব্ধ)।
        """
        risk_level = news_ctx.get("risk_level", "LOW") if news_ctx else "LOW"
        upcoming   = news_ctx.get("upcoming_events", []) if news_ctx else []

        high_impact_usd = any(
            kw in str(e).upper() for e in upcoming for kw in ("FOMC", "FED", "NFP", "CPI")
        )

        if risk_level == "HIGH" or high_impact_usd:
            return {
                "elevated":           True,
                "confidence_penalty": 20,
                "note": "High-impact USD event nearby (FOMC/NFP/CPI) — macro confidence reduced",
            }

        return {"elevated": False, "confidence_penalty": 0, "note": "No major event risk detected"}

    # ═══════════════════════════════════════════════════════════
    # ⭐ MACRO + SMC FUSION  ⭐⭐⭐⭐⭐
    # ═══════════════════════════════════════════════════════════

    def fuse_with_smc(self, intermarket_result: dict, smc_ctx: dict, session_ctx: dict = None) -> dict:
        """
        Macro bias + SMC confluence (+ Session, optional) একসাথে মিলে
        গেলে high probability setup — doc-এর "Macro + SMC Fusion"
        example:
            Macro: Risk-Off -> USD strong
            Technical: EURUSD liquidity sweep + Bearish CHoCH
            Session: London
            = High Probability SELL
        """
        smc_ctx     = smc_ctx or {}
        session_ctx = session_ctx or {}

        macro_bias  = intermarket_result["bias"]["pair_bias"]
        macro_score = intermarket_result["macro_score"]
        smc_signal  = smc_ctx.get("smc_signal", "WAIT")
        smc_score   = smc_ctx.get("smc_score", 0)

        aligned = (
            (macro_bias == "BUY"  and smc_signal == "BUY") or
            (macro_bias == "SELL" and smc_signal == "SELL")
        )

        # P0 fix (audit C8): renamed `fusion_score` → `macro_fusion_score` to
        # disambiguate from analysis/session_analyzer.py:402 `fusion_score` (which
        # uses a different formula: smc_score*0.6 + session gate). Both used to
        # write the same dict key with different semantics — downstream consumers
        # (decision_agent.py:295, trade_permission.py:213, master_analyst.py:475,
        # confluence_engine.py:269) couldn't tell which source produced the value.
        # Now this module's score is unambiguously the macro+SMC fusion.
        macro_fusion_score = round(macro_score * 0.4 + smc_score * 0.5 + (10 if aligned else 0))
        macro_fusion_score = min(100, macro_fusion_score)

        if session_ctx.get("is_overlap") and aligned:
            macro_fusion_score = min(100, macro_fusion_score + 5)

        if macro_fusion_score >= 85 and aligned:
            grade = "A+"
        elif macro_fusion_score >= 70 and aligned:
            grade = "A"
        elif macro_fusion_score >= MIN_FUSION_SCORE:
            grade = "B"
        else:
            grade = "INVALID"

        signal = macro_bias if (aligned and macro_fusion_score >= MIN_FUSION_SCORE) else "WAIT"

        return {
            "fusion_allowed": aligned and macro_fusion_score >= MIN_FUSION_SCORE,
            "macro_fusion_score": macro_fusion_score,  # renamed from fusion_score
            # Backward-compat: keep fusion_score as alias so existing readers
            # that explicitly want the macro fusion can still find it, but
            # prefer macro_fusion_score in new code.
            "fusion_score":   macro_fusion_score,
            "fusion_grade":   grade,
            "aligned":        aligned,
            "signal":         signal,
            "note": (
                f"Macro({macro_bias}) {'aligns' if aligned else 'conflicts'} with "
                f"SMC({smc_signal}) — macro fusion score {macro_fusion_score}/100"
            ),
        }

    # ═══════════════════════════════════════════════════════════
    # ⭐ HISTORICAL MACRO MEMORY
    # ═══════════════════════════════════════════════════════════

    def _save_memory(self, result: dict) -> None:
        try:
            os.makedirs("memory", exist_ok=True)
            history = self._load_memory()
            macro = result["macro_data"]
            entry = {
                "timestamp":    result["timestamp"],
                "pair":         result["pair"],
                "regime":       result["regime"].get("environment"),
                "macro_score":  result["macro_score"],
                "pair_bias":    result["bias"].get("pair_bias"),
                "dxy_value":    macro.get("dxy", {}).get("value"),
                "dxy_trend":    macro.get("dxy", {}).get("trend"),
                "gold_change":  macro.get("gold", {}).get("change_pct"),
                "oil_change":   macro.get("oil", {}).get("change_pct"),
                "bond_yield":   macro.get("us10y", {}).get("value"),
                "sp500_change": macro.get("sp500", {}).get("change_pct"),
                "vix_value":    macro.get("vix", {}).get("value"),
            }
            history.append(entry)
            with open(MACRO_MEMORY_PATH, "w") as f:
                json.dump(history[-500:], f, indent=2)
        except Exception as e:
            log.warning(f"[IntermarketEngine] Memory save failed: {e}")

    def _load_memory(self) -> list:
        if not os.path.exists(MACRO_MEMORY_PATH):
            return []
        try:
            with open(MACRO_MEMORY_PATH) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Suppressed exception at line 466: {e}")
            return []

    def get_regime_history(self, pair: str = None, limit: int = 20) -> list:
        """পূর্ববর্তী Risk-On/Risk-Off episode-এ pair কেমন macro context পেয়েছিল তা দেখার জন্য raw history।"""
        history = self._load_memory()
        if pair:
            clean = pair.upper().replace("/", "").replace("=X", "")[:6]
            history = [h for h in history if h.get("pair") == clean]
        return history[-limit:]

    # ═══════════════════════════════════════════════════════════
    # AI CONTEXT  (MasterAnalyst handoff)
    # ═══════════════════════════════════════════════════════════

    def get_ai_context(self, result: dict) -> dict:
        macro   = result["macro_data"]
        bias    = result["bias"]
        regime  = result["regime"]
        confirm = result["confirmation"]
        event   = result["event_risk"]

        return {
            "dxy_trend":               macro.get("dxy", {}).get("trend"),
            "dxy_change_pct":          macro.get("dxy", {}).get("change_pct"),
            "gold_trend":              macro.get("gold", {}).get("trend"),
            "oil_trend":               macro.get("oil", {}).get("trend"),
            "us10y_trend":             macro.get("us10y", {}).get("trend"),
            "sp500_trend":             macro.get("sp500", {}).get("trend"),
            "vix_value":               macro.get("vix", {}).get("value"),
            "vix_trend":               macro.get("vix", {}).get("trend"),
            "macro_regime":            regime.get("environment"),
            "macro_regime_confidence": regime.get("confidence"),
            "trading_mode":            regime.get("trading_mode"),
            "usd_bias":                bias.get("usd_bias"),
            "usd_confirmations":       bias.get("usd_confirmations"),
            "macro_pair_bias":         bias.get("pair_bias"),
            "macro_currency_bias":     bias.get("currency_bias"),
            "macro_score":             result.get("macro_score", 0),
            "cross_asset_confirmed":   confirm.get("confirmed"),
            "cross_asset_note":        confirm.get("note"),
            "event_risk_elevated":     event.get("elevated"),
            "event_risk_penalty":      event.get("confidence_penalty"),
            # FIX: result["correlations"] is now {"pair": ..., "matrix": {...}}
            # so .get("matrix", {}) correctly returns the nested correlation dict.
            "macro_correlations":      result.get("correlations", {}).get("matrix", {}),
        }

    # ═══════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # FIX: removed self.corr_engine.print_summary(result["correlations"])
    # — that method expects a full analyze() result dict (with
    # "volatility_state", "atr_current", etc.), not the lightweight
    # {"pair": ..., "matrix": ...} dict we build in calculate_correlations().
    # Calling it with the wrong shape always logged "Pair: ?" and
    # printed zeros for every ATR field on the first cycle.
    # The correlation matrix is still available via get_ai_context()
    # under the "macro_correlations" key for downstream consumers.
    # ═══════════════════════════════════════════════════════════

    def print_summary(self, result: dict) -> None:
        bar = "═" * 58
        print(f"\n{bar}")
        print(f"  🌎  INTERMARKET ANALYSIS ENGINE  (Day 65) — {result['pair']}")
        print(bar)

        self.macro_provider.print_summary(result["macro_data"])
        self.risk_engine.print_summary(result["regime"])
        # NOTE: corr_engine.print_summary() intentionally removed here.
        # It printed "Pair: ?" because it received the lightweight matrix
        # dict instead of a full analyze() result. The correlation data
        # is surfaced via get_ai_context()["macro_correlations"] instead.

        bias = result["bias"]
        print("  ── Macro Bias ──")
        print(f"  USD Bias     : {bias['usd_bias']}")
        for c in bias["usd_confirmations"]:
            print(f"    • {c}")
        print(f"  Pair Bias    : {bias['pair_bias']}  ({bias['pair']})")
        print()
        print(f"  Macro Score  : {result['macro_score']}/100")
        print(f"  Confirmation : {result['confirmation']['note']}")
        print(f"  Event Risk   : {result['event_risk']['note']}")
        print(bar + "\n")


# ═══════════════════════════════════════════════════════════════
# QUICK RUN — Direct test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = IntermarketEngine()
    result = engine.analyze("GBPUSD")
    engine.print_summary(result)

    ctx = engine.get_ai_context(result)
    print("AI Context (for MasterAnalyst):")
    for k, v in ctx.items():
        print(f"  {k:<28}: {v}")