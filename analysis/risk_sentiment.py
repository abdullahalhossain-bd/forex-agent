# analysis/risk_sentiment.py  —  Day 65 | Global Risk Sentiment Engine
# ============================================================
# SP500 + VIX (+ DXY) দেখে market-এর overall risk appetite বের করে:
#
#   RISK_ON  -> SP500 ↑, VIX ↓  -> AUD/NZD/GBP/CAD buy bias, USD/JPY/CHF sell bias
#   RISK_OFF -> SP500 ↓, VIX ↑  -> USD/JPY/CHF buy bias, AUD/NZD/GBP/CAD sell bias
#
# এছাড়া VIX Fear Index আলাদাভাবে trading mode
# (NORMAL / CAUTIOUS / DEFENSIVE) নির্ধারণ করে।
# ============================================================

from utils.logger import get_logger

log = get_logger("risk_sentiment")

RISK_ON_ASSETS  = ["AUD", "NZD", "GBP", "CAD"]
RISK_OFF_ASSETS = ["USD", "JPY", "CHF"]

# VIX value -> fear label  (lo inclusive, hi exclusive)
VIX_FEAR_LEVELS = {
    "LOW":      (0, 15),
    "NORMAL":   (15, 20),
    "ELEVATED": (20, 26),
    "HIGH":     (26, 35),
    "EXTREME":  (35, 999),
}

FEAR_TO_MODE = {
    "LOW":      "NORMAL",
    "NORMAL":   "NORMAL",
    "ELEVATED": "CAUTIOUS",
    "HIGH":     "DEFENSIVE",
    "EXTREME":  "DEFENSIVE",
}


class RiskSentimentEngine:
    """
    Usage:
        engine = RiskSentimentEngine()
        result = engine.analyze(macro_data)   # macro_data = MacroDataProvider().get_all()
        ctx    = engine.get_ai_context(result)
    """

    def analyze(self, macro_data: dict) -> dict:
        sp500 = macro_data.get("sp500", {})
        vix   = macro_data.get("vix", {})
        dxy   = macro_data.get("dxy", {})

        sp_trend  = sp500.get("trend", "NEUTRAL")
        vix_trend = vix.get("trend", "NEUTRAL")
        vix_value = vix.get("value")

        environment, confidence, reasons = self._classify_environment(
            sp_trend, vix_trend, dxy.get("trend")
        )
        fear_level, trading_mode = self._classify_fear(vix_value)
        preferred, avoid         = self._preferred_assets(environment)

        result = {
            "environment":      environment,
            "confidence":       confidence,
            "reasons":          reasons,
            "preferred_assets": preferred,
            "avoid_assets":     avoid,
            "vix_value":        vix_value,
            "fear_level":       fear_level,
            "trading_mode":     trading_mode,
            "sp500_trend":      sp_trend,
            "vix_trend":        vix_trend,
        }

        log.info(
            f"[RiskSentiment] {environment} (conf {confidence}%) | "
            f"VIX={vix_value} ({fear_level}) | Mode={trading_mode}"
        )
        return result

    # ═══════════════════════════════════════════════════════════
    # ENVIRONMENT CLASSIFICATION
    # ═══════════════════════════════════════════════════════════

    def _classify_environment(self, sp_trend: str, vix_trend: str, dxy_trend: str):
        reasons = []
        score   = 0.0   # positive = risk-on, negative = risk-off

        if sp_trend == "BULLISH":
            score += 1
            reasons.append("S&P500 rising — risk appetite improving")
        elif sp_trend == "BEARISH":
            score -= 1
            reasons.append("S&P500 falling — risk appetite deteriorating")

        if vix_trend == "BEARISH":      # VIX falling = fear receding = risk-on
            score += 1
            reasons.append("VIX falling — fear receding")
        elif vix_trend == "BULLISH":
            score -= 1
            reasons.append("VIX rising — fear increasing")

        if dxy_trend == "BEARISH":
            score += 0.5
            reasons.append("DXY weakening — supports risk-on flows")
        elif dxy_trend == "BULLISH":
            score -= 0.5
            reasons.append("DXY strengthening — supports risk-off flows")

        if score >= 1.5:
            environment, confidence = "RISK_ON", min(90, 60 + int(score * 10))
        elif score <= -1.5:
            environment, confidence = "RISK_OFF", min(90, 60 + int(abs(score) * 10))
        else:
            environment, confidence = "NEUTRAL", 40

        return environment, confidence, reasons

    def _classify_fear(self, vix_value):
        if vix_value is None:
            return "UNKNOWN", "NORMAL"

        fear = "EXTREME"
        for label, (lo, hi) in VIX_FEAR_LEVELS.items():
            if lo <= vix_value < hi:
                fear = label
                break

        return fear, FEAR_TO_MODE.get(fear, "NORMAL")

    def _preferred_assets(self, environment: str):
        if environment == "RISK_ON":
            return RISK_ON_ASSETS, RISK_OFF_ASSETS
        if environment == "RISK_OFF":
            return RISK_OFF_ASSETS, RISK_ON_ASSETS
        return [], []

    # ═══════════════════════════════════════════════════════════
    # AI CONTEXT
    # ═══════════════════════════════════════════════════════════

    def get_ai_context(self, result: dict) -> dict:
        return {
            "risk_environment":      result["environment"],
            "risk_confidence":       result["confidence"],
            "risk_preferred_assets": result["preferred_assets"],
            "risk_avoid_assets":     result["avoid_assets"],
            "vix_value":             result["vix_value"],
            "vix_fear_level":        result["fear_level"],
            "trading_mode":          result["trading_mode"],
        }

    # ═══════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════════

    def print_summary(self, result: dict) -> None:
        bar  = "─" * 48
        icon = {"RISK_ON": "🟢", "RISK_OFF": "🔴", "NEUTRAL": "🟡"}.get(result["environment"], "⚪")
        print(f"\n{bar}")
        print("  ⚖️  RISK SENTIMENT  (Day 65)")
        print(bar)
        print(f"  Environment : {icon} {result['environment']}  ({result['confidence']}%)")
        print(f"  VIX         : {result['vix_value']}  [{result['fear_level']}]")
        print(f"  Mode        : {result['trading_mode']}")
        print(f"  Preferred   : {', '.join(result['preferred_assets']) or '-'}")
        print(f"  Avoid       : {', '.join(result['avoid_assets']) or '-'}")
        print("  ── Reasons ──")
        for r in result["reasons"]:
            print(f"  • {r}")
        print(bar + "\n")