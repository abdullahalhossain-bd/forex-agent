"""
test_barrier_fixes.py — Verify the 6 trade-blocking barriers are fixed.

Tests each barrier the user identified:
  Barrier 1: DecisionAgent voting promotes rule signal when master+LLM both WAIT
  Barrier 2: MasterAnalyst error → master_ctx gets safe default (not {})
  Barrier 4: placeholder_risk approved=False no longer blocks voting
  Barrier 6: MAX_LLM_CALLS_PER_CYCLE is 8 (not 5)

Run:
    python test_barrier_fixes.py

Exits 0 if all tests pass, 1 otherwise.
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

import logging
logging.basicConfig(level=logging.WARNING)

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


print("=" * 70)
print("  BARRIER FIX VERIFICATION")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────
# Barrier 6: MAX_LLM_CALLS_PER_CYCLE = 8
# ──────────────────────────────────────────────────────────────────
print("\n── Barrier 6: MAX_LLM_CALLS_PER_CYCLE ──")
from config import MAX_LLM_CALLS_PER_CYCLE
check("MAX_LLM_CALLS_PER_CYCLE >= 8",
      MAX_LLM_CALLS_PER_CYCLE >= 8,
      f"actual={MAX_LLM_CALLS_PER_CYCLE}")

# ──────────────────────────────────────────────────────────────────
# Barrier 1: DecisionAgent rule-signal promotion
# ──────────────────────────────────────────────────────────────────
print("\n── Barrier 1: DecisionAgent rule-signal promotion ──")
import inspect
from agents.decision_agent import DecisionAgent
src = inspect.getsource(DecisionAgent.decide)
check("rule-signal promotion code present",
      "Barrier-1 fix" in src and "promoted to 3 votes" in src,
      "code path exists in DecisionAgent.decide")

# Simulate: master=WAIT, llm=WAIT, rule=BUY(65%) → should return BUY
da = DecisionAgent()
fake_market = {
    "symbol": "EURUSD", "timeframe": "15m",
    "ind_ctx": {"close": 1.0850, "trend": "bullish"},
    "regime": {"regime": "TRENDING", "volatility": "NORMAL"},
}
fake_analysis = {
    "final_signal": "WAIT",  # ← not BUY/SELL, so TEST_MODE bypass won't fire
    "signal": {"signal": "BUY", "confidence": 65, "entry": 1.0850},
    "llm": {"signal": "WAIT", "confidence": 0},
    "news": {"trade_allowed": True},
    "news_ctx": {"news_trade_allowed": True},
    "sentiment_ctx": {"sentiment_bias": "NEUTRAL", "sentiment_score": 0},
    "conflict": {"has_conflict": False, "confidence_adjustment": 0},
    "master_ctx": {
        "master_signal": "WAIT",
        "master_confidence": 0,
        "master_entry": None, "master_sl": None, "master_tp1": None,
        "master_story": "LLM unavailable",
        "master_risks": [], "master_critique": "",
    },
}
# Use placeholder_risk with approved=True so we get past barrier 4 test
fake_risk = {"approved": True, "lot": 0.01, "sl_pips": 15, "tp_pips": 30, "rr_ratio": 2.0}
try:
    result = da.decide(fake_market, fake_analysis, fake_risk)
    check("rule signal promoted → decision=BUY",
          result.get("decision") == "BUY",
          f"actual decision={result.get('decision')} conf={result.get('confidence')}")
except Exception as e:
    check("rule signal promoted → decision=BUY", False, f"raised: {e}")

# ──────────────────────────────────────────────────────────────────
# Barrier 4: placeholder_risk no longer blocks voting
# ──────────────────────────────────────────────────────────────────
print("\n── Barrier 4: placeholder_risk no longer blocks voting ──")
check("placeholder detection code present",
      "Barrier-4 fix" in src and "_is_placeholder" in src,
      "code path exists in DecisionAgent.decide")

# Same scenario but with placeholder_risk (approved=False, lot=0, sl_pips=0, etc.)
fake_placeholder = {
    "approved": False,  # ← key: False because final_signal is WAIT
    "lot": 0, "sl_pips": 0, "tp_pips": 0, "rr_ratio": 0,
}
try:
    result = da.decide(fake_market, fake_analysis, fake_placeholder)
    check("placeholder_risk no longer blocks voting",
          result.get("decision") == "BUY",
          f"actual decision={result.get('decision')} (should be BUY from rule promotion)")
except Exception as e:
    check("placeholder_risk no longer blocks voting", False, f"raised: {e}")

# ──────────────────────────────────────────────────────────────────
# Barrier 2: MasterAnalyst error → master_ctx safe default
# ──────────────────────────────────────────────────────────────────
print("\n── Barrier 2: MasterAnalyst error → master_ctx safe default ──")
from agents.analysis_agent import AnalysisAgent
src_aa = inspect.getsource(AnalysisAgent.run)
check("master_ctx safe default code present",
      "Barrier 2" in src_aa and "LLM unavailable — rule engine fallback" in src_aa,
      "except block populates master_ctx with rule signal")

# ──────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"  Result: {PASS} PASS, {FAIL} FAIL")
print("=" * 70)
sys.exit(0 if FAIL == 0 else 1)
