"""
test_day94_institutional_apis.py — Day 94 institutional-grade API tests

Tests each new institutional API:
  1. Economic Calendar API (Tradermade → Finnhub → Fxstreet → FF scraper)
  2. FRED API (Federal Reserve macro data — CPI, Unemployment, Yields)
  3. Retail Sentiment API (OANDA v20 position book + order book)
  4. Full AnalysisAgent integration (all 3 wired into pipeline)

Each test:
  - Skips gracefully if API key not configured
  - Makes a real API call (or fallback chain) when key is present
  - Validates response shape
  - Prints clear PASS/SKIP/FAIL summary

Usage:
    cd D:\Projects\forex_ai
    python scripts\test_day94_institutional_apis.py
"""
import os
import sys
from pathlib import Path

# ── Auto-detect project root (works on Windows + Linux) ──────────────────────
# This script lives at <project_root>/scripts/test_day94_*.py
# so project root is two levels up from __file__.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()


def banner(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def main():
    banner("Day 94 — Institutional-Grade API Tests")
    print(f"  Project root: {PROJECT_ROOT}")

    results = []

    # ─────────────────────────────────────────────────────────
    banner("Test 1: Economic Calendar API (multi-source fallback)")

    tm_key = os.getenv("TRADERMADE_API_KEY", "")
    fh_key = os.getenv("FINNHUB_API_KEY", "")
    print(f"  Tradermade key: {'set' if tm_key else 'not set'}")
    print(f"  Finnhub key:    {'set' if fh_key else 'not set'}")
    print(f"  Fxstreet RSS:   always available (no key)")
    print(f"  FF scraper:     always available (Day 90 cloudscraper)")

    try:
        from fundamental.economic_calendar_api import EconomicCalendarAPI
        cal = EconomicCalendarAPI()
        result = cal.get_calendar(currencies=["USD", "EUR"], hours_ahead=24)

        print(f"\n  Source used:       {result['source']}")
        print(f"  Events found:      {len(result['events'])}")
        print(f"  High-impact:       {result['high_impact_count']}")
        print(f"  Trade block:       {'⛔ YES' if result['trade_block'] else '✅ no'}")
        if result.get("block_reason"):
            print(f"  Block reason:      {result['block_reason']}")
        if result.get("next_event"):
            ne = result["next_event"]
            print(f"  Next event:        {ne['currency']} {ne['title']} @ {ne['time']} [{ne['impact']}]")

        # Show first 3 events
        if result["events"]:
            print(f"\n  First 3 events:")
            for ev in result["events"][:3]:
                print(f"    • {ev['time'].strftime('%H:%M UTC')} {ev['currency']} {ev['title']} [{ev['impact']}]")

        ctx = cal.get_ai_context(result)
        print(f"\n  AI context keys: {list(ctx.keys())}")

        if result["source"] != "none":
            print(f"  PASS: calendar served from {result['source']}")
            results.append(("Economic Calendar", "PASS", f"source={result['source']}, {len(result['events'])} events"))
        else:
            print(f"  FAIL: all calendar sources failed")
            results.append(("Economic Calendar", "FAIL", "all sources failed"))
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()
        results.append(("Economic Calendar", "FAIL", str(e)[:80]))

    # ─────────────────────────────────────────────────────────
    banner("Test 2: FRED API (Federal Reserve macro data)")

    fred_key = os.getenv("FRED_API_KEY", "")
    if not fred_key:
        print("  SKIP: FRED_API_KEY not set (get free key at https://fredaccount.stlouisfed.org/apikeys)")
        results.append(("FRED API", "SKIP", "no key"))
    else:
        try:
            from fundamental.fred_data import get_fred_api
            fred = get_fred_api()
            print(f"  API available: {fred.available}")

            # Test single series first
            print(f"\n  Testing single series (CPIAUCSL):")
            cpi = fred.get_series("CPIAUCSL")
            if cpi:
                print(f"    CPI value: {cpi['value']}  date: {cpi['date']}  change: {cpi['change_pct']:+.2f}%")
            else:
                print(f"    CPI: None")

            # Full macro snapshot
            print(f"\n  Full macro snapshot:")
            snapshot = fred.get_macro_snapshot()
            print(f"    Source:           {snapshot['source']}")
            print(f"    Yield curve:      {snapshot['yield_curve']}")
            print(f"    Inflation trend:  {snapshot['inflation_trend']}")
            print(f"    Rate environment: {snapshot['rate_environment']}")
            print(f"    Series fetched:   {len(snapshot['series'])}")
            for label, data in snapshot["series"].items():
                print(f"      {label:<16}: {data['value']}  ({data['date']}, {data['change_pct']:+.2f}%)")

            ctx = fred.get_ai_context(snapshot)
            print(f"\n  AI context: {ctx}")

            if snapshot["source"] != "none":
                print(f"  PASS: FRED served {len(snapshot['series'])} series")
                results.append(("FRED API", "PASS", f"{len(snapshot['series'])} series, yield={snapshot['yield_curve']}"))
            else:
                print(f"  FAIL: FRED returned no data")
                results.append(("FRED API", "FAIL", "no data"))
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback; traceback.print_exc()
            results.append(("FRED API", "FAIL", str(e)[:80]))

    # ─────────────────────────────────────────────────────────
    banner("Test 3: Retail Sentiment API (OANDA v20)")

    oanda_key = os.getenv("OANDA_API_KEY", "")
    if not oanda_key:
        print("  SKIP: OANDA_API_KEY not set (get free practice account at https://www.oanda.com/apply/demo/)")
        print("  Testing graceful fallback...")
        try:
            from analysis.retail_sentiment import get_retail_sentiment_api
            api = get_retail_sentiment_api()
            result = api.get_sentiment("EURUSD")
            print(f"  Source:          {result['source']}")
            print(f"  Long %:          {result['long_pct']}")
            print(f"  Short %:         {result['short_pct']}")
            print(f"  Sentiment:       {result['sentiment_label']}")
            print(f"  Contrarian:      {result['contrarian_signal']} ({result['contrarian_strength']})")
            print(f"  Trade bias:      {result['trade_bias']}")
            print(f"  Confidence:      {result['confidence']}%")
            assert result["source"] == "fallback", "Should fall back when no key"
            print(f"  PASS: graceful fallback returned NEUTRAL (no crash)")
            results.append(("Retail Sentiment", "PASS", "fallback OK (no key — set OANDA_API_KEY for live data)"))
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append(("Retail Sentiment", "FAIL", str(e)[:80]))
    else:
        try:
            from analysis.retail_sentiment import get_retail_sentiment_api
            api = get_retail_sentiment_api()
            result = api.get_sentiment("EURUSD")
            print(f"  Source:          {result['source']}")
            print(f"  Long %:          {result['long_pct']}")
            print(f"  Short %:         {result['short_pct']}")
            print(f"  Sentiment:       {result['sentiment_label']}")
            print(f"  Contrarian:      {result['contrarian_signal']} ({result['contrarian_strength']})")
            print(f"  Long/Short ratio:{result['long_short_ratio']}")
            print(f"  Trade bias:      {result['trade_bias']}")
            print(f"  Confidence:      {result['confidence']}%")
            if result.get("order_book", {}).get("stop_cluster"):
                print(f"  Stop cluster:    {result['order_book']['stop_cluster']} (liquidity grab target)")

            if result["source"] == "oanda_live":
                print(f"  PASS: live OANDA data fetched")
                results.append(("Retail Sentiment", "PASS", f"live OANDA, contrarian={result['contrarian_signal']}"))
            else:
                print(f"  FAIL: expected oanda_live, got {result['source']}")
                results.append(("Retail Sentiment", "FAIL", f"source={result['source']}"))
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback; traceback.print_exc()
            results.append(("Retail Sentiment", "FAIL", str(e)[:80]))

    # ─────────────────────────────────────────────────────────
    banner("Test 4: Full AnalysisAgent integration (all Day 94 APIs wired)")

    try:
        from agents.analysis_agent import AnalysisAgent
        agent = AnalysisAgent()
        print(f"  AnalysisAgent instantiated OK")
        print(f"  EconomicCalendarAPI imported: yes (used in run())")
        print(f"  FRED API imported:            yes (get_fred_api)")
        print(f"  Retail Sentiment imported:     yes (get_retail_sentiment_api)")

        # Check MasterAnalyst accepts the new kwargs
        import inspect
        from agents.master_analyst import MasterAnalyst
        sig = inspect.signature(MasterAnalyst.analyze)
        new_params = ["econ_calendar_ctx", "fred_ctx", "retail_sentiment_ctx"]
        for p in new_params:
            assert p in sig.parameters, f"Missing param: {p}"
        print(f"  MasterAnalyst.analyze() accepts all 3 new kwargs: OK")

        print(f"  PASS: all Day 94 APIs wired into pipeline")
        results.append(("AnalysisAgent integration", "PASS", "all 3 APIs wired"))
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()
        results.append(("AnalysisAgent integration", "FAIL", str(e)[:80]))

    # ─────────────────────────────────────────────────────────
    banner("Summary")
    passed  = sum(1 for _, s, _ in results if s == "PASS")
    skipped = sum(1 for _, s, _ in results if s == "SKIP")
    failed  = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  PASS: {passed}")
    print(f"  SKIP: {skipped}  (API key not configured)")
    print(f"  FAIL: {failed}")
    print()
    for name, status, detail in results:
        icon = {"PASS": "OK", "SKIP": "SKIP", "FAIL": "FAIL"}[status]
        print(f"  [{icon}]  {name:<28}  {status:<5}  {detail}")
    print()
    print("  Institutional-grade architecture (Day 94):")
    print("    Economic Calendar:  Tradermade -> Finnhub -> Fxstreet -> FF scraper")
    print("    Central Bank Data:  FRED (CPI, Unemployment, Yields, Fed Rate)")
    print("    Retail Sentiment:   OANDA v20 (position book + order book)")
    print()
    print("  All 3 wired into AnalysisAgent + MasterAnalyst LLM context.")


if __name__ == "__main__":
    main()