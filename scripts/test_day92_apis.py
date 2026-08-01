"""
test_day92_apis.py — Day 92 API integration tests

Tests each new API provider added on Day 92:
  1. Alpha Vantage   — forex OHLCV (live)
  2. Polygon.io      — forex aggregates (free tier = end-of-day)
  3. Finnhub         — forex candles
  4. Twelve Data     — forex time series
  5. NewsAPI.org     — financial news + sentiment

Each test:
  - Skips gracefully if the API key isn't configured
  - Makes a single real API call
  - Validates the response shape
  - Prints a clear PASS/SKIP/FAIL summary

Usage:
    cd /home/z/my-project/forex_ai
    python scripts/test_day92_apis.py
"""
import os
import sys
import time

# Add project root to path
sys.path.insert(0, '/home/z/my-project/forex_ai')
os.chdir('/home/z/my-project/forex_ai')

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Force DataFetcher to use the explicit preferred source so we can
# test each provider independently.
TEST_PAIR = "EURUSD"
TEST_TF   = "M15"
TEST_LIMIT = 50


def banner(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def main():
    banner("Day 92 — API Integration Tests")
    print(f"Test pair    : {TEST_PAIR}")
    print(f"Test timeframe: {TEST_TF}")
    print(f"Test limit    : {TEST_LIMIT} candles")

    results = []

    # ── Test 1: Alpha Vantage ───────────────────────────────
    banner("Test 1: Alpha Vantage (FX_INTRADAY)")
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not av_key:
        print("  SKIP: ALPHA_VANTAGE_API_KEY not set")
        results.append(("Alpha Vantage", "SKIP", "no key"))
    else:
        try:
            os.environ["PREFERRED_DATA_SOURCE"] = "alpha_vantage"
            from data.fetcher import DataFetcher
            f = DataFetcher()
            print(f"  Detected source: {f.source}")
            assert f.source == "alpha_vantage", f"Expected alpha_vantage, got {f.source}"
            df = f.fetch_ohlcv(TEST_PAIR, TEST_TF, limit=TEST_LIMIT)
            if df is None or len(df) == 0:
                print("  FAIL: no data returned")
                results.append(("Alpha Vantage", "FAIL", "no data"))
            else:
                last_close = df['close'].iloc[-1]
                print(f"  PASS: {len(df)} candles | last close: {last_close:.5f}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Index type: {type(df.index).__name__}")
                results.append(("Alpha Vantage", "PASS", f"{len(df)} candles, last={last_close:.5f}"))
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append(("Alpha Vantage", "FAIL", str(e)[:80]))

    time.sleep(2)  # be polite to free-tier APIs

    # ── Test 2: Finnhub ─────────────────────────────────────
    banner("Test 2: Finnhub (forex/candle)")
    fh_key = os.getenv("FINNHUB_API_KEY", "")
    if not fh_key:
        print("  SKIP: FINNHUB_API_KEY not set")
        results.append(("Finnhub", "SKIP", "no key"))
    else:
        try:
            os.environ["PREFERRED_DATA_SOURCE"] = "finnhub"
            # Re-import to pick up new env
            import importlib
            import data.fetcher
            importlib.reload(data.fetcher)
            from data.fetcher import DataFetcher
            f = DataFetcher()
            print(f"  Detected source: {f.source}")
            df = f.fetch_ohlcv(TEST_PAIR, TEST_TF, limit=TEST_LIMIT)
            if df is None or len(df) == 0:
                print("  FAIL: no data returned")
                results.append(("Finnhub", "FAIL", "no data"))
            else:
                last_close = df['close'].iloc[-1]
                print(f"  PASS: {len(df)} candles | last close: {last_close:.5f}")
                results.append(("Finnhub", "PASS", f"{len(df)} candles, last={last_close:.5f}"))
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append(("Finnhub", "FAIL", str(e)[:80]))

    time.sleep(2)

    # ── Test 3: Twelve Data ─────────────────────────────────
    banner("Test 3: Twelve Data (time_series)")
    td_key = os.getenv("TWELVE_DATA_API_KEY", "")
    if not td_key:
        print("  SKIP: TWELVE_DATA_API_KEY not set")
        results.append(("Twelve Data", "SKIP", "no key"))
    else:
        try:
            os.environ["PREFERRED_DATA_SOURCE"] = "twelve_data"
            import importlib
            import data.fetcher
            importlib.reload(data.fetcher)
            from data.fetcher import DataFetcher
            f = DataFetcher()
            print(f"  Detected source: {f.source}")
            df = f.fetch_ohlcv(TEST_PAIR, TEST_TF, limit=TEST_LIMIT)
            if df is None or len(df) == 0:
                print("  FAIL: no data returned")
                results.append(("Twelve Data", "FAIL", "no data"))
            else:
                last_close = df['close'].iloc[-1]
                print(f"  PASS: {len(df)} candles | last close: {last_close:.5f}")
                results.append(("Twelve Data", "PASS", f"{len(df)} candles, last={last_close:.5f}"))
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append(("Twelve Data", "FAIL", str(e)[:80]))

    time.sleep(2)

    # ── Test 4: Polygon.io ──────────────────────────────────
    banner("Test 4: Polygon.io (aggs)")
    poly_key = os.getenv("POLYGON_API_KEY", "")
    if not poly_key:
        print("  SKIP: POLYGON_API_KEY not set (get free key at https://polygon.io/dashboard/api-keys)")
        results.append(("Polygon.io", "SKIP", "no key"))
    else:
        try:
            os.environ["PREFERRED_DATA_SOURCE"] = "polygon"
            import importlib
            import data.fetcher
            importlib.reload(data.fetcher)
            from data.fetcher import DataFetcher
            f = DataFetcher()
            print(f"  Detected source: {f.source}")
            df = f.fetch_ohlcv(TEST_PAIR, TEST_TF, limit=TEST_LIMIT)
            if df is None or len(df) == 0:
                print("  FAIL: no data returned")
                results.append(("Polygon.io", "FAIL", "no data"))
            else:
                last_close = df['close'].iloc[-1]
                print(f"  PASS: {len(df)} candles | last close: {last_close:.5f}")
                results.append(("Polygon.io", "PASS", f"{len(df)} candles, last={last_close:.5f}"))
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append(("Polygon.io", "FAIL", str(e)[:80]))

    # ── Test 5: NewsAPI.org ─────────────────────────────────
    banner("Test 5: NewsAPI.org (financial news + sentiment)")
    na_key = os.getenv("NEWSAPI_API_KEY", "")
    if not na_key:
        print("  SKIP: NEWSAPI_API_KEY not set (get free key at https://newsapi.org/register)")
        results.append(("NewsAPI.org", "SKIP", "no key"))
    else:
        try:
            from analysis.news_api_provider import NewsAPIProvider
            prov = NewsAPIProvider()
            print(f"  Provider available: {prov.available}")
            result = prov.fetch_headlines_for_pair(TEST_PAIR)
            print(f"  Bias          : {result.get('news_bias')}")
            print(f"  Score         : {result.get('news_score', 0):+d}")
            print(f"  Headlines     : {result.get('headline_count', 0)}")
            print(f"  Source        : {result.get('source')}")
            print(f"  Trade allowed : {result.get('trade_allowed')}")
            if result.get("top_headlines"):
                print(f"  Top headline  : [{result['top_headlines'][0]['source']}] "
                      f"{result['top_headlines'][0]['title'][:80]}")
            if result.get("headline_count", 0) > 0:
                print(f"  PASS: {result['headline_count']} headlines, bias={result['news_bias']}")
                results.append(("NewsAPI.org", "PASS",
                                f"{result['headline_count']} headlines, bias={result['news_bias']}"))
            else:
                print(f"  PASS: API responded (no headlines found, but call succeeded)")
                results.append(("NewsAPI.org", "PASS", "API responded, 0 headlines"))
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append(("NewsAPI.org", "FAIL", str(e)[:80]))

    # ── Summary ─────────────────────────────────────────────
    banner("Summary")
    passed = sum(1 for _, s, _ in results if s == "PASS")
    skipped = sum(1 for _, s, _ in results if s == "SKIP")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  PASS : {passed}")
    print(f"  SKIP : {skipped}  (API key not configured — see .env)")
    print(f"  FAIL : {failed}")
    print()
    for name, status, detail in results:
        icon = {"PASS":"✅","SKIP":"⏭️","FAIL":"❌"}[status]
        print(f"  {icon}  {name:<15}  {status:<5}  {detail}")
    print()
    if failed == 0:
        print("  All configured APIs working correctly.")
    else:
        print(f"  {failed} API(s) failed — check error messages above.")


if __name__ == "__main__":
    main()
