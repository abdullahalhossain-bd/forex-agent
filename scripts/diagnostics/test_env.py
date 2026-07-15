"""
test_env.py — Full Environment Check
=====================================
Covers ALL API keys / services defined in .env:
  - TwelveData, Alpha Vantage, Finnhub, Polygon
  - OpenRouter, Groq, Cerebras, SambaNova, Gemini
  - Telegram Bot
  - FRED (Federal Reserve)
  - NewsAPI
  - Tradermade, Trading Economics
  - OANDA, Myfxbook
  - Hugging Face
  - MT5 Demo (connectivity ping)

Run:
    pip install requests python-dotenv
    python test_env.py
"""

import os
import json
import time
import socket
import requests
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env file loaded via python-dotenv\n")
except ImportError:
    print("⚠️  python-dotenv not installed. Install: pip install python-dotenv")
    print("    Continuing with system environment variables...\n")

TIMEOUT = 10  # seconds per request
results = []

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def ok(label, detail=""):
    msg = f"✅ {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    results.append(("OK", label, detail))

def fail(label, detail=""):
    msg = f"❌ {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    results.append(("FAIL", label, detail))

def warn(label, detail=""):
    msg = f"⚠️  {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    results.append(("WARN", label, detail))

def skip(label, detail="Key not set in .env"):
    msg = f"⏭️  {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    results.append(("SKIP", label, detail))

def safe_get(url, headers=None, timeout=TIMEOUT):
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        return r
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None

def safe_post(url, headers=None, json_body=None, timeout=TIMEOUT):
    try:
        r = requests.post(url, headers=headers or {}, json=json_body or {}, timeout=timeout)
        return r
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  FULL ENVIRONMENT CHECK  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
print()

# ─────────────────────────────────────────────────────────────
# Safety / mode flags (informational — no API call needed)
# ─────────────────────────────────────────────────────────────
print("── Bot Configuration ─────────────────────────────────")
ABSOLUTE_SAFETY  = os.getenv("ABSOLUTE_SAFETY", "").lower()
TEST_MODE        = os.getenv("TEST_MODE", "").lower()
TRADING_MODE     = os.getenv("TRADING_MODE", "")
SIMULATION_MODE  = os.getenv("SIMULATION_MODE", "").lower()
EXECUTION_MODE   = os.getenv("EXECUTION_MODE", "")
APPROVAL_MODE    = os.getenv("APPROVAL_MODE", "")
LOOP_INTERVAL    = os.getenv("LOOP_INTERVAL_SEC", "")
MAX_LOT          = os.getenv("MAX_LOT", "")
DAILY_LOSS_LIMIT = os.getenv("DAILY_LOSS_LIMIT_PCT", "")

print(f"   ABSOLUTE_SAFETY      = {ABSOLUTE_SAFETY or '(not set)'}")
print(f"   TEST_MODE            = {TEST_MODE or '(not set)'}")
print(f"   TRADING_MODE         = {TRADING_MODE or '(not set)'}")
print(f"   SIMULATION_MODE      = {SIMULATION_MODE or '(not set)'}")
print(f"   EXECUTION_MODE       = {EXECUTION_MODE or '(not set)'}")
print(f"   APPROVAL_MODE        = {APPROVAL_MODE or '(not set)'}")
print(f"   LOOP_INTERVAL_SEC    = {LOOP_INTERVAL or '(not set)'}")
print(f"   MAX_LOT              = {MAX_LOT or '(not set)'}")
print(f"   DAILY_LOSS_LIMIT_PCT = {DAILY_LOSS_LIMIT or '(not set)'}")

if ABSOLUTE_SAFETY == "true":
    ok("Safety flags", "ABSOLUTE_SAFETY=true ✔  SIMULATION_MODE=" + SIMULATION_MODE)
else:
    warn("Safety flags", "ABSOLUTE_SAFETY is not 'true' — verify before live trading")
print()

# ─────────────────────────────────────────────────────────────
# 1. TWELVE DATA
# ─────────────────────────────────────────────────────────────
print("── Market Data APIs ──────────────────────────────────")
key = os.getenv("TWELVE_DATA_API_KEY", "")
if not key:
    skip("TwelveData")
else:
    url = f"https://api.twelvedata.com/time_series?symbol=EUR/USD&interval=1min&outputsize=1&apikey={key}"
    r = safe_get(url)
    if r is None:
        fail("TwelveData", "Connection error")
    else:
        data = r.json()
        if data.get("status") == "error":
            fail("TwelveData", data.get("message", "Unknown error"))
        elif "values" in data:
            latest = data["values"][0]
            ok("TwelveData", f"EUR/USD close={latest.get('close')}  (interval=1min)")
        else:
            warn("TwelveData", f"Unexpected response: {str(data)[:120]}")

# ─────────────────────────────────────────────────────────────
# 2. ALPHA VANTAGE
# ─────────────────────────────────────────────────────────────
key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
if not key:
    skip("Alpha Vantage")
else:
    url = (f"https://www.alphavantage.co/query"
           f"?function=CURRENCY_EXCHANGE_RATE&from_currency=EUR&to_currency=USD&apikey={key}")
    r = safe_get(url)
    if r is None:
        fail("Alpha Vantage", "Connection error")
    else:
        data = r.json()
        if "Realtime Currency Exchange Rate" in data:
            rate = data["Realtime Currency Exchange Rate"].get("5. Exchange Rate", "?")
            ok("Alpha Vantage", f"EUR/USD rate={rate}")
        elif "Note" in data:
            warn("Alpha Vantage", "Rate limit hit (25 req/day free tier exceeded)")
        elif "Information" in data:
            warn("Alpha Vantage", data["Information"][:120])
        else:
            fail("Alpha Vantage", str(data)[:120])

# ─────────────────────────────────────────────────────────────
# 3. FINNHUB
# ─────────────────────────────────────────────────────────────
key = os.getenv("FINNHUB_API_KEY", "")
if not key:
    skip("Finnhub")
else:
    url = f"https://finnhub.io/api/v1/forex/rates?base=USD&token={key}"
    r = safe_get(url)
    if r is None:
        fail("Finnhub", "Connection error")
    else:
        data = r.json()
        if "quote" in data or "base" in data:
            ok("Finnhub", f"Forex rates received (base={data.get('base', '?')})")
        elif data.get("error"):
            fail("Finnhub", data["error"])
        else:
            # Stock ping fallback (free tier)
            url2 = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={key}"
            r2 = safe_get(url2)
            if r2 and r2.json().get("c"):
                warn("Finnhub", "Key valid (stocks OK) but forex candles are premium-only on free tier")
            else:
                fail("Finnhub", str(data)[:120])

# ─────────────────────────────────────────────────────────────
# 4. POLYGON
# ─────────────────────────────────────────────────────────────
key = os.getenv("POLYGON_API_KEY", "")
if not key:
    skip("Polygon.io")
else:
    url = f"https://api.polygon.io/v2/aggs/ticker/C:EURUSD/prev?adjusted=true&apiKey={key}"
    r = safe_get(url)
    if r is None:
        fail("Polygon.io", "Connection error")
    else:
        data = r.json()
        if data.get("status") in ("OK", "ok") and data.get("resultsCount", 0) > 0:
            c = data["results"][0].get("c", "?")
            ok("Polygon.io", f"EUR/USD prev-day close={c}")
        elif data.get("status") == "NOT_AUTHORIZED":
            warn("Polygon.io", "Key valid but free tier = end-of-day only, no intraday")
        else:
            fail("Polygon.io", data.get("error", str(data)[:120]))

# ─────────────────────────────────────────────────────────────
# 5. TRADERMADE
# ─────────────────────────────────────────────────────────────
key = os.getenv("TRADERMADE_API_KEY", "")
if not key:
    skip("Tradermade", "Key not set (replaced by Trading Economics + Investing.com RSS in Day 95)")
else:
    url = f"https://marketdata.tradermade.com/api/v1/live?currency=EURUSD&api_key={key}"
    r = safe_get(url)
    if r is None:
        fail("Tradermade", "Connection error")
    else:
        data = r.json()
        if data.get("quotes"):
            q = data["quotes"][0]
            ok("Tradermade", f"EUR/USD mid={q.get('mid', '?')}")
        elif data.get("error"):
            fail("Tradermade", str(data.get("error"))[:120])
        else:
            warn("Tradermade", f"Unexpected: {str(data)[:120]}")

# ─────────────────────────────────────────────────────────────
# 6. TRADING ECONOMICS
# ─────────────────────────────────────────────────────────────
key = os.getenv("TRADINGECONOMICS_API_KEY", "")
if not key:
    skip("Trading Economics", "Key not set (fall back to Investing.com RSS + FF scraper)")
else:
    url = f"https://api.tradingeconomics.com/calendar?c={key}&f=json"
    r = safe_get(url)
    if r is None:
        fail("Trading Economics", "Connection error")
    elif r.status_code == 200:
        try:
            data = r.json()
            count = len(data) if isinstance(data, list) else "?"
            ok("Trading Economics", f"Economic calendar: {count} events returned")
        except Exception:
            ok("Trading Economics", f"HTTP 200 (non-JSON response: {r.text[:60]})")
    elif r.status_code in (401, 403):
        fail("Trading Economics", f"HTTP {r.status_code} — Key invalid")
    else:
        fail("Trading Economics", f"HTTP {r.status_code}: {r.text[:100]}")

print()
print("── LLM / AI APIs ─────────────────────────────────────")

# ─────────────────────────────────────────────────────────────
# 7. GROQ (multi-key)
# ─────────────────────────────────────────────────────────────
groq_keys = []
for suffix in [""] + [f"_{i}" for i in range(1, 10)]:
    k = os.getenv(f"GROQ_API_KEY{suffix}", "")
    if k:
        groq_keys.append((f"GROQ_API_KEY{suffix}", k))

if not groq_keys:
    skip("Groq", "All GROQ_API_KEY_* commented out in .env")
else:
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    for env_name, key in groq_keys:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}
        r = safe_post(url, headers=headers, json_body=body)
        if r is None:
            fail(f"Groq ({env_name})", "Connection error")
        elif r.status_code == 200:
            ok(f"Groq ({env_name})", f"model={model}  → Active")
        elif r.status_code in (401, 403):
            fail(f"Groq ({env_name})", f"HTTP {r.status_code} — Key invalid/expired")
        elif r.status_code == 429:
            warn(f"Groq ({env_name})", "Rate limited (key exists but TPD/RPM exceeded)")
        else:
            fail(f"Groq ({env_name})", f"HTTP {r.status_code}: {r.text[:100]}")
        time.sleep(0.5)

# ─────────────────────────────────────────────────────────────
# 8. CEREBRAS
# ─────────────────────────────────────────────────────────────
key = os.getenv("CEREBRAS_API_KEY", "")
if not key:
    skip("Cerebras")
else:
    # Use CEREBRAS_MODEL from .env (llama3.1-8b-instruct), not hardcoded llama3.1-8b
    model = os.getenv("CEREBRAS_MODEL", "llama3.1-8b-instruct")
    base_url = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}
    r = safe_post(url, headers=headers, json_body=body)
    if r is None:
        fail("Cerebras", "Connection error (likely Cloudflare bot-filter blocking VPS — known issue)")
    elif r.status_code == 200:
        ok("Cerebras", f"model={model}  → Active")
    elif r.status_code == 403:
        warn("Cerebras", "403 Cloudflare block — key likely valid but VPS IP is filtered (known issue)")
    elif r.status_code == 401:
        fail("Cerebras", "401 — Key invalid")
    elif r.status_code == 404:
        warn("Cerebras", f"404 — model '{model}' not found. Check https://inference-docs.cerebras.ai for current model names")
    else:
        fail("Cerebras", f"HTTP {r.status_code}: {r.text[:100]}")

# ─────────────────────────────────────────────────────────────
# 9. SAMBANOVA
# ─────────────────────────────────────────────────────────────
key = os.getenv("SAMBANOVA_API_KEY", "")
if not key:
    skip("SambaNova")
else:
    model = os.getenv("SAMBANOVA_MODEL", "DeepSeek-V3")
    base_url = os.getenv("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1")
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}
    r = safe_post(url, headers=headers, json_body=body)
    if r is None:
        fail("SambaNova", "Connection error")
    elif r.status_code == 200:
        ok("SambaNova", f"model={model}  → Active")
    elif r.status_code == 410:
        warn("SambaNova", (
            f"410 Gone — model '{model}' deprecated. "
            "Check https://docs.sambanova.ai/cloud/docs/get-started/supported-models "
            "for current free-tier model names and update SAMBANOVA_MODEL in .env"
        ))
        # Try to list models as a hint
        r2 = safe_get(f"{base_url}/models", headers=headers)
        if r2 and r2.status_code == 200:
            try:
                models = [m["id"] for m in r2.json().get("data", [])][:5]
                if models:
                    print(f"     ↳ Available models (first 5): {', '.join(models)}")
            except Exception:
                pass
    elif r.status_code in (401, 403):
        fail("SambaNova", f"HTTP {r.status_code} — Key invalid")
    else:
        fail("SambaNova", f"HTTP {r.status_code}: {r.text[:100]}")

# ─────────────────────────────────────────────────────────────
# 10. OPENROUTER
# ─────────────────────────────────────────────────────────────
key = os.getenv("OPENROUTER_API_KEY", "")
if not key:
    skip("OpenRouter")
else:
    # Key validity + credit check
    r = safe_get("https://openrouter.ai/api/v1/auth/key",
                 headers={"Authorization": f"Bearer {key}"})
    if r is None:
        fail("OpenRouter", "Connection error")
    elif r.status_code == 200 and "data" in r.json():
        info = r.json()["data"]
        limit = info.get("limit", "unlimited")
        used  = info.get("usage", 0)
        ok("OpenRouter", f"Key valid  |  usage=${used:.4f}  limit=${limit}")

        # Test primary model
        model = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        url2 = f"{base_url}/chat/completions"
        headers2 = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/your-bot", "X-Title": "ForexBot"}
        body2 = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}
        r2 = safe_post(url2, headers=headers2, json_body=body2)
        if r2 and r2.status_code == 200:
            ok("OpenRouter model test", f"model={model}  → Active")
        elif r2 and r2.status_code == 429:
            warn("OpenRouter model test", f"model={model}  → Rate limited")
        elif r2:
            warn("OpenRouter model test", f"model={model}  → HTTP {r2.status_code}: {r2.text[:80]}")

        # Test fallback models
        for fb_env in ("OPENROUTER_MODEL_FALLBACK_1", "OPENROUTER_MODEL_FALLBACK_2"):
            fb_model = os.getenv(fb_env, "")
            if not fb_model:
                continue
            body3 = {"model": fb_model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}
            r3 = safe_post(url2, headers=headers2, json_body=body3)
            if r3 and r3.status_code == 200:
                ok(f"OpenRouter fallback ({fb_env})", f"model={fb_model}  → Active")
            elif r3 and r3.status_code == 429:
                warn(f"OpenRouter fallback ({fb_env})", f"model={fb_model}  → Rate limited")
            elif r3:
                warn(f"OpenRouter fallback ({fb_env})", f"model={fb_model}  → HTTP {r3.status_code}")
            time.sleep(0.3)
    else:
        fail("OpenRouter", f"HTTP {r.status_code}: {r.text[:100]}")

# ─────────────────────────────────────────────────────────────
# 11. GEMINI (multi-key)
# ─────────────────────────────────────────────────────────────
gemini_keys = []
k0 = os.getenv("GEMINI_API_KEY", "")
if k0:
    gemini_keys.append(("GEMINI_API_KEY", k0))
for i in range(1, 10):
    k = os.getenv(f"GEMINI_API_KEY_{i}", "")
    if k:
        gemini_keys.append((f"GEMINI_API_KEY_{i}", k))

if not gemini_keys:
    skip("Gemini", "All GEMINI_API_KEY_* commented out in .env")
else:
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    for env_name, key in gemini_keys:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={key}")
        body = {"contents": [{"parts": [{"text": "ping"}]}]}
        r = safe_post(url, json_body=body)
        if r is None:
            fail(f"Gemini ({env_name})", "Connection error")
        elif r.status_code == 200:
            ok(f"Gemini ({env_name})", f"model={model}  → Active")
        elif r.status_code == 400:
            fail(f"Gemini ({env_name})", "400 Bad Request — Key format invalid (should be AIza...39 chars)")
        elif r.status_code in (401, 403):
            fail(f"Gemini ({env_name})", f"HTTP {r.status_code} — Key invalid/expired")
        elif r.status_code == 429:
            warn(f"Gemini ({env_name})", "Rate limited (free tier RPM exceeded)")
        else:
            fail(f"Gemini ({env_name})", f"HTTP {r.status_code}: {r.text[:100]}")

print()
print("── Alert / Notification APIs ─────────────────────────")

# ─────────────────────────────────────────────────────────────
# 12. TELEGRAM
# ─────────────────────────────────────────────────────────────
token = os.getenv("TELEGRAM_TOKEN", "")
chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
if not token:
    skip("Telegram Bot")
else:
    r = safe_get(f"https://api.telegram.org/bot{token}/getMe")
    if r is None:
        fail("Telegram Bot", "Connection error")
    elif r.status_code == 200 and r.json().get("ok"):
        bot = r.json()["result"]
        ok("Telegram Bot", f"@{bot.get('username')}  (id={bot.get('id')})")
        # Verify chat_id can receive messages (getChat check — no message sent)
        if chat_id:
            r2 = safe_get(f"https://api.telegram.org/bot{token}/getChat?chat_id={chat_id}")
            if r2 and r2.status_code == 200 and r2.json().get("ok"):
                chat = r2.json().get("result", {})
                ctype = chat.get("type", "?")
                ok("Telegram Chat ID", f"chat_id={chat_id} valid  (type={ctype})")
            else:
                warn("Telegram Chat ID", f"chat_id={chat_id} — bot not in chat or ID wrong")
        else:
            warn("Telegram Chat ID", "TELEGRAM_CHAT_ID not set — alerts will not reach any chat")
    else:
        fail("Telegram Bot", f"HTTP {r.status_code}: {r.text[:100]}")

print()
print("── News / Economic Data APIs ─────────────────────────")

# ─────────────────────────────────────────────────────────────
# 13. NEWSAPI
# ─────────────────────────────────────────────────────────────
key = os.getenv("NEWSAPI_API_KEY", "")
if not key:
    skip("NewsAPI")
else:
    url = f"https://newsapi.org/v2/top-headlines?category=business&pageSize=1&apiKey={key}"
    r = safe_get(url)
    if r is None:
        fail("NewsAPI", "Connection error")
    else:
        data = r.json()
        if data.get("status") == "ok":
            ok("NewsAPI", f"totalResults={data.get('totalResults', '?')}")
        elif data.get("code") == "maximumResultsReached":
            warn("NewsAPI", "Free tier daily limit reached (100 req/day)")
        else:
            fail("NewsAPI", data.get("message", str(data)[:120]))

# ─────────────────────────────────────────────────────────────
# 14. FRED (Federal Reserve)
# ─────────────────────────────────────────────────────────────
key = os.getenv("FRED_API_KEY", "")
if not key:
    skip("FRED")
else:
    url = f"https://api.stlouisfed.org/fred/series?series_id=GDP&api_key={key}&file_type=json"
    r = safe_get(url)
    if r is None:
        fail("FRED", "Connection error")
    else:
        data = r.json()
        if "seriess" in data:
            ok("FRED", f"Series: {data['seriess'][0].get('title', 'GDP')}")
        elif data.get("error_code"):
            fail("FRED", data.get("error_message", str(data)[:120]))
        else:
            fail("FRED", str(data)[:120])

print()
print("── Broker / Sentiment APIs ───────────────────────────")

# ─────────────────────────────────────────────────────────────
# 15. OANDA (practice / live)
# ─────────────────────────────────────────────────────────────
key     = os.getenv("OANDA_API_KEY", "")
acct_id = os.getenv("OANDA_ACCOUNT_ID", "")
use_practice = os.getenv("OANDA_USE_PRACTICE", "true").lower() == "true"
if not key:
    skip("OANDA", "Key not set (replaced by Myfxbook Community Outlook in Day 95)")
else:
    base = "https://api-fxpractice.oanda.com" if use_practice else "https://api-fxtrade.oanda.com"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    r = safe_get(f"{base}/v3/accounts", headers=headers)
    if r is None:
        fail("OANDA", "Connection error")
    elif r.status_code == 200:
        accounts = r.json().get("accounts", [])
        ids = [a.get("id") for a in accounts]
        env_label = "practice" if use_practice else "live"
        ok("OANDA", f"Key valid ({env_label})  |  accounts: {ids}")
        if acct_id and acct_id not in ids:
            warn("OANDA Account ID", f"OANDA_ACCOUNT_ID={acct_id} not found in account list: {ids}")
    elif r.status_code in (401, 403):
        fail("OANDA", f"HTTP {r.status_code} — Key invalid or wrong environment (practice vs live)")
    else:
        fail("OANDA", f"HTTP {r.status_code}: {r.text[:100]}")

# ─────────────────────────────────────────────────────────────
# 16. MYFXBOOK
# ─────────────────────────────────────────────────────────────
email    = os.getenv("MYFXBOOK_EMAIL", "")
password = os.getenv("MYFXBOOK_PASSWORD", "")
if not email or not password:
    # Public community outlook needs no auth — test it anyway
    r = safe_get("https://www.myfxbook.com/api/get-community-outlook.json?session=0")
    if r is None:
        warn("Myfxbook (public)", "Public community outlook endpoint unreachable")
    elif r.status_code == 200:
        try:
            data = r.json()
            if data.get("error") is False or "symbols" in data:
                ok("Myfxbook (public)", "Community outlook endpoint reachable (no auth needed)")
            else:
                warn("Myfxbook (public)", f"Endpoint reachable but returned: {str(data)[:80]}")
        except Exception:
            warn("Myfxbook (public)", f"Non-JSON response: {r.text[:80]}")
    else:
        warn("Myfxbook (public)", f"HTTP {r.status_code}")
    skip("Myfxbook (authenticated)", "MYFXBOOK_EMAIL / MYFXBOOK_PASSWORD not set — using public data only")
else:
    # Attempt authenticated session
    r = safe_get(
        f"https://www.myfxbook.com/api/login.json?email={email}&password={password}"
    )
    if r is None:
        fail("Myfxbook", "Connection error")
    elif r.status_code == 200:
        data = r.json()
        if data.get("error") is False:
            session = data.get("session", "")
            ok("Myfxbook", f"Authenticated  |  session={session[:12]}...")
        else:
            fail("Myfxbook", data.get("message", "Login failed")[:120])
    else:
        fail("Myfxbook", f"HTTP {r.status_code}: {r.text[:100]}")

print()
print("── Hugging Face ──────────────────────────────────────")

# ─────────────────────────────────────────────────────────────
# 17. HUGGING FACE
# ─────────────────────────────────────────────────────────────
key = os.getenv("HF_TOKEN", "")
if not key:
    skip("Hugging Face")
else:
    r = safe_get("https://huggingface.co/api/whoami-v2",
                 headers={"Authorization": f"Bearer {key}"})
    if r is None:
        fail("Hugging Face", "Connection error")
    elif r.status_code == 200:
        data = r.json()
        ok("Hugging Face", f"Logged in as: {data.get('name', data.get('type', '?'))}")
    else:
        fail("Hugging Face", f"HTTP {r.status_code}: {r.text[:100]}")

print()
print("── MT5 / Broker Connectivity ─────────────────────────")

# ─────────────────────────────────────────────────────────────
# 18. MT5 Demo (TCP connectivity ping — no MT5 library needed)
# ─────────────────────────────────────────────────────────────
MT5_SERVER   = os.getenv("MT5_SERVER", "")
MT5_LOGIN    = os.getenv("MT5_LOGIN", "")
SIMULATION   = os.getenv("SIMULATION_MODE", "true").lower()
EXEC_MODE    = os.getenv("EXECUTION_MODE", "")

if not MT5_SERVER:
    skip("MT5 Demo", "MT5_SERVER not set in .env")
elif SIMULATION == "true":
    # Can still do a TCP ping to the MetaQuotes demo gateway
    # MetaQuotes-Demo resolves to demo.mt5.com:443 (WebAPI)
    MT5_HOST = "demo.mt5.com"
    MT5_PORT = 443
    try:
        s = socket.create_connection((MT5_HOST, MT5_PORT), timeout=5)
        s.close()
        ok("MT5 Demo (TCP ping)", (
            f"demo.mt5.com:443 reachable  |  login={MT5_LOGIN}  server={MT5_SERVER}  "
            f"[SIMULATION_MODE=true — no real orders sent]"
        ))
    except (socket.timeout, OSError) as e:
        warn("MT5 Demo (TCP ping)", f"demo.mt5.com:443 unreachable: {e}  (VPS firewall or DNS issue)")
else:
    # Simulation off — remind user MetaTrader5 Python library is Windows-only
    warn("MT5 Demo", (
        f"SIMULATION_MODE={SIMULATION} but MetaTrader5 Python lib is Windows-only. "
        "This check only verifies TCP connectivity to the broker gateway."
    ))
    MT5_HOST = "demo.mt5.com"
    MT5_PORT = 443
    try:
        s = socket.create_connection((MT5_HOST, MT5_PORT), timeout=5)
        s.close()
        ok("MT5 Demo (TCP ping)", f"demo.mt5.com:443 reachable  |  login={MT5_LOGIN}")
    except (socket.timeout, OSError) as e:
        warn("MT5 Demo (TCP ping)", f"demo.mt5.com:443 unreachable: {e}")

# MetaQuotes WebAPI (used by MQL5 EA remote monitoring)
r = safe_get("https://www.mql5.com/en/signals", timeout=5)
if r and r.status_code == 200:
    ok("MQL5.com reachability", "www.mql5.com is reachable from this VPS")
else:
    warn("MQL5.com reachability", "www.mql5.com unreachable — possible VPS routing issue")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  SUMMARY")
print("=" * 60)

counts = {"OK": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
for status, label, detail in results:
    counts[status] += 1

print(f"  ✅ OK    : {counts['OK']}")
print(f"  ❌ FAIL  : {counts['FAIL']}")
print(f"  ⚠️  WARN  : {counts['WARN']}")
print(f"  ⏭️  SKIP  : {counts['SKIP']}")
print()

if counts["FAIL"] > 0:
    print("  Failed services:")
    for status, label, detail in results:
        if status == "FAIL":
            print(f"    • {label}: {detail}")
    print()

if counts["WARN"] > 0:
    print("  Warnings:")
    for status, label, detail in results:
        if status == "WARN":
            print(f"    • {label}: {detail}")
    print()

if counts["SKIP"] > 0:
    print("  Skipped (no key / disabled):")
    for status, label, detail in results:
        if status == "SKIP":
            print(f"    • {label}")
    print()

print("=" * 60)
print("  Check complete.")
print("=" * 60)