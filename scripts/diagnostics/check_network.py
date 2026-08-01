#!/usr/bin/env python3
"""
check_network.py — Network connectivity diagnostic for forex_ai
================================================================

Checks whether your PC can reach the external services that forex_ai
depends on:
  - Telegram Bot API    (api.telegram.org)
  - Groq LLM API        (api.groq.com)
  - Gemini LLM API      (generativelanguage.googleapis.com)
  - yfinance / Yahoo    (query1.finance.yahoo.com)
  - Forex Factory news  (www.forexfactory.com)
  - Alpha Vantage       (www.alphavantage.co)
  - Finnhub             (finnhub.io)

If you see "BLOCKED" for any service, that's why LLM keys get disabled
and Telegram alerts fail.  Common causes:
  - Corporate / ISP firewall blocking the host
  - Geo-blocked region (e.g. Google APIs blocked in some countries)
  - DNS resolver issue (try 8.8.8.8 / 1.1.1.1)
  - Proxy / VPN configuration needed

Usage:
    python check_network.py
"""

from __future__ import annotations

import os
import socket
import ssl
import sys
import time
import urllib.request
from pathlib import Path

# ── Load .env so we can use proxy settings if configured ────────────
try:
    from dotenv import load_dotenv
    for env_path in (Path.cwd() / ".env", Path.home() / ".env"):
        if env_path.exists():
            load_dotenv(str(env_path))
            break
except ImportError:
    pass

# ── Color helpers ───────────────────────────────────────────────────
_IS_TTY = sys.stdout.isatty()
def _c(text: str, color: str) -> str:
    if not _IS_TTY:
        return text
    codes = {"red": "31", "green": "32", "yellow": "33", "cyan": "36", "bold": "1"}
    return f"\033[{codes.get(color, '0')}m{text}\033[0m"


# ── Services to check ──────────────────────────────────────────────
SERVICES = [
    # (label, host, port, path, expect_status)
    # Any 4xx response means "we reached the server, it just rejected
    # our request because we didn't authenticate" — that's a PASS for
    # connectivity purposes.
    ("Telegram API",      "api.telegram.org",                    443, "/bot000:getMe",           (401, 403, 404)),
    ("Groq LLM",          "api.groq.com",                        443, "/openai/v1/models",       (401, 403, 404)),
    ("Gemini LLM",        "generativelanguage.googleapis.com",   443, "/v1beta/models",          (401, 403, 404)),
    ("yfinance / Yahoo",  "query1.finance.yahoo.com",            443, "/v8/finance/chart/AAPL",  (200,)),
    ("Forex Factory",     "www.forexfactory.com",                443, "/calendar",               (200, 403)),
    ("Alpha Vantage",     "www.alphavantage.co",                 443, "/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo", (200,)),
    ("Finnhub",           "finnhub.io",                          443, "/api/v1",                 (401, 403, 404)),
    ("GitHub (for git)",  "github.com",                          443, "/",                        (200, 301)),
    ("PyPI (for pip)",    "pypi.org",                            443, "/",                        (200, 301)),
    ("Google DNS",        "8.8.8.8",                              53, "",                         ()),  # special — TCP test only
]


def _check_dns(host: str) -> bool:
    """Can we resolve the hostname?"""
    try:
        socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)
        return True
    except socket.gaierror:
        return False


def _check_tcp(host: str, port: int, timeout: float = 5.0) -> bool:
    """Can we open a TCP connection to host:port?"""
    try:
        with socket.create_connection((host, port), timeout=timeout) as _:
            return True
    except (socket.timeout, OSError):
        return False


def _check_https(host: str, path: str, timeout: float = 8.0) -> tuple[int | None, str]:
    """Can we complete an HTTPS request? Returns (status_code, error_or_ok)."""
    url = f"https://{host}{path}"
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent": "forex_ai/check_network"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.status, "OK"
    except urllib.error.HTTPError as e:
        # HTTP error means the connection WORKED — server returned a status.
        # That's still "reachable" for our purposes.
        return e.code, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return None, f"URLError: {e.reason}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def check_service(label: str, host: str, port: int, path: str, expect_statuses: tuple) -> dict:
    """Full check: DNS → TCP → HTTPS."""
    result = {
        "label": label,
        "host": host,
        "dns": False,
        "tcp": False,
        "https_status": None,
        "https_msg": "",
        "verdict": "BLOCKED",
    }

    # 1. DNS
    result["dns"] = _check_dns(host)
    if not result["dns"]:
        result["verdict"] = "DNS_FAIL"
        return result

    # Special case — DNS-only host (e.g. 8.8.8.8 has no path)
    if not path:
        result["tcp"] = _check_tcp(host, port)
        result["verdict"] = "OK" if result["tcp"] else "TCP_FAIL"
        return result

    # 2. TCP
    result["tcp"] = _check_tcp(host, port)
    if not result["tcp"]:
        result["verdict"] = "TCP_FAIL"
        return result

    # 3. HTTPS request
    status, msg = _check_https(host, path)
    result["https_status"] = status
    result["https_msg"] = msg
    if status is None:
        result["verdict"] = "HTTPS_FAIL"
    elif expect_statuses and status not in expect_statuses:
        result["verdict"] = f"UNEXPECTED_{status}"
    else:
        result["verdict"] = "OK"

    return result


def main() -> int:
    # ── Banner ─────────────────────────────────────────────────────
    print()
    print(_c("=" * 64, "bold"))
    print(_c("  forex_ai — Network Connectivity Check", "bold"))
    print(_c("=" * 64, "bold"))
    print(f"  Time   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  OS     : {sys.platform}")
    print(f"  Python : {sys.version.split()[0]}")

    # ── Proxy env vars ─────────────────────────────────────────────
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                  "NO_PROXY", "no_proxy", "ALL_PROXY", "all_proxy"]
    proxies_set = [(v, os.environ.get(v, "(none)")) for v in proxy_vars
                   if os.environ.get(v)]
    if proxies_set:
        print(f"  Proxy  : {_c('CONFIGURED', 'cyan')}")
        for name, val in proxies_set:
            print(f"    {name} = {val[:80]}")
    else:
        print(f"  Proxy  : {_c('none (direct connection)', 'cyan')}")

    print()
    print(_c("[ Checking reachability of forex_ai dependencies ]", "bold"))
    print()

    # ── Run checks ─────────────────────────────────────────────────
    print(f"  {'Service':<22} {'DNS':<5} {'TCP':<5} {'HTTPS':<7} {'Verdict'}")
    print(f"  {'-'*22} {'-'*5} {'-'*5} {'-'*7} {'-'*20}")

    all_ok = True
    critical_failures = []
    for label, host, port, path, expect in SERVICES:
        r = check_service(label, host, port, path, expect)

        dns_s = _c("✓", "green") if r["dns"] else _c("✗", "red")
        tcp_s = _c("✓", "green") if r["tcp"] else _c("✗", "red")
        https_s = (
            _c(str(r["https_status"]), "green") if r["https_status"]
            else _c("—", "yellow") if not path
            else _c("✗", "red")
        )

        verdict = r["verdict"]
        if verdict == "OK":
            verdict_s = _c("✓ OK", "green")
        elif verdict in ("DNS_FAIL", "TCP_FAIL", "HTTPS_FAIL"):
            verdict_s = _c(f"✗ {verdict}", "red")
            all_ok = False
            critical_failures.append((label, host, verdict, r["https_msg"]))
        else:
            verdict_s = _c(f"? {verdict}", "yellow")

        print(f"  {label:<22} {dns_s:<5} {tcp_s:<5} {https_s:<7} {verdict_s}")

        if r["https_msg"] and r["https_msg"] != "OK":
            print(f"    {_c('↳', 'gray')} {r['https_msg'][:80]}")

    # ── Summary ────────────────────────────────────────────────────
    print()
    print(_c("=" * 64, "bold"))
    if all_ok:
        print(_c("  ✓ All services reachable. Network is healthy.", "green"))
        print()
        print("  If LLM keys are still disabled, run:")
        print("    python -c \"from core.llm_key_manager import get_llm_key_manager; get_llm_key_manager().reset_keys()\"")
    else:
        print(_c("  ✗ Some services are unreachable.", "red"))
        print()
        print(_c("  Failed services:", "red"))
        for label, host, verdict, msg in critical_failures:
            print(f"    ✗ {label} ({host}) — {verdict}")
            if msg:
                print(f"      {msg[:80]}")
        print()
        print(_c("  Common fixes:", "yellow"))
        print("    1. Check if a VPN / proxy is needed for Groq / Gemini / Telegram")
        print("    2. Try changing DNS to 8.8.8.8 (Google) or 1.1.1.1 (Cloudflare)")
        print("    3. Disable firewall temporarily to test")
        print("    4. For region-blocked services, set HTTPS_PROXY in .env:")
        print("       HTTPS_PROXY=http://your-proxy:port")
        print()
        print("  After fixing the network, reset LLM key state:")
        print("    python -c \"from core.llm_key_manager import get_llm_key_manager; get_llm_key_manager().reset_keys()\"")
    print(_c("=" * 64, "bold"))
    print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
