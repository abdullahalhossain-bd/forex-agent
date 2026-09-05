"""Static forensic audit for AnalysisAgent and its analysis/ dependency graph.

Usage: python scripts/audit_analysis_historical_safety.py

The audit is intentionally conservative: any wall-clock/network/cache/current-
state primitive is flagged for manual review. It never declares a module safe
just because an exception handler exists.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
TOKENS = {
    "wall_clock": ("datetime.now", "datetime.utcnow", "datetime.today", "date.today", "time.time"),
    "network": ("requests.", "httpx.", "urllib.", "yfinance", "fredapi", "NewsAPI", "newsapi", "aiohttp"),
    "live_feed": ("MT5DataFeed", "get_live_feed", "mt5.", "MetaTrader5"),
    "cache": ("cache", "lru_cache", "redis", "sqlite"),
    "current_date": ("CURRENT_DATE", "today", "now()"),
}


def scan(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    hits = []
    for category, needles in TOKENS.items():
        for needle in needles:
            for n, line in enumerate(text.splitlines(), 1):
                if needle in line:
                    hits.append({"category": category, "needle": needle, "line": n, "text": line.strip()[:240]})
    try:
        ast.parse(text)
        syntax = "OK"
    except SyntaxError as exc:
        syntax = f"ERROR: {exc}"
    return {"file": str(path.relative_to(ROOT)), "syntax": syntax, "hits": hits}


def main() -> None:
    files = sorted(ANALYSIS.glob("*.py"))
    report = [scan(p) for p in files]
    out = ROOT / "backtest" / "results" / "analysis_historical_safety_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    flagged = [r for r in report if r["hits"] or r["syntax"] != "OK"]
    print(f"Analysis modules scanned: {len(report)}")
    print(f"Flagged modules: {len(flagged)}")
    for r in flagged:
        print(f"- {r['file']}: {len(r['hits'])} findings")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
