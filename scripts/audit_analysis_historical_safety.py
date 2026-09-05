"""Forensic audit of AnalysisAgent and its local dependency graph.

The scanner is conservative: wall-clock, network/live-feed, mutable-cache and
current-date primitives are findings requiring manual classification. A hit is
never considered safe merely because a module catches exceptions.
"""
from __future__ import annotations
import ast, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "agents" / "analysis_agent.py"
PACKAGE_ROOTS = {"analysis", "fundamental", "system", "ml", "strategy", "ai", "core"}
TOKENS = {
    "wall_clock": ("datetime.now", "datetime.utcnow", "datetime.today", "date.today", "time.time"),
    "network": ("requests.", "httpx.", "urllib.", "yfinance", "fredapi", "NewsAPI", "newsapi", "aiohttp", "urlopen"),
    "live_feed": ("MT5DataFeed", "get_live_feed", "MetaTrader5", "mt5.", "copy_rates", "symbol_info_tick"),
    "cache": ("lru_cache", "redis", "sqlite", "cache.get", "cache.set", "cached"),
    "current_date": ("CURRENT_DATE", "today", "now()", "current_time"),
}

def module_name(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)

def local_imports(path: Path) -> set[str]:
    try: tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError: return set()
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return {x for x in found if x in PACKAGE_ROOTS}

def resolve_graph() -> list[Path]:
    queue = [ENTRY]; seen = set()
    while queue:
        path = queue.pop()
        if path in seen or not path.exists(): continue
        seen.add(path)
        for root in local_imports(path):
            pkg = ROOT / root
            for child in pkg.rglob("*.py"):
                # Include imported package files; bounded to known project roots.
                if child not in seen and module_name(child).startswith(root + "."):
                    queue.append(child)
    return sorted(seen)

def scan(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(); hits = []
    for category, needles in TOKENS.items():
        for needle in needles:
            for n, line in enumerate(lines, 1):
                if needle in line:
                    hits.append({"category": category, "needle": needle, "line": n, "text": line.strip()[:240]})
    try: ast.parse(text); syntax = "OK"
    except SyntaxError as exc: syntax = f"ERROR: {exc}"
    return {"file": str(path.relative_to(ROOT)), "module": module_name(path),
            "syntax": syntax, "hits": hits}

def main() -> None:
    files = resolve_graph()
    report = [scan(p) for p in files]
    out = ROOT / "backtest" / "results" / "analysis_historical_safety_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    flagged = [r for r in report if r["hits"] or r["syntax"] != "OK"]
    print(f"AnalysisAgent dependency files scanned: {len(report)}")
    print(f"Flagged files requiring review: {len(flagged)}")
    for r in flagged:
        print(f"- {r['file']}: {len(r['hits'])} findings")
    print(f"Report: {out}")

if __name__ == "__main__": main()
