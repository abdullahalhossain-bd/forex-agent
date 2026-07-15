# scripts/mcp_docs_server.py — MCP server exposing project docs to AI agents
# =============================================================================
# Minimal MCP (Model Context Protocol) server that lets an AI coding agent
# query the project's documentation. Tools exposed:
#
#   list_docs()              → list all .md files in repo root + docs/
#   read_doc(path)           → return contents of a doc file
#   search_docs(query)       → simple substring search across all docs
#
# This is a STUB. It implements the MCP-over-stdio protocol just enough to
# be discoverable by clients like Claude Code, Cursor, etc. For full MCP
# semantics, use the official `mcp` Python SDK (`pip install mcp`).
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_DIRS = (PROJECT_ROOT, PROJECT_ROOT / "docs")


def list_docs() -> list[str]:
    out = []
    for d in DOC_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.md")):
            out.append(str(p.relative_to(PROJECT_ROOT)))
    return out


def read_doc(rel_path: str) -> str:
    p = PROJECT_ROOT / rel_path
    if not p.exists() or not p.is_file():
        return f"NOT FOUND: {rel_path}"
    if p.suffix != ".md":
        return f"NOT A MARKDOWN FILE: {rel_path}"
    return p.read_text(encoding="utf-8")


def search_docs(query: str) -> list[dict]:
    q = query.lower()
    hits = []
    for d in DOC_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.md")):
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            if q in text.lower():
                # find first matching line for context
                for line in text.splitlines():
                    if q in line.lower():
                        hits.append({
                            "file": str(p.relative_to(PROJECT_ROOT)),
                            "context": line.strip()[:200],
                        })
                        break
    return hits


# ── Minimal MCP-over-stdio protocol ─────────────────────────────────────────
# Responds to JSON-RPC 2.0 messages on stdin. Implements just enough to be
# useful: tools/list and tools/call. For a full MCP server, use the official
# `mcp` SDK.

def _handle(request: dict) -> dict:
    method = request.get("method")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "forex-ai-docs", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "list_docs",
                        "description": "List all markdown docs in the Forex AI repo.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "read_doc",
                        "description": "Read a markdown doc by relative path.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                    {
                        "name": "search_docs",
                        "description": "Substring search across all docs.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                ]
            },
        }

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "list_docs":
            result = list_docs()
        elif name == "read_doc":
            result = read_doc(args.get("path", ""))
        elif name == "search_docs":
            result = search_docs(args.get("query", ""))
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {name}"},
            }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = _handle(request)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
