# scripts/mcp_journal_server.py — MCP server exposing the Trading-as-Git journal
# =============================================================================
# Lets an AI coding agent inspect (and reject) trade intentions in the
# Forex AI journal without giving it PUSH permission. Tools exposed:
#
#   journal_status()                 → counts + recent items per state
#   journal_show(id)                 → full intention JSON
#   journal_reject(id, reason)       → reject a staged/committed intention
#
# PUSH is intentionally NOT exposed — that requires the orchestrator's
# execution_router, which an MCP agent should not invoke directly.
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running both from repo root and from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.trading_as_git import TradingJournal  # noqa: E402


def journal_status() -> dict:
    j = TradingJournal()
    return {
        "staged":    [i.to_dict() for i in j.list_staged()],
        "committed": [i.to_dict() for i in j.list_committed()],
        "pushed":    [i.to_dict() for i in j.list_pushed(limit=10)],
        "rejected":  [i.to_dict() for i in j.list_rejected(limit=10)],
    }


def journal_show(intention_id: str) -> dict:
    j = TradingJournal()
    i = j.get(intention_id)
    return i.to_dict() if i else {"error": "not found", "id": intention_id}


def journal_reject(intention_id: str, reason: str) -> dict:
    j = TradingJournal()
    try:
        i = j.reject(intention_id, reason)
        return {"ok": True, "intention": i.to_dict()}
    except FileNotFoundError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ── Minimal MCP-over-stdio protocol ─────────────────────────────────────────

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
                "serverInfo": {"name": "forex-ai-journal", "version": "0.1.0"},
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
                        "name": "journal_status",
                        "description": "Show Forex AI Trading-as-Git journal status.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "journal_show",
                        "description": "Show a single trade intention by ID.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"id": {"type": "string"}},
                            "required": ["id"],
                        },
                    },
                    {
                        "name": "journal_reject",
                        "description": "Reject a staged or committed trade intention.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["id", "reason"],
                        },
                    },
                ]
            },
        }

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        try:
            if name == "journal_status":
                result = journal_status()
            elif name == "journal_show":
                result = journal_show(args.get("id", ""))
            elif name == "journal_reject":
                result = journal_reject(args.get("id", ""), args.get("reason", ""))
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {name}"},
                }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": f"{type(e).__name__}: {e}"},
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
