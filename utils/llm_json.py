"""Helpers for safely parsing LLM JSON responses.

The trading stack receives free-form text from LLM providers that is often
wrapped in Markdown fences, prefixed with prose, or followed by commentary.
This module extracts the JSON payload and raises a clearer exception when no
usable JSON is present.
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_llm_json(raw: Any) -> Any:
    """Parse JSON emitted by an LLM from a possibly noisy text response.

    The parser is intentionally defensive:
    - accepts already-decoded dict/list values,
    - strips Markdown code fences,
    - removes trailing commas before closing brackets,
    - extracts the first balanced JSON object/array when prose surrounds it,
    - raises a clearer ``json.JSONDecodeError`` when no JSON payload exists.
    """
    if raw is None:
        raise json.JSONDecodeError("LLM response was empty.", "", 0)

    if isinstance(raw, (dict, list, int, float, bool)):
        return raw

    if not isinstance(raw, str):
        raise TypeError(f"parse_llm_json expected str or JSON-compatible value, got {type(raw).__name__}")

    text = raw.strip()
    if not text:
        raise json.JSONDecodeError("LLM response was empty.", "", 0)

    # Normalize common LLM formatting quirks.
    cleaned = text.replace("```json", "").replace("```JSON", "")
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

    for candidate in _candidate_payloads(cleaned):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError(
        "LLM response did not contain a JSON payload. It may have echoed the prompt or failed to produce JSON. Please retry.",
        text,
        0,
    )


def _candidate_payloads(text: str) -> list[str]:
    """Yield candidate JSON strings from the input text."""
    candidates: list[str] = []

    if not text:
        return candidates

    # Try the whole cleaned block first.
    candidates.append(text)

    # Then try to isolate the first balanced object/array if prose surrounds it.
    for start_char in ("{", "["):
        start = text.find(start_char)
        while start != -1:
            end = _find_matching_bracket(text, start)
            if end != -1:
                candidates.append(text[start:end + 1])
                break
            start = text.find(start_char, start + 1)

    # Also try the fenced payload after removing wrappers.
    if text.startswith("{") or text.startswith("["):
        candidates.append(text)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    return unique_candidates


def _find_matching_bracket(text: str, start: int) -> int:
    """Find the matching closing bracket for an opening brace/bracket."""
    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return idx

    return -1


__all__ = ["parse_llm_json"]
