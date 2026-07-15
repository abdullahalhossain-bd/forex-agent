# utils/llm_json.py  —  Robust JSON parsing for LLM responses
# ============================================================
# Day 99+ FIX (Issue #1): LLM providers (Groq, Cerebras, OpenRouter,
# SambaNova, Gemini) often return JSON wrapped in Markdown code fences
# (```json ... ```), or with extra prose before/after the JSON object,
# or with trailing commas / smart quotes that json.loads rejects.
# Calling json.loads() directly on such a response raises
# JSONDecodeError and crashes the trading cycle.
#
# This module exposes a single function — `parse_llm_json(raw)` — that
# pre-processes the response and extracts the first complete JSON
# object, handling:
#   - Markdown ```json ... ``` code blocks
#   - Trailing prose after the JSON object
#   - Leading prose before the JSON object
#   - Trailing commas (common LLM mistake; tolerated)
#   - Smart quotes (" " ' ') that break strict JSON
#   - Empty / None input (raises clean JSONDecodeError)
#
# It is used by agents/master_analyst.py and ai/ai_analyst.py to
# guarantee that a "dirty" LLM response never crashes the pipeline.
# ============================================================

from __future__ import annotations

import json
import re
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("llm_json")


# Pre-compiled regexes (LLM responses are frequent; compile once).

# Match ```json ... ``` or ``` ... ``` fenced code blocks.
# We capture the inside so we can extract just the JSON content.
_FENCE_PATTERN = re.compile(
    r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)

# Match trailing commas inside objects/arrays (e.g. {"a": 1,} or [1, 2,])
# which are valid in JS but not in strict JSON. The LLM emits these often.
_TRAILING_COMMA_PATTERN = re.compile(
    r",\s*([}\]])"
)

# Smart quote → straight quote replacement map. LLMs occasionally emit
# smart quotes around keys or string values, especially when the prompt
# contains them. json.loads rejects them outright.
_SMART_QUOTE_MAP = {
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u2013": "-",  # en dash (sometimes in numbers)
    "\u2014": "-",  # em dash
}


def _strip_markdown_fences(text: str) -> str:
    """If the text is (or contains) a ```json ... ``` fenced block,
    return just the content inside the fence. Otherwise return the
    original text unchanged.
    """
    match = _FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text


def _normalize_smart_quotes(text: str) -> str:
    """Replace smart quotes / dashes with their ASCII equivalents so
    json.loads() doesn't choke on them.
    """
    if not text:
        return text
    for smart, plain in _SMART_QUOTE_MAP.items():
        if smart in text:
            text = text.replace(smart, plain)
    return text


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] (common LLM mistake)."""
    return _TRAILING_COMMA_PATTERN.sub(r"\1", text)


def parse_llm_json(raw: Optional[str]) -> Any:
    """Parse a JSON object out of an LLM response string.

    Tries (in order):
      1. Strip Markdown ```json ... ``` fences if present.
      2. Normalize smart quotes → ASCII.
      3. Strip trailing commas.
      4. Try json.loads on the cleaned text.
      5. If that fails, find the first '{' and use
         JSONDecoder().raw_decode() to extract one complete object
         (this naturally stops at the matching '}', ignoring any
         trailing prose).

    Args:
        raw: the raw LLM response string (may be None or empty).

    Returns:
        The parsed JSON value (usually a dict).

    Raises:
        json.JSONDecodeError: if no JSON object can be extracted
            (including when `raw` is None or empty).
    """
    if raw is None:
        log.error("[llm_json] received None input")
        raise json.JSONDecodeError("LLM response is None", "", 0)

    text = raw.strip()
    if not text:
        log.error("[llm_json] received empty input")
        raise json.JSONDecodeError("Empty LLM response", "", 0)

    # ── Step 1: strip markdown fences ─────────────────────────
    text = _strip_markdown_fences(text)

    # ── Step 2: normalize smart quotes ────────────────────────
    text = _normalize_smart_quotes(text)

    # ── Step 3: strip trailing commas ─────────────────────────
    text = _strip_trailing_commas(text)

    text = text.strip()

    # ── Step 4: try strict json.loads first ───────────────────
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass  # fall through to step 5

    # ── Step 5: find first '{' and decode one complete object ─
    start = text.find("{")
    if start < 0:
        log.error(
            f"[llm_json] no JSON object found in response "
            f"(len={len(text)}, preview={text[:120]!r})"
        )
        raise json.JSONDecodeError(
            "No JSON object ('{') found in LLM response", text, 0
        )

    try:
        data, end_idx = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        log.error(
            f"[llm_json] could not decode JSON object starting at "
            f"offset {start}: {e}. Preview: {text[start:start+200]!r}"
        )
        raise

    # Log if there was trailing prose (informational only — common
    # and not an error in itself).
    trailing = text[start + end_idx:].strip()
    if trailing:
        log.debug(
            f"[llm_json] ignored {len(trailing)} chars of trailing "
            f"prose after JSON object"
        )

    return data


def parse_llm_json_or(raw: Optional[str], default: Any) -> Any:
    """Like parse_llm_json, but returns `default` instead of raising
    on any parse failure. Use this in code paths that want to silently
    fall back (e.g. "if the LLM gave us bad JSON, just use the rule
    engine's signal instead of crashing the cycle").
    """
    try:
        return parse_llm_json(raw)
    except json.JSONDecodeError as e:
        log.warning(
            f"[llm_json] parse failed, returning default: {e.msg}"
        )
        return default
