import json
from pathlib import Path

import pytest

from utils.llm_json import parse_llm_json
from utils.safe_json import safe_read_json, safe_write_json


def test_parse_llm_json_handles_markdown_fence_and_trailing_text():
    raw = 'Sure, here is the response:\n```json\n{"signal": "BUY", "confidence": 85}\n```\nThanks!'

    parsed = parse_llm_json(raw)

    assert parsed["signal"] == "BUY"
    assert parsed["confidence"] == 85


def test_parse_llm_json_raises_helpful_error_for_non_json_prompt_echo():
    bad_response = "We need to produce JSON with fields: market_story, key_levels"

    with pytest.raises(json.JSONDecodeError) as excinfo:
        parse_llm_json(bad_response)

    message = str(excinfo.value).lower()
    assert "echoed" in message or "retry" in message


def test_safe_json_round_trip(tmp_path: Path):
    target = tmp_path / "state.json"
    payload = {"signal": "WAIT", "confidence": 0}

    safe_write_json(target, payload)
    reloaded = safe_read_json(target)

    assert reloaded == payload
