import json
import sys
import types

import pytest

import config
from core import llm_gateway
from core.llm_key_manager import LLMKeyManager


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def test_ollama_api_url_does_not_duplicate_api_path():
    assert llm_gateway.ollama_api_url("https://example.test") == "https://example.test/api/chat"
    assert llm_gateway.ollama_api_url("https://example.test/api") == "https://example.test/api/chat"
    assert llm_gateway.ollama_api_url("https://example.test/api/chat") == "https://example.test/api/chat"


def test_remote_ollama_sends_required_payload_and_parses_content(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"message": {"content": '{"signal":"WAIT","confidence":0}'}, "thinking": "ignore"})

    monkeypatch.setattr(llm_gateway, "urlopen", fake_urlopen)
    monkeypatch.setattr("config.LLM_REMOTE_URL", "https://example.test")
    monkeypatch.setattr("config.LLM_MODEL", "qwen3:14b")
    result = llm_gateway.call_remote_ollama([{"role": "user", "content": "Return JSON"}], timeout=3, retries=0)

    assert json.loads(result)["signal"] == "WAIT"
    assert captured["url"] == "https://example.test/api/chat"
    assert captured["timeout"] == 3
    assert captured["payload"]["model"] == "qwen3:14b"
    assert captured["payload"]["think"] is False
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["options"]["temperature"] == 0


def test_remote_ollama_retries_invalid_json(monkeypatch):
    responses = iter([
        FakeResponse({"message": {"content": "not json"}}),
        FakeResponse({"message": {"content": '{"signal":"WAIT"}'}}),
    ])
    monkeypatch.setattr(llm_gateway, "urlopen", lambda request, timeout: next(responses))
    monkeypatch.setattr(llm_gateway.time, "sleep", lambda _: None)
    assert json.loads(llm_gateway.call_remote_ollama([], timeout=1, retries=1))["signal"] == "WAIT"


def test_remote_ollama_raises_after_invalid_json_retries(monkeypatch):
    monkeypatch.setattr(llm_gateway, "urlopen", lambda request, timeout: FakeResponse({"message": {"content": "bad"}}))
    monkeypatch.setattr(llm_gateway.time, "sleep", lambda _: None)
    with pytest.raises(llm_gateway.LLMGatewayError):
        llm_gateway.call_remote_ollama([], timeout=1, retries=1)


def test_local_backend_disables_legacy_provider_clients(monkeypatch):
    monkeypatch.setattr(config, "LLM_LOCAL", True)
    monkeypatch.setenv("GROQ_API_KEY_1", "test-groq-key")
    monkeypatch.setenv("GEMINI_API_KEY_1", "test-gemini-key")

    fake_groq = types.SimpleNamespace(Groq=lambda **kwargs: object())
    fake_google = types.ModuleType("google")
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = lambda **kwargs: object()
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "groq", fake_groq)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    manager = LLMKeyManager()
    assert manager.get_groq_client() is None
    assert manager.get_gemini_client() is None