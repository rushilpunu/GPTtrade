from __future__ import annotations

import json
import sys
import types

import pytest

from agent.llm_clients import (
    FakeLLMClient,
    GeminiClient,
    LLMClientError,
    OpenAIClient,
    create_llm_client,
)


class _FakeOpenAIError(Exception):
    pass


class _FakeOpenAIResponse:
    def __init__(self, content: str) -> None:
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


class _FakeOpenAI:
    def __init__(self, api_key: str, create_fn) -> None:
        self.api_key = api_key
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create_fn)
        )


class _FakeGenerateContentConfig:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeGeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeGenAIClient:
    def __init__(self, api_key: str, capture) -> None:
        self.api_key = api_key
        self.models = types.SimpleNamespace(
            generate_content=lambda **kwargs: capture(kwargs)
        )


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, create_fn) -> None:
    openai_module = types.ModuleType("openai")

    def _factory(api_key: str):
        return _FakeOpenAI(api_key, create_fn)

    openai_module.OpenAI = _factory
    openai_module.OpenAIError = _FakeOpenAIError
    monkeypatch.setitem(sys.modules, "openai", openai_module)


def _install_fake_gemini(monkeypatch: pytest.MonkeyPatch, capture) -> None:
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = lambda api_key: _FakeGenAIClient(api_key, capture)

    types_module = types.ModuleType("google.genai.types")
    types_module.GenerateContentConfig = _FakeGenerateContentConfig

    google_module = types.ModuleType("google")
    google_module.genai = genai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)


def test_fake_client_returns_fixed_response() -> None:
    response = {"status": "ok"}
    client = FakeLLMClient(fixed_response=response)

    result = client.generate_json("system", {"foo": "bar"}, {"type": "object"})

    assert result == response


def test_openai_client_schema_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _create_fn(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _FakeOpenAIResponse("{not json}")
        return _FakeOpenAIResponse('{"status":"ok"}')

    _install_fake_openai(monkeypatch, _create_fn)

    client = OpenAIClient(api_key="key", model="gpt-test", temperature=0.2, max_tokens=256)
    schema = {
        "type": "object",
        "required": ["status"],
        "properties": {"status": {"type": "string"}},
        "additionalProperties": False,
    }
    payload = {"foo": "bar"}

    result = client.generate_json("system prompt", payload, schema)

    assert result == {"status": "ok"}
    assert len(calls) == 2

    request = calls[0]
    assert request["model"] == "gpt-test"
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 256
    assert request["response_format"] == {"type": "json_object"}

    messages = request["messages"]
    assert messages[0] == {"role": "system", "content": "system prompt"}
    expected_payload = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    assert messages[1] == {"role": "user", "content": expected_payload}


def test_gemini_client_schema_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = []

    def _capture(kwargs):
        captured.append(kwargs)
        return _FakeGeminiResponse('{"signal":"buy"}')

    _install_fake_gemini(monkeypatch, _capture)

    client = GeminiClient(api_key="key", model="gemini-test", temperature=0.3, max_tokens=512)
    schema = {
        "type": "object",
        "required": ["signal"],
        "properties": {"signal": {"type": "string"}},
    }
    payload = {"symbol": "AAPL"}

    result = client.generate_json("system prompt", payload, schema)

    assert result == {"signal": "buy"}
    assert len(captured) == 1

    request = captured[0]
    assert request["model"] == "gemini-test"
    expected_payload = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    assert request["contents"] == f"system prompt\n\nUser payload:\n{expected_payload}"

    config = request["config"]
    assert config.temperature == 0.3
    assert config.max_output_tokens == 512
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == schema


def test_create_llm_client_factory() -> None:
    assert isinstance(create_llm_client("openai", "key"), OpenAIClient)
    assert isinstance(create_llm_client("gemini", "key"), GeminiClient)
    fake = create_llm_client("fake", "key", fixed_response={"ok": True})
    assert isinstance(fake, FakeLLMClient)

    with pytest.raises(LLMClientError):
        create_llm_client("unknown", "key")


def test_client_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    def _create_fn(**_kwargs):
        raise _FakeOpenAIError("boom")

    _install_fake_openai(monkeypatch, _create_fn)

    client = OpenAIClient(api_key="key")

    with pytest.raises(LLMClientError) as excinfo:
        client.generate_json("system", {"foo": "bar"}, {"type": "object"})

    assert "OpenAI API request failed" in str(excinfo.value)
