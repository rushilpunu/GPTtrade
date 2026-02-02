"""LLM client abstractions for structured JSON output."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


class LLMClientError(RuntimeError):
    """Raised when an LLM client fails to produce a valid structured response."""


class LLMClient(ABC):
    """Abstract base class for LLM clients that return JSON output."""

    @abstractmethod
    def generate_json(
        self, system_prompt: str, user_payload: Dict[str, Any], schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate JSON that conforms to the provided schema."""
        raise NotImplementedError


@dataclass
class OpenAIClient(LLMClient):
    """OpenAI client implementation for JSON-mode responses."""

    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1024

    def generate_json(
        self, system_prompt: str, user_payload: Dict[str, Any], schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            from openai import OpenAI
            from openai import OpenAIError
        except Exception as exc:
            raise LLMClientError("openai library is not installed or failed to import") from exc

        client = OpenAI(api_key=self.api_key)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    user_payload, ensure_ascii=True, separators=(",", ":")
                ),
            },
        ]

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI response did not include content")
                data = _load_json(content)
                _validate_against_schema(data, schema)
                return data
            except (json.JSONDecodeError, ValueError, SchemaValidationError) as exc:
                last_error = exc
                if attempt < 2:
                    continue
                raise LLMClientError(
                    "Failed to parse or validate OpenAI response"
                ) from exc
            except OpenAIError as exc:
                raise LLMClientError("OpenAI API request failed") from exc
            except Exception as exc:
                raise LLMClientError("Unexpected OpenAI client failure") from exc

        raise LLMClientError("Failed to generate JSON from OpenAI") from last_error


@dataclass
class GeminiClient(LLMClient):
    """Gemini client implementation for JSON responses with schema guidance."""

    api_key: str
    model: str = "gemini-2.0-flash"
    temperature: float = 0.1
    max_tokens: int = 1024

    def generate_json(
        self, system_prompt: str, user_payload: Dict[str, Any], schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise LLMClientError(
                "google-genai library is not installed or failed to import"
            ) from exc

        client = genai.Client(api_key=self.api_key)
        payload = json.dumps(user_payload, ensure_ascii=True, separators=(",", ":"))
        contents = f"{system_prompt}\n\nUser payload:\n{payload}"
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
            response_json_schema=schema,
        )

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                content = getattr(response, "text", None)
                if not content:
                    content = _extract_gemini_text(response)
                data = _load_json(content)
                _validate_against_schema(data, schema)
                return data
            except (json.JSONDecodeError, ValueError, SchemaValidationError) as exc:
                last_error = exc
                if attempt < 2:
                    continue
                raise LLMClientError(
                    "Failed to parse or validate Gemini response"
                ) from exc
            except Exception as exc:
                raise LLMClientError("Gemini API request failed") from exc

        raise LLMClientError("Failed to generate JSON from Gemini") from last_error


@dataclass
class FakeLLMClient(LLMClient):
    """Fake LLM client for tests that returns a fixed response."""

    fixed_response: Dict[str, Any]

    def generate_json(
        self, system_prompt: str, user_payload: Dict[str, Any], schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.fixed_response


def create_llm_client(provider: str, api_key: str, **kwargs: Any) -> LLMClient:
    """Factory for LLM clients."""
    normalized = provider.strip().lower()
    if normalized in {"openai", "oai"}:
        return OpenAIClient(api_key=api_key, **kwargs)
    if normalized in {"gemini", "google", "gcp"}:
        return GeminiClient(api_key=api_key, **kwargs)
    if normalized in {"fake", "test"}:
        fixed_response = kwargs.get("fixed_response", {})
        return FakeLLMClient(fixed_response=fixed_response)
    raise LLMClientError(f"Unsupported LLM provider: {provider}")


class SchemaValidationError(ValueError):
    """Raised when JSON output fails schema validation."""


def _load_json(content: str) -> Dict[str, Any]:
    if content is None:
        raise ValueError("LLM response was empty")
    return json.loads(content)


def _validate_against_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    try:
        import jsonschema
    except Exception:
        _validate_schema_minimal(data, schema)
        return

    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as exc:
        raise SchemaValidationError("JSON did not match schema") from exc


def _validate_schema_minimal(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    if schema.get("type") == "object":
        if not isinstance(data, dict):
            raise SchemaValidationError("Expected object response")
        required = schema.get("required", [])
        for key in required:
            if key not in data:
                raise SchemaValidationError(f"Missing required key: {key}")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate_value_type(key, value, properties[key])
            elif schema.get("additionalProperties") is False:
                raise SchemaValidationError(f"Unexpected key: {key}")


def _validate_value_type(key: str, value: Any, prop_schema: Dict[str, Any]) -> None:
    expected = prop_schema.get("type")
    if expected is None:
        return
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
        "null": type(None),
    }
    expected_types = type_map.get(expected)
    if expected_types and not isinstance(value, expected_types):
        raise SchemaValidationError(
            f"Key '{key}' expected type {expected} but got {type(value).__name__}"
        )
