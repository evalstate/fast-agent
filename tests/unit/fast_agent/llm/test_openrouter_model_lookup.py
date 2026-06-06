from __future__ import annotations

import httpx
import pytest

from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.openrouter_model_lookup import (
    OpenRouterModelLookupResult,
    clear_openrouter_model_cache,
    list_openrouter_model_specs_sync,
    lookup_openrouter_models,
    register_runtime_openrouter_models,
)
from fast_agent.llm.provider_types import Provider


def _make_result() -> OpenRouterModelLookupResult:
    return OpenRouterModelLookupResult.model_validate(
        {
            "models": [
                {
                    "id": "google/gemini-2.5-pro",
                    "context_length": 1_048_576,
                    "architecture": {"input_modalities": [" text ", "IMAGE"]},
                    "top_provider": {
                        "context_length": 1_048_576,
                        "max_completion_tokens": 65_536,
                    },
                    "supported_parameters": [" STRUCTURED_OUTPUTS "],
                },
                {
                    "id": "openai/gpt-4.1-mini",
                    "context_length": 128_000,
                    "architecture": {"input_modalities": ["text"]},
                    "top_provider": {
                        "context_length": 128_000,
                        "max_completion_tokens": 16_384,
                    },
                    "supported_parameters": [" RESPONSE_FORMAT "],
                },
            ]
        }
    )


class _FakeAsyncClient:
    responses: list[httpx.Response] = []
    calls: list[tuple[str, dict[str, str]]] = []

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(self, url: str, *, headers: dict[str, str]) -> httpx.Response:
        self.calls.append((url, headers))
        return self.responses[len(self.calls) - 1]


def _openrouter_response(status_code: int, payload: object) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("GET", "https://openrouter.example/models/user"),
    )


@pytest.mark.asyncio
async def test_lookup_openrouter_models_caches_successful_response(monkeypatch) -> None:
    clear_openrouter_model_cache()
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = [
        _openrouter_response(
            200,
            {
                "data": [
                    {
                        "id": "openai/gpt-4.1-mini",
                        "architecture": {"input_modalities": ["text"]},
                    }
                ]
            },
        )
    ]
    monkeypatch.setattr(
        "fast_agent.llm.openrouter_model_lookup.httpx.AsyncClient",
        _FakeAsyncClient,
    )

    first = await lookup_openrouter_models("or-test", "https://openrouter.example/")
    second = await lookup_openrouter_models("or-test", "https://openrouter.example")

    assert [model.id for model in first.models] == ["openai/gpt-4.1-mini"]
    assert second == first
    assert len(_FakeAsyncClient.calls) == 1
    assert _FakeAsyncClient.calls[0] == (
        "https://openrouter.example/models/user",
        {"Authorization": "Bearer or-test"},
    )
    clear_openrouter_model_cache()


@pytest.mark.asyncio
async def test_lookup_openrouter_models_does_not_cache_auth_errors(monkeypatch) -> None:
    clear_openrouter_model_cache()
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = [
        _openrouter_response(401, {"error": "bad key"}),
        _openrouter_response(200, {"data": []}),
    ]
    monkeypatch.setattr(
        "fast_agent.llm.openrouter_model_lookup.httpx.AsyncClient",
        _FakeAsyncClient,
    )

    rejected = await lookup_openrouter_models("or-test", "https://openrouter.example")
    refreshed = await lookup_openrouter_models("or-test", "https://openrouter.example")

    assert rejected.error == "OpenRouter API key rejected while listing available models"
    assert refreshed.error is None
    assert len(_FakeAsyncClient.calls) == 2
    clear_openrouter_model_cache()


def test_list_openrouter_model_specs_registers_runtime_metadata(monkeypatch) -> None:
    ModelDatabase.clear_runtime_model_params(Provider.OPENROUTER)

    def _stub_lookup(*args, **kwargs):
        return _make_result()

    monkeypatch.setattr(
        "fast_agent.llm.openrouter_model_lookup.lookup_openrouter_models_sync",
        _stub_lookup,
    )

    specs = list_openrouter_model_specs_sync(api_key="or-test")

    assert "openrouter.google/gemini-2.5-pro" in specs
    assert "openrouter.openai/gpt-4.1-mini" in specs

    params = ModelDatabase.get_model_params("openrouter.google/gemini-2.5-pro")
    assert params is not None
    assert params.context_window == 1_048_576
    assert params.max_output_tokens == 65_536
    assert "image/png" in params.tokenizes
    assert params.json_mode == "schema"

    text_params = ModelDatabase.get_model_params("openrouter.openai/gpt-4.1-mini")
    assert text_params is not None
    assert text_params.json_mode == "object"

    ModelDatabase.clear_runtime_model_params(Provider.OPENROUTER)


def test_openrouter_runtime_registration_does_not_override_static_models(monkeypatch) -> None:
    ModelDatabase.clear_runtime_model_params(Provider.OPENROUTER)
    lookup_result = OpenRouterModelLookupResult.model_validate(
        {
            "models": [
                {
                    "id": "moonshotai/kimi-k2",
                    "context_length": 999_999,
                    "architecture": {"input_modalities": ["text"]},
                    "top_provider": {
                        "context_length": 999_999,
                        "max_completion_tokens": 999_999,
                    },
                    "supported_parameters": ["structured_outputs"],
                }
            ]
        }
    )

    monkeypatch.setattr(
        "fast_agent.llm.openrouter_model_lookup.lookup_openrouter_models_sync",
        lambda *args, **kwargs: lookup_result,
    )

    _ = list_openrouter_model_specs_sync(api_key="or-test")

    # Static metadata should remain unchanged for known models.
    assert ModelDatabase.get_max_output_tokens("moonshotai/kimi-k2") == 16384
    assert ModelDatabase.get_default_provider("moonshotai/kimi-k2") == Provider.HUGGINGFACE

    ModelDatabase.clear_runtime_model_params(Provider.OPENROUTER)


def test_openrouter_runtime_registration_ignores_boolean_limit_metadata() -> None:
    ModelDatabase.clear_runtime_model_params(Provider.OPENROUTER)
    lookup_result = OpenRouterModelLookupResult.model_validate(
        {
            "models": [
                {
                    "id": "example/bool-limits",
                    "context_length": 64_000,
                    "architecture": {"input_modalities": ["text"]},
                    "top_provider": {
                        "context_length": True,
                        "max_completion_tokens": False,
                    },
                }
            ]
        }
    )

    try:
        assert register_runtime_openrouter_models(lookup_result) == 1
        params = ModelDatabase.get_model_params("openrouter.example/bool-limits")
        assert params is not None
        assert params.context_window == 64_000
        assert params.max_output_tokens == 16_384
    finally:
        ModelDatabase.clear_runtime_model_params(Provider.OPENROUTER)
