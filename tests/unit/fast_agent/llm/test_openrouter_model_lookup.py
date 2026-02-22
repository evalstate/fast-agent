from __future__ import annotations

from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.openrouter_model_lookup import (
    OpenRouterModelLookupResult,
    list_openrouter_model_specs_sync,
)
from fast_agent.llm.provider_types import Provider


def _make_result() -> OpenRouterModelLookupResult:
    return OpenRouterModelLookupResult.model_validate(
        {
            "models": [
                {
                    "id": "google/gemini-2.5-pro",
                    "context_length": 1_048_576,
                    "architecture": {"input_modalities": ["text", "image"]},
                    "top_provider": {
                        "context_length": 1_048_576,
                        "max_completion_tokens": 65_536,
                    },
                    "supported_parameters": ["structured_outputs"],
                },
                {
                    "id": "openai/gpt-4.1-mini",
                    "context_length": 128_000,
                    "architecture": {"input_modalities": ["text"]},
                    "top_provider": {
                        "context_length": 128_000,
                        "max_completion_tokens": 16_384,
                    },
                    "supported_parameters": ["response_format"],
                },
            ]
        }
    )


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

    ModelDatabase.clear_runtime_model_params(Provider.OPENROUTER)
