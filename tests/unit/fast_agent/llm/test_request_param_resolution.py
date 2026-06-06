from __future__ import annotations

from types import SimpleNamespace

import pytest

from fast_agent.llm.request_param_resolution import (
    get_provider_config,
    initialize_base_default_params,
    resolve_config_default_model,
)


class _BrokenProviderSectionConfig:
    @property
    def openai(self) -> object:
        raise AttributeError("provider section failed")


class _BrokenDefaultModelConfig:
    openai = None

    @property
    def anthropic(self) -> object:
        return SimpleNamespace(default_model="claude-sonnet-4-6")


class _BrokenDefaultModel:
    @property
    def default_model(self) -> str:
        raise AttributeError("default model failed")


def test_get_provider_config_skips_missing_sections() -> None:
    provider_config = get_provider_config(
        context_config=SimpleNamespace(anthropic=SimpleNamespace(default_model="sonnet")),
        provider_value="openai",
        fallback_sections=("anthropic",),
    )
    assert provider_config is not None
    assert provider_config.default_model == "sonnet"


def test_get_provider_config_does_not_mask_property_attribute_errors() -> None:
    with pytest.raises(AttributeError, match="provider section failed"):
        get_provider_config(
            context_config=_BrokenProviderSectionConfig(),
            provider_value="openai",
        )


def test_resolve_config_default_model_does_not_mask_default_model_property_errors() -> None:
    with pytest.raises(AttributeError, match="default model failed"):
        resolve_config_default_model(
            context_config=SimpleNamespace(openai=_BrokenDefaultModel()),
            provider_value="openai",
        )


def test_resolve_config_default_model_uses_fallback_after_null_section() -> None:
    assert (
        resolve_config_default_model(
            context_config=_BrokenDefaultModelConfig(),
            provider_value="openai",
            fallback_sections=("anthropic",),
        )
        == "claude-sonnet-4-6"
    )


def test_initialize_base_default_params_normalizes_model_name() -> None:
    params = initialize_base_default_params(
        instruction=None,
        kwargs={"model": " gpt-4.1-mini "},
    )

    assert params.model == "gpt-4.1-mini"


def test_initialize_base_default_params_treats_blank_model_as_missing() -> None:
    params = initialize_base_default_params(
        instruction=None,
        kwargs={"model": "   "},
    )

    assert params.model is None
    assert params.maxTokens == 16384
