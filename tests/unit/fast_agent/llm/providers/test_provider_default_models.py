"""
Testing notes:

- This module owns provider-level fallback behavior when a caller selects a
  provider but omits an explicit model.
- Keep these tests focused on config/env precedence and the request model that
  results from provider defaults.
- Avoid duplicating alias parsing, catalog curation, or capability-table
  assertions here; those belong in the dedicated model_factory,
  model_selection_catalog, and model_database test modules.
"""

import os

from fast_agent.config import (
    AzureSettings,
    DeepSeekSettings,
    HuggingFaceSettings,
    OpenAISettings,
    OpenResponsesSettings,
    OpenRouterSettings,
    Settings,
)
from fast_agent.constants import DEFAULT_MAX_ITERATIONS
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.google.llm_google_native import GoogleNativeLLM
from fast_agent.llm.provider.openai.llm_azure import AzureOpenAILLM
from fast_agent.llm.provider.openai.llm_deepseek import DeepSeekLLM
from fast_agent.llm.provider.openai.llm_generic import GenericLLM
from fast_agent.llm.provider.openai.llm_google_oai import GoogleOaiLLM
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider.openai.llm_openrouter import OpenRouterLLM
from fast_agent.llm.provider.openai.openresponses import OpenResponsesLLM
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider_types import Provider


def test_openai_provider_default_model_used_when_model_missing() -> None:
    settings = Settings(openai=OpenAISettings(default_model="gpt-4.1-mini"))
    llm = OpenAILLM(context=Context(config=settings), model="")

    assert llm.default_request_params.model == "gpt-4.1-mini"


def test_openai_provider_default_model_alias_is_resolved() -> None:
    settings = Settings(
        openai=OpenAISettings(default_model="$system.fast"),
        model_references={"system": {"fast": "gpt-4.1-mini"}},
    )
    llm = OpenAILLM(context=Context(config=settings), model="")

    assert llm.default_request_params.model == "gpt-4.1-mini"


def test_openai_explicit_model_overrides_provider_default() -> None:
    settings = Settings(openai=OpenAISettings(default_model="gpt-4.1-mini"))
    llm = OpenAILLM(context=Context(config=settings), model="gpt-4.1")

    assert llm.default_request_params.model == "gpt-4.1"


def test_provider_defaults_use_global_max_iterations() -> None:
    settings = Settings()
    context = Context(config=settings)

    assert (
        GenericLLM(context=context).default_request_params.max_iterations == DEFAULT_MAX_ITERATIONS
    )
    assert (
        GoogleOaiLLM(context=context).default_request_params.max_iterations
        == DEFAULT_MAX_ITERATIONS
    )
    assert (
        GoogleNativeLLM(context=context).default_request_params.max_iterations
        == DEFAULT_MAX_ITERATIONS
    )


def test_responses_provider_default_model_used_when_model_missing() -> None:
    settings = Settings(responses=OpenAISettings(default_model="gpt-5.1"))
    llm = ResponsesLLM(context=Context(config=settings), model="")

    assert llm.default_request_params.model == "gpt-5.1"


def test_responses_falls_back_to_openai_provider_config_default_model() -> None:
    settings = Settings(openai=OpenAISettings(default_model="gpt-5.1"))
    llm = ResponsesLLM(context=Context(config=settings), model="")

    assert llm.default_request_params.model == "gpt-5.1"


def test_openresponses_provider_default_model_used_when_model_missing() -> None:
    settings = Settings(openresponses=OpenResponsesSettings(default_model="gpt-oss-120b"))
    llm = OpenResponsesLLM(context=Context(config=settings), model="")

    assert llm.default_request_params.model == "gpt-oss-120b"


def test_openresponses_provider_default_base_url_used_when_config_missing() -> None:
    llm = OpenResponsesLLM(context=Context(config=Settings()), model="gpt-oss-120b")

    assert llm._base_url() == "http://localhost:8080/v1"


def test_openresponses_provider_env_base_url_overrides_default() -> None:
    original = os.getenv("OPENRESPONSES_BASE_URL")
    try:
        os.environ["OPENRESPONSES_BASE_URL"] = "http://localhost:9090/v1"
        llm = OpenResponsesLLM(context=Context(config=Settings()), model="gpt-oss-120b")

        assert llm._base_url() == "http://localhost:9090/v1"
    finally:
        if original is None:
            if "OPENRESPONSES_BASE_URL" in os.environ:
                del os.environ["OPENRESPONSES_BASE_URL"]
        else:
            os.environ["OPENRESPONSES_BASE_URL"] = original


def test_openresponses_provider_does_not_inherit_openai_base_url() -> None:
    settings = Settings(openai=OpenAISettings(base_url="https://gateway.example/v1"))
    llm = OpenResponsesLLM(context=Context(config=settings), model="gpt-oss-120b")

    assert llm._base_url() == "http://localhost:8080/v1"


def test_openrouter_provider_default_model_used_when_model_missing() -> None:
    ModelDatabase.clear_runtime_model_params(provider=Provider.OPENROUTER)
    try:
        settings = Settings(
            openrouter=OpenRouterSettings(default_model="google/gemini-2.0-flash-exp")
        )
        llm = OpenRouterLLM(context=Context(config=settings), model="")

        assert llm.default_request_params.model == "google/gemini-2.0-flash-exp"
    finally:
        ModelDatabase.clear_runtime_model_params(provider=Provider.OPENROUTER)


def test_openrouter_provider_base_url_prefers_config_over_env() -> None:
    original_env = os.environ.get("OPENROUTER_BASE_URL")
    os.environ["OPENROUTER_BASE_URL"] = "https://env-openrouter.example/v1"
    try:
        settings = Settings(openrouter=OpenRouterSettings(base_url="https://config.example/v1"))
        llm = OpenRouterLLM(context=Context(config=settings), model="openai/gpt-4.1-mini")

        assert llm._base_url() == "https://config.example/v1"
    finally:
        if original_env is None:
            os.environ.pop("OPENROUTER_BASE_URL", None)
        else:
            os.environ["OPENROUTER_BASE_URL"] = original_env


def test_huggingface_provider_default_model_used_with_provider_suffix() -> None:
    settings = Settings(
        hf=HuggingFaceSettings(
            default_model="moonshotai/kimi-k2-instruct",
            default_provider="fireworks-ai",
        )
    )
    llm = HuggingFaceLLM(context=Context(config=settings), model="")

    assert llm.default_request_params.model == "moonshotai/kimi-k2-instruct"

    request = llm._prepare_api_request(
        [{"role": "user", "content": "hi"}],
        None,
        llm.default_request_params,
    )
    assert request["model"] == "moonshotai/kimi-k2-instruct:fireworks-ai"


def test_deepseek_provider_defaults_to_v4_flash() -> None:
    llm = DeepSeekLLM(context=Context(config=Settings()), model="")

    assert llm.default_request_params.model == "deepseek-v4-flash"


def test_deepseek_provider_config_default_model_used_when_model_missing() -> None:
    settings = Settings(deepseek=DeepSeekSettings(default_model="deepseek-v4-pro"))
    llm = DeepSeekLLM(context=Context(config=settings), model="")

    assert llm.default_request_params.model == "deepseek-v4-pro"


def test_deepseek_v4_request_enables_thinking_by_default() -> None:
    llm = DeepSeekLLM(context=Context(config=Settings()), model="deepseek-v4-pro")

    request = llm._prepare_api_request(
        [{"role": "user", "content": "hi"}],
        None,
        llm.default_request_params,
    )

    assert request["reasoning_effort"] == "high"
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}


def test_deepseek_v4_request_maps_reasoning_and_can_disable_thinking() -> None:
    medium_llm = DeepSeekLLM(
        context=Context(config=Settings()),
        model="deepseek-v4-pro",
        reasoning_effort="medium",
    )
    medium_request = medium_llm._prepare_api_request(
        [{"role": "user", "content": "hi"}],
        None,
        medium_llm.default_request_params,
    )
    assert medium_request["reasoning_effort"] == "high"
    assert medium_request["extra_body"] == {"thinking": {"type": "enabled"}}

    max_llm = DeepSeekLLM(
        context=Context(config=Settings()),
        model="deepseek-v4-pro",
        reasoning_effort="xhigh",
    )
    max_request = max_llm._prepare_api_request(
        [{"role": "user", "content": "hi"}],
        None,
        max_llm.default_request_params,
    )
    assert max_request["reasoning_effort"] == "max"
    assert max_request["extra_body"] == {"thinking": {"type": "enabled"}}

    off_llm = DeepSeekLLM(
        context=Context(config=Settings()),
        model="deepseek-v4-pro",
        reasoning_effort=False,
    )
    off_request = off_llm._prepare_api_request(
        [{"role": "user", "content": "hi"}],
        None,
        off_llm.default_request_params,
    )
    assert "reasoning_effort" not in off_request
    assert off_request["extra_body"] == {"thinking": {"type": "disabled"}}

    none_llm = DeepSeekLLM(
        context=Context(config=Settings()),
        model="deepseek-v4-pro",
        reasoning_effort="none",
    )
    none_request = none_llm._prepare_api_request(
        [{"role": "user", "content": "hi"}],
        None,
        none_llm.default_request_params,
    )
    assert "reasoning_effort" not in none_request
    assert none_request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_azure_uses_azure_deployment_when_default_model_unset() -> None:
    settings = Settings(
        azure=AzureSettings(
            api_key="test-key",
            base_url="https://example.openai.azure.com/",
            azure_deployment="deployment-model",
        )
    )
    llm = AzureOpenAILLM(context=Context(config=settings), model="")

    assert llm.default_request_params.model == "deployment-model"


def test_azure_default_model_overrides_azure_deployment() -> None:
    settings = Settings(
        azure=AzureSettings(
            api_key="test-key",
            base_url="https://example.openai.azure.com/",
            azure_deployment="deployment-model",
            default_model="preferred-model",
        )
    )
    llm = AzureOpenAILLM(context=Context(config=settings), model="")

    assert llm.default_request_params.model == "preferred-model"
