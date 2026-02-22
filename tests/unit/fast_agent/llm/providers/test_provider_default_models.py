from fast_agent.config import (
    AzureSettings,
    HuggingFaceSettings,
    OpenAISettings,
    OpenResponsesSettings,
    OpenRouterSettings,
    Settings,
)
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.openai.llm_azure import AzureOpenAILLM
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


def test_openai_explicit_model_overrides_provider_default() -> None:
    settings = Settings(openai=OpenAISettings(default_model="gpt-4.1-mini"))
    llm = OpenAILLM(context=Context(config=settings), model="gpt-4.1")

    assert llm.default_request_params.model == "gpt-4.1"


def test_responses_provider_default_model_used_when_model_missing() -> None:
    settings = Settings(responses=OpenAISettings(default_model="gpt-5.1"))
    llm = ResponsesLLM(context=Context(config=settings), model="")

    assert llm.default_request_params.model == "gpt-5.1"


def test_openresponses_provider_default_model_used_when_model_missing() -> None:
    settings = Settings(openresponses=OpenResponsesSettings(default_model="gpt-oss-120b"))
    llm = OpenResponsesLLM(context=Context(config=settings), model="")

    assert llm.default_request_params.model == "gpt-oss-120b"


def test_openrouter_provider_default_model_used_when_model_missing() -> None:
    ModelDatabase.clear_runtime_model_params(provider=Provider.OPENROUTER)
    try:
        settings = Settings(openrouter=OpenRouterSettings(default_model="google/gemini-2.0-flash-exp"))
        llm = OpenRouterLLM(context=Context(config=settings), model="")

        assert llm.default_request_params.model == "google/gemini-2.0-flash-exp"
    finally:
        ModelDatabase.clear_runtime_model_params(provider=Provider.OPENROUTER)


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
