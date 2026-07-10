from fast_agent.config import MetaAISettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.openai.metaai_responses import (
    DEFAULT_META_AI_MODEL,
    MetaAIResponsesLLM,
)
from fast_agent.llm.provider_types import Provider


def test_metaai_responses_provider_defaults_to_sse_transport() -> None:
    llm = MetaAIResponsesLLM(
        context=Context(config=Settings(metaai=MetaAISettings(api_key="test-key"))),
        model="muse-spark-1.1",
    )

    assert llm.provider == Provider.META_AI
    assert llm.configured_transport == "sse"


def test_metaai_responses_default_model_used_when_model_missing() -> None:
    llm = MetaAIResponsesLLM(
        context=Context(config=Settings(metaai=MetaAISettings(api_key="test-key"))),
        model="",
    )

    assert llm.default_request_params.model == DEFAULT_META_AI_MODEL


def test_metaai_responses_uses_metaai_config_fallback() -> None:
    settings = Settings(
        metaai=MetaAISettings(
            api_key="meta-key",
            base_url="https://gateway.example/metaai/v1",
            default_headers={"X-Test": "1"},
            default_model="muse-spark-1.1",
        )
    )
    llm = MetaAIResponsesLLM(context=Context(config=settings), model="")

    assert llm._api_key() == "meta-key"
    assert llm._base_url() == "https://gateway.example/metaai/v1"
    assert llm._default_headers() == {"X-Test": "1"}
    assert llm.default_request_params.model == "muse-spark-1.1"
