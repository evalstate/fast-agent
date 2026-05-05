from fast_agent.config import Settings, XAIResponsesSettings, XAISettings
from fast_agent.context import Context
from fast_agent.llm.provider.openai.responses_websocket import (
    StatelessResponsesWsPlanner,
    resolve_responses_ws_url,
)
from fast_agent.llm.provider.openai.xai_responses import (
    DEFAULT_XAI_RESPONSES_MODEL,
    XAIResponsesLLM,
)
from fast_agent.llm.provider_types import Provider


def test_xairesponses_provider_defaults_to_sse_transport() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
    )

    assert llm.provider == Provider.XAI
    assert llm.configured_transport == "sse"


def test_xairesponses_alias_remains_supported() -> None:
    llm = XAIResponsesLLM(
        provider=Provider.XAI_RESPONSES,
        context=Context(config=Settings(xairesponses=XAIResponsesSettings(api_key="test-key"))),
        model="grok-4.3",
    )

    assert llm.provider == Provider.XAI_RESPONSES
    assert llm._api_key() == "test-key"


def test_xairesponses_default_model_used_when_model_missing() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="",
    )

    assert llm.default_request_params.model == DEFAULT_XAI_RESPONSES_MODEL


def test_xairesponses_uses_xai_config_fallback() -> None:
    settings = Settings(
        xai=XAISettings(
            api_key="xai-key",
            base_url="https://gateway.example/xai/v1",
            default_headers={"X-Test": "1"},
            default_model="grok-4",
        )
    )
    llm = XAIResponsesLLM(context=Context(config=settings), model="")

    assert llm._api_key() == "xai-key"
    assert llm._base_url() == "https://gateway.example/xai/v1"
    assert llm._default_headers() == {"X-Test": "1"}
    assert llm.default_request_params.model == "grok-4"


def test_xairesponses_websocket_url_uses_responses_endpoint() -> None:
    assert resolve_responses_ws_url("https://api.x.ai/v1") == "wss://api.x.ai/v1/responses"


def test_xairesponses_websocket_headers_are_not_openai_beta_headers() -> None:
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(
                xai=XAISettings(
                    api_key="test-key",
                    default_headers={"X-Test": "1"},
                )
            )
        ),
        model="grok-4.3",
    )

    headers = llm._build_websocket_headers()

    assert headers["Authorization"] == "Bearer test-key"
    assert headers["X-Test"] == "1"
    assert "OpenAI-Beta" not in headers


def test_xairesponses_uses_stateless_websocket_planner() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
    )

    assert isinstance(llm._new_ws_request_planner(), StatelessResponsesWsPlanner)


def test_xairesponses_builds_conservative_response_payload() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
    )
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]

    args = llm._build_response_args(input_items, llm.default_request_params, tools=None)

    assert args["model"] == "grok-4.3"
    assert args["store"] is False
    assert args["input"] == input_items
    assert args["parallel_tool_calls"] is False
    assert "include" not in args
    assert "reasoning" not in args
    assert "service_tier" not in args
    assert "stream" not in args
    assert "background" not in args
