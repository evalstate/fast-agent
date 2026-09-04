from types import SimpleNamespace

import pytest
from mcp.types import ImageContent, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import DeepSeekSettings, OpenAIWebSearchSettings, Settings
from fast_agent.constants import REASONING
from fast_agent.context import Context
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.openai.llm_deepseek import (
    DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_DEEPSEEK_REASONING_EFFORT,
    SUPPORTED_DEEPSEEK_MODELS,
    DeepSeekResponsesLLM,
)
from fast_agent.llm.provider_types import Provider


def _input_items() -> list[dict[str, object]]:
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]


def test_deepseek_responses_defaults_to_sse_and_official_endpoint() -> None:
    llm = DeepSeekResponsesLLM(context=Context(config=Settings()), model="")

    assert llm.provider == Provider.DEEPSEEK
    assert llm.configured_transport == "sse"
    assert llm.default_request_params.model == DEFAULT_DEEPSEEK_MODEL
    assert llm._base_url() == DEEPSEEK_BASE_URL


def test_deepseek_responses_uses_provider_configuration() -> None:
    settings = Settings(
        deepseek=DeepSeekSettings(
            api_key="deepseek-key",
            base_url="https://gateway.example/deepseek",
            default_model=DEFAULT_DEEPSEEK_MODEL,
            default_headers={"X-Test": "1"},
        )
    )
    llm = DeepSeekResponsesLLM(context=Context(config=settings), model="")

    assert llm._api_key() == "deepseek-key"
    assert llm._base_url() == "https://gateway.example/deepseek"
    assert llm._default_headers() == {"X-Test": "1"}


@pytest.mark.parametrize("model", SUPPORTED_DEEPSEEK_MODELS)
def test_deepseek_responses_accepts_native_models(model: str) -> None:
    llm = DeepSeekResponsesLLM(context=Context(config=Settings()), model=model)

    assert llm.default_request_params.model == model


@pytest.mark.parametrize("model", ["deepseek-chat", "deepseek-reasoner"])
def test_deepseek_responses_rejects_models_not_migrated_to_responses(model: str) -> None:
    with pytest.raises(ModelConfigError, match="DeepSeek Responses supports"):
        DeepSeekResponsesLLM(context=Context(config=Settings()), model=model)


def test_deepseek_factory_builds_sse_responses_adapter() -> None:
    factory = ModelFactory.create_factory("deepseek?reasoning=low")
    llm = factory(LlmAgent(AgentConfig(name="test")))

    assert isinstance(llm, DeepSeekResponsesLLM)
    assert llm.configured_transport == "sse"


def test_deepseek_vision_model_serializes_inline_image_for_responses() -> None:
    llm = DeepSeekResponsesLLM(
        context=Context(config=Settings()),
        model="deepseek-v4-flash-vision-exp",
    )

    parts = llm._convert_content_parts(
        [
            TextContent(type="text", text="What is in this image?"),
            ImageContent(type="image", data="aW1hZ2U=", mime_type="image/png"),
        ],
        role="user",
    )

    assert parts == [
        {"type": "input_text", "text": "What is in this image?"},
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,aW1hZ2U=",
        },
    ]


def test_deepseek_factory_forwards_web_search_override() -> None:
    factory = ModelFactory.create_factory("deepseek?web_search=true")
    llm = factory(LlmAgent(AgentConfig(name="test")))
    assert isinstance(llm, DeepSeekResponsesLLM)

    args = llm._build_response_args(
        _input_items(),
        llm.default_request_params,
        tools=None,
    )

    assert args["tools"] == [{"type": "web_search"}]
    assert llm.web_search_enabled is True


def test_deepseek_rejects_websocket_transport() -> None:
    with pytest.raises(ModelConfigError, match="WebSocket transport"):
        ModelFactory.parse_model_string("deepseek?transport=ws")


@pytest.mark.parametrize(
    ("reasoning", "wire_effort"),
    [
        (None, DEFAULT_DEEPSEEK_REASONING_EFFORT),
        (True, DEFAULT_DEEPSEEK_REASONING_EFFORT),
        ("low", "low"),
        ("high", "high"),
        ("max", "max"),
        ("none", "none"),
        (False, "none"),
    ],
)
def test_deepseek_responses_builds_native_reasoning_payload(
    reasoning: str | bool | None,
    wire_effort: str,
) -> None:
    llm = DeepSeekResponsesLLM(
        context=Context(config=Settings()),
        model=DEFAULT_DEEPSEEK_MODEL,
        reasoning_effort=reasoning,
    )

    args = llm._build_response_args(
        _input_items(),
        llm.default_request_params,
        tools=None,
    )

    assert args["reasoning"] == {"effort": wire_effort}
    assert "include" not in args
    assert "parallel_tool_calls" not in args
    assert "service_tier" not in args
    assert "store" not in args


def test_deepseek_responses_enables_sanitized_web_search_tool() -> None:
    settings = Settings(
        deepseek=DeepSeekSettings(
            web_search=OpenAIWebSearchSettings(
                enabled=True,
                search_context_size="high",
            )
        )
    )
    llm = DeepSeekResponsesLLM(context=Context(config=settings))

    args = llm._build_response_args(
        _input_items(),
        llm.default_request_params,
        tools=None,
    )

    assert args["tools"] == [{"type": "web_search"}]
    assert "include" not in args
    assert llm.web_search_enabled is True


def test_deepseek_responses_preserves_structured_output_format() -> None:
    llm = DeepSeekResponsesLLM(context=Context(config=Settings()))
    request_params = llm.default_request_params.model_copy(
        update={
            "response_format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        }
    )

    args = llm._build_response_args(_input_items(), request_params, tools=None)

    assert args["text"]["format"]["type"] == "json_schema"
    assert args["text"]["format"]["name"] == "answer"


def test_deepseek_reasoning_text_items_are_exposed_on_reasoning_channel() -> None:
    llm = DeepSeekResponsesLLM(context=Context(config=Settings()))
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=[],
                content=[
                    SimpleNamespace(
                        type="reasoning_text",
                        text="DeepSeek chain of thought",
                    )
                ],
            )
        ]
    )

    blocks = llm._extract_reasoning_summary(response, [])

    assert blocks == [TextContent(type="text", text="DeepSeek chain of thought")]


def test_deepseek_plain_text_reasoning_is_replayed_for_tool_continuation() -> None:
    llm = DeepSeekResponsesLLM(context=Context(config=Settings()))

    items = llm._extract_encrypted_reasoning_items(
        {REASONING: [TextContent(type="text", text="Call the weather tool")]}
    )

    assert items == [
        {
            "type": "reasoning",
            "content": [
                {
                    "type": "reasoning_text",
                    "text": "Call the weather tool",
                }
            ],
        }
    ]
