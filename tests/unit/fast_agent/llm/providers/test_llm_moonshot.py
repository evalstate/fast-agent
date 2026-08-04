from collections.abc import AsyncIterator

import pytest
from mcp.types import TextContent, Tool
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import MoonshotSettings, OpenAISettings, Settings
from fast_agent.constants import REASONING
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.openai.llm_moonshot import (
    MOONSHOT_BASE_URL,
    MoonshotLLM,
)
from fast_agent.llm.provider.openai.llm_openai import EmptyStreamError
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types import RequestParams

_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "echo_value",
        "description": "Echo a value.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    },
}
_STRUCTURED_TOOL = Tool(
    name="echo_value",
    description="Echo a value.",
    input_schema={
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    },
)


def _request(
    llm: MoonshotLLM,
    *,
    tools: list[ChatCompletionToolParam] | None = None,
) -> dict[str, object]:
    return llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        tools,
        RequestParams(model="kimi-k3"),
    )


def test_moonshot_model_routes_are_native_without_changing_hf_kimi_aliases() -> None:
    qualified = ModelFactory.parse_model_string("moonshot.kimi-k3")
    bare = ModelFactory.parse_model_string("kimi-k3")
    alias = ModelFactory.parse_model_string("kimik3")
    legacy_alias = ModelFactory.parse_model_string("kimi")
    legacy_repo = ModelFactory.parse_model_string("moonshotai/Kimi-K2.6")

    assert qualified.provider == Provider.MOONSHOT
    assert qualified.model_name == "kimi-k3"
    assert bare.provider == Provider.MOONSHOT
    assert alias.provider == Provider.MOONSHOT
    assert legacy_alias.provider == Provider.HUGGINGFACE
    assert legacy_repo.provider == Provider.HUGGINGFACE
    assert ModelFactory._load_provider_class(Provider.MOONSHOT) is MoonshotLLM


def test_moonshot_defaults_to_kimi_k3_and_official_endpoint() -> None:
    llm = MoonshotLLM(context=Context(config=Settings()), model="")

    assert llm.default_request_params.model == "kimi-k3"
    assert llm._base_url() == MOONSHOT_BASE_URL


def test_moonshot_config_overrides_model_endpoint_credentials_and_headers(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "environment-key")
    settings = Settings(
        moonshot=MoonshotSettings(
            api_key="configured-key",
            base_url="https://gateway.example/v1",
            default_model="kimi-custom",
            default_headers={"X-Test": "moonshot"},
        )
    )
    llm = MoonshotLLM(context=Context(config=settings), model="")

    assert llm.default_request_params.model == "kimi-custom"
    assert llm._base_url() == "https://gateway.example/v1"
    assert llm._api_key() == "configured-key"
    assert llm._default_headers() == {"X-Test": "moonshot"}


def test_moonshot_kimi_k3_capabilities_match_native_contract() -> None:
    params = ModelDatabase.get_model_params("kimi-k3", provider=Provider.MOONSHOT)

    assert params is not None
    assert params.context_window == 1_048_576
    assert params.max_output_tokens == 131_072
    assert params.default_provider == Provider.MOONSHOT
    assert params.reasoning == "reasoning_content"
    assert params.stream_mode == "manual"
    assert params.json_mode == "schema"
    assert params.structured_tool_policy == "no_tools"
    assert params.shell_output_byte_limit == 16_000
    assert "image/png" in params.tokenizes
    assert "video/mp4" in params.tokenizes
    assert "application/pdf" not in params.tokenizes


def test_moonshot_reasoning_defaults_to_max_and_supports_documented_efforts() -> None:
    default = _request(MoonshotLLM(context=Context(config=Settings()), model="kimi-k3"))
    low = _request(
        MoonshotLLM(
            context=Context(config=Settings()),
            model="kimi-k3",
            reasoning_effort=ReasoningEffortSetting(kind="effort", value="low"),
        )
    )
    high = _request(
        MoonshotLLM(
            context=Context(config=Settings()),
            model="kimi-k3",
            reasoning_effort=ReasoningEffortSetting(kind="effort", value="high"),
        )
    )

    assert default["reasoning_effort"] == "max"
    assert low["reasoning_effort"] == "low"
    assert high["reasoning_effort"] == "high"


def test_moonshot_always_on_reasoning_ignores_openai_and_disable_settings() -> None:
    settings = Settings(openai=OpenAISettings(reasoning="none"))
    inherited = _request(MoonshotLLM(context=Context(config=settings), model="kimi-k3"))
    disabled = _request(
        MoonshotLLM(
            context=Context(config=Settings()),
            model="kimi-k3",
            reasoning_effort=False,
        )
    )

    assert inherited["reasoning_effort"] == "max"
    assert disabled["reasoning_effort"] == "max"


def test_moonshot_request_uses_completion_tokens_and_streamed_parallel_tools() -> None:
    llm = MoonshotLLM(context=Context(config=Settings()), model="kimi-k3")
    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        [_TOOL],
        RequestParams(model="kimi-k3", max_tokens=2048),
    )

    assert request["max_completion_tokens"] == 2048
    assert request["stream_options"] == {"include_usage": True}
    assert request["tools"] == [_TOOL]
    assert "parallel_tool_calls" not in request
    assert "temperature" not in request
    assert "top_p" not in request


def test_moonshot_factory_default_uses_api_output_default() -> None:
    factory = ModelFactory.create_factory("kimik3")
    llm = factory(LlmAgent(AgentConfig(name="test")))

    assert isinstance(llm, MoonshotLLM)
    assert llm.default_request_params.max_tokens == 131_072
    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        llm.default_request_params,
    )
    assert request["max_completion_tokens"] == 131_072


def test_moonshot_omits_fixed_and_unsupported_sampling_fields() -> None:
    llm = MoonshotLLM(context=Context(config=Settings()), model="kimi-k3")
    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        RequestParams(
            model="kimi-k3",
            temperature=0.2,
            metadata={
                "top_p": 0.2,
                "top_k": 5,
                "min_p": 0.1,
                "n": 2,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.2,
                "repetition_penalty": 1.1,
            },
        ),
    )

    assert not set(request).intersection(
        {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
        }
    )


def test_moonshot_structured_output_uses_strict_json_schema_without_tools() -> None:
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    llm = MoonshotLLM(context=Context(config=Settings()), model="kimi-k3")

    messages, params = llm._prepare_structured_request(
        [Prompt.user("return json")],
        RequestParams(model="kimi-k3", structured_schema=schema),
        [_STRUCTURED_TOOL],
    )

    assert messages == [Prompt.user("return json")]
    assert params.response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": schema,
            "strict": True,
        },
    }


def test_moonshot_replays_reasoning_content_as_a_separate_assistant_field() -> None:
    llm = MoonshotLLM(context=Context(config=Settings()), model="kimi-k3")
    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="answer")],
        channels={REASONING: [TextContent(type="text", text="private reasoning")]},
    )

    converted = llm._convert_extended_messages_to_provider([message])

    assert converted == [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "private reasoning",
        }
    ]


async def _stream_chunks(
    chunks: list[ChatCompletionChunk],
) -> AsyncIterator[ChatCompletionChunk]:
    for chunk in chunks:
        yield chunk


def _chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    choice_usage: dict[str, int] | None = None,
) -> ChatCompletionChunk:
    choice: dict[str, object] = {
        "index": 0,
        "delta": {"content": content},
        "finish_reason": finish_reason,
    }
    if choice_usage is not None:
        choice["usage"] = choice_usage
    return ChatCompletionChunk.model_validate(
        {
            "id": "chunk",
            "created": 0,
            "model": "kimi-k3",
            "object": "chat.completion.chunk",
            "choices": [choice],
        }
    )


@pytest.mark.asyncio
async def test_moonshot_manual_stream_accepts_usage_inside_final_choice() -> None:
    llm = MoonshotLLM(context=Context(config=Settings()), model="kimi-k3")
    chunks = [
        _chunk(content="answer"),
        _chunk(
            finish_reason="stop",
            choice_usage={
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        ),
    ]

    completion, _ = await llm._process_stream_manual(_stream_chunks(chunks), "kimi-k3")

    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 10


@pytest.mark.asyncio
async def test_moonshot_manual_stream_rejects_partial_content_without_finish() -> None:
    llm = MoonshotLLM(context=Context(config=Settings()), model="kimi-k3")

    with pytest.raises(EmptyStreamError, match="without a finish reason"):
        await llm._process_stream_manual(
            _stream_chunks([_chunk(content="partial")]),
            "kimi-k3",
        )


@pytest.mark.asyncio
async def test_moonshot_embeds_remote_image_urls(monkeypatch) -> None:
    llm = MoonshotLLM(context=Context(config=Settings()), model="kimi-k3")

    async def download(_url: str) -> tuple[bytes | None, str | None]:
        return b"png", "image/png"

    monkeypatch.setattr(llm, "_download_remote_file", download)
    message: ChatCompletionMessageParam = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            }
        ],
    }

    normalized = await llm._embed_remote_media(message)

    assert normalized["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,cG5n"},
        }
    ]
