import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp_types import TextContent
from openai.types.responses import ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from fast_agent.config import Settings, XAISettings, XAIWebSearchSettings
from fast_agent.constants import OPENAI_ASSISTANT_MESSAGE_ITEMS
from fast_agent.context import Context
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider.openai.responses_websocket import (
    ResponsesWebSocketError,
    StatelessResponsesWsPlanner,
    resolve_responses_ws_url,
)
from fast_agent.llm.provider.openai.tool_stream_state import OpenAIToolStreamState
from fast_agent.llm.provider.openai.xai_responses import (
    DEFAULT_XAI_MODEL,
    GROK_45_HIGH_STREAMING_TIMEOUT,
    XAIResponsesLLM,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.llm.request_params import RequestParams
from fast_agent.llm.usage_tracking import UsageSchema
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

REPO_ROOT = Path(__file__).resolve().parents[5]


class _XAIStreamingHarness(XAIResponsesLLM):
    def __init__(self) -> None:
        super().__init__(
            context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
            model="grok-4.3",
            x_search=True,
        )
        self.events: list[tuple[str, dict]] = []

    def _notify_tool_stream_listeners(self, event_type, payload=None) -> None:
        self.events.append((event_type, payload or {}))


def test_xai_responses_provider_defaults_to_websocket_transport() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
    )

    assert llm.provider == Provider.XAI
    assert llm.configured_transport == "websocket"


def test_xai_websocket_usage_preserves_missing_cache_write_as_unknown() -> None:
    payload = json.loads(
        (
            REPO_ROOT
            / "tests"
            / "fixtures"
            / "llm_traces"
            / "sanitized"
            / "xai_responses_websocket_usage_20260715.json"
        ).read_text()
    )
    input_details = payload.pop("input_tokens_details")
    output_details = payload.pop("output_tokens_details")
    usage = ResponseUsage.model_construct(
        **payload,
        input_tokens_details=InputTokensDetails.model_construct(**input_details),
        output_tokens_details=OutputTokensDetails.model_construct(**output_details),
    )
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.5",
    )

    turn = llm._translate_responses_usage(
        usage,
        provider=Provider.XAI,
        model="grok-4.5",
    )

    assert turn.usage_schema is UsageSchema.OPENAI_RESPONSES_COMPATIBLE
    assert turn.prompt.total == 373
    assert turn.prompt.uncached is None
    assert turn.prompt.cache_read == 128
    assert turn.prompt.cache_write is None
    assert turn.completion.total == 138
    assert turn.completion.reasoning == 124


def test_xai_responses_default_model_used_when_model_missing() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="",
    )

    assert llm.default_request_params.model == DEFAULT_XAI_MODEL


@pytest.mark.parametrize(
    ("model", "reasoning_effort", "expected_timeout"),
    [
        ("grok-4.5", "high", GROK_45_HIGH_STREAMING_TIMEOUT),
        ("grok-4.5", "medium", 120.0),
        ("grok-4.5", "low", 120.0),
        ("grok-4.3", "high", 120.0),
    ],
)
def test_xai_grok_45_high_reasoning_gets_extended_streaming_timeout(
    model: str,
    reasoning_effort: str,
    expected_timeout: float,
) -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model=model,
        reasoning_effort=reasoning_effort,
    )

    assert llm.default_request_params.streaming_timeout == expected_timeout


@pytest.mark.parametrize("streaming_timeout", [45.0, None])
def test_xai_explicit_streaming_timeout_overrides_high_reasoning_default(
    streaming_timeout: float | None,
) -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.5",
        reasoning_effort="high",
        request_params=RequestParams(streaming_timeout=streaming_timeout),
    )

    assert llm.default_request_params.streaming_timeout == streaming_timeout


def test_xai_implicit_request_timeout_does_not_block_high_reasoning_default() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.5",
        reasoning_effort="high",
        request_params=RequestParams(model="grok-4.5", use_history=False),
    )

    assert llm.default_request_params.streaming_timeout == GROK_45_HIGH_STREAMING_TIMEOUT


def test_xai_responses_uses_xai_config_fallback() -> None:
    settings = Settings(
        xai=XAISettings(
            api_key="xai-key",
            base_url="https://gateway.example/xai/v1",
            default_headers={"X-Test": "1"},
            default_model="grok-4.5",
        )
    )
    llm = XAIResponsesLLM(context=Context(config=settings), model="")

    assert llm._api_key() == "xai-key"
    assert llm._base_url() == "https://gateway.example/xai/v1"
    assert llm._default_headers() == {"X-Test": "1"}
    assert llm.default_request_params.model == "grok-4.5"


def test_xai_responses_websocket_url_uses_responses_endpoint() -> None:
    assert resolve_responses_ws_url("https://api.x.ai/v1") == "wss://api.x.ai/v1/responses"


def test_xai_responses_websocket_headers_are_not_openai_beta_headers() -> None:
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


def test_xai_websocket_disables_client_generated_keepalive_pings() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
    )

    assert llm._websocket_keepalive_options() == {"ping_interval": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key_source", ["config", "environment", "init"])
async def test_xai_api_key_401_does_not_enter_oauth_refresh(
    monkeypatch, api_key_source: str
) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    settings = Settings(xai=XAISettings())
    init_api_key: str | None = None
    if api_key_source == "config":
        settings = Settings(xai=XAISettings(api_key="configured-key"))
    elif api_key_source == "environment":
        monkeypatch.setenv("XAI_API_KEY", "environment-key")
    else:
        init_api_key = "init-key"

    rejected = ResponsesWebSocketError("rejected API key", status=401)

    async def reject_connection(
        self, url: str, headers: dict[str, str], timeout_seconds: float | None
    ):
        del self, url, headers, timeout_seconds
        raise rejected

    monkeypatch.setattr(ResponsesLLM, "_create_websocket_connection", reject_connection)
    llm = XAIResponsesLLM(
        context=Context(config=settings),
        model="grok-4.3",
        api_key=init_api_key,
    )

    with pytest.raises(ResponsesWebSocketError) as exc_info:
        await llm._create_websocket_connection("wss://api.x.ai/v1/responses", {}, None)

    assert exc_info.value is rejected


def test_xai_responses_uses_stateless_websocket_planner() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
    )

    assert isinstance(llm._new_ws_request_planner(), StatelessResponsesWsPlanner)


def test_xai_responses_builds_parallel_response_payload_with_default_reasoning() -> None:
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
    assert args["parallel_tool_calls"] is True
    assert args["include"] == ["reasoning.encrypted_content"]
    assert args["reasoning"] == {"effort": "high"}
    assert "service_tier" not in args
    assert "stream" not in args
    assert "background" not in args


def test_xai_responses_builds_payload_with_selected_reasoning_effort() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
        reasoning_effort="high",
    )
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]

    args = llm._build_response_args(input_items, llm.default_request_params, tools=None)

    assert llm.reasoning_effort == ReasoningEffortSetting(kind="effort", value="high")
    assert args["reasoning"] == {"effort": "high"}


@pytest.mark.parametrize("model", ["grok-4.5", "grok-4.6"])
def test_xai_responses_builds_experimental_streaming_payload(model: str) -> None:
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(
                xai=XAISettings(
                    api_key="test-key",
                    reasoning_summary="concise",
                    stream_tool_calls=True,
                )
            )
        ),
        model=model,
    )

    args = llm._build_response_args([], llm.default_request_params, tools=None)

    assert args["reasoning"] == {"effort": "high", "summary": "concise"}
    assert args["extra_body"] == {"stream_tool_calls": True}


def test_xai_responses_flattens_stream_tool_calls_for_websocket() -> None:
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(xai=XAISettings(api_key="test-key", stream_tool_calls=True))
        ),
        model="grok-4.6",
    )
    args = llm._build_response_args([], llm.default_request_params, tools=None)

    llm._prepare_websocket_arguments(args)

    assert args["stream_tool_calls"] is True
    assert "extra_body" not in args


def test_xai_responses_rejects_unverified_experimental_model() -> None:
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(xai=XAISettings(api_key="test-key", stream_tool_calls=True))
        ),
        model="grok-4.3",
    )

    with pytest.raises(ModelConfigError, match="supported only for grok-4.5, grok-4.6"):
        llm._build_response_args([], llm.default_request_params, tools=None)


def test_xai_grok_46_builds_payload_with_xhigh_reasoning() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.6",
        reasoning_effort="xhigh",
    )

    args = llm._build_response_args([], llm.default_request_params, tools=None)

    assert llm.reasoning_effort == ReasoningEffortSetting(kind="effort", value="xhigh")
    assert args["reasoning"] == {"effort": "xhigh"}


def test_xai_prompt_cache_key_is_stable_per_conversation_and_rotates_on_clear() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.6",
    )

    first = llm._build_response_args([], llm.default_request_params, tools=None)
    second = llm._build_response_args([], llm.default_request_params, tools=None)
    first_key = first["prompt_cache_key"]

    assert isinstance(first_key, str)
    assert first_key
    assert second["prompt_cache_key"] == first_key
    assert "extra_body" not in first

    planned = llm._new_ws_request_planner().plan(first)
    assert planned.arguments["prompt_cache_key"] == first_key

    llm.clear()
    after_clear = llm._build_response_args([], llm.default_request_params, tools=None)
    assert after_clear["prompt_cache_key"] != first_key


@pytest.mark.parametrize("model", ["grok-4.5", "grok-4.6"])
def test_xai_replays_distinct_assistant_messages_when_provider_reuses_item_id(
    model: str,
) -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model=model,
    )
    messages: list[PromptMessageExtended] = []
    for user_text, assistant_text in (
        ("good evening", "Hello."),
        ("write an essay", "The essay."),
        ("was that fun?", "Yes."),
    ):
        messages.append(
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text=user_text)],
            )
        )
        raw_item = {
            "type": "message",
            "id": "msg_reused",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": assistant_text}],
        }
        messages.append(
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text=assistant_text)],
                channels={
                    OPENAI_ASSISTANT_MESSAGE_ITEMS: [
                        TextContent(type="text", text=json.dumps(raw_item))
                    ]
                },
            )
        )

    input_items = llm._convert_to_provider_format(messages)
    args = llm._build_response_args(input_items, llm.default_request_params, tools=None)
    planned = llm._new_ws_request_planner().plan(args)
    replayed = planned.arguments["input"]

    assert [item["role"] for item in replayed] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [item["content"][0]["text"] for item in replayed if item["role"] == "assistant"] == [
        "Hello.",
        "The essay.",
        "Yes.",
    ]
    assert all("id" not in item for item in replayed if item["role"] == "assistant")


def test_xai_responses_advertises_web_search() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
    )

    assert llm.web_search_supported is True
    assert llm.web_search_enabled is False


def test_xai_responses_builds_web_search_tool_when_enabled() -> None:
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(
                xai=XAISettings(
                    api_key="test-key",
                    web_search=XAIWebSearchSettings(enabled=True),
                )
            )
        ),
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

    assert args["tools"] == [{"type": "web_search"}]
    assert args["include"] == ["reasoning.encrypted_content"]


def test_xai_responses_builds_xai_web_search_options() -> None:
    llm = XAIResponsesLLM(
        context=Context(
            config=Settings(
                xai=XAISettings(
                    api_key="test-key",
                    web_search=XAIWebSearchSettings(
                        enabled=True,
                        excluded_domains=["example.com"],
                        enable_image_understanding=True,
                    ),
                )
            )
        ),
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

    assert args["tools"] == [
        {
            "type": "web_search",
            "filters": {"excluded_domains": ["example.com"]},
            "enable_image_understanding": True,
        }
    ]


def test_xai_web_search_rejects_conflicting_domain_filters() -> None:
    with pytest.raises(ValueError):
        XAIWebSearchSettings(
            allowed_domains=["example.com"],
            excluded_domains=["example.org"],
        )


def test_xai_web_search_domain_filter_limit() -> None:
    domains = [f"domain-{index}.example.com" for index in range(6)]

    with pytest.raises(ValueError, match="at most 5 domains"):
        XAIWebSearchSettings(allowed_domains=domains)

    with pytest.raises(ValueError, match="at most 5 domains"):
        XAIWebSearchSettings(excluded_domains=domains)


def test_xai_responses_advertises_x_search() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
    )

    assert llm.x_search_supported is True
    assert llm.x_search_enabled is False


def test_xai_responses_builds_x_search_tool_when_enabled() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
        x_search=True,
    )
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]

    args = llm._build_response_args(input_items, llm.default_request_params, tools=None)

    assert args["tools"] == [{"type": "x_search"}]


def test_xai_responses_stream_renders_x_search_internal_calls_as_remote_tools() -> None:
    harness = _XAIStreamingHarness()

    handled = harness._handle_responses_output_item_added(
        event=SimpleNamespace(
            output_index=0,
            item_id="fc_1",
            item=SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="xs_1",
                name="x_keyword_search",
            ),
        ),
        tool_state=OpenAIToolStreamState(),
        notified_tool_indices=set(),
        model="grok-4.3",
    )

    assert handled is True
    assert len(harness.events) == 1
    event_type, payload = harness.events[0]
    assert event_type == "start"
    assert payload["tool_name"] == "x_keyword_search"
    assert payload["presentation_family"] == "remote_tool"
    assert payload["preserve_details"] is True
    assert payload["tool_display_name"] == "remote tool: x_keyword_search"


def test_xai_responses_filters_x_search_internal_function_calls() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
        x_search=True,
    )
    response = SimpleNamespace(
        model="grok-4.3",
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="x_keyword_search",
                arguments='{"query":"evalstate"}',
            )
        ],
    )

    assert llm._extract_tool_calls(response) is None


def test_xai_responses_records_x_search_internal_calls_as_server_metadata() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
        x_search=True,
    )
    response = SimpleNamespace(
        model="grok-4.3",
        output=[
            SimpleNamespace(
                type="custom_tool_call",
                id="ctc_1",
                call_id="xs_1",
                name="x_keyword_search",
                input='{"query":"from:evalstate","limit":"5"}',
                status="completed",
            )
        ],
    )

    payloads = llm._extract_provider_mcp_metadata(response)

    assert len(payloads) == 1
    assert isinstance(payloads[0], TextContent)
    payload = json.loads(payloads[0].text)
    assert payload == {
        "type": "server_tool_use",
        "provider_tool_type": "x_search_call",
        "name": "x_keyword_search",
        "id": "xs_1",
        "status": "completed",
        "arguments": '{"query":"from:evalstate","limit":"5"}',
        "input": {"query": "from:evalstate", "limit": "5"},
    }


def test_xai_responses_records_clean_x_search_tool_use_id() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
        x_search=True,
    )
    response = SimpleNamespace(
        model="grok-4.3",
        output=[
            SimpleNamespace(
                type="custom_tool_call",
                id="  ctc_1  ",
                call_id=123,
                name="x_keyword_search",
                input="{}",
            )
        ],
    )

    payloads = llm._extract_provider_mcp_metadata(response)

    assert len(payloads) == 1
    assert isinstance(payloads[0], TextContent)
    payload = json.loads(payloads[0].text)
    assert payload["id"] == "ctc_1"


def test_xai_responses_falls_back_to_arguments_when_input_is_not_text() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
        x_search=True,
    )
    response = SimpleNamespace(
        model="grok-4.3",
        output=[
            SimpleNamespace(
                type="custom_tool_call",
                id="ctc_1",
                name="x_keyword_search",
                input={"query": "ignored-provider-shape"},
                arguments='{"query":"from:evalstate"}',
            )
        ],
    )

    payloads = llm._extract_provider_mcp_metadata(response)

    assert len(payloads) == 1
    assert isinstance(payloads[0], TextContent)
    payload = json.loads(payloads[0].text)
    assert payload["arguments"] == '{"query":"from:evalstate"}'
    assert payload["input"] == {"query": "from:evalstate"}


def test_xai_responses_preserves_regular_function_calls_when_x_search_enabled() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
        x_search=True,
    )
    response = SimpleNamespace(
        model="grok-4.3",
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="local_tool",
                arguments='{"value":1}',
            )
        ],
    )

    tool_calls = llm._extract_tool_calls(response)

    assert tool_calls is not None
    assert tool_calls["call_1"].params.name == "local_tool"
