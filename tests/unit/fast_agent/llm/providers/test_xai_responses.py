import json
from types import SimpleNamespace

import pytest
from mcp.types import TextContent

from fast_agent.config import Settings, XAISettings, XAIWebSearchSettings
from fast_agent.context import Context
from fast_agent.llm.provider.openai.responses_websocket import (
    StatelessResponsesWsPlanner,
    resolve_responses_ws_url,
)
from fast_agent.llm.provider.openai.tool_stream_state import OpenAIToolStreamState
from fast_agent.llm.provider.openai.xai_responses import (
    DEFAULT_XAI_MODEL,
    XAIResponsesLLM,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting


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


def test_xai_responses_default_model_used_when_model_missing() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="",
    )

    assert llm.default_request_params.model == DEFAULT_XAI_MODEL


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


def test_xai_responses_uses_stateless_websocket_planner() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
    )

    assert isinstance(llm._new_ws_request_planner(), StatelessResponsesWsPlanner)


def test_xai_responses_builds_conservative_response_payload_with_default_reasoning() -> None:
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
    assert args["reasoning"] == {"effort": "low"}
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


def test_xai_responses_builds_payload_with_reasoning_none() -> None:
    llm = XAIResponsesLLM(
        context=Context(config=Settings(xai=XAISettings(api_key="test-key"))),
        model="grok-4.3",
        reasoning_effort="none",
    )
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]

    args = llm._build_response_args(input_items, llm.default_request_params, tools=None)

    assert llm.reasoning_effort == ReasoningEffortSetting(kind="effort", value="none")
    assert args["reasoning"] == {"effort": "none"}


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
    assert "include" not in args


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
