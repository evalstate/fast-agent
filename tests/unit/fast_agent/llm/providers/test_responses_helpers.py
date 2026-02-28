import base64
import json
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import ImageContent, TextContent
from openai import AsyncOpenAI
from openai.types.responses import ResponseFunctionToolCall
from pydantic import ValidationError

from fast_agent.config import (
    CodexResponsesSettings,
    OpenAISettings,
    OpenAIUserLocationSettings,
    OpenAIWebSearchSettings,
    Settings,
)
from fast_agent.constants import OPENAI_REASONING_ENCRYPTED
from fast_agent.context import Context
from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM
from fast_agent.llm.provider.openai.responses import (
    RESPONSE_INCLUDE_WEB_SEARCH_SOURCES,
    ResponsesLLM,
)
from fast_agent.llm.provider.openai.responses_content import ResponsesContentMixin
from fast_agent.llm.provider.openai.responses_files import ResponsesFileMixin
from fast_agent.llm.provider.openai.responses_output import ResponsesOutputMixin
from fast_agent.llm.provider.openai.responses_streaming import ResponsesStreamingMixin
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class _ContentHarness(ResponsesContentMixin):
    def __init__(self) -> None:
        self.logger = get_logger("test.responses.content")
        self._tool_call_id_map = {}


class _FileHarness(ResponsesFileMixin):
    def __init__(self) -> None:
        self._file_id_cache: dict[str, str] = {}

    async def _upload_file_bytes(self, client, data, filename, mime_type) -> str:
        return f"file_{len(data)}"


class _StreamingHarness(ResponsesStreamingMixin):
    def __init__(self) -> None:
        self.logger = get_logger("test.responses.streaming")
        self.name = "test"
        self._events: list[tuple[str, dict]] = []

    def _notify_tool_stream_listeners(self, event_type, payload=None) -> None:
        self._events.append((event_type, payload or {}))

    def _notify_stream_listeners(self, chunk) -> None:
        del chunk

    def _update_streaming_progress(self, content, model, estimated_tokens):
        del content, model
        return estimated_tokens

    def chat_turn(self) -> int:
        return 1

    @property
    def events(self) -> list[tuple[str, dict]]:
        return self._events


class _OutputHarness(ResponsesOutputMixin):
    def __init__(self, provider: Provider = Provider.RESPONSES) -> None:
        self.logger = get_logger("test.responses.output")
        self._tool_call_id_map = {}
        self._provider = provider
        self.captured_usages: list[Any] = []

    @property
    def provider(self) -> Provider:
        return self._provider

    def _finalize_turn_usage(self, usage) -> None:
        self.captured_usages.append(usage)

    def _normalize_tool_ids(self, tool_use_id: str | None) -> tuple[str, str]:
        tool_use_id = tool_use_id or ""
        if tool_use_id.startswith("fc_"):
            suffix = tool_use_id[len("fc_") :]
            return tool_use_id, f"call_{suffix}"
        if tool_use_id.startswith("call_"):
            suffix = tool_use_id[len("call_") :]
            return f"fc_{suffix}", tool_use_id
        return f"fc_{tool_use_id}", f"call_{tool_use_id}"


class _LoggerSpy:
    def __init__(self) -> None:
        self.info_calls: list[tuple[str, dict[str, Any]]] = []
        self.warning_calls: list[tuple[str, dict[str, Any]]] = []

    def info(self, message: str, **data: Any) -> None:
        self.info_calls.append((message, data))

    def warning(self, message: str, **data: Any) -> None:
        self.warning_calls.append((message, data))


def test_record_usage_uses_harness_provider() -> None:
    harness = _OutputHarness(provider=Provider.CODEX_RESPONSES)

    usage = SimpleNamespace(input_tokens=12, output_tokens=8, total_tokens=20)
    harness._record_usage(usage, "gpt-5.3-codex")

    assert len(harness.captured_usages) == 1
    recorded = harness.captured_usages[0]
    assert recorded.provider == Provider.CODEX_RESPONSES


class _FakeResponsesStream:
    def __init__(self, events: list[Any], final_response: Any) -> None:
        self._events = events
        self._final_response = final_response

    def __aiter__(self):
        self._iterator = iter(self._events)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def get_final_response(self):
        return self._final_response


def _build_responses_llm_with_web_search(
    web_search: OpenAIWebSearchSettings,
    *,
    override: bool | None = None,
) -> ResponsesLLM:
    settings = Settings(
        responses=OpenAISettings(
            api_key="test-key",
            web_search=web_search,
        )
    )
    context = Context(config=settings)
    kwargs: dict[str, Any] = {
        "context": context,
        "model": "gpt-5-mini",
        "name": "responses-web-search-test",
    }
    if override is not None:
        kwargs["web_search"] = override
    return ResponsesLLM(**kwargs)


def _build_codex_responses_llm_with_web_search(
    web_search: OpenAIWebSearchSettings | None = None,
    *,
    override: bool | None = None,
) -> CodexResponsesLLM:
    settings = Settings(
        codexresponses=CodexResponsesSettings(
            api_key="test-key",
            web_search=web_search or OpenAIWebSearchSettings(),
        )
    )
    context = Context(config=settings)
    kwargs: dict[str, Any] = {
        "context": context,
        "model": "gpt-5.3-codex",
        "name": "codex-responses-web-search-test",
    }
    if override is not None:
        kwargs["web_search"] = override
    return CodexResponsesLLM(**kwargs)


def test_openai_web_search_domain_allowlist_limit() -> None:
    domains = [f"domain-{index}.example.com" for index in range(101)]
    with pytest.raises(ValidationError):
        OpenAIWebSearchSettings(enabled=True, allowed_domains=domains)


def test_convert_content_parts_text_and_image():
    harness = _ContentHarness()
    image_data = base64.b64encode(b"image-bytes").decode("ascii")

    parts = harness._convert_content_parts(
        [
            TextContent(type="text", text="Hello"),
            ImageContent(type="image", data=image_data, mimeType="image/png"),
        ],
        role="user",
    )

    assert parts[0] == {"type": "input_text", "text": "Hello"}
    assert parts[1]["type"] == "input_image"
    assert parts[1]["image_url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_normalize_input_file_data_to_file_id():
    harness = _FileHarness()
    client = AsyncOpenAI(api_key="test")
    file_bytes = b"%PDF-1.4 dummy"
    file_data = base64.b64encode(file_bytes).decode("ascii")

    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Summarize"},
                {"type": "input_file", "file_data": file_data, "filename": "sample.pdf"},
            ],
        }
    ]

    normalized = await harness._normalize_input_files(client, input_items)
    content = normalized[0]["content"]
    assert content[0] == {"type": "input_text", "text": "Summarize"}
    assert content[1] == {"type": "input_file", "file_id": f"file_{len(file_bytes)}"}


@pytest.mark.asyncio
async def test_normalize_input_image_file_url(tmp_path):
    harness = _FileHarness()
    client = AsyncOpenAI(api_key="test")
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"file://{image_path}"},
            ],
        }
    ]

    normalized = await harness._normalize_input_files(client, input_items)
    content = normalized[0]["content"]
    assert content[0] == {"type": "input_image", "file_id": f"file_{len(image_path.read_bytes())}"}


def test_tool_fallback_notifications():
    harness = _StreamingHarness()
    tool_call = ResponseFunctionToolCall(
        arguments="{}",
        call_id="call_123",
        name="weather",
        type="function_call",
    )

    harness._emit_tool_notification_fallback([tool_call], set(), model="gpt-test")

    events = harness.events
    assert [event for event, _payload in events] == ["start", "stop"]
    assert events[0][1]["tool_use_id"] == "call_123"
    assert events[0][1]["tool_name"] == "weather"


def test_dedupes_duplicate_reasoning_ids():
    harness = _ContentHarness()
    payload = {"type": "reasoning", "encrypted_content": "abc", "id": "rs_dup"}
    reasoning_block = TextContent(type="text", text=json.dumps(payload))
    channels = {OPENAI_REASONING_ENCRYPTED: [reasoning_block]}

    messages = [
        PromptMessageExtended(role="assistant", content=[], channels=channels),
        PromptMessageExtended(role="assistant", content=[], channels=channels),
    ]

    items = harness._convert_extended_messages_to_provider(messages)
    reasoning_items = [item for item in items if item.get("type") == "reasoning"]
    assert len(reasoning_items) == 1


def test_responses_tool_use_id_prefers_call_id_when_available():
    """
    Responses streaming emits tool_use_id=call_id; tool execution must use the same
    identifier to avoid duplicated tool cards in ACP clients.
    """
    harness = _OutputHarness()
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_123",
                call_id="call_123",
                name="weather",
                arguments="{}",
            )
        ]
    )

    tool_calls = harness._extract_tool_calls(response)
    assert tool_calls is not None
    assert list(tool_calls.keys()) == ["call_123"]
    assert harness._tool_call_id_map["call_123"] == "call_123"


def test_responses_tool_use_id_falls_back_to_item_id_when_call_id_missing():
    harness = _OutputHarness()
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_456",
                call_id=None,
                name="weather",
                arguments="{}",
            )
        ]
    )

    tool_calls = harness._extract_tool_calls(response)
    assert tool_calls is not None
    assert list(tool_calls.keys()) == ["fc_456"]
    assert harness._tool_call_id_map["fc_456"] == "call_456"


def test_responses_filters_duplicate_tool_calls_across_turns():
    harness = _OutputHarness()

    first_response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_123",
                call_id="call_123",
                name="weather",
                arguments="{}",
            )
        ]
    )
    second_response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_123",
                call_id="call_123",
                name="weather",
                arguments="{}",
            ),
            SimpleNamespace(
                type="function_call",
                id="fc_456",
                call_id="call_456",
                name="stocks",
                arguments="{}",
            ),
        ]
    )

    first_tool_calls = harness._extract_tool_calls(first_response)
    assert first_tool_calls is not None
    assert list(first_tool_calls.keys()) == ["call_123"]
    assert harness._consume_tool_call_diagnostics() is None

    second_tool_calls = harness._extract_tool_calls(second_response)
    assert second_tool_calls is not None
    assert list(second_tool_calls.keys()) == ["call_456"]

    diagnostics = harness._consume_tool_call_diagnostics()
    assert diagnostics is not None
    assert diagnostics["kind"] == "duplicate_tool_calls_filtered"
    assert diagnostics["duplicate_count"] == 1
    assert diagnostics["duplicate_tool_call_ids"] == ["call_123"]
    assert diagnostics["raw_function_call_count"] == 2
    assert diagnostics["new_function_call_count"] == 1

    assert harness._consume_tool_call_diagnostics() is None


def test_responses_duplicate_tool_calls_emit_stop_progress_events() -> None:
    harness = _OutputHarness()
    logger_spy = _LoggerSpy()
    harness.logger = logger_spy  # type: ignore[assignment]
    harness.name = "assistant"  # type: ignore[attr-defined]

    first_response = SimpleNamespace(
        model="gpt-test",
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_123",
                call_id="call_123",
                name="execute",
                arguments="{}",
            )
        ],
    )
    duplicate_response = SimpleNamespace(
        model="gpt-test",
        output=[
            SimpleNamespace(
                type="function_call",
                id="fc_123",
                call_id="call_123",
                name="execute",
                arguments="{}",
            )
        ],
    )

    harness._extract_tool_calls(first_response)
    harness._extract_tool_calls(duplicate_response)

    stop_events = [
        data
        for message, data in logger_spy.info_calls
        if message == "Filtered duplicate Responses tool call"
    ]
    assert len(stop_events) == 1
    event_data = stop_events[0].get("data", {})
    assert event_data.get("progress_action") == ProgressAction.CALLING_TOOL
    assert event_data.get("tool_event") == "stop"
    assert event_data.get("tool_use_id") == "call_123"


def test_build_response_args_includes_openai_web_search_tool() -> None:
    llm = _build_responses_llm_with_web_search(
        OpenAIWebSearchSettings(
            enabled=True,
            allowed_domains=["openai.com"],
            external_web_access=False,
            search_context_size="high",
            user_location=OpenAIUserLocationSettings(
                city="Minneapolis",
                region="Minnesota",
                country="US",
                timezone="America/Chicago",
            ),
        )
    )

    args = llm._build_response_args(
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "latest news"}],
            }
        ],
        request_params=RequestParams(model="gpt-5-mini"),
        tools=None,
    )

    tools_payload = args.get("tools")
    assert isinstance(tools_payload, list)
    assert len(tools_payload) == 1

    web_tool = tools_payload[0]
    assert web_tool["type"] == "web_search"
    assert web_tool["filters"]["allowed_domains"] == ["openai.com"]
    assert web_tool["external_web_access"] is False
    assert web_tool["search_context_size"] == "high"
    assert web_tool["user_location"]["timezone"] == "America/Chicago"

    include_payload = args.get("include")
    assert isinstance(include_payload, list)
    assert RESPONSE_INCLUDE_WEB_SEARCH_SOURCES in include_payload


def test_web_search_override_disables_configured_web_search_tool() -> None:
    llm = _build_responses_llm_with_web_search(
        OpenAIWebSearchSettings(enabled=True),
        override=False,
    )

    args = llm._build_response_args(
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "latest news"}],
            }
        ],
        request_params=RequestParams(model="gpt-5-mini"),
        tools=None,
    )

    assert "tools" not in args
    assert llm.web_search_enabled is False

    include_payload = args.get("include")
    assert isinstance(include_payload, list)
    assert RESPONSE_INCLUDE_WEB_SEARCH_SOURCES not in include_payload


def test_responses_web_search_enabled_property_tracks_config() -> None:
    llm = _build_responses_llm_with_web_search(OpenAIWebSearchSettings(enabled=True))
    assert llm.web_search_enabled is True


def test_codex_web_search_defaults_disabled() -> None:
    llm = _build_codex_responses_llm_with_web_search()

    assert llm.web_search_enabled is False

    args = llm._build_response_args(
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "latest news"}],
            }
        ],
        request_params=RequestParams(model="gpt-5.3-codex"),
        tools=None,
    )

    assert args["tool_choice"] == "auto"
    assert "tools" not in args


def test_codex_web_search_enabled_adds_tool_payload() -> None:
    llm = _build_codex_responses_llm_with_web_search(
        OpenAIWebSearchSettings(
            enabled=True,
            allowed_domains=["openai.com"],
            external_web_access=True,
            search_context_size="high",
        )
    )

    assert llm.web_search_enabled is True

    args = llm._build_response_args(
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "latest news"}],
            }
        ],
        request_params=RequestParams(model="gpt-5.3-codex"),
        tools=None,
    )

    assert args["tool_choice"] == "auto"
    tools_payload = args.get("tools")
    assert isinstance(tools_payload, list)
    assert len(tools_payload) == 1
    web_tool = tools_payload[0]
    assert web_tool["type"] == "web_search"
    assert web_tool["filters"]["allowed_domains"] == ["openai.com"]
    assert web_tool["external_web_access"] is True

    include_payload = args.get("include")
    assert isinstance(include_payload, list)
    assert RESPONSE_INCLUDE_WEB_SEARCH_SOURCES in include_payload


def test_extract_web_search_metadata_captures_tool_and_citations() -> None:
    harness = _OutputHarness()
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="web_search_call",
                id="ws_123",
                status="completed",
                action=SimpleNamespace(
                    type="search",
                    queries=["openai news"],
                    sources=[SimpleNamespace(type="url", url="https://openai.com/blog")],
                ),
            ),
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(
                        type="output_text",
                        text="OpenAI announced updates.",
                        annotations=[
                            SimpleNamespace(
                                type="url_citation",
                                start_index=0,
                                end_index=24,
                                title="OpenAI Blog",
                                url="https://openai.com/blog",
                            )
                        ],
                    )
                ],
            ),
        ]
    )

    web_tools, citations = harness._extract_web_search_metadata(response)

    assert len(web_tools) == 1
    tool_block = web_tools[0]
    assert isinstance(tool_block, TextContent)
    tool_payload = json.loads(tool_block.text)
    assert tool_payload["type"] == "server_tool_use"
    assert tool_payload["name"] == "web_search"
    assert tool_payload["action"] == "search"

    citation_urls = {
        json.loads(citation.text).get("url")
        for citation in citations
        if isinstance(citation, TextContent)
    }
    assert "https://openai.com/blog" in citation_urls


def test_tool_fallback_notifications_support_web_search_call() -> None:
    harness = _StreamingHarness()
    web_search_call = SimpleNamespace(
        type="web_search_call",
        id="ws_123",
    )

    harness._emit_tool_notification_fallback([web_search_call], set(), model="gpt-test")

    events = harness.events
    assert [event for event, _payload in events] == ["start", "stop"]
    assert events[0][1]["tool_use_id"] == "ws_123"
    assert events[0][1]["tool_name"] == "web_search"


@pytest.mark.asyncio
async def test_stream_process_emits_web_search_status_events() -> None:
    harness = _StreamingHarness()
    final_response = SimpleNamespace(
        output=[SimpleNamespace(type="web_search_call", id="ws_123")],
        usage=None,
    )
    stream = _FakeResponsesStream(
        events=[
            SimpleNamespace(
                type="response.output_item.added",
                output_index=0,
                item=SimpleNamespace(type="web_search_call", id="ws_123"),
                item_id="ws_123",
            ),
            SimpleNamespace(
                type="response.web_search_call.searching",
                output_index=0,
                item_id="ws_123",
            ),
            SimpleNamespace(
                type="response.web_search_call.completed",
                output_index=0,
                item_id="ws_123",
            ),
            SimpleNamespace(type="response.completed", response=final_response),
        ],
        final_response=final_response,
    )

    await harness._process_stream(stream, model="gpt-test", capture_filename=None)

    event_types = [event for event, _payload in harness.events]
    assert event_types.count("start") >= 1
    assert "status" in event_types
    assert event_types.count("stop") >= 1

    first_start_payload = next(
        payload
        for event_type, payload in harness.events
        if event_type == "start"
    )
    assert first_start_payload["tool_name"] == "web_search"
    assert first_start_payload["tool_display_name"] == "Searching the web"
    assert first_start_payload["chunk"] == "starting search..."
