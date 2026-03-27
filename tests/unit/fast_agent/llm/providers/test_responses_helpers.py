import base64
import json
import os
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from mcp.types import (
    BlobResourceContents,
    CallToolRequest,
    CallToolRequestParams,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from openai import AsyncOpenAI
from openai.types.responses import ResponseFunctionToolCall
from pydantic import AnyUrl, ValidationError

from fast_agent.config import (
    CodexResponsesSettings,
    OpenAISettings,
    OpenAIUserLocationSettings,
    OpenAIWebSearchSettings,
    OpenResponsesSettings,
    Settings,
)
from fast_agent.constants import OPENAI_ASSISTANT_MESSAGE_ITEMS, OPENAI_REASONING_ENCRYPTED
from fast_agent.context import Context
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM
from fast_agent.llm.provider.openai.openresponses import (
    OpenResponsesLLM,
    _OpenResponsesRawStream,
)
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
from fast_agent.tools.apply_patch_tool import build_apply_patch_tool
from fast_agent.types import COMMENTARY_PHASE


class _ContentHarness(ResponsesContentMixin):
    def __init__(self) -> None:
        self.logger = get_logger("test.responses.content")
        self._tool_call_id_map = {}
        self._tool_name_map = {}
        self._tool_kind_map = {}


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


class _FakeAcloseResponse:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _FakeRawResponsesSequence:
    def __init__(self, events: list[Any]) -> None:
        self._events = list(events)
        self.response = _FakeAcloseResponse()

    def __aiter__(self):
        return self

    async def __anext__(self) -> Any:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


class _FakeOpenResponsesApi:
    def __init__(self, raw_stream: _FakeRawResponsesSequence) -> None:
        self.raw_stream = raw_stream
        self.create_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeRawResponsesSequence:
        self.create_calls.append(kwargs)
        return self.raw_stream

    def stream(self, **kwargs: Any) -> Any:
        self.stream_calls.append(kwargs)
        raise AssertionError("OpenResponses should not use the SDK accumulator stream path")


class _FakeOpenResponsesClient:
    def __init__(self, responses_api: _FakeOpenResponsesApi) -> None:
        self.responses = responses_api

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb


class _OpenResponsesCompletionHarness(OpenResponsesLLM):
    def __init__(self, client: _FakeOpenResponsesClient) -> None:
        settings = Settings(openresponses=OpenResponsesSettings(api_key="test-key"))
        super().__init__(
            context=Context(config=settings),
            model="openai/gpt-oss-120b",
            name="openresponses-raw-stream-test",
        )
        self._client = client
        self.seen_stream: Any | None = None

    def _responses_client(self) -> Any:
        return self._client

    async def _normalize_input_files(
        self,
        client: Any,
        input_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del client
        return input_items

    def _build_response_args(
        self,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Any] | None,
    ) -> dict[str, Any]:
        del request_params, tools
        return {
            "input": input_items,
            "model": "openai/gpt-oss-120b",
        }

    async def _process_stream(
        self,
        stream: Any,
        model: str,
        capture_filename: Any,
    ) -> tuple[Any, list[str]]:
        del model, capture_filename
        self.seen_stream = stream
        async for _event in stream:
            pass
        return await stream.get_final_response(), []


class _OutputHarness(ResponsesOutputMixin):
    def __init__(self, provider: Provider = Provider.RESPONSES) -> None:
        self.logger = get_logger("test.responses.output")
        self._tool_call_id_map = {}
        self._tool_name_map = {}
        self._tool_kind_map = {}
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


def test_convert_tool_calls_serializes_apply_patch_as_custom_tool_call() -> None:
    harness = _ContentHarness()
    harness._tool_kind_map["call_patch"] = "custom"

    items = harness._convert_tool_calls(
        {
            "call_patch": CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name="apply_patch",
                    arguments={
                        "input": (
                            "*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch\n"
                        )
                    },
                ),
            )
        }
    )

    assert items == [
        {
            "type": "custom_tool_call",
            "call_id": "call_patch",
            "name": "apply_patch",
            "input": ("*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch"),
        }
    ]


def test_convert_tool_results_serializes_apply_patch_as_custom_tool_call_output() -> None:
    harness = _ContentHarness()
    harness._tool_kind_map["call_patch"] = "custom"

    result = SimpleNamespace(content=[TextContent(type="text", text="Success")], isError=False)
    items = harness._convert_tool_results({"call_patch": result})

    assert items == [
        {
            "type": "custom_tool_call_output",
            "call_id": "call_patch",
            "output": "Success",
        }
    ]


def test_convert_tool_calls_keeps_namespaced_apply_patch_as_function_call() -> None:
    harness = _ContentHarness()
    harness._tool_kind_map["call_patch"] = "function"

    items = harness._convert_tool_calls(
        {
            "call_patch": CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name="docs__apply_patch",
                    arguments={"input": "*** Begin Patch\n*** End Patch\n"},
                ),
            )
        }
    )

    assert items == [
        {
            "type": "function_call",
            "id": "fc_patch",
            "call_id": "call_patch",
            "name": "docs__apply_patch",
            "arguments": json.dumps({"input": "*** Begin Patch\n*** End Patch\n"}),
        }
    ]


def test_extract_tool_calls_tracks_custom_kind_for_history_replay() -> None:
    harness = _OutputHarness()
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="custom_tool_call",
                id="ctc_123",
                call_id="call_patch",
                name="apply_patch",
                input="*** Begin Patch\n*** End Patch\n",
            )
        ]
    )

    tool_calls = harness._extract_tool_calls(response)
    assert tool_calls is not None
    assert harness._tool_kind_map["call_patch"] == "custom"
    assert harness._tool_kind_map["ctc_123"] == "custom"


def test_extract_custom_tool_call_maps_apply_patch_input() -> None:
    harness = _OutputHarness()
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="custom_tool_call",
                id="ctc_123",
                call_id="call_patch",
                name="apply_patch",
                input=("*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch\n"),
            )
        ]
    )

    tool_calls = harness._extract_tool_calls(response)
    assert tool_calls is not None
    request = tool_calls["call_patch"]
    assert request.params.name == "apply_patch"
    assert request.params.arguments == {
        "input": ("*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch\n")
    }


def test_build_response_args_serializes_apply_patch_as_custom_tool() -> None:
    llm = _build_responses_family_llm(Provider.RESPONSES, model_name="gpt-5.4")

    args = llm._build_response_args(
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "patch it"}],
            }
        ],
        request_params=RequestParams(model="gpt-5.4"),
        tools=[build_apply_patch_tool()],
    )

    tools_payload = args.get("tools")
    assert isinstance(tools_payload, list)
    assert len(tools_payload) == 1

    tool_payload = tools_payload[0]
    assert tool_payload["type"] == "custom"
    assert tool_payload["name"] == "apply_patch"
    assert tool_payload["description"] == (
        "Use the `apply_patch` tool to edit files. "
        "This is a FREEFORM tool, so do not wrap the patch in JSON."
    )
    assert tool_payload["format"]["type"] == "grammar"
    assert tool_payload["format"]["syntax"] == "lark"
    assert "*** Begin Patch" in tool_payload["format"]["definition"]


def test_record_usage_uses_harness_provider() -> None:
    harness = _OutputHarness(provider=Provider.CODEX_RESPONSES)

    usage = SimpleNamespace(input_tokens=12, output_tokens=8, total_tokens=20)
    harness._record_usage(usage, "gpt-5.3-codex")

    assert len(harness.captured_usages) == 1
    recorded = harness.captured_usages[0]
    assert recorded.provider == Provider.CODEX_RESPONSES


def test_codex_responses_display_model_uses_infinity_marker() -> None:
    llm = CodexResponsesLLM(provider=Provider.CODEX_RESPONSES, model="gpt-5.3-codex")

    assert llm._display_model("gpt-5.3-codex") == "∞gpt-5.3-codex"


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


def _build_responses_family_llm(
    provider: Provider,
    *,
    model_name: str | None = None,
    configured_service_tier: Literal["fast", "flex"] | None = None,
) -> ResponsesLLM:
    if provider == Provider.RESPONSES:
        settings = Settings(
            responses=OpenAISettings(api_key="test-key", service_tier=configured_service_tier)
        )
        model = model_name or "gpt-5-mini"
        llm_class = ResponsesLLM
    elif provider == Provider.OPENRESPONSES:
        settings = Settings(
            openresponses=OpenResponsesSettings(
                api_key="test-key",
                service_tier=configured_service_tier,
            )
        )
        model = model_name or "gpt-5-mini"
        llm_class = OpenResponsesLLM
    elif provider == Provider.CODEX_RESPONSES:
        if configured_service_tier == "flex":
            raise AssertionError("codexresponses does not support flex service tier")
        codex_service_tier: Literal["fast"] | None = configured_service_tier
        settings = Settings(
            codexresponses=CodexResponsesSettings(
                api_key="test-key",
                service_tier=codex_service_tier,
            )
        )
        model = model_name or "gpt-5.3-codex"
        llm_class = CodexResponsesLLM
    else:
        raise AssertionError(f"unexpected provider: {provider}")

    context = Context(config=settings)
    return llm_class(context=context, model=model, name=f"{provider.value}-service-tier-test")


def test_responses_provider_defaults_to_websocket_preferred_transport() -> None:
    llm = _build_responses_family_llm(Provider.RESPONSES, model_name="gpt-5.4")

    assert llm.configured_transport == "auto"


def test_codexresponses_provider_defaults_to_websocket_preferred_transport() -> None:
    llm = _build_responses_family_llm(Provider.CODEX_RESPONSES, model_name="gpt-5.3-codex")

    assert llm.configured_transport == "auto"


def test_openresponses_provider_keeps_sse_transport_default() -> None:
    llm = _build_responses_family_llm(Provider.OPENRESPONSES, model_name="gpt-5-mini")

    assert llm.configured_transport == "sse"


@pytest.mark.asyncio
async def test_openresponses_raw_stream_tracks_final_response() -> None:
    final_response = SimpleNamespace(id="resp_final")
    raw_stream = _FakeRawResponsesSequence(
        [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.completed", response=final_response),
        ]
    )
    wrapped_stream = _OpenResponsesRawStream(raw_stream)

    seen_event_types: list[str] = []
    async for event in wrapped_stream:
        seen_event_types.append(event.type)

    assert seen_event_types == ["response.created", "response.completed"]
    assert await wrapped_stream.get_final_response() is final_response


@pytest.mark.asyncio
async def test_openresponses_sse_completion_uses_raw_create_stream() -> None:
    final_response = SimpleNamespace(id="resp_final")
    raw_stream = _FakeRawResponsesSequence(
        [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.completed", response=final_response),
        ]
    )
    responses_api = _FakeOpenResponsesApi(raw_stream)
    llm = _OpenResponsesCompletionHarness(_FakeOpenResponsesClient(responses_api))

    response, streamed_summary, normalized_input = await llm._responses_completion_sse(
        input_items=[{"type": "message", "role": "user", "content": []}],
        request_params=RequestParams(),
        tools=None,
        model_name="openai/gpt-oss-120b",
    )

    assert response is final_response
    assert streamed_summary == []
    assert normalized_input == [{"type": "message", "role": "user", "content": []}]
    assert llm.seen_stream is not None
    assert responses_api.stream_calls == []
    assert len(responses_api.create_calls) == 1
    assert responses_api.create_calls[0]["stream"] is True


def test_openresponses_client_allows_missing_api_key() -> None:
    original = os.getenv("OPENRESPONSES_API_KEY")
    try:
        if "OPENRESPONSES_API_KEY" in os.environ:
            del os.environ["OPENRESPONSES_API_KEY"]

        llm = OpenResponsesLLM(context=Context(config=Settings()), model="openai/gpt-oss-120b")

        assert llm._api_key() == ""
        assert isinstance(llm._responses_client(), AsyncOpenAI)
    finally:
        if original is None:
            if "OPENRESPONSES_API_KEY" in os.environ:
                del os.environ["OPENRESPONSES_API_KEY"]
        else:
            os.environ["OPENRESPONSES_API_KEY"] = original


def test_openresponses_client_preserves_explicit_empty_api_key() -> None:
    llm = OpenResponsesLLM(
        context=Context(config=Settings()),
        model="openai/gpt-oss-120b",
        api_key="",
    )

    assert llm._api_key() == ""
    assert isinstance(llm._responses_client(), AsyncOpenAI)


def test_openresponses_sampling_overrides_route_non_responses_fields_to_extra_body() -> None:
    llm = OpenResponsesLLM(
        context=Context(config=Settings()),
        model="openai/gpt-oss-120b",
    )

    args = llm._build_response_args(
        [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        RequestParams(top_k=40, min_p=0.05),
        None,
    )

    assert "top_k" not in args
    assert "min_p" not in args
    extra_body = args.get("extra_body")
    assert isinstance(extra_body, dict)
    assert extra_body["top_k"] == 40
    assert extra_body["min_p"] == 0.05


def test_explicit_responses_transport_override_is_preserved() -> None:
    settings = Settings(responses=OpenAISettings(api_key="test-key", transport="sse"))
    context = Context(config=settings)

    llm = ResponsesLLM(context=context, model="gpt-5.4", name="responses-explicit-sse")

    assert llm.configured_transport == "sse"


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


def test_convert_content_parts_embedded_text_resource_inlines_as_input_text():
    harness = _ContentHarness()
    resource = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=AnyUrl("file:///tmp/example.py"),
            mimeType="text/x-python",
            text="print('hello')",
        ),
    )

    parts = harness._convert_content_parts([resource], role="user")

    assert parts == [
        {
            "type": "input_text",
            "text": (
                '<fastagent:file title="example.py" mimetype="text/x-python">\n'
                "print('hello')\n"
                "</fastagent:file>"
            ),
        }
    ]


def test_convert_content_parts_office_resource_stays_as_input_file():
    harness = _ContentHarness()
    docx_data = base64.b64encode(b"PK\x03\x04docx-bytes").decode("ascii")
    resource = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=AnyUrl("file:///tmp/example.docx"),
            mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            blob=docx_data,
        ),
    )

    parts = harness._convert_content_parts([resource], role="user")

    assert parts == [
        {
            "type": "input_file",
            "file_data": docx_data,
            "filename": "example.docx",
        }
    ]


def test_convert_content_parts_image_resource_link_uses_remote_input_image():
    harness = _ContentHarness()
    resource = ResourceLink(
        type="resource_link",
        uri=AnyUrl("https://example.com/image.png"),
        mimeType="image/png",
        name="image.png",
    )

    parts = harness._convert_content_parts([resource], role="user")

    assert parts == [
        {
            "type": "input_image",
            "image_url": "https://example.com/image.png",
        }
    ]


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


def test_tool_fallback_notifications_for_custom_tool_call() -> None:
    harness = _StreamingHarness()
    tool_call = SimpleNamespace(
        call_id="call_patch",
        name="apply_patch",
        type="custom_tool_call",
    )

    harness._emit_tool_notification_fallback([tool_call], set(), model="gpt-test")

    events = harness.events
    assert [event for event, _payload in events] == ["start", "stop"]
    assert events[0][1]["tool_use_id"] == "call_patch"
    assert events[0][1]["tool_name"] == "apply_patch"


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


def test_extract_raw_assistant_message_items_preserves_phase_metadata() -> None:
    harness = _OutputHarness()
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                id="msg_123",
                role="assistant",
                status="completed",
                phase="commentary",
                content=[SimpleNamespace(type="output_text", text="Let me inspect that first.")],
            )
        ]
    )

    raw_items, message_phase = harness._extract_raw_assistant_message_items(response)

    assert message_phase == COMMENTARY_PHASE
    assert len(raw_items) == 1
    assert isinstance(raw_items[0], TextContent)
    payload = json.loads(raw_items[0].text)
    assert payload["type"] == "message"
    assert payload["phase"] == "commentary"
    assert payload["content"][0]["text"] == "Let me inspect that first."


def test_convert_extended_messages_to_provider_includes_assistant_phase() -> None:
    harness = _ContentHarness()
    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="Working through the plan")],
        phase=COMMENTARY_PHASE,
    )

    items = harness._convert_extended_messages_to_provider([message])

    assert items == [
        {
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Working through the plan"}],
        }
    ]


def test_convert_extended_messages_to_provider_uses_raw_assistant_items_channel() -> None:
    harness = _ContentHarness()
    raw_item = {
        "type": "message",
        "id": "msg_123",
        "role": "assistant",
        "status": "completed",
        "phase": "final_answer",
        "content": [{"type": "output_text", "text": "Final answer"}],
    }
    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="Final answer")],
        channels={
            OPENAI_ASSISTANT_MESSAGE_ITEMS: [TextContent(type="text", text=json.dumps(raw_item))]
        },
    )

    items = harness._convert_extended_messages_to_provider([message])

    assert items == [raw_item]


def test_extract_reasoning_summary_trims_streamed_fallback() -> None:
    harness = _OutputHarness()
    response = SimpleNamespace(output=[])

    blocks = harness._extract_reasoning_summary(
        response,
        ["Inspecting interactive prompt for error handling", "\n\n"],
    )

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextContent)
    assert blocks[0].text == "Inspecting interactive prompt for error handling"


def test_extract_reasoning_summary_omits_whitespace_only_streamed_fallback() -> None:
    harness = _OutputHarness()
    response = SimpleNamespace(output=[])

    blocks = harness._extract_reasoning_summary(response, ["\n", "   ", "\t"])

    assert blocks == []


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


@pytest.mark.parametrize(
    ("service_tier", "wire_value"),
    [("fast", "priority"), ("flex", "flex")],
)
def test_build_response_args_maps_service_tier_values(
    service_tier: Literal["fast", "flex"],
    wire_value: str,
) -> None:
    llm = _build_responses_family_llm(Provider.RESPONSES)

    args = llm._build_response_args(
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        request_params=RequestParams(model="gpt-5-mini", service_tier=service_tier),
        tools=None,
    )

    assert args["service_tier"] == wire_value


def test_build_response_args_omits_service_tier_when_unset() -> None:
    llm = _build_responses_family_llm(Provider.RESPONSES)

    args = llm._build_response_args(
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        request_params=RequestParams(model="gpt-5-mini"),
        tools=None,
    )

    assert "service_tier" not in args


def test_codexresponses_settings_reject_flex_service_tier() -> None:
    with pytest.raises(ValidationError):
        CodexResponsesSettings.model_validate({"api_key": "test-key", "service_tier": "flex"})


@pytest.mark.parametrize(
    ("provider", "expected_tiers"),
    [
        (Provider.RESPONSES, ("fast", "flex")),
        (Provider.CODEX_RESPONSES, ("fast",)),
    ],
)
def test_responses_family_llms_report_service_tier_support(
    provider: Provider,
    expected_tiers: tuple[str, ...],
) -> None:
    llm = _build_responses_family_llm(provider)

    assert llm.service_tier_supported is True
    assert llm.available_service_tiers == expected_tiers
    assert llm.service_tier is None


def test_openresponses_llm_hides_interactive_provider_controls() -> None:
    llm = _build_responses_family_llm(
        Provider.OPENRESPONSES,
        model_name="unsloth/Qwen3.5-9B-GGUF",
    )

    assert llm.service_tier_supported is False
    assert llm.web_search_supported is False
    assert llm.service_tier is None


def test_responses_llm_set_service_tier_updates_live_state() -> None:
    llm = _build_responses_family_llm(Provider.RESPONSES)

    llm.set_service_tier("fast")
    assert llm.service_tier == "fast"

    llm.set_service_tier(None)
    assert llm.service_tier is None


def test_codexresponses_set_service_tier_rejects_flex() -> None:
    llm = _build_responses_family_llm(Provider.CODEX_RESPONSES)

    with pytest.raises(ValueError, match="standard"):
        llm.set_service_tier("flex")


def test_responses_chatgpt_model_reports_fast_only_service_tier() -> None:
    llm = _build_responses_family_llm(
        Provider.RESPONSES,
        model_name="gpt-5.3-chat-latest",
    )

    assert llm.available_service_tiers == ("fast",)


def test_responses_chatgpt_model_rejects_flex_service_tier() -> None:
    llm = _build_responses_family_llm(
        Provider.RESPONSES,
        model_name="gpt-5.3-chat-latest",
    )

    with pytest.raises(ValueError, match="standard"):
        llm.set_service_tier("flex")


@pytest.mark.parametrize(
    ("provider", "configured_service_tier", "expected_wire_value"),
    [
        (Provider.RESPONSES, "fast", "priority"),
        (Provider.OPENRESPONSES, "flex", "flex"),
        (Provider.CODEX_RESPONSES, "fast", "priority"),
    ],
)
def test_configured_service_tier_defaults_apply_for_responses_family(
    provider: Provider,
    configured_service_tier: Literal["fast", "flex"],
    expected_wire_value: str,
) -> None:
    llm = _build_responses_family_llm(
        provider,
        configured_service_tier=configured_service_tier,
    )

    args = llm._build_response_args(
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        request_params=RequestParams(model=llm.default_request_params.model),
        tools=None,
    )

    assert llm.default_request_params.service_tier == configured_service_tier
    assert args["service_tier"] == expected_wire_value


def test_request_service_tier_overrides_configured_default() -> None:
    llm = _build_responses_family_llm(
        Provider.RESPONSES,
        configured_service_tier="fast",
    )

    args = llm._build_response_args(
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        request_params=RequestParams(model="gpt-5-mini", service_tier="flex"),
        tools=None,
    )

    assert args["service_tier"] == "flex"


def test_codexresponses_request_service_tier_rejects_flex() -> None:
    llm = _build_responses_family_llm(Provider.CODEX_RESPONSES)

    with pytest.raises(ModelConfigError, match="does not support service tier 'flex'"):
        llm._build_response_args(
            input_items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
            request_params=RequestParams(model="gpt-5.3-codex", service_tier="flex"),
            tools=None,
        )


def test_responses_chatgpt_request_service_tier_rejects_flex() -> None:
    llm = _build_responses_family_llm(
        Provider.RESPONSES,
        model_name="gpt-5.3-chat-latest",
    )

    with pytest.raises(ModelConfigError, match="gpt-5.3-chat-latest"):
        llm._build_response_args(
            input_items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
            request_params=RequestParams(model="gpt-5.3-chat-latest", service_tier="flex"),
            tools=None,
        )


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


def test_openresponses_web_search_does_not_inherit_openai_settings() -> None:
    settings = Settings(
        openai=OpenAISettings(
            api_key="test-key",
            web_search=OpenAIWebSearchSettings(enabled=True),
        )
    )
    llm = OpenResponsesLLM(
        context=Context(config=settings),
        model="unsloth/Qwen3.5-9B-GGUF",
        name="openresponses-web-search-test",
    )

    assert llm.web_search_enabled is False

    args = llm._build_response_args(
        input_items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "latest news"}],
            }
        ],
        request_params=RequestParams(model="unsloth/Qwen3.5-9B-GGUF"),
        tools=None,
    )

    assert "tools" not in args


def test_openresponses_web_search_uses_own_provider_settings() -> None:
    settings = Settings(
        openresponses=OpenResponsesSettings(
            api_key="test-key",
            web_search=OpenAIWebSearchSettings(enabled=True),
        )
    )
    llm = OpenResponsesLLM(
        context=Context(config=settings),
        model="unsloth/Qwen3.5-9B-GGUF",
        name="openresponses-web-search-test",
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
        request_params=RequestParams(model="unsloth/Qwen3.5-9B-GGUF"),
        tools=None,
    )

    tools_payload = args.get("tools")
    assert isinstance(tools_payload, list)
    assert len(tools_payload) == 1
    assert tools_payload[0]["type"] == "web_search"


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
        payload for event_type, payload in harness.events if event_type == "start"
    )
    assert first_start_payload["tool_name"] == "web_search"
    assert first_start_payload["tool_display_name"] == "Searching the web"
    assert first_start_payload["chunk"] == "starting search..."
