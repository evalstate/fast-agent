import base64
import json
from types import SimpleNamespace

import pytest
from mcp.types import ImageContent, TextContent
from openai import AsyncOpenAI
from openai.types.responses import ResponseFunctionToolCall

from fast_agent.constants import OPENAI_REASONING_ENCRYPTED
from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.provider.openai.responses_content import ResponsesContentMixin
from fast_agent.llm.provider.openai.responses_files import ResponsesFileMixin
from fast_agent.llm.provider.openai.responses_output import ResponsesOutputMixin
from fast_agent.llm.provider.openai.responses_streaming import ResponsesStreamingMixin
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

    def chat_turn(self) -> int:
        return 1

    @property
    def events(self) -> list[tuple[str, dict]]:
        return self._events


class _OutputHarness(ResponsesOutputMixin):
    def __init__(self) -> None:
        self.logger = get_logger("test.responses.output")
        self._tool_call_id_map = {}

    def _finalize_turn_usage(self, usage) -> None:  # pragma: no cover
        return None

    def _normalize_tool_ids(self, tool_use_id: str | None) -> tuple[str, str]:
        tool_use_id = tool_use_id or ""
        if tool_use_id.startswith("fc_"):
            suffix = tool_use_id[len("fc_") :]
            return tool_use_id, f"call_{suffix}"
        if tool_use_id.startswith("call_"):
            suffix = tool_use_id[len("call_") :]
            return f"fc_{suffix}", tool_use_id
        return f"fc_{tool_use_id}", f"call_{tool_use_id}"


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
