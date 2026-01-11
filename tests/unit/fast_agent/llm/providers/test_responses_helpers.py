import base64
import json

import pytest
from mcp.types import ImageContent, TextContent
from openai import AsyncOpenAI
from openai.types.responses import ResponseFunctionToolCall

from fast_agent.constants import OPENAI_REASONING_ENCRYPTED
from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.provider.openai.responses_content import ResponsesContentMixin
from fast_agent.llm.provider.openai.responses_files import ResponsesFileMixin
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
