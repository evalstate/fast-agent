from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.provider.anthropic.beta_types import (
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
)
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "tests" / "support").is_dir()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_replay = importlib.import_module("tests.support.llm_trace_replay")


class _AnthropicReplayHarness(AnthropicLLM):
    def __init__(self) -> None:
        self.logger = get_logger("test.anthropic.replay")
        self.name = "test"
        self.stream_events: list[dict[str, Any]] = []
        self.tool_events: list[dict[str, Any]] = []

    def _notify_tool_stream_listeners(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.tool_events.append(
            {
                "event_type": event_type,
                "payload": payload or {},
            }
        )

    def _notify_stream_listeners(self, chunk: Any) -> None:
        self.stream_events.append(
            {
                "text": getattr(chunk, "text", ""),
                "is_reasoning": bool(getattr(chunk, "is_reasoning", False)),
            }
        )

    def _update_streaming_progress(
        self,
        content: str,
        model: str,
        estimated_tokens: int,
    ) -> int:
        del content, model
        return estimated_tokens

    def chat_turn(self) -> int:
        return 1


class _SyntheticAnthropicStream:
    def __init__(
        self,
        events: list[Any],
        *,
        final_message: Message | None = None,
        final_error: Exception | None = None,
    ) -> None:
        self._events = events
        self._index = 0
        self._final_message = final_message
        self._final_error = final_error

    def __aiter__(self) -> _SyntheticAnthropicStream:
        self._index = 0
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event

    async def get_final_message(self) -> Message:
        if self._final_error is not None:
            raise self._final_error
        if self._final_message is None:
            raise RuntimeError("missing final message")
        return self._final_message


def _anthropic_message(*, text: str = "", stop_reason: str = "end_turn") -> Message:
    content = [{"type": "text", "text": text}] if text else []
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": "claude-sonnet-4-6",
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
    )


def _fixture_params() -> list[Any]:
    return [
        pytest.param(case, fixture, id=case.id)
        for case, fixture in _replay.load_replay_fixtures("anthropic")
    ]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(("case", "fixture"), _fixture_params())
async def test_anthropic_stream_replay(case: Any, fixture: Any) -> None:
    harness = _AnthropicReplayHarness()
    message, _thinking_segments, _streamed_text_segments = await harness._process_stream(
        fixture.anthropic_stream(),
        model=fixture.meta()["resolved_model"],
        capture_filename=None,
    )

    summary = _replay.summarize_anthropic_replay(
        message=message,
        stream_events=harness.stream_events,
        tool_events=harness.tool_events,
    )
    _replay.assert_replay_case(case, fixture, summary)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_anthropic_server_tool_start_stop_without_delta() -> None:
    harness = _AnthropicReplayHarness()
    stream = _SyntheticAnthropicStream(
        [
            RawContentBlockStartEvent.model_validate(
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {"query": "capital of France"},
                        "caller": {"type": "direct"},
                    },
                }
            ),
            RawContentBlockStopEvent.model_validate(
                {
                    "type": "content_block_stop",
                    "index": 2,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {"query": "capital of France"},
                        "caller": {"type": "direct"},
                    },
                }
            ),
        ],
        final_message=_anthropic_message(),
    )

    message, thinking_segments, streamed_text_segments = await harness._process_stream(
        cast("Any", stream),
        model="claude-sonnet-4-6",
        capture_filename=None,
    )

    assert thinking_segments == []
    assert streamed_text_segments == []
    assert message.stop_reason == "end_turn"
    assert harness.tool_events == [
        {
            "event_type": "start",
            "payload": {
                "tool_name": "web_search",
                "presentation_family": "web_search",
                "preserve_details": False,
                "tool_display_name": "Searching the web",
                "chunk": '{"query": "capital of France"}',
                "tool_use_id": "srvtoolu_1",
                "index": 2,
            },
        },
        {
            "event_type": "stop",
            "payload": {
                "tool_name": "web_search",
                "presentation_family": "web_search",
                "preserve_details": False,
                "tool_display_name": "Searching the web",
                "tool_use_id": "srvtoolu_1",
                "index": 2,
            },
        },
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_anthropic_mcp_tool_start_stop_without_delta() -> None:
    harness = _AnthropicReplayHarness()
    stream = _SyntheticAnthropicStream(
        [
            RawContentBlockStartEvent.model_validate(
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {
                        "type": "mcp_tool_use",
                        "id": "mcptoolu_1",
                        "name": "hf_hub_query",
                        "server_name": "huggingface_mcp",
                        "input": {},
                    },
                }
            ),
            RawContentBlockStopEvent.model_validate(
                {
                    "type": "content_block_stop",
                    "index": 2,
                    "content_block": {
                        "type": "mcp_tool_use",
                        "id": "mcptoolu_1",
                        "name": "hf_hub_query",
                        "server_name": "huggingface_mcp",
                        "input": {"message": "top models"},
                    },
                }
            ),
        ],
        final_message=_anthropic_message(),
    )

    message, thinking_segments, streamed_text_segments = await harness._process_stream(
        cast("Any", stream),
        model="claude-sonnet-4-6",
        capture_filename=None,
    )

    assert thinking_segments == []
    assert streamed_text_segments == []
    assert message.stop_reason == "end_turn"
    assert harness.tool_events == [
        {
            "event_type": "start",
            "payload": {
                "tool_name": "huggingface_mcp/hf_hub_query",
                "server_name": "huggingface_mcp",
                "presentation_family": "remote_tool",
                "preserve_details": True,
                "tool_display_name": "remote tool: hf_hub_query",
                "chunk": "{}",
                "tool_use_id": "mcptoolu_1",
                "index": 2,
            },
        },
        {
            "event_type": "replace",
            "payload": {
                "tool_name": "huggingface_mcp/hf_hub_query",
                "server_name": "huggingface_mcp",
                "presentation_family": "remote_tool",
                "preserve_details": True,
                "tool_use_id": "mcptoolu_1",
                "index": 2,
                "chunk": '{"message": "top models"}',
            },
        },
        {
            "event_type": "stop",
            "payload": {
                "tool_name": "huggingface_mcp/hf_hub_query",
                "server_name": "huggingface_mcp",
                "presentation_family": "remote_tool",
                "preserve_details": True,
                "tool_display_name": "remote tool: hf_hub_query",
                "tool_use_id": "mcptoolu_1",
                "index": 2,
            },
        },
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_anthropic_stream_validation_fallback_uses_streamed_text() -> None:
    harness = _AnthropicReplayHarness()
    stream = _SyntheticAnthropicStream(
        [
            RawContentBlockDeltaEvent.model_validate(
                {
                    "type": "content_block_delta",
                    "index": 7,
                    "delta": {
                        "type": "text_delta",
                        "text": "hello",
                    },
                }
            )
        ],
        final_error=ValueError("BetaTextBlock input should be a valid string for text"),
    )

    message, thinking_segments, streamed_text_segments = await harness._process_stream(
        cast("Any", stream),
        model="claude-sonnet-4-6",
        capture_filename=None,
    )

    assert thinking_segments == []
    assert streamed_text_segments == ["hello"]
    assert message.id == "msg_stream_fallback"
    assert message.stop_reason == "end_turn"
    assert harness.stream_events == [{"text": "hello", "is_reasoning": False}]
    assert harness.tool_events == [
        {
            "event_type": "text",
            "payload": {"chunk": "hello", "index": 7},
        }
    ]
