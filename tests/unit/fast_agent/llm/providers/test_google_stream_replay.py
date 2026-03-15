from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest
from google.genai import types as google_types

from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.provider.google.llm_google_native import GoogleNativeLLM

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "tests" / "support").is_dir()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_replay = importlib.import_module("tests.support.llm_trace_replay")


class _GoogleReplayHarness(GoogleNativeLLM):
    def __init__(self) -> None:
        self.logger = get_logger("test.google.replay")
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


class _SyntheticGoogleStream:
    def __init__(self, events: list[google_types.GenerateContentResponse]) -> None:
        self._events = events
        self._index = 0
        self.closed = False

    def __aiter__(self) -> _SyntheticGoogleStream:
        self._index = 0
        return self

    async def __anext__(self) -> google_types.GenerateContentResponse:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event

    async def aclose(self) -> None:
        self.closed = True


def _google_chunk(
    *,
    text: str | None = None,
    thought: bool | None = None,
    function_call: dict[str, Any] | None = None,
    finish_reason: str | None = None,
) -> google_types.GenerateContentResponse:
    part: dict[str, Any] = {}
    if text is not None:
        part["text"] = text
    if thought is not None:
        part["thought"] = thought
    if function_call is not None:
        part["function_call"] = function_call
    return google_types.GenerateContentResponse.model_validate(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [part],
                    },
                    "finish_reason": finish_reason,
                }
            ]
        }
    )


def _fixture_params() -> list[Any]:
    return [
        pytest.param(case, fixture, id=case.id)
        for case, fixture in _replay.load_replay_fixtures("google")
    ]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(("case", "fixture"), _fixture_params())
async def test_google_stream_replay(case: Any, fixture: Any) -> None:
    harness = _GoogleReplayHarness()
    final_response = await harness._consume_google_stream(
        fixture.google_stream(),
        model=fixture.meta()["resolved_model"],
    )

    summary = _replay.summarize_google_replay(
        final_response=final_response,
        stream_events=harness.stream_events,
        tool_events=harness.tool_events,
    )
    _replay.assert_replay_case(case, fixture, summary)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_google_stream_plain_text_reconstructs_final_response() -> None:
    harness = _GoogleReplayHarness()
    stream = _SyntheticGoogleStream(
        [
            _google_chunk(text="Hel"),
            _google_chunk(text="lo", finish_reason="STOP"),
        ]
    )

    final_response = await harness._consume_google_stream(
        stream,
        model="gemini-2.0-flash",
    )

    assert stream.closed is True
    assert harness.stream_events == [
        {"text": "Hel", "is_reasoning": False},
        {"text": "lo", "is_reasoning": False},
    ]
    assert harness.tool_events == [
        {"event_type": "text", "payload": {"chunk": "Hel"}},
        {"event_type": "text", "payload": {"chunk": "lo"}},
    ]
    assert final_response is not None
    candidates = final_response.candidates or []
    assert candidates
    content = candidates[0].content
    assert content is not None
    parts = content.parts or []
    assert len(parts) == 1
    assert parts[0].text == "Hello"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_google_stream_reasoning_emits_stream_events_without_text_tool_events() -> None:
    harness = _GoogleReplayHarness()
    stream = _SyntheticGoogleStream(
        [
            _google_chunk(text="thinking", thought=True),
            _google_chunk(text="answer", finish_reason="STOP"),
        ]
    )

    final_response = await harness._consume_google_stream(
        stream,
        model="gemini-2.0-flash",
    )

    assert stream.closed is True
    assert harness.stream_events == [
        {"text": "thinking", "is_reasoning": True},
        {"text": "answer", "is_reasoning": False},
    ]
    assert harness.tool_events == [
        {"event_type": "text", "payload": {"chunk": "answer"}},
    ]
    assert final_response is not None
    candidates = final_response.candidates or []
    assert candidates
    content = candidates[0].content
    assert content is not None
    parts = content.parts or []
    assert len(parts) == 1
    assert parts[0].text == "answer"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_google_stream_tool_call_uses_final_arguments_and_closes_stream() -> None:
    harness = _GoogleReplayHarness()
    stream = _SyntheticGoogleStream(
        [
            _google_chunk(
                function_call={
                    "name": "weather",
                    "args": {"city": "P"},
                }
            ),
            _google_chunk(
                function_call={
                    "name": "weather",
                    "args": {"city": "Paris"},
                },
                finish_reason="STOP",
            ),
        ]
    )

    final_response = await harness._consume_google_stream(
        stream,
        model="gemini-2.0-flash",
    )

    assert stream.closed is True
    assert harness.tool_events == [
        {
            "event_type": "start",
            "payload": {"tool_name": "weather", "tool_use_id": "tool_1_0", "index": 0},
        },
        {
            "event_type": "delta",
            "payload": {
                "tool_name": "weather",
                "tool_use_id": "tool_1_0",
                "index": 0,
                "chunk": '{"city":"P"}',
            },
        },
        {
            "event_type": "delta",
            "payload": {
                "tool_name": "weather",
                "tool_use_id": "tool_1_0",
                "index": 0,
                "chunk": '{"city":"Paris"}',
            },
        },
        {
            "event_type": "stop",
            "payload": {"tool_name": "weather", "tool_use_id": "tool_1_0", "index": 0},
        },
    ]
    assert final_response is not None
    candidates = final_response.candidates or []
    assert candidates
    content = candidates[0].content
    assert content is not None
    parts = content.parts or []
    assert len(parts) == 1
    assert parts[0].function_call is not None
    assert parts[0].function_call.name == "weather"
    assert parts[0].function_call.args == {"city": "Paris"}
