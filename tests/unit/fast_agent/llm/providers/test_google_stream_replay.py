from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from google.genai import types as google_types
from mcp_types import CallToolResult, TextContent

from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.provider.google.google_converter import GoogleConverter
from fast_agent.llm.provider.google.llm_google_native import GoogleNativeLLM
from fast_agent.llm.provider.streaming_timeouts import StreamIdleTimeoutError

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


class _IdleGoogleStream:
    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self) -> _IdleGoogleStream:
        return self

    async def __anext__(self) -> google_types.GenerateContentResponse:
        await asyncio.sleep(1)
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class _SimulatedGoogleModels:
    def __init__(self, stream: _IdleGoogleStream) -> None:
        self.stream = stream

    async def generate_content_stream(self, **_kwargs: Any) -> _IdleGoogleStream:
        return self.stream


class _SimulatedGoogleClient:
    def __init__(self, stream: _IdleGoogleStream) -> None:
        self.aio = type("_Aio", (), {"models": _SimulatedGoogleModels(stream)})()


@pytest.mark.asyncio
async def test_google_stream_enforces_between_event_idle_timeout() -> None:
    harness = _GoogleReplayHarness()
    stream = _IdleGoogleStream()

    with pytest.raises(
        StreamIdleTimeoutError,
        match="No stream events were received for 0.01 seconds",
    ):
        await harness._stream_generate_content(
            model="gemini-test",
            contents=[],
            config=google_types.GenerateContentConfig(),
            client=cast("Any", _SimulatedGoogleClient(stream)),
            timeout_seconds=0.01,
        )

    assert stream.closed


def _google_chunk(
    *,
    text: str | None = None,
    thought: bool | None = None,
    thought_signature: bytes | None = None,
    function_call: dict[str, Any] | None = None,
    finish_reason: str | None = None,
) -> google_types.GenerateContentResponse:
    part: dict[str, Any] = {}
    if text is not None:
        part["text"] = text
    if thought is not None:
        part["thought"] = thought
    if thought_signature is not None:
        part["thought_signature"] = thought_signature
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
    assert len(parts) == 2
    assert parts[0].text == "thinking"
    assert parts[0].thought is True
    assert parts[1].text == "answer"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_google_stream_tool_call_uses_final_arguments_and_closes_stream() -> None:
    harness = _GoogleReplayHarness()
    stream = _SyntheticGoogleStream(
        [
            _google_chunk(
                thought_signature=b"sig",
                function_call={
                    "id": "call_weather",
                    "name": "weather",
                    "args": {"city": "P"},
                },
            ),
            _google_chunk(
                thought_signature=b"sig",
                function_call={
                    "id": "call_weather",
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
            "payload": {"tool_name": "weather", "tool_use_id": "call_weather", "index": 0},
        },
        {
            "event_type": "delta",
            "payload": {
                "tool_name": "weather",
                "tool_use_id": "call_weather",
                "index": 0,
                "chunk": '{"city":"P"}',
            },
        },
        {
            "event_type": "delta",
            "payload": {
                "tool_name": "weather",
                "tool_use_id": "call_weather",
                "index": 0,
                "chunk": '{"city":"Paris"}',
            },
        },
        {
            "event_type": "stop",
            "payload": {"tool_name": "weather", "tool_use_id": "call_weather", "index": 0},
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
    assert parts[0].function_call.id == "call_weather"
    assert parts[0].thought_signature == b"sig"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("with_ids", [True, False])
@pytest.mark.parametrize("separate_chunks", [True, False])
async def test_gemini4_parallel_calls_preserve_arguments_ids_and_signature_placement(
    with_ids: bool,
    separate_chunks: bool,
) -> None:
    harness = _GoogleReplayHarness()
    calls = [
        google_types.Part(
            function_call=google_types.FunctionCall(
                id="call_paris" if with_ids else None,
                name="weather",
                args={"city": "Paris"},
            ),
            thought_signature=b"first-call-signature",
        ),
        google_types.Part(
            function_call=google_types.FunctionCall(
                id="call_london" if with_ids else None,
                name="weather",
                args={"city": "London"},
            ),
        ),
    ]
    stream = _SyntheticGoogleStream(
        [
            google_types.GenerateContentResponse(
                candidates=[
                    google_types.Candidate(
                        content=google_types.Content(role="model", parts=parts),
                        finish_reason=(
                            google_types.FinishReason.STOP if parts[-1] is calls[-1] else None
                        ),
                    )
                ]
            )
            for parts in ([calls[:1], calls[1:]] if separate_chunks else [calls])
        ]
    )
    response = await harness._consume_google_stream(stream, model="gemini-4-hydrogen")
    assert response is not None
    assert response.candidates is not None
    content = response.candidates[0].content
    assert content is not None
    assert content.parts is not None
    assert len(content.parts) == 2
    first, second = content.parts
    assert first.function_call is not None
    assert second.function_call is not None
    assert first.function_call.args == {"city": "Paris"}
    assert second.function_call.args == {"city": "London"}
    assert first.function_call.id != second.function_call.id
    if with_ids:
        assert first.function_call.id == "call_paris"
        assert second.function_call.id == "call_london"
    assert first.thought_signature == b"first-call-signature"
    assert second.thought_signature is None
    assert [event["event_type"] for event in harness.tool_events] == [
        "start",
        "delta",
        "stop",
        "start",
        "delta",
        "stop",
    ]
    provider_calls = harness._provider_google_tool_calls(content)
    results = GoogleConverter().convert_function_results_to_google(
        [
            (
                name,
                call_id,
                CallToolResult(
                    content=[TextContent(type="text", text=str(args["city"]))],
                    is_error=index == 1,
                ),
            )
            for index, (call_id, name, args) in enumerate(provider_calls)
        ]
    )
    assert len(results) == 1
    assert results[0].role == "user"
    assert results[0].parts is not None
    responses = [part.function_response for part in results[0].parts]
    assert all(response is not None for response in responses)
    for function_response, (call_id, name, _) in zip(responses, provider_calls, strict=True):
        assert function_response is not None
        assert function_response.id == call_id
        assert function_response.name == name
    assert responses[0] is not None
    assert responses[0].response == {"result": "Paris"}
    assert responses[1] is not None
    assert responses[1].response == {"error": "London"}


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("thought", [True, False])
async def test_gemini4_stream_keeps_signed_text_parts_separate(thought: bool) -> None:
    harness = _GoogleReplayHarness()
    response = await harness._consume_google_stream(
        _SyntheticGoogleStream(
            [
                _google_chunk(text="unsigned", thought=thought),
                _google_chunk(text="signed", thought=thought, thought_signature=b"sig"),
                _google_chunk(text="also signed", thought=thought, thought_signature=b"sig"),
                _google_chunk(text="tail", thought=thought, finish_reason="STOP"),
            ]
        ),
        model="gemini-4-carbon",
    )
    assert response is not None
    assert response.candidates is not None
    content = response.candidates[0].content
    assert content is not None
    assert content.parts is not None
    assert [part.text for part in content.parts] == ["unsigned", "signed", "also signed", "tail"]
    assert [part.thought_signature for part in content.parts] == [None, b"sig", b"sig", None]
    assert all(bool(part.thought) is thought for part in content.parts)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", None])
@pytest.mark.parametrize("thought", [True, False, None])
async def test_gemini4_stream_preserves_empty_signature_part(
    text: str | None, thought: bool | None
) -> None:
    harness = _GoogleReplayHarness()
    signature_chunk = _google_chunk(
        text=text,
        thought=thought,
        thought_signature=b"last-signature",
        finish_reason="STOP",
    )
    response = await harness._consume_google_stream(
        _SyntheticGoogleStream(
            [
                _google_chunk(text="answer"),
                signature_chunk,
            ]
        ),
        model="gemini-4-argon",
    )
    assert response is not None
    assert response.candidates is not None
    content = response.candidates[0].content
    assert content is not None
    assert content.parts is not None
    assert [part.text for part in content.parts] == ["answer", text]
    assert signature_chunk.candidates is not None
    signature_content = signature_chunk.candidates[0].content
    assert signature_content is not None
    assert signature_content.parts is not None
    assert content.parts[-1] == signature_content.parts[0]
