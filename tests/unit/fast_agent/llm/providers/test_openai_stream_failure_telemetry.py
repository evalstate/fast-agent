"""Chat-completions streaming diagnostics.

The chat-completions transport wraps its stream with the shared idle-timeout
helper, so stream timing exists for every provider on that path (HuggingFace
routes included). These tests cover what the transport does with it: record how
far a broken stream got, and report extended inter-event gaps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from openai.types.chat import ChatCompletionChunk

from fast_agent.context import Context
from fast_agent.core.logging.logger import Logger
from fast_agent.llm.provider.openai.llm_openai import (
    OpenAILLM,
    _OpenAICompletionRequest,
)
from fast_agent.llm.provider.streaming_timeouts import with_stream_idle_timeout
from fast_agent.llm.request_params import RequestParams

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from fast_agent.core.logging.events import EventContext, EventType

MODEL = "zai-org/glm-5.2"


class _RecordingLogger(Logger):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def event(
        self,
        etype: EventType,
        ename: str | None,
        message: str,
        context: EventContext | None,
        data: dict,
    ) -> None:
        self.events.append({"type": etype, "message": message, "data": data})


def _text_chunk(text: str) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
    )


class _TruncatedStream:
    """Yield chunks, then fail the way a truncated chunked response does."""

    def __init__(self, chunks: list[ChatCompletionChunk]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> _TruncatedStream:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            return next(self._chunks)
        except StopIteration:
            raise httpx.ReadError(
                "Response payload is not completed: <TransferEncodingError: 400, "
                "message='Not enough data to satisfy transfer length header.'>"
            ) from None


class _SimulatedCompletions:
    def __init__(self, stream: _TruncatedStream) -> None:
        self._stream = stream

    async def create(self, **_kwargs: Any) -> _TruncatedStream:
        return self._stream


class _SimulatedClient:
    def __init__(self, stream: _TruncatedStream) -> None:
        self.chat = type("_Chat", (), {"completions": _SimulatedCompletions(stream)})()


class _SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _request(llm: OpenAILLM) -> _OpenAICompletionRequest:
    return _OpenAICompletionRequest(
        params=RequestParams(model=MODEL, streaming_timeout=120.0),
        model_name=MODEL,
        messages=[],
        arguments={"model": MODEL, "messages": [], "stream": True},
    )


@pytest.mark.asyncio
async def test_truncated_chat_stream_records_events_received() -> None:
    llm = OpenAILLM(context=Context(), model=MODEL)
    client = _SimulatedClient(_TruncatedStream([_text_chunk("par"), _text_chunk("tial")]))

    with pytest.raises(httpx.ReadError):
        await llm._create_openai_streaming_response(
            cast("AsyncOpenAI", client),
            _request(llm),
            None,
        )

    # Two chunks arrived before the payload was truncated; retry telemetry needs
    # that count to tell an early failure from a nearly-complete one.
    assert llm._stream_failure_events_received == 2


@pytest.mark.asyncio
async def test_successful_chat_stream_leaves_no_failure_telemetry() -> None:
    llm = OpenAILLM(context=Context(), model=MODEL)
    llm._stream_failure_events_received = 7

    client = _SimulatedClient(_TruncatedStream([]))
    with pytest.raises(httpx.ReadError):
        await llm._create_openai_streaming_response(
            cast("AsyncOpenAI", client),
            _request(llm),
            None,
        )

    assert llm._stream_failure_events_received == 0


@pytest.mark.asyncio
async def test_extended_inter_event_gap_is_reported() -> None:
    llm = OpenAILLM(context=Context(), model=MODEL)
    logger = _RecordingLogger()
    llm.logger = logger

    # 2s to first event, then a 12s gap, then 3s.
    timed_stream = with_stream_idle_timeout(
        _TruncatedStream([_text_chunk("a"), _text_chunk("b"), _text_chunk("c")]),
        idle_timeout_seconds=None,
        monotonic=_SequenceClock([0.0, 2.0, 5.0, 17.0, 20.0, 23.0]),
    )
    for _ in range(3):
        await timed_stream.__anext__()

    llm._record_stream_gap_observation(timed_stream.timing, model=MODEL)

    (event,) = [e for e in logger.events if "inter-event gap" in e["message"]]
    timing = event["data"]["data"]["stream_timing"]
    assert timing["inter_event_waits_over_10s"] == 1
    assert timing["max_inter_event_wait_ms"] == 12000.0
    assert timing["events_received"] == 3


@pytest.mark.asyncio
async def test_steady_chat_stream_reports_no_gap() -> None:
    llm = OpenAILLM(context=Context(), model=MODEL)
    logger = _RecordingLogger()
    llm.logger = logger

    timed_stream = with_stream_idle_timeout(
        _TruncatedStream([_text_chunk("a"), _text_chunk("b")]),
        idle_timeout_seconds=None,
        monotonic=_SequenceClock([0.0, 0.5, 1.0, 1.5]),
    )
    for _ in range(2):
        await timed_stream.__anext__()

    llm._record_stream_gap_observation(timed_stream.timing, model=MODEL)

    assert logger.events == []
