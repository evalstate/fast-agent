"""Anthropic Messages streaming diagnostics (shared with Copilot Messages).

The stream is wrapped with the shared timing helper; these tests cover what
the provider does with the resulting timing: every failed attempt is logged with
trial-local call/attempt identifiers (stream-start failures distinguished from idle
established streams), and successful timing lands in the provider diagnostics channel.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from anthropic import Timeout
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaUsage
from httpx2 import ReadTimeout
from mcp_types import TextContent

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.constants import FAST_AGENT_RETRY
from fast_agent.context import Context
from fast_agent.core.logging.logger import Logger
from fast_agent.llm.provider.anthropic.llm_anthropic import (
    ANTHROPIC_CACHE_DIAGNOSTICS_CHANNEL,
    AnthropicLLM,
)
from fast_agent.llm.provider.copilot.messages import CopilotMessagesLLM
from fast_agent.llm.provider.streaming_timeouts import StreamTiming

if TYPE_CHECKING:
    from fast_agent.core.logging.events import EventContext, EventType

MODEL = "claude-test"
COPILOT_MODEL = "claude-sonnet-5"
HANG = "hang"


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
        self.events.append({"type": etype, "message": message, "data": data["data"]})

    def stream_attempts(self) -> list[dict[str, Any]]:
        return [e["data"] for e in self.events if e["message"] == "Provider stream attempt failed"]


def _message() -> BetaMessage:
    return BetaMessage(
        id="msg_1",
        type="message",
        role="assistant",
        content=[BetaTextBlock(type="text", text="hi")],
        model=MODEL,
        stop_reason="end_turn",
        usage=BetaUsage(input_tokens=1, output_tokens=1),
    )


class _ScriptedStream:
    """Anthropic-style stream context manager driven by a script.

    Script items are opaque events (yielded as-is), ``HANG`` (HTTP read timeout),
    or an exception (raised mid-stream). ``start=HANG`` never finishes entering.
    """

    def __init__(self, script: list[Any], *, start: Any = None) -> None:
        self._script = script
        self._start = start

    async def __aenter__(self) -> _ScriptedStream:
        if self._start is HANG:
            await asyncio.Event().wait()
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def __aiter__(self):
        for item in self._script:
            if item is HANG:
                raise ReadTimeout("HTTP stream stalled")
            if isinstance(item, BaseException):
                raise item
            yield item

    async def get_final_message(self) -> BetaMessage:
        return _message()


class _Harness:
    """An Anthropic-family LLM, a recording logger and a client that plays scripted streams."""

    def __init__(
        self,
        scripts: list[_ScriptedStream],
        *,
        llm_type: type[AnthropicLLM] = AnthropicLLM,
        retries: int = 0,
    ) -> None:
        context = Context()
        context.config = Settings(anthropic=AnthropicSettings(api_key="test-key"))
        model = MODEL if llm_type is AnthropicLLM else COPILOT_MODEL
        self.llm = llm_type(context=context, model=model)
        self.llm.retry_count = retries
        self.llm.retry_backoff_seconds = 0.0
        self.logger = _RecordingLogger()
        self.llm.logger = self.logger
        self.scripts = list(scripts)
        self.client = SimpleNamespace(
            timeout=Timeout(600, connect=5),
            beta=SimpleNamespace(
                messages=SimpleNamespace(stream=lambda **_: self.scripts.pop(0)),
            ),
        )

    async def run(self, timeout: float | None, *_args: object) -> None:
        await self.llm._execute_anthropic_stream(
            anthropic=self.client,
            arguments={},
            model=MODEL,
            capture_filename=None,
            timeout_seconds=timeout,
        )


def _diagnostics(llm: AnthropicLLM, *, cache: bool = False) -> dict[str, Any]:
    channels = llm._anthropic_response_channels(
        _message(), MODEL, [], None, cache_diagnostics_enabled=cache
    )
    assert channels is not None
    [block] = channels[ANTHROPIC_CACHE_DIAGNOSTICS_CHANNEL]
    assert isinstance(block, TextContent)
    return json.loads(block.text)


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_type", [AnthropicLLM, CopilotMessagesLLM])
async def test_successful_stream_timing_reaches_diagnostics(llm_type: type[AnthropicLLM]) -> None:
    h = _Harness([_ScriptedStream(["a", "b", "c"])], llm_type=llm_type)

    await h.run(5.0)

    assert h.logger.events == []
    timing = _diagnostics(h.llm)["stream_timing"]
    assert timing["events_received"] == 3
    assert timing["timed_out"] is False
    assert timing["inter_event_waits_over_10s"] == 0
    assert "timed_out_wait_ms" not in timing


@pytest.mark.asyncio
async def test_disabled_timeout_still_records_timing() -> None:
    h = _Harness([_ScriptedStream(["a", "b"])])

    await h.run(None)

    assert h.logger.events == []
    assert _diagnostics(h.llm)["stream_timing"]["events_received"] == 2


def test_long_gap_success_warns_and_merges_with_cache_diagnostics() -> None:
    h = _Harness([])
    h.llm._record_stream_outcome(
        StreamTiming(
            events_received=4,
            first_event_wait_seconds=0.5,
            max_inter_event_wait_seconds=12.5,
            inter_event_waits_over_threshold=1,
            timed_out_wait_seconds=None,
        ),
        error=None,
        model=MODEL,
        timeout_seconds=150.0,
    )

    [warning] = h.logger.events
    assert warning["type"] == "warning"
    assert warning["data"]["stream_timing"]["max_inter_event_wait_ms"] == 12500.0
    assert warning["data"]["stream_timing"]["inter_event_waits_over_10s"] == 1
    diagnostics = _diagnostics(h.llm, cache=True)
    assert diagnostics["kind"] == "anthropic_cache_diagnosis"
    assert diagnostics["stream_timing"]["events_received"] == 4


@pytest.mark.asyncio
async def test_zero_event_read_timeout_is_an_established_stream_failure() -> None:
    h = _Harness([_ScriptedStream([HANG])])

    with pytest.raises(ReadTimeout):
        await h.run(0.01)

    [attempt] = h.logger.stream_attempts()
    assert attempt["phase"] == "stream"
    assert attempt["error_type"] == "ReadTimeout"
    assert attempt["timeout_seconds"] == 0.01
    timing = attempt["stream_timing"]
    assert timing["events_received"] == 0
    assert timing["timed_out"] is False
    assert "timed_out_wait_ms" not in timing
    assert timing["first_event_wait_ms"] is None
    assert h.llm._stream_failure_events_received == 0
    assert set(attempt) == {
        "model",
        "phase",
        "timeout_seconds",
        "call",
        "attempt",
        "max_attempts",
        "stream_timing",
        "error_type",
    }


@pytest.mark.asyncio
async def test_stream_start_timeout_is_distinguished_from_idle() -> None:
    h = _Harness([_ScriptedStream([], start=HANG)])

    with pytest.raises(TimeoutError, match="did not start"):
        await h.run(0.01)

    [attempt] = h.logger.stream_attempts()
    assert attempt["phase"] == "start"
    assert attempt["error_type"] == "TimeoutError"
    assert attempt["stream_timing"]["events_received"] == 0
    assert attempt["stream_timing"]["timed_out"] is True
    assert h.llm._stream_failure_events_received is None


@pytest.mark.asyncio
async def test_mid_stream_timeout_keeps_event_count() -> None:
    h = _Harness([_ScriptedStream(["a", "b", HANG])])

    with pytest.raises(ReadTimeout):
        await h.run(0.01)

    [attempt] = h.logger.stream_attempts()
    assert attempt["stream_timing"]["events_received"] == 2
    assert attempt["stream_timing"]["first_event_wait_ms"] is not None
    assert h.llm._stream_failure_events_received == 2


@pytest.mark.asyncio
async def test_mid_stream_error_is_recorded_without_timeout() -> None:
    h = _Harness([_ScriptedStream(["a", RuntimeError("reset")])])

    with pytest.raises(RuntimeError):
        await h.run(0.5)

    [attempt] = h.logger.stream_attempts()
    assert attempt["phase"] == "stream"
    assert attempt["error_type"] == "RuntimeError"
    assert attempt["stream_timing"] == {
        "events_received": 1,
        "first_event_wait_ms": attempt["stream_timing"]["first_event_wait_ms"],
        "max_inter_event_wait_ms": None,
        "inter_event_waits_over_10s": 0,
        "timed_out": False,
    }


@pytest.mark.asyncio
async def test_retry_recovery_identifies_attempts_and_clears_stale_timing() -> None:
    h = _Harness([_ScriptedStream(["a", HANG]), _ScriptedStream(["a", "b", "c"])], retries=2)

    response = await h.llm._execute_with_retry(h.run, 0.01)

    assert response is None
    [failed] = h.logger.stream_attempts()
    assert (failed["call"], failed["attempt"], failed["max_attempts"]) == (1, 1, 3)
    assert h.llm._last_stream_timing is not None
    assert h.llm._last_stream_timing["events_received"] == 3


@pytest.mark.asyncio
async def test_retry_exhaustion_retains_every_attempt() -> None:
    h = _Harness([_ScriptedStream([HANG]), _ScriptedStream(["a", HANG])], retries=1)

    with pytest.raises(ReadTimeout):
        await h.llm._execute_with_retry(h.run, 0.01)

    first, last = h.logger.stream_attempts()
    assert (first["call"], first["attempt"], first["max_attempts"]) == (1, 1, 2)
    assert (last["call"], last["attempt"], last["max_attempts"]) == (1, 2, 2)
    assert first["stream_timing"]["events_received"] == 0
    assert last["stream_timing"]["events_received"] == 1
    assert h.llm._last_stream_timing is None

    # A later call gets a fresh call id so its attempts cannot be confused with the first.
    h.scripts += [_ScriptedStream([HANG]), _ScriptedStream([HANG])]
    with pytest.raises(ReadTimeout):
        await h.llm._execute_with_retry(h.run, 0.01)
    assert h.logger.stream_attempts()[-1]["call"] == 2


@pytest.mark.asyncio
async def test_retry_channel_and_stream_logs_agree_on_progress() -> None:
    h = _Harness([_ScriptedStream(["a", "b", HANG]), _ScriptedStream(["a"])], retries=1)

    async def attempt(_messages: object):
        await h.run(0.01)
        return await h.llm._finalize_anthropic_response(
            response=_message(),
            model=MODEL,
            messages=[],
            thinking_segments=[],
            streamed_text_segments=[],
            structured_mode=None,
            structured_model=None,
        )

    response = await h.llm._execute_with_retry(attempt, [])

    channels = response.channels or {}
    [retry_block] = channels[FAST_AGENT_RETRY]
    assert isinstance(retry_block, TextContent)
    [retry] = json.loads(retry_block.text)["retries"]
    [failed] = h.logger.stream_attempts()
    assert retry["stream_events_received"] == failed["stream_timing"]["events_received"] == 2
    [diag_block] = channels[ANTHROPIC_CACHE_DIAGNOSTICS_CHANNEL]
    assert isinstance(diag_block, TextContent)
    assert json.loads(diag_block.text)["stream_timing"]["events_received"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_type", [AnthropicLLM, CopilotMessagesLLM])
@pytest.mark.parametrize("mode", ["pings", "stall", "disabled"])
async def test_sdk_http_read_timeout_and_filtered_pings(
    llm_type: type[AnthropicLLM], mode: str
) -> None:
    """Use real SDK SSE parsing and HTTP reads, not parsed-event test doubles."""
    from anthropic import AsyncAnthropic
    from httpx2 import AsyncClient, Request

    def sse(event: str, data: dict[str, object]) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    start_message = _message().model_dump()
    start_message.update(content=[], stop_reason=None)
    start = sse("message_start", {"type": "message_start", "message": start_message})
    stop = sse("message_stop", {"type": "message_stop"})
    server_done = asyncio.Event()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            for line in headers.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    await reader.readexactly(int(line.split(b":", 1)[1]))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
            )
            # Exercise both the first-event wait and a later parsed-event gap.
            for event in (start, stop):
                for _ in range(8):
                    await asyncio.sleep(0.05)
                    if mode == "pings":
                        writer.write(sse("ping", {"type": "ping"}))
                        await writer.drain()
                writer.write(event)
                await writer.drain()
        except ConnectionError:
            # Expected when the read timeout closes the client connection.
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            server_done.set()

    request_timeouts: list[object] = []

    async def record_timeout(request: Request) -> None:
        request_timeouts.append(request.extensions["timeout"])

    h = _Harness([], llm_type=llm_type)
    client_timeout = Timeout(connect=1, read=0.1, write=2, pool=3)
    timeout_seconds = None if mode == "disabled" else 0.2
    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with (
        server,
        AsyncAnthropic(
            api_key="test-key",
            base_url=f"http://127.0.0.1:{port}",
            timeout=client_timeout,
            max_retries=0,
            http_client=AsyncClient(event_hooks={"request": [record_timeout]}),
        ) as client,
    ):
        try:
            call = h.llm._execute_anthropic_stream(
                anthropic=client,
                arguments={
                    "model": MODEL,
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                model=MODEL,
                capture_filename=None,
                timeout_seconds=timeout_seconds,
            )
            if mode == "stall":
                with pytest.raises(ReadTimeout):
                    await asyncio.wait_for(call, 3)
                assert h.llm._stream_failure_events_received == 0
            else:
                response, _, _ = await asyncio.wait_for(call, 3)
                assert response.id == "msg_1"
                timing = _diagnostics(h.llm)["stream_timing"]
                # Pings are absent from parsed events, despite keeping reads alive.
                assert timing["events_received"] == 2
                assert timing["first_event_wait_ms"] >= 200
                assert timing["max_inter_event_wait_ms"] >= 200
            assert request_timeouts == [
                {"connect": 1, "read": timeout_seconds, "write": 2, "pool": 3}
            ]
            assert client.timeout is client_timeout
            assert client_timeout.read == 0.1
        finally:
            await asyncio.wait_for(server_done.wait(), 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_timeout", [None, 7.0, Timeout(9, connect=1)])
@pytest.mark.parametrize("streaming_timeout", [None, 0.5])
async def test_request_timeout_normalizes_sdk_client_defaults(
    client_timeout: float | Timeout | None, streaming_timeout: float | None
) -> None:
    from anthropic import AsyncAnthropic
    from httpx2 import AsyncClient, MockTransport, Request, Response

    async def respond(request: Request) -> Response:
        expected = Timeout(client_timeout)
        expected.read = streaming_timeout
        assert request.extensions["timeout"] == expected.as_dict()
        message = {"type": "message_start", "message": _message().model_dump()}
        return Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                f"event: message_start\ndata: {json.dumps(message)}\n\n"
                'event: message_stop\ndata: {"type":"message_stop"}\n\n'
            ),
        )

    h = _Harness([])
    async with AsyncAnthropic(
        api_key="test-key",
        timeout=client_timeout,
        http_client=AsyncClient(transport=MockTransport(respond)),
    ) as client:
        response, _, _ = await h.llm._execute_anthropic_stream(
            anthropic=client,
            arguments={"model": MODEL, "max_tokens": 1, "messages": []},
            model=MODEL,
            capture_filename=None,
            timeout_seconds=streaming_timeout,
        )
        assert response.id == "msg_1"
        assert client.timeout == client_timeout
