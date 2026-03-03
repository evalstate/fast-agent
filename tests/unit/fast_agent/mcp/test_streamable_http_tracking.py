import logging
from typing import TYPE_CHECKING, cast

import anyio
import pytest
from mcp.client.streamable_http import MAX_RECONNECTION_ATTEMPTS
from mcp.shared.message import SessionMessage

from fast_agent.mcp.streamable_http_tracking import ChannelTrackingStreamableHTTPTransport

if TYPE_CHECKING:
    import httpx

pytestmark = pytest.mark.asyncio


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _Client:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    async def delete(self, url: str, headers: dict[str, str] | None = None) -> _Response:
        del url, headers
        return _Response(self.status_code)


class _FailingClient:
    async def delete(self, url: str, headers: dict[str, str] | None = None) -> _Response:
        del url, headers
        raise RuntimeError("network down")


class _FakeEventSourceResponse:
    def raise_for_status(self) -> None:
        return None


class _DropEventSource:
    response = _FakeEventSourceResponse()

    async def aiter_sse(self):
        raise RuntimeError("stream dropped")
        yield  # pragma: no cover


class _StopEventSource:
    def __init__(self, *, on_complete) -> None:
        self.response = _FakeEventSourceResponse()
        self._on_complete = on_complete

    async def aiter_sse(self):
        self._on_complete()
        if False:  # pragma: no cover
            yield


class _ScriptedSSEConnection:
    def __init__(self, mode: str, *, on_complete=None) -> None:
        self._mode = mode
        self._on_complete = on_complete

    async def __aenter__(self):
        if self._mode == "drop":
            return _DropEventSource()
        if self._mode == "stop":
            return _StopEventSource(on_complete=self._on_complete)
        raise RuntimeError(f"Unknown scripted mode: {self._mode}")

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False


class _ScriptedTransport(ChannelTrackingStreamableHTTPTransport):
    def __init__(self, script: list[str]) -> None:
        super().__init__("https://example.com/mcp")
        self._script = script
        self.open_calls = 0

    def _open_sse_connection(self, client, method: str, url: str, *, headers: dict[str, str]):
        del client, method, url, headers
        mode = self._script[self.open_calls]
        self.open_calls += 1
        if mode == "stop":
            return _ScriptedSSEConnection(mode, on_complete=lambda: setattr(self, "session_id", None))
        return _ScriptedSSEConnection(mode)

    async def _sleep_before_reconnect(self, delay_ms: int) -> None:
        del delay_ms
        return


def _transport() -> ChannelTrackingStreamableHTTPTransport:
    transport = ChannelTrackingStreamableHTTPTransport("https://example.com/mcp")
    transport.session_id = "session-123"
    return transport


@pytest.fixture
def _capture_transport_logger(caplog):
    """Capture transport logs even when ``fast_agent`` logger propagation is disabled."""

    target_logger = logging.getLogger("fast_agent.mcp.streamable_http_tracking")
    original_level = target_logger.level
    target_logger.addHandler(caplog.handler)
    target_logger.setLevel(logging.WARNING)

    try:
        yield
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(original_level)


async def test_terminate_session_accepts_202_without_warning(caplog, _capture_transport_logger) -> None:
    transport = _transport()

    await transport.terminate_session(cast("httpx.AsyncClient", _Client(202)))

    assert "Session termination failed" not in caplog.text


async def test_terminate_session_logs_warning_for_unexpected_status(
    caplog, _capture_transport_logger
) -> None:
    transport = _transport()

    await transport.terminate_session(cast("httpx.AsyncClient", _Client(500)))

    assert "Session termination failed: 500" in caplog.text


async def test_terminate_session_logs_warning_on_exception(caplog, _capture_transport_logger) -> None:
    transport = _transport()

    await transport.terminate_session(cast("httpx.AsyncClient", _FailingClient()))

    assert "Session termination failed: network down" in caplog.text


async def test_get_stream_resets_error_counter_after_successful_retry() -> None:
    if MAX_RECONNECTION_ATTEMPTS <= 1:
        pytest.skip("Reconnect-attempt reset requires MAX_RECONNECTION_ATTEMPTS > 1")

    # Multiple stream drops should not exhaust the retry budget as long as each reconnect succeeds.
    script = ["drop"] * (MAX_RECONNECTION_ATTEMPTS + 2) + ["stop"]
    transport = _ScriptedTransport(script)
    transport.session_id = "session-123"
    read_stream_writer, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](
        10
    )

    try:
        await transport.handle_get_stream(cast("httpx.AsyncClient", object()), read_stream_writer)
    finally:
        await read_stream_writer.aclose()
        await read_stream.aclose()

    assert transport.open_calls == len(script)
