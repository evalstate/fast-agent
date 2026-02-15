import logging
from typing import TYPE_CHECKING, cast

import pytest

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
