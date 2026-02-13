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


async def test_terminate_session_accepts_202_without_warning(caplog) -> None:
    transport = _transport()

    with caplog.at_level("WARNING"):
        await transport.terminate_session(cast("httpx.AsyncClient", _Client(202)))

    assert "Session termination failed" not in caplog.text


async def test_terminate_session_logs_warning_for_unexpected_status(caplog) -> None:
    transport = _transport()

    with caplog.at_level("WARNING"):
        await transport.terminate_session(cast("httpx.AsyncClient", _Client(500)))

    assert "Session termination failed: 500" in caplog.text


async def test_terminate_session_logs_warning_on_exception(caplog) -> None:
    transport = _transport()

    with caplog.at_level("WARNING"):
        await transport.terminate_session(cast("httpx.AsyncClient", _FailingClient()))

    assert "Session termination failed: network down" in caplog.text
