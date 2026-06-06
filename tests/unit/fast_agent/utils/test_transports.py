from __future__ import annotations

from fast_agent.utils.transports import (
    MCP_CLIENT_TRANSPORTS,
    MCP_REMOTE_TRANSPORTS,
    is_mcp_client_transport,
    uses_mcp_remote_transport,
    uses_protocol_stdio,
)


def test_mcp_transport_groups_are_derived_from_literals() -> None:
    assert MCP_CLIENT_TRANSPORTS == frozenset({"stdio", "sse", "http"})
    assert MCP_REMOTE_TRANSPORTS == frozenset({"sse", "http"})


def test_mcp_client_transport_identifies_supported_client_transports() -> None:
    assert is_mcp_client_transport("stdio")
    assert is_mcp_client_transport("sse")
    assert is_mcp_client_transport("http")
    assert is_mcp_client_transport(" HTTP ")
    assert not is_mcp_client_transport("websocket")
    assert not is_mcp_client_transport(None)


def test_mcp_remote_transport_identifies_url_backed_transports() -> None:
    assert uses_mcp_remote_transport("sse")
    assert uses_mcp_remote_transport("http")
    assert uses_mcp_remote_transport(" SSE ")
    assert not uses_mcp_remote_transport("stdio")
    assert not uses_mcp_remote_transport(None)


def test_uses_protocol_stdio_identifies_protocol_stream_transports() -> None:
    assert uses_protocol_stdio("stdio")
    assert uses_protocol_stdio("acp")
    assert uses_protocol_stdio(" ACP ")


def test_uses_protocol_stdio_ignores_http_style_transports() -> None:
    assert not uses_protocol_stdio("http")
    assert not uses_protocol_stdio("sse")
    assert not uses_protocol_stdio(None)
