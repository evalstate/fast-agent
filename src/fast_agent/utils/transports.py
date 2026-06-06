"""Shared transport predicates."""

from __future__ import annotations

from typing import Literal, get_args

from fast_agent.utils.text import strip_casefold

McpClientTransport = Literal["stdio", "sse", "http"]
McpRemoteTransport = Literal["sse", "http"]

MCP_CLIENT_TRANSPORTS = frozenset(get_args(McpClientTransport))
MCP_REMOTE_TRANSPORTS = frozenset(get_args(McpRemoteTransport))
PROTOCOL_STDIO_TRANSPORTS = frozenset({"stdio", "acp"})


def _normalize_transport(transport: str | None) -> str:
    return strip_casefold(transport) if transport is not None else ""


def _transport_in(transport: str | None, supported: frozenset[str]) -> bool:
    return _normalize_transport(transport) in supported


def is_mcp_client_transport(transport: str | None) -> bool:
    return _transport_in(transport, MCP_CLIENT_TRANSPORTS)


def uses_mcp_remote_transport(transport: str | None) -> bool:
    return _transport_in(transport, MCP_REMOTE_TRANSPORTS)


def uses_protocol_stdio(transport: str | None) -> bool:
    """Return True when a transport owns stdout/stderr as a protocol channel."""
    return _transport_in(transport, PROTOCOL_STDIO_TRANSPORTS)
