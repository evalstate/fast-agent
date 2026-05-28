from __future__ import annotations

from typing import TYPE_CHECKING

from acp.schema import EnvVariable, HttpHeader, HttpMcpServer, McpServerStdio, SseMcpServer

from fast_agent.config import MCPServerSettings

if TYPE_CHECKING:
    from collections.abc import Sequence

ACPConfiguredMCPServer = HttpMcpServer | SseMcpServer | McpServerStdio


def convert_acp_mcp_server(server: ACPConfiguredMCPServer) -> MCPServerSettings:
    if isinstance(server, McpServerStdio):
        return MCPServerSettings(
            name=server.name,
            transport="stdio",
            command=server.command,
            args=list(server.args or []),
            env=_env_map(server.env),
        )

    if isinstance(server, HttpMcpServer):
        return MCPServerSettings(
            name=server.name,
            transport="http",
            url=server.url,
            headers=_header_map(server.headers),
        )

    if isinstance(server, SseMcpServer):
        return MCPServerSettings(
            name=server.name,
            transport="sse",
            url=server.url,
            headers=_header_map(server.headers),
        )

    raise TypeError(f"Unsupported ACP MCP server type: {type(server)!r}")


def _env_map(env: Sequence[EnvVariable] | None) -> dict[str, str] | None:
    if not env:
        return None
    return {
        str(item.name): str(item.value)
        for item in env
        if item.name is not None and item.value is not None
    } or None


def _header_map(headers: list[HttpHeader] | None) -> dict[str, str] | None:
    if not headers:
        return None
    return {header.name: header.value for header in headers} or None
