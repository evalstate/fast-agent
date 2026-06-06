"""Runtime capabilities exposed to plugin command actions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.mcp_aggregator import MCPAttachOptions, MCPAttachResult, MCPDetachResult


AttachMcpServerCallback = Callable[
    [str, str, "MCPServerSettings | None", "MCPAttachOptions | None"],
    Awaitable["MCPAttachResult"],
]
DetachMcpServerCallback = Callable[[str, str], Awaitable["MCPDetachResult"]]
ListMcpServersCallback = Callable[[str], Awaitable[list[str]]]


class PluginRuntime(Protocol):
    """Stable live-runtime capabilities exposed to plugin command actions."""

    async def attach_mcp_server(
        self,
        *,
        server_name: str,
        agent_name: str | None = None,
        server_config: "MCPServerSettings | None" = None,
        options: "MCPAttachOptions | None" = None,
    ) -> "MCPAttachResult":
        """Attach an MCP server to a running MCP-capable agent and refresh instructions."""

    async def detach_mcp_server(
        self,
        *,
        server_name: str,
        agent_name: str | None = None,
    ) -> "MCPDetachResult":
        """Detach an MCP server from a running MCP-capable agent and refresh instructions."""

    async def list_attached_mcp_servers(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[str, ...]:
        """List MCP servers attached to a running MCP-capable agent."""

    async def list_configured_detached_mcp_servers(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[str, ...]:
        """List configured MCP servers that are not currently attached."""


@dataclass(frozen=True, slots=True)
class PluginRuntimeFacade:
    """Callback-backed implementation of plugin runtime capabilities."""

    current_agent_name: str
    attach_mcp_server_callback: AttachMcpServerCallback | None = None
    detach_mcp_server_callback: DetachMcpServerCallback | None = None
    list_attached_mcp_servers_callback: ListMcpServersCallback | None = None
    list_configured_detached_mcp_servers_callback: ListMcpServersCallback | None = None

    def _target_agent_name(self, agent_name: str | None) -> str:
        return agent_name or self.current_agent_name

    async def attach_mcp_server(
        self,
        *,
        server_name: str,
        agent_name: str | None = None,
        server_config: "MCPServerSettings | None" = None,
        options: "MCPAttachOptions | None" = None,
    ) -> "MCPAttachResult":
        if self.attach_mcp_server_callback is None:
            raise RuntimeError("Runtime MCP server attachment is not available.")
        return await self.attach_mcp_server_callback(
            self._target_agent_name(agent_name),
            server_name,
            server_config,
            options,
        )

    async def detach_mcp_server(
        self,
        *,
        server_name: str,
        agent_name: str | None = None,
    ) -> "MCPDetachResult":
        if self.detach_mcp_server_callback is None:
            raise RuntimeError("Runtime MCP server detachment is not available.")
        return await self.detach_mcp_server_callback(
            self._target_agent_name(agent_name),
            server_name,
        )

    async def list_attached_mcp_servers(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[str, ...]:
        if self.list_attached_mcp_servers_callback is None:
            raise RuntimeError("Runtime MCP server listing is not available.")
        servers = await self.list_attached_mcp_servers_callback(self._target_agent_name(agent_name))
        return tuple(servers)

    async def list_configured_detached_mcp_servers(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[str, ...]:
        if self.list_configured_detached_mcp_servers_callback is None:
            raise RuntimeError("Configured MCP server listing is not available.")
        servers = await self.list_configured_detached_mcp_servers_callback(
            self._target_agent_name(agent_name)
        )
        return tuple(servers)
