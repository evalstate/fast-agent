from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
    from fast_agent.mcp.client_connection import MCPClientConnection
    from fast_agent.mcp.interfaces import ServerRegistryProtocol

logger = get_logger(__name__)


@asynccontextmanager
async def gen_client(
    server_name: str,
    server_registry: ServerRegistryProtocol,
    *,
    server_config: MCPServerSettings | None = None,
    publish_capabilities: bool = True,
    callback_runtime: MCPClientCallbackRuntime | None = None,
    trigger_oauth: bool | None = None,
) -> AsyncIterator[MCPClientConnection]:
    """
    Create an on-demand high-level client for the specified server.
    Handles server startup through the public high-level ``mcp.client.Client``.
    For attached runtimes, use MCPConnectionManager instead.
    """
    if not server_registry:
        raise ValueError(
            "Server registry not found in the context. Please specify one either on this method, or in the context."
        )

    config = server_config or server_registry.get_server_config(server_name)
    if config is None:
        raise ValueError(f"Server '{server_name}' not found in registry.")

    from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
    from fast_agent.mcp.client_gateway import MCPClientHooks, open_request_scoped_client

    callbacks = callback_runtime or MCPClientCallbackRuntime(
        server_name=server_name, server_config=config
    )
    hooks = MCPClientHooks(
        active_home=server_registry.active_home,
        no_home=server_registry.no_home,
    )
    async with open_request_scoped_client(
        server_name=server_name,
        config=config,
        callback_runtime=callbacks,
        trigger_oauth=trigger_oauth,
        hooks=hooks,
    ) as connection:
        if publish_capabilities and connection.server_capabilities is not None:
            server_registry.set_server_capabilities(server_name, connection.server_capabilities)
        yield connection
