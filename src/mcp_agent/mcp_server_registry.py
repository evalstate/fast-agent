"""
This module defines a `ServerRegistry` class for managing MCP server configurations
and initialization logic.

The class loads server configurations from a YAML file,
supports dynamic registration of initialization hooks, and provides methods for
server initialization.
"""

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncGenerator, Callable, Dict

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession, stdio_client
from mcp.client.sse import sse_client
from mcp.client.stdio import (
    StdioServerParameters,
    get_default_environment,
)
from mcp.client.streamable_http import GetSessionIdCallback, streamablehttp_client

from mcp_agent.config import (
    MCPServerAuthSettings,
    MCPServerSettings,
    Settings,
    get_settings,
)
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.hf_auth import add_hf_auth_header
from mcp_agent.mcp.logger_textio import get_stderr_handler
from mcp_agent.mcp.mcp_connection_manager import (
    MCPConnectionManager,
    _add_none_to_context,
)

logger = get_logger(__name__)

InitHookCallable = Callable[[ClientSession | None, MCPServerAuthSettings | None], bool]
"""
A type alias for an initialization hook function that is invoked after MCP server initialization.

Args:
    session (ClientSession | None): The client session for the server connection.
    auth (MCPServerAuthSettings | None): The authentication configuration for the server.

Returns:
    bool: Result of the post-init hook (false indicates failure).
"""


class ServerRegistry:
    """
    A registry for managing server configurations and initialization logic.

    The `ServerRegistry` class is responsible for loading server configurations
    from a YAML file, registering initialization hooks, initializing servers,
    and executing post-initialization hooks dynamically.

    Attributes:
        config_path (str): Path to the YAML configuration file.
        registry (Dict[str, MCPServerSettings]): Loaded server configurations.
        init_hooks (Dict[str, InitHookCallable]): Registered initialization hooks.
    """

    def __init__(self, config: Settings | None = None, config_path: str | None = None) -> None:
        """
        Initialize the ServerRegistry with a configuration file.

        Args:
            config (Settings): The Settings object containing the server configurations.
            config_path (str): Path to the YAML configuration file.
        """
        if config is None:
            self.registry = self.load_registry_from_file(config_path)
        elif config.mcp is not None and hasattr(config.mcp, 'servers') and config.mcp.servers is not None:
            # Ensure config.mcp exists, has a 'servers' attribute, and it's not None
            self.registry = config.mcp.servers
        else:
            # Default to an empty dictionary if config.mcp is None or has no 'servers'
            self.registry = {}

        self.init_hooks: Dict[str, InitHookCallable] = {}
        self.connection_manager = MCPConnectionManager(self)

    def load_registry_from_file(
        self, config_path: str | None = None
    ) -> Dict[str, MCPServerSettings]:
        """
        Load the YAML configuration file and validate it.

        Returns:
            Dict[str, MCPServerSettings]: A dictionary of server configurations.

        Raises:
            ValueError: If the configuration is invalid.
        """
        servers = {} 

        settings = get_settings(config_path)
        
        if settings.mcp is not None and hasattr(settings.mcp, 'servers') and settings.mcp.servers is not None:
            return settings.mcp.servers
        
        return servers

    @asynccontextmanager
    async def start_server(
        self,
        server_name: str,
        client_session_factory: Callable[
            [
                MemoryObjectReceiveStream,
                MemoryObjectSendStream,
                timedelta | None,
                GetSessionIdCallback | None,
            ],
            ClientSession,
        ] = ClientSession,
    ) -> AsyncGenerator[ClientSession, None]:
        """
        Starts the server process based on its configuration. To initialize, call initialize_server

        Args:
            server_name (str): The name of the server to initialize.

        Returns:
            StdioServerParameters: The server parameters for stdio transport.

        Raises:
            ValueError: If the server is not found or has an unsupported transport.
        """
        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        config = self.registry[server_name]

        read_timeout_seconds = (
            timedelta(config.read_timeout_seconds) if config.read_timeout_seconds else None
        )

        if config.transport == "stdio":
            if not config.command or not config.args:
                raise ValueError(
                    f"Command and args are required for stdio transport: {server_name}"
                )

            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env={**get_default_environment(), **(config.env or {})},
                cwd=config.cwd,
            )

            # Create a stderr handler that logs to our application logger
            async with _add_none_to_context(
                stdio_client(server_params, errlog=get_stderr_handler(server_name))
            ) as (
                read_stream,
                write_stream,
                _,
            ):
                session = client_session_factory(
                    read_stream,
                    write_stream,
                    read_timeout_seconds,
                    None,  # No callback for stdio
                )
                async with session:
                    logger.info(f"{server_name}: Connected to server using stdio transport.")
                    try:
                        yield session
                    finally:
                        logger.debug(f"{server_name}: Closed session to server")

        elif config.transport == "sse":
            if not config.url:
                raise ValueError(f"URL is required for SSE transport: {server_name}")

            # Apply HuggingFace authentication if appropriate
            headers = add_hf_auth_header(config.url, config.headers)

            # Use sse_client to get the read and write streams
            async with _add_none_to_context(
                sse_client(
                    config.url,
                    headers,
                    sse_read_timeout=config.read_transport_sse_timeout_seconds,
                )
            ) as (read_stream, write_stream, _):
                session = client_session_factory(
                    read_stream,
                    write_stream,
                    read_timeout_seconds,
                    None,  # No callback for stdio
                )
                async with session:
                    logger.info(f"{server_name}: Connected to server using SSE transport.")
                    try:
                        yield session
                    finally:
                        logger.debug(f"{server_name}: Closed session to server")
        elif config.transport == "http":
            if not config.url:
                raise ValueError(f"URL is required for HTTP transport: {server_name}")

            # Apply HuggingFace authentication if appropriate
            headers = add_hf_auth_header(config.url, config.headers)

            async with streamablehttp_client(config.url, headers) as (
                read_stream,
                write_stream,
                _,
            ):
                session = client_session_factory(
                    read_stream,
                    write_stream,
                    read_timeout_seconds,
                    None,  # No callback for stdio
                )
                async with session:
                    logger.info(f"{server_name}: Connected to server using HTTP transport.")
                    try:
                        yield session
                    finally:
                        logger.debug(f"{server_name}: Closed session to server")

        # Unsupported transport
        else:
            raise ValueError(f"Unsupported transport: {config.transport}")

    @asynccontextmanager
    async def initialize_server(
        self,
        server_name: str,
        client_session_factory: Callable[
            [
                MemoryObjectReceiveStream,
                MemoryObjectSendStream,
                timedelta | None,
                GetSessionIdCallback,
            ],
            ClientSession,
        ] = ClientSession,
        init_hook: InitHookCallable = None,
    ) -> AsyncGenerator[ClientSession, None]:
        """
        Initialize a server based on its configuration.
        After initialization, also calls any registered or provided initialization hook for the server.

        Args:
            server_name (str): The name of the server to initialize.
            init_hook (InitHookCallable): Optional initialization hook function to call after initialization.

        Returns:
            StdioServerParameters: The server parameters for stdio transport.

        Raises:
            ValueError: If the server is not found or has an unsupported transport.
        """

        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        config = self.registry[server_name]

        async with self.start_server(
            server_name, client_session_factory=client_session_factory
        ) as session:
            try:
                logger.info(f"{server_name}: Initializing server...")
                await session.initialize()
                logger.info(f"{server_name}: Initialized.")

                intialization_callback = (
                    init_hook if init_hook is not None else self.init_hooks.get(server_name)
                )

                if intialization_callback:
                    logger.info(f"{server_name}: Executing init hook")
                    intialization_callback(session, config.auth)

                logger.info(f"{server_name}: Up and running!")
                yield session
            finally:
                logger.info(f"{server_name}: Ending server session.")

    def register_init_hook(self, server_name: str, hook: InitHookCallable) -> None:
        """
        Register an initialization hook for a specific server. This will get called
        after the server is initialized.

        Args:
            server_name (str): The name of the server.
            hook (callable): The initialization function to register.
        """
        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        self.init_hooks[server_name] = hook

    def execute_init_hook(self, server_name: str, session=None) -> bool:
        """
        Execute the initialization hook for a specific server.

        Args:
            server_name (str): The name of the server.
            session: The session object to pass to the initialization hook.
        """
        if server_name in self.init_hooks:
            hook = self.init_hooks[server_name]
            config = self.registry[server_name]
            logger.info(f"Executing init hook for '{server_name}'")
            return hook(session, config.auth)
        else:
            logger.info(f"No init hook registered for '{server_name}'")

    def get_server_config(self, server_name: str) -> MCPServerSettings | None:
        """
        Get the configuration for a specific server.

        Args:
            server_name (str): The name of the server.

        Returns:
            MCPServerSettings: The server configuration.
        """

        server_config = self.registry.get(server_name)
        if server_config is None:
            logger.warning(f"Server '{server_name}' not found in registry.")
            return None
        elif server_config.name is None:
            server_config.name = server_name
        return server_config
