"""
Manages the lifecycle of multiple MCP server connections.
"""

import asyncio
import traceback
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    AsyncGenerator,
    Callable,
    Dict,
    Optional,
)

from anyio import Event, Lock, create_task_group
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from httpx import HTTPStatusError
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import (
    StdioServerParameters,
    get_default_environment,
    stdio_client,
)
from mcp.client.streamable_http import GetSessionIdCallback, streamablehttp_client
from mcp.types import JSONRPCMessage, ServerCapabilities

from mcp_agent.config import MCPServerSettings
from mcp_agent.context_dependent import ContextDependent
from mcp_agent.core.exceptions import ServerInitializationError
from mcp_agent.event_progress import ProgressAction
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.logger_textio import get_stderr_handler
from mcp_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

if TYPE_CHECKING:
    from mcp_agent.context import Context
    from mcp_agent.mcp_server_registry import InitHookCallable, ServerRegistry

logger = get_logger(__name__)


class StreamingContextAdapter:
    """Adapter to provide a 3-value context from a 2-value context manager"""

    def __init__(self, context_manager):
        self.context_manager = context_manager
        self.cm_instance = None

    async def __aenter__(self):
        self.cm_instance = await self.context_manager.__aenter__()
        read_stream, write_stream = self.cm_instance
        return read_stream, write_stream, None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.context_manager.__aexit__(exc_type, exc_val, exc_tb)


def _add_none_to_context(context_manager):
    """Helper to add a None value to context managers that return 2 values instead of 3"""
    return StreamingContextAdapter(context_manager)


class ServerConnection:
    """
    Represents a long-lived MCP server connection, including:
    - The ClientSession to the server
    - The transport streams (via stdio/sse, etc.)
    """

    def __init__(
        self,
        server_name: str,
        server_config: MCPServerSettings,
        transport_context_factory: Callable[
            [],
            AsyncGenerator[
                tuple[
                    MemoryObjectReceiveStream[JSONRPCMessage | Exception],
                    MemoryObjectSendStream[JSONRPCMessage],
                    GetSessionIdCallback | None,
                ],
                None,
            ],
        ],
        client_session_factory: Callable[
            [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
            ClientSession,
        ],
        init_hook: Optional["InitHookCallable"] = None,
    ) -> None:
        self.server_name = server_name
        self.server_config = server_config
        self.session: ClientSession | None = None
        self._client_session_factory = client_session_factory
        self._init_hook = init_hook
        self._transport_context_factory = transport_context_factory
        # Signal that session is fully up and initialized
        self._initialized_event = Event()

        # Signal we want to shut down
        self._shutdown_event = Event()

        # Track error state
        self._error_occurred = False
        self._error_message = None

    def is_healthy(self) -> bool:
        """Check if the server connection is healthy and ready to use."""
        return self.session is not None and not self._error_occurred

    def reset_error_state(self) -> None:
        """Reset the error state, allowing reconnection attempts."""
        self._error_occurred = False
        self._error_message = None

    def request_shutdown(self) -> None:
        """
        Request the server to shut down. Signals the server lifecycle task to exit.
        """
        self._shutdown_event.set()

    async def wait_for_shutdown_request(self) -> None:
        """
        Wait until the shutdown event is set.
        """
        await self._shutdown_event.wait()

    async def initialize_session(self) -> None:
        """
        Initializes the server connection and session.
        Must be called within an async context.
        """

        result = await self.session.initialize()

        self.server_capabilities = result.capabilities
        # If there's an init hook, run it
        if self._init_hook:
            logger.info(f"{self.server_name}: Executing init hook.")
            self._init_hook(self.session, self.server_config.auth)

        # Now the session is ready for use
        self._initialized_event.set()

    async def wait_for_initialized(self) -> None:
        """
        Wait until the session is fully initialized.
        """
        await self._initialized_event.wait()

    def create_session(
        self,
        read_stream: MemoryObjectReceiveStream,
        send_stream: MemoryObjectSendStream,
    ) -> ClientSession:
        """
        Create a new session instance for this server connection.
        """

        read_timeout = (
            timedelta(seconds=self.server_config.read_timeout_seconds)
            if self.server_config.read_timeout_seconds
            else None
        )

        session = self._client_session_factory(
            read_stream, 
            send_stream, 
            read_timeout,
            server_config=self.server_config
        )

        self.session = session

        return session


async def _server_lifecycle_task(server_conn: ServerConnection) -> None:
    """
    Manage the lifecycle of a single server connection.
    Runs inside the MCPConnectionManager's shared TaskGroup.
    """
    server_name = server_conn.server_name
    try:
        transport_context = server_conn._transport_context_factory()

        async with transport_context as (read_stream, write_stream, _):
            server_conn.create_session(read_stream, write_stream)

            async with server_conn.session:
                await server_conn.initialize_session()
                await server_conn.wait_for_shutdown_request()

    except HTTPStatusError as http_exc:
        logger.error(
            f"{server_name}: Lifecycle task encountered HTTP error: {http_exc}",
            exc_info=True,
            data={
                "progress_action": ProgressAction.FATAL_ERROR,
                "server_name": server_name,
            },
        )
        server_conn._error_occurred = True
        server_conn._error_message = f"HTTP Error: {http_exc.response.status_code} {http_exc.response.reason_phrase} for URL: {http_exc.request.url}"
        server_conn._initialized_event.set()
        # No raise - let get_server handle it with a friendly message

    except Exception as exc:
        logger.error(
            f"{server_name}: Lifecycle task encountered an error: {exc}",
            exc_info=True,
            data={
                "progress_action": ProgressAction.FATAL_ERROR,
                "server_name": server_name,
            },
        )
        server_conn._error_occurred = True

        if "ExceptionGroup" in type(exc).__name__ and hasattr(exc, "exceptions"):
            # Handle ExceptionGroup better by extracting the actual errors
            error_messages = []
            for subexc in exc.exceptions:
                if isinstance(subexc, HTTPStatusError):
                    # Special handling for HTTP errors to make them more user-friendly
                    error_messages.append(
                        f"HTTP Error: {subexc.response.status_code} {subexc.response.reason_phrase} for URL: {subexc.request.url}"
                    )
                else:
                    error_messages.append(f"Error: {type(subexc).__name__}: {subexc}")
                if hasattr(subexc, "__cause__") and subexc.__cause__:
                    error_messages.append(
                        f"Caused by: {type(subexc.__cause__).__name__}: {subexc.__cause__}"
                    )
            server_conn._error_message = error_messages
        else:
            # For regular exceptions, keep the traceback but format it more cleanly
            server_conn._error_message = traceback.format_exception(exc)

        # If there's an error, we should also set the event so that
        # 'get_server' won't hang
        server_conn._initialized_event.set()
        # No raise - allow graceful exit


class MCPConnectionManager(ContextDependent):
    """
    Manages the lifecycle of multiple MCP server connections.
    Integrates with the application context system for proper resource management.
    """

    def __init__(
        self, server_registry: "ServerRegistry", context: Optional["Context"] = None
    ) -> None:
        super().__init__(context=context)
        self.server_registry = server_registry
        self.running_servers: Dict[str, ServerConnection] = {}
        self._lock = Lock()
        # Manage our own task group - independent of task context
        self._task_group = None
        self._task_group_active = False

    async def __aenter__(self):
        # Create a task group that isn't tied to a specific task
        self._task_group = create_task_group()
        # Enter the task group context
        await self._task_group.__aenter__()
        self._task_group_active = True
        self._tg = self._task_group
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure clean shutdown of all connections before exiting."""
        try:
            # First request all servers to shutdown
            await self.disconnect_all()

            # Add a small delay to allow for clean shutdown
            await asyncio.sleep(0.5)

            # Then close the task group if it's active
            if self._task_group_active:
                await self._task_group.__aexit__(exc_type, exc_val, exc_tb)
                self._task_group_active = False
                self._task_group = None
                self._tg = None
        except Exception as e:
            logger.error(f"Error during connection manager shutdown: {e}")

    async def launch_server(
        self,
        server_name: str,
        client_session_factory: Callable[
            [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
            ClientSession,
        ],
        init_hook: Optional["InitHookCallable"] = None,
    ) -> ServerConnection:
        """
        Connect to a server and return a RunningServer instance that will persist
        until explicitly disconnected.
        """
        # Create task group if it doesn't exist yet - make this method more resilient
        if not self._task_group_active:
            self._task_group = create_task_group()
            await self._task_group.__aenter__()
            self._task_group_active = True
            self._tg = self._task_group
            logger.info(f"Auto-created task group for server: {server_name}")

        config = self.server_registry.registry.get(server_name)
        if not config:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        logger.debug(f"{server_name}: Found server configuration=", data=config.model_dump())

        def transport_context_factory():
            if config.transport == "stdio":
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args if config.args is not None else [],
                    env={**get_default_environment(), **(config.env or {})},
                    cwd=config.cwd,
                )
                # Create custom error handler to ensure all output is captured
                error_handler = get_stderr_handler(server_name)
                # Explicitly ensure we're using our custom logger for stderr
                logger.debug(f"{server_name}: Creating stdio client with custom error handler")
                return _add_none_to_context(stdio_client(server_params, errlog=error_handler))
            elif config.transport == "sse":
                return _add_none_to_context(
                    sse_client(
                        config.url,
                        config.headers,
                        sse_read_timeout=config.read_transport_sse_timeout_seconds,
                    )
                )
            elif config.transport == "http":
                return streamablehttp_client(config.url, config.headers)
            else:
                raise ValueError(f"Unsupported transport: {config.transport}")

        server_conn = ServerConnection(
            server_name=server_name,
            server_config=config,
            transport_context_factory=transport_context_factory,
            client_session_factory=client_session_factory,
            init_hook=init_hook or self.server_registry.init_hooks.get(server_name),
        )

        async with self._lock:
            # Check if already running
            if server_name in self.running_servers:
                return self.running_servers[server_name]

            self.running_servers[server_name] = server_conn
            self._tg.start_soon(_server_lifecycle_task, server_conn)

        logger.info(f"{server_name}: Up and running with a persistent connection!")
        return server_conn

    async def get_server(
        self,
        server_name: str,
        client_session_factory: Callable,
        init_hook: Optional["InitHookCallable"] = None,
    ) -> ServerConnection:
        """
        Get a running server instance, launching it if needed.
        """
        # Get the server connection if it's already running and healthy
        async with self._lock:
            server_conn = self.running_servers.get(server_name)
            if server_conn and server_conn.is_healthy():
                return server_conn

            # If server exists but isn't healthy, remove it so we can create a new one
            if server_conn:
                logger.info(f"{server_name}: Server exists but is unhealthy, recreating...")
                self.running_servers.pop(server_name)
                server_conn.request_shutdown()

        # Launch the connection
        server_conn = await self.launch_server(
            server_name=server_name,
            client_session_factory=client_session_factory,
            init_hook=init_hook,
        )

        # Wait until it's fully initialized, or an error occurs
        await server_conn.wait_for_initialized()

        # Check if the server is healthy after initialization
        if not server_conn.is_healthy():
            error_msg = server_conn._error_message or "Unknown error"

            # Format the error message for better display
            if isinstance(error_msg, list):
                # Join the list with newlines for better readability
                formatted_error = "\n".join(error_msg)
            else:
                formatted_error = str(error_msg)

            raise ServerInitializationError(
                f"MCP Server: '{server_name}': Failed to initialize - see details. Check fastagent.config.yaml?",
                formatted_error,
            )

        return server_conn

    async def get_server_capabilities(self, server_name: str) -> ServerCapabilities | None:
        """Get the capabilities of a specific server."""
        server_conn = await self.get_server(
            server_name, client_session_factory=MCPAgentClientSession
        )
        return server_conn.server_capabilities if server_conn else None

    async def disconnect_server(self, server_name: str) -> None:
        """
        Disconnect a specific server if it's running under this connection manager.
        """
        logger.info(f"{server_name}: Disconnecting persistent connection to server...")

        async with self._lock:
            server_conn = self.running_servers.pop(server_name, None)
        if server_conn:
            server_conn.request_shutdown()
            logger.info(f"{server_name}: Shutdown signal sent (lifecycle task will exit).")
        else:
            logger.info(f"{server_name}: No persistent connection found. Skipping server shutdown")

    async def disconnect_all(self) -> None:
        """Disconnect all servers that are running under this connection manager."""
        # Get a copy of servers to shutdown
        servers_to_shutdown = []

        async with self._lock:
            if not self.running_servers:
                return

            # Make a copy of the servers to shut down
            servers_to_shutdown = list(self.running_servers.items())
            # Clear the dict immediately to prevent any new access
            self.running_servers.clear()

        # Release the lock before waiting for servers to shut down
        for name, conn in servers_to_shutdown:
            logger.info(f"{name}: Requesting shutdown...")
            conn.request_shutdown()
