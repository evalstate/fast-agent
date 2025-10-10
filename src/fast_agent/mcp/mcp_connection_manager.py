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
from mcp.client.stdio import (
    StdioServerParameters,
    get_default_environment,
)
from mcp.client.streamable_http import GetSessionIdCallback
from mcp.types import Implementation, JSONRPCMessage, ServerCapabilities

from fast_agent.config import MCPServerSettings
from fast_agent.context_dependent import ContextDependent
from fast_agent.core.exceptions import ServerInitializationError
from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction
from fast_agent.mcp.logger_textio import get_stderr_handler
from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
from fast_agent.mcp.oauth_client import build_oauth_provider
from fast_agent.mcp.sse_tracking import tracking_sse_client
from fast_agent.mcp.stdio_tracking_simple import tracking_stdio_client
from fast_agent.mcp.streamable_http_tracking import tracking_streamablehttp_client
from fast_agent.mcp.transport_tracking import TransportChannelMetrics

if TYPE_CHECKING:
    from mcp.client.auth import OAuthClientProvider

    from fast_agent.context import Context
    from fast_agent.mcp_server_registry import ServerRegistry

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


def _prepare_headers_and_auth(
    server_config: MCPServerSettings,
) -> tuple[dict[str, str], Optional["OAuthClientProvider"], set[str]]:
    """
    Prepare request headers and determine if OAuth authentication should be used.

    Returns a copy of the headers, an OAuth auth provider when applicable, and the set
    of user-supplied authorization header keys.
    """
    headers: dict[str, str] = dict(server_config.headers or {})
    auth_header_keys = {"authorization", "x-hf-authorization"}
    user_provided_auth_keys = {key for key in headers if key.lower() in auth_header_keys}

    # OAuth is only relevant for SSE/HTTP transports and should be skipped when the
    # user has already supplied explicit Authorization headers.
    if server_config.transport not in ("sse", "http") or user_provided_auth_keys:
        return headers, None, user_provided_auth_keys

    oauth_auth = build_oauth_provider(server_config)
    if oauth_auth is not None:
        # Scrub Authorization headers so OAuth-managed credentials are the only ones sent.
        for header_name in (
            "Authorization",
            "authorization",
            "X-HF-Authorization",
            "x-hf-authorization",
        ):
            headers.pop(header_name, None)

    return headers, oauth_auth, user_provided_auth_keys


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
    ) -> None:
        self.server_name = server_name
        self.server_config = server_config
        self.session: ClientSession | None = None
        self._client_session_factory = client_session_factory
        self._transport_context_factory = transport_context_factory
        # Signal that session is fully up and initialized
        self._initialized_event = Event()

        # Signal we want to shut down
        self._shutdown_event = Event()

        # Track error state
        self._error_occurred = False
        self._error_message = None

        # Server instructions from initialization
        self.server_instructions: str | None = None
        self.server_capabilities: ServerCapabilities | None = None
        self.server_implementation: Implementation | None = None
        self.client_capabilities: dict | None = None
        self.server_instructions_available: bool = False
        self.server_instructions_enabled: bool = (
            server_config.include_instructions if server_config else True
        )
        self.session_id: str | None = None
        self._get_session_id_cb: GetSessionIdCallback | None = None
        self.transport_metrics: TransportChannelMetrics | None = None

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
        assert self.session, "Session must be created before initialization"
        result = await self.session.initialize()

        self.server_capabilities = result.capabilities
        # InitializeResult exposes server info via `serverInfo`; keep fallback for older fields
        implementation = getattr(result, "serverInfo", None)
        if implementation is None:
            implementation = getattr(result, "implementation", None)
        self.server_implementation = implementation

        raw_instructions = getattr(result, "instructions", None)
        self.server_instructions_available = bool(raw_instructions)

        # Store instructions if provided by the server and enabled in config
        if self.server_config.include_instructions:
            self.server_instructions = raw_instructions
            if self.server_instructions:
                logger.debug(
                    f"{self.server_name}: Received server instructions",
                    data={"instructions": self.server_instructions},
                )
        else:
            self.server_instructions = None
            if self.server_instructions_available:
                logger.debug(
                    f"{self.server_name}: Server instructions disabled by configuration",
                    data={"instructions": raw_instructions},
                )
            else:
                logger.debug(f"{self.server_name}: No server instructions provided")

        # If there's an init hook, run it

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
            server_config=self.server_config,
            transport_metrics=self.transport_metrics,
        )

        self.session = session
        self.client_capabilities = getattr(session, "client_capabilities", None)

        return session


async def _server_lifecycle_task(server_conn: ServerConnection) -> None:
    """
    Manage the lifecycle of a single server connection.
    Runs inside the MCPConnectionManager's shared TaskGroup.
    """
    server_name = server_conn.server_name
    try:
        transport_context = server_conn._transport_context_factory()

        async with transport_context as (read_stream, write_stream, get_session_id_cb):
            server_conn._get_session_id_cb = get_session_id_cb

            if get_session_id_cb is not None:
                try:
                    server_conn.session_id = get_session_id_cb()
                except Exception:
                    logger.debug(f"{server_name}: Unable to retrieve session id from transport")
            elif server_conn.server_config.transport == "stdio":
                server_conn.session_id = "local"

            server_conn.create_session(read_stream, write_stream)

            async with server_conn.session:
                await server_conn.initialize_session()

                if get_session_id_cb is not None:
                    try:
                        server_conn.session_id = get_session_id_cb() or server_conn.session_id
                    except Exception:
                        logger.debug(f"{server_name}: Unable to refresh session id after init")
                elif server_conn.server_config.transport == "stdio":
                    server_conn.session_id = "local"

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
            def extract_errors(exception_group):
                """Recursively extract meaningful errors from ExceptionGroups"""
                messages = []
                for subexc in exception_group.exceptions:
                    if "ExceptionGroup" in type(subexc).__name__ and hasattr(subexc, "exceptions"):
                        # Recursively handle nested ExceptionGroups
                        messages.extend(extract_errors(subexc))
                    elif isinstance(subexc, HTTPStatusError):
                        # Special handling for HTTP errors to make them more user-friendly
                        messages.append(
                            f"HTTP Error: {subexc.response.status_code} {subexc.response.reason_phrase} for URL: {subexc.request.url}"
                        )
                    else:
                        # Show the exception type and message, plus the root cause if available
                        error_msg = f"{type(subexc).__name__}: {subexc}"
                        messages.append(error_msg)

                        # If there's a root cause, show that too as it's often the most informative
                        if hasattr(subexc, "__cause__") and subexc.__cause__:
                            messages.append(
                                f"Caused by: {type(subexc.__cause__).__name__}: {subexc.__cause__}"
                            )
                return messages

            error_messages = extract_errors(exc)
            # If we didn't extract any meaningful errors, fall back to the original exception
            if not error_messages:
                error_messages = [f"{type(exc).__name__}: {exc}"]
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
        self._mcp_sse_filter_added = False

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

    def _suppress_mcp_sse_errors(self) -> None:
        """Suppress MCP library's 'Error in sse_reader' messages."""
        if self._mcp_sse_filter_added:
            return

        import logging

        class MCPSSEErrorFilter(logging.Filter):
            def filter(self, record):
                return not (
                    record.name == "mcp.client.sse" and "Error in sse_reader" in record.getMessage()
                )

        mcp_sse_logger = logging.getLogger("mcp.client.sse")
        mcp_sse_logger.addFilter(MCPSSEErrorFilter())
        self._mcp_sse_filter_added = True

    async def launch_server(
        self,
        server_name: str,
        client_session_factory: Callable[
            [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
            ClientSession,
        ],
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

        config = self.server_registry.get_server_config(server_name)
        if not config:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        logger.debug(f"{server_name}: Found server configuration=", data=config.model_dump())

        timeline_steps = 20
        timeline_seconds = 30
        try:
            ctx = self.context
        except RuntimeError:
            ctx = None

        config_obj = getattr(ctx, "config", None)
        timeline_config = getattr(config_obj, "mcp_timeline", None)
        if timeline_config:
            timeline_steps = getattr(timeline_config, "steps", timeline_steps)
            timeline_seconds = getattr(timeline_config, "step_seconds", timeline_seconds)

        transport_metrics = (
            TransportChannelMetrics(
                bucket_seconds=timeline_seconds,
                bucket_count=timeline_steps,
            )
            if config.transport in ("http", "sse", "stdio")
            else None
        )

        def transport_context_factory():
            if config.transport == "stdio":
                if not config.command:
                    raise ValueError(
                        f"Server '{server_name}' uses stdio transport but no command is specified"
                    )
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

                channel_hook = transport_metrics.record_event if transport_metrics else None
                return _add_none_to_context(
                    tracking_stdio_client(
                        server_params, channel_hook=channel_hook, errlog=error_handler
                    )
                )
            elif config.transport == "sse":
                if not config.url:
                    raise ValueError(
                        f"Server '{server_name}' uses sse transport but no url is specified"
                    )
                # Suppress MCP library error spam
                self._suppress_mcp_sse_errors()
                headers, oauth_auth, user_auth_keys = _prepare_headers_and_auth(config)
                if user_auth_keys:
                    logger.debug(
                        f"{server_name}: Using user-specified auth header(s); skipping OAuth provider.",
                        user_auth_headers=sorted(user_auth_keys),
                    )
                channel_hook = None
                if transport_metrics is not None:

                    def channel_hook(event):
                        try:
                            transport_metrics.record_event(event)
                        except Exception:  # pragma: no cover - defensive guard
                            logger.debug(
                                "%s: transport metrics hook failed",
                                server_name,
                                exc_info=True,
                            )

                return tracking_sse_client(
                    config.url,
                    headers,
                    sse_read_timeout=config.read_transport_sse_timeout_seconds,
                    auth=oauth_auth,
                    channel_hook=channel_hook,
                )
            elif config.transport == "http":
                if not config.url:
                    raise ValueError(
                        f"Server '{server_name}' uses http transport but no url is specified"
                    )
                headers, oauth_auth, user_auth_keys = _prepare_headers_and_auth(config)
                if user_auth_keys:
                    logger.debug(
                        f"{server_name}: Using user-specified auth header(s); skipping OAuth provider.",
                        user_auth_headers=sorted(user_auth_keys),
                    )
                channel_hook = None
                if transport_metrics is not None:

                    def channel_hook(event):
                        try:
                            transport_metrics.record_event(event)
                        except Exception:  # pragma: no cover - defensive guard
                            logger.debug(
                                "%s: transport metrics hook failed",
                                server_name,
                                exc_info=True,
                            )

                return tracking_streamablehttp_client(
                    config.url,
                    headers,
                    auth=oauth_auth,
                    channel_hook=channel_hook,
                )
            else:
                raise ValueError(f"Unsupported transport: {config.transport}")

        server_conn = ServerConnection(
            server_name=server_name,
            server_config=config,
            transport_context_factory=transport_context_factory,
            client_session_factory=client_session_factory,
        )

        if transport_metrics is not None:
            server_conn.transport_metrics = transport_metrics

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
