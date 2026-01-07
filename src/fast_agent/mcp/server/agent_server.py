"""
Enhanced AgentMCPServer with robust shutdown handling for SSE transport.
"""

import asyncio
import logging
import os
import signal
import time
from contextlib import AsyncExitStack, asynccontextmanager
from importlib.metadata import version as get_version
from typing import Any, AsyncContextManager, Awaitable, Callable, Literal, Protocol, cast

from mcp.server.fastmcp import Context as MCPContext
from mcp.server.fastmcp import FastMCP

import fast_agent.core
import fast_agent.core.prompt
from fast_agent.core.fastagent import AgentInstance
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.tool_progress import MCPToolProgressManager
from fast_agent.types import RequestParams
from fast_agent.utils.async_utils import run_sync

logger = get_logger(__name__)


def _get_oauth_config() -> tuple[str | None, list[str], str]:
    """
    Read OAuth configuration from environment variables.

    Returns:
        Tuple of (provider, scopes, resource_url).
        provider is None if OAuth is not enabled.
    """
    oauth_provider = os.environ.get("FAST_AGENT_SERVE_OAUTH", "").lower()

    # Normalize provider aliases
    if oauth_provider in ("hf", "huggingface"):
        oauth_provider = "huggingface"
    elif not oauth_provider:
        oauth_provider = None

    # Parse scopes from comma-separated string
    oauth_scopes_str = os.environ.get("FAST_AGENT_OAUTH_SCOPES", "")
    oauth_scopes = [s.strip() for s in oauth_scopes_str.split(",") if s.strip()] or ["access"]

    # Resource URL defaults to localhost:8000
    resource_url = os.environ.get("FAST_AGENT_OAUTH_RESOURCE_URL", "http://localhost:8000")

    return oauth_provider, oauth_scopes, resource_url


TransportMode = Literal["http", "sse", "stdio"]
McpTransportMode = Literal["streamable-http", "sse", "stdio"]


class _LocalSseTransport(Protocol):
    connect_sse: Callable[..., AsyncContextManager[Any]]
    _read_stream_writers: dict[Any, Any]


class _FastMCPLocalExtensions(Protocol):
    _sse_transport: _LocalSseTransport | None
    _lifespan_state: str
    _on_shutdown: Callable[[], Awaitable[None]]
    _server_should_exit: bool


class AgentMCPServer:
    """Exposes FastAgent agents as MCP tools through an MCP server."""

    def __init__(
        self,
        primary_instance: AgentInstance,
        create_instance: Callable[[], Awaitable[AgentInstance]],
        dispose_instance: Callable[[AgentInstance], Awaitable[None]],
        instance_scope: str,
        server_name: str = "FastAgent-MCP-Server",
        server_description: str | None = None,
        tool_description: str | None = None,
        host: str = "0.0.0.0",
        get_registry_version: Callable[[], int] | None = None,
        reload_callback: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        """Initialize the server with the provided agent app."""
        self.primary_instance = primary_instance
        self._create_instance_task = create_instance
        self._dispose_instance_task = dispose_instance
        self._instance_scope = instance_scope
        self._get_registry_version = get_registry_version
        self._reload_callback = reload_callback
        self._primary_registry_version = getattr(primary_instance, "registry_version", 0)
        self._shared_instance_lock = asyncio.Lock()
        self._shared_active_requests = 0
        self._stale_instances: list[AgentInstance] = []

        # Check for OAuth configuration
        oauth_provider, oauth_scopes, resource_url = _get_oauth_config()
        token_verifier = None
        auth_settings = None

        if oauth_provider == "huggingface":
            from mcp.server.auth.settings import AuthSettings
            from pydantic import AnyHttpUrl

            from fast_agent.mcp.auth.presence import PresenceTokenVerifier

            token_verifier = PresenceTokenVerifier(provider="huggingface", scopes=oauth_scopes)
            auth_settings = AuthSettings(
                issuer_url=AnyHttpUrl("https://huggingface.co"),
                resource_server_url=AnyHttpUrl(resource_url),
                required_scopes=oauth_scopes,
            )
            logger.info(
                f"OAuth enabled for provider '{oauth_provider}'",
                name="oauth_enabled",
                provider=oauth_provider,
                scopes=oauth_scopes,
                resource_url=resource_url,
            )

        self.mcp_server: FastMCP = FastMCP(
            name=server_name,
            instructions=self._build_instructions(server_description),
            token_verifier=token_verifier,
            auth=auth_settings,
            host=host,
        )
        self._set_mcp_version()
        self._configure_capabilities()

        # Register root route for HTTP/SSE transport info
        @self.mcp_server.custom_route("/", methods=["GET"])
        async def root_info(request):
            from starlette.responses import PlainTextResponse

            version = get_version("fast-agent-mcp")
            return PlainTextResponse(
                f"fast-agent mcp server (v{version}) - see https://fast-agent.ai for more information."
            )

        if self._instance_scope == "request":
            # Ensure FastMCP does not attempt to maintain sessions for stateless mode
            self.mcp_server.settings.stateless_http = True
        self._tool_description = tool_description
        self._shared_instance_active = True
        self._registered_agents: set[str] = set(primary_instance.agents.keys())
        # Shutdown coordination
        self._graceful_shutdown_event = asyncio.Event()
        self._force_shutdown_event = asyncio.Event()
        self._shutdown_timeout = 5.0  # Seconds to wait for graceful shutdown

        # Resource management
        self._exit_stack = AsyncExitStack()
        self._active_connections: set[object] = set()

        # Server state
        self._server_task = None

        # Standard logging channel so we appear alongside Uvicorn/logging output
        self.std_logger = logging.getLogger("fast_agent.server")

        # Connection-scoped instance tracking
        self._connection_instances: dict[int, AgentInstance] = {}
        self._connection_cleanup_tasks: dict[int, Callable[[], Awaitable[None]]] = {}
        self._connection_lock = asyncio.Lock()

        # Set up agent tools
        self.setup_tools()

        logger.info(
            f"AgentMCPServer initialized with {len(primary_instance.agents)} agents",
            name="mcp_server_initialized",
            agent_count=len(primary_instance.agents),
            instance_scope=instance_scope,
        )

    def setup_tools(self) -> None:
        """Register all agents as MCP tools."""
        for agent_name in self.primary_instance.agents.keys():
            self.register_agent_tools(agent_name)
        if self._reload_callback is not None:
            self._register_reload_tool()

    def register_agent_tools(self, agent_name: str) -> None:
        """Register tools for a specific agent."""
        self._registered_agents.add(agent_name)

        # Basic send message tool
        tool_description = (
            self._tool_description.format(agent=agent_name)
            if self._tool_description and "{agent}" in self._tool_description
            else self._tool_description
        )

        agent = self.primary_instance.agents.get(agent_name)
        agent_description = None
        if agent is not None:
            config = getattr(agent, "config", None)
            agent_description = getattr(config, "description", None)

        @self.mcp_server.tool(
            name=f"{agent_name}_send",
            description=tool_description
            or agent_description
            or f"Send a message to the {agent_name} agent",
            structured_output=False,
            # MCP 1.10.1 turns every tool in to a structured output
        )
        async def send_message(message: str, ctx: MCPContext) -> str:
            """Send a message to the agent and return its response."""
            # Extract bearer token from auth context for token passthrough
            from fast_agent.mcp.auth.context import request_bearer_token

            bearer_token = None
            try:
                from mcp.server.auth.middleware.auth_context import get_access_token

                access_token = get_access_token()
                if access_token:
                    bearer_token = access_token.token
            except Exception:
                # Auth context not available (e.g., no auth configured)
                pass

            # Set the token in our contextvar for LLM provider access
            saved_token = request_bearer_token.set(bearer_token)
            report_progress = self._build_progress_reporter(ctx)
            request_params = RequestParams(
                tool_execution_handler=MCPToolProgressManager(report_progress),
                emit_loop_progress=True,
            )
            try:
                instance = await self._acquire_instance(ctx)
                agent = instance.app[agent_name]
                agent_context = getattr(agent, "context", None)

                # Define the function to execute
                async def execute_send():
                    start = time.perf_counter()
                    logger.info(
                        f"MCP request received for agent '{agent_name}'",
                        name="mcp_request_start",
                        agent=agent_name,
                        session=self._session_identifier(ctx),
                    )
                    self.std_logger.info(
                        "MCP request received for agent '%s' (scope=%s)",
                        agent_name,
                        self._instance_scope,
                    )

                    response = await agent.send(message, request_params=request_params)
                    duration = time.perf_counter() - start

                    logger.info(
                        f"Agent '{agent_name}' completed MCP request",
                        name="mcp_request_complete",
                        agent=agent_name,
                        duration=duration,
                        session=self._session_identifier(ctx),
                    )
                    self.std_logger.info(
                        "Agent '%s' completed MCP request in %.2fs (scope=%s)",
                        agent_name,
                        duration,
                        self._instance_scope,
                    )
                    return response

                try:
                    # Execute with bridged context
                    if agent_context and ctx:
                        return await self.with_bridged_context(agent_context, ctx, execute_send)
                    return await execute_send()
                finally:
                    await self._release_instance(ctx, instance)
            finally:
                # Always reset the contextvar
                request_bearer_token.reset(saved_token)

        if self._instance_scope != "request":
            # Register a history prompt for this agent
            @self.mcp_server.prompt(
                name=f"{agent_name}_history",
                description=f"Conversation history for the {agent_name} agent",
            )
            async def get_history_prompt(ctx: MCPContext) -> list:
                """Return the conversation history as MCP messages."""
                instance = await self._acquire_instance(ctx)
                agent = instance.app[agent_name]
                try:
                    multipart_history = agent.message_history
                    if not multipart_history:
                        return []

                    # Convert the multipart message history to standard PromptMessages
                    prompt_messages = fast_agent.core.prompt.Prompt.from_multipart(
                        multipart_history
                    )
                    # In FastMCP, we need to return the raw list of messages
                    return [{"role": msg.role, "content": msg.content} for msg in prompt_messages]
                finally:
                    await self._release_instance(ctx, instance, reuse_connection=True)

    def _register_missing_agents(self, instance: AgentInstance) -> None:
        new_agents = set(instance.agents.keys())
        missing = new_agents - self._registered_agents
        for agent_name in sorted(missing):
            self.register_agent_tools(agent_name)

    def _register_reload_tool(self) -> None:
        @self.mcp_server.tool(
            name="reload_agent_cards",
            description="Reload AgentCards from disk",
            structured_output=False,
        )
        async def reload_agent_cards(ctx: MCPContext) -> str:
            if not self._reload_callback:
                return "Reload not available."
            changed = await self._reload_callback()
            if not changed:
                return "No AgentCard changes detected."

            if self._instance_scope == "shared":
                await self._maybe_refresh_shared_instance()
                return "Reloaded AgentCards."

            if self._instance_scope == "connection":
                session_key = self._connection_key(ctx)
                new_instance = await self._create_instance_task()
                old_instance = None
                async with self._connection_lock:
                    old_instance = self._connection_instances.get(session_key)
                    self._connection_instances[session_key] = new_instance
                self._register_missing_agents(new_instance)
                if old_instance is not None:
                    await self._dispose_instance_task(old_instance)
                return "Reloaded AgentCards."

            # request scope: register tools from a fresh instance, then dispose it
            new_instance = await self._create_instance_task()
            try:
                self._register_missing_agents(new_instance)
            finally:
                await self._dispose_instance_task(new_instance)
            return "Reloaded AgentCards."

    def _set_mcp_version(self) -> None:
        try:
            self.mcp_server._mcp_server.version = get_version("fast-agent-mcp")
            return
        except Exception:
            pass
        try:
            self.mcp_server._mcp_server.version = get_version("fast-agent")
        except Exception:
            pass

    def _configure_capabilities(self) -> None:
        from mcp import types

        handlers = self.mcp_server._mcp_server.request_handlers
        for request_type in (
            types.ListResourcesRequest,
            types.ReadResourceRequest,
            types.ListResourceTemplatesRequest,
        ):
            handlers.pop(request_type, None)

        if self._instance_scope == "request":
            for request_type in (
                types.ListPromptsRequest,
                types.GetPromptRequest,
            ):
                handlers.pop(request_type, None)

    def _build_instructions(self, server_description: str | None) -> str:
        agent_count = len(self.primary_instance.agents)
        base = server_description or f"This server provides access to {agent_count} agents."
        if self._instance_scope == "request":
            scope_info = "do NOT retain history between your requests"
        elif self._instance_scope == "connection":
            scope_info = "retain history between tool calls."
        else:
            scope_info = "retain history between tool calls."
        return (
            f"{base} Use the  `{self._name_for_send_tool()}` tools to send messages to agents. "
            f"Instance mode is {self._instance_scope} "
            f"Agents ({scope_info})"
        )

    def _name_for_send_tool(self) -> str:
        return "<agent>_send"

    def _build_progress_reporter(
        self, ctx: MCPContext
    ) -> Callable[[float, float | None, str | None], Awaitable[None]]:
        async def report_progress(
            progress: float, total: float | None = None, message: str | None = None
        ) -> None:
            try:
                meta = ctx.request_context.meta
                progress_token = meta.progressToken if meta else None
                if progress_token is None:
                    return

                from mcp import types

                await ctx.request_context.session.send_notification(
                    types.ServerNotification(
                        types.ProgressNotification(
                            params=types.ProgressNotificationParams(
                                progressToken=progress_token,
                                progress=progress,
                                total=total,
                                message=message,
                            )
                        )
                    ),
                    related_request_id=ctx.request_context.request_id,
                )
            except Exception:
                pass

        return report_progress

    async def _acquire_instance(self, ctx: MCPContext | None) -> AgentInstance:
        if self._instance_scope == "shared":
            await self._maybe_refresh_shared_instance()
            self._shared_active_requests += 1
            return self.primary_instance

        if self._instance_scope == "request":
            return await self._create_instance_task()

        # Connection scope
        assert ctx is not None, "Context is required for connection-scoped instances"
        session_key = self._connection_key(ctx)
        async with self._connection_lock:
            instance = self._connection_instances.get(session_key)
            if instance is None:
                instance = await self._create_instance_task()
                self._connection_instances[session_key] = instance
                self._register_session_cleanup(ctx, session_key)
            return instance

    async def _release_instance(
        self,
        ctx: MCPContext | None,
        instance: AgentInstance,
        *,
        reuse_connection: bool = False,
    ) -> None:
        if self._instance_scope == "shared":
            if self._shared_active_requests > 0:
                self._shared_active_requests -= 1
            await self._dispose_stale_instances_if_idle()
        elif self._instance_scope == "request":
            await self._dispose_instance_task(instance)
        elif self._instance_scope == "connection" and reuse_connection is False:
            # Connection-scoped instances persist until session cleanup
            pass

    def _connection_key(self, ctx: MCPContext) -> int:
        return id(ctx.session)

    def _register_session_cleanup(self, ctx: MCPContext, session_key: int) -> None:
        async def cleanup() -> None:
            instance = self._connection_instances.pop(session_key, None)
            if instance is not None:
                await self._dispose_instance_task(instance)

        exit_stack = getattr(ctx.session, "_exit_stack", None)
        if exit_stack is not None:
            exit_stack.push_async_callback(cleanup)
        else:
            self._connection_cleanup_tasks[session_key] = cleanup

    def _session_identifier(self, ctx: MCPContext | None) -> str | None:
        if ctx is None:
            return None
        request = getattr(ctx.request_context, "request", None)
        if request is not None:
            return request.headers.get("mcp-session-id")
        return None

    async def _maybe_refresh_shared_instance(self) -> None:
        if not self._get_registry_version:
            return
        latest_version = self._get_registry_version()
        if latest_version <= self._primary_registry_version:
            return

        async with self._shared_instance_lock:
            latest_version = self._get_registry_version()
            if latest_version <= self._primary_registry_version:
                return

            new_instance = await self._create_instance_task()
            old_instance = self.primary_instance
            self.primary_instance = new_instance
            self._primary_registry_version = getattr(
                new_instance, "registry_version", latest_version
            )
            self._stale_instances.append(old_instance)

            new_agents = set(new_instance.agents.keys())
            missing = new_agents - self._registered_agents
            for agent_name in sorted(missing):
                self.register_agent_tools(agent_name)

    async def _dispose_stale_instances_if_idle(self) -> None:
        if self._shared_active_requests:
            return
        if not self._stale_instances:
            return

        stale = list(self._stale_instances)
        self._stale_instances.clear()
        for instance in stale:
            await self._dispose_instance_task(instance)

    async def _dispose_primary_instance(self) -> None:
        if self._shared_instance_active:
            try:
                await self._dispose_instance_task(self.primary_instance)
            finally:
                self._shared_instance_active = False

    async def _dispose_all_stale_instances(self) -> None:
        if not self._stale_instances:
            return
        stale = list(self._stale_instances)
        self._stale_instances.clear()
        for instance in stale:
            await self._dispose_instance_task(instance)

    async def _dispose_all_connection_instances(self) -> None:
        pending_cleanups = list(self._connection_cleanup_tasks.values())
        self._connection_cleanup_tasks.clear()
        for cleanup in pending_cleanups:
            await cleanup()

        async with self._connection_lock:
            instances = list(self._connection_instances.values())
            self._connection_instances.clear()

        for instance in instances:
            await self._dispose_instance_task(instance)

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful and forced shutdown."""
        loop = asyncio.get_running_loop()

        def handle_signal(is_term=False):
            # Use asyncio.create_task to handle the signal in an async-friendly way
            asyncio.create_task(self._handle_shutdown_signal(is_term))

        # Register handlers for SIGINT (Ctrl+C) and SIGTERM
        for sig, is_term in [(signal.SIGINT, False), (signal.SIGTERM, True)]:
            import platform

            if platform.system() != "Windows":
                loop.add_signal_handler(sig, lambda term=is_term: handle_signal(term))

        logger.debug("Signal handlers installed")

    async def _handle_shutdown_signal(self, is_term=False):
        """Handle shutdown signals with proper escalation."""
        signal_name = "SIGTERM" if is_term else "SIGINT (Ctrl+C)"

        # If force shutdown already requested, exit immediately
        if self._force_shutdown_event.is_set():
            logger.info("Force shutdown already in progress, exiting immediately...")
            os._exit(1)

        # If graceful shutdown already in progress, escalate to forced
        if self._graceful_shutdown_event.is_set():
            logger.info(f"Second {signal_name} received. Forcing shutdown...")
            self._force_shutdown_event.set()
            # Allow a brief moment for final cleanup, then force exit
            await asyncio.sleep(0.5)
            os._exit(1)

        # First signal - initiate graceful shutdown
        logger.info(f"{signal_name} received. Starting graceful shutdown...")
        print(f"\n{signal_name} received. Starting graceful shutdown...")
        print("Press Ctrl+C again to force exit.")
        self._graceful_shutdown_event.set()

    def run(
        self,
        transport: TransportMode = "http",
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        """Run the MCP server synchronously."""
        if transport in ["sse", "http"]:
            self.mcp_server.settings.host = host
            self.mcp_server.settings.port = port

            # For synchronous run, we can use the simpler approach
            try:
                # Add any server attributes that might help with shutdown
                if not hasattr(self.mcp_server, "_server_should_exit"):
                    setattr(self.mcp_server, "_server_should_exit", False)

                # Run the server
                mcp_transport: McpTransportMode = (
                    "streamable-http" if transport == "http" else transport
                )
                self.mcp_server.run(transport=mcp_transport)
            except KeyboardInterrupt:
                print("\nServer stopped by user (CTRL+C)")
            except SystemExit as e:
                # Handle normal exit
                print(f"\nServer exiting with code {e.code}")
                # Re-raise to allow normal exit process
                raise
            except Exception as e:
                print(f"\nServer error: {e}")
            finally:
                # Run an async cleanup in a new event loop
                try:
                    run_sync(self.shutdown)
                except (SystemExit, KeyboardInterrupt):
                    # These are expected during shutdown
                    pass
        else:  # stdio
            try:
                self.mcp_server.run(transport="stdio")
            except KeyboardInterrupt:
                print("\nServer stopped by user (CTRL+C)")
            finally:
                # Minimal cleanup for stdio
                run_sync(self._cleanup_stdio)

    async def run_async(
        self, transport: TransportMode = "http", host: str = "0.0.0.0", port: int = 8000
    ) -> None:
        """Run the MCP server asynchronously with improved shutdown handling."""
        # Use different handling strategies based on transport type
        if transport in ["sse", "http"]:
            # For SSE/HTTP, use our enhanced shutdown handling
            self._setup_signal_handlers()

            self.mcp_server.settings.host = host
            self.mcp_server.settings.port = port

            # Start the server in a separate task so we can monitor it
            if transport == "http":
                http_transport: Literal["http", "sse"] = "http"
            elif transport == "sse":
                http_transport = "sse"
            else:
                raise ValueError("HTTP/SSE handler received stdio transport")
            self._server_task = asyncio.create_task(self._run_server_with_shutdown(http_transport))

            try:
                # Wait for the server task to complete
                await self._server_task
            except (asyncio.CancelledError, KeyboardInterrupt):
                # Both cancellation and KeyboardInterrupt are expected during shutdown
                logger.info("Server stopped via cancellation or interrupt")
                print("\nServer stopped")
            except SystemExit as e:
                # Handle normal exit cleanly
                logger.info(f"Server exiting with code {e.code}")
                print(f"\nServer exiting with code {e.code}")
                # If this is exit code 0, let it propagate for normal exit
                if e.code == 0:
                    raise
            except Exception as e:
                logger.error(f"Server error: {e}", exc_info=True)
                print(f"\nServer error: {e}")
            finally:
                # Only do minimal cleanup - don't try to be too clever
                await self._cleanup_stdio()
                print("\nServer shutdown complete.")
        else:  # stdio
            # For STDIO, use simpler approach that respects STDIO lifecycle
            try:
                # Run directly without extra monitoring or signal handlers
                # This preserves the natural lifecycle of STDIO connections
                await self.mcp_server.run_stdio_async()
            except (asyncio.CancelledError, KeyboardInterrupt):
                logger.info("Server stopped (CTRL+C)")
                print("\nServer stopped (CTRL+C)")
            except SystemExit as e:
                # Handle normal exit cleanly
                logger.info(f"Server exiting with code {e.code}")
                print(f"\nServer exiting with code {e.code}")
                # If this is exit code 0, let it propagate for normal exit
                if e.code == 0:
                    raise
            # Only perform minimal cleanup needed for STDIO
            await self._cleanup_stdio()

    async def _run_server_with_shutdown(self, transport: Literal["http", "sse"]):
        """Run the server with proper shutdown handling."""
        # This method is used for SSE/HTTP transport
        if transport not in ["sse", "http"]:
            raise ValueError("This method should only be used with SSE or HTTP transport")

        # Start a monitor task for shutdown
        shutdown_monitor = asyncio.create_task(self._monitor_shutdown())

        try:
            # Patch SSE server to track connections if needed
            mcp_ext = cast("_FastMCPLocalExtensions", self.mcp_server)
            sse_transport = getattr(mcp_ext, "_sse_transport", None)
            if sse_transport is not None:
                # Store the original connect_sse method
                original_connect = sse_transport.connect_sse

                # Create a wrapper that tracks connections
                @asynccontextmanager
                async def tracked_connect_sse(*args, **kwargs):
                    async with original_connect(*args, **kwargs) as streams:
                        self._active_connections.add(streams)
                        try:
                            yield streams
                        finally:
                            self._active_connections.discard(streams)

                # Replace with our tracking version
                sse_transport.connect_sse = tracked_connect_sse

            # Run the server based on transport type
            if transport == "sse":
                await self.mcp_server.run_sse_async()
            elif transport == "http":
                # Check if HF OAuth is enabled - if so, wrap app with header middleware
                oauth_provider = os.environ.get("FAST_AGENT_SERVE_OAUTH", "").lower()
                if oauth_provider in ("hf", "huggingface"):
                    await self._run_http_with_hf_middleware()
                else:
                    await self.mcp_server.run_streamable_http_async()
        finally:
            # Cancel the monitor when the server exits
            shutdown_monitor.cancel()
            try:
                await shutdown_monitor
            except asyncio.CancelledError:
                pass

    async def _run_http_with_hf_middleware(self) -> None:
        """Run HTTP server with HuggingFace header normalization middleware.

        This wraps the Starlette app with middleware that copies X-HF-Authorization
        to Authorization header, enabling HuggingFace Spaces authentication.
        """
        import uvicorn

        from fast_agent.mcp.auth.middleware import HFAuthHeaderMiddleware

        # Get the Starlette app from FastMCP
        starlette_app = self.mcp_server.streamable_http_app()

        # Wrap with our header normalization middleware
        wrapped_app = HFAuthHeaderMiddleware(starlette_app)

        # Run uvicorn with the wrapped app
        config = uvicorn.Config(
            wrapped_app,
            host=self.mcp_server.settings.host,
            port=self.mcp_server.settings.port,
            log_level=self.mcp_server.settings.log_level.lower(),
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def _monitor_shutdown(self):
        """Monitor for shutdown signals and coordinate proper shutdown sequence."""
        try:
            # Wait for graceful shutdown request
            await self._graceful_shutdown_event.wait()
            logger.info("Graceful shutdown initiated")

            # Two possible paths:
            # 1. Wait for force shutdown
            # 2. Wait for shutdown timeout
            force_shutdown_task = asyncio.create_task(self._force_shutdown_event.wait())
            timeout_task = asyncio.create_task(asyncio.sleep(self._shutdown_timeout))

            done, pending = await asyncio.wait(
                [force_shutdown_task, timeout_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel the remaining task
            for task in pending:
                task.cancel()

            # Determine shutdown reason
            if force_shutdown_task in done:
                logger.info("Force shutdown requested by user")
                print("\nForce shutdown initiated...")
            else:
                logger.info(f"Graceful shutdown timed out after {self._shutdown_timeout} seconds")
                print(f"\nGraceful shutdown timed out after {self._shutdown_timeout} seconds")

                os._exit(0)

        except asyncio.CancelledError:
            # Monitor was cancelled - clean exit
            pass
        except Exception as e:
            logger.error(f"Error in shutdown monitor: {e}", exc_info=True)

    async def _close_sse_connections(self):
        """Force close all SSE connections."""
        # Close tracked connections
        for conn in list(self._active_connections):
            try:
                close = getattr(conn, "close", None)
                if callable(close):
                    await close()
                else:
                    aclose = getattr(conn, "aclose", None)
                    if callable(aclose):
                        await aclose()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
            self._active_connections.discard(conn)

        # Access the SSE transport if it exists to close stream writers
        mcp_ext = cast("_FastMCPLocalExtensions", self.mcp_server)
        sse = getattr(mcp_ext, "_sse_transport", None)
        if sse is not None:
            # Close all read stream writers
            writers = list(sse._read_stream_writers.items())
            for session_id, writer in writers:
                try:
                    logger.debug(f"Closing SSE connection: {session_id}")
                    # Instead of aclose, try to close more gracefully
                    # Send a special event to notify client, then close
                    try:
                        if hasattr(writer, "send") and not getattr(writer, "_closed", False):
                            try:
                                # Try to send a close event if possible
                                await writer.send(Exception("Server shutting down"))
                            except (AttributeError, asyncio.CancelledError):
                                pass
                    except Exception:
                        pass

                    # Now close the stream
                    await writer.aclose()
                    sse._read_stream_writers.pop(session_id, None)
                except Exception as e:
                    logger.error(f"Error closing SSE connection {session_id}: {e}")

        # If we have a ASGI lifespan hook, try to signal closure
        if getattr(mcp_ext, "_lifespan_state", None) == "started":
            logger.debug("Attempting to signal ASGI lifespan shutdown")
            try:
                on_shutdown = getattr(mcp_ext, "_on_shutdown", None)
                if on_shutdown is not None:
                    await on_shutdown()
            except Exception as e:
                logger.error(f"Error during ASGI lifespan shutdown: {e}")

    async def with_bridged_context(self, agent_context, mcp_context, func, *args, **kwargs):
        """
        Execute a function with bridged context between MCP and agent

        Args:
            agent_context: The agent's context object
            mcp_context: The MCP context from the tool call
            func: The function to execute
            args, kwargs: Arguments to pass to the function
        """
        # Store original progress reporter if it exists
        original_progress_reporter = None
        if hasattr(agent_context, "progress_reporter"):
            original_progress_reporter = agent_context.progress_reporter

        # Store MCP context in agent context for nested calls
        agent_context.mcp_context = mcp_context

        # Create bridged progress reporter
        async def bridged_progress(progress, total=None) -> None:
            if mcp_context:
                await mcp_context.report_progress(progress, total)
            if original_progress_reporter:
                await original_progress_reporter(progress, total)

        # Install bridged progress reporter
        if hasattr(agent_context, "progress_reporter"):
            agent_context.progress_reporter = bridged_progress

        try:
            # Call the function
            return await func(*args, **kwargs)
        finally:
            # Restore original progress reporter
            if hasattr(agent_context, "progress_reporter"):
                agent_context.progress_reporter = original_progress_reporter

            # Remove MCP context reference
            if hasattr(agent_context, "mcp_context"):
                delattr(agent_context, "mcp_context")

    async def _cleanup_stdio(self):
        """Minimal cleanup for STDIO transport to avoid keeping process alive."""
        logger.info("Performing minimal STDIO cleanup")

        await self._dispose_primary_instance()
        await self._dispose_all_stale_instances()
        await self._dispose_all_connection_instances()

        logger.info("STDIO cleanup complete")

    async def shutdown(self):
        """Gracefully shutdown the MCP server and its resources."""
        logger.info("Running full shutdown procedure")

        # Skip if already in shutdown
        if self._graceful_shutdown_event.is_set():
            return

        # Signal shutdown
        self._graceful_shutdown_event.set()

        try:
            # Close SSE connections
            await self._close_sse_connections()

            # Close any resources in the exit stack
            await self._exit_stack.aclose()

            # Dispose connection-scoped instances
            await self._dispose_all_connection_instances()

            # Dispose shared instance if still active
            await self._dispose_primary_instance()
            await self._dispose_all_stale_instances()
        except Exception as e:
            # Log any errors but don't let them prevent shutdown
            logger.error(f"Error during shutdown: {e}", exc_info=True)
        finally:
            logger.info("Full shutdown complete")

    async def _cleanup_minimal(self):
        """Perform minimal cleanup before simulating a KeyboardInterrupt."""
        logger.info("Performing minimal cleanup before interrupt")

        # Only close SSE connection writers directly
        mcp_ext = cast("_FastMCPLocalExtensions", self.mcp_server)
        sse = getattr(mcp_ext, "_sse_transport", None)
        if sse is not None:
            # Close all read stream writers
            for session_id, writer in list(sse._read_stream_writers.items()):
                try:
                    await writer.aclose()
                except Exception:
                    # Ignore errors during cleanup
                    pass

        # Clear active connections set to prevent further operations
        self._active_connections.clear()
