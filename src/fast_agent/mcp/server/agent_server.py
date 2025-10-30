"""
Enhanced AgentMCPServer with robust shutdown handling for SSE transport.
"""

import asyncio
import logging
import os
import signal
import time
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Awaitable, Callable, Set

from mcp.server.fastmcp import Context as MCPContext
from mcp.server.fastmcp import FastMCP

import fast_agent.core
import fast_agent.core.prompt
from fast_agent.core.fastagent import AgentInstance
from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


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
    ) -> None:
        """Initialize the server with the provided agent app."""
        self.primary_instance = primary_instance
        self._create_instance_task = create_instance
        self._dispose_instance_task = dispose_instance
        self._instance_scope = instance_scope
        self.mcp_server: FastMCP = FastMCP(
            name=server_name,
            instructions=server_description
            or f"This server provides access to {len(primary_instance.agents)} agents",
        )
        if self._instance_scope == "request":
            # Ensure FastMCP does not attempt to maintain sessions for stateless mode
            self.mcp_server.settings.stateless_http = True
        self._tool_description = tool_description
        self._shared_instance_active = True
        # Shutdown coordination
        self._graceful_shutdown_event = asyncio.Event()
        self._force_shutdown_event = asyncio.Event()
        self._shutdown_timeout = 5.0  # Seconds to wait for graceful shutdown

        # Resource management
        self._exit_stack = AsyncExitStack()
        self._active_connections: Set[any] = set()

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

    def register_agent_tools(self, agent_name: str) -> None:
        """Register tools for a specific agent."""

        # Basic send message tool
        tool_description = (
            self._tool_description.format(agent=agent_name)
            if self._tool_description and "{agent}" in self._tool_description
            else self._tool_description
        )

        @self.mcp_server.tool(
            name=f"{agent_name}_send",
            description=tool_description or f"Send a message to the {agent_name} agent",
            structured_output=False,
            # MCP 1.10.1 turns every tool in to a structured output
        )
        async def send_message(message: str, ctx: MCPContext) -> str:
            """Send a message to the agent and return its response."""
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

                response = await agent.send(message)
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
                if not hasattr(agent, "_llm") or agent._llm is None:
                    return []

                # Convert the multipart message history to standard PromptMessages
                multipart_history = agent._llm.message_history
                prompt_messages = fast_agent.core.prompt.Prompt.from_multipart(multipart_history)

                # In FastMCP, we need to return the raw list of messages
                return [{"role": msg.role, "content": msg.content} for msg in prompt_messages]
            finally:
                await self._release_instance(ctx, instance, reuse_connection=True)

    async def _acquire_instance(self, ctx: MCPContext | None) -> AgentInstance:
        if self._instance_scope == "shared":
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
        if self._instance_scope == "request":
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

    async def _dispose_primary_instance(self) -> None:
        if self._shared_instance_active:
            try:
                await self._dispose_instance_task(self.primary_instance)
            finally:
                self._shared_instance_active = False

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

    def run(self, transport: str = "http", host: str = "0.0.0.0", port: int = 8000) -> None:
        """Run the MCP server synchronously."""
        if transport in ["sse", "http"]:
            self.mcp_server.settings.host = host
            self.mcp_server.settings.port = port

            # For synchronous run, we can use the simpler approach
            try:
                # Add any server attributes that might help with shutdown
                if not hasattr(self.mcp_server, "_server_should_exit"):
                    self.mcp_server._server_should_exit = False

                # Run the server
                self.mcp_server.run(transport=transport)
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
                    asyncio.run(self.shutdown())
                except (SystemExit, KeyboardInterrupt):
                    # These are expected during shutdown
                    pass
        else:  # stdio
            try:
                self.mcp_server.run(transport=transport)
            except KeyboardInterrupt:
                print("\nServer stopped by user (CTRL+C)")
            finally:
                # Minimal cleanup for stdio
                asyncio.run(self._cleanup_stdio())

    async def run_async(
        self, transport: str = "http", host: str = "0.0.0.0", port: int = 8000
    ) -> None:
        """Run the MCP server asynchronously with improved shutdown handling."""
        # Use different handling strategies based on transport type
        if transport in ["sse", "http"]:
            # For SSE/HTTP, use our enhanced shutdown handling
            self._setup_signal_handlers()

            self.mcp_server.settings.host = host
            self.mcp_server.settings.port = port

            # Start the server in a separate task so we can monitor it
            self._server_task = asyncio.create_task(self._run_server_with_shutdown(transport))

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

    async def _run_server_with_shutdown(self, transport: str):
        """Run the server with proper shutdown handling."""
        # This method is used for SSE/HTTP transport
        if transport not in ["sse", "http"]:
            raise ValueError("This method should only be used with SSE or HTTP transport")

        # Start a monitor task for shutdown
        shutdown_monitor = asyncio.create_task(self._monitor_shutdown())

        try:
            # Patch SSE server to track connections if needed
            if hasattr(self.mcp_server, "_sse_transport") and self.mcp_server._sse_transport:
                # Store the original connect_sse method
                original_connect = self.mcp_server._sse_transport.connect_sse

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
                self.mcp_server._sse_transport.connect_sse = tracked_connect_sse

            # Run the server based on transport type
            if transport == "sse":
                await self.mcp_server.run_sse_async()
            elif transport == "http":
                await self.mcp_server.run_streamable_http_async()
        finally:
            # Cancel the monitor when the server exits
            shutdown_monitor.cancel()
            try:
                await shutdown_monitor
            except asyncio.CancelledError:
                pass

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
                if hasattr(conn, "close"):
                    await conn.close()
                elif hasattr(conn, "aclose"):
                    await conn.aclose()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
            self._active_connections.discard(conn)

        # Access the SSE transport if it exists to close stream writers
        if (
            hasattr(self.mcp_server, "_sse_transport")
            and self.mcp_server._sse_transport is not None
        ):
            sse = self.mcp_server._sse_transport

            # Close all read stream writers
            if hasattr(sse, "_read_stream_writers"):
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
        if (
            hasattr(self.mcp_server, "_lifespan_state")
            and self.mcp_server._lifespan_state == "started"
        ):
            logger.debug("Attempting to signal ASGI lifespan shutdown")
            try:
                if hasattr(self.mcp_server, "_on_shutdown"):
                    await self.mcp_server._on_shutdown()
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
        except Exception as e:
            # Log any errors but don't let them prevent shutdown
            logger.error(f"Error during shutdown: {e}", exc_info=True)
        finally:
            logger.info("Full shutdown complete")

    async def _cleanup_minimal(self):
        """Perform minimal cleanup before simulating a KeyboardInterrupt."""
        logger.info("Performing minimal cleanup before interrupt")

        # Only close SSE connection writers directly
        if (
            hasattr(self.mcp_server, "_sse_transport")
            and self.mcp_server._sse_transport is not None
        ):
            sse = self.mcp_server._sse_transport

            # Close all read stream writers
            if hasattr(sse, "_read_stream_writers"):
                for session_id, writer in list(sse._read_stream_writers.items()):
                    try:
                        await writer.aclose()
                    except Exception:
                        # Ignore errors during cleanup
                        pass

        # Clear active connections set to prevent further operations
        self._active_connections.clear()
