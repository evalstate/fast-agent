"""
AgentACPServer - Exposes FastAgent agents via the Agent Client Protocol (ACP).

This implementation allows fast-agent to act as an ACP agent, enabling editors
and other clients to interact with fast-agent agents over stdio using the ACP protocol.
"""

import asyncio
import logging
import uuid
from typing import Awaitable, Callable

from acp import Agent as ACPAgent
from acp import InitializeRequest, InitializeResponse, NewSessionRequest, NewSessionResponse
from acp import PromptRequest, PromptResponse
from acp import AgentSideConnection
from acp.schema import (
    AgentCapabilities,
    Implementation,
    PromptCapabilities,
    StopReason,
)
from acp.helpers import text_block, session_notification, update_agent_message_text
from acp.stdio import stdio_streams

from fast_agent.core.fastagent import AgentInstance
from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


class AgentACPServer(ACPAgent):
    """
    Exposes FastAgent agents as an ACP agent through stdio.

    This server:
    - Handles ACP connection initialization and capability negotiation
    - Manages sessions (maps sessionId to AgentInstance)
    - Routes prompts to the appropriate fast-agent agent
    - Returns responses in ACP format
    """

    def __init__(
        self,
        primary_instance: AgentInstance,
        create_instance: Callable[[], Awaitable[AgentInstance]],
        dispose_instance: Callable[[AgentInstance], Awaitable[None]],
        instance_scope: str,
        server_name: str = "fast-agent-acp",
        server_version: str = "0.1.0",
    ) -> None:
        """
        Initialize the ACP server.

        Args:
            primary_instance: The primary agent instance (used in shared mode)
            create_instance: Factory function to create new agent instances
            dispose_instance: Function to dispose of agent instances
            instance_scope: How to scope instances ('shared', 'connection', or 'request')
            server_name: Name of the server for capability advertisement
            server_version: Version of the server
        """
        super().__init__()

        self.primary_instance = primary_instance
        self._create_instance_task = create_instance
        self._dispose_instance_task = dispose_instance
        self._instance_scope = instance_scope
        self.server_name = server_name
        self.server_version = server_version

        # Session management
        self.sessions: dict[str, AgentInstance] = {}
        self._session_lock = asyncio.Lock()

        # Connection reference (set during run_async)
        self._connection: AgentSideConnection | None = None

        # For simplicity, use the first agent as the primary agent
        # In the future, we could add routing logic to select different agents
        self.primary_agent_name = list(primary_instance.agents.keys())[0] if primary_instance.agents else None

        logger.info(
            f"AgentACPServer initialized",
            name="acp_server_initialized",
            agent_count=len(primary_instance.agents),
            instance_scope=instance_scope,
            primary_agent=self.primary_agent_name,
        )

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        """
        Handle ACP initialization request.

        Negotiates protocol version and advertises capabilities.
        """
        try:
            logger.info(
                "ACP initialize request",
                name="acp_initialize",
                client_protocol=params.protocolVersion,
                client_info=params.clientInfo,
            )

            # Build our capabilities
            agent_capabilities = AgentCapabilities(
                prompts=PromptCapabilities(
                    supportedTypes=["text"],  # Start with text only
                ),
                # We don't support loadSession yet
                loadSession=False,
            )

            # Build agent info using Implementation type
            agent_info = Implementation(
                name=self.server_name,
                version=self.server_version,
            )

            response = InitializeResponse(
                protocolVersion=params.protocolVersion,  # Echo back the client's version
                agentCapabilities=agent_capabilities,
                agentInfo=agent_info,
                authMethods=[],  # No authentication for now
            )

            logger.info(
                "ACP initialize response sent",
                name="acp_initialize_response",
                protocol_version=response.protocolVersion,
            )

            return response
        except Exception as e:
            logger.error(f"Error in initialize: {e}", name="acp_initialize_error", exc_info=True)
            print(f"ERROR in initialize: {e}", file=__import__('sys').stderr)
            raise

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:
        """
        Handle new session request.

        Creates a new session and maps it to an AgentInstance based on instance_scope.
        """
        session_id = str(uuid.uuid4())

        logger.info(
            "ACP new session request",
            name="acp_new_session",
            session_id=session_id,
            instance_scope=self._instance_scope,
        )

        async with self._session_lock:
            # Determine which instance to use based on scope
            if self._instance_scope == "shared":
                # All sessions share the primary instance
                instance = self.primary_instance
            elif self._instance_scope in ["connection", "request"]:
                # Create a new instance for this session
                instance = await self._create_instance_task()
            else:
                # Default to shared
                instance = self.primary_instance

            self.sessions[session_id] = instance

        logger.info(
            "ACP new session created",
            name="acp_new_session_created",
            session_id=session_id,
            total_sessions=len(self.sessions),
        )

        return NewSessionResponse(sessionId=session_id)

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        """
        Handle prompt request.

        Extracts the prompt text, sends it to the fast-agent agent, and sends the response
        back to the client via sessionUpdate notifications.
        """
        session_id = params.sessionId

        logger.info(
            "ACP prompt request",
            name="acp_prompt",
            session_id=session_id,
        )

        # Get the agent instance for this session
        async with self._session_lock:
            instance = self.sessions.get(session_id)

        if not instance:
            logger.error(
                "ACP prompt error: session not found",
                name="acp_prompt_error",
                session_id=session_id,
            )
            # Return an error response
            return PromptResponse(stopReason=StopReason.REFUSAL)

        # Extract text content from the prompt
        text_parts = []
        for content_block in params.prompt:
            if hasattr(content_block, 'type') and content_block.type == "text":
                text_parts.append(content_block.text)

        prompt_text = "\n".join(text_parts)

        logger.info(
            "Sending prompt to fast-agent",
            name="acp_prompt_send",
            session_id=session_id,
            agent=self.primary_agent_name,
            prompt_length=len(prompt_text),
        )

        # Send to the fast-agent agent with streaming support
        try:
            if self.primary_agent_name:
                agent = instance.agents[self.primary_agent_name]

                # Set up streaming if connection is available
                stream_listener = None
                if self._connection:
                    # Accumulator for streamed chunks
                    accumulated_chunks = []
                    update_lock = asyncio.Lock()

                    async def send_stream_update(text: str):
                        """Send sessionUpdate with accumulated text so far."""
                        try:
                            async with update_lock:
                                message_chunk = update_agent_message_text(text)
                                notification = session_notification(session_id, message_chunk)
                                await self._connection.sessionUpdate(notification)
                        except Exception as e:
                            logger.error(
                                f"Error sending stream update: {e}",
                                name="acp_stream_error",
                                exc_info=True,
                            )

                    def on_stream_chunk(chunk: str):
                        """
                        Sync callback from fast-agent streaming.
                        Accumulates chunks and sends updates to ACP client.
                        """
                        accumulated_chunks.append(chunk)
                        current_text = "".join(accumulated_chunks)

                        logger.debug(
                            f"Stream chunk received: {len(chunk)} chars, total: {len(current_text)}",
                            name="acp_stream_chunk",
                            session_id=session_id,
                            chunk_count=len(accumulated_chunks),
                        )

                        # Send update asynchronously (don't await in sync callback)
                        asyncio.create_task(send_stream_update(current_text))

                    # Register the stream listener
                    stream_listener = on_stream_chunk
                    agent.add_stream_listener(stream_listener)

                    logger.info(
                        "Streaming enabled for prompt",
                        name="acp_streaming_enabled",
                        session_id=session_id,
                    )

                try:
                    # This will trigger streaming callbacks as chunks arrive
                    response_text = await agent.send(prompt_text)

                    logger.info(
                        "Received complete response from fast-agent",
                        name="acp_prompt_response",
                        session_id=session_id,
                        response_length=len(response_text),
                    )

                    # Send final sessionUpdate with complete response
                    if self._connection and response_text:
                        try:
                            message_chunk = update_agent_message_text(response_text)
                            notification = session_notification(session_id, message_chunk)
                            await self._connection.sessionUpdate(notification)
                            logger.info(
                                "Sent final sessionUpdate with complete response",
                                name="acp_final_update",
                                session_id=session_id,
                            )
                        except Exception as e:
                            logger.error(
                                f"Error sending final update: {e}",
                                name="acp_final_update_error",
                                exc_info=True,
                            )

                finally:
                    # Clean up stream listener
                    if stream_listener and hasattr(agent, '_stream_listeners'):
                        agent._stream_listeners.discard(stream_listener)
                        logger.info(
                            "Removed stream listener",
                            name="acp_streaming_cleanup",
                            session_id=session_id,
                        )

            else:
                logger.error("No primary agent available")
        except Exception as e:
            logger.error(
                f"Error processing prompt: {e}",
                name="acp_prompt_error",
                exc_info=True,
            )
            import sys
            import traceback
            print(f"ERROR processing prompt: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise

        # Return success
        return PromptResponse(
            stopReason=StopReason.END_TURN,
        )

    async def run_async(self) -> None:
        """
        Run the ACP server over stdio.

        This creates the stdio streams and sets up the ACP connection.
        """
        logger.info("Starting ACP server on stdio")
        print(f"Starting FastAgent '{self.server_name}' in ACP mode", file=__import__('sys').stderr)
        print(f"Instance scope: {self._instance_scope}", file=__import__('sys').stderr)
        print("Press Ctrl+C to stop", file=__import__('sys').stderr)

        try:
            # Get stdio streams
            reader, writer = await stdio_streams()

            # Create the ACP connection
            # Note: AgentSideConnection expects (writer, reader) order
            # - input_stream (writer) = where agent writes TO client
            # - output_stream (reader) = where agent reads FROM client
            connection = AgentSideConnection(
                lambda conn: self,
                writer,  # input_stream = StreamWriter for agent output
                reader,  # output_stream = StreamReader for agent input
            )

            # Store the connection reference so we can send sessionUpdate notifications
            self._connection = connection

            logger.info("ACP connection established, waiting for messages")

            # Keep the connection alive
            # The connection will handle incoming messages automatically
            # We just need to wait until it's closed or interrupted
            try:
                # Wait indefinitely - the connection will process messages in the background
                # The Connection class automatically starts a receive loop on creation
                shutdown_event = asyncio.Event()
                await shutdown_event.wait()
            except (asyncio.CancelledError, KeyboardInterrupt):
                logger.info("ACP server shutting down")
                print("\nServer stopped (Ctrl+C)", file=__import__('sys').stderr)
            finally:
                # Close the connection properly
                await connection._conn.close()

        except Exception as e:
            logger.error(f"ACP server error: {e}", name="acp_server_error", exc_info=True)
            raise

        finally:
            # Clean up sessions
            await self._cleanup_sessions()

    async def _cleanup_sessions(self) -> None:
        """Clean up all sessions and dispose of agent instances."""
        logger.info(f"Cleaning up {len(self.sessions)} sessions")

        async with self._session_lock:
            # Dispose of non-shared instances
            if self._instance_scope in ["connection", "request"]:
                for session_id, instance in self.sessions.items():
                    if instance != self.primary_instance:
                        try:
                            await self._dispose_instance_task(instance)
                        except Exception as e:
                            logger.error(
                                f"Error disposing instance for session {session_id}: {e}",
                                name="acp_cleanup_error",
                            )

            # Dispose of primary instance
            if self.primary_instance:
                try:
                    await self._dispose_instance_task(self.primary_instance)
                except Exception as e:
                    logger.error(
                        f"Error disposing primary instance: {e}",
                        name="acp_cleanup_error",
                    )

            self.sessions.clear()

        logger.info("ACP cleanup complete")
