"""
ACP Elicitation State Management.

Manages session-scoped state for interactive elicitation when running via ACP.
Coordinates between the elicitation handler (waiting for response) and the
prompt handler (receiving user input).
"""

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from acp import AgentSideConnection
    from mcp.types import ElicitRequestParams

logger = get_logger(__name__)


@dataclass
class PendingElicitation:
    """Represents a pending elicitation request waiting for user response."""

    request_id: str
    """Unique identifier for this elicitation request."""

    params: "ElicitRequestParams"
    """Original elicitation parameters from MCP server."""

    response_future: asyncio.Future[dict[str, Any]]
    """Future that will be resolved with the user's response."""

    agent_name: str | None = None
    """Name of the agent that triggered the elicitation."""

    server_name: str | None = None
    """Name of the MCP server that requested elicitation."""

    field_names: list[str] = field(default_factory=list)
    """List of field names being collected (for parsing responses)."""


class ACPElicitationState:
    """
    Manages elicitation state for ACP sessions.

    This class coordinates between:
    1. The elicitation handler that sends questions and waits for responses
    2. The ACP prompt handler that receives user input

    Thread-safe for concurrent access across sessions.
    """

    def __init__(self) -> None:
        """Initialize the elicitation state manager."""
        # Map session_id -> pending elicitation
        self._pending: dict[str, PendingElicitation] = {}
        self._lock = asyncio.Lock()

        # Map session_id -> ACP connection (for sending updates)
        self._connections: dict[str, "AgentSideConnection"] = {}

    async def register_connection(
        self, session_id: str, connection: "AgentSideConnection"
    ) -> None:
        """
        Register an ACP connection for a session.

        Args:
            session_id: The ACP session ID
            connection: The ACP connection for sending updates
        """
        async with self._lock:
            self._connections[session_id] = connection
            logger.debug(
                "Registered ACP connection for elicitation",
                name="acp_elicitation_connection_registered",
                session_id=session_id,
            )

    async def unregister_connection(self, session_id: str) -> None:
        """
        Unregister an ACP connection when session ends.

        Args:
            session_id: The ACP session ID
        """
        async with self._lock:
            self._connections.pop(session_id, None)
            # Also cancel any pending elicitation for this session
            pending = self._pending.pop(session_id, None)
            if pending and not pending.response_future.done():
                pending.response_future.cancel()
            logger.debug(
                "Unregistered ACP connection for elicitation",
                name="acp_elicitation_connection_unregistered",
                session_id=session_id,
            )

    def get_connection(self, session_id: str) -> "AgentSideConnection | None":
        """
        Get the ACP connection for a session.

        Args:
            session_id: The ACP session ID

        Returns:
            The ACP connection or None if not registered
        """
        return self._connections.get(session_id)

    async def start_elicitation(
        self,
        session_id: str,
        request_id: str,
        params: "ElicitRequestParams",
        agent_name: str | None = None,
        server_name: str | None = None,
        field_names: list[str] | None = None,
    ) -> asyncio.Future[dict[str, Any]]:
        """
        Start a new elicitation request for a session.

        Args:
            session_id: The ACP session ID
            request_id: Unique identifier for this elicitation
            params: The elicitation parameters from MCP
            agent_name: Name of the requesting agent
            server_name: Name of the MCP server
            field_names: List of field names being collected

        Returns:
            A Future that will be resolved with the user's response

        Raises:
            RuntimeError: If there's already a pending elicitation for this session
        """
        async with self._lock:
            if session_id in self._pending:
                raise RuntimeError(
                    f"Session {session_id} already has a pending elicitation"
                )

            # Create the future for the response
            loop = asyncio.get_event_loop()
            response_future: asyncio.Future[dict[str, Any]] = loop.create_future()

            # Create the pending elicitation record
            pending = PendingElicitation(
                request_id=request_id,
                params=params,
                response_future=response_future,
                agent_name=agent_name,
                server_name=server_name,
                field_names=field_names or [],
            )

            self._pending[session_id] = pending

            logger.info(
                "Started ACP elicitation",
                name="acp_elicitation_started",
                session_id=session_id,
                request_id=request_id,
                field_count=len(pending.field_names),
            )

            return response_future

    async def has_pending_elicitation(self, session_id: str) -> bool:
        """
        Check if a session has a pending elicitation.

        Args:
            session_id: The ACP session ID

        Returns:
            True if there's a pending elicitation
        """
        async with self._lock:
            return session_id in self._pending

    async def get_pending_elicitation(
        self, session_id: str
    ) -> PendingElicitation | None:
        """
        Get the pending elicitation for a session without removing it.

        Args:
            session_id: The ACP session ID

        Returns:
            The pending elicitation or None
        """
        async with self._lock:
            return self._pending.get(session_id)

    async def resolve_elicitation(
        self, session_id: str, response: dict[str, Any]
    ) -> bool:
        """
        Resolve a pending elicitation with the user's response.

        Args:
            session_id: The ACP session ID
            response: The parsed response data

        Returns:
            True if an elicitation was resolved, False if none was pending
        """
        async with self._lock:
            pending = self._pending.pop(session_id, None)
            if pending is None:
                return False

            if not pending.response_future.done():
                pending.response_future.set_result(response)
                logger.info(
                    "Resolved ACP elicitation",
                    name="acp_elicitation_resolved",
                    session_id=session_id,
                    request_id=pending.request_id,
                )
                return True

            return False

    async def cancel_elicitation(
        self, session_id: str, reason: str = "cancelled"
    ) -> bool:
        """
        Cancel a pending elicitation.

        Args:
            session_id: The ACP session ID
            reason: Reason for cancellation

        Returns:
            True if an elicitation was cancelled, False if none was pending
        """
        async with self._lock:
            pending = self._pending.pop(session_id, None)
            if pending is None:
                return False

            if not pending.response_future.done():
                pending.response_future.set_exception(
                    asyncio.CancelledError(reason)
                )
                logger.info(
                    "Cancelled ACP elicitation",
                    name="acp_elicitation_cancelled",
                    session_id=session_id,
                    request_id=pending.request_id,
                    reason=reason,
                )
                return True

            return False


# Global singleton for ACP elicitation state
_acp_elicitation_state: ACPElicitationState | None = None


def get_acp_elicitation_state() -> ACPElicitationState:
    """
    Get the global ACP elicitation state manager.

    Returns:
        The singleton ACPElicitationState instance
    """
    global _acp_elicitation_state
    if _acp_elicitation_state is None:
        _acp_elicitation_state = ACPElicitationState()
    return _acp_elicitation_state
