"""
ACP-based interactive elicitation handler.

This module provides an elicitation handler that works over ACP by converting
form-based elicitations into interactive Q&A sessions. Instead of displaying
a form UI, it asks questions one at a time through ACP's normal prompt flow.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult

from fast_agent.acp.acp_elicitation_context import (
    ACPElicitationContext,
    ElicitationPhase,
)
from fast_agent.acp.acp_elicitation_context import (
    ElicitationResult as InternalResult,
)
from fast_agent.acp.elicitation_questions import format_validation_error
from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp import ClientSession

logger = get_logger(__name__)


# Type for the message sender callback
MessageSenderT = Callable[[str], Awaitable[None]]


class ACPElicitationOrchestrator:
    """
    Orchestrates an interactive elicitation session over ACP.

    This class manages the back-and-forth conversation flow when collecting
    form data interactively. It's designed to be integrated with the ACP server
    to handle elicitation requests that arrive during agent execution.
    """

    def __init__(
        self,
        context: ACPElicitationContext,
        send_message: MessageSenderT,
    ):
        """
        Initialize the orchestrator.

        Args:
            context: The elicitation context with schema and state
            send_message: Async callback to send messages to the user via ACP
        """
        self.context = context
        self.send_message = send_message
        self._completion_event = asyncio.Event()

    async def start(self) -> None:
        """
        Start the elicitation session by sending the introduction.

        After calling this, the orchestrator waits for responses via handle_response().
        """
        intro_message = self.context.get_intro_message()
        await self.send_message(intro_message)

        # Move to intro phase - next response will be processed
        self.context.phase = ElicitationPhase.INTRO

        logger.info(
            "Started ACP elicitation session",
            name="acp_elicitation_started",
            session_id=self.context.session_id,
            server_name=self.context.server_name,
            question_count=self.context.total_questions,
        )

    async def handle_response(self, response: str) -> bool:
        """
        Handle a user response.

        This processes the response based on the current phase and sends
        the next message (question, error, or confirmation).

        Args:
            response: The user's response text

        Returns:
            True if the elicitation is complete, False if more input needed
        """
        if self.context.is_complete:
            return True

        phase = self.context.phase

        if phase == ElicitationPhase.INTRO:
            next_message = self.context.process_intro_response(response)
            if next_message:
                await self.send_message(next_message)
            elif self.context.is_complete:
                return True
            else:
                # Moved to questioning phase - send first question
                question_msg = self.context.get_current_question_message()
                if question_msg:
                    await self.send_message(question_msg)

        elif phase == ElicitationPhase.QUESTIONING:
            next_message, error = self.context.process_question_response(response)

            if error:
                # Validation error - send error message and re-ask
                error_msg = format_validation_error(error)
                question_msg = self.context.get_current_question_message()
                await self.send_message(f"{error_msg}\n\n{question_msg}")
            elif next_message:
                await self.send_message(next_message)
            elif self.context.is_complete:
                return True

        elif phase == ElicitationPhase.CONFIRMATION:
            next_message = self.context.process_confirmation_response(response)
            if next_message:
                await self.send_message(next_message)
            elif self.context.is_complete:
                return True

        return self.context.is_complete

    def get_result(self) -> InternalResult | None:
        """Get the final result once complete."""
        return self.context.get_result()


# Registry for active elicitation sessions by ACP session ID
_active_elicitations: dict[str, ACPElicitationOrchestrator] = {}


def get_active_elicitation(session_id: str) -> ACPElicitationOrchestrator | None:
    """Get the active elicitation orchestrator for a session, if any."""
    return _active_elicitations.get(session_id)


def set_active_elicitation(session_id: str, orchestrator: ACPElicitationOrchestrator | None) -> None:
    """Set or clear the active elicitation for a session."""
    if orchestrator is None:
        _active_elicitations.pop(session_id, None)
    else:
        _active_elicitations[session_id] = orchestrator


def has_active_elicitation(session_id: str) -> bool:
    """Check if a session has an active elicitation."""
    return session_id in _active_elicitations


async def create_acp_elicitation_handler(
    session_id: str,
    send_message: MessageSenderT,
) -> Callable[[RequestContext["ClientSession", Any], ElicitRequestParams], Awaitable[ElicitResult]]:
    """
    Create an elicitation handler configured for a specific ACP session.

    This factory creates a handler that:
    1. Converts elicitation requests into interactive Q&A sessions
    2. Uses the provided send_message callback to communicate with the user
    3. Tracks state across multiple prompt/response cycles

    Args:
        session_id: The ACP session ID
        send_message: Async callback to send messages to the user

    Returns:
        An elicitation handler function compatible with MCP
    """

    async def acp_elicitation_handler(
        context: RequestContext["ClientSession", Any],
        params: ElicitRequestParams,
    ) -> ElicitResult:
        """
        Handle an elicitation request via interactive ACP Q&A.

        This handler doesn't return immediately - it starts an interactive
        session and waits for the user to complete the Q&A flow.
        """
        logger.info(
            "ACP elicitation request received",
            name="acp_elicitation_request",
            session_id=session_id,
            message=params.message[:100] if params.message else None,
        )

        # Get agent and server names from context
        from fast_agent.mcp.helpers.server_config_helpers import get_server_config
        from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

        agent_name = "Agent"
        server_name = "MCP Server"

        if hasattr(context, "session") and isinstance(context.session, MCPAgentClientSession):
            agent_name = context.session.agent_name or agent_name

        server_config = get_server_config(context)
        if server_config:
            server_name = server_config.name or server_name

        # Create the elicitation context
        schema = params.requestedSchema or {"properties": {}, "required": []}
        elicitation_context = ACPElicitationContext(
            session_id=session_id,
            schema=schema,
            message=params.message,
            agent_name=agent_name,
            server_name=server_name,
        )

        # Check if there are any questions to ask
        if not elicitation_context.questions:
            # No fields in schema - just show message and auto-accept
            await send_message(
                f"---\n**{server_name}**\n\n{params.message}\n\n_No input required._\n---"
            )
            return ElicitResult(action="accept", content={})

        # Create orchestrator
        orchestrator = ACPElicitationOrchestrator(
            context=elicitation_context,
            send_message=send_message,
        )

        # Register as active elicitation
        set_active_elicitation(session_id, orchestrator)

        try:
            # Start the session - this sends the intro
            await orchestrator.start()

            # Wait for completion via the async event mechanism
            # The ACP server will call orchestrator.handle_response() for each prompt
            while not elicitation_context.is_complete:
                response = await elicitation_context.wait_for_response()
                await orchestrator.handle_response(response)

            # Get the result
            result = orchestrator.get_result()
            if result is None:
                return ElicitResult(action="cancel")

            logger.info(
                "ACP elicitation completed",
                name="acp_elicitation_completed",
                session_id=session_id,
                action=result.action,
            )

            return ElicitResult(action=result.action, content=result.data)

        finally:
            # Clean up
            set_active_elicitation(session_id, None)

    return acp_elicitation_handler


async def auto_cancel_acp_elicitation_handler(
    context: RequestContext["ClientSession", Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    """
    Handler that automatically cancels elicitation requests in ACP mode.

    Use this when you want to advertise elicitation capability but
    automatically decline all requests (e.g., for non-interactive ACP clients).
    """
    logger.info(f"Auto-cancelling ACP elicitation request: {params.message}")
    return ElicitResult(action="cancel")
