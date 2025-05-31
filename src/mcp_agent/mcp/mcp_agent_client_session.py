"""
A derived client session for the MCP Agent framework.
It adds logging and supports sampling requests.
"""

from datetime import timedelta
from typing import TYPE_CHECKING, Optional

from mcp import ClientSession, ServerNotification
from mcp.shared.session import (
    ReceiveResultT,
    RequestId,
    SendNotificationT,
    SendRequestT,
    SendResultT,
)
from mcp.types import ErrorData, ListRootsResult, Root, ToolListChangedNotification
from pydantic import FileUrl

from mcp_agent.context_dependent import ContextDependent
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.sampling import sample

if TYPE_CHECKING:
    from mcp_agent.config import MCPServerSettings

logger = get_logger(__name__)


async def list_roots(ctx: ClientSession) -> ListRootsResult:
    """List roots callback that will be called by the MCP library."""

    roots = []
    if (
        hasattr(ctx, "session")
        and hasattr(ctx.session, "server_config")
        and ctx.session.server_config
        and hasattr(ctx.session.server_config, "roots")
        and ctx.session.server_config.roots
    ):
        roots = [
            Root(
                uri=FileUrl(
                    root.server_uri_alias or root.uri,
                ),
                name=root.name,
            )
            for root in ctx.session.server_config.roots
        ]
    return ListRootsResult(roots=roots or [])


class MCPAgentClientSession(ClientSession, ContextDependent):
    """
    MCP Agent framework acts as a client to the servers providing tools/resources/prompts for the agent workloads.
    This is a simple client session for those server connections, and supports
        - handling sampling requests
        - notifications
        - MCP root configuration

    Developers can extend this class to add more custom functionality as needed
    """

    def __init__(self, *args, **kwargs) -> None:
        # Extract server_name if provided in kwargs
        self.session_server_name = kwargs.pop("server_name", None)
        # Extract the notification callbacks if provided
        self._tool_list_changed_callback = kwargs.pop("tool_list_changed_callback", None)

        super().__init__(*args, **kwargs, list_roots_callback=list_roots, sampling_callback=sample)
        self.server_config: Optional[MCPServerSettings] = None

    async def send_request(
        self,
        request: SendRequestT,
        result_type: type[ReceiveResultT],
        request_read_timeout_seconds: timedelta | None = None,
    ) -> ReceiveResultT:
        logger.info("send_request: request=", data=request.model_dump())
        logger.debug("send_request: request=", data=request.model_dump())
        try:
            result = await super().send_request(
                request, result_type, request_read_timeout_seconds=request_read_timeout_seconds
            )
            logger.debug("send_request: response=", data=result.model_dump())
            return result
        except Exception as e:
            logger.error(f"send_request failed: {str(e)}")
            raise

    async def send_notification(self, notification: SendNotificationT) -> None:
        logger.debug("send_notification:", data=notification.model_dump())
        try:
            return await super().send_notification(notification)
        except Exception as e:
            logger.error("send_notification failed", data=e)
            raise

    async def _send_response(
        self, request_id: RequestId, response: SendResultT | ErrorData
    ) -> None:
        logger.debug(
            f"send_response: request_id={request_id}, response=",
            data=response.model_dump(),
        )
        return await super()._send_response(request_id, response)

    async def _received_notification(self, notification: ServerNotification) -> None:
        """
        Can be overridden by subclasses to handle a notification without needing
        to listen on the message stream.
        """
        logger.info(
            "_received_notification: notification=",
            data=notification.model_dump(),
        )

        # Call parent notification handler first
        await super()._received_notification(notification)

        # Then process our specific notification types
        match notification.root:
            case ToolListChangedNotification():
                # Simple notification handling - just call the callback if it exists
                if self._tool_list_changed_callback and self.session_server_name:
                    logger.info(
                        f"Tool list changed for server '{self.session_server_name}', triggering callback"
                    )
                    # Use asyncio.create_task to prevent blocking the notification handler
                    import asyncio
                    asyncio.create_task(self._handle_tool_list_change_callback(self.session_server_name))
                else:
                    logger.debug(
                        f"Tool list changed for server '{self.session_server_name}' but no callback registered"
                    )

        return None

    async def _handle_tool_list_change_callback(self, server_name: str) -> None:
        """
        Helper method to handle tool list change callback in a separate task
        to prevent blocking the notification handler
        """
        try:
            await self._tool_list_changed_callback(server_name)
        except Exception as e:
            logger.error(f"Error in tool list changed callback: {e}")

    async def send_progress_notification(
        self, progress_token: str | int, progress: float, total: float | None = None
    ) -> None:
        """
        Sends a progress notification for a request that is currently being
        processed.
        """
        logger.debug(
            "send_progress_notification: progress_token={progress_token}, progress={progress}, total={total}"
        )
        return await super().send_progress_notification(
            progress_token=progress_token, progress=progress, total=total
        )
