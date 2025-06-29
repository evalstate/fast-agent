"""
A derived client session for the MCP Agent framework.
It adds logging and supports sampling requests.
"""

from datetime import timedelta
from typing import TYPE_CHECKING

from mcp import ClientSession, ServerNotification
from mcp.shared.message import MessageMetadata
from mcp.shared.session import (
    ProgressFnT,
    ReceiveResultT,
    SendRequestT,
)
from mcp.types import Implementation, ListRootsResult, Root, ToolListChangedNotification
from pydantic import FileUrl

from mcp_agent.context_dependent import ContextDependent
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.helpers.server_config_helpers import get_server_config
from mcp_agent.mcp.sampling import sample

if TYPE_CHECKING:
    from mcp_agent.config import MCPServerSettings

logger = get_logger(__name__)


async def list_roots(ctx: ClientSession) -> ListRootsResult:
    """List roots callback that will be called by the MCP library."""

    if server_config := get_server_config(ctx):
        if server_config.roots:
            roots = [
                Root(
                    uri=FileUrl(
                        root.server_uri_alias or root.uri,
                    ),
                    name=root.name,
                )
                for root in server_config.roots
            ]
            return ListRootsResult(roots=roots)

    return ListRootsResult(roots=[])


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
        from importlib.metadata import version

        version = version("fast-agent-mcp") or "dev"
        fast_agent: Implementation = Implementation(name="fast-agent-mcp", version=version)

        self.session_server_name = kwargs.pop("server_name", None)
        # Extract the notification callbacks if provided
        self._tool_list_changed_callback = kwargs.pop("tool_list_changed_callback", None)
        # Extract server_config if provided
        self.server_config: MCPServerSettings | None = kwargs.pop("server_config", None)
        # Extract agent_model if provided (for auto_sampling fallback)
        self.agent_model: str | None = kwargs.pop("agent_model", None)

        # Only register callbacks if the server_config has the relevant settings
        list_roots_cb = list_roots if (self.server_config and self.server_config.roots) else None

        # Register sampling callback if either:
        # 1. Sampling is explicitly configured, OR
        # 2. Application-level auto_sampling is enabled
        sampling_cb = None
        if (
            self.server_config
            and hasattr(self.server_config, "sampling")
            and self.server_config.sampling
        ):
            # Explicit sampling configuration
            sampling_cb = sample
        elif self._should_enable_auto_sampling():
            # Auto-sampling enabled at application level
            sampling_cb = sample

        super().__init__(
            *args,
            **kwargs,
            list_roots_callback=list_roots_cb,
            sampling_callback=sampling_cb,
            client_info=fast_agent,
        )

    def _should_enable_auto_sampling(self) -> bool:
        """Check if auto_sampling is enabled at the application level."""
        try:
            from mcp_agent.context import get_current_context

            context = get_current_context()
            if context and context.config:
                return getattr(context.config, "auto_sampling", True)
        except Exception:
            pass
        return True  # Default to True if can't access config

    async def send_request(
        self,
        request: SendRequestT,
        result_type: type[ReceiveResultT],
        request_read_timeout_seconds: timedelta | None = None,
        metadata: MessageMetadata | None = None,
        progress_callback: ProgressFnT | None = None,
    ) -> ReceiveResultT:
        logger.debug("send_request: request=", data=request.model_dump())
        try:
            result = await super().send_request(
                request=request,
                result_type=result_type,
                request_read_timeout_seconds=request_read_timeout_seconds,
                metadata=metadata,
                progress_callback=progress_callback,
            )
            logger.debug(
                "send_request: response=",
                data=result.model_dump() if result is not None else "no response returned",
            )
            return result
        except Exception as e:
            logger.error(f"send_request failed: {str(e)}")
            raise

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

                    asyncio.create_task(
                        self._handle_tool_list_change_callback(self.session_server_name)
                    )
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
