"""
A derived client session for the MCP Agent framework.
It adds logging and supports sampling requests.
"""

import json
from datetime import timedelta
from typing import TYPE_CHECKING

from mcp import ClientSession, ServerNotification
from mcp.shared.message import MessageMetadata
from mcp.shared.session import (
    ProgressFnT,
    ReceiveResultT,
    SendRequestT,
)
from mcp.types import (
    ElicitRequestParams,
    ElicitResult,
    Implementation,
    ListRootsResult,
    Root,
    ToolListChangedNotification,
)
from pydantic import FileUrl

from mcp_agent.context_dependent import ContextDependent
from mcp_agent.human_input.elicitation_handler import elicitation_input_callback
from mcp_agent.human_input.types import HumanInputRequest
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


async def elicit(ctx: ClientSession, params: ElicitRequestParams) -> ElicitResult:
    """
    Elicit a response from the user using enhanced input handler.
    """
    logger.info(f"Eliciting response for params: {params}")
    
    # Get server config for additional context
    server_config = get_server_config(ctx)
    server_name = server_config.name if server_config else "Unknown Server"
    server_info = {"command": server_config.command} if server_config and server_config.command else None
    
    # Get agent name - try multiple sources in order of preference
    agent_name: str | None = None
    
    # 1. Check if we have an MCPAgentClientSession in the context
    if hasattr(ctx, "session") and isinstance(ctx.session, MCPAgentClientSession):
        agent_name = ctx.session.agent_name
    
    # 2. If no agent name yet, use a sensible default
    if not agent_name:
        agent_name = "Unknown Agent"
    
    # Create human input request
    request = HumanInputRequest(
        prompt=params.message,
        description=f"Schema: {params.requestedSchema}" if params.requestedSchema else None,
        request_id=f"elicit_{id(params)}",
        metadata={
            "agent_name": agent_name,
            "server_name": server_name,
            "elicitation": True,
            "requested_schema": params.requestedSchema,
        }
    )
    
    try:
        # Call the enhanced elicitation handler
        response = await elicitation_input_callback(
            request=request,
            agent_name=agent_name,
            server_name=server_name,
            server_info=server_info,
        )
        
        # Check for special action responses
        response_data = response.response.strip()
        action_metadata = response.metadata.get("action") if response.metadata else None
        
        # Handle special responses
        if response_data == "__DECLINED__":
            return ElicitResult(action="decline")
        elif response_data == "__CANCELLED__":
            return ElicitResult(action="cancel")
        elif response_data == "__DISABLE_SERVER__":
            # Log that user wants to disable elicitation for this server
            logger.warning(f"User requested to disable elicitation for server: {server_name}")
            # For now, just cancel - in a full implementation, this would update server config
            return ElicitResult(action="cancel")
        
        # Parse response based on schema if provided
        if params.requestedSchema:
            # Check if the response is already JSON (from our form)
            try:
                # Try to parse as JSON first (from schema-driven form)
                content = json.loads(response_data)
                # Validate that all required fields are present
                required_fields = params.requestedSchema.get("required", [])
                for field in required_fields:
                    if field not in content:
                        logger.warning(f"Missing required field '{field}' in elicitation response")
                        return ElicitResult(action="decline")
            except json.JSONDecodeError:
                # Not JSON, try to handle as simple text response
                # This is a fallback for simple schemas or text-based responses
                properties = params.requestedSchema.get("properties", {})
                if len(properties) == 1:
                    # Single field schema - try to parse based on type
                    field_name = list(properties.keys())[0]
                    field_def = properties[field_name]
                    field_type = field_def.get("type")
                    
                    if field_type == "boolean":
                        # Parse boolean values
                        if response_data.lower() in ["yes", "y", "true", "1"]:
                            content = {field_name: True}
                        elif response_data.lower() in ["no", "n", "false", "0"]:
                            content = {field_name: False}
                        else:
                            return ElicitResult(action="decline")
                    elif field_type == "string":
                        content = {field_name: response_data}
                    elif field_type in ["number", "integer"]:
                        try:
                            value = int(response_data) if field_type == "integer" else float(response_data)
                            content = {field_name: value}
                        except ValueError:
                            return ElicitResult(action="decline")
                    else:
                        # Unknown type, just pass as string
                        content = {field_name: response_data}
                else:
                    # Multiple fields but text response - can't parse reliably
                    logger.warning("Text response provided for multi-field schema")
                    return ElicitResult(action="decline")
        else:
            # No schema, just return the raw response
            content = {"response": response_data}
        
        # Return the response wrapped in ElicitResult with accept action
        return ElicitResult(
            action="accept",
            content=content
        )
    except (KeyboardInterrupt, EOFError, TimeoutError):
        # User cancelled or timeout
        return ElicitResult(action="cancel")


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
        # Extract agent_name if provided
        self.agent_name: str | None = kwargs.pop("agent_name", None)

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
            elicitation_callback=elicit,
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
