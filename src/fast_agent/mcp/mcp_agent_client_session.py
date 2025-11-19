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
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    GetPromptRequest,
    GetPromptRequestParams,
    GetPromptResult,
    Implementation,
    ListRootsResult,
    ReadResourceRequest,
    ReadResourceRequestParams,
    ReadResourceResult,
    Root,
    ToolListChangedNotification,
)
from pydantic import FileUrl

from fast_agent.context_dependent import ContextDependent
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.server_config_helpers import get_server_config
from fast_agent.mcp.sampling import sample

if TYPE_CHECKING:
    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.transport_tracking import TransportChannelMetrics

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

        self.session_server_name = kwargs.pop("server_name", None)
        # Extract the notification callbacks if provided
        self._tool_list_changed_callback = kwargs.pop("tool_list_changed_callback", None)
        # Extract server_config if provided
        self.server_config: MCPServerSettings | None = kwargs.pop("server_config", None)
        # Extract agent_model if provided (for auto_sampling fallback)
        self.agent_model: str | None = kwargs.pop("agent_model", None)
        # Extract agent_name if provided
        self.agent_name: str | None = kwargs.pop("agent_name", None)
        # Extract api_key if provided
        self.api_key: str | None = kwargs.pop("api_key", None)
        # Extract custom elicitation handler if provided
        custom_elicitation_handler = kwargs.pop("elicitation_handler", None)
        # Extract optional context for ContextDependent mixin without passing it to ClientSession
        self._context = kwargs.pop("context", None)
        # Extract transport metrics tracker if provided
        self._transport_metrics: TransportChannelMetrics | None = kwargs.pop(
            "transport_metrics", None
        )

        # Track the effective elicitation mode for diagnostics
        self.effective_elicitation_mode: str | None = "none"

        version = version("fast-agent-mcp") or "dev"
        fast_agent: Implementation = Implementation(name="fast-agent-mcp", version=version)
        if self.server_config and self.server_config.implementation:
            fast_agent = self.server_config.implementation

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

        # Use custom elicitation handler if provided, otherwise resolve using factory
        if custom_elicitation_handler is not None:
            elicitation_handler = custom_elicitation_handler
        else:
            # Try to resolve using factory
            elicitation_handler = None
            try:
                from fast_agent.agents.agent_types import AgentConfig
                from fast_agent.context import get_current_context
                from fast_agent.mcp.elicitation_factory import resolve_elicitation_handler

                context = get_current_context()
                if context and context.config:
                    # Create a minimal agent config for the factory
                    agent_config = AgentConfig(
                        name=self.agent_name or "unknown",
                        model=self.agent_model or "unknown",
                        elicitation_handler=None,
                    )
                    elicitation_handler = resolve_elicitation_handler(
                        agent_config, context.config, self.server_config
                    )
            except Exception:
                # If factory resolution fails, we'll use default fallback
                pass

            # Fallback to forms handler only if factory resolution wasn't attempted
            if elicitation_handler is None and not self.server_config:
                from fast_agent.mcp.elicitation_handlers import forms_elicitation_handler

                elicitation_handler = forms_elicitation_handler

        # Determine effective elicitation mode for diagnostics
        if self.server_config and getattr(self.server_config, "elicitation", None):
            self.effective_elicitation_mode = self.server_config.elicitation.mode or "forms"
        elif elicitation_handler is not None:
            # Use global config if available to distinguish auto-cancel
            try:
                from fast_agent.context import get_current_context

                context = get_current_context()
                mode = None
                if context and getattr(context, "config", None):
                    elicitation_cfg = getattr(context.config, "elicitation", None)
                    if isinstance(elicitation_cfg, dict):
                        mode = elicitation_cfg.get("mode")
                    else:
                        mode = getattr(elicitation_cfg, "mode", None)
                self.effective_elicitation_mode = (mode or "forms").lower()
            except Exception:
                self.effective_elicitation_mode = "forms"
        else:
            self.effective_elicitation_mode = "none"

        super().__init__(
            *args,
            **kwargs,
            list_roots_callback=list_roots_cb,
            sampling_callback=sampling_cb,
            client_info=fast_agent,
            elicitation_callback=elicitation_handler,
        )

    def _should_enable_auto_sampling(self) -> bool:
        """Check if auto_sampling is enabled at the application level."""
        try:
            from fast_agent.context import get_current_context

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
        request_id = getattr(self, "_request_id", None)
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
            self._attach_transport_channel(request_id, result)
            return result
        except Exception as e:
            # Handle connection errors cleanly
            # Looking at the MCP SDK, this should probably handle MCPError
            from anyio import ClosedResourceError

            if isinstance(e, ClosedResourceError):
                # Show clean offline message and convert to ConnectionError
                from fast_agent.ui import console

                console.console.print(
                    f"[dim red]MCP server {self.session_server_name} offline[/dim red]"
                )
                raise ConnectionError(f"MCP server {self.session_server_name} offline") from e
            else:
                logger.error(f"send_request failed: {str(e)}")
                raise

    def _attach_transport_channel(self, request_id, result) -> None:
        if self._transport_metrics is None or request_id is None or result is None:
            return
        channel = self._transport_metrics.consume_response_channel(request_id)
        if not channel:
            return
        try:
            setattr(result, "transport_channel", channel)
        except Exception:
            # If result cannot be mutated, ignore silently
            pass

    async def _received_notification(self, notification: ServerNotification) -> None:
        """
        Can be overridden by subclasses to handle a notification without needing
        to listen on the message stream.
        """
        logger.debug(
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

    # TODO -- decide whether to make this override type safe or not (modify SDK)
    async def call_tool(
        self, name: str, arguments: dict | None = None, _meta: dict | None = None, **kwargs
    ) -> CallToolResult:
        """Call a tool with optional metadata support."""
        if _meta:
            from mcp.types import RequestParams

            # Safe merge - preserve existing meta fields like progressToken
            existing_meta = kwargs.get("meta")
            if existing_meta:
                meta_dict = (
                    existing_meta.model_dump() if hasattr(existing_meta, "model_dump") else {}
                )
                meta_dict.update(_meta)
                meta_obj = RequestParams.Meta(**meta_dict)
            else:
                meta_obj = RequestParams.Meta(**_meta)

            # Create CallToolRequestParams without meta, then add _meta via model_dump
            params = CallToolRequestParams(name=name, arguments=arguments)
            params_dict = params.model_dump(by_alias=True)
            params_dict["_meta"] = meta_obj.model_dump()

            # Create request with proper types
            request = CallToolRequest(
                method="tools/call", params=CallToolRequestParams.model_validate(params_dict)
            )

            return await self.send_request(request, CallToolResult)
        else:
            return await super().call_tool(name, arguments, **kwargs)

    async def read_resource(
        self, uri: str, _meta: dict | None = None, **kwargs
    ) -> ReadResourceResult:
        """Read a resource with optional metadata support."""
        if _meta:
            from mcp.types import RequestParams

            # Safe merge - preserve existing meta fields like progressToken
            existing_meta = kwargs.get("meta")
            if existing_meta:
                meta_dict = (
                    existing_meta.model_dump() if hasattr(existing_meta, "model_dump") else {}
                )
                meta_dict.update(_meta)
                meta_obj = RequestParams.Meta(**meta_dict)
            else:
                meta_obj = RequestParams.Meta(**_meta)

            request = ReadResourceRequest(
                method="resources/read", params=ReadResourceRequestParams(uri=uri, meta=meta_obj)
            )
            return await self.send_request(request, ReadResourceResult)
        else:
            return await super().read_resource(uri, **kwargs)

    async def get_prompt(
        self, name: str, arguments: dict | None = None, _meta: dict | None = None, **kwargs
    ) -> GetPromptResult:
        """Get a prompt with optional metadata support."""
        if _meta:
            from mcp.types import RequestParams

            # Safe merge - preserve existing meta fields like progressToken
            existing_meta = kwargs.get("meta")
            if existing_meta:
                meta_dict = (
                    existing_meta.model_dump() if hasattr(existing_meta, "model_dump") else {}
                )
                meta_dict.update(_meta)
                meta_obj = RequestParams.Meta(**meta_dict)
            else:
                meta_obj = RequestParams.Meta(**_meta)

            request = GetPromptRequest(
                method="prompts/get",
                params=GetPromptRequestParams(name=name, arguments=arguments, meta=meta_obj),
            )
            return await self.send_request(request, GetPromptResult)
        else:
            return await super().get_prompt(name, arguments, **kwargs)
