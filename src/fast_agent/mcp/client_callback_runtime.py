"""Fast-agent callback configuration for an MCP client connection."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import TYPE_CHECKING, cast

from mcp_types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    CreateMessageResultWithTools,
    ErrorData,
    Implementation,
    ListRootsResult,
    ProgressNotification,
    Root,
    SamplingCapability,
    SamplingToolsCapability,
    ToolListChangedNotification,
)
from pydantic import FileUrl

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.elicitation_factory import (
    resolve_elicitation_handler,
    resolve_global_elicitation_mode,
)
from fast_agent.mcp.elicitation_handlers import (
    forms_elicitation_handler,
    make_forms_elicitation_handler,
)
from fast_agent.mcp.sampling import resolve_auto_sampling_enabled, sample

if TYPE_CHECKING:
    from mcp.client.session import (
        ClientRequestContext,
        ElicitationFnT,
        ListRootsFnT,
        MessageHandlerFnT,
        SamplingFnT,
    )
    from mcp.client.subscriptions import ServerEvent

    from fast_agent.config import MCPServerSettings
    from fast_agent.context import Context
    from fast_agent.mcp.mcp_aggregator import MCPAggregator
    from fast_agent.mcp.url_elicitation_required import URLElicitationDisplayItem

logger = get_logger(__name__)

type ToolListChangedCallback = Callable[[str], Awaitable[None]]
type TransportNotificationHandler = Callable[[str], None]
type AgentModelResolver = Callable[[], str | None]


@dataclass(slots=True)
class MCPClientCallbackRuntime:
    """Compose fast-agent callbacks and capabilities for an MCP SDK client.

    The object deliberately owns client-specific state rather than attaching it
    to an SDK protocol object. Its callback attributes are passed directly to
    ``mcp.client.Client``.
    """

    server_name: str | None
    server_config: MCPServerSettings | None
    agent_model: str | None = None
    agent_model_resolver: AgentModelResolver | None = None
    agent_name: str | None = None
    api_key: str | None = None
    custom_elicitation_handler: ElicitationFnT | None = None
    aggregator: MCPAggregator | None = None
    context: Context | None = None
    tool_list_changed_callback: ToolListChangedCallback | None = None
    transport_notification_handler: TransportNotificationHandler | None = None
    subscription_ready: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    effective_elicitation_mode: str = field(init=False)
    client_info: Implementation = field(init=False)
    list_roots_callback: ListRootsFnT | None = field(init=False)
    sampling_callback: SamplingFnT | None = field(init=False)
    sampling_capabilities: SamplingCapability | None = field(init=False)
    elicitation_callback: ElicitationFnT | None = field(init=False)
    message_handler: MessageHandlerFnT = field(init=False)
    _pending_url_elicitations: list[URLElicitationDisplayItem] = field(
        init=False,
        default_factory=list,
    )

    def __post_init__(self) -> None:
        if self.aggregator is None:
            self.subscription_ready.set()
        self.client_info = self._client_implementation()
        self.list_roots_callback = self._make_list_roots_callback()
        self.sampling_callback = self._make_sampling_callback()
        self.sampling_capabilities = self._make_sampling_capabilities()
        self.elicitation_callback = self._resolve_elicitation_handler()
        self.effective_elicitation_mode = self._resolve_effective_elicitation_mode()
        self.message_handler = self._handle_message

    @property
    def display_server_name(self) -> str:
        """Return the configured server name suitable for UI and notifications."""
        if self.server_name:
            return self.server_name
        if self.server_config and self.server_config.name:
            return self.server_config.name
        return "unknown"

    def _client_implementation(self) -> Implementation:
        if self.server_config and self.server_config.implementation:
            return self.server_config.implementation
        return Implementation(name="fast-agent-mcp", version=version("fast-agent-mcp") or "dev")

    def _current_agent_model(self) -> str | None:
        if self.agent_model_resolver is not None:
            return self.agent_model_resolver()
        return self.agent_model

    def _make_list_roots_callback(self) -> ListRootsFnT | None:
        if self.server_config is None or not self.server_config.roots:
            return None
        roots = self.server_config.roots

        async def list_roots(context: ClientRequestContext) -> ListRootsResult:
            del context
            return ListRootsResult(
                roots=[
                    Root(uri=FileUrl(root.server_uri_alias or root.uri), name=root.name)
                    for root in roots
                ]
            )

        return cast("ListRootsFnT", list_roots)

    def _make_sampling_callback(self) -> SamplingFnT | None:
        if self.server_config and self.server_config.sampling:
            return self._sampling_callback
        if self._should_enable_auto_sampling():
            return self._sampling_callback
        return None

    async def _sampling_callback(
        self,
        context: ClientRequestContext,
        params: CreateMessageRequestParams,
    ) -> CreateMessageResult | CreateMessageResultWithTools | ErrorData:
        del context
        return await sample(
            params,
            server_name=self.display_server_name,
            server_config=self.server_config,
            agent_model=self._current_agent_model(),
            api_key=self.api_key,
            app_context=self._app_context(),
        )

    def _make_sampling_capabilities(self) -> SamplingCapability | None:
        if self.sampling_callback is None:
            return None
        return SamplingCapability(tools=SamplingToolsCapability())

    def _resolve_elicitation_handler(self) -> ElicitationFnT | None:
        if self.custom_elicitation_handler is not None:
            return self.custom_elicitation_handler

        app_context = self._app_context()
        if app_context is not None and app_context.config is not None:
            handler = resolve_elicitation_handler(
                AgentConfig(
                    name=self.agent_name or "unknown",
                    model=self._current_agent_model() or "unknown",
                    elicitation_handler=None,
                ),
                app_context.config,
                self.server_config,
            )
            if handler is forms_elicitation_handler:
                return self._forms_elicitation_handler()
            return handler

        if self.server_config is not None:
            return None

        return self._forms_elicitation_handler()

    def _forms_elicitation_handler(self) -> ElicitationFnT:
        server_info = None
        if self.server_config is not None and self.server_config.command is not None:
            server_info = {"command": self.server_config.command}
        return make_forms_elicitation_handler(
            agent_name=self.agent_name or "Unknown Agent",
            server_name=self.display_server_name,
            server_info=server_info,
            queue_url_elicitation=self.queue_url_elicitation,
        )

    def _resolve_effective_elicitation_mode(self) -> str:
        if self.server_config and self.server_config.elicitation is not None:
            return self.server_config.elicitation.mode or "forms"
        if self.elicitation_callback is None:
            return "none"

        app_context = self._app_context()
        if app_context is not None and app_context.config is not None:
            return resolve_global_elicitation_mode(app_context.config) or "forms"
        return "forms"

    def _should_enable_auto_sampling(self) -> bool:
        return resolve_auto_sampling_enabled(self._app_context())

    def _app_context(self) -> Context | None:
        if self.context is not None:
            return self.context

        from fast_agent.context import get_initialized_context

        return get_initialized_context()

    def queue_url_elicitation(
        self,
        *,
        message: str,
        url: str,
        elicitation_id: str | None,
    ) -> bool:
        """Queue URL elicitation for attachment to the active request result."""
        from fast_agent.mcp.url_elicitation_required import URLElicitationDisplayItem

        self._pending_url_elicitations.append(
            URLElicitationDisplayItem(
                message=message,
                url=url,
                elicitation_id=elicitation_id or "",
            )
        )
        return True

    def consume_pending_url_elicitations(self) -> list[URLElicitationDisplayItem]:
        """Return and clear URL elicitations received during the active request."""
        items = self._pending_url_elicitations
        self._pending_url_elicitations = []
        return items

    def discard_pending_url_elicitations(self) -> None:
        """Discard URL elicitations when the associated request fails."""
        self._pending_url_elicitations = []

    async def _handle_message(self, message: object) -> None:
        if isinstance(message, ProgressNotification) and self.transport_notification_handler:
            self.transport_notification_handler("notifications/progress")
        if (
            isinstance(message, ToolListChangedNotification)
            and self.tool_list_changed_callback is not None
        ):
            asyncio.create_task(self._notify_tool_list_changed())
        if self.aggregator is not None and not isinstance(
            message, ProgressNotification | Exception
        ):
            asyncio.create_task(self._notify_server_notification(message))

    async def _notify_tool_list_changed(self) -> None:
        if self.tool_list_changed_callback is None:
            return
        try:
            await self.tool_list_changed_callback(self.display_server_name)
        except Exception as exc:
            logger.error(f"Error in tool list changed callback: {exc}")

    async def _notify_server_notification(self, notification: object) -> None:
        if self.aggregator is None:
            return
        callback = self.aggregator.server_notification_callback
        if callback is None:
            return
        try:
            await callback(self.display_server_name, notification)
        except Exception as exc:
            logger.warning(
                f"Error in server notification callback for '{self.display_server_name}': {exc}"
            )

    async def handle_subscription_event(self, event: ServerEvent) -> None:
        """Bridge a typed modern subscription event to authoritative derived state."""
        if self.aggregator is None:
            return
        await self.aggregator.handle_subscription_event(self.display_server_name, event)

    def subscription_resource_uris(self) -> tuple[str, ...]:
        """Return the aggregator's canonical materialized UI resource selection."""
        if self.aggregator is None:
            return ()
        return self.aggregator.selected_materialized_resource_uris(self.display_server_name)

    async def refresh_subscription_state(self) -> tuple[str, ...]:
        """Force authoritative attached discovery after a listen acknowledgment."""
        if self.aggregator is None:
            return ()
        return await self.aggregator.refresh_subscription_state(self.display_server_name)

    async def wait_until_subscription_ready(self) -> None:
        """Wait until initial attachment discovery has been committed."""
        await self.subscription_ready.wait()

    def mark_subscription_ready(self) -> None:
        """Release the listener after initial attachment discovery commits."""
        self.subscription_ready.set()
