"""
A derived client session for the MCP Agent framework.
It adds logging and supports sampling requests.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import suppress
from typing import TYPE_CHECKING, Any, TypeVar, cast

from mcp import ClientSession
from mcp.client._input_required import run_input_required_driver
from mcp.client.subscriptions import (
    PromptsListChanged,
    ResourcesListChanged,
    ServerEvent,
    ToolsListChanged,
)
from mcp_types import (
    URL_ELICITATION_REQUIRED,
    CallToolResult,
    ClientRequest,
    EmptyResult,
    ErrorData,
    GetPromptResult,
    Implementation,
    InputRequest,
    InputRequiredResult,
    InputResponse,
    InputResponses,
    ListResourcesResult,
    ListRootsResult,
    PaginatedRequestParams,
    ProgressNotification,
    ReadResourceResult,
    Request,
    RequestParamsMeta,
    Root,
    SamplingCapability,
    SamplingToolsCapability,
    ToolListChangedNotification,
)
from pydantic import AnyUrl, BaseModel, FileUrl, TypeAdapter
from typing_extensions import Literal

from fast_agent.context_dependent import ContextDependent
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.server_config_helpers import get_server_config
from fast_agent.mcp.sampling import resolve_auto_sampling_enabled, sample
from fast_agent.mcp.tool_result_metadata import (
    set_url_elicitation_required_payload,
    update_tool_result_display_metadata,
    url_elicitation_required_payload,
)
from fast_agent.mcp.url_elicitation_required import (
    URLElicitationDisplayItem,
    URLElicitationRequiredDisplayPayload,
    build_url_elicitation_required_display_payload,
)
from fast_agent.utils.env import env_flag
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from mcp.client.session import ClientRequestContext, ListRootsFnT, SamplingFnT
    from mcp.shared.dispatcher import ProgressFnT
    from mcp.shared.message import ClientMessageMetadata

    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.transport_tracking import TransportChannelMetrics

logger = get_logger(__name__)
ReceiveResultT = TypeVar("ReceiveResultT", bound=BaseModel)


class DirectoryReadRequestParams(PaginatedRequestParams):
    """Parameters for the SEP-2640 ``resources/directory/read`` method."""

    uri: str


class DirectoryReadRequest(
    Request[DirectoryReadRequestParams, Literal["resources/directory/read"]]
):
    """List the direct children of a directory resource (SEP-2640)."""

    method: Literal["resources/directory/read"] = "resources/directory/read"
    params: DirectoryReadRequestParams


def _progress_trace_enabled() -> bool:
    return env_flag("FAST_AGENT_TRACE_MCP_PROGRESS")


def _progress_trace(message: str) -> None:
    if not _progress_trace_enabled():
        return
    print(f"[mcp-progress-trace] {message}", file=sys.stderr, flush=True)


_URL_ELICITATION_RESULT_PREFIX = "fast-agent-url-elicitation-required:"


async def list_roots(context: ClientRequestContext) -> ListRootsResult:
    """List roots callback that will be called by the MCP library."""

    if (server_config := get_server_config(context.session)) and server_config.roots:
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

    _pending_url_elicitations: list[URLElicitationDisplayItem]

    def __init__(self, read_stream, write_stream, read_timeout=None, **kwargs) -> None:
        custom_elicitation_handler = self._pop_fast_agent_kwargs(kwargs)
        self._initialize_session_state()

        fast_agent = self._client_implementation()
        list_roots_cb = self._make_list_roots_callback()
        sampling_cb = self._make_sampling_callback()
        sampling_caps = self._make_sampling_capabilities(sampling_cb)
        elicitation_handler = self._resolve_elicitation_handler(custom_elicitation_handler)
        self.effective_elicitation_mode = self._resolve_effective_elicitation_mode(
            elicitation_handler
        )
        self._discard_managed_client_kwargs(kwargs)

        super().__init__(
            read_stream,
            write_stream,
            read_timeout,
            **kwargs,
            list_roots_callback=list_roots_cb,
            sampling_callback=sampling_cb,
            sampling_capabilities=sampling_caps,
            client_info=fast_agent,
            elicitation_callback=elicitation_handler,
            message_handler=self._handle_message,
        )

    def _pop_fast_agent_kwargs(self, kwargs: dict[str, Any]) -> Any:
        self.session_server_name = kwargs.pop("server_name", None)
        self._tool_list_changed_callback = kwargs.pop("tool_list_changed_callback", None)
        self._aggregator = kwargs.pop("aggregator", None)
        self.server_config: MCPServerSettings | None = kwargs.pop("server_config", None)
        self.agent_model: str | None = kwargs.pop("agent_model", None)
        self.agent_name: str | None = kwargs.pop("agent_name", None)
        self.api_key: str | None = kwargs.pop("api_key", None)
        self._context = kwargs.pop("context", None)
        self._transport_metrics: TransportChannelMetrics | None = kwargs.pop(
            "transport_metrics", None
        )
        return kwargs.pop("elicitation_handler", None)

    def _initialize_session_state(self) -> None:
        self.effective_elicitation_mode: str | None = "none"
        self._offline_notified = False
        self._pending_url_elicitations = []

    def _client_implementation(self) -> Implementation:
        from importlib.metadata import version

        if self.server_config and self.server_config.implementation:
            return self.server_config.implementation
        fast_agent_version = version("fast-agent-mcp") or "dev"
        return Implementation(name="fast-agent-mcp", version=fast_agent_version)

    def _make_list_roots_callback(self) -> ListRootsFnT | None:
        if self.server_config and self.server_config.roots:
            return cast("ListRootsFnT", list_roots)
        return None

    def _make_sampling_callback(self) -> SamplingFnT | None:
        if (
            self.server_config and self.server_config.sampling
        ) or self._should_enable_auto_sampling():
            return cast("SamplingFnT", sample)
        return None

    @staticmethod
    def _make_sampling_capabilities(
        sampling_cb: SamplingFnT | None,
    ) -> SamplingCapability | None:
        if sampling_cb is None:
            return None
        return SamplingCapability(tools=SamplingToolsCapability())

    def _resolve_elicitation_handler(self, custom_elicitation_handler: Any) -> Any | None:
        if custom_elicitation_handler is not None:
            return custom_elicitation_handler

        elicitation_handler = self._resolve_configured_elicitation_handler()
        if elicitation_handler is not None or self.server_config:
            return elicitation_handler

        from fast_agent.mcp.elicitation_handlers import forms_elicitation_handler

        return forms_elicitation_handler

    def _resolve_configured_elicitation_handler(self) -> Any | None:
        try:
            from fast_agent.agents.agent_types import AgentConfig
            from fast_agent.context import get_current_context
            from fast_agent.mcp.elicitation_factory import resolve_elicitation_handler

            context = get_current_context()
            if not context or not context.config:
                return None
            agent_config = AgentConfig(
                name=self.agent_name or "unknown",
                model=self.agent_model or "unknown",
                elicitation_handler=None,
            )
            return resolve_elicitation_handler(agent_config, context.config, self.server_config)
        except Exception:
            return None

    def _resolve_effective_elicitation_mode(self, elicitation_handler: Any | None) -> str:
        if self.server_config and self.server_config.elicitation is not None:
            return self.server_config.elicitation.mode or "forms"
        if elicitation_handler is None:
            return "none"
        return self._global_elicitation_mode() or "forms"

    @staticmethod
    def _global_elicitation_mode() -> str | None:
        from fast_agent.context import get_current_context
        from fast_agent.mcp.elicitation_factory import resolve_global_elicitation_mode

        context = get_current_context()
        if not context or not context.config:
            return None
        return resolve_global_elicitation_mode(context.config)

    @staticmethod
    def _discard_managed_client_kwargs(kwargs: dict[str, Any]) -> None:
        for key in (
            "list_roots_callback",
            "sampling_callback",
            "sampling_capabilities",
            "client_info",
            "elicitation_callback",
            "message_handler",
        ):
            kwargs.pop(key, None)

    def _should_enable_auto_sampling(self) -> bool:
        """Check if auto_sampling is enabled at the application level."""
        try:
            from fast_agent.context import get_current_context

            return resolve_auto_sampling_enabled(get_current_context())
        except Exception:
            return True

    async def send_request(
        self,
        request: ClientRequest | Request[Any, Any],
        result_type: type[ReceiveResultT] | TypeAdapter[ReceiveResultT],
        request_read_timeout_seconds: float | None = None,
        metadata: ClientMessageMetadata | None = None,
        progress_callback: ProgressFnT | None = None,
    ) -> ReceiveResultT:
        logger.debug("send_request: request=", data=request.model_dump())
        request_id = None
        is_ping_request = self._is_ping_request(request)
        request_method = request.method

        self._trace_request_progress(
            "outbound-request",
            request_method=request_method,
            request_id=request_id,
            progress_callback=progress_callback,
            extra=f" progress_token={request_id!r}",
        )
        self._register_ping_request(is_ping_request, request_id)
        try:
            result = await super().send_request(
                # NOTE: request must be positional due to an upstream bug in
                # opentelemetry-instrumentation-mcp (seen in 0.52.1) where the
                # wrapper expects args[0] and can return None when request is
                # only provided via kwargs.
                # TODO: revert to keyword argument once upstream handles kwargs.
                request,
                result_type=result_type,
                request_read_timeout_seconds=request_read_timeout_seconds,
                metadata=metadata,
                progress_callback=progress_callback,
            )
            self._handle_successful_request(
                result,
                is_ping_request=is_ping_request,
                request_id=request_id,
                request_method=request_method,
                progress_callback=progress_callback,
            )
            return result
        except Exception as e:
            self._handle_failed_request(
                e,
                is_ping_request=is_ping_request,
                request_id=request_id,
                request_method=request_method,
                progress_callback=progress_callback,
            )
            raise

    def _trace_request_progress(
        self,
        event: str,
        *,
        request_method: str,
        request_id: Any,
        progress_callback: ProgressFnT | None,
        extra: str = "",
    ) -> None:
        if progress_callback is None or request_id is None:
            return
        _progress_trace(
            f"{event} "
            f"server={self.session_server_name or 'unknown'} "
            f"method={request_method} "
            f"request_id={request_id!r}"
            f"{extra}"
        )

    def _register_ping_request(self, is_ping_request: bool, request_id: Any) -> None:
        if is_ping_request and request_id is not None and self._transport_metrics is not None:
            self._transport_metrics.register_ping_request(request_id)

    def _discard_ping_request(self, is_ping_request: bool, request_id: Any) -> None:
        if is_ping_request and request_id is not None and self._transport_metrics is not None:
            self._transport_metrics.discard_ping_request(request_id)

    def _handle_successful_request(
        self,
        result: ReceiveResultT,
        *,
        is_ping_request: bool,
        request_id: Any,
        request_method: str,
        progress_callback: ProgressFnT | None,
    ) -> None:
        logger.debug(
            "send_request: response=",
            data=result.model_dump() if result is not None else "no response returned",
        )

        self._trace_request_progress(
            "request-complete",
            request_method=request_method,
            request_id=request_id,
            progress_callback=progress_callback,
        )

        self._attach_transport_channel(request_id, result)
        self._attach_url_elicitation_payload_from_result(
            result,
            request_method=request_method,
        )
        self._attach_pending_url_elicitation_payload_for_request(
            result,
            request_method=request_method,
        )
        self._discard_ping_request(is_ping_request, request_id)
        self._offline_notified = False

    def _handle_failed_request(
        self,
        exc: Exception,
        *,
        is_ping_request: bool,
        request_id: Any,
        request_method: str,
        progress_callback: ProgressFnT | None,
    ) -> None:
        from anyio import ClosedResourceError

        from fast_agent.core.exceptions import ServerSessionTerminatedError

        self._discard_pending_url_elicitation_payload()
        self._trace_request_progress(
            "request-error",
            request_method=request_method,
            request_id=request_id,
            progress_callback=progress_callback,
            extra=f" error={type(exc).__name__}: {exc}",
        )
        self._discard_ping_request(is_ping_request, request_id)

        if self._is_session_terminated_error(exc):
            raise ServerSessionTerminatedError(
                server_name=self.session_server_name or "unknown",
                details="Server returned 404 - session may have expired due to server restart",
            ) from exc

        if self._is_url_elicitation_required_error(exc):
            self._attach_url_elicitation_required_payload(exc, request_method)

        if isinstance(exc, ClosedResourceError):
            self._raise_connection_offline(exc)

        logger.error(f"send_request failed: {exc!s}")

    def _raise_connection_offline(self, exc: Exception) -> None:
        if not self._offline_notified:
            from fast_agent.ui import console

            console.console.print(
                f"[dim red]MCP server {self.session_server_name} offline[/dim red]"
            )
            self._offline_notified = True
        raise ConnectionError(f"MCP server {self.session_server_name} offline") from exc

    @staticmethod
    def _is_ping_request(request: Any) -> bool:
        method = request.method
        if not isinstance(method, str):
            return False
        method_lower = strip_casefold(method)
        return method_lower == "ping" or method_lower.endswith(("/ping", ".ping"))

    def _is_session_terminated_error(self, exc: Exception) -> bool:
        """Check if exception is a session terminated error (code 32600 from 404)."""
        from mcp.shared.exceptions import MCPError

        from fast_agent.core.exceptions import ServerSessionTerminatedError

        return (
            isinstance(exc, MCPError)
            and exc.code == ServerSessionTerminatedError.SESSION_TERMINATED_CODE
        )

    def _is_url_elicitation_required_error(self, exc: Exception) -> bool:
        """Check if exception is URL elicitation required error (-32042)."""
        from mcp.shared.exceptions import MCPError

        return isinstance(exc, MCPError) and exc.code == URL_ELICITATION_REQUIRED

    def _attach_url_elicitation_required_payload(self, exc: Exception, request_method: str) -> None:
        """Attach parsed URL elicitation data to exception for deferred display."""
        from mcp.shared.exceptions import MCPError

        if not isinstance(exc, MCPError):
            return

        server_name = self.session_server_name or "unknown"
        payload = build_url_elicitation_required_display_payload(
            exc.data,
            server_name=server_name,
            request_method=request_method,
        )
        set_url_elicitation_required_payload(exc, payload)

    def queue_url_elicitation_for_active_request(
        self,
        *,
        message: str,
        url: str,
        elicitation_id: str | None,
    ) -> bool:
        """Queue URL elicitation for the next successful request result."""
        self._pending_url_elicitations.append(
            URLElicitationDisplayItem(
                message=message,
                url=url,
                elicitation_id=elicitation_id or "",
            )
        )
        return True

    def _attach_pending_url_elicitation_payload_for_request(
        self,
        result: object,
        *,
        request_method: str,
    ) -> None:
        payload = self._consume_pending_url_elicitation_payload(request_method=request_method)
        if payload is None:
            return
        with suppress(Exception):
            set_url_elicitation_required_payload(result, payload)

    def _consume_pending_url_elicitation_payload(
        self,
        *,
        request_method: str,
    ) -> URLElicitationRequiredDisplayPayload | None:
        if not self._pending_url_elicitations:
            return None

        items = self._pending_url_elicitations
        self._pending_url_elicitations = []
        return URLElicitationRequiredDisplayPayload(
            server_name=self.session_server_name or "unknown",
            request_method=request_method,
            elicitations=items,
            issues=[],
        )

    def _discard_pending_url_elicitation_payload(self) -> None:
        self._pending_url_elicitations = []

    def _attach_url_elicitation_payload_from_result(
        self,
        result: object,
        *,
        request_method: str,
    ) -> None:
        if not isinstance(result, CallToolResult) or not result.is_error or not result.content:
            return

        first_block = result.content[0]
        text = getattr(first_block, "text", None)
        if not isinstance(text, str) or _URL_ELICITATION_RESULT_PREFIX not in text:
            return

        _, _, raw_payload = text.partition(_URL_ELICITATION_RESULT_PREFIX)
        try:
            payload_data = json.loads(raw_payload.strip())
        except json.JSONDecodeError:
            return
        if not isinstance(payload_data, dict):
            return

        payload = build_url_elicitation_required_display_payload(
            payload_data,
            server_name=self.session_server_name or "unknown",
            request_method=request_method,
        )
        with suppress(Exception):
            set_url_elicitation_required_payload(result, payload)

    @staticmethod
    def get_url_elicitation_required_payload(
        exc: object,
    ) -> URLElicitationRequiredDisplayPayload | None:
        """Return deferred URL elicitation display payload when present."""
        return url_elicitation_required_payload(exc)

    def _attach_transport_channel(self, request_id, result) -> None:
        if self._transport_metrics is None or request_id is None or result is None:
            return
        channel = self._transport_metrics.consume_response_channel(request_id)
        if not channel:
            return
        with suppress(Exception):
            update_tool_result_display_metadata(result, {"transport_channel": channel})

    async def _handle_message(self, message: object) -> None:
        if isinstance(message, ToolListChangedNotification):
            if self._tool_list_changed_callback and self.session_server_name:
                asyncio.create_task(
                    self._handle_tool_list_change_callback(self.session_server_name)
                )
        if self._aggregator and not isinstance(message, ProgressNotification | Exception):
            asyncio.create_task(self._handle_server_notification(message))

    async def _handle_server_notification(self, notification: object) -> None:
        """Forward server notifications to the registered callback."""
        _cb = (
            getattr(self._aggregator, "server_notification_callback", None)
            if self._aggregator
            else None
        )
        if not _cb:
            return
        try:
            await _cb(
                self.session_server_name or "unknown",
                notification,
            )
        except Exception as e:
            logger.warning(
                f"Error in server notification callback for '{self.session_server_name}': {e}"
            )

    async def handle_subscription_event(self, event: ServerEvent) -> None:
        if self._aggregator is None or self.session_server_name is None:
            return
        if isinstance(event, ToolsListChanged):
            await self._aggregator._handle_tool_list_changed(self.session_server_name)
        elif isinstance(event, PromptsListChanged):
            await self._aggregator._fetch_and_cache_prompts(self.session_server_name)
        elif isinstance(event, ResourcesListChanged):
            await self._aggregator._refresh_server_resources(self.session_server_name)

    async def _dispatch_input(
        self,
        key: str,
        request: InputRequest,
    ) -> InputResponse | ErrorData:
        from mcp.client.session import ClientRequestContext

        context = ClientRequestContext(
            session=self,
            request_id=key,
            meta=request.params.meta if request.params else None,
        )
        return await self.dispatch_input_request(context, request)

    async def _handle_tool_list_change_callback(self, server_name: str) -> None:
        """
        Helper method to handle tool list change callback in a separate task
        to prevent blocking the notification handler
        """
        try:
            await self._tool_list_changed_callback(server_name)
        except Exception as e:
            logger.error(f"Error in tool list changed callback: {e}")

    async def call_tool_complete(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
        progress_callback: ProgressFnT | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Call a tool with optional metadata and progress callback support.

        Always uses our overridden send_request to ensure session terminated errors
        are properly detected and converted to ServerSessionTerminatedError.
        """
        first = await super().call_tool(
            name,
            arguments,
            read_timeout_seconds,
            progress_callback,
            meta=cast("RequestParamsMeta | None", meta),
            allow_input_required=True,
        )
        if not isinstance(first, InputRequiredResult):
            return first

        async def retry(
            responses: InputResponses | None,
            request_state: str | None,
        ) -> CallToolResult | InputRequiredResult:
            return await super(MCPAgentClientSession, self).call_tool(
                name,
                arguments,
                read_timeout_seconds,
                progress_callback,
                input_responses=responses,
                request_state=request_state,
                meta=cast("RequestParamsMeta | None", meta),
                allow_input_required=True,
            )

        return await run_input_required_driver(
            first,
            dispatch=self._dispatch_input,
            retry=retry,
        )

    async def ping(self, read_timeout_seconds: float | None = None) -> EmptyResult:
        """Send a ping request to check server liveness."""
        del read_timeout_seconds
        return await self.send_ping()

    async def read_resource_complete(
        self,
        uri: AnyUrl | str,
        *,
        meta: RequestParamsMeta | None = None,
    ) -> ReadResourceResult:
        """Read a resource with optional metadata support.

        Always uses our overridden send_request to ensure session terminated errors
        are properly detected and converted to ServerSessionTerminatedError.
        """
        first = await super().read_resource(str(uri), meta=meta, allow_input_required=True)
        if not isinstance(first, InputRequiredResult):
            return first

        async def retry(
            responses: InputResponses | None,
            request_state: str | None,
        ) -> ReadResourceResult | InputRequiredResult:
            return await super(MCPAgentClientSession, self).read_resource(
                str(uri),
                input_responses=responses,
                request_state=request_state,
                meta=meta,
                allow_input_required=True,
            )

        return await run_input_required_driver(
            first,
            dispatch=self._dispatch_input,
            retry=retry,
        )

    async def read_directory(
        self,
        uri: AnyUrl | str,
        *,
        cursor: str | None = None,
    ) -> ListResourcesResult:
        """List the direct children of a directory resource (SEP-2640).

        Sends ``resources/directory/read`` and reuses the ``resources/list``
        result shape (``resources`` + ``nextCursor``). Callers MUST only invoke
        this against servers that declared ``directoryRead``. The method is not in
        the closed ``ClientRequest`` union, so the bare request goes through our
        ``send_request`` override; wrapping it would emit union serializer warnings.
        """
        params = DirectoryReadRequestParams(uri=str(uri), cursor=cursor)
        request = DirectoryReadRequest(method="resources/directory/read", params=params)
        return await self.send_request(request, ListResourcesResult)

    async def get_prompt_complete(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
        *,
        meta: RequestParamsMeta | None = None,
    ) -> GetPromptResult:
        """Get a prompt with optional metadata support.

        Always uses our overridden send_request to ensure session terminated errors
        are properly detected and converted to ServerSessionTerminatedError.
        """
        first = await super().get_prompt(
            name,
            arguments,
            meta=meta,
            allow_input_required=True,
        )
        if not isinstance(first, InputRequiredResult):
            return first

        async def retry(
            responses: InputResponses | None,
            request_state: str | None,
        ) -> GetPromptResult | InputRequiredResult:
            return await super(MCPAgentClientSession, self).get_prompt(
                name,
                arguments,
                input_responses=responses,
                request_state=request_state,
                meta=meta,
                allow_input_required=True,
            )

        return await run_input_required_driver(
            first,
            dispatch=self._dispatch_input,
            retry=retry,
        )
