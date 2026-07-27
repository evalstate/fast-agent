"""High-level MCP client connection owned by fast-agent."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from mcp.client import Client, Transport
from mcp.shared.exceptions import MCPError
from mcp_types import (
    CallToolResult,
    CompleteResult,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    PaginatedRequestParams,
    PromptReference,
    ReadResourceResult,
    Request,
    RequestParamsMeta,
    ResourceTemplateReference,
    TextContent,
)
from mcp_types.version import MODERN_PROTOCOL_VERSIONS

from fast_agent.core.exceptions import ServerSessionTerminatedError
from fast_agent.mcp.tool_result_metadata import set_url_elicitation_required_payload
from fast_agent.mcp.url_elicitation_required import (
    URLElicitationRequiredDisplayPayload,
    build_url_elicitation_required_display_payload,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

    from mcp.shared.dispatcher import ProgressFnT

    from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime

URL_ELICITATION_REQUIRED = -32042
T = TypeVar("T")


class DirectoryReadRequestParams(PaginatedRequestParams):
    """Parameters for the SEP-2640 directory-read extension."""

    uri: str


class DirectoryReadRequest(
    Request[DirectoryReadRequestParams, Literal["resources/directory/read"]]
):
    """Request for the SEP-2640 directory-read extension."""

    method: Literal["resources/directory/read"] = "resources/directory/read"
    params: DirectoryReadRequestParams


class MCPClientConnection:
    """Compose the SDK client with fast-agent callback and extension behavior."""

    def __init__(
        self,
        transport: Transport,
        callbacks: MCPClientCallbackRuntime,
        *,
        read_timeout_seconds: float | None = None,
        cache: bool = True,
    ) -> None:
        self.callbacks = callbacks
        self.client = Client(
            transport,
            mode="auto",
            read_timeout_seconds=read_timeout_seconds,
            sampling_callback=callbacks.sampling_callback,
            sampling_capabilities=callbacks.sampling_capabilities,
            list_roots_callback=callbacks.list_roots_callback,
            elicitation_callback=callbacks.elicitation_callback,
            message_handler=callbacks.message_handler,
            client_info=callbacks.client_info,
            cache=None if cache else False,
        )

    async def __aenter__(self) -> MCPClientConnection:
        await self.client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.client.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def session(self):
        """Expose the SDK session only for diagnostics and extension requests."""
        return self.client.session

    @property
    def protocol_version(self) -> str:
        return self.client.protocol_version

    @property
    def server_info(self):
        return self.client.server_info

    @property
    def server_capabilities(self):
        return self.client.server_capabilities

    @property
    def instructions(self) -> str | None:
        return self.client.instructions

    @property
    def discover_result(self):
        return self.client.session.discover_result

    @property
    def effective_elicitation_mode(self) -> str:
        return self.callbacks.effective_elicitation_mode

    def listen(
        self,
        *,
        tools_list_changed: bool = False,
        prompts_list_changed: bool = False,
        resources_list_changed: bool = False,
        resource_subscriptions: tuple[str, ...] = (),
    ):
        return self.client.listen(
            tools_list_changed=tools_list_changed,
            prompts_list_changed=prompts_list_changed,
            resources_list_changed=resources_list_changed,
            resource_subscriptions=resource_subscriptions,
        )

    async def ping(self, read_timeout_seconds: float | None = None) -> object:
        """Send a legacy health ping; modern runtimes never call this."""
        del read_timeout_seconds
        return await self.client.send_ping()  # ty: ignore[deprecated]

    async def list_tools(
        self,
        *,
        cursor: str | None = None,
        meta: RequestParamsMeta | None = None,
    ) -> ListToolsResult:
        return await self.client.list_tools(cursor=cursor, meta=meta)

    async def list_prompts(
        self,
        *,
        cursor: str | None = None,
        meta: RequestParamsMeta | None = None,
    ) -> ListPromptsResult:
        return await self.client.list_prompts(cursor=cursor, meta=meta)

    async def list_resources(
        self,
        *,
        cursor: str | None = None,
        meta: RequestParamsMeta | None = None,
    ) -> ListResourcesResult:
        return await self.client.list_resources(cursor=cursor, meta=meta)

    async def list_resource_templates(
        self,
        *,
        cursor: str | None = None,
        meta: RequestParamsMeta | None = None,
    ) -> ListResourceTemplatesResult:
        return await self.client.list_resource_templates(cursor=cursor, meta=meta)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
        progress_callback: ProgressFnT | None = None,
        *,
        meta: RequestParamsMeta | None = None,
    ) -> CallToolResult:
        return await self._interactive_operation(
            "tools/call",
            self.client.call_tool(
                name,
                arguments,
                read_timeout_seconds,
                progress_callback,
                meta=meta,
            ),
        )

    async def read_resource(
        self,
        uri: str,
        *,
        meta: RequestParamsMeta | None = None,
    ) -> ReadResourceResult:
        return await self._interactive_operation(
            "resources/read",
            self.client.read_resource(uri, meta=meta),
        )

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
        *,
        meta: RequestParamsMeta | None = None,
    ) -> GetPromptResult:
        return await self._interactive_operation(
            "prompts/get",
            self.client.get_prompt(name, arguments, meta=meta),
        )

    async def complete(
        self,
        ref: ResourceTemplateReference | PromptReference,
        argument: dict[str, str],
        context_arguments: dict[str, str] | None = None,
    ) -> CompleteResult:
        return await self.client.complete(ref, argument, context_arguments)

    async def read_directory(
        self,
        uri: str,
        *,
        cursor: str | None = None,
    ) -> ListResourcesResult:
        request = DirectoryReadRequest(params=DirectoryReadRequestParams(uri=uri, cursor=cursor))
        return await self.client.session.send_request(request, ListResourcesResult)

    async def _interactive_operation(self, method: str, operation: Awaitable[T]) -> T:
        self.callbacks.discard_pending_url_elicitations()
        try:
            result = await operation
        except MCPError as exc:
            self.callbacks.discard_pending_url_elicitations()
            if (
                exc.code == ServerSessionTerminatedError.SESSION_TERMINATED_CODE
                and self.protocol_version not in MODERN_PROTOCOL_VERSIONS
            ):
                raise ServerSessionTerminatedError(
                    server_name=self.callbacks.display_server_name,
                    details="Server returned 404 - runtime may need replacement",
                ) from exc
            if exc.code == URL_ELICITATION_REQUIRED:
                payload = build_url_elicitation_required_display_payload(
                    exc.data,
                    server_name=self.callbacks.display_server_name,
                    request_method=method,
                )
                set_url_elicitation_required_payload(exc, payload)
            raise

        self._attach_pending_url_elicitations(result, method)
        self._attach_url_elicitation_result(result, method)
        return result

    def _attach_pending_url_elicitations(self, result: object, method: str) -> None:
        items = self.callbacks.consume_pending_url_elicitations()
        if not items:
            return
        payload = URLElicitationRequiredDisplayPayload(
            server_name=self.callbacks.display_server_name,
            request_method=method,
            elicitations=items,
            issues=[],
        )
        set_url_elicitation_required_payload(result, payload)

    def _attach_url_elicitation_result(self, result: object, method: str) -> None:
        if not isinstance(result, CallToolResult) or not result.is_error or not result.content:
            return
        first = result.content[0]
        text = first.text if isinstance(first, TextContent) else None
        marker = "fast-agent-url-elicitation-required:"
        if not isinstance(text, str) or marker not in text:
            return
        _, _, encoded = text.partition(marker)
        with suppress(json.JSONDecodeError):
            data = json.loads(encoded.strip())
            if isinstance(data, dict):
                payload = build_url_elicitation_required_display_payload(
                    data,
                    server_name=self.callbacks.display_server_name,
                    request_method=method,
                )
                set_url_elicitation_required_payload(result, payload)


class MCPTransportAdapter:
    """Adapt fast-agent's legacy three-value transport context to SDK Transport."""

    def __init__(self, context: AbstractAsyncContextManager[tuple[Any, Any, Any]]) -> None:
        self._context = context

    async def __aenter__(self):
        read_stream, write_stream, _session_id = await self._context.__aenter__()
        return read_stream, write_stream

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        return await self._context.__aexit__(exc_type, exc_val, exc_tb)
