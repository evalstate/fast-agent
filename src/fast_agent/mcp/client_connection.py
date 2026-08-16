"""High-level MCP client connection owned by fast-agent."""

from __future__ import annotations

import json
from contextlib import AsyncExitStack, suppress
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from mcp.client import CacheConfig, CacheMode, Client, Transport
from mcp.shared.exceptions import MCPError
from mcp_types import (
    INVALID_REQUEST,
    CallToolResult,
    CompleteResult,
    ContentBlock,
    DiscoverResult,
    EmbeddedResource,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    PromptReference,
    ReadResourceResult,
    RequestParamsMeta,
    ResourceLink,
    ResourceTemplateReference,
    TextContent,
)
from mcp_types.version import LATEST_MODERN_VERSION, MODERN_PROTOCOL_VERSIONS

from fast_agent.core.exceptions import ServerSessionTerminatedError
from fast_agent.mcp.skills_extension import (
    DirectoryReadRequest,
    DirectoryReadRequestParams,
    GetSkillRequest,
    GetSkillRequestParams,
    GetSkillResult,
    ListSkillsRequest,
    ListSkillsRequestParams,
    ListSkillsResult,
)
from fast_agent.mcp.tool_result_metadata import set_url_elicitation_required_payload
from fast_agent.mcp.uri_security import SANITIZED_INLINE_RESOURCE_URI, is_file_uri
from fast_agent.mcp.url_elicitation_required import (
    URLElicitationRequiredDisplayPayload,
    build_url_elicitation_required_display_payload,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from types import TracebackType

    from mcp.shared.dispatcher import ProgressFnT

    from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime

URL_ELICITATION_REQUIRED = -32042
T = TypeVar("T")


def sdk_connect_mode(protocol_mode: Literal["auto", "modern", "legacy"]) -> str:
    """Map fast-agent's era-oriented setting to the SDK connect mode."""
    return LATEST_MODERN_VERSION if protocol_mode == "modern" else protocol_mode


class _ForcedModernClient(Client):
    """Enter atomically after a successful modern discovery request."""

    async def __aenter__(self) -> _ForcedModernClient:
        if self._entered:
            raise RuntimeError("Client is already entered; cannot reenter")
        self._entered = True

        error: Exception | None = None
        async with AsyncExitStack() as exit_stack:
            session = await self._build_session(exit_stack)
            session = await exit_stack.enter_async_context(session)
            # Let cancellation unwind the nested task groups in-place; re-raise
            # ordinary negotiation failures only after a clean exit.
            try:
                raw = await session.send_discover(LATEST_MODERN_VERSION)
                session.adopt(DiscoverResult.model_validate(raw))
            except Exception as exc:
                error = exc
            else:
                self._session = session
                self._exit_stack = exit_stack.pop_all()
                return self

        assert error is not None
        raise error


class MCPClientConnection:
    """Compose the SDK client with fast-agent callback and extension behavior."""

    def __init__(
        self,
        transport: Transport,
        callbacks: MCPClientCallbackRuntime,
        *,
        read_timeout_seconds: float | None = None,
        cache: bool = True,
        protocol_mode: Literal["auto", "modern", "legacy"] = "auto",
    ) -> None:
        self.callbacks = callbacks
        client_type = _ForcedModernClient if protocol_mode == "modern" else Client
        self.client = client_type(
            transport,
            mode=sdk_connect_mode(protocol_mode),
            read_timeout_seconds=read_timeout_seconds,
            sampling_callback=callbacks.sampling_callback,
            sampling_capabilities=callbacks.sampling_capabilities,
            list_roots_callback=callbacks.list_roots_callback,
            elicitation_callback=callbacks.elicitation_callback,
            message_handler=callbacks.message_handler,
            client_info=callbacks.client_info,
            cache=CacheConfig() if cache else None,
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
    def discover_result(self) -> DiscoverResult | None:
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
        return await self.client.session.send_ping()

    async def list_tools(
        self,
        *,
        cursor: str | None = None,
        meta: RequestParamsMeta | None = None,
        cache_mode: CacheMode = "use",
    ) -> ListToolsResult:
        return await self._request(
            self.client.list_tools(cursor=cursor, meta=meta, cache_mode=cache_mode)
        )

    async def list_prompts(
        self,
        *,
        cursor: str | None = None,
        meta: RequestParamsMeta | None = None,
        cache_mode: CacheMode = "use",
    ) -> ListPromptsResult:
        return await self._request(
            self.client.list_prompts(cursor=cursor, meta=meta, cache_mode=cache_mode)
        )

    async def list_resources(
        self,
        *,
        cursor: str | None = None,
        meta: RequestParamsMeta | None = None,
        cache_mode: CacheMode = "use",
    ) -> ListResourcesResult:
        return await self._request(
            self.client.list_resources(cursor=cursor, meta=meta, cache_mode=cache_mode)
        )

    async def list_resource_templates(
        self,
        *,
        cursor: str | None = None,
        meta: RequestParamsMeta | None = None,
        cache_mode: CacheMode = "use",
    ) -> ListResourceTemplatesResult:
        return await self._request(
            self.client.list_resource_templates(
                cursor=cursor,
                meta=meta,
                cache_mode=cache_mode,
            )
        )

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
            self._sanitize_call_tool_result,
        )

    async def read_resource(
        self,
        uri: str,
        *,
        meta: RequestParamsMeta | None = None,
        cache_mode: CacheMode = "use",
    ) -> ReadResourceResult:
        return await self._interactive_operation(
            "resources/read",
            self.client.read_resource(uri, meta=meta, cache_mode=cache_mode),
            self._sanitize_read_resource_result,
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
            self._sanitize_get_prompt_result,
        )

    async def complete(
        self,
        ref: ResourceTemplateReference | PromptReference,
        argument: dict[str, str],
        context_arguments: dict[str, str] | None = None,
    ) -> CompleteResult:
        return await self._request(self.client.complete(ref, argument, context_arguments))

    async def list_skills(self, *, cursor: str | None = None) -> ListSkillsResult:
        """List skills published under the pinned SEP-2640 draft."""
        request = ListSkillsRequest(params=ListSkillsRequestParams(cursor=cursor))
        return await self._request(self.client.session.send_request(request, ListSkillsResult))

    async def get_skill(self, uri: str) -> GetSkillResult:
        """Get a single skill entry under the pinned SEP-2640 draft."""
        request = GetSkillRequest(params=GetSkillRequestParams(uri=uri))
        return await self._request(self.client.session.send_request(request, GetSkillResult))

    async def read_directory(
        self,
        uri: str,
        *,
        cursor: str | None = None,
    ) -> ListResourcesResult:
        request = DirectoryReadRequest(params=DirectoryReadRequestParams(uri=uri, cursor=cursor))
        return await self._request(self.client.session.send_request(request, ListResourcesResult))

    async def _request(self, operation: Awaitable[T]) -> T:
        try:
            return await operation
        except MCPError as exc:
            if (
                exc.code == INVALID_REQUEST
                and exc.message == "Session terminated"
                and self.protocol_version not in MODERN_PROTOCOL_VERSIONS
            ):
                raise ServerSessionTerminatedError(
                    server_name=self.callbacks.display_server_name,
                    details="Server returned 404 - runtime may need replacement",
                ) from exc
            raise

    async def _interactive_operation(
        self,
        method: str,
        operation: Awaitable[T],
        sanitize: Callable[[T], T],
    ) -> T:
        self.callbacks.discard_pending_url_elicitations()
        try:
            result = sanitize(await self._request(operation))
        except ServerSessionTerminatedError:
            self.callbacks.discard_pending_url_elicitations()
            raise
        except MCPError as exc:
            self.callbacks.discard_pending_url_elicitations()
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

    def _sanitize_call_tool_result(self, result: CallToolResult) -> CallToolResult:
        content = [self._sanitize_content_block(item) for item in result.content]
        if all(new is old for new, old in zip(content, result.content, strict=True)):
            return result
        return result.model_copy(update={"content": content})

    def _sanitize_get_prompt_result(self, result: GetPromptResult) -> GetPromptResult:
        messages = [
            message.model_copy(update={"content": self._sanitize_content_block(message.content)})
            for message in result.messages
        ]
        if all(
            new.content is old.content for new, old in zip(messages, result.messages, strict=True)
        ):
            return result
        return result.model_copy(update={"messages": messages})

    def _sanitize_read_resource_result(self, result: ReadResourceResult) -> ReadResourceResult:
        contents = [
            resource.model_copy(update={"uri": SANITIZED_INLINE_RESOURCE_URI})
            if is_file_uri(str(resource.uri))
            else resource
            for resource in result.contents
        ]
        if all(new is old for new, old in zip(contents, result.contents, strict=True)):
            return result
        return result.model_copy(update={"contents": contents})

    @staticmethod
    def _sanitize_content_block(content: ContentBlock) -> ContentBlock:
        if isinstance(content, ResourceLink) and is_file_uri(str(content.uri)):
            return TextContent(
                type="text",
                text="[Local file attachment from a remote MCP server was blocked.]",
            )
        if isinstance(content, EmbeddedResource) and is_file_uri(str(content.resource.uri)):
            resource = content.resource.model_copy(update={"uri": SANITIZED_INLINE_RESOURCE_URI})
            return content.model_copy(update={"resource": resource})
        return content
