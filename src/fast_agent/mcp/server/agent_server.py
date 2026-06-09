"""Agent MCP server."""

import logging
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import version as get_version
from typing import Any, Literal, cast

from fastmcp import Context as MCPContext
from fastmcp import FastMCP
from fastmcp.prompts import Message
from fastmcp.server.auth import RemoteAuthProvider
from fastmcp.server.dependencies import get_access_token
from pydantic import AnyHttpUrl
from starlette.middleware import Middleware

import fast_agent.core.prompt
from fast_agent.core.agent_instance_factory import CallableAgentInstanceFactory
from fast_agent.core.fastagent import AgentInstance
from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.request_params import (
    ResponseMode,
    ToolResultMode,
    response_mode_to_tool_result_mode,
    tool_result_mode_allows_response_mode,
)
from fast_agent.mcp.auth.middleware import HFAuthHeaderMiddleware
from fast_agent.mcp.prompts.prompt_server import convert_to_fastmcp_messages
from fast_agent.mcp.server.instance_lease_pool import (
    AgentInstanceLease,
    InstanceScope,
    InstanceScopeValue,
    ScopedAgentInstancePool,
)
from fast_agent.mcp.tool_progress import MCPToolProgressManager
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.utils.async_utils import run_sync
from fast_agent.utils.text import strip_casefold

logger = get_logger(__name__)
_CONTEXT_ATTR_UNSET = object()


def _restore_context_attribute(
    target: Any,
    name: str,
    original_value: object,
) -> None:
    if original_value is _CONTEXT_ATTR_UNSET:
        with suppress(AttributeError):
            delattr(target, name)
        return
    setattr(target, name, original_value)


def _get_request_bearer_token() -> str | None:
    """Return the authenticated bearer token for the current MCP request."""
    access_token = get_access_token()
    if access_token is None:
        return None
    return access_token.token


def _get_fast_agent_version() -> str | None:
    for package_name in ("fast-agent-mcp", "fast-agent"):
        try:
            return get_version(package_name)
        except Exception:
            continue
    return None


def _normalize_serve_oauth_provider(provider: str | None) -> str | None:
    if provider is None:
        return None

    oauth_provider = strip_casefold(provider)
    if oauth_provider in {"hf", "huggingface"}:
        return "huggingface"
    if not oauth_provider:
        return None
    return oauth_provider


def _get_oauth_config() -> tuple[str | None, list[str], str]:
    """
    Read OAuth configuration from environment variables.

    Returns:
        Tuple of (provider, scopes, resource_url).
        provider is None if OAuth is not enabled.
    """
    oauth_provider = _normalize_serve_oauth_provider(os.environ.get("FAST_AGENT_SERVE_OAUTH"))

    oauth_scopes_str = os.environ.get("FAST_AGENT_OAUTH_SCOPES", "")
    oauth_scopes = [scope.strip() for scope in oauth_scopes_str.split(",") if scope.strip()] or [
        "access"
    ]
    resource_url = os.environ.get("FAST_AGENT_OAUTH_RESOURCE_URL", "http://localhost:8000")
    return oauth_provider, oauth_scopes, resource_url


def _history_to_fastmcp_messages(
    message_history: list[PromptMessageExtended],
) -> list[Message]:
    """Convert stored agent history into FastMCP prompt messages."""
    prompt_messages = fast_agent.core.prompt.Prompt.from_multipart(message_history)
    return convert_to_fastmcp_messages(prompt_messages)


TransportMode = Literal["http", "stdio"]


@dataclass(frozen=True, slots=True)
class _AgentToolRegistration:
    agent_name: str
    tool_name: str
    description: str
    response_mode_enabled: bool


@dataclass(frozen=True, slots=True)
class _AgentMetadata:
    description: str | None = None
    default_request_params: RequestParams | None = None


def _agent_metadata(agent: Any | None) -> _AgentMetadata:
    if agent is None:
        return _AgentMetadata()

    config = getattr(agent, "config", None)
    if config is None:
        return _AgentMetadata()

    description = getattr(config, "description", None)
    default_request_params = getattr(config, "default_request_params", None)
    return _AgentMetadata(
        description=description if isinstance(description, str) else None,
        default_request_params=(
            default_request_params if isinstance(default_request_params, RequestParams) else None
        ),
    )


class AgentMCPServer:
    """Exposes FastAgent agents as MCP tools through an MCP server."""

    def __init__(
        self,
        primary_instance: AgentInstance,
        create_instance: Callable[[], Awaitable[AgentInstance]],
        dispose_instance: Callable[[AgentInstance], Awaitable[None]],
        instance_scope: InstanceScopeValue,
        server_name: str = "FastAgent-MCP-Server",
        server_description: str | None = None,
        tool_description: str | None = None,
        host: str = "0.0.0.0",
        get_registry_version: Callable[[], int] | None = None,
        reload_callback: Callable[[], Awaitable[bool]] | None = None,
        tool_name_template: str | None = None,
    ) -> None:
        self._primary_instance = primary_instance
        self._instance_factory = CallableAgentInstanceFactory(
            create=create_instance,
            dispose=dispose_instance,
        )
        self._create_instance_task = self._instance_factory.create_instance
        self._dispose_instance_task = self._instance_factory.dispose_instance
        self._instance_scope = InstanceScope(instance_scope)
        self._default_host = host
        self._reload_callback = reload_callback
        self._tool_description = tool_description
        self._tool_name_template = tool_name_template or "{agent}"
        if "{agent}" not in self._tool_name_template:
            raise ValueError("tool_name_template must include '{agent}'.")

        oauth_provider, oauth_scopes, resource_url = _get_oauth_config()
        auth_provider = None
        if oauth_provider == "huggingface":
            from fast_agent.mcp.auth.presence import HuggingFaceTokenVerifier

            token_verifier = HuggingFaceTokenVerifier(
                provider="huggingface",
                scopes=oauth_scopes,
                base_url=resource_url,
            )
            auth_provider = RemoteAuthProvider(
                token_verifier=token_verifier,
                authorization_servers=[AnyHttpUrl("https://huggingface.co")],
                base_url=AnyHttpUrl(resource_url),
                scopes_supported=oauth_scopes,
                resource_name=server_name,
            )
            logger.info(
                f"OAuth enabled for provider '{oauth_provider}'",
                name="oauth_enabled",
                provider=oauth_provider,
                scopes=oauth_scopes,
                resource_url=resource_url,
            )

        self.mcp_server = FastMCP(
            name=server_name,
            instructions=self._build_instructions(server_description),
            version=_get_fast_agent_version(),
            auth=auth_provider,
        )

        @self.mcp_server.custom_route("/", methods=["GET"])
        async def root_info(request):
            del request
            from starlette.responses import PlainTextResponse

            version = _get_fast_agent_version() or "unknown"
            return PlainTextResponse(
                f"fast-agent mcp server (v{version}) - see https://fast-agent.ai for more information."
            )

        self._registered_agents: set[str] = set(primary_instance.agents.keys())
        self.std_logger = logging.getLogger("fast_agent.server")
        self._instance_pool = ScopedAgentInstancePool(
            primary_instance=primary_instance,
            instance_factory=self._instance_factory,
            instance_scope=instance_scope,
            get_registry_version=get_registry_version,
            register_missing_agents=self._register_missing_agents,
        )
        self._connection_instances = self._instance_pool.connection_instances
        self._stale_instances = self._instance_pool.stale_instances

        self.setup_tools()

        logger.info(
            f"AgentMCPServer initialized with {len(primary_instance.agents)} agents",
            name="mcp_server_initialized",
            agent_count=len(primary_instance.agents),
            instance_scope=self._instance_scope.value,
        )

    @property
    def primary_instance(self) -> AgentInstance:
        if "_instance_pool" not in self.__dict__:
            return self._primary_instance
        return self._instance_pool.primary_instance

    @primary_instance.setter
    def primary_instance(self, instance: AgentInstance) -> None:
        if "_instance_pool" in self.__dict__:
            self._instance_pool.primary_instance = instance
        self._primary_instance = instance

    def setup_tools(self) -> None:
        """Register all agents as MCP tools."""
        for agent_name in self.primary_instance.agents:
            self.register_agent_tools(agent_name)
        if self._reload_callback is not None:
            self._register_reload_tool()

    @staticmethod
    def _agent_tool_result_mode(agent: Any | None) -> ToolResultMode:
        request_params = _agent_metadata(agent).default_request_params
        if request_params is None:
            return "postprocess"
        return request_params.tool_result_mode

    def register_agent_tools(self, agent_name: str) -> None:
        """Register tools for a specific agent."""
        self._registered_agents.add(agent_name)
        registration = self._agent_tool_registration(agent_name)

        if registration.response_mode_enabled:

            @self.mcp_server.tool(
                name=registration.tool_name,
                description=registration.description,
                output_schema=None,
            )
            async def send_message(
                message: str,
                ctx: MCPContext,
                response_mode: Literal["inherit", "postprocess", "passthrough"] = "inherit",
            ) -> str:
                return await self._send_agent_message(agent_name, message, ctx, response_mode)

        else:

            @self.mcp_server.tool(
                name=registration.tool_name,
                description=registration.description,
                output_schema=None,
            )
            async def send_message(message: str, ctx: MCPContext) -> str:
                return await self._send_agent_message(agent_name, message, ctx)

        self._register_agent_history_prompt(agent_name)

    def _agent_tool_registration(self, agent_name: str) -> _AgentToolRegistration:
        agent = self.primary_instance.agents.get(agent_name)
        return _AgentToolRegistration(
            agent_name=agent_name,
            tool_name=self._tool_name_template.format(agent=agent_name),
            description=self._agent_tool_description(agent_name, agent),
            response_mode_enabled=tool_result_mode_allows_response_mode(
                self._agent_tool_result_mode(agent)
            ),
        )

    def _agent_tool_description(self, agent_name: str, agent: Any | None) -> str:
        tool_description = (
            self._tool_description.format(agent=agent_name)
            if self._tool_description and "{agent}" in self._tool_description
            else self._tool_description
        )
        metadata = _agent_metadata(agent)
        return (
            tool_description or metadata.description or f"Send a message to the {agent_name} agent"
        )

    async def _send_agent_message(
        self,
        agent_name: str,
        message: str,
        ctx: MCPContext,
        response_mode: ResponseMode | None = None,
    ) -> str:
        from fast_agent.mcp.auth.context import request_bearer_token

        saved_token = request_bearer_token.set(_get_request_bearer_token())
        request_params = self._request_params_for_tool_call(ctx, response_mode)
        try:
            instance = await self._acquire_instance(ctx)
            try:
                return await self._execute_agent_send(
                    agent_name,
                    instance.app[agent_name],
                    message,
                    request_params,
                    ctx,
                )
            finally:
                await self._release_instance(ctx, instance)
        finally:
            request_bearer_token.reset(saved_token)

    def _request_params_for_tool_call(
        self,
        ctx: MCPContext,
        response_mode: ResponseMode | None,
    ) -> RequestParams:
        request_param_overrides: dict[str, Any] = {
            "tool_execution_handler": MCPToolProgressManager(self._build_progress_reporter(ctx)),
            "emit_loop_progress": True,
        }
        if response_mode is not None:
            tool_result_mode = response_mode_to_tool_result_mode(response_mode)
            if tool_result_mode is not None:
                request_param_overrides["tool_result_mode"] = tool_result_mode
        return RequestParams(**request_param_overrides)

    async def _execute_agent_send(
        self,
        agent_name: str,
        agent_instance: Any,
        message: str,
        request_params: RequestParams,
        ctx: MCPContext,
    ) -> str:
        async def execute_send() -> str:
            start = time.perf_counter()
            self._log_agent_send_started(agent_name, ctx)
            response = await agent_instance.send(message, request_params=request_params)
            self._log_agent_send_completed(agent_name, ctx, time.perf_counter() - start)
            return response

        agent_context = getattr(agent_instance, "context", None)
        if agent_context is not None:
            return await self.with_bridged_context(agent_context, ctx, execute_send)
        return await execute_send()

    def _log_agent_send_started(self, agent_name: str, ctx: MCPContext) -> None:
        logger.info(
            f"MCP request received for agent '{agent_name}'",
            name="mcp_request_start",
            agent=agent_name,
            session=self._session_identifier(ctx),
        )
        self.std_logger.info(
            "MCP request received for agent '%s' (scope=%s)",
            agent_name,
            self._instance_scope.value,
        )

    def _log_agent_send_completed(
        self,
        agent_name: str,
        ctx: MCPContext,
        duration: float,
    ) -> None:
        logger.info(
            f"Agent '{agent_name}' completed MCP request",
            name="mcp_request_complete",
            agent=agent_name,
            duration=duration,
            session=self._session_identifier(ctx),
        )
        self.std_logger.info(
            "Agent '%s' completed MCP request in %.2fs (scope=%s)",
            agent_name,
            duration,
            self._instance_scope.value,
        )

    def _register_agent_history_prompt(self, agent_name: str) -> None:
        if self._instance_scope is InstanceScope.REQUEST:
            return

        @self.mcp_server.prompt(
            name=f"{agent_name}_history",
            description=f"Conversation history for the {agent_name} agent",
        )
        async def get_history_prompt(ctx: MCPContext) -> list[Message]:
            instance = await self._acquire_instance(ctx)
            agent_instance = instance.app[agent_name]
            try:
                multipart_history = agent_instance.message_history
                if not multipart_history:
                    return []

                return _history_to_fastmcp_messages(multipart_history)
            finally:
                await self._release_instance(ctx, instance, reuse_connection=True)

    def _register_missing_agents(self, instance: AgentInstance) -> None:
        new_agents = set(instance.agents.keys())
        for agent_name in sorted(new_agents - self._registered_agents):
            self.register_agent_tools(agent_name)

    def _register_reload_tool(self) -> None:
        @self.mcp_server.tool(
            name="reload_agent_cards",
            description="Reload AgentCards",
            output_schema=None,
        )
        async def reload_agent_cards(ctx: MCPContext) -> str:
            if not self._reload_callback:
                return "Reload not available."

            changed = await self._reload_callback()
            if not changed:
                return "No AgentCard changes detected."

            await self._instance_pool.reload_current_scope(ctx)
            return "Reloaded AgentCards."

    def _build_instructions(self, server_description: str | None) -> str:
        agent_count = len(self.primary_instance.agents)
        base = server_description or f"This server provides access to {agent_count} agents."
        scope_info = (
            "do NOT retain history between your requests"
            if self._instance_scope is InstanceScope.REQUEST
            else "retain history between tool calls."
        )
        return (
            f"{base} Use the `{self._name_for_send_tool()}` tools to send messages to agents. "
            f"Instance mode is {self._instance_scope.value}. Agents ({scope_info})"
        )

    def _name_for_send_tool(self) -> str:
        return self._tool_name_template.format(agent="<agent>")

    def _build_progress_reporter(
        self, ctx: MCPContext
    ) -> Callable[[float, float | None, str | None], Awaitable[None]]:
        async def report_progress(
            progress: float,
            total: float | None = None,
            message: str | None = None,
        ) -> None:
            with suppress(Exception):
                await ctx.report_progress(progress, total, message)

        return report_progress

    async def _acquire_instance(self, ctx: MCPContext | None) -> AgentInstance:
        return (await self._instance_pool.acquire(ctx)).instance

    async def _release_instance(
        self,
        ctx: MCPContext | None,
        instance: AgentInstance,
        *,
        reuse_connection: bool = False,
    ) -> None:
        await self._instance_pool.release(
            AgentInstanceLease(instance, reuse_connection=reuse_connection),
            ctx,
        )

    def _connection_key(self, ctx: MCPContext) -> int:
        return self._instance_pool.connection_key(ctx)

    def _register_session_cleanup(self, ctx: MCPContext, session_key: int) -> None:
        self._instance_pool.register_session_cleanup(ctx, session_key)

    def _session_identifier(self, ctx: MCPContext | None) -> str | None:
        if ctx is None or ctx.request_context is None:
            return None
        request = getattr(ctx.request_context, "request", None)
        if request is None:
            return None
        headers = getattr(request, "headers", None)
        return headers.get("mcp-session-id") if headers is not None else None

    async def _maybe_refresh_shared_instance(self) -> None:
        await self._instance_pool.maybe_refresh_shared_instance()

    async def _dispose_stale_instances_if_idle(self) -> None:
        await self._instance_pool.dispose_stale_instances_if_idle()

    async def _dispose_primary_instance(self) -> None:
        await self._instance_pool.dispose_primary_instance()

    async def _dispose_all_stale_instances(self) -> None:
        await self._instance_pool.dispose_all_stale_instances()

    async def _dispose_all_connection_instances(self) -> None:
        await self._instance_pool.dispose_all_connection_instances()

    async def _dispose_instance_safely(self, instance: AgentInstance, *, phase: str) -> None:
        await self._instance_pool.dispose_instance_safely(instance, phase=phase)

    def _http_middleware(self) -> list[Middleware] | None:
        oauth_provider = _normalize_serve_oauth_provider(os.environ.get("FAST_AGENT_SERVE_OAUTH"))
        if oauth_provider != "huggingface":
            return None
        return [Middleware(cast("Any", HFAuthHeaderMiddleware))]

    def http_app(self):
        """Return a FastMCP HTTP ASGI app configured for the current instance scope."""
        return self.mcp_server.http_app(
            transport="http",
            middleware=self._http_middleware(),
            stateless_http=self._instance_scope is InstanceScope.REQUEST,
        )

    def run(
        self,
        transport: TransportMode = "http",
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        """Run the MCP server synchronously."""
        try:
            if transport == "http":
                self.mcp_server.run(
                    transport="http",
                    host=host,
                    port=port,
                    middleware=self._http_middleware(),
                    stateless_http=self._instance_scope is InstanceScope.REQUEST,
                )
                return
            if transport == "stdio":
                self.mcp_server.run(transport="stdio")
                return
            raise ValueError(f"Unsupported MCP server transport: {transport}")
        except KeyboardInterrupt:
            print("\nServer stopped by user (CTRL+C)")
        finally:
            run_sync(self.shutdown)

    async def run_async(
        self,
        transport: TransportMode = "http",
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        """Run the MCP server asynchronously."""
        try:
            if transport == "http":
                await self.mcp_server.run_http_async(
                    transport="http",
                    host=host,
                    port=port,
                    middleware=self._http_middleware(),
                    stateless_http=self._instance_scope is InstanceScope.REQUEST,
                )
                return
            if transport == "stdio":
                await self.mcp_server.run_stdio_async()
                return
            raise ValueError(f"Unsupported MCP server transport: {transport}")
        finally:
            await self.shutdown()

    async def with_bridged_context(
        self,
        agent_context: Any,
        mcp_context: MCPContext,
        func: Callable[..., Awaitable[str]],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Execute a function with bridged context between MCP and agent."""
        original_progress_reporter = getattr(
            agent_context,
            "progress_reporter",
            _CONTEXT_ATTR_UNSET,
        )
        original_mcp_context = getattr(agent_context, "mcp_context", _CONTEXT_ATTR_UNSET)
        agent_context.mcp_context = mcp_context

        async def bridged_progress(
            progress: float,
            total: float | None = None,
            message: str | None = None,
        ) -> None:
            await mcp_context.report_progress(progress, total, message)
            if (
                original_progress_reporter is _CONTEXT_ATTR_UNSET
                or original_progress_reporter is None
            ):
                return
            progress_reporter = cast(
                "Callable[[float, float | None, str | None], Awaitable[None]]",
                original_progress_reporter,
            )
            try:
                await progress_reporter(progress, total, message)
            except TypeError:
                legacy_progress_reporter = cast(
                    "Callable[[float, float | None], Awaitable[None]]",
                    original_progress_reporter,
                )
                await legacy_progress_reporter(progress, total)

        if original_progress_reporter is not _CONTEXT_ATTR_UNSET:
            agent_context.progress_reporter = bridged_progress

        try:
            return await func(*args, **kwargs)
        finally:
            _restore_context_attribute(
                agent_context,
                "progress_reporter",
                original_progress_reporter,
            )
            _restore_context_attribute(agent_context, "mcp_context", original_mcp_context)

    async def shutdown(self) -> None:
        """Dispose all managed agent instances."""
        await self._instance_pool.shutdown()
