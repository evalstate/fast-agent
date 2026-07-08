"""Reusable FastMCP adapter for harness applications."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from fastmcp import Context as MCPContext  # noqa: TC002 - FastMCP inspects tool annotations.
from fastmcp.server.dependencies import get_access_token, get_context
from fastmcp.tools import Tool
from mcp.types import TextContent
from pydantic import PrivateAttr

from fast_agent.core.agent_tool_shape import (
    AgentToolArgumentRenderer,
    default_agent_tool_schema,
    render_agent_tool_arguments,
    render_structured_args,
)
from fast_agent.core.harness_app import AppOpenRequest, HarnessApp
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.tool_progress import MCPToolProgressManager
from fast_agent.types import AgentAuth, AgentRequest, AgentResponse, PromptMessageExtended

if TYPE_CHECKING:
    from fastmcp import FastMCP

MCPHarnessSessionScope = Literal["connection", "request"]
SessionCleanup = Callable[[str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class HarnessMCPAdapterOptions:
    """Options for adapting FastMCP tool calls to a ``HarnessApp``."""

    default_agent: str | None = None
    session_scope: MCPHarnessSessionScope = "connection"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tool_name: str = "send"
    tool_description: str | None = None
    cleanup_session: SessionCleanup | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class HarnessMCPSessionPlan:
    """Resolved session ids for one MCP tool call."""

    scope: MCPHarnessSessionScope
    open_session_id: str | None
    request_session_id: str | None
    mcp_session_id: str | None = None
    requested_session_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class MCPProgressReporter:
    """Bridge harness app progress reports to MCP progress notifications."""

    def __init__(self, ctx: MCPContext) -> None:
        self._ctx = ctx

    async def report(
        self,
        message: str,
        *,
        progress: float | None = None,
        total: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        del metadata
        await self._ctx.report_progress(progress or 0, total, message)


class HarnessMCPCallContext:
    """One FastMCP request context adapted for harness invocation."""

    def __init__(
        self,
        adapter: HarnessMCPAdapter,
        ctx: MCPContext,
        *,
        session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._adapter = adapter
        self._ctx = ctx
        self._mcp_session_id = adapter.mcp_session_id(ctx)
        self._auth = adapter.agent_auth()
        self._request_params = adapter.request_params(ctx)
        self._session_plan = adapter.session_plan(
            mcp_session_id=self._mcp_session_id,
            requested_session_id=session_id,
            metadata=metadata,
        )

    @property
    def mcp_session_id(self) -> str | None:
        return self._mcp_session_id

    @property
    def auth(self) -> AgentAuth | None:
        return self._auth

    @property
    def request_params(self) -> RequestParams:
        return self._request_params

    @property
    def session_plan(self) -> HarnessMCPSessionPlan:
        return self._session_plan

    async def invoke_agent(
        self,
        *,
        agent: str | None = None,
        message: str | PromptMessageExtended | None = None,
        arguments: Mapping[str, Any] | None = None,
        render_arguments: AgentToolArgumentRenderer | None = None,
        input_schema: dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentResponse:
        return await self._adapter.invoke_with_context(
            self,
            agent=agent,
            message=message,
            arguments=arguments,
            render_arguments=render_arguments,
            input_schema=input_schema,
            metadata=metadata,
        )


class HarnessMCPAdapter:
    """Reusable bridge from FastMCP request context to ``HarnessApp``."""

    def __init__(self, app: HarnessApp, options: HarnessMCPAdapterOptions) -> None:
        self._app = app
        self._options = options

    @property
    def options(self) -> HarnessMCPAdapterOptions:
        return self._options

    def call_context(
        self,
        ctx: MCPContext,
        *,
        session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> HarnessMCPCallContext:
        return HarnessMCPCallContext(self, ctx, session_id=session_id, metadata=metadata)

    async def invoke_agent(
        self,
        *,
        ctx: MCPContext,
        agent: str | None = None,
        message: str | PromptMessageExtended | None = None,
        arguments: Mapping[str, Any] | None = None,
        session_id: str | None = None,
        render_arguments: AgentToolArgumentRenderer | None = None,
        input_schema: dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentResponse:
        call = self.call_context(ctx, session_id=session_id, metadata=metadata)
        return await self.invoke_with_context(
            call,
            agent=agent,
            message=message,
            arguments=arguments,
            render_arguments=render_arguments,
            input_schema=input_schema,
        )

    async def invoke_with_context(
        self,
        call: HarnessMCPCallContext,
        *,
        agent: str | None = None,
        message: str | PromptMessageExtended | None = None,
        arguments: Mapping[str, Any] | None = None,
        render_arguments: AgentToolArgumentRenderer | None = None,
        input_schema: dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentResponse:
        request = self.agent_request(
            call,
            agent=agent,
            message=message,
            arguments=arguments,
            render_arguments=render_arguments,
            input_schema=input_schema,
            metadata=metadata,
        )
        session_plan = call.session_plan
        with self._request_bearer_context(call.auth):
            try:
                async with self._app.open(
                    AppOpenRequest(
                        session_id=session_plan.open_session_id,
                        agent=request.agent,
                        metadata=session_plan.metadata,
                    )
                ) as session:
                    return await session.invoke(request)
            finally:
                cleanup_session = self._options.cleanup_session
                if (
                    cleanup_session is not None
                    and session_plan.scope == "request"
                    and session_plan.open_session_id is not None
                ):
                    await cleanup_session(session_plan.open_session_id)

    def agent_request(
        self,
        call: HarnessMCPCallContext,
        *,
        agent: str | None = None,
        message: str | PromptMessageExtended | None = None,
        arguments: Mapping[str, Any] | None = None,
        render_arguments: AgentToolArgumentRenderer | None = None,
        input_schema: dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentRequest:
        if (message is None) == (arguments is None):
            raise ValueError("Exactly one of message or arguments is required.")

        request_message = (
            self._message_from_arguments(
                arguments,
                render_arguments=render_arguments,
                input_schema=input_schema,
            )
            if arguments is not None
            else self._message_from_prompt(message)
        )
        request_metadata = {
            **dict(self._options.metadata),
            **dict(call.session_plan.metadata),
            **dict(metadata or {}),
        }
        request_state: dict[str, Any] = {}
        if arguments is not None:
            request_state["mcp_arguments"] = dict(arguments)

        return AgentRequest(
            message=request_message,
            agent=agent or self._options.default_agent,
            session_id=call.session_plan.request_session_id,
            auth=call.auth,
            params=call.request_params,
            metadata=request_metadata,
            state=request_state,
            progress=MCPProgressReporter(call._ctx),
        )

    def session_plan(
        self,
        *,
        mcp_session_id: str | None,
        requested_session_id: str | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> HarnessMCPSessionPlan:
        base_metadata = {
            "transport": "mcp",
            "session_scope": self._options.session_scope,
        }
        if mcp_session_id is not None:
            base_metadata["mcp_session_id"] = mcp_session_id
        if requested_session_id is not None:
            base_metadata["requested_session_id"] = requested_session_id

        if self._options.session_scope == "request":
            transient_id = f"request-{uuid4().hex}"
            return HarnessMCPSessionPlan(
                scope="request",
                open_session_id=transient_id,
                request_session_id=transient_id,
                mcp_session_id=mcp_session_id,
                requested_session_id=requested_session_id,
                metadata={
                    **base_metadata,
                    "harness_session_id": transient_id,
                    **dict(metadata or {}),
                },
            )

        effective_session_id = requested_session_id or mcp_session_id
        return HarnessMCPSessionPlan(
            scope="connection",
            open_session_id=effective_session_id,
            request_session_id=effective_session_id,
            mcp_session_id=mcp_session_id,
            requested_session_id=requested_session_id,
            metadata={
                **base_metadata,
                **({"harness_session_id": effective_session_id} if effective_session_id else {}),
                **dict(metadata or {}),
            },
        )

    def request_params(self, ctx: MCPContext) -> RequestParams:
        return RequestParams(
            tool_execution_handler=MCPToolProgressManager(self._build_progress_reporter(ctx)),
            emit_loop_progress=True,
        )

    @staticmethod
    def agent_auth() -> AgentAuth | None:
        access_token = get_access_token()
        if access_token is None:
            token = HarnessMCPAdapter._raw_bearer_token_from_request()
            if token:
                return AgentAuth.bearer(token, provider="huggingface")
            return None
        claims = dict(access_token.claims)
        provider = claims.get("provider")
        if not isinstance(provider, str):
            provider = "huggingface" if "huggingface_userinfo" in claims else None
        subject = access_token.subject
        if subject is None:
            claim_subject = claims.get("sub")
            subject = claim_subject if isinstance(claim_subject, str) else None
        return AgentAuth.bearer(
            access_token.token,
            provider=provider,
            subject=subject,
            client_id=access_token.client_id,
            scopes=tuple(access_token.scopes),
            claims=claims,
        )

    @staticmethod
    def _raw_bearer_token_from_request() -> str | None:
        try:
            ctx = get_context()
        except Exception:
            return None
        request_context = getattr(ctx, "request_context", None)
        request = getattr(request_context, "request", None)
        headers = getattr(request, "headers", None)
        if headers is None:
            return None
        for header_name in ("authorization", "x-hf-authorization"):
            value = headers.get(header_name)
            if isinstance(value, str) and value.lower().startswith("bearer "):
                token = value[7:].strip()
                return token or None
        return None

    @staticmethod
    def mcp_session_id(ctx: MCPContext) -> str | None:
        if ctx.request_context is None:
            return None
        request = getattr(ctx.request_context, "request", None)
        if request is None:
            return None
        headers = getattr(request, "headers", None)
        if headers is None:
            return None
        session_id = headers.get("mcp-session-id")
        return session_id if isinstance(session_id, str) else None

    def register_default_tools(self, mcp: FastMCP) -> None:
        @mcp.tool(
            name=self._options.tool_name,
            description=self.tool_description(),
            output_schema=None,
        )
        async def send(
            message: str,
            ctx: MCPContext,
            session_id: str | None = None,
            agent: str | None = None,
        ) -> str:
            response = await self.invoke_agent(
                ctx=ctx,
                agent=agent,
                message=message,
                session_id=session_id,
            )
            return response.text_content()

    def register_agent_tool(
        self,
        mcp: FastMCP,
        *,
        name: str,
        agent: str | None = None,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        session_id: str | None = None,
        render_arguments: AgentToolArgumentRenderer | None = None,
    ) -> Tool:
        return mcp.add_tool(
            _HarnessAgentTool(
                adapter=self,
                name=name,
                agent=agent,
                description=description,
                input_schema=input_schema,
                session_id=session_id,
                render_arguments=render_arguments,
            )
        )

    def tool_description(self) -> str:
        description = self._options.tool_description or "Send a message to the fast-agent application."
        return description.replace("{agent}", self._options.default_agent or "agent")

    @staticmethod
    def _build_progress_reporter(ctx: MCPContext):
        async def report_progress(
            progress: float,
            total: float | None = None,
            message: str | None = None,
        ) -> None:
            await ctx.report_progress(progress, total, message)

        return report_progress

    @staticmethod
    def _message_from_prompt(message: str | PromptMessageExtended | None) -> PromptMessageExtended:
        if isinstance(message, PromptMessageExtended):
            return message
        if isinstance(message, str):
            return PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text=message)],
            )
        raise ValueError("message is required.")

    @staticmethod
    def _message_from_arguments(
        arguments: Mapping[str, Any] | None,
        *,
        render_arguments: AgentToolArgumentRenderer | None = None,
        input_schema: dict[str, Any] | None = None,
    ) -> PromptMessageExtended:
        if arguments is None:
            raise ValueError("arguments are required.")
        rendered = (
            render_structured_args(arguments)
            if input_schema is None and render_arguments is None
            else render_agent_tool_arguments(
                arguments,
                configured_schema=input_schema,
                render_arguments=render_arguments,
            )
        )
        return PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text=rendered)],
        )

    @staticmethod
    @contextmanager
    def _request_bearer_context(auth: AgentAuth | None):
        if auth is None:
            yield
            return

        from fast_agent.mcp.auth.context import request_bearer_token

        token = request_bearer_token.set(auth.token)
        try:
            yield
        finally:
            request_bearer_token.reset(token)


class _HarnessAgentTool(Tool):
    _adapter: HarnessMCPAdapter = PrivateAttr()
    _agent: str | None = PrivateAttr(default=None)
    _session_id: str | None = PrivateAttr(default=None)
    _input_schema: dict[str, Any] | None = PrivateAttr(default=None)
    _render_arguments: AgentToolArgumentRenderer | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        adapter: HarnessMCPAdapter,
        name: str,
        agent: str | None,
        description: str | None,
        input_schema: dict[str, Any] | None,
        session_id: str | None,
        render_arguments: AgentToolArgumentRenderer | None,
    ) -> None:
        resolved_input_schema = input_schema or default_agent_tool_schema()
        super().__init__(
            name=name,
            description=description,
            parameters=resolved_input_schema,
            output_schema=None,
        )
        self._adapter = adapter
        self._agent = agent
        self._session_id = session_id
        self._input_schema = resolved_input_schema
        self._render_arguments = render_arguments

    async def run(self, arguments: dict[str, Any]):
        response = await self._adapter.invoke_agent(
            ctx=get_context(),
            agent=self._agent,
            arguments=arguments,
            session_id=self._session_id,
            input_schema=self._input_schema,
            render_arguments=self._render_arguments,
        )
        return self.convert_result(response.text_content())


__all__ = [
    "HarnessMCPAdapter",
    "HarnessMCPAdapterOptions",
    "HarnessMCPCallContext",
    "HarnessMCPSessionPlan",
    "MCPHarnessSessionScope",
    "MCPProgressReporter",
]
