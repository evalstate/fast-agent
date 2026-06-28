"""MCP adapter for harness applications."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from fastmcp import FastMCP
from fastmcp.server.auth import RemoteAuthProvider
from pydantic import AnyHttpUrl
from starlette.middleware import Middleware

from fast_agent.core.harness import HarnessSessions
from fast_agent.core.harness_app import (
    HarnessApp,
    HarnessSessionsAppProvider,
    load_harness_app,
)
from fast_agent.core.harness_persistence import FileHarnessSessionPersistence
from fast_agent.mcp.auth.middleware import HFAuthHeaderMiddleware
from fast_agent.mcp.server.common import (
    TransportMode,
    get_fast_agent_version,
    get_oauth_config,
    normalize_serve_oauth_provider,
)
from fast_agent.mcp.server.harness_adapter import (
    HarnessMCPAdapter,
    HarnessMCPAdapterOptions,
    MCPHarnessSessionScope,
    MCPProgressReporter,
    SessionCleanup,
)

if TYPE_CHECKING:
    from fast_agent.config import Settings
    from fast_agent.core.agent_instance_factory import AgentInstanceFactory
    from fast_agent.mcp.server.instance_lease_pool import InstanceScopeValue
    from fast_agent.tools.session_environment import ShellExecutor


@dataclass(frozen=True, slots=True)
class HarnessMCPAppServerOptions:
    """Configuration for the default harness MCP application server."""

    server_name: str
    server_description: str | None = None
    tool_name: str = "send"
    tool_description: str | None = None
    default_agent: str | None = None
    stateless_http: bool = False
    session_scope: MCPHarnessSessionScope = "connection"
    cleanup_session: SessionCleanup | None = None


@dataclass(frozen=True, slots=True)
class HarnessMCPAppRuntimeOptions:
    """Options for running the default harness MCP app runtime."""

    server_name: str
    server_description: str | None = None
    tool_name: str | None = None
    tool_description: str | None = None
    default_agent: str | None = None
    transport: TransportMode = "http"
    host: str = "127.0.0.1"
    port: int = 8000
    instance_scope: InstanceScopeValue = "shared"


@dataclass(slots=True)
class HarnessMCPAppRuntime:
    """Owned MCP app server plus its harness session manager."""

    server: HarnessMCPAppServer
    sessions: HarnessSessions

    async def close(self) -> None:
        await self.sessions.close_all()


class HarnessMCPAppServer:
    """Expose a ``HarnessApp`` as a single MCP application tool."""

    def __init__(self, app: HarnessApp, options: HarnessMCPAppServerOptions) -> None:
        self._app = app
        self._options = options
        self.adapter = HarnessMCPAdapter(
            app,
            HarnessMCPAdapterOptions(
                default_agent=options.default_agent,
                session_scope=options.session_scope,
                tool_name=options.tool_name,
                tool_description=options.tool_description,
                cleanup_session=options.cleanup_session,
            ),
        )
        self.mcp_server = FastMCP(
            name=options.server_name,
            instructions=self._instructions(),
            version=get_fast_agent_version(),
            auth=self._auth_provider(),
        )
        self._register_routes()
        self.adapter.register_default_tools(self.mcp_server)

    def _auth_provider(self) -> RemoteAuthProvider | None:
        oauth_provider, oauth_scopes, resource_url = get_oauth_config()
        if oauth_provider != "huggingface":
            return None

        from fast_agent.mcp.auth.providers.huggingface import HuggingFaceTokenVerifier

        token_verifier = HuggingFaceTokenVerifier()
        return RemoteAuthProvider(
            token_verifier=token_verifier,
            authorization_servers=[AnyHttpUrl("https://huggingface.co")],
            base_url=AnyHttpUrl(resource_url),
            scopes_supported=oauth_scopes,
            resource_name=self._options.server_name,
        )

    def _register_routes(self) -> None:
        @self.mcp_server.custom_route("/", methods=["GET"])
        async def root_info(request):
            del request
            from starlette.responses import PlainTextResponse

            version = get_fast_agent_version() or "unknown"
            return PlainTextResponse(
                f"fast-agent harness mcp server (v{version}) - see https://fast-agent.ai for more information."
            )

    async def _send(
        self,
        message: str,
        *,
        ctx,
        session_id: str | None,
        agent: str | None,
    ) -> str:
        response = await self.adapter.invoke_agent(
            ctx=ctx,
            agent=agent,
            message=message,
            session_id=session_id,
        )
        return response.text_content()

    def _tool_description(self) -> str:
        description = (
            self._options.tool_description or "Send a message to the fast-agent application."
        )
        return description.replace("{agent}", self._options.default_agent or "agent")

    def _instructions(self) -> str:
        description = self._options.server_description or "This server exposes a fast-agent app."
        return f"{description} Use `{self._options.tool_name}` to send messages."

    def _http_middleware(self) -> list[Middleware] | None:
        oauth_provider = normalize_serve_oauth_provider(os.environ.get("FAST_AGENT_SERVE_OAUTH"))
        if oauth_provider != "huggingface":
            return None
        return [Middleware(cast("Any", HFAuthHeaderMiddleware))]

    async def run_async(
        self,
        transport: TransportMode = "http",
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        if transport == "http":
            await self.mcp_server.run_http_async(
                transport="http",
                host=host,
                port=port,
                middleware=self._http_middleware(),
                stateless_http=self._options.stateless_http,
            )
            return
        if transport == "stdio":
            await self.mcp_server.run_stdio_async()
            return
        raise ValueError(f"Unsupported MCP server transport: {transport}")


def create_harness_mcp_app_runtime(
    *,
    instance_factory: AgentInstanceFactory,
    shell_executor: ShellExecutor,
    settings: Settings | None,
    options: HarnessMCPAppRuntimeOptions,
) -> HarnessMCPAppRuntime:
    """Create the default harness MCP app runtime without starting transport."""
    persistence = (
        FileHarnessSessionPersistence(settings.environment_dir)
        if settings is not None and not settings._fast_agent_noenv and settings.session_history
        else None
    )
    sessions = HarnessSessions(
        instance_factory=instance_factory,
        persistence=persistence,
        shell_executor=shell_executor,
    )
    session_provider = HarnessSessionsAppProvider(sessions)
    app = load_harness_app(
        session_provider=session_provider,
        settings=settings,
    )
    server = HarnessMCPAppServer(
        app,
        HarnessMCPAppServerOptions(
            server_name=options.server_name,
            server_description=options.server_description,
            tool_name=options.tool_name or "send",
            tool_description=options.tool_description,
            default_agent=options.default_agent,
            stateless_http=options.instance_scope == "request",
            session_scope="request" if options.instance_scope == "request" else "connection",
            cleanup_session=sessions.delete if options.instance_scope == "request" else None,
        ),
    )
    return HarnessMCPAppRuntime(server=server, sessions=sessions)


async def run_harness_mcp_app_server(
    *,
    instance_factory: AgentInstanceFactory,
    shell_executor: ShellExecutor,
    settings: Settings | None,
    options: HarnessMCPAppRuntimeOptions,
) -> None:
    """Run the default harness MCP app server and close sessions on exit."""
    runtime = create_harness_mcp_app_runtime(
        instance_factory=instance_factory,
        shell_executor=shell_executor,
        settings=settings,
        options=options,
    )
    try:
        await runtime.server.run_async(
            transport=options.transport,
            host=options.host,
            port=options.port,
        )
    finally:
        await runtime.close()


__all__ = [
    "HarnessMCPAppRuntime",
    "HarnessMCPAppRuntimeOptions",
    "HarnessMCPAppServer",
    "HarnessMCPAppServerOptions",
    "MCPProgressReporter",
    "create_harness_mcp_app_runtime",
    "run_harness_mcp_app_server",
]
