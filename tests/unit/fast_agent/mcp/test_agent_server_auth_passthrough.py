from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance
from fast_agent.mcp.auth.context import request_bearer_token
from fast_agent.mcp.server.agent_server import (
    AgentMCPServer,
    _get_oauth_config,
    _normalize_serve_oauth_provider,
)

if TYPE_CHECKING:
    from fastmcp import Context as MCPContext
    from fastmcp.tools import FunctionTool

    from fast_agent.interfaces import AgentProtocol


class _AuthCapturingAgent:
    def __init__(self) -> None:
        self.config = SimpleNamespace(default_request_params=None, description=None)
        self.captured_tokens: list[str | None] = []

    async def send(self, message: str, request_params=None) -> str:
        del message, request_params
        self.captured_tokens.append(request_bearer_token.get())
        return "ok"

    async def shutdown(self) -> None:
        return None


class _NoopNotificationSession:
    async def send_notification(self, *_args, **_kwargs) -> None:
        return None


class _ProgressContext:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float | None, str | None]] = []

    async def report_progress(
        self,
        progress: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        self.calls.append((progress, total, message))


def _build_test_context() -> object:
    request_context = SimpleNamespace(
        meta=None,
        request=SimpleNamespace(headers={}),
        request_id="req-1",
        session=_NoopNotificationSession(),
    )
    return SimpleNamespace(session=object(), request_context=request_context)


async def _build_server(agent: _AuthCapturingAgent) -> AgentMCPServer:
    async def create_instance() -> AgentInstance:
        wrapped = cast("AgentProtocol", agent)
        app = AgentApp({"worker": wrapped})
        return AgentInstance(app=app, agents={"worker": wrapped})

    async def dispose_instance(instance: AgentInstance) -> None:
        await instance.shutdown()

    primary = await create_instance()
    return AgentMCPServer(
        primary_instance=primary,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        instance_scope="shared",
    )


@pytest.mark.unit
def test_serve_oauth_provider_normalizes_huggingface_aliases() -> None:
    assert _normalize_serve_oauth_provider(" HF ") == "huggingface"
    assert _normalize_serve_oauth_provider(" HuggingFace ") == "huggingface"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_agent_mcp_server_defaults_to_loopback_host() -> None:
    agent = _AuthCapturingAgent()
    server = await _build_server(agent)

    assert server._default_host == "127.0.0.1"
    await server.shutdown()


@pytest.mark.unit
def test_serve_oauth_provider_treats_blank_as_disabled() -> None:
    assert _normalize_serve_oauth_provider("   ") is None
    assert _normalize_serve_oauth_provider(None) is None


@pytest.mark.unit
def test_get_oauth_config_normalizes_provider_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAST_AGENT_SERVE_OAUTH", " HF ")

    oauth_provider, _scopes, _resource_url = _get_oauth_config()

    assert oauth_provider == "huggingface"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_middleware_uses_normalized_oauth_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAST_AGENT_SERVE_OAUTH", " HuggingFace ")
    server = await _build_server(_AuthCapturingAgent())

    middleware = server._http_middleware()

    assert middleware is not None
    assert len(middleware) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_send_tool_passes_authenticated_bearer_token_via_contextvar() -> None:
    agent = _AuthCapturingAgent()
    server = await _build_server(agent)
    tool = cast("FunctionTool", await server.mcp_server.get_tool("worker"))
    ctx = _build_test_context()
    authenticated_user = AuthenticatedUser(
        AccessToken(token="request-token", client_id="client-id", scopes=["access"])
    )

    saved_auth_context = auth_context_var.set(authenticated_user)
    try:
        assert request_bearer_token.get() is None
        response = await tool.fn(message="hello", ctx=ctx)
    finally:
        auth_context_var.reset(saved_auth_context)

    assert response == "ok"
    assert agent.captured_tokens == ["request-token"]
    assert request_bearer_token.get() is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_send_tool_restores_prior_request_token_when_no_authenticated_user_exists() -> None:
    agent = _AuthCapturingAgent()
    server = await _build_server(agent)
    tool = cast("FunctionTool", await server.mcp_server.get_tool("worker"))
    ctx = _build_test_context()

    saved_request_token = request_bearer_token.set("stale-token")
    try:
        response = await tool.fn(message="hello", ctx=ctx)
        assert response == "ok"
        assert agent.captured_tokens == [None]
        assert request_bearer_token.get() == "stale-token"
    finally:
        request_bearer_token.reset(saved_request_token)

    assert request_bearer_token.get() is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bridged_context_restores_existing_agent_context_attributes() -> None:
    agent = _AuthCapturingAgent()
    server = await _build_server(agent)
    mcp_context = _ProgressContext()
    original_mcp_context = object()
    original_progress_calls: list[tuple[float, float | None, str | None]] = []

    async def original_progress_reporter(
        progress: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        original_progress_calls.append((progress, total, message))

    agent_context = SimpleNamespace(
        mcp_context=original_mcp_context,
        progress_reporter=original_progress_reporter,
    )

    async def run_with_context() -> str:
        assert agent_context.mcp_context is mcp_context
        await agent_context.progress_reporter(0.5, 1.0, "halfway")
        return "ok"

    response = await server.with_bridged_context(
        agent_context,
        cast("MCPContext", mcp_context),
        run_with_context,
    )

    assert response == "ok"
    assert agent_context.mcp_context is original_mcp_context
    assert agent_context.progress_reporter is original_progress_reporter
    assert mcp_context.calls == [(0.5, 1.0, "halfway")]
    assert original_progress_calls == [(0.5, 1.0, "halfway")]
