from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance
from fast_agent.mcp.auth.context import request_bearer_token
from fast_agent.mcp.server.agent_server import AgentMCPServer

if TYPE_CHECKING:
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
@pytest.mark.asyncio
async def test_send_tool_passes_authenticated_bearer_token_via_contextvar() -> None:
    agent = _AuthCapturingAgent()
    server = await _build_server(agent)
    tool = server.mcp_server._tool_manager._tools["worker"]
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
    tool = server.mcp_server._tool_manager._tools["worker"]
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
