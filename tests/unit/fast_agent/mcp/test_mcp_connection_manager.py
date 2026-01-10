
import asyncio

import pytest

from fast_agent.config import MCPServerSettings
from fast_agent.mcp.mcp_connection_manager import (
    ServerConnection,
    _prepare_headers_and_auth,
    _server_lifecycle_task,
)


def test_prepare_headers_respects_user_authorization(monkeypatch):
    config = MCPServerSettings(
        name="test",
        transport="sse",
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer user-token"},
    )

    def _builder(_config):
        raise AssertionError("OAuth provider should not be built when Authorization header is set.")

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.build_oauth_provider",
        _builder,
    )

    headers, auth, user_keys = _prepare_headers_and_auth(config)

    assert headers == {"Authorization": "Bearer user-token"}
    assert headers is not config.headers
    assert auth is None
    assert user_keys == {"Authorization"}


def test_prepare_headers_respects_case_insensitive_authorization(monkeypatch):
    config = MCPServerSettings(
        name="test",
        transport="http",
        url="https://example.com/mcp",
        headers={"authorization": "Bearer user-token"},
    )

    def _builder(_config):
        raise AssertionError("OAuth provider should not be built when authorization header is set.")

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.build_oauth_provider",
        _builder,
    )

    headers, auth, user_keys = _prepare_headers_and_auth(config)

    assert headers == {"authorization": "Bearer user-token"}
    assert auth is None
    assert user_keys == {"authorization"}


def test_prepare_headers_invokes_oauth_when_no_auth_headers(monkeypatch):
    config = MCPServerSettings(
        name="test",
        transport="sse",
        url="https://example.com/mcp",
        headers={"Accept": "application/json"},
    )

    sentinel = object()
    calls: list[MCPServerSettings] = []

    def _builder(received_config: MCPServerSettings):
        calls.append(received_config)
        return sentinel

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.build_oauth_provider",
        _builder,
    )

    headers, auth, user_keys = _prepare_headers_and_auth(config)

    assert headers == {"Accept": "application/json"}
    assert auth is sentinel
    assert user_keys == set()
    assert calls == [config]


@pytest.mark.asyncio
async def test_server_lifecycle_sets_initialized_on_startup_failure():
    class DummyTransportContext:
        async def __aenter__(self):
            return object(), object(), None

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def initialize(self):
            raise RuntimeError("boom")

    def session_factory(*_args, **_kwargs):
        return DummySession()

    server_conn = ServerConnection(
        server_name="test-server",
        server_config=MCPServerSettings(name="test-server", url="http://example.com/mcp"),
        transport_context_factory=DummyTransportContext,
        client_session_factory=session_factory,
    )

    lifecycle_task = asyncio.create_task(_server_lifecycle_task(server_conn))
    try:
        await asyncio.wait_for(server_conn.wait_for_initialized(), timeout=1.0)
    finally:
        await lifecycle_task

    assert server_conn._error_occurred is True
