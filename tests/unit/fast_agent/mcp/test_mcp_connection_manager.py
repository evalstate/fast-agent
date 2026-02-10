
import asyncio
import time
from typing import Any, cast

import pytest

from fast_agent.config import MCPServerSettings
from fast_agent.mcp.mcp_connection_manager import (
    MCPConnectionManager,
    ServerConnection,
    _is_oauth_timeout_message,
    _prepare_headers_and_auth,
    _server_lifecycle_task,
    _wait_for_initialized_with_startup_budget,
)


def test_prepare_headers_respects_user_authorization(monkeypatch):
    config = MCPServerSettings(
        name="test",
        transport="sse",
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer user-token"},
    )

    def _builder(_config, **_kwargs):
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

    def _builder(_config, **_kwargs):
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

    def _builder(received_config: MCPServerSettings, **_kwargs):
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


def _make_server_connection() -> ServerConnection:
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
            return None

    def session_factory(*_args, **_kwargs):
        return DummySession()

    return ServerConnection(
        server_name="test-server",
        server_config=MCPServerSettings(name="test-server", url="http://example.com/mcp"),
        transport_context_factory=DummyTransportContext,
        client_session_factory=session_factory,
    )


@pytest.mark.asyncio
async def test_startup_timeout_budget_excludes_oauth_wait_window() -> None:
    server_conn = _make_server_connection()

    async def _drive_events() -> None:
        await asyncio.sleep(0.02)
        server_conn.mark_oauth_wait_start()
        await asyncio.sleep(0.14)
        server_conn.mark_oauth_wait_end()
        await asyncio.sleep(0.06)
        server_conn._initialized_event.set()

    driver = asyncio.create_task(_drive_events())
    await _wait_for_initialized_with_startup_budget(
        server_conn,
        startup_timeout_seconds=0.1,
        poll_interval_seconds=0.01,
    )
    await driver


@pytest.mark.asyncio
async def test_startup_timeout_budget_still_times_out_for_non_oauth_hang() -> None:
    server_conn = _make_server_connection()

    with pytest.raises(TimeoutError):
        await _wait_for_initialized_with_startup_budget(
            server_conn,
            startup_timeout_seconds=0.05,
            poll_interval_seconds=0.01,
        )


@pytest.mark.asyncio
async def test_startup_timeout_budget_resumes_after_oauth_wait_ends() -> None:
    server_conn = _make_server_connection()

    async def _drive_events() -> None:
        await asyncio.sleep(0.01)
        server_conn.mark_oauth_wait_start()
        await asyncio.sleep(0.07)
        server_conn.mark_oauth_wait_end()

    started = time.monotonic()
    driver = asyncio.create_task(_drive_events())

    with pytest.raises(TimeoutError):
        await _wait_for_initialized_with_startup_budget(
            server_conn,
            startup_timeout_seconds=0.05,
            poll_interval_seconds=0.01,
        )

    await driver
    elapsed = time.monotonic() - started
    assert elapsed >= 0.10


class _DummyRegistry:
    def get_server_config(self, _server_name: str):
        return MCPServerSettings(name="demo", url="http://example.com/mcp")


@pytest.mark.asyncio
async def test_get_server_cancellation_cleans_up_pending_connection() -> None:
    manager = MCPConnectionManager(server_registry=cast("Any", _DummyRegistry()))
    server_conn = _make_server_connection()

    async def _fake_launch_server(*_args, **_kwargs):
        manager.running_servers["demo"] = server_conn
        return server_conn

    manager.launch_server = _fake_launch_server  # type: ignore[method-assign]

    task = asyncio.create_task(
        manager.get_server(
            "demo",
            client_session_factory=lambda *_args, **_kwargs: object(),
            startup_timeout_seconds=10.0,
        )
    )

    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert "demo" not in manager.running_servers
    assert server_conn._shutdown_event.is_set()
    assert server_conn._oauth_abort_event.is_set()


def test_is_oauth_timeout_message_requires_real_timeout_markers() -> None:
    assert _is_oauth_timeout_message("OAuth authorization timed out") is True
    assert _is_oauth_timeout_message("OAuth authorization was not completed in time.") is True
    assert _is_oauth_timeout_message("OAuth callback timeout") is True

    # Guard against false positives from words like 'RuntimeError' containing 'time'.
    assert (
        _is_oauth_timeout_message(
            "RuntimeError: OAuth local callback server unavailable and paste fallback is disabled"
        )
        is False
    )
