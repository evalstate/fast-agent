import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import httpx2
import pytest
from anyio import CancelScope
from mcp.client.streamable_http import streamable_http_client
from mcp.client.subscriptions import (
    ResourcesListChanged,
    ResourceUpdated,
    SubscriptionLost,
)
from mcp.shared.exceptions import MCPError
from mcp_types import DiscoverResult, SubscriptionFilter

from fast_agent.config import MCPServerAuthSettings, MCPServerSettings, Settings
from fast_agent.context import Context
from fast_agent.core.exceptions import ServerInitializationError
from fast_agent.mcp.auth.context import request_bearer_token
from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
from fast_agent.mcp.client_gateway import (
    _managed_http_transport_context,
    _prepare_headers_and_auth,
)
from fast_agent.mcp.client_gateway import (
    is_http_auth_challenge as _is_http_auth_challenge_error,
)
from fast_agent.mcp.mcp_connection_manager import (
    MCPConnectionManager,
    ServerConnection,
    _format_oauth_registration_404_details,
    _is_oauth_registration_404_message,
    _is_oauth_timeout_message,
    _run_subscription_loop,
    _server_lifecycle_task,
    _wait_for_initialized_with_startup_budget,
    _wait_for_shutdown_with_optional_ping,
)
from fast_agent.mcp.oauth_client import OAuthEventHandler
from fast_agent.mcp.transport_tracking import TransportChannelMetrics

if TYPE_CHECKING:
    from fast_agent.mcp.client_connection import MCPClientConnection


@pytest.mark.asyncio
async def test_http_response_hook_captures_session_id_and_auth_challenge() -> None:
    config = MCPServerSettings(name="test", transport="http", url="https://example.com/mcp")
    connection = ServerConnection(
        "test",
        config,
        cast(
            "Callable[[], MCPClientConnection]",
            lambda: streamable_http_client(config.url or ""),
        ),
        MCPClientCallbackRuntime(server_name="test", server_config=config),
    )
    response = httpx2.Response(
        401,
        headers={"Mcp-Session-Id": "session-123"},
        request=httpx2.Request("POST", config.url or ""),
    )

    await connection.capture_http_response(response)

    assert connection._auth_challenge_received is True
    assert connection.session_id == "session-123"


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
        "fast_agent.mcp.client_gateway.build_oauth_provider",
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
        "fast_agent.mcp.client_gateway.build_oauth_provider",
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
        "fast_agent.mcp.client_gateway.build_oauth_provider",
        _builder,
    )

    headers, auth, user_keys = _prepare_headers_and_auth(config, trigger_oauth=True)

    assert headers == {"Accept": "application/json"}
    assert auth is sentinel
    assert user_keys == set()
    assert calls == [config]


def test_prepare_headers_auto_mode_does_not_build_oauth(monkeypatch):
    config = MCPServerSettings(
        name="test",
        transport="sse",
        url="https://example.com/mcp",
    )

    def _builder(_config, **_kwargs):
        raise AssertionError("OAuth provider should not be built in auto mode.")

    monkeypatch.setattr(
        "fast_agent.mcp.client_gateway.build_oauth_provider",
        _builder,
    )

    headers, auth, user_keys = _prepare_headers_and_auth(config, trigger_oauth=None)

    assert headers == {}
    assert auth is None
    assert user_keys == set()


def test_prepare_headers_forwards_hf_request_token() -> None:
    config = MCPServerSettings(
        name="test",
        transport="http",
        url="https://huggingface.co/mcp",
        auth=MCPServerAuthSettings(forward="huggingface"),
    )

    saved_token = request_bearer_token.set("request-token")
    try:
        headers, auth, user_keys = _prepare_headers_and_auth(config, trigger_oauth=True)
    finally:
        request_bearer_token.reset(saved_token)

    assert headers == {"Authorization": "Bearer request-token"}
    assert auth is None
    assert user_keys == {"Authorization"}


def test_forward_hf_config_does_not_capture_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")

    config = MCPServerSettings(
        name="test",
        transport="http",
        url="https://huggingface.co/mcp",
        auth=MCPServerAuthSettings(forward="huggingface"),
    )

    assert config.headers is None


def test_prepare_headers_forwards_hf_space_request_token() -> None:
    config = MCPServerSettings(
        name="test",
        transport="http",
        url="https://demo.hf.space/mcp",
        auth=MCPServerAuthSettings(forward="huggingface"),
    )

    saved_token = request_bearer_token.set("request-token")
    try:
        headers, auth, user_keys = _prepare_headers_and_auth(config, trigger_oauth=True)
    finally:
        request_bearer_token.reset(saved_token)

    assert headers == {"X-HF-Authorization": "Bearer request-token"}
    assert auth is None
    assert user_keys == {"X-HF-Authorization"}


def test_prepare_headers_forward_preserves_explicit_authorization() -> None:
    config = MCPServerSettings(
        name="test",
        transport="http",
        url="https://huggingface.co/mcp",
        headers={"Authorization": "Bearer explicit"},
        auth=MCPServerAuthSettings(forward="huggingface"),
    )

    saved_token = request_bearer_token.set("request-token")
    try:
        headers, auth, user_keys = _prepare_headers_and_auth(config, trigger_oauth=True)
    finally:
        request_bearer_token.reset(saved_token)

    assert headers == {"Authorization": "Bearer explicit"}
    assert auth is None
    assert user_keys == {"Authorization"}


@pytest.mark.asyncio
async def test_managed_http_transport_context_closes_client_after_transport() -> None:
    class _FakeClient:
        def __init__(self) -> None:
            self.entered = False
            self.exited = False

        async def __aenter__(self):
            self.entered = True
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            self.exited = True
            return False

    class _FakeTransportContext:
        def __init__(self) -> None:
            self.entered = False
            self.exited = False

        async def __aenter__(self):
            self.entered = True
            return object(), object(), None

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            self.exited = True
            return None

    client = cast("Any", _FakeClient())
    transport_context = _FakeTransportContext()

    async with _managed_http_transport_context(client, transport_context) as streams:
        assert streams[2] is None
        assert transport_context.entered is True
        assert transport_context.exited is False
        assert client.entered is True
        assert client.exited is False

    assert transport_context.exited is True
    assert client.exited is True


@pytest.mark.asyncio
async def test_server_lifecycle_sets_initialized_on_startup_failure():
    class DummyTransportContext:
        async def __aenter__(self):
            return object(), object(), None

        async def __aexit__(self, exc_type, exc, tb):
            return None

    server_conn = ServerConnection(
        server_name="test-server",
        server_config=MCPServerSettings(name="test-server", url="http://example.com/mcp"),
        client_connection_factory=cast("Callable[[], MCPClientConnection]", DummyTransportContext),
        callback_runtime=_callback_runtime(),
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

    return ServerConnection(
        server_name="test-server",
        server_config=MCPServerSettings(name="test-server", url="http://example.com/mcp"),
        client_connection_factory=cast("Callable[[], MCPClientConnection]", DummyTransportContext),
        callback_runtime=_callback_runtime(),
    )


def _callback_runtime() -> MCPClientCallbackRuntime:
    return MCPClientCallbackRuntime(
        server_name="test-server",
        server_config=MCPServerSettings(name="test-server", url="http://example.com/mcp"),
    )


@pytest.mark.asyncio
async def test_subscription_limit_error_is_not_retried() -> None:
    class RejectedSubscription:
        async def __aenter__(self):
            raise MCPError(-32000, "Subscription limit reached")

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    class SubscriptionLimitClient:
        discover_result = DiscoverResult.model_validate(
            {
                "supportedVersions": ["2026-07-28"],
                "capabilities": {"tools": {"listChanged": True}},
            }
        )

        def __init__(self) -> None:
            self.listen_calls = 0

        def listen(self, **kwargs):
            del kwargs
            self.listen_calls += 1
            return RejectedSubscription()

    server_conn = _make_server_connection()
    client = SubscriptionLimitClient()
    server_conn.client = cast("Any", client)
    server_conn.transport_metrics = TransportChannelMetrics()

    await asyncio.wait_for(_run_subscription_loop(server_conn), timeout=0.1)

    assert client.listen_calls == 1
    assert server_conn.subscription_state == "error"
    listen = server_conn.transport_metrics.snapshot().listen
    assert listen is not None
    assert listen.last_error == "Subscription limit reached"


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
    active_home = None
    no_home = False

    def get_server_config(self, _server_name: str):
        return MCPServerSettings(name="demo", url="http://example.com/mcp")


@pytest.mark.parametrize(
    ("with_user_oauth_handler", "expected_console_output"),
    [(False, True), (True, False)],
)
def test_managed_connection_preserves_oauth_console_output_without_user_handler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_user_oauth_handler: bool,
    expected_console_output: bool,
) -> None:
    manager = MCPConnectionManager(server_registry=cast("Any", _DummyRegistry()))
    config = MCPServerSettings(name="demo", url="http://example.com/mcp")
    server_conn = _make_server_connection()
    captured_hooks: list[Any] = []
    sentinel = object()

    def _fake_create_client_connection(*_args, hooks, **_kwargs):
        captured_hooks.append(hooks)
        return sentinel

    async def _user_oauth_handler(_event) -> None:
        return None

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.create_client_connection",
        _fake_create_client_connection,
    )
    factory = manager._client_connection_factory(
        [server_conn],
        server_name="demo",
        config=config,
        oauth_mode="auto",
        oauth_active=True,
        oauth_event_handler=_user_oauth_handler if with_user_oauth_handler else None,
        allow_oauth_paste_fallback=True,
        transport_metrics=None,
    )

    assert factory() is sentinel
    assert len(captured_hooks) == 1
    assert captured_hooks[0].oauth_event_handler is not None
    assert captured_hooks[0].emit_oauth_console_output is expected_console_output


def test_disabled_mcp_diagnostics_skips_timeline_metrics_and_ping_history() -> None:
    settings = Settings.model_validate({"mcp": {"diagnostics": {"enabled": False}}})
    manager = MCPConnectionManager(
        server_registry=cast("Any", _DummyRegistry()),
        context=Context(config=settings),
    )
    config = MCPServerSettings(name="demo", url="http://example.com/mcp")

    assert manager._launch_transport_metrics(config) is None

    connection = ServerConnection(
        server_name="demo",
        server_config=config,
        client_connection_factory=lambda: cast("Any", object()),
        callback_runtime=_callback_runtime(),
    )
    connection.record_ping_event("ping")

    assert connection.build_ping_activity_buckets(30, 2) == ["none", "none"]


class _DummyStdioRegistry:
    active_home = None
    no_home = False

    def __init__(self, config: MCPServerSettings) -> None:
        self._config = config

    def get_server_config(self, _server_name: str):
        return self._config


@pytest.mark.asyncio
async def test_get_server_cancellation_cleans_up_pending_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = MCPConnectionManager(server_registry=cast("Any", _DummyRegistry()))
    server_conn = _make_server_connection()
    lifecycle_complete = asyncio.Event()

    async def _run_lifecycle() -> None:
        await server_conn.wait_for_shutdown_request()
        lifecycle_complete.set()
        server_conn._lifecycle_complete_event.set()

    async def _fake_launch_server(
        *,
        server_name: str,
        server_config: MCPServerSettings | None,
        callback_runtime: MCPClientCallbackRuntime,
        startup_timeout_seconds: float | None = None,
        trigger_oauth: bool | None = None,
        oauth_event_handler: OAuthEventHandler | None = None,
        allow_oauth_paste_fallback: bool = True,
    ) -> ServerConnection:
        del server_name, server_config, callback_runtime, startup_timeout_seconds
        del trigger_oauth, oauth_event_handler, allow_oauth_paste_fallback
        manager.running_servers["demo"] = server_conn
        asyncio.create_task(_run_lifecycle())
        return server_conn

    monkeypatch.setattr(manager, "launch_server", _fake_launch_server)

    task = asyncio.create_task(
        manager.get_server(
            "demo",
            callback_runtime=_callback_runtime(),
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
    assert lifecycle_complete.is_set()


@pytest.mark.asyncio
async def test_get_server_startup_timeout_cancels_blocked_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = MCPConnectionManager(server_registry=cast("Any", _DummyRegistry()))
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class HangingTransportContext:
        async def __aenter__(self):
            entered.set()
            try:
                await asyncio.Event().wait()
            except BaseException:
                cancelled.set()
                raise

        async def __aexit__(self, exc_type, exc, tb):
            return None

    server_conn = ServerConnection(
        server_name="demo",
        server_config=MCPServerSettings(
            name="demo",
            transport="http",
            url="http://127.0.0.1:9/mcp",
        ),
        client_connection_factory=cast(
            "Callable[[], MCPClientConnection]", HangingTransportContext
        ),
        callback_runtime=_callback_runtime(),
    )

    async def _fake_launch_server(
        *,
        server_name: str,
        server_config: MCPServerSettings | None,
        callback_runtime: MCPClientCallbackRuntime,
        startup_timeout_seconds: float | None = None,
        trigger_oauth: bool | None = None,
        oauth_event_handler: OAuthEventHandler | None = None,
        allow_oauth_paste_fallback: bool = True,
    ) -> ServerConnection:
        del server_name, server_config, callback_runtime, startup_timeout_seconds
        del trigger_oauth, oauth_event_handler, allow_oauth_paste_fallback
        manager.running_servers["demo"] = server_conn
        asyncio.create_task(_server_lifecycle_task(server_conn))
        await entered.wait()
        return server_conn

    monkeypatch.setattr(manager, "launch_server", _fake_launch_server)

    with pytest.raises(ServerInitializationError):
        await manager.get_server(
            "demo",
            callback_runtime=_callback_runtime(),
            startup_timeout_seconds=0.01,
        )

    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    assert "demo" not in manager.running_servers
    assert server_conn._shutdown_event.is_set()
    assert server_conn._oauth_abort_event.is_set()
    assert server_conn._lifecycle_complete_event.is_set()


@pytest.mark.asyncio
async def test_get_server_retries_with_oauth_after_401_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = MCPConnectionManager(server_registry=cast("Any", _DummyRegistry()))
    unhealthy = _make_server_connection()
    unhealthy._error_occurred = True
    unhealthy._error_message = "HTTP Error: 401 Unauthorized for URL: http://example.com/mcp"

    healthy = _make_server_connection()
    healthy.client = cast("Any", object())

    calls: list[bool | None] = []

    async def _fake_launch_and_wait_for_server(
        *,
        server_name: str,
        server_config: MCPServerSettings | None,
        callback_runtime: MCPClientCallbackRuntime,
        startup_timeout_seconds: float | None,
        trigger_oauth: bool | None,
        oauth_event_handler: OAuthEventHandler | None,
        allow_oauth_paste_fallback: bool,
        timeout_action: str,
    ) -> ServerConnection:
        del server_name, server_config, callback_runtime, startup_timeout_seconds
        del oauth_event_handler, allow_oauth_paste_fallback, timeout_action
        trigger = trigger_oauth
        calls.append(trigger)
        manager._server_oauth_mode["demo"] = "force" if trigger is True else "auto"
        manager._server_oauth_active["demo"] = trigger is True
        return healthy if trigger is True else unhealthy

    async def _fake_retry_server_with_oauth(
        *,
        server_name: str,
        server_conn: ServerConnection,
        server_config: MCPServerSettings | None,
        callback_runtime: MCPClientCallbackRuntime,
        startup_timeout_seconds: float | None,
        oauth_event_handler: OAuthEventHandler | None,
        allow_oauth_paste_fallback: bool,
        timeout_action: str,
    ) -> ServerConnection:
        del server_name, server_conn, server_config, callback_runtime, startup_timeout_seconds
        del oauth_event_handler, allow_oauth_paste_fallback, timeout_action
        calls.append(True)
        manager._server_oauth_mode["demo"] = "force"
        manager._server_oauth_active["demo"] = True
        return healthy

    monkeypatch.setattr(manager, "_launch_and_wait_for_server", _fake_launch_and_wait_for_server)
    monkeypatch.setattr(manager, "_retry_server_with_oauth", _fake_retry_server_with_oauth)

    server_conn = await manager.get_server(
        "demo",
        callback_runtime=_callback_runtime(),
    )

    assert server_conn is healthy
    assert calls == [None, True]


@pytest.mark.asyncio
async def test_get_server_formats_stdio_missing_executable_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingStdioClient:
        async def __aenter__(self):
            raise FileNotFoundError(2, "No such file or directory")

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

    def _failing_stdio_client(*_args, **_kwargs):
        return _FailingStdioClient()

    monkeypatch.setattr(
        "fast_agent.mcp.client_gateway.tracking_stdio_client",
        _failing_stdio_client,
    )

    manager = MCPConnectionManager(
        server_registry=cast(
            "Any",
            _DummyStdioRegistry(
                MCPServerSettings(
                    name="demo",
                    transport="stdio",
                    command="missing-mcp-server",
                    args=["serve"],
                )
            ),
        )
    )

    async with manager:
        with pytest.raises(ServerInitializationError) as exc_info:
            await manager.get_server(
                "demo",
                callback_runtime=_callback_runtime(),
                startup_timeout_seconds=1.0,
            )

    assert exc_info.value.message == "MCP Server: 'demo': Failed to start stdio server."
    details = exc_info.value.details
    assert "Failed to start stdio MCP server command: missing-mcp-server serve." in details
    assert "Executable not found on PATH: missing-mcp-server" in details
    assert "Traceback" not in details


@pytest.mark.asyncio
async def test_get_server_formats_stdio_missing_cwd_without_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class _FailingStdioClient:
        async def __aenter__(self):
            raise FileNotFoundError(2, "No such file or directory")

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

    def _failing_stdio_client(*_args, **_kwargs):
        return _FailingStdioClient()

    missing_cwd = str(tmp_path / "missing-dir")

    monkeypatch.setattr(
        "fast_agent.mcp.client_gateway.tracking_stdio_client",
        _failing_stdio_client,
    )

    manager = MCPConnectionManager(
        server_registry=cast(
            "Any",
            _DummyStdioRegistry(
                MCPServerSettings(
                    name="demo",
                    transport="stdio",
                    command="python",
                    args=["-m", "demo_server"],
                    cwd=missing_cwd,
                )
            ),
        )
    )

    async with manager:
        with pytest.raises(ServerInitializationError) as exc_info:
            await manager.get_server(
                "demo",
                callback_runtime=_callback_runtime(),
                startup_timeout_seconds=1.0,
            )

    assert exc_info.value.message == "MCP Server: 'demo': Failed to start stdio server."
    details = exc_info.value.details
    assert "Working directory not found" in details
    assert missing_cwd in details
    assert "Traceback" not in details


@pytest.mark.asyncio
async def test_get_server_stdio_timeout_includes_recent_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MCPServerSettings(
        name="demo",
        transport="stdio",
        command="npx",
        args=["-y", "@wonderwhy-er/desktop-commander@latest"],
    )
    manager = MCPConnectionManager(server_registry=cast("Any", _DummyStdioRegistry(config)))
    server_conn = ServerConnection(
        server_name="demo",
        server_config=config,
        client_connection_factory=lambda: cast("Any", object()),
        callback_runtime=_callback_runtime(),
    )
    server_conn.record_stdio_stderr("npm notice downloading desktop-commander")
    server_conn.record_stdio_stderr("npm warn request took longer than expected")

    async def _run_lifecycle() -> None:
        await server_conn.wait_for_shutdown_request()
        server_conn._lifecycle_complete_event.set()

    async def _fake_launch_server(
        *,
        server_name: str,
        server_config: MCPServerSettings | None,
        callback_runtime: MCPClientCallbackRuntime,
        startup_timeout_seconds: float | None = None,
        trigger_oauth: bool | None = None,
        oauth_event_handler: OAuthEventHandler | None = None,
        allow_oauth_paste_fallback: bool = True,
    ) -> ServerConnection:
        del server_name, server_config, callback_runtime, startup_timeout_seconds
        del trigger_oauth, oauth_event_handler, allow_oauth_paste_fallback
        manager.running_servers["demo"] = server_conn
        asyncio.create_task(_run_lifecycle())
        return server_conn

    monkeypatch.setattr(manager, "launch_server", _fake_launch_server)

    with pytest.raises(ServerInitializationError) as exc_info:
        await manager.get_server(
            "demo",
            callback_runtime=_callback_runtime(),
            startup_timeout_seconds=0.01,
        )

    details = exc_info.value.details
    assert "Try increasing --timeout or verify server/network startup." in details
    assert "Recent stderr from stdio server:" in details
    assert "npm notice downloading desktop-commander" in details
    assert "npm warn request took longer than expected" in details


@pytest.mark.asyncio
async def test_connection_manager_exit_skips_grace_sleep_without_running_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoRunningServersManager(MCPConnectionManager):
        async def disconnect_all(self) -> bool:
            return False

    manager = _NoRunningServersManager(server_registry=cast("Any", _DummyRegistry()))
    task_group = asyncio.TaskGroup()
    await task_group.__aenter__()
    manager._task_group_active = True
    manager._task_group = task_group

    async def _unexpected_sleep(_delay: float) -> None:
        raise AssertionError("shutdown grace sleep should be skipped")

    monkeypatch.setattr(asyncio, "sleep", _unexpected_sleep)

    await manager.__aexit__(None, None, None)

    assert manager._task_group_active is False
    assert manager._task_group is None


@pytest.mark.asyncio
async def test_connection_manager_exit_needs_no_fixed_shutdown_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RunningServersManager(MCPConnectionManager):
        async def disconnect_all(self) -> bool:
            return True

    manager = _RunningServersManager(server_registry=cast("Any", _DummyRegistry()))
    task_group = asyncio.TaskGroup()
    await task_group.__aenter__()
    manager._task_group_active = True
    manager._task_group = task_group

    async def _unexpected_sleep(_delay: float) -> None:
        raise AssertionError("lifecycle completion replaces fixed shutdown sleeps")

    monkeypatch.setattr(asyncio, "sleep", _unexpected_sleep)

    await manager.__aexit__(None, None, None)

    assert manager._task_group_active is False
    assert manager._task_group is None


class _DerivedStateSimulator:
    def __init__(
        self,
        *,
        refresh_states: list[tuple[str, ...]] | None = None,
        event_states: list[tuple[str, ...]] | None = None,
    ) -> None:
        self.current_uris: tuple[str, ...] = ()
        self.refresh_states = list(refresh_states or [])
        self.event_states = list(event_states or [])
        self.refresh_count = 0
        self.events: list[object] = []

    def selected_materialized_resource_uris(self, server_name: str) -> tuple[str, ...]:
        del server_name
        return self.current_uris

    async def refresh_subscription_state(self, server_name: str) -> tuple[str, ...]:
        del server_name
        self.refresh_count += 1
        if self.refresh_states:
            self.current_uris = self.refresh_states.pop(0)
        return self.current_uris

    async def handle_subscription_event(self, server_name: str, event: object) -> None:
        del server_name
        self.events.append(event)
        if self.event_states:
            self.current_uris = self.event_states.pop(0)


class _ListenerContextSimulator:
    def __init__(
        self,
        client: "_ListenerClientSimulator",
        events: list[object],
        honored: SubscriptionFilter,
    ) -> None:
        self.client = client
        self.events = list(events)
        self.honored = honored

    async def __aenter__(self):
        self.client.active_listeners += 1
        self.client.max_active_listeners = max(
            self.client.max_active_listeners,
            self.client.active_listeners,
        )
        self.client.listen_opened.set()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        self.client.active_listeners -= 1
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.events:
            event = self.events.pop(0)
            if isinstance(event, BaseException):
                raise event
            return event
        await self.client.hold_open.wait()
        raise StopAsyncIteration


class _ListenerClientSimulator:
    def __init__(
        self,
        discover_result: DiscoverResult | None,
        *,
        scripts: list[list[object]] | None = None,
        honored_filters: list[SubscriptionFilter] | None = None,
    ) -> None:
        self.discover_result = discover_result
        self.scripts = list(scripts or [])
        self.honored_filters = list(honored_filters or [])
        self.listen_calls: list[dict[str, Any]] = []
        self.listen_opened = asyncio.Event()
        self.hold_open = asyncio.Event()
        self.active_listeners = 0
        self.max_active_listeners = 0

    def listen(self, **kwargs: Any) -> _ListenerContextSimulator:
        self.listen_calls.append(kwargs)
        events = self.scripts.pop(0) if self.scripts else []
        honored = (
            self.honored_filters.pop(0)
            if self.honored_filters
            else SubscriptionFilter.model_validate(kwargs)
        )
        return _ListenerContextSimulator(self, events, honored)


async def _wait_until(predicate: Callable[[], bool]) -> None:
    while not predicate():
        await asyncio.sleep(0.001)


def _modern_discovery(capabilities: dict[str, object]) -> DiscoverResult:
    return DiscoverResult.model_validate(
        {
            "supportedVersions": ["2026-07-28"],
            "capabilities": capabilities,
        }
    )


def _subscription_server(
    client: _ListenerClientSimulator,
    derived_state: _DerivedStateSimulator,
    *,
    ready: bool = True,
) -> ServerConnection:
    server_conn = _make_server_connection()
    server_conn.client = cast("Any", client)
    server_conn.protocol_era = "modern"
    server_conn._callback_runtime = MCPClientCallbackRuntime(
        server_name="test-server",
        server_config=server_conn.server_config,
        aggregator=cast("Any", derived_state),
    )
    if ready:
        server_conn._callback_runtime.mark_subscription_ready()
    return server_conn


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("discover_result", "expected_filter"),
    [
        (None, None),
        (_modern_discovery({}), None),
        (
            _modern_discovery({"tools": {"listChanged": True}}),
            {
                "tools_list_changed": True,
                "prompts_list_changed": False,
                "resources_list_changed": False,
                "resource_subscriptions": (),
            },
        ),
        (
            _modern_discovery({"prompts": {"listChanged": True}}),
            {
                "tools_list_changed": False,
                "prompts_list_changed": True,
                "resources_list_changed": False,
                "resource_subscriptions": (),
            },
        ),
        (
            _modern_discovery({"resources": {"subscribe": False, "listChanged": True}}),
            {
                "tools_list_changed": False,
                "prompts_list_changed": False,
                "resources_list_changed": True,
                "resource_subscriptions": (),
            },
        ),
        (
            _modern_discovery({"resources": {"subscribe": True}}),
            {
                "tools_list_changed": False,
                "prompts_list_changed": False,
                "resources_list_changed": False,
                "resource_subscriptions": (),
            },
        ),
    ],
)
async def test_modern_listener_capability_matrix(
    discover_result: DiscoverResult | None,
    expected_filter: dict[str, Any] | None,
) -> None:
    client = _ListenerClientSimulator(discover_result)
    server_conn = _subscription_server(client, _DerivedStateSimulator())

    wait_task = asyncio.create_task(_wait_for_shutdown_with_optional_ping(server_conn))
    if expected_filter is None:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    else:
        await asyncio.wait_for(client.listen_opened.wait(), timeout=1)
    server_conn.request_shutdown()
    await wait_task

    assert client.listen_calls == ([] if expected_filter is None else [expected_filter])
    assert server_conn.subscription_state == ("disabled" if expected_filter is None else "open")


@pytest.mark.asyncio
async def test_dropped_subscription_epoch_converges_after_reack_refresh() -> None:
    client = _ListenerClientSimulator(
        _modern_discovery({"tools": {"listChanged": True}}),
        scripts=[
            [SubscriptionLost("simulated dropped epoch")],
            [],
        ],
    )
    derived_state = _DerivedStateSimulator(
        refresh_states=[("ui://epoch/one",), ("ui://epoch/two",)]
    )
    server_conn = _subscription_server(client, derived_state)

    wait_task = asyncio.create_task(_wait_for_shutdown_with_optional_ping(server_conn))
    await asyncio.wait_for(
        _wait_until(lambda: derived_state.refresh_count == 2),
        timeout=1,
    )
    server_conn.request_shutdown()
    await wait_task

    assert derived_state.current_uris == ("ui://epoch/two",)
    assert len(client.listen_calls) == 2


@pytest.mark.asyncio
async def test_partial_subscription_acknowledgment_reports_degraded_state() -> None:
    client = _ListenerClientSimulator(
        _modern_discovery({"tools": {"listChanged": True}}),
        honored_filters=[SubscriptionFilter()],
    )
    derived_state = _DerivedStateSimulator()
    server_conn = _subscription_server(client, derived_state)

    wait_task = asyncio.create_task(_wait_for_shutdown_with_optional_ping(server_conn))
    await asyncio.wait_for(
        _wait_until(lambda: server_conn.subscription_state == "partial"),
        timeout=1,
    )
    server_conn.request_shutdown()
    await wait_task

    assert derived_state.refresh_count == 1
    assert len(client.listen_calls) == 1


@pytest.mark.asyncio
async def test_initial_attachment_commit_rotates_to_materialized_resource_uris() -> None:
    client = _ListenerClientSimulator(
        _modern_discovery({"resources": {"subscribe": True}}),
        scripts=[[], []],
    )
    derived_state = _DerivedStateSimulator(
        refresh_states=[
            ("ui://component/initial",),
            ("ui://component/initial",),
        ]
    )
    server_conn = _subscription_server(client, derived_state, ready=False)

    wait_task = asyncio.create_task(_wait_for_shutdown_with_optional_ping(server_conn))
    await asyncio.wait_for(client.listen_opened.wait(), timeout=1)
    assert derived_state.refresh_count == 0

    server_conn._callback_runtime.mark_subscription_ready()
    await asyncio.wait_for(_wait_until(lambda: len(client.listen_calls) == 2), timeout=1)
    server_conn.request_shutdown()
    await wait_task

    assert [call["resource_subscriptions"] for call in client.listen_calls] == [
        (),
        ("ui://component/initial",),
    ]
    assert client.max_active_listeners == 1


@pytest.mark.asyncio
async def test_resource_events_rotate_serial_listener_without_planned_backoff() -> None:
    client = _ListenerClientSimulator(
        _modern_discovery({"resources": {"subscribe": True, "listChanged": True}}),
        scripts=[
            [],
            [ResourcesListChanged()],
            [ResourceUpdated(uri="ui://component/two")],
            [],
        ],
    )
    derived_state = _DerivedStateSimulator(
        refresh_states=[
            ("ui://component/one",),
            ("ui://component/one",),
            ("ui://component/two",),
            ("ui://component/three",),
        ],
        event_states=[
            ("ui://component/two",),
            ("ui://component/three",),
        ],
    )
    server_conn = _subscription_server(client, derived_state)

    wait_task = asyncio.create_task(_wait_for_shutdown_with_optional_ping(server_conn))
    await asyncio.wait_for(_wait_until(lambda: len(client.listen_calls) == 4), timeout=0.2)
    server_conn.request_shutdown()
    await wait_task

    assert [call["resource_subscriptions"] for call in client.listen_calls] == [
        (),
        ("ui://component/one",),
        ("ui://component/two",),
        ("ui://component/three",),
    ]
    assert [type(event) for event in derived_state.events] == [
        ResourcesListChanged,
        ResourceUpdated,
    ]
    assert client.max_active_listeners == 1
    assert client.active_listeners == 0


@pytest.mark.asyncio
async def test_connection_manager_does_not_leak_cancel_scope_into_caller() -> None:
    manager = MCPConnectionManager(server_registry=cast("Any", _DummyRegistry()))

    with CancelScope():
        await manager.__aenter__()

    await manager.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_disconnect_server_waits_for_lifecycle_cleanup() -> None:
    manager = MCPConnectionManager(server_registry=cast("Any", _DummyRegistry()))
    server_conn = _make_server_connection()
    manager.running_servers["demo"] = server_conn
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()

    async def _run_lifecycle() -> None:
        await server_conn.wait_for_shutdown_request()
        cleanup_started.set()
        await allow_cleanup.wait()
        server_conn._lifecycle_complete_event.set()

    lifecycle_task = asyncio.create_task(_run_lifecycle())
    disconnect_task = asyncio.create_task(manager.disconnect_server("demo"))

    await cleanup_started.wait()
    assert not disconnect_task.done()
    assert manager.running_servers["demo"] is server_conn

    allow_cleanup.set()
    await disconnect_task
    await lifecycle_task

    assert "demo" not in manager.running_servers


@pytest.mark.asyncio
async def test_reconnect_does_not_overlap_old_runtime_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = MCPConnectionManager(server_registry=cast("Any", _DummyRegistry()))
    old_conn = _make_server_connection()
    new_conn = _make_server_connection()
    new_conn.client = cast("Any", object())
    manager.running_servers["demo"] = old_conn
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    launch_started = asyncio.Event()

    async def _run_old_lifecycle() -> None:
        await old_conn.wait_for_shutdown_request()
        cleanup_started.set()
        await allow_cleanup.wait()
        old_conn._lifecycle_complete_event.set()

    async def _fake_launch_and_wait_for_server(**_kwargs: Any) -> ServerConnection:
        launch_started.set()
        manager.running_servers["demo"] = new_conn
        return new_conn

    lifecycle_task = asyncio.create_task(_run_old_lifecycle())
    monkeypatch.setattr(manager, "_launch_and_wait_for_server", _fake_launch_and_wait_for_server)
    reconnect_task = asyncio.create_task(
        manager.reconnect_server("demo", callback_runtime=_callback_runtime())
    )

    await cleanup_started.wait()
    assert not launch_started.is_set()

    allow_cleanup.set()
    assert await reconnect_task is new_conn
    await lifecycle_task

    assert launch_started.is_set()
    assert manager.running_servers["demo"] is new_conn


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

    # Guard against traceback text that mentions oauth variable names and timeout kwargs
    # without any real OAuth timeout happening.
    assert (
        _is_oauth_timeout_message(
            "ImportError: Using SOCKS proxy, but the 'socksio' package is not installed. auth=oauth_auth timeout=10"
        )
        is False
    )


def test_is_oauth_registration_404_message_detects_registration_failures() -> None:
    assert (
        _is_oauth_registration_404_message(
            "OAuthRegistrationError: Registration failed: 404 404 page not found"
        )
        is True
    )
    assert _is_oauth_registration_404_message("HTTP Error: 404 Not Found for URL: /mcp") is False


def test_is_http_auth_challenge_error_detects_401_responses() -> None:
    assert _is_http_auth_challenge_error("HTTP Error: 401 Unauthorized for URL: /mcp") is True
    assert _is_http_auth_challenge_error("401 Client Error: Unauthorized for url") is True
    assert _is_http_auth_challenge_error("WWW-Authenticate: Bearer realm=example") is True
    assert _is_http_auth_challenge_error("HTTP Error: 404 Not Found for URL: /mcp") is False


def test_format_oauth_registration_404_details_includes_copilot_hint() -> None:
    details = _format_oauth_registration_404_details(
        "OAuthRegistrationError: Registration failed: 404 404 page not found",
        "https://githubcopilot.com/mcp/",
    )
    assert "dynamic client registration" in details
    assert "--client-metadata-url" in details
    assert "--auth <token>" in details
    assert "GitHub Copilot MCP" in details


def test_oauth_traceback_filter_suppresses_non_debug_oauth_flow_errors() -> None:
    manager = MCPConnectionManager(server_registry=cast("Any", _DummyRegistry()))
    oauth_logger = logging.getLogger("mcp.client.auth.oauth2")
    initial_filter_count = len(oauth_logger.filters)
    root_logger = logging.getLogger()
    original_level = root_logger.level

    try:
        root_logger.setLevel(logging.INFO)
        manager._suppress_mcp_oauth_cancel_errors()
        added_filters = oauth_logger.filters[initial_filter_count:]
        assert added_filters
        oauth_filter = added_filters[-1]
        assert isinstance(oauth_filter, logging.Filter)

        record = logging.LogRecord(
            name="mcp.client.auth.oauth2",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="OAuth flow error",
            args=(),
            exc_info=(RuntimeError, RuntimeError("boom"), None),
        )
        assert oauth_filter.filter(record) is False
    finally:
        root_logger.setLevel(original_level)
        for filt in oauth_logger.filters[initial_filter_count:]:
            oauth_logger.removeFilter(filt)
