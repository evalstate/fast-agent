import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, cast

import pytest
from anyio import create_task_group
from httpx import HTTPStatusError, Request, Response
from mcp import ClientSession

from fast_agent.config import MCPServerSettings
from fast_agent.core.exceptions import ServerInitializationError
from fast_agent.mcp.interfaces import ClientSessionFactory
from fast_agent.mcp.mcp_connection_manager import (
    MCPConnectionManager,
    ServerConnection,
    _format_lifecycle_exception_group_errors,
    _format_oauth_registration_404_details,
    _format_ping_shutdown_error,
    _format_user_auth_skip_oauth_message,
    _is_http_auth_challenge_error,
    _is_oauth_registration_404_message,
    _is_oauth_timeout_message,
    _managed_http_transport_context,
    _pingable_session,
    _prepare_headers_and_auth,
    _server_lifecycle_task,
    _wait_for_initialized_with_startup_budget,
)
from fast_agent.mcp.oauth_client import OAuthEventHandler


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

    prepared_auth = _prepare_headers_and_auth(config)

    assert prepared_auth.headers == {"Authorization": "Bearer user-token"}
    assert prepared_auth.headers is not config.headers
    assert prepared_auth.oauth_provider is None
    assert prepared_auth.user_auth_keys == {"Authorization"}


def test_prepare_headers_respects_case_insensitive_authorization(monkeypatch):
    config = MCPServerSettings(
        name="test",
        transport="http",
        url="https://example.com/mcp",
        headers={" authorization ": "Bearer user-token"},
    )

    def _builder(_config, **_kwargs):
        raise AssertionError("OAuth provider should not be built when authorization header is set.")

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.build_oauth_provider",
        _builder,
    )

    prepared_auth = _prepare_headers_and_auth(config)

    assert prepared_auth.headers == {" authorization ": "Bearer user-token"}
    assert prepared_auth.oauth_provider is None
    assert prepared_auth.user_auth_keys == {" authorization "}


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

    prepared_auth = _prepare_headers_and_auth(config, trigger_oauth=True)

    assert prepared_auth.headers == {"Accept": "application/json"}
    assert prepared_auth.oauth_provider is sentinel
    assert prepared_auth.user_auth_keys == set()
    assert calls == [config]


def test_prepare_headers_forced_oauth_scrubs_user_authorization(monkeypatch):
    config = MCPServerSettings(
        name="test",
        transport="http",
        url="https://example.com/mcp",
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer stale-token",
            "X-HF-Authorization": "Bearer stale-hf-token",
        },
    )

    sentinel = object()

    def _builder(_config, **_kwargs):
        return sentinel

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.build_oauth_provider",
        _builder,
    )

    prepared_auth = _prepare_headers_and_auth(
        config,
        trigger_oauth=True,
        oauth_mode="force",
    )

    assert prepared_auth.headers == {"Accept": "application/json"}
    assert prepared_auth.oauth_provider is sentinel
    assert prepared_auth.user_auth_keys == {"Authorization", "X-HF-Authorization"}


def test_prepare_headers_auto_oauth_defers_to_user_authorization(monkeypatch):
    config = MCPServerSettings(
        name="test",
        transport="http",
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer user-token"},
    )

    def _builder(_config, **_kwargs):
        raise AssertionError("Auto OAuth should defer to explicit user Authorization.")

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.build_oauth_provider",
        _builder,
    )

    prepared_auth = _prepare_headers_and_auth(
        config,
        trigger_oauth=True,
        oauth_mode="auto",
    )

    assert prepared_auth.headers == {"Authorization": "Bearer user-token"}
    assert prepared_auth.oauth_provider is None
    assert prepared_auth.user_auth_keys == {"Authorization"}


def test_prepare_headers_auto_mode_does_not_build_oauth(monkeypatch):
    config = MCPServerSettings(
        name="test",
        transport="sse",
        url="https://example.com/mcp",
    )

    def _builder(_config, **_kwargs):
        raise AssertionError("OAuth provider should not be built in auto mode.")

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.build_oauth_provider",
        _builder,
    )

    prepared_auth = _prepare_headers_and_auth(config, trigger_oauth=None)

    assert prepared_auth.headers == {}
    assert prepared_auth.oauth_provider is None
    assert prepared_auth.user_auth_keys == set()


def test_format_lifecycle_exception_group_errors_flattens_nested_causes() -> None:
    request = Request("GET", "https://example.com/mcp")
    response = Response(503, request=request)
    http_error = HTTPStatusError("service unavailable", request=request, response=response)
    caused_error = RuntimeError("startup failed")
    caused_error.__cause__ = ValueError("bad config")
    exception_group = ExceptionGroup(
        "outer",
        [
            ExceptionGroup("inner", [http_error]),
            caused_error,
        ],
    )

    assert _format_lifecycle_exception_group_errors(exception_group) == [
        "HTTP Error: 503 Service Unavailable for URL: https://example.com/mcp",
        "RuntimeError: startup failed",
        "Caused by: ValueError: bad config",
    ]


def test_format_ping_shutdown_error_uses_singular_count() -> None:
    assert _format_ping_shutdown_error(1, RuntimeError("timeout")) == (
        "Ping failed 1 time; last error: timeout"
    )


def test_format_ping_shutdown_error_uses_plural_count() -> None:
    assert _format_ping_shutdown_error(3, RuntimeError("timeout")) == (
        "Ping failed 3 times; last error: timeout"
    )


def test_pingable_session_narrows_optional_ping_capability() -> None:
    class _Pingable:
        async def ping(self, read_timeout_seconds=None) -> object:
            del read_timeout_seconds
            return object()

    session = _Pingable()

    assert _pingable_session(session) is session
    assert _pingable_session(object()) is None
    assert _pingable_session(None) is None


def test_format_user_auth_skip_oauth_message_uses_singular_header_count() -> None:
    assert _format_user_auth_skip_oauth_message("docs", {"Authorization"}) == (
        "docs: Using user-specified 1 auth header; skipping OAuth provider."
    )


def test_format_user_auth_skip_oauth_message_uses_plural_header_count() -> None:
    assert _format_user_auth_skip_oauth_message(
        "docs",
        {"Authorization", "X-HF-Authorization"},
    ) == "docs: Using user-specified 2 auth headers; skipping OAuth provider."


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


def _dummy_client_session_factory(*_args: Any, **_kwargs: Any) -> ClientSession:
    return cast("ClientSession", object())


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


class _DummyStdioRegistry:
    def __init__(self, config: MCPServerSettings) -> None:
        self._config = config

    def get_server_config(self, _server_name: str):
        return self._config


@pytest.mark.asyncio
async def test_launch_server_invalid_startup_timeout_has_no_manager_side_effects() -> None:
    manager = MCPConnectionManager(server_registry=cast("Any", _DummyRegistry()))

    with pytest.raises(ValueError, match="startup_timeout_seconds must be > 0"):
        await manager.launch_server(
            server_name="demo",
            client_session_factory=_dummy_client_session_factory,
            startup_timeout_seconds=0,
        )

    assert manager.running_servers == {}
    assert manager._server_oauth_mode == {}
    assert manager._server_oauth_active == {}
    assert manager._task_group_active is False


@pytest.mark.asyncio
async def test_get_server_cancellation_cleans_up_pending_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = MCPConnectionManager(server_registry=cast("Any", _DummyRegistry()))
    server_conn = _make_server_connection()

    async def _fake_launch_server(
        *,
        server_name: str,
        client_session_factory: ClientSessionFactory,
        startup_timeout_seconds: float | None = None,
        trigger_oauth: bool | None = None,
        oauth_event_handler: OAuthEventHandler | None = None,
        allow_oauth_paste_fallback: bool = True,
    ) -> ServerConnection:
        del server_name, client_session_factory, startup_timeout_seconds
        del trigger_oauth, oauth_event_handler, allow_oauth_paste_fallback
        manager.running_servers["demo"] = server_conn
        return server_conn

    monkeypatch.setattr(manager, "launch_server", _fake_launch_server)

    task = asyncio.create_task(
        manager.get_server(
            "demo",
            client_session_factory=_dummy_client_session_factory,
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
        transport_context_factory=HangingTransportContext,
        client_session_factory=_dummy_client_session_factory,
    )

    async def _fake_launch_server(
        *,
        server_name: str,
        client_session_factory: ClientSessionFactory,
        startup_timeout_seconds: float | None = None,
        trigger_oauth: bool | None = None,
        oauth_event_handler: OAuthEventHandler | None = None,
        allow_oauth_paste_fallback: bool = True,
    ) -> ServerConnection:
        del server_name, client_session_factory, startup_timeout_seconds
        del trigger_oauth, oauth_event_handler, allow_oauth_paste_fallback
        manager.running_servers["demo"] = server_conn
        asyncio.create_task(_server_lifecycle_task(server_conn))
        await entered.wait()
        return server_conn

    monkeypatch.setattr(manager, "launch_server", _fake_launch_server)

    with pytest.raises(ServerInitializationError):
        await manager.get_server(
            "demo",
            client_session_factory=_dummy_client_session_factory,
            startup_timeout_seconds=0.01,
        )

    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    assert "demo" not in manager.running_servers
    assert server_conn._shutdown_event.is_set()
    assert server_conn._oauth_abort_event.is_set()


@pytest.mark.asyncio
async def test_get_server_retries_with_oauth_after_401_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = MCPConnectionManager(server_registry=cast("Any", _DummyRegistry()))
    unhealthy = _make_server_connection()
    unhealthy._error_occurred = True
    unhealthy._error_message = "HTTP Error: 401 Unauthorized for URL: http://example.com/mcp"

    healthy = _make_server_connection()
    healthy.session = cast("Any", object())

    calls: list[bool | None] = []

    async def _fake_launch_and_wait_for_server(
        *,
        server_name: str,
        client_session_factory: ClientSessionFactory,
        startup_timeout_seconds: float | None,
        trigger_oauth: bool | None,
        oauth_event_handler: OAuthEventHandler | None,
        allow_oauth_paste_fallback: bool,
        timeout_action: str,
    ) -> ServerConnection:
        del server_name, client_session_factory, startup_timeout_seconds
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
        client_session_factory: ClientSessionFactory,
        startup_timeout_seconds: float | None,
        oauth_event_handler: OAuthEventHandler | None,
        allow_oauth_paste_fallback: bool,
        timeout_action: str,
    ) -> ServerConnection:
        del server_name, server_conn, client_session_factory, startup_timeout_seconds
        del oauth_event_handler, allow_oauth_paste_fallback, timeout_action
        calls.append(True)
        manager._server_oauth_mode["demo"] = "force"
        manager._server_oauth_active["demo"] = True
        return healthy

    monkeypatch.setattr(manager, "_launch_and_wait_for_server", _fake_launch_and_wait_for_server)
    monkeypatch.setattr(manager, "_retry_server_with_oauth", _fake_retry_server_with_oauth)

    server_conn = await manager.get_server(
        "demo",
        client_session_factory=_dummy_client_session_factory,
    )

    assert server_conn is healthy
    assert calls == [None, True]


@pytest.mark.asyncio
async def test_get_server_formats_stdio_missing_executable_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @asynccontextmanager
    async def _failing_stdio_client(*_args, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory")
        yield

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.tracking_stdio_client",
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
                client_session_factory=_dummy_client_session_factory,
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
    @asynccontextmanager
    async def _failing_stdio_client(*_args, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory")
        yield

    missing_cwd = str(tmp_path / "missing-dir")

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.tracking_stdio_client",
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
                client_session_factory=_dummy_client_session_factory,
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
        transport_context_factory=lambda: cast("Any", object()),
        client_session_factory=_dummy_client_session_factory,
    )
    server_conn.record_stdio_stderr("npm notice downloading desktop-commander")
    server_conn.record_stdio_stderr("npm warn request took longer than expected")

    async def _fake_launch_server(
        *,
        server_name: str,
        client_session_factory: ClientSessionFactory,
        startup_timeout_seconds: float | None = None,
        trigger_oauth: bool | None = None,
        oauth_event_handler: OAuthEventHandler | None = None,
        allow_oauth_paste_fallback: bool = True,
    ) -> ServerConnection:
        del server_name, client_session_factory, startup_timeout_seconds
        del trigger_oauth, oauth_event_handler, allow_oauth_paste_fallback
        manager.running_servers["demo"] = server_conn
        return server_conn

    monkeypatch.setattr(manager, "launch_server", _fake_launch_server)

    with pytest.raises(ServerInitializationError) as exc_info:
        await manager.get_server(
            "demo",
            client_session_factory=_dummy_client_session_factory,
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
    task_group = create_task_group()
    await task_group.__aenter__()
    manager._task_group_active = True
    manager._task_group = task_group
    manager._tg = task_group

    async def _unexpected_sleep(_delay: float) -> None:
        raise AssertionError("shutdown grace sleep should be skipped")

    monkeypatch.setattr(asyncio, "sleep", _unexpected_sleep)

    await manager.__aexit__(None, None, None)

    assert manager._task_group_active is False
    assert manager._task_group is None
    assert manager._tg is None


@pytest.mark.asyncio
async def test_connection_manager_exit_waits_briefly_after_requesting_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RunningServersManager(MCPConnectionManager):
        async def disconnect_all(self) -> bool:
            return True

    manager = _RunningServersManager(server_registry=cast("Any", _DummyRegistry()))
    task_group = create_task_group()
    await task_group.__aenter__()
    manager._task_group_active = True
    manager._task_group = task_group
    manager._tg = task_group
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    await manager.__aexit__(None, None, None)

    assert sleep_calls == [0.5]
    assert manager._task_group_active is False
    assert manager._task_group is None
    assert manager._tg is None


def test_is_oauth_timeout_message_requires_real_timeout_markers() -> None:
    assert _is_oauth_timeout_message("OAuth authorization timed out") is True
    assert _is_oauth_timeout_message("  OAUTH   AUTHORIZATION   TIMED   OUT  ") is True
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
    assert _is_oauth_registration_404_message(" OAUTH REGISTRATION FAILED: 404 ") is True
    assert _is_oauth_registration_404_message("HTTP Error: 404 Not Found for URL: /mcp") is False


def test_is_http_auth_challenge_error_detects_401_responses() -> None:
    assert _is_http_auth_challenge_error("HTTP Error: 401 Unauthorized for URL: /mcp") is True
    assert _is_http_auth_challenge_error("  HTTP   ERROR:   401   UNAUTHORIZED  ") is True
    assert _is_http_auth_challenge_error("401 Client Error: Unauthorized for url") is True
    assert _is_http_auth_challenge_error("WWW-Authenticate: Bearer realm=example") is True
    assert _is_http_auth_challenge_error("HTTP Error: 404 Not Found for URL: /mcp") is False


def test_format_oauth_registration_404_details_includes_copilot_hint() -> None:
    details = _format_oauth_registration_404_details(
        "OAuthRegistrationError: Registration failed: 404 404 page not found",
        "HTTPS://API.GITHUBCOPILOT.COM/mcp/",
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
