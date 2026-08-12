import json
import sys
from pathlib import Path

import httpx2
import pytest

from fast_agent.config import MCPServerSettings
from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
from fast_agent.mcp.client_gateway import (
    MCPClientHooks,
    _http_diagnostic_hooks,
    is_http_auth_challenge,
    open_request_scoped_client,
)
from fast_agent.mcp.gen_client import gen_client
from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager
from fast_agent.mcp.transport_tracking import TransportChannelMetrics
from fast_agent.mcp_server_registry import ServerRegistry


def test_auth_challenge_classifier_walks_exception_groups_and_causes() -> None:
    request = httpx2.Request("POST", "https://example.com/mcp")
    challenge = httpx2.HTTPStatusError(
        "opaque SDK failure",
        request=request,
        response=httpx2.Response(401, request=request),
    )
    wrapper = RuntimeError("client startup failed")
    wrapper.__cause__ = ExceptionGroup("transport cleanup", [ValueError("closed"), challenge])

    assert is_http_auth_challenge(wrapper)
    assert not is_http_auth_challenge(RuntimeError("request failed"))
    assert is_http_auth_challenge(
        RuntimeError("request failed"),
        response_challenged=True,
    )


@pytest.mark.asyncio
async def test_http_diagnostic_hooks_track_post_get_and_resumption() -> None:
    metrics = TransportChannelMetrics()
    hooks = _http_diagnostic_hooks(
        "docs",
        MCPClientHooks(transport_metrics=metrics),
    )
    assert hooks is not None

    async def server(request: httpx2.Request) -> httpx2.Response:
        if request.content == b"not-json":
            return httpx2.Response(400, request=request)
        if request.method == "POST":
            payload = json.loads(request.content)
            assert isinstance(payload, dict)
            content_type = (
                "text/event-stream; charset=utf-8"
                if payload["method"] in {"tools/call", "subscriptions/listen"}
                else "application/json; charset=utf-8"
            )
            return httpx2.Response(
                200,
                headers={"content-type": content_type},
                request=request,
            )
        return httpx2.Response(
            405,
            request=request,
        )

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(server),
        event_hooks=hooks,
    ) as client:
        await client.post(
            "https://example.com/mcp",
            headers={"Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {},
            },
        )
        await client.post(
            "https://example.com/mcp",
            headers={"Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "generate_image", "arguments": {}},
            },
        )
        await client.post(
            "https://example.com/mcp",
            headers={"Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": "listen-1",
                "method": "subscriptions/listen",
                "params": {"notifications": {"toolsListChanged": True}},
            },
        )
        await client.get(
            "https://example.com/mcp",
            headers={"Accept": "text/event-stream", "Last-Event-ID": "event-7"},
        )
        await client.post(
            "https://example.com/mcp",
            headers={"Accept": "application/json"},
            content=b"not-json",
        )

    snapshot = metrics.snapshot()
    assert snapshot.post_json is not None
    assert snapshot.post_json.request_count == 1
    assert snapshot.post_json.state == "error"
    assert snapshot.post_json.last_error == "HTTP 400"
    assert snapshot.post_sse is not None
    assert snapshot.post_sse.request_count == 1
    assert snapshot.listen is not None
    assert snapshot.listen.request_count == 1
    assert snapshot.get is not None
    assert snapshot.get.last_status_code == 405
    assert snapshot.resumption is not None
    assert snapshot.resumption.request_count == 1
    assert snapshot.resumption.last_message_summary == "event-7"


@pytest.mark.asyncio
async def test_http_discover_400_is_discovery_not_post_error() -> None:
    metrics = TransportChannelMetrics()
    hooks = _http_diagnostic_hooks(
        "docs",
        MCPClientHooks(transport_metrics=metrics),
    )
    assert hooks is not None

    async def server(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, request=request)

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(server),
        event_hooks=hooks,
    ) as client:
        await client.post(
            "https://example.com/mcp",
            headers={"Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "server/discover",
                "params": {},
            },
        )

    snapshot = metrics.snapshot()
    assert snapshot.discovery is not None
    assert snapshot.discovery.state == "failed"
    assert snapshot.discovery.status_code == 400
    assert snapshot.discovery.detail == "HTTP 400"
    assert snapshot.post_json is not None
    assert snapshot.post_json.request_count == 1
    assert snapshot.post_json.state != "error"
    assert snapshot.post_json.last_error is None


@pytest.mark.asyncio
async def test_http_diagnostic_hooks_classify_only_final_redirect_response() -> None:
    metrics = TransportChannelMetrics()
    hooks = _http_diagnostic_hooks(
        "docs",
        MCPClientHooks(transport_metrics=metrics),
    )
    assert hooks is not None

    async def server(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/mcp":
            return httpx2.Response(
                307,
                headers={"location": "/tools"},
                request=request,
            )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(server),
        event_hooks=hooks,
        follow_redirects=True,
    ) as client:
        await client.post(
            "https://example.com/mcp",
            headers={"Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "generate_image", "arguments": {}},
            },
        )

    snapshot = metrics.snapshot()
    assert snapshot.post_json is None
    assert snapshot.post_sse is not None
    assert snapshot.post_sse.request_count == 1


@pytest.mark.asyncio
async def test_request_startup_is_plain_first_and_escalates_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MCPServerSettings(
        name="demo",
        transport="http",
        url="https://example.com/mcp",
    )
    callbacks = MCPClientCallbackRuntime(server_name="demo", server_config=config)
    attempts: list[bool] = []
    oauth_failure = RuntimeError("OAuth startup failed")

    class _Connection:
        def __init__(self, oauth_active: bool, hooks: MCPClientHooks) -> None:
            self.oauth_active = oauth_active
            self.hooks = hooks

        async def __aenter__(self):
            attempts.append(self.oauth_active)
            if not self.oauth_active:
                handler = self.hooks.http_response_handler
                assert handler is not None
                await handler(
                    httpx2.Response(
                        401,
                        request=httpx2.Request("POST", config.url or ""),
                    )
                )
                raise RuntimeError("opaque SDK failure")
            raise oauth_failure

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            raise AssertionError("A failed startup must not be exited twice")

    def _create_client_connection(*, oauth_active, hooks, **kwargs):
        del kwargs
        return _Connection(oauth_active, hooks)

    monkeypatch.setattr(
        "fast_agent.mcp.client_gateway.create_client_connection",
        _create_client_connection,
    )

    with pytest.raises(RuntimeError) as raised:
        async with open_request_scoped_client(
            server_name="demo",
            config=config,
            callback_runtime=callbacks,
        ):
            raise AssertionError("startup did not fail")

    assert raised.value is oauth_failure
    assert attempts == [False, True]


@pytest.mark.asyncio
async def test_request_startup_does_not_replace_explicit_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MCPServerSettings(
        name="demo",
        transport="http",
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer user-token"},
    )
    callbacks = MCPClientCallbackRuntime(server_name="demo", server_config=config)
    attempts: list[bool] = []

    class _Connection:
        async def __aenter__(self):
            attempts.append(False)
            raise RuntimeError("401 Unauthorized")

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            raise AssertionError("A failed startup must not be exited")

    def _create_client_connection(**kwargs):
        assert kwargs["oauth_active"] is False
        return _Connection()

    monkeypatch.setattr(
        "fast_agent.mcp.client_gateway.create_client_connection",
        _create_client_connection,
    )

    with pytest.raises(RuntimeError, match="401 Unauthorized"):
        async with open_request_scoped_client(
            server_name="demo",
            config=config,
            callback_runtime=callbacks,
        ):
            raise AssertionError("startup did not fail")

    assert attempts == [False]


@pytest.mark.asyncio
async def test_gen_client_uses_real_stdio_simulator() -> None:
    server = (
        Path(__file__).parents[3]
        / "integration"
        / "server_instructions"
        / "server_without_instructions.py"
    )
    config = MCPServerSettings(
        name="simulator",
        transport="stdio",
        command=sys.executable,
        args=[str(server)],
    )
    registry = ServerRegistry()
    registry.register_central("simulator", config)

    async with gen_client("simulator", server_registry=registry) as client:
        result = await client.list_tools()

    assert {tool.name for tool in result.tools} >= {"echo", "ping"}
    assert registry.get_server_capabilities("simulator") is not None


@pytest.mark.asyncio
async def test_gen_client_uses_attachment_local_config_without_publishing_capabilities() -> None:
    server = (
        Path(__file__).parents[3]
        / "integration"
        / "server_instructions"
        / "server_without_instructions.py"
    )
    registry = ServerRegistry()
    registry.register_central(
        "simulator",
        MCPServerSettings(
            transport="stdio",
            command="missing-registry-command",
        ),
    )
    override = MCPServerSettings(
        transport="stdio",
        command=sys.executable,
        args=[str(server)],
    )

    async with gen_client(
        "simulator",
        server_registry=registry,
        server_config=override,
        publish_capabilities=False,
    ) as client:
        result = await client.list_tools()

    assert {tool.name for tool in result.tools} >= {"echo", "ping"}
    stored = registry.get_server_config("simulator")
    assert stored is not None
    assert stored.command == "missing-registry-command"
    assert registry.get_server_capabilities("simulator") is None


@pytest.mark.asyncio
async def test_connection_manager_uses_attachment_local_config() -> None:
    server = (
        Path(__file__).parents[3]
        / "integration"
        / "server_instructions"
        / "server_without_instructions.py"
    )
    registry = ServerRegistry()
    registry.register_central(
        "simulator",
        MCPServerSettings(transport="stdio", command="missing-registry-command"),
    )
    override = MCPServerSettings(
        transport="stdio",
        command=sys.executable,
        args=[str(server)],
    )
    callbacks = MCPClientCallbackRuntime(
        server_name="simulator",
        server_config=override,
    )

    async with MCPConnectionManager(registry) as manager:
        connection = await manager.get_server(
            "simulator",
            server_config=override,
            callback_runtime=callbacks,
        )
        assert connection.client is not None
        result = await connection.client.list_tools()

    assert {tool.name for tool in result.tools} >= {"echo", "ping"}
    stored = registry.get_server_config("simulator")
    assert stored is not None
    assert stored.command == "missing-registry-command"
