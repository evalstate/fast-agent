import sys
from pathlib import Path

import httpx2
import pytest

from fast_agent.config import MCPServerSettings
from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
from fast_agent.mcp.client_gateway import (
    MCPClientHooks,
    is_http_auth_challenge,
    open_request_scoped_client,
)
from fast_agent.mcp.gen_client import gen_client
from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager
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
