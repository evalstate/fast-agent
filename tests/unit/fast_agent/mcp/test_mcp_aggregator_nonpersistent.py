from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import cast

import pytest
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from fast_agent.config import MCPServerSettings
from fast_agent.context import Context
from fast_agent.mcp.gen_client import gen_client
from fast_agent.mcp.interfaces import ServerInitializerProtocol
from fast_agent.mcp.mcp_aggregator import MCPAggregator
from fast_agent.mcp_server_registry import ServerRegistry


def _build_context(configs: dict[str, MCPServerSettings]) -> Context:
    registry = ServerRegistry()
    registry.registry = configs
    return Context(server_registry=registry)


def _stdio_server(name: str) -> MCPServerSettings:
    return MCPServerSettings(name=name, transport="stdio", command="echo")


@pytest.mark.asyncio
async def test_gen_client_accepts_initializer_protocol() -> None:
    session = object()

    class StubRegistry:
        @asynccontextmanager
        async def initialize_server(self, server_name: str, client_session_factory=None):
            del client_session_factory
            assert server_name == "alpha"
            yield session

    registry = StubRegistry()

    assert isinstance(registry, ServerInitializerProtocol)

    async with gen_client("alpha", registry) as client:
        assert client is session


@pytest.mark.asyncio
async def test_server_registry_initialize_server_initializes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ServerRegistry()
    registry.registry = {"alpha": _stdio_server("alpha")}
    sentinel_capabilities = object()
    init_calls: list[str] = []

    @asynccontextmanager
    async def fake_transport_context(**kwargs):
        assert kwargs["server_name"] == "alpha"
        yield object(), object(), None

    class DummySession:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        async def initialize(self):
            init_calls.append("initialize")
            return SimpleNamespace(capabilities=sentinel_capabilities)

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.create_transport_context",
        fake_transport_context,
    )

    def session_factory(*args, **kwargs) -> ClientSession:
        return cast("ClientSession", DummySession(*args, **kwargs))

    async with registry.initialize_server("alpha", session_factory) as session:
        assert isinstance(session, DummySession)
        assert session.kwargs["server_config"].name == "alpha"

    assert init_calls == ["initialize"]
    capabilities = await registry.get_server_capabilities("alpha", session_factory)
    assert capabilities is sentinel_capabilities


@pytest.mark.asyncio
async def test_get_capabilities_nonpersistent_caches_result() -> None:
    context = _build_context({"alpha": _stdio_server("alpha")})
    aggregator = MCPAggregator(
        server_names=["alpha"],
        connection_persistence=False,
        context=context,
    )
    sentinel_capabilities = object()
    calls: list[str] = []

    async def fake_get_server_capabilities(server_name: str, client_session_factory=None):
        del client_session_factory
        calls.append(server_name)
        return sentinel_capabilities

    context.server_registry.get_server_capabilities = fake_get_server_capabilities  # type: ignore[method-assign]

    assert await aggregator.get_capabilities("alpha") is sentinel_capabilities
    assert await aggregator.get_capabilities("alpha") is sentinel_capabilities
    assert calls == ["alpha"]


@pytest.mark.asyncio
async def test_fetch_server_tools_re_raises_infrastructure_failure() -> None:
    context = _build_context({"alpha": _stdio_server("alpha")})

    class InfraAggregator(MCPAggregator):
        async def server_supports_feature(self, server_name: str, feature: str) -> bool:
            del server_name, feature
            return False

        async def _execute_on_server(
            self,
            server_name: str,
            operation_type: str,
            operation_name: str,
            method_name: str,
            method_args=None,
            error_factory=None,
            progress_callback=None,
        ):
            del (
                server_name,
                operation_type,
                operation_name,
                method_name,
                method_args,
                error_factory,
                progress_callback,
            )
            raise AttributeError("'ServerRegistry' object has no attribute 'initialize_server'")

    aggregator = InfraAggregator(
        server_names=["alpha"],
        connection_persistence=False,
        context=context,
    )

    with pytest.raises(AttributeError):
        await aggregator._fetch_server_tools("alpha")


@pytest.mark.asyncio
async def test_fetch_server_tools_returns_empty_for_method_not_found() -> None:
    context = _build_context({"alpha": _stdio_server("alpha")})

    class NoToolsAggregator(MCPAggregator):
        async def server_supports_feature(self, server_name: str, feature: str) -> bool:
            del server_name, feature
            return False

        async def _execute_on_server(
            self,
            server_name: str,
            operation_type: str,
            operation_name: str,
            method_name: str,
            method_args=None,
            error_factory=None,
            progress_callback=None,
        ):
            del (
                server_name,
                operation_type,
                operation_name,
                method_name,
                method_args,
                error_factory,
                progress_callback,
            )
            raise McpError(ErrorData(code=-32601, message="Method not found"))

    aggregator = NoToolsAggregator(
        server_names=["alpha"],
        connection_persistence=False,
        context=context,
    )

    assert await aggregator._fetch_server_tools("alpha") == []
