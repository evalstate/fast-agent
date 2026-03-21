from contextlib import asynccontextmanager

import pytest
from mcp.shared.exceptions import McpError
from mcp.types import (
    ErrorData,
    Implementation,
    InitializeResult,
    ListToolsResult,
    ServerCapabilities,
    Tool,
)

from fast_agent.config import MCPServerSettings
from fast_agent.context import Context
from fast_agent.mcp.gen_client import gen_client
from fast_agent.mcp.interfaces import ServerInitializerProtocol
from fast_agent.mcp.mcp_aggregator import METHOD_NOT_FOUND_ERROR_CODE, MCPAggregator
from fast_agent.mcp_server_registry import ServerRegistry


def _build_context(configs: dict[str, MCPServerSettings]) -> Context:
    registry = ServerRegistry()
    registry.registry = configs
    return Context(server_registry=registry)


# ---------------------------------------------------------------------------
# Test 1: initialize_server creates and tears down session
# ---------------------------------------------------------------------------


class _DummySession:
    """Minimal stub that records initialize() calls."""

    def __init__(self) -> None:
        self.initialized = False
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.closed = True
        return None

    async def initialize(self):
        self.initialized = True
        return InitializeResult(
            protocolVersion="2025-03-26",
            capabilities=ServerCapabilities(tools={}),
            serverInfo=Implementation(name="stub", version="0.1"),
        )


@pytest.mark.asyncio
async def test_initialize_server_creates_and_tears_down_session(monkeypatch) -> None:
    registry = ServerRegistry()
    registry.registry = {
        "demo": MCPServerSettings(name="demo", transport="stdio", command="echo"),
    }

    session = _DummySession()
    transport_entered = False
    transport_exited = False

    @asynccontextmanager
    async def _fake_transport(server_name, config):
        nonlocal transport_entered, transport_exited
        transport_entered = True
        yield (object(), object(), None)
        transport_exited = True

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.create_transport_context",
        _fake_transport,
    )

    def _fake_factory(read_stream, write_stream, read_timeout):
        return session

    async with registry.initialize_server(
        "demo", client_session_factory=_fake_factory
    ) as yielded_session:
        assert yielded_session is session
        assert session.initialized is True
        assert transport_entered is True

    assert session.closed is True
    assert transport_exited is True
    assert "demo" in registry._init_results
    assert registry._init_results["demo"].capabilities is not None


# ---------------------------------------------------------------------------
# Test 2: get_capabilities returns real capabilities in non-persistent mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_capabilities_nonpersistent_returns_real_capabilities(
    monkeypatch,
) -> None:
    context = _build_context(
        {"alpha": MCPServerSettings(name="alpha", transport="stdio", command="echo")}
    )

    expected_caps = ServerCapabilities(tools={}, prompts={})

    @asynccontextmanager
    async def _fake_initialize_server(self, server_name, client_session_factory=None):
        self._init_results[server_name] = InitializeResult(
            protocolVersion="2025-03-26",
            capabilities=expected_caps,
            serverInfo=Implementation(name="stub", version="0.1"),
        )
        yield _DummySession()

    monkeypatch.setattr(
        ServerRegistry,
        "initialize_server",
        _fake_initialize_server,
    )

    aggregator = MCPAggregator(
        server_names=["alpha"],
        connection_persistence=False,
        context=context,
    )

    caps = await aggregator.get_capabilities("alpha")
    assert caps is expected_caps


# ---------------------------------------------------------------------------
# Test 3: get_capabilities caches result (second call does not reconnect)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_capabilities_nonpersistent_caches_result(monkeypatch) -> None:
    context = _build_context(
        {"alpha": MCPServerSettings(name="alpha", transport="stdio", command="echo")}
    )

    expected_caps = ServerCapabilities(tools={})
    init_count = 0

    @asynccontextmanager
    async def _counting_initialize(self, server_name, client_session_factory=None):
        nonlocal init_count
        init_count += 1
        self._init_results[server_name] = InitializeResult(
            protocolVersion="2025-03-26",
            capabilities=expected_caps,
            serverInfo=Implementation(name="stub", version="0.1"),
        )
        yield _DummySession()

    monkeypatch.setattr(
        ServerRegistry,
        "initialize_server",
        _counting_initialize,
    )

    aggregator = MCPAggregator(
        server_names=["alpha"],
        connection_persistence=False,
        context=context,
    )

    caps1 = await aggregator.get_capabilities("alpha")
    caps2 = await aggregator.get_capabilities("alpha")

    assert caps1 is expected_caps
    assert caps2 is expected_caps
    assert init_count == 1


# ---------------------------------------------------------------------------
# Test 4: _fetch_server_tools re-raises infrastructure error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_server_tools_reraises_infrastructure_error() -> None:
    context = _build_context({})

    class _InfraErrorAggregator(MCPAggregator):
        async def server_supports_feature(self, server_name, feature):
            return False

        async def _execute_on_server(
            self,
            server_name,
            operation_type,
            operation_name,
            method_name,
            method_args=None,
            error_factory=None,
            progress_callback=None,
        ):
            raise AttributeError("broken transport")

    aggregator = _InfraErrorAggregator(
        server_names=["broken"],
        connection_persistence=False,
        context=context,
    )

    with pytest.raises(AttributeError, match="broken transport"):
        await aggregator._fetch_server_tools("broken")


# ---------------------------------------------------------------------------
# Test 5: _fetch_server_tools returns empty for McpError on optimistic probe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_server_tools_returns_empty_for_mcp_error() -> None:
    context = _build_context({})

    class _McpErrorAggregator(MCPAggregator):
        async def server_supports_feature(self, server_name, feature):
            return False

        async def _execute_on_server(
            self,
            server_name,
            operation_type,
            operation_name,
            method_name,
            method_args=None,
            error_factory=None,
            progress_callback=None,
        ):
            raise McpError(
                ErrorData(
                    code=METHOD_NOT_FOUND_ERROR_CODE,
                    message="Method not found",
                )
            )

    aggregator = _McpErrorAggregator(
        server_names=["no-tools"],
        connection_persistence=False,
        context=context,
    )

    tools = await aggregator._fetch_server_tools("no-tools")
    assert tools == []


# ---------------------------------------------------------------------------
# Test 6: _fetch_server_tools returns tools on success (non-persistent)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_server_tools_nonpersistent_success() -> None:
    context = _build_context({})

    class _ToolsAggregator(MCPAggregator):
        async def server_supports_feature(self, server_name, feature):
            return True

        async def _execute_on_server(
            self,
            server_name,
            operation_type,
            operation_name,
            method_name,
            method_args=None,
            error_factory=None,
            progress_callback=None,
        ):
            return ListToolsResult(
                tools=[
                    Tool(name="read_file", inputSchema={"type": "object"}),
                    Tool(name="write_file", inputSchema={"type": "object"}),
                ]
            )

    aggregator = _ToolsAggregator(
        server_names=["fs"],
        connection_persistence=False,
        context=context,
    )

    tools = await aggregator._fetch_server_tools("fs")
    assert [t.name for t in tools] == ["read_file", "write_file"]


# ---------------------------------------------------------------------------
# Test 7: gen_client accepts ServerInitializerProtocol (narrow protocol)
# ---------------------------------------------------------------------------


class _DummyInitializer:
    """Stub implementing only ServerInitializerProtocol, no connection_manager."""

    @asynccontextmanager
    async def initialize_server(self, server_name, client_session_factory=None):
        session = _DummySession()
        yield session


@pytest.mark.asyncio
async def test_gen_client_accepts_initializer_protocol() -> None:
    stub = _DummyInitializer()
    assert isinstance(stub, ServerInitializerProtocol)

    async with gen_client("demo", server_registry=stub) as session:
        assert session is not None


# ---------------------------------------------------------------------------
# Test 8: connect/disconnect still require full ServerRegistryProtocol
# ---------------------------------------------------------------------------


def test_connect_requires_full_protocol() -> None:
    """ServerInitializerProtocol alone is not sufficient for connect/disconnect."""
    from fast_agent.mcp.interfaces import ServerRegistryProtocol

    stub = _DummyInitializer()

    # The narrow protocol should satisfy ServerInitializerProtocol
    assert isinstance(stub, ServerInitializerProtocol)
    # But NOT ServerRegistryProtocol (missing connection_manager, registry, etc.)
    assert not isinstance(stub, ServerRegistryProtocol)


# ---------------------------------------------------------------------------
# Test 9: _fetch_server_tools re-raises McpError when server advertised tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_server_tools_reraises_mcp_error_when_tools_advertised() -> None:
    context = _build_context({})

    class _AdvertisedButBrokenAggregator(MCPAggregator):
        async def server_supports_feature(self, server_name, feature):
            return True

        async def _execute_on_server(
            self,
            server_name,
            operation_type,
            operation_name,
            method_name,
            method_args=None,
            error_factory=None,
            progress_callback=None,
        ):
            raise McpError(ErrorData(code=-32600, message="Invalid request"))

    aggregator = _AdvertisedButBrokenAggregator(
        server_names=["broken"],
        connection_persistence=False,
        context=context,
    )

    with pytest.raises(McpError):
        await aggregator._fetch_server_tools("broken")
