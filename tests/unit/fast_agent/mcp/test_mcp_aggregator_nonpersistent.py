from contextlib import asynccontextmanager

import pytest
from mcp.shared.exceptions import McpError
from mcp.types import (
    ErrorData,
    Implementation,
    InitializeResult,
    ListToolsResult,
    PromptsCapability,
    ServerCapabilities,
    Tool,
    ToolsCapability,
)

from fast_agent.config import MCPServerSettings
from fast_agent.context import Context
from fast_agent.mcp.gen_client import gen_client
from fast_agent.mcp.interfaces import ServerInitializerProtocol
from fast_agent.mcp.mcp_aggregator import (
    METHOD_NOT_FOUND_ERROR_CODE,
    MCPAggregator,
    _is_capability_probe_error,
)
from fast_agent.mcp_server_registry import ServerRegistry


def _build_context(configs: dict[str, MCPServerSettings]) -> Context:
    registry = ServerRegistry()
    registry.registry = configs
    return Context(server_registry=registry)


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
            capabilities=ServerCapabilities(tools=ToolsCapability()),
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

    def _fake_factory(read_stream, write_stream, read_timeout, **kwargs):
        return session

    async with registry.initialize_server(
        "demo", client_session_factory=_fake_factory
    ) as yielded_session:
        assert yielded_session is session
        assert session.initialized is True
        assert transport_entered is True

    assert session.closed is True
    assert transport_exited is True
    assert registry.get_server_capabilities("demo") is not None


@pytest.mark.asyncio
async def test_get_capabilities_nonpersistent_returns_real_capabilities(
    monkeypatch,
) -> None:
    context = _build_context(
        {"alpha": MCPServerSettings(name="alpha", transport="stdio", command="echo")}
    )

    expected_caps = ServerCapabilities(tools=ToolsCapability(), prompts=PromptsCapability())

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


@pytest.mark.asyncio
async def test_get_capabilities_nonpersistent_caches_result(monkeypatch) -> None:
    context = _build_context(
        {"alpha": MCPServerSettings(name="alpha", transport="stdio", command="echo")}
    )

    expected_caps = ServerCapabilities(tools=ToolsCapability())
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


@pytest.mark.asyncio
async def test_fetch_server_tools_returns_empty_for_not_implemented_error() -> None:
    context = _build_context({})

    class _NotImplAggregator(MCPAggregator):
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
            raise NotImplementedError("list_tools not supported")

    aggregator = _NotImplAggregator(
        server_names=["legacy"],
        connection_persistence=False,
        context=context,
    )

    tools = await aggregator._fetch_server_tools("legacy")
    assert tools == []


@pytest.mark.asyncio
async def test_fetch_server_tools_returns_empty_for_method_not_found_message() -> None:
    """McpError with 'method not found' in message (without -32601 code) degrades gracefully."""
    context = _build_context({})

    class _MsgOnlyAggregator(MCPAggregator):
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
            raise McpError(ErrorData(code=-32000, message="Method not found on this server"))

    aggregator = _MsgOnlyAggregator(
        server_names=["msg-only"],
        connection_persistence=False,
        context=context,
    )

    tools = await aggregator._fetch_server_tools("msg-only")
    assert tools == []


@pytest.mark.asyncio
async def test_fetch_server_tools_reraises_non_probe_mcp_error() -> None:
    """McpError that is NOT a capability probe (e.g. -32600 Invalid request) re-raises."""
    context = _build_context({})

    class _InvalidRequestAggregator(MCPAggregator):
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
            raise McpError(ErrorData(code=-32600, message="Invalid request"))

    aggregator = _InvalidRequestAggregator(
        server_names=["bad-req"],
        connection_persistence=False,
        context=context,
    )

    with pytest.raises(McpError):
        await aggregator._fetch_server_tools("bad-req")


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


class _DummyInitializer:
    """Stub implementing only ServerInitializerProtocol, no connection_manager."""

    @asynccontextmanager
    async def initialize_server(self, server_name, client_session_factory=None):
        session = _DummySession()
        yield session

    def get_server_capabilities(self, server_name):
        return None


@pytest.mark.asyncio
async def test_gen_client_accepts_initializer_protocol() -> None:
    stub = _DummyInitializer()
    assert isinstance(stub, ServerInitializerProtocol)

    async with gen_client("demo", server_registry=stub) as session:
        assert session is not None


def test_connect_requires_full_protocol() -> None:
    """ServerInitializerProtocol alone is not sufficient for connect/disconnect."""
    from fast_agent.mcp.interfaces import ServerRegistryProtocol

    stub = _DummyInitializer()

    assert isinstance(stub, ServerInitializerProtocol)
    assert not isinstance(stub, ServerRegistryProtocol)


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


def test_is_capability_probe_error_with_not_implemented_error() -> None:
    assert _is_capability_probe_error(NotImplementedError("not supported")) is True


def test_is_capability_probe_error_with_method_not_found_code() -> None:
    exc = McpError(ErrorData(code=METHOD_NOT_FOUND_ERROR_CODE, message="Method not found"))
    assert _is_capability_probe_error(exc) is True


def test_is_capability_probe_error_with_method_not_found_message() -> None:
    exc = McpError(ErrorData(code=-32000, message="Method not found on server"))
    assert _is_capability_probe_error(exc) is True


def test_is_capability_probe_error_rejects_infrastructure_errors() -> None:
    assert _is_capability_probe_error(RuntimeError("connection lost")) is False
    assert _is_capability_probe_error(AttributeError("no such attr")) is False
    exc = McpError(ErrorData(code=-32600, message="Invalid request"))
    assert _is_capability_probe_error(exc) is False


@pytest.mark.asyncio
async def test_detach_server_clears_capabilities_cache(monkeypatch) -> None:
    context = _build_context(
        {"alpha": MCPServerSettings(name="alpha", transport="stdio", command="echo")}
    )

    expected_caps = ServerCapabilities(tools=ToolsCapability())

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

    # Simulate that the server was attached (normally done by load_servers)
    aggregator._attached_server_names.append("alpha")
    await aggregator.detach_server("alpha")

    assert aggregator._capabilities_cache.get("alpha") is None


@pytest.mark.asyncio
async def test_reset_runtime_indexes_clears_capabilities_cache() -> None:
    context = _build_context({})

    aggregator = MCPAggregator(
        server_names=[],
        connection_persistence=False,
        context=context,
    )

    # Manually populate the cache
    aggregator._capabilities_cache["alpha"] = ServerCapabilities(tools=ToolsCapability())
    assert aggregator._capabilities_cache.get("alpha") is not None

    await aggregator._reset_runtime_indexes()

    assert aggregator._capabilities_cache.get("alpha") is None
