from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

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
    MCPAttachOptions,
    _is_capability_probe_error,
)
from fast_agent.mcp.skybridge import SkybridgeServerConfig
from fast_agent.mcp_server_registry import ServerRegistry

if TYPE_CHECKING:
    from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager


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


def _make_stub_aggregator(
    context: Context,
    server_name: str,
    *,
    supports_tools: bool = False,
    execute_result: object | None = None,
    execute_error: Exception | None = None,
) -> MCPAggregator:
    """Create a stub aggregator with configurable _execute_on_server behavior."""

    class _Stub(MCPAggregator):
        async def server_supports_feature(self, server_name, feature):
            return supports_tools

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
            if execute_error is not None:
                raise execute_error
            return execute_result

    return _Stub(
        server_names=[server_name],
        connection_persistence=False,
        context=context,
    )


# ---------------------------------------------------------------------------
# ServerRegistry.initialize_server
# ---------------------------------------------------------------------------


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
    async def _fake_transport(server_name, config, trigger_oauth=None):
        del trigger_oauth
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
async def test_handle_auth_challenge_reports_retry_failure() -> None:
    class _FailingManager:
        async def reconnect_server(self, server_name, client_session_factory, trigger_oauth=None):
            del server_name, client_session_factory, trigger_oauth
            raise RuntimeError("OAuth callback timed out")

    aggregator = MCPAggregator(
        server_names=[],
        connection_persistence=True,
        context=None,
    )
    aggregator._persistent_connection_manager = cast("MCPConnectionManager", _FailingManager())

    async def _try_execute(session) -> None:
        del session
        raise AssertionError("try_execute should not run when reconnect fails")

    result, success = await aggregator._handle_auth_challenge(
        "alpha",
        _try_execute,
        lambda message: message,
        RuntimeError("401 Unauthorized"),
    )

    assert success is False
    assert result == "OAuth callback timed out"


@pytest.mark.asyncio
async def test_initialize_server_forwards_server_config_to_custom_factory(monkeypatch) -> None:
    registry = ServerRegistry()
    server_config = MCPServerSettings(name="demo", transport="stdio", command="echo")
    registry.registry = {"demo": server_config}

    session = _DummySession()
    captured_server_config = None

    @asynccontextmanager
    async def _fake_transport(server_name, config, trigger_oauth=None):
        del trigger_oauth
        yield (object(), object(), None)

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.create_transport_context",
        _fake_transport,
    )

    def _fake_factory(read_stream, write_stream, read_timeout, **kwargs):
        del read_stream, write_stream, read_timeout
        nonlocal captured_server_config
        captured_server_config = kwargs.get("server_config")
        return session

    async with registry.initialize_server(
        "demo", client_session_factory=_fake_factory
    ) as yielded_session:
        assert yielded_session is session

    assert captured_server_config is server_config


@pytest.mark.asyncio
async def test_initialize_server_retries_with_oauth_after_401(monkeypatch) -> None:
    registry = ServerRegistry()
    registry.registry = {
        "demo": MCPServerSettings(name="demo", transport="http", url="https://example.com"),
    }

    trigger_history: list[bool | None] = []

    @asynccontextmanager
    async def _fake_transport(server_name, config, trigger_oauth=None):
        del server_name, config
        trigger_history.append(trigger_oauth)
        yield (object(), object(), None)

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.create_transport_context",
        _fake_transport,
    )

    class _ChallengeSession(_DummySession):
        def __init__(self, oauth_enabled: bool) -> None:
            super().__init__()
            self._oauth_enabled = oauth_enabled

        async def initialize(self):
            if not self._oauth_enabled:
                raise RuntimeError("401 Unauthorized")
            return await super().initialize()

    def _fake_factory(read_stream, write_stream, read_timeout, **kwargs):
        del read_stream, write_stream, read_timeout, kwargs
        return _ChallengeSession(oauth_enabled=bool(trigger_history[-1]))

    async with registry.initialize_server(
        "demo", client_session_factory=_fake_factory
    ) as session:
        assert isinstance(session, _ChallengeSession)
        assert session.initialized is True

    assert trigger_history == [False, True]


@pytest.mark.asyncio
async def test_execute_on_server_nonpersistent_retries_with_oauth_after_401(
    monkeypatch,
) -> None:
    context = _build_context(
        {
            "alpha": MCPServerSettings(
                name="alpha", transport="http", url="https://example.com"
            )
        }
    )
    aggregator = MCPAggregator(
        server_names=["alpha"],
        connection_persistence=False,
        context=context,
    )

    trigger_history: list[bool | None] = []

    class _RetryClient:
        def __init__(self, trigger_oauth: bool | None) -> None:
            self._trigger_oauth = trigger_oauth

        async def list_tools(self) -> ListToolsResult:
            if self._trigger_oauth is not True:
                raise RuntimeError("401 Unauthorized")
            return ListToolsResult(tools=[Tool(name="echo", inputSchema={"type": "object"})])

    @asynccontextmanager
    async def _fake_gen_client(
        server_name,
        server_registry,
        client_session_factory=_DummySession,
        *,
        trigger_oauth=None,
    ):
        del server_name, server_registry, client_session_factory
        trigger_history.append(trigger_oauth)
        yield _RetryClient(trigger_oauth)

    monkeypatch.setattr("fast_agent.mcp.mcp_aggregator.gen_client", _fake_gen_client)

    result = await aggregator._execute_on_server(
        "alpha",
        "tools/list",
        "",
        "list_tools",
    )

    assert isinstance(result, ListToolsResult)
    assert [tool.name for tool in result.tools] == ["echo"]
    assert trigger_history == [None, True]


# ---------------------------------------------------------------------------
# get_capabilities (non-persistent path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_capabilities_nonpersistent_returns_real_capabilities(
    monkeypatch,
) -> None:
    context = _build_context(
        {"alpha": MCPServerSettings(name="alpha", transport="stdio", command="echo")}
    )

    expected_caps = ServerCapabilities(tools=ToolsCapability(), prompts=PromptsCapability())

    @asynccontextmanager
    async def _fake_initialize_server(self, server_name, client_session_factory=None, trigger_oauth=None):
        del trigger_oauth
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
    async def _counting_initialize(self, server_name, client_session_factory=None, trigger_oauth=None):
        del trigger_oauth
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
async def test_get_capabilities_returns_none_when_initialize_raises(monkeypatch) -> None:
    """get_capabilities degrades gracefully when initialize_server raises."""
    context = _build_context(
        {"broken": MCPServerSettings(name="broken", transport="stdio", command="echo")}
    )

    @asynccontextmanager
    async def _exploding_initialize(self, server_name, client_session_factory=None, trigger_oauth=None):
        del trigger_oauth
        raise RuntimeError("server crashed on startup")
        yield  # pragma: no cover — makes this a valid async generator

    monkeypatch.setattr(
        ServerRegistry,
        "initialize_server",
        _exploding_initialize,
    )

    aggregator = MCPAggregator(
        server_names=["broken"],
        connection_persistence=False,
        context=context,
    )

    result = await aggregator.get_capabilities("broken")
    assert result is None
    assert "broken" not in aggregator._capabilities_cache


# ---------------------------------------------------------------------------
# _fetch_server_tools — error propagation
# ---------------------------------------------------------------------------


def _make_mcp_error_none_code(message: str) -> McpError:
    """Build an McpError whose error code is None (simulates servers that omit it)."""
    error_data = ErrorData(code=-1, message=message)
    error_data.code = None  # type: ignore[assignment]
    return McpError(error_data)


@pytest.mark.asyncio
async def test_fetch_server_tools_reraises_infrastructure_error() -> None:
    aggregator = _make_stub_aggregator(
        _build_context({}),
        "broken",
        execute_error=AttributeError("broken transport"),
    )
    with pytest.raises(AttributeError, match="broken transport"):
        await aggregator._fetch_server_tools("broken")


@pytest.mark.asyncio
async def test_fetch_server_tools_returns_empty_for_mcp_error() -> None:
    aggregator = _make_stub_aggregator(
        _build_context({}),
        "no-tools",
        execute_error=McpError(
            ErrorData(code=METHOD_NOT_FOUND_ERROR_CODE, message="Method not found")
        ),
    )
    tools = await aggregator._fetch_server_tools("no-tools")
    assert tools == []


@pytest.mark.asyncio
async def test_fetch_server_tools_returns_empty_for_not_implemented_error() -> None:
    aggregator = _make_stub_aggregator(
        _build_context({}),
        "legacy",
        execute_error=NotImplementedError("list_tools not supported"),
    )
    tools = await aggregator._fetch_server_tools("legacy")
    assert tools == []


@pytest.mark.asyncio
async def test_fetch_server_tools_returns_empty_for_method_not_found_message() -> None:
    """McpError with 'method not found' in message (without -32601 code) degrades gracefully."""
    aggregator = _make_stub_aggregator(
        _build_context({}),
        "msg-only",
        execute_error=_make_mcp_error_none_code("Method not found on this server"),
    )
    tools = await aggregator._fetch_server_tools("msg-only")
    assert tools == []


@pytest.mark.asyncio
async def test_fetch_server_tools_reraises_non_probe_mcp_error() -> None:
    """McpError that is NOT a capability probe (e.g. -32600 Invalid request) re-raises."""
    aggregator = _make_stub_aggregator(
        _build_context({}),
        "bad-req",
        execute_error=McpError(ErrorData(code=-32600, message="Invalid request")),
    )
    with pytest.raises(McpError):
        await aggregator._fetch_server_tools("bad-req")


@pytest.mark.asyncio
async def test_fetch_server_tools_nonpersistent_success() -> None:
    aggregator = _make_stub_aggregator(
        _build_context({}),
        "fs",
        supports_tools=True,
        execute_result=ListToolsResult(
            tools=[
                Tool(name="read_file", inputSchema={"type": "object"}),
                Tool(name="write_file", inputSchema={"type": "object"}),
            ]
        ),
    )
    tools = await aggregator._fetch_server_tools("fs")
    assert [t.name for t in tools] == ["read_file", "write_file"]


@pytest.mark.asyncio
async def test_fetch_server_tools_reraises_mcp_error_when_tools_advertised() -> None:
    aggregator = _make_stub_aggregator(
        _build_context({}),
        "broken",
        supports_tools=True,
        execute_error=McpError(ErrorData(code=-32600, message="Invalid request")),
    )
    with pytest.raises(McpError):
        await aggregator._fetch_server_tools("broken")


# ---------------------------------------------------------------------------
# gen_client protocol boundary
# ---------------------------------------------------------------------------


class _DummyInitializer:
    """Stub implementing only ServerInitializerProtocol, no connection_manager."""

    @asynccontextmanager
    async def initialize_server(self, server_name, client_session_factory=None, trigger_oauth=None):
        del trigger_oauth
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


# ---------------------------------------------------------------------------
# _is_capability_probe_error
# ---------------------------------------------------------------------------


def test_is_capability_probe_error_with_not_implemented_error() -> None:
    assert _is_capability_probe_error(NotImplementedError("not supported")) is True


def test_is_capability_probe_error_with_method_not_found_code() -> None:
    exc = McpError(ErrorData(code=METHOD_NOT_FOUND_ERROR_CODE, message="Method not found"))
    assert _is_capability_probe_error(exc) is True


def test_is_capability_probe_error_with_method_not_found_message_no_code() -> None:
    """Message fallback only triggers when the server omitted the error code."""
    exc = McpError(ErrorData(code=0, message="Method not found on server"))
    # code=0 is truthy but not None — message fallback should NOT trigger
    assert _is_capability_probe_error(exc) is False

    # When code is genuinely absent (None), message fallback works
    exc2 = _make_mcp_error_none_code("Method not found on server")
    assert _is_capability_probe_error(exc2) is True


def test_is_capability_probe_error_rejects_infrastructure_errors() -> None:
    assert _is_capability_probe_error(RuntimeError("connection lost")) is False
    assert _is_capability_probe_error(AttributeError("no such attr")) is False
    exc = McpError(ErrorData(code=-32600, message="Invalid request"))
    assert _is_capability_probe_error(exc) is False
    # Different code + "method not found" in message should NOT match
    exc2 = McpError(ErrorData(code=-32000, message="Method not found on server"))
    assert _is_capability_probe_error(exc2) is False


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detach_server_clears_capabilities_cache(monkeypatch) -> None:
    context = _build_context(
        {"alpha": MCPServerSettings(name="alpha", transport="stdio", command="echo")}
    )

    expected_caps = ServerCapabilities(tools=ToolsCapability())

    @asynccontextmanager
    async def _fake_initialize_server(self, server_name, client_session_factory=None, trigger_oauth=None):
        del trigger_oauth
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


@pytest.mark.asyncio
async def test_attach_server_force_reconnect_refreshes_capabilities_cache() -> None:
    capability_generations = [
        ServerCapabilities(tools=ToolsCapability()),
        ServerCapabilities(prompts=PromptsCapability()),
    ]

    class _SequencedRegistry(ServerRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.registry = {
                "alpha": MCPServerSettings(name="alpha", transport="stdio", command="echo")
            }
            self.initialize_count = 0

        @asynccontextmanager
        async def initialize_server(
            self,
            server_name,
            client_session_factory=None,
            trigger_oauth=None,
        ):
            del client_session_factory, trigger_oauth
            capabilities = capability_generations[min(self.initialize_count, 1)]
            self.initialize_count += 1
            self._init_results[server_name] = InitializeResult(
                protocolVersion="2025-03-26",
                capabilities=capabilities,
                serverInfo=Implementation(name="stub", version="0.1"),
            )
            yield _DummySession()

    registry = _SequencedRegistry()
    context = Context(server_registry=registry)

    class _ReconnectAwareAggregator(MCPAggregator):
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
            del operation_type, operation_name, method_args, error_factory, progress_callback
            capabilities = self._require_server_registry().get_server_capabilities(server_name)
            if method_name == "list_tools":
                if capabilities and capabilities.tools:
                    return ListToolsResult(
                        tools=[Tool(name="echo", inputSchema={"type": "object"})]
                    )
                raise McpError(
                    ErrorData(code=METHOD_NOT_FOUND_ERROR_CODE, message="Method not found")
                )
            if method_name == "list_prompts":
                prompts = (
                    [SimpleNamespace(name="new-prompt")]
                    if capabilities and capabilities.prompts
                    else []
                )
                return SimpleNamespace(prompts=prompts)
            raise AssertionError(f"Unexpected MCP method: {method_name}")

        async def _evaluate_skybridge_for_server(
            self, server_name: str
        ) -> tuple[str, SkybridgeServerConfig]:
            return server_name, SkybridgeServerConfig(server_name=server_name)

    aggregator = _ReconnectAwareAggregator(
        server_names=["alpha"],
        connection_persistence=False,
        context=context,
    )

    first_caps = await aggregator.get_capabilities("alpha")
    assert first_caps is capability_generations[0]

    aggregator._attached_server_names.append("alpha")
    result = await aggregator.attach_server(
        server_name="alpha",
        options=MCPAttachOptions(force_reconnect=True),
    )

    assert registry.initialize_count == 2
    assert aggregator._capabilities_cache["alpha"] is capability_generations[1]
    assert result.prompts_added == ["new-prompt"]
    assert result.tools_total == 0
    assert result.prompts_total == 1
