from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from mcp.client import CacheMode
from mcp.shared.exceptions import MCPError as McpError
from mcp_types import (
    CallToolResult,
    ErrorData,
    ListPromptsResult,
    ListToolsResult,
    Prompt,
    PromptsCapability,
    ServerCapabilities,
    TextContent,
    Tool,
    ToolsCapability,
)

from fast_agent.config import MCPServerAuthSettings, MCPServerSettings
from fast_agent.context import Context
from fast_agent.mcp.app_integrations import AppServerConfig
from fast_agent.mcp.auth.context import request_bearer_token
from fast_agent.mcp.mcp_aggregator import (
    METHOD_NOT_FOUND_ERROR_CODE,
    MCPAggregator,
    MCPAttachOptions,
    _is_capability_probe_error,
)
from fast_agent.mcp_server_registry import ServerRegistry

if TYPE_CHECKING:
    from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager


def _build_context(configs: dict[str, MCPServerSettings]) -> Context:
    registry = ServerRegistry()
    registry.registry = configs
    return Context(server_registry=registry)


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


@pytest.mark.asyncio
async def test_handle_auth_challenge_reports_retry_failure() -> None:
    class _FailingManager:
        async def reconnect_server(self, server_name, callback_runtime, trigger_oauth=None):
            del server_name, callback_runtime, trigger_oauth
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
async def test_connection_error_does_not_replay_tool_call() -> None:
    context = _build_context({"alpha": MCPServerSettings(name="alpha")})
    aggregator = MCPAggregator(
        server_names=["alpha"],
        connection_persistence=True,
        context=context,
    )
    call_count = 0

    class _ToolClient:
        async def call_tool(self, **kwargs):
            nonlocal call_count
            del kwargs
            call_count += 1
            raise ConnectionError("response lost after dispatch")

    class _ReconnectedClient:
        async def call_tool(self, **kwargs):
            nonlocal call_count
            del kwargs
            call_count += 1
            return CallToolResult(content=[TextContent(type="text", text="replayed")])

    class _PersistentManager:
        reconnect_count = 0

        async def get_server(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(client=_ToolClient(), negotiation="adopt")

        async def reconnect_server(self, *args, **kwargs):
            callback_runtime = kwargs["callback_runtime"]
            del args, kwargs
            self.reconnect_count += 1
            return SimpleNamespace(
                client=_ReconnectedClient(),
                negotiation="adopt",
                _callback_runtime=callback_runtime,
            )

    manager = _PersistentManager()
    aggregator._persistent_connection_manager = cast("MCPConnectionManager", manager)

    result = await aggregator._execute_on_server(
        "alpha",
        "tools/call",
        "write",
        "call_tool",
        method_args={"name": "write", "arguments": {}},
        error_factory=lambda message: CallToolResult(
            is_error=True,
            content=[TextContent(type="text", text=message)],
        ),
    )

    assert result.is_error is True
    assert isinstance(result.content[0], TextContent)
    assert "response lost after dispatch" in result.content[0].text
    assert call_count == 1
    assert manager.reconnect_count == 1


@pytest.mark.asyncio
async def test_connection_error_replays_list_operation_once() -> None:
    context = _build_context({"alpha": MCPServerSettings(name="alpha")})
    aggregator = MCPAggregator(
        server_names=["alpha"],
        connection_persistence=True,
        context=context,
    )
    call_count = 0

    class _DisconnectedClient:
        async def list_tools(self):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("connection dropped")

    class _ReconnectedClient:
        async def list_tools(self):
            nonlocal call_count
            call_count += 1
            return ListToolsResult(tools=[Tool(name="echo", input_schema={"type": "object"})])

    class _PersistentManager:
        reconnect_count = 0

        async def get_server(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(client=_DisconnectedClient(), negotiation="adopt")

        async def reconnect_server(self, *args, **kwargs):
            callback_runtime = kwargs["callback_runtime"]
            del args, kwargs
            self.reconnect_count += 1
            return SimpleNamespace(
                client=_ReconnectedClient(),
                negotiation="adopt",
                _callback_runtime=callback_runtime,
            )

    manager = _PersistentManager()
    aggregator._persistent_connection_manager = cast("MCPConnectionManager", manager)

    result = await aggregator._execute_on_server(
        "alpha",
        "tools/list",
        "",
        "list_tools",
    )

    assert [tool.name for tool in result.tools] == ["echo"]
    assert call_count == 2
    assert manager.reconnect_count == 1


@pytest.mark.asyncio
async def test_execute_on_server_nonpersistent_retries_with_oauth_after_401(
    monkeypatch,
) -> None:
    context = _build_context(
        {"alpha": MCPServerSettings(name="alpha", transport="http", url="https://example.com")}
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
            return ListToolsResult(tools=[Tool(name="echo", input_schema={"type": "object"})])

    @asynccontextmanager
    async def _fake_gen_client(
        server_name,
        server_registry,
        *,
        callback_runtime=None,
        trigger_oauth=None,
    ):
        del server_name, server_registry, callback_runtime
        trigger_history.append(trigger_oauth)
        yield _RetryClient(trigger_oauth)

    monkeypatch.setattr("fast_agent.mcp.mcp_aggregator.gen_client", _fake_gen_client)

    result = await aggregator._execute_on_server(
        "alpha",
        "tools/list",
        "",
        "list_tools",
        error_factory=lambda _: ListToolsResult(tools=[]),
    )

    assert isinstance(result, ListToolsResult)
    assert [tool.name for tool in result.tools] == ["echo"]
    assert trigger_history == [None, True]


@pytest.mark.asyncio
async def test_execute_on_server_uses_request_scoped_connection_for_forwarded_hf_auth(
    monkeypatch,
) -> None:
    context = _build_context(
        {
            "hf": MCPServerSettings(
                name="hf",
                transport="http",
                url="https://huggingface.co/mcp",
                auth=MCPServerAuthSettings(forward="huggingface"),
            )
        }
    )
    aggregator = MCPAggregator(
        server_names=["hf"],
        connection_persistence=True,
        context=context,
    )

    class _PersistentManager:
        async def get_server(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("persistent connection must not be reused for forwarded auth")

    class _RequestClient:
        async def call_tool(self, **kwargs):
            del kwargs
            return CallToolResult(content=[TextContent(type="text", text="ok")])

        async def read_resource(self, **kwargs):
            raise AssertionError(kwargs)

        async def get_prompt(self, **kwargs):
            raise AssertionError(kwargs)

    gen_client_calls: list[str] = []

    @asynccontextmanager
    async def _fake_gen_client(
        server_name,
        server_registry,
        *,
        callback_runtime=None,
        trigger_oauth=None,
    ):
        del server_registry, callback_runtime, trigger_oauth
        gen_client_calls.append(server_name)
        yield _RequestClient()

    monkeypatch.setattr("fast_agent.mcp.mcp_aggregator.gen_client", _fake_gen_client)
    aggregator._persistent_connection_manager = cast("MCPConnectionManager", _PersistentManager())

    token = request_bearer_token.set("request-token")
    try:
        result = await aggregator._execute_on_server(
            "hf",
            "tools/call",
            "hf_whoami",
            "call_tool",
            method_args={"name": "hf_whoami", "arguments": {}},
        )
    finally:
        request_bearer_token.reset(token)

    assert isinstance(result, CallToolResult)
    assert gen_client_calls == ["hf"]


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
    async def _fake_gen_client(*args, **kwargs):
        del args, kwargs
        yield SimpleNamespace(server_capabilities=expected_caps)

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_aggregator.gen_client",
        _fake_gen_client,
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
    async def _counting_gen_client(*args, **kwargs):
        del args, kwargs
        nonlocal init_count
        init_count += 1
        yield SimpleNamespace(server_capabilities=expected_caps)

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_aggregator.gen_client",
        _counting_gen_client,
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
    """get_capabilities degrades gracefully when gateway startup raises."""
    context = _build_context(
        {"broken": MCPServerSettings(name="broken", transport="stdio", command="echo")}
    )

    @asynccontextmanager
    async def _exploding_gen_client(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("server crashed on startup")
        yield

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_aggregator.gen_client",
        _exploding_gen_client,
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
    error_data = ErrorData.model_construct(code=None, message=message)
    error = McpError(0, message)
    error.error = error_data
    return error


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
        execute_error=McpError.from_error_data(
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
        execute_error=McpError.from_error_data(ErrorData(code=-32600, message="Invalid request")),
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
                Tool(name="read_file", input_schema={"type": "object"}),
                Tool(name="write_file", input_schema={"type": "object"}),
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
        execute_error=McpError.from_error_data(ErrorData(code=-32600, message="Invalid request")),
    )
    with pytest.raises(McpError):
        await aggregator._fetch_server_tools("broken")


# ---------------------------------------------------------------------------
# _is_capability_probe_error
# ---------------------------------------------------------------------------


def test_is_capability_probe_error_with_not_implemented_error() -> None:
    assert _is_capability_probe_error(NotImplementedError("not supported")) is True


def test_is_capability_probe_error_with_method_not_found_code() -> None:
    exc = McpError.from_error_data(
        ErrorData(code=METHOD_NOT_FOUND_ERROR_CODE, message="Method not found")
    )
    assert _is_capability_probe_error(exc) is True


def test_is_capability_probe_error_with_method_not_found_message_no_code() -> None:
    """Message fallback only triggers when the server omitted the error code."""
    exc = McpError.from_error_data(ErrorData(code=0, message="Method not found on server"))
    # code=0 is truthy but not None — message fallback should NOT trigger
    assert _is_capability_probe_error(exc) is False

    # When code is genuinely absent (None), message fallback works
    exc2 = _make_mcp_error_none_code("Method not found on server")
    assert _is_capability_probe_error(exc2) is True


def test_is_capability_probe_error_rejects_infrastructure_errors() -> None:
    assert _is_capability_probe_error(RuntimeError("connection lost")) is False
    assert _is_capability_probe_error(AttributeError("no such attr")) is False
    exc = McpError.from_error_data(ErrorData(code=-32600, message="Invalid request"))
    assert _is_capability_probe_error(exc) is False
    # Different code + "method not found" in message should NOT match
    exc2 = McpError.from_error_data(ErrorData(code=-32000, message="Method not found on server"))
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
    async def _fake_gen_client(*args, **kwargs):
        del args, kwargs
        yield SimpleNamespace(server_capabilities=expected_caps)

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_aggregator.gen_client",
        _fake_gen_client,
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
async def test_attach_server_force_reconnect_refreshes_capabilities_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    registry = _SequencedRegistry()
    context = Context(server_registry=registry)

    @asynccontextmanager
    async def _sequenced_gen_client(*args, **kwargs):
        del args, kwargs
        capabilities = capability_generations[min(registry.initialize_count, 1)]
        registry.initialize_count += 1
        registry.set_server_capabilities("alpha", capabilities)
        yield SimpleNamespace(server_capabilities=capabilities)

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_aggregator.gen_client",
        _sequenced_gen_client,
    )

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
                        tools=[Tool(name="echo", input_schema={"type": "object"})]
                    )
                raise McpError.from_error_data(
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

        async def _evaluate_app_integrations_for_server(
            self,
            server_name: str,
            *,
            cache_mode: CacheMode = "use",
        ) -> tuple[str, AppServerConfig]:
            del cache_mode
            return server_name, AppServerConfig(server_name=server_name)

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


@pytest.mark.asyncio
async def test_list_prompts_does_not_cache_transient_list_failure() -> None:
    context = _build_context({"alpha": MCPServerSettings(name="alpha", transport="stdio")})

    class _PromptListAggregator(MCPAggregator):
        calls = 0

        async def get_capabilities(self, server_name):
            assert server_name == "alpha"
            return ServerCapabilities(prompts=PromptsCapability())

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
            del operation_type, operation_name, method_args, progress_callback
            assert server_name == "alpha"
            assert method_name == "list_prompts"
            self.calls += 1
            if self.calls == 1:
                return error_factory("temporary failure") if error_factory else None
            return ListPromptsResult(prompts=[Prompt(name="available")])

    aggregator = _PromptListAggregator(
        server_names=["alpha"],
        connection_persistence=False,
        context=context,
    )
    aggregator.initialized = True

    first = await aggregator.list_prompts(server_name="alpha")
    assert first == {"alpha": []}
    assert "alpha" not in aggregator._prompt_cache

    second = await aggregator.list_prompts(server_name="alpha")
    assert [prompt.name for prompt in second["alpha"]] == ["available"]
    assert [prompt.name for prompt in aggregator._prompt_cache["alpha"]] == ["available"]
