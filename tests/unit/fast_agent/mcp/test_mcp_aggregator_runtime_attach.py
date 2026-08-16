import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from mcp.client import CacheMode
from mcp_types import (
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    Prompt,
    ReadResourceResult,
    Resource,
    ServerCapabilities,
    TextResourceContents,
    Tool,
)

from fast_agent.config import MCPServerSettings
from fast_agent.context import Context
from fast_agent.mcp.app_integrations import AppServerConfig
from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
from fast_agent.mcp.mcp_aggregator import (
    MCPAggregator,
    MCPAttachOptions,
    MCPAttachResult,
    MCPDetachResult,
    NamespacedTool,
)
from fast_agent.mcp.skills_extension import ListSkillsResult, SkillEntry, SkillResource
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.ui.console_display import ConsoleDisplay

if TYPE_CHECKING:
    from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager


def _build_context(configs: dict[str, MCPServerSettings]) -> Context:
    registry = ServerRegistry()
    for name, config in configs.items():
        registry.register_central(name, config)
    return Context(server_registry=registry)


class _RecordingAggregator(MCPAggregator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attach_calls: list[str] = []

    async def attach_server(self, *, server_name: str, server_config=None, options=None):
        self.attach_calls.append(server_name)
        if server_name not in self._attached_server_names:
            self._attached_server_names.append(server_name)
        return MCPAttachResult(
            server_name=server_name,
            transport="stdio",
            attached=True,
            already_attached=False,
            tools_added=[],
            prompts_added=[],
            warnings=[],
        )


class _FailingStartupAggregator(_RecordingAggregator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.detach_calls: list[str] = []

    async def attach_server(self, *, server_name: str, server_config=None, options=None):
        if server_name == "beta":
            raise RuntimeError("beta failed")
        return await super().attach_server(
            server_name=server_name,
            server_config=server_config,
            options=options,
        )

    async def detach_server(self, server_name: str) -> MCPDetachResult:
        self.detach_calls.append(server_name)
        self._attached_server_names.remove(server_name)
        return MCPDetachResult(
            server_name=server_name,
            detached=True,
            tools_removed=[],
            prompts_removed=[],
        )


@pytest.mark.asyncio
async def test_load_servers_routes_startup_connections_through_attach_server() -> None:
    context = _build_context(
        {
            "alpha": MCPServerSettings(name="alpha", transport="stdio", command="echo"),
            "beta": MCPServerSettings(
                name="beta", transport="stdio", command="echo", load_on_start=False
            ),
        }
    )

    aggregator = _RecordingAggregator(
        server_names=["alpha", "beta"],
        connection_persistence=False,
        context=context,
    )

    await aggregator.load_servers()

    assert aggregator.attach_calls == ["alpha"]
    assert aggregator.list_attached_servers() == ["alpha"]

    await aggregator.load_servers(force_connect=True)

    assert aggregator.attach_calls == ["alpha", "alpha", "beta"]
    assert aggregator.list_attached_servers() == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_load_servers_rolls_back_cli_owned_startup_batch() -> None:
    registry = ServerRegistry()
    configs = {
        name: MCPServerSettings(name=name, transport="stdio", command="echo")
        for name in ("alpha", "beta")
    }
    registry.register_runtime_batch(configs, owner="cli-startup")
    aggregator = _FailingStartupAggregator(
        server_names=["alpha", "beta"],
        connection_persistence=False,
        context=Context(server_registry=registry),
    )

    with pytest.raises(RuntimeError, match="beta failed"):
        await aggregator.load_servers()

    assert aggregator.detach_calls == ["alpha"]
    assert aggregator.list_attached_servers() == []
    assert registry.registry == {}


@pytest.mark.asyncio
async def test_detach_server_removes_runtime_indexes() -> None:
    context = _build_context({})

    aggregator = MCPAggregator(
        server_names=["alpha"],
        connection_persistence=False,
        context=context,
    )

    namespaced_tool = NamespacedTool(
        tool=Tool(name="demo", input_schema={"type": "object"}),
        server_name="alpha",
        namespaced_tool_name="alpha.demo",
    )
    aggregator.server_names = ["alpha"]
    aggregator._attached_server_names = ["alpha"]
    aggregator._namespaced_tool_map = {"alpha.demo": namespaced_tool}
    aggregator._server_to_tool_map = {"alpha": [namespaced_tool]}
    aggregator._prompt_cache = {"alpha": []}
    aggregator._app_integration_configs = {"alpha": AppServerConfig(server_name="alpha")}

    result = await aggregator.detach_server("alpha")

    assert result.detached is True
    assert result.tools_removed == ["alpha.demo"]
    assert result.prompts_removed == []
    assert aggregator.list_attached_servers() == []
    assert aggregator._namespaced_tool_map == {}
    assert aggregator._server_to_tool_map == {}
    assert aggregator._prompt_cache == {}
    assert aggregator._app_integration_configs == {}


def test_list_configured_detached_servers_includes_registry_entries() -> None:
    context = _build_context(
        {
            "alpha": MCPServerSettings(name="alpha", transport="stdio", command="echo"),
            "beta": MCPServerSettings(name="beta", transport="stdio", command="echo"),
        }
    )

    aggregator = MCPAggregator(
        server_names=["alpha"],
        connection_persistence=False,
        context=context,
    )
    aggregator.server_names = ["alpha"]
    aggregator._attached_server_names = ["alpha"]

    assert aggregator.list_configured_detached_servers() == ["beta"]


def test_supplemental_attached_servers_are_not_reported_as_detached() -> None:
    context = _build_context(
        {
            "alpha": MCPServerSettings(name="alpha", transport="stdio", command="echo"),
            "stripe": MCPServerSettings(
                name="stripe",
                management="provider",
                transport="http",
                url="https://mcp.stripe.com",
            ),
        }
    )

    aggregator = MCPAggregator(
        server_names=["alpha"],
        connection_persistence=False,
        context=context,
    )
    aggregator._attached_server_names = ["alpha"]
    aggregator.set_supplemental_attached_servers(["stripe"])

    assert aggregator.list_attached_servers() == ["alpha", "stripe"]
    assert aggregator.list_configured_detached_servers() == []


@pytest.mark.asyncio
async def test_fetch_server_tools_optimistic_fallback_when_capability_missing() -> None:
    context = _build_context({})

    class _FallbackAggregator(MCPAggregator):
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
            return ListToolsResult(tools=[Tool(name="echo", input_schema={"type": "object"})])

    aggregator = _FallbackAggregator(
        server_names=["alpha"],
        connection_persistence=False,
        context=context,
    )

    tools = await aggregator._fetch_server_tools("alpha")
    assert [tool.name for tool in tools] == ["echo"]


@pytest.mark.asyncio
async def test_runtime_server_is_published_only_after_discovery() -> None:
    context = _build_context({})

    class _CapabilityAwareAggregator(MCPAggregator):
        async def get_capabilities(self, server_name: str):
            del server_name
            return ServerCapabilities.model_validate({"tools": {}, "prompts": {}})

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
                method_args,
                error_factory,
                progress_callback,
            )
            if method_name == "list_tools":
                assert context.server_registry is not None
                assert context.server_registry.get_server_config("runtime") is None
                return ListToolsResult(tools=[Tool(name="echo", input_schema={"type": "object"})])
            if method_name == "list_prompts":
                assert context.server_registry is not None
                assert context.server_registry.get_server_config("runtime") is None
                return SimpleNamespace(prompts=[SimpleNamespace(name="demo-prompt")])
            raise AssertionError(f"Unexpected MCP method: {method_name}")

        async def _evaluate_app_integrations_for_server(
            self,
            server_name: str,
            *,
            cache_mode: CacheMode = "use",
        ) -> tuple[str, AppServerConfig]:
            del cache_mode
            return server_name, AppServerConfig(server_name=server_name)

    aggregator = _CapabilityAwareAggregator(
        server_names=[],
        connection_persistence=False,
        context=context,
    )

    result = await aggregator.attach_server(
        server_name="runtime",
        server_config=MCPServerSettings(name="runtime", transport="stdio", command="echo"),
        options=MCPAttachOptions(),
    )

    assert len(result.tools_added) == 1
    assert result.tools_added[0].endswith("echo")
    assert result.prompts_added == ["demo-prompt"]
    assert result.tools_total == 1
    assert result.prompts_total == 1
    assert aggregator.server_names == ["runtime"]
    assert context.server_registry is not None
    assert context.server_registry.get_server_origin("runtime") == "runtime"


@pytest.mark.asyncio
@pytest.mark.parametrize("connection_persistence", [False, True])
@pytest.mark.parametrize("failure", ["exception", "cancel"])
async def test_attachment_discovery_failure_rolls_back_transaction(
    connection_persistence: bool,
    failure: str,
) -> None:
    context = _build_context({})
    disconnected: list[str] = []

    class _Manager:
        async def disconnect_server(self, server_name: str) -> None:
            disconnected.append(server_name)

    class _FailingAggregator(MCPAggregator):
        async def _connect_persistent_server(self, server_name, server_config, attach_options):
            del server_name, server_config, attach_options

        async def get_capabilities(self, server_name: str):
            del server_name
            return ServerCapabilities.model_validate({"tools": {}, "prompts": {}})

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
            del (
                server_name,
                operation_type,
                operation_name,
                method_args,
                error_factory,
                progress_callback,
            )
            if method_name == "list_tools":
                return ListToolsResult(tools=[Tool(name="staged", input_schema={"type": "object"})])
            if method_name == "list_prompts":
                if failure == "cancel":
                    raise asyncio.CancelledError
                raise RuntimeError("discovery failed")
            raise AssertionError(method_name)

    aggregator = _FailingAggregator(
        server_names=[],
        connection_persistence=connection_persistence,
        context=context,
    )
    if connection_persistence:
        aggregator._persistent_connection_manager = cast("MCPConnectionManager", _Manager())
    aggregator._capabilities_cache["runtime"] = ServerCapabilities()
    assert context.server_registry is not None
    context.server_registry.set_server_capabilities("runtime", ServerCapabilities())

    error = asyncio.CancelledError if failure == "cancel" else RuntimeError
    with pytest.raises(error):
        await aggregator.attach_server(
            server_name="runtime",
            server_config=MCPServerSettings(
                name="runtime",
                transport="stdio",
                command="echo",
            ),
        )

    assert context.server_registry.get_server_config("runtime") is None
    assert context.server_registry.get_server_capabilities("runtime") is None
    assert "runtime" not in aggregator._capabilities_cache
    assert aggregator.list_attached_servers() == []
    assert aggregator.server_names == []
    assert aggregator._namespaced_tool_map == {}
    assert aggregator._prompt_cache == {}
    assert disconnected == (["runtime"] if connection_persistence else [])


@pytest.mark.asyncio
async def test_failed_forced_reconnect_clears_stale_attachment_indexes() -> None:
    config = MCPServerSettings(name="central", transport="stdio", command="echo")
    context = _build_context({"central": config})

    class _FailingAggregator(MCPAggregator):
        async def _connect_persistent_server(self, server_name, server_config, attach_options):
            del server_name, server_config, attach_options
            raise RuntimeError("reconnect failed")

    aggregator = _FailingAggregator(
        server_names=["central"],
        connection_persistence=True,
        context=context,
    )
    tool = NamespacedTool(
        tool=Tool(name="stale", input_schema={"type": "object"}),
        server_name="central",
        namespaced_tool_name="central-stale",
    )
    aggregator._attached_server_names = ["central"]
    aggregator._server_to_tool_map["central"] = [tool]
    aggregator._namespaced_tool_map[tool.namespaced_tool_name] = tool

    with pytest.raises(RuntimeError, match="reconnect failed"):
        await aggregator.attach_server(
            server_name="central",
            options=MCPAttachOptions(force_reconnect=True),
        )

    assert aggregator.list_attached_servers() == []
    assert aggregator.server_names == []
    assert aggregator._server_to_tool_map == {}
    assert aggregator._namespaced_tool_map == {}
    assert context.server_registry is not None
    assert context.server_registry.get_server_origin("central") == "central"


@pytest.mark.asyncio
async def test_attachment_reconnect_override_is_local_and_not_published() -> None:
    context = _build_context({})
    supplied = MCPServerSettings(
        name="runtime",
        transport="stdio",
        command="echo",
        reconnect_on_disconnect=False,
    )

    class _OverrideAggregator(MCPAggregator):
        async def get_capabilities(self, server_name: str):
            config = self._server_config(server_name)
            assert config is not None
            assert config.reconnect_on_disconnect is True
            return ServerCapabilities()

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
            del (
                server_name,
                operation_type,
                operation_name,
                method_args,
                error_factory,
                progress_callback,
            )
            if method_name == "list_tools":
                return ListToolsResult(tools=[])
            raise AssertionError(method_name)

    aggregator = _OverrideAggregator(
        server_names=[],
        connection_persistence=False,
        context=context,
    )
    await aggregator.attach_server(
        server_name="runtime",
        server_config=supplied,
        options=MCPAttachOptions(reconnect_on_disconnect=True),
    )

    assert supplied.reconnect_on_disconnect is False
    assert context.server_registry is not None
    published = context.server_registry.get_server_config("runtime")
    assert published is not None
    assert published.reconnect_on_disconnect is False
    attached = aggregator._server_config("runtime")
    assert attached is not None
    assert attached.reconnect_on_disconnect is True


@pytest.mark.asyncio
async def test_detach_removes_only_runtime_owned_definition() -> None:
    context = _build_context(
        {"central": MCPServerSettings(name="central", transport="stdio", command="echo")}
    )
    assert context.server_registry is not None
    context.server_registry.register_card(
        "card",
        MCPServerSettings(name="card", transport="stdio", command="echo"),
    )
    aggregator = MCPAggregator(
        server_names=["central", "card", "runtime"],
        connection_persistence=False,
        context=context,
    )
    context.server_registry.register_runtime(
        "runtime",
        MCPServerSettings(name="runtime", transport="stdio", command="echo"),
        owner=aggregator._runtime_definition_owner,
    )
    aggregator._attached_server_names = ["central", "card", "runtime"]

    for name in ["central", "card", "runtime"]:
        result = await aggregator.detach_server(name)
        assert result.detached is True

    assert context.server_registry.get_server_config("central") is not None
    assert context.server_registry.get_server_config("card") is not None
    assert context.server_registry.get_server_config("runtime") is None


@pytest.mark.asyncio
async def test_interactive_startup_definition_transfers_to_attachment_owner() -> None:
    context = _build_context({})
    aggregator = MCPAggregator(
        server_names=["runtime"],
        connection_persistence=False,
        context=context,
    )
    assert context.server_registry is not None
    config = MCPServerSettings(name="runtime", transport="stdio", command="echo")
    context.server_registry.register_runtime(config.name or "runtime", config, owner="cli-startup")

    await aggregator._commit_server_attachment(
        "runtime",
        cast(
            "Any",
            SimpleNamespace(
                tools=[],
                prompts=[],
                skill_registry=None,
                app_integration_config=AppServerConfig(server_name="runtime"),
                capabilities=ServerCapabilities(),
            ),
        ),
    )

    assert context.server_registry.get_runtime_owners("runtime") == frozenset(
        {aggregator._runtime_definition_owner}
    )

    await aggregator.detach_server("runtime")

    assert context.server_registry.get_server_config("runtime") is None


@pytest.mark.asyncio
async def test_subscription_refresh_uses_public_refresh_mode_and_commits_canonical_uris() -> None:
    context = _build_context(
        {"alpha": MCPServerSettings(name="alpha", transport="stdio", command="echo")}
    )

    class _RefreshContractAggregator(MCPAggregator):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.calls: list[tuple[str, dict[str, Any]]] = []
            self.fail_resource_list = False

        async def get_capabilities(self, server_name: str) -> ServerCapabilities:
            del server_name
            return ServerCapabilities.model_validate(
                {
                    "tools": {"listChanged": True},
                    "prompts": {"listChanged": True},
                    "resources": {"subscribe": True, "listChanged": True},
                }
            )

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
                error_factory,
                progress_callback,
            )
            args = dict(method_args or {})
            self.calls.append((method_name, args))
            if method_name == "list_tools":
                tool = Tool.model_validate(
                    {
                        "name": "render",
                        "inputSchema": {"type": "object"},
                        "_meta": {
                            "ui": {
                                "resourceUri": "ui://component/app",
                                "visibility": ["model", "app"],
                            }
                        },
                    }
                )
                return ListToolsResult(tools=[tool])
            if method_name == "list_prompts":
                return ListPromptsResult(prompts=[Prompt(name="draft")])
            if method_name == "list_resources":
                if self.fail_resource_list:
                    raise RuntimeError("transient resource-list failure")
                return ListResourcesResult(
                    resources=[
                        Resource(name="App", uri="ui://component/app"),
                    ]
                )
            if method_name == "read_resource":
                return ReadResourceResult(
                    contents=[
                        TextResourceContents(
                            uri="ui://component/app",
                            mime_type="text/html;profile=mcp-app",
                            text="<html />",
                        )
                    ]
                )
            raise AssertionError(method_name)

    aggregator = _RefreshContractAggregator(
        server_names=["alpha"],
        connection_persistence=False,
        context=context,
    )
    aggregator._attached_server_names = ["alpha"]
    runtime = MCPClientCallbackRuntime(
        server_name="alpha",
        server_config=context.server_registry.get_server_config("alpha")
        if context.server_registry
        else None,
        aggregator=aggregator,
    )

    uris = await runtime.refresh_subscription_state()

    assert uris == ("ui://component/app",)
    assert aggregator.selected_materialized_resource_uris("alpha") == uris
    assert set(aggregator._namespaced_tool_map) == {"alpha__render"}
    assert [prompt.name for prompt in aggregator._prompt_cache["alpha"]] == ["draft"]
    assert [method for method, _ in aggregator.calls] == [
        "list_tools",
        "list_prompts",
        "list_resources",
        "read_resource",
    ]
    assert all(args["cache_mode"] == "refresh" for _, args in aggregator.calls)

    committed = aggregator._app_integration_configs["alpha"]
    aggregator.fail_resource_list = True
    with pytest.raises(RuntimeError, match="transient resource-list failure"):
        await runtime.refresh_subscription_state()
    assert aggregator._app_integration_configs["alpha"] is committed


@pytest.mark.asyncio
async def test_card_tool_refresh_preserves_visible_namespace() -> None:
    context = _build_context({})
    assert context.server_registry is not None
    internal_name = "card-source-revision-docs"
    context.server_registry.register_card(
        internal_name,
        MCPServerSettings(name="docs", transport="stdio", command="echo"),
    )

    class _RefreshAggregator(MCPAggregator):
        async def server_supports_feature(self, server_name: str, feature: str) -> bool:
            del server_name
            return feature == "tools"

        async def _execute_on_server(self, *args, **kwargs):
            del args, kwargs
            return ListToolsResult(
                tools=[
                    Tool(
                        name="search",
                        input_schema={},
                        output_schema={
                            "type": "object",
                            "properties": {"matches": {"type": "array"}},
                        },
                    )
                ]
            )

    aggregator = _RefreshAggregator(
        server_names=[internal_name],
        connection_persistence=False,
        context=context,
    )

    class _SilentDisplay(ConsoleDisplay):
        async def show_tool_update(
            self,
            updated_server: str,
            agent_name: str | None = None,
        ) -> None:
            del updated_server, agent_name

    aggregator.display = _SilentDisplay(config=None)

    await aggregator._refresh_server_tools(internal_name)

    assert set(aggregator._namespaced_tool_map) == {"docs__search"}
    assert aggregator._namespaced_tool_map["docs__search"].tool.output_schema == {
        "type": "object",
        "properties": {"matches": {"type": "array"}},
    }


@pytest.mark.asyncio
async def test_card_grouped_apis_accept_and_return_visible_namespace() -> None:
    context = _build_context(
        {
            "docs": MCPServerSettings(
                name="docs",
                transport="stdio",
                command="unrelated-central",
            )
        }
    )
    assert context.server_registry is not None
    internal_name = "card-source-revision-docs"
    context.server_registry.register_card(
        internal_name,
        MCPServerSettings(name="docs", transport="stdio", command="echo"),
    )

    class _CollectionAggregator(MCPAggregator):
        async def server_supports_feature(self, server_name: str, feature: str) -> bool:
            del server_name, feature
            return True

        async def _execute_on_server(self, *args, **kwargs):
            del args, kwargs
            return ListToolsResult(tools=[Tool(name="search", input_schema={})])

    aggregator = _CollectionAggregator(
        server_names=[internal_name],
        connection_persistence=False,
        context=context,
    )
    aggregator.initialized = True
    aggregator._prompt_cache[internal_name] = [Prompt(name="summarize")]

    prompts = await aggregator.list_prompts("docs")
    tools = await aggregator.list_mcp_tools("docs")
    resolved_prompt = aggregator._resolve_prompt_name("summarize", "docs")

    assert list(prompts) == ["docs"]
    assert list(tools) == ["docs"]
    assert resolved_prompt.server_name == internal_name


@pytest.mark.asyncio
async def test_close_releases_runtime_definition_owner() -> None:
    context = _build_context({})
    aggregator = MCPAggregator(
        server_names=[],
        connection_persistence=False,
        context=context,
    )
    assert context.server_registry is not None
    config = MCPServerSettings(name="runtime", transport="stdio", command="echo")
    context.server_registry.register_runtime(
        "runtime",
        config,
        owner=aggregator._runtime_definition_owner,
    )
    aggregator._attachment_configs["runtime"] = config

    await aggregator.close()

    assert context.server_registry.get_server_config("runtime") is None
    assert aggregator._attachment_configs == {}


@pytest.mark.asyncio
async def test_detach_closes_local_runtime_but_keeps_shared_definition() -> None:
    context = _build_context({})
    aggregator = MCPAggregator(
        server_names=["runtime"],
        connection_persistence=True,
        context=context,
    )
    assert context.server_registry is not None
    config = MCPServerSettings(name="runtime", transport="stdio", command="echo")
    context.server_registry.register_runtime(
        "runtime",
        config,
        owner=aggregator._runtime_definition_owner,
    )
    context.server_registry.register_runtime("runtime", config, owner="other")
    context.server_registry.register_attachment(
        "runtime",
        owner=aggregator._attachment_owner,
    )
    context.server_registry.register_attachment("runtime", owner="other")
    aggregator._attached_server_names = ["runtime"]
    disconnected: list[str] = []

    class _Manager:
        async def disconnect_server(self, server_name: str) -> None:
            disconnected.append(server_name)

    aggregator._persistent_connection_manager = cast("MCPConnectionManager", _Manager())

    await aggregator.detach_server("runtime")

    assert disconnected == ["runtime"]
    assert context.server_registry.get_server_config("runtime") is not None
    assert context.server_registry.get_runtime_owners("runtime") == frozenset({"other"})
    assert context.server_registry.get_attachment_owners("runtime") == frozenset({"other"})


@pytest.mark.asyncio
async def test_attached_result_uses_cached_mcp_skill_registry() -> None:
    context = _build_context({})

    class _NoResultRegistryScanAggregator(MCPAggregator):
        async def _scan_mcp_skill_registry(
            self,
            server_name: str,
            *,
            cache_mode: CacheMode = "use",
        ):
            del cache_mode
            raise AssertionError(f"unexpected registry scan from result for {server_name}")

    aggregator = _NoResultRegistryScanAggregator(
        server_names=["runtime"],
        connection_persistence=False,
        context=context,
    )
    aggregator._attached_server_names = ["runtime"]

    result = await aggregator._attached_result(
        server_name="runtime",
        resolved_config=MCPServerSettings(
            name="runtime", transport="http", url="https://example.com/mcp"
        ),
        already_attached=False,
        existing_tool_names=set(),
        existing_prompt_names=set(),
        app_integration_config=AppServerConfig(server_name="runtime"),
    )

    assert result.skills_total is None


@pytest.mark.asyncio
async def test_refresh_attached_server_cache_discovers_mcp_skill_registry() -> None:
    context = _build_context({})
    skill_uri = "skill://demo/SKILL.md"
    entry = SkillEntry(
        uri=skill_uri,
        frontmatter={"name": "demo", "description": "Demo skill"},
        resources=[
            SkillResource(
                uri=skill_uri,
                digest="sha256:0000000000000000000000000000000000000000000000000000000000000000",
            )
        ],
    )

    class _RegistryCachingAggregator(MCPAggregator):
        async def get_capabilities(self, server_name: str):
            del server_name
            return ServerCapabilities.model_validate(
                {"resources": {}, "extensions": {"io.modelcontextprotocol/skills": {}}}
            )

        async def server_supports_feature(self, server_name: str, feature: str) -> bool:
            del server_name
            return feature in {"resources", "tools", "prompts"}

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
            del server_name, operation_type, operation_name, error_factory, progress_callback
            if method_name == "list_tools":
                return ListToolsResult(tools=[])
            if method_name == "list_prompts":
                return SimpleNamespace(prompts=[])
            if method_name == "list_skills":
                assert method_args is None
                return ListSkillsResult(skills=[entry])
            raise AssertionError(f"Unexpected MCP method: {method_name}")

        async def _evaluate_app_integrations_for_server(
            self,
            server_name: str,
            *,
            cache_mode: CacheMode = "use",
        ) -> tuple[str, AppServerConfig]:
            del cache_mode
            return server_name, AppServerConfig(server_name=server_name)

    aggregator = _RegistryCachingAggregator(
        server_names=["runtime"],
        connection_persistence=False,
        context=context,
    )

    await aggregator._refresh_attached_server_cache("runtime")
    aggregator.initialized = True
    aggregator._attached_server_names = ["runtime"]

    registries = await aggregator.list_mcp_skill_registries()

    assert len(registries) == 1
    assert registries[0].server_name == "runtime"
    assert [skill.name for skill in registries[0].skills] == ["demo"]
    assert await aggregator._mcp_skills_total("runtime") == 1


@pytest.mark.asyncio
async def test_collect_server_status_does_not_probe_detached_capabilities() -> None:
    context = _build_context(
        {
            "deferred": MCPServerSettings(
                name="deferred",
                transport="http",
                url="https://example.com/mcp",
                load_on_start=False,
            )
        }
    )

    class _NoCapabilityProbeAggregator(MCPAggregator):
        async def get_capabilities(self, server_name: str):
            raise AssertionError(f"unexpected capability probe for {server_name}")

    aggregator = _NoCapabilityProbeAggregator(
        server_names=["deferred"],
        connection_persistence=True,
        context=context,
    )

    status = await aggregator.collect_server_status()

    assert status["deferred"].mcp_skills_enabled is False
    assert aggregator.list_attached_servers() == []


@pytest.mark.asyncio
async def test_list_mcp_skill_registries_scans_only_attached_servers() -> None:
    context = _build_context(
        {
            "attached": MCPServerSettings(
                name="attached",
                transport="http",
                url="https://attached.example/mcp",
            ),
            "detached": MCPServerSettings(
                name="detached",
                transport="http",
                url="https://detached.example/mcp",
            ),
        }
    )

    class _RegistryScanAggregator(MCPAggregator):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.capability_probes: list[str] = []

        async def get_capabilities(self, server_name: str):
            self.capability_probes.append(server_name)
            return ServerCapabilities()

    aggregator = _RegistryScanAggregator(
        server_names=["attached", "detached"],
        connection_persistence=False,
        context=context,
    )
    aggregator.initialized = True
    aggregator._attached_server_names = ["attached"]

    registries = await aggregator.list_mcp_skill_registries()

    assert registries == []
    assert aggregator.capability_probes == ["attached"]
