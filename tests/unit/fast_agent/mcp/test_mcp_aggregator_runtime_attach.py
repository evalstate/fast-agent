from types import SimpleNamespace

import pytest
from mcp.types import (
    ListToolsResult,
    ReadResourceResult,
    ServerCapabilities,
    TextResourceContents,
    Tool,
)
from pydantic import AnyUrl

from fast_agent.config import MCPServerSettings
from fast_agent.context import Context
from fast_agent.mcp.mcp_aggregator import (
    MCPAggregator,
    MCPAttachOptions,
    MCPAttachResult,
    NamespacedTool,
)
from fast_agent.mcp.skybridge import SkybridgeServerConfig
from fast_agent.mcp_server_registry import ServerRegistry


def _build_context(configs: dict[str, MCPServerSettings]) -> Context:
    registry = ServerRegistry()
    registry.registry = configs
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
async def test_detach_server_removes_runtime_indexes() -> None:
    context = _build_context({})

    aggregator = MCPAggregator(
        server_names=["alpha"],
        connection_persistence=False,
        context=context,
    )

    namespaced_tool = NamespacedTool(
        tool=Tool(name="demo", inputSchema={"type": "object"}),
        server_name="alpha",
        namespaced_tool_name="alpha.demo",
    )
    aggregator.server_names = ["alpha"]
    aggregator._attached_server_names = ["alpha"]
    aggregator._namespaced_tool_map = {"alpha.demo": namespaced_tool}
    aggregator._server_to_tool_map = {"alpha": [namespaced_tool]}
    aggregator._prompt_cache = {"alpha": []}
    aggregator._skybridge_configs = {"alpha": SkybridgeServerConfig(server_name="alpha")}

    result = await aggregator.detach_server("alpha")

    assert result.detached is True
    assert result.tools_removed == ["alpha.demo"]
    assert result.prompts_removed == []
    assert aggregator.list_attached_servers() == []
    assert aggregator._namespaced_tool_map == {}
    assert aggregator._server_to_tool_map == {}
    assert aggregator._prompt_cache == {}
    assert aggregator._skybridge_configs == {}


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
            return ListToolsResult(
                tools=[Tool(name="echo", inputSchema={"type": "object"})]
            )

    aggregator = _FallbackAggregator(
        server_names=["alpha"],
        connection_persistence=False,
        context=context,
    )

    tools = await aggregator._fetch_server_tools("alpha")
    assert [tool.name for tool in tools] == ["echo"]


@pytest.mark.asyncio
async def test_attach_server_registers_runtime_server_before_prompt_discovery() -> None:
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
                return ListToolsResult(
                    tools=[Tool(name="echo", inputSchema={"type": "object"})]
                )
            if method_name == "list_prompts":
                return SimpleNamespace(prompts=[SimpleNamespace(name="demo-prompt")])
            raise AssertionError(f"Unexpected MCP method: {method_name}")

        async def _evaluate_skybridge_for_server(
            self, server_name: str
        ) -> tuple[str, SkybridgeServerConfig]:
            return server_name, SkybridgeServerConfig(server_name=server_name)

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


@pytest.mark.asyncio
async def test_attached_result_uses_cached_mcp_skill_registry() -> None:
    context = _build_context({})

    class _NoResultRegistryScanAggregator(MCPAggregator):
        async def _scan_mcp_skill_registry(self, server_name: str):
            raise AssertionError(f"unexpected registry scan from result for {server_name}")

    aggregator = _NoResultRegistryScanAggregator(
        server_names=["runtime"],
        connection_persistence=False,
        context=context,
    )
    aggregator._attached_server_names = ["runtime"]

    result = await aggregator._attached_result(
        server_name="runtime",
        resolved_config=MCPServerSettings(name="runtime", transport="http", url="https://example.com/mcp"),
        already_attached=False,
        existing_tool_names=set(),
        existing_prompt_names=set(),
        skybridge_config=SkybridgeServerConfig(server_name="runtime"),
    )

    assert result.skills_total is None


@pytest.mark.asyncio
async def test_refresh_attached_server_cache_discovers_mcp_skill_registry() -> None:
    context = _build_context({})
    index_uri = "skill://index.json"
    skill_uri = "skill://demo/SKILL.md"
    index_text = (
        '{"skills":[{"name":"demo","description":"Demo skill","type":"skill-md",'
        f'"url":"{skill_uri}","digest":"sha256:'
        '0000000000000000000000000000000000000000000000000000000000000000"}]}'
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
            if method_name == "read_resource":
                assert method_args == {"uri": AnyUrl(index_uri)}
                return ReadResourceResult(
                    contents=[
                        TextResourceContents(
                            uri=AnyUrl(index_uri),
                            mimeType="application/json",
                            text=index_text,
                        )
                    ]
                )
            raise AssertionError(f"Unexpected MCP method: {method_name}")

        async def _evaluate_skybridge_for_server(
            self, server_name: str
        ) -> tuple[str, SkybridgeServerConfig]:
            return server_name, SkybridgeServerConfig(server_name=server_name)

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
