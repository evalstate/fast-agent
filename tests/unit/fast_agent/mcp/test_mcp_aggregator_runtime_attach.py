import pytest
from mcp.types import ListToolsResult, Tool

from fast_agent.config import MCPServerSettings
from fast_agent.context import Context
from fast_agent.mcp.mcp_aggregator import (
    MCPAggregator,
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
