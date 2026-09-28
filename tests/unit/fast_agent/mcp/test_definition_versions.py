"""Digest mode: server definition versions replace the snapshot TTL and gate tool calls."""

import time
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from mcp.shared.exceptions import MCPError
from mcp_types import CallToolResult, ListToolsResult, TextContent, Tool

from fast_agent.config import MCPServerSettings, MCPToolCacheSettings
from fast_agent.context import Context
from fast_agent.mcp.common import create_namespaced_name
from fast_agent.mcp.definition_versions import (
    DEFINITION_VERSION_MISMATCH,
    KNOWN_DEFINITION_VERSIONS_META,
    DefinitionVersions,
    stale_definitions,
)
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.mcp_aggregator import DEFINITIONS_CHANGED_MESSAGE, MCPAggregator, NamespacedTool
from fast_agent.mcp.tool_catalog_cache import ToolCatalogCache, ToolSnapshot
from fast_agent.mcp_server_registry import ServerRegistry

TOOL = Tool(name="search", input_schema={"type": "object"})


def _config(tmp_path, *, include_instructions: bool = True) -> MCPServerSettings:
    return MCPServerSettings(
        command="server",
        connection_policy="deferred",
        include_instructions=include_instructions,
        tool_cache=MCPToolCacheSettings(directory=str(tmp_path), ttl_seconds=60),
    )


def _save(cache: ToolCatalogCache, versions: DefinitionVersions, *, age: float) -> None:
    cache.save(
        ToolSnapshot(
            key=cache.key,
            fetched_at=time.time() - age,
            tools=[TOOL],
            instructions="Use search.",
            definition_versions=versions,
        )
    )


def test_digest_snapshot_ignores_ttl_only_when_it_covers_rendered_definitions(tmp_path):
    cache = ToolCatalogCache(_config(tmp_path))
    full = DefinitionVersions(tools="sha256:t", instructions="sha256:i")
    _save(cache, full, age=3600)
    assert cache.load() is not None
    # Instructions are rendered from the snapshot, so they need a digest too.
    _save(cache, DefinitionVersions(tools="sha256:t"), age=3600)
    assert cache.load() is None
    _save(cache, DefinitionVersions(), age=3600)
    assert cache.load() is None
    tools_only = ToolCatalogCache(_config(tmp_path, include_instructions=False))
    _save(tools_only, DefinitionVersions(tools="sha256:t"), age=3600)
    assert tools_only.load() is not None


def test_mismatch_errors_name_stale_targets_and_default_to_everything():
    named = MCPError(DEFINITION_VERSION_MISMATCH, "changed", {"stale": ["instructions"]})
    assert stale_definitions(named) == {"instructions"}
    assert stale_definitions(MCPError(DEFINITION_VERSION_MISMATCH, "changed")) == {
        "tools",
        "instructions",
    }
    assert stale_definitions(MCPError(-32602, "invalid params")) is None
    assert stale_definitions(RuntimeError("boom")) is None


def _connected_aggregator(tmp_path) -> MCPAggregator:
    registry = ServerRegistry()
    registry.register_central("hf", _config(tmp_path))
    aggregator = MCPAggregator(
        server_names=["hf"], connection_persistence=False, context=Context(server_registry=registry)
    )
    aggregator.initialized = True
    namespaced = NamespacedTool(
        tool=TOOL, server_name="hf", namespaced_tool_name=create_namespaced_name("hf", "search")
    )
    aggregator._server_to_tool_map["hf"] = [namespaced]
    aggregator._namespaced_tool_map[namespaced.namespaced_tool_name] = namespaced
    aggregator._definition_versions["hf"] = DefinitionVersions(
        tools="sha256:t", instructions="sha256:i"
    )
    return aggregator


@pytest.mark.asyncio
async def test_calls_echo_known_versions_and_mismatch_refreshes_without_executing(
    monkeypatch, tmp_path
):
    aggregator = _connected_aggregator(tmp_path)
    sent: list[dict[str, Any]] = []
    mismatch = True

    async def execute(**kwargs: Any) -> CallToolResult:
        sent.append(kwargs["method_args"])
        if mismatch:
            raise MCPError(DEFINITION_VERSION_MISMATCH, "changed", {"stale": ["tools"]})
        return CallToolResult(content=[TextContent(type="text", text="ran")])

    refresh = AsyncMock()
    monkeypatch.setattr(aggregator, "_execute_on_server", execute)
    monkeypatch.setattr(aggregator, "_refresh_server_tools", refresh)
    generation = aggregator.definitions_generation

    result = await aggregator.call_tool("hf__search", {"q": "x"})

    assert sent[0]["meta"] == {
        KNOWN_DEFINITION_VERSIONS_META: {"tools": "sha256:t", "instructions": "sha256:i"}
    }
    assert result.is_error
    assert get_text(result.content[0]) == DEFINITIONS_CHANGED_MESSAGE
    refresh.assert_awaited_once_with("hf")
    assert aggregator.definitions_generation > generation

    mismatch = False
    result = await aggregator.call_tool("hf__search", {"q": "x"})
    assert not result.is_error


@pytest.mark.asyncio
async def test_tool_runner_relists_listed_tools_when_definitions_change():
    from fast_agent.agents.tool_runner import ToolRunner

    class _Agent:
        tool_definitions_generation = 0
        listed = 0

        async def list_tools(self) -> ListToolsResult:
            self.listed += 1
            return ListToolsResult(tools=[TOOL])

    agent = _Agent()
    runner = ToolRunner(agent=cast("Any", agent), messages=[])
    await runner._ensure_tools_ready()
    await runner._ensure_tools_ready()
    assert agent.listed == 1
    agent.tool_definitions_generation = 1
    await runner._ensure_tools_ready()
    assert agent.listed == 2

    pinned = ToolRunner(agent=cast("Any", agent), messages=[], tools=[TOOL])
    agent.tool_definitions_generation = 2
    await pinned._ensure_tools_ready()
    assert agent.listed == 2  # caller-supplied tools are never replaced


def test_unpartitioned_network_server_persists_only_digest_snapshots(tmp_path):
    # e.g. `--url https://huggingface.co/mcp?anon`: no auth_identity is configured.
    config = MCPServerSettings(
        transport="http",
        url="https://example.com/mcp",
        tool_cache=MCPToolCacheSettings(directory=str(tmp_path)),
    )
    cache = ToolCatalogCache(config)
    ttl_only = ToolSnapshot(key=cache.key, fetched_at=time.time(), tools=[TOOL])
    assert not cache.save(ttl_only)
    assert cache.load() is None
    digest = ttl_only.model_copy(
        update={
            "definition_versions": DefinitionVersions(tools="sha256:t", instructions="sha256:i")
        }
    )
    assert cache.save(digest)
    assert cache.load() == digest
