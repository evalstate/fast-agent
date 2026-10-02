"""Digest mode: server definition digests replace the snapshot TTL and gate tool calls."""

import time
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from mcp.shared.exceptions import MCPError
from mcp_types import CallToolResult, ListToolsResult, TextContent, Tool

from fast_agent.config import MCPServerSettings, MCPToolCacheSettings
from fast_agent.context import Context
from fast_agent.mcp.common import create_namespaced_name
from fast_agent.mcp.definition_digests import (
    DIGEST_MISMATCH,
    KNOWN_DIGESTS_META,
    SERVER_DISCOVER,
    STALE_DIGESTS_META,
    TOOLS_LIST,
    Digests,
    rejected_digests,
    signalled_digests,
)
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.mcp_aggregator import DEFINITIONS_CHANGED_MESSAGE, MCPAggregator, NamespacedTool
from fast_agent.mcp.tool_catalog_cache import ToolCatalogCache, ToolSnapshot
from fast_agent.mcp_server_registry import ServerRegistry

TOOL = Tool(name="search", input_schema={"type": "object"})


def _config(tmp_path, *, include_instructions: bool = True) -> MCPServerSettings:
    return MCPServerSettings(
        command="server",
        connection_policy="lazy",
        include_instructions=include_instructions,
        tool_cache=MCPToolCacheSettings(directory=str(tmp_path), ttl_seconds=60),
    )


FULL: Digests = {TOOLS_LIST: "sha256:t", SERVER_DISCOVER: "sha256:d"}


def _save(cache: ToolCatalogCache, digests: Digests, *, age: float) -> None:
    cache.save(
        ToolSnapshot(
            key=cache.key,
            fetched_at=time.time() - age,
            tools=[TOOL],
            instructions="Use search.",
            digests=digests,
        )
    )


def test_digest_snapshot_ignores_ttl_only_when_it_covers_rendered_definitions(tmp_path):
    cache = ToolCatalogCache(_config(tmp_path))
    _save(cache, FULL, age=3600)
    assert cache.load() is not None
    # Instructions are rendered from the snapshot, so they need the discovery digest too.
    _save(cache, {TOOLS_LIST: "sha256:t"}, age=3600)
    assert cache.load() is None
    _save(cache, {}, age=3600)
    assert cache.load() is None
    tools_only = ToolCatalogCache(_config(tmp_path, include_instructions=False))
    _save(tools_only, {TOOLS_LIST: "sha256:t"}, age=3600)
    assert tools_only.load() is not None


def test_stale_digests_name_methods_and_rejections_default_to_everything():
    named = MCPError(DIGEST_MISMATCH, "changed", {"staleDigests": [SERVER_DISCOVER]})
    assert rejected_digests(named) == {SERVER_DISCOVER}
    # Methods we never sent (or no list at all) still mean our view is stale.
    for data in (None, {"staleDigests": ["skills/list"]}):
        assert rejected_digests(MCPError(DIGEST_MISMATCH, "changed", data)) == {
            TOOLS_LIST,
            SERVER_DISCOVER,
        }
    assert rejected_digests(MCPError(-32602, "invalid params")) is None
    assert rejected_digests(RuntimeError("boom")) is None
    # A served result only flags what it names.
    assert signalled_digests({STALE_DIGESTS_META: [TOOLS_LIST, "skills/list"]}) == {TOOLS_LIST}
    assert signalled_digests(None) == frozenset()


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
    aggregator._definition_digests["hf"] = dict(FULL)
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
            raise MCPError(DIGEST_MISMATCH, "changed", {"staleDigests": [TOOLS_LIST]})
        return CallToolResult(content=[TextContent(type="text", text="ran")])

    refresh = AsyncMock()
    monkeypatch.setattr(aggregator, "_execute_on_server", execute)
    monkeypatch.setattr(aggregator, "_refresh_server_tools", refresh)
    generation = aggregator.definitions_generation

    result = await aggregator.call_tool("hf__search", {"q": "x"})

    assert sent[0]["meta"] == {KNOWN_DIGESTS_META: FULL}
    assert result.is_error
    assert get_text(result.content[0]) == DEFINITIONS_CHANGED_MESSAGE
    refresh.assert_awaited_once_with("hf")
    assert aggregator.definitions_generation > generation

    mismatch = False
    result = await aggregator.call_tool("hf__search", {"q": "x"})
    assert not result.is_error


@pytest.mark.asyncio
async def test_signalled_stale_digests_keep_the_result_and_refresh(monkeypatch, tmp_path):
    aggregator = _connected_aggregator(tmp_path)
    served = CallToolResult(
        content=[TextContent(type="text", text="ran")],
        _meta={STALE_DIGESTS_META: [TOOLS_LIST]},
    )
    refresh = AsyncMock()
    monkeypatch.setattr(aggregator, "_execute_on_server", AsyncMock(return_value=served))
    monkeypatch.setattr(aggregator, "_refresh_server_tools", refresh)
    generation = aggregator.definitions_generation

    result = await aggregator.call_tool("hf__search", {"q": "x"})

    assert not result.is_error
    assert get_text(result.content[0]) == "ran"
    refresh.assert_awaited_once_with("hf")
    assert aggregator.definitions_generation > generation


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
    digest = ttl_only.model_copy(update={"digests": dict(FULL)})
    assert cache.save(digest)
    assert cache.load() == digest


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["eager", "lazy"])
async def test_snapshot_servers_never_hold_the_first_prompt(monkeypatch, tmp_path, policy):
    import asyncio

    config = MCPServerSettings(
        command="server",
        connection_policy=policy,
        include_instructions=False,
        tool_cache=MCPToolCacheSettings(directory=str(tmp_path)),
    )
    cache = ToolCatalogCache(config)
    _save(cache, {TOOLS_LIST: "sha256:t"}, age=3600)
    registry = ServerRegistry()
    registry.register_central("hf", config)
    context = Context(server_registry=registry, background_mcp_startup=True)
    aggregator = MCPAggregator(server_names=["hf"], connection_persistence=False, context=context)
    release = asyncio.Event()
    attached: list[str] = []

    async def attach(*, server_name, server_config, options):
        attached.append(server_name)
        await release.wait()

    monkeypatch.setattr(aggregator, "_attach_server_locked", attach)
    await aggregator.__aenter__()
    await asyncio.wait_for(context.mcp_startup.wait(), 1)
    # The prompt gate is clear and the snapshot's tools are usable already.
    assert not context.mcp_startup.pending
    assert [tool.name for tool in (await aggregator.list_tools()).tools] == ["hf__search"]
    await asyncio.sleep(0)
    # Eager connects behind the snapshot; lazy waits for the first tool call.
    assert attached == (["hf"] if policy == "eager" else [])
    release.set()
    await asyncio.gather(*aggregator._snapshot_connects)
    assert ("hf" in aggregator._deferred_servers) is (policy == "lazy")
    await aggregator.close()


@pytest.mark.asyncio
async def test_matching_discovery_digest_skips_tools_list(monkeypatch, tmp_path):
    from types import SimpleNamespace

    aggregator = _connected_aggregator(tmp_path)
    aggregator._deferred_tool_definitions["hf"] = {"search": TOOL}
    # Discovery only vouches for instructions; tools are checked on every call.
    live = SimpleNamespace(
        definition_digests={SERVER_DISCOVER: "sha256:d"}, server_instructions=None
    )
    aggregator._persistent_connection_manager = cast(
        "Any", SimpleNamespace(running_servers={"hf": live})
    )
    listed = AsyncMock(return_value=ListToolsResult(tools=[TOOL]))
    monkeypatch.setattr(aggregator, "_execute_on_server", listed)
    assert await aggregator._fetch_server_tools("hf") == [TOOL]
    listed.assert_not_awaited()
    # Changed instructions (or an explicit refresh) list tools from the server.
    live.definition_digests = {SERVER_DISCOVER: "sha256:new"}
    await aggregator._fetch_server_tools("hf")
    listed.assert_awaited_once()


@pytest.mark.asyncio
async def test_listing_restarts_when_pages_disagree_on_the_digest(monkeypatch, tmp_path):
    from types import SimpleNamespace

    aggregator = _connected_aggregator(tmp_path)
    live = SimpleNamespace(definition_digests={}, server_instructions=None)
    aggregator._persistent_connection_manager = cast(
        "Any", SimpleNamespace(running_servers={"hf": live})
    )
    # First pass changes between pages; the restarted pass is consistent.
    pages = iter(
        [("sha256:a", "next"), ("sha256:b", None), ("sha256:b", "next"), ("sha256:b", None)]
    )
    cursors: list[str | None] = []

    async def execute(**kwargs: Any) -> ListToolsResult:
        cursors.append(kwargs["method_args"].get("cursor"))
        digest, next_cursor = next(pages)
        live.definition_digests = {TOOLS_LIST: digest}
        return ListToolsResult(tools=[TOOL], next_cursor=next_cursor)

    monkeypatch.setattr(aggregator, "_execute_on_server", execute)
    tools, digest = await aggregator._list_all_tools("hf", "refresh")
    assert cursors == [None, "next", None, "next"]
    assert (len(tools), digest) == (2, "sha256:b")


def test_mcp_connect_option_applies_to_startup_targets():
    from fast_agent.cli.runtime.request_builders import _materialize_startup_mcp_servers

    merge = _materialize_startup_mcp_servers(
        server_list=None,
        urls=["https://huggingface.co/mcp?anon"],
        auth=None,
        client_metadata_url=None,
        stdio_commands=None,
        protocol_mode=None,
        connection_policy="lazy",
    )
    assert merge.servers is not None
    assert {config.connection_policy for config in merge.servers.values()} == {"lazy"}
