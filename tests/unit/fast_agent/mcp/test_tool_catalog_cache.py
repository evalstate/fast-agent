import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from mcp_types import ListToolsResult, Tool

from fast_agent.config import MCPServerSettings, MCPToolCacheSettings
from fast_agent.context import Context
from fast_agent.mcp.mcp_aggregator import MCPAggregator
from fast_agent.mcp.tool_catalog_cache import ToolCatalogCache, ToolSnapshot
from fast_agent.mcp_server_registry import ServerRegistry


def test_disk_opt_out_identity_expiry_and_corruption(tmp_path):
    config = MCPServerSettings(
        url="https://example.com/mcp",
        transport="http",
        access_token="private-token",
        tool_cache=MCPToolCacheSettings(enabled=False, auth_identity="account-a"),
    )
    config.tool_cache.directory = str(tmp_path)
    cache = ToolCatalogCache(config)
    snapshot = ToolSnapshot(key=cache.key, fetched_at=time.time(), tools=[])
    cache.save(snapshot)
    assert not list(tmp_path.iterdir())
    config.tool_cache.enabled = True
    cache.save(snapshot)
    assert cache.load() == snapshot
    assert "private-token" not in cache.path.read_text()
    assert cache.path.stat().st_mode & 0o077 == 0
    changed = ToolCatalogCache(config.model_copy(update={"access_token": "other"}))
    assert changed.load() is None
    cache.save(snapshot.model_copy(update={"fetched_at": 0}))
    assert cache.load() is None
    cache.path.write_text("not json")
    assert cache.load() is None
    cache.clear()
    assert not cache.path.exists()


@pytest.mark.asyncio
async def test_paginated_snapshot_reused_without_connect_and_clear(tmp_path):
    config = MCPServerSettings(
        command="echo",
        connection_policy="deferred",
        include_instructions=False,
        tool_cache=MCPToolCacheSettings(enabled=True, directory=str(tmp_path)),
    )
    registry = ServerRegistry()
    registry.register_central("test", config)
    context = Context(server_registry=registry)
    aggregator = MCPAggregator(server_names=["test"], context=context)
    aggregator.server_supports_feature = AsyncMock(return_value=True)
    aggregator._execute_on_server = AsyncMock(
        side_effect=[
            ListToolsResult(tools=[Tool(name="first", input_schema={})], next_cursor="page2"),
            ListToolsResult(
                tools=[Tool(name="second", input_schema={}, _meta={"fastmcp": {"tags": []}})]
            ),
        ]
    )
    tools = await aggregator._fetch_server_tools("test", cache_mode="refresh")
    assert len(tools) == 2
    assert aggregator._execute_on_server.call_args.kwargs["method_args"] == {
        "cache_mode": "refresh",
        "cursor": "page2",
    }
    reused = MCPAggregator(server_names=["test"], context=context)
    reused.attach_server = AsyncMock()
    await reused.load_servers()
    reused.attach_server.assert_not_called()
    assert len((await reused.list_tools()).tools) == 2
    assert reused._tool_cache_info["test"].source == "disk"
    await reused.clear_tool_cache("test")
    assert not (await reused.list_tools()).tools
    assert ToolCatalogCache(config).load() is None


@pytest.mark.asyncio
async def test_deferred_changed_schema_never_executes(tmp_path):
    config = MCPServerSettings(
        command="echo",
        connection_policy="deferred",
        include_instructions=False,
        tool_cache=MCPToolCacheSettings(enabled=True, directory=str(tmp_path)),
    )
    cache = ToolCatalogCache(config)
    original = Tool(name="action", input_schema={"type": "object"})
    cache.save(ToolSnapshot(key=cache.key, fetched_at=time.time(), tools=[original]))
    registry = ServerRegistry()
    registry.register_central("test", config)
    aggregator = MCPAggregator(server_names=["test"], context=Context(server_registry=registry))
    await aggregator.load_servers()
    advertised = (await aggregator.list_tools()).tools[0].name

    async def changed_catalog(*args, **kwargs):
        cache.save(
            ToolSnapshot(
                key=cache.key,
                fetched_at=time.time(),
                tools=[
                    Tool(name="action", input_schema={"type": "object", "required": ["confirm"]})
                ],
            )
        )
        await aggregator._restore_tool_catalog("test")

    async def connect(**kwargs):
        await asyncio.sleep(0)
        await changed_catalog()

    aggregator._attach_server_locked = AsyncMock(side_effect=connect)
    aggregator._refresh_attached_server_cache = AsyncMock()
    aggregator._execute_on_server = AsyncMock()
    results = await asyncio.gather(
        aggregator.call_tool(advertised), aggregator.call_tool(advertised)
    )
    assert all(result.is_error for result in results)
    assert (await aggregator.call_tool(advertised)).is_error
    aggregator._attach_server_locked.assert_awaited_once()
    aggregator._execute_on_server.assert_not_called()
    aggregator._refresh_attached_server_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeated_pagination_cursor_is_not_persisted(tmp_path):
    config = MCPServerSettings(
        command="echo", tool_cache=MCPToolCacheSettings(enabled=True, directory=str(tmp_path))
    )
    registry = ServerRegistry()
    registry.register_central("test", config)
    aggregator = MCPAggregator(server_names=["test"], context=Context(server_registry=registry))
    aggregator.server_supports_feature = AsyncMock(return_value=True)
    aggregator._execute_on_server = AsyncMock(
        return_value=ListToolsResult(tools=[], next_cursor="same")
    )
    with pytest.raises(ValueError, match="pagination"):
        await aggregator._fetch_server_tools("test")
    assert ToolCatalogCache(config).load() is None


def test_identity_ignores_launch_environment_but_partitions_explicit_env(tmp_path, monkeypatch):
    config = MCPServerSettings(command="server", env={"ACCOUNT": "one"})
    config.tool_cache = MCPToolCacheSettings(enabled=True, directory=str(tmp_path))
    first = ToolCatalogCache(config)
    monkeypatch.setenv("SHLVL", "987")
    monkeypatch.setenv("UNRELATED_LAUNCH_ID", "new")
    assert ToolCatalogCache(config).key == first.key
    config.env = {"ACCOUNT": "two"}
    assert ToolCatalogCache(config).key != first.key


def test_network_cache_requires_explicit_partition(tmp_path):
    config = MCPServerSettings(
        transport="http",
        url="https://example.com/mcp",
        tool_cache=MCPToolCacheSettings(enabled=True, directory=str(tmp_path)),
    )
    cache = ToolCatalogCache(config)
    cache.save(ToolSnapshot(key=cache.key, fetched_at=time.time()))
    assert cache.load() is None
    assert not list(tmp_path.iterdir())


def test_disk_failure_is_optional(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    config = MCPServerSettings(
        command="server",
        tool_cache=MCPToolCacheSettings(enabled=True, directory=str(blocker / "cache")),
    )
    cache = ToolCatalogCache(config)
    cache.save(ToolSnapshot(key=cache.key, fetched_at=time.time()))
    assert cache.load() is None
    cache.clear()


@pytest.mark.asyncio
async def test_deferred_snapshot_does_not_omit_instructions_or_app_validation(tmp_path):
    config = MCPServerSettings(
        command="server",
        connection_policy="deferred",
        tool_cache=MCPToolCacheSettings(enabled=True, directory=str(tmp_path)),
    )
    cache = ToolCatalogCache(config)
    cache.save(
        ToolSnapshot(
            key=cache.key, fetched_at=time.time(), tools=[Tool(name="action", input_schema={})]
        )
    )
    registry = ServerRegistry()
    registry.register_central("test", config)
    aggregator = MCPAggregator(server_names=["test"], context=Context(server_registry=registry))
    assert not await aggregator._restore_tool_catalog("test")
    config.include_instructions = False
    cache = ToolCatalogCache(config)
    cache.save(
        ToolSnapshot(
            key=cache.key,
            fetched_at=time.time(),
            tools=[Tool(name="action", input_schema={}, _meta={"ui": {"visibility": ["app"]}})],
        )
    )
    assert not await aggregator._restore_tool_catalog("test")


@pytest.mark.asyncio
async def test_clear_preserves_live_advertisements_and_bypasses_sdk_cache(tmp_path):
    config = MCPServerSettings(
        command="server",
        connection_policy="deferred",
        include_instructions=False,
        tool_cache=MCPToolCacheSettings(enabled=True, directory=str(tmp_path)),
    )
    cache = ToolCatalogCache(config)
    cache.save(
        ToolSnapshot(
            key=cache.key, fetched_at=time.time(), tools=[Tool(name="action", input_schema={})]
        )
    )
    registry = ServerRegistry()
    registry.register_central("test", config)
    aggregator = MCPAggregator(server_names=["test"], context=Context(server_registry=registry))
    await aggregator.load_servers()
    # Model the transition to a connected catalog, retaining its advertisement.
    aggregator._deferred_servers.clear()
    before = await aggregator.list_tools()
    await aggregator.clear_tool_cache("test")
    assert await aggregator.list_tools() == before
    assert cache.load() is None
    aggregator.server_supports_feature = AsyncMock(return_value=True)
    aggregator._execute_on_server = AsyncMock(return_value=ListToolsResult(tools=[]))
    await aggregator._fetch_server_tools("test")
    assert aggregator._execute_on_server.call_args.kwargs["method_args"]["cache_mode"] == "refresh"


def test_secret_values_do_not_collide_or_leak(tmp_path):
    from pydantic import SecretStr

    from fast_agent.config import MCPServerAuthSettings

    config = MCPServerSettings(
        transport="http",
        url="https://example.com/mcp",
        auth=MCPServerAuthSettings.model_validate({"client_secret": SecretStr("first-secret")}),
        tool_cache=MCPToolCacheSettings(
            enabled=True, directory=str(tmp_path), auth_identity="account"
        ),
    )
    first = ToolCatalogCache(config)
    first.save(ToolSnapshot(key=first.key, fetched_at=time.time()))
    config.auth = MCPServerAuthSettings.model_validate(
        {"client_secret": SecretStr("second-secret")}
    )
    assert ToolCatalogCache(config).key != first.key
    assert "first-secret" not in first.path.read_text()


@pytest.mark.asyncio
async def test_pagination_has_bound_even_for_unique_cursors():
    registry = ServerRegistry()
    registry.register_central("test", MCPServerSettings(command="server"))
    aggregator = MCPAggregator(server_names=["test"], context=Context(server_registry=registry))
    aggregator.server_supports_feature = AsyncMock(return_value=True)
    count = 0

    async def page(**kwargs):
        nonlocal count
        count += 1
        return ListToolsResult(tools=[], next_cursor=str(count))

    aggregator._execute_on_server = AsyncMock(side_effect=page)
    with pytest.raises(ValueError, match="pagination"):
        await aggregator._fetch_server_tools("test")
    assert count <= 1000


@pytest.mark.asyncio
async def test_caller_arriving_during_hydration_cannot_use_new_schema(tmp_path):
    config = MCPServerSettings(
        command="server",
        connection_policy="deferred",
        include_instructions=False,
        tool_cache=MCPToolCacheSettings(enabled=True, directory=str(tmp_path)),
    )
    cache = ToolCatalogCache(config)
    cache.save(
        ToolSnapshot(
            key=cache.key, fetched_at=time.time(), tools=[Tool(name="action", input_schema={})]
        )
    )
    registry = ServerRegistry()
    registry.register_central("test", config)
    aggregator = MCPAggregator(server_names=["test"], context=Context(server_registry=registry))
    await aggregator.load_servers()
    name = (await aggregator.list_tools()).tools[0].name
    published = asyncio.Event()
    finish = asyncio.Event()

    async def connect(**kwargs):
        # Attachment publishes live tools before the final authoritative refresh.
        aggregator._server_to_tool_map["test"][0].tool.input_schema = {
            "type": "object",
            "required": ["confirmation"],
        }
        published.set()
        await finish.wait()

    aggregator._attach_server_locked = AsyncMock(side_effect=connect)
    aggregator._refresh_attached_server_cache = AsyncMock()
    aggregator._execute_on_server = AsyncMock()
    first = asyncio.create_task(aggregator.call_tool(name))
    await published.wait()
    second = asyncio.create_task(aggregator.call_tool(name))
    await asyncio.sleep(0)
    finish.set()
    results = await asyncio.gather(first, second)
    assert all(result.is_error for result in results)
    assert (await aggregator.call_tool(name)).is_error
    aggregator._execute_on_server.assert_not_called()


@pytest.mark.parametrize("home_source", ["default", "environment", "settings", "resolved"])
def test_default_cache_follows_active_home(tmp_path, monkeypatch, home_source):
    from fast_agent.config import Settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
    monkeypatch.delenv("FAST_AGENT_RUNTIME_HOME", raising=False)
    settings = Settings()
    home = tmp_path / ".fast-agent"
    if home_source == "environment":
        home = tmp_path / "environment-home"
        monkeypatch.setenv("FAST_AGENT_HOME", str(home))
    elif home_source == "settings":
        home = tmp_path / "configured-home"
        settings.home = str(home)
    elif home_source == "resolved":
        home = tmp_path / "cli-home"
        settings._fast_agent_home = str(home)

    config = MCPServerSettings(command="server")
    registry = ServerRegistry()
    registry.register_central("test", config)
    aggregator = MCPAggregator(
        server_names=["test"], context=Context(config=settings, server_registry=registry)
    )
    cache = aggregator._catalog_cache("test")
    assert cache is not None
    snapshot = ToolSnapshot(key=cache.key, fetched_at=time.time())
    cache.save(snapshot)
    assert cache.path.parent == home / "cache" / "mcp-tools"
    assert cache.load() == snapshot
    assert cache.path.stat().st_mode & 0o077 == 0


def test_aggregator_cache_respects_no_home_and_explicit_directory(tmp_path):
    from fast_agent.config import Settings

    settings = Settings(home=str(tmp_path / "disabled-home"))
    settings._fast_agent_no_home = True
    config = MCPServerSettings(command="server")
    registry = ServerRegistry()
    registry.register_central("test", config)
    aggregator = MCPAggregator(
        server_names=["test"], context=Context(config=settings, server_registry=registry)
    )
    assert aggregator._catalog_cache("test") is None
    assert not (tmp_path / "disabled-home").exists()

    config.tool_cache.directory = str(tmp_path / "explicit-cache")
    registry.register_central("test", config)
    cache = aggregator._catalog_cache("test")
    assert cache is not None
    cache.save(ToolSnapshot(key=cache.key, fetched_at=time.time()))
    assert cache.path.parent == tmp_path / "explicit-cache"
    assert cache.load() is not None
    config.tool_cache.enabled = False
    registry.register_central("test", config)
    assert aggregator._catalog_cache("test") is None
