import asyncio
from unittest.mock import AsyncMock

import pytest

from fast_agent.config import MCPServerSettings
from fast_agent.context import Context
from fast_agent.mcp.mcp_aggregator import MCPAggregator, MCPAttachOptions
from fast_agent.mcp.startup import MCPStartup
from fast_agent.mcp_server_registry import ServerRegistry


def test_snapshots_are_stable_and_owner_scoped() -> None:
    startup = MCPStartup()
    startup.set_status("first", "server", "pending")
    before = startup.snapshot()
    startup.set_status("first", "server", "ready")
    startup.set_status("second", "server", "auth", "Please authenticate")
    assert before[0].state == "pending"
    assert startup.snapshot("first")[0].state == "ready"
    assert startup.get_startup_errors()[0].failure_detail == "Please authenticate"
    startup.clear("second")
    assert not startup.get_startup_errors()


@pytest.mark.asyncio
async def test_background_startup_is_parallel_observable_and_cancelled(monkeypatch) -> None:
    registry = ServerRegistry()
    for name in ("slow", "broken", "ready", "deferred"):
        registry.register_central(
            name, MCPServerSettings(command="unused", load_on_start=name != "deferred")
        )
    context = Context(server_registry=registry, background_mcp_startup=True)
    aggregator = MCPAggregator(
        server_names=["slow", "broken", "ready", "deferred"],
        connection_persistence=False,
        context=context,
    )
    slow_started = asyncio.Event()
    slow_cancelled = asyncio.Event()
    ready_finished = asyncio.Event()
    calls: list[str] = []

    async def attach(*, server_name, server_config, options):
        calls.append(server_name)
        if server_name == "slow":
            slow_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                slow_cancelled.set()
        elif server_name == "broken":
            raise RuntimeError("independent failure")
        else:
            ready_finished.set()

    # Keep real attachment locking so this detects global serialization regressions.
    monkeypatch.setattr(aggregator, "_attach_server_locked", attach)
    await aggregator.__aenter__()
    assert {entry.state for entry in aggregator.startup_status} == {"pending"}
    await asyncio.wait_for(slow_started.wait(), 1)
    await asyncio.wait_for(ready_finished.wait(), 1)
    assert aggregator.get_startup_errors("broken")[0].failure_detail == "independent failure"
    assert context.mcp_startup.snapshot() == aggregator.startup_status
    status = await asyncio.wait_for(aggregator.collect_server_status(), 1)
    assert status["broken"].error_message is not None
    assert "independent failure" in status["broken"].error_message
    assert "/mcp error" in status["broken"].error_message
    assert status["slow"].error_message == "initializing..."
    # UI reads do not wait for pending servers or schedule duplicate startup.
    await asyncio.wait_for(aggregator.list_tools(), 1)
    assert sorted(calls) == ["broken", "ready", "slow"]
    await asyncio.wait_for(aggregator.close(), 1)
    assert slow_cancelled.is_set()
    assert aggregator.get_startup_errors("slow")[0].failure_detail == "Startup cancelled"


@pytest.mark.asyncio
async def test_eager_startup_waits_and_does_not_repeat(monkeypatch) -> None:
    registry = ServerRegistry()
    registry.register_central("server", MCPServerSettings(command="unused"))
    aggregator = MCPAggregator(
        server_names=["server"],
        connection_persistence=False,
        context=Context(server_registry=registry),
    )
    attach = AsyncMock()
    monkeypatch.setattr(aggregator, "_attach_server_locked", attach)
    await aggregator.__aenter__()
    assert aggregator.startup_status[0].state == "ready"
    await aggregator.__aenter__()
    await aggregator.list_tools()
    assert attach.await_count == 1
    await aggregator.close()


@pytest.mark.asyncio
async def test_connection_requests_share_inflight_startup(monkeypatch) -> None:
    from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
    from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager, ServerConnection

    config = MCPServerSettings(command="unused")
    registry = ServerRegistry()
    registry.register_central("server", config)
    manager = MCPConnectionManager(registry, context=Context(server_registry=registry))

    def unused_factory():
        raise AssertionError("transport is not used in this test")

    runtime = MCPClientCallbackRuntime(server_name="server", server_config=config)
    connection = ServerConnection("server", config, unused_factory, runtime)
    started = asyncio.Event()
    release = asyncio.Event()
    launches = 0

    async def healthy(server_name, server_config):
        return connection if started.is_set() else None

    async def launch(**kwargs):
        nonlocal launches
        launches += 1
        started.set()
        await release.wait()
        return connection

    monkeypatch.setattr(manager, "_healthy_running_server", healthy)
    monkeypatch.setattr(manager, "_launch_and_wait_for_server", launch)
    monkeypatch.setattr(manager, "_healthy_or_retry_server", AsyncMock(return_value=connection))
    first = asyncio.create_task(manager.get_server("server", callback_runtime=runtime))
    await started.wait()
    second = asyncio.create_task(manager.get_server("server", callback_runtime=runtime))
    await asyncio.sleep(0)
    assert not second.done()
    release.set()
    assert await first is await second is connection
    assert launches == 1


@pytest.mark.asyncio
async def test_auth_wait_is_visible_and_clears_on_success(monkeypatch) -> None:
    from fast_agent.mcp.oauth_client import OAuthEvent

    registry = ServerRegistry()
    registry.register_central("server", MCPServerSettings(command="unused"))
    context = Context(server_registry=registry)
    aggregator = MCPAggregator(
        server_names=["server"],
        connection_persistence=False,
        context=context,
    )

    async def attach(*, server_name, server_config, options):
        assert options.oauth_event_handler is not None
        await options.oauth_event_handler(
            OAuthEvent("wait_start", server_name, message="Authorize in browser")
        )
        status = context.mcp_startup.snapshot()[0]
        assert status.state == "auth"
        assert status.failure_detail == "Authorize in browser"
        await options.oauth_event_handler(OAuthEvent("wait_end", server_name))
        assert aggregator.startup_status[0].state == "pending"

    monkeypatch.setattr(aggregator, "_attach_server_locked", attach)
    await aggregator.__aenter__()
    assert aggregator.startup_status[0].state == "ready"
    assert not aggregator.get_startup_errors()
    await aggregator.close()


@pytest.mark.asyncio
async def test_failed_recovery_updates_live_error_and_preserves_resolved_history(
    monkeypatch,
) -> None:
    registry = ServerRegistry()
    registry.register_central("server", MCPServerSettings(command="unused"))
    aggregator = MCPAggregator(
        server_names=["server"],
        connection_persistence=False,
        context=Context(server_registry=registry),
    )
    attach = AsyncMock(side_effect=RuntimeError("first failure"))
    monkeypatch.setattr(aggregator, "_attach_server_locked", attach)
    await aggregator.__aenter__()
    first = aggregator.get_startup_errors()[0]
    assert first.transport == "stdio"
    assert first.timestamp is not None
    assert first.duration_seconds >= 0
    attach.side_effect = RuntimeError("recovery failed")
    with pytest.raises(RuntimeError, match="recovery failed"):
        await aggregator.attach_server(server_name="server")
    assert aggregator.get_startup_errors()[0].failure_detail == "recovery failed"
    attach.side_effect = None
    await aggregator.attach_server(server_name="server")
    assert not aggregator.get_startup_errors()
    assert {s.failure_detail for s in aggregator.startup_history} == {
        "first failure",
        "recovery failed",
    }
    assert all(s.resolved_at is not None for s in aggregator.startup_history)
    assert first.resolved_at is None  # Previously returned snapshots stay immutable.
    await aggregator.close()


@pytest.mark.asyncio
async def test_reconnect_waits_for_inflight_startup(monkeypatch) -> None:
    from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
    from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager

    config = MCPServerSettings(command="unused")
    registry = ServerRegistry()
    registry.register_central("server", config)
    manager = MCPConnectionManager(registry, context=Context(server_registry=registry))
    runtime = MCPClientCallbackRuntime(server_name="server", server_config=config)
    started = asyncio.Event()
    disconnect = AsyncMock()

    async def launch(**kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "_healthy_running_server", AsyncMock(return_value=None))
    monkeypatch.setattr(manager, "_launch_and_wait_for_server", launch)
    monkeypatch.setattr(manager, "disconnect_server", disconnect)
    first = asyncio.create_task(manager.get_server("server", callback_runtime=runtime))
    await started.wait()
    reconnect = asyncio.create_task(manager.reconnect_server("server", callback_runtime=runtime))
    await asyncio.sleep(0)
    assert not reconnect.done()
    disconnect.assert_not_awaited()
    reconnect.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reconnect
    assert not first.done()  # Cancelling a waiter must not cancel the launch owner.
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first


@pytest.mark.asyncio
async def test_first_prompt_gate_waits_for_startup_and_survives_waiter_cancel(
    monkeypatch, tmp_path
) -> None:
    import time

    from mcp_types import Tool

    from fast_agent.config import MCPToolCacheSettings
    from fast_agent.mcp.tool_catalog_cache import ToolCatalogCache, ToolSnapshot

    registry = ServerRegistry()
    registry.register_central("live", MCPServerSettings(command="unused"))
    cached = MCPServerSettings(
        command="unused",
        connection_policy="deferred",
        include_instructions=False,
        tool_cache=MCPToolCacheSettings(directory=str(tmp_path)),
    )
    registry.register_central("cached", cached)
    cache = ToolCatalogCache(cached)
    cache.save(
        ToolSnapshot(
            key=cache.key,
            fetched_at=time.time(),
            tools=[Tool(name="tool", input_schema={"type": "object"})],
        )
    )
    context = Context(server_registry=registry, background_mcp_startup=True)
    aggregator = MCPAggregator(
        server_names=["live", "cached"], connection_persistence=False, context=context
    )
    release = asyncio.Event()

    async def attach(*, server_name, server_config, options):
        await release.wait()

    monkeypatch.setattr(aggregator, "_attach_server_locked", attach)
    await aggregator.__aenter__()
    assert context.mcp_startup.pending
    waiter = asyncio.create_task(context.mcp_startup.wait())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    # Cancelling a prompt that was waiting must not cancel startup itself.
    release.set()
    await asyncio.wait_for(context.mcp_startup.wait(), 1)
    assert not context.mcp_startup.pending
    # Snapshot-restored deferred servers are usable, not stuck pending.
    assert {entry.server_name: entry.state for entry in aggregator.startup_status} == {
        "live": "ready",
        "cached": "ready",
    }
    await aggregator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("background", "stored_tokens", "trigger_oauth"),
    [(True, False, False), (True, True, None), (False, False, None)],
)
async def test_background_startup_never_begins_an_interactive_login(
    monkeypatch, background: bool, stored_tokens: bool, trigger_oauth: bool | None
) -> None:
    registry = ServerRegistry()
    registry.register_central(
        "remote", MCPServerSettings(transport="http", url="https://example.com/mcp")
    )
    aggregator = MCPAggregator(
        server_names=["remote"],
        connection_persistence=False,
        context=Context(server_registry=registry, background_mcp_startup=background),
    )
    monkeypatch.setattr(
        aggregator, "_has_stored_oauth_tokens", AsyncMock(return_value=stored_tokens)
    )
    captured: list[MCPAttachOptions] = []

    async def attach(*, server_name, server_config, options: MCPAttachOptions) -> None:
        captured.append(options)

    monkeypatch.setattr(aggregator, "_attach_server_locked", attach)
    await aggregator._start_server("remote")
    assert captured[0].trigger_oauth is trigger_oauth
