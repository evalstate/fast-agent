from unittest.mock import AsyncMock

import pytest

from fast_agent.config import MCPServerSettings
from fast_agent.context import Context
from fast_agent.mcp.mcp_aggregator import MCPAggregator
from fast_agent.mcp.tool_catalog_cache import ToolCacheInfo
from fast_agent.mcp_server_registry import ServerRegistry


def build_aggregator() -> MCPAggregator:
    registry = ServerRegistry()
    registry.register_central("server", MCPServerSettings(command="unused"))
    return MCPAggregator(
        server_names=["server"],
        connection_persistence=False,
        context=Context(server_registry=registry),
    )


@pytest.mark.asyncio
async def test_refresh_records_failure_and_recovery_without_duplicate_discovery(monkeypatch):
    aggregator = build_aggregator()
    attach = AsyncMock(side_effect=RuntimeError("connect failed"))
    refresh = AsyncMock()
    monkeypatch.setattr(aggregator, "_attach_server_locked", attach)
    monkeypatch.setattr(aggregator, "_refresh_attached_server_cache", refresh)
    with pytest.raises(RuntimeError, match="connect failed"):
        await aggregator.refresh_tool_cache()
    assert aggregator.get_startup_errors()
    attach.side_effect = None
    await aggregator.refresh_tool_cache()
    assert not aggregator.get_startup_errors()
    refresh.assert_not_awaited()

    aggregator._attached_server_names.append("server")
    refresh.side_effect = RuntimeError("discovery failed")
    with pytest.raises(RuntimeError, match="discovery failed"):
        await aggregator.refresh_tool_cache()
    assert aggregator.get_startup_errors()[0].phase == "discovery"
    refresh.side_effect = None
    await aggregator.refresh_tool_cache()
    assert not aggregator.get_startup_errors()


@pytest.mark.asyncio
async def test_clear_retains_live_provenance_and_close_filters_context():
    aggregator = build_aggregator()
    aggregator._attached_server_names.append("server")
    info = ToolCacheInfo(source="live", fetched_at=1, expires_at=2, tool_count=0)
    aggregator._tool_cache_info["server"] = info
    await aggregator.clear_tool_cache()
    assert aggregator._tool_cache_info["server"] == info
    aggregator._startup.set_status(aggregator._attachment_owner, "server", "error", "failed")
    await aggregator.close()
    assert aggregator.get_startup_errors()
    assert not aggregator.context.mcp_startup.get_startup_errors()
    with pytest.raises(RuntimeError, match="closed"):
        await aggregator.refresh_tool_cache()


@pytest.mark.asyncio
async def test_detach_failed_server_clears_active_error_keeps_history():
    aggregator = build_aggregator()
    aggregator._startup.set_status(aggregator._attachment_owner, "server", "error", "failed")
    await aggregator.detach_server("server")
    assert not aggregator.get_startup_errors()
    assert aggregator.startup_history[0].failure_detail == "failed"


@pytest.mark.asyncio
async def test_listing_tools_never_connects_deferred_server(monkeypatch):
    # UI reads (completion, /tools, banners) call list_tools; they must not spawn or connect.
    aggregator = build_aggregator()
    aggregator.initialized = True
    aggregator._deferred_servers.add("server")
    aggregator._tool_cache_info["server"] = ToolCacheInfo(
        source="disk", fetched_at=1, expires_at=2, tool_count=0
    )
    attach = AsyncMock(side_effect=AssertionError("list_tools connected a deferred server"))
    monkeypatch.setattr(aggregator, "_attach_server_locked", attach)
    await aggregator.list_tools()
    attach.assert_not_awaited()
