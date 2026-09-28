from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers.mcp_runtime import handle_mcp_cache
from fast_agent.mcp.mcp_aggregator import MCPAggregator
from fast_agent.mcp.tool_catalog_cache import ToolCacheInfo
from fast_agent.ui.command_payloads import CommandError, McpCacheCommand
from fast_agent.ui.enhanced_prompt import parse_special_input
from fast_agent.ui.mcp_display import format_tool_cache


@pytest.mark.parametrize(
    "command,target",
    [
        ('/mcp cache clear "my server"', "my server"),
        ("/mcp cache clear all", None),
        ("/mcp cache clear", None),
        ("/mcp refresh docs", "docs"),
        ("/mcp refresh", None),
        ("/mcp refresh all", None),
    ],
)
@pytest.mark.asyncio
async def test_cache_command_reaches_aggregator(command, target):
    payload = parse_special_input(command)
    assert isinstance(payload, McpCacheCommand)
    aggregator = Mock(spec=MCPAggregator)
    aggregator.clear_tool_cache = AsyncMock()
    aggregator.refresh_tool_cache = AsyncMock()
    agent = Mock(spec=McpAgent)
    agent.aggregator = aggregator
    ctx = Mock(spec=CommandContext)
    ctx.agent_provider = Mock()
    ctx.agent_provider._agent.return_value = agent
    outcome = await handle_mcp_cache(ctx, agent_name="main", value=payload.value)
    assert not any(message.channel == "error" for message in outcome.messages)
    if "refresh" in command:
        aggregator.refresh_tool_cache.assert_awaited_once_with(target)
        aggregator.clear_tool_cache.assert_not_called()
    else:
        aggregator.clear_tool_cache.assert_awaited_once_with(target)
        aggregator.refresh_tool_cache.assert_not_called()


@pytest.mark.parametrize(
    "command",
    [
        "/mcp cache docs",
        "/mcp cache clear a b",
        "/mcp refresh a b",
        '/mcp refresh ""',
    ],
)
def test_invalid_cache_commands_are_rejected(command):
    assert isinstance(parse_special_input(command), CommandError)


def test_cache_display_describes_origin_change_detection_and_persistence():
    now = datetime.now(timezone.utc).timestamp()
    digest = ToolCacheInfo(source="live", fetched_at=now - 4, tool_count=4, digest=True)
    assert (
        format_tool_cache(digest)
        == "tools: 4 · live 4s ago · digest checked per call · memory only"
    )
    saved = ToolCacheInfo(
        source="disk", fetched_at=now - 120, tool_count=3, expires_at=now + 60, persisted=True
    )
    assert "disk 2m ago" in format_tool_cache(saved)
    assert "reusable for" in format_tool_cache(saved) and "saved" in format_tool_cache(saved)
    saved.expires_at = now - 60
    assert "expired" in format_tool_cache(saved)
    # Without persistence a TTL means nothing to the user; don't show one.
    unsaved = ToolCacheInfo(source="live", fetched_at=now, tool_count=1, expires_at=now + 60)
    assert "reusable" not in format_tool_cache(unsaved)
    assert format_tool_cache(None) == "no tool catalog recorded"


@pytest.mark.asyncio
async def test_harness_cache_refresh_and_failure():
    from fast_agent.commands.harness import execute_harness_command
    from fast_agent.core.exceptions import AgentConfigError

    agent = Mock(spec=McpAgent)
    agent.name = "main"
    agent.agent_registry = {}
    agent.context = None
    agent.aggregator = Mock(spec=MCPAggregator)
    agent.aggregator.refresh_tool_cache = AsyncMock()
    rendered = await execute_harness_command(agent, '/mcp refresh "my server"')
    assert "refreshed" in rendered
    agent.aggregator.refresh_tool_cache.assert_awaited_once_with("my server")
    agent.aggregator.refresh_tool_cache.side_effect = ValueError("Unknown MCP server")
    with pytest.raises(AgentConfigError):
        await execute_harness_command(agent, "/mcp refresh missing")


@pytest.mark.asyncio
async def test_acp_cache_routes_to_shared_handler():
    from fast_agent.acp.slash_commands import SlashCommandHandler
    from fast_agent.core.fastagent import AgentInstance

    agent = Mock(spec=McpAgent)
    agent.name = "main"
    agent.acp_commands = {}
    agent.aggregator = Mock(spec=MCPAggregator)
    agent.aggregator.clear_tool_cache = AsyncMock()
    app = Mock()
    app._agent.return_value = agent
    instance = AgentInstance(app=app, agents={"main": agent}, registry_version=0)
    handler = SlashCommandHandler(
        session_id="cache-test", instance=instance, primary_agent_name="main"
    )
    rendered = await handler.execute_command("mcp", 'cache clear "my server"')
    assert "cleared" in rendered
    agent.aggregator.clear_tool_cache.assert_awaited_once_with("my server")


def test_cache_help_and_completion_are_discoverable():
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    from fast_agent.commands.command_discovery import render_command_detail_markdown
    from fast_agent.ui.enhanced_prompt import AgentCompleter

    assert "/mcp cache" in (render_command_detail_markdown("mcp", "cache") or "")
    assert "/mcp refresh" in (render_command_detail_markdown("mcp", "refresh") or "")
    completer = AgentCompleter(agents=["main"])
    for prefix, expected in [
        ("/mcp cache ", "clear"),
        ("/mcp cache clear ", "all"),
        ("/mcp refresh ", "all"),
    ]:
        completions = completer.get_completions(Document(prefix), CompleteEvent())
        assert expected in {completion.text for completion in completions}


@pytest.mark.asyncio
async def test_cache_summary_does_not_refresh_or_infer_missing_provenance():
    from fast_agent.mcp.mcp_aggregator import ServerStatus

    agent = Mock(spec=McpAgent)
    aggregator = Mock(spec=MCPAggregator)
    agent.aggregator = aggregator
    aggregator.collect_server_status = AsyncMock(
        return_value={
            "docs": ServerStatus(server_name="docs"),
        }
    )
    ctx = Mock(spec=CommandContext)
    ctx.agent_provider = Mock()
    ctx.agent_provider._agent.return_value = agent
    outcome = await handle_mcp_cache(ctx, agent_name="main", value="cache")
    assert "docs: no tool catalog recorded" in outcome.messages[0].plain_text()
    aggregator.refresh_tool_cache.assert_not_called()
    aggregator.clear_tool_cache.assert_not_called()


@pytest.mark.asyncio
async def test_cache_summary_uses_real_aggregator_without_connecting():
    from fast_agent.config import MCPServerSettings
    from fast_agent.context import Context
    from fast_agent.mcp_server_registry import ServerRegistry

    registry = ServerRegistry()
    registry.register_central("manual", MCPServerSettings(command="unused", load_on_start=False))
    aggregator = MCPAggregator(server_names=["manual"], context=Context(server_registry=registry))
    agent = Mock(spec=McpAgent)
    agent.aggregator = aggregator
    ctx = Mock(spec=CommandContext)
    ctx.agent_provider = Mock()
    ctx.agent_provider._agent.return_value = agent
    outcome = await handle_mcp_cache(ctx, agent_name="main", value="cache")
    assert not any(message.channel == "error" for message in outcome.messages)
    assert "manual: no tool catalog recorded" in outcome.messages[0].plain_text()
    assert not aggregator.startup_status
