from unittest.mock import Mock

import pytest

from fast_agent.acp.slash.handlers.mcp import handle_mcp
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.commands.harness import _execute_mcp_command
from fast_agent.commands.mcp_diagnostics import render_mcp_diagnostics
from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.ui.command_payloads import CommandError, McpDiagnosticsCommand
from fast_agent.ui.prompt.parser import parse_special_input


@pytest.fixture
def agent() -> McpAgent:
    agent = McpAgent(config=AgentConfig("diagnostics"), context=Context(config=Settings()))
    startup = agent.aggregator._startup
    owner = agent.aggregator._attachment_owner
    startup.set_status(
        owner,
        "broken server",
        "error",
        "Launch failed: executable missing\nAuthorization: Bearer hidden\nhttps://user:pass@example.com/mcp?token=hidden\nrefresh_token=private",
    )
    startup.set_status(owner, "login", "auth", "OAuth authorization required")
    startup.set_status(owner, "waiting", "pending")
    startup.set_status("another-agent", "other", "error", "other-owner-secret")
    return agent


@pytest.mark.parametrize("command", ["/mcp error", '/mcp error "broken server"', "/MCP AUTH"])
def test_parser(command: str) -> None:
    assert isinstance(parse_special_input(command), McpDiagnosticsCommand)


@pytest.mark.parametrize(
    "command", ["/mcp error a b", "/mcp auth a", '/mcp error "', '/mcp error ""']
)
def test_invalid_parser(command: str) -> None:
    assert isinstance(parse_special_input(command), CommandError)


@pytest.mark.asyncio
async def test_surfaces_share_safe_diagnostics(agent: McpAgent) -> None:
    expected = render_mcp_diagnostics(agent, ["error"])
    assert "executable missing" in expected
    assert "hidden" not in expected
    assert "private" not in expected
    assert "user:pass" not in expected
    assert "other-owner-secret" not in expected
    assert "still starting" in expected
    assert "/mcp attach" in expected
    assert await _execute_mcp_command(agent, "error") == expected
    handler = Mock()
    handler._get_current_agent.return_value = agent
    assert await handle_mcp(handler, "error") == expected


def test_filter_and_auth_recovery(agent: McpAgent) -> None:
    selected = render_mcp_diagnostics(agent, ["error", "broken server"])
    assert "OAuth authorization required" not in selected
    auth = render_mcp_diagnostics(agent, ["auth"])
    assert "fast-agent auth mcp login" in auth
    assert "does not log in" in auth
    assert "executable missing" not in auth
    assert "Unknown startup server" in render_mcp_diagnostics(agent, ["error", "missing"])
    assert "unavailable" in render_mcp_diagnostics(object(), ["auth"])


@pytest.mark.asyncio
async def test_tui_handler_is_read_only_and_shared(agent: McpAgent) -> None:
    from fast_agent.commands.handlers.display import handle_mcp_diagnostics

    ctx = Mock()
    ctx.agent_provider._agent.return_value = agent
    before = agent.aggregator.startup_status
    outcome = await handle_mcp_diagnostics(
        ctx, agent_name=agent.name, value='error "broken server"'
    )
    assert outcome.messages[0].text == render_mcp_diagnostics(agent, ["error", "broken server"])
    assert agent.aggregator.startup_status == before


@pytest.mark.parametrize("action", ["error", "auth"])
def test_discovery(action: str) -> None:
    from fast_agent.commands.command_discovery import render_command_detail_markdown

    for model_facing in (True, False):
        detail = render_command_detail_markdown("mcp", action, model_facing=model_facing)
        assert detail is not None
        assert f"/mcp {action}" in detail


def test_ready_and_empty_state(agent: McpAgent) -> None:
    agent.aggregator._startup.clear(agent.aggregator._attachment_owner)
    assert "No startup errors" in render_mcp_diagnostics(agent, ["error"])
    assert "No authentication waits" in render_mcp_diagnostics(agent, ["auth"])


@pytest.mark.parametrize(
    "detail",
    [
        'Authorization: "Bearer sensitive value"',
        'refresh_token="sensitive value"',
        "Cookie: session=sensitive; other=sensitive",
        "Bearer sensitive",
        "https://user:sensitive@example.com/mcp?code=sensitive#sensitive",
    ],
)
def test_external_failure_secrets_are_redacted(agent: McpAgent, detail: str) -> None:
    agent.aggregator._startup.set_status(
        agent.aggregator._attachment_owner, "sensitive-server", "error", detail
    )
    result = render_mcp_diagnostics(agent, ["error", "sensitive-server"])
    # The server name is intentionally retained, but no secret from its detail is.
    assert "sensitive" not in result.replace("sensitive-server", "server")


def test_diagnostics_render_metadata_and_redacted_recovery_history(agent: McpAgent) -> None:
    startup = agent.aggregator._startup
    owner = agent.aggregator._attachment_owner
    startup.set_status(owner, "broken server", "ready")
    text = render_mcp_diagnostics(agent, ["error"])
    assert "Resolved" in text
    assert "executable missing" in text
    assert "hidden" not in text
    assert "private" not in text
    assert "Phase: authentication" in text
    assert "duration:" in text
    assert "recorded:" in text
