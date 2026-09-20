"""Live MCP startup toolbar contracts."""

from pathlib import Path

from fast_agent.mcp.startup import MCPStartup
from fast_agent.ui.prompt.status_bar.mcp_startup import render_mcp_startup_status
from fast_agent.ui.prompt.status_bar.renderer import (
    ShellToolbarState,
    _resolve_toolbar_identity_segment,
)


def test_status_priority_and_recovery() -> None:
    startup = MCPStartup()
    assert render_mcp_startup_status(startup.snapshot()) is None
    startup.set_status("agent", "one", "ready")
    startup.set_status("agent", "two", "pending")
    assert render_mcp_startup_status(startup.snapshot()) == "MCP 1/2"
    startup.set_status("agent", "three", "auth")
    assert "MCP AUTH · /mcp auth" in (render_mcp_startup_status(startup.snapshot()) or "")
    startup.set_status("agent", "four", "error")
    assert "MCP ERR · /mcp" in (render_mcp_startup_status(startup.snapshot()) or "")
    startup.set_status("agent", "four", "ready")
    assert "MCP AUTH" in (render_mcp_startup_status(startup.snapshot()) or "")
    startup.set_status("agent", "three", "ready")
    assert render_mcp_startup_status(startup.snapshot()) == "MCP 3/4"
    startup.set_status("agent", "two", "ready")
    assert render_mcp_startup_status(startup.snapshot()) is None


def test_shell_path_never_replaces_active_status() -> None:
    result = _resolve_toolbar_identity_segment(
        shell_state=ShellToolbarState(
            enabled=True, working_dir=Path("/very/long/path"), show_path_segment=True
        ),
        middle="",
        agent_identity_segment="agent",
        mode_style="ansigreen",
        mode_text="STD",
        version_segment="MCP ERR · /mcp",
        notification_segment="",
        copy_notice_segment="",
        shell_path_switch_delay_seconds=0,
        preserve_status=True,
    )
    assert result.html == "MCP ERR · /mcp"
