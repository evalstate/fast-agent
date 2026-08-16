"""Prompt command help text utilities."""

from __future__ import annotations

from fast_agent.commands.command_catalog import CommandSpec, get_command_spec
from fast_agent.commands.session_export_help import SESSION_EXPORT_USAGE

CATALOG_HELP_COMMANDS = (
    "skills",
    "packs",
    "plugins",
    "model",
    "agent",
    "subagents",
    "card",
    "check",
)

HELP_TOPIC_DESCRIPTIONS = {
    "status": "Explain the interactive status bar",
}


def _catalog_help_lines(command_names: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for command_name in command_names:
        spec = get_command_spec(command_name)
        if spec is None:
            raise ValueError(f"unknown command catalog entry: {command_name}")
        lines.extend(_command_help_lines(spec))
    return lines


def _command_help_lines(spec: CommandSpec) -> list[str]:
    lines = [f"  /{spec.command:<13} - {spec.summary}"]
    for action in spec.actions:
        usage = action.usage or f"/{spec.command} {action.action}"
        lines.append(f"  {usage:<42} - {action.help}")
    return lines


def render_help_lines(*, show_webclear_help: bool) -> list[str]:
    lines = [
        "[bold]Available Commands:[/bold]",
        "  /help          - Show this help",
        "  /help status   - Explain the interactive status bar",
        "  /system        - Show the current system prompt",
        "  /prompt <name> - Load a Prompt File or use MCP Prompt",
        "  /attach [path|url ...|clear] - Stage or clear file/^file: or URL/^url: attachments",
        "  /usage         - Show current usage statistics",
    ]
    lines.extend(_catalog_help_lines(CATALOG_HELP_COMMANDS))
    lines.extend(
        [
            "  /history [agent_name] - Show chat history overview (quote names that match subcommands)",
            "  /history show [agent_name] - Show per-turn timing summaries",
            "  /history clear all [agent_name] - Clear conversation history (keeps templates)",
            "  /history clear last [agent_name] - Remove the most recent message from history",
            "  /compact [instructions] - Compact history into a checkpoint summary",
            "  /compact preview - Show what compaction would keep (no model call)",
            "  /compact prompt - Show the active compaction prompt",
            "  /markdown      - Show last assistant message without markdown formatting",
            "  /environment   - List configured execution environments",
            "  /process [--history] - Show active or retained finished shell processes",
            "  /mcp           - Show detailed MCP server status for the active agent",
            "  /mcp status    - Show detailed MCP server status for the active agent",
            "  /mcp list      - List configured and attached MCP servers",
            "  /mcp attach <name> - Attach a configured MCP server",
            "  /mcp connect <target> - Connect an ad-hoc MCP target",
            "      [dim]flags: --name --auth <token-value> --timeout --oauth/--no-oauth --reconnect[/dim]",
            '      [dim]example: /mcp connect "C:\\Program Files\\Tool\\tool.exe" --flag[/dim]',
            "  /mcp disconnect <name> - Disconnect attached MCP server",
            "  /mcp reconnect <name> - Reconnect attached MCP server",
            "  /connect <name|target> - Attach configured name or connect an ad-hoc target",
            "  /history save [filename] - Save current chat history to a file",
            "      [dim]Tip: Use a .json extension for MCP-compatible JSON; any other extension saves Markdown.[/dim]",
            "      [dim]Default: Timestamped filename (e.g., 25_01_15_14_30-conversation.json)[/dim]",
            "  /history load <filename> - Load chat history from a file",
            "  /history <turn> - Show a prior user turn in full",
            "  /history rewind <turn> - Rewind to a prior user turn",
            "  /history detail <turn> - Show a prior user turn in full",
            "  /history review [turn] - Review the latest or a specified turn in full",
            "  /history fix [agent_name] - Remove the last pending tool call",
        ]
    )
    if show_webclear_help:
        lines.append(
            "  /history webclear [agent_name] - Strip web tool/citation metadata from history"
        )
    lines.extend(
        [
            "  /resume [id|number] - Resume the last or specified session",
            "  /session list - List recent sessions",
            "  /session new [title] - Create a new session",
            "  /session resume [id|number] - Resume the last or specified session",
            "  /session title <text> - Set the current session title",
            "  /session fork [title] - Fork the current session",
            "  /session delete <id|number|all> - Delete a session or all sessions",
            "  /session pin <title> - Set title and pin the current session",
            "  /session unpin - Unpin the current session",
            f"  {SESSION_EXPORT_USAGE} - Export a session trace",
            "  /reload        - Reload AgentCards",
            "  @agent_name    - Switch to agent",
            "  #agent_name <msg> - Send message to agent (no space after #); '# Heading' stays plain text",
            "  STOP           - Return control back to the workflow",
            "  EXIT           - Exit fast-agent, terminating any running workflows",
            "",
            "[bold]Keyboard Shortcuts:[/bold]",
            "  Enter          - Accept completion menu selection (if open), otherwise submit/new line",
            "  Ctrl+Enter     - Always submit (in any mode)",
            "  Ctrl+Space     - Open completion menu",
            "  Tab / Shift+Tab - Next/previous completion item (when menu is open)",
            "  Shift+Tab      - Cycle service tier (when completion menu is closed)",
            "  F5             - Cycle mode (Standard / Delegate / Orchestrate / Harness-only)",
            "  F6             - Cycle reasoning (when supported)",
            "  F7             - Cycle verbosity (when supported)",
            "  F8             - Toggle web search (when supported)",
            "  F9             - Toggle web fetch (when supported)",
            "  F10            - Clear staged ^file:/^url: attachments",
            "  Ctrl+T         - Toggle multiline mode",
            "  Ctrl+E         - Edit in external editor",
            "  Ctrl+Y         - Copy last assistant response to clipboard",
            "  Ctrl+L         - Redraw the screen",
            "  Ctrl+U         - Clear input",
            "  Ctrl+C         - Cancel current operation (press twice quickly to exit)",
            "  Ctrl+D         - End prompt session (same as STOP)",
            "  Up/Down        - Navigate history",
        ]
    )
    return lines


def render_status_bar_help_lines() -> list[str]:
    return [
        "[bold]Interactive Status Bar (left → right):[/bold]",
        "  status bar",
        "  ├─ Agent",
        "  │  └─ <name>  active agent",
        "  ├─ Activity",
        "  │  ├─ ↻  managed shell processes: dim idle, yellow active, red near the limit",
        "  │  ├─ ↳  subagent delegation: green enabled, dim disabled",
        "  │  └─ ⌘  harness tools: green enabled, dim disabled",
        "  ├─ Model",
        "  │  ├─ T V D  text, vision, and document support",
        "  │  │  └─ green supported; reversed white unsupported; red related content error",
        "  │  ├─ ▲ / ▲1…▲9 / ▲+  no draft attachments / count / ten or more",
        "  │  │  └─ green usable; red missing, unknown, or unsupported",
        "  │  ├─ ⣀…⣿ (paired: ⢀…⢸ ⡀…⡇)  reasoning, then verbosity gauges",
        "  │  │  └─ fuller and green → yellow → red mean higher; dim inactive; blue auto",
        "  │  ├─ ∞<model>  plan (OAuth login/monthly token plan)",
        "  │  ├─ ▼<model>  overlay",
        "  │  ├─ »  service tier: dim standard, blue flex, red fast",
        "  │  ├─ ⊕  web search: green enabled, dim disabled",
        "  │  └─ ⇣  web fetch: green enabled, dim disabled",
        "  ├─ Context",
        "  │  └─ <percent> used, or a zero-padded turn count when usage is unavailable",
        "  ├─ Mode",
        "  │  └─ NRM normal input; MLT multiline input",
        "  └─ Right side",
        "     ├─ <working directory> / fast-agent <version>",
        "     ├─ ◀  notifications, sampling, elicitation, warnings, or tool updates",
        "     └─ transient copy notice",
        "",
        "[dim]Unsupported controls are omitted. This topic explains TUI toolbar icons.[/dim]",
    ]
