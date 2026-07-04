"""Prompt command help text utilities."""

from __future__ import annotations

from fast_agent.commands.command_catalog import CommandSpec, get_command_spec
from fast_agent.commands.session_export_help import SESSION_EXPORT_USAGE

CATALOG_HELP_COMMANDS = ("skills", "cards", "plugins", "model", "models", "check")


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
            "  /mcpstatus     - Show MCP server status summary for the active agent",
            "  /mcp list      - List attached runtime MCP servers",
            "  /mcp connect <target> - Connect MCP server at runtime",
            "      [dim]flags: --name --auth <token-value> --timeout --oauth/--no-oauth --reconnect[/dim]",
            '      [dim]example: /mcp connect "C:\\Program Files\\Tool\\tool.exe" --flag[/dim]',
            "  /mcp disconnect <name> - Disconnect attached MCP server",
            "  /mcp reconnect <name> - Reconnect attached MCP server",
            "  /connect <target> - Alias for /mcp connect",
            "  /history save [filename] - Save current chat history to a file",
            "      [dim]Tip: Use a .json extension for MCP-compatible JSON; any other extension saves Markdown.[/dim]",
            "      [dim]Default: Timestamped filename (e.g., 25_01_15_14_30-conversation.json)[/dim]",
            "  /history load <filename> - Load chat history from a file",
            "  /history <turn> - Show a prior user turn in full",
            "  /history rewind <turn> - Rewind to a prior user turn",
            "  /history detail <turn> - Show a prior user turn in full",
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
            "  /card <filename> [--tool [remove]] - Load an AgentCard (attach/remove as tool)",
            "  /agent <name> --tool [remove] - Attach/remove an agent as a tool",
            "  /agent [name] --dump - Print an AgentCard to screen",
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
