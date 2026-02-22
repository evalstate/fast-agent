"""Special command handling utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rich import print as rich_print

from fast_agent.commands.handlers import history as history_handlers
from fast_agent.core.exceptions import PromptExitError
from fast_agent.ui.command_payloads import (
    CommandPayload,
    ListSessionsCommand,
    SelectPromptCommand,
    ShowMarkdownCommand,
    ShowSystemCommand,
    ShowUsageCommand,
    SwitchAgentCommand,
    is_command_payload,
)
from fast_agent.ui.prompt.command_help import render_help_lines

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp


def handle_special_commands(
    command: str | CommandPayload | None,
    agent_app: "AgentApp | bool | None" = None,
    *,
    available_agents: set[str] | None = None,
) -> bool | CommandPayload:
    """Handle special input commands."""
    if not command:
        return False

    if is_command_payload(command):
        return cast("CommandPayload", command)

    available = available_agents or set()

    if command == "HELP":
        show_webclear_help = False
        if agent_app and agent_app is not True:
            for agent_name in sorted(available):
                try:
                    agent_obj = agent_app._agent(agent_name)
                except Exception:
                    continue
                if history_handlers.web_tools_enabled_for_agent(agent_obj):
                    show_webclear_help = True
                    break

        rich_print()
        for line in render_help_lines(show_webclear_help=show_webclear_help):
            rich_print(line)
        return True

    if command == "SESSION_HELP":
        return ListSessionsCommand(show_help=True)

    if isinstance(command, str) and command.upper() == "EXIT":
        raise PromptExitError("User requested to exit fast-agent session")

    if command == "SHOW_USAGE":
        return ShowUsageCommand()

    if command == "SHOW_SYSTEM":
        return ShowSystemCommand()

    if command == "MARKDOWN":
        return ShowMarkdownCommand()

    if command == "SELECT_PROMPT" or (
        isinstance(command, str) and command.startswith("SELECT_PROMPT:")
    ):
        if agent_app:
            prompt_name = None
            if isinstance(command, str) and command.startswith("SELECT_PROMPT:"):
                prompt_name = command.split(":", 1)[1].strip()
            return SelectPromptCommand(prompt_index=None, prompt_name=prompt_name)

        rich_print("[yellow]Prompt selection is not available outside of an agent context[/yellow]")
        return True

    if isinstance(command, str) and command.startswith("SWITCH:"):
        agent_name = command.split(":", 1)[1]
        if agent_name in available:
            if agent_app:
                return SwitchAgentCommand(agent_name=agent_name)
            rich_print("[yellow]Agent switching not available in this context[/yellow]")
        else:
            rich_print(f"[red]Unknown agent: {agent_name}[/red]")
        return True

    return False


async def handle_special_commands_async(
    command: str | CommandPayload | None,
    agent_app: "AgentApp | bool | None" = None,
    *,
    available_agents: set[str] | None = None,
) -> bool | CommandPayload:
    """Async wrapper preserved for callsites."""
    return handle_special_commands(command, agent_app, available_agents=available_agents)
