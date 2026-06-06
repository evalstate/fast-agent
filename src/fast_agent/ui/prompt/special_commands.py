"""Special command handling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from rich import print as rich_print
from rich.text import Text

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
    from collections.abc import Callable

    from fast_agent.core.agent_app import AgentApp


class _SpecialCommandKind(StrEnum):
    SIMPLE = "simple"
    HELP = "help"
    EXIT = "exit"
    SELECT_PROMPT = "select_prompt"
    SWITCH = "switch"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class _ParsedSpecialCommand:
    kind: _SpecialCommandKind
    argument: str | None = None


_SELECT_PROMPT_PREFIX = "SELECT_PROMPT:"
_SWITCH_PREFIX = "SWITCH:"

_SIMPLE_COMMANDS: dict[str, Callable[[], CommandPayload]] = {
    "SESSION_HELP": lambda: ListSessionsCommand(show_help=True),
    "SHOW_USAGE": ShowUsageCommand,
    "SHOW_SYSTEM": ShowSystemCommand,
    "MARKDOWN": ShowMarkdownCommand,
}


def _agent_app_available(agent_app: "AgentApp | bool | None") -> bool:
    return bool(agent_app)


def _agent_app_object(agent_app: "AgentApp | bool | None") -> "AgentApp | None":
    return agent_app if agent_app and agent_app is not True else None


def _help_includes_webclear(
    agent_app: "AgentApp | bool | None",
    *,
    available_agents: set[str],
) -> bool:
    app = _agent_app_object(agent_app)
    if app is None:
        return False

    for agent_name in sorted(available_agents):
        try:
            agent_obj = app._agent(agent_name)
        except Exception:
            continue
        if history_handlers.web_tools_enabled_for_agent(agent_obj):
            return True
    return False


def _handle_help(
    agent_app: "AgentApp | bool | None",
    *,
    available_agents: set[str],
) -> bool:
    rich_print()
    for line in render_help_lines(
        show_webclear_help=_help_includes_webclear(
            agent_app,
            available_agents=available_agents,
        )
    ):
        rich_print(line)
    return True


def _handle_select_prompt(
    prompt_name: str | None,
    agent_app: "AgentApp | bool | None",
) -> bool | SelectPromptCommand:
    if _agent_app_available(agent_app):
        return SelectPromptCommand(prompt_index=None, prompt_name=prompt_name)

    rich_print("[yellow]Prompt selection is not available outside of an agent context[/yellow]")
    return True


def _handle_switch_agent(
    agent_name: str,
    agent_app: "AgentApp | bool | None",
    *,
    available_agents: set[str],
) -> bool | SwitchAgentCommand:
    if agent_name not in available_agents:
        rich_print(Text(f"Unknown agent: {agent_name}", style="red"))
        return True

    if _agent_app_available(agent_app):
        return SwitchAgentCommand(agent_name=agent_name)

    rich_print("[yellow]Agent switching not available in this context[/yellow]")
    return True


def _parse_special_command(command: str) -> _ParsedSpecialCommand:
    if command in _SIMPLE_COMMANDS:
        return _ParsedSpecialCommand(_SpecialCommandKind.SIMPLE, command)

    if command == "HELP":
        return _ParsedSpecialCommand(_SpecialCommandKind.HELP)

    if command.upper() == "EXIT":
        return _ParsedSpecialCommand(_SpecialCommandKind.EXIT)

    if command == "SELECT_PROMPT":
        return _ParsedSpecialCommand(_SpecialCommandKind.SELECT_PROMPT)

    if command.startswith(_SELECT_PROMPT_PREFIX):
        prompt_name = command.removeprefix(_SELECT_PROMPT_PREFIX).strip()
        return _ParsedSpecialCommand(_SpecialCommandKind.SELECT_PROMPT, prompt_name)

    if command.startswith(_SWITCH_PREFIX):
        return _ParsedSpecialCommand(
            _SpecialCommandKind.SWITCH,
            command.removeprefix(_SWITCH_PREFIX),
        )

    return _ParsedSpecialCommand(_SpecialCommandKind.UNKNOWN)


def _handle_special_command_string(
    command: str,
    agent_app: "AgentApp | bool | None",
    *,
    available_agents: set[str],
) -> bool | CommandPayload:
    parsed = _parse_special_command(command)
    match parsed.kind:
        case _SpecialCommandKind.SIMPLE:
            if parsed.argument is None:
                return False
            return _SIMPLE_COMMANDS[parsed.argument]()
        case _SpecialCommandKind.HELP:
            return _handle_help(agent_app, available_agents=available_agents)
        case _SpecialCommandKind.EXIT:
            raise PromptExitError("User requested to exit fast-agent session")
        case _SpecialCommandKind.SELECT_PROMPT:
            return _handle_select_prompt(parsed.argument, agent_app)
        case _SpecialCommandKind.SWITCH:
            return _handle_switch_agent(
                parsed.argument or "",
                agent_app,
                available_agents=available_agents,
            )
        case _SpecialCommandKind.UNKNOWN:
            return False


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
        return command

    if not isinstance(command, str):
        return False

    available = available_agents or set()
    return _handle_special_command_string(
        command,
        agent_app,
        available_agents=available,
    )


async def handle_special_commands_async(
    command: str | CommandPayload | None,
    agent_app: "AgentApp | bool | None" = None,
    *,
    available_agents: set[str] | None = None,
) -> bool | CommandPayload:
    """Async wrapper preserved for callsites."""
    return handle_special_commands(command, agent_app, available_agents=available_agents)
