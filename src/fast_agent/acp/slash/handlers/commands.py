"""Command discovery slash handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.commands.command_catalog import suggest_command_name
from fast_agent.commands.command_discovery import (
    parse_commands_discovery_arguments,
    render_command_detail_markdown,
    render_commands_index_markdown,
    render_commands_json,
)
from fast_agent.utils.markdown import markdown_code_span
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from fast_agent.acp.slash_commands import SlashCommandHandler


def _available_command_names(handler: "SlashCommandHandler") -> set[str]:
    return {command.name for command in handler.get_available_commands()}


def _resolve_available_command_name(command_name: str, available_names: set[str]) -> str | None:
    if command_name in available_names:
        return command_name

    normalized = strip_casefold(command_name)
    matches = [name for name in available_names if strip_casefold(name) == normalized]
    if len(matches) == 1:
        return matches[0]
    return None


def _render_unknown_command_family(command_name: str) -> str:
    suggestions = suggest_command_name(command_name)
    suggestion_line = ""
    if suggestions:
        suggestion_line = "\nDid you mean: " + ", ".join(
            markdown_code_span(f"/{name}") for name in suggestions
        )
    return (
        f"# commands\n\nUnknown command family: {markdown_code_span(command_name)}.\n"
        f"Use `/commands` to list available commands.{suggestion_line}"
    )


def _render_missing_metadata(command_name: str, action_name: str | None) -> str:
    command = f"/{command_name} {action_name}" if action_name is not None else command_name
    return f"# commands\n\nNo discovery metadata for {markdown_code_span(command)} yet."


async def handle_commands(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    command_args = (arguments or "").strip()

    try:
        request = parse_commands_discovery_arguments(command_args)
    except ValueError as exc:
        return f"# commands\n\n{exc}"

    available_names = _available_command_names(handler)

    if request.as_json:
        return render_commands_json(
            command_name=request.command_name,
            action_name=request.action_name,
            command_names=available_names,
        )

    if request.command_name is None:
        return render_commands_index_markdown(command_names=available_names)

    command_name = _resolve_available_command_name(request.command_name, available_names)
    if command_name is None:
        return _render_unknown_command_family(request.command_name)

    detail = render_command_detail_markdown(command_name, request.action_name)
    if detail is not None:
        return detail

    return _render_missing_metadata(command_name, request.action_name)
