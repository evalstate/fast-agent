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

if TYPE_CHECKING:
    from fast_agent.acp.slash_commands import SlashCommandHandler


def _available_command_names(handler: "SlashCommandHandler") -> set[str]:
    names = {name.lower() for name in handler._get_allowed_session_commands()}
    names.add("commands")
    return names


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
            command_names=available_names,
        )

    if request.command_name is None:
        return render_commands_index_markdown(command_names=available_names)

    if request.command_name.lower() not in available_names:
        suggestions = suggest_command_name(request.command_name)
        suggestion_line = ""
        if suggestions:
            suggestion_line = "\nDid you mean: " + ", ".join(f"`/{name}`" for name in suggestions)
        return (
            f"# commands\n\nUnknown command family: `{request.command_name}`.\n"
            f"Use `/commands` to list available commands.{suggestion_line}"
        )

    detail = render_command_detail_markdown(request.command_name)
    if detail is not None:
        return detail

    return f"# commands\n\nNo discovery metadata for `{request.command_name}` yet."
