"""Slash command routing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agent.acp.slash_commands import SlashCommandHandler


class UnknownSlashCommandError(KeyError):
    """Raised when no slash command route exists for a command name."""


async def execute(handler: "SlashCommandHandler", command_name: str, arguments: str) -> str:
    match command_name:
        case "status":
            return await handler._handle_status(arguments)
        case "tools":
            return await handler._handle_tools()
        case "commands":
            return await handler._handle_commands(arguments)
        case "skills":
            return await handler._handle_skills(arguments)
        case "cards":
            return await handler._handle_cards(arguments)
        case "history":
            return await handler._handle_history(arguments)
        case "clear":
            return await handler._handle_clear(arguments)
        case "save":
            return await handler._handle_save(arguments)
        case "load":
            return await handler._handle_load(arguments)
        case "model":
            return await handler._handle_model(arguments)
        case "models":
            return await handler._handle_models(arguments)
        case "session":
            return await handler._handle_session(arguments)
        case "card":
            return await handler._handle_card(arguments)
        case "agent":
            return await handler._handle_agent(arguments)
        case "mcp":
            return await handler._handle_mcp(arguments)
        case "reload":
            return await handler._handle_reload()
        case _:
            raise UnknownSlashCommandError(command_name)
