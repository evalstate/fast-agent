"""Slash command routing helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agent.acp.slash_commands import SlashCommandHandler


class UnknownSlashCommandError(KeyError):
    """Raised when no slash command route exists for a command name."""


RouteHandler = Callable[["SlashCommandHandler", str], Awaitable[str]]


async def _status(handler: "SlashCommandHandler", arguments: str) -> str:
    return await handler._handle_status(arguments)


async def _tools(handler: "SlashCommandHandler", arguments: str) -> str:
    del arguments
    return await handler._handle_tools()


async def _commands(handler: "SlashCommandHandler", arguments: str) -> str:
    return await handler._handle_commands(arguments)


async def _skills(handler: "SlashCommandHandler", arguments: str) -> str:
    return await handler._handle_skills(arguments)


async def _cards(handler: "SlashCommandHandler", arguments: str) -> str:
    return await handler._handle_cards(arguments)


async def _history(handler: "SlashCommandHandler", arguments: str) -> str:
    return await handler._handle_history(arguments)


async def _clear(handler: "SlashCommandHandler", arguments: str) -> str:
    return await handler._handle_clear(arguments)


async def _model(handler: "SlashCommandHandler", arguments: str) -> str:
    return await handler._handle_model(arguments)


async def _session(handler: "SlashCommandHandler", arguments: str) -> str:
    return await handler._handle_session(arguments)


async def _card(handler: "SlashCommandHandler", arguments: str) -> str:
    return await handler._handle_card(arguments)


async def _agent(handler: "SlashCommandHandler", arguments: str) -> str:
    return await handler._handle_agent(arguments)


async def _mcp(handler: "SlashCommandHandler", arguments: str) -> str:
    return await handler._handle_mcp(arguments)


async def _reload(handler: "SlashCommandHandler", arguments: str) -> str:
    del arguments
    return await handler._handle_reload()


_ROUTES: dict[str, RouteHandler] = {
    "status": _status,
    "tools": _tools,
    "commands": _commands,
    "skills": _skills,
    "cards": _cards,
    "history": _history,
    "clear": _clear,
    "model": _model,
    "session": _session,
    "card": _card,
    "agent": _agent,
    "mcp": _mcp,
    "reload": _reload,
}


def routed_command_names() -> frozenset[str]:
    """Return built-in slash command names handled by the dispatcher."""
    return frozenset(_ROUTES)


async def execute(handler: "SlashCommandHandler", command_name: str, arguments: str) -> str:
    route = _ROUTES.get(command_name)
    if route is None:
        raise UnknownSlashCommandError(command_name)
    return await route(handler, arguments)
