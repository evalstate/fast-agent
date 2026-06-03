"""Shared MCP command-intent parsing across TUI and ACP surfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Final, Literal

from fast_agent.commands.option_parsing import (
    ValueOption,
    is_long_option_token,
    read_value_option,
)
from fast_agent.utils.action_normalization import normalize_action_token
from fast_agent.utils.text import strip_to_none

McpTopLevelAction = Literal["list", "connect", "session", "disconnect", "reconnect"]
McpSessionAction = Literal["jar", "new", "use", "clear", "list"]
McpServerNameAction = Literal["disconnect", "reconnect"]
_NewSessionValueName = Literal["title"]

MCP_TOP_LEVEL_ACTIONS: tuple[McpTopLevelAction, ...] = (
    "list",
    "connect",
    "session",
    "disconnect",
    "reconnect",
)
MCP_SERVER_NAME_ACTIONS: tuple[McpServerNameAction, ...] = (
    "disconnect",
    "reconnect",
)
MCP_TOP_LEVEL_ACTION_DESCRIPTIONS: dict[str, str] = {
    "list": "List currently attached MCP servers",
    "connect": "Connect a new MCP server",
    "session": "Inspect and control MCP data-layer sessions",
    "disconnect": "Disconnect an attached MCP server",
    "reconnect": "Reconnect an attached MCP server",
}
MCP_SESSION_ACTION_DESCRIPTIONS: dict[str, str] = {
    "list": "List sessions (all connected servers by default) and highlight active",
    "jar": "Show all stored sessions grouped by target/mcp name",
    "new": "Create a new MCP session",
    "create": "Alias for new; create a new MCP session",
    "use": "Switch active session to an existing session id",
    "resume": "Alias for use; switch active session to an existing session id",
    "clear": "Clear one session entry or the full local store",
}
MCP_SESSION_NEW_ACTIONS: tuple[str, ...] = ("new", "create")
MCP_SESSION_SERVER_SCOPED_ACTIONS: tuple[str, ...] = ("list", "jar", "new", "create")
MCP_SESSION_USE_ACTIONS: tuple[str, ...] = ("use", "resume")
MCP_SESSION_CLEAR_ACTION: Final = "clear"


@dataclass(frozen=True, slots=True)
class McpSessionIntent:
    action: McpSessionAction
    server_identity: str | None
    session_id: str | None
    title: str | None
    clear_all: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class McpServerNameIntent:
    server_name: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class McpNoArgsIntent:
    error: str | None


@dataclass(frozen=True, slots=True)
class _ParsedClearSessionArgs:
    server_identity: str | None = None
    clear_all: bool = False
    error: str | None = None

    def to_intent(self) -> McpSessionIntent:
        if self.error is not None:
            return _session_intent(
                "clear",
                server_identity=self.server_identity,
                clear_all=self.clear_all,
                error=self.error,
            )

        if self.clear_all and self.server_identity is not None:
            return _session_intent(
                "clear",
                server_identity=self.server_identity,
                clear_all=self.clear_all,
                error="Use either a server name or --all",
            )

        return _session_intent(
            "clear",
            server_identity=self.server_identity,
            clear_all=self.clear_all,
        )


McpSessionIntentParser = Callable[[list[str]], McpSessionIntent]


MCP_SESSION_USAGE = (
    "Usage: /mcp session [list [server]|jar [server]|new [server] [--title <title>]|"
    "create [server] [--title <title>]|use <server> <session_id>|clear <server|--all>]"
)
_SESSION_LIST_USAGE = "Usage: /mcp session list [<server_or_mcp_name>]"
_SESSION_JAR_USAGE = "Usage: /mcp session jar [<server_or_mcp_name>]"
_SESSION_USE_USAGE = "Usage: /mcp session use <server_or_mcp_name> <session_id>"
_SESSION_CLEAR_USAGE = "Usage: /mcp session clear <server|--all>"
_SESSION_TITLE_OPTIONS: tuple[ValueOption[_NewSessionValueName], ...] = (
    ValueOption("title", ("--title",)),
)
_SESSION_CLEAR_ALL_FLAG: Final = "--all"


def is_mcp_top_level_action(action: str) -> bool:
    return action in MCP_TOP_LEVEL_ACTIONS


def is_mcp_server_name_action(action: str) -> bool:
    return action in MCP_SERVER_NAME_ACTIONS


def parse_mcp_server_name_tokens(tokens: list[str], *, usage: str) -> McpServerNameIntent:
    if len(tokens) != 2:
        return McpServerNameIntent(server_name=None, error=usage)
    server_name = strip_to_none(tokens[1])
    if server_name is None:
        return McpServerNameIntent(server_name=None, error=usage)
    return McpServerNameIntent(server_name=server_name, error=None)


def parse_mcp_no_args_tokens(tokens: list[str], *, usage: str) -> McpNoArgsIntent:
    if len(tokens) != 1:
        return McpNoArgsIntent(error=usage)
    return McpNoArgsIntent(error=None)


def parse_mcp_session_tokens(session_tokens: list[str]) -> McpSessionIntent:
    if not session_tokens:
        return _session_intent("list")

    raw_action = session_tokens[0]
    normalized_action = strip_to_none(raw_action)
    if normalized_action is None:
        return _session_intent("list", error=MCP_SESSION_USAGE)

    action = normalize_action_token(normalized_action)
    args = session_tokens[1:]

    parser = _MCP_SESSION_PARSERS.get(action)
    if parser is not None:
        return parser(args)

    if is_long_option_token(normalized_action):
        return _session_intent("list", error=f"Unknown flag: {normalized_action}")

    return _session_intent(
        "list",
        server_identity=normalized_action,
        error=None if not args else MCP_SESSION_USAGE,
    )


def _session_intent(
    action: McpSessionAction,
    *,
    server_identity: str | None = None,
    session_id: str | None = None,
    title: str | None = None,
    clear_all: bool = False,
    error: str | None = None,
) -> McpSessionIntent:
    return McpSessionIntent(
        action=action,
        server_identity=server_identity,
        session_id=session_id,
        title=title,
        clear_all=clear_all,
        error=error,
    )


def _parse_single_optional_arg_session(
    args: list[str],
    *,
    action: Literal["list", "jar"],
    usage: str,
) -> McpSessionIntent:
    if len(args) > 1:
        return _session_intent(action, error=usage)
    if not args:
        return _session_intent(action)
    server_identity = strip_to_none(args[0])
    if server_identity is None:
        return _session_intent(action, error=usage)
    return _session_intent(action, server_identity=server_identity)


def _parse_new_session(args: list[str]) -> McpSessionIntent:
    server_identity: str | None = None
    title: str | None = None

    def new_session_intent(*, error: str | None = None) -> McpSessionIntent:
        return _session_intent(
            "new",
            server_identity=server_identity,
            title=title,
            error=error,
        )

    idx = 0
    while idx < len(args):
        token = args[idx]
        parsed_title = read_value_option(
            args,
            idx,
            _SESSION_TITLE_OPTIONS,
            allow_flag_like_value=True,
        )
        if parsed_title.matched:
            if parsed_title.error is not None:
                return new_session_intent(error=parsed_title.error)
            if title is not None:
                return new_session_intent(error="Duplicate flag: --title")
            title = parsed_title.require_value()
            idx = parsed_title.next_index
            continue
        if is_long_option_token(token):
            return new_session_intent(error=f"Unknown flag: {token}")
        server_name = strip_to_none(token)
        if server_name is None:
            return new_session_intent(error=MCP_SESSION_USAGE)
        if server_identity is None:
            server_identity = server_name
        else:
            return new_session_intent(error=f"Unexpected argument: {token}")
        idx += 1

    return new_session_intent()


def _parse_use_session(args: list[str]) -> McpSessionIntent:
    if len(args) != 2:
        return _session_intent("use", error=_SESSION_USE_USAGE)
    server_identity = strip_to_none(args[0])
    session_id = strip_to_none(args[1])
    if server_identity is None or session_id is None:
        return _session_intent("use", error=_SESSION_USE_USAGE)
    return _session_intent("use", server_identity=server_identity, session_id=session_id)


def _parse_clear_session_args(args: list[str]) -> _ParsedClearSessionArgs:
    if not args:
        return _ParsedClearSessionArgs(error=_SESSION_CLEAR_USAGE)

    clear_all = False
    server_identity: str | None = None
    for token in args:
        normalized = strip_to_none(token)
        if normalized is None:
            return _ParsedClearSessionArgs(
                server_identity=server_identity,
                clear_all=clear_all,
                error=_SESSION_CLEAR_USAGE,
            )
        if normalized == _SESSION_CLEAR_ALL_FLAG:
            if clear_all:
                return _ParsedClearSessionArgs(
                    server_identity=server_identity,
                    clear_all=clear_all,
                    error=f"Duplicate flag: {_SESSION_CLEAR_ALL_FLAG}",
                )
            clear_all = True
            continue
        if is_long_option_token(normalized):
            return _ParsedClearSessionArgs(
                server_identity=server_identity,
                clear_all=clear_all,
                error=f"Unknown flag: {normalized}",
            )
        if server_identity is not None:
            return _ParsedClearSessionArgs(
                server_identity=server_identity,
                clear_all=clear_all,
                error=f"Unexpected argument: {normalized}",
            )
        server_identity = normalized
    return _ParsedClearSessionArgs(
        server_identity=server_identity,
        clear_all=clear_all,
    )


def _parse_clear_session(args: list[str]) -> McpSessionIntent:
    return _parse_clear_session_args(args).to_intent()


_MCP_SESSION_PARSERS: dict[str, McpSessionIntentParser] = {
    "list": partial(
        _parse_single_optional_arg_session,
        action="list",
        usage=_SESSION_LIST_USAGE,
    ),
    "jar": partial(
        _parse_single_optional_arg_session,
        action="jar",
        usage=_SESSION_JAR_USAGE,
    ),
    "new": _parse_new_session,
    "create": _parse_new_session,
    "resume": _parse_use_session,
    "use": _parse_use_session,
    "clear": _parse_clear_session,
}
