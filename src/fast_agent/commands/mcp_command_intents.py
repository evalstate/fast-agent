"""Shared MCP command-intent parsing across TUI and ACP surfaces."""

from __future__ import annotations

from typing import Literal

from fast_agent.ui.command_payloads import McpSessionCommand


def parse_mcp_session_tokens(session_tokens: list[str]) -> McpSessionCommand:
    if not session_tokens:
        return McpSessionCommand(
            action="list",
            server_identity=None,
            session_id=None,
            title=None,
            clear_all=False,
            error=None,
        )

    action = session_tokens[0].lower()
    args = session_tokens[1:]

    if action == "list":
        return _parse_single_optional_arg_session(
            action="list",
            args=args,
            usage="Usage: /mcp session list [<server_or_mcp_name>]",
        )
    if action == "jar":
        return _parse_single_optional_arg_session(
            action="jar",
            args=args,
            usage="Usage: /mcp session jar [<server_or_mcp_name>]",
        )
    if action in {"new", "create"}:
        return _parse_new_session(args)
    if action in {"resume", "use"}:
        return _parse_use_session(args)
    if action == "clear":
        return _parse_clear_session(args)

    return McpSessionCommand(
        action="list",
        server_identity=action,
        session_id=None,
        title=None,
        clear_all=False,
        error=(
            None
            if not args
            else "Usage: /mcp session [list [server]|jar [server]|new [server] [--title <title>]|use <server> <session_id>|clear [server|--all]]"
        ),
    )


def _parse_single_optional_arg_session(
    *,
    action: Literal["list", "jar"],
    args: list[str],
    usage: str,
) -> McpSessionCommand:
    if len(args) > 1:
        return McpSessionCommand(
            action=action,
            server_identity=None,
            session_id=None,
            title=None,
            clear_all=False,
            error=usage,
        )
    return McpSessionCommand(
        action=action,
        server_identity=args[0] if args else None,
        session_id=None,
        title=None,
        clear_all=False,
        error=None,
    )


def _parse_new_session(args: list[str]) -> McpSessionCommand:
    server_identity: str | None = None
    title: str | None = None
    parse_error: str | None = None
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == "--title":
            idx += 1
            if idx >= len(args):
                parse_error = "Missing value for --title"
                break
            title = args[idx]
        elif token.startswith("--title="):
            title = token.split("=", 1)[1] or None
            if title is None:
                parse_error = "Missing value for --title"
                break
        elif token.startswith("--"):
            parse_error = f"Unknown flag: {token}"
            break
        elif server_identity is None:
            server_identity = token
        else:
            parse_error = f"Unexpected argument: {token}"
            break
        idx += 1

    return McpSessionCommand(
        action="new",
        server_identity=server_identity,
        session_id=None,
        title=title,
        clear_all=False,
        error=parse_error,
    )


def _parse_use_session(args: list[str]) -> McpSessionCommand:
    if len(args) != 2:
        return McpSessionCommand(
            action="use",
            server_identity=None,
            session_id=None,
            title=None,
            clear_all=False,
            error="Usage: /mcp session use <server_or_mcp_name> <session_id>",
        )
    return McpSessionCommand(
        action="use",
        server_identity=args[0],
        session_id=args[1],
        title=None,
        clear_all=False,
        error=None,
    )


def _parse_clear_session(args: list[str]) -> McpSessionCommand:
    clear_all = False
    server_identity: str | None = None
    parse_error: str | None = None
    for token in args:
        if token == "--all":
            clear_all = True
            continue
        if token.startswith("--"):
            parse_error = f"Unknown flag: {token}"
            break
        if server_identity is not None:
            parse_error = f"Unexpected argument: {token}"
            break
        server_identity = token

    if clear_all and server_identity is not None:
        parse_error = "Use either a server name or --all"

    if not clear_all and server_identity is None:
        clear_all = True

    return McpSessionCommand(
        action="clear",
        server_identity=server_identity,
        session_id=None,
        title=None,
        clear_all=clear_all,
        error=parse_error,
    )
