"""Shared MCP command-intent parsing across TUI and ACP surfaces."""

from __future__ import annotations

import shlex
from typing import Literal, cast

from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers
from fast_agent.ui.command_payloads import McpConnectCommand, McpConnectMode, McpSessionCommand


def _rebuild_target_text(tokens: list[str]) -> str:
    if not tokens:
        return ""

    rebuilt_parts: list[str] = []
    for token in tokens:
        if token == "" or any(char.isspace() for char in token):
            rebuilt_parts.append(shlex.quote(token))
        else:
            rebuilt_parts.append(token)
    return " ".join(rebuilt_parts)


def parse_mcp_connect_tokens(connect_tokens: list[str]) -> McpConnectCommand:
    if not connect_tokens:
        return McpConnectCommand(
            target_text="",
            parsed_mode="stdio",
            server_name=None,
            auth_token=None,
            timeout_seconds=None,
            trigger_oauth=None,
            reconnect_on_disconnect=None,
            force_reconnect=False,
            error=(
                "Usage: /mcp connect <target> [--name <server>] [--auth <token-value>] "
                "[--timeout <seconds>] [--oauth|--no-oauth] [--reconnect|--no-reconnect]"
            ),
        )

    connect_text = _rebuild_target_text(connect_tokens).strip()
    try:
        parsed = mcp_runtime_handlers.parse_connect_input(connect_text)
    except ValueError as exc:
        return McpConnectCommand(
            target_text="",
            parsed_mode="stdio",
            server_name=None,
            auth_token=None,
            timeout_seconds=None,
            trigger_oauth=None,
            reconnect_on_disconnect=None,
            force_reconnect=False,
            error=str(exc),
        )

    return McpConnectCommand(
        target_text=parsed.target_text,
        parsed_mode=cast(
            "McpConnectMode",
            mcp_runtime_handlers.infer_connect_mode(parsed.target_text),
        ),
        server_name=parsed.server_name,
        auth_token=parsed.auth_token,
        timeout_seconds=parsed.timeout_seconds,
        trigger_oauth=parsed.trigger_oauth,
        reconnect_on_disconnect=parsed.reconnect_on_disconnect,
        force_reconnect=parsed.force_reconnect,
        error=None,
    )


def build_mcp_connect_runtime_target(
    command: McpConnectCommand,
    *,
    redact_auth: bool = False,
) -> str:
    runtime_target = command.target_text
    if command.server_name:
        runtime_target += f" --name {command.server_name}"
    if command.auth_token:
        auth_token = "[REDACTED]" if redact_auth else command.auth_token
        runtime_target += f" --auth {shlex.quote(auth_token)}"
    if command.timeout_seconds is not None:
        runtime_target += f" --timeout {command.timeout_seconds}"
    if command.trigger_oauth is True:
        runtime_target += " --oauth"
    elif command.trigger_oauth is False:
        runtime_target += " --no-oauth"
    if command.reconnect_on_disconnect is False:
        runtime_target += " --no-reconnect"
    if command.force_reconnect:
        runtime_target += " --reconnect"
    return runtime_target


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
        if server_identity is None:
            server_identity = token
        else:
            parse_error = f"Unexpected argument: {token}"
            break

    if parse_error is None and clear_all and server_identity is not None:
        parse_error = "Use either --all or a specific server, not both"

    if parse_error is None and not clear_all and server_identity is None:
        clear_all = True

    return McpSessionCommand(
        action="clear",
        server_identity=server_identity,
        session_id=None,
        title=None,
        clear_all=clear_all,
        error=parse_error,
    )
