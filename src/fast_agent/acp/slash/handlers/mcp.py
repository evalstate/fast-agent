"""MCP slash command handlers."""

from __future__ import annotations

import asyncio
import shlex
from typing import TYPE_CHECKING, cast

from acp.helpers import text_block, tool_content
from acp.schema import ToolCallProgress, ToolCallStart

from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers

if TYPE_CHECKING:
    from fast_agent.acp.command_io import ACPCommandIO
    from fast_agent.acp.slash_commands import SlashCommandHandler


def _parse_mcp_server_name_argument(
    tokens: list[str],
    *,
    heading: str,
    subcommand: str,
) -> str | None:
    if len(tokens) < 2:
        return f"{heading}\n\nUsage: /mcp {subcommand} <server_name>"
    return None


def _mcp_usage_text(heading: str) -> str:
    return (
        f"{heading}\n\n"
        "Usage:\n"
        "- /mcp list\n"
        "- /mcp connect <target> [--name <server>] [--auth <token>] [--timeout <seconds>] "
        "[--oauth|--no-oauth] [--reconnect|--no-reconnect]\n"
        "- /mcp session [list [server]|jar [server]|new [server] [--title <title>]|"
        "use <server> <session_id>|clear [server|--all]]\n"
        "- /mcp disconnect <server_name>\n"
        "- /mcp reconnect <server_name>"
    )


async def _refresh_acp_instruction_cache(handler: "SlashCommandHandler") -> None:
    if not handler._acp_context:
        return
    agent = handler._get_current_agent()
    await handler._acp_context.invalidate_instruction_cache(
        handler.current_agent_name,
        getattr(agent, "instruction", None) if agent else None,
    )
    await handler._acp_context.send_available_commands_update()


async def handle_mcp(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    heading = "mcp"
    args = (arguments or "").strip()
    if not args:
        args = "list"

    try:
        tokens = shlex.split(args)
    except ValueError as exc:
        return f"{heading}\n\nInvalid arguments: {exc}"

    if not tokens:
        tokens = ["list"]
    subcmd = tokens[0].lower()

    if subcmd in {"help", "--help", "-h"}:
        return _mcp_usage_text(heading)

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    manager = cast("mcp_runtime_handlers.McpRuntimeManager", handler.instance.app)

    if subcmd == "list":
        if handler._list_attached_mcp_servers_callback is None:
            return "mcp\n\nRuntime MCP server listing is not available."
        outcome = await mcp_runtime_handlers.handle_mcp_list(
            ctx,
            manager=manager,
            agent_name=handler.current_agent_name,
        )
        return handler._format_outcome_as_markdown(outcome, heading, io=io)

    if subcmd == "connect":
        if handler._attach_mcp_server_callback is None:
            return "mcp\n\nRuntime MCP server attachment is not available."
        if len(tokens) < 2:
            return (
                f"{heading}\n\n"
                "Usage: /mcp connect <target> [--name <server>] [--auth <token-value>] [--timeout <seconds>] "
                "[--oauth|--no-oauth] [--reconnect|--no-reconnect]"
            )
        target_text = " ".join(tokens[1:])
        tool_call_id = handler._build_tool_call_id()
        oauth_authorization_url: str | None = None

        connect_label = "MCP server"
        try:
            parsed_connect = mcp_runtime_handlers.parse_connect_input(target_text)
            if parsed_connect.server_name:
                connect_label = f"MCP server '{parsed_connect.server_name}'"
            elif parsed_connect.target_text:
                first_target_token = parsed_connect.target_text.split()[0]
                connect_label = f"MCP target '{first_target_token}'"
        except Exception:
            pass
        tool_call_title = f"Connect {connect_label}"

        async def _send_connect_tool_update(*, title: str, status: str, message: str | None = None) -> None:
            if handler._acp_context is None:
                return
            try:
                content = [tool_content(text_block(message))] if message else None
                await handler._acp_context.send_session_update(
                    ToolCallProgress(
                        tool_call_id=tool_call_id,
                        title=title,
                        status=status,  # type: ignore[arg-type]
                        content=content,
                        session_update="tool_call_update",
                    )
                )
            except Exception:
                return

        async def _send_connect_progress(message: str) -> None:
            nonlocal oauth_authorization_url

            if message.startswith("Open this link to authorize:"):
                oauth_authorization_url = message.split(":", 1)[1].strip() or None

            if (
                oauth_authorization_url
                and (
                    "Waiting for OAuth callback" in message
                    or "Waiting for pasted OAuth callback URL" in message
                )
                and "OAuth authorization link:" not in message
            ):
                message = f"{message}\nOAuth authorization link: {oauth_authorization_url}"

            if handler._acp_context is not None and "Waiting for OAuth callback" in message:
                if "Stop/Cancel" not in message:
                    message = f"{message}\nTo cancel, use your ACP client's Stop/Cancel action."
                if "fast-agent auth login" not in message:
                    message = (
                        f"{message}\n"
                        "If the browser cannot reach the callback host, run "
                        "`fast-agent auth login <server-name-or-mcp-name>` on the "
                        "fast-agent host, then retry `/mcp connect ...`."
                    )

            if handler._acp_context is None:
                await handler._send_progress_update(message)
                return
            await _send_connect_tool_update(
                title=tool_call_title,
                status="in_progress",
                message=message,
            )

        if handler._acp_context is not None:
            try:
                await handler._acp_context.send_session_update(
                    ToolCallStart(
                        tool_call_id=tool_call_id,
                        title=f"{tool_call_title} (open for details)",
                        kind="fetch",
                        status="in_progress",
                        session_update="tool_call",
                    )
                )
                await _send_connect_tool_update(
                    title=tool_call_title,
                    status="in_progress",
                    message=(
                        f"{target_text}\n"
                        "Open this tool call to view OAuth links and live connection status."
                    ),
                )
            except Exception:
                pass

        try:
            outcome = await mcp_runtime_handlers.handle_mcp_connect(
                ctx,
                manager=manager,
                agent_name=handler.current_agent_name,
                target_text=target_text,
                on_progress=_send_connect_progress,
            )
        except asyncio.CancelledError:
            await _send_connect_tool_update(
                title="Connection cancelled",
                status="failed",
                message="Connection cancelled by client.",
            )
            raise

        if handler._acp_context is not None and oauth_authorization_url:
            outcome.messages = [
                message
                for message in outcome.messages
                if not str(message.text).startswith("OAuth authorization link:")
            ]

        has_error = any(msg.channel == "error" for msg in outcome.messages)
        failure_details = None
        completion_details = None
        if has_error:
            first_error = next((msg for msg in outcome.messages if msg.channel == "error"), None)
            if first_error is not None:
                failure_details = str(first_error.text)
        else:
            success_message = next(
                (
                    str(msg.text)
                    for msg in outcome.messages
                    if (
                        "Connected MCP server" in str(msg.text)
                        or "Reconnected MCP server" in str(msg.text)
                        or "already attached" in str(msg.text).lower()
                    )
                ),
                "MCP connection complete.",
            )
            oauth_link_message = next(
                (
                    str(msg.text)
                    for msg in outcome.messages
                    if str(msg.text).startswith("OAuth authorization link:")
                ),
                None,
            )
            completion_details = (
                f"{success_message}\n{oauth_link_message}" if oauth_link_message else success_message
            )
        await _send_connect_tool_update(
            title=tool_call_title,
            status="failed" if has_error else "completed",
            message=failure_details if has_error else completion_details,
        )

        if has_error:
            if failure_details:
                await handler._send_progress_update(f"❌ {failure_details}")
        elif completion_details:
            await handler._send_progress_update(f"✅ {completion_details}")

        if handler._acp_context:
            agent = handler._get_current_agent()
            await handler._acp_context.invalidate_instruction_cache(
                handler.current_agent_name,
                getattr(agent, "instruction", None) if agent else None,
            )
            await handler._acp_context.send_available_commands_update()
        return handler._format_outcome_as_markdown(outcome, heading, io=io)

    if subcmd == "session":
        session_tokens = tokens[1:]
        action = "list"
        server_identity: str | None = None
        session_id: str | None = None
        title: str | None = None
        clear_all = False

        if session_tokens:
            action = session_tokens[0].lower()
            args = session_tokens[1:]

            if action == "list":
                if len(args) > 1:
                    return f"{heading}\n\nUsage: /mcp session list [<server_or_mcp_name>]"
                server_identity = args[0] if args else None
            elif action == "jar":
                if len(args) > 1:
                    return f"{heading}\n\nUsage: /mcp session jar [<server_or_mcp_name>]"
                server_identity = args[0] if args else None
            elif action in {"new", "create"}:
                idx = 0
                while idx < len(args):
                    token = args[idx]
                    if token == "--title":
                        idx += 1
                        if idx >= len(args):
                            return f"{heading}\n\nMissing value for --title"
                        title = args[idx]
                    elif token.startswith("--title="):
                        title = token.split("=", 1)[1] or None
                        if title is None:
                            return f"{heading}\n\nMissing value for --title"
                    elif token.startswith("--"):
                        return f"{heading}\n\nUnknown flag: {token}"
                    elif server_identity is None:
                        server_identity = token
                    else:
                        return f"{heading}\n\nUnexpected argument: {token}"
                    idx += 1
                action = "new"
            elif action in {"resume", "use"}:
                if len(args) != 2:
                    return (
                        f"{heading}\n\n"
                        "Usage: /mcp session use <server_or_mcp_name> <session_id>"
                    )
                server_identity, session_id = args
                action = "use"
            elif action == "clear":
                for token in args:
                    if token == "--all":
                        clear_all = True
                        continue
                    if token.startswith("--"):
                        return f"{heading}\n\nUnknown flag: {token}"
                    if server_identity is None:
                        server_identity = token
                    else:
                        return f"{heading}\n\nUnexpected argument: {token}"

                if clear_all and server_identity is not None:
                    return f"{heading}\n\nUse either --all or a specific server, not both"

                if not clear_all and server_identity is None:
                    clear_all = True
            else:
                if args:
                    return (
                        f"{heading}\n\n"
                        "Usage: /mcp session [list [server]|jar [server]|new [server] [--title <title>]"
                        "|use <server> <session_id>|clear [server|--all]]"
                    )
                server_identity = action
                action = "list"

        session_action = cast("mcp_runtime_handlers.McpSessionAction", action)

        outcome = await mcp_runtime_handlers.handle_mcp_session(
            ctx,
            agent_name=handler.current_agent_name,
            action=session_action,
            server_identity=server_identity,
            session_id=session_id,
            title=title,
            clear_all=clear_all,
        )
        return handler._format_outcome_as_markdown(outcome, heading, io=io)

    if subcmd == "disconnect":
        if handler._detach_mcp_server_callback is None:
            return "mcp\n\nRuntime MCP server detachment is not available."
        usage_error = _parse_mcp_server_name_argument(
            tokens,
            heading=heading,
            subcommand="disconnect",
        )
        if usage_error:
            return usage_error
        outcome = await mcp_runtime_handlers.handle_mcp_disconnect(
            ctx,
            manager=manager,
            agent_name=handler.current_agent_name,
            server_name=tokens[1],
        )
        await _refresh_acp_instruction_cache(handler)
        return handler._format_outcome_as_markdown(outcome, heading, io=io)

    if subcmd == "reconnect":
        if handler._attach_mcp_server_callback is None:
            return "mcp\n\nRuntime MCP server attachment is not available."
        usage_error = _parse_mcp_server_name_argument(
            tokens,
            heading=heading,
            subcommand="reconnect",
        )
        if usage_error:
            return usage_error
        outcome = await mcp_runtime_handlers.handle_mcp_reconnect(
            ctx,
            manager=manager,
            agent_name=handler.current_agent_name,
            server_name=tokens[1],
        )
        await _refresh_acp_instruction_cache(handler)
        return handler._format_outcome_as_markdown(outcome, heading, io=io)

    return _mcp_usage_text(heading)
