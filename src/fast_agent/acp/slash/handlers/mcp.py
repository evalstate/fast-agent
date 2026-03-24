"""MCP slash command handlers."""

from __future__ import annotations

import asyncio
import shlex
from typing import TYPE_CHECKING, cast

from acp.helpers import text_block, tool_content
from acp.schema import (
    ContentToolCallContent,
    FileEditToolCallContent,
    TerminalToolCallContent,
    ToolCallProgress,
    ToolCallStart,
)

from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers
from fast_agent.commands.mcp_command_intents import parse_mcp_session_tokens
from fast_agent.mcp.connect_targets import parse_connect_command_text, render_connect_request
from fast_agent.utils.slash_commands import split_subcommand_and_remainder

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
        "  Example: /mcp connect \"C:\\Program Files\\Tool\\tool.exe\" --flag\n"
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


async def _send_connect_tool_update(
    handler: "SlashCommandHandler",
    *,
    tool_call_id: str,
    title: str,
    status: str,
    message: str | None = None,
) -> None:
    if handler._acp_context is None:
        return
    try:
        content: (
            list[
                ContentToolCallContent
                | FileEditToolCallContent
                | TerminalToolCallContent
            ]
            | None
        ) = [tool_content(text_block(message))] if message else None
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


def _connect_tool_call_title(request) -> str:
    connect_label = "MCP server"
    if request.target.server_name:
        connect_label = f"MCP server '{request.target.server_name}'"
    else:
        target_text = render_connect_request(request)
        first_target_token = target_text.split()[0]
        connect_label = f"MCP target '{first_target_token}'"
    return f"Connect {connect_label}"


def _rewrite_connect_progress_message(
    handler: "SlashCommandHandler",
    *,
    message: str,
    oauth_authorization_url: str | None,
) -> tuple[str, str | None]:
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

    return message, oauth_authorization_url


async def _start_connect_tool_call(
    handler: "SlashCommandHandler",
    *,
    tool_call_id: str,
    tool_call_title: str,
    display_target: str,
) -> None:
    if handler._acp_context is None:
        return
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
            handler,
            tool_call_id=tool_call_id,
            title=tool_call_title,
            status="in_progress",
            message=(
                f"{display_target}\n"
                "Open this tool call to view OAuth links and live connection status."
            ),
        )
    except Exception:
        return


def _summarize_connect_outcome(outcome) -> tuple[bool, str | None, str | None]:
    has_error = any(msg.channel == "error" for msg in outcome.messages)
    if has_error:
        first_error = next((msg for msg in outcome.messages if msg.channel == "error"), None)
        failure_details = str(first_error.text) if first_error is not None else None
        return True, failure_details, None

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
    return False, None, completion_details


def _strip_oauth_link_messages(outcome, *, oauth_authorization_url: str | None) -> None:
    if oauth_authorization_url is None:
        return
    outcome.messages = [
        message
        for message in outcome.messages
        if not str(message.text).startswith("OAuth authorization link:")
    ]


async def _emit_connect_completion_progress(
    handler: "SlashCommandHandler",
    *,
    has_error: bool,
    failure_details: str | None,
    completion_details: str | None,
) -> None:
    if has_error:
        if failure_details:
            await handler._send_progress_update(f"❌ {failure_details}")
        return
    if completion_details:
        await handler._send_progress_update(f"✅ {completion_details}")


async def _handle_mcp_list_command(
    handler: "SlashCommandHandler",
    *,
    heading: str,
    ctx,
    io: "ACPCommandIO",
    manager,
    tokens: list[str] | None = None,
) -> str:
    del tokens
    if handler._list_attached_mcp_servers_callback is None:
        return "mcp\n\nRuntime MCP server listing is not available."
    outcome = await mcp_runtime_handlers.handle_mcp_list(
        ctx,
        manager=manager,
        agent_name=handler.current_agent_name,
    )
    return handler._format_outcome_as_markdown(outcome, heading, io=io)


async def _handle_mcp_connect_command(
    handler: "SlashCommandHandler",
    *,
    heading: str,
    ctx,
    io: "ACPCommandIO",
    manager,
    remainder: str,
) -> str:
    if handler._attach_mcp_server_callback is None:
        return "mcp\n\nRuntime MCP server attachment is not available."
    if not remainder:
        return (
            f"{heading}\n\nUsage: /mcp connect <target> [--name <server>] [--auth <token>] "
            "[--timeout <seconds>] [--oauth|--no-oauth] [--reconnect|--no-reconnect]"
        )
    try:
        request = parse_connect_command_text(remainder)
    except ValueError as exc:
        return f"{heading}\n\n{exc}"

    display_target = render_connect_request(request, redact_auth=True)
    tool_call_id = handler._build_tool_call_id()
    oauth_authorization_url: str | None = None
    tool_call_title = _connect_tool_call_title(request)

    async def _send_connect_progress(message: str) -> None:
        nonlocal oauth_authorization_url

        message, oauth_authorization_url = _rewrite_connect_progress_message(
            handler,
            message=message,
            oauth_authorization_url=oauth_authorization_url,
        )

        if handler._acp_context is None:
            await handler._send_progress_update(message)
            return
        await _send_connect_tool_update(
            handler,
            tool_call_id=tool_call_id,
            title=tool_call_title,
            status="in_progress",
            message=message,
        )

    await _start_connect_tool_call(
        handler,
        tool_call_id=tool_call_id,
        tool_call_title=tool_call_title,
        display_target=display_target,
    )

    try:
        outcome = await mcp_runtime_handlers.handle_mcp_connect(
            ctx,
            manager=manager,
            agent_name=handler.current_agent_name,
            request=request,
            on_progress=_send_connect_progress,
        )
    except asyncio.CancelledError:
        await _send_connect_tool_update(
            handler,
            tool_call_id=tool_call_id,
            title="Connection cancelled",
            status="failed",
            message="Connection cancelled by client.",
        )
        raise

    if handler._acp_context is not None:
        _strip_oauth_link_messages(
            outcome,
            oauth_authorization_url=oauth_authorization_url,
        )

    has_error, failure_details, completion_details = _summarize_connect_outcome(outcome)
    await _send_connect_tool_update(
        handler,
        tool_call_id=tool_call_id,
        title=tool_call_title,
        status="failed" if has_error else "completed",
        message=failure_details if has_error else completion_details,
    )

    await _emit_connect_completion_progress(
        handler,
        has_error=has_error,
        failure_details=failure_details,
        completion_details=completion_details,
    )

    if handler._acp_context:
        agent = handler._get_current_agent()
        await handler._acp_context.invalidate_instruction_cache(
            handler.current_agent_name,
            getattr(agent, "instruction", None) if agent else None,
        )
        await handler._acp_context.send_available_commands_update()
    return handler._format_outcome_as_markdown(outcome, heading, io=io)


async def _handle_mcp_session_command(
    handler: "SlashCommandHandler",
    *,
    heading: str,
    ctx,
    io: "ACPCommandIO",
    tokens: list[str],
    manager=None,
) -> str:
    del manager
    parsed_session = parse_mcp_session_tokens(tokens[1:])
    if parsed_session.error:
        return f"{heading}\n\n{parsed_session.error}"

    outcome = await mcp_runtime_handlers.handle_mcp_session(
        ctx,
        agent_name=handler.current_agent_name,
        action=parsed_session.action,
        server_identity=parsed_session.server_identity,
        session_id=parsed_session.session_id,
        title=parsed_session.title,
        clear_all=parsed_session.clear_all,
    )
    return handler._format_outcome_as_markdown(outcome, heading, io=io)


async def _handle_mcp_disconnect_command(
    handler: "SlashCommandHandler",
    *,
    heading: str,
    ctx,
    io: "ACPCommandIO",
    manager,
    tokens: list[str],
) -> str:
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


async def _handle_mcp_reconnect_command(
    handler: "SlashCommandHandler",
    *,
    heading: str,
    ctx,
    io: "ACPCommandIO",
    manager,
    tokens: list[str],
) -> str:
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


async def handle_mcp(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    heading = "mcp"
    args = (arguments or "").strip() or "list"
    subcmd_text, remainder = split_subcommand_and_remainder(args)
    subcmd = (subcmd_text or "list").lower()

    if subcmd in {"help", "--help", "-h"}:
        return _mcp_usage_text(heading)

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    manager = cast("mcp_runtime_handlers.McpRuntimeManager", handler.instance.app)

    command_handlers = {
        "list": _handle_mcp_list_command,
        "session": _handle_mcp_session_command,
        "disconnect": _handle_mcp_disconnect_command,
        "reconnect": _handle_mcp_reconnect_command,
    }
    if subcmd == "connect":
        return await _handle_mcp_connect_command(
            handler,
            heading=heading,
            ctx=ctx,
            io=io,
            manager=manager,
            remainder=remainder,
        )

    try:
        tokens = shlex.split(args)
    except ValueError as exc:
        return f"{heading}\n\nInvalid arguments: {exc}"

    handler_func = command_handlers.get(subcmd)
    if handler_func is None:
        return _mcp_usage_text(heading)

    return await handler_func(
        handler,
        heading=heading,
        ctx=ctx,
        io=io,
        manager=manager,
        tokens=tokens,
    )
